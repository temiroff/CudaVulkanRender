// denoiser.cpp — Intel OIDN 1.x wrapper.
// Compiled unconditionally; OIDN code is gated by OIDN_ENABLED.
#include "denoiser.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#ifdef OIDN_ENABLED
#include <OpenImageDenoise/oidn.h>
#endif

void denoiser_init(DenoiserState& state, int width, int height)
{
    denoiser_free(state);
    state.width  = width;
    state.height = height;

#ifdef OIDN_ENABLED
    OIDNDevice dev = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
    oidnCommitDevice(dev);

    const char* err = nullptr;
    if (oidnGetDeviceError(dev, &err) != OIDN_ERROR_NONE) {
        std::cerr << "[oidn] device init error: " << (err ? err : "unknown") << "\n";
        oidnReleaseDevice(dev);
        return;
    }

    OIDNFilter filter = oidnNewFilter(dev, "RT");
    oidnSetFilter1b(filter, "hdr",  true);   // input is linear HDR
    oidnSetFilter1b(filter, "srgb", false);  // output stays linear

    state.device    = dev;
    state.filter    = filter;
    state.available = true;
    std::cout << "[oidn] initialized CPU denoiser " << width << "x" << height << "\n";
#else
    (void)width; (void)height;
#endif
}

void denoiser_free(DenoiserState& state)
{
#ifdef OIDN_ENABLED
    if (state.filter) oidnReleaseFilter((OIDNFilter)state.filter);
    if (state.device) oidnReleaseDevice((OIDNDevice)state.device);
#endif
    state = DenoiserState{};
}

bool denoiser_run(DenoiserState& state,
                  const float4* d_input, float4* d_output,
                  int width, int height)
{
#ifdef OIDN_ENABLED
    if (!state.available || !state.device || !state.filter) return false;

    const int n = width * height;

    // ── 1. Download accumulation buffer from GPU (read-only) ─────────────────
    std::vector<float4> h_input((size_t)n);
    cudaMemcpy(h_input.data(), d_input, (size_t)n * sizeof(float4), cudaMemcpyDeviceToHost);

    // ── 2. Convert float4 → packed float3 ────────────────────────────────────
    std::vector<float> h_color ((size_t)n * 3);
    std::vector<float> h_out   ((size_t)n * 3);
    for (int i = 0; i < n; ++i) {
        h_color[i * 3 + 0] = h_input[i].x;
        h_color[i * 3 + 1] = h_input[i].y;
        h_color[i * 3 + 2] = h_input[i].z;
    }

    // ── 3. Set images, commit, execute ───────────────────────────────────────
    OIDNFilter filter = (OIDNFilter)state.filter;
    oidnSetSharedFilterImage(filter, "color",
        h_color.data(), OIDN_FORMAT_FLOAT3, (size_t)width, (size_t)height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "output",
        h_out.data(),   OIDN_FORMAT_FLOAT3, (size_t)width, (size_t)height, 0, 0, 0);
    oidnCommitFilter(filter);
    oidnExecuteFilter(filter);

    const char* err = nullptr;
    if (oidnGetDeviceError((OIDNDevice)state.device, &err) != OIDN_ERROR_NONE) {
        std::cerr << "[oidn] execute error: " << (err ? err : "unknown") << "\n";
        return false;
    }

    // ── 4. Convert float3 → float4 and upload to d_output (NOT d_input) ──────
    std::vector<float4> h_result((size_t)n);
    for (int i = 0; i < n; ++i) {
        h_result[i] = { h_out[i*3+0], h_out[i*3+1], h_out[i*3+2], 1.f };
    }
    cudaMemcpy(d_output, h_result.data(), (size_t)n * sizeof(float4), cudaMemcpyHostToDevice);
    return true;

#else
    (void)state; (void)d_input; (void)d_output; (void)width; (void)height;
    return false;
#endif
}
