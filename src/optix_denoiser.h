#pragma once
#include <cuda_runtime.h>

// OptiX 7.x/8.x GPU denoiser — uses RTX tensor cores.
// Falls back to no-op when OPTIX_ENABLED is not defined.
// Download OptiX SDK: https://developer.nvidia.com/designworks/optix/download

struct OptixDenoiserState {
    void*              ctx          = nullptr;   // OptixDeviceContext (opaque)
    void*              denoiser     = nullptr;   // OptixDenoiser (opaque)
    unsigned long long d_state      = 0;         // CUdeviceptr
    unsigned long long d_scratch    = 0;         // CUdeviceptr
    unsigned long long d_output     = 0;         // CUdeviceptr — separate output buffer
    size_t             state_size   = 0;
    size_t             scratch_size = 0;
    int                width        = 0;
    int                height       = 0;
    bool               available    = false;
};

// Init/resize denoiser for given resolution. Safe to call on resize.
bool optix_denoiser_init(OptixDenoiserState& s, int w, int h);

// Run GPU denoiser: reads d_input, writes to d_output (d_input is never modified).
// d_output must be a GPU float4 buffer of the same size as d_input.
void optix_denoiser_run(OptixDenoiserState& s, const float4* d_input, float4* d_output);

// Free all resources.
void optix_denoiser_free(OptixDenoiserState& s);
