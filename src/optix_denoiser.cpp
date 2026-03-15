#include "optix_denoiser.h"
#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>

#ifdef OPTIX_ENABLED
#  include <optix.h>
#  include <optix_stubs.h>                      // optixInit, optixInitWithHandle
#  include <optix_function_table_definition.h>  // define function pointers — ONE TU only
#endif

// ─────────────────────────────────────────────────────────────────────────────

#ifdef OPTIX_ENABLED

static void optix_log_cb(unsigned level, const char* tag, const char* msg, void*)
{
    if (level <= 3)
        fprintf(stderr, "[OptiX][L%u][%s]: %s\n", level, tag, msg);
}

bool optix_denoiser_init(OptixDenoiserState& s, int w, int h)
{
    optix_denoiser_free(s);

    // Load the OptiX shared library
    if (optixInit() != OPTIX_SUCCESS) {
        fprintf(stderr, "[optix] optixInit() failed — is the OptiX runtime installed?\n");
        fprintf(stderr, "[optix] Driver must be >= 520.00. Get SDK: developer.nvidia.com/designworks/optix/download\n");
        return false;
    }

    // Grab the current CUDA context (created by cudaSetDevice earlier)
    CUcontext cu_ctx = nullptr;
    cuCtxGetCurrent(&cu_ctx);
    if (!cu_ctx) {
        fprintf(stderr, "[optix] no active CUDA context\n");
        return false;
    }

    // Create OptiX device context
    OptixDeviceContextOptions ctx_opts{};
    ctx_opts.logCallbackFunction = optix_log_cb;
    ctx_opts.logCallbackLevel    = 3;
    OptixDeviceContext ctx = nullptr;
    if (optixDeviceContextCreate(cu_ctx, &ctx_opts, &ctx) != OPTIX_SUCCESS) {
        fprintf(stderr, "[optix] device context creation failed\n");
        return false;
    }
    s.ctx = ctx;

    // Create HDR AI denoiser (beauty-pass only — no albedo/normal guides needed)
    OptixDenoiserOptions dn_opts{};
    dn_opts.guideAlbedo = 0;
    dn_opts.guideNormal = 0;
    OptixDenoiser dn = nullptr;
    if (optixDenoiserCreate(ctx, OPTIX_DENOISER_MODEL_KIND_HDR, &dn_opts, &dn) != OPTIX_SUCCESS) {
        fprintf(stderr, "[optix] denoiser creation failed\n");
        optixDeviceContextDestroy(ctx);
        s.ctx = nullptr;
        return false;
    }
    s.denoiser = dn;

    // Query memory requirements for this resolution
    OptixDenoiserSizes sizes{};
    optixDenoiserComputeMemoryResources(dn, (unsigned)w, (unsigned)h, &sizes);
    s.state_size   = sizes.stateSizeInBytes;
    s.scratch_size = sizes.withoutOverlapScratchSizeInBytes;

    cudaMalloc((void**)&s.d_state,   s.state_size);
    cudaMalloc((void**)&s.d_scratch, s.scratch_size);
    cudaMalloc((void**)&s.d_output,  (size_t)w * h * sizeof(float4));

    // Setup (upload model weights etc.) — blocking
    optixDenoiserSetup(dn, /*stream*/0,
        (unsigned)w, (unsigned)h,
        (CUdeviceptr)s.d_state,   s.state_size,
        (CUdeviceptr)s.d_scratch, s.scratch_size);

    s.width     = w;
    s.height    = h;
    s.available = true;
    printf("[optix] GPU HDR denoiser ready %dx%d  (state=%zuKB  scratch=%zuKB)\n",
           w, h, s.state_size / 1024, s.scratch_size / 1024);
    return true;
}

void optix_denoiser_run(OptixDenoiserState& s, const float4* d_input, float4* d_output)
{
    if (!s.available) return;

    OptixDenoiserParams params{};
    params.hdrIntensity = (CUdeviceptr)0;
    params.blendFactor  = 0.0f;

    OptixImage2D input_img{};
    input_img.data               = (CUdeviceptr)d_input;
    input_img.width              = (unsigned)s.width;
    input_img.height             = (unsigned)s.height;
    input_img.rowStrideInBytes   = (unsigned)(s.width * sizeof(float4));
    input_img.pixelStrideInBytes = (unsigned)sizeof(float4);
    input_img.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

    // OptiX writes to its own internal scratch buffer, then we copy to d_output
    OptixImage2D scratch_img = input_img;
    scratch_img.data = (CUdeviceptr)s.d_output;

    OptixDenoiserGuideLayer guide{};
    OptixDenoiserLayer layer{};
    layer.input  = input_img;
    layer.output = scratch_img;

    optixDenoiserInvoke(
        (OptixDenoiser)s.denoiser,
        /*stream*/0,
        &params,
        (CUdeviceptr)s.d_state,   s.state_size,
        &guide,
        &layer, 1,
        /*offsetX*/0, /*offsetY*/0,
        (CUdeviceptr)s.d_scratch, s.scratch_size);

    // Copy denoised result to caller-provided d_output (never touches d_input)
    cudaMemcpyAsync(d_output, (void*)s.d_output,
                    (size_t)s.width * s.height * sizeof(float4),
                    cudaMemcpyDeviceToDevice, 0);
}

void optix_denoiser_free(OptixDenoiserState& s)
{
    if (s.d_output)  cudaFree((void*)s.d_output);
    if (s.d_scratch) cudaFree((void*)s.d_scratch);
    if (s.d_state)   cudaFree((void*)s.d_state);
    if (s.denoiser)  optixDenoiserDestroy((OptixDenoiser)s.denoiser);
    if (s.ctx)       optixDeviceContextDestroy((OptixDeviceContext)s.ctx);
    s = OptixDenoiserState{};
}

#else  // ── OPTIX_ENABLED not defined ────────────────────────────────────────

bool optix_denoiser_init(OptixDenoiserState&, int, int) { return false; }
void optix_denoiser_run (OptixDenoiserState&, const float4*, float4*) {}
void optix_denoiser_free(OptixDenoiserState&)            {}

#endif
