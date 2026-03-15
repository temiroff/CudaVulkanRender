#pragma once
#include <cuda_runtime.h>

// Wraps Intel OIDN 1.x CPU denoiser.
// Falls back to a no-op when OIDN_ENABLED is not defined at build time.
struct DenoiserState {
    void* device      = nullptr;  // OIDNDevice (opaque)
    void* filter      = nullptr;  // OIDNFilter (opaque)
    int   width       = 0;
    int   height      = 0;
    bool  available   = false;    // true when OIDN_ENABLED and init succeeded
};

// (Re-)initialize for the given resolution. Safe to call on resize.
void denoiser_init(DenoiserState& state, int width, int height);

// Release all OIDN resources.
void denoiser_free(DenoiserState& state);

// Denoise d_input into d_output (d_input is never modified).
// d_output must be a GPU float4 buffer of the same size.
// Returns false if denoiser is not available or fails.
bool denoiser_run(DenoiserState& state,
                  const float4* d_input, float4* d_output,
                  int width, int height);
