#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// ─────────────────────────────────────────────
//  GPU texture helpers — RGBA8 → CUDA texture object
// ─────────────────────────────────────────────

struct GpuTexture {
    cudaArray_t         array = nullptr;
    cudaTextureObject_t tex   = 0;
};

// Upload an RGBA8 image to a CUDA texture object.
// The texture is configured for normalized float reads, linear filtering,
// and repeat wrap mode.  width*height*4 bytes are read from data.
// srgb=true: RGB channels are linearised (sRGB→linear) before upload so the
//   path tracer always works in linear light.  Alpha is left as-is.
GpuTexture gpu_texture_upload_rgba8(const uint8_t* data, int width, int height,
                                    bool srgb = false);

// Destroy the texture object and free the underlying cudaArray.
// Both handles are zeroed after the call.
void gpu_texture_free(GpuTexture& t);

// Collect all .tex handles into a device-side array and return the device
// pointer.  The caller is responsible for calling cudaFree on the result.
// Returns nullptr if textures is empty.
cudaTextureObject_t* gpu_textures_upload_handles(const std::vector<GpuTexture>& textures);
