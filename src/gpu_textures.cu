#include "gpu_textures.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// ─────────────────────────────────────────────
//  Internal helper: abort on CUDA error
// ─────────────────────────────────────────────

static void cuda_check(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",
                     cudaGetErrorString(err), (int)err, file, line);
        std::abort();
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

// ─────────────────────────────────────────────
//  gpu_texture_upload_rgba8
// ─────────────────────────────────────────────

// Fast sRGB byte → linear float → linear byte (IEC 61966-2-1 approximation).
// pow(x/255, 2.2)*255 — accurate enough for rendering; avoids full piecewise formula.
static inline uint8_t srgb_to_linear_u8(uint8_t v)
{
    float f = (float)v / 255.f;
    f = std::pow(f, 2.2f);
    int i = (int)(f * 255.f + 0.5f);
    if (i < 0)   i = 0;
    if (i > 255) i = 255;
    return (uint8_t)i;
}

GpuTexture gpu_texture_upload_rgba8(const uint8_t* data, int width, int height,
                                    bool srgb)
{
    GpuTexture t;

    // If sRGB, convert RGB channels to linear before uploading.
    // Alpha channel is left unchanged.
    std::vector<uint8_t> linear_buf;
    if (srgb) {
        const size_t npx = (size_t)width * height;
        linear_buf.resize(npx * 4);
        for (size_t i = 0; i < npx; ++i) {
            linear_buf[i * 4 + 0] = srgb_to_linear_u8(data[i * 4 + 0]);
            linear_buf[i * 4 + 1] = srgb_to_linear_u8(data[i * 4 + 1]);
            linear_buf[i * 4 + 2] = srgb_to_linear_u8(data[i * 4 + 2]);
            linear_buf[i * 4 + 3] = data[i * 4 + 3]; // alpha unchanged
        }
        data = linear_buf.data();
    }

    // 1. Allocate a 2-D CUDA array for uchar4 (RGBA, 8-bit per channel).
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&t.array, &desc, (size_t)width, (size_t)height));

    // 2. Copy host pixels into the array (no pitch on the source side).
    CUDA_CHECK(cudaMemcpy2DToArray(
        t.array,
        /*wOffset=*/0, /*hOffset=*/0,
        data,
        /*spitch (bytes per source row)=*/(size_t)width * 4,
        /*width  (bytes per row to copy)=*/(size_t)width * 4,
        (size_t)height,
        cudaMemcpyHostToDevice));

    // 3. Build resource descriptor pointing at the array.
    cudaResourceDesc resDesc{};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = t.array;

    // 4. Build texture descriptor.
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeNormalizedFloat; // uchar4 → float4 [0,1]
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&t.tex, &resDesc, &texDesc, nullptr));
    return t;
}

// ─────────────────────────────────────────────
//  gpu_texture_free
// ─────────────────────────────────────────────

void gpu_texture_free(GpuTexture& t)
{
    if (t.tex != 0) {
        CUDA_CHECK(cudaDestroyTextureObject(t.tex));
        t.tex = 0;
    }
    if (t.array != nullptr) {
        CUDA_CHECK(cudaFreeArray(t.array));
        t.array = nullptr;
    }
}

// ─────────────────────────────────────────────
//  gpu_textures_upload_handles
// ─────────────────────────────────────────────

cudaTextureObject_t* gpu_textures_upload_handles(const std::vector<GpuTexture>& textures)
{
    if (textures.empty()) {
        return nullptr;
    }

    // Collect the opaque texture-object handles on the host.
    std::vector<cudaTextureObject_t> handles;
    handles.reserve(textures.size());
    for (const GpuTexture& gt : textures) {
        handles.push_back(gt.tex);
    }

    // Allocate device memory and copy.
    const size_t bytes = handles.size() * sizeof(cudaTextureObject_t);
    cudaTextureObject_t* d_handles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_handles, bytes));
    CUDA_CHECK(cudaMemcpy(d_handles, handles.data(), bytes, cudaMemcpyHostToDevice));

    return d_handles;
}
