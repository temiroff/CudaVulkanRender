#include "gpu_textures.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

// ─────────────────────────────────────────────
//  Internal helper: abort on CUDA error
// ─────────────────────────────────────────────

static bool cuda_check(cudaError_t err, const char* file, int line, std::string* err_out = nullptr)
{
    if (err == cudaSuccess) return true;

    char msg[256];
    std::snprintf(msg, sizeof(msg), "CUDA error %s (%d) at %s:%d",
                  cudaGetErrorString(err), (int)err, file, line);
    std::fprintf(stderr, "%s\n", msg);
    if (err_out) *err_out = msg;
    return false;
}
#define CUDA_CHECK(x, err_out) cuda_check((x), __FILE__, __LINE__, (err_out))

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

static int mip_level_count(int width, int height)
{
    int levels = 1;
    while (width > 1 || height > 1) {
        width = (width > 1) ? (width >> 1) : 1;
        height = (height > 1) ? (height >> 1) : 1;
        ++levels;
    }
    return levels;
}

static std::vector<uint8_t> downsample_rgba8_box(const std::vector<uint8_t>& src, int src_w, int src_h)
{
    int dst_w = (src_w > 1) ? (src_w >> 1) : 1;
    int dst_h = (src_h > 1) ? (src_h >> 1) : 1;
    std::vector<uint8_t> dst((size_t)dst_w * dst_h * 4, 0);

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int accum[4] = { 0, 0, 0, 0 };
            int taps = 0;
            for (int oy = 0; oy < 2; ++oy) {
                for (int ox = 0; ox < 2; ++ox) {
                    int sx = std::min(src_w - 1, x * 2 + ox);
                    int sy = std::min(src_h - 1, y * 2 + oy);
                    const uint8_t* p = &src[((size_t)sy * src_w + sx) * 4];
                    accum[0] += p[0];
                    accum[1] += p[1];
                    accum[2] += p[2];
                    accum[3] += p[3];
                    ++taps;
                }
            }
            uint8_t* out = &dst[((size_t)y * dst_w + x) * 4];
            out[0] = (uint8_t)((accum[0] + taps / 2) / taps);
            out[1] = (uint8_t)((accum[1] + taps / 2) / taps);
            out[2] = (uint8_t)((accum[2] + taps / 2) / taps);
            out[3] = (uint8_t)((accum[3] + taps / 2) / taps);
        }
    }

    return dst;
}

GpuTexture gpu_texture_upload_rgba8(const uint8_t* data, int width, int height,
                                    bool srgb, std::string* err_out)
{
    GpuTexture t;

    if (!data || width <= 0 || height <= 0) {
        if (err_out) *err_out = "gpu_texture_upload_rgba8: invalid input image";
        return t;
    }

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

    // 1. Build mip chain on the host.
    std::vector<std::vector<uint8_t>> mip_pixels;
    std::vector<int> mip_widths;
    std::vector<int> mip_heights;
    mip_pixels.reserve((size_t)mip_level_count(width, height));
    mip_widths.reserve(mip_pixels.capacity());
    mip_heights.reserve(mip_pixels.capacity());

    mip_pixels.emplace_back(data, data + (size_t)width * height * 4);
    mip_widths.push_back(width);
    mip_heights.push_back(height);
    while (mip_widths.back() > 1 || mip_heights.back() > 1) {
        mip_pixels.push_back(downsample_rgba8_box(mip_pixels.back(), mip_widths.back(), mip_heights.back()));
        mip_widths.push_back((mip_widths.back() > 1) ? (mip_widths.back() >> 1) : 1);
        mip_heights.push_back((mip_heights.back() > 1) ? (mip_heights.back() >> 1) : 1);
    }

    // 2. Allocate a 2-D CUDA mipmapped array for uchar4 (RGBA, 8-bit per channel).
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    unsigned int flags = cudaArrayDefault;
    if (!CUDA_CHECK(cudaMallocMipmappedArray(&t.mip_array, &desc,
                                             make_cudaExtent((size_t)width, (size_t)height, 0),
                                             (unsigned int)mip_pixels.size(), flags), err_out))
        return t;

    // 3. Copy each mip level into the mipmapped array.
    for (size_t level = 0; level < mip_pixels.size(); ++level) {
        cudaArray_t level_array = nullptr;
        if (!CUDA_CHECK(cudaGetMipmappedArrayLevel(&level_array, t.mip_array, (unsigned int)level), err_out)) {
            cudaFreeMipmappedArray(t.mip_array);
            t.mip_array = nullptr;
            return t;
        }
        if (!CUDA_CHECK(cudaMemcpy2DToArray(
            level_array,
            /*wOffset=*/0, /*hOffset=*/0,
            mip_pixels[level].data(),
            /*spitch=*/(size_t)mip_widths[level] * 4,
            /*width=*/(size_t)mip_widths[level] * 4,
            (size_t)mip_heights[level],
            cudaMemcpyHostToDevice), err_out)) {
            cudaFreeMipmappedArray(t.mip_array);
            t.mip_array = nullptr;
            return t;
        }
    }

    // 4. Build resource descriptor pointing at the mipmapped array.
    cudaResourceDesc resDesc{};
    resDesc.resType                         = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap              = t.mip_array;

    // 5. Build texture descriptor.
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0]        = cudaAddressModeWrap;
    texDesc.addressMode[1]        = cudaAddressModeWrap;
    texDesc.filterMode            = cudaFilterModeLinear;
    texDesc.mipmapFilterMode      = cudaFilterModeLinear;
    texDesc.readMode              = cudaReadModeNormalizedFloat; // uchar4 → float4 [0,1]
    texDesc.normalizedCoords      = 1;
    texDesc.minMipmapLevelClamp   = 0.f;
    texDesc.maxMipmapLevelClamp   = (float)(mip_pixels.size() - 1);
    texDesc.maxAnisotropy         = 1;

    if (!CUDA_CHECK(cudaCreateTextureObject(&t.tex, &resDesc, &texDesc, nullptr), err_out)) {
        cudaFreeMipmappedArray(t.mip_array);
        t.mip_array = nullptr;
        return t;
    }
    return t;
}

// ─────────────────────────────────────────────
//  gpu_texture_free
// ─────────────────────────────────────────────

void gpu_texture_free(GpuTexture& t)
{
    if (t.tex != 0) {
        cuda_check(cudaDestroyTextureObject(t.tex), __FILE__, __LINE__);
        t.tex = 0;
    }
    if (t.mip_array != nullptr) {
        cuda_check(cudaFreeMipmappedArray(t.mip_array), __FILE__, __LINE__);
        t.mip_array = nullptr;
    }
}

// ─────────────────────────────────────────────
//  gpu_textures_upload_handles
// ─────────────────────────────────────────────

cudaTextureObject_t* gpu_textures_upload_handles(const std::vector<GpuTexture>& textures,
                                                 std::string* err_out)
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
    if (!CUDA_CHECK(cudaMalloc(&d_handles, bytes), err_out))
        return nullptr;
    if (!CUDA_CHECK(cudaMemcpy(d_handles, handles.data(), bytes, cudaMemcpyHostToDevice), err_out)) {
        cudaFree(d_handles);
        return nullptr;
    }

    return d_handles;
}
