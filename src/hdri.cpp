// hdri.cpp — HDRI loading via OpenEXR (all compression types) or stb_image (HDR/PNG/JPG).
// Upload to a CUDA float4 texture.
//
// OpenEXR is from the bundled USD devkit — supports DWAA, DWAB, PIZ, B44, ZIP, etc.
// stb_image.h implementation is compiled in gltf_loader.cpp — include only declarations here.
#include "hdri.h"
#include <stb_image.h>
// OpenEXR — handles all compression types (DWAA, DWAB, PIZ, B44, PXR24, ZIP…)
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <cstring>

HdriMap hdri_load(const std::string& path)
{
    HdriMap map;
    int w = 0, h = 0;
    float* data    = nullptr;  // float4 RGBA linear, w*h*4 floats
    bool   is_exr  = false;
    bool   data_needs_free = false; // true → call free(), false → stbi_image_free()

    // Detect extension (compare lowercase)
    auto ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);

    if (ext == ".exr") {
        is_exr = true;
        try {
            // ImfRgbaFile handles ALL EXR compression types (DWAA, DWAB, PIZ, B44, ZIP…)
            Imf::RgbaInputFile file(path.c_str());
            Imath::Box2i dw = file.dataWindow();
            w = dw.max.x - dw.min.x + 1;
            h = dw.max.y - dw.min.y + 1;

            // Read into OpenEXR Rgba (half-float RGBA)
            Imf::Array2D<Imf::Rgba> pixels(h, w);
            file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * w, 1, w);
            file.readPixels(dw.min.y, dw.max.y);

            // Convert half-float Rgba → float4
            data = (float*)malloc((size_t)w * h * 4 * sizeof(float));
            data_needs_free = true;
            if (!data) {
                std::cerr << "[hdri] out of memory for EXR: " << path << "\n";
                return map;
            }
            float* dst = data;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const Imf::Rgba& px = pixels[y][x];
                    *dst++ = (float)px.r;
                    *dst++ = (float)px.g;
                    *dst++ = (float)px.b;
                    *dst++ = (float)px.a;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[hdri] EXR load failed: " << e.what() << "\n";
            return map;
        }
    } else {
        // stbi_loadf handles .hdr, .png, .jpg, .tiff, etc.
        int ch = 0;
        data = stbi_loadf(path.c_str(), &w, &h, &ch, 4);
        data_needs_free = false;
        if (!data) {
            std::cerr << "[hdri] failed to load: " << path << "\n";
            return map;
        }
    }

    // Allocate a CUDA array for float4
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&map.arr, &fmt, (size_t)w, (size_t)h);

    // Upload — pitch = w * 4 floats * sizeof(float)
    cudaMemcpy2DToArray(
        map.arr, 0, 0,
        data,
        (size_t)w * 4 * sizeof(float),
        (size_t)w * 4 * sizeof(float),
        (size_t)h,
        cudaMemcpyHostToDevice);

    if (data_needs_free)
        free(data);
    else
        stbi_image_free(data);

    // Create texture object with bilinear filtering + wrap on U, clamp on V
    cudaResourceDesc res{};
    res.resType         = cudaResourceTypeArray;
    res.res.array.array = map.arr;

    cudaTextureDesc td{};
    td.addressMode[0]   = cudaAddressModeWrap;
    td.addressMode[1]   = cudaAddressModeClamp;
    td.filterMode       = cudaFilterModeLinear;
    td.readMode         = cudaReadModeElementType;
    td.normalizedCoords = 1;

    cudaCreateTextureObject(&map.tex, &res, &td, nullptr);

    map.width  = w;
    map.height = h;
    map.loaded = true;
    std::cout << "[hdri] loaded " << w << "x" << h << " from " << path
              << (is_exr ? " (EXR/OpenEXR)" : "") << "\n";
    return map;
}

void hdri_free(HdriMap& map)
{
    if (map.tex) cudaDestroyTextureObject(map.tex);
    if (map.arr) cudaFreeArray(map.arr);
    map = HdriMap{};
}
