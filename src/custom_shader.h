#pragma once
#include <string>
#include <vector>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  Runtime CUDA material shader compiler (NVRTC)
//
//  The user defines ONE function:
//
//    __device__ MatOut custom_material(float2 uv, float3 pos, float3 normal, int frame)
//
//  MatOut is pre-defined in the wrapper — no #include needed:
//
//    struct MatOut {
//        float3 base_color;  // [0,1]  albedo
//        float  roughness;   // [0,1]  0=mirror, 1=matte
//        float  metallic;    // [0,1]  0=dielectric, 1=metal
//        float3 emissive;    // [0,∞)  HDR glow — use (0,0,0) for none
//        float3 normal_ts;   // tangent-space normal — (0,0,1) = flat
//    };
//
//  All four outputs (base_color, metallic_rough, emissive, normal) are uploaded
//  as separate GPU textures and wired to the matching material slots.
// ─────────────────────────────────────────────────────────────────────────────

extern const char* CUSTOM_SHADER_DEFAULT_CODE;

struct CustomShaderResult {
    bool        success   = false;
    std::string error_log;

    // Four RGBA8 textures — all width×height×4 bytes, or empty on failure.
    std::vector<uint8_t> base_pixels;      // RGB = base_color,  A = 255
    std::vector<uint8_t> mr_pixels;        // R=0, G=roughness, B=metallic, A=255
    std::vector<uint8_t> emissive_pixels;  // RGB = emissive,   A = 255
    std::vector<uint8_t> normal_pixels;    // RGB = tangent-space normal encoded [0,1], A=255

    int width  = 256;
    int height = 256;
};

CustomShaderResult custom_shader_run(const std::string& user_code,
                                     int width, int height,
                                     int frame_count);
