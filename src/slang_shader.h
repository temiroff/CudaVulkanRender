#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  Runtime Slang material shader compiler
//
//  The user defines ONE function in Slang:
//
//    MatOut custom_material(float2 uv, float3 pos, float3 normal, int frame)
//
//  MatOut is pre-defined in the wrapper — no import needed:
//
//    struct MatOut {
//        float3 base_color;  // [0,1]  albedo
//        float  roughness;   // [0,1]  0=mirror, 1=matte
//        float  metallic;    // [0,1]  0=dielectric, 1=metal
//        float3 emissive;    // [0,∞)  HDR glow
//        float3 normal_ts;   // tangent-space normal — float3(0,0,1) = flat
//    };
//
//  The same four textures (base, metallic_rough, emissive, normal) are produced.
// ─────────────────────────────────────────────────────────────────────────────

// Must be called once after VkDevice is created.
// slangc_path: full path to slangc.exe (or "slangc" if it is in PATH).
void slang_shader_init(VkDevice device, VkPhysicalDevice phys,
                       VkCommandPool cmd_pool, VkQueue queue,
                       const char* slangc_path);

struct SlangShaderResult {
    bool        success = false;
    std::string error_log;

    std::vector<uint8_t> base_pixels;
    std::vector<uint8_t> mr_pixels;
    std::vector<uint8_t> emissive_pixels;
    std::vector<uint8_t> normal_pixels;

    int width  = 0;
    int height = 0;
};

SlangShaderResult slang_shader_run(const std::string& user_code,
                                   int width, int height, int frame_count);

extern const char* SLANG_SHADER_DEFAULT_CODE;
