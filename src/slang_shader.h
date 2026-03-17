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
//  'frame' increments every render frame — use it for animation.
//
//  Usage for static textures:
//      SlangShaderResult r = slang_shader_run(code, 256, 256, frame);
//
//  Usage for animated textures (cheap per-frame re-dispatch):
//      SlangShaderPipeline p = slang_shader_compile(code, 256, 256);
//      // each frame:
//      SlangShaderResult r = slang_shader_dispatch(p, frame);
//      // on cleanup:
//      slang_shader_destroy(p);
// ─────────────────────────────────────────────────────────────────────────────

// Must be called once after VkDevice is created.
void slang_shader_init(VkDevice device, VkPhysicalDevice phys,
                       VkCommandPool cmd_pool, VkQueue queue,
                       const char* slangc_path);

// Result of one dispatch — RGBA8 pixel data for 4 material maps.
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

// ─────────────────────────────────────────────────────────────────────────────
//  Persistent compiled pipeline — compile once, dispatch every frame cheaply.
// ─────────────────────────────────────────────────────────────────────────────
struct SlangShaderPipeline {
    bool        valid = false;
    std::string error_log;
    int         tex_w = 0;
    int         tex_h = 0;

    // All Vulkan objects owned by this pipeline:
    VkShaderModule        shmod     = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl       = VK_NULL_HANDLE;
    VkPipelineLayout      pl        = VK_NULL_HANDLE;
    VkPipeline            pipe      = VK_NULL_HANDLE;
    VkDescriptorPool      desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet       desc_set  = VK_NULL_HANDLE;
    VkBuffer              stage_buf = VK_NULL_HANDLE;
    VkDeviceMemory        stage_mem = VK_NULL_HANDLE;

    struct ImgBundle {
        VkImage        img  = VK_NULL_HANDLE;
        VkDeviceMemory mem  = VK_NULL_HANDLE;
        VkImageView    view = VK_NULL_HANDLE;
    };
    ImgBundle imgs[4];
};

// Compile Slang source → persistent Vulkan pipeline (expensive: launches slangc).
// Check result.valid; on failure result.error_log is set.
SlangShaderPipeline slang_shader_compile(const std::string& user_code,
                                         int width, int height);

// Re-run an already-compiled pipeline with a new frame count (cheap: GPU only).
SlangShaderResult   slang_shader_dispatch(SlangShaderPipeline& pipeline,
                                          int frame_count);

// Destroy all Vulkan objects owned by the pipeline (safe to call on invalid).
void                slang_shader_destroy(SlangShaderPipeline& pipeline);

// Convenience one-shot: compile + dispatch + destroy (same as before).
SlangShaderResult   slang_shader_run(const std::string& user_code,
                                     int width, int height, int frame_count);

extern const char* SLANG_SHADER_DEFAULT_CODE;
