#pragma once
#include <vulkan/vulkan.h>

// Must match push constant layout in post.slang
struct PostPushConstants {
    float exposure           = 0.0f;   // EV stops: 0=neutral, +1=2x brighter, -1=2x darker
    int   tone_map_mode      = 3;      // 0=None, 1=Reinhard, 2=ACES, 3=AgX, 4=Filmic, 5=Khronos
    float bloom_strength     = 0.0f;   // 0 = disabled
    float bloom_threshold    = 0.8f;
    float vignette_strength  = 0.224f;
    float saturation         = 1.0f;
    int   width              = 0;
    int   height             = 0;
    float vignette_falloff   = 1.712f;
    int   look               = 0;      // contrast look: 0=None,1=VeryLow..7=VeryHigh (Blender-style)
    float gamma              = 1.0f;   // display gamma: 1=neutral (Blender default)
};

struct PostProcess {
    VkDevice              device          = VK_NULL_HANDLE;
    VkPipeline            pipeline        = VK_NULL_HANDLE;
    VkPipelineLayout      layout          = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl             = VK_NULL_HANDLE;
    VkDescriptorSet       compute_desc    = VK_NULL_HANDLE;  // for compute shader
    VkDescriptorPool      pool            = VK_NULL_HANDLE;
    // Display image (LDR output, sampled by ImGui)
    VkImage               display_image   = VK_NULL_HANDLE;
    VkDeviceMemory        display_memory  = VK_NULL_HANDLE;
    VkImageView           display_view    = VK_NULL_HANDLE;
    VkSampler             display_sampler = VK_NULL_HANDLE;
    VkDescriptorSet       imgui_desc      = VK_NULL_HANDLE;  // ImGui::Image() handle
    // AgX 3D LUT (binding 3) — exact Blender AgX_Base_sRGB.cube
    VkImage               lut_image       = VK_NULL_HANDLE;
    VkDeviceMemory        lut_memory      = VK_NULL_HANDLE;
    VkImageView           lut_view        = VK_NULL_HANDLE;
    VkSampler             lut_sampler     = VK_NULL_HANDLE;
    int                   lut_size        = 0;  // N for NxNxN (0 = not loaded)
    int width  = 0;
    int height = 0;
};

// Call once at startup.
// hdr_view/hdr_sampler: the CUDA interop image view + sampler (combined image sampler, SHADER_READ_ONLY)
// imgui_pool: VulkanContext::descriptorPool (for ImGui::Image registration)
PostProcess post_process_create(
    VkDevice         device,
    VkPhysicalDevice phys,
    VkDescriptorPool imgui_pool,
    VkImageView      hdr_view,
    VkSampler        hdr_sampler,
    VkCommandPool    cmd_pool,
    VkQueue          queue,
    int width, int height);

// Call on viewport resize. Recreates the display image and updates descriptors.
void post_process_resize(
    PostProcess&     pp,
    VkPhysicalDevice phys,
    VkDescriptorPool imgui_pool,
    VkCommandPool    cmd_pool,
    VkQueue          queue,
    VkImageView      hdr_view,
    VkSampler        hdr_sampler,
    int new_w, int new_h);

// Dispatch the compute shader. Call inside a command buffer, outside a render pass.
// Inserts the necessary pipeline barriers for the display image (GENERAL layout for write,
// then memory barrier so fragment shader (ImGui) can read it).
void post_process_dispatch(
    PostProcess&           pp,
    VkCommandBuffer        cb,
    const PostPushConstants& params);

void post_process_destroy(PostProcess& pp);
