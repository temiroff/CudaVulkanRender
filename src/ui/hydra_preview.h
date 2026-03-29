#pragma once
#include <string>
#include <vulkan/vulkan.h>

struct AnimPanelState;

struct HydraPreviewState {
    // Camera — eye position, look target, and FOV, synced from main viewport each frame.
    float pos[3]   = { 0.f, 1.f, 5.f };
    float pivot[3] = { 0.f, 0.f, 0.f };
    float vfov     = 60.f;

    // RMB pick — set pick_requested + pixel before tick; read results after.
    bool  pick_requested    = false;
    int   pick_px           = 0;
    int   pick_py           = 0;
    float pick_result_dist  = -1.f;          // distance eye→hit, -1 = miss
    float pick_result_hit[3] = { 0.f, 0.f, 0.f }; // world-space hit position

    // Opaque USD/Hydra objects — defined in .cpp to keep USD headers out of here
    void* stage_ref  = nullptr;   // UsdStageRefPtr*
    void* engine_ptr = nullptr;   // UsdImagingGLEngine*
    bool  loaded     = false;

    // Offscreen WGL context so HdStorm GL Hgi has an OpenGL 4.5 context to use.
    void* wgl_hwnd  = nullptr;    // HWND
    void* wgl_hdc   = nullptr;    // HDC
    void* wgl_hglrc = nullptr;    // HGLRC

    // Vulkan upload resources for displaying Hydra output in ImGui.
    VkDevice         vk_device         = VK_NULL_HANDLE;
    VkPhysicalDevice vk_physical       = VK_NULL_HANDLE;
    VkQueue          vk_queue          = VK_NULL_HANDLE;
    uint32_t         vk_queue_family   = UINT32_MAX;
    VkCommandPool    vk_command_pool   = VK_NULL_HANDLE;
    VkSampler        vk_sampler        = VK_NULL_HANDLE;
    VkImage          vk_image          = VK_NULL_HANDLE;
    VkDeviceMemory   vk_image_memory   = VK_NULL_HANDLE;
    VkImageView      vk_image_view     = VK_NULL_HANDLE;
    VkBuffer         vk_staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory   vk_staging_memory = VK_NULL_HANDLE;
    void*            vk_staging_mapped = nullptr;
    size_t           vk_staging_size   = 0;
    VkImageLayout    vk_image_layout   = VK_IMAGE_LAYOUT_UNDEFINED;
    int              tex_w             = 0;
    int              tex_h             = 0;
    VkDescriptorSet  imgui_desc        = VK_NULL_HANDLE;
};

// Initialize Hydra engine and upload resources.
bool hydra_preview_init(
    HydraPreviewState& hp,
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkQueue graphics_queue,
    uint32_t graphics_queue_family,
    VkCommandPool command_pool);

// Release all resources.
void hydra_preview_destroy(HydraPreviewState& hp);

// Load a USD file. The stage is kept alive for animation playback.
void hydra_preview_load   (HydraPreviewState& hp, const std::string& path);

// Render Hydra at anim.current_time into an internal Vulkan texture.
bool hydra_preview_tick(HydraPreviewState& hp, const AnimPanelState& anim, int width, int height);

// Descriptor for ImGui::Image().
VkDescriptorSet hydra_preview_descriptor(const HydraPreviewState& hp);
