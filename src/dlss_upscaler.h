#pragma once
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>
#include "scene.h"
#include "cuda_interop.h"

// DLSS (Deep Learning Super Sampling) via NVIDIA Streamline SDK.
// Falls back gracefully when DLSS_ENABLED is not defined.
//
// Download Streamline SDK (open source, prebuilt binaries):
//   https://github.com/NVIDIA-RTX/Streamline/releases
// Set DLSS_SDK_DIR in cmake-gui to the extracted SDK root (contains include/sl.h).
//
// !! IMPORTANT: dlss_pre_vulkan_init() MUST be called before any vkCreateInstance !!
//
// Integration flow:
//   1. dlss_pre_vulkan_init(plugin_dir)  — before vulkan_context_create()
//   2. dlss_init(...)                    — after VkDevice is created
//   3. Per frame: dlss_get_jitter(), dlss_upscale()
//   4. dlss_free()                       — at shutdown

struct DlssState {
    // Streamline viewport ID (arbitrary uint32 — we use 0)
    uint32_t viewport_id = 0;

    // Vulkan resources — DLSS output image (full resolution, RGBA16F)
    VkImage        output_image  = VK_NULL_HANDLE;
    VkDeviceMemory output_memory = VK_NULL_HANDLE;
    VkImageView    output_view   = VK_NULL_HANDLE;

    // Motion vector image (render-res, camera/object motion in pixel space)
    CudaInterop    mv_interop;

    // Depth image (render-res, non-linear depth)
    CudaInterop    depth_interop;

    float2*        motion_buffer = nullptr;
    float*         depth_buffer  = nullptr;

    VkDevice         device    = VK_NULL_HANDLE;
    VkPhysicalDevice phys_dev  = VK_NULL_HANDLE;
    VkCommandPool    cmd_pool  = VK_NULL_HANDLE;
    VkQueue          queue     = VK_NULL_HANDLE;

    int render_w   = 0;   // low-res input width
    int render_h   = 0;   // low-res input height
    int display_w  = 0;   // full-res output width
    int display_h  = 0;   // full-res output height
    int quality    = 1;   // 0=performance, 1=balanced, 2=quality

    int  frame_idx   = 0;
    float jitter_x   = 0.f;
    float jitter_y   = 0.f;
    Camera prev_camera{};
    bool   prev_camera_valid = false;
    Camera prev_render_camera{};
    bool   prev_render_camera_valid = false;
    bool available   = false;
    bool initialized = false;
};

struct DlssDebugState {
    uint64_t estimated_vram_bytes = 0;
};

struct DlssFeatureSupport {
    bool query_ok = false;
    bool dlss_sr_supported = false;
    bool dlss_rr_supported = false;
    bool dlss_fg_supported = false;
};

// Quality mode → scale factor: 0→0.5, 1→0.667, 2→0.75
float dlss_scale_factor(int quality_mode);

// STEP 1: Must be called before any Vulkan call (before vkCreateInstance).
// plugin_dir: path to the directory containing sl.dlss.dll, sl.common.dll etc.
//             (typically the same directory as your .exe, or the SDK bin/x64/ dir).
// Safe to call when DLSS_ENABLED is not defined — does nothing.
void dlss_pre_vulkan_init(const char* plugin_dir);

// STEP 2: Initialize DLSS after VkDevice is created.
// display_w/h: full output resolution. quality_mode: 0-2.
// Computed render resolution returned in render_w_out / render_h_out.
bool dlss_init(DlssState& s,
               VkInstance instance, VkPhysicalDevice phys, VkDevice device,
               VkCommandPool cmd_pool, VkQueue queue,
               int display_w, int display_h, int quality_mode, float render_scale,
               int& render_w_out, int& render_h_out);

// Halton jitter for the current frame (call before dispatching the path tracer).
void dlss_get_jitter(DlssState& s, float& jx, float& jy);

// Run DLSS upscaling, recording work into cmd_buf.
// input_image/view/memory: the low-res rendered image (CUDA interop surface).
// After this call, s.output_image contains the upscaled full-res result.
void dlss_upscale(DlssState& s, VkCommandBuffer cmd_buf,
                  VkImage input_image, VkImageView input_view,
                  VkDeviceMemory input_memory, float exposure,
                  const Camera& camera, float vfov_deg, float aspect_ratio,
                  bool reset_history);

// Free all resources.
void dlss_free(DlssState& s);

// Query live DLSS runtime state from Streamline for the current viewport.
bool dlss_get_debug_state(const DlssState& s, DlssDebugState& out_state);

// Query which DLSS feature families are supported by the current adapter/runtime.
bool dlss_query_feature_support(VkInstance instance, VkPhysicalDevice phys, DlssFeatureSupport& out_support);

// Shut down the global Streamline runtime before destroying VkDevice/VkInstance.
void dlss_shutdown();
