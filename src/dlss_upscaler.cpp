#include "dlss_upscaler.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <windows.h>

#ifdef DLSS_ENABLED
#  include <sl.h>
#  include <sl_dlss.h>
#  include <sl_consts.h>    // sl::Constants (jitter, mvec, camera matrices)
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Shared helpers (always compiled)
// ─────────────────────────────────────────────────────────────────────────────

float dlss_scale_factor(int q)
{
    static const float scales[] = { 0.5f, 0.667f, 0.75f };
    return scales[q < 0 ? 0 : q > 2 ? 2 : q];
}

// Halton sequence base-2 (x) and base-3 (y) for sub-pixel jitter
static float halton(int i, int base)
{
    float f = 1.f, r = 0.f;
    while (i > 0) { f /= (float)base; r += f * (float)(i % base); i /= base; }
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
#ifdef DLSS_ENABLED
// ─────────────────────────────────────────────────────────────────────────────

// ── Vulkan image helpers ─────────────────────────────────────────────────────

static uint32_t find_mem_type(VkPhysicalDevice phys, uint32_t filter,
                               VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

static bool create_image(VkDevice dev, VkPhysicalDevice phys,
                          uint32_t w, uint32_t h, VkFormat fmt,
                          VkImageUsageFlags usage,
                          VkImage& img, VkDeviceMemory& mem, VkImageView& view)
{
    VkImageCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = fmt;
    ci.extent        = { w, h, 1 };
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev, &ci, nullptr, &img) != VK_SUCCESS) return false;

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, img, &mr);
    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = find_mem_type(phys, mr.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS) return false;
    vkBindImageMemory(dev, img, mem, 0);

    VkImageViewCreateInfo vi{};
    vi.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image                       = img;
    vi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    vi.format                      = fmt;
    vi.subresourceRange.aspectMask =
        (fmt == VK_FORMAT_D16_UNORM || fmt == VK_FORMAT_D32_SFLOAT)
        ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(dev, &vi, nullptr, &view) != VK_SUCCESS) return false;
    return true;
}

static void destroy_image(VkDevice dev, VkImage& img, VkDeviceMemory& mem, VkImageView& view)
{
    if (view) { vkDestroyImageView(dev, view, nullptr); view = VK_NULL_HANDLE; }
    if (img)  { vkDestroyImage   (dev, img,  nullptr); img  = VK_NULL_HANDLE; }
    if (mem)  { vkFreeMemory     (dev, mem,  nullptr); mem  = VK_NULL_HANDLE; }
}

static VkCommandBuffer begin_cmd(VkDevice dev, VkCommandPool pool)
{
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev, &ai, &cb);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

static void end_cmd(VkDevice dev, VkCommandPool pool, VkCommandBuffer cb, VkQueue q)
{
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(q);
    vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static void transition_image(VkCommandBuffer cb, VkImage img,
                               VkImageLayout old_layout, VkImageLayout new_layout)
{
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout           = old_layout;
    b.newLayout           = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = img;
    b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    b.srcAccessMask       = 0;
    b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b);
}

// ── Streamline quality enum ──────────────────────────────────────────────────

static sl::DLSSMode to_sl_quality(int q)
{
    switch (q) {
        case 0:  return sl::DLSSMode::eMaxPerformance;  // 50%
        case 2:  return sl::DLSSMode::eMaxQuality;      // 75%
        default: return sl::DLSSMode::eBalanced;        // 67%
    }
}

// ── Plugin directory (set by dlss_pre_vulkan_init) ───────────────────────────
static std::vector<std::string> g_plugin_paths;

// ─────────────────────────────────────────────────────────────────────────────
//  dlss_pre_vulkan_init — MUST be called before vkCreateInstance
// ─────────────────────────────────────────────────────────────────────────────

void dlss_pre_vulkan_init(const char* plugin_dir)
{
    if (plugin_dir && plugin_dir[0])
        g_plugin_paths = { std::string(plugin_dir) };

    // SL v2.x requires wide-char plugin paths
    static std::vector<std::wstring> g_wide_paths;
    g_wide_paths.clear();
    for (auto& p : g_plugin_paths) {
        int n = MultiByteToWideChar(CP_UTF8, 0, p.c_str(), -1, nullptr, 0);
        std::wstring w(n, 0);
        MultiByteToWideChar(CP_UTF8, 0, p.c_str(), -1, w.data(), n);
        g_wide_paths.push_back(std::move(w));
    }
    const wchar_t* paths[8] = {};
    uint32_t num_paths = 0;
    for (auto& w : g_wide_paths)
        if (num_paths < 8) paths[num_paths++] = w.c_str();

    sl::Preferences pref{};
    pref.showConsole         = false;
    pref.logLevel            = sl::LogLevel::eOff;
    pref.pathsToPlugins      = (num_paths > 0) ? paths : nullptr;
    pref.numPathsToPlugins   = num_paths;
    pref.engine              = sl::EngineType::eCustom;
    pref.renderAPI           = sl::RenderAPI::eVulkan;
    pref.flags               = sl::PreferenceFlags::eDisableCLStateTracking;

    sl::Result res = slInit(pref, sl::kSDKVersion);
    if (res != sl::Result::eOk) {
        fprintf(stderr, "[dlss] slInit failed (%d) — DLSS unavailable\n", (int)res);
    } else {
        printf("[dlss] Streamline SDK initialized (pre-Vulkan)\n");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  dlss_init
// ─────────────────────────────────────────────────────────────────────────────

bool dlss_init(DlssState& s,
               VkInstance /*instance*/, VkPhysicalDevice phys, VkDevice device,
               VkCommandPool cmd_pool, VkQueue queue,
               int display_w, int display_h, int quality_mode,
               int& render_w_out, int& render_h_out)
{
    dlss_free(s);
    s.device    = device;
    s.phys_dev  = phys;
    s.cmd_pool  = cmd_pool;
    s.queue     = queue;
    s.quality   = quality_mode;
    s.display_w = display_w;
    s.display_h = display_h;

    // ── Check DLSS support on this adapter ───────────────────────────────────
    sl::AdapterInfo adapter{};
    sl::Result sr = slIsFeatureSupported(sl::kFeatureDLSS, adapter);
    if (sr != sl::Result::eOk) {
        fprintf(stderr, "[dlss] DLSS not supported on this GPU (code %d)\n", (int)sr);
        fprintf(stderr, "[dlss] Requires RTX GPU + driver >= 520\n");
        return false;
    }

    // ── Query optimal render resolution ──────────────────────────────────────
    sl::DLSSOptimalSettings optimal{};
    sl::DLSSOptions         options{};
    options.mode         = to_sl_quality(quality_mode);
    options.outputWidth  = (uint32_t)display_w;
    options.outputHeight = (uint32_t)display_h;

    sl::ViewportHandle viewport{ s.viewport_id };
    slDLSSGetOptimalSettings(options, optimal);

    s.render_w = (int)optimal.optimalRenderWidth;
    s.render_h = (int)optimal.optimalRenderHeight;
    if (s.render_w < 1) s.render_w = (int)(display_w * dlss_scale_factor(quality_mode));
    if (s.render_h < 1) s.render_h = (int)(display_h * dlss_scale_factor(quality_mode));
    render_w_out = s.render_w;
    render_h_out = s.render_h;
    printf("[dlss] render=%dx%d  display=%dx%d  mode=%d\n",
           s.render_w, s.render_h, display_w, display_h, quality_mode);

    // Set DLSS options for this viewport
    slDLSSSetOptions(viewport, options);

    // ── Allocate auxiliary images ─────────────────────────────────────────────
    // Output: full-res RGBA16F
    if (!create_image(device, phys,
                      (uint32_t)display_w, (uint32_t)display_h,
                      VK_FORMAT_R16G16B16A16_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      s.output_image, s.output_memory, s.output_view)) {
        fprintf(stderr, "[dlss] Failed to create output image\n");
        return false;
    }

    // Motion vectors: render-res RG16F (zero = no camera motion data for path tracer)
    if (!create_image(device, phys,
                      (uint32_t)s.render_w, (uint32_t)s.render_h,
                      VK_FORMAT_R16G16_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      s.mv_image, s.mv_memory, s.mv_view)) {
        fprintf(stderr, "[dlss] Failed to create motion vector image\n");
        return false;
    }

    // Depth: render-res R32F (zero = path tracer has no geometry depth output)
    if (!create_image(device, phys,
                      (uint32_t)s.render_w, (uint32_t)s.render_h,
                      VK_FORMAT_R32_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      s.depth_image, s.depth_memory, s.depth_view)) {
        fprintf(stderr, "[dlss] Failed to create depth image\n");
        return false;
    }

    // Transition all images to GENERAL layout
    {
        VkCommandBuffer cb = begin_cmd(device, cmd_pool);
        transition_image(cb, s.output_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        transition_image(cb, s.mv_image,     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        transition_image(cb, s.depth_image,  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        end_cmd(device, cmd_pool, cb, queue);
    }

    s.available   = true;
    s.initialized = true;
    s.frame_idx   = 0;
    printf("[dlss] DLSS initialized via Streamline SDK\n");
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  dlss_get_jitter
// ─────────────────────────────────────────────────────────────────────────────

void dlss_get_jitter(DlssState& s, float& jx, float& jy)
{
    // 16-sample Halton sequence shifted to [-0.5, 0.5]
    int i = (s.frame_idx % 16) + 1;
    jx = halton(i, 2) - 0.5f;
    jy = halton(i, 3) - 0.5f;
    ++s.frame_idx;
}

// ─────────────────────────────────────────────────────────────────────────────
//  dlss_upscale
// ─────────────────────────────────────────────────────────────────────────────

void dlss_upscale(DlssState& s, VkCommandBuffer cmd_buf,
                  VkImage input_image, VkImageView input_view,
                  VkDeviceMemory input_memory, float /*exposure*/)
{
    if (!s.available) return;

    float jx = 0.f, jy = 0.f;
    dlss_get_jitter(s, jx, jy);

    sl::ViewportHandle viewport{ s.viewport_id };

    // ── Obtain per-frame token ────────────────────────────────────────────────
    sl::FrameToken* frame_token = nullptr;
    slGetNewFrameToken(frame_token);

    // ── Build Streamline resource descriptors ─────────────────────────────────
    // ResourceType::eTex2d is used for VkImage resources in SL v2.x
    // Color input (render-res, in GENERAL layout after CUDA write)
    sl::Resource colorRes(sl::ResourceType::eTex2d,
        (void*)input_image, (void*)input_memory, (void*)input_view,
        (uint32_t)VK_IMAGE_LAYOUT_GENERAL);
    // DLSS output (display-res)
    sl::Resource outputRes(sl::ResourceType::eTex2d,
        (void*)s.output_image, (void*)s.output_memory, (void*)s.output_view,
        (uint32_t)VK_IMAGE_LAYOUT_GENERAL);
    // Motion vectors (all zero — static/path-traced scene)
    sl::Resource mvRes(sl::ResourceType::eTex2d,
        (void*)s.mv_image, (void*)s.mv_memory, (void*)s.mv_view,
        (uint32_t)VK_IMAGE_LAYOUT_GENERAL);
    // Depth (all zero)
    sl::Resource depthRes(sl::ResourceType::eTex2d,
        (void*)s.depth_image, (void*)s.depth_memory, (void*)s.depth_view,
        (uint32_t)VK_IMAGE_LAYOUT_GENERAL);

    sl::Extent renderExtent  { 0, 0, (uint32_t)s.render_w,  (uint32_t)s.render_h  };
    sl::Extent displayExtent { 0, 0, (uint32_t)s.display_w, (uint32_t)s.display_h };

    sl::ResourceTag tags[] = {
        { &colorRes,  sl::kBufferTypeScalingInputColor,
          sl::ResourceLifecycle::eValidUntilPresent, &renderExtent  },
        { &outputRes, sl::kBufferTypeScalingOutputColor,
          sl::ResourceLifecycle::eValidUntilPresent, &displayExtent },
        { &mvRes,     sl::kBufferTypeMotionVectors,
          sl::ResourceLifecycle::eValidUntilPresent, &renderExtent  },
        { &depthRes,  sl::kBufferTypeDepth,
          sl::ResourceLifecycle::eValidUntilPresent, &renderExtent  },
    };
    slSetTag(viewport, tags, 4, cmd_buf);

    // ── Per-frame constants (jitter, camera params) ───────────────────────────
    sl::Constants constants{};
    constants.jitterOffset   = { jx, jy };
    constants.mvecScale      = { 1.f, 1.f };
    constants.cameraNear     = 0.01f;
    constants.cameraFar      = 10000.f;
    constants.reset          = (s.frame_idx <= 1) ? sl::Boolean::eTrue
                                                  : sl::Boolean::eFalse;
    slSetConstants(constants, *frame_token, viewport);

    // ── Evaluate DLSS ─────────────────────────────────────────────────────────
    const sl::BaseStructure* inputs[] = {
        static_cast<sl::BaseStructure*>(&viewport)
    };
    slEvaluateFeature(sl::kFeatureDLSS, *frame_token, inputs, 1, cmd_buf);

    // ── Blit upscaled output → interop image ─────────────────────────────────
    // DLSS wrote full-res result into s.output_image (R16G16B16A16_SFLOAT).
    // The viewport always reads from input_image (interop, R32G32B32A32_SFLOAT).
    // Copy back so the viewer sees the upscaled image, not the small raw render.
    {
        auto barrier = [&](VkImage img, VkImageLayout from, VkImageLayout to,
                           VkAccessFlags src_access, VkAccessFlags dst_access,
                           VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = from;
            b.newLayout           = to;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = img;
            b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            b.srcAccessMask       = src_access;
            b.dstAccessMask       = dst_access;
            vkCmdPipelineBarrier(cmd_buf, src_stage, dst_stage,
                                  0, 0, nullptr, 0, nullptr, 1, &b);
        };

        // output_image: GENERAL → TRANSFER_SRC
        barrier(s.output_image,
                VK_IMAGE_LAYOUT_GENERAL,       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_WRITE_BIT,    VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        // input_image (interop): UNDEFINED → TRANSFER_DST
        // Use UNDEFINED as old layout since we're overwriting the entire image —
        // this is correct regardless of whether the image was last used as
        // SHADER_READ_ONLY_OPTIMAL (first DLSS frame) or GENERAL (subsequent frames).
        barrier(input_image,
                VK_IMAGE_LAYOUT_UNDEFINED,     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0,                             VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkImageBlit region{};
        region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.srcOffsets[0]  = { 0, 0, 0 };
        region.srcOffsets[1]  = { (int32_t)s.display_w, (int32_t)s.display_h, 1 };
        region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.dstOffsets[0]  = { 0, 0, 0 };
        region.dstOffsets[1]  = { (int32_t)s.display_w, (int32_t)s.display_h, 1 };
        vkCmdBlitImage(cmd_buf,
                       s.output_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       input_image,    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region, VK_FILTER_LINEAR);

        // Restore both images to GENERAL for next frame
        barrier(s.output_image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_TRANSFER_READ_BIT,           VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Restore to SHADER_READ_ONLY_OPTIMAL — the post-process compute shader
        // and the ImGui descriptor were both created with this expected layout.
        barrier(input_image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT,           VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  dlss_free
// ─────────────────────────────────────────────────────────────────────────────

void dlss_free(DlssState& s)
{
    if (s.device == VK_NULL_HANDLE) { s = DlssState{}; return; }

    vkDeviceWaitIdle(s.device);

    // Streamline feature cleanup — SL owns the DLSS handle internally
    if (s.initialized) {
        sl::ViewportHandle vp{ s.viewport_id };
        slFreeResources(sl::kFeatureDLSS, vp);
    }

    destroy_image(s.device, s.output_image, s.output_memory, s.output_view);
    destroy_image(s.device, s.mv_image,     s.mv_memory,     s.mv_view);
    destroy_image(s.device, s.depth_image,  s.depth_memory,  s.depth_view);

    s = DlssState{};
}

// ─────────────────────────────────────────────────────────────────────────────
#else  // DLSS_ENABLED not defined — stub implementations
// ─────────────────────────────────────────────────────────────────────────────

void dlss_pre_vulkan_init(const char*) {}

bool dlss_init(DlssState&, VkInstance, VkPhysicalDevice, VkDevice, VkCommandPool, VkQueue,
               int display_w, int display_h, int quality,
               int& rw, int& rh)
{
    float scale = dlss_scale_factor(quality);
    rw = (int)(display_w * scale);
    rh = (int)(display_h * scale);
    return false;
}

void dlss_get_jitter(DlssState&, float& jx, float& jy) { jx = jy = 0.f; }

void dlss_upscale(DlssState&, VkCommandBuffer,
                  VkImage, VkImageView, VkDeviceMemory, float) {}

void dlss_free(DlssState&) {}

#endif
