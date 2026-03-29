#include "dlss_upscaler.h"
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <windows.h>
#include <vulkan/vulkan_win32.h>

#ifdef DLSS_ENABLED
#  include <sl.h>
#  include <sl_dlss.h>
#  include <sl_consts.h>       // sl::Constants (jitter, mvec, camera matrices)
#  include <sl_helpers.h>
#  include <sl_helpers_vk.h>   // slSetVulkanInfo — required before any feature query
#  include <sl_matrix_helpers.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Shared helpers (always compiled)
// ─────────────────────────────────────────────────────────────────────────────

float dlss_scale_factor(int q)
{
    static const float scales[] = { 0.333f, 0.5f, 0.667f, 0.75f, 0.77f, 1.0f };
    return scales[q < 0 ? 0 : q > 5 ? 5 : q];
}

// Halton sequence base-2 (x) and base-3 (y) for sub-pixel jitter
static float halton(int i, int base)
{
    float f = 1.f, r = 0.f;
    while (i > 0) { f /= (float)base; r += f * (float)(i % base); i /= base; }
    return r;
}

#ifdef DLSS_ENABLED
static sl::float4x4 make_perspective_matrix(float vfov_deg, float aspect, float z_near, float z_far)
{
    const float f = 1.0f / tanf(vfov_deg * 0.5f * 3.14159265f / 180.0f);
    const float a = z_far / (z_far - z_near);
    const float b = -(z_near * z_far) / (z_far - z_near);
    sl::float4x4 m{};
    // Streamline expects row-major matrices and multiplies them as matrix * vector.
    // Our camera view space uses +Z forward, so w_clip must be z_view.
    m[0] = sl::float4(f / aspect, 0.f, 0.f, 0.f);
    m[1] = sl::float4(0.f, f, 0.f, 0.f);
    m[2] = sl::float4(0.f, 0.f, a, b);
    m[3] = sl::float4(0.f, 0.f, 1.f, 0.f);
    return m;
}

static sl::float4x4 make_camera_to_world_matrix(const Camera& camera)
{
    sl::float3 right(camera.u.x, camera.u.y, camera.u.z);
    sl::float3 up(camera.v.x, camera.v.y, camera.v.z);
    sl::float3 fwd(-camera.w.x, -camera.w.y, -camera.w.z);
    sl::vectorNormalize(right);
    sl::vectorNormalize(fwd);
    sl::vectorCrossProduct(up, fwd, right);
    sl::vectorNormalize(up);
    return {
        sl::float4(right.x,          right.y,          right.z,          0.f),
        sl::float4(up.x,             up.y,             up.z,             0.f),
        sl::float4(fwd.x,            fwd.y,            fwd.z,            0.f),
        sl::float4(camera.origin.x,  camera.origin.y,  camera.origin.z,  1.f)
    };
}
#endif

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

static sl::Resource make_vk_resource(VkImage image, VkDeviceMemory memory, VkImageView view,
                                     VkImageLayout layout, uint32_t width, uint32_t height,
                                     VkFormat format, VkImageUsageFlags usage, VkImageCreateFlags flags = 0,
                                     uint32_t mip_levels = 1, uint32_t array_layers = 1)
{
    sl::Resource res(sl::ResourceType::eTex2d,
        (void*)image, (void*)memory, (void*)view, (uint32_t)layout);
    res.width = width;
    res.height = height;
    res.nativeFormat = (uint32_t)format;
    res.mipLevels = mip_levels;
    res.arrayLayers = array_layers;
    res.flags = (uint32_t)flags;
    res.usage = (uint32_t)usage;
    return res;
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

static bool format_to_channel_desc(VkFormat format, cudaChannelFormatDesc& desc)
{
    switch (format) {
        case VK_FORMAT_R32_SFLOAT:
            desc = cudaCreateChannelDesc<float>();
            return true;
        case VK_FORMAT_R32G32_SFLOAT:
            desc = cudaCreateChannelDesc<float2>();
            return true;
        default:
            return false;
    }
}

static bool create_cuda_interop_image(VkDevice dev, VkPhysicalDevice phys,
                                      VkCommandPool pool, VkQueue queue,
                                      uint32_t w, uint32_t h, VkFormat fmt,
                                      VkImageUsageFlags usage, CudaInterop& out)
{
    cudaChannelFormatDesc channel_desc{};
    if (!format_to_channel_desc(fmt, channel_desc)) return false;

    out.width = (int)w;
    out.height = (int)h;

    VkExternalMemoryImageCreateInfo ext_info{};
    ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
#ifdef _WIN32
    ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkImageCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.pNext         = &ext_info;
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = fmt;
    ci.extent        = { w, h, 1 };
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev, &ci, nullptr, &out.image) != VK_SUCCESS) return false;

    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(dev, out.image, &mr);

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
#ifdef _WIN32
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &export_info;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_mem_type(phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX) return false;
    if (vkAllocateMemory(dev, &ai, nullptr, &out.memory) != VK_SUCCESS) return false;
    vkBindImageMemory(dev, out.image, out.memory, 0);

#ifdef _WIN32
    HANDLE win_handle = nullptr;
    VkMemoryGetWin32HandleInfoKHR get_info{};
    get_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    get_info.memory = out.memory;
    get_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    auto get_handle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryWin32HandleKHR");
    if (!get_handle || get_handle(dev, &get_info, &win_handle) != VK_SUCCESS) return false;
#else
    int fd = -1;
    VkMemoryGetFdInfoKHR get_info{};
    get_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    get_info.memory = out.memory;
    get_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto get_handle = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
    if (!get_handle || get_handle(dev, &get_info, &fd) != VK_SUCCESS) return false;
#endif

    cudaExternalMemoryHandleDesc mem_desc{};
#ifdef _WIN32
    mem_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    mem_desc.handle.win32.handle = win_handle;
#else
    mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = fd;
#endif
    mem_desc.size = mr.size;
    if (cudaImportExternalMemory(&out.ext_mem, &mem_desc) != cudaSuccess) return false;

    cudaExternalMemoryMipmappedArrayDesc mip_desc{};
    mip_desc.offset = 0;
    mip_desc.formatDesc = channel_desc;
    mip_desc.extent = make_cudaExtent(w, h, 0);
    mip_desc.flags = cudaArraySurfaceLoadStore;
    mip_desc.numLevels = 1;
    if (cudaExternalMemoryGetMappedMipmappedArray(&out.mip_array, out.ext_mem, &mip_desc) != cudaSuccess) return false;
    if (cudaGetMipmappedArrayLevel(&out.array, out.mip_array, 0) != cudaSuccess) return false;

    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = out.array;
    if (cudaCreateSurfaceObject(&out.surface, &res_desc) != cudaSuccess) return false;

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = out.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(dev, &vi, nullptr, &out.image_view) != VK_SUCCESS) return false;

    VkCommandBuffer cb = begin_cmd(dev, pool);
    transition_image(cb, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    end_cmd(dev, pool, cb, queue);
    return true;
}

static void destroy_cuda_interop_image(VkDevice dev, CudaInterop& ci)
{
    if (ci.surface)   { cudaDestroySurfaceObject(ci.surface); ci.surface = 0; }
    if (ci.mip_array) { cudaFreeMipmappedArray(ci.mip_array); ci.mip_array = nullptr; }
    if (ci.ext_mem)   { cudaDestroyExternalMemory(ci.ext_mem); ci.ext_mem = nullptr; }
    if (ci.image_view){ vkDestroyImageView(dev, ci.image_view, nullptr); ci.image_view = VK_NULL_HANDLE; }
    if (ci.memory)    { vkFreeMemory(dev, ci.memory, nullptr); ci.memory = VK_NULL_HANDLE; }
    if (ci.image)     { vkDestroyImage(dev, ci.image, nullptr); ci.image = VK_NULL_HANDLE; }
    ci.array = nullptr;
}

// ── Streamline quality enum ──────────────────────────────────────────────────

static sl::DLSSMode to_sl_quality(int q)
{
    switch (q) {
        case 0:  return sl::DLSSMode::eUltraPerformance;
        case 1:  return sl::DLSSMode::eMaxPerformance;
        case 3:  return sl::DLSSMode::eMaxQuality;
        case 4:  return sl::DLSSMode::eUltraQuality;
        case 5:  return sl::DLSSMode::eDLAA;
        default: return sl::DLSSMode::eBalanced;
    }
}

// ── Plugin directory (set by dlss_pre_vulkan_init) ───────────────────────────
static std::vector<std::string> g_plugin_paths;
static bool                     g_sl_init_ok = false;  // true only if slInit succeeded

static void sl_log_callback(sl::LogType type, const char* msg)
{
    const char* tag = (type == sl::LogType::eError) ? "[SL ERROR]" :
                      (type == sl::LogType::eWarn)  ? "[SL WARN] " : "[SL INFO] ";
    fprintf(stderr, "%s %s\n", tag, msg);
    fflush(stderr);
}

static const char* get_env_str(const char* name)
{
    const char* value = std::getenv(name);
    return (value && value[0]) ? value : nullptr;
}

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
    static const sl::Feature features_to_load[] = { sl::kFeatureDLSS };
    pref.showConsole         = false;
    pref.logLevel            = sl::LogLevel::eDefault;  // errors + warnings + basic init info
    pref.logMessageCallback  = sl_log_callback;
    pref.pathsToPlugins      = (num_paths > 0) ? paths : nullptr;
    pref.numPathsToPlugins   = num_paths;
    pref.featuresToLoad      = features_to_load;
    pref.numFeaturesToLoad   = static_cast<uint32_t>(sizeof(features_to_load) / sizeof(features_to_load[0]));
    pref.engine              = sl::EngineType::eCustom;
    pref.renderAPI           = sl::RenderAPI::eVulkan;
    pref.flags               = sl::PreferenceFlags::eDisableCLStateTracking;
    if (const char* app_id = get_env_str("STREAMLINE_APP_ID"))
        pref.applicationId = static_cast<uint32_t>(std::strtoul(app_id, nullptr, 10));
    if (const char* project_id = get_env_str("STREAMLINE_PROJECT_ID"))
        pref.projectId = project_id;

    sl::Result res = slInit(pref, sl::kSDKVersion);
    if (res != sl::Result::eOk) {
        fprintf(stderr, "[dlss] slInit result: %s (%d)\n", sl::getResultAsStr(res), (int)res);
        fprintf(stderr, "[dlss] slInit failed (code %d) — check SL log above for reason\n", (int)res);
        fflush(stderr);
        g_sl_init_ok = false;
    } else {
        fprintf(stderr, "[dlss] Streamline SDK initialized OK\n");
        fflush(stderr);
        g_sl_init_ok = true;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  dlss_init
// ─────────────────────────────────────────────────────────────────────────────

bool dlss_init(DlssState& s,
               VkInstance instance, VkPhysicalDevice phys, VkDevice device,
               VkCommandPool cmd_pool, VkQueue queue,
               int display_w, int display_h, int quality_mode, float render_scale,
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

    // ── Bail early if slInit never succeeded ────────────────────────────────
    if (!g_sl_init_ok) {
        static bool s_warned = false;
        if (!s_warned) {
            fprintf(stderr, "[dlss] dlss_init skipped — slInit failed at startup\n");
            fflush(stderr);
            s_warned = true;
        }
        return false;
    }

    // With sl.interposer linked, Streamline hooks vkCreateInstance/vkCreateDevice
    // and initializes plugins during device creation.



    // slSetVulkanInfo is only needed when bypassing the interposer entirely —

    // ── Diagnostic: is sl.dlss.dll actually loaded into the process? ─────────
    {
        HMODULE hDlss = GetModuleHandleA("sl.dlss.dll");
        if (!hDlss) hDlss = GetModuleHandleA("sl.dlss_d.dll");
        if (hDlss) {
            fprintf(stderr, "[dlss] sl.dlss plugin module found in process (OK)\n");
        } else {
            fprintf(stderr, "[dlss] sl.dlss plugin module not visible via GetModuleHandle yet\n");
            fprintf(stderr, "[dlss] WARNING: if you are using Streamline production DLLs without NVIDIA onboarding,\n"
                            "[dlss]   slIsFeatureSupported may report FeatureNotSupported even when the files are present\n"
                            "[dlss]   for local development prefer the development DLLs copied by CMake\n");
        }
        fflush(stderr);
    }

    // ── Find the physical device enumeration index (required by AdapterInfo) ─
    // slIsFeatureSupported correlates the physical device by both handle AND
    // enumeration index; providing only the handle can return code 31.
    uint32_t phys_index = 0;
    {
        uint32_t dev_count = 0;
        vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
        std::vector<VkPhysicalDevice> devs(dev_count);
        vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());
        for (uint32_t i = 0; i < dev_count; ++i)
            if (devs[i] == phys) { phys_index = i; break; }
        fprintf(stderr, "[dlss] physical device index = %u  (of %u)\n", phys_index, dev_count);
        fflush(stderr);
    }

    // ── Check DLSS support on this adapter ───────────────────────────────────
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = phys;
    // Note: vkPhysicalDeviceIndex is not a member of this SL SDK version's AdapterInfo;
    // phys_index logged above for diagnostic purposes only.
    (void)phys_index;
    sl::Result sr = slIsFeatureSupported(sl::kFeatureDLSS, adapter);
    if (sr != sl::Result::eOk) {
        static bool s_warned = false;
        if (!s_warned) {
            fprintf(stderr, "[dlss] slIsFeatureSupported(kFeatureDLSS) result: %s (%d)\n",
                    sl::getResultAsStr(sr), (int)sr);
            const char* reason =
                (int)sr == 7  ? "no supported adapter found by SL" :
                (int)sr == 8  ? "adapter not supported (need RTX GPU)" :
                (int)sr == 9  ? "no plugins found — sl.dlss.dll missing next to exe" :
                (int)sr == 21 ? "driver too old (need >= 520)" :
                (int)sr == 22 ? "feature not supported on this GPU" :
                (int)sr == 32 ? "feature blocked by production runtime or unsupported adapter; use development DLLs for local testing or set STREAMLINE_APP_ID / STREAMLINE_PROJECT_ID" :
                                "unknown error";
            fprintf(stderr, "[dlss] slIsFeatureSupported(kFeatureDLSS) failed: code %d — %s\n",
                    (int)sr, reason);
            fflush(stderr);
            s_warned = true;
        }
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

    auto clamp_int = [](int v, int lo, int hi) {
        return (v < lo) ? lo : (v > hi ? hi : v);
    };
    auto round_even = [](float v) {
        int iv = (int)std::lround(v);
        return (iv > 1) ? (iv & ~1) : 1;
    };

    const float safe_scale = (render_scale < 0.01f) ? 0.01f : (render_scale > 1.0f ? 1.0f : render_scale);
    int desired_w = round_even((float)display_w * safe_scale);
    int desired_h = round_even((float)display_h * safe_scale);

    int min_w = optimal.renderWidthMin  ? (int)optimal.renderWidthMin  : 1;
    int min_h = optimal.renderHeightMin ? (int)optimal.renderHeightMin : 1;
    int max_w = optimal.renderWidthMax  ? (int)optimal.renderWidthMax  : display_w;
    int max_h = optimal.renderHeightMax ? (int)optimal.renderHeightMax : display_h;

    s.render_w = clamp_int(desired_w, min_w, max_w);
    s.render_h = clamp_int(desired_h, min_h, max_h);

    if (s.render_w < 1) s.render_w = round_even((float)display_w * dlss_scale_factor(quality_mode));
    if (s.render_h < 1) s.render_h = round_even((float)display_h * dlss_scale_factor(quality_mode));
    render_w_out = s.render_w;
    render_h_out = s.render_h;
    printf("[dlss] render=%dx%d  display=%dx%d  requested_scale=%.3f  mode=%d  optimal=%ux%u  range=[%u..%u]x[%u..%u]\n",
           s.render_w, s.render_h, display_w, display_h, safe_scale, quality_mode,
           optimal.optimalRenderWidth, optimal.optimalRenderHeight,
           optimal.renderWidthMin, optimal.renderWidthMax,
           optimal.renderHeightMin, optimal.renderHeightMax);

    // Set DLSS options for this viewport
    slDLSSSetOptions(viewport, options);

    // ── Allocate auxiliary images ─────────────────────────────────────────────
    // Output: full-res RGBA16F
    if (!create_image(device, phys,
                      (uint32_t)display_w, (uint32_t)display_h,
                      VK_FORMAT_R16G16B16A16_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      s.output_image, s.output_memory, s.output_view)) {
        fprintf(stderr, "[dlss] Failed to create output image\n");
        return false;
    }

    // Motion vectors: render-res RG32F written from CUDA in pixel space.
    if (!create_cuda_interop_image(device, phys, cmd_pool, queue,
                                   (uint32_t)s.render_w, (uint32_t)s.render_h,
                                   VK_FORMAT_R32G32_SFLOAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                   s.mv_interop)) {
        fprintf(stderr, "[dlss] Failed to create motion vector image\n");
        return false;
    }

    // Depth: render-res R32F written from CUDA from the primary hit.
    if (!create_cuda_interop_image(device, phys, cmd_pool, queue,
                                   (uint32_t)s.render_w, (uint32_t)s.render_h,
                                   VK_FORMAT_R32_SFLOAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                   s.depth_interop)) {
        fprintf(stderr, "[dlss] Failed to create depth image\n");
        return false;
    }
    cudaMalloc(&s.motion_buffer, (size_t)s.render_w * s.render_h * sizeof(float2));
    cudaMalloc(&s.depth_buffer,  (size_t)s.render_w * s.render_h * sizeof(float));
    cudaMemset(s.motion_buffer, 0, (size_t)s.render_w * s.render_h * sizeof(float2));
    cudaMemset(s.depth_buffer,  0, (size_t)s.render_w * s.render_h * sizeof(float));

    // Transition output image to GENERAL layout
    {
        VkCommandBuffer cb = begin_cmd(device, cmd_pool);
        transition_image(cb, s.output_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        end_cmd(device, cmd_pool, cb, queue);
    }

    s.available   = true;
    s.initialized = true;
    s.frame_idx   = 0;
    s.prev_camera_valid = false;
    s.prev_render_camera_valid = false;
    printf("[dlss] DLSS initialized via Streamline SDK\n");
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  dlss_get_jitter
// ─────────────────────────────────────────────────────────────────────────────

void dlss_get_jitter(DlssState& s, float& jx, float& jy)
{
    // This renderer already performs progressive temporal accumulation in d_accum.
    // Feeding an additional DLSS jitter sequence into that accumulated buffer
    // creates visible shimmer, so keep the DLSS path unjittered here.
    jx = 0.f;
    jy = 0.f;
    s.jitter_x = jx;
    s.jitter_y = jy;
}

// ─────────────────────────────────────────────────────────────────────────────
//  dlss_upscale
// ─────────────────────────────────────────────────────────────────────────────

void dlss_upscale(DlssState& s, VkCommandBuffer cmd_buf,
                  VkImage input_image, VkImageView input_view,
                  VkDeviceMemory input_memory, float /*exposure*/,
                  const Camera& camera, float vfov_deg, float aspect_ratio,
                  bool reset_history)
{
    if (!s.available) return;

    const float jx = s.jitter_x;
    const float jy = s.jitter_y;

    sl::ViewportHandle viewport{ s.viewport_id };

    // ── Obtain per-frame token ────────────────────────────────────────────────
    sl::FrameToken* frame_token = nullptr;
    slGetNewFrameToken(frame_token);

    // ── Build Streamline resource descriptors ─────────────────────────────────
    // ResourceType::eTex2d is used for VkImage resources in SL v2.x
    // Color input: physical image is display_w × display_h (rt_w × rt_h).
    // The resource dimensions must match the actual VkImage allocation so that
    // Streamline computes correct row strides. renderExtent tells DLSS which
    // sub-region to read (the render_w × render_h top-left that the pathtracer wrote).
    sl::Resource colorRes = make_vk_resource(
        input_image, input_memory, input_view,
        VK_IMAGE_LAYOUT_GENERAL,
        (uint32_t)s.display_w, (uint32_t)s.display_h,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    // DLSS output (display-res)
    sl::Resource outputRes = make_vk_resource(
        s.output_image, s.output_memory, s.output_view,
        VK_IMAGE_LAYOUT_GENERAL,
        (uint32_t)s.display_w, (uint32_t)s.display_h,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    // Motion vectors (all zero — static/path-traced scene)
    sl::Resource mvRes = make_vk_resource(
        s.mv_interop.image, s.mv_interop.memory, s.mv_interop.image_view,
        VK_IMAGE_LAYOUT_GENERAL,
        (uint32_t)s.render_w, (uint32_t)s.render_h,
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    // Depth
    sl::Resource depthRes = make_vk_resource(
        s.depth_interop.image, s.depth_interop.memory, s.depth_interop.image_view,
        VK_IMAGE_LAYOUT_GENERAL,
        (uint32_t)s.render_w, (uint32_t)s.render_h,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    sl::Extent renderExtent  { 0, 0, (uint32_t)s.render_w,  (uint32_t)s.render_h  };
    sl::Extent displayExtent { 0, 0, (uint32_t)s.display_w, (uint32_t)s.display_h };

    sl::ResourceTag tags[] = {
        { &colorRes,  sl::kBufferTypeScalingInputColor,
          sl::ResourceLifecycle::eValidUntilEvaluate, &renderExtent  },
        { &outputRes, sl::kBufferTypeScalingOutputColor,
          sl::ResourceLifecycle::eValidUntilEvaluate, &displayExtent },
        { &mvRes,     sl::kBufferTypeMotionVectors,
          sl::ResourceLifecycle::eValidUntilEvaluate, &renderExtent  },
        { &depthRes,  sl::kBufferTypeDepth,
          sl::ResourceLifecycle::eValidUntilEvaluate, &renderExtent  },
    };
    slSetTag(viewport, tags, 4, cmd_buf);

    // ── Per-frame constants (jitter, camera params) ───────────────────────────
    sl::Constants constants{};
    constants.jitterOffset   = { jx, jy };
    constants.mvecScale      = { 1.f / (float)std::max(1, s.render_w),
                                 1.f / (float)std::max(1, s.render_h) };
    constants.cameraNear     = 0.01f;
    constants.cameraFar      = 10000.f;
    constants.cameraPos      = { camera.origin.x, camera.origin.y, camera.origin.z };
    constants.cameraUp       = { camera.v.x, camera.v.y, camera.v.z };
    constants.cameraRight    = { camera.u.x, camera.u.y, camera.u.z };
    constants.cameraFwd      = { -camera.w.x, -camera.w.y, -camera.w.z };
    constants.cameraFOV      = vfov_deg * (3.14159265f / 180.f);
    constants.cameraAspectRatio = aspect_ratio;
    constants.motionVectorsInvalidValue = 0.f;
    constants.depthInverted  = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eFalse;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    constants.reset          = (reset_history || s.frame_idx <= 1) ? sl::Boolean::eTrue
                                                                   : sl::Boolean::eFalse;
    constants.cameraViewToClip = make_perspective_matrix(vfov_deg, aspect_ratio, constants.cameraNear, constants.cameraFar);
    sl::matrixFullInvert(constants.clipToCameraView, constants.cameraViewToClip);
    {
        const bool have_prev = s.prev_camera_valid && !reset_history && s.frame_idx > 0;
        const Camera& prev_camera = have_prev ? s.prev_camera : camera;
        sl::float4x4 camera_view_to_world = make_camera_to_world_matrix(camera);
        sl::float4x4 prev_camera_view_to_world = make_camera_to_world_matrix(prev_camera);
        sl::float4x4 camera_to_prev_camera{};
        sl::calcCameraToPrevCamera(camera_to_prev_camera,
                                   camera_view_to_world,
                                   prev_camera_view_to_world);
        sl::float4x4 clip_to_prev_camera_view{};
        sl::matrixMul(clip_to_prev_camera_view, constants.clipToCameraView, camera_to_prev_camera);
        sl::matrixMul(constants.clipToPrevClip, clip_to_prev_camera_view, constants.cameraViewToClip);
        sl::matrixFullInvert(constants.prevClipToClip, constants.clipToPrevClip);
    }
    slSetConstants(constants, *frame_token, viewport);

    // ── Evaluate DLSS ─────────────────────────────────────────────────────────
    const sl::BaseStructure* inputs[] = {
        static_cast<sl::BaseStructure*>(&viewport)
    };
    slEvaluateFeature(sl::kFeatureDLSS, *frame_token, inputs, 1, cmd_buf);
    s.prev_camera = camera;
    s.prev_camera_valid = true;

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

    ++s.frame_idx;
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
    destroy_cuda_interop_image(s.device, s.mv_interop);
    destroy_cuda_interop_image(s.device, s.depth_interop);
    if (s.motion_buffer) { cudaFree(s.motion_buffer); s.motion_buffer = nullptr; }
    if (s.depth_buffer)  { cudaFree(s.depth_buffer);  s.depth_buffer  = nullptr; }
    s.prev_camera_valid = false;
    s.prev_render_camera_valid = false;

    s = DlssState{};
}

bool dlss_get_debug_state(const DlssState& s, DlssDebugState& out_state)
{
    out_state = {};
    if (!g_sl_init_ok || !s.initialized || !s.available)
        return false;

    sl::ViewportHandle viewport{ s.viewport_id };
    sl::DLSSState state{};
    sl::Result res = slDLSSGetState(viewport, state);
    if (res != sl::Result::eOk)
        return false;

    out_state.estimated_vram_bytes = state.estimatedVRAMUsageInBytes;
    return true;
}

bool dlss_query_feature_support(VkInstance, VkPhysicalDevice phys, DlssFeatureSupport& out_support)
{
    out_support = {};
    if (!g_sl_init_ok || phys == VK_NULL_HANDLE)
        return false;

    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = phys;

    out_support.query_ok = true;
    out_support.dlss_sr_supported = (slIsFeatureSupported(sl::kFeatureDLSS, adapter) == sl::Result::eOk);
    out_support.dlss_rr_supported = (slIsFeatureSupported(sl::kFeatureDLSS_RR, adapter) == sl::Result::eOk);
    out_support.dlss_fg_supported = (slIsFeatureSupported(sl::kFeatureDLSS_G, adapter) == sl::Result::eOk);
    return true;
}

void dlss_shutdown()
{
    if (!g_sl_init_ok)
        return;

    sl::Result res = slShutdown();
    if (res != sl::Result::eOk) {
        fprintf(stderr, "[dlss] slShutdown result: %s (%d)\n", sl::getResultAsStr(res), (int)res);
        fflush(stderr);
    }
    g_sl_init_ok = false;
}

// ─────────────────────────────────────────────────────────────────────────────
#else  // DLSS_ENABLED not defined — stub implementations
// ─────────────────────────────────────────────────────────────────────────────

void dlss_pre_vulkan_init(const char*) {}

bool dlss_init(DlssState&, VkInstance, VkPhysicalDevice, VkDevice, VkCommandPool, VkQueue,
               int display_w, int display_h, int quality, float render_scale,
               int& rw, int& rh)
{
    float scale = (render_scale < 0.01f) ? 0.01f : (render_scale > 1.0f ? 1.0f : render_scale);
    rw = (int)(display_w * scale);
    rh = (int)(display_h * scale);
    return false;
}

void dlss_get_jitter(DlssState&, float& jx, float& jy) { jx = jy = 0.f; }

void dlss_upscale(DlssState&, VkCommandBuffer,
                  VkImage, VkImageView, VkDeviceMemory, float,
                  const Camera&, float, float, bool) {}

void dlss_free(DlssState&) {}

bool dlss_get_debug_state(const DlssState&, DlssDebugState&)
{
    return false;
}

bool dlss_query_feature_support(VkInstance, VkPhysicalDevice, DlssFeatureSupport& out_support)
{
    out_support = {};
    return false;
}

void dlss_shutdown() {}

#endif
