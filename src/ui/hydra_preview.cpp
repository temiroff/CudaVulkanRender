#include "hydra_preview.h"
#include "anim_panel.h"

#include <backends/imgui_impl_vulkan.h>

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/base/gf/range1d.h>
#include <pxr/base/gf/range2d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/tf/token.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/glf/simpleLight.h>
#include <pxr/imaging/glf/simpleMaterial.h>
#include <pxr/usdImaging/usdImagingGL/engine.h>
#include <pxr/usdImaging/usdImagingGL/renderParams.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/boundableLightBase.h>
#include <pxr/usd/usdLux/nonboundableLightBase.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/primRange.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <gl/GL.h>
#endif

PXR_NAMESPACE_USING_DIRECTIVE

static UsdStageRefPtr& get_stage(HydraPreviewState& hp)
{
    return *static_cast<UsdStageRefPtr*>(hp.stage_ref);
}

static UsdImagingGLEngine& get_engine(HydraPreviewState& hp)
{
    return *static_cast<UsdImagingGLEngine*>(hp.engine_ptr);
}

// ---- WGL offscreen OpenGL context (needed so HdStorm GL Hgi can initialize) ----
#ifdef _WIN32
typedef HGLRC (WINAPI* PFN_wglCreateContextAttribsARB)(HDC, HGLRC, const int*);

static bool wgl_create_context(HydraPreviewState& hp)
{
    static bool wc_registered = false;
    HINSTANCE hInst = GetModuleHandle(nullptr);
    if (!wc_registered) {
        WNDCLASSA wc = {};
        wc.style         = CS_OWNDC;
        wc.lpfnWndProc   = DefWindowProcA;
        wc.hInstance     = hInst;
        wc.lpszClassName = "HydraOffscreenGL";
        if (!RegisterClassA(&wc)) {
            std::cerr << "[hydra_preview] RegisterClassA failed: " << GetLastError() << "\n";
            return false;
        }
        wc_registered = true;
    }

    HWND hwnd = CreateWindowExA(0, "HydraOffscreenGL", "",
                                WS_OVERLAPPEDWINDOW, 0, 0, 1, 1,
                                nullptr, nullptr, hInst, nullptr);
    if (!hwnd) {
        std::cerr << "[hydra_preview] CreateWindowExA failed: " << GetLastError() << "\n";
        return false;
    }

    HDC hdc = GetDC(hwnd);
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize      = sizeof(pfd);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int pf = ChoosePixelFormat(hdc, &pfd);
    if (!pf || !SetPixelFormat(hdc, pf, &pfd)) {
        std::cerr << "[hydra_preview] SetPixelFormat failed: " << GetLastError() << "\n";
        ReleaseDC(hwnd, hdc); DestroyWindow(hwnd);
        return false;
    }

    // Legacy context needed to bootstrap wglCreateContextAttribsARB.
    HGLRC legacy = wglCreateContext(hdc);
    if (!legacy) {
        std::cerr << "[hydra_preview] wglCreateContext (legacy) failed\n";
        ReleaseDC(hwnd, hdc); DestroyWindow(hwnd);
        return false;
    }
    wglMakeCurrent(hdc, legacy);

    auto wglCreateContextAttribsARB = reinterpret_cast<PFN_wglCreateContextAttribsARB>(
        wglGetProcAddress("wglCreateContextAttribsARB"));

    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(legacy);

    if (!wglCreateContextAttribsARB) {
        std::cerr << "[hydra_preview] wglCreateContextAttribsARB not found\n";
        ReleaseDC(hwnd, hdc); DestroyWindow(hwnd);
        return false;
    }

    const int attribs[] = {
        0x2091, 4,          // WGL_CONTEXT_MAJOR_VERSION_ARB
        0x2092, 5,          // WGL_CONTEXT_MINOR_VERSION_ARB
        0x9126, 0x00000002, // WGL_CONTEXT_PROFILE_MASK_ARB = COMPATIBILITY
        0
    };
    HGLRC ctx = wglCreateContextAttribsARB(hdc, nullptr, attribs);
    if (!ctx) {
        std::cerr << "[hydra_preview] wglCreateContextAttribsARB (4.5 core) failed: "
                  << GetLastError() << "\n";
        ReleaseDC(hwnd, hdc); DestroyWindow(hwnd);
        return false;
    }

    if (!wglMakeCurrent(hdc, ctx)) {
        std::cerr << "[hydra_preview] wglMakeCurrent failed\n";
        wglDeleteContext(ctx); ReleaseDC(hwnd, hdc); DestroyWindow(hwnd);
        return false;
    }

    hp.wgl_hwnd  = hwnd;
    hp.wgl_hdc   = hdc;
    hp.wgl_hglrc = ctx;
    std::cerr << "[hydra_preview] WGL offscreen OpenGL 4.5 context ready\n";
    return true;
}

static void wgl_destroy_context(HydraPreviewState& hp)
{
    if (hp.wgl_hglrc) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(static_cast<HGLRC>(hp.wgl_hglrc));
        hp.wgl_hglrc = nullptr;
    }
    if (hp.wgl_hwnd) {
        if (hp.wgl_hdc) {
            ReleaseDC(static_cast<HWND>(hp.wgl_hwnd), static_cast<HDC>(hp.wgl_hdc));
            hp.wgl_hdc = nullptr;
        }
        DestroyWindow(static_cast<HWND>(hp.wgl_hwnd));
        hp.wgl_hwnd = nullptr;
    }
}
#endif // _WIN32

static std::string get_env_var(const char* name)
{
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        std::string out(value);
        free(value);
        return out;
    }
    return {};
#else
    const char* cur = std::getenv(name);
    return (cur && cur[0]) ? std::string(cur) : std::string();
#endif
}

static std::string append_env_path(const char* name, const std::string& extra)
{
    std::string merged = get_env_var(name);
    if (merged.empty()) return extra;
    if (merged.find(extra) == std::string::npos) {
        merged += ";";
        merged += extra;
    }
    return merged;
}

static void configure_usd_runtime_paths()
{
    namespace fs = std::filesystem;
    fs::path exe_dir = fs::current_path();
#ifdef _WIN32
    char module_path[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, module_path, MAX_PATH) > 0)
        exe_dir = fs::path(module_path).parent_path();
#endif
    const fs::path local_usd_plugin_dir = exe_dir / "usd";
    const std::string fallback_usd_plugin_dir =
        "C:/Program Files/Autodesk/MayaUSD/Maya2025/0.29.0/mayausd/USD/plugin/usd";

    std::string plugin_path = local_usd_plugin_dir.string();
    if (!fs::exists(local_usd_plugin_dir))
        plugin_path = fallback_usd_plugin_dir;

    const std::string merged_plugin_path = append_env_path("PXR_PLUGINPATH_NAME", plugin_path);
#ifdef _WIN32
    _putenv_s("PXR_PLUGINPATH_NAME", merged_plugin_path.c_str());
#else
    setenv("PXR_PLUGINPATH_NAME", merged_plugin_path.c_str(), 1);
#endif
    std::cerr << "[hydra_preview] PXR_PLUGINPATH_NAME=" << merged_plugin_path << "\n";
}

static uint32_t find_memory_type(VkPhysicalDevice physical, uint32_t type_filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

static bool immediate_submit(HydraPreviewState& hp, const std::function<void(VkCommandBuffer)>& record)
{
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = hp.vk_command_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(hp.vk_device, &ai, &cb) != VK_SUCCESS)
        return false;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) {
        vkFreeCommandBuffers(hp.vk_device, hp.vk_command_pool, 1, &cb);
        return false;
    }

    record(cb);

    if (vkEndCommandBuffer(cb) != VK_SUCCESS) {
        vkFreeCommandBuffers(hp.vk_device, hp.vk_command_pool, 1, &cb);
        return false;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    if (vkQueueSubmit(hp.vk_queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(hp.vk_device, hp.vk_command_pool, 1, &cb);
        return false;
    }

    vkQueueWaitIdle(hp.vk_queue);
    vkFreeCommandBuffers(hp.vk_device, hp.vk_command_pool, 1, &cb);
    return true;
}

static void destroy_upload_resources(HydraPreviewState& hp)
{
    if (hp.vk_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(hp.vk_device);
    }

    if (hp.imgui_desc) {
        ImGui_ImplVulkan_RemoveTexture(hp.imgui_desc);
        hp.imgui_desc = VK_NULL_HANDLE;
    }
    if (hp.vk_staging_buffer) {
        if (hp.vk_staging_mapped)
            vkUnmapMemory(hp.vk_device, hp.vk_staging_memory);
        vkDestroyBuffer(hp.vk_device, hp.vk_staging_buffer, nullptr);
        hp.vk_staging_buffer = VK_NULL_HANDLE;
    }
    if (hp.vk_staging_memory) {
        vkFreeMemory(hp.vk_device, hp.vk_staging_memory, nullptr);
        hp.vk_staging_memory = VK_NULL_HANDLE;
    }
    hp.vk_staging_mapped = nullptr;
    hp.vk_staging_size = 0;

    if (hp.vk_image_view) {
        vkDestroyImageView(hp.vk_device, hp.vk_image_view, nullptr);
        hp.vk_image_view = VK_NULL_HANDLE;
    }
    if (hp.vk_image) {
        vkDestroyImage(hp.vk_device, hp.vk_image, nullptr);
        hp.vk_image = VK_NULL_HANDLE;
    }
    if (hp.vk_image_memory) {
        vkFreeMemory(hp.vk_device, hp.vk_image_memory, nullptr);
        hp.vk_image_memory = VK_NULL_HANDLE;
    }

    hp.vk_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    hp.tex_w = 0;
    hp.tex_h = 0;
}

static bool ensure_upload_resources(HydraPreviewState& hp, int width, int height)
{
    if (width <= 0 || height <= 0) return false;
    if (hp.vk_image && hp.tex_w == width && hp.tex_h == height) return true;

    destroy_upload_resources(hp);

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = { (uint32_t)width, (uint32_t)height, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(hp.vk_device, &ici, nullptr, &hp.vk_image) != VK_SUCCESS)
        return false;

    VkMemoryRequirements imr{};
    vkGetImageMemoryRequirements(hp.vk_device, hp.vk_image, &imr);
    VkMemoryAllocateInfo iai{};
    iai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    iai.allocationSize = imr.size;
    iai.memoryTypeIndex = find_memory_type(
        hp.vk_physical, imr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (iai.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(hp.vk_device, &iai, nullptr, &hp.vk_image_memory) != VK_SUCCESS)
        return false;
    vkBindImageMemory(hp.vk_device, hp.vk_image, hp.vk_image_memory, 0);

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = hp.vk_image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    if (vkCreateImageView(hp.vk_device, &ivci, nullptr, &hp.vk_image_view) != VK_SUCCESS)
        return false;

    const VkDeviceSize staging_bytes = (VkDeviceSize)width * (VkDeviceSize)height * 4;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = staging_bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(hp.vk_device, &bci, nullptr, &hp.vk_staging_buffer) != VK_SUCCESS)
        return false;

    VkMemoryRequirements bmr{};
    vkGetBufferMemoryRequirements(hp.vk_device, hp.vk_staging_buffer, &bmr);
    VkMemoryAllocateInfo bai{};
    bai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bai.allocationSize = bmr.size;
    bai.memoryTypeIndex = find_memory_type(
        hp.vk_physical,
        bmr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (bai.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(hp.vk_device, &bai, nullptr, &hp.vk_staging_memory) != VK_SUCCESS)
        return false;
    vkBindBufferMemory(hp.vk_device, hp.vk_staging_buffer, hp.vk_staging_memory, 0);
    if (vkMapMemory(hp.vk_device, hp.vk_staging_memory, 0, staging_bytes, 0, &hp.vk_staging_mapped) != VK_SUCCESS)
        return false;

    hp.vk_staging_size = (size_t)staging_bytes;
    hp.tex_w = width;
    hp.tex_h = height;
    hp.vk_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    hp.imgui_desc = ImGui_ImplVulkan_AddTexture(
        hp.vk_sampler, hp.vk_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    return hp.imgui_desc != VK_NULL_HANDLE;
}

static float half_to_float(uint16_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;

    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

static inline uint8_t to_u8(float v)
{
    v = std::clamp(v, 0.0f, 1.0f);
    return (uint8_t)(v * 255.0f + 0.5f);
}

static bool read_color_aov_rgba8(UsdImagingGLEngine& eng, std::vector<uint8_t>& out_rgba)
{
    HdRenderBuffer* rb = eng.GetAovRenderBuffer(HdAovTokens->color);
    if (!rb) return false;

    rb->Resolve();
    void* src = rb->Map();
    if (!src) return false;

    const unsigned w = rb->GetWidth();
    const unsigned h = rb->GetHeight();
    const HdFormat fmt = rb->GetFormat();
    out_rgba.resize((size_t)w * (size_t)h * 4u);

    if (fmt == HdFormatUNorm8Vec4) {
        std::memcpy(out_rgba.data(), src, out_rgba.size());
    } else if (fmt == HdFormatFloat32Vec4) {
        const float* p = static_cast<const float*>(src);
        for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; ++i) {
            out_rgba[i * 4 + 0] = to_u8(p[i * 4 + 0]);
            out_rgba[i * 4 + 1] = to_u8(p[i * 4 + 1]);
            out_rgba[i * 4 + 2] = to_u8(p[i * 4 + 2]);
            out_rgba[i * 4 + 3] = to_u8(p[i * 4 + 3]);
        }
    } else if (fmt == HdFormatFloat16Vec4) {
        const uint16_t* p = static_cast<const uint16_t*>(src);
        for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; ++i) {
            out_rgba[i * 4 + 0] = to_u8(half_to_float(p[i * 4 + 0]));
            out_rgba[i * 4 + 1] = to_u8(half_to_float(p[i * 4 + 1]));
            out_rgba[i * 4 + 2] = to_u8(half_to_float(p[i * 4 + 2]));
            out_rgba[i * 4 + 3] = to_u8(half_to_float(p[i * 4 + 3]));
        }
    } else {
        std::fill(out_rgba.begin(), out_rgba.end(), 0);
    }

    rb->Unmap();
    return true;
}

// Read any active AOV buffer and convert to RGBA8 for display.
// Handles float1 (depth), float3/float4 (normal/color), and int (primId).
// out_w/out_h: set to actual render buffer dimensions (may differ from window size).
static bool read_active_aov_rgba8(UsdImagingGLEngine& eng, TfToken aov_token,
                                   std::vector<uint8_t>& out_rgba,
                                   int& out_w, int& out_h)
{
    HdRenderBuffer* rb = eng.GetAovRenderBuffer(aov_token);
    if (!rb) return false;

    rb->Resolve();
    void* src = rb->Map();
    if (!src) return false;

    const unsigned w = rb->GetWidth();
    const unsigned h = rb->GetHeight();
    out_w = (int)w;
    out_h = (int)h;
    const HdFormat fmt = rb->GetFormat();
    const size_t n = (size_t)w * (size_t)h;
    out_rgba.resize(n * 4u);

    // UNorm8Vec4 (same as color — Neye on HdStorm returns this)
    if (fmt == HdFormatUNorm8Vec4) {
        std::memcpy(out_rgba.data(), src, n * 4u);
        rb->Unmap();
        return true;
    }

    // UNorm8Vec3
    if (fmt == HdFormatUNorm8Vec3) {
        const uint8_t* p = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < n; ++i) {
            out_rgba[i*4+0] = p[i*3+0];
            out_rgba[i*4+1] = p[i*3+1];
            out_rgba[i*4+2] = p[i*3+2];
            out_rgba[i*4+3] = 255;
        }
        rb->Unmap();
        return true;
    }

    // Float32 single-channel (depth)
    if (fmt == HdFormatFloat32) {
        const float* p = static_cast<const float*>(src);
        // Auto-range: find min/max of valid (non-background) pixels
        float mn = 1e30f, mx = -1e30f;
        for (size_t i = 0; i < n; ++i) {
            if (p[i] < 1e10f && p[i] > -1e10f) { mn = std::min(mn, p[i]); mx = std::max(mx, p[i]); }
        }
        float range = std::max(1e-5f, mx - mn);
        for (size_t i = 0; i < n; ++i) {
            bool is_bg = (p[i] >= 1e10f || p[i] <= -1e10f);
            if (is_bg) {
                out_rgba[i*4+0] = 0; out_rgba[i*4+1] = 0;
                out_rgba[i*4+2] = 0; out_rgba[i*4+3] = 0;  // transparent bg
            } else {
                float t = std::max(0.f, std::min(1.f, (p[i] - mn) / range));
                // Log-scale: spreads near/mid range, compresses far
                float log_t = log1pf(t * 99.f) / log1pf(99.f); // steeper curve
                uint8_t v = (uint8_t)((1.f - log_t) * 255.f + 0.5f); // close=bright
                out_rgba[i*4+0] = v; out_rgba[i*4+1] = v;
                out_rgba[i*4+2] = v; out_rgba[i*4+3] = 255;
            }
        }
    }
    // Float32 vec3 (normals)
    else if (fmt == HdFormatFloat32Vec3) {
        const float* p = static_cast<const float*>(src);
        for (size_t i = 0; i < n; ++i) {
            out_rgba[i*4+0] = to_u8(p[i*3+0] * 0.5f + 0.5f);
            out_rgba[i*4+1] = to_u8(p[i*3+1] * 0.5f + 0.5f);
            out_rgba[i*4+2] = to_u8(p[i*3+2] * 0.5f + 0.5f);
            out_rgba[i*4+3] = 255;
        }
    }
    // Int32 single-channel (primId) — hash to colors
    else if (fmt == HdFormatInt32) {
        const int32_t* p = static_cast<const int32_t*>(src);
        for (size_t i = 0; i < n; ++i) {
            if (p[i] < 0) {
                out_rgba[i*4+0]=0; out_rgba[i*4+1]=0; out_rgba[i*4+2]=0; out_rgba[i*4+3]=255;
            } else {
                unsigned h = (unsigned)p[i] * 2654435761u;
                out_rgba[i*4+0] = (uint8_t)(((h>>0)&0xFF)*0.7f+76.5f);
                out_rgba[i*4+1] = (uint8_t)(((h>>8)&0xFF)*0.7f+76.5f);
                out_rgba[i*4+2] = (uint8_t)(((h>>16)&0xFF)*0.7f+76.5f);
                out_rgba[i*4+3] = 255;
            }
        }
    }
    // Unknown format — fill black
    else {
        std::cerr << "[hydra_aov] unhandled format " << (int)fmt << "\n";
        std::fill(out_rgba.begin(), out_rgba.end(), (uint8_t)0);
        for (size_t i = 0; i < n; ++i) out_rgba[i*4+3] = 255;
    }

    rb->Unmap();
    return true;
}

static bool upload_rgba8(HydraPreviewState& hp, const uint8_t* rgba, int width, int height)
{
    const size_t bytes = (size_t)width * (size_t)height * 4u;
    if (!hp.vk_staging_mapped || hp.vk_staging_size < bytes) return false;

    std::memcpy(hp.vk_staging_mapped, rgba, bytes);

    return immediate_submit(hp, [&](VkCommandBuffer cb) {
        VkImageMemoryBarrier to_dst{};
        to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_dst.oldLayout = hp.vk_image_layout;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = hp.vk_image;
        to_dst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_dst.subresourceRange.levelCount = 1;
        to_dst.subresourceRange.layerCount = 1;
        to_dst.srcAccessMask = (hp.vk_image_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            ? VK_ACCESS_SHADER_READ_BIT
            : 0;
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            cb,
            (hp.vk_image_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &to_dst);

        VkBufferImageCopy bic{};
        bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bic.imageSubresource.layerCount = 1;
        bic.imageExtent = { (uint32_t)width, (uint32_t)height, 1 };
        vkCmdCopyBufferToImage(
            cb,
            hp.vk_staging_buffer,
            hp.vk_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bic);

        VkImageMemoryBarrier to_sample{};
        to_sample.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.image = hp.vk_image;
        to_sample.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_sample.subresourceRange.levelCount = 1;
        to_sample.subresourceRange.layerCount = 1;
        to_sample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &to_sample);
    });
}

static GfMatrix4d make_view(const HydraPreviewState& hp)
{
    GfVec3d eye(hp.pos[0], hp.pos[1], hp.pos[2]);
    GfVec3d ctr(hp.pivot[0], hp.pivot[1], hp.pivot[2]);

    GfVec3d f = (ctr - eye).GetNormalized();
    GfVec3d r_raw = GfCross(f, GfVec3d(0, 1, 0));
    GfVec3d r = (r_raw.GetLength() < 1e-6) ? GfVec3d(1, 0, 0) : r_raw.GetNormalized();
    GfVec3d u = GfCross(r, f);

    return GfMatrix4d(
         r[0],          u[0],         -f[0],        0.0,
         r[1],          u[1],         -f[1],        0.0,
         r[2],          u[2],         -f[2],        0.0,
        -GfDot(r, eye), -GfDot(u, eye), GfDot(f, eye), 1.0);
}

static GfMatrix4d make_proj(const HydraPreviewState& hp, int w, int h)
{
    // Auto near/far: scale to eye–pivot distance so both tiny objects and
    // large scenes (Kitchen_set) get adequate depth range without clipping.
    float dx = hp.pos[0] - hp.pivot[0];
    float dy = hp.pos[1] - hp.pivot[1];
    float dz = hp.pos[2] - hp.pivot[2];
    double dist = sqrt((double)dx*dx + (double)dy*dy + (double)dz*dz);
    if (dist < 0.001) dist = 0.001;

    const double near_  = dist * 0.0005;               // 1/2000 of eye-pivot dist
    const double far_   = dist * 20000.0;
    const double aspect = (h > 0) ? static_cast<double>(w) / h : 1.0;

    // SetPerspective(vfov_deg, isVerticalFov, aspect, near, far) — no window ambiguity.
    GfFrustum frustum;
    frustum.SetPerspective(hp.vfov, /*isFovVertical=*/true, aspect, near_, far_);
    return frustum.ComputeProjectionMatrix();
}

static bool recreate_engine(HydraPreviewState& hp);

bool hydra_preview_init(
    HydraPreviewState& hp,
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkQueue graphics_queue,
    uint32_t graphics_queue_family,
    VkCommandPool command_pool)
{
    configure_usd_runtime_paths();

    auto fail_init = [&hp]() -> bool {
        if (hp.engine_ptr) {
            delete static_cast<UsdImagingGLEngine*>(hp.engine_ptr);
            hp.engine_ptr = nullptr;
        }
        if (hp.vk_sampler) {
            vkDestroySampler(hp.vk_device, hp.vk_sampler, nullptr);
            hp.vk_sampler = VK_NULL_HANDLE;
        }
#ifdef _WIN32
        wgl_destroy_context(hp);
#endif
        return false;
    };

    hp.vk_device = device;
    hp.vk_physical = physical_device;
    hp.vk_queue = graphics_queue;
    hp.vk_queue_family = graphics_queue_family;
    hp.vk_command_pool = command_pool;

    // Create an offscreen OpenGL 4.5 context so HdStorm's GL Hgi can initialize.
#ifdef _WIN32
    if (!wgl_create_context(hp)) {
        std::cerr << "[hydra_preview] Failed to create WGL offscreen context\n";
        return false;
    }
#endif

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 1.0f;
    if (vkCreateSampler(hp.vk_device, &sci, nullptr, &hp.vk_sampler) != VK_SUCCESS)
        return fail_init();

    // Log available plugins for diagnostics, then create the engine via the shared helper.
    {
        TfTokenVector renderer_plugins = UsdImagingGLEngine::GetRendererPlugins();
        if (renderer_plugins.empty()) {
            std::cerr << "[hydra_preview] No Hydra renderer plugins found\n";
            return fail_init();
        }
        std::cerr << "[hydra_preview] Available renderer plugins:";
        for (const TfToken& t : renderer_plugins)
            std::cerr << " " << t.GetString();
        std::cerr << "\n";
    }

    if (!recreate_engine(hp)) {
        std::cerr << "[hydra_preview] No supported renderer plugin found\n";
        return fail_init();
    }
    return true;
}

void hydra_preview_destroy(HydraPreviewState& hp)
{
    destroy_upload_resources(hp);

    if (hp.vk_sampler) {
        vkDestroySampler(hp.vk_device, hp.vk_sampler, nullptr);
        hp.vk_sampler = VK_NULL_HANDLE;
    }

    if (hp.engine_ptr) {
        delete static_cast<UsdImagingGLEngine*>(hp.engine_ptr);
        hp.engine_ptr = nullptr;
    }
    if (hp.stage_ref) {
        delete static_cast<UsdStageRefPtr*>(hp.stage_ref);
        hp.stage_ref = nullptr;
    }
#ifdef _WIN32
    wgl_destroy_context(hp);
#endif
    hp.loaded = false;
}

// Recreate the UsdImagingGLEngine in-place (same GL context) so its Hydra scene
// delegate starts fresh.  Must be called with the WGL context current.
static bool recreate_engine(HydraPreviewState& hp)
{
    if (hp.engine_ptr) {
        delete static_cast<UsdImagingGLEngine*>(hp.engine_ptr);
        hp.engine_ptr = nullptr;
    }

    hp.engine_ptr = new UsdImagingGLEngine();
    auto& eng = get_engine(hp);
    eng.SetEnablePresentation(false);

    TfTokenVector plugins = UsdImagingGLEngine::GetRendererPlugins();
    for (const TfToken& t : plugins) {
        if (eng.SetRendererPlugin(t)) {
            // Log all AOVs this renderer supports
            TfTokenVector aovs = eng.GetRendererAovs();
            std::cerr << "[hydra_preview] Renderer '" << t << "' supports "
                      << aovs.size() << " AOVs:";
            for (const TfToken& a : aovs)
                std::cerr << " " << a;
            std::cerr << "\n";

            eng.SetRendererAov(HdAovTokens->color);
            return true;
        }
    }
    // No plugin found — clean up and signal failure.
    delete static_cast<UsdImagingGLEngine*>(hp.engine_ptr);
    hp.engine_ptr = nullptr;
    return false;
}

void hydra_preview_load(HydraPreviewState& hp, const std::string& path)
{
    if (path.empty()) return;

    UsdStageRefPtr stage = UsdStage::Open(path);
    if (!stage) {
        std::cerr << "[hydra_preview] Failed to open: " << path << "\n";
        hp.loaded = false;
        return;
    }

    // Replace the stage reference.
    if (hp.stage_ref)
        delete static_cast<UsdStageRefPtr*>(hp.stage_ref);
    hp.stage_ref = new UsdStageRefPtr(stage);

    // Recreate the engine so its Hydra scene delegate is clean (no stale old-stage data).
#ifdef _WIN32
    if (hp.wgl_hdc && hp.wgl_hglrc)
        wglMakeCurrent(static_cast<HDC>(hp.wgl_hdc), static_cast<HGLRC>(hp.wgl_hglrc));
#endif
    if (!recreate_engine(hp)) {
        std::cerr << "[hydra_preview] Failed to recreate engine for: " << path << "\n";
        hp.loaded = false;
        return;
    }

    hp.loaded = true;
    std::cout << "[hydra_preview] Loaded: " << path << "\n";
}

bool hydra_preview_tick(HydraPreviewState& hp, const AnimPanelState& anim, int width, int height)
{
    if (!hp.engine_ptr || !hp.loaded || !hp.stage_ref) return true;
    if (width <= 0 || height <= 0) return true;

    if (!ensure_upload_resources(hp, width, height))
        return false;

    auto& eng = get_engine(hp);
    auto& stage = get_stage(hp);

    // Ensure the GL context is current on this thread before calling Render.
#ifdef _WIN32
    if (hp.wgl_hdc && hp.wgl_hglrc)
        wglMakeCurrent(static_cast<HDC>(hp.wgl_hdc), static_cast<HGLRC>(hp.wgl_hglrc));
#endif

    // Flush any stale GL errors that accumulated during init or non-HdStorm GL calls,
    // so HgiGLPostPendingGLErrors doesn't misattribute them to HdStorm operations.
    while (glGetError() != GL_NO_ERROR) {}

    eng.SetRenderBufferSize(GfVec2i(width, height));
    eng.SetRenderViewport(GfVec4d(0, 0, width, height));
    eng.SetCameraState(make_view(hp), make_proj(hp, width, height));

    // Apply the same axis correction as usd_loader.cpp: X→X, Y→−Z, Z→+Y for Z-up
    // stages, and bake metersPerUnit so cm-authored stages match the viewport's
    // meters-space units.
    {
        GfMatrix4d root(1.0);
        TfToken up_axis = UsdGeomGetStageUpAxis(stage);
        if (up_axis == UsdGeomTokens->z) {
            root = GfMatrix4d(
                1,  0,  0, 0,
                0,  0, -1, 0,
                0,  1,  0, 0,
                0,  0,  0, 1);
        }
        double mpu = UsdGeomGetStageMetersPerUnit(stage);
        if (mpu <= 0.0) mpu = 1.0;
        if (mpu != 1.0) {
            GfMatrix4d scale_m(1.0);
            scale_m.SetScale(GfVec3d(mpu, mpu, mpu));
            root = root * scale_m;
        }
        eng.SetRootTransform(root);
    }

    // World-space lighting that follows the camera.
    // Compute light directions from the actual camera vectors so they
    // rotate with the view.  w=0 → directional.
    {
        GfVec3d eye(hp.pos[0], hp.pos[1], hp.pos[2]);
        GfVec3d ctr(hp.pivot[0], hp.pivot[1], hp.pivot[2]);
        GfVec3d fwd = (ctr - eye);
        if (fwd.GetLength() < 1e-6) fwd = GfVec3d(0, 0, -1);
        fwd.Normalize();
        GfVec3d right_raw = GfCross(fwd, GfVec3d(0, 1, 0));
        GfVec3d right = (right_raw.GetLength() < 1e-6) ? GfVec3d(1, 0, 0)
                                                        : right_raw.GetNormalized();
        GfVec3d up = GfCross(right, fwd).GetNormalized();

        // Light position with w=0 is the direction FROM which light arrives.
        // Negate forward so light shines from behind the camera onto the object.
        GfVec3d back = -fwd;
        // Key: behind-right-above camera, shining onto the object
        GfVec3d key = (back + right * 0.5 + up * 0.7).GetNormalized();
        // Fill: behind-left-above camera
        GfVec3d fill = (back - right * 0.4 + up * 0.3).GetNormalized();

        GlfSimpleLightVector lights(2);
        // Key light
        lights[0].SetPosition(GfVec4f(key[0], key[1], key[2], 0.f));
        lights[0].SetDiffuse(GfVec4f(0.9f, 0.9f, 0.9f, 1.f));
        lights[0].SetAmbient(GfVec4f(0.3f, 0.3f, 0.3f, 1.f));
        // Fill light
        lights[1].SetPosition(GfVec4f(fill[0], fill[1], fill[2], 0.f));
        lights[1].SetDiffuse(GfVec4f(0.4f, 0.4f, 0.45f, 1.f));
        lights[1].SetAmbient(GfVec4f(0.0f, 0.0f, 0.0f, 1.f));

        GlfSimpleMaterial material;
        material.SetAmbient(GfVec4f(0.2f, 0.2f, 0.2f, 1.f));
        material.SetSpecular(GfVec4f(0.15f, 0.15f, 0.15f, 1.f));
        material.SetShininess(64.f);
        eng.SetLightingState(lights, material, GfVec4f(0.15f, 0.15f, 0.15f, 1.f));
    }

    UsdImagingGLRenderParams params;
    params.frame = UsdTimeCode(anim.current_time);
    params.complexity = 1.1f;  // >1.0 triggers subdivision refinement for smooth normals
    params.drawMode = UsdImagingGLDrawMode::DRAW_SHADED_SMOOTH;
    params.enableLighting = true;
    params.enableSceneMaterials = true;
    params.enableSceneLights = false;
    params.cullStyle = UsdImagingGLCullStyle::CULL_STYLE_BACK_UNLESS_DOUBLE_SIDED;
    params.clearColor = GfVec4f(0.18f, 0.18f, 0.18f, 1.f);

    // Select AOV before rendering
    // HdStorm supports: color, depth, Neye, primId
    TfToken active_aov = HdAovTokens->color;
    switch (hp.aov_mode) {
        default:
        case 0: active_aov = HdAovTokens->color;  break;
        case 1: active_aov = HdAovTokens->depth;  break;
        case 2: active_aov = HdAovTokens->Neye;   break;
        case 3: active_aov = HdAovTokens->primId;  break;
    }

    eng.SetRendererAov(active_aov);
    eng.Render(stage->GetPseudoRoot(), params);

    // Focus pick: cast a single ray at the requested pixel via a 1×1 pick frustum.
    hp.pick_result_dist = -1.f;
    if (hp.pick_requested) {
        hp.pick_requested = false;

        float ddx = hp.pos[0]-hp.pivot[0], ddy = hp.pos[1]-hp.pivot[1], ddz = hp.pos[2]-hp.pivot[2];
        double dist = sqrt((double)ddx*ddx + (double)ddy*ddy + (double)ddz*ddz);
        if (dist < 0.001) dist = 0.001;
        const double near_ = dist * 0.0005;
        const double far_  = dist * 20000.0;
        const double aspect = (double)width / height;

        // Build the same base frustum as make_proj so window coords are consistent.
        GfFrustum base_frust;
        base_frust.SetPerspective(hp.vfov, /*isVertical=*/true, aspect, near_, far_);
        GfRange2d win = base_frust.GetWindow();
        const double half_w = win.GetMax()[0];
        const double half_h = win.GetMax()[1];

        // Map pixel centre to frustum window coords (Y flipped: screen top = frustum bottom).
        const double u  = (hp.pick_px + 0.5) / width;
        const double v  = 1.0 - (hp.pick_py + 0.5) / height;
        const double wx = win.GetMin()[0] + u * (win.GetMax()[0] - win.GetMin()[0]);
        const double wy = win.GetMin()[1] + v * (win.GetMax()[1] - win.GetMin()[1]);
        const double pw = (win.GetMax()[0] - win.GetMin()[0]) / (2.0 * width);
        const double ph = (win.GetMax()[1] - win.GetMin()[1]) / (2.0 * height);

        GfFrustum pick_frust;
        pick_frust.SetProjectionType(GfFrustum::Perspective);
        pick_frust.SetNearFar(GfRange1d(near_, far_));
        pick_frust.SetWindow(GfRange2d(GfVec2d(wx-pw, wy-ph), GfVec2d(wx+pw, wy+ph)));

        GfVec3d hit_pt, hit_norm;
        if (eng.TestIntersection(make_view(hp), pick_frust.ComputeProjectionMatrix(),
                                 stage->GetPseudoRoot(), params, &hit_pt, &hit_norm)) {
            GfVec3d eye(hp.pos[0], hp.pos[1], hp.pos[2]);
            hp.pick_result_dist    = (float)(hit_pt - eye).GetLength();
            hp.pick_result_hit[0]  = (float)hit_pt[0];
            hp.pick_result_hit[1]  = (float)hit_pt[1];
            hp.pick_result_hit[2]  = (float)hit_pt[2];
        }
    }

    std::vector<uint8_t> rgba;
    int buf_w = width, buf_h = height;
    bool aov_ok = false;
    if (hp.aov_mode == 0) {
        aov_ok = read_color_aov_rgba8(eng, rgba);
    } else {
        aov_ok = read_active_aov_rgba8(eng, active_aov, rgba, buf_w, buf_h);
    }
    if (!aov_ok) {
        // AOV failed — fall back to color and reset mode
        eng.SetRendererAov(HdAovTokens->color);
        hp.aov_mode = 0;
        if (!read_color_aov_rgba8(eng, rgba))
            return false;
        buf_w = width; buf_h = height;
    }

    // Ensure upload resources match the actual buffer size
    if (buf_w != hp.tex_w || buf_h != hp.tex_h) {
        if (!ensure_upload_resources(hp, buf_w, buf_h))
            return false;
    }

    if (!upload_rgba8(hp, rgba.data(), buf_w, buf_h))
        return false;

    hp.vk_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}

VkDescriptorSet hydra_preview_descriptor(const HydraPreviewState& hp)
{
    return hp.imgui_desc;
}

std::vector<uint8_t> hydra_preview_read_color(HydraPreviewState& hp)
{
    std::vector<uint8_t> raw;
    if (!hp.engine_ptr) return raw;
    if (!read_color_aov_rgba8(get_engine(hp), raw)) return {};

    int w = hp.tex_w, h = hp.tex_h;
    if (w <= 0 || h <= 0 || (int)raw.size() != w * h * 4) return raw;

    // Flip Y (OpenGL bottom-up → top-down) and apply sRGB gamma
    std::vector<uint8_t> result(raw.size());
    auto to_srgb = [](uint8_t v) -> uint8_t {
        float c = v / 255.f;
        float s = (c <= 0.0031308f)
            ? c * 12.92f
            : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
        return (uint8_t)(std::min(1.f, std::max(0.f, s)) * 255.f + 0.5f);
    };
    for (int y = 0; y < h; ++y) {
        int src_row = (h - 1 - y) * w * 4;
        int dst_row = y * w * 4;
        for (int x = 0; x < w; ++x) {
            int si = src_row + x * 4;
            int di = dst_row + x * 4;
            result[di + 0] = to_srgb(raw[si + 0]);
            result[di + 1] = to_srgb(raw[si + 1]);
            result[di + 2] = to_srgb(raw[si + 2]);
            result[di + 3] = raw[si + 3];
        }
    }
    return result;
}
