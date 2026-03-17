#include "slang_shader.h"
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cctype>
#include <system_error>

// ─────────────────────────────────────────────────────────────────────────────
//  Default starter code
// ─────────────────────────────────────────────────────────────────────────────

const char* SLANG_SHADER_DEFAULT_CODE = R"(
// MatOut — full material output.  Same meaning as the CUDA version.
// Available built-ins: float2/3/4, sin, cos, sqrt, pow, abs, min, max, floor,
//                      frac(x), lerp(a,b,t), clamp(x,lo,hi), saturate(x)

MatOut custom_material(float2 uv, float3 pos, float3 normal, int frame)
{
    MatOut m;

    float t = frame * 0.02f;

    float wave = sin(uv.x * 20.0f + t) * cos(uv.y * 20.0f - t);

    float3 color = float3(
        0.5f + 0.5f * sin(t + uv.x * 5.0f),
        0.5f + 0.5f * sin(t + uv.y * 5.0f),
        0.8f + wave * 0.2f
    );

    m.base_color = color;
    m.roughness  = 0.2f + abs(wave) * 0.4f;
    m.metallic   = 0.0f;

    m.emissive = float3(
        abs(wave) * 0.5f,
        abs(wave) * 0.2f,
        abs(wave)
    );

    m.normal_ts = float3(0.0f, 0.0f, 1.0f);

    return m;
}
)";

// ─────────────────────────────────────────────────────────────────────────────
//  Slang compute kernel wrapper — injected around user code
// ─────────────────────────────────────────────────────────────────────────────

static const char* SLANG_WRAPPER = R"(
// ── MatOut struct ─────────────────────────────────────────────────────────────
struct MatOut
{
    float3 base_color;
    float  roughness;
    float  metallic;
    float3 emissive;
    float3 normal_ts;
};

// ── User code ─────────────────────────────────────────────────────────────────
%USER_CODE%

// ── Outputs (4 storage images) ────────────────────────────────────────────────
[[vk::binding(0, 0)]] RWTexture2D<float4> base_out;
[[vk::binding(1, 0)]] RWTexture2D<float4> mr_out;
[[vk::binding(2, 0)]] RWTexture2D<float4> emis_out;
[[vk::binding(3, 0)]] RWTexture2D<float4> norm_out;

struct PushParams { int width; int height; int frame; };
[[vk::push_constant]] PushParams p;

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int2 px = int2(tid.xy);
    if (px.x >= p.width || px.y >= p.height) return;

    float2 uv = float2((float)px.x / (float)p.width,
                       (float)px.y / (float)p.height);
    float3 pos    = float3(0.0, 0.0, 0.0);
    float3 normal = float3(0.0, 1.0, 0.0);

    MatOut m = custom_material(uv, pos, normal, p.frame);

    base_out[px] = float4(clamp(m.base_color, 0.0, 1.0), 1.0);

    // glTF layout: G = roughness, B = metallic
    mr_out[px] = float4(0.0, clamp(m.roughness, 0.0, 1.0),
                             clamp(m.metallic,  0.0, 1.0), 1.0);

    emis_out[px] = float4(clamp(m.emissive, 0.0, 1.0), 1.0);

    // Encode tangent-space normal → [0,1]
    float3 n = normalize(length(m.normal_ts) > 1e-4 ? m.normal_ts : float3(0,0,1));
    norm_out[px] = float4(n * 0.5 + 0.5, 1.0);
}
)";

// ─────────────────────────────────────────────────────────────────────────────
//  Global Vulkan context (set by slang_shader_init)
// ─────────────────────────────────────────────────────────────────────────────

static VkDevice         g_device = VK_NULL_HANDLE;
static VkPhysicalDevice g_phys   = VK_NULL_HANDLE;
static VkCommandPool    g_pool   = VK_NULL_HANDLE;
static VkQueue          g_queue  = VK_NULL_HANDLE;
static std::string      g_slangc = "slangc";

void slang_shader_init(VkDevice dev, VkPhysicalDevice phys,
                       VkCommandPool pool, VkQueue queue,
                       const char* slangc_path)
{
    g_device = dev; g_phys = phys; g_pool = pool; g_queue = queue;
    if (slangc_path && slangc_path[0]) g_slangc = slangc_path;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

// Strip HLSL/Slang semantic annotations (": POSITION", ": SV_Position", etc.)
static std::string strip_hlsl_semantics(const std::string& src)
{
    std::string out;
    out.reserve(src.size());
    size_t i = 0;
    while (i < src.size()) {
        if (src[i] == ':') {
            size_t j = i + 1;
            while (j < src.size() && (src[j] == ' ' || src[j] == '\t')) ++j;
            if (j < src.size() && (isupper((unsigned char)src[j]) || src[j] == '_')) {
                size_t k = j;
                while (k < src.size() && (isalnum((unsigned char)src[k]) || src[k] == '_')) ++k;
                bool all_upper = true;
                for (size_t ci = j; ci < k; ++ci)
                    if (islower((unsigned char)src[ci])) { all_upper = false; break; }
                bool sv_prefix = (k - j >= 3 &&
                                  src[j]=='S' && src[j+1]=='V' && src[j+2]=='_');
                if (all_upper || sv_prefix) {
                    size_t m2 = k;
                    while (m2 < src.size() && (src[m2] == ' ' || src[m2] == '\t')) ++m2;
                    if (m2 < src.size() && (src[m2] == ';' || src[m2] == ',' ||
                                            src[m2] == ')' || src[m2] == '\n')) {
                        i = k;
                        continue;
                    }
                }
            }
        }
        out += src[i++];
    }
    return out;
}

static std::string build_source(const std::string& user_code)
{
    std::string code = strip_hlsl_semantics(user_code);

    // Backward compat: if user wrote a rasterization-style entry point
    // (e.g. "float4 fragmentMain(VSOutput input)"), wrap it as custom_material.
    // We look for either "fragmentMain" or "float4 pixelMain" / "float4 PSMain".
    bool raster_style = (code.find("fragmentMain") != std::string::npos ||
                         code.find("pixelMain")    != std::string::npos ||
                         code.find("PSMain")       != std::string::npos);
    if (raster_style) {
        // Rename all known entry-point variants so they don't conflict, then
        // inject a custom_material adapter that calls into the user's function.
        // We pass uv as the only meaningful varying — pos/normal are dummy.
        auto rename = [&](const std::string& from, const std::string& to) {
            size_t p = 0;
            while ((p = code.find(from, p)) != std::string::npos) {
                code.replace(p, from.size(), to);
                p += to.size();
            }
        };
        rename("fragmentMain", "_sl_frag_legacy");
        rename("pixelMain",    "_sl_pixel_legacy");
        rename("PSMain",       "_sl_ps_legacy");

        // Determine which renamed function exists and call it.
        // We try to call the first one that was renamed.
        std::string call_fn;
        if (user_code.find("fragmentMain") != std::string::npos) call_fn = "_sl_frag_legacy";
        else if (user_code.find("pixelMain") != std::string::npos) call_fn = "_sl_pixel_legacy";
        else call_fn = "_sl_ps_legacy";

        // The legacy function likely takes a struct argument. We construct a
        // minimal VSOutput-compatible struct and forward UV into it.
        code += R"(

// ── Auto-generated VSOutput stub ────────────────────────────────────────────
struct VSOutput { float2 texcoord; float4 position; };

// ── Adapter: rasterization-style → compute-style MatOut ─────────────────────
MatOut custom_material(float2 uv, float3 pos, float3 normal, int frame)
{
    VSOutput vso;
    vso.texcoord = uv;
    vso.position = float4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
    float4 r = )" + call_fn + R"((vso);
    MatOut m;
    m.base_color = r.xyz;
    m.roughness  = 0.5;
    m.metallic   = 0.0;
    m.emissive   = float3(0.0, 0.0, 0.0);
    m.normal_ts  = float3(0.0, 0.0, 1.0);
    return m;
}
)";
    }

    std::string s(SLANG_WRAPPER);
    size_t pos = s.find("%USER_CODE%");
    if (pos != std::string::npos) s.replace(pos, 11, code);
    return s;
}

static bool compile_slang(const std::string& src,
                           std::vector<uint32_t>& spirv_out,
                           std::string& err_out)
{
    namespace fs = std::filesystem;
    static std::atomic<uint64_t> s_temp_seq{0};
    auto tmp   = fs::temp_directory_path();
    uint64_t stamp = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t seq   = s_temp_seq.fetch_add(1, std::memory_order_relaxed);
    std::string stem = "sl_mat_" + std::to_string(stamp) + "_" + std::to_string(seq);
    auto slang_path = (tmp / (stem + ".slang")).string();
    auto spv_path   = (tmp / (stem + ".spv")).string();

    { std::ofstream f(slang_path); if (!f) { err_out = "Cannot write temp file"; return false; } f << src; }

    // Use forward slashes — avoids Windows path quoting issues with backslashes.
    // Wrap executable + all args in an outer cmd /C "..." so quoted paths work.
    auto to_fwd = [](std::string p) {
        for (char& c : p) if (c == '\\') c = '/';
        return p;
    };
    std::string exe_f  = to_fwd(g_slangc);
    std::string src_f  = to_fwd(slang_path);
    std::string dst_f  = to_fwd(spv_path);

    std::string cmd = "cmd /C \"\"" + exe_f + "\""
        " \"" + src_f + "\""
        " -target spirv"
        " -profile sm_6_0"
        " -stage compute"
        " -entry main"
        " -o \"" + dst_f + "\""
        " 2>&1\"";

    // Check slangc is reachable before launching (avoids cryptic Windows errors).
    // Only check fs::exists for absolute paths; bare "slangc" relies on PATH.
    if (g_slangc.find('/') != std::string::npos ||
        g_slangc.find('\\') != std::string::npos) {
        if (!fs::exists(g_slangc)) {
            err_out = "slangc not found at: " + g_slangc + "\n"
                      "Rebuild with a valid SLANG_SDK_DIR or ensure the Vulkan SDK is installed.";
            fs::remove(slang_path);
            return false;
        }
    }

    fprintf(stderr, "[slang] running: %s\n", cmd.c_str());
    FILE* fp = _popen(cmd.c_str(), "r");
    if (!fp) { err_out = "Failed to launch slangc — set SLANG_SDK_DIR in cmake"; return false; }
    char buf[512];
    while (fgets(buf, sizeof(buf), fp)) {
        err_out += buf;
        fprintf(stderr, "[slangc] %s", buf);   // always visible in console
    }
    int rc = _pclose(fp);
    fprintf(stderr, "[slang] exit code: %d  spv exists: %d\n",
            rc, (int)fs::exists(spv_path));

    if (rc != 0 || !fs::exists(spv_path)) {
        if (err_out.empty()) err_out = "slangc failed (exit " + std::to_string(rc) + ")";
        fs::remove(slang_path);
        return false;
    }

    std::ifstream spv(spv_path, std::ios::binary | std::ios::ate);
    if (!spv) { err_out = "Cannot read SPIR-V output"; fs::remove(slang_path); return false; }
    std::streamoff raw_sz = spv.tellg();
    fprintf(stderr, "[slang] SPIR-V file size: %lld bytes\n", (long long)raw_sz);
    fflush(stderr);
    if (raw_sz <= 0 || raw_sz > 64 * 1024 * 1024) {
        err_out = "SPIR-V size invalid: " + std::to_string((long long)raw_sz);
        fs::remove(slang_path); fs::remove(spv_path);
        return false;
    }
    size_t sz    = (size_t)raw_sz;
    size_t words = sz / 4;
    try { spirv_out.resize(words); }
    catch (...) { err_out = "Out of memory allocating SPIR-V buffer"; return false; }
    spv.seekg(0);
    spv.read(reinterpret_cast<char*>(spirv_out.data()), (std::streamsize)(words * 4));
    fprintf(stderr, "[slang] SPIR-V read OK — %zu words, magic=0x%08X\n",
            words, words > 0 ? spirv_out[0] : 0u);
    fflush(stderr);
    spv.close();

    std::error_code rm_err;
    fs::remove(slang_path, rm_err);
    if (rm_err) {
        fprintf(stderr, "[slang] warning: failed to remove temp source '%s': %s\n",
                slang_path.c_str(), rm_err.message().c_str());
        rm_err.clear();
    }
    fs::remove(spv_path, rm_err);
    if (rm_err) {
        fprintf(stderr, "[slang] warning: failed to remove temp spv '%s': %s\n",
                spv_path.c_str(), rm_err.message().c_str());
    }
    fprintf(stderr, "[slang] compile_slang returning success\n");
    fflush(stderr);
    return true;
}

static uint32_t find_mem(VkPhysicalDevice phys, uint32_t filter, VkMemoryPropertyFlags f)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & f) == f)
            return i;
    return UINT32_MAX;
}

static bool supports_storage_image(VkPhysicalDevice phys, VkFormat format)
{
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(phys, format, &props);
    return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

static VkCommandBuffer begin_cmd()
{
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = g_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(g_device, &ai, &cb) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) {
        vkFreeCommandBuffers(g_device, g_pool, 1, &cb);
        return VK_NULL_HANDLE;
    }
    return cb;
}

static bool end_cmd(VkCommandBuffer cb, std::string& err_out)
{
    VkResult end_res = vkEndCommandBuffer(cb);
    if (end_res != VK_SUCCESS) {
        err_out = "vkEndCommandBuffer failed (VkResult=" + std::to_string((int)end_res) + ")";
        vkFreeCommandBuffers(g_device, g_pool, 1, &cb);
        return false;
    }

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkResult submit_res = vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE);
    if (submit_res != VK_SUCCESS) {
        err_out = "vkQueueSubmit failed (VkResult=" + std::to_string((int)submit_res) + ")";
        vkFreeCommandBuffers(g_device, g_pool, 1, &cb);
        return false;
    }

    VkResult idle_res = vkQueueWaitIdle(g_queue);
    if (idle_res != VK_SUCCESS) {
        err_out = "vkQueueWaitIdle failed (VkResult=" + std::to_string((int)idle_res) + ")";
        vkFreeCommandBuffers(g_device, g_pool, 1, &cb);
        return false;
    }

    vkFreeCommandBuffers(g_device, g_pool, 1, &cb);
    return true;
}

static inline uint8_t f2u(float v)
{
    int i = (int)(v * 255.f + .5f); return (uint8_t)(i < 0 ? 0 : i > 255 ? 255 : i);
}

static void f32_to_u8(const float* src, std::vector<uint8_t>& out, int npix)
{
    out.resize((size_t)npix * 4);
    for (int i = 0; i < npix; ++i) {
        out[i*4+0] = f2u(src[i*4+0]);
        out[i*4+1] = f2u(src[i*4+1]);
        out[i*4+2] = f2u(src[i*4+2]);
        out[i*4+3] = f2u(src[i*4+3]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  slang_shader_compile — build persistent pipeline (expensive: runs slangc)
// ─────────────────────────────────────────────────────────────────────────────

SlangShaderPipeline slang_shader_compile(const std::string& user_code,
                                         int width, int height)
{
    SlangShaderPipeline p;
    p.tex_w = width; p.tex_h = height;

    if (g_device == VK_NULL_HANDLE) {
        p.error_log = "Slang: Vulkan not initialized — call slang_shader_init()";
        return p;
    }
    if (width <= 0 || height <= 0) {
        p.error_log = "Slang: invalid output size";
        return p;
    }
    if (!supports_storage_image(g_phys, VK_FORMAT_R32G32B32A32_SFLOAT)) {
        p.error_log = "VK_FORMAT_R32G32B32A32_SFLOAT does not support storage images on this GPU";
        return p;
    }

    // 1. Compile Slang → SPIR-V
    std::vector<uint32_t> spirv;
    if (!compile_slang(build_source(user_code), spirv, p.error_log)) return p;

    fprintf(stderr, "[slang] compile_slang returned spirv.size()=%zu\n", spirv.size());
    fflush(stderr);

    // Save SPIR-V for spirv-val inspection
    {
        std::ofstream dbg("C:/Temp/sl_mat_debug.spv", std::ios::binary);
        if (dbg) dbg.write(reinterpret_cast<const char*>(spirv.data()), spirv.size() * 4);
    }

    if (spirv.size() < 5 || spirv[0] != 0x07230203u) {
        p.error_log = "slangc produced invalid SPIR-V (magic mismatch, words="
                      + std::to_string(spirv.size()) + ")";
        return p;
    }
    fprintf(stderr, "[slang] SPIR-V valid: %zu words magic=0x%08X\n", spirv.size(), spirv[0]);
    fflush(stderr);

    // 2. Shader module
    fprintf(stderr, "[slang] calling vkCreateShaderModule (codeSize=%zu)...\n", spirv.size() * 4);
    fflush(stderr);
    VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = spirv.size() * 4; smi.pCode = spirv.data();
    VkResult shmod_res = vkCreateShaderModule(g_device, &smi, nullptr, &p.shmod);
    fprintf(stderr, "[slang] vkCreateShaderModule returned %d\n", (int)shmod_res);
    fflush(stderr);
    if (shmod_res != VK_SUCCESS) {
        p.error_log = "vkCreateShaderModule failed (VkResult=" + std::to_string((int)shmod_res) + ")";
        return p;
    }

    // 3. Descriptor set layout — 4 storage images
    VkDescriptorSetLayoutBinding binds[4]{};
    for (int i = 0; i < 4; ++i) {
        binds[i].binding         = (uint32_t)i;
        binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl_ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dsl_ci.bindingCount = 4; dsl_ci.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g_device, &dsl_ci, nullptr, &p.dsl) != VK_SUCCESS) {
        p.error_log = "vkCreateDescriptorSetLayout failed";
        slang_shader_destroy(p); return p;
    }

    // 4. Pipeline layout (push constants: width, height, frame — 3 ints)
    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * (uint32_t)sizeof(int) };
    VkPipelineLayoutCreateInfo pl_ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pl_ci.setLayoutCount = 1; pl_ci.pSetLayouts = &p.dsl;
    pl_ci.pushConstantRangeCount = 1; pl_ci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(g_device, &pl_ci, nullptr, &p.pl) != VK_SUCCESS) {
        p.error_log = "vkCreatePipelineLayout failed";
        slang_shader_destroy(p); return p;
    }

    // 5. Compute pipeline
    VkComputePipelineCreateInfo cp_ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cp_ci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, p.shmod, "main", nullptr };
    cp_ci.layout = p.pl;
    fprintf(stderr, "[slang] calling vkCreateComputePipelines...\n"); fflush(stderr);
    if (vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.pipe) != VK_SUCCESS) {
        p.error_log = "vkCreateComputePipelines failed";
        slang_shader_destroy(p); return p;
    }
    fprintf(stderr, "[slang] pipeline created OK\n"); fflush(stderr);

    // 6. 4 output images (R32G32B32A32_SFLOAT, STORAGE | TRANSFER_SRC)
    for (auto& b : p.imgs) {
        VkImageCreateInfo ic{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ic.imageType     = VK_IMAGE_TYPE_2D;
        ic.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
        ic.extent        = { (uint32_t)width, (uint32_t)height, 1 };
        ic.mipLevels     = ic.arrayLayers = 1;
        ic.samples       = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ic.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(g_device, &ic, nullptr, &b.img) != VK_SUCCESS) {
            p.error_log = "vkCreateImage failed"; slang_shader_destroy(p); return p;
        }
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_device, b.img, &mr);
        uint32_t mt = find_mem(g_phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mt == UINT32_MAX) { p.error_log = "No device-local memory"; slang_shader_destroy(p); return p; }
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        if (vkAllocateMemory(g_device, &mai, nullptr, &b.mem) != VK_SUCCESS) {
            p.error_log = "vkAllocateMemory failed (image)"; slang_shader_destroy(p); return p;
        }
        if (vkBindImageMemory(g_device, b.img, b.mem, 0) != VK_SUCCESS) {
            p.error_log = "vkBindImageMemory failed"; slang_shader_destroy(p); return p;
        }
        VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vi.image = b.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(g_device, &vi, nullptr, &b.view) != VK_SUCCESS) {
            p.error_log = "vkCreateImageView failed"; slang_shader_destroy(p); return p;
        }
    }

    // 7. Staging buffer (all 4 images contiguous)
    VkDeviceSize buf_bytes = (VkDeviceSize)width * height * 4 * sizeof(float);
    {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = buf_bytes * 4; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(g_device, &bci, nullptr, &p.stage_buf) != VK_SUCCESS) {
            p.error_log = "vkCreateBuffer failed"; slang_shader_destroy(p); return p;
        }
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_device, p.stage_buf, &mr);
        uint32_t ht = find_mem(g_phys, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ht == UINT32_MAX) { p.error_log = "No host-visible memory"; slang_shader_destroy(p); return p; }
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = mr.size; mai.memoryTypeIndex = ht;
        if (vkAllocateMemory(g_device, &mai, nullptr, &p.stage_mem) != VK_SUCCESS) {
            p.error_log = "vkAllocateMemory failed (staging)"; slang_shader_destroy(p); return p;
        }
        if (vkBindBufferMemory(g_device, p.stage_buf, p.stage_mem, 0) != VK_SUCCESS) {
            p.error_log = "vkBindBufferMemory failed"; slang_shader_destroy(p); return p;
        }
    }

    // 8. Descriptor pool + set
    {
        VkDescriptorPoolSize pool_sz{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 };
        VkDescriptorPoolCreateInfo dp_ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dp_ci.maxSets = 1; dp_ci.poolSizeCount = 1; dp_ci.pPoolSizes = &pool_sz;
        if (vkCreateDescriptorPool(g_device, &dp_ci, nullptr, &p.desc_pool) != VK_SUCCESS) {
            p.error_log = "vkCreateDescriptorPool failed"; slang_shader_destroy(p); return p;
        }
        VkDescriptorSetAllocateInfo dsa{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsa.descriptorPool = p.desc_pool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &p.dsl;
        if (vkAllocateDescriptorSets(g_device, &dsa, &p.desc_set) != VK_SUCCESS) {
            p.error_log = "vkAllocateDescriptorSets failed"; slang_shader_destroy(p); return p;
        }
    }

    // 9. Update descriptor set (image views don't change between dispatches)
    {
        VkDescriptorImageInfo dii[4]{};
        VkWriteDescriptorSet  wd[4]{};
        for (int i = 0; i < 4; ++i) {
            dii[i] = { VK_NULL_HANDLE, p.imgs[i].view, VK_IMAGE_LAYOUT_GENERAL };
            wd[i]  = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            wd[i].dstSet          = p.desc_set;
            wd[i].dstBinding      = (uint32_t)i;
            wd[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            wd[i].descriptorCount = 1;
            wd[i].pImageInfo      = &dii[i];
        }
        vkUpdateDescriptorSets(g_device, 4, wd, 0, nullptr);
    }

    p.valid = true;
    fprintf(stderr, "[slang] pipeline ready for dispatch (%dx%d)\n", width, height);
    fflush(stderr);
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
//  slang_shader_dispatch — re-run compiled pipeline with new frame (cheap)
// ─────────────────────────────────────────────────────────────────────────────

SlangShaderResult slang_shader_dispatch(SlangShaderPipeline& p, int frame_count)
{
    SlangShaderResult res;
    res.width = p.tex_w; res.height = p.tex_h;

    if (!p.valid) { res.error_log = "Pipeline not compiled"; return res; }

    VkCommandBuffer cb = begin_cmd();
    if (cb == VK_NULL_HANDLE) {
        res.error_log = "vkAllocateCommandBuffers / vkBeginCommandBuffer failed";
        return res;
    }

    // Transition UNDEFINED → GENERAL (safe every dispatch; content is fully overwritten)
    VkImageMemoryBarrier barriers[4]{};
    for (int i = 0; i < 4; ++i) {
        barriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[i].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[i].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].image               = p.imgs[i].img;
        barriers[i].subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barriers[i].dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 4, barriers);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &p.desc_set, 0, nullptr);

    int pc[3] = { p.tex_w, p.tex_h, frame_count };
    vkCmdPushConstants(cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);

    vkCmdDispatch(cb,
                  ((uint32_t)p.tex_w + 15) / 16,
                  ((uint32_t)p.tex_h + 15) / 16, 1);

    // Transition GENERAL → TRANSFER_SRC, copy to staging buffer
    for (int i = 0; i < 4; ++i) {
        barriers[i].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 4, barriers);

    VkDeviceSize buf_bytes = (VkDeviceSize)p.tex_w * p.tex_h * 4 * sizeof(float);
    VkDeviceSize off = 0;
    for (int i = 0; i < 4; ++i) {
        VkBufferImageCopy copy{};
        copy.bufferOffset     = off;
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.imageExtent      = { (uint32_t)p.tex_w, (uint32_t)p.tex_h, 1 };
        vkCmdCopyImageToBuffer(cb, p.imgs[i].img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               p.stage_buf, 1, &copy);
        off += buf_bytes;
    }

    if (!end_cmd(cb, res.error_log)) return res;

    void* mapped = nullptr;
    if (vkMapMemory(g_device, p.stage_mem, 0, buf_bytes * 4, 0, &mapped) != VK_SUCCESS) {
        res.error_log = "vkMapMemory failed"; return res;
    }
    {
        auto* fdata = reinterpret_cast<const float*>(mapped);
        int   npix  = p.tex_w * p.tex_h;
        f32_to_u8(fdata,            res.base_pixels,     npix);
        f32_to_u8(fdata + npix * 4, res.mr_pixels,       npix);
        f32_to_u8(fdata + npix * 8, res.emissive_pixels, npix);
        f32_to_u8(fdata + npix *12, res.normal_pixels,   npix);
    }
    vkUnmapMemory(g_device, p.stage_mem);

    res.success = true;
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
//  slang_shader_destroy — free all Vulkan objects owned by the pipeline
// ─────────────────────────────────────────────────────────────────────────────

void slang_shader_destroy(SlangShaderPipeline& p)
{
    if (g_device == VK_NULL_HANDLE) return;
    if (p.desc_pool) vkDestroyDescriptorPool(g_device, p.desc_pool, nullptr);
    for (auto& b : p.imgs) {
        if (b.view) vkDestroyImageView(g_device, b.view, nullptr);
        if (b.img)  vkDestroyImage(g_device, b.img,  nullptr);
        if (b.mem)  vkFreeMemory(g_device, b.mem,    nullptr);
    }
    if (p.stage_buf) vkDestroyBuffer(g_device, p.stage_buf, nullptr);
    if (p.stage_mem) vkFreeMemory(g_device, p.stage_mem, nullptr);
    if (p.pipe)      vkDestroyPipeline(g_device, p.pipe, nullptr);
    if (p.pl)        vkDestroyPipelineLayout(g_device, p.pl, nullptr);
    if (p.dsl)       vkDestroyDescriptorSetLayout(g_device, p.dsl, nullptr);
    if (p.shmod)     vkDestroyShaderModule(g_device, p.shmod, nullptr);
    p = SlangShaderPipeline{};
}

// ─────────────────────────────────────────────────────────────────────────────
//  slang_shader_run — convenience one-shot (backward compat)
// ─────────────────────────────────────────────────────────────────────────────

SlangShaderResult slang_shader_run(const std::string& user_code,
                                   int width, int height, int frame_count)
{
    SlangShaderPipeline p = slang_shader_compile(user_code, width, height);
    if (!p.valid) {
        SlangShaderResult res;
        res.error_log = p.error_log;
        return res;
    }
    SlangShaderResult res = slang_shader_dispatch(p, frame_count);
    slang_shader_destroy(p);
    return res;
}
