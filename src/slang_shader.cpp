#include "slang_shader.h"
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cctype>

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

    // Checkerboard
    int cx = int(uv.x * 8.0);
    int cy = int(uv.y * 8.0);
    float c = ((cx + cy) % 2 == 0) ? 1.0 : 0.15;

    m.base_color = float3(c * 0.9, c * 0.5, c * 0.2);
    m.roughness  = 0.4;
    m.metallic   = 0.0;
    m.emissive   = float3(0.0, 0.0, 0.0);
    m.normal_ts  = float3(0.0, 0.0, 1.0);  // flat

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

struct Params { int width; int height; int frame; };
[[vk::push_constant]] ConstantBuffer<Params> p;

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int2 px = int2(tid.xy);
    if (px.x >= p.width || px.y >= p.height) return;

    float2 uv     = float2((float)px.x / (float)p.width,
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
    auto tmp        = fs::temp_directory_path();
    auto slang_path = (tmp / "sl_mat.slang").string();
    auto spv_path   = (tmp / "sl_mat.spv").string();

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
        " -target spirv -stage compute -entry main"
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

    FILE* fp = _popen(cmd.c_str(), "r");
    if (!fp) { err_out = "Failed to launch slangc — set SLANG_SDK_DIR in cmake"; return false; }
    char buf[512];
    while (fgets(buf, sizeof(buf), fp)) err_out += buf;
    int rc = _pclose(fp);

    if (rc != 0 || !fs::exists(spv_path)) {
        if (err_out.empty()) err_out = "slangc failed (exit " + std::to_string(rc) + ")";
        fs::remove(slang_path);
        return false;
    }

    std::ifstream spv(spv_path, std::ios::binary | std::ios::ate);
    if (!spv) { err_out = "Cannot read SPIR-V output"; return false; }
    size_t sz = (size_t)spv.tellg();
    spirv_out.resize(sz / 4);
    spv.seekg(0);
    spv.read(reinterpret_cast<char*>(spirv_out.data()), (std::streamsize)sz);

    fs::remove(slang_path);
    fs::remove(spv_path);
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

static VkCommandBuffer begin_cmd()
{
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = g_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(g_device, &ai, &cb);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

static void end_cmd(VkCommandBuffer cb)
{
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_queue);
    vkFreeCommandBuffers(g_device, g_pool, 1, &cb);
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
//  slang_shader_run
// ─────────────────────────────────────────────────────────────────────────────

SlangShaderResult slang_shader_run(const std::string& user_code,
                                   int width, int height, int frame_count)
{
    SlangShaderResult res;
    res.width = width; res.height = height;

    if (g_device == VK_NULL_HANDLE) {
        res.error_log = "Slang: Vulkan not initialized — call slang_shader_init()";
        return res;
    }

    // 1. Compile Slang → SPIR-V
    std::vector<uint32_t> spirv;
    if (!compile_slang(build_source(user_code), spirv, res.error_log)) return res;

    // 2. Shader module — guard against empty/corrupt SPIR-V before Vulkan sees it
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) {
        res.error_log = "slangc produced invalid SPIR-V (magic mismatch)"; return res;
    }
    VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = spirv.size() * 4; smi.pCode = spirv.data();
    VkShaderModule shmod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g_device, &smi, nullptr, &shmod) != VK_SUCCESS) {
        res.error_log = "vkCreateShaderModule failed"; return res;
    }

    // 3. Descriptor set layout — 4 storage images
    VkDescriptorSetLayoutBinding binds[4]{};
    for (int i = 0; i < 4; ++i) {
        binds[i].binding        = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binds[i].descriptorCount= 1;
        binds[i].stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl_ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dsl_ci.bindingCount = 4; dsl_ci.pBindings = binds;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(g_device, &dsl_ci, nullptr, &dsl);

    // 4. Pipeline layout (push constants: 3 ints)
    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * (uint32_t)sizeof(int) };
    VkPipelineLayoutCreateInfo pl_ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pl_ci.setLayoutCount = 1; pl_ci.pSetLayouts = &dsl;
    pl_ci.pushConstantRangeCount = 1; pl_ci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    vkCreatePipelineLayout(g_device, &pl_ci, nullptr, &pl);

    // 5. Compute pipeline
    VkComputePipelineCreateInfo cp_ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cp_ci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shmod, "main", nullptr };
    cp_ci.layout = pl;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipeline) != VK_SUCCESS) {
        res.error_log = "vkCreateComputePipelines failed";
        vkDestroyShaderModule(g_device, shmod, nullptr);
        vkDestroyDescriptorSetLayout(g_device, dsl, nullptr);
        vkDestroyPipelineLayout(g_device, pl, nullptr);
        return res;
    }

    // Pre-declare all variables that sl_cleanup needs, so gotos don't skip initializations
    VkDeviceSize        buf_bytes  = 0;
    VkBuffer            stage_buf  = VK_NULL_HANDLE;
    VkDeviceMemory      stage_mem  = VK_NULL_HANDLE;
    VkDescriptorPool    desc_pool  = VK_NULL_HANDLE;
    VkDescriptorSet     desc_set   = VK_NULL_HANDLE;
    void*               mapped_ptr = nullptr;

    // 6. 4 output images (R32G32B32A32_SFLOAT, STORAGE | TRANSFER_SRC)
    struct ImgBundle { VkImage img = VK_NULL_HANDLE;
                       VkDeviceMemory mem = VK_NULL_HANDLE;
                       VkImageView view = VK_NULL_HANDLE; };
    ImgBundle imgs[4]{};
    for (auto& b : imgs) {
        VkImageCreateInfo ic{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format    = VK_FORMAT_R32G32B32A32_SFLOAT;
        ic.extent    = { (uint32_t)width, (uint32_t)height, 1 };
        ic.mipLevels = ic.arrayLayers = 1;
        ic.samples   = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling    = VK_IMAGE_TILING_OPTIMAL;
        ic.usage     = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(g_device, &ic, nullptr, &b.img) != VK_SUCCESS) {
            res.error_log = "vkCreateImage failed for output texture";
            goto sl_cleanup;
        }

        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_device, b.img, &mr);
        uint32_t mem_type = find_mem(g_phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mem_type == UINT32_MAX) {
            res.error_log = "No device-local memory type found";
            goto sl_cleanup;
        }
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = mem_type;
        if (vkAllocateMemory(g_device, &mai, nullptr, &b.mem) != VK_SUCCESS) {
            res.error_log = "vkAllocateMemory failed for output texture";
            goto sl_cleanup;
        }
        vkBindImageMemory(g_device, b.img, b.mem, 0);

        VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vi.image    = b.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(g_device, &vi, nullptr, &b.view) != VK_SUCCESS) {
            res.error_log = "vkCreateImageView failed for output texture";
            goto sl_cleanup;
        }
    }

    // 7. Staging buffer for readback
    buf_bytes = (VkDeviceSize)width * height * 4 * sizeof(float);
    {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = buf_bytes * 4; // one contiguous buffer for all 4 images
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(g_device, &bci, nullptr, &stage_buf) != VK_SUCCESS) {
            res.error_log = "vkCreateBuffer failed for staging buffer";
            goto sl_cleanup;
        }
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_device, stage_buf, &mr);
        uint32_t hv_type = find_mem(g_phys, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (hv_type == UINT32_MAX) {
            res.error_log = "No host-visible memory type found";
            goto sl_cleanup;
        }
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = hv_type;
        if (vkAllocateMemory(g_device, &mai, nullptr, &stage_mem) != VK_SUCCESS) {
            res.error_log = "vkAllocateMemory failed for staging buffer";
            goto sl_cleanup;
        }
        vkBindBufferMemory(g_device, stage_buf, stage_mem, 0);
    }

    // 8. Descriptor pool + set
    {
        VkDescriptorPoolSize pool_sz{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 };
        VkDescriptorPoolCreateInfo dp_ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dp_ci.maxSets = 1; dp_ci.poolSizeCount = 1; dp_ci.pPoolSizes = &pool_sz;
        vkCreateDescriptorPool(g_device, &dp_ci, nullptr, &desc_pool);

        VkDescriptorSetAllocateInfo dsa{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsa.descriptorPool = desc_pool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
        vkAllocateDescriptorSets(g_device, &dsa, &desc_set);
    }

    // 9. Record: transition → bind → dispatch → transition → copy
    {
        VkCommandBuffer cb = begin_cmd();

        // Transition all images UNDEFINED → GENERAL
        VkImageMemoryBarrier barriers[4]{};
        for (int i = 0; i < 4; ++i) {
            barriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[i].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[i].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].image               = imgs[i].img;
            barriers[i].subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            barriers[i].dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 4, barriers);

        // Update descriptors (images are now GENERAL)
        VkDescriptorImageInfo dii[4]{};
        VkWriteDescriptorSet  wd[4]{};
        for (int i = 0; i < 4; ++i) {
            dii[i] = { VK_NULL_HANDLE, imgs[i].view, VK_IMAGE_LAYOUT_GENERAL };
            wd[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            wd[i].dstSet         = desc_set;
            wd[i].dstBinding     = (uint32_t)i;
            wd[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            wd[i].descriptorCount= 1;
            wd[i].pImageInfo     = &dii[i];
        }
        vkUpdateDescriptorSets(g_device, 4, wd, 0, nullptr);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &desc_set, 0, nullptr);

        int pc[3] = { width, height, frame_count };
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);

        vkCmdDispatch(cb, ((uint32_t)width + 15) / 16, ((uint32_t)height + 15) / 16, 1);

        // Transition GENERAL → TRANSFER_SRC + copy each image to staging buffer
        for (int i = 0; i < 4; ++i) {
            barriers[i].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 4, barriers);

        VkDeviceSize offset = 0;
        for (int i = 0; i < 4; ++i) {
            VkBufferImageCopy copy{};
            copy.bufferOffset      = offset;
            copy.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copy.imageExtent       = { (uint32_t)width, (uint32_t)height, 1 };
            vkCmdCopyImageToBuffer(cb, imgs[i].img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   stage_buf, 1, &copy);
            offset += buf_bytes;
        }

        end_cmd(cb);
    }

    // 10. Read back + convert to RGBA8
    vkMapMemory(g_device, stage_mem, 0, buf_bytes * 4, 0, &mapped_ptr);
    {
        auto* fdata = reinterpret_cast<const float*>(mapped_ptr);
        int   npix  = width * height;
        f32_to_u8(fdata,             res.base_pixels,     npix);
        f32_to_u8(fdata + npix*4,   res.mr_pixels,       npix);
        f32_to_u8(fdata + npix*4*2, res.emissive_pixels, npix);
        f32_to_u8(fdata + npix*4*3, res.normal_pixels,   npix);
    }
    vkUnmapMemory(g_device, stage_mem);

    res.success = true;

    sl_cleanup:
    // 11. Cleanup — safe to call vkDestroy on VK_NULL_HANDLE (no-op per spec)
    if (desc_pool)  vkDestroyDescriptorPool(g_device, desc_pool, nullptr);
    for (auto& b : imgs) {
        if (b.view) vkDestroyImageView(g_device, b.view, nullptr);
        if (b.img)  vkDestroyImage(g_device, b.img,  nullptr);
        if (b.mem)  vkFreeMemory(g_device, b.mem,    nullptr);
    }
    if (stage_buf)  vkDestroyBuffer(g_device, stage_buf, nullptr);
    if (stage_mem)  vkFreeMemory(g_device, stage_mem, nullptr);
    if (pipeline)   vkDestroyPipeline(g_device, pipeline, nullptr);
    if (pl)         vkDestroyPipelineLayout(g_device, pl, nullptr);
    if (dsl)        vkDestroyDescriptorSetLayout(g_device, dsl, nullptr);
    if (shmod)      vkDestroyShaderModule(g_device, shmod, nullptr);

    return res;
}
