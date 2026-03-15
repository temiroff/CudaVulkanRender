#include "post_process.h"

#include <vulkan/vulkan.h>
#include <backends/imgui_impl_vulkan.h>

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <string>

#include <windows.h>

static std::filesystem::path exe_dir()
{
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

// Find a suitable memory type index.
static uint32_t find_memory_type(VkPhysicalDevice phys,
                                 uint32_t          type_filter,
                                 VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("post_process: no suitable memory type");
}

// Run a one-time command buffer (submit and wait).
static void begin_one_time(VkDevice device, VkCommandPool pool, VkCommandBuffer& cb)
{
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &ai, &cb);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
}

static void end_one_time(VkDevice device, VkCommandPool pool, VkCommandBuffer cb, VkQueue queue)
{
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cb);
}

// ─────────────────────────────────────────────────────────────────────────────
//  create_display_image — helper that creates the R16G16B16A16_SFLOAT display image,
//  its view, sampler, and transitions it to VK_IMAGE_LAYOUT_GENERAL.
// ─────────────────────────────────────────────────────────────────────────────

static void create_display_image(PostProcess&     pp,
                                 VkPhysicalDevice phys,
                                 VkCommandPool    cmd_pool,
                                 VkQueue          queue)
{
    // ── Create image ─────────────────────────────────────────────────────────
    VkImageCreateInfo img_ci{};
    img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType     = VK_IMAGE_TYPE_2D;
    img_ci.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    img_ci.extent        = { (uint32_t)pp.width, (uint32_t)pp.height, 1 };
    img_ci.mipLevels     = 1;
    img_ci.arrayLayers   = 1;
    img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(pp.device, &img_ci, nullptr, &pp.display_image);

    // ── Allocate and bind device-local memory ────────────────────────────────
    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(pp.device, pp.display_image, &mem_req);

    VkMemoryAllocateInfo alloc_i{};
    alloc_i.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_i.allocationSize  = mem_req.size;
    alloc_i.memoryTypeIndex = find_memory_type(phys, mem_req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(pp.device, &alloc_i, nullptr, &pp.display_memory);
    vkBindImageMemory(pp.device, pp.display_image, pp.display_memory, 0);

    // ── Create image view ────────────────────────────────────────────────────
    VkImageViewCreateInfo view_ci{};
    view_ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image                           = pp.display_image;
    view_ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format                          = VK_FORMAT_R16G16B16A16_SFLOAT;
    view_ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view_ci.subresourceRange.baseMipLevel   = 0;
    view_ci.subresourceRange.levelCount     = 1;
    view_ci.subresourceRange.baseArrayLayer = 0;
    view_ci.subresourceRange.layerCount     = 1;
    vkCreateImageView(pp.device, &view_ci, nullptr, &pp.display_view);

    // ── Create sampler (linear, clamp to edge) ───────────────────────────────
    VkSamplerCreateInfo samp_ci{};
    samp_ci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samp_ci.magFilter     = VK_FILTER_LINEAR;
    samp_ci.minFilter     = VK_FILTER_LINEAR;
    samp_ci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samp_ci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_ci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_ci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_ci.maxLod        = VK_LOD_CLAMP_NONE;
    vkCreateSampler(pp.device, &samp_ci, nullptr, &pp.display_sampler);

    // ── Transition to VK_IMAGE_LAYOUT_GENERAL via one-time command buffer ────
    VkCommandBuffer cb;
    begin_one_time(pp.device, cmd_pool, cb);

    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = pp.display_image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask                   = 0;
    barrier.dstAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    end_one_time(pp.device, cmd_pool, cb, queue);
}

// ─────────────────────────────────────────────────────────────────────────────
//  load_agx_lut — parses AgX_Base_sRGB.cube and uploads to a 3D Vulkan texture.
//  Returns true on success; pp.lut_* fields are populated.
// ─────────────────────────────────────────────────────────────────────────────

static bool load_agx_lut(PostProcess&     pp,
                         VkPhysicalDevice phys,
                         VkCommandPool    cmd_pool,
                         VkQueue          queue,
                         const std::string& cube_path)
{
    std::ifstream f(cube_path);
    if (!f.is_open()) {
        std::cerr << "[PostProcess] AgX LUT not found: " << cube_path << "\n";
        return false;
    }

    int N = 0;
    std::vector<float> data;  // packed R,G,B,A (A=1) triples in .cube order
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.substr(0, 11) == "LUT_3D_SIZE") {
            std::istringstream ss(line);
            std::string tok; ss >> tok >> N;
            data.reserve((size_t)N * N * N * 4);
            continue;
        }
        // Skip non-data lines (TITLE, DOMAIN_*)
        if (!std::isdigit((unsigned char)line[0]) &&
            line[0] != '-' && line[0] != '+' && line[0] != '.') continue;
        if (N == 0) continue;
        std::istringstream ss(line);
        float r, g, b;
        if (ss >> r >> g >> b) {
            data.push_back(r);
            data.push_back(g);
            data.push_back(b);
            data.push_back(1.0f);
        }
    }

    if (N == 0 || (int)data.size() != N * N * N * 4) {
        std::cerr << "[PostProcess] AgX LUT parse error: expected " << N*N*N
                  << " entries, got " << data.size()/4 << "\n";
        return false;
    }

    // ── Create Vulkan 3D image ────────────────────────────────────────────────
    VkImageCreateInfo img_ci{};
    img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType     = VK_IMAGE_TYPE_3D;
    img_ci.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
    img_ci.extent        = { (uint32_t)N, (uint32_t)N, (uint32_t)N };
    img_ci.mipLevels     = 1;
    img_ci.arrayLayers   = 1;
    img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(pp.device, &img_ci, nullptr, &pp.lut_image) != VK_SUCCESS) {
        std::cerr << "[PostProcess] Failed to create AgX LUT 3D image\n";
        return false;
    }

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(pp.device, pp.lut_image, &mem_req);
    VkMemoryAllocateInfo alloc_i{};
    alloc_i.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_i.allocationSize  = mem_req.size;
    alloc_i.memoryTypeIndex = find_memory_type(phys, mem_req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(pp.device, &alloc_i, nullptr, &pp.lut_memory);
    vkBindImageMemory(pp.device, pp.lut_image, pp.lut_memory, 0);

    // ── Upload via staging buffer ─────────────────────────────────────────────
    VkDeviceSize buf_size = data.size() * sizeof(float);

    VkBuffer       stg_buf; VkDeviceMemory stg_mem;
    VkBufferCreateInfo bc{};
    bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bc.size  = buf_size;
    bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(pp.device, &bc, nullptr, &stg_buf);
    VkMemoryRequirements smr;
    vkGetBufferMemoryRequirements(pp.device, stg_buf, &smr);
    VkMemoryAllocateInfo sai{};
    sai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    sai.allocationSize = smr.size;
    sai.memoryTypeIndex = find_memory_type(phys, smr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(pp.device, &sai, nullptr, &stg_mem);
    vkBindBufferMemory(pp.device, stg_buf, stg_mem, 0);

    void* mapped;
    vkMapMemory(pp.device, stg_mem, 0, buf_size, 0, &mapped);
    memcpy(mapped, data.data(), buf_size);
    vkUnmapMemory(pp.device, stg_mem);

    // ── One-time command: transition + copy + transition ─────────────────────
    VkCommandBuffer cb;
    begin_one_time(pp.device, cmd_pool, cb);

    // Transition: UNDEFINED → TRANSFER_DST
    VkImageMemoryBarrier bar1{};
    bar1.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar1.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
    bar1.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar1.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    bar1.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    bar1.image                           = pp.lut_image;
    bar1.subresourceRange                = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    bar1.srcAccessMask                   = 0;
    bar1.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,nullptr,0,nullptr,1,&bar1);

    // Copy buffer → 3D image
    VkBufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageOffset       = {0, 0, 0};
    region.imageExtent       = { (uint32_t)N, (uint32_t)N, (uint32_t)N };
    vkCmdCopyBufferToImage(cb, stg_buf, pp.lut_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition: TRANSFER_DST → SHADER_READ_ONLY
    VkImageMemoryBarrier bar2 = bar1;
    bar2.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar2.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bar2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,nullptr,0,nullptr,1,&bar2);

    end_one_time(pp.device, cmd_pool, cb, queue);

    // Cleanup staging
    vkDestroyBuffer(pp.device, stg_buf, nullptr);
    vkFreeMemory(pp.device, stg_mem, nullptr);

    // ── Create image view (3D) ────────────────────────────────────────────────
    VkImageViewCreateInfo view_ci{};
    view_ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image                           = pp.lut_image;
    view_ci.viewType                        = VK_IMAGE_VIEW_TYPE_3D;
    view_ci.format                          = VK_FORMAT_R32G32B32A32_SFLOAT;
    view_ci.subresourceRange                = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(pp.device, &view_ci, nullptr, &pp.lut_view);

    pp.lut_size = N;
    std::cout << "[PostProcess] AgX LUT loaded: " << N << "x" << N << "x" << N << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  update_compute_desc — writes descriptor bindings into compute_desc.
// ─────────────────────────────────────────────────────────────────────────────

static void update_compute_desc(PostProcess& pp,
                                VkImageView  hdr_view,
                                VkSampler    hdr_sampler)
{
    // Binding 0: sampled image (HDR input — image only, no sampler)
    VkDescriptorImageInfo img_info{};
    img_info.sampler     = VK_NULL_HANDLE;
    img_info.imageView   = hdr_view;
    img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write0{};
    write0.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write0.dstSet          = pp.compute_desc;
    write0.dstBinding      = 0;
    write0.dstArrayElement = 0;
    write0.descriptorCount = 1;
    write0.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write0.pImageInfo      = &img_info;

    // Binding 1: sampler (HDR input — sampler only, no image view)
    VkDescriptorImageInfo samp_info{};
    samp_info.sampler     = hdr_sampler;
    samp_info.imageView   = VK_NULL_HANDLE;
    samp_info.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkWriteDescriptorSet write1{};
    write1.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write1.dstSet          = pp.compute_desc;
    write1.dstBinding      = 1;
    write1.dstArrayElement = 0;
    write1.descriptorCount = 1;
    write1.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
    write1.pImageInfo      = &samp_info;

    // Binding 2: storage image (LDR output display image, always GENERAL)
    VkDescriptorImageInfo ldr_info{};
    ldr_info.sampler     = VK_NULL_HANDLE;
    ldr_info.imageView   = pp.display_view;
    ldr_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write2{};
    write2.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write2.dstSet          = pp.compute_desc;
    write2.dstBinding      = 2;
    write2.dstArrayElement = 0;
    write2.descriptorCount = 1;
    write2.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write2.pImageInfo      = &ldr_info;

    // Binding 3: AgX 3D LUT (sampled image only — shader reuses binding-1 sampler)
    VkDescriptorImageInfo lut_info{};
    lut_info.sampler     = VK_NULL_HANDLE;
    lut_info.imageView   = pp.lut_view;
    lut_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write3{};
    write3.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write3.dstSet          = pp.compute_desc;
    write3.dstBinding      = 3;
    write3.dstArrayElement = 0;
    write3.descriptorCount = 1;
    write3.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write3.pImageInfo      = &lut_info;

    uint32_t n_writes = (pp.lut_view != VK_NULL_HANDLE) ? 4 : 3;
    VkWriteDescriptorSet writes[] = { write0, write1, write2, write3 };
    vkUpdateDescriptorSets(pp.device, n_writes, writes, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  post_process_create
// ─────────────────────────────────────────────────────────────────────────────

PostProcess post_process_create(
    VkDevice         device,
    VkPhysicalDevice phys,
    VkDescriptorPool imgui_pool,
    VkImageView      hdr_view,
    VkSampler        hdr_sampler,
    VkCommandPool    cmd_pool,
    VkQueue          queue,
    int width, int height)
{
    PostProcess pp;
    pp.device = device;
    pp.width  = width;
    pp.height = height;

    // ── Load SPIR-V from post.spv (next to the executable) ──────────────────
    auto spv_path = exe_dir() / "post.spv";
    std::ifstream spv_file(spv_path, std::ios::binary | std::ios::ate);
    if (!spv_file.is_open()) {
        std::cerr << "[PostProcess] ERROR: could not open " << spv_path
                  << " — compile post.slang with slangc first.\n";
        return {};
    }
    auto spv_size = (size_t)spv_file.tellg();
    spv_file.seekg(0);
    std::vector<char> spv_data(spv_size);
    spv_file.read(spv_data.data(), (std::streamsize)spv_size);
    spv_file.close();

    // ── Create shader module ─────────────────────────────────────────────────
    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv_size;
    sm_ci.pCode    = reinterpret_cast<const uint32_t*>(spv_data.data());
    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &sm_ci, nullptr, &shader_module) != VK_SUCCESS) {
        std::cerr << "[PostProcess] ERROR: failed to create shader module from post.spv\n";
        return {};
    }

    // ── Create descriptor set layout ────────────────────────────────────────
    VkDescriptorSetLayoutBinding bindings[4] = {};

    // Binding 0: sampled image (HDR input, image-only)
    bindings[0].binding            = 0;
    bindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount    = 1;
    bindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = nullptr;

    // Binding 1: sampler (HDR input, sampler-only)
    bindings[1].binding            = 1;
    bindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount    = 1;
    bindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].pImmutableSamplers = nullptr;

    // Binding 2: storage image (LDR output)
    bindings[2].binding            = 2;
    bindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount    = 1;
    bindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].pImmutableSamplers = nullptr;

    // Binding 3: AgX 3D LUT (sampled image only — reuses binding-1 sampler in shader)
    bindings[3].binding            = 3;
    bindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[3].descriptorCount    = 1;
    bindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[3].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 4;
    dsl_ci.pBindings    = bindings;
    vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &pp.dsl);

    // ── Create push constant range ───────────────────────────────────────────
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset     = 0;
    pc_range.size       = sizeof(PostPushConstants);

    // ── Create pipeline layout ───────────────────────────────────────────────
    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &pp.dsl;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pc_range;
    vkCreatePipelineLayout(device, &pl_ci, nullptr, &pp.layout);

    // ── Create compute pipeline ──────────────────────────────────────────────
    VkComputePipelineCreateInfo cp_ci{};
    cp_ci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_ci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_ci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module = shader_module;
    cp_ci.stage.pName  = "main";
    cp_ci.layout       = pp.layout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pp.pipeline);

    // Shader module no longer needed after pipeline creation
    vkDestroyShaderModule(device, shader_module, nullptr);

    // ── Create descriptor pool ───────────────────────────────────────────────
    VkDescriptorPoolSize pool_sizes[3] = {};
    pool_sizes[0].type            = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[0].descriptorCount = 4;  // binding 0 (HDR) + binding 3 (LUT)
    pool_sizes[1].type            = VK_DESCRIPTOR_TYPE_SAMPLER;
    pool_sizes[1].descriptorCount = 2;
    pool_sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[2].descriptorCount = 2;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dp_ci.maxSets       = 4;
    dp_ci.poolSizeCount = 3;
    dp_ci.pPoolSizes    = pool_sizes;
    vkCreateDescriptorPool(device, &dp_ci, nullptr, &pp.pool);

    // ── Allocate compute descriptor set ─────────────────────────────────────
    VkDescriptorSetAllocateInfo ds_alloc{};
    ds_alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc.descriptorPool     = pp.pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts        = &pp.dsl;
    vkAllocateDescriptorSets(device, &ds_alloc, &pp.compute_desc);

    // ── Create display image ─────────────────────────────────────────────────
    create_display_image(pp, phys, cmd_pool, queue);

    // ── Load AgX 3D LUT ──────────────────────────────────────────────────────
    // Look next to exe first, then fall back to Blender installation.
    {
        auto lut_next_to_exe = exe_dir() / "luts" / "AgX_Base_sRGB.cube";
        const std::string blender_lut =
            "C:/Program Files/Blender Foundation/Blender 5.0/5.0/datafiles/colormanagement/luts/AgX_Base_sRGB.cube";
        bool loaded = load_agx_lut(pp, phys, cmd_pool, queue, lut_next_to_exe.string());
        if (!loaded)
            loaded = load_agx_lut(pp, phys, cmd_pool, queue, blender_lut);
        if (!loaded)
            std::cerr << "[PostProcess] WARNING: AgX LUT not loaded — AgX mode will use analytical fallback\n";
    }

    // ── Write descriptor set ─────────────────────────────────────────────────
    update_compute_desc(pp, hdr_view, hdr_sampler);

    // ── Register display image with ImGui (GENERAL layout) ───────────────────
    pp.imgui_desc = ImGui_ImplVulkan_AddTexture(
        pp.display_sampler,
        pp.display_view,
        VK_IMAGE_LAYOUT_GENERAL);

    return pp;
}

// ─────────────────────────────────────────────────────────────────────────────
//  post_process_resize
// ─────────────────────────────────────────────────────────────────────────────

void post_process_resize(
    PostProcess&     pp,
    VkPhysicalDevice phys,
    VkDescriptorPool imgui_pool,
    VkCommandPool    cmd_pool,
    VkQueue          queue,
    VkImageView      hdr_view,
    VkSampler        hdr_sampler,
    int new_w, int new_h)
{
    if (pp.device == VK_NULL_HANDLE) return;

    // ── Remove old ImGui registration ────────────────────────────────────────
    if (pp.imgui_desc != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(pp.imgui_desc);
        pp.imgui_desc = VK_NULL_HANDLE;
    }

    // ── Destroy old display image resources ──────────────────────────────────
    if (pp.display_view   != VK_NULL_HANDLE) {
        vkDestroyImageView(pp.device, pp.display_view, nullptr);
        pp.display_view = VK_NULL_HANDLE;
    }
    if (pp.display_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(pp.device, pp.display_sampler, nullptr);
        pp.display_sampler = VK_NULL_HANDLE;
    }
    if (pp.display_image  != VK_NULL_HANDLE) {
        vkDestroyImage(pp.device, pp.display_image, nullptr);
        pp.display_image = VK_NULL_HANDLE;
    }
    if (pp.display_memory != VK_NULL_HANDLE) {
        vkFreeMemory(pp.device, pp.display_memory, nullptr);
        pp.display_memory = VK_NULL_HANDLE;
    }

    // ── Update dimensions ────────────────────────────────────────────────────
    pp.width  = new_w;
    pp.height = new_h;

    // ── Recreate display image ───────────────────────────────────────────────
    create_display_image(pp, phys, cmd_pool, queue);

    // ── Update compute descriptor set ────────────────────────────────────────
    update_compute_desc(pp, hdr_view, hdr_sampler);

    // ── Re-register with ImGui (GENERAL layout) ──────────────────────────────
    pp.imgui_desc = ImGui_ImplVulkan_AddTexture(
        pp.display_sampler,
        pp.display_view,
        VK_IMAGE_LAYOUT_GENERAL);
}

// ─────────────────────────────────────────────────────────────────────────────
//  post_process_dispatch
// ─────────────────────────────────────────────────────────────────────────────

void post_process_dispatch(
    PostProcess&            pp,
    VkCommandBuffer         cb,
    const PostPushConstants& params_in)
{
    // ── Barrier: wait for previous frame's fragment read, then allow compute write ──
    // Display image stays in GENERAL throughout its lifetime.
    // This barrier covers both the memory dependency (compute write visible to
    // subsequent fragment reads) and the ordering dependency (no concurrent access).
    VkImageMemoryBarrier pre_barrier{};
    pre_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pre_barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    pre_barrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    pre_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.image                           = pp.display_image;
    pre_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    pre_barrier.subresourceRange.baseMipLevel   = 0;
    pre_barrier.subresourceRange.levelCount     = 1;
    pre_barrier.subresourceRange.baseArrayLayer = 0;
    pre_barrier.subresourceRange.layerCount     = 1;
    pre_barrier.srcAccessMask                   = VK_ACCESS_SHADER_READ_BIT;
    pre_barrier.dstAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &pre_barrier);

    // ── Bind compute pipeline and resources ───────────────────────────────────
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
        pp.layout, 0, 1, &pp.compute_desc, 0, nullptr);

    // ── Push constants (inject width/height) ─────────────────────────────────
    PostPushConstants pc = params_in;
    pc.width  = pp.width;
    pc.height = pp.height;
    vkCmdPushConstants(cb, pp.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PostPushConstants), &pc);

    // ── Dispatch ─────────────────────────────────────────────────────────────
    uint32_t gx = ((uint32_t)pp.width  + 15u) / 16u;
    uint32_t gy = ((uint32_t)pp.height + 15u) / 16u;
    vkCmdDispatch(cb, gx, gy, 1);

    // ── Barrier: compute write done → fragment shader (ImGui) can read ───────
    VkImageMemoryBarrier post_barrier{};
    post_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    post_barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    post_barrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    post_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    post_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    post_barrier.image                           = pp.display_image;
    post_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    post_barrier.subresourceRange.baseMipLevel   = 0;
    post_barrier.subresourceRange.levelCount     = 1;
    post_barrier.subresourceRange.baseArrayLayer = 0;
    post_barrier.subresourceRange.layerCount     = 1;
    post_barrier.srcAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
    post_barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &post_barrier);
}

// ─────────────────────────────────────────────────────────────────────────────
//  post_process_destroy
// ─────────────────────────────────────────────────────────────────────────────

void post_process_destroy(PostProcess& pp)
{
    if (pp.device == VK_NULL_HANDLE) return;

    vkDeviceWaitIdle(pp.device);

    // Remove ImGui registration
    if (pp.imgui_desc != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(pp.imgui_desc);
        pp.imgui_desc = VK_NULL_HANDLE;
    }

    // Destroy display image resources
    if (pp.display_view    != VK_NULL_HANDLE) vkDestroyImageView(pp.device, pp.display_view,    nullptr);
    if (pp.display_sampler != VK_NULL_HANDLE) vkDestroySampler  (pp.device, pp.display_sampler, nullptr);
    if (pp.display_image   != VK_NULL_HANDLE) vkDestroyImage    (pp.device, pp.display_image,   nullptr);
    if (pp.display_memory  != VK_NULL_HANDLE) vkFreeMemory      (pp.device, pp.display_memory,  nullptr);

    // Destroy AgX LUT resources
    if (pp.lut_view   != VK_NULL_HANDLE) vkDestroyImageView(pp.device, pp.lut_view,  nullptr);
    if (pp.lut_image  != VK_NULL_HANDLE) vkDestroyImage    (pp.device, pp.lut_image, nullptr);
    if (pp.lut_memory != VK_NULL_HANDLE) vkFreeMemory      (pp.device, pp.lut_memory, nullptr);

    // Destroy Vulkan pipeline objects
    if (pp.pipeline != VK_NULL_HANDLE) vkDestroyPipeline       (pp.device, pp.pipeline, nullptr);
    if (pp.layout   != VK_NULL_HANDLE) vkDestroyPipelineLayout  (pp.device, pp.layout,   nullptr);
    if (pp.dsl      != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(pp.device, pp.dsl,   nullptr);
    if (pp.pool     != VK_NULL_HANDLE) vkDestroyDescriptorPool  (pp.device, pp.pool,     nullptr);

    pp = {};
}
