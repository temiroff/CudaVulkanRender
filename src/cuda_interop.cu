#include "cuda_interop.h"
#include <backends/imgui_impl_vulkan.h>
#include <stdexcept>
#include <cstring>

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
        if ((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("No suitable memory type");
}

static VkCommandBuffer begin_single_time(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cb;
    vkAllocateCommandBuffers(device, &ai, &cb);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

static void end_single_time(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cb);
}

// ─────────────────────────────────────────────
//  Create
// ─────────────────────────────────────────────

CudaInterop cuda_interop_create(
    VkDevice device, VkPhysicalDevice physicalDevice,
    VkCommandPool commandPool, VkQueue graphicsQueue,
    int width, int height)
{
    CudaInterop ci;
    ci.width  = width;
    ci.height = height;

    // ── 1. Create VkImage with external memory flag ──
    VkExternalMemoryImageCreateInfo ext_info{};
    ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
#ifdef _WIN32
    ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkImageCreateInfo img_info{};
    img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.pNext         = &ext_info;
    img_info.imageType     = VK_IMAGE_TYPE_2D;
    img_info.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
    img_info.extent        = { (uint32_t)width, (uint32_t)height, 1 };
    img_info.mipLevels     = 1;
    img_info.arrayLayers   = 1;
    img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &img_info, nullptr, &ci.image) != VK_SUCCESS)
        throw std::runtime_error("Failed to create interop image");

    // ── 2. Allocate exportable memory ──
    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(device, ci.image, &mem_req);

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
#ifdef _WIN32
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext           = &export_info;
    alloc_info.allocationSize  = mem_req.size;
    alloc_info.memoryTypeIndex = find_memory_type(physicalDevice, mem_req.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &alloc_info, nullptr, &ci.memory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate interop memory");

    vkBindImageMemory(device, ci.image, ci.memory, 0);

    // ── 3. Export handle ──
#ifdef _WIN32
    HANDLE win_handle;
    VkMemoryGetWin32HandleInfoKHR get_info{};
    get_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    get_info.memory     = ci.memory;
    get_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    auto vkGetMemoryWin32HandleKHR_fn = (PFN_vkGetMemoryWin32HandleKHR)
        vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!vkGetMemoryWin32HandleKHR_fn || vkGetMemoryWin32HandleKHR_fn(device, &get_info, &win_handle) != VK_SUCCESS)
        throw std::runtime_error("Failed to get Win32 memory handle");
#else
    int fd = -1;
    VkMemoryGetFdInfoKHR get_info{};
    get_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    get_info.memory     = ci.memory;
    get_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)
        vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR || vkGetMemoryFdKHR(device, &get_info, &fd) != VK_SUCCESS)
        throw std::runtime_error("Failed to get FD memory handle");
#endif

    // ── 4. Import into CUDA ──
    cudaExternalMemoryHandleDesc cuda_mem_desc{};
#ifdef _WIN32
    cuda_mem_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    cuda_mem_desc.handle.win32.handle = win_handle;
#else
    cuda_mem_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    cuda_mem_desc.handle.fd = fd;
#endif
    cuda_mem_desc.size = mem_req.size;
    cudaImportExternalMemory(&ci.ext_mem, &cuda_mem_desc);

    // ── 5. Map as CUDA mipmapped array ──
    cudaExternalMemoryMipmappedArrayDesc mip_desc{};
    mip_desc.offset     = 0;
    mip_desc.formatDesc = cudaCreateChannelDesc<float4>();
    mip_desc.extent     = make_cudaExtent(width, height, 0);
    mip_desc.flags      = cudaArraySurfaceLoadStore;
    mip_desc.numLevels  = 1;
    cudaExternalMemoryGetMappedMipmappedArray(&ci.mip_array, ci.ext_mem, &mip_desc);
    cudaGetMipmappedArrayLevel(&ci.array, ci.mip_array, 0);

    // ── 6. Create CUDA surface object ──
    cudaResourceDesc res_desc{};
    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = ci.array;
    cudaCreateSurfaceObject(&ci.surface, &res_desc);

    // ── 7. VkImageView ──
    VkImageViewCreateInfo view_info{};
    view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image                           = ci.image;
    view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format                          = VK_FORMAT_R32G32B32A32_SFLOAT;
    view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount     = 1;
    view_info.subresourceRange.layerCount     = 1;
    vkCreateImageView(device, &view_info, nullptr, &ci.image_view);

    // ── 8. VkSampler ──
    VkSamplerCreateInfo samp_info{};
    samp_info.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samp_info.magFilter    = VK_FILTER_LINEAR;
    samp_info.minFilter    = VK_FILTER_LINEAR;
    samp_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device, &samp_info, nullptr, &ci.sampler);

    // ── 9. Transition to shader-read layout ──
    cuda_interop_transition(device, commandPool, graphicsQueue,
        ci, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // ── 10. Register with ImGui (returns VkDescriptorSet used as ImTextureID) ──
    ci.descriptor = ImGui_ImplVulkan_AddTexture(
        ci.sampler, ci.image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    return ci;
}

// ─────────────────────────────────────────────
//  Layout transition
// ─────────────────────────────────────────────

void cuda_interop_transition(
    VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue,
    CudaInterop& ci, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkCommandBuffer cb = begin_single_time(device, commandPool);

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = ci.image;
    barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkPipelineStageFlags src_stage, dst_stage;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }

    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    end_single_time(device, commandPool, graphicsQueue, cb);
}

// ─────────────────────────────────────────────
//  Destroy
// ─────────────────────────────────────────────

void cuda_interop_destroy(VkDevice device, CudaInterop& ci) {
    if (ci.descriptor) ImGui_ImplVulkan_RemoveTexture(ci.descriptor);

    cudaDestroySurfaceObject(ci.surface);
    cudaFreeMipmappedArray(ci.mip_array);
    cudaDestroyExternalMemory(ci.ext_mem);

    if (ci.sampler)    vkDestroySampler(device, ci.sampler, nullptr);
    if (ci.image_view) vkDestroyImageView(device, ci.image_view, nullptr);
    if (ci.memory)     vkFreeMemory(device, ci.memory, nullptr);
    if (ci.image)      vkDestroyImage(device, ci.image, nullptr);
}
