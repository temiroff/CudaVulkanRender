#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>
#include <cstdint>

// Holds everything needed to share a Vulkan image with CUDA
struct CudaInterop {
    // Vulkan objects
    VkImage         image      = VK_NULL_HANDLE;
    VkDeviceMemory  memory     = VK_NULL_HANDLE;
    VkImageView     image_view = VK_NULL_HANDLE;
    VkSampler       sampler    = VK_NULL_HANDLE;
    VkDescriptorSet descriptor = VK_NULL_HANDLE; // registered via ImGui_ImplVulkan_AddTexture

    // CUDA objects
    cudaExternalMemory_t ext_mem   = nullptr;
    cudaMipmappedArray_t mip_array = nullptr;
    cudaArray_t          array     = nullptr;
    cudaSurfaceObject_t  surface   = 0;

    int width  = 0;
    int height = 0;
};

// Create a Vulkan image + export handle + import into CUDA as a surface.
// The ImGui descriptor set is registered via ImGui_ImplVulkan_AddTexture().
CudaInterop cuda_interop_create(
    VkDevice         device,
    VkPhysicalDevice physicalDevice,
    VkCommandPool    commandPool,
    VkQueue          graphicsQueue,
    int width, int height
);

// Transition the image layout
void cuda_interop_transition(
    VkDevice      device,
    VkCommandPool commandPool,
    VkQueue       graphicsQueue,
    CudaInterop&  ci,
    VkImageLayout oldLayout,
    VkImageLayout newLayout
);

// Free all resources (also calls ImGui_ImplVulkan_RemoveTexture)
void cuda_interop_destroy(VkDevice device, CudaInterop& ci);
