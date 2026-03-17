#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cstdint>

struct VulkanContext {
    VkInstance               instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR             surface        = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice = VK_NULL_HANDLE;
    VkDevice                 device         = VK_NULL_HANDLE;

    uint32_t graphicsFamily = UINT32_MAX;
    uint32_t presentFamily  = UINT32_MAX;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue  = VK_NULL_HANDLE;

    VkSwapchainKHR           swapchain      = VK_NULL_HANDLE;
    std::vector<VkImage>     swapImages;
    std::vector<VkImageView> swapViews;
    VkFormat                 swapFormat     = VK_FORMAT_UNDEFINED;
    VkExtent2D               swapExtent     = {};

    VkRenderPass             renderPass     = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    VkCommandPool            commandPool    = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    VkDescriptorPool         descriptorPool = VK_NULL_HANDLE;

    // Sync — correct semaphore ownership model:
    //  imageAvailable[MAX_FRAMES]      — per frame-in-flight, used for vkAcquireNextImageKHR
    //  renderFinished[swapImageCount]  — per swapchain image, safe because image I won't be
    //                                    re-acquired until its previous present completes
    //  inFlight[MAX_FRAMES]            — GPU fence per frame-in-flight
    static constexpr int MAX_FRAMES = 2;
    VkSemaphore              imageAvailable[MAX_FRAMES] = {};
    std::vector<VkSemaphore> renderFinished;              // size == swapchain image count
    VkFence                  inFlight[MAX_FRAMES]        = {};
    std::vector<VkFence>     imagesInFlight;              // per swapchain image — which fence last submitted it
    int                      currentFrame               = 0;

    bool enableValidation = true;
};

VulkanContext vk_create(GLFWwindow* window, bool enableValidation = true);
void          vk_recreate_swapchain(VulkanContext& ctx, GLFWwindow* window);
void          vk_destroy(VulkanContext& ctx);

// Returns index of acquired swapchain image, or UINT32_MAX on resize needed
uint32_t vk_begin_frame(VulkanContext& ctx);
void     vk_end_frame(VulkanContext& ctx, uint32_t imageIndex);
