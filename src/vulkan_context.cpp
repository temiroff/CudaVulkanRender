#include "vulkan_context.h"
#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

// ─────────────────────────────────────────────
//  Validation layer callback
// ─────────────────────────────────────────────

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[VK] " << data->pMessage << "\n";
    return VK_FALSE;
}

static VkDebugUtilsMessengerEXT create_debug_messenger(VkInstance instance) {
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debug_callback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (!func) throw std::runtime_error("vkCreateDebugUtilsMessengerEXT not found");

    VkDebugUtilsMessengerEXT messenger;
    if (func(instance, &ci, nullptr, &messenger) != VK_SUCCESS)
        throw std::runtime_error("Failed to create debug messenger");
    return messenger;
}

// ─────────────────────────────────────────────
//  Instance
// ─────────────────────────────────────────────

static VkInstance create_instance(bool enableValidation) {
    uint32_t count;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> exts(glfwExts, glfwExts + count);

    if (enableValidation) {
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // External memory extensions for CUDA interop
    exts.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

    VkApplicationInfo app_info{};
    app_info.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "CudaPathTracer";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    const char* validationLayer = "VK_LAYER_KHRONOS_validation";

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app_info;
    ci.enabledExtensionCount   = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    if (enableValidation) {
        ci.enabledLayerCount   = 1;
        ci.ppEnabledLayerNames = &validationLayer;
    }

    VkInstance instance;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create VkInstance");
    return instance;
}

// ─────────────────────────────────────────────
//  Physical device selection
// ─────────────────────────────────────────────

static VkPhysicalDevice pick_physical_device(VkInstance instance, VkSurfaceKHR surface,
                                              uint32_t& graphicsFamily, uint32_t& presentFamily)
{
    uint32_t count;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    for (auto& dev : devices) {
        // Prefer discrete GPU
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        uint32_t qcount;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, queues.data());

        int gfx = -1, pres = -1;
        for (uint32_t i = 0; i < qcount; ++i) {
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) gfx = i;
            VkBool32 support;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &support);
            if (support) pres = i;
        }

        if (gfx >= 0 && pres >= 0 && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            graphicsFamily = gfx;
            presentFamily  = pres;
            std::cout << "GPU: " << props.deviceName << "\n";
            return dev;
        }
    }

    // Fallback: first device with required queues
    for (auto& dev : devices) {
        uint32_t qcount;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, queues.data());
        int gfx = -1, pres = -1;
        for (uint32_t i = 0; i < qcount; ++i) {
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) gfx = i;
            VkBool32 support;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &support);
            if (support) pres = i;
        }
        if (gfx >= 0 && pres >= 0) {
            graphicsFamily = gfx;
            presentFamily  = pres;
            return dev;
        }
    }
    throw std::runtime_error("No suitable GPU found");
}

// ─────────────────────────────────────────────
//  Logical device
// ─────────────────────────────────────────────

static VkDevice create_device(VkPhysicalDevice phys, uint32_t gfx, uint32_t pres) {
    std::set<uint32_t> families = { gfx, pres };
    float priority = 1.f;

    std::vector<VkDeviceQueueCreateInfo> queue_cis;
    for (uint32_t f : families) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = f;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &priority;
        queue_cis.push_back(qi);
    }

    std::vector<const char*> exts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
    };

    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy                 = VK_TRUE;
    features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    features.shaderStorageImageReadWithoutFormat  = VK_TRUE;

    VkDeviceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount    = (uint32_t)queue_cis.size();
    ci.pQueueCreateInfos       = queue_cis.data();
    ci.enabledExtensionCount   = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    ci.pEnabledFeatures        = &features;

    VkDevice device;
    if (vkCreateDevice(phys, &ci, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create logical device");
    return device;
}

// ─────────────────────────────────────────────
//  Swapchain
// ─────────────────────────────────────────────

static void create_swapchain(VulkanContext& ctx, GLFWwindow* window) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physicalDevice, ctx.surface, &caps);

    // Prefer BGRA8 SRGB
    uint32_t fmt_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice, ctx.surface, &fmt_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice, ctx.surface, &fmt_count, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            chosen = f;
    ctx.swapFormat = chosen.format;

    // Prefer mailbox, fallback FIFO
    uint32_t pm_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice, ctx.surface, &pm_count, nullptr);
    std::vector<VkPresentModeKHR> present_modes(pm_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice, ctx.surface, &pm_count, present_modes.data());

    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto& pm : present_modes)
        if (pm == VK_PRESENT_MODE_MAILBOX_KHR) present_mode = pm;

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    ctx.swapExtent = { (uint32_t)w, (uint32_t)h };
    ctx.swapExtent.width  = std::clamp(ctx.swapExtent.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    ctx.swapExtent.height = std::clamp(ctx.swapExtent.height, caps.minImageExtent.height, caps.maxImageExtent.height);

    uint32_t img_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) img_count = std::min(img_count, caps.maxImageCount);

    VkSwapchainCreateInfoKHR sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = ctx.surface;
    sci.minImageCount    = img_count;
    sci.imageFormat      = chosen.format;
    sci.imageColorSpace  = chosen.colorSpace;
    sci.imageExtent      = ctx.swapExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = present_mode;
    sci.clipped          = VK_TRUE;

    uint32_t family_indices[] = { ctx.graphicsFamily, ctx.presentFamily };
    if (ctx.graphicsFamily != ctx.presentFamily) {
        sci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        sci.queueFamilyIndexCount = 2;
        sci.pQueueFamilyIndices   = family_indices;
    } else {
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    if (vkCreateSwapchainKHR(ctx.device, &sci, nullptr, &ctx.swapchain) != VK_SUCCESS)
        throw std::runtime_error("Failed to create swapchain");

    vkGetSwapchainImagesKHR(ctx.device, ctx.swapchain, &img_count, nullptr);
    ctx.swapImages.resize(img_count);
    vkGetSwapchainImagesKHR(ctx.device, ctx.swapchain, &img_count, ctx.swapImages.data());

    ctx.swapViews.resize(img_count);
    for (uint32_t i = 0; i < img_count; ++i) {
        VkImageViewCreateInfo iv{};
        iv.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image                           = ctx.swapImages[i];
        iv.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        iv.format                          = ctx.swapFormat;
        iv.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.levelCount     = 1;
        iv.subresourceRange.layerCount     = 1;
        vkCreateImageView(ctx.device, &iv, nullptr, &ctx.swapViews[i]);
    }
}

static void create_render_pass(VulkanContext& ctx) {
    VkAttachmentDescription color_att{};
    color_att.format         = ctx.swapFormat;
    color_att.samples        = VK_SAMPLE_COUNT_1_BIT;
    color_att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color_att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color_att.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &color_ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 1;
    rp_info.pAttachments    = &color_att;
    rp_info.subpassCount    = 1;
    rp_info.pSubpasses      = &subpass;
    rp_info.dependencyCount = 1;
    rp_info.pDependencies   = &dep;
    vkCreateRenderPass(ctx.device, &rp_info, nullptr, &ctx.renderPass);
}

static void create_framebuffers(VulkanContext& ctx) {
    ctx.framebuffers.resize(ctx.swapViews.size());
    for (size_t i = 0; i < ctx.swapViews.size(); ++i) {
        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = ctx.renderPass;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments    = &ctx.swapViews[i];
        fb_info.width           = ctx.swapExtent.width;
        fb_info.height          = ctx.swapExtent.height;
        fb_info.layers          = 1;
        vkCreateFramebuffer(ctx.device, &fb_info, nullptr, &ctx.framebuffers[i]);
    }
}

// ─────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────

VulkanContext vk_create(GLFWwindow* window, bool enableValidation) {
    VulkanContext ctx;
    ctx.enableValidation = enableValidation;
    ctx.instance = create_instance(enableValidation);

    if (enableValidation)
        ctx.debugMessenger = create_debug_messenger(ctx.instance);

    if (glfwCreateWindowSurface(ctx.instance, window, nullptr, &ctx.surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");

    ctx.physicalDevice = pick_physical_device(ctx.instance, ctx.surface,
                                              ctx.graphicsFamily, ctx.presentFamily);
    ctx.device = create_device(ctx.physicalDevice, ctx.graphicsFamily, ctx.presentFamily);

    vkGetDeviceQueue(ctx.device, ctx.graphicsFamily, 0, &ctx.graphicsQueue);
    vkGetDeviceQueue(ctx.device, ctx.presentFamily,  0, &ctx.presentQueue);

    create_swapchain(ctx, window);
    create_render_pass(ctx);
    create_framebuffers(ctx);

    // Command pool
    VkCommandPoolCreateInfo cp_info{};
    cp_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_info.queueFamilyIndex = ctx.graphicsFamily;
    cp_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(ctx.device, &cp_info, nullptr, &ctx.commandPool);

    // Command buffers
    ctx.commandBuffers.resize(ctx.swapImages.size());
    VkCommandBufferAllocateInfo cb_alloc{};
    cb_alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_alloc.commandPool        = ctx.commandPool;
    cb_alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_alloc.commandBufferCount = (uint32_t)ctx.commandBuffers.size();
    vkAllocateCommandBuffers(ctx.device, &cb_alloc, ctx.commandBuffers.data());

    // Descriptor pool for ImGui + interop texture
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
    };
    VkDescriptorPoolCreateInfo dp_info{};
    dp_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dp_info.maxSets       = 16;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes    = pool_sizes;
    vkCreateDescriptorPool(ctx.device, &dp_info, nullptr, &ctx.descriptorPool);

    // Sync objects
    VkSemaphoreCreateInfo sem{};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fen{};
    fen.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fen.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // One acquire semaphore per frame-in-flight
    for (int i = 0; i < VulkanContext::MAX_FRAMES; ++i)
        vkCreateSemaphore(ctx.device, &sem, nullptr, &ctx.imageAvailable[i]);

    // One renderFinished semaphore per swapchain image — safe reuse because image I
    // won't be re-acquired until its previous presentation completes
    ctx.renderFinished.resize(ctx.swapImages.size());
    for (auto& s : ctx.renderFinished)
        vkCreateSemaphore(ctx.device, &sem, nullptr, &s);

    // Per-image in-flight fence tracking — VK_NULL_HANDLE = not yet submitted
    ctx.imagesInFlight.assign(ctx.swapImages.size(), VK_NULL_HANDLE);

    for (int i = 0; i < VulkanContext::MAX_FRAMES; ++i)
        vkCreateFence(ctx.device, &fen, nullptr, &ctx.inFlight[i]);

    return ctx;
}

void vk_recreate_swapchain(VulkanContext& ctx, GLFWwindow* window) {
    vkDeviceWaitIdle(ctx.device);
    for (auto fb : ctx.framebuffers) vkDestroyFramebuffer(ctx.device, fb, nullptr);
    for (auto iv : ctx.swapViews)    vkDestroyImageView(ctx.device, iv, nullptr);
    vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
    create_swapchain(ctx, window);
    create_framebuffers(ctx);

    // Resize renderFinished if swapchain image count changed
    VkSemaphoreCreateInfo sem{};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (auto s : ctx.renderFinished) vkDestroySemaphore(ctx.device, s, nullptr);
    ctx.renderFinished.resize(ctx.swapImages.size());
    for (auto& s : ctx.renderFinished) vkCreateSemaphore(ctx.device, &sem, nullptr, &s);

    // Reset per-image fence tracking after swapchain recreate
    ctx.imagesInFlight.assign(ctx.swapImages.size(), VK_NULL_HANDLE);
}

uint32_t vk_begin_frame(VulkanContext& ctx) {
    vkWaitForFences(ctx.device, 1, &ctx.inFlight[ctx.currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t image_index;
    VkResult result = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX,
                                             ctx.imageAvailable[ctx.currentFrame],
                                             VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) return UINT32_MAX;

    // If a previous frame is still using this swapchain image's command buffer,
    // wait for that specific fence before resetting.  With 2 frames-in-flight but
    // 3 swapchain images, image_index and currentFrame can diverge, so the
    // inFlight[currentFrame] wait above does NOT guarantee this image is free.
    if (ctx.imagesInFlight[image_index] != VK_NULL_HANDLE)
        vkWaitForFences(ctx.device, 1, &ctx.imagesInFlight[image_index], VK_TRUE, UINT64_MAX);

    // Record which fence now owns this image's slot
    ctx.imagesInFlight[image_index] = ctx.inFlight[ctx.currentFrame];

    vkResetFences(ctx.device, 1, &ctx.inFlight[ctx.currentFrame]);
    vkResetCommandBuffer(ctx.commandBuffers[image_index], 0);
    return image_index;
}

void vk_end_frame(VulkanContext& ctx, uint32_t imageIndex) {
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &ctx.imageAvailable[ctx.currentFrame];
    si.pWaitDstStageMask    = wait_stages;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &ctx.commandBuffers[imageIndex];
    // Use renderFinished indexed by swapchain image — safe because image I won't be
    // re-acquired until its present completes, so this semaphore is free to reuse.
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &ctx.renderFinished[imageIndex];
    vkQueueSubmit(ctx.graphicsQueue, 1, &si, ctx.inFlight[ctx.currentFrame]);

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &ctx.renderFinished[imageIndex];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &ctx.swapchain;
    pi.pImageIndices      = &imageIndex;
    vkQueuePresentKHR(ctx.presentQueue, &pi);

    ctx.currentFrame = (ctx.currentFrame + 1) % VulkanContext::MAX_FRAMES;
}

void vk_destroy(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    for (int i = 0; i < VulkanContext::MAX_FRAMES; ++i)
        vkDestroySemaphore(ctx.device, ctx.imageAvailable[i], nullptr);
    for (auto s : ctx.renderFinished)
        vkDestroySemaphore(ctx.device, s, nullptr);
    for (int i = 0; i < VulkanContext::MAX_FRAMES; ++i)
        vkDestroyFence(ctx.device, ctx.inFlight[i], nullptr);

    vkDestroyDescriptorPool(ctx.device, ctx.descriptorPool, nullptr);
    vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);

    for (auto fb : ctx.framebuffers) vkDestroyFramebuffer(ctx.device, fb, nullptr);
    for (auto iv : ctx.swapViews)    vkDestroyImageView(ctx.device, iv, nullptr);

    vkDestroyRenderPass(ctx.device, ctx.renderPass, nullptr);
    vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroySurfaceKHR(ctx.instance, ctx.surface, nullptr);

    if (ctx.enableValidation && ctx.debugMessenger) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(ctx.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func) func(ctx.instance, ctx.debugMessenger, nullptr);
    }

    vkDestroyInstance(ctx.instance, nullptr);
}
