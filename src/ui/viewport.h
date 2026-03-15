#pragma once

#include <vulkan/vulkan.h>
#include <imgui.h>

struct ViewportPanel {
    ImVec2 size     = { 800, 600 };
    ImVec2 origin   = {};           // screen-space top-left of the image
    bool   resized  = false;

    // Per-frame mouse events
    bool   hovered       = false;
    bool   lmb_clicked   = false;   // clean single click (no drag threshold exceeded)
    bool   lmb_dragging   = false;   // drag in progress (>3px threshold)
    ImVec2 lmb_drag_delta = {};      // per-frame mouse delta while dragging
    bool   hdri_dragging  = false;   // Shift+LMB → rotate HDRI
    ImVec2 hdri_drag_delta = {};     // per-frame delta for HDRI yaw/pitch
    bool   rmb_clicked   = false;   // RMB released without drag → set orbit pivot
    bool   mmb_dragging  = false;   // MMB held + dragging → pan camera
    ImVec2 mmb_drag_delta = {};
    ImVec2 mouse_uv       = {};     // [0,1] position in viewport
    float  scroll_y       = 0.f;
};

void viewport_draw(ViewportPanel& vp, VkDescriptorSet descriptor);
