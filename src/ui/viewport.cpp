#include "viewport.h"
#include "control_panel.h"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

void viewport_draw(ViewportPanel& vp, VkDescriptorSet descriptor, ControlPanelState& controls) {
    vp.lmb_clicked    = false;
    vp.lmb_dragging   = false;
    vp.lmb_drag_delta = { 0.f, 0.f };
    vp.rmb_clicked    = false;
    vp.mmb_dragging   = false;
    vp.mmb_drag_delta = { 0.f, 0.f };
    vp.scroll_y       = 0.f;
    vp.hovered        = false;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_MenuBar);

    if (ImGui::BeginMenuBar()) {
        control_panel_draw_main_menu(controls);

        // AOV pass selector in the menu bar
        // Maps combo index → ViewportPassMode enum value
        static const char* vp_aov_items[] = {
            "Beauty", "Solid", "Rasterized", "Depth", "Normal", "Albedo",
            "Metallic", "Roughness", "Emission", "SegId"
        };
        static const int vp_aov_map[] = {
            (int)ViewportPassMode::Final,
            (int)ViewportPassMode::Solid,
            (int)ViewportPassMode::Rasterized,
            (int)ViewportPassMode::DlssDepth,
            (int)ViewportPassMode::Normal,
            (int)ViewportPassMode::Albedo,
            (int)ViewportPassMode::Metallic,
            (int)ViewportPassMode::Roughness,
            (int)ViewportPassMode::Emission,
            (int)ViewportPassMode::Segmentation,
        };
        // Find current combo index from enum value
        int combo_idx = 0;
        for (int i = 0; i < IM_ARRAYSIZE(vp_aov_map); ++i)
            if (vp_aov_map[i] == controls.viewport_pass) { combo_idx = i; break; }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.f);
        if (ImGui::Combo("##vp_aov", &combo_idx,
                         vp_aov_items, IM_ARRAYSIZE(vp_aov_items)))
            controls.viewport_pass = vp_aov_map[combo_idx];

        ImGui::EndMenuBar();
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 1.f) avail.x = 1.f;
    if (avail.y < 1.f) avail.y = 1.f;

    vp.resized = (avail.x != vp.size.x || avail.y != vp.size.y);
    vp.size    = avail;
    vp.origin  = ImGui::GetCursorScreenPos();

    // InvisibleButton owns mouse input (Image does not)
    ImGui::InvisibleButton("##vp", avail,
        ImGuiButtonFlags_MouseButtonLeft |
        ImGuiButtonFlags_MouseButtonRight |
        ImGuiButtonFlags_MouseButtonMiddle);

    // Draw CUDA texture over the button area
    ImGui::GetWindowDrawList()->AddImage(
        (ImTextureID)(uintptr_t)descriptor,
        vp.origin,
        ImVec2(vp.origin.x + avail.x, vp.origin.y + avail.y));

    bool item_hovered = ImGui::IsItemHovered();
    bool item_active  = ImGui::IsItemActive();
    vp.hovered = item_hovered;

    // Track whether drag threshold was ever exceeded during each press
    static bool s_drag_exceeded = false;
    static bool s_rdrag_exceeded = false;
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))  s_drag_exceeded  = false;
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) s_rdrag_exceeded = false;

    if (item_hovered || item_active) {
        ImVec2 mouse = ImGui::GetIO().MousePos;
        vp.mouse_uv = {
            (mouse.x - vp.origin.x) / avail.x,
            (mouse.y - vp.origin.y) / avail.y
        };

        // Drag: only after moving >3px from click point
        // Shift+LMB → HDRI rotation; plain LMB → camera orbit
        if (item_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 3.f)) {
            s_drag_exceeded = true;
            ImVec2 delta    = ImGui::GetIO().MouseDelta;
            bool   shift    = ImGui::GetIO().KeyShift;
            if (shift) {
                vp.hdri_dragging   = true;
                vp.hdri_drag_delta = delta;
            } else {
                vp.lmb_dragging   = true;
                vp.lmb_drag_delta = delta;
            }
        }

        // LMB click: fires on release only if no drag occurred
        if (item_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !s_drag_exceeded)
            vp.lmb_clicked = true;

        // MMB drag → pan
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 1.f)) {
            vp.mmb_dragging   = true;
            vp.mmb_drag_delta = ImGui::GetIO().MouseDelta;
        }

        // RMB drag tracking (mirrors LMB logic)
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right, 3.f))
            s_rdrag_exceeded = true;

        // RMB click: fires on release only if no drag occurred
        if (item_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && !s_rdrag_exceeded)
            vp.rmb_clicked = true;

        if (item_hovered)
            vp.scroll_y = ImGui::GetIO().MouseWheel;
    }

    ImGui::End();
    ImGui::PopStyleVar();
}
