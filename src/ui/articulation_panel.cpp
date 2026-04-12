#include "articulation_panel.h"
#include <imgui.h>
#include <cmath>

bool articulation_panel_draw(UrdfArticulation* handle)
{
    if (!ImGui::Begin("Articulation")) { ImGui::End(); return false; }

    if (!handle || urdf_joint_count(handle) == 0) {
        ImGui::TextDisabled("No URDF loaded");
        ImGui::End();
        return false;
    }

    bool changed = false;
    int n = urdf_joint_count(handle);
    UrdfJointInfo* joints = urdf_joint_info(handle);

    if (ImGui::Button("Reset All")) {
        for (int i = 0; i < n; ++i) joints[i].angle = 0.0f;
        changed = true;
    }

    ImGui::Separator();

    for (int i = 0; i < n; ++i) {
        UrdfJointInfo& j = joints[i];
        ImGui::PushID(i);

        // Joint name as label
        ImGui::Text("%s", j.name);

        // Reset button on same line, right-aligned
        float btn_w = 24.f;
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - btn_w + ImGui::GetCursorPosX());
        if (ImGui::SmallButton("R")) { j.angle = 0; changed = true; }

        if (j.type == 0 || j.type == 2) {
            float deg = j.angle * (180.0f / 3.14159265f);
            float lo_deg = j.lower * (180.0f / 3.14159265f);
            float hi_deg = j.upper * (180.0f / 3.14159265f);
            if (j.type == 2) { lo_deg = -360.0f; hi_deg = 360.0f; }
            if (fabsf(deg) < 0.5f) deg = 0.0f;

            ImGui::SetNextItemWidth(-1.f);
            if (ImGui::SliderFloat("##slider", &deg, lo_deg, hi_deg, "%.1f deg")) {
                if (fabsf(deg) < 0.5f) deg = 0.0f;
                j.angle = deg * (3.14159265f / 180.0f);
                changed = true;
            }
        }
        else if (j.type == 1) {
            float mm = j.angle * 1000.0f;
            float lo = j.lower * 1000.0f;
            float hi = j.upper * 1000.0f;
            if (fabsf(mm) < 0.1f) mm = 0.0f;

            ImGui::SetNextItemWidth(-1.f);
            if (ImGui::SliderFloat("##slider", &mm, lo, hi, "%.1f mm")) {
                if (fabsf(mm) < 0.1f) mm = 0.0f;
                j.angle = mm * 0.001f;
                changed = true;
            }
        }

        if (i < n - 1) ImGui::Spacing();
        ImGui::PopID();
    }

    ImGui::End();
    return changed;
}
