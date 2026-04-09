#include "articulation_panel.h"
#include <imgui.h>
#include <cmath>
#include <algorithm>

bool articulation_panel_draw(UrdfArticulation* handle)
{
    if (!ImGui::Begin("Articulation")) { ImGui::End(); return false; }

    if (!handle || urdf_joint_count(handle) == 0) {
        ImGui::TextDisabled("No URDF loaded — load a .urdf file to articulate joints.");
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

        if (j.type == 0 || j.type == 2) {
            // Revolute / continuous — show in degrees, store in radians
            float deg = j.angle * (180.0f / 3.14159265f);
            float lo_deg = j.lower * (180.0f / 3.14159265f);
            float hi_deg = j.upper * (180.0f / 3.14159265f);

            if (j.type == 2) { lo_deg = -360.0f; hi_deg = 360.0f; }

            // Snap to zero when very close
            if (fabsf(deg) < 0.5f) deg = 0.0f;

            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            if (ImGui::SliderFloat(j.name, &deg, lo_deg, hi_deg, "%.1f")) {
                // Snap to zero in a small dead zone
                if (fabsf(deg) < 0.5f) deg = 0.0f;
                j.angle = deg * (3.14159265f / 180.0f);
                changed = true;
            }
            // Double-click to type exact value
            if (ImGui::IsItemDeactivatedAfterEdit()) changed = true;
        }
        else if (j.type == 1) {
            // Prismatic — show in mm, store in meters
            float mm = j.angle * 1000.0f;
            float lo = j.lower * 1000.0f;
            float hi = j.upper * 1000.0f;

            if (fabsf(mm) < 0.1f) mm = 0.0f;

            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            if (ImGui::SliderFloat(j.name, &mm, lo, hi, "%.1f mm")) {
                if (fabsf(mm) < 0.1f) mm = 0.0f;
                j.angle = mm * 0.001f;
                changed = true;
            }
        }

        ImGui::PopID();
    }

    ImGui::End();
    return changed;
}
