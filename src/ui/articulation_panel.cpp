#include "articulation_panel.h"
#include <imgui.h>
#include <cmath>

bool articulation_panel_draw(UrdfArticulation* handle, bool playback_active)
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

    if (playback_active) ImGui::BeginDisabled();

    if (ImGui::Button("Reset All")) {
        for (int i = 0; i < n; ++i) joints[i].angle = 0.0f;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear Locks")) urdf_ik_clear_all_locks(handle);

    // IK lock — which joint the solver leaves alone (controlled by
    // sliders or the [ / ] hotkeys). Default "Auto" = last movable joint.
    {
        int eff_lock = urdf_ik_lock_joint_effective(handle);
        int stored   = urdf_ik_lock_joint(handle);
        const char* preview = (stored < 0)
            ? (eff_lock >= 0 ? joints[eff_lock].name : "Auto")
            : joints[stored].name;
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::BeginCombo("IK lock ( [ / ] )", preview)) {
            if (ImGui::Selectable("Auto (last movable)", stored < 0))
                urdf_set_ik_lock_joint(handle, -1);
            for (int i = 0; i < n; ++i) {
                if (joints[i].type != 0 && joints[i].type != 2) continue;
                ImGui::PushID(i);
                if (ImGui::Selectable(joints[i].name, stored == i))
                    urdf_set_ik_lock_joint(handle, i);
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
    }

    ImGui::Separator();

    int locked = urdf_ik_lock_joint_effective(handle);
    for (int i = 0; i < n; ++i) {
        UrdfJointInfo& j = joints[i];
        ImGui::PushID(i);

        // Joint name as label — mark the primary ([ / ] hotkey) lock.
        if (i == locked) ImGui::Text("[L] %s", j.name);
        else             ImGui::Text("%s", j.name);

        // Right-aligned buttons: L (per-joint IK freeze toggle) + R (reset)
        float pair_w = 50.f;
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - pair_w + ImGui::GetCursorPosX());
        bool is_locked = urdf_ik_is_locked(handle, i);
        if (is_locked) ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(180, 120, 40, 255));
        if (ImGui::SmallButton("L")) urdf_ik_set_locked(handle, i, !is_locked);
        if (is_locked) ImGui::PopStyleColor();
        ImGui::SameLine();
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

    if (playback_active) ImGui::EndDisabled();

    ImGui::End();
    return changed;
}
