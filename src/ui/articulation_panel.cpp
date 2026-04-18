#include "articulation_panel.h"
#include "robot_demo_panel.h"   // RobotDemoState + gripper_toggle
#include <imgui.h>
#include <cmath>
#include <vector>

bool articulation_panel_draw(UrdfArticulation* handle, bool playback_active,
                             RobotDemoState* gripper_state)
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

    // Gripper Close/Open — snaps fingers to their lower/upper limits. The
    // grasp pipeline tests proximity against every finger link (not just a
    // single tool-frame point), so the object attaches if any finger is
    // within the grip threshold when Close is pressed.
    if (gripper_state) {
        std::vector<int> fingers;
        urdf_gripper_finger_indices(handle, fingers);
        if (!fingers.empty()) {
            bool is_closed = gripper_state->gripper_closed;

            if (gripper_state->grasp.active) {
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.f), "Holding object");
            } else {
                ImGui::TextDisabled(is_closed ? "Closed" : "Open");
            }

            float avail = ImGui::GetContentRegionAvail().x;
            float bw    = (avail - ImGui::GetStyle().ItemSpacing.x) * 0.5f;

            if (is_closed) ImGui::BeginDisabled();
            if (ImGui::Button("Close", ImVec2(bw, 0.f))) {
                if (gripper_toggle(*gripper_state, handle, /*close=*/true))
                    changed = true;
            }
            if (is_closed) ImGui::EndDisabled();

            ImGui::SameLine();

            if (!is_closed) ImGui::BeginDisabled();
            if (ImGui::Button("Open", ImVec2(bw, 0.f))) {
                if (gripper_toggle(*gripper_state, handle, /*close=*/false))
                    changed = true;
            }
            if (!is_closed) ImGui::EndDisabled();

            // Tuning — tolerance for auto-attach and how far forward the
            // grip reference sits from the ee origin.
            ImGui::SetNextItemWidth(avail);
            ImGui::SliderFloat("##grip_thresh", &gripper_state->grasp_threshold,
                               0.05f, 0.50f, "grip threshold %.2f m");
            ImGui::SetNextItemWidth(avail);
            ImGui::SliderFloat("##grip_fwd", &gripper_state->grip_forward,
                               0.00f, 0.20f, "grip forward %.2f m");
        }
    }

    // IK lock — which joint the solver leaves alone (controlled by
    // sliders or the [ / ] hotkeys). Default "Auto" = last movable joint.
    {
        int eff_lock = urdf_ik_lock_joint_effective(handle);
        int stored   = urdf_ik_lock_joint(handle);
        const char* preview = (stored < 0)
            ? (eff_lock >= 0 ? joints[eff_lock].name : "Auto")
            : joints[stored].name;
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::BeginCombo("IK lock", preview)) {
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

    // Keyboard joint — which joint [ / ] drives. "Follow IK lock" keeps the
    // legacy behavior; picking a specific joint decouples keyboard control
    // from the IK-lock choice so the user can nudge any joint by hand.
    {
        int kb_stored = urdf_kb_joint(handle);
        int kb_eff    = urdf_kb_joint_effective(handle);
        const char* kb_preview = (kb_stored < 0)
            ? (kb_eff >= 0 ? joints[kb_eff].name : "Follow IK lock")
            : joints[kb_stored].name;
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::BeginCombo("Keyboard ( [ / ] )", kb_preview)) {
            if (ImGui::Selectable("Follow IK lock", kb_stored < 0))
                urdf_set_kb_joint(handle, -1);
            for (int i = 0; i < n; ++i) {
                // Revolute, prismatic, continuous — anything drivable.
                int t = joints[i].type;
                if (t != 0 && t != 1 && t != 2) continue;
                ImGui::PushID(i + 10000);
                if (ImGui::Selectable(joints[i].name, kb_stored == i))
                    urdf_set_kb_joint(handle, i);
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
    }

    ImGui::Separator();

    int locked = urdf_ik_lock_joint_effective(handle);
    int kb_eff = urdf_kb_joint_effective(handle);
    for (int i = 0; i < n; ++i) {
        UrdfJointInfo& j = joints[i];
        ImGui::PushID(i);

        // Joint name as label — mark the IK lock [L] and keyboard joint [K].
        const char* l_tag = (i == locked) ? "[L]" : "";
        const char* k_tag = (i == kb_eff) ? "[K]" : "";
        if (l_tag[0] || k_tag[0]) ImGui::Text("%s%s %s", l_tag, k_tag, j.name);
        else                       ImGui::Text("%s", j.name);

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
