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
            // Face-pick grasp setup
            {
                bool fp = gripper_state->face_pick_mode;
                if (fp) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.65f, 0.1f, 1.f));
                if (ImGui::Button(fp ? "Face pick  [ACTIVE — click a face]" : "Face pick",
                                  ImVec2(avail, 0.f)))
                    gripper_state->face_pick_mode = !fp;
                if (fp) ImGui::PopStyleColor();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Activate face-pick mode, then:\n"
                                      "  Click a finger-pad face  → green dot + grip normal\n"
                                      "  Click an object face     → red dot  + target normal\n"
                                      "Pick & Place will align the two faces for the grasp.");

                // Grip face status
                if (gripper_state->grip_face_set) {
                    ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.f), "Grip face:   set");
                    ImGui::SameLine();
                    if (ImGui::SmallButton("X##gf")) {
                        gripper_state->grip_face_set = false;
                        gripper_state->grip_face_tri_idx = -1;
                    }
                } else {
                    ImGui::TextDisabled("Grip face:   not set");
                }

                // Object face status
                if (gripper_state->pick_face_set) {
                    ImGui::TextColored(ImVec4(0.9f, 0.35f, 0.35f, 1.f), "Target face: set");
                    ImGui::SameLine();
                    if (ImGui::SmallButton("X##pf")) {
                        gripper_state->pick_face_set = false;
                        gripper_state->pick_face_obj_idx = -1;
                        gripper_state->pick_face_tri_idx = -1;
                    }
                } else {
                    ImGui::TextDisabled("Target face: not set");
                }

                ImGui::Text("Grip offset: %.3f  %.3f  %.3f",
                            gripper_state->grip_local[0],
                            gripper_state->grip_local[1],
                            gripper_state->grip_local[2]);
                if (ImGui::Button("Reset grip to EE origin", ImVec2(avail, 0.f))) {
                    gripper_state->grip_local[0]  = 0.f;
                    gripper_state->grip_local[1]  = 0.f;
                    gripper_state->grip_local[2]  = 0.f;
                    gripper_state->grip_face_set  = false;
                    gripper_state->pick_face_set  = false;
                    gripper_state->pick_face_obj_idx = -1;
                    gripper_state->grip_face_tri_idx = -1;
                    gripper_state->pick_face_tri_idx = -1;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Clear grip offset and both face picks.");
            }
        }

        // Explicit attach/detach — always shown, even when finger joints
        // aren't recognized, because attach-nearest falls back to the EE
        // grip point. Bypasses the proximity threshold that Close uses.
        {
            float avail = ImGui::GetContentRegionAvail().x;
            float bw    = (avail - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
            bool holding = gripper_state->grasp.active;

            if (holding) ImGui::BeginDisabled();
            if (ImGui::Button("Attach", ImVec2(bw, 0.f)))
                gripper_state->request_attach = true;
            if (holding) ImGui::EndDisabled();

            ImGui::SameLine();

            if (!holding) ImGui::BeginDisabled();
            if (ImGui::Button("Detach", ImVec2(bw, 0.f)))
                gripper_state->request_detach = true;
            if (!holding) ImGui::EndDisabled();
        }

        // Pick & Place — finds the nearest pickup*/pickups* prop and runs a full
        // pregrasp → grasp → lift → place → release → retract trajectory,
        // auto-picking the best approach direction (top or ±X/±Z) by IK
        // feasibility. Place target = the IK gizmo's current world position.
        {
            float avail = ImGui::GetContentRegionAvail().x;
            bool holding  = gripper_state->grasp.active;
            bool playing  = playback_active;
            bool disabled = holding || playing;

            // Approach mode: let the user force top-down (wrist perpendicular
            // to ground, fingers down) or side (wrist parallel, horizontal
            // fingers). Auto tries top first then falls back to sides.
            {
                const char* kModes[] = { "Auto", "Top", "Side" };
                int mode = gripper_state->pick_approach_mode;
                if (mode < 0 || mode > 2) mode = 0;
                ImGui::SetNextItemWidth(avail);
                if (ImGui::Combo("##pickmode", &mode, kModes, IM_ARRAYSIZE(kModes)))
                    gripper_state->pick_approach_mode = mode;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Grasp approach for Pick & Place:\n"
                        "  Auto — top-down first, sides as fallback.\n"
                        "  Top  — wrist perpendicular to ground, fingers down.\n"
                        "  Side — wrist parallel to ground, horizontal fingers.");
            }

            if (disabled) ImGui::BeginDisabled();
            if (ImGui::Button("Pick & Place", ImVec2(avail, 0.f)))
                gripper_state->request_pick_and_place = true;
            if (disabled) ImGui::EndDisabled();

            if (ImGui::IsItemHovered() && !disabled)
                ImGui::SetTooltip(
                    "Pick the nearest mesh named pickup*/pickups* and drop it 30 cm away on the floor.\n"
                    "Approach direction (within the selected mode) and drop side chosen\n"
                    "automatically by IK feasibility.");

            ImGui::Checkbox("Debug: target line", &gripper_state->show_pick_debug_line);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Draw a live line from the gripper grip point to the\n"
                    "nearest pickup* mesh centroid, so you can verify the\n"
                    "planner is aimed at the right object.");

            // Last-run status. Green = planned OK, red = failure.
            if (gripper_state->last_pick_msg[0]) {
                ImVec4 col = (gripper_state->last_pick_status > 0)
                                 ? ImVec4(0.4f, 0.9f, 0.4f, 1.f)
                                 : ImVec4(0.95f, 0.55f, 0.55f, 1.f);
                ImGui::TextColored(col, "%s", gripper_state->last_pick_msg);
            }
        }
    }

    // EE link — the IK chain runs from root up to this link.
    // If the arm has more joints than the IK gizmo shows, pick a deeper link.
    {
        std::vector<std::string> link_names;
        urdf_link_names(handle, link_names);
        const char* cur_ee = urdf_get_ee_link_name(handle);
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::BeginCombo("EE link", cur_ee ? cur_ee : "?")) {
            for (auto& ln : link_names) {
                bool sel = cur_ee && (ln == cur_ee);
                if (ImGui::Selectable(ln.c_str(), sel))
                    urdf_set_ee_link_name(handle, ln.c_str());
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "End-effector link: the IK chain ends here.\n"
                "If the blue grab-dots don't cover all arm joints,\n"
                "pick a deeper link (e.g. the hand or gripper base).");
    }

    // IK lock — which joint the solver leaves alone. Default "None" lets
    // every movable joint participate in viewport IK.
    {
        int eff_lock = urdf_ik_lock_joint_effective(handle);
        int stored   = urdf_ik_lock_joint(handle);
        const char* preview = (stored < 0)
            ? "None (all joints)"
            : joints[stored].name;
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::BeginCombo("IK lock", preview)) {
            if (ImGui::Selectable("None (all joints)", stored < 0))
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

    {
        UrdfIkYawDebug yd{};
        urdf_ik_yaw_debug(handle, &yd);
        if (ImGui::CollapsingHeader("IK yaw debug")) {
            ImGui::Text("State: %s", yd.reason[0] ? yd.reason : "none");
            ImGui::Text("Joint: %s  idx=%d chain=%d",
                        yd.joint_name[0] ? yd.joint_name : "-",
                        yd.joint_idx, yd.chain_idx);
            ImGui::Text("Axis world: %.3f  %.3f  %.3f",
                        yd.axis_world[0], yd.axis_world[1], yd.axis_world[2]);
            ImGui::Text("Joint pos:  %.3f  %.3f  %.3f",
                        yd.joint_world[0], yd.joint_world[1], yd.joint_world[2]);
            ImGui::Text("EE pos:     %.3f  %.3f  %.3f",
                        yd.ee_world[0], yd.ee_world[1], yd.ee_world[2]);
            ImGui::Text("Target:     %.3f  %.3f  %.3f",
                        yd.target_world[0], yd.target_world[1], yd.target_world[2]);
            ImGui::Text("Gates: radial %.2f  ext %.2f  target %.2f",
                        yd.radial_gate, yd.extension_gate, yd.target_gate);
            ImGui::Text("Yaw err %.3f  gain %.3f  step %.4f  dot %.4f",
                        yd.yaw_err, yd.yaw_gain, yd.yaw_step, yd.drive_dot);
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

    int locked = urdf_ik_lock_joint(handle) >= 0 ? urdf_ik_lock_joint_effective(handle) : -1;
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
