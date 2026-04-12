#include "robot_demo_panel.h"
#include <imgui.h>
#include <cmath>
#include <fstream>
#include <algorithm>

// nlohmann/json bundled with tinygltf
#include "json.hpp"
using json = nlohmann::json;

// Windows native file dialog
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>

static bool win32_save_dialog(char* out_path, int max_len)
{
    OPENFILENAMEA ofn = {};
    out_path[0] = '\0';
    ofn.lStructSize  = sizeof(ofn);
    ofn.hwndOwner    = nullptr;
    ofn.lpstrFilter  = "Robot Demo JSON (*.json)\0*.json\0All Files\0*.*\0";
    ofn.lpstrFile    = out_path;
    ofn.nMaxFile     = max_len;
    ofn.lpstrDefExt  = "json";
    ofn.Flags        = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
    return GetSaveFileNameA(&ofn) != 0;
}

static bool win32_open_dialog(char* out_path, int max_len)
{
    OPENFILENAMEA ofn = {};
    out_path[0] = '\0';
    ofn.lStructSize  = sizeof(ofn);
    ofn.hwndOwner    = nullptr;
    ofn.lpstrFilter  = "Robot Demo JSON (*.json)\0*.json\0All Files\0*.*\0";
    ofn.lpstrFile    = out_path;
    ofn.nMaxFile     = max_len;
    ofn.Flags        = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&ofn) != 0;
}
#endif

// ── Catmull-Rom interpolation ────────────────────────────────────────────────

static float catmull_rom(float p0, float p1, float p2, float p3, float t)
{
    // Centripetal Catmull-Rom with tau = 0.5
    float t2 = t * t, t3 = t2 * t;
    return 0.5f * ((2.f * p1) +
                   (-p0 + p2) * t +
                   (2.f*p0 - 5.f*p1 + 4.f*p2 - p3) * t2 +
                   (-p0 + 3.f*p1 - 3.f*p2 + p3) * t3);
}

// Interpolate all joint angles at time t using Catmull-Rom spline.
// Returns interpolated angles + whether gripper is closed at this time.
static bool interpolate_waypoints(const std::vector<RobotWaypoint>& wps,
                                  float t, std::vector<float>& out_angles,
                                  bool& out_gripper)
{
    if (wps.empty()) return false;
    if (wps.size() == 1) {
        out_angles = wps[0].angles;
        out_gripper = wps[0].gripper_closed;
        return true;
    }

    // Clamp t to trajectory range
    float t0 = wps.front().time;
    float t1 = wps.back().time;
    t = std::max(t0, std::min(t1, t));

    // Find segment: wps[seg] .. wps[seg+1]
    int seg = 0;
    for (int i = 0; i < (int)wps.size() - 1; i++) {
        if (t >= wps[i].time && t <= wps[i+1].time) { seg = i; break; }
        if (i == (int)wps.size() - 2) seg = i; // clamp to last segment
    }

    float seg_start = wps[seg].time;
    float seg_end   = wps[seg + 1].time;
    float seg_len   = seg_end - seg_start;
    float u = (seg_len > 1e-6f) ? (t - seg_start) / seg_len : 0.f;

    // Gripper: snap to the target waypoint's state
    out_gripper = (u < 0.5f) ? wps[seg].gripper_closed : wps[seg + 1].gripper_closed;

    // 4 control points for Catmull-Rom (clamp at boundaries)
    int i0 = std::max(0, seg - 1);
    int i1 = seg;
    int i2 = seg + 1;
    int i3 = std::min((int)wps.size() - 1, seg + 2);

    int n_joints = (int)wps[i1].angles.size();
    out_angles.resize(n_joints);

    for (int j = 0; j < n_joints; j++) {
        float p0 = (j < (int)wps[i0].angles.size()) ? wps[i0].angles[j] : 0.f;
        float p1 = (j < (int)wps[i1].angles.size()) ? wps[i1].angles[j] : 0.f;
        float p2 = (j < (int)wps[i2].angles.size()) ? wps[i2].angles[j] : 0.f;
        float p3 = (j < (int)wps[i3].angles.size()) ? wps[i3].angles[j] : 0.f;
        out_angles[j] = catmull_rom(p0, p1, p2, p3, u);
    }
    return true;
}

// ── Tick (playback) ──────────────────────────────────────────────────────────

bool robot_demo_tick(RobotDemoState& state, UrdfArticulation* handle, float dt)
{
    if (!state.playing || !handle || state.waypoints.size() < 2)
        return false;

    state.playback_time += dt * state.playback_speed;

    float end_time = state.waypoints.back().time;
    if (state.playback_time > end_time) {
        if (state.loop) {
            state.playback_time = state.waypoints.front().time;
            // Reset grasp on loop restart
            state.needs_grasp_reset = true;
            state.prev_gripper_closed = false;
        } else {
            state.playback_time = end_time;
            state.playing = false;
        }
    }

    // Interpolate joints
    std::vector<float> angles;
    bool gripper_closed = false;
    if (!interpolate_waypoints(state.waypoints, state.playback_time, angles, gripper_closed))
        return false;

    // Write interpolated angles into the articulation handle
    int n = urdf_joint_count(handle);
    UrdfJointInfo* joints = urdf_joint_info(handle);
    for (int i = 0; i < n && i < (int)angles.size(); i++)
        joints[i].angle = angles[i];

    // Track gripper state for grasp transitions
    state.gripper_closed = gripper_closed;

    return true;
}

// ── Grasp update ─────────────────────────────────────────────────────────────

// 3x3 matrix helpers for rotation
static void mat3_from_mat4(const float* m16, float* m9) {
    m9[0]=m16[0]; m9[1]=m16[1]; m9[2]=m16[2];
    m9[3]=m16[4]; m9[4]=m16[5]; m9[5]=m16[6];
    m9[6]=m16[8]; m9[7]=m16[9]; m9[8]=m16[10];
}
static void mat3_transpose(const float* in, float* out) {
    out[0]=in[0]; out[1]=in[3]; out[2]=in[6];
    out[3]=in[1]; out[4]=in[4]; out[5]=in[7];
    out[6]=in[2]; out[7]=in[5]; out[8]=in[8];
}
static float3 mat3_mul_vec(const float* m, float3 v) {
    return make_float3(m[0]*v.x + m[1]*v.y + m[2]*v.z,
                       m[3]*v.x + m[4]*v.y + m[5]*v.z,
                       m[6]*v.x + m[7]*v.y + m[8]*v.z);
}

void robot_demo_update_grasp(RobotDemoState& state, UrdfArticulation* handle,
                             std::vector<Sphere>& spheres,
                             std::vector<MeshObject>& objects,
                             std::vector<Triangle>& all_prims)
{
    if (!handle) return;

    float3 ee = urdf_end_effector_pos(handle);

    // Detect gripper transitions
    bool just_closed = (state.gripper_closed && !state.prev_gripper_closed);
    bool just_opened = (!state.gripper_closed && state.prev_gripper_closed);
    state.prev_gripper_closed = state.gripper_closed;

    if (just_closed && !state.grasp.active) {
        float best_dist = state.grasp_threshold;
        int   best_idx  = -1;
        bool  best_is_mesh = false;

        // Check spheres
        for (int i = 0; i < (int)spheres.size(); i++) {
            if (spheres[i].radius > 50.f) continue;
            float3 d = make_float3(spheres[i].center.x - ee.x,
                                   spheres[i].center.y - ee.y,
                                   spheres[i].center.z - ee.z);
            float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
            if (dist < best_dist) {
                best_dist = dist; best_idx = i; best_is_mesh = false;
            }
        }

        // Check mesh objects by centroid
        for (int i = 0; i < (int)objects.size(); i++) {
            if (objects[i].hidden) continue;
            float3 c = objects[i].centroid;
            float3 d = make_float3(c.x - ee.x, c.y - ee.y, c.z - ee.z);
            float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
            if (dist < best_dist) {
                best_dist = dist; best_idx = i; best_is_mesh = true;
            }
        }

        if (best_idx >= 0) {
            state.grasp.active  = true;
            state.grasp.is_mesh = best_is_mesh;
            state.grasp.obj_idx = best_idx;

            // Get ee full transform at grab time
            float ee_mat[16];
            urdf_fk_ee_transform(handle, ee_mat);
            float ee_rot[9];
            mat3_from_mat4(ee_mat, ee_rot);
            mat3_transpose(ee_rot, state.grasp.grab_inv_rot);  // inverse = transpose for rotation

            if (best_is_mesh) {
                float3 c = objects[best_idx].centroid;
                if (!state.grasp.has_original) {
                    state.grasp.original_pos = c;
                    state.grasp.has_original = true;
                }
                // Store all vertices in ee-local space for rotation support,
                // and save original world positions for reset
                int target_id = objects[best_idx].obj_id;
                state.grasp.local_verts.clear();
                state.grasp.original_verts.clear();
                for (auto& tri : all_prims) {
                    if (tri.obj_id != target_id) continue;
                    float3 verts[3] = { tri.v0, tri.v1, tri.v2 };
                    for (auto& v : verts) {
                        // Save original world position
                        state.grasp.original_verts.push_back(v);
                        // Transform to ee-local space
                        float3 rel = make_float3(v.x - ee.x, v.y - ee.y, v.z - ee.z);
                        state.grasp.local_verts.push_back(mat3_mul_vec(state.grasp.grab_inv_rot, rel));
                    }
                }
                state.grasp.offset = make_float3(c.x - ee.x, c.y - ee.y, c.z - ee.z);
            } else {
                state.grasp.offset = make_float3(
                    spheres[best_idx].center.x - ee.x,
                    spheres[best_idx].center.y - ee.y,
                    spheres[best_idx].center.z - ee.z);
                if (!state.grasp.has_original) {
                    state.grasp.original_pos = spheres[best_idx].center;
                    state.grasp.has_original = true;
                }
            }
        }
    }

    if (just_opened && state.grasp.active) {
        state.grasp.active = false;
    }

    // Move attached object each frame
    if (!state.grasp.active || state.grasp.obj_idx < 0) return;

    if (!state.grasp.is_mesh) {
        // Sphere: position only (no rotation for spheres)
        float3 new_pos = make_float3(ee.x + state.grasp.offset.x,
                                     ee.y + state.grasp.offset.y,
                                     ee.z + state.grasp.offset.z);
        if (state.grasp.obj_idx < (int)spheres.size())
            spheres[state.grasp.obj_idx].center = new_pos;
    } else {
        // Mesh: apply full transform (position + rotation)
        if (state.grasp.obj_idx < (int)objects.size()) {
            float ee_mat[16];
            urdf_fk_ee_transform(handle, ee_mat);
            float cur_rot[9];
            mat3_from_mat4(ee_mat, cur_rot);

            int target_id = objects[state.grasp.obj_idx].obj_id;
            int vi = 0;
            float3 centroid_sum = make_float3(0, 0, 0);
            int vert_count = 0;

            for (auto& tri : all_prims) {
                if (tri.obj_id != target_id) continue;
                float3* tv[3] = { &tri.v0, &tri.v1, &tri.v2 };
                for (int k = 0; k < 3; k++) {
                    if (vi < (int)state.grasp.local_verts.size()) {
                        // local_vert is in ee-local space at grab time
                        // new world pos = ee_pos + cur_rot * local_vert
                        float3 world = mat3_mul_vec(cur_rot, state.grasp.local_verts[vi]);
                        tv[k]->x = ee.x + world.x;
                        tv[k]->y = ee.y + world.y;
                        tv[k]->z = ee.z + world.z;
                        centroid_sum.x += tv[k]->x;
                        centroid_sum.y += tv[k]->y;
                        centroid_sum.z += tv[k]->z;
                        vert_count++;
                    }
                    vi++;
                }
            }
            if (vert_count > 0) {
                float inv = 1.f / (float)vert_count;
                objects[state.grasp.obj_idx].centroid = make_float3(
                    centroid_sum.x * inv, centroid_sum.y * inv, centroid_sum.z * inv);
            }
        }
    }
}

// Reset grasped object to its original position + rotation.
void robot_demo_reset_grasp(RobotDemoState& state,
                            std::vector<Sphere>& spheres,
                            std::vector<MeshObject>& objects,
                            std::vector<Triangle>& all_prims)
{
    if (!state.grasp.has_original) return;

    if (!state.grasp.is_mesh) {
        if (state.grasp.obj_idx >= 0 && state.grasp.obj_idx < (int)spheres.size())
            spheres[state.grasp.obj_idx].center = state.grasp.original_pos;
    } else {
        if (state.grasp.obj_idx >= 0 && state.grasp.obj_idx < (int)objects.size()) {
            int target_id = objects[state.grasp.obj_idx].obj_id;
            int vi = 0;

            if (!state.grasp.original_verts.empty()) {
                // Restore exact original vertices (position + rotation)
                for (auto& tri : all_prims) {
                    if (tri.obj_id != target_id) continue;
                    if (vi + 2 < (int)state.grasp.original_verts.size()) {
                        tri.v0 = state.grasp.original_verts[vi];
                        tri.v1 = state.grasp.original_verts[vi + 1];
                        tri.v2 = state.grasp.original_verts[vi + 2];
                    }
                    vi += 3;
                }
            }
            objects[state.grasp.obj_idx].centroid = state.grasp.original_pos;
        }
    }
    state.grasp.active = false;
}

// ── Save / Load JSON ─────────────────────────────────────────────────────────

bool robot_demo_save(const RobotDemoState& state, const char* path)
{
    json j;
    json wps = json::array();
    for (auto& wp : state.waypoints) {
        json w;
        w["time"] = wp.time;
        w["angles"] = wp.angles;
        w["gripper_closed"] = wp.gripper_closed;
        wps.push_back(w);
    }
    j["waypoints"] = wps;

    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << j.dump(2);
    return true;
}

bool robot_demo_load(RobotDemoState& state, const char* path)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;

    json j;
    try { f >> j; } catch (...) { return false; }

    state.waypoints.clear();
    if (!j.contains("waypoints")) return false;

    for (auto& w : j["waypoints"]) {
        RobotWaypoint wp;
        wp.time = w.value("time", 0.f);
        wp.gripper_closed = w.value("gripper_closed", false);
        if (w.contains("angles") && w["angles"].is_array()) {
            for (auto& a : w["angles"])
                wp.angles.push_back(a.get<float>());
        }
        state.waypoints.push_back(std::move(wp));
    }
    return !state.waypoints.empty();
}

// ── ImGui Panel ──────────────────────────────────────────────────────────────

void robot_demo_panel_draw(RobotDemoState& state, UrdfArticulation* handle)
{
    if (!ImGui::Begin("Robot Demo")) { ImGui::End(); return; }

    if (!handle || urdf_joint_count(handle) == 0) {
        ImGui::TextDisabled("No URDF loaded");
        ImGui::End();
        return;
    }

    int n_joints = urdf_joint_count(handle);
    UrdfJointInfo* joints = urdf_joint_info(handle);

    // Mode toggle
    if (ImGui::RadioButton("Record", state.recording)) {
        state.recording = true;
        state.playing = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Playback", !state.recording)) {
        state.recording = false;
    }

    ImGui::Separator();

    // ── Record mode ──────────────────────────────────────────────────────
    if (state.recording) {
        // IK mode + Gripper toggle
        ImGui::Checkbox("IK gizmo", &state.ik_enabled);
        ImGui::Checkbox("Gripper closed", &state.gripper_closed);

        // Add waypoint button
        if (ImGui::Button("Add Waypoint", ImVec2(-1, 0))) {
            RobotWaypoint wp;
            // Auto-assign time: last waypoint time + 1s, or 0 for first
            wp.time = state.waypoints.empty() ? 0.f :
                      state.waypoints.back().time + 1.0f;
            wp.gripper_closed = state.gripper_closed;
            wp.angles.resize(n_joints);
            for (int i = 0; i < n_joints; i++)
                wp.angles[i] = joints[i].angle;
            state.waypoints.push_back(std::move(wp));
        }

        ImGui::Spacing();

        // Waypoint list
        int remove_idx = -1;
        for (int i = 0; i < (int)state.waypoints.size(); i++) {
            ImGui::PushID(i);
            auto& wp = state.waypoints[i];

            // Time editor
            ImGui::SetNextItemWidth(60.f);
            ImGui::DragFloat("##t", &wp.time, 0.05f, 0.f, 100.f, "%.1fs");
            ImGui::SameLine();

            // Gripper indicator
            ImGui::Text(wp.gripper_closed ? "[GRIP]" : "[OPEN]");
            ImGui::SameLine();

            // Preview: clicking a waypoint loads its pose
            char label[32];
            snprintf(label, sizeof(label), "WP %d", i);
            if (ImGui::SmallButton(label)) {
                for (int j = 0; j < n_joints && j < (int)wp.angles.size(); j++)
                    joints[j].angle = wp.angles[j];
                state.gripper_closed = wp.gripper_closed;
            }
            ImGui::SameLine();

            // Update: overwrite this waypoint with current pose
            if (ImGui::SmallButton("Upd")) {
                wp.gripper_closed = state.gripper_closed;
                wp.angles.resize(n_joints);
                for (int j = 0; j < n_joints; j++)
                    wp.angles[j] = joints[j].angle;
            }
            ImGui::SameLine();

            // Delete
            if (ImGui::SmallButton("X")) remove_idx = i;

            ImGui::PopID();
        }

        if (remove_idx >= 0)
            state.waypoints.erase(state.waypoints.begin() + remove_idx);
    }

    // ── Play mode ────────────────────────────────────────────────────────
    else {
        if (state.waypoints.size() < 2) {
            ImGui::TextDisabled("Need at least 2 waypoints");
        } else {
            // Play / Pause / Stop
            if (state.playing) {
                if (ImGui::Button("Pause")) state.playing = false;
            } else {
                if (ImGui::Button("Play")) {
                    state.playing = true;
                    if (state.playback_time >= state.waypoints.back().time) {
                        state.playback_time = state.waypoints.front().time;
                        state.needs_grasp_reset = true;
                    }
                    state.prev_gripper_closed = false;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                state.playing = false;
                state.playback_time = state.waypoints.front().time;
                state.needs_grasp_reset = true;
                state.prev_gripper_closed = false;
            }

            // Timeline scrubber — dragging updates the robot pose immediately
            float t_min = state.waypoints.front().time;
            float t_max = state.waypoints.back().time;
            ImGui::SetNextItemWidth(-1.f);
            if (ImGui::SliderFloat("##timeline", &state.playback_time,
                                   t_min, t_max, "%.2fs"))
            {
                // Manual scrub: apply pose at this time
                std::vector<float> angles;
                bool grip = false;
                if (interpolate_waypoints(state.waypoints, state.playback_time, angles, grip)) {
                    int n = urdf_joint_count(handle);
                    UrdfJointInfo* jts = urdf_joint_info(handle);
                    for (int i = 0; i < n && i < (int)angles.size(); i++)
                        jts[i].angle = angles[i];
                    state.gripper_closed = grip;
                    state.scrub_changed = true;
                }
            }

            // Speed
            ImGui::SetNextItemWidth(120.f);
            ImGui::SliderFloat("Speed", &state.playback_speed, 0.1f, 3.0f, "%.1fx");
            ImGui::SameLine();
            ImGui::Checkbox("Loop", &state.loop);
        }

        // Grasp info
        if (state.grasp.active) {
            if (state.grasp.is_mesh)
                ImGui::TextColored(ImVec4(0.3f, 1.f, 0.3f, 1.f), "Grasping mesh obj %d",
                                   state.grasp.obj_idx);
            else
                ImGui::TextColored(ImVec4(0.3f, 1.f, 0.3f, 1.f), "Grasping sphere %d",
                                   state.grasp.obj_idx);
        }
    }

    ImGui::Separator();

    // ── Save / Load (both modes) ─────────────────────────────────────────
    ImGui::Text("Trajectory (%d waypoints)", (int)state.waypoints.size());

    if (ImGui::Button("Save...")) {
#ifdef _WIN32
        char path[512];
        if (win32_save_dialog(path, sizeof(path)))
            robot_demo_save(state, path);
#else
        robot_demo_save(state, "robot_demo.json");
#endif
    }
    ImGui::SameLine();
    if (ImGui::Button("Load...")) {
#ifdef _WIN32
        char path[512];
        if (win32_open_dialog(path, sizeof(path))) {
            if (robot_demo_load(state, path)) {
                state.playing = false;
                state.playback_time = 0.f;
                state.grasp.active = false;
            }
        }
#else
        if (robot_demo_load(state, "robot_demo.json")) {
            state.playing = false;
            state.playback_time = 0.f;
            state.grasp.active = false;
        }
#endif
    }

    // Grasp threshold
    ImGui::SetNextItemWidth(120.f);
    ImGui::SliderFloat("Grasp dist", &state.grasp_threshold, 0.01f, 1.0f, "%.2f");

    ImGui::End();
}
