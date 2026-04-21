#include "robot_demo_panel.h"
#include "../sim/props_physics.h"
#include <imgui.h>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstring>
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

static bool step_bool_at_time(const std::vector<RobotWaypoint>& wps, float t,
                              bool RobotWaypoint::*field)
{
    if (wps.empty()) return false;
    if (t <= wps.front().time) return wps.front().*field;
    for (int i = 0; i < (int)wps.size() - 1; i++) {
        if (t < wps[i + 1].time) return wps[i].*field;
    }
    return wps.back().*field;
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

    // Gripper: hold each keyed state until the next waypoint time. Flipping at
    // the segment midpoint makes close/attach events fire before the arm has
    // actually reached the keyed pose.
    out_gripper = step_bool_at_time(wps, t, &RobotWaypoint::gripper_closed);

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
        // A dwell segment is intentional: the arm should stay exactly at the
        // keyed pose while the gripper closes/opens. Catmull-Rom would still
        // move through equal endpoints because neighboring waypoints define
        // non-zero tangents.
        out_angles[j] = (fabsf(p2 - p1) < 1e-7f)
            ? p1
            : catmull_rom(p0, p1, p2, p3, u);
    }
    return true;
}

// ── Tick (playback) ──────────────────────────────────────────────────────────

// Resolve the attach flag at the current playback time. Attach/detach is a
// discrete event and must happen at the keyed waypoint, not halfway through
// the previous segment.
static bool sample_attached(const std::vector<RobotWaypoint>& wps, float t)
{
    return step_bool_at_time(wps, t, &RobotWaypoint::attached);
}

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
            state.prev_attached = false;
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

    // Attach/detach keyframing — raise an intent whenever we cross a
    // transition; main.cpp consumes it alongside the button-driven ones.
    // Backwards-compat: if the loaded trajectory never sets attached=true
    // anywhere (saved before the flag existed, or user only used gripper
    // close/open), fall back to treating gripper_closed as the attach
    // signal so old recordings still pick objects up on replay.
    bool any_keyed_attach = false;
    for (const auto& w : state.waypoints) {
        if (w.attached) { any_keyed_attach = true; break; }
    }
    bool want_attached = any_keyed_attach
        ? sample_attached(state.waypoints, state.playback_time)
        : gripper_closed;
    if (want_attached != state.prev_attached) {
        if (want_attached) state.request_attach = true;
        else               state.request_detach = true;
        state.prev_attached = want_attached;
    }

    return true;
}

// ── Gripper snap toggle ──────────────────────────────────────────────────────

bool gripper_toggle(RobotDemoState& state, UrdfArticulation* handle, bool close)
{
    if (!handle) return false;
    std::vector<int> fingers;
    urdf_gripper_finger_indices(handle, fingers);
    if (fingers.empty()) return false;

    int n = urdf_joint_count(handle);
    UrdfJointInfo* joints = urdf_joint_info(handle);
    bool changed = false;
    for (int ji : fingers) {
        if (ji < 0 || ji >= n) continue;
        UrdfJointInfo& j = joints[ji];
        float lo = j.lower, hi = j.upper;
        if (!(hi > lo)) { lo = 0.f; hi = 0.f; }
        float target = close ? lo : hi;
        if (j.angle != target) { j.angle = target; changed = true; }
    }
    state.gripper_closed = close;
    return changed;
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

static void compute_world_aabb_mesh(int obj_id,
                                    const std::vector<Triangle>& all_prims,
                                    float3& out_min, float3& out_max);

static float3 ee_origin(UrdfArticulation* handle)
{
    float m[16];
    urdf_fk_ee_transform(handle, m);
    return make_float3(m[3], m[7], m[11]);
}

// Extract one of the 6 local EE axes from the row-major 4×4 transform m,
// Grip anchor = EE origin + grip_local offset rotated into world space.
// grip_local is in EE-local frame; {0,0,0} = EE origin exactly.
// Drag the green debug dot in the viewport to position it — the offset is
// stored in EE-local so it follows the arm as it moves.
static float3 gripper_anchor(UrdfArticulation* handle, const float grip_local[3])
{
    float m[16];
    urdf_fk_ee_transform(handle, m);
    float ox = m[3],  oy = m[7],  oz = m[11];
    float wx = m[0]*grip_local[0] + m[1]*grip_local[1] + m[2]*grip_local[2];
    float wy = m[4]*grip_local[0] + m[5]*grip_local[1] + m[6]*grip_local[2];
    float wz = m[8]*grip_local[0] + m[9]*grip_local[1] + m[10]*grip_local[2];
    return make_float3(ox + wx, oy + wy, oz + wz);
}

// Shared attach: find nearest object to any contact point and attach it.
// If threshold is finite, only attach when distance is within it. Returns
// true if an object was attached.
static bool try_attach_nearest(RobotDemoState& state, UrdfArticulation* handle,
                               std::vector<Sphere>& spheres,
                               std::vector<MeshObject>& objects,
                               std::vector<Triangle>& all_prims,
                               float threshold)
{
    if (!handle || state.grasp.active) return false;

    float3 ee = gripper_anchor(handle, state.grip_local);

    std::vector<float3> contact_pts;
    urdf_gripper_finger_worlds(handle, contact_pts);
    contact_pts.push_back(ee);

    auto min_dist_to_contact = [&](float3 p) {
        float best = 1e30f;
        for (const auto& c : contact_pts) {
            float3 d = make_float3(p.x - c.x, p.y - c.y, p.z - c.z);
            float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
            if (dist < best) best = dist;
        }
        return best;
    };

    float best_dist = threshold;
    int   best_idx  = -1;
    bool  best_is_mesh = false;

    for (int i = 0; i < (int)spheres.size(); i++) {
        if (spheres[i].radius > 50.f) continue;
        float dist = min_dist_to_contact(spheres[i].center);
        if (dist < best_dist) { best_dist = dist; best_idx = i; best_is_mesh = false; }
    }

    std::vector<float3> mesh_centers((size_t)objects.size(), make_float3(0, 0, 0));
    for (int i = 0; i < (int)objects.size(); i++) {
        if (objects[i].hidden || objects[i].is_robot_part) continue;
        float3 mn, mx;
        compute_world_aabb_mesh(objects[i].obj_id, all_prims, mn, mx);
        if (mn.x > mx.x) continue;
        float3 center = make_float3(0.5f * (mn.x + mx.x),
                                    0.5f * (mn.y + mx.y),
                                    0.5f * (mn.z + mx.z));
        mesh_centers[(size_t)i] = center;
        float dist = min_dist_to_contact(center);
        if (dist < best_dist) { best_dist = dist; best_idx = i; best_is_mesh = true; }
    }

    if (best_idx < 0) return false;

    state.grasp.active  = true;
    state.grasp.is_mesh = best_is_mesh;
    state.grasp.obj_idx = best_idx;

    float ee_mat[16];
    urdf_fk_ee_transform(handle, ee_mat);
    float ee_rot[9];
    mat3_from_mat4(ee_mat, ee_rot);
    mat3_transpose(ee_rot, state.grasp.grab_inv_rot);

    if (best_is_mesh) {
        float3 c = mesh_centers[(size_t)best_idx];
        if (!state.grasp.has_original) {
            state.grasp.original_pos = c;
            state.grasp.has_original = true;
        }
        int target_id = objects[best_idx].obj_id;
        state.grasp.local_verts.clear();
        state.grasp.original_verts.clear();
        for (auto& tri : all_prims) {
            if (tri.obj_id != target_id) continue;
            float3 verts[3] = { tri.v0, tri.v1, tri.v2 };
            for (auto& v : verts) {
                state.grasp.original_verts.push_back(v);
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

    // Newly attached object cancels any in-progress free-fall for it.
    // Tell MuJoCo to drop the body so we don't double-simulate it.
    for (auto it = state.falling.begin(); it != state.falling.end(); ) {
        if (it->is_mesh == best_is_mesh && it->obj_idx == best_idx) {
            if (state.props_world && it->mujoco_handle >= 0)
                props_physics_remove(state.props_world, it->mujoco_handle);
            it = state.falling.erase(it);
        } else {
            ++it;
        }
    }

    return true;
}

void robot_demo_update_grasp(RobotDemoState& state, UrdfArticulation* handle,
                             std::vector<Sphere>& spheres,
                             std::vector<MeshObject>& objects,
                             std::vector<Triangle>& all_prims)
{
    if (!handle) return;

    float3 ee = gripper_anchor(handle, state.grip_local);

    // Detect gripper transitions
    bool just_closed = (state.gripper_closed && !state.prev_gripper_closed);
    bool just_opened = (!state.gripper_closed && state.prev_gripper_closed);
    state.prev_gripper_closed = state.gripper_closed;

    if (just_closed && !state.grasp.active) {
        try_attach_nearest(state, handle, spheres, objects, all_prims,
                           state.grasp_threshold);
    }

    if (just_opened && state.grasp.active) {
        // Classic "open the gripper" release — matches the record/playback
        // behavior before explicit Attach/Detach existed. Drop as free-fall
        // so the object obeys gravity after the arm lets go.
        FallingObject fo;
        fo.is_mesh  = state.grasp.is_mesh;
        fo.obj_idx  = state.grasp.obj_idx;
        fo.velocity = state.last_ee_vel;
        state.falling.push_back(fo);
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
    // Any in-flight bounces are cancelled on loop/reset — otherwise the
    // object would keep physics-falling past the restart.
    if (state.props_world) props_physics_clear(state.props_world);
    state.falling.clear();

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

// ── Explicit attach / detach buttons ─────────────────────────────────────────

bool robot_demo_force_attach(RobotDemoState& state, UrdfArticulation* handle,
                             std::vector<Sphere>& spheres,
                             std::vector<MeshObject>& objects,
                             std::vector<Triangle>& all_prims)
{
    // No threshold — grab whatever is closest to any contact point.
    return try_attach_nearest(state, handle, spheres, objects, all_prims, 1e30f);
}

bool robot_demo_snap_attach(RobotDemoState& state, UrdfArticulation* handle,
                            std::vector<Sphere>& spheres,
                            std::vector<MeshObject>& objects,
                            std::vector<Triangle>& all_prims,
                            int obj_idx, bool is_mesh)
{
    if (!handle || state.grasp.active || obj_idx < 0) return false;

    // Anchor the cube at the gripper contact center, derived from jaw/finger
    // link origins. This avoids Panda-specific +Z tool-frame assumptions and
    // avoids pinning SO-101 pickups to the wrist or moving_jaw pivot.
    float ee_mat_dbg[16];
    urdf_fk_ee_transform(handle, ee_mat_dbg);
    float3 ee = gripper_anchor(handle, state.grip_local);

    std::fprintf(stderr,
        "[snap_attach] is_mesh=%d obj_idx=%d\n"
        "              grip_anchor= (%.4f, %.4f, %.4f)\n"
        "              ee_origin  = (%.4f, %.4f, %.4f)\n",
        is_mesh ? 1 : 0, obj_idx,
        ee.x, ee.y, ee.z,
        ee_mat_dbg[3], ee_mat_dbg[7], ee_mat_dbg[11]);

    // Pre-teleport snapshot so reset/loop sends the cube back to its
    // SCENE position, not to wherever the grip point happened to be at
    // attach time. try_attach_nearest skips overwriting original_pos when
    // has_original is true, but it unconditionally rewrites original_verts
    // — we restore that slot after the call.
    std::vector<float3> saved_original_verts;
    float3 saved_original_pos = {0, 0, 0};
    bool   saved_has_original = false;

    if (is_mesh) {
        if (obj_idx >= (int)objects.size()) return false;
        if (objects[obj_idx].hidden) return false;

        float3 mn, mx;
        compute_world_aabb_mesh(objects[obj_idx].obj_id, all_prims, mn, mx);
        if (mn.x > mx.x) return false;
        float3 center = make_float3(0.5f*(mn.x+mx.x), 0.5f*(mn.y+mx.y), 0.5f*(mn.z+mx.z));

        int target_id = objects[obj_idx].obj_id;

        // Capture originals FIRST (pre-teleport).
        if (!state.grasp.has_original) {
            state.grasp.original_pos = center;
            state.grasp.has_original = true;
        }
        saved_original_verts.reserve(all_prims.size() * 3);
        for (auto& tri : all_prims) {
            if (tri.obj_id != target_id) continue;
            saved_original_verts.push_back(tri.v0);
            saved_original_verts.push_back(tri.v1);
            saved_original_verts.push_back(tri.v2);
        }
        saved_original_pos  = state.grasp.original_pos;
        saved_has_original  = true;

        // Teleport the mesh so the chosen contact point coincides with the
        // gripper anchor. Face-pick uses the clicked object face point;
        // legacy pickup uses the object centroid.
        float3 attach_point = center;
        if (state.pick_face_set && state.pick_face_obj_idx == obj_idx) {
            attach_point = make_float3(state.pick_face_world_pos[0],
                                       state.pick_face_world_pos[1],
                                       state.pick_face_world_pos[2]);
        }
        float3 d = make_float3(ee.x - attach_point.x,
                               ee.y - attach_point.y,
                               ee.z - attach_point.z);

        std::fprintf(stderr,
            "              cube_aabb  = (%.4f..%.4f, %.4f..%.4f, %.4f..%.4f)\n"
            "              cube_center= (%.4f, %.4f, %.4f)  name='%s'\n"
            "              delta d    = (%.4f, %.4f, %.4f)  |d|=%.4f m\n",
            mn.x, mx.x, mn.y, mx.y, mn.z, mx.z,
            center.x, center.y, center.z,
            objects[obj_idx].name,
            d.x, d.y, d.z,
            std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z));

        for (auto& tri : all_prims) {
            if (tri.obj_id != target_id) continue;
            tri.v0.x += d.x; tri.v0.y += d.y; tri.v0.z += d.z;
            tri.v1.x += d.x; tri.v1.y += d.y; tri.v1.z += d.z;
            tri.v2.x += d.x; tri.v2.y += d.y; tri.v2.z += d.z;
        }
        objects[obj_idx].centroid.x += d.x;
        objects[obj_idx].centroid.y += d.y;
        objects[obj_idx].centroid.z += d.z;
    } else {
        if (obj_idx >= (int)spheres.size()) return false;
        if (!state.grasp.has_original) {
            state.grasp.original_pos = spheres[obj_idx].center;
            state.grasp.has_original = true;
        }
        spheres[obj_idx].center = ee;
    }

    // Normal attach flow — finds the now-coincident object and sets
    // offset ≈ 0 + captures local_verts in EE-local frame.
    bool ok = try_attach_nearest(state, handle, spheres, objects, all_prims, 1e30f);
    if (!ok) return false;

    // Restore pre-teleport originals so reset puts the cube back in the
    // scene (not at the grip point where it was teleported).
    if (is_mesh && saved_has_original) {
        state.grasp.original_pos   = saved_original_pos;
        state.grasp.original_verts = std::move(saved_original_verts);
    }
    return true;
}

bool robot_demo_force_detach(RobotDemoState& state)
{
    if (!state.grasp.active || state.grasp.obj_idx < 0) return false;

    FallingObject fo;
    fo.is_mesh  = state.grasp.is_mesh;
    fo.obj_idx  = state.grasp.obj_idx;
    // Inherit the end-effector's velocity at the moment of release so the
    // object "throws" naturally when the arm is moving.
    fo.velocity = state.last_ee_vel;
    fo.stopped  = false;
    state.falling.push_back(fo);

    state.grasp.active = false;
    return true;
}

// ── Pick & Place planner ─────────────────────────────────────────────────────
//
// Strategy:
//   1. Find the pickup mesh object whose centroid is closest to the EE grip
//      point. Skip robot parts, environment, hidden objects, and any mesh not
//      named pickup*/pickups*.
//   2. Compute its world AABB. Try 5 approach directions in order of
//      preference (top first, then ±X, ±Z). For each direction, run pose IK
//      on three critical waypoints (pregrasp, grasp, lift). The first
//      direction where all three converge is selected.
//   3. Place waypoints also need to solve. The place phase always uses the
//      same grasp orientation so the gripper doesn't rotate mid-carry.
//   4. Joint angles are captured at each waypoint by running pose IK, then
//      restored to the original pose before playback starts.
//
// Assumptions:
//   - World is Y-up (matches the existing gravity/ground_y/z_to_y setup).
//   - Hand +Z is the approach axis (matches grip_forward's documented frame).
//   - The place target is a world-space point; the cube is set down so its
//      centroid lands at that point with the same orientation as the grasp.

static void compute_world_aabb_mesh(int obj_id,
                                    const std::vector<Triangle>& all_prims,
                                    float3& out_min, float3& out_max)
{
    out_min = make_float3( 1e30f,  1e30f,  1e30f);
    out_max = make_float3(-1e30f, -1e30f, -1e30f);
    for (const auto& tri : all_prims) {
        if (tri.obj_id != obj_id) continue;
        const float3 v[3] = { tri.v0, tri.v1, tri.v2 };
        for (int i = 0; i < 3; ++i) {
            if (v[i].x < out_min.x) out_min.x = v[i].x;
            if (v[i].y < out_min.y) out_min.y = v[i].y;
            if (v[i].z < out_min.z) out_min.z = v[i].z;
            if (v[i].x > out_max.x) out_max.x = v[i].x;
            if (v[i].y > out_max.y) out_max.y = v[i].y;
            if (v[i].z > out_max.z) out_max.z = v[i].z;
        }
    }
}

static float3 safe_normalize(float3 v, float3 fallback)
{
    float l2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (l2 < 1e-12f) return fallback;
    float inv = 1.f / std::sqrt(l2);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

// Build a row-major 4×4 rotation where the EE axes (cols) are ex, ey, ez
// expressed in world coordinates. Hand +Z is the approach axis.
static void make_ee_rotation(float3 ex, float3 ey, float3 ez, float out[16])
{
    out[ 0] = ex.x; out[ 1] = ey.x; out[ 2] = ez.x; out[ 3] = 0.f;
    out[ 4] = ex.y; out[ 5] = ey.y; out[ 6] = ez.y; out[ 7] = 0.f;
    out[ 8] = ex.z; out[ 9] = ey.z; out[10] = ez.z; out[11] = 0.f;
    out[12] = 0.f;  out[13] = 0.f;  out[14] = 0.f;  out[15] = 1.f;
}

static void make_face_aligned_rotation(const float current_R[9],
                                       float3 grip_normal_local,
                                       float3 target_grip_normal_world,
                                       float out[16])
{
    float3 nL = safe_normalize(grip_normal_local, make_float3(0, 0, 1));
    float3 nW = safe_normalize(target_grip_normal_world, make_float3(0, 0, 1));

    const float3 local_axes[] = {
        make_float3(1, 0, 0),
        make_float3(0, 1, 0),
        make_float3(0, 0, 1),
    };
    float3 uL = local_axes[0];
    float best = 1e30f;
    for (float3 a : local_axes) {
        float d = fabsf(dot(a, nL));
        if (d < best) {
            best = d;
            uL = a;
        }
    }
    uL = safe_normalize(uL - nL * dot(uL, nL), make_float3(1, 0, 0));
    float3 vL = safe_normalize(cross(nL, uL), make_float3(0, 1, 0));

    float3 uW_hint = mat3_mul_vec(current_R, uL);
    float3 uW = safe_normalize(uW_hint - nW * dot(uW_hint, nW), make_float3(0, 1, 0));
    if (fabsf(dot(uW, nW)) > 0.95f)
        uW = safe_normalize(cross(make_float3(0, 1, 0), nW), make_float3(1, 0, 0));
    float3 vW = safe_normalize(cross(nW, uW), make_float3(0, 1, 0));

    // R = [uW vW nW] * transpose([uL vL nL]).
    float colsW[3][3] = {
        { uW.x, uW.y, uW.z },
        { vW.x, vW.y, vW.z },
        { nW.x, nW.y, nW.z },
    };
    float colsL[3][3] = {
        { uL.x, uL.y, uL.z },
        { vL.x, vL.y, vL.z },
        { nL.x, nL.y, nL.z },
    };
    float R[9] = {};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R[r*3 + c] =
                colsW[0][r] * colsL[0][c] +
                colsW[1][r] * colsL[1][c] +
                colsW[2][r] * colsL[2][c];
        }
    }
    out[ 0] = R[0]; out[ 1] = R[1]; out[ 2] = R[2]; out[ 3] = 0.f;
    out[ 4] = R[3]; out[ 5] = R[4]; out[ 6] = R[5]; out[ 7] = 0.f;
    out[ 8] = R[6]; out[ 9] = R[7]; out[10] = R[8]; out[11] = 0.f;
    out[12] = 0.f;  out[13] = 0.f;  out[14] = 0.f;  out[15] = 1.f;
}

// Enumerate approach directions. `approach` points FROM cube TO gripper
// (so grasp_pos = cube_center + approach * offset). For each we pick an
// orthogonal EE frame such that hand +Y is roughly aligned with world +Y
// (keeps the gripper level for side grasps).
struct ApproachDir {
    const char* name;
    float3      approach;    // unit vector from cube to gripper
    float3      ex, ey, ez;  // EE axes in world
};

static const ApproachDir kApproachDirs[] = {
    // Top-down: gripper sits above cube, descends along -Y.
    { "top",
      { 0.f,  1.f,  0.f},
      {-1.f,  0.f,  0.f},
      { 0.f,  0.f, -1.f},
      { 0.f, -1.f,  0.f} },
    // +X side: gripper sits at +X from cube, descends along -X.
    { "+X",
      { 1.f,  0.f,  0.f},
      { 0.f,  0.f, -1.f},
      { 0.f,  1.f,  0.f},
      {-1.f,  0.f,  0.f} },
    // -X side.
    { "-X",
      {-1.f,  0.f,  0.f},
      { 0.f,  0.f,  1.f},
      { 0.f,  1.f,  0.f},
      { 1.f,  0.f,  0.f} },
    // +Z side.
    { "+Z",
      { 0.f,  0.f,  1.f},
      { 1.f,  0.f,  0.f},
      { 0.f,  1.f,  0.f},
      { 0.f,  0.f, -1.f} },
    // -Z side.
    { "-Z",
      { 0.f,  0.f, -1.f},
      {-1.f,  0.f,  0.f},
      { 0.f,  1.f,  0.f},
      { 0.f,  0.f,  1.f} },
};

// Save / restore joint angles so failed planning attempts don't leave the arm
// in a weird intermediate pose.
static void snapshot_joint_angles(UrdfArticulation* h, std::vector<float>& out)
{
    int n = urdf_joint_count(h);
    out.resize((size_t)n);
    UrdfJointInfo* info = urdf_joint_info(h);
    for (int i = 0; i < n; ++i) out[i] = info[i].angle;
}

static void restore_joint_angles(UrdfArticulation* h, const std::vector<float>& in)
{
    int n = urdf_joint_count(h);
    UrdfJointInfo* info = urdf_joint_info(h);
    int limit = std::min(n, (int)in.size());
    for (int i = 0; i < limit; ++i) info[i].angle = in[i];
}

// Half-extent of the object AABB projected onto the approach axis. This tells
// us how far the gripper needs to stand off from the centroid along that
// axis to sit at the surface.
static float half_extent_along(float3 half, float3 axis)
{
    return fabsf(half.x * axis.x) + fabsf(half.y * axis.y) + fabsf(half.z * axis.z);
}

static bool is_pickup_object_name(const char* raw_name)
{
    if (!raw_name || !raw_name[0]) return false;

    const char* leaf = raw_name;
    for (const char* p = raw_name; *p; ++p) {
        if (*p == '/' || *p == '\\') leaf = p + 1;
    }

    char lower[128];
    int n = 0;
    for (; leaf[n] && n < (int)sizeof(lower) - 1; ++n)
        lower[n] = (char)std::tolower((unsigned char)leaf[n]);
    lower[n] = '\0';

    const char* stem = nullptr;
    if (std::strncmp(lower, "pickups", 7) == 0) {
        stem = lower + 7;
    } else if (std::strncmp(lower, "pickup", 6) == 0) {
        stem = lower + 6;
    } else {
        return false;
    }

    // Accept pickup, pickups, pickup001, pickups_001, pickups-001, etc.
    for (const char* p = stem; *p; ++p) {
        unsigned char c = (unsigned char)*p;
        if (std::isdigit(c)) continue;
        if (*p == '_' || *p == '-' || *p == '.' || *p == ' ')
            continue;
        return false;
    }
    return true;
}

// Debug: return the grip anchor + nearest pickup centroid the planner would
// target. Same selection rule as robot_demo_pick_and_place step 1 so the
// overlay line matches what a press of the button would aim at.
bool robot_demo_pick_target_preview(const RobotDemoState& state,
                                    UrdfArticulation* handle,
                                    const std::vector<MeshObject>& objects,
                                    const std::vector<Triangle>& all_prims,
                                    float3& out_from_ee,
                                    float3& out_to_target)
{
    if (!handle || urdf_joint_count(handle) == 0) return false;

    float3 ee = gripper_anchor(handle, state.grip_local);
    if (state.pick_face_set) {
        out_from_ee = ee;
        out_to_target = make_float3(state.pick_face_world_pos[0],
                                    state.pick_face_world_pos[1],
                                    state.pick_face_world_pos[2]);
        return true;
    }

    float  best_d2  = 1e30f;
    int    best_idx = -1;
    float3 best_pos = {0, 0, 0};

    for (int i = 0; i < (int)objects.size(); ++i) {
        if (objects[i].hidden || objects[i].is_robot_part || objects[i].environment)
            continue;
        if (!is_pickup_object_name(objects[i].name))
            continue;
        float3 mn, mx;
        compute_world_aabb_mesh(objects[i].obj_id, all_prims, mn, mx);
        if (mn.x > mx.x) continue;
        float3 center = make_float3(0.5f*(mn.x+mx.x), 0.5f*(mn.y+mx.y), 0.5f*(mn.z+mx.z));
        float dx = center.x - ee.x, dy = center.y - ee.y, dz = center.z - ee.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_d2) { best_d2 = d2; best_idx = i; best_pos = center; }
    }

    if (best_idx < 0) return false;
    out_from_ee   = ee;
    out_to_target = best_pos;
    return true;
}

bool robot_demo_pick_and_place(RobotDemoState& state, UrdfArticulation* handle,
                               const std::vector<Sphere>& spheres,
                               const std::vector<MeshObject>& objects,
                               const std::vector<Triangle>& all_prims)
{
    state.last_pick_status = 0;
    state.last_pick_msg[0] = '\0';

    if (!handle || urdf_joint_count(handle) == 0) {
        state.last_pick_status = -1;
        std::snprintf(state.last_pick_msg, sizeof(state.last_pick_msg),
                      "no URDF loaded");
        return false;
    }
    if (state.grasp.active) {
        state.last_pick_status = -1;
        std::snprintf(state.last_pick_msg, sizeof(state.last_pick_msg),
                      "already holding an object");
        return false;
    }

    // Push ground height into the IK solver so it can repel joints above floor.
    urdf_set_ik_ground_y(handle, state.ground_y);

    // ── 1. Find nearest named pickup mesh ────────────────────────────────────
    float3 ee = gripper_anchor(handle, state.grip_local);

    float  best_d2    = 1e30f;
    int    best_idx   = -1;
    bool   best_mesh  = false;
    float3 best_pos   = {0,0,0};
    float3 best_half  = {0,0,0};
    bool   use_face_pick = false;

    auto consider = [&](float3 pos, float3 half, int idx, bool is_mesh) {
        float3 d = make_float3(pos.x - ee.x, pos.y - ee.y, pos.z - ee.z);
        float d2 = d.x*d.x + d.y*d.y + d.z*d.z;
        if (d2 < best_d2) {
            best_d2 = d2; best_idx = idx; best_mesh = is_mesh;
            best_pos = pos; best_half = half;
        }
    };

    (void)spheres; // Pick & Place only targets named mesh pickups.
    for (int i = 0; i < (int)objects.size(); ++i) {
        if (objects[i].hidden || objects[i].is_robot_part || objects[i].environment)
            continue;
        if (!is_pickup_object_name(objects[i].name))
            continue;
        float3 mn, mx;
        compute_world_aabb_mesh(objects[i].obj_id, all_prims, mn, mx);
        if (mn.x > mx.x) continue;   // no triangles found
        float3 center = make_float3(0.5f*(mn.x+mx.x), 0.5f*(mn.y+mx.y), 0.5f*(mn.z+mx.z));
        float3 half   = make_float3(0.5f*(mx.x-mn.x), 0.5f*(mx.y-mn.y), 0.5f*(mx.z-mn.z));
        consider(center, half, i, true);
    }

    if (state.pick_face_set &&
        state.pick_face_obj_idx >= 0 &&
        state.pick_face_obj_idx < (int)objects.size())
    {
        const MeshObject& obj = objects[state.pick_face_obj_idx];
        if (!obj.hidden && !obj.is_robot_part && !obj.environment) {
            float3 mn, mx;
            compute_world_aabb_mesh(obj.obj_id, all_prims, mn, mx);
            if (mn.x <= mx.x) {
                best_idx = state.pick_face_obj_idx;
                best_mesh = true;
                best_pos = make_float3(state.pick_face_world_pos[0],
                                       state.pick_face_world_pos[1],
                                       state.pick_face_world_pos[2]);
                best_half = make_float3(0.5f*(mx.x-mn.x),
                                        0.5f*(mx.y-mn.y),
                                        0.5f*(mx.z-mn.z));
                use_face_pick = true;
            }
        }
    }

    if (best_idx < 0) {
        state.last_pick_status = -1;
        std::snprintf(state.last_pick_msg, sizeof(state.last_pick_msg),
                      "no pickup* object in scene");
        return false;
    }

    // ── 2. Snapshot current pose (waypoint 0 + restore target) ───────────────
    std::vector<float> saved_angles;
    snapshot_joint_angles(handle, saved_angles);

    const float grip        = 0.f; (void)grip;
    const float standoff    = 0.12f;    // pregrasp offset along approach axis
    const float lift_y      = 0.20f;    // preferred lift above grasp height (world +Y)
    const float margin      = 0.005f;   // tiny padding so fingers don't clip
    // Hard floor for the wrist/EE origin. The IK solver has no collision
    // avoidance, so without this the top-down grasp of a floor cube drives
    // the arm straight down through the ground. Applied to every waypoint
    // (pick + place, every approach direction).
    const float min_ee_y    = state.ground_y + 0.10f;
    (void)grip;

    float ee_mat0[16];
    urdf_fk_ee_transform(handle, ee_mat0);
    float ee_rot0[9], ee_inv_rot0[9];
    mat3_from_mat4(ee_mat0, ee_rot0);
    mat3_transpose(ee_rot0, ee_inv_rot0);
    float3 anchor0 = gripper_anchor(handle, state.grip_local);
    float3 ee0     = make_float3(ee_mat0[3], ee_mat0[7], ee_mat0[11]);
    float3 anchor_local = mat3_mul_vec(ee_inv_rot0,
        make_float3(anchor0.x - ee0.x, anchor0.y - ee0.y, anchor0.z - ee0.z));

    // ── 3. Try each approach direction until all critical poses solve ────────
    // Two passes: first with full pose IK (position + orientation), then
    // position-only fallback. The fallback covers rigs whose tool-frame
    // convention doesn't match the "hand +Z is approach" assumption — the
    // arm still reaches each waypoint position; the attach system uses
    // whatever EE rotation the IK lands in.
    const float tol = 0.02f;   // 2 cm — plenty for grasping, avoids false negatives
    auto fk_pos_error = [&](float3 target) -> float {
        float3 p = urdf_fk_ee_pos(handle);
        float dx = p.x - target.x;
        float dy = p.y - target.y;
        float dz = p.z - target.z;
        return sqrtf(dx*dx + dy*dy + dz*dz);
    };
    // Set every revolute joint to the midpoint of its limits — gives the IK a
    // folded starting configuration that avoids the "arm fully extended" local
    // minimum that often causes DLS to converge to a straight-arm pose.
    auto set_neutral = [&]() {
        int n = urdf_joint_count(handle);
        UrdfJointInfo* info = urdf_joint_info(handle);
        for (int i = 0; i < n; ++i) {
            if (info[i].type == 0 && info[i].upper > info[i].lower)
                info[i].angle = 0.5f * (info[i].lower + info[i].upper);
        }
    };
    auto try_pose = [&](float3 pos, const float R[16]) -> bool {
        restore_joint_angles(handle, saved_angles);
        if (urdf_solve_ik_pose(handle, pos, R, 60, tol) && fk_pos_error(pos) <= tol)
            return true;
        set_neutral();
        return urdf_solve_ik_pose(handle, pos, R, 60, tol) && fk_pos_error(pos) <= tol;
    };
    auto try_pos_only = [&](float3 pos) -> bool {
        restore_joint_angles(handle, saved_angles);
        if (urdf_solve_ik(handle, pos, 60, tol) && fk_pos_error(pos) <= tol)
            return true;
        set_neutral();
        return urdf_solve_ik(handle, pos, 60, tol) && fk_pos_error(pos) <= tol;
    };

    // Auto drop zones — offsets from the cube in world XZ, keeping Y on the
    // ground.
    const float drop_dist = 0.30f;
    const float3 drop_offsets[] = {
        { drop_dist,  0.f,  0.f     },
        {-drop_dist,  0.f,  0.f     },
        { 0.f,        0.f,  drop_dist},
        { 0.f,        0.f, -drop_dist},
    };
    static const char* kDropNames[] = { "+X", "-X", "+Z", "-Z" };
    const float lift_candidates[] = { lift_y, 0.14f, 0.10f, 0.07f, 0.04f };

    int dir_chosen = -1;
    int drop_chosen = -1;
    bool used_orient = false;
    bool used_retract_fallback = false;
    bool used_lift_fallback = false;
    bool used_place_fallback = false;
    bool used_face_alignment = use_face_pick && state.grip_face_set;
    bool used_debug_unreachable_place = false;
    char chosen_approach_name[32] = {};
    float R_grasp[16];
    float3 grasp_pos, pregrasp_pos, lift_pos;
    float3 place_pregrasp_pos, place_grasp_pos, place_retract_pos;

    bool debug_place_valid = false;
    bool debug_place_orient = false;
    bool debug_lift_fallback = false;
    int  debug_place_dir = -1;
    int  debug_place_drop = -1;
    char debug_place_name[32] = {};
    float debug_R[16] = {};
    float3 debug_pregrasp_pos = {};
    float3 debug_grasp_pos = {};
    float3 debug_lift_pos = {};
    float3 debug_place_pregrasp_pos = {};

    // Track the farthest stage reached across all attempts so the failure
    // message tells the user whether it was the pick or the place that
    // stalled.
    const char* kStages[] = { "pregrasp", "grasp", "lift",
                              "place-pregrasp", "place", "retract" };
    int best_stage_reached = -1;

    // Direction enumeration honors the user's approach-mode selection.
    // 0 = Auto (top first, then sides); 1 = Top only; 2 = Side only.
    // kApproachDirs layout: [0]=top, [1..4]=+X,-X,+Z,-Z.
    int  dir_order[5];
    int  dir_count = 0;
    if (used_face_alignment) {
        dir_order[0] = -1;
        dir_count = 1;
    } else {
        switch (state.pick_approach_mode) {
            case 1:  // Top only — wrist perpendicular to ground
                dir_order[0] = 0;
                dir_count    = 1;
                break;
            case 2:  // Side only — wrist parallel to ground
                dir_order[0] = 1; dir_order[1] = 2; dir_order[2] = 3; dir_order[3] = 4;
                dir_count    = 4;
                break;
            default: // Auto
                dir_order[0] = 0; dir_order[1] = 1; dir_order[2] = 2;
                dir_order[3] = 3; dir_order[4] = 4;
                dir_count    = 5;
                break;
        }
    }

    for (int pass = 0; pass < 2 && dir_chosen < 0; ++pass) {
        const bool with_orientation = (pass == 0);

        for (int oi = 0; oi < dir_count; ++oi) {
            int d = dir_order[oi];
            float R_try[16];
            float3 approach = {0, 1, 0};
            const char* try_name = "face";
            if (used_face_alignment) {
                approach = safe_normalize(
                    make_float3(state.pick_face_world_normal[0],
                                state.pick_face_world_normal[1],
                                state.pick_face_world_normal[2]),
                    make_float3(0, 1, 0));
                float3 grip_n_local = make_float3(state.grip_face_normal_local[0],
                                                  state.grip_face_normal_local[1],
                                                  state.grip_face_normal_local[2]);
                make_face_aligned_rotation(
                    ee_rot0, grip_n_local,
                    make_float3(-approach.x, -approach.y, -approach.z),
                    R_try);
            } else {
                const ApproachDir& A = kApproachDirs[d];
                approach = A.approach;
                try_name = A.name;
                make_ee_rotation(A.ex, A.ey, A.ez, R_try);
            }

            auto solve = [&](float3 pos) -> bool {
                return with_orientation ? try_pose(pos, R_try) : try_pos_only(pos);
            };
            auto ee_target_for_anchor = [&](float3 anchor_target) -> float3 {
                float R3[9];
                mat3_from_mat4(R_try, R3);
                float3 off = mat3_mul_vec(R3, anchor_local);
                return make_float3(anchor_target.x - off.x,
                                   anchor_target.y - off.y,
                                   anchor_target.z - off.z);
            };

            // Target the gripper contact anchor at the cube center, converted
            // into the wrist/hand EE-origin target needed for this orientation.
            //   approach is unit vector FROM cube TO gripper, so with
            //   distance 0 the grip anchor sits exactly at cube_center. Use
            //   `margin` as a tiny standoff so the IK target isn't literally
            //   inside the cube, which can make solvers wobble.
            float he       = half_extent_along(best_half, approach);
            (void)he;
            float3 g_anchor  = make_float3(best_pos.x + approach.x * margin,
                                           best_pos.y + approach.y * margin,
                                           best_pos.z + approach.z * margin);
            float3 pg_anchor = make_float3(g_anchor.x + approach.x * standoff,
                                           g_anchor.y + approach.y * standoff,
                                           g_anchor.z + approach.z * standoff);
            float3 g_pos     = ee_target_for_anchor(g_anchor);
            float3 pg_pos    = ee_target_for_anchor(pg_anchor);

            // Ground clearance — the IK solver has no collision avoidance,
            // so a top-down pick of a low cube will straighten the arm
            // through the floor. Clamp every waypoint's wrist target; the
            // cube is snap-teleported to the EE origin at attach time, so
            // the grip doesn't need to be exactly at the cube's center.
            if (g_pos.y  < min_ee_y) g_pos.y  = min_ee_y;
            if (pg_pos.y < min_ee_y) pg_pos.y = min_ee_y;

            int stage = -1;
            if (!solve(pg_pos)) { if (stage > best_stage_reached) best_stage_reached = stage; continue; }
            stage = 0;  // reached pregrasp
            if (!solve(g_pos))  { if (stage > best_stage_reached) best_stage_reached = stage; continue; }
            stage = 1;  // reached grasp

            float3 lf_pos = make_float3(0, 0, 0);
            float chosen_lift_y = 0.f;
            bool lift_ok = false;
            for (float ly : lift_candidates) {
                float3 lf_anchor = make_float3(g_anchor.x, g_anchor.y + ly, g_anchor.z);
                float3 candidate = ee_target_for_anchor(lf_anchor);
                if (candidate.y < min_ee_y) candidate.y = min_ee_y;
                if (solve(candidate)) {
                    lf_pos = candidate;
                    chosen_lift_y = ly;
                    lift_ok = true;
                    break;
                }
            }
            if (!lift_ok) { if (stage > best_stage_reached) best_stage_reached = stage; continue; }
            stage = 2;  // reached lift

            float cube_half_y = best_half.y;
            float ground_y    = state.ground_y;

            for (int dr = 0; dr < (int)(sizeof(drop_offsets)/sizeof(drop_offsets[0])); ++dr) {
                float3 drop_center = make_float3(
                    best_pos.x + drop_offsets[dr].x,
                    ground_y   + cube_half_y,
                    best_pos.z + drop_offsets[dr].z);

                // Target grip anchor at drop_center (same convention as pick).
                float3 p_anchor_base = make_float3(drop_center.x + approach.x * margin,
                                                   drop_center.y + approach.y * margin,
                                                   drop_center.z + approach.z * margin);
                float3 p_pre_a  = make_float3(p_anchor_base.x + approach.x * standoff,
                                             p_anchor_base.y + approach.y * standoff
                                                 + (approach.y < 0.5f ? chosen_lift_y : 0.f),
                                             p_anchor_base.z + approach.z * standoff);
                float3 p_pre   = ee_target_for_anchor(p_pre_a);

                // Same floor-clearance clamp for place waypoints.
                if (p_pre.y < min_ee_y) p_pre.y = min_ee_y;

                int ds = 2;
                if (!solve(p_pre))  {
                    if (!debug_place_valid) {
                        debug_place_valid = true;
                        debug_place_orient = with_orientation;
                        debug_lift_fallback = chosen_lift_y < lift_y - 1e-5f;
                        debug_place_dir = d;
                        debug_place_drop = dr;
                        std::snprintf(debug_place_name, sizeof(debug_place_name),
                                      "%s", try_name);
                        for (int k = 0; k < 16; ++k) debug_R[k] = R_try[k];
                        debug_pregrasp_pos = pg_pos;
                        debug_grasp_pos = g_pos;
                        debug_lift_pos = lf_pos;
                        debug_place_pregrasp_pos = p_pre;
                    }
                    if (ds > best_stage_reached) best_stage_reached = ds;
                    continue;
                }
                ds = 3;

                float3 p_grasp = make_float3(0, 0, 0);
                float chosen_place_raise = 0.f;
                bool place_ok = false;
                const float place_raise_candidates[] = { 0.f, 0.03f, 0.06f, 0.10f, 0.14f };
                for (float raise : place_raise_candidates) {
                    float3 p_anchor = make_float3(p_anchor_base.x,
                                                  p_anchor_base.y + raise,
                                                  p_anchor_base.z);
                    float3 candidate = ee_target_for_anchor(p_anchor);
                    if (candidate.y < min_ee_y) candidate.y = min_ee_y;
                    if (solve(candidate)) {
                        p_grasp = candidate;
                        chosen_place_raise = raise;
                        place_ok = true;
                        break;
                    }
                }
                if (!place_ok) {
                    // The arm can carry the object to the pre-place pose but
                    // cannot descend to the requested release pose. Complete
                    // the task by releasing at pre-place instead of rejecting
                    // the whole trajectory.
                    p_grasp = p_pre;
                    chosen_place_raise = chosen_lift_y;
                    place_ok = true;
                }

                ds = 4;
                float3 p_ret_a  = make_float3(p_anchor_base.x,
                                               p_anchor_base.y + chosen_lift_y + chosen_place_raise,
                                               p_anchor_base.z);
                float3 p_ret   = ee_target_for_anchor(p_ret_a);
                if (p_ret.y < min_ee_y) p_ret.y = min_ee_y;
                bool retract_ok = solve(p_ret);
                if (!retract_ok) {
                    // Retraction is cosmetic after release. Do not reject an
                    // otherwise valid pick/place just because this final pose
                    // is outside the small arm's comfortable workspace.
                    p_ret = p_pre;
                }
                ds = 5;

                dir_chosen      = d;
                drop_chosen     = dr;
                used_orient     = with_orientation;
                std::snprintf(chosen_approach_name, sizeof(chosen_approach_name),
                              "%s", try_name);
                used_retract_fallback = !retract_ok;
                used_lift_fallback = chosen_lift_y < lift_y - 1e-5f;
                used_place_fallback = chosen_place_raise > 1e-5f;
                for (int k = 0; k < 16; ++k) R_grasp[k] = R_try[k];
                grasp_pos          = g_pos;
                pregrasp_pos       = pg_pos;
                lift_pos           = lf_pos;
                place_pregrasp_pos = p_pre;
                place_grasp_pos    = p_grasp;
                place_retract_pos  = p_ret;
                break;
            }
            if (dir_chosen >= 0) break;
        }
    }

    // Always restore before we either build waypoints or bail out.
    restore_joint_angles(handle, saved_angles);

    if (dir_chosen < 0 && debug_place_valid) {
        dir_chosen = debug_place_dir;
        drop_chosen = debug_place_drop;
        used_orient = debug_place_orient;
        used_debug_unreachable_place = true;
        used_lift_fallback = debug_lift_fallback;
        used_place_fallback = true;
        used_retract_fallback = true;
        std::snprintf(chosen_approach_name, sizeof(chosen_approach_name),
                      "%s", debug_place_name);
        for (int k = 0; k < 16; ++k) R_grasp[k] = debug_R[k];
        pregrasp_pos = debug_pregrasp_pos;
        grasp_pos = debug_grasp_pos;
        lift_pos = debug_lift_pos;
        place_pregrasp_pos = debug_place_pregrasp_pos;
        place_grasp_pos = debug_place_pregrasp_pos;
        place_retract_pos = debug_place_pregrasp_pos;
    }

    if (dir_chosen < 0) {
        state.last_pick_status = -2;
        const char* stalled = (best_stage_reached < 0)
            ? "pregrasp"
            : kStages[best_stage_reached + 1];
        std::snprintf(state.last_pick_msg, sizeof(state.last_pick_msg),
                      "unreachable — stalled at %s", stalled);
        return false;
    }

    // ── 4. Capture joint angles at each waypoint ─────────────────────────────
    const int n_joints = urdf_joint_count(handle);
    auto solve_current = [&](float3 pos) -> bool {
        bool ok = used_orient
            ? urdf_solve_ik_pose(handle, pos, R_grasp, 60, tol)
            : urdf_solve_ik(handle, pos, 60, tol);
        return ok && fk_pos_error(pos) <= tol;
    };
    auto capture = [&](float3 pos, std::vector<float>& q) -> bool {
        if (!solve_current(pos)) {
            // Fallback keeps the stricter reachability behavior, but the
            // normal path is sequential so adjacent waypoints stay on the
            // same IK branch and wrist joints keep moving smoothly.
            restore_joint_angles(handle, saved_angles);
            if (!solve_current(pos)) return false;
        }
        q.resize((size_t)n_joints);
        UrdfJointInfo* info = urdf_joint_info(handle);
        for (int i = 0; i < n_joints; ++i) q[i] = info[i].angle;
        return true;
    };
    auto capture_best_effort = [&](float3 pos, std::vector<float>& q) -> bool {
        if (used_orient)
            urdf_solve_ik_pose(handle, pos, R_grasp, 90, tol);
        else
            urdf_solve_ik(handle, pos, 90, tol);
        q.resize((size_t)n_joints);
        UrdfJointInfo* info = urdf_joint_info(handle);
        for (int i = 0; i < n_joints; ++i) q[i] = info[i].angle;
        return true;
    };

    restore_joint_angles(handle, saved_angles);
    std::vector<float> q_pregrasp;
    std::vector<float> q_grasp;
    std::vector<float> q_lift;
    std::vector<float> q_place_pre;
    std::vector<float> q_place_grasp;
    std::vector<float> q_place_retract;
    bool capture_ok =
        capture(pregrasp_pos, q_pregrasp) &&
        capture(grasp_pos, q_grasp) &&
        capture(lift_pos, q_lift);
    if (capture_ok && used_debug_unreachable_place) {
        capture_ok =
            capture_best_effort(place_pregrasp_pos, q_place_pre) &&
            capture_best_effort(place_grasp_pos, q_place_grasp) &&
            capture_best_effort(place_retract_pos, q_place_retract);
    } else if (capture_ok) {
        capture_ok =
            capture(place_pregrasp_pos, q_place_pre) &&
            capture(place_grasp_pos, q_place_grasp) &&
            capture(place_retract_pos, q_place_retract);
    }
    if (!capture_ok)
    {
        restore_joint_angles(handle, saved_angles);
        state.last_pick_status = -2;
        std::snprintf(state.last_pick_msg, sizeof(state.last_pick_msg),
                      "IK capture drifted from planned pose");
        return false;
    }

    // Restore BEFORE setting waypoints so the arm starts at the current pose
    // and playback interpolates forward from there.
    restore_joint_angles(handle, saved_angles);

    // ── 5. Build the waypoint trajectory ─────────────────────────────────────
    std::vector<RobotWaypoint> wps;
    auto push = [&](float t, const std::vector<float>& q, bool closed, bool attached) {
        RobotWaypoint w;
        w.time            = t;
        w.angles          = q;
        w.gripper_closed  = closed;
        w.attached        = attached;
        wps.push_back(w);
    };

    push(0.0f,  saved_angles,      false, false);
    push(1.5f,  q_pregrasp,        false, false);
    push(3.0f,  q_grasp,           false, false);
    push(3.6f,  q_grasp,           true,  true);    // close + attach
    push(5.0f,  q_lift,            true,  true);
    push(6.5f,  q_place_pre,       true,  true);
    push(8.0f,  q_place_grasp,     true,  true);
    push(8.6f,  q_place_grasp,     false, false);   // open + release
    push(10.0f, q_place_retract,   false, false);

    state.waypoints     = std::move(wps);
    state.playing       = true;
    state.recording     = false;
    state.playback_time = 0.f;
    state.playback_speed = 1.0f;
    state.needs_grasp_reset = false;

    // Tell the attach keyframe handler to snap THIS specific object to the
    // EE origin when the attach waypoint fires (instead of falling back to
    // nearest-search, which can drift a few cm if IK landed short or used
    // position-only fallback with arbitrary EE orientation).
    state.planned_pick_valid   = true;
    state.planned_pick_obj_idx = best_idx;
    state.planned_pick_is_mesh = best_mesh;

    state.last_pick_status = 1;
    char mode_msg[64];
    std::snprintf(mode_msg, sizeof(mode_msg), "%s%s%s%s%s",
                  used_orient ? "pose IK" : "position-only",
                  used_debug_unreachable_place ? ", debug unreachable place" : "",
                  used_lift_fallback ? ", low lift" : "",
                  used_place_fallback ? ", high release" : "",
                  used_retract_fallback ? ", fallback retract" : "");
    std::snprintf(state.last_pick_msg, sizeof(state.last_pick_msg),
                  "picking %s via %s, dropping %s (%s)",
                  best_mesh && best_idx < (int)objects.size()
                      ? objects[best_idx].name : (best_mesh ? "mesh" : "prop"),
                  chosen_approach_name[0] ? chosen_approach_name :
                      (dir_chosen >= 0 ? kApproachDirs[dir_chosen].name : "face"),
                  kDropNames[drop_chosen],
                  mode_msg);
    return true;
}

// Translate a mesh object by (dx, dy, dz): every triangle vertex + centroid.
static void translate_mesh(int obj_idx,
                           std::vector<MeshObject>& objects,
                           std::vector<Triangle>& all_prims,
                           float dx, float dy, float dz)
{
    if (obj_idx < 0 || obj_idx >= (int)objects.size()) return;
    int target_id = objects[obj_idx].obj_id;
    for (auto& tri : all_prims) {
        if (tri.obj_id != target_id) continue;
        tri.v0.x += dx; tri.v0.y += dy; tri.v0.z += dz;
        tri.v1.x += dx; tri.v1.y += dy; tri.v1.z += dz;
        tri.v2.x += dx; tri.v2.y += dy; tri.v2.z += dz;
    }
    objects[obj_idx].centroid.x += dx;
    objects[obj_idx].centroid.y += dy;
    objects[obj_idx].centroid.z += dz;
}

// Return the minimum world-y over the object's triangle vertices.
static float mesh_min_y(int obj_idx,
                        const std::vector<MeshObject>& objects,
                        const std::vector<Triangle>& all_prims)
{
    if (obj_idx < 0 || obj_idx >= (int)objects.size()) return 0.f;
    int target_id = objects[obj_idx].obj_id;
    float min_y = 1e30f;
    for (const auto& tri : all_prims) {
        if (tri.obj_id != target_id) continue;
        if (tri.v0.y < min_y) min_y = tri.v0.y;
        if (tri.v1.y < min_y) min_y = tri.v1.y;
        if (tri.v2.y < min_y) min_y = tri.v2.y;
    }
    return (min_y < 1e29f) ? min_y : 0.f;
}

// Rotate a vector by a unit quaternion (w, x, y, z).
// Uses the cross-product form: v' = v + 2*q.w*(q.xyz × v) + 2*(q.xyz × (q.xyz × v)).
static float3 quat_rotate(const float q[4], float3 v)
{
    float3 u = make_float3(q[1], q[2], q[3]);
    float  w = q[0];
    float3 uxv = make_float3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x);
    float3 uxuxv = make_float3(
        u.y * uxv.z - u.z * uxv.y,
        u.z * uxv.x - u.x * uxv.z,
        u.x * uxv.y - u.y * uxv.x);
    return make_float3(
        v.x + 2.f * (w * uxv.x + uxuxv.x),
        v.y + 2.f * (w * uxv.y + uxuxv.y),
        v.z + 2.f * (w * uxv.z + uxuxv.z));
}

// Compute the AABB of a mesh object's triangles.
static void mesh_aabb(int obj_idx,
                      const std::vector<MeshObject>& objects,
                      const std::vector<Triangle>& all_prims,
                      float3& out_min, float3& out_max)
{
    out_min = make_float3( 1e30f,  1e30f,  1e30f);
    out_max = make_float3(-1e30f, -1e30f, -1e30f);
    if (obj_idx < 0 || obj_idx >= (int)objects.size()) return;
    int target_id = objects[obj_idx].obj_id;
    for (const auto& tri : all_prims) {
        if (tri.obj_id != target_id) continue;
        const float3 vs[3] = { tri.v0, tri.v1, tri.v2 };
        for (const auto& v : vs) {
            if (v.x < out_min.x) out_min.x = v.x;
            if (v.y < out_min.y) out_min.y = v.y;
            if (v.z < out_min.z) out_min.z = v.z;
            if (v.x > out_max.x) out_max.x = v.x;
            if (v.y > out_max.y) out_max.y = v.y;
            if (v.z > out_max.z) out_max.z = v.z;
        }
    }
}

// Register a freshly-detached FallingObject with the MuJoCo auxiliary
// world. Captures body-local vertex offsets so we can rigidly transform
// the mesh as MuJoCo integrates the body's pose.
static void register_prop(RobotDemoState& state, FallingObject& fo,
                          std::vector<Sphere>& spheres,
                          std::vector<MeshObject>& objects,
                          std::vector<Triangle>& all_prims)
{
    if (!state.props_world || !props_physics_available()) return;
    if (fo.mujoco_handle >= 0) return;  // already registered

    props_physics_set_ground(state.props_world, state.ground_y);

    if (!fo.is_mesh) {
        if (fo.obj_idx < 0 || fo.obj_idx >= (int)spheres.size()) return;
        const Sphere& s = spheres[fo.obj_idx];
        float vol = (4.f / 3.f) * 3.14159265f * s.radius * s.radius * s.radius;
        float mass = std::max(0.01f, vol * state.prop_density);
        fo.body_centroid = s.center;
        fo.mujoco_handle = props_physics_add_sphere(
            state.props_world,
            s.center.x, s.center.y, s.center.z,
            s.radius,
            fo.velocity.x, fo.velocity.y, fo.velocity.z,
            mass);
    } else {
        if (fo.obj_idx < 0 || fo.obj_idx >= (int)objects.size()) return;
        float3 aabb_min, aabb_max;
        mesh_aabb(fo.obj_idx, objects, all_prims, aabb_min, aabb_max);
        float sx = std::max(1e-4f, aabb_max.x - aabb_min.x);
        float sy = std::max(1e-4f, aabb_max.y - aabb_min.y);
        float sz = std::max(1e-4f, aabb_max.z - aabb_min.z);
        // Centroid = AABB center. The body's world frame origin sits here;
        // per-vertex offsets are stored relative to it so they ride along
        // with MuJoCo's rigid transform.
        float3 centroid = make_float3(
            0.5f * (aabb_min.x + aabb_max.x),
            0.5f * (aabb_min.y + aabb_max.y),
            0.5f * (aabb_min.z + aabb_max.z));
        fo.body_centroid = centroid;

        int target_id = objects[fo.obj_idx].obj_id;
        fo.body_local_verts.clear();
        for (const auto& tri : all_prims) {
            if (tri.obj_id != target_id) continue;
            const float3 vs[3] = { tri.v0, tri.v1, tri.v2 };
            for (const auto& v : vs) {
                fo.body_local_verts.push_back(make_float3(
                    v.x - centroid.x, v.y - centroid.y, v.z - centroid.z));
            }
        }

        float mass = std::max(0.01f, sx * sy * sz * state.prop_density);
        fo.mujoco_handle = props_physics_add_box(
            state.props_world,
            centroid.x, centroid.y, centroid.z,
            1.f, 0.f, 0.f, 0.f,       // initial quat = identity
            sx, sy, sz,
            fo.velocity.x, fo.velocity.y, fo.velocity.z,
            0.f, 0.f, 0.f,             // no initial angular velocity
            mass);
    }
}

// Apply MuJoCo's integrated pose to the corresponding mesh/sphere.
static bool apply_mujoco_pose(RobotDemoState& state, FallingObject& fo,
                              std::vector<Sphere>& spheres,
                              std::vector<MeshObject>& objects,
                              std::vector<Triangle>& all_prims)
{
    if (!state.props_world || fo.mujoco_handle < 0) return false;
    float pos[3], quat[4];
    if (!props_physics_get_pose(state.props_world, fo.mujoco_handle, pos, quat))
        return false;

    if (!fo.is_mesh) {
        if (fo.obj_idx < 0 || fo.obj_idx >= (int)spheres.size()) return false;
        spheres[fo.obj_idx].center = make_float3(pos[0], pos[1], pos[2]);
        return true;
    }

    if (fo.obj_idx < 0 || fo.obj_idx >= (int)objects.size()) return false;
    int target_id = objects[fo.obj_idx].obj_id;
    int vi = 0;
    float3 centroid_sum = make_float3(0, 0, 0);
    int vert_count = 0;

    for (auto& tri : all_prims) {
        if (tri.obj_id != target_id) continue;
        float3* tv[3] = { &tri.v0, &tri.v1, &tri.v2 };
        for (int k = 0; k < 3; ++k) {
            if (vi < (int)fo.body_local_verts.size()) {
                float3 rot = quat_rotate(quat, fo.body_local_verts[vi]);
                tv[k]->x = pos[0] + rot.x;
                tv[k]->y = pos[1] + rot.y;
                tv[k]->z = pos[2] + rot.z;
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
        objects[fo.obj_idx].centroid = make_float3(
            centroid_sum.x * inv, centroid_sum.y * inv, centroid_sum.z * inv);
    }
    return true;
}

bool robot_demo_update_falling(RobotDemoState& state,
                               PropsPhysics* props_world, float dt,
                               std::vector<Sphere>& spheres,
                               std::vector<MeshObject>& objects,
                               std::vector<Triangle>& all_prims)
{
    // The first call that sees a non-null props_world latches it onto
    // state so helpers (e.g. try_attach_nearest) can remove bodies on
    // re-attach without being passed the pointer.
    if (props_world && state.props_world != props_world)
        state.props_world = props_world;

    if (state.falling.empty() || dt <= 0.f) return false;

    const bool use_mujoco = (props_world != nullptr && props_physics_available());

    // ── MuJoCo path ──────────────────────────────────────────────────────
    if (use_mujoco) {
        // Register any newly-queued falling objects (handle == -1). This
        // also seeds their initial velocity from the EE throw.
        for (auto& fo : state.falling) {
            if (fo.mujoco_handle < 0)
                register_prop(state, fo, spheres, objects, all_prims);
        }

        props_physics_set_ground(props_world, state.ground_y);
        props_physics_step(props_world, dt);

        bool moved = false;
        for (auto& fo : state.falling) {
            if (apply_mujoco_pose(state, fo, spheres, objects, all_prims))
                moved = true;
        }
        return moved;
    }

    // ── Fallback integrator (no MuJoCo) ──────────────────────────────────
    // Semi-implicit Euler with restitution + friction. Translation only.
    const float kSettleV = 0.30f;
    bool moved = false;
    for (auto& fo : state.falling) {
        if (fo.stopped) continue;

        fo.velocity.y += state.gravity * dt;
        float dx = fo.velocity.x * dt;
        float dy = fo.velocity.y * dt;
        float dz = fo.velocity.z * dt;

        if (!fo.is_mesh) {
            if (fo.obj_idx < 0 || fo.obj_idx >= (int)spheres.size()) { fo.stopped = true; continue; }
            Sphere& s = spheres[fo.obj_idx];
            float rest_y = state.ground_y + s.radius;
            float new_cx = s.center.x + dx, new_cy = s.center.y + dy, new_cz = s.center.z + dz;
            if (new_cy <= rest_y && fo.velocity.y < 0.f) {
                s.center.x = new_cx; s.center.y = rest_y; s.center.z = new_cz;
                float bv = -fo.velocity.y * state.restitution;
                if (bv < kSettleV) { fo.velocity = make_float3(0,0,0); fo.stopped = true; }
                else { fo.velocity.y = bv; float k = 1.f - state.friction; fo.velocity.x *= k; fo.velocity.z *= k; }
            } else {
                s.center.x = new_cx; s.center.y = new_cy; s.center.z = new_cz;
            }
            moved = true;
        } else {
            if (fo.obj_idx < 0 || fo.obj_idx >= (int)objects.size()) { fo.stopped = true; continue; }
            float min_y = mesh_min_y(fo.obj_idx, objects, all_prims);
            float new_min_y = min_y + dy;
            if (new_min_y <= state.ground_y && fo.velocity.y < 0.f) {
                float adj = state.ground_y - min_y;
                translate_mesh(fo.obj_idx, objects, all_prims, dx, adj, dz);
                float bv = -fo.velocity.y * state.restitution;
                if (bv < kSettleV) { fo.velocity = make_float3(0,0,0); fo.stopped = true; }
                else { fo.velocity.y = bv; float k = 1.f - state.friction; fo.velocity.x *= k; fo.velocity.z *= k; }
            } else {
                translate_mesh(fo.obj_idx, objects, all_prims, dx, dy, dz);
            }
            moved = true;
        }
    }
    auto it = std::remove_if(state.falling.begin(), state.falling.end(),
        [](const FallingObject& f) { return f.stopped; });
    state.falling.erase(it, state.falling.end());
    return moved;
}

void robot_demo_track_ee_velocity(RobotDemoState& state,
                                  UrdfArticulation* handle, float dt)
{
    if (!handle || dt <= 0.f) return;
    float3 ee = gripper_anchor(handle, state.grip_local);
    if (state.prev_ee_valid) {
        state.last_ee_vel = make_float3(
            (ee.x - state.prev_ee_pos.x) / dt,
            (ee.y - state.prev_ee_pos.y) / dt,
            (ee.z - state.prev_ee_pos.z) / dt);
    }
    state.prev_ee_pos   = ee;
    state.prev_ee_valid = true;
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
        w["attached"] = wp.attached;
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
        wp.attached = w.value("attached", false);
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
            wp.attached = state.grasp.active;
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

            // Gripper + attach indicators. [HOLD] = object attached at this
            // waypoint; drives force_attach/detach during playback.
            ImGui::Text(wp.gripper_closed ? "[GRIP]" : "[OPEN]");
            ImGui::SameLine();
            if (wp.attached) {
                ImGui::TextColored(ImVec4(0.3f, 1.f, 0.3f, 1.f), "[HOLD]");
                ImGui::SameLine();
            }

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
                wp.attached = state.grasp.active;
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
                    state.prev_attached = false;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                state.playing = false;
                state.playback_time = state.waypoints.front().time;
                state.needs_grasp_reset = true;
                state.prev_gripper_closed = false;
                state.prev_attached = false;
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

                    // Mirror the play-tick attach/detach logic so scrubbing
                    // across an `attached` transition fires the same attach
                    // request as Play does. Without this, dragging into a
                    // grasped region leaves the object behind.
                    bool any_keyed_attach = false;
                    for (const auto& w : state.waypoints) {
                        if (w.attached) { any_keyed_attach = true; break; }
                    }
                    bool want_attached = any_keyed_attach
                        ? sample_attached(state.waypoints, state.playback_time)
                        : grip;
                    if (want_attached != state.prev_attached) {
                        if (want_attached) state.request_attach = true;
                        else               state.request_detach = true;
                        state.prev_attached = want_attached;
                    }
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
