#include "sensor_panel.h"
#include <imgui.h>
#include <cmath>

// ── GPU render-target management ─────────────────────────────────────────────
// Sensor output is a standalone CUDA array (not Vulkan-interop — it's never
// presented directly, only blitted into the main surface). Reallocated when
// the configured resolution changes.
static void sensor_ensure_target(SensorState& s)
{
    if (s.d_array && s.alloc_w == s.width && s.alloc_h == s.height) return;

    if (s.surface)  { cudaDestroySurfaceObject(s.surface); s.surface = 0; }
    if (s.d_array)  { cudaFreeArray(s.d_array);            s.d_array = nullptr; }

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&s.d_array, &fmt, s.width, s.height, cudaArraySurfaceLoadStore);

    cudaResourceDesc rd{};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = s.d_array;
    cudaCreateSurfaceObject(&s.surface, &rd);

    s.alloc_w = s.width;
    s.alloc_h = s.height;
}

void sensor_state_destroy(SensorState& state)
{
    if (state.surface) { cudaDestroySurfaceObject(state.surface); state.surface = 0; }
    if (state.d_array) { cudaFreeArray(state.d_array);            state.d_array = nullptr; }
    rasterizer_destroy(state.raster);
    state.alloc_w = state.alloc_h = 0;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
static inline float3 vsub(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline float3 vadd(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline float3 vmul(float3 a, float s)  { return make_float3(a.x*s, a.y*s, a.z*s); }
static inline float3 vcross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x);
}
static inline float vdot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float3 vnorm(float3 v) {
    float l = sqrtf(vdot(v, v));
    return (l > 1e-7f) ? make_float3(v.x/l, v.y/l, v.z/l) : v;
}

static float3 xform_origin(const float m[16]) { return make_float3(m[3], m[7], m[11]); }

// Rotate v around unit axis a by angle (radians). Rodrigues.
static float3 rot_axis(float3 v, float3 a, float ang)
{
    float c = cosf(ang), s = sinf(ang);
    float d = vdot(a, v);
    float3 cr = vcross(a, v);
    return make_float3(v.x*c + cr.x*s + a.x*d*(1.f-c),
                       v.y*c + cr.y*s + a.y*d*(1.f-c),
                       v.z*c + cr.z*s + a.z*d*(1.f-c));
}

// World origin of the chosen attachment (EE or joint).
static float3 attach_origin(const SensorState& s, UrdfArticulation* h)
{
    float m[16];
    if (s.attach_joint < 0) urdf_fk_ee_transform(h, m);
    else                    urdf_joint_world_transform(h, s.attach_joint, m);
    return xform_origin(m);
}

// World origin of the chain-parent of the attachment, for deriving a
// geometric "forward" (points from parent toward this joint, i.e. outward
// along the arm). Falls back to the base at (0,0,0) when no parent exists.
static float3 parent_origin(const SensorState& s, UrdfArticulation* h)
{
    if (s.attach_joint < 0) {
        // EE: previous is the last movable joint. urdf_joint_info is in
        // chain order, so the last entry is closest to the tip.
        int n = urdf_joint_count(h);
        if (n > 0) {
            float m[16];
            urdf_joint_world_transform(h, n - 1, m);
            return xform_origin(m);
        }
        return make_float3(0.f, 0.f, 0.f);
    }
    if (s.attach_joint > 0) {
        float m[16];
        urdf_joint_world_transform(h, s.attach_joint - 1, m);
        return xform_origin(m);
    }
    return make_float3(0.f, 0.f, 0.f);
}

// Decide which of the joint's ±local axes map to forward and up. Runs once
// per attachment change (result cached in SensorState), so later frames reuse
// the same mapping and the basis stays locked to the joint through rotations.
static void sensor_calibrate_axes(SensorState& state, UrdfArticulation* handle)
{
    float m[16];
    if (state.attach_joint < 0) urdf_fk_ee_transform(handle, m);
    else                        urdf_joint_world_transform(handle, state.attach_joint, m);

    float3 axes[3] = {
        vnorm(make_float3(m[0], m[4], m[8])),
        vnorm(make_float3(m[1], m[5], m[9])),
        vnorm(make_float3(m[2], m[6], m[10])),
    };
    float3 j_pos   = xform_origin(m);
    float3 p_pos   = parent_origin(state, handle);
    float3 arm_raw = vsub(j_pos, p_pos);
    float3 arm_dir = (vdot(arm_raw, arm_raw) > 1e-10f)
        ? vnorm(arm_raw) : axes[2];

    int   fa = 0;
    float bf = fabsf(vdot(axes[0], arm_dir));
    for (int i = 1; i < 3; ++i) {
        float d = fabsf(vdot(axes[i], arm_dir));
        if (d > bf) { bf = d; fa = i; }
    }
    float fs = (vdot(axes[fa], arm_dir) >= 0.f) ? 1.f : -1.f;

    const float3 world_up = make_float3(0.f, 1.f, 0.f);
    int   ua = -1;
    float bu = -1.f;
    for (int i = 0; i < 3; ++i) {
        if (i == fa) continue;
        float d = fabsf(vdot(axes[i], world_up));
        if (d > bu) { bu = d; ua = i; }
    }
    float us = (vdot(axes[ua], world_up) >= 0.f) ? 1.f : -1.f;

    state.fwd_ax          = fa;
    state.fwd_sign        = fs;
    state.up_ax           = ua;
    state.up_sign         = us;
    state.axis_cached_for = state.attach_joint;
}

bool sensor_get_world_frame(const SensorState& state_in, UrdfArticulation* handle,
                            float3& out_origin, float3& out_right,
                            float3& out_up, float3& out_forward)
{
    if (!handle) return false;

    // Refresh the axis cache iff the attachment changed.
    SensorState& state = const_cast<SensorState&>(state_in);
    if (state.axis_cached_for != state.attach_joint)
        sensor_calibrate_axes(state, handle);

    // Attached joint's world transform. Its columns are the joint's local
    // axes — they rotate with the joint, and the cached indices below pick
    // which of them are "forward" and "up" for this attachment.
    float m[16];
    if (state.attach_joint < 0) urdf_fk_ee_transform(handle, m);
    else                        urdf_joint_world_transform(handle, state.attach_joint, m);

    float3 j_pos = xform_origin(m);
    float3 axes[3] = {
        vnorm(make_float3(m[0], m[4], m[8])),
        vnorm(make_float3(m[1], m[5], m[9])),
        vnorm(make_float3(m[2], m[6], m[10])),
    };

    float3 fwd = vmul(axes[state.fwd_ax], state.fwd_sign);
    float3 up  = vmul(axes[state.up_ax],  state.up_sign);

    // Orthonormalize — right = up × fwd, up = fwd × right. The result is
    // still expressed in the joint's local axes (just re-labeled), so the
    // whole basis rotates with the joint.
    float3 right = vnorm(vcross(up,    fwd));
    up           = vnorm(vcross(fwd,   right));

    // Apply rotation offsets in the derived frame:
    // yaw around up, then pitch around (yaw-rotated) right, then roll around forward.
    const float D2R = 3.14159265358979323846f / 180.f;
    float yaw   = state.rot_deg[0] * D2R;
    float pitch = state.rot_deg[1] * D2R;
    float roll  = state.rot_deg[2] * D2R;

    fwd   = rot_axis(fwd,   up, yaw);
    right = rot_axis(right, up, yaw);

    fwd = rot_axis(fwd, right, pitch);
    up  = rot_axis(up,  right, pitch);

    right = rot_axis(right, fwd, roll);
    up    = rot_axis(up,    fwd, roll);

    // Translation offset applied in the final frame, centred on the joint origin.
    float3 camo = vadd(j_pos,
                   vadd(vmul(right, state.offset[0]),
                   vadd(vmul(up,    state.offset[1]),
                        vmul(fwd,   state.offset[2]))));

    out_origin  = camo;
    out_right   = right;
    out_up      = up;
    out_forward = fwd;
    return true;
}

// ── Build a Camera from the attachment transform ─────────────────────────────
// Dead-attached to the chosen joint/EE: the camera basis tracks the joint's
// orientation exactly, so wrist roll rolls the image too. Offset is applied in
// the joint's local frame.
static Camera sensor_camera_for_render(const SensorState& s,
                                        UrdfArticulation* h,
                                        int w, int h_px)
{
    float3 origin, right, up, fwd;
    sensor_get_world_frame(s, h, origin, right, up, fwd);
    (void)right;

    float3 target = vadd(origin, fwd);
    float aspect = (h_px > 0) ? (float)w / (float)h_px : 1.f;
    return Camera::make(origin, target, up, s.fov_deg, aspect,
                        /*aperture=*/0.f, /*focus_dist=*/1.f);
}

// ── UI ───────────────────────────────────────────────────────────────────────
void sensor_panel_draw(SensorState& state, UrdfArticulation* handle)
{
    if (!ImGui::Begin("Sensors")) { ImGui::End(); return; }

    if (!handle) {
        ImGui::TextDisabled("No URDF loaded");
        ImGui::End();
        return;
    }

    ImGui::TextUnformatted("Gripper camera");
    ImGui::Checkbox("Enabled", &state.enabled);
    ImGui::SameLine();
    ImGui::Checkbox("Show gizmo", &state.show_gizmo);

    // Attachment — EE by default, or any joint in the articulation.
    int n = urdf_joint_count(handle);
    UrdfJointInfo* joints = urdf_joint_info(handle);
    const char* attach_preview = (state.attach_joint < 0 || state.attach_joint >= n)
        ? "End effector" : joints[state.attach_joint].name;
    ImGui::SetNextItemWidth(-1.f);
    if (ImGui::BeginCombo("Attach to", attach_preview)) {
        if (ImGui::Selectable("End effector", state.attach_joint < 0))
            state.attach_joint = -1;
        for (int i = 0; i < n; ++i) {
            ImGui::PushID(i + 30000);
            if (ImGui::Selectable(joints[i].name, state.attach_joint == i))
                state.attach_joint = i;
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }

    ImGui::SetNextItemWidth(-1.f);
    ImGui::SliderFloat("FOV", &state.fov_deg, 20.f, 120.f, "%.0f deg");
    // Offset in the derived camera frame (right, up, forward).
    ImGui::SetNextItemWidth(-1.f);
    ImGui::SliderFloat3("Offset (r,u,f) m", state.offset, -0.30f, 0.30f, "%.3f");
    // Rotation offsets — yaw/pitch/roll around derived basis.
    ImGui::SetNextItemWidth(-1.f);
    ImGui::SliderFloat3("Rotate (yaw,pitch,roll) deg", state.rot_deg, -180.f, 180.f, "%.1f");
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset##rot")) {
        state.rot_deg[0] = state.rot_deg[1] = state.rot_deg[2] = 0.f;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Recalibrate"))
        state.axis_cached_for = -999;   // force re-pick on next frame

    const int sizes[][2] = { {160, 120}, {320, 240}, {480, 360}, {640, 480} };
    int cur = 1;
    for (int i = 0; i < 4; ++i)
        if (state.width == sizes[i][0] && state.height == sizes[i][1]) cur = i;
    const char* labels[] = { "160x120", "320x240", "480x360", "640x480" };
    ImGui::SetNextItemWidth(-1.f);
    if (ImGui::Combo("Resolution", &cur, labels, 4)) {
        state.width  = sizes[cur][0];
        state.height = sizes[cur][1];
    }

    int corner_idx = (int)state.corner;
    const char* corners[] = { "Top Left", "Top Right", "Bottom Left", "Bottom Right" };
    ImGui::SetNextItemWidth(-1.f);
    if (ImGui::Combo("Corner", &corner_idx, corners, 4))
        state.corner = (SensorCorner)corner_idx;

    ImGui::End();
}

// ── Render + blit ────────────────────────────────────────────────────────────
void sensor_render_and_blit(SensorState& state,
                            UrdfArticulation* handle,
                            cudaSurfaceObject_t main_surface,
                            int main_w, int main_h,
                            Triangle* d_triangles, int num_triangles,
                            GpuMaterial* d_materials, int num_materials,
                            cudaTextureObject_t* d_textures,
                            float3* d_obj_colors, int num_obj_colors,
                            int color_mode, float3 bg_color,
                            float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    if (!state.enabled || !handle || num_triangles <= 0) return;
    if (state.width <= 0 || state.height <= 0) return;

    sensor_ensure_target(state);
    if (!state.surface) return;

    Camera cam = sensor_camera_for_render(state, handle, state.width, state.height);

    rasterizer_render(state.raster, state.surface, state.width, state.height,
                      cam,
                      d_triangles, num_triangles,
                      d_materials, num_materials, d_textures,
                      d_obj_colors, num_obj_colors,
                      color_mode, bg_color,
                      key_dir, fill_dir, rim_dir);

    // Position the PiP in the chosen corner with an 8 px margin.
    const int M = 8;
    int off_x = 0, off_y = 0;
    switch (state.corner) {
    case SensorCorner::TopLeft:     off_x = M;                      off_y = M; break;
    case SensorCorner::TopRight:    off_x = main_w - state.width - M; off_y = M; break;
    case SensorCorner::BottomLeft:  off_x = M;                      off_y = main_h - state.height - M; break;
    case SensorCorner::BottomRight: off_x = main_w - state.width - M; off_y = main_h - state.height - M; break;
    }

    sensor_blit_corner(main_surface, main_w, main_h,
                       state.d_array, state.width, state.height,
                       off_x, off_y);
}
