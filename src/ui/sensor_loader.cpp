#include "sensor_loader.h"

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/tf/token.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

// ── small float3 helpers ─────────────────────────────────────────────────────
namespace {

struct V3 { float x, y, z; };
static inline V3 v3(float x, float y, float z) { return {x, y, z}; }
static inline V3 sub(V3 a, V3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline float dot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline V3 cross(V3 a, V3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static inline V3 norm(V3 v) {
    float l = std::sqrt(dot(v, v));
    return (l > 1e-7f) ? v3(v.x/l, v.y/l, v.z/l) : v;
}
static inline V3 col_of(const float m[16], int c) {
    return v3(m[0*4+c], m[1*4+c], m[2*4+c]);
}
static inline V3 pos_of(const float m[16]) { return v3(m[3], m[7], m[11]); }

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

// Status helper — writes into state.status and returns `ok`.
static bool set_status(SensorState& st, bool ok, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(st.status, sizeof(st.status), fmt, ap);
    va_end(ap);
    return ok;
}

// Match joint hint (from "sensor_camera_<hint>") to a URDF joint index.
// Returns -1 for end-effector-like hints ("ee"/"endeffector"/"tip"),
// -2 when no match was found.
static int match_joint(UrdfArticulation* h, const std::string& hint_raw) {
    std::string hint = to_lower(hint_raw);
    if (hint.empty()) return -2;
    if (hint == "ee" || hint == "end_effector" || hint == "endeffector" ||
        hint == "tip" || hint == "eef") return -1;

    int n = urdf_joint_count(h);
    UrdfJointInfo* ji = urdf_joint_info(h);
    // Exact match first.
    for (int i = 0; i < n; ++i) {
        if (to_lower(ji[i].name) == hint) return i;
    }
    // Substring (hint ⊂ joint_name) — picks the shortest joint name that
    // contains the hint, to prefer the most specific rather than a random
    // longer joint that happens to contain it.
    int best = -2;
    size_t best_len = (size_t)-1;
    for (int i = 0; i < n; ++i) {
        std::string jn = to_lower(ji[i].name);
        if (jn.find(hint) != std::string::npos && jn.size() < best_len) {
            best_len = jn.size();
            best = i;
        }
    }
    return best;
}

} // namespace

// ── Main entry point ─────────────────────────────────────────────────────────
bool sensor_load_from_usd(const std::string& path,
                           UrdfArticulation*  handle,
                           SensorState&       state) {
    if (!handle) return set_status(state, false, "No URDF loaded");

    UsdStageRefPtr stage = UsdStage::Open(path);
    if (!stage) return set_status(state, false, "Failed to open USD: %s", path.c_str());

    // ── Stage → app-space (Y-up, meters) transform ───────────────────────────
    GfMatrix4d axis_correction(1.0);
    TfToken up_axis = UsdGeomGetStageUpAxis(stage);
    if (up_axis == UsdGeomTokens->z) {
        axis_correction = GfMatrix4d(1,0,0,0,  0,0,-1,0,  0,1,0,0,  0,0,0,1);
    }
    double mpu = UsdGeomGetStageMetersPerUnit(stage);
    if (mpu <= 0.0) mpu = 1.0;
    GfMatrix4d scale_m(1.0);
    scale_m.SetScale(GfVec3d(mpu, mpu, mpu));
    GfMatrix4d stage_to_app = axis_correction * scale_m;

    // ── Find a "sensor_camera_*" root prim ───────────────────────────────────
    UsdPrim sensor_root;
    std::string joint_hint;
    for (const UsdPrim& p : stage->GetPseudoRoot().GetChildren()) {
        std::string n = p.GetName().GetString();
        if (n.rfind("sensor_camera_", 0) == 0) {
            sensor_root = p;
            joint_hint  = n.substr(std::strlen("sensor_camera_"));
            break;
        }
    }
    // Fallback: scan whole tree for a prim with that naming convention.
    if (!sensor_root) {
        for (const UsdPrim& p : stage->Traverse()) {
            std::string n = p.GetName().GetString();
            if (n.rfind("sensor_camera_", 0) == 0) {
                sensor_root = p;
                joint_hint  = n.substr(std::strlen("sensor_camera_"));
                break;
            }
        }
    }
    if (!sensor_root)
        return set_status(state, false,
            "No 'sensor_camera_<joint>' prim in USD");

    // ── Find a descendant named "camera_attach" ──────────────────────────────
    UsdPrim attach;
    for (const UsdPrim& p : sensor_root.GetDescendants()) {
        if (p.GetName().GetString() == "camera_attach") { attach = p; break; }
    }
    if (!attach)
        return set_status(state, false,
            "'%s' has no 'camera_attach' child", sensor_root.GetName().GetText());

    // ── Match joint hint to URDF ────────────────────────────────────────────
    int joint_idx = match_joint(handle, joint_hint);
    if (joint_idx == -2)
        return set_status(state, false,
            "No URDF joint matches hint '%s'", joint_hint.c_str());

    // ── Compute camera_attach's world pose in app space ─────────────────────
    UsdGeomXformable xf(attach);
    GfMatrix4d local_to_world = xf.ComputeLocalToWorldTransform(UsdTimeCode::Default());
    GfMatrix4d world_app = local_to_world * stage_to_app;

    // Extract position + normalized local axes. GfMatrix4d is row-major row-
    // vector: rows 0..2 = local X/Y/Z axes expressed in world; row 3 = origin.
    V3 a_pos = v3((float)world_app[3][0], (float)world_app[3][1], (float)world_app[3][2]);
    V3 a_x   = norm(v3((float)world_app[0][0], (float)world_app[0][1], (float)world_app[0][2]));
    V3 a_y   = norm(v3((float)world_app[1][0], (float)world_app[1][1], (float)world_app[1][2]));
    V3 a_z   = norm(v3((float)world_app[2][0], (float)world_app[2][1], (float)world_app[2][2]));

    // Camera convention: camera_attach is authored with local +Y = look
    // direction and local +X = up — the natural "point it at the target"
    // orientation for a Maya locator (grab the +Y handle, drag it onto the
    // target; tilt around +X to roll). Right is then cam_up × cam_fwd,
    // which for orthonormal attach axes gives +Z. Basis is right-handed
    // (det = +1 since a_x × a_y = a_z).
    V3 cam_forward = a_y;
    V3 cam_up      = a_x;
    V3 cam_right   = cross(cam_up, cam_forward);  // = +a_z for orthonormal axes

    // ── Store baseline in joint-local space ─────────────────────────────────
    // Instead of solving yaw/pitch/roll against a heuristic basis (which breaks
    // when the chosen basis is left-handed vs. the attach basis), we stash the
    // camera_attach pose in the joint's local frame and let the runtime push
    // it back through the joint's world transform each frame. Offset/rotation
    // sliders then apply *on top* of this baseline — at zero offset, the
    // gizmo lands exactly on the camera_attach prim.
    float mj[16];
    if (joint_idx < 0) urdf_fk_ee_transform(handle, mj);
    else               urdf_joint_world_transform(handle, joint_idx, mj);

    // World → joint-local: v_local = R^T · (v - t), d_local = R^T · d.
    // mj is row-major with R in its 3x3 upper-left and t in column 3.
    auto w2l_pos = [&](V3 p) -> V3 {
        V3 d = { p.x - mj[3], p.y - mj[7], p.z - mj[11] };
        return V3{ mj[0]*d.x + mj[4]*d.y + mj[8]*d.z,
                   mj[1]*d.x + mj[5]*d.y + mj[9]*d.z,
                   mj[2]*d.x + mj[6]*d.y + mj[10]*d.z };
    };
    auto w2l_dir = [&](V3 d) -> V3 {
        return V3{ mj[0]*d.x + mj[4]*d.y + mj[8]*d.z,
                   mj[1]*d.x + mj[5]*d.y + mj[9]*d.z,
                   mj[2]*d.x + mj[6]*d.y + mj[10]*d.z };
    };

    V3 o_l = w2l_pos(a_pos);
    V3 r_l = w2l_dir(cam_right);
    V3 u_l = w2l_dir(cam_up);
    V3 f_l = w2l_dir(cam_forward);

    // ── Commit to SensorState ───────────────────────────────────────────────
    state.attach_joint   = joint_idx;
    state.offset[0]      = state.offset[1]      = state.offset[2]      = 0.f;
    state.rot_deg[0]     = state.rot_deg[1]     = state.rot_deg[2]     = 0.f;
    state.has_base_pose  = true;
    state.base_origin_local[0] = o_l.x; state.base_origin_local[1] = o_l.y; state.base_origin_local[2] = o_l.z;
    state.base_right_local [0] = r_l.x; state.base_right_local [1] = r_l.y; state.base_right_local [2] = r_l.z;
    state.base_up_local    [0] = u_l.x; state.base_up_local    [1] = u_l.y; state.base_up_local    [2] = u_l.z;
    state.base_fwd_local   [0] = f_l.x; state.base_fwd_local   [1] = f_l.y; state.base_fwd_local   [2] = f_l.z;
    // Mark the axis cache as current-for-this-joint so sensor_get_world_frame
    // doesn't fire the "attachment changed" branch and clobber has_base_pose.
    state.axis_cached_for = joint_idx;

    // Arm the geometry-follower so main.cpp knows to capture the next import.
    state.follow.armed  = true;
    state.follow.active = false;
    state.follow.tri_idx.clear();
    state.follow.v_local.clear();
    state.follow.n_local.clear();

    const char* target = (joint_idx < 0) ? "end effector"
                                         : urdf_joint_info(handle)[joint_idx].name;
    return set_status(state, true, "Attached to %s (hint '%s')",
                      target, joint_hint.c_str());
}

// ── Geometry follower ────────────────────────────────────────────────────────
// The sensor's imported USD geometry is pinned to the *attach joint*, not to
// the sensor's camera frame. That means offset/rotation sliders move only the
// virtual camera (gizmo + render) while the physical geo stays locked to the
// joint, exactly as if it were a URDF child link.
namespace {

// Joint (or end-effector) world matrix in row-major float[16].
static bool joint_world_matrix(UrdfArticulation* h, int joint_idx, float m[16]) {
    if (!h) return false;
    if (joint_idx < 0) urdf_fk_ee_transform(h, m);
    else               urdf_joint_world_transform(h, joint_idx, m);
    return true;
}

// Apply inverse of a rigid transform (rotation R + translation t) to a world
// point p, yielding a joint-local point L = R^T · (p - t).
// R is stored as the 3x3 block of a row-major 4x4 matrix m:
//   m[0..2]=R row0, m[4..6]=R row1, m[8..10]=R row2, t = (m[3],m[7],m[11]).
static inline V3 world_to_local_pos(const float m[16], V3 p) {
    V3 d = { p.x - m[3], p.y - m[7], p.z - m[11] };
    // R^T · d: columns of R become rows.
    return {
        m[0]*d.x + m[4]*d.y + m[8] *d.z,
        m[1]*d.x + m[5]*d.y + m[9] *d.z,
        m[2]*d.x + m[6]*d.y + m[10]*d.z
    };
}
static inline V3 world_to_local_dir(const float m[16], V3 d) {
    return {
        m[0]*d.x + m[4]*d.y + m[8] *d.z,
        m[1]*d.x + m[5]*d.y + m[9] *d.z,
        m[2]*d.x + m[6]*d.y + m[10]*d.z
    };
}
static inline V3 local_to_world_pos(const float m[16], V3 L) {
    return {
        m[0]*L.x + m[1]*L.y + m[2] *L.z + m[3],
        m[4]*L.x + m[5]*L.y + m[6] *L.z + m[7],
        m[8]*L.x + m[9]*L.y + m[10]*L.z + m[11]
    };
}
static inline V3 local_to_world_dir(const float m[16], V3 L) {
    return {
        m[0]*L.x + m[1]*L.y + m[2] *L.z,
        m[4]*L.x + m[5]*L.y + m[6] *L.z,
        m[8]*L.x + m[9]*L.y + m[10]*L.z
    };
}

} // namespace

void sensor_geo_capture(SensorState& state, UrdfArticulation* handle,
                         int first_obj_id, int last_obj_id,
                         std::vector<Triangle>& all_prims) {
    state.follow.tri_idx.clear();
    state.follow.v_local.clear();
    state.follow.n_local.clear();
    state.follow.active = false;

    if (!handle || last_obj_id < first_obj_id) return;

    float mj[16];
    if (!joint_world_matrix(handle, state.attach_joint, mj)) return;

    for (int i = 0; i < (int)all_prims.size(); ++i) {
        const Triangle& t = all_prims[i];
        if (t.obj_id < first_obj_id || t.obj_id > last_obj_id) continue;
        state.follow.tri_idx.push_back(i);
        V3 v0 = world_to_local_pos(mj, v3(t.v0.x, t.v0.y, t.v0.z));
        V3 v1 = world_to_local_pos(mj, v3(t.v1.x, t.v1.y, t.v1.z));
        V3 v2 = world_to_local_pos(mj, v3(t.v2.x, t.v2.y, t.v2.z));
        V3 n0 = world_to_local_dir(mj, v3(t.n0.x, t.n0.y, t.n0.z));
        V3 n1 = world_to_local_dir(mj, v3(t.n1.x, t.n1.y, t.n1.z));
        V3 n2 = world_to_local_dir(mj, v3(t.n2.x, t.n2.y, t.n2.z));
        state.follow.v_local.push_back(make_float3(v0.x, v0.y, v0.z));
        state.follow.v_local.push_back(make_float3(v1.x, v1.y, v1.z));
        state.follow.v_local.push_back(make_float3(v2.x, v2.y, v2.z));
        state.follow.n_local.push_back(make_float3(n0.x, n0.y, n0.z));
        state.follow.n_local.push_back(make_float3(n1.x, n1.y, n1.z));
        state.follow.n_local.push_back(make_float3(n2.x, n2.y, n2.z));
    }

    state.follow.obj_id_start = first_obj_id;
    state.follow.obj_id_end   = last_obj_id;
    state.follow.joint_idx    = state.attach_joint;
    state.follow.active       = !state.follow.tri_idx.empty();
    state.follow.armed        = false;
    state.follow.has_cached   = false;
}

bool sensor_geo_update(SensorState& state, UrdfArticulation* handle,
                        std::vector<Triangle>& all_prims) {
    if (!state.follow.active || !handle) return false;
    if (state.follow.tri_idx.empty())   return false;

    float mj[16];
    if (!joint_world_matrix(handle, state.follow.joint_idx, mj)) return false;

    // Skip work when the joint hasn't moved since the last apply.
    if (state.follow.has_cached) {
        bool same = true;
        for (int i = 0; i < 16; ++i)
            if (std::fabs(mj[i] - state.follow.cached_mj[i]) > 1e-6f) {
                same = false; break;
            }
        if (same) return false;
    }
    for (int i = 0; i < 16; ++i) state.follow.cached_mj[i] = mj[i];
    state.follow.has_cached = true;

    for (int i = 0; i < (int)state.follow.tri_idx.size(); ++i) {
        int idx = state.follow.tri_idx[i];
        if (idx < 0 || idx >= (int)all_prims.size()) continue;
        Triangle& t = all_prims[idx];
        float3 L;
        L = state.follow.v_local[3*i + 0];
        V3 w0 = local_to_world_pos(mj, v3(L.x, L.y, L.z));
        L = state.follow.v_local[3*i + 1];
        V3 w1 = local_to_world_pos(mj, v3(L.x, L.y, L.z));
        L = state.follow.v_local[3*i + 2];
        V3 w2 = local_to_world_pos(mj, v3(L.x, L.y, L.z));
        L = state.follow.n_local[3*i + 0];
        V3 m0 = local_to_world_dir(mj, v3(L.x, L.y, L.z));
        L = state.follow.n_local[3*i + 1];
        V3 m1 = local_to_world_dir(mj, v3(L.x, L.y, L.z));
        L = state.follow.n_local[3*i + 2];
        V3 m2 = local_to_world_dir(mj, v3(L.x, L.y, L.z));
        t.v0 = make_float3(w0.x, w0.y, w0.z);
        t.v1 = make_float3(w1.x, w1.y, w1.z);
        t.v2 = make_float3(w2.x, w2.y, w2.z);
        t.n0 = make_float3(m0.x, m0.y, m0.z);
        t.n1 = make_float3(m1.x, m1.y, m1.z);
        t.n2 = make_float3(m2.x, m2.y, m2.z);
    }
    return true;
}
