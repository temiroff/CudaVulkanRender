#pragma once
// Internal data structures shared by urdf_loader.cpp and mjcf_loader.cpp.
// Public consumers include urdf_loader.h instead.

#include "scene.h"
#include "urdf_loader.h"   // UrdfJointInfo + UrdfArticulation forward decl
#include "gltf_loader.h"   // TextureImage, MeshObject
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

// ── 4x4 matrix ──────────────────────────────────────────────────────────────
struct Mat4 {
    float m[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

    static Mat4 identity() { return Mat4{}; }

    static Mat4 from_rpy_xyz(float roll, float pitch, float yaw,
                              float x, float y, float z)
    {
        float cr = cosf(roll),  sr = sinf(roll);
        float cp = cosf(pitch), sp = sinf(pitch);
        float cy = cosf(yaw),   sy = sinf(yaw);
        Mat4 T;
        T.m[0][0] = cy*cp; T.m[0][1] = cy*sp*sr - sy*cr; T.m[0][2] = cy*sp*cr + sy*sr; T.m[0][3] = x;
        T.m[1][0] = sy*cp; T.m[1][1] = sy*sp*sr + cy*cr; T.m[1][2] = sy*sp*cr - cy*sr; T.m[1][3] = y;
        T.m[2][0] = -sp;   T.m[2][1] = cp*sr;             T.m[2][2] = cp*cr;             T.m[2][3] = z;
        T.m[3][0] = 0;     T.m[3][1] = 0;                 T.m[3][2] = 0;                 T.m[3][3] = 1;
        return T;
    }

    // MJCF quaternion convention: w x y z
    static Mat4 from_quat_wxyz_xyz(float qw, float qx, float qy, float qz,
                                    float x, float y, float z)
    {
        float n = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        if (n > 1e-8f) { qw/=n; qx/=n; qy/=n; qz/=n; }
        float xx = qx*qx, yy = qy*qy, zz = qz*qz;
        float xy = qx*qy, xz = qx*qz, yz = qy*qz;
        float wx = qw*qx, wy = qw*qy, wz = qw*qz;
        Mat4 T;
        T.m[0][0] = 1 - 2*(yy+zz); T.m[0][1] = 2*(xy - wz);   T.m[0][2] = 2*(xz + wy);   T.m[0][3] = x;
        T.m[1][0] = 2*(xy + wz);   T.m[1][1] = 1 - 2*(xx+zz); T.m[1][2] = 2*(yz - wx);   T.m[1][3] = y;
        T.m[2][0] = 2*(xz - wy);   T.m[2][1] = 2*(yz + wx);   T.m[2][2] = 1 - 2*(xx+yy); T.m[2][3] = z;
        T.m[3][0] = 0;             T.m[3][1] = 0;             T.m[3][2] = 0;             T.m[3][3] = 1;
        return T;
    }

    // MJCF <body axisangle="ax ay az angle"> — rotation around an axis.
    static Mat4 from_axisangle_xyz(float ax, float ay, float az, float angle,
                                    float x, float y, float z)
    {
        float len = sqrtf(ax*ax + ay*ay + az*az);
        if (len > 1e-8f) { ax/=len; ay/=len; az/=len; }
        float c = cosf(angle), s = sinf(angle), t = 1.0f - c;
        Mat4 T;
        T.m[0][0] = t*ax*ax + c;    T.m[0][1] = t*ax*ay - s*az; T.m[0][2] = t*ax*az + s*ay; T.m[0][3] = x;
        T.m[1][0] = t*ax*ay + s*az; T.m[1][1] = t*ay*ay + c;    T.m[1][2] = t*ay*az - s*ax; T.m[1][3] = y;
        T.m[2][0] = t*ax*az - s*ay; T.m[2][1] = t*ay*az + s*ax; T.m[2][2] = t*az*az + c;    T.m[2][3] = z;
        T.m[3][0] = 0;              T.m[3][1] = 0;              T.m[3][2] = 0;              T.m[3][3] = 1;
        return T;
    }

    Mat4 operator*(const Mat4& b) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                r.m[i][j] = 0;
                for (int k = 0; k < 4; ++k)
                    r.m[i][j] += m[i][k] * b.m[k][j];
            }
        return r;
    }

    float3 transform_point(float3 p) const {
        return make_float3(
            m[0][0]*p.x + m[0][1]*p.y + m[0][2]*p.z + m[0][3],
            m[1][0]*p.x + m[1][1]*p.y + m[1][2]*p.z + m[1][3],
            m[2][0]*p.x + m[2][1]*p.y + m[2][2]*p.z + m[2][3]);
    }

    float3 transform_normal(float3 n) const {
        float3 r = make_float3(
            m[0][0]*n.x + m[0][1]*n.y + m[0][2]*n.z,
            m[1][0]*n.x + m[1][1]*n.y + m[1][2]*n.z,
            m[2][0]*n.x + m[2][1]*n.y + m[2][2]*n.z);
        float len = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        if (len > 1e-7f) { r.x /= len; r.y /= len; r.z /= len; }
        return r;
    }
};

// ── Raw mesh (intermediate geometry from file formats) ──────────────────────
struct DaeMaterial {
    float4 diffuse  = make_float4(0.7f, 0.7f, 0.7f, 1.0f);
    float  specular_intensity = 0.25f;
    float  shininess = 0.0f;
};

struct RawMesh {
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<float2> uvs;        // optional — aligned to vertices when present
    std::vector<int>    indices;
    std::vector<int>    mat_ids;
    std::vector<DaeMaterial> dae_materials;
};

// ── Kinematic tree (URDF and MJCF share this representation) ────────────────
struct URDFLink {
    std::string name;
    std::string visual_mesh_path;   // resolved path, empty = transform-only link
    Mat4        visual_origin;
    // If non-empty, geometry is provided inline (MJCF primitives: plane/box/sphere/...).
    // Takes precedence over visual_mesh_path — no disk load needed.
    RawMesh     inline_mesh;
};

struct URDFJoint {
    std::string name;
    std::string type;     // "revolute", "prismatic", "continuous", "fixed"
    std::string parent;
    std::string child;
    Mat4        origin;
    float3      axis  = {0, 0, 1};
    float       lower = 0.0f;
    float       upper = 0.0f;
};

// ── Per-link mesh cache used during articulation (repose) ───────────────────
struct LinkMeshCache {
    std::string link_name;
    RawMesh     raw;
    int         tri_start = 0;
    int         tri_count = 0;
    int         obj_id = 0;
    std::vector<int> mat_ids;
    int         link_mat_base = 0;
};

// ── Opaque articulation handle (full def — consumers use UrdfArticulation*) ─
struct UrdfArticulation {
    std::unordered_map<std::string, URDFLink> links;
    std::vector<URDFJoint>                    joints;
    std::string                               root_link;
    std::unordered_map<std::string, std::vector<const URDFJoint*>> children_map;

    std::vector<UrdfJointInfo>                joint_infos;
    std::unordered_map<std::string, int>      joint_name_to_idx;

    std::vector<LinkMeshCache>                mesh_caches;

    Mat4                                      z_to_y;
    // Extra transform applied to the root link before the kinematic chain,
    // in Z-up (pre-z_to_y) space. Used by the physics backend to inject
    // freejoint motion (root-body drift/rotation) into rendering. Identity
    // by default, so non-sim flows are unaffected.
    Mat4                                      root_xform = Mat4::identity();
    // Link name that should receive root_xform. For plain URDFs this equals
    // root_link. For MJCFs, root_link is the synthetic __world__ (which also
    // carries floor geoms as siblings of the robot), so we need to scope
    // root_xform to only the robot's top-level body — otherwise the floor
    // drifts with the freejoint delta.
    std::string                               root_body_link;

    std::string                               ee_link_name;
    float3                                    ee_world_pos = {0, 0, 0};

    // Optional floor height for IK ground repulsion. Set via
    // urdf_set_ik_ground_y(); default -FLT_MAX = disabled.
    float                                     ik_ground_y  = -1e30f;
    float                                     ik_view_topdown = 0.f;

    // Primary locked joint — excluded from IK. -1 = auto (last movable).
    // Index into joint_infos.
    int                                       ik_lock_joint = -1;

    // Joint driven by [ / ] keyboard shortcuts. -1 = follow ik_lock_joint.
    int                                       kb_joint      = -1;

    // Per-joint IK lock. When set, the solver preserves the WORLD rotation
    // of the joint's child link (so a vertical tool stays vertical while the
    // arm moves). Parallel to joint_infos; sized during finalize().
    // ik_locked_rot stores the captured world rotation of the child link at
    // the moment the lock was enabled — it's the orientation target.
    std::vector<uint8_t>                      ik_locked_mask;
    std::vector<Mat4>                         ik_locked_rot;

    UrdfIkYawDebug                            ik_yaw_debug{};

    int                                       total_tris = 0;
};

// ── Minimal XML parser (shared by URDF and MJCF) ────────────────────────────
struct XmlAttr {
    std::string name, value;
};

struct XmlNode {
    std::string                tag;
    std::string                text;
    std::vector<XmlAttr>       attrs;
    std::vector<XmlNode>       children;

    const XmlNode* child(const std::string& t) const {
        for (auto& c : children) if (c.tag == t) return &c;
        return nullptr;
    }

    std::vector<const XmlNode*> children_with_tag(const std::string& t) const {
        std::vector<const XmlNode*> out;
        for (auto& c : children) if (c.tag == t) out.push_back(&c);
        return out;
    }

    std::string attr(const std::string& name, const std::string& def = "") const {
        for (auto& a : attrs) if (a.name == name) return a.value;
        return def;
    }
};

XmlNode parse_xml(const std::string& xml_str);
std::string read_file_text(const std::string& path);

// ── Shared mesh loaders (implemented in urdf_loader.cpp) ────────────────────
bool load_stl(const std::string& path, RawMesh& out);
bool load_obj(const std::string& path, RawMesh& out);
bool load_dae(const std::string& path, RawMesh& out);
bool load_link_mesh(const std::string& mesh_path, RawMesh& raw);

// ── Post-parse articulation builder — shared by URDF and MJCF ───────────────
// Given a populated UrdfArticulation (links, joints, root_link set), finishes
// initialization: builds children_map, joint_infos, joint_name_to_idx, sets
// z_to_y, loads per-link meshes for articulation, finds end-effector, and
// computes initial ee position from initial_tris.
void urdf_articulation_finalize(UrdfArticulation* h,
                                const std::vector<Triangle>& initial_tris);

// ── Axis-angle rotation helper ──────────────────────────────────────────────
Mat4 axis_angle_rotation(float3 axis, float angle);
