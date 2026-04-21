#pragma once
#include "scene.h"
#include "gltf_loader.h"   // reuse MeshObject, TextureImage
#include <vector>
#include <string>

// Parse a .urdf file.
// Resolves mesh references (COLLADA .dae, STL .stl), applies the kinematic
// chain transforms, and converts all visual geometry into flat Triangle lists
// with default PBR materials — identical output to gltf_load / usd_load.
// Returns true on success.
bool urdf_load(const std::string&          path,
               std::vector<Triangle>&      triangles_out,
               std::vector<GpuMaterial>&   materials_out,
               std::vector<TextureImage>&  textures_out,
               std::vector<MeshObject>&    objects_out);

// ── URDF Articulation (persistent state for interactive joint control) ───────

struct UrdfArticulation;   // opaque handle

// Open a URDF for interactive articulation. Call after urdf_load().
// Caches the kinematic chain, raw meshes, and initial triangle mapping.
UrdfArticulation* urdf_articulation_open(const std::string& path,
                                          const std::vector<Triangle>& initial_tris);

// Joint info for UI
struct UrdfJointInfo {
    char  name[64];
    int   type;          // 0=revolute, 1=prismatic, 2=continuous, 3=fixed
    float lower;         // limit (radians for revolute, meters for prismatic)
    float upper;
    float angle;         // current value (radians or meters)
};

// Get the list of articulated joints (excludes fixed joints).
int  urdf_joint_count(const UrdfArticulation* h);
UrdfJointInfo* urdf_joint_info(UrdfArticulation* h);  // array of joint_count

// Get the world-space position of the end-effector (deepest leaf link origin).
float3 urdf_end_effector_pos(UrdfArticulation* h);

// Solve inverse kinematics: adjust joint angles so the end-effector reaches
// target_pos. Uses weighted damped least-squares Jacobian with joint limits,
// reach clamping, and null-space joint centering.
// Returns true if converged within tolerance.
bool urdf_solve_ik(UrdfArticulation* h, float3 target_pos,
                   int max_iters = 10, float tolerance = 0.005f);

// Solve full 6-DOF pose IK: drive the end-effector to both target_pos and
// target_rot_mat16 (row-major 4×4, top-left 3×3 used as R_target).
// Pass target_rot_mat16 = nullptr to fall back to position-only.
// When a rotation target is given, the primary IK lock is ignored internally
// so every movable joint can contribute to the orientation.
bool urdf_solve_ik_pose(UrdfArticulation* h, float3 target_pos,
                        const float* target_rot_mat16,
                        int max_iters = 20, float tolerance = 0.005f);

// Solve IK for a specific joint in the chain (not the ee).
// Only adjusts joints [0..joint_idx) to bring joint_idx toward target_pos.
// joint_idx: 0-based movable joint index (e.g. 2 = 3rd movable joint).
bool urdf_solve_ik_joint(UrdfArticulation* h, int joint_idx, float3 target_pos,
                         int max_iters = 6, float tolerance = 0.005f);

// Get the world-space position of the Nth movable joint (0-based).
float3 urdf_joint_pos(UrdfArticulation* h, int joint_idx);

// Get the UrdfJointInfo for the Nth movable joint in the IK chain (0-based).
// Returns nullptr if out of range. Allows reading name/angle and writing angle
// directly for per-joint viewport control.
UrdfJointInfo* urdf_chain_joint_info(UrdfArticulation* h, int movable_idx);

// Get the number of movable joints in the IK chain.
int urdf_ik_chain_length(UrdfArticulation* h);

// Get / set the end-effector link name. The IK chain goes from the root up to
// (and including) this link. Changing it takes effect immediately — the chain
// is rebuilt on every IK call.
// Set a floor Y height so the IK solver adds null-space repulsion when any
// joint position would go below it. Pass -FLT_MAX (default) to disable.
void urdf_set_ik_ground_y(UrdfArticulation* h, float ground_y);

const char* urdf_get_ee_link_name(const UrdfArticulation* h);
void        urdf_set_ee_link_name(UrdfArticulation* h, const char* link_name);

// Return all link names in the articulation, ordered by depth from the root
// (breadth-first). Useful for populating a "choose EE link" combo.
void urdf_link_names(const UrdfArticulation* h, std::vector<std::string>& out);

// Which joint IK leaves alone (so the user can set its angle manually).
// -1 means no primary IK lock. Index is into urdf_joint_info().
int  urdf_ik_lock_joint(UrdfArticulation* h);
void urdf_set_ik_lock_joint(UrdfArticulation* h, int joint_idx);

// Resolve the effective locked joint index for UI/keyboard fallback.
// Returns -1 if nothing is movable.
int  urdf_ik_lock_joint_effective(UrdfArticulation* h);

// Joint rotated by the [ / ] keyboard shortcuts. -1 = "follow IK lock"
// (the default, preserves the original hotkey behaviour).
int  urdf_kb_joint(UrdfArticulation* h);
void urdf_set_kb_joint(UrdfArticulation* h, int joint_idx);

// Resolve the effective keyboard-driven joint: falls back to the IK-locked
// joint when kb_joint is -1. Returns -1 if nothing is available.
int  urdf_kb_joint_effective(UrdfArticulation* h);

// Per-joint multi-lock. Any joint marked locked is excluded from IK, so
// the solver has to find a solution without moving it. The primary lock
// (above) is implicitly locked as well. Joint index is into urdf_joint_info().
bool urdf_ik_is_locked(UrdfArticulation* h, int joint_idx);
void urdf_ik_set_locked(UrdfArticulation* h, int joint_idx, bool locked);
void urdf_ik_clear_all_locks(UrdfArticulation* h);

// Compute end-effector position via forward kinematics (from current joint angles).
// Unlike urdf_end_effector_pos which returns a cached mesh centroid, this computes
// the FK chain tip directly — consistent with what the IK solver uses internally.
float3 urdf_fk_ee_pos(UrdfArticulation* h);

// Get the full 4x4 end-effector transform (position + rotation) from FK chain.
// out_mat: 16 floats in row-major order (m[row][col]).
void urdf_fk_ee_transform(UrdfArticulation* h, float* out_mat16);

// World-space transform of any joint (by index into urdf_joint_info array),
// with the joint's current angle applied. Writes identity when the joint
// isn't in the IK chain. Row-major 4x4.
void urdf_joint_world_transform(UrdfArticulation* h, int joint_idx, float* out_mat16);

// Re-evaluate geometry from current joint angles.
// Updates triangles_out in-place (must be same size as initial_tris).
// Returns true if any geometry changed.
bool urdf_repose(UrdfArticulation* h, std::vector<Triangle>& triangles_out);

// Override the root-link base transform (in articulation-native / Z-up space).
// Use this to feed freejoint motion from a physics backend into rendering.
// m16 is row-major 4x4. Pass the identity (or call urdf_clear_root_xform)
// to return to default behavior.
void urdf_set_root_xform(UrdfArticulation* h, const float m16[16]);
void urdf_clear_root_xform(UrdfArticulation* h);

// Re-evaluate geometry using sim-provided per-body world transforms (Z-up,
// row-major 4x4 per body in body_mats_flat, names.size()*16 floats).
// For any articulation link whose name matches a body name, the link's world
// transform is snapped to z_to_y * body_world instead of being computed by
// reimplemented FK. Links not in the map inherit via the joint chain from
// their parent as usual (synthetic joint-frame links and geom-child links).
// This keeps rendering exactly consistent with the sim — no divergence from
// joint pos/axis/quat reinterpretation. Returns true if geometry changed.
bool urdf_repose_with_body_xforms(UrdfArticulation* h,
                                   const std::vector<std::string>& body_names,
                                   const std::vector<float>& body_mats_flat,
                                   std::vector<Triangle>& triangles_out);

// Indices (into urdf_joint_info array) of gripper finger joints — joints
// whose name contains "finger" and are drivable (revolute, prismatic, or
// continuous). Empty if no fingers are recognizable. Used by the UI to
// expose a dedicated Open/Close gripper control.
void urdf_gripper_finger_indices(UrdfArticulation* h, std::vector<int>& out);

// World-space "grip point" — where a grasped object should be anchored.
// Computed as the EE link origin projected forward along the hand's local
// +Z axis (the conventional direction fingers extend in URDF/MJCF gripper
// definitions) by `forward_offset` meters. For Panda, 0.08–0.10 m puts this
// between the fingertips. Falls back to urdf_end_effector_pos when FK fails.
float3 urdf_gripper_grip_point(UrdfArticulation* h, float forward_offset);

// World-space origins of every link whose name contains "finger", computed
// by walking the full kinematic tree with the current joint angles applied.
// Used for proximity-based grasp detection so the "grab zone" covers both
// fingertips rather than a single tool-frame point.
void urdf_gripper_finger_worlds(UrdfArticulation* h, std::vector<float3>& out);

// Free all cached data.
void urdf_articulation_close(UrdfArticulation* h);
