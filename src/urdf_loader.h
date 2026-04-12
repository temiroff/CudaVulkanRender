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
// target_pos. Uses CCD (Cyclic Coordinate Descent) with joint limits.
// Returns true if converged within tolerance.
bool urdf_solve_ik(UrdfArticulation* h, float3 target_pos,
                   int max_iters = 10, float tolerance = 0.005f);

// Solve IK for a specific joint in the chain (not the ee).
// Only adjusts joints [0..joint_idx) to bring joint_idx toward target_pos.
// joint_idx: 0-based movable joint index (e.g. 2 = 3rd movable joint).
bool urdf_solve_ik_joint(UrdfArticulation* h, int joint_idx, float3 target_pos,
                         int max_iters = 6, float tolerance = 0.005f);

// Get the world-space position of the Nth movable joint (0-based).
float3 urdf_joint_pos(UrdfArticulation* h, int joint_idx);

// Get the number of movable joints in the IK chain.
int urdf_ik_chain_length(UrdfArticulation* h);

// Compute end-effector position via forward kinematics (from current joint angles).
// Unlike urdf_end_effector_pos which returns a cached mesh centroid, this computes
// the FK chain tip directly — consistent with what the IK solver uses internally.
float3 urdf_fk_ee_pos(UrdfArticulation* h);

// Re-evaluate geometry from current joint angles.
// Updates triangles_out in-place (must be same size as initial_tris).
// Returns true if any geometry changed.
bool urdf_repose(UrdfArticulation* h, std::vector<Triangle>& triangles_out);

// Free all cached data.
void urdf_articulation_close(UrdfArticulation* h);
