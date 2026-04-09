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

// Re-evaluate geometry from current joint angles.
// Updates triangles_out in-place (must be same size as initial_tris).
// Returns true if any geometry changed.
bool urdf_repose(UrdfArticulation* h, std::vector<Triangle>& triangles_out);

// Free all cached data.
void urdf_articulation_close(UrdfArticulation* h);
