#pragma once
#include "scene.h"
#include "gltf_loader.h"   // MeshObject, TextureImage
#include "urdf_loader.h"   // UrdfArticulation (shared articulation type)
#include <vector>
#include <string>

// ─────────────────────────────────────────────
//  MuJoCo MJCF (.xml) loader.
//
//  Supports the subset of MJCF needed to render and articulate robots from
//  google-deepmind's mujoco_menagerie:
//    - <include file="..."/> recursive resolution
//    - <compiler angle="radian|degree" meshdir="..." autolimits="true"/>
//    - <default class="..."> inheritance (nested, with childclass attr)
//    - <asset> <mesh> (.stl / .obj) and <material> (rgba-based)
//    - <worldbody> nested <body> with pos/quat/euler/axisangle
//    - <geom type="mesh"> with class-based visual/collision filtering
//    - <joint> with type="hinge|slide" and range limits
//
//  Physics-only elements (<inertial>, <contact>, <actuator>, <tendon>,
//  <equality>, <keyframe>) are ignored.
//
//  Articulation uses the same UrdfArticulation handle as urdf_loader, so
//  existing UI panels (articulation_panel, robot_demo_panel) work unchanged.
// ─────────────────────────────────────────────

bool mjcf_load(const std::string&          path,
               std::vector<Triangle>&      triangles_out,
               std::vector<GpuMaterial>&   materials_out,
               std::vector<TextureImage>&  textures_out,
               std::vector<MeshObject>&    objects_out);

// Open an MJCF for interactive articulation. Returns a UrdfArticulation*
// compatible with urdf_repose, urdf_solve_ik, urdf_joint_info, and all UI
// panels that accept a URDF handle.
UrdfArticulation* mjcf_articulation_open(const std::string& path,
                                          const std::vector<Triangle>& initial_tris);
