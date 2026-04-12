#pragma once

#include "../scene.h"
#include "../gltf_loader.h"
#include "../urdf_loader.h"
#include <vector>
#include <string>

// ── Waypoint: a snapshot of all joint angles + gripper state ─────────────────

struct RobotWaypoint {
    float              time;            // seconds from start of trajectory
    std::vector<float> angles;          // one per joint (same order as urdf_joint_info)
    bool               gripper_closed;  // true = fingers closed
};

// ── Grasp state: tracks an attached scene object ─────────────────────────────

struct GraspState {
    bool   active     = false;  // object currently attached to end-effector
    bool   is_mesh    = false;  // true = mesh object, false = sphere
    int    obj_idx    = -1;     // index in prims_sorted (sphere) or objects[] (mesh)
    float3 offset     = {};     // object center - ee_pos at grab time (in ee local space)
    float  grab_inv_rot[9] = {}; // inverse of ee rotation at grab time (3x3, row-major)
    float3 original_pos = {};   // object position before first grab (for reset)
    bool   has_original = false;
    // Per-vertex local offsets (in ee-local space at grab time) for rotation support
    std::vector<float3> local_verts;  // vertex positions relative to ee at grab time
    // Original world-space vertices for reset (v0,v1,v2 per triangle, same order as local_verts)
    std::vector<float3> original_verts;
};

// ── Robot Demo panel state ───────────────────────────────────────────────────

struct RobotDemoState {
    std::vector<RobotWaypoint> waypoints;

    // Playback
    bool  playing        = false;
    bool  loop           = true;
    float playback_time  = 0.f;   // current time in seconds
    float playback_speed = 1.0f;

    // Recording
    bool  recording        = true;   // start in record mode
    bool  gripper_closed   = false;  // current gripper toggle in UI
    bool  ik_enabled       = false;  // IK drag mode for posing

    // Grasp
    GraspState grasp;
    float grasp_threshold = 0.15f;  // proximity distance for auto-attach

    // Internal
    bool prev_gripper_closed = false;
    bool scrub_changed       = false;  // set by timeline scrub, cleared by main loop
    bool needs_grasp_reset   = false;  // set on loop/reset, cleared after reset
};

// Draw the Robot Demo ImGui panel. Returns nothing — state is mutated directly.
void robot_demo_panel_draw(RobotDemoState& state, UrdfArticulation* handle);

// Advance playback by dt seconds. If playing, interpolates joint angles and
// writes them into the UrdfArticulation handle. Returns true if joints changed.
bool robot_demo_tick(RobotDemoState& state, UrdfArticulation* handle, float dt);

// Update grasp attachment: move grasped object to follow end-effector,
// or attach/detach on gripper state transitions.
// Checks both spheres and mesh objects for proximity.
void robot_demo_update_grasp(RobotDemoState& state, UrdfArticulation* handle,
                             std::vector<Sphere>& spheres,
                             std::vector<MeshObject>& objects,
                             std::vector<Triangle>& all_prims);

// Reset grasped object to its original position (call on playback restart/loop).
void robot_demo_reset_grasp(RobotDemoState& state,
                            std::vector<Sphere>& spheres,
                            std::vector<MeshObject>& objects,
                            std::vector<Triangle>& all_prims);

// Save/load trajectory to/from JSON.
bool robot_demo_save(const RobotDemoState& state, const char* path);
bool robot_demo_load(RobotDemoState& state, const char* path);
