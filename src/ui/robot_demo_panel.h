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
    bool               attached = false; // true = object attached at this waypoint
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

// ── Falling object: simple gravity integrator for post-detach free-fall ──────

struct FallingObject {
    bool   is_mesh = false;
    int    obj_idx = -1;
    float3 velocity = {0, 0, 0};
    bool   stopped  = false;
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
    float grasp_threshold = 0.25f;  // proximity distance for auto-attach (m)
    float grip_forward    = 0.08f;  // offset from ee origin along hand +Z (m)

    // Free-fall physics (applied to detached objects)
    std::vector<FallingObject> falling;
    float gravity    = -9.81f;   // along world +Y
    float ground_y   = 0.f;      // floor height; falling objects rest on this plane
    float restitution = 0.40f;   // bounce coefficient on ground hit (0=dead, 1=elastic)
    float friction    = 0.30f;   // horizontal velocity damping on ground contact

    // EE velocity tracking — detached objects inherit this so arm motion
    // produces a natural throw arc instead of a dead drop.
    float3 prev_ee_pos   = {0, 0, 0};
    bool   prev_ee_valid = false;
    float3 last_ee_vel   = {0, 0, 0};

    // Internal
    bool prev_gripper_closed = false;
    bool prev_attached       = false; // for waypoint-driven attach transitions
    bool scrub_changed       = false; // set by timeline scrub, cleared by main loop
    bool needs_grasp_reset   = false; // set on loop/reset, cleared after reset

    // UI intents (set by buttons or by playback, consumed by main loop)
    bool request_attach = false;
    bool request_detach = false;
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

// Snap fingers to lower (close=true) or upper (open=false) limits, flip
// state.gripper_closed so the next grasp update sees the transition.
// Returns true if finger angles changed.
bool gripper_toggle(RobotDemoState& state, UrdfArticulation* handle, bool close);

// Force-attach the nearest scene object to the end-effector, ignoring the
// proximity threshold. Returns true if an object was attached.
bool robot_demo_force_attach(RobotDemoState& state, UrdfArticulation* handle,
                             std::vector<Sphere>& spheres,
                             std::vector<MeshObject>& objects,
                             std::vector<Triangle>& all_prims);

// Detach the currently grasped object and schedule it for free-fall.
// Returns true if something was detached.
bool robot_demo_force_detach(RobotDemoState& state);

// Advance free-fall simulation by dt seconds. Moves each falling object
// under gravity, stopping when it reaches state.ground_y. Returns true if
// any object's geometry moved (caller must re-upload/refit).
bool robot_demo_update_falling(RobotDemoState& state, float dt,
                               std::vector<Sphere>& spheres,
                               std::vector<MeshObject>& objects,
                               std::vector<Triangle>& all_prims);

// Update the cached EE velocity (used as initial velocity when an object
// is released). Call every frame with the real-time delta.
void robot_demo_track_ee_velocity(RobotDemoState& state,
                                  UrdfArticulation* handle, float dt);
