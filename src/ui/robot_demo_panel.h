#pragma once

#include "../scene.h"
#include "../gltf_loader.h"
#include "../urdf_loader.h"
#include <vector>
#include <string>

struct PropsPhysics;  // forward decl — real MuJoCo auxiliary world (sim/props_physics.h)

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
    float3 velocity = {0, 0, 0};   // initial velocity (used at registration time)
    bool   stopped  = false;       // simple integrator only; MuJoCo ignores

    // MuJoCo binding. Populated when the prop is registered with the
    // auxiliary physics world. -1 means "not yet registered" — the next
    // update_falling tick will create the body.
    int    mujoco_handle = -1;
    float3 body_centroid = {0, 0, 0};  // rigid pivot; world pose from MuJoCo
    std::vector<float3> body_local_verts;  // per-vertex offsets from body_centroid
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
    // Grip anchor offset in EE-local space (metres). Default zero = EE origin.
    // Drag the green debug dot in the viewport to position it between the jaws;
    // the offset is stored here so it follows the arm as it moves.
    float grip_local[3]   = {0.f, 0.f, 0.f};

    // Face-pick grasp setup.
    // Click finger face → grip_local updated + grip_face_normal_local stored.
    // Click object face → pick_face_world_pos/normal stored.
    // IK planner uses face normals to derive approach direction and orientation.
    bool  face_pick_mode             = false;
    bool  grip_face_set              = false;
    int   grip_face_tri_idx          = -1;              // mesh.all_prims triangle index
    float grip_face_bary[2]          = {0.f, 0.f};      // Moller u/v on selected triangle
    int   grip_face_normal_sign      = 1;               // preserves clicked side of triangle
    float grip_face_normal_local[3]  = {0.f, 0.f, 0.f}; // EE-local outward normal of grip face
    bool  pick_face_set              = false;
    int   pick_face_obj_idx          = -1;              // objects[] index for picked object face
    int   pick_face_tri_idx          = -1;              // mesh.all_prims triangle index
    float pick_face_bary[2]          = {0.f, 0.f};      // Moller u/v on selected triangle
    int   pick_face_normal_sign      = 1;               // preserves clicked side of triangle
    float pick_face_world_pos[3]     = {0.f, 0.f, 0.f}; // world-space contact point on object
    float pick_face_world_normal[3]  = {0.f, 0.f, 0.f}; // outward normal of object face

    // Free-fall physics (applied to detached objects). When props_world is
    // set, MuJoCo runs the rigid-body sim. Otherwise a simple translation-
    // only integrator runs as a fallback. Main owns the PropsPhysics
    // lifetime; the state just borrows the pointer.
    PropsPhysics* props_world = nullptr;
    std::vector<FallingObject> falling;
    float gravity    = -9.81f;   // along world +Y (used by fallback integrator only)
    float ground_y   = 0.f;      // floor height; mirrored into MuJoCo on every rebuild
    float restitution = 0.40f;   // fallback only
    float friction    = 0.30f;   // fallback only
    float prop_density = 400.f;  // kg/m^3 for mesh-AABB mass estimation

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
    bool request_pick_and_place = false;  // "Pick & Place" button in articulation panel

    // Pick & Place approach mode: 0 = Auto (top first, sides as fallback),
    // 1 = Top only (wrist perpendicular to ground, fingers pointing down),
    // 2 = Side only (wrist parallel to ground, horizontal fingers).
    int pick_approach_mode = 0;

    // Debug overlay: draw a live line from the current gripper grip anchor
    // to the nearest pickup* mesh's centroid so the user can visually verify
    // the planner will aim at the right object.
    bool show_pick_debug_line = false;

    // Planned snap-attach target — set by pick_and_place so the scheduled
    // attach teleports THIS exact object to the EE origin, avoiding the
    // "cube floats with a gap" bug when IK lands a bit off target or drops
    // into position-only fallback (EE orientation is then arbitrary, so the
    // grip point lands up to grip_forward away from the cube). Cleared on
    // consumption and on playback stop.
    bool  planned_pick_valid   = false;
    int   planned_pick_obj_idx = -1;
    bool  planned_pick_is_mesh = false;

    // Last plan result (for status text in the UI)
    int   last_pick_status   = 0;   // 0=none, 1=success, -1=no object, -2=unreachable
    char  last_pick_msg[96]  = {};  // human-readable summary
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

// Snap-attach a SPECIFIC object to the EE origin: first teleports the
// object so its centroid coincides with the current EE origin, then
// attaches with zero offset. Used by Pick & Place so the planned cube is
// always picked up cleanly (no drift from IK tolerance or position-only
// fallback orientation ambiguity). Returns true on success.
bool robot_demo_snap_attach(RobotDemoState& state, UrdfArticulation* handle,
                            std::vector<Sphere>& spheres,
                            std::vector<MeshObject>& objects,
                            std::vector<Triangle>& all_prims,
                            int obj_idx, bool is_mesh);

// Plan and queue a pick-and-place trajectory:
//   1. Find the nearest mesh prop named pickup*/pickups* to the end-effector.
//   2. Auto-select the best approach direction (top, ±X, ±Z) by IK feasibility.
//   3. Auto-compute a drop zone 30 cm along world +X from the cube, sitting
//      on the floor (state.ground_y + half-height). If that isn't reachable,
//      try -X, +Z, -Z until one solves.
//   4. Build 8 waypoints (start → pregrasp → grasp → close+attach → lift →
//      place_pregrasp → place → open+release → retract) using pose IK.
//   5. Start playback with the arm at waypoint 0 (the current pose).
// Writes status into state.last_pick_status / state.last_pick_msg. Returns
// true if a feasible plan was found and playback was started.
bool robot_demo_pick_and_place(RobotDemoState& state, UrdfArticulation* handle,
                               const std::vector<Sphere>& spheres,
                               const std::vector<MeshObject>& objects,
                               const std::vector<Triangle>& all_prims);

// Debug helper: returns the two endpoints the planner would aim at — the
// current gripper grip anchor and the centroid of the nearest pickup mesh.
// Returns false if no pickup* mesh is present (no line to draw).
bool robot_demo_pick_target_preview(const RobotDemoState& state,
                                    UrdfArticulation* handle,
                                    const std::vector<MeshObject>& objects,
                                    const std::vector<Triangle>& all_prims,
                                    float3& out_from_ee,
                                    float3& out_to_target);

// Detach the currently grasped object and schedule it for free-fall.
// Returns true if something was detached.
bool robot_demo_force_detach(RobotDemoState& state);

// Advance free-fall simulation by dt seconds. When props_world is non-null
// and MuJoCo is compiled in, full 6-DoF rigid-body physics runs (gravity,
// tumbling, restitution from contact). Otherwise falls back to a simple
// translation-only integrator. Returns true if any object moved.
bool robot_demo_update_falling(RobotDemoState& state,
                               PropsPhysics* props_world, float dt,
                               std::vector<Sphere>& spheres,
                               std::vector<MeshObject>& objects,
                               std::vector<Triangle>& all_prims);

// Update the cached EE velocity (used as initial velocity when an object
// is released). Call every frame with the real-time delta.
void robot_demo_track_ee_velocity(RobotDemoState& state,
                                  UrdfArticulation* handle, float dt);
