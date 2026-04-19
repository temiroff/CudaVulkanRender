#pragma once

// Auxiliary MuJoCo world for "prop" objects — the small free-bodies that
// the gripper picks up and drops. Distinct from the main sim backend (which
// owns the robot): lets us spawn / despawn cubes and spheres at runtime
// without touching the robot's MJCF.
//
// Y-up world. Gravity along -Y. Floor is a plane at a configurable height.
// Compiled to stubs when MUJOCO_ENABLED is off so callers don't need to
// #ifdef at the call site.

struct PropsPhysics;

// Create / destroy the world. Lazily rebuilds the mjModel whenever props
// are added or removed.
PropsPhysics* props_physics_create();
void          props_physics_destroy(PropsPhysics*);

// true when the build has MuJoCo. false → all other functions are no-ops
// and add_* returns -1.
bool props_physics_available();

// Floor plane y-coordinate. Takes effect on the next model rebuild.
void props_physics_set_ground(PropsPhysics*, float ground_y);

// Add a rigid box. size_{x,y,z} are full extents (not half). Returns a
// handle (>=0) or -1 on failure. Initial linear + angular velocity are
// applied after the model is built.
int props_physics_add_box(PropsPhysics*,
                          float pos_x, float pos_y, float pos_z,
                          float qw, float qx, float qy, float qz,
                          float size_x, float size_y, float size_z,
                          float vx, float vy, float vz,
                          float wx, float wy, float wz,
                          float mass);

// Add a rigid sphere.
int props_physics_add_sphere(PropsPhysics*,
                             float pos_x, float pos_y, float pos_z,
                             float radius,
                             float vx, float vy, float vz,
                             float mass);

// Remove a prop by handle. No-op if handle is unknown.
void props_physics_remove(PropsPhysics*, int handle);

// Clear every prop (used on playback loop/reset).
void props_physics_clear(PropsPhysics*);

// Step the world by dt seconds. Sub-steps at the solver timestep.
void props_physics_step(PropsPhysics*, float dt);

// Read current world pose. out_pos = (x,y,z). out_quat = (w,x,y,z).
// Returns false if the handle is unknown.
bool props_physics_get_pose(PropsPhysics*, int handle,
                            float out_pos[3], float out_quat[4]);
