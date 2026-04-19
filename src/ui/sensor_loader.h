#pragma once

#include "sensor_panel.h"
#include "../urdf_loader.h"
#include <string>

// ── Sensor USD loader ────────────────────────────────────────────────────────
// Reads a USD file with the convention:
//
//   /sensor_camera_<joint>          Xform (root — suffix names the attach joint)
//     └── camera_attach             Prim whose world transform is the
//                                   desired camera origin + orientation.
//
// Matches <joint> against the current URDF's joint names (case-insensitive
// substring; "ee"/"endeffector"/"tip" → end-effector) and writes
// attach_joint, offset and rot_deg into `state` so the sensor lands exactly
// at camera_attach's world pose, with full tracking as the joint moves.
//
// Returns true on success; either way, a human-readable message is written
// to `state.status` for display in the UI.
bool sensor_load_from_usd(const std::string& path,
                          UrdfArticulation*  handle,
                          SensorState&       state);

// Capture the newly-imported sensor triangles (obj_id in [first_id..last_id])
// into camera-local coordinates. Call once, right after the USD import
// completes, when state.follow.armed is true.
void sensor_geo_capture(SensorState& state, UrdfArticulation* handle,
                        int first_obj_id, int last_obj_id,
                        std::vector<Triangle>& all_prims);

// Reposition the sensor's triangles to ride with the current joint pose.
// Returns true if triangle positions changed (caller should re-upload to GPU
// and refit the BVH). No-op when follow.active is false.
bool sensor_geo_update(SensorState& state, UrdfArticulation* handle,
                       std::vector<Triangle>& all_prims);
