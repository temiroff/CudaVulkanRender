#pragma once
#include "../urdf_loader.h"

struct RobotDemoState;  // ui/robot_demo_panel.h

// Draw the URDF Articulation ImGui panel.
// Returns true if any joint angle was changed (caller should repose + rebuild BVH).
// When playback_active is true, sliders are read-only (showing playback-driven values).
// If gripper_state is non-null and the articulation has recognizable finger
// joints, the panel shows Close/Open buttons that start a smooth animation on
// gripper_state->gripper_anim; the caller ticks the animation each frame and
// feeds the result through the existing grasp pipeline.
bool articulation_panel_draw(UrdfArticulation* handle, bool playback_active = false,
                             RobotDemoState* gripper_state = nullptr);
