#pragma once
#include "../urdf_loader.h"

// Draw the URDF Articulation ImGui panel.
// Returns true if any joint angle was changed (caller should repose + rebuild BVH).
// When playback_active is true, sliders are read-only (showing playback-driven values).
bool articulation_panel_draw(UrdfArticulation* handle, bool playback_active = false);
