#pragma once
#include "../urdf_loader.h"

// Draw the URDF Articulation ImGui panel.
// Returns true if any joint angle was changed (caller should repose + rebuild BVH).
bool articulation_panel_draw(UrdfArticulation* handle);
