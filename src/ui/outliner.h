#pragma once
#include "../scene.h"
#include "../gltf_loader.h"
#include <vector>
#include <unordered_set>

// Draws the "Outliner" ImGui window.
// selected_mesh_obj  — primary selection (single leaf, for gizmo/silhouette)
// multi_sel          — all selected obj indices (leaves under a group click)
// Returns true if selection changed.
bool outliner_draw(
    const std::vector<Sphere>&     spheres,
    const std::vector<MeshObject>& mesh_objects,
    int&                           selected_sphere,
    int&                           selected_mesh_obj,
    std::unordered_set<int>&       multi_sel);
