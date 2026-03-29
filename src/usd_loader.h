#pragma once
#include "scene.h"
#include "gltf_loader.h"   // reuse MeshObject, TextureImage
#include <vector>
#include <string>

// Parse a .usd / .usda / .usdc / .usdz file.
// Extracts UsdGeomMesh prims → Triangle list, UsdPreviewSurface → GpuMaterial,
// and any bound textures → TextureImage.
// Returns true on success.
bool usd_load(const std::string&          path,
              std::vector<Triangle>&      triangles_out,
              std::vector<GpuMaterial>&   materials_out,
              std::vector<TextureImage>&  textures_out,
              std::vector<MeshObject>&    objects_out);

// ── USD Animation ────────────────────────────────────────────────────────────
// Opaque handle that keeps a USD stage open with cached topology/UVs/materials.
// usd_load_frame() re-evaluates only time-varying data (points, normals, xforms).
struct UsdAnimHandle;

// Open a stage and cache per-mesh topology for animated playback.
// Call after usd_load() on the same path.
UsdAnimHandle* usd_anim_open(const std::string& path,
                             const std::vector<Triangle>& initial_tris);

// Re-evaluate geometry at a given time code.  Writes into triangles_out
// (must be pre-sized to match initial_tris).  Returns true on success.
bool usd_load_frame(UsdAnimHandle* h, float time,
                    std::vector<Triangle>& triangles_out);

// Check if the stage has any time-varying data (animated points/xforms).
bool usd_has_animation(UsdAnimHandle* h);

// Release the stage and all cached data.
void usd_anim_close(UsdAnimHandle* h);
