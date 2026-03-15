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
