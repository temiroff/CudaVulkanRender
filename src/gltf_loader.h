#pragma once
#include "scene.h"
#include <vector>
#include <string>
#include <cstdint>

// ─────────────────────────────────────────────
//  Raw image decoded from a glTF asset
// ─────────────────────────────────────────────

struct TextureImage {
    std::vector<uint8_t> pixels; // RGBA8, row-major, width*height*4 bytes
    int  width  = 0;
    int  height = 0;
    bool srgb   = false; // true → baseColor/emissive (need sRGB→linear on upload)
};

// ─────────────────────────────────────────────
//  Mesh object (one per glTF mesh primitive)
// ─────────────────────────────────────────────

struct MeshObject {
    int    obj_id;          // matches Triangle::obj_id
    float3 centroid;        // world-space AABB centroid (updated on move)
    char   name[128];       // node name for display
    bool   hidden = false;  // excluded from render when true
    bool   environment = false; // world-level prop (ground, walls) — skip camera auto-fit
};

// ─────────────────────────────────────────────
//  glTF / glb loader
// ─────────────────────────────────────────────

// Parse a .gltf (ASCII) or .glb (binary) file.
// On success returns true and populates the output vectors:
//   triangles_out  — flat, world-space triangle list (each tri has obj_id)
//   materials_out  — PBR materials mapped to GpuMaterial
//   textures_out   — decoded RGBA8 images (index matches GpuMaterial tex fields)
//   objects_out    — one MeshObject per glTF mesh primitive
// The last entry in materials_out is always a default grey Lambertian.
// On failure prints the tinygltf error to stderr and returns false.
bool gltf_load(const std::string&          path,
               std::vector<Triangle>&      triangles_out,
               std::vector<GpuMaterial>&   materials_out,
               std::vector<TextureImage>&  textures_out,
               std::vector<MeshObject>&    objects_out);
