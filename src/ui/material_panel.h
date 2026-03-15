#pragma once
#include "../scene.h"
#include "../gltf_loader.h"
#include <vector>
#include <cstdint>
#include <string>

// ── What changed this frame ───────────────────────────────────────────────────
struct MaterialPanelChange {
    bool materials = false;  // PBR sliders changed
    bool triangles = false;  // material assignment changed

    // Custom shader output (CUDA or Slang) — all four maps, or all empty
    std::vector<uint8_t> shader_base_pixels;      // RGB=base_color,    A=255
    std::vector<uint8_t> shader_mr_pixels;         // R=0,G=rough,B=metal,A=255
    std::vector<uint8_t> shader_emissive_pixels;   // RGB=emissive,      A=255
    std::vector<uint8_t> shader_normal_pixels;     // RGB=normal_ts[0,1],A=255

    int shader_w   = 0;
    int shader_h   = 0;
    int shader_mat = -1;

    // "Reset to BSDF" — restores original material before shader was applied
    bool shader_reset     = false;
    int  shader_reset_mat = -1;

    explicit operator bool() const {
        return materials || triangles || !shader_base_pixels.empty() || shader_reset;
    }
};

MaterialPanelChange material_panel_draw(
    int                             selected_obj,
    std::vector<Triangle>&          all_prims,
    std::vector<Triangle>&          prims,
    const std::vector<MeshObject>&  objects,
    std::vector<GpuMaterial>&       host_mats,
    int                             frame_count);
