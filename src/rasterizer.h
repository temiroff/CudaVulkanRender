#pragma once

#include "scene.h"
#include <cuda_runtime.h>

struct RasterizerState {
    float*    d_zbuffer             = nullptr;
    int*      d_large_tris          = nullptr;   // indices of screen-filling triangles
    int*      d_large_count         = nullptr;   // atomic counter
    Triangle* d_large_clipped_tris  = nullptr;   // generated near-plane-clipped large triangles
    int*      d_large_clipped_count = nullptr;   // atomic counter
    int       zbuf_w                = 0;
    int       zbuf_h                = 0;
};

void rasterizer_init(RasterizerState& state);
void rasterizer_resize(RasterizerState& state, int w, int h);
void rasterizer_destroy(RasterizerState& state);

void rasterizer_render(
    RasterizerState& state,
    cudaSurfaceObject_t surface, int width, int height,
    Camera cam,
    Triangle* d_triangles, int num_triangles,
    GpuMaterial* d_materials, int num_materials,
    cudaTextureObject_t* d_textures,
    float3* d_obj_colors, int num_obj_colors,
    int color_mode, float3 bg_color,
    float3 key_dir, float3 fill_dir, float3 rim_dir);
