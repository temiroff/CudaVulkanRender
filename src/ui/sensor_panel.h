#pragma once

#include "../scene.h"
#include "../urdf_loader.h"
#include "../rasterizer.h"
#include <cuda_runtime.h>

// ── Gripper camera sensor ────────────────────────────────────────────────────
// A small rasterized view attached to the articulation's end-effector, blitted
// as a PiP overlay on the main viewport. Keeps things minimal: one sensor,
// rasterizer only, fixed corner placement (user pickable).

enum class SensorCorner : int {
    TopLeft = 0, TopRight = 1, BottomLeft = 2, BottomRight = 3
};

struct SensorState {
    bool         enabled      = false;
    float        fov_deg      = 60.f;
    float        forward_m    = 0.10f;    // distance from ee origin along local +Z
    int          width        = 320;
    int          height       = 240;
    SensorCorner corner       = SensorCorner::TopRight;

    // Internal — GPU-side render target. Allocated lazily on first render.
    cudaArray_t           d_array = nullptr;
    cudaSurfaceObject_t   surface = 0;
    int                   alloc_w = 0;
    int                   alloc_h = 0;
    RasterizerState       raster{};
};

// Draw the Sensors ImGui panel (enable/disable, fov, offset, corner).
void sensor_panel_draw(SensorState& state, UrdfArticulation* handle);

// Render the sensor into its private surface, then blit it with a white
// border into a corner of `main_surface`. No-op if disabled or no handle.
void sensor_render_and_blit(SensorState& state,
                            UrdfArticulation* handle,
                            cudaSurfaceObject_t main_surface,
                            int main_w, int main_h,
                            Triangle* d_triangles, int num_triangles,
                            GpuMaterial* d_materials, int num_materials,
                            cudaTextureObject_t* d_textures,
                            float3* d_obj_colors, int num_obj_colors,
                            int color_mode, float3 bg_color,
                            float3 key_dir, float3 fill_dir, float3 rim_dir);

// Release GPU resources.
void sensor_state_destroy(SensorState& state);

// CUDA-side blit helper (implemented in sensor_blit.cu).
void sensor_blit_corner(cudaSurfaceObject_t dst, int dst_w, int dst_h,
                        cudaArray_t src_array, int src_w, int src_h,
                        int off_x, int off_y);
