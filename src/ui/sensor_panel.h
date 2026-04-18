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
    // Attachment: -1 = end-effector (default), else index into urdf_joint_info.
    int          attach_joint = -1;
    // Offset from the attachment origin to the camera, in the *derived* camera
    // frame (x = right, y = up, z = forward). Forward is toward the scene.
    // Default = {0,0,0}: camera starts exactly at the joint origin, so the
    // gizmo sits on the joint before any user calibration.
    float        offset[3]    = {0.f, 0.f, 0.f};
    // Rotation offsets (deg), applied to the derived base frame:
    //   rot_deg[0] = yaw   (around up),
    //   rot_deg[1] = pitch (around right),
    //   rot_deg[2] = roll  (around forward).
    float        rot_deg[3]   = {0.f, 0.f, 0.f};
    int          width        = 320;
    int          height       = 240;
    SensorCorner corner       = SensorCorner::TopRight;
    // Debug: draw a wireframe camera body + frustum in the viewport.
    bool         show_gizmo   = false;

    // Internal — GPU-side render target. Allocated lazily on first render.
    cudaArray_t           d_array = nullptr;
    cudaSurfaceObject_t   surface = 0;
    int                   alloc_w = 0;
    int                   alloc_h = 0;
    RasterizerState       raster{};

    // Cached axis mapping (joint-local) for the current attachment. Recomputed
    // only when `attach_joint` changes, so the camera's basis strictly follows
    // joint rotation without flipping when local axes cross alignment boundaries.
    // axis_cached_for = sentinel value meaning "uninitialized".
    int   axis_cached_for = -999;
    int   fwd_ax          = 2;   // 0,1,2 → local X,Y,Z
    int   up_ax           = 1;
    float fwd_sign        = 1.f;
    float up_sign         = 1.f;
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

// Compute the camera's world-space frame (origin + orthonormal basis: right,
// up, forward). Useful for drawing a viewport gizmo. Returns false if the
// articulation is missing. The returned frame matches what the sensor renders
// from — "dead attached" to the chosen joint/EE with the configured offset.
bool sensor_get_world_frame(const SensorState& state, UrdfArticulation* handle,
                            float3& out_origin, float3& out_right,
                            float3& out_up, float3& out_forward);

// CUDA-side blit helper (implemented in sensor_blit.cu).
void sensor_blit_corner(cudaSurfaceObject_t dst, int dst_w, int dst_h,
                        cudaArray_t src_array, int src_w, int src_h,
                        int off_x, int off_y);
