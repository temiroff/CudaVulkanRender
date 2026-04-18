#include "sensor_panel.h"
#include <imgui.h>
#include <cmath>

// ── GPU render-target management ─────────────────────────────────────────────
// Sensor output is a standalone CUDA array (not Vulkan-interop — it's never
// presented directly, only blitted into the main surface). Reallocated when
// the configured resolution changes.
static void sensor_ensure_target(SensorState& s)
{
    if (s.d_array && s.alloc_w == s.width && s.alloc_h == s.height) return;

    if (s.surface)  { cudaDestroySurfaceObject(s.surface); s.surface = 0; }
    if (s.d_array)  { cudaFreeArray(s.d_array);            s.d_array = nullptr; }

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&s.d_array, &fmt, s.width, s.height, cudaArraySurfaceLoadStore);

    cudaResourceDesc rd{};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = s.d_array;
    cudaCreateSurfaceObject(&s.surface, &rd);

    s.alloc_w = s.width;
    s.alloc_h = s.height;
}

void sensor_state_destroy(SensorState& state)
{
    if (state.surface) { cudaDestroySurfaceObject(state.surface); state.surface = 0; }
    if (state.d_array) { cudaFreeArray(state.d_array);            state.d_array = nullptr; }
    rasterizer_destroy(state.raster);
    state.alloc_w = state.alloc_h = 0;
}

// ── Build a Camera from the end-effector transform ───────────────────────────
// EE-local +Z points "forward" out of the gripper (URDF/MJCF convention), +Y
// is "up" in the hand frame. Camera sits at EE origin + forward_m along local
// +Z so it shoots out between the fingers, facing the same direction.
static Camera sensor_camera_from_ee(UrdfArticulation* h,
                                    float forward_m, float fov_deg,
                                    int w, int h_px)
{
    float m[16];
    urdf_fk_ee_transform(h, m);

    float3 origin = make_float3(m[3],  m[7],  m[11]);
    float3 fwd    = make_float3(m[2],  m[6],  m[10]);  // local +Z
    float3 up     = make_float3(m[1],  m[5],  m[9]);   // local +Y

    float3 eye    = make_float3(origin.x + fwd.x * forward_m,
                                 origin.y + fwd.y * forward_m,
                                 origin.z + fwd.z * forward_m);
    float3 target = make_float3(eye.x + fwd.x, eye.y + fwd.y, eye.z + fwd.z);

    float aspect = (h_px > 0) ? (float)w / (float)h_px : 1.f;
    return Camera::make(eye, target, up, fov_deg, aspect,
                        /*aperture=*/0.f, /*focus_dist=*/1.f);
}

// ── UI ───────────────────────────────────────────────────────────────────────
void sensor_panel_draw(SensorState& state, UrdfArticulation* handle)
{
    if (!ImGui::Begin("Sensors")) { ImGui::End(); return; }

    if (!handle) {
        ImGui::TextDisabled("No URDF loaded");
        ImGui::End();
        return;
    }

    ImGui::TextUnformatted("Gripper camera");
    ImGui::Checkbox("Enabled", &state.enabled);

    ImGui::SetNextItemWidth(-1.f);
    ImGui::SliderFloat("FOV", &state.fov_deg, 20.f, 120.f, "%.0f deg");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::SliderFloat("Forward offset", &state.forward_m, 0.00f, 0.30f, "%.3f m");

    const int sizes[][2] = { {160, 120}, {320, 240}, {480, 360}, {640, 480} };
    int cur = 1;
    for (int i = 0; i < 4; ++i)
        if (state.width == sizes[i][0] && state.height == sizes[i][1]) cur = i;
    const char* labels[] = { "160x120", "320x240", "480x360", "640x480" };
    ImGui::SetNextItemWidth(-1.f);
    if (ImGui::Combo("Resolution", &cur, labels, 4)) {
        state.width  = sizes[cur][0];
        state.height = sizes[cur][1];
    }

    int corner_idx = (int)state.corner;
    const char* corners[] = { "Top Left", "Top Right", "Bottom Left", "Bottom Right" };
    ImGui::SetNextItemWidth(-1.f);
    if (ImGui::Combo("Corner", &corner_idx, corners, 4))
        state.corner = (SensorCorner)corner_idx;

    ImGui::End();
}

// ── Render + blit ────────────────────────────────────────────────────────────
void sensor_render_and_blit(SensorState& state,
                            UrdfArticulation* handle,
                            cudaSurfaceObject_t main_surface,
                            int main_w, int main_h,
                            Triangle* d_triangles, int num_triangles,
                            GpuMaterial* d_materials, int num_materials,
                            cudaTextureObject_t* d_textures,
                            float3* d_obj_colors, int num_obj_colors,
                            int color_mode, float3 bg_color,
                            float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    if (!state.enabled || !handle || num_triangles <= 0) return;
    if (state.width <= 0 || state.height <= 0) return;

    sensor_ensure_target(state);
    if (!state.surface) return;

    Camera cam = sensor_camera_from_ee(handle, state.forward_m, state.fov_deg,
                                        state.width, state.height);

    rasterizer_render(state.raster, state.surface, state.width, state.height,
                      cam,
                      d_triangles, num_triangles,
                      d_materials, num_materials, d_textures,
                      d_obj_colors, num_obj_colors,
                      color_mode, bg_color,
                      key_dir, fill_dir, rim_dir);

    // Position the PiP in the chosen corner with an 8 px margin.
    const int M = 8;
    int off_x = 0, off_y = 0;
    switch (state.corner) {
    case SensorCorner::TopLeft:     off_x = M;                      off_y = M; break;
    case SensorCorner::TopRight:    off_x = main_w - state.width - M; off_y = M; break;
    case SensorCorner::BottomLeft:  off_x = M;                      off_y = main_h - state.height - M; break;
    case SensorCorner::BottomRight: off_x = main_w - state.width - M; off_y = main_h - state.height - M; break;
    }

    sensor_blit_corner(main_surface, main_w, main_h,
                       state.d_array, state.width, state.height,
                       off_x, off_y);
}
