#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <cuda_runtime.h>

#include "vulkan_context.h"
#include "cuda_interop.h"
#include "ui/gpu_arch_window.h"
#include "ui/gpu_arch_sm_tracker.h"
#include "cupti_profiler.h"
#include "pathtracer.h"
#include "bvh.h"
#include "scene.h"
#include "gpu_textures.h"
#include "gltf_loader.h"
#include "usd_loader.h"
#include "urdf_loader.h"
#include "mjcf_loader.h"
#include "restir.h"
#include "rasterizer.h"
#include "hdri.h"
#include "denoiser.h"
#include "optix_denoiser.h"
#include "optix_renderer.h"
#include "dlss_upscaler.h"
#include "post_process.h"
#include "nim_vlm.h"
#include "nim_cosmos.h"
#include "slang_shader.h"
#include "stb_image_write.h"
#include "batch_processor.h"
#include <tinyexr.h>
#include "ui/viewport.h"
#include "ui/control_panel.h"
#include "ui/outliner.h"
#include "ui/stats_panel.h"
#include "ui/material_panel.h"
#include "ui/anim_panel.h"
#include "ui/articulation_panel.h"
#include "ui/robot_demo_panel.h"
#include "ui/hydra_preview.h"

#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <cmath>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

static bool env_flag_enabled(const char* name)
{
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || !value || !value[0]) {
        free(value);
        return false;
    }
    const char c = value[0];
    free(value);
    return c == '1' || c == 'y' || c == 'Y' ||
           c == 't' || c == 'T' || c == 'o' || c == 'O';
}

static LONG WINAPI win_unhandled_exception_filter(EXCEPTION_POINTERS* ep)
{
    if (ep && ep->ExceptionRecord) {
        fprintf(stderr, "[fatal] SEH 0x%08X at %p\n",
                (unsigned)ep->ExceptionRecord->ExceptionCode,
                ep->ExceptionRecord->ExceptionAddress);
    } else {
        fprintf(stderr, "[fatal] SEH (no exception record)\n");
    }
    fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

static bool safe_cupti_init(CuptiProfiler& profiler, int device_index)
{
#ifdef _WIN32
    __try {
        return profiler.init(device_index);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::cerr << "[cupti] init crashed (SEH); disabling CUPTI profiler\n";
        return false;
    }
#else
    return profiler.init(device_index);
#endif
}

static void safe_cupti_begin_frame(CuptiProfiler& profiler)
{
#ifdef _WIN32
    __try {
        profiler.begin_frame();
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::cerr << "[cupti] begin_frame crashed (SEH); skipping CUPTI sample\n";
    }
#else
    profiler.begin_frame();
#endif
}

static void safe_cupti_end_frame(CuptiProfiler& profiler)
{
#ifdef _WIN32
    __try {
        profiler.end_frame();
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::cerr << "[cupti] end_frame crashed (SEH); skipping CUPTI sample\n";
    }
#else
    profiler.end_frame();
#endif
}

static bool safe_hydra_preview_init(
    HydraPreviewState& hp,
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkQueue graphics_queue,
    uint32_t graphics_queue_family,
    VkCommandPool command_pool)
{
#ifdef _WIN32
    __try {
        return hydra_preview_init(
            hp, device, physical_device, graphics_queue, graphics_queue_family, command_pool);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::cerr << "[hydra_preview] init crashed (SEH), preview disabled\n";
        return false;
    }
#else
    return hydra_preview_init(
        hp, device, physical_device, graphics_queue, graphics_queue_family, command_pool);
#endif
}

static bool safe_hydra_preview_tick(
    HydraPreviewState& hp,
    const AnimPanelState& anim,
    int width,
    int height)
{
#ifdef _WIN32
    __try {
        return hydra_preview_tick(hp, anim, width, height);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::cerr << "[hydra_preview] tick crashed (SEH), closing preview\n";
        return false;
    }
#else
    return hydra_preview_tick(hp, anim, width, height);
#endif
}

static bool safe_hydra_preview_load(HydraPreviewState& hp, const std::string& path)
{
#ifdef _WIN32
    __try {
        hydra_preview_load(hp, path);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::cerr << "[hydra_preview] load crashed (SEH), closing preview\n";
        return false;
    }
#else
    hydra_preview_load(hp, path);
    return true;
#endif
}

static void safe_hydra_preview_destroy(HydraPreviewState& hp)
{
#ifdef _WIN32
    __try {
        hydra_preview_destroy(hp);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::cerr << "[hydra_preview] destroy crashed (SEH)\n";
    }
#else
    hydra_preview_destroy(hp);
#endif
}

// ─────────────────────────────────────────────
//  Scene
// ─────────────────────────────────────────────

static std::vector<Sphere> make_scene() {
    std::vector<Sphere> s;
    s.push_back({ make_float3(0.f, -1000.f, 0.f), 1000.f,
        { make_float3(0.5f, 0.5f, 0.5f), {}, 0.f, 0.f, MatType::Lambertian } });
    s.push_back({ make_float3(0.f, 1.f, 0.f), 1.f,
        { {}, {}, 0.f, 1.5f, MatType::Dielectric } });
    s.push_back({ make_float3(-4.f, 1.f, 0.f), 1.f,
        { make_float3(0.4f, 0.2f, 0.1f), {}, 0.f, 0.f, MatType::Lambertian } });
    s.push_back({ make_float3(4.f, 1.f, 0.f), 1.f,
        { make_float3(0.7f, 0.6f, 0.5f), {}, 0.f, 0.f, MatType::Metal } });
    s.push_back({ make_float3(0.f, 5.f, -3.f), 1.5f,
        { {}, make_float3(8.f, 8.f, 7.f), 0.f, 0.f, MatType::Emissive } });
    for (int a = -5; a < 5; ++a) {
        for (int b = -5; b < 5; ++b) {
            float cx = a + 0.9f * sinf((float)(a * 7 + b * 13));
            float cz = b + 0.9f * cosf((float)(a * 11 + b * 5));
            if (sqrtf(cx*cx + cz*cz) < 2.f) continue;
            int t = (a + b * 11) % 3;
            Material mat{};
            if (t == 0)
                mat = { make_float3(fabsf(sinf(a*1.f)), fabsf(cosf(b*1.f)), 0.5f), {}, 0.f, 0.f, MatType::Lambertian };
            else if (t == 1)
                mat = { make_float3(0.5f+0.5f*fabsf(sinf(a*2.f)), 0.5f+0.5f*fabsf(cosf(b*2.f)), 0.3f),
                        {}, fabsf(sinf((float)(a+b)))*0.5f, 0.f, MatType::Metal };
            else
                mat = { {}, {}, 0.f, 1.5f, MatType::Dielectric };
            s.push_back({ make_float3(cx, 0.2f, cz), 0.2f, mat });
        }
    }
    return s;
}

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

static int pick_sphere(const std::vector<Sphere>& prims, float3 origin, float3 dir) {
    float best_t = 1e20f;
    int   best   = -1;
    for (int i = 0; i < (int)prims.size(); ++i) {
        if (prims[i].radius > 100.f) continue;   // skip environment/ground sphere
        float3 oc = { origin.x - prims[i].center.x,
                      origin.y - prims[i].center.y,
                      origin.z - prims[i].center.z };
        float a  = dot(dir, dir);
        float b  = dot(oc, dir);
        float c  = dot(oc, oc) - prims[i].radius * prims[i].radius;
        float disc = b*b - a*c;
        if (disc < 0.f) continue;
        float t = (-b - sqrtf(disc)) / a;
        if (t < 1e-3f) t = (-b + sqrtf(disc)) / a;
        if (t > 1e-3f && t < best_t) { best_t = t; best = i; }
    }
    return best;
}

// Möller–Trumbore ray-triangle test (CPU, for picking).
// Returns the index into objects[] of the closest hit mesh object, or -1.
static int pick_mesh_object(const std::vector<Triangle>& tris,
                             const std::vector<MeshObject>& objects,
                             float3 origin, float3 dir)
{
    float best_t  = 1e20f;
    int   best_id = -1;

    for (const Triangle& tri : tris) {
        float3 e1 = { tri.v1.x - tri.v0.x, tri.v1.y - tri.v0.y, tri.v1.z - tri.v0.z };
        float3 e2 = { tri.v2.x - tri.v0.x, tri.v2.y - tri.v0.y, tri.v2.z - tri.v0.z };
        float3 h  = cross(dir, e2);
        float  a  = dot(e1, h);
        if (fabsf(a) < 1e-8f) continue;
        float  f  = 1.f / a;
        float3 s  = { origin.x - tri.v0.x, origin.y - tri.v0.y, origin.z - tri.v0.z };
        float  u  = f * dot(s, h);
        if (u < 0.f || u > 1.f) continue;
        float3 q  = cross(s, e1);
        float  v  = f * dot(dir, q);
        if (v < 0.f || u + v > 1.f) continue;
        float  t  = f * dot(e2, q);
        if (t < 1e-3f || t >= best_t) continue;
        best_t  = t;
        best_id = tri.obj_id;
    }

    if (best_id < 0) return -1;
    // Find index in objects[] matching best_id
    for (int i = 0; i < (int)objects.size(); ++i)
        if (objects[i].obj_id == best_id) return i;
    return -1;
}

// Ray-cast to find the exact world-space surface hit point.
// Tests mesh triangles first (if any), then spheres. Returns true on hit.
static bool pick_surface_point(
    const std::vector<Triangle>& tris,
    const std::vector<Sphere>&   spheres,
    float3 origin, float3 dir,
    float3& hit_point)
{
    float best_t = 1e20f;
    bool  found  = false;

    // Mesh triangles — Möller–Trumbore
    for (const Triangle& tri : tris) {
        float3 e1 = { tri.v1.x-tri.v0.x, tri.v1.y-tri.v0.y, tri.v1.z-tri.v0.z };
        float3 e2 = { tri.v2.x-tri.v0.x, tri.v2.y-tri.v0.y, tri.v2.z-tri.v0.z };
        float3 h  = cross(dir, e2);
        float  a  = dot(e1, h);
        if (fabsf(a) < 1e-8f) continue;
        float  f = 1.f / a;
        float3 s = { origin.x-tri.v0.x, origin.y-tri.v0.y, origin.z-tri.v0.z };
        float  u = f * dot(s, h);
        if (u < 0.f || u > 1.f) continue;
        float3 q = cross(s, e1);
        float  v = f * dot(dir, q);
        if (v < 0.f || u+v > 1.f) continue;
        float  t = f * dot(e2, q);
        if (t > 1e-3f && t < best_t) { best_t = t; found = true; }
    }

    // Spheres (skip environment)
    for (const Sphere& sp : spheres) {
        if (sp.radius > 100.f) continue;
        float3 oc = { origin.x-sp.center.x, origin.y-sp.center.y, origin.z-sp.center.z };
        float  a  = dot(dir, dir);
        float  b  = dot(oc, dir);
        float  c  = dot(oc, oc) - sp.radius * sp.radius;
        float  disc = b*b - a*c;
        if (disc < 0.f) continue;
        float  t = (-b - sqrtf(disc)) / a;
        if (t < 1e-3f) t = (-b + sqrtf(disc)) / a;
        if (t > 1e-3f && t < best_t) { best_t = t; found = true; }
    }

    if (found)
        hit_point = { origin.x + dir.x*best_t,
                      origin.y + dir.y*best_t,
                      origin.z + dir.z*best_t };
    return found;
}

// Project world point → screen pixel. Returns forward depth (negative = behind camera).
static float world_to_screen(const Camera& cam, float3 p,
                              ImVec2 vp_origin, ImVec2 vp_size,
                              float vfov_deg, float aspect, ImVec2& out)
{
    float ox = p.x - cam.origin.x, oy = p.y - cam.origin.y, oz = p.z - cam.origin.z;
    float depth = -(ox*cam.w.x + oy*cam.w.y + oz*cam.w.z); // -cam.w = forward
    if (depth <= 0.f) { out = {0,0}; return -1.f; }
    float sx = ox*cam.u.x + oy*cam.u.y + oz*cam.u.z;
    float sy = ox*cam.v.x + oy*cam.v.y + oz*cam.v.z;
    float half_h = tanf(vfov_deg * 0.5f * 3.14159265f / 180.f);
    float u = sx / (depth * 2.f * half_h * aspect) + 0.5f;
    float v = sy / (depth * 2.f * half_h) + 0.5f;
    out.x = vp_origin.x + u * vp_size.x;
    out.y = vp_origin.y + (1.f - v) * vp_size.y; // flip Y for screen
    return depth;
}

static float seg_dist_sq(float px, float py,
                          float ax, float ay, float bx, float by)
{
    float dx = bx-ax, dy = by-ay;
    float len2 = dx*dx + dy*dy;
    float t = len2 > 0.f ? ((px-ax)*dx + (py-ay)*dy) / len2 : 0.f;
    t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    float ex = px - (ax+t*dx), ey = py - (ay+t*dy);
    return ex*ex + ey*ey;
}

// Draw XYZ translate gizmo. Returns active axis (1=X,2=Y,3=Z, 4=center/camera-space, 0=none).
// out_delta: world-space movement to apply this frame.
static int draw_move_gizmo(ImDrawList* dl, const Camera& cam, float3 center,
                            float vfov, float aspect,
                            ImVec2 vp_origin, ImVec2 vp_size,
                            float3& out_delta)
{
    static int drag_axis = 0;
    out_delta = {0.f, 0.f, 0.f};

    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) drag_axis = 0;

    ImVec2 cs;
    float depth = world_to_screen(cam, center, vp_origin, vp_size, vfov, aspect, cs);
    if (depth <= 0.f) return 0;

    const float ARROW_PX = 80.f;
    const float HIT_PX   = 8.f;
    const float TIP_PX   = 10.f;
    const float CENTER_SZ = 7.f;

    float half_h = tanf(vfov * 0.5f * 3.14159265f / 180.f);
    float scale  = depth * 2.f * half_h / vp_size.y * ARROW_PX;

    float3 axis_dir[4]  = { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} };
    ImU32  axis_col[4]  = { 0,
        IM_COL32(220, 60,  60, 255),
        IM_COL32( 60,200,  60, 255),
        IM_COL32( 60,120, 220, 255) };

    ImVec2 tip[4];
    for (int i = 1; i <= 3; ++i) {
        float3 w = {
            center.x + axis_dir[i].x * scale,
            center.y + axis_dir[i].y * scale,
            center.z + axis_dir[i].z * scale };
        world_to_screen(cam, w, vp_origin, vp_size, vfov, aspect, tip[i]);
    }

    ImVec2 mouse = ImGui::GetIO().MousePos;
    int hovered = 0;
    if (drag_axis == 0) {
        // Check center square first
        if (fabsf(mouse.x - cs.x) < CENTER_SZ && fabsf(mouse.y - cs.y) < CENTER_SZ)
            hovered = 4;
        else {
            for (int i = 1; i <= 3; ++i) {
                if (seg_dist_sq(mouse.x, mouse.y,
                    cs.x, cs.y, tip[i].x, tip[i].y) < HIT_PX*HIT_PX) {
                    hovered = i; break;
                }
            }
        }
    }

    if (hovered > 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        drag_axis = hovered;

    int display = drag_axis > 0 ? drag_axis : hovered;

    // Draw axes
    for (int i = 1; i <= 3; ++i) {
        ImU32 c = (i == display) ? IM_COL32(255, 255, 100, 255) : axis_col[i];
        dl->AddLine(cs, tip[i], c, 2.5f);
        float ax = tip[i].x - cs.x, ay = tip[i].y - cs.y;
        float len = sqrtf(ax*ax + ay*ay);
        if (len > 0.f) {
            ax /= len; ay /= len;
            float px = -ay, py = ax;
            dl->AddTriangleFilled(
                tip[i],
                { tip[i].x - ax*TIP_PX + px*(TIP_PX*0.4f), tip[i].y - ay*TIP_PX + py*(TIP_PX*0.4f) },
                { tip[i].x - ax*TIP_PX - px*(TIP_PX*0.4f), tip[i].y - ay*TIP_PX - py*(TIP_PX*0.4f) },
                c);
        }
    }
    // Center square (camera-space translate)
    ImU32 cc = (display == 4) ? IM_COL32(255, 255, 100, 255) : IM_COL32(255, 255, 255, 180);
    dl->AddRectFilled({ cs.x - CENTER_SZ, cs.y - CENTER_SZ },
                      { cs.x + CENTER_SZ, cs.y + CENTER_SZ }, cc);

    // Apply drag
    ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (drag_axis > 0 && (delta.x != 0.f || delta.y != 0.f)) {
        float wpp = scale / ARROW_PX;
        if (drag_axis == 4) {
            // Camera-space translate: mouse X → cam.u, mouse Y → -cam.v
            out_delta = {
                (cam.u.x * delta.x - cam.v.x * delta.y) * wpp,
                (cam.u.y * delta.x - cam.v.y * delta.y) * wpp,
                (cam.u.z * delta.x - cam.v.z * delta.y) * wpp };
        } else {
            float3 ad = axis_dir[drag_axis];
            float sax = tip[drag_axis].x - cs.x;
            float say = tip[drag_axis].y - cs.y;
            float slen = sqrtf(sax*sax + say*say);
            if (slen > 0.f) {
                float proj = (delta.x * sax + delta.y * say) / slen;
                float move = proj * wpp;
                out_delta = { ad.x * move, ad.y * move, ad.z * move };
            }
        }
    }

    return (drag_axis > 0) ? drag_axis : hovered;
}

// Draw XYZ rotation gizmo (circles per axis). Returns active axis (1=X,2=Y,3=Z, 0=none).
// out_angles: rotation in radians around each world axis this frame.
static int draw_rotate_gizmo(ImDrawList* dl, const Camera& cam, float3 center,
                              float vfov, float aspect,
                              ImVec2 vp_origin, ImVec2 vp_size,
                              float3& out_angles)
{
    static int drag_axis = 0;
    out_angles = {0.f, 0.f, 0.f};

    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) drag_axis = 0;

    ImVec2 cs;
    float depth = world_to_screen(cam, center, vp_origin, vp_size, vfov, aspect, cs);
    if (depth <= 0.f) return 0;

    const float RADIUS_PX = 70.f;
    const float HIT_PX    = 10.f;
    const int   SEGMENTS  = 48;

    float3 axis_dir[4] = { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} };
    ImU32  axis_col[4] = { 0,
        IM_COL32(220, 60,  60, 255),
        IM_COL32( 60,200,  60, 255),
        IM_COL32( 60,120, 220, 255) };

    float half_h = tanf(vfov * 0.5f * 3.14159265f / 180.f);
    float scale  = depth * 2.f * half_h / vp_size.y * RADIUS_PX;

    // Draw circles and detect hover
    ImVec2 mouse = ImGui::GetIO().MousePos;
    int hovered = 0;

    for (int ax = 1; ax <= 3; ++ax) {
        // Two perpendicular directions to the axis
        float3 u_dir, v_dir;
        if (ax == 1)      { u_dir = {0,1,0}; v_dir = {0,0,1}; }
        else if (ax == 2) { u_dir = {1,0,0}; v_dir = {0,0,1}; }
        else              { u_dir = {1,0,0}; v_dir = {0,1,0}; }

        ImVec2 pts[SEGMENTS + 1];
        for (int s = 0; s <= SEGMENTS; ++s) {
            float angle = (float)s / SEGMENTS * 2.f * 3.14159265f;
            float3 wp = {
                center.x + (u_dir.x * cosf(angle) + v_dir.x * sinf(angle)) * scale,
                center.y + (u_dir.y * cosf(angle) + v_dir.y * sinf(angle)) * scale,
                center.z + (u_dir.z * cosf(angle) + v_dir.z * sinf(angle)) * scale };
            world_to_screen(cam, wp, vp_origin, vp_size, vfov, aspect, pts[s]);
        }

        // Check hover distance to circle
        if (drag_axis == 0) {
            for (int s = 0; s < SEGMENTS; ++s) {
                if (seg_dist_sq(mouse.x, mouse.y,
                    pts[s].x, pts[s].y, pts[s+1].x, pts[s+1].y) < HIT_PX*HIT_PX) {
                    hovered = ax; break;
                }
            }
        }

        int display = drag_axis > 0 ? drag_axis : hovered;
        ImU32 c = (ax == display) ? IM_COL32(255, 255, 100, 255) : axis_col[ax];
        for (int s = 0; s < SEGMENTS; ++s)
            dl->AddLine(pts[s], pts[s+1], c, 2.0f);
    }

    dl->AddCircleFilled(cs, 4.f, IM_COL32(255, 255, 255, 200));

    if (hovered > 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        drag_axis = hovered;

    // Compute rotation from mouse drag
    ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (drag_axis > 0 && (delta.x != 0.f || delta.y != 0.f)) {
        // Rotation speed: screen pixels → radians
        float rot_speed = 0.01f;
        float rot = (delta.x + delta.y) * rot_speed;
        if (drag_axis == 1) out_angles.x = rot;
        if (drag_axis == 2) out_angles.y = rot;
        if (drag_axis == 3) out_angles.z = rot;
    }

    return (drag_axis > 0) ? drag_axis : hovered;
}

// Draw XYZ scale gizmo (lines with boxes). Returns active axis (1=X,2=Y,3=Z, 4=uniform, 0=none).
// out_scale: scale delta per axis this frame (add to 1.0).
static int draw_scale_gizmo(ImDrawList* dl, const Camera& cam, float3 center,
                             float vfov, float aspect,
                             ImVec2 vp_origin, ImVec2 vp_size,
                             float3& out_scale)
{
    static int drag_axis = 0;
    out_scale = {0.f, 0.f, 0.f};

    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) drag_axis = 0;

    ImVec2 cs;
    float depth = world_to_screen(cam, center, vp_origin, vp_size, vfov, aspect, cs);
    if (depth <= 0.f) return 0;

    const float ARROW_PX  = 70.f;
    const float HIT_PX    = 8.f;
    const float BOX_PX    = 5.f;
    const float CENTER_SZ = 7.f;

    float half_h = tanf(vfov * 0.5f * 3.14159265f / 180.f);
    float scale  = depth * 2.f * half_h / vp_size.y * ARROW_PX;

    float3 axis_dir[4] = { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} };
    ImU32  axis_col[4] = { 0,
        IM_COL32(220, 60,  60, 255),
        IM_COL32( 60,200,  60, 255),
        IM_COL32( 60,120, 220, 255) };

    ImVec2 tip[4];
    for (int i = 1; i <= 3; ++i) {
        float3 w = {
            center.x + axis_dir[i].x * scale,
            center.y + axis_dir[i].y * scale,
            center.z + axis_dir[i].z * scale };
        world_to_screen(cam, w, vp_origin, vp_size, vfov, aspect, tip[i]);
    }

    ImVec2 mouse = ImGui::GetIO().MousePos;
    int hovered = 0;
    if (drag_axis == 0) {
        // Center square first (uniform scale)
        if (fabsf(mouse.x - cs.x) < CENTER_SZ && fabsf(mouse.y - cs.y) < CENTER_SZ)
            hovered = 4;
        else {
            for (int i = 1; i <= 3; ++i) {
                if (seg_dist_sq(mouse.x, mouse.y,
                    cs.x, cs.y, tip[i].x, tip[i].y) < HIT_PX*HIT_PX) {
                    hovered = i; break;
                }
            }
        }
    }

    if (hovered > 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        drag_axis = hovered;

    int display = drag_axis > 0 ? drag_axis : hovered;

    for (int i = 1; i <= 3; ++i) {
        ImU32 c = (i == display) ? IM_COL32(255, 255, 100, 255) : axis_col[i];
        dl->AddLine(cs, tip[i], c, 2.5f);
        dl->AddRectFilled(
            { tip[i].x - BOX_PX, tip[i].y - BOX_PX },
            { tip[i].x + BOX_PX, tip[i].y + BOX_PX }, c);
    }
    // Center square (uniform scale)
    ImU32 cc = (display == 4) ? IM_COL32(255, 255, 100, 255) : IM_COL32(255, 255, 255, 180);
    dl->AddRectFilled({ cs.x - CENTER_SZ, cs.y - CENTER_SZ },
                      { cs.x + CENTER_SZ, cs.y + CENTER_SZ }, cc);

    ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (drag_axis > 0 && (delta.x != 0.f || delta.y != 0.f)) {
        if (drag_axis == 4) {
            // Uniform scale: horizontal drag right = bigger
            float s = (delta.x - delta.y) * 0.003f;
            out_scale = { s, s, s };
        } else {
            float3 ad = axis_dir[drag_axis];
            float sax = tip[drag_axis].x - cs.x;
            float say = tip[drag_axis].y - cs.y;
            float slen = sqrtf(sax*sax + say*say);
            if (slen > 0.f) {
                float proj = (delta.x * sax + delta.y * say) / slen;
                float s = proj * 0.005f;
                if (drag_axis == 1) out_scale.x = s;
                if (drag_axis == 2) out_scale.y = s;
                if (drag_axis == 3) out_scale.z = s;
            }
        }
    }

    return (drag_axis > 0) ? drag_axis : hovered;
}

// ─────────────────────────���───────────────────
//  Silhouette edge detection (CPU, for selection outline)
// ────────��────────────────────────────────────

struct EdgeKey {
    int64_t x0,y0,z0, x1,y1,z1;
    bool operator==(const EdgeKey& o) const {
        return x0==o.x0&&y0==o.y0&&z0==o.z0&&x1==o.x1&&y1==o.y1&&z1==o.z1;
    }
};
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
        size_t h = 0;
        auto mix = [&](int64_t v){
            h ^= std::hash<int64_t>{}(v) + 0x9e3779b9ull + (h<<6) + (h>>2);
        };
        mix(k.x0);mix(k.y0);mix(k.z0);mix(k.x1);mix(k.y1);mix(k.z1);
        return h;
    }
};
struct EdgeInfo {
    float3 a, b;
    float3 fn[2];   // up to 2 adjacent face normals
    int    count = 0;
};

static EdgeKey make_edge_key(float3 a, float3 b) {
    // Quantise to 1/1000 world units so float equality is stable
    auto q = [](float v){ return (int64_t)(v * 1000.f); };
    int64_t ax=q(a.x),ay=q(a.y),az=q(a.z);
    int64_t bx=q(b.x),by=q(b.y),bz=q(b.z);
    // Canonical ordering so edge A→B and B→A map to the same key
    if (ax>bx||(ax==bx&&ay>by)||(ax==bx&&ay==by&&az>bz)) {
        std::swap(ax,bx); std::swap(ay,by); std::swap(az,bz);
    }
    return {ax,ay,az,bx,by,bz};
}

// Returns world-space silhouette edge pairs for a given obj_id.
// Falls back to AABB wireframe if the mesh is too large (> LIMIT tris).
static std::vector<std::pair<float3,float3>>
compute_silhouette(const std::vector<Triangle>& all_tris, int obj_id,
                   float3 cam_pos,
                   float3& aabb_mn_out, float3& aabb_mx_out)
{
    constexpr int LIMIT = 200000;  // fall back to AABB above this

    // Gather this object's triangles and its AABB
    float3 mn = make_float3( 1e30f, 1e30f, 1e30f);
    float3 mx = make_float3(-1e30f,-1e30f,-1e30f);
    int    cnt = 0;
    for (auto& t : all_tris) {
        if (t.obj_id != obj_id) continue;
        ++cnt;
        for (auto& v : {t.v0, t.v1, t.v2}) {
            if (v.x<mn.x) mn.x=v.x; if (v.x>mx.x) mx.x=v.x;
            if (v.y<mn.y) mn.y=v.y; if (v.y>mx.y) mx.y=v.y;
            if (v.z<mn.z) mn.z=v.z; if (v.z>mx.z) mx.z=v.z;
        }
    }
    aabb_mn_out = mn; aabb_mx_out = mx;

    std::vector<std::pair<float3,float3>> result;

    if (cnt > LIMIT) {
        // AABB wireframe fallback — 12 edges of the bounding box
        float3 corners[8] = {
            {mn.x,mn.y,mn.z},{mx.x,mn.y,mn.z},{mx.x,mx.y,mn.z},{mn.x,mx.y,mn.z},
            {mn.x,mn.y,mx.z},{mx.x,mn.y,mx.z},{mx.x,mx.y,mx.z},{mn.x,mx.y,mx.z},
        };
        int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},  // near face
            {4,5},{5,6},{6,7},{7,4},  // far face
            {0,4},{1,5},{2,6},{3,7},  // connecting
        };
        for (auto& e : edges)
            result.push_back({corners[e[0]], corners[e[1]]});
        return result;
    }

    // Build edge → adjacent face normals map
    std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> edge_map;
    edge_map.reserve(cnt * 3);

    for (auto& t : all_tris) {
        if (t.obj_id != obj_id) continue;
        float3 e1 = {t.v1.x-t.v0.x, t.v1.y-t.v0.y, t.v1.z-t.v0.z};
        float3 e2 = {t.v2.x-t.v0.x, t.v2.y-t.v0.y, t.v2.z-t.v0.z};
        float3 fn = cross(e1, e2);  // face normal (unnormalized)

        auto add = [&](float3 a, float3 b) {
            auto key = make_edge_key(a, b);
            auto& ei = edge_map[key];
            ei.a = a; ei.b = b;
            if (ei.count < 2) ei.fn[ei.count] = fn;
            ++ei.count;
        };
        add(t.v0, t.v1);
        add(t.v1, t.v2);
        add(t.v2, t.v0);
    }

    // Collect silhouette edges
    for (auto& [key, ei] : edge_map) {
        float3 mid = {(ei.a.x+ei.b.x)*0.5f,
                      (ei.a.y+ei.b.y)*0.5f,
                      (ei.a.z+ei.b.z)*0.5f};
        float3 view = {cam_pos.x-mid.x, cam_pos.y-mid.y, cam_pos.z-mid.z};

        if (ei.count == 1) {
            // Boundary / open edge — always silhouette
            result.push_back({ei.a, ei.b});
        } else {
            bool f0 = (dot(ei.fn[0], view) > 0.f);
            bool f1 = (dot(ei.fn[1], view) > 0.f);
            if (f0 != f1)
                result.push_back({ei.a, ei.b});
        }
    }
    return result;
}

// Returns world-space silhouette edges for the UNION of multiple obj_ids.
// Edges shared between two selected objects are interior and are excluded.
static std::vector<std::pair<float3,float3>>
compute_silhouette_merged(const std::vector<Triangle>& all_tris,
                          const std::vector<int>& obj_ids,
                          float3 cam_pos,
                          float3& aabb_mn_out, float3& aabb_mx_out)
{
    constexpr int LIMIT = 200000;
    std::unordered_set<int> id_set(obj_ids.begin(), obj_ids.end());

    float3 mn = make_float3( 1e30f, 1e30f, 1e30f);
    float3 mx = make_float3(-1e30f,-1e30f,-1e30f);
    int cnt = 0;
    for (auto& t : all_tris) {
        if (!id_set.count(t.obj_id)) continue;
        ++cnt;
        for (auto v : {t.v0, t.v1, t.v2}) {
            if (v.x<mn.x) mn.x=v.x; if (v.x>mx.x) mx.x=v.x;
            if (v.y<mn.y) mn.y=v.y; if (v.y>mx.y) mx.y=v.y;
            if (v.z<mn.z) mn.z=v.z; if (v.z>mx.z) mx.z=v.z;
        }
    }
    aabb_mn_out = mn; aabb_mx_out = mx;

    std::vector<std::pair<float3,float3>> result;
    if (cnt == 0) return result;

    if (cnt > LIMIT) {
        float3 corners[8] = {
            {mn.x,mn.y,mn.z},{mx.x,mn.y,mn.z},{mx.x,mx.y,mn.z},{mn.x,mx.y,mn.z},
            {mn.x,mn.y,mx.z},{mx.x,mn.y,mx.z},{mx.x,mx.y,mx.z},{mn.x,mx.y,mx.z},
        };
        int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
        };
        for (auto& e : edges)
            result.push_back({corners[e[0]], corners[e[1]]});
        return result;
    }

    std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> edge_map;
    edge_map.reserve(cnt * 3);
    for (auto& t : all_tris) {
        if (!id_set.count(t.obj_id)) continue;
        float3 e1 = {t.v1.x-t.v0.x, t.v1.y-t.v0.y, t.v1.z-t.v0.z};
        float3 e2 = {t.v2.x-t.v0.x, t.v2.y-t.v0.y, t.v2.z-t.v0.z};
        float3 fn = cross(e1, e2);
        auto add = [&](float3 a, float3 b) {
            auto key = make_edge_key(a, b);
            auto& ei = edge_map[key];
            ei.a = a; ei.b = b;
            if (ei.count < 2) ei.fn[ei.count] = fn;
            ++ei.count;
        };
        add(t.v0, t.v1);
        add(t.v1, t.v2);
        add(t.v2, t.v0);
    }
    for (auto& [key, ei] : edge_map) {
        float3 mid = {(ei.a.x+ei.b.x)*0.5f, (ei.a.y+ei.b.y)*0.5f, (ei.a.z+ei.b.z)*0.5f};
        float3 view = {cam_pos.x-mid.x, cam_pos.y-mid.y, cam_pos.z-mid.z};
        if (ei.count == 1) {
            result.push_back({ei.a, ei.b});
        } else {
            bool f0 = (dot(ei.fn[0], view) > 0.f);
            bool f1 = (dot(ei.fn[1], view) > 0.f);
            if (f0 != f1)
                result.push_back({ei.a, ei.b});
        }
    }
    return result;
}

// Rebuild BVH from current prims_sorted, re-upload, track selected sphere by center.
static void rebuild_bvh(std::vector<BVHNode>& bvh_nodes,
                         std::vector<Sphere>& prims_sorted,
                         BVHNode** d_bvh, Sphere** d_prims,
                         int& selected, float3 track_center)
{
    cudaFree(*d_bvh); cudaFree(*d_prims);
    *d_bvh = nullptr; *d_prims = nullptr;
    bvh_nodes.clear();                    // must clear or new root won't be at index 0
    std::vector<Sphere> src = prims_sorted;
    bvh_build(src, bvh_nodes, prims_sorted);
    bvh_upload(bvh_nodes, prims_sorted, d_bvh, d_prims);
    if (selected >= 0) {
        selected = -1;
        for (int i = 0; i < (int)prims_sorted.size(); ++i) {
            if (prims_sorted[i].center.x == track_center.x &&
                prims_sorted[i].center.y == track_center.y &&
                prims_sorted[i].center.z == track_center.z) {
                selected = i; break;
            }
        }
    }
}

// ─────────────────────────────────────────────
//  Mesh state (glTF)
// ─────────────────────────────────────────────

struct MeshState {
    std::vector<BVHNode>    bvh_nodes;
    std::vector<Triangle>   all_prims;  // canonical full list (never BVH-reordered)
    std::vector<Triangle>   prims;      // BVH-sorted visible subset (on GPU)
    std::vector<int>        prim_remap; // prim_remap[i] = all_prims index for prims[i]
    std::vector<MeshObject> objects;    // one per glTF primitive
    std::vector<GpuTexture> gpu_textures;

    BVHNode*             d_bvh      = nullptr;
    Triangle*            d_prims    = nullptr;
    GpuMaterial*         d_mats     = nullptr;
    cudaTextureObject_t* d_tex_hdls = nullptr;
    float3*              d_obj_colors = nullptr;
    int                  num_mats   = 0;
    int                  num_texs   = 0;
    int                  num_obj_colors = 0;
    size_t               tex_bytes  = 0;   // total device bytes used by all textures

    // Version counter — bumped whenever prims change so OptiX RT re-uploads BVH
    unsigned long long scene_version = 0;

    // ReSTIR
    std::vector<GpuMaterial> host_mats;   // CPU copy of materials (for emissive extraction)
    EmissiveTri*         d_emissive    = nullptr;
    int                  num_emissive  = 0;
    Reservoir*           d_reservoirs  = nullptr;
    Reservoir*           d_reservoirs2 = nullptr; // ping-pong for spatial reuse

    void free_gpu() {
        cudaFree(d_bvh);    d_bvh    = nullptr;
        cudaFree(d_prims);  d_prims  = nullptr;
        cudaFree(d_mats);   d_mats   = nullptr;
        cudaFree(d_tex_hdls); d_tex_hdls = nullptr;
        cudaFree(d_obj_colors); d_obj_colors = nullptr;
        cudaFree(d_emissive);   d_emissive   = nullptr;
        cudaFree(d_reservoirs); d_reservoirs = nullptr;
        cudaFree(d_reservoirs2);d_reservoirs2= nullptr;
        for (auto& t : gpu_textures) gpu_texture_free(t);
        gpu_textures.clear();
        bvh_nodes.clear();
        all_prims.clear();
        prims.clear();
        prim_remap.clear();
        objects.clear();
        host_mats.clear();
        num_mats = 0; num_texs = 0; num_obj_colors = 0; num_emissive = 0; tex_bytes = 0;
    }
};

static bool is_usd_scene_path(const std::string& path)
{
    auto ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);
    return (ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz");
}

static bool is_urdf_path(const std::string& path)
{
    auto ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);
    return (ext == ".urdf");
}

static bool is_mjcf_path(const std::string& path)
{
    auto ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);
    return (ext == ".xml");
}

static bool load_gltf_into(const std::string& path, MeshState& ms,
                            ControlPanelState& ctrl)
{
    ms.free_gpu();

    std::vector<Triangle>     tris;
    std::vector<GpuMaterial>  mats;
    std::vector<TextureImage> tex_images;

    // Route by extension
    const bool is_usd  = is_usd_scene_path(path);
    const bool is_urdf = is_urdf_path(path);
    const bool is_mjcf = is_mjcf_path(path);

    bool load_ok = is_urdf
        ? urdf_load(path, tris, mats, tex_images, ms.objects)
        : is_mjcf
            ? mjcf_load(path, tris, mats, tex_images, ms.objects)
            : is_usd
                ? usd_load (path, tris, mats, tex_images, ms.objects)
                : gltf_load(path, tris, mats, tex_images, ms.objects);
    if (!load_ok) return false;

    // Upload textures
    ms.tex_bytes = 0;
    for (auto& ti : tex_images) {
        if (ti.width > 0 && ti.height > 0 && !ti.pixels.empty()) {
            ms.gpu_textures.push_back(gpu_texture_upload_rgba8(ti.pixels.data(), ti.width, ti.height, ti.srgb));
            ms.tex_bytes += (size_t)ti.width * ti.height * 4; // RGBA8 bytes on device
        }
    }
    ms.d_tex_hdls = gpu_textures_upload_handles(ms.gpu_textures);
    ms.num_texs   = (int)ms.gpu_textures.size();

    // Upload materials
    if (!mats.empty()) {
        cudaMalloc(&ms.d_mats, mats.size() * sizeof(GpuMaterial));
        cudaMemcpy(ms.d_mats, mats.data(), mats.size() * sizeof(GpuMaterial), cudaMemcpyHostToDevice);
        ms.num_mats   = (int)mats.size();
        ms.host_mats  = mats;  // keep CPU copy for ReSTIR emissive extraction
    }

    // Generate random per-object colors (deterministic hash per obj_id)
    {
        int max_id = -1;
        for (auto& obj : ms.objects) if (obj.obj_id > max_id) max_id = obj.obj_id;
        if (max_id >= 0) {
            std::vector<float3> palette(max_id + 1);
            for (int i = 0; i <= max_id; ++i) {
                // PCG-style hash for vivid, well-distributed hues
                unsigned h = (unsigned)i * 2654435761u;
                float r = (float)((h >> 0) & 0xFF) / 255.f * 0.7f + 0.3f;
                float g = (float)((h >> 8) & 0xFF) / 255.f * 0.7f + 0.3f;
                float b = (float)((h >> 16) & 0xFF) / 255.f * 0.7f + 0.3f;
                palette[i] = make_float3(r, g, b);
            }
            cudaMalloc(&ms.d_obj_colors, palette.size() * sizeof(float3));
            cudaMemcpy(ms.d_obj_colors, palette.data(),
                       palette.size() * sizeof(float3), cudaMemcpyHostToDevice);
            ms.num_obj_colors = (int)palette.size();
        }
    }

    // Keep canonical copy before BVH reorders triangles
    ms.all_prims = tris;

    // Build + upload triangle BVH
    ms.bvh_nodes.clear();
    bvh_build_triangles(tris, ms.bvh_nodes, ms.prims);
    bvh_upload_triangles(ms.bvh_nodes, ms.prims, &ms.d_bvh, &ms.d_prims);

    // Extract emissive triangles for ReSTIR DI
    if (!ms.host_mats.empty()) {
        auto em_tris = restir_extract_emissive(tris, ms.host_mats);
        if (!em_tris.empty()) {
            cudaMalloc(&ms.d_emissive, em_tris.size() * sizeof(EmissiveTri));
            cudaMemcpy(ms.d_emissive, em_tris.data(),
                       em_tris.size() * sizeof(EmissiveTri), cudaMemcpyHostToDevice);
            ms.num_emissive = (int)em_tris.size();
        }
    }

    ctrl.mesh_loaded   = true;
    ctrl.num_mesh_tris = (int)ms.prims.size();
    ++ms.scene_version;  // signal OptiX RT to re-build hardware BVH

    // Compute AABB directly from the raw triangle list (before BVH reorders them)
    if (!tris.empty()) {
        float3 mn = make_float3( 1e30f,  1e30f,  1e30f);
        float3 mx = make_float3(-1e30f, -1e30f, -1e30f);
        for (const Triangle& tri : tris) {
            float3 verts[3] = { tri.v0, tri.v1, tri.v2 };
            for (auto& v : verts) {
                if (v.x < mn.x) mn.x = v.x;  if (v.x > mx.x) mx.x = v.x;
                if (v.y < mn.y) mn.y = v.y;  if (v.y > mx.y) mx.y = v.y;
                if (v.z < mn.z) mn.z = v.z;  if (v.z > mx.z) mx.z = v.z;
            }
        }

        float cx = (mn.x + mx.x) * 0.5f;
        float cy = (mn.y + mx.y) * 0.5f;
        float cz = (mn.z + mx.z) * 0.5f;

        float diag = sqrtf(
            (mx.x-mn.x)*(mx.x-mn.x) +
            (mx.y-mn.y)*(mx.y-mn.y) +
            (mx.z-mn.z)*(mx.z-mn.z));
        float dist = diag * 0.8f;
        if (dist < 0.5f) dist = 0.5f;

        // Orbit pivot = centroid; camera positioned above-back, looking at centroid
        ctrl.orbit_pivot[0] = cx;
        ctrl.orbit_pivot[1] = cy;
        ctrl.orbit_pivot[2] = cz;

        ctrl.pos[0] = cx;
        ctrl.pos[1] = cy + dist * 0.3f;
        ctrl.pos[2] = cz + dist;

        // Also update cam_from/cam_at immediately so any same-frame sync
        // (e.g. Hydra preview) gets the correct auto-fit camera right away.
        ctrl.cam_from[0] = ctrl.pos[0];
        ctrl.cam_from[1] = ctrl.pos[1];
        ctrl.cam_from[2] = ctrl.pos[2];
        ctrl.cam_at[0]   = ctrl.orbit_pivot[0];
        ctrl.cam_at[1]   = ctrl.orbit_pivot[1];
        ctrl.cam_at[2]   = ctrl.orbit_pivot[2];

        // Auto-scale move speed to ~5% of scene diagonal so navigation feels
        // natural regardless of whether the scene is centimetres or kilometres.
        ctrl.move_speed = fmaxf(1.5f, diag * 0.15f);

        // Force Orbit mode so cam_at = orbit_pivot is immediately applied
        ctrl.interact_mode  = InteractMode::Orbit;
        ctrl.selected_sphere   = -1;
        ctrl.selected_mesh_obj = -1;
    }
    return true;
}

static void rebuild_visible_bvh(MeshState& ms);  // forward decl

// Import a 3D file into the existing scene (append, don't replace).
// Offsets obj_ids and mat_indices so they don't clash with existing data.
static bool import_into_scene(const std::string& path, MeshState& ms,
                               ControlPanelState& ctrl)
{
    std::vector<Triangle>     tris;
    std::vector<GpuMaterial>  mats;
    std::vector<TextureImage> tex_images;
    std::vector<MeshObject>   new_objects;

    const bool is_usd  = is_usd_scene_path(path);
    const bool is_urdf = is_urdf_path(path);
    const bool is_mjcf = is_mjcf_path(path);

    bool load_ok = is_urdf
        ? urdf_load(path, tris, mats, tex_images, new_objects)
        : is_mjcf
            ? mjcf_load(path, tris, mats, tex_images, new_objects)
            : is_usd
                ? usd_load (path, tris, mats, tex_images, new_objects)
                : gltf_load(path, tris, mats, tex_images, new_objects);
    if (!load_ok) return false;

    // Compute offsets
    int obj_id_offset = 0;
    for (auto& obj : ms.objects)
        if (obj.obj_id >= obj_id_offset) obj_id_offset = obj.obj_id + 1;
    int mat_offset = ms.num_mats;
    int tex_offset = ms.num_texs;

    // Offset triangle references
    for (auto& tri : tris) {
        tri.obj_id  += obj_id_offset;
        tri.mat_idx += mat_offset;
        // Offset texture indices in UVs (if they reference texture array)
    }
    for (auto& obj : new_objects)
        obj.obj_id += obj_id_offset;
    for (auto& m : mats) {
        if (m.base_color_tex    >= 0) m.base_color_tex    += tex_offset;
        if (m.metallic_rough_tex >= 0) m.metallic_rough_tex += tex_offset;
        if (m.normal_tex        >= 0) m.normal_tex        += tex_offset;
        if (m.emissive_tex      >= 0) m.emissive_tex      += tex_offset;
    }

    // Append textures
    for (auto& ti : tex_images) {
        if (ti.width > 0 && ti.height > 0 && !ti.pixels.empty()) {
            ms.gpu_textures.push_back(gpu_texture_upload_rgba8(ti.pixels.data(), ti.width, ti.height, ti.srgb));
            ms.tex_bytes += (size_t)ti.width * ti.height * 4;
        }
    }
    // Re-upload texture handle array
    cudaFree(ms.d_tex_hdls); ms.d_tex_hdls = nullptr;
    ms.d_tex_hdls = gpu_textures_upload_handles(ms.gpu_textures);
    ms.num_texs   = (int)ms.gpu_textures.size();

    // Append materials
    ms.host_mats.insert(ms.host_mats.end(), mats.begin(), mats.end());
    cudaFree(ms.d_mats); ms.d_mats = nullptr;
    if (!ms.host_mats.empty()) {
        cudaMalloc(&ms.d_mats, ms.host_mats.size() * sizeof(GpuMaterial));
        cudaMemcpy(ms.d_mats, ms.host_mats.data(),
                   ms.host_mats.size() * sizeof(GpuMaterial), cudaMemcpyHostToDevice);
        ms.num_mats = (int)ms.host_mats.size();
    }

    // Append objects
    ms.objects.insert(ms.objects.end(), new_objects.begin(), new_objects.end());

    // Append triangles
    ms.all_prims.insert(ms.all_prims.end(), tris.begin(), tris.end());

    // Regenerate per-object colors for ALL objects
    {
        int max_id = -1;
        for (auto& obj : ms.objects) if (obj.obj_id > max_id) max_id = obj.obj_id;
        if (max_id >= 0) {
            std::vector<float3> palette(max_id + 1);
            for (int i = 0; i <= max_id; ++i) {
                unsigned h = (unsigned)i * 2654435761u;
                float r = (float)((h >> 0) & 0xFF) / 255.f * 0.7f + 0.3f;
                float g = (float)((h >> 8) & 0xFF) / 255.f * 0.7f + 0.3f;
                float b = (float)((h >> 16) & 0xFF) / 255.f * 0.7f + 0.3f;
                palette[i] = make_float3(r, g, b);
            }
            cudaFree(ms.d_obj_colors); ms.d_obj_colors = nullptr;
            cudaMalloc(&ms.d_obj_colors, palette.size() * sizeof(float3));
            cudaMemcpy(ms.d_obj_colors, palette.data(),
                       palette.size() * sizeof(float3), cudaMemcpyHostToDevice);
            ms.num_obj_colors = (int)palette.size();
        }
    }

    // Rebuild BVH with all triangles
    rebuild_visible_bvh(ms);

    ctrl.mesh_loaded   = true;
    ctrl.num_mesh_tris = (int)ms.all_prims.size();
    ++ms.scene_version;

    std::cerr << "[import] added " << tris.size() << " triangles, "
              << new_objects.size() << " objects from " << path << '\n';
    return true;
}

// Rebuild GPU BVH from only the visible objects in all_prims.
// ms.prims becomes the sorted visible subset; all_prims stays untouched.
static void rebuild_visible_bvh(MeshState& ms)
{
    std::unordered_map<int,bool> hidden_map;
    for (auto& obj : ms.objects)
        hidden_map[obj.obj_id] = obj.hidden;

    // Build visible list and record which all_prims index each came from
    std::vector<Triangle> visible;
    std::vector<int> vis_to_all;  // vis_to_all[k] = all_prims index
    visible.reserve(ms.all_prims.size());
    vis_to_all.reserve(ms.all_prims.size());
    for (int j = 0; j < (int)ms.all_prims.size(); j++) {
        if (!hidden_map[ms.all_prims[j].obj_id]) {
            visible.push_back(ms.all_prims[j]);
            vis_to_all.push_back(j);
        }
    }

    cudaFree(ms.d_bvh);   ms.d_bvh   = nullptr;
    cudaFree(ms.d_prims); ms.d_prims = nullptr;
    ms.bvh_nodes.clear();
    ms.prims.clear();
    ms.prim_remap.clear();

    if (!visible.empty()) {
        // BVH build reorders visible → prims (sorted by spatial locality)
        bvh_build_triangles(visible, ms.bvh_nodes, ms.prims);
        bvh_upload_triangles(ms.bvh_nodes, ms.prims, &ms.d_bvh, &ms.d_prims);

        // Build prim_remap: prims[i] → all_prims index.
        // BVH sort permuted the visible array. Match by finding which visible
        // entry each prims[i] corresponds to (bit-exact vertex match pre-repose).
        ms.prim_remap.resize(ms.prims.size(), -1);
        // Build lookup from v0 bit pattern to vis_to_all index for O(N) matching
        struct V0Key {
            float x, y, z;
            int obj_id;
            bool operator==(const V0Key& o) const {
                return x == o.x && y == o.y && z == o.z && obj_id == o.obj_id;
            }
        };
        struct V0Hash {
            size_t operator()(const V0Key& k) const {
                size_t h = 0;
                h ^= std::hash<float>{}(k.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<float>{}(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<float>{}(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<int>{}(k.obj_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };
        // Map from v0+obj_id → list of all_prims indices (handles duplicates)
        std::unordered_multimap<V0Key, int, V0Hash> key_to_all;
        for (int k = 0; k < (int)visible.size(); k++) {
            V0Key key = { visible[k].v0.x, visible[k].v0.y, visible[k].v0.z, visible[k].obj_id };
            key_to_all.insert({ key, vis_to_all[k] });
        }
        for (int i = 0; i < (int)ms.prims.size(); i++) {
            V0Key key = { ms.prims[i].v0.x, ms.prims[i].v0.y, ms.prims[i].v0.z, ms.prims[i].obj_id };
            auto range = key_to_all.equal_range(key);
            for (auto it = range.first; it != range.second; ++it) {
                ms.prim_remap[i] = it->second;
                key_to_all.erase(it);  // consume so duplicates get unique mappings
                break;
            }
        }
    }
}

// Lightweight GPU re-upload: skip BVH rebuild, just memcpy updated triangles.
// Uses prim_remap to copy transformed data from all_prims into BVH-sorted order,
// keeping the BVH topology valid (only AABBs are stale — acceptable during drag).
static void reupload_visible_prims(MeshState& ms)
{
    if (ms.all_prims.empty() || !ms.d_prims || ms.prims.empty()) return;

    // Update prims from all_prims using BVH remap (preserves BVH sort order)
    if (!ms.prim_remap.empty() && ms.prim_remap.size() == ms.prims.size()) {
        for (int i = 0; i < (int)ms.prims.size(); i++) {
            int src = ms.prim_remap[i];
            if (src >= 0 && src < (int)ms.all_prims.size()) {
                // Copy only geometry (v0-v2, n0-n2) — UVs/tangents/ids stay the same
                ms.prims[i].v0 = ms.all_prims[src].v0;
                ms.prims[i].v1 = ms.all_prims[src].v1;
                ms.prims[i].v2 = ms.all_prims[src].v2;
                ms.prims[i].n0 = ms.all_prims[src].n0;
                ms.prims[i].n1 = ms.all_prims[src].n1;
                ms.prims[i].n2 = ms.all_prims[src].n2;
            }
        }
    } else {
        // No remap (e.g. first time, or rasterized-only) — direct copy
        bool any_hidden = false;
        for (auto& obj : ms.objects)
            if (obj.hidden) { any_hidden = true; break; }

        if (!any_hidden && ms.prims.size() == ms.all_prims.size()) {
            ms.prims = ms.all_prims;
        } else {
            std::unordered_map<int,bool> hidden_map;
            for (auto& obj : ms.objects)
                hidden_map[obj.obj_id] = obj.hidden;
            ms.prims.clear();
            ms.prims.reserve(ms.all_prims.size());
            for (auto& t : ms.all_prims)
                if (!hidden_map[t.obj_id]) ms.prims.push_back(t);
        }
    }

    cudaMemcpyAsync(ms.d_prims, ms.prims.data(),
                    ms.prims.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
}

// ─────────────────────────────────────────────
//  Window-state persistence
// ─────────────────────────────────────────────

static std::filesystem::path exe_dir_main()
{
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
}

struct WinState { int x, y, w, h; bool maximized; bool show_gpu_arch = false; };

static WinState load_win_state()
{
    WinState s{ 100, 100, 1600, 900, false };
    auto path = exe_dir_main() / "window.ini";
    std::ifstream f(path);
    if (!f) return s;
    f >> s.x >> s.y >> s.w >> s.h >> s.maximized;
    f >> s.show_gpu_arch;  // optional field — silently ignored if absent (old window.ini)

    // Clamp degenerate sizes
    if (s.w < 400) s.w = 1600;
    if (s.h < 300) s.h = 900;

    // Make sure the title bar is reachable on some monitor (32px safety margin)
    POINT pt{ s.x + s.w / 2, s.y + 16 };
    HMONITOR hmon = MonitorFromPoint(pt, MONITOR_DEFAULTTONULL);
    if (!hmon) {
        // Saved position is offscreen — reset to primary monitor default
        s.x = 100; s.y = 100;
        hmon = MonitorFromPoint(POINT{s.x, s.y}, MONITOR_DEFAULTTOPRIMARY);
    }

    // Clamp to monitor work area — prevents the window from opening
    // larger than the screen (catches stale window.ini from old bugs).
    if (hmon) {
        MONITORINFO mi{}; mi.cbSize = sizeof(mi);
        if (GetMonitorInfoA(hmon, &mi)) {
            int mw = mi.rcWork.right  - mi.rcWork.left;
            int mh = mi.rcWork.bottom - mi.rcWork.top;
            if (s.w > mw) s.w = mw;
            if (s.h > mh) s.h = mh;
            // Also clamp position so window stays on-screen
            if (s.x < mi.rcWork.left)   s.x = mi.rcWork.left;
            if (s.y < mi.rcWork.top)    s.y = mi.rcWork.top;
            if (s.x + s.w > mi.rcWork.right)  s.x = mi.rcWork.right  - s.w;
            if (s.y + s.h > mi.rcWork.bottom) s.y = mi.rcWork.bottom - s.h;
        }
    }
    return s;
}

static void save_win_state(GLFWwindow* win, bool s_show_gpu_arch)
{
    HWND hwnd = glfwGetWin32Window(win);

    // Check maximized state via Win32 (works even if GLFW already started teardown).
    WINDOWPLACEMENT wp{};
    wp.length = sizeof(wp);
    bool is_max = GetWindowPlacement(hwnd, &wp) && (wp.showCmd == SW_SHOWMAXIMIZED);

    WinState s{};
    s.maximized = is_max;

    if (is_max) {
        // When maximized, glfwGetWindowPos/Size return the maximized rect.
        // Save the restored rect from GetWindowPlacement instead so we
        // restore to the correct non-maximized size next launch.
        s.x = wp.rcNormalPosition.left;
        s.y = wp.rcNormalPosition.top;
        s.w = wp.rcNormalPosition.right  - wp.rcNormalPosition.left;
        s.h = wp.rcNormalPosition.bottom - wp.rcNormalPosition.top;
    } else {
        // Use GLFW getters — these return the visible window rect without
        // the invisible DWP extended-frame borders that GetWindowPlacement
        // includes on Windows 10/11, preventing the window from growing
        // by ~14px on every save→load cycle.
        glfwGetWindowPos(win, &s.x, &s.y);
        glfwGetWindowSize(win, &s.w, &s.h);
    }

    // Sanity: don't save a degenerate size
    if (s.w < 400 || s.h < 300) return;

    auto path = exe_dir_main() / "window.ini";
    std::ofstream f(path);
    if (f) f << s.x << ' ' << s.y << ' ' << s.w << ' ' << s.h << ' '
               << s.maximized << ' ' << s_show_gpu_arch << '\n';
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

int main() {
    try {
    SetUnhandledExceptionFilter(win_unhandled_exception_filter);
    const bool streamline_disabled = env_flag_enabled("CUDA_VK_DISABLE_STREAMLINE");
    const bool cupti_enabled = !env_flag_enabled("CUDA_VK_DISABLE_CUPTI");

    // Streamline DLSS: slInit() must be called before any vkCreateInstance.
    // plugin_dir = exe directory where sl.dlss.dll, sl.common.dll etc. are deployed.
    if (!streamline_disabled) {
        char exe_buf[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exe_buf, MAX_PATH);
        std::filesystem::path exe_dir = std::filesystem::path(exe_buf).parent_path();
        dlss_pre_vulkan_init(exe_dir.string().c_str());
    } else {
        fprintf(stderr, "[dlss] Streamline disabled by CUDA_VK_DISABLE_STREAMLINE\n");
        fflush(stderr);
    }
    if (!glfwInit()) throw std::runtime_error("GLFW init failed");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    WinState wstate = load_win_state();
    GLFWwindow* window = glfwCreateWindow(wstate.w, wstate.h, "CUDA Path Tracer", nullptr, nullptr);
    if (!window) throw std::runtime_error("GLFW window creation failed");
    glfwSetWindowPos(window, wstate.x, wstate.y);
    if (wstate.maximized) glfwMaximizeWindow(window);

    VulkanContext vk = vk_create(window, true);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // Persist panel layout next to the exe so it survives working-dir changes
    static std::string imgui_ini_path = (exe_dir_main() / "imgui.ini").string();
    io.IniFilename = imgui_ini_path.c_str();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance        = vk.instance;
    init_info.PhysicalDevice  = vk.physicalDevice;
    init_info.Device          = vk.device;
    init_info.QueueFamily     = vk.graphicsFamily;
    init_info.Queue           = vk.graphicsQueue;
    init_info.ApiVersion      = VK_API_VERSION_1_3;
    init_info.DescriptorPoolSize           = 64;
    init_info.PipelineInfoMain.RenderPass  = vk.renderPass;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.MinImageCount   = 2;
    init_info.ImageCount      = (uint32_t)vk.swapImages.size();
    ImGui_ImplVulkan_Init(&init_info);

    int rt_w = 800, rt_h = 600;
    CudaInterop interop = cuda_interop_create(
        vk.device, vk.physicalDevice, vk.commandPool, vk.graphicsQueue, rt_w, rt_h);

    std::vector<Sphere>  prims_in = make_scene();
    std::vector<BVHNode> bvh_nodes;
    std::vector<Sphere>  prims_sorted;
    bvh_build(prims_in, bvh_nodes, prims_sorted);

    BVHNode* d_bvh   = nullptr;
    Sphere*  d_prims = nullptr;
    bvh_upload(bvh_nodes, prims_sorted, &d_bvh, &d_prims);

    float4*      d_accum = pathtracer_alloc_accum(rt_w, rt_h);

    // CUDA software rasterizer state (z-buffer, lazy-allocated on first use)
    RasterizerState raster_state{};
    rasterizer_init(raster_state);

    // DLSS auxiliary buffers — depth (float) and motion vectors (float2), render-res
    float*  d_depth  = nullptr;  cudaMalloc(&d_depth,  (size_t)rt_w * rt_h * sizeof(float));
    float2* d_motion = nullptr;  cudaMalloc(&d_motion, (size_t)rt_w * rt_h * sizeof(float2));
    Camera prev_cam = {};
    bool   has_prev_cam = false;

    // ReSTIR reservoir buffers (reallocated on resize)
    // Allocated in MeshState lazily at first use so they match actual viewport size

    ViewportPanel            vp;
    ControlPanelState        ctrl;
    ctrl.show_gpu_arch = wstate.show_gpu_arch;
    AnimPanelState           anim;
    HydraPreviewState        hp_preview;
    bool                     hydra_preview_ready = false;
    std::string              hydra_loaded_scene_path;
    MeshState                mesh;
    std::unordered_set<int>  multi_sel;   // obj indices currently selected
    int frame_count = 0;

    // USD animation — persistent stage handle for per-frame geometry re-evaluation
    UsdAnimHandle*           usd_anim_handle = nullptr;
    float                    usd_last_frame_time = -1e30f;

    // URDF articulation — persistent handle for interactive joint control
    UrdfArticulation*        urdf_artic_handle = nullptr;
    RobotDemoState           robot_demo;

    HdriMap        hdri;
    DenoiserState  denoiser;
    denoiser_init(denoiser, rt_w, rt_h);
    ctrl.denoise_available = denoiser.available;

    OptixDenoiserState  optix_dn;
    OptixRendererState  optix_rt;
    DlssState           dlss;
    bool dlss_active = false;   // true when DLSS initialized successfully

    NvmlContext     nvml;
    GpuHardwareInfo hw_info;
    GpuLiveStats    live_stats;
    stats_init(nvml, hw_info);

    GpuArchWindowState arch_state{};
    gpu_arch_window_init(arch_state,
                         hw_info.compute_major, hw_info.compute_minor,
                         hw_info.l2_cache_bytes / (1024 * 1024));
    arch_state.vram_total = hw_info.total_vram_bytes;

    SmTracker sm_tracker{};
    sm_tracker_init(sm_tracker, hw_info.sm_count);

    CuptiProfiler cupti_profiler;
    arch_state.cupti_available = cupti_enabled ? safe_cupti_init(cupti_profiler, 0) : false;
    arch_state.profiling_enabled = arch_state.cupti_available; // default ON when CUPTI is available
    if (arch_state.cupti_available)
        printf("[main] CUPTI hardware counters available\n");
    else
        printf("[main] CUPTI not available — using NVML estimates\n");

    float arch_timer_ms = 0.f;   // accumulates frame_ms; snapshot every 500ms

    // OptiX GPU denoiser (replaces OIDN when available and enabled)
#ifdef OPTIX_ENABLED
    if (optix_denoiser_init(optix_dn, rt_w, rt_h))
        printf("[main] OptiX GPU denoiser available\n");

    // OptiX RT init is deferred: background thread only starts when user enables the
    // checkbox (lazy init). This avoids competing with model/HDRI loads on startup.
    std::atomic<bool> optix_rt_init_done{false};
    std::atomic<bool> optix_rt_init_started{false};
    std::thread optix_rt_thread;
#endif

    // DLSS — only init when user enables it (done on demand in frame loop)
#ifdef DLSS_ENABLED
    ctrl.dlss_has_sdk = !streamline_disabled;
#endif

    PostProcess post;
    post = post_process_create(vk.device, vk.physicalDevice,
        vk.descriptorPool,
        interop.image_view, interop.sampler,
        vk.commandPool, vk.graphicsQueue,
        rt_w, rt_h);
    if (!post.pipeline) {
        std::cerr << "[main] Post-processing disabled (post.spv not loaded).\n";
        ctrl.post_enabled = false;
    }

    // Slang runtime material shader support — requires slangc in PATH or SLANGC_PATH
#ifdef SLANGC_PATH
    slang_shader_init(vk.device, vk.physicalDevice, vk.commandPool, vk.graphicsQueue,
                      SLANGC_PATH);
#else
    slang_shader_init(vk.device, vk.physicalDevice, vk.commandPool, vk.graphicsQueue,
                      "slangc");
#endif

    // NIM VLM recognition thread
    std::thread nim_thread;
    // Cosmos Transfer thread
    std::thread cosmos_thread;

    // Auto-ping NIM endpoint on startup (background, non-blocking)
    nim_thread = std::thread([cfg = ctrl.nim_cfg, &ctrl]() {
        bool ok = nim_vlm_check_connection(cfg);
        ctrl.nim_connection_ok     = ok;
        ctrl.nim_connection_tested = true;
    });

    BatchProcessor batch;

    // Denoiser display buffer — denoiser writes here, d_accum stays pristine
    float4* d_display = pathtracer_alloc_accum(rt_w, rt_h);

    // ── Auto-load Sponza if bundled alongside the exe ─────────────────
    {
        // Look for assets/Sponza/glTF/Sponza.gltf next to the executable
        // (or next to the working directory, whichever exists first)
        const char* candidates[] = {
            "assets/Sponza/glTF/Sponza.gltf",
            "../assets/Sponza/glTF/Sponza.gltf",
        };
        for (auto* p : candidates) {
            if (std::filesystem::exists(p)) {
                std::cout << "Auto-loading " << p << "\n";
                strncpy_s(ctrl.gltf_path, sizeof(ctrl.gltf_path), p, _TRUNCATE);
                load_gltf_into(p, mesh, ctrl);
                if (is_usd_scene_path(p))
                    usd_anim_handle = usd_anim_open(p, mesh.all_prims);
                break;
            }
        }
    }

    double prev_mx = 0.0, prev_my = 0.0;
    bool   rmb_was_down = false;
    bool   pending_rt_resize = false;
    int    pending_rt_w = 0;
    int    pending_rt_h = 0;

    auto t_last = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        auto t_now = std::chrono::high_resolution_clock::now();
        float frame_ms = std::chrono::duration<float, std::milli>(t_now - t_last).count();
        if (frame_ms < 0.001f) frame_ms = 0.001f;
        t_last = t_now;

        if (pending_rt_resize && pending_rt_w > 0 && pending_rt_h > 0) {
            vkDeviceWaitIdle(vk.device);
            cuda_interop_destroy(vk.device, interop);
            rt_w = pending_rt_w; rt_h = pending_rt_h;
            interop = cuda_interop_create(vk.device, vk.physicalDevice,
                                          vk.commandPool, vk.graphicsQueue, rt_w, rt_h);
            cudaFree(d_accum); cudaFree(d_display);
            cudaFree(d_depth);  cudaMalloc(&d_depth,  (size_t)rt_w * rt_h * sizeof(float));
            cudaFree(d_motion); cudaMalloc(&d_motion, (size_t)rt_w * rt_h * sizeof(float2));
            d_accum   = pathtracer_alloc_accum(rt_w, rt_h);
            d_display = pathtracer_alloc_accum(rt_w, rt_h);
            has_prev_cam = false;
            // ReSTIR reservoir buffers — reallocate on resize
            cudaFree(mesh.d_reservoirs);  mesh.d_reservoirs  = restir_alloc_reservoirs(rt_w, rt_h);
            cudaFree(mesh.d_reservoirs2); mesh.d_reservoirs2 = restir_alloc_reservoirs(rt_w, rt_h);
            // Denoiser re-init for new resolution
            denoiser_init(denoiser, rt_w, rt_h);
            ctrl.denoise_available = denoiser.available;
#ifdef OPTIX_ENABLED
            if (optix_dn.available)
                optix_denoiser_init(optix_dn, rt_w, rt_h);
            if (optix_rt.available)
                optix_renderer_resize(optix_rt, rt_w, rt_h);
#endif
            // DLSS needs reinit at new resolution
            if (dlss_active) { dlss_free(dlss); dlss_active = false; }
            post_process_resize(post, vk.physicalDevice,
                vk.descriptorPool,
                vk.commandPool, vk.graphicsQueue,
                interop.image_view, interop.sampler,
                rt_w, rt_h);
            frame_count = 0;
            pending_rt_resize = false;
        }

        // Animation tick + Hydra preview panel state
        anim_tick(anim, frame_ms * 0.001f);
        const std::string active_scene_path = ctrl.gltf_path;
        const bool active_scene_is_usd = !active_scene_path.empty() && is_usd_scene_path(active_scene_path);
        if (anim.preview_open && !active_scene_is_usd)
            anim.preview_open = false;

        uint32_t img_idx = vk_begin_frame(vk);
        if (img_idx == UINT32_MAX) { vk_recreate_swapchain(vk, window); continue; }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                     ImGuiDockNodeFlags_NoUndocking);

        bool cam_changed = control_panel_draw(ctrl);

        // USD animation: re-evaluate geometry when the time code changes.
        // Runs whenever the timeline moves (play or scrub), even if no
        // time samples were auto-detected (skeletal anim may not flag).
        if (usd_anim_handle && active_scene_is_usd &&
            anim.current_time != usd_last_frame_time)
        {
            usd_last_frame_time = anim.current_time;
            if (usd_load_frame(usd_anim_handle, anim.current_time, mesh.all_prims)) {
                rebuild_visible_bvh(mesh);
                ++mesh.scene_version;  // trigger OptiX RT re-upload
                cam_changed = true;    // reset accumulation
            }
        }
        // URDF articulation: re-pose geometry when joint sliders change.
        // During interactive drag: fast reupload + BVH refit (all viewport modes).
        // On slider release: full BVH rebuild for optimal tree quality.
        {
            static bool s_artic_was_active = false;
            bool artic_changed = articulation_panel_draw(urdf_artic_handle, robot_demo.playing) &&
                                 urdf_repose(urdf_artic_handle, mesh.all_prims);
            if (artic_changed) {
                reupload_visible_prims(mesh);
                // Refit BVH AABBs from updated vertex positions (O(N), no rebuild)
                if (mesh.d_bvh && !mesh.bvh_nodes.empty())
                    bvh_refit_triangles(mesh.bvh_nodes, mesh.prims, mesh.d_bvh);
                ++mesh.scene_version;
                cam_changed = true;
                s_artic_was_active = true;
            } else if (s_artic_was_active) {
                // Sliders stopped — full BVH rebuild for optimal tree quality
                rebuild_visible_bvh(mesh);
                cam_changed = true;
                ++mesh.scene_version;
                s_artic_was_active = false;
            }
        }

        // Robot demo: tick playback + draw panel
        {
            float dt = frame_ms * 0.001f;
            bool demo_changed = robot_demo_tick(robot_demo, urdf_artic_handle, dt);
            robot_demo_panel_draw(robot_demo, urdf_artic_handle);
            // Scrub also triggers a pose update
            demo_changed = demo_changed || robot_demo.scrub_changed;
            robot_demo.scrub_changed = false;

            // Reset grasped object to original position on loop/restart
            if (robot_demo.needs_grasp_reset) {
                robot_demo_reset_grasp(robot_demo, prims_sorted, mesh.objects, mesh.all_prims);
                robot_demo.needs_grasp_reset = false;
                // Re-upload moved geometry
                reupload_visible_prims(mesh);
                if (mesh.d_bvh && !mesh.bvh_nodes.empty())
                    bvh_refit_triangles(mesh.bvh_nodes, mesh.prims, mesh.d_bvh);
                rebuild_bvh(bvh_nodes, prims_sorted, &d_bvh, &d_prims,
                            ctrl.selected_sphere,
                            ctrl.selected_sphere >= 0
                                ? prims_sorted[ctrl.selected_sphere].center
                                : make_float3(0,0,0));
                ++mesh.scene_version;
                cam_changed = true;
            }

            if (demo_changed && urdf_artic_handle) {
                urdf_repose(urdf_artic_handle, mesh.all_prims);
                reupload_visible_prims(mesh);
                if (mesh.d_bvh && !mesh.bvh_nodes.empty())
                    bvh_refit_triangles(mesh.bvh_nodes, mesh.prims, mesh.d_bvh);
                ++mesh.scene_version;
                cam_changed = true;

                // Move grasped object to follow end-effector
                robot_demo_update_grasp(robot_demo, urdf_artic_handle,
                                        prims_sorted, mesh.objects, mesh.all_prims);
                if (robot_demo.grasp.active) {
                    if (!robot_demo.grasp.is_mesh) {
                        rebuild_bvh(bvh_nodes, prims_sorted, &d_bvh, &d_prims,
                                    ctrl.selected_sphere,
                                    ctrl.selected_sphere >= 0
                                        ? prims_sorted[ctrl.selected_sphere].center
                                        : make_float3(0,0,0));
                    } else {
                        // Mesh moved — re-upload triangles
                        reupload_visible_prims(mesh);
                        if (mesh.d_bvh && !mesh.bvh_nodes.empty())
                            bvh_refit_triangles(mesh.bvh_nodes, mesh.prims, mesh.d_bvh);
                        ++mesh.scene_version;
                    }
                }
            }
        }

        anim_panel_draw(anim, ctrl.gltf_path, active_scene_is_usd);
        viewport_draw(vp, (ctrl.post_enabled && post.imgui_desc) ? post.imgui_desc : interop.descriptor, ctrl);
        if (anim.preview_open && active_scene_is_usd) {
            if (!hydra_preview_ready) {
                hydra_preview_ready = safe_hydra_preview_init(
                    hp_preview,
                    vk.device,
                    vk.physicalDevice,
                    vk.graphicsQueue,
                    vk.graphicsFamily,
                    vk.commandPool);
                if (!hydra_preview_ready)
                    anim.preview_open = false;
            }
            if (hydra_preview_ready && hydra_loaded_scene_path != active_scene_path) {
                hydra_preview_ready = safe_hydra_preview_load(hp_preview, active_scene_path);
                if (hydra_preview_ready) hydra_loaded_scene_path = active_scene_path;
                else anim.preview_open = false;
            }

            ImGui::Begin("Hydra Preview");
            {
                static const char* hydra_aov_items[] = {
                    "Color", "Depth", "Neye", "PrimId"
                };
                // Map combo index → aov_mode: 0=color, 1=depth, 2=Neye, 3=primId
                static const int hydra_aov_map[] = { 0, 1, 2, 3 };
                int combo_idx = 0;
                for (int i = 0; i < IM_ARRAYSIZE(hydra_aov_map); ++i)
                    if (hydra_aov_map[i] == hp_preview.aov_mode) { combo_idx = i; break; }
                ImGui::SetNextItemWidth(90.f);
                if (ImGui::Combo("AOV##hydra", &combo_idx,
                                 hydra_aov_items, IM_ARRAYSIZE(hydra_aov_items)))
                    hp_preview.aov_mode = hydra_aov_map[combo_idx];
            }
            ImVec2 avail = ImGui::GetContentRegionAvail();
            // Always render at the current window size so it tracks resize.
            int hw = std::max(1, (int)avail.x);
            int hh = std::max(1, (int)avail.y);

            // Always sync Hydra camera FROM the main viewport state.
            // The Hydra local camera controls below may override this,
            // and if they do they write back to ctrl.pos/orbit_pivot.
            hp_preview.pos[0] = ctrl.pos[0];
            hp_preview.pos[1] = ctrl.pos[1];
            hp_preview.pos[2] = ctrl.pos[2];
            hp_preview.pivot[0] = ctrl.orbit_pivot[0];
            hp_preview.pivot[1] = ctrl.orbit_pivot[1];
            hp_preview.pivot[2] = ctrl.orbit_pivot[2];
            hp_preview.vfov = ctrl.vfov;

            // Camera controls — identical to main viewport: LMB=orbit, MMB=pan, scroll=zoom.
            // These override the synced values above and write back to ctrl.
            ImVec2 cursor_before_cam = ImGui::GetCursorPos();
            if (avail.x > 4.0f && avail.y > 4.0f) {
                ImGui::InvisibleButton("##hydra_cam", avail,
                    ImGuiButtonFlags_MouseButtonLeft |
                    ImGuiButtonFlags_MouseButtonMiddle |
                    ImGuiButtonFlags_MouseButtonRight);
                bool hydra_cam_changed = false;
                if (ImGui::IsItemActive()) {
                    ImVec2 delta = ImGui::GetIO().MouseDelta;

                    // LMB — arcball orbit (identical to main viewport)
                    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                        float ox = hp_preview.pos[0] - hp_preview.pivot[0];
                        float oy = hp_preview.pos[1] - hp_preview.pivot[1];
                        float oz = hp_preview.pos[2] - hp_preview.pivot[2];
                        float d  = sqrtf(ox*ox + oy*oy + oz*oz);
                        if (d < 0.001f) d = 0.001f;
                        float theta = atan2f(ox, -oz);
                        float phi   = asinf(std::clamp(oy / d, -1.f, 1.f));
                        theta += delta.x * ctrl.look_sens * (3.14159265f / 180.f);
                        phi   += delta.y * ctrl.look_sens * (3.14159265f / 180.f);
                        phi    = std::clamp(phi, -1.552f, 1.552f);
                        hp_preview.pos[0] = hp_preview.pivot[0] + d * cosf(phi) * sinf(theta);
                        hp_preview.pos[1] = hp_preview.pivot[1] + d * sinf(phi);
                        hp_preview.pos[2] = hp_preview.pivot[2] - d * cosf(phi) * cosf(theta);
                        hydra_cam_changed = true;

                    // MMB — screen-space pan (identical to main viewport)
                    } else if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                        float ox = hp_preview.pos[0] - hp_preview.pivot[0];
                        float oy = hp_preview.pos[1] - hp_preview.pivot[1];
                        float oz = hp_preview.pos[2] - hp_preview.pivot[2];
                        float d  = sqrtf(ox*ox + oy*oy + oz*oz);
                        if (d < 0.001f) d = 0.001f;
                        float fx = -ox/d, fy = -oy/d, fz = -oz/d;
                        float ux = -fz, uy = 0.f, uz = fx;
                        float ulen = sqrtf(ux*ux + uz*uz);
                        if (ulen > 1e-4f) { ux /= ulen; uz /= ulen; }
                        float vx = uy*fz - uz*fy;
                        float vy = uz*fx - ux*fz;
                        float vz = ux*fy - uy*fx;
                        float half_h = tanf(hp_preview.vfov * 0.5f * 3.14159265f / 180.f);
                        float scale  = d * 2.f * half_h / hh;
                        float pdx = (-delta.x * ux + delta.y * vx) * scale;
                        float pdy = (-delta.x * uy + delta.y * vy) * scale;
                        float pdz = (-delta.x * uz + delta.y * vz) * scale;
                        hp_preview.pos[0]   += pdx; hp_preview.pivot[0] += pdx;
                        hp_preview.pos[1]   += pdy; hp_preview.pivot[1] += pdy;
                        hp_preview.pos[2]   += pdz; hp_preview.pivot[2] += pdz;
                        hydra_cam_changed = true;
                    }
                }
                if (ImGui::IsItemHovered()) {
                    float scroll = ImGui::GetIO().MouseWheel;
                    if (scroll != 0.0f) {
                        float ox = hp_preview.pos[0] - hp_preview.pivot[0];
                        float oy = hp_preview.pos[1] - hp_preview.pivot[1];
                        float oz = hp_preview.pos[2] - hp_preview.pivot[2];
                        float scale = 1.f - scroll * 0.1f;
                        if (scale < 0.01f) scale = 0.01f;
                        hp_preview.pos[0] = hp_preview.pivot[0] + ox * scale;
                        hp_preview.pos[1] = hp_preview.pivot[1] + oy * scale;
                        hp_preview.pos[2] = hp_preview.pivot[2] + oz * scale;
                        hydra_cam_changed = true;
                    }
                }
                if (hydra_cam_changed) {
                    ctrl.pos[0] = hp_preview.pos[0];
                    ctrl.pos[1] = hp_preview.pos[1];
                    ctrl.pos[2] = hp_preview.pos[2];
                    ctrl.orbit_pivot[0] = hp_preview.pivot[0];
                    ctrl.orbit_pivot[1] = hp_preview.pivot[1];
                    ctrl.orbit_pivot[2] = hp_preview.pivot[2];
                    cam_changed = true;
                }
                // RMB click (no drag) → pick focus distance at that pixel.
                if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    ImVec2 mp = ImGui::GetMousePos();
                    ImVec2 rm = ImGui::GetItemRectMin();
                    hp_preview.pick_px = (int)(mp.x - rm.x);
                    hp_preview.pick_py = (int)(mp.y - rm.y);
                    hp_preview.pick_requested = true;
                }
                // Rewind cursor so Image() draws over the same region.
                ImGui::SetCursorPos(cursor_before_cam);
            }

            if (hydra_preview_ready) {
                hydra_preview_ready = safe_hydra_preview_tick(hp_preview, anim, hw, hh);
                if (!hydra_preview_ready)
                    anim.preview_open = false;
            }

            // RMB pick result: move orbit pivot to surface hit (same as viewport RMB),
            // and update focus distance to match the hit depth.
            if (hp_preview.pick_result_dist > 0.f) {
                ctrl.orbit_pivot[0] = hp_preview.pick_result_hit[0];
                ctrl.orbit_pivot[1] = hp_preview.pick_result_hit[1];
                ctrl.orbit_pivot[2] = hp_preview.pick_result_hit[2];
                ctrl.focus_dist     = hp_preview.pick_result_dist;
                // Also update the Hydra camera pivot so the preview
                // orbits around the new hit point (window is focused,
                // so the normal sync path is skipped).
                hp_preview.pivot[0] = hp_preview.pick_result_hit[0];
                hp_preview.pivot[1] = hp_preview.pick_result_hit[1];
                hp_preview.pivot[2] = hp_preview.pick_result_hit[2];
                cam_changed = true;
            }

            VkDescriptorSet hydra_desc = hydra_preview_descriptor(hp_preview);
            if (hydra_preview_ready && hydra_desc && avail.x > 4.0f && avail.y > 4.0f) {
                ImGui::Image((ImTextureID)hydra_desc, avail, ImVec2(0,1), ImVec2(1,0));
            } else if (!hydra_preview_ready) {
                ImGui::TextDisabled("Hydra preview unavailable.");
            }
            ImGui::End();
        }
        // Outliner — selection from here doesn't reset accumulation
        outliner_draw(prims_sorted, mesh.objects,
                      ctrl.selected_sphere, ctrl.selected_mesh_obj, multi_sel);
        stats_draw(hw_info, live_stats);

        // GPU Architecture Viewer (toggled from Tools menu)
        arch_state.cupti_available = cupti_profiler.is_initialized();
        if (ctrl.show_gpu_arch) {
            arch_state.gpu_util_pct = live_stats.gpu_util_pct;
            arch_state.mem_util_pct = live_stats.mem_util_pct;
            arch_state.vram_used    = live_stats.vram_used_bytes;
            arch_state.vram_total   = hw_info.total_vram_bytes;
            gpu_arch_window_draw(arch_state, hw_info.name, hw_info.sm_count, &ctrl.show_gpu_arch);
        }

        batch_panel_draw(batch);

        // Material panel — edit / reassign / custom shader for selected object
        {
            auto mc = material_panel_draw(ctrl.selected_mesh_obj,
                                          mesh.all_prims, mesh.prims,
                                          mesh.objects, mesh.host_mats,
                                          frame_count);

            if (mc.materials && !mesh.host_mats.empty()) {
                // host_mats may have grown (new/duplicate) — reallocate GPU buffer
                cudaFree(mesh.d_mats);
                cudaMalloc(&mesh.d_mats,
                           mesh.host_mats.size() * sizeof(GpuMaterial));
                cudaMemcpy(mesh.d_mats, mesh.host_mats.data(),
                           mesh.host_mats.size() * sizeof(GpuMaterial),
                           cudaMemcpyHostToDevice);
                mesh.num_mats = (int)mesh.host_mats.size();
            }

            if (mc.triangles && mesh.d_prims && !mesh.prims.empty()) {
                cudaMemcpy(mesh.d_prims, mesh.prims.data(),
                           mesh.prims.size() * sizeof(Triangle),
                           cudaMemcpyHostToDevice);
            }

            // Custom shader (CUDA or Slang): upload 4 textures — base_color,
            // metallic_rough, emissive, normal — and assign them to the material.
            if (!mc.shader_base_pixels.empty() &&
                mc.shader_mat >= 0 && mc.shader_mat < (int)mesh.host_mats.size()) {

                auto upload = [&](const std::vector<uint8_t>& px, bool srgb) -> int {
                    if (px.empty()) return -1;
                    GpuTexture t = gpu_texture_upload_rgba8(px.data(), mc.shader_w, mc.shader_h, srgb);
                    mesh.gpu_textures.push_back(t);
                    mesh.tex_bytes += (size_t)mc.shader_w * mc.shader_h * 4;
                    return (int)mesh.gpu_textures.size() - 1;
                };

                GpuMaterial& m = mesh.host_mats[mc.shader_mat];
                m.base_color_tex     = upload(mc.shader_base_pixels,     /*srgb=*/false);
                m.metallic_rough_tex = upload(mc.shader_mr_pixels,       /*srgb=*/false);
                m.emissive_tex       = upload(mc.shader_emissive_pixels,  /*srgb=*/false);
                m.normal_tex         = upload(mc.shader_normal_pixels,    /*srgb=*/false);
                // Reset multipliers so textures drive values directly
                m.base_color   = make_float4(1.f, 1.f, 1.f, 1.f);
                m.roughness    = 1.0f;
                m.metallic     = 0.0f;
                m.emissive_factor = make_float4(1.f, 1.f, 1.f, 1.f);

                cudaFree(mesh.d_tex_hdls);
                mesh.d_tex_hdls = gpu_textures_upload_handles(mesh.gpu_textures);
                mesh.num_texs   = (int)mesh.gpu_textures.size();

                cudaFree(mesh.d_mats);
                cudaMalloc(&mesh.d_mats, mesh.host_mats.size() * sizeof(GpuMaterial));
                cudaMemcpy(mesh.d_mats, mesh.host_mats.data(),
                           mesh.host_mats.size() * sizeof(GpuMaterial),
                           cudaMemcpyHostToDevice);
                mesh.num_mats = (int)mesh.host_mats.size();
            }

            // ── Reset to original BSDF ────────────────────────────────
            // material_panel already restored host_mats[mat_idx]; we just
            // need to re-upload the GPU material buffer so the change is visible.
            if (mc.shader_reset &&
                mc.shader_reset_mat >= 0 &&
                mc.shader_reset_mat < (int)mesh.host_mats.size()) {
                cudaFree(mesh.d_mats);
                cudaMalloc(&mesh.d_mats, mesh.host_mats.size() * sizeof(GpuMaterial));
                cudaMemcpy(mesh.d_mats, mesh.host_mats.data(),
                           mesh.host_mats.size() * sizeof(GpuMaterial),
                           cudaMemcpyHostToDevice);
                mesh.num_mats = (int)mesh.host_mats.size();
            }

            if (mc) cam_changed = true;
        }

        // ── glTF load ─────────────────────────────────────────────────
        if (ctrl.load_gltf_requested) {
            ctrl.load_gltf_requested = false;
            if (load_gltf_into(ctrl.gltf_path, mesh, ctrl)) {
                // Replace sphere scene: free GPU spheres and clear CPU lists
                cudaFree(d_bvh);  d_bvh  = nullptr;
                cudaFree(d_prims); d_prims = nullptr;
                bvh_nodes.clear();
                prims_sorted.clear();
                ctrl.selected_sphere = -1;
                multi_sel.clear();
                cam_changed = true;

                // Open persistent USD stage for animation playback
                if (usd_anim_handle) { usd_anim_close(usd_anim_handle); usd_anim_handle = nullptr; }
                usd_last_frame_time = -1e30f;
                if (is_usd_scene_path(ctrl.gltf_path)) {
                    usd_anim_handle = usd_anim_open(ctrl.gltf_path, mesh.all_prims);
                }

                // Open URDF articulation handle for interactive joint control
                if (urdf_artic_handle) { urdf_articulation_close(urdf_artic_handle); urdf_artic_handle = nullptr; }
                robot_demo = RobotDemoState{};
                if (is_urdf_path(ctrl.gltf_path)) {
                    urdf_artic_handle = urdf_articulation_open(ctrl.gltf_path, mesh.all_prims);
                } else if (is_mjcf_path(ctrl.gltf_path)) {
                    urdf_artic_handle = mjcf_articulation_open(ctrl.gltf_path, mesh.all_prims);
                }
            }
        }

        // ── Import into existing scene ────────────────────────────────
        if (ctrl.import_requested) {
            ctrl.import_requested = false;
            if (import_into_scene(ctrl.gltf_path, mesh, ctrl)) {
                cam_changed = true;

                // Open URDF articulation for the imported file
                if (is_urdf_path(ctrl.gltf_path)) {
                    if (urdf_artic_handle) { urdf_articulation_close(urdf_artic_handle); urdf_artic_handle = nullptr; }
                    urdf_artic_handle = urdf_articulation_open(ctrl.gltf_path, mesh.all_prims);
                    robot_demo = RobotDemoState{};
                } else if (is_mjcf_path(ctrl.gltf_path)) {
                    if (urdf_artic_handle) { urdf_articulation_close(urdf_artic_handle); urdf_artic_handle = nullptr; }
                    urdf_artic_handle = mjcf_articulation_open(ctrl.gltf_path, mesh.all_prims);
                    robot_demo = RobotDemoState{};
                }
            }
        }

        // ── HDRI load / clear ─────────────────────────────────────────
        if (ctrl.load_hdri_requested) {
            ctrl.load_hdri_requested = false;
            hdri_free(hdri);
            if (ctrl.hdri_path[0] != '\0') {
                hdri = hdri_load(ctrl.hdri_path);
                ctrl.hdri_loaded = hdri.loaded;
            } else {
                ctrl.hdri_loaded = false;
            }
            cam_changed = true;
        }

        // ── NIM VLM recognition ───────────────────────────────────────
        // Auto-trigger: fire when accumulation has settled for nim_auto_frames
        if (ctrl.nim_auto_enabled && !ctrl.nim_busy && !ctrl.nim_request
            && frame_count > 0 && frame_count == ctrl.nim_auto_frames)
        {
            ctrl.nim_request = true;
        }

        // Handle connection ping request
        if (ctrl.nim_ping_request && !ctrl.nim_busy) {
            ctrl.nim_ping_request = false;
            NimVlmConfig ping_cfg = ctrl.nim_cfg;
            if (nim_thread.joinable()) nim_thread.join();
            nim_thread = std::thread([ping_cfg, &ctrl]() {
                bool ok = nim_vlm_check_connection(ping_cfg);
                ctrl.nim_connection_ok     = ok;
                ctrl.nim_connection_tested = true;
            });
        }

        // Handle Docker launch request
        if (ctrl.nim_docker_launch_req) {
            ctrl.nim_docker_launch_req = false;
            nim_vlm_launch_docker(ctrl.nim_cfg, ctrl.nim_docker_error, sizeof(ctrl.nim_docker_error));
        }

        // ── Save Render as EXR ───────────────────────────────────────
        // EXR save is deferred to after the render — see below ++frame_count

        if (ctrl.nim_request && !ctrl.nim_busy) {
            ctrl.nim_request = false;
            std::vector<float> cpu_accum((size_t)rt_w * rt_h * 4);
            cudaMemcpy(cpu_accum.data(), d_accum,
                       cpu_accum.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            ctrl.nim_busy   = true;
            ctrl.nim_result = {};
            if (nim_thread.joinable()) nim_thread.join();
            nim_thread = std::thread([
                cfg    = ctrl.nim_cfg,
                pixels = std::move(cpu_accum),
                w = rt_w, h = rt_h,
                &ctrl]() mutable
            {
                NimVlmResult r = nim_vlm_recognize(cfg, pixels.data(), w, h);
                ctrl.nim_result = r;
                ctrl.nim_busy   = false;
                // Update connection status based on recognition result
                ctrl.nim_connection_ok     = r.success || r.error_msg[0] == '\0';
                ctrl.nim_connection_tested = true;
            });
        }

        // ── Batch Asset Library ───────────────────────────────────────
        // State: Loading — main loads the model and resets accumulation
        if (batch.state == BatchState::Loading && batch.do_load) {
            batch.do_load = false;
            const std::string& model_path = batch.files[batch.current_idx];
            bool ok = load_gltf_into(model_path, mesh, ctrl);
            if (ok) {
                // Clear sphere scene to avoid geometry conflicts
                cudaFree(d_bvh);   d_bvh   = nullptr;
                cudaFree(d_prims); d_prims = nullptr;
                bvh_nodes.clear();
                prims_sorted.clear();
                ctrl.selected_sphere   = -1;
                ctrl.selected_mesh_obj = -1;
                multi_sel.clear();
                // Open USD anim handle for batch
                if (usd_anim_handle) { usd_anim_close(usd_anim_handle); usd_anim_handle = nullptr; }
                usd_last_frame_time = -1e30f;
                if (is_usd_scene_path(model_path))
                    usd_anim_handle = usd_anim_open(model_path, mesh.all_prims);
            }
            pathtracer_reset_accum(d_accum, rt_w, rt_h);
            frame_count = 0;
            cam_changed = true;
            if (ok) {
                batch.state = BatchState::Rendering;
                snprintf(batch.status, sizeof(batch.status),
                         "Rendering [%d/%d]: %.50s",
                         batch.current_idx + 1, (int)batch.files.size(),
                         std::filesystem::path(model_path).filename().string().c_str());
            } else {
                // Skip broken model
                snprintf(batch.status, sizeof(batch.status),
                         "Load failed, skipping [%d/%d]",
                         batch.current_idx + 1, (int)batch.files.size());
                batch.current_idx++;
                if (batch.current_idx >= (int)batch.files.size()) {
                    batch.state = BatchState::AllDone;
                    snprintf(batch.status, sizeof(batch.status),
                             "All done! %d models saved.", (int)batch.log.size());
                } else {
                    batch.do_load = true;
                }
            }
        }

        // State: Rendering — check if we've accumulated enough frames
        if (batch.state == BatchState::Rendering &&
            frame_count >= batch.config.target_frames)
        {
            if (batch.config.use_nim) {
                batch.state       = BatchState::Recognizing;
                batch.nim_started = false;
                batch.nim_result  = {};
                snprintf(batch.status, sizeof(batch.status),
                         "Recognizing [%d/%d]...",
                         batch.current_idx + 1, (int)batch.files.size());
            } else {
                // Skip NIM: use filename as description
                batch.nim_result = {};
                auto stem = std::filesystem::path(batch.files[batch.current_idx]).stem().string();
                snprintf(batch.nim_result.category,    sizeof(batch.nim_result.category),    "prop");
                snprintf(batch.nim_result.description, sizeof(batch.nim_result.description), "%s", stem.c_str());
                snprintf(batch.nim_result.object_type, sizeof(batch.nim_result.object_type), "%s", stem.c_str());
                batch_save(batch);
                // Advance
                batch.current_idx++;
                if (batch.current_idx >= (int)batch.files.size()) {
                    batch.state = BatchState::AllDone;
                    snprintf(batch.status, sizeof(batch.status),
                             "All done! %d models saved.", (int)batch.log.size());
                } else {
                    batch.state   = BatchState::Loading;
                    batch.do_load = true;
                    snprintf(batch.status, sizeof(batch.status),
                             "Loading [%d/%d]...",
                             batch.current_idx + 1, (int)batch.files.size());
                }
            }
        }

        // State: Recognizing — launch NIM thread once
        if (batch.state == BatchState::Recognizing &&
            !batch.nim_started && !batch.nim_busy)
        {
            batch.nim_started = true;
            batch.nim_busy    = true;
            if (nim_thread.joinable()) nim_thread.join();
            std::vector<float> px((size_t)rt_w * rt_h * 4);
            cudaMemcpy(px.data(), d_accum, px.size() * sizeof(float), cudaMemcpyDeviceToHost);
            NimVlmConfig nim_cfg = ctrl.nim_cfg;
            nim_thread = std::thread([nim_cfg, pixels = std::move(px), w = rt_w, h = rt_h, &batch]() mutable {
                batch.nim_result = nim_vlm_recognize(nim_cfg, pixels.data(), w, h);
                batch.nim_busy   = false;
            });
        }

        // State: Recognizing — NIM thread completed
        if (batch.state == BatchState::Recognizing &&
            batch.nim_started && !batch.nim_busy)
        {
            if (!batch.nim_result.success) {
                // Fallback to filename
                auto stem = std::filesystem::path(batch.files[batch.current_idx]).stem().string();
                snprintf(batch.nim_result.category,    sizeof(batch.nim_result.category),    "prop");
                snprintf(batch.nim_result.description, sizeof(batch.nim_result.description), "%s", stem.c_str());
                snprintf(batch.nim_result.object_type, sizeof(batch.nim_result.object_type), "%s", stem.c_str());
            }
            batch_save(batch);
            // Advance to next model
            batch.current_idx++;
            if (batch.current_idx >= (int)batch.files.size()) {
                batch.state = BatchState::AllDone;
                snprintf(batch.status, sizeof(batch.status),
                         "All done! %d models saved.", (int)batch.log.size());
            } else {
                batch.state       = BatchState::Loading;
                batch.do_load     = true;
                batch.nim_started = false;
                snprintf(batch.status, sizeof(batch.status),
                         "Loading [%d/%d]...",
                         batch.current_idx + 1, (int)batch.files.size());
            }
        }

        // ── Resize ───────────────────────────────────────────────────
        if (vp.resized && (int)vp.size.x > 0 && (int)vp.size.y > 0) {
            pending_rt_w = (int)vp.size.x;
            pending_rt_h = (int)vp.size.y;
            pending_rt_resize = true;
        }

        // ── Look direction ────────────────────────────────────────────
        float yaw_rad   = ctrl.yaw   * (3.14159265f / 180.f);
        float pitch_rad = ctrl.pitch * (3.14159265f / 180.f);
        float look_x =  cosf(pitch_rad) * sinf(yaw_rad);
        float look_y =  sinf(pitch_rad);
        float look_z = -cosf(pitch_rad) * cosf(yaw_rad);

        // ── RMB free-look ─────────────────────────────────────────────
        bool rmb_active = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
                          && !io.WantCaptureMouse;
        {
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);
            if (rmb_active && rmb_was_down) {
                float dx = (float)(mx - prev_mx);
                float dy = (float)(my - prev_my);
                if (dx != 0.f || dy != 0.f) {
                    ctrl.yaw   += dx * ctrl.look_sens;
                    ctrl.pitch -= dy * ctrl.look_sens;
                    if (ctrl.pitch >  89.f) ctrl.pitch =  89.f;
                    if (ctrl.pitch < -89.f) ctrl.pitch = -89.f;
                    yaw_rad   = ctrl.yaw   * (3.14159265f / 180.f);
                    pitch_rad = ctrl.pitch * (3.14159265f / 180.f);
                    look_x =  cosf(pitch_rad) * sinf(yaw_rad);
                    look_y =  sinf(pitch_rad);
                    look_z = -cosf(pitch_rad) * cosf(yaw_rad);
                    cam_changed = true;
                }
            }
            glfwSetInputMode(window, GLFW_CURSOR,
                rmb_active ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            prev_mx = mx; prev_my = my;
            rmb_was_down = rmb_active;
        }

        // ── WASD (all modes; RMB held always passes keys through) ─────
        // Forward/right are derived from the actual camera look direction so they
        // stay correct after orbiting (ctrl.yaw isn't updated during orbit).
        // Q = up, R = down (world Y). Up/down arrow keys are reserved for orbit.
        if (!io.WantCaptureKeyboard || rmb_active) {
            float dt  = frame_ms * 0.001f;
            float spd = ctrl.move_speed * dt;
            bool orbit = (ctrl.interact_mode == InteractMode::Orbit);

            // Compute horizontal forward and right from the current view direction.
            // In orbit mode use pos→pivot projection onto XZ plane.
            // In FPS mode use yaw angle (pitch ignored so movement stays horizontal).
            float fwd_x, fwd_z, rgt_x, rgt_z;
            if (orbit) {
                float dx = ctrl.orbit_pivot[0] - ctrl.pos[0];
                float dz = ctrl.orbit_pivot[2] - ctrl.pos[2];
                float d  = sqrtf(dx*dx + dz*dz);
                if (d > 0.001f) { fwd_x = dx/d; fwd_z = dz/d; }
                else            { fwd_x = sinf(yaw_rad); fwd_z = -cosf(yaw_rad); }
                rgt_x =  fwd_z; rgt_z = -fwd_x;   // 90° CW around Y
            } else {
                fwd_x =  sinf(yaw_rad); fwd_z = -cosf(yaw_rad);
                rgt_x =  cosf(yaw_rad); rgt_z =  sinf(yaw_rad);
            }

            auto pan = [&](float dx, float dy, float dz) {
                ctrl.pos[0] += dx; ctrl.pos[1] += dy; ctrl.pos[2] += dz;
                if (orbit) {
                    ctrl.orbit_pivot[0] += dx;
                    ctrl.orbit_pivot[1] += dy;
                    ctrl.orbit_pivot[2] += dz;
                }
                cam_changed = true;
            };
            // In Move mode, W/E/R are gizmo shortcuts — don't move camera
            bool move_mode = (ctrl.interact_mode == InteractMode::Move);
            if (!move_mode && (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS))
                pan( fwd_x*spd, 0.f,  fwd_z*spd);
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS)
                pan(-fwd_x*spd, 0.f, -fwd_z*spd);
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS)
                pan( rgt_x*spd, 0.f,  rgt_z*spd);
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                pan(-rgt_x*spd, 0.f, -rgt_z*spd);
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                pan(0.f,  spd, 0.f);
            if (!move_mode && (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS))
                pan(0.f, -spd, 0.f);
            // R was used for nothing in WASD — no conflict
        }

        // ── Mode-change sync (no jump on switch) ─────────────────────
        {
            static InteractMode s_prev = ctrl.interact_mode;
            if (ctrl.interact_mode != s_prev) {
                if (ctrl.interact_mode == InteractMode::Orbit) {
                    // FPS → Orbit: place pivot directly ahead so camera doesn't move
                    ctrl.orbit_pivot[0] = ctrl.pos[0] + look_x * ctrl.orbit_dist;
                    ctrl.orbit_pivot[1] = ctrl.pos[1] + look_y * ctrl.orbit_dist;
                    ctrl.orbit_pivot[2] = ctrl.pos[2] + look_z * ctrl.orbit_dist;
                } else if (ctrl.interact_mode == InteractMode::Select) {
                    // Orbit/Move → FPS: derive yaw/pitch from pos→pivot direction
                    float dx = ctrl.orbit_pivot[0] - ctrl.pos[0];
                    float dy = ctrl.orbit_pivot[1] - ctrl.pos[1];
                    float dz = ctrl.orbit_pivot[2] - ctrl.pos[2];
                    float d  = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (d > 0.001f) {
                        ctrl.pitch = asinf(fmaxf(-1.f, fminf(1.f, dy/d))) * (180.f / 3.14159265f);
                        float cp = cosf(ctrl.pitch * 3.14159265f / 180.f);
                        if (cp > 1e-5f)
                            ctrl.yaw = atan2f(dx/d, -dz/d) * (180.f / 3.14159265f);
                    }
                }
                // Orbit ↔ Move: no sync needed, both share orbit state
                s_prev = ctrl.interact_mode;
            }
        }

        // ── W / E / R gizmo mode shortcuts (Move mode only) ──────────
        if (ctrl.interact_mode == InteractMode::Move && !io.WantCaptureKeyboard) {
            if (ImGui::IsKeyPressed(ImGuiKey_W, false))
                ctrl.gizmo_mode = GizmoMode::Translate;
            if (ImGui::IsKeyPressed(ImGuiKey_E, false))
                ctrl.gizmo_mode = GizmoMode::Rotate;
            if (ImGui::IsKeyPressed(ImGuiKey_R, false))
                ctrl.gizmo_mode = GizmoMode::Scale;
        }

        // ── F key: frame selected object (or entire scene if nothing selected) ──
        if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
            float3 mn = make_float3( 1e30f,  1e30f,  1e30f);
            float3 mx = make_float3(-1e30f, -1e30f, -1e30f);
            bool has_sel = ctrl.selected_mesh_obj >= 0 && !multi_sel.empty();

            for (const Triangle& tri : mesh.all_prims) {
                if (has_sel) {
                    // Only selected objects
                    bool in_sel = false;
                    for (int idx : multi_sel) {
                        if (idx >= 0 && idx < (int)mesh.objects.size() &&
                            tri.obj_id == mesh.objects[idx].obj_id) { in_sel = true; break; }
                    }
                    if (!in_sel) continue;
                }
                float3 verts[3] = { tri.v0, tri.v1, tri.v2 };
                for (auto& v : verts) {
                    if (v.x < mn.x) mn.x = v.x; if (v.x > mx.x) mx.x = v.x;
                    if (v.y < mn.y) mn.y = v.y; if (v.y > mx.y) mx.y = v.y;
                    if (v.z < mn.z) mn.z = v.z; if (v.z > mx.z) mx.z = v.z;
                }
            }

            // Also consider spheres if no mesh
            if (mn.x > mx.x && !prims_sorted.empty()) {
                for (auto& s : prims_sorted) {
                    if (s.center.x - s.radius < mn.x) mn.x = s.center.x - s.radius;
                    if (s.center.x + s.radius > mx.x) mx.x = s.center.x + s.radius;
                    if (s.center.y - s.radius < mn.y) mn.y = s.center.y - s.radius;
                    if (s.center.y + s.radius > mx.y) mx.y = s.center.y + s.radius;
                    if (s.center.z - s.radius < mn.z) mn.z = s.center.z - s.radius;
                    if (s.center.z + s.radius > mx.z) mx.z = s.center.z + s.radius;
                }
            }

            if (mn.x <= mx.x) {
                float cx = (mn.x + mx.x) * 0.5f;
                float cy = (mn.y + mx.y) * 0.5f;
                float cz = (mn.z + mx.z) * 0.5f;
                float dx = mx.x - mn.x, dy = mx.y - mn.y, dz = mx.z - mn.z;
                float diag = sqrtf(dx*dx + dy*dy + dz*dz);
                float dist = diag * 0.8f / tanf(ctrl.vfov * 0.5f * 3.14159265f / 180.f);

                ctrl.orbit_pivot[0] = cx;
                ctrl.orbit_pivot[1] = cy;
                ctrl.orbit_pivot[2] = cz;

                // Place camera looking from current direction but at correct distance
                float fx = ctrl.pos[0] - cx, fy = ctrl.pos[1] - cy, fz = ctrl.pos[2] - cz;
                float fl = sqrtf(fx*fx + fy*fy + fz*fz);
                if (fl > 1e-5f) { fx/=fl; fy/=fl; fz/=fl; }
                else { fx = 0; fy = 0.3f; fz = 1.0f; }

                ctrl.pos[0] = cx + fx * dist;
                ctrl.pos[1] = cy + fy * dist;
                ctrl.pos[2] = cz + fz * dist;
                ctrl.cam_from[0] = ctrl.pos[0];
                ctrl.cam_from[1] = ctrl.pos[1];
                ctrl.cam_from[2] = ctrl.pos[2];
                ctrl.cam_at[0] = cx; ctrl.cam_at[1] = cy; ctrl.cam_at[2] = cz;
                ctrl.orbit_dist = dist;
                ctrl.interact_mode = InteractMode::Orbit;
                cam_changed = true;
            }
        }

        // ── ORBIT drag + scroll: arcball (pos rotates around pivot) ──
        // Alt held = temporary orbit override regardless of current mode.
        // Move mode shares the orbit camera — orbit with Alt or when not on gizmo.
        // Release light drag if mouse is up (unconditional — prevents stuck state).
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
            ctrl.dragging_light = -1;
        // Suppress camera drag when the light HUD or IK gizmo has captured the mouse.
        static bool s_ik_gizmo_active = false;
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
            s_ik_gizmo_active = false;
        bool light_hud_active = (ctrl.dragging_light >= 0) || s_ik_gizmo_active;
        bool alt_orbit = io.KeyAlt && vp.hovered;
        if ((ctrl.interact_mode == InteractMode::Orbit || alt_orbit) && !light_hud_active) {
            if (vp.lmb_dragging) {
                float ox = ctrl.pos[0] - ctrl.orbit_pivot[0];
                float oy = ctrl.pos[1] - ctrl.orbit_pivot[1];
                float oz = ctrl.pos[2] - ctrl.orbit_pivot[2];
                float dist = sqrtf(ox*ox + oy*oy + oz*oz);
                if (dist < 0.001f) dist = 0.001f;
                float theta = atan2f(ox, -oz);
                float phi   = asinf(fmaxf(-1.f, fminf(1.f, oy / dist)));
                theta += vp.lmb_drag_delta.x * ctrl.look_sens * (3.14159265f / 180.f);
                phi   += vp.lmb_drag_delta.y * ctrl.look_sens * (3.14159265f / 180.f);
                phi    = fmaxf(-1.552f, fminf(1.552f, phi));  // clamp ~±89°
                ctrl.pos[0] = ctrl.orbit_pivot[0] + dist * cosf(phi) * sinf(theta);
                ctrl.pos[1] = ctrl.orbit_pivot[1] + dist * sinf(phi);
                ctrl.pos[2] = ctrl.orbit_pivot[2] - dist * cosf(phi) * cosf(theta);
                cam_changed = true;
            }
            if (vp.hovered && vp.scroll_y != 0.f) {
                float ox = ctrl.pos[0] - ctrl.orbit_pivot[0];
                float oy = ctrl.pos[1] - ctrl.orbit_pivot[1];
                float oz = ctrl.pos[2] - ctrl.orbit_pivot[2];
                float scale = 1.f - vp.scroll_y * 0.1f;
                if (scale < 0.01f) scale = 0.01f;
                ctrl.pos[0] = ctrl.orbit_pivot[0] + ox * scale;
                ctrl.pos[1] = ctrl.orbit_pivot[1] + oy * scale;
                ctrl.pos[2] = ctrl.orbit_pivot[2] + oz * scale;
                cam_changed = true;
            }
        }

        // ── Camera derivation ─────────────────────────────────────────
        // Move mode shares orbit camera (look at pivot, not FPS yaw/pitch)
        if (ctrl.interact_mode == InteractMode::Orbit ||
            ctrl.interact_mode == InteractMode::Move  || alt_orbit) {
            ctrl.cam_from[0] = ctrl.pos[0];
            ctrl.cam_from[1] = ctrl.pos[1];
            ctrl.cam_from[2] = ctrl.pos[2];
            ctrl.cam_at[0]   = ctrl.orbit_pivot[0];
            ctrl.cam_at[1]   = ctrl.orbit_pivot[1];
            ctrl.cam_at[2]   = ctrl.orbit_pivot[2];
            // Keep orbit_dist in sync for display + FPS→Orbit initial pivot
            float dx = ctrl.pos[0]-ctrl.orbit_pivot[0];
            float dy = ctrl.pos[1]-ctrl.orbit_pivot[1];
            float dz = ctrl.pos[2]-ctrl.orbit_pivot[2];
            ctrl.orbit_dist = sqrtf(dx*dx + dy*dy + dz*dz);
        } else {
            ctrl.cam_from[0] = ctrl.pos[0];
            ctrl.cam_from[1] = ctrl.pos[1];
            ctrl.cam_from[2] = ctrl.pos[2];
            ctrl.cam_at[0]   = ctrl.pos[0] + look_x;
            ctrl.cam_at[1]   = ctrl.pos[1] + look_y;
            ctrl.cam_at[2]   = ctrl.pos[2] + look_z;
        }

        Camera cam = Camera::make(
            make_float3(ctrl.cam_from[0], ctrl.cam_from[1], ctrl.cam_from[2]),
            make_float3(ctrl.cam_at[0],   ctrl.cam_at[1],   ctrl.cam_at[2]),
            make_float3(0.f, 1.f, 0.f),
            ctrl.vfov, (float)rt_w / (float)rt_h,
            ctrl.aperture, ctrl.focus_dist);

        // ── MMB drag: pan in screen space using built camera vectors ──
        if (vp.mmb_dragging &&
            (vp.mmb_drag_delta.x != 0.f || vp.mmb_drag_delta.y != 0.f))
        {
            float dist = ctrl.orbit_dist > 0.1f ? ctrl.orbit_dist : 0.1f;
            float half_h = tanf(ctrl.vfov * 0.5f * 3.14159265f / 180.f);
            float scale  = dist * 2.f * half_h / vp.size.y;

            // cam.u = screen right, cam.v = screen up (always correct after Camera::make)
            float dx = (-vp.mmb_drag_delta.x * cam.u.x + vp.mmb_drag_delta.y * cam.v.x) * scale;
            float dy = (-vp.mmb_drag_delta.x * cam.u.y + vp.mmb_drag_delta.y * cam.v.y) * scale;
            float dz = (-vp.mmb_drag_delta.x * cam.u.z + vp.mmb_drag_delta.y * cam.v.z) * scale;

            ctrl.pos[0] += dx; ctrl.pos[1] += dy; ctrl.pos[2] += dz;
            ctrl.orbit_pivot[0] += dx;
            ctrl.orbit_pivot[1] += dy;
            ctrl.orbit_pivot[2] += dz;

            // Rebuild cam with updated position so gizmos/picking this frame are correct
            cam = Camera::make(
                make_float3(ctrl.pos[0], ctrl.pos[1], ctrl.pos[2]),
                make_float3(ctrl.orbit_pivot[0], ctrl.orbit_pivot[1], ctrl.orbit_pivot[2]),
                make_float3(0.f, 1.f, 0.f),
                ctrl.vfov, (float)rt_w / (float)rt_h,
                ctrl.aperture, ctrl.focus_dist);
            cam_changed = true;
        }

        // ── IK gizmo: XYZ move gizmo at end-effector for IK dragging ─
        bool ik_gizmo_consuming = false;
        if (robot_demo.ik_enabled && robot_demo.recording && urdf_artic_handle &&
            !robot_demo.playing)
        {
            float3 ee = urdf_end_effector_pos(urdf_artic_handle);
            float ik_aspect = (rt_w > 0 && rt_h > 0) ? (float)rt_w / (float)rt_h : 1.f;

            // Persistent IK target — accumulates gizmo deltas instead of using ee pos.
            static float3 ik_target = {0, 0, 0};
            static bool   ik_target_init = false;
            static int    ik_drag_axis = 0;
            bool ik_needs_solve = false;

            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) ik_drag_axis = 0;

            // Sync target to FK ee when not dragging (matches what IK solver sees)
            if (ik_drag_axis == 0) {
                ik_target = urdf_fk_ee_pos(urdf_artic_handle);
                ik_target_init = true;
            }

            ImVec2 ee_screen;
            float ee_depth = world_to_screen(cam, ee, vp.origin, vp.size, ctrl.vfov, ik_aspect, ee_screen);

            if (ee_depth > 0.f) {
                ImDrawList* dl = ImGui::GetForegroundDrawList();
                const float ARROW_PX = 80.f;
                const float HIT_PX   = 8.f;
                const float TIP_PX   = 10.f;
                const float CENTER_SZ = 9.f;

                float half_h = tanf(ctrl.vfov * 0.5f * 3.14159265f / 180.f);
                float scale  = ee_depth * 2.f * half_h / vp.size.y * ARROW_PX;

                float3 axis_dir[4]  = { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} };
                ImU32  axis_col[4]  = { 0,
                    IM_COL32(255, 100, 100, 255),
                    IM_COL32(100, 255, 100, 255),
                    IM_COL32(100, 150, 255, 255) };

                ImVec2 tip[4];
                for (int i = 1; i <= 3; ++i) {
                    float3 w = { ee.x + axis_dir[i].x * scale,
                                 ee.y + axis_dir[i].y * scale,
                                 ee.z + axis_dir[i].z * scale };
                    world_to_screen(cam, w, vp.origin, vp.size, ctrl.vfov, ik_aspect, tip[i]);
                }

                ImVec2 mouse = ImGui::GetIO().MousePos;
                int hovered = 0;
                if (ik_drag_axis == 0) {
                    if (fabsf(mouse.x - ee_screen.x) < CENTER_SZ &&
                        fabsf(mouse.y - ee_screen.y) < CENTER_SZ)
                        hovered = 4;
                    else {
                        for (int i = 1; i <= 3; ++i) {
                            if (seg_dist_sq(mouse.x, mouse.y,
                                ee_screen.x, ee_screen.y, tip[i].x, tip[i].y) < HIT_PX*HIT_PX) {
                                hovered = i; break;
                            }
                        }
                    }
                }

                if (hovered > 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    ik_drag_axis = hovered;
                    s_ik_gizmo_active = true;
                }

                ik_gizmo_consuming = (ik_drag_axis > 0) && ImGui::IsMouseDown(ImGuiMouseButton_Left);
                int display = ik_drag_axis > 0 ? ik_drag_axis : hovered;

                // Draw axes
                for (int i = 1; i <= 3; ++i) {
                    ImU32 c = (i == display) ? IM_COL32(255, 255, 100, 255) : axis_col[i];
                    dl->AddLine(ee_screen, tip[i], c, 2.5f);
                    float ax = tip[i].x - ee_screen.x, ay = tip[i].y - ee_screen.y;
                    float len = sqrtf(ax*ax + ay*ay);
                    if (len > 0.f) {
                        ax /= len; ay /= len;
                        float px = -ay, py = ax;
                        dl->AddTriangleFilled(
                            tip[i],
                            { tip[i].x - ax*TIP_PX + px*(TIP_PX*0.4f), tip[i].y - ay*TIP_PX + py*(TIP_PX*0.4f) },
                            { tip[i].x - ax*TIP_PX - px*(TIP_PX*0.4f), tip[i].y - ay*TIP_PX - py*(TIP_PX*0.4f) },
                            c);
                    }
                }
                // Center diamond
                ImU32 cc = (display == 4) ? IM_COL32(255, 255, 100, 255) : IM_COL32(255, 200, 100, 220);
                ImVec2 diamond[4] = {
                    { ee_screen.x, ee_screen.y - CENTER_SZ },
                    { ee_screen.x + CENTER_SZ, ee_screen.y },
                    { ee_screen.x, ee_screen.y + CENTER_SZ },
                    { ee_screen.x - CENTER_SZ, ee_screen.y }
                };
                dl->AddConvexPolyFilled(diamond, 4, cc);
                dl->AddPolyline(diamond, 4, IM_COL32(255, 255, 255, 150), ImDrawFlags_Closed, 1.5f);
                dl->AddText({ ee_screen.x + CENTER_SZ + 4.f, ee_screen.y - 6.f },
                            IM_COL32(255, 200, 100, 220), "IK");

                // Apply drag: accumulate delta onto ik_target, then solve IK to target
                ImVec2 delta = ImGui::GetIO().MouseDelta;
                if (ik_drag_axis > 0 && (delta.x != 0.f || delta.y != 0.f)) {
                    float wpp = scale / ARROW_PX;
                    float3 world_delta = {0, 0, 0};

                    if (ik_drag_axis == 4) {
                        // Camera-space translate
                        world_delta = {
                            (cam.u.x * delta.x - cam.v.x * delta.y) * wpp,
                            (cam.u.y * delta.x - cam.v.y * delta.y) * wpp,
                            (cam.u.z * delta.x - cam.v.z * delta.y) * wpp };
                    } else {
                        float3 ad = axis_dir[ik_drag_axis];
                        float sax = tip[ik_drag_axis].x - ee_screen.x;
                        float say = tip[ik_drag_axis].y - ee_screen.y;
                        float slen = sqrtf(sax*sax + say*say);
                        if (slen > 0.f) {
                            float proj = (delta.x * sax + delta.y * say) / slen;
                            float move = proj * wpp;
                            world_delta = { ad.x * move, ad.y * move, ad.z * move };
                        }
                    }

                    // Accumulate onto the persistent target (NOT the ee position)
                    ik_target.x += world_delta.x;
                    ik_target.y += world_delta.y;
                    ik_target.z += world_delta.z;

                    ik_needs_solve = true;
                }
            }

            // ── Chain grab points (one per movable joint, except first/last) ─
            static int grab_dragging = -1;  // which joint index is being dragged
            static float3 grab_target = {0,0,0};
            bool grab_needs_solve = false;

            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) grab_dragging = -1;

            int chain_len = urdf_ik_chain_length(urdf_artic_handle);

            if (chain_len >= 3 && ik_drag_axis == 0) {
                ImDrawList* dl = ImGui::GetForegroundDrawList();
                ImVec2 mouse = ImGui::GetIO().MousePos;

                // Draw grab points for joints 1 .. chain_len-2
                for (int ji = 1; ji < chain_len - 1; ji++) {
                    float3 jpos = urdf_joint_pos(urdf_artic_handle, ji);
                    ImVec2 js;
                    float jd = world_to_screen(cam, jpos, vp.origin, vp.size,
                                               ctrl.vfov, ik_aspect, js);
                    if (jd <= 0.f) continue;

                    const float R = 6.f;
                    float dx = mouse.x - js.x, dy = mouse.y - js.y;
                    bool hov = (dx*dx + dy*dy) <= (R + 5.f) * (R + 5.f);
                    bool active = (grab_dragging == ji);

                    if (hov && grab_dragging < 0 && !ik_gizmo_consuming &&
                        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        grab_dragging = ji;
                        grab_target = jpos;
                        s_ik_gizmo_active = true;
                    }
                    if (active) ik_gizmo_consuming = true;

                    // Color gradient along chain: blue at base → cyan at tip
                    float t = (float)ji / (float)(chain_len - 1);
                    int r = (int)(80 + 50 * t), g = (int)(150 + 80 * t), b = 230;
                    ImU32 col = active ? IM_COL32(100, 240, 255, 255)
                              : hov   ? IM_COL32(120, 220, 255, 240)
                                      : IM_COL32(r, g, b, 180);

                    dl->AddCircleFilled(js, R, col, 10);
                    dl->AddCircle(js, R, IM_COL32(255, 255, 255, 140), 10, 1.f);

                    // Drag
                    if (active) {
                        ImVec2 delta = ImGui::GetIO().MouseDelta;
                        if (delta.x != 0.f || delta.y != 0.f) {
                            float half_h = tanf(ctrl.vfov * 0.5f * 3.14159265f / 180.f);
                            float wpp = jd * 2.f * half_h / vp.size.y;
                            grab_target.x += (cam.u.x * delta.x - cam.v.x * delta.y) * wpp;
                            grab_target.y += (cam.u.y * delta.x - cam.v.y * delta.y) * wpp;
                            grab_target.z += (cam.u.z * delta.x - cam.v.z * delta.y) * wpp;
                            grab_needs_solve = true;
                        }

                        // Target line
                        ImVec2 ts;
                        if (world_to_screen(cam, grab_target, vp.origin, vp.size,
                                            ctrl.vfov, ik_aspect, ts) > 0.f)
                            dl->AddLine(js, ts, IM_COL32(100, 240, 255, 100), 1.f);
                    }
                }
            }

            // Solve: grab point takes priority over ee gizmo
            if (grab_needs_solve && grab_dragging >= 1) {
                urdf_solve_ik_joint(urdf_artic_handle, grab_dragging, grab_target);
                urdf_repose(urdf_artic_handle, mesh.all_prims);
                reupload_visible_prims(mesh);
                if (mesh.d_bvh && !mesh.bvh_nodes.empty())
                    bvh_refit_triangles(mesh.bvh_nodes, mesh.prims, mesh.d_bvh);
                ++mesh.scene_version;
                cam_changed = true;
            } else if (ik_needs_solve) {
                urdf_solve_ik(urdf_artic_handle, ik_target);
                urdf_repose(urdf_artic_handle, mesh.all_prims);
                reupload_visible_prims(mesh);
                if (mesh.d_bvh && !mesh.bvh_nodes.empty())
                    bvh_refit_triangles(mesh.bvh_nodes, mesh.prims, mesh.d_bvh);
                ++mesh.scene_version;
                cam_changed = true;
            }
        }

        // ── Gizmo (Move mode, sphere selected) ───────────────────────
        float3 gizmo_delta   = {0,0,0};
        int    gizmo_result  = 0;
        bool   gizmo_consuming = false;
        bool   mesh_moved    = false;   // set when gizmo translates mesh objects

        // ── Sphere gizmo ──────────────────────────────────────────────
        if (ctrl.overlay_gizmo && ctrl.selected_sphere >= 0 && ctrl.interact_mode == InteractMode::Move && !alt_orbit) {
            gizmo_result = draw_move_gizmo(
                ImGui::GetForegroundDrawList(), cam,
                prims_sorted[ctrl.selected_sphere].center,
                ctrl.vfov, (float)rt_w / (float)rt_h,
                vp.origin, vp.size, gizmo_delta);

            gizmo_consuming = (gizmo_result > 0) &&
                              ImGui::IsMouseDown(ImGuiMouseButton_Left);

            if (gizmo_delta.x != 0.f || gizmo_delta.y != 0.f || gizmo_delta.z != 0.f) {
                int idx = ctrl.selected_sphere;
                float3 updated = {
                    prims_sorted[idx].center.x + gizmo_delta.x,
                    prims_sorted[idx].center.y + gizmo_delta.y,
                    prims_sorted[idx].center.z + gizmo_delta.z };
                prims_sorted[idx].center = updated;
                rebuild_bvh(bvh_nodes, prims_sorted, &d_bvh, &d_prims,
                            ctrl.selected_sphere, updated);
                ctrl.orbit_pivot[0] = prims_sorted[ctrl.selected_sphere].center.x;
                ctrl.orbit_pivot[1] = prims_sorted[ctrl.selected_sphere].center.y;
                ctrl.orbit_pivot[2] = prims_sorted[ctrl.selected_sphere].center.z;
                cam_changed = true;
            }
        }

        // ── Mesh object gizmo (W=translate, E=rotate, R=scale) ────────
        if (ctrl.overlay_gizmo && ctrl.selected_mesh_obj >= 0 &&
            ctrl.selected_mesh_obj < (int)mesh.objects.size() &&
            !multi_sel.empty() &&
            ctrl.interact_mode == InteractMode::Move && !alt_orbit)
        {
            MeshObject& mobj = mesh.objects[ctrl.selected_mesh_obj];
            float3 pivot = mobj.centroid;

            // Dispatch gizmo by mode
            float3 gizmo_rotate = {0,0,0};
            float3 gizmo_scale_d = {0,0,0};
            int gr = 0;

            if (ctrl.gizmo_mode == GizmoMode::Translate) {
                gr = draw_move_gizmo(
                    ImGui::GetForegroundDrawList(), cam, pivot,
                    ctrl.vfov, (float)rt_w / (float)rt_h,
                    vp.origin, vp.size, gizmo_delta);
            } else if (ctrl.gizmo_mode == GizmoMode::Rotate) {
                gr = draw_rotate_gizmo(
                    ImGui::GetForegroundDrawList(), cam, pivot,
                    ctrl.vfov, (float)rt_w / (float)rt_h,
                    vp.origin, vp.size, gizmo_rotate);
            } else if (ctrl.gizmo_mode == GizmoMode::Scale) {
                gr = draw_scale_gizmo(
                    ImGui::GetForegroundDrawList(), cam, pivot,
                    ctrl.vfov, (float)rt_w / (float)rt_h,
                    vp.origin, vp.size, gizmo_scale_d);
            }

            if (!gizmo_consuming)
                gizmo_consuming = (gr > 0) && ImGui::IsMouseDown(ImGuiMouseButton_Left);

            static bool s_mesh_dragging = false;
            bool dragging_now = (gr > 0) && ImGui::IsMouseDown(ImGuiMouseButton_Left);
            bool drag_released = s_mesh_dragging && !dragging_now;
            s_mesh_dragging = dragging_now;

            bool any_change = false;

            // ── TRANSLATE ──
            if (ctrl.gizmo_mode == GizmoMode::Translate &&
                (gizmo_delta.x != 0.f || gizmo_delta.y != 0.f || gizmo_delta.z != 0.f))
            {
                for (int idx : multi_sel) {
                    if (idx < 0 || idx >= (int)mesh.objects.size()) continue;
                    MeshObject& obj = mesh.objects[idx];
                    int oid = obj.obj_id;
                    auto translate_tri = [&](Triangle& tri) {
                        if (tri.obj_id != oid) return;
                        tri.v0.x += gizmo_delta.x; tri.v0.y += gizmo_delta.y; tri.v0.z += gizmo_delta.z;
                        tri.v1.x += gizmo_delta.x; tri.v1.y += gizmo_delta.y; tri.v1.z += gizmo_delta.z;
                        tri.v2.x += gizmo_delta.x; tri.v2.y += gizmo_delta.y; tri.v2.z += gizmo_delta.z;
                    };
                    for (Triangle& tri : mesh.all_prims) translate_tri(tri);
                    for (Triangle& tri : mesh.prims)     translate_tri(tri);
                    obj.centroid.x += gizmo_delta.x;
                    obj.centroid.y += gizmo_delta.y;
                    obj.centroid.z += gizmo_delta.z;
                }
                any_change = true;
            }

            // ── ROTATE ──
            if (ctrl.gizmo_mode == GizmoMode::Rotate &&
                (gizmo_rotate.x != 0.f || gizmo_rotate.y != 0.f || gizmo_rotate.z != 0.f))
            {
                // Build rotation matrix around pivot
                float cx = cosf(gizmo_rotate.x), sx = sinf(gizmo_rotate.x);
                float cy = cosf(gizmo_rotate.y), sy = sinf(gizmo_rotate.y);
                float cz = cosf(gizmo_rotate.z), sz = sinf(gizmo_rotate.z);

                // Combined Rz * Ry * Rx
                auto rotate_pt = [&](float3 p) -> float3 {
                    float3 r = p;
                    // Rx
                    float y1 = cx*r.y - sx*r.z, z1 = sx*r.y + cx*r.z; r.y = y1; r.z = z1;
                    // Ry
                    float x2 = cy*r.x + sy*r.z, z2 = -sy*r.x + cy*r.z; r.x = x2; r.z = z2;
                    // Rz
                    float x3 = cz*r.x - sz*r.y, y3 = sz*r.x + cz*r.y; r.x = x3; r.y = y3;
                    return r;
                };

                for (int idx : multi_sel) {
                    if (idx < 0 || idx >= (int)mesh.objects.size()) continue;
                    MeshObject& obj = mesh.objects[idx];
                    int oid = obj.obj_id;
                    auto rotate_tri = [&](Triangle& tri) {
                        if (tri.obj_id != oid) return;
                        // Rotate around pivot
                        auto rot = [&](float3 v) -> float3 {
                            float3 local = { v.x - pivot.x, v.y - pivot.y, v.z - pivot.z };
                            float3 r = rotate_pt(local);
                            return { r.x + pivot.x, r.y + pivot.y, r.z + pivot.z };
                        };
                        tri.v0 = rot(tri.v0); tri.v1 = rot(tri.v1); tri.v2 = rot(tri.v2);
                        tri.n0 = rotate_pt(tri.n0); tri.n1 = rotate_pt(tri.n1); tri.n2 = rotate_pt(tri.n2);
                    };
                    for (Triangle& tri : mesh.all_prims) rotate_tri(tri);
                    for (Triangle& tri : mesh.prims)     rotate_tri(tri);
                    // Centroid rotates too
                    float3 lc = { obj.centroid.x - pivot.x, obj.centroid.y - pivot.y, obj.centroid.z - pivot.z };
                    float3 rc = rotate_pt(lc);
                    obj.centroid = { rc.x + pivot.x, rc.y + pivot.y, rc.z + pivot.z };
                }
                any_change = true;
            }

            // ── SCALE ──
            if (ctrl.gizmo_mode == GizmoMode::Scale &&
                (gizmo_scale_d.x != 0.f || gizmo_scale_d.y != 0.f || gizmo_scale_d.z != 0.f))
            {
                float sx = 1.f + gizmo_scale_d.x;
                float sy = 1.f + gizmo_scale_d.y;
                float sz = 1.f + gizmo_scale_d.z;

                for (int idx : multi_sel) {
                    if (idx < 0 || idx >= (int)mesh.objects.size()) continue;
                    MeshObject& obj = mesh.objects[idx];
                    int oid = obj.obj_id;
                    auto scale_tri = [&](Triangle& tri) {
                        if (tri.obj_id != oid) return;
                        auto sc = [&](float3 v) -> float3 {
                            return { pivot.x + (v.x - pivot.x) * sx,
                                     pivot.y + (v.y - pivot.y) * sy,
                                     pivot.z + (v.z - pivot.z) * sz };
                        };
                        tri.v0 = sc(tri.v0); tri.v1 = sc(tri.v1); tri.v2 = sc(tri.v2);
                        // Normals: inverse-transpose scale
                        auto sn = [&](float3 n) -> float3 {
                            float3 r = { n.x / sx, n.y / sy, n.z / sz };
                            float l = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
                            if (l > 1e-7f) { r.x/=l; r.y/=l; r.z/=l; }
                            return r;
                        };
                        tri.n0 = sn(tri.n0); tri.n1 = sn(tri.n1); tri.n2 = sn(tri.n2);
                    };
                    for (Triangle& tri : mesh.all_prims) scale_tri(tri);
                    for (Triangle& tri : mesh.prims)     scale_tri(tri);
                    float3 lc = { obj.centroid.x - pivot.x, obj.centroid.y - pivot.y, obj.centroid.z - pivot.z };
                    obj.centroid = { pivot.x + lc.x*sx, pivot.y + lc.y*sy, pivot.z + lc.z*sz };
                }
                any_change = true;
            }

            if (any_change) {
                if (ctrl.viewport_pass == (int)ViewportPassMode::Rasterized) {
                    // Rasterized: no BVH needed, just re-upload triangles (fast)
                    reupload_visible_prims(mesh);
                } else {
                    // Solid/Beauty: need valid BVH for ray casting
                    rebuild_visible_bvh(mesh);
                }
                mesh_moved = true; cam_changed = true;
                ++mesh.scene_version;
            }

            // Full BVH rebuild on mouse release (ensures clean state when switching modes)
            if (drag_released) {
                rebuild_visible_bvh(mesh);
                cam_changed = true;
                ++mesh.scene_version;
            }
        }

        // ── Selection outline + orbit pivot indicator ─────────────────
        {
            auto* dl = ImGui::GetForegroundDrawList();
            // Clip all overlay drawing to the viewport so nothing bleeds into panels
            ImVec2 vp_clip_max = ImVec2(vp.origin.x + vp.size.x, vp.origin.y + vp.size.y);
            dl->PushClipRect(vp.origin, vp_clip_max, true);
            float aspect = (float)rt_w / (float)rt_h;

            // Draw silhouette outline for selected mesh objects.
            // Strategy:
            //   • 1 object selected   → per-object silhouette edges
            //   • N>1 objects selected → merged silhouette (inner edges excluded)
            //   • Camera moving       → draw stale cached edges (no recompute)
            //   • Camera stable ≥ N frames → recompute
            if (ctrl.overlay_selection && !multi_sel.empty() && !mesh.all_prims.empty()) {
                struct SilEntry {
                    std::vector<std::pair<float3,float3>> edges;
                    float3 mn, mx;
                };
                // Cache keyed by a hash of the selection set (-1 = merged multi-selection)
                static std::unordered_map<int, SilEntry> s_sil_cache;
                static std::unordered_set<int>           s_last_sel;
                static int                               s_stable = 0;

                constexpr int STABLE_FRAMES = 5;

                float3 cp = make_float3(ctrl.cam_from[0], ctrl.cam_from[1], ctrl.cam_from[2]);

                s_stable = cam_changed ? 0 : s_stable + 1;

                if (mesh_moved || multi_sel != s_last_sel) {
                    s_sil_cache.clear();
                    s_last_sel = multi_sel;
                    s_stable   = 0;
                }

                const ImU32 col_out = IM_COL32(255, 160, 30, 230);
                const ImU32 col_shd = IM_COL32(  0,   0,  0, 120);

                bool single = (multi_sel.size() == 1);

                if (single) {
                    // ── Single object: per-object silhouette ─────────────
                    int idx = *multi_sel.begin();
                    if (idx < (int)mesh.objects.size() && !mesh.objects[idx].hidden) {
                        int oid = mesh.objects[idx].obj_id;
                        if (!s_sil_cache.count(oid) && s_stable >= STABLE_FRAMES) {
                            SilEntry e;
                            e.edges = compute_silhouette(mesh.all_prims, oid, cp, e.mn, e.mx);
                            s_sil_cache[oid] = std::move(e);
                        }
                        auto it = s_sil_cache.find(oid);
                        if (it != s_sil_cache.end()) {
                            for (auto& [wa, wb] : it->second.edges) {
                                ImVec2 sa, sb;
                                float da = world_to_screen(cam, wa, vp.origin, vp.size, ctrl.vfov, aspect, sa);
                                float db = world_to_screen(cam, wb, vp.origin, vp.size, ctrl.vfov, aspect, sb);
                                if (da <= 0.f || db <= 0.f) continue;
                                dl->AddLine(sa, sb, col_shd, 3.f);
                                dl->AddLine(sa, sb, col_out, 1.6f);
                            }
                        }
                    }
                } else {
                    // ── Multiple objects: merged silhouette (no inner edges) ─
                    constexpr int MERGED_KEY = -1;
                    if (!s_sil_cache.count(MERGED_KEY) && s_stable >= STABLE_FRAMES) {
                        std::vector<int> obj_ids;
                        for (int idx : multi_sel) {
                            if (idx < (int)mesh.objects.size() && !mesh.objects[idx].hidden)
                                obj_ids.push_back(mesh.objects[idx].obj_id);
                        }
                        if (!obj_ids.empty()) {
                            SilEntry e;
                            e.edges = compute_silhouette_merged(mesh.all_prims, obj_ids, cp, e.mn, e.mx);
                            s_sil_cache[MERGED_KEY] = std::move(e);
                        }
                    }
                    auto it = s_sil_cache.find(MERGED_KEY);
                    if (it != s_sil_cache.end()) {
                        for (auto& [wa, wb] : it->second.edges) {
                            ImVec2 sa, sb;
                            float da = world_to_screen(cam, wa, vp.origin, vp.size, ctrl.vfov, aspect, sa);
                            float db = world_to_screen(cam, wb, vp.origin, vp.size, ctrl.vfov, aspect, sb);
                            if (da <= 0.f || db <= 0.f) continue;
                            dl->AddLine(sa, sb, col_shd, 3.f);
                            dl->AddLine(sa, sb, col_out, 1.6f);
                        }
                    }
                }
            }

            // Draw ring around selected sphere
            if (ctrl.overlay_selection && ctrl.selected_sphere >= 0 && ctrl.selected_sphere < (int)prims_sorted.size()) {
                const Sphere& sel = prims_sorted[ctrl.selected_sphere];
                ImVec2 cs;
                float depth = world_to_screen(cam, sel.center, vp.origin, vp.size,
                                              ctrl.vfov, aspect, cs);
                if (depth > 0.f) {
                    // Edge point offset by radius in camera-right direction → screen radius
                    float3 edge_pt = {
                        sel.center.x + cam.u.x * sel.radius,
                        sel.center.y + cam.u.y * sel.radius,
                        sel.center.z + cam.u.z * sel.radius };
                    ImVec2 es;
                    world_to_screen(cam, edge_pt, vp.origin, vp.size, ctrl.vfov, aspect, es);
                    float scr_r = sqrtf((es.x-cs.x)*(es.x-cs.x) + (es.y-cs.y)*(es.y-cs.y));
                    if (scr_r < 5.f) scr_r = 5.f;
                    // Black outline for contrast, then bright ring
                    dl->AddCircle(cs, scr_r + 2.f, IM_COL32(0, 0, 0, 180), 48, 3.f);
                    dl->AddCircle(cs, scr_r,        IM_COL32(255, 200, 50, 255), 48, 2.f);
                }
            }

            // Orbit pivot crosshair (orbit mode only)
            if (ctrl.overlay_orbit && ctrl.interact_mode == InteractMode::Orbit) {
                float3 piv = { ctrl.orbit_pivot[0], ctrl.orbit_pivot[1], ctrl.orbit_pivot[2] };
                ImVec2 ps;
                if (world_to_screen(cam, piv, vp.origin, vp.size, ctrl.vfov, aspect, ps) > 0.f) {
                    const float R = 7.f;
                    dl->AddCircle(ps, R, IM_COL32(100, 200, 255, 200), 16, 1.5f);
                    dl->AddLine({ps.x - R, ps.y}, {ps.x + R, ps.y}, IM_COL32(100, 200, 255, 180), 1.f);
                    dl->AddLine({ps.x, ps.y - R}, {ps.x, ps.y + R}, IM_COL32(100, 200, 255, 180), 1.f);
                }
            }
            // ── Viewport lights HUD (Solid/Rasterized only) ─────────────
            if (ctrl.overlay_lights &&
                (ctrl.viewport_pass == (int)ViewportPassMode::Solid ||
                 ctrl.viewport_pass == (int)ViewportPassMode::Rasterized))
            {
                // Draw a 2D hemisphere diagram in the bottom-right corner
                // showing where each light is relative to the camera.
                // Camera looks into the diagram center; lights are dots on the dome.
                const float HUD_R  = 60.f;  // hemisphere radius in pixels
                const float PAD    = 20.f;
                const float PI_F   = 3.14159265f;
                ImVec2 center = { vp.origin.x + vp.size.x - HUD_R - PAD,
                                  vp.origin.y + vp.size.y - HUD_R - PAD };

                // Background circle (hemisphere dome outline)
                dl->AddCircleFilled(center, HUD_R + 2.f, IM_COL32(0, 0, 0, 100), 32);
                dl->AddCircle(center, HUD_R, IM_COL32(255, 255, 255, 80), 32, 1.f);
                // Crosshair at center (camera look direction)
                dl->AddLine({ center.x - 6.f, center.y }, { center.x + 6.f, center.y },
                            IM_COL32(255, 255, 255, 60), 1.f);
                dl->AddLine({ center.x, center.y - 6.f }, { center.x, center.y + 6.f },
                            IM_COL32(255, 255, 255, 60), 1.f);

                // Light directions from control panel offsets (same formula as shading)
                auto norm3 = [](float3 v) -> float3 {
                    float l = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
                    if (l < 1e-7f) return make_float3(0.f, 0.f, 1.f);
                    return make_float3(v.x/l, v.y/l, v.z/l);
                };
                float3 key_dir  = norm3(cam.w + cam.u * ctrl.light_key_u  + cam.v * ctrl.light_key_v);
                float3 fill_dir = norm3(cam.w + cam.u * ctrl.light_fill_u + cam.v * ctrl.light_fill_v);
                float3 rim_dir  = norm3(cam.w * -1.f + cam.u * ctrl.light_rim_u + cam.v * ctrl.light_rim_v);

                // Project direction onto HUD disc: returns screen position
                auto dir_to_disc = [&](float3 dir) -> ImVec2 {
                    float hx = dot(dir, cam.u);
                    float hy = -dot(dir, cam.v);  // screen Y is flipped
                    float hz = -dot(dir, cam.w);   // positive = in front of camera
                    float angle = atan2f(hy, hx);
                    float r = acosf(fmaxf(-1.f, fminf(1.f, hz))) / PI_F;
                    return { center.x + cosf(angle) * r * HUD_R,
                             center.y + sinf(angle) * r * HUD_R };
                };

                struct LightHUD { float3 dir; ImU32 col; const char* label; float size; int idx; };
                LightHUD lights[3] = {
                    { key_dir,  IM_COL32(255, 220, 100, 255), "Key",  9.f, 0 },
                    { fill_dir, IM_COL32(100, 150, 255, 240), "Fill", 7.f, 1 },
                    { rim_dir,  IM_COL32(220, 220, 220, 220), "Rim",  6.f, 2 },
                };

                ImVec2 mp = ImGui::GetMousePos();

                // Handle drag interaction — only pick up clicks inside the HUD disc
                float hud_dx = mp.x - center.x, hud_dy = mp.y - center.y;
                bool mouse_in_hud = (hud_dx*hud_dx + hud_dy*hud_dy) <= (HUD_R + 12.f) * (HUD_R + 12.f);

                if (mouse_in_hud && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    // Check if click is on any light dot
                    for (auto& lv : lights) {
                        ImVec2 dp = dir_to_disc(lv.dir);
                        float ddx = mp.x - dp.x, ddy = mp.y - dp.y;
                        if (ddx*ddx + ddy*ddy <= (lv.size + 4.f) * (lv.size + 4.f)) {
                            ctrl.dragging_light = lv.idx;
                            break;
                        }
                    }
                }

                if (ctrl.dragging_light >= 0 && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.f)) {
                    // Convert mouse position on disc back to light u,v offsets
                    float dx = (mp.x - center.x) / HUD_R;
                    float dy = (mp.y - center.y) / HUD_R;
                    float disc_r = sqrtf(dx*dx + dy*dy);
                    if (disc_r > 0.95f) { // clamp to disc edge
                        dx *= 0.95f / disc_r;
                        dy *= 0.95f / disc_r;
                        disc_r = 0.95f;
                    }

                    // Clamp disc_r per light type to stay in valid hemisphere
                    // Key/Fill live in back hemisphere (disc_r > 0.5), Rim in front (disc_r < 0.5)
                    bool is_rim = (ctrl.dragging_light == 2);
                    if (!is_rim) disc_r = fmaxf(disc_r, 0.55f);  // keep behind camera
                    else         disc_r = fminf(disc_r, 0.45f);   // keep in front of camera

                    float theta = disc_r * PI_F;
                    float angle = atan2f(dy, dx);
                    float sin_t = sinf(theta), cos_t = cosf(theta);

                    // Recover u,v offsets from disc position using correct inverse:
                    // Key/Fill (w=+1): u = -sin/cos * cos(angle), v = sin/cos * sin(angle)
                    // Rim     (w=-1): u =  sin/cos * cos(angle), v = -sin/cos * sin(angle)
                    float tan_t = (fabsf(cos_t) > 0.01f) ? sin_t / cos_t : (cos_t < 0.f ? -100.f : 100.f);
                    // Clamp to keep offsets reasonable
                    tan_t = fmaxf(-5.f, fminf(5.f, tan_t));

                    if (ctrl.dragging_light == 0) { // Key
                        ctrl.light_key_u  = -tan_t * cosf(angle);
                        ctrl.light_key_v  =  tan_t * sinf(angle);
                    } else if (ctrl.dragging_light == 1) { // Fill
                        ctrl.light_fill_u = -tan_t * cosf(angle);
                        ctrl.light_fill_v =  tan_t * sinf(angle);
                    } else { // Rim
                        ctrl.light_rim_u  =  tan_t * cosf(angle);
                        ctrl.light_rim_v  = -tan_t * sinf(angle);
                    }

                    // Recompute dragged light direction for immediate visual feedback
                    if (ctrl.dragging_light == 0)
                        lights[0].dir = norm3(cam.w + cam.u * ctrl.light_key_u + cam.v * ctrl.light_key_v);
                    else if (ctrl.dragging_light == 1)
                        lights[1].dir = norm3(cam.w + cam.u * ctrl.light_fill_u + cam.v * ctrl.light_fill_v);
                    else
                        lights[2].dir = norm3(cam.w * -1.f + cam.u * ctrl.light_rim_u + cam.v * ctrl.light_rim_v);
                }

                // Draw light dots
                for (auto& lv : lights) {
                    ImVec2 dp = dir_to_disc(lv.dir);

                    // Highlight when hovered or being dragged
                    float draw_size = lv.size;
                    ImU32 draw_col = lv.col;
                    float mdx = mp.x - dp.x, mdy = mp.y - dp.y;
                    bool hovered = (mdx*mdx + mdy*mdy <= (lv.size + 4.f) * (lv.size + 4.f));
                    if (hovered || ctrl.dragging_light == lv.idx) {
                        draw_size += 2.f;
                        draw_col = (draw_col & 0x00FFFFFFu) | 0xFF000000u;
                    }

                    // Line from center to light
                    dl->AddLine(center, dp, (lv.col & 0x00FFFFFFu) | 0x50000000u, 1.f);

                    // Light dot
                    dl->AddCircleFilled(dp, draw_size, draw_col, 12);
                    dl->AddCircle(dp, draw_size, IM_COL32(255, 255, 255, 120), 12, 1.f);

                    // Label
                    dl->AddText({ dp.x + draw_size + 3.f, dp.y - 6.f }, lv.col, lv.label);
                }

                // "CAM" label at center
                dl->AddText({ center.x - 10.f, center.y + 8.f },
                            IM_COL32(255, 255, 255, 100), "CAM");
            }
            // ── NIM VLM result overlay (bottom-left of viewport) ─────────
            if (ctrl.overlay_nim && ctrl.nim_result.success && ctrl.nim_result.object_type[0] != '\0') {
                const float pad     = 10.f;
                const float fs      = ImGui::GetFontSize();
                ImFont*     font    = ImGui::GetFont();

                // Fixed inner width — clamp between 260 and 520, or 38% of viewport
                float inner_w = std::max(260.f, std::min(vp.size.x * 0.38f, 520.f));
                float box_w   = inner_w + pad * 2.f;

                // Build text lines
                char line_cat[160], line_obj[160], line_desc[512], line_tags[512];
                snprintf(line_cat,  sizeof(line_cat),  "Category  %s", ctrl.nim_result.category);
                snprintf(line_obj,  sizeof(line_obj),  "Type      %s", ctrl.nim_result.object_type);
                snprintf(line_desc, sizeof(line_desc), "%s", ctrl.nim_result.description);
                snprintf(line_tags, sizeof(line_tags), "Tags  %s",     ctrl.nim_result.tags);

                // Measure wrapped heights
                auto wh = [&](const char* s) {
                    return ImGui::CalcTextSize(s, nullptr, false, inner_w).y;
                };
                float box_h = pad + wh(line_obj) + 2.f
                            + 1.f + 4.f  // separator
                            + wh(line_cat) + 2.f
                            + wh(line_desc) + 2.f
                            + wh(line_tags) + 2.f
                            + pad;

                float bx = vp.origin.x + pad;
                float by = vp.origin.y + vp.size.y - pad - box_h;

                dl->AddRectFilled(ImVec2(bx, by), ImVec2(bx+box_w, by+box_h),
                    IM_COL32(10, 10, 10, 180), 4.f);
                dl->AddRect(ImVec2(bx, by), ImVec2(bx+box_w, by+box_h),
                    IM_COL32(255, 160, 30, 120), 4.f, 0, 1.f);

                float cy = by + pad;
                auto draw_wrapped = [&](const char* s, ImU32 col, float gap = 2.f) {
                    dl->AddText(font, fs, ImVec2(bx+pad, cy), col, s, nullptr, inner_w);
                    cy += ImGui::CalcTextSize(s, nullptr, false, inner_w).y + gap;
                };

                draw_wrapped(line_obj, IM_COL32(255, 200, 80, 255));
                dl->AddLine(ImVec2(bx+pad, cy+1.f), ImVec2(bx+box_w-pad, cy+1.f),
                    IM_COL32(255, 160, 30, 60), 1.f);
                cy += 5.f;
                draw_wrapped(line_cat,  IM_COL32(160, 160, 160, 220));
                draw_wrapped(line_desc, IM_COL32(210, 210, 210, 220));
                draw_wrapped(line_tags, IM_COL32(120, 180, 255, 200));
            }

            dl->PopClipRect();
        }

        // ── Shift+LMB drag: rotate HDRI environment ───────────────────
        if (vp.hdri_dragging && ctrl.hdri_loaded) {
            // Horizontal drag → yaw; 0.3 deg/px feels responsive
            ctrl.hdri_yaw_deg += vp.hdri_drag_delta.x * 0.3f;
            // Keep in [-180, 180] for display
            while (ctrl.hdri_yaw_deg >  180.f) ctrl.hdri_yaw_deg -= 360.f;
            while (ctrl.hdri_yaw_deg < -180.f) ctrl.hdri_yaw_deg += 360.f;
            cam_changed = true;
        }
        vp.hdri_dragging   = false;
        vp.hdri_drag_delta = {};

        // Suppress viewport clicks/drags while gizmo owns the mouse, or Alt-orbit
        if (gizmo_consuming || alt_orbit) {
            vp.lmb_clicked  = false;
            vp.lmb_dragging = false;
        }

        // ── SELECT: LMB click (all modes) ────────────────────────────
        if (vp.lmb_clicked) {
            float u = vp.mouse_uv.x;
            float v = 1.f - vp.mouse_uv.y;
            float3 ray_dir = normalize(
                cam.lower_left + u*cam.horizontal + v*cam.vertical - cam.origin);

            // Prefer mesh objects when a mesh is loaded, else pick spheres
            // Selection never touches camera or orbit pivot — no cam_changed here
            if (!mesh.prims.empty()) {
                // Skip hidden objects in viewport picking
                std::unordered_set<int> hidden_ids;
                for (auto& obj : mesh.objects)
                    if (obj.hidden) hidden_ids.insert(obj.obj_id);
                std::vector<Triangle> pickable;
                for (auto& t : mesh.all_prims)
                    if (!hidden_ids.count(t.obj_id)) pickable.push_back(t);

                ctrl.selected_mesh_obj = pick_mesh_object(pickable, mesh.objects, cam.origin, ray_dir);
                ctrl.selected_sphere   = -1;
                multi_sel.clear();
                if (ctrl.selected_mesh_obj >= 0)
                    multi_sel.insert(ctrl.selected_mesh_obj);
            } else {
                ctrl.selected_sphere   = pick_sphere(prims_sorted, cam.origin, ray_dir);
                ctrl.selected_mesh_obj = -1;
                multi_sel.clear();
            }
        }

        // ── Hide / Unhide (Ctrl+H = hide, Shift+H = unhide) ──────────
        if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_H)) {
            bool hide = !io.KeyShift;  // Shift+H = unhide, H or Ctrl+H = hide
            bool vis_changed = false;
            for (int idx : multi_sel) {
                if (idx < (int)mesh.objects.size() &&
                    mesh.objects[idx].hidden != hide) {
                    mesh.objects[idx].hidden = hide;
                    vis_changed = true;
                }
            }
            if (vis_changed) {
                rebuild_visible_bvh(mesh);
                cam_changed = true;
            }
        }

        // ── RMB click: set orbit pivot to exact surface hit point ────
        if (vp.rmb_clicked) {
            float u = vp.mouse_uv.x;
            float v = 1.f - vp.mouse_uv.y;
            float3 ray_dir = normalize(
                cam.lower_left + u*cam.horizontal + v*cam.vertical - cam.origin);
            float3 hit;
            if (pick_surface_point(mesh.prims, prims_sorted, cam.origin, ray_dir, hit)) {
                ctrl.orbit_pivot[0] = hit.x;
                ctrl.orbit_pivot[1] = hit.y;
                ctrl.orbit_pivot[2] = hit.z;
                float dx = hit.x - cam.origin.x;
                float dy = hit.y - cam.origin.y;
                float dz = hit.z - cam.origin.z;
                ctrl.focus_dist = sqrtf(dx*dx + dy*dy + dz*dz);
                cam_changed = true;
            }
        }

        // ── Reset accumulation ────────────────────────────────────────
        if (cam_changed) {
            pathtracer_reset_accum(d_accum, rt_w, rt_h);
            frame_count = 0;
        }

        // Rebuild cam after this frame's orbit adjustments
        cam = Camera::make(
            make_float3(ctrl.cam_from[0], ctrl.cam_from[1], ctrl.cam_from[2]),
            make_float3(ctrl.cam_at[0],   ctrl.cam_at[1],   ctrl.cam_at[2]),
            make_float3(0.f, 1.f, 0.f),
            ctrl.vfov, (float)rt_w / (float)rt_h,
            ctrl.aperture, ctrl.focus_dist);

        // ── Path tracer ───────────────────────────────────────────────
        PathTracerParams pt{};
        pt.surface      = interop.surface;
        pt.accum_buffer = d_accum;
        pt.width        = rt_w;
        pt.height       = rt_h;
        pt.cam          = cam;
        pt.bvh          = d_bvh;
        pt.prims        = d_prims;
        pt.num_prims    = (int)prims_sorted.size();
        // Triangle mesh
        pt.tri_bvh          = mesh.d_bvh;
        pt.triangles        = mesh.d_prims;
        pt.num_triangles    = (int)mesh.prims.size();
        pt.gpu_materials    = mesh.d_mats;
        pt.num_gpu_materials = mesh.num_mats;
        pt.textures         = mesh.d_tex_hdls;
        pt.num_textures     = mesh.num_texs;
        pt.frame_count  = frame_count;
        pt.spp          = ctrl.spp;
        pt.max_depth    = ctrl.max_depth;
        pt.color_mode       = ctrl.color_mode;
        pt.obj_colors       = mesh.d_obj_colors;
        pt.num_obj_colors   = mesh.num_obj_colors;
        pt.hdri_tex         = hdri.loaded ? hdri.tex : 0;
        pt.hdri_intensity   = ctrl.hdri_intensity;
        pt.hdri_yaw         = ctrl.hdri_yaw_deg * (3.14159265f / 180.f);
        pt.hdri_bg_blur     = ctrl.hdri_bg_blur;
        pt.hdri_bg_opacity  = ctrl.hdri_bg_opacity;
        pt.bg_color         = make_float3(ctrl.bg_color[0], ctrl.bg_color[1], ctrl.bg_color[2]);
        pt.firefly_clamp        = ctrl.firefly_clamp;
        pt.sm_tracker_bitmask   = sm_tracker.d_bitmask;

        // ── DLSS / resolution scaling ─────────────────────────────────────────
        int render_w = rt_w, render_h = rt_h;
        if (ctrl.dlss_enabled) {
#ifdef DLSS_ENABLED
            // Init DLSS on first enable or quality/scale change
            static int   s_last_dlss_quality = -1;
            static float s_last_dlss_scale   = -1.f;
            bool scale_changed = (s_last_dlss_scale != ctrl.dlss_scale);
            if (!dlss_active || s_last_dlss_quality != ctrl.dlss_quality || scale_changed) {
                dlss_free(dlss);
                dlss_active = dlss_init(dlss,
                    vk.instance, vk.physicalDevice, vk.device,
                    vk.commandPool, vk.graphicsQueue,
                    rt_w, rt_h, ctrl.dlss_quality, ctrl.dlss_scale,
                    render_w, render_h);
                s_last_dlss_quality = ctrl.dlss_quality;
                s_last_dlss_scale   = ctrl.dlss_scale;
                if (!dlss_active) {
                    render_w = rt_w;
                    render_h = rt_h;
                    fprintf(stderr, "[dlss] init failed — falling back to full-resolution render\n");
                }
            } else if (dlss_active) {
                render_w = dlss.render_w;
                render_h = dlss.render_h;
            }
#else
            // Software bicubic scaling — pathtracer renders small, sw_upscale fills surface.
            // skip_surface_write=1: suppress the partial top-left write to the full surface.
            // (Neural DLSS does NOT set this — it reads the partial surface write directly.)
            float sc = std::max(0.01f, std::min(1.f, ctrl.dlss_scale));
            render_w = std::max(1, (int)(rt_w * sc));
            render_h = std::max(1, (int)(rt_h * sc));
            pt.skip_surface_write = (render_w < rt_w || render_h < rt_h) ? 1 : 0;
#endif
            pt.width  = render_w;
            pt.height = render_h;
        } else if (dlss_active) {
            dlss_free(dlss);
            dlss_active = false;
        }

        // ── OptiX RT: lazy init + pick up result once ready ──────────
#ifdef OPTIX_ENABLED
        // Start background init the first time the user enables OptiX RT.
        if (ctrl.optix_rt_enabled && !optix_rt_init_started.load(std::memory_order_relaxed)) {
            optix_rt_init_started.store(true, std::memory_order_relaxed);
            if (optix_rt_thread.joinable()) optix_rt_thread.join(); // safety
            optix_rt_thread = std::thread([&]() {
                cudaSetDevice(0);
                printf("[main] Compiling OptiX RT shaders (nvrtc — may take 30-60s first run)...\n");
                fflush(stdout);
                bool ok = optix_renderer_init(optix_rt, rt_w, rt_h);
                if (ok)  printf("[main] OptiX RT renderer available (hardware RT cores)\n");
                else     printf("[main] OptiX RT renderer unavailable: %s\n", optix_rt.last_error);
                fflush(stdout);
                optix_rt_init_done.store(true, std::memory_order_seq_cst);
            });
        }

        // Once the background thread finishes, expose result to UI.
        if (!ctrl.optix_rt_available && optix_rt_init_done.load(std::memory_order_seq_cst)) {
            ctrl.optix_rt_available = optix_rt.available;
            if (!optix_rt.available)
                strncpy_s(ctrl.optix_rt_last_error, sizeof(ctrl.optix_rt_last_error),
                          optix_rt.last_error, _TRUNCATE);
        }

        // ── OptiX RT: upload scene when mesh changes ──────────────────
        if (ctrl.optix_rt_enabled && optix_rt.available && !mesh.prims.empty()) {
            if (optix_rt.scene_version != mesh.scene_version) {
                if (!optix_renderer_upload_scene(optix_rt, mesh.prims, mesh.scene_version))
                    printf("[main] OptiX RT scene upload failed: %s\n", optix_rt.last_error);
            }
        }
        // Sync runtime status to UI every frame
        ctrl.optix_rt_scene_ready = optix_rt.scene_ready;
        if (optix_rt.last_error[0] && !ctrl.optix_rt_last_error[0])
            strncpy_s(ctrl.optix_rt_last_error, sizeof(ctrl.optix_rt_last_error),
                      optix_rt.last_error, _TRUNCATE);
#endif

        // ── Dispatch: ReSTIR, OptiX RT, or standard path tracer ──────
        bool use_restir = ctrl.restir_enabled && mesh.num_emissive > 0 && mesh.d_bvh;
        bool use_optix_rt = ctrl.optix_rt_enabled && optix_rt.available && optix_rt.scene_ready;

        // Solid/Rasterized modes write directly to the surface — skip pathtracer entirely.
        bool skip_pathtracer = (ctrl.viewport_pass == (int)ViewportPassMode::Solid ||
                                ctrl.viewport_pass == (int)ViewportPassMode::Rasterized);

        // CUPTI: start profiling pass when checkbox is enabled
        if (arch_state.profiling_enabled && arch_state.cupti_available)
            safe_cupti_begin_frame(cupti_profiler);
        if (skip_pathtracer) {
            // No-op: Solid/Rasterized kernels run in the viewport pass switch below.
        } else if (use_restir) {
            // Lazy-alloc reservoirs if not yet done (first frame or after resize w/o viewport event)
            if (!mesh.d_reservoirs) {
                mesh.d_reservoirs  = restir_alloc_reservoirs(render_w, render_h);
                mesh.d_reservoirs2 = restir_alloc_reservoirs(render_w, render_h);
            }
            ReSTIRParams rp{};
            rp.surface          = interop.surface;
            rp.accum_buffer     = d_accum;
            rp.width            = render_w;
            rp.height           = render_h;
            rp.cam              = cam;
            rp.tri_bvh          = mesh.d_bvh;
            rp.triangles        = mesh.d_prims;
            rp.num_triangles    = (int)mesh.prims.size();
            rp.gpu_materials    = mesh.d_mats;
            rp.num_gpu_materials= mesh.num_mats;
            rp.textures         = mesh.d_tex_hdls;
            rp.num_textures     = mesh.num_texs;
            rp.emissive_tris    = mesh.d_emissive;
            rp.num_emissive     = mesh.num_emissive;
            rp.reservoirs       = mesh.d_reservoirs;
            rp.reservoirs_tmp   = mesh.d_reservoirs2;
            rp.frame_count      = frame_count;
            rp.spp              = ctrl.spp;
            rp.max_depth        = ctrl.max_depth;
            rp.num_candidates   = ctrl.restir_candidates;
            rp.spatial_radius   = ctrl.restir_radius;
            rp.hdri_tex         = hdri.loaded ? hdri.tex : 0;
            rp.hdri_intensity   = ctrl.hdri_intensity;
            rp.hdri_yaw         = ctrl.hdri_yaw_deg * (3.14159265f / 180.f);
            restir_launch(rp);
        } else if (use_optix_rt) {
#ifdef OPTIX_ENABLED
            optix_renderer_render(optix_rt, pt);
#endif
        } else {
            pathtracer_launch(pt);
        }

        // ── DLSS aux buffers: depth + motion vectors ─────────────────────────
        {
            DlssAuxParams aux{};
            aux.width         = render_w;
            aux.height        = render_h;
            aux.cam           = cam;
            aux.prev_cam      = prev_cam;
            aux.has_prev_camera = has_prev_cam ? 1 : 0;
            aux.camera_near   = 0.01f;
            aux.camera_far    = 1000.f;
            aux.bvh           = d_bvh;
            aux.prims         = d_prims;
            aux.num_prims     = (int)prims_sorted.size();
            aux.tri_bvh       = mesh.d_bvh;
            aux.triangles     = mesh.d_prims;
            aux.num_triangles = (int)mesh.prims.size();
            aux.depth_buffer  = d_depth;
            aux.motion_buffer = d_motion;
            pathtracer_write_dlss_aux(aux);
        }
        prev_cam     = cam;
        has_prev_cam = true;

        cudaDeviceSynchronize();

        // ── Software bicubic upscale (sw-scale mode only, not neural DLSS) ──
        // skip_surface_write is only set in the #else (no DLSS_ENABLED) branch,
        // so this never fires when neural DLSS is active.
        if (ctrl.dlss_enabled && pt.skip_surface_write)
            pathtracer_sw_upscale(d_accum, render_w, render_h, interop.surface, rt_w, rt_h);

        // ── OptiX RT: blit accum → surface so it's visible (raygen writes d_accum only) ─
        // Skip when neural DLSS is active — DLSS reads d_accum and writes surface itself.
        if (use_optix_rt && !dlss_active)
            pathtracer_blit_surface(d_accum, interop.surface, rt_w, rt_h);

        // CUPTI: end profiling pass and update real metrics
        if (arch_state.profiling_enabled && arch_state.cupti_available) {
            safe_cupti_end_frame(cupti_profiler);
            CuptiMetrics cm = cupti_profiler.get_metrics();
            if (cm.valid) {
                arch_state.cuda_active_pct   = cm.cuda_active_pct;
                arch_state.tex_active_pct    = cm.tex_active_pct;
                arch_state.ldst_active_pct   = cm.ldst_active_pct;
                arch_state.sfu_active_pct    = cm.sfu_active_pct;
                arch_state.tensor_active_pct = cm.tensor_active_pct;
                arch_state.has_unit_counters = true;
            }
        } else {
            arch_state.has_unit_counters = false;
        }

        if (!skip_pathtracer)
            ++frame_count;

        // ── Save Render (EXR — with or without tonemapping) ─────────
        if (ctrl.save_exr_requested) {
            ctrl.save_exr_requested = false;
            if (ctrl.save_exr_path[0] != '\0' && frame_count > 0) {
                int sw = rt_w, sh = rt_h;
                std::vector<float> r(sw * sh), g(sw * sh), b(sw * sh);

                if (ctrl.save_exr_tonemapped && ctrl.post_enabled && post.pipeline) {
                    // Read the post-processed image — exact viewport content
                    // (AgX/ACES tonemap, bloom, exposure, all baked in)
                    auto h2f = [](uint16_t h) -> float {
                        uint32_t sign = (h & 0x8000u) << 16;
                        uint32_t exp = (h >> 10) & 0x1Fu;
                        uint32_t mant = h & 0x3FFu;
                        if (exp == 0) { if (mant == 0) { float f; uint32_t u = sign; memcpy(&f,&u,4); return f; }
                            exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF;
                            uint32_t u = sign | ((exp + 112) << 23) | (mant << 13); float f; memcpy(&f,&u,4); return f;
                        } else if (exp == 31) { uint32_t u = sign | 0x7F800000u | (mant << 13); float f; memcpy(&f,&u,4); return f; }
                        uint32_t u = sign | ((exp + 112) << 23) | (mant << 13); float f; memcpy(&f,&u,4); return f;
                    };
                    std::vector<uint16_t> hf = post_process_readback(
                        post, vk.physicalDevice, vk.commandPool, vk.graphicsQueue);
                    sw = post.width; sh = post.height;
                    r.resize(sw * sh); g.resize(sw * sh); b.resize(sw * sh);
                    for (int i = 0; i < sw * sh; ++i) {
                        r[i] = h2f(hf[i * 4 + 0]);
                        g[i] = h2f(hf[i * 4 + 1]);
                        b[i] = h2f(hf[i * 4 + 2]);
                    }
                    std::cout << "[save] Saving tonemapped EXR from post-process ("
                              << sw << "x" << sh << ")\n";
                } else {
                    // Raw linear HDR from accum buffer (already a running average)
                    std::vector<float> px((size_t)rt_w * rt_h * 4);
                    cudaMemcpy(px.data(), d_accum, px.size() * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    for (int i = 0; i < sw * sh; ++i) {
                        r[i] = px[i * 4 + 0];
                        g[i] = px[i * 4 + 1];
                        b[i] = px[i * 4 + 2];
                    }
                }

                EXRHeader header; InitEXRHeader(&header);
                EXRImage  image;  InitEXRImage(&image);
                image.num_channels = 3;
                float* channels[3] = { b.data(), g.data(), r.data() };
                image.images = reinterpret_cast<unsigned char**>(channels);
                image.width  = sw;
                image.height = sh;
                header.num_channels = 3;
                header.channels = static_cast<EXRChannelInfo*>(malloc(sizeof(EXRChannelInfo)*3));
                strncpy_s(header.channels[0].name, sizeof(header.channels[0].name), "B", _TRUNCATE);
                strncpy_s(header.channels[1].name, sizeof(header.channels[1].name), "G", _TRUNCATE);
                strncpy_s(header.channels[2].name, sizeof(header.channels[2].name), "R", _TRUNCATE);
                header.pixel_types           = static_cast<int*>(malloc(sizeof(int)*3));
                header.requested_pixel_types = static_cast<int*>(malloc(sizeof(int)*3));
                int exr_type = ctrl.save_exr_half ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;
                for (int i = 0; i < 3; ++i) {
                    header.pixel_types[i]           = TINYEXR_PIXELTYPE_FLOAT;
                    header.requested_pixel_types[i] = exr_type;
                }
                const char* err = nullptr;
                int ret = SaveEXRImageToFile(&image, &header, ctrl.save_exr_path, &err);
                if (ret == TINYEXR_SUCCESS)
                    std::cout << "[save] Wrote EXR: " << ctrl.save_exr_path << "\n";
                else if (err) { std::cerr << "[save] EXR error: " << err << "\n"; FreeEXRErrorMessage(err); }
                free(header.channels);
                free(header.pixel_types);
                free(header.requested_pixel_types);
            }
        }

        // ── Cosmos Transfer: readback AOVs and send to Cosmos ────────
        if (ctrl.cosmos_request && !ctrl.cosmos_busy) {
            ctrl.cosmos_request = false;
            ctrl.cosmos_busy    = true;
            ctrl.cosmos_error[0] = '\0';
            snprintf(ctrl.cosmos_cfg.status, sizeof(ctrl.cosmos_cfg.status), "Reading back AOVs...");

            int cw = render_w, ch = render_h;

            // ── Beauty readback ──────────────────────────────────────────
            std::vector<uint8_t> beauty;
            if (ctrl.cosmos_source == 1 && hydra_preview_ready) {
                // Hydra source — read whatever AOV is currently displayed
                beauty = hydra_preview_read_color(hp_preview);
                if (beauty.empty()) {
                    // Fallback to pathtracer if Hydra read failed
                    beauty.resize(cw * ch * 4);
                    pathtracer_readback_surface(interop.surface, rt_w, rt_h, beauty.data());
                } else {
                    // Hydra buffer may be a different size — update cw/ch
                    // (read_color returns w*h*4 bytes, dimensions from the render buffer)
                    // For now assume it matches the Hydra panel size
                    cw = hp_preview.tex_w;
                    ch = hp_preview.tex_h;
                }
            } else if (ctrl.post_enabled && post.pipeline) {
                // Path Tracer + post-process: read tonemapped display image
                beauty.resize(cw * ch * 4);
                std::vector<uint16_t> hf = post_process_readback(
                    post, vk.physicalDevice, vk.commandPool, vk.graphicsQueue);
                int pw = post.width, ph = post.height;
                auto h2f = [](uint16_t h) -> float {
                    uint32_t sign = (h & 0x8000u) << 16;
                    uint32_t exp  = (h >> 10) & 0x1Fu;
                    uint32_t mant = h & 0x3FFu;
                    if (exp == 0) { if (mant == 0) { float f; uint32_t u = sign; memcpy(&f,&u,4); return f; }
                        exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF;
                        uint32_t u = sign | ((exp + 112) << 23) | (mant << 13); float f; memcpy(&f,&u,4); return f;
                    } else if (exp == 31) { uint32_t u = sign | 0x7F800000u | (mant << 13); float f; memcpy(&f,&u,4); return f; }
                    uint32_t u = sign | ((exp + 112) << 23) | (mant << 13); float f; memcpy(&f,&u,4); return f;
                };
                for (int y = 0; y < std::min(ch, ph); ++y) {
                    for (int x = 0; x < std::min(cw, pw); ++x) {
                        int si = y * pw + x;
                        int di = y * cw + x;
                        float r = h2f(hf[si*4+0]);
                        float g = h2f(hf[si*4+1]);
                        float b = h2f(hf[si*4+2]);
                        beauty[di*4+0] = (uint8_t)std::min(255.f, powf(std::max(0.f,std::min(1.f,r)), 1.f/2.2f) * 255.f + 0.5f);
                        beauty[di*4+1] = (uint8_t)std::min(255.f, powf(std::max(0.f,std::min(1.f,g)), 1.f/2.2f) * 255.f + 0.5f);
                        beauty[di*4+2] = (uint8_t)std::min(255.f, powf(std::max(0.f,std::min(1.f,b)), 1.f/2.2f) * 255.f + 0.5f);
                        beauty[di*4+3] = 255;
                    }
                }
            } else {
                // Path Tracer, no post-process — read CUDA surface directly
                beauty.resize(cw * ch * 4);
                pathtracer_readback_surface(interop.surface, rt_w, rt_h, beauty.data());
            }

            // Readback depth + seg from pathtracer (always, regardless of source)
            int dw = render_w, dh = render_h;  // pathtracer dimensions
            float dmin = 0.f, dmax = 1.f;
            pathtracer_depth_range(d_depth, dw * dh, &dmin, &dmax);
            std::vector<uint8_t> depth_aov(dw * dh * 4);
            pathtracer_readback_depth(d_depth, dw, dh, dmin, dmax, depth_aov.data());

            std::vector<uint8_t> seg_aov(dw * dh * 4);
            pathtracer_readback_seg(dw, dh, cam,
                d_bvh, d_prims, (int)prims_sorted.size(),
                mesh.d_bvh, mesh.d_prims, (int)mesh.prims.size(),
                mesh.d_obj_colors, mesh.num_obj_colors, seg_aov.data());

            cudaDeviceSynchronize();
            snprintf(ctrl.cosmos_cfg.status, sizeof(ctrl.cosmos_cfg.status), "Sending to Cosmos...");

            // Save AOVs to disk for debugging
            std::filesystem::create_directories("debug");
            stbi_write_png("debug/cosmos_beauty.png", cw, ch, 4, beauty.data(), cw * 4);
            stbi_write_png("debug/cosmos_depth.png",  dw, dh, 4, depth_aov.data(), dw * 4);
            stbi_write_png("debug/cosmos_seg.png",    dw, dh, 4, seg_aov.data(), dw * 4);
            std::cout << "[cosmos] Saved AOV PNGs to debug/\n";

            if (cosmos_thread.joinable()) cosmos_thread.join();
            cosmos_thread = std::thread([
                cfg = ctrl.cosmos_cfg,
                beauty_data = std::move(beauty),
                depth_data  = std::move(depth_aov),
                seg_data    = std::move(seg_aov),
                cw, ch, &ctrl]()
            {
                CosmosResult r = cosmos_transfer(cfg,
                    beauty_data.data(), depth_data.data(), seg_data.data(), cw, ch);

                if (r.success) {
                    if (ctrl.cosmos_result_pixels) {
                        delete[] ctrl.cosmos_result_pixels;
                        ctrl.cosmos_result_pixels = nullptr;
                    }
                    ctrl.cosmos_result_pixels = r.pixels;
                    ctrl.cosmos_result_w      = r.width;
                    ctrl.cosmos_result_h      = r.height;
                    ctrl.cosmos_has_result    = true;
                    ctrl.cosmos_show_result   = true;
                    ctrl.cosmos_error[0]      = '\0';
                    stbi_write_png("debug/cosmos_result.png",
                                   r.width, r.height, 4, r.pixels, r.width * 4);
                    snprintf(ctrl.cosmos_cfg.status, sizeof(ctrl.cosmos_cfg.status),
                             "Saved to debug/cosmos_result.png");
                } else {
                    strncpy_s(ctrl.cosmos_error, sizeof(ctrl.cosmos_error),
                              r.error, _TRUNCATE);
                    snprintf(ctrl.cosmos_cfg.status, sizeof(ctrl.cosmos_cfg.status), "Failed");
                }
                ctrl.cosmos_busy = false;
            });
        }

        // ── Denoiser: always reads d_accum, writes to d_display ──────────────
        // d_accum is NEVER modified — it stays as the raw accumulated data.
        // d_display is blitted to the surface so the viewport shows the denoised result.
        // d_display_valid: true once the denoiser has written at least one frame.
        // On non-denoise frames we re-blit d_display so the viewport never shows
        // raw noisy data while denoising is enabled (prevents the back-and-forth flicker).
        static bool d_display_valid = false;
        bool did_denoise = false;
        bool denoising_active = false;
        const bool dlss_final_input = dlss_active &&
                                      ctrl.dlss_enabled &&
                                      ctrl.viewport_pass == (int)ViewportPassMode::Final;
        auto present_for_viewport = [&](const float4* src) {
            if (dlss_final_input) {
                // For DLSS final mode, keep render-res data in the top-left region.
                // DLSS reads this region as input and writes the full-res result later.
                pathtracer_blit_surface(src, interop.surface, render_w, render_h);
            } else {
                pathtracer_sw_upscale(src, render_w, render_h, interop.surface, rt_w, rt_h);
            }
        };

        // Camera moved → invalidate stale denoised frame so the fresh pathtracer
        // output shows through. The denoiser runs immediately this same frame
        // (every=1 when cam_changed) so d_display_valid goes true again right away.
        if (cam_changed) d_display_valid = false;

#ifdef OPTIX_ENABLED
        if (ctrl.optix_enabled && optix_dn.available) {
            denoising_active = true;
            bool run_dn = ctrl.denoise_on_demand;
            ctrl.denoise_on_demand = false;
            if (!run_dn && frame_count > 0) {
                // cam_changed: always denoise this frame to avoid showing stale result
                // first 16 frames: run every frame for fast settle; then per user setting
                int every = (cam_changed || frame_count <= 16) ? 1 : ctrl.denoise_every_n;
                run_dn = (frame_count % every == 0);
            }
            if (run_dn) {
                // OptiX denoiser is initialized for rt_w×rt_h and reads d_accum with
                // rt_w row stride. When DLSS scaling is active (render_w < rt_w),
                // d_accum has render_w stride — skip to avoid skewed/zoomed output.
                if (render_w == rt_w && render_h == rt_h) {
                    optix_denoiser_run(optix_dn, d_accum, d_display);
                    present_for_viewport(d_display);
                    d_display_valid = true;
                }
                did_denoise = true;
            }
        } else
#endif
        if (ctrl.denoise_available) {
            denoising_active = ctrl.denoise_enabled;
            bool run_dn = ctrl.denoise_on_demand;
            ctrl.denoise_on_demand = false;
            if (!run_dn && ctrl.denoise_enabled && frame_count > 0) {
                // Run every frame for first 16 frames to avoid visible blink on settle
                int every = (frame_count <= 16) ? 1 : ctrl.denoise_every_n;
                run_dn = (frame_count % every == 0);
            }
            if (run_dn) {
                // Pass render dimensions: d_accum is packed at render_w stride when
                // DLSS scaling is active (render_w < rt_w). OIDN works with any size.
                denoiser_run(denoiser, d_accum, d_display, render_w, render_h);
                present_for_viewport(d_display);
                d_display_valid = true;
                did_denoise = true;
            }
        }

        // On non-denoise frames: if denoising is active and we have a valid denoised
        // frame, re-blit it so the viewport always shows the denoised result.
        if (!did_denoise && denoising_active && d_display_valid)
            present_for_viewport(d_display);

        // When denoising is turned off, invalidate so we don't show stale data.
        if (!denoising_active) d_display_valid = false;
        (void)did_denoise;

        // ── Viewport pass override ────────────────────────────────────
        {
            // Compute per-frame depth range and max motion magnitude only when needed.
            // These are used for auto-normalization so both passes always span 0..1.
            // Computed lazily: only when depth or motion is actually being displayed.
            const bool need_depth = (ctrl.viewport_pass == (int)ViewportPassMode::DlssDepth) ||
                                    (ctrl.dlss_enabled && ctrl.dlss_debug &&
                                     ctrl.dlss_overlay_pass == (int)ViewportPassMode::DlssDepth);
            const bool need_motion = (ctrl.viewport_pass == (int)ViewportPassMode::DlssMotion) ||
                                     (ctrl.dlss_enabled && ctrl.dlss_debug &&
                                      ctrl.dlss_overlay_pass == (int)ViewportPassMode::DlssMotion);

            float depth_min = 0.f, depth_max = 1.f;
            if (need_depth)
                pathtracer_depth_range(d_depth, render_w * render_h, &depth_min, &depth_max);

            float motion_maxmag = 0.f;
            if (need_motion)
                motion_maxmag = pathtracer_motion_maxmag(d_motion, render_w * render_h);

            // Compute world-space 3-point light directions from control panel offsets
            auto normalize3 = [](float3 v) -> float3 {
                float l = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
                if (l < 1e-7f) return make_float3(0.f, 0.f, 1.f);
                return make_float3(v.x/l, v.y/l, v.z/l);
            };
            float3 light_key_dir  = normalize3(cam.w + cam.u * ctrl.light_key_u  + cam.v * ctrl.light_key_v);
            float3 light_fill_dir = normalize3(cam.w + cam.u * ctrl.light_fill_u + cam.v * ctrl.light_fill_v);
            float3 light_rim_dir  = normalize3(cam.w * -1.f + cam.u * ctrl.light_rim_u + cam.v * ctrl.light_rim_v);

            switch ((ViewportPassMode)ctrl.viewport_pass) {

            case ViewportPassMode::RawRender:
            case ViewportPassMode::DlssInput:
                // Show the raw pathtracer output (before DLSS) scaled to fill the screen.
                // At lower render %, image looks blurrier (fewer source pixels).
                // At 100%, no upscaling — crisp native pixels.
                pathtracer_sw_upscale(d_accum, render_w, render_h,
                                      interop.surface, rt_w, rt_h);
                break;

            case ViewportPassMode::Denoised:
                // Show denoised if available, else fall back to raw accum.
                // Use sw_upscale so render_w-stride d_accum/d_display maps correctly
                // when DLSS scaling is active (render_w < rt_w).
                if (d_display_valid)
                    pathtracer_sw_upscale(d_display, render_w, render_h, interop.surface, rt_w, rt_h);
                else
                    pathtracer_sw_upscale(d_accum, render_w, render_h, interop.surface, rt_w, rt_h);
                break;

            case ViewportPassMode::DlssDepth:
                // Auto-normalized: closest surface = white, farthest = black.
                pathtracer_visualize_depth(d_depth, render_w, render_h,
                                           depth_min, depth_max,
                                           interop.surface, rt_w, rt_h);
                break;

            case ViewportPassMode::DlssMotion:
                // When camera is nearly static, generate fake motion vectors from
                // a small virtual orbit so the visualization isn't just black dots.
                if (motion_maxmag < 0.5f) {
                    float fake_yaw = 0.02f; // small virtual rotation
                    float cx = ctrl.orbit_pivot[0], cy = ctrl.orbit_pivot[1], cz = ctrl.orbit_pivot[2];
                    float ox = ctrl.pos[0] - cx, oy = ctrl.pos[1] - cy, oz = ctrl.pos[2] - cz;
                    float c = cosf(fake_yaw), s = sinf(fake_yaw);
                    float fx = ox*c + oz*s, fz = -ox*s + oz*c;
                    Camera fake_prev = Camera::make(
                        make_float3(cx + fx, cy + oy, cz + fz),
                        make_float3(cx, cy, cz),
                        make_float3(0.f, 1.f, 0.f),
                        ctrl.vfov, (float)rt_w / (float)rt_h,
                        ctrl.aperture, ctrl.focus_dist);
                    DlssAuxParams fake_aux{};
                    fake_aux.width = render_w; fake_aux.height = render_h;
                    fake_aux.cam = pt.cam; fake_aux.prev_cam = fake_prev;
                    fake_aux.has_prev_camera = 1;
                    fake_aux.camera_near = 0.01f; fake_aux.camera_far = 1000.f;
                    fake_aux.bvh = d_bvh; fake_aux.prims = d_prims;
                    fake_aux.num_prims = (int)prims_sorted.size();
                    fake_aux.tri_bvh = mesh.d_bvh; fake_aux.triangles = mesh.d_prims;
                    fake_aux.num_triangles = (int)mesh.prims.size();
                    fake_aux.motion_buffer = d_motion;
                    pathtracer_write_dlss_aux(fake_aux);
                    cudaDeviceSynchronize();
                    float fake_max = pathtracer_motion_maxmag(d_motion, render_w * render_h);
                    pathtracer_visualize_motion(d_motion, render_w, render_h,
                                                fake_max, interop.surface, rt_w, rt_h);
                } else {
                    pathtracer_visualize_motion(d_motion, render_w, render_h,
                                                motion_maxmag, interop.surface, rt_w, rt_h);
                }
                break;

            case ViewportPassMode::Normal:
                pathtracer_visualize_material(render_w, render_h, cam,
                    d_bvh, d_prims, (int)prims_sorted.size(),
                    mesh.d_bvh, mesh.d_prims, (int)mesh.prims.size(),
                    mesh.d_mats, mesh.num_mats, mesh.d_tex_hdls,
                    mesh.d_obj_colors, mesh.num_obj_colors,
                    0, interop.surface, rt_w, rt_h);
                break;
            case ViewportPassMode::Albedo:
                pathtracer_visualize_material(render_w, render_h, cam,
                    d_bvh, d_prims, (int)prims_sorted.size(),
                    mesh.d_bvh, mesh.d_prims, (int)mesh.prims.size(),
                    mesh.d_mats, mesh.num_mats, mesh.d_tex_hdls,
                    mesh.d_obj_colors, mesh.num_obj_colors,
                    1, interop.surface, rt_w, rt_h);
                break;
            case ViewportPassMode::Metallic:
                pathtracer_visualize_material(render_w, render_h, cam,
                    d_bvh, d_prims, (int)prims_sorted.size(),
                    mesh.d_bvh, mesh.d_prims, (int)mesh.prims.size(),
                    mesh.d_mats, mesh.num_mats, mesh.d_tex_hdls,
                    mesh.d_obj_colors, mesh.num_obj_colors,
                    2, interop.surface, rt_w, rt_h);
                break;
            case ViewportPassMode::Roughness:
                pathtracer_visualize_material(render_w, render_h, cam,
                    d_bvh, d_prims, (int)prims_sorted.size(),
                    mesh.d_bvh, mesh.d_prims, (int)mesh.prims.size(),
                    mesh.d_mats, mesh.num_mats, mesh.d_tex_hdls,
                    mesh.d_obj_colors, mesh.num_obj_colors,
                    3, interop.surface, rt_w, rt_h);
                break;
            case ViewportPassMode::Emission:
                pathtracer_visualize_material(render_w, render_h, cam,
                    d_bvh, d_prims, (int)prims_sorted.size(),
                    mesh.d_bvh, mesh.d_prims, (int)mesh.prims.size(),
                    mesh.d_mats, mesh.num_mats, mesh.d_tex_hdls,
                    mesh.d_obj_colors, mesh.num_obj_colors,
                    4, interop.surface, rt_w, rt_h);
                break;
            case ViewportPassMode::Segmentation:
                pathtracer_visualize_material(render_w, render_h, cam,
                    d_bvh, d_prims, (int)prims_sorted.size(),
                    mesh.d_bvh, mesh.d_prims, (int)mesh.prims.size(),
                    mesh.d_mats, mesh.num_mats, mesh.d_tex_hdls,
                    mesh.d_obj_colors, mesh.num_obj_colors,
                    5, interop.surface, rt_w, rt_h);
                break;

            case ViewportPassMode::Solid:
                pathtracer_solid_shade(render_w, render_h, cam,
                    d_bvh, d_prims, (int)prims_sorted.size(),
                    mesh.d_bvh, mesh.d_prims, (int)mesh.prims.size(),
                    mesh.d_mats, mesh.num_mats, mesh.d_tex_hdls,
                    mesh.d_obj_colors, mesh.num_obj_colors,
                    ctrl.color_mode,
                    make_float3(ctrl.bg_color[0], ctrl.bg_color[1], ctrl.bg_color[2]),
                    light_key_dir, light_fill_dir, light_rim_dir,
                    interop.surface, rt_w, rt_h);
                break;

            case ViewportPassMode::Rasterized:
                rasterizer_render(raster_state, interop.surface, rt_w, rt_h, cam,
                    mesh.d_prims, (int)mesh.prims.size(),
                    mesh.d_mats, mesh.num_mats, mesh.d_tex_hdls,
                    mesh.d_obj_colors, mesh.num_obj_colors,
                    ctrl.color_mode,
                    make_float3(ctrl.bg_color[0], ctrl.bg_color[1], ctrl.bg_color[2]),
                    light_key_dir, light_fill_dir, light_rim_dir);
                break;

            case ViewportPassMode::Final:
            default:
                break;
            }

            // Debug PiP overlay: small inset of the selected overlay pass.
            // Always constrain to the top-left pip_w × pip_h region so this
            // overlay never replaces the main viewport — only adds a small inset.
            // Show regardless of DLSS % so the pip stays a fixed rt_w/3 size.
            if (ctrl.dlss_enabled && ctrl.dlss_debug) {
                int pip_w = rt_w / 3, pip_h = rt_h / 3;
                switch ((ViewportPassMode)ctrl.dlss_overlay_pass) {
                case ViewportPassMode::DlssDepth:
                    pathtracer_visualize_depth(d_depth, render_w, render_h,
                                               depth_min, depth_max,
                                               interop.surface, pip_w, pip_h);
                    break;
                case ViewportPassMode::DlssMotion:
                    pathtracer_visualize_motion(d_motion, render_w, render_h,
                                                motion_maxmag,
                                                interop.surface, pip_w, pip_h);
                    break;
                default:
                    // Raw Render / DLSS Input: show small native render as PiP
                    pathtracer_sw_upscale_debug_pip(d_accum, render_w, render_h,
                                                    interop.surface, rt_w, rt_h);
                    break;
                }
            }
        }

        // Ensure all CUDA writes to interop.surface are complete before Vulkan reads it.
        // (First cudaDeviceSynchronize covers the render dispatch; this one covers
        //  denoiser blits, viewport pass overrides, and debug PiP.)
        cudaDeviceSynchronize();

        // ── GPU Stats update ──────────────────────────────────────────
        {
            size_t sz_bvh   = mesh.d_bvh   ? mesh.bvh_nodes.size() * sizeof(BVHNode)     : 0;
            size_t sz_tris  = mesh.d_prims ? mesh.prims.size()      * sizeof(Triangle)   : 0;
            size_t sz_mats  = mesh.d_mats  ? (size_t)mesh.num_mats  * sizeof(GpuMaterial): 0;
            size_t sz_texs  = mesh.tex_bytes;
            size_t sz_accum = (size_t)rt_w * rt_h * sizeof(float4);
            size_t sz_rng   = 0; // PCG hash RNG — no GPU state buffer
            size_t sz_res   = mesh.d_reservoirs
                ? (size_t)rt_w * rt_h * sizeof(Reservoir) * 2 : 0;
            size_t sz_hdri  = hdri.loaded
                ? (size_t)hdri.width * hdri.height * sizeof(float4) : 0;

            stats_update(nvml, live_stats, hw_info,
                render_w, render_h,
                ctrl.optix_enabled,
                use_optix_rt /*RT cores: hardware traversal via OptiX*/,
                dlss_active && ctrl.dlss_enabled,
                mesh.num_texs > 0, hdri.loaded,
                sz_bvh, sz_tris, sz_mats, sz_texs,
                sz_accum, sz_rng, sz_res, sz_hdri);
        }

        ctrl.frame_ms      = frame_ms;
        ctrl.mrays_per_sec = (float)render_w * render_h * ctrl.spp * ctrl.max_depth / (frame_ms * 1000.f);
        ctrl.frame_count   = frame_count;

        // ── Render ────────────────────────────────────────────────────
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();

        VkCommandBuffer cb = vk.commandBuffers[img_idx];
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cb, &bi);

        // ── DLSS upscaling — must run BEFORE post-process ────────────────────
        // DLSS reads render_w×render_h from interop.image and blit-backs
        // the upscaled full-res result into interop.image so post-process
        // can read the full-resolution HDR content.
        // Skip DLSS for all non-Final passes — the viewport overrides above
        // already wrote the correct full-res debug view; running DLSS on top
        // would re-read the render_w×render_h sub-region and zoom it back in.
#ifdef DLSS_ENABLED
        if (dlss_active && ctrl.dlss_enabled && ctrl.viewport_pass == (int)ViewportPassMode::Final) {
            dlss_upscale(dlss, cb,
                         interop.image, interop.image_view, interop.memory,
                         ctrl.post.exposure, cam,
                         ctrl.vfov, (float)rt_w / (float)rt_h,
                         cam_changed);
        }
#endif

        // Post-processing compute pass — reads interop.image (full-res after DLSS).
        // Must run every frame regardless of viewport pass: the dispatch includes
        // Vulkan image-layout transitions that keep post.display_image in the correct
        // layout. Skipping it on data passes would leave the image in a broken state
        // for the next frame and black-out the entire viewport.
        if (ctrl.post_enabled && post.pipeline) {
            PostPushConstants pc = ctrl.post;
            post_process_dispatch(post, cb, pc);
        }

        VkClearValue clear = {{ {0.1f, 0.1f, 0.1f, 1.f} }};
        VkRenderPassBeginInfo rp{};
        rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass        = vk.renderPass;
        rp.framebuffer       = vk.framebuffers[img_idx];
        rp.renderArea.extent = vk.swapExtent;
        rp.clearValueCount   = 1;
        rp.pClearValues      = &clear;
        vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
        ImGui_ImplVulkan_RenderDrawData(draw_data, cb);
        vkCmdEndRenderPass(cb);
        vkEndCommandBuffer(cb);

        vk_end_frame(vk, img_idx);

        // Multi-viewport: render any ImGui windows torn off to a second monitor
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        // SM tracker snapshot every 500ms
        arch_timer_ms += frame_ms;
        if (arch_timer_ms >= 500.f) {
            arch_timer_ms = 0.f;
            sm_tracker_read_reset(sm_tracker, arch_state.sm_active);

            // Fallback: when OptiX RT (or other non-CUDA path) is active the
            // pathtrace_kernel never fires, so the bitmask stays zero.
            // Approximate SM activity from NVML GPU utilisation instead.
            bool any_active = false;
            for (int w = 0; w < GpuArchWindowState::MAX_SM_WORDS; w++)
                if (arch_state.sm_active[w]) { any_active = true; break; }
            if (!any_active && arch_state.gpu_util_pct > 0 && hw_info.sm_count > 0) {
                int n = (hw_info.sm_count * (int)arch_state.gpu_util_pct + 99) / 100;
                n = std::min(n, hw_info.sm_count);
                // Spread active SMs evenly across all GPCs rather than filling
                // linearly (linear would make all early GPCs look 100% busy while
                // later GPCs stay dark — not representative of real scheduling).
                for (int i = 0; i < n; i++) {
                    int idx = (int)((int64_t)i * hw_info.sm_count / n);
                    arch_state.sm_active[idx >> 5] |= (1u << (idx & 31));
                }
            }

            // Workload-specific fallback activity (only when CUPTI unit counters
            // are not available). Do not override real hardware counter samples.
            if (!arch_state.has_unit_counters) {
                float util = (float)arch_state.gpu_util_pct;
                arch_state.rt_active_pct     = use_optix_rt ? util : 0.f;
                arch_state.tensor_active_pct = dlss_active  ? util : 0.f;
                // Texture units can be exercised by mesh textures and HDRI sampling.
                const bool uses_textures = (mesh.num_texs > 0) || hdri.loaded;
                arch_state.tex_active_pct = (util > 0.f && uses_textures) ? util : 0.f;
            }
        }
    }

    // Save window position/size FIRST — before any cleanup can alter window state
    save_win_state(window, ctrl.show_gpu_arch);

    vkDeviceWaitIdle(vk.device);
    batch_stop(batch);
    if (nim_thread.joinable()) nim_thread.join();
    if (cosmos_thread.joinable()) cosmos_thread.join();
    if (ctrl.cosmos_result_pixels) { delete[] ctrl.cosmos_result_pixels; ctrl.cosmos_result_pixels = nullptr; }
#ifdef OPTIX_ENABLED
    if (optix_rt_thread.joinable()) optix_rt_thread.join();
#endif
    optix_denoiser_free(optix_dn);
    optix_renderer_free(optix_rt);
    dlss_free(dlss);
    dlss_shutdown();
    sm_tracker_destroy(sm_tracker);
    stats_shutdown(nvml);
    post_process_destroy(post);
    if (usd_anim_handle) { usd_anim_close(usd_anim_handle); usd_anim_handle = nullptr; }
    mesh.free_gpu();
    cudaFree(d_accum); cudaFree(d_display);
    cudaFree(d_depth); cudaFree(d_motion);
    rasterizer_destroy(raster_state);
    cudaFree(d_bvh); cudaFree(d_prims);
    cuda_interop_destroy(vk.device, interop);
    if (hydra_preview_ready)
        safe_hydra_preview_destroy(hp_preview);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vk_destroy(vk);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[fatal] %s\n", e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "[fatal] Unknown exception\n");
        return 1;
    }
}
