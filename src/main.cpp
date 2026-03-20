#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <cuda_runtime.h>

#include "vulkan_context.h"
#include "cuda_interop.h"
#include "pathtracer.h"
#include "bvh.h"
#include "scene.h"
#include "gpu_textures.h"
#include "gltf_loader.h"
#include "usd_loader.h"
#include "restir.h"
#include "hdri.h"
#include "denoiser.h"
#include "optix_denoiser.h"
#include "optix_renderer.h"
#include "dlss_upscaler.h"
#include "post_process.h"
#include "nim_vlm.h"
#include "slang_shader.h"
#include "batch_processor.h"
#include <tinyexr.h>
#include "ui/viewport.h"
#include "ui/control_panel.h"
#include "ui/outliner.h"
#include "ui/stats_panel.h"
#include "ui/material_panel.h"

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

// Draw XYZ translate gizmo. Returns active axis (1=X,2=Y,3=Z, 0=none).
// out_delta: world-space movement to apply this frame.
static int draw_move_gizmo(ImDrawList* dl, const Camera& cam, float3 center,
                            float vfov, float aspect,
                            ImVec2 vp_origin, ImVec2 vp_size,
                            float3& out_delta)
{
    static int drag_axis = 0;
    out_delta = {0.f, 0.f, 0.f};

    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) drag_axis = 0;

    ImVec2 cs; // center screen pos
    float depth = world_to_screen(cam, center, vp_origin, vp_size, vfov, aspect, cs);
    if (depth <= 0.f) return 0;

    // Gizmo arrow length in screen pixels
    const float ARROW_PX = 80.f;
    const float HIT_PX   = 8.f;
    const float TIP_PX   = 10.f;

    // World-space scale so arrow = ARROW_PX screen pixels
    float half_h = tanf(vfov * 0.5f * 3.14159265f / 180.f);
    float scale  = depth * 2.f * half_h / vp_size.y * ARROW_PX;

    float3 axis_dir[4]  = { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} };
    ImU32  axis_col[4]  = { 0,
        IM_COL32(220, 60,  60, 255),   // X red
        IM_COL32( 60,200,  60, 255),   // Y green
        IM_COL32( 60,120, 220, 255) }; // Z blue

    ImVec2 tip[4];
    for (int i = 1; i <= 3; ++i) {
        float3 w = {
            center.x + axis_dir[i].x * scale,
            center.y + axis_dir[i].y * scale,
            center.z + axis_dir[i].z * scale };
        world_to_screen(cam, w, vp_origin, vp_size, vfov, aspect, tip[i]);
    }

    // Hover detection (only when not dragging)
    ImVec2 mouse = ImGui::GetIO().MousePos;
    int hovered = 0;
    if (drag_axis == 0) {
        for (int i = 1; i <= 3; ++i) {
            if (seg_dist_sq(mouse.x, mouse.y,
                            cs.x, cs.y, tip[i].x, tip[i].y) < HIT_PX*HIT_PX) {
                hovered = i; break;
            }
        }
    }

    if (hovered > 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        drag_axis = hovered;

    int display = drag_axis > 0 ? drag_axis : hovered;

    // Draw
    for (int i = 1; i <= 3; ++i) {
        ImU32 c = (i == display) ? IM_COL32(255, 255, 100, 255) : axis_col[i];
        dl->AddLine(cs, tip[i], c, 2.5f);
        // Arrowhead
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
    dl->AddCircleFilled(cs, 4.f, IM_COL32(255, 255, 255, 200));

    // Apply drag movement
    ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (drag_axis > 0 && (delta.x != 0.f || delta.y != 0.f)) {
        float3 ad = axis_dir[drag_axis];
        float sax = tip[drag_axis].x - cs.x;
        float say = tip[drag_axis].y - cs.y;
        float slen = sqrtf(sax*sax + say*say);
        if (slen > 0.f) {
            float proj = (delta.x * sax + delta.y * say) / slen;
            float wpp  = scale / ARROW_PX; // world units per screen pixel
            float move = proj * wpp;
            out_delta = { ad.x * move, ad.y * move, ad.z * move };
        }
    }

    return (drag_axis > 0) ? drag_axis : hovered;
}

// ─────────────────────────────────────────────
//  Silhouette edge detection (CPU, for selection outline)
// ─────────────────────────────────────────────

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
        objects.clear();
        host_mats.clear();
        num_mats = 0; num_texs = 0; num_obj_colors = 0; num_emissive = 0; tex_bytes = 0;
    }
};

static bool load_gltf_into(const std::string& path, MeshState& ms,
                            ControlPanelState& ctrl)
{
    ms.free_gpu();

    std::vector<Triangle>     tris;
    std::vector<GpuMaterial>  mats;
    std::vector<TextureImage> tex_images;

    // Route by extension
    auto ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);
    bool is_usd = (ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz");

    bool load_ok = is_usd
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

// Rebuild GPU BVH from only the visible objects in all_prims.
// ms.prims becomes the sorted visible subset; all_prims stays untouched.
static void rebuild_visible_bvh(MeshState& ms)
{
    std::unordered_map<int,bool> hidden_map;
    for (auto& obj : ms.objects)
        hidden_map[obj.obj_id] = obj.hidden;

    std::vector<Triangle> visible;
    visible.reserve(ms.all_prims.size());
    for (auto& t : ms.all_prims)
        if (!hidden_map[t.obj_id]) visible.push_back(t);

    cudaFree(ms.d_bvh);   ms.d_bvh   = nullptr;
    cudaFree(ms.d_prims); ms.d_prims = nullptr;
    ms.bvh_nodes.clear();
    ms.prims.clear();

    if (!visible.empty()) {
        bvh_build_triangles(visible, ms.bvh_nodes, ms.prims);
        bvh_upload_triangles(ms.bvh_nodes, ms.prims, &ms.d_bvh, &ms.d_prims);
    }
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

struct WinState { int x, y, w, h; bool maximized; };

static WinState load_win_state()
{
    WinState s{ 100, 100, 1600, 900, false };
    auto path = exe_dir_main() / "window.ini";
    std::ifstream f(path);
    if (!f) return s;
    f >> s.x >> s.y >> s.w >> s.h >> s.maximized;

    // Clamp degenerate sizes
    if (s.w < 400) s.w = 1600;
    if (s.h < 300) s.h = 900;

    // Make sure the title bar is reachable on some monitor (32px safety margin)
    POINT pt{ s.x + s.w / 2, s.y + 16 };
    if (!MonitorFromPoint(pt, MONITOR_DEFAULTTONULL)) {
        // Saved position is offscreen — reset to primary monitor default
        s.x = 100; s.y = 100;
    }
    return s;
}

static void save_win_state(GLFWwindow* win)
{
    // GetWindowPlacement returns rcNormalPosition — the restored rect —
    // correctly even when the window is currently maximized (no async issues).
    HWND hwnd = glfwGetWin32Window(win);
    WINDOWPLACEMENT wp{};
    wp.length = sizeof(wp);
    if (!GetWindowPlacement(hwnd, &wp)) return;

    WinState s{};
    s.maximized = (wp.showCmd == SW_SHOWMAXIMIZED);
    s.x = wp.rcNormalPosition.left;
    s.y = wp.rcNormalPosition.top;
    s.w = wp.rcNormalPosition.right  - wp.rcNormalPosition.left;
    s.h = wp.rcNormalPosition.bottom - wp.rcNormalPosition.top;

    // Sanity: don't save a degenerate size
    if (s.w < 400 || s.h < 300) return;

    auto path = exe_dir_main() / "window.ini";
    std::ofstream f(path);
    if (f) f << s.x << ' ' << s.y << ' ' << s.w << ' ' << s.h << ' ' << s.maximized << '\n';
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

int main() {
    if (!glfwInit()) throw std::runtime_error("GLFW init failed");
    // Streamline DLSS: slInit() must be called before any vkCreateInstance.
    // plugin_dir = exe directory where sl.dlss.dll, sl.common.dll etc. are deployed.
    {
        char exe_buf[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exe_buf, MAX_PATH);
        std::filesystem::path exe_dir = std::filesystem::path(exe_buf).parent_path();
        dlss_pre_vulkan_init(exe_dir.string().c_str());
    }

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

    // DLSS auxiliary buffers — depth (float) and motion vectors (float2), render-res
    float*  d_depth  = nullptr;  cudaMalloc(&d_depth,  (size_t)rt_w * rt_h * sizeof(float));
    float2* d_motion = nullptr;  cudaMalloc(&d_motion, (size_t)rt_w * rt_h * sizeof(float2));
    Camera prev_cam = {};
    bool   has_prev_cam = false;

    // ReSTIR reservoir buffers (reallocated on resize)
    // Allocated in MeshState lazily at first use so they match actual viewport size

    ViewportPanel            vp;
    ControlPanelState        ctrl;
    MeshState                mesh;
    std::unordered_set<int>  multi_sel;   // obj indices currently selected
    int frame_count = 0;

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
    ctrl.dlss_has_sdk = true;
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
                break;
            }
        }
    }

    double prev_mx = 0.0, prev_my = 0.0;
    bool   rmb_was_down = false;

    auto t_last = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        auto t_now = std::chrono::high_resolution_clock::now();
        float frame_ms = std::chrono::duration<float, std::milli>(t_now - t_last).count();
        if (frame_ms < 0.001f) frame_ms = 0.001f;
        t_last = t_now;

        uint32_t img_idx = vk_begin_frame(vk);
        if (img_idx == UINT32_MAX) { vk_recreate_swapchain(vk, window); continue; }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        bool cam_changed = control_panel_draw(ctrl);
        viewport_draw(vp, (ctrl.post_enabled && post.imgui_desc) ? post.imgui_desc : interop.descriptor, ctrl);
        // Outliner — selection from here doesn't reset accumulation
        outliner_draw(prims_sorted, mesh.objects,
                      ctrl.selected_sphere, ctrl.selected_mesh_obj, multi_sel);
        stats_draw(hw_info, live_stats);
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
        if (ctrl.save_exr_requested) {
            ctrl.save_exr_requested = false;
            if (ctrl.save_exr_path[0] != '\0' && frame_count > 0) {
                std::vector<float> px((size_t)rt_w * rt_h * 4);
                cudaMemcpy(px.data(), d_accum, px.size() * sizeof(float), cudaMemcpyDeviceToHost);
                // Divide accumulated sum by frame count to get average HDR values
                float inv = 1.f / (float)frame_count;
                std::vector<float> r(rt_w * rt_h), g(rt_w * rt_h), b(rt_w * rt_h);
                for (int i = 0; i < rt_w * rt_h; ++i) {
                    r[i] = px[i * 4 + 0] * inv;
                    g[i] = px[i * 4 + 1] * inv;
                    b[i] = px[i * 4 + 2] * inv;
                }
                EXRHeader header; InitEXRHeader(&header);
                EXRImage  image;  InitEXRImage(&image);
                image.num_channels = 3;
                float* channels[3] = { b.data(), g.data(), r.data() }; // EXR: BGR order
                image.images = reinterpret_cast<unsigned char**>(channels);
                image.width  = rt_w;
                image.height = rt_h;
                header.num_channels = 3;
                header.channels     = static_cast<EXRChannelInfo*>(
                    malloc(sizeof(EXRChannelInfo) * 3));
                strncpy(header.channels[0].name, "B", 255);
                strncpy(header.channels[1].name, "G", 255);
                strncpy(header.channels[2].name, "R", 255);
                header.pixel_types           = static_cast<int*>(malloc(sizeof(int) * 3));
                header.requested_pixel_types = static_cast<int*>(malloc(sizeof(int) * 3));
                for (int i = 0; i < 3; ++i) {
                    header.pixel_types[i]           = TINYEXR_PIXELTYPE_FLOAT;
                    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
                }
                const char* err = nullptr;
                int ret = SaveEXRImageToFile(&image, &header, ctrl.save_exr_path, &err);
                if (ret != TINYEXR_SUCCESS && err) FreeEXRErrorMessage(err);
                free(header.channels);
                free(header.pixel_types);
                free(header.requested_pixel_types);
            }
        }

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
            vkDeviceWaitIdle(vk.device);
            cuda_interop_destroy(vk.device, interop);
            rt_w = (int)vp.size.x; rt_h = (int)vp.size.y;
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
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS)
                pan( fwd_x*spd, 0.f,  fwd_z*spd);
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS)
                pan(-fwd_x*spd, 0.f, -fwd_z*spd);
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS)
                pan( rgt_x*spd, 0.f,  rgt_z*spd);
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                pan(-rgt_x*spd, 0.f, -rgt_z*spd);
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                pan(0.f,  spd, 0.f);
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
                pan(0.f, -spd, 0.f);
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
                } else {
                    // Orbit → FPS: derive yaw/pitch from pos→pivot direction
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
                    // pos is already the camera world position — no sync needed
                }
                s_prev = ctrl.interact_mode;
            }
        }

        // ── ORBIT drag + scroll: arcball (pos rotates around pivot) ──
        // Alt held = temporary orbit override regardless of current mode
        bool alt_orbit = io.KeyAlt && vp.hovered;
        if (ctrl.interact_mode == InteractMode::Orbit || alt_orbit) {
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
        if (ctrl.interact_mode == InteractMode::Orbit || alt_orbit) {
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

        // ── Mesh object gizmo ─────────────────────────────────────────
        if (ctrl.overlay_gizmo && ctrl.selected_mesh_obj >= 0 &&
            ctrl.selected_mesh_obj < (int)mesh.objects.size() &&
            !multi_sel.empty() &&
            ctrl.interact_mode == InteractMode::Move && !alt_orbit)
        {
            // Gizmo centred on the primary selected object
            MeshObject& mobj = mesh.objects[ctrl.selected_mesh_obj];
            int gr = draw_move_gizmo(
                ImGui::GetForegroundDrawList(), cam,
                mobj.centroid,
                ctrl.vfov, (float)rt_w / (float)rt_h,
                vp.origin, vp.size, gizmo_delta);

            if (!gizmo_consuming)
                gizmo_consuming = (gr > 0) && ImGui::IsMouseDown(ImGuiMouseButton_Left);

            // Track drag-end to trigger full BVH rebuild
            static bool s_mesh_dragging = false;
            bool dragging_now = (gr > 0) && ImGui::IsMouseDown(ImGuiMouseButton_Left);
            bool drag_released = s_mesh_dragging && !dragging_now;
            s_mesh_dragging = dragging_now;

            if (gizmo_delta.x != 0.f || gizmo_delta.y != 0.f || gizmo_delta.z != 0.f) {
                // Translate ALL objects in the current selection group
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

                // During drag: expand every BVH node AABB by the movement delta
                // so no moved triangle gets culled. O(nodes) — instant.
                // Full tight rebuild happens on mouse release.
                float3 d = gizmo_delta;
                for (auto& node : mesh.bvh_nodes) {
                    if (d.x > 0.f) node.aabb.mx.x += d.x; else node.aabb.mn.x += d.x;
                    if (d.y > 0.f) node.aabb.mx.y += d.y; else node.aabb.mn.y += d.y;
                    if (d.z > 0.f) node.aabb.mx.z += d.z; else node.aabb.mn.z += d.z;
                }
                if (mesh.d_bvh && !mesh.bvh_nodes.empty())
                    cudaMemcpy(mesh.d_bvh, mesh.bvh_nodes.data(),
                               mesh.bvh_nodes.size() * sizeof(BVHNode),
                               cudaMemcpyHostToDevice);
                if (mesh.d_prims && !mesh.prims.empty())
                    cudaMemcpy(mesh.d_prims, mesh.prims.data(),
                               mesh.prims.size() * sizeof(Triangle),
                               cudaMemcpyHostToDevice);
                mesh_moved  = true;
                cam_changed = true;
                ++mesh.scene_version;  // OptiX RT re-uploads BVH
            }

            // Full BVH rebuild when the user releases the gizmo
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
                // Camera position unchanged — only the look-at point shifts
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
        pt.firefly_clamp    = ctrl.firefly_clamp;

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
                std::strncpy(ctrl.optix_rt_last_error, optix_rt.last_error,
                             sizeof(ctrl.optix_rt_last_error) - 1);
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
            std::strncpy(ctrl.optix_rt_last_error, optix_rt.last_error, sizeof(ctrl.optix_rt_last_error) - 1);
#endif

        // ── Dispatch: ReSTIR, OptiX RT, or standard path tracer ──────
        bool use_restir = ctrl.restir_enabled && mesh.num_emissive > 0 && mesh.d_bvh;
        bool use_optix_rt = ctrl.optix_rt_enabled && optix_rt.available && optix_rt.scene_ready;
        if (use_restir) {
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
        if (use_optix_rt)
            pathtracer_blit_surface(d_accum, interop.surface, rt_w, rt_h);

        ++frame_count;

        // ── Denoiser: always reads d_accum, writes to d_display ──────────────
        // d_accum is NEVER modified — it stays as the raw accumulated data.
        // d_display is blitted to the surface so the viewport shows the denoised result.
        // d_display_valid: true once the denoiser has written at least one frame.
        // On non-denoise frames we re-blit d_display so the viewport never shows
        // raw noisy data while denoising is enabled (prevents the back-and-forth flicker).
        static bool d_display_valid = false;
        bool did_denoise = false;
        bool denoising_active = false;

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
                    pathtracer_sw_upscale(d_display, render_w, render_h, interop.surface, rt_w, rt_h);
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
                pathtracer_sw_upscale(d_display, render_w, render_h, interop.surface, rt_w, rt_h);
                d_display_valid = true;
                did_denoise = true;
            }
        }

        // On non-denoise frames: if denoising is active and we have a valid denoised
        // frame, re-blit it so the viewport always shows the denoised result.
        if (!did_denoise && denoising_active && d_display_valid)
            pathtracer_sw_upscale(d_display, render_w, render_h, interop.surface, rt_w, rt_h);

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
                // Auto-normalized: fastest pixel = full saturation, static = black.
                pathtracer_visualize_motion(d_motion, render_w, render_h,
                                            motion_maxmag,
                                            interop.surface, rt_w, rt_h);
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
    }

    vkDeviceWaitIdle(vk.device);
    batch_stop(batch);
    if (nim_thread.joinable()) nim_thread.join();
#ifdef OPTIX_ENABLED
    if (optix_rt_thread.joinable()) optix_rt_thread.join();
#endif
    optix_denoiser_free(optix_dn);
    optix_renderer_free(optix_rt);
    dlss_free(dlss);
    stats_shutdown(nvml);
    post_process_destroy(post);
    mesh.free_gpu();
    cudaFree(d_accum); cudaFree(d_display);
    cudaFree(d_depth); cudaFree(d_motion);
    cudaFree(d_bvh); cudaFree(d_prims);
    cuda_interop_destroy(vk.device, interop);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vk_destroy(vk);
    save_win_state(window);   // write window.ini before destroying the window
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
