#include "bvh.h"
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
//  CPU BVH Build  (median split)
// ─────────────────────────────────────────────

static AABB sphere_aabb(const Sphere& s) {
    float3 r = make_float3(s.radius, s.radius, s.radius);
    return { make_float3(s.center.x - r.x, s.center.y - r.y, s.center.z - r.z),
             make_float3(s.center.x + r.x, s.center.y + r.y, s.center.z + r.z) };
}

static AABB triangle_aabb(const Triangle& t) {
    float3 mn = make_float3(
        fminf(fminf(t.v0.x, t.v1.x), t.v2.x),
        fminf(fminf(t.v0.y, t.v1.y), t.v2.y),
        fminf(fminf(t.v0.z, t.v1.z), t.v2.z));
    float3 mx = make_float3(
        fmaxf(fmaxf(t.v0.x, t.v1.x), t.v2.x),
        fmaxf(fmaxf(t.v0.y, t.v1.y), t.v2.y),
        fmaxf(fmaxf(t.v0.z, t.v1.z), t.v2.z));
    // Pad degenerate (flat) triangles so AABB is never zero-thickness
    const float eps = 1e-4f;
    if (mx.x - mn.x < eps) { mn.x -= eps; mx.x += eps; }
    if (mx.y - mn.y < eps) { mn.y -= eps; mx.y += eps; }
    if (mx.z - mn.z < eps) { mn.z -= eps; mx.z += eps; }
    return { mn, mx };
}

static AABB aabb_union(AABB a, AABB b) {
    return {
        make_float3(fminf(a.mn.x,b.mn.x), fminf(a.mn.y,b.mn.y), fminf(a.mn.z,b.mn.z)),
        make_float3(fmaxf(a.mx.x,b.mx.x), fmaxf(a.mx.y,b.mx.y), fmaxf(a.mx.z,b.mx.z))
    };
}

static float3 aabb_centroid(const AABB& a) {
    return make_float3((a.mn.x+a.mx.x)*0.5f, (a.mn.y+a.mx.y)*0.5f, (a.mn.z+a.mx.z)*0.5f);
}

// ─────────────────────────────────────────────
//  Sphere BVH build
// ─────────────────────────────────────────────

struct SphereCtx { std::vector<BVHNode>& nodes; std::vector<Sphere>& prims; };

static int build_recursive(SphereCtx& ctx, int start, int end) {
    int node_idx = (int)ctx.nodes.size();
    ctx.nodes.emplace_back();
    BVHNode& node = ctx.nodes.back();

    AABB box = sphere_aabb(ctx.prims[start]);
    for (int i = start + 1; i < end; ++i)
        box = aabb_union(box, sphere_aabb(ctx.prims[i]));
    node.aabb = box;

    int count = end - start;
    if (count <= 4) {
        node.left = node.right = -1;
        node.prim_start = start;
        node.prim_count = count;
        return node_idx;
    }

    float3 extent = make_float3(box.mx.x-box.mn.x, box.mx.y-box.mn.y, box.mx.z-box.mn.z);
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > (&extent.x)[axis]) axis = 2;

    std::sort(ctx.prims.begin() + start, ctx.prims.begin() + end,
        [axis](const Sphere& a, const Sphere& b) {
            return (&aabb_centroid(sphere_aabb(a)).x)[axis] <
                   (&aabb_centroid(sphere_aabb(b)).x)[axis];
        });

    int mid = start + count / 2;
    node.prim_start = -1;
    node.prim_count = 0;

    int left  = build_recursive(ctx, start, mid);
    int right = build_recursive(ctx, mid,   end);
    ctx.nodes[node_idx].left  = left;
    ctx.nodes[node_idx].right = right;
    return node_idx;
}

void bvh_build(
    const std::vector<Sphere>& prims_in,
    std::vector<BVHNode>&      nodes_out,
    std::vector<Sphere>&       prims_out)
{
    prims_out = prims_in;
    nodes_out.reserve(prims_in.size() * 2);
    SphereCtx ctx{ nodes_out, prims_out };
    build_recursive(ctx, 0, (int)prims_out.size());
}

void bvh_upload(
    const std::vector<BVHNode>& nodes,
    const std::vector<Sphere>&  prims,
    BVHNode** d_nodes,
    Sphere**  d_prims)
{
    cudaMalloc(d_nodes, nodes.size() * sizeof(BVHNode));
    cudaMalloc(d_prims, prims.size() * sizeof(Sphere));
    cudaMemcpy(*d_nodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_prims, prims.data(), prims.size() * sizeof(Sphere),  cudaMemcpyHostToDevice);
}

// ─────────────────────────────────────────────
//  Triangle BVH build
// ─────────────────────────────────────────────

struct TriCtx { std::vector<BVHNode>& nodes; std::vector<Triangle>& prims; };

static int build_recursive_tri(TriCtx& ctx, int start, int end) {
    int node_idx = (int)ctx.nodes.size();
    ctx.nodes.emplace_back();
    BVHNode& node = ctx.nodes.back();

    AABB box = triangle_aabb(ctx.prims[start]);
    for (int i = start + 1; i < end; ++i)
        box = aabb_union(box, triangle_aabb(ctx.prims[i]));
    node.aabb = box;

    int count = end - start;
    if (count <= 8) {
        node.left = node.right = -1;
        node.prim_start = start;
        node.prim_count = count;
        return node_idx;
    }

    float3 extent = make_float3(box.mx.x-box.mn.x, box.mx.y-box.mn.y, box.mx.z-box.mn.z);
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > (&extent.x)[axis]) axis = 2;

    std::sort(ctx.prims.begin() + start, ctx.prims.begin() + end,
        [axis](const Triangle& a, const Triangle& b) {
            float ca = ((&a.v0.x)[axis] + (&a.v1.x)[axis] + (&a.v2.x)[axis]) / 3.f;
            float cb = ((&b.v0.x)[axis] + (&b.v1.x)[axis] + (&b.v2.x)[axis]) / 3.f;
            return ca < cb;
        });

    int mid = start + count / 2;
    node.prim_start = -1;
    node.prim_count = 0;

    int left  = build_recursive_tri(ctx, start, mid);
    int right = build_recursive_tri(ctx, mid,   end);
    ctx.nodes[node_idx].left  = left;
    ctx.nodes[node_idx].right = right;
    return node_idx;
}

void bvh_build_triangles(
    const std::vector<Triangle>& prims_in,
    std::vector<BVHNode>&        nodes_out,
    std::vector<Triangle>&       prims_out)
{
    prims_out = prims_in;
    nodes_out.reserve(prims_in.size() / 4 * 2 + 4);
    TriCtx ctx{ nodes_out, prims_out };
    build_recursive_tri(ctx, 0, (int)prims_out.size());
}

void bvh_upload_triangles(
    const std::vector<BVHNode>&  nodes,
    const std::vector<Triangle>& prims,
    BVHNode**  d_nodes,
    Triangle** d_prims)
{
    cudaMalloc(d_nodes, nodes.size() * sizeof(BVHNode));
    cudaMalloc(d_prims, prims.size() * sizeof(Triangle));
    cudaMemcpy(*d_nodes, nodes.data(), nodes.size() * sizeof(BVHNode),   cudaMemcpyHostToDevice);
    cudaMemcpy(*d_prims, prims.data(), prims.size() * sizeof(Triangle),   cudaMemcpyHostToDevice);
}

// ─────────────────────────────────────────────
//  Device — sphere intersection
// ─────────────────────────────────────────────

__device__ static bool sphere_hit(const Sphere& s, const Ray& r, float t_min, float t_max, HitRecord& rec) {
    float3 oc = r.origin - s.center;
    float a  = dot(r.dir, r.dir);
    float hb = dot(oc, r.dir);
    float c  = dot(oc, oc) - s.radius * s.radius;
    float disc = hb*hb - a*c;
    if (disc < 0.f) return false;

    float sqrt_disc = sqrtf(disc);
    float root = (-hb - sqrt_disc) / a;
    if (root < t_min || root > t_max) {
        root = (-hb + sqrt_disc) / a;
        if (root < t_min || root > t_max) return false;
    }

    rec.t = root;
    rec.p = r.at(root);
    float3 outward = (rec.p - s.center) / s.radius;
    rec.set_face_normal(r, outward);
    rec.geom_normal = rec.normal;   // sphere: geometric = shading
    rec.mat         = s.mat;
    rec.gpu_mat_idx = -1;
    rec.uv          = make_float2(0.f, 0.f);
    rec.tangent     = make_float4(0.f, 0.f, 0.f, 0.f);
    return true;
}

// ─────────────────────────────────────────────
//  Device — Möller–Trumbore triangle intersection
// ─────────────────────────────────────────────

__device__ static bool tri_hit(const Triangle& tri, const Ray& r,
                                float t_min, float t_max, HitRecord& rec)
{
    const float EPS = 1e-7f;
    float3 e1 = tri.v1 - tri.v0;
    float3 e2 = tri.v2 - tri.v0;
    float3 h  = cross(r.dir, e2);
    float  a  = dot(e1, h);
    if (fabsf(a) < EPS) return false;   // parallel

    float  f = 1.f / a;
    float3 s = r.origin - tri.v0;
    float  u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return false;

    float3 q = cross(s, e1);
    float  v = f * dot(r.dir, q);
    if (v < 0.f || u + v > 1.f) return false;

    float t = f * dot(e2, q);
    if (t < t_min || t > t_max) return false;

    float w = 1.f - u - v;
    float3 shading_n = normalize(tri.n0 * w + tri.n1 * u + tri.n2 * v);
    // Geometric normal from triangle edges (not interpolated)
    float3 raw_geom  = cross(e1, e2);
    float  geom_len  = sqrtf(dot(raw_geom, raw_geom));
    float3 geom_n    = (geom_len > 1e-12f) ? raw_geom / geom_len : shading_n;

    rec.t  = t;
    rec.p  = r.at(t);
    rec.uv = make_float2(
        tri.uv0.x * w + tri.uv1.x * u + tri.uv2.x * v,
        tri.uv0.y * w + tri.uv1.y * u + tri.uv2.y * v);
    rec.gpu_mat_idx = tri.mat_idx;
    rec.obj_id      = tri.obj_id;
    rec.set_face_normal(r, shading_n);
    // Store face-corrected geometric normal (same front/back convention as shading normal)
    rec.geom_normal  = (dot(r.dir, geom_n) < 0.f) ? geom_n
                     : make_float3(-geom_n.x, -geom_n.y, -geom_n.z);
    // Interpolate tangent (xyz = direction, w = sign)
    rec.tangent = make_float4(
        tri.t0.x * w + tri.t1.x * u + tri.t2.x * v,
        tri.t0.y * w + tri.t1.y * u + tri.t2.y * v,
        tri.t0.z * w + tri.t1.z * u + tri.t2.z * v,
        tri.t0.w);  // bitangent sign — same for all three vertices in a primitive
    return true;
}

// ─────────────────────────────────────────────
//  Device — sphere BVH traversal
// ─────────────────────────────────────────────

__device__ bool bvh_hit(
    const BVHNode* nodes, const Sphere* prims,
    const Ray& r, float t_min, float t_max,
    HitRecord& rec)
{
    int stack[64]; int top = 0;
    stack[top++] = 0;
    bool hit_any = false;
    float closest = t_max;

    while (top > 0) {
        const BVHNode& node = nodes[stack[--top]];
        if (!node.aabb.hit(r, t_min, closest)) continue;
        if (node.prim_count > 0) {
            for (int i = node.prim_start; i < node.prim_start + node.prim_count; ++i) {
                HitRecord tmp;
                if (sphere_hit(prims[i], r, t_min, closest, tmp))
                    { hit_any = true; closest = tmp.t; rec = tmp; }
            }
        } else {
            if (node.left  >= 0) stack[top++] = node.left;
            if (node.right >= 0) stack[top++] = node.right;
        }
    }
    return hit_any;
}

// ─────────────────────────────────────────────
//  Device — triangle BVH traversal
// ─────────────────────────────────────────────

__device__ bool bvh_hit_triangles(
    const BVHNode* nodes, const Triangle* prims,
    const Ray& r, float t_min, float t_max,
    HitRecord& rec)
{
    int stack[64]; int top = 0;
    stack[top++] = 0;
    bool hit_any = false;
    float closest = t_max;

    while (top > 0) {
        const BVHNode& node = nodes[stack[--top]];
        if (!node.aabb.hit(r, t_min, closest)) continue;
        if (node.prim_count > 0) {
            for (int i = node.prim_start; i < node.prim_start + node.prim_count; ++i) {
                HitRecord tmp;
                if (tri_hit(prims[i], r, t_min, closest, tmp))
                    { hit_any = true; closest = tmp.t; rec = tmp; }
            }
        } else {
            if (node.left  >= 0) stack[top++] = node.left;
            if (node.right >= 0) stack[top++] = node.right;
        }
    }
    return hit_any;
}
