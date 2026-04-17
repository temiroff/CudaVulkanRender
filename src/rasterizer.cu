#include "rasterizer.h"
#include "shading_common.cuh"
#include <cfloat>

// ─────────────────────────────────────────────
//  State management
// ─────────────────────────────────────────────

// Max large triangles per frame (screen-filling tris — rarely more than a dozen)
static constexpr int MAX_LARGE_TRIS = 512;
static constexpr int MAX_LARGE_CLIPPED_TRIS = 1024;

void rasterizer_init(RasterizerState& state)
{
    state.d_zbuffer = nullptr;
    state.d_large_tris = nullptr;
    state.d_large_count = nullptr;
    state.d_large_clipped_tris = nullptr;
    state.d_large_clipped_count = nullptr;
    state.zbuf_w = state.zbuf_h = 0;
}

void rasterizer_resize(RasterizerState& state, int w, int h)
{
    if (state.zbuf_w == w && state.zbuf_h == h) return;
    cudaFree(state.d_zbuffer);
    state.d_zbuffer = nullptr;
    state.zbuf_w = w;
    state.zbuf_h = h;
    if (w > 0 && h > 0)
        cudaMalloc(&state.d_zbuffer, (size_t)w * h * sizeof(float));

    // Allocate large-triangle list once
    if (!state.d_large_tris) {
        cudaMalloc(&state.d_large_tris,  MAX_LARGE_TRIS * sizeof(int));
        cudaMalloc(&state.d_large_count, sizeof(int));
    }
    if (!state.d_large_clipped_tris) {
        cudaMalloc(&state.d_large_clipped_tris,  MAX_LARGE_CLIPPED_TRIS * sizeof(Triangle));
        cudaMalloc(&state.d_large_clipped_count, sizeof(int));
    }
}

void rasterizer_destroy(RasterizerState& state)
{
    cudaFree(state.d_zbuffer);
    cudaFree(state.d_large_tris);
    cudaFree(state.d_large_count);
    cudaFree(state.d_large_clipped_tris);
    cudaFree(state.d_large_clipped_count);
    state.d_zbuffer     = nullptr;
    state.d_large_tris  = nullptr;
    state.d_large_count = nullptr;
    state.d_large_clipped_tris = nullptr;
    state.d_large_clipped_count = nullptr;
    state.zbuf_w = state.zbuf_h = 0;
}

// ─────────────────────────────────────────────
//  Clear kernel: reset z-buffer + fill background
// ─────────────────────────────────────────────

// bg_srgb is pre-converted to sRGB on the CPU to avoid per-pixel powf.
__global__ void raster_clear_kernel(float* zbuf, cudaSurfaceObject_t surf,
                                    int w, int h, float3 bg_srgb)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    zbuf[y * w + x] = FLT_MAX;
    surf2Dwrite(make_float4(bg_srgb.x, bg_srgb.y, bg_srgb.z, 1.f),
                surf, x * (int)sizeof(float4), y);
}

// ─────────────────────────────────────────────
//  Projection: invert Camera ray generation
// ─────────────────────────────────────────────

__device__ inline bool project(float3 P, const Camera& cam,
                               float inv_vw, float inv_vh,
                               int w, int h,
                               float& out_sx, float& out_sy, float& out_z)
{
    float3 d = P - cam.origin;
    out_z = -dot(d, cam.w);
    if (out_z < 1e-4f) return false;

    float u_ndc = dot(d, cam.u) / out_z * inv_vw + 0.5f;
    float v_ndc = dot(d, cam.v) / out_z * inv_vh + 0.5f;

    out_sx = u_ndc * (float)w;
    out_sy = (1.f - v_ndc) * (float)h;
    return true;
}

struct RasterVertex {
    float3 pos;
    float3 normal;
    float2 uv;
    float4 tangent;
};

__device__ inline float3 normalize_or(float3 v, float3 fallback)
{
    float len2 = dot(v, v);
    if (len2 <= 1e-12f) return fallback;
    return v / sqrtf(len2);
}

__device__ inline RasterVertex raster_lerp(const RasterVertex& a, const RasterVertex& b, float t)
{
    RasterVertex out;
    out.pos = a.pos + (b.pos - a.pos) * t;
    out.normal = normalize_or(a.normal + (b.normal - a.normal) * t, a.normal);
    out.uv = make_float2(
        a.uv.x + (b.uv.x - a.uv.x) * t,
        a.uv.y + (b.uv.y - a.uv.y) * t);
    out.tangent = make_float4(
        a.tangent.x + (b.tangent.x - a.tangent.x) * t,
        a.tangent.y + (b.tangent.y - a.tangent.y) * t,
        a.tangent.z + (b.tangent.z - a.tangent.z) * t,
        a.tangent.w + (b.tangent.w - a.tangent.w) * t);
    return out;
}

__device__ inline float camera_depth(const float3& pos, const Camera& cam)
{
    return -dot(pos - cam.origin, cam.w);
}

__device__ inline float2 perspective_uv_grad(float u0, float u1, float u2,
                                             float z0, float z1, float z2,
                                             float lambda0, float lambda1, float lambda2,
                                             float signed_area,
                                             float sx0, float sy0,
                                             float sx1, float sy1,
                                             float sx2, float sy2)
{
    float inv_z0 = 1.f / z0;
    float inv_z1 = 1.f / z1;
    float inv_z2 = 1.f / z2;

    float q = lambda0 * inv_z0 + lambda1 * inv_z1 + lambda2 * inv_z2;
    if (fabsf(q) <= 1e-12f || fabsf(signed_area) <= 1e-12f)
        return make_float2(0.f, 0.f);

    float a = lambda0 * (u0 * inv_z0) + lambda1 * (u1 * inv_z1) + lambda2 * (u2 * inv_z2);
    float inv_q2 = 1.f / (q * q);

    float dl0dx = (sy1 - sy2) / signed_area;
    float dl1dx = (sy2 - sy0) / signed_area;
    float dl2dx = (sy0 - sy1) / signed_area;
    float dl0dy = (sx2 - sx1) / signed_area;
    float dl1dy = (sx0 - sx2) / signed_area;
    float dl2dy = (sx1 - sx0) / signed_area;

    float da_dx = dl0dx * (u0 * inv_z0) + dl1dx * (u1 * inv_z1) + dl2dx * (u2 * inv_z2);
    float dq_dx = dl0dx * inv_z0 + dl1dx * inv_z1 + dl2dx * inv_z2;
    float da_dy = dl0dy * (u0 * inv_z0) + dl1dy * (u1 * inv_z1) + dl2dy * (u2 * inv_z2);
    float dq_dy = dl0dy * inv_z0 + dl1dy * inv_z1 + dl2dy * inv_z2;

    return make_float2(
        (da_dx * q - a * dq_dx) * inv_q2,
        (da_dy * q - a * dq_dy) * inv_q2);
}

__device__ inline float4 sample_texture_grad(cudaTextureObject_t tex, float u, float v,
                                             float dudx, float dvdx, float dudy, float dvdy)
{
    return tex2DGrad<float4>(tex, u, v, make_float2(dudx, dvdx), make_float2(dudy, dvdy));
}

__device__ inline bool clip_triangle_near_plane(const Triangle& tri, const Camera& cam,
                                                float near_z, Triangle out_tris[2], int& out_count)
{
    RasterVertex input[4];
    input[0] = { tri.v0, tri.n0, tri.uv0, tri.t0 };
    input[1] = { tri.v1, tri.n1, tri.uv1, tri.t1 };
    input[2] = { tri.v2, tri.n2, tri.uv2, tri.t2 };

    RasterVertex output[4];
    int in_count = 3;
    int out_vertices = 0;

    for (int i = 0; i < in_count; ++i) {
        const RasterVertex& cur = input[i];
        const RasterVertex& nxt = input[(i + 1) % in_count];
        float z_cur = camera_depth(cur.pos, cam);
        float z_nxt = camera_depth(nxt.pos, cam);
        bool cur_inside = (z_cur >= near_z);
        bool nxt_inside = (z_nxt >= near_z);

        if (cur_inside && nxt_inside) {
            output[out_vertices++] = nxt;
        } else if (cur_inside && !nxt_inside) {
            float denom = z_nxt - z_cur;
            float t = (fabsf(denom) > 1e-12f) ? ((near_z - z_cur) / denom) : 0.f;
            output[out_vertices++] = raster_lerp(cur, nxt, t);
        } else if (!cur_inside && nxt_inside) {
            float denom = z_nxt - z_cur;
            float t = (fabsf(denom) > 1e-12f) ? ((near_z - z_cur) / denom) : 0.f;
            output[out_vertices++] = raster_lerp(cur, nxt, t);
            output[out_vertices++] = nxt;
        }
    }

    if (out_vertices < 3) {
        out_count = 0;
        return false;
    }

    auto write_tri = [&](const RasterVertex& a, const RasterVertex& b, const RasterVertex& c,
                         Triangle& dst) {
        dst = tri;
        dst.v0 = a.pos; dst.v1 = b.pos; dst.v2 = c.pos;
        dst.n0 = a.normal; dst.n1 = b.normal; dst.n2 = c.normal;
        dst.uv0 = a.uv; dst.uv1 = b.uv; dst.uv2 = c.uv;
        dst.t0 = a.tangent; dst.t1 = b.tangent; dst.t2 = c.tangent;
    };

    write_tri(output[0], output[1], output[2], out_tris[0]);
    if (out_vertices == 3) {
        out_count = 1;
        return true;
    }

    write_tri(output[0], output[2], output[3], out_tris[1]);
    out_count = 2;
    return true;
}

// ─────────────────────────────────────────────
//  Shared: shade a pixel with color mode support
// ─────────────────────────────────────────────

// Compute final shaded color for a rasterized pixel.
// color_mode: 0=shaders, 1=greyscale, 2=random obj colors, 3=USD flat N·L
__device__ inline float3 raster_shade(
    float3 N, float3 hit_pos, float tu, float tv,
    float dudx, float dvdx, float dudy, float dvdy,
    int obj_id,
    const Camera& cam,
    GpuMaterial* materials, int num_materials,
    cudaTextureObject_t* textures,
    float3* obj_colors, int num_obj_colors,
    int color_mode, int mat_idx,
    float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    float3 base = make_float3(0.8f, 0.8f, 0.8f);
    float met = 0.f, rough = 0.5f;

    // Color mode 1: grey lambert (0.18 = perceptual mid-grey in linear)
    if (color_mode == 1) {
        base = make_float3(0.18f, 0.18f, 0.18f);
        met = 0.f; rough = 0.6f;
    } else if (color_mode == 2 && obj_colors &&
        obj_id >= 0 && obj_id < num_obj_colors) {
        base = obj_colors[obj_id];
    } else if (mat_idx >= 0 && mat_idx < num_materials) {
        const GpuMaterial& m = materials[mat_idx];
        if (m.base_color_tex >= 0 && textures) {
            float4 t = sample_texture_grad(textures[m.base_color_tex], tu, tv, dudx, dvdx, dudy, dvdy);
            float4 bc = m.base_color;
            bc = make_float4(bc.x * t.x, bc.y * t.y, bc.z * t.z, bc.w * t.w);
            base = make_float3(bc.x, bc.y, bc.z);
        } else {
            base = make_float3(m.base_color.x, m.base_color.y, m.base_color.z);
        }
        met = m.metallic;
        rough = m.roughness;
        if (m.metallic_rough_tex >= 0 && textures) {
            float4 mr = sample_texture_grad(textures[m.metallic_rough_tex], tu, tv, dudx, dvdx, dudy, dvdy);
            rough *= mr.y;
            met   *= mr.z;
        }
    }

    float3 c;
    if (color_mode == 3) {
        // USD flat N·L
        float3 L = make_float3(0.4f, 0.7f, -0.5f);
        float nl = fabsf(dot(N, L));
        c = base * (0.3f + 0.7f * nl);
    } else {
        float3 V = normalize(cam.origin - hit_pos);
        c = three_point_shade_view(N, V, key_dir, fill_dir, rim_dir, base, met, rough);
    }

    // Linear → sRGB
    c.x = powf(fmaxf(0.f, c.x), 1.f / 2.2f);
    c.y = powf(fmaxf(0.f, c.y), 1.f / 2.2f);
    c.z = powf(fmaxf(0.f, c.z), 1.f / 2.2f);
    return c;
}

// shade_and_write: z-test + shade + surf2Dwrite for large-tri pass
__device__ inline void shade_and_write(
    const Triangle& tri,
    float w0, float w1, float w2,
    float z0, float z1, float z2,
    float signed_area,
    float sx0, float sy0,
    float sx1, float sy1,
    float sx2, float sy2,
    bool back_face, int px, int py, int w,
    GpuMaterial* materials, int num_materials,
    cudaTextureObject_t* textures,
    float3* obj_colors, int num_obj_colors,
    int color_mode,
    const Camera& cam, int* zbuf_int, cudaSurfaceObject_t surf,
    float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    float inv_z0 = 1.f / z0;
    float inv_z1 = 1.f / z1;
    float inv_z2 = 1.f / z2;
    float inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2;
    if (inv_z <= 1e-12f) return;

    float depth = 1.f / inv_z;
    float pc_w0 = (w0 * inv_z0) * depth;
    float pc_w1 = (w1 * inv_z1) * depth;
    float pc_w2 = (w2 * inv_z2) * depth;

    int z_int = __float_as_int(depth);
    int old   = atomicMin(&zbuf_int[py * w + px], z_int);
    if (z_int >= old) return;

    float3 N = make_float3(
        pc_w0 * tri.n0.x + pc_w1 * tri.n1.x + pc_w2 * tri.n2.x,
        pc_w0 * tri.n0.y + pc_w1 * tri.n1.y + pc_w2 * tri.n2.y,
        pc_w0 * tri.n0.z + pc_w1 * tri.n1.z + pc_w2 * tri.n2.z);
    float nl = sqrtf(N.x*N.x + N.y*N.y + N.z*N.z);
    if (nl > 1e-7f) { N.x /= nl; N.y /= nl; N.z /= nl; }
    float3 hit_pos = make_float3(
        tri.v0.x * pc_w0 + tri.v1.x * pc_w1 + tri.v2.x * pc_w2,
        tri.v0.y * pc_w0 + tri.v1.y * pc_w1 + tri.v2.y * pc_w2,
        tri.v0.z * pc_w0 + tri.v1.z * pc_w1 + tri.v2.z * pc_w2);
    if (dot(N, hit_pos - cam.origin) > 0.f) { N.x = -N.x; N.y = -N.y; N.z = -N.z; }

    float tu = pc_w0 * tri.uv0.x + pc_w1 * tri.uv1.x + pc_w2 * tri.uv2.x;
    float tv = pc_w0 * tri.uv0.y + pc_w1 * tri.uv1.y + pc_w2 * tri.uv2.y;
    float2 du = perspective_uv_grad(tri.uv0.x, tri.uv1.x, tri.uv2.x,
                                    z0, z1, z2, w0, w1, w2, signed_area,
                                    sx0, sy0, sx1, sy1, sx2, sy2);
    float2 dv = perspective_uv_grad(tri.uv0.y, tri.uv1.y, tri.uv2.y,
                                    z0, z1, z2, w0, w1, w2, signed_area,
                                    sx0, sy0, sx1, sy1, sx2, sy2);

    float3 c = raster_shade(N, hit_pos, tu, tv, du.x, dv.x, du.y, dv.y, tri.obj_id, cam,
                            materials, num_materials, textures,
                            obj_colors, num_obj_colors, color_mode, tri.mat_idx,
                            key_dir, fill_dir, rim_dir);

    surf2Dwrite(make_float4(c.x, c.y, c.z, 1.f),
                surf, px * (int)sizeof(float4), py);
}

// ─────────────────────────────────────────────
//  Pass 1: Per-triangle kernel (small triangles)
//  Large triangles are appended to a list for pass 2.
// ─────────────────────────────────────────────

static constexpr int LARGE_TRI_THRESHOLD = 128 * 128;

__device__ inline void rasterize_triangle(
    const Triangle& tri,
    GpuMaterial* materials, int num_materials,
    cudaTextureObject_t* textures,
    float3* obj_colors, int num_obj_colors, int color_mode,
    Camera cam, float inv_vw, float inv_vh,
    int* zbuf_int, cudaSurfaceObject_t surf,
    int w, int h,
    int* large_tri_list, int* large_tri_count, int max_large,
    Triangle* large_clipped_tris, int* large_clipped_count, int max_large_clipped,
    int tri_id_for_large_list,
    float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    float sx0, sy0, z0, sx1, sy1, z1, sx2, sy2, z2;
    if (!project(tri.v0, cam, inv_vw, inv_vh, w, h, sx0, sy0, z0)) return;
    if (!project(tri.v1, cam, inv_vw, inv_vh, w, h, sx1, sy1, z1)) return;
    if (!project(tri.v2, cam, inv_vw, inv_vh, w, h, sx2, sy2, z2)) return;

    float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
    if (area == 0.f) return;
    bool back_face = (area > 0.f);

    int min_x = max(0, (int)floorf(fminf(fminf(sx0, sx1), sx2)));
    int max_x = min(w - 1, (int)ceilf(fmaxf(fmaxf(sx0, sx1), sx2)));
    int min_y = max(0, (int)floorf(fminf(fminf(sy0, sy1), sy2)));
    int max_y = min(h - 1, (int)ceilf(fmaxf(fmaxf(sy0, sy1), sy2)));
    if (min_x > max_x || min_y > max_y) return;

    long long bbox_pixels = (long long)(max_x - min_x + 1) * (max_y - min_y + 1);
    if (bbox_pixels > LARGE_TRI_THRESHOLD) {
        if (tri_id_for_large_list >= 0) {
            int slot = atomicAdd(large_tri_count, 1);
            if (slot < max_large) {
                large_tri_list[slot] = tri_id_for_large_list;
                return;
            }
        } else if (large_clipped_tris && large_clipped_count) {
            int slot = atomicAdd(large_clipped_count, 1);
            if (slot < max_large_clipped) {
                large_clipped_tris[slot] = tri;
                return;
            }
        }
    }

    float inv_area = 1.f / fabsf(area);
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            float x = (float)px + 0.5f;
            float y = (float)py + 0.5f;

            float w0 = (sx1 - x) * (sy2 - y) - (sx2 - x) * (sy1 - y);
            float w1 = (sx2 - x) * (sy0 - y) - (sx0 - x) * (sy2 - y);
            float w2 = (sx0 - x) * (sy1 - y) - (sx1 - x) * (sy0 - y);
            if (back_face) {
                if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
            } else {
                if (w0 > 0.f || w1 > 0.f || w2 > 0.f) continue;
            }

            w0 = fabsf(w0) * inv_area;
            w1 = fabsf(w1) * inv_area;
            w2 = fabsf(w2) * inv_area;

            shade_and_write(tri, w0, w1, w2, z0, z1, z2,
                            area, sx0, sy0, sx1, sy1, sx2, sy2, back_face,
                            px, py, w, materials, num_materials, textures,
                            obj_colors, num_obj_colors, color_mode,
                            cam, zbuf_int, surf,
                            key_dir, fill_dir, rim_dir);
        }
    }
}

__global__ void raster_tri_kernel(
    Triangle* triangles, int num_triangles,
    GpuMaterial* materials, int num_materials,
    cudaTextureObject_t* textures,
    float3* obj_colors, int num_obj_colors, int color_mode,
    Camera cam, float inv_vw, float inv_vh,
    int* zbuf_int, cudaSurfaceObject_t surf,
    int w, int h,
    int* large_tri_list, int* large_tri_count, int max_large,
    Triangle* large_clipped_tris, int* large_clipped_count, int max_large_clipped,
    float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_triangles) return;

    const Triangle& tri = triangles[tid];

    // Near-plane clip
    const float NEAR = 0.01f;
    float z0c = -dot(tri.v0 - cam.origin, cam.w);
    float z1c = -dot(tri.v1 - cam.origin, cam.w);
    float z2c = -dot(tri.v2 - cam.origin, cam.w);
    if (z0c < NEAR || z1c < NEAR || z2c < NEAR) {
        Triangle clipped[2];
        int clipped_count = 0;
        if (!clip_triangle_near_plane(tri, cam, NEAR, clipped, clipped_count))
            return;
        for (int i = 0; i < clipped_count; ++i) {
            rasterize_triangle(clipped[i], materials, num_materials, textures,
                               obj_colors, num_obj_colors, color_mode,
                               cam, inv_vw, inv_vh,
                               zbuf_int, surf, w, h,
                               large_tri_list, large_tri_count, max_large,
                               large_clipped_tris, large_clipped_count, max_large_clipped,
                               -1,
                               key_dir, fill_dir, rim_dir);
        }
        return;
    }

    rasterize_triangle(tri, materials, num_materials, textures,
                       obj_colors, num_obj_colors, color_mode,
                       cam, inv_vw, inv_vh,
                       zbuf_int, surf, w, h,
                       large_tri_list, large_tri_count, max_large,
                       large_clipped_tris, large_clipped_count, max_large_clipped,
                       tid,
                       key_dir, fill_dir, rim_dir);
}

// ─────────────────────────────────────────────
//  Pass 2: Per-pixel kernel for large triangles
//  One thread per screen pixel, iterates over the (small) large-tri list.
// ─────────────────────────────────────────────

__global__ void raster_large_tri_kernel(
    Triangle* triangles,
    Triangle* clipped_triangles,
    GpuMaterial* materials, int num_materials,
    cudaTextureObject_t* textures,
    float3* obj_colors, int num_obj_colors, int color_mode,
    Camera cam, float inv_vw, float inv_vh,
    int* zbuf_int, cudaSurfaceObject_t surf,
    int w, int h,
    int* large_tri_list, int* d_large_count, int max_large,
    int* d_large_clipped_count, int max_large_clipped,
    float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    // Read count from device memory — avoids D2H sync stall
    int num_large = min(*d_large_count, max_large);
    int num_large_clipped = min(*d_large_clipped_count, max_large_clipped);
    if (num_large <= 0 && num_large_clipped <= 0) return;

    float x = (float)px + 0.5f;
    float y = (float)py + 0.5f;

    for (int li = 0; li < num_large; ++li) {
        const Triangle& tri = triangles[large_tri_list[li]];

        // Project vertices
        float sx0, sy0, z0, sx1, sy1, z1, sx2, sy2, z2;
        if (!project(tri.v0, cam, inv_vw, inv_vh, w, h, sx0, sy0, z0)) continue;
        if (!project(tri.v1, cam, inv_vw, inv_vh, w, h, sx1, sy1, z1)) continue;
        if (!project(tri.v2, cam, inv_vw, inv_vh, w, h, sx2, sy2, z2)) continue;

        float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
        if (area == 0.f) continue;
        bool back_face = (area > 0.f);
        float inv_area = 1.f / fabsf(area);

        // Barycentric test
        float w0 = (sx1 - x) * (sy2 - y) - (sx2 - x) * (sy1 - y);
        float w1 = (sx2 - x) * (sy0 - y) - (sx0 - x) * (sy2 - y);
        float w2 = (sx0 - x) * (sy1 - y) - (sx1 - x) * (sy0 - y);
        if (back_face) {
            if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
        } else {
            if (w0 > 0.f || w1 > 0.f || w2 > 0.f) continue;
        }

        w0 = fabsf(w0) * inv_area;
        w1 = fabsf(w1) * inv_area;
        w2 = fabsf(w2) * inv_area;

        shade_and_write(tri, w0, w1, w2, z0, z1, z2,
                        area, sx0, sy0, sx1, sy1, sx2, sy2, back_face,
                        px, py, w, materials, num_materials, textures,
                        obj_colors, num_obj_colors, color_mode,
                        cam, zbuf_int, surf,
                        key_dir, fill_dir, rim_dir);
    }

    for (int li = 0; li < num_large_clipped; ++li) {
        const Triangle& tri = clipped_triangles[li];

        float sx0, sy0, z0, sx1, sy1, z1, sx2, sy2, z2;
        if (!project(tri.v0, cam, inv_vw, inv_vh, w, h, sx0, sy0, z0)) continue;
        if (!project(tri.v1, cam, inv_vw, inv_vh, w, h, sx1, sy1, z1)) continue;
        if (!project(tri.v2, cam, inv_vw, inv_vh, w, h, sx2, sy2, z2)) continue;

        float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
        if (area == 0.f) continue;
        bool back_face = (area > 0.f);
        float inv_area = 1.f / fabsf(area);

        float w0 = (sx1 - x) * (sy2 - y) - (sx2 - x) * (sy1 - y);
        float w1 = (sx2 - x) * (sy0 - y) - (sx0 - x) * (sy2 - y);
        float w2 = (sx0 - x) * (sy1 - y) - (sx1 - x) * (sy0 - y);
        if (back_face) {
            if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
        } else {
            if (w0 > 0.f || w1 > 0.f || w2 > 0.f) continue;
        }

        w0 = fabsf(w0) * inv_area;
        w1 = fabsf(w1) * inv_area;
        w2 = fabsf(w2) * inv_area;

        shade_and_write(tri, w0, w1, w2, z0, z1, z2,
                        area, sx0, sy0, sx1, sy1, sx2, sy2, back_face,
                        px, py, w, materials, num_materials, textures,
                        obj_colors, num_obj_colors, color_mode,
                        cam, zbuf_int, surf,
                        key_dir, fill_dir, rim_dir);
    }
}

// ─────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────

void rasterizer_render(
    RasterizerState& state,
    cudaSurfaceObject_t surface, int width, int height,
    Camera cam,
    Triangle* d_triangles, int num_triangles,
    GpuMaterial* d_materials, int num_materials,
    cudaTextureObject_t* d_textures,
    float3* d_obj_colors, int num_obj_colors,
    int color_mode, float3 bg_color,
    float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    if (width <= 0 || height <= 0) return;
    rasterizer_resize(state, width, height);

    // Clear z-buffer and background (pre-convert bg to sRGB on CPU)
    float3 bg_srgb = make_float3(
        powf(fmaxf(0.f, bg_color.x), 1.f / 2.2f),
        powf(fmaxf(0.f, bg_color.y), 1.f / 2.2f),
        powf(fmaxf(0.f, bg_color.z), 1.f / 2.2f));
    {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        raster_clear_kernel<<<grid, block>>>(state.d_zbuffer, surface,
                                             width, height, bg_srgb);
    }

    if (num_triangles <= 0 || !d_triangles) return;

    // Derive projection constants from Camera
    float3 hor = cam.horizontal;
    float3 ver = cam.vertical;
    float3 diff = cam.origin - cam.lower_left
                  - make_float3(hor.x*0.5f, hor.y*0.5f, hor.z*0.5f)
                  - make_float3(ver.x*0.5f, ver.y*0.5f, ver.z*0.5f);
    float focus_dist = dot(diff, cam.w);

    float hor_len = length(hor);
    float ver_len = length(ver);
    float inv_vw = (hor_len > 1e-7f) ? (focus_dist / hor_len) : 1.f;
    float inv_vh = (ver_len > 1e-7f) ? (focus_dist / ver_len) : 1.f;

    // Reset large-triangle counter (async — no CPU stall)
    cudaMemsetAsync(state.d_large_count, 0, sizeof(int));
    cudaMemsetAsync(state.d_large_clipped_count, 0, sizeof(int));

    // Pass 1: rasterize small triangles, collect large ones
    {
        int block_size = 256;
        int grid_size  = (num_triangles + block_size - 1) / block_size;
        raster_tri_kernel<<<grid_size, block_size>>>(
            d_triangles, num_triangles,
            d_materials, num_materials, d_textures,
            d_obj_colors, num_obj_colors, color_mode,
            cam, inv_vw, inv_vh,
            reinterpret_cast<int*>(state.d_zbuffer), surface,
            width, height,
            state.d_large_tris, state.d_large_count, MAX_LARGE_TRIS,
            state.d_large_clipped_tris, state.d_large_clipped_count, MAX_LARGE_CLIPPED_TRIS,
            key_dir, fill_dir, rim_dir);
    }

    // Pass 2: per-pixel rasterize large triangles.
    // Always launch — kernel reads d_large_count from device and early-exits if zero.
    // This avoids a cudaMemcpy D2H sync stall between Pass 1 and Pass 2.
    {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        raster_large_tri_kernel<<<grid, block>>>(
            d_triangles, state.d_large_clipped_tris,
            d_materials, num_materials, d_textures,
            d_obj_colors, num_obj_colors, color_mode,
            cam, inv_vw, inv_vh,
            reinterpret_cast<int*>(state.d_zbuffer), surface,
            width, height,
            state.d_large_tris, state.d_large_count, MAX_LARGE_TRIS,
            state.d_large_clipped_count, MAX_LARGE_CLIPPED_TRIS,
            key_dir, fill_dir, rim_dir);
    }
}
