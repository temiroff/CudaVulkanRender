#include "rasterizer.h"
#include "shading_common.cuh"
#include <cfloat>

// ─────────────────────────────────────────────
//  State management
// ─────────────────────────────────────────────

// Max large triangles per frame (screen-filling tris — rarely more than a dozen)
static constexpr int MAX_LARGE_TRIS = 512;

void rasterizer_init(RasterizerState& state)
{
    state.d_zbuffer    = nullptr;
    state.d_large_tris = nullptr;
    state.d_large_count = nullptr;
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
}

void rasterizer_destroy(RasterizerState& state)
{
    cudaFree(state.d_zbuffer);
    cudaFree(state.d_large_tris);
    cudaFree(state.d_large_count);
    state.d_zbuffer     = nullptr;
    state.d_large_tris  = nullptr;
    state.d_large_count = nullptr;
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

// ─────────────────────────────────────────────
//  Shared: shade a pixel with color mode support
// ─────────────────────────────────────────────

// Compute final shaded color for a rasterized pixel.
// color_mode: 0=shaders, 1=greyscale, 2=random obj colors, 3=USD flat N·L
__device__ inline float3 raster_shade(
    float3 N, float tu, float tv, int obj_id,
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
        base = sample_base_color(m, textures, tu, tv);
        met = m.metallic;
        rough = m.roughness;
        if (m.metallic_rough_tex >= 0 && textures) {
            float4 mr = tex2D<float4>(textures[m.metallic_rough_tex], tu, tv);
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
        c = three_point_shade(N, cam.w, key_dir, fill_dir, rim_dir, base, met, rough);
    }

    // Linear → sRGB
    c.x = powf(fmaxf(0.f, c.x), 1.f / 2.2f);
    c.y = powf(fmaxf(0.f, c.y), 1.f / 2.2f);
    c.z = powf(fmaxf(0.f, c.z), 1.f / 2.2f);
    return c;
}

// shade_and_write: z-test + shade + surf2Dwrite for large-tri pass
__device__ inline void shade_and_write(
    const Triangle& tri, float w0, float w1, float w2, float depth,
    bool back_face, int px, int py, int w,
    GpuMaterial* materials, int num_materials,
    cudaTextureObject_t* textures,
    float3* obj_colors, int num_obj_colors,
    int color_mode,
    const Camera& cam, int* zbuf_int, cudaSurfaceObject_t surf,
    float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    int z_int = __float_as_int(depth);
    int old   = atomicMin(&zbuf_int[py * w + px], z_int);
    if (z_int >= old) return;

    float3 N = make_float3(
        w0 * tri.n0.x + w1 * tri.n1.x + w2 * tri.n2.x,
        w0 * tri.n0.y + w1 * tri.n1.y + w2 * tri.n2.y,
        w0 * tri.n0.z + w1 * tri.n1.z + w2 * tri.n2.z);
    float nl = sqrtf(N.x*N.x + N.y*N.y + N.z*N.z);
    if (nl > 1e-7f) { N.x /= nl; N.y /= nl; N.z /= nl; }
    float3 hit_pos = make_float3(
        tri.v0.x * w0 + tri.v1.x * w1 + tri.v2.x * w2,
        tri.v0.y * w0 + tri.v1.y * w1 + tri.v2.y * w2,
        tri.v0.z * w0 + tri.v1.z * w1 + tri.v2.z * w2);
    if (dot(N, hit_pos - cam.origin) > 0.f) { N.x = -N.x; N.y = -N.y; N.z = -N.z; }

    float tu = w0 * tri.uv0.x + w1 * tri.uv1.x + w2 * tri.uv2.x;
    float tv = w0 * tri.uv0.y + w1 * tri.uv1.y + w2 * tri.uv2.y;

    float3 c = raster_shade(N, tu, tv, tri.obj_id, cam,
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

__global__ void raster_tri_kernel(
    Triangle* triangles, int num_triangles,
    GpuMaterial* materials, int num_materials,
    cudaTextureObject_t* textures,
    float3* obj_colors, int num_obj_colors, int color_mode,
    Camera cam, float inv_vw, float inv_vh,
    int* zbuf_int, cudaSurfaceObject_t surf,
    int w, int h,
    int* large_tri_list, int* large_tri_count, int max_large,
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
    if (z0c < NEAR || z1c < NEAR || z2c < NEAR) return;

    // Project vertices
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

    // Large triangle? Defer to pass 2.
    long long bbox_pixels = (long long)(max_x - min_x + 1) * (max_y - min_y + 1);
    if (bbox_pixels > LARGE_TRI_THRESHOLD) {
        int slot = atomicAdd(large_tri_count, 1);
        if (slot < max_large)
            large_tri_list[slot] = tid;
        return;
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

            float depth = w0 * z0 + w1 * z1 + w2 * z2;

            shade_and_write(tri, w0, w1, w2, depth, back_face,
                            px, py, w, materials, num_materials, textures,
                            obj_colors, num_obj_colors, color_mode,
                            cam, zbuf_int, surf,
                            key_dir, fill_dir, rim_dir);
        }
    }
}

// ─────────────────────────────────────────────
//  Pass 2: Per-pixel kernel for large triangles
//  One thread per screen pixel, iterates over the (small) large-tri list.
// ─────────────────────────────────────────────

__global__ void raster_large_tri_kernel(
    Triangle* triangles,
    GpuMaterial* materials, int num_materials,
    cudaTextureObject_t* textures,
    float3* obj_colors, int num_obj_colors, int color_mode,
    Camera cam, float inv_vw, float inv_vh,
    int* zbuf_int, cudaSurfaceObject_t surf,
    int w, int h,
    int* large_tri_list, int* d_large_count, int max_large,
    float3 key_dir, float3 fill_dir, float3 rim_dir)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    // Read count from device memory — avoids D2H sync stall
    int num_large = min(*d_large_count, max_large);
    if (num_large <= 0) return;

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

        float depth = w0 * z0 + w1 * z1 + w2 * z2;

        shade_and_write(tri, w0, w1, w2, depth, back_face,
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
            key_dir, fill_dir, rim_dir);
    }

    // Pass 2: per-pixel rasterize large triangles.
    // Always launch — kernel reads d_large_count from device and early-exits if zero.
    // This avoids a cudaMemcpy D2H sync stall between Pass 1 and Pass 2.
    {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        raster_large_tri_kernel<<<grid, block>>>(
            d_triangles,
            d_materials, num_materials, d_textures,
            d_obj_colors, num_obj_colors, color_mode,
            cam, inv_vw, inv_vh,
            reinterpret_cast<int*>(state.d_zbuffer), surface,
            width, height,
            state.d_large_tris, state.d_large_count, MAX_LARGE_TRIS,
            key_dir, fill_dir, rim_dir);
    }
}
