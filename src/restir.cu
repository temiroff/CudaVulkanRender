#include "restir.h"
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <cstring>

// ── PCG hash RNG (same as pathtracer.cu — no global state) ───────────────────

__device__ inline unsigned int pcg_hash(unsigned int seed)
{
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline float rs_rand(unsigned int& s)
{
    s = pcg_hash(s);
    return (float)(s >> 8) * (1.f / (float)(1u << 24));
}

__device__ inline float3 rs_rand_unit(unsigned int& s) {
    float3 p;
    do { p = make_float3(rs_rand(s)*2.f-1.f, rs_rand(s)*2.f-1.f, rs_rand(s)*2.f-1.f); }
    while (dot(p,p) >= 1.f);
    float l = sqrtf(dot(p,p));
    return make_float3(p.x/l, p.y/l, p.z/l);
}

// ── Geometry helpers ─────────────────────────────────────────────────────────

__device__ inline float3 rs_cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ inline float3 rs_normalize(float3 v) {
    float l = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return l < 1e-8f ? make_float3(0.f,1.f,0.f) : make_float3(v.x/l, v.y/l, v.z/l);
}
__device__ inline float rs_dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline float rs_len(float3 v) { return sqrtf(rs_dot(v,v)); }

// Sample a uniform random point on a triangle
__device__ float3 sample_tri(const EmissiveTri& lt, float r1, float r2) {
    float sq = sqrtf(r1);
    float u  = 1.f - sq;
    float v  = r2 * sq;
    return make_float3(
        (1.f-u-v)*lt.v0.x + u*lt.v1.x + v*lt.v2.x,
        (1.f-u-v)*lt.v0.y + u*lt.v1.y + v*lt.v2.y,
        (1.f-u-v)*lt.v0.z + u*lt.v1.z + v*lt.v2.z);
}

// Geometric term: G = cos(θ_surface) * cos(θ_light) / dist²
__device__ float geometry_term(float3 p, float3 n_p, float3 q, float3 n_q) {
    float3 pq  = make_float3(q.x-p.x, q.y-p.y, q.z-p.z);
    float  d2  = fmaxf(1e-6f, rs_dot(pq,pq));
    float3 dir = make_float3(pq.x/sqrtf(d2), pq.y/sqrtf(d2), pq.z/sqrtf(d2));
    float  cos_p = fmaxf(0.f,  rs_dot(n_p, dir));
    float3 nq_neg = make_float3(-n_q.x, -n_q.y, -n_q.z);
    float  cos_q = fmaxf(0.f,  rs_dot(nq_neg, dir));
    return cos_p * cos_q / d2;
}

// p_hat: unshadowed target PDF proportional to contribution
__device__ float eval_phat(float3 p, float3 n, const EmissiveTri& lt, float3 lp, float3 ln) {
    float g = geometry_term(p, n, lp, ln);
    float em = lt.emission.x * 0.2126f + lt.emission.y * 0.7152f + lt.emission.z * 0.0722f;
    return fmaxf(0.f, em * g);
}

__device__ inline float3 tri_normal(const EmissiveTri& lt)
{
    return rs_normalize(rs_cross(
        make_float3(lt.v1.x - lt.v0.x, lt.v1.y - lt.v0.y, lt.v1.z - lt.v0.z),
        make_float3(lt.v2.x - lt.v0.x, lt.v2.y - lt.v0.y, lt.v2.z - lt.v0.z)));
}

__device__ inline float3 tri_centroid(const EmissiveTri& lt)
{
    return make_float3((lt.v0.x + lt.v1.x + lt.v2.x) / 3.f,
                       (lt.v0.y + lt.v1.y + lt.v2.y) / 3.f,
                       (lt.v0.z + lt.v1.z + lt.v2.z) / 3.f);
}

__device__ inline float camera_focus_distance(const Camera& cam)
{
    float3 plane_center = cam.lower_left + cam.horizontal * 0.5f + cam.vertical * 0.5f;
    return fmaxf(1e-6f, rs_dot(cam.origin - plane_center, cam.w));
}

__device__ inline bool project_world_to_pixel(const Camera& cam, float3 world,
                                              int width, int height,
                                              float& out_x, float& out_y, float& out_depth)
{
    float3 rel = world - cam.origin;
    float x_view = rs_dot(rel, cam.u);
    float y_view = rs_dot(rel, cam.v);
    float z_view = -rs_dot(rel, cam.w);
    if (z_view <= 1e-4f) return false;

    float focus_dist = camera_focus_distance(cam);
    float half_w = fmaxf(1e-6f, rs_len(cam.horizontal) * 0.5f);
    float half_h = fmaxf(1e-6f, rs_len(cam.vertical) * 0.5f);
    float plane_scale = focus_dist / z_view;
    float ndc_x = (x_view * plane_scale) / half_w;
    float ndc_y = (y_view * plane_scale) / half_h;

    out_x = (ndc_x * 0.5f + 0.5f) * (float)width;
    out_y = (0.5f - ndc_y * 0.5f) * (float)height;
    out_depth = z_view;
    return true;
}

__device__ inline bool wrs_merge(Reservoir& r, int candidate,
                                 float weight_sum, int candidate_count,
                                 float rng_val)
{
    if (candidate_count <= 0 || weight_sum <= 0.f) return false;
    r.w_sum += weight_sum;
    r.M += candidate_count;
    if (rng_val * r.w_sum < weight_sum) {
        r.y = candidate;
        return true;
    }
    return false;
}

__device__ inline void restir_write_hit_info(ReSTIRHitInfo* info, int idx,
                                             float3 hit_p, float3 hit_n, float depth)
{
    if (!info) return;
    info[idx].world_pos = make_float4(hit_p.x, hit_p.y, hit_p.z, depth);
    info[idx].normal    = make_float4(hit_n.x, hit_n.y, hit_n.z, 1.f);
}

__device__ inline void restir_write_invalid_hit(ReSTIRHitInfo* info, int idx)
{
    if (!info) return;
    info[idx].world_pos = make_float4(0.f, 0.f, 0.f, 0.f);
    info[idx].normal    = make_float4(0.f, 0.f, 0.f, 0.f);
}

__device__ inline bool restir_hit_valid(const ReSTIRHitInfo& info)
{
    return info.normal.w > 0.f && info.world_pos.w > 0.f;
}

__device__ inline bool restir_hit_compatible(const ReSTIRHitInfo& a, const ReSTIRHitInfo& b)
{
    if (!restir_hit_valid(a) || !restir_hit_valid(b)) return false;
    float3 ap = make_float3(a.world_pos.x, a.world_pos.y, a.world_pos.z);
    float3 bp = make_float3(b.world_pos.x, b.world_pos.y, b.world_pos.z);
    float3 an = rs_normalize(make_float3(a.normal.x, a.normal.y, a.normal.z));
    float3 bn = rs_normalize(make_float3(b.normal.x, b.normal.y, b.normal.z));
    float dist = rs_len(ap - bp);
    float max_dist = fmaxf(0.02f, 0.02f * fmaxf(a.world_pos.w, b.world_pos.w));
    return dist <= max_dist && rs_dot(an, bn) >= 0.9f;
}

__host__ __device__ inline bool restir_mode_uses_spatial(const ReSTIRParams& p)
{
    return p.reuse_mode == (int)ReSTIRReuseMode::Spatial ||
           p.reuse_mode == (int)ReSTIRReuseMode::TemporalSpatial;
}

__host__ __device__ inline bool restir_mode_uses_temporal(const ReSTIRParams& p)
{
    return p.reuse_mode == (int)ReSTIRReuseMode::Temporal ||
           p.reuse_mode == (int)ReSTIRReuseMode::TemporalSpatial;
}

__device__ inline float reservoir_reuse_weight(const Reservoir& r, float3 hit_p, float3 hit_n,
                                               const EmissiveTri& lt)
{
    if (r.y < 0 || r.M <= 0 || r.W <= 0.f) return 0.f;
    float3 lp = tri_centroid(lt);
    float3 ln = tri_normal(lt);
    float q = eval_phat(hit_p, hit_n, lt, lp, ln);
    return (q > 1e-10f) ? (r.W * (float)r.M * q) : 0.f;
}

// Weighted Reservoir Sampling update — returns true if sample was selected
__device__ bool wrs_update(Reservoir& r, int candidate, float weight, float rng_val) {
    return wrs_merge(r, candidate, weight, 1, rng_val);
}

// ── Visibility test (shadow ray) ────────────────────────────────────────────

__device__ bool is_visible(const ReSTIRParams& p, float3 origin, float3 target) {
    float3 dir = make_float3(target.x-origin.x, target.y-origin.y, target.z-origin.z);
    float  dist = rs_len(dir);
    if (dist < 1e-4f) return true;
    dir = make_float3(dir.x/dist, dir.y/dist, dir.z/dist);
    Ray shadow_ray;
    shadow_ray.origin = make_float3(origin.x + dir.x*1e-3f,
                                    origin.y + dir.y*1e-3f,
                                    origin.z + dir.z*1e-3f);
    shadow_ray.dir = dir;
    HitRecord rec;
    return !bvh_hit_triangles(p.tri_bvh, p.triangles, shadow_ray, 1e-3f, dist - 1e-3f, rec);
}

// ── Primary ray + RIS kernel ─────────────────────────────────────────────────

__global__ void restir_ris_kernel(ReSTIRParams p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    int idx = y * p.width + x;
    unsigned int rng = pcg_hash((unsigned int)idx ^ pcg_hash((unsigned int)p.frame_count * 3u + 1u));
    Reservoir& res = p.reservoirs[idx];
    res = Reservoir{};

    if (p.num_emissive == 0 || !p.tri_bvh) {
        // No lights — reservoir stays empty, shade kernel will use sky
        restir_write_invalid_hit(p.hit_info, idx);
        return;
    }

    // ── Primary ray ──────────────────────────────────────────────────────────
    float fu = ((float)x + rs_rand(rng)) / (float)p.width;
    float fv = ((float)y + rs_rand(rng)) / (float)p.height;
    Ray ray;
    ray.origin = p.cam.origin;
    ray.dir    = rs_normalize(make_float3(
        p.cam.lower_left.x + fu*p.cam.horizontal.x + fv*p.cam.vertical.x - p.cam.origin.x,
        p.cam.lower_left.y + fu*p.cam.horizontal.y + fv*p.cam.vertical.y - p.cam.origin.y,
        p.cam.lower_left.z + fu*p.cam.horizontal.z + fv*p.cam.vertical.z - p.cam.origin.z));

    HitRecord rec;
    if (!bvh_hit_triangles(p.tri_bvh, p.triangles, ray, 1e-3f, 1e20f, rec)) {
        // Miss — store sentinel y=-2 so shade kernel can draw sky
        res.y = -2;
        restir_write_invalid_hit(p.hit_info, idx);
        return;
    }

    float3 hit_p = rec.p;
    float3 hit_n = rec.normal;
    float hit_depth = rs_len(hit_p - p.cam.origin);
    restir_write_hit_info(p.hit_info, idx, hit_p, hit_n, hit_depth);

    // ── Initial RIS: sample M candidates ─────────────────────────────────────
    float total_area = 0.f;
    for (int i = 0; i < p.num_emissive; ++i) total_area += p.emissive_tris[i].area;

    for (int i = 0; i < p.num_candidates; ++i) {
        // Pick a random emissive triangle proportional to area
        float r_area = rs_rand(rng) * total_area;
        int   li = 0;
        float accum = 0.f;
        for (int j = 0; j < p.num_emissive-1; ++j) {
            accum += p.emissive_tris[j].area;
            if (accum >= r_area) { li = j; break; }
            li = j + 1;
        }
        const EmissiveTri& lt = p.emissive_tris[li];

        // Uniform sample on triangle
        float3 lp = sample_tri(lt, rs_rand(rng), rs_rand(rng));
        float3 ln = rs_normalize(rs_cross(
            make_float3(lt.v1.x-lt.v0.x, lt.v1.y-lt.v0.y, lt.v1.z-lt.v0.z),
            make_float3(lt.v2.x-lt.v0.x, lt.v2.y-lt.v0.y, lt.v2.z-lt.v0.z)));

        // Source PDF: uniform over all emissive area
        float source_pdf = lt.area / total_area / lt.area;  // = 1 / total_area
        float phat       = eval_phat(hit_p, hit_n, lt, lp, ln);
        float weight     = (source_pdf > 1e-10f) ? (phat / source_pdf) : 0.f;

        wrs_update(res, li, weight, rs_rand(rng));
    }

    if (restir_mode_uses_temporal(p) &&
        p.has_prev_frame && p.prev_reservoirs && p.prev_hit_info) {
        float prev_x = 0.f, prev_y = 0.f, prev_depth = 0.f;
        if (project_world_to_pixel(p.prev_cam, hit_p, p.width, p.height,
                                   prev_x, prev_y, prev_depth)) {
            int px = (int)floorf(prev_x);
            int py = (int)floorf(prev_y);
            if (px >= 0 && px < p.width && py >= 0 && py < p.height) {
                int prev_idx = py * p.width + px;
                const ReSTIRHitInfo prev_hit = p.prev_hit_info[prev_idx];
                const ReSTIRHitInfo cur_hit  = p.hit_info[idx];
                const Reservoir& prev_res    = p.prev_reservoirs[prev_idx];
                if (prev_res.y >= 0 &&
                    restir_hit_compatible(cur_hit, prev_hit) &&
                    prev_res.y < p.num_emissive) {
                    const EmissiveTri& lt = p.emissive_tris[prev_res.y];
                    float reuse_weight = reservoir_reuse_weight(prev_res, hit_p, hit_n, lt);
                    wrs_merge(res, prev_res.y, reuse_weight, prev_res.M, rs_rand(rng));
                }
            }
        }
    }

    // Compute unbiased W
    if (res.y >= 0 && res.M > 0) {
        const EmissiveTri& lt = p.emissive_tris[res.y];
        float3 lp = tri_centroid(lt);
        float3 ln = tri_normal(lt);
        float phat = eval_phat(hit_p, hit_n, lt, lp, ln);
        res.W = (phat > 1e-10f) ? (res.w_sum / ((float)res.M * phat)) : 0.f;
    }
}

// ── Spatial reuse kernel ──────────────────────────────────────────────────────

__global__ void restir_spatial_kernel(ReSTIRParams p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    int idx = y * p.width + x;
    // PCG seed: mix pixel index, frame count, and pass ID for independent streams
    unsigned int rng = pcg_hash((unsigned int)idx ^ pcg_hash((unsigned int)p.frame_count * 3u + 0u));

    Reservoir r = p.reservoirs[idx];
    if (r.y == -2) { p.reservoirs_tmp[idx] = r; return; } // sky hit, skip
    if (!p.hit_info) { p.reservoirs_tmp[idx] = r; return; }

    const ReSTIRHitInfo cur_hit = p.hit_info[idx];
    if (!restir_hit_valid(cur_hit)) { p.reservoirs_tmp[idx] = r; return; }
    float3 hit_p = make_float3(cur_hit.world_pos.x, cur_hit.world_pos.y, cur_hit.world_pos.z);
    float3 hit_n = rs_normalize(make_float3(cur_hit.normal.x, cur_hit.normal.y, cur_hit.normal.z));

    // Combine with k=4 random spatial neighbors
    const int K = 4;
    for (int k = 0; k < K; ++k) {
        float angle  = rs_rand(rng) * 6.28318f;
        float radius = rs_rand(rng) * (float)p.spatial_radius;
        int nx = x + (int)(cosf(angle) * radius);
        int ny = y + (int)(sinf(angle) * radius);
        if (nx < 0 || nx >= p.width || ny < 0 || ny >= p.height) continue;

        const Reservoir& nb = p.reservoirs[ny * p.width + nx];
        if (nb.y < 0 || nb.M == 0) continue;
        if (nb.y >= p.num_emissive) continue;

        const ReSTIRHitInfo nb_hit = p.hit_info[ny * p.width + nx];
        if (!restir_hit_compatible(cur_hit, nb_hit)) continue;

        const EmissiveTri& lt = p.emissive_tris[nb.y];
        float weight = reservoir_reuse_weight(nb, hit_p, hit_n, lt);
        wrs_merge(r, nb.y, weight, nb.M, rs_rand(rng));
    }

    // Recompute W against the current shading point after neighbor reuse.
    if (r.y >= 0 && r.M > 0 && r.y < p.num_emissive) {
        const EmissiveTri& lt = p.emissive_tris[r.y];
        float phat = eval_phat(hit_p, hit_n, lt, tri_centroid(lt), tri_normal(lt));
        r.W = (phat > 1e-10f) ? (r.w_sum / ((float)r.M * phat)) : 0.f;
    }
    p.reservoirs_tmp[idx] = r;
}

// ── Shade helper ─────────────────────────────────────────────────────────────

__device__ float3 sky_color(const ReSTIRParams& p, float3 dir) {
    if (p.hdri_tex != 0) {
        float phi   = atan2f(dir.x, -dir.z) + p.hdri_yaw;
        float theta = acosf(fmaxf(-1.f, fminf(1.f, dir.y)));
        float u     = phi / (2.f * 3.14159265f);
        if (u < 0.f) u += 1.f; if (u > 1.f) u -= 1.f;
        float v     = theta / 3.14159265f;
        float4 c    = tex2D<float4>(p.hdri_tex, u, v);
        return make_float3(c.x * p.hdri_intensity,
                           c.y * p.hdri_intensity,
                           c.z * p.hdri_intensity);
    }
    float t = 0.5f * (rs_normalize(dir).y + 1.f);
    return make_float3(
        1.f*(1.f-t) + 0.5f*t,
        1.f*(1.f-t) + 0.7f*t,
        1.f*(1.f-t) + 1.0f*t);
}

__device__ float3 shade_pixel(const ReSTIRParams& p, int x, int y,
                               const Reservoir& res, unsigned int& rng)
{
    float fu = ((float)x + 0.5f) / (float)p.width;
    float fv = ((float)y + 0.5f) / (float)p.height;
    Ray ray;
    ray.origin = p.cam.origin;
    ray.dir    = rs_normalize(make_float3(
        p.cam.lower_left.x + fu*p.cam.horizontal.x + fv*p.cam.vertical.x - p.cam.origin.x,
        p.cam.lower_left.y + fu*p.cam.horizontal.y + fv*p.cam.vertical.y - p.cam.origin.y,
        p.cam.lower_left.z + fu*p.cam.horizontal.z + fv*p.cam.vertical.z - p.cam.origin.z));

    // Sky case (flagged by ris kernel)
    if (res.y == -2) return sky_color(p, ray.dir);

    HitRecord rec;
    if (!bvh_hit_triangles(p.tri_bvh, p.triangles, ray, 1e-3f, 1e20f, rec))
        return sky_color(p, ray.dir);

    float3 hit_p = rec.p;
    float3 hit_n = rec.normal;

    // Base color + emissive from material
    float3 base_col = make_float3(0.8f, 0.8f, 0.8f);
    float3 emis_col = make_float3(0.f, 0.f, 0.f);
    if (rec.gpu_mat_idx >= 0 && rec.gpu_mat_idx < p.num_gpu_materials) {
        const GpuMaterial& mat = p.gpu_materials[rec.gpu_mat_idx];
        float4 bc = mat.base_color;
        if (mat.base_color_tex >= 0) {
            float4 tx = tex2D<float4>(p.textures[mat.base_color_tex], rec.uv.x, rec.uv.y);
            bc.x *= tx.x; bc.y *= tx.y; bc.z *= tx.z;
        }
        base_col = make_float3(bc.x, bc.y, bc.z);
        emis_col = make_float3(mat.emissive_factor.x, mat.emissive_factor.y, mat.emissive_factor.z);
    }

    float3 radiance = emis_col;  // surface self-emission

    // ── Direct illumination via ReSTIR reservoir ──────────────────────────
    if (res.y >= 0 && res.W > 0.f && p.num_emissive > 0) {
        const EmissiveTri& lt = p.emissive_tris[res.y];
        float3 lp = sample_tri(lt, rs_rand(rng), rs_rand(rng));
        float3 ln = rs_normalize(rs_cross(
            make_float3(lt.v1.x-lt.v0.x, lt.v1.y-lt.v0.y, lt.v1.z-lt.v0.z),
            make_float3(lt.v2.x-lt.v0.x, lt.v2.y-lt.v0.y, lt.v2.z-lt.v0.z)));

        if (is_visible(p, hit_p, lp)) {
            float3 to_l  = make_float3(lp.x-hit_p.x, lp.y-hit_p.y, lp.z-hit_p.z);
            float  dist  = rs_len(to_l);
            float3 l_dir = make_float3(to_l.x/dist, to_l.y/dist, to_l.z/dist);

            float cos_s = fmaxf(0.f, rs_dot(hit_n, l_dir));
            float cos_l = fmaxf(0.f, rs_dot(ln, make_float3(-l_dir.x,-l_dir.y,-l_dir.z)));
            float brdf  = cos_s / 3.14159265f;
            float g     = cos_l / fmaxf(1e-6f, dist*dist);
            radiance.x += base_col.x * lt.emission.x * brdf * g * res.W * lt.area;
            radiance.y += base_col.y * lt.emission.y * brdf * g * res.W * lt.area;
            radiance.z += base_col.z * lt.emission.z * brdf * g * res.W * lt.area;
        }
    }

    // ── One indirect bounce ───────────────────────────────────────────────
    if (p.max_depth > 1) {
        float3 bdir = rs_normalize(make_float3(
            hit_n.x + rs_rand_unit(rng).x,
            hit_n.y + rs_rand_unit(rng).y,
            hit_n.z + rs_rand_unit(rng).z));
        Ray bounce; bounce.origin = make_float3(
            hit_p.x + hit_n.x*1e-3f,
            hit_p.y + hit_n.y*1e-3f,
            hit_p.z + hit_n.z*1e-3f);
        bounce.dir = bdir;
        float cos_theta = fmaxf(0.f, rs_dot(hit_n, bdir));

        HitRecord b_rec;
        float3 indirect;
        if (bvh_hit_triangles(p.tri_bvh, p.triangles, bounce, 1e-3f, 1e20f, b_rec)) {
            indirect = make_float3(0.f,0.f,0.f);
            if (b_rec.gpu_mat_idx >= 0 && b_rec.gpu_mat_idx < p.num_gpu_materials) {
                float4 ef = p.gpu_materials[b_rec.gpu_mat_idx].emissive_factor;
                indirect  = make_float3(ef.x, ef.y, ef.z);
            }
        } else {
            indirect = sky_color(p, bdir);
        }
        radiance.x += base_col.x * indirect.x * cos_theta;
        radiance.y += base_col.y * indirect.y * cos_theta;
        radiance.z += base_col.z * indirect.z * cos_theta;
    }

    return radiance;
}

// ── Shade kernel ─────────────────────────────────────────────────────────────

__global__ void restir_shade_kernel(ReSTIRParams p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    int idx = y * p.width + x;
    unsigned int rng = pcg_hash((unsigned int)idx ^ pcg_hash((unsigned int)p.frame_count * 3u + 2u));
    const Reservoir& res = p.reservoirs[idx];

    float3 radiance = shade_pixel(p, x, y, res, rng);

    // Progressive accumulation
    int    pix_idx = (p.height - 1 - y) * p.width + x;
    float4 prev    = p.accum_buffer[pix_idx];
    float  fc      = (float)(p.frame_count);
    float4 accum   = make_float4(
        (prev.x * fc + radiance.x) / (fc + 1.f),
        (prev.y * fc + radiance.y) / (fc + 1.f),
        (prev.z * fc + radiance.z) / (fc + 1.f),
        1.f);
    p.accum_buffer[pix_idx] = accum;

    // Write raw linear HDR — tone mapping done in Slang post-process pass
    surf2Dwrite(make_float4(accum.x, accum.y, accum.z, 1.f),
                p.surface, x * (int)sizeof(float4), y, cudaBoundaryModeClamp);
}

// ── Host launch ──────────────────────────────────────────────────────────────

void restir_launch(const ReSTIRParams& p, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((p.width + 15) / 16, (p.height + 15) / 16);

    restir_ris_kernel<<<grid, block, 0, stream>>>(p);

    if (restir_mode_uses_spatial(p) && p.reservoirs_tmp) {
        restir_spatial_kernel<<<grid, block, 0, stream>>>(p);
        // Swap tmp into primary so shade reads the spatially-reused reservoirs
        ReSTIRParams p2 = p;
        p2.reservoirs = p.reservoirs_tmp;  // shade reads tmp
        restir_shade_kernel<<<grid, block, 0, stream>>>(p2);
    } else {
        restir_shade_kernel<<<grid, block, 0, stream>>>(p);
    }
}

// ── Host: alloc reservoirs ───────────────────────────────────────────────────

Reservoir* restir_alloc_reservoirs(int width, int height) {
    Reservoir* ptr = nullptr;
    cudaMalloc(&ptr, (size_t)width * height * sizeof(Reservoir));
    cudaMemset(ptr, 0, (size_t)width * height * sizeof(Reservoir));
    return ptr;
}

// ── Host: extract emissive triangles ────────────────────────────────────────

std::vector<EmissiveTri> restir_extract_emissive(
    const std::vector<Triangle>&    tris,
    const std::vector<GpuMaterial>& mats)
{
    std::vector<EmissiveTri> out;
    for (const auto& tri : tris) {
        if (tri.mat_idx < 0 || tri.mat_idx >= (int)mats.size()) continue;
        const GpuMaterial& mat = mats[tri.mat_idx];
        float em = mat.emissive_factor.x + mat.emissive_factor.y + mat.emissive_factor.z;
        if (em < 0.01f) continue;
        EmissiveTri et;
        et.v0       = tri.v0;
        et.v1       = tri.v1;
        et.v2       = tri.v2;
        et.emission = make_float3(mat.emissive_factor.x, mat.emissive_factor.y, mat.emissive_factor.z);
        // Area via cross product
        float3 e1   = make_float3(tri.v1.x-tri.v0.x, tri.v1.y-tri.v0.y, tri.v1.z-tri.v0.z);
        float3 e2   = make_float3(tri.v2.x-tri.v0.x, tri.v2.y-tri.v0.y, tri.v2.z-tri.v0.z);
        float3 cr   = make_float3(e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z, e1.x*e2.y-e1.y*e2.x);
        et.area     = 0.5f * sqrtf(cr.x*cr.x + cr.y*cr.y + cr.z*cr.z);
        if (et.area < 1e-8f) continue;
        out.push_back(et);
    }
    return out;
}
