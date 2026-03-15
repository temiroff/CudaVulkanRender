#include "pathtracer.h"
#include <cuda_runtime.h>
#include <cstring>

// ─────────────────────────────────────────────
//  PCG hash RNG — zero global memory, seed from coords + frame
// ─────────────────────────────────────────────

__device__ inline unsigned int pcg_hash(unsigned int seed)
{
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Advance seed and return a float in [0, 1)
__device__ inline float rand_f(unsigned int& seed)
{
    seed = pcg_hash(seed);
    return (float)(seed >> 8) * (1.f / (float)(1u << 24));
}

// Seed from pixel index and frame count — each frame is an independent sample
__device__ inline unsigned int make_seed(int idx, int frame)
{
    return pcg_hash((unsigned int)idx ^ pcg_hash((unsigned int)frame));
}

__device__ inline float3 rand_in_unit_sphere(unsigned int& seed) {
    float3 p;
    do { p = make_float3(rand_f(seed)*2.f-1.f, rand_f(seed)*2.f-1.f, rand_f(seed)*2.f-1.f); }
    while (dot(p, p) >= 1.f);
    return p;
}

__device__ inline float3 rand_unit_vector(unsigned int& seed) {
    return normalize(rand_in_unit_sphere(seed));
}

__device__ inline float3 rand_in_unit_disk(unsigned int& seed) {
    float3 p;
    do { p = make_float3(rand_f(seed)*2.f-1.f, rand_f(seed)*2.f-1.f, 0.f); }
    while (dot(p,p) >= 1.f);
    return p;
}

// ─────────────────────────────────────────────
//  Camera ray
// ─────────────────────────────────────────────

__device__ inline Ray camera_ray(const Camera& cam, float u, float v, unsigned int& rng) {
    float3 rd     = cam.lens_radius * rand_in_unit_disk(rng);
    float3 offset = cam.u * rd.x + cam.v * rd.y;
    return {
        cam.origin + offset,
        normalize(cam.lower_left + u*cam.horizontal + v*cam.vertical - cam.origin - offset)
    };
}

// ─────────────────────────────────────────────
//  Ray-origin offset — black terminator fix
//  Offsets p along the geometric normal so the scattered ray never starts
//  below the geometric surface, preventing hard dark lines at grazing angles.
// ─────────────────────────────────────────────

__device__ inline float3 offset_ray_origin(const float3& p, const float3& geom_n, const float3& dir)
{
    // Push p 0.1 mm (1e-4) toward the side the scattered ray points to.
    float s = dot(dir, geom_n) >= 0.f ? 1.f : -1.f;
    return make_float3(p.x + geom_n.x * s * 1e-4f,
                       p.y + geom_n.y * s * 1e-4f,
                       p.z + geom_n.z * s * 1e-4f);
}

// ─────────────────────────────────────────────
//  Scatter — sphere materials
// ─────────────────────────────────────────────

__device__ inline bool scatter_lambertian(
    const Ray& r_in, const HitRecord& rec,
    float3& attenuation, Ray& scattered, unsigned int& rng)
{
    float3 dir = rec.normal + rand_unit_vector(rng);
    if (fabsf(dir.x) < 1e-8f && fabsf(dir.y) < 1e-8f && fabsf(dir.z) < 1e-8f)
        dir = rec.normal;
    float3 d = normalize(dir);
    scattered   = { offset_ray_origin(rec.p, rec.geom_normal, d), d };
    attenuation = rec.mat.albedo;
    return true;
}

__device__ inline bool scatter_metal(
    const Ray& r_in, const HitRecord& rec,
    float3& attenuation, Ray& scattered, unsigned int& rng)
{
    float3 reflected = reflect(normalize(r_in.dir), rec.normal);
    float3 d = normalize(reflected + rec.mat.roughness * rand_in_unit_sphere(rng));
    scattered   = { offset_ray_origin(rec.p, rec.geom_normal, d), d };
    attenuation = rec.mat.albedo;
    return dot(d, rec.normal) > 0.f;
}

__device__ inline float schlick(float cos_theta, float ior) {
    float r0 = (1.f - ior) / (1.f + ior); r0 *= r0;
    return r0 + (1.f - r0) * powf(1.f - cos_theta, 5.f);
}

__device__ inline bool scatter_dielectric(
    const Ray& r_in, const HitRecord& rec,
    float3& attenuation, Ray& scattered, unsigned int& rng)
{
    attenuation = make_float3(1.f, 1.f, 1.f);
    float ratio = rec.front_face ? (1.f / rec.mat.ior) : rec.mat.ior;
    float3 unit_dir = normalize(r_in.dir);
    float cos_theta = fminf(dot(make_float3(-unit_dir.x,-unit_dir.y,-unit_dir.z), rec.normal), 1.f);
    float sin_theta = sqrtf(1.f - cos_theta*cos_theta);
    bool cannot_refract = ratio * sin_theta > 1.f;
    float3 dir;
    if (cannot_refract || schlick(cos_theta, ratio) > rand_f(rng)) {
        dir = reflect(unit_dir, rec.normal);
    } else {
        float3 r_perp  = ratio * (unit_dir + cos_theta * rec.normal);
        float3 r_paral = make_float3(-1.f,-1.f,-1.f) * rec.normal * sqrtf(fabsf(1.f - dot(r_perp,r_perp)));
        dir = normalize(r_perp + r_paral);
    }
    scattered = { offset_ray_origin(rec.p, rec.geom_normal, dir), dir };
    return true;
}

// ─────────────────────────────────────────────
//  Scatter — glTF PBR metallic-roughness
// ─────────────────────────────────────────────

__device__ inline float4 sample_tex(const cudaTextureObject_t* textures, int idx, float2 uv) {
    return tex2D<float4>(textures[idx], uv.x, uv.y);
}
__device__ inline float4 mul4(const float4& a, const float4& b)
{
    return make_float4(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z,
        a.w * b.w);
}
__device__ bool scatter_gpu_material(
    const Ray& r_in, HitRecord& rec,
    const GpuMaterial& mat,
    const cudaTextureObject_t* textures,
    float3& attenuation, float3& emission, Ray& scattered,
    unsigned int& rng)
{
    // Base color
    float4 base = mat.base_color;
    if (mat.base_color_tex >= 0)
        base = mul4(base, sample_tex(textures, mat.base_color_tex, rec.uv));

    // Metallic / roughness
    float metallic  = mat.metallic;
    float roughness = mat.roughness;
    if (mat.metallic_rough_tex >= 0) {
        float4 mr = sample_tex(textures, mat.metallic_rough_tex, rec.uv);
        roughness *= mr.y;   // G channel
        metallic  *= mr.z;   // B channel
    }

    // Emissive
    float4 emis = mat.emissive_factor;
    if (mat.emissive_tex >= 0) {
        float4 et = sample_tex(textures, mat.emissive_tex, rec.uv);
        emis.x *= et.x; emis.y *= et.y; emis.z *= et.z;
    }
    emission = make_float3(emis.x, emis.y, emis.z);

    attenuation = make_float3(base.x, base.y, base.z);

    // Normal map — apply before scatter so the modified normal is used everywhere
    if (mat.normal_tex >= 0 && rec.tangent.w != 0.f) {
        float4 ns = sample_tex(textures, mat.normal_tex, rec.uv);
        // Decode [0,1] → [-1,1], flip Y to match OpenGL convention used by glTF
        float3 nm = make_float3(ns.x * 2.f - 1.f, ns.y * 2.f - 1.f, ns.z * 2.f - 1.f);
        float  nm_len = sqrtf(nm.x*nm.x + nm.y*nm.y + nm.z*nm.z);
        if (nm_len > 1e-7f) { nm.x /= nm_len; nm.y /= nm_len; nm.z /= nm_len; }
        // Reconstruct TBN basis in world space
        float3 T = make_float3(rec.tangent.x, rec.tangent.y, rec.tangent.z);
        float3 N = rec.normal;
        float3 B = cross(N, T) * rec.tangent.w;  // bitangent sign
        // Transform sample into world space
        float3 mapped = make_float3(
            T.x*nm.x + B.x*nm.y + N.x*nm.z,
            T.y*nm.x + B.y*nm.y + N.y*nm.z,
            T.z*nm.x + B.z*nm.y + N.z*nm.z);
        float ml = sqrtf(dot(mapped, mapped));
        if (ml > 1e-7f) rec.normal = mapped / ml;
    }

    // Opacity: if base_color alpha is very low, treat as transparent (passthrough)
    if (base.w < 0.1f) {
        scattered = { offset_ray_origin(rec.p, rec.geom_normal, r_in.dir), r_in.dir };
        attenuation = make_float3(1.f, 1.f, 1.f);
        return true;
    }

    // ── GGX PBR metallic-roughness BSDF ─────────────────────────────────────
    // Implements the Disney/glTF PBR model:
    //   F0     = lerp(0.04, base_color, metallic)   [specular reflectance at 0°]
    //   Fdiff  = base_color * (1-metallic)           [diffuse albedo, 0 for metals]
    //   At each bounce: choose specular or diffuse by comparing F0 lum to rand.

    float3 albedo     = make_float3(base.x, base.y, base.z);
    float  alpha      = roughness * roughness;           // GGX alpha = roughness²
    alpha             = fmaxf(alpha, 0.001f);            // avoid delta singularity

    float3 N = rec.normal;
    float3 V = make_float3(-r_in.dir.x, -r_in.dir.y, -r_in.dir.z);  // view direction
    float  NdotV = fmaxf(dot(N, V), 0.f);

    // F0: dielectric = 0.04, metal = albedo
    float3 F0 = make_float3(
        0.04f + (albedo.x - 0.04f) * metallic,
        0.04f + (albedo.y - 0.04f) * metallic,
        0.04f + (albedo.z - 0.04f) * metallic);

    // Schlick Fresnel at grazing angle (use NdotV as approximation)
    float  one_minus_v = 1.f - NdotV;
    float  one_minus_v5 = one_minus_v * one_minus_v * one_minus_v * one_minus_v * one_minus_v;
    float3 F = make_float3(
        F0.x + (1.f - F0.x) * one_minus_v5,
        F0.y + (1.f - F0.y) * one_minus_v5,
        F0.z + (1.f - F0.z) * one_minus_v5);

    // Probability of choosing specular lobe = luminance(F)
    float F_lum       = 0.2126f * F.x + 0.7152f * F.y + 0.0722f * F.z;
    bool  do_specular = (rand_f(rng) < F_lum);

    if (do_specular) {
        // ── GGX specular ─────────────────────────────────────────────────────
        // Sample GGX half-vector in tangent space (simplified isotropic NDF)
        float r1 = rand_f(rng);
        float r2 = rand_f(rng);
        // GGX importance sampling: theta = atan(alpha * sqrt(r1/(1-r1)))
        float cos_theta2 = (1.f - r1) / (1.f + (alpha * alpha - 1.f) * r1);
        float cos_theta  = sqrtf(fmaxf(cos_theta2, 0.f));
        float sin_theta  = sqrtf(fmaxf(1.f - cos_theta2, 0.f));
        float phi        = 2.f * 3.14159265f * r2;

        // Build local frame around N
        float3 up   = (fabsf(N.z) < 0.999f) ? make_float3(0.f, 0.f, 1.f)
                                              : make_float3(1.f, 0.f, 0.f);
        float3 T    = normalize(cross(up, N));
        float3 B    = cross(N, T);

        // Half-vector in world space
        float3 H = normalize(
            T * (sin_theta * cosf(phi)) +
            B * (sin_theta * sinf(phi)) +
            N * cos_theta);

        float3 L = reflect(r_in.dir, H);   // reflect view around H

        float NdotL = dot(N, L);
        if (NdotL <= 0.f) return false;    // below hemisphere

        // Weight: F / F_lum (cancel the probability; F already accounts for color)
        attenuation = make_float3(F.x / fmaxf(F_lum, 1e-5f),
                                  F.y / fmaxf(F_lum, 1e-5f),
                                  F.z / fmaxf(F_lum, 1e-5f));
        float3 Ln = normalize(L);
        scattered = { offset_ray_origin(rec.p, rec.geom_normal, Ln), Ln };
    } else {
        // ── Lambertian diffuse ───────────────────────────────────────────────
        // Diffuse albedo: base_color * (1-metallic). Metals have zero diffuse.
        float3 diffuse_albedo = make_float3(
            albedo.x * (1.f - metallic),
            albedo.y * (1.f - metallic),
            albedo.z * (1.f - metallic));

        // Weight: diffuse_albedo / (1 - F_lum)  — cancel probability
        float inv_p = 1.f / fmaxf(1.f - F_lum, 1e-5f);
        attenuation = make_float3(
            diffuse_albedo.x * inv_p,
            diffuse_albedo.y * inv_p,
            diffuse_albedo.z * inv_p);

        float3 dir = N + rand_unit_vector(rng);
        if (fabsf(dir.x) < 1e-8f && fabsf(dir.y) < 1e-8f && fabsf(dir.z) < 1e-8f)
            dir = N;
        float3 dn = normalize(dir);
        scattered = { offset_ray_origin(rec.p, rec.geom_normal, dn), dn };
    }
    return true;
}

// ─────────────────────────────────────────────
//  Sky / HDRI sampling
// ─────────────────────────────────────────────

__device__ float3 sample_sky(const PathTracerParams& p, float3 dir)
{
    if (p.hdri_tex != 0) {
        // Equirectangular mapping
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
    // Default gradient sky
    float  t   = 0.5f * (normalize(dir).y + 1.f);
    return lerp(make_float3(1.f, 1.f, 1.f), make_float3(0.5f, 0.7f, 1.f), t);
}

// ─────────────────────────────────────────────
//  Path trace loop
// ─────────────────────────────────────────────

__device__ float3 trace_path(Ray ray, const PathTracerParams& p, unsigned int& rng)
{
    float3 throughput = make_float3(1.f, 1.f, 1.f);
    float3 radiance   = make_float3(0.f, 0.f, 0.f);

    for (int depth = 0; depth < p.max_depth; ++depth) {
        // Test sphere BVH (skip if no spheres)
        HitRecord rec;
        bool any_hit = (p.num_prims > 0) && bvh_hit(p.bvh, p.prims, ray, 1e-3f, 1e20f, rec);
        float t_max  = any_hit ? rec.t : 1e20f;

        // Test triangle BVH (only if mesh is loaded and might be closer)
        if (p.tri_bvh && p.num_triangles > 0) {
            HitRecord tri_rec;
            if (bvh_hit_triangles(p.tri_bvh, p.triangles, ray, 1e-3f, t_max, tri_rec)) {
                rec     = tri_rec;
                any_hit = true;
            }
        }

        if (!any_hit) {
            radiance += throughput * sample_sky(p, ray.dir);
            break;
        }

        Ray    scattered;
        float3 attenuation;
        float3 emission = make_float3(0.f, 0.f, 0.f);
        bool   did_scatter = false;

        if (rec.gpu_mat_idx >= 0 && rec.gpu_mat_idx < p.num_gpu_materials) {
            // Mesh / glTF material
            did_scatter = scatter_gpu_material(
                ray, rec, p.gpu_materials[rec.gpu_mat_idx],
                p.textures, attenuation, emission, scattered, rng);
            // Color mode override (greyscale applied to final radiance below, not here)
            if (p.color_mode == 2 && p.obj_colors &&
                       rec.obj_id >= 0 && rec.obj_id < p.num_obj_colors) {
                // Random per-object color
                attenuation = p.obj_colors[rec.obj_id];
            }
        } else {
            // Sphere inline material
            emission = rec.mat.emission;
            switch (rec.mat.type) {
                case MatType::Lambertian:
                    did_scatter = scatter_lambertian(ray, rec, attenuation, scattered, rng); break;
                case MatType::Metal:
                    did_scatter = scatter_metal(ray, rec, attenuation, scattered, rng); break;
                case MatType::Dielectric:
                    did_scatter = scatter_dielectric(ray, rec, attenuation, scattered, rng); break;
                case MatType::Emissive:
                    did_scatter = false; break;
            }
        }

        radiance += throughput * emission;
        if (!did_scatter) break;
        throughput = throughput * attenuation;
        ray = scattered;
    }
    return radiance;
}

// ─────────────────────────────────────────────
//  Main kernel
// ─────────────────────────────────────────────

__global__ void pathtrace_kernel(PathTracerParams p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    int idx = y * p.width + x;
    // PCG seed: unique per pixel × frame — no global RNG state needed
    unsigned int rng = make_seed(idx, p.frame_count);

    float3 color = make_float3(0.f, 0.f, 0.f);
    for (int s = 0; s < p.spp; ++s) {
        // Vary seed per sample within the same frame
        unsigned int srng = pcg_hash(rng ^ (unsigned int)s);
        float u =        (x + rand_f(srng)) / (float)(p.width  - 1);
        float v = 1.f - (y + rand_f(srng)) / (float)(p.height - 1);
        Ray ray = camera_ray(p.cam, u, v, srng);
        float3 sample = trace_path(ray, p, srng);
        // Firefly suppression: clamp luminance of outlier samples
        if (p.firefly_clamp > 0.f) {
            float lum = 0.2126f * sample.x + 0.7152f * sample.y + 0.0722f * sample.z;
            if (lum > p.firefly_clamp)
                sample = sample * (p.firefly_clamp / lum);
        }
        color += sample;
    }
    color = color / (float)p.spp;

    // Greyscale: convert fully-shaded PBR result to luminance AFTER all bounces,
    // so reflections, normal maps, roughness, and lighting are all captured.
    if (p.color_mode == 1) {
        float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
        color = make_float3(lum, lum, lum);
    }

    float4 prev  = p.accum_buffer[idx];
    float  t     = 1.f / (float)(p.frame_count + 1);
    float3 accum = lerp(make_float3(prev.x, prev.y, prev.z), color, t);
    p.accum_buffer[idx] = make_float4(accum.x, accum.y, accum.z, 1.f);

    // Write raw linear HDR — tone mapping done in Slang post-process pass
    surf2Dwrite(make_float4(accum.x, accum.y, accum.z, 1.f), p.surface, x * (int)sizeof(float4), y);

}

// ─────────────────────────────────────────────
//  Host API
// ─────────────────────────────────────────────

void pathtracer_launch(const PathTracerParams& p, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((p.width  + block.x - 1) / block.x,
              (p.height + block.y - 1) / block.y);
    pathtrace_kernel<<<grid, block, 0, stream>>>(p);
}

float4* pathtracer_alloc_accum(int width, int height) {
    float4* buf;
    cudaMalloc(&buf, (size_t)width * height * sizeof(float4));
    cudaMemset(buf, 0, (size_t)width * height * sizeof(float4));
    return buf;
}

void pathtracer_reset_accum(float4* accum, int width, int height) {
    cudaMemset(accum, 0, (size_t)width * height * sizeof(float4));
}

__global__ void blit_surface_kernel(const float4* src, cudaSurfaceObject_t surf, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float4 px = src[y * w + x];
    surf2Dwrite(px, surf, x * (int)sizeof(float4), y);
}

void pathtracer_blit_surface(const float4* d_src, cudaSurfaceObject_t surface, int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    blit_surface_kernel<<<grid, block>>>(d_src, surface, width, height);
}
