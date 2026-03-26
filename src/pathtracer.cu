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

__device__ inline Ray camera_ray_center(const Camera& cam, float u, float v)
{
    return {
        cam.origin,
        normalize(cam.lower_left + u * cam.horizontal + v * cam.vertical - cam.origin)
    };
}

__device__ inline float camera_focus_distance(const Camera& cam)
{
    float3 plane_center = cam.lower_left + cam.horizontal * 0.5f + cam.vertical * 0.5f;
    return fmaxf(1e-6f, dot(cam.origin - plane_center, cam.w));
}

__device__ inline float linear_depth_to_device_depth(float z_view, float z_near, float z_far)
{
    z_view = fmaxf(z_view, fmaxf(1e-6f, z_near));
    z_near = fmaxf(1e-6f, z_near);
    z_far = fmaxf(z_near + 1e-3f, z_far);
    float a = z_far / (z_far - z_near);
    float b = -(z_near * z_far) / (z_far - z_near);
    return fminf(0.99999f, fmaxf(0.f, a + b / z_view));
}

__device__ inline bool project_world_to_pixel(const Camera& cam, float3 world,
                                              int width, int height,
                                              float& out_x, float& out_y, float& out_depth)
{
    float3 rel = world - cam.origin;
    float x_view = dot(rel, cam.u);
    float y_view = dot(rel, cam.v);
    float z_view = -dot(rel, cam.w);
    if (z_view <= 1e-4f) return false;

    float focus_dist = camera_focus_distance(cam);
    float half_w = fmaxf(1e-6f, length(cam.horizontal) * 0.5f);
    float half_h = fmaxf(1e-6f, length(cam.vertical) * 0.5f);
    float plane_scale = focus_dist / z_view;
    float ndc_x = (x_view * plane_scale) / half_w;
    float ndc_y = (y_view * plane_scale) / half_h;

    out_x = (ndc_x * 0.5f + 0.5f) * (float)width;
    out_y = (0.5f - ndc_y * 0.5f) * (float)height;
    out_depth = z_view;
    return true;
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
    attenuation = rec.albedo;
    return true;
}

__device__ inline bool scatter_metal(
    const Ray& r_in, const HitRecord& rec,
    float3& attenuation, Ray& scattered, unsigned int& rng)
{
    float3 reflected = reflect(normalize(r_in.dir), rec.normal);
    float3 d = normalize(reflected + rec.roughness * rand_in_unit_sphere(rng));
    scattered   = { offset_ray_origin(rec.p, rec.geom_normal, d), d };
    attenuation = rec.albedo;
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
    float ratio = rec.front_face ? (1.f / rec.ior) : rec.ior;
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
    if (mat.base_color_tex >= 0) {
        float4 tex = sample_tex(textures, mat.base_color_tex, make_float2(rec.u, rec.v));
        base = mat.custom_shader ? tex : mul4(base, tex);
    }

    // Metallic / roughness
    float metallic  = mat.metallic;
    float roughness = mat.roughness;
    if (mat.metallic_rough_tex >= 0) {
        float4 mr = sample_tex(textures, mat.metallic_rough_tex, make_float2(rec.u, rec.v));
        if (mat.custom_shader) {
            roughness = mr.y;   // G channel — direct from shader
            metallic  = mr.z;   // B channel — direct from shader
        } else {
            roughness *= mr.y;
            metallic  *= mr.z;
        }
    }

    // Emissive
    float4 emis = mat.emissive_factor;
    if (mat.emissive_tex >= 0) {
        float4 et = sample_tex(textures, mat.emissive_tex, make_float2(rec.u, rec.v));
        if (mat.custom_shader) {
            emis.x = et.x; emis.y = et.y; emis.z = et.z;
        } else {
            emis.x *= et.x; emis.y *= et.y; emis.z *= et.z;
        }
    }
    emission = make_float3(emis.x, emis.y, emis.z);

    attenuation = make_float3(base.x, base.y, base.z);

    // Normal map — apply before scatter so the modified normal is used everywhere
    if (mat.normal_tex >= 0 && rec.tangent.w != 0.f) {
        float4 ns = sample_tex(textures, mat.normal_tex, make_float2(rec.u, rec.v));
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
        // Sequential traversal into one HitRecord — no second copy needed.
        // Sphere BVH runs first; if it hits, t_max shrinks so the triangle BVH
        // only overwrites rec when it finds something strictly closer.
        HitRecord rec;
        bool any_hit = false;
        float t_max  = 1e20f;

        if (p.num_prims > 0 && bvh_hit(p.bvh, p.prims, ray, 1e-3f, t_max, rec)) {
            any_hit = true;
            t_max   = rec.t;    // shrink — triangles must beat this
        }
        if (p.tri_bvh && p.num_triangles > 0 &&
                bvh_hit_triangles(p.tri_bvh, p.triangles, ray, 1e-3f, t_max, rec)) {
            any_hit = true;     // rec already holds the closer triangle hit
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
            emission = rec.emission;
            switch ((MatType)rec.mat_type) {
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

// Hint to NVCC: 256 threads per block. Keeps scheduling assumptions correct.
__launch_bounds__(256)
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

    // Write raw linear HDR — tone mapping done in Slang post-process pass.
    // Skip when an upscale pass will write the full surface itself (avoids partial-surface flicker).
    if (!p.skip_surface_write)
        surf2Dwrite(make_float4(accum.x, accum.y, accum.z, 1.f), p.surface, x * (int)sizeof(float4), y);

}

__global__ void dlss_aux_kernel(DlssAuxParams p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    float u = ((float)x + 0.5f) / (float)max(1, p.width);
    float v = 1.f - ((float)y + 0.5f) / (float)max(1, p.height);
    Ray ray = camera_ray_center(p.cam, u, v);

    HitRecord rec;
    bool any_hit = (p.num_prims > 0) && bvh_hit(p.bvh, p.prims, ray, 1e-3f, 1e20f, rec);
    float t_max = any_hit ? rec.t : 1e20f;
    if (p.tri_bvh && p.num_triangles > 0) {
        HitRecord tri_rec;
        if (bvh_hit_triangles(p.tri_bvh, p.triangles, ray, 1e-3f, t_max, tri_rec)) {
            rec = tri_rec;
            any_hit = true;
        }
    }

    float depth_value = 1.f;  // 1.0 = background sentinel
    float2 motion_value = make_float2(0.f, 0.f);
    if (any_hit) {
        float world_x = 0.f, world_y = 0.f, world_depth = 0.f;
        if (project_world_to_pixel(p.cam, rec.p, p.width, p.height, world_x, world_y, world_depth)) {
            // Log-scale depth: spreads near objects across the full 0..1 range.
            // Linear depth/far compresses everything into the first 1-5% for typical
            // scenes (far=1000, objects at 2-50 units), giving no visual differentiation.
            float far = fmaxf(p.camera_near + 1e-3f, p.camera_far);
            depth_value = fminf(0.9999f, fmaxf(0.f, log1pf(world_depth) / log1pf(far)));

            if (p.has_prev_camera) {
                float prev_x = 0.f, prev_y = 0.f, prev_depth = 0.f;
                if (project_world_to_pixel(p.prev_cam, rec.p, p.width, p.height, prev_x, prev_y, prev_depth))
                    motion_value = make_float2(prev_x - ((float)x + 0.5f),
                                               prev_y - ((float)y + 0.5f));
            }
        }
    }

    int idx = y * p.width + x;
    if (p.depth_buffer) p.depth_buffer[idx] = depth_value;
    if (p.motion_buffer) p.motion_buffer[idx] = motion_value;
    if (p.depth_surface)
        surf2Dwrite(depth_value, p.depth_surface, x * (int)sizeof(float), y);
    if (p.motion_surface)
        surf2Dwrite(motion_value, p.motion_surface, x * (int)sizeof(float2), y);
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

void pathtracer_write_dlss_aux(const DlssAuxParams& p, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((p.width + block.x - 1) / block.x,
              (p.height + block.y - 1) / block.y);
    dlss_aux_kernel<<<grid, block, 0, stream>>>(p);
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

__device__ inline float3 tonemap_for_viewport(float3 hdr, float exposure, int mode)
{
    float exp_scale = exp2f(exposure);
    hdr = make_float3(hdr.x * exp_scale, hdr.y * exp_scale, hdr.z * exp_scale);
    hdr.x = fmaxf(hdr.x, 0.f);
    hdr.y = fmaxf(hdr.y, 0.f);
    hdr.z = fmaxf(hdr.z, 0.f);

    float3 out = hdr;
    switch (mode) {
        case 0:
            out.x = fminf(hdr.x, 1.f);
            out.y = fminf(hdr.y, 1.f);
            out.z = fminf(hdr.z, 1.f);
            break;
        case 1:
            out.x = hdr.x / (1.f + hdr.x);
            out.y = hdr.y / (1.f + hdr.y);
            out.z = hdr.z / (1.f + hdr.z);
            break;
        case 2:
        case 3:
        case 4:
        case 5:
        default: {
            auto aces = [](float x) {
                float y = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
                return fminf(fmaxf(y, 0.f), 1.f);
            };
            out.x = aces(hdr.x);
            out.y = aces(hdr.y);
            out.z = aces(hdr.z);
            break;
        }
    }
    return out;
}

__global__ void blit_surface_tonemap_kernel(const float4* src, cudaSurfaceObject_t surf, int w, int h,
                                            float exposure, int tone_map_mode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float4 px = src[y * w + x];
    float3 tm = tonemap_for_viewport(make_float3(px.x, px.y, px.z), exposure, tone_map_mode);
    surf2Dwrite(make_float4(tm.x, tm.y, tm.z, 1.f), surf, x * (int)sizeof(float4), y);
}

void pathtracer_blit_surface_tonemap(const float4* d_src, cudaSurfaceObject_t surface, int width, int height,
                                     float exposure, int tone_map_mode)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    blit_surface_tonemap_kernel<<<grid, block>>>(d_src, surface, width, height, exposure, tone_map_mode);
}

// ── Bicubic (Mitchell-Netravali B=1/3 C=1/3) weight for one axis ─────────────
__device__ float mitchell(float x)
{
    const float B = 1.f / 3.f, C = 1.f / 3.f;
    x = fabsf(x);
    if (x < 1.f)
        return ((12.f - 9.f*B - 6.f*C)*x*x*x
              + (-18.f + 12.f*B + 6.f*C)*x*x
              + (6.f - 2.f*B)) / 6.f;
    if (x < 2.f)
        return ((-B - 6.f*C)*x*x*x
              + (6.f*B + 30.f*C)*x*x
              + (-12.f*B - 48.f*C)*x
              + (8.f*B + 24.f*C)) / 6.f;
    return 0.f;
}

__device__ float4 accum_fetch(const float4* a, int x, int y, int w, int h)
{
    x = max(0, min(w - 1, x));
    y = max(0, min(h - 1, y));
    return a[y * w + x];
}

// ── Software upscale: bicubic (Mitchell) sample from accum (running average)
//    at render_w×render_h, write to full dst_w×dst_h surface. ────────────────
// NOTE: d_accum already holds the per-pixel running average — do NOT divide
//       by frame_count here or the image fades to black over time.
__global__ void sw_upscale_kernel(
    const float4* accum, int src_w, int src_h,
    cudaSurfaceObject_t surf, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // Map dst pixel to src (accum) coordinate
    float sx = ((float)dx + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
    float sy = ((float)dy + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
    int ix = (int)floorf(sx);
    int iy = (int)floorf(sy);

    // 4×4 bicubic neighbourhood
    float4 result = make_float4(0.f, 0.f, 0.f, 0.f);
    float  wsum   = 0.f;
    for (int jy = -1; jy <= 2; ++jy) {
        float wy = mitchell(sy - (float)(iy + jy));
        for (int jx = -1; jx <= 2; ++jx) {
            float wx = mitchell(sx - (float)(ix + jx));
            float  w  = wx * wy;
            float4 p  = accum_fetch(accum, ix + jx, iy + jy, src_w, src_h);
            result.x += p.x * w;
            result.y += p.y * w;
            result.z += p.z * w;
            wsum += w;
        }
    }
    if (wsum > 0.f) { result.x /= wsum; result.y /= wsum; result.z /= wsum; }
    // Clamp to avoid ringing artefacts going negative in HDR
    result.x = fmaxf(result.x, 0.f);
    result.y = fmaxf(result.y, 0.f);
    result.z = fmaxf(result.z, 0.f);
    result.w = 1.f;
    surf2Dwrite(result, surf, dx * (int)sizeof(float4), dy);
}

void pathtracer_sw_upscale(const float4* d_accum,
                            int src_w, int src_h,
                            cudaSurfaceObject_t surface,
                            int dst_w, int dst_h)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    sw_upscale_kernel<<<grid, block>>>(d_accum, src_w, src_h, surface, dst_w, dst_h);
}

// ── Debug PiP: draw small render (nearest-neighbour) in top-left corner ─────
__global__ void sw_pip_kernel(
    const float4* accum, int src_w, int src_h,
    cudaSurfaceObject_t surf, int pip_w, int pip_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= pip_w || dy >= pip_h) return;

    const int B = 2;  // border width in pixels
    // White border
    if (dx < B || dy < B || dx >= pip_w - B || dy >= pip_h - B) {
        surf2Dwrite(make_float4(1.f, 1.f, 1.f, 1.f), surf,
                    dx * (int)sizeof(float4), dy);
        return;
    }

    // Nearest-neighbour from running-average accum — shows actual render pixels
    int inner_w = pip_w - 2 * B;
    int inner_h = pip_h - 2 * B;
    int sx = (dx - B) * src_w / inner_w;
    int sy = (dy - B) * src_h / inner_h;
    sx = min(sx, src_w - 1);
    sy = min(sy, src_h - 1);

    float4 c = accum[sy * src_w + sx];
    c.w = 1.f;
    surf2Dwrite(c, surf, dx * (int)sizeof(float4), dy);
}

void pathtracer_sw_upscale_debug_pip(const float4* d_accum,
                                      int src_w, int src_h,
                                      cudaSurfaceObject_t surface,
                                      int dst_w, int dst_h)
{
    // PiP box = 1/3 of the full surface in each dimension
    int pip_w = dst_w / 3;
    int pip_h = dst_h / 3;
    dim3 block(16, 16);
    dim3 grid((pip_w + 15) / 16, (pip_h + 15) / 16);
    sw_pip_kernel<<<grid, block>>>(d_accum, src_w, src_h, surface, pip_w, pip_h);
}

// ── Depth auto-range reduction ────────────────────────────────────────────
// Works for non-negative floats: IEEE 754 positive floats are ordered the
// same as their unsigned int representations, so int atomics are valid.
__global__ void depth_range_kernel(const float* depth, int n, float* out_min, float* out_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = depth[i];
    if (d >= 1.f - 1e-4f) return;  // background sentinel — skip
    atomicMin((int*)out_min, __float_as_int(d));
    atomicMax((int*)out_max, __float_as_int(d));
}

void pathtracer_depth_range(const float* d_depth, int n, float* h_min, float* h_max)
{
    // Use two dedicated device floats (allocated once, reused each frame).
    static float* s_buf = nullptr;
    if (!s_buf) cudaMalloc(&s_buf, 2 * sizeof(float));

    // Init: min = 1.f (will only decrease), max = 0.f (will only increase).
    float init[2] = { 1.f, 0.f };
    cudaMemcpy(s_buf, init, sizeof(init), cudaMemcpyHostToDevice);
    depth_range_kernel<<<(n + 255) / 256, 256>>>(d_depth, n, s_buf, s_buf + 1);

    float result[2];
    cudaMemcpy(result, s_buf, sizeof(result), cudaMemcpyDeviceToHost); // implicit sync
    if (result[0] >= result[1]) { *h_min = 0.f; *h_max = 1.f; }      // all background
    else                         { *h_min = result[0]; *h_max = result[1]; }
}

__global__ void visualize_depth_kernel(const float* depth, int src_w, int src_h,
                                       float min_depth, float max_depth,
                                       cudaSurfaceObject_t surf, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    int sx = min(src_w - 1, max(0, (dx * src_w) / max(1, dst_w)));
    int sy = min(src_h - 1, max(0, (dy * src_h) / max(1, dst_h)));
    float d = depth[sy * src_w + sx];
    // Auto-normalized: min_depth maps to white (close), max_depth maps to black (far).
    // Background (d >= 1 - eps) stays black regardless.
    float vis = 0.f;
    if (d < 1.f - 1e-4f) {
        float range = fmaxf(1e-5f, max_depth - min_depth);
        float t = fminf(1.f, fmaxf(0.f, (d - min_depth) / range));
        vis = sqrtf(1.f - t);   // close=bright, far=dark, sqrt for perceptual linearity
    }
    surf2Dwrite(make_float4(vis, vis, vis, 1.f), surf, dx * (int)sizeof(float4), dy);
}

void pathtracer_visualize_depth(const float* d_depth, int src_w, int src_h,
                                float min_depth, float max_depth,
                                cudaSurfaceObject_t surface, int dst_w, int dst_h)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    visualize_depth_kernel<<<grid, block>>>(d_depth, src_w, src_h, min_depth, max_depth,
                                            surface, dst_w, dst_h);
}

// ── Motion max-magnitude reduction ───────────────────────────────────────
__global__ void motion_maxmag_kernel(const float2* motion, int n, float* out_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 mv = motion[i];
    float mag = sqrtf(mv.x * mv.x + mv.y * mv.y);
    atomicMax((int*)out_max, __float_as_int(mag));
}

float pathtracer_motion_maxmag(const float2* d_motion, int n)
{
    static float* s_buf = nullptr;
    if (!s_buf) cudaMalloc(&s_buf, sizeof(float));

    float init = 0.f;
    cudaMemcpy(s_buf, &init, sizeof(float), cudaMemcpyHostToDevice);
    motion_maxmag_kernel<<<(n + 255) / 256, 256>>>(d_motion, n, s_buf);

    float result;
    cudaMemcpy(&result, s_buf, sizeof(float), cudaMemcpyDeviceToHost); // implicit sync
    return (result < 0.001f) ? 0.f : result;  // 0 = no motion this frame
}

__global__ void visualize_motion_kernel(const float2* motion, int src_w, int src_h,
                                        float max_mag,
                                        cudaSurfaceObject_t surf, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    int sx = min(src_w - 1, max(0, (dx * src_w) / max(1, dst_w)));
    int sy = min(src_h - 1, max(0, (dy * src_h) / max(1, dst_h)));
    float2 mv = motion[sy * src_w + sx];

    float px_x  = mv.x;
    float px_y  = mv.y;
    float mag_px = sqrtf(px_x * px_x + px_y * px_y);

    // Zero motion → black.  Non-zero → HSV: hue = direction, value = magnitude.
    // Normalise against the frame's maximum magnitude so even sub-pixel motions
    // span the full brightness range (0 = no motion, 1 = fastest motion in frame).
    float r = 0.f, g = 0.f, b = 0.f;
    if (mag_px > 0.001f) {
        float norm = (max_mag > 0.001f) ? (mag_px / max_mag) : 1.f;
        float mag_v = fminf(1.f, norm);                      // normalised [0..1]
        float angle = atan2f(px_y, px_x);                    // [-pi, pi]
        float hue   = (angle + 3.14159265f) / (2.f * 3.14159265f);  // [0..1]
        float h6    = hue * 6.f;
        int   hi    = (int)h6 % 6;
        float f     = h6 - floorf(h6);
        float q     = mag_v * (1.f - f);
        float t     = mag_v * f;
        switch (hi) {
            case 0:  r = mag_v; g = t;     b = 0.f;    break;
            case 1:  r = q;     g = mag_v; b = 0.f;    break;
            case 2:  r = 0.f;   g = mag_v; b = t;      break;
            case 3:  r = 0.f;   g = q;     b = mag_v;  break;
            case 4:  r = t;     g = 0.f;   b = mag_v;  break;
            default: r = mag_v; g = 0.f;   b = q;      break;
        }
    }
    surf2Dwrite(make_float4(r, g, b, 1.f), surf, dx * (int)sizeof(float4), dy);
}

void pathtracer_visualize_motion(const float2* d_motion, int src_w, int src_h,
                                 float max_mag,
                                 cudaSurfaceObject_t surface, int dst_w, int dst_h)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    visualize_motion_kernel<<<grid, block>>>(d_motion, src_w, src_h, max_mag, surface, dst_w, dst_h);
}

__global__ void sw_upscale_tonemap_kernel(
    const float4* accum, int src_w, int src_h,
    cudaSurfaceObject_t surf, int dst_w, int dst_h,
    float exposure, int tone_map_mode)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = ((float)dx + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
    float sy = ((float)dy + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
    int ix = (int)floorf(sx);
    int iy = (int)floorf(sy);

    float4 result = make_float4(0.f, 0.f, 0.f, 0.f);
    float  wsum   = 0.f;
    for (int jy = -1; jy <= 2; ++jy) {
        float wy = mitchell(sy - (float)(iy + jy));
        for (int jx = -1; jx <= 2; ++jx) {
            float wx = mitchell(sx - (float)(ix + jx));
            float  w  = wx * wy;
            float4 p  = accum_fetch(accum, ix + jx, iy + jy, src_w, src_h);
            result.x += p.x * w;
            result.y += p.y * w;
            result.z += p.z * w;
            wsum += w;
        }
    }
    if (wsum > 0.f) { result.x /= wsum; result.y /= wsum; result.z /= wsum; }
    float3 tm = tonemap_for_viewport(make_float3(result.x, result.y, result.z), exposure, tone_map_mode);
    surf2Dwrite(make_float4(tm.x, tm.y, tm.z, 1.f), surf, dx * (int)sizeof(float4), dy);
}

void pathtracer_sw_upscale_tonemap(const float4* d_accum,
                                   int src_w, int src_h,
                                   cudaSurfaceObject_t surface,
                                   int dst_w, int dst_h,
                                   float exposure, int tone_map_mode)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    sw_upscale_tonemap_kernel<<<grid, block>>>(d_accum, src_w, src_h, surface, dst_w, dst_h,
                                               exposure, tone_map_mode);
}
