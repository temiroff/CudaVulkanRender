#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>

#include "scene.h"

struct OptixLaunchParams {
    float4*             accum_buffer;
    int                 width;
    int                 height;
    Camera              cam;
    Triangle*           triangles;
    int                 num_triangles;
    GpuMaterial*        gpu_materials;
    int                 num_gpu_materials;
    cudaTextureObject_t* textures;
    int                 num_textures;
    int                 frame_count;
    int                 spp;
    int                 max_depth;
    int                 color_mode;
    cudaTextureObject_t hdri_tex;
    float               hdri_intensity;
    float               hdri_yaw;
    float               firefly_clamp;
    OptixTraversableHandle traversable;
};

extern "C" {
__constant__ OptixLaunchParams params;
}

struct RadiancePRD {
    int       hit;
    HitRecord rec;
};

static __forceinline__ __device__ unsigned int pcg_hash(unsigned int seed)
{
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static __forceinline__ __device__ float rand_f(unsigned int& seed)
{
    seed = pcg_hash(seed);
    return (float)(seed >> 8) * (1.f / (float)(1u << 24));
}

static __forceinline__ __device__ unsigned int make_seed(int idx, int frame)
{
    return pcg_hash((unsigned int)idx ^ pcg_hash((unsigned int)frame));
}

static __forceinline__ __device__ float3 rand_in_unit_sphere(unsigned int& seed)
{
    float3 p;
    do {
        p = make_float3(rand_f(seed) * 2.f - 1.f,
                        rand_f(seed) * 2.f - 1.f,
                        rand_f(seed) * 2.f - 1.f);
    } while (dot(p, p) >= 1.f);
    return p;
}

static __forceinline__ __device__ float3 rand_unit_vector(unsigned int& seed)
{
    return normalize(rand_in_unit_sphere(seed));
}

static __forceinline__ __device__ float3 rand_in_unit_disk(unsigned int& seed)
{
    float3 p;
    do {
        p = make_float3(rand_f(seed) * 2.f - 1.f,
                        rand_f(seed) * 2.f - 1.f,
                        0.f);
    } while (dot(p, p) >= 1.f);
    return p;
}

static __forceinline__ __device__ Ray camera_ray(const Camera& cam, float u, float v, unsigned int& rng)
{
    float3 rd = cam.lens_radius * rand_in_unit_disk(rng);
    float3 offset = cam.u * rd.x + cam.v * rd.y;
    Ray ray;
    ray.origin = cam.origin + offset;
    ray.dir = normalize(cam.lower_left + u * cam.horizontal + v * cam.vertical - cam.origin - offset);
    return ray;
}

static __forceinline__ __device__ float4 sample_tex(const cudaTextureObject_t* textures, int idx, float2 uv)
{
    return tex2D<float4>(textures[idx], uv.x, uv.y);
}

static __forceinline__ __device__ float4 mul4(const float4& a, const float4& b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

static __forceinline__ __device__ float3 offset_ray_origin(const float3& p, const float3& geom_n, const float3& dir)
{
    float s = dot(dir, geom_n) >= 0.f ? 1.f : -1.f;
    return make_float3(p.x + geom_n.x * s * 1e-4f,
                       p.y + geom_n.y * s * 1e-4f,
                       p.z + geom_n.z * s * 1e-4f);
}

static __forceinline__ __device__ float3 hashed_object_color(int obj_id)
{
    unsigned h = (unsigned)obj_id * 2654435761u;
    float r = (float)((h >> 0) & 0xFF) / 255.f * 0.7f + 0.3f;
    float g = (float)((h >> 8) & 0xFF) / 255.f * 0.7f + 0.3f;
    float b = (float)((h >> 16) & 0xFF) / 255.f * 0.7f + 0.3f;
    return make_float3(r, g, b);
}

static __forceinline__ __device__ bool scatter_gpu_material(
    const Ray& r_in, HitRecord& rec,
    const GpuMaterial& mat,
    const cudaTextureObject_t* textures,
    float3& attenuation, float3& emission, Ray& scattered,
    unsigned int& rng)
{
    float4 base = mat.base_color;
    if (mat.base_color_tex >= 0) {
        float4 tex = sample_tex(textures, mat.base_color_tex, rec.uv);
        base = mat.custom_shader ? tex : mul4(base, tex);
    }

    float metallic = mat.metallic;
    float roughness = mat.roughness;
    if (mat.metallic_rough_tex >= 0) {
        float4 mr = sample_tex(textures, mat.metallic_rough_tex, rec.uv);
        if (mat.custom_shader) {
            roughness = mr.y;
            metallic = mr.z;
        } else {
            roughness *= mr.y;
            metallic *= mr.z;
        }
    }

    float4 emis = mat.emissive_factor;
    if (mat.emissive_tex >= 0) {
        float4 et = sample_tex(textures, mat.emissive_tex, rec.uv);
        if (mat.custom_shader) {
            emis.x = et.x; emis.y = et.y; emis.z = et.z;
        } else {
            emis.x *= et.x; emis.y *= et.y; emis.z *= et.z;
        }
    }
    emission = make_float3(emis.x, emis.y, emis.z);
    attenuation = make_float3(base.x, base.y, base.z);

    if (mat.normal_tex >= 0 && rec.tangent.w != 0.f) {
        float4 ns = sample_tex(textures, mat.normal_tex, rec.uv);
        float3 nm = make_float3(ns.x * 2.f - 1.f, ns.y * 2.f - 1.f, ns.z * 2.f - 1.f);
        float nm_len = sqrtf(nm.x * nm.x + nm.y * nm.y + nm.z * nm.z);
        if (nm_len > 1e-7f) {
            nm.x /= nm_len;
            nm.y /= nm_len;
            nm.z /= nm_len;
        }
        float3 T = make_float3(rec.tangent.x, rec.tangent.y, rec.tangent.z);
        float3 N = rec.normal;
        float3 B = cross(N, T) * rec.tangent.w;
        float3 mapped = make_float3(
            T.x * nm.x + B.x * nm.y + N.x * nm.z,
            T.y * nm.x + B.y * nm.y + N.y * nm.z,
            T.z * nm.x + B.z * nm.y + N.z * nm.z);
        float ml = sqrtf(dot(mapped, mapped));
        if (ml > 1e-7f)
            rec.normal = mapped / ml;
    }

    if (base.w < 0.1f) {
        scattered.origin = offset_ray_origin(rec.p, rec.geom_normal, r_in.dir);
        scattered.dir = r_in.dir;
        attenuation = make_float3(1.f, 1.f, 1.f);
        return true;
    }

    float3 albedo = make_float3(base.x, base.y, base.z);
    float alpha = roughness * roughness;
    alpha = fmaxf(alpha, 0.001f);

    float3 N = rec.normal;
    float3 V = make_float3(-r_in.dir.x, -r_in.dir.y, -r_in.dir.z);
    float NdotV = fmaxf(dot(N, V), 0.f);

    float3 F0 = make_float3(
        0.04f + (albedo.x - 0.04f) * metallic,
        0.04f + (albedo.y - 0.04f) * metallic,
        0.04f + (albedo.z - 0.04f) * metallic);

    float one_minus_v = 1.f - NdotV;
    float one_minus_v5 = one_minus_v * one_minus_v * one_minus_v * one_minus_v * one_minus_v;
    float3 F = make_float3(
        F0.x + (1.f - F0.x) * one_minus_v5,
        F0.y + (1.f - F0.y) * one_minus_v5,
        F0.z + (1.f - F0.z) * one_minus_v5);

    float F_lum = 0.2126f * F.x + 0.7152f * F.y + 0.0722f * F.z;
    bool do_specular = (rand_f(rng) < F_lum);

    if (do_specular) {
        float r1 = rand_f(rng);
        float r2 = rand_f(rng);
        float cos_theta2 = (1.f - r1) / (1.f + (alpha * alpha - 1.f) * r1);
        float cos_theta = sqrtf(fmaxf(cos_theta2, 0.f));
        float sin_theta = sqrtf(fmaxf(1.f - cos_theta2, 0.f));
        float phi = 2.f * 3.14159265f * r2;

        float3 up = (fabsf(N.z) < 0.999f) ? make_float3(0.f, 0.f, 1.f)
                                           : make_float3(1.f, 0.f, 0.f);
        float3 T = normalize(cross(up, N));
        float3 B = cross(N, T);
        float3 H = normalize(
            T * (sin_theta * cosf(phi)) +
            B * (sin_theta * sinf(phi)) +
            N * cos_theta);
        float3 L = reflect(r_in.dir, H);
        float NdotL = dot(N, L);
        if (NdotL <= 0.f)
            return false;

        attenuation = make_float3(F.x / fmaxf(F_lum, 1e-5f),
                                  F.y / fmaxf(F_lum, 1e-5f),
                                  F.z / fmaxf(F_lum, 1e-5f));
        float3 Ln = normalize(L);
        scattered.origin = offset_ray_origin(rec.p, rec.geom_normal, Ln);
        scattered.dir = Ln;
    } else {
        float3 diffuse_albedo = make_float3(
            albedo.x * (1.f - metallic),
            albedo.y * (1.f - metallic),
            albedo.z * (1.f - metallic));
        float inv_p = 1.f / fmaxf(1.f - F_lum, 1e-5f);
        attenuation = make_float3(diffuse_albedo.x * inv_p,
                                  diffuse_albedo.y * inv_p,
                                  diffuse_albedo.z * inv_p);
        float3 dir = N + rand_unit_vector(rng);
        if (fabsf(dir.x) < 1e-8f && fabsf(dir.y) < 1e-8f && fabsf(dir.z) < 1e-8f)
            dir = N;
        float3 dn = normalize(dir);
        scattered.origin = offset_ray_origin(rec.p, rec.geom_normal, dn);
        scattered.dir = dn;
    }
    return true;
}

static __forceinline__ __device__ float3 sample_sky(float3 dir)
{
    if (params.hdri_tex != 0) {
        float phi = atan2f(dir.x, -dir.z) + params.hdri_yaw;
        float theta = acosf(fmaxf(-1.f, fminf(1.f, dir.y)));
        float u = phi / (2.f * 3.14159265f);
        if (u < 0.f) u += 1.f;
        if (u > 1.f) u -= 1.f;
        float v = theta / 3.14159265f;
        float4 c = tex2D<float4>(params.hdri_tex, u, v);
        return make_float3(c.x * params.hdri_intensity,
                           c.y * params.hdri_intensity,
                           c.z * params.hdri_intensity);
    }

    float t = 0.5f * (normalize(dir).y + 1.f);
    return lerp(make_float3(1.f, 1.f, 1.f), make_float3(0.5f, 0.7f, 1.f), t);
}

static __forceinline__ __device__ void pack_pointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = static_cast<unsigned int>(uptr >> 32);
    i1 = static_cast<unsigned int>(uptr & 0xFFFFFFFFu);
}

template <typename T>
static __forceinline__ __device__ T* unpack_pointer(unsigned int i0, unsigned int i1)
{
    unsigned long long uptr = (static_cast<unsigned long long>(i0) << 32) | i1;
    return reinterpret_cast<T*>(uptr);
}

template <typename T>
static __forceinline__ __device__ T* get_prd()
{
    return unpack_pointer<T>(optixGetPayload_0(), optixGetPayload_1());
}

static __forceinline__ __device__ float3 trace_path(Ray ray, unsigned int& rng)
{
    float3 throughput = make_float3(1.f, 1.f, 1.f);
    float3 radiance = make_float3(0.f, 0.f, 0.f);

    for (int depth = 0; depth < params.max_depth; ++depth) {
        RadiancePRD prd = {};
        unsigned int u0 = 0;
        unsigned int u1 = 0;
        pack_pointer(&prd, u0, u1);
        optixTrace(
            params.traversable,
            ray.origin,
            ray.dir,
            1e-3f,
            1e16f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,
            1,
            0,
            u0,
            u1);

        if (!prd.hit) {
            radiance += throughput * sample_sky(ray.dir);
            break;
        }

        HitRecord rec = prd.rec;
        if (rec.gpu_mat_idx < 0 || rec.gpu_mat_idx >= params.num_gpu_materials) {
            float3 fallback = make_float3(fabsf(rec.normal.x), fabsf(rec.normal.y), fabsf(rec.normal.z));
            radiance += throughput * fallback;
            break;
        }

        Ray scattered;
        float3 attenuation = make_float3(1.f, 1.f, 1.f);
        float3 emission = make_float3(0.f, 0.f, 0.f);
        bool did_scatter = scatter_gpu_material(
            ray,
            rec,
            params.gpu_materials[rec.gpu_mat_idx],
            params.textures,
            attenuation,
            emission,
            scattered,
            rng);

        if (params.color_mode == 2 && rec.obj_id >= 0)
            attenuation = hashed_object_color(rec.obj_id);

        radiance += throughput * emission;
        if (!did_scatter)
            break;

        throughput = throughput * attenuation;
        ray = scattered;
    }

    return radiance;
}

extern "C" __global__ void __closesthit__ch()
{
    RadiancePRD* prd = get_prd<RadiancePRD>();
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const Triangle& tri = params.triangles[prim_idx];
    const float2 bary = optixGetTriangleBarycentrics();
    const float b1 = bary.x;
    const float b2 = bary.y;
    const float b0 = 1.f - b1 - b2;

    HitRecord rec = {};
    rec.t = optixGetRayTmax();
    const float3 origin = optixGetWorldRayOrigin();
    const float3 dir = optixGetWorldRayDirection();
    rec.p = origin + dir * rec.t;

    float3 outward_n = tri.n0 * b0 + tri.n1 * b1 + tri.n2 * b2;
    float outward_len = sqrtf(dot(outward_n, outward_n));
    float3 face_n = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
    if (outward_len > 1e-7f)
        outward_n = outward_n / outward_len;
    else
        outward_n = face_n;

    rec.geom_normal = face_n;
    rec.uv = tri.uv0 * b0 + tri.uv1 * b1 + tri.uv2 * b2;
    rec.tangent = make_float4(
        tri.t0.x * b0 + tri.t1.x * b1 + tri.t2.x * b2,
        tri.t0.y * b0 + tri.t1.y * b1 + tri.t2.y * b2,
        tri.t0.z * b0 + tri.t1.z * b1 + tri.t2.z * b2,
        tri.t0.w * b0 + tri.t1.w * b1 + tri.t2.w * b2);
    rec.gpu_mat_idx = tri.mat_idx;
    rec.obj_id = tri.obj_id;

    Ray ray;
    ray.origin = origin;
    ray.dir = dir;
    rec.set_face_normal(ray, outward_n);

    prd->hit = 1;
    prd->rec = rec;
}

extern "C" __global__ void __miss__ms()
{
    RadiancePRD* prd = get_prd<RadiancePRD>();
    prd->hit = 0;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dim = optixGetLaunchDimensions();
    const int x = static_cast<int>(launch_idx.x);
    const int y = static_cast<int>(launch_idx.y);
    if (x >= params.width || y >= params.height)
        return;

    const int idx = y * params.width + x;
    unsigned int rng = make_seed(idx, params.frame_count);

    float3 color = make_float3(0.f, 0.f, 0.f);
    const float width_denom = (float)max(1, params.width - 1);
    const float height_denom = (float)max(1, params.height - 1);

    for (int s = 0; s < params.spp; ++s) {
        unsigned int srng = pcg_hash(rng ^ (unsigned int)s);
        float u = (x + rand_f(srng)) / width_denom;
        float v = 1.f - (y + rand_f(srng)) / height_denom;
        Ray ray = camera_ray(params.cam, u, v, srng);
        float3 sample = trace_path(ray, srng);
        if (params.firefly_clamp > 0.f) {
            float lum = 0.2126f * sample.x + 0.7152f * sample.y + 0.0722f * sample.z;
            if (lum > params.firefly_clamp)
                sample = sample * (params.firefly_clamp / lum);
        }
        color += sample;
    }

    color = color / (float)params.spp;
    if (params.color_mode == 1) {
        float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
        color = make_float3(lum, lum, lum);
    }

    float4 prev = params.accum_buffer[idx];
    float t = 1.f / (float)(params.frame_count + 1);
    float3 accum = lerp(make_float3(prev.x, prev.y, prev.z), color, t);
    params.accum_buffer[idx] = make_float4(accum.x, accum.y, accum.z, 1.f);
}
