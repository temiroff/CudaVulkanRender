#pragma once

#include <cuda_runtime.h>

// ─────────────────────────────────────────────
//  Math helpers (CUDA device + host)
// ─────────────────────────────────────────────

__host__ __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float3 operator*(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float3 operator*(float s, float3 a) { return a * s; }
__host__ __device__ inline float3 operator/(float3 a, float s) { return make_float3(a.x/s, a.y/s, a.z/s); }
__host__ __device__ inline float3& operator+=(float3& a, float3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }

__host__ __device__ inline float dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ inline float length(float3 v) { return sqrtf(dot(v, v)); }
__host__ __device__ inline float3 normalize(float3 v) { return v / length(v); }
__host__ __device__ inline float3 lerp(float3 a, float3 b, float t) { return a * (1.f - t) + b * t; }
__host__ __device__ inline float3 reflect(float3 v, float3 n) { return v - 2.f * dot(v, n) * n; }

// ─────────────────────────────────────────────
//  Camera
// ─────────────────────────────────────────────

struct Camera {
    float3 origin;
    float3 lower_left;
    float3 horizontal;
    float3 vertical;
    float3 u, v, w;     // basis vectors
    float  lens_radius;

    __host__ __device__ static Camera make(
        float3 look_from, float3 look_at, float3 vup,
        float vfov_deg, float aspect, float aperture, float focus_dist)
    {
        Camera cam;
        float theta = vfov_deg * 3.14159265f / 180.f;
        float h = tanf(theta * 0.5f);
        float viewport_h = 2.f * h;
        float viewport_w = aspect * viewport_h;

        cam.w = normalize(look_from - look_at);
        cam.u = normalize(cross(vup, cam.w));
        cam.v = cross(cam.w, cam.u);

        cam.origin     = look_from;
        cam.horizontal = focus_dist * viewport_w * cam.u;
        cam.vertical   = focus_dist * viewport_h * cam.v;
        cam.lower_left = cam.origin - cam.horizontal*0.5f - cam.vertical*0.5f - focus_dist*cam.w;
        cam.lens_radius = aperture * 0.5f;
        return cam;
    }
};

// ─────────────────────────────────────────────
//  Ray
// ─────────────────────────────────────────────

struct Ray {
    float3 origin;
    float3 dir;
    __host__ __device__ float3 at(float t) const { return origin + dir * t; }
};

// ─────────────────────────────────────────────
//  Material (sphere / inline)
// ─────────────────────────────────────────────

enum class MatType : int {
    Lambertian = 0,
    Metal      = 1,
    Dielectric = 2,
    Emissive   = 3,
};

struct Material {
    float3  albedo;
    float3  emission;
    float   roughness;   // metal fuzz
    float   ior;         // dielectric IOR
    MatType type;
};

// ─────────────────────────────────────────────
//  Primitive — Sphere
// ─────────────────────────────────────────────

struct Sphere {
    float3   center;
    float    radius;
    Material mat;
};

// ─────────────────────────────────────────────
//  Primitive — Triangle (for glTF meshes)
// ─────────────────────────────────────────────

struct Triangle {
    float3 v0, v1, v2;     // vertex positions (world space)
    float3 n0, n1, n2;     // per-vertex normals (world space, unit)
    float2 uv0, uv1, uv2;  // texture coordinates
    // Tangent vectors (xyz = world-space direction, w = bitangent sign +1/-1)
    // w == 0 means no tangent data — skip normal mapping for this triangle
    float4 t0, t1, t2;
    int    mat_idx;         // index into GpuMaterial array
    int    obj_id;          // which glTF mesh primitive this triangle belongs to
};

// ─────────────────────────────────────────────
//  GPU-resident PBR material (glTF metallic-roughness)
// ─────────────────────────────────────────────

struct GpuMaterial {
    float4 base_color;          // linear RGBA
    float  metallic;
    float  roughness;
    float4 emissive_factor;     // linear RGB (w unused)
    int    base_color_tex;      // image index into texture array, -1 = none
    int    metallic_rough_tex;  // G channel = roughness, B channel = metallic
    int    normal_tex;          // tangent-space normal map (-1 = unused)
    int    emissive_tex;        // -1 = none
    // 1 = custom shader owns all textures — read them as absolute values, not factors
    int    custom_shader;
};

// ─────────────────────────────────────────────
//  Hit record
// ─────────────────────────────────────────────

struct HitRecord {
    float3   p;
    float3   normal;
    float3   geom_normal;   // un-interpolated face normal — used for ray-origin offset
    float4   tangent;       // xyz = world tangent, w = bitangent sign (0 = no tangent)
    Material mat;           // inline sphere material (used when gpu_mat_idx < 0)
    float    t;
    bool     front_face;
    float2   uv;            // texture coordinates (mesh hits)
    int      gpu_mat_idx;   // -1 = sphere (use inline mat), >= 0 = mesh GpuMaterial
    int      obj_id = -1;  // mesh object id for per-object shading modes

    __host__ __device__ void set_face_normal(const Ray& r, float3 outward_n) {
        front_face = dot(r.dir, outward_n) < 0.f;
        normal = front_face ? outward_n : make_float3(-outward_n.x, -outward_n.y, -outward_n.z);
    }
};

// ─────────────────────────────────────────────
//  AABB (for BVH)
// ─────────────────────────────────────────────

struct AABB {
    float3 mn, mx;

    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max) const {
        for (int i = 0; i < 3; ++i) {
            float inv = 1.f / ((&r.dir.x)[i]);
            float t0 = ((&mn.x)[i] - (&r.origin.x)[i]) * inv;
            float t1 = ((&mx.x)[i] - (&r.origin.x)[i]) * inv;
            if (inv < 0.f) { float tmp = t0; t0 = t1; t1 = tmp; }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min) return false;
        }
        return true;
    }
};
