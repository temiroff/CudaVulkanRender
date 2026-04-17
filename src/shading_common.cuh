#pragma once

#include "scene.h"
#include <cuda_runtime.h>

// Blinn-Phong specular helper: returns specular intensity for a single light.
__device__ inline float blinn_spec(float3 N, float3 L, float3 V, float shininess)
{
    float3 H = normalize(L + V);
    float  ndh = fmaxf(0.f, dot(N, H));
    return powf(ndh, shininess);
}

// Camera-relative 3-point Blinn-Phong lighting for solid/rasterized viewport modes.
// N: surface normal (world space, unit), V: surface→camera view direction (world space, unit),
// key_dir/fill_dir/rim_dir: pre-computed unit light directions (world space),
// base_color: material albedo (linear RGB),
// metallic/roughness: PBR params (roughness→shininess, metallic→spec color).
__device__ inline float3 three_point_shade_view(float3 N, float3 V,
                                                float3 key_dir, float3 fill_dir, float3 rim_dir,
                                                float3 base_color,
                                                float metallic = 0.f, float roughness = 0.5f)
{
    float ndv = fmaxf(0.f, dot(N, V));
    float fresnel = powf(1.f - ndv, 5.f);

    // Roughness → Blinn-Phong shininess (perceptual mapping)
    float smooth = 1.f - roughness;
    float shininess = 2.f + smooth * smooth * 254.f;  // range [2, 256]

    // Specular color: dielectrics get white spec, metals tint with base color
    float3 spec_color = make_float3(
        metallic * base_color.x + (1.f - metallic) * 0.04f,
        metallic * base_color.y + (1.f - metallic) * 0.04f,
        metallic * base_color.z + (1.f - metallic) * 0.04f);

    // Specular intensity scales with smoothness
    float spec_strength = 0.5f + smooth * 0.5f;
    float grazing_boost = 1.f + fresnel * (1.5f + smooth);

    // Key light
    float  key_ndl = fmaxf(0.f, dot(N, key_dir));
    float3 key_tint = make_float3(1.0f, 0.98f, 0.95f);
    float3 key_diff = base_color * key_tint * (key_ndl * 0.55f);
    float3 key_spec = spec_color * key_tint * (blinn_spec(N, key_dir, V, shininess) * key_ndl * spec_strength * 0.6f * grazing_boost);

    // Fill light
    float  fill_ndl = fmaxf(0.f, dot(N, fill_dir));
    float3 fill_tint = make_float3(0.9f, 0.92f, 1.0f);
    float3 fill_diff = base_color * fill_tint * (fill_ndl * 0.25f);
    float3 fill_spec = spec_color * fill_tint * (blinn_spec(N, fill_dir, V, shininess) * fill_ndl * spec_strength * 0.2f * grazing_boost);

    // Rim light
    float  rim_ndl = fmaxf(0.f, dot(N, rim_dir));
    float3 rim_diff = base_color * (rim_ndl * 0.1f);
    float3 rim_spec = make_float3(0.f, 0.f, 0.f);

    // Ambient
    float3 ambient = base_color * (0.08f + 0.04f * fresnel);

    float3 c = ambient + key_diff + key_spec + fill_diff + fill_spec + rim_diff + rim_spec;
    c.x = fminf(c.x, 1.f);
    c.y = fminf(c.y, 1.f);
    c.z = fminf(c.z, 1.f);
    return c;
}

__device__ inline float3 three_point_shade(float3 N, float3 cam_w,
                                           float3 key_dir, float3 fill_dir, float3 rim_dir,
                                           float3 base_color,
                                           float metallic = 0.f, float roughness = 0.5f)
{
    // cam_w points from the target toward the camera, which matches surface→camera.
    return three_point_shade_view(N, cam_w, key_dir, fill_dir, rim_dir,
                                  base_color, metallic, roughness);
}

// Sample base color from a GpuMaterial, applying texture if present.
__device__ inline float3 sample_base_color(const GpuMaterial& m,
                                           cudaTextureObject_t* textures,
                                           float u, float v)
{
    float4 bc = m.base_color;
    if (m.base_color_tex >= 0 && textures) {
        float4 t = tex2D<float4>(textures[m.base_color_tex], u, v);
        bc = make_float4(bc.x * t.x, bc.y * t.y, bc.z * t.z, bc.w * t.w);
    }
    return make_float3(bc.x, bc.y, bc.z);
}
