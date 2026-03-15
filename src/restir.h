#pragma once
#include "scene.h"
#include "bvh.h"
#include <cuda_runtime.h>
#include <vector>

// ─────────────────────────────────────────────
//  ReSTIR Direct Illumination
//  (Reservoir-based Spatiotemporal Importance Resampling)
//  Reference: Bitterli et al. 2020
// ─────────────────────────────────────────────

// Light source — one entry per emissive triangle
struct EmissiveTri {
    float3 v0, v1, v2;   // world-space vertices
    float3 emission;       // emissive radiance (linear RGB)
    float  area;           // triangle area
    float  total_area_inv; // 1 / sum of all emissive areas  (filled by extract fn)
};

// Per-pixel reservoir (Weighted Reservoir Sampling state)
struct Reservoir {
    int   y     = -1;    // selected emissive tri index, -1 = none
    float w_sum = 0.f;   // running sum of weights
    float W     = 0.f;   // unbiased contribution weight = w_sum / (M * p_hat(y))
    int   M     = 0;     // number of candidates seen
};

struct ReSTIRParams {
    cudaSurfaceObject_t  surface;
    float4*              accum_buffer;
    int                  width, height;
    Camera               cam;

    // Triangle BVH for shadow rays + primary hits
    BVHNode*             tri_bvh;
    Triangle*            triangles;
    int                  num_triangles;
    GpuMaterial*         gpu_materials;
    int                  num_gpu_materials;
    cudaTextureObject_t* textures;
    int                  num_textures;

    // Emissive light list
    EmissiveTri*         emissive_tris;
    int                  num_emissive;

    // Reservoir ping-pong buffers (both size = width*height)
    Reservoir*           reservoirs;       // primary output / spatial input
    Reservoir*           reservoirs_tmp;   // spatial output (swapped before shade)

    int                  frame_count;
    int                  spp;              // samples per pixel per frame
    int                  max_depth;        // indirect bounces after direct

    // Feature toggles
    int                  num_candidates;   // M (initial RIS samples, default 8)
    bool                 spatial_reuse;    // combine k spatial neighbor reservoirs
    int                  spatial_radius;   // pixel radius for spatial neighbors (default 30)

    // HDRI environment (same as PathTracerParams)
    cudaTextureObject_t  hdri_tex;
    float                hdri_intensity;
    float                hdri_yaw;
};

// ── Host API ────────────────────────────────────────────────────────────────

// Extract emissive triangles from loaded mesh data (CPU)
std::vector<EmissiveTri> restir_extract_emissive(
    const std::vector<Triangle>&    tris,
    const std::vector<GpuMaterial>& mats);

// Alloc per-pixel reservoir buffer on device
Reservoir* restir_alloc_reservoirs(int width, int height);

// Full ReSTIR DI pass: initial RIS → optional spatial → shade → write accum
// Always call restir_launch; it internally runs the correct kernel sequence.
void restir_launch(const ReSTIRParams& p, cudaStream_t stream = 0);
