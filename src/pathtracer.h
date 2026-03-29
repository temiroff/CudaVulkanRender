#pragma once

#include "scene.h"
#include "bvh.h"
#include <cuda_runtime.h>

struct PathTracerParams {
    cudaSurfaceObject_t surface;      // Vulkan texture — kernel writes here
    float4*             accum_buffer; // progressive accumulation (device ptr)
    int                 width;
    int                 height;
    Camera              cam;

    // Sphere BVH
    BVHNode*            bvh;
    Sphere*             prims;
    int                 num_prims;

    // Triangle mesh BVH (null if no mesh loaded)
    BVHNode*            tri_bvh;
    Triangle*           triangles;
    int                 num_triangles;

    // Mesh materials + textures (device ptrs)
    GpuMaterial*             gpu_materials;
    int                      num_gpu_materials;
    cudaTextureObject_t*     textures;       // device array of tex object handles
    int                      num_textures;

    int                 frame_count;  // increments each frame; reset on camera move
    int                 spp;          // samples per pixel per frame
    int                 max_depth;    // max ray bounce depth

    // Color mode: 0 = shaders, 1 = greyscale, 2 = random per object, 3 = USD base color
    int                 color_mode;
    float3*             obj_colors;     // device array indexed by obj_id
    int                 num_obj_colors;

    // HDRI environment
    cudaTextureObject_t hdri_tex;       // equirectangular float4 texture, 0 = use gradient
    float               hdri_intensity; // exposure multiplier (default 1.0)
    float               hdri_yaw;       // rotation offset in radians

    // Firefly suppression: clamp per-sample luminance to this value (0 = disabled)
    float               firefly_clamp;  // e.g. 10.0f

    // If true, skip the surf2Dwrite — caller will upscale d_accum to surface separately.
    // Use when render_w < viewport_w so the partial surface write doesn't flicker.
    int                 skip_surface_write;

    // SM activity tracker — device bitmask; bit i set when SM i executes this kernel.
    // Null-safe: macro checks for null before writing.
    uint32_t*           sm_tracker_bitmask;  // may be nullptr
};

struct DlssAuxParams {
    int                 width;
    int                 height;
    Camera              cam;
    Camera              prev_cam;
    float               camera_near;
    float               camera_far;
    int                 has_prev_camera;

    BVHNode*            bvh;
    Sphere*             prims;
    int                 num_prims;

    BVHNode*            tri_bvh;
    Triangle*           triangles;
    int                 num_triangles;

    cudaSurfaceObject_t depth_surface;
    cudaSurfaceObject_t motion_surface;
    float*              depth_buffer;
    float2*             motion_buffer;
};

// Launch path trace kernel
void pathtracer_launch(const PathTracerParams& p, cudaStream_t stream = 0);

// Allocate accumulation buffer
float4* pathtracer_alloc_accum(int width, int height);

// Reset accumulation (call when camera moves)
void pathtracer_reset_accum(float4* accum, int width, int height);

// Write an arbitrary float4 GPU buffer to a cudaSurfaceObject (e.g. after denoising)
void pathtracer_blit_surface(const float4* d_src, cudaSurfaceObject_t surface, int width, int height);
void pathtracer_blit_surface_tonemap(const float4* d_src, cudaSurfaceObject_t surface, int width, int height,
                                     float exposure, int tone_map_mode);
// Bilinear upscale accum (running average, src_w×src_h) → full surface (dst_w×dst_h)
void pathtracer_sw_upscale(const float4* d_accum, int src_w, int src_h,
                            cudaSurfaceObject_t surface, int dst_w, int dst_h);
void pathtracer_sw_upscale_tonemap(const float4* d_accum, int src_w, int src_h,
                                   cudaSurfaceObject_t surface, int dst_w, int dst_h,
                                   float exposure, int tone_map_mode);
// Debug: draw the small render as a PiP in the top-left of an already-upscaled surface
void pathtracer_sw_upscale_debug_pip(const float4* d_accum, int src_w, int src_h,
                                     cudaSurfaceObject_t surface, int dst_w, int dst_h);

// Generate DLSS auxiliary buffers from primary hits.
void pathtracer_write_dlss_aux(const DlssAuxParams& p, cudaStream_t stream = 0);

// Visualize scalar/motion debug buffers into an RGBA surface.
// pathtracer_depth_range: compute per-frame [min,max] of non-background depth values.
// pathtracer_motion_maxmag: compute per-frame max pixel-space motion magnitude.
// Call these before the corresponding visualize functions to get auto-normalized output.
void pathtracer_depth_range(const float* d_depth, int n, float* h_min, float* h_max);
float pathtracer_motion_maxmag(const float2* d_motion, int n);

void pathtracer_visualize_depth(const float* d_depth, int src_w, int src_h,
                                float min_depth, float max_depth,
                                cudaSurfaceObject_t surface, int dst_w, int dst_h);
// max_mag: result of pathtracer_motion_maxmag (0 = no motion this frame → all black).
void pathtracer_visualize_motion(const float2* d_motion, int src_w, int src_h,
                                 float max_mag,
                                 cudaSurfaceObject_t surface, int dst_w, int dst_h);
