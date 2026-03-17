#pragma once

#include "pathtracer.h"
#include <vector>

struct OptixRendererState {
    void*              ctx             = nullptr; // OptixDeviceContext
    void*              module          = nullptr; // OptixModule
    void*              pipeline        = nullptr; // OptixPipeline
    void*              raygen_pg       = nullptr; // OptixProgramGroup
    void*              miss_pg         = nullptr; // OptixProgramGroup
    void*              hitgroup_pg     = nullptr; // OptixProgramGroup
    void*              d_launch_params = nullptr; // CUdeviceptr
    void*              d_vertices      = nullptr; // CUdeviceptr
    void*              d_indices       = nullptr; // CUdeviceptr
    void*              d_gas_buffer    = nullptr; // CUdeviceptr
    void*              d_sbt_raygen    = nullptr; // CUdeviceptr
    void*              d_sbt_miss      = nullptr; // CUdeviceptr
    void*              d_sbt_hitgroup  = nullptr; // CUdeviceptr
    unsigned long long gas_handle      = 0;       // OptixTraversableHandle
    unsigned long long scene_version   = 0;
    int                width           = 0;
    int                height          = 0;
    int                num_triangles   = 0;
    bool               available       = false;
    bool               initialized     = false;
    bool               scene_ready     = false;
    char               last_error[512] = {};
};

bool optix_renderer_init(OptixRendererState& s, int width, int height);
bool optix_renderer_upload_scene(OptixRendererState& s, const std::vector<Triangle>& tris,
                                 unsigned long long scene_version);
bool optix_renderer_render(OptixRendererState& s, const PathTracerParams& p, cudaStream_t stream = 0);
void optix_renderer_resize(OptixRendererState& s, int width, int height);
void optix_renderer_free(OptixRendererState& s);
