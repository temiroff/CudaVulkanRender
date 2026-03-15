#pragma once
#include <cuda_runtime.h>
#include <string>

struct HdriMap {
    cudaTextureObject_t tex    = 0;
    cudaArray_t         arr    = nullptr;
    int                 width  = 0;
    int                 height = 0;
    bool                loaded = false;
};

// Load an equirectangular HDR image (.hdr / Radiance RGBE) and upload to GPU.
// Returns an HdriMap with loaded=true on success.
HdriMap hdri_load(const std::string& path);

// Free all GPU resources.
void hdri_free(HdriMap& map);
