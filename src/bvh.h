#pragma once

#include "scene.h"
#include <vector>

// ─────────────────────────────────────────────
//  BVH node — fits in a flat array (GPU-friendly)
// ─────────────────────────────────────────────

struct BVHNode {
    AABB aabb;
    int  left;       // child index or -1
    int  right;      // child index or -1
    int  prim_start; // first prim index (-1 if interior)
    int  prim_count; // 0 if interior node
};

// ─────────────────────────────────────────────
//  Sphere BVH
// ─────────────────────────────────────────────

void bvh_build(
    const std::vector<Sphere>& prims_in,
    std::vector<BVHNode>&      nodes_out,
    std::vector<Sphere>&       prims_out
);

void bvh_upload(
    const std::vector<BVHNode>& nodes,
    const std::vector<Sphere>&  prims,
    BVHNode** d_nodes,
    Sphere**  d_prims
);

__device__ bool bvh_hit(
    const BVHNode* nodes, const Sphere* prims,
    const Ray& r, float t_min, float t_max,
    HitRecord& rec
);

// ─────────────────────────────────────────────
//  Triangle BVH
// ─────────────────────────────────────────────

void bvh_build_triangles(
    const std::vector<Triangle>& prims_in,
    std::vector<BVHNode>&        nodes_out,
    std::vector<Triangle>&       prims_out
);

void bvh_upload_triangles(
    const std::vector<BVHNode>&  nodes,
    const std::vector<Triangle>& prims,
    BVHNode**  d_nodes,
    Triangle** d_prims
);

// Refit BVH AABBs in-place from updated triangle positions (O(N), no realloc).
void bvh_refit_triangles(
    std::vector<BVHNode>& nodes,
    const std::vector<Triangle>& prims,
    BVHNode* d_nodes
);

__device__ bool bvh_hit_triangles(
    const BVHNode* nodes, const Triangle* prims,
    const Ray& r, float t_min, float t_max,
    HitRecord& rec
);
