#include "sensor_panel.h"
#include <cuda_runtime.h>

// Nearest-neighbour blit from a CUDA array (texture object) into a corner of
// the main output surface, with a white border. Source is assumed to be the
// same resolution as the destination region (no scaling), so src_w==dst_region_w.
__global__ static void sensor_blit_kernel(
    cudaSurfaceObject_t dst, int dst_w, int dst_h,
    cudaTextureObject_t src_tex, int src_w, int src_h,
    int off_x, int off_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src_w || y >= src_h) return;

    int dx = off_x + x;
    int dy = off_y + y;
    if (dx < 0 || dy < 0 || dx >= dst_w || dy >= dst_h) return;

    const int B = 2;
    float4 c;
    if (x < B || y < B || x >= src_w - B || y >= src_h - B) {
        c = make_float4(1.f, 1.f, 1.f, 1.f);
    } else {
        // Sample centre of source pixel with normalized coords.
        float u = (x + 0.5f) / (float)src_w;
        float v = (y + 0.5f) / (float)src_h;
        c = tex2D<float4>(src_tex, u, v);
        c.w = 1.f;
    }
    surf2Dwrite(c, dst, dx * (int)sizeof(float4), dy);
}

void sensor_blit_corner(cudaSurfaceObject_t dst, int dst_w, int dst_h,
                        cudaArray_t src_array, int src_w, int src_h,
                        int off_x, int off_y)
{
    if (!src_array || src_w <= 0 || src_h <= 0) return;

    // Wrap the source array in a short-lived texture object for sampling.
    cudaResourceDesc rd{};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = src_array;
    cudaTextureDesc td{};
    td.addressMode[0]    = cudaAddressModeClamp;
    td.addressMode[1]    = cudaAddressModeClamp;
    td.filterMode        = cudaFilterModePoint;
    td.readMode          = cudaReadModeElementType;
    td.normalizedCoords  = 1;
    cudaTextureObject_t src_tex = 0;
    cudaCreateTextureObject(&src_tex, &rd, &td, nullptr);

    dim3 block(16, 16);
    dim3 grid((src_w + 15) / 16, (src_h + 15) / 16);
    sensor_blit_kernel<<<grid, block>>>(dst, dst_w, dst_h,
                                        src_tex, src_w, src_h,
                                        off_x, off_y);

    cudaDestroyTextureObject(src_tex);
}
