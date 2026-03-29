#include "gpu_arch_sm_tracker.h"
#include <cstring>

void sm_tracker_init(SmTracker& t, int sm_count)
{
    t.sm_count = sm_count;
    t.words    = (sm_count + 31) / 32;
    cudaMalloc(&t.d_bitmask, t.words * sizeof(uint32_t));
    cudaMemset(t.d_bitmask, 0, t.words * sizeof(uint32_t));
}

void sm_tracker_read_reset(SmTracker& t, uint32_t* dst)
{
    if (!t.d_bitmask || !dst) return;
    cudaMemcpy(dst, t.d_bitmask, t.words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemset(t.d_bitmask, 0, t.words * sizeof(uint32_t));
}

void sm_tracker_destroy(SmTracker& t)
{
    if (t.d_bitmask) { cudaFree(t.d_bitmask); t.d_bitmask = nullptr; }
    t.words = t.sm_count = 0;
}
