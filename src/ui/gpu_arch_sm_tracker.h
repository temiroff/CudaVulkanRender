#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// ── Device-side macro ────────────────────────────────────────────────────────
// Place SM_TRACKER_RECORD(tracker.d_bitmask) anywhere inside a __global__ kernel
// to record that the current SM executed work during this frame window.
// d_bitmask must be a device uint32_t pointer, zero-initialised before each window.
#define SM_TRACKER_RECORD(d_bitmask)                                    \
    do {                                                                \
        uint32_t __smid;                                                \
        asm("mov.u32 %0, %smid;" : "=r"(__smid));                      \
        atomicOr((d_bitmask) + (__smid >> 5), 1u << (__smid & 31));   \
    } while (0)

// ── Host-side tracker ────────────────────────────────────────────────────────
struct SmTracker {
    uint32_t* d_bitmask = nullptr;   // device memory: [ceil(sm_count/32)] uint32_t words
    int       words     = 0;
    int       sm_count  = 0;
};

// Allocate the device bitmask buffer. Call once after CUDA init.
void sm_tracker_init(SmTracker& t, int sm_count);

// Copy the current bitmask to dst_host (size >= t.words uint32_t),
// then zero the device buffer for the next measurement window.
// Call every 500ms on the host.
void sm_tracker_read_reset(SmTracker& t, uint32_t* dst_host);

void sm_tracker_destroy(SmTracker& t);
