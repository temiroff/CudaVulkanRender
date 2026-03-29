#pragma once
#include <stdint.h>

// Architecture topology constants — derived from compute capability.
// GPC/TPC counts are not queryable via CUDA; they come from a static lookup table.
struct ArchTopology {
    int  gpcs          = 0;
    int  tpcs_per_gpc  = 0;
    int  sms_per_tpc   = 0;
    int  full_die_sms  = 0;      // gpcs * tpcs_per_gpc * sms_per_tpc
    int  fp32_only     = 64;     // FP32-only ALU count per SM  (blue)
    int  fp32_int32    = 64;     // second partition count per SM (cyan) — FP32/INT32 on Ada/Ampere
    int  tensor_cores  = 0;      // tensor core units per SM
    int  rt_cores      = 0;      // RT core units per SM  (0 = no RT)
    int  tex_units     = 4;      // texture units per SM
    int  fp64_units    = 2;      // FP64 (double precision) ALUs per SM
    int  ldst_units    = 32;     // load/store units per SM
    int  sfu_units     = 16;     // special function units per SM (sin/cos/rcp/rsqrt)
    int  warp_schedulers = 4;   // warp schedulers per SM (each issues 1 warp/cycle)
    int  reg_file_kb   = 256;   // register file size per SM in KB (65536 × 32-bit regs)
    int  l1_shared_kb  = 128;    // max L1/shared mem per SM in KB
    const char* tensor_label  = "";
    const char* rt_label      = "";
    const char* part_b_label  = "FP32/INT32";  // label for second ALU partition
};

enum class ArchLevel { GPU = 0, GPC, SM };

struct GpuArchWindowState {
    bool         open        = true;
    ArchLevel    level       = ArchLevel::GPU;
    int          sel_gpc     = -1;
    int          sel_sm      = -1;
    ArchTopology topo;
    bool         topo_ready  = false;
    int          l2_cache_mb = 0;     // from cudaDeviceProp.l2CacheSize / (1024*1024)

    // Per-SM activity bitmask: bit i set means SM i was active in the last 500ms window.
    // Populated from SM_TRACKER_RECORD in CUDA kernels (main app)
    // or approximated from NVML overall util (standalone).
    static constexpr int MAX_SM_WORDS = 8;  // supports up to 256 SMs
    uint32_t sm_active[MAX_SM_WORDS] = {};

    // Live metrics — caller fills these from GpuLiveStats or NVML each update tick
    uint32_t gpu_util_pct = 0;
    uint32_t mem_util_pct = 0;
    uint64_t vram_used    = 0;
    uint64_t vram_total   = 0;

    // Unit activity sampling.
    // When profiling_enabled = false, only SM-level activity is shown (safe, zero overhead).
    // When true, callers should populate the *_active_pct fields from hardware counters
    // (CUPTI Profiler API).  Fields left at 0 are shown as "no data".
    bool  profiling_enabled   = false;
    bool  cupti_available     = false;  // set by caller after cupti_profiler.init()

    // Per-unit activity [0,100]. Caller fills from CUPTI when profiling_enabled.
    // CUDA/LD/ST/SFU are approximated from SM bitmask when profiling is off.
    // Tensor/RT/TEX are 0 unless caller provides real counter data.
    float cuda_active_pct   = 0.f;
    float ldst_active_pct   = 0.f;
    float sfu_active_pct    = 0.f;
    float tensor_active_pct = 0.f;  // requires CUPTI — stays 0 without it
    float rt_active_pct     = 0.f;  // requires CUPTI — stays 0 without it
    float tex_active_pct    = 0.f;  // requires CUPTI — stays 0 without it
    bool  has_unit_counters  = false; // true when *_active_pct come from real HW counters
};

// Call once after CUDA init. Populates topology from compute capability and sets L2 size.
void gpu_arch_window_init(GpuArchWindowState& s,
                          int compute_major, int compute_minor,
                          int l2_cache_mb);

// Call every frame when the window should be visible.
// hw_name  : device name from cudaDeviceProp.name
// sm_count : actual enabled SM count from cudaDeviceProp.multiProcessorCount
// p_open   : window close button writes false here; pass nullptr to use s.open
void gpu_arch_window_draw(GpuArchWindowState& s,
                          const char*         hw_name,
                          int                 sm_count,
                          bool*               p_open = nullptr);
