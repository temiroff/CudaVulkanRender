#pragma once
#include <cstdint>

// Per-unit SM-pipe utilisation collected from hardware counters via CUPTI.
// All values in [0, 100] % of peak sustained throughput.
struct CuptiMetrics {
    float cuda_active_pct   = 0.f;  // sm__pipe_alu_cycles_active   — FP32 ALU
    float tex_active_pct    = 0.f;  // sm__pipe_l1tex_cycles_active  — TEX units
    float ldst_active_pct   = 0.f;  // sm__pipe_lsu_cycles_active    — LD/ST units
    float sfu_active_pct    = 0.f;  // sm__pipe_xu_cycles_active     — SFU (sin/cos/rcp)
    float tensor_active_pct = 0.f;  // sm__pipe_tensor_cycles_active — Tensor cores
    bool  valid             = false; // true when values come from real HW counters
};

#ifdef CUPTI_ENABLED

// CUPTI Profiler wrapper — thin state machine around the CUPTI Profiler API +
// NvPerf metrics evaluator.  All methods are no-ops on error (graceful degradation).
//
// Usage per frame:
//   profiler.begin_frame();   // wraps kernel dispatch
//   ... launch CUDA kernels ...
//   profiler.end_frame();     // reads counters ~every N frames
//   CuptiMetrics m = profiler.get_metrics();
class CuptiProfiler {
public:
    // Call once after cudaSetDevice().  Returns false if CUPTI init fails
    // (driver doesn't support profiling — e.g. vGPU, container restrictions).
    bool init(int device_index);

    void shutdown();

    // Wrap render-kernel dispatch; safe to call every frame.
    void begin_frame();
    void end_frame();

    CuptiMetrics get_metrics() const { return m_metrics; }
    bool is_initialized()      const { return m_ok; }

    ~CuptiProfiler() { shutdown(); }

private:
    bool         m_ok         = false;
    int          m_device     = 0;
    int          m_frame      = 0;
    bool         m_capturing  = false;  // true between begin_frame and end_frame on capture frames
    CuptiMetrics m_metrics;

    // Opaque internal state (avoids leaking CUPTI headers into every TU)
    struct Impl;
    Impl* m_impl = nullptr;
};

#else  // !CUPTI_ENABLED — stub so the rest of the code compiles unchanged

class CuptiProfiler {
public:
    bool init(int)             { return false; }
    void shutdown()            {}
    void begin_frame()         {}
    void end_frame()           {}
    CuptiMetrics get_metrics() const { return {}; }
    bool is_initialized()      const { return false; }
};

#endif // CUPTI_ENABLED
