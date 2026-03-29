#ifdef CUPTI_ENABLED

#include "cupti_profiler.h"

#include <cupti_profiler_target.h>  // CUpti_Profiler_* (session/pass/range/config)
#include <cupti_target.h>           // CUpti_Device_GetChipName_Params
#include <nvperf_host.h>            // NVPW host: CounterDataBuilder, MetricsEvaluator
#include <nvperf_cuda_host.h>       // NVPW_CUDA_*: RawMetricsConfig, MetricsEvaluator init
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <cmath>

// ── Error-check macros ─────────────────────────────────────────────────────────
#define CUPTI_CHECK_RET(expr, ret) do { \
    CUptiResult _r = (expr); \
    if (_r != CUPTI_SUCCESS) { \
        const char* _msg = nullptr; \
        cuptiGetResultString(_r, &_msg); \
        std::cerr << "[cupti] " #expr " failed: " << (_msg ? _msg : "?") \
                  << " (" << (int)_r << ")\n"; \
        return (ret); \
    } \
} while(0)

#define NVPW_CHECK_RET(expr, ret) do { \
    NVPA_Status _s = (expr); \
    if (_s != NVPA_STATUS_SUCCESS) { \
        std::cerr << "[cupti] NVPW " #expr " failed: " << (int)_s << "\n"; \
        return (ret); \
    } \
} while(0)

#define CUPTI_CHECK(expr)  CUPTI_CHECK_RET(expr, false)
#define NVPW_CHECK(expr)   NVPW_CHECK_RET(expr, false)

// ── Logical metrics + candidate aliases (driver/chip dependent naming) ────────
static constexpr int k_num_metrics = 5;
enum MetricSlot {
    METRIC_CUDA = 0,
    METRIC_TEX,
    METRIC_LDST,
    METRIC_SFU,
    METRIC_TENSOR
};

static const char* k_metric_labels[k_num_metrics] = {
    "CUDA", "TEX", "LD/ST", "SFU", "Tensor"
};

static const char* k_metric_candidates[k_num_metrics][4] = {
    { // CUDA
      "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
      "sm__throughput.avg.pct_of_peak_sustained_elapsed",
      "smsp__throughput.avg.pct_of_peak_sustained_elapsed",
      nullptr },
    { // TEX
      "sm__pipe_l1tex_cycles_active.avg.pct_of_peak_sustained_active",
      "smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active",
      "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active",
      "l1tex__throughput.avg.pct_of_peak_sustained_elapsed" },
    { // LD/ST
      "sm__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_active",
      "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
      "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
      "smsp__lsu_busy.avg.pct_of_peak_sustained_elapsed" },
    { // SFU
      "sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active",
      "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
      "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
      "smsp__xu_busy.avg.pct_of_peak_sustained_elapsed" },
    { // Tensor
      "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
      "smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
      "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
      "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active" },
};

static std::array<std::string, k_num_metrics> s_resolved_metric_names{};

// How often to collect (every N frames, ~100ms at 100fps)
static constexpr int k_sample_interval = 10;

// ── Internal state ─────────────────────────────────────────────────────────────
struct CuptiProfiler::Impl {
    std::string          chip_name;
    std::vector<uint8_t> config_image;
    std::vector<uint8_t> counter_data_prefix;
    std::vector<uint8_t> counter_data_image;
    std::vector<uint8_t> counter_data_scratch;
    bool session_open = false;
    bool pass_open    = false;
    bool range_open   = false;
};

// ── Resolve high-level metric names → raw hardware counter requests ────────────
// AddMetrics on RawMetricsConfig/CounterDataBuilder expects raw counter names
// (e.g. "sm__pipe_alu_cycles_active.sum"), not the high-level derived names
// (e.g. "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active").
// We use a chip-only MetricsEvaluator + GetMetricRawDependencies to resolve them.
static std::vector<NVPA_RawMetricRequest>
resolve_raw_requests(const std::string& chip_name,
                     std::vector<std::string>& raw_name_storage)  // keeps string memory alive
{
    std::vector<NVPA_RawMetricRequest> reqs;
    std::vector<std::string> unique_raw_names;
    static std::unordered_map<std::string, bool> s_metric_warned_once;
    static std::unordered_set<std::string> s_resolved_logged_once;

    auto convert_metric = [&](NVPW_MetricsEvaluator* eval,
                              const char* metric_name,
                              NVPW_MetricEvalRequest* out_req) -> bool {
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params conv{};
        conv.structSize                  = NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE;
        conv.pMetricsEvaluator           = eval;
        conv.pMetricName                 = metric_name;
        conv.pMetricEvalRequest          = out_req;
        conv.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        return NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&conv) == NVPA_STATUS_SUCCESS;
    };

    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calc{};
    calc.structSize = NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE;
    calc.pChipName  = chip_name.c_str();
    if (NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calc) != NVPA_STATUS_SUCCESS) {
        std::cerr << "[cupti] resolve_raw: CalculateScratchBufferSize failed\n";
        return reqs;
    }

    std::vector<uint8_t> scratch(calc.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params ei{};
    ei.structSize        = NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE;
    ei.pScratchBuffer    = scratch.data();
    ei.scratchBufferSize = scratch.size();
    ei.pChipName         = chip_name.c_str();  // virtual-device mode — no counter data needed
    if (NVPW_CUDA_MetricsEvaluator_Initialize(&ei) != NVPA_STATUS_SUCCESS) {
        std::cerr << "[cupti] resolve_raw: MetricsEvaluator_Initialize (chip-only) failed\n";
        return reqs;
    }
    NVPW_MetricsEvaluator* evaluator = ei.pMetricsEvaluator;

    std::unordered_set<std::string> seen;
    s_resolved_metric_names.fill(std::string{});
    for (int i = 0; i < k_num_metrics; i++) {
        NVPW_MetricEvalRequest eval_req{};
        const char* chosen_metric = nullptr;
        for (int c = 0; c < 4 && k_metric_candidates[i][c]; c++) {
            if (convert_metric(evaluator, k_metric_candidates[i][c], &eval_req)) {
                chosen_metric = k_metric_candidates[i][c];
                break;
            }
        }
        if (!chosen_metric) {
            const std::string warn_key = std::string("slot:") + k_metric_labels[i];
            if (!s_metric_warned_once[warn_key]) {
                std::cerr << "[cupti] metric unsupported on this driver/GPU: "
                          << k_metric_labels[i] << " (all aliases failed, skipping)\n";
                s_metric_warned_once[warn_key] = true;
            }
            continue;
        }
        s_resolved_metric_names[i] = chosen_metric;

        // Get raw counter dependencies — two-call pattern: first get count, then fill.
        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params dep{};
        dep.structSize                  = NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE;
        dep.pMetricsEvaluator           = evaluator;
        dep.pMetricEvalRequests         = &eval_req;
        dep.numMetricEvalRequests       = 1;
        dep.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        dep.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        dep.ppRawDependencies           = nullptr;  // first call: query count
        dep.numRawDependencies          = 0;
        if (NVPW_MetricsEvaluator_GetMetricRawDependencies(&dep) != NVPA_STATUS_SUCCESS) {
            std::cerr << "[cupti] resolve_raw: GetMetricRawDependencies (count) failed for " << chosen_metric << "\n";
            continue;
        }

        size_t n = dep.numRawDependencies;
        std::vector<const char*> raw_ptrs(n);
        dep.ppRawDependencies  = raw_ptrs.data();  // second call: fill pointers
        dep.numRawDependencies = n;
        if (NVPW_MetricsEvaluator_GetMetricRawDependencies(&dep) != NVPA_STATUS_SUCCESS) {
            std::cerr << "[cupti] resolve_raw: GetMetricRawDependencies (fill) failed for " << chosen_metric << "\n";
            continue;
        }

        for (size_t j = 0; j < dep.numRawDependencies; j++) {
            std::string raw = raw_ptrs[j];
            if (seen.insert(raw).second)
                unique_raw_names.push_back(std::move(raw));
        }
    }

    NVPW_MetricsEvaluator_Destroy_Params d{};
    d.structSize        = NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE;
    d.pMetricsEvaluator = evaluator;
    NVPW_MetricsEvaluator_Destroy(&d);

    // Keep all strings stable first, then create requests that point to them.
    raw_name_storage.clear();
    raw_name_storage.reserve(unique_raw_names.size());
    for (auto& n : unique_raw_names)
        raw_name_storage.push_back(std::move(n));

    reqs.reserve(raw_name_storage.size());
    for (const auto& raw_name : raw_name_storage) {
        NVPA_RawMetricRequest r{};
        r.structSize    = NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE;
        r.pMetricName   = raw_name.c_str();
        r.isolated      = true;
        r.keepInstances = true;
        reqs.push_back(r);
    }

    if (s_resolved_logged_once.insert(chip_name).second) {
        int resolved_slots = 0;
        for (const auto& name : s_resolved_metric_names) {
            if (!name.empty()) resolved_slots++;
        }
        std::cerr << "[cupti] resolved " << reqs.size() << " raw counters for "
                  << resolved_slots << "/" << k_num_metrics << " logical metrics\n";
    }
    return reqs;
}

// ── Build config image ─────────────────────────────────────────────────────────
static bool build_config_image(const std::string& chip_name,
                               std::vector<uint8_t>& out)
{
    std::vector<std::string> raw_name_storage;
    std::vector<NVPA_RawMetricRequest> raw_reqs = resolve_raw_requests(chip_name, raw_name_storage);
    if (raw_reqs.empty()) {
        std::cerr << "[cupti] build_config: no raw counters resolved\n";
        return false;
    }

    NVPW_CUDA_RawMetricsConfig_Create_V2_Params create{};
    create.structSize    = NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE;
    create.activityKind  = NVPA_ACTIVITY_KIND_PROFILER;
    create.pChipName     = chip_name.c_str();
    NVPW_CHECK(NVPW_CUDA_RawMetricsConfig_Create_V2(&create));
    NVPA_RawMetricsConfig* cfg = create.pRawMetricsConfig;

    bool ok = false;
    do {
        NVPW_RawMetricsConfig_BeginPassGroup_Params bpg{};
        bpg.structSize        = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE;
        bpg.pRawMetricsConfig = cfg;
        bpg.maxPassCount      = 1;
        if (NVPW_RawMetricsConfig_BeginPassGroup(&bpg) != NVPA_STATUS_SUCCESS) break;

        // Add all resolved raw counters at once
        NVPW_RawMetricsConfig_AddMetrics_Params add{};
        add.structSize         = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE;
        add.pRawMetricsConfig  = cfg;
        add.pRawMetricRequests = raw_reqs.data();
        add.numMetricRequests  = raw_reqs.size();
        if (NVPW_RawMetricsConfig_AddMetrics(&add) != NVPA_STATUS_SUCCESS) {
            std::cerr << "[cupti] build_config: AddMetrics failed\n";
            break;
        }

        NVPW_RawMetricsConfig_EndPassGroup_Params epg{};
        epg.structSize        = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE;
        epg.pRawMetricsConfig = cfg;
        if (NVPW_RawMetricsConfig_EndPassGroup(&epg) != NVPA_STATUS_SUCCESS) break;

        NVPW_RawMetricsConfig_GenerateConfigImage_Params gen{};
        gen.structSize           = NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE;
        gen.pRawMetricsConfig    = cfg;
        gen.mergeAllPassGroups   = true;
        if (NVPW_RawMetricsConfig_GenerateConfigImage(&gen) != NVPA_STATUS_SUCCESS) break;

        NVPW_RawMetricsConfig_GetConfigImage_Params get{};
        get.structSize        = NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE;
        get.pRawMetricsConfig = cfg;
        get.bytesAllocated    = 0;
        get.pBuffer           = nullptr;
        if (NVPW_RawMetricsConfig_GetConfigImage(&get) != NVPA_STATUS_SUCCESS) break;
        out.resize(get.bytesCopied);
        get.bytesAllocated = get.bytesCopied;
        get.pBuffer        = out.data();
        if (NVPW_RawMetricsConfig_GetConfigImage(&get) != NVPA_STATUS_SUCCESS) break;

        ok = true;
    } while(false);

    NVPW_RawMetricsConfig_Destroy_Params destroy{};
    destroy.structSize        = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE;
    destroy.pRawMetricsConfig = cfg;
    NVPW_RawMetricsConfig_Destroy(&destroy);
    return ok;
}

// ── Build counter data prefix ──────────────────────────────────────────────────
static bool build_counter_data_prefix(const std::string& chip_name,
                                      std::vector<uint8_t>& out)
{
    std::vector<std::string> raw_name_storage;
    std::vector<NVPA_RawMetricRequest> raw_reqs = resolve_raw_requests(chip_name, raw_name_storage);
    if (raw_reqs.empty()) {
        std::cerr << "[cupti] build_prefix: no raw counters resolved\n";
        return false;
    }

    NVPW_CUDA_CounterDataBuilder_Create_Params create{};
    create.structSize = NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE;
    create.pChipName  = chip_name.c_str();
    NVPW_CHECK(NVPW_CUDA_CounterDataBuilder_Create(&create));
    NVPA_CounterDataBuilder* builder = create.pCounterDataBuilder;

    bool ok = false;
    do {
        NVPW_CounterDataBuilder_AddMetrics_Params add{};
        add.structSize          = NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE;
        add.pCounterDataBuilder = builder;
        add.pRawMetricRequests  = raw_reqs.data();
        add.numMetricRequests   = raw_reqs.size();
        if (NVPW_CounterDataBuilder_AddMetrics(&add) != NVPA_STATUS_SUCCESS) {
            std::cerr << "[cupti] build_prefix: AddMetrics failed\n";
            break;
        }

        // Query prefix size then copy
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params get{};
        get.structSize          = NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE;
        get.pCounterDataBuilder = builder;
        get.bytesAllocated      = 0;
        get.pBuffer             = nullptr;
        if (NVPW_CounterDataBuilder_GetCounterDataPrefix(&get) != NVPA_STATUS_SUCCESS) break;
        out.resize(get.bytesCopied);
        get.bytesAllocated = get.bytesCopied;
        get.pBuffer        = out.data();
        if (NVPW_CounterDataBuilder_GetCounterDataPrefix(&get) != NVPA_STATUS_SUCCESS) break;

        ok = true;
    } while(false);

    NVPW_CounterDataBuilder_Destroy_Params destroy{};
    destroy.structSize          = NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE;
    destroy.pCounterDataBuilder = builder;
    NVPW_CounterDataBuilder_Destroy(&destroy);
    return ok;
}

// ── Allocate + init counter data image ────────────────────────────────────────
static bool alloc_counter_data(const std::vector<uint8_t>& prefix,
                               std::vector<uint8_t>& image_out,
                               std::vector<uint8_t>& scratch_out)
{
    CUpti_Profiler_CounterDataImageOptions opts{};
    opts.structSize            = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    opts.pCounterDataPrefix    = prefix.data();
    opts.counterDataPrefixSize = prefix.size();
    opts.maxNumRanges          = 1;
    opts.maxNumRangeTreeNodes  = 1;
    opts.maxRangeNameLength    = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calc_size{};
    calc_size.structSize              = CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE;
    calc_size.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    calc_size.pOptions                = &opts;
    CUPTI_CHECK(cuptiProfilerCounterDataImageCalculateSize(&calc_size));
    image_out.resize(calc_size.counterDataImageSize);

    CUpti_Profiler_CounterDataImage_Initialize_Params img_init{};
    img_init.structSize                    = CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE;
    img_init.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    img_init.pOptions                      = &opts;
    img_init.counterDataImageSize          = image_out.size();
    img_init.pCounterDataImage             = image_out.data();
    CUPTI_CHECK(cuptiProfilerCounterDataImageInitialize(&img_init));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params calc_scratch{};
    calc_scratch.structSize           = CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE;
    calc_scratch.counterDataImageSize = image_out.size();
    calc_scratch.pCounterDataImage    = image_out.data();
    CUPTI_CHECK(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&calc_scratch));
    scratch_out.resize(calc_scratch.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params scratch_init{};
    scratch_init.structSize                   = CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE;
    scratch_init.counterDataImageSize         = image_out.size();
    scratch_init.pCounterDataImage            = image_out.data();
    scratch_init.counterDataScratchBufferSize = scratch_out.size();
    scratch_init.pCounterDataScratchBuffer    = scratch_out.data();
    CUPTI_CHECK(cuptiProfilerCounterDataImageInitializeScratchBuffer(&scratch_init));
    return true;
}

// ── Re-initialise counter data image (cheap reset between sessions) ───────────
static void reset_counter_data(std::vector<uint8_t>& image,
                                std::vector<uint8_t>& scratch,
                                const std::vector<uint8_t>& prefix)
{
    CUpti_Profiler_CounterDataImageOptions opts{};
    opts.structSize            = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    opts.pCounterDataPrefix    = prefix.data();
    opts.counterDataPrefixSize = prefix.size();
    opts.maxNumRanges          = 1;
    opts.maxNumRangeTreeNodes  = 1;
    opts.maxRangeNameLength    = 64;

    CUpti_Profiler_CounterDataImage_Initialize_Params img_init{};
    img_init.structSize                    = CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE;
    img_init.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    img_init.pOptions                      = &opts;
    img_init.counterDataImageSize          = image.size();
    img_init.pCounterDataImage             = image.data();
    cuptiProfilerCounterDataImageInitialize(&img_init);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params scratch_init{};
    scratch_init.structSize                   = CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE;
    scratch_init.counterDataImageSize         = image.size();
    scratch_init.pCounterDataImage            = image.data();
    scratch_init.counterDataScratchBufferSize = scratch.size();
    scratch_init.pCounterDataScratchBuffer    = scratch.data();
    cuptiProfilerCounterDataImageInitializeScratchBuffer(&scratch_init);
}

// ── Decode collected counter data ─────────────────────────────────────────────
static CuptiMetrics decode_metrics(const std::string& chip_name,
                                   const std::vector<uint8_t>& counter_data)
{
    CuptiMetrics out{};

    // Create metrics evaluator — using pCounterDataImage gives an actual-device evaluator
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calc{};
    calc.structSize  = NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE;
    calc.pChipName   = chip_name.c_str();
    if (NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calc) != NVPA_STATUS_SUCCESS) {
        std::cerr << "[cupti] decode: CalculateScratchBufferSize failed\n";
        return out;
    }

    std::vector<uint8_t> scratch(calc.scratchBufferSize);

    NVPW_CUDA_MetricsEvaluator_Initialize_Params eval_init{};
    eval_init.structSize            = NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE;
    eval_init.pScratchBuffer        = scratch.data();
    eval_init.scratchBufferSize     = scratch.size();
    eval_init.pCounterDataImage     = counter_data.data();
    eval_init.counterDataImageSize  = counter_data.size();
    if (NVPW_CUDA_MetricsEvaluator_Initialize(&eval_init) != NVPA_STATUS_SUCCESS) {
        std::cerr << "[cupti] decode: MetricsEvaluator_Initialize failed (counter data sz=" << counter_data.size() << ")\n";
        return out;
    }
    NVPW_MetricsEvaluator* evaluator = eval_init.pMetricsEvaluator;

    // Set device attributes from the counter data image
    NVPW_MetricsEvaluator_SetDeviceAttributes_Params dev_attr{};
    dev_attr.structSize           = NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE;
    dev_attr.pMetricsEvaluator    = evaluator;
    dev_attr.pCounterDataImage    = counter_data.data();
    dev_attr.counterDataImageSize = counter_data.size();
    if (NVPW_MetricsEvaluator_SetDeviceAttributes(&dev_attr) != NVPA_STATUS_SUCCESS) {
        std::cerr << "[cupti] decode: SetDeviceAttributes failed\n";
        NVPW_MetricsEvaluator_Destroy_Params d{};
        d.structSize = NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE;
        d.pMetricsEvaluator = evaluator;
        NVPW_MetricsEvaluator_Destroy(&d);
        return out;
    }

    // Convert metric names → MetricEvalRequests.
    // Track which output field each successfully-converted request maps to so we can
    // evaluate only valid requests and write results to the correct fields.
    NVPW_MetricEvalRequest valid_requests[k_num_metrics] = {};
    int                    valid_src[k_num_metrics]      = {};  // original metric index
    int                    valid_count                   = 0;

    auto convert_metric = [&](NVPW_MetricsEvaluator* eval,
                              const char* metric_name,
                              NVPW_MetricEvalRequest* out_req) -> bool {
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params conv{};
        conv.structSize                  = NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE;
        conv.pMetricsEvaluator           = eval;
        conv.pMetricName                 = metric_name;
        conv.pMetricEvalRequest          = out_req;
        conv.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        return NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&conv) == NVPA_STATUS_SUCCESS;
    };

    for (int i = 0; i < k_num_metrics; i++) {
        if (s_resolved_metric_names[i].empty()) continue;
        if (convert_metric(evaluator, s_resolved_metric_names[i].c_str(), &valid_requests[valid_count])) {
            valid_src[valid_count] = i;
            valid_count++;
        }
    }

    if (valid_count > 0) {
        double values[k_num_metrics] = {};
        NVPW_MetricsEvaluator_EvaluateToGpuValues_Params eval{};
        eval.structSize                  = NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE;
        eval.pMetricsEvaluator           = evaluator;
        eval.pMetricEvalRequests         = valid_requests;
        eval.numMetricEvalRequests       = (size_t)valid_count;   // only the valid ones
        eval.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        eval.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        eval.pCounterDataImage           = counter_data.data();
        eval.counterDataImageSize        = counter_data.size();
        eval.rangeIndex                  = 0;
        eval.isolated                    = true;
        eval.pMetricValues               = values;

        NVPA_Status eval_status = NVPW_MetricsEvaluator_EvaluateToGpuValues(&eval);
        if (eval_status == NVPA_STATUS_SUCCESS) {
            auto normalize_pct = [](double v) -> float {
                if (!std::isfinite(v) || v < 0.0) return 0.0f;
                // Some stacks report pct metrics as 0..1, others as 0..100.
                if (v <= 1.0) v *= 100.0;
                if (v > 100.0) v = 100.0;
                return (float)v;
            };
            // Map each evaluated value back to its output field by original index.
            float* fields[k_num_metrics] = {
                &out.cuda_active_pct,   // 0 — FP32 ALU
                &out.tex_active_pct,    // 1 — TEX
                &out.ldst_active_pct,   // 2 — LD/ST
                &out.sfu_active_pct,    // 3 — SFU
                &out.tensor_active_pct, // 4 — Tensor
            };
            for (int i = 0; i < valid_count; i++)
                *fields[valid_src[i]] = normalize_pct(values[i]);
            out.valid = true;
        } else {
            std::cerr << "[cupti] decode: EvaluateToGpuValues failed: " << (int)eval_status << "\n";
        }
    } else {
        std::cerr << "[cupti] decode: no valid metric eval requests\n";
    }

    NVPW_MetricsEvaluator_Destroy_Params destroy{};
    destroy.structSize      = NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE;
    destroy.pMetricsEvaluator = evaluator;
    NVPW_MetricsEvaluator_Destroy(&destroy);
    return out;
}

// ── CuptiProfiler::init ────────────────────────────────────────────────────────
bool CuptiProfiler::init(int device_index)
{
    m_device = device_index;
    m_impl   = new Impl();

    // 1. Init CUPTI profiler interface
    CUpti_Profiler_Initialize_Params cupti_init{};
    cupti_init.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
    CUPTI_CHECK(cuptiProfilerInitialize(&cupti_init));

    // 2. Get chip name (e.g. "ad102" for RTX 4090)
    CUpti_Device_GetChipName_Params chip_params{};
    chip_params.structSize  = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
    chip_params.deviceIndex = (uint32_t)device_index;
    CUPTI_CHECK(cuptiDeviceGetChipName(&chip_params));
    m_impl->chip_name = chip_params.pChipName;
    for (char& c : m_impl->chip_name) c = (char)tolower((unsigned char)c);
    printf("[cupti] chip: %s\n", m_impl->chip_name.c_str());

    // 3. Init NvPerf host library
    NVPW_InitializeHost_Params nvpw_init{};
    nvpw_init.structSize = NVPW_InitializeHost_Params_STRUCT_SIZE;
    NVPW_CHECK(NVPW_InitializeHost(&nvpw_init));

    // 4. Build config image
    if (!build_config_image(m_impl->chip_name, m_impl->config_image)) {
        std::cerr << "[cupti] failed to build config image\n";
        return false;
    }

    // 5. Build counter data prefix
    if (!build_counter_data_prefix(m_impl->chip_name, m_impl->counter_data_prefix)) {
        std::cerr << "[cupti] failed to build counter data prefix\n";
        return false;
    }

    // 6. Allocate counter data image (1 range per session)
    if (!alloc_counter_data(m_impl->counter_data_prefix,
                            m_impl->counter_data_image,
                            m_impl->counter_data_scratch)) {
        std::cerr << "[cupti] failed to allocate counter data image\n";
        return false;
    }

    m_ok = true;
    printf("[cupti] profiler ready (%d metrics, 1 range/session)\n", k_num_metrics);
    return true;
}

// ── CuptiProfiler::shutdown ────────────────────────────────────────────────────
void CuptiProfiler::shutdown()
{
    if (!m_impl) return;

    if (m_impl->range_open) {
        CUpti_Profiler_PopRange_Params pop{};
        pop.structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE;
        cuptiProfilerPopRange(&pop);
        m_impl->range_open = false;
    }
    if (m_impl->pass_open) {
        CUpti_Profiler_DisableProfiling_Params dis{};
        dis.structSize = CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE;
        cuptiProfilerDisableProfiling(&dis);

        CUpti_Profiler_EndPass_Params ep{};
        ep.structSize = CUpti_Profiler_EndPass_Params_STRUCT_SIZE;
        cuptiProfilerEndPass(&ep);
        m_impl->pass_open = false;
    }
    if (m_impl->session_open) {
        CUpti_Profiler_UnsetConfig_Params uc{};
        uc.structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE;
        cuptiProfilerUnsetConfig(&uc);

        CUpti_Profiler_EndSession_Params es{};
        es.structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE;
        cuptiProfilerEndSession(&es);
        m_impl->session_open = false;
    }

    CUpti_Profiler_DeInitialize_Params deinit{};
    deinit.structSize = CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE;
    cuptiProfilerDeInitialize(&deinit);

    delete m_impl;
    m_impl = nullptr;
    m_ok   = false;
}

// ── CuptiProfiler::begin_frame ────────────────────────────────────────────────
void CuptiProfiler::begin_frame()
{
    if (!m_ok) return;
    // Only open a session on every k_sample_interval-th frame.
    // m_capturing is set here and cleared in end_frame — both run the SAME frame.
    if (m_frame % k_sample_interval != 0) return;
    // Reset counter data for a fresh session
    reset_counter_data(m_impl->counter_data_image,
                       m_impl->counter_data_scratch,
                       m_impl->counter_data_prefix);

    // Begin session
    CUpti_Profiler_BeginSession_Params bs{};
    bs.structSize                   = CUpti_Profiler_BeginSession_Params_STRUCT_SIZE;
    bs.ctx                          = nullptr;
    bs.counterDataImageSize         = m_impl->counter_data_image.size();
    bs.pCounterDataImage            = m_impl->counter_data_image.data();
    bs.counterDataScratchBufferSize = m_impl->counter_data_scratch.size();
    bs.pCounterDataScratchBuffer    = m_impl->counter_data_scratch.data();
    bs.range                        = CUPTI_UserRange;
    bs.replayMode                   = CUPTI_UserReplay;
    bs.maxRangesPerPass             = 1;
    bs.maxLaunchesPerPass           = 64;
    {
        CUptiResult r = cuptiProfilerBeginSession(&bs);
        if (r != CUPTI_SUCCESS) {
            const char* msg = nullptr; cuptiGetResultString(r, &msg);
            std::cerr << "[cupti] begin_frame: BeginSession failed: " << (msg?msg:"?") << " (" << (int)r << ")\n";
            if (r == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES)
                std::cerr << "[cupti] Fix: run as Administrator, or set HKLM\\SYSTEM\\CurrentControlSet\\Services\\nvlddmkm\\Global\\NVreg_RestrictProfilingToAdminUsers=0 and reboot\n";
            m_ok = false;  // stop retrying — caller will see is_initialized()==false
            return;
        }
    }
    m_impl->session_open = true;

    // Set config
    CUpti_Profiler_SetConfig_Params sc{};
    sc.structSize         = CUpti_Profiler_SetConfig_Params_STRUCT_SIZE;
    sc.pConfig            = m_impl->config_image.data();
    sc.configSize         = m_impl->config_image.size();
    sc.minNestingLevel    = 1;
    sc.numNestingLevels   = 1;
    sc.passIndex          = 0;
    sc.targetNestingLevel = 1;
    {
        CUptiResult r = cuptiProfilerSetConfig(&sc);
        if (r != CUPTI_SUCCESS) {
            const char* msg = nullptr; cuptiGetResultString(r, &msg);
            std::cerr << "[cupti] begin_frame: SetConfig failed: " << (msg?msg:"?") << " (" << (int)r << ")\n";
            return;
        }
    }

    // Begin pass
    CUpti_Profiler_BeginPass_Params bp{};
    bp.structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE;
    {
        CUptiResult r = cuptiProfilerBeginPass(&bp);
        if (r != CUPTI_SUCCESS) {
            const char* msg = nullptr; cuptiGetResultString(r, &msg);
            std::cerr << "[cupti] begin_frame: BeginPass failed: " << (msg?msg:"?") << " (" << (int)r << ")\n";
            return;
        }
    }
    m_impl->pass_open = true;

    // Enable profiling + push range
    CUpti_Profiler_EnableProfiling_Params ep{};
    ep.structSize = CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE;
    {
        CUptiResult r = cuptiProfilerEnableProfiling(&ep);
        if (r != CUPTI_SUCCESS) {
            const char* msg = nullptr; cuptiGetResultString(r, &msg);
            std::cerr << "[cupti] begin_frame: EnableProfiling failed: " << (msg?msg:"?") << " (" << (int)r << ")\n";
            return;
        }
    }

    CUpti_Profiler_PushRange_Params pr{};
    pr.structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE;
    pr.pRangeName = "frame";
    {
        CUptiResult r = cuptiProfilerPushRange(&pr);
        if (r != CUPTI_SUCCESS) {
            const char* msg = nullptr; cuptiGetResultString(r, &msg);
            std::cerr << "[cupti] begin_frame: PushRange failed: " << (msg?msg:"?") << " (" << (int)r << ")\n";
            return;
        }
    }
    m_impl->range_open = true;
    m_capturing = true;  // signal end_frame to close this session
}

// ── CuptiProfiler::end_frame ──────────────────────────────────────────────────
void CuptiProfiler::end_frame()
{
    if (!m_ok) return;

    // Only close the session we opened in begin_frame this same frame.
    bool do_collect = m_capturing;
    m_capturing = false;
    m_frame++;

    if (!do_collect) return;
    if (!m_impl->session_open || !m_impl->pass_open) return;

    // Pop range + disable
    if (m_impl->range_open) {
        CUpti_Profiler_PopRange_Params pop{};
        pop.structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE;
        cuptiProfilerPopRange(&pop);
        m_impl->range_open = false;
    }

    CUpti_Profiler_DisableProfiling_Params dis{};
    dis.structSize = CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE;
    cuptiProfilerDisableProfiling(&dis);

    // EndPass — check allPassesSubmitted; if 0, GPU needs more passes (shouldn't
    // happen for 5 SM-pipe counters on Ada but handle gracefully).
    CUpti_Profiler_EndPass_Params ep{};
    ep.structSize = CUpti_Profiler_EndPass_Params_STRUCT_SIZE;
    if (cuptiProfilerEndPass(&ep) != CUPTI_SUCCESS) {
        std::cerr << "[cupti] EndPass failed\n";
        // close session anyway to avoid leaking
        CUpti_Profiler_UnsetConfig_Params uc{}; uc.structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE; cuptiProfilerUnsetConfig(&uc);
        CUpti_Profiler_EndSession_Params es{}; es.structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE; cuptiProfilerEndSession(&es);
        m_impl->pass_open = m_impl->session_open = false;
        return;
    }
    m_impl->pass_open = false;

    if (!ep.allPassesSubmitted) {
        // Multi-pass required — metrics not collected this interval; try again next time.
        // (All 5 SM-pipe counters fit in a single pass on Ada/Ampere, so this shouldn't happen.)
        std::cerr << "[cupti] allPassesSubmitted=0 — multi-pass required, skipping decode\n";
        CUpti_Profiler_UnsetConfig_Params uc{}; uc.structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE; cuptiProfilerUnsetConfig(&uc);
        CUpti_Profiler_EndSession_Params es{}; es.structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE; cuptiProfilerEndSession(&es);
        m_impl->session_open = false;
        return;
    }

    // Flush counter data to CPU-visible memory
    CUpti_Profiler_FlushCounterData_Params flush{};
    flush.structSize = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE;
    if (cuptiProfilerFlushCounterData(&flush) != CUPTI_SUCCESS) {
        std::cerr << "[cupti] FlushCounterData failed\n";
    }

    // Ensure GPU has finished writing counter data before we decode.
    cudaDeviceSynchronize();

    // End session
    CUpti_Profiler_UnsetConfig_Params uc{};
    uc.structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE;
    cuptiProfilerUnsetConfig(&uc);

    CUpti_Profiler_EndSession_Params es{};
    es.structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE;
    cuptiProfilerEndSession(&es);
    m_impl->session_open = false;

    // Decode. Keep last valid sample if this decode attempt fails.
    CuptiMetrics decoded = decode_metrics(m_impl->chip_name, m_impl->counter_data_image);
    if (decoded.valid)
        m_metrics = decoded;
}

#endif // CUPTI_ENABLED
