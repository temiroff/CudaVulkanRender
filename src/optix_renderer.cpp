#include "optix_renderer.h"

#include <cuda.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef OPTIX_ENABLED
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

namespace {

struct OptixLaunchParams {
    float4*             accum_buffer;
    int                 width;
    int                 height;
    Camera              cam;
    Triangle*           triangles;
    int                 num_triangles;
    GpuMaterial*        gpu_materials;
    int                 num_gpu_materials;
    cudaTextureObject_t* textures;
    int                 num_textures;
    int                 frame_count;
    int                 spp;
    int                 max_depth;
    int                 color_mode;
    cudaTextureObject_t hdri_tex;
    float               hdri_intensity;
    float               hdri_yaw;
    float               hdri_bg_blur;
    float               hdri_bg_opacity;
    float3              bg_color;
    float               firefly_clamp;
    OptixTraversableHandle traversable;
};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T    data;
};

struct EmptyRecordData {};

static void set_error(OptixRendererState& s, const std::string& msg)
{
    std::snprintf(s.last_error, sizeof(s.last_error), "%s", msg.c_str());
}

static std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void optix_log_cb(unsigned int level, const char* tag, const char* msg, void*)
{
    std::fprintf(stderr, "[optix][%u][%s] %s\n", level, tag ? tag : "?", msg ? msg : "");
}

static bool compile_optix_ptx(OptixRendererState& s, std::string& out_ptx)
{
    const std::filesystem::path src_path = std::filesystem::path(__FILE__).parent_path() / "optix_renderer_device.cu";
    printf("[optix_rt] Compiling %s via nvrtc...\n", src_path.string().c_str());
    fflush(stdout);
    const std::string src = read_text_file(src_path);
    if (src.empty()) {
        set_error(s, std::string("failed to read device source: ") + src_path.string());
        return false;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const std::string arch = std::string("--gpu-architecture=compute_") +
                             std::to_string(prop.major) + std::to_string(prop.minor);

    std::vector<std::string> opt_strings = {
        arch,
        "--std=c++17",
        "--use_fast_math",
        "--device-as-default-execution-space",
        "--relocatable-device-code=true",
        "-I" + std::filesystem::path(__FILE__).parent_path().string(),
        "-I" OPTIX_INCLUDE_DIR,
        "-I" CUDA_INCLUDE_DIR,
    };
    std::vector<const char*> opts;
    opts.reserve(opt_strings.size());
    for (const std::string& opt : opt_strings)
        opts.push_back(opt.c_str());

    nvrtcProgram prog = nullptr;
    nvrtcResult nvr = nvrtcCreateProgram(&prog, src.c_str(), src_path.filename().string().c_str(), 0, nullptr, nullptr);
    if (nvr != NVRTC_SUCCESS) {
        set_error(s, std::string("nvrtcCreateProgram failed: ") + nvrtcGetErrorString(nvr));
        return false;
    }

    nvr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::string log;
    if (log_size > 1) {
        log.resize(log_size - 1);
        nvrtcGetProgramLog(prog, log.data());
    }
    if (nvr != NVRTC_SUCCESS) {
        set_error(s, log.empty() ? std::string("nvrtcCompileProgram failed: ") + nvrtcGetErrorString(nvr) : log);
        printf("[optix_rt] nvrtc compile failed: %s\n", s.last_error);
        fflush(stdout);
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptx_size = 0;
    nvrtcGetPTXSize(prog, &ptx_size);
    out_ptx.resize(ptx_size);
    nvrtcGetPTX(prog, out_ptx.data());
    nvrtcDestroyProgram(&prog);
    printf("[optix_rt] nvrtc OK — PTX size %zu bytes\n", ptx_size);
    fflush(stdout);
    return true;
}

static bool create_pipeline(OptixRendererState& s)
{
    std::string ptx;
    if (!compile_optix_ptx(s, ptx))
        return false;

    OptixDeviceContext ctx = static_cast<OptixDeviceContext>(s.ctx);

    OptixModuleCompileOptions module_opts = {};
    module_opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipeline_opts = {};
    pipeline_opts.usesMotionBlur = 0;
    pipeline_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_opts.numPayloadValues = 2;
    pipeline_opts.numAttributeValues = 2;
    pipeline_opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_opts.pipelineLaunchParamsVariableName = "params";
    pipeline_opts.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[4096] = {};
    size_t log_size = sizeof(log);
    OptixModule module = nullptr;
    OptixResult res = optixModuleCreate(
        ctx,
        &module_opts,
        &pipeline_opts,
        ptx.c_str(),
        ptx.size(),
        log,
        &log_size,
        &module);
    if (res != OPTIX_SUCCESS) {
        printf("[optix_rt] optixModuleCreate failed (res=%d): %s\n", (int)res, log);
        fflush(stdout);
        set_error(s, std::string("optixModuleCreate failed: ") + log);
        return false;
    }
    printf("[optix_rt] Module created OK.\n");
    fflush(stdout);
    s.module = module;

    OptixProgramGroupOptions pg_opts = {};
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module;
    raygen_desc.raygen.entryFunctionName = "__raygen__rg";

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module;
    miss_desc.miss.entryFunctionName = "__miss__ms";

    OptixProgramGroupDesc hit_desc = {};
    hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_desc.hitgroup.moduleCH = module;
    hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    log_size = sizeof(log);
    OptixProgramGroup raygen_pg = nullptr;
    res = optixProgramGroupCreate(ctx, &raygen_desc, 1, &pg_opts, log, &log_size, &raygen_pg);
    if (res != OPTIX_SUCCESS) {
        set_error(s, std::string("optixProgramGroupCreate(raygen) failed: ") + log);
        return false;
    }
    s.raygen_pg = raygen_pg;

    log_size = sizeof(log);
    OptixProgramGroup miss_pg = nullptr;
    res = optixProgramGroupCreate(ctx, &miss_desc, 1, &pg_opts, log, &log_size, &miss_pg);
    if (res != OPTIX_SUCCESS) {
        set_error(s, std::string("optixProgramGroupCreate(miss) failed: ") + log);
        return false;
    }
    s.miss_pg = miss_pg;

    log_size = sizeof(log);
    OptixProgramGroup hit_pg = nullptr;
    res = optixProgramGroupCreate(ctx, &hit_desc, 1, &pg_opts, log, &log_size, &hit_pg);
    if (res != OPTIX_SUCCESS) {
        set_error(s, std::string("optixProgramGroupCreate(hitgroup) failed: ") + log);
        return false;
    }
    s.hitgroup_pg = hit_pg;

    OptixProgramGroup groups[] = { raygen_pg, miss_pg, hit_pg };
    OptixPipelineLinkOptions link_opts = {};
    link_opts.maxTraceDepth = 1;
    link_opts.maxTraversableGraphDepth = 1;

    log_size = sizeof(log);
    OptixPipeline pipeline = nullptr;
    res = optixPipelineCreate(ctx, &pipeline_opts, &link_opts, groups, 3, log, &log_size, &pipeline);
    if (res != OPTIX_SUCCESS) {
        set_error(s, std::string("optixPipelineCreate failed: ") + log);
        return false;
    }
    s.pipeline = pipeline;

    OptixStackSizes stack_sizes = {};
    optixUtilAccumulateStackSizes(raygen_pg, &stack_sizes, pipeline);
    optixUtilAccumulateStackSizes(miss_pg, &stack_sizes, pipeline);
    optixUtilAccumulateStackSizes(hit_pg, &stack_sizes, pipeline);

    unsigned int dc_stack_from_traversal = 0;
    unsigned int dc_stack_from_state = 0;
    unsigned int cc_stack = 0;
    unsigned int continuation_stack = 0;
    optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0,
                               &dc_stack_from_traversal,
                               &dc_stack_from_state,
                               &continuation_stack);
    res = optixPipelineSetStackSize(
        pipeline,
        dc_stack_from_traversal,
        dc_stack_from_state,
        continuation_stack,
        1);
    if (res != OPTIX_SUCCESS) {
        set_error(s, "optixPipelineSetStackSize failed");
        return false;
    }

    SbtRecord<EmptyRecordData> rg = {};
    SbtRecord<EmptyRecordData> ms = {};
    SbtRecord<EmptyRecordData> hg = {};
    optixSbtRecordPackHeader(raygen_pg, &rg);
    optixSbtRecordPackHeader(miss_pg, &ms);
    optixSbtRecordPackHeader(hit_pg, &hg);

    cudaMalloc(&s.d_sbt_raygen, sizeof(rg));
    cudaMalloc(&s.d_sbt_miss, sizeof(ms));
    cudaMalloc(&s.d_sbt_hitgroup, sizeof(hg));
    cudaMemcpy(s.d_sbt_raygen, &rg, sizeof(rg), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_sbt_miss, &ms, sizeof(ms), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_sbt_hitgroup, &hg, sizeof(hg), cudaMemcpyHostToDevice);
    cudaMalloc(&s.d_launch_params, sizeof(OptixLaunchParams));
    return true;
}

static void free_scene_buffers(OptixRendererState& s)
{
    cudaFree(s.d_vertices); s.d_vertices = nullptr;
    cudaFree(s.d_indices); s.d_indices = nullptr;
    cudaFree(s.d_gas_buffer); s.d_gas_buffer = nullptr;
    s.gas_handle = 0;
    s.scene_ready = false;
    s.num_triangles = 0;
}

} // namespace
#endif

bool optix_renderer_init(OptixRendererState& s, int width, int height)
{
#ifdef OPTIX_ENABLED
    s.width = width;
    s.height = height;
    if (s.initialized)
        return s.available;

    optix_renderer_free(s);

    if (optixInit() != OPTIX_SUCCESS) {
        set_error(s, "optixInit failed");
        return false;
    }

    CUcontext cu_ctx = nullptr;
    cuCtxGetCurrent(&cu_ctx);
    if (!cu_ctx) {
        set_error(s, "no active CUDA context for OptiX renderer");
        return false;
    }

    OptixDeviceContextOptions ctx_opts = {};
    ctx_opts.logCallbackFunction = optix_log_cb;
    ctx_opts.logCallbackLevel = 4;
    OptixDeviceContext ctx = nullptr;
    if (optixDeviceContextCreate(cu_ctx, &ctx_opts, &ctx) != OPTIX_SUCCESS) {
        set_error(s, "optixDeviceContextCreate failed");
        return false;
    }
    s.ctx = ctx;

    if (!create_pipeline(s)) {
        // Preserve the error message before free() resets the struct.
        char saved_err[512] = {};
        std::snprintf(saved_err, sizeof(saved_err), "%s", s.last_error);
        printf("[optix_rt] create_pipeline failed: %s\n", saved_err[0] ? saved_err : "(no message)");
        fflush(stdout);
        optix_renderer_free(s);
        std::snprintf(s.last_error, sizeof(s.last_error), "%s", saved_err[0] ? saved_err : "create_pipeline failed");
        return false;
    }

    s.available = true;
    s.initialized = true;
    printf("[optix_rt] Pipeline ready.\n");
    fflush(stdout);
    return true;
#else
    (void)s; (void)width; (void)height;
    return false;
#endif
}

bool optix_renderer_upload_scene(OptixRendererState& s, const std::vector<Triangle>& tris,
                                 unsigned long long scene_version)
{
#ifdef OPTIX_ENABLED
    if (!s.initialized && !optix_renderer_init(s, s.width, s.height))
        return false;

    if (tris.empty()) {
        free_scene_buffers(s);
        s.scene_version = scene_version;
        return false;
    }

    if (s.scene_ready && s.scene_version == scene_version && s.num_triangles == (int)tris.size())
        return true;

    free_scene_buffers(s);

    std::vector<float3> vertices;
    std::vector<uint3> indices;
    vertices.reserve(tris.size() * 3);
    indices.reserve(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        const Triangle& tri = tris[i];
        vertices.push_back(tri.v0);
        vertices.push_back(tri.v1);
        vertices.push_back(tri.v2);
        indices.push_back(make_uint3((unsigned)(i * 3 + 0),
                                     (unsigned)(i * 3 + 1),
                                     (unsigned)(i * 3 + 2)));
    }

    cudaMalloc(&s.d_vertices, vertices.size() * sizeof(float3));
    cudaMalloc(&s.d_indices, indices.size() * sizeof(uint3));
    cudaMemcpy(s.d_vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_indices, indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice);

    CUdeviceptr vertex_buffer = reinterpret_cast<CUdeviceptr>(s.d_vertices);
    CUdeviceptr index_buffer = reinterpret_cast<CUdeviceptr>(s.d_indices);
    unsigned int flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.numVertices = (unsigned int)vertices.size();
    build_input.triangleArray.vertexBuffers = &vertex_buffer;
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets = (unsigned int)indices.size();
    build_input.triangleArray.indexBuffer = index_buffer;
    build_input.triangleArray.flags = flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OptixDeviceContext ctx = static_cast<OptixDeviceContext>(s.ctx);
    OptixResult res = optixAccelComputeMemoryUsage(ctx, &accel_opts, &build_input, 1, &sizes);
    if (res != OPTIX_SUCCESS) {
        set_error(s, "optixAccelComputeMemoryUsage failed");
        free_scene_buffers(s);
        return false;
    }

    CUdeviceptr temp_buffer = 0;
    cudaMalloc(reinterpret_cast<void**>(&temp_buffer), sizes.tempSizeInBytes);
    cudaMalloc(&s.d_gas_buffer, sizes.outputSizeInBytes);

    OptixTraversableHandle gas = 0;
    res = optixAccelBuild(
        ctx,
        0,
        &accel_opts,
        &build_input,
        1,
        temp_buffer,
        sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(s.d_gas_buffer),
        sizes.outputSizeInBytes,
        &gas,
        nullptr,
        0);
    cudaFree(reinterpret_cast<void*>(temp_buffer));
    if (res != OPTIX_SUCCESS) {
        set_error(s, "optixAccelBuild failed");
        free_scene_buffers(s);
        return false;
    }

    s.gas_handle = gas;
    s.scene_version = scene_version;
    s.num_triangles = (int)tris.size();
    s.scene_ready = true;
    s.available = true;
    return true;
#else
    (void)s; (void)tris; (void)scene_version;
    return false;
#endif
}

bool optix_renderer_render(OptixRendererState& s, const PathTracerParams& p, cudaStream_t stream)
{
#ifdef OPTIX_ENABLED
    if (!s.initialized || !s.scene_ready || !s.pipeline)
        return false;

    OptixLaunchParams launch = {};
    launch.accum_buffer = p.accum_buffer;
    launch.width = p.width;
    launch.height = p.height;
    launch.cam = p.cam;
    launch.triangles = p.triangles;
    launch.num_triangles = p.num_triangles;
    launch.gpu_materials = p.gpu_materials;
    launch.num_gpu_materials = p.num_gpu_materials;
    launch.textures = p.textures;
    launch.num_textures = p.num_textures;
    launch.frame_count = p.frame_count;
    launch.spp = p.spp;
    launch.max_depth = p.max_depth;
    launch.color_mode = p.color_mode;
    launch.hdri_tex = p.hdri_tex;
    launch.hdri_intensity = p.hdri_intensity;
    launch.hdri_yaw = p.hdri_yaw;
    launch.hdri_bg_blur = p.hdri_bg_blur;
    launch.hdri_bg_opacity = p.hdri_bg_opacity;
    launch.bg_color = p.bg_color;
    launch.firefly_clamp = p.firefly_clamp;
    launch.traversable = static_cast<OptixTraversableHandle>(s.gas_handle);

    cudaMemcpyAsync(s.d_launch_params, &launch, sizeof(launch), cudaMemcpyHostToDevice, stream);

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(s.d_sbt_raygen);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(s.d_sbt_miss);
    sbt.missRecordStrideInBytes = sizeof(SbtRecord<EmptyRecordData>);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(s.d_sbt_hitgroup);
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<EmptyRecordData>);
    sbt.hitgroupRecordCount = 1;

    OptixResult res = optixLaunch(
        static_cast<OptixPipeline>(s.pipeline),
        stream,
        reinterpret_cast<CUdeviceptr>(s.d_launch_params),
        sizeof(launch),
        &sbt,
        (unsigned int)p.width,
        (unsigned int)p.height,
        1);
    if (res != OPTIX_SUCCESS) {
        set_error(s, "optixLaunch failed");
        return false;
    }
    return true;
#else
    (void)s; (void)p; (void)stream;
    return false;
#endif
}

void optix_renderer_resize(OptixRendererState& s, int width, int height)
{
    s.width = width;
    s.height = height;
}

void optix_renderer_free(OptixRendererState& s)
{
#ifdef OPTIX_ENABLED
    free_scene_buffers(s);
    cudaFree(s.d_launch_params); s.d_launch_params = nullptr;
    cudaFree(s.d_sbt_raygen); s.d_sbt_raygen = nullptr;
    cudaFree(s.d_sbt_miss); s.d_sbt_miss = nullptr;
    cudaFree(s.d_sbt_hitgroup); s.d_sbt_hitgroup = nullptr;

    if (s.hitgroup_pg) optixProgramGroupDestroy(static_cast<OptixProgramGroup>(s.hitgroup_pg));
    if (s.miss_pg) optixProgramGroupDestroy(static_cast<OptixProgramGroup>(s.miss_pg));
    if (s.raygen_pg) optixProgramGroupDestroy(static_cast<OptixProgramGroup>(s.raygen_pg));
    if (s.pipeline) optixPipelineDestroy(static_cast<OptixPipeline>(s.pipeline));
    if (s.module) optixModuleDestroy(static_cast<OptixModule>(s.module));
    if (s.ctx) optixDeviceContextDestroy(static_cast<OptixDeviceContext>(s.ctx));
#endif
    s = OptixRendererState{};
}

#ifndef OPTIX_ENABLED
bool optix_renderer_init(OptixRendererState&, int, int) { return false; }
bool optix_renderer_upload_scene(OptixRendererState&, const std::vector<Triangle>&, unsigned long long) { return false; }
bool optix_renderer_render(OptixRendererState&, const PathTracerParams&, cudaStream_t) { return false; }
void optix_renderer_resize(OptixRendererState&, int, int) {}
void optix_renderer_free(OptixRendererState&) {}
#endif

