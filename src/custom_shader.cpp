#include "custom_shader.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
//  Default starter code
// ─────────────────────────────────────────────────────────────────────────────

const char* CUSTOM_SHADER_DEFAULT_CODE = R"(
// MatOut — full material output.  All fields are pre-defined; just fill them.
// Available: float2/3/4, make_float2/3/4, sinf, cosf, sqrtf, powf, fabs,
//            fminf, fmaxf, floorf, ceilf, fmodf, fracf(x)=x-floorf(x)

__device__ MatOut custom_material(float2 uv, float3 pos, float3 normal, int frame)
{
    MatOut m;

    // Checkerboard
    int cx = (int)(uv.x * 8.f);
    int cy = (int)(uv.y * 8.f);
    float c = ((cx + cy) % 2 == 0) ? 1.f : 0.15f;

    m.base_color = make_float3(c * 0.9f, c * 0.5f, c * 0.2f);
    m.roughness  = 0.4f;
    m.metallic   = 0.0f;
    m.emissive   = make_float3(0.f, 0.f, 0.f);   // no glow
    m.normal_ts  = make_float3(0.f, 0.f, 1.f);   // flat

    return m;
}
)";

// ─────────────────────────────────────────────────────────────────────────────
//  Kernel wrapper
// ─────────────────────────────────────────────────────────────────────────────

static const char* KERNEL_WRAPPER = R"(
// ── MatOut struct ─────────────────────────────────────────────────────────────
struct MatOut {
    float3 base_color;   // [0,1]   albedo / diffuse
    float  roughness;    // [0,1]   0=mirror, 1=matte
    float  metallic;     // [0,1]   0=dielectric, 1=metal
    float3 emissive;     // [0,∞)   HDR emissive glow; (0,0,0) = none
    float3 normal_ts;    // tangent-space normal — (0,0,1) = flat (no bump)
};

// ── User code ─────────────────────────────────────────────────────────────────
%USER_CODE%

// ── Driver kernel ─────────────────────────────────────────────────────────────
extern "C" __global__
void gen_tex_kernel(
    float4* __restrict__ base_out,
    float4* __restrict__ mr_out,
    float4* __restrict__ emis_out,
    float4* __restrict__ norm_out,
    int w, int h, int frame)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    float2 uv     = make_float2((float)x / (float)w, (float)y / (float)h);
    float3 pos    = make_float3(0.f, 0.f, 0.f);
    float3 normal = make_float3(0.f, 1.f, 0.f);

    MatOut m = custom_material(uv, pos, normal, frame);

    // Clamp helpers
    #define C01(v) (fmaxf(0.f, fminf(1.f, (v))))

    int idx = y * w + x;

    // Base color — linear RGB, fully opaque
    base_out[idx] = make_float4(C01(m.base_color.x), C01(m.base_color.y), C01(m.base_color.z), 1.f);

    // Metallic-roughness — glTF layout: G=roughness, B=metallic
    mr_out[idx] = make_float4(0.f, C01(m.roughness), C01(m.metallic), 1.f);

    // Emissive — HDR allowed (clamped to [0,1] here; set mat.emissive_factor for scale)
    emis_out[idx] = make_float4(C01(m.emissive.x), C01(m.emissive.y), C01(m.emissive.z), 1.f);

    // Normal — normalize tangent-space, encode to [0,1]
    float3 n = m.normal_ts;
    float  nl = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (nl > 1e-4f) { n.x /= nl; n.y /= nl; n.z /= nl; }
    else            { n = make_float3(0.f, 0.f, 1.f); }
    norm_out[idx] = make_float4(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, 1.f);

    #undef C01
}
)";

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Strip HLSL/Slang semantic annotations (": POSITION", ": SV_Target", etc.)
// from struct member declarations. These are rasterization-only and invalid in CUDA.
static std::string strip_hlsl_semantics(const std::string& src)
{
    // Strips HLSL/Slang semantic annotations from struct member declarations:
    //   ": POSITION"    ": TEXCOORD0"   (all-caps)
    //   ": SV_Position" ": SV_Target"   (SV_ system-value prefix, mixed case)
    std::string out;
    out.reserve(src.size());
    size_t i = 0;
    while (i < src.size()) {
        if (src[i] == ':') {
            size_t j = i + 1;
            while (j < src.size() && (src[j] == ' ' || src[j] == '\t')) ++j;
            if (j < src.size() && (isupper((unsigned char)src[j]) || src[j] == '_')) {
                // Scan full identifier
                size_t k = j;
                while (k < src.size() && (isalnum((unsigned char)src[k]) || src[k] == '_')) ++k;
                // Accept if all-uppercase/digits/underscores OR starts with SV_
                bool all_upper = true;
                for (size_t ci = j; ci < k; ++ci)
                    if (islower((unsigned char)src[ci])) { all_upper = false; break; }
                bool sv_prefix = (k - j >= 3 &&
                                  src[j]=='S' && src[j+1]=='V' && src[j+2]=='_');
                if (all_upper || sv_prefix) {
                    size_t m2 = k;
                    while (m2 < src.size() && (src[m2] == ' ' || src[m2] == '\t')) ++m2;
                    if (m2 < src.size() && (src[m2] == ';' || src[m2] == ',' ||
                                            src[m2] == ')' || src[m2] == '\n')) {
                        i = k;
                        continue;
                    }
                }
            }
        }
        out += src[i++];
    }
    return out;
}

static std::string build_source(const std::string& user_code)
{
    std::string code = strip_hlsl_semantics(user_code);

    // Backward compatibility: if the user's function returns float4 (old API),
    // rename it to _cm_legacy and wrap it in a MatOut adapter so old shaders
    // still compile without modification.
    bool legacy = (code.find("float4 custom_material") != std::string::npos);
    if (legacy) {
        size_t p = 0;
        const std::string old_sig = "float4 custom_material";
        const std::string new_sig = "float4 _cm_legacy";
        while ((p = code.find(old_sig, p)) != std::string::npos) {
            code.replace(p, old_sig.size(), new_sig);
            p += new_sig.size();
        }
        code += R"(
// Auto-generated adapter: converts old float4 return to MatOut
__device__ MatOut custom_material(float2 uv, float3 pos, float3 normal, int frame) {
    float4 r = _cm_legacy(uv, pos, normal, frame);
    MatOut m;
    m.base_color = make_float3(r.x, r.y, r.z);
    m.roughness  = r.w;
    m.metallic   = 0.f;
    m.emissive   = make_float3(0.f, 0.f, 0.f);
    m.normal_ts  = make_float3(0.f, 0.f, 1.f);
    return m;
}
)";
    }

    std::string src(KERNEL_WRAPPER);
    const std::string marker = "%USER_CODE%";
    size_t pos = src.find(marker);
    if (pos != std::string::npos)
        src.replace(pos, marker.size(), code);
    return src;
}

static inline uint8_t f32_to_u8(float v)
{
    int i = (int)(v * 255.f + 0.5f);
    return (uint8_t)(i < 0 ? 0 : i > 255 ? 255 : i);
}

// ─────────────────────────────────────────────────────────────────────────────
//  custom_shader_run
// ─────────────────────────────────────────────────────────────────────────────

CustomShaderResult custom_shader_run(const std::string& user_code,
                                     int width, int height,
                                     int frame_count)
{
    CustomShaderResult res;
    res.width  = width;
    res.height = height;

    std::string src = build_source(user_code);

    // ── Compile ──────────────────────────────────────────────────────────────
    nvrtcProgram prog = nullptr;
    nvrtcResult nvr = nvrtcCreateProgram(&prog, src.c_str(),
                                          "custom_material.cu", 0, nullptr, nullptr);
    if (nvr != NVRTC_SUCCESS) {
        res.error_log = std::string("nvrtcCreateProgram: ") + nvrtcGetErrorString(nvr);
        return res;
    }

    const char* opts[] = {
        "--gpu-architecture=compute_70",
        "--std=c++14",
        "--device-as-default-execution-space",
    };
    nvr = nvrtcCompileProgram(prog, 3, opts);

    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
        res.error_log.resize(log_size - 1);
        nvrtcGetProgramLog(prog, &res.error_log[0]);
    }
    if (nvr != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        return res;
    }

    // ── PTX → CUDA module ────────────────────────────────────────────────────
    size_t ptx_size = 0;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    cuInit(0);
    CUmodule   cu_mod  = nullptr;
    CUfunction cu_func = nullptr;
    CUresult cr = cuModuleLoadData(&cu_mod, ptx.data());
    if (cr != CUDA_SUCCESS) {
        const char* s = nullptr; cuGetErrorString(cr, &s);
        res.error_log = std::string("cuModuleLoadData: ") + (s ? s : "?");
        return res;
    }
    cr = cuModuleGetFunction(&cu_func, cu_mod, "gen_tex_kernel");
    if (cr != CUDA_SUCCESS) {
        const char* s = nullptr; cuGetErrorString(cr, &s);
        res.error_log = std::string("cuModuleGetFunction: ") + (s ? s : "?");
        cuModuleUnload(cu_mod);
        return res;
    }

    // ── Allocate 4 GPU output buffers ─────────────────────────────────────────
    size_t npix = (size_t)width * height;
    float4 *d_base = nullptr, *d_mr = nullptr, *d_emis = nullptr, *d_norm = nullptr;
    if (cudaMalloc(&d_base, npix * sizeof(float4)) != cudaSuccess ||
        cudaMalloc(&d_mr,   npix * sizeof(float4)) != cudaSuccess ||
        cudaMalloc(&d_emis, npix * sizeof(float4)) != cudaSuccess ||
        cudaMalloc(&d_norm, npix * sizeof(float4)) != cudaSuccess) {
        res.error_log = "cudaMalloc failed for output buffers";
        cudaFree(d_base); cudaFree(d_mr); cudaFree(d_emis); cudaFree(d_norm);
        cuModuleUnload(cu_mod);
        return res;
    }

    // ── Launch ───────────────────────────────────────────────────────────────
    dim3 block(16, 16);
    dim3 grid(((unsigned)width  + 15) / 16,
              ((unsigned)height + 15) / 16);

    void* args[] = { &d_base, &d_mr, &d_emis, &d_norm, &width, &height, &frame_count };
    cr = cuLaunchKernel(cu_func,
                        grid.x, grid.y, 1,
                        block.x, block.y, 1,
                        0, nullptr, args, nullptr);
    cudaDeviceSynchronize();

    if (cr != CUDA_SUCCESS) {
        const char* s = nullptr; cuGetErrorString(cr, &s);
        res.error_log = std::string("cuLaunchKernel: ") + (s ? s : "?");
        cudaFree(d_base); cudaFree(d_mr); cudaFree(d_emis); cudaFree(d_norm);
        cuModuleUnload(cu_mod);
        return res;
    }

    // ── Read back + convert to RGBA8 ──────────────────────────────────────────
    auto readback = [&](float4* d_buf, std::vector<uint8_t>& out) {
        std::vector<float4> tmp(npix);
        cudaMemcpy(tmp.data(), d_buf, npix * sizeof(float4), cudaMemcpyDeviceToHost);
        out.resize(npix * 4);
        for (size_t i = 0; i < npix; ++i) {
            out[i*4+0] = f32_to_u8(tmp[i].x);
            out[i*4+1] = f32_to_u8(tmp[i].y);
            out[i*4+2] = f32_to_u8(tmp[i].z);
            out[i*4+3] = f32_to_u8(tmp[i].w);
        }
    };

    readback(d_base, res.base_pixels);
    readback(d_mr,   res.mr_pixels);
    readback(d_emis, res.emissive_pixels);
    readback(d_norm, res.normal_pixels);

    cudaFree(d_base); cudaFree(d_mr); cudaFree(d_emis); cudaFree(d_norm);
    cuModuleUnload(cu_mod);

    res.success = true;
    return res;
}
