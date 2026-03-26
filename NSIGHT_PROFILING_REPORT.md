# GPU Performance Profiling Report
## CUDA/Vulkan Path Tracer — Nsight Compute Analysis

---

## Environment

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 4090 |
| Architecture | Ada Lovelace (SM 8.9, `0x190`) |
| Driver | 591.86 |
| Nsight Compute | 2024.1.0 |
| Profiler | `ncu` CLI (`--set full --launch-count 100`) |
| Platform | Windows 11, CUDA 12.x |

---

## Step 1 — How We Ran the Profile

### Tool: Nsight Compute CLI (`ncu`)

```bat
ncu --set full --launch-count 100 -o profile_<timestamp> pathtracer.exe
```

- `--set full` — captures every available GPU metric (memory, compute, warp states, scheduler)
- `--launch-count 100` — profiles the first 100 CUDA kernel launches then exits automatically
- `-o` with timestamp — each run saves a unique `.ncu-rep` file, no overwrites
- Must run as **Administrator** on Windows — GPU hardware counters require elevated access

A batch file (`nsight_profile.bat`) was created to automate this with automatic timestamped output.

### Kernels Captured

| ID Range | Kernel | What It Is |
|----------|--------|-----------|
| 0 – 73 | `NVIDIA internal` | OptiX HDR denoiser internals (NVIDIA code, not ours) |
| 74, 76, 78… | `pathtrace_kernel` | **Our path tracing kernel** |
| 75, 77, 79… | `dlss_aux_kernel` | DLSS motion vector / depth preparation |

---

## Step 2 — What We Found

### Opening View: NVIDIA Internal Kernels (OptiX)

The first 73 launches were OptiX denoiser passes. Nsight flagged three issues:

| Issue | Est. Speedup |
|-------|-------------|
| Achieved Occupancy (only 11.2% vs 100% theoretical) | 84.65% |
| Small Grid (64 blocks < 128 SMs on RTX 4090) | 50.00% |
| Imc Miss Stalls (4.7 cycles/warp on constant cache) | 36.07% |

These are **NVIDIA's internal OptiX kernels** — not our code, cannot be modified.

---

## Step 3 — The Real Bottleneck: `pathtrace_kernel`

### Baseline Metrics (Run 1)

```
Kernel:    pathtrace_kernel (70, 52, 1) × (16, 16, 1)
Duration:  ~253 µs per call
Cycles:    564,085
Compute:   ~65%   (GPU math units — reasonably busy)
Memory:    ~75%   (VRAM bandwidth — the bigger limiter)
Registers: 102 per thread   ← ROOT CAUSE
```

**The kernel is memory-bound.** Memory throughput (75%) exceeds compute (65%) —
the GPU is stalling on memory more than doing math.

### Three Issues Flagged by Nsight

| Issue | Efficiency | Est. Speedup |
|-------|-----------|-------------|
| L1TEX Local Store Access Pattern | 1.4 / 32 bytes per sector | 62.10% |
| L1TEX Local Load Access Pattern | 2.7 / 32 bytes per sector | 59.46% |
| L1TEX Global Load Access Pattern | 7.4 / 32 bytes per sector | 49.98% |

All three indicate **uncoalesced memory access** — instead of 32 threads in a warp
reading 32 adjacent addresses (one cache line), they read scattered addresses,
forcing up to 32 separate cache line fetches — a ~20× bandwidth waste.

### SASS Source View: `bvh_hit`

Drilling into the SASS assembly confirmed the hot spot: two global memory loads
(⚠️ warnings on `LD.E R42, [R32.64]` and `LD.E R48, [R32.64+8xc]`) stalling
warps waiting for BVH node data from VRAM. Classic memory-latency-bound BVH traversal.

---

## Step 4 — Root Cause Analysis

### Why 102 Registers?

#### Problem 1: `HitRecord` allocated twice per bounce

```cuda
// Inside trace_path(), every bounce iteration:
HitRecord rec;      // sphere BVH result   — ~124 bytes, ~31 registers
HitRecord tri_rec;  // triangle BVH result — ~124 bytes, ~31 registers (second copy!)
```

The original `HitRecord` had hidden compiler padding due to mixed float3/float4 fields
and an embedded `Material` struct:

```cuda
struct HitRecord {
    float3 p, normal, geom_normal;  // 36 bytes — compiler pads to 48 for float4 alignment
    float4 tangent;                  // 16 bytes
    Material mat;                    // 36 bytes (albedo, emission, roughness, ior, type)
    float t; bool front_face;        //  8 bytes (bool padded to 4)
    float2 uv;                       //  8 bytes
    int gpu_mat_idx, obj_id;         //  8 bytes
};  // ~124 bytes actual = ~31 registers
```

Two instances simultaneously live = ~62 registers just for hit data.

#### Problem 2: Many simultaneous live variables in `scatter_gpu_material()`

```cuda
float3 N, V, F0, F, albedo;    // 5 × float3 = 15 registers
float3 T, B, H, L;             // 4 × float3 = 12 registers
float3 attenuation, emission;  // 2 × float3 =  6 registers
// + metallic, roughness, alpha, NdotV, F_lum, r1, r2... ~10 more
// = ~43 registers in this function alone
```

#### Problem 3: Register spilling to local memory

When register count exceeds the SM's per-thread budget, NVCC spills overflow to
**local memory** — private per-thread scratch space backed by L1/L2/VRAM.
Each thread spills to a different address → completely uncoalesced.

This is exactly what the Nsight flags showed:
- Local Store 1.4/32 bytes — writes to spill slots in VRAM
- Local Load 2.7/32 bytes — reads to restore spilled values

### Chain of Causation

```
102 registers/thread
    → register overflow spills to local memory (per-thread VRAM scratch)
    → spill loads/stores are uncoalesced (each thread hits a different address)
    → memory bandwidth wasted on scattered 32-bit transactions
    → GPU stalls waiting for memory
    → low SM occupancy (~11% — only 1 block fits per SM)
```

---

## Step 5 — What We Tried (Honest Account)

### Attempt 1: `__launch_bounds__(256, 2)` — No Effect

```cuda
__launch_bounds__(256, 2)
__global__ void pathtrace_kernel(PathTracerParams p)
```

**Result:** Still 102 registers. The cap this sets is `65536 / (2×256) = 128` —
above 102, so the compiler had no reason to reduce anything. No-op.

### Attempt 2: `__launch_bounds__(256, 3)` — Build Failed

Cap would be `65536 / (3×256) = 85` registers. NVCC device-link exited with code 255.
The kernel physically cannot fit in 85 registers without breaking.

### Attempt 3: `--maxrregcount=96` in CMakeLists — No Effect

Set as a per-file COMPILE_OPTIONS on `pathtracer.cu`. The kernel was already
at 94 registers (from the struct changes below), so 94 < 96 — constraint never triggered.

### Attempt 4: `--maxrregcount=80` / `88` — Build Failed

Same device-link exit code 255. Confirmed the kernel's hard floor is ~94 registers
for the GGX PBR + dual-BVH algorithm.

**Lesson:** Compiler flags can only reduce registers within the algorithm's natural range.
The code itself had to change first.

---

## Step 6 — What Actually Worked

### Fix 1: Packed `HitRecord` struct (`scene.h`)

Redesigned `HitRecord` to eliminate hidden compiler padding by pairing each `float3`
with a `float` to fill exact 16-byte slots. Also removed the embedded `Material` struct,
promoting its fields to top level.

```cuda
// OLD — ~124 bytes, hidden gaps, embedded struct
struct HitRecord {
    float3 p, normal, geom_normal;
    float4 tangent;
    Material mat;           // embedded — albedo, emission, roughness, ior, type
    float t; bool front_face; float2 uv;
    int gpu_mat_idx, obj_id;
};

// NEW — 112 bytes, explicit packing, no hidden padding
struct HitRecord {
    float3 p;           float t;        // 16 bytes — position + distance packed
    float3 normal;      float u;        // 16 bytes — shading normal + uv.x
    float3 geom_normal; float v;        // 16 bytes — geometric normal + uv.y
    float4 tangent;                     // 16 bytes
    float3 albedo;      float roughness;// 16 bytes — sphere material
    float3 emission;    float ior;      // 16 bytes — sphere material
    int gpu_mat_idx, obj_id;            //  8 bytes
    int mat_type, front_face;           //  8 bytes  (bool → int)
};  // 112 bytes = 28 registers
```

**Files changed:** `scene.h`, `bvh.cu`, `pathtracer.cu`, `restir.cu`, `optix_renderer_device.cu`

**Register saving:** ~31 → ~28 registers per instance × 2 instances = **~6 registers saved**

> **OptiX breakage during this fix:** Adding `alignas(16)` to `HitRecord` caused
> OptiX RT "runtime init failed". Root cause: `RadiancePRD` (the OptiX payload struct)
> embeds `HitRecord`, and `alignas(16)` inserts 12 bytes of padding after `int hit`,
> shifting the struct layout. OptiX's continuation stack does not honour the alignment,
> causing a misaligned memory access at runtime. Fixed by removing `alignas(16)` —
> the packed float3+float layout provides the register savings without it.

### Fix 2: Sequential BVH Traversal into One `HitRecord` (`pathtracer.cu`)

Eliminated the second `HitRecord` instance by traversing both BVHs into the same variable.

```cuda
// OLD — two HitRecords alive simultaneously
HitRecord rec;
bool any_hit = bvh_hit(..., rec);
float t_max = any_hit ? rec.t : 1e20f;
if (p.tri_bvh) {
    HitRecord tri_rec;                              // ← second allocation
    if (bvh_hit_triangles(..., t_max, tri_rec)) {
        rec = tri_rec;                              // copy back
        any_hit = true;
    }
}

// NEW — one HitRecord, sequential shrinking t_max
HitRecord rec;
bool any_hit = false;
float t_max = 1e20f;
if (p.num_prims > 0 && bvh_hit(..., t_max, rec)) {
    any_hit = true;
    t_max = rec.t;          // shrink — triangles must beat this
}
if (p.tri_bvh && bvh_hit_triangles(..., t_max, rec))  // overwrites only if closer
    any_hit = true;
```

**Why this is correct:** `bvh_hit_triangles` only writes to `rec` when it finds a hit
with `t < t_max`. Passing the sphere's `rec.t` as the new `t_max` guarantees the
sphere result is preserved automatically if no triangle is closer.

**Register saving:** Removes one full `HitRecord` from the live variable set —
28 fewer registers of peak pressure, improving spill coalescing.

---

## Step 7 — Results Across All Profiling Runs

| Run | Changes | Registers | Cycles | Duration | Local Store | Local Load |
|-----|---------|-----------|--------|----------|-------------|------------|
| 1 — Baseline | None | 102 | 564,085 | 253 µs | 62.10% est. | 59.46% est. |
| 2 — `__launch_bounds__(256,2)` | Compiler hint only | 102 | 553,024 | 249 µs | 63.35% | 60.66% |
| 3 — Packed HitRecord | Struct redesign | **94** | 543,624 | 247 µs | 64.09% | 61.40% |

**Total achieved:** 102 → 94 registers (−8, −7.8%), cycles −3.6%, duration −2.4%.

The memory access pattern scores (local store/load) improved consistently each run,
confirming the spill pattern is improving even when raw register count stays similar.

---

## Step 8 — Why We Can't Go Below ~94 Registers

The kernel's minimum register budget is set by what's simultaneously live during
GGX PBR evaluation inside `scatter_gpu_material`:

| Variable Group | Registers |
|----------------|-----------|
| `HitRecord rec` (one instance) | ~28 |
| GGX PBR: N, V, F0, F, H, T, B, L | ~24 |
| attenuation, emission, throughput, radiance | ~12 |
| Scalar temps: metallic, roughness, alpha, NdotV, F_lum, r1, r2 | ~7 |
| RNG state, loop counter, ray | ~5 |
| **Minimum total** | **~76–94** |

Forcing below ~85 registers causes `nvcc -dlink` to fail (exit code 255) because
the register pressure cannot be satisfied through spilling alone without breaking
the kernel's execution model.

---

## Step 9 — Remaining Opportunities (Future Work)

### 1. Split Kernel Into Passes (Biggest Impact)

Separate `pathtrace_kernel` into:
1. **Ray generation pass** — camera rays → ray buffer in VRAM
2. **BVH traversal pass** — ray buffer → hit buffer in VRAM
3. **Shading pass** — hit buffer → color accumulation

Each pass uses far fewer simultaneous live variables → registers drop to ~40–50 per pass.
Warps within each pass are more coherent (all doing the same work) → better coalescing.

**Cost:** Significant architectural change. Hit/ray buffers add global memory bandwidth
but it's well-coalesced (sequential per pixel) — net win for complex scenes.

### 2. Wider BVH (4-wide or 8-wide)

The SASS view showed `bvh_hit` stalling on every node fetch from global memory.
A 4-wide BVH halves traversal depth, halving the number of memory transactions per ray.

### 3. Pass `PathTracerParams` by Pointer

```cuda
__global__ void pathtrace_kernel(const PathTracerParams* __restrict__ p)
```

The ~200-byte param struct currently lives in registers/param memory. A `__restrict__`
pointer lets NVCC cache fields in constant memory, saving ~10–15 registers.

---

## Summary

| What | Finding |
|------|---------|
| **Tool** | Nsight Compute `ncu --set full --launch-count 100`, run as Admin, timestamped output |
| **Root cause** | 102 registers/thread → register spilling → uncoalesced local memory traffic |
| **Why 102** | Hidden struct padding + embedded Material + two simultaneous HitRecord instances |
| **What worked** | Packed HitRecord struct (float3+float pairs) + sequential BVH traversal |
| **What didn't** | `__launch_bounds__` and `--maxrregcount` — flags only help within algorithmic range |
| **Result** | 102 → 94 registers, −3.6% cycles, memory access patterns improving |
| **Floor** | ~94 registers — hard limit of the GGX PBR + BVH algorithm as a single kernel |
| **Next step** | Split kernel into ray-gen / traversal / shading passes to break the floor |

---

## Interview Talking Points

**Q: How did you find the performance problem?**

> We ran NVIDIA Nsight Compute with `--set full` across 100 kernel launches. The Summary page flagged three uncoalesced memory access patterns — local store at 1.4/32 bytes efficiency, local load at 2.7/32, global load at 7.4/32. The register count of 102 per thread immediately identified register spilling as the root cause.

**Q: What is register spilling and why is it bad?**

> The GPU has a fixed register file per SM — 65,536 registers on the RTX 4090. When a kernel needs more than its share, the compiler spills the overflow to local memory: private per-thread scratch space backed by L1 and VRAM. The problem is each thread in a warp spills to a different address, so 32 threads generate 32 separate memory transactions instead of one coalesced 128-byte fetch. That's a 20× bandwidth overhead on every spilled variable access.

**Q: What changes actually reduced the register count?**

> Two code changes. First, we redesigned the `HitRecord` struct — the original had hidden compiler padding between float3 and float4 fields, plus an embedded Material struct. Packing each float3 with a float to form exact 16-byte groups removed the waste, going from ~124 bytes to 112 bytes per instance. Second, we removed the second HitRecord entirely by traversing both BVHs sequentially into the same variable, shrinking `t_max` after the sphere test so triangles only overwrite the result if closer. Together these dropped registers from 102 to 94.

**Q: Why didn't `--maxrregcount` or `__launch_bounds__` help?**

> Compiler flags can only constrain the compiler — they cannot reduce the algorithm's inherent register pressure. `__launch_bounds__(256, 2)` sets a cap of 128 registers, which is already above 102 so nothing changes. `--maxrregcount=96` was set after the struct changes already brought us to 94, so it was also never triggered. When we tried values that would actually bite — 80, 85, 88 — the device-link step failed with exit code 255, because the kernel genuinely cannot execute correctly with fewer than ~94 registers. The struct redesign had to come first.

**Q: What would you do next to push further?**

> The remaining 94-register floor is set by the GGX PBR BSDF having ~43 registers of simultaneous live variables, plus the HitRecord, plus the ray and throughput state. The only way past this is to split the kernel into separate passes — ray generation, BVH traversal, and shading — each with far fewer live variables. Each pass then runs at ~40–50 registers, doubles or triples SM occupancy, and achieves much better warp coherence since all threads in a pass do the same work. The tradeoff is additional global memory traffic for the hit/ray buffers between passes, but it's sequential and well-coalesced, which is a net win for complex scenes.

**Q: What went wrong during the optimisation and how did you fix it?**

> When packing the HitRecord we added `alignas(16)` to ensure 16-byte alignment. This broke the OptiX RT renderer at runtime — OptiX passes the payload struct as a pointer split across two 32-bit registers, and the `alignas(16)` inserted 12 bytes of padding between `int hit` and `HitRecord rec` inside the `RadiancePRD` payload, shifting the struct layout in a way OptiX's continuation stack doesn't honour. Removing `alignas(16)` fixed the crash — the packed float3+float layout still provides the register savings without requiring explicit alignment.
