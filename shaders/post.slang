// post.slang — Slang compute shader for post-processing
// Input:  HDR image from CUDA path tracer  (binding 0, combined image sampler)
// Output: LDR post-processed image          (binding 1, storage image)
// Push constants: exposure, tone_map_mode, bloom_strength, bloom_threshold,
//                 vignette_strength, saturation, width, height

// ── Descriptors ──────────────────────────────────────────────────────────────

// Binding 0: sampled image  (VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
[[vk::binding(0, 0)]] Texture2D<float4> hdr_input;
// Binding 1: sampler        (VK_DESCRIPTOR_TYPE_SAMPLER)
[[vk::binding(1, 0)]] SamplerState hdr_sampler;
// Binding 2: storage image  (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
[[vk::binding(2, 0)]] RWTexture2D<float4> ldr_output;
// Binding 3: AgX 3D LUT     (VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampled via binding 1)
[[vk::binding(3, 0)]] Texture3D<float4> agx_lut;

// ── Push constants ───────────────────────────────────────────────────────────

struct PostPushConstants {
    float exposure;           // overall exposure multiplier
    int   tone_map_mode;      // 0=None, 1=Reinhard, 2=ACES, 3=AgX, 4=Filmic, 5=Khronos
    float bloom_strength;     // how much bloom to add (0..2)
    float bloom_threshold;    // luminance threshold for bloom extraction
    float vignette_strength;  // 0 = no vignette, 1 = heavy vignette
    float saturation;         // 1 = original, 0 = greyscale, 2 = hyper-saturated
    int   width;
    int   height;
    float vignette_falloff;   // 1 = linear, 2 = smooth, 4 = very soft
    int   look;               // contrast look: 0=None, 1=VeryLow..7=VeryHigh
    float gamma;              // display gamma: 1=neutral, >1=brighter midtones (Blender default=1)
};

[[vk::push_constant]] PostPushConstants params;

// ── Helpers ──────────────────────────────────────────────────────────────────

float luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

// ── Tone-mapping operators ───────────────────────────────────────────────────

// Mode 0: clamp (no tone map)
float3 tonemap_clamp(float3 x) {
    return clamp(x, float3(0.f, 0.f, 0.f), float3(1.f, 1.f, 1.f));
}

// Mode 1: Reinhard luminance-preserving
float3 tonemap_reinhard(float3 x) {
    float lum = luminance(x);
    if (lum <= 0.f) return float3(0.f, 0.f, 0.f);
    float lum_out = lum / (1.f + lum);
    return clamp(x * (lum_out / lum), float3(0.f, 0.f, 0.f), float3(1.f, 1.f, 1.f));
}

// Mode 2: ACES fitted (Narkowicz 2015)
float3 tonemap_aces(float3 x) {
    float3 result = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
    return clamp(result, float3(0.f, 0.f, 0.f), float3(1.f, 1.f, 1.f));
}

// Mode 4: Filmic (Lottes 2016) — closely matches Blender "Filmic" (pre-4.0 default).
// Stays in linear space; sRGB OETF is applied afterward like all other modes.
float3 tonemap_filmic(float3 x)
{
    x = max(x, float3(0.f, 0.f, 0.f));
    // Contrast / shoulder / reference params tuned to match Blender Filmic Low Contrast
    const float a       = 1.60f;   // contrast
    const float d       = 0.977f;  // shoulder
    const float hdrMax  = 8.0f;
    const float midIn   = 0.18f;
    const float midOut  = 0.267f;  // Blender Filmic maps 0.18 → ~0.267
    float b = (-pow(midIn,  a) + pow(hdrMax, a)   * midOut) /
              ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    float c = (pow(hdrMax, a * d) * pow(midIn, a) -
               pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
              ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    float3 r = pow(x, float3(a, a, a)) /
               (pow(x, float3(a*d, a*d, a*d)) * b + c);
    return clamp(r, float3(0.f, 0.f, 0.f), float3(1.f, 1.f, 1.f));
}

// Mode 5: Khronos PBR Neutral — spec from KhronosGroup/ToneMapping (2023).
// Preserves low-luminance colors exactly; gently compresses highlights.
float3 tonemap_khronos_pbr(float3 color)
{
    color = max(color, float3(0.f, 0.f, 0.f));
    const float startCompression = 0.8f - 0.04f;   // = 0.76
    const float desaturation     = 0.15f;

    // Black-point offset (removes muddy shadow bias)
    float x = min(color.r, min(color.g, color.b));
    float offset = (x < 0.08f) ? (x - 6.25f * x * x) : 0.04f;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    const float d    = 1.f - startCompression;
    float newPeak    = 1.f - d * d / (peak + d - startCompression);
    color           *= newPeak / peak;

    float g = 1.f - 1.f / (desaturation * (peak - newPeak) + 1.f);
    return lerp(color, float3(newPeak, newPeak, newPeak), g);
}

// Mode 3: AgX — exact Blender 5.0 color management.
//
// Uses Blender's actual AgX_Base_sRGB.cube 3D LUT (binding 3).
// Pipeline matches Blender's OCIO config exactly:
//   1. Linear sRGB → E-Gamut (combined sRGB→XYZ_D65→XYZ_E→E-Gamut matrix)
//   2. Log2 normalize with [-12.47393, 12.5260688117] (25 stops, Blender range)
//   3. Sample AgX_Base_sRGB.cube 3D LUT (tetrahedral → trilinear hardware approx)
//   4. Decode Rec.1886 gamma (LUT output is power-2.4 encoded) → linear for display
//
// Combined sRGB → E-Gamut matrix (computed from Blender OCIO config matrices):
//   sRGB → XYZ_D65 (IEC 61966-2-1) × XYZ_D65 → XYZ_E (Bradford D65→E) × XYZ_E → E-Gamut

float3 tonemap_agx(float3 x)
{
    x = max(x, float3(0.f, 0.f, 0.f));

    // 1. sRGB → E-Gamut (combined matrix from Blender OCIO config)
    float3 eg = float3(
        dot(x, float3(0.52473429f, 0.25089006f, 0.14290110f)),
        dot(x, float3(0.07638113f, 0.81678924f, 0.14679310f)),
        dot(x, float3(0.06439888f, 0.16921565f, 0.83836502f))
    );

    // 2. Log2 normalize to [0,1]: Blender range [-12.47393, +12.5260688117] = 25 stops
    const float min_ev = -12.47393f;
    const float max_ev =  12.5260688117f;
    float3 log_eg = (log2(max(eg, float3(1e-10f, 1e-10f, 1e-10f))) - min_ev) / (max_ev - min_ev);
    log_eg = clamp(log_eg, float3(0.f, 0.f, 0.f), float3(1.f, 1.f, 1.f));

    // 3. Sample AgX_Base_sRGB.cube (trilinear)
    float3 encoded = agx_lut.SampleLevel(hdr_sampler, log_eg, 0).rgb;

    // 4. Decode Rec.1886 EOTF (gamma 2.4) → linear sRGB for display
    //    LUT output is Rec.1886-encoded; our display image is linear float.
    return pow(max(encoded, float3(0.f, 0.f, 0.f)), float3(2.4f, 2.4f, 2.4f));
}

// ── Bloom ────────────────────────────────────────────────────────────────────
// Correct soft bloom: Gaussian blur of NEIGHBOURING bright pixels only.
// The current pixel does NOT sample itself — that prevents the "doubling"
// of specular highlights. Result is normalised by total Gaussian weight so
// bloom_strength is a true linear scale (0.1 = subtle, 1.0 = strong glow).
float3 compute_bloom(float2 uv, float threshold) {
    float2 ps = float2(1.f / (float)params.width, 1.f / (float)params.height);

    // Golden-ratio spiral, 48 samples, radius 2..28 px, σ = 10 px.
    // Starting at radius 2 (not 0) ensures we never sample the centre pixel.
    const int   N        = 48;
    const float R_MIN    = 2.f;
    const float R_MAX    = 28.f;
    const float SIGMA2   = 100.f;           // σ = 10 px — soft, wide falloff
    const float PHI_STEP = 2.39996323f;     // golden angle ≈ 137.5°

    float3 bloom  = float3(0.f, 0.f, 0.f);
    float  weight = 0.f;

    for (int i = 0; i < N; ++i) {
        float r  = R_MIN + (float)i * ((R_MAX - R_MIN) / (float)(N - 1));
        float gw = exp(-(r * r) / (2.f * SIGMA2));

        float phi  = (float)i * PHI_STEP;
        float2 off = float2(cos(phi), sin(phi)) * (r * ps);

        float2 suv = clamp(uv + off, float2(0.f, 0.f), float2(1.f, 1.f));
        float3 col = hdr_input.SampleLevel(hdr_sampler, suv, 0).rgb;
        float  lum = luminance(col);

        // Only the above-threshold energy leaks — preserves hue of bright source
        float above = max(lum - threshold, 0.f);
        bloom  += col * (above / max(lum, 1e-6f)) * gw;
        weight += gw;
    }

    // Proper normalisation: divide by total weight — no extra multiplier.
    // bloom_strength in the caller is the sole gain control.
    return (weight > 1e-6f) ? bloom / weight : float3(0.f, 0.f, 0.f);
}

// ── Contrast Look (Blender-style: None / Very Low … Very High) ───────────────
// Applied in LDR [0,1] space using a power curve — preserves black at 0 and
// white at 1, no dark crushing. gamma < 1 = lower contrast, gamma > 1 = higher.
float3 apply_look(float3 c, int look)
{
    // 0=None,1=VeryLow,2=Low,3=MedLow,4=Med,5=MedHigh,6=High,7=VeryHigh
    // gamma values: <1 lifts shadows (low contrast), >1 crushes them (high contrast)
    const float gammas[8] = { 1.0f, 0.70f, 0.80f, 0.90f, 1.0f, 1.15f, 1.35f, 1.60f };
    if (look <= 0 || look > 7) return c;
    float g = gammas[look];
    return pow(max(c, float3(0.f, 0.f, 0.f)), float3(g, g, g));
}

// ── Entry point ──────────────────────────────────────────────────────────────

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int2 px = int2(tid.xy);

    // Guard against out-of-bounds threads
    if (px.x >= params.width || px.y >= params.height)
        return;

    // UV coords (0..1), Y-up convention for Vulkan
    float2 uv = float2((float)px.x + 0.5f, (float)px.y + 0.5f) /
                float2((float)params.width, (float)params.height);

    // ── Sample HDR input ─────────────────────────────────────────────────────
    float3 hdr = hdr_input.SampleLevel(hdr_sampler, uv, 0).rgb;

    // ── Apply exposure (EV stops, matching Blender: 0=neutral) ──────────────
    hdr *= pow(2.f, params.exposure);

    // ── Bloom ────────────────────────────────────────────────────────────────
    if (params.bloom_strength > 0.f) {
        float3 bloom = compute_bloom(uv, params.bloom_threshold);
        hdr += bloom * params.bloom_strength;
    }

    // ── Tone mapping ─────────────────────────────────────────────────────────
    float3 ldr;
    if (params.tone_map_mode == 1)
        ldr = tonemap_reinhard(hdr);
    else if (params.tone_map_mode == 2)
        ldr = tonemap_aces(hdr);
    else if (params.tone_map_mode == 3)
        ldr = tonemap_agx(hdr);
    else if (params.tone_map_mode == 4)
        ldr = tonemap_filmic(hdr);
    else if (params.tone_map_mode == 5)
        ldr = tonemap_khronos_pbr(hdr);
    else
        ldr = tonemap_clamp(hdr);  // mode 0: clamp

    // ── Contrast Look ────────────────────────────────────────────────────────
    ldr = apply_look(ldr, params.look);

    // ── Gamma (Blender Color Management — default 1.0 = neutral) ─────────────
    {
        float g = clamp(params.gamma, 0.1f, 10.f);  // safety clamp against corrupt push-constant
        if (g < 0.999f || g > 1.001f)
            ldr = pow(max(ldr, float3(0.f, 0.f, 0.f)), float3(1.f / g, 1.f / g, 1.f / g));
    }

    // ── Saturation ───────────────────────────────────────────────────────────
    {
        float lum = luminance(ldr);
        ldr = lerp(float3(lum, lum, lum), ldr, params.saturation);
        ldr = clamp(ldr, float3(0.f, 0.f, 0.f), float3(1.f, 1.f, 1.f));
    }

    // ── Vignette ─────────────────────────────────────────────────────────────
    // Radial distance [0..1] where 1 = corner. Soft power curve controlled
    // by vignette_falloff (1=linear, 2=smooth quad, 4=very soft quartic).
    {
        float2 d    = uv - float2(0.5f, 0.5f);
        float  dist = length(d) * 1.4142f;  // normalise so corner = 1.0
        float  edge = saturate(dist * params.vignette_strength * 1.5f);
        float  vig  = pow(1.f - edge, max(params.vignette_falloff, 0.5f));
        ldr        *= vig;
    }

    // ── Write output ─────────────────────────────────────────────────────────
    // The swapchain is VK_FORMAT_B8G8R8A8_SRGB — hardware applies sRGB OETF
    // automatically on framebuffer write.  We output LINEAR [0,1] values here;
    // applying sRGB in the shader too would double-encode and blow everything out.
    ldr_output[px] = float4(saturate(ldr), 1.f);
}
