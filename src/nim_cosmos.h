#pragma once
#include <cstdint>

// ── NVIDIA Cosmos Transfer — AI-augmented rendering via AOV control signals ──
// Sends beauty + depth + segmentation (primId) to Cosmos-Transfer2.5
// and returns a stylized/augmented frame.
//
// Uses the same NVIDIA NIM cloud endpoint as nim_vlm (integrate.api.nvidia.com).

struct CosmosConfig {
    // ── Endpoint ─────────────────────────────────────────────────────────────
    char  host[256]    = "integrate.api.nvidia.com";
    int   port         = 443;
    bool  use_https    = true;
    char  api_key[512] = "";   // NGC API key — copied from NimVlmConfig at init
    char  endpoint[256] = "/v1/infer";  // /v1/infer for both cloud and local NIM

    // ── Model ────────────────────────────────────────────────────────────────
    char  model[256]   = "nvidia/cosmos-transfer2p5-2b";

    // ── Prompt & controls ────────────────────────────────────────────────────
    char  prompt[1024] = "photorealistic cinematic render, high detail, volumetric lighting";
    float depth_weight = 0.9f;
    float seg_weight   = 0.9f;
    int   seed         = 42;

    // ── State ────────────────────────────────────────────────────────────────
    bool  busy         = false;   // true while request is in flight
    bool  has_result   = false;   // true when a result frame is ready
    char  status[256]  = "";      // status text for UI
};

struct CosmosResult {
    bool     success      = false;
    char     error[512]   = {};
    // Decoded output frame (RGBA8, caller frees with delete[])
    uint8_t* pixels       = nullptr;
    int      width        = 0;
    int      height       = 0;
};

// Send AOV buffers to Cosmos-Transfer and get back a stylized frame.
// All buffers are RGBA8 (uint8_t, 4 bytes per pixel).
// Runs synchronously — always call from a background std::thread.
CosmosResult cosmos_transfer(
    const CosmosConfig& cfg,
    const uint8_t*      beauty_rgba8,   // color AOV
    const uint8_t*      depth_rgba8,    // depth AOV (greyscale encoded as RGB)
    const uint8_t*      seg_rgba8,      // primId AOV (ID colors)
    int                 width,
    int                 height);
