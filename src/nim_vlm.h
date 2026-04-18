#pragma once

// ── NVIDIA NIM VLM object recognition ───────────────────────────────────────
// Sends the current render to a self-hosted NVIDIA NIM Vision Language Model
// endpoint (OpenAI-compatible) and returns structured scene recognition.
//
// The NIM container is started with:
//   docker run -it --gpus all -p 8000:8000 nvcr.io/nim/<model>
// Typical models: nvidia/nemotron-nano-12b-v2-vl
//                 nvidia/llama-3.1-nemotron-nano-vl-8b-v1
//                 meta/llama-3.2-90b-vision-instruct
// (microsoft/phi-3.5-vision-instruct was retired by NIM on 2026-04-15.)

struct NimVlmConfig {
    // ── Endpoint ──────────────────────────────────────────────────────────────
    // Cloud: host = "integrate.api.nvidia.com", port = 443, use_https = true
    // Local: host = "localhost",                port = 8000, use_https = false
    char  host[256]    = "integrate.api.nvidia.com";
    int   port         = 443;
    bool  use_https    = true;
    char  api_key[512] = "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";   // NVIDIA API key — leave empty for local Docker

    // ── Model ─────────────────────────────────────────────────────────────────
    char  model[256]   = "nvidia/nemotron-nano-12b-v2-vl";
    float temperature  = 0.05f;
    int   max_tokens   = 256;
};

// Ping the NIM endpoint (GET /v1/models) — returns true if reachable.
// Runs synchronously; call from a background thread.
bool nim_vlm_check_connection(const NimVlmConfig& cfg);

// Launch a local NIM container via Docker (non-blocking, opens a cmd window).
// Requires Docker Desktop with GPU support installed.
// Returns false and writes to error_out if Docker is not running.
bool nim_vlm_launch_docker(const NimVlmConfig& cfg, char* error_out = nullptr, int error_len = 0);

struct NimVlmResult {
    char category   [128] = {};  // broad category: furniture, vehicle, architecture …
    char object_type[256] = {};  // specific type: "wooden chair", "sports car" …
    char description[512] = {};  // 2-3 sentence visual description
    char tags       [256] = {};  // comma-separated hashtags for search
    bool success          = false;
    char error_msg  [512] = {};
};

// Send pixels to the NIM VLM endpoint and return recognised scene info.
// pixels_rgba_f32: CPU-side buffer of float4 (4 floats per pixel, linear HDR).
// Runs synchronously — always call from a background std::thread.
NimVlmResult nim_vlm_recognize(
    const NimVlmConfig& cfg,
    const float*        pixels_rgba_f32,
    int                 width,
    int                 height);

// Multi-angle variant — sends up to 'count' renders in a single request.
// pixels[i] points to a float4 RGBA buffer of width*height pixels.
NimVlmResult nim_vlm_recognize_multi(
    const NimVlmConfig& cfg,
    const float* const* pixels,
    int                 count,
    int                 width,
    int                 height);
