#pragma once

#include <string>
#include <vector>
#include "nim_vlm.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Batch Asset Library Builder
//
//  Scans a folder of 3D models, renders each one with path tracing,
//  optionally runs NIM VLM recognition, then saves:
//    output_dir/category/Nice_Name/render.jpg
//    output_dir/category/Nice_Name/metadata.json
// ─────────────────────────────────────────────────────────────────────────────

enum class BatchState {
    Idle,
    Loading,      // waiting for main.cpp to call load_gltf_into
    Rendering,    // accumulating frames
    Recognizing,  // NIM VLM recognition in progress
    AllDone,
    Stopped
};

struct BatchConfig {
    char input_dir[512]  = "F:\\PROJECTS\\NVDIA\\CudaVulkan\\model\\Assets";
    char output_dir[512] = "asset_library";
    int  target_frames   = 8;
    int  jpeg_quality    = 85;
    bool use_nim         = true;
    bool skip_existing   = true;
};

struct BatchLogEntry {
    std::string source_path;
    std::string category;
    std::string object_name;
    std::string out_dir;
};

static constexpr int BATCH_NUM_ANGLES = 3;

struct BatchProcessor {
    BatchConfig  config;
    BatchState   state = BatchState::Idle;

    std::vector<std::string> files;  // discovered model paths
    int  current_idx = 0;

    // ── Multi-angle rendering ──────────────────────────────────────────
    int   angle_idx   = 0;       // which angle is currently rendering (0..BATCH_NUM_ANGLES-1)
    float bsphere_r   = 1.f;     // bounding-sphere radius computed after load
    float saved_vfov  = 45.f;    // user's vfov saved before batch overrides it
    std::vector<float> angle_pixels[BATCH_NUM_ANGLES];  // captured HDR pixels per angle
    int  render_w   = 0;
    int  render_h   = 0;

    // ── Flags / requests for main.cpp ─────────────────────────────────
    bool do_load     = false;   // main.cpp should load files[current_idx]
    bool cam_repoint = false;   // main.cpp should reposition camera to angle_idx
    bool nim_busy    = false;   // batch NIM thread is running
    bool nim_started = false;

    NimVlmResult nim_result{};

    char status[256] = "Idle";
    std::vector<BatchLogEntry> log;
};

// ── API ──────────────────────────────────────────────────────────────────────

// Scan dir recursively for .gltf/.glb/.usd/.usda/.usdc/.usdz
std::vector<std::string> batch_scan_files(const char* dir);

// Replace unsafe chars with underscores, collapse runs, trim to max_len
std::string batch_sanitize(const std::string& s, int max_len = 48);

// Write render.jpg + render_front.jpg + render_side.jpg + metadata.json
void batch_save(BatchProcessor& bp);

// Begin processing: scan files, reset state
void batch_start(BatchProcessor& bp);

// Abort processing
void batch_stop(BatchProcessor& bp);

// ImGui panel (opens its own window)
void batch_panel_draw(BatchProcessor& bp);
