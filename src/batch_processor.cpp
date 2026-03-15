#include "batch_processor.h"

#include <filesystem>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <imgui.h>

// stb_image_write — implementation compiled in stb_write_impl.cpp
#include "stb_image_write.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static float aces_ch(float x)
{
    if (x <= 0.f) return 0.f;
    float v = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
    return v < 0.f ? 0.f : v > 1.f ? 1.f : v;
}

static void jpeg_cb(void* ctx, void* data, int size)
{
    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
    const uint8_t* p = static_cast<const uint8_t*>(data);
    v->insert(v->end(), p, p + size);
}

static std::vector<uint8_t> encode_jpeg(const float* px, int w, int h, int quality)
{
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            size_t si = (static_cast<size_t>(row) * w + col) * 4;
            float r = std::powf(aces_ch(px[si + 0]), 1.f / 2.2f);
            float g = std::powf(aces_ch(px[si + 1]), 1.f / 2.2f);
            float b = std::powf(aces_ch(px[si + 2]), 1.f / 2.2f);
            size_t di = (static_cast<size_t>(row) * w + col) * 3;
            rgb[di + 0] = static_cast<uint8_t>(r * 255.f + 0.5f);
            rgb[di + 1] = static_cast<uint8_t>(g * 255.f + 0.5f);
            rgb[di + 2] = static_cast<uint8_t>(b * 255.f + 0.5f);
        }
    }
    std::vector<uint8_t> buf;
    buf.reserve(static_cast<size_t>(w) * h);
    stbi_write_jpg_to_func(jpeg_cb, &buf, w, h, 3, rgb.data(), quality);
    return buf;
}

// Escape backslashes and quotes for embedding in JSON strings
static std::string json_esc(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if      (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else                { out += c; }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::string> batch_scan_files(const char* dir)
{
    static const char* EXTS[] = {
        ".gltf", ".glb", ".usd", ".usda", ".usdc", ".usdz", nullptr
    };

    std::vector<std::string> result;
    std::error_code ec;
    namespace fs = std::filesystem;

    for (auto& entry : fs::recursive_directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        auto ext = entry.path().extension().string();
        for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        for (int i = 0; EXTS[i]; ++i) {
            if (ext == EXTS[i]) {
                result.push_back(entry.path().string());
                break;
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::string batch_sanitize(const std::string& s, int max_len)
{
    std::string out;
    out.reserve(s.size());
    bool last_under = true;  // start true to suppress leading underscore
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-') {
            out += static_cast<char>(c);
            last_under = false;
        } else if (!last_under) {
            out += '_';
            last_under = true;
        }
    }
    // Trim trailing underscore
    while (!out.empty() && out.back() == '_') out.pop_back();
    if (static_cast<int>(out.size()) > max_len) out.resize(max_len);
    if (out.empty()) out = "model";
    return out;
}

void batch_save(BatchProcessor& bp, const float* pixels_rgba_f32, int w, int h)
{
    namespace fs = std::filesystem;
    const std::string& src = bp.files[bp.current_idx];

    // ── Determine names from NIM result or filename fallback ─────────────────
    std::string category  = "prop";
    std::string nice_name = fs::path(src).stem().string();
    std::string obj_type  = nice_name;

    if (bp.nim_result.success) {
        if (bp.nim_result.category[0] != '\0')  category  = bp.nim_result.category;
        if (bp.nim_result.nice_name[0] != '\0') nice_name = bp.nim_result.nice_name;
        if (bp.nim_result.object_type[0]!= '\0')obj_type  = bp.nim_result.object_type;
    }

    // Lowercase category for consistent folder naming
    for (auto& c : category) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));

    std::string cat_safe  = batch_sanitize(category,  32);
    std::string name_safe = batch_sanitize(nice_name, 48);

    fs::path out_dir = fs::path(bp.config.output_dir) / cat_safe / name_safe;

    // ── Skip existing ─────────────────────────────────────────────────────────
    if (bp.config.skip_existing && fs::exists(out_dir / "render.jpg")) {
        BatchLogEntry e;
        e.source_path = src;
        e.category    = cat_safe;
        e.nice_name   = name_safe;
        e.out_dir     = out_dir.string();
        bp.log.push_back(e);
        snprintf(bp.status, sizeof(bp.status), "Skipped (exists): %s/%s",
                 cat_safe.c_str(), name_safe.c_str());
        return;
    }

    // ── Create output directory ───────────────────────────────────────────────
    std::error_code ec;
    fs::create_directories(out_dir, ec);

    // ── Write render.jpg ──────────────────────────────────────────────────────
    auto jpeg = encode_jpeg(pixels_rgba_f32, w, h, bp.config.jpeg_quality);
    if (!jpeg.empty()) {
        auto jpg_path = (out_dir / "render.jpg").string();
        FILE* fj = nullptr;
        fopen_s(&fj, jpg_path.c_str(), "wb");
        if (fj) { fwrite(jpeg.data(), 1, jpeg.size(), fj); fclose(fj); }
    }

    // ── Write metadata.json ───────────────────────────────────────────────────
    {
        char date_str[32] = "unknown";
        time_t t = time(nullptr);
        struct tm tm_buf{};
        localtime_s(&tm_buf, &t);
        strftime(date_str, sizeof(date_str), "%Y-%m-%d", &tm_buf);

        auto json_path = (out_dir / "metadata.json").string();
        FILE* fm = nullptr;
        fopen_s(&fm, json_path.c_str(), "w");
        if (FILE* f = fm) {
            fprintf(f, "{\n");
            fprintf(f, "  \"source\": \"%s\",\n",      json_esc(src).c_str());
            fprintf(f, "  \"category\": \"%s\",\n",    json_esc(cat_safe).c_str());
            fprintf(f, "  \"object\": \"%s\",\n",      json_esc(obj_type).c_str());
            fprintf(f, "  \"nice_name\": \"%s\",\n",   json_esc(name_safe).c_str());
            fprintf(f, "  \"render_resolution\": [%d, %d],\n", w, h);
            fprintf(f, "  \"frames_accumulated\": %d,\n", bp.config.target_frames);
            fprintf(f, "  \"nim_used\": %s,\n",        bp.nim_result.success ? "true" : "false");
            fprintf(f, "  \"date\": \"%s\"\n",         date_str);
            fprintf(f, "}\n");
            fclose(f);
        }
    }

    BatchLogEntry e;
    e.source_path = src;
    e.category    = cat_safe;
    e.nice_name   = name_safe;
    e.out_dir     = out_dir.string();
    bp.log.push_back(e);

    snprintf(bp.status, sizeof(bp.status), "Saved: %s/%s",
             cat_safe.c_str(), name_safe.c_str());
}

void batch_start(BatchProcessor& bp)
{
    bp.files = batch_scan_files(bp.config.input_dir);
    if (bp.files.empty()) {
        snprintf(bp.status, sizeof(bp.status),
                 "No models found in: %.200s", bp.config.input_dir);
        return;
    }
    bp.current_idx  = 0;
    bp.nim_busy     = false;
    bp.nim_started  = false;
    bp.nim_result   = {};
    bp.log.clear();
    bp.do_load      = true;
    bp.state        = BatchState::Loading;
    snprintf(bp.status, sizeof(bp.status),
             "Loading [1/%d]...", static_cast<int>(bp.files.size()));
}

void batch_stop(BatchProcessor& bp)
{
    bp.state = BatchState::Stopped;
    snprintf(bp.status, sizeof(bp.status),
             "Stopped at %d / %d", bp.current_idx, static_cast<int>(bp.files.size()));
}

// ─────────────────────────────────────────────────────────────────────────────
//  ImGui panel
// ─────────────────────────────────────────────────────────────────────────────

void batch_panel_draw(BatchProcessor& bp)
{
    ImGui::SetNextWindowSize(ImVec2(420, 360), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Batch Asset Library")) { ImGui::End(); return; }

    bool running = (bp.state == BatchState::Loading    ||
                    bp.state == BatchState::Rendering  ||
                    bp.state == BatchState::Recognizing);

    // ── Config ────────────────────────────────────────────────────────────────
    ImGui::BeginDisabled(running);
    ImGui::InputText("Input Folder",  bp.config.input_dir,  sizeof(bp.config.input_dir));
    ImGui::InputText("Output Folder", bp.config.output_dir, sizeof(bp.config.output_dir));
    ImGui::SliderInt("Frames / Model", &bp.config.target_frames, 8, 512);
    ImGui::SliderInt("JPEG Quality",   &bp.config.jpeg_quality,  50, 100);
    ImGui::Checkbox("Use NIM VLM", &bp.config.use_nim);
    ImGui::SameLine();
    ImGui::Checkbox("Skip Existing",   &bp.config.skip_existing);
    ImGui::EndDisabled();

    ImGui::Separator();

    // ── Controls ──────────────────────────────────────────────────────────────
    if (running) {
        if (ImGui::Button("Stop")) batch_stop(bp);
        ImGui::SameLine();
        float progress = bp.files.empty()
            ? 0.f
            : static_cast<float>(bp.current_idx) / static_cast<float>(bp.files.size());
        char prog_label[32];
        snprintf(prog_label, sizeof(prog_label), "%d / %d",
                 bp.current_idx, static_cast<int>(bp.files.size()));
        ImGui::ProgressBar(progress, ImVec2(-1.f, 0.f), prog_label);
    } else {
        bool can_start = (bp.config.input_dir[0] != '\0');
        ImGui::BeginDisabled(!can_start);
        if (ImGui::Button("Start Batch Processing")) batch_start(bp);
        ImGui::EndDisabled();
        if (!can_start) {
            ImGui::SameLine();
            ImGui::TextDisabled("(set input folder first)");
        }

        if (bp.state == BatchState::AllDone) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "Done!");
        }
    }

    // ── Status ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.9f, 0.85f, 0.3f, 1.f), "Status: %s", bp.status);

    // ── Log ───────────────────────────────────────────────────────────────────
    if (!bp.log.empty()) {
        ImGui::Separator();
        ImGui::Text("Completed (%d):", static_cast<int>(bp.log.size()));
        float log_h = ImGui::GetTextLineHeightWithSpacing() * 5.5f;
        if (ImGui::BeginChild("##batch_log", ImVec2(0, log_h), true)) {
            for (auto it = bp.log.rbegin(); it != bp.log.rend(); ++it) {
                ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f),
                    "%s / %s", it->category.c_str(), it->nice_name.c_str());
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", it->source_path.c_str());
            }
        }
        ImGui::EndChild();
    }

    ImGui::End();
}
