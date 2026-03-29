#include "gpu_arch_window.h"
#include <imgui.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

// ── Topology lookup ───────────────────────────────────────────────────────────
// GPC/TPC/SM-per-TPC counts are architecture constants — not queryable via CUDA API.
// SM count, L2 size, memory etc. are queried at runtime by the caller.

static ArchTopology topology_from_cc(int major, int minor)
{
    ArchTopology t;
    // Ada/Ampere defaults (most common)
    t.fp32_only    = 64;
    t.fp32_int32   = 64;
    t.tex_units    = 4;
    t.part_b_label = "FP32/INT32";

    int cc = major * 10 + minor;
    switch (cc) {
    case 70:  // Volta GV100 — 64 FP32 + 32 FP64 per SM, 32 FP64 (datacenter)
        t.gpcs=6; t.tpcs_per_gpc=7; t.sms_per_tpc=2;
        t.fp32_only=64; t.fp32_int32=32;
        t.tensor_cores=8; t.rt_cores=0; t.l1_shared_kb=96;
        t.ldst_units=32; t.sfu_units=16; t.fp64_units=32;
        t.warp_schedulers=4; t.reg_file_kb=256;
        t.tensor_label="Gen 1 (FP16)"; t.rt_label="None";
        t.part_b_label="FP64";
        break;
    case 75:  // Turing TU10x — 64 FP32 + 64 INT32 (parallel, separate pipelines)
        t.gpcs=6; t.tpcs_per_gpc=6; t.sms_per_tpc=2;
        t.fp32_only=64; t.fp32_int32=64;
        t.tensor_cores=8; t.rt_cores=1; t.l1_shared_kb=96;
        t.ldst_units=32; t.sfu_units=16; t.fp64_units=2;
        t.warp_schedulers=4; t.reg_file_kb=256;
        t.tensor_label="Gen 2 (INT8/FP16)"; t.rt_label="Gen 1";
        t.part_b_label="INT32";
        break;
    case 80:  // Ampere GA100 (A100) — no RT cores, 32 FP64 (datacenter)
        t.gpcs=8; t.tpcs_per_gpc=8; t.sms_per_tpc=2;
        t.tensor_cores=4; t.rt_cores=0; t.l1_shared_kb=192;
        t.ldst_units=32; t.sfu_units=16; t.fp64_units=32;
        t.warp_schedulers=4; t.reg_file_kb=256;
        t.tensor_label="Gen 3 (TF32/BF16/INT8)"; t.rt_label="None";
        break;
    case 86: case 87:  // Ampere GA102/GA104 (RTX 3090 / 3080 Ti) — consumer, 2 FP64
        t.gpcs=7; t.tpcs_per_gpc=6; t.sms_per_tpc=2;
        t.tensor_cores=4; t.rt_cores=1; t.l1_shared_kb=128;
        t.ldst_units=32; t.sfu_units=16; t.fp64_units=2;
        t.warp_schedulers=4; t.reg_file_kb=256;
        t.tensor_label="Gen 3 (TF32/BF16/INT8)"; t.rt_label="Gen 2";
        break;
    case 89:  // Ada Lovelace AD102 (RTX 4090) — 128 of 144 die SMs enabled, 2 FP64
        t.gpcs=12; t.tpcs_per_gpc=6; t.sms_per_tpc=2;
        t.tensor_cores=4; t.rt_cores=1; t.l1_shared_kb=128;
        t.ldst_units=32; t.sfu_units=16; t.fp64_units=2;
        t.warp_schedulers=4; t.reg_file_kb=256;
        t.tensor_label="Gen 4 (FP8/INT8)"; t.rt_label="Gen 3";
        break;
    case 90:  // Hopper GH100 — no consumer RT cores, 32 FP64 (datacenter)
        t.gpcs=8; t.tpcs_per_gpc=9; t.sms_per_tpc=2;
        t.tensor_cores=4; t.rt_cores=0; t.l1_shared_kb=228;
        t.ldst_units=32; t.sfu_units=16; t.fp64_units=32;
        t.warp_schedulers=4; t.reg_file_kb=256;
        t.tensor_label="Gen 4 (FP8/INT8)"; t.rt_label="None";
        break;
    default:
        t.gpcs=4; t.tpcs_per_gpc=4; t.sms_per_tpc=2;
        t.tensor_cores=4; t.rt_cores=0; t.l1_shared_kb=64;
        t.ldst_units=32; t.sfu_units=16; t.fp64_units=2;
        t.warp_schedulers=4; t.reg_file_kb=256;
        t.tensor_label="?"; t.rt_label="?";
        break;
    }
    t.full_die_sms = t.gpcs * t.tpcs_per_gpc * t.sms_per_tpc;
    return t;
}

void gpu_arch_window_init(GpuArchWindowState& s, int major, int minor, int l2_mb)
{
    s.topo        = topology_from_cc(major, minor);
    s.topo_ready  = true;
    s.l2_cache_mb = l2_mb;
}

// ── Color palette ─────────────────────────────────────────────────────────────

// Active — bright, saturated
static constexpr ImU32 COL_FP32_ACT  = IM_COL32( 80, 155, 255, 255);  // vivid blue
static constexpr ImU32 COL_FPI_ACT   = IM_COL32(  0, 220, 245, 255);  // vivid cyan
static constexpr ImU32 COL_TC_ACT    = IM_COL32( 55, 255,  80, 255);  // vivid green
static constexpr ImU32 COL_RT_ACT    = IM_COL32(255, 165,  30, 255);  // vivid orange
static constexpr ImU32 COL_TEX_ACT   = IM_COL32(200,  75, 255, 255);  // vivid purple
static constexpr ImU32 COL_L1_ACT    = IM_COL32(250, 205,  20, 255);  // vivid amber
static constexpr ImU32 COL_LDST_ACT  = IM_COL32(255, 110,  90, 255);  // vivid coral/red-orange
static constexpr ImU32 COL_SFU_ACT   = IM_COL32(180, 255, 180, 255);  // vivid light green
static constexpr ImU32 COL_FP64_ACT  = IM_COL32(220,  80, 220, 255);  // vivid magenta
static constexpr ImU32 COL_WS_ACT    = IM_COL32(255, 200,  80, 255);  // vivid gold/amber

// Idle — very dark fill + dim type-coloured border so you can still read what it is
static constexpr ImU32 COL_CORE_IDLE = IM_COL32( 38,  44,  56, 255);  // near-black core pixel
static constexpr ImU32 COL_BLOCK_IDL = IM_COL32( 35,  40,  50, 255);  // near-black block fill
static constexpr ImU32 COL_FP32_DIM  = IM_COL32( 22,  48,  88, 255);  // dim blue border
static constexpr ImU32 COL_FPI_DIM   = IM_COL32(  0,  55,  68, 255);  // dim cyan border
static constexpr ImU32 COL_TC_DIM    = IM_COL32( 22,  65,  22, 255);  // dim green border
static constexpr ImU32 COL_RT_DIM    = IM_COL32( 88,  48,   8, 255);  // dim orange border
static constexpr ImU32 COL_TEX_DIM   = IM_COL32( 58,  18,  72, 255);  // dim purple border
static constexpr ImU32 COL_L1_DIM    = IM_COL32( 78,  62,   8, 255);  // dim amber border

// GPC / SM grid
static constexpr ImU32 COL_SM_ACTIVE = IM_COL32( 25, 215,  75, 255);
static constexpr ImU32 COL_SM_IDLE   = IM_COL32( 62,  78,  95, 255);
static constexpr ImU32 COL_DISABLED  = IM_COL32( 32,  32,  32, 255);
static constexpr ImU32 COL_BLOCK_IDL_FILL = IM_COL32( 52,  56,  62, 255);  // grey fill for idle blocks
static constexpr ImU32 COL_BLOCK_IDL_BORD = IM_COL32( 78,  82,  90, 255);  // grey border for idle blocks
static constexpr ImU32 COL_BLOCK_IDL_TEXT = IM_COL32( 95,  99, 108, 255);  // grey label for idle blocks
static constexpr ImU32 COL_GPC_BASE  = IM_COL32( 38,  50,  68, 255);
static constexpr ImU32 COL_BORDER    = IM_COL32( 85, 108, 128, 255);
static constexpr ImU32 COL_BORDER_ACT= IM_COL32( 50, 210,  80, 200);
static constexpr ImU32 COL_TEXT      = IM_COL32(220, 220, 220, 255);
static constexpr ImU32 COL_DIM       = IM_COL32(140, 140, 140, 255);
static constexpr ImU32 COL_HOVER     = IM_COL32( 65,  88, 112, 255);

// ── Utilities ─────────────────────────────────────────────────────────────────

static inline bool sm_is_active(const GpuArchWindowState& s, int sm)
{
    if (sm < 0 || sm >= GpuArchWindowState::MAX_SM_WORDS * 32) return false;
    return (s.sm_active[sm >> 5] >> (sm & 31)) & 1u;
}

static inline int gpc_first_sm(const ArchTopology& t, int gpc)
{
    return gpc * t.tpcs_per_gpc * t.sms_per_tpc;
}

// Draw N small squares in a rows×cols grid.
// active=true   → bright active_col squares
// disabled=true → flat uniform grey (SM not present on die)
// otherwise     → near-black squares with a subtle dim_col tint (idle)
static constexpr ImU32 COL_CORE_DIS = IM_COL32(42, 44, 48, 255);  // flat grey for disabled
static constexpr ImU32 COL_BORD_DIS = IM_COL32(55, 57, 62, 255);  // subtle grey border

static void draw_core_grid(ImDrawList* dl, ImVec2 origin,
                            int count, int cols,
                            float cell, float gap,
                            ImU32 active_col, ImU32 /*dim_col*/,
                            bool active, bool disabled = false)
{
    for (int i = 0; i < count; i++) {
        float x0 = origin.x + (i % cols) * (cell + gap);
        float y0 = origin.y + (i / cols) * (cell + gap);
        if (disabled) {
            dl->AddRectFilled({ x0, y0 }, { x0 + cell, y0 + cell }, COL_CORE_DIS, 1.f);
            dl->AddRect({ x0, y0 }, { x0 + cell, y0 + cell }, COL_BORD_DIS, 1.f);
        } else if (active) {
            dl->AddRectFilled({ x0, y0 }, { x0 + cell, y0 + cell }, active_col, 1.f);
        } else {
            // Idle: plain grey — same visual language as tensor/RT/TEX blocks
            dl->AddRectFilled({ x0, y0 }, { x0 + cell, y0 + cell }, COL_BLOCK_IDL_FILL, 1.f);
            dl->AddRect({ x0, y0 }, { x0 + cell, y0 + cell }, COL_BLOCK_IDL_BORD, 1.f);
        }
    }
}

// Draw a section header with a status dot:
//   active=true  → green filled dot
//   disabled=true → flat grey dot (SM not present)
//   otherwise    → hollow dim dot (idle)
static void draw_section_header(ImDrawList* dl, const char* label,
                                 ImVec4 label_col_act, ImVec4 label_col_idle,
                                 bool active, bool disabled = false)
{
    ImVec2  p   = ImGui::GetCursorScreenPos();
    float   th  = ImGui::GetTextLineHeight();
    float   r   = 4.5f;
    float   cx  = p.x + r + 1.f;
    float   cy  = p.y + th * 0.5f;
    ImU32   dot = active   ? IM_COL32(40, 245, 100, 255)
                : disabled ? IM_COL32(55, 57, 65, 255)
                :            IM_COL32(65, 70, 80, 255);
    dl->AddCircleFilled({ cx, cy }, r, dot);
    if (!active)
        dl->AddCircle({ cx, cy }, r, disabled ? IM_COL32(68, 70, 78, 255) : IM_COL32(85, 90, 100, 255));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + r * 2.f + 6.f);
    ImVec4 col = active   ? label_col_act
               : disabled ? ImVec4(0.38f, 0.39f, 0.42f, 1.f)
               :            label_col_idle;
    ImGui::TextColored(col, "%s", label);
}

// ── Level: GPU overview ───────────────────────────────────────────────────────

static void draw_level_gpu(GpuArchWindowState& s, const char* hw_name, int sm_count)
{
    const ArchTopology& t   = s.topo;
    ImDrawList*         dl  = ImGui::GetWindowDrawList();

    ImGui::Text("GPU: %s", hw_name);
    ImGui::Text("%d / %d SMs enabled  ·  %d GPCs  ·  %d TPCs/GPC  ·  %d SMs/TPC",
                sm_count, t.full_die_sms, t.gpcs, t.tpcs_per_gpc, t.sms_per_tpc);

    // "Real counters" checkbox — enables CUPTI hardware profiling (~5-15% GPU overhead)
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, s.profiling_enabled
        ? ImVec4(0.4f, 1.f, 0.5f, 1.f)   // green when active
        : ImVec4(0.6f, 0.6f, 0.6f, 1.f)); // grey when off
    ImGui::Checkbox("Real counters (CUPTI)", &s.profiling_enabled);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Query actual hardware pipe utilisation via CUPTI.\n"
            "Shows real TEX / LD-ST / SFU / Tensor activity.\n"
            "Adds ~5-15%% GPU overhead while enabled.");
    if (s.profiling_enabled && s.cupti_available && s.has_unit_counters) {
        ImGui::SameLine();
        ImGui::TextDisabled("  live HW data");
    } else if (s.profiling_enabled && s.cupti_available) {
        ImGui::SameLine();
        ImGui::TextDisabled("  live HW data (partial)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "CUPTI is active, but some unit metrics are unsupported on this driver/GPU.\n"
                "Supported units still update from real hardware counters.");
    } else if (s.profiling_enabled && !s.cupti_available) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.4f, 0.4f, 1.f));
        ImGui::Text("  not available");
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "CUPTI init failed.\n"
                "Common causes:\n"
                "  - Another profiler (Nsight) is attached\n"
                "  - Driver does not allow GPU profiling\n"
                "  - cupti64_*.dll not found next to exe");
    }
    ImGui::Spacing();

    // ── GPC grid ─────────────────────────────────────────────────────────────
    const int   COLS   = (t.gpcs <= 6) ? t.gpcs : 4;   // 3×4 grid for 12 GPCs
    const float PAD    = 6.f;
    const float GPC_H  = 68.f;
    const float avail  = ImGui::GetContentRegionAvail().x;
    const float GPC_W  = (avail - (COLS - 1) * PAD) / (float)COLS;
    const int   sms_pg = t.tpcs_per_gpc * t.sms_per_tpc;

    for (int g = 0; g < t.gpcs; g++) {
        if (g % COLS != 0) ImGui::SameLine(0.f, PAD);

        int first   = gpc_first_sm(t, g);
        int enabled = std::max(0, std::min(first + sms_pg, sm_count) - first);
        int active  = 0;
        for (int si = first; si < first + enabled; si++)
            if (sm_is_active(s, si)) active++;

        ImVec2 p0  = ImGui::GetCursorScreenPos();
        ImVec2 sz  = { GPC_W, GPC_H };
        char   bid[16]; snprintf(bid, sizeof(bid), "##g%d", g);
        ImGui::InvisibleButton(bid, sz);
        bool hov = ImGui::IsItemHovered();
        bool clk = ImGui::IsItemClicked();

        // Green tint proportional to active SM fraction
        float frac = enabled > 0 ? (float)active / enabled : 0.f;
        ImU32 fill = hov ? COL_HOVER
                        : IM_COL32((int)(38 + frac * 25),
                                   (int)(50 + frac * 120),
                                   (int)(68),
                                   255);
        ImU32 bord = active > 0 ? COL_BORDER_ACT : COL_BORDER;
        dl->AddRectFilled(p0, { p0.x + GPC_W, p0.y + GPC_H }, fill, 5.f);
        dl->AddRect       (p0, { p0.x + GPC_W, p0.y + GPC_H }, bord, 5.f);

        char tmp[32];
        snprintf(tmp, sizeof(tmp), "GPC %d", g + 1);
        dl->AddText({ p0.x + 7, p0.y + 7 }, COL_TEXT, tmp);

        if (enabled < sms_pg)
            snprintf(tmp, sizeof(tmp), "%d/%d SMs", enabled, sms_pg);
        else
            snprintf(tmp, sizeof(tmp), "%d SMs", enabled);
        dl->AddText({ p0.x + 7, p0.y + 26 }, COL_DIM, tmp);

        if (active > 0) {
            snprintf(tmp, sizeof(tmp), "%d active", active);
            dl->AddText({ p0.x + 7, p0.y + 46 }, COL_SM_ACTIVE, tmp);
        }

        if (hov) ImGui::SetTooltip("GPC %d\n%d/%d SMs enabled\n%d active\nClick to inspect",
                                   g + 1, enabled, sms_pg, active);
        if (clk) { s.sel_gpc = g; s.level = ArchLevel::GPC; }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ── Unit activity sampling toggle ────────────────────────────────────────
    // ── Unit totals + active counts ───────────────────────────────────────────
    {
        // Active SM count from bitmask — used to estimate SM-correlated unit activity.
        int active_sms = 0;
        for (int i = 0; i < sm_count; i++)
            if ((s.sm_active[i >> 5] >> (i & 31)) & 1u) active_sms++;

        // Derive SM-correlated estimates from the SM bitmask only when we do not
        // have real HW unit counters from CUPTI.
        if (!s.has_unit_counters && sm_count > 0) {
            float frac = (float)active_sms / sm_count;
            s.cuda_active_pct = frac * 100.f;
            s.ldst_active_pct = frac * 100.f;
            s.sfu_active_pct  = frac * 100.f;
        }
        // tensor / rt / tex are left as-is — caller sets them from render mode
        // (rt_active_pct when OptiX is running, tensor_active_pct when DLSS is on, etc.)

        int total_fp32  = sm_count * (t.fp32_only + t.fp32_int32);
        int total_tc    = sm_count * t.tensor_cores;
        int total_rt    = sm_count * t.rt_cores;
        int total_tex   = sm_count * t.tex_units;
        int total_l1_mb = sm_count * t.l1_shared_kb / 1024;
        int total_ldst  = sm_count * t.ldst_units;
        int total_sfu   = sm_count * t.sfu_units;
        int total_fp64  = sm_count * t.fp64_units;
        int total_ws    = sm_count * t.warp_schedulers;

        constexpr float C0 = 145.f, C1 = 280.f;
        char buf2[64];

        // SM-correlated row: always shows "~active / total" derived from SM bitmask.
        auto row_sm = [&](ImVec4 col, const char* label,
                          int total, float pct, const char* detail) {
            ImGui::TextColored(col, "%s", label);
            ImGui::SameLine(C0);
            int active = (int)(pct / 100.f * total + 0.5f);
            if (s.has_unit_counters)
                snprintf(buf2, sizeof(buf2), "%d / %d", active, total);
            else
                snprintf(buf2, sizeof(buf2), "~%d / %d", active, total);
            ImGui::Text("%s", buf2);
            if (detail && detail[0]) { ImGui::SameLine(C1); ImGui::TextDisabled("%s", detail); }
        };

        // Workload-specific row: shows "active / total" if caller set pct > 0,
        // otherwise shows "— / total" (grey) — not estimated, not fabricated.
        auto row_wp = [&](ImVec4 col, const char* label,
                          int total, float pct, const char* detail) {
            ImGui::TextColored(col, "%s", label);
            ImGui::SameLine(C0);
            if (pct > 0.f || s.has_unit_counters) {
                int active = (int)(pct / 100.f * total + 0.5f);
                snprintf(buf2, sizeof(buf2), "%d / %d", active, total);
                ImGui::Text("%s", buf2);
            } else {
                snprintf(buf2, sizeof(buf2), "-- / %d", total);
                ImGui::TextDisabled("%s", buf2);
            }
            if (detail && detail[0]) { ImGui::SameLine(C1); ImGui::TextDisabled("%s", detail); }
        };

        row_sm({ 0.55f, 0.75f, 1.f,  1.f }, "CUDA Cores",
               total_fp32, s.cuda_active_pct, "FP32 + FP32/INT32 per SM");
        row_sm({ 1.f,   0.43f, 0.35f,1.f }, "LD/ST Units",
               total_ldst, s.ldst_active_pct, "load / store / atomics");
        row_sm({ 0.70f, 1.f,   0.70f,1.f }, "SFU",
               total_sfu, s.sfu_active_pct,  "sin / cos / rcp / rsqrt");
        row_sm({ 0.87f, 0.32f, 0.87f,1.f }, "FP64 Units",
               total_fp64, s.cuda_active_pct,
               t.fp64_units == 32 ? "double prec. (datacenter)" : "double prec. (1/64 rate)");
        row_sm({ 1.f,   0.80f, 0.32f,1.f }, "Warp Schedulers",
               total_ws, s.cuda_active_pct,  "1 warp issued / scheduler / cycle");

        ImGui::Separator();

        // Workload-specific: grey until caller sets the pct fields
        snprintf(buf2, sizeof(buf2), "%s", t.tensor_label);
        row_wp({ 0.22f, 1.f,   0.32f,1.f }, "Tensor Cores",
               total_tc, s.tensor_active_pct, buf2);
        if (total_rt > 0) {
            snprintf(buf2, sizeof(buf2), "%s", t.rt_label);
            row_wp({ 1.f,   0.66f, 0.12f,1.f }, "RT Cores",
                   total_rt, s.rt_active_pct, buf2);
        }
        row_wp({ 0.80f, 0.30f, 1.f,  1.f }, "TEX Units",
               total_tex, s.tex_active_pct, nullptr);

        // L1/Shared — static size, no activity metric
        ImGui::TextColored({ 0.98f, 0.82f, 0.10f, 1.f }, "L1 / Shared");
        ImGui::SameLine(C0);
        if (total_l1_mb > 0) ImGui::Text("%d MB total", total_l1_mb);
        else                  ImGui::Text("%d KB × %d SMs", t.l1_shared_kb, sm_count);
        ImGui::SameLine(C1);
        ImGui::TextDisabled("(%d active SMs)", active_sms);

        // Register file — static size
        ImGui::TextColored({ 0.75f, 0.92f, 1.f, 1.f }, "Reg File");
        ImGui::SameLine(C0);
        ImGui::Text("%d KB × %d SMs  =  %d MB", t.reg_file_kb, sm_count,
                    sm_count * t.reg_file_kb / 1024);
        ImGui::SameLine(C1);
        ImGui::TextDisabled("65536 × 32-bit regs / SM");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ── Memory / live stats ───────────────────────────────────────────────────
    constexpr float LBL_W = 120.f;
    constexpr float BAR_W = 240.f;
    char buf[64];

    // L2 cache (static — no live utilisation API)
    ImGui::Text("L2 Cache:");
    ImGui::SameLine(LBL_W);
    if (s.l2_cache_mb > 0)
        ImGui::Text("%d MB", s.l2_cache_mb);
    else
        ImGui::Text("unknown");
    ImGui::SameLine();
    ImGui::TextDisabled("  (no live API)");

    ImGui::Spacing();

    // VRAM used / total
    float vram_tot_gb = (float)(s.vram_total) / (1024.f * 1024.f * 1024.f);
    float vram_use_gb = (float)(s.vram_used)  / (1024.f * 1024.f * 1024.f);
    ImGui::Text("VRAM used:");
    ImGui::SameLine(LBL_W);
    snprintf(buf, sizeof(buf), "%.1f / %.1f GB", vram_use_gb, vram_tot_gb);
    ImGui::ProgressBar(vram_tot_gb > 0.f ? vram_use_gb / vram_tot_gb : 0.f,
                       { BAR_W, 0.f }, buf);

    // DRAM controller busy %
    ImGui::Text("DRAM busy:");
    ImGui::SameLine(LBL_W);
    snprintf(buf, sizeof(buf), "%u%%", s.mem_util_pct);
    ImGui::ProgressBar(s.mem_util_pct / 100.f, { BAR_W, 0.f }, buf);

    // SM utilisation %
    ImGui::Text("SM util:");
    ImGui::SameLine(LBL_W);
    snprintf(buf, sizeof(buf), "%u%%", s.gpu_util_pct);
    ImGui::ProgressBar(s.gpu_util_pct / 100.f, { BAR_W, 0.f }, buf);
}

// ── Level: GPC detail — SM grid grouped by TPC ───────────────────────────────

static void draw_level_gpc(GpuArchWindowState& s, int sm_count)
{
    const ArchTopology& t  = s.topo;
    ImDrawList*         dl = ImGui::GetWindowDrawList();

    int first_sm = gpc_first_sm(t, s.sel_gpc);

    ImGui::Text("GPC %d  ·  %d TPCs  ·  %d SMs per TPC",
                s.sel_gpc + 1, t.tpcs_per_gpc, t.sms_per_tpc);
    ImGui::Spacing();

    const float TPC_PAD   = 8.f;   // gap between TPC containers
    const float INNER_PAD = 5.f;   // padding inside TPC container around SMs
    const float SM_GAP    = 4.f;   // gap between the two SM blocks inside a TPC
    const float HEADER_H  = ImGui::GetTextLineHeight() + 8.f;  // TPC label bar height
    const float SM_H      = 58.f;
    const float avail     = ImGui::GetContentRegionAvail().x;
    const float COL_W     = (avail - (t.tpcs_per_gpc - 1) * TPC_PAD) / (float)t.tpcs_per_gpc;
    const float SM_W      = COL_W - INNER_PAD * 2.f;
    // Total height of one TPC container box
    const float TPC_H     = HEADER_H + INNER_PAD
                           + t.sms_per_tpc * SM_H
                           + (t.sms_per_tpc - 1) * SM_GAP
                           + INNER_PAD;

    // Use a draw-list splitter so TPC container backgrounds (channel 0) sit
    // visually behind the SM blocks and interaction buttons (channel 1).
    ImDrawListSplitter splitter;
    splitter.Split(dl, 2);

    for (int tpc = 0; tpc < t.tpcs_per_gpc; tpc++) {
        if (tpc != 0) ImGui::SameLine(0.f, TPC_PAD);
        ImGui::BeginGroup();

        ImVec2 tpc_p0 = ImGui::GetCursorScreenPos();
        ImVec2 tpc_p1 = { tpc_p0.x + COL_W, tpc_p0.y + TPC_H };

        // Count active/enabled SMs in this TPC for header tinting
        int tpc_active  = 0;
        int tpc_enabled = 0;
        for (int si = 0; si < t.sms_per_tpc; si++) {
            int gsm = first_sm + tpc * t.sms_per_tpc + si;
            if (gsm < sm_count) {
                tpc_enabled++;
                if (sm_is_active(s, gsm)) tpc_active++;
            }
        }

        // ── Channel 0: TPC container box (drawn behind SM content) ───────────
        splitter.SetCurrentChannel(dl, 0);
        {
            ImU32 box_fill  = IM_COL32(22, 27, 36, 255);
            ImU32 box_bord  = tpc_active  > 0 ? IM_COL32(48, 185, 72, 170)
                            : tpc_enabled > 0 ? IM_COL32(55, 68, 84, 255)
                            :                   IM_COL32(45, 48, 54, 255);
            ImU32 hdr_fill  = tpc_active  > 0 ? IM_COL32(20, 52, 30, 255)
                            : tpc_enabled > 0 ? IM_COL32(28, 35, 48, 255)
                            :                   IM_COL32(26, 28, 32, 255);

            // Outer box
            dl->AddRectFilled(tpc_p0, tpc_p1, box_fill, 6.f);
            dl->AddRect      (tpc_p0, tpc_p1, box_bord, 6.f, 0, 1.5f);

            // Header bar (rounded top only)
            dl->AddRectFilled(tpc_p0, { tpc_p1.x, tpc_p0.y + HEADER_H },
                              hdr_fill, 6.f, ImDrawFlags_RoundCornersTop);

            // Divider line between header and SM area
            dl->AddLine({ tpc_p0.x + 1.f,        tpc_p0.y + HEADER_H },
                        { tpc_p1.x - 1.f,        tpc_p0.y + HEADER_H },
                        box_bord, 1.f);

            // TPC label centred in header
            char tpc_lbl[16]; snprintf(tpc_lbl, sizeof(tpc_lbl), "TPC %d", tpc + 1);
            ImVec2 lsz = ImGui::CalcTextSize(tpc_lbl);
            ImU32  lcol = tpc_active  > 0 ? IM_COL32( 80, 220, 110, 255)
                        : tpc_enabled > 0 ? IM_COL32(130, 155, 185, 255)
                        :                   IM_COL32( 90,  92,  98, 255);
            dl->AddText({ tpc_p0.x + (COL_W - lsz.x) * 0.5f,
                          tpc_p0.y + (HEADER_H - lsz.y) * 0.5f }, lcol, tpc_lbl);
        }

        // ── Channel 1: SM blocks (interactive, drawn on top) ─────────────────
        splitter.SetCurrentChannel(dl, 1);

        // Reserve the header area so ImGui cursor lands below it
        ImGui::Dummy({ COL_W, HEADER_H + INNER_PAD });

        for (int s_in_tpc = 0; s_in_tpc < t.sms_per_tpc; s_in_tpc++) {
            if (s_in_tpc != 0) {
                // Position cursor: indent by INNER_PAD, advance by SM_GAP
                ImVec2 prev = ImGui::GetCursorScreenPos();
                ImGui::SetCursorScreenPos({ tpc_p0.x + INNER_PAD, prev.y + SM_GAP - ImGui::GetStyle().ItemSpacing.y });
            } else {
                ImVec2 cur = ImGui::GetCursorScreenPos();
                ImGui::SetCursorScreenPos({ tpc_p0.x + INNER_PAD, cur.y });
            }

            int   gsm     = first_sm + tpc * t.sms_per_tpc + s_in_tpc;
            bool  enabled = gsm < sm_count;
            bool  active  = enabled && sm_is_active(s, gsm);

            ImVec2 p0  = ImGui::GetCursorScreenPos();
            char   bid[24]; snprintf(bid, sizeof(bid), "##sm%d", gsm);
            ImGui::InvisibleButton(bid, { SM_W, SM_H });
            bool hov = ImGui::IsItemHovered();
            bool clk = ImGui::IsItemClicked();

            ImU32 fill = !enabled ? (hov ? IM_COL32(58, 62, 70, 255) : COL_DISABLED)
                       :  active  ? COL_SM_ACTIVE
                       :  hov     ? COL_HOVER
                       :            COL_SM_IDLE;
            ImU32 bord = !enabled ? IM_COL32(65, 68, 75, 255)
                       :  active  ? COL_BORDER_ACT
                       :            COL_BORDER;
            dl->AddRectFilled(p0, { p0.x + SM_W, p0.y + SM_H }, fill, 4.f);
            dl->AddRect      (p0, { p0.x + SM_W, p0.y + SM_H }, bord, 4.f);

            char sm_lbl[16]; snprintf(sm_lbl, sizeof(sm_lbl), "SM %d", gsm + 1);
            dl->AddText({ p0.x + 5, p0.y + 5 }, COL_TEXT, sm_lbl);

            if (!enabled)
                dl->AddText({ p0.x + 5, p0.y + 24 }, IM_COL32(100, 100, 108, 255), "disabled");
            else if (active)
                dl->AddText({ p0.x + 5, p0.y + 24 }, IM_COL32(15, 255, 90, 255), "ACTIVE");
            else
                dl->AddText({ p0.x + 5, p0.y + 24 }, IM_COL32(115, 135, 158, 255), "idle");

            if (hov)
                ImGui::SetTooltip("SM %d  (%s)\nPart of TPC %d\nClick to inspect internals",
                                  gsm + 1,
                                  !enabled ? "Disabled" : active ? "Active" : "Idle",
                                  tpc + 1);
            if (clk) { s.sel_sm = gsm; s.level = ArchLevel::SM; }
        }

        // Advance cursor past the entire TPC box before EndGroup
        ImVec2 cur = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos({ tpc_p0.x, tpc_p0.y + TPC_H });
        ImGui::Dummy({ COL_W, 0.f });

        ImGui::EndGroup();
    }

    splitter.Merge(dl);
}

// ── Level: SM internal breakdown ─────────────────────────────────────────────

static void draw_level_sm(GpuArchWindowState& s, int sm_count)
{
    const ArchTopology& t    = s.topo;
    ImDrawList*         dl   = ImGui::GetWindowDrawList();
    bool                enbl = s.sel_sm < sm_count;
    bool                act  = enbl && sm_is_active(s, s.sel_sm);

    // ── Status banner ─────────────────────────────────────────────────────────
    {
        ImVec2 p0  = ImGui::GetCursorScreenPos();
        float  bw  = ImGui::GetContentRegionAvail().x;
        float  bh  = 26.f;
        ImU32  bg  = act  ? IM_COL32(12, 55, 22, 255)
                   : enbl ? IM_COL32(32, 36, 44, 255)
                   :        IM_COL32(28, 28, 32, 255);   // disabled: darkest
        ImU32  fg  = act  ? IM_COL32(40, 245, 100, 255)
                   : enbl ? IM_COL32(110, 115, 125, 255)
                   :        IM_COL32(95, 98, 108, 255);
        ImU32  brd = act  ? IM_COL32(45, 210, 80, 200)
                   : enbl ? IM_COL32(70, 76, 88, 255)
                   :        IM_COL32(60, 62, 70, 255);
        dl->AddRectFilled(p0, { p0.x + bw, p0.y + bh }, bg, 4.f);
        dl->AddRect       (p0, { p0.x + bw, p0.y + bh }, brd, 4.f);
        float cy = p0.y + bh * 0.5f;
        dl->AddCircleFilled({ p0.x + 14.f, cy }, 5.f, fg);
        const char* state_str = act ? "ACTIVE" : (enbl ? "IDLE" : "DISABLED");
        char txt[48]; snprintf(txt, sizeof(txt), "SM %d  —  %s", s.sel_sm + 1, state_str);
        ImVec2 ts = ImGui::CalcTextSize(txt);
        dl->AddText({ p0.x + 26.f, p0.y + (bh - ts.y) * 0.5f }, fg, txt);
        ImGui::Dummy({ bw, bh });
    }
    ImGui::Spacing();

    ImGui::BeginChild("##sm_scroll", { 0.f, 0.f }, false, ImGuiWindowFlags_HorizontalScrollbar);

    char lbl[96];
    const float cell = 7.f, gap = 1.f, fp_cols = 8;

    bool dis = !enbl;  // shorthand: hardware not present on this GPU

    // Workload-specific units: only active when caller has set their pct fields.
    bool tc_act  = !dis && s.tensor_active_pct > 0.f;
    bool rt_act  = !dis && s.rt_active_pct     > 0.f;

    // SM-correlated units follow the SM bitmask — BUT if any workload unit is active
    // the SMs must be running shader code for it (OptiX raygen/closest-hit, DLSS etc.),
    // even if the CUDA pathtrace_kernel SM tracker never fired.
    bool sm_running = act || tc_act || rt_act;
    bool tex_act = !dis && s.tex_active_pct > 0.f;

    // ── Partition A: FP32-only ────────────────────────────────────────────────
    snprintf(lbl, sizeof(lbl), "FP32  (%d cores)", t.fp32_only);
    draw_section_header(dl, lbl, { 0.32f, 0.62f, 1.f, 1.f }, { 0.25f, 0.30f, 0.38f, 1.f }, sm_running, dis);
    {
        int    rows = (t.fp32_only + (int)fp_cols - 1) / (int)fp_cols;
        ImVec2 ori  = ImGui::GetCursorScreenPos();
        draw_core_grid(dl, ori, t.fp32_only, (int)fp_cols, cell, gap, COL_FP32_ACT, COL_FP32_DIM, sm_running, dis);
        ImGui::Dummy({ fp_cols * (cell + gap), rows * (cell + gap) + 2.f });
    }
    ImGui::Spacing();

    // ── Partition B: FP32/INT32 ───────────────────────────────────────────────
    if (t.fp32_int32 > 0) {
        snprintf(lbl, sizeof(lbl), "%s  (%d cores)", t.part_b_label, t.fp32_int32);
        draw_section_header(dl, lbl, { 0.f, 0.88f, 0.98f, 1.f }, { 0.f, 0.25f, 0.30f, 1.f }, sm_running, dis);
        {
            int    rows = (t.fp32_int32 + (int)fp_cols - 1) / (int)fp_cols;
            ImVec2 ori  = ImGui::GetCursorScreenPos();
            draw_core_grid(dl, ori, t.fp32_int32, (int)fp_cols, cell, gap, COL_FPI_ACT, COL_FPI_DIM, sm_running, dis);
            ImGui::Dummy({ fp_cols * (cell + gap), rows * (cell + gap) + 2.f });
        }
        ImGui::Spacing();
    }

    // ── Tensor Cores ──────────────────────────────────────────────────────────
    if (t.tensor_cores > 0) {
        snprintf(lbl, sizeof(lbl), "Tensor Cores  %d ×  %s", t.tensor_cores, t.tensor_label);
        draw_section_header(dl, lbl, { 0.22f, 1.f, 0.32f, 1.f }, { 0.12f, 0.28f, 0.12f, 1.f }, tc_act, dis);
        {
            const float TW = 56.f, TH = 40.f, TGAP = 6.f;
            ImVec2 ori = ImGui::GetCursorScreenPos();
            for (int i = 0; i < t.tensor_cores; i++) {
                float x0  = ori.x + i * (TW + TGAP);
                ImU32 fill = dis ? COL_CORE_DIS : tc_act ? COL_TC_ACT  : COL_BLOCK_IDL_FILL;
                ImU32 bord = dis ? COL_BORD_DIS : tc_act ? IM_COL32(45, 220, 65, 220) : COL_BLOCK_IDL_BORD;
                ImU32 tlbl = dis ? IM_COL32(65, 67, 72, 255)
                           : tc_act ? IM_COL32(10, 30, 10, 255) : COL_BLOCK_IDL_TEXT;
                dl->AddRectFilled({ x0, ori.y }, { x0 + TW, ori.y + TH }, fill, 5.f);
                dl->AddRect       ({ x0, ori.y }, { x0 + TW, ori.y + TH }, bord, 5.f);
                char tc[8]; snprintf(tc, sizeof(tc), "TC%d", i + 1);
                ImVec2 ts = ImGui::CalcTextSize(tc);
                dl->AddText({ x0 + (TW - ts.x) * 0.5f, ori.y + (TH - ts.y) * 0.5f }, tlbl, tc);
            }
            ImGui::Dummy({ t.tensor_cores * (TW + TGAP), TH + 2.f });
        }
        ImGui::Spacing();
    }

    // ── RT Core ───────────────────────────────────────────────────────────────
    if (t.rt_cores > 0) {
        snprintf(lbl, sizeof(lbl), "RT Core  %d ×  %s", t.rt_cores, t.rt_label);
        draw_section_header(dl, lbl, { 1.f, 0.66f, 0.12f, 1.f }, { 0.35f, 0.20f, 0.04f, 1.f }, rt_act, dis);
        {
            const float RW = 72.f, RH = 40.f, RGAP = 6.f;
            ImVec2 ori = ImGui::GetCursorScreenPos();
            for (int i = 0; i < t.rt_cores; i++) {
                float x0  = ori.x + i * (RW + RGAP);
                ImU32 fill = dis ? COL_CORE_DIS : rt_act ? COL_RT_ACT  : COL_BLOCK_IDL_FILL;
                ImU32 bord = dis ? COL_BORD_DIS : rt_act ? IM_COL32(255, 155, 20, 220) : COL_BLOCK_IDL_BORD;
                ImU32 tlbl = dis ? IM_COL32(65, 67, 72, 255)
                           : rt_act ? IM_COL32(30, 14, 4, 255) : COL_BLOCK_IDL_TEXT;
                dl->AddRectFilled({ x0, ori.y }, { x0 + RW, ori.y + RH }, fill, 5.f);
                dl->AddRect       ({ x0, ori.y }, { x0 + RW, ori.y + RH }, bord, 5.f);
                float th = ImGui::GetTextLineHeight();
                dl->AddText({ x0 + 8.f, ori.y + (RH - th) * 0.5f }, tlbl, "RT Core");
            }
            ImGui::Dummy({ t.rt_cores * (RW + RGAP), RH + 2.f });
        }
        ImGui::Spacing();
    }

    // ── Texture Units ─────────────────────────────────────────────────────────
    {
        snprintf(lbl, sizeof(lbl), "Texture Units  (%d)", t.tex_units);
        draw_section_header(dl, lbl, { 0.80f, 0.30f, 1.f, 1.f }, { 0.25f, 0.10f, 0.30f, 1.f }, tex_act, dis);
        {
            const float TUW = 50.f, TUH = 32.f, TUGAP = 6.f;
            ImVec2 ori = ImGui::GetCursorScreenPos();
            for (int i = 0; i < t.tex_units; i++) {
                float x0  = ori.x + i * (TUW + TUGAP);
                ImU32 fill = dis ? COL_CORE_DIS : tex_act ? COL_TEX_ACT  : COL_BLOCK_IDL_FILL;
                ImU32 bord = dis ? COL_BORD_DIS : tex_act ? IM_COL32(185, 60, 245, 220) : COL_BLOCK_IDL_BORD;
                ImU32 tlbl = dis ? IM_COL32(65, 67, 72, 255)
                           : tex_act ? COL_TEXT : COL_BLOCK_IDL_TEXT;
                dl->AddRectFilled({ x0, ori.y }, { x0 + TUW, ori.y + TUH }, fill, 4.f);
                dl->AddRect       ({ x0, ori.y }, { x0 + TUW, ori.y + TUH }, bord, 4.f);
                char tu[8]; snprintf(tu, sizeof(tu), "TEX%d", i + 1);
                ImVec2 ts = ImGui::CalcTextSize(tu);
                dl->AddText({ x0 + (TUW - ts.x) * 0.5f, ori.y + (TUH - ts.y) * 0.5f }, tlbl, tu);
            }
            ImGui::Dummy({ t.tex_units * (TUW + TUGAP), TUH + 2.f });
        }
        ImGui::Spacing();
    }

    // ── L1 / Shared Memory ────────────────────────────────────────────────────
    {
        snprintf(lbl, sizeof(lbl), "L1 Cache / Shared Memory  (%d KB, configurable split)",
                 t.l1_shared_kb);
        draw_section_header(dl, lbl, { 0.98f, 0.82f, 0.10f, 1.f }, { 0.32f, 0.26f, 0.04f, 1.f }, sm_running, dis);
        {
            float bw = std::min(ImGui::GetContentRegionAvail().x - 4.f, 400.f);
            const float BH = 28.f;
            ImVec2 p0  = ImGui::GetCursorScreenPos();
            ImU32  fill = dis ? COL_CORE_DIS : sm_running ? COL_L1_ACT  : COL_BLOCK_IDL_FILL;
            ImU32  bord = dis ? COL_BORD_DIS : sm_running ? IM_COL32(240, 195, 15, 220) : COL_BLOCK_IDL_BORD;
            ImU32  tlbl = dis ? IM_COL32(65, 67, 72, 255)
                        : sm_running ? IM_COL32(28, 18, 4, 255) : COL_BLOCK_IDL_TEXT;
            dl->AddRectFilled(p0, { p0.x + bw, p0.y + BH }, fill, 5.f);
            dl->AddRect       (p0, { p0.x + bw, p0.y + BH }, bord, 5.f);
            snprintf(lbl, sizeof(lbl), "%d KB", t.l1_shared_kb);
            ImVec2 ts = ImGui::CalcTextSize(lbl);
            dl->AddText({ p0.x + (bw - ts.x) * 0.5f, p0.y + (BH - ts.y) * 0.5f }, tlbl, lbl);
            ImGui::Dummy({ bw, BH });
        }
    }

    // ── LD/ST Units ───────────────────────────────────────────────────────────
    if (t.ldst_units > 0) {
        ImGui::Spacing();
        snprintf(lbl, sizeof(lbl), "LD/ST Units  (%d)  — load/store, atomics", t.ldst_units);
        draw_section_header(dl, lbl, { 1.f, 0.43f, 0.35f, 1.f }, { 0.32f, 0.16f, 0.12f, 1.f }, sm_running, dis);
        {
            const float UW = 22.f, UH = 18.f, UGAP = 3.f;
            const int   UCOLS = 16;
            ImVec2 ori = ImGui::GetCursorScreenPos();
            int rows = (t.ldst_units + UCOLS - 1) / UCOLS;
            for (int i = 0; i < t.ldst_units; i++) {
                int   col = i % UCOLS, row = i / UCOLS;
                float x0  = ori.x + col * (UW + UGAP);
                float y0  = ori.y + row * (UH + UGAP);
                ImU32 fill = dis ? COL_CORE_DIS : sm_running ? COL_LDST_ACT : COL_BLOCK_IDL_FILL;
                ImU32 bord = dis ? COL_BORD_DIS : sm_running ? IM_COL32(240, 80, 60, 200) : COL_BLOCK_IDL_BORD;
                dl->AddRectFilled({ x0, y0 }, { x0 + UW, y0 + UH }, fill, 2.f);
                dl->AddRect      ({ x0, y0 }, { x0 + UW, y0 + UH }, bord, 2.f);
            }
            ImGui::Dummy({ UCOLS * (UW + UGAP), rows * (UH + UGAP) + 2.f });
        }
    }

    // ── SFU (Special Function Units) ──────────────────────────────────────────
    if (t.sfu_units > 0) {
        ImGui::Spacing();
        snprintf(lbl, sizeof(lbl), "SFU  (%d)  — sin / cos / rcp / rsqrt / exp", t.sfu_units);
        draw_section_header(dl, lbl, { 0.70f, 1.f, 0.70f, 1.f }, { 0.18f, 0.28f, 0.18f, 1.f }, sm_running, dis);
        {
            const float UW = 32.f, UH = 18.f, UGAP = 4.f;
            ImVec2 ori = ImGui::GetCursorScreenPos();
            for (int i = 0; i < t.sfu_units; i++) {
                float x0  = ori.x + i * (UW + UGAP);
                ImU32 fill = dis ? COL_CORE_DIS : sm_running ? COL_SFU_ACT : COL_BLOCK_IDL_FILL;
                ImU32 bord = dis ? COL_BORD_DIS : sm_running ? IM_COL32(130, 230, 130, 200) : COL_BLOCK_IDL_BORD;
                ImU32 tlbl = dis ? IM_COL32(65, 67, 72, 255)
                           : sm_running ? IM_COL32(15, 40, 15, 255) : COL_BLOCK_IDL_TEXT;
                dl->AddRectFilled({ x0, ori.y }, { x0 + UW, ori.y + UH }, fill, 2.f);
                dl->AddRect      ({ x0, ori.y }, { x0 + UW, ori.y + UH }, bord, 2.f);
                char su[6]; snprintf(su, sizeof(su), "S%d", i + 1);
                ImVec2 ts = ImGui::CalcTextSize(su);
                dl->AddText({ x0 + (UW - ts.x) * 0.5f, ori.y + (UH - ts.y) * 0.5f }, tlbl, su);
            }
            ImGui::Dummy({ t.sfu_units * (UW + UGAP), UH + 2.f });
        }
    }

    // ── FP64 Units (double precision ALUs) ────────────────────────────────────
    if (t.fp64_units > 0) {
        ImGui::Spacing();
        snprintf(lbl, sizeof(lbl), "FP64  (%d)  — double precision ALUs", t.fp64_units);
        draw_section_header(dl, lbl, { 0.87f, 0.32f, 0.87f, 1.f }, { 0.30f, 0.10f, 0.30f, 1.f }, sm_running, dis);
        {
            const float UW = 40.f, UH = 22.f, UGAP = 5.f;
            ImVec2 ori = ImGui::GetCursorScreenPos();
            // clamp columns so they don't overflow on large datacenter counts (32)
            const int   UCOLS = (t.fp64_units <= 8) ? t.fp64_units : 16;
            int rows = (t.fp64_units + UCOLS - 1) / UCOLS;
            for (int i = 0; i < t.fp64_units; i++) {
                int   col = i % UCOLS, row = i / UCOLS;
                float x0  = ori.x + col * (UW + UGAP);
                float y0  = ori.y + row * (UH + UGAP);
                ImU32 fill = dis ? COL_CORE_DIS : sm_running ? COL_FP64_ACT : COL_BLOCK_IDL_FILL;
                ImU32 bord = dis ? COL_BORD_DIS : sm_running ? IM_COL32(210, 60, 210, 200) : COL_BLOCK_IDL_BORD;
                ImU32 tlbl = dis ? IM_COL32(65, 67, 72, 255)
                           : sm_running ? IM_COL32(28, 8, 28, 255) : COL_BLOCK_IDL_TEXT;
                dl->AddRectFilled({ x0, y0 }, { x0 + UW, y0 + UH }, fill, 2.f);
                dl->AddRect      ({ x0, y0 }, { x0 + UW, y0 + UH }, bord, 2.f);
                char fu[8]; snprintf(fu, sizeof(fu), "D%d", i + 1);
                ImVec2 ts = ImGui::CalcTextSize(fu);
                dl->AddText({ x0 + (UW - ts.x) * 0.5f, y0 + (UH - ts.y) * 0.5f }, tlbl, fu);
            }
            ImGui::Dummy({ UCOLS * (UW + UGAP), rows * (UH + UGAP) + 2.f });
        }
    }

    // ── Warp Schedulers ───────────────────────────────────────────────────────
    if (t.warp_schedulers > 0) {
        ImGui::Spacing();
        snprintf(lbl, sizeof(lbl), "Warp Schedulers  (%d)  — each issues 1 warp/cycle", t.warp_schedulers);
        draw_section_header(dl, lbl, { 1.f, 0.80f, 0.32f, 1.f }, { 0.34f, 0.26f, 0.06f, 1.f }, sm_running, dis);
        {
            const float WW = 68.f, WH = 36.f, WGAP = 6.f;
            ImVec2 ori = ImGui::GetCursorScreenPos();
            for (int i = 0; i < t.warp_schedulers; i++) {
                float x0  = ori.x + i * (WW + WGAP);
                ImU32 fill = dis ? COL_CORE_DIS : sm_running ? COL_WS_ACT : COL_BLOCK_IDL_FILL;
                ImU32 bord = dis ? COL_BORD_DIS : sm_running ? IM_COL32(245, 180, 55, 200) : COL_BLOCK_IDL_BORD;
                ImU32 tlbl = dis ? IM_COL32(65, 67, 72, 255)
                           : sm_running ? IM_COL32(30, 20, 4, 255) : COL_BLOCK_IDL_TEXT;
                dl->AddRectFilled({ x0, ori.y }, { x0 + WW, ori.y + WH }, fill, 4.f);
                dl->AddRect      ({ x0, ori.y }, { x0 + WW, ori.y + WH }, bord, 4.f);
                char ws[8]; snprintf(ws, sizeof(ws), "WS%d", i + 1);
                ImVec2 ts = ImGui::CalcTextSize(ws);
                dl->AddText({ x0 + (WW - ts.x) * 0.5f, ori.y + (WH - ts.y) * 0.5f }, tlbl, ws);
            }
            ImGui::Dummy({ t.warp_schedulers * (WW + WGAP), WH + 2.f });
        }
    }

    // ── Register File ─────────────────────────────────────────────────────────
    if (t.reg_file_kb > 0) {
        ImGui::Spacing();
        snprintf(lbl, sizeof(lbl), "Register File  (%d KB = 65536 × 32-bit regs)", t.reg_file_kb);
        draw_section_header(dl, lbl, { 0.75f, 0.92f, 1.f, 1.f }, { 0.22f, 0.28f, 0.34f, 1.f }, sm_running, dis);
        {
            float bw = std::min(ImGui::GetContentRegionAvail().x - 4.f, 400.f);
            const float BH = 22.f;
            ImVec2 p0  = ImGui::GetCursorScreenPos();
            ImU32  fill = dis ? COL_CORE_DIS : sm_running ? IM_COL32(140, 210, 255, 255) : COL_BLOCK_IDL_FILL;
            ImU32  bord = dis ? COL_BORD_DIS : sm_running ? IM_COL32(100, 195, 255, 200) : COL_BLOCK_IDL_BORD;
            ImU32  tlbl = dis ? IM_COL32(65, 67, 72, 255)
                        : sm_running ? IM_COL32(8, 18, 28, 255) : COL_BLOCK_IDL_TEXT;
            dl->AddRectFilled(p0, { p0.x + bw, p0.y + BH }, fill, 4.f);
            dl->AddRect       (p0, { p0.x + bw, p0.y + BH }, bord, 4.f);
            snprintf(lbl, sizeof(lbl), "%d KB", t.reg_file_kb);
            ImVec2 ts = ImGui::CalcTextSize(lbl);
            dl->AddText({ p0.x + (bw - ts.x) * 0.5f, p0.y + (BH - ts.y) * 0.5f }, tlbl, lbl);
            ImGui::Dummy({ bw, BH });
        }
    }

    ImGui::EndChild();
}

// ── Breadcrumb bar ────────────────────────────────────────────────────────────

static void draw_breadcrumb(GpuArchWindowState& s, const char* hw_name)
{
    if (s.level == ArchLevel::GPU) {
        ImGui::TextUnformatted(hw_name);
        return;
    }

    if (ImGui::SmallButton(hw_name))
        { s.level = ArchLevel::GPU; s.sel_gpc = -1; s.sel_sm = -1; }

    if (s.level >= ArchLevel::GPC) {
        ImGui::SameLine(); ImGui::TextDisabled(">");
        ImGui::SameLine();
        char buf[32]; snprintf(buf, sizeof(buf), "GPC %d", s.sel_gpc + 1);
        if (s.level == ArchLevel::GPC) {
            ImGui::TextUnformatted(buf);
        } else {
            if (ImGui::SmallButton(buf)) { s.level = ArchLevel::GPC; s.sel_sm = -1; }
            ImGui::SameLine(); ImGui::TextDisabled(">");
            ImGui::SameLine();
            snprintf(buf, sizeof(buf), "SM %d", s.sel_sm + 1);
            ImGui::TextUnformatted(buf);
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

void gpu_arch_window_draw(GpuArchWindowState& s,
                          const char* hw_name,
                          int sm_count,
                          bool* p_open)
{
    if (!s.topo_ready) return;

    bool& vis = p_open ? *p_open : s.open;
    ImGui::SetNextWindowSize({ 780.f, 590.f }, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("GPU Architecture", &vis, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End(); return;
    }

    // ESC: navigate up one level
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        ImGui::IsKeyPressed(ImGuiKey_Escape, false))
    {
        if      (s.level == ArchLevel::SM)  { s.level = ArchLevel::GPC; s.sel_sm  = -1; }
        else if (s.level == ArchLevel::GPC) { s.level = ArchLevel::GPU; s.sel_gpc = -1; }
    }

    draw_breadcrumb(s, hw_name);
    ImGui::Separator();
    ImGui::Spacing();

    switch (s.level) {
        case ArchLevel::GPU: draw_level_gpu(s, hw_name, sm_count); break;
        case ArchLevel::GPC: draw_level_gpc(s, sm_count);          break;
        case ArchLevel::SM:  draw_level_sm(s, sm_count);             break;
    }

    ImGui::End();
}
