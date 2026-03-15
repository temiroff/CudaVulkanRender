#include "material_panel.h"
#include "../custom_shader.h"
#include "../slang_shader.h"
#include <imgui.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<int> find_mat_indices(int obj_id,
                                          const std::vector<Triangle>& tris)
{
    std::vector<int> result;
    for (const Triangle& t : tris) {
        if (t.obj_id != obj_id) continue;
        if (std::find(result.begin(), result.end(), t.mat_idx) == result.end())
            result.push_back(t.mat_idx);
    }
    return result;
}

static void assign_material(int obj_id, int new_mat_idx,
                             std::vector<Triangle>& all_prims,
                             std::vector<Triangle>& prims)
{
    for (Triangle& t : all_prims) if (t.obj_id == obj_id) t.mat_idx = new_mat_idx;
    for (Triangle& t : prims)     if (t.obj_id == obj_id) t.mat_idx = new_mat_idx;
}

// HDR colour + intensity widget. Returns true on change.
static bool color_intensity_edit(const char* label, float4& factor)
{
    float intensity = std::max({ factor.x, factor.y, factor.z });
    float col[3];
    if (intensity > 0.f) {
        col[0] = factor.x / intensity; col[1] = factor.y / intensity;
        col[2] = factor.z / intensity;
    } else { col[0] = col[1] = col[2] = 1.f; }

    bool changed = false;
    char id[64];

    ImGui::PushItemWidth(120.f);
    snprintf(id, sizeof(id), "##col_%s", label);
    if (ImGui::ColorEdit3(id, col,
            ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR |
            ImGuiColorEditFlags_NoAlpha)) {
        factor.x = col[0] * intensity; factor.y = col[1] * intensity;
        factor.z = col[2] * intensity;
        changed = true;
    }
    ImGui::PopItemWidth();

    ImGui::SameLine();
    snprintf(id, sizeof(id), "Intensity##int_%s", label);
    ImGui::PushItemWidth(-1.f);
    if (ImGui::DragFloat(id, &intensity, 0.05f, 0.f, 100.f, "%.3f")) {
        factor.x = col[0] * intensity; factor.y = col[1] * intensity;
        factor.z = col[2] * intensity;
        changed = true;
    }
    ImGui::PopItemWidth();
    return changed;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tab 1 — Edit Material
// ─────────────────────────────────────────────────────────────────────────────

static bool draw_edit_tab(GpuMaterial& mat)
{
    bool changed = false;

    // Base Color
    ImGui::Text("Base Color");
    ImGui::SameLine(130.f);
    float bc[4] = { mat.base_color.x, mat.base_color.y,
                    mat.base_color.z, mat.base_color.w };
    ImGui::PushItemWidth(-1.f);
    if (ImGui::ColorEdit4("##base_color", bc,
            ImGuiColorEditFlags_Float | ImGuiColorEditFlags_AlphaBar)) {
        mat.base_color = { bc[0], bc[1], bc[2], bc[3] }; changed = true;
    }
    ImGui::PopItemWidth();

    // Metallic
    ImGui::Text("Metallic");
    ImGui::SameLine(130.f);
    ImGui::PushItemWidth(-1.f);
    if (ImGui::SliderFloat("##metallic", &mat.metallic, 0.f, 1.f, "%.3f"))
        changed = true;
    ImGui::PopItemWidth();

    // Roughness
    ImGui::Text("Roughness");
    ImGui::SameLine(130.f);
    ImGui::PushItemWidth(-1.f);
    if (ImGui::SliderFloat("##roughness", &mat.roughness, 0.f, 1.f, "%.3f"))
        changed = true;
    ImGui::PopItemWidth();

    ImGui::Spacing(); ImGui::Separator();

    // Emissive
    ImGui::Text("Emissive");
    if (color_intensity_edit("emissive", mat.emissive_factor)) changed = true;

    ImGui::Spacing(); ImGui::Separator();

    // Presets
    if (ImGui::CollapsingHeader("Presets")) {
        struct Preset { const char* name; float4 color; float m, r; };
        static const Preset P[] = {
            { "Matte Plastic",  {0.80f,0.80f,0.80f,1.f}, 0.f, 0.90f },
            { "Glossy Plastic", {0.80f,0.80f,0.80f,1.f}, 0.f, 0.15f },
            { "Rough Metal",    {0.80f,0.80f,0.80f,1.f}, 1.f, 0.70f },
            { "Polished Metal", {0.90f,0.90f,0.90f,1.f}, 1.f, 0.05f },
            { "Brushed Metal",  {0.80f,0.80f,0.80f,1.f}, 1.f, 0.40f },
            { "Gold",           {1.00f,0.76f,0.33f,1.f}, 1.f, 0.20f },
            { "Copper",         {0.95f,0.64f,0.54f,1.f}, 1.f, 0.35f },
            { "Mirror",         {0.95f,0.95f,0.95f,1.f}, 1.f, 0.00f },
            { "Glass-like",     {1.00f,1.00f,1.00f,1.f}, 0.f, 0.00f },
        };
        for (int i = 0; i < (int)(sizeof(P)/sizeof(P[0])); ++i) {
            if (i % 3 != 0) ImGui::SameLine();
            if (ImGui::Button(P[i].name, ImVec2(113.f, 0.f))) {
                mat.base_color = P[i].color; mat.metallic = P[i].m;
                mat.roughness  = P[i].r; changed = true;
            }
        }
        ImGui::Spacing();
    }

    // Texture info
    if (ImGui::CollapsingHeader("Textures  (read-only)")) {
        auto row = [&](const char* lbl, int idx) {
            ImGui::Text("%-22s", lbl); ImGui::SameLine();
            if (idx >= 0) {
                char buf[32]; snprintf(buf, sizeof(buf), "#%d", idx);
                ImGui::TextColored(ImVec4(0.5f,0.9f,0.5f,1.f), "%s", buf);
            } else { ImGui::TextDisabled("none"); }
        };
        row("Base Color Tex",  mat.base_color_tex);
        row("Metal/Rough Tex", mat.metallic_rough_tex);
        row("Normal Tex",      mat.normal_tex);
        row("Emissive Tex",    mat.emissive_tex);
    }

    ImGui::Spacing();
    if (ImGui::Button("Reset to Default", ImVec2(-1.f, 0.f))) {
        mat.base_color = {0.8f,0.8f,0.8f,1.f}; mat.metallic = 0.f;
        mat.roughness  = 0.9f;
        mat.emissive_factor = {0.f,0.f,0.f,1.f};
        mat.base_color_tex = mat.metallic_rough_tex =
        mat.normal_tex     = mat.emissive_tex = -1;
        changed = true;
    }
    return changed;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tab 2 — Assign Material (library)
// ─────────────────────────────────────────────────────────────────────────────

// Returns mat_idx to assign, or -1. Sets new_created if host_mats grew.
static int draw_assign_tab(int current_mat, std::vector<GpuMaterial>& host_mats,
                            bool& new_created)
{
    int assign_to = -1;
    new_created = false;

    if (ImGui::Button("+ New Material")) {
        GpuMaterial blank{};
        blank.base_color      = {0.8f,0.8f,0.8f,1.f};
        blank.metallic        = 0.f;
        blank.roughness       = 0.9f;
        blank.emissive_factor = {0.f,0.f,0.f,1.f};
        blank.base_color_tex  = -1;
        blank.emissive_tex    = -1;
        // Inherit normal/metallic-roughness maps from current material so the
        // model's soft shading normals are preserved when switching materials.
        if (current_mat >= 0 && current_mat < (int)host_mats.size()) {
            blank.normal_tex        = host_mats[current_mat].normal_tex;
            blank.metallic_rough_tex = host_mats[current_mat].metallic_rough_tex;
        } else {
            blank.normal_tex = blank.metallic_rough_tex = -1;
        }
        host_mats.push_back(blank);
        assign_to = (int)host_mats.size() - 1;
        new_created = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Duplicate Current") &&
        current_mat >= 0 && current_mat < (int)host_mats.size()) {
        host_mats.push_back(host_mats[current_mat]);
        assign_to = (int)host_mats.size() - 1;
        new_created = true;
    }
    ImGui::Separator();

    ImGui::BeginChild("##mat_lib", ImVec2(0.f, 0.f), false);
    for (int i = 0; i < (int)host_mats.size(); ++i) {
        const GpuMaterial& m = host_mats[i];
        bool is_cur = (i == current_mat);

        char sw_id[32]; snprintf(sw_id, sizeof(sw_id), "##sw%d", i);
        ImGui::ColorButton(sw_id, {m.base_color.x, m.base_color.y, m.base_color.z, 1.f},
            ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip,
            ImVec2(16.f, 16.f));
        ImGui::SameLine();

        char lbl[80];
        snprintf(lbl, sizeof(lbl), "#%-3d  M:%.2f  R:%.2f%s",
                 i, m.metallic, m.roughness, is_cur ? "  [assigned]" : "");

        if (is_cur) ImGui::PushStyleColor(ImGuiCol_Text, {1.f,0.85f,0.3f,1.f});
        ImGui::Text("%s", lbl);
        if (is_cur) ImGui::PopStyleColor();

        if (!is_cur) {
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 54.f);
            char btn[32]; snprintf(btn, sizeof(btn), "Assign##a%d", i);
            if (ImGui::SmallButton(btn)) assign_to = i;
        }
    }
    ImGui::EndChild();
    return assign_to;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tab 3 — Custom Shader (NVRTC / Slang-compatible)
// ─────────────────────────────────────────────────────────────────────────────

// Per-material shader code buffer (keyed by mat_idx)
// Per-material shader code — separate for each language
static std::unordered_map<int, std::string>    s_cuda_code;
static std::unordered_map<int, std::string>    s_slang_code;
// Original GpuMaterial snapshot saved at the moment "Compile & Apply" is first pressed.
// Used to restore the BSDF when the user hits "Reset to Original BSDF".
static std::unordered_map<int, GpuMaterial>    s_original_mat;

struct ShaderEditState {
    bool        compiled   = false;
    bool        has_error  = false;
    std::string log;
    int         tex_w      = 256;
    int         tex_h      = 256;
    int         lang       = 0;   // 0 = CUDA, 1 = Slang
};
static std::unordered_map<int, ShaderEditState> s_shader_state;

// Returns a MaterialPanelChange with shader pixels on "Compile & Apply"; else empty fields.
static MaterialPanelChange draw_shader_tab(int mat_idx, GpuMaterial& mat, int frame_count)
{
    if (s_cuda_code.find(mat_idx)  == s_cuda_code.end())  s_cuda_code[mat_idx]  = CUSTOM_SHADER_DEFAULT_CODE;
    if (s_slang_code.find(mat_idx) == s_slang_code.end()) s_slang_code[mat_idx] = SLANG_SHADER_DEFAULT_CODE;
    if (s_shader_state.find(mat_idx) == s_shader_state.end()) s_shader_state[mat_idx] = {};

    auto& state = s_shader_state[mat_idx];
    bool is_slang = (state.lang == 1);
    auto& code  = is_slang ? s_slang_code[mat_idx] : s_cuda_code[mat_idx];

    // ── Language selector ─────────────────────────────────────────────────────
    ImGui::Text("Language");
    ImGui::SameLine(80.f);
    ImGui::SetNextItemWidth(100.f);
    static const char* langs[] = { "CUDA (NVRTC)", "Slang" };
    if (ImGui::BeginCombo("##lang", langs[state.lang])) {
        for (int i = 0; i < 2; ++i)
            if (ImGui::Selectable(langs[i], state.lang == i)) {
                state.lang = i; state.compiled = false; state.has_error = false;
            }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::Text("Texture");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.f);
    static const int sizes[] = {64, 128, 256, 512, 1024};
    static const char* size_names[] = {"64","128","256","512","1024"};
    int cur_w_idx = 2;
    for (int i = 0; i < 5; ++i) if (sizes[i] == state.tex_w) cur_w_idx = i;
    if (ImGui::BeginCombo("##texsz", size_names[cur_w_idx])) {
        for (int i = 0; i < 5; ++i)
            if (ImGui::Selectable(size_names[i], i == cur_w_idx))
                state.tex_w = state.tex_h = sizes[i];
        ImGui::EndCombo();
    }
    ImGui::SameLine(); ImGui::TextDisabled("px");

    ImGui::TextDisabled(is_slang
        ? "Return MatOut from custom_material(uv, pos, normal, frame) — Slang syntax"
        : "Return MatOut from __device__ custom_material(uv, pos, normal, frame) — CUDA C++");
    ImGui::Separator();

    // ── Code editor ──────────────────────────────────────────────────────────
    constexpr int EDITOR_BUF = 8192;
    static char s_edit_buf[EDITOR_BUF] = {};
    static int  s_last_mat  = -1;
    static int  s_last_lang = -1;

    if (s_last_mat != mat_idx || s_last_lang != state.lang) {
        s_last_mat  = mat_idx;
        s_last_lang = state.lang;
        int n = (int)code.size();
        if (n >= EDITOR_BUF) n = EDITOR_BUF - 1;
        memcpy(s_edit_buf, code.c_str(), (size_t)n);
        s_edit_buf[n] = '\0';
    }

    float editor_h = ImGui::GetContentRegionAvail().y - 80.f;
    if (editor_h < 100.f) editor_h = 100.f;

    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.10f, 0.10f, 0.12f, 1.f));
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts.Size > 1
                    ? ImGui::GetIO().Fonts->Fonts[1] : nullptr);
    ImGui::InputTextMultiline("##shader_code", s_edit_buf, EDITOR_BUF,
                               ImVec2(-1.f, editor_h),
                               ImGuiInputTextFlags_AllowTabInput);
    ImGui::PopFont();
    ImGui::PopStyleColor();
    code = s_edit_buf;

    ImGui::Spacing();

    bool has_original = (s_original_mat.find(mat_idx) != s_original_mat.end());

    // ── Compile button + (optional) Reset button ──────────────────────────────
    float btn_w = has_original ? ImGui::GetContentRegionAvail().x * 0.65f : -1.f;
    bool clicked_compile = ImGui::Button("Compile & Apply", ImVec2(btn_w, 0.f));

    bool clicked_reset = false;
    if (has_original) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.55f, 0.20f, 0.18f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.75f, 0.30f, 0.28f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.90f, 0.40f, 0.38f, 1.f));
        clicked_reset = ImGui::Button("Reset to BSDF", ImVec2(-1.f, 0.f));
        ImGui::PopStyleColor(3);
    }

    if (state.has_error) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.4f, 0.3f, 1.f));
        ImGui::TextWrapped("Error: %s", state.log.c_str());
        ImGui::PopStyleColor();
    } else if (state.compiled) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.f, 0.5f, 1.f));
        ImGui::Text("OK — base_color + roughness + metallic + emissive + normal applied.");
        ImGui::PopStyleColor();
        if (!state.log.empty()) ImGui::TextDisabled("Warnings: %s", state.log.c_str());
    } else {
        ImGui::TextDisabled("Press Compile & Apply to run the shader.");
    }
    if (has_original)
        ImGui::TextDisabled("Original BSDF saved — Reset to BSDF restores it.");

    // ── Run compilation ───────────────────────────────────────────────────────
    MaterialPanelChange result;
    if (clicked_compile) {
        // Save original material the very first time a shader is applied
        if (s_original_mat.find(mat_idx) == s_original_mat.end())
            s_original_mat[mat_idx] = mat;

        bool ok = false;
        std::string err;
        if (is_slang) {
            SlangShaderResult sr = slang_shader_run(code, state.tex_w, state.tex_h, frame_count);
            ok  = sr.success;
            err = sr.error_log;
            if (ok) {
                result.shader_base_pixels     = std::move(sr.base_pixels);
                result.shader_mr_pixels       = std::move(sr.mr_pixels);
                result.shader_emissive_pixels = std::move(sr.emissive_pixels);
                result.shader_normal_pixels   = std::move(sr.normal_pixels);
            }
        } else {
            CustomShaderResult cr = custom_shader_run(code, state.tex_w, state.tex_h, frame_count);
            ok  = cr.success;
            err = cr.error_log;
            if (ok) {
                result.shader_base_pixels     = std::move(cr.base_pixels);
                result.shader_mr_pixels       = std::move(cr.mr_pixels);
                result.shader_emissive_pixels = std::move(cr.emissive_pixels);
                result.shader_normal_pixels   = std::move(cr.normal_pixels);
            }
        }
        state.compiled  = ok;
        state.has_error = !ok;
        state.log       = err;
        if (ok) {
            result.shader_w   = state.tex_w;
            result.shader_h   = state.tex_h;
            result.shader_mat = mat_idx;
        }
    }

    // ── Reset to original BSDF ────────────────────────────────────────────────
    if (clicked_reset && has_original) {
        mat = s_original_mat[mat_idx];            // restore PBR properties + textures
        s_original_mat.erase(mat_idx);            // forget snapshot
        state.compiled  = false;
        state.has_error = false;
        state.log.clear();
        result.materials         = true;          // tell main.cpp to re-upload GpuMaterial
        result.shader_reset      = true;
        result.shader_reset_mat  = mat_idx;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

MaterialPanelChange material_panel_draw(
    int                            selected_obj,
    std::vector<Triangle>&         all_prims,
    std::vector<Triangle>&         prims,
    const std::vector<MeshObject>& objects,
    std::vector<GpuMaterial>&      host_mats,
    int                            frame_count)
{
    MaterialPanelChange result;

    ImGui::SetNextWindowSize(ImVec2(400, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin("Material");

    if (selected_obj < 0 || selected_obj >= (int)objects.size() || host_mats.empty()) {
        ImGui::TextDisabled("No object selected.");
        ImGui::TextDisabled("Click an object in the Outliner.");
        ImGui::End();
        return result;
    }

    const MeshObject& obj = objects[selected_obj];
    ImGui::TextColored(ImVec4(1.f,0.85f,0.3f,1.f), "%s", obj.name);
    ImGui::Separator();

    std::vector<int> mat_ids = find_mat_indices(obj.obj_id, all_prims);
    if (mat_ids.empty()) {
        ImGui::TextDisabled("(no triangles for this object)");
        ImGui::End();
        return result;
    }

    // Sub-material selector (only shown when object has multiple materials)
    static int s_edit_slot = 0;
    if (s_edit_slot >= (int)mat_ids.size()) s_edit_slot = 0;
    if (mat_ids.size() > 1) {
        ImGui::Text("Sub-material:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.f);
        char preview[32]; snprintf(preview, sizeof(preview), "#%d", mat_ids[s_edit_slot]);
        if (ImGui::BeginCombo("##submat", preview)) {
            for (int ti = 0; ti < (int)mat_ids.size(); ++ti) {
                char opt[32]; snprintf(opt, sizeof(opt), "#%d", mat_ids[ti]);
                if (ImGui::Selectable(opt, ti == s_edit_slot)) s_edit_slot = ti;
            }
            ImGui::EndCombo();
        }
        ImGui::Separator();
    }

    int mat_idx = mat_ids[s_edit_slot];
    if (mat_idx < 0 || mat_idx >= (int)host_mats.size()) {
        ImGui::TextDisabled("Mat index out of range.");
        ImGui::End();
        return result;
    }

    // ── Tab bar ───────────────────────────────────────────────────────────────
    if (!ImGui::BeginTabBar("##mat_tabs")) { ImGui::End(); return result; }

    // Tab 1: Edit
    if (ImGui::BeginTabItem("Edit")) {
        if (draw_edit_tab(host_mats[mat_idx])) result.materials = true;
        ImGui::EndTabItem();
    }

    // Tab 2: Assign
    if (ImGui::BeginTabItem("Assign")) {
        ImGui::TextDisabled("Assign any scene material to \"%s\".", obj.name);
        ImGui::Spacing();
        bool new_created = false;
        int assign_to = draw_assign_tab(mat_idx, host_mats, new_created);
        if (assign_to >= 0) {
            assign_material(obj.obj_id, assign_to, all_prims, prims);
            result.materials = new_created;
            result.triangles = true;
        }
        ImGui::EndTabItem();
    }

    // Tab 3: Custom Shader
    if (ImGui::BeginTabItem("Custom Shader")) {
        MaterialPanelChange sr = draw_shader_tab(mat_idx, host_mats[mat_idx], frame_count);
        if (!sr.shader_base_pixels.empty()) {
            result.shader_base_pixels     = std::move(sr.shader_base_pixels);
            result.shader_mr_pixels       = std::move(sr.shader_mr_pixels);
            result.shader_emissive_pixels = std::move(sr.shader_emissive_pixels);
            result.shader_normal_pixels   = std::move(sr.shader_normal_pixels);
            result.shader_w   = sr.shader_w;
            result.shader_h   = sr.shader_h;
            result.shader_mat = sr.shader_mat;
        }
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
    ImGui::End();
    return result;
}
