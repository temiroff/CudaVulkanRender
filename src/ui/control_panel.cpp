#include "control_panel.h"
#include <imgui.h>
#include <cstring>
#include <cmath>
#include <string>
#include <algorithm>

// ?????? Native Windows file picker ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
#include <windows.h>
#include <shobjidl.h>   // IFileOpenDialog

static std::string open_gltf_picker() {
    std::string result;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE) {
        IFileOpenDialog* pfd = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
                                       IID_IFileOpenDialog, reinterpret_cast<void**>(&pfd))))
        {
            COMDLG_FILTERSPEC types[] = {
                { L"3D Scenes (glTF / USD / URDF)", L"*.gltf;*.glb;*.usd;*.usda;*.usdc;*.usdz;*.urdf" },
                { L"glTF / GLB",                    L"*.gltf;*.glb"                                    },
                { L"USD / USDA / USDC",             L"*.usd;*.usda;*.usdc;*.usdz"                      },
                { L"URDF (Robot Description)",      L"*.urdf"                                           },
                { L"All Files",                     L"*.*"                                              }
            };
            pfd->SetFileTypes(5, types);
            pfd->SetFileTypeIndex(1);
            pfd->SetTitle(L"Open 3D Scene");

            if (SUCCEEDED(pfd->Show(nullptr))) {
                IShellItem* psi = nullptr;
                if (SUCCEEDED(pfd->GetResult(&psi))) {
                    PWSTR path = nullptr;
                    if (SUCCEEDED(psi->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path) {
                        int n = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
                        result.resize(n - 1);
                        WideCharToMultiByte(CP_UTF8, 0, path, -1, result.data(), n, nullptr, nullptr);
                        CoTaskMemFree(path);
                    }
                    psi->Release();
                }
            }
            pfd->Release();
        }
        CoUninitialize();
    }
    return result;
}

static const char* mat_names[] = { "Lambertian", "Metal", "Dielectric", "Emissive" };

static const char* viewport_pass_name(int pass)
{
    switch ((ViewportPassMode)pass) {
        case ViewportPassMode::Final:        return "Final";
        case ViewportPassMode::RawRender:    return "Raw Render";
        case ViewportPassMode::Denoised:     return "Denoised";
        case ViewportPassMode::DlssInput:    return "DLSS Input";
        case ViewportPassMode::DlssDepth:    return "Depth";
        case ViewportPassMode::DlssMotion:   return "Motion";
        case ViewportPassMode::Normal:       return "Normal";
        case ViewportPassMode::Albedo:       return "Albedo";
        case ViewportPassMode::Metallic:     return "Metallic";
        case ViewportPassMode::Roughness:    return "Roughness";
        case ViewportPassMode::Emission:     return "Emission";
        case ViewportPassMode::Segmentation: return "Segmentation";
        case ViewportPassMode::Solid:        return "Solid";
        case ViewportPassMode::Rasterized:   return "Rasterized";
        default:                             return "Final";
    }
}

static std::string save_render_picker(bool /*unused*/) {
    std::string result;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE) {
        IFileSaveDialog* pfd = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_ALL,
                                       IID_IFileSaveDialog, reinterpret_cast<void**>(&pfd))))
        {
            COMDLG_FILTERSPEC types[] = {
                { L"OpenEXR Image", L"*.exr" },
                { L"All Files",     L"*.*"   }
            };
            pfd->SetFileTypes(2, types);
            pfd->SetFileTypeIndex(1);
            pfd->SetDefaultExtension(L"exr");
            pfd->SetTitle(L"Save Render as EXR");
            pfd->SetFileName(L"render.exr");
            if (SUCCEEDED(pfd->Show(nullptr))) {
                IShellItem* psi = nullptr;
                if (SUCCEEDED(pfd->GetResult(&psi))) {
                    PWSTR path = nullptr;
                    if (SUCCEEDED(psi->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path) {
                        int n = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
                        result.resize(n - 1);
                        WideCharToMultiByte(CP_UTF8, 0, path, -1, result.data(), n, nullptr, nullptr);
                        CoTaskMemFree(path);
                    }
                    psi->Release();
                }
            }
            pfd->Release();
        }
        CoUninitialize();
    }
    return result;
}

static std::string open_hdri_picker() {
    std::string result;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE) {
        IFileOpenDialog* pfd = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
                                       IID_IFileOpenDialog, reinterpret_cast<void**>(&pfd))))
        {
            COMDLG_FILTERSPEC types[] = {
                { L"HDR/EXR Images", L"*.hdr;*.HDR;*.exr;*.EXR" },
                { L"All Files",      L"*.*"                      }
            };
            pfd->SetFileTypes(2, types);
            pfd->SetFileTypeIndex(1);
            pfd->SetTitle(L"Open HDRI Environment");
            if (SUCCEEDED(pfd->Show(nullptr))) {
                IShellItem* psi = nullptr;
                if (SUCCEEDED(pfd->GetResult(&psi))) {
                    PWSTR path = nullptr;
                    if (SUCCEEDED(psi->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path) {
                        int n = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
                        result.resize(n - 1);
                        WideCharToMultiByte(CP_UTF8, 0, path, -1, result.data(), n, nullptr, nullptr);
                        CoTaskMemFree(path);
                    }
                    psi->Release();
                }
            }
            pfd->Release();
        }
        CoUninitialize();
    }
    return result;
}

bool control_panel_draw(ControlPanelState& s) {
    bool cam_changed = false;

    ImGui::Begin("Controls");

    // ?????? Menu bar ??????????????????????????????????????????????????????????????????????????????????????????
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            ImGui::MenuItem("Apply Tonemapping on Save", nullptr, &s.save_exr_tonemapped);
            ImGui::MenuItem("16-bit Half EXR", nullptr, &s.save_exr_half);
            ImGui::Separator();
            const char* label = s.save_exr_half ? "Save Render as EXR (16-bit)..."
                                                 : "Save Render as EXR (32-bit)...";
            if (ImGui::MenuItem(label, "Ctrl+S")) {
                std::string p = save_render_picker(false);
                if (!p.empty()) {
                    strncpy_s(s.save_exr_path, sizeof(s.save_exr_path), p.c_str(), _TRUNCATE);
                    s.save_exr_requested = true;
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Tools")) {
            ImGui::MenuItem("GPU Architecture Viewer", nullptr, &s.show_gpu_arch);
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // ?????? Interaction mode toggle ??????????????????????????????????????????
    ImGui::Text("Mode:");
    ImGui::SameLine();
    if (ImGui::RadioButton("Select", s.interact_mode == InteractMode::Select))
        s.interact_mode = InteractMode::Select;
    ImGui::SameLine();
    if (ImGui::RadioButton("Move", s.interact_mode == InteractMode::Move))
        s.interact_mode = InteractMode::Move;
    ImGui::SameLine();
    if (ImGui::RadioButton("Orbit", s.interact_mode == InteractMode::Orbit))
        s.interact_mode = InteractMode::Orbit;

    if (s.interact_mode == InteractMode::Move) {
        ImGui::Text("Gizmo:");
        ImGui::SameLine();
        if (ImGui::RadioButton("W Translate", s.gizmo_mode == GizmoMode::Translate))
            s.gizmo_mode = GizmoMode::Translate;
        ImGui::SameLine();
        if (ImGui::RadioButton("E Rotate", s.gizmo_mode == GizmoMode::Rotate))
            s.gizmo_mode = GizmoMode::Rotate;
        ImGui::SameLine();
        if (ImGui::RadioButton("R Scale", s.gizmo_mode == GizmoMode::Scale))
            s.gizmo_mode = GizmoMode::Scale;
    }

    ImGui::Separator();

    // ?????? Camera ??????????????????????????????????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Pos: (%.2f, %.2f, %.2f)", s.pos[0], s.pos[1], s.pos[2]);
        ImGui::Text("Yaw: %.1f  Pitch: %.1f", s.yaw, s.pitch);
        if (s.interact_mode == InteractMode::Orbit) {
            ImGui::Text("Pivot: (%.2f, %.2f, %.2f)  Dist: %.2f",
                s.orbit_pivot[0], s.orbit_pivot[1], s.orbit_pivot[2], s.orbit_dist);
            ImGui::TextDisabled("LMB drag: orbit   Scroll: zoom");
        } else {
            ImGui::TextDisabled("WASD/Arrows: move   RMB+drag: look");
        }
        ImGui::Separator();
        cam_changed |= ImGui::SliderFloat("Move Speed", &s.move_speed, 0.5f, 50.f);
        cam_changed |= ImGui::SliderFloat("Look Sens",  &s.look_sens,  0.01f, 1.f);
        ImGui::Separator();
        cam_changed |= ImGui::SliderFloat("VFov",       &s.vfov,       5.f, 120.f);
        cam_changed |= ImGui::SliderFloat("Aperture",   &s.aperture,   0.f,   5.0f);
        cam_changed |= ImGui::DragFloat("Focus Dist", &s.focus_dist, 0.01f, 0.001f, 100000.f, "%.3f", ImGuiSliderFlags_Logarithmic);
    }

    ImGui::Separator();

    // ?????? Selected object info ??????????????????????????????????????????????????????
    bool any_selected = (s.selected_sphere >= 0 || s.selected_mesh_obj >= 0);
    if (any_selected) {
        if (ImGui::CollapsingHeader("Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (s.selected_sphere >= 0)
                ImGui::Text("Sphere #%d", s.selected_sphere);
            else
                ImGui::Text("Mesh obj #%d", s.selected_mesh_obj);
            ImGui::TextDisabled("Move mode: drag gizmo arrows");
            if (ImGui::Button("Deselect")) {
                s.selected_sphere   = -1;
                s.selected_mesh_obj = -1;
            }
        }
        ImGui::Separator();
    }

    // ?????? Render Settings ??????????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
        cam_changed |= ImGui::SliderInt("SPP / frame", &s.spp,       1, 32);
        cam_changed |= ImGui::SliderInt("Max Depth",   &s.max_depth, 1, 32);
        static const char* color_modes[] = { "Shaders", "Grey", "Random Colors", "USD Colors" };
        cam_changed |= ImGui::Combo("Color Mode", &s.color_mode, color_modes, 4);
        ImGui::Separator();
        cam_changed |= ImGui::SliderFloat("Firefly Clamp", &s.firefly_clamp, 0.f, 100.f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Clamp per-sample luminance to suppress bright speckles.\n0 = disabled. 5-20 recommended with HDRI.");
        ImGui::Text("Accumulated samples: %d  (%d frames ?? %d spp)",
                    s.frame_count * s.spp, s.frame_count, s.spp);
        if (ImGui::Button("Reset Accumulation")) cam_changed = true;
    }

    ImGui::Separator();

    // ?????? Performance Stats ???????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%.2f Mrays/sec", s.mrays_per_sec);
        ImGui::Text("%.2f ms / frame", s.frame_ms);

        s.frame_times[s.ft_offset] = s.frame_ms;
        s.ft_offset = (s.ft_offset + 1) % ControlPanelState::HISTORY;

        char overlay[32];
        snprintf(overlay, sizeof(overlay), "%.1f ms", s.frame_ms);
        ImGui::PlotLines("##ft", s.frame_times, ControlPanelState::HISTORY, s.ft_offset,
                         overlay, 0.f, 50.f, ImVec2(0, 60));
    }

    ImGui::Separator();

    // ?????? glTF Model ?????????????????????????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("Scene (glTF / USD)", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (s.mesh_loaded)
            ImGui::TextColored(ImVec4(0.4f,1.f,0.4f,1.f), "Loaded  %d triangles", s.num_mesh_tris);
        else
            ImGui::TextDisabled("No mesh loaded");

        // Show just the filename, not the full path, so it fits in the panel
        const char* display = s.gltf_path[0] ? s.gltf_path : "(none)";
        // Find last slash to show only filename
        const char* fname = display;
        for (const char* p = display; *p; ++p)
            if (*p == '/' || *p == '\\') fname = p + 1;
        ImGui::TextDisabled("File: %s", fname);

        float btn_w = ImGui::GetContentRegionAvail().x;

        // Browse button — replaces entire scene
        if (ImGui::Button("Browse...", ImVec2(btn_w, 32.f))) {
            std::string picked = open_gltf_picker();
            if (!picked.empty()) {
                strncpy_s(s.gltf_path, sizeof(s.gltf_path), picked.c_str(), _TRUNCATE);
                s.load_gltf_requested = true;
            }
        }

        // Import button — adds to existing scene
        if (ImGui::Button("Import into Scene...", ImVec2(btn_w, 28.f))) {
            std::string picked = open_gltf_picker();
            if (!picked.empty()) {
                strncpy_s(s.gltf_path, sizeof(s.gltf_path), picked.c_str(), _TRUNCATE);
                s.import_requested = true;
            }
        }

        // Reload button (if a file is already picked)
        if (s.gltf_path[0] != '\0') {
            if (ImGui::Button("Reload", ImVec2(btn_w, 0.f)))
                s.load_gltf_requested = true;
        }
    }

    ImGui::Separator();

    // ?????? HDRI Environment ??????????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("HDRI Environment", ImGuiTreeNodeFlags_DefaultOpen)) {
        float btn_w = ImGui::GetContentRegionAvail().x;

        if (s.hdri_loaded) {
            ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "HDRI loaded");
        } else {
            ImGui::TextDisabled("No HDRI ??? using gradient sky");
        }

        // Show filename
        const char* hdri_disp = s.hdri_path[0] ? s.hdri_path : "(none)";
        const char* hdri_fname = hdri_disp;
        for (const char* p = hdri_disp; *p; ++p)
            if (*p == '/' || *p == '\\') hdri_fname = p + 1;
        if (s.hdri_path[0]) ImGui::TextDisabled("%s", hdri_fname);

        if (ImGui::Button("Browse HDRI...", ImVec2(btn_w, 28.f))) {
            std::string picked = open_hdri_picker();
            if (!picked.empty()) {
                strncpy_s(s.hdri_path, sizeof(s.hdri_path), picked.c_str(), _TRUNCATE);
                s.load_hdri_requested = true;
            }
        }
        if (s.hdri_loaded) {
            if (ImGui::Button("Remove HDRI", ImVec2(btn_w, 0.f))) {
                s.hdri_path[0]       = '\0';
                s.hdri_loaded        = false;
                s.load_hdri_requested = true;  // signal main to clear
                cam_changed = true;
            }
            cam_changed |= ImGui::SliderFloat("Intensity##hdri", &s.hdri_intensity, 0.f, 10.f);
            cam_changed |= ImGui::SliderFloat("Rotation##hdri",  &s.hdri_yaw_deg,  -180.f, 180.f);
        }
        ImGui::Separator();
        ImGui::TextDisabled("Background");
        cam_changed |= ImGui::SliderFloat("BG Blur##hdri",    &s.hdri_bg_blur,    0.f, 1.f, "%.2f");
        cam_changed |= ImGui::SliderFloat("BG Opacity##hdri", &s.hdri_bg_opacity, 0.f, 1.f, "%.2f");
        cam_changed |= ImGui::ColorEdit3("BG Color##hdri", s.bg_color, ImGuiColorEditFlags_Float);
    }

    ImGui::Separator();

    // ?????? Denoiser (OIDN) ?????????????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("Denoiser (OIDN)", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (s.denoise_available) {
            ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "OIDN ready (CPU)");
            ImGui::Checkbox("Auto-denoise##dn",   &s.denoise_enabled);
            if (s.denoise_enabled) {
                ImGui::Indent(12.f);
                ImGui::SliderInt("Every N frames##dn", &s.denoise_every_n, 1, 256);
                ImGui::TextDisabled("Frame %d / %d", s.frame_count % s.denoise_every_n, s.denoise_every_n);
                ImGui::Unindent(12.f);
            }
            if (ImGui::Button("Denoise Now")) s.denoise_on_demand = true;
            ImGui::TextDisabled("CPU denoiser ??? fast at convergence");
        } else {
            ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "OIDN not found");
            ImGui::TextDisabled("Install OIDN SDK and rebuild with:");
            ImGui::TextDisabled("-DOIDN_ROOT=<path>");
            ImGui::TextDisabled("Or install Houdini/Blender which ships OIDN");
        }
    }

    ImGui::Separator();

    // ?????? Post Processing ?????????????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("Post Processing", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Post-Processing", &s.post_enabled);
        if (s.post_enabled) {
            static const char* tone_map_items[] = {
                "Standard (clamp)",    // 0  Blender: Standard / Raw
                "Reinhard",            // 1
                "ACES",                // 2  Blender: ACES 1.3 approx
                "AgX",                 // 3  Blender: AgX  (4.0+ default, exact)
                "Filmic",              // 4  Blender: Filmic (pre-4.0 default)
                "Khronos PBR Neutral", // 5  Blender: Khronos PBR Neutral
            };
            int prev_mode = s.post.tone_map_mode;
            ImGui::Combo("Tone Map", &s.post.tone_map_mode, tone_map_items, 6);
            (void)prev_mode;
            static const char* look_items[] = {
                "None",
                "Very Low Contrast",
                "Low Contrast",
                "Medium Low Contrast",
                "Medium Contrast",
                "Medium High Contrast",
                "High Contrast",
                "Very High Contrast",
            };
            ImGui::Combo("Look", &s.post.look, look_items, 8);
            ImGui::SliderFloat("Exposure (EV)", &s.post.exposure, -4.f, 4.f, "%.2f EV");
            ImGui::Separator();
            ImGui::SliderFloat("Bloom Strength",    &s.post.bloom_strength,  0.f, 2.f);
            ImGui::SliderFloat("Bloom Threshold",   &s.post.bloom_threshold, 0.2f, 2.0f);
            ImGui::Separator();
            ImGui::SliderFloat("Vignette",          &s.post.vignette_strength, 0.f, 1.f);
            ImGui::SliderFloat("Vig. Softness",     &s.post.vignette_falloff,  0.5f, 6.f);
            ImGui::Separator();
            ImGui::SliderFloat("Saturation",        &s.post.saturation,        0.f, 2.f);
            ImGui::SliderFloat("Gamma",             &s.post.gamma,             0.2f, 4.f, "%.2f");
        }
    }

    ImGui::Separator();

    // ?????? AI Object Recognition ???????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("AI Object Recognition (NIM VLM)")) {

        // ?????? Mode toggle ??????????????????????????????????????????????????????????????????????????????
        bool cloud_mode = s.nim_cfg.use_https;
        if (ImGui::RadioButton("Cloud API (api.nvidia.com)", cloud_mode)) {
            s.nim_cfg.use_https = true;
            strncpy_s(s.nim_cfg.host, sizeof(s.nim_cfg.host), "integrate.api.nvidia.com", _TRUNCATE);
            s.nim_cfg.port = 443;
            s.nim_connection_tested = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Local Docker", !cloud_mode)) {
            s.nim_cfg.use_https = false;
            strncpy_s(s.nim_cfg.host, sizeof(s.nim_cfg.host), "localhost", _TRUNCATE);
            s.nim_cfg.port = 8000;
            s.nim_connection_tested = false;
        }

        ImGui::Spacing();

        if (cloud_mode) {
            // Cloud: show API key field
            ImGui::SetNextItemWidth(-1.f);
            ImGui::InputText("API Key", s.nim_cfg.api_key, sizeof(s.nim_cfg.api_key),
                             ImGuiInputTextFlags_Password);
            ImGui::TextDisabled("Get a free key at build.nvidia.com");
        } else {
            // Local: show host/port + Docker launch
            ImGui::SetNextItemWidth(140.f);
            ImGui::InputText("Host", s.nim_cfg.host, sizeof(s.nim_cfg.host));
            ImGui::SameLine();
            ImGui::SetNextItemWidth(65.f);
            ImGui::InputInt("Port##nimport", &s.nim_cfg.port);
            s.nim_cfg.port = std::max(1, std::min(65535, s.nim_cfg.port));

            if (ImGui::Button("Launch NIM Docker", ImVec2(-1.f, 0.f))) {
                s.nim_docker_launch_req = true;
                s.nim_docker_error[0] = '\0';  // clear previous error
            }
            ImGui::TextDisabled("Requires Docker Desktop + NGC API Key above");
            if (s.nim_docker_error[0] != '\0')
                ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s", s.nim_docker_error);
        }

        // ── Model dropdown ───────────────────────────────────────────────
        {
            static const char* nim_model_names[] = {
                "Phi-3.5 Vision (Microsoft)",
                "Llama 3.2 90B Vision (Meta)",
                "Nemotron Nano 12B VL (NVIDIA)",
                "Nemotron Nano VL 8B (NVIDIA)",
            };
            static const char* nim_model_ids[] = {
                "microsoft/phi-3.5-vision-instruct",
                "meta/llama-3.2-90b-vision-instruct",
                "nvidia/nemotron-nano-12b-v2-vl",
                "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            };
            static const int nim_model_count = sizeof(nim_model_ids) / sizeof(nim_model_ids[0]);

            // Find current selection (default to 0 if model string doesn't match any)
            int current = 0;
            for (int i = 0; i < nim_model_count; ++i) {
                if (strcmp(s.nim_cfg.model, nim_model_ids[i]) == 0) { current = i; break; }
            }

            ImGui::SetNextItemWidth(-1.f);
            if (ImGui::Combo("Model", &current, nim_model_names, nim_model_count)) {
                strncpy_s(s.nim_cfg.model, sizeof(s.nim_cfg.model),
                          nim_model_ids[current], _TRUNCATE);
            }
        }

        ImGui::Separator();

        // ?????? Connection status + test ??????????????????????????????????????????
        {
            float bw = ImGui::GetContentRegionAvail().x - 10.f;
            if (s.nim_connection_tested) {
                if (s.nim_connection_ok) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f,1.f,0.4f,1.f));
                    ImGui::Text("??? Connected");
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f,0.35f,0.3f,1.f));
                    ImGui::Text("??? Unreachable");
                }
                ImGui::PopStyleColor();
                ImGui::SameLine(bw - 80.f);
            }
            if (!s.nim_busy) {
                if (ImGui::SmallButton("Test Connection"))
                    s.nim_ping_request = true;
            }
        }

        ImGui::Separator();

        // ?????? Auto-recognize ?????????????????????????????????????????????????????????????????????
        ImGui::Checkbox("Auto-recognize", &s.nim_auto_enabled);
        if (s.nim_auto_enabled) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80.f);
            ImGui::SliderInt("##nimframes", &s.nim_auto_frames, 32, 512, "%d frames");
            ImGui::TextDisabled("Fires automatically after scene settles");
        }

        ImGui::Separator();

        // ?????? Manual trigger ????????????????????????????????????????????????????????????????????????
        if (s.nim_busy) {
            static float anim = 0.f;
            anim += ImGui::GetIO().DeltaTime * 2.f;
            int dots = (int)(anim) % 4;
            char label[32];
            snprintf(label, sizeof(label), "Recognizing%.*s", dots, "...");
            ImGui::BeginDisabled();
            ImGui::Button(label, ImVec2(-1.f, 0.f));
            ImGui::EndDisabled();
        } else {
            if (ImGui::Button("Recognize Scene Now", ImVec2(-1.f, 0.f)))
                s.nim_request = true;
        }

        ImGui::Separator();

        // ?????? Results ??????????????????????????????????????????????????????????????????????????????????????????
        if (s.nim_result.success) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.f, 0.6f, 1.f));
            ImGui::TextWrapped("%s", s.nim_result.object_type);
            ImGui::PopStyleColor();
            ImGui::Spacing();
            ImGui::TextDisabled("Category");
            ImGui::SameLine(90.f);
            ImGui::Text("%s", s.nim_result.category);
            if (s.nim_result.description[0] != '\0') {
                ImGui::TextDisabled("Desc    ");
                ImGui::SameLine(90.f);
                ImGui::TextWrapped("%s", s.nim_result.description);
            }
            if (s.nim_result.tags[0] != '\0') {
                ImGui::TextDisabled("Tags    ");
                ImGui::SameLine(90.f);
                ImGui::TextWrapped("%s", s.nim_result.tags);
            }
        } else if (s.nim_result.error_msg[0] != '\0') {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.4f, 0.4f, 1.f));
            ImGui::TextWrapped("Error: %s", s.nim_result.error_msg);
            ImGui::PopStyleColor();
        } else {
            ImGui::TextDisabled("Press 'Recognize Scene Now' or enable Auto-recognize");
        }
    }

    ImGui::Separator();

    // ── Cosmos Transfer (AI-augmented rendering) ─────────────────────
    if (ImGui::CollapsingHeader("Cosmos Transfer (AI Render)")) {

        // Inherit API key from NIM VLM if empty
        if (s.cosmos_cfg.api_key[0] == '\0' && s.nim_cfg.api_key[0] != '\0')
            strncpy_s(s.cosmos_cfg.api_key, sizeof(s.cosmos_cfg.api_key),
                      s.nim_cfg.api_key, _TRUNCATE);

        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputText("API Key##cosmos", s.cosmos_cfg.api_key,
                         sizeof(s.cosmos_cfg.api_key), ImGuiInputTextFlags_Password);

        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputText("Model##cosmos", s.cosmos_cfg.model, sizeof(s.cosmos_cfg.model));

        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputText("Endpoint##cosmos", s.cosmos_cfg.endpoint, sizeof(s.cosmos_cfg.endpoint));
        ImGui::TextDisabled("Host: %s:%d", s.cosmos_cfg.host, s.cosmos_cfg.port);

        ImGui::Separator();
        ImGui::TextDisabled("Prompt");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputTextMultiline("##cosmos_prompt", s.cosmos_cfg.prompt,
                                  sizeof(s.cosmos_cfg.prompt), ImVec2(-1.f, 60.f));

        ImGui::SetNextItemWidth(140.f);
        static const char* cosmos_src[] = { "Path Tracer", "Hydra Preview" };
        ImGui::Combo("Source##cosmos", &s.cosmos_source, cosmos_src, 2);

        ImGui::SetNextItemWidth(120.f);
        ImGui::SliderFloat("Depth Weight", &s.cosmos_cfg.depth_weight, 0.f, 1.f, "%.2f");
        ImGui::SetNextItemWidth(120.f);
        ImGui::SliderFloat("Seg Weight", &s.cosmos_cfg.seg_weight, 0.f, 1.f, "%.2f");
        ImGui::SetNextItemWidth(80.f);
        ImGui::InputInt("Seed##cosmos", &s.cosmos_cfg.seed);

        ImGui::Separator();

        // Send button
        if (s.cosmos_busy) {
            static float cosmos_anim = 0.f;
            cosmos_anim += ImGui::GetIO().DeltaTime * 2.f;
            int dots = (int)(cosmos_anim) % 4;
            char label[64];
            snprintf(label, sizeof(label), "Generating%.*s", dots, "...");
            ImGui::BeginDisabled();
            ImGui::Button(label, ImVec2(-1.f, 0.f));
            ImGui::EndDisabled();
            if (s.cosmos_cfg.status[0] != '\0')
                ImGui::TextDisabled("%s", s.cosmos_cfg.status);
        } else {
            if (ImGui::Button("Send AOVs to Cosmos", ImVec2(-1.f, 0.f)))
                s.cosmos_request = true;
        }

        // Result display
        if (s.cosmos_has_result) {
            ImGui::TextColored(ImVec4(0.3f, 1.f, 0.4f, 1.f),
                               "Result ready (%dx%d)", s.cosmos_result_w, s.cosmos_result_h);
            ImGui::Checkbox("Show Result Overlay", &s.cosmos_show_result);
        }
        if (s.cosmos_error[0] != '\0') {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.4f, 0.4f, 1.f));
            ImGui::TextWrapped("Error: %s", s.cosmos_error);
            ImGui::PopStyleColor();
        }
    }

    ImGui::Separator();

    // ?????? GPU Features ???????????????????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("GPU Features", ImGuiTreeNodeFlags_DefaultOpen)) {

        // ?????? Baseline FPS capture ??????????????????????????????????????????????????????
        float cur_fps = s.frame_ms > 0.f ? 1000.f / s.frame_ms : 0.f;
        if (!s.fps_locked) {
            if (ImGui::Button("Lock Baseline FPS")) {
                s.fps_baseline = cur_fps;
                s.fps_locked   = true;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("%.1f fps now", cur_fps);
        } else {
            float delta = cur_fps - s.fps_baseline;
            ImGui::Text("Baseline: %.1f fps", s.fps_baseline);
            ImGui::SameLine();
            if (delta >= 0.f)
                ImGui::TextColored(ImVec4(0.3f,1.f,0.3f,1.f), "+%.1f fps", delta);
            else
                ImGui::TextColored(ImVec4(1.f,0.4f,0.4f,1.f), "%.1f fps", delta);
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset")) { s.fps_locked = false; s.fps_baseline = 0.f; }
        }

        ImGui::Separator();

        // ?????? ReSTIR DI ????????????????????????????????????????????????????????????????????????????????????
        ImGui::TextColored(ImVec4(0.5f,0.9f,1.f,1.f), "ReSTIR DI");
        ImGui::SameLine();
        ImGui::TextDisabled("(CUDA, always available)");
        cam_changed |= ImGui::Checkbox("Enable ReSTIR##restir", &s.restir_enabled);
        if (s.restir_enabled) {
            ImGui::Indent(12.f);
            static const char* restir_mode_items[] = {
                "RIS Only",
                "Spatial Reuse",
                "Temporal Reuse",
                "Temporal + Spatial",
            };
            cam_changed |= ImGui::Combo("Reuse mode##restir_mode", &s.restir_mode,
                                        restir_mode_items, IM_ARRAYSIZE(restir_mode_items));
            cam_changed |= ImGui::SliderInt("Candidates (M)##rc", &s.restir_candidates, 1, 64);
            const bool mode_uses_spatial =
                s.restir_mode == (int)ReSTIRReuseMode::Spatial ||
                s.restir_mode == (int)ReSTIRReuseMode::TemporalSpatial;
            if (mode_uses_spatial)
                cam_changed |= ImGui::SliderInt("Radius px##rr", &s.restir_radius, 5, 60);
            ImGui::TextDisabled("RIS = direct-light candidate sampling only");
            ImGui::TextDisabled("Temporal = previous-frame reprojection, Spatial = neighbor reuse");
            ImGui::Unindent(12.f);
        }

        ImGui::Separator();

        // ?????? OptiX GPU Denoiser ????????????????????????????????????????????????????????????
        ImGui::Separator();

        ImGui::TextColored(ImVec4(0.4f,1.f,1.f,1.f), "OptiX RT Renderer");
#ifdef OPTIX_ENABLED
        ImGui::SameLine();
        ImGui::TextDisabled("(SDK found)");
        cam_changed |= ImGui::Checkbox("Enable OptiX RT renderer##optix_rt", &s.optix_rt_enabled);
        // Runtime availability status
        if (s.optix_rt_available) {
            if (s.optix_rt_scene_ready) {
                ImGui::TextColored(ImVec4(0.4f,1.f,0.4f,1.f), "  Hardware BVH ready");
            } else {
                ImGui::TextColored(ImVec4(1.f,0.85f,0.2f,1.f), "  Load a glTF mesh to use OptiX RT");
                ImGui::TextDisabled("  (spheres use CUDA path tracer, not OptiX RT)");
            }
        } else {
            ImGui::TextColored(ImVec4(1.f,0.4f,0.4f,1.f), "  Runtime init failed");
            if (s.optix_rt_last_error[0])
                ImGui::TextWrapped("  %s", s.optix_rt_last_error);
        }
        ImGui::TextDisabled("Mesh triangles only - same visual output, faster via hardware RT cores");
        ImGui::TextDisabled("CUDA path tracer used for spheres; ReSTIR stays on CUDA");
#else
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1.f), "(SDK not found)");
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.4f);
        bool _dummy_optix_rt = false;
        ImGui::Checkbox("Enable OptiX RT renderer##optix_rt", &_dummy_optix_rt);
        ImGui::PopStyleVar();
        ImGui::TextDisabled("Download OptiX SDK from developer.nvidia.com");
        ImGui::TextDisabled("Then rebuild with OPTIX_ENABLED=ON");
#endif
        ImGui::TextColored(ImVec4(1.f,0.8f,0.3f,1.f), "OptiX GPU Denoiser");
#ifdef OPTIX_ENABLED
        ImGui::SameLine();
        ImGui::TextDisabled("(SDK found)");
        cam_changed |= ImGui::Checkbox("Enable OptiX denoiser##optix", &s.optix_enabled);
        ImGui::TextDisabled("AI denoiser on GPU ??? overrides OIDN when enabled");
        if (s.optix_enabled) {
            ImGui::Indent(12.f);
            ImGui::SliderInt("Every N frames##optix_dn", &s.denoise_every_n, 1, 128);
            if (s.denoise_every_n == 1)
                ImGui::TextDisabled("Denoising every frame (best quality, RTX recommended)");
            else
                ImGui::TextDisabled("Frame %d / %d", s.frame_count % s.denoise_every_n, s.denoise_every_n);
            if (ImGui::Button("Denoise Now##optix")) s.denoise_on_demand = true;
            ImGui::Unindent(12.f);
        }
#else
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1.f), "(SDK not found)");
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.4f);
        bool _dummy_optix = false;
        ImGui::Checkbox("Enable OptiX denoiser##optix", &_dummy_optix);
        ImGui::PopStyleVar();
        ImGui::TextDisabled("Download OptiX SDK from developer.nvidia.com");
        ImGui::TextDisabled("Then rebuild with OPTIX_ENABLED=ON");
#endif

        ImGui::Separator();

        // ?????? DLSS / Resolution Scale ?????????????????????????????????????????????
#ifdef DLSS_ENABLED
        ImGui::TextColored(ImVec4(0.5f,1.f,0.5f,1.f), "DLSS (NGX)");
        cam_changed |= ImGui::Checkbox("Enable DLSS##dlss", &s.dlss_enabled);
#else
        ImGui::TextColored(ImVec4(0.5f,1.f,0.5f,1.f), "Resolution Scale");
        cam_changed |= ImGui::Checkbox("Enable##rs_sw", &s.dlss_enabled);
#endif
if (s.dlss_enabled) {
            ImGui::Indent(12.f);

            auto dlss_mode_name = [](int q) -> const char* {
                switch (q) {
                    case 0: return "Ultra Performance";
                    case 1: return "Performance";
                    case 3: return "Quality";
                    case 4: return "Ultra Quality";
                    case 5: return "DLAA";
                    default: return "Balanced";
                }
            };

            // Continuous scale slider: 1% ??? 100%
            int pct = (int)std::roundf(s.dlss_scale * 100.f);
            if (ImGui::SliderInt("Render %##dlss_pct", &pct, 1, 100, "%d%%")) {
                s.dlss_scale = pct / 100.f;
                if      (pct <= 39) s.dlss_quality = 0;  // Ultra Performance
                else if (pct <= 58) s.dlss_quality = 1;  // Performance
                else if (pct <= 71) s.dlss_quality = 2;  // Balanced
                else if (pct <= 84) s.dlss_quality = 3;  // Quality
                else if (pct < 100) s.dlss_quality = 4;  // Ultra Quality
                else                s.dlss_quality = 5;  // DLAA
                cam_changed = true;
            }

            ImGui::TextDisabled("~%d x %d px (of viewport)",
                (int)(s.dlss_scale * 1920), (int)(s.dlss_scale * 1080));
            ImGui::Checkbox("Debug overlay##dlss_dbg", &s.dlss_debug);
            if (s.dlss_debug) {
                const char* overlay_items[] = {
                    "Raw Render",
                    "Denoised",
                    "DLSS Input",
                    "DLSS Depth",
                    "DLSS Motion",
                };
                int overlay_idx = std::max(0, s.dlss_overlay_pass - 1);
                if (ImGui::Combo("Overlay pass##dlss_overlay", &overlay_idx,
                                 overlay_items, IM_ARRAYSIZE(overlay_items)))
                    s.dlss_overlay_pass = overlay_idx + 1;
            }
#ifdef DLSS_ENABLED
            if (ImGui::TreeNodeEx("DLSS Debug Panel", ImGuiTreeNodeFlags_DefaultOpen)) {
                const float requested_pct = s.dlss_scale * 100.f;
                const float actual_pct_x = (s.dlss_output_w_actual > 0)
                    ? (100.f * (float)s.dlss_render_w_actual / (float)s.dlss_output_w_actual) : 0.f;
                const float actual_pct_y = (s.dlss_output_h_actual > 0)
                    ? (100.f * (float)s.dlss_render_h_actual / (float)s.dlss_output_h_actual) : 0.f;

                ImGui::Text("Runtime: %s", s.dlss_runtime_active ? "Active" : "Inactive");
                ImGui::Text("Auto mode: %s", dlss_mode_name(s.dlss_quality));
                ImGui::Text("Viewport pass: %s", viewport_pass_name(s.viewport_pass));
                ImGui::Text("Requested scale: %.1f%%", requested_pct);
                if (s.dlss_runtime_active) {
                    ImGui::Text("Input render: %d x %d", s.dlss_render_w_actual, s.dlss_render_h_actual);
                    ImGui::Text("Output display: %d x %d", s.dlss_output_w_actual, s.dlss_output_h_actual);
                    ImGui::Text("Actual scale: %.1f%% x %.1f%%", actual_pct_x, actual_pct_y);
                } else {
                    ImGui::TextDisabled("Input render: not active");
                    ImGui::TextDisabled("Output display: not active");
                }

                if (s.dlss_state_valid) {
                    const double vram_mb = (double)s.dlss_estimated_vram_bytes / (1024.0 * 1024.0);
                    ImGui::Text("Estimated VRAM: %.1f MB", vram_mb);
                } else {
                    ImGui::TextDisabled("Estimated VRAM: unavailable");
                }

                ImGui::SeparatorText("Feature Support");
                if (s.dlss_feature_query_ok) {
                    ImGui::Text("Super Resolution: %s", s.dlss_sr_supported ? "Supported" : "Not supported");
                    ImGui::Text("Ray Reconstruction: %s", s.dlss_rr_supported ? "Supported" : "Not supported");
                    ImGui::Text("Frame Generation: %s", s.dlss_fg_supported ? "Supported" : "Not supported");
                    ImGui::TextDisabled("This app currently integrates Super Resolution only.");
                    ImGui::TextDisabled("Ray Reconstruction and Frame Generation need extra render/present integration.");
                } else {
                    ImGui::TextDisabled("Feature support query unavailable");
                }

                ImGui::TextDisabled("This panel shows live Streamline DLSS state for the current viewport.");
                ImGui::TreePop();
            }
#endif
#ifndef DLSS_ENABLED
            ImGui::TextDisabled("Bilinear upscale. NGX SDK = neural DLSS.");
#endif
            ImGui::Unindent(12.f);
        }
#ifndef DLSS_ENABLED
        ImGui::TextDisabled("For neural DLSS: get NGX SDK from developer.nvidia.com");
#endif
    }

    ImGui::Separator();

    // ?????? Debug ????????????????????????????????????????????????????????????????????????????????????????????????
    if (ImGui::CollapsingHeader("Debug")) {
        ImGui::Checkbox("BVH wireframe", &s.show_bvh_dbg);
    }

    ImGui::End();

    s.camera_dirty = cam_changed;
    return cam_changed;
}

void control_panel_draw_main_menu(ControlPanelState& s) {
    if (ImGui::BeginMenu("File")) {
        ImGui::MenuItem("Apply Tonemapping on Save", nullptr, &s.save_exr_tonemapped);
        ImGui::MenuItem("16-bit Half EXR", nullptr, &s.save_exr_half);
        ImGui::Separator();
        const char* label = s.save_exr_half ? "Save Render as EXR (16-bit)..."
                                             : "Save Render as EXR (32-bit)...";
        if (ImGui::MenuItem(label, "Ctrl+S")) {
            std::string p = save_render_picker(false);
            if (!p.empty()) {
                strncpy_s(s.save_exr_path, sizeof(s.save_exr_path), p.c_str(), _TRUNCATE);
                s.save_exr_requested = true;
            }
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Overlays")) {
        ImGui::MenuItem("Selection outline",      nullptr, &s.overlay_selection);
        ImGui::MenuItem("Move gizmo",             nullptr, &s.overlay_gizmo);
        ImGui::MenuItem("Orbit pivot",            nullptr, &s.overlay_orbit);
        ImGui::MenuItem("Scene recognition card", nullptr, &s.overlay_nim);
        ImGui::MenuItem("Viewport lights",        nullptr, &s.overlay_lights);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Tools")) {
        ImGui::MenuItem("GPU Architecture Viewer", nullptr, &s.show_gpu_arch);
        ImGui::EndMenu();
    }
}

