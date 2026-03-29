#include "anim_panel.h"
#include <imgui.h>
#include <algorithm>
#include <cstdio>

void anim_tick(AnimPanelState& a, float dt_seconds)
{
    if (!a.playing) return;
    a.current_time += dt_seconds * a.fps;
    if (a.current_time >= a.end_time) {
        if (a.loop) {
            a.current_time = a.start_time;
        } else {
            a.current_time = a.end_time;
            a.playing = false;
        }
    }
}

void anim_panel_draw(AnimPanelState& a, const char* active_scene_path, bool active_scene_is_usd)
{
    if (!ImGui::Begin("Animation")) { ImGui::End(); return; }

    ImGui::SeparatorText("Hydra USD Preview");
    const bool has_scene_path = active_scene_path && active_scene_path[0] != '\0';
    if (has_scene_path) {
        const char* fname = active_scene_path;
        for (const char* p = active_scene_path; *p; ++p)
            if (*p == '/' || *p == '\\') fname = p + 1;
        ImGui::TextDisabled("Scene: %s", fname);
    } else {
        ImGui::TextDisabled("Scene: (none)");
    }

    if (!active_scene_is_usd) {
        ImGui::TextDisabled("Hydra preview requires a USD scene (.usd/.usda/.usdc/.usdz)");
        if (a.preview_open) a.preview_open = false;
        ImGui::BeginDisabled();
    }

    {
        const char* label = a.preview_open ? "Close Preview Panel" : "Open Preview Panel";
        if (ImGui::Button(label))
            a.preview_open = !a.preview_open;
    }

    if (!active_scene_is_usd) {
        ImGui::EndDisabled();
    }

    ImGui::Separator();

    if (ImGui::Button("|<")) {
        a.current_time = a.start_time;
        a.playing      = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("<")) {
        a.playing = false;
        a.current_time = std::max(a.start_time, a.current_time - 1.0f);
    }
    ImGui::SameLine();
    if (ImGui::Button(a.playing ? "Pause" : " Play ")) {
        a.playing = !a.playing;
        if (a.playing && a.current_time >= a.end_time)
            a.current_time = a.start_time;
    }
    ImGui::SameLine();
    if (ImGui::Button(">")) {
        a.playing = false;
        a.current_time = std::min(a.end_time, a.current_time + 1.0f);
    }
    ImGui::SameLine();
    if (ImGui::Button(">|")) {
        a.current_time = a.end_time;
        a.playing      = false;
    }

    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::SliderFloat("##timeline", &a.current_time, a.start_time, a.end_time, "%.1f"))
        a.playing = false;

    char info[64];
    std::snprintf(info, sizeof(info), "Frame %.1f / %.1f  @  %.1f fps",
                  a.current_time, a.end_time, a.fps);
    ImGui::TextDisabled("%s", info);

    ImGui::Separator();
    ImGui::SetNextItemWidth(80.f);
    ImGui::DragFloat("Start##anim", &a.start_time, 1.f, 0.f, a.end_time - 1.f, "%.0f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.f);
    ImGui::DragFloat("End##anim",   &a.end_time,   1.f, a.start_time + 1.f, 100000.f, "%.0f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70.f);
    ImGui::DragFloat("FPS##anim",   &a.fps,        0.5f, 1.f, 120.f, "%.1f");

    ImGui::Checkbox("Loop", &a.loop);

    ImGui::End();
}
