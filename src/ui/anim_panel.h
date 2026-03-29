#pragma once

struct AnimPanelState {
    float current_time = 0.0f;
    float start_time   = 0.0f;
    float end_time     = 120.0f;
    float fps          = 24.0f;
    bool  playing      = false;
    bool  loop         = true;
    bool  preview_open = false;
};

// Advance animation time by dt_seconds. Call once per frame before drawing.
void anim_tick(AnimPanelState& a, float dt_seconds);

// Draw the Animation ImGui panel.
void anim_panel_draw(AnimPanelState& a, const char* active_scene_path, bool active_scene_is_usd);
