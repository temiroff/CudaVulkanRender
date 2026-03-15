#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D cuda_texture;

void main() {
    out_color = texture(cuda_texture, v_uv);
}
