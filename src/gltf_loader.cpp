// ─────────────────────────────────────────────
//  gltf_loader.cpp
//  Loads .gltf / .glb files and converts them into flat GPU-ready data.
//  Depends on tinygltf (header-only, single-compilation-unit pattern).
// ─────────────────────────────────────────────

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include "gltf_loader.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>

// ─────────────────────────────────────────────
//  Local 4×4 column-major matrix (CPU only)
// ─────────────────────────────────────────────

struct Mat4 {
    float m[16]; // column-major: m[col*4 + row]

    static Mat4 identity()
    {
        Mat4 r{};
        r.m[0]  = 1.f; r.m[5]  = 1.f;
        r.m[10] = 1.f; r.m[15] = 1.f;
        return r;
    }

    // Column-major matrix multiply: this * rhs
    Mat4 operator*(const Mat4& rhs) const
    {
        Mat4 res{};
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0.f;
                for (int k = 0; k < 4; ++k) {
                    // this[k][row] * rhs[col][k]
                    sum += m[k * 4 + row] * rhs.m[col * 4 + k];
                }
                res.m[col * 4 + row] = sum;
            }
        }
        return res;
    }

    // Transform a point (w=1)
    float3 transform_point(float3 p) const
    {
        float x = m[0]*p.x + m[4]*p.y + m[8]*p.z  + m[12];
        float y = m[1]*p.x + m[5]*p.y + m[9]*p.z  + m[13];
        float z = m[2]*p.x + m[6]*p.y + m[10]*p.z + m[14];
        return make_float3(x, y, z);
    }

    // Transform a direction / normal using the upper 3×3 only (no translation).
    // Note: for correct normal transformation one should use the
    // inverse-transpose, but per the specification the upper-3×3 is used
    // here for simplicity (suitable for uniform-scale scenes).
    float3 transform_normal(float3 n) const
    {
        float x = m[0]*n.x + m[4]*n.y + m[8]*n.z;
        float y = m[1]*n.x + m[5]*n.y + m[9]*n.z;
        float z = m[2]*n.x + m[6]*n.y + m[10]*n.z;
        return make_float3(x, y, z);
    }
};

// ─────────────────────────────────────────────
//  Build a Mat4 from a glTF node's TRS or matrix field
// ─────────────────────────────────────────────

static Mat4 node_local_transform(const tinygltf::Node& node)
{
    // Case 1: explicit 4×4 matrix (16 doubles, column-major)
    if (node.matrix.size() == 16) {
        Mat4 m;
        for (int i = 0; i < 16; ++i) {
            m.m[i] = static_cast<float>(node.matrix[i]);
        }
        return m;
    }

    // Case 2: decomposed TRS
    Mat4 T = Mat4::identity();
    Mat4 R = Mat4::identity();
    Mat4 S = Mat4::identity();

    // Translation
    if (node.translation.size() == 3) {
        T.m[12] = static_cast<float>(node.translation[0]);
        T.m[13] = static_cast<float>(node.translation[1]);
        T.m[14] = static_cast<float>(node.translation[2]);
    }

    // Rotation — quaternion (x, y, z, w) to column-major rotation matrix
    if (node.rotation.size() == 4) {
        float qx = static_cast<float>(node.rotation[0]);
        float qy = static_cast<float>(node.rotation[1]);
        float qz = static_cast<float>(node.rotation[2]);
        float qw = static_cast<float>(node.rotation[3]);

        float x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
        float xx = qx * x2, yy = qy * y2, zz = qz * z2;
        float xy = qx * y2, xz = qx * z2, yz = qy * z2;
        float wx = qw * x2, wy = qw * y2, wz = qw * z2;

        // Column 0
        R.m[0] = 1.f - (yy + zz);
        R.m[1] = xy + wz;
        R.m[2] = xz - wy;
        R.m[3] = 0.f;
        // Column 1
        R.m[4] = xy - wz;
        R.m[5] = 1.f - (xx + zz);
        R.m[6] = yz + wx;
        R.m[7] = 0.f;
        // Column 2
        R.m[8]  = xz + wy;
        R.m[9]  = yz - wx;
        R.m[10] = 1.f - (xx + yy);
        R.m[11] = 0.f;
        // Column 3 (no translation in rotation matrix)
        R.m[12] = R.m[13] = R.m[14] = 0.f;
        R.m[15] = 1.f;
    }

    // Scale
    if (node.scale.size() == 3) {
        S.m[0]  = static_cast<float>(node.scale[0]);
        S.m[5]  = static_cast<float>(node.scale[1]);
        S.m[10] = static_cast<float>(node.scale[2]);
    }

    // Local = T * R * S  (column-major, applied right-to-left)
    return T * R * S;
}

// ─────────────────────────────────────────────
//  Accessor byte-stride helper
// ─────────────────────────────────────────────

// Returns the real byte stride for an accessor (uses bufferView stride when
// set, otherwise the tightly-packed element size via acc.ByteStride()).
static int accessor_stride(const tinygltf::Accessor& acc,
                            const tinygltf::BufferView& bv)
{
    return acc.ByteStride(bv); // tinygltf already handles the fallback
}

// ─────────────────────────────────────────────
//  Recursive node traversal
// ─────────────────────────────────────────────

static void traverse_node(
    const tinygltf::Model&    model,
    int                       node_idx,
    const Mat4&               parent_world,
    int                       default_mat_idx,
    int&                      next_obj_id,
    std::vector<Triangle>&    triangles_out,
    std::vector<MeshObject>&  objects_out)
{
    const tinygltf::Node& node = model.nodes[node_idx];

    // Compute this node's world transform
    Mat4 local = node_local_transform(node);
    Mat4 world = parent_world * local;

    // Process mesh (if any)
    if (node.mesh >= 0) {
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];

        for (const tinygltf::Primitive& prim : mesh.primitives) {
            // Only handle triangle primitives
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }

            // ── Locate required attributes ─────────────────────────────

            auto find_attr = [&](const char* name) -> int {
                auto it = prim.attributes.find(name);
                return (it != prim.attributes.end()) ? it->second : -1;
            };

            int pos_acc_idx    = find_attr("POSITION");
            int norm_acc_idx   = find_attr("NORMAL");
            int uv_acc_idx     = find_attr("TEXCOORD_0");
            int tan_acc_idx    = find_attr("TANGENT");

            if (pos_acc_idx < 0) {
                // Cannot render without positions
                continue;
            }

            // ── Positions ─────────────────────────────────────────────

            const tinygltf::Accessor&   pos_acc = model.accessors[pos_acc_idx];
            const tinygltf::BufferView& pos_bv  = model.bufferViews[pos_acc.bufferView];
            const tinygltf::Buffer&     pos_buf = model.buffers[pos_bv.buffer];

            const uint8_t* pos_base = pos_buf.data.data()
                                    + pos_bv.byteOffset
                                    + pos_acc.byteOffset;
            int pos_stride = accessor_stride(pos_acc, pos_bv);

            auto get_pos = [&](size_t i) -> float3 {
                const float* f = reinterpret_cast<const float*>(pos_base + i * pos_stride);
                return make_float3(f[0], f[1], f[2]);
            };

            // ── Normals (optional) ────────────────────────────────────

            const uint8_t* norm_base   = nullptr;
            int            norm_stride = 0;

            if (norm_acc_idx >= 0) {
                const tinygltf::Accessor&   na = model.accessors[norm_acc_idx];
                const tinygltf::BufferView& nb = model.bufferViews[na.bufferView];
                norm_base   = model.buffers[nb.buffer].data.data()
                            + nb.byteOffset + na.byteOffset;
                norm_stride = accessor_stride(na, nb);
            }

            auto get_norm = [&](size_t i) -> float3 {
                if (!norm_base) return make_float3(0.f, 0.f, 0.f); // placeholder
                const float* f = reinterpret_cast<const float*>(norm_base + i * norm_stride);
                return make_float3(f[0], f[1], f[2]);
            };

            // ── UVs (optional) ────────────────────────────────────────

            const uint8_t* uv_base   = nullptr;
            int            uv_stride = 0;

            if (uv_acc_idx >= 0) {
                const tinygltf::Accessor&   ua = model.accessors[uv_acc_idx];
                const tinygltf::BufferView& ub = model.bufferViews[ua.bufferView];
                uv_base   = model.buffers[ub.buffer].data.data()
                          + ub.byteOffset + ua.byteOffset;
                uv_stride = accessor_stride(ua, ub);
            }

            auto get_uv = [&](size_t i) -> float2 {
                if (!uv_base) return make_float2(0.f, 0.f);
                const float* f = reinterpret_cast<const float*>(uv_base + i * uv_stride);
                return make_float2(f[0], f[1]);
            };

            // ── Tangents (optional, VEC4: xyz=direction w=sign) ──────

            const uint8_t* tan_base   = nullptr;
            int            tan_stride = 0;

            if (tan_acc_idx >= 0) {
                const tinygltf::Accessor&   ta = model.accessors[tan_acc_idx];
                const tinygltf::BufferView& tb = model.bufferViews[ta.bufferView];
                tan_base   = model.buffers[tb.buffer].data.data()
                           + tb.byteOffset + ta.byteOffset;
                tan_stride = accessor_stride(ta, tb);
            }

            // Returns float4(xyz=local tangent direction, w=bitangent sign).
            // Returns zero-w when no tangent data is present.
            auto get_tan = [&](size_t i) -> float4 {
                if (!tan_base) return make_float4(0.f, 0.f, 0.f, 0.f);
                const float* f = reinterpret_cast<const float*>(tan_base + i * tan_stride);
                return make_float4(f[0], f[1], f[2], f[3]);
            };

            // ── Indices ───────────────────────────────────────────────

            // Collect all vertex indices into a flat list of uint32_t
            std::vector<uint32_t> indices;

            if (prim.indices >= 0) {
                const tinygltf::Accessor&   idx_acc = model.accessors[prim.indices];
                const tinygltf::BufferView& idx_bv  = model.bufferViews[idx_acc.bufferView];
                const uint8_t* idx_base = model.buffers[idx_bv.buffer].data.data()
                                        + idx_bv.byteOffset + idx_acc.byteOffset;
                int idx_stride = accessor_stride(idx_acc, idx_bv);

                indices.resize(idx_acc.count);
                for (size_t i = 0; i < idx_acc.count; ++i) {
                    const uint8_t* p = idx_base + i * idx_stride;
                    switch (idx_acc.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            indices[i] = static_cast<uint32_t>(*p);
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            indices[i] = static_cast<uint32_t>(
                                *reinterpret_cast<const uint16_t*>(p));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                        default:
                            indices[i] = *reinterpret_cast<const uint32_t*>(p);
                            break;
                    }
                }
            } else {
                // Non-indexed: one index per vertex
                indices.resize(pos_acc.count);
                for (size_t i = 0; i < pos_acc.count; ++i) {
                    indices[i] = static_cast<uint32_t>(i);
                }
            }

            // ── Material index ────────────────────────────────────────

            int mat_idx = (prim.material >= 0) ? prim.material : default_mat_idx;

            // ── Assign object id for this primitive ───────────────────

            int this_obj_id = next_obj_id++;

            // ── Build triangles ───────────────────────────────────────

            const size_t tri_count = indices.size() / 3;
            triangles_out.reserve(triangles_out.size() + tri_count);

            // Track AABB for centroid
            float3 aabb_mn = make_float3( 1e30f,  1e30f,  1e30f);
            float3 aabb_mx = make_float3(-1e30f, -1e30f, -1e30f);

            for (size_t t = 0; t < tri_count; ++t) {
                uint32_t i0 = indices[t * 3 + 0];
                uint32_t i1 = indices[t * 3 + 1];
                uint32_t i2 = indices[t * 3 + 2];

                // Local-space vertices
                float3 lv0 = get_pos(i0);
                float3 lv1 = get_pos(i1);
                float3 lv2 = get_pos(i2);

                // World-space vertices
                float3 wv0 = world.transform_point(lv0);
                float3 wv1 = world.transform_point(lv1);
                float3 wv2 = world.transform_point(lv2);

                // Normals
                float3 wn0, wn1, wn2;
                if (norm_base) {
                    wn0 = normalize(world.transform_normal(get_norm(i0)));
                    wn1 = normalize(world.transform_normal(get_norm(i1)));
                    wn2 = normalize(world.transform_normal(get_norm(i2)));
                } else {
                    // Flat normal from world-space edge vectors
                    float3 flat = normalize(cross(wv1 - wv0, wv2 - wv0));
                    wn0 = wn1 = wn2 = flat;
                }

                // Tangents — transform xyz by world rotation, preserve w (sign)
                auto world_tan = [&](size_t vi) -> float4 {
                    float4 lt = get_tan(vi);
                    if (lt.w == 0.f) return make_float4(0.f, 0.f, 0.f, 0.f);
                    float3 wt = world.transform_normal(make_float3(lt.x, lt.y, lt.z));
                    float  wlen = sqrtf(wt.x*wt.x + wt.y*wt.y + wt.z*wt.z);
                    if (wlen > 1e-7f) { wt.x /= wlen; wt.y /= wlen; wt.z /= wlen; }
                    return make_float4(wt.x, wt.y, wt.z, lt.w);
                };

                Triangle tri;
                tri.v0 = wv0; tri.v1 = wv1; tri.v2 = wv2;
                tri.n0 = wn0; tri.n1 = wn1; tri.n2 = wn2;
                tri.uv0 = get_uv(i0);
                tri.uv1 = get_uv(i1);
                tri.uv2 = get_uv(i2);
                tri.t0 = world_tan(i0);
                tri.t1 = world_tan(i1);
                tri.t2 = world_tan(i2);
                tri.mat_idx = mat_idx;
                tri.obj_id  = this_obj_id;

                // Expand AABB
                auto expand = [](float3& mn, float3& mx, float3 p) {
                    if (p.x < mn.x) mn.x = p.x; if (p.x > mx.x) mx.x = p.x;
                    if (p.y < mn.y) mn.y = p.y; if (p.y > mx.y) mx.y = p.y;
                    if (p.z < mn.z) mn.z = p.z; if (p.z > mx.z) mx.z = p.z;
                };
                expand(aabb_mn, aabb_mx, wv0);
                expand(aabb_mn, aabb_mx, wv1);
                expand(aabb_mn, aabb_mx, wv2);

                triangles_out.push_back(tri);
            }

            // Register MeshObject for this primitive
            if (tri_count > 0) {
                MeshObject obj{};
                obj.obj_id   = this_obj_id;
                obj.centroid = make_float3(
                    (aabb_mn.x + aabb_mx.x) * 0.5f,
                    (aabb_mn.y + aabb_mx.y) * 0.5f,
                    (aabb_mn.z + aabb_mx.z) * 0.5f);
                snprintf(obj.name, sizeof(obj.name), "%s[%d]",
                         node.name.empty() ? "mesh" : node.name.c_str(),
                         this_obj_id);
                objects_out.push_back(obj);
            }
        }
    }

    // Recurse into children
    for (int child_idx : node.children) {
        traverse_node(model, child_idx, world, default_mat_idx,
                      next_obj_id, triangles_out, objects_out);
    }
}

// ─────────────────────────────────────────────
//  Public entry point
// ─────────────────────────────────────────────

bool gltf_load(const std::string&          path,
               std::vector<Triangle>&      triangles_out,
               std::vector<GpuMaterial>&   materials_out,
               std::vector<TextureImage>&  textures_out,
               std::vector<MeshObject>&    objects_out)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    // Detect binary (.glb) vs. ASCII (.gltf) by extension
    bool is_glb = (path.size() >= 4 &&
                   path.compare(path.size() - 4, 4, ".glb") == 0);

    bool ok = is_glb
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!warn.empty()) {
        std::cerr << "[gltf_loader] warning: " << warn << '\n';
    }
    if (!ok) {
        std::cerr << "[gltf_loader] error loading '" << path << "': " << err << '\n';
        return false;
    }

    // ── 1. Images ────────────────────────────────────────────────────────

    textures_out.reserve(model.images.size());

    for (const tinygltf::Image& img : model.images) {
        TextureImage ti;
        ti.width  = img.width;
        ti.height = img.height;

        if (img.component == 4) {
            // Already RGBA
            ti.pixels = img.image;
        } else if (img.component == 3) {
            // RGB → RGBA (add alpha = 255)
            const size_t npx = (size_t)img.width * (size_t)img.height;
            ti.pixels.resize(npx * 4);
            const uint8_t* src = img.image.data();
            uint8_t*       dst = ti.pixels.data();
            for (size_t i = 0; i < npx; ++i) {
                dst[i * 4 + 0] = src[i * 3 + 0];
                dst[i * 4 + 1] = src[i * 3 + 1];
                dst[i * 4 + 2] = src[i * 3 + 2];
                dst[i * 4 + 3] = 255u;
            }
        } else {
            // Unsupported channel count — store a 1×1 white placeholder
            ti.width  = 1;
            ti.height = 1;
            ti.pixels = { 255u, 255u, 255u, 255u };
        }

        textures_out.push_back(std::move(ti));
    }

    // ── 1b. Tag sRGB images ───────────────────────────────────────────────
    // glTF spec: baseColorTexture and emissiveTexture are sRGB-encoded.
    // metallicRoughness, normal, occlusion are linear.
    // We pre-tag here so the uploader can linearize on upload.
    for (const tinygltf::Material& mat : model.materials) {
        auto mark_srgb = [&](int tex_idx) {
            if (tex_idx < 0 || tex_idx >= (int)model.textures.size()) return;
            int img_idx = model.textures[tex_idx].source;
            if (img_idx >= 0 && img_idx < (int)textures_out.size())
                textures_out[img_idx].srgb = true;
        };
        mark_srgb(mat.pbrMetallicRoughness.baseColorTexture.index);
        mark_srgb(mat.emissiveTexture.index);
    }

    // ── 2. Materials ─────────────────────────────────────────────────────

    // Helper: glTF texture index → image index (or -1)
    auto tex_to_img = [&](int tex_idx) -> int {
        if (tex_idx < 0) return -1;
        return model.textures[tex_idx].source;
    };

    materials_out.reserve(model.materials.size() + 1);

    for (const tinygltf::Material& mat : model.materials) {
        GpuMaterial gm{};

        const tinygltf::PbrMetallicRoughness& pbr = mat.pbrMetallicRoughness;

        // Base color factor (RGBA)
        gm.base_color = {
            (pbr.baseColorFactor.size() >= 1) ? (float)pbr.baseColorFactor[0] : 1.f,
            (pbr.baseColorFactor.size() >= 2) ? (float)pbr.baseColorFactor[1] : 1.f,
            (pbr.baseColorFactor.size() >= 3) ? (float)pbr.baseColorFactor[2] : 1.f,
            (pbr.baseColorFactor.size() >= 4) ? (float)pbr.baseColorFactor[3] : 1.f
        };

        gm.metallic  = (float)pbr.metallicFactor;
        gm.roughness = (float)pbr.roughnessFactor;

        // Emissive factor (RGB, pad to float4 with w=1)
        gm.emissive_factor = {
            (mat.emissiveFactor.size() >= 1) ? (float)mat.emissiveFactor[0] : 0.f,
            (mat.emissiveFactor.size() >= 2) ? (float)mat.emissiveFactor[1] : 0.f,
            (mat.emissiveFactor.size() >= 3) ? (float)mat.emissiveFactor[2] : 0.f,
            1.f
        };

        // Texture image indices (map texture index → image index)
        gm.base_color_tex      = tex_to_img(pbr.baseColorTexture.index);
        gm.metallic_rough_tex  = tex_to_img(pbr.metallicRoughnessTexture.index);
        gm.normal_tex          = tex_to_img(mat.normalTexture.index);
        gm.emissive_tex        = tex_to_img(mat.emissiveTexture.index);

        materials_out.push_back(gm);
    }

    // ── 3. Default fallback material (grey Lambertian) ────────────────────

    const int default_mat_idx = (int)materials_out.size();
    {
        GpuMaterial def{};
        def.base_color       = { 0.8f, 0.8f, 0.8f, 1.f };
        def.metallic         = 0.f;
        def.roughness        = 0.9f;
        def.emissive_factor  = { 0.f, 0.f, 0.f, 1.f };
        def.base_color_tex   = -1;
        def.metallic_rough_tex = -1;
        def.normal_tex       = -1;
        def.emissive_tex     = -1;
        materials_out.push_back(def);
    }

    // ── 4. Node graph traversal ───────────────────────────────────────────

    const Mat4 identity = Mat4::identity();

    // Use the default scene if available, otherwise fall back to scene 0.
    int scene_idx = model.defaultScene >= 0 ? model.defaultScene : 0;

    int next_obj_id = 0;
    if (scene_idx < (int)model.scenes.size()) {
        const tinygltf::Scene& scene = model.scenes[scene_idx];
        for (int root_idx : scene.nodes) {
            traverse_node(model, root_idx, identity,
                          default_mat_idx, next_obj_id, triangles_out, objects_out);
        }
    }

    // ── 5. Summary ───────────────────────────────────────────────────────

    std::cout << "glTF loaded: "
              << triangles_out.size()  << " triangles, "
              << materials_out.size()  << " materials, "
              << textures_out.size()   << " textures, "
              << objects_out.size()    << " objects\n";

    return true;
}
