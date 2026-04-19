// ─────────────────────────────────────────────
//  usd_loader.cpp
//  Loads .usd/.usda/.usdc/.usdz and converts to
//  the same flat GPU-ready data as gltf_loader.
// ─────────────────────────────────────────────

#include "usd_loader.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/bakeSkinning.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/metrics.h>    // UsdGeomGetStageUpAxis
#include <pxr/usd/usdGeom/tokens.h>    // UsdGeomTokens->leftHanded/rightHanded
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/types.h>   // UsdShadeAttributeType
#include <pxr/usd/usdShade/tokens.h>  // UsdShadeTokens->materialBind
#include <pxr/usd/usdGeom/subset.h>   // UsdGeomSubset — per-face material bindings
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/vt/array.h>
// stb_image — tinygltf already defines STB_IMAGE_IMPLEMENTATION in gltf_loader.cpp.
// Include only the declarations here; the definitions are shared via the linker.
#include <stb_image.h>

// miniz — direct ZIP extraction for USDZ packages (already compiled via tinyexr deps)
#include "miniz.h"

#include <iostream>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>  // std::swap

PXR_NAMESPACE_USING_DIRECTIVE

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

static float3 gf3(const GfVec3f& v) { return make_float3(v[0], v[1], v[2]); }

static float3 transform_point(const GfMatrix4d& m, const GfVec3f& p)
{
    GfVec3d r = m.Transform(GfVec3d(p[0], p[1], p[2]));
    return make_float3((float)r[0], (float)r[1], (float)r[2]);
}

static float3 transform_normal(const GfMatrix4d& m, const GfVec3f& n)
{
    // normals transform by inverse-transpose upper 3×3
    GfMatrix4d inv = m.GetInverse();
    GfVec3d r(
        inv[0][0]*n[0] + inv[1][0]*n[1] + inv[2][0]*n[2],
        inv[0][1]*n[0] + inv[1][1]*n[1] + inv[2][1]*n[2],
        inv[0][2]*n[0] + inv[1][2]*n[1] + inv[2][2]*n[2]);
    float len = (float)r.GetLength();
    if (len > 1e-7f) r /= len;
    return make_float3((float)r[0], (float)r[1], (float)r[2]);
}

static float3 safe_normalize(float3 v)
{
    float l = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (l < 1e-7f) return make_float3(0.f, 1.f, 0.f);
    return make_float3(v.x/l, v.y/l, v.z/l);
}

static float dot3(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static float3 cross3(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x);
}

// ─────────────────────────────────────────────
//  Texture loading
// ─────────────────────────────────────────────

// Extract a file entry from a USDZ (ZIP) archive into memory.
// sub_path: path inside the archive (e.g. "textures/diffuse.png").
// Falls back to case-insensitive search and basename-only match.
static std::vector<uint8_t> extract_from_usdz(const std::string& zip_path,
                                               const std::string& sub_path)
{
    mz_zip_archive zip{};
    if (!mz_zip_reader_init_file(&zip, zip_path.c_str(), 0))
        return {};

    auto lower = [](std::string s) {
        for (auto& c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        return s;
    };
    auto normalize = [](std::string s) {
        for (auto& c : s) if (c == '\\') c = '/';
        return s;
    };

    const std::string sub_norm  = normalize(sub_path);
    const std::string sub_lower = lower(sub_norm);
    // basename for last-resort match
    std::string sub_base = sub_lower;
    auto slash = sub_base.rfind('/');
    if (slash != std::string::npos) sub_base = sub_base.substr(slash + 1);

    int idx = mz_zip_reader_locate_file(&zip, sub_norm.c_str(), nullptr, 0);

    if (idx < 0) {
        int n = (int)mz_zip_reader_get_num_files(&zip);
        for (int i = 0; i < n; ++i) {
            mz_zip_archive_file_stat stat{};
            if (!mz_zip_reader_file_stat(&zip, i, &stat)) continue;
            std::string fname = lower(normalize(stat.m_filename));
            if (fname == sub_lower) { idx = i; break; }
            // Match on trailing basename
            auto sl = fname.rfind('/');
            if (sl != std::string::npos && fname.substr(sl + 1) == sub_base)
                idx = i;  // keep looking for a better match
        }
    }

    if (idx < 0) { mz_zip_reader_end(&zip); return {}; }

    size_t sz = 0;
    void* data = mz_zip_reader_extract_to_heap(&zip, (mz_uint)idx, &sz, 0);
    mz_zip_reader_end(&zip);
    if (!data) return {};

    std::vector<uint8_t> result(static_cast<uint8_t*>(data),
                                 static_cast<uint8_t*>(data) + sz);
    mz_free(data);
    return result;
}

// Load a texture embedded inside a USDZ archive.
static int load_texture_usdz(const std::string& zip_path,
                              const std::string& sub_path,
                              std::vector<TextureImage>& textures_out,
                              std::unordered_map<std::string,int>& tex_cache,
                              bool is_srgb)
{
    std::string cache_key = zip_path + "[" + sub_path + "]" + (is_srgb ? "|srgb" : "|lin");
    auto it = tex_cache.find(cache_key);
    if (it != tex_cache.end()) return it->second;

    std::vector<uint8_t> bytes = extract_from_usdz(zip_path, sub_path);
    if (bytes.empty()) {
        std::cerr << "[usd_loader] cannot extract from USDZ '" << zip_path
                  << "': " << sub_path << '\n';
        tex_cache[cache_key] = -1;
        return -1;
    }

    int w, h, ch;
    unsigned char* stbi_data = stbi_load_from_memory(
        bytes.data(), (int)bytes.size(), &w, &h, &ch, 4);
    if (!stbi_data) {
        std::cerr << "[usd_loader] stbi failed for USDZ entry: " << sub_path << '\n';
        tex_cache[cache_key] = -1;
        return -1;
    }

    TextureImage ti;
    ti.width  = w;
    ti.height = h;
    ti.srgb   = is_srgb;
    ti.pixels.assign(stbi_data, stbi_data + (size_t)w * h * 4);
    stbi_image_free(stbi_data);

    int idx = (int)textures_out.size();
    textures_out.push_back(std::move(ti));
    tex_cache[cache_key] = idx;
    return idx;
}

// Load a texture from a plain file path.
static int load_texture_file(const std::string& path,
                              std::vector<TextureImage>& textures_out,
                              std::unordered_map<std::string,int>& tex_cache,
                              bool is_srgb = false)
{
    if (path.empty()) return -1;
    std::string cache_key = path + (is_srgb ? "|srgb" : "|lin");
    auto it = tex_cache.find(cache_key);
    if (it != tex_cache.end()) return it->second;

    int w = 0, h = 0, ch = 0;
    unsigned char* stbi_data = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!stbi_data) {
        std::cerr << "[usd_loader] cannot load texture: " << path << '\n';
        tex_cache[cache_key] = -1;
        return -1;
    }

    TextureImage ti;
    ti.width  = w;
    ti.height = h;
    ti.srgb   = is_srgb;
    ti.pixels.assign(stbi_data, stbi_data + (size_t)w * h * 4);
    stbi_image_free(stbi_data);

    int idx = (int)textures_out.size();
    textures_out.push_back(std::move(ti));
    tex_cache[cache_key] = idx;
    return idx;
}

// Resolve a USD asset path relative to the stage's root layer.
static std::string resolve_asset(const std::string& asset_path,
                                  const std::string& stage_dir)
{
    if (asset_path.empty()) return {};
    namespace fs = std::filesystem;
    // If already absolute, use as-is
    fs::path p(asset_path);
    if (p.is_absolute() && fs::exists(p)) return asset_path;
    // Try relative to stage directory
    fs::path rel = fs::path(stage_dir) / p;
    if (fs::exists(rel)) return rel.string();
    return {};
}

// Extract UsdPreviewSurface inputs into GpuMaterial.
// stage_path is the full file path (not dir) so we can build USDZ bracket paths.
static GpuMaterial read_preview_surface(
    const UsdShadeShader&    shader,
    const std::string&       stage_path,
    std::vector<TextureImage>& textures_out,
    std::unordered_map<std::string,int>& tex_cache)
{
    namespace fs = std::filesystem;
    const std::string stage_dir = fs::path(stage_path).parent_path().string();
    const bool is_usdz = [&]{
        auto ext = fs::path(stage_path).extension().string();
        for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        return ext == ".usdz";
    }();

    GpuMaterial gm{};
    gm.base_color         = make_float4(0.8f, 0.8f, 0.8f, 1.f);
    gm.metallic           = 0.f;
    gm.roughness          = 0.5f;
    gm.emissive_factor    = make_float4(0.f, 0.f, 0.f, 1.f);
    gm.base_color_tex     = -1;
    gm.metallic_rough_tex = -1;
    gm.normal_tex         = -1;
    gm.emissive_tex       = -1;
    gm.normal_y_flip      = 1;  // USD normal maps use DirectX convention (Y-flipped)

    // Helper: resolve a texture shader's file input to a texture index
    auto get_tex_idx = [&](const UsdShadeShader& texShader, bool is_srgb) -> int {
        UsdShadeInput fileInp = texShader.GetInput(TfToken("file"));
        if (!fileInp) return -1;
        SdfAssetPath ap;
        if (!fileInp.Get(&ap)) return -1;

        // For USDZ: textures are embedded in the ZIP; extract directly with miniz.
        if (is_usdz) {
            std::string asset   = ap.GetAssetPath();
            std::string resolved = ap.GetResolvedPath();
            std::cerr << "[usd_loader] USDZ tex  asset='" << asset
                      << "'  resolved='" << resolved << "'\n";
            std::string sub = asset.empty() ? resolved : asset;
            // Case 1: bracket path "anything.usdz[sub/path.png]"
            auto bracket = sub.find('[');
            if (bracket != std::string::npos && sub.back() == ']')
                sub = sub.substr(bracket + 1, sub.size() - bracket - 2);
            else {
                // Case 2: plain relative "./0/tex.png"
                if (sub.size() >= 2 && sub[0] == '.' && (sub[1] == '/' || sub[1] == '\\'))
                    sub = sub.substr(2);
            }
            for (auto& c : sub) if (c == '\\') c = '/';
            std::cerr << "[usd_loader] USDZ sub='" << sub << "'\n";
            if (!sub.empty())
                return load_texture_usdz(stage_path, sub, textures_out, tex_cache, is_srgb);
            return -1;
        }

        // Non-USDZ: use resolved path or relative-to-stage lookup
        std::string resolved = ap.GetResolvedPath();
        if (resolved.empty()) resolved = resolve_asset(ap.GetAssetPath(), stage_dir);
        return load_texture_file(resolved, textures_out, tex_cache, is_srgb);
    };

    // Helper: follow a shader input connection, return the source shader (or invalid)
    auto get_connected_shader = [&](const UsdShadeInput& inp) -> UsdShadeShader {
        UsdShadeConnectableAPI src;
        TfToken srcName; UsdShadeAttributeType srcType;
        if (UsdShadeConnectableAPI::GetConnectedSource(inp, &src, &srcName, &srcType))
            return UsdShadeShader(src.GetPrim());
        return UsdShadeShader();
    };

    // Helper: read a color3 input — texture connection takes priority,
    // scalar value is read as fallback/tint.  A UsdPreviewSurface input
    // can have BOTH a scalar (e.g. white) AND a texture connection.
    auto read_color3 = [&](const char* name, float4& out, bool is_srgb) {
        UsdShadeInput inp = shader.GetInput(TfToken(name));
        if (!inp) return;
        // Check texture connection first (takes priority)
        UsdShadeShader src = get_connected_shader(inp);
        if (src) {
            int idx = get_tex_idx(src, is_srgb);
            if (idx >= 0) {
                if (strcmp(name, "diffuseColor") == 0 || strcmp(name, "baseColor") == 0)
                    gm.base_color_tex = idx;
                else if (strcmp(name, "emissiveColor") == 0)
                    gm.emissive_tex = idx;
            }
        }
        // Also read scalar value (used as tint multiplier or fallback)
        GfVec3f v;
        if (inp.Get(&v))
            out = make_float4(v[0], v[1], v[2], 1.f);
    };

    auto read_float = [&](const char* name, float& out) {
        UsdShadeInput inp = shader.GetInput(TfToken(name));
        if (!inp) return;
        inp.Get(&out);
    };

    // Helper: read a single-channel float input that may be a texture.
    // Stores the texture index in tex_idx_out if connected.
    // A UsdPreviewSurface input can have BOTH a scalar default AND a texture
    // connection — always check for a connection first, fall back to scalar.
    auto read_float_tex = [&](const char* name, float& scalar_out, int& tex_idx_out) {
        UsdShadeInput inp = shader.GetInput(TfToken(name));
        if (!inp) return;
        // Check texture connection first (takes priority over scalar)
        UsdShadeShader src = get_connected_shader(inp);
        if (src) {
            tex_idx_out = get_tex_idx(src, /*is_srgb=*/false);
        }
        // Also read scalar (used as multiplier or fallback when no texture)
        inp.Get(&scalar_out);
    };

    {
        TfToken shader_id;
        shader.GetShaderId(&shader_id);
        std::cerr << "[usd_loader] UsdPreviewSurface id='" << shader_id << "'\n";
    }

    read_color3("diffuseColor",  gm.base_color,       /*is_srgb=*/true);
    read_color3("baseColor",     gm.base_color,       /*is_srgb=*/true);
    read_color3("emissiveColor", gm.emissive_factor,  /*is_srgb=*/true);

    // Metallic and roughness — linear data textures.
    // USD UsdPreviewSurface stores metallic in R and roughness in R of separate textures
    // (unlike glTF which packs them together). We read them into separate slots and
    // build a packed metallic-roughness texture if both point to the same file.
    int metallic_tex_idx  = -1;
    int roughness_tex_idx = -1;
    read_float_tex("metallic",  gm.metallic,  metallic_tex_idx);
    read_float_tex("roughness", gm.roughness, roughness_tex_idx);

    // When a texture is connected, treat the scalar as a multiplier.
    // If the scalar is still at the default 0.0 (metallic) or wasn't
    // explicitly authored, use 1.0 so the texture value isn't zeroed out.
    if (metallic_tex_idx >= 0 && gm.metallic == 0.f)
        gm.metallic = 1.f;
    if (roughness_tex_idx >= 0 && gm.roughness == 0.f)
        gm.roughness = 1.f;

    // The renderer reads roughness from G and metallic from B (glTF convention).
    // USD often stores them as separate single-channel textures with data in R.
    // Pack into a single combined texture with correct channel layout.
    if (metallic_tex_idx >= 0 && metallic_tex_idx == roughness_tex_idx) {
        // Same texture — already packed (Blender export style)
        gm.metallic_rough_tex = metallic_tex_idx;
    } else if (metallic_tex_idx >= 0 && roughness_tex_idx >= 0) {
        // Two separate textures — pack R channels into G=roughness, B=metallic
        const TextureImage& rough_img = textures_out[roughness_tex_idx];
        const TextureImage& metal_img = textures_out[metallic_tex_idx];
        TextureImage packed;
        packed.width  = rough_img.width;
        packed.height = rough_img.height;
        packed.srgb   = false;
        packed.pixels.resize(packed.width * packed.height * 4);
        for (int i = 0; i < packed.width * packed.height; ++i) {
            uint8_t roughness_val = rough_img.pixels[i * 4];  // R channel
            uint8_t metallic_val  = 0;
            // Sample metallic texture (may differ in size)
            int mx = (i % packed.width)  * metal_img.width  / packed.width;
            int my = (i / packed.width)  * metal_img.height / packed.height;
            int mi = my * metal_img.width + mx;
            if (mi >= 0 && mi < metal_img.width * metal_img.height)
                metallic_val = metal_img.pixels[mi * 4];  // R channel
            packed.pixels[i * 4 + 0] = 0;              // R (unused)
            packed.pixels[i * 4 + 1] = roughness_val;  // G = roughness
            packed.pixels[i * 4 + 2] = metallic_val;   // B = metallic
            packed.pixels[i * 4 + 3] = 255;
        }
        gm.metallic_rough_tex = (int)textures_out.size();
        textures_out.push_back(std::move(packed));
    } else if (roughness_tex_idx >= 0) {
        // Roughness only — remap R→G channel
        const TextureImage& rough_img = textures_out[roughness_tex_idx];
        TextureImage packed;
        packed.width  = rough_img.width;
        packed.height = rough_img.height;
        packed.srgb   = false;
        packed.pixels.resize(packed.width * packed.height * 4);
        for (int i = 0; i < packed.width * packed.height; ++i) {
            packed.pixels[i * 4 + 0] = 0;
            packed.pixels[i * 4 + 1] = rough_img.pixels[i * 4];  // R → G
            packed.pixels[i * 4 + 2] = (uint8_t)(gm.metallic * 255.f);
            packed.pixels[i * 4 + 3] = 255;
        }
        gm.metallic_rough_tex = (int)textures_out.size();
        textures_out.push_back(std::move(packed));
    } else if (metallic_tex_idx >= 0) {
        // Metallic only — remap R→B channel
        const TextureImage& metal_img = textures_out[metallic_tex_idx];
        TextureImage packed;
        packed.width  = metal_img.width;
        packed.height = metal_img.height;
        packed.srgb   = false;
        packed.pixels.resize(packed.width * packed.height * 4);
        for (int i = 0; i < packed.width * packed.height; ++i) {
            packed.pixels[i * 4 + 0] = 0;
            packed.pixels[i * 4 + 1] = (uint8_t)(gm.roughness * 255.f);
            packed.pixels[i * 4 + 2] = metal_img.pixels[i * 4];  // R → B
            packed.pixels[i * 4 + 3] = 255;
        }
        gm.metallic_rough_tex = (int)textures_out.size();
        textures_out.push_back(std::move(packed));
    }

    read_float("opacity",   gm.base_color.w);

    // Normal map — linear (NOT sRGB)
    {
        UsdShadeInput normalInp = shader.GetInput(TfToken("normal"));
        if (normalInp) {
            UsdShadeShader ns = get_connected_shader(normalInp);
            if (ns) gm.normal_tex = get_tex_idx(ns, /*is_srgb=*/false);
        }
    }

    return gm;
}

// ── MDL shader support (Omniverse / Kit exports) ─────────────────────────────
//
// MDL shaders in USD have info:id = "mdl" and store the specific material name
// (e.g. "OmniPBR", "OmniGlass") in info:mdl:sourceAsset:subIdentifier.
// Inputs are direct scalar/asset values on the shader prim — no UsdUVTexture
// intermediates like UsdPreviewSurface uses.

static std::string mdl_variant(const UsdShadeShader& shader)
{
    UsdPrim prim = shader.GetPrim();

    // 1. info:mdl:sourceAsset:subIdentifier — "OmniPBR", "OmniGlass", etc.
    //    This is the canonical Omniverse attribute.
    {
        UsdAttribute attr = prim.GetAttribute(TfToken("info:mdl:sourceAsset:subIdentifier"));
        if (attr) {
            TfToken tok;
            if (attr.Get(&tok) && !tok.IsEmpty()) return tok.GetString();
            std::string s;
            if (attr.Get(&s) && !s.empty()) return s;
        }
    }

    // 2. info:mdl:sourceAsset — e.g. @OmniPBR.mdl@ — strip path and extension.
    {
        UsdAttribute attr = prim.GetAttribute(TfToken("info:mdl:sourceAsset"));
        if (attr) {
            SdfAssetPath ap;
            if (attr.Get(&ap)) {
                std::string asset = ap.GetAssetPath();
                auto sl = asset.rfind('/');
                if (sl != std::string::npos) asset = asset.substr(sl + 1);
                auto dot = asset.rfind('.');
                if (dot != std::string::npos) asset = asset.substr(0, dot);
                if (!asset.empty()) return asset;
            }
        }
    }

    // 3. info:id fallback (some exporters embed the name here directly)
    TfToken id; shader.GetShaderId(&id);
    return id.GetString();
}

static bool is_mdl_shader(const UsdShadeShader& shader)
{
    // Check info:id first (fast path)
    TfToken id; shader.GetShaderId(&id);
    const std::string& s = id.GetString();
    if (s == "mdl")                              return true;
    if (s.find("OmniPBR")   != std::string::npos) return true;
    if (s.find("OmniGlass")  != std::string::npos) return true;
    if (s.find("Omni")       != std::string::npos) return true;

    // Omniverse Kit exports often leave info:id empty and use
    // info:mdl:sourceAsset / info:mdl:sourceAsset:subIdentifier instead.
    UsdPrim prim = shader.GetPrim();
    if (prim.HasAttribute(TfToken("info:mdl:sourceAsset")))            return true;
    if (prim.HasAttribute(TfToken("info:mdl:sourceAsset:subIdentifier"))) return true;

    return false;
}

// Parse an OmniPBR .mdl file and extract texture paths + scalar defaults.
// The MDL format for textures is:   param_name: texture_2d("./path", gamma),
// and for scalars:                  param_name: value,
// We do a simple line-by-line scan — no full MDL parser needed.
static void parse_mdl_file(
    const std::string&           mdl_path,   // absolute path to the .mdl file
    GpuMaterial&                 gm,
    std::vector<TextureImage>&   textures_out,
    std::unordered_map<std::string,int>& tex_cache,
    bool                         is_usdz)
{
    namespace fs = std::filesystem;
    std::ifstream f(mdl_path);
    if (!f.is_open()) return;

    const std::string mdl_dir = fs::path(mdl_path).parent_path().string();

    // Helper: load a texture given a path relative to the .mdl file.
    auto load_tex = [&](const std::string& rel, bool srgb) -> int {
        if (rel.empty()) return -1;
        std::string abs = (fs::path(mdl_dir) / rel).string();
        // normalise separators
        for (auto& c : abs) if (c == '\\') c = '/';
        return load_texture_file(abs, textures_out, tex_cache, srgb);
    };

    // Extract the quoted path from a texture_2d("./path", ...) token.
    auto extract_tex_path = [](const std::string& line) -> std::string {
        auto q1 = line.find('"');
        if (q1 == std::string::npos) return {};
        auto q2 = line.find('"', q1 + 1);
        if (q2 == std::string::npos) return {};
        std::string p = line.substr(q1 + 1, q2 - q1 - 1);
        // strip leading "./"
        if (p.size() >= 2 && p[0] == '.' && (p[1] == '/' || p[1] == '\\'))
            p = p.substr(2);
        return p;
    };

    std::string line;
    while (std::getline(f, line)) {
        // trim leading whitespace
        size_t s = line.find_first_not_of(" \t");
        if (s == std::string::npos) continue;
        line = line.substr(s);

        // diffuse / albedo texture
        if (line.find("diffuse_texture") != std::string::npos ||
            line.find("albedo_texture")  != std::string::npos) {
            if (gm.base_color_tex < 0) {
                std::string p = extract_tex_path(line);
                if (!p.empty()) gm.base_color_tex = load_tex(p, /*srgb=*/true);
            }
        }
        // roughness texture
        else if (line.find("reflectionroughness_texture") != std::string::npos ||
                 line.find("roughness_texture")           != std::string::npos) {
            if (gm.metallic_rough_tex < 0) {
                std::string p = extract_tex_path(line);
                if (!p.empty()) gm.metallic_rough_tex = load_tex(p, false);
            }
        }
        // metallic texture
        else if (line.find("metallic_texture") != std::string::npos &&
                 line.find("influence") == std::string::npos) {
            if (gm.metallic_rough_tex < 0) {
                std::string p = extract_tex_path(line);
                if (!p.empty()) gm.metallic_rough_tex = load_tex(p, false);
            }
        }
        // normal map
        else if (line.find("normalmap_texture") != std::string::npos) {
            if (gm.normal_tex < 0) {
                std::string p = extract_tex_path(line);
                if (!p.empty()) gm.normal_tex = load_tex(p, false);
            }
        }
        // emissive texture
        else if (line.find("emissive_mask_texture") != std::string::npos ||
                 line.find("emissive_texture")      != std::string::npos) {
            if (gm.emissive_tex < 0) {
                std::string p = extract_tex_path(line);
                if (!p.empty()) gm.emissive_tex = load_tex(p, true);
            }
        }
        // scalar: diffuse_color_constant: color(r, g, b)
        else if (line.find("diffuse_color_constant") != std::string::npos) {
            float r = 0.8f, g = 0.8f, b = 0.8f;
            auto p = line.find('(');
            if (p != std::string::npos)
                sscanf(line.c_str() + p + 1, "%f, %f, %f", &r, &g, &b);
            gm.base_color = make_float4(r, g, b, gm.base_color.w);
        }
        // scalar: reflection_roughness_constant
        else if (line.find("reflection_roughness_constant") != std::string::npos &&
                 line.find("influence") == std::string::npos) {
            auto p = line.find(':');
            if (p != std::string::npos) sscanf(line.c_str() + p + 1, "%f", &gm.roughness);
        }
        // scalar: metallic_constant
        else if (line.find("metallic_constant") != std::string::npos &&
                 line.find("texture") == std::string::npos) {
            auto p = line.find(':');
            if (p != std::string::npos) sscanf(line.c_str() + p + 1, "%f", &gm.metallic);
        }
    }
}

// Derive the .mdl file path from the inline variant string.
// ".::materials::FountainGrass_Mat::FountainGrass_Mat" →
//   {stage_dir}/materials/FountainGrass_Mat.mdl
static std::string mdl_file_from_variant(const std::string& variant,
                                          const std::string& stage_dir)
{
    // Variant format: .::module_path::MaterialName::MaterialName
    // Split on "::" and take the second token as the module path.
    // e.g. [".", "materials", "FountainGrass_Mat", "FountainGrass_Mat"]
    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos < variant.size()) {
        size_t end = variant.find("::", pos);
        if (end == std::string::npos) end = variant.size();
        std::string tok = variant.substr(pos, end - pos);
        if (!tok.empty() && tok != ".") parts.push_back(tok);
        pos = (end == variant.size()) ? end : end + 2;
    }
    namespace fs = std::filesystem;
    if (parts.size() < 2) {
        // Plain name with no "::" prefix (e.g. "fallleaves", "TreeBark_07").
        // Convention: look for {stage_dir}/materials/{variant}.mdl
        if (!variant.empty() && variant.find("::") == std::string::npos)
            return (fs::path(stage_dir) / "materials" / (variant + ".mdl")).string();
        return {};
    }
    // parts[0] = module dir component (e.g. "materials")
    // parts[1] = module name (e.g. "FountainGrass_Mat")
    return (fs::path(stage_dir) / parts[0] / (parts[1] + ".mdl")).string();
}

static GpuMaterial read_mdl_shader(
    const UsdShadeShader&         shader,
    const std::string&            stage_path,
    std::vector<TextureImage>&    textures_out,
    std::unordered_map<std::string,int>& tex_cache)
{
    namespace fs = std::filesystem;
    const std::string stage_dir = fs::path(stage_path).parent_path().string();
    const bool is_usdz = [&]{
        std::string ext = fs::path(stage_path).extension().string();
        for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        return ext == ".usdz";
    }();

    GpuMaterial gm{};
    gm.base_color         = make_float4(0.8f, 0.8f, 0.8f, 1.f);
    gm.metallic           = 0.f;
    gm.roughness          = 0.5f;
    gm.emissive_factor    = make_float4(0.f, 0.f, 0.f, 1.f);
    gm.base_color_tex     = -1;
    gm.metallic_rough_tex = -1;
    gm.normal_tex         = -1;
    gm.emissive_tex       = -1;
    gm.normal_y_flip      = 1;  // USD/MDL normal maps use DirectX convention

    const std::string variant = mdl_variant(shader);
    std::cerr << "[usd_loader] MDL variant='" << variant << "'\n";

    // Read an asset-path input directly as a texture.
    // MDL textures are stored as SdfAssetPath inputs, not through UsdUVTexture nodes.
    auto get_mdl_tex = [&](const char* input_name, bool is_srgb) -> int {
        UsdShadeInput inp = shader.GetInput(TfToken(input_name));
        if (!inp) return -1;
        SdfAssetPath ap;
        if (!inp.Get(&ap)) return -1;
        std::string asset    = ap.GetAssetPath();
        std::string resolved = ap.GetResolvedPath();
        if (asset.empty() && resolved.empty()) return -1;

        if (is_usdz) {
            std::string sub = asset.empty() ? resolved : asset;
            auto bracket = sub.find('[');
            if (bracket != std::string::npos && sub.back() == ']')
                sub = sub.substr(bracket + 1, sub.size() - bracket - 2);
            else if (sub.size() >= 2 && sub[0] == '.' && (sub[1] == '/' || sub[1] == '\\'))
                sub = sub.substr(2);
            for (auto& c : sub) if (c == '\\') c = '/';
            return sub.empty() ? -1
                               : load_texture_usdz(stage_path, sub, textures_out, tex_cache, is_srgb);
        }
        if (resolved.empty()) resolved = resolve_asset(asset, stage_dir);
        return load_texture_file(resolved, textures_out, tex_cache, is_srgb);
    };

    auto read_color3 = [&](const char* name, float4& out) {
        UsdShadeInput inp = shader.GetInput(TfToken(name));
        if (!inp) return;
        GfVec3f v;
        if (inp.Get(&v)) out = make_float4(v[0], v[1], v[2], out.w);
    };

    auto read_float = [&](const char* name, float& out) {
        UsdShadeInput inp = shader.GetInput(TfToken(name));
        if (!inp) return;
        inp.Get(&out);
    };

    bool is_glass = variant.find("Glass") != std::string::npos ||
                    variant.find("glass") != std::string::npos;

    if (is_glass) {
        // OmniGlass: map to low-roughness transparent PBR
        // (no dedicated dielectric path in GpuMaterial, so we approximate)
        gm.roughness    = 0.f;
        gm.metallic     = 0.f;
        gm.base_color.w = 0.08f;   // mostly transparent
        read_color3("glass_color",        gm.base_color);
        read_float ("frosting_roughness", gm.roughness);
        // Keep alpha low to approximate transmission
        float depth = 1.f;
        read_float("depth", depth);
        gm.base_color.w = std::max(0.02f, std::min(0.35f, 1.f / std::max(depth, 0.01f)));
    } else {
        // Try OmniPBR names first, then fall back to common aliases used by
        // inline MDL materials and other Omniverse exporters.

        // ── Diffuse / base color ──────────────────────────────────────────────
        static const char* DIFFUSE_NAMES[] = {
            "diffuse_color_constant", "albedo_color", "diffuse_color",
            "base_color", "baseColor", "diffuse", nullptr };
        for (int i = 0; DIFFUSE_NAMES[i]; i++) {
            UsdShadeInput inp = shader.GetInput(TfToken(DIFFUSE_NAMES[i]));
            if (!inp) continue;
            GfVec3f v;
            if (inp.Get(&v)) { gm.base_color = make_float4(v[0], v[1], v[2], 1.f); break; }
        }
        // albedo_brightness — scalar multiplier on the base color (inline MDL convention)
        bool has_albedo_brightness = false;
        {
            UsdShadeInput inp = shader.GetInput(TfToken("albedo_brightness"));
            if (inp) {
                has_albedo_brightness = true;
                float brightness = 1.f;
                if (inp.Get(&brightness) && brightness != 1.f) {
                    gm.base_color.x *= brightness;
                    gm.base_color.y *= brightness;
                    gm.base_color.z *= brightness;
                }
            }
        }

        bool got_scalar = false;

        // ── Roughness ────────────────────────────────────────────────────────
        static const char* ROUGH_NAMES[] = {
            "reflection_roughness_constant", "roughness_constant", "roughness",
            "specular_roughness", "specularRoughness", nullptr };
        for (int i = 0; ROUGH_NAMES[i]; i++) {
            UsdShadeInput inp = shader.GetInput(TfToken(ROUGH_NAMES[i]));
            if (inp && inp.Get(&gm.roughness)) { got_scalar = true; break; }
        }

        // ── Metallic ─────────────────────────────────────────────────────────
        static const char* METAL_NAMES[] = {
            "metallic_constant", "metallic", "metallicAmount",
            "metalness", "reflectivity", nullptr };
        for (int i = 0; METAL_NAMES[i]; i++) {
            UsdShadeInput inp = shader.GetInput(TfToken(METAL_NAMES[i]));
            if (inp && inp.Get(&gm.metallic)) { got_scalar = true; break; }
        }

        // ── Opacity ──────────────────────────────────────────────────────────
        static const char* OPACITY_NAMES[] = {
            "opacity_constant", "opacity", "alpha", nullptr };
        for (int i = 0; OPACITY_NAMES[i]; i++) {
            UsdShadeInput inp = shader.GetInput(TfToken(OPACITY_NAMES[i]));
            if (inp && inp.Get(&gm.base_color.w)) { got_scalar = true; break; }
        }

        // ── Textures ─────────────────────────────────────────────────────────
        // Diffuse / albedo texture
        static const char* DIFF_TEX_NAMES[] = {
            "diffuse_texture", "albedo_texture", "base_color_texture",
            "diffuseColorMap", "albedoMap", "baseColorMap", nullptr };
        for (int i = 0; DIFF_TEX_NAMES[i]; i++) {
            int idx = get_mdl_tex(DIFF_TEX_NAMES[i], /*srgb=*/true);
            if (idx >= 0) { gm.base_color_tex = idx; break; }
        }

        // Roughness texture
        static const char* ROUGH_TEX_NAMES[] = {
            "reflectionroughness_texture", "roughness_texture",
            "roughnessMap", "specularRoughnessMap", nullptr };
        int rough_tex = -1;
        for (int i = 0; ROUGH_TEX_NAMES[i]; i++) {
            rough_tex = get_mdl_tex(ROUGH_TEX_NAMES[i], /*srgb=*/false);
            if (rough_tex >= 0) break;
        }

        // Metallic texture
        static const char* METAL_TEX_NAMES[] = {
            "metallic_texture", "metallicMap", "metalnessMap", nullptr };
        int metal_tex = -1;
        for (int i = 0; METAL_TEX_NAMES[i]; i++) {
            metal_tex = get_mdl_tex(METAL_TEX_NAMES[i], /*srgb=*/false);
            if (metal_tex >= 0) break;
        }

        // Normal map
        static const char* NORM_TEX_NAMES[] = {
            "normalmap_texture", "normal_texture", "normalMap",
            "bumpMap", "bump_texture", nullptr };
        for (int i = 0; NORM_TEX_NAMES[i]; i++) {
            int idx = get_mdl_tex(NORM_TEX_NAMES[i], /*srgb=*/false);
            if (idx >= 0) { gm.normal_tex = idx; break; }
        }

        // Emissive texture
        static const char* EMIS_TEX_NAMES[] = {
            "emissive_color_texture", "emissive_texture",
            "emissiveMap", "emissiveColorMap", nullptr };
        int emis_tex = -1;
        for (int i = 0; EMIS_TEX_NAMES[i]; i++) {
            emis_tex = get_mdl_tex(EMIS_TEX_NAMES[i], /*srgb=*/true);
            if (emis_tex >= 0) { gm.emissive_tex = emis_tex; break; }
        }

        // ORM (Occlusion/Roughness/Metallic) combined texture — used by TreeBark etc.
        {
            bool orm_enabled = false;
            UsdShadeInput ein = shader.GetInput(TfToken("enable_ORM_texture"));
            if (ein) ein.Get(&orm_enabled);
            if (orm_enabled && rough_tex < 0 && metal_tex < 0) {
                static const char* ORM_NAMES[] = {
                    "ORM_texture", "orm_texture", "ORMMap", nullptr };
                for (int i = 0; ORM_NAMES[i]; i++) {
                    int orm_tex = get_mdl_tex(ORM_NAMES[i], /*srgb=*/false);
                    if (orm_tex >= 0) { rough_tex = orm_tex; metal_tex = orm_tex; break; }
                }
            }
        }

        // Pack metallic + roughness into glTF layout (G=roughness, B=metallic).
        // Same logic as read_preview_surface: USD stores them as separate R-channel
        // textures, but the renderer reads G and B.
        if (metal_tex >= 0 && metal_tex == rough_tex) {
            gm.metallic_rough_tex = metal_tex;  // already packed (e.g. ORM)
        } else if (metal_tex >= 0 && rough_tex >= 0) {
            const TextureImage& ri = textures_out[rough_tex];
            const TextureImage& mi = textures_out[metal_tex];
            TextureImage packed;
            packed.width = ri.width; packed.height = ri.height; packed.srgb = false;
            packed.pixels.resize(packed.width * packed.height * 4);
            for (int i = 0; i < packed.width * packed.height; ++i) {
                int mx = (i % packed.width) * mi.width / packed.width;
                int my = (i / packed.width) * mi.height / packed.height;
                int mi_idx = my * mi.width + mx;
                packed.pixels[i*4+0] = 0;
                packed.pixels[i*4+1] = ri.pixels[i*4];  // R → G (roughness)
                packed.pixels[i*4+2] = (mi_idx >= 0 && mi_idx < mi.width*mi.height)
                                       ? mi.pixels[mi_idx*4] : 0;  // R → B (metallic)
                packed.pixels[i*4+3] = 255;
            }
            gm.metallic_rough_tex = (int)textures_out.size();
            textures_out.push_back(std::move(packed));
        } else if (rough_tex >= 0) {
            const TextureImage& ri = textures_out[rough_tex];
            TextureImage packed;
            packed.width = ri.width; packed.height = ri.height; packed.srgb = false;
            packed.pixels.resize(packed.width * packed.height * 4);
            for (int i = 0; i < packed.width * packed.height; ++i) {
                packed.pixels[i*4+0] = 0;
                packed.pixels[i*4+1] = ri.pixels[i*4];  // R → G
                packed.pixels[i*4+2] = (uint8_t)(gm.metallic * 255.f);
                packed.pixels[i*4+3] = 255;
            }
            gm.metallic_rough_tex = (int)textures_out.size();
            textures_out.push_back(std::move(packed));
        } else if (metal_tex >= 0) {
            const TextureImage& mi = textures_out[metal_tex];
            TextureImage packed;
            packed.width = mi.width; packed.height = mi.height; packed.srgb = false;
            packed.pixels.resize(packed.width * packed.height * 4);
            for (int i = 0; i < packed.width * packed.height; ++i) {
                packed.pixels[i*4+0] = 0;
                packed.pixels[i*4+1] = (uint8_t)(gm.roughness * 255.f);
                packed.pixels[i*4+2] = mi.pixels[i*4];  // R → B
                packed.pixels[i*4+3] = 255;
            }
            gm.metallic_rough_tex = (int)textures_out.size();
            textures_out.push_back(std::move(packed));
        }

        // Emissive scalar
        bool enable_emission = false;
        {
            UsdShadeInput ein = shader.GetInput(TfToken("enable_emission"));
            if (ein) ein.Get(&enable_emission);
        }
        if (enable_emission) {
            static const char* EMIS_COLOR_NAMES[] = {
                "emissive_color", "emissiveColor", "emission_color", nullptr };
            for (int i = 0; EMIS_COLOR_NAMES[i]; i++) {
                UsdShadeInput inp = shader.GetInput(TfToken(EMIS_COLOR_NAMES[i]));
                if (!inp) continue;
                GfVec3f v;
                if (inp.Get(&v)) { gm.emissive_factor = make_float4(v[0], v[1], v[2], 1.f); break; }
            }
            float intensity = 1.f;
            read_float("emissive_intensity", intensity);
            gm.emissive_factor.x *= intensity;
            gm.emissive_factor.y *= intensity;
            gm.emissive_factor.z *= intensity;
        }

        // For inline MDL variants (variant starts with ".::"), the .mdl file is
        // the authoritative source for textures — USD inputs only carry scalar
        // overrides.  Always parse the .mdl when any texture slot is still empty.
        {
            std::string mdl_path = mdl_file_from_variant(variant, stage_dir);
            if (!mdl_path.empty() &&
                (gm.base_color_tex < 0 || gm.metallic_rough_tex < 0 || gm.normal_tex < 0)) {
                parse_mdl_file(mdl_path, gm, textures_out, tex_cache, is_usdz);
            }
        }

        bool got_anything = (gm.base_color_tex >= 0 || gm.metallic_rough_tex >= 0
                          || gm.normal_tex >= 0 || gm.emissive_tex >= 0
                          || gm.metallic != 0.f || has_albedo_brightness
                          || got_scalar);
        if (!got_anything) {
            auto all_inputs = shader.GetInputs();
            if (!all_inputs.empty()) {
                std::cerr << "[usd_loader] MDL '" << variant
                          << "' — no recognised inputs found. Available inputs:";
                for (auto& inp : all_inputs)
                    std::cerr << "  " << inp.GetBaseName();
                std::cerr << '\n';
            }
        }
    }

    return gm;
}

// ─────────────────────────────────────────────
//  Main loader
// ─────────────────────────────────────────────

// USD emits a warning when ComputeBoundMaterial() is called on a prim that has
// a material:binding relationship but the MaterialBindingAPI schema is not applied.
// This is a common Omniverse Kit / AECO exporter issue.  When the API is missing
// we read the relationship directly — same result, no warning.
static UsdShadeMaterial get_bound_material(const UsdPrim& prim)
{
    if (prim.HasAPI<UsdShadeMaterialBindingAPI>())
        return UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();

    // Direct relationship fallback — handles prims with the rel but no applied API.
    UsdRelationship rel = prim.GetRelationship(TfToken("material:binding"));
    if (!rel) return UsdShadeMaterial(UsdPrim{});
    SdfPathVector targets;
    rel.GetForwardedTargets(&targets);
    if (targets.empty()) return UsdShadeMaterial(UsdPrim{});
    return UsdShadeMaterial(prim.GetStage()->GetPrimAtPath(targets[0]));
}

bool usd_load(const std::string&          path,
              std::vector<Triangle>&      triangles_out,
              std::vector<GpuMaterial>&   materials_out,
              std::vector<TextureImage>&  textures_out,
              std::vector<MeshObject>&    objects_out)
{
    UsdStageRefPtr stage = UsdStage::Open(path);
    if (!stage) {
        std::cerr << "[usd_loader] failed to open: " << path << '\n';
        return false;
    }

    std::string stage_dir = std::filesystem::path(path).parent_path().string();

    // Build a Z-up → Y-up correction if the stage uses Z-up.
    // USD uses row-vector convention: point * matrix.
    // We POST-multiply so the correction is applied to world-space points.
    // We want: new_X=old_X, new_Y=old_Z, new_Z=-old_Y
    // For row vectors result[j] = sum_i p[i]*M[i][j], so:
    //   M[2][1]=1 (col Y gets row Z)  and  M[1][2]=-1 (col Z gets -row Y)
    TfToken up_axis = UsdGeomGetStageUpAxis(stage);
    GfMatrix4d axis_correction(1.0);   // identity by default (Y-up stages)
    if (up_axis == UsdGeomTokens->z) {
        axis_correction = GfMatrix4d(
            1,  0,  0, 0,   // row X → X
            0,  0, -1, 0,   // row Y → -Z
            0,  1,  0, 0,   // row Z → +Y
            0,  0,  0, 1);
        std::cout << "[usd_loader] Z-up stage — up_axis=" << up_axis << "  applying Z→Y correction\n";
    } else {
        std::cout << "[usd_loader] Y-up stage — up_axis=" << up_axis << "  no correction\n";
    }

    // metersPerUnit: stages authored in cm (Maya default) set this to 0.01.
    // Bake it into the same correction matrix so every point lands in meters.
    double mpu = UsdGeomGetStageMetersPerUnit(stage);
    if (mpu <= 0.0) mpu = 1.0;
    if (mpu != 1.0) {
        GfMatrix4d scale_m(1.0);
        scale_m.SetScale(GfVec3d(mpu, mpu, mpu));
        axis_correction = axis_correction * scale_m;
        std::cout << "[usd_loader] metersPerUnit=" << mpu
                  << " applied (stage units -> meters)\n";
    }

    UsdGeomXformCache xform_cache;
    std::unordered_map<std::string, int> tex_cache;   // path → texture index
    std::unordered_map<std::string, int> mat_cache;   // prim path → material index

    // Default material
    GpuMaterial default_mat{};
    default_mat.base_color        = make_float4(0.8f, 0.8f, 0.8f, 1.f);
    default_mat.metallic          = 0.f;
    default_mat.roughness         = 0.9f;
    default_mat.emissive_factor   = make_float4(0.f, 0.f, 0.f, 1.f);
    default_mat.base_color_tex    = -1;
    default_mat.metallic_rough_tex= -1;
    default_mat.normal_tex        = -1;
    default_mat.emissive_tex      = -1;
    default_mat.normal_y_flip     = 1;
    int default_mat_idx = 0;
    materials_out.push_back(default_mat);

    int next_obj_id = 0;

    // ── Material resolution helper (reused for mesh-level and GeomSubset-level) ──
    auto resolve_material = [&](const UsdShadeMaterial& mat) -> int {
        if (!mat) return default_mat_idx;
        std::string mat_path = mat.GetPrim().GetPath().GetString();
        auto it = mat_cache.find(mat_path);
        if (it != mat_cache.end()) return it->second;

        UsdShadeShader surface_shader;
        // Try all render-context surface outputs (universal + named)
        for (UsdShadeOutput surf_out : {
                mat.GetSurfaceOutput(),
                mat.GetSurfaceOutput(UsdShadeTokens->universalRenderContext) }) {
            if (!surf_out || surface_shader) continue;
            // Prefer newer GetConnectedSources (plural) — works with USD 21.11+
            UsdShadeSourceInfoVector sources =
                UsdShadeConnectableAPI::GetConnectedSources(surf_out);
            if (!sources.empty()) {
                surface_shader = UsdShadeShader(sources[0].source.GetPrim());
            } else {
                // Legacy singular API fallback
                UsdShadeConnectableAPI src; TfToken srcName; UsdShadeAttributeType srcType;
                if (UsdShadeConnectableAPI::GetConnectedSource(surf_out, &src, &srcName, &srcType))
                    surface_shader = UsdShadeShader(src.GetPrim());
            }
        }
        // Last resort: walk descendants for any known shader type
        if (!surface_shader) {
            for (const UsdPrim& child : mat.GetPrim().GetDescendants()) {
                UsdShadeShader sh(child);
                if (!sh) continue;
                TfToken id; sh.GetShaderId(&id);
                if (id == TfToken("UsdPreviewSurface") ||
                    id == TfToken("ND_UsdPreviewSurface_surfaceshader") ||
                    is_mdl_shader(sh)) {
                    surface_shader = sh; break;
                }
            }
        }
        GpuMaterial gm = default_mat;
        if (surface_shader) {
            if (is_mdl_shader(surface_shader))
                gm = read_mdl_shader(surface_shader, path, textures_out, tex_cache);
            else
                gm = read_preview_surface(surface_shader, path, textures_out, tex_cache);
        }
        int idx = (int)materials_out.size();
        materials_out.push_back(gm);
        mat_cache[mat_path] = idx;
        return idx;
    };

    // Traverse with instance proxy expansion so instanced scenes (like Kitchen_set)
    // are visited. UsdTraverseInstanceProxies() opts into proxy prim traversal.
    for (const UsdPrim& prim : stage->Traverse(UsdTraverseInstanceProxies())) {
        if (!prim.IsA<UsdGeomMesh>()) continue;

        UsdGeomMesh geom(prim);

        // ── Orientation: leftHanded meshes need winding reversal ──────
        // USD default is rightHanded; USDZ from Apple/Reality Composer
        // often uses leftHanded.  Detect via attribute or world xform sign.
        TfToken orientation = UsdGeomTokens->rightHanded;
        geom.GetOrientationAttr().Get(&orientation);
        bool flip_winding = (orientation == UsdGeomTokens->leftHanded);

        // ── World transform ───────────────────────────────────────────
        // Post-multiply: world * correction so correction applies AFTER world transform.
        // (USD row-vector convention: point is transformed as  p * matrix)
        GfMatrix4d world = xform_cache.GetLocalToWorldTransform(prim) * axis_correction;

        // A negative-determinant world transform also flips winding
        if (world.GetDeterminant() < 0.0)
            flip_winding = !flip_winding;

        // ── Topology ──────────────────────────────────────────────────
        VtIntArray face_vertex_counts, face_vertex_indices;
        geom.GetFaceVertexCountsAttr().Get(&face_vertex_counts);
        geom.GetFaceVertexIndicesAttr().Get(&face_vertex_indices);

        VtVec3fArray points;
        geom.GetPointsAttr().Get(&points);
        if (points.empty()) continue;

        // ── Normals ───────────────────────────────────────────────────
        // 1. Try the mesh 'normals' attribute
        // 2. Try primvars:normals (some exporters store normals as primvar)
        // 3. Compute smooth vertex normals by averaging face normals
        VtVec3fArray normals;
        TfToken normals_interp;
        geom.GetNormalsAttr().Get(&normals);
        normals_interp = geom.GetNormalsInterpolation();

        if (normals.empty()) {
            UsdGeomPrimvarsAPI nAPI(prim);
            UsdGeomPrimvar nPv = nAPI.GetPrimvar(TfToken("normals"));
            if (nPv && nPv.IsDefined()) {
                nPv.Get(&normals);
                normals_interp = nPv.GetInterpolation();
            }
        }

        // If no normals found, leave empty — flat normals are computed
        // per-triangle in the loop below as fallback.

        // ── UVs: look for st or st0 primvar ──────────────────────────
        UsdGeomPrimvarsAPI pvAPI(prim);
        VtVec2fArray uvs;
        TfToken uv_interp = UsdGeomTokens->faceVarying;
        VtIntArray uv_indices;

        {
            static const char* UV_NAMES[] = { "st", "st0", "UVMap", "map1", nullptr };
            bool found_uv = false;
            for (int ui = 0; UV_NAMES[ui]; ++ui) {
                UsdGeomPrimvar pv = pvAPI.GetPrimvar(TfToken(UV_NAMES[ui]));
                if (pv && pv.IsDefined()) {
                    pv.Get(&uvs);
                    pv.GetIndices(&uv_indices);
                    uv_interp = pv.GetInterpolation();
                    std::cerr << "[usd_loader] UV primvar='" << UV_NAMES[ui]
                              << "'  count=" << uvs.size()
                              << "  interp=" << uv_interp << '\n';
                    found_uv = true;
                    break;
                }
            }
            if (!found_uv) {
                // List all available primvars so we know what names to add
                std::cerr << "[usd_loader] NO UV primvar found on '"
                          << prim.GetPath() << "'. Available:";
                for (auto& pv : pvAPI.GetPrimvars())
                    std::cerr << " " << pv.GetName();
                std::cerr << '\n';
            }
        }

        // ── Material ──────────────────────────────────────────────────
        // Mesh-level fallback binding
        int mat_idx = default_mat_idx;
        {
            UsdShadeMaterial mat = get_bound_material(prim);
            if (mat) {
                mat_idx = resolve_material(mat);
                std::cerr << "[usd_loader] mesh '" << prim.GetPath()
                          << "' -> mat_idx=" << mat_idx << '\n';
            } else {
                std::cerr << "[usd_loader] mesh '" << prim.GetPath()
                          << "' -> no mesh-level binding\n";
            }
        }

        // ── Per-face materials from GeomSubset children ───────────────
        // Many exporters (Reality Composer, Blender, Maya) bind materials at
        // the face-group level via UsdGeomSubset prims with familyName=materialBind.
        std::unordered_map<int, int> face_mat_map;  // face_idx → mat_idx
        for (const UsdPrim& child : prim.GetChildren()) {
            if (!child.IsA<UsdGeomSubset>()) continue;
            UsdGeomSubset subset(child);
            TfToken family;
            subset.GetFamilyNameAttr().Get(&family);
            if (family != UsdShadeTokens->materialBind) continue;

            VtIntArray face_indices;
            subset.GetIndicesAttr().Get(&face_indices);

            UsdShadeMaterial sub_mat = get_bound_material(child);
            int sub_mat_idx = sub_mat ? resolve_material(sub_mat) : mat_idx;
            for (int fi : face_indices)
                face_mat_map[fi] = sub_mat_idx;
        }

        // ── Build triangles from polygons ─────────────────────────────
        int obj_id = next_obj_id++;
        float3 aabb_mn = make_float3( 1e30f,  1e30f,  1e30f);
        float3 aabb_mx = make_float3(-1e30f, -1e30f, -1e30f);

        int fv_cursor = 0;   // index into face_vertex_indices
        int face_idx  = 0;

        for (int fvc : face_vertex_counts) {
            // Fan-triangulate each polygon
            for (int t = 1; t < fvc - 1; ++t) {
                int i0 = face_vertex_indices[fv_cursor];
                int i1 = face_vertex_indices[fv_cursor + t];
                int i2 = face_vertex_indices[fv_cursor + t + 1];

                if (i0 >= (int)points.size() || i1 >= (int)points.size() || i2 >= (int)points.size())
                    continue;

                float3 wv0 = transform_point(world, points[i0]);
                float3 wv1 = transform_point(world, points[i1]);
                float3 wv2 = transform_point(world, points[i2]);

                // Normals
                auto get_norm = [&](int face_v_idx, int vert_idx) -> float3 {
                    if (normals.empty()) return make_float3(0.f, 0.f, 0.f);
                    int ni = -1;
                    if (normals_interp == UsdGeomTokens->faceVarying)
                        ni = face_v_idx;
                    else if (normals_interp == UsdGeomTokens->vertex)
                        ni = vert_idx;
                    else if (normals_interp == UsdGeomTokens->uniform)
                        ni = face_idx;
                    if (ni < 0 || ni >= (int)normals.size()) return make_float3(0.f,0.f,0.f);
                    return transform_normal(world, normals[ni]);
                };

                float3 wn0 = get_norm(fv_cursor,       i0);
                float3 wn1 = get_norm(fv_cursor + t,   i1);
                float3 wn2 = get_norm(fv_cursor+t+1,   i2);

                // Fall back to flat normal if none
                float3 flat = safe_normalize(cross3(
                    make_float3(wv1.x-wv0.x, wv1.y-wv0.y, wv1.z-wv0.z),
                    make_float3(wv2.x-wv0.x, wv2.y-wv0.y, wv2.z-wv0.z)));
                if (wn0.x == 0.f && wn0.y == 0.f && wn0.z == 0.f) wn0 = flat;
                if (wn1.x == 0.f && wn1.y == 0.f && wn1.z == 0.f) wn1 = flat;
                if (wn2.x == 0.f && wn2.y == 0.f && wn2.z == 0.f) wn2 = flat;

                // UVs
                auto get_uv = [&](int face_v_idx, int vert_idx) -> float2 {
                    if (uvs.empty()) return make_float2(0.f, 0.f);
                    int ui = -1;
                    if (!uv_indices.empty()) {
                        // Indexed primvar
                        if (uv_interp == UsdGeomTokens->faceVarying && face_v_idx < (int)uv_indices.size())
                            ui = uv_indices[face_v_idx];
                        else if (uv_interp == UsdGeomTokens->vertex && vert_idx < (int)uv_indices.size())
                            ui = uv_indices[vert_idx];
                    } else {
                        if (uv_interp == UsdGeomTokens->faceVarying)
                            ui = face_v_idx;
                        else if (uv_interp == UsdGeomTokens->vertex)
                            ui = vert_idx;
                    }
                    if (ui < 0 || ui >= (int)uvs.size()) return make_float2(0.f, 0.f);
                    // USD uses bottom-left UV origin; flip V to match Vulkan/glTF top-left
                    return make_float2(uvs[ui][0], 1.f - uvs[ui][1]);
                };

                // Flip winding for leftHanded orientation or negative-det xform
                if (flip_winding) {
                    std::swap(wv1, wv2);
                    std::swap(wn1, wn2);
                }

                Triangle tri{};
                tri.v0 = wv0; tri.v1 = wv1; tri.v2 = wv2;
                tri.n0 = safe_normalize(wn0);
                tri.n1 = safe_normalize(wn1);
                tri.n2 = safe_normalize(wn2);

                float2 tuv0 = get_uv(fv_cursor,     i0);
                float2 tuv1 = get_uv(fv_cursor + t, i1);
                float2 tuv2 = get_uv(fv_cursor+t+1, i2);
                if (flip_winding) std::swap(tuv1, tuv2);
                tri.uv0 = tuv0; tri.uv1 = tuv1; tri.uv2 = tuv2;

                // Compute tangent from UV edges (MikkTSpace-style)
                // Required for normal mapping — shader skips when tangent.w == 0
                {
                    float3 e1 = make_float3(wv1.x-wv0.x, wv1.y-wv0.y, wv1.z-wv0.z);
                    float3 e2 = make_float3(wv2.x-wv0.x, wv2.y-wv0.y, wv2.z-wv0.z);
                    float du1 = tuv1.x - tuv0.x, dv1 = tuv1.y - tuv0.y;
                    float du2 = tuv2.x - tuv0.x, dv2 = tuv2.y - tuv0.y;
                    float det = du1*dv2 - du2*dv1;
                    float3 T;
                    if (fabsf(det) > 1e-8f) {
                        float inv = 1.f / det;
                        T = safe_normalize(make_float3(
                            (dv2*e1.x - dv1*e2.x) * inv,
                            (dv2*e1.y - dv1*e2.y) * inv,
                            (dv2*e1.z - dv1*e2.z) * inv));
                    } else {
                        // Degenerate UV — pick an arbitrary tangent perpendicular to normal
                        float3 n = tri.n0;
                        float3 up = (fabsf(n.z) < 0.999f)
                            ? make_float3(0.f,0.f,1.f) : make_float3(1.f,0.f,0.f);
                        T = safe_normalize(cross3(up, n));
                    }
                    // Bitangent sign: +1 if (T × N) agrees with B, else -1
                    float3 B = cross3(tri.n0, T);
                    float3 e2cross = cross3(e1, e2);
                    float bsign = (dot3(B, make_float3(
                        (du2*e1.x - du1*e2.x), (du2*e1.y - du1*e2.y),
                        (du2*e1.z - du1*e2.z))) >= 0.f) ? 1.f : -1.f;
                    float4 tang = make_float4(T.x, T.y, T.z, bsign);
                    tri.t0 = tang; tri.t1 = tang; tri.t2 = tang;
                }
                // Per-face material from GeomSubset overrides mesh-level mat
                auto fm = face_mat_map.find(face_idx);
                tri.mat_idx = (fm != face_mat_map.end()) ? fm->second : mat_idx;
                tri.obj_id  = obj_id;

                // AABB
                for (float3 v : { wv0, wv1, wv2 }) {
                    if (v.x < aabb_mn.x) aabb_mn.x = v.x; if (v.x > aabb_mx.x) aabb_mx.x = v.x;
                    if (v.y < aabb_mn.y) aabb_mn.y = v.y; if (v.y > aabb_mx.y) aabb_mx.y = v.y;
                    if (v.z < aabb_mn.z) aabb_mn.z = v.z; if (v.z > aabb_mx.z) aabb_mx.z = v.z;
                }

                triangles_out.push_back(tri);
            }

            fv_cursor += fvc;
            ++face_idx;
        }

        // Register MeshObject
        if (aabb_mn.x < 1e29f) {
            MeshObject obj{};
            obj.obj_id   = obj_id;
            obj.centroid = make_float3(
                (aabb_mn.x + aabb_mx.x) * 0.5f,
                (aabb_mn.y + aabb_mx.y) * 0.5f,
                (aabb_mn.z + aabb_mx.z) * 0.5f);
            snprintf(obj.name, sizeof(obj.name), "%s", prim.GetPath().GetString().c_str());
            objects_out.push_back(obj);
        }
    }

    std::cout << "[usd_loader] loaded: "
              << triangles_out.size() << " triangles, "
              << materials_out.size() << " materials, "
              << textures_out.size()  << " textures, "
              << objects_out.size()   << " objects  from " << path << '\n';

    return !triangles_out.empty();
}

// ─────────────────────────────────────────────
//  USD Animation support
// ─────────────────────────────────────────────

// Per-mesh cached topology (time-invariant data)
struct MeshCache {
    SdfPath              prim_path;
    VtIntArray           face_vertex_counts;
    VtIntArray           face_vertex_indices;
    TfToken              orientation;
    TfToken              subdiv_scheme;
    VtVec2fArray         uvs;
    VtIntArray           uv_indices;
    TfToken              uv_interp;
    int                  mat_idx;
    std::unordered_map<int,int> face_mat_map;
    int                  obj_id;
    int                  tri_start;   // offset into output triangle array
    int                  tri_count;   // number of triangles this mesh produces

};

struct UsdAnimHandle {
    UsdStageRefPtr           stage;
    GfMatrix4d               axis_correction;
    std::vector<MeshCache>   meshes;
    bool                     has_time_samples;
    bool                     skinning_baked = false;
};

UsdAnimHandle* usd_anim_open(const std::string& path,
                             const std::vector<Triangle>& initial_tris)
{
    UsdStageRefPtr stage = UsdStage::Open(path);
    if (!stage) return nullptr;

    auto* h = new UsdAnimHandle();
    h->stage = stage;

    // Axis correction (same as usd_load)
    TfToken up_axis = UsdGeomGetStageUpAxis(stage);
    h->axis_correction = GfMatrix4d(1.0);
    if (up_axis == UsdGeomTokens->z) {
        h->axis_correction = GfMatrix4d(
            1,  0,  0, 0,
            0,  0, -1, 0,
            0,  1,  0, 0,
            0,  0,  0, 1);
    }

    double mpu = UsdGeomGetStageMetersPerUnit(stage);
    if (mpu <= 0.0) mpu = 1.0;
    if (mpu != 1.0) {
        GfMatrix4d scale_m(1.0);
        scale_m.SetScale(GfVec3d(mpu, mpu, mpu));
        h->axis_correction = h->axis_correction * scale_m;
    }

    h->has_time_samples = false;
    int tri_cursor = 0;
    int next_obj_id = 0;

    // Bake skeletal skinning: deforms rest-pose points into time-sampled
    // world-space points directly on the stage.  After baking, GetPointsAttr()
    // returns deformed points at any time code — no manual LBS needed.
    for (const UsdPrim& p : stage->Traverse(UsdTraverseInstanceProxies())) {
        if (p.IsA<UsdSkelRoot>()) {
            std::cerr << "[usd_anim] baking skeletal skinning for SkelRoot '"
                      << p.GetPath() << "'...\n";
            UsdSkelBakeSkinning(UsdSkelRoot(p));
            h->skinning_baked = true;
            h->has_time_samples = true;
        }
    }
    if (h->skinning_baked)
        std::cerr << "[usd_anim] skinning bake complete\n";

    // Traverse in the same order as usd_load to match triangle layout
    for (const UsdPrim& prim : stage->Traverse(UsdTraverseInstanceProxies())) {
        if (!prim.IsA<UsdGeomMesh>()) continue;
        UsdGeomMesh geom(prim);

        MeshCache mc;
        mc.prim_path = prim.GetPath();
        geom.GetFaceVertexCountsAttr().Get(&mc.face_vertex_counts);
        geom.GetFaceVertexIndicesAttr().Get(&mc.face_vertex_indices);
        mc.orientation = UsdGeomTokens->rightHanded;
        geom.GetOrientationAttr().Get(&mc.orientation);
        mc.subdiv_scheme = TfToken();
        geom.GetSubdivisionSchemeAttr().Get(&mc.subdiv_scheme);

        // Check for time samples on points, xform, or any ancestor xform
        if (geom.GetPointsAttr().GetNumTimeSamples() > 1)
            h->has_time_samples = true;
        // Walk up the prim hierarchy — animations often live on parent xforms
        for (UsdPrim p = prim; p.IsValid(); p = p.GetParent()) {
            UsdGeomXformable xf(p);
            if (!xf) continue;
            std::vector<double> times;
            xf.GetTimeSamples(&times);
            if (times.size() > 1) {
                h->has_time_samples = true;
                break;
            }
        }

        // Cache UVs
        mc.uv_interp = UsdGeomTokens->faceVarying;
        {
            UsdGeomPrimvarsAPI pvAPI(prim);
            static const char* UV_NAMES[] = { "st", "st0", "UVMap", "map1", nullptr };
            for (int ui = 0; UV_NAMES[ui]; ++ui) {
                UsdGeomPrimvar pv = pvAPI.GetPrimvar(TfToken(UV_NAMES[ui]));
                if (pv && pv.IsDefined()) {
                    pv.Get(&mc.uvs);
                    pv.GetIndices(&mc.uv_indices);
                    mc.uv_interp = pv.GetInterpolation();
                    break;
                }
            }
        }

        // Match obj_id and material from initial load
        mc.obj_id = next_obj_id++;

        // Find mat_idx from the initial triangles (first tri of this mesh)
        // Count triangles this mesh produces
        int tri_count = 0;
        for (int fvc : mc.face_vertex_counts)
            tri_count += std::max(0, fvc - 2);

        mc.tri_start = tri_cursor;
        mc.tri_count = tri_count;

        // Read mat_idx and face_mat_map from initial triangles
        if (tri_cursor < (int)initial_tris.size())
            mc.mat_idx = initial_tris[tri_cursor].mat_idx;
        else
            mc.mat_idx = 0;

        // Build face_mat_map by scanning initial tris for varying mat_idx
        {
            int fv = 0, fi = 0, ti = tri_cursor;
            for (int fvc : mc.face_vertex_counts) {
                for (int t = 1; t < fvc - 1; ++t) {
                    if (ti < (int)initial_tris.size()) {
                        int m = initial_tris[ti].mat_idx;
                        if (m != mc.mat_idx)
                            mc.face_mat_map[fi] = m;
                        ti++;
                    }
                }
                fv += fvc;
                fi++;
            }
        }

        tri_cursor += tri_count;
        h->meshes.push_back(std::move(mc));
    }

    // Also check the stage's time code range — if start != end, it's animated
    if (!h->has_time_samples) {
        double start = stage->GetStartTimeCode();
        double end   = stage->GetEndTimeCode();
        if (end > start)
            h->has_time_samples = true;
    }

    std::cout << "[usd_anim] opened stage with " << h->meshes.size()
              << " meshes, " << tri_cursor << " triangles, "
              << (h->has_time_samples ? "ANIMATED" : "static") << "\n";
    return h;
}

bool usd_load_frame(UsdAnimHandle* h, float time,
                    std::vector<Triangle>& triangles_out)
{
    if (!h || !h->stage) return false;

    UsdTimeCode tc(time);
    UsdGeomXformCache xform_cache(tc);

    for (const MeshCache& mc : h->meshes) {
        UsdPrim prim = h->stage->GetPrimAtPath(mc.prim_path);
        if (!prim) continue;
        UsdGeomMesh geom(prim);

        // Time-varying: xform + points
        GfMatrix4d world = xform_cache.GetLocalToWorldTransform(prim) * h->axis_correction;

        VtVec3fArray points;
        geom.GetPointsAttr().Get(&points, tc);
        if (points.empty()) continue;

        // Skinning is already baked into the stage — GetPointsAttr().Get()
        // returns deformed points at any time code.  No manual LBS needed.

        bool flip_winding = (mc.orientation == UsdGeomTokens->leftHanded);
        if (world.GetDeterminant() < 0.0)
            flip_winding = !flip_winding;

        // Normals (re-evaluate at this time code)
        VtVec3fArray normals;
        TfToken normals_interp;
        geom.GetNormalsAttr().Get(&normals, tc);
        normals_interp = geom.GetNormalsInterpolation();

        if (normals.empty()) {
            UsdGeomPrimvarsAPI nAPI(prim);
            UsdGeomPrimvar nPv = nAPI.GetPrimvar(TfToken("normals"));
            if (nPv && nPv.IsDefined()) {
                nPv.Get(&normals, tc);
                normals_interp = nPv.GetInterpolation();
            }
        }

        // If no normals, flat normals computed per-triangle below as fallback.

        // Build triangles using cached topology
        int tri_idx = mc.tri_start;
        int fv_cursor = 0;
        int face_idx = 0;

        for (int fvc : mc.face_vertex_counts) {
            for (int t = 1; t < fvc - 1; ++t) {
                if (tri_idx >= (int)triangles_out.size()) break;

                int i0 = mc.face_vertex_indices[fv_cursor];
                int i1 = mc.face_vertex_indices[fv_cursor + t];
                int i2 = mc.face_vertex_indices[fv_cursor + t + 1];
                if (i0 >= (int)points.size() || i1 >= (int)points.size() || i2 >= (int)points.size()) {
                    tri_idx++;
                    continue;
                }

                float3 wv0 = transform_point(world, points[i0]);
                float3 wv1 = transform_point(world, points[i1]);
                float3 wv2 = transform_point(world, points[i2]);

                // Normals
                auto get_norm = [&](int face_v_idx, int vert_idx) -> float3 {
                    if (normals.empty()) return make_float3(0.f, 0.f, 0.f);
                    int ni = -1;
                    if (normals_interp == UsdGeomTokens->faceVarying)
                        ni = face_v_idx;
                    else if (normals_interp == UsdGeomTokens->vertex)
                        ni = vert_idx;
                    else if (normals_interp == UsdGeomTokens->uniform)
                        ni = face_idx;
                    if (ni < 0 || ni >= (int)normals.size()) return make_float3(0.f,0.f,0.f);
                    return transform_normal(world, normals[ni]);
                };

                float3 wn0 = get_norm(fv_cursor,       i0);
                float3 wn1 = get_norm(fv_cursor + t,   i1);
                float3 wn2 = get_norm(fv_cursor+t+1,   i2);

                float3 flat = safe_normalize(cross3(
                    make_float3(wv1.x-wv0.x, wv1.y-wv0.y, wv1.z-wv0.z),
                    make_float3(wv2.x-wv0.x, wv2.y-wv0.y, wv2.z-wv0.z)));
                if (wn0.x == 0.f && wn0.y == 0.f && wn0.z == 0.f) wn0 = flat;
                if (wn1.x == 0.f && wn1.y == 0.f && wn1.z == 0.f) wn1 = flat;
                if (wn2.x == 0.f && wn2.y == 0.f && wn2.z == 0.f) wn2 = flat;

                if (flip_winding) {
                    std::swap(wv1, wv2);
                    std::swap(wn1, wn2);
                }

                Triangle& tri = triangles_out[tri_idx];
                tri.v0 = wv0; tri.v1 = wv1; tri.v2 = wv2;
                tri.n0 = safe_normalize(wn0);
                tri.n1 = safe_normalize(wn1);
                tri.n2 = safe_normalize(wn2);

                // UVs from cache
                auto get_uv = [&](int face_v_idx, int vert_idx) -> float2 {
                    if (mc.uvs.empty()) return make_float2(0.f, 0.f);
                    int ui = -1;
                    if (!mc.uv_indices.empty()) {
                        if (mc.uv_interp == UsdGeomTokens->faceVarying && face_v_idx < (int)mc.uv_indices.size())
                            ui = mc.uv_indices[face_v_idx];
                        else if (mc.uv_interp == UsdGeomTokens->vertex && vert_idx < (int)mc.uv_indices.size())
                            ui = mc.uv_indices[vert_idx];
                    } else {
                        if (mc.uv_interp == UsdGeomTokens->faceVarying)
                            ui = face_v_idx;
                        else if (mc.uv_interp == UsdGeomTokens->vertex)
                            ui = vert_idx;
                    }
                    if (ui < 0 || ui >= (int)mc.uvs.size()) return make_float2(0.f, 0.f);
                    return make_float2(mc.uvs[ui][0], 1.f - mc.uvs[ui][1]);
                };

                float2 tuv0 = get_uv(fv_cursor,     i0);
                float2 tuv1 = get_uv(fv_cursor + t, i1);
                float2 tuv2 = get_uv(fv_cursor+t+1, i2);
                if (flip_winding) std::swap(tuv1, tuv2);
                tri.uv0 = tuv0; tri.uv1 = tuv1; tri.uv2 = tuv2;

                // Tangent
                {
                    float3 e1 = make_float3(wv1.x-wv0.x, wv1.y-wv0.y, wv1.z-wv0.z);
                    float3 e2 = make_float3(wv2.x-wv0.x, wv2.y-wv0.y, wv2.z-wv0.z);
                    float du1 = tuv1.x - tuv0.x, dv1 = tuv1.y - tuv0.y;
                    float du2 = tuv2.x - tuv0.x, dv2 = tuv2.y - tuv0.y;
                    float det = du1*dv2 - du2*dv1;
                    float3 T;
                    if (fabsf(det) > 1e-8f) {
                        float inv = 1.f / det;
                        T = safe_normalize(make_float3(
                            (dv2*e1.x - dv1*e2.x) * inv,
                            (dv2*e1.y - dv1*e2.y) * inv,
                            (dv2*e1.z - dv1*e2.z) * inv));
                    } else {
                        float3 n = tri.n0;
                        float3 up = (fabsf(n.z) < 0.999f)
                            ? make_float3(0.f,0.f,1.f) : make_float3(1.f,0.f,0.f);
                        T = safe_normalize(cross3(up, n));
                    }
                    float3 B = cross3(tri.n0, T);
                    float bsign = (dot3(B, make_float3(
                        (du2*e1.x - du1*e2.x), (du2*e1.y - du1*e2.y),
                        (du2*e1.z - du1*e2.z))) >= 0.f) ? 1.f : -1.f;
                    float4 tang = make_float4(T.x, T.y, T.z, bsign);
                    tri.t0 = tang; tri.t1 = tang; tri.t2 = tang;
                }

                // Material from cache
                auto fm = mc.face_mat_map.find(face_idx);
                tri.mat_idx = (fm != mc.face_mat_map.end()) ? fm->second : mc.mat_idx;
                tri.obj_id  = mc.obj_id;

                tri_idx++;
            }
            fv_cursor += fvc;
            face_idx++;
        }
    }
    return true;
}

bool usd_has_animation(UsdAnimHandle* h)
{
    return h && h->has_time_samples;
}

void usd_anim_close(UsdAnimHandle* h)
{
    delete h;
}
