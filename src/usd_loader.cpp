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
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/metrics.h>    // UsdGeomGetStageUpAxis
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

    // Helper: read a color3 input — either inline value or texture connection.
    // is_srgb=true for base/emissive (sRGB-encoded), false for data textures.
    auto read_color3 = [&](const char* name, float4& out, bool is_srgb) {
        UsdShadeInput inp = shader.GetInput(TfToken(name));
        if (!inp) return;
        GfVec3f v;
        if (inp.Get(&v)) {
            out = make_float4(v[0], v[1], v[2], 1.f);
        } else {
            UsdShadeShader src = get_connected_shader(inp);
            if (!src) return;
            int idx = get_tex_idx(src, is_srgb);
            if (idx < 0) return;
            if (strcmp(name, "diffuseColor") == 0 || strcmp(name, "baseColor") == 0)
                gm.base_color_tex = idx;
            else if (strcmp(name, "emissiveColor") == 0)
                gm.emissive_tex = idx;
        }
    };

    auto read_float = [&](const char* name, float& out) {
        UsdShadeInput inp = shader.GetInput(TfToken(name));
        if (!inp) return;
        inp.Get(&out);
    };

    // Helper: read a single-channel float input that may be a texture.
    // Stores the texture index in tex_idx_out if connected.
    auto read_float_tex = [&](const char* name, float& scalar_out, int& tex_idx_out) {
        UsdShadeInput inp = shader.GetInput(TfToken(name));
        if (!inp) return;
        if (!inp.Get(&scalar_out)) {
            UsdShadeShader src = get_connected_shader(inp);
            if (src) tex_idx_out = get_tex_idx(src, /*is_srgb=*/false);
        }
    };

    // Print shader id so we can confirm it's UsdPreviewSurface
    {
        TfToken shader_id;
        shader.GetShaderId(&shader_id);
        std::cerr << "[usd_loader] shader id='" << shader_id << "'\n";
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

    // If metallic and roughness use the same texture (common in USD exports from Blender),
    // use it as the combined metallic-roughness map (G=roughness, B=metallic like glTF).
    if (metallic_tex_idx >= 0 && metallic_tex_idx == roughness_tex_idx)
        gm.metallic_rough_tex = metallic_tex_idx;
    else if (roughness_tex_idx >= 0)
        gm.metallic_rough_tex = roughness_tex_idx;  // roughness in G channel
    else if (metallic_tex_idx >= 0)
        gm.metallic_rough_tex = metallic_tex_idx;   // metallic in B channel

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

// ─────────────────────────────────────────────
//  Main loader
// ─────────────────────────────────────────────

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
        // Last resort: walk descendants looking for a UsdPreviewSurface shader
        if (!surface_shader) {
            for (const UsdPrim& child : mat.GetPrim().GetDescendants()) {
                UsdShadeShader sh(child);
                if (!sh) continue;
                TfToken id; sh.GetShaderId(&id);
                if (id == TfToken("UsdPreviewSurface") ||
                    id == TfToken("ND_UsdPreviewSurface_surfaceshader")) {
                    surface_shader = sh; break;
                }
            }
        }
        GpuMaterial gm = surface_shader
            ? read_preview_surface(surface_shader, path, textures_out, tex_cache)
            : default_mat;
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

        // ── World transform ───────────────────────────────────────────
        // Post-multiply: world * correction so correction applies AFTER world transform.
        // (USD row-vector convention: point is transformed as  p * matrix)
        GfMatrix4d world = xform_cache.GetLocalToWorldTransform(prim) * axis_correction;

        // ── Topology ──────────────────────────────────────────────────
        VtIntArray face_vertex_counts, face_vertex_indices;
        geom.GetFaceVertexCountsAttr().Get(&face_vertex_counts);
        geom.GetFaceVertexIndicesAttr().Get(&face_vertex_indices);

        VtVec3fArray points;
        geom.GetPointsAttr().Get(&points);
        if (points.empty()) continue;

        // ── Normals (optional) ────────────────────────────────────────
        VtVec3fArray normals;
        TfToken normals_interp;
        geom.GetNormalsAttr().Get(&normals);
        normals_interp = geom.GetNormalsInterpolation();

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
            UsdShadeMaterialBindingAPI binding(prim);
            UsdShadeMaterial mat = binding.ComputeBoundMaterial();
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

            UsdShadeMaterialBindingAPI sub_bind(child);
            UsdShadeMaterial sub_mat = sub_bind.ComputeBoundMaterial();
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

                Triangle tri{};
                tri.v0 = wv0; tri.v1 = wv1; tri.v2 = wv2;
                tri.n0 = safe_normalize(wn0);
                tri.n1 = safe_normalize(wn1);
                tri.n2 = safe_normalize(wn2);
                tri.uv0 = get_uv(fv_cursor,     i0);
                tri.uv1 = get_uv(fv_cursor + t, i1);
                tri.uv2 = get_uv(fv_cursor+t+1, i2);
                tri.t0 = make_float4(0.f, 0.f, 0.f, 0.f);  // no tangents yet
                tri.t1 = make_float4(0.f, 0.f, 0.f, 0.f);
                tri.t2 = make_float4(0.f, 0.f, 0.f, 0.f);
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
