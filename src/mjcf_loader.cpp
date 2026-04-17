// ─────────────────────────────────────────────
//  mjcf_loader.cpp
//  Loads MuJoCo MJCF (.xml) robot description files.
//  Targets the mujoco_menagerie set (Franka, Go2, Shadow Hand, etc.).
//
//  Supported:
//    - <include file="..."/> recursive resolution
//    - <compiler angle meshdir autolimits>
//    - <default class> inheritance
//    - <asset> <mesh> (.stl, .obj) and <material>
//    - <worldbody> nested <body> with pos/quat/euler/axisangle
//    - <geom type="mesh"> with visual/collision class filtering
//    - <joint> type="hinge"|"slide" with range limits
//
//  Produces a UrdfArticulation* so it reuses the URDF articulation/IK
//  pipeline and UI panels unchanged.
// ─────────────────────────────────────────────

#include "mjcf_loader.h"
#include "urdf_internal.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────
//  Small parsing helpers
// ─────────────────────────────────────────────

static std::vector<float> parse_floats_str(const std::string& s) {
    std::vector<float> out;
    std::istringstream ss(s);
    float v;
    while (ss >> v) out.push_back(v);
    return out;
}

static float3 parse_vec3(const std::string& s, float3 def = {0, 0, 0}) {
    auto v = parse_floats_str(s);
    if (v.size() >= 3) return make_float3(v[0], v[1], v[2]);
    return def;
}

static float parse_float_attr(const std::string& s, float def) {
    if (s.empty()) return def;
    try { return std::stof(s); } catch (...) { return def; }
}

// Build a Mat4 from MJCF body-style transform attributes.
// Reads pos + one of {quat, euler, axisangle}. angle_scale converts MJCF
// angle units to radians (1 for radian, π/180 for degree).
static Mat4 parse_mjcf_transform(float3 pos, const std::string& quat,
                                  const std::string& euler, const std::string& axisangle,
                                  float angle_scale)
{
    if (!quat.empty()) {
        auto v = parse_floats_str(quat);
        if (v.size() >= 4)
            return Mat4::from_quat_wxyz_xyz(v[0], v[1], v[2], v[3], pos.x, pos.y, pos.z);
    }
    if (!euler.empty()) {
        auto v = parse_floats_str(euler);
        if (v.size() >= 3) {
            // MJCF default euler sequence is "xyz" (intrinsic x-y-z); matches our RPY.
            return Mat4::from_rpy_xyz(v[0] * angle_scale, v[1] * angle_scale,
                                     v[2] * angle_scale, pos.x, pos.y, pos.z);
        }
    }
    if (!axisangle.empty()) {
        auto v = parse_floats_str(axisangle);
        if (v.size() >= 4)
            return Mat4::from_axisangle_xyz(v[0], v[1], v[2], v[3] * angle_scale,
                                            pos.x, pos.y, pos.z);
    }
    // Translation only
    Mat4 m = Mat4::identity();
    m.m[0][3] = pos.x; m.m[1][3] = pos.y; m.m[2][3] = pos.z;
    return m;
}

// ─────────────────────────────────────────────
//  MJCF scene data
// ─────────────────────────────────────────────

struct MjcfCompiler {
    float       angle_scale = 1.0f;        // 1.0 = radians; π/180 = degrees
    std::string meshdir;
    std::string texturedir;
    bool        autolimits = false;
};

struct MjcfMaterialDef {
    float4 rgba      = make_float4(1, 1, 1, 1);
    float  specular  = 0.0f;
    float  shininess = 0.5f;
    float  reflectance = 0.0f;
};

struct MjcfMeshDef {
    std::string file_abs;
    float3      scale     = {1, 1, 1};
};

using AttrMap = std::unordered_map<std::string, std::string>;

struct MjcfDefault {
    std::string                              parent_name;
    std::unordered_map<std::string, AttrMap> tag_defaults;
};

struct MjcfGeom {
    std::string type;            // "mesh", "box", "plane", "sphere", ...
    std::string mesh_name;
    std::string material_name;
    Mat4        local_xform;
    float4      rgba     = make_float4(1, 1, 1, 1);
    bool        has_rgba = false;
    int         group    = 0;
    float3      size     = {0, 0, 0};
};

struct MjcfJoint {
    std::string name;
    std::string type;            // "hinge" or "slide"
    float3      axis     = {0, 0, 1};
    float3      pos      = {0, 0, 0};
    float       range_lo = 0.0f;
    float       range_hi = 0.0f;
    bool        has_range = false;
    bool        limited   = false;
};

struct MjcfBody {
    std::string            name;
    Mat4                   local_xform = Mat4::identity();
    std::string            childclass;
    std::vector<MjcfJoint> joints;
    std::vector<MjcfGeom>  geoms;
    std::vector<MjcfBody>  children;
};

struct MjcfScene {
    MjcfCompiler                                    compiler;
    std::string                                     base_dir;
    std::string                                     meshdir_abs;
    std::unordered_map<std::string, MjcfDefault>    defaults;
    std::unordered_map<std::string, MjcfMaterialDef> materials;
    std::unordered_map<std::string, MjcfMeshDef>    meshes;
    MjcfBody                                        worldbody;
};

// ─────────────────────────────────────────────
//  Default class resolution (attribute lookup)
// ─────────────────────────────────────────────

// The MJCF "main" default is the one without a class attribute. We store it
// under the name "main" internally; references to class="" also resolve there.
static std::string resolve_default_attr(const MjcfScene& scene,
                                         const std::string& class_name,
                                         const std::string& tag,
                                         const std::string& attr_name,
                                         const std::string& fallback)
{
    std::string cur = class_name.empty() ? std::string("main") : class_name;

    for (int guard = 0; guard < 32; ++guard) {
        auto it = scene.defaults.find(cur);
        if (it == scene.defaults.end()) break;

        auto tit = it->second.tag_defaults.find(tag);
        if (tit != it->second.tag_defaults.end()) {
            auto ait = tit->second.find(attr_name);
            if (ait != tit->second.end()) return ait->second;
        }
        if (it->second.parent_name.empty()) break;
        cur = it->second.parent_name;
    }
    return fallback;
}

// Resolve an attribute for an element: inline attr wins, then class defaults.
static std::string resolve_element_attr(const MjcfScene& scene,
                                         const XmlNode& elem,
                                         const std::string& effective_class,
                                         const std::string& attr_name,
                                         const std::string& fallback)
{
    std::string inline_val = elem.attr(attr_name);
    if (!inline_val.empty()) return inline_val;
    return resolve_default_attr(scene, effective_class, elem.tag, attr_name, fallback);
}

// ─────────────────────────────────────────────
//  Include resolution
// ─────────────────────────────────────────────

// Returns the first <mujoco> node found in the parsed XML (or the root if it is).
static const XmlNode* find_mujoco_root(const XmlNode& doc)
{
    if (doc.tag == "mujoco") return &doc;
    for (auto& c : doc.children)
        if (c.tag == "mujoco") return &c;
    return nullptr;
}

// Recursively replace <include file="..."/> elements by the root-level content
// of the referenced file's <mujoco> node. Includes are resolved relative to
// the directory of the file containing them.
static void expand_includes(XmlNode& mujoco_node, const std::string& xml_dir,
                            std::unordered_set<std::string>& visited)
{
    std::vector<XmlNode> out;
    out.reserve(mujoco_node.children.size());

    for (auto& child : mujoco_node.children) {
        if (child.tag == "include") {
            std::string file = child.attr("file");
            if (file.empty()) continue;
            fs::path inc_path = fs::path(xml_dir) / file;
            std::string canonical;
            try { canonical = fs::weakly_canonical(inc_path).string(); }
            catch (...) { canonical = inc_path.string(); }
            if (visited.count(canonical)) {
                std::cerr << "[mjcf_loader] skipping cyclic include: " << canonical << '\n';
                continue;
            }
            visited.insert(canonical);

            std::string xml_text = read_file_text(canonical);
            if (xml_text.empty()) {
                std::cerr << "[mjcf_loader] cannot read include: " << canonical << '\n';
                continue;
            }
            XmlNode included = parse_xml(xml_text);
            const XmlNode* mjr = find_mujoco_root(included);
            if (!mjr) {
                std::cerr << "[mjcf_loader] include missing <mujoco>: " << canonical << '\n';
                continue;
            }
            // Recurse: expand nested includes first, using the included file's dir
            XmlNode expanded = *mjr;
            expand_includes(expanded, fs::path(canonical).parent_path().string(), visited);
            for (auto& c : expanded.children) out.push_back(std::move(c));
        } else {
            out.push_back(std::move(child));
        }
    }
    mujoco_node.children = std::move(out);
}

// ─────────────────────────────────────────────
//  Default tree parsing
// ─────────────────────────────────────────────

// Walks a <default> subtree, collecting per-class tag defaults. Outer <default>
// (no class attr) is named "main". Nested <default class="X"> inherits from
// its parent default's name.
static void collect_defaults(const XmlNode& default_node,
                              const std::string& class_name,
                              const std::string& parent_name,
                              MjcfScene& scene)
{
    // Ensure entry exists
    MjcfDefault& d = scene.defaults[class_name];
    d.parent_name = parent_name;

    for (auto& child : default_node.children) {
        if (child.tag == "default") {
            std::string sub_class = child.attr("class");
            if (sub_class.empty()) continue;   // malformed
            collect_defaults(child, sub_class, class_name, scene);
        } else {
            // Record all attributes as defaults for this tag
            AttrMap am;
            for (auto& a : child.attrs) am[a.name] = a.value;
            if (!am.empty()) {
                auto& tag_map = d.tag_defaults[child.tag];
                for (auto& [k, v] : am) tag_map[k] = v;
            }
        }
    }
}

// ─────────────────────────────────────────────
//  Asset parsing
// ─────────────────────────────────────────────

static void parse_asset(const XmlNode& asset, MjcfScene& scene)
{
    for (auto& child : asset.children) {
        if (child.tag == "mesh") {
            MjcfMeshDef md;
            std::string name = child.attr("name");
            std::string file = child.attr("file");
            // MJCF mesh name defaults to filename without extension if "name" absent
            if (name.empty() && !file.empty()) {
                name = fs::path(file).stem().string();
            }
            if (name.empty() || file.empty()) continue;

            fs::path abs = fs::path(scene.meshdir_abs) / file;
            md.file_abs = abs.string();

            std::string scale_str = child.attr("scale");
            if (!scale_str.empty()) {
                auto v = parse_floats_str(scale_str);
                if (v.size() >= 3) md.scale = make_float3(v[0], v[1], v[2]);
            }

            scene.meshes[name] = md;
        } else if (child.tag == "material") {
            MjcfMaterialDef mm;
            std::string name = child.attr("name");
            if (name.empty()) continue;
            std::string cls = child.attr("class");

            // rgba: inline, else from class default, else default white
            std::string rgba_str = resolve_element_attr(scene, child, cls, "rgba", "1 1 1 1");
            auto rv = parse_floats_str(rgba_str);
            if (rv.size() >= 4) mm.rgba = make_float4(rv[0], rv[1], rv[2], rv[3]);
            else if (rv.size() == 3) mm.rgba = make_float4(rv[0], rv[1], rv[2], 1.0f);

            mm.specular   = parse_float_attr(
                resolve_element_attr(scene, child, cls, "specular", "0"), 0.0f);
            mm.shininess  = parse_float_attr(
                resolve_element_attr(scene, child, cls, "shininess", "0.5"), 0.5f);
            mm.reflectance = parse_float_attr(
                resolve_element_attr(scene, child, cls, "reflectance", "0"), 0.0f);

            scene.materials[name] = mm;
        }
        // <texture>, <skin>, <hfield> — not handled for MVP
    }
}

// ─────────────────────────────────────────────
//  Worldbody / body tree parsing
// ─────────────────────────────────────────────

static void parse_geom(const XmlNode& geom_node, const std::string& body_class,
                        const MjcfScene& scene, MjcfGeom& out)
{
    std::string cls = geom_node.attr("class");
    if (cls.empty()) cls = body_class;

    auto A = [&](const std::string& name, const std::string& def) {
        return resolve_element_attr(scene, geom_node, cls, name, def);
    };

    out.type          = A("type", "sphere");   // MJCF default is sphere, but we'll skip unknowns
    out.mesh_name     = A("mesh", "");
    out.material_name = A("material", "");
    out.group         = (int)parse_float_attr(A("group", "0"), 0);

    // rgba: inline or default-class (may be empty), overrides material colour if set.
    std::string rgba_str = A("rgba", "");
    if (!rgba_str.empty()) {
        auto v = parse_floats_str(rgba_str);
        if (v.size() >= 4) { out.rgba = make_float4(v[0], v[1], v[2], v[3]); out.has_rgba = true; }
        else if (v.size() == 3) { out.rgba = make_float4(v[0], v[1], v[2], 1); out.has_rgba = true; }
    }

    float3 pos = parse_vec3(A("pos", "0 0 0"));
    out.local_xform = parse_mjcf_transform(
        pos,
        A("quat", ""),
        A("euler", ""),
        A("axisangle", ""),
        scene.compiler.angle_scale);

    std::string size_str = A("size", "");
    if (!size_str.empty()) out.size = parse_vec3(size_str);
}

static void parse_joint(const XmlNode& joint_node, const std::string& body_class,
                         const MjcfScene& scene, MjcfJoint& out)
{
    std::string cls = joint_node.attr("class");
    if (cls.empty()) cls = body_class;

    auto A = [&](const std::string& name, const std::string& def) {
        return resolve_element_attr(scene, joint_node, cls, name, def);
    };

    out.name = A("name", "");
    out.type = A("type", "hinge");
    out.axis = parse_vec3(A("axis", "0 0 1"), make_float3(0, 0, 1));
    out.pos  = parse_vec3(A("pos", "0 0 0"));

    std::string range_str = A("range", "");
    if (!range_str.empty()) {
        auto v = parse_floats_str(range_str);
        if (v.size() >= 2) {
            out.range_lo = v[0] * (out.type == "slide" ? 1.0f : scene.compiler.angle_scale);
            out.range_hi = v[1] * (out.type == "slide" ? 1.0f : scene.compiler.angle_scale);
            out.has_range = true;
        }
    }

    std::string limited_str = A("limited", "");
    if (limited_str == "true" || scene.compiler.autolimits) out.limited = true;
    if (out.has_range && limited_str != "false") out.limited = true;
}

static void parse_body(const XmlNode& body_node, const std::string& inherited_class,
                        const MjcfScene& scene, MjcfBody& out)
{
    out.name = body_node.attr("name");

    // childclass attribute changes the default class for all descendants
    out.childclass = body_node.attr("childclass");
    std::string effective_class = out.childclass.empty() ? inherited_class : out.childclass;

    // Transform: pos + quat/euler/axisangle (no class inheritance for body transform)
    float3 pos = parse_vec3(body_node.attr("pos", "0 0 0"));
    out.local_xform = parse_mjcf_transform(pos,
        body_node.attr("quat"),
        body_node.attr("euler"),
        body_node.attr("axisangle"),
        scene.compiler.angle_scale);

    for (auto& child : body_node.children) {
        if (child.tag == "geom") {
            MjcfGeom g;
            parse_geom(child, effective_class, scene, g);
            out.geoms.push_back(g);
        } else if (child.tag == "joint") {
            MjcfJoint j;
            parse_joint(child, effective_class, scene, j);
            out.joints.push_back(j);
        } else if (child.tag == "body") {
            MjcfBody sub;
            parse_body(child, effective_class, scene, sub);
            out.children.push_back(std::move(sub));
        }
        // Skip <inertial>, <site>, <camera>, <light>, <composite>, etc.
    }
}

// ─────────────────────────────────────────────
//  Top-level MJCF parsing
// ─────────────────────────────────────────────

static bool parse_mjcf(const std::string& path, MjcfScene& scene)
{
    std::string xml_text = read_file_text(path);
    if (xml_text.empty()) {
        std::cerr << "[mjcf_loader] cannot read: " << path << '\n';
        return false;
    }

    XmlNode doc = parse_xml(xml_text);
    const XmlNode* mjr_ptr = find_mujoco_root(doc);
    if (!mjr_ptr) {
        std::cerr << "[mjcf_loader] no <mujoco> root element in " << path << '\n';
        return false;
    }
    XmlNode mujoco = *mjr_ptr;

    scene.base_dir = fs::path(path).parent_path().string();

    // 1. Expand <include> recursively
    std::unordered_set<std::string> visited;
    try { visited.insert(fs::weakly_canonical(fs::path(path)).string()); } catch (...) {}
    expand_includes(mujoco, scene.base_dir, visited);

    // 2. <compiler>
    for (auto* comp : mujoco.children_with_tag("compiler")) {
        std::string angle = comp->attr("angle");
        if (angle == "degree") scene.compiler.angle_scale = 3.14159265f / 180.0f;
        else                   scene.compiler.angle_scale = 1.0f;

        std::string md = comp->attr("meshdir");
        if (!md.empty()) scene.compiler.meshdir = md;

        std::string td = comp->attr("texturedir");
        if (!td.empty()) scene.compiler.texturedir = td;

        scene.compiler.autolimits = (comp->attr("autolimits") == "true");
    }

    // Resolve meshdir to absolute path relative to base_dir
    if (scene.compiler.meshdir.empty()) scene.meshdir_abs = scene.base_dir;
    else scene.meshdir_abs = (fs::path(scene.base_dir) / scene.compiler.meshdir).string();

    // 3. <default> — seed "main" then recurse
    // The root default in MJCF has no class attribute (it's the top-level <default>).
    // Ensure "main" exists even if there's no root default (so resolve_* doesn't error).
    scene.defaults["main"].parent_name = "";

    for (auto* def : mujoco.children_with_tag("default")) {
        // Named top-level <default class="X"> also exists (MJCF allows this).
        std::string root_class = def->attr("class");
        if (root_class.empty()) {
            // The true root default: collect its direct children under "main"
            for (auto& sub : def->children) {
                if (sub.tag == "default") {
                    std::string sub_class = sub.attr("class");
                    if (sub_class.empty()) continue;
                    collect_defaults(sub, sub_class, "main", scene);
                } else {
                    AttrMap am;
                    for (auto& a : sub.attrs) am[a.name] = a.value;
                    auto& tm = scene.defaults["main"].tag_defaults[sub.tag];
                    for (auto& [k, v] : am) tm[k] = v;
                }
            }
        } else {
            collect_defaults(*def, root_class, "main", scene);
        }
    }

    // 4. <asset>
    for (auto* asset : mujoco.children_with_tag("asset")) {
        parse_asset(*asset, scene);
    }

    // 5. <worldbody>
    scene.worldbody.name = "";
    scene.worldbody.local_xform = Mat4::identity();
    for (auto* wb : mujoco.children_with_tag("worldbody")) {
        // Worldbody is like a body with no transform; parse its children directly
        for (auto& c : wb->children) {
            if (c.tag == "body") {
                MjcfBody sub;
                parse_body(c, "", scene, sub);
                scene.worldbody.children.push_back(std::move(sub));
            } else if (c.tag == "geom") {
                MjcfGeom g;
                parse_geom(c, "", scene, g);
                scene.worldbody.geoms.push_back(g);
            }
            // Skip <light>, <camera>, <site>.
        }
    }

    std::cerr << "[mjcf_loader] parsed " << path
              << ": defaults=" << scene.defaults.size()
              << " materials=" << scene.materials.size()
              << " meshes=" << scene.meshes.size()
              << " top-bodies=" << scene.worldbody.children.size() << '\n';

    return true;
}

// ─────────────────────────────────────────────
//  Visibility filter
// ─────────────────────────────────────────────
// A geom is "visual" if it's a mesh or primitive we can render AND it's not
// marked as collision-only. Heuristic: group 2 is visual (Menagerie convention),
// group 3 is collision. Also accept group 0 (default) as visual if the geom has
// a material or mesh attached.

static bool is_visual_geom(const MjcfGeom& g)
{
    if (g.type != "mesh") return false;          // MVP: only mesh-type visuals
    if (g.mesh_name.empty()) return false;
    if (g.group == 3 || g.group == 4) return false;
    return true;
}

// ─────────────────────────────────────────────
//  Convert MJCF → URDF-compatible kinematic structures
//
//  For each MJCF body with N visual geoms:
//    • One URDFLink per body (no mesh) carrying the body-local transform
//    • One URDFLink per geom (mesh=geom.mesh.file, visual_origin=geom.local_xform)
//    • Fixed URDFJoints glueing the geom-links to their body-link
//  For each joint inside a body, a revolute/prismatic URDFJoint links parent
//  body-link to current body-link (the body's local_xform becomes the joint
//  origin). Bodies with NO joints get a fixed URDFJoint from parent.
// ─────────────────────────────────────────────

struct MjcfFlatten {
    std::unordered_map<std::string, URDFLink> links;
    std::vector<URDFJoint>                    joints;
    std::string                               root_link;
    // Per-geom link (in emission order): resolved material index and rgba overlay
    struct GeomLinkMaterial {
        std::string geom_link_name;
        int         gpu_mat_idx;
    };
    std::vector<GeomLinkMaterial> geom_materials;
};

// Unique ID counter to disambiguate synthetic names (unnamed bodies, unnamed joints)
static int mjcf_unique_id_counter = 0;

static std::string make_unique_body_name(const std::string& hint)
{
    int id = ++mjcf_unique_id_counter;
    std::string base = hint.empty() ? std::string("body") : hint;
    return base + "__mj" + std::to_string(id);
}

// Walks the body tree. Every body gets a URDFLink named by its body name (with
// uniquification for empty names). Body-local transform becomes the parent joint
// origin; its movable joints become URDFJoints with that origin; movable joints
// beyond the first are chained via zero-origin links (composite joint bodies).
static void flatten_body(const MjcfBody& body, const std::string& parent_link_name,
                          MjcfScene& scene, MjcfFlatten& flat, int& obj_counter,
                          std::vector<GpuMaterial>& materials_out,
                          const std::function<int(const std::string&, float4, bool)>& get_or_add_material)
{
    // Assign a unique link name for this body
    std::string body_link = body.name.empty()
        ? make_unique_body_name("body")
        : body.name;

    // Avoid collisions with existing links (e.g. meshes-with-same-name-as-body edge case)
    if (flat.links.count(body_link)) {
        body_link = make_unique_body_name(body_link);
    }

    // Create the body's URDFLink (no mesh — it's a transform node)
    URDFLink L;
    L.name = body_link;
    L.visual_origin = Mat4::identity();
    flat.links[body_link] = L;

    // Connect parent → body with a URDFJoint for each MJCF joint on this body.
    // For bodies with no MJCF joints (fixed welded bodies), emit a single fixed joint.
    // For bodies with multiple MJCF joints, we insert synthetic intermediate links:
    //   parent_link → [J0] → link#0 → [J1] → link#1 → ... → body_link
    // Each synthetic link has no geoms; only body_link carries the geoms.
    std::string prev_link = parent_link_name;
    Mat4 pending_origin = body.local_xform;  // applied on the first emitted joint only
    bool first = true;

    auto emit_joint = [&](const std::string& child_link, const std::string& jname,
                          const std::string& jtype, float3 axis, float lo, float hi,
                          bool has_range, Mat4 origin)
    {
        URDFJoint J;
        J.name   = jname.empty()
            ? (std::string("j_") + prev_link + "_" + child_link)
            : jname;
        J.parent = prev_link;
        J.child  = child_link;
        J.origin = origin;
        J.axis   = axis;
        J.type   = jtype;
        if (has_range) { J.lower = lo; J.upper = hi; }
        flat.joints.push_back(J);
    };

    if (body.joints.empty()) {
        // Fixed weld
        emit_joint(body_link, "", "fixed", make_float3(0, 0, 1), 0, 0, false, pending_origin);
    } else {
        for (size_t k = 0; k < body.joints.size(); ++k) {
            const auto& j = body.joints[k];
            std::string child;
            if (k + 1 == body.joints.size()) child = body_link;
            else {
                child = make_unique_body_name(body_link + "_j" + std::to_string(k));
                URDFLink syn; syn.name = child; syn.visual_origin = Mat4::identity();
                flat.links[child] = syn;
            }

            std::string jtype_urdf = (j.type == "slide") ? "prismatic" : "revolute";
            // MJCF 'hinge' with no range → continuous (URDF convention)
            if (j.type == "hinge" && !j.has_range && !j.limited)
                jtype_urdf = "continuous";

            // Use body's local transform for the first emitted joint, identity after
            Mat4 origin = first ? pending_origin : Mat4::identity();
            first = false;

            // NOTE: MJCF joints may also have a <joint pos="..."/> offset from the body
            // origin. For the targeted Menagerie robots (Panda, etc.) joint pos is ~0.
            // We ignore joint pos here — supporting it requires composing into origin.

            emit_joint(child, j.name, jtype_urdf, j.axis,
                       j.range_lo, j.range_hi, j.has_range, origin);
            prev_link = child;
        }
    }

    // Attach each visual geom as a child fixed-joint link of body_link.
    for (size_t gi = 0; gi < body.geoms.size(); ++gi) {
        const MjcfGeom& g = body.geoms[gi];
        if (!is_visual_geom(g)) continue;

        auto mit = scene.meshes.find(g.mesh_name);
        if (mit == scene.meshes.end()) continue;

        std::string geom_link = make_unique_body_name(body_link + "_g" + std::to_string(gi));
        URDFLink GL;
        GL.name             = geom_link;
        GL.visual_mesh_path = mit->second.file_abs;
        GL.visual_origin    = g.local_xform;
        flat.links[geom_link] = GL;

        URDFJoint FJ;
        FJ.name   = geom_link + "_fixed";
        FJ.parent = body_link;
        FJ.child  = geom_link;
        FJ.type   = "fixed";
        FJ.origin = Mat4::identity();
        flat.joints.push_back(FJ);

        // Determine material (rgba override > geom.material > grey fallback)
        float4 rgba = make_float4(0.72f, 0.72f, 0.74f, 1.0f);
        bool have_override = false;
        if (g.has_rgba) { rgba = g.rgba; have_override = true; }
        else if (!g.material_name.empty()) {
            auto mmit = scene.materials.find(g.material_name);
            if (mmit != scene.materials.end()) { rgba = mmit->second.rgba; have_override = true; }
        }
        int midx = get_or_add_material(g.material_name, rgba, have_override);
        flat.geom_materials.push_back({ geom_link, midx });
    }

    // Recurse into children
    for (auto& sub : body.children) {
        flatten_body(sub, body_link, scene, flat, obj_counter, materials_out, get_or_add_material);
    }
}

// ─────────────────────────────────────────────
//  Public API — mjcf_load
// ─────────────────────────────────────────────

bool mjcf_load(const std::string&          path,
               std::vector<Triangle>&      triangles_out,
               std::vector<GpuMaterial>&   materials_out,
               std::vector<TextureImage>&  /*textures_out*/,
               std::vector<MeshObject>&    objects_out)
{
    mjcf_unique_id_counter = 0;

    MjcfScene scene;
    if (!parse_mjcf(path, scene)) return false;

    // Flatten MJCF body tree → URDF-compatible link/joint lists.
    // Material cache: MJCF material name → GpuMaterial index.
    std::unordered_map<std::string, int> mat_name_to_idx;

    auto material_from_mjcf = [&](const MjcfMaterialDef& mm) -> GpuMaterial {
        GpuMaterial gm{};
        gm.base_color       = mm.rgba;
        // Specular/shininess heuristic mapping to PBR roughness/metallic.
        // MJCF shininess is in [0,1]; 1 = sharp (low roughness), 0 = diffuse.
        gm.metallic         = mm.specular > 0.5f ? 0.3f : 0.05f;
        gm.roughness        = std::max(0.1f, 1.0f - mm.shininess);
        gm.emissive_factor  = make_float4(0, 0, 0, 0);
        gm.base_color_tex   = -1;
        gm.metallic_rough_tex = -1;
        gm.normal_tex       = -1;
        gm.emissive_tex     = -1;
        gm.custom_shader    = 0;
        gm.normal_y_flip    = 0;
        return gm;
    };

    auto make_grey_material = [&](float4 rgba) -> GpuMaterial {
        GpuMaterial gm{};
        gm.base_color       = rgba;
        gm.metallic         = 0.1f;
        gm.roughness        = 0.5f;
        gm.emissive_factor  = make_float4(0, 0, 0, 0);
        gm.base_color_tex   = -1;
        gm.metallic_rough_tex = -1;
        gm.normal_tex       = -1;
        gm.emissive_tex     = -1;
        gm.custom_shader    = 0;
        gm.normal_y_flip    = 0;
        return gm;
    };

    auto get_or_add_material = [&](const std::string& name, float4 rgba, bool have_override) -> int {
        // Key by material name when available (to dedupe), else by a synthetic key for inline rgbas.
        std::string key = !name.empty() ? name : std::string("__inline_") + std::to_string(materials_out.size());
        auto it = mat_name_to_idx.find(key);
        if (it != mat_name_to_idx.end()) return it->second;

        int idx = (int)materials_out.size();
        if (!name.empty()) {
            auto mit = scene.materials.find(name);
            if (mit != scene.materials.end()) {
                GpuMaterial gm = material_from_mjcf(mit->second);
                if (have_override) gm.base_color = rgba;   // geom's inline rgba wins
                materials_out.push_back(gm);
            } else {
                materials_out.push_back(make_grey_material(rgba));
            }
        } else {
            materials_out.push_back(make_grey_material(rgba));
        }
        mat_name_to_idx[key] = idx;
        return idx;
    };

    MjcfFlatten flat;
    // Synthetic root link (worldbody)
    flat.root_link = "__world__";
    URDFLink world_link;
    world_link.name = flat.root_link;
    world_link.visual_origin = Mat4::identity();
    flat.links[flat.root_link] = world_link;

    int obj_counter = 0;
    for (auto& top : scene.worldbody.children) {
        flatten_body(top, flat.root_link, scene, flat, obj_counter,
                     materials_out, get_or_add_material);
    }

    if (flat.links.size() <= 1) {
        std::cerr << "[mjcf_loader] no renderable bodies in " << path << '\n';
        return false;
    }

    // Build children_map for traversal order
    std::unordered_map<std::string, std::vector<const URDFJoint*>> children_map;
    for (auto& j : flat.joints) children_map[j.parent].push_back(&j);

    // Z-up → Y-up correction (same as urdf_loader)
    Mat4 z_to_y = Mat4::identity();
    z_to_y.m[0][0] =  1; z_to_y.m[0][1] =  0; z_to_y.m[0][2] =  0;
    z_to_y.m[1][0] =  0; z_to_y.m[1][1] =  0; z_to_y.m[1][2] =  1;
    z_to_y.m[2][0] =  0; z_to_y.m[2][1] = -1; z_to_y.m[2][2] =  0;

    // Lookup: geom_link name → material idx (for triangle emission)
    std::unordered_map<std::string, int> geom_link_to_mat;
    for (auto& gm : flat.geom_materials) geom_link_to_mat[gm.geom_link_name] = gm.gpu_mat_idx;

    // Walk kinematic chain in the order articulation would traverse. Emit triangles
    // with per-geom materials. Consistent ordering matches mjcf_articulation_open().
    std::function<void(const std::string&, Mat4)> build_link;
    build_link = [&](const std::string& link_name, Mat4 world_xform) {
        auto lit = flat.links.find(link_name);
        if (lit == flat.links.end()) return;
        const URDFLink& link = lit->second;

        if (!link.visual_mesh_path.empty() && fs::exists(link.visual_mesh_path)) {
            RawMesh raw;
            if (load_link_mesh(link.visual_mesh_path, raw) && !raw.vertices.empty()) {
                int obj_id = obj_counter++;
                int mat_idx = 0;
                auto mit = geom_link_to_mat.find(link_name);
                if (mit != geom_link_to_mat.end()) mat_idx = mit->second;

                const Mat4& local = link.visual_origin;
                const Mat4& world = world_xform;

                int num_tris = (int)raw.indices.size() / 3;
                float3 centroid = make_float3(0, 0, 0);
                int cnt = 0;

                for (int ti = 0; ti < num_tris; ++ti) {
                    int i0 = raw.indices[ti*3];
                    int i1 = raw.indices[ti*3+1];
                    int i2 = raw.indices[ti*3+2];

                    Triangle tri{};
                    tri.v0 = world.transform_point(local.transform_point(raw.vertices[i0]));
                    tri.v1 = world.transform_point(local.transform_point(raw.vertices[i1]));
                    tri.v2 = world.transform_point(local.transform_point(raw.vertices[i2]));

                    if (i0 < (int)raw.normals.size() && i1 < (int)raw.normals.size() &&
                        i2 < (int)raw.normals.size()) {
                        tri.n0 = world.transform_normal(local.transform_normal(raw.normals[i0]));
                        tri.n1 = world.transform_normal(local.transform_normal(raw.normals[i1]));
                        tri.n2 = world.transform_normal(local.transform_normal(raw.normals[i2]));
                    } else {
                        float3 e1 = tri.v1 - tri.v0;
                        float3 e2 = tri.v2 - tri.v0;
                        float3 fn = make_float3(e1.y*e2.z-e1.z*e2.y,
                                                e1.z*e2.x-e1.x*e2.z,
                                                e1.x*e2.y-e1.y*e2.x);
                        float len = sqrtf(fn.x*fn.x + fn.y*fn.y + fn.z*fn.z);
                        if (len > 1e-7f) { fn.x/=len; fn.y/=len; fn.z/=len; }
                        tri.n0 = tri.n1 = tri.n2 = fn;
                    }

                    tri.uv0 = tri.uv1 = tri.uv2 = make_float2(0, 0);
                    tri.t0 = tri.t1 = tri.t2 = make_float4(0, 0, 0, 0);
                    tri.mat_idx = mat_idx;
                    tri.obj_id  = obj_id;
                    triangles_out.push_back(tri);

                    centroid.x += tri.v0.x + tri.v1.x + tri.v2.x;
                    centroid.y += tri.v0.y + tri.v1.y + tri.v2.y;
                    centroid.z += tri.v0.z + tri.v1.z + tri.v2.z;
                    cnt += 3;
                }

                MeshObject obj{};
                obj.obj_id = obj_id;
                obj.hidden = false;
                if (cnt > 0) {
                    float inv = 1.0f / (float)cnt;
                    obj.centroid = make_float3(centroid.x * inv, centroid.y * inv, centroid.z * inv);
                }
                snprintf(obj.name, sizeof(obj.name), "%s", link_name.c_str());
                objects_out.push_back(obj);
            }
        }

        auto jit = children_map.find(link_name);
        if (jit != children_map.end()) {
            for (auto* jp : jit->second) {
                Mat4 child_xform = world_xform * jp->origin;   // no joint angle at load (angle=0)
                build_link(jp->child, child_xform);
            }
        }
    };

    build_link(flat.root_link, z_to_y);

    std::cerr << "[mjcf_loader] total: " << triangles_out.size()
              << " triangles, " << objects_out.size() << " objects, "
              << materials_out.size() << " materials\n";

    return !triangles_out.empty();
}

// ─────────────────────────────────────────────
//  Public API — mjcf_articulation_open
// ─────────────────────────────────────────────

UrdfArticulation* mjcf_articulation_open(const std::string& path,
                                          const std::vector<Triangle>& initial_tris)
{
    mjcf_unique_id_counter = 0;

    MjcfScene scene;
    if (!parse_mjcf(path, scene)) return nullptr;

    // We need to replicate the same flattening as mjcf_load() to produce a
    // consistent URDF structure. (The triangle emission order in mjcf_load
    // walks the kinematic chain, so link_mesh loading during articulation
    // finalize follows the same order → mesh_caches align with triangle ids.)
    auto* h = new UrdfArticulation;
    MjcfFlatten flat;
    flat.root_link = "__world__";
    URDFLink world_link;
    world_link.name = flat.root_link;
    world_link.visual_origin = Mat4::identity();
    flat.links[flat.root_link] = world_link;

    // Materials aren't needed here — articulation reposes existing triangles in-place,
    // preserving their mat_idx. We just pass a no-op callback.
    std::vector<GpuMaterial> unused_materials;
    auto noop_mat = [](const std::string&, float4, bool) -> int { return 0; };

    int obj_counter = 0;
    for (auto& top : scene.worldbody.children) {
        flatten_body(top, flat.root_link, scene, flat, obj_counter,
                     unused_materials, noop_mat);
    }

    // Populate UrdfArticulation's links, joints, root_link — finalize does the rest.
    h->links      = flat.links;
    h->joints     = flat.joints;
    h->root_link  = flat.root_link;

    urdf_articulation_finalize(h, initial_tris);
    return h;
}
