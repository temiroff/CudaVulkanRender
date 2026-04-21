// ─────────────────────────────────────────────
//  urdf_loader.cpp
//  Loads .urdf robot description files and converts
//  all visual geometry into flat GPU-ready data
//  (same output format as gltf_loader / usd_loader).
//
//  Supported mesh formats:
//    .stl  — binary STL (direct parsing)
//    .dae  — COLLADA (XML geometry extraction)
//
//  Kinematic chain transforms are baked into world space.
// ─────────────────────────────────────────────

#include "urdf_loader.h"
#include "urdf_internal.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <cstring>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <set>

namespace fs = std::filesystem;

static Mat4 mat4_mul_rowmajor(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += a.m[i][k] * b.m[k][j];
            r.m[i][j] = s;
        }
    return r;
}

// ─────────────────────────────────────────────
//  Minimal XML parser (read-only, no validation)
//  Just enough to parse URDF, MJCF, and COLLADA files.
//  XmlAttr/XmlNode are defined in urdf_internal.h.
// ─────────────────────────────────────────────

static void skip_ws(const char*& p) { while (*p && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p; }

static std::string parse_name(const char*& p) {
    const char* s = p;
    while (*p && *p!='>' && *p!='/' && *p!='=' && *p!=' ' && *p!='\t' && *p!='\n' && *p!='\r') ++p;
    return std::string(s, p);
}

static std::string parse_attr_value(const char*& p) {
    char q = *p++;
    const char* s = p;
    while (*p && *p != q) ++p;
    std::string val(s, p);
    if (*p) ++p;
    return val;
}

static bool parse_node(const char*& p, XmlNode& node) {
    skip_ws(p);
    if (*p != '<') return false;
    ++p;

    // Skip processing instructions and comments
    if (*p == '?') { while (*p && !(*p=='?' && *(p+1)=='>')) ++p; if(*p) p+=2; return false; }
    if (*p == '!' && *(p+1)=='-' && *(p+2)=='-') {
        p += 3;
        while (*p && !(*p=='-' && *(p+1)=='-' && *(p+2)=='>')) ++p;
        if (*p) p += 3;
        return false;
    }
    if (*p == '!') { while (*p && *p!='>') ++p; if(*p) ++p; return false; }

    node.tag = parse_name(p);

    // Attributes
    while (true) {
        skip_ws(p);
        if (*p == '/' && *(p+1) == '>') { p += 2; return true; }
        if (*p == '>') { ++p; break; }
        if (!*p) return false;
        XmlAttr a;
        a.name = parse_name(p);
        skip_ws(p);
        if (*p == '=') { ++p; skip_ws(p); a.value = parse_attr_value(p); }
        node.attrs.push_back(a);
    }

    // Children and text
    while (*p) {
        skip_ws(p);
        if (*p == '<' && *(p+1) == '/') {
            p += 2;
            while (*p && *p != '>') ++p;
            if (*p) ++p;
            return true;
        }
        if (*p == '<') {
            XmlNode child;
            if (parse_node(p, child))
                node.children.push_back(std::move(child));
        } else {
            const char* s = p;
            while (*p && *p != '<') ++p;
            node.text += std::string(s, p);
        }
    }
    return true;
}

XmlNode parse_xml(const std::string& xml_str) {
    const char* p = xml_str.c_str();
    XmlNode root;
    while (*p) {
        skip_ws(p);
        if (!*p) break;
        XmlNode node;
        if (parse_node(p, node) && !node.tag.empty())
            root.children.push_back(std::move(node));
    }
    if (root.children.size() == 1)
        return std::move(root.children[0]);
    root.tag = "_root";
    return root;
}

std::string read_file_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ─────────────────────────────────────────────
//  Math helpers — Mat4 is defined in urdf_internal.h
// ─────────────────────────────────────────────

static Mat4 parse_origin(const XmlNode* origin) {
    if (!origin) return Mat4::identity();
    float x=0,y=0,z=0,roll=0,pitch=0,yaw=0;
    std::string xyz_str = origin->attr("xyz", "0 0 0");
    std::string rpy_str = origin->attr("rpy", "0 0 0");
    sscanf(xyz_str.c_str(), "%f %f %f", &x, &y, &z);
    sscanf(rpy_str.c_str(), "%f %f %f", &roll, &pitch, &yaw);
    return Mat4::from_rpy_xyz(roll, pitch, yaw, x, y, z);
}

// ─────────────────────────────────────────────
//  Mesh path resolution
// ─────────────────────────────────────────────

static std::string resolve_mesh_path(const std::string& filename,
                                      const std::string& urdf_dir) {
    if (filename.substr(0, 10) == "package://") {
        // package://franka_description/meshes/visual/link0.dae
        auto rest = filename.substr(10);
        auto slash = rest.find('/');
        std::string pkg = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
        std::string rel = (slash != std::string::npos) ? rest.substr(slash + 1) : "";

        // Walk up from urdf_dir to find package root
        fs::path search(urdf_dir);
        for (int i = 0; i < 10; ++i) {
            fs::path candidate = search / pkg / rel;
            if (fs::exists(candidate)) return candidate.string();
            if (search.filename().string() == pkg) {
                candidate = search / rel;
                if (fs::exists(candidate)) return candidate.string();
            }
            search = search.parent_path();
        }
        return (fs::path(urdf_dir) / rel).string();
    }
    return (fs::path(urdf_dir) / filename).string();
}

// ─────────────────────────────────────────────
//  STL binary loader
// ─────────────────────────────────────────────

// DaeMaterial and RawMesh are defined in urdf_internal.h

bool load_stl(const std::string& path, RawMesh& out) {
    // Slurp the whole file in one syscall. iostream doing 50-byte reads per
    // triangle is catastrophically slow on Windows for meshes with 20k+ tris.
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize sz = f.tellg();
    if (sz < 84) return false;
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf((size_t)sz);
    if (!f.read(reinterpret_cast<char*>(buf.data()), sz)) return false;

    uint32_t num_tris = 0;
    std::memcpy(&num_tris, buf.data() + 80, 4);
    if (num_tris == 0 || num_tris > 10000000) return false;
    if ((size_t)sz < 84 + (size_t)num_tris * 50) return false;

    out.vertices.resize(num_tris * 3);
    out.normals.resize(num_tris * 3);
    out.indices.resize(num_tris * 3);

    const uint8_t* p = buf.data() + 84;
    for (uint32_t i = 0; i < num_tris; ++i, p += 50) {
        float data[12];
        std::memcpy(data, p, 48);
        float3 fn = make_float3(data[0], data[1], data[2]);
        int base = (int)i * 3;
        out.vertices[base + 0] = make_float3(data[3], data[4],  data[5]);
        out.vertices[base + 1] = make_float3(data[6], data[7],  data[8]);
        out.vertices[base + 2] = make_float3(data[9], data[10], data[11]);
        out.normals[base + 0] = fn;
        out.normals[base + 1] = fn;
        out.normals[base + 2] = fn;
        out.indices[base + 0] = base;
        out.indices[base + 1] = base + 1;
        out.indices[base + 2] = base + 2;
    }
    return true;
}

// ─────────────────────────────────────────────
//  COLLADA (.dae) geometry loader
//  Extracts <mesh> geometry from the first
//  <geometry> element. Handles <triangles> and
//  <polylist> with triangulated quads.
// ─────────────────────────────────────────────

static std::vector<float> parse_float_array(const std::string& text) {
    std::vector<float> out;
    std::istringstream ss(text);
    float v;
    while (ss >> v) out.push_back(v);
    return out;
}

static std::vector<int> parse_int_array(const std::string& text) {
    std::vector<int> out;
    std::istringstream ss(text);
    int v;
    while (ss >> v) out.push_back(v);
    return out;
}

// Recursively find a node with given tag
static const XmlNode* find_node(const XmlNode& root, const std::string& tag) {
    if (root.tag == tag) return &root;
    for (auto& c : root.children) {
        auto* found = find_node(c, tag);
        if (found) return found;
    }
    return nullptr;
}

// Find all nodes with given tag
static void find_all_nodes(const XmlNode& root, const std::string& tag,
                            std::vector<const XmlNode*>& results) {
    if (root.tag == tag) results.push_back(&root);
    for (auto& c : root.children)
        find_all_nodes(c, tag, results);
}

// Parse a COLLADA <matrix> (4x4, column-major in the text) into our row-major Mat4.
static Mat4 parse_collada_matrix(const std::string& text) {
    auto vals = parse_float_array(text);
    if (vals.size() < 16) return Mat4::identity();
    // COLLADA stores row-major in the text (row0 col0..3, row1 col0..3, ...)
    Mat4 m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            m.m[r][c] = vals[r * 4 + c];
    return m;
}

static float4 parse_color4(const std::string& text) {
    auto v = parse_float_array(text);
    if (v.size() >= 4) return make_float4(v[0], v[1], v[2], v[3]);
    if (v.size() >= 3) return make_float4(v[0], v[1], v[2], 1.0f);
    return make_float4(0.7f, 0.7f, 0.7f, 1.0f);
}

bool load_dae(const std::string& path, RawMesh& out) {
    std::string xml_str = read_file_text(path);
    if (xml_str.empty()) return false;

    XmlNode doc = parse_xml(xml_str);

    // ── 1. Parse effects: effect_id → DaeMaterial ──
    std::unordered_map<std::string, DaeMaterial> effects;
    {
        std::vector<const XmlNode*> effect_nodes;
        find_all_nodes(doc, "effect", effect_nodes);
        for (auto* eff : effect_nodes) {
            std::string eid = eff->attr("id");
            if (eid.empty()) continue;
            DaeMaterial dm;
            // Find <phong>, <lambert>, or <blinn> technique
            const XmlNode* tech = nullptr;
            std::vector<const XmlNode*> phongs;
            find_all_nodes(*eff, "phong", phongs);
            if (!phongs.empty()) tech = phongs[0];
            if (!tech) { find_all_nodes(*eff, "lambert", phongs); if (!phongs.empty()) tech = phongs[0]; }
            if (!tech) { find_all_nodes(*eff, "blinn", phongs);   if (!phongs.empty()) tech = phongs[0]; }
            if (tech) {
                auto* diff = tech->child("diffuse");
                if (diff) {
                    auto* col = diff->child("color");
                    if (col) dm.diffuse = parse_color4(col->text);
                }
                auto* spec = tech->child("specular");
                if (spec) {
                    auto* col = spec->child("color");
                    if (col) {
                        auto sv = parse_float_array(col->text);
                        if (sv.size() >= 3) dm.specular_intensity = (sv[0] + sv[1] + sv[2]) / 3.0f;
                    }
                }
                auto* shin = tech->child("shininess");
                if (shin) {
                    auto* fv = shin->child("float");
                    if (fv) dm.shininess = std::atof(fv->text.c_str());
                }
            }
            effects[eid] = dm;
        }
    }

    // ── 2. Parse materials: material_id → effect_id ──
    std::unordered_map<std::string, std::string> mat_to_effect;
    {
        std::vector<const XmlNode*> mat_nodes;
        find_all_nodes(doc, "material", mat_nodes);
        for (auto* mn : mat_nodes) {
            std::string mid = mn->attr("id");
            auto* inst = mn->child("instance_effect");
            if (inst) {
                std::string url = inst->attr("url");
                if (!url.empty() && url[0] == '#') url = url.substr(1);
                mat_to_effect[mid] = url;
            }
        }
    }

    // Resolve material_id → DaeMaterial
    auto resolve_material = [&](const std::string& mat_id) -> DaeMaterial {
        auto it = mat_to_effect.find(mat_id);
        if (it != mat_to_effect.end()) {
            auto eit = effects.find(it->second);
            if (eit != effects.end()) return eit->second;
        }
        return DaeMaterial{};
    };

    // ── 3. Parse sources: source_id → float_array ──
    std::unordered_map<std::string, std::vector<float>> sources;
    {
        std::vector<const XmlNode*> source_nodes;
        find_all_nodes(doc, "source", source_nodes);
        for (auto* src : source_nodes) {
            std::string id = src->attr("id");
            auto* fa = src->child("float_array");
            if (fa && !id.empty())
                sources[id] = parse_float_array(fa->text);
        }
    }

    // ── 4. Index geometry by id ──
    struct GeomData {
        const XmlNode* mesh_node;
        std::string    pos_source_id;
    };
    std::unordered_map<std::string, GeomData> geometries;
    {
        std::vector<const XmlNode*> geom_nodes;
        find_all_nodes(doc, "geometry", geom_nodes);
        for (auto* geom : geom_nodes) {
            std::string gid = geom->attr("id");
            auto* mesh_node = geom->child("mesh");
            if (!mesh_node) continue;
            std::string pos_source_id;
            auto* vertices_node = mesh_node->child("vertices");
            if (vertices_node) {
                for (auto& inp : vertices_node->children) {
                    if (inp.tag == "input" && inp.attr("semantic") == "POSITION") {
                        pos_source_id = inp.attr("source");
                        if (!pos_source_id.empty() && pos_source_id[0] == '#')
                            pos_source_id = pos_source_id.substr(1);
                    }
                }
            }
            geometries[gid] = { mesh_node, pos_source_id };
        }
    }

    // ── 5. Helper: emit triangles from a <triangles>/<polylist> element ──
    auto process_prim = [&](const XmlNode* prim_node, const std::string& pos_source_id,
                             int dae_mat_idx, const Mat4& node_xform) {
        int prim_count = std::atoi(prim_node->attr("count", "0").c_str());
        if (prim_count == 0) return;

        int max_offset = 0;
        int vertex_offset = -1, normal_offset = -1;
        std::string normal_source_id;

        for (auto& inp : prim_node->children) {
            if (inp.tag != "input") continue;
            int off = std::atoi(inp.attr("offset", "0").c_str());
            max_offset = std::max(max_offset, off);
            std::string semantic = inp.attr("semantic");
            if (semantic == "VERTEX") vertex_offset = off;
            else if (semantic == "NORMAL") {
                normal_offset = off;
                normal_source_id = inp.attr("source");
                if (!normal_source_id.empty() && normal_source_id[0] == '#')
                    normal_source_id = normal_source_id.substr(1);
            }
        }
        int stride = max_offset + 1;

        auto* p_node = prim_node->child("p");
        if (!p_node) return;
        auto p_indices = parse_int_array(p_node->text);

        const std::vector<float>* pos_data = nullptr;
        if (sources.count(pos_source_id)) pos_data = &sources[pos_source_id];
        if (!pos_data) return;

        const std::vector<float>* norm_data = nullptr;
        if (!normal_source_id.empty() && sources.count(normal_source_id))
            norm_data = &sources[normal_source_id];

        std::vector<int> vcounts;
        if (prim_node->tag == "polylist") {
            auto* vc = prim_node->child("vcount");
            if (vc) vcounts = parse_int_array(vc->text);
        }

        int idx_pos = 0;
        int num_polys = (prim_node->tag == "triangles") ? prim_count : (int)vcounts.size();

        for (int poly = 0; poly < num_polys; ++poly) {
            int nverts = (prim_node->tag == "triangles") ? 3 :
                         (poly < (int)vcounts.size() ? vcounts[poly] : 3);

            std::vector<float3> face_verts, face_norms;
            for (int vi = 0; vi < nverts; ++vi) {
                int base_idx = idx_pos + vi * stride;
                if (base_idx + stride > (int)p_indices.size()) break;

                if (vertex_offset >= 0) {
                    int pi = p_indices[base_idx + vertex_offset];
                    if (pi * 3 + 2 < (int)pos_data->size()) {
                        float3 p = make_float3((*pos_data)[pi*3], (*pos_data)[pi*3+1], (*pos_data)[pi*3+2]);
                        face_verts.push_back(node_xform.transform_point(p));
                    }
                }
                if (normal_offset >= 0 && norm_data) {
                    int ni = p_indices[base_idx + normal_offset];
                    if (ni * 3 + 2 < (int)norm_data->size()) {
                        float3 n = make_float3((*norm_data)[ni*3], (*norm_data)[ni*3+1], (*norm_data)[ni*3+2]);
                        face_norms.push_back(node_xform.transform_normal(n));
                    }
                }
            }
            idx_pos += nverts * stride;

            for (int vi = 1; vi + 1 < (int)face_verts.size(); ++vi) {
                int base = (int)out.vertices.size();
                out.vertices.push_back(face_verts[0]);
                out.vertices.push_back(face_verts[vi]);
                out.vertices.push_back(face_verts[vi + 1]);

                if ((int)face_norms.size() > vi + 1) {
                    out.normals.push_back(face_norms[0]);
                    out.normals.push_back(face_norms[vi]);
                    out.normals.push_back(face_norms[vi + 1]);
                } else {
                    float3 e1 = face_verts[vi] - face_verts[0];
                    float3 e2 = face_verts[vi+1] - face_verts[0];
                    float3 fn = make_float3(e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z, e1.x*e2.y-e1.y*e2.x);
                    float len = sqrtf(fn.x*fn.x + fn.y*fn.y + fn.z*fn.z);
                    if (len > 1e-7f) { fn.x/=len; fn.y/=len; fn.z/=len; }
                    out.normals.push_back(fn); out.normals.push_back(fn); out.normals.push_back(fn);
                }

                out.indices.push_back(base);
                out.indices.push_back(base + 1);
                out.indices.push_back(base + 2);
                out.mat_ids.push_back(dae_mat_idx);
            }
        }
    };

    // ── 6. Walk <visual_scene> → <node> elements with transforms ──
    // Each <node> may have a <matrix>, then <instance_geometry> referencing a geometry + material.
    std::vector<const XmlNode*> scene_nodes;
    find_all_nodes(doc, "visual_scene", scene_nodes);

    // Helper: process a single <node>
    std::function<void(const XmlNode*, Mat4)> walk_node;
    walk_node = [&](const XmlNode* node, Mat4 parent_xform) {
        Mat4 local = Mat4::identity();
        auto* mat_node = node->child("matrix");
        if (mat_node) local = parse_collada_matrix(mat_node->text);
        Mat4 world = parent_xform * local;

        // Process <instance_geometry> children
        for (auto& child : node->children) {
            if (child.tag == "instance_geometry") {
                std::string url = child.attr("url");
                if (!url.empty() && url[0] == '#') url = url.substr(1);

                auto git = geometries.find(url);
                if (git == geometries.end()) continue;

                // Resolve material binding: <bind_material>/<technique_common>/<instance_material>
                // symbol → target mapping
                std::unordered_map<std::string, std::string> mat_bindings;
                auto* bind = child.child("bind_material");
                if (bind) {
                    auto* tech = bind->child("technique_common");
                    if (tech) {
                        for (auto& im : tech->children) {
                            if (im.tag == "instance_material") {
                                std::string sym = im.attr("symbol");
                                std::string tgt = im.attr("target");
                                if (!tgt.empty() && tgt[0] == '#') tgt = tgt.substr(1);
                                mat_bindings[sym] = tgt;
                            }
                        }
                    }
                }

                auto* mesh_node = git->second.mesh_node;
                const std::string& pos_src = git->second.pos_source_id;

                // Process each <triangles>/<polylist> — they have a "material" attr (symbol)
                auto emit = [&](const XmlNode* prim) {
                    std::string sym = prim->attr("material");
                    std::string mat_id;
                    auto bit = mat_bindings.find(sym);
                    if (bit != mat_bindings.end()) mat_id = bit->second;

                    DaeMaterial dm = resolve_material(mat_id);
                    // Find or add material
                    int dae_mat_idx = (int)out.dae_materials.size();
                    out.dae_materials.push_back(dm);

                    process_prim(prim, pos_src, dae_mat_idx, world);
                };

                for (auto* tri : mesh_node->children_with_tag("triangles")) emit(tri);
                for (auto* poly : mesh_node->children_with_tag("polylist")) emit(poly);
            }
        }

        // Recurse into child <node>s
        for (auto& child : node->children) {
            if (child.tag == "node") walk_node(&child, world);
        }
    };

    if (!scene_nodes.empty()) {
        for (auto* vs : scene_nodes) {
            for (auto& child : vs->children) {
                if (child.tag == "node") walk_node(&child, Mat4::identity());
            }
        }
    }

    // Fallback: if no visual_scene found, load geometry directly (no transforms)
    if (out.vertices.empty()) {
        DaeMaterial fallback_dm;
        int dae_mat_idx = (int)out.dae_materials.size();
        out.dae_materials.push_back(fallback_dm);

        for (auto& [gid, gd] : geometries) {
            auto* mesh_node = gd.mesh_node;
            for (auto* tri : mesh_node->children_with_tag("triangles"))
                process_prim(tri, gd.pos_source_id, dae_mat_idx, Mat4::identity());
            for (auto* poly : mesh_node->children_with_tag("polylist"))
                process_prim(poly, gd.pos_source_id, dae_mat_idx, Mat4::identity());
        }
    }

    return !out.vertices.empty();
}

// ─────────────────────────────────────────────
//  URDF data structures — defined in urdf_internal.h
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
//  URDF parser
// ─────────────────────────────────────────────

static bool parse_urdf(const std::string& urdf_path,
                        std::unordered_map<std::string, URDFLink>& links,
                        std::vector<URDFJoint>& joints,
                        std::string& root_link_name)
{
    std::string xml_str = read_file_text(urdf_path);
    if (xml_str.empty()) {
        std::cerr << "[urdf_loader] cannot read: " << urdf_path << '\n';
        return false;
    }

    XmlNode doc = parse_xml(xml_str);
    if (doc.tag != "robot") {
        // Might be wrapped
        auto* robot = doc.child("robot");
        if (!robot) robot = find_node(doc, "robot");
        if (!robot) {
            std::cerr << "[urdf_loader] no <robot> element found\n";
            return false;
        }
        doc = *robot;
    }

    std::string urdf_dir = fs::path(urdf_path).parent_path().string();
    std::set<std::string> child_links;

    // Parse <link> elements
    for (auto& link_node : doc.children_with_tag("link")) {
        URDFLink link;
        link.name = link_node->attr("name");
        link.visual_origin = Mat4::identity();

        auto* visual = link_node->child("visual");
        if (visual) {
            auto* origin = visual->child("origin");
            link.visual_origin = parse_origin(origin);

            auto* geom = visual->child("geometry");
            if (geom) {
                auto* mesh_elem = geom->child("mesh");
                if (mesh_elem) {
                    link.visual_mesh_path = resolve_mesh_path(
                        mesh_elem->attr("filename"), urdf_dir);
                }
            }
        }

        links[link.name] = link;
    }

    // Parse <joint> elements
    for (auto& joint_node : doc.children_with_tag("joint")) {
        URDFJoint j;
        j.name   = joint_node->attr("name");
        j.type   = joint_node->attr("type");
        j.origin = parse_origin(joint_node->child("origin"));

        auto* parent = joint_node->child("parent");
        auto* child  = joint_node->child("child");
        if (parent) j.parent = parent->attr("link");
        if (child)  j.child  = child->attr("link");

        auto* axis_node = joint_node->child("axis");
        if (axis_node) {
            float ax=0, ay=0, az=1;
            sscanf(axis_node->attr("xyz", "0 0 1").c_str(), "%f %f %f", &ax, &ay, &az);
            j.axis = make_float3(ax, ay, az);
        }

        auto* limit_node = joint_node->child("limit");
        if (limit_node) {
            j.lower = (float)std::atof(limit_node->attr("lower", "0").c_str());
            j.upper = (float)std::atof(limit_node->attr("upper", "0").c_str());
        }

        joints.push_back(j);
        child_links.insert(j.child);
    }

    // Find root link (not a child of any joint)
    for (auto& [name, _] : links) {
        if (child_links.find(name) == child_links.end()) {
            root_link_name = name;
            break;
        }
    }

    std::cerr << "[urdf_loader] links=" << links.size()
              << " joints=" << joints.size()
              << " root=" << root_link_name << '\n';
    return true;
}

// ─────────────────────────────────────────────
//  Build Triangle/Material output
// ─────────────────────────────────────────────

bool load_link_mesh(const std::string& mesh_path, RawMesh& raw) {
    if (mesh_path.empty() || !fs::exists(mesh_path)) return false;

    // In-memory cache keyed by canonical path + file size. Same mesh is loaded
    // multiple times: once by mjcf_load (builds triangles), again by
    // urdf_articulation_finalize (builds mesh_caches for repose). Caching makes
    // the second and third calls effectively free.
    struct CacheKey { std::string path; uintmax_t size; };
    static std::unordered_map<std::string, RawMesh> s_cache;
    std::string canonical;
    try { canonical = fs::weakly_canonical(mesh_path).string(); }
    catch (...) { canonical = mesh_path; }
    uintmax_t fsz = 0;
    std::error_code ec;
    fsz = fs::file_size(mesh_path, ec);
    std::string key = canonical + "|" + std::to_string(fsz);

    auto it = s_cache.find(key);
    if (it != s_cache.end()) { raw = it->second; return true; }

    auto ext = fs::path(mesh_path).extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);

    bool ok = false;
    if      (ext == ".stl") ok = load_stl(mesh_path, raw);
    else if (ext == ".dae") ok = load_dae(mesh_path, raw);
    else if (ext == ".obj") ok = load_obj(mesh_path, raw);
    else { std::cerr << "[urdf_loader] unsupported mesh format: " << ext << '\n'; return false; }

    if (ok) s_cache[key] = raw;
    return ok;
}

// ─────────────────────────────────────────────
//  OBJ loader (MJCF meshes)
//  Minimal Wavefront OBJ parser — v/vn/f only.
//  Triangulates polygons as fans. Ignores textures,
//  groups, materials (MJCF assigns materials externally
//  per-geom, not per-face).
// ─────────────────────────────────────────────

bool load_obj(const std::string& path, RawMesh& out)
{
    std::ifstream f(path);
    if (!f) return false;

    std::vector<float3> positions;
    std::vector<float3> normals;
    positions.reserve(1024);
    normals.reserve(1024);

    // Deduplicate (v,vn) pairs so we can emit unique vertices.
    struct PairKey { int v, n; };
    auto make_key = [](int v, int n) -> long long {
        return ((long long)(uint32_t)v << 32) | (uint32_t)n;
    };
    std::unordered_map<long long, int> key_to_vidx;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const char* p = line.c_str();
        while (*p == ' ' || *p == '\t') ++p;
        if (p[0] == '#' || p[0] == '\0') continue;

        if (p[0] == 'v' && (p[1] == ' ' || p[1] == '\t')) {
            float x, y, z;
            if (sscanf(p + 1, "%f %f %f", &x, &y, &z) == 3)
                positions.push_back(make_float3(x, y, z));
        } else if (p[0] == 'v' && p[1] == 'n') {
            float x, y, z;
            if (sscanf(p + 2, "%f %f %f", &x, &y, &z) == 3)
                normals.push_back(make_float3(x, y, z));
        } else if (p[0] == 'f' && (p[1] == ' ' || p[1] == '\t')) {
            // Parse face tokens like "v/vt/vn" or "v//vn" or "v"
            std::vector<int> face_v;
            std::vector<int> face_n;
            const char* q = p + 1;
            while (*q) {
                while (*q == ' ' || *q == '\t') ++q;
                if (!*q || *q == '\n' || *q == '\r') break;

                int vi = 0, ti = 0, ni = 0;
                // parse vertex index
                bool neg = false;
                if (*q == '-') { neg = true; ++q; }
                while (*q >= '0' && *q <= '9') { vi = vi*10 + (*q - '0'); ++q; }
                if (neg) vi = -vi;

                if (*q == '/') {
                    ++q;
                    // texture index (skip)
                    bool tneg = false;
                    if (*q == '-') { tneg = true; ++q; }
                    while (*q >= '0' && *q <= '9') { ti = ti*10 + (*q - '0'); ++q; }
                    if (tneg) ti = -ti;

                    if (*q == '/') {
                        ++q;
                        bool nneg = false;
                        if (*q == '-') { nneg = true; ++q; }
                        while (*q >= '0' && *q <= '9') { ni = ni*10 + (*q - '0'); ++q; }
                        if (nneg) ni = -ni;
                    }
                }

                // OBJ indices are 1-based; negative = relative-from-end
                int v_idx = (vi > 0) ? vi - 1 : (int)positions.size() + vi;
                int n_idx = (ni > 0) ? ni - 1 : (ni < 0 ? (int)normals.size() + ni : -1);

                face_v.push_back(v_idx);
                face_n.push_back(n_idx);

                // skip any remaining chars to next whitespace
                while (*q && *q != ' ' && *q != '\t' && *q != '\n' && *q != '\r') ++q;
            }

            if (face_v.size() < 3) continue;

            // Fan triangulation
            auto emit_vertex = [&](int v, int n) -> int {
                long long k = make_key(v, n);
                auto it = key_to_vidx.find(k);
                if (it != key_to_vidx.end()) return it->second;

                if (v < 0 || v >= (int)positions.size()) return -1;
                float3 pos = positions[v];
                float3 nrm;
                if (n >= 0 && n < (int)normals.size()) nrm = normals[n];
                else nrm = make_float3(0, 0, 0);    // will be replaced by face normal

                int idx = (int)out.vertices.size();
                out.vertices.push_back(pos);
                out.normals.push_back(nrm);
                key_to_vidx[k] = idx;
                return idx;
            };

            int i0 = emit_vertex(face_v[0], face_n[0]);
            if (i0 < 0) continue;
            for (size_t k = 1; k + 1 < face_v.size(); ++k) {
                int i1 = emit_vertex(face_v[k],   face_n[k]);
                int i2 = emit_vertex(face_v[k+1], face_n[k+1]);
                if (i1 < 0 || i2 < 0) continue;

                // If any vertex lacks a normal, synthesize one from the face.
                bool missing_normal =
                    (face_n[0]   < 0) ||
                    (face_n[k]   < 0) ||
                    (face_n[k+1] < 0);
                if (missing_normal) {
                    float3 v0 = out.vertices[i0];
                    float3 v1 = out.vertices[i1];
                    float3 v2 = out.vertices[i2];
                    float3 e1 = make_float3(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z);
                    float3 e2 = make_float3(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z);
                    float3 fn = make_float3(e1.y*e2.z - e1.z*e2.y,
                                            e1.z*e2.x - e1.x*e2.z,
                                            e1.x*e2.y - e1.y*e2.x);
                    float len = sqrtf(fn.x*fn.x + fn.y*fn.y + fn.z*fn.z);
                    if (len > 1e-7f) { fn.x/=len; fn.y/=len; fn.z/=len; }
                    if (face_n[0]   < 0) out.normals[i0] = fn;
                    if (face_n[k]   < 0) out.normals[i1] = fn;
                    if (face_n[k+1] < 0) out.normals[i2] = fn;
                }

                out.indices.push_back(i0);
                out.indices.push_back(i1);
                out.indices.push_back(i2);
            }
        }
    }

    return !out.vertices.empty();
}

static float3 compute_centroid(const std::vector<float3>& verts) {
    float3 sum = make_float3(0, 0, 0);
    for (auto& v : verts) { sum.x += v.x; sum.y += v.y; sum.z += v.z; }
    float n = (float)verts.size();
    if (n > 0) { sum.x /= n; sum.y /= n; sum.z /= n; }
    return sum;
}

bool urdf_load(const std::string&          path,
               std::vector<Triangle>&      triangles_out,
               std::vector<GpuMaterial>&   materials_out,
               std::vector<TextureImage>&  textures_out,
               std::vector<MeshObject>&    objects_out)
{
    std::unordered_map<std::string, URDFLink> links;
    std::vector<URDFJoint> joints;
    std::string root_link;

    if (!parse_urdf(path, links, joints, root_link))
        return false;

    // Build parent→children map
    std::unordered_map<std::string, std::vector<const URDFJoint*>> children_map;
    for (auto& j : joints)
        children_map[j.parent].push_back(&j);

    // URDF is Z-up; the renderer is Y-up.  Apply the same correction as usd_loader:
    //   (x,y,z) → (x, z, -y)   — 90° rotation around X
    // This is applied AFTER all URDF-space transforms (outermost).
    Mat4 z_to_y;
    z_to_y.m[0][0] =  1; z_to_y.m[0][1] =  0; z_to_y.m[0][2] =  0; z_to_y.m[0][3] = 0;
    z_to_y.m[1][0] =  0; z_to_y.m[1][1] =  0; z_to_y.m[1][2] =  1; z_to_y.m[1][3] = 0;
    z_to_y.m[2][0] =  0; z_to_y.m[2][1] = -1; z_to_y.m[2][2] =  0; z_to_y.m[2][3] = 0;
    z_to_y.m[3][0] =  0; z_to_y.m[3][1] =  0; z_to_y.m[3][2] =  0; z_to_y.m[3][3] = 1;

    int obj_counter = 0;
    // Base material index — new materials added per .dae file
    int mat_base = (int)materials_out.size();

    // Recursive traversal of kinematic chain
    std::function<void(const std::string&, Mat4)> build_link;
    build_link = [&](const std::string& link_name, Mat4 world_xform) {
        auto it = links.find(link_name);
        if (it == links.end()) return;

        const URDFLink& link = it->second;

        // Two-stage transform: visual_origin operates in link-local space,
        // then world_xform (z_to_y * joint chain) brings to renderer world space.
        // Applied separately to avoid matrix composition precision issues.
        const Mat4& local_xform = link.visual_origin;
        const Mat4& world = world_xform;

        // Load mesh
        RawMesh raw;
        if (load_link_mesh(link.visual_mesh_path, raw) && !raw.vertices.empty()) {
            int obj_id = obj_counter++;

            // Register materials from this mesh
            int link_mat_base = (int)materials_out.size();
            if (!raw.dae_materials.empty()) {
                for (auto& dm : raw.dae_materials) {
                    GpuMaterial gm{};
                    gm.base_color       = dm.diffuse;
                    // Convert Phong specular → PBR roughness/metallic approximation
                    gm.metallic         = dm.specular_intensity > 0.5f ? 0.4f : 0.1f;
                    gm.roughness        = dm.shininess > 0.0f ? std::max(0.1f, 1.0f - dm.shininess / 512.0f) : 0.5f;
                    gm.emissive_factor  = make_float4(0, 0, 0, 0);
                    gm.base_color_tex   = -1;
                    gm.metallic_rough_tex = -1;
                    gm.normal_tex       = -1;
                    gm.emissive_tex     = -1;
                    gm.custom_shader    = 0;
                    gm.normal_y_flip    = 0;
                    materials_out.push_back(gm);
                }
            } else {
                // Fallback: default gray material
                GpuMaterial gm{};
                gm.base_color       = make_float4(0.72f, 0.72f, 0.74f, 1.0f);
                gm.metallic         = 0.2f;
                gm.roughness        = 0.5f;
                gm.emissive_factor  = make_float4(0, 0, 0, 0);
                gm.base_color_tex   = -1;
                gm.metallic_rough_tex = -1;
                gm.normal_tex       = -1;
                gm.emissive_tex     = -1;
                gm.custom_shader    = 0;
                gm.normal_y_flip    = 0;
                materials_out.push_back(gm);
            }

            // Transform vertices and normals to world space, emit triangles
            int num_tris = (int)raw.indices.size() / 3;
            for (int ti = 0; ti < num_tris; ++ti) {
                int i0 = raw.indices[ti*3];
                int i1 = raw.indices[ti*3+1];
                int i2 = raw.indices[ti*3+2];

                Triangle tri{};
                tri.v0 = world.transform_point(local_xform.transform_point(raw.vertices[i0]));
                tri.v1 = world.transform_point(local_xform.transform_point(raw.vertices[i1]));
                tri.v2 = world.transform_point(local_xform.transform_point(raw.vertices[i2]));

                if (i0 < (int)raw.normals.size() && i1 < (int)raw.normals.size() && i2 < (int)raw.normals.size()) {
                    tri.n0 = world.transform_normal(local_xform.transform_normal(raw.normals[i0]));
                    tri.n1 = world.transform_normal(local_xform.transform_normal(raw.normals[i1]));
                    tri.n2 = world.transform_normal(local_xform.transform_normal(raw.normals[i2]));
                } else {
                    float3 e1 = tri.v1 - tri.v0;
                    float3 e2 = tri.v2 - tri.v0;
                    float3 fn = make_float3(
                        e1.y*e2.z - e1.z*e2.y,
                        e1.z*e2.x - e1.x*e2.z,
                        e1.x*e2.y - e1.y*e2.x);
                    float len = sqrtf(fn.x*fn.x + fn.y*fn.y + fn.z*fn.z);
                    if (len > 1e-7f) { fn.x/=len; fn.y/=len; fn.z/=len; }
                    tri.n0 = tri.n1 = tri.n2 = fn;
                }

                tri.uv0 = tri.uv1 = tri.uv2 = make_float2(0, 0);
                tri.t0 = tri.t1 = tri.t2 = make_float4(0, 0, 0, 0);

                // Assign material from COLLADA per-triangle binding
                if (ti < (int)raw.mat_ids.size())
                    tri.mat_idx = link_mat_base + raw.mat_ids[ti];
                else
                    tri.mat_idx = link_mat_base;

                tri.obj_id  = obj_id;
                triangles_out.push_back(tri);
            }

            // Create MeshObject
            MeshObject obj{};
            obj.obj_id = obj_id;
            obj.hidden = false;
            obj.is_robot_part = true;

            // Compute centroid from transformed vertices
            float3 sum = make_float3(0, 0, 0);
            for (auto& v : raw.vertices) {
                float3 wv = world.transform_point(local_xform.transform_point(v));
                sum.x += wv.x; sum.y += wv.y; sum.z += wv.z;
            }
            float n = (float)raw.vertices.size();
            if (n > 0) { sum.x /= n; sum.y /= n; sum.z /= n; }
            obj.centroid = sum;

            snprintf(obj.name, sizeof(obj.name), "%s", link.name.c_str());
            objects_out.push_back(obj);


            std::cerr << "[urdf_loader]   + " << link.name
                      << ": " << num_tris << " triangles, "
                      << raw.dae_materials.size() << " materials\n";
        } else if (!link.visual_mesh_path.empty()) {
            std::cerr << "[urdf_loader]   - " << link.name
                      << ": mesh not loaded (" << link.visual_mesh_path << ")\n";
        }

        // Recurse into child links via joints
        auto jit = children_map.find(link_name);
        if (jit != children_map.end()) {
            for (auto* joint : jit->second) {
                Mat4 child_xform = world_xform * joint->origin;
                build_link(joint->child, child_xform);
            }
        }
    };

    build_link(root_link, z_to_y);

    std::cerr << "[urdf_loader] total: " << triangles_out.size()
              << " triangles, " << objects_out.size() << " objects\n";

    return !triangles_out.empty();
}

// ─────────────────────────────────────────────
//  URDF Articulation — persistent state
//  (LinkMeshCache and UrdfArticulation defined in urdf_internal.h)
// ─────────────────────────────────────────────

Mat4 axis_angle_rotation(float3 axis, float angle)
{
    float ax = axis.x, ay = axis.y, az = axis.z;
    float len = sqrtf(ax*ax + ay*ay + az*az);
    if (len > 1e-7f) { ax/=len; ay/=len; az/=len; }

    float c = cosf(angle), s = sinf(angle), t = 1.0f - c;
    Mat4 R;
    R.m[0][0] = t*ax*ax + c;     R.m[0][1] = t*ax*ay - s*az;  R.m[0][2] = t*ax*az + s*ay;  R.m[0][3] = 0;
    R.m[1][0] = t*ax*ay + s*az;  R.m[1][1] = t*ay*ay + c;     R.m[1][2] = t*ay*az - s*ax;  R.m[1][3] = 0;
    R.m[2][0] = t*ax*az - s*ay;  R.m[2][1] = t*ay*az + s*ax;  R.m[2][2] = t*az*az + c;     R.m[2][3] = 0;
    R.m[3][0] = 0;               R.m[3][1] = 0;               R.m[3][2] = 0;               R.m[3][3] = 1;
    return R;
}

void urdf_articulation_finalize(UrdfArticulation* h,
                                const std::vector<Triangle>& initial_tris)
{
    if (!h) return;

    // Build children map
    h->children_map.clear();
    for (auto& j : h->joints)
        h->children_map[j.parent].push_back(&j);
    if (h->root_body_link.empty())
        h->root_body_link = h->root_link;

    // Axis correction (Z-up → Y-up)
    h->z_to_y = Mat4::identity();
    h->z_to_y.m[0][0] =  1; h->z_to_y.m[0][1] =  0; h->z_to_y.m[0][2] =  0;
    h->z_to_y.m[1][0] =  0; h->z_to_y.m[1][1] =  0; h->z_to_y.m[1][2] =  1;
    h->z_to_y.m[2][0] =  0; h->z_to_y.m[2][1] = -1; h->z_to_y.m[2][2] =  0;

    // Build joint info list (only movable joints). Limits come directly from URDFJoint.
    h->joint_infos.clear();
    h->joint_name_to_idx.clear();
    for (auto& j : h->joints) {
        if (j.type == "fixed") continue;
        UrdfJointInfo ji{};
        snprintf(ji.name, sizeof(ji.name), "%s", j.name.c_str());
        if (j.type == "revolute")       ji.type = 0;
        else if (j.type == "prismatic") ji.type = 1;
        else if (j.type == "continuous") ji.type = 2;
        else ji.type = 3;

        ji.lower = j.lower;
        ji.upper = j.upper;
        ji.angle = 0;
        h->joint_name_to_idx[j.name] = (int)h->joint_infos.size();
        h->joint_infos.push_back(ji);
    }
    h->ik_locked_mask.assign(h->joint_infos.size(), 0);
    h->ik_locked_rot.assign(h->joint_infos.size(), Mat4::identity());

    // Cache mesh data per link by re-loading meshes
    // Walk the kinematic chain and record which triangles belong to each link
    int tri_cursor = 0;
    std::function<void(const std::string&)> cache_link;
    cache_link = [&](const std::string& link_name) {
        auto it = h->links.find(link_name);
        if (it == h->links.end()) return;
        const URDFLink& link = it->second;

        bool have_inline = !link.inline_mesh.vertices.empty();
        bool have_file   = !link.visual_mesh_path.empty() && fs::exists(link.visual_mesh_path);
        if (have_inline || have_file) {
            LinkMeshCache mc;
            mc.link_name = link_name;
            mc.tri_start = tri_cursor;

            bool loaded = have_inline
                ? (mc.raw = link.inline_mesh, !mc.raw.vertices.empty())
                : (load_link_mesh(link.visual_mesh_path, mc.raw) && !mc.raw.vertices.empty());

            if (loaded) {
                mc.tri_count = (int)mc.raw.indices.size() / 3;
                mc.mat_ids = mc.raw.mat_ids;
                mc.obj_id = tri_cursor > 0 ? initial_tris[tri_cursor].obj_id : 0;
                if (tri_cursor < (int)initial_tris.size())
                    mc.link_mat_base = initial_tris[tri_cursor].mat_idx;
                else
                    mc.link_mat_base = 0;
                tri_cursor += mc.tri_count;
                h->mesh_caches.push_back(std::move(mc));
            }
        }

        // Recurse children
        auto jit = h->children_map.find(link_name);
        if (jit != h->children_map.end()) {
            for (auto* joint : jit->second)
                cache_link(joint->child);
        }
    };
    cache_link(h->root_link);
    h->total_tris = tri_cursor;

    // Find end-effector. A gripper's fingers/jaws are leaves, but the wrist or
    // hand above them is the real pose target. Walk all links (not just leaves)
    // and pick the deepest link that is not part of the gripper mechanism.
    // This matters for SO-101: otherwise the EE becomes moving_jaw, so pose IK
    // drives a jaw pivot instead of rolling the wrist into a grasp frame.
    {
        auto is_gripper_link_name = [](const std::string& n) {
            std::string lower;
            lower.reserve(n.size());
            for (char c : n) lower += (char)std::tolower((unsigned char)c);
            return lower.find("finger")  != std::string::npos ||
                   lower.find("tip")     != std::string::npos ||
                   lower.find("jaw")     != std::string::npos ||
                   lower.find("gripper") != std::string::npos ||
                   lower.find("claw")    != std::string::npos ||
                   lower.find("thumb")   != std::string::npos ||
                   lower.find("knuckle") != std::string::npos;
        };
        std::string best;
        int best_depth = -1;
        std::function<void(const std::string&, int)> walk;
        walk = [&](const std::string& link_name, int depth) {
            if (!is_gripper_link_name(link_name) && depth > best_depth) {
                best = link_name;
                best_depth = depth;
            }
            auto jit = h->children_map.find(link_name);
            if (jit == h->children_map.end()) return;
            for (auto* joint : jit->second) walk(joint->child, depth + 1);
        };
        walk(h->root_link, 0);
        if (best.empty()) {
            // Degenerate case: no non-finger link at all — take deepest leaf.
            std::function<void(const std::string&, int)> leaf_walk;
            leaf_walk = [&](const std::string& link_name, int depth) {
                auto jit = h->children_map.find(link_name);
                bool has_children = (jit != h->children_map.end() && !jit->second.empty());
                if (!has_children && depth > best_depth) {
                    best = link_name;
                    best_depth = depth;
                }
                if (has_children)
                    for (auto* joint : jit->second) leaf_walk(joint->child, depth + 1);
            };
            leaf_walk(h->root_link, 0);
        }
        h->ee_link_name = best;
    }

    // Compute initial ee position from the centroid of all mesh caches that
    // are descendants of ee_link_name (including itself).
    if (!h->mesh_caches.empty() && !h->ee_link_name.empty()) {
        // Precompute the set of links in the ee subtree via BFS over children_map.
        std::unordered_set<std::string> ee_subtree;
        std::function<void(const std::string&)> walk;
        walk = [&](const std::string& n) {
            ee_subtree.insert(n);
            auto it = h->children_map.find(n);
            if (it == h->children_map.end()) return;
            for (auto* j : it->second) walk(j->child);
        };
        walk(h->ee_link_name);

        float3 sum = make_float3(0, 0, 0);
        int count = 0;
        for (auto& mc : h->mesh_caches) {
            if (!ee_subtree.count(mc.link_name)) continue;
            for (int ti = 0; ti < mc.tri_count; ++ti) {
                int idx = mc.tri_start + ti;
                if (idx >= (int)initial_tris.size()) break;
                const Triangle& t = initial_tris[idx];
                sum.x += t.v0.x + t.v1.x + t.v2.x;
                sum.y += t.v0.y + t.v1.y + t.v2.y;
                sum.z += t.v0.z + t.v1.z + t.v2.z;
                count += 3;
            }
        }
        if (count > 0) {
            float inv = 1.f / (float)count;
            h->ee_world_pos = make_float3(sum.x * inv, sum.y * inv, sum.z * inv);
        }
    }

    std::cerr << "[urdf_articulation] finalized: " << h->joint_infos.size()
              << " joints, " << h->mesh_caches.size() << " mesh caches, "
              << h->total_tris << " triangles, ee=" << h->ee_link_name
              << " pos=(" << h->ee_world_pos.x << "," << h->ee_world_pos.y
              << "," << h->ee_world_pos.z << ")\n";
}

UrdfArticulation* urdf_articulation_open(const std::string& path,
                                          const std::vector<Triangle>& initial_tris)
{
    auto* h = new UrdfArticulation;
    if (!parse_urdf(path, h->links, h->joints, h->root_link)) {
        delete h;
        return nullptr;
    }
    urdf_articulation_finalize(h, initial_tris);
    return h;
}

int urdf_joint_count(const UrdfArticulation* h) {
    return h ? (int)h->joint_infos.size() : 0;
}

UrdfJointInfo* urdf_joint_info(UrdfArticulation* h) {
    return h && !h->joint_infos.empty() ? h->joint_infos.data() : nullptr;
}

float3 urdf_end_effector_pos(UrdfArticulation* h)
{
    // Returns the cached ee position computed during the last urdf_repose() call.
    // This is the centroid of the last (deepest) link's mesh vertices,
    // guaranteed to use the exact same transform chain as the visible geometry.
    return h ? h->ee_world_pos : make_float3(0, 0, 0);
}

// ── CCD Inverse Kinematics ──────────────────────────────────────────────────

// Internal FK: compute the world-space position of every joint origin and the
// end-effector from current joint_infos angles. No dependency on cached data.
// Returns: joint_positions[i] and joint_axes[i] for each entry in chain,
//          plus ee_pos for the end-effector.
struct IKChainEntry {
    int              ji;   // index into joint_infos (-1 for fixed)
    const URDFJoint* jnt;
};

static bool ik_collect_chain(UrdfArticulation* h,
                             std::vector<IKChainEntry>& chain)
{
    chain.clear();
    std::function<bool(const std::string&)> walk;
    walk = [&](const std::string& link_name) -> bool {
        if (link_name == h->ee_link_name) return true;
        auto jit = h->children_map.find(link_name);
        if (jit == h->children_map.end()) return false;
        for (auto* joint : jit->second) {
            auto idx_it = h->joint_name_to_idx.find(joint->name);
            int ji = (idx_it != h->joint_name_to_idx.end()) ? idx_it->second : -1;
            chain.push_back({ ji, joint });
            if (walk(joint->child)) return true;
            chain.pop_back();
        }
        return false;
    };
    return walk(h->root_link);
}

// Run forward kinematics on the chain, computing world-space joint positions,
// axes, and the final ee position. Uses current joint_infos angles directly.
// Optional per-chain-entry xforms: out_xforms[i] = world transform *after*
// joint i's rotation — i.e. the world transform of chain[i].jnt->child.
static float3 ik_forward(UrdfArticulation* h,
                         const std::vector<IKChainEntry>& chain,
                         std::vector<float3>& joint_pos,
                         std::vector<float3>& joint_axis,
                         Mat4* out_ee_xform = nullptr,
                         std::vector<Mat4>* out_xforms = nullptr)
{
    joint_pos.resize(chain.size());
    joint_axis.resize(chain.size());
    if (out_xforms) out_xforms->resize(chain.size());

    Mat4 xform = h->z_to_y;
    UrdfJointInfo* joints = h->joint_infos.data();

    for (int i = 0; i < (int)chain.size(); i++) {
        const URDFJoint* jnt = chain[i].jnt;
        xform = xform * jnt->origin;

        // Record joint world position and axis BEFORE applying joint rotation
        joint_pos[i]  = xform.transform_point(make_float3(0, 0, 0));
        float3 wa = xform.transform_normal(jnt->axis);
        float al = sqrtf(wa.x*wa.x + wa.y*wa.y + wa.z*wa.z);
        if (al > 1e-7f) { wa.x /= al; wa.y /= al; wa.z /= al; }
        joint_axis[i] = wa;

        // Apply joint angle
        int ji = chain[i].ji;
        if (ji >= 0) {
            float angle = joints[ji].angle;
            int type = joints[ji].type;
            if ((type == 0 || type == 2) && angle != 0.0f)
                xform = xform * axis_angle_rotation(jnt->axis, angle);
            else if (type == 1 && angle != 0.0f) {
                Mat4 T = Mat4::identity();
                T.m[0][3] = jnt->axis.x * angle;
                T.m[1][3] = jnt->axis.y * angle;
                T.m[2][3] = jnt->axis.z * angle;
                xform = xform * T;
            }
        }

        if (out_xforms) (*out_xforms)[i] = xform;
    }

    if (out_ee_xform) *out_ee_xform = xform;
    return xform.transform_point(make_float3(0, 0, 0));
}

float3 urdf_fk_ee_pos(UrdfArticulation* h)
{
    if (!h) return make_float3(0, 0, 0);
    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain) || chain.empty())
        return h ? h->ee_world_pos : make_float3(0, 0, 0);
    std::vector<float3> jp, ja;
    return ik_forward(h, chain, jp, ja);
}

void urdf_fk_ee_transform(UrdfArticulation* h, float* out_mat16)
{
    if (!h || !out_mat16) return;
    // Identity fallback
    for (int i = 0; i < 16; i++) out_mat16[i] = (i % 5 == 0) ? 1.f : 0.f;

    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain) || chain.empty()) return;
    std::vector<float3> jp, ja;
    Mat4 ee_xform;
    ik_forward(h, chain, jp, ja, &ee_xform);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out_mat16[r * 4 + c] = ee_xform.m[r][c];
}

void urdf_joint_world_transform(UrdfArticulation* h, int joint_idx, float* out_mat16)
{
    if (!out_mat16) return;
    for (int i = 0; i < 16; i++) out_mat16[i] = (i % 5 == 0) ? 1.f : 0.f;
    if (!h || joint_idx < 0 || joint_idx >= (int)h->joint_infos.size()) return;

    std::unordered_map<std::string, float> angles;
    for (auto& ji : h->joint_infos) angles[ji.name] = ji.angle;

    bool found = false;
    std::function<void(const std::string&, Mat4)> walk =
        [&](const std::string& link_name, Mat4 world_xform) {
            if (found) return;
            if (!h->root_body_link.empty() && link_name == h->root_body_link)
                world_xform = world_xform * h->root_xform;

            auto jit = h->children_map.find(link_name);
            if (jit == h->children_map.end()) return;

            for (const URDFJoint* joint : jit->second) {
                Mat4 jx = world_xform * joint->origin;
                auto idx_it = h->joint_name_to_idx.find(joint->name);
                int ji = (idx_it == h->joint_name_to_idx.end()) ? -1 : idx_it->second;

                Mat4 child_xform = jx;
                auto ait = angles.find(joint->name);
                float a = (ait != angles.end()) ? ait->second : 0.f;
                if (joint->type == "revolute" || joint->type == "continuous") {
                    child_xform = child_xform * axis_angle_rotation(joint->axis, a);
                } else if (joint->type == "prismatic") {
                    Mat4 slide = Mat4::identity();
                    slide.m[0][3] = joint->axis.x * a;
                    slide.m[1][3] = joint->axis.y * a;
                    slide.m[2][3] = joint->axis.z * a;
                    child_xform = child_xform * slide;
                }

                if (ji == joint_idx) {
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            out_mat16[r * 4 + c] = child_xform.m[r][c];
                    found = true;
                    return;
                }

                walk(joint->child, child_xform);
                if (found) return;
            }
        };

    walk(h->root_link, h->z_to_y);
}

// Helper: map movable joint indices from chain
static void ik_movable_indices(UrdfArticulation* h,
                               const std::vector<IKChainEntry>& chain,
                               std::vector<int>& movable)
{
    movable.clear();
    UrdfJointInfo* joints = h->joint_infos.data();
    for (int ci = 0; ci < (int)chain.size(); ci++)
        if (chain[ci].ji >= 0 &&
            (joints[chain[ci].ji].type == 0 ||
             joints[chain[ci].ji].type == 1 ||
             joints[chain[ci].ji].type == 2))
            movable.push_back(ci);
}

int urdf_ik_chain_length(UrdfArticulation* h)
{
    if (!h) return 0;
    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain)) return 0;
    std::vector<int> movable;
    ik_movable_indices(h, chain, movable);
    return (int)movable.size();
}

int urdf_ik_lock_joint(UrdfArticulation* h)
{
    return h ? h->ik_lock_joint : -1;
}

void urdf_set_ik_ground_y(UrdfArticulation* h, float ground_y)
{
    if (h) h->ik_ground_y = ground_y;
}

const char* urdf_get_ee_link_name(const UrdfArticulation* h)
{
    return h ? h->ee_link_name.c_str() : "";
}

void urdf_set_ee_link_name(UrdfArticulation* h, const char* link_name)
{
    if (h && link_name) h->ee_link_name = link_name;
}

void urdf_link_names(const UrdfArticulation* h, std::vector<std::string>& out)
{
    out.clear();
    if (!h) return;
    // BFS from root so the list is ordered depth-first (arm order).
    std::vector<std::string> queue;
    queue.push_back(h->root_link);
    for (int qi = 0; qi < (int)queue.size(); ++qi) {
        out.push_back(queue[qi]);
        auto it = h->children_map.find(queue[qi]);
        if (it != h->children_map.end())
            for (auto* jnt : it->second)
                queue.push_back(jnt->child);
    }
}

void urdf_set_ik_lock_joint(UrdfArticulation* h, int joint_idx)
{
    if (!h) return;
    if (joint_idx < -1 || joint_idx >= (int)h->joint_infos.size()) return;
    h->ik_lock_joint = joint_idx;
}

int urdf_ik_lock_joint_effective(UrdfArticulation* h)
{
    if (!h || h->joint_infos.empty()) return -1;
    int n = (int)h->joint_infos.size();
    if (h->ik_lock_joint >= 0 && h->ik_lock_joint < n) {
        int t = h->joint_infos[h->ik_lock_joint].type;
        if (t == 0 || t == 2) return h->ik_lock_joint;
    }
    // UI/keyboard fallback: last movable (revolute/continuous) joint.
    for (int i = n - 1; i >= 0; --i) {
        int t = h->joint_infos[i].type;
        if (t == 0 || t == 2) return i;
    }
    return -1;
}

int urdf_kb_joint(UrdfArticulation* h)
{
    return h ? h->kb_joint : -1;
}

void urdf_set_kb_joint(UrdfArticulation* h, int joint_idx)
{
    if (!h) return;
    if (joint_idx < -1 || joint_idx >= (int)h->joint_infos.size()) return;
    h->kb_joint = joint_idx;
}

int urdf_kb_joint_effective(UrdfArticulation* h)
{
    if (!h || h->joint_infos.empty()) return -1;
    int n = (int)h->joint_infos.size();
    if (h->kb_joint >= 0 && h->kb_joint < n) {
        int t = h->joint_infos[h->kb_joint].type;
        // Any drivable joint: revolute, prismatic, or continuous.
        if (t == 0 || t == 1 || t == 2) return h->kb_joint;
    }
    // Default: the IK-locked joint (what [ / ] rotated before this API).
    return urdf_ik_lock_joint_effective(h);
}

bool urdf_ik_is_locked(UrdfArticulation* h, int joint_idx)
{
    if (!h || joint_idx < 0 || joint_idx >= (int)h->ik_locked_mask.size()) return false;
    return h->ik_locked_mask[joint_idx] != 0;
}

void urdf_ik_set_locked(UrdfArticulation* h, int joint_idx, bool locked)
{
    if (!h || joint_idx < 0 || joint_idx >= (int)h->ik_locked_mask.size()) return;
    h->ik_locked_mask[joint_idx] = locked ? 1 : 0;

    if (!locked) return;

    // Snapshot the child link's current world rotation — the solver will
    // preserve this while IK rearranges the rest of the arm. Without this
    // freeze, the joint would keep its *angle* fixed but the link could
    // swing arbitrarily as ancestors rotate.
    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain)) return;
    std::vector<float3> jpos, jaxis;
    std::vector<Mat4>  xforms;
    ik_forward(h, chain, jpos, jaxis, nullptr, &xforms);
    for (int ci = 0; ci < (int)chain.size(); ++ci) {
        if (chain[ci].ji == joint_idx) {
            h->ik_locked_rot[joint_idx] = xforms[ci];
            break;
        }
    }
}

void urdf_ik_clear_all_locks(UrdfArticulation* h)
{
    if (!h) return;
    std::fill(h->ik_locked_mask.begin(), h->ik_locked_mask.end(), 0);
}

float3 urdf_gripper_grip_point(UrdfArticulation* h, float forward_offset)
{
    if (!h) return make_float3(0, 0, 0);
    float m[16];
    urdf_fk_ee_transform(h, m);
    // m is row-major 4x4; column 2 (m[0][2], m[1][2], m[2][2]) is the hand's
    // local +Z axis in world space. Origin is column 3.
    float ox = m[3], oy = m[7], oz = m[11];
    float zx = m[2], zy = m[6], zz = m[10];
    return make_float3(ox + zx * forward_offset,
                       oy + zy * forward_offset,
                       oz + zz * forward_offset);
}

// Case-insensitive substring match against a list of gripper-joint keywords.
// URDF authors pick from a grab-bag of names (finger, gripper, jaw, knuckle,
// driver, claw, hand, thumb), so we cover the common ones instead of just
// "finger" — otherwise the Articulation panel hides its Close/Open block on
// any gripper that deviates from the Panda naming convention.
static bool name_is_gripperlike(const std::string& nm)
{
    std::string lower;
    lower.reserve(nm.size());
    for (char c : nm) lower += (char)tolower((unsigned char)c);
    static const char* kw[] = {
        "finger", "gripper", "jaw", "knuckle", "driver", "claw", "thumb",
    };
    for (const char* k : kw) if (lower.find(k) != std::string::npos) return true;
    return false;
}

void urdf_gripper_finger_indices(UrdfArticulation* h, std::vector<int>& out)
{
    out.clear();
    if (!h) return;
    int n = (int)h->joint_infos.size();
    for (int i = 0; i < n; ++i) {
        int t = h->joint_infos[i].type;
        if (t != 0 && t != 1 && t != 2) continue;   // driveable only
        if (name_is_gripperlike(h->joint_infos[i].name)) out.push_back(i);
    }
}

void urdf_gripper_finger_worlds(UrdfArticulation* h, std::vector<float3>& out)
{
    out.clear();
    if (!h) return;

    // Joint-name → current angle map. Same data the repose walker uses.
    std::unordered_map<std::string, float> angles;
    for (auto& ji : h->joint_infos) angles[ji.name] = ji.angle;

    // Recursive tree walk mirroring urdf_repose's joint-transform logic, but
    // recording origin positions instead of transforming geometry.
    std::function<void(const std::string&, Mat4)> walk;
    walk = [&](const std::string& link_name, Mat4 world_xform) {
        if (link_name == h->root_body_link)
            world_xform = world_xform * h->root_xform;

        if (name_is_gripperlike(link_name))
            out.push_back(world_xform.transform_point(make_float3(0, 0, 0)));

        auto jit = h->children_map.find(link_name);
        if (jit == h->children_map.end()) return;
        for (auto* joint : jit->second) {
            Mat4 jx = world_xform * joint->origin;
            float a = 0.f;
            auto ait = angles.find(joint->name);
            if (ait != angles.end()) a = ait->second;
            if ((joint->type == "revolute" || joint->type == "continuous") && a != 0.f) {
                jx = jx * axis_angle_rotation(joint->axis, a);
            } else if (joint->type == "prismatic" && a != 0.f) {
                Mat4 slide = Mat4::identity();
                slide.m[0][3] = joint->axis.x * a;
                slide.m[1][3] = joint->axis.y * a;
                slide.m[2][3] = joint->axis.z * a;
                jx = jx * slide;
            }
            walk(joint->child, jx);
        }
    };

    walk(h->root_link, h->z_to_y);
}

float3 urdf_joint_pos(UrdfArticulation* h, int joint_idx)
{
    if (!h) return make_float3(0, 0, 0);
    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain) || chain.empty())
        return make_float3(0, 0, 0);
    std::vector<int> movable;
    ik_movable_indices(h, chain, movable);
    if (joint_idx < 0 || joint_idx >= (int)movable.size())
        return make_float3(0, 0, 0);
    std::vector<float3> jp, ja;
    ik_forward(h, chain, jp, ja);
    return jp[movable[joint_idx]];
}



UrdfJointInfo* urdf_chain_joint_info(UrdfArticulation* h, int movable_idx)
{
    if (!h) return nullptr;
    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain) || chain.empty()) return nullptr;
    std::vector<int> movable;
    ik_movable_indices(h, chain, movable);
    if (movable_idx < 0 || movable_idx >= (int)movable.size()) return nullptr;
    int ji = chain[movable[movable_idx]].ji;
    if (ji < 0 || ji >= (int)h->joint_infos.size()) return nullptr;
    return &h->joint_infos[ji];
}

// CCD helper: compute the rotation angle for one joint to bring 'current' toward 'target'
// around the joint's axis. Returns the signed angle (already damped).
static float ccd_angle_for_joint(float3 j_pos, float3 j_axis,
                                 float3 current, float3 target, float damping)
{
    float3 to_cur = make_float3(current.x - j_pos.x, current.y - j_pos.y, current.z - j_pos.z);
    float3 to_tgt = make_float3(target.x  - j_pos.x, target.y  - j_pos.y, target.z  - j_pos.z);

    // Project onto plane perpendicular to joint axis
    float dc = to_cur.x*j_axis.x + to_cur.y*j_axis.y + to_cur.z*j_axis.z;
    float dt = to_tgt.x*j_axis.x + to_tgt.y*j_axis.y + to_tgt.z*j_axis.z;
    float3 pc = make_float3(to_cur.x - dc*j_axis.x, to_cur.y - dc*j_axis.y, to_cur.z - dc*j_axis.z);
    float3 pt = make_float3(to_tgt.x - dt*j_axis.x, to_tgt.y - dt*j_axis.y, to_tgt.z - dt*j_axis.z);

    float lc = sqrtf(pc.x*pc.x + pc.y*pc.y + pc.z*pc.z);
    float lt = sqrtf(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    if (lc < 1e-6f || lt < 1e-6f) return 0.f;

    float dv = (pc.x*pt.x + pc.y*pt.y + pc.z*pt.z) / (lc * lt);
    dv = fmaxf(-1.f, fminf(1.f, dv));
    float angle = acosf(dv);

    float3 cr = make_float3(pc.y*pt.z - pc.z*pt.y,
                            pc.z*pt.x - pc.x*pt.z,
                            pc.x*pt.y - pc.y*pt.x);
    float sign = (cr.x*j_axis.x + cr.y*j_axis.y + cr.z*j_axis.z) > 0 ? 1.f : -1.f;
    return angle * sign * damping;
}

static void clamp_joint(UrdfJointInfo& info, float PI)
{
    if (info.type == 0) {
        if (info.angle < info.lower) info.angle = info.lower;
        if (info.angle > info.upper) info.angle = info.upper;
    } else if (info.type == 2) {
        while (info.angle >  PI) info.angle -= 2.f * PI;
        while (info.angle < -PI) info.angle += 2.f * PI;
    }
}

// Gauss-Jordan in-place inversion of an N×N row-major matrix with partial
// pivoting. Returns false if singular — in DLS we add λ²I so this shouldn't
// happen, but we keep the guard.
static bool invert_nxn(std::vector<float>& A, int N)
{
    std::vector<float> aug((size_t)N * 2 * N, 0.f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) aug[i*2*N + j] = A[i*N + j];
        aug[i*2*N + N + i] = 1.f;
    }
    for (int p = 0; p < N; ++p) {
        int pr = p;
        float pv = fabsf(aug[p*2*N + p]);
        for (int r = p+1; r < N; ++r) {
            float v = fabsf(aug[r*2*N + p]);
            if (v > pv) { pv = v; pr = r; }
        }
        if (pv < 1e-12f) return false;
        if (pr != p) {
            for (int j = 0; j < 2*N; ++j) std::swap(aug[p*2*N + j], aug[pr*2*N + j]);
        }
        float invp = 1.f / aug[p*2*N + p];
        for (int j = 0; j < 2*N; ++j) aug[p*2*N + j] *= invp;
        for (int r = 0; r < N; ++r) {
            if (r == p) continue;
            float f = aug[r*2*N + p];
            if (f == 0.f) continue;
            for (int j = 0; j < 2*N; ++j) aug[r*2*N + j] -= f * aug[p*2*N + j];
        }
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i*N + j] = aug[i*2*N + N + j];
    return true;
}

// Extract world-frame angular error ω that rotates R_current → R_target.
// For small drift, ω ≈ 0.5 * vee(ΔR - ΔRᵀ) where ΔR = R_target · R_currentᵀ.
static void angular_error_world(const Mat4& R_target, const Mat4& R_current,
                                 float& ox, float& oy, float& oz)
{
    float D[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            D[i][j] = R_target.m[i][0]*R_current.m[j][0]
                    + R_target.m[i][1]*R_current.m[j][1]
                    + R_target.m[i][2]*R_current.m[j][2];
        }
    ox = 0.5f * (D[2][1] - D[1][2]);
    oy = 0.5f * (D[0][2] - D[2][0]);
    oz = 0.5f * (D[1][0] - D[0][1]);
}

// Shared DLS core used by both urdf_solve_ik (target_ci < 0 → drive EE tip)
// and urdf_solve_ik_joint (target_ci ≥ 0 → drive that joint's origin point).
// Honors primary-lock (excluded from DOF) and per-joint orientation locks
// (included in DOF, with angular-error rows added to the Jacobian so the
// locked link's world rotation is preserved).
static bool urdf_solve_dls(UrdfArticulation* h, float3 target_pos,
                            int target_ci, int max_iters, float tolerance,
                            const Mat4* ee_rot_target = nullptr)
{
    if (!h) return false;
    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain) || chain.empty()) return false;

    UrdfJointInfo* joints = h->joint_infos.data();
    const float PI          = 3.14159265f;
    const float lambda2     = 0.05f * 0.05f;
    const float max_err_lin = 0.15f;   // m per step
    const float max_err_ang = 0.20f;   // rad per step
    const float max_dq      = 0.20f;   // per-joint step cap

    // EE orientation target only applies when driving the EE tip. When
    // supplied, the primary IK lock is overridden: hitting a 6-DOF pose
    // usually needs every joint, and excluding one (typically the wrist)
    // makes the task infeasible.
    const bool have_ee_rot = (target_ci < 0 && ee_rot_target != nullptr);

    int primary_lock = (h->ik_lock_joint >= 0) ? urdf_ik_lock_joint_effective(h) : -1;
    if (have_ee_rot) primary_lock = -1;
    int use_limit    = (target_ci < 0) ? (int)chain.size() : target_ci;

    // Movable DOF: everything we're allowed to drive. Per-joint freezes
    // remain in the set — they need to move to preserve orientation.
    std::vector<int> use_ci;
    use_ci.reserve(use_limit);
    for (int ci = 0; ci < use_limit; ++ci) {
        int ji = chain[ci].ji;
        if (ji < 0) continue;
        if (ji == primary_lock) continue;
        int t = joints[ji].type;
        if (t == 0 || t == 1 || t == 2) use_ci.push_back(ci);
    }
    const int N = (int)use_ci.size();
    if (N == 0) return false;

    // Weighted DLS: per-joint stiffness w[k] (higher = harder to move). The
    // first two driven revolute joints on a serial arm (base twist, shoulder
    // pitch on Panda/UR/Kinova) swing through the largest arcs in world space
    // and dominate the unweighted least-squares solution. Up-weighting them
    // pushes IK to prefer moving the elbow/wrist instead. Implemented by
    // scaling Jacobian columns by 1/sqrt(w) so the existing DLS math works
    // unchanged; final Δq is rescaled on step application.
    std::vector<float> inv_sqrt_w(N, 1.f);
    {
        int rev_seen = 0;
        for (int k = 0; k < N; ++k) {
            int ji = chain[use_ci[k]].ji;
            if (joints[ji].type != 0) continue;   // revolute only
            if (rev_seen == 0)      inv_sqrt_w[k] = 1.f / sqrtf(5.0f);
            else if (rev_seen == 1) inv_sqrt_w[k] = 1.f / sqrtf(2.5f);
            ++rev_seen;
        }
    }

    // Orientation constraints — one per user-frozen joint that has at
    // least one movable ancestor to drive its child link.
    struct Cons { int ci; Mat4 R_target; };
    std::vector<Cons> cons;
    for (int ci = 0; ci < (int)chain.size(); ++ci) {
        int ji = chain[ci].ji;
        if (ji < 0) continue;
        if (ji >= (int)h->ik_locked_mask.size()) continue;
        if (!h->ik_locked_mask[ji]) continue;
        if (ji == primary_lock) continue;
        bool has_upstream_mover = false;
        for (int k = 0; k < N; ++k)
            if (use_ci[k] <= ci) { has_upstream_mover = true; break; }
        if (!has_upstream_mover) continue;
        cons.push_back({ ci, h->ik_locked_rot[ji] });
    }
    const int K = (int)cons.size();
    const int ee_rot_row = have_ee_rot ? (3 + 3 * K) : -1;
    const int M = 3 + 3 * K + (have_ee_rot ? 3 : 0);

    std::vector<float3> jpos, jaxis;
    std::vector<Mat4>   xforms;
    std::vector<float>  J((size_t)M * N, 0.f);
    std::vector<float>  e((size_t)M, 0.f);
    std::vector<float>  A((size_t)M * M, 0.f);
    std::vector<float>  y((size_t)M, 0.f);
    std::vector<float>  dq((size_t)N, 0.f);

    bool   reach_init = false;
    float3 base_pos   = make_float3(0, 0, 0);
    float  reach_safe = 0.f;        // hard cap: max straight-line reach
    float  reach_max  = 0.f;        // full straight-line arm reach
    const float reach_margin = 1.0f;

    Mat4 ee_xform_local;
    for (int iter = 0; iter < max_iters; ++iter) {
        float3 ee = ik_forward(h, chain, jpos, jaxis,
                                have_ee_rot ? &ee_xform_local : nullptr,
                                K > 0 ? &xforms : nullptr);
        float3 point = (target_ci < 0) ? ee : jpos[target_ci];

        // Compute arm reach once. Segment lengths between successive joint
        // positions are invariant under rotations (fixed link offsets), so
        // the sum gives straight-line max reach regardless of pose.
        if (!reach_init && !jpos.empty()) {
            base_pos = jpos[0];
            int reach_end = (target_ci < 0) ? (int)chain.size() - 1 : target_ci;
            if (reach_end > (int)jpos.size() - 1) reach_end = (int)jpos.size() - 1;
            float r = 0.f;
            for (int i = 1; i <= reach_end; ++i) {
                float dx = jpos[i].x - jpos[i-1].x;
                float dy = jpos[i].y - jpos[i-1].y;
                float dz = jpos[i].z - jpos[i-1].z;
                r += sqrtf(dx*dx + dy*dy + dz*dz);
            }
            if (target_ci < 0 && reach_end >= 0) {
                float dx = ee.x - jpos[reach_end].x;
                float dy = ee.y - jpos[reach_end].y;
                float dz = ee.z - jpos[reach_end].z;
                r += sqrtf(dx*dx + dy*dy + dz*dz);
            }
            reach_max  = r;
            reach_safe = r * reach_margin;
            reach_init = true;
        }

        // Clamp target within reach budget. The viewport IK handle should be
        // able to pull the arm all the way to its physical reach/limit, so the
        // cap is the full straight-line reach rather than a conservative 97%.
        float3 tgt = target_pos;
        if (reach_init && reach_safe > 0.f) {
            float tx = tgt.x - base_pos.x;
            float ty = tgt.y - base_pos.y;
            float tz = tgt.z - base_pos.z;
            float td = sqrtf(tx*tx + ty*ty + tz*tz);
            if (td > reach_safe && td > 1e-6f) {
                float s = reach_safe / td;
                tgt.x = base_pos.x + tx * s;
                tgt.y = base_pos.y + ty * s;
                tgt.z = base_pos.z + tz * s;
            }
        }

        float ex = tgt.x - point.x;
        float ey = tgt.y - point.y;
        float ez = tgt.z - point.z;
        float err_p = sqrtf(ex*ex + ey*ey + ez*ez);

        // Per-constraint angular errors (world frame).
        std::vector<float> aerr((size_t)(3 * K), 0.f);
        float max_ang = 0.f;
        for (int c = 0; c < K; ++c) {
            float ox, oy, oz;
            angular_error_world(cons[c].R_target, xforms[cons[c].ci], ox, oy, oz);
            aerr[3*c + 0] = ox;
            aerr[3*c + 1] = oy;
            aerr[3*c + 2] = oz;
            float m = sqrtf(ox*ox + oy*oy + oz*oz);
            if (m > max_ang) max_ang = m;
        }

        // EE orientation error (only when an EE rotation target was supplied).
        float ee_aerr[3] = { 0.f, 0.f, 0.f };
        if (have_ee_rot) {
            float ox, oy, oz;
            angular_error_world(*ee_rot_target, ee_xform_local, ox, oy, oz);
            ee_aerr[0] = ox; ee_aerr[1] = oy; ee_aerr[2] = oz;
            float m = sqrtf(ox*ox + oy*oy + oz*oz);
            if (m > max_ang) max_ang = m;
        }

        if (err_p < tolerance && max_ang < 0.01f) return true;

        // Clamp per-task errors so a single iteration stays well-scaled.
        if (err_p > max_err_lin) {
            float s = max_err_lin / err_p;
            ex *= s; ey *= s; ez *= s;
        }
        for (int c = 0; c < K; ++c) {
            float mx = aerr[3*c+0], my = aerr[3*c+1], mz = aerr[3*c+2];
            float m = sqrtf(mx*mx + my*my + mz*mz);
            if (m > max_err_ang && m > 1e-9f) {
                float s = max_err_ang / m;
                aerr[3*c+0] *= s; aerr[3*c+1] *= s; aerr[3*c+2] *= s;
            }
        }
        if (have_ee_rot) {
            float mx = ee_aerr[0], my = ee_aerr[1], mz = ee_aerr[2];
            float m = sqrtf(mx*mx + my*my + mz*mz);
            if (m > max_err_ang && m > 1e-9f) {
                float s = max_err_ang / m;
                ee_aerr[0] *= s; ee_aerr[1] *= s; ee_aerr[2] *= s;
            }
        }

        // Assemble full error stack.
        e[0] = ex; e[1] = ey; e[2] = ez;
        for (int c = 0; c < K; ++c) {
            e[3 + 3*c + 0] = aerr[3*c + 0];
            e[3 + 3*c + 1] = aerr[3*c + 1];
            e[3 + 3*c + 2] = aerr[3*c + 2];
        }
        if (have_ee_rot) {
            e[ee_rot_row + 0] = ee_aerr[0];
            e[ee_rot_row + 1] = ee_aerr[1];
            e[ee_rot_row + 2] = ee_aerr[2];
        }

        // Build the M×N Jacobian.
        std::fill(J.begin(), J.end(), 0.f);
        for (int k = 0; k < N; ++k) {
            int ci = use_ci[k];
            int t  = joints[chain[ci].ji].type;
            float3 ax = jaxis[ci];
            // Position rows (target point).
            if (t == 1) {
                J[0*N + k] = ax.x;
                J[1*N + k] = ax.y;
                J[2*N + k] = ax.z;
            } else {
                float rx = point.x - jpos[ci].x;
                float ry = point.y - jpos[ci].y;
                float rz = point.z - jpos[ci].z;
                J[0*N + k] = ax.y*rz - ax.z*ry;
                J[1*N + k] = ax.z*rx - ax.x*rz;
                J[2*N + k] = ax.x*ry - ax.y*rx;
            }
            // Angular rows: joint k contributes to constraint c iff k ≤ c's
            // chain index (it's the locked link itself or upstream). Prismatic
            // contributes nothing angular.
            for (int c = 0; c < K; ++c) {
                if (ci > cons[c].ci) continue;
                if (t == 1) continue;
                int row = 3 + 3*c;
                J[(row+0)*N + k] = ax.x;
                J[(row+1)*N + k] = ax.y;
                J[(row+2)*N + k] = ax.z;
            }
            // EE orientation rows: every movable revolute/continuous joint in
            // the chain is upstream of the EE (target_ci < 0 ⇒ use_limit spans
            // the whole chain), so the angular contribution is the world axis.
            if (have_ee_rot && t != 1) {
                J[(ee_rot_row+0)*N + k] = ax.x;
                J[(ee_rot_row+1)*N + k] = ax.y;
                J[(ee_rot_row+2)*N + k] = ax.z;
            }
        }

        // Apply per-joint weights: J̃ = J · W^(-1/2). From here on, everything
        // (A, y, dq, null-space bias) operates on J̃; the real joint delta is
        // recovered by multiplying dq[k] by inv_sqrt_w[k] on step application.
        for (int k = 0; k < N; ++k) {
            float s = inv_sqrt_w[k];
            if (s == 1.f) continue;
            for (int r = 0; r < M; ++r) J[r*N + k] *= s;
        }

        // A = J Jᵀ + λ² I  (M×M).
        std::fill(A.begin(), A.end(), 0.f);
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < M; ++c) {
                float s = 0.f;
                for (int k = 0; k < N; ++k) s += J[r*N + k] * J[c*N + k];
                A[r*M + c] = s;
            }
        for (int r = 0; r < M; ++r) A[r*M + r] += lambda2;

        if (!invert_nxn(A, M)) break;

        // y = A⁻¹ e  (M).
        for (int r = 0; r < M; ++r) {
            float s = 0.f;
            for (int c = 0; c < M; ++c) s += A[r*M + c] * e[c];
            y[r] = s;
        }

        // Δq_task = Jᵀ y  (N).
        for (int k = 0; k < N; ++k) {
            float s = 0.f;
            for (int r = 0; r < M; ++r) s += J[r*N + k] * y[r];
            dq[k] = s;
        }

        // Mild null-space bias toward joint center. Keep this only while the
        // arm is comfortably inside its workspace; near full extension the
        // user is explicitly asking to reach the boundary, so do not fold the
        // joints away from their limits.
        {
            float extension = 0.f;
            if (reach_init && reach_max > 1e-6f) {
                float dx = point.x - base_pos.x;
                float dy = point.y - base_pos.y;
                float dz = point.z - base_pos.z;
                extension = sqrtf(dx*dx + dy*dy + dz*dz) / reach_max;
            }
            float bias_gain = 0.04f * fmaxf(0.f, 1.f - extension / 0.85f);
            std::vector<float> qb((size_t)N, 0.f);
            for (int k = 0; k < N; ++k) {
                UrdfJointInfo& info = joints[chain[use_ci[k]].ji];
                if (info.type != 0) continue;   // revolute only
                float lo = info.lower, hi = info.upper;
                if (!(hi > lo)) continue;       // skip unlimited/degenerate
                float center = 0.5f * (lo + hi);
                // qb is in weighted space (q̃b = W^(1/2)·qb_joint), so divide
                // the joint-space bias by inv_sqrt_w[k] = 1/√w.
                qb[k] = bias_gain * (center - info.angle) / inv_sqrt_w[k];
            }
            // J qb  (M)
            std::vector<float> Jqb((size_t)M, 0.f);
            for (int r = 0; r < M; ++r) {
                float s = 0.f;
                for (int k = 0; k < N; ++k) s += J[r*N + k] * qb[k];
                Jqb[r] = s;
            }
            // A⁻¹ (J qb)  (M)
            std::vector<float> AiJqb((size_t)M, 0.f);
            for (int r = 0; r < M; ++r) {
                float s = 0.f;
                for (int c = 0; c < M; ++c) s += A[r*M + c] * Jqb[c];
                AiJqb[r] = s;
            }
            // Δq += qb − Jᵀ A⁻¹ J qb
            for (int k = 0; k < N; ++k) {
                float s = 0.f;
                for (int r = 0; r < M; ++r) s += J[r*N + k] * AiJqb[r];
                dq[k] += qb[k] - s;
            }
        }

        // Ground repulsion: for each joint whose world Y is below ik_ground_y,
        // add a null-space bias that rotates upstream joints to lift it back up.
        // Uses the Y-row of the per-joint position Jacobian (same cross-product
        // formula as the primary task, but evaluated at each joint origin rather
        // than the EE).
        if (h->ik_ground_y > -1e29f) {
            const float gnd       = h->ik_ground_y;
            const float gnd_margin = 0.04f;   // start repelling 4 cm above floor
            const float gnd_gain   = 2.0f;
            std::vector<float> qg((size_t)N, 0.f);
            bool any_penetrating = false;
            for (int ci = 0; ci < (int)chain.size(); ++ci) {
                float pen = (gnd + gnd_margin) - jpos[ci].y;
                if (pen <= 0.f) continue;
                any_penetrating = true;
                // Accumulate dJoint_Y/dq[k] for all DOF k upstream of ci.
                for (int k = 0; k < N; ++k) {
                    if (use_ci[k] > ci) continue;   // joint k is downstream
                    int t = joints[chain[use_ci[k]].ji].type;
                    if (t != 0 && t != 2) continue; // revolute/continuous only
                    float3 ax = jaxis[use_ci[k]];
                    float3 r  = { jpos[ci].x - jpos[use_ci[k]].x,
                                  jpos[ci].y - jpos[use_ci[k]].y,
                                  jpos[ci].z - jpos[use_ci[k]].z };
                    // (axis × r).y
                    float dy = ax.z * r.x - ax.x * r.z;
                    qg[k] += gnd_gain * pen * dy / inv_sqrt_w[k];
                }
            }
            if (any_penetrating) {
                // Project onto null space and add.
                std::vector<float> Jqg((size_t)M, 0.f);
                for (int r = 0; r < M; ++r) {
                    float s = 0.f;
                    for (int k = 0; k < N; ++k) s += J[r*N+k] * qg[k];
                    Jqg[r] = s;
                }
                std::vector<float> AiJqg((size_t)M, 0.f);
                for (int r = 0; r < M; ++r) {
                    float s = 0.f;
                    for (int c = 0; c < M; ++c) s += A[r*M+c] * Jqg[c];
                    AiJqg[r] = s;
                }
                for (int k = 0; k < N; ++k) {
                    float s = 0.f;
                    for (int r = 0; r < M; ++r) s += J[r*N+k] * AiJqg[r];
                    dq[k] += qg[k] - s;
                }
            }
        }

        // Step cap operates on the real joint-space delta dq[k]·inv_sqrt_w[k],
        // not the weighted-space dq[k], so max_dq is still the true radians
        // cap per iteration regardless of weighting.
        float max_abs = 0.f;
        for (int k = 0; k < N; ++k) {
            float a = fabsf(dq[k] * inv_sqrt_w[k]);
            if (a > max_abs) max_abs = a;
        }
        float scale = 1.f;
        if (max_abs > max_dq && max_abs > 1e-9f) scale = max_dq / max_abs;

        for (int k = 0; k < N; ++k) {
            UrdfJointInfo& info = joints[chain[use_ci[k]].ji];
            info.angle += scale * dq[k] * inv_sqrt_w[k];
            clamp_joint(info, PI);
        }
    }
    return false;
}

bool urdf_solve_ik(UrdfArticulation* h, float3 target_pos,
                   int max_iters, float tolerance)
{
    return urdf_solve_dls(h, target_pos, -1, max_iters, tolerance);
}

bool urdf_solve_ik_pose(UrdfArticulation* h, float3 target_pos,
                        const float* target_rot_mat16,
                        int max_iters, float tolerance)
{
    if (!target_rot_mat16)
        return urdf_solve_dls(h, target_pos, -1, max_iters, tolerance, nullptr);
    Mat4 R_target;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            R_target.m[r][c] = target_rot_mat16[r * 4 + c];
    return urdf_solve_dls(h, target_pos, -1, max_iters, tolerance, &R_target);
}

bool urdf_solve_ik_joint(UrdfArticulation* h, int joint_idx, float3 target_pos,
                         int max_iters, float tolerance)
{
    if (!h) return false;
    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain) || chain.empty()) return false;
    std::vector<int> movable;
    ik_movable_indices(h, chain, movable);
    if (joint_idx < 1 || joint_idx >= (int)movable.size()) return false;
    return urdf_solve_dls(h, target_pos, movable[joint_idx], max_iters, tolerance);
}

bool urdf_repose(UrdfArticulation* h, std::vector<Triangle>& triangles_out)
{
    if (!h || h->mesh_caches.empty()) return false;

    // Build joint angle map: joint_name → angle
    std::unordered_map<std::string, float> angles;
    for (auto& ji : h->joint_infos)
        angles[ji.name] = ji.angle;

    // Walk kinematic chain, computing world transforms, and update triangles
    int cache_idx = 0;

    // Accumulate end-effector centroid only for the designated ee_link_name.
    // (Previously this tracked "last link visited", which broke when non-robot
    //  links like a worldbody ground plane were added after the robot's tip.)
    float3 ee_centroid = make_float3(0, 0, 0);
    int    ee_vert_count = 0;

    std::function<void(const std::string&, Mat4, bool)> repose_link;
    repose_link = [&](const std::string& link_name, Mat4 world_xform, bool under_ee) {
        if (link_name == h->root_body_link)
            world_xform = world_xform * h->root_xform;

        // Once we enter the ee subtree, all descendants contribute to the centroid.
        if (!under_ee && link_name == h->ee_link_name) under_ee = true;

        if (cache_idx < (int)h->mesh_caches.size() &&
            h->mesh_caches[cache_idx].link_name == link_name)
        {
            auto& mc = h->mesh_caches[cache_idx];
            const Mat4& local_xform = h->links[link_name].visual_origin;
            const Mat4& world = world_xform;
            bool accumulate_ee = under_ee;

            // Update triangles in-place
            for (int ti = 0; ti < mc.tri_count; ++ti) {
                int idx = mc.tri_start + ti;
                if (idx >= (int)triangles_out.size()) break;

                int i0 = mc.raw.indices[ti*3];
                int i1 = mc.raw.indices[ti*3+1];
                int i2 = mc.raw.indices[ti*3+2];

                Triangle& tri = triangles_out[idx];
                tri.v0 = world.transform_point(local_xform.transform_point(mc.raw.vertices[i0]));
                tri.v1 = world.transform_point(local_xform.transform_point(mc.raw.vertices[i1]));
                tri.v2 = world.transform_point(local_xform.transform_point(mc.raw.vertices[i2]));

                if (accumulate_ee) {
                    ee_centroid.x += tri.v0.x + tri.v1.x + tri.v2.x;
                    ee_centroid.y += tri.v0.y + tri.v1.y + tri.v2.y;
                    ee_centroid.z += tri.v0.z + tri.v1.z + tri.v2.z;
                    ee_vert_count += 3;
                }

                if (i0 < (int)mc.raw.normals.size() && i1 < (int)mc.raw.normals.size() && i2 < (int)mc.raw.normals.size()) {
                    tri.n0 = world.transform_normal(local_xform.transform_normal(mc.raw.normals[i0]));
                    tri.n1 = world.transform_normal(local_xform.transform_normal(mc.raw.normals[i1]));
                    tri.n2 = world.transform_normal(local_xform.transform_normal(mc.raw.normals[i2]));
                } else {
                    float3 e1 = tri.v1 - tri.v0;
                    float3 e2 = tri.v2 - tri.v0;
                    float3 fn = make_float3(e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z, e1.x*e2.y-e1.y*e2.x);
                    float len = sqrtf(fn.x*fn.x + fn.y*fn.y + fn.z*fn.z);
                    if (len > 1e-7f) { fn.x/=len; fn.y/=len; fn.z/=len; }
                    tri.n0 = tri.n1 = tri.n2 = fn;
                }
            }
            ++cache_idx;
        }

        // Recurse children with joint transforms
        auto jit = h->children_map.find(link_name);
        if (jit != h->children_map.end()) {
            for (auto* joint : jit->second) {
                Mat4 joint_xform = world_xform * joint->origin;

                // Apply joint angle
                float angle = 0;
                auto ait = angles.find(joint->name);
                if (ait != angles.end()) angle = ait->second;

                if ((joint->type == "revolute" || joint->type == "continuous") && angle != 0.0f) {
                    joint_xform = joint_xform * axis_angle_rotation(joint->axis, angle);
                } else if (joint->type == "prismatic" && angle != 0.0f) {
                    Mat4 T = Mat4::identity();
                    T.m[0][3] = joint->axis.x * angle;
                    T.m[1][3] = joint->axis.y * angle;
                    T.m[2][3] = joint->axis.z * angle;
                    joint_xform = joint_xform * T;
                }

                repose_link(joint->child, joint_xform, under_ee);
            }
        }
    };

    repose_link(h->root_link, h->z_to_y, false);

    // Cache ee position as centroid of the designated ee link's mesh
    if (ee_vert_count > 0) {
        float inv = 1.f / (float)ee_vert_count;
        h->ee_world_pos = make_float3(ee_centroid.x * inv,
                                      ee_centroid.y * inv,
                                      ee_centroid.z * inv);
    }

    return true;
}

void urdf_articulation_close(UrdfArticulation* h)
{
    delete h;
}

void urdf_set_root_xform(UrdfArticulation* h, const float m16[16])
{
    if (!h || !m16) return;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            h->root_xform.m[r][c] = m16[r * 4 + c];
}

void urdf_clear_root_xform(UrdfArticulation* h)
{
    if (!h) return;
    h->root_xform = Mat4::identity();
}

// Sim-driven repose. Each articulation link whose name matches a sim body is
// snapped to z_to_y * body_world_mujoco, exactly matching MuJoCo's xpos/xmat.
// Links not in the map (synthetic joint-frame links, fixed-joint geom wrappers,
// the synthetic __world__ root) inherit via the joint chain from their parent —
// but every geom that renders is the fixed child of a body that *is* in the
// map, so its world transform = body_world * visual_origin = MuJoCo's geom
// pose. The reimplemented FK is bypassed for any link that matters.
bool urdf_repose_with_body_xforms(UrdfArticulation* h,
                                   const std::vector<std::string>& body_names,
                                   const std::vector<float>& body_mats_flat,
                                   std::vector<Triangle>& triangles_out)
{
    if (!h || h->mesh_caches.empty()) return false;
    if (body_names.empty() || body_mats_flat.size() != body_names.size() * 16) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr, "[urdf_repose_with_body_xforms] EMPTY INPUT — falling back to FK. names=%zu mats=%zu\n",
                         body_names.size(), body_mats_flat.size());
            warned = true;
        }
        return urdf_repose(h, triangles_out);
    }

    // Build name → Mat4 (Z-up) map from parallel vectors.
    std::unordered_map<std::string, Mat4> body_world;
    body_world.reserve(body_names.size());
    for (size_t i = 0; i < body_names.size(); ++i) {
        Mat4 m;
        const float* src = body_mats_flat.data() + i * 16;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m.m[r][c] = src[r * 4 + c];
        body_world.emplace(body_names[i], m);
    }

    {
        static int printed = 0;
        if (printed < 1) {
            std::fprintf(stderr, "[snap] === SIM BODIES (%zu) ===\n", body_names.size());
            for (size_t i = 0; i < body_names.size(); ++i) {
                const float* M = body_mats_flat.data() + i * 16;
                bool matched = h->links.count(body_names[i]) > 0;
                std::fprintf(stderr, "[snap]  %s '%s' pos=(%.3f,%.3f,%.3f)\n",
                             matched ? "MATCH" : "MISS ",
                             body_names[i].c_str(), M[3], M[7], M[11]);
            }
            std::fprintf(stderr, "[snap] === ARTIC LINKS (%zu, showing first 20) ===\n", h->links.size());
            int k = 0;
            for (const auto& kv : h->links) {
                if (k++ >= 20) break;
                bool in_sim = body_world.count(kv.first) > 0;
                std::fprintf(stderr, "[snap]  %s '%s'\n",
                             in_sim ? "SIM  " : "chain", kv.first.c_str());
            }
            ++printed;
        }
    }

    int cache_idx = 0;
    float3 ee_centroid = make_float3(0, 0, 0);
    int    ee_vert_count = 0;

    std::function<void(const std::string&, Mat4, bool)> walk;
    walk = [&](const std::string& link_name, Mat4 chain_world, bool under_ee) {
        // Snap to sim-computed world if this link corresponds to a sim body.
        // NOTE: Mat4::operator* miscomputes A*B as B*A for this specific call
        // pattern under MSVC /O2 (root cause not identified — possibly an
        // RVO/aliasing codegen bug). Call the free-function variant instead;
        // it is immune.
        Mat4 world_xform = chain_world;
        auto sit = body_world.find(link_name);
        bool snapped = sit != body_world.end();
        if (snapped)
            world_xform = mat4_mul_rowmajor(h->z_to_y, sit->second);

        if (!under_ee && link_name == h->ee_link_name) under_ee = true;

        if (cache_idx < (int)h->mesh_caches.size() &&
            h->mesh_caches[cache_idx].link_name == link_name)
        {
            auto& mc = h->mesh_caches[cache_idx];
            const Mat4& local_xform = h->links[link_name].visual_origin;
            const Mat4& world = world_xform;

            for (int ti = 0; ti < mc.tri_count; ++ti) {
                int idx = mc.tri_start + ti;
                if (idx >= (int)triangles_out.size()) break;

                int i0 = mc.raw.indices[ti*3];
                int i1 = mc.raw.indices[ti*3+1];
                int i2 = mc.raw.indices[ti*3+2];

                Triangle& tri = triangles_out[idx];
                tri.v0 = world.transform_point(local_xform.transform_point(mc.raw.vertices[i0]));
                tri.v1 = world.transform_point(local_xform.transform_point(mc.raw.vertices[i1]));
                tri.v2 = world.transform_point(local_xform.transform_point(mc.raw.vertices[i2]));

                if (under_ee) {
                    ee_centroid.x += tri.v0.x + tri.v1.x + tri.v2.x;
                    ee_centroid.y += tri.v0.y + tri.v1.y + tri.v2.y;
                    ee_centroid.z += tri.v0.z + tri.v1.z + tri.v2.z;
                    ee_vert_count += 3;
                }

                if (i0 < (int)mc.raw.normals.size() && i1 < (int)mc.raw.normals.size() && i2 < (int)mc.raw.normals.size()) {
                    tri.n0 = world.transform_normal(local_xform.transform_normal(mc.raw.normals[i0]));
                    tri.n1 = world.transform_normal(local_xform.transform_normal(mc.raw.normals[i1]));
                    tri.n2 = world.transform_normal(local_xform.transform_normal(mc.raw.normals[i2]));
                } else {
                    float3 e1 = tri.v1 - tri.v0;
                    float3 e2 = tri.v2 - tri.v0;
                    float3 fn = make_float3(e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z, e1.x*e2.y-e1.y*e2.x);
                    float len = sqrtf(fn.x*fn.x + fn.y*fn.y + fn.z*fn.z);
                    if (len > 1e-7f) { fn.x/=len; fn.y/=len; fn.z/=len; }
                    tri.n0 = tri.n1 = tri.n2 = fn;
                }
            }
            ++cache_idx;
        }

        auto jit = h->children_map.find(link_name);
        if (jit == h->children_map.end()) return;

        for (auto* joint : jit->second) {
            // Only bother chaining when the child isn't a sim body — children
            // that are bodies will be snapped anyway, so the composed chain
            // value is unused. Still, compute it so non-body descendants
            // (geom wrappers) inherit a reasonable parent world even if a
            // body is missing from the sim map.
            Mat4 child_chain = mat4_mul_rowmajor(world_xform, joint->origin);
            // Joint angle is irrelevant here — if the child is a body, we
            // snap; if not (geom/fixed), angle is 0 by construction.
            walk(joint->child, child_chain, under_ee);
        }
    };

    walk(h->root_link, h->z_to_y, false);

    if (ee_vert_count > 0) {
        float inv = 1.f / (float)ee_vert_count;
        h->ee_world_pos = make_float3(ee_centroid.x * inv,
                                      ee_centroid.y * inv,
                                      ee_centroid.z * inv);
    }

    return true;
}
