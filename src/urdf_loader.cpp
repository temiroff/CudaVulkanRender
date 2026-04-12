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

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <set>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────
//  Minimal XML parser (read-only, no validation)
//  Just enough to parse URDF and COLLADA files.
// ─────────────────────────────────────────────

struct XmlAttr {
    std::string name, value;
};

struct XmlNode {
    std::string                tag;
    std::string                text;       // inner text (non-element content)
    std::vector<XmlAttr>       attrs;
    std::vector<XmlNode>       children;

    const XmlNode* child(const std::string& t) const {
        for (auto& c : children) if (c.tag == t) return &c;
        return nullptr;
    }

    std::vector<const XmlNode*> children_with_tag(const std::string& t) const {
        std::vector<const XmlNode*> out;
        for (auto& c : children) if (c.tag == t) out.push_back(&c);
        return out;
    }

    std::string attr(const std::string& name, const std::string& def = "") const {
        for (auto& a : attrs) if (a.name == name) return a.value;
        return def;
    }
};

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

static XmlNode parse_xml(const std::string& xml_str) {
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

static std::string read_file_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ─────────────────────────────────────────────
//  Math: 4x4 matrix, RPY rotation
// ─────────────────────────────────────────────

struct Mat4 {
    float m[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

    static Mat4 identity() { Mat4 r; return r; }

    static Mat4 from_rpy_xyz(float roll, float pitch, float yaw,
                              float x, float y, float z) {
        float cr = cosf(roll),  sr = sinf(roll);
        float cp = cosf(pitch), sp = sinf(pitch);
        float cy = cosf(yaw),   sy = sinf(yaw);
        Mat4 T;
        T.m[0][0] = cy*cp;  T.m[0][1] = cy*sp*sr - sy*cr;  T.m[0][2] = cy*sp*cr + sy*sr;  T.m[0][3] = x;
        T.m[1][0] = sy*cp;  T.m[1][1] = sy*sp*sr + cy*cr;  T.m[1][2] = sy*sp*cr - cy*sr;  T.m[1][3] = y;
        T.m[2][0] = -sp;    T.m[2][1] = cp*sr;              T.m[2][2] = cp*cr;              T.m[2][3] = z;
        T.m[3][0] = 0;      T.m[3][1] = 0;                  T.m[3][2] = 0;                  T.m[3][3] = 1;
        return T;
    }

    Mat4 operator*(const Mat4& b) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                r.m[i][j] = 0;
                for (int k = 0; k < 4; ++k)
                    r.m[i][j] += m[i][k] * b.m[k][j];
            }
        return r;
    }

    float3 transform_point(float3 p) const {
        return make_float3(
            m[0][0]*p.x + m[0][1]*p.y + m[0][2]*p.z + m[0][3],
            m[1][0]*p.x + m[1][1]*p.y + m[1][2]*p.z + m[1][3],
            m[2][0]*p.x + m[2][1]*p.y + m[2][2]*p.z + m[2][3]);
    }

    float3 transform_normal(float3 n) const {
        // Normals transform by upper 3x3 (assumes orthonormal rotation)
        float3 r = make_float3(
            m[0][0]*n.x + m[0][1]*n.y + m[0][2]*n.z,
            m[1][0]*n.x + m[1][1]*n.y + m[1][2]*n.z,
            m[2][0]*n.x + m[2][1]*n.y + m[2][2]*n.z);
        float len = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        if (len > 1e-7f) { r.x /= len; r.y /= len; r.z /= len; }
        return r;
    }
};

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

struct DaeMaterial {
    float4 diffuse  = make_float4(0.7f, 0.7f, 0.7f, 1.0f);
    float  specular_intensity = 0.25f;
    float  shininess = 0.0f;
};

struct RawMesh {
    std::vector<float3> vertices;
    std::vector<float3> normals;    // per-vertex
    std::vector<int>    indices;    // triangle indices (3 per face)
    std::vector<int>    mat_ids;    // per-triangle material index into dae_materials
    std::vector<DaeMaterial> dae_materials;
};

static bool load_stl(const std::string& path, RawMesh& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char header[80];
    f.read(header, 80);

    uint32_t num_tris = 0;
    f.read(reinterpret_cast<char*>(&num_tris), 4);
    if (num_tris == 0 || num_tris > 10000000) return false;

    // Each STL triangle: 12 bytes normal + 3×12 bytes vertices + 2 bytes attr = 50 bytes
    out.vertices.reserve(num_tris * 3);
    out.normals.reserve(num_tris * 3);
    out.indices.reserve(num_tris * 3);

    for (uint32_t i = 0; i < num_tris; ++i) {
        float data[12]; // normal(3) + v0(3) + v1(3) + v2(3)
        f.read(reinterpret_cast<char*>(data), 48);
        uint16_t attr;
        f.read(reinterpret_cast<char*>(&attr), 2);

        float3 fn = make_float3(data[0], data[1], data[2]);
        float3 v0 = make_float3(data[3], data[4], data[5]);
        float3 v1 = make_float3(data[6], data[7], data[8]);
        float3 v2 = make_float3(data[9], data[10], data[11]);

        int base = (int)out.vertices.size();
        out.vertices.push_back(v0);
        out.vertices.push_back(v1);
        out.vertices.push_back(v2);
        out.normals.push_back(fn);
        out.normals.push_back(fn);
        out.normals.push_back(fn);
        out.indices.push_back(base);
        out.indices.push_back(base + 1);
        out.indices.push_back(base + 2);
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

static bool load_dae(const std::string& path, RawMesh& out) {
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
//  URDF data structures
// ─────────────────────────────────────────────

struct URDFLink {
    std::string name;
    std::string visual_mesh_path;   // resolved filesystem path
    Mat4        visual_origin;
};

struct URDFJoint {
    std::string name;
    std::string type;   // revolute, prismatic, fixed, continuous
    std::string parent;
    std::string child;
    Mat4        origin;
    float3      axis = {0, 0, 1};
};

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

static bool load_link_mesh(const std::string& mesh_path, RawMesh& raw) {
    if (mesh_path.empty() || !fs::exists(mesh_path)) return false;

    auto ext = fs::path(mesh_path).extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);

    if (ext == ".stl") return load_stl(mesh_path, raw);
    if (ext == ".dae") return load_dae(mesh_path, raw);

    std::cerr << "[urdf_loader] unsupported mesh format: " << ext << '\n';
    return false;
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
// ─────────────────────────────────────────────

// Cached mesh data per link (topology + raw verts for re-posing)
struct LinkMeshCache {
    std::string link_name;
    RawMesh     raw;
    int         tri_start;   // index into the flat Triangle array
    int         tri_count;
    int         obj_id;
    // Per-triangle material indices (relative to link_mat_base)
    std::vector<int> mat_ids;
    int         link_mat_base;
};

struct UrdfArticulation {
    // Parsed URDF data
    std::unordered_map<std::string, URDFLink> links;
    std::vector<URDFJoint>                    joints;
    std::string                               root_link;
    std::unordered_map<std::string, std::vector<const URDFJoint*>> children_map;

    // Articulated joints (excludes fixed)
    std::vector<UrdfJointInfo> joint_infos;
    // Map from joint name → index into joint_infos
    std::unordered_map<std::string, int> joint_name_to_idx;

    // Cached mesh data per link
    std::vector<LinkMeshCache> mesh_caches;

    // Axis correction
    Mat4 z_to_y;

    // End-effector: deepest leaf link (cached on open)
    std::string ee_link_name;
    float3 ee_world_pos = {0, 0, 0};  // updated by urdf_repose

    int total_tris = 0;
};

static Mat4 axis_angle_rotation(float3 axis, float angle)
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

UrdfArticulation* urdf_articulation_open(const std::string& path,
                                          const std::vector<Triangle>& initial_tris)
{
    auto* h = new UrdfArticulation;

    // Re-parse URDF (lightweight — just XML, no mesh loading)
    if (!parse_urdf(path, h->links, h->joints, h->root_link)) {
        delete h;
        return nullptr;
    }

    // Build children map
    for (auto& j : h->joints)
        h->children_map[j.parent].push_back(&j);

    // Axis correction
    h->z_to_y = Mat4::identity();
    h->z_to_y.m[0][0] =  1; h->z_to_y.m[0][1] =  0; h->z_to_y.m[0][2] =  0;
    h->z_to_y.m[1][0] =  0; h->z_to_y.m[1][1] =  0; h->z_to_y.m[1][2] =  1;
    h->z_to_y.m[2][0] =  0; h->z_to_y.m[2][1] = -1; h->z_to_y.m[2][2] =  0;

    // Build joint info list (only movable joints)
    for (auto& j : h->joints) {
        if (j.type == "fixed") continue;
        UrdfJointInfo ji{};
        snprintf(ji.name, sizeof(ji.name), "%s", j.name.c_str());
        if (j.type == "revolute")       ji.type = 0;
        else if (j.type == "prismatic") ji.type = 1;
        else if (j.type == "continuous") ji.type = 2;
        else ji.type = 3;

        // Parse limits from URDF joints
        // (limits were stored during parse_urdf but not kept — re-parse)
        ji.lower = 0; ji.upper = 0; ji.angle = 0;
        h->joint_name_to_idx[j.name] = (int)h->joint_infos.size();
        h->joint_infos.push_back(ji);
    }

    // Re-parse to get limits (parse_urdf doesn't store them in URDFJoint currently)
    {
        std::string xml_str = read_file_text(path);
        XmlNode doc = parse_xml(xml_str);
        if (doc.tag != "robot") {
            auto* robot = find_node(doc, "robot");
            if (robot) doc = *robot;
        }
        for (auto& jn : doc.children_with_tag("joint")) {
            std::string name = jn->attr("name");
            auto it = h->joint_name_to_idx.find(name);
            if (it == h->joint_name_to_idx.end()) continue;
            auto* limit = jn->child("limit");
            if (limit) {
                h->joint_infos[it->second].lower = std::atof(limit->attr("lower", "0").c_str());
                h->joint_infos[it->second].upper = std::atof(limit->attr("upper", "0").c_str());
            }
        }
    }

    // Cache mesh data per link by re-loading meshes
    // Walk the kinematic chain and record which triangles belong to each link
    int tri_cursor = 0;
    std::function<void(const std::string&)> cache_link;
    cache_link = [&](const std::string& link_name) {
        auto it = h->links.find(link_name);
        if (it == h->links.end()) return;
        const URDFLink& link = it->second;

        if (!link.visual_mesh_path.empty() && fs::exists(link.visual_mesh_path)) {
            LinkMeshCache mc;
            mc.link_name = link_name;
            mc.tri_start = tri_cursor;

            if (load_link_mesh(link.visual_mesh_path, mc.raw) && !mc.raw.vertices.empty()) {
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

    // Find end-effector: deepest leaf link, preferring names without "finger"/"tip"
    {
        std::string best_leaf;
        int best_depth = -1;
        std::function<void(const std::string&, int)> find_ee;
        find_ee = [&](const std::string& link_name, int depth) {
            auto jit = h->children_map.find(link_name);
            bool has_children = (jit != h->children_map.end() && !jit->second.empty());
            if (!has_children) {
                // Leaf link — prefer non-finger/tip links, or deepest overall
                bool is_finger = (link_name.find("finger") != std::string::npos ||
                                  link_name.find("tip") != std::string::npos);
                if (depth > best_depth || (!is_finger && depth >= best_depth)) {
                    best_leaf = link_name;
                    best_depth = depth;
                }
            }
            if (has_children) {
                for (auto* joint : jit->second)
                    find_ee(joint->child, depth + 1);
            }
        };
        find_ee(h->root_link, 0);
        h->ee_link_name = best_leaf;
    }

    // Compute initial ee position from the last mesh cache's triangle centroid
    if (!h->mesh_caches.empty()) {
        auto& last_mc = h->mesh_caches.back();
        float3 sum = make_float3(0, 0, 0);
        int count = 0;
        for (int ti = 0; ti < last_mc.tri_count; ++ti) {
            int idx = last_mc.tri_start + ti;
            if (idx >= (int)initial_tris.size()) break;
            const Triangle& t = initial_tris[idx];
            sum.x += t.v0.x + t.v1.x + t.v2.x;
            sum.y += t.v0.y + t.v1.y + t.v2.y;
            sum.z += t.v0.z + t.v1.z + t.v2.z;
            count += 3;
        }
        if (count > 0) {
            float inv = 1.f / (float)count;
            h->ee_world_pos = make_float3(sum.x * inv, sum.y * inv, sum.z * inv);
        }
    }

    std::cerr << "[urdf_articulation] opened: " << h->joint_infos.size()
              << " joints, " << h->mesh_caches.size() << " mesh caches, "
              << h->total_tris << " triangles, ee=" << h->ee_link_name
              << " pos=(" << h->ee_world_pos.x << "," << h->ee_world_pos.y
              << "," << h->ee_world_pos.z << ")\n";

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
static float3 ik_forward(UrdfArticulation* h,
                         const std::vector<IKChainEntry>& chain,
                         std::vector<float3>& joint_pos,
                         std::vector<float3>& joint_axis)
{
    joint_pos.resize(chain.size());
    joint_axis.resize(chain.size());

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
    }

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

// Helper: map movable joint indices from chain
static void ik_movable_indices(UrdfArticulation* h,
                               const std::vector<IKChainEntry>& chain,
                               std::vector<int>& movable)
{
    movable.clear();
    UrdfJointInfo* joints = h->joint_infos.data();
    for (int ci = 0; ci < (int)chain.size(); ci++)
        if (chain[ci].ji >= 0 && (joints[chain[ci].ji].type == 0 || joints[chain[ci].ji].type == 2))
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

bool urdf_solve_ik(UrdfArticulation* h, float3 target_pos,
                   int max_iters, float tolerance)
{
    if (!h) return false;

    std::vector<IKChainEntry> chain;
    if (!ik_collect_chain(h, chain) || chain.empty()) return false;

    UrdfJointInfo* joints = h->joint_infos.data();
    const float PI = 3.14159265f;

    std::vector<float3> jpos, jaxis;

    for (int iter = 0; iter < max_iters; iter++) {
        float3 ee = ik_forward(h, chain, jpos, jaxis);

        float3 diff = make_float3(target_pos.x - ee.x, target_pos.y - ee.y, target_pos.z - ee.z);
        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
        if (dist < tolerance) return true;

        // CCD: iterate joints from tip to base
        for (int ci = (int)chain.size() - 1; ci >= 0; ci--) {
            int ji = chain[ci].ji;
            if (ji < 0) continue;
            UrdfJointInfo& info = joints[ji];
            if (info.type != 0 && info.type != 2) continue;

            ee = ik_forward(h, chain, jpos, jaxis);
            float a = ccd_angle_for_joint(jpos[ci], jaxis[ci], ee, target_pos, 0.5f);
            info.angle += a;
            clamp_joint(info, PI);
        }
    }

    float3 ee = ik_forward(h, chain, jpos, jaxis);
    float3 diff = make_float3(target_pos.x - ee.x, target_pos.y - ee.y, target_pos.z - ee.z);
    return sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z) < tolerance;
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

    UrdfJointInfo* joints = h->joint_infos.data();
    const float PI = 3.14159265f;
    std::vector<float3> jpos, jaxis;

    int target_ci = movable[joint_idx];

    for (int iter = 0; iter < max_iters; iter++) {
        ik_forward(h, chain, jpos, jaxis);
        float3 cur = jpos[target_ci];

        float3 diff = make_float3(target_pos.x - cur.x, target_pos.y - cur.y, target_pos.z - cur.z);
        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
        if (dist < tolerance) return true;

        // CCD on joints [0..joint_idx) only
        for (int mi = joint_idx - 1; mi >= 0; mi--) {
            int ci = movable[mi];
            UrdfJointInfo& info = joints[chain[ci].ji];

            ik_forward(h, chain, jpos, jaxis);
            float a = ccd_angle_for_joint(jpos[ci], jaxis[ci],
                                          jpos[target_ci], target_pos, 0.5f);
            info.angle += a;
            clamp_joint(info, PI);
        }
    }

    ik_forward(h, chain, jpos, jaxis);
    float3 cur = jpos[target_ci];
    float3 diff = make_float3(target_pos.x - cur.x, target_pos.y - cur.y, target_pos.z - cur.z);
    return sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z) < tolerance;
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

    // Track the last link's vertex centroid as end-effector position
    float3 last_link_centroid = make_float3(0, 0, 0);
    int    last_link_vert_count = 0;

    std::function<void(const std::string&, Mat4)> repose_link;
    repose_link = [&](const std::string& link_name, Mat4 world_xform) {
        if (cache_idx < (int)h->mesh_caches.size() &&
            h->mesh_caches[cache_idx].link_name == link_name)
        {
            auto& mc = h->mesh_caches[cache_idx];
            const Mat4& local_xform = h->links[link_name].visual_origin;
            const Mat4& world = world_xform;

            // Reset centroid tracking for this link (overwrite with later links)
            last_link_centroid = make_float3(0, 0, 0);
            last_link_vert_count = 0;

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

                // Accumulate for ee centroid
                last_link_centroid.x += tri.v0.x + tri.v1.x + tri.v2.x;
                last_link_centroid.y += tri.v0.y + tri.v1.y + tri.v2.y;
                last_link_centroid.z += tri.v0.z + tri.v1.z + tri.v2.z;
                last_link_vert_count += 3;

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

                repose_link(joint->child, joint_xform);
            }
        }
    };

    repose_link(h->root_link, h->z_to_y);

    // Cache ee position as centroid of the last link's mesh
    if (last_link_vert_count > 0) {
        float inv = 1.f / (float)last_link_vert_count;
        h->ee_world_pos = make_float3(last_link_centroid.x * inv,
                                      last_link_centroid.y * inv,
                                      last_link_centroid.z * inv);
    }

    return true;
}

void urdf_articulation_close(UrdfArticulation* h)
{
    delete h;
}
