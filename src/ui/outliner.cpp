#include "outliner.h"
#include <imgui.h>
#include <map>
#include <string>
#include <cstdio>

// ── Hierarchy node ─────────────────────────────────────────────────────────

struct HNode {
    std::string  label;
    std::string  full_path;
    int          obj_idx  = -1;   // index into mesh_objects (-1 = group)
    std::vector<int> leaf_objs;   // all descendant obj indices (populated post-build)
    std::map<std::string, HNode> children;
};

static void insert_path(HNode& root, const std::string& path, int idx)
{
    HNode* cur = &root;
    size_t pos = (!path.empty() && path[0] == '/') ? 1 : 0;
    std::string accum;
    for (;;) {
        size_t sep = path.find('/', pos);
        std::string seg = path.substr(pos, sep == std::string::npos ? sep : sep - pos);
        if (seg.empty()) break;
        accum += '/' + seg;
        HNode& child = cur->children[seg];
        if (child.label.empty()) { child.label = seg; child.full_path = accum; }
        cur = &child;
        if (sep == std::string::npos) break;
        pos = sep + 1;
    }
    cur->obj_idx = idx;
}

// Bottom-up: fill leaf_objs for every node so group clicks know their coverage.
static void collect_leaves(HNode& node)
{
    node.leaf_objs.clear();
    if (node.obj_idx >= 0)
        node.leaf_objs.push_back(node.obj_idx);
    for (auto& [k, child] : node.children) {
        collect_leaves(child);
        for (int i : child.leaf_objs)
            node.leaf_objs.push_back(i);
    }
}

// ── Recursive draw ──────────────────────────────────────────────────────────

static void draw_hnode(
    const HNode&                         node,
    const std::vector<MeshObject>&       objs,
    int&                                 sel_sphere,
    int&                                 sel_mesh,
    std::unordered_set<int>&             multi_sel,
    bool&                                changed)
{
    for (auto& [key, child] : node.children) {
        bool is_leaf = child.children.empty();

        // Determine selection state for this node
        bool any_sel = false, all_sel = true;
        for (int i : child.leaf_objs) {
            if (multi_sel.count(i)) any_sel = true;
            else                    all_sel = false;
        }
        if (child.leaf_objs.empty()) all_sel = false;
        bool is_sel = all_sel || (is_leaf && any_sel);

        // Determine hidden state
        bool all_hidden = !child.leaf_objs.empty();
        bool any_hidden = false;
        for (int i : child.leaf_objs) {
            if (i < (int)objs.size() && objs[i].hidden) any_hidden = true;
            else                                          all_hidden = false;
        }

        // Style: grey out fully hidden nodes, dim partially hidden
        if (all_hidden)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.45f, 0.45f, 0.6f));
        else if (any_hidden)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.75f, 0.75f, 0.8f));

        ImGuiTreeNodeFlags flags =
            ImGuiTreeNodeFlags_SpanFullWidth |
            ImGuiTreeNodeFlags_OpenOnArrow;
        if (is_leaf)
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (is_sel)
            flags |= ImGuiTreeNodeFlags_Selected;

        char id_buf[512];
        snprintf(id_buf, sizeof(id_buf), "%s##%s",
                 child.label.c_str(), child.full_path.c_str());

        bool open = ImGui::TreeNodeEx(id_buf, flags);

        if (all_hidden || any_hidden)
            ImGui::PopStyleColor();

        // Click (not on the expand arrow): select this node's subtree
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            multi_sel.clear();
            for (int i : child.leaf_objs)
                multi_sel.insert(i);
            // Primary = first leaf (used for gizmo / silhouette)
            sel_mesh   = child.leaf_objs.empty() ? -1 : child.leaf_objs.front();
            sel_sphere = -1;
            changed    = true;
        }

        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", child.full_path.c_str());

        if (open && !is_leaf) {
            draw_hnode(child, objs, sel_sphere, sel_mesh, multi_sel, changed);
            ImGui::TreePop();
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

bool outliner_draw(
    const std::vector<Sphere>&     spheres,
    const std::vector<MeshObject>& mesh_objects,
    int&                           selected_sphere,
    int&                           selected_mesh_obj,
    std::unordered_set<int>&       multi_sel)
{
    // Rebuild tree only when the object list changes
    static size_t s_last_size = SIZE_MAX;
    static HNode  s_tree;
    if (mesh_objects.size() != s_last_size) {
        s_tree      = HNode{};
        s_last_size = mesh_objects.size();
        for (int i = 0; i < (int)mesh_objects.size(); ++i)
            insert_path(s_tree, mesh_objects[i].name, i);
        collect_leaves(s_tree);
    }

    bool changed = false;

    ImGui::SetNextWindowSize(ImVec2(300, 520), ImGuiCond_FirstUseEver);
    ImGui::Begin("Outliner");

    // ── Mesh hierarchy ──────────────────────────────────────────────
    if (!mesh_objects.empty()) {
        char header[64];
        snprintf(header, sizeof(header), "Scene  (%d objects)##scene_root",
                 (int)mesh_objects.size());

        bool scene_open = ImGui::TreeNodeEx(header,
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanFullWidth);

        if (scene_open) {
            draw_hnode(s_tree, mesh_objects,
                       selected_sphere, selected_mesh_obj, multi_sel, changed);
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    // ── Spheres ──────────────────────────────────────────────────────
    int vis_spheres = 0;
    for (auto& s : spheres) if (s.radius <= 100.f) ++vis_spheres;

    if (vis_spheres > 0) {
        bool sp_open = ImGui::TreeNodeEx("##spheres",
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanFullWidth,
            "Spheres  (%d)", vis_spheres);

        if (sp_open) {
            int di = 0;
            for (int i = 0; i < (int)spheres.size(); ++i) {
                if (spheres[i].radius > 100.f) continue;
                const char* type = "?";
                switch (spheres[i].mat.type) {
                    case MatType::Lambertian: type = "Lambertian"; break;
                    case MatType::Metal:      type = "Metal";      break;
                    case MatType::Dielectric: type = "Glass";      break;
                    case MatType::Emissive:   type = "Emissive";   break;
                }
                char lbl[80];
                snprintf(lbl, sizeof(lbl), "Sphere %d  [%s]##sph%d", di, type, i);
                bool is_sel = (selected_sphere == i);
                ImGuiTreeNodeFlags f =
                    ImGuiTreeNodeFlags_Leaf   |
                    ImGuiTreeNodeFlags_SpanFullWidth |
                    ImGuiTreeNodeFlags_NoTreePushOnOpen |
                    (is_sel ? ImGuiTreeNodeFlags_Selected : 0);
                ImGui::TreeNodeEx(lbl, f);
                if (ImGui::IsItemClicked()) {
                    selected_sphere   = i;
                    selected_mesh_obj = -1;
                    multi_sel.clear();
                    changed = true;
                }
                ++di;
            }
            ImGui::TreePop();
        }
    }

    ImGui::End();
    return changed;
}
