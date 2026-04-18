// MuJoCo CPU backend. Links against libmujoco (Apache-2.0) provided via
// FetchContent. Only compiled when MUJOCO_ENABLED is defined; the factory in
// sim_backend.cpp returns nullptr otherwise.

#ifdef MUJOCO_ENABLED

#include "sim_backend.h"

#include <mujoco/mujoco.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <unordered_map>

namespace fs = std::filesystem;

namespace mjx_helpers {

// Use C fopen — std::ifstream on Windows occasionally failed to open the file
// after a prior mj_loadXML had errored on the same path (likely a stale
// exclusive handle inside MuJoCo's resource provider).
static std::string read_text_file(const fs::path& p) {
    FILE* f = nullptr;
#ifdef _WIN32
    _wfopen_s(&f, p.wstring().c_str(), L"rb");
#else
    f = std::fopen(p.string().c_str(), "rb");
#endif
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); return {}; }
    std::string out((size_t)sz, '\0');
    size_t rd = std::fread(&out[0], 1, (size_t)sz, f);
    std::fclose(f);
    if (rd != (size_t)sz) return {};
    return out;
}

// Scan the body opening at `body_open` for a direct-child <freejoint/> or
// <joint type="free".../>. Returns true if one is already present, in which
// case we don't need to inject anything — the model is already free-base.
static size_t find_tag_end(const std::string& xml, size_t tag_open) {
    size_t q = tag_open;
    while (q < xml.size() && xml[q] != '>') {
        if (xml[q] == '"') {
            size_t cq = xml.find('"', q + 1);
            if (cq == std::string::npos) return std::string::npos;
            q = cq + 1;
        } else if (xml[q] == '\'') {
            size_t cq = xml.find('\'', q + 1);
            if (cq == std::string::npos) return std::string::npos;
            q = cq + 1;
        } else {
            ++q;
        }
    }
    return (q < xml.size()) ? q : std::string::npos;
}

static bool body_already_has_freejoint(const std::string& xml, size_t body_open_gt) {
    // Walk the body's content until we hit the matching </body> or a nested
    // <body ... — both signal we've left direct-children territory.
    size_t p = body_open_gt + 1;
    while (p < xml.size()) {
        size_t lt = xml.find('<', p);
        if (lt == std::string::npos) return false;
        if (xml.compare(lt, 7, "</body>") == 0) return false;
        if (xml.compare(lt, 6, "<body ") == 0 || xml.compare(lt, 6, "<body\t") == 0
            || xml.compare(lt, 6, "<body>") == 0 || xml.compare(lt, 6, "<body\n") == 0
            || xml.compare(lt, 6, "<body\r") == 0) {
            return false;
        }
        if (xml.compare(lt, 10, "<freejoint") == 0) return true;
        if (xml.compare(lt, 6, "<joint") == 0) {
            size_t tag_end = xml.find('>', lt);
            if (tag_end != std::string::npos) {
                std::string tag = xml.substr(lt, tag_end - lt);
                if (tag.find("type=\"free\"") != std::string::npos
                    || tag.find("type='free'") != std::string::npos) {
                    return true;
                }
            }
        }
        // Skip past this tag.
        size_t tag_end = xml.find('>', lt);
        if (tag_end == std::string::npos) return false;
        p = tag_end + 1;
    }
    return false;
}

static size_t find_matching_body_close(const std::string& xml, size_t body_open) {
    size_t p = body_open;
    int depth = 0;
    while (p < xml.size()) {
        size_t lt = xml.find('<', p);
        if (lt == std::string::npos) return std::string::npos;
        if (xml.compare(lt, 6, "<body ") == 0 || xml.compare(lt, 6, "<body\t") == 0
            || xml.compare(lt, 6, "<body>") == 0 || xml.compare(lt, 6, "<body\n") == 0
            || xml.compare(lt, 6, "<body\r") == 0) {
            ++depth;
            size_t gt = find_tag_end(xml, lt + 1);
            if (gt == std::string::npos) return std::string::npos;
            p = gt + 1;
            continue;
        }
        if (xml.compare(lt, 7, "</body>") == 0) {
            --depth;
            if (depth == 0) return lt;
            p = lt + 7;
            continue;
        }
        size_t gt = find_tag_end(xml, lt + 1);
        if (gt == std::string::npos) return std::string::npos;
        p = gt + 1;
    }
    return std::string::npos;
}

static bool body_subtree_has_joint(const std::string& xml, size_t body_open) {
    size_t body_close = find_matching_body_close(xml, body_open);
    if (body_close == std::string::npos) return false;
    size_t joint = xml.find("<joint", body_open);
    size_t freejoint = xml.find("<freejoint", body_open);
    return (joint != std::string::npos && joint < body_close)
        || (freejoint != std::string::npos && freejoint < body_close);
}

enum class InjectResult {
    NoBody,         // no <body> found under worldbody — try an include
    AlreadyFree,    // body already has a freejoint — caller should skip VFS
    Injected,       // <freejoint/> inserted into xml
};

// Inject a <freejoint/> into the articulated root body under <worldbody>.
// Prefer a top-level body whose subtree contains joints; that avoids making
// static environment bodies (ground, room, props) fall when free-base is on.
static InjectResult inject_freejoint(std::string& xml) {
    size_t wb = xml.find("<worldbody");
    if (wb == std::string::npos) return InjectResult::NoBody;
    size_t p = wb;
    size_t chosen_body = std::string::npos;
    size_t fallback_body = std::string::npos;
    while (true) {
        p = xml.find("<body", p);
        if (p == std::string::npos) break;
        char c = (p + 5 < xml.size()) ? xml[p + 5] : '\0';
        if (!(c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '>' || c == '/')) {
            p += 5;
            continue;
        }

        // Only consider direct children of worldbody.
        size_t scan = wb;
        int depth = -1;
        while (scan < p) {
            size_t lt = xml.find('<', scan);
            if (lt == std::string::npos || lt >= p) break;
            if (xml.compare(lt, 11, "<worldbody ") == 0 || xml.compare(lt, 11, "<worldbody>") == 0
                || xml.compare(lt, 11, "<worldbody\n") == 0 || xml.compare(lt, 11, "<worldbody\r") == 0
                || xml.compare(lt, 11, "<worldbody\t") == 0) {
                ++depth;
            } else if (xml.compare(lt, 6, "<body ") == 0 || xml.compare(lt, 6, "<body\t") == 0
                || xml.compare(lt, 6, "<body>") == 0 || xml.compare(lt, 6, "<body\n") == 0
                || xml.compare(lt, 6, "<body\r") == 0) {
                ++depth;
            } else if (xml.compare(lt, 7, "</body>") == 0 || xml.compare(lt, 12, "</worldbody>") == 0) {
                --depth;
            }
            size_t gt = find_tag_end(xml, lt + 1);
            if (gt == std::string::npos) return InjectResult::NoBody;
            scan = gt + 1;
        }
        if (depth != 0) {
            p += 5;
            continue;
        }

        if (fallback_body == std::string::npos) fallback_body = p;
        if (body_subtree_has_joint(xml, p)) {
            chosen_body = p;
            break;
        }
        p += 5;
    }
    if (chosen_body == std::string::npos) chosen_body = fallback_body;
    if (chosen_body == std::string::npos) return InjectResult::NoBody;

    size_t q = find_tag_end(xml, chosen_body + 1);
    if (q == std::string::npos) return InjectResult::NoBody;
    if (q > 0 && xml[q - 1] == '/') return InjectResult::NoBody;
    if (body_already_has_freejoint(xml, q)) return InjectResult::AlreadyFree;
    xml.insert(q + 1, "\n    <freejoint/>");
    return InjectResult::Injected;
}

// Pull out the first `file="X"` value from the first <include ...> tag.
static std::string first_include_file(const std::string& xml) {
    size_t p = 0;
    while ((p = xml.find("<include", p)) != std::string::npos) {
        size_t end = xml.find('>', p);
        if (end == std::string::npos) return {};
        size_t fp = xml.find("file", p);
        if (fp != std::string::npos && fp < end) {
            size_t q1 = xml.find('"', fp);
            if (q1 != std::string::npos && q1 < end) {
                size_t q2 = xml.find('"', q1 + 1);
                if (q2 != std::string::npos && q2 < end)
                    return xml.substr(q1 + 1, q2 - q1 - 1);
            }
        }
        p = end + 1;
    }
    return {};
}

} // namespace mjx_helpers

namespace {

class MujocoSim final : public ISimBackend {
public:
    MujocoSim() = default;

    ~MujocoSim() override {
        release();
    }

    bool load(const std::string& path) override {
        release();
        // Clear the global "last XML" cache — after a failed load it can
        // retain stale resource-provider state that makes the next load
        // report "resource not found" on a path that plainly exists.
        mj_freeLastXML();

        std::array<char, 1024> err{};
        std::fprintf(stderr, "[sim_mujoco] loading %s (free_base=%d)\n",
                     path.c_str(), m_free_base ? 1 : 0);
        m_model = m_free_base
            ? load_xml_with_freejoint(path, err)
            : mj_loadXML(path.c_str(), nullptr, err.data(), (int)err.size());
        if (!m_model) {
            std::fprintf(stderr, "[sim_mujoco] mj_loadXML failed: %s\n", err.data());
            return false;
        }
        if (err[0]) std::fprintf(stderr, "[sim_mujoco] load warnings: %s\n", err.data());
        m_data = mj_makeData(m_model);
        if (!m_data) {
            std::fprintf(stderr, "[sim_mujoco] mj_makeData failed\n");
            mj_deleteModel(m_model); m_model = nullptr;
            return false;
        }

        m_original_gravity[0] = m_model->opt.gravity[0];
        m_original_gravity[1] = m_model->opt.gravity[1];
        m_original_gravity[2] = m_model->opt.gravity[2];

        apply_initial_state();
        build_joint_index();
        std::fprintf(stderr, "[sim_mujoco] loaded njnt=%d nq=%d timestep=%.4f\n",
                     m_model->njnt, m_model->nq, m_model->opt.timestep);
        std::fprintf(stderr, "[sim_mujoco] movable joints=%zu, actuator-driven=%zu\n",
                     m_jnt_qposadr.size(), m_jnt_to_actuator.size());

        std::snprintf(m_name, sizeof(m_name),
                      "MuJoCo %d.%d.%d",
                      mjVERSION_HEADER / 100,
                      (mjVERSION_HEADER / 10) % 10,
                      mjVERSION_HEADER % 10);
        return true;
    }

    void step(float wall_dt) override {
        if (!m_model || !m_data) return;
        double ts = m_model->opt.timestep;
        if (ts <= 0.0) return;
        // Catch up to wall clock, capped so a stall doesn't freeze the UI.
        int n = (int)std::min(50.0, std::floor(wall_dt / ts));
        if (n < 1) n = 1;
        for (int i = 0; i < n; ++i) {
            mj_step(m_model, m_data);
            // If physics diverges (qacc_warnstart tripped), bail rather than
            // spamming NaN through the rest of the app.
            if (m_data->warning[mjWARN_BADQACC].number > 0) {
                std::fprintf(stderr, "[sim_mujoco] solver diverged — resetting\n");
                mj_resetData(m_model, m_data);
                return;
            }
        }
    }

    void reset() override {
        if (!m_model || !m_data) return;
        apply_initial_state();
    }

    void read_angles_by_name(const std::vector<std::string>& names,
                             std::vector<float>& out) override
    {
        out.assign(names.size(), std::nanf(""));
        if (!m_model || !m_data) return;
        for (size_t i = 0; i < names.size(); ++i) {
            auto it = m_jnt_qposadr.find(names[i]);
            if (it == m_jnt_qposadr.end()) continue;
            out[i] = (float)m_data->qpos[it->second];
        }
    }

    void write_angles_by_name(const std::vector<std::string>& names,
                              const std::vector<float>& values) override
    {
        if (!m_model || !m_data) return;
        size_t n = std::min(names.size(), values.size());
        for (size_t i = 0; i < n; ++i) {
            auto it = m_jnt_qposadr.find(names[i]);
            if (it == m_jnt_qposadr.end()) continue;
            m_data->qpos[it->second] = (mjtNum)values[i];
            auto itv = m_jnt_dofadr.find(names[i]);
            if (itv != m_jnt_dofadr.end())
                m_data->qvel[itv->second] = 0.0;
        }
        mj_forward(m_model, m_data);
    }

    void write_ctrl_by_joint_name(const std::vector<std::string>& names,
                                   const std::vector<float>& values) override
    {
        if (!m_model || !m_data) return;
        size_t n = std::min(names.size(), values.size());
        for (size_t i = 0; i < n; ++i) {
            auto it = m_jnt_to_actuator.find(names[i]);
            if (it == m_jnt_to_actuator.end()) continue;
            int a = it->second;
            mjtNum v = (mjtNum)values[i];
            // Respect ctrlrange if the actuator has one (actuator_ctrllimited).
            if (m_model->actuator_ctrllimited[a]) {
                mjtNum lo = m_model->actuator_ctrlrange[2*a + 0];
                mjtNum hi = m_model->actuator_ctrlrange[2*a + 1];
                if (v < lo) v = lo;
                if (v > hi) v = hi;
            }
            m_data->ctrl[a] = v;
        }
    }

    void set_free_base(bool on) override {
        m_free_base = on;   // takes effect on next load()
    }

    bool read_root_xform(float out_m16[16]) override {
        for (int i = 0; i < 16; ++i) out_m16[i] = (i % 5 == 0) ? 1.f : 0.f;
        if (!m_model || !m_data || m_root_body_id < 1 || m_root_body_id >= m_model->nbody) return false;
        const mjtNum* p = &m_data->xpos[3 * m_root_body_id];
        const mjtNum* R = &m_data->xmat[9 * m_root_body_id]; // row-major 3x3
        out_m16[0]  = (float)R[0]; out_m16[1]  = (float)R[1]; out_m16[2]  = (float)R[2]; out_m16[3]  = (float)p[0];
        out_m16[4]  = (float)R[3]; out_m16[5]  = (float)R[4]; out_m16[6]  = (float)R[5]; out_m16[7]  = (float)p[1];
        out_m16[8]  = (float)R[6]; out_m16[9]  = (float)R[7]; out_m16[10] = (float)R[8]; out_m16[11] = (float)p[2];
        out_m16[12] = 0; out_m16[13] = 0; out_m16[14] = 0; out_m16[15] = 1;
        return true;
    }

    void read_body_xforms(std::vector<std::string>& names,
                          std::vector<float>& mats_flat) override {
        names.clear();
        mats_flat.clear();
        if (!m_model || !m_data) return;
        names.reserve((size_t)m_model->nbody);
        mats_flat.reserve((size_t)m_model->nbody * 16);
        // Body 0 is the world frame (fixed at the origin). Skip it.
        for (int b = 1; b < m_model->nbody; ++b) {
            const char* nm = mj_id2name(m_model, mjOBJ_BODY, b);
            if (!nm || !*nm) continue; // must match an MJCF body name — unnamed bodies have no stable mapping
            const mjtNum* p = &m_data->xpos[3 * b];
            const mjtNum* R = &m_data->xmat[9 * b]; // row-major 3x3
            names.emplace_back(nm);
            float m[16] = {
                (float)R[0], (float)R[1], (float)R[2], (float)p[0],
                (float)R[3], (float)R[4], (float)R[5], (float)p[1],
                (float)R[6], (float)R[7], (float)R[8], (float)p[2],
                0.f,         0.f,         0.f,         1.f
            };
            mats_flat.insert(mats_flat.end(), m, m + 16);
        }
    }

    void set_gravity_enabled(bool on) override {
        if (!m_model) return;
        if (on) {
            m_model->opt.gravity[0] = m_original_gravity[0];
            m_model->opt.gravity[1] = m_original_gravity[1];
            m_model->opt.gravity[2] = m_original_gravity[2];
        } else {
            m_model->opt.gravity[0] = 0;
            m_model->opt.gravity[1] = 0;
            m_model->opt.gravity[2] = 0;
        }
    }

    const char* name() const override { return m_name; }

private:
    // Load `path` but inject a <freejoint/> at the first body encountered —
    // either in the top-level file or in its first <include>. The modified
    // text is pushed through a VFS so we never touch disk. 3.1.6 has no
    // mjSpec API; this is the least-bad workaround.
    mjModel* load_xml_with_freejoint(const std::string& path,
                                     std::array<char, 1024>& err) {
        using namespace mjx_helpers;
        std::string main_text = read_text_file(path);
        if (main_text.empty()) {
            std::snprintf(err.data(), err.size(),
                          "free-base: could not read %s", path.c_str());
            return nullptr;
        }

        // Prefer modifying the main file if it already has a body; otherwise
        // dive into the first <include>.
        std::string modified_text;
        std::string modified_name;

        auto try_inject = [&](std::string& text, const std::string& name) -> InjectResult {
            auto r = inject_freejoint(text);
            if (r == InjectResult::Injected) {
                modified_text = std::move(text);
                modified_name = name;
            }
            return r;
        };

        std::string tentative = main_text;
        auto r = try_inject(tentative, path);
        if (r == InjectResult::NoBody) {
            // Walk include chain until we find one with a <body>.
            std::string inc = first_include_file(main_text);
            while (!inc.empty()) {
                fs::path inc_path = fs::path(path).parent_path() / inc;
                std::string inc_text = read_text_file(inc_path);
                if (inc_text.empty()) {
                    std::snprintf(err.data(), err.size(),
                                  "free-base: could not read include %s",
                                  inc_path.string().c_str());
                    return nullptr;
                }
                r = try_inject(inc_text, inc);
                if (r != InjectResult::NoBody) break;
                inc = first_include_file(inc_text);
            }
            if (r == InjectResult::NoBody) {
                std::snprintf(err.data(), err.size(),
                              "free-base: no <body> in worldbody (searched main + includes)");
                return nullptr;
            }
        }

        if (r == InjectResult::AlreadyFree) {
            // Root body already has a <freejoint/> (e.g. Spot, Go1). The
            // model is already free-base; plain load is the right thing.
            std::fprintf(stderr, "[sim_mujoco] free-base: root already has a freejoint — loading as-is\n");
            return mj_loadXML(path.c_str(), nullptr, err.data(), (int)err.size());
        }

        // mjVFS is ~20MB (mjMAXVFS × mjMAXVFSNAME + bookkeeping) — heap it.
        auto vfs = std::make_unique<mjVFS>();
        mj_defaultVFS(vfs.get());
        int rc = mj_addBufferVFS(vfs.get(), modified_name.c_str(),
                                 modified_text.data(),
                                 (int)modified_text.size());
        if (rc != 0) {
            std::snprintf(err.data(), err.size(),
                          "free-base: mj_addBufferVFS rc=%d", rc);
            mj_deleteVFS(vfs.get());
            return nullptr;
        }
        mjModel* m = mj_loadXML(path.c_str(), vfs.get(),
                                err.data(), (int)err.size());
        mj_deleteVFS(vfs.get());
        return m;
    }

    void release() {
        if (m_data)  { mj_deleteData(m_data);  m_data  = nullptr; }
        if (m_model) { mj_deleteModel(m_model); m_model = nullptr; }
        m_jnt_qposadr.clear();
        m_jnt_dofadr.clear();
    }

    // Match MuJoCo's `simulate` viewer: if the model defines keyframes, the
    // first one is the authored "home" pose and that's what the viewer loads.
    // Without this, mj_makeData/mj_resetData leave qpos=qpos0 (usually all
    // zeros) — which for Menagerie arms/quadrupeds is a degenerate config
    // where actuators can't hold the body against gravity and the robot
    // visibly collapses or slams joint limits the moment physics runs.
    // Also seed ctrl from the key to match (many keyframes store ctrl too),
    // so position-controlled actuators start tracking the home pose.
    void apply_initial_state() {
        if (!m_model || !m_data) return;
        if (m_model->nkey > 0) {
            mj_resetDataKeyframe(m_model, m_data, 0);
            std::fprintf(stderr, "[sim_mujoco] reset to keyframe 0 (%s)\n",
                         mj_id2name(m_model, mjOBJ_KEY, 0)
                           ? mj_id2name(m_model, mjOBJ_KEY, 0) : "unnamed");
        } else {
            mj_resetData(m_model, m_data);
        }
        // Propagate kinematics immediately so xpos/xmat reflect the new qpos
        // before the first read_body_xforms — otherwise a newly-loaded model
        // reports stale (pre-reset) transforms for one frame.
        mj_forward(m_model, m_data);
    }

    // Map joint names (hinge/slide only — the ones our UrdfArticulation cares
    // about) to their qpos/dof index, and find a 1:1 position actuator per
    // joint when one exists (used as the PD target for articulation-driven
    // control).
    void build_joint_index() {
        m_jnt_qposadr.clear();
        m_jnt_dofadr.clear();
        m_jnt_to_actuator.clear();
        m_root_body_id = -1;
        for (int j = 0; j < m_model->njnt; ++j) {
            int type = m_model->jnt_type[j];
            if (type != mjJNT_HINGE && type != mjJNT_SLIDE) continue;
            const char* nm = mj_id2name(m_model, mjOBJ_JOINT, j);
            if (!nm) continue;
            m_jnt_qposadr[nm] = m_model->jnt_qposadr[j];
            m_jnt_dofadr[nm]  = m_model->jnt_dofadr[j];

            int b = m_model->jnt_bodyid[j];
            while (b > 0 && m_model->body_parentid[b] > 0) {
                b = m_model->body_parentid[b];
            }
            if (b > 0 && (m_root_body_id < 0 || b < m_root_body_id))
                m_root_body_id = b;
        }
        // Walk actuators; for each joint-transmission actuator, bind it to
        // the joint it drives. Tendons/sites aren't mapped here (finger pair
        // on Franka goes through a tendon — those joints stay uncontrolled).
        for (int a = 0; a < m_model->nu; ++a) {
            if (m_model->actuator_trntype[a] != mjTRN_JOINT) continue;
            int j = m_model->actuator_trnid[2*a];
            if (j < 0 || j >= m_model->njnt) continue;
            const char* nm = mj_id2name(m_model, mjOBJ_JOINT, j);
            if (!nm) continue;
            // Keep first actuator found per joint (common case: 1:1).
            m_jnt_to_actuator.emplace(nm, a);
        }
        if (m_root_body_id < 0 && m_model->nbody >= 2)
            m_root_body_id = 1;
    }

    mjModel* m_model = nullptr;
    mjData*  m_data  = nullptr;
    bool     m_free_base = false;
    mjtNum   m_original_gravity[3] = {0, 0, -9.81};
    std::unordered_map<std::string, int> m_jnt_qposadr;
    std::unordered_map<std::string, int> m_jnt_dofadr;
    std::unordered_map<std::string, int> m_jnt_to_actuator;
    int m_root_body_id = -1;
    char m_name[32] = "MuJoCo";
};

} // namespace

std::unique_ptr<ISimBackend> make_mujoco_backend() {
    return std::make_unique<MujocoSim>();
}

#endif // MUJOCO_ENABLED
