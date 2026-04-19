// Auxiliary MuJoCo world for prop objects. See props_physics.h.
//
// Implementation strategy:
//   - Maintain a list of PropDesc entries (shape, size, mass, colour,
//     current pose + velocity). Each one corresponds to one free-body in
//     the compiled model.
//   - Whenever add/remove happens, mark the world dirty. On the next
//     step(), regenerate the MJCF string from the current prop list and
//     reload the model via mjVFS. Copy the last-known pose/velocity for
//     every surviving prop into the fresh qpos/qvel so in-flight motion
//     is preserved across rebuilds.
//   - After each step, write the integrated pose back into PropDesc so the
//     next rebuild (if any) resumes from the true simulated state, not
//     the add-time state.
//
// MuJoCo is natively Z-up; we configure gravity along -Y and orient the
// floor plane so its normal points along +Y. This keeps the backend in
// agreement with the rest of the renderer's world frame.

#include "props_physics.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef MUJOCO_ENABLED
#include <mujoco/mujoco.h>
#endif

namespace {

struct PropDesc {
    int   handle  = -1;
    int   shape   = 0;      // 0 = box, 1 = sphere
    float size[3] = {0,0,0}; // box: full extents; sphere: size[0] = radius
    float mass    = 1.0f;
    // Latest known state — refreshed every step.
    float pos[3]  = {0,0,0};
    float quat[4] = {1,0,0,0};
    float lvel[3] = {0,0,0};
    float avel[3] = {0,0,0};
};

} // anon

#ifndef MUJOCO_ENABLED

// ── Stub implementation: MuJoCo not compiled in ─────────────────────────────

struct PropsPhysics { int dummy; };

PropsPhysics* props_physics_create() { return nullptr; }
void          props_physics_destroy(PropsPhysics*) {}
bool          props_physics_available() { return false; }
void          props_physics_set_ground(PropsPhysics*, float) {}
int           props_physics_add_box(PropsPhysics*, float, float, float,
                                    float, float, float, float,
                                    float, float, float,
                                    float, float, float,
                                    float, float, float,
                                    float) { return -1; }
int           props_physics_add_sphere(PropsPhysics*, float, float, float,
                                       float,
                                       float, float, float,
                                       float) { return -1; }
void          props_physics_remove(PropsPhysics*, int) {}
void          props_physics_clear(PropsPhysics*) {}
void          props_physics_step(PropsPhysics*, float) {}
bool          props_physics_get_pose(PropsPhysics*, int, float[3], float[4]) { return false; }

#else

// ── Real implementation ─────────────────────────────────────────────────────

struct PropsPhysics {
    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    std::vector<PropDesc> props;
    int   next_handle = 0;
    float ground_y    = 0.f;
    bool  dirty       = true;   // needs rebuild before next step
};

// Generate the MJCF string from the current prop list.
static std::string build_mjcf(const PropsPhysics* p)
{
    // Gravity along -Y (world is Y-up). Softer contacts so bounce is visible.
    // solref negative = physical (time-constant, damping-ratio). A stiff but
    // slightly damped value gives a few perceptible bounces before the
    // cube settles.
    std::string xml;
    xml.reserve(2048 + p->props.size() * 256);
    xml +=
        "<mujoco model=\"props_world\">\n"
        "  <option gravity=\"0 -9.81 0\" integrator=\"RK4\" timestep=\"0.002\"/>\n"
        "  <default>\n"
        "    <geom friction=\"0.8 0.02 0.001\" solref=\"0.004 1\" solimp=\"0.9 0.95 0.001\" condim=\"4\"/>\n"
        "  </default>\n"
        "  <worldbody>\n";

    // Floor: MuJoCo's plane has its normal along local +Z. zaxis=\"0 1 0\"
    // rotates that so the normal points along world +Y. Size is (x, z, thick)
    // in plane-local space; we use a big patch so cubes don't escape.
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "    <geom name=\"floor\" type=\"plane\" size=\"100 100 0.1\" "
            "zaxis=\"0 1 0\" pos=\"0 %.6f 0\" rgba=\"0.4 0.4 0.4 1\"/>\n",
            p->ground_y);
        xml += buf;
    }

    for (const auto& pd : p->props) {
        char buf[1024];
        if (pd.shape == 0) {
            std::snprintf(buf, sizeof(buf),
                "    <body name=\"prop_%d\" pos=\"%.6f %.6f %.6f\">\n"
                "      <freejoint/>\n"
                "      <geom type=\"box\" size=\"%.6f %.6f %.6f\" mass=\"%.6f\" rgba=\"0.8 0.5 0.2 1\"/>\n"
                "    </body>\n",
                pd.handle,
                pd.pos[0], pd.pos[1], pd.pos[2],
                pd.size[0] * 0.5f, pd.size[1] * 0.5f, pd.size[2] * 0.5f,
                pd.mass);
        } else {
            std::snprintf(buf, sizeof(buf),
                "    <body name=\"prop_%d\" pos=\"%.6f %.6f %.6f\">\n"
                "      <freejoint/>\n"
                "      <geom type=\"sphere\" size=\"%.6f\" mass=\"%.6f\" rgba=\"0.8 0.5 0.2 1\"/>\n"
                "    </body>\n",
                pd.handle,
                pd.pos[0], pd.pos[1], pd.pos[2],
                pd.size[0],
                pd.mass);
        }
        xml += buf;
    }

    xml +=
        "  </worldbody>\n"
        "</mujoco>\n";
    return xml;
}

static void destroy_sim(PropsPhysics* p)
{
    if (p->data)  { mj_deleteData(p->data);   p->data  = nullptr; }
    if (p->model) { mj_deleteModel(p->model); p->model = nullptr; }
}

static void rebuild(PropsPhysics* p)
{
    destroy_sim(p);
    p->dirty = false;

    if (p->props.empty()) {
        // Nothing to simulate; keep model null.
        return;
    }

    std::string xml = build_mjcf(p);

    // mjVFS is ~20MB (see sim_mujoco.cpp) — must be heap-allocated or it
    // blows the stack the first time rebuild() runs.
    auto vfs = std::make_unique<mjVFS>();
    mj_defaultVFS(vfs.get());
    int add_rc = mj_addBufferVFS(vfs.get(), "props.xml", xml.data(), (int)xml.size());
    if (add_rc != 0) {
        std::fprintf(stderr, "[props_physics] mj_addBufferVFS failed (rc=%d)\n", add_rc);
        mj_deleteVFS(vfs.get());
        return;
    }

    char err[1024] = {0};
    p->model = mj_loadXML("props.xml", vfs.get(), err, (int)sizeof(err));
    mj_deleteVFS(vfs.get());

    if (!p->model) {
        std::fprintf(stderr, "[props_physics] mj_loadXML failed: %s\n", err);
        std::fprintf(stderr, "[props_physics] XML was:\n%s\n", xml.c_str());
        return;
    }
    if (err[0]) {
        std::fprintf(stderr, "[props_physics] load warnings: %s\n", err);
    }

    p->data = mj_makeData(p->model);
    if (!p->data) {
        std::fprintf(stderr, "[props_physics] mj_makeData failed\n");
        mj_deleteModel(p->model);
        p->model = nullptr;
        return;
    }

    // Seed qpos/qvel from stored PropDesc state so in-flight motion persists
    // across rebuilds (e.g. a cube mid-bounce when another cube is added).
    for (const auto& pd : p->props) {
        char name[64];
        std::snprintf(name, sizeof(name), "prop_%d", pd.handle);
        int bid = mj_name2id(p->model, mjOBJ_BODY, name);
        if (bid < 0) continue;
        int jnt = p->model->body_jntadr[bid];
        if (jnt < 0 || p->model->jnt_type[jnt] != mjJNT_FREE) continue;
        int qadr = p->model->jnt_qposadr[jnt];
        int vadr = p->model->jnt_dofadr[jnt];

        p->data->qpos[qadr + 0] = pd.pos[0];
        p->data->qpos[qadr + 1] = pd.pos[1];
        p->data->qpos[qadr + 2] = pd.pos[2];
        p->data->qpos[qadr + 3] = pd.quat[0]; // w
        p->data->qpos[qadr + 4] = pd.quat[1]; // x
        p->data->qpos[qadr + 5] = pd.quat[2]; // y
        p->data->qpos[qadr + 6] = pd.quat[3]; // z

        p->data->qvel[vadr + 0] = pd.lvel[0];
        p->data->qvel[vadr + 1] = pd.lvel[1];
        p->data->qvel[vadr + 2] = pd.lvel[2];
        p->data->qvel[vadr + 3] = pd.avel[0];
        p->data->qvel[vadr + 4] = pd.avel[1];
        p->data->qvel[vadr + 5] = pd.avel[2];
    }

    mj_forward(p->model, p->data);
}

// After mj_step: refresh PropDesc cache from the sim state so a later
// rebuild resumes exactly from where we left off.
static void refresh_prop_cache(PropsPhysics* p)
{
    if (!p->model || !p->data) return;
    for (auto& pd : p->props) {
        char name[64];
        std::snprintf(name, sizeof(name), "prop_%d", pd.handle);
        int bid = mj_name2id(p->model, mjOBJ_BODY, name);
        if (bid < 0) continue;
        int jnt = p->model->body_jntadr[bid];
        if (jnt < 0 || p->model->jnt_type[jnt] != mjJNT_FREE) continue;
        int qadr = p->model->jnt_qposadr[jnt];
        int vadr = p->model->jnt_dofadr[jnt];

        pd.pos[0]  = (float)p->data->qpos[qadr + 0];
        pd.pos[1]  = (float)p->data->qpos[qadr + 1];
        pd.pos[2]  = (float)p->data->qpos[qadr + 2];
        pd.quat[0] = (float)p->data->qpos[qadr + 3];
        pd.quat[1] = (float)p->data->qpos[qadr + 4];
        pd.quat[2] = (float)p->data->qpos[qadr + 5];
        pd.quat[3] = (float)p->data->qpos[qadr + 6];

        pd.lvel[0] = (float)p->data->qvel[vadr + 0];
        pd.lvel[1] = (float)p->data->qvel[vadr + 1];
        pd.lvel[2] = (float)p->data->qvel[vadr + 2];
        pd.avel[0] = (float)p->data->qvel[vadr + 3];
        pd.avel[1] = (float)p->data->qvel[vadr + 4];
        pd.avel[2] = (float)p->data->qvel[vadr + 5];
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

PropsPhysics* props_physics_create()
{
    auto* p = new PropsPhysics();
    return p;
}

void props_physics_destroy(PropsPhysics* p)
{
    if (!p) return;
    destroy_sim(p);
    delete p;
}

bool props_physics_available() { return true; }

void props_physics_set_ground(PropsPhysics* p, float ground_y)
{
    if (!p) return;
    if (p->ground_y != ground_y) {
        p->ground_y = ground_y;
        p->dirty = true;
    }
}

int props_physics_add_box(PropsPhysics* p,
                          float pos_x, float pos_y, float pos_z,
                          float qw, float qx, float qy, float qz,
                          float size_x, float size_y, float size_z,
                          float vx, float vy, float vz,
                          float wx, float wy, float wz,
                          float mass)
{
    if (!p) return -1;
    PropDesc pd;
    pd.handle = p->next_handle++;
    pd.shape = 0;
    pd.size[0] = size_x; pd.size[1] = size_y; pd.size[2] = size_z;
    pd.mass = (mass > 0.f) ? mass : 1.0f;
    pd.pos[0] = pos_x; pd.pos[1] = pos_y; pd.pos[2] = pos_z;
    pd.quat[0] = qw; pd.quat[1] = qx; pd.quat[2] = qy; pd.quat[3] = qz;
    pd.lvel[0] = vx; pd.lvel[1] = vy; pd.lvel[2] = vz;
    pd.avel[0] = wx; pd.avel[1] = wy; pd.avel[2] = wz;
    p->props.push_back(pd);
    p->dirty = true;
    return pd.handle;
}

int props_physics_add_sphere(PropsPhysics* p,
                             float pos_x, float pos_y, float pos_z,
                             float radius,
                             float vx, float vy, float vz,
                             float mass)
{
    if (!p) return -1;
    PropDesc pd;
    pd.handle = p->next_handle++;
    pd.shape = 1;
    pd.size[0] = radius;
    pd.mass = (mass > 0.f) ? mass : 1.0f;
    pd.pos[0] = pos_x; pd.pos[1] = pos_y; pd.pos[2] = pos_z;
    pd.quat[0] = 1.f;
    pd.lvel[0] = vx; pd.lvel[1] = vy; pd.lvel[2] = vz;
    p->props.push_back(pd);
    p->dirty = true;
    return pd.handle;
}

void props_physics_remove(PropsPhysics* p, int handle)
{
    if (!p) return;
    for (auto it = p->props.begin(); it != p->props.end(); ++it) {
        if (it->handle == handle) { p->props.erase(it); p->dirty = true; return; }
    }
}

void props_physics_clear(PropsPhysics* p)
{
    if (!p) return;
    p->props.clear();
    p->dirty = true;
}

void props_physics_step(PropsPhysics* p, float dt)
{
    if (!p || dt <= 0.f) return;
    if (p->dirty) rebuild(p);
    if (!p->model || !p->data) return;

    // Sub-step to the solver timestep so wall-clock and sim-time track.
    double ts = p->model->opt.timestep;
    if (ts <= 0.0) ts = 0.002;
    int n = (int)(dt / ts + 0.5);
    if (n < 1) n = 1;
    // Guard against pathological dt spikes (alt-tab, pause).
    if (n > 200) n = 200;
    for (int i = 0; i < n; ++i) mj_step(p->model, p->data);
    refresh_prop_cache(p);
}

bool props_physics_get_pose(PropsPhysics* p, int handle,
                            float out_pos[3], float out_quat[4])
{
    if (!p) return false;
    for (const auto& pd : p->props) {
        if (pd.handle != handle) continue;
        out_pos[0]  = pd.pos[0];
        out_pos[1]  = pd.pos[1];
        out_pos[2]  = pd.pos[2];
        out_quat[0] = pd.quat[0];
        out_quat[1] = pd.quat[1];
        out_quat[2] = pd.quat[2];
        out_quat[3] = pd.quat[3];
        return true;
    }
    return false;
}

#endif // MUJOCO_ENABLED
