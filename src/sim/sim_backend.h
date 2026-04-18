#pragma once
// ISimBackend — pluggable physics simulation interface.
//
// The viewer owns rendering; a backend owns physics state. The main loop calls
// step() to advance time, then reads joint angles back into the URDF
// articulation so the existing repose/BVH pipeline renders the current pose.
//
// Backends are independent — e.g. CPU MuJoCo for interactive viewing vs. a
// GPU-batched backend (MJX/Isaac) for training. Add a backend by
// implementing this interface and extending make_sim_backend().

#include <memory>
#include <string>
#include <vector>

enum class SimBackendId {
    None,       // Kinematic only (existing IK-driven behavior)
    MuJoCo,     // In-process CPU MuJoCo
    // MJX, Bullet, Isaac, ...
};

class ISimBackend {
public:
    virtual ~ISimBackend() = default;

    // Load a scene/robot from a file path. Returns false on failure.
    // The same MJCF the renderer consumes should work here unchanged.
    virtual bool load(const std::string& path) = 0;

    // Advance the simulation by wall_dt seconds. Internally the backend may
    // sub-step at its native timestep.
    virtual void step(float wall_dt) = 0;

    // Reset all state to t=0 (initial qpos from the model).
    virtual void reset() = 0;

    // Read current joint angles, one per requested joint name. Names that the
    // backend doesn't recognize are filled with NaN so the caller can detect
    // mismatches. Out vector is resized to names.size().
    virtual void read_angles_by_name(const std::vector<std::string>& names,
                                     std::vector<float>& out) = 0;

    // Overwrite backend joint positions (used for teleporting during UI drag
    // or to seed sim state on activation).
    virtual void write_angles_by_name(const std::vector<std::string>& names,
                                      const std::vector<float>& values) = 0;

    // Drive position-controlled actuators: for each joint name, set the ctrl
    // of its associated actuator (if any) to the target position. Joints
    // without a 1:1 position actuator are silently skipped — dynamics will
    // keep them at their default spring/damper equilibrium.
    virtual void write_ctrl_by_joint_name(const std::vector<std::string>& names,
                                          const std::vector<float>& values) = 0;

    // Gravity toggle — convenience wrapper so UI can flip between free-fall
    // and zero-g without touching XML.
    virtual void set_gravity_enabled(bool on) = 0;

    // Read the root body's world transform (Z-up, row-major 4x4) into
    // out_m16. Returns true if the backend wrote a transform. Used to feed
    // freejoint motion (drop / tumble) into the renderer's articulation so
    // the visual base tracks the simulated base. Identity for backends that
    // can't report a root transform.
    virtual bool read_root_xform(float out_m16[16]) {
        for (int i = 0; i < 16; ++i) out_m16[i] = (i % 5 == 0) ? 1.f : 0.f;
        return false;
    }

    // Read world transforms for every named body in the simulation (Z-up,
    // row-major 4x4 per body, stored as 16 contiguous floats in mats_flat).
    // names.size() == mats_flat.size() / 16. The world body (id 0) is
    // skipped. Consumers use this to drive the renderer's per-link world
    // transforms directly from the sim, bypassing any re-implemented FK —
    // this is the only way to guarantee the rendered pose exactly matches
    // what the sim computed.
    virtual void read_body_xforms(std::vector<std::string>& names,
                                  std::vector<float>& mats_flat) {
        names.clear();
        mats_flat.clear();
    }

    // Request a freejoint be injected at the root body on the *next* load().
    // Menagerie manipulators are fixed-base by default; enabling this lets the
    // root body fall / tumble so the scene behaves like a standalone rigid
    // body sim. No-op if the backend doesn't support it. Takes effect on
    // reload; does not retroactively modify an already-loaded model.
    virtual void set_free_base(bool on) { (void)on; }

    // Human-readable backend name ("MuJoCo 3.2.7", ...)
    virtual const char* name() const = 0;
};

// Factory. Returns nullptr if the backend isn't compiled in (e.g. MuJoCo
// requested but MUJOCO_ENABLED is not defined).
std::unique_ptr<ISimBackend> make_sim_backend(SimBackendId id);

// Short identifier for the UI dropdown. Stable across builds.
const char* sim_backend_label(SimBackendId id);
