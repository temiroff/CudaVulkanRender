#include "sim_backend.h"

// Forward declarations for backend constructors (each in its own .cpp).
#ifdef MUJOCO_ENABLED
std::unique_ptr<ISimBackend> make_mujoco_backend();
#endif

std::unique_ptr<ISimBackend> make_sim_backend(SimBackendId id) {
    switch (id) {
    case SimBackendId::MuJoCo:
#ifdef MUJOCO_ENABLED
        return make_mujoco_backend();
#else
        return nullptr;
#endif
    case SimBackendId::None:
    default:
        return nullptr;
    }
}

const char* sim_backend_label(SimBackendId id) {
    switch (id) {
    case SimBackendId::None:   return "Kinematic (IK only)";
    case SimBackendId::MuJoCo: return "MuJoCo (CPU)";
    }
    return "?";
}
