# Inverse Kinematics Solver — Weighted Damped Least-Squares (DLS) Jacobian

Implementation: `src/urdf_loader.cpp` → `urdf_solve_dls()`
Public entry points: `urdf_solve_ik()` (drive EE tip) and `urdf_solve_ik_joint()` (drive a specific joint's origin).

## The Problem

A robot arm is a chain of joints connected by rigid links. Given a desired point (end-effector or a named joint) in world space, find joint angles that reach it — while respecting joint limits, keeping user-locked links' orientations, and avoiding the near-singularity where the arm fully straightens.

## Why Damped Least-Squares Jacobian

| Method | Verdict |
|--------|---------|
| Analytical | Exact but only for specific geometries. Rejected — URDF/MJCF rig can be any topology. |
| CCD | Simple and robust but can't handle multi-row task spaces (position + orientation constraints), has no natural null-space for secondary objectives, and looks "robotic". Kept as legacy helper (`ccd_angle_for_joint`) but unused by the public API. |
| **DLS Jacobian** | Handles an arbitrary M-row task (position + K×3 orientation rows), resolves singularities with damping λ², exposes a null space for joint-centering bias. This is the active solver. |

## Task Formulation

Each iteration solves

```
Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e
```

where:

- `q ∈ ℝᴺ` — the movable DOF subset (revolute + prismatic + continuous). Primary-locked joint excluded. Per-joint frozen joints are *kept* in the DOF set — they need to move to preserve their child link's orientation.
- `e ∈ ℝᴹ` — error stack. First 3 rows are the linear error for the driven point. Each per-joint orientation lock adds 3 angular-error rows. So `M = 3 + 3K`.
- `J ∈ ℝᴹˣᴺ` — Jacobian of the task w.r.t. `q`. Revolute column = `axis × (point − joint_pos)`; prismatic column = `axis`; angular rows use `axis` directly (prismatic contributes zero).
- `λ² = 0.0025` — damping that keeps `J Jᵀ + λ² I` invertible at singularities.

The linear system is built explicitly (`A = J Jᵀ + λ² I`, `M×M`), inverted, and applied: `y = A⁻¹ e`, `Δq = Jᵀ y`. `M` is small (typically 3 to ~12), so a plain inversion is fine.

## Per-Joint Stiffness Weighting

The base twist and shoulder pitch on a serial arm swing through the largest arcs in world space and dominate an unweighted LS solution. That produces ugly "base-first" motion. Weights discourage that:

```
w[base_revolute]     = 5.0
w[shoulder_revolute] = 2.5
w[others]            = 1.0
```

Implemented by scaling Jacobian columns by `1/√w`, solving in the weighted space, then rescaling `Δq[k] *= 1/√w[k]` on step application. The existing DLS math is unchanged. Net effect: IK prefers moving the elbow/wrist.

## Orientation Preservation (Per-Joint Locks)

If the user locks a joint's *rotation* (via `ik_locked_mask`), the lock records that link's current world rotation `R_target`. On every iteration:

```
angular_error_world(R_target, R_current)  →  ω ≈ 0.5 · vee(ΔR − ΔRᵀ)
```

These 3-vectors are appended to `e` and the corresponding Jacobian rows are added. Only joints **upstream** of (or equal to) the locked link contribute — joints downstream can't reorient it.

## End-Effector Orientation Target (Pose IK)

`urdf_solve_ik_pose(h, target_pos, target_rot_mat16, ...)` adds a 6-DOF task: both a position and a world rotation for the EE. Implementation:

- 3 extra angular rows appended to `e` and `J` at row index `ee_rot_row = 3 + 3K`.
- Error: `angular_error_world(R_target, R_current_ee)` using the EE's world transform captured from `ik_forward`'s `out_ee_xform`.
- Jacobian: every movable revolute/continuous joint in the chain is upstream of the EE, so its angular column is the world axis directly (prismatic contributes zero).
- Error clamp: `‖ω‖ ≤ max_err_ang = 0.20 rad` per iteration (same cap as per-joint locks).
- **Primary IK lock is overridden** when a rotation target is supplied. A 6-DOF pose target generally needs every joint; holding one (typically the wrist) fixed usually makes the task infeasible.

This enables arbitrary approach directions — side grasps, angled picks, re-orientation — not just top-down. Default `max_iters` is 20 (vs 10 for position-only) because the coupled position+orientation task converges more slowly.

## Reach Clamping

The solver computes the arm's straight-line max reach once (sum of fixed link lengths), then clamps the target to `0.92 × reach_max` projected along `base → target`. Consequence: the arm can never be commanded to fully straighten, which is exactly where DLS stalls and motion looks jittery.

## Null-Space Joint Centering

After the primary `Δq_task = Jᵀ A⁻¹ e`, a secondary bias pushes revolute joints toward the middle of their limits:

```
qb[k] = gain · (center[k] − angle[k])
Δq   += (I − Jᵀ A⁻¹ J) · qb     // projected onto null space of the task
```

The projection guarantees the bias can't fight the primary task. Gain is `0.08` baseline and ramps up sharply once the arm extends past 75% of its reach — so as the arm approaches singularity, joints actively fold (elbow bends) instead of committing to a straight-line solution.

This is what fixes the classic "IK only rotates the base" failure near the reach limit.

## Step Caps

Per iteration, after solving:

```
max_err_lin = 0.15 m
max_err_ang = 0.20 rad
max_dq      = 0.20 rad    (per joint, in real joint space)
```

`e` is clamped before the solve (keeps the linearization honest). `Δq` is clamped after the solve. Joint limits are enforced with `clamp_joint()` on every step.

## Termination

Iterate until `‖e_pos‖ < tolerance` **and** `max‖ω_c‖ < 0.01 rad`, up to `max_iters`. Returns `true` on convergence, `false` on timeout or singular `A` (shouldn't happen with λ² > 0).

## Public API

```cpp
// Drive the end-effector tip to target_pos (position only).
bool urdf_solve_ik(UrdfArticulation* h, float3 target_pos,
                   int max_iters = 10, float tolerance = 0.005f);

// Drive the end-effector to both target_pos and target_rot_mat16
// (row-major 4×4, top-left 3×3 used). Pass nullptr for position-only.
bool urdf_solve_ik_pose(UrdfArticulation* h, float3 target_pos,
                        const float* target_rot_mat16,
                        int max_iters = 20, float tolerance = 0.005f);

// Drive joint_idx's origin to target_pos (used by grab-drag on a mid-chain joint).
bool urdf_solve_ik_joint(UrdfArticulation* h, int joint_idx, float3 target_pos,
                         int max_iters = 6, float tolerance = 0.005f);
```

All three dispatch to the same `urdf_solve_dls(h, target_pos, target_ci, max_iters, tolerance, ee_rot_target)` core. `target_ci < 0` means "drive the EE tip"; otherwise `target_ci` is the chain index of the driven joint. `ee_rot_target` is honored only when `target_ci < 0`.

## Known Limits / Extensions

- **Position-only reach clamp.** Doesn't consider orientation feasibility — an in-reach position may still be unreachable with a given target orientation. Convergence of `urdf_solve_ik_pose` is not guaranteed at workspace edges; callers should check the return value and fall back to a pregrasp offset.
- **Orientation target replaces primary lock.** When `ee_rot_target` is set, the user's primary IK lock is ignored for that solve. This is deliberate (see "End-Effector Orientation Target") but surprising if you expect lock semantics to stack.
- **No collision avoidance.** Null-space is currently spent on joint-centering; it could host a second secondary task (e.g., distance to a repulsive obstacle field).
- **CPU-only.** `M` and `N` are small enough that GPU offload wouldn't help. Matrix inversion is `O(M³)` with `M ≤ ~18` in practice (3 position + 3 EE rotation + up to K×3 per-link locks).
