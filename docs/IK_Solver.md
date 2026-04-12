# Inverse Kinematics Solver — CCD (Cyclic Coordinate Descent)

## The Problem

A robot arm is a chain of joints connected by rigid links. Given a desired end-effector (tip) position, find the joint angles that reach it.

## Why CCD

| Method | Pros | Cons |
|--------|------|------|
| **Analytical** | Exact, fast | Only works for specific robot geometries (6-DOF with known structure). Can't generalize. |
| **Jacobian** | Physically accurate, smooth | Needs matrix inversion, can be singular (arm fully stretched = divide by zero), complex to implement |
| **CCD** | Works for any chain, handles joint limits naturally, simple to implement, stable | Not physically optimal, can look "robotic" |

CCD was chosen because:
1. The URDF could be any robot — 3 joints, 7 joints, whatever. CCD doesn't care.
2. Joint limits are trivial — just clamp after each step.
3. No matrix math — no Jacobian, no pseudoinverse, no singularity handling.
4. Real-time friendly — each iteration is just trig on each joint.

## Algorithm

### Step 1: Collect the Chain

`ik_collect_chain()` walks the URDF kinematic tree from root to end-effector, recording every joint along the path (including fixed joints for transform propagation).

```
root → joint0 → joint1 → joint2 → ... → ee_link
```

### Step 2: Forward Kinematics

`ik_forward()` computes every joint's world-space position and rotation axis by walking the chain and multiplying 4x4 transforms:

```
xform = z_to_y_correction
for each joint in chain:
    xform = xform * joint.origin           // joint frame
    record joint_pos = xform * (0,0,0)     // world position
    record joint_axis = xform * joint.axis  // world axis
    xform = xform * rotation(axis, angle)   // apply current angle
ee_pos = xform * (0,0,0)                   // end-effector position
```

This is the same transform chain used by `urdf_repose()` for rendering, ensuring consistency.

### Step 3: CCD Iteration (10 iterations, tip-to-base)

For each joint, working from the tip toward the base:

```
         joint_pos ●─────────────── ● ee (current position)
                    \
                     \  we want ee here
                      \
                       ● target
```

**a)** Get the joint's world position and rotation axis from the latest FK.

**b)** Compute two vectors from the joint origin:
- `to_ee` = current ee position - joint position
- `to_target` = target position - joint position

**c)** Project both vectors onto the plane perpendicular to the joint axis. A revolute joint can only rotate around its axis, so the component along the axis is irrelevant:
```
projected = vector - dot(vector, axis) * axis
```

**d)** Compute the angle between the two projected vectors:
```
angle = acos(clamp(dot(p_ee, p_target) / (|p_ee| * |p_target|), -1, 1))
```

**e)** Determine sign (CW vs CCW) using the cross product:
```
sign = dot(cross(p_ee, p_target), axis) > 0 ? +1 : -1
```

**f)** Apply 0.5 damping — only rotate halfway:
```
angle *= sign * 0.5
```
Without damping, the base joint might swing the whole arm past the target, then the next joint overcorrects, causing oscillation.

**g)** Clamp to joint limits:
- Revolute: clamp to [lower, upper]
- Continuous: wrap to [-PI, PI]

**h)** Recompute FK — this joint changed, so the ee position changed. The next joint sees the updated state.

### Convergence

Each iteration gets roughly 50% closer (due to 0.5 damping):
- After 10 iterations: ~99.9% convergence
- Tolerance check: if ee is within 5mm of target, stop early

## Per-Joint Grab Points

`urdf_solve_ik_joint(handle, joint_idx, target)` runs the same CCD algorithm but with a different scope:

- Only adjusts joints `[0 .. joint_idx)` (before the target joint)
- Tries to bring `joint_idx`'s world position toward the drag target
- Joints after `joint_idx` are passengers — they follow passively

This is like grabbing a chain in the middle and lifting. The joints before the grab point rotate to reach; the joints after just hang.

## Why Tip-to-Base Order

**Base-to-tip** (wrong): The base joint makes a huge swing moving everything downstream. The next joint tries to correct, but it also moves everything after it. Chaotic, oscillating.

**Tip-to-base** (correct): The tip joint makes a small local adjustment. The next joint up makes a slightly larger correction. The base joint makes the final gross adjustment. Each joint builds on the previous correction. Stable convergence.

## Why Not FABRIK?

FABRIK (Forward And Backward Reaching IK) is another popular simple solver. It alternately pulls the chain from the tip backward, then from the base forward. It's faster per iteration but treats joints as **ball joints** — it doesn't naturally handle specific rotation axes or joint limits.

For a robot with constrained rotation axes (e.g., joint 1 rotates around Z, joint 2 around Y), CCD is the right choice because it respects the axis constraint at every step. FABRIK would need post-hoc projection onto valid axis rotations, which adds complexity and can break convergence.

## Implementation Files

- **IK solver**: `src/urdf_loader.cpp` — `urdf_solve_ik()`, `urdf_solve_ik_joint()`, `ik_forward()`, `ik_collect_chain()`
- **IK gizmo**: `src/main.cpp` — XYZ move gizmo at end-effector, per-joint grab points
- **FK ee position**: `urdf_fk_ee_pos()`, `urdf_fk_ee_transform()` — consistent with solver internals
- **Cached ee position**: `urdf_end_effector_pos()` — mesh centroid from last `urdf_repose()`, used for display

## Key Design Decisions

1. **FK recomputed after every joint change** inside the CCD loop. Without this, the solver sees stale ee positions and converges to wrong solutions.

2. **Persistent `ik_target`** in the gizmo code. Mouse deltas accumulate onto the target (not the ee position), preventing a feedback loop where ee movement amplifies gizmo delta.

3. **`urdf_fk_ee_pos()` for gizmo snap** instead of `urdf_end_effector_pos()`. The solver uses `ik_forward()` internally, so the gizmo target must match that reference frame. Using the mesh centroid (from `urdf_end_effector_pos`) would cause a small offset/jump on drag start.

4. **Separate `urdf_solve_ik_joint()`** for grab points rather than modifying the main solver. Simpler, no interaction between ee target and grab target — they're independent solves.
