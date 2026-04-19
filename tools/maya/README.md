# Maya tools

## `import_mjcf.py` — MuJoCo MJCF → Maya importer

Imports a MuJoCo `.xml` (MJCF) robot description into Maya as a nested body
hierarchy with materials, matching the physical scale of the CUDA/Vulkan app
so any extra parts authored in Maya line up 1:1 after export.

### What it builds
- A transform for every `<body>`, parented to match the MJCF tree, with local
  `pos` and `quat` applied.
- Every `class="visual"` (or unclassed) mesh geom imported from disk (STL/OBJ
  via `<compiler meshdir>`), materials applied from `<asset>/<material rgba>`.
- Every `class="collision"` geom (including primitives — `box` / `sphere` /
  `cylinder` / `capsule` — from MJCF `size`/`pos`) imported into a hidden,
  non-selectable display layer `<root>_collisions` (translucent red). Toggle
  it on in the Display Layer Editor to double-check collision shells.
- `<default>` class inheritance resolved, so geoms that leave out `type` /
  `size` / `pos` (e.g. Franka Panda's fingertip pads) get the values from
  their class default.
- Optional ground plane + directional key light when the MJCF declares them.
- Top-level `rotateX = -90` so MuJoCo Z-up renders as Maya Y-up.

### Scale contract
The importer **respects the current scene linear unit** — it does NOT touch
`currentUnit`, grid, or camera clip planes. MJCF is meters by convention.
The script reads `currentUnit(q=True, linear=True)` and scales:

| Scene unit | factor |
|---|---|
| m  | 1.0 |
| cm | 100.0 |
| mm | 1000.0 |
| in | 39.3701 |
| ft | 3.2808 |

Every `<body>` / `<geom>` `pos` is multiplied by `factor` on setAttr, and
every imported mesh transform is scaled by `factor` so its raw meter-space
vertex data displays at correct physical size. Body groups keep `scale = 1`,
so extra parts authored in the current scene unit and parented under any
body group track it without extra scaling.

When exporting USD back to the CUDA/Vulkan app: the app reads
`metersPerUnit` from the USD stage, so a Maya scene in cm → USD with
`metersPerUnit=0.01` → the app sizes it correctly.

### Use it

Maya Script Editor → Python tab:

```python
exec(open(r"F:/PROJECTS/NVDIA/CudaVulkan/tools/maya/import_mjcf.py").read(),
     globals())

# Programmatic:
import_mjcf(r"F:/PROJECTS/.../scene.xml", root_name="panda_root")

# Or UI:
show_ui()
```

`show_ui()` opens a window with a file browser, optional root name, and
toggles for floor / directional light / collision import. Running `show_ui()`
again replaces the previous window.

### Adding custom parts
Body groups are scale = 1, so create your part, parent it under the target
body group, and set its translate in scene units (meters if scene is `m`,
cm if scene is `cm`). It follows the body.

### Known gotcha
STL imports land at the world root and can collide with a same-named body
group (e.g. `moving_jaw_so101_v1`). The importer renames every incoming mesh
to `<body>__<mesh>_<kind><i>` **before** parenting. Do not remove that step.
