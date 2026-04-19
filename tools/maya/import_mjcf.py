"""
MJCF -> Maya importer. Paste into a Maya Python session (via the maya MCP
`execute_code`) to define `import_mjcf(path, ...)`, then call it.

Scale contract: respects the CURRENT scene linear unit. MJCF values (meters)
are converted on the fly: translates multiplied by a meter-to-scene factor,
imported STL/OBJ meshes scaled by the same factor so raw vertex data (also
meters) displays at correct physical size. No scene settings are touched —
unit, grid, camera clip planes are left as the user has them.
"""
import os
import math
import xml.etree.ElementTree as ET

import maya.cmds as cmds
from maya.api.OpenMaya import MQuaternion


# ── Unit handling ────────────────────────────────────────────────────────────

# Maya currentUnit strings -> meters-per-unit.
_UNIT_METERS = {
    "mm":        0.001,
    "millimeter": 0.001,
    "cm":        0.01,
    "centimeter": 0.01,
    "m":         1.0,
    "meter":     1.0,
    "km":        1000.0,
    "kilometer": 1000.0,
    "in":        0.0254,
    "inch":      0.0254,
    "ft":        0.3048,
    "foot":      0.3048,
    "yd":        0.9144,
    "yard":      0.9144,
    "mi":        1609.344,
    "mile":      1609.344,
}


def _meter_to_scene_factor():
    """Return the multiplier that converts a value in meters into the current
    scene's linear unit. e.g. scene=cm -> 100.0, scene=m -> 1.0."""
    u = cmds.currentUnit(q=True, linear=True)
    mpu = _UNIT_METERS.get(u, 1.0)
    if mpu <= 0:
        return 1.0
    return 1.0 / mpu


# ── MJCF parsing ─────────────────────────────────────────────────────────────

def _load_mjcf(path):
    """Parse an MJCF file and follow any `<include>` directives. Returns a
    single merged ElementTree root (a synthetic <mujoco> element)."""
    path = os.path.abspath(path).replace("\\", "/")
    base = os.path.dirname(path)
    tree = ET.parse(path)
    root = tree.getroot()

    merged = ET.Element("mujoco")
    _merge_into(merged, root, base)
    return merged, base


def _merge_into(dst, src, base_dir):
    for child in list(src):
        if child.tag == "include":
            f = child.get("file")
            if not f:
                continue
            inc_path = os.path.join(base_dir, f).replace("\\", "/")
            sub_tree = ET.parse(inc_path)
            _merge_into(dst, sub_tree.getroot(), os.path.dirname(inc_path))
        else:
            dst.append(child)


def _parse_vec(s, default):
    return [float(x) for x in s.split()] if s else list(default)


def _collect_geom_defaults(root):
    """Walk <default> tree and build a map of class name -> dict of <geom>
    attribute defaults. MJCF defaults are nested, so child classes inherit
    from their parent's <geom>. Returns {class_name: {attr: value, ...}}
    including an implicit "main" entry for the top-level <default>."""
    result = {}

    def walk(node, inherited):
        cls = node.get("class") or "main"
        merged = dict(inherited)
        g = node.find("geom")
        if g is not None:
            for k, v in g.attrib.items():
                merged[k] = v
        result[cls] = merged
        for child in node.findall("default"):
            walk(child, merged)

    for top in root.findall("default"):
        walk(top, {})
    return result


def _resolve_geom(g, defaults):
    """Return a dict-like object that provides geom attributes, falling back
    to <default class=...> when the geom itself omits an attribute."""
    cls = g.get("class")
    # Explicit "childclass" / "main" fallback per MJCF semantics is complex;
    # we cover the common case: a named class overrides, otherwise use "main".
    base = dict(defaults.get("main", {}))
    if cls and cls in defaults:
        base.update(defaults[cls])
    base.update({k: v for k, v in g.attrib.items() if v is not None})
    return base


def _collect_assets(root, base_dir):
    """Returns (meshes, materials).
    meshes:    name -> absolute file path
    materials: name -> [r,g,b,a]
    """
    meshdir = "assets"
    for c in root.findall("compiler"):
        if c.get("meshdir"):
            meshdir = c.get("meshdir")
    mesh_base = os.path.join(base_dir, meshdir).replace("\\", "/")

    meshes = {}
    materials = {}
    for asset in root.findall("asset"):
        for me in asset.findall("mesh"):
            fn = me.get("file")
            if not fn:
                continue
            nm = me.get("name") or os.path.splitext(os.path.basename(fn))[0]
            meshes[nm] = os.path.join(mesh_base, fn).replace("\\", "/")
        for m in asset.findall("material"):
            rgba = _parse_vec(m.get("rgba"), (0.8, 0.8, 0.8, 1.0))
            while len(rgba) < 4:
                rgba.append(1.0)
            materials[m.get("name")] = rgba

    return meshes, materials


# ── Maya helpers ─────────────────────────────────────────────────────────────

_SHADER_CACHE = {}


def _quat_to_euler_deg(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    w, x, y, z = w / n, x / n, y / n, z / n
    e = MQuaternion(x, y, z, w).asEulerRotation()
    return (math.degrees(e.x), math.degrees(e.y), math.degrees(e.z))


def _set_trs(node, pos, quat, factor):
    rx, ry, rz = _quat_to_euler_deg(*quat)
    cmds.setAttr(node + ".translate",
                 pos[0] * factor, pos[1] * factor, pos[2] * factor)
    cmds.setAttr(node + ".rotate", rx, ry, rz)


def _get_shader(name, rgba):
    if name in _SHADER_CACHE:
        return _SHADER_CACHE[name]
    sh = cmds.shadingNode("lambert", asShader=True, name=name + "_mat")
    cmds.setAttr(sh + ".color", rgba[0], rgba[1], rgba[2], type="double3")
    if rgba[3] < 1.0:
        t = 1.0 - rgba[3]
        cmds.setAttr(sh + ".transparency", t, t, t, type="double3")
    sg = cmds.sets(empty=True, renderable=True, noSurfaceShader=True,
                   name=sh + "SG")
    cmds.connectAttr(sh + ".outColor", sg + ".surfaceShader", force=True)
    _SHADER_CACHE[name] = (sh, sg)
    return _SHADER_CACHE[name]


def _get_collision_shader():
    """Translucent red lambert for collision visualization."""
    if "__collision__" in _SHADER_CACHE:
        return _SHADER_CACHE["__collision__"]
    sh = cmds.shadingNode("lambert", asShader=True, name="mjcf_collision_mat")
    cmds.setAttr(sh + ".color", 1.0, 0.2, 0.2, type="double3")
    cmds.setAttr(sh + ".transparency", 0.65, 0.65, 0.65, type="double3")
    sg = cmds.sets(empty=True, renderable=True, noSurfaceShader=True,
                   name=sh + "SG")
    cmds.connectAttr(sh + ".outColor", sg + ".surfaceShader", force=True)
    _SHADER_CACHE["__collision__"] = (sh, sg)
    return _SHADER_CACHE["__collision__"]


def _primitive_from_geom(attrs, name, factor):
    """Build a Maya primitive for a non-mesh collision <geom>. `attrs` is a
    dict with MJCF defaults already merged in. Returns the transform name or
    None if the type is unsupported. Sizes follow MuJoCo conventions (meters):
      box:      size = (hx, hy, hz)          — half-extents
      sphere:   size[0] = radius
      capsule:  size[0] = radius, size[1] = half-length (along local Z)
      cylinder: size[0] = radius, size[1] = half-length (along local Z)
    """
    gtype = attrs.get("type")
    size = _parse_vec(attrs.get("size"), ())
    if gtype == "box" and len(size) >= 3:
        w = 2.0 * size[0] * factor
        h = 2.0 * size[1] * factor
        d = 2.0 * size[2] * factor
        return cmds.polyCube(w=w, h=h, d=d, name=name)[0]
    if gtype == "sphere" and len(size) >= 1:
        r = size[0] * factor
        return cmds.polySphere(r=r, name=name)[0]
    if gtype in ("cylinder", "capsule") and len(size) >= 2:
        r = size[0] * factor
        hh = 2.0 * size[1] * factor  # full height
        # MuJoCo cylinders extend along local Z; Maya polyCylinder is Y-axis
        # by default. Use axis=(0,0,1) so half-length runs along Z.
        node = cmds.polyCylinder(r=r, h=hh, axis=(0, 0, 1), name=name)[0]
        return node
    return None


def _load_stl_plugin():
    for p in ("stlTranslator", "stlImport"):
        try:
            if not cmds.pluginInfo(p, q=True, loaded=True):
                cmds.loadPlugin(p)
        except Exception:
            pass


def _import_mesh_file(abs_path):
    ext = os.path.splitext(abs_path)[1].lower()
    before = set(cmds.ls(assemblies=True))
    if ext == ".stl":
        cmds.file(abs_path, i=True, type="STLImport", ignoreVersion=True,
                  preserveReferences=False, mergeNamespacesOnClash=True,
                  namespace=":")
    elif ext == ".obj":
        cmds.file(abs_path, i=True, type="OBJ", ignoreVersion=True,
                  preserveReferences=False, mergeNamespacesOnClash=True,
                  namespace=":")
    else:
        cmds.warning("unsupported mesh extension: " + abs_path)
        return None
    new = list(set(cmds.ls(assemblies=True)) - before)
    return new[0] if new else None


# ── Main importer ────────────────────────────────────────────────────────────

def import_mjcf(mjcf_path, root_name=None, add_floor=True, add_key_light=True,
                add_collisions=True):
    """
    Parse an MJCF and build it in the CURRENT Maya scene. Scene unit, grid,
    and camera clip planes are NOT modified — the importer reads the current
    linear unit and converts MJCF meter values to match.

    Args:
        mjcf_path: absolute path to the .xml
        root_name: top-level Maya group name (defaults to <file_stem>_root)
        add_floor: create a plane if MJCF declares a floor geom
        add_key_light: add a directional key light if MJCF declares one
        add_collisions: also import class="collision" geoms (mesh + box +
            sphere + cylinder/capsule) into a display layer that is hidden
            and templated (non-selectable) by default

    Returns:
        str: the root group name (e.g. 'SO101_root')
    """
    _load_stl_plugin()
    factor = _meter_to_scene_factor()

    if root_name is None:
        stem = os.path.splitext(os.path.basename(mjcf_path))[0]
        root_name = stem + "_root"

    merged, base = _load_mjcf(mjcf_path)
    meshes, materials = _collect_assets(merged, base)
    geom_defaults = _collect_geom_defaults(merged)

    _SHADER_CACHE.clear()

    # Root: MuJoCo is Z-up, Maya is Y-up.
    root = cmds.group(empty=True, name=root_name)
    cmds.setAttr(root + ".rotateX", -90)

    worldbody = merged.find("worldbody")
    if worldbody is None:
        cmds.warning("MJCF has no <worldbody>")
        return root

    if add_floor:
        for g in worldbody.findall("geom"):
            if g.get("type") == "plane":
                w = 4.0 * factor  # 4m in scene units
                floor = cmds.polyPlane(w=w, h=w, sx=20, sy=20,
                                       name=(g.get("name") or "floor"))[0]
                floor = cmds.parent(floor, root)[0]
                _, sg = _get_shader("groundplane", [0.15, 0.25, 0.35, 1.0])
                cmds.sets(floor, e=True, forceElement=sg)
                break

    if add_key_light:
        for lt in worldbody.findall("light"):
            if lt.get("directional", "false") == "true":
                dl = cmds.directionalLight(name="mj_key_light")
                cmds.parent(dl, root)
                cmds.setAttr(dl + ".rotate", -90, 0, 0)
                break

    collision_nodes = []

    def walk(body, parent):
        name = body.get("name") or "body"
        pos = _parse_vec(body.get("pos"), (0, 0, 0))
        quat = _parse_vec(body.get("quat"), (1, 0, 0, 0))

        grp = cmds.group(empty=True, name=name)
        grp = cmds.parent(grp, parent)[0]
        _set_trs(grp, pos, quat, factor)

        for i, g in enumerate(body.findall("geom")):
            attrs = _resolve_geom(g, geom_defaults)
            cls = g.get("class", "")
            is_collision = "collision" in cls
            if is_collision and not add_collisions:
                continue
            if cls and not is_collision and cls != "visual":
                continue

            gtype = attrs.get("type")
            mesh_name = attrs.get("mesh")
            gname = attrs.get("name") or ""

            # Mesh geoms: accept explicit `type="mesh"` OR unspecified type
            # when a `mesh=` asset is named (MJCF defaults, e.g. Panda, put
            # `type="mesh"` on the <default> geom and omit on individuals).
            if mesh_name and (gtype is None or gtype == "mesh"):
                if mesh_name not in meshes:
                    cmds.warning("unknown mesh asset: " + str(mesh_name))
                    continue
                imported = _import_mesh_file(meshes[mesh_name])
                if not imported:
                    continue
                # Rename BEFORE parenting — dodges body/mesh name collisions
                # (e.g. body 'moving_jaw_so101_v1' + same-named STL).
                suffix = "col" if is_collision else "geom"
                unique = "{}__{}_{}{}".format(name, mesh_name, suffix, i)
                imported = cmds.rename(imported, unique)
                imported_full = cmds.ls(imported, long=True)[0]
                node = cmds.parent(imported_full, grp)[0]

                _set_trs(node,
                         _parse_vec(attrs.get("pos"), (0, 0, 0)),
                         _parse_vec(attrs.get("quat"), (1, 0, 0, 0)),
                         factor)
                # Scale the mesh itself so its meter-space vertex data displays
                # at correct physical size in the current scene unit.
                if factor != 1.0:
                    cmds.setAttr(node + ".scale", factor, factor, factor)

                if is_collision:
                    _, sg = _get_collision_shader()
                    shapes = cmds.listRelatives(node, ad=True, type="mesh",
                                                fullPath=True) or []
                    if shapes:
                        cmds.sets(shapes, e=True, forceElement=sg)
                    collision_nodes.append(node)
                else:
                    mat = attrs.get("material")
                    if mat and mat in materials:
                        _, sg = _get_shader(mat, materials[mat])
                        shapes = cmds.listRelatives(node, ad=True, type="mesh",
                                                    fullPath=True) or []
                        if shapes:
                            cmds.sets(shapes, e=True, forceElement=sg)
                continue

            # Primitive collision geoms (box/sphere/cylinder/capsule). Skip
            # visual-class primitives; MJCF visuals are almost always meshes,
            # and primitives here would need per-type shader handling anyway.
            if is_collision and gtype in ("box", "sphere", "cylinder", "capsule"):
                prim_name = "{}__{}_{}{}".format(
                    name, (gname or gtype), "col", i)
                prim = _primitive_from_geom(attrs, prim_name, factor)
                if not prim:
                    continue
                node = cmds.parent(prim, grp)[0]
                _set_trs(node,
                         _parse_vec(attrs.get("pos"), (0, 0, 0)),
                         _parse_vec(attrs.get("quat"), (1, 0, 0, 0)),
                         factor)
                _, sg = _get_collision_shader()
                shapes = cmds.listRelatives(node, ad=True, type="mesh",
                                            fullPath=True) or []
                if shapes:
                    cmds.sets(shapes, e=True, forceElement=sg)
                collision_nodes.append(node)

        for cb in body.findall("body"):
            walk(cb, grp)

    for b in worldbody.findall("body"):
        walk(b, root)

    # Collisions → hidden, templated display layer so the user can toggle
    # them on for inspection without cluttering the default view.
    if add_collisions and collision_nodes:
        layer = root_name + "_collisions"
        if cmds.objExists(layer):
            try:
                cmds.delete(layer)
            except Exception:
                pass
        layer = cmds.createDisplayLayer(name=layer, empty=True, noRecurse=True)
        cmds.editDisplayLayerMembers(layer, collision_nodes, noRecurse=True)
        cmds.setAttr(layer + ".visibility", 0)
        cmds.setAttr(layer + ".displayType", 2)  # 2 = Reference (non-selectable)

    cmds.select(root, replace=True)
    try:
        cmds.viewFit(all=True)
    except Exception:
        pass
    cmds.select(clear=True)

    print("[mjcf] imported {} -> {} (scene unit: {}, factor: {:.3f})".format(
        mjcf_path, root, cmds.currentUnit(q=True, linear=True), factor))
    return root


# ── UI ───────────────────────────────────────────────────────────────────────

_UI_WIN = "mjcfImporterWin"


def show_ui():
    """Open the MJCF Importer window in Maya. File-pick an .xml, set options,
    click Import. Runs `import_mjcf` with the current scene's unit — no
    settings are changed."""
    if cmds.window(_UI_WIN, exists=True):
        cmds.deleteUI(_UI_WIN)

    win = cmds.window(_UI_WIN, title="MJCF Importer",
                      widthHeight=(460, 240), sizeable=True)
    form = cmds.columnLayout(adjustableColumn=True, rowSpacing=6,
                             columnAttach=("both", 10), columnOffset=("both", 4))

    cmds.text(label="Import a MuJoCo MJCF (.xml) robot into the current scene.",
              align="left")
    cmds.text(label="Scene unit is read live — no Maya settings are changed.",
              align="left", font="smallObliqueLabelFont")
    cmds.separator(style="in", height=6)

    path_field = cmds.textFieldButtonGrp(
        label="MJCF file",
        text="",
        buttonLabel="Browse…",
        columnWidth3=(70, 260, 80),
        adjustableColumn=2)

    root_field = cmds.textFieldGrp(
        label="Root group",
        placeholderText="(auto: <filename>_root)",
        columnWidth2=(70, 340),
        adjustableColumn=2)

    floor_cb = cmds.checkBox(label="Create ground plane if present", value=True)
    light_cb = cmds.checkBox(label="Create directional key light if present",
                             value=True)
    col_cb = cmds.checkBox(
        label="Import collisions (hidden display layer, non-selectable)",
        value=True)

    cmds.separator(style="in", height=6)
    status = cmds.text(label="Ready.", align="left")

    def _browse(*_):
        # Default to the last used folder or the user's IsaacSim path
        default_dir = ""
        current = cmds.textFieldButtonGrp(path_field, q=True, text=True)
        if current and os.path.isfile(current):
            default_dir = os.path.dirname(current)
        elif os.path.isdir(r"F:/PROJECTS/IsaacSim/exts/isaacsim.asset.importer.urdf/data/urdf/robots"):
            default_dir = r"F:/PROJECTS/IsaacSim/exts/isaacsim.asset.importer.urdf/data/urdf/robots"
        picked = cmds.fileDialog2(
            fileMode=1,  # existing file
            caption="Select MJCF (.xml)",
            fileFilter="MuJoCo XML (*.xml);;All Files (*.*)",
            startingDirectory=default_dir or None,
            dialogStyle=2)
        if picked:
            p = picked[0].replace("\\", "/")
            cmds.textFieldButtonGrp(path_field, e=True, text=p)

    def _do_import(*_):
        path = cmds.textFieldButtonGrp(path_field, q=True, text=True).strip()
        if not path or not os.path.isfile(path):
            cmds.text(status, e=True, label="⚠ Pick a valid .xml file first.")
            return
        rn = cmds.textFieldGrp(root_field, q=True, text=True).strip() or None
        af = cmds.checkBox(floor_cb, q=True, value=True)
        al = cmds.checkBox(light_cb, q=True, value=True)
        ac = cmds.checkBox(col_cb, q=True, value=True)
        try:
            r = import_mjcf(path, root_name=rn, add_floor=af, add_key_light=al,
                            add_collisions=ac)
            unit = cmds.currentUnit(q=True, linear=True)
            cmds.text(status, e=True,
                      label=u"✓ Imported → {}  (scene unit: {})".format(r, unit))
        except Exception as e:
            cmds.text(status, e=True, label="✗ " + str(e))
            raise

    cmds.textFieldButtonGrp(path_field, e=True, buttonCommand=_browse)
    cmds.rowLayout(numberOfColumns=2, adjustableColumn=1,
                   columnAttach=(1, "both", 0))
    cmds.text(label="")
    cmds.button(label="Import", height=32, width=120, command=_do_import)
    cmds.setParent("..")

    cmds.showWindow(win)
    return win
