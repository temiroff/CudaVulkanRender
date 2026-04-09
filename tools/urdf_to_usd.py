#!/usr/bin/env python3
"""
URDF → USD Converter for CudaVulkan Path Tracer
================================================
Parses a URDF file, loads .dae/.stl visual meshes via Assimp/trimesh,
builds a USD stage with proper kinematic hierarchy and transforms,
and outputs a .usda that the renderer can load directly.

Usage:
    python urdf_to_usd.py <urdf_path> [--output out.usda] [--joints j1=0.5 j2=-1.0 ...]

Dependencies:
    pip install trimesh numpy lxml
    Requires OpenUSD Python bindings (pxr) — already available if you built the renderer.
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh not found. Install with: pip install trimesh")
    sys.exit(1)

try:
    from pxr import Usd, UsdGeom, Gf, Sdf, Vt, UsdShade
except ImportError:
    print("ERROR: pxr (OpenUSD) not found. Make sure USD Python bindings are on PYTHONPATH.")
    sys.exit(1)


# ── URDF Parsing ─────────────────────────────────────────────────────────────

def parse_origin(elem):
    """Parse <origin rpy="..." xyz="..."/> into a 4x4 matrix."""
    if elem is None:
        return np.eye(4)
    xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]

    # Roll-pitch-yaw (XYZ extrinsic = ZYX intrinsic)
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,           cp*sr,           cp*cr  ],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def resolve_mesh_path(filename, urdf_dir):
    """Resolve package:// or relative mesh paths."""
    if filename.startswith("package://"):
        # package://franka_description/meshes/visual/link0.dae
        # Walk up from urdf_dir to find the package root
        parts = filename[len("package://"):].split("/", 1)
        pkg_name = parts[0]
        rel_path = parts[1] if len(parts) > 1 else ""

        # Search upward for a directory matching the package name
        search = Path(urdf_dir)
        for _ in range(10):
            candidate = search / pkg_name / rel_path
            if candidate.exists():
                return str(candidate)
            # Also check if we're inside the package already
            if search.name == pkg_name:
                candidate = search / rel_path
                if candidate.exists():
                    return str(candidate)
            search = search.parent

        # Fallback: try relative to urdf_dir
        return str(Path(urdf_dir) / rel_path)
    else:
        return str(Path(urdf_dir) / filename)


class URDFLink:
    def __init__(self, name):
        self.name = name
        self.visual_mesh = None      # resolved file path
        self.visual_origin = np.eye(4)


class URDFJoint:
    def __init__(self, name, jtype, parent, child, origin, axis):
        self.name = name
        self.type = jtype            # revolute, prismatic, fixed, continuous
        self.parent = parent
        self.child = child
        self.origin = origin         # 4x4 transform
        self.axis = axis             # [x, y, z]
        self.lower = 0.0
        self.upper = 0.0


def parse_urdf(urdf_path):
    """Parse URDF file into links and joints."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = str(Path(urdf_path).parent)

    links = {}
    joints = []
    children = set()

    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        link = URDFLink(name)

        visual = link_elem.find("visual")
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None:
                mesh_elem = geom.find("mesh")
                if mesh_elem is not None:
                    link.visual_mesh = resolve_mesh_path(
                        mesh_elem.get("filename"), urdf_dir
                    )
            origin_elem = visual.find("origin")
            link.visual_origin = parse_origin(origin_elem)

        links[name] = link

    for joint_elem in root.findall("joint"):
        name = joint_elem.get("name")
        jtype = joint_elem.get("type")
        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")
        origin = parse_origin(joint_elem.find("origin"))
        axis_elem = joint_elem.find("axis")
        axis = [float(v) for v in axis_elem.get("xyz").split()] if axis_elem is not None else [0, 0, 1]

        j = URDFJoint(name, jtype, parent, child, origin, axis)

        limit_elem = joint_elem.find("limit")
        if limit_elem is not None:
            j.lower = float(limit_elem.get("lower", "0"))
            j.upper = float(limit_elem.get("upper", "0"))

        joints.append(j)
        children.add(child)

    # Find root link (not a child of any joint)
    root_link = None
    for lname in links:
        if lname not in children:
            root_link = lname
            break

    return links, joints, root_link


# ── Mesh Loading ─────────────────────────────────────────────────────────────

def load_mesh(filepath):
    """Load a mesh file (.dae, .stl, .obj) and return vertices, normals, face indices."""
    if not os.path.exists(filepath):
        print(f"  WARNING: mesh not found: {filepath}")
        return None

    scene = trimesh.load(filepath, force="scene" if filepath.endswith(".dae") else None)

    # Flatten scene to single mesh
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            return None
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    if not isinstance(mesh, trimesh.Trimesh):
        return None

    return mesh


# ── USD Stage Building ───────────────────────────────────────────────────────

def sanitize_name(name):
    """Make a URDF name safe for USD prim paths."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def mat4_to_gf(m):
    """Convert numpy 4x4 to Gf.Matrix4d."""
    return Gf.Matrix4d(
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    )


def add_mesh_to_stage(stage, prim_path, mesh, xform=np.eye(4)):
    """Add a trimesh.Trimesh as a UsdGeomMesh under prim_path."""
    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

    verts = mesh.vertices.astype(np.float64)
    # Apply visual origin transform
    if not np.allclose(xform, np.eye(4)):
        ones = np.ones((len(verts), 1))
        h = np.hstack([verts, ones])
        verts = (xform @ h.T).T[:, :3]

    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    usd_mesh.CreatePointsAttr(points)

    faces = mesh.faces
    face_counts = [3] * len(faces)
    face_indices = faces.flatten().tolist()
    usd_mesh.CreateFaceVertexCountsAttr(face_counts)
    usd_mesh.CreateFaceVertexIndicesAttr(face_indices)

    # Normals
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        norms = mesh.vertex_normals
        if not np.allclose(xform[:3, :3], np.eye(3)):
            R_inv_T = np.linalg.inv(xform[:3, :3]).T
            norms = (R_inv_T @ norms.T).T
            norms = norms / (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8)
        normals = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in norms]
        usd_mesh.CreateNormalsAttr(normals)
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # Simple gray material
    mat_path = prim_path + "/material"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.7, 0.72))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.3)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(usd_mesh).Bind(material)

    return usd_mesh


def apply_joint_rotation(origin_mat, axis, angle_rad):
    """Apply a joint rotation around axis to the origin transform."""
    ax = np.array(axis, dtype=np.float64)
    ax = ax / (np.linalg.norm(ax) + 1e-12)

    c, s = math.cos(angle_rad), math.sin(angle_rad)
    t = 1 - c
    x, y, z = ax

    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])

    rot4 = np.eye(4)
    rot4[:3, :3] = R

    return origin_mat @ rot4


def build_usd(links, joints, root_link, joint_angles, output_path):
    """Build a USD stage from parsed URDF data."""
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # URDF uses meters

    # Build parent→children map
    children_map = {}  # parent_link → [(joint, child_link)]
    for j in joints:
        children_map.setdefault(j.parent, []).append(j)

    # Recursive traversal
    def build_link(link_name, parent_path):
        link = links[link_name]
        safe = sanitize_name(link_name)
        prim_path = f"{parent_path}/{safe}"

        xform = UsdGeom.Xform.Define(stage, prim_path)

        # Load and add visual mesh
        if link.visual_mesh:
            mesh = load_mesh(link.visual_mesh)
            if mesh is not None:
                mesh_path = f"{prim_path}/visual"
                add_mesh_to_stage(stage, mesh_path, mesh, link.visual_origin)
                print(f"  + {link_name}: {len(mesh.faces)} triangles")
            else:
                print(f"  - {link_name}: no mesh loaded")
        else:
            print(f"  - {link_name}: no visual geometry")

        # Process child joints
        for j in children_map.get(link_name, []):
            child_link = links[j.child]

            # Compute joint transform
            joint_xform = j.origin.copy()

            # Apply joint angle if provided
            angle = joint_angles.get(j.name, 0.0)
            if j.type in ("revolute", "continuous") and angle != 0.0:
                joint_xform = apply_joint_rotation(joint_xform, j.axis, angle)
            elif j.type == "prismatic" and angle != 0.0:
                translation = np.array(j.axis) * angle
                T = np.eye(4)
                T[:3, 3] = translation
                joint_xform = joint_xform @ T

            # Create joint xform prim
            joint_safe = sanitize_name(j.name)
            joint_path = f"{prim_path}/{joint_safe}"
            joint_xf = UsdGeom.Xform.Define(stage, joint_path)

            # Set transform
            gf_mat = mat4_to_gf(joint_xform)
            joint_xf.AddTransformOp().Set(gf_mat)

            # Recurse into child link under the joint prim
            build_link(j.child, joint_path)

    # Start from root
    robot_path = "/panda"
    robot_xf = UsdGeom.Xform.Define(stage, robot_path)
    stage.SetDefaultPrim(robot_xf.GetPrim())

    # Place root link directly under robot
    build_link(root_link, robot_path)

    stage.GetRootLayer().Save()
    print(f"\nSaved: {output_path}")
    return output_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert URDF to USD for CudaVulkan renderer")
    parser.add_argument("urdf", help="Path to .urdf file")
    parser.add_argument("--output", "-o", default=None, help="Output .usda path")
    parser.add_argument("--joints", "-j", nargs="*", default=[],
                        help="Joint angles as name=radians (e.g. panda_joint1=0.5)")
    args = parser.parse_args()

    urdf_path = os.path.abspath(args.urdf)
    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found: {urdf_path}")
        sys.exit(1)

    output = args.output or os.path.splitext(urdf_path)[0] + ".usda"

    # Parse joint angle overrides
    joint_angles = {}
    for jspec in args.joints:
        if "=" in jspec:
            name, val = jspec.split("=", 1)
            joint_angles[name] = float(val)

    print(f"URDF: {urdf_path}")
    print(f"Output: {output}")
    if joint_angles:
        print(f"Joint angles: {joint_angles}")
    print()

    links, joints, root_link = parse_urdf(urdf_path)
    print(f"Links: {len(links)}, Joints: {len(joints)}, Root: {root_link}\n")

    build_usd(links, joints, root_link, joint_angles, output)

    print(f"\nTo load in CudaVulkan: drag-and-drop or open '{output}'")


if __name__ == "__main__":
    main()
