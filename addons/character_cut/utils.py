# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometry helpers for Character Cut (CharacterCut)."""

from __future__ import annotations

import bmesh
import bpy
from mathutils import Matrix, Vector


ADDON_ID = "character_cut"
DRAFT_COLLECTION = "CharacterCut_Черновик"
PARTS_COLLECTION = "CharacterCut_Детали"
CUT_PREFIX = "CC_Разрез"
CONNECTOR_PREFIX = "CC_Коннектор"
SOURCE_PREFIX = "CC_Источник"


def ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def unlink_from_all_collections(obj: bpy.types.Object) -> None:
    for col in list(obj.users_collection):
        col.objects.unlink(obj)


def link_only(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    unlink_from_all_collections(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)


def active_mesh(context) -> bpy.types.Object | None:
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None
    return obj


def scene_unit_scale_to_mm(scene: bpy.types.Scene) -> float:
    """Return multiplier that converts Blender length units to millimeters."""
    us = scene.unit_settings
    if us.system == "NONE":
        # Assume scene already authored in mm (1 BU = 1 mm) — common print workflow.
        return 1.0
    # Blender stores lengths in meters internally when metric is used.
    # unit_scale is additional scale factor (default 1.0 → 1 BU = 1 m).
    meters_per_bu = us.scale_length
    return meters_per_bu * 1000.0


def mm_to_bu(scene: bpy.types.Scene, mm: float) -> float:
    to_mm = scene_unit_scale_to_mm(scene)
    if to_mm == 0:
        return mm
    return mm / to_mm


def apply_scale_rotation(obj: bpy.types.Object) -> None:
    mat = obj.matrix_world.copy()
    loc, rot, scale = mat.decompose()
    # Apply rotation+scale into mesh, keep location.
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.transform(
        bm,
        matrix=rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale.to_4d()),
        verts=bm.verts,
    )
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj.matrix_world = Matrix.Translation(loc)


def mesh_is_manifold(obj: bpy.types.Object) -> tuple[bool, str]:
    """Basic closed-manifold check via bmesh edges."""
    if obj.type != "MESH":
        return False, "Объект не является сеткой (MESH)."
    mesh = obj.data
    if len(mesh.polygons) == 0:
        return False, "Сетка пуста."

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    non_manifold = [e for e in bm.edges if not e.is_manifold]
    wire = [e for e in bm.edges if e.is_wire]
    bm.free()

    if non_manifold or wire:
        return (
            False,
            f"Сетка не manifold: рёбер non-manifold={len(non_manifold)}, wire={len(wire)}. "
            "Нужна закрытая сплошная модель.",
        )
    return True, "Сетка закрытая (manifold)."


def duplicate_object(obj: bpy.types.Object, name: str) -> bpy.types.Object:
    new_mesh = obj.data.copy()
    new_obj = obj.copy()
    new_obj.data = new_mesh
    new_obj.name = name
    new_mesh.name = name
    for col in obj.users_collection:
        col.objects.link(new_obj)
    return new_obj


def object_bbox_world(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def ensure_boolean_exact(mod: bpy.types.BooleanModifier) -> None:
    if hasattr(mod, "solver"):
        try:
            mod.solver = "EXACT"
        except Exception:
            pass


def apply_boolean(target: bpy.types.Object, cutter: bpy.types.Object, operation: str) -> bool:
    """Apply boolean modifier; return False on failure."""
    context = bpy.context
    view_layer = context.view_layer
    view_layer.objects.active = target
    target.select_set(True)

    mod = target.modifiers.new(name="CC_Boolean", type="BOOLEAN")
    mod.operation = operation
    mod.object = cutter
    ensure_boolean_exact(mod)
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return True
    except Exception:
        if mod.name in target.modifiers:
            target.modifiers.remove(mod)
        return False


def bisect_mesh_to_parts(
    source: bpy.types.Object,
    plane_co: Vector,
    plane_no: Vector,
    name_a: str,
    name_b: str,
) -> tuple[bpy.types.Object | None, bpy.types.Object | None, str]:
    """
    Split a mesh into two closed parts using bisect.
    Returns (positive-side, negative-side, error_message).
    """
    plane_no = plane_no.normalized()
    if plane_no.length < 1e-8:
        return None, None, "Нормаль плоскости нулевая."

    parts: list[bpy.types.Object] = []
    for clear_outer, name in ((True, name_a), (False, name_b)):
        obj = duplicate_object(source, name)
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        # Transform plane into object local space.
        inv = obj.matrix_world.inverted()
        local_co = inv @ plane_co
        local_no = (inv.to_3x3() @ plane_no).normalized()

        result = bmesh.ops.bisect_plane(
            bm,
            geom=geom,
            dist=1e-6,
            plane_co=local_co,
            plane_no=local_no,
            clear_outer=clear_outer,
            clear_inner=not clear_outer,
        )
        # Fill cut faces.
        cut_edges = [e for e in result.get("geom_cut", []) if isinstance(e, bmesh.types.BMEdge)]
        if cut_edges:
            try:
                bmesh.ops.edgeloop_fill(bm, edges=cut_edges)
            except Exception:
                try:
                    bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)
                except Exception:
                    pass

        if len(bm.faces) == 0:
            bm.free()
            bpy.data.objects.remove(obj, do_unlink=True)
            return None, None, "Булева/бисекция дала пустую деталь. Проверьте положение реза."

        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        parts.append(obj)

    return parts[0], parts[1], ""


def create_connector_mesh(
    scene: bpy.types.Scene,
    shape: str,
    radius_mm: float,
    depth_mm: float,
    name: str,
) -> bpy.types.Object:
    """Create pin solid along +Z (local), centered at origin, extending ±depth/2."""
    r = mm_to_bu(scene, radius_mm)
    d = mm_to_bu(scene, depth_mm)
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()
    if shape == "BOX":
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=Vector((r * 2.0, r * 2.0, d)), verts=bm.verts)
    elif shape == "TAPERED":
        # Truncated cone: wider at socket entry (-Z), narrower tip (+Z).
        segs = 24
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=segs,
            radius1=r * 1.15,
            radius2=r * 0.75,
            depth=d,
        )
    else:  # CYLINDER
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=24,
            radius1=r,
            radius2=r,
            depth=d,
        )
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def orient_object_z_to_vector(obj: bpy.types.Object, direction: Vector, location: Vector) -> None:
    direction = direction.normalized()
    if direction.length < 1e-8:
        direction = Vector((0, 0, 1))
    quat = direction.to_track_quat("Z", "Y")
    obj.matrix_world = Matrix.Translation(location) @ quat.to_matrix().to_4x4()


def add_pin_socket(
    scene: bpy.types.Scene,
    pin_side: bpy.types.Object,
    socket_side: bpy.types.Object,
    location: Vector,
    normal: Vector,
    shape: str,
    radius_mm: float,
    depth_mm: float,
    clearance_mm: float,
) -> str:
    """
    Boolean-union pin into pin_side; difference enlarged pin from socket_side.
    Returns empty string on success, else error.
    """
    pin = create_connector_mesh(scene, shape, radius_mm, depth_mm, "CC_tmp_pin")
    orient_object_z_to_vector(pin, normal, location)

    socket_cutter = create_connector_mesh(
        scene,
        shape,
        radius_mm + clearance_mm,
        depth_mm + clearance_mm * 2.0,
        "CC_tmp_socket",
    )
    orient_object_z_to_vector(socket_cutter, normal, location)

    ok_pin = apply_boolean(pin_side, pin, "UNION")
    ok_sock = apply_boolean(socket_side, socket_cutter, "DIFFERENCE")

    bpy.data.objects.remove(pin, do_unlink=True)
    bpy.data.objects.remove(socket_cutter, do_unlink=True)

    if not ok_pin or not ok_sock:
        return (
            "Ошибка Boolean при создании штифта/гнезда. "
            "Проверьте manifold-геометрию и размер коннектора."
        )
    return ""


def delete_object_and_data(obj: bpy.types.Object | None) -> None:
    if obj is None:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and getattr(data, "users", 1) == 0:
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
