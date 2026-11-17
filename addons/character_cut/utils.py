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


def newell_normal(points: list[Vector]) -> Vector:
    """Polygon normal via Newell's method."""
    n = Vector((0.0, 0.0, 0.0))
    count = len(points)
    if count < 3:
        return Vector((0.0, 0.0, 1.0))
    for i in range(count):
        a = points[i]
        b = points[(i + 1) % count]
        n.x += (a.y - b.y) * (a.z + b.z)
        n.y += (a.z - b.z) * (a.x + b.x)
        n.z += (a.x - b.x) * (a.y + b.y)
    if n.length < 1e-12:
        # Fallback from first triangle.
        n = (points[1] - points[0]).cross(points[2] - points[0])
    if n.length < 1e-12:
        return Vector((0.0, 0.0, 1.0))
    return n.normalized()


def close_polyline(points: list[Vector], eps: float = 1e-5) -> list[Vector]:
    """Return open ring (no duplicate last) suitable for a closed polygon."""
    if len(points) < 3:
        return list(points)
    pts = [p.copy() for p in points]
    if (pts[0] - pts[-1]).length <= eps:
        pts = pts[:-1]
    return pts


def create_halfspace_prism(
    points_world: list[Vector],
    normal: Vector,
    extent_positive: float,
    extent_negative: float = 0.05,
    name: str = "CC_tmp_halfspace",
) -> bpy.types.Object | None:
    """
    Build a localized prism from a closed polyline.
    Occupies the polygon extruded along ±normal — used as Boolean half-space
    so only geometry inside the path band is separated (not an infinite plane).
    """
    poly = close_polyline(points_world)
    if len(poly) < 3:
        return None
    normal = normal.normalized()
    if normal.length < 1e-8:
        return None

    bottom = [p - normal * extent_negative for p in poly]
    top = [p + normal * extent_positive for p in poly]
    n = len(poly)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    verts = [v.to_tuple() for v in bottom] + [v.to_tuple() for v in top]
    faces: list[tuple[int, ...]] = []
    # Bottom (reversed winding for outward normal roughly -N)
    faces.append(tuple(range(n - 1, -1, -1)))
    # Top
    faces.append(tuple(range(n, 2 * n)))
    # Sides
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))

    mesh.from_pydata(verts, [], faces)
    mesh.update()
    # Triangulate for more reliable booleans.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj["character_cut_role"] = "HALFSPACE"
    return obj


def create_path_guide_object(
    points_world: list[Vector],
    name: str,
    closed: bool = True,
) -> bpy.types.Object:
    """Visible editable polyline/polygon guide stored as a mesh."""
    poly = close_polyline(points_world)
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    if len(poly) < 2:
        mesh.from_pydata([p.to_tuple() for p in poly], [], [])
    else:
        edges = [(i, i + 1) for i in range(len(poly) - 1)]
        faces: list[tuple[int, ...]] = []
        if closed and len(poly) >= 3:
            edges.append((len(poly) - 1, 0))
            faces.append(tuple(range(len(poly))))
        mesh.from_pydata([p.to_tuple() for p in poly], edges, faces)
        if faces:
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.to_mesh(mesh)
            bm.free()
    mesh.update()
    obj.display_type = "WIRE"
    obj.show_in_front = True
    obj.hide_render = True
    obj["character_cut_role"] = "PATH"
    obj["character_cut_closed"] = closed
    return obj


def path_points_from_object(obj: bpy.types.Object) -> list[Vector]:
    if obj is None or obj.type != "MESH":
        return []
    return [obj.matrix_world @ v.co for v in obj.data.vertices]


def split_mesh_by_path_loop(
    source: bpy.types.Object,
    points_world: list[Vector],
    name_pos: str,
    name_neg: str,
    extent_positive: float | None = None,
    extent_negative: float = 0.05,
    extrude_normal: Vector | None = None,
) -> tuple[bpy.types.Object | None, bpy.types.Object | None, Vector, Vector, str]:
    """
    Split source with a localized half-space built from a closed surface path.

    The prism only spans the polyline footprint (e.g. around an arm), so a waist
    path will not slice a hanging limb elsewhere — unlike an infinite plane.

    ``extrude_normal`` (connector +Z) points toward the side kept in INTERSECT
    (positive / pin-side volume). Point it along the limb distal direction.
    Returns (part_positive, part_negative, center, normal, error).
    """
    poly = close_polyline(points_world)
    if len(poly) < 3:
        return None, None, Vector(), Vector(), "Нужен замкнутый путь (≥3 точки)."

    plane_n = newell_normal(poly)
    center = sum((p for p in poly), Vector()) / len(poly)

    if extrude_normal is not None and extrude_normal.length > 1e-8:
        normal = extrude_normal.normalized()
        # Keep extrusion roughly aligned with the path plane; if nearly parallel
        # to the polygon, fall back to Newell so the prism still has volume.
        if abs(normal.dot(plane_n)) < 0.25:
            normal = plane_n
    else:
        normal = plane_n

    mn, mx = object_bbox_world(source)
    diag = (mx - mn).length
    if extent_positive is None:
        extent_positive = max(diag * 0.75, 1.0)

    half = create_halfspace_prism(
        poly,
        normal,
        extent_positive=extent_positive,
        extent_negative=extent_negative,
        name="CC_tmp_path_halfspace",
    )
    if half is None:
        return None, None, center, normal, "Не удалось построить поверхность реза."

    part_pos = duplicate_object(source, name_pos)
    part_neg = duplicate_object(source, name_neg)

    ok_pos = apply_boolean(part_pos, half, "INTERSECT")
    ok_neg = apply_boolean(part_neg, half, "DIFFERENCE")
    delete_object_and_data(half)

    if not ok_pos or not ok_neg:
        delete_object_and_data(part_pos)
        delete_object_and_data(part_neg)
        return (
            None,
            None,
            center,
            normal,
            "Boolean по кривому резу не удался. Уточните путь или геометрию.",
        )

    # Empty-result guard.
    if len(part_pos.data.polygons) == 0 or len(part_neg.data.polygons) == 0:
        delete_object_and_data(part_pos)
        delete_object_and_data(part_neg)
        return (
            None,
            None,
            center,
            normal,
            "Кривой рез дал пустую деталь — путь не пересекает нужную область.",
        )

    return part_pos, part_neg, center, normal, ""


def path_intersects_object(points_world: list[Vector], obj: bpy.types.Object, pad: float = 0.01) -> bool:
    """Cheap bbox overlap test between path points and object."""
    if not points_world:
        return False
    mn, mx = object_bbox_world(obj)
    mn = mn - Vector((pad, pad, pad))
    mx = mx + Vector((pad, pad, pad))
    for p in points_world:
        if mn.x <= p.x <= mx.x and mn.y <= p.y <= mx.y and mn.z <= p.z <= mx.z:
            return True
    # Also: path centroid inside expanded bbox
    c = sum((p for p in points_world), Vector()) / len(points_world)
    return mn.x <= c.x <= mx.x and mn.y <= c.y <= mx.y and mn.z <= c.z <= mx.z
