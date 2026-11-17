# SPDX-License-Identifier: GPL-3.0-or-later
"""Easy Mode cut and Draft Mode Build Parts."""

from __future__ import annotations

import bpy
from mathutils import Vector

from .. import utils


def _plane_world_frame(plane_obj: bpy.types.Object) -> tuple[Vector, Vector]:
    """Return (origin, normal) in world space. Plane local +Z is cut normal."""
    mw = plane_obj.matrix_world
    origin = mw.translation.copy()
    normal = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
    return origin, normal


def _connector_world(connector_obj: bpy.types.Object | None, fallback_co: Vector, fallback_no: Vector) -> tuple[Vector, Vector]:
    if connector_obj is None:
        return fallback_co, fallback_no
    mw = connector_obj.matrix_world
    loc = mw.translation.copy()
    normal = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
    if normal.length < 1e-8:
        normal = fallback_no
    return loc, normal


def _split_with_connector(
    scene: bpy.types.Scene,
    source: bpy.types.Object,
    plane_co: Vector,
    plane_no: Vector,
    conn_co: Vector,
    conn_no: Vector,
    shape: str,
    radius_mm: float,
    depth_mm: float,
    clearance_mm: float,
    pin_on_positive: bool,
    name_prefix: str,
) -> tuple[bpy.types.Object | None, bpy.types.Object | None, str]:
    pos_name = f"{name_prefix}_плюс"
    neg_name = f"{name_prefix}_минус"
    part_pos, part_neg, err = utils.bisect_mesh_to_parts(source, plane_co, plane_no, pos_name, neg_name)
    if err:
        return None, None, err

    pin_side = part_pos if pin_on_positive else part_neg
    socket_side = part_neg if pin_on_positive else part_pos
    # Pin sticks along cut normal from the cut face.
    pin_normal = conn_no if pin_on_positive else -conn_no
    err2 = utils.add_pin_socket(
        scene,
        pin_side,
        socket_side,
        conn_co,
        pin_normal,
        shape,
        radius_mm,
        depth_mm,
        clearance_mm,
    )
    if err2:
        return part_pos, part_neg, err2
    return part_pos, part_neg, ""


def _split_curved_with_connector(
    scene: bpy.types.Scene,
    source: bpy.types.Object,
    points: list,
    conn_co: Vector | None,
    conn_no: Vector | None,
    shape: str,
    radius_mm: float,
    depth_mm: float,
    clearance_mm: float,
    pin_on_positive: bool,
    name_prefix: str,
    band_extent: float = 0.0,
) -> tuple[bpy.types.Object | None, bpy.types.Object | None, str]:
    extent = band_extent if band_extent > 0.0 else None
    extrude = conn_no if conn_no is not None else None
    part_pos, part_neg, center, normal, err = utils.split_mesh_by_path_loop(
        source,
        points,
        f"{name_prefix}_плюс",
        f"{name_prefix}_минус",
        extent_positive=extent,
        extrude_normal=extrude,
    )
    if err:
        return None, None, err

    c_co = conn_co if conn_co is not None else center
    c_no = conn_no if conn_no is not None else normal
    pin_side = part_pos if pin_on_positive else part_neg
    socket_side = part_neg if pin_on_positive else part_pos
    pin_normal = c_no if pin_on_positive else -c_no
    err2 = utils.add_pin_socket(
        scene,
        pin_side,
        socket_side,
        c_co,
        pin_normal,
        shape,
        radius_mm,
        depth_mm,
        clearance_mm,
    )
    if err2:
        return part_pos, part_neg, err2
    return part_pos, part_neg, ""


class CHARACTER_CUT_OT_easy_cut(bpy.types.Operator):
    bl_idname = "character_cut.easy_cut"
    bl_label = "Быстрый рез + штифт"
    bl_description = "Один рез активной плоскостью (или через центр) → две детали со штифтом/гнездом"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.character_cut
        src = settings.source_object if settings.source_object else utils.active_mesh(context)
        if src is None or src.type != "MESH":
            self.report({"ERROR"}, "Выберите MESH-модель")
            return {"CANCELLED"}

        ok, msg = utils.mesh_is_manifold(src)
        if not ok:
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        # Prefer active draft plane / CC plane; else cut through bbox center horizontally.
        plane_obj = context.active_object
        if plane_obj and plane_obj.get("character_cut_role") == "PLANE":
            plane_co, plane_no = _plane_world_frame(plane_obj)
            # Find matching connector among draft items.
            conn_obj = None
            for item in settings.draft_cuts:
                if item.plane_object == plane_obj:
                    conn_obj = item.connector_object
                    break
            conn_co, conn_no = _connector_world(conn_obj, plane_co, plane_no)
        else:
            mn, mx = utils.object_bbox_world(src)
            plane_co = (mn + mx) * 0.5
            plane_no = Vector((0, 0, 1))
            conn_co, conn_no = plane_co, plane_no

        work = utils.duplicate_object(src, f"{utils.SOURCE_PREFIX}_{src.name}")
        parts_col = utils.ensure_collection(utils.PARTS_COLLECTION)

        part_a, part_b, err = _split_with_connector(
            context.scene,
            work,
            plane_co,
            plane_no,
            conn_co,
            conn_no,
            settings.shape,
            settings.radius_mm,
            settings.depth_mm,
            settings.clearance_mm,
            settings.pin_on_positive,
            f"{src.name}_часть",
        )
        utils.delete_object_and_data(work)

        if part_a is None or part_b is None:
            settings.last_status = err or "Сбой реза"
            self.report({"ERROR"}, err or "Сбой реза")
            return {"CANCELLED"}

        utils.link_only(part_a, parts_col)
        utils.link_only(part_b, parts_col)
        if not settings.keep_original:
            utils.delete_object_and_data(src)
            settings.source_object = None

        if err:
            settings.last_status = f"Детали созданы с предупреждением: {err}"
            self.report({"WARNING"}, err)
        else:
            settings.last_status = f"Быстрый рез готов → коллекция «{utils.PARTS_COLLECTION}»"
            self.report({"INFO"}, "Детали со штифтом созданы")
        return {"FINISHED"}


class CHARACTER_CUT_OT_build_parts(bpy.types.Operator):
    bl_idname = "character_cut.build_parts"
    bl_label = "Собрать детали"
    bl_description = "Применить все включённые резы черновика и создать коллекцию деталей"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.character_cut
        src = settings.source_object if settings.source_object else utils.active_mesh(context)
        if src is None or src.type != "MESH":
            self.report({"ERROR"}, "Не задан исходный MESH")
            return {"CANCELLED"}

        enabled = [
            item
            for item in settings.draft_cuts
            if item.enabled
            and (
                (item.cut_type == "CURVED" and item.path_object)
                or (item.cut_type != "CURVED" and item.plane_object)
            )
        ]
        if not enabled:
            self.report({"ERROR"}, "Нет включённых резов в черновике")
            return {"CANCELLED"}

        ok, msg = utils.mesh_is_manifold(src)
        if not ok:
            settings.last_status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        parts_col = utils.ensure_collection(utils.PARTS_COLLECTION)
        # Start with one working piece (duplicate of source).
        queue = [utils.duplicate_object(src, f"{utils.SOURCE_PREFIX}_build")]
        warnings: list[str] = []

        for cut_i, item in enumerate(enabled):
            next_queue: list[bpy.types.Object] = []

            if item.cut_type == "CURVED":
                points = utils.path_points_from_object(item.path_object)
                if len(points) < 3:
                    warnings.append(f"Кривой рез #{cut_i}: мало точек")
                    next_queue.extend(queue)
                    queue = next_queue
                    continue
                poly = utils.close_polyline(points)
                normal = utils.newell_normal(poly)
                center = sum((p for p in poly), Vector()) / len(poly)
                conn_co, conn_no = _connector_world(item.connector_object, center, normal)

                for piece_i, piece in enumerate(queue):
                    if not utils.path_intersects_object(poly, piece):
                        next_queue.append(piece)
                        continue
                    part_a, part_b, err = _split_curved_with_connector(
                        context.scene,
                        piece,
                        poly,
                        conn_co,
                        conn_no,
                        item.shape,
                        item.radius_mm,
                        item.depth_mm,
                        item.clearance_mm,
                        item.pin_on_positive,
                        f"{src.name}_r{cut_i}_{piece_i}",
                        band_extent=item.band_extent,
                    )
                    utils.delete_object_and_data(piece)
                    if part_a is None or part_b is None:
                        warnings.append(err or f"Сбой кривого реза #{cut_i}")
                        continue
                    if err:
                        warnings.append(err)
                    next_queue.extend([part_a, part_b])
            else:
                plane_co, plane_no = _plane_world_frame(item.plane_object)
                conn_co, conn_no = _connector_world(item.connector_object, plane_co, plane_no)

                for piece_i, piece in enumerate(queue):
                    # Skip pieces that barely intersect the plane.
                    mn, mx = utils.object_bbox_world(piece)
                    corners = [
                        Vector((x, y, z))
                        for x in (mn.x, mx.x)
                        for y in (mn.y, mx.y)
                        for z in (mn.z, mx.z)
                    ]
                    dots = [((c - plane_co).dot(plane_no)) for c in corners]
                    if all(d > 1e-4 for d in dots) or all(d < -1e-4 for d in dots):
                        next_queue.append(piece)
                        continue

                    part_a, part_b, err = _split_with_connector(
                        context.scene,
                        piece,
                        plane_co,
                        plane_no,
                        conn_co,
                        conn_no,
                        item.shape,
                        item.radius_mm,
                        item.depth_mm,
                        item.clearance_mm,
                        item.pin_on_positive,
                        f"{src.name}_r{cut_i}_{piece_i}",
                    )
                    utils.delete_object_and_data(piece)
                    if part_a is None or part_b is None:
                        warnings.append(err or f"Сбой реза #{cut_i}")
                        continue
                    if err:
                        warnings.append(err)
                    next_queue.extend([part_a, part_b])

            queue = next_queue
            if not queue:
                settings.last_status = "Все детали потеряны после реза — проверьте плоскости/пути"
                self.report({"ERROR"}, settings.last_status)
                return {"CANCELLED"}

        for i, part in enumerate(queue):
            part.name = f"{src.name}_деталь_{i+1:02d}"
            utils.link_only(part, parts_col)

        if not settings.keep_original:
            utils.delete_object_and_data(src)
            settings.source_object = None

        status = f"Собрано деталей: {len(queue)} → «{utils.PARTS_COLLECTION}»"
        if warnings:
            status += f" | предупреждений: {len(warnings)}"
        settings.last_status = status
        self.report({"WARNING"} if warnings else {"INFO"}, status)
        return {"FINISHED"}


class CHARACTER_CUT_OT_place_cut_from_view(bpy.types.Operator):
    bl_idname = "character_cut.place_cut_from_view"
    bl_label = "Рез из вида"
    bl_description = "Добавить плоскость реза через центр модели, перпендикулярно виду"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.character_cut
        src = settings.source_object if settings.source_object else utils.active_mesh(context)
        if src is None:
            self.report({"ERROR"}, "Выберите MESH")
            return {"CANCELLED"}
        mn, mx = utils.object_bbox_world(src)
        center = (mn + mx) * 0.5
        size = max((mx - mn).length * 0.6, 0.05)
        rv3d = context.region_data
        if rv3d is None:
            normal = Vector((0, 0, 1))
        else:
            normal = (rv3d.view_rotation @ Vector((0, 0, -1))).normalized()
        # Reuse draft helper via operator props path — call add logic inline.
        from . import draft as draft_ops

        draft_ops._add_draft_item(context, f"Вид_{len(settings.draft_cuts)+1}", center, normal, size)
        settings.source_object = src
        settings.last_status = "Плоскость из текущего вида"
        self.report({"INFO"}, "Плоскость добавлена из вида")
        return {"FINISHED"}


CLASSES = (
    CHARACTER_CUT_OT_easy_cut,
    CHARACTER_CUT_OT_build_parts,
    CHARACTER_CUT_OT_place_cut_from_view,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
