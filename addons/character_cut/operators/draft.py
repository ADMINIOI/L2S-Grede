# SPDX-License-Identifier: GPL-3.0-or-later
"""Draft-mode cut planes and character presets."""

from __future__ import annotations

import bpy
from mathutils import Vector

from .. import utils


def _source(context) -> bpy.types.Object | None:
    settings = context.scene.character_cut
    if settings.source_object and settings.source_object.type == "MESH":
        return settings.source_object
    return utils.active_mesh(context)


def _make_plane_guide(name: str, location: Vector, normal: Vector, size: float) -> bpy.types.Object:
    # Visual guide: thin cube used as cut plane gizmo (editable by user).
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    # Flat slab in XY, thin in Z — Z aligns to cut normal.
    bmesh.ops.scale(bm, vec=Vector((size, size, size * 0.01)), verts=bm.verts)
    bm.to_mesh(mesh)
    bm.free()

    utils.orient_object_z_to_vector(obj, normal, location)
    obj.display_type = "WIRE"
    obj.show_in_front = True
    obj.hide_render = True
    obj["character_cut_role"] = "PLANE"
    return obj


def _make_connector_empty(name: str, location: Vector, normal: Vector) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "SINGLE_ARROW"
    empty.empty_display_size = 0.02
    bpy.context.scene.collection.objects.link(empty)
    utils.orient_object_z_to_vector(empty, normal, location)
    empty["character_cut_role"] = "CONNECTOR"
    return empty


def _add_draft_item(
    context,
    name: str,
    location: Vector,
    normal: Vector,
    size: float,
) -> None:
    settings = context.scene.character_cut
    draft_col = utils.ensure_collection(utils.DRAFT_COLLECTION)

    plane = _make_plane_guide(f"{utils.CUT_PREFIX}_{name}", location, normal, size)
    connector = _make_connector_empty(f"{utils.CONNECTOR_PREFIX}_{name}", location, normal)
    utils.link_only(plane, draft_col)
    utils.link_only(connector, draft_col)
    connector.parent = plane
    connector.matrix_parent_inverse = plane.matrix_world.inverted()

    item = settings.draft_cuts.add()
    item.name = name
    item.enabled = True
    item.cut_type = "STRAIGHT"
    item.plane_object = plane
    item.path_object = None
    item.connector_object = connector
    item.shape = settings.shape
    item.radius_mm = settings.radius_mm
    item.depth_mm = settings.depth_mm
    item.clearance_mm = settings.clearance_mm
    item.pin_on_positive = settings.pin_on_positive
    settings.active_draft_index = len(settings.draft_cuts) - 1

    bpy.ops.object.select_all(action="DESELECT")
    plane.select_set(True)
    context.view_layer.objects.active = plane


def add_curved_draft_item(
    context,
    name: str,
    points: list,
    closed: bool = True,
) -> None:
    """Register a curved/polyline draft cut from world-space points."""
    settings = context.scene.character_cut
    draft_col = utils.ensure_collection(utils.DRAFT_COLLECTION)

    poly = utils.close_polyline([Vector(p) for p in points])
    normal = utils.newell_normal(poly)
    center = sum((p for p in poly), Vector()) / max(len(poly), 1)

    path = utils.create_path_guide_object(
        poly,
        name=f"{utils.CUT_PREFIX}_{name}",
        closed=closed,
    )
    # Reddish wire look
    path.color = (1.0, 0.2, 0.15, 1.0)
    connector = _make_connector_empty(f"{utils.CONNECTOR_PREFIX}_{name}", center, normal)
    utils.link_only(path, draft_col)
    utils.link_only(connector, draft_col)
    connector.parent = path
    connector.matrix_parent_inverse = path.matrix_world.inverted()

    item = settings.draft_cuts.add()
    item.name = name
    item.enabled = True
    item.cut_type = "CURVED"
    item.plane_object = None
    item.path_object = path
    item.connector_object = connector
    item.shape = settings.shape
    item.radius_mm = settings.radius_mm
    item.depth_mm = settings.depth_mm
    item.clearance_mm = settings.clearance_mm
    item.pin_on_positive = settings.pin_on_positive
    item.band_extent = 0.0
    settings.active_draft_index = len(settings.draft_cuts) - 1

    bpy.ops.object.select_all(action="DESELECT")
    path.select_set(True)
    context.view_layer.objects.active = path


class CHARACTER_CUT_OT_add_straight_cut(bpy.types.Operator):
    bl_idname = "character_cut.add_straight_cut"
    bl_label = "Прямой рез"
    bl_description = "Добавить плоскость прямого реза в режиме Черновик"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        src = _source(context)
        if src is None:
            self.report({"ERROR"}, "Выберите модель персонажа (MESH)")
            return {"CANCELLED"}

        ok, msg = utils.mesh_is_manifold(src)
        if not ok:
            self.report({"WARNING"}, msg)

        mn, mx = utils.object_bbox_world(src)
        center = (mn + mx) * 0.5
        size = max((mx - mn).length * 0.6, 0.05)
        # Default: horizontal cut through center (useful for torso / limbs).
        normal = Vector((0, 0, 1))
        _add_draft_item(context, f"Прямой_{len(context.scene.character_cut.draft_cuts)+1}", center, normal, size)
        context.scene.character_cut.source_object = src
        context.scene.character_cut.last_status = "Добавлен прямой рез — двигайте/вращайте плоскость"
        self.report({"INFO"}, "Прямой рез добавлен. Перемещайте плоскость и стрелку коннектора.")
        return {"FINISHED"}


class CHARACTER_CUT_OT_add_character_preset(bpy.types.Operator):
    bl_idname = "character_cut.add_character_preset"
    bl_label = "Пресет персонажа"
    bl_description = "Быстрый рез по анатомии персонажа"
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(
        name="Пресет",
        items=(
            ("HEAD", "Голова", "Горизонтальный рез шеи/головы"),
            ("WAIST", "Пояс", "Горизонтальный рез талии"),
            ("ARM_L", "Рука Л", "Вертикальный рез левой руки"),
            ("ARM_R", "Рука П", "Вертикальный рез правой руки"),
            ("LEG_L", "Нога Л", "Рез левой ноги"),
            ("LEG_R", "Нога П", "Рез правой ноги"),
        ),
        default="HEAD",
    )

    def execute(self, context):
        src = _source(context)
        if src is None:
            self.report({"ERROR"}, "Выберите модель персонажа (MESH)")
            return {"CANCELLED"}

        mn, mx = utils.object_bbox_world(src)
        size = max((mx - mn).length * 0.55, 0.05)
        cx = (mn.x + mx.x) * 0.5
        cy = (mn.y + mx.y) * 0.5
        cz = (mn.z + mx.z) * 0.5
        dx = mx.x - mn.x
        dz = mx.z - mn.z

        presets = {
            "HEAD": (Vector((cx, cy, mn.z + dz * 0.82)), Vector((0, 0, 1)), "Голова"),
            "WAIST": (Vector((cx, cy, mn.z + dz * 0.48)), Vector((0, 0, 1)), "Пояс"),
            "ARM_L": (Vector((mn.x + dx * 0.22, cy, mn.z + dz * 0.62)), Vector((1, 0, 0)), "Рука_Л"),
            "ARM_R": (Vector((mn.x + dx * 0.78, cy, mn.z + dz * 0.62)), Vector((1, 0, 0)), "Рука_П"),
            "LEG_L": (Vector((mn.x + dx * 0.35, cy, mn.z + dz * 0.28)), Vector((0, 0, 1)), "Нога_Л"),
            "LEG_R": (Vector((mn.x + dx * 0.65, cy, mn.z + dz * 0.28)), Vector((0, 0, 1)), "Нога_П"),
        }
        loc, normal, label = presets[self.preset]
        _add_draft_item(context, label, loc, normal, size)
        context.scene.character_cut.source_object = src
        context.scene.character_cut.last_status = f"Пресет: {label}"
        self.report({"INFO"}, f"Добавлен рез «{label}» — подправьте плоскость под модель")
        return {"FINISHED"}


class CHARACTER_CUT_OT_toggle_draft(bpy.types.Operator):
    bl_idname = "character_cut.toggle_draft"
    bl_label = "Вкл/выкл рез"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        settings = context.scene.character_cut
        if self.index < 0 or self.index >= len(settings.draft_cuts):
            return {"CANCELLED"}
        item = settings.draft_cuts[self.index]
        item.enabled = not item.enabled
        for obj in (item.plane_object, item.path_object, item.connector_object):
            if obj:
                obj.hide_viewport = not item.enabled
        return {"FINISHED"}


class CHARACTER_CUT_OT_remove_draft(bpy.types.Operator):
    bl_idname = "character_cut.remove_draft"
    bl_label = "Удалить рез"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        settings = context.scene.character_cut
        if self.index < 0 or self.index >= len(settings.draft_cuts):
            return {"CANCELLED"}
        item = settings.draft_cuts[self.index]
        utils.delete_object_and_data(item.plane_object)
        utils.delete_object_and_data(item.path_object)
        utils.delete_object_and_data(item.connector_object)
        settings.draft_cuts.remove(self.index)
        settings.active_draft_index = max(0, min(settings.active_draft_index, len(settings.draft_cuts) - 1))
        settings.last_status = "Рез удалён"
        return {"FINISHED"}


class CHARACTER_CUT_OT_clear_drafts(bpy.types.Operator):
    bl_idname = "character_cut.clear_drafts"
    bl_label = "Очистить черновик"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.character_cut
        while len(settings.draft_cuts):
            item = settings.draft_cuts[0]
            utils.delete_object_and_data(item.plane_object)
            utils.delete_object_and_data(item.path_object)
            utils.delete_object_and_data(item.connector_object)
            settings.draft_cuts.remove(0)
        settings.last_status = "Черновик очищен"
        return {"FINISHED"}


class CHARACTER_CUT_OT_select_draft_plane(bpy.types.Operator):
    bl_idname = "character_cut.select_draft_plane"
    bl_label = "Выбрать рез"
    bl_options = {"REGISTER"}

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        settings = context.scene.character_cut
        if self.index < 0 or self.index >= len(settings.draft_cuts):
            return {"CANCELLED"}
        item = settings.draft_cuts[self.index]
        target = item.path_object if item.cut_type == "CURVED" else item.plane_object
        if target is None:
            self.report({"ERROR"}, "Объект реза отсутствует")
            return {"CANCELLED"}
        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        context.view_layer.objects.active = target
        settings.active_draft_index = self.index
        return {"FINISHED"}


CLASSES = (
    CHARACTER_CUT_OT_add_straight_cut,
    CHARACTER_CUT_OT_add_character_preset,
    CHARACTER_CUT_OT_toggle_draft,
    CHARACTER_CUT_OT_remove_draft,
    CHARACTER_CUT_OT_clear_drafts,
    CHARACTER_CUT_OT_select_draft_plane,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
