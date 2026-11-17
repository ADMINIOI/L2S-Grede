# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive polyline / curved cut — draw path on mesh surface."""

from __future__ import annotations

import bpy
import gpu
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .. import utils
from . import draft as draft_ops


# Viewport overlay colors (CatPin-inspired: red path, yellow preview, green active).
COL_PATH = (0.95, 0.15, 0.12, 1.0)
COL_PREVIEW = (1.0, 0.92, 0.15, 1.0)
COL_ACTIVE = (0.2, 0.95, 0.25, 1.0)
COL_CLOSE = (0.3, 0.85, 1.0, 1.0)


def _ray_cast_mesh(context, event, target: bpy.types.Object):
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None, None
    coord = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    # Ray in object local space for object.ray_cast
    inv = target.matrix_world.inverted()
    origin_l = inv @ origin
    direction_l = (inv.to_3x3() @ direction).normalized()
    ok, location, normal, _face = target.ray_cast(origin_l, direction_l)
    if not ok:
        # Fallback: scene ray cast
        depsgraph = context.evaluated_depsgraph_get()
        result, loc_w, n_w, *_rest = context.scene.ray_cast(depsgraph, origin, direction)
        if not result:
            return None, None
        return loc_w, n_w.normalized()
    loc_w = target.matrix_world @ location
    n_w = (target.matrix_world.to_3x3() @ normal).normalized()
    return loc_w, n_w


def _screen_dist(context, world_a: Vector, event) -> float:
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return 1e9
    a2 = view3d_utils.location_3d_to_region_2d(region, rv3d, world_a)
    if a2 is None:
        return 1e9
    return (Vector((a2.x, a2.y)) - Vector((event.mouse_region_x, event.mouse_region_y))).length


def _draw_overlay(self, context):
    if context.area is None:
        return
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    gpu.state.line_width(3.0)

    pts = self.points
    if len(pts) >= 2:
        coords = [p.copy() for p in pts]
        if self.closed and len(pts) >= 3:
            coords.append(pts[0].copy())
        batch = batch_for_shader(shader, "LINE_STRIP", {"pos": coords})
        shader.bind()
        shader.uniform_float("color", COL_PATH)
        batch.draw(shader)

    # Rubber-band preview
    if pts and self.hover is not None and not self.closed:
        preview_end = pts[0] if self.hover_closes else self.hover
        batch = batch_for_shader(
            shader,
            "LINES",
            {"pos": [pts[-1].copy(), preview_end.copy()]},
        )
        shader.bind()
        shader.uniform_float("color", COL_CLOSE if self.hover_closes else COL_PREVIEW)
        batch.draw(shader)

    # Vertices
    if pts:
        batch = batch_for_shader(shader, "POINTS", {"pos": [p.copy() for p in pts]})
        shader.bind()
        try:
            gpu.state.point_size(8.0)
        except Exception:
            pass
        shader.uniform_float("color", COL_PATH)
        batch.draw(shader)
        # Active (last) vertex green
        batch = batch_for_shader(shader, "POINTS", {"pos": [pts[-1].copy()]})
        shader.bind()
        try:
            gpu.state.point_size(12.0)
        except Exception:
            pass
        shader.uniform_float("color", COL_ACTIVE)
        batch.draw(shader)
        if self.closed or self.hover_closes:
            batch = batch_for_shader(shader, "POINTS", {"pos": [pts[0].copy()]})
            shader.bind()
            shader.uniform_float("color", COL_CLOSE)
            batch.draw(shader)

    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.blend_set("NONE")
    gpu.state.line_width(1.0)


class CHARACTER_CUT_OT_draw_curved_cut(bpy.types.Operator):
    bl_idname = "character_cut.draw_curved_cut"
    bl_label = "Нарисовать рез"
    bl_description = (
        "Кривой рез: кликайте по поверхности, чтобы отметить путь точек. "
        "Замкните петлю вокруг руки/ноги — разрежется только эта область, не весь персонаж. "
        "ЛКМ — точка, Backspace — отмена точки, C — замкнуть, Enter/ПКМ — подтвердить, Esc — выход"
    )
    bl_options = {"REGISTER", "UNDO"}

    # When True (Easy mode), build parts immediately after confirm.
    easy_build: bpy.props.BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, event):
        settings = context.scene.character_cut
        src = settings.source_object if settings.source_object else utils.active_mesh(context)
        if src is None or src.type != "MESH":
            self.report({"ERROR"}, "Выберите MESH-модель персонажа")
            return {"CANCELLED"}
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"ERROR"}, "Откройте 3D Viewport")
            return {"CANCELLED"}

        self.target = src
        settings.source_object = src
        self.points: list[Vector] = []
        self.hover: Vector | None = None
        self.hover_normal: Vector | None = None
        self.hover_closes = False
        self.closed = False
        self._handle = None

        wm = context.window_manager
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay, (self, context), "WINDOW", "POST_VIEW"
        )
        wm.modal_handler_add(self)
        context.workspace.status_text_set(
            "Кривой рез: ЛКМ — точка | Backspace — назад | C — замкнуть | Enter/ПКМ — ОК | Esc — отмена"
        )
        settings.last_status = "Рисование реза… кликайте по модели"
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _remove_handler(self):
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            self._handle = None

    def _update_hover(self, context, event):
        loc, normal = _ray_cast_mesh(context, event, self.target)
        self.hover = loc
        self.hover_normal = normal
        self.hover_closes = False
        if loc is not None and len(self.points) >= 3 and not self.closed:
            if _screen_dist(context, self.points[0], event) < 18.0:
                self.hover_closes = True

    def _add_point(self, context, event):
        if self.closed:
            return
        loc, normal = _ray_cast_mesh(context, event, self.target)
        if loc is None:
            self.report({"WARNING"}, "Кликните по поверхности модели")
            return
        # Close if near first point
        if len(self.points) >= 3 and _screen_dist(context, self.points[0], event) < 18.0:
            self.closed = True
            context.scene.character_cut.last_status = "Путь замкнут — Enter/ПКМ чтобы подтвердить"
            return
        # Avoid tiny duplicate clicks
        if self.points and (self.points[-1] - loc).length < 1e-4:
            return
        self.points.append(loc.copy())
        context.scene.character_cut.last_status = f"Точек: {len(self.points)}"

    def _undo_point(self, context):
        if self.closed:
            self.closed = False
            context.scene.character_cut.last_status = "Контур снова открыт"
            return
        if self.points:
            self.points.pop()
            context.scene.character_cut.last_status = f"Точек: {len(self.points)}"

    def _confirm(self, context):
        settings = context.scene.character_cut
        pts = list(self.points)
        if self.closed or (len(pts) >= 3):
            # Auto-close if enough points
            if not self.closed and len(pts) >= 3:
                self.closed = True
        if len(pts) < 3:
            self.report({"ERROR"}, "Нужно минимум 3 точки для кривого реза")
            return {"RUNNING_MODAL"}

        name = f"Кривой_{len(settings.draft_cuts) + 1}"
        draft_ops.add_curved_draft_item(context, name, pts, closed=True)
        settings.last_status = (
            f"Кривой рез «{name}» добавлен. Подправьте точки или соберите детали."
        )

        if self.easy_build or settings.mode == "EASY":
            bpy.ops.character_cut.build_parts()

        self._cleanup(context)
        self.report({"INFO"}, "Кривой рез сохранён")
        return {"FINISHED"}

    def _cleanup(self, context):
        self._remove_handler()
        context.workspace.status_text_set(None)
        if context.area:
            context.area.tag_redraw()

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()

        if event.type == "MOUSEMOVE":
            self._update_hover(context, event)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._add_point(context, event)
            return {"RUNNING_MODAL"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            return self._confirm(context)

        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            return self._confirm(context)

        if event.type == "ESC" and event.value == "PRESS":
            self._cleanup(context)
            context.scene.character_cut.last_status = "Рисование отменено"
            self.report({"INFO"}, "Отменено")
            return {"CANCELLED"}

        if event.type == "BACK_SPACE" and event.value == "PRESS":
            self._undo_point(context)
            return {"RUNNING_MODAL"}

        if event.type == "C" and event.value == "PRESS":
            if len(self.points) >= 3:
                self.closed = True
                context.scene.character_cut.last_status = "Путь замкнут (C)"
            return {"RUNNING_MODAL"}

        # Allow navigation
        if event.type in {
            "MIDDLEMOUSE",
            "WHEELUPMOUSE",
            "WHEELDOWNMOUSE",
            "TRACKPADPAN",
            "TRACKPADZOOM",
        }:
            return {"PASS_THROUGH"}

        return {"RUNNING_MODAL"}


class CHARACTER_CUT_OT_apply_polyline_points(bpy.types.Operator):
    """Headless / script helper: apply a closed polyline cut from coded points."""

    bl_idname = "character_cut.apply_polyline_points"
    bl_label = "Применить полилайн-рез"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    # Packed as "x,y,z;x,y,z;..."
    points_csv: bpy.props.StringProperty(default="")
    easy_build: bpy.props.BoolProperty(default=False)
    name: bpy.props.StringProperty(default="Кривой")

    def execute(self, context):
        settings = context.scene.character_cut
        src = settings.source_object if settings.source_object else utils.active_mesh(context)
        if src is None:
            self.report({"ERROR"}, "Нет MESH-источника")
            return {"CANCELLED"}
        pts: list[Vector] = []
        for chunk in self.points_csv.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [float(x) for x in chunk.split(",")]
            if len(parts) >= 3:
                pts.append(Vector(parts[:3]))
        if len(pts) < 3:
            self.report({"ERROR"}, "Мало точек")
            return {"CANCELLED"}

        settings.source_object = src
        draft_ops.add_curved_draft_item(context, self.name, pts, closed=True)
        if self.easy_build or settings.mode == "EASY":
            return bpy.ops.character_cut.build_parts()
        settings.last_status = f"Кривой рез «{self.name}» добавлен"
        return {"FINISHED"}


CLASSES = (
    CHARACTER_CUT_OT_draw_curved_cut,
    CHARACTER_CUT_OT_apply_polyline_points,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
