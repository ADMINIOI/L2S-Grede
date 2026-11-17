# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare scene / mesh for cutting."""

from __future__ import annotations

import bpy

from .. import utils


class CHARACTER_CUT_OT_set_mm_units(bpy.types.Operator):
    bl_idname = "character_cut.set_mm_units"
    bl_label = "Единицы: миллиметры"
    bl_description = "Переключить сцену на метрику с длиной в миллиметрах"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        us = context.scene.unit_settings
        us.system = "METRIC"
        us.scale_length = 0.001  # 1 BU = 1 mm
        us.length_unit = "MILLIMETERS"
        context.scene.character_cut.last_status = "Единицы: 1 ед. = 1 мм"
        self.report({"INFO"}, "Сцена: миллиметры (1 BU = 1 мм)")
        return {"FINISHED"}


class CHARACTER_CUT_OT_apply_transforms(bpy.types.Operator):
    bl_idname = "character_cut.apply_transforms"
    bl_label = "Применить масштаб и поворот"
    bl_description = "Применить Scale и Rotation к активной сетке (обязательно перед резом)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = utils.active_mesh(context)
        if obj is None:
            self.report({"ERROR"}, "Выберите MESH-объект")
            return {"CANCELLED"}
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        context.scene.character_cut.last_status = f"Применены трансформы: {obj.name}"
        self.report({"INFO"}, "Масштаб и поворот применены")
        return {"FINISHED"}


class CHARACTER_CUT_OT_check_manifold(bpy.types.Operator):
    bl_idname = "character_cut.check_manifold"
    bl_label = "Проверить manifold"
    bl_description = "Проверить, что сетка закрытая (нужно для Boolean)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        obj = utils.active_mesh(context)
        if obj is None:
            self.report({"ERROR"}, "Выберите MESH-объект")
            return {"CANCELLED"}
        ok, msg = utils.mesh_is_manifold(obj)
        context.scene.character_cut.last_status = msg
        self.report({"INFO"} if ok else {"ERROR"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class CHARACTER_CUT_OT_set_source(bpy.types.Operator):
    """Kept for compatibility; not shown in the main panel (selection is enough)."""

    bl_idname = "character_cut.set_source"
    bl_label = "Запомнить персонажа"
    bl_description = "Необязательно: активный MESH запоминается сам при рисовании реза"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = utils.active_mesh(context)
        if obj is None:
            self.report({"ERROR"}, "Выберите MESH-объект")
            return {"CANCELLED"}
        utils.remember_character(context, obj)
        context.scene.character_cut.last_status = f"Персонаж: {obj.name}"
        self.report({"INFO"}, f"Персонаж: {obj.name} (обычно не нужно — достаточно выделения)")
        return {"FINISHED"}


CLASSES = (
    CHARACTER_CUT_OT_set_mm_units,
    CHARACTER_CUT_OT_apply_transforms,
    CHARACTER_CUT_OT_check_manifold,
    CHARACTER_CUT_OT_set_source,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
