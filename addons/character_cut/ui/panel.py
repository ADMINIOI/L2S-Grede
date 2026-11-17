# SPDX-License-Identifier: GPL-3.0-or-later
"""N-panel UI — русский интерфейс с иконками."""

from __future__ import annotations

import bpy

from .. import utils


class CHARACTER_CUT_PT_main(bpy.types.Panel):
    bl_label = "Character Cut"
    bl_idname = "CHARACTER_CUT_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Рез персонажа"

    def draw_header(self, context):
        self.layout.label(text="", icon="MOD_BOOLEAN")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.character_cut

        # --- Status ---
        box = layout.box()
        row = box.row(align=True)
        row.label(text="Статус", icon="INFO")
        if settings.last_status:
            box.label(text=settings.last_status, icon="DOT")
        else:
            box.label(text="Выберите модель → подготовка → рез", icon="HAND")

        # --- Prepare ---
        box = layout.box()
        box.label(text="1. Подготовка", icon="TOOL_SETTINGS")
        col = box.column(align=True)
        col.operator("character_cut.set_mm_units", icon="DRIVER_DISTANCE")
        col.operator("character_cut.apply_transforms", icon="OBJECT_ORIGIN")
        col.operator("character_cut.check_manifold", icon="MESH_DATA")
        col.operator("character_cut.set_source", icon="USER")
        if settings.source_object:
            row = box.row()
            row.label(text=f"Источник: {settings.source_object.name}", icon="OUTLINER_OB_MESH")

        # --- Mode ---
        box = layout.box()
        box.label(text="2. Режим", icon="UV_SYNC_SELECT")
        row = box.row(align=True)
        row.prop(settings, "mode", expand=True)

        # --- Connector defaults ---
        box = layout.box()
        box.label(text="3. Коннектор (штифт / гнездо)", icon="MESH_CYLINDER")
        col = box.column(align=True)
        col.prop(settings, "shape", text="")
        col.prop(settings, "radius_mm", icon="MESH_CIRCLE")
        col.prop(settings, "depth_mm", icon="EMPTY_SINGLE_ARROW")
        col.prop(settings, "fit_preset", text="Пресет")
        col.prop(settings, "clearance_mm", icon="MOD_SOLIDIFY")
        col.prop(settings, "pin_on_positive", icon="ARROW_LEFTRIGHT")
        col.prop(settings, "keep_original", icon="PINNED")

        # --- Cuts ---
        box = layout.box()
        box.label(text="4. Резы", icon="MOD_EDGESPLIT")

        if settings.mode == "DRAFT":
            col = box.column(align=True)
            col.operator("character_cut.add_straight_cut", icon="MESH_PLANE")
            row = col.row(align=True)
            row.scale_y = 1.25
            row.operator("character_cut.draw_curved_cut", text="Кривой рез (нарисовать)", icon="CURVE_BEZCURVE")
            col.operator("character_cut.place_cut_from_view", icon="VIEW3D")

            box.label(
                text="Кривой рез: петля вокруг руки/ноги — без среза всего тела",
                icon="INFO",
            )

            # Character presets with icons
            box.label(text="Плоскости-пресеты (осторожно с висящими руками)", icon="ARMATURE_DATA")
            grid = box.grid_flow(columns=3, align=True, even_columns=True)
            op = grid.operator("character_cut.add_character_preset", text="Голова", icon="USER")
            op.preset = "HEAD"
            op = grid.operator("character_cut.add_character_preset", text="Пояс", icon="SNAP_MIDPOINT")
            op.preset = "WAIST"
            op = grid.operator("character_cut.add_character_preset", text="Рука Л", icon="TRIA_LEFT")
            op.preset = "ARM_L"
            op = grid.operator("character_cut.add_character_preset", text="Рука П", icon="TRIA_RIGHT")
            op.preset = "ARM_R"
            op = grid.operator("character_cut.add_character_preset", text="Нога Л", icon="SORT_ASC")
            op.preset = "LEG_L"
            op = grid.operator("character_cut.add_character_preset", text="Нога П", icon="SORT_DESC")
            op.preset = "LEG_R"

            # Draft list
            box.label(text="Список резов", icon="OUTLINER")
            if not settings.draft_cuts:
                box.label(text="Пока пусто — добавьте рез", icon="ADD")
            for i, item in enumerate(settings.draft_cuts):
                row = box.row(align=True)
                icon = "CHECKBOX_HLT" if item.enabled else "CHECKBOX_DEHLT"
                tog = row.operator("character_cut.toggle_draft", text="", icon=icon)
                tog.index = i
                type_icon = "CURVE_BEZCURVE" if item.cut_type == "CURVED" else "MESH_PLANE"
                label = f"{'⌁ ' if item.cut_type == 'CURVED' else ''}{item.name}"
                sel = row.operator("character_cut.select_draft_plane", text=label, icon=type_icon)
                sel.index = i
                rem = row.operator("character_cut.remove_draft", text="", icon="X")
                rem.index = i

                # Per-cut connector overrides when selected
                if i == settings.active_draft_index:
                    sub = box.box()
                    sub.label(text=f"Настройки: {item.name}", icon="SETTINGS")
                    sub.prop(item, "enabled", icon="HIDE_OFF")
                    sub.label(text=f"Тип: {'кривой путь' if item.cut_type == 'CURVED' else 'прямая плоскость'}")
                    sub.prop(item, "shape")
                    sub.prop(item, "radius_mm")
                    sub.prop(item, "depth_mm")
                    sub.prop(item, "clearance_mm")
                    sub.prop(item, "pin_on_positive")
                    if item.cut_type == "CURVED":
                        sub.prop(item, "band_extent")

            row = box.row(align=True)
            row.operator("character_cut.clear_drafts", icon="TRASH")
            row = box.row()
            row.scale_y = 1.4
            row.operator("character_cut.build_parts", icon="MOD_BUILD")

        else:
            col = box.column(align=True)
            col.label(text="Быстрый режим: плоскость или нарисованный путь", icon="QUESTION")
            col.operator("character_cut.add_straight_cut", text="Плоскость для быстрого реза", icon="MESH_PLANE")
            row = col.row(align=True)
            row.scale_y = 1.25
            op = row.operator(
                "character_cut.draw_curved_cut",
                text="Нарисовать рез + собрать",
                icon="CURVE_BEZCURVE",
            )
            op.easy_build = True
            row = box.row()
            row.scale_y = 1.4
            row.operator("character_cut.easy_cut", text="Плоскость → детали", icon="PLAY")

        # --- Export ---
        box = layout.box()
        box.label(text="5. Экспорт", icon="EXPORT")
        col = box.column(align=True)
        col.prop(settings, "export_format", text="")
        col.prop(settings, "export_dir")
        row = box.row()
        row.scale_y = 1.2
        row.operator("character_cut.export_parts", icon="FILE_FOLDER")

        # --- Help ---
        box = layout.box()
        box.label(text="Важно", icon="ERROR")
        col = box.column(align=True)
        col.scale_y = 0.85
        col.label(text="• Только закрытые manifold-сетки", icon="MESH_ICOSPHERE")
        col.label(text="• Сцена в мм, Apply Scale/Rotation", icon="DRIVER_DISTANCE")
        col.label(text="• Зазор калибруйте тестовой печатью", icon="MOD_SOLIDIFY")
        col.label(text="• Кривой рез: замкните петлю вокруг детали", icon="CURVE_BEZCURVE")
        col.label(text="• ЛКМ точка · Backspace · C замкнуть · Enter ОК", icon="DOT")


CLASSES = (CHARACTER_CUT_PT_main,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
