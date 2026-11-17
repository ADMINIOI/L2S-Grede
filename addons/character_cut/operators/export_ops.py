# SPDX-License-Identifier: GPL-3.0-or-later
"""Export parts collection to STL / OBJ / FBX."""

from __future__ import annotations

import os

import bpy

from .. import utils


class CHARACTER_CUT_OT_export_parts(bpy.types.Operator):
    bl_idname = "character_cut.export_parts"
    bl_label = "Экспорт деталей"
    bl_description = "Экспортировать все MESH из коллекции деталей в папку"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = context.scene.character_cut
        col = bpy.data.collections.get(utils.PARTS_COLLECTION)
        if col is None:
            self.report({"ERROR"}, f"Коллекция «{utils.PARTS_COLLECTION}» не найдена. Сначала соберите детали.")
            return {"CANCELLED"}

        meshes = [o for o in col.objects if o.type == "MESH"]
        if not meshes:
            self.report({"ERROR"}, "В коллекции деталей нет MESH-объектов")
            return {"CANCELLED"}

        export_dir = bpy.path.abspath(settings.export_dir)
        if not export_dir:
            self.report({"ERROR"}, "Укажите папку экспорта")
            return {"CANCELLED"}
        os.makedirs(export_dir, exist_ok=True)

        fmt = settings.export_format
        # Deselect all, export one-by-one to avoid merging.
        bpy.ops.object.select_all(action="DESELECT")
        exported = 0
        errors = []

        for obj in meshes:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
            safe = bpy.path.clean_name(obj.name)
            path = os.path.join(export_dir, f"{safe}.{fmt.lower()}")
            try:
                if fmt == "STL":
                    # Blender 4.2: wm.stl_export
                    if hasattr(bpy.ops.wm, "stl_export"):
                        bpy.ops.wm.stl_export(
                            filepath=path,
                            export_selected_objects=True,
                            ascii_format=False,
                        )
                    else:
                        bpy.ops.export_mesh.stl(filepath=path, use_selection=True)
                elif fmt == "OBJ":
                    if hasattr(bpy.ops.wm, "obj_export"):
                        bpy.ops.wm.obj_export(
                            filepath=path,
                            export_selected_objects=True,
                        )
                    else:
                        bpy.ops.export_scene.obj(filepath=path, use_selection=True)
                elif fmt == "FBX":
                    bpy.ops.export_scene.fbx(
                        filepath=path,
                        use_selection=True,
                        object_types={"MESH"},
                    )
                else:
                    errors.append(f"Неизвестный формат: {fmt}")
                    continue
                exported += 1
            except Exception as exc:
                errors.append(f"{obj.name}: {exc}")

        status = f"Экспортировано {exported}/{len(meshes)} → {export_dir}"
        if errors:
            status += f" | ошибок: {len(errors)}"
        settings.last_status = status
        self.report({"WARNING"} if errors else {"INFO"}, status)
        return {"FINISHED"} if exported else {"CANCELLED"}


CLASSES = (CHARACTER_CUT_OT_export_parts,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
