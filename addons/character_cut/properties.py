# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene / object properties for Character Cut."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


class CharacterCutDraftItem(bpy.types.PropertyGroup):
    name: StringProperty(name="Имя", default="Разрез")
    enabled: BoolProperty(name="Включён", default=True)
    cut_type: EnumProperty(
        name="Тип реза",
        items=(
            ("STRAIGHT", "Прямой", "Бесконечная плоскость"),
            ("CURVED", "Кривой", "Локальный путь по поверхности"),
        ),
        default="STRAIGHT",
    )
    plane_object: PointerProperty(name="Плоскость", type=bpy.types.Object)
    path_object: PointerProperty(name="Путь", type=bpy.types.Object)
    connector_object: PointerProperty(name="Коннектор", type=bpy.types.Object)
    pin_on_positive: BoolProperty(
        name="Штифт на + стороне",
        description="Если включено — штифт на стороне нормали плоскости/пути",
        default=True,
    )
    band_extent: FloatProperty(
        name="Глубина пояса реза",
        description="Насколько далеко вытянуть локальный объём реза вдоль нормали пути",
        default=0.0,
        min=0.0,
        soft_max=500.0,
    )
    shape: EnumProperty(
        name="Форма",
        items=(
            ("CYLINDER", "Цилиндр", "Круглый штифт"),
            ("TAPERED", "Конус", "Самоцентрирующийся конус"),
            ("BOX", "Коробка", "Квадратный штифт против проворота"),
        ),
        default="CYLINDER",
    )
    radius_mm: FloatProperty(name="Радиус (мм)", default=2.0, min=0.2, soft_max=20.0)
    depth_mm: FloatProperty(name="Глубина (мм)", default=6.0, min=0.5, soft_max=40.0)
    clearance_mm: FloatProperty(
        name="Зазор (мм)",
        description="Насколько гнездо больше штифта. Обычно 0.1–0.3 мм",
        default=0.15,
        min=0.0,
        soft_max=1.0,
        precision=3,
    )


class CharacterCutSettings(bpy.types.PropertyGroup):
    mode: EnumProperty(
        name="Режим",
        description="Черновик — планирование резов; Быстрый — сразу детали",
        items=(
            ("DRAFT", "Черновик", "Несколько резов без разрушения оригинала"),
            ("EASY", "Быстрый", "Один рез → сразу детали со штифтом"),
        ),
        default="DRAFT",
    )
    shape: EnumProperty(
        name="Форма коннектора",
        items=(
            ("CYLINDER", "Цилиндр", "Круглый штифт"),
            ("TAPERED", "Конус", "Самоцентрирующийся"),
            ("BOX", "Коробка", "Против проворота"),
        ),
        default="CYLINDER",
    )
    radius_mm: FloatProperty(
        name="Радиус / половина (мм)",
        default=2.0,
        min=0.2,
        soft_max=20.0,
    )
    depth_mm: FloatProperty(
        name="Глубина (мм)",
        default=6.0,
        min=0.5,
        soft_max=40.0,
    )
    clearance_mm: FloatProperty(
        name="Зазор (мм)",
        description="Зазор гнезда. Смола ~0.1–0.2; FDM часто 0.15–0.3",
        default=0.15,
        min=0.0,
        soft_max=1.0,
        precision=3,
    )
    pin_on_positive: BoolProperty(
        name="Штифт на стороне нормали",
        default=True,
    )
    keep_original: BoolProperty(
        name="Сохранить оригинал",
        description="Не удалять исходный объект при сборке",
        default=True,
    )
    fit_preset: EnumProperty(
        name="Пресет зазора",
        items=(
            ("RESIN_TIGHT", "Смола плотно", "0.10 мм"),
            ("RESIN", "Смола", "0.15 мм"),
            ("FDM", "FDM", "0.25 мм"),
            ("LOOSE", "Свободно", "0.35 мм"),
            ("CUSTOM", "Свой", "Ручное значение"),
        ),
        default="RESIN",
        update=lambda self, context: _apply_fit_preset(self),
    )
    export_format: EnumProperty(
        name="Формат",
        items=(
            ("STL", "STL", "Стандарт для слайсеров"),
            ("OBJ", "OBJ", "Wavefront OBJ"),
            ("FBX", "FBX", "Autodesk FBX"),
        ),
        default="STL",
    )
    export_dir: StringProperty(
        name="Папка экспорта",
        subtype="DIR_PATH",
        default="//character_cut_export/",
    )
    active_draft_index: IntProperty(name="Активный рез", default=0, min=0)
    draft_cuts: bpy.props.CollectionProperty(type=CharacterCutDraftItem)
    last_status: StringProperty(name="Статус", default="")
    source_object: PointerProperty(name="Источник", type=bpy.types.Object)


def _apply_fit_preset(settings: CharacterCutSettings) -> None:
    mapping = {
        "RESIN_TIGHT": 0.10,
        "RESIN": 0.15,
        "FDM": 0.25,
        "LOOSE": 0.35,
    }
    if settings.fit_preset in mapping:
        settings.clearance_mm = mapping[settings.fit_preset]


CLASSES = (
    CharacterCutDraftItem,
    CharacterCutSettings,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.character_cut = PointerProperty(type=CharacterCutSettings)


def unregister():
    del bpy.types.Scene.character_cut
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
