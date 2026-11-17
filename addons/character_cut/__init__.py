# SPDX-License-Identifier: GPL-3.0-or-later
"""
Character Cut — аддон Blender для разрезания персонажей на печатные детали
со штифтами и гнёздами (workflow Cut → Connect → Print).

Русский интерфейс, панель N: «Рез персонажа».
"""

bl_info = {
    "name": "Character Cut (Рез персонажа)",
    "author": "L2S-Grede / Cursor Agent",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Рез персонажа",
    "description": (
        "Разрезание 3D-персонажей на печатные части со штифтом/гнездом. "
        "Режим Черновик и Быстрый рез. Экспорт STL/OBJ/FBX."
    ),
    "warning": "Нужны закрытые manifold-сетки; сцена в мм",
    "doc_url": "",
    "category": "Mesh",
}

from . import properties, operators, ui


def register():
    properties.register()
    operators.register()
    ui.register()
    print("Character Cut: зарегистрирован (панель «Рез персонажа»)")


def unregister():
    ui.unregister()
    operators.unregister()
    properties.unregister()
    print("Character Cut: снят с регистрации")


if __name__ == "__main__":
    register()
