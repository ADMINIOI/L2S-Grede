# SPDX-License-Identifier: GPL-3.0-or-later
from . import prepare, draft, cut, draw_cut, export_ops

MODULES = (prepare, draft, cut, draw_cut, export_ops)


def register():
    for mod in MODULES:
        mod.register()


def unregister():
    for mod in reversed(MODULES):
        mod.unregister()
