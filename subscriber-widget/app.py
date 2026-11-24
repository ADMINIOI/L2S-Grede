"""Transparent always-on-top rings for public AxisBox subscriber counts."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from counts import REFRESH_SECONDS, ProfileCount, fetch_all
from render import (
    DIAL,
    GAP,
    GOAL_CY,
    RADIUS,
    clip_to_circles,
    encode_png,
    flatten_chroma,
    render_dial,
    ring_fraction,
)

# Windows treats this exact color as transparent. It is not used in the rings.
CHROMA = "#010203"
PLATFORMS = ("TikTok", "Instagram", "Facebook")
WIDGET_W = DIAL * len(PLATFORMS) + GAP * (len(PLATFORMS) - 1)
WIDGET_H = DIAL


def state_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "axisbox-subscriber-widget" / "window.json"


def enable_windows_dpi() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            return


def print_once() -> int:
    payload = [item.to_dict() for item in fetch_all()]
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Окно счётчиков подписчиков AxisBox")
    parser.add_argument("--once", action="store_true", help="Print the public counts as JSON and exit")
    args = parser.parse_args(argv)
    if args.once:
        return print_once()
    try:
        import tkinter as tk
    except ImportError:
        sys.stderr.write("Нужен Python с tkinter. Установите его с python.org и включите tcl/tk.\n")
        return 1
    enable_windows_dpi()
    Widget(tk).mainloop()
    return 0


class Widget:
    def __init__(self, tk, *, autostart: bool = True) -> None:
        self._tk = tk
        self.root = tk.Tk()
        self.root.title("Подписчики")
        self.root.configure(bg=CHROMA)
        self.root.resizable(False, False)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", CHROMA)
        except tk.TclError:
            pass
        self._busy = False
        self._after_id = None
        self._autostart = autostart
        self._goals: dict[str, int] = {}
        self._results: dict[str, ProfileCount] = {}
        self._editor = None
        self._drag = None
        self._shape_queued = False
        self._photo = None
        self._rgba = b""
        self._menu_platform = PLATFORMS[0]
        self.canvas = tk.Canvas(
            self.root,
            width=WIDGET_W,
            height=WIDGET_H,
            bg=CHROMA,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag_window)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._popup)
        self.root.bind("<F5>", lambda _event: self.refresh())
        self.root.bind("<Button-3>", self._popup)
        self._menu = tk.Menu(self.root, tearoff=0)
        self._menu.add_command(label="Обновить", command=self.refresh)
        self._menu.add_command(label="Задать цель", command=self._edit_from_menu)
        self._menu.add_separator()
        self._menu.add_command(label="Закрыть", command=self.close)
        self._restore_state()
        if autostart:
            self.root.after(150, self.refresh)

    def mainloop(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self._save_state()
        self.root.destroy()

    def refresh(self) -> None:
        if self._busy:
            return
        self._commit_editor(save=True)
        self._busy = True
        threading.Thread(target=self._load, daemon=True).start()

    def show(self, results: list[ProfileCount]) -> None:
        self._commit_editor(save=True)
        self._results = {item.platform: item for item in results}
        self._busy = False
        self._draw()
        if self._autostart:
            self._schedule()

    def _load(self) -> None:
        try:
            results = fetch_all()
        except Exception:
            results = []
        self.root.after(0, lambda results=results: self.show(results))

    def _schedule(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(REFRESH_SECONDS * 1000, self.refresh)

    def _draw(self) -> None:
        strips = []
        for platform in PLATFORMS:
            item = self._results.get(platform)
            count = item.count if item is not None else None
            goal = self._goals.get(platform)
            _width, _height, pixels = render_dial(platform, self._count_text(item), self._goal_text(goal), ring_fraction(count, goal))
            strips.append(pixels)
        self._rgba = clip_to_circles(WIDGET_W, WIDGET_H, _join(strips))
        flat = flatten_chroma(WIDGET_W, WIDGET_H, self._rgba)
        self._photo = self._tk.PhotoImage(data=base64.b64encode(encode_png(WIDGET_W, WIDGET_H, flat)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self._queue_shape()

    def _count_text(self, item: ProfileCount | None) -> str:
        if item is None:
            return "…"
        if item.count is None:
            return "—"
        return f"{item.count:,}".replace(",", " ")

    def _goal_text(self, goal: int | None) -> str:
        if not goal:
            return "цель"
        return f"{goal:,}".replace(",", " ")

    def _platform_at(self, x: int) -> str | None:
        slot = DIAL + GAP
        index = x // slot
        if index < 0 or index >= len(PLATFORMS):
            return None
        if x - index * slot >= DIAL:
            return None
        return PLATFORMS[index]

    def _goal_hit(self, x: int, y: int) -> str | None:
        platform = self._platform_at(x)
        if platform is None:
            return None
        local_x = x - PLATFORMS.index(platform) * (DIAL + GAP)
        if abs(local_x - DIAL / 2) <= 52 and abs(y - GOAL_CY) <= 16:
            return platform
        return None

    def _press(self, event) -> None:
        self._drag = {
            "x": event.x_root,
            "y": event.y_root,
            "win_x": self.root.winfo_x(),
            "win_y": self.root.winfo_y(),
            "moved": False,
            "local_x": event.x,
            "local_y": event.y,
        }

    def _drag_window(self, event) -> None:
        if not self._drag:
            return
        dx = event.x_root - self._drag["x"]
        dy = event.y_root - self._drag["y"]
        if abs(dx) + abs(dy) > 4:
            self._drag["moved"] = True
            self._commit_editor(save=True)
            self.root.geometry(f"+{self._drag['win_x'] + dx}+{self._drag['win_y'] + dy}")

    def _release(self, event) -> None:
        drag = self._drag
        self._drag = None
        if not drag or drag.get("moved"):
            return
        platform = self._goal_hit(drag["local_x"], drag["local_y"])
        if platform:
            self._begin_edit(platform)

    def _popup(self, event) -> str:
        platform = self._platform_at(getattr(event, "x", 0))
        if platform:
            self._menu_platform = platform
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()
        return "break"

    def _edit_from_menu(self) -> None:
        self._begin_edit(self._menu_platform)

    def _begin_edit(self, platform: str) -> None:
        self._commit_editor(save=True)
        index = PLATFORMS.index(platform)
        entry = self._tk.Entry(
            self.canvas,
            bg="#0c0c1a",
            fg="#00d6e8",
            insertbackground="#00d6e8",
            relief=self._tk.FLAT,
            justify="center",
            highlightthickness=0,
            bd=0,
            width=6,
        )
        goal = self._goals.get(platform)
        if goal:
            entry.insert(0, str(goal))
            entry.selection_range(0, self._tk.END)
        entry.place(x=index * (DIAL + GAP) + DIAL / 2, y=GOAL_CY, anchor="center", width=72, height=22)
        entry.update_idletasks()
        entry.focus_force()
        self._focus_keys(entry)
        entry.after(80, lambda: self._focus_keys(entry))
        entry.bind("<Return>", lambda _event: self._commit_editor(save=True))
        entry.bind("<Escape>", lambda _event: self._commit_editor(save=False))
        entry.after(250, lambda: entry.bind("<FocusOut>", lambda _event: self._commit_editor(save=True)))
        self._editor = (platform, entry)

    def _focus_keys(self, widget) -> None:
        """Borderless windows are not focused by the window manager, so claim the keyboard."""
        if sys.platform == "win32":
            return
        import ctypes
        from ctypes import c_int, c_ulong, c_void_p

        try:
            window_id = int(widget.winfo_id())
        except self._tk.TclError:
            return
        x11 = ctypes.CDLL("libX11.so.6")
        x11.XOpenDisplay.restype = c_void_p
        x11.XSetInputFocus.argtypes = [c_void_p, c_ulong, c_int, c_ulong]
        display = x11.XOpenDisplay(None)
        if not display:
            return
        x11.XSetInputFocus(display, c_ulong(window_id), 2, 0)
        x11.XFlush(display)
        x11.XCloseDisplay(display)

    def _commit_editor(self, save: bool) -> None:
        editor = self._editor
        if editor is None:
            return
        platform, entry = editor
        self._editor = None
        text = entry.get().strip().replace(" ", "")
        entry.destroy()
        if save and text.isdigit() and int(text) > 0:
            self._goals[platform] = int(text)
            self._save_state()
        elif save and text == "":
            self._goals.pop(platform, None)
            self._save_state()
        self._draw()

    def _restore_state(self) -> None:
        try:
            data = json.loads(state_path().read_text(encoding="utf-8"))
            x = int(data["x"])
            y = int(data["y"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.root.geometry("+80+80")
            self._draw()
            return
        goals = data.get("goals") if isinstance(data, dict) else None
        if isinstance(goals, dict):
            for platform in PLATFORMS:
                value = goals.get(platform)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    self._goals[platform] = value
        self.root.geometry(f"+{x}+{y}")
        self._draw()

    def _save_state(self) -> None:
        path = state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"x": self.root.winfo_x(), "y": self.root.winfo_y(), "goals": self._goals}
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            return

    def _queue_shape(self) -> None:
        if sys.platform == "win32" or self._shape_queued:
            return
        self._shape_queued = True
        self.root.after(40, self._apply_shape)

    def _apply_shape(self) -> None:
        self._shape_queued = False
        try:
            self.root.update_idletasks()
            if self.root.winfo_width() < 2 or not self._rgba:
                self._queue_shape()
                return
            _shape_alpha(self.root, WIDGET_W, WIDGET_H, self.canvas.winfo_x(), self.canvas.winfo_y())
        except (OSError, ValueError):
            return


def _inside_dial(x: int, y: int) -> bool:
    slot = DIAL + GAP
    limit = (RADIUS - 1.25) * (RADIUS - 1.25)
    for index in range(len(PLATFORMS)):
        dx = x + 0.5 - (index * slot + DIAL / 2)
        dy = y + 0.5 - DIAL / 2
        if dx * dx + dy * dy <= limit:
            return True
    return False


def _join(strips: list[bytes]) -> bytes:
    out = bytearray(WIDGET_W * WIDGET_H * 4)
    for index, pixels in enumerate(strips):
        left = index * (DIAL + GAP)
        for y in range(DIAL):
            source = y * DIAL * 4
            dest = (y * WIDGET_W + left) * 4
            out[dest : dest + DIAL * 4] = pixels[source : source + DIAL * 4]
    return bytes(out)


def _shape_alpha(window, img_w: int, img_h: int, origin_x: int, origin_y: int) -> None:
    """Punch transparent pixels out of the window on X11. Windows uses -transparentcolor."""
    if sys.platform == "win32":
        return
    import ctypes
    from ctypes import c_char_p, c_int, c_uint, c_ulong, c_void_p

    x11 = ctypes.CDLL("libX11.so.6")
    xext = ctypes.CDLL("libXext.so.6")
    x11.XOpenDisplay.restype = c_void_p
    x11.XCreateBitmapFromData.argtypes = [c_void_p, c_ulong, c_char_p, c_uint, c_uint]
    x11.XCreateBitmapFromData.restype = c_ulong
    xext.XShapeCombineMask.argtypes = [c_void_p, c_ulong, c_int, c_int, c_int, c_ulong, c_int]
    x11.XFreePixmap.argtypes = [c_void_p, c_ulong]
    x11.XFlush.argtypes = [c_void_p]
    x11.XQueryTree.argtypes = [
        c_void_p,
        c_ulong,
        ctypes.POINTER(c_ulong),
        ctypes.POINTER(c_ulong),
        ctypes.POINTER(ctypes.POINTER(c_ulong)),
        ctypes.POINTER(c_uint),
    ]
    display = x11.XOpenDisplay(None)
    if not display:
        return
    root = c_ulong()
    parent = c_ulong()
    children = ctypes.POINTER(c_ulong)()
    count = c_uint()
    x11.XQueryTree(
        display,
        c_ulong(window.winfo_id()),
        ctypes.byref(root),
        ctypes.byref(parent),
        ctypes.byref(children),
        ctypes.byref(count),
    )
    target = parent.value or window.winfo_id()
    width, height = window.winfo_width(), window.winfo_height()
    if width < 2 or height < 2:
        x11.XCloseDisplay(display)
        return
    # Pad to bytes and use the server's LSB bit order. A 32-bit pad tears the rim.
    stride = (width + 7) // 8
    mask = bytearray(stride * height)
    for y in range(img_h):
        window_y = origin_y + y
        if not 0 <= window_y < height:
            continue
        for x in range(img_w):
            window_x = origin_x + x
            if not 0 <= window_x < width:
                continue
            if not _inside_dial(x, y):
                continue
            mask[window_y * stride + window_x // 8] |= 1 << (window_x % 8)
    pixmap = x11.XCreateBitmapFromData(display, c_ulong(target), ctypes.create_string_buffer(bytes(mask)), width, height)
    xext.XShapeCombineMask(display, c_ulong(target), 0, 0, 0, pixmap, 0)
    x11.XFlush(display)
    x11.XFreePixmap(display, pixmap)
    x11.XCloseDisplay(display)


if __name__ == "__main__":
    raise SystemExit(main())
