"""Transparent always-on-top rings for public AxisBox subscriber counts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from counts import REFRESH_SECONDS, ProfileCount, fetch_all

# Windows treats this exact color as transparent. It is not used in the rings.
CHROMA = "#010203"
TRACK = "#656667"
ARC_START = "#FFC0A3"
ARC_END = "#FFE6A8"
CAPSULE = "#42CB7E"
CAPSULE_TEXT = "#141414"
PLATFORMS = ("TikTok", "Instagram", "Facebook")

DIAL_W = 136
DIAL_H = 184
RING_CX = DIAL_W / 2
RING_CY = 68
RING_R = 52
RING_STROKE = 15
CAPSULE_CY = 152
CAPSULE_H = 34


def state_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "axisbox-subscriber-widget" / "window.json"


def ring_fraction(count: int | None, goal: int | None) -> float:
    """How much of the ring the current count fills toward a goal."""
    if isinstance(count, bool) or not isinstance(count, int):
        return 0.0
    if isinstance(goal, bool) or not isinstance(goal, int) or goal <= 0:
        return 0.0
    return max(0.0, min(1.0, count / goal))


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
        from tkinter import font as tkfont
    except ImportError:
        sys.stderr.write("Нужен Python с tkinter. Установите его с python.org и включите tcl/tk.\n")
        return 1
    enable_windows_dpi()
    Widget(tk, tkfont).mainloop()
    return 0


def _mix(start: str, end: str, t: float) -> str:
    def channel(text: str, offset: int) -> int:
        return int(text[offset : offset + 2], 16)

    return "#{:02x}{:02x}{:02x}".format(
        round(channel(start, 1) + (channel(end, 1) - channel(start, 1)) * t),
        round(channel(start, 3) + (channel(end, 3) - channel(start, 3)) * t),
        round(channel(start, 5) + (channel(end, 5) - channel(start, 5)) * t),
    )


def _point(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    radians = math.radians(angle)
    return cx + radius * math.cos(radians), cy - radius * math.sin(radians)


class Widget:
    def __init__(self, tk, tkfont, *, autostart: bool = True) -> None:
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
        self._font_family = self._pick_font(tkfont)
        self.font_count = tkfont.Font(family=self._font_family, size=13, weight="bold")
        self._busy = False
        self._after_id = None
        self._autostart = autostart
        self._goals: dict[str, int] = {}
        self._results: dict[str, ProfileCount] = {}
        self._canvases: dict[str, tk.Canvas] = {}
        self._editor = None
        self._drag = None
        self._shape_queued = False
        self._capsule_widths: dict[object, int] = {}
        self._build()
        self._restore_state()
        self._use_dark_titlebar()
        self.root.bind("<F5>", lambda _event: self.refresh())
        self.root.bind("<Button-3>", self._popup)
        self._menu = tk.Menu(self.root, tearoff=0)
        self._menu.add_command(label="Обновить", command=self.refresh)
        self._menu.add_command(label="Задать цель", command=self._edit_from_menu)
        self._menu.add_separator()
        self._menu.add_command(label="Закрыть", command=self.close)
        self._menu_platform = PLATFORMS[0]
        if autostart:
            self.root.after(150, self.refresh)
        else:
            self._draw()

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

    def _build(self) -> None:
        tk = self._tk
        shell = tk.Frame(self.root, bg=CHROMA)
        shell.pack()
        for index, platform in enumerate(PLATFORMS):
            canvas = tk.Canvas(
                shell,
                width=DIAL_W,
                height=DIAL_H,
                bg=CHROMA,
                highlightthickness=0,
                bd=0,
                cursor="hand2",
            )
            canvas.grid(row=0, column=index, padx=6)
            canvas.bind("<ButtonPress-1>", lambda event, name=platform: self._press(name, event))
            canvas.bind("<B1-Motion>", self._drag_window)
            canvas.bind("<ButtonRelease-1>", lambda event, name=platform: self._release(name, event))
            canvas.bind("<Button-3>", lambda event, name=platform: self._popup(event, name))
            self._canvases[platform] = canvas

    def _draw(self) -> None:
        for platform, canvas in self._canvases.items():
            canvas.delete("all")
            item = self._results.get(platform)
            count = item.count if item is not None else None
            goal = self._goals.get(platform)
            self._draw_ring(canvas, ring_fraction(count, goal))
            self._draw_logo(canvas, platform)
            self._draw_capsule(canvas, self._count_text(item))
        self._queue_shape()

    def _count_text(self, item: ProfileCount | None) -> str:
        if item is None:
            return "…"
        if item.count is None:
            return "—"
        return f"{item.count:,}".replace(",", " ")

    def _draw_ring(self, canvas, fraction: float) -> None:
        radius = RING_R - RING_STROKE / 2
        canvas.create_oval(
            RING_CX - radius,
            RING_CY - radius,
            RING_CX + radius,
            RING_CY + radius,
            outline=TRACK,
            width=RING_STROKE,
        )
        if fraction <= 0:
            return
        # Clockwise from the lower left, matching the sample ring.
        start = 225.0
        sweep = -359.8 * fraction
        steps = max(24, int(abs(sweep) / 4))
        for index in range(steps):
            angle_a = start + sweep * index / steps
            angle_b = start + sweep * (index + 1) / steps
            x0, y0 = _point(RING_CX, RING_CY, radius, angle_a)
            x1, y1 = _point(RING_CX, RING_CY, radius, angle_b)
            canvas.create_line(
                x0,
                y0,
                x1,
                y1,
                fill=_mix(ARC_START, ARC_END, index / max(1, steps - 1)),
                width=RING_STROKE,
                capstyle=self._tk.ROUND,
            )

    def _draw_logo(self, canvas, platform: str) -> None:
        if platform == "TikTok":
            self._logo_tiktok(canvas)
        elif platform == "Instagram":
            self._logo_instagram(canvas)
        else:
            self._logo_facebook(canvas)

    def _logo_tiktok(self, canvas) -> None:
        for dx, dy, color in ((-2.4, -1.6, "#25F4EE"), (2.4, 1.8, "#FE2C55"), (0, 0, "#161616")):
            self._note(canvas, RING_CX, RING_CY - 2, dx, dy, color)

    def _note(self, canvas, cx: float, cy: float, dx: float, dy: float, color: str) -> None:
        canvas.create_oval(cx - 13 + dx, cy + 2 + dy, cx + 3 + dx, cy + 16 + dy, fill=color, outline="")
        canvas.create_rectangle(cx + 0.5 + dx, cy - 15 + dy, cx + 4.5 + dx, cy + 8 + dy, fill=color, outline="")
        canvas.create_polygon(
            cx + 4.5 + dx,
            cy - 15 + dy,
            cx + 16 + dx,
            cy - 7 + dy,
            cx + 16 + dx,
            cy - 3 + dy,
            cx + 4.5 + dx,
            cy - 9 + dy,
            fill=color,
            outline="",
        )

    def _logo_instagram(self, canvas) -> None:
        self._round_rect(canvas, RING_CX - 18, RING_CY - 18, RING_CX + 18, RING_CY + 18, 8, "#E1306C")
        canvas.create_oval(RING_CX - 9, RING_CY - 9, RING_CX + 9, RING_CY + 9, outline="#ffffff", width=2)
        canvas.create_oval(RING_CX - 3.5, RING_CY - 3.5, RING_CX + 3.5, RING_CY + 3.5, fill="#ffffff", outline="")
        canvas.create_oval(RING_CX + 7, RING_CY - 12, RING_CX + 10, RING_CY - 9, fill="#ffffff", outline="")

    def _logo_facebook(self, canvas) -> None:
        canvas.create_oval(RING_CX - 18, RING_CY - 18, RING_CX + 18, RING_CY + 18, fill="#1877F2", outline="")
        canvas.create_text(
            RING_CX + 1,
            RING_CY + 1,
            text="f",
            fill="#ffffff",
            font=(self._font_family, 22, "bold"),
        )

    def _round_rect(self, canvas, x1: float, y1: float, x2: float, y2: float, radius: float, color: str) -> None:
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        canvas.create_polygon(points, smooth=True, fill=color, outline="")

    def _draw_capsule(self, canvas, text: str) -> None:
        width = max(78, self.font_count.measure(text) + 28)
        self._capsule_widths[canvas] = width
        height = CAPSULE_H
        cx, cy = RING_CX, CAPSULE_CY
        left, right = cx - width / 2, cx + width / 2
        top, bottom = cy - height / 2, cy + height / 2
        canvas.create_oval(left, top, left + height, bottom, fill=CAPSULE, outline="")
        canvas.create_oval(right - height, top, right, bottom, fill=CAPSULE, outline="")
        canvas.create_rectangle(left + height / 2, top, right - height / 2, bottom, fill=CAPSULE, outline="")
        canvas.create_text(cx, cy, text=text, fill=CAPSULE_TEXT, font=self.font_count)

    def _capsule_hit(self, y: float) -> bool:
        return abs(y - CAPSULE_CY) <= CAPSULE_H / 2 + 4

    def _press(self, platform: str, event) -> None:
        self._drag = {
            "platform": platform,
            "x": event.x_root,
            "y": event.y_root,
            "win_x": self.root.winfo_x(),
            "win_y": self.root.winfo_y(),
            "moved": False,
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

    def _release(self, platform: str, event) -> None:
        drag = self._drag
        self._drag = None
        if not drag or drag.get("moved") or drag.get("platform") != platform:
            return
        if self._capsule_hit(drag["local_y"]):
            self._begin_edit(platform)

    def _popup(self, event, platform: str | None = None) -> None:
        self._menu_platform = platform or self._menu_platform
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()
        return "break"

    def _edit_from_menu(self) -> None:
        self._begin_edit(self._menu_platform)

    def _begin_edit(self, platform: str) -> None:
        self._commit_editor(save=True)
        canvas = self._canvases[platform]
        entry = self._tk.Entry(
            canvas,
            bg=CAPSULE,
            fg=CAPSULE_TEXT,
            insertbackground=CAPSULE_TEXT,
            relief=self._tk.FLAT,
            justify="center",
            font=self.font_count,
            highlightthickness=0,
            bd=0,
            width=6,
        )
        goal = self._goals.get(platform)
        if goal:
            entry.insert(0, str(goal))
            entry.selection_range(0, self._tk.END)
        entry.place(x=RING_CX, y=CAPSULE_CY, anchor="center", width=72, height=24)
        entry.update_idletasks()
        entry.focus_force()
        self._focus_keys(entry)
        entry.after(80, lambda: self._focus_keys(entry))
        entry.bind("<Return>", lambda _event: self._commit_editor(save=True))
        entry.bind("<Escape>", lambda _event: self._commit_editor(save=False))
        # The click that opened the field can fire FocusOut before the user types.
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

    def _pick_font(self, tkfont) -> str:
        try:
            families = set(tkfont.families())
        except self._tk.TclError:
            return "TkDefaultFont"
        for name in ("Segoe UI", "Arial", "DejaVu Sans", "Liberation Sans", "Tahoma"):
            if name in families:
                return name
        return "TkDefaultFont"

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

    def _use_dark_titlebar(self) -> None:
        if sys.platform != "win32":
            return
        import ctypes

        self.root.update_idletasks()
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            value = ctypes.c_int(1)
            for attribute in (20, 19):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
        except (AttributeError, OSError):
            return

    def _dial_boxes(self) -> list[tuple[float, float, float]]:
        boxes = []
        for index, canvas in enumerate(self._canvases.values()):
            origin_x = 6 + index * (DIAL_W + 12)
            boxes.append((origin_x + RING_CX, RING_CY, float(self._capsule_widths.get(canvas, 78))))
        return boxes

    def _queue_shape(self) -> None:
        if sys.platform == "win32" or self._shape_queued:
            return
        self._shape_queued = True
        self.root.after(40, self._apply_shape)

    def _apply_shape(self) -> None:
        self._shape_queued = False
        try:
            self.root.update_idletasks()
            if self.root.winfo_width() < 2:
                self._queue_shape()
                return
            _shape_window(self.root, self._dial_boxes())
        except (OSError, ValueError):
            return


def _sample(buffer: bytes, row: int, x: int, step: int) -> tuple[int, int, int]:
    pixel = row + x * step
    return buffer[pixel + 2], buffer[pixel + 1], buffer[pixel]


def _fringe(color: tuple[int, int, int]) -> bool:
    """Anti-aliased pixels blended into the chroma key. They read as black spikes."""
    return max(color) < 70


def _pill_contains(dx: float, y: float, capsule_width: float) -> tuple[bool, bool]:
    """Return whether the point is inside the capsule, and whether it sits on the rim."""
    radius = CAPSULE_H / 2
    body = capsule_width / 2 - radius
    cap_dy = y - CAPSULE_CY
    if abs(cap_dy) > radius:
        return False, False
    if abs(dx) <= max(0.0, body):
        return True, abs(cap_dy) > radius - 2
    cap_dx = abs(dx) - body
    distance = (cap_dx * cap_dx + cap_dy * cap_dy) ** 0.5
    return distance <= radius, distance > radius - 2


def _pixel_visible(x: int, y: int, dials: list[tuple[float, float, float]], buffer: bytes, row: int, step: int) -> bool:
    """Keep a smooth ring and capsule, and only the opaque logo pixels inside the ring."""
    outer = RING_R + 1
    inner = RING_R - RING_STROKE - 1
    color = _sample(buffer, row, x, step)
    for center_x, center_y, capsule_width in dials:
        dx = x - center_x
        dy = y - center_y
        distance = (dx * dx + dy * dy) ** 0.5
        if inner <= distance <= outer:
            on_rim = distance > outer - 2 or distance < inner + 2
            return not (on_rim and _fringe(color))
        inside, on_rim = _pill_contains(dx, y, capsule_width)
        if inside:
            # The digits are near-black. Only the anti-aliased rim is dropped.
            return not (on_rim and _fringe(color))
        if distance < inner and abs(color[0] - 1) + abs(color[1] - 2) + abs(color[2] - 3) > 30:
            return True
    return False


def _shape_window(window, dials: list[tuple[float, float, float]]) -> None:
    """Punch the chroma key out of the window on X11. Windows uses -transparentcolor."""
    if sys.platform == "win32":
        return
    import ctypes
    from ctypes import POINTER, Structure, c_char_p, c_int, c_uint, c_ulong, c_void_p

    class XImage(Structure):
        _fields_ = [
            ("width", c_int), ("height", c_int), ("xoffset", c_int), ("format", c_int),
            ("data", c_void_p), ("byte_order", c_int), ("bitmap_unit", c_int),
            ("bitmap_bit_order", c_int), ("bitmap_pad", c_int), ("depth", c_int),
            ("bytes_per_line", c_int), ("bits_per_pixel", c_int),
            ("red_mask", c_ulong), ("green_mask", c_ulong), ("blue_mask", c_ulong),
        ]

    x11 = ctypes.CDLL("libX11.so.6")
    xext = ctypes.CDLL("libXext.so.6")
    x11.XOpenDisplay.restype = c_void_p
    x11.XGetImage.restype = POINTER(XImage)
    x11.XCreateBitmapFromData.argtypes = [c_void_p, c_ulong, c_char_p, c_uint, c_uint]
    x11.XCreateBitmapFromData.restype = c_ulong
    xext.XShapeCombineMask.argtypes = [c_void_p, c_ulong, c_int, c_int, c_int, c_ulong, c_int]
    x11.XFreePixmap.argtypes = [c_void_p, c_ulong]
    x11.XDestroyImage.argtypes = [POINTER(XImage)]
    x11.XFlush.argtypes = [c_void_p]
    x11.XQueryTree.argtypes = [
        c_void_p, c_ulong, POINTER(c_ulong), POINTER(c_ulong), POINTER(POINTER(c_ulong)), POINTER(c_uint)
    ]
    display = x11.XOpenDisplay(None)
    if not display:
        return
    root = c_ulong()
    parent = c_ulong()
    children = POINTER(c_ulong)()
    count = c_uint()
    x11.XQueryTree(display, c_ulong(window.winfo_id()), ctypes.byref(root), ctypes.byref(parent), ctypes.byref(children), ctypes.byref(count))
    target = parent.value or window.winfo_id()
    width, height = window.winfo_width(), window.winfo_height()
    if width < 2 or height < 2:
        x11.XCloseDisplay(display)
        return
    image = x11.XGetImage(display, c_ulong(target), 0, 0, width, height, c_ulong(0xFFFFFFFF), 2)
    if not image:
        x11.XCloseDisplay(display)
        return
    info = image.contents
    buffer = ctypes.string_at(info.data, info.bytes_per_line * info.height)
    step = max(1, info.bits_per_pixel // 8)
    stride = (width + 7) // 8
    mask = bytearray(stride * height)
    for y in range(height):
        row = y * info.bytes_per_line
        for x in range(width):
            if not _pixel_visible(x, y, dials, buffer, row, step):
                continue
            mask[y * stride + x // 8] |= 0x80 >> (x % 8)
    pixmap = x11.XCreateBitmapFromData(display, c_ulong(target), ctypes.create_string_buffer(bytes(mask)), width, height)
    xext.XShapeCombineMask(display, c_ulong(target), 0, 0, 0, pixmap, 0)
    x11.XFlush(display)
    x11.XFreePixmap(display, pixmap)
    x11.XDestroyImage(image)
    x11.XCloseDisplay(display)


if __name__ == "__main__":
    raise SystemExit(main())
