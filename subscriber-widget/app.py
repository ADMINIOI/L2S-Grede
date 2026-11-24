"""Small always-on-top window for public AxisBox subscriber counts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from counts import REFRESH_SECONDS, ProfileCount, fetch_all

BG = "#12141a"
TEXT = "#f4f6fb"
MUTED = "#9aa3b2"
ERROR = "#ffb4b4"
BUTTON = "#2c3444"
BUTTON_ACTIVE = "#3d475c"

ACCENTS = {
    "TikTok": "#fe2c55",
    "Instagram": "#ee4d8b",
    "Facebook": "#4c8dff",
}


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
        from tkinter import font as tkfont
    except ImportError:
        sys.stderr.write("Нужен Python с tkinter. Установите его с python.org и включите tcl/tk.\n")
        return 1
    enable_windows_dpi()
    Widget(tk, tkfont).mainloop()
    return 0


class Widget:
    def __init__(self, tk, tkfont, *, autostart: bool = True) -> None:
        self._tk = tk
        self.root = tk.Tk()
        self.root.title("Подписчики")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        family = self._font_family(tkfont)
        self.font_header = tkfont.Font(family=family, size=13, weight="bold")
        self.font_platform = tkfont.Font(family=family, size=11, weight="bold")
        self.font_handle = tkfont.Font(family=family, size=9)
        self.font_count = tkfont.Font(family=family, size=16, weight="bold")
        self.font_small = tkfont.Font(family=family, size=9)
        self._busy = False
        self._after_id = None
        self._autostart = autostart
        self._rows: dict[str, dict[str, tk.Label]] = {}
        self._build()
        self._restore_position()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<F5>", lambda _event: self.refresh())
        self._use_dark_titlebar()
        if autostart:
            self.root.after(150, self.refresh)

    def mainloop(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self._save_position()
        self.root.destroy()

    def refresh(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.button.configure(state=self._tk.DISABLED, text="…")
        self.status.configure(text="обновление…")
        threading.Thread(target=self._load, daemon=True).start()

    def show(self, results: list[ProfileCount]) -> None:
        if not results:
            self.status.configure(text="нет ответа")
            self._busy = False
            self.button.configure(state=self._tk.NORMAL, text="Обновить")
            if self._autostart:
                self._schedule()
            return
        by_name = {item.platform: item for item in results}
        for platform, row in self._rows.items():
            item = by_name.get(platform)
            if item is None:
                continue
            row["handle"].configure(text=item.handle)
            if item.count is None:
                row["count"].configure(text="—", fg=MUTED)
                row["error"].configure(text=item.error or "счётчик не найден")
                row["error"].grid()
            else:
                row["count"].configure(text=_format_count(item.count), fg=TEXT)
                row["error"].configure(text="")
                row["error"].grid_remove()
        self.status.configure(text=f"обновлено {self._now()}")
        self._busy = False
        self.button.configure(state=self._tk.NORMAL, text="Обновить")
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
        shell = tk.Frame(self.root, bg=BG, padx=14, pady=12)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.grid_columnconfigure(0, minsize=292)

        header = tk.Frame(shell, bg=BG)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="Подписчики", bg=BG, fg=TEXT, font=self.font_header).grid(row=0, column=0, sticky="w")
        self.button = tk.Button(
            header,
            text="Обновить",
            command=self.refresh,
            bg=BUTTON,
            fg=TEXT,
            activebackground=BUTTON_ACTIVE,
            activeforeground=TEXT,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            padx=10,
            pady=4,
            font=self.font_small,
            cursor="hand2",
        )
        self.button.grid(row=0, column=1, sticky="e")

        defaults = {"TikTok": "@axisbox.am", "Instagram": "@axisbox", "Facebook": "…"}
        for index, platform in enumerate(("TikTok", "Instagram", "Facebook"), start=1):
            self._rows[platform] = self._build_row(shell, index, platform, defaults[platform])

        self.status = tk.Label(shell, text="", bg=BG, fg=MUTED, font=self.font_small, anchor="w")
        self.status.grid(row=4, column=0, sticky="ew", pady=(8, 0))

    def _build_row(self, parent, index: int, platform: str, handle_text: str) -> dict:
        tk = self._tk
        frame = tk.Frame(parent, bg=BG)
        frame.grid(row=index, column=0, sticky="ew", pady=(10, 0))
        frame.grid_columnconfigure(1, weight=1)
        accent = tk.Frame(frame, bg=ACCENTS[platform], width=3)
        accent.grid(row=0, column=0, rowspan=3, sticky="ns", padx=(0, 8))
        name = tk.Label(frame, text=platform, bg=BG, fg=TEXT, font=self.font_platform, anchor="w")
        name.grid(row=0, column=1, sticky="w")
        count = tk.Label(frame, text="…", bg=BG, fg=MUTED, font=self.font_count, anchor="e")
        count.grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 0))
        handle = tk.Label(frame, text=handle_text, bg=BG, fg=MUTED, font=self.font_handle, anchor="w")
        handle.grid(row=1, column=1, sticky="w")
        error = tk.Label(frame, text="", bg=BG, fg=ERROR, font=self.font_small, anchor="w", justify="left", wraplength=280)
        error.grid(row=2, column=1, columnspan=2, sticky="w")
        error.grid_remove()
        return {"handle": handle, "count": count, "error": error}

    def _font_family(self, tkfont) -> str:
        try:
            families = set(tkfont.families())
        except self._tk.TclError:
            return "TkDefaultFont"
        for name in ("Segoe UI", "Arial", "DejaVu Sans", "Liberation Sans", "Tahoma"):
            if name in families:
                return name
        return "TkDefaultFont"

    def _now(self) -> str:
        import datetime

        return datetime.datetime.now().strftime("%H:%M")

    def _restore_position(self) -> None:
        path = state_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            x = int(data["x"])
            y = int(data["y"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.root.geometry("+48+48")
            return
        self.root.geometry(f"+{x}+{y}")

    def _save_position(self) -> None:
        path = state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"x": self.root.winfo_x(), "y": self.root.winfo_y()}
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


def _format_count(count: int) -> str:
    return f"{count:,}".replace(",", " ")


if __name__ == "__main__":
    raise SystemExit(main())
