"""Smooth dial images: an antialiased ring, the real network icon, the count, and the goal."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

DIAL = 176
GAP = 18
RADIUS = 80.0
STROKE = 13.0
ICON_SIZE = 42
ICON_CY = 64
COUNT_CY = 104
GOAL_CY = 128
COUNT_PX = 28
GOAL_PX = 14

DISC = (12, 12, 26)
TRACK = (32, 36, 62)
ARC = (0, 214, 232)
COUNT_COLOR = (0, 214, 232)
GOAL_COLOR = (154, 160, 180)
CHROMA = (1, 2, 3)

_ICON_DIR = Path(__file__).resolve().parent / "icons"
_ICONS = {"TikTok": "tiktok.png", "Instagram": "instagram.png", "Facebook": "facebook.png"}
_icon_cache: dict[str, tuple[int, int, bytes]] = {}
_text_cache: dict[tuple[str, int, bool], tuple[int, int, bytes]] = {}


def ring_fraction(count: int | None, goal: int | None) -> float:
    """How much of the ring the current count fills toward a goal."""
    if isinstance(count, bool) or not isinstance(count, int):
        return 0.0
    if isinstance(goal, bool) or not isinstance(goal, int) or goal <= 0:
        return 0.0
    return max(0.0, min(1.0, count / goal))


def render_dial(platform: str, count_text: str, goal_text: str, fraction: float) -> tuple[int, int, bytes]:
    """Return width, height, and straight RGBA for one dial."""
    pixels = _ring_pixels(max(0.0, min(1.0, fraction)))
    _blit_icon(pixels, platform)
    _blit_text(pixels, count_text, COUNT_PX, True, COUNT_COLOR, DIAL / 2, COUNT_CY)
    _blit_text(pixels, goal_text, GOAL_PX, False, GOAL_COLOR, DIAL / 2, GOAL_CY)
    return DIAL, DIAL, pixels


def clip_to_circles(width: int, height: int, rgba: bytes) -> bytes:
    """Keep a clean circular silhouette so the transparent edge is not a sawtooth."""
    out = bytearray(rgba)
    slot = DIAL + GAP
    # Stay inside the solid stroke. The antialiased rim outside it reads as a dark gear.
    limit = (RADIUS - 1.25) * (RADIUS - 1.25)
    for y in range(height):
        for x in range(width):
            inside = False
            for index in range(len(_ICONS)):
                center_x = index * slot + DIAL / 2
                dx = x + 0.5 - center_x
                dy = y + 0.5 - DIAL / 2
                if dx * dx + dy * dy <= limit:
                    inside = True
                    break
            pixel = (y * width + x) * 4
            out[pixel + 3] = 255 if inside else 0
    return bytes(out)


def flatten_chroma(width: int, height: int, rgba: bytes) -> bytes:
    """Opaque RGB. Pixels that are mostly transparent become the window's chroma key."""
    out = bytearray(width * height * 3)
    for index in range(width * height):
        source = index * 4
        alpha = rgba[source + 3]
        dest = index * 3
        if alpha < 128:
            out[dest : dest + 3] = CHROMA
        else:
            out[dest] = rgba[source]
            out[dest + 1] = rgba[source + 1]
            out[dest + 2] = rgba[source + 2]
    return bytes(out)


def encode_png(width: int, height: int, rgb: bytes) -> bytes:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * stride : (y + 1) * stride])
    return _png(width, height, 2, bytes(raw))


def _ring_pixels(fraction: float) -> bytearray:
    scale = 2
    side = DIAL * scale
    center = side / 2
    radius = RADIUS * scale
    stroke = STROKE * scale
    midline = radius - stroke / 2
    inner = midline - stroke / 2
    sweep = fraction * math.tau
    full = fraction >= 0.999
    start_x, start_y = midline * math.sin(0.0), -midline * math.cos(0.0)
    end_x, end_y = midline * math.sin(sweep), -midline * math.cos(sweep)
    cap = stroke / 2 + 0.7
    buffer = bytearray(side * side * 4)
    cover = _cover
    hypot = math.hypot
    atan2 = math.atan2
    tau = math.tau
    for y in range(side):
        dy = y + 0.5 - center
        row = y * side
        for x in range(side):
            dx = x + 0.5 - center
            distance = hypot(dx, dy)
            disc = cover(inner + 1.15 - distance)
            ring = cover(stroke / 2 + 0.7 - abs(distance - midline))
            if disc <= 0 and ring <= 0:
                continue
            red = green = blue = alpha = 0
            if disc > 0:
                red, green, blue, alpha = _over(red, green, blue, alpha, DISC, disc)
            if ring > 0:
                red, green, blue, alpha = _over(red, green, blue, alpha, TRACK, ring)
            arc = 0.0
            if full:
                arc = ring
            elif fraction > 0:
                if atan2(dx, -dy) % tau <= sweep:
                    arc = ring
                arc = max(arc, cover(cap - hypot(dx - start_x, dy - start_y)))
                arc = max(arc, cover(cap - hypot(dx - end_x, dy - end_y)))
            if arc > 0:
                red, green, blue, alpha = _over(red, green, blue, alpha, ARC, arc)
            pixel = (row + x) * 4
            buffer[pixel] = red
            buffer[pixel + 1] = green
            buffer[pixel + 2] = blue
            buffer[pixel + 3] = alpha
    return _downsample(buffer, side, scale)


def _cover(distance: float) -> float:
    if distance <= 0:
        return 0.0
    if distance >= 1:
        return 1.0
    return distance


def _over(red: int, green: int, blue: int, alpha: int, color: tuple[int, int, int], coverage: float) -> tuple[int, int, int, int]:
    amount = int(round(coverage * 255))
    if amount <= 0:
        return red, green, blue, alpha
    if amount >= 255 or alpha == 0:
        return color[0], color[1], color[2], 255 if amount >= 255 else amount
    keep = 255 - amount
    return (
        (color[0] * amount + red * keep) // 255,
        (color[1] * amount + green * keep) // 255,
        (color[2] * amount + blue * keep) // 255,
        amount + (alpha * keep) // 255,
    )


def _downsample(buffer: bytearray, side: int, scale: int) -> bytearray:
    width = side // scale
    out = bytearray(width * width * 4)
    area = scale * scale
    for y in range(width):
        for x in range(width):
            red = green = blue = alpha = 0
            for sy in range(scale):
                row = (y * scale + sy) * side
                for sx in range(scale):
                    pixel = (row + x * scale + sx) * 4
                    red += buffer[pixel]
                    green += buffer[pixel + 1]
                    blue += buffer[pixel + 2]
                    alpha += buffer[pixel + 3]
            pixel = (y * width + x) * 4
            out[pixel] = red // area
            out[pixel + 1] = green // area
            out[pixel + 2] = blue // area
            out[pixel + 3] = alpha // area
    return out


def _blit_icon(pixels: bytearray, platform: str) -> None:
    icon = _load_icon(platform)
    if icon is None:
        return
    width, height, source = icon
    scale = min(ICON_SIZE / width, ICON_SIZE / height)
    draw_w = max(1, int(round(width * scale)))
    draw_h = max(1, int(round(height * scale)))
    left = int(round(DIAL / 2 - draw_w / 2))
    top = int(round(ICON_CY - draw_h / 2))
    for y in range(draw_h):
        if not 0 <= top + y < DIAL:
            continue
        sy = min(height - 1, int((y + 0.5) * height / draw_h))
        for x in range(draw_w):
            if not 0 <= left + x < DIAL:
                continue
            sx = min(width - 1, int((x + 0.5) * width / draw_w))
            sample = (sy * width + sx) * 4
            alpha = source[sample + 3]
            if alpha < 8:
                continue
            _blend(pixels, left + x, top + y, source[sample], source[sample + 1], source[sample + 2], alpha)


def _blit_text(pixels: bytearray, text: str, pixel_size: int, bold: bool, color: tuple[int, int, int], cx: float, cy: float) -> None:
    if not text:
        return
    mask = _text_mask(text, pixel_size, bold)
    if mask is None:
        return
    width, height, alpha = mask
    left = int(round(cx - width / 2))
    top = int(round(cy - height / 2))
    for y in range(height):
        if not 0 <= top + y < DIAL:
            continue
        for x in range(width):
            if not 0 <= left + x < DIAL:
                continue
            coverage = alpha[y * width + x]
            if coverage < 8:
                continue
            _blend(pixels, left + x, top + y, color[0], color[1], color[2], coverage)


def _blend(pixels: bytearray, x: int, y: int, red: int, green: int, blue: int, alpha: int) -> None:
    pixel = (y * DIAL + x) * 4
    keep = 255 - alpha
    pixels[pixel] = (red * alpha + pixels[pixel] * keep) // 255
    pixels[pixel + 1] = (green * alpha + pixels[pixel + 1] * keep) // 255
    pixels[pixel + 2] = (blue * alpha + pixels[pixel + 2] * keep) // 255
    pixels[pixel + 3] = alpha + (pixels[pixel + 3] * keep) // 255


def _load_icon(platform: str) -> tuple[int, int, bytes] | None:
    cached = _icon_cache.get(platform)
    if cached is not None:
        return cached
    path = _ICON_DIR / _ICONS.get(platform, "")
    if not path.is_file():
        return None
    loaded = _decode_png(path.read_bytes())
    if loaded is None:
        return None
    _icon_cache[platform] = loaded
    return loaded


def _text_mask(text: str, pixel_size: int, bold: bool) -> tuple[int, int, bytes] | None:
    key = (text, pixel_size, bold)
    cached = _text_cache.get(key)
    if cached is not None:
        return cached
    rendered = _text_mask_cairo(text, pixel_size, bold) or _text_mask_gdiplus(text, pixel_size, bold)
    if rendered is not None:
        _text_cache[key] = rendered
    return rendered


def _text_mask_cairo(text: str, pixel_size: int, bold: bool) -> tuple[int, int, bytes] | None:
    import ctypes
    from ctypes import c_char_p, c_double, c_int, c_void_p

    try:
        cairo = ctypes.CDLL("libcairo.so.2")
    except OSError:
        return None

    class Extents(ctypes.Structure):
        _fields_ = [
            ("x_bearing", c_double),
            ("y_bearing", c_double),
            ("width", c_double),
            ("height", c_double),
            ("x_advance", c_double),
            ("y_advance", c_double),
        ]

    cairo.cairo_image_surface_create.argtypes = [c_int, c_int, c_int]
    cairo.cairo_image_surface_create.restype = c_void_p
    cairo.cairo_create.argtypes = [c_void_p]
    cairo.cairo_create.restype = c_void_p
    cairo.cairo_select_font_face.argtypes = [c_void_p, c_char_p, c_int, c_int]
    cairo.cairo_set_font_size.argtypes = [c_void_p, c_double]
    cairo.cairo_text_extents.argtypes = [c_void_p, c_char_p, ctypes.POINTER(Extents)]
    cairo.cairo_move_to.argtypes = [c_void_p, c_double, c_double]
    cairo.cairo_show_text.argtypes = [c_void_p, c_char_p]
    cairo.cairo_set_source_rgba.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
    cairo.cairo_image_surface_get_data.argtypes = [c_void_p]
    cairo.cairo_image_surface_get_data.restype = c_void_p
    cairo.cairo_image_surface_get_stride.argtypes = [c_void_p]
    cairo.cairo_image_surface_get_stride.restype = c_int
    cairo.cairo_surface_flush.argtypes = [c_void_p]
    cairo.cairo_destroy.argtypes = [c_void_p]
    cairo.cairo_surface_destroy.argtypes = [c_void_p]

    encoded = text.encode("utf-8")
    scratch = cairo.cairo_image_surface_create(0, 8, 8)
    context = cairo.cairo_create(scratch)
    family = b"DejaVu Sans"
    cairo.cairo_select_font_face(context, family, 0, 1 if bold else 0)
    cairo.cairo_set_font_size(context, float(pixel_size))
    extents = Extents()
    cairo.cairo_text_extents(context, encoded, ctypes.byref(extents))
    cairo.cairo_destroy(context)
    cairo.cairo_surface_destroy(scratch)
    if extents.width <= 0 or extents.height <= 0:
        return None
    pad = 4
    width = max(1, int(math.ceil(extents.width)) + pad * 2)
    height = max(1, int(math.ceil(extents.height)) + pad * 2)
    surface = cairo.cairo_image_surface_create(0, width, height)
    context = cairo.cairo_create(surface)
    cairo.cairo_select_font_face(context, family, 0, 1 if bold else 0)
    cairo.cairo_set_font_size(context, float(pixel_size))
    cairo.cairo_set_source_rgba(context, 1, 1, 1, 1)
    cairo.cairo_move_to(context, pad - extents.x_bearing, pad - extents.y_bearing)
    cairo.cairo_show_text(context, encoded)
    cairo.cairo_surface_flush(surface)
    stride = cairo.cairo_image_surface_get_stride(surface)
    raw = ctypes.string_at(cairo.cairo_image_surface_get_data(surface), stride * height)
    alpha = bytearray(width * height)
    for y in range(height):
        row = y * stride
        for x in range(width):
            alpha[y * width + x] = raw[row + x * 4 + 3]
    cairo.cairo_destroy(context)
    cairo.cairo_surface_destroy(surface)
    return _trim_alpha(width, height, alpha)


def _text_mask_gdiplus(text: str, pixel_size: int, bold: bool) -> tuple[int, int, bytes] | None:
    import sys

    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import byref, c_float, c_int, c_ubyte, c_uint, c_uint32, c_void_p, c_wchar_p

    gdiplus = ctypes.windll.gdiplus

    class StartupInput(ctypes.Structure):
        _fields_ = [
            ("GdiplusVersion", c_uint32),
            ("DebugEventCallback", c_void_p),
            ("SuppressBackgroundThread", c_int),
            ("SuppressExternalCodecs", c_int),
        ]

    class RectF(ctypes.Structure):
        _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]

    class Rect(ctypes.Structure):
        _fields_ = [("x", c_int), ("y", c_int), ("w", c_int), ("h", c_int)]

    class BitmapData(ctypes.Structure):
        _fields_ = [
            ("Width", c_uint),
            ("Height", c_uint),
            ("Stride", c_int),
            ("PixelFormat", c_int),
            ("Scan0", c_void_p),
            ("Reserved", c_void_p),
        ]

    token = ctypes.c_ulong()
    if gdiplus.GdiplusStartup(byref(token), byref(StartupInput(1, None, 0, 0)), None) != 0:
        return None
    width = max(8, pixel_size * (len(text) + 1))
    height = max(8, int(pixel_size * 2))
    bitmap = c_void_p()
    graphics = c_void_p()
    family = c_void_p()
    font = c_void_p()
    brush = c_void_p()
    pixel_format = 0x26200A
    try:
        if gdiplus.GdipCreateBitmapFromScan0(width, height, 0, pixel_format, None, byref(bitmap)) != 0:
            return None
        if gdiplus.GdipGetImageGraphicsContext(bitmap, byref(graphics)) != 0:
            return None
        gdiplus.GdipSetTextRenderingHint(graphics, 4)
        gdiplus.GdipSetSmoothingMode(graphics, 4)
        if gdiplus.GdipCreateFontFamilyFromName(c_wchar_p("Segoe UI"), None, byref(family)) != 0:
            return None
        if gdiplus.GdipCreateFont(family, c_float(pixel_size), 1 if bold else 0, 2, byref(font)) != 0:
            return None
        if gdiplus.GdipCreateSolidFill(0xFFFFFFFF, byref(brush)) != 0:
            return None
        rect = RectF(0, 0, float(width), float(height))
        if gdiplus.GdipDrawString(graphics, c_wchar_p(text), len(text), font, byref(rect), None, brush) != 0:
            return None
        locked = BitmapData()
        if gdiplus.GdipBitmapLockBits(bitmap, byref(Rect(0, 0, width, height)), 1, pixel_format, byref(locked)) != 0:
            return None
        raw = ctypes.string_at(locked.Scan0, abs(locked.Stride) * height)
        alpha = bytearray(width * height)
        stride = locked.Stride
        for y in range(height):
            row = y * stride
            for x in range(width):
                alpha[y * width + x] = raw[row + x * 4 + 3]
        gdiplus.GdipBitmapUnlockBits(bitmap, byref(locked))
        return _trim_alpha(width, height, alpha)
    finally:
        if brush:
            gdiplus.GdipDeleteBrush(brush)
        if font:
            gdiplus.GdipDeleteFont(font)
        if family:
            gdiplus.GdipDeleteFontFamily(family)
        if graphics:
            gdiplus.GdipDeleteGraphics(graphics)
        if bitmap:
            gdiplus.GdipDisposeImage(bitmap)
        gdiplus.GdiplusShutdown(token)


def _trim_alpha(width: int, height: int, alpha: bytearray) -> tuple[int, int, bytes] | None:
    min_x, min_y, max_x, max_y = width, height, -1, -1
    for y in range(height):
        row = y * width
        for x in range(width):
            if alpha[row + x] > 12:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < min_x:
        return None
    trimmed_w = max_x - min_x + 1
    trimmed_h = max_y - min_y + 1
    trimmed = bytearray(trimmed_w * trimmed_h)
    for y in range(trimmed_h):
        source = (min_y + y) * width + min_x
        trimmed[y * trimmed_w : (y + 1) * trimmed_w] = alpha[source : source + trimmed_w]
    return trimmed_w, trimmed_h, bytes(trimmed)


def _decode_png(data: bytes) -> tuple[int, int, bytes] | None:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    idat = b""
    width = height = color = 0
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + size]
        pos += 12 + size
        if kind == b"IHDR":
            width, height, _bit, color = struct.unpack(">IIBB", chunk[:10])
        elif kind == b"IDAT":
            idat += chunk
        elif kind == b"IEND":
            break
    if color not in (2, 6) or width < 1 or height < 1:
        return None
    channels = 4 if color == 6 else 3
    raw = zlib.decompress(idat)
    stride = width * channels
    rows: list[bytearray] = []
    index = 0
    previous = bytearray(stride)
    for _y in range(height):
        filter_type = raw[index]
        index += 1
        row = bytearray(raw[index : index + stride])
        index += stride
        if filter_type == 1:
            for x in range(stride):
                row[x] = (row[x] + (row[x - channels] if x >= channels else 0)) & 255
        elif filter_type == 2:
            for x in range(stride):
                row[x] = (row[x] + previous[x]) & 255
        elif filter_type == 3:
            for x in range(stride):
                left = row[x - channels] if x >= channels else 0
                row[x] = (row[x] + ((left + previous[x]) // 2)) & 255
        elif filter_type == 4:
            for x in range(stride):
                left = row[x - channels] if x >= channels else 0
                up = previous[x]
                up_left = previous[x - channels] if x >= channels else 0
                row[x] = (row[x] + _paeth(left, up, up_left)) & 255
        previous = row
        rows.append(row)
    rgba = bytearray(width * height * 4)
    for y, row in enumerate(rows):
        for x in range(width):
            source = x * channels
            dest = (y * width + x) * 4
            rgba[dest] = row[source]
            rgba[dest + 1] = row[source + 1]
            rgba[dest + 2] = row[source + 2]
            rgba[dest + 3] = row[source + 3] if channels == 4 else 255
    return width, height, bytes(rgba)


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    distance_left = abs(estimate - left)
    distance_up = abs(estimate - up)
    distance_up_left = abs(estimate - up_left)
    if distance_left <= distance_up and distance_left <= distance_up_left:
        return left
    if distance_up <= distance_up_left:
        return up
    return up_left


def _png(width: int, height: int, color_type: int, raw: bytes) -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
