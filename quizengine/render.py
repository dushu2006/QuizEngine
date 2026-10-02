"""Scene rasterizer: turns a :class:`~quizengine.scenes.Scene` into real pixels.

Uses Pillow so fixtures are genuine images -- the preprocessing pipeline, OCR
adapters, template matching and pixel-diff verification all run on actual
raster data, not on mocks.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import Box
from .scenes import Scene, SceneOption, SceneText, blend

_FONT_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
)


@functools.lru_cache(maxsize=1)
def _font_paths() -> Optional[Tuple[str, str]]:
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            return (regular, bold if Path(bold).exists() else regular)
    return None


@functools.lru_cache(maxsize=128)
def _font(px: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    px = max(6, int(px))
    paths = _font_paths()
    if paths is not None:
        try:
            return ImageFont.truetype(paths[1] if bold else paths[0], px)
        except Exception:  # pragma: no cover - corrupt font file
            pass
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # pragma: no cover - Pillow < 10
        return ImageFont.load_default()


def render_scene(scene: Scene) -> np.ndarray:
    """Render ``scene`` to an ``H x W x 3`` uint8 RGB array."""
    from PIL import Image, ImageDraw

    width, height = scene.width, scene.height
    image = Image.new("RGB", (width, height), tuple(scene.background))
    draw = ImageDraw.Draw(image)

    if scene.screen_role == "blank":
        # Uniform frame: std-dev 0 -> capture validation must reject it (FR-7.1.2).
        return np.asarray(image, dtype=np.uint8)

    if scene.screen_role == "lock_screen":
        _draw_lock_screen(draw, scene)

    for rect in scene.rects:
        _draw_rect(draw, rect, scene)

    _draw_options(draw, scene)

    if scene.question_text and scene.screen_role not in {"end_state"}:
        region = scene.question_region or _fallback_question_region(scene)
        _draw_text(
            draw,
            SceneText(
                text=scene.question_text,
                box=region,
                font_px=int(scene.question_font_px * scene.zoom),
                color=(17, 17, 17) if scene.theme == "light" else (238, 238, 242),
                role="question",
                bold=True,
                confidence=scene.question_confidence,
                contrast=scene.question_contrast,
            ),
            scene,
            wrap=True,
        )

    for text in scene.texts:
        _draw_text(draw, text, scene, wrap=text.role in {"question", "result", "header"})

    _draw_navigation(draw, scene)
    _draw_overlays(draw, scene)

    return np.asarray(image, dtype=np.uint8)


def render_uniform(size: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """A deliberately invalid frame (used to test blank-frame rejection)."""
    array = np.zeros((int(size[1]), int(size[0]), 3), dtype=np.uint8)
    array[:, :] = color
    return array


def render_noise(size: Tuple[int, int], seed: int = 0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(int(size[1]), int(size[0]), 3), dtype=np.uint8)
    if scale != 1.0:
        array = np.clip(array.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return array


# --------------------------------------------------------------------------- #
# drawing primitives
# --------------------------------------------------------------------------- #
def _draw_rect(draw: Any, spec: Dict[str, Any], scene: Scene) -> None:
    box = tuple(spec.get("box", (0, 0, 0, 0)))
    fill = spec.get("fill")
    outline = spec.get("outline")
    width = int(spec.get("width", 1))
    radius = int(spec.get("radius", 0))
    if fill is not None:
        if radius:
            draw.rounded_rectangle(box, radius=radius, fill=tuple(fill))
        else:
            draw.rectangle(box, fill=tuple(fill))
    if outline is not None:
        if radius:
            draw.rounded_rectangle(box, radius=radius, outline=tuple(outline), width=width)
        else:
            draw.rectangle(box, outline=tuple(outline), width=width)


def _draw_options(draw: Any, scene: Scene) -> None:
    accent = (37, 99, 235) if scene.theme == "light" else (96, 165, 250)
    border = (203, 208, 216) if scene.theme == "light" else (70, 76, 90)
    card_fill = (255, 255, 255) if scene.theme == "light" else (32, 34, 42)
    text_color = (17, 17, 17) if scene.theme == "light" else (238, 238, 242)

    for option in scene.options:
        x, y, w, h = option.hit_box
        selected = option.selected_marker.value != "none"
        style = option.style

        if style in {"card", "tile"}:
            fill = tuple(card_fill) if not selected else _tint(card_fill, accent, 0.16)
            outline = tuple(accent) if selected else tuple(border)
            draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=fill, outline=outline, width=2 if selected else 1)
        elif style == "button":
            fill = tuple(accent) if selected else tuple(card_fill)
            draw.rounded_rectangle((x, y, x + w, y + h), radius=6, fill=fill, outline=tuple(border), width=1)
        elif style == "text_only" and selected:
            # Real "text-only button" selected states: an accent bar plus accent text.
            bar = max(3, int(h * 0.08))
            draw.rectangle((x, y, x + bar, y + h), fill=tuple(accent))
            draw.rectangle((x, y + h - bar, x + w, y + h), fill=tuple(accent))

        if style == "radio":
            marker_size = max(14, min(int(h * 0.34), 22))
            marker_box = option.marker_box or (
                x + max(8, int(h * 0.2)),
                y + (h - marker_size) // 2,
                marker_size,
                marker_size,
            )
            mx, my, mw, mh = marker_box
            draw.ellipse((mx, my, mx + mw, my + mh), outline=tuple(accent if selected else border), width=2)
            if selected and option.selected_marker.value in {"dot", "highlight"}:
                inset = max(3, min(mw, mh) // 4)
                draw.ellipse((mx + inset, my + inset, mx + mw - inset, my + mh - inset), fill=tuple(accent))
        elif style == "checkbox":
            marker_size = max(14, min(int(h * 0.34), 22))
            marker_box = option.marker_box or (
                x + max(8, int(h * 0.2)),
                y + (h - marker_size) // 2,
                marker_size,
                marker_size,
            )
            mx, my, mw, mh = marker_box
            draw.rectangle((mx, my, mx + mw, my + mh), outline=tuple(accent if selected else border), width=2)
            if selected and option.selected_marker.value in {"check", "highlight"}:
                draw.line((mx + 3, my + mh // 2, mx + mw // 2, my + mh - 4), fill=tuple(accent), width=2)
                draw.line((mx + mw // 2, my + mh - 4, mx + mw - 3, my + 3), fill=tuple(accent), width=2)

        text_box = option.text_box or _option_text_box(option)
        color = tuple(option.color) if option.color != (17, 17, 17) else text_color
        if style == "button" and selected:
            color = (255, 255, 255)
        elif style == "text_only" and selected:
            color = tuple(accent)
        _draw_text(
            draw,
            SceneText(
                text=option.text,
                box=text_box,
                font_px=int(option.font_px * scene.zoom),
                color=color,  # type: ignore[arg-type]
                role="option",
                confidence=option.confidence,
                contrast=option.contrast,
            ),
            scene,
            wrap=False,
        )


def _draw_navigation(draw: Any, scene: Scene) -> None:
    accent = (37, 99, 235) if scene.theme == "light" else (96, 165, 250)
    muted = (226, 229, 234) if scene.theme == "light" else (58, 62, 74)
    text_color = (17, 17, 17) if scene.theme == "light" else (238, 238, 242)

    for button in (scene.navigation.prev_btn, scene.navigation.next_btn, scene.navigation.submit_btn):
        if button is None:
            continue
        x, y, w, h = button.box
        is_primary = button.text.strip().lower() in {"next", "continue", "submit", "finish", "next question"}
        fill = tuple(accent) if (is_primary and button.enabled) else tuple(muted)
        draw.rounded_rectangle((x, y, x + w, y + h), radius=6, fill=fill, width=0)
        color = (255, 255, 255) if (is_primary and button.enabled) else text_color
        if not button.enabled:
            color = blend(color, tuple(muted), 0.45)  # type: ignore[arg-type]
        _draw_text(
            draw,
            SceneText(
                text=button.text,
                box=_inset(button.box, 6),
                font_px=int(button.font_px * scene.zoom),
                color=color,  # type: ignore[arg-type]
                role="nav",
                align="center",
            ),
            scene,
            wrap=False,
        )

    if scene.navigation.progress_text and scene.navigation.progress_box:
        _draw_text(
            draw,
            SceneText(
                text=scene.navigation.progress_text,
                box=scene.navigation.progress_box,
                font_px=int(14 * scene.zoom),
                color=tuple(blend(text_color, tuple(scene.background), 0.55)),  # type: ignore[arg-type]
                role="progress",
                align="right",
            ),
            scene,
            wrap=False,
        )


def _draw_overlays(draw: Any, scene: Scene) -> None:
    for overlay in scene.overlays:
        x, y, w, h = overlay.box
        draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=tuple(overlay.background), width=0)
        if overlay.text:
            _draw_text(
                draw,
                SceneText(
                    text=overlay.text,
                    box=_inset(overlay.box, 10),
                    font_px=int(overlay.font_px * scene.zoom),
                    color=tuple(overlay.color),  # type: ignore[arg-type]
                    role="overlay",
                ),
                scene,
                wrap=True,
            )
        if overlay.close_btn is not None:
            cx, cy, cw, ch = overlay.close_btn.box
            draw.rectangle((cx, cy, cx + cw, cy + ch), outline=(200, 200, 210), width=1)
            _draw_text(
                draw,
                SceneText(
                    text=overlay.close_btn.text or "X",
                    box=overlay.close_btn.box,
                    font_px=int(overlay.close_btn.font_px),
                    color=(240, 240, 245),  # type: ignore[arg-type]
                    role="overlay",
                    align="center",
                ),
                scene,
                wrap=False,
            )


def _draw_lock_screen(draw: Any, scene: Scene) -> None:
    """Windows-style lock / secure desktop signature used by AC-7.1.1."""
    width, height = scene.width, scene.height
    for band in range(0, height, max(1, height // 12)):
        shade = 12 + int(18 * (band / max(1, height)))
        draw.rectangle((0, band, width, band + max(1, height // 12)), fill=(shade, shade, shade + 6))
    clock_box = (width // 2 - 160, height // 3, width // 2 + 160, height // 3 + 90)
    _draw_text(
        draw,
        SceneText(text="03:14", box=clock_box, font_px=64, color=(235, 235, 240), role="lock", align="center"),  # type: ignore[arg-type]
        scene,
        wrap=False,
    )
    prompt_box = (width // 2 - 260, int(height * 0.72), width // 2 + 260, int(height * 0.72) + 40)
    _draw_text(
        draw,
        SceneText(
            text="Press Ctrl+Alt+Del to unlock",
            box=prompt_box,
            font_px=20,
            color=(220, 220, 228),  # type: ignore[arg-type]
            role="lock",
            align="center",
        ),
        scene,
        wrap=False,
    )
    for text in scene.texts:
        _draw_text(draw, text, scene, wrap=False)


def _draw_text(draw: Any, text: SceneText, scene: Scene, *, wrap: bool) -> None:
    if not text.text:
        return
    font = _font(text.font_px, text.bold)
    color = tuple(text.color)
    if text.contrast < 1.0:
        color = tuple(blend(color, tuple(scene.background), text.contrast))  # type: ignore[assignment]
    x, y, w, h = text.box
    if w <= 0 or h <= 0:
        return

    if not wrap:
        _draw_single_line(draw, text.text, (x, y, w, h), font, color, text.align)
        return

    lines = _wrap_lines(draw, text.text, font, w)
    line_height = _line_height(draw, font)
    total = line_height * len(lines)
    cursor_y = y + max(0, (h - total) // 2)
    for line in lines:
        if cursor_y + line_height > y + h + line_height:
            break
        _draw_single_line(draw, line, (x, cursor_y, w, line_height), font, color, text.align)
        cursor_y += line_height


def _draw_single_line(draw: Any, line: str, box: Box, font: Any, color: Tuple[int, int, int], align: str) -> None:
    x, y, w, h = box
    if align == "center":
        text_w = draw.textlength(line, font=font)
        draw.text((x + max(0, (w - text_w) / 2), y), line, font=font, fill=color)
    elif align == "right":
        text_w = draw.textlength(line, font=font)
        draw.text((x + max(0, w - text_w), y), line, font=font, fill=color)
    else:
        draw.text((x, y), line, font=font, fill=color)


def _line_height(draw: Any, font: Any) -> int:
    try:
        metrics = font.getmetrics()
        return int(metrics[0] + metrics[1]) + 2
    except Exception:  # pragma: no cover
        return 20


def _wrap_lines(draw: Any, text: str, font: Any, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _inset(box: Box, px: int) -> Box:
    x, y, w, h = box
    return (x + px, y + px, max(1, w - 2 * px), max(1, h - 2 * px))


def _option_text_box(option: SceneOption) -> Box:
    from .scenes import _option_text_box as _impl

    return _impl(option)


def _fallback_question_region(scene: Scene) -> Box:
    from .scenes import _default_question_region

    return _default_question_region(scene)


def _tint(color: Tuple[int, int, int], accent: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    return tuple(int(round(c * (1 - amount) + a * amount)) for c, a in zip(color, accent))  # type: ignore[return-value]


def upscale(pixels: np.ndarray, factor: float, interpolation: str = "bicubic") -> np.ndarray:
    """FR-7.1.6 zoom-crop upscale.  OpenCV when available, Pillow otherwise."""
    if abs(factor - 1.0) < 1e-6:
        return pixels
    try:
        import cv2

        mode = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }.get(interpolation, cv2.INTER_CUBIC)
        return cv2.resize(pixels, None, fx=factor, fy=factor, interpolation=mode)
    except ImportError:
        from PIL import Image

        resample = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }.get(interpolation, Image.Resampling.BICUBIC)
        height, width = pixels.shape[:2]
        image = Image.fromarray(pixels)
        return np.asarray(image.resize((int(width * factor), int(height * factor)), resample), dtype=np.uint8)


def crop(pixels: np.ndarray, box: Box, pad: int = 0) -> np.ndarray:
    """Crop ``box`` out of a frame, padded and clipped to the frame bounds."""
    height, width = pixels.shape[:2]
    x, y, w, h = box
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(width, x + w + pad)
    bottom = min(height, y + h + pad)
    if right <= left or bottom <= top:
        return np.zeros((0, 0, 3), dtype=pixels.dtype)
    return pixels[top:bottom, left:right]


def encode_png_b64(pixels: np.ndarray) -> Optional[str]:
    """Base64 PNG of a pixel array -- the only image payload models ever see."""
    if pixels is None:
        return None
    array = np.asarray(pixels)
    if array.size == 0:
        return None
    try:
        from .models.openai_compat import encode_image

        return encode_image(array)
    except Exception:
        return None


def crop_b64(frame: Any, region: Optional[Box], *, pad: int = 4) -> Optional[str]:
    """Crop ``region`` out of ``frame`` and encode it (FR-16.1: crops only)."""
    if frame is None or getattr(frame, "pixels", None) is None or region is None:
        return None
    pixels = crop(np.asarray(frame.pixels), region, pad=pad)
    return encode_png_b64(pixels) if pixels.size else None
