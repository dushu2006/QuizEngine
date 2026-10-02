"""Text preprocessing pipeline (FR-7.2.2) -- MANDATORY sequence, raw OCR forbidden.

Order (non-negotiable)::

    upscale x2 -> grayscale -> bilateral denoise
              -> adaptive threshold (auto-select Otsu vs Sauvola by contrast score)
              -> optional deskew

Every geometric step is accumulated into a single affine matrix so that bounding
boxes coming *out* of OCR can be mapped exactly back into frame coordinates
(``PreprocessResult.unmap_box``).  Without that, an upscale+deskew pipeline would
silently produce boxes that are right in the preprocessed image and wrong on
screen -- which would break click targeting (L1).

OpenCV is used when available; a pure NumPy/Pillow fallback keeps the pipeline
functional (degraded: no bilateral denoise, no deskew) on minimal installs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import PreprocessConfig
from ..geometry import Box, box_from_x2y2

try:  # pragma: no cover - import guard
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None


@dataclass
class PreprocessResult:
    """Preprocessed image + the geometry mapping back to the source frame."""

    image: np.ndarray
    matrix: np.ndarray  # 2x3 affine: frame coords -> preprocessed coords
    steps: List[str] = field(default_factory=list)
    contrast_score: float = 0.0
    threshold_mode: str = "none"
    skew_degrees: float = 0.0
    scale: float = 1.0
    latency_ms: float = 0.0
    engine: str = "opencv"
    source_size: Tuple[int, int] = (0, 0)

    @property
    def inverse(self) -> np.ndarray:
        return _invert_affine(self.matrix)

    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        m = self.matrix
        return (float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2]))

    def unmap_point(self, x: float, y: float) -> Tuple[float, float]:
        m = self.inverse
        return (float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2]))

    def unmap_box(self, box: Box) -> Box:
        """Map a box from preprocessed pixels back to frame pixels."""
        x, y, w, h = box
        left, top = self.unmap_point(x, y)
        right, bottom = self.unmap_point(x + w, y + h)
        return box_from_x2y2(int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))

    def map_box(self, box: Box) -> Box:
        x, y, w, h = box
        left, top = self.map_point(x, y)
        right, bottom = self.map_point(x + w, y + h)
        return box_from_x2y2(int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))

    def describe(self) -> Dict[str, Any]:
        return {
            "steps": list(self.steps),
            "contrast_score": round(self.contrast_score, 4),
            "threshold_mode": self.threshold_mode,
            "skew_degrees": round(self.skew_degrees, 3),
            "scale": round(self.scale, 3),
            "latency_ms": round(self.latency_ms, 2),
            "engine": self.engine,
            "size": [int(self.image.shape[1]), int(self.image.shape[0])],
        }


def has_opencv() -> bool:
    return _cv2 is not None


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
def preprocess(image: np.ndarray, config: PreprocessConfig) -> PreprocessResult:
    """Run the FR-7.2.2 sequence.  Returns the binarized image + geometry map."""
    started = time.perf_counter()
    source = np.asarray(image)
    if source.ndim == 3 and source.shape[2] == 4:
        source = source[:, :, :3]
    source_height, source_width = source.shape[:2]
    steps: List[str] = []
    scale = 1.0

    # -- 1. upscale -------------------------------------------------------- #
    factor = float(config.upscale_factor)
    if factor != 1.0:
        if _cv2 is not None:
            interp = _cv2.INTER_CUBIC
            source = _cv2.resize(source, None, fx=factor, fy=factor, interpolation=interp)
        else:
            from ..render import upscale as _pil_upscale

            source = _pil_upscale(source, factor, "bicubic")
        scale = factor
        steps.append(f"upscale_x{factor:g}")
    matrix = _scale_matrix(scale)

    # -- 2. grayscale ------------------------------------------------------ #
    gray = to_grayscale(source)
    if config.grayscale:
        steps.append("grayscale")

    # -- 3. bilateral denoise ---------------------------------------------- #
    denoised = gray
    if config.denoise:
        denoised, applied = _bilateral(gray, config.denoise_d, config.denoise_sigma)
        if applied:
            steps.append("bilateral_denoise")

    # -- 4. adaptive threshold (auto-select Otsu vs Sauvola) ---------------- #
    contrast = contrast_score(denoised)
    binary, mode = binarize(denoised, config, contrast)
    steps.append(f"threshold_{mode}")

    # -- 5. optional deskew ------------------------------------------------- #
    skew = 0.0
    if config.deskew:
        skew = estimate_skew(binary)
        if abs(skew) >= 0.4 and abs(skew) <= config.deskew_max_degrees:
            binary, rotate_matrix = _rotate(binary, skew)
            matrix = _compose_affine(rotate_matrix, matrix)
            steps.append(f"deskew_{skew:+.2f}deg")
        else:
            skew = 0.0

    latency = (time.perf_counter() - started) * 1000.0
    return PreprocessResult(
        image=binary,
        matrix=matrix,
        steps=steps,
        contrast_score=contrast,
        threshold_mode=mode,
        skew_degrees=skew,
        scale=scale,
        latency_ms=latency,
        engine="opencv" if _cv2 is not None else "numpy",
        source_size=(source_width, source_height),
    )


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if _cv2 is not None:
        return _cv2.cvtColor(image, _cv2.COLOR_RGB2GRAY)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.clip(image[:, :, :3].astype(np.float32) @ weights, 0, 255).astype(np.uint8)


def contrast_score(gray: np.ndarray) -> float:
    """Normalized (p95 - p5) / 255 -- the Otsu-vs-Sauvola selector signal.

    High contrast (crisp text on a flat background) -> global Otsu is enough.
    Low contrast (washed-out / OCR-hostile tier, FR-13.3) -> local Sauvola.
    """
    if gray.size == 0:
        return 0.0
    values = gray.astype(np.float32).ravel()
    if values.size > 200_000:  # subsample for speed on large frames
        values = values[:: max(1, values.size // 200_000)]
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    return max(0.0, min(1.0, (high - low) / 255.0))


def otsu_threshold(gray: np.ndarray) -> int:
    """Classic Otsu, implemented in NumPy so it works without OpenCV."""
    if gray.size == 0:
        return 127
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127
    weights = hist / total
    means = np.arange(256, dtype=np.float64)
    valid = weights > 0
    mean_total = float((weights * means).sum())
    best_threshold, best_variance = 127, -1.0
    weight_bg = 0.0
    sum_bg = 0.0
    for threshold in range(256):
        weight_bg += weights[threshold]
        if weight_bg == 0:
            continue
        weight_fg = 1.0 - weight_bg
        if weight_fg == 0:
            break
        sum_bg += weights[threshold] * means[threshold]
        mean_bg = sum_bg / weight_bg
        mean_fg = (mean_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance, best_threshold = variance, threshold
    del valid
    return int(best_threshold)


def sauvola_threshold(gray: np.ndarray, window: int = 15, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Sauvola local thresholding via integral images (no skimage dependency)."""
    image = gray.astype(np.float64)
    window = max(3, int(window) | 1)
    pad = window // 2
    integral = _integral(image)
    integral_sq = _integral(image * image)
    height, width = image.shape
    ys = np.clip(np.arange(height) - pad, 0, height - 1)
    ye = np.clip(np.arange(height) + pad, 0, height - 1)
    xs = np.clip(np.arange(width) - pad, 0, width - 1)
    xe = np.clip(np.arange(width) + pad, 0, width - 1)
    ys2, ye2 = ys[:, None], ye[:, None]
    xs2, xe2 = xs[None, :], xe[None, :]
    area = (ye2 - ys2 + 1) * (xe2 - xs2 + 1)
    total = _sum_window(integral, ys, ye, xs, xe)
    total_sq = _sum_window(integral_sq, ys, ye, xs, xe)
    mean = total / area
    variance = np.maximum(total_sq / area - mean * mean, 0.0)
    std = np.sqrt(variance)
    return mean * (1.0 + k * (std / r - 1.0))


def _integral(image: np.ndarray) -> np.ndarray:
    padded = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = image.cumsum(axis=0).cumsum(axis=1)
    return padded


def _sum_window(integral: np.ndarray, ys: np.ndarray, ye: np.ndarray, xs: np.ndarray, xe: np.ndarray) -> np.ndarray:
    ys2, ye2 = ys[:, None], ye[:, None]
    xs2, xe2 = xs[None, :], xe[None, :]
    return (
        integral[ye2 + 1, xe2 + 1]
        - integral[ys2, xe2 + 1]
        - integral[ye2 + 1, xs2]
        + integral[ys2, xs2]
    )


def binarize(gray: np.ndarray, config: PreprocessConfig, contrast: float) -> Tuple[np.ndarray, str]:
    """Threshold to a text-is-dark-on-light image (what OCR engines expect)."""
    if not config.adaptive_threshold:
        return gray, "none"

    auto = config.threshold_auto_select
    # Low contrast -> local (Sauvola).  High contrast -> global (Otsu) is faster
    # and more stable on flat backgrounds.
    use_sauvola = (contrast < 0.34) if auto else True
    mode = "sauvola" if use_sauvola else "otsu"

    if use_sauvola:
        thresholds = sauvola_threshold(gray, window=config.threshold_block_size, k=0.2 + config.threshold_c / 100.0)
        binary = (gray.astype(np.float64) > thresholds).astype(np.uint8) * 255
    else:
        if _cv2 is not None:
            value, binary = _cv2.threshold(gray, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
            threshold_value = int(value)
        else:
            threshold_value = otsu_threshold(gray)
            binary = (gray > threshold_value).astype(np.uint8) * 255

    # Normalize polarity: background should be white (255), ink black (0).
    if float((binary == 0).mean()) > 0.5:
        binary = 255 - binary
        mode += "+invert"
    return binary.astype(np.uint8), mode


def estimate_skew(binary: np.ndarray, max_degrees: float = 15.0) -> float:
    """Estimate text skew from the ink mask.  0.0 when it cannot be determined."""
    ink = binary < 128
    if not ink.any():
        return 0.0
    if _cv2 is not None:
        coords = np.column_stack(np.where(ink))
        if coords.shape[0] < 50:
            return 0.0
        try:
            rect = _cv2.minAreaRect(coords.astype(np.float32))
        except Exception:  # pragma: no cover
            return 0.0
        angle = float(rect[-1])
        if angle > 45:
            angle -= 90
        if angle < -45:
            angle += 90
        return angle if abs(angle) <= max_degrees else 0.0
    # NumPy fallback: projection-profile search over a small angle range.
    best_angle, best_score = 0.0, -1.0
    for tenth in range(-int(max_degrees * 10), int(max_degrees * 10) + 1):
        angle = tenth / 10.0
        rotated, _ = _rotate(binary, angle)
        profile = (rotated < 128).sum(axis=1).astype(np.float64)
        score = float(profile.var())
        if score > best_score:
            best_angle, best_score = angle, score
    return best_angle


def _rotate(image: np.ndarray, degrees: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate about the centre; returns the image and its 2x3 affine matrix."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    if _cv2 is not None:
        matrix = _cv2.getRotationMatrix2D(center, degrees, 1.0)
        rotated = _cv2.warpAffine(
            image, matrix, (width, height), flags=_cv2.INTER_LINEAR, borderMode=_cv2.BORDER_REPLICATE
        )
        return rotated, matrix.astype(np.float64)
    # Fallback: nearest-neighbour manual warp (small angles only).
    radians = np.deg2rad(-degrees)
    cos, sin = np.cos(radians), np.sin(radians)
    matrix = np.array(
        [
            [cos, -sin, center[0] - (cos * center[0] - sin * center[1])],
            [sin, cos, center[1] - (sin * center[0] + cos * center[1])],
        ],
        dtype=np.float64,
    )
    ys, xs = np.mgrid[0:height, 0:width]
    src_x = matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]
    src_y = matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]
    src_x = np.clip(np.round(src_x).astype(int), 0, width - 1)
    src_y = np.clip(np.round(src_y).astype(int), 0, height - 1)
    return image[src_y, src_x], matrix


def _bilateral(gray: np.ndarray, d: int, sigma: float) -> Tuple[np.ndarray, bool]:
    if _cv2 is not None:
        try:
            return _cv2.bilateralFilter(gray, d=int(d), sigmaColor=float(sigma), sigmaSpace=float(sigma)), True
        except Exception:  # pragma: no cover - unsupported depth
            return gray, False
    # NumPy fallback: 3x3 edge-preserving-ish smoothing.
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    out = np.zeros_like(gray, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
    return np.clip(out, 0, 255).astype(np.uint8), True


def _scale_matrix(scale: float) -> np.ndarray:
    return np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]], dtype=np.float64)


def _compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affines (``outer`` applied after ``inner``) as 2x3."""
    a = np.vstack([np.asarray(outer, dtype=np.float64), [0.0, 0.0, 1.0]])
    b = np.vstack([np.asarray(inner, dtype=np.float64), [0.0, 0.0, 1.0]])
    return (a @ b)[:2]


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:, :2]
    translation = matrix[:, 2:]
    inverse_linear = np.linalg.inv(linear)
    return np.hstack([inverse_linear, -inverse_linear @ translation])


def prepare_for_ocr(image: np.ndarray, config: PreprocessConfig, *, allow_raw: bool = False) -> PreprocessResult:
    """The only sanctioned way to hand pixels to an OCR engine.

    ``allow_raw`` exists for fixture replay where frames are already clean
    renders; :func:`quizengine.config._post_validate` refuses it for real OCR
    engines because FR-7.2.2 forbids raw input to OCR.
    """
    if allow_raw:
        gray = to_grayscale(np.asarray(image))
        return PreprocessResult(
            image=gray,
            matrix=_scale_matrix(1.0),
            steps=["raw(passthrough)"],
            contrast_score=contrast_score(gray),
            threshold_mode="none",
            engine="raw",
            source_size=(gray.shape[1], gray.shape[0]),
        )
    return preprocess(image, config)


__all__ = [
    "PreprocessResult",
    "preprocess",
    "prepare_for_ocr",
    "to_grayscale",
    "contrast_score",
    "otsu_threshold",
    "sauvola_threshold",
    "binarize",
    "estimate_skew",
    "has_opencv",
]
