"""Capture Module (PRD section 7.1).

Responsibility: produce a *validated, fresh, correctly-cropped* frame of the
target region -- and nothing else.  This module is the single legal source of
pixels in the system (**AC-7.1.2**).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import EngineConfig
from ..contracts import FailureCode, Frame, RunEventName, State
from ..failures import FailureSignal
from ..geometry import Box, box_clip
from ..render import crop, upscale
from .backends import build_backend
from .base import CaptureBackend, MonitorInfo
from .ring_buffer import FrameRingBuffer, frame_memory_mb
from .validation import FrameValidator, ValidityResult, frame_hash


class CaptureModule:
    """FR-7.1.1 -- FR-7.1.7."""

    def __init__(
        self,
        config: EngineConfig,
        backend: Optional[CaptureBackend] = None,
        *,
        telemetry: Any = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        perf: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.capture_config = config.capture
        self.telemetry = telemetry
        self._clock = clock
        self._sleep = sleep
        self._perf = perf
        self.backend = backend if backend is not None else build_backend(config.capture)
        self.validator = FrameValidator(config.capture, config.safety)
        self.ring = FrameRingBuffer(
            max_frames=config.capture.ring_buffer_frames,
            max_memory_mb=config.capture.ring_buffer_memory_mb,
        )
        self.monitors: List[MonitorInfo] = self.backend.monitors()
        self.monitor = self._select_monitor(config.capture.monitor)
        self._seq = 0
        self._last_hash: Optional[str] = None
        self._last_frame: Optional[Frame] = None
        self.stats: Dict[str, Any] = {
            "captures": 0,
            "retries": 0,
            "rejections": {},
            "stale_by_age": 0,
            "zoom_crops": 0,
            "region_captures": 0,
            "geometry_adaptations": 0,
            "distinct_geometries": 1,
        }
        self._geometry_adaptations = 0
        self._geometries_seen: set = set()

    # -- monitor geometry --------------------------------------------------- #
    def _select_monitor(self, index: int) -> MonitorInfo:
        if not self.monitors:
            raise FailureSignal(
                FailureCode.CAPTURE_FAILURE,
                "capture backend reported no monitors",
                origin_state=State.CAPTURING,
            )
        for monitor in self.monitors:
            if monitor.index == index:
                return monitor
        # L6 never assume: fall back to the primary/last display and say so.
        chosen = next((m for m in self.monitors if m.is_primary), self.monitors[-1])
        self._log(
            f"capture.monitor={index} not present; using {chosen.name} {chosen.size_px}",
            event=RunEventName.FRAME_REJECTED,
            code=FailureCode.CAPTURE_FAILURE,
        )
        return chosen

    @property
    def expected_size(self) -> Tuple[int, int]:
        """Expected frame size: the configured region, else the whole monitor."""
        if self.capture_config.region is not None:
            return (int(self.capture_config.region[2]), int(self.capture_config.region[3]))
        return self.monitor.size_px

    #: How many *distinct* display geometries one run may adapt to (bounded, L3).
    #: Flipping back and forth between two known geometries (monitor switch, DPI
    #: change, window resize) is not itself suspicious; an unbounded number of
    #: different sizes is.
    MAX_DISTINCT_GEOMETRIES = 6

    def _adapt_geometry(self, width: int, height: int, correlation_id: Optional[str] = None) -> bool:
        """Re-enumerate displays after a dimension rejection.

        Returns True when the backend now reports the size we actually grabbed,
        meaning the *display* changed rather than the capture being broken.
        """
        tolerance = int(self.capture_config.dimension_tolerance_px)
        previous = self.monitor.size_px
        geometry = (int(width), int(height))
        known = geometry in self._geometries_seen or previous == geometry
        if not known and len(self._geometries_seen) >= self.MAX_DISTINCT_GEOMETRIES:
            return False
        monitor = self.reselect_monitor()
        if abs(monitor.size_px[0] - width) > tolerance or abs(monitor.size_px[1] - height) > tolerance:
            return False
        self._geometries_seen.add(geometry)
        self._geometries_seen.add((int(monitor.size_px[0]), int(monitor.size_px[1])))
        self._geometry_adaptations += 1
        self.stats["geometry_adaptations"] = self._geometry_adaptations
        self.stats["distinct_geometries"] = len(self._geometries_seen)
        self._log(
            f"display geometry changed {previous[0]}x{previous[1]} -> {monitor.size_px[0]}x{monitor.size_px[1]}; "
            f"adapting (distinct geometries {len(self._geometries_seen)}/{self.MAX_DISTINCT_GEOMETRIES})",
            correlation_id=correlation_id,
            width=monitor.size_px[0],
            height=monitor.size_px[1],
        )
        return True

    def reselect_monitor(self, index: Optional[int] = None) -> MonitorInfo:
        """Re-enumerate displays (hot-plug / resolution change) -- L6."""
        self.monitors = self.backend.monitors()
        self.monitor = self._select_monitor(index if index is not None else self.capture_config.monitor)
        return self.monitor

    # -- capture ------------------------------------------------------------ #
    def capture(
        self,
        *,
        expect_change: bool = False,
        correlation_id: Optional[str] = None,
        region: Optional[Box] = None,
    ) -> Frame:
        """Grab and validate a frame; retries per FR-7.1.3, then signals failure."""
        retries = max(0, int(self.capture_config.retries))
        backoff_s = float(self.capture_config.retry_backoff_ms) / 1000.0
        last_result: Optional[ValidityResult] = None
        last_error: Optional[str] = None
        target_region = region if region is not None else self.capture_config.region
        expected = (int(target_region[2]), int(target_region[3])) if target_region is not None else self.expected_size

        for attempt in range(1, retries + 1):
            started = self._perf()
            pixels: Optional[np.ndarray] = None
            try:
                pixels = self.backend.grab(self.monitor.index, target_region)
            except Exception as exc:  # backend raised -> retry, never crash blind (L4)
                last_error = f"{type(exc).__name__}: {exc}"
                last_result = ValidityResult(valid=False, reason="null", detail=last_error)
            else:
                last_result = self.validator.validate(
                    pixels,
                    expected,
                    previous_hash=self._last_hash,
                    expect_change=expect_change,
                    text_layer=self.backend.text_layer_hint(),
                    display_state=self.backend.display_state(),
                )
            latency_ms = (self._perf() - started) * 1000.0

            if last_result is not None and last_result.valid and pixels is not None:
                frame = self._accept(pixels, last_result, latency_ms, correlation_id, target_region)
                return frame

            reason = (last_result.reason if last_result else "null") or "null"
            detail = (last_result.detail if last_result else last_error) or ""
            self._record_rejection(reason)
            self._emit_rejection(reason, detail, attempt, latency_ms, correlation_id)

            if reason == "dimensions" and pixels is not None:
                # L6: never assume the display geometry is frozen.  A monitor
                # hot-plug, a DPI change or a window resize shows up as a
                # dimension rejection; re-enumerate and adapt (bounded).
                height, width = pixels.shape[:2]
                if self._adapt_geometry(int(width), int(height), correlation_id):
                    last_result = self.validator.validate(
                        pixels,
                        self.expected_size,
                        previous_hash=self._last_hash,
                        expect_change=expect_change,
                        text_layer=self.backend.text_layer_hint(),
                        display_state=self.backend.display_state(),
                    )
                    if last_result.valid:
                        frame = self._accept(pixels, last_result, latency_ms, correlation_id, target_region)
                        return frame

            if reason == "lock_screen":
                # FR-7.1.2: a lock screen / secure desktop is an *environment*
                # event, not a capture retry.  Hand it to the gatekeeper path.
                code = (
                    FailureCode.RESTRICTED_ENVIRONMENT
                    if self.config.safety.halt_on_lock_screen
                    else FailureCode.CAPTURE_FAILURE
                )
                raise FailureSignal(
                    code,
                    f"capture rejected: {detail}",
                    origin_state=State.CAPTURING,
                    detail={"reason": reason, "attempt": attempt, "monitor": self.monitor.name},
                )

            if attempt <= retries - 1:
                self.stats["retries"] += 1
                if backoff_s > 0:
                    self._sleep(backoff_s)

        raise FailureSignal(
            FailureCode.CAPTURE_FAILURE,
            f"capture failed after {retries} attempt(s): "
            f"{(last_result.detail if last_result else last_error) or 'unknown reason'}",
            origin_state=State.CAPTURING,
            detail={
                "reason": (last_result.reason if last_result else "null"),
                "attempts": retries,
                "monitor": self.monitor.name,
                "expected_size": list(expected),
                "backend_error": last_error,
            },
        )

    def _accept(
        self,
        pixels: np.ndarray,
        result: ValidityResult,
        latency_ms: float,
        correlation_id: Optional[str],
        region: Optional[Box],
    ) -> Frame:
        self._seq += 1
        height, width = pixels.shape[:2]
        annotation = self.backend.annotation()
        frame = Frame(
            seq=self._seq,
            ts=self._clock(),
            monitor_id=self.monitor.index,
            dpi_scale=self.monitor.dpi_scale,
            size_px=(width, height),
            hash=result.hash or frame_hash(pixels),
            backend=self.backend.name,
            backend_meta={
                "std_dev": round(result.std_dev, 3),
                "region": list(region) if region else None,
                "monitor_name": self.monitor.name,
                "annotation_available": annotation is not None,
                # Fixture ground truth.  Present only for synthetic/replay
                # backends; the annotation OCR adapter is the sole consumer and
                # is refused for real capture backends (see perception.ocr).
                "fixture_annotation": annotation,
            },
            pixels=pixels,
        )
        self.ring.push(frame)
        self._last_hash = frame.hash
        self._last_frame = frame
        self.stats["captures"] += 1
        if self.telemetry is not None:
            self.telemetry.observe_latency("capture_latency_ms", latency_ms)
            self.telemetry.event(
                RunEventName.FRAME_CAPTURED,
                state=State.CAPTURING,
                latency_ms=latency_ms,
                module="capture",
                correlation_id=correlation_id,
                seq=frame.seq,
                size=f"{width}x{height}",
                hash=frame.hash[:16],
            )
        return frame

    def _record_rejection(self, reason: str) -> None:
        counters: Dict[str, int] = self.stats["rejections"]
        counters[reason] = counters.get(reason, 0) + 1
        metric = {
            "blank": "frames_rejected_blank",
            "stale": "frames_rejected_stale",
            "lock_screen": "frames_rejected_lock_screen",
        }.get(reason)
        if metric is not None and self.telemetry is not None:
            self.telemetry.metrics.inc(metric)
        if self.telemetry is not None:
            self.telemetry.metrics.inc("capture_failures")

    def _emit_rejection(
        self, reason: str, detail: str, attempt: int, latency_ms: float, correlation_id: Optional[str]
    ) -> None:
        self._log(
            f"frame rejected ({reason}) on attempt {attempt}: {detail}",
            event=RunEventName.FRAME_REJECTED,
            code=FailureCode.CAPTURE_FAILURE,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
            module="capture",
            reason=reason,
            attempt=attempt,
        )

    # -- freshness (FR-7.1.5) ---------------------------------------------- #
    def is_fresh(self, frame: Frame, now: Optional[float] = None) -> bool:
        return frame.is_fresh(self.capture_config.max_frame_age_ms, now if now is not None else self._clock())

    def require_fresh(
        self, frame: Frame, *, now: Optional[float] = None, correlation_id: Optional[str] = None
    ) -> Frame:
        """Discard an over-age frame and re-capture (FR-7.1.5).

        Perception may only consume fresh frames; anything older than
        ``capture.max_frame_age_ms`` is thrown away here, at the boundary.
        """
        current = now if now is not None else self._clock()
        if frame.is_fresh(self.capture_config.max_frame_age_ms, current):
            return frame
        age = frame.age_ms(current)
        self.stats["stale_by_age"] += 1
        self._log(
            f"frame {frame.seq} aged out ({age:.0f}ms > {self.capture_config.max_frame_age_ms:.0f}ms); re-capturing",
            event=RunEventName.FRAME_REJECTED,
            code=FailureCode.CAPTURE_FAILURE,
            module="capture",
            correlation_id=correlation_id,
            reason="age",
            age_ms=round(age, 1),
        )
        return self.capture(correlation_id=correlation_id)

    # -- crops (FR-7.1.6) --------------------------------------------------- #
    def capture_zoom_window(
        self,
        box: Box,
        *,
        scale: Optional[float] = None,
        correlation_id: Optional[str] = None,
        source_frame: Optional[Frame] = None,
        pad_px: int = 4,
    ) -> Frame:
        """Capture a zoomed crop of an element/question region (x2 bicubic default).

        Always takes a *fresh* capture (L1) unless an explicit ``source_frame``
        is supplied by a caller that just captured it in this same cycle.
        """
        factor = float(scale if scale is not None else self.capture_config.zoom_crop_scale)
        frame = source_frame if source_frame is not None else self.capture(correlation_id=correlation_id)
        if frame.pixels is None:
            raise FailureSignal(
                FailureCode.CAPTURE_FAILURE, "cannot zoom a frame without pixel data", origin_state=State.CAPTURING
            )
        clipped = box_clip(box, frame.width, frame.height)
        region_pixels = crop(np.asarray(frame.pixels), clipped, pad=pad_px)
        if region_pixels.size == 0:
            raise FailureSignal(
                FailureCode.CAPTURE_FAILURE,
                f"zoom window {box} is outside the frame {frame.size_px}",
                origin_state=State.CAPTURING,
                detail={"box": list(box), "frame": list(frame.size_px)},
            )
        zoomed = upscale(region_pixels, factor, self.capture_config.zoom_interpolation)
        self._seq += 1
        zoom_frame = Frame(
            seq=self._seq,
            ts=self._clock(),
            monitor_id=frame.monitor_id,
            dpi_scale=frame.dpi_scale * factor,
            size_px=(int(zoomed.shape[1]), int(zoomed.shape[0])),
            hash=frame_hash(zoomed),
            backend=self.backend.name,
            data_ref=frame.data_ref,
            backend_meta={
                "zoom_of": frame.seq,
                "source_box": list(clipped),
                "scale": factor,
                "interpolation": self.capture_config.zoom_interpolation,
                "purpose": "zoom_window",
            },
            pixels=zoomed,
        )
        self.ring.push(zoom_frame)
        self.stats["zoom_crops"] += 1
        if self.telemetry is not None:
            self.telemetry.event(
                RunEventName.FRAME_CAPTURED,
                state=State.CAPTURING,
                module="capture",
                correlation_id=correlation_id,
                seq=zoom_frame.seq,
                zoom_of=frame.seq,
                size=f"{zoom_frame.width}x{zoom_frame.height}",
            )
        return zoom_frame

    def capture_region(
        self, box: Box, *, expect_change: bool = False, correlation_id: Optional[str] = None
    ) -> Frame:
        """Targeted re-capture of a sub-region (FR-7.3.2 recovery step)."""
        frame = self.capture(expect_change=expect_change, correlation_id=correlation_id, region=None)
        if frame.pixels is None:
            return frame
        clipped = box_clip(box, frame.width, frame.height)
        region_pixels = crop(np.asarray(frame.pixels), clipped)
        if region_pixels.size == 0:
            return frame
        self._seq += 1
        region_frame = Frame(
            seq=self._seq,
            ts=frame.ts,
            monitor_id=frame.monitor_id,
            dpi_scale=frame.dpi_scale,
            size_px=(int(region_pixels.shape[1]), int(region_pixels.shape[0])),
            hash=frame_hash(region_pixels),
            backend=self.backend.name,
            backend_meta={"region_of": frame.seq, "source_box": list(clipped), "purpose": "region_recapture"},
            pixels=region_pixels,
        )
        self.ring.push(region_frame)
        self.stats["region_captures"] += 1
        return region_frame

    # -- access ------------------------------------------------------------- #
    @property
    def last_frame(self) -> Optional[Frame]:
        return self._last_frame

    @property
    def last_hash(self) -> Optional[str]:
        return self._last_hash

    def forensic_frames(self, count: int = 5) -> List[Frame]:
        """Last N frames for a failure artifact bundle (FR-7.11.3)."""
        return self.ring.last(count)

    def annotation(self) -> Optional[Dict[str, Any]]:
        """Ground-truth annotation, when the backend is a fixture backend."""
        return self.backend.annotation()

    def foreground_window(self) -> Optional[str]:
        return self.backend.foreground_window()

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.name,
            "monitor": {"name": self.monitor.name, "size": list(self.monitor.size_px), "dpi": self.monitor.dpi_scale},
            "expected_size": list(self.expected_size),
            "seq": self._seq,
            "ring": self.ring.stats,
            "stats": dict(self.stats),
            "validator_stats": dict(self.validator.stats),
        }

    def memory_mb(self) -> float:
        return sum(frame_memory_mb(f) for f in self.ring)

    # -- logging ------------------------------------------------------------ #
    def _log(self, message: str, **kwargs: Any) -> None:
        if self.telemetry is None:
            return
        event = kwargs.pop("event", RunEventName.FRAME_CAPTURED)
        module = kwargs.pop("module", "capture")
        code = kwargs.pop("code", None)
        latency_ms = kwargs.pop("latency_ms", 0.0)
        correlation_id = kwargs.pop("correlation_id", None)
        self.telemetry.event(
            event,
            state=State.CAPTURING,
            code=code,
            latency_ms=latency_ms,
            module=module,
            correlation_id=correlation_id,
            message=message,
            **kwargs,
        )


__all__ = ["CaptureModule"]
