"""Cross-tier reconciliation (FR-7.2.7).

Rules, verbatim from the PRD:

* Tier 1 OCR text wins for **content** -- characters are ground truth.
* Tier 2 wins for **spatial semantics** -- which box is an option, what is
  selected, where the question region is.
* Disagreement beyond tolerance -> confidence penalty (consumed by section 7.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import PerceptionConfig
from ..contracts import (
    LayoutType,
    NavButton,
    NavigationPerception,
    OptionPerception,
    Overlay,
    PerceptionResult,
    RegionKind,
    SelectedMarker,
)
from ..geometry import Box, as_box, box_iou
from .tier2 import Tier2Analysis

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", (text or "").lower()).strip()


@dataclass
class ReconciliationResult:
    perception: PerceptionResult
    flags: List[str] = field(default_factory=list)
    #: 0..1 -- how much the two tiers agreed (feeds FR-7.5.1).
    agreement: float = 1.0
    #: 0..1 penalty applied to composite confidence.
    penalty: float = 0.0
    matched_pairs: int = 0
    tier2_only_options: int = 0
    tier1_only_options: int = 0

    def describe(self) -> Dict[str, Any]:
        return {
            "agreement": round(self.agreement, 3),
            "penalty": round(self.penalty, 3),
            "flags": list(self.flags),
            "matched_pairs": self.matched_pairs,
            "tier2_only_options": self.tier2_only_options,
            "tier1_only_options": self.tier1_only_options,
            "option_count": len(self.perception.options),
            "layout_type": self.perception.layout_type.value,
        }


class Reconciler:
    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config

    # -- entry -------------------------------------------------------------- #
    def reconcile(
        self, tier1: PerceptionResult, tier2: Optional[Tier2Analysis]
    ) -> ReconciliationResult:
        if tier2 is None:
            return ReconciliationResult(
                perception=tier1,
                flags=["tier2_not_run"],
                agreement=max(0.35, tier1.tier1_confidence),
                penalty=0.0,
            )

        flags: List[str] = []
        if tier2.is_echo:
            flags.append(f"tier2_source={tier2.source}")

        options, option_stats = self._options(tier1, tier2, flags)
        layout = self._layout(tier1, tier2, flags)
        question_text, question_region = self._question(tier1, tier2, flags)
        navigation = self._navigation(tier1, tier2, flags)
        overlays = self._overlays(tier1, tier2, flags)

        mismatch_count = option_stats["mismatches"] + len([f for f in flags if f.startswith("disagree")])
        penalty = min(0.5, self.config.reconciliation_confidence_penalty * mismatch_count)
        if tier2.is_echo:
            penalty = min(0.5, penalty + self.config.reconciliation_confidence_penalty * 0.5)
        agreement = max(0.0, min(1.0, 1.0 - penalty))

        perception = tier1.model_copy(
            update={
                "layout_type": layout,
                "question_text": question_text,
                "question_region": question_region,
                "options": options,
                "navigation": navigation,
                "overlays": overlays,
                "tier2_used": True,
                "reconciliation_flags": flags,
                "tier2_confidence": tier2.confidence,
            }
        )
        return ReconciliationResult(
            perception=perception,
            flags=flags,
            agreement=agreement,
            penalty=penalty,
            matched_pairs=option_stats["matched"],
            tier2_only_options=option_stats["tier2_only"],
            tier1_only_options=option_stats["tier1_only"],
        )

    # -- options ------------------------------------------------------------ #
    def _options(
        self, tier1: PerceptionResult, tier2: Tier2Analysis, flags: List[str]
    ) -> Tuple[List[OptionPerception], Dict[str, int]]:
        tolerance = self.config.reconciliation_tolerance_iou
        stats = {"matched": 0, "tier2_only": 0, "tier1_only": 0, "mismatches": 0}
        used_tier2: set[int] = set()
        merged: List[OptionPerception] = []

        for first in tier1.options:
            best_index, best_score = None, 0.0
            for position, second in enumerate(tier2.options):
                if position in used_tier2:
                    continue
                second_box = as_box(second.hit_box)
                spatial = box_iou(first.hit_box, second_box)
                textual = 1.0 if normalize(first.text) == normalize(second.text) and first.text else 0.0
                score = max(spatial, 0.55 * spatial + 0.45 * textual, textual * 0.8)
                if score > best_score:
                    best_index, best_score = position, score
            if best_index is not None and best_score >= tolerance:
                second = tier2.options[best_index]
                used_tier2.add(best_index)
                stats["matched"] += 1
                if normalize(first.text) != normalize(second.text) and second.text.strip():
                    flags.append(
                        f"disagree:option_text[{first.handle}] tier1={first.text[:40]!r} tier2={second.text[:40]!r} (tier1 wins for content)"
                    )
                    stats["mismatches"] += 1
                if first.selected_marker != second.selected_marker:
                    flags.append(
                        f"disagree:selected_marker[{first.handle}] tier1={first.selected_marker.value} "
                        f"tier2={second.selected_marker.value} (tier2 wins for spatial semantics)"
                    )
                    stats["mismatches"] += 1
                merged.append(
                    OptionPerception(
                        index=first.index,
                        handle=first.handle,
                        text=first.text or second.text,  # Tier 1 wins for content
                        hit_box=as_box(second.hit_box),  # Tier 2 wins for space
                        text_box=as_box(second.text_box) if second.text_box else first.text_box,
                        text_conf=first.text_conf or second.text_conf,
                        selected_marker=second.selected_marker,  # Tier 2 wins for state
                    )
                )
            else:
                stats["tier1_only"] += 1
                merged.append(first)

        for position, second in enumerate(tier2.options):
            if position in used_tier2:
                continue
            stats["tier2_only"] += 1
            flags.append(f"option_only_in_tier2:{second.handle or second.text[:24]!r}")
            merged.append(
                OptionPerception(
                    index=len(merged),
                    handle=second.handle or f"opt_{len(merged)}",
                    text=second.text,
                    hit_box=as_box(second.hit_box),
                    text_box=as_box(second.text_box) if second.text_box else None,
                    text_conf=second.text_conf,
                    selected_marker=second.selected_marker,
                )
            )

        if len(tier1.options) != len(tier2.options):
            flags.append(
                f"disagree:option_count tier1={len(tier1.options)} tier2={len(tier2.options)}"
            )
            stats["mismatches"] += 1

        # Re-index in reading order and give every option a stable per-frame handle.
        merged.sort(key=lambda o: (o.hit_box[1], o.hit_box[0]))
        reindexed: List[OptionPerception] = []
        for index, option in enumerate(merged):
            reindexed.append(option.model_copy(update={"index": index, "handle": f"opt_{index}"}))
        return reindexed, stats

    # -- other axes --------------------------------------------------------- #
    def _layout(self, tier1: PerceptionResult, tier2: Tier2Analysis, flags: List[str]) -> LayoutType:
        if tier2.layout_type == LayoutType.UNKNOWN:
            return tier1.layout_type
        if tier1.layout_type != LayoutType.UNKNOWN and tier2.layout_type != tier1.layout_type:
            flags.append(f"disagree:layout tier1={tier1.layout_type.value} tier2={tier2.layout_type.value} (tier2 wins)")
        return tier2.layout_type

    def _question(
        self, tier1: PerceptionResult, tier2: Tier2Analysis, flags: List[str]
    ) -> Tuple[str, Optional[Box]]:
        text = tier1.question_text.strip()
        if not text:
            text = tier2.question_text.strip()
            if text:
                flags.append("question_text_recovered_by_tier2")
        elif normalize(text) != normalize(tier2.question_text) and tier2.question_text.strip():
            flags.append("disagree:question_text (tier1 wins for content)")
        region: Optional[Box] = as_box(tier2.question_region) if tier2.question_region else tier1.question_region
        return text, region

    def _navigation(
        self, tier1: PerceptionResult, tier2: Tier2Analysis, flags: List[str]
    ) -> NavigationPerception:
        def merge(first: Optional[NavButton], second: Any, handle: str) -> Optional[NavButton]:
            if second is not None and first is not None:
                return NavButton(
                    handle=first.handle or handle,
                    box=as_box(second.box),  # Tier 2 wins for space
                    text=first.text or second.text,  # Tier 1 wins for content
                    enabled=bool(second.enabled),
                )
            if second is not None:
                return NavButton(handle=second.handle or handle, box=as_box(second.box), text=second.text, enabled=bool(second.enabled))
            return first

        merged = NavigationPerception(
            next_btn=merge(tier1.navigation.next_btn, tier2.navigation.next_btn, "nav_next"),
            prev_btn=merge(tier1.navigation.prev_btn, tier2.navigation.prev_btn, "nav_prev"),
            submit_btn=merge(tier1.navigation.submit_btn, tier2.navigation.submit_btn, "nav_submit"),
            progress_text=tier1.navigation.progress_text or tier2.navigation.progress_text,
            progress_current=tier1.navigation.progress_current
            if tier1.navigation.progress_current is not None
            else tier2.navigation.progress_current,
            progress_total=tier1.navigation.progress_total
            if tier1.navigation.progress_total is not None
            else tier2.navigation.progress_total,
        )
        if (tier1.navigation.next_btn is None) != (tier2.navigation.next_btn is None):
            flags.append("disagree:next_button_presence")
        return merged

    def _overlays(self, tier1: PerceptionResult, tier2: Tier2Analysis, flags: List[str]) -> List[Overlay]:
        merged: List[Overlay] = list(tier1.overlays)
        for candidate in tier2.overlays:
            box = as_box(candidate.box)
            match = next((o for o in merged if box_iou(o.box, box) > 0.3), None)
            if match is None:
                merged.append(
                    Overlay(
                        handle=candidate.handle or f"overlay_{len(merged)}",
                        box=box,
                        text=candidate.text,
                        kind=candidate.kind,
                        dismissible=candidate.dismissible,
                    )
                )
                flags.append(f"overlay_only_in_tier2:{candidate.kind.value}")
                continue
            if match.kind.value == "unknown" and candidate.kind.value != "unknown":
                match.kind = candidate.kind
            if match.dismissible is None and candidate.dismissible is not None:
                match.dismissible = candidate.dismissible
            if candidate.text and len(candidate.text) > len(match.text):
                match.text = candidate.text
            match.box = box
        return merged


def region_of(perception: PerceptionResult, kind: RegionKind) -> Optional[Box]:
    for proposal in perception.regions:
        if proposal.kind == kind:
            return proposal.box
    return None


__all__ = ["Reconciler", "ReconciliationResult", "normalize", "region_of"]
