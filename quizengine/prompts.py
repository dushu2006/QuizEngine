"""Prompt and response-schema builders for every model-facing call.

Kept in one place so the *contract* with a model is reviewable as a unit:
what we ask for, what schema we enforce, and what we are allowed to send
(FR-16.1: crops only, never full-screen dumps).

Every builder returns a :class:`~quizengine.contracts.ModelRequest`, so no module
needs to know which provider -- or which vendor -- answers it (FR-6.2).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from .contracts import ModelMessage, ModelRequest, OptionPerception, PerceptionResult, Question
from .geometry import Box, box_union_all

SYSTEM_IDENTITY = (
    "You are the reasoning component of QuizEngine, a vision-driven agent that answers single-choice "
    "questions shown on a screen in an authorized test environment. You answer only from the content "
    "given to you. You never attempt to bypass, evade or disable any security, proctoring or "
    "human-verification mechanism, and you never interact with CAPTCHAs. When you are not sure, say so "
    "with a low confidence value instead of guessing."
)

STRUCTURED_OUTPUT_RULE = (
    "Respond with ONLY a JSON object that matches the given schema. No prose, no markdown fences."
)

# --------------------------------------------------------------------------- #
# Tier-2 perception (FR-7.2.5 / FR-7.2.6)
# --------------------------------------------------------------------------- #
PERCEPTION_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "QuizEngineTier2Perception",
    "type": "object",
    "additionalProperties": False,
    "required": ["layout_type", "question_text", "options", "navigation", "overlays", "confidence"],
    "properties": {
        "layout_type": {
            "type": "string",
            "enum": [
                "vertical_options",
                "horizontal_options",
                "card_grid",
                "tile",
                "text_only_buttons",
                "unknown",
            ],
        },
        "question_region": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x, y, w, h] in the coordinates of the supplied crop",
        },
        "question_text": {"type": "string", "description": "verbatim transcription of the question"},
        "verbatim_element_indices": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
            "description": "indices of the Tier-1 text elements that make up the question text",
        },
        "options": {
            "type": "array",
            "minItems": 0,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "handle", "text", "hit_box", "selected_marker"],
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "handle": {"type": "string"},
                    "text": {"type": "string"},
                    "hit_box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "text_box": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "text_conf": {"type": "number", "minimum": 0, "maximum": 1},
                    "selected_marker": {"type": "string", "enum": ["none", "dot", "check", "highlight"]},
                },
            },
        },
        "navigation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["next_btn", "prev_btn", "progress_text"],
            "properties": {
                "next_btn": {"$ref": "#/definitions/button"},
                "prev_btn": {"$ref": "#/definitions/button"},
                "submit_btn": {"$ref": "#/definitions/button"},
                "progress_text": {"type": ["string", "null"]},
                "progress_current": {"type": ["integer", "null"]},
                "progress_total": {"type": ["integer", "null"]},
            },
        },
        "overlays": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["handle", "box", "kind"],
                "properties": {
                    "handle": {"type": "string"},
                    "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "text": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["toast", "modal", "loading", "cookie_banner", "captcha", "human_verification", "unknown"],
                    },
                    "dismissible": {"type": ["boolean", "null"]},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
    "definitions": {
        "button": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["handle", "box"],
            "properties": {
                "handle": {"type": "string"},
                "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "text": {"type": "string"},
                "enabled": {"type": "boolean"},
            },
        }
    },
}

PERCEPTION_SYSTEM_PROMPT = (
    SYSTEM_IDENTITY
    + "\n\nYour task in this call is STRUCTURAL PERCEPTION of a screenshot crop of a quiz interface. "
    "Report what is on screen: the layout type, the question text verbatim, every answer option with a "
    "pixel-accurate clickable bounding box (the whole row or tile, not just its text), whether each "
    "option shows a selected-state marker (radio dot, checkbox tick, or highlight), the navigation "
    "affordances, and any popup/toast/overlay. "
    "If you see a CAPTCHA or a human-verification challenge, report it as an overlay of that kind and "
    "do nothing else. "
    + STRUCTURED_OUTPUT_RULE
)


def tier1_hints(perception: PerceptionResult, *, max_options: int = 8) -> Dict[str, Any]:
    """Tier-1 proposals passed to Tier 2 as hints (FR-7.2.5)."""
    return {
        "layout_type": perception.layout_type.value,
        "question_region": list(perception.question_region) if perception.question_region else None,
        "question_text": perception.question_text,
        "options": [
            {
                "index": option.index,
                "handle": option.handle,
                "text": option.text,
                "hit_box": list(option.hit_box),
                "text_box": list(option.text_box) if option.text_box else None,
                "text_conf": round(option.text_conf, 3),
                "selected_marker": option.selected_marker.value,
            }
            for option in perception.options[:max_options]
        ],
        "navigation": {
            "next_btn": _button_hint(perception.navigation.next_btn),
            "prev_btn": _button_hint(perception.navigation.prev_btn),
            "submit_btn": _button_hint(perception.navigation.submit_btn),
            "progress_text": perception.navigation.progress_text,
        },
        "overlays": [
            {"handle": o.handle, "box": list(o.box), "text": o.text, "kind": o.kind.value}
            for o in perception.overlays
        ],
        "text_blocks": [
            {"text": b.text, "box": list(b.box), "confidence": round(b.confidence, 3)}
            for b in perception.text_blocks[:40]
        ],
        "tier1_confidence": round(perception.tier1_confidence, 3),
        "ambiguity": perception.reconciliation_flags,
    }


def _button_hint(button: Any) -> Optional[Dict[str, Any]]:
    if button is None:
        return None
    return {"handle": button.handle, "box": list(button.box), "text": button.text, "enabled": button.enabled}


def build_perception_request(
    *,
    tier1: PerceptionResult,
    crop_b64: Optional[str],
    crop_box: Optional[Box],
    ambiguity: Sequence[str],
    correlation_id: Optional[str] = None,
    timeout_s: float = 30.0,
    max_tokens: int = 1600,
) -> ModelRequest:
    payload: Dict[str, Any] = {
        "task": "structural_perception",
        "crop_box_in_frame": list(crop_box) if crop_box else None,
        "tier1_hints": tier1_hints(tier1),
        "why_tier2": list(ambiguity),
        "instructions": (
            "Coordinates must be expressed in the crop's pixel space. Confirm or correct the Tier-1 "
            "hints; do not invent options that are not visible."
        ),
    }
    messages = [
        ModelMessage(role="system", content=PERCEPTION_SYSTEM_PROMPT),
        ModelMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=False),
            images_b64=[crop_b64] if crop_b64 else [],
        ),
    ]
    return ModelRequest(
        task="perceive",
        messages=messages,
        response_schema=PERCEPTION_SCHEMA,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        correlation_id=correlation_id,
    )


def perception_crop_box(tier1: PerceptionResult, frame_size: Box, pad_ratio: float = 0.06) -> Box:
    """The minimal crop that contains everything Tier 2 must see (FR-16.1).

    Never the whole screen: only the union of the question region, the option hit
    areas, the navigation proposal and any overlay.
    """
    from .geometry import box_clip, box_pad_relative

    boxes: List[Box] = []
    if tier1.question_region:
        boxes.append(tier1.question_region)
    boxes.extend(option.hit_box for option in tier1.options)
    for button in (tier1.navigation.next_btn, tier1.navigation.prev_btn, tier1.navigation.submit_btn):
        if button is not None:
            boxes.append(button.box)
    boxes.extend(overlay.box for overlay in tier1.overlays)
    if not boxes:
        boxes.extend(block.box for block in tier1.text_blocks[:20])
    if not boxes:
        width, height = frame_size[2], frame_size[3]
        return (int(width * 0.05), int(height * 0.05), int(width * 0.9), int(height * 0.9))
    union = box_union_all(boxes) or frame_size
    padded = box_pad_relative(union, pad_ratio, pad_ratio)
    return box_clip(padded, frame_size[2], frame_size[3])


# --------------------------------------------------------------------------- #
# Solver (FR-7.4.1 step 3)
# --------------------------------------------------------------------------- #
SOLVER_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "QuizEngineDecision",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "confidence", "rationale"],
    "properties": {
        "answer": {
            "type": ["string", "null"],
            "description": "single letter A-Z naming the chosen option, or null when unanswerable",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 1200},
    },
}

SOLVER_SYSTEM_PROMPT = (
    SYSTEM_IDENTITY
    + "\n\nYour task in this call is to ANSWER one single-choice question. You are given the question "
    "text, the options labelled A..N, and any visual/mathematical context that was extracted from the "
    "screen. Reason step by step internally, then output the letter of the correct option. "
    "Report a calibrated confidence: 0.9+ only when you are certain, below 0.6 when you are guessing. "
    "If the question cannot be answered from the given content, return {\"answer\": null, "
    "\"confidence\": 0, \"rationale\": \"...\"}. "
    + STRUCTURED_OUTPUT_RULE
)


def build_solver_request(
    question: Question,
    *,
    context: Optional[Dict[str, Any]] = None,
    image_b64: Optional[str] = None,
    correlation_id: Optional[str] = None,
    temperature: float = 0.0,
    timeout_s: float = 30.0,
    max_tokens: int = 700,
) -> ModelRequest:
    letters = question.letters()
    payload: Dict[str, Any] = {
        "task": "answer_single_choice_question",
        "question": question.text,
        "options": [f"{letter}. {text}" for letter, text in zip(letters, question.option_texts())],
        "option_count": len(question.options),
        "type": question.type.value,
        "flags": question.flags.to_wire(),
        "context": context or {},
    }
    messages = [
        ModelMessage(role="system", content=SOLVER_SYSTEM_PROMPT),
        ModelMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=False),
            images_b64=[image_b64] if image_b64 else [],
        ),
    ]
    return ModelRequest(
        task="solve",
        messages=messages,
        response_schema=SOLVER_SCHEMA,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        correlation_id=correlation_id,
    )


# --------------------------------------------------------------------------- #
# Extraction escalations (FR-7.3.4 math, FR-7.3.5 images)
# --------------------------------------------------------------------------- #
MATH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["latex", "confidence"],
    "properties": {
        "latex": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
}


def build_math_request(
    *, ocr_text: str, image_b64: Optional[str], correlation_id: Optional[str] = None, timeout_s: float = 30.0
) -> ModelRequest:
    payload = {
        "task": "transcribe_mathematics",
        "ocr_text": ocr_text,
        "instructions": "Transcribe the mathematical expression exactly, in LaTeX.",
    }
    return ModelRequest(
        task="transcribe_math",
        messages=[
            ModelMessage(
                role="system",
                content=SYSTEM_IDENTITY + "\n\nTranscribe mathematics into LaTeX. " + STRUCTURED_OUTPUT_RULE,
            ),
            ModelMessage(
                role="user", content=json.dumps(payload, ensure_ascii=False), images_b64=[image_b64] if image_b64 else []
            ),
        ],
        response_schema=MATH_SCHEMA,
        temperature=0.0,
        max_tokens=400,
        timeout_s=timeout_s,
        correlation_id=correlation_id,
    )


IMAGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["description", "confidence"],
    "properties": {
        "description": {"type": "string"},
        "values": {"type": "object", "additionalProperties": True},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


def build_image_request(
    *, ocr_text: str, image_b64: Optional[str], correlation_id: Optional[str] = None, timeout_s: float = 30.0
) -> ModelRequest:
    payload = {
        "task": "describe_visual_content",
        "ocr_text": ocr_text,
        "instructions": (
            "Describe the image so that a text-only reasoner can answer a question about it. "
            "For charts and tables, extract the axis labels and every readable data value."
        ),
    }
    return ModelRequest(
        task="describe_image",
        messages=[
            ModelMessage(
                role="system",
                content=SYSTEM_IDENTITY + "\n\nDescribe visual content factually. " + STRUCTURED_OUTPUT_RULE,
            ),
            ModelMessage(
                role="user", content=json.dumps(payload, ensure_ascii=False), images_b64=[image_b64] if image_b64 else []
            ),
        ],
        response_schema=IMAGE_SCHEMA,
        temperature=0.0,
        max_tokens=700,
        timeout_s=timeout_s,
        correlation_id=correlation_id,
    )


POPUP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "dismissible", "confidence"],
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["toast", "modal", "loading", "cookie_banner", "captcha", "human_verification", "unknown"],
        },
        "dismissible": {"type": ["boolean", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
}


def build_popup_request(
    *, text: str, image_b64: Optional[str], correlation_id: Optional[str] = None, timeout_s: float = 20.0
) -> ModelRequest:
    payload = {
        "task": "classify_overlay",
        "text": text,
        "instructions": (
            "Classify this overlay. If it is a CAPTCHA or any human-verification challenge, report that "
            "kind; the agent will stop rather than interact with it."
        ),
    }
    return ModelRequest(
        task="classify_popup",
        messages=[
            ModelMessage(
                role="system",
                content=SYSTEM_IDENTITY + "\n\nClassify an on-screen overlay. " + STRUCTURED_OUTPUT_RULE,
            ),
            ModelMessage(
                role="user", content=json.dumps(payload, ensure_ascii=False), images_b64=[image_b64] if image_b64 else []
            ),
        ],
        response_schema=POPUP_SCHEMA,
        temperature=0.0,
        max_tokens=200,
        timeout_s=timeout_s,
        correlation_id=correlation_id,
    )


__all__ = [
    "IMAGE_SCHEMA",
    "MATH_SCHEMA",
    "PERCEPTION_SCHEMA",
    "POPUP_SCHEMA",
    "SOLVER_SCHEMA",
    "STRUCTURED_OUTPUT_RULE",
    "SYSTEM_IDENTITY",
    "build_image_request",
    "build_math_request",
    "build_perception_request",
    "build_popup_request",
    "build_solver_request",
    "perception_crop_box",
    "tier1_hints",
]
