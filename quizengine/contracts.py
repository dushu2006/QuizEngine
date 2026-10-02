"""Data contracts (PRD section 8) -- the *only* way modules talk to each other.

Design law **[L5] modularity by interface**: capture, perception, extraction,
solving, confidence, action and orchestration exchange nothing but the Pydantic
models below.  No module may import another module's internals.

Schema evolution rule (section 8): **additive-only within a major version**.
Every wire model carries ``schema_version`` pinned to a :class:`~typing.Literal`
so that a producer emitting an unknown major version fails validation loudly
instead of silently mis-parsing.  Fields added by this implementation beyond the
PRD's JSON sketches are marked ``# ADDITIVE`` and always have defaults, so a
document-shaped payload still validates.

Interpretation note: the PRD sketch for ``Frame`` reads
``"hash": "sha1": "<ref>"``.  That is modelled as ``hash = "sha1:<hex>"`` (the
content digest of the pixel buffer) plus a separate ``data_ref`` pointing at a
persisted artifact.  Pixel data itself lives in ``Frame.pixels`` which is
excluded from every serialization -- frames are metadata on the wire, memory in
process.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .geometry import Box, as_box

SCHEMA_MAJOR_VERSION = 1
SchemaVersion = Literal["v1"]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class LayoutType(str, Enum):
    """Section 8 ``layout_type``."""

    VERTICAL_OPTIONS = "vertical_options"
    HORIZONTAL_OPTIONS = "horizontal_options"
    CARD_GRID = "card_grid"
    TILE = "tile"
    TEXT_ONLY_BUTTONS = "text_only_buttons"  # ADDITIVE (section 3.1 layout list)
    UNKNOWN = "unknown"


class SelectedMarker(str, Enum):
    NONE = "none"
    DOT = "dot"
    CHECK = "check"
    HIGHLIGHT = "highlight"


class QuestionType(str, Enum):
    SINGLE_CHOICE = "single_choice"
    CHECKBOX_TILE = "checkbox_tile"
    # Reserved slots (section 19): parsed but refused by the v1.0 pipeline.
    MULTI_SELECT = "multi_select"
    FILL_IN_BLANK = "fill_in_blank"
    DRAG_AND_DROP = "drag_and_drop"
    UNKNOWN = "unknown"


#: Question types the v1.0 pipeline can actually answer (section 3.1/3.2).
SUPPORTED_QUESTION_TYPES = frozenset({QuestionType.SINGLE_CHOICE, QuestionType.CHECKBOX_TILE})


class RegionKind(str, Enum):
    """Tier-1 region proposals (FR-7.2.3)."""

    QUESTION = "question"
    OPTIONS = "options"
    NAVIGATION = "navigation"
    HEADER = "header"
    NOISE = "noise"


class ScreenTransition(str, Enum):
    """FR-7.3.3 transition classification."""

    SAME_QUESTION = "same_question"
    NEW_QUESTION = "new_question"
    PARTIAL_SCROLL = "partial_scroll"
    POPUP = "popup"
    END_STATE = "end_state"  # ADDITIVE (FR-7.8.4 end-state detector output)
    UNKNOWN = "unknown"


class SolverStrategy(str, Enum):
    """FR-7.4.1 strategy chain, in cascade order."""

    LOCAL_EXACT = "local_exact"
    LOCAL_FUZZY = "local_fuzzy"
    LOCAL_RULES = "local_rules"
    LLM_REASONING = "llm_reasoning"
    VLM_REANALYSIS = "vlm_reanalysis"
    SELF_CONSISTENCY = "self_consistency"  # ADDITIVE (FR-7.4.3)
    HUMAN = "human"  # ADDITIVE (UC-8 operator answer)
    NONE = "none"


class ActionType(str, Enum):
    CLICK = "click"
    MOVE = "move"
    SCROLL = "scroll"
    KEY = "key"
    TYPE = "type"


#: Actions that mutate the world and therefore MUST be verified (L2, AC-14.3).
STATE_CHANGING_ACTIONS = frozenset({ActionType.CLICK, ActionType.SCROLL, ActionType.KEY, ActionType.TYPE})


class ExpectedEffectType(str, Enum):
    """Intent ledger vocabulary (FR-7.9.3)."""

    SELECTION_CHANGED = "selection_changed"
    NAVIGATION = "navigation"
    CONTENT_SHIFT = "content_shift"
    DISMISS_OVERLAY = "dismiss_overlay"
    NONE = "none"


class UncertaintyPolicy(str, Enum):
    """FR-7.5.2 mid/low-confidence behaviour."""

    ACT = "act"
    VERIFY_AGAIN = "verify_again"
    PAUSE_HUMAN = "pause_human"


class State(str, Enum):
    """Finite state machine states (section 10)."""

    IDLE = "IDLE"
    GATE_CHECK = "GATE_CHECK"
    CAPTURING = "CAPTURING"
    PERCEIVING = "PERCEIVING"
    EXTRACTING = "EXTRACTING"
    DECIDING = "DECIDING"
    PRE_ACTION_RESOLVE = "PRE_ACTION_RESOLVE"
    ACTING = "ACTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    NAVIGATING = "NAVIGATING"
    END_DETECTED = "END_DETECTED"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    REPORTING = "REPORTING"
    DONE = "DONE"
    RUN_HALTED = "RUN_HALTED"
    RUN_ABORTED = "RUN_ABORTED"
    FAILED_SAFE = "FAILED_SAFE"


TERMINAL_STATES = frozenset({State.DONE, State.FAILED_SAFE})
FAILURE_STATES = frozenset({State.RUN_HALTED, State.RUN_ABORTED, State.FAILED_SAFE})


class RunEventName(str, Enum):
    """Telemetry event vocabulary (FR-7.10.4, FR-7.13.1)."""

    RUN_STARTED = "RUN_STARTED"
    RUN_COMPLETE = "RUN_COMPLETE"
    RUN_ABORTED = "RUN_ABORTED"
    RUN_HALTED = "RUN_HALTED"
    STATE_TRANSITION = "STATE_TRANSITION"
    ILLEGAL_TRANSITION = "ILLEGAL_TRANSITION"
    FRAME_CAPTURED = "FRAME_CAPTURED"
    FRAME_REJECTED = "FRAME_REJECTED"
    PERCEPTION_DONE = "PERCEPTION_DONE"
    PERCEPTION_LOW_CONFIDENCE = "PERCEPTION_LOW_CONFIDENCE"
    QUESTION_EXTRACTED = "QUESTION_EXTRACTED"
    EXTRACTION_FAILURE = "EXTRACTION_FAILURE"
    DECISION_MADE = "DECISION_MADE"
    SOLVER_STRATEGY_FAILED = "SOLVER_STRATEGY_FAILED"  # ADDITIVE: FR-7.13.4 attempt trace
    INTENT_DECLARED = "INTENT_DECLARED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    ACTION_VERIFIED = "ACTION_VERIFIED"
    ACTION_UNVERIFIED = "ACTION_UNVERIFIED"
    NAVIGATION_SUCCESS = "NAVIGATION_SUCCESS"
    NAVIGATION_STUCK = "NAVIGATION_STUCK"
    END_STATE_DETECTED = "END_STATE_DETECTED"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    RECOVERY_SUCCESS = "RECOVERY_SUCCESS"
    RECOVERY_EXHAUSTED = "RECOVERY_EXHAUSTED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    HUMAN_RESPONSE = "HUMAN_RESPONSE"
    GATE_PASSED = "GATE_PASSED"
    UNSUPPORTED_ENVIRONMENT = "UNSUPPORTED_ENVIRONMENT"
    FOCUS_LOST = "FOCUS_LOST"
    MODEL_CALL = "MODEL_CALL"
    SESSION_SAVED = "SESSION_SAVED"
    SESSION_RESUMED = "SESSION_RESUMED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class FailureCode(str, Enum):
    """Failure catalog (section 11)."""

    CAPTURE_FAILURE = "CAPTURE_FAILURE"
    EXTRACTION_FAILURE = "EXTRACTION_FAILURE"
    ACTION_UNVERIFIED = "ACTION_UNVERIFIED"
    NAVIGATION_STUCK = "NAVIGATION_STUCK"
    PERCEPTION_LOW_CONFIDENCE = "PERCEPTION_LOW_CONFIDENCE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_SCHEMA_FAILURE = "MODEL_SCHEMA_FAILURE"  # ADDITIVE (FR-7.2.6)
    FOCUS_LOST = "FOCUS_LOST"
    POPUP_UNKNOWN = "POPUP_UNKNOWN"
    RESTRICTED_ENVIRONMENT = "RESTRICTED_ENVIRONMENT"
    SOLVER_LOW_CONFIDENCE = "SOLVER_LOW_CONFIDENCE"
    SOLVER_TIMEOUT = "SOLVER_TIMEOUT"
    SOLVER_NO_ANSWER = "SOLVER_NO_ANSWER"  # ADDITIVE
    CONFIG_INVALID = "CONFIG_INVALID"  # ADDITIVE (NFR-17.4)
    ATTESTATION_MISSING = "ATTESTATION_MISSING"  # ADDITIVE (FR-7.14.1)
    STALE_COORDINATES = "STALE_COORDINATES"  # ADDITIVE (L1 / AC-14.2)
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"  # ADDITIVE (FR-7.10.2)
    UNSUPPORTED_QUESTION_TYPE = "UNSUPPORTED_QUESTION_TYPE"  # ADDITIVE (section 3.2)
    ILLEGAL_TRANSITION = "ILLEGAL_TRANSITION"  # ADDITIVE (FR-7.10.1)
    OPERATOR_STOP = "OPERATOR_STOP"  # ADDITIVE (FR-7.10.3)
    TIER2_FAILURE = "TIER2_FAILURE"
    MALFORMED_MODEL_OUTPUT = "MALFORMED_MODEL_OUTPUT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    CLICK_NOT_REGISTERED = "CLICK_NOT_REGISTERED"
    TRANSITION_NOT_DETECTED = "TRANSITION_NOT_DETECTED"
    POPUP_DETECTED = "POPUP_DETECTED"
    NETWORK_ERROR = "NETWORK_ERROR"
    UNSUPPORTED_ENVIRONMENT = "UNSUPPORTED_ENVIRONMENT"
    CANCELLED = "CANCELLED"


class RunOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED_SAFE = "failed_safe"
    ABORTED = "aborted"
    HALTED = "halted"
    STOPPED_BY_OPERATOR = "stopped_by_operator"


class OverlayKind(str, Enum):
    TOAST = "toast"
    MODAL = "modal"
    LOADING = "loading"
    COOKIE_BANNER = "cookie_banner"
    CAPTCHA = "captcha"  # restricted-environment indicator (section 3.3.2)
    HUMAN_VERIFICATION = "human_verification"
    UNKNOWN = "unknown"


#: Overlays that make the environment restricted rather than merely noisy.
RESTRICTED_OVERLAY_KINDS = frozenset({OverlayKind.CAPTCHA, OverlayKind.HUMAN_VERIFICATION})


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #
class SchemaModel(BaseModel):
    """Base for every wire contract: strict, versioned, JSON-safe."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        ser_json_timedelta="float",
    )

    schema_version: SchemaVersion = "v1"

    def to_wire(self) -> Dict[str, Any]:
        """JSON-safe dump (enums -> values), excluding in-memory pixel payloads."""
        return self.model_dump(mode="json", by_alias=False)


# --------------------------------------------------------------------------- #
# Capture (section 7.1)
# --------------------------------------------------------------------------- #
class Frame(SchemaModel):
    """A validated capture of the target region.

    ``pixels`` holds the raw ``H x W x 3`` uint8 array in memory only; it is
    excluded from serialization so a ``Frame`` can be logged, hashed and stored
    in a session file without dragging image data along.
    """

    seq: int = Field(..., ge=0)
    ts: float = Field(..., description="capture time, unix seconds")
    monitor_id: int = 0
    dpi_scale: float = Field(1.0, gt=0)
    size_px: Tuple[int, int]
    hash: str = Field(..., description="content digest, formatted 'sha1:<hex>'")
    data_ref: Optional[str] = Field(None, description="path of a persisted frame artifact")
    backend: str = Field("synthetic", description="capture backend id (FR-7.1.1)")
    backend_meta: Dict[str, Any] = Field(default_factory=dict)  # ADDITIVE
    pixels: Any = Field(default=None, exclude=True)

    @field_validator("size_px", mode="before")
    @classmethod
    def _coerce_size(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        return value

    @field_validator("hash")
    @classmethod
    def _check_hash(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError("hash must be formatted '<algo>:<digest>' e.g. 'sha1:abc123'")
        return value

    # -- convenience ------------------------------------------------------- #
    @property
    def width(self) -> int:
        return int(self.size_px[0])

    @property
    def height(self) -> int:
        return int(self.size_px[1])

    @property
    def full_box(self) -> Box:
        return (0, 0, self.width, self.height)

    def age_ms(self, now: float | None = None) -> float:
        """Age of the frame in milliseconds (FR-7.1.5 freshness contract)."""
        return max(0.0, ((now if now is not None else time.time()) - self.ts) * 1000.0)

    def is_fresh(self, max_frame_age_ms: float, now: float | None = None) -> bool:
        return self.age_ms(now) <= max_frame_age_ms

    def has_pixels(self) -> bool:
        return self.pixels is not None


class CaptureFailureDetail(SchemaModel):
    """Why a frame was rejected (FR-7.1.2)."""

    reason: Literal["null", "dimensions", "blank", "stale", "lock_screen", "backend_error"]
    detail: str = ""
    attempts: int = 1


# --------------------------------------------------------------------------- #
# Perception (section 7.2)
# --------------------------------------------------------------------------- #
class TextBlock(SchemaModel):
    """One OCR line with its bounding box (FR-7.2.1)."""

    text: str
    box: Box
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str = "ocr"  # ADDITIVE: which engine produced it

    @field_validator("box", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        return as_box(value) if isinstance(value, (list, tuple)) else value


class RegionProposal(SchemaModel):
    """Tier-1 region segmentation proposal (FR-7.2.3).

    Explicitly a *proposal generator*: Tier 2 confirms or overrides it.
    """

    kind: RegionKind
    box: Box
    score: float = Field(0.5, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)

    @field_validator("box", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        return as_box(value) if isinstance(value, (list, tuple)) else value


class OptionPerception(SchemaModel):
    """A perceived answer option.

    ``hit_box`` is the *clickable* area (full row/tile), ``text_box`` is the text
    inside it (FR-7.2.4).  ``handle`` is the semantic element id that actions
    must reference (FR-7.7.1).
    """

    index: int = Field(..., ge=0)
    handle: str
    text: str = ""
    hit_box: Box
    text_box: Optional[Box] = None
    text_conf: float = Field(0.0, ge=0.0, le=1.0)
    selected_marker: SelectedMarker = SelectedMarker.NONE

    @field_validator("hit_box", "text_box", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value


class NavButton(SchemaModel):
    handle: str
    box: Box
    text: str = ""
    enabled: bool = True  # ADDITIVE (greyed-out Next buttons are common)

    @field_validator("box", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        return as_box(value) if isinstance(value, (list, tuple)) else value


class NavigationPerception(SchemaModel):
    next_btn: Optional[NavButton] = None
    prev_btn: Optional[NavButton] = None
    submit_btn: Optional[NavButton] = None  # ADDITIVE
    progress_text: Optional[str] = None
    progress_current: Optional[int] = None  # ADDITIVE (FR-7.8.4 end-state evidence)
    progress_total: Optional[int] = None  # ADDITIVE


class Overlay(SchemaModel):
    """A popup / toast / modal detected on top of the quiz (FR-7.2.5)."""

    handle: str
    box: Box
    text: str = ""
    kind: OverlayKind = OverlayKind.UNKNOWN
    dismissible: Optional[bool] = None  # ADDITIVE: None = unknown (POPUP_UNKNOWN)
    close_btn: Optional[NavButton] = None  # ADDITIVE: verified-dismissal target only

    @field_validator("box", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        return as_box(value) if isinstance(value, (list, tuple)) else value


class PerceptionResult(SchemaModel):
    """Structured description of one frame (section 8)."""

    frame_seq: int = Field(..., ge=0)
    layout_type: LayoutType = LayoutType.UNKNOWN
    question_region: Optional[Box] = None
    question_text: str = ""
    options: List[OptionPerception] = Field(default_factory=list)
    navigation: NavigationPerception = Field(default_factory=NavigationPerception)
    overlays: List[Overlay] = Field(default_factory=list)
    tier2_used: bool = False
    reconciliation_flags: List[str] = Field(default_factory=list)
    # ADDITIVE -- Tier-1 evidence retained for extraction, telemetry and tests.
    text_blocks: List[TextBlock] = Field(default_factory=list)
    regions: List[RegionProposal] = Field(default_factory=list)
    ocr_engine: str = "none"
    latency_ms: float = 0.0
    tier1_confidence: float = Field(0.0, ge=0.0, le=1.0)
    tier2_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    end_state_evidence: List[str] = Field(default_factory=list)

    @field_validator("question_region", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value

    def option_by_handle(self, handle: str) -> Optional[OptionPerception]:
        for option in self.options:
            if option.handle == handle:
                return option
        return None

    def option_by_index(self, index: int) -> Optional[OptionPerception]:
        for option in self.options:
            if option.index == index:
                return option
        return None

    def selected_options(self) -> List[OptionPerception]:
        return [o for o in self.options if o.selected_marker != SelectedMarker.NONE]

    def has_question_like_region(self) -> bool:
        return bool(self.question_text.strip()) and len(self.options) >= 2


# --------------------------------------------------------------------------- #
# Extraction (section 7.3)
# --------------------------------------------------------------------------- #
class ContentFlags(SchemaModel):
    has_math: bool = False
    has_image: bool = False
    has_table: bool = False
    has_chart: bool = False  # ADDITIVE (section 3.1 content list)
    low_contrast: bool = False  # ADDITIVE (FR-13.3 OCR-hostile tier)


class QuestionOption(SchemaModel):
    index: int = Field(..., ge=0)
    text: str
    # ADDITIVE: carried through from perception so answer->box binding stays in
    # one place (FR-7.4.2).  Document-shaped payloads omit them and still parse.
    handle: Optional[str] = None
    hit_box: Optional[Box] = None
    text_conf: float = Field(0.0, ge=0.0, le=1.0)
    selected_marker: SelectedMarker = SelectedMarker.NONE

    @field_validator("hit_box", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value


class Question(SchemaModel):
    """Validated output of section 7.3."""

    hash: str = Field(..., description="stable id: sha1 of normalized text + ordinal")
    ordinal: int = Field(..., ge=0)
    text: str
    type: QuestionType = QuestionType.SINGLE_CHOICE
    options: List[QuestionOption] = Field(..., min_length=2)
    flags: ContentFlags = Field(default_factory=ContentFlags)
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    # ADDITIVE
    content_hash: str = Field("", description="sha1 of normalized text only")
    frame_seq: int = -1
    question_region: Optional[Box] = None
    layout_type: LayoutType = LayoutType.UNKNOWN
    source: Literal["tier1", "tier2", "reconciled", "fixture"] = "reconciled"

    @field_validator("question_region", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value

    def option_texts(self) -> List[str]:
        return [o.text for o in self.options]

    def letters(self) -> List[str]:
        """A, B, C, ... labels used in solver prompts (FR-7.4.1.3)."""
        return [chr(ord("A") + i) for i in range(len(self.options))]

    def option_index_for_letter(self, letter: str) -> Optional[int]:
        letter = letter.strip().upper()
        if len(letter) == 1 and "A" <= letter <= "Z":
            index = ord(letter) - ord("A")
            return index if index < len(self.options) else None
        return None


class TransitionAssessment(SchemaModel):
    """FR-7.3.3: how the screen changed between two verified observations."""

    transition: ScreenTransition
    previous_question_hash: Optional[str] = None
    question_hash: Optional[str] = None
    already_answered: bool = False
    reasons: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Solver (section 7.4)
# --------------------------------------------------------------------------- #
class SolverSample(SchemaModel):
    """One independent sample in a self-consistency vote (FR-7.4.3)."""

    option_index: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""
    temperature: float = 0.7


class Decision(SchemaModel):
    """Output of section 7.4.  Pure data -- no screen access happened here."""

    question_hash: str
    option_index: int = Field(..., ge=0)
    strategy: SolverStrategy
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""
    samples: Optional[List[SolverSample]] = None
    # ADDITIVE
    latency_ms: float = 0.0
    provider: Optional[str] = None
    attempts: int = 1
    letter: Optional[str] = None

    @field_validator("letter", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        return None if value in ("", None) else value


class StrategyAttempt(SchemaModel):
    """One link of the strategy chain that was tried (FR-7.13.4 decision trace)."""

    strategy: SolverStrategy
    attempted: bool = True
    produced_answer: bool = False
    option_index: Optional[int] = None
    confidence: float = 0.0
    accepted: bool = False
    skipped_reason: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Confidence (section 7.5)
# --------------------------------------------------------------------------- #
class ConfidenceBreakdown(SchemaModel):
    """Composite score components + weights (FR-7.5.1)."""

    ocr_confidence: float = Field(..., ge=0.0, le=1.0)
    perception_agreement: float = Field(..., ge=0.0, le=1.0)
    solver_confidence: float = Field(..., ge=0.0, le=1.0)
    penalty: float = Field(0.0, ge=0.0, le=1.0)
    weights: Dict[str, float] = Field(default_factory=dict)
    composite: float = Field(..., ge=0.0, le=1.0)
    tier: Literal["high", "mid", "low"]
    policy: UncertaintyPolicy
    reasons: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Action / binding / verification (sections 7.6, 7.7, 7.9)
# --------------------------------------------------------------------------- #
class ExpectedEffect(SchemaModel):
    """The declared intent that drives generic verification (FR-7.9.3)."""

    type: ExpectedEffectType
    target_handle: Optional[str] = None
    region: Optional[Box] = None
    min_region_change_pct: float = Field(5.0, ge=0.0, le=100.0)
    expected_marker: Optional[SelectedMarker] = None
    expected_transition: Optional[ScreenTransition] = None

    @field_validator("region", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value


class Intent(SchemaModel):
    """Orchestrator -> actuator + verifier.

    ``frame_seq`` is the frame in which ``target_box`` was validated.  The
    actuator refuses to execute when it does not match the live frame: that is
    the audit hook for **[L1]** / **[AC-14.2]**.
    """

    intent_id: str  # ADDITIVE: stable id linking ledger <-> verification record
    action: ActionType
    handle: Optional[str] = Field(None, description="semantic element id (FR-7.7.1)")
    target_box: Optional[Box] = None
    expected_effect: ExpectedEffect
    max_wait_ms: int = Field(3000, gt=0)
    verify: Literal["standard", "strict", "none"] = "standard"
    frame_seq: int = Field(-1, description="frame the box was validated in (L1)")
    # ADDITIVE
    correlation_id: Optional[str] = None
    scroll_delta: Optional[int] = None
    key_name: Optional[str] = None
    type_text: Optional[str] = None
    jitter_seed: Optional[int] = None

    @field_validator("target_box", mode="before")
    @classmethod
    def _coerce_box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value

    @property
    def is_state_changing(self) -> bool:
        return self.action in STATE_CHANGING_ACTIONS


class ActionRecord(SchemaModel):
    """What the actuator physically did (intent ledger entry)."""

    intent_id: str
    action: ActionType
    point: Optional[Tuple[int, int]] = None
    box: Optional[Box] = None
    frame_seq_at_execution: int
    started_ts: float
    finished_ts: float
    backend: str
    jitter_px: float = 0.0
    move_duration_s: float = 0.0
    refused: bool = False
    refusal_reason: Optional[str] = None


class VerificationRecord(SchemaModel):
    """Proof that an action achieved its declared effect (AC-14.3)."""

    intent_id: str
    passed: bool
    evidence: List[str] = Field(default_factory=list)
    pre_frame_seq: int
    post_frame_seq: int
    region_change_pct: float = 0.0
    marker_before: Optional[SelectedMarker] = None
    marker_after: Optional[SelectedMarker] = None
    elapsed_ms: float = 0.0
    attempts: int = 1
    expected_effect_type: ExpectedEffectType
    transition: Optional[ScreenTransition] = None


class ElementResolution(SchemaModel):
    """FR-7.7.2 re-resolution outcome, immediately before an action."""

    handle: str
    resolved_box: Box
    match_confidence: float = Field(..., ge=0.0, le=1.0)
    displacement_px: float = 0.0
    method: Literal["template_match", "phase_correlation", "full_reperception", "relative_geometry"]
    frame_seq: int
    escalated: bool = False


# --------------------------------------------------------------------------- #
# Telemetry / persistence (sections 7.12, 7.13)
# --------------------------------------------------------------------------- #
class RunEvent(SchemaModel):
    ts: float
    run_id: str
    state: State
    event: RunEventName
    code: Optional[FailureCode] = None
    latency_ms: float = 0.0
    trace_img: Optional[str] = None
    # ADDITIVE
    correlation_id: Optional[str] = None
    detail: Dict[str, Any] = Field(default_factory=dict)


class DecisionTrace(SchemaModel):
    """The operator's primary review surface (FR-7.13.4)."""

    correlation_id: str
    question_hash: str
    ordinal: int
    question_text: str = ""
    option_texts: List[str] = Field(default_factory=list)
    attempts: List[StrategyAttempt] = Field(default_factory=list)
    decision: Optional[Decision] = None
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    verification: Optional[VerificationRecord] = None
    navigation_verified: Optional[bool] = None
    frame_seq: int = -1
    total_latency_ms: float = 0.0
    outcome: str = "pending"

    def human_summary(self) -> str:
        decision = self.decision
        if decision is None:
            return f"[{self.ordinal}] {self.outcome} (no decision)"
        letter = decision.letter or chr(ord("A") + decision.option_index)
        chain = " -> ".join(
            f"{a.strategy.value}({a.confidence:.2f}{',accepted' if a.accepted else ''})"
            for a in self.attempts
            if a.attempted
        )
        verified = "n/a" if self.verification is None else ("verified" if self.verification.passed else "UNVERIFIED")
        return (
            f"[{self.ordinal}] {decision.strategy.value} -> {letter} "
            f"conf={decision.confidence:.2f} composite="
            f"{self.confidence_breakdown.composite:.2f} {verified} | {chain}"
            if self.confidence_breakdown
            else f"[{self.ordinal}] {decision.strategy.value} -> {letter} conf={decision.confidence:.2f} {verified} | {chain}"
        )


class AnsweredQuestion(SchemaModel):
    """Session ledger entry (FR-7.12.1)."""

    hash: str
    ordinal: int
    chosen_index: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: float
    verified: bool = False
    decision_trace_id: Optional[str] = None
    content_hash: str = ""
    strategy: SolverStrategy = SolverStrategy.NONE


class BudgetCounters(SchemaModel):
    """Global run budget (FR-7.10.2)."""

    questions_answered: int = 0
    consecutive_failures: int = 0
    total_failures: int = 0
    recoveries_invoked: int = 0
    recoveries_succeeded: int = 0
    elapsed_s: float = 0.0
    cycles: int = 0
    human_interventions: int = 0
    model_calls: int = 0


class SessionState(SchemaModel):
    """Crash-recovery / resume payload (FR-7.12.1, L9)."""

    run_id: str
    started_at: float
    platform_profile: Dict[str, Any] = Field(default_factory=dict)
    questions_answered: List[AnsweredQuestion] = Field(default_factory=list)
    state: State = State.IDLE
    budget_counters: BudgetCounters = Field(default_factory=BudgetCounters)
    # ADDITIVE
    last_verified_at: Optional[float] = None
    last_question_hash: Optional[str] = None
    resume_count: int = 0
    attestation_text: Optional[str] = None
    config_fingerprint: Optional[str] = None

    def answered_hashes(self) -> set[str]:
        return {q.hash for q in self.questions_answered}

    def answered_content_hashes(self) -> set[str]:
        return {q.content_hash for q in self.questions_answered if q.content_hash}


class FailureBundleRef(SchemaModel):
    code: FailureCode
    path: str
    created_ts: float
    state: State
    frames: List[str] = Field(default_factory=list)


class EnvIndicator(SchemaModel):
    kind: Literal["process", "window_title", "overlay", "display_state", "config"]
    name: str
    detail: str = ""
    restricted: bool = True


class EnvScanResult(SchemaModel):
    """Section 7.14 environment scan output."""

    ts: float
    restricted: bool = False
    indicators: List[EnvIndicator] = Field(default_factory=list)
    scanned_processes: int = 0
    halt_reason: Optional[str] = None

    @property
    def summary(self) -> str:
        if not self.restricted:
            return f"clear ({self.scanned_processes} processes scanned)"
        names = ", ".join(i.name for i in self.indicators[:4])
        return f"RESTRICTED: {names}"


class Attestation(SchemaModel):
    """Recorded operator authorization (FR-7.14.1)."""

    text: str
    operator: str = "local"
    recorded_at: float = Field(default_factory=time.time)
    method: Literal["cli", "config", "session"] = "cli"


class RunReport(SchemaModel):
    """Terminal-state report (FR-7.12.2)."""

    run_id: str
    outcome: RunOutcome
    started_at: float
    finished_at: float
    questions: List[AnsweredQuestion] = Field(default_factory=list)
    decision_traces: List[DecisionTrace] = Field(default_factory=list)
    total_time_s: float = 0.0
    failure_bundles: List[FailureBundleRef] = Field(default_factory=list)
    env_scan_summary: str = ""
    attestation: Optional[Attestation] = None
    budget_counters: BudgetCounters = Field(default_factory=BudgetCounters)
    # ADDITIVE
    accuracy: Optional[float] = None
    correct: Optional[int] = None
    incorrect: Optional[int] = None
    unverified_actions: int = 0
    illegal_transitions: int = 0
    stale_coordinate_violations: int = 0
    halted_code: Optional[FailureCode] = None
    halted_detail: str = ""
    config_fingerprint: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at


# --------------------------------------------------------------------------- #
# Model provider abstraction (FR-6.2)
# --------------------------------------------------------------------------- #
class ModelMessage(SchemaModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images_b64: List[str] = Field(default_factory=list)


class ModelRequest(SchemaModel):
    task: Literal["solve", "perceive", "transcribe_math", "describe_image", "classify_popup"]
    messages: List[ModelMessage] = Field(default_factory=list)
    response_schema: Optional[Dict[str, Any]] = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0)
    timeout_s: float = Field(30.0, gt=0)
    correlation_id: Optional[str] = None


class ModelResponse(SchemaModel):
    text: str = ""
    parsed: Optional[Dict[str, Any]] = None
    provider: str
    model: str = ""
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    attempts: int = 1
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SolverResult(SchemaModel):
    """Vendor-neutral, validated model answer (never an executable action)."""

    selected_option_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    provider: str
    model: str
    latency_ms: float = Field(0.0, ge=0.0)
    answer_text: Optional[str] = None
    verification_status: Optional[Literal["unverified", "agreed", "disagreed"]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("selected_option_id")
    @classmethod
    def _option_identifier_only(cls, value: str) -> str:
        # This is a logical option identifier, not a coordinate or UI selector.
        import re
        if not re.fullmatch(r"option_[1-9][0-9]*", value):
            raise ValueError("selected_option_id must use the option_N contract")
        return value


class ModelFailure(SchemaModel):
    code: FailureCode
    provider: str
    detail: str
    attempts: int
