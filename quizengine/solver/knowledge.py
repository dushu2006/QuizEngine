"""Local knowledge-base strategies (FR-7.4.1 step 1: local exact + fuzzy).

The knowledge base is the QuizForge answer key (section 13.4) or any operator
supplied JSON/YAML of ``{question, answer, options?}`` pairs.  It is consulted
before any model call, which keeps the common case offline, deterministic and
free (**L6** local-first, FR-16.1 no crops leave the machine).

Re-ordering safe: when an entry records its own option list the answer is
resolved to *text* first and only then bound against the live Question, so a
shuffled option order cannot produce a wrong click (FR-7.4.2).
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from ..config import SolverConfig
from ..contracts import Question, SolverStrategy
from .base import SolverContext, SolverStrategyBase, StrategyResult, bind_answer_to_index, normalize


@dataclass(frozen=True)
class KnowledgeEntry:
    question: str
    answer: str
    options: tuple = ()
    tags: tuple = ()
    source: str = "inline"

    @property
    def normalized(self) -> str:
        return normalize(self.question)


@dataclass(frozen=True)
class KnowledgeMatch:
    entry: KnowledgeEntry
    ratio: float
    kind: str  # exact | normalized | fuzzy
    answer_index: Optional[int] = None
    answer_text: Optional[str] = None


class KnowledgeBase:
    """Deterministic lookup over Q&A pairs."""

    def __init__(self, entries: Sequence[KnowledgeEntry] = (), *, fuzzy_threshold: float = 0.90) -> None:
        self.entries: List[KnowledgeEntry] = list(entries)
        self.fuzzy_threshold = float(fuzzy_threshold)
        self._by_raw: Dict[str, KnowledgeEntry] = {}
        self._by_norm: Dict[str, KnowledgeEntry] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        self._by_raw = {entry.question.strip(): entry for entry in self.entries}
        self._by_norm = {}
        for entry in self.entries:
            self._by_norm.setdefault(entry.normalized, entry)

    # -- loading ----------------------------------------------------------- #
    @classmethod
    def load(cls, path: Union[str, Path], *, fuzzy_threshold: float = 0.90) -> "KnowledgeBase":
        file_path = Path(path).expanduser()
        text = file_path.read_text(encoding="utf-8")
        payload = json.loads(text) if file_path.suffix.lower() == ".json" else yaml.safe_load(text)
        return cls.from_payload(payload, fuzzy_threshold=fuzzy_threshold, source=str(file_path))

    @classmethod
    def from_payload(
        cls, payload: Any, *, fuzzy_threshold: float = 0.90, source: str = "inline"
    ) -> "KnowledgeBase":
        entries: List[KnowledgeEntry] = []
        if payload is None:
            return cls(entries, fuzzy_threshold=fuzzy_threshold)
        if isinstance(payload, dict):
            pairs = payload.get("pairs") or payload.get("entries") or payload.get("questions")
            if isinstance(pairs, list):
                items: List[Any] = list(pairs)
            else:
                # flat mapping: {"question text": "answer text"}
                items = [{"question": k, "answer": v} for k, v in payload.items() if isinstance(v, str)]
        elif isinstance(payload, list):
            items = list(payload)
        else:
            raise ValueError(f"unsupported knowledge-base payload type: {type(payload).__name__}")

        for item in items:
            if isinstance(item, str):
                continue
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or item.get("q") or "").strip()
            answer = item.get("answer")
            if answer is None:
                answer = item.get("correct") or item.get("a") or item.get("solution")
            if not question or answer is None:
                continue
            if isinstance(answer, (list, tuple)):
                answer_text = str(answer[0]) if answer else ""
            elif isinstance(answer, dict):
                answer_text = str(answer.get("text") or answer.get("letter") or "")
            else:
                answer_text = str(answer)
            options = tuple(str(o) for o in (item.get("options") or ()))
            tags = tuple(str(t) for t in (item.get("tags") or ()))
            entries.append(
                KnowledgeEntry(
                    question=question,
                    answer=answer_text.strip(),
                    options=options,
                    tags=tags,
                    source=str(item.get("source") or source),
                )
            )
        return cls(entries, fuzzy_threshold=fuzzy_threshold)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    # -- lookup ------------------------------------------------------------ #
    def lookup(self, question_text: str, options: Sequence[str] = ()) -> Optional[KnowledgeMatch]:
        raw = (question_text or "").strip()
        if not raw:
            return None
        entry = self._by_raw.get(raw)
        if entry is not None:
            return self._resolve(entry, 1.0, "exact", options)
        norm = normalize(raw)
        entry = self._by_norm.get(norm)
        if entry is not None:
            return self._resolve(entry, 1.0, "normalized", options)
        best: Optional[KnowledgeMatch] = None
        for candidate in self.entries:
            ratio = difflib.SequenceMatcher(None, norm, candidate.normalized).ratio()
            if ratio < self.fuzzy_threshold:
                continue
            match = self._resolve(candidate, ratio, "fuzzy", options)
            if match.answer_index is None:
                continue
            if best is None or ratio > best.ratio:
                best = match
        return best

    def _resolve(
        self, entry: KnowledgeEntry, ratio: float, kind: str, options: Sequence[str]
    ) -> KnowledgeMatch:
        """Map the stored answer onto the *live* option order."""
        answer_text = entry.answer
        if entry.options:
            stored_norm = [normalize(o) for o in entry.options]
            try:
                position = stored_norm.index(normalize(answer_text))
            except ValueError:
                position = None
                # the entry may store a letter instead of the text
                if position is None and len(answer_text) == 1 and answer_text.isalpha():
                    position = ord(answer_text.upper()) - ord("A")
            if position is not None and position < len(entry.options):
                answer_text = entry.options[position]
        index = None
        if options:
            for position, option in enumerate(options):
                if normalize(option) == normalize(answer_text):
                    index = position
                    break
            if index is None and len(answer_text) == 1 and answer_text.isalpha():
                position = ord(answer_text.upper()) - ord("A")
                index = position if position < len(options) else None
            if index is None:
                close = difflib.get_close_matches(normalize(answer_text), [normalize(o) for o in options], n=1, cutoff=0.85)
                if close:
                    index = [normalize(o) for o in options].index(close[0])
        return KnowledgeMatch(entry=entry, ratio=ratio, kind=kind, answer_index=index, answer_text=answer_text)


class LocalExactStrategy(SolverStrategyBase):
    strategy = SolverStrategy.LOCAL_EXACT

    def __init__(self, config: SolverConfig, knowledge: Optional[KnowledgeBase] = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.knowledge = knowledge or KnowledgeBase(fuzzy_threshold=config.fuzzy_threshold)

    def available(self) -> bool:
        return bool(self.knowledge)

    def unavailable_reason(self) -> str:
        return "no knowledge base configured"

    def solve(self, question: Question, context: SolverContext) -> Optional[StrategyResult]:
        match = self.knowledge.lookup(question.text, [o.text for o in question.options])
        if match is None or match.kind == "fuzzy" or match.answer_index is None:
            return None
        confidence = 0.99 if match.kind == "exact" else 0.97
        return StrategyResult(
            option_index=match.answer_index,
            confidence=confidence * max(0.9, min(1.0, match.ratio)),
            rationale=(
                f"local knowledge base ({match.kind} match, source={match.entry.source}): "
                f"{match.entry.answer!r}"
            ),
            strategy=self.strategy,
            raw={"kind": match.kind, "ratio": round(match.ratio, 4), "source": match.entry.source},
        )


class LocalFuzzyStrategy(SolverStrategyBase):
    strategy = SolverStrategy.LOCAL_FUZZY

    def __init__(self, config: SolverConfig, knowledge: Optional[KnowledgeBase] = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.knowledge = knowledge or KnowledgeBase(fuzzy_threshold=config.fuzzy_threshold)

    def available(self) -> bool:
        return bool(self.knowledge)

    def unavailable_reason(self) -> str:
        return "no knowledge base configured"

    def solve(self, question: Question, context: SolverContext) -> Optional[StrategyResult]:
        match = self.knowledge.lookup(question.text, [o.text for o in question.options])
        if match is None or match.answer_index is None:
            return None
        if match.ratio < self.config.fuzzy_threshold:
            return None
        confidence = 0.90 + 0.08 * max(0.0, min(1.0, (match.ratio - 0.90) / 0.10))
        return StrategyResult(
            option_index=match.answer_index,
            confidence=min(0.97, confidence),
            rationale=(
                f"local knowledge base (fuzzy match {match.ratio:.2f} >= "
                f"{self.config.fuzzy_threshold}, source={match.entry.source})"
            ),
            strategy=self.strategy,
            raw={"kind": "fuzzy", "ratio": round(match.ratio, 4), "source": match.entry.source},
        )


__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeMatch",
    "LocalExactStrategy",
    "LocalFuzzyStrategy",
]
