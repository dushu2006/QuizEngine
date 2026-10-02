"""Deterministic local rule solver (FR-7.4.1 step 2).

Pure-Python handlers for question shapes that need no model at all: arithmetic,
percentages, unit conversion, date arithmetic, sequences and roman numerals.
Each handler either returns a clean answer with high confidence or ``None`` --
never a guess -- so the cascade can fall through to the LLM.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..arith import find_expression, numbers_in, numbers_match, safe_eval
from ..config import SolverConfig
from ..contracts import Question, SolverStrategy
from .base import SolverContext, SolverStrategyBase, StrategyResult

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B, %Y",
    "%Y/%m/%d",
)
_DATE_RE = re.compile(
    r"(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})"
)

_TIME_UNITS = {
    "second": 1.0,
    "sec": 1.0,
    "s": 1.0,
    "minute": 60.0,
    "min": 60.0,
    "hour": 3600.0,
    "hr": 3600.0,
    "h": 3600.0,
    "day": 86400.0,
    "d": 86400.0,
    "week": 604800.0,
    "wk": 604800.0,
    "fortnight": 1209600.0,
    "year": 31557600.0,
    "yr": 31557600.0,
}
_LENGTH_UNITS = {
    "millimeter": 0.001,
    "millimetre": 0.001,
    "mm": 0.001,
    "centimeter": 0.01,
    "centimetre": 0.01,
    "cm": 0.01,
    "meter": 1.0,
    "metre": 1.0,
    "m": 1.0,
    "kilometer": 1000.0,
    "kilometre": 1000.0,
    "km": 1000.0,
    "inch": 0.0254,
    "in": 0.0254,
    "foot": 0.3048,
    "feet": 0.3048,
    "ft": 0.3048,
    "yard": 0.9144,
    "yd": 0.9144,
    "mile": 1609.344,
    "mi": 1609.344,
}
_MASS_UNITS = {
    "milligram": 1e-6,
    "mg": 1e-6,
    "gram": 0.001,
    "g": 0.001,
    "kilogram": 1.0,
    "kg": 1.0,
    "tonne": 1000.0,
    "ton": 1000.0,
    "t": 1000.0,
    "pound": 0.45359237,
    "lb": 0.45359237,
    "ounce": 0.028349523,
    "oz": 0.028349523,
}
_VOLUME_UNITS = {
    "milliliter": 0.001,
    "millilitre": 0.001,
    "ml": 0.001,
    "liter": 1.0,
    "litre": 1.0,
    "l": 1.0,
    "gallon": 3.785411784,
    "gal": 3.785411784,
}
_UNIT_FAMILIES: Tuple[Dict[str, float], ...] = (
    _LENGTH_UNITS,
    _MASS_UNITS,
    _VOLUME_UNITS,
    _TIME_UNITS,
)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


@dataclass
class RuleHit:
    value: float
    label: str
    confidence: float
    as_text: Optional[str] = None


class LocalRulesStrategy(SolverStrategyBase):
    strategy = SolverStrategy.LOCAL_RULES

    def __init__(self, config: SolverConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.handlers: List[Tuple[str, Callable[[str, Sequence[str]], Optional[RuleHit]]]] = [
            ("arithmetic", self._arithmetic),
            ("percentage", self._percentage),
            ("unit_conversion", self._unit_conversion),
            ("date_difference", self._date_difference),
            ("date_offset", self._date_offset),
            ("weekday", self._weekday),
            ("sequence", self._sequence),
            ("roman_numeral", self._roman_numeral),
            ("counting", self._counting),
        ]

    def solve(self, question: Question, context: SolverContext) -> Optional[StrategyResult]:
        text = question.text
        if context.extra.get("math_latex"):
            # Prefer the VLM/OCR LaTeX reading of the expression when present.
            text = str(context.extra["math_latex"]).replace("\\frac", "frac") + " " + text
        option_texts = [o.text for o in question.options]
        for name, handler in self.handlers:
            try:
                hit = handler(text, option_texts)
            except Exception:
                hit = None
            if hit is None:
                continue
            index = self._match_option(hit, option_texts)
            if index is None:
                continue
            return StrategyResult(
                option_index=index,
                confidence=hit.confidence,
                rationale=f"local rule '{name}': computed {hit.label}",
                strategy=self.strategy,
                raw={"rule": name, "value": hit.value, "label": hit.label},
            )
        return None

    # -- option matching --------------------------------------------------- #
    @staticmethod
    def _match_option(hit: RuleHit, option_texts: Sequence[str]) -> Optional[int]:
        if hit.as_text is not None:
            wanted = normalize(hit.as_text)
            for index, option in enumerate(option_texts):
                if normalize(option) == wanted:
                    return index
            for index, option in enumerate(option_texts):
                if wanted and wanted in normalize(option):
                    return index
        for index, option in enumerate(option_texts):
            if numbers_match(hit.value, option, tolerance=_tolerance(hit.value)):
                return index
        return None

    # -- handlers ---------------------------------------------------------- #
    @staticmethod
    def _arithmetic(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        expression = find_expression(text)
        if expression is None:
            return None
        value = safe_eval(expression)
        if value is None:
            return None
        return RuleHit(value=value, label=f"{expression} = {_fmt(value)}", confidence=0.96)

    @staticmethod
    def _percentage(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s*cent)\s*(?:of|from)\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE
        )
        if not match:
            match = re.search(
                r"(?:what\s+(?:is|are)\s+)(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)?\s*(\d+(?:\.\d+)?)",
                text,
                re.IGNORECASE,
            )
        if not match:
            return None
        percent, base = float(match.group(1)), float(match.group(2))
        value = percent / 100.0 * base
        return RuleHit(value=value, label=f"{percent}% of {base} = {_fmt(value)}", confidence=0.95)

    @staticmethod
    def _unit_conversion(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        lowered = text.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]+)\s*(?:is|to|in|as|equals|=)\s*(?:how\s+many\s+)?([a-z]+)", lowered)
        if not match:
            match = re.search(r"how\s+many\s+([a-z]+)\s*(?:are|is)?\s*(?:there\s+)?in\s*(\d+(?:\.\d+)?)\s*([a-z]+)", lowered)
            if not match:
                return None
            target, value, source = match.group(1), float(match.group(2)), match.group(3)
        else:
            value, source, target = float(match.group(1)), match.group(2), match.group(3)
        source = source.rstrip("s") if source not in ("s", "in", "l", "t", "m", "g", "h", "d") else source
        target = target.rstrip("s") if target not in ("s", "in", "l", "t", "m", "g", "h", "d") else target
        for family in _UNIT_FAMILIES:
            if source in family and target in family:
                converted = value * family[source] / family[target]
                return RuleHit(
                    value=converted,
                    label=f"{value} {source} -> {_fmt(converted)} {target}",
                    confidence=0.93,
                )
        return None

    @staticmethod
    def _date_difference(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        lowered = text.lower()
        if not re.search(r"(how\s+many\s+days|days\s+between|difference\s+in\s+days)", lowered):
            return None
        dates = _parse_dates(text)
        if len(dates) < 2:
            return None
        first, second = sorted(dates)[:2]
        days = (second - first).days
        return RuleHit(value=float(days), label=f"{first.date()} to {second.date()} = {days} day(s)", confidence=0.94)

    @staticmethod
    def _date_offset(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        lowered = text.lower()
        match = re.search(r"(\d+)\s*(day|week|month|year)s?\s*(after|before|from)", lowered)
        if not match:
            return None
        dates = _parse_dates(text)
        if not dates:
            return None
        amount, unit, direction = int(match.group(1)), match.group(2), match.group(3)
        base = dates[0]
        if unit == "day":
            delta = _dt.timedelta(days=amount)
        elif unit == "week":
            delta = _dt.timedelta(weeks=amount)
        elif unit == "month":
            delta = _dt.timedelta(days=amount * 30)
        else:
            delta = _dt.timedelta(days=amount * 365)
        result = base + delta if direction == "after" else base - delta
        return RuleHit(
            value=float(result.toordinal()),
            label=result.strftime("%d %B %Y"),
            confidence=0.9,
            as_text=_date_variants(result),
        )

    @staticmethod
    def _weekday(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        if not re.search(r"(day\s+of\s+the\s+week|which\s+day|what\s+day)", text, re.IGNORECASE):
            return None
        dates = _parse_dates(text)
        if not dates:
            return None
        weekday = dates[0].strftime("%A")
        return RuleHit(
            value=float(dates[0].weekday()),
            label=weekday,
            confidence=0.93,
            as_text=_weekday_variants(weekday),
        )

    @staticmethod
    def _sequence(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
        if len(numbers) < 3:
            return None
        if not re.search(r"(next|comes|missing|continue|sequence|series)", text, re.IGNORECASE):
            return None
        diffs = [b - a for a, b in zip(numbers, numbers[1:])]
        if len(set(diffs)) == 1:
            return RuleHit(value=numbers[-1] + diffs[0], label=f"arithmetic sequence -> {_fmt(numbers[-1] + diffs[0])}", confidence=0.9)
        ratios = [b / a for a, b in zip(numbers, numbers[1:]) if a]
        if ratios and all(abs(r - ratios[0]) < 1e-9 for r in ratios):
            return RuleHit(value=numbers[-1] * ratios[0], label=f"geometric sequence -> {_fmt(numbers[-1] * ratios[0])}", confidence=0.9)
        return None

    @staticmethod
    def _roman_numeral(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        if not re.search(r"roman", text, re.IGNORECASE):
            return None
        match = re.search(r"\b([IVXLCDM]{2,})\b", text)
        target_arabic = re.search(r"(\d{1,4})", text)
        if match:
            value = _roman_to_int(match.group(1))
            if value is not None:
                return RuleHit(value=float(value), label=f"{match.group(1)} = {value}", confidence=0.93)
        if target_arabic:
            numeral = _int_to_roman(int(target_arabic.group(1)))
            if numeral:
                return RuleHit(value=float(int(target_arabic.group(1))), label=numeral, confidence=0.93, as_text=numeral)
        return None

    @staticmethod
    def _counting(text: str, options: Sequence[str]) -> Optional[RuleHit]:
        """'How many letters/words in <quoted string>'."""
        match = re.search(r"how\s+many\s+(letters|words|vowels|consonants|digits)\s+.*?[\"'“]([^\"'”]+)[\"'”]", text, re.IGNORECASE)
        if not match:
            return None
        kind, payload = match.group(1).lower(), match.group(2)
        if kind == "letters":
            value = sum(1 for c in payload if c.isalpha())
        elif kind == "words":
            value = len(payload.split())
        elif kind == "digits":
            value = sum(1 for c in payload if c.isdigit())
        elif kind == "vowels":
            value = sum(1 for c in payload.lower() if c in "aeiou")
        else:
            value = sum(1 for c in payload.lower() if c.isalpha() and c not in "aeiou")
        return RuleHit(value=float(value), label=f"{kind} in {payload!r} = {value}", confidence=0.92)


# -- helpers ------------------------------------------------------------------ #
def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z\u00c0-\u024f]+", " ", (text or "").lower()).split())


def _tolerance(value: float) -> float:
    magnitude = abs(value)
    if magnitude >= 1000:
        return 0.5
    if magnitude >= 1:
        return 1e-6
    return 1e-9


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


def _parse_dates(text: str) -> List[_dt.datetime]:
    found: List[_dt.datetime] = []
    for chunk in _DATE_RE.findall(text or ""):
        for fmt in _DATE_FORMATS:
            try:
                parsed = _dt.datetime.strptime(chunk.strip(), fmt)
            except ValueError:
                continue
            if parsed not in found:
                found.append(parsed)
            break
    return found


def _date_variants(moment: _dt.datetime) -> str:
    return moment.strftime("%d %B %Y")


def _weekday_variants(weekday: str) -> str:
    return weekday


def _roman_to_int(numeral: str) -> Optional[int]:
    total = 0
    previous = 0
    for char in reversed(numeral.upper()):
        value = _ROMAN_VALUES.get(char)
        if value is None:
            return None
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total


def _int_to_roman(value: int) -> Optional[str]:
    if not 1 <= value <= 3999:
        return None
    pairs = (
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    )
    out: List[str] = []
    for number, symbol in pairs:
        while value >= number:
            out.append(symbol)
            value -= number
    return "".join(out)


__all__ = ["LocalRulesStrategy", "RuleHit"]
