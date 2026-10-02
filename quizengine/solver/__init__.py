"""Solver Module (PRD section 7.4): cascade, binding, self-consistency, budget."""

from __future__ import annotations

from .base import SolverContext, SolverStrategyBase, StrategyResult, bind_answer_to_index, make_decision, normalize
from .knowledge import KnowledgeBase, KnowledgeEntry, KnowledgeMatch, LocalExactStrategy, LocalFuzzyStrategy
from .llm import LLMStrategy, VLMStrategy
from .module import SolverModule, SolverOutcome
from .rules import LocalRulesStrategy, RuleHit
from .self_consistency import SelfConsistency

__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeMatch",
    "LLMStrategy",
    "LocalExactStrategy",
    "LocalFuzzyStrategy",
    "LocalRulesStrategy",
    "RuleHit",
    "SelfConsistency",
    "SolverContext",
    "SolverModule",
    "SolverOutcome",
    "SolverStrategyBase",
    "StrategyResult",
    "VLMStrategy",
    "bind_answer_to_index",
    "make_decision",
    "normalize",
]
