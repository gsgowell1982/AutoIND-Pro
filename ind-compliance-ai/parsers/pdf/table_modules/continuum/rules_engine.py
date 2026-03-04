# Version: v1.0.1
# Optimization Summary:
# - Introduce a lightweight semantic rule-engine scaffold for phased rollout.
# - Support shadow-mode execution to validate rule effects without mutating output.
# - Provide deterministic execution order and per-rule change accounting.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RuleContext:
    rows: list[Any]
    grid: list[list[str | None]]
    raw_evidence: Any
    logical_col_count: int


@dataclass
class RuleRunRecord:
    rule_id: str
    mode: str
    changed_cells: int


@dataclass
class RuleEngineReport:
    mode: str
    total_changed_cells: int = 0
    records: list[RuleRunRecord] = field(default_factory=list)


class SemanticRule:
    """Rule interface for semantic table post-normalization repairs."""

    rule_id: str = "unnamed_rule"

    def run(self, ctx: RuleContext) -> int:
        raise NotImplementedError


class CallableSemanticRule(SemanticRule):
    """Adapter to register existing normalization functions as semantic rules."""

    def __init__(self, rule_id: str, fn: Callable[[list[Any], list[list[str | None]], Any, int], int]) -> None:
        self.rule_id = rule_id
        self._fn = fn

    def run(self, ctx: RuleContext) -> int:
        return int(self._fn(ctx.rows, ctx.grid, ctx.raw_evidence, ctx.logical_col_count) or 0)


def run_semantic_rule_engine(
    *,
    rows: list[Any],
    grid: list[list[str | None]],
    raw_evidence: Any,
    logical_col_count: int,
    rules: list[SemanticRule],
    shadow_mode: bool,
) -> RuleEngineReport:
    """Run semantic rules in apply or shadow mode.

    Shadow mode executes rules on deep-copied data and only reports potential deltas.
    """
    if shadow_mode:
        work_rows = deepcopy(rows)
        work_grid = deepcopy(grid)
        mode = "shadow"
    else:
        work_rows = rows
        work_grid = grid
        mode = "apply"

    ctx = RuleContext(
        rows=work_rows,
        grid=work_grid,
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
    )
    report = RuleEngineReport(mode=mode)

    for rule in rules:
        changed = max(0, int(rule.run(ctx)))
        report.total_changed_cells += changed
        report.records.append(
            RuleRunRecord(
                rule_id=rule.rule_id,
                mode=mode,
                changed_cells=changed,
            )
        )

    return report


__all__ = [
    "RuleContext",
    "RuleRunRecord",
    "RuleEngineReport",
    "SemanticRule",
    "CallableSemanticRule",
    "run_semantic_rule_engine",
]
