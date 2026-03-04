# Version: v1.0.0
# Optimization Summary:
# - Centralize semantic rule registration and execution orchestration.
# - Keep normalization module thin while preserving existing behavior.
# - Provide apply/shadow helpers for staged enterprise rollout.

from __future__ import annotations

from typing import Any, Callable

from .rules_engine import CallableSemanticRule, run_semantic_rule_engine


SemanticFn = Callable[[list[Any], list[list[str | None]], Any, int], int]


def build_default_semantic_rules(
    recover_key_identifier_cells_fn: SemanticFn,
    repair_directory_listing_structure_fn: SemanticFn,
    merge_filename_continuations_fn: Callable[[list[Any], list[list[str | None]], int], int],
) -> list[CallableSemanticRule]:
    return [
        CallableSemanticRule(
            "recover_key_identifier_cells",
            lambda rs, gd, rw, lc: recover_key_identifier_cells_fn(rs, gd, rw, lc),
        ),
        CallableSemanticRule(
            "repair_directory_listing_structure",
            lambda rs, gd, rw, lc: repair_directory_listing_structure_fn(rs, gd, rw, lc),
        ),
        CallableSemanticRule(
            "merge_filename_continuations",
            lambda rs, gd, _rw, lc: merge_filename_continuations_fn(rs, gd, lc),
        ),
    ]


def run_apply_semantic_rules(
    *,
    rows: list[Any],
    grid: list[list[str | None]],
    raw_evidence: Any,
    logical_col_count: int,
    rules: list[CallableSemanticRule],
) -> int:
    report = run_semantic_rule_engine(
        rows=rows,
        grid=grid,
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
        rules=rules,
        shadow_mode=False,
    )
    return int(report.total_changed_cells)


def run_shadow_semantic_rules(
    *,
    rows: list[Any],
    grid: list[list[str | None]],
    raw_evidence: Any,
    logical_col_count: int,
    rules: list[CallableSemanticRule],
    emit_diagnostics: bool,
) -> dict[str, Any]:
    report = run_semantic_rule_engine(
        rows=rows,
        grid=grid,
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
        rules=rules,
        shadow_mode=True,
    )
    if not emit_diagnostics:
        return {"mode": report.mode, "total_changed_cells": int(report.total_changed_cells)}
    return {
        "mode": report.mode,
        "total_changed_cells": int(report.total_changed_cells),
        "records": [
            {
                "rule_id": item.rule_id,
                "mode": item.mode,
                "changed_cells": int(item.changed_cells),
            }
            for item in report.records
        ],
    }


__all__ = [
    "build_default_semantic_rules",
    "run_apply_semantic_rules",
    "run_shadow_semantic_rules",
]
