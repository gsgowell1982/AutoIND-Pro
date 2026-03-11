# Version: v1.1.2
# Optimization Summary:
# - Introduce configurable PDF parser settings using TOML.
# - Centralize cross-page stitching thresholds for enterprise tuning.
# - Provide cached loading with safe defaults and optional env override.
# - Add non-destructive table-content diagnostics controls.
# - Add semantic rule-engine policy for phased enterprise rollout.
# - Add table-detection policy for supplemental candidate merge and two-column gating.
# - Expose preceding-text guard thresholds used by cross-page stitching to avoid false continuations.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import tomllib
from typing import Any


@dataclass(slots=True)
class CrossPageStitchingThresholds:
    prev_near_bottom_ratio: float = 0.72
    curr_near_top_ratio: float = 0.28
    hint_glossary_similarity_min: float = 0.22
    default_similarity_min: float = 0.28
    high_similarity_link_threshold: float = 0.82
    title_context_similarity_threshold: float = 0.42
    context_overlap_min: float = 0.45
    overlap_min_without_hint: float = 0.35
    low_similarity_guard_threshold: float = 0.50
    low_similarity_guard_overlap_min: float = 0.55
    preceding_text_gap_max: float = 32.0
    preceding_text_min_length: int = 12
    preceding_text_ignore_top_margin: float = 24.0
    preceding_text_overlap_min: float = 0.25


@dataclass(slots=True)
class TableContentPolicy:
    enable_supplement_writeback: bool = False
    enable_missing_content_diagnostics: bool = True
    diagnostic_min_candidates_per_table: int = 3
    diagnostic_max_candidate_ratio: float = 0.08
    diagnostic_min_text_length: int = 2
    suppress_col0_diagnostics_for_continuation: bool = True


@dataclass(slots=True)
class RuleEnginePolicy:
    enabled: bool = False
    shadow_mode: bool = True
    apply_writeback: bool = False
    emit_diagnostics: bool = True


@dataclass(slots=True)
class TableDetectionPolicy:
    supplemental_dedup_overlap_threshold: float = 0.30
    enable_two_column_guard: bool = True
    two_column_balance_tolerance: float = 0.30
    two_column_gutter_ratio_min: float = 0.06
    two_column_gutter_ratio_max: float = 0.30
    two_column_min_words: int = 120
    two_column_min_tabular_score: float = 0.72
    min_rows_for_supplemental_candidate: int = 3
    min_cols_for_supplemental_candidate: int = 2


@dataclass(slots=True)
class SamePageMergePolicy:
    gap_ratio_min: float = -0.1
    gap_ratio_max: float = 0.35
    header_similarity_threshold: float = 0.0
    respect_preceding_text_barrier: bool = True


@dataclass(slots=True)
class PdfParserSettings:
    cross_page_stitching: CrossPageStitchingThresholds
    table_content_policy: TableContentPolicy
    rule_engine_policy: RuleEnginePolicy
    table_detection_policy: TableDetectionPolicy
    same_page_merge_policy: SamePageMergePolicy


def _project_root() -> Path:
    # parsers/pdf/settings.py -> project root at parents[2]
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return _project_root() / "config" / "pdf_parser.toml"


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


@lru_cache(maxsize=1)
def get_pdf_parser_settings() -> PdfParserSettings:
    cfg_path = Path(os.environ.get("PDF_PARSER_CONFIG_PATH", str(_default_config_path())))
    thresholds = CrossPageStitchingThresholds()
    table_content_policy = TableContentPolicy()
    rule_engine_policy = RuleEnginePolicy()
    table_detection_policy = TableDetectionPolicy()
    same_page_merge_policy = SamePageMergePolicy()

    if not cfg_path.exists():
        return PdfParserSettings(
            cross_page_stitching=thresholds,
            table_content_policy=table_content_policy,
            rule_engine_policy=rule_engine_policy,
            table_detection_policy=table_detection_policy,
            same_page_merge_policy=same_page_merge_policy,
        )

    try:
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return PdfParserSettings(
            cross_page_stitching=thresholds,
            table_content_policy=table_content_policy,
            rule_engine_policy=rule_engine_policy,
            table_detection_policy=table_detection_policy,
            same_page_merge_policy=same_page_merge_policy,
        )

    section = data.get("cross_page_stitching", {}) if isinstance(data, dict) else {}
    thresholds.prev_near_bottom_ratio = _to_float(section.get("prev_near_bottom_ratio"), thresholds.prev_near_bottom_ratio)
    thresholds.curr_near_top_ratio = _to_float(section.get("curr_near_top_ratio"), thresholds.curr_near_top_ratio)
    thresholds.hint_glossary_similarity_min = _to_float(section.get("hint_glossary_similarity_min"), thresholds.hint_glossary_similarity_min)
    thresholds.default_similarity_min = _to_float(section.get("default_similarity_min"), thresholds.default_similarity_min)
    thresholds.high_similarity_link_threshold = _to_float(section.get("high_similarity_link_threshold"), thresholds.high_similarity_link_threshold)
    thresholds.title_context_similarity_threshold = _to_float(section.get("title_context_similarity_threshold"), thresholds.title_context_similarity_threshold)
    thresholds.context_overlap_min = _to_float(section.get("context_overlap_min"), thresholds.context_overlap_min)
    thresholds.overlap_min_without_hint = _to_float(section.get("overlap_min_without_hint"), thresholds.overlap_min_without_hint)
    thresholds.low_similarity_guard_threshold = _to_float(section.get("low_similarity_guard_threshold"), thresholds.low_similarity_guard_threshold)
    thresholds.low_similarity_guard_overlap_min = _to_float(section.get("low_similarity_guard_overlap_min"), thresholds.low_similarity_guard_overlap_min)
    thresholds.preceding_text_gap_max = _to_float(section.get("preceding_text_gap_max"), thresholds.preceding_text_gap_max)
    thresholds.preceding_text_min_length = _to_int(section.get("preceding_text_min_length"), thresholds.preceding_text_min_length)
    thresholds.preceding_text_ignore_top_margin = _to_float(section.get("preceding_text_ignore_top_margin"), thresholds.preceding_text_ignore_top_margin)
    thresholds.preceding_text_overlap_min = _to_float(section.get("preceding_text_overlap_min"), thresholds.preceding_text_overlap_min)

    table_section = data.get("table_content_policy", {}) if isinstance(data, dict) else {}
    table_content_policy.enable_supplement_writeback = _to_bool(
        table_section.get("enable_supplement_writeback"),
        table_content_policy.enable_supplement_writeback,
    )
    table_content_policy.enable_missing_content_diagnostics = _to_bool(
        table_section.get("enable_missing_content_diagnostics"),
        table_content_policy.enable_missing_content_diagnostics,
    )
    table_content_policy.diagnostic_min_candidates_per_table = _to_int(
        table_section.get("diagnostic_min_candidates_per_table"),
        table_content_policy.diagnostic_min_candidates_per_table,
    )
    table_content_policy.diagnostic_max_candidate_ratio = _to_float(
        table_section.get("diagnostic_max_candidate_ratio"),
        table_content_policy.diagnostic_max_candidate_ratio,
    )
    table_content_policy.diagnostic_min_text_length = _to_int(
        table_section.get("diagnostic_min_text_length"),
        table_content_policy.diagnostic_min_text_length,
    )
    table_content_policy.suppress_col0_diagnostics_for_continuation = _to_bool(
        table_section.get("suppress_col0_diagnostics_for_continuation"),
        table_content_policy.suppress_col0_diagnostics_for_continuation,
    )

    rule_engine_section = data.get("rule_engine_policy", {}) if isinstance(data, dict) else {}
    rule_engine_policy.enabled = _to_bool(
        rule_engine_section.get("enabled"),
        rule_engine_policy.enabled,
    )
    rule_engine_policy.shadow_mode = _to_bool(
        rule_engine_section.get("shadow_mode"),
        rule_engine_policy.shadow_mode,
    )
    rule_engine_policy.apply_writeback = _to_bool(
        rule_engine_section.get("apply_writeback"),
        rule_engine_policy.apply_writeback,
    )
    rule_engine_policy.emit_diagnostics = _to_bool(
        rule_engine_section.get("emit_diagnostics"),
        rule_engine_policy.emit_diagnostics,
    )

    detection_section = data.get("table_detection_policy", {}) if isinstance(data, dict) else {}
    table_detection_policy.supplemental_dedup_overlap_threshold = _to_float(
        detection_section.get("supplemental_dedup_overlap_threshold"),
        table_detection_policy.supplemental_dedup_overlap_threshold,
    )
    table_detection_policy.enable_two_column_guard = _to_bool(
        detection_section.get("enable_two_column_guard"),
        table_detection_policy.enable_two_column_guard,
    )
    table_detection_policy.two_column_balance_tolerance = _to_float(
        detection_section.get("two_column_balance_tolerance"),
        table_detection_policy.two_column_balance_tolerance,
    )
    table_detection_policy.two_column_gutter_ratio_min = _to_float(
        detection_section.get("two_column_gutter_ratio_min"),
        table_detection_policy.two_column_gutter_ratio_min,
    )
    table_detection_policy.two_column_gutter_ratio_max = _to_float(
        detection_section.get("two_column_gutter_ratio_max"),
        table_detection_policy.two_column_gutter_ratio_max,
    )
    table_detection_policy.two_column_min_words = _to_int(
        detection_section.get("two_column_min_words"),
        table_detection_policy.two_column_min_words,
    )
    table_detection_policy.two_column_min_tabular_score = _to_float(
        detection_section.get("two_column_min_tabular_score"),
        table_detection_policy.two_column_min_tabular_score,
    )
    table_detection_policy.min_rows_for_supplemental_candidate = _to_int(
        detection_section.get("min_rows_for_supplemental_candidate"),
        table_detection_policy.min_rows_for_supplemental_candidate,
    )
    table_detection_policy.min_cols_for_supplemental_candidate = _to_int(
        detection_section.get("min_cols_for_supplemental_candidate"),
        table_detection_policy.min_cols_for_supplemental_candidate,
    )

    merge_section = data.get("same_page_merge_policy", {}) if isinstance(data, dict) else {}
    same_page_merge_policy.gap_ratio_min = _to_float(
        merge_section.get("gap_ratio_min"),
        same_page_merge_policy.gap_ratio_min,
    )
    same_page_merge_policy.gap_ratio_max = _to_float(
        merge_section.get("gap_ratio_max"),
        same_page_merge_policy.gap_ratio_max,
    )
    same_page_merge_policy.header_similarity_threshold = _to_float(
        merge_section.get("header_similarity_threshold"),
        same_page_merge_policy.header_similarity_threshold,
    )
    same_page_merge_policy.respect_preceding_text_barrier = _to_bool(
        merge_section.get("respect_preceding_text_barrier"),
        same_page_merge_policy.respect_preceding_text_barrier,
    )

    return PdfParserSettings(
        cross_page_stitching=thresholds,
        table_content_policy=table_content_policy,
        rule_engine_policy=rule_engine_policy,
        table_detection_policy=table_detection_policy,
        same_page_merge_policy=same_page_merge_policy,
    )
