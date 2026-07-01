# Version: v1.0.1
# Optimization Summary:
# - Extract column inference and column mapping logic from normalization layer.
# - Preserve existing algorithms and return values to keep parser output stable.
# - Provide reusable layout utilities for continuum-oriented rule evolution.
#
# v1.0.1 (2026-03-12):
# - Fix column mapping bug in build_column_mapping_with_parent_bbox().
# - Change max(1, ...) to max(0, ...) to allow mapping to first column.
# - This fixes the issue where continuation table's first column was always empty.

from __future__ import annotations

from typing import Any


def _textual_cell_length(cell: Any) -> int:
    return len(str(getattr(cell, "text", "") or "").strip())


def _select_dense_content_anchor_pattern(
    *,
    row_patterns: list[tuple[int, ...]],
    physical_col_count: int,
    rows: list[Any] | None = None,
) -> tuple[int, ...] | None:
    if not row_patterns or physical_col_count <= 0:
        return None

    max_cols_in_any_row = max((len(pattern) for pattern in row_patterns if pattern), default=0)
    if max_cols_in_any_row < 4 or physical_col_count <= max_cols_in_any_row:
        return None

    dense_candidates: list[tuple[float, int, tuple[int, ...]]] = []
    evidence_rows = rows or []
    for row_index, pattern in enumerate(row_patterns):
        if len(pattern) != max_cols_in_any_row:
            continue
        if len(set(pattern)) != len(pattern):
            continue
        if any(col < 0 or col >= physical_col_count for col in pattern):
            continue

        content_length = 0
        if row_index < len(evidence_rows):
            for cell in getattr(evidence_rows[row_index], "cells", []) or []:
                if getattr(cell, "physical_col", None) in pattern:
                    content_length += _textual_cell_length(cell)

        frequency = sum(1 for other in row_patterns if tuple(other) == tuple(pattern))
        later_row_weight = row_index / max(1, len(row_patterns) - 1)
        score = frequency * 1000 + content_length + later_row_weight
        dense_candidates.append((score, row_index, tuple(pattern)))

    if not dense_candidates:
        return None

    dense_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return dense_candidates[0][2]


def _build_mapping_from_anchor_pattern(
    physical_col_count: int,
    logical_col_count: int,
    anchor_pattern: tuple[int, ...] | None,
) -> list[list[int]] | None:
    if not anchor_pattern or len(anchor_pattern) != logical_col_count:
        return None
    if physical_col_count <= 0 or logical_col_count <= 0:
        return None

    anchors = list(anchor_pattern)
    mapping: list[list[int]] = [[anchor] for anchor in anchors]
    anchor_to_logical = {anchor: idx for idx, anchor in enumerate(anchors)}

    for physical_col in range(physical_col_count):
        if physical_col in anchor_to_logical:
            continue
        nearest_logical = min(
            range(logical_col_count),
            key=lambda idx: (abs(physical_col - anchors[idx]), idx),
        )
        mapping[nearest_logical].append(physical_col)

    for cols in mapping:
        cols.sort()
    return mapping


def infer_logical_column_count(
    raw_evidence: Any,
    parent_col_count: int | None,
    grid_score: float,
) -> tuple[int, str]:
    _ = grid_score
    physical_col_count = raw_evidence.physical_col_count
    row_patterns = raw_evidence.row_patterns

    if not row_patterns:
        return max(physical_col_count, 1), "fallback"

    cols_per_row = [len(pattern) for pattern in row_patterns if pattern]
    if not cols_per_row:
        return max(physical_col_count, 1), "fallback"

    max_cols_in_any_row = max(cols_per_row)

    if parent_col_count is not None and parent_col_count > 0:
        return parent_col_count, "inherit"

    if str(getattr(raw_evidence, "source", "") or "") == "text_aligned_borderless_grid":
        return max(physical_col_count, 1), "text_aligned_physical_columns"

    anchor_pattern = _select_dense_content_anchor_pattern(
        row_patterns=row_patterns,
        physical_col_count=physical_col_count,
        rows=list(getattr(raw_evidence, "rows", []) or []),
    )
    if anchor_pattern:
        return len(anchor_pattern), "content_anchor_columns"

    all_content_cols: set[int] = set()
    for pattern in row_patterns:
        all_content_cols.update(pattern)

    if not all_content_cols:
        return max(physical_col_count, 1), "fallback"

    sorted_content_cols = sorted(all_content_cols)
    clusters: list[list[int]] = []
    current_cluster: list[int] = []

    for col_idx in sorted_content_cols:
        if current_cluster and col_idx > current_cluster[-1] + 1:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(col_idx)
    if current_cluster:
        clusters.append(current_cluster)

    num_clusters = len(clusters)
    if num_clusters == max_cols_in_any_row:
        return num_clusters, "content_clusters"
    if num_clusters == 1 and max_cols_in_any_row > 1:
        return max_cols_in_any_row, "max_content_cols"
    if num_clusters >= 2:
        return num_clusters, "content_clusters"
    return max(max_cols_in_any_row, 1), "max_content_cols"


def analyze_content_distribution(
    raw_evidence: Any,
) -> dict[str, Any]:
    row_patterns = raw_evidence.row_patterns
    physical_col_count = raw_evidence.physical_col_count

    if not row_patterns:
        return {"logical_col_count": physical_col_count, "confidence": 0.0}

    col_frequency: dict[int, int] = {}
    for pattern in row_patterns:
        for col in pattern:
            col_frequency[col] = col_frequency.get(col, 0) + 1

    if not col_frequency:
        return {"logical_col_count": physical_col_count, "confidence": 0.0}

    sorted_cols = sorted(col_frequency.keys())
    clusters = [[sorted_cols[0]]]
    for col in sorted_cols[1:]:
        prev_col = clusters[-1][-1]
        co_occurrence = sum(
            1 for pattern in row_patterns if col in pattern and prev_col in pattern
        )
        solo_occurrence = sum(
            1 for pattern in row_patterns if col in pattern or prev_col in pattern
        )
        if solo_occurrence > 0 and co_occurrence / solo_occurrence >= 0.5:
            clusters[-1].append(col)
        else:
            clusters.append([col])

    logical_col_count = len(clusters)
    confidence = min(1.0, len(row_patterns) / 5.0)
    return {
        "logical_col_count": logical_col_count,
        "confidence": confidence,
        "clusters": clusters,
    }


def analyze_column_clustering(
    row_patterns: list[tuple[int, ...]],
    physical_col_count: int,
) -> dict[str, Any]:
    if not row_patterns:
        return {"logical_col_count": physical_col_count, "confidence": 0.0}

    all_non_empty_cols = set()
    for pattern in row_patterns:
        all_non_empty_cols.update(pattern)
    if not all_non_empty_cols:
        return {"logical_col_count": physical_col_count, "confidence": 0.0}

    sorted_cols = sorted(all_non_empty_cols)
    clusters = [[sorted_cols[0]]]
    for col in sorted_cols[1:]:
        prev_col = clusters[-1][-1]
        if col - prev_col <= 1:
            clusters[-1].append(col)
        else:
            clusters.append([col])

    logical_col_count = len(clusters)
    max_cols_per_row = max(len(p) for p in row_patterns) if row_patterns else 1
    confidence = min(1.0, logical_col_count / max(1, max_cols_per_row))
    return {
        "logical_col_count": logical_col_count,
        "confidence": confidence,
        "clusters": clusters,
    }


def build_column_mapping(
    physical_col_count: int,
    logical_col_count: int,
    row_patterns: list[tuple[int, ...]] | None = None,
    table_bbox: tuple[float, float, float, float] | None = None,
    parent_bbox: tuple[float, float, float, float] | None = None,
) -> list[list[int]]:
    if logical_col_count <= 0 or physical_col_count <= 0:
        return []

    if parent_bbox and table_bbox:
        return build_column_mapping_with_parent_bbox(
            physical_col_count,
            logical_col_count,
            table_bbox,
            parent_bbox,
        )

    if not row_patterns:
        mapping: list[list[int]] = [[] for _ in range(logical_col_count)]
        for physical_col in range(physical_col_count):
            rel_pos = (physical_col + 0.5) / physical_col_count
            logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
            mapping[logical_col].append(physical_col)
        return mapping

    all_content_cols: set[int] = set()
    for pattern in row_patterns:
        all_content_cols.update(pattern)

    if not all_content_cols:
        mapping = [[] for _ in range(logical_col_count)]
        for physical_col in range(physical_col_count):
            rel_pos = (physical_col + 0.5) / physical_col_count
            logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
            mapping[logical_col].append(physical_col)
        return mapping

    anchor_mapping = _build_mapping_from_anchor_pattern(
        physical_col_count,
        logical_col_count,
        _select_dense_content_anchor_pattern(
            row_patterns=list(row_patterns),
            physical_col_count=physical_col_count,
        ),
    )
    if anchor_mapping is not None:
        return anchor_mapping

    sorted_content_cols = sorted(all_content_cols)
    clusters: list[list[int]] = []
    current_cluster: list[int] = []
    for col_idx in sorted_content_cols:
        if current_cluster and col_idx > current_cluster[-1] + 1:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(col_idx)
    if current_cluster:
        clusters.append(current_cluster)

    if len(clusters) == logical_col_count:
        mapping = [[] for _ in range(logical_col_count)]
        for logical_col, cluster in enumerate(clusters):
            mapping[logical_col] = list(cluster)

        all_mapped = set()
        for cluster in clusters:
            all_mapped.update(cluster)

        for physical_col in range(physical_col_count):
            if physical_col in all_mapped:
                continue
            min_dist = float("inf")
            nearest_logical = 0
            for logical_col, cluster in enumerate(clusters):
                for cluster_col in cluster:
                    dist = abs(physical_col - cluster_col)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_logical = logical_col
            mapping[nearest_logical].append(physical_col)

        for logical_col in range(logical_col_count):
            mapping[logical_col].sort()
        return mapping

    mapping = [[] for _ in range(logical_col_count)]
    for physical_col in range(physical_col_count):
        rel_pos = (physical_col + 0.5) / physical_col_count
        logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
        mapping[logical_col].append(physical_col)
    return mapping


def build_column_mapping_with_parent_bbox(
    physical_col_count: int,
    logical_col_count: int,
    table_bbox: tuple[float, float, float, float],
    parent_bbox: tuple[float, float, float, float],
) -> list[list[int]]:
    mapping: list[list[int]] = [[] for _ in range(logical_col_count)]
    table_width = table_bbox[2] - table_bbox[0]
    parent_width = parent_bbox[2] - parent_bbox[0]

    if table_width <= 0 or parent_width <= 0:
        for physical_col in range(physical_col_count):
            rel_pos = (physical_col + 0.5) / physical_col_count
            logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
            mapping[logical_col].append(physical_col)
        return mapping

    x_offset = table_bbox[0] - parent_bbox[0]
    first_col_width = parent_width / logical_col_count
    first_col_truncated = x_offset > first_col_width * 0.2

    if first_col_truncated:
        for physical_col in range(physical_col_count):
            col_x_start = table_bbox[0] + physical_col * table_width / physical_col_count
            col_x_end = table_bbox[0] + (physical_col + 1) * table_width / physical_col_count
            col_x_center = (col_x_start + col_x_end) / 2
            rel_x = (col_x_center - parent_bbox[0]) / parent_width
            logical_col = int(rel_x * logical_col_count)
            # v1.0.1: Changed max(1, ...) to max(0, ...) to allow mapping to first column
            # Previously, max(1, ...) forced skipping the first column when first_col_truncated=True,
            # causing mapping[0] = [] and resulting in empty first column in the grid.
            logical_col = max(0, min(logical_col, logical_col_count - 1))
            mapping[logical_col].append(physical_col)
    else:
        parent_col_boundaries = []
        for i in range(logical_col_count + 1):
            boundary = parent_bbox[0] + i * parent_width / logical_col_count
            parent_col_boundaries.append(boundary)

        for physical_col in range(physical_col_count):
            col_x_start = table_bbox[0] + physical_col * table_width / physical_col_count
            col_x_end = table_bbox[0] + (physical_col + 1) * table_width / physical_col_count
            col_x_center = (col_x_start + col_x_end) / 2
            logical_col = 0
            for i in range(logical_col_count):
                if parent_col_boundaries[i] <= col_x_center < parent_col_boundaries[i + 1]:
                    logical_col = i
                    break
                if col_x_center >= parent_col_boundaries[-1]:
                    logical_col = logical_col_count - 1
            mapping[logical_col].append(physical_col)

    return mapping


__all__ = [
    "infer_logical_column_count",
    "analyze_content_distribution",
    "analyze_column_clustering",
    "build_column_mapping",
    "build_column_mapping_with_parent_bbox",
]
