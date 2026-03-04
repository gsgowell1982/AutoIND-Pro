# Version: v1.0.0
# Optimization Summary:
# - Extract column inference and column mapping logic from normalization layer.
# - Preserve existing algorithms and return values to keep parser output stable.
# - Provide reusable layout utilities for continuum-oriented rule evolution.

from __future__ import annotations

from typing import Any


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
            logical_col = max(1, min(logical_col, logical_col_count - 1))
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
