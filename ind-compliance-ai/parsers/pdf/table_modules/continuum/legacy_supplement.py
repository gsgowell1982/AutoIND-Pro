# Version: v1.0.0
# Optimization Summary:
# - Extract legacy supplement writeback logic from normalization module.
# - Preserve original algorithm and behavior for backward compatibility.
# - Isolate optional mutation path for enterprise auditability.

from __future__ import annotations

from typing import Any


def supplement_missing_content(
    rows: list[Any],
    grid: list[list[str | None]],
    raw_evidence: Any,
    logical_col_count: int,
    parent_bbox: tuple[float, float, float, float] | None = None,
) -> int:
    """Legacy supplement writeback path.

    This keeps the original non-empty-cell-only writeback behavior and is
    intentionally invoked only when policy `enable_supplement_writeback=true`.
    """
    if not raw_evidence.spans:
        return 0

    supplemented_count = 0

    table_bbox = raw_evidence.bbox
    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]
    if table_width <= 0 or table_height <= 0:
        return 0

    if parent_bbox and len(parent_bbox) == 4:
        reference_x0 = parent_bbox[0]
        reference_x1 = parent_bbox[2]
    else:
        reference_x0 = table_bbox[0]
        reference_x1 = table_bbox[2]
    reference_width = reference_x1 - reference_x0

    all_spans = raw_evidence.spans
    median_height = sorted([s.height for s in all_spans])[len(all_spans) // 2] if all_spans else 8.0
    x_gap_threshold = table_width / logical_col_count * 0.25

    def y_overlap_ratio(s1: Any, s2: Any) -> float:
        y0_max = max(s1.y0, s2.y0)
        y1_min = min(s1.y1, s2.y1)
        if y1_min <= y0_max:
            return 0.0
        return (y1_min - y0_max) / min(s1.height, s2.height)

    y_tolerance = median_height * 0.5
    y_groups: dict[int, list[Any]] = {}
    for span in all_spans:
        y_key = int(span.y_center / y_tolerance)
        y_groups.setdefault(y_key, []).append(span)

    merged_cells: list[dict[str, Any]] = []
    for group in y_groups.values():
        group_sorted = sorted(group, key=lambda s: s.x0)
        clusters: list[list[Any]] = []

        for span in group_sorted:
            merged = False
            for cluster in clusters:
                for existing in cluster:
                    y_overlap = y_overlap_ratio(span, existing)
                    x_gap = span.x0 - existing.x1
                    size_diff = abs(span.size - existing.size)
                    if y_overlap > 0.7 and x_gap < x_gap_threshold and size_diff < 2:
                        cluster.append(span)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                clusters.append([span])

        for cluster in clusters:
            cluster_sorted = sorted(cluster, key=lambda s: s.x0)
            text = " ".join(s.text.strip() for s in cluster_sorted)
            x0 = min(s.x0 for s in cluster)
            x1 = max(s.x1 for s in cluster)
            y0 = min(s.y0 for s in cluster)
            y1 = max(s.y1 for s in cluster)
            merged_cells.append(
                {
                    "text": text,
                    "x_center": (x0 + x1) / 2,
                    "y_center": (y0 + y1) / 2,
                    "y0": y0,
                    "y1": y1,
                }
            )

    cell_rows: list[list[dict[str, Any]]] = []
    for cell in sorted(merged_cells, key=lambda c: c["y_center"]):
        found = False
        for row in cell_rows:
            row_y0 = min(c["y0"] for c in row)
            row_y1 = max(c["y1"] for c in row)
            cell_h = cell["y1"] - cell["y0"]
            if cell_h > 0:
                overlap = (min(row_y1, cell["y1"]) - max(row_y0, cell["y0"])) / cell_h
                if overlap > 0.5:
                    row.append(cell)
                    found = True
                    break
        if not found:
            cell_rows.append([cell])

    for row_cells in cell_rows:
        row_y_center = sum(c["y_center"] for c in row_cells) / len(row_cells)
        logical_row = None
        for row_idx in range(len(rows)):
            row_y0 = table_bbox[1] + table_height * row_idx / len(rows)
            row_y1 = table_bbox[1] + table_height * (row_idx + 1) / len(rows)
            row_center = (row_y0 + row_y1) / 2
            if abs(row_y_center - row_center) < table_height / len(rows) * 0.6:
                logical_row = row_idx
                break

        if logical_row is None or logical_row >= len(grid):
            continue

        for cell in row_cells:
            rel_x = (cell["x_center"] - reference_x0) / reference_width
            logical_col = max(0, min(int(rel_x * logical_col_count), logical_col_count - 1))
            cell_text = cell["text"].strip()
            if not cell_text:
                continue
            if grid[logical_row][logical_col] is not None:
                continue

            grid[logical_row][logical_col] = cell_text
            for norm_cell in rows[logical_row].cells:
                if norm_cell.logical_col == logical_col:
                    norm_cell.text = cell_text
                    norm_cell.supplemented = True
                    break
            supplemented_count += 1

    return supplemented_count


__all__ = ["supplement_missing_content"]
