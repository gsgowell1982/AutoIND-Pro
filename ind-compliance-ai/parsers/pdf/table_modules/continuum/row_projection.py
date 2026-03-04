# Version: v1.0.0
# Optimization Summary:
# - Extract row/cell projection and column cluster construction from normalization layer.
# - Keep algorithms unchanged while reducing normalization module size.
# - Support typed object construction through injected constructors.

from __future__ import annotations

from typing import Any, Callable


def build_column_clusters(
    column_mapping: list[list[int]],
    table_bbox: tuple[float, float, float, float],
    physical_col_count: int,
    cluster_cls: Callable[..., Any],
) -> list[Any]:
    clusters = []
    table_width = table_bbox[2] - table_bbox[0]

    for logical_col, physical_cols in enumerate(column_mapping):
        if not physical_cols:
            continue
        avg_physical = sum(physical_cols) / len(physical_cols)
        center_x = table_bbox[0] + table_width * (avg_physical + 0.5) / physical_col_count
        clusters.append(
            cluster_cls(
                logical_col=logical_col,
                physical_cols=physical_cols,
                center_x=center_x,
            )
        )
    return clusters


def normalize_rows_and_cells(
    raw_evidence: Any,
    column_mapping: list[list[int]],
    logical_col_count: int,
    cell_cls: Callable[..., Any],
    row_cls: Callable[..., Any],
) -> tuple[list[Any], list[list[str | None]]]:
    physical_to_logical: dict[int, int] = {}
    for logical_col, physical_cols in enumerate(column_mapping):
        for physical_col in physical_cols:
            physical_to_logical[physical_col] = logical_col

    rows: list[Any] = []
    grid: list[list[str | None]] = []

    for raw_row in raw_evidence.rows:
        normalized_cells: list[Any] = [None] * logical_col_count
        grid_row: list[str | None] = [None] * logical_col_count
        cell_texts: dict[int, list[str]] = {i: [] for i in range(logical_col_count)}

        for raw_cell in raw_row.cells:
            physical_col = raw_cell.physical_col
            logical_col = physical_to_logical.get(physical_col, physical_col)
            if raw_cell.text:
                cell_texts[logical_col].append(raw_cell.text)

        for logical_col in range(logical_col_count):
            physical_cols = column_mapping[logical_col] if logical_col < len(column_mapping) else []
            text = " ".join(cell_texts[logical_col]) if cell_texts[logical_col] else None
            cell = cell_cls(
                logical_row=raw_row.physical_row,
                logical_col=logical_col,
                physical_row=raw_row.physical_row,
                physical_col_start=physical_cols[0] if physical_cols else logical_col,
                physical_col_end=physical_cols[-1] if physical_cols else logical_col,
                physical_colspan=len(physical_cols),
                text=text,
            )
            normalized_cells[logical_col] = cell
            grid_row[logical_col] = text

        rows.append(
            row_cls(
                logical_row=raw_row.physical_row,
                physical_row=raw_row.physical_row,
                cells=[c for c in normalized_cells if c is not None],
            )
        )
        grid.append(grid_row)

    return rows, grid


__all__ = [
    "build_column_clusters",
    "normalize_rows_and_cells",
]
