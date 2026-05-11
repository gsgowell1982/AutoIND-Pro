"""Stable table-structure contracts for parser regression auditing.

This module intentionally does not repair or reinterpret tables. It summarizes
already-built table ASTs into compact, deterministic contracts so parser changes
can be reviewed for shape drift before user-facing regressions appear.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


GRID_KEYS = ("raw_grid", "display_grid", "data_grid", "grid")


def summarize_document_tables(parsed_document: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic table-structure summary for one parsed document."""

    tables = [
        summarize_table(table)
        for table in list(parsed_document.get("table_asts") or [])
        if isinstance(table, dict)
    ]
    return {
        "table_count": len(tables),
        "continued_table_count": sum(1 for table in tables if table.get("continued_from")),
        "review_required_table_count": sum(1 for table in tables if table.get("review_required")),
        "tables": tables,
    }


def summarize_table(table: dict[str, Any]) -> dict[str, Any]:
    """Summarize one table AST without preserving full cell text."""

    views = {
        key: _summarize_grid(table.get(key))
        for key in GRID_KEYS
        if isinstance(table.get(key), list)
    }
    return {
        "table_id": str(table.get("table_id") or ""),
        "page": _int_or_none(table.get("page")),
        "bbox": _rounded_bbox(table.get("bbox")),
        "title_present": bool(str(table.get("title") or "").strip()),
        "col_count": _int_or_none(table.get("col_count")),
        "logical_col_count": _int_or_none(table.get("logical_col_count")),
        "physical_col_count": _int_or_none(table.get("physical_col_count")),
        "row_count": _int_or_none(table.get("row_count")),
        "raw_row_count": _int_or_none(table.get("raw_row_count")),
        "display_row_count": _int_or_none(table.get("display_row_count")),
        "data_row_count": _int_or_none(table.get("data_row_count")),
        "data_start_row": _int_or_none(table.get("data_start_row")),
        "header_row_index": _int_or_none(table.get("header_row_index")),
        "continued_from": _string_or_none(table.get("continued_from")),
        "continued_to": _string_list(table.get("continued_to")),
        "header_inherited": bool(table.get("header_inherited")),
        "review_required": bool(table.get("review_required")),
        "review_reasons": [str(item) for item in table.get("review_reasons") or []],
        "risk_flags": [str(item) for item in table.get("risk_flags") or []],
        "views": views,
        "fingerprints": {
            key: _fingerprint_grid(table.get(key))
            for key in GRID_KEYS
            if isinstance(table.get(key), list)
        },
    }


def diff_table_contracts(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare two table contract summaries by stable table id."""

    before_tables = _tables_by_id(before)
    after_tables = _tables_by_id(after)
    before_ids = set(before_tables)
    after_ids = set(after_tables)
    added = sorted(after_ids - before_ids)
    removed = sorted(before_ids - after_ids)

    changed = []
    for table_id in sorted(before_ids & after_ids):
        changed_fields = _changed_fields(before_tables[table_id], after_tables[table_id])
        if changed_fields:
            changed.append({"table_id": table_id, "changed_fields": changed_fields})

    return {
        "matches": not added and not removed and not changed,
        "table_count_before": int(before.get("table_count", len(before_tables)) or 0),
        "table_count_after": int(after.get("table_count", len(after_tables)) or 0),
        "added_tables": added,
        "removed_tables": removed,
        "changed_tables": changed,
    }


def _summarize_grid(grid: object) -> dict[str, Any]:
    rows = [list(row) if isinstance(row, list) else [] for row in list(grid or [])]
    widths = [len(row) for row in rows]
    max_width = max(widths, default=0)
    non_empty_by_column = [0] * max_width
    empty_by_column = [0] * max_width
    non_empty_cell_count = 0

    for row in rows:
        for col_idx in range(max_width):
            value = row[col_idx] if col_idx < len(row) else None
            if _has_text(value):
                non_empty_by_column[col_idx] += 1
                non_empty_cell_count += 1
            else:
                empty_by_column[col_idx] += 1

    return {
        "row_count": len(rows),
        "max_col_count": max_width,
        "row_widths": widths,
        "non_empty_cell_count": non_empty_cell_count,
        "non_empty_cells_by_column": non_empty_by_column,
        "empty_cells_by_column": empty_by_column,
    }


def _fingerprint_grid(grid: object) -> str:
    rows = [list(row) if isinstance(row, list) else [] for row in list(grid or [])]
    normalized = [
        [
            {
                "empty": not _has_text(value),
                "length_bucket": _text_length_bucket(value),
                "line_count": _line_count(value),
            }
            for value in row
        ]
        for row in rows
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for key in (
        "page",
        "bbox",
        "title_present",
        "col_count",
        "logical_col_count",
        "physical_col_count",
        "row_count",
        "raw_row_count",
        "display_row_count",
        "data_row_count",
        "data_start_row",
        "header_row_index",
        "continued_from",
        "continued_to",
        "header_inherited",
        "review_required",
        "risk_flags",
    ):
        if before.get(key) != after.get(key):
            fields.append(key)

    for grid_key in sorted(set(before.get("views", {})) | set(after.get("views", {}))):
        before_view = (before.get("views") or {}).get(grid_key, {})
        after_view = (after.get("views") or {}).get(grid_key, {})
        for view_key in (
            "row_count",
            "max_col_count",
            "row_widths",
            "non_empty_cell_count",
            "non_empty_cells_by_column",
            "empty_cells_by_column",
        ):
            if before_view.get(view_key) != after_view.get(view_key):
                fields.append(f"views.{grid_key}.{view_key}")

    for grid_key in sorted(set(before.get("fingerprints", {})) | set(after.get("fingerprints", {}))):
        before_fp = (before.get("fingerprints") or {}).get(grid_key)
        after_fp = (after.get("fingerprints") or {}).get(grid_key)
        if before_fp != after_fp:
            fields.append(f"fingerprints.{grid_key}")

    return fields


def _tables_by_id(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for table in list(contract.get("tables") or []):
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("table_id") or "").strip()
        if table_id:
            result[table_id] = table
    return result


def _rounded_bbox(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [round(float(item), 2) for item in value]
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _has_text(value: object) -> bool:
    return bool(str(value or "").strip())


def _line_count(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    return text.count("\n") + 1


def _text_length_bucket(value: object) -> int:
    length = len(str(value or "").strip())
    if length <= 0:
        return 0
    if length <= 4:
        return 1
    if length <= 12:
        return 2
    if length <= 32:
        return 3
    if length <= 80:
        return 4
    return 5
