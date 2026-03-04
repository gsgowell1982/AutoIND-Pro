# Version: v1.0.1
# Optimization Summary:
# - Migrate semantic repair logic out of normalization for maintainability.
# - Keep rules evidence-driven and non-destructive (fill-empty / merge-broken only).
# - Provide reusable functions for rule-engine apply and shadow execution.

from __future__ import annotations

import re
from typing import Any


def recover_key_identifier_cells(
    rows: list[Any],
    grid: list[list[str | None]],
    raw_evidence: Any,
    logical_col_count: int,
) -> int:
    if not rows or logical_col_count <= 0:
        return 0

    table_bbox = raw_evidence.bbox
    table_width = table_bbox[2] - table_bbox[0]
    if table_width <= 0:
        return 0

    app_number_pattern = re.compile(r"^[xX]\d{6,}$")
    sequence_pattern = re.compile(r"^\d{4}$")
    short_sequence_pattern = re.compile(r"^\d{2}$")
    module_pattern = re.compile(r"^[mM]\d{1,2}$")
    region_codes = {"cn", "us", "eu", "jp", "kr", "au", "ca", "uk"}
    folder_keys = {"dtd", "util", "style"}

    first_col_right = table_bbox[0] + table_width / max(1, logical_col_count)
    recovered = 0

    def classify_key_token(text: str) -> str | None:
        token = text.strip()
        if app_number_pattern.fullmatch(token):
            return "application_id"
        if sequence_pattern.fullmatch(token):
            return "sequence_code"
        if short_sequence_pattern.fullmatch(token):
            return "short_sequence_code"
        if module_pattern.fullmatch(token):
            return "module_code"
        if token.lower() in region_codes:
            return "region_code"
        if token.lower() in folder_keys:
            return "folder_key"
        return None

    def find_row_index_by_span_y(y_center: float) -> int | None:
        for idx, raw_row in enumerate(raw_evidence.rows):
            if raw_row.bbox and raw_row.y0 <= y_center <= raw_row.y1:
                return idx
        table_height = raw_evidence.bbox[3] - raw_evidence.bbox[1]
        if table_height <= 0:
            return None
        rel = (y_center - raw_evidence.bbox[1]) / table_height
        guess = int(rel * max(1, len(rows)))
        if 0 <= guess < len(rows):
            return guess
        return None

    evidence_candidates: list[tuple[str, float, float, str]] = []
    for span in raw_evidence.spans:
        evidence_candidates.append(((span.text or "").strip(), span.x_center, span.y_center, "span"))
    for word in getattr(raw_evidence, "words", []):
        evidence_candidates.append(((word.text or "").strip(), word.x_center, word.y_center, "word"))

    for span_text, x_center, y_center, source_kind in evidence_candidates:
        if not span_text or " " in span_text:
            continue
        token_type = classify_key_token(span_text)
        if token_type is None:
            continue
        if x_center > first_col_right:
            continue

        row_idx = find_row_index_by_span_y(y_center)
        if row_idx is None or row_idx >= len(rows):
            continue
        if grid[row_idx][0] is not None:
            continue
        if logical_col_count > 1:
            mid_text = str(grid[row_idx][1] or "").strip()
            if mid_text and mid_text == span_text:
                continue
        if logical_col_count > 1 and all(grid[row_idx][c] is None for c in range(1, logical_col_count)):
            continue

        raw_row = raw_evidence.rows[row_idx] if row_idx < len(raw_evidence.rows) else None
        if raw_row is None:
            continue

        row_col0 = raw_row.cells[0] if raw_row.cells else None
        bbox_shifted = False
        if row_col0 and row_col0.bbox:
            bbox_shifted = row_col0.bbox[0] > (table_bbox[0] + table_width * 0.35)
        else:
            bbox_shifted = True
        if not bbox_shifted:
            continue

        row_context = " ".join(str(grid[row_idx][c] or "") for c in range(1, logical_col_count)).lower()
        if token_type == "sequence_code":
            if ("序列" not in row_context) and ("sequence" not in row_context):
                continue
        elif token_type == "short_sequence_code":
            if ("1." not in row_context) and ("章节" not in row_context):
                continue
        elif token_type == "region_code":
            if ("regional" not in row_context) and ("区域" not in row_context) and ("region" not in row_context):
                continue
        elif token_type == "folder_key":
            if ("文件夹" not in row_context) and ("folder" not in row_context):
                continue

        grid[row_idx][0] = span_text
        for norm_cell in rows[row_idx].cells:
            if norm_cell.logical_col == 0:
                norm_cell.text = span_text
                norm_cell.supplemented = True
                norm_cell.supplement_reason = f"key_identifier_span_fallback:{token_type}:{source_kind}"
                break
        recovered += 1

    return recovered


def repair_directory_listing_structure(
    rows: list[Any],
    grid: list[list[str | None]],
    raw_evidence: Any,
    logical_col_count: int,
) -> int:
    if logical_col_count != 3 or not rows or not grid:
        return 0
    if raw_evidence.physical_col_count < 5:
        return 0

    desc_hits = 0
    for row in grid:
        desc = (row[2] or "") if len(row) > 2 else ""
        if "文件夹" in desc or "folder" in desc.lower():
            desc_hits += 1
    if desc_hits < 6:
        return 0

    repaired = 0

    def _set_cell(row_idx: int, col_idx: int, text: str, reason: str) -> bool:
        if not (0 <= row_idx < len(grid) and 0 <= col_idx < logical_col_count):
            return False
        text = text.strip()
        if not text:
            return False
        current = grid[row_idx][col_idx]
        if current is not None and str(current).strip():
            return False
        grid[row_idx][col_idx] = text
        for norm_cell in rows[row_idx].cells:
            if norm_cell.logical_col == col_idx:
                norm_cell.text = text
                norm_cell.supplemented = True
                norm_cell.supplement_reason = reason
                break
        return True

    def _smart_join_tokens(items: list[tuple[float, float, str]]) -> str:
        if not items:
            return ""
        items = sorted(items, key=lambda it: (it[0], it[1]))
        out = ""
        prev_x1 = None
        cjk_re = re.compile(r"[\u4e00-\u9fff]")
        for x0, x1, token in items:
            token = token.strip()
            if not token:
                continue
            if not out:
                out = token
                prev_x1 = x1
                continue
            gap = (x0 - prev_x1) if prev_x1 is not None else 0.0
            need_space = gap > 1.5
            if not need_space:
                last_ch = out[-1]
                first_ch = token[0]
                if cjk_re.match(last_ch) and re.match(r"[0-9A-Za-z]", first_ch):
                    need_space = True
                elif re.match(r"[0-9A-Za-z]", last_ch) and cjk_re.match(first_ch):
                    need_space = True
            out = f"{out} {token}" if need_space else f"{out}{token}"
            prev_x1 = x1
        return out.strip()

    table_x0, _, table_x1, _ = raw_evidence.bbox
    table_w = max(1e-6, table_x1 - table_x0)
    right_band_x = table_x0 + table_w * 0.45

    def _rebuild_row_desc_text(raw_row: Any) -> str:
        if not raw_row.bbox:
            return ""
        y0, y1 = raw_row.y0, raw_row.y1
        row_spans = [
            s for s in raw_evidence.spans
            if (y0 <= s.y_center <= y1) and (s.x_center >= right_band_x) and str(s.text or "").strip()
        ]
        if row_spans:
            span_items = [(float(s.x0), float(s.x1), str(s.text).strip()) for s in row_spans]
            text = _smart_join_tokens(span_items)
            if text:
                return text
        row_words = [
            w for w in getattr(raw_evidence, "words", [])
            if (y0 <= w.y_center <= y1) and (w.x_center >= right_band_x) and str(w.text or "").strip()
        ]
        if row_words:
            word_items = [(float(w.x0), float(w.x1), str(w.text).strip()) for w in row_words]
            return _smart_join_tokens(word_items)
        return ""

    for idx, raw_row in enumerate(raw_evidence.rows):
        if idx >= len(grid):
            break
        if grid[idx][2] is not None:
            continue
        mid_code = str(grid[idx][1] or "").strip()
        if not re.fullmatch(r"\d{2}", mid_code):
            continue
        text = _rebuild_row_desc_text(raw_row)
        if not text:
            continue
        if _set_cell(idx, 2, text, "directory_row_desc_from_evidence"):
            repaired += 1

    for idx, row in enumerate(grid):
        head = row[0] if len(row) > 0 else None
        if not head or "\n" not in str(head):
            continue
        lines = [ln.strip() for ln in str(head).splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        leader = lines[0].lower()
        if leader not in {"dtd", "util", "style"}:
            continue

        if grid[idx][0] != lines[0]:
            grid[idx][0] = lines[0]
            for norm_cell in rows[idx].cells:
                if norm_cell.logical_col == 0:
                    norm_cell.text = lines[0]
                    norm_cell.supplemented = True
                    norm_cell.supplement_reason = "directory_multiline_head_split"
                    break
            repaired += 1

        existing_col2 = {
            str(grid[k][1]).strip().lower()
            for k in range(idx + 1, len(grid))
            if len(grid[k]) > 1 and grid[k][1] and str(grid[k][1]).strip()
        }
        targets = lines[1:]
        cursor = idx + 1
        for item in targets:
            if item.strip().lower() in existing_col2:
                continue
            while cursor < len(grid):
                desc = grid[cursor][2] if len(grid[cursor]) > 2 else None
                row_empty = all((c is None) or (str(c).strip() == "") for c in grid[cursor])
                if (desc is not None and str(desc).strip()) and not row_empty:
                    break
                cursor += 1
            if cursor >= len(grid):
                break
            if _set_cell(cursor, 1, item, "directory_multiline_tail_split"):
                repaired += 1
            cursor += 1

    return repaired


def detect_filename_semantic_columns(
    grid: list[list[str | None]],
    logical_col_count: int,
) -> set[int]:
    if logical_col_count <= 0:
        return set()

    ext_pat = re.compile(r"\.(?:xml|xsl|xsd|dtd|txt|pdf|csv|zip)$", re.IGNORECASE)
    token_pat = re.compile(r"^[A-Za-z0-9._-]+$")
    candidates: set[int] = set()

    for col in range(logical_col_count):
        non_empty = 0
        score = 0
        for row in grid:
            if col >= len(row):
                continue
            text = str(row[col] or "").strip()
            if not text:
                continue
            non_empty += 1
            parts = [p for p in re.split(r"\s+", text) if p]
            if any(ext_pat.search(p) for p in parts):
                score += 2
            elif len(parts) == 1 and token_pat.fullmatch(parts[0]):
                score += 1
        if non_empty >= 3 and score / max(1, non_empty) >= 0.9:
            candidates.add(col)

    return candidates


def merge_filename_continuations(
    rows: list[Any],
    grid: list[list[str | None]],
    logical_col_count: int,
) -> int:
    if logical_col_count <= 0 or len(grid) < 2:
        return 0

    filename_cols = detect_filename_semantic_columns(grid, logical_col_count)
    if not filename_cols:
        return 0

    prefix_pat = re.compile(r"^[A-Za-z0-9._-]+-$")
    suffix_pat = re.compile(r"^[A-Za-z0-9._-]+\.(?:xml|xsl|xsd|dtd|txt|pdf|csv|zip)$", re.IGNORECASE)
    merged_count = 0

    def row_other_cols_empty(row_idx: int, target_col: int) -> bool:
        for c in range(logical_col_count):
            if c == target_col:
                continue
            val = str(grid[row_idx][c] or "").strip() if c < len(grid[row_idx]) else ""
            if val:
                return False
        return True

    for col in sorted(filename_cols):
        row_idx = 0
        while row_idx < len(grid) - 1:
            current = str(grid[row_idx][col] or "").strip() if col < len(grid[row_idx]) else ""
            nxt = str(grid[row_idx + 1][col] or "").strip() if col < len(grid[row_idx + 1]) else ""
            if not current or not nxt:
                row_idx += 1
                continue
            if not prefix_pat.fullmatch(current):
                row_idx += 1
                continue
            if not suffix_pat.fullmatch(nxt):
                row_idx += 1
                continue
            if not row_other_cols_empty(row_idx + 1, col):
                row_idx += 1
                continue

            merged = f"{current}{nxt}"
            if not suffix_pat.fullmatch(merged):
                row_idx += 1
                continue

            grid[row_idx][col] = merged
            grid[row_idx + 1][col] = None

            for cell in rows[row_idx].cells:
                if cell.logical_col == col:
                    cell.text = merged
                    cell.supplemented = True
                    cell.supplement_reason = "filename_continuation_merge:head"
                    break
            for cell in rows[row_idx + 1].cells:
                if cell.logical_col == col:
                    cell.text = None
                    cell.supplemented = True
                    cell.supplement_reason = "filename_continuation_merge:tail_consumed"
                    break

            merged_count += 1
            row_idx += 2

    return merged_count


__all__ = [
    "recover_key_identifier_cells",
    "repair_directory_listing_structure",
    "detect_filename_semantic_columns",
    "merge_filename_continuations",
]
