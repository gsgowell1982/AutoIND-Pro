# Version: v1.0.2
# Optimization Summary:
# - Preserve single ownership when directory filename tails are promoted into the
#   filename column by trimming duplicated filename prefixes from descriptions.
# - Keep semantic repairs non-destructive to observed facts unless a promoted
#   value has an exact matching prefix in the companion description cell.
# - Migrate semantic repair logic out of normalization for maintainability.
# - Keep rules evidence-driven and non-destructive (fill-empty / merge-broken only).
# - Provide reusable functions for rule-engine apply and shadow execution.

from __future__ import annotations

import re
from typing import Any


_FILENAME_PREFIX_STRIP_RE_TEMPLATE = r"^{prefix}(?:[\s\u3000:：-]+)(?P<rest>.+)$"


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

    def _rewrite_cell(row_idx: int, col_idx: int, text: str, reason: str) -> bool:
        if not (0 <= row_idx < len(grid) and 0 <= col_idx < logical_col_count):
            return False
        text = text.strip()
        if not text:
            return False
        current = str(grid[row_idx][col_idx] or "").strip()
        if current == text:
            return False
        grid[row_idx][col_idx] = text
        for norm_cell in rows[row_idx].cells:
            if norm_cell.logical_col == col_idx:
                norm_cell.text = text
                norm_cell.supplemented = True
                norm_cell.supplement_reason = reason
                break
        return True

    def _clear_cell(row_idx: int, col_idx: int, reason: str) -> bool:
        if not (0 <= row_idx < len(grid) and 0 <= col_idx < logical_col_count):
            return False
        current = str(grid[row_idx][col_idx] or "").strip()
        if not current:
            return False
        grid[row_idx][col_idx] = None
        for norm_cell in rows[row_idx].cells:
            if norm_cell.logical_col == col_idx:
                norm_cell.text = None
                norm_cell.supplemented = True
                norm_cell.supplement_reason = reason
                break
        return True

    def _line_tokens(text: str | None) -> list[str]:
        return [line.strip() for line in str(text or "").splitlines() if line.strip()]

    def _subsequent_same_column_values(row_idx: int, col_idx: int) -> set[str]:
        values: set[str] = set()
        for later in grid[row_idx + 1 :]:
            if col_idx >= len(later):
                continue
            value = str(later[col_idx] or "").strip()
            if value:
                values.add(value.lower())
        return values

    def _remove_embedded_child_filename_lists() -> int:
        changed = 0
        for row_idx, row in enumerate(grid):
            if len(row) < 3:
                continue
            folder_key = str(row[0] or "").strip().lower()
            if folder_key not in {"dtd", "util", "style"}:
                continue
            for col_idx in range(1, min(logical_col_count, len(row) - 1)):
                lines = _line_tokens(row[col_idx])
                if len(lines) < 2:
                    continue
                subsequent_values = _subsequent_same_column_values(row_idx, col_idx)
                if not subsequent_values:
                    continue
                embedded_children = [line for line in lines[1:] if line.lower() in subsequent_values]
                if len(embedded_children) < 2:
                    continue
                if len(embedded_children) != len(lines) - 1:
                    continue
                if lines[0].lower() and lines[0].lower() != folder_key:
                    continue
                if _clear_cell(row_idx, col_idx, "directory_embedded_child_filename_list_removed"):
                    changed += 1
        return changed

    def _strip_promoted_filename_prefix(description: str, filename: str) -> str:
        desc = description.strip()
        name = filename.strip()
        if not desc or not name:
            return desc
        pattern = re.compile(
            _FILENAME_PREFIX_STRIP_RE_TEMPLATE.format(prefix=re.escape(name)),
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.match(desc)
        if not match:
            return desc
        rest = str(match.group("rest") or "").strip()
        return rest or desc

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

    repaired += _remove_embedded_child_filename_lists()

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

    for idx, row in enumerate(grid):
        if len(row) < 3:
            continue
        filename = str(row[1] or "").strip()
        description = str(row[2] or "").strip()
        if not filename or not description:
            continue
        trimmed_description = _strip_promoted_filename_prefix(description, filename)
        if trimmed_description == description:
            continue
        if _rewrite_cell(idx, 2, trimmed_description, "directory_desc_prefix_trim"):
            repaired += 1

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


def reconstruct_filename_path_cells_from_text_layer(
    rows: list[Any],
    grid: list[list[str | None]],
    raw_evidence: Any,
    logical_col_count: int,
) -> int:
    """Prefer word/span text for filename/path cells when table extraction corrupts tokens.

    PyMuPDF table cell extraction can treat low-position glyphs such as
    underscores as line fragments, while the page text layer still exposes the
    correct filename/path token. This repair is intentionally limited to
    filename/path semantic columns and only rewrites a cell when same-cell
    geometry provides a stronger, content-compatible candidate.
    """
    if logical_col_count <= 0 or not rows or not grid:
        return 0

    filename_cols = detect_filename_semantic_columns(grid, logical_col_count)
    if not filename_cols:
        filename_cols = _detect_filename_columns_from_headers_or_raw_cells(grid, raw_evidence, logical_col_count)
    if not filename_cols:
        return 0

    changed = 0
    row_lookup = {getattr(row, "physical_row", idx): row for idx, row in enumerate(rows)}

    for raw_row in getattr(raw_evidence, "rows", []) or []:
        row_idx = int(getattr(raw_row, "physical_row", -1))
        if row_idx < 0 or row_idx >= len(grid):
            continue
        normalized_row = row_lookup.get(row_idx) or (rows[row_idx] if row_idx < len(rows) else None)
        if normalized_row is None:
            continue

        for logical_col in sorted(filename_cols):
            if logical_col >= logical_col_count or logical_col >= len(grid[row_idx]):
                continue
            current = str(grid[row_idx][logical_col] or "").strip()
            if not _looks_like_reconstructable_filename_cell(current):
                continue
            raw_cell = _raw_cell_for_logical_column(raw_row, logical_col, logical_col_count)
            cell_bbox = getattr(raw_cell, "bbox", None) if raw_cell is not None else None
            if not cell_bbox:
                continue

            candidate = _best_text_layer_filename_candidate(raw_evidence, cell_bbox, current)
            if not candidate:
                continue
            if candidate == current:
                continue
            if not _filename_candidate_matches_cell_text(candidate, current):
                continue

            grid[row_idx][logical_col] = candidate
            _rewrite_normalized_cell(
                normalized_row,
                logical_col,
                candidate,
                "filename_path_text_layer_reconstruction",
                _candidate_source_kind(raw_evidence, cell_bbox, candidate),
            )
            changed += 1

    return changed


def _detect_filename_columns_from_headers_or_raw_cells(
    grid: list[list[str | None]],
    raw_evidence: Any,
    logical_col_count: int,
) -> set[int]:
    candidates: set[int] = set()
    header_terms = ("filename", "file", "folder", "path", "href", "link", "文件", "路径")
    for row in grid[:3]:
        for col in range(min(logical_col_count, len(row))):
            text = str(row[col] or "").strip().lower()
            if text and any(term in text for term in header_terms):
                candidates.add(col)

    for col in range(logical_col_count):
        hits = 0
        non_empty = 0
        for raw_row in getattr(raw_evidence, "rows", []) or []:
            raw_cell = _raw_cell_for_logical_column(raw_row, col, logical_col_count)
            text = str(getattr(raw_cell, "text", "") or "").strip()
            if not text:
                continue
            non_empty += 1
            if _looks_like_reconstructable_filename_cell(text):
                hits += 1
        if non_empty >= 2 and hits / max(1, non_empty) >= 0.5:
            candidates.add(col)
    return candidates


def _raw_cell_for_logical_column(raw_row: Any, logical_col: int, logical_col_count: int) -> Any | None:
    cells = list(getattr(raw_row, "cells", []) or [])
    if not cells:
        return None
    if logical_col_count <= 0:
        return cells[logical_col] if logical_col < len(cells) else None
    physical_count = len(cells)
    if physical_count == logical_col_count:
        return cells[logical_col] if logical_col < physical_count else None
    target_center = (logical_col + 0.5) / logical_col_count
    best_cell = None
    best_distance = float("inf")
    for cell in cells:
        physical_col = int(getattr(cell, "physical_col", 0) or 0)
        physical_center = (physical_col + 0.5) / max(1, physical_count)
        distance = abs(physical_center - target_center)
        if distance < best_distance:
            best_cell = cell
            best_distance = distance
    return best_cell


def _looks_like_reconstructable_filename_cell(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if not re.search(r"\.(?:xml|xsl|xsd|dtd|txt|pdf|csv|zip)\b", candidate, re.IGNORECASE):
        return False
    compact = re.sub(r"\s+", "", candidate)
    if re.search(r"[_\\/][A-Za-z0-9]", compact):
        return True
    if re.search(r"[A-Za-z0-9][_-][A-Za-z0-9]", compact):
        return True
    if re.search(r"\b\d{3,}\s+[A-Za-z0-9][A-Za-z0-9.-]*\.", candidate):
        return True
    if re.search(r"[\\/][A-Za-z0-9]+(?:\s+|_)[A-Za-z0-9.-]+\.", candidate):
        return True
    return False


def _best_text_layer_filename_candidate(raw_evidence: Any, bbox: Any, current: str) -> str | None:
    for source_kind, items in (
        ("word", getattr(raw_evidence, "words", []) or []),
        ("span", getattr(raw_evidence, "spans", []) or []),
    ):
        candidates = _collect_filename_candidates_from_items(items, bbox)
        if not candidates:
            continue
        for candidate in candidates:
            _ = source_kind
            if _filename_candidate_matches_cell_text(candidate, current):
                return candidate
    return None


def _collect_filename_candidates_from_items(items: list[Any], bbox: Any) -> list[str]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    in_cell: list[tuple[float, float, str]] = []
    for item in items:
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        item_x0 = float(getattr(item, "x0", 0.0) or 0.0)
        item_y0 = float(getattr(item, "y0", 0.0) or 0.0)
        item_x1 = float(getattr(item, "x1", 0.0) or 0.0)
        item_y1 = float(getattr(item, "y1", 0.0) or 0.0)
        cx = (item_x0 + item_x1) / 2
        cy = (item_y0 + item_y1) / 2
        if not (x0 - 1.0 <= cx <= x1 + 1.0 and y0 - 1.0 <= cy <= y1 + 1.0):
            continue
        in_cell.append((item_y0, item_x0, text))

    if not in_cell:
        return []

    ordered_texts = [text for _, _, text in sorted(in_cell, key=lambda item: (item[0], item[1]))]
    candidates: list[str] = []
    for text in ordered_texts:
        if _is_strong_filename_token(text):
            candidates.append(text)
    joined_without_space = "".join(ordered_texts).strip()
    if _is_strong_filename_token(joined_without_space):
        candidates.append(joined_without_space)
    joined_with_space = " ".join(ordered_texts).strip()
    if _looks_like_reconstructable_filename_cell(joined_with_space):
        candidates.append(joined_with_space)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _is_strong_filename_token(text: str) -> bool:
    candidate = str(text or "").strip()
    if not re.search(r"\.(?:xml|xsl|xsd|dtd|txt|pdf|csv|zip)\b", candidate, re.IGNORECASE):
        return False
    if " " in candidate:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._\\/\-]+", candidate))


def _filename_candidate_matches_cell_text(candidate: str, current: str) -> bool:
    cand = _filename_match_key(candidate)
    cur = _filename_match_key(current)
    if not cand or not cur:
        return False
    if cand == cur:
        return True
    if not _has_filename_separator_damage_evidence(current):
        return False
    loose_cand = re.sub(r"[_\\/\-]+", "", cand)
    loose_cur = re.sub(r"[_\\/\-]+", "", cur)
    if loose_cand and loose_cand == loose_cur:
        return True
    return False


def _has_filename_separator_damage_evidence(text: str) -> bool:
    current = str(text or "")
    if not re.search(r"\s", current):
        return False
    if re.search(r"[_\\/\-]\s+", current):
        return True
    if re.search(r"\s+[_\\/\-]", current):
        return True
    if re.search(r"\b\d{3,}\s+[A-Za-z0-9][A-Za-z0-9.-]*\.", current):
        return True
    if re.search(r"[\\/][A-Za-z0-9]+(?:\s+|_)[A-Za-z0-9.-]+\.", current):
        return True
    return False


def _filename_match_key(text: str) -> str:
    key = str(text or "").strip().lower()
    key = re.sub(r"\s+", "", key)
    key = key.replace("\u3000", "")
    return key


def _candidate_source_kind(raw_evidence: Any, bbox: Any, candidate: str) -> str:
    for source_kind, items in (
        ("word", getattr(raw_evidence, "words", []) or []),
        ("span", getattr(raw_evidence, "spans", []) or []),
    ):
        if candidate in _collect_filename_candidates_from_items(items, bbox):
            return source_kind
    return "text_layer"


def _rewrite_normalized_cell(
    normalized_row: Any,
    logical_col: int,
    text: str,
    reason: str,
    source_kind: str,
) -> None:
    for cell in getattr(normalized_row, "cells", []) or []:
        if int(getattr(cell, "logical_col", -1) or -1) != logical_col:
            continue
        cell.text = text
        cell.supplemented = True
        cell.supplement_reason = f"{reason}:{source_kind}"
        return


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
    "reconstruct_filename_path_cells_from_text_layer",
    "detect_filename_semantic_columns",
    "merge_filename_continuations",
]
