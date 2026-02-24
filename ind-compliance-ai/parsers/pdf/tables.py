from __future__ import annotations

from collections import Counter
import hashlib
import re
import statistics
from typing import Any

from .layout import _drawing_rect_to_tuple
from .shared import (
    _Word,
    _bbox_intersection_ratio,
    _bbox_to_list,
    _bbox_union,
    _clean_text,
    _compact_text,
    _horizontal_overlap_ratio,
)

TABLE_TITLE_PATTERN = re.compile(
    r"^\s*(?:附?表\s*[0-9A-Za-z一二三四五六七八九十零〇.\-]+|table\s*[0-9A-Za-z.\-]+)\s*(?:[.:：、\-]|\s+)",
    re.IGNORECASE,
)
TABLE_SECTION_HINT_PATTERN = re.compile(r"(术语表|词汇表|名词表|名词解释|参数表|清单|附录表|glossary)", re.IGNORECASE)
TOC_LINE_PATTERN = re.compile(r"[\.·…]{4,}\s*\d{1,3}\s*$")
TOC_HEADING_PATTERN = re.compile(r"^\s*(目录|contents)\s*$", re.IGNORECASE)
REFERENCE_ROW_PATTERN = re.compile(r"^\s*(?:\d+\.\s+|[（(]?\d+[)）])")


def _is_glossary_hint_text(text: str) -> bool:
    normalized = _clean_text(text).lower()
    return "术语表" in normalized or "词汇表" in normalized or "glossary" in normalized


def _cluster_rows(words: list[_Word]) -> list[dict[str, Any]]:
    if not words:
        return []
    median_height = statistics.median([word.height for word in words]) if words else 8.0
    row_tol = max(2.5, median_height * 0.55)
    rows: list[dict[str, Any]] = []

    for word in sorted(words, key=lambda item: (item.yc, item.x0)):
        placed = False
        for row in rows:
            if abs(word.yc - row["yc"]) <= row_tol:
                row["words"].append(word)
                row["yc"] = (row["yc"] * row["n"] + word.yc) / (row["n"] + 1)
                row["n"] += 1
                placed = True
                break
        if not placed:
            rows.append({"yc": word.yc, "n": 1, "words": [word]})

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        row_words = sorted(row["words"], key=lambda item: item.x0)
        row_bbox = _bbox_union([(item.x0, item.y0, item.x1, item.y1) for item in row_words])
        normalized_rows.append(
            {
                "words": row_words,
                "bbox": row_bbox,
                "y0": row_bbox[1],
                "y1": row_bbox[3],
                "height": max(0.1, row_bbox[3] - row_bbox[1]),
            }
        )

    normalized_rows.sort(key=lambda item: item["y0"])
    return normalized_rows


def _group_table_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    dense_rows_in_current = 0
    pending_sparse: list[dict[str, Any]] = []
    for row in rows:
        is_dense_row = len(row["words"]) >= 2
        if not current:
            if not is_dense_row:
                if pending_sparse:
                    prev_sparse = pending_sparse[-1]
                    gap = row["y0"] - prev_sparse["y1"]
                    avg_height = (row["height"] + prev_sparse["height"]) / 2
                    overlap = _horizontal_overlap_ratio(prev_sparse["bbox"], row["bbox"])
                    if gap > max(22.0, avg_height * 2.2) or overlap < 0.08:
                        pending_sparse = []
                pending_sparse.append(row)
                if len(pending_sparse) > 4:
                    pending_sparse = pending_sparse[-4:]
                continue
            leading_sparse_rows: list[dict[str, Any]] = []
            if pending_sparse:
                eligible_sparse_indexes: list[int] = []
                for index, sparse_row in enumerate(pending_sparse):
                    vertical_gap = row["y0"] - sparse_row["y1"]
                    overlap = _horizontal_overlap_ratio(sparse_row["bbox"], row["bbox"])
                    # Keep sparse-leading rows only when they are tightly adjacent to the first dense row.
                    if -2.0 <= vertical_gap <= 12.0 and overlap >= 0.16:
                        eligible_sparse_indexes.append(index)
                if eligible_sparse_indexes:
                    include_flags = [False] * len(pending_sparse)
                    for index in eligible_sparse_indexes:
                        include_flags[index] = True
                    first_included = min(eligible_sparse_indexes)
                    # Backtrack contiguous sparse rows (e.g. "回复/撤回") so the full leading line is preserved.
                    for index in range(first_included - 1, -1, -1):
                        current_sparse = pending_sparse[index]
                        next_sparse = pending_sparse[index + 1]
                        gap = next_sparse["y0"] - current_sparse["y1"]
                        overlap = _horizontal_overlap_ratio(current_sparse["bbox"], next_sparse["bbox"])
                        if -2.0 <= gap <= 12.0 and overlap >= 0.16:
                            include_flags[index] = True
                            continue
                        break
                    leading_sparse_rows = [item for index, item in enumerate(pending_sparse) if include_flags[index]]
            current = leading_sparse_rows + [row]
            dense_rows_in_current = 1
            pending_sparse = []
            continue
        previous = current[-1]

        if is_dense_row:
            vertical_gap = row["y0"] - previous["y1"]
            avg_height = (row["height"] + previous["height"]) / 2
            overlap = _horizontal_overlap_ratio(previous["bbox"], row["bbox"])
            if vertical_gap <= max(7.0, avg_height * 1.35) and overlap >= 0.18:
                current.append(row)
                dense_rows_in_current += 1
                continue
            if len(current) >= 2 and dense_rows_in_current >= 2:
                groups.append(current)
            current = [row]
            dense_rows_in_current = 1
            pending_sparse = []
            continue

        # Sparse rows are treated as continuation lines only when they tightly follow.
        vertical_gap = row["y0"] - previous["y1"]
        avg_height = (row["height"] + previous["height"]) / 2
        overlap = _horizontal_overlap_ratio(previous["bbox"], row["bbox"])
        if vertical_gap <= max(4.5, avg_height * 0.9) and overlap >= 0.35:
            current.append(row)
            continue
        if len(current) >= 2 and dense_rows_in_current >= 2:
            groups.append(current)
        current = []
        dense_rows_in_current = 0
        pending_sparse = [row]

    if len(current) >= 2 and dense_rows_in_current >= 2:
        groups.append(current)
    return [group for group in groups if sum(1 for row in group if len(row["words"]) >= 2) >= 2]


def _row_text_from_words(words: list[_Word]) -> str:
    return _clean_text(" ".join(word.text for word in words))


def _is_heading_like_row_text(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return bool(
        re.match(r"^\d+(?:\.\d+)*\s*[\u4e00-\u9fffA-Za-z]{1,20}表$", cleaned)
        or re.match(r"^表\s*[0-9A-Za-z一二三四五六七八九十零〇.\-]+\b", cleaned)
        or "术语表" in cleaned
    )


def _is_section_heading_like_tail(words: list[_Word]) -> bool:
    if not words:
        return False
    parts = [word.text for word in words]
    if len(parts) == 1:
        return bool(re.match(r"^\d+(?:\.\d+){1,4}$", parts[0]))
    if len(parts) == 2 and re.match(r"^\d+(?:\.\d+){1,4}$", parts[0]):
        return len(parts[1]) <= 12 and bool(re.search(r"[\u4e00-\u9fffA-Za-z]", parts[1]))
    merged = _row_text_from_words(words)
    return bool(re.match(r"^\d+(?:\.\d+){1,4}\s+\S+$", merged)) and len(parts) <= 3


def _prune_non_table_edge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return rows
    pruned = list(rows)
    # Drop leading table-heading rows like "6. 术语表".
    while len(pruned) >= 2:
        first_words = pruned[0]["words"]
        second_words = pruned[1]["words"]
        if len(first_words) <= 3 and len(second_words) >= 2 and _is_heading_like_row_text(_row_text_from_words(first_words)):
            pruned = pruned[1:]
            continue
        break

    # Drop trailing section heading rows like "2.3 序列信息".
    while len(pruned) >= 2:
        last_words = pruned[-1]["words"]
        prev_words = pruned[-2]["words"]
        if len(prev_words) >= 2 and _is_section_heading_like_tail(last_words):
            pruned = pruned[:-1]
            continue
        break

    # Drop leading narrative lines introducing tables and in-table caption rows.
    while len(pruned) >= 2:
        first_text = _row_text_from_words(pruned[0]["words"])
        second_text = _row_text_from_words(pruned[1]["words"])
        if not first_text:
            pruned = pruned[1:]
            continue
        is_intro_line = bool(re.search(r"(如下表|见下表|详见表|见表)", first_text))
        is_caption_line = bool(TABLE_TITLE_PATTERN.search(first_text))
        if is_intro_line or is_caption_line:
            pruned = pruned[1:]
            continue
        second_is_intro_or_caption = bool(re.search(r"(如下表|见下表|详见表|见表)", second_text)) or bool(TABLE_TITLE_PATTERN.search(second_text))
        first_compact_len = len(_compact_text(first_text))
        first_looks_narrative = (
            first_compact_len >= 10
            and not TABLE_TITLE_PATTERN.search(first_text)
            and not re.search(r"(申请编号|序列号|序列类型|序列描述|注册行为|扩展节点标题|名词|定义)", first_text)
            and not re.fullmatch(r"[A-Za-z0-9.\-_/]+", first_text)
        )
        if first_looks_narrative and second_is_intro_or_caption:
            pruned = pruned[1:]
            continue
        if (
            len(pruned[0]["words"]) == 1
            and re.search(r"[。；;]$", first_text)
            and (bool(re.search(r"(如下表|见下表|详见表|见表)", second_text)) or bool(TABLE_TITLE_PATTERN.search(second_text)))
        ):
            pruned = pruned[1:]
            continue
        break

    # Drop trailing narrative lines introducing figures/next sections.
    while len(pruned) >= 2:
        last_text = _row_text_from_words(pruned[-1]["words"])
        if not last_text:
            pruned = pruned[:-1]
            continue
        if re.search(r"(如下图|见下图|如图|见图)", last_text) or (re.search(r"图\s*\d+", last_text) and "所示" in last_text):
            pruned = pruned[:-1]
            continue
        break

    return pruned


def _assign_rows_to_columns(
    rows: list[dict[str, Any]],
    column_centers: list[float],
) -> tuple[list[dict[int, list[_Word]]], list[str], list[int]]:
    row_cell_maps: list[dict[int, list[_Word]]] = []
    row_texts: list[str] = []
    row_non_empty_counts: list[int] = []
    for row in rows:
        cell_map: dict[int, list[_Word]] = {}
        for word in row["words"]:
            col_index = _closest_column_index(word.xc, column_centers)
            cell_map.setdefault(col_index, []).append(word)
        row_cell_maps.append(cell_map)
        row_texts.append(_clean_text(" ".join(word.text for word in sorted(row["words"], key=lambda item: item.x0))))
        row_non_empty_counts.append(sum(1 for words in cell_map.values() if words))
    return row_cell_maps, row_texts, row_non_empty_counts


def _is_header_like_row(words: list[_Word]) -> bool:
    if len(words) < 2:
        return False
    merged_text = _clean_text(" ".join(word.text for word in words))
    if not merged_text:
        return False
    token_texts = [_clean_text(word.text) for word in words if _clean_text(word.text)]
    max_token_len = max((len(_compact_text(token)) for token in token_texts), default=0)
    compact_len = len(_compact_text(merged_text))
    if len(token_texts) >= 2:
        first_token = _compact_text(token_texts[0])
        second_token = _compact_text(token_texts[1])
        if first_token and second_token.startswith(first_token) and len(second_token) >= len(first_token) + 4:
            # Common glossary continuation pattern: "术语 术语是..." -> data row, not header.
            return False
    header_keywords = [
        "序列",
        "相关序列",
        "申请类型",
        "注册行为类型",
        "序列类型",
        "序列描述",
        "名词",
        "定义",
        "文件夹",
        "文件",
        "命名规则",
        "扩展节点标题",
    ]
    if any(keyword in merged_text for keyword in header_keywords):
        # Header keywords should appear in concise labels, not in long definition sentences.
        if max_token_len <= 10 and compact_len <= 24 and not re.search(r"[，。；;]", merged_text):
            return True
    # Action-heavy rows are likely data rows in continuation pages.
    data_keywords = [
        "首次提交",
        "回复",
        "撤回",
        "补充申请",
        "再注册",
        "新适应症",
        "备案",
        "报告",
        "格式转换",
        "生产工艺变更",
    ]
    if any(keyword in merged_text for keyword in data_keywords):
        return False
    # Continuation pages often start with long definition lines; avoid treating them as header rows.
    if compact_len >= 18:
        return False
    if max_token_len >= 12:
        return False
    if re.search(r"[，。；;：:]", merged_text):
        return False
    if re.search(r"\b\d{3,}\b", merged_text):
        return False
    digit_count = sum(char.isdigit() for char in merged_text)
    alpha_count = sum(char.isalpha() for char in merged_text)
    return digit_count <= max(1, alpha_count // 6)


def _infer_header_column_centers(header_words: list[_Word]) -> list[float]:
    if len(header_words) < 2:
        return []
    ordered = sorted(header_words, key=lambda item: item.x0)
    median_width = statistics.median([word.width for word in ordered]) if ordered else 24.0
    merge_gap = min(10.0, max(4.0, median_width * 0.2))

    groups: list[list[_Word]] = [[ordered[0]]]
    for word in ordered[1:]:
        previous = groups[-1][-1]
        gap = word.x0 - previous.x1
        if gap <= merge_gap:
            groups[-1].append(word)
        else:
            groups.append([word])

    centers: list[float] = []
    for group in groups:
        centers.append(sum(item.xc for item in group) / len(group))
    return centers


def _refine_column_centers(
    initial_column_centers: list[float],
    row_non_empty_counts: list[int],
) -> list[float]:
    if len(initial_column_centers) <= 2:
        return initial_column_centers
    candidates = [count for count in row_non_empty_counts if count >= 2]
    if not candidates:
        return initial_column_centers

    target_by_mode = Counter(candidates).most_common(1)[0][0]
    if target_by_mode >= len(initial_column_centers):
        return initial_column_centers
    if target_by_mode < 2:
        return initial_column_centers
    # Only collapse one suspicious split-column per table in this pass.
    target_by_mode = max(target_by_mode, len(initial_column_centers) - 1)

    centers = list(sorted(initial_column_centers))
    while len(centers) > target_by_mode:
        if len(centers) <= 2:
            break
        nearest_pair_index = min(
            range(len(centers) - 1),
            key=lambda index: centers[index + 1] - centers[index],
        )
        merged_center = (centers[nearest_pair_index] + centers[nearest_pair_index + 1]) / 2
        centers = centers[:nearest_pair_index] + [merged_center] + centers[nearest_pair_index + 2 :]
    return centers


def _cluster_columns(rows: list[dict[str, Any]]) -> list[float]:
    words = [word for row in rows for word in row["words"]]
    if not words:
        return []
    median_width = statistics.median([word.width for word in words]) if words else 12.0
    col_tol = min(max(8.0, median_width * 0.75), 42.0)
    centers = sorted(word.xc for word in words)
    if not centers:
        return []

    clusters: list[list[float]] = [[centers[0]]]
    for center in centers[1:]:
        if abs(center - clusters[-1][-1]) <= col_tol:
            clusters[-1].append(center)
        else:
            clusters.append([center])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _closest_column_index(x_center: float, column_centers: list[float]) -> int:
    return min(range(len(column_centers)), key=lambda index: abs(column_centers[index] - x_center))


def _column_hash(column_signature: list[float]) -> str:
    signature = ",".join(f"{value:.3f}" for value in column_signature)
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]


def _build_single_table_ast(
    rows: list[dict[str, Any]],
    page_number: int,
    page_height: float,
    table_id: str,
) -> dict[str, Any] | None:
    rows = _prune_non_table_edge_rows(rows)
    if len(rows) < 2:
        return None

    header_row_candidate_index: int | None = None
    for index, row in enumerate(rows[: min(3, len(rows))]):
        if _is_header_like_row(row["words"]):
            header_row_candidate_index = index
            break
    header_centers = _infer_header_column_centers(rows[header_row_candidate_index]["words"]) if header_row_candidate_index is not None else []

    initial_column_centers = _cluster_columns(rows)
    if len(initial_column_centers) < 1 and 1 <= len(header_centers) <= 24:
        initial_column_centers = header_centers
    if len(initial_column_centers) < 1 or len(initial_column_centers) > 24:
        return None

    _, row_texts, initial_non_empty_counts = _assign_rows_to_columns(rows, initial_column_centers)
    column_centers = _refine_column_centers(initial_column_centers, initial_non_empty_counts)
    if 1 <= len(header_centers) <= 24 and len(header_centers) >= len(column_centers):
        # Prefer header-aligned columns when header expresses the real table schema.
        column_centers = header_centers

    row_cell_maps, row_texts, row_non_empty_counts = _assign_rows_to_columns(rows, column_centers)
    if len(column_centers) < 1 or len(column_centers) > 24:
        return None

    table_bbox = _bbox_union([row["bbox"] for row in rows])
    table_width = max(1.0, table_bbox[2] - table_bbox[0])
    column_signature = [round((center - table_bbox[0]) / table_width, 3) for center in column_centers]

    header_row_index = -1
    for index, cell_map in enumerate(row_cell_maps[: min(3, len(row_cell_maps))]):
        non_empty = row_non_empty_counts[index]
        if non_empty >= 2 and _is_header_like_row(rows[index]["words"]):
            header_row_index = index
            break
    if header_row_index < 0 and len(column_centers) == 1 and row_cell_maps:
        # Single-column tables often have a one-cell header.
        header_row_index = 0

    header: list[dict[str, Any]] = []
    if header_row_index >= 0 and row_cell_maps:
        header_cells = row_cell_maps[header_row_index]
        for col_index in range(len(column_centers)):
            words = sorted(header_cells.get(col_index, []), key=lambda item: item.x0)
            text = _clean_text(" ".join(word.text for word in words))
            header.append({"text": text or f"Column {col_index + 1}", "col": col_index + 1})
    else:
        for col_index in range(len(column_centers)):
            header.append({"text": f"Column {col_index + 1}", "col": col_index + 1})

    cells: list[dict[str, Any]] = []
    latest_cell_by_column: dict[int, dict[str, Any]] = {}
    data_start_index = header_row_index + 1 if header_row_index >= 0 else 0
    data_row_maps = row_cell_maps[data_start_index:]
    data_row_non_empty_counts = row_non_empty_counts[data_start_index:]
    logical_data_row_index = 0

    for raw_row_index, cell_map in enumerate(data_row_maps, start=1):
        non_empty_this_row = data_row_non_empty_counts[raw_row_index - 1]
        row_has_single_column = non_empty_this_row == 1
        continuation_col_index = next((index for index, words in cell_map.items() if words), -1)
        row_fragments: list[dict[str, Any]] = []
        for col_index in range(len(column_centers)):
            words = sorted(cell_map.get(col_index, []), key=lambda item: item.x0)
            if not words:
                continue
            cell_text = _clean_text(" ".join(word.text for word in words))
            if not cell_text:
                continue
            bbox = _bbox_union([(word.x0, word.y0, word.x1, word.y1) for word in words])
            covered_columns = [
                index
                for index, center in enumerate(column_centers)
                if bbox[0] - 2 <= center <= bbox[2] + 2
            ]
            colspan = max(1, len(covered_columns))
            row_fragments.append(
                {
                    "col_index": col_index,
                    "text": cell_text,
                    "bbox": bbox,
                    "colspan": colspan,
                }
            )

        if not row_fragments:
            continue

        present_cols = sorted(fragment["col_index"] for fragment in row_fragments)
        # Multi-column continuation row: no leading key columns, only sparse trailing cells.
        can_merge_sparse_continuation = (
            non_empty_this_row <= 2
            and present_cols
            and present_cols[0] >= 2
        )
        if can_merge_sparse_continuation:
            can_merge_all = True
            for fragment in row_fragments:
                previous_cell = latest_cell_by_column.get(int(fragment["col_index"]))
                if previous_cell is None:
                    can_merge_all = False
                    break
                previous_bbox = tuple(float(item) for item in previous_cell.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                current_bbox = tuple(float(item) for item in fragment["bbox"])
                vertical_gap = current_bbox[1] - previous_bbox[3]
                if vertical_gap > max(24.0, (current_bbox[3] - current_bbox[1]) * 2.4):
                    can_merge_all = False
                    break
            if can_merge_all:
                for fragment in row_fragments:
                    col_index = int(fragment["col_index"])
                    previous_cell = latest_cell_by_column[col_index]
                    previous_bbox = tuple(float(item) for item in previous_cell.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                    current_bbox = tuple(float(item) for item in fragment["bbox"])
                    merged_bbox = _bbox_union([previous_bbox, current_bbox])
                    previous_cell["text"] = f"{previous_cell['text']}\n{fragment['text']}"
                    previous_cell["bbox"] = _bbox_to_list(merged_bbox)
                    previous_cell["rowspan"] = int(previous_cell.get("rowspan", 1)) + 1
                continue

        new_cells_for_this_row: list[dict[str, Any]] = []
        for fragment in row_fragments:
            col_index = int(fragment["col_index"])
            bbox = tuple(float(item) for item in fragment["bbox"])
            cell_text = str(fragment["text"])
            colspan = int(fragment["colspan"])

            if len(column_centers) > 1 and row_has_single_column and col_index == continuation_col_index:
                previous_cell = latest_cell_by_column.get(col_index)
                if previous_cell is not None:
                    previous_bbox = tuple(float(item) for item in previous_cell.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                    vertical_gap = bbox[1] - previous_bbox[3]
                    if vertical_gap <= max(16.0, (bbox[3] - bbox[1]) * 1.8):
                        merged_bbox = _bbox_union([previous_bbox, bbox])
                        previous_cell["text"] = f"{previous_cell['text']}\n{cell_text}"
                        previous_cell["bbox"] = _bbox_to_list(merged_bbox)
                        previous_cell["rowspan"] = int(previous_cell.get("rowspan", 1)) + 1
                        continue

            new_cells_for_this_row.append(
                {
                    "row": 0,
                    "col": col_index + 1,
                    "text": cell_text,
                    "bbox": _bbox_to_list(bbox),
                    "rowspan": 1,
                    "colspan": colspan,
                }
            )

        if not new_cells_for_this_row:
            continue

        logical_data_row_index += 1
        for new_cell in new_cells_for_this_row:
            new_cell["row"] = logical_data_row_index
            cells.append(new_cell)
            latest_cell_by_column[int(new_cell["col"]) - 1] = new_cell

    if not cells:
        return None

    logical_total_rows = (header_row_index + 1 if header_row_index >= 0 else 0) + max(1, logical_data_row_index)

    logical_data_row_count = max(1, logical_data_row_index)
    coverage_ratio = len(cells) / max(1.0, logical_data_row_count * len(column_centers))
    aligned_ratio = sum(1 for count in data_row_non_empty_counts if count >= 2) / max(1, len(data_row_non_empty_counts))
    numeric_ratio = sum(1 for cell in cells if re.search(r"\d", str(cell.get("text", "")))) / max(1, len(cells))
    structure_score = min(1.0, 0.45 * aligned_ratio + 0.35 * min(1.0, coverage_ratio) + 0.2 * numeric_ratio)

    return {
        "block_type": "table",
        "table_id": table_id,
        "page": page_number,
        "bbox": _bbox_to_list(table_bbox),
        "header": header,
        "cells": cells,
        "row_count": logical_total_rows,
        "col_count": len(column_centers),
        "column_signature": column_signature,
        "column_hash": _column_hash(column_signature),
        "header_row_index": header_row_index + 1 if header_row_index >= 0 else 0,
        "structure_score": round(structure_score, 3),
        "row_texts": row_texts,
        "near_page_bottom": table_bbox[3] >= page_height * 0.72,
        "near_page_top": table_bbox[1] <= page_height * 0.28,
    }


def _find_table_title_block(
    table_bbox: tuple[float, float, float, float],
    text_blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in text_blocks:
        text = _clean_text(block.get("text", ""))
        if not text or not TABLE_TITLE_PATTERN.search(text):
            continue
        bbox = tuple(block["bbox"])
        vertical_gap = table_bbox[1] - bbox[3]
        inside_top_band = (
            bbox[1] >= table_bbox[1] - 8
            and bbox[3] <= table_bbox[1] + max(72.0, (table_bbox[3] - table_bbox[1]) * 0.28)
        )
        if (vertical_gap < -8 or vertical_gap > 160) and not inside_top_band:
            continue
        overlap = _horizontal_overlap_ratio(table_bbox, bbox)
        if overlap < 0.12:
            continue
        if inside_top_band:
            distance = 0.0
        else:
            distance = vertical_gap if vertical_gap >= 0 else abs(vertical_gap) + 25
        candidates.append((distance, block))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _find_section_table_hint(
    table_bbox: tuple[float, float, float, float],
    text_blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in text_blocks:
        text = _clean_text(block.get("text", ""))
        if not text or not TABLE_SECTION_HINT_PATTERN.search(text):
            continue
        bbox = tuple(block["bbox"])
        vertical_gap = table_bbox[1] - bbox[3]
        if vertical_gap < -8 or vertical_gap > 180:
            continue
        overlap = _horizontal_overlap_ratio(table_bbox, bbox)
        if overlap < 0.1:
            continue
        candidates.append((max(0.0, vertical_gap), block))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _is_toc_context(
    table_bbox: tuple[float, float, float, float],
    text_blocks: list[dict[str, Any]],
) -> bool:
    table_top = table_bbox[1]
    for block in text_blocks:
        text = _clean_text(block.get("text", ""))
        if not text:
            continue
        bbox = tuple(block["bbox"])
        if bbox[3] > table_top + 15:
            continue
        vertical_gap = table_top - bbox[3]
        if vertical_gap < 0 or vertical_gap > 220:
            continue
        if TOC_HEADING_PATTERN.search(text):
            return True
    return False


def _table_toc_row_ratio(table_ast: dict[str, Any]) -> float:
    row_texts = [str(item) for item in table_ast.get("row_texts", [])]
    if not row_texts:
        return 0.0
    toc_rows = 0.0
    for row_text in row_texts:
        cleaned = _clean_text(row_text)
        if not cleaned:
            continue
        if TOC_LINE_PATTERN.search(cleaned):
            toc_rows += 1.0
            continue
        tokens = cleaned.split()
        if len(tokens) >= 3 and re.fullmatch(r"\d+(?:\.\d+){1,4}", tokens[0]) and re.fullmatch(r"\d{1,3}", tokens[-1]):
            middle = " ".join(tokens[1:-1])
            has_textual_middle = bool(re.search(r"[A-Za-z\u4e00-\u9fff]", middle))
            has_dot_leader = bool(re.search(r"[\.·…]{2,}", middle))
            if has_textual_middle and (has_dot_leader or len(middle) >= 3):
                toc_rows += 0.75
                continue
        if re.search(r"[\.·…]{2,}", cleaned) and re.search(r"\d{1,3}\s*$", cleaned):
            toc_rows += 0.65
    return min(1.0, toc_rows / max(1, len(row_texts)))


def _table_grid_line_score(
    table_bbox: tuple[float, float, float, float],
    page_drawings: list[dict[str, Any]],
) -> float:
    overlap_count = 0
    for drawing in page_drawings:
        rect = _drawing_rect_to_tuple(drawing)
        if rect is None:
            continue
        if _bbox_intersection_ratio(rect, table_bbox) >= 0.18:
            overlap_count += 1
    return min(1.0, overlap_count / 8.0)


def _table_reference_like_ratio(table_ast: dict[str, Any]) -> float:
    row_texts = [str(item) for item in table_ast.get("row_texts", []) if str(item).strip()]
    if not row_texts:
        return 0.0
    score = 0.0
    for row_text in row_texts:
        cleaned = _clean_text(row_text)
        if not cleaned:
            continue
        row_score = 0.0
        lowered = cleaned.lower()
        if "——《" in cleaned or ("《" in cleaned and "》" in cleaned):
            row_score += 0.9
        if REFERENCE_ROW_PATTERN.match(cleaned):
            row_score += 0.75
        if re.search(r"\bV\d+(?:\.\d+){1,3}\b", cleaned, re.IGNORECASE):
            row_score += 0.45
        if "ich" in lowered:
            row_score += 0.35
        if re.search(r"(specification|document|technical|change request)", lowered):
            row_score += 0.25
        score += min(1.0, row_score)
    return min(1.0, score / max(1, len(row_texts)))


def _find_continuation_hint(
    candidate_table: dict[str, Any],
    accepted_tables: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not bool(candidate_table.get("near_page_top")):
        return None
    best_parent: dict[str, Any] | None = None
    best_similarity = 0.0
    current_page = int(candidate_table.get("page", 0))
    for previous in reversed(accepted_tables[-12:]):
        if int(previous.get("page", 0)) != current_page - 1:
            continue
        previous_has_title = bool(str(previous.get("title", "")).strip())
        previous_section_hint_text = str(previous.get("section_hint", ""))
        previous_is_glossary_context = _is_glossary_hint_text(previous_section_hint_text) or (
            str(previous.get("continuation_context", "")).strip().lower() == "glossary"
        )
        previous_has_section_hint = bool(previous_section_hint_text.strip()) or previous_is_glossary_context
        previous_has_chain = bool(previous.get("continuation_hint")) or bool(previous.get("continued_from"))
        previous_col_count = int(previous.get("col_count", 0))
        candidate_col_count = int(candidate_table.get("col_count", 0))
        if not bool(previous.get("near_page_bottom")) and not previous_has_title and not previous_has_section_hint:
            continue
        similarity = _column_similarity(
            list(previous.get("column_signature", [])),
            list(candidate_table.get("column_signature", [])),
        )
        if (
            previous_is_glossary_context
            and bool(candidate_table.get("near_page_top"))
            and abs(previous_col_count - candidate_col_count) <= 1
            and similarity >= 0.22
        ):
            similarity += 0.25
        elif (
            previous_has_chain
            and bool(candidate_table.get("near_page_top"))
            and abs(previous_col_count - candidate_col_count) <= 1
            and similarity >= 0.45
        ):
            similarity += 0.15
        elif (
            (previous_has_title or previous_has_section_hint)
            and bool(candidate_table.get("near_page_top"))
            and abs(previous_col_count - candidate_col_count) <= 1
            and similarity >= 0.28
        ):
            similarity += 0.25
        elif (previous_has_title or previous_has_section_hint) and bool(candidate_table.get("near_page_top")) and similarity >= 0.38:
            similarity += 0.08
        if similarity > best_similarity:
            best_similarity = similarity
            best_parent = previous
    if best_parent is None or best_similarity < 0.42:
        return None
    return {
        "table_id": str(best_parent.get("table_id", "")),
        "similarity": round(min(1.0, best_similarity), 3),
    }


def _is_valid_table_candidate(
    table_ast: dict[str, Any],
    title_block: dict[str, Any] | None,
    section_hint_block: dict[str, Any] | None,
    toc_context: bool,
    grid_line_score: float,
    continuation_hint: dict[str, Any] | None,
) -> bool:
    score = float(table_ast.get("structure_score", 0.0))
    row_count = int(table_ast.get("row_count", 0))
    col_count = int(table_ast.get("col_count", 0))
    toc_row_ratio = float(table_ast.get("toc_row_ratio", 0.0))
    reference_like_ratio = _table_reference_like_ratio(table_ast)
    if toc_context or toc_row_ratio >= 0.38:
        return False

    if (
        title_block is None
        and continuation_hint is None
        and reference_like_ratio >= 0.75
    ):
        return False

    if (
        title_block is None
        and continuation_hint is None
        and section_hint_block is None
        and grid_line_score < 0.3
        and reference_like_ratio >= 0.42
    ):
        return False

    if (
        title_block is None
        and continuation_hint is None
        and grid_line_score < 0.2
        and col_count >= 4
        and row_count <= 8
        and reference_like_ratio >= 0.25
    ):
        return False

    if title_block is not None:
        if col_count == 1:
            return row_count >= 4 and score >= 0.15
        return score >= 0.32 and row_count >= 2 and col_count >= 2

    if continuation_hint is not None:
        is_glossary_context = str(table_ast.get("continuation_context", "")).strip().lower() == "glossary"
        if row_count < 4 and not is_glossary_context:
            return False
        if is_glossary_context and row_count >= 2 and col_count >= 2:
            # Glossary continuation pages can end with only a few rows, but should keep table grid signals.
            return grid_line_score >= 0.45 and score >= 0.15
        if col_count <= 3 and row_count >= 3:
            return score >= 0.15
        return row_count >= 2 and col_count >= 2 and score >= 0.42

    if section_hint_block is not None:
        section_text = _clean_text(str(section_hint_block.get("text", ""))).lower()
        if "术语表" in section_text or "glossary" in section_text:
            return row_count >= 4 and col_count >= 2 and score >= 0.25
        return row_count >= 4 and col_count >= 2 and score >= 0.55

    if grid_line_score >= 0.45:
        return row_count >= 4 and col_count >= 2 and score >= 0.56

    # Without caption/hint/grid, keep only very strong grids.
    strong_multi_col = row_count >= 5 and col_count >= 3 and score >= 0.9 and toc_row_ratio <= 0.15
    strong_two_col = row_count >= 8 and col_count >= 2 and score >= 0.93 and toc_row_ratio <= 0.1
    return strong_multi_col or strong_two_col


def _merge_two_table_fragments(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    primary["bbox"] = _bbox_to_list(_bbox_union([primary_bbox, secondary_bbox]))

    current_max_row = max((int(cell.get("row", 0)) for cell in primary.get("cells", [])), default=0)
    secondary_cells: list[dict[str, Any]] = []
    for cell in secondary.get("cells", []):
        cell_copy = dict(cell)
        cell_copy["row"] = int(cell_copy.get("row", 0)) + current_max_row
        secondary_cells.append(cell_copy)
    primary.setdefault("cells", [])
    primary["cells"].extend(secondary_cells)

    primary["row_count"] = int(primary.get("row_count", 0)) + int(secondary.get("row_count", 0))
    primary["col_count"] = max(int(primary.get("col_count", 0)), int(secondary.get("col_count", 0)))
    primary["structure_score"] = round(
        max(float(primary.get("structure_score", 0.0)), float(secondary.get("structure_score", 0.0))),
        3,
    )
    primary["grid_line_score"] = round(
        max(float(primary.get("grid_line_score", 0.0)), float(secondary.get("grid_line_score", 0.0))),
        3,
    )
    primary["near_page_bottom"] = bool(primary.get("near_page_bottom")) or bool(secondary.get("near_page_bottom"))
    primary["near_page_top"] = bool(primary.get("near_page_top")) and bool(secondary.get("near_page_top"))
    if not primary.get("title") and secondary.get("title"):
        primary["title"] = secondary.get("title")
        primary["title_block_id"] = secondary.get("title_block_id")
    if not primary.get("section_hint") and secondary.get("section_hint"):
        primary["section_hint"] = secondary.get("section_hint")
        primary["section_hint_block_id"] = secondary.get("section_hint_block_id")
    primary.setdefault("merged_from", [])
    primary["merged_from"].append(str(secondary.get("table_id", "")))
    primary.setdefault("row_texts", [])
    primary["row_texts"].extend([str(item) for item in secondary.get("row_texts", [])])
    primary["toc_row_ratio"] = round(_table_toc_row_ratio(primary), 3)
    return primary


def _can_merge_table_fragments_on_same_page(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    if int(primary.get("page", 0)) != int(secondary.get("page", 0)):
        return False
    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    vertical_gap = secondary_bbox[1] - primary_bbox[3]
    if vertical_gap < -10 or vertical_gap > 52:
        return False
    overlap = _horizontal_overlap_ratio(primary_bbox, secondary_bbox)
    if overlap < 0.5:
        return False
    similarity = _column_similarity(
        list(primary.get("column_signature", [])),
        list(secondary.get("column_signature", [])),
    )
    if similarity < 0.68:
        return False
    if float(primary.get("toc_row_ratio", 0.0)) > 0.25 or float(secondary.get("toc_row_ratio", 0.0)) > 0.25:
        return False
    return True


def _merge_same_page_table_fragments(page_tables: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not page_tables:
        return [], 0
    sorted_tables = sorted(page_tables, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    merged_tables: list[dict[str, Any]] = [dict(sorted_tables[0])]
    merge_count = 0
    for table in sorted_tables[1:]:
        candidate = dict(table)
        previous = merged_tables[-1]
        if _can_merge_table_fragments_on_same_page(previous, candidate):
            merged_tables[-1] = _merge_two_table_fragments(previous, candidate)
            merge_count += 1
            continue
        merged_tables.append(candidate)
    return merged_tables, merge_count


def _normalized_table_title(table: dict[str, Any]) -> str:
    return _compact_text(str(table.get("title", "")))


def _can_merge_tables_with_same_title(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    if int(primary.get("page", 0)) != int(secondary.get("page", 0)):
        return False
    primary_title = _normalized_table_title(primary)
    secondary_title = _normalized_table_title(secondary)
    if not primary_title or not secondary_title or primary_title != secondary_title:
        return False

    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    vertical_gap = secondary_bbox[1] - primary_bbox[3]
    if vertical_gap < -12 or vertical_gap > 320:
        return False
    overlap = _horizontal_overlap_ratio(primary_bbox, secondary_bbox)
    if overlap < 0.4:
        return False
    similarity = _column_similarity(
        list(primary.get("column_signature", [])),
        list(secondary.get("column_signature", [])),
    )
    if similarity >= 0.45:
        return True
    # Allow merge for same title when fragments are vertically adjacent and almost same width.
    return overlap >= 0.92 and abs(vertical_gap) <= 10


def _merge_same_title_tables(page_tables: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not page_tables:
        return [], 0
    sorted_tables = sorted(page_tables, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    merged_tables: list[dict[str, Any]] = []
    merge_count = 0
    for table in sorted_tables:
        candidate = dict(table)
        if not merged_tables:
            merged_tables.append(candidate)
            continue
        previous = merged_tables[-1]
        if _can_merge_tables_with_same_title(previous, candidate):
            merged_tables[-1] = _merge_two_table_fragments(previous, candidate)
            merge_count += 1
            continue
        merged_tables.append(candidate)
    return merged_tables, merge_count


def _column_similarity(signature_a: list[float], signature_b: list[float]) -> float:
    if not signature_a or not signature_b:
        return 0.0
    tolerance = 0.07
    matched = 0
    for value in signature_a:
        if any(abs(value - other) <= tolerance for other in signature_b):
            matched += 1
    return matched / max(len(signature_a), len(signature_b))


def _header_similarity(header_a: list[dict[str, Any]], header_b: list[dict[str, Any]]) -> float:
    texts_a = [str(item.get("text", "")).strip().lower() for item in header_a if str(item.get("text", "")).strip()]
    texts_b = [str(item.get("text", "")).strip().lower() for item in header_b if str(item.get("text", "")).strip()]
    if not texts_a or not texts_b:
        return 0.0
    overlap = len(set(texts_a) & set(texts_b))
    return overlap / max(len(set(texts_a)), len(set(texts_b)))


def _stitch_cross_page_tables(
    table_asts: list[dict[str, Any]],
    page_heights: dict[int, float],
) -> int:
    if not table_asts:
        return 0
    stitched_count = 0
    sorted_tables = sorted(table_asts, key=lambda item: (item["page"], item["bbox"][1]))

    for previous, current in zip(sorted_tables, sorted_tables[1:]):
        if current["page"] != previous["page"] + 1:
            continue
        prev_page_height = max(1.0, page_heights.get(previous["page"], 1.0))
        curr_page_height = max(1.0, page_heights.get(current["page"], 1.0))
        prev_near_bottom = previous["bbox"][3] >= prev_page_height * 0.72
        curr_near_top = current["bbox"][1] <= curr_page_height * 0.28
        similarity = _column_similarity(previous["column_signature"], current["column_signature"])
        previous_has_title = bool(str(previous.get("title", "")).strip())
        previous_section_hint_text = str(previous.get("section_hint", ""))
        previous_is_glossary_context = _is_glossary_hint_text(previous_section_hint_text) or (
            str(previous.get("continuation_context", "")).strip().lower() == "glossary"
        )
        previous_has_section_hint = bool(previous_section_hint_text.strip()) or previous_is_glossary_context
        previous_col_count = int(previous.get("col_count", 0))
        current_col_count = int(current.get("col_count", 0))
        continuation_hint = current.get("continuation_hint")
        hint_matches_previous = (
            isinstance(continuation_hint, dict)
            and str(continuation_hint.get("table_id", "")).strip() == str(previous.get("table_id", "")).strip()
        )
        context_continuation = (
            (previous_has_title or previous_has_section_hint)
            and curr_near_top
            and abs(previous_col_count - current_col_count) <= 1
            and (
                (previous_is_glossary_context and similarity >= 0.22)
                or (not previous_is_glossary_context and similarity >= 0.28)
            )
        )
        is_likely_continuation = (
            (prev_near_bottom and curr_near_top)
            or similarity >= 0.82
            or ((previous_has_title or previous_has_section_hint) and curr_near_top and similarity >= 0.42)
            or context_continuation
            or hint_matches_previous
        )
        min_similarity = 0.22 if hint_matches_previous and previous_is_glossary_context else 0.28
        if similarity < min_similarity or not is_likely_continuation:
            continue
        if str(current.get("title", "")).strip():
            # A titled table on the current page is treated as a new table, not continuation.
            continue

        previous.setdefault("continued_to", [])
        previous["continued_to"].append(current["table_id"])
        current["continued_from"] = previous["table_id"]
        current["cross_page_similarity"] = round(similarity, 3)
        continuation_source: dict[str, Any] = {
            "source_table_id": str(previous.get("table_id", "")),
            "strategy": "cross_page_stitch",
            "similarity": round(similarity, 3),
            "inherited_fields": [],
        }
        if isinstance(continuation_hint, dict):
            hint_table_id = str(continuation_hint.get("table_id", "")).strip()
            if hint_table_id:
                continuation_source["hint_table_id"] = hint_table_id
            hint_similarity = continuation_hint.get("similarity")
            if isinstance(hint_similarity, (int, float)):
                continuation_source["hint_similarity"] = round(float(hint_similarity), 3)
        if not str(current.get("title", "")).strip() and str(previous.get("title", "")).strip():
            current["title"] = previous["title"]
            current["title_inherited"] = True
            continuation_source["inherited_fields"].append("title")

        header_similarity = _header_similarity(previous.get("header", []), current.get("header", []))
        if header_similarity < 0.45:
            current["header"] = previous.get("header", [])
            current["header_inherited"] = True
            continuation_source["inherited_fields"].append("header")
        current["continuation_source"] = continuation_source
        stitched_count += 1
    return stitched_count


def _renumber_table_ids(table_asts: list[dict[str, Any]]) -> None:
    if not table_asts:
        return
    ordered = sorted(
        table_asts,
        key=lambda item: (int(item.get("page", 0)), float(item.get("bbox", [0.0, 0.0, 0.0, 0.0])[1]), float(item.get("bbox", [0.0, 0.0, 0.0, 0.0])[0])),
    )
    id_mapping: dict[str, str] = {}
    for index, table in enumerate(ordered, start=1):
        old_id = str(table.get("table_id", "")).strip()
        new_id = f"tbl_{index:03d}"
        if old_id:
            id_mapping[old_id] = new_id
        table["table_id"] = new_id

    for table in table_asts:
        continued_from = str(table.get("continued_from", "")).strip()
        if continued_from:
            table["continued_from"] = id_mapping.get(continued_from, continued_from)

        continued_to = table.get("continued_to")
        if isinstance(continued_to, list):
            table["continued_to"] = [id_mapping.get(str(item), str(item)) for item in continued_to]

        merged_from = table.get("merged_from")
        if isinstance(merged_from, list):
            table["merged_from"] = [id_mapping.get(str(item), str(item)) for item in merged_from]

        continuation_hint = table.get("continuation_hint")
        if isinstance(continuation_hint, dict):
            hint_table_id = str(continuation_hint.get("table_id", "")).strip()
            if hint_table_id:
                continuation_hint["table_id"] = id_mapping.get(hint_table_id, hint_table_id)

        continuation_source = table.get("continuation_source")
        if isinstance(continuation_source, dict):
            source_table_id = str(continuation_source.get("source_table_id", "")).strip()
            if source_table_id:
                continuation_source["source_table_id"] = id_mapping.get(source_table_id, source_table_id)
            hint_table_id = str(continuation_source.get("hint_table_id", "")).strip()
            if hint_table_id:
                continuation_source["hint_table_id"] = id_mapping.get(hint_table_id, hint_table_id)

