from __future__ import annotations

from typing import Any


def _toc_block_sort_key(toc_block: dict[str, Any]) -> tuple[int, float, float]:
    bbox = list(toc_block.get("bbox", []) or [])
    return (
        int(toc_block.get("page", 0) or 0),
        float(bbox[1]) if len(bbox) == 4 else 0.0,
        float(bbox[0]) if len(bbox) == 4 else 0.0,
    )


def _toc_entry_sort_key(entry: dict[str, Any]) -> tuple[float, float]:
    bbox = list(entry.get("bbox", []) or [])
    return (
        float(bbox[1]) if len(bbox) == 4 else float(entry.get("sort_y0", 0.0) or 0.0),
        float(bbox[0]) if len(bbox) == 4 else float(entry.get("sort_x0", 0.0) or 0.0),
    )


def _next_toc_index(toc_blocks: list[dict[str, Any]]) -> int:
    highest = 0
    for toc_block in toc_blocks:
        toc_id = str(toc_block.get("toc_id") or "").strip()
        if not toc_id.startswith("toc_"):
            continue
        suffix = toc_id.split("_", 1)[1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest


def _toc_entry_source_block_ids(entry: dict[str, Any]) -> list[str]:
    block_ids: list[str] = []
    for raw_value in entry.get("source_block_ids", []) or []:
        normalized = str(raw_value or "").strip()
        if normalized and normalized not in block_ids:
            block_ids.append(normalized)
    single_block_id = str(entry.get("source_block_id") or "").strip()
    if single_block_id and single_block_id not in block_ids:
        block_ids.append(single_block_id)
    return block_ids


def _toc_entry_block_relation(
    toc_bbox: list[Any],
    candidate_bbox: list[Any],
) -> str:
    candidate_top = float(candidate_bbox[1])
    candidate_bottom = float(candidate_bbox[3])
    toc_top = float(toc_bbox[1])
    toc_bottom = float(toc_bbox[3])
    if candidate_bottom <= toc_top:
        return "above"
    if candidate_top >= toc_bottom:
        return "below"
    return "overlap"


def _last_toc_entry_outline_key(
    entries: list[dict[str, Any]],
) -> tuple[str, tuple[int, ...] | str] | None:
    for entry in reversed(entries):
        outline_key = _toc_outline_sort_key(str(entry.get("outline_index") or "").strip())
        if outline_key is not None:
            return outline_key
    return None


def _toc_entry_page_locator_key(entry: dict[str, Any]) -> tuple[str, int] | None:
    locator_kind = str(entry.get("page_locator_kind") or "")
    locator_value = entry.get("page_locator_value")
    if locator_kind in {"arabic", "roman"} and locator_value is not None:
        return locator_kind, int(locator_value)
    return None


def _toc_outline_coverage_ratio(entries: list[dict[str, Any]]) -> float:
    if not entries:
        return 0.0
    outlined = sum(1 for entry in entries if str(entry.get("outline_index") or "").strip())
    return outlined / len(entries)


def _first_root_outline_key(toc_block: dict[str, Any]) -> tuple[str, tuple[int, ...] | str] | None:
    root_keys = [
        _toc_outline_sort_key(str(entry.get("outline_index") or "").strip())
        for entry in toc_block.get("entries", []) or []
        if str(entry.get("outline_index") or "").strip() and int(entry.get("level", 1) or 1) <= 1
    ]
    root_keys = [key for key in root_keys if key is not None]
    if not root_keys:
        return None
    return root_keys[0]


def _first_toc_entry_outline_key(
    entries: list[dict[str, Any]],
) -> tuple[str, tuple[int, ...] | str] | None:
    for entry in entries:
        outline_key = _toc_outline_sort_key(str(entry.get("outline_index") or "").strip())
        if outline_key is not None:
            return outline_key
    return None


def _last_root_outline_key(toc_block: dict[str, Any]) -> tuple[str, tuple[int, ...] | str] | None:
    root_keys = [
        _toc_outline_sort_key(str(entry.get("outline_index") or "").strip())
        for entry in toc_block.get("entries", []) or []
        if str(entry.get("outline_index") or "").strip() and int(entry.get("level", 1) or 1) <= 1
    ]
    root_keys = [key for key in root_keys if key is not None]
    if not root_keys:
        return None
    return root_keys[-1]


def _toc_horizontal_overlap_ratio(
    previous_bbox: list[float] | tuple[float, float, float, float],
    current_bbox: list[float] | tuple[float, float, float, float],
) -> float:
    previous_left, _, previous_right, _ = [float(value) for value in previous_bbox]
    current_left, _, current_right, _ = [float(value) for value in current_bbox]
    overlap = min(previous_right, current_right) - max(previous_left, current_left)
    if overlap <= 0:
        return 0.0
    previous_width = max(1.0, previous_right - previous_left)
    current_width = max(1.0, current_right - current_left)
    return overlap / min(previous_width, current_width)


def _toc_outline_sort_key(outline_index: str) -> tuple[str, tuple[int, ...] | str] | None:
    candidate = str(outline_index or "").strip()
    if not candidate:
        return None
    if candidate.startswith("APPENDIX "):
        return ("appendix", candidate)
    if all(part.isdigit() for part in candidate.split(".") if part):
        segments = tuple(int(part) for part in candidate.split(".") if part)
        return ("numeric", segments)
    roman_value = _roman_to_int(candidate)
    if roman_value is not None:
        return ("roman", (roman_value,))
    if len(candidate) == 1 and candidate.isalpha():
        return ("alpha", (ord(candidate.upper()) - ord("A") + 1,))
    return None


def _first_known_page_locator(entries: list[dict[str, Any]]) -> tuple[str, int] | None:
    for entry in entries:
        locator_kind = str(entry.get("page_locator_kind") or "")
        locator_value = entry.get("page_locator_value")
        if locator_kind in {"arabic", "roman"} and locator_value is not None:
            return locator_kind, int(locator_value)
    return None


def _last_known_page_locator(entries: list[dict[str, Any]]) -> tuple[str, int] | None:
    for entry in reversed(entries):
        locator_kind = str(entry.get("page_locator_kind") or "")
        locator_value = entry.get("page_locator_value")
        if locator_kind in {"arabic", "roman"} and locator_value is not None:
            return locator_kind, int(locator_value)
    return None


def _roman_to_int(text: str) -> int | None:
    candidate = str(text or "").strip().upper()
    if not candidate:
        return None
    if any(char not in {"I", "V", "X", "L", "C", "D", "M"} for char in candidate):
        return None
    roman_values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }
    total = 0
    previous_value = 0
    for char in reversed(candidate):
        value = roman_values[char]
        if value < previous_value:
            total -= value
        else:
            total += value
            previous_value = value
    return total


def _build_toc_sequence_tree(
    sequence_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], int]:
    ordered_entries = sorted(
        [dict(entry) for entry in sequence_entries],
        key=lambda entry: int(entry.get("sequence_entry_index", 0) or 0),
    )
    node_by_sequence_index: dict[int, dict[str, Any]] = {}

    for entry in ordered_entries:
        sequence_entry_index = int(entry.get("sequence_entry_index", 0) or 0)
        if sequence_entry_index <= 0:
            continue
        node_by_sequence_index[sequence_entry_index] = {
            "sequence_entry_index": sequence_entry_index,
            "entry_index": int(entry.get("entry_index", 0) or 0),
            "page": int(entry.get("page", 0) or 0),
            "toc_id": entry.get("toc_id"),
            "outline_index": entry.get("outline_index"),
            "outline_depth": int(entry.get("outline_depth", 0) or 0),
            "text": entry.get("text"),
            "page_locator": entry.get("page_locator"),
            "page_locator_kind": entry.get("page_locator_kind"),
            "page_locator_value": entry.get("page_locator_value"),
            "level": int(entry.get("level", 1) or 1),
            "parent_sequence_entry_index": entry.get("parent_sequence_entry_index"),
            "section_anchor_sequence_entry_index": entry.get("section_anchor_sequence_entry_index"),
            "children": [],
        }

    root_nodes: list[dict[str, Any]] = []
    for sequence_entry_index in sorted(node_by_sequence_index):
        node = node_by_sequence_index[sequence_entry_index]
        parent_sequence_entry_index = node.get("parent_sequence_entry_index")
        if parent_sequence_entry_index:
            parent_node = node_by_sequence_index.get(int(parent_sequence_entry_index))
            if parent_node:
                parent_node["children"].append(node)
                continue
        root_nodes.append(node)

    leaf_entry_indices: list[int] = []
    max_branching_factor = 0
    for node in node_by_sequence_index.values():
        child_count = len(node["children"])
        node["child_count"] = child_count
        node["has_children"] = child_count > 0
        max_branching_factor = max(max_branching_factor, child_count)
        if child_count == 0:
            leaf_entry_indices.append(int(node["sequence_entry_index"]))

    return root_nodes, leaf_entry_indices, max_branching_factor
