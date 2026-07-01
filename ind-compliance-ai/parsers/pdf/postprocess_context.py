from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

_MODULE_SIGNAL_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("M1", "module_1", re.compile(r"(?<![a-z0-9])(?:module[\s._/-]*1|m[\s._/-]*1)(?![a-z0-9])", re.IGNORECASE)),
    ("M2", "module_2", re.compile(r"(?<![a-z0-9])(?:module[\s._/-]*2|m[\s._/-]*2)(?![a-z0-9])", re.IGNORECASE)),
    (
        "M3",
        "module_3",
        re.compile(
            r"(?<![a-z0-9])(?:module[\s._/-]*3|m[\s._/-]*3)(?![a-z0-9])|(?<!\d)3\.2(?:\.[sp])?(?!\d)",
            re.IGNORECASE,
        ),
    ),
    ("M4", "module_4", re.compile(r"(?<![a-z0-9])(?:module[\s._/-]*4|m[\s._/-]*4)(?![a-z0-9])", re.IGNORECASE)),
    ("M5", "module_5", re.compile(r"(?<![a-z0-9])(?:module[\s._/-]*5|m[\s._/-]*5)(?![a-z0-9])", re.IGNORECASE)),
)


def _infer_module_context(path: Path) -> dict[str, Any] | None:
    source_values = [str(path.name or "").strip(), str(path).strip()]
    for raw_value in source_values:
        normalized = raw_value.lower().replace("\\", "/")
        for module_label, signal_hit, pattern in _MODULE_SIGNAL_PATTERNS:
            if pattern.search(normalized):
                return {
                    "module_label": module_label,
                    "module_signal_hit": signal_hit,
                    "module_signal_source": "filename" if raw_value == str(path.name or "").strip() else "source_path",
                    "anchor_source": "document_classification",
                    "anchor_confidence": 0.9,
                }
    return None


def _merge_section_context(
    section_context: dict[str, Any] | None,
    module_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not section_context and not module_context:
        return None
    merged: dict[str, Any] = {}
    if module_context:
        merged.update(module_context)
    if section_context:
        merged.update(section_context)
    return merged


def _resolve_section_context_for_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    page_heading_anchors: list[dict[str, Any]],
    active_section_context: dict[str, Any] | None,
    module_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    current_y = float(bbox[1]) if len(bbox) >= 2 else float("inf")
    applicable_anchor: dict[str, Any] | None = None
    for anchor in page_heading_anchors:
        anchor_bbox = list(anchor.get("anchor_bbox", []))
        anchor_y = float(anchor_bbox[1]) if len(anchor_bbox) >= 2 else float("-inf")
        if anchor_y <= current_y:
            applicable_anchor = anchor
        else:
            break
    return _merge_section_context(applicable_anchor or active_section_context, module_context)


def _attach_section_context(
    target: dict[str, Any],
    section_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if section_context:
        target["section_context"] = deepcopy(section_context)
    return target
