# Version: v1.0.12
# Optimization Summary:
# - Add audit-oriented table quality metrics to parse metadata.
# - Expose low-confidence/review-required/continuation counts and continuation similarity stats.
# - Persist active parser threshold snapshot in metadata for traceable auditing.
# - Add non-destructive missing-content diagnostic summary counters.
# - Repair semantic-only cross-page boundary row splits after rebuilding raw and
#   semantic row views.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

from parsers.common.atomic_fact_extractor import extract_atomic_facts

from .settings import get_pdf_parser_settings
from .shared import _Word, _bbox_to_list, _clean_text, _compact_text
from .table_modules.postprocess import (
    _align_header_row,
    _refresh_row_texts_from_grid,
    compact_leading_key_carry_forward_rows,
    compact_vector_ocr_placeholder_tail_column_pairs,
    compact_vector_ocr_sparse_anchor_rows,
    repair_cross_page_boundary_row_splits,
)
from .tables import (
    _annotate_toc_diagnostics,
    _build_toc_text_block_entries,
    _parent_outline_index,
    _populate_toc_block_from_raw_entries,
)
from .text_blocks import _suppress_table_text_blocks
from .text_blocks import order_text_blocks_for_reading
from .types import PDF_PARSER_HINT, PdfPipelineState

_DISPLAY_EQUATION_FUNCTION_RE = re.compile(
    r"\b(?:relu|softmax|sigmoid|tanh|max|min|argmax|argmin|exp|log|sin|cos|drop)\b",
    re.IGNORECASE,
)
_DISPLAY_EQUATION_MARKER_RE = re.compile(r"^\((\d{1,3})\)$")
_DISPLAY_EQUATION_TERMINAL_PUNCTUATION = {
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "\u3002",
    "\uff0c",
    "\uff1b",
    "\uff1a",
    "\uff01",
    "\uff1f",
}
_DISPLAY_EQUATION_TRAILING_TRIM_CHARS = {'"', "'", "\u201c", "\u201d", "\u2018", "\u2019", "`"}
_DISPLAY_EQUATION_FRAGMENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "when",
    "where",
    "which",
    "with",
}
_DISPLAY_EQUATION_CLUSTER_PROSE_CUES = (
    "where",
    "which",
    "means",
    "samples",
    "sample",
    "labels",
    "label",
    "dataset",
    "training",
    "class",
    "classes",
    "kernel",
    "function",
    "parameter",
    "probability",
    "containing",
    "respectively",
)
_DISPLAY_EQUATION_HARD_PROSE_CUES = (
    "algorithm",
    "initialization",
    "output",
    "compute",
    "repeat",
    "update",
    "until",
    "end if",
    "theorem",
    "proof",
    "gradient descent",
    "feature weights",
    "training samples",
    "sample labels",
    "stopping criterion",
    "using eq.",
    "eq.",
    "pseudo code",
    "global minimizer",
    "initial point",
    "point of f v",
    "line search",
    "direction",
    "constraint",
    "direct gradient method",
)
_ALGORITHM_TITLE_RE = re.compile(r"^algorithm\s+(?P<number>\d+)(?:[.:]|\b)", re.IGNORECASE)
_ALGORITHM_SECTION_HEADING_RE = re.compile(r"^\d+(?:\.\d+)+\.\s+\S")
_ALGORITHM_STEP_RE = re.compile(
    r"^(?:\d+\.\s*|initialization:|output:|input:|repeat\b|until\b|for\b|if\b|else\b|elseif\b|end if\b|end\b|set\b|update\b|compute\b|calculate\b)",
    re.IGNORECASE,
)
_ALGORITHM_BOUNDARY_PREFIXES = (
    "this section",
    "there are",
    "in this section",
    "the experiments",
    "the proposed",
    "for algorithm",
    "for comparison",
)
_ALGORITHM_TERMINAL_CHARS = {".", ";", ":", "!", "?", "\u3002", "\uff1b", "\uff1a", "\uff01", "\uff1f"}
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
_NUMBERED_SECTION_HEADING_RE = re.compile(
    r"^(?P<outline>\d+(?:\.\d+)*)(?:\.)?\s+(?P<title>\S.*)$"
)
_DATE_LIKE_HEADING_SUFFIX_RE = re.compile(r"^(?:年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?|年\s*\d{1,2}\s*月|年)$")


def _looks_like_equation_false_positive_text(text: str) -> bool:
    lowered = _clean_text(text).lower()
    if not lowered:
        return False
    if any(cue in lowered for cue in _DISPLAY_EQUATION_HARD_PROSE_CUES):
        return True
    if re.match(r"^\d+\.\s", lowered):
        return True
    return lowered.startswith(("theorem", "proof", "algorithm", "initialization", "output", "set ", "compute ", "repeat", "update ", "until "))


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


def _build_heading_section_anchor(
    text_block: dict[str, Any],
    module_context: dict[str, Any] | None,
    active_section_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if str(text_block.get("block_type", "text") or "text") != "text":
        return None
    text = _clean_text(str(text_block.get("text", "")))
    if not text:
        return None
    semantic_role = str(text_block.get("semantic_role", "") or "").strip()
    if semantic_role == "reference_heading":
        return _merge_section_context(
            {
                "outline_index": "references",
                "outline_path": "references",
                "section_title": text,
                "section_level": 1,
                "anchor_source": "reference_heading",
                "anchor_confidence": 0.95,
                "anchor_page": int(text_block.get("page", 0) or 0),
                "anchor_bbox": list(text_block.get("bbox", [])),
            },
            module_context,
        )
    if semantic_role == "reference_entry":
        return None
    match = _NUMBERED_SECTION_HEADING_RE.match(text)
    if not match:
        return None
    outline_index = str(match.group("outline") or "").strip().rstrip(".")
    section_title = _clean_text(str(match.group("title") or ""))
    if not outline_index or not section_title:
        return None
    if _looks_like_date_heading_false_positive(text, outline_index, section_title):
        return None
    if _looks_like_regressive_root_heading_candidate(outline_index, active_section_context):
        return None
    return _merge_section_context(
        {
            "outline_index": outline_index,
            "outline_path": outline_index,
            "section_title": section_title,
            "section_level": outline_index.count(".") + 1,
            "anchor_source": "heading",
            "anchor_confidence": 0.97,
            "anchor_page": int(text_block.get("page", 0) or 0),
            "anchor_bbox": list(text_block.get("bbox", [])),
        },
        module_context,
    )


def _looks_like_regressive_root_heading_candidate(
    outline_index: str,
    active_section_context: dict[str, Any] | None,
) -> bool:
    candidate = str(outline_index or "").strip()
    if not candidate or "." in candidate:
        return False
    if not candidate.isdigit():
        return False
    active_outline_index = str((active_section_context or {}).get("outline_index") or "").strip()
    if not active_outline_index:
        return False
    active_root = active_outline_index.split(".", 1)[0]
    if not active_root.isdigit():
        return False
    return int(candidate) < int(active_root)


def _looks_like_date_heading_false_positive(
    full_text: str,
    outline_index: str,
    section_title: str,
) -> bool:
    normalized_outline = str(outline_index or "").strip()
    normalized_title = _clean_text(str(section_title or ""))
    normalized_full_text = _clean_text(str(full_text or ""))
    if not normalized_outline.isdigit():
        return False
    if len(normalized_outline) != 4:
        return False
    year_value = int(normalized_outline)
    if year_value < 1900 or year_value > 2099:
        return False
    if _DATE_LIKE_HEADING_SUFFIX_RE.match(normalized_title):
        return True
    return bool(re.fullmatch(r"\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?", normalized_full_text))


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


def _extract_algorithm_ref(text: str) -> str | None:
    match = _ALGORITHM_TITLE_RE.match(_clean_text(text))
    if not match:
        return None
    return f"Algorithm {match.group('number')}"


def _looks_like_algorithm_title(text: str) -> bool:
    return _extract_algorithm_ref(text) is not None


def _looks_like_algorithm_section_heading(text: str) -> bool:
    return bool(_ALGORITHM_SECTION_HEADING_RE.match(_clean_text(text)))


def _looks_like_algorithm_body_line(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if _ALGORITHM_STEP_RE.match(cleaned):
        return True
    if "algorithm " in lowered and "=" in cleaned:
        return True
    if "algorithm " in lowered and "(" in cleaned:
        return True
    if "=" in cleaned and len(re.findall(r"[A-Za-z]+", cleaned)) <= 10:
        return True
    return False


def _looks_like_algorithm_continuation_seed(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned or cleaned.isdigit():
        return False
    if _looks_like_algorithm_section_heading(cleaned):
        return False
    if _looks_like_algorithm_body_line(cleaned):
        return True
    alpha_tokens = re.findall(r"[A-Za-z]+", cleaned)
    if "=" in cleaned and len(alpha_tokens) <= 8:
        return True
    return 1 <= len(alpha_tokens) <= 5 and any(char.isalpha() for char in cleaned)


def _looks_like_algorithm_boundary_text(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if _looks_like_algorithm_section_heading(cleaned):
        return True
    if _looks_like_algorithm_title(cleaned):
        return True
    return lowered.startswith(_ALGORITHM_BOUNDARY_PREFIXES)


def _algorithm_same_lane(
    seed_block: dict[str, Any],
    candidate_block: dict[str, Any],
) -> bool:
    seed_lane = str(seed_block.get("layout_lane", "") or "").strip()
    candidate_lane = str(candidate_block.get("layout_lane", "") or "").strip()
    if not seed_lane or not candidate_lane:
        return True
    return seed_lane == candidate_lane


def _algorithm_vertical_gap(
    previous_block: dict[str, Any],
    current_block: dict[str, Any],
) -> float:
    previous_bbox = list(previous_block.get("bbox", []))
    current_bbox = list(current_block.get("bbox", []))
    if len(previous_bbox) < 4 or len(current_bbox) < 4:
        return 999.0
    return float(current_bbox[1]) - float(previous_bbox[3])


def _algorithm_line_is_opener(text: str) -> bool:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    return _looks_like_algorithm_title(cleaned) or lowered.startswith(
        ("initialization:", "output:", "input:", "repeat", "until", "for ", "if ", "set ", "update ", "compute ", "calculate ")
    ) or bool(re.match(r"^\d+\.\s*$", cleaned))


def _algorithm_line_is_open_ended(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    if cleaned.endswith("-"):
        return True
    return cleaned[-1] not in _ALGORITHM_TERMINAL_CHARS


def _can_absorb_algorithm_line(
    consumed_blocks: list[dict[str, Any]],
    candidate_block: dict[str, Any],
) -> bool:
    if not consumed_blocks:
        return False
    candidate_text = _clean_text(str(candidate_block.get("text", "")))
    if not candidate_text or _looks_like_algorithm_boundary_text(candidate_text):
        return False
    previous_block = consumed_blocks[-1]
    previous_text = _clean_text(str(previous_block.get("text", "")))
    if not _algorithm_same_lane(consumed_blocks[0], candidate_block):
        return False
    max_gap = max(
        16.0,
        _bbox_height(list(previous_block.get("bbox", []))) * 1.75,
        _bbox_height(list(candidate_block.get("bbox", []))) * 1.75,
    )
    if _algorithm_vertical_gap(previous_block, candidate_block) > max_gap:
        return False
    if _looks_like_algorithm_body_line(candidate_text):
        return True
    if _algorithm_line_is_opener(previous_text):
        return True
    if _algorithm_line_is_open_ended(previous_text):
        return True
    previous_bbox = list(previous_block.get("bbox", []))
    candidate_bbox = list(candidate_block.get("bbox", []))
    if len(previous_bbox) >= 4 and len(candidate_bbox) >= 4:
        if float(candidate_bbox[0]) >= float(previous_bbox[0]) + 10.0:
            return True
    return False


def _build_algorithm_text_block(
    consumed_blocks: list[dict[str, Any]],
    *,
    algorithm_id: str,
    algorithm_ref: str,
    title: str,
    continued_from_previous_page: bool,
    continues_to_next_page: bool,
) -> dict[str, Any]:
    bbox = _bbox_union([list(block.get("bbox", [])) for block in consumed_blocks])
    content_lines = [_clean_text(str(block.get("text", ""))) for block in consumed_blocks if _clean_text(str(block.get("text", "")))]
    source_block_ids = [str(block.get("block_id", "")).strip() for block in consumed_blocks if str(block.get("block_id", "")).strip()]
    source_block_indices: list[int] = []
    for block in consumed_blocks:
        source_block_indices.extend(list(block.get("source_block_indices", []) or []))

    algorithm_block = {
        "block_type": "algorithm",
        "block_id": algorithm_id,
        "algorithm_id": algorithm_id,
        "page": int(consumed_blocks[0].get("page", 0) or 0),
        "bbox": bbox,
        "text": "\n".join(content_lines).strip(),
        "content_text": "\n".join(content_lines).strip(),
        "semantic_role": "algorithm_pseudocode",
        "unit_role": "algorithm",
        "algorithm_ref": algorithm_ref,
        "title": title,
        "source": "algorithm-grouping",
        "source_block_ids": source_block_ids,
        "source_block_indices": sorted(set(int(item) for item in source_block_indices)),
        "line_count": len(content_lines),
        "lines": content_lines,
        "continued_from_previous_page": continued_from_previous_page,
        "continues_to_next_page": continues_to_next_page,
    }
    layout_lane = str(consumed_blocks[0].get("layout_lane", "") or "").strip()
    if layout_lane:
        algorithm_block["layout_lane"] = layout_lane
    layout_mode = str(consumed_blocks[0].get("layout_mode", "") or "").strip()
    if layout_mode:
        algorithm_block["layout_mode"] = layout_mode
    if consumed_blocks[0].get("layout_confidence") is not None:
        algorithm_block["layout_confidence"] = float(consumed_blocks[0].get("layout_confidence", 0.0) or 0.0)
    return algorithm_block


def _build_algorithm_projection(text_block: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm_id": str(text_block.get("algorithm_id") or text_block.get("block_id") or "").strip(),
        "page": int(text_block.get("page", 0) or 0),
        "bbox": list(text_block.get("bbox", [])),
        "algorithm_ref": str(text_block.get("algorithm_ref", "") or "").strip(),
        "title": str(text_block.get("title", "") or "").strip(),
        "content_text": str(text_block.get("content_text", "") or "").strip(),
        "lines": list(text_block.get("lines", []) or []),
        "line_count": int(text_block.get("line_count", 0) or 0),
        "continued_from_previous_page": bool(text_block.get("continued_from_previous_page", False)),
        "continues_to_next_page": bool(text_block.get("continues_to_next_page", False)),
        "source_block_ids": list(text_block.get("source_block_ids", []) or []),
        "semantic_role": "algorithm_pseudocode",
    }


def _promote_algorithm_blocks(
    text_blocks: list[dict[str, Any]],
    *,
    page_number: int,
    page_height: float,
    next_algorithm_index: int,
    carry_state: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, int]:
    if not text_blocks:
        return [], [], None, next_algorithm_index

    projected: list[dict[str, Any]] = []
    page_algorithm_blocks: list[dict[str, Any]] = []
    idx = 0
    active_carry_state = dict(carry_state) if carry_state else None
    top_continuation_limit = min(160.0, page_height * 0.24)

    while idx < len(text_blocks):
        block = text_blocks[idx]
        if str(block.get("block_type", "text") or "text") != "text":
            projected.append(dict(block))
            idx += 1
            continue

        text = _clean_text(str(block.get("text", "")))
        bbox = list(block.get("bbox", []))
        y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0

        default_ref = None
        default_title = ""
        continued_from_previous_page = False
        if (
            active_carry_state
            and y0 <= top_continuation_limit
            and _looks_like_algorithm_continuation_seed(text)
            and not _looks_like_algorithm_title(text)
        ):
            default_ref = str(active_carry_state.get("algorithm_ref", "") or "").strip() or None
            default_title = str(active_carry_state.get("title", "") or "").strip()
            continued_from_previous_page = True
        elif active_carry_state and y0 > top_continuation_limit:
            active_carry_state = None

        if not default_ref and not _looks_like_algorithm_title(text):
            projected.append(dict(block))
            idx += 1
            continue

        consumed_blocks: list[dict[str, Any]] = []
        algorithm_ref = default_ref
        title = default_title
        cursor = idx

        while cursor < len(text_blocks):
            candidate = text_blocks[cursor]
            if str(candidate.get("block_type", "text") or "text") != "text":
                break
            candidate_text = _clean_text(str(candidate.get("text", "")))
            if not candidate_text:
                break
            if not consumed_blocks:
                if _looks_like_algorithm_title(candidate_text):
                    algorithm_ref = _extract_algorithm_ref(candidate_text)
                    title = candidate_text
                    consumed_blocks.append(dict(candidate))
                    cursor += 1
                    continue
                if default_ref and _looks_like_algorithm_continuation_seed(candidate_text):
                    consumed_blocks.append(dict(candidate))
                    cursor += 1
                    continue
                break

            if _looks_like_algorithm_title(candidate_text):
                break
            if not _can_absorb_algorithm_line(consumed_blocks, candidate):
                break
            consumed_blocks.append(dict(candidate))
            cursor += 1

        if not consumed_blocks or not algorithm_ref:
            projected.append(dict(block))
            idx += 1
            continue

        last_bbox = list(consumed_blocks[-1].get("bbox", []))
        last_bottom = float(last_bbox[3]) if len(last_bbox) >= 4 else 0.0
        continues_to_next_page = cursor >= len(text_blocks) and last_bottom >= page_height - max(72.0, page_height * 0.1)
        algorithm_id = f"alg_{next_algorithm_index:03d}"
        next_algorithm_index += 1

        algorithm_block = _build_algorithm_text_block(
            consumed_blocks,
            algorithm_id=algorithm_id,
            algorithm_ref=algorithm_ref,
            title=title or (algorithm_ref if continued_from_previous_page else ""),
            continued_from_previous_page=continued_from_previous_page,
            continues_to_next_page=continues_to_next_page,
        )
        projected.append(algorithm_block)
        page_algorithm_blocks.append(_build_algorithm_projection(algorithm_block))
        active_carry_state = (
            {
                "algorithm_ref": algorithm_ref,
                "title": title,
            }
            if continues_to_next_page
            else None
        )
        idx = cursor

    return projected, page_algorithm_blocks, active_carry_state, next_algorithm_index

_LITERATURE_AUTHOR_TOKEN_RE = re.compile(
    r"\b[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){0,2}\d(?:,\d+)*(?:[†‡*])?"
)
_PERSON_NAME_LINE_RE = re.compile(r"^[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3}$")
_AFFILIATION_KEYWORDS = (
    "department",
    "school",
    "college",
    "university",
    "hospital",
    "institute",
    "academy",
    "laboratory",
    "laboratories",
    "lab",
    "center",
    "centre",
    "faculty",
    "clinic",
)
_REFERENCE_HEADING_NORMALIZED = {
    "references",
    "bibliography",
    "workscited",
    "literaturecited",
}
_REFERENCE_STOP_HEADING_NORMALIZED = {
    "publishersnote",
    "publishernote",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "appendices",
    "supplementaryinformation",
    "supplementalmaterial",
    "supplementarymaterial",
    "footnotes",
    "notes",
}
_REFERENCE_ENTRY_INLINE_RE = re.compile(r"^\[?(?P<number>\d{1,3})\]?(?:[.)])\s+\S")
_REFERENCE_ENTRY_STANDALONE_RE = re.compile(r"^\[?(?P<number>\d{1,3})\]?(?:[.)])?$")
_REFERENCE_CITATION_CUE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|doi|arxiv|pp?\.\s*\d|vol\.\s*\d|proceedings|conference|journal|trans\.|intell syst|bioinformatics|appl sci|learn cybern|bmc",
    re.IGNORECASE,
)
_REFERENCE_AUTHOR_LIST_RE = re.compile(
    r"^[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+)?(?:,\s*[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+)?){1,}"
)


_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_LITERATURE_AUTHOR_LINE_HINT_RE = re.compile(r"[,*†‡]|(?:\b\w+\d(?:,\d)*)")


def build_pdf_parse_result(path: Path, state: PdfPipelineState) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    text_bounding_boxes: list[dict[str, Any]] = []
    image_blocks: list[dict[str, Any]] = []
    algorithm_blocks: list[dict[str, Any]] = []
    equation_blocks: list[dict[str, Any]] = []
    content_evidence: list[dict[str, Any]] = []
    content_units: list[dict[str, Any]] = []
    document_ast_pages: list[dict[str, Any]] = []
    analysis_page_texts: list[str] = []
    total_table_text_suppressed = 0

    _promote_cross_page_toc_edge_fragments(state)
    for page_payload in state.page_payloads:
        _promote_page_text_toc_fragments(page_payload)
    _stitch_cross_page_toc_sequences(state.toc_nodes)

    reference_state = {
        "active": False,
        "entry_open": False,
        "reference_number": None,
        "entry_index": 0,
    }
    module_context = _infer_module_context(path)
    next_algorithm_index = 1
    algorithm_carry_state: dict[str, Any] | None = None
    active_section_context: dict[str, Any] | None = _merge_section_context(None, module_context)
    for page_payload in state.page_payloads:
        page_number = int(page_payload["page_number"])
        ordered_analysis_blocks = order_text_blocks_for_reading(
            page_payload["text_blocks"],
            page_payload.get("layout_profile"),
        )
        analysis_text = "\n".join(item["text"] for item in ordered_analysis_blocks).strip()
        if analysis_text:
            analysis_page_texts.append(analysis_text)

        page_toc_blocks = page_payload.get("toc_blocks", [])
        visible_text_blocks, table_text_suppressed_count = _suppress_table_text_blocks(
            ordered_analysis_blocks,
            page_payload["tables"] + page_toc_blocks,
        )
        total_table_text_suppressed += table_text_suppressed_count
        projected_text_blocks = _promote_display_equation_blocks(
            visible_text_blocks,
            page_width=float(page_payload["width"]),
            page_words=list(page_payload.get("page_words", []) or []),
        )
        projected_text_blocks, page_algorithm_blocks, algorithm_carry_state, next_algorithm_index = _promote_algorithm_blocks(
            projected_text_blocks,
            page_number=page_number,
            page_height=float(page_payload["height"]),
            next_algorithm_index=next_algorithm_index,
            carry_state=algorithm_carry_state,
        )
        reference_state = _annotate_reference_section_text_blocks(
            projected_text_blocks,
            page_height=float(page_payload["height"]),
            reference_state=reference_state,
        )
        algorithm_blocks.extend(page_algorithm_blocks)
        page_equation_blocks = [
            _build_equation_projection(text_block)
            for text_block in projected_text_blocks
            if str(text_block.get("block_type", "text") or "text") == "equation"
        ]
        equation_blocks.extend(page_equation_blocks)

        page_text = "\n".join(item["text"] for item in projected_text_blocks).strip()
        text_page_nodes: list[dict[str, Any]] = []
        other_page_nodes: list[dict[str, Any]] = []
        page_heading_anchors: list[dict[str, Any]] = []
        page_text_evidence: list[dict[str, Any]] = []
        current_page_section_context = active_section_context

        for text_block in projected_text_blocks:
            heading_anchor = _build_heading_section_anchor(
                text_block,
                module_context,
                current_page_section_context,
            )
            if heading_anchor is not None:
                text_block["semantic_role"] = str(text_block.get("semantic_role") or "section_heading")
                text_block["unit_role"] = "section_heading"
                current_page_section_context = deepcopy(heading_anchor)
                active_section_context = deepcopy(heading_anchor)
                page_heading_anchors.append(deepcopy(heading_anchor))
            applicable_section_context = _resolve_section_context_for_bbox(
                list(text_block.get("bbox", [])),
                page_heading_anchors=page_heading_anchors,
                active_section_context=current_page_section_context,
                module_context=module_context,
            )
            text_evidence = _build_text_content_evidence(text_block)
            _attach_section_context(text_evidence, applicable_section_context)
            page_text_evidence.append(text_evidence)
            bbox = text_block["bbox"]
            text_bounding_boxes.append(
                {
                    "id": text_block["block_id"],
                    "page_number": page_number,
                    "text": text_block["text"],
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                    "block_type": str(text_block.get("block_type", "text") or "text"),
                    "semantic_role": str(text_block.get("semantic_role", "") or ""),
                }
            )
            page_node = {
                "block_type": str(text_block.get("block_type", "text") or "text"),
                "block_id": text_block["block_id"],
                "page": page_number,
                "bbox": bbox,
                "text": text_block["text"],
                "source": text_block.get("source", "text-layer"),
                "semantic_role": str(text_block.get("semantic_role", "") or ""),
            }
            _attach_section_context(page_node, applicable_section_context)
            text_page_nodes.append(page_node)
        _refine_literature_front_matter_text_evidence(
            page_text_evidence,
            page_height=float(page_payload["height"]),
        )
        content_evidence.extend(page_text_evidence)

        for image_block in page_payload["images"]:
            bbox = image_block["bbox"]
            image_blocks.append(image_block)
            text_bounding_boxes.append(
                {
                    "id": image_block["image_id"],
                    "page_number": page_number,
                    "text": f"[IMAGE] {image_block.get('title', '')}".strip(),
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                }
            )
            image_node = {
                "block_type": "image",
                "block_id": image_block["image_id"],
                "page": page_number,
                "bbox": bbox,
                "image_id": image_block["image_id"],
                "figure_ref": image_block.get("figure_ref"),
                "title": image_block.get("title", ""),
                "text_recovery": image_block.get("text_recovery", {}),
                "image_kind_guess": image_block.get("image_kind_guess"),
                "content_text": image_block.get("content_text", ""),
            }
            image_section_context = _resolve_section_context_for_bbox(
                bbox,
                page_heading_anchors=page_heading_anchors,
                active_section_context=current_page_section_context,
                module_context=module_context,
            )
            _attach_section_context(image_node, image_section_context)
            other_page_nodes.append(image_node)
            image_evidence = _build_image_content_evidence(image_block)
            _attach_section_context(image_evidence, image_section_context)
            content_evidence.append(image_evidence)

        for table_ast in page_payload["tables"]:
            bbox = table_ast["bbox"]
            text_bounding_boxes.append(
                {
                    "id": table_ast["table_id"],
                    "page_number": page_number,
                    "text": f"[TABLE] {table_ast['table_id']}",
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                }
            )
            table_node = {
                "block_type": "table",
                "block_id": table_ast["table_id"],
                "page": page_number,
                "bbox": bbox,
                "table_id": table_ast["table_id"],
                "column_hash": table_ast.get("column_hash"),
                "title": table_ast.get("title", ""),
            }
            table_section_context = _resolve_section_context_for_bbox(
                bbox,
                page_heading_anchors=page_heading_anchors,
                active_section_context=current_page_section_context,
                module_context=module_context,
            )
            _attach_section_context(table_node, table_section_context)
            other_page_nodes.append(table_node)
            table_evidence = _build_table_content_evidence(table_ast)
            _attach_section_context(table_evidence, table_section_context)
            content_evidence.append(table_evidence)

        for toc_block in page_toc_blocks:
            bbox = toc_block["bbox"]
            toc_id = toc_block["toc_id"]
            toc_title = toc_block.get("title") or toc_block.get("toc_title") or ""
            toc_diagnostics = toc_block.get("toc_diagnostics") or {}
            text_bounding_boxes.append(
                {
                    "id": toc_id,
                    "page_number": page_number,
                    "text": f"[TOC] {toc_title}".strip(),
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                }
            )
            toc_node = {
                "block_type": "toc",
                "block_id": toc_id,
                "page": page_number,
                "bbox": bbox,
                "toc_id": toc_id,
                "source_candidate_id": toc_block.get("source_candidate_id"),
                "title": toc_title,
                "title_inferred": bool(toc_block.get("title_inferred", False)),
                "entry_count": int(toc_block.get("entry_count", 0) or 0),
                "raw_entry_row_count": int(toc_block.get("raw_entry_row_count", 0) or 0),
                "wrapped_entry_count": int(toc_block.get("wrapped_entry_count", 0) or 0),
                "max_entry_level": int(toc_block.get("max_entry_level", 0) or 0),
                "missing_page_locator_count": int(toc_block.get("missing_page_locator_count", 0) or 0),
                "review_required": bool(toc_diagnostics.get("review_required", False)),
                "review_item_count": int(toc_diagnostics.get("review_item_count", 0) or 0),
                "aggregated_review_item_count": int(toc_diagnostics.get("aggregated_review_item_count", 0) or 0),
            }
            toc_section_context = _merge_section_context(
                {
                    "toc_sequence_id": toc_block.get("toc_sequence_id"),
                    "section_title": toc_title or None,
                    "anchor_source": "toc",
                    "anchor_confidence": 0.98,
                    "anchor_page": page_number,
                    "anchor_bbox": list(bbox),
                },
                module_context,
            )
            _attach_section_context(toc_node, toc_section_context)
            other_page_nodes.append(toc_node)
            toc_evidence = _build_toc_content_evidence(toc_block)
            _attach_section_context(toc_evidence, toc_section_context)
            content_evidence.append(toc_evidence)

        page_bbox_nodes = _merge_page_nodes_with_text_reading_order(text_page_nodes, other_page_nodes)
        document_ast_pages.append({"page": page_number, "blocks": page_bbox_nodes})
        pages.append(
            {
                "page_number": page_number,
                "width": float(page_payload["width"]),
                "height": float(page_payload["height"]),
                "text": page_text,
                "block_count": len(projected_text_blocks),
                "image_count": len(page_payload["images"]),
                "algorithm_count": len(page_algorithm_blocks),
                "equation_count": len(page_equation_blocks),
                "table_count": len(page_payload["tables"]),
                "toc_count": len(page_toc_blocks),
                "layout_mode": str((page_payload.get("layout_profile") or {}).get("mode", "single_column")),
                "layout_confidence": float((page_payload.get("layout_profile") or {}).get("confidence", 0.0) or 0.0),
                "semantic_merge_count": int(page_payload["semantic_merge_count"]),
                "image_text_recovered_count": int(page_payload["image_text_recovered_count"]),
                "header_footer_filtered_count": int(page_payload.get("header_footer_filtered", 0)),
                "table_text_suppressed_count": table_text_suppressed_count,
                "table_fragment_merge_count": int(page_payload.get("table_fragment_merge_count", 0)),
            }
        )

    settings = get_pdf_parser_settings()
    cfg = settings.cross_page_stitching
    table_policy = settings.table_content_policy
    cfg_snapshot = {
        "prev_near_bottom_ratio": cfg.prev_near_bottom_ratio,
        "curr_near_top_ratio": cfg.curr_near_top_ratio,
        "hint_glossary_similarity_min": cfg.hint_glossary_similarity_min,
        "default_similarity_min": cfg.default_similarity_min,
        "high_similarity_link_threshold": cfg.high_similarity_link_threshold,
        "title_context_similarity_threshold": cfg.title_context_similarity_threshold,
        "context_overlap_min": cfg.context_overlap_min,
        "overlap_min_without_hint": cfg.overlap_min_without_hint,
        "low_similarity_guard_threshold": cfg.low_similarity_guard_threshold,
        "low_similarity_guard_overlap_min": cfg.low_similarity_guard_overlap_min,
    }

    table_nodes = state.table_asts
    toc_nodes = state.toc_nodes
    for table in table_nodes:
        _align_header_row(table)
        _refresh_row_texts_from_grid(table)
        compact_vector_ocr_placeholder_tail_column_pairs(table)
        compact_vector_ocr_sparse_anchor_rows(table)
    cross_page_boundary_row_merge_count = repair_cross_page_boundary_row_splits(table_nodes)
    table_lookup = {
        str(table.get("table_id", "")).strip(): table
        for table in table_nodes
        if str(table.get("table_id", "")).strip()
    }
    for table in table_nodes:
        compact_leading_key_carry_forward_rows(table, table_lookup=table_lookup)
    continuation_tables = [t for t in table_nodes if t.get("is_continuation")]
    review_required_tables = [t for t in table_nodes if t.get("review_required")]
    low_confidence_tables = [t for t in table_nodes if float(t.get("confidence", 0.0) or 0.0) < 0.75]
    diagnostic_tables = [t for t in table_nodes if (t.get("diagnostics") or {}).get("possible_missing_table_content")]
    diagnostic_candidate_total = sum(int((t.get("diagnostics") or {}).get("candidate_count", 0) or 0) for t in table_nodes)
    review_required_toc_blocks = [toc for toc in toc_nodes if (toc.get("toc_diagnostics") or {}).get("review_required")]
    toc_review_item_total = sum(
        int((toc.get("toc_diagnostics") or {}).get("review_item_count", 0) or 0)
        for toc in toc_nodes
    )
    aggregated_toc_review_item_total = sum(
        int((toc.get("toc_diagnostics") or {}).get("aggregated_review_item_count", 0) or 0)
        for toc in toc_nodes
    )
    toc_sequences = _build_toc_sequences(toc_nodes, state.page_payloads)
    continuation_similarities = [
        float(t.get("cross_page_similarity"))
        for t in continuation_tables
        if t.get("cross_page_similarity") is not None
    ]

    merged_text = "\n\n".join(analysis_page_texts)
    content_evidence_counts = _count_content_evidence_types(content_evidence)
    content_units = _build_content_units(content_evidence)
    content_unit_counts = _count_content_unit_types(content_units)
    fact_extraction_corpus, fact_extraction_unit_count = _build_fact_extraction_corpus(content_units)
    atomic_facts = extract_atomic_facts(fact_extraction_corpus)
    return {
        "filename": path.name,
        "source_path": str(path),
        "source_type": "pdf",
        "pages": pages,
        "bounding_boxes": text_bounding_boxes,
        "image_blocks": image_blocks,
        "algorithm_blocks": algorithm_blocks,
        "equation_blocks": equation_blocks,
        "content_evidence": content_evidence,
        "content_units": content_units,
        "figures": state.figure_nodes,
        "toc_blocks": toc_nodes,
        "toc_sequences": toc_sequences,
        "table_asts": state.table_asts,
        "tables": state.table_asts,
        "document_ast": {
            "source_type": "pdf",
            "pages": document_ast_pages,
            "table_refs": [table["table_id"] for table in state.table_asts],
            "toc_refs": [toc["toc_id"] for toc in toc_nodes],
            "toc_sequence_refs": [sequence["toc_sequence_id"] for sequence in toc_sequences],
            "image_refs": [image["image_id"] for image in image_blocks],
            "algorithm_refs": [algorithm["algorithm_id"] for algorithm in algorithm_blocks],
            "equation_refs": [equation["equation_id"] for equation in equation_blocks],
            "content_evidence_refs": [item["evidence_id"] for item in content_evidence],
            "content_unit_refs": [item["unit_id"] for item in content_units],
        },
        "text": merged_text,
        "atomic_facts": atomic_facts,
        "metadata": {
            "page_count": len(pages),
            "bounding_box_count": len(text_bounding_boxes),
            "image_count": len(image_blocks),
            "algorithm_count": len(algorithm_blocks),
            "equation_count": len(equation_blocks),
            "table_count": len(state.table_asts),
            "toc_count": len(toc_nodes),
            "toc_sequence_count": len(toc_sequences),
            "embedded_outline_count": int(state.embedded_outline_count or 0),
            "embedded_outline_depth": int(state.embedded_outline_depth or 0),
            "embedded_file_count": int(state.embedded_file_count or 0),
            "embedded_file_names": [str(item).strip() for item in state.embedded_file_names if str(item).strip()],
            "link_annotation_count": int(state.link_annotation_count or 0),
            "navigational_link_count": int(state.navigational_link_count or 0),
            "internal_link_count": int(state.internal_link_count or 0),
            "external_file_link_count": int(state.external_file_link_count or 0),
            "external_uri_link_count": int(state.external_uri_link_count or 0),
            "external_file_link_targets": [str(item).strip() for item in state.external_file_link_targets if str(item).strip()],
            "link_annotation_xref_page_map": [
                {"xref": int(xref), "page": int(page)}
                for xref, page in sorted(state.link_annotation_xref_page_map.items())
                if int(xref) > 0 and int(page) > 0
            ],
            "non_link_annotation_count": int(state.non_link_annotation_count or 0),
            "non_link_annotation_types": [str(item).strip() for item in state.non_link_annotation_types if str(item).strip()],
            "multi_page_toc_sequence_count": sum(1 for sequence in toc_sequences if int(sequence.get("page_count", 0) or 0) > 1),
            "content_evidence_count": len(content_evidence),
            "content_evidence_counts": content_evidence_counts,
            "content_unit_count": len(content_units),
            "content_unit_counts": content_unit_counts,
            "fact_extraction_unit_count": fact_extraction_unit_count,
            "fact_extraction_corpus_length": len(fact_extraction_corpus),
            "raw_table_candidates": state.counters.raw_table_candidates,
            "accepted_table_candidates": state.counters.accepted_table_candidates,
            "toc_block_count": state.counters.toc_block_count,
            "review_required_toc_count": len(review_required_toc_blocks),
            "toc_review_item_count": toc_review_item_total,
            "aggregated_toc_review_item_count": aggregated_toc_review_item_total,
            "continuation_table_count": len(continuation_tables),
            "review_required_table_count": len(review_required_tables),
            "low_confidence_table_count": len(low_confidence_tables),
            "possible_missing_content_table_count": len(diagnostic_tables),
            "possible_missing_content_candidate_count": diagnostic_candidate_total,
            "continuation_similarity_avg": (
                round(sum(continuation_similarities) / len(continuation_similarities), 3)
                if continuation_similarities else None
            ),
            "continuation_similarity_min": (
                round(min(continuation_similarities), 3) if continuation_similarities else None
            ),
            "continuation_similarity_max": (
                round(max(continuation_similarities), 3) if continuation_similarities else None
            ),
            "cross_page_boundary_row_merge_count": cross_page_boundary_row_merge_count,
            "figure_count": len(state.figure_nodes),
            "cross_page_table_links": state.counters.cross_page_table_links,
            "semantic_merge_count": state.counters.semantic_merge_count,
            "image_text_recovered_count": state.counters.image_text_recovered_count,
            "rejected_table_candidates": state.counters.rejected_table_candidates,
            "table_fragment_merge_count": state.counters.table_fragment_merge_count,
            "duplicate_image_blocks_removed": state.counters.duplicate_image_blocks_removed,
            "header_footer_filtered_count": state.counters.header_footer_filtered_count,
            "table_text_suppressed_count": total_table_text_suppressed,
            "parser_hint": PDF_PARSER_HINT,
            "parser_config_snapshot": {
                "cross_page_stitching": cfg_snapshot,
                "table_content_policy": {
                    "enable_supplement_writeback": table_policy.enable_supplement_writeback,
                    "enable_missing_content_diagnostics": table_policy.enable_missing_content_diagnostics,
                    "diagnostic_min_candidates_per_table": table_policy.diagnostic_min_candidates_per_table,
                    "diagnostic_max_candidate_ratio": table_policy.diagnostic_max_candidate_ratio,
                    "diagnostic_min_text_length": table_policy.diagnostic_min_text_length,
                    "suppress_col0_diagnostics_for_continuation": table_policy.suppress_col0_diagnostics_for_continuation,
                },
            },
        },
    }


def _merge_page_nodes_with_text_reading_order(
    text_nodes: list[dict[str, Any]],
    other_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not text_nodes:
        return sorted(other_nodes, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    if not other_nodes:
        return list(text_nodes)

    merged: list[dict[str, Any]] = []
    ordered_other = sorted(other_nodes, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    text_index = 0

    for other in ordered_other:
        other_top = float(other["bbox"][1])
        while text_index < len(text_nodes):
            text_node = text_nodes[text_index]
            text_center_y = (float(text_node["bbox"][1]) + float(text_node["bbox"][3])) / 2.0
            if text_center_y > other_top:
                break
            merged.append(text_node)
            text_index += 1
        merged.append(other)

    if text_index < len(text_nodes):
        merged.extend(text_nodes[text_index:])
    return merged


def _is_reference_heading_text(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 64:
        return False
    return _compact_text(compact) in _REFERENCE_HEADING_NORMALIZED


def _match_reference_entry_start(text: str) -> tuple[str, bool] | None:
    compact = _clean_text(text)
    if not compact:
        return None
    inline_match = _REFERENCE_ENTRY_INLINE_RE.match(compact)
    if inline_match:
        return inline_match.group("number"), True
    standalone_match = _REFERENCE_ENTRY_STANDALONE_RE.match(compact)
    if standalone_match:
        return standalone_match.group("number"), False
    return None


def _looks_like_reference_boundary_heading(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 96:
        return False
    if _is_reference_heading_text(compact):
        return False
    if _match_reference_entry_start(compact):
        return False
    normalized = _compact_text(compact)
    if normalized in _REFERENCE_STOP_HEADING_NORMALIZED:
        return True
    if compact[-1] in {".", ",", ";", ":", "!", "?", "\u3002", "\uff0c", "\uff1b", "\uff1a", "\uff01", "\uff1f"}:
        return False
    if _REFERENCE_CITATION_CUE_RE.search(compact):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'`.-]*|\d+", compact)
    return 1 <= len(words) <= 8


def _looks_like_reference_page_boilerplate(
    text: str,
    bbox: list[float] | tuple[float, float, float, float],
    page_height: float,
) -> bool:
    compact = _clean_text(text)
    if not compact:
        return False
    y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0
    top_band = max(64.0, page_height * 0.08)
    if y0 > top_band:
        return False
    lowered = compact.lower()
    if _looks_like_equation_false_positive_text(compact):
        return False
    if lowered.startswith(("where ", "with ")):
        return False
    if re.fullmatch(r"page\s+\d+\s+of\s+\d+", lowered):
        return True
    if re.search(r"\(\d{4}\)\s*\d+(?::\d+)?", compact) and len(compact) <= 180:
        return True
    return False


def _looks_like_reference_entry_seed(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 360:
        return False
    if _looks_like_reference_boundary_heading(compact):
        return False
    if _REFERENCE_CITATION_CUE_RE.search(compact):
        return True
    return bool(_REFERENCE_AUTHOR_LIST_RE.match(compact))


def _looks_like_reference_entry_continuation(
    text: str,
    bbox: list[float] | tuple[float, float, float, float],
    page_height: float,
) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 360:
        return False
    if _looks_like_reference_page_boilerplate(compact, bbox, page_height):
        return False
    if _looks_like_reference_boundary_heading(compact):
        return False
    return True


def _find_reference_continuation_start_index(
    projected_text_blocks: list[dict[str, Any]],
    page_height: float,
    previous_entry_open: bool,
) -> int | None:
    eligible_count = 0
    search_depth = 10
    top_limit = max(180.0, page_height * 0.35)
    for index, block in enumerate(projected_text_blocks):
        if str(block.get("block_type", "text") or "text") != "text":
            continue
        text = _clean_text(str(block.get("text", "")))
        if not text:
            continue
        bbox = list(block.get("bbox", []))
        if _looks_like_reference_page_boilerplate(text, bbox, page_height):
            continue
        if _looks_like_reference_boundary_heading(text):
            return None
        if _match_reference_entry_start(text):
            return index
        if previous_entry_open and _looks_like_reference_entry_seed(text):
            return index
        eligible_count += 1
        y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0
        if eligible_count >= search_depth or y0 >= top_limit:
            break
    return None


def _mark_reference_heading(text_block: dict[str, Any]) -> None:
    text_block["semantic_role"] = "reference_heading"
    text_block["unit_role"] = "section_heading"
    text_block.pop("reference_number", None)
    text_block.pop("reference_entry_index", None)
    text_block.pop("reference_entry_start", None)
    text_block.pop("reference_continuation", None)
    text_block.pop("reference_page_continuation", None)


def _mark_reference_entry(
    text_block: dict[str, Any],
    reference_number: str | None,
    entry_index: int,
    *,
    entry_start: bool,
    page_continuation: bool,
) -> None:
    text_block["semantic_role"] = "reference_entry"
    text_block["unit_role"] = "entry"
    if reference_number:
        text_block["reference_number"] = reference_number
    else:
        text_block.pop("reference_number", None)
    text_block["reference_entry_index"] = entry_index
    text_block["reference_entry_start"] = bool(entry_start)
    text_block["reference_continuation"] = not bool(entry_start)
    if page_continuation:
        text_block["reference_page_continuation"] = True
    else:
        text_block.pop("reference_page_continuation", None)


def _annotate_reference_section_text_blocks(
    projected_text_blocks: list[dict[str, Any]],
    *,
    page_height: float,
    reference_state: dict[str, Any],
) -> dict[str, Any]:
    carry_active = bool(reference_state.get("active"))
    carry_entry_open = bool(reference_state.get("entry_open"))
    current_reference_number = str(reference_state.get("reference_number") or "").strip() or None
    current_entry_index = int(reference_state.get("entry_index", 0) or 0)

    heading_index: int | None = None
    for index, block in enumerate(projected_text_blocks):
        if str(block.get("block_type", "text") or "text") != "text":
            continue
        if _is_reference_heading_text(str(block.get("text", ""))):
            heading_index = index
            break

    continuation_start_index: int | None = None
    if heading_index is None and carry_active:
        continuation_start_index = _find_reference_continuation_start_index(
            projected_text_blocks,
            page_height=page_height,
            previous_entry_open=carry_entry_open,
        )

    if heading_index is None and continuation_start_index is None:
        return {
            "active": False,
            "entry_open": False,
            "reference_number": None,
            "entry_index": current_entry_index,
        }

    in_reference_section = False
    entry_open = carry_entry_open if continuation_start_index is not None else False
    page_continuation_mode = continuation_start_index is not None

    for index, block in enumerate(projected_text_blocks):
        if str(block.get("block_type", "text") or "text") != "text":
            continue
        text = _clean_text(str(block.get("text", "")))
        if not text:
            continue

        if heading_index is not None and index == heading_index:
            _mark_reference_heading(block)
            in_reference_section = True
            entry_open = False
            current_reference_number = None
            page_continuation_mode = False
            continue

        if not in_reference_section:
            if continuation_start_index is None or index != continuation_start_index:
                continue
            in_reference_section = True

        if _looks_like_reference_boundary_heading(text):
            in_reference_section = False
            entry_open = False
            current_reference_number = None
            break

        match = _match_reference_entry_start(text)
        if match:
            next_reference_number, _ = match
            if not entry_open or next_reference_number != current_reference_number:
                current_entry_index += 1
            current_reference_number = next_reference_number
            entry_open = True
            _mark_reference_entry(
                block,
                current_reference_number,
                current_entry_index,
                entry_start=True,
                page_continuation=page_continuation_mode,
            )
            page_continuation_mode = False
            continue

        bbox = list(block.get("bbox", []))
        if entry_open and _looks_like_reference_entry_continuation(text, bbox, page_height):
            _mark_reference_entry(
                block,
                current_reference_number,
                current_entry_index,
                entry_start=False,
                page_continuation=page_continuation_mode,
            )
            page_continuation_mode = False
            continue

        if not entry_open and _looks_like_reference_entry_seed(text):
            current_entry_index += 1
            entry_open = True
            _mark_reference_entry(
                block,
                current_reference_number,
                current_entry_index,
                entry_start=True,
                page_continuation=page_continuation_mode,
            )
            page_continuation_mode = False
            continue

        in_reference_section = False
        entry_open = False
        current_reference_number = None
        break

    return {
        "active": in_reference_section and entry_open,
        "entry_open": in_reference_section and entry_open,
        "reference_number": current_reference_number if in_reference_section and entry_open else None,
        "entry_index": current_entry_index,
    }


def _looks_like_literature_author_line(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 320 or not any(ch.isdigit() for ch in compact):
        return False
    author_token_count = len(_LITERATURE_AUTHOR_TOKEN_RE.findall(compact))
    marker_count = (
        compact.count(",")
        + compact.count("*")
        + compact.count("\u2020")
        + compact.count("\u2021")
    )
    return author_token_count >= 2 and marker_count >= 2


def _looks_like_author_affiliation_line(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 360:
        return False
    lowered = compact.lower()
    if _looks_like_equation_false_positive_text(compact):
        return False
    if not any(keyword in lowered for keyword in _AFFILIATION_KEYWORDS):
        return False
    if re.match(r"^(?:\d+|[*])\s*", compact):
        return True
    return compact.count(";") >= 1 or compact.count(",") >= 2


def _looks_like_contact_name_line(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 80 or any(ch.isdigit() for ch in compact):
        return False
    return bool(_PERSON_NAME_LINE_RE.fullmatch(compact))


def _set_text_evidence_role(
    evidence: dict[str, Any],
    semantic_role: str,
    unit_role: str,
) -> None:
    evidence["semantic_role"] = semantic_role
    content_text = _clean_text(str(evidence.get("content_text", "")))
    evidence["segments"] = [{"role": unit_role, "text": content_text}]


def _is_literature_front_page(text_evidence: list[dict[str, Any]]) -> bool:
    if not text_evidence:
        return False
    marker_roles = {
        "citation_metadata",
        "publication_label",
        "author_line",
        "author_affiliation",
        "correspondence",
        "author_note",
        "contact_email",
    }
    marker_count = sum(
        1
        for evidence in text_evidence
        if str(evidence.get("semantic_role", "") or "") in marker_roles
    )
    has_abstract = any(
        _compact_text(str(evidence.get("content_text", ""))) == "abstract"
        for evidence in text_evidence
    )
    return marker_count >= 2 and has_abstract


def _refine_literature_front_matter_text_evidence(
    text_evidence: list[dict[str, Any]],
    page_height: float,
) -> None:
    if not _is_literature_front_page(text_evidence):
        return

    ordered = sorted(
        text_evidence,
        key=lambda item: (
            float((item.get("bbox") or [float("inf"), float("inf")])[1]),
            float((item.get("bbox") or [float("inf")])[0]),
        ),
    )
    abstract_top = next(
        (
            float((evidence.get("bbox") or [0.0, 0.0])[1])
            for evidence in ordered
            if _compact_text(str(evidence.get("content_text", ""))) == "abstract"
        ),
        None,
    )
    top_metadata_limit = min(96.0, max(72.0, float(abstract_top or 96.0) - 120.0))
    correspondence_mode = False
    license_mode = False

    for evidence in ordered:
        text = _clean_text(str(evidence.get("content_text", "")))
        if not text:
            continue
        lowered = text.lower()
        normalized = _compact_text(text)
        bbox = list(evidence.get("bbox", []))
        y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0
        semantic_role = str(evidence.get("semantic_role", "") or "")
        unit_role = str(((evidence.get("segments") or [{}])[0]).get("role", "body") or "body")

        if license_mode:
            _set_text_evidence_role(evidence, "license_notice", "metadata")
            continue
        if semantic_role == "license_notice":
            license_mode = True
            continue
        if normalized == "keywords" or lowered.startswith("keywords "):
            _set_text_evidence_role(evidence, "keyword_metadata", "metadata")
            continue
        if correspondence_mode:
            if _looks_like_contact_name_line(text):
                _set_text_evidence_role(evidence, "contact_name", "metadata")
                continue
            if semantic_role in {"contact_email", "author_note", "author_affiliation"}:
                continue
            if unit_role != "metadata":
                correspondence_mode = False
        if semantic_role == "correspondence":
            correspondence_mode = True
            continue
        if unit_role == "body" and y0 <= top_metadata_limit and len(text) <= 160 and not re.search(r"[.!?;:]", text):
            _set_text_evidence_role(evidence, "publication_masthead", "metadata")


def _classify_text_block_semantic_role(text_block: dict[str, Any]) -> tuple[str, str]:
    explicit_role = str(text_block.get("semantic_role", "") or "").strip()
    explicit_unit_role = str(text_block.get("unit_role", "") or "").strip()
    if explicit_role:
        return explicit_role, explicit_unit_role or "body"

    text = _clean_text(str(text_block.get("text", "")))
    if not text:
        return "text_block", "body"

    normalized = _compact_text(text)
    lowered = text.lower()
    bbox = list(text_block.get("bbox", []))
    y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0

    if _EMAIL_RE.search(text):
        return "contact_email", "metadata"
    if "contributed equally" in lowered:
        return "author_note", "metadata"
    if "full list of author information" in lowered:
        return "author_note", "metadata"
    if lowered.startswith("*correspondence") or lowered.startswith("correspondence"):
        return "correspondence", "metadata"
    if lowered.startswith("open access") or normalized in {"openaccess", "research"}:
        return "publication_label", "metadata"
    if lowered.startswith("keywords "):
        return "keyword_metadata", "metadata"
    if "creativecommons" in normalized or "licensedunder" in normalized or "copyright" in normalized or text.startswith("©"):
        return "license_notice", "metadata"
    if "doi.org/" in lowered or "https://doi.org/" in lowered:
        return "citation_metadata", "metadata"
    if y0 <= 96.0 and ("et al." in lowered or re.search(r"\(\d{4}\)\s*\d+(?::\d+|\(\d+\))", text)):
        return "citation_metadata", "metadata"
    if _looks_like_literature_author_line(text):
        return "author_line", "metadata"
    if _looks_like_author_affiliation_line(text):
        return "author_affiliation", "metadata"
    return "text_block", "body"


def _bbox_center_y(bbox: list[float] | tuple[float, float, float, float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return (float(bbox[1]) + float(bbox[3])) / 2.0


def _bbox_height(bbox: list[float] | tuple[float, float, float, float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_union(bboxes: list[list[float] | tuple[float, float, float, float]]) -> list[float]:
    valid = [bbox for bbox in bboxes if len(bbox) >= 4]
    if not valid:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(float(bbox[0]) for bbox in valid),
        min(float(bbox[1]) for bbox in valid),
        max(float(bbox[2]) for bbox in valid),
        max(float(bbox[3]) for bbox in valid),
    ]


def _looks_like_display_equation_text(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) < 5 or len(compact) > 140:
        return False
    if compact[:1] in {"•", "-", "*"}:
        return False
    if compact[-1] in _DISPLAY_EQUATION_TERMINAL_PUNCTUATION:
        return False
    if "http" in compact.lower() or "doi" in compact.lower():
        return False
    if not re.search(r"[A-Za-z]", compact):
        return False
    if not re.search(r"[=≈≠≤≥]", compact):
        return False

    lowered = compact.lower()
    if re.match(r"^[ijkmn]\s*=\s*\d+(?:\s*[()⎝⎞])?$", compact):
        return False
    if (
        re.match(r"^[ijkmn]\s*=\s*\d+\b", compact)
        and len(compact) < 22
        and not re.search(r"[\[\]\(\)\{\};,+\-*/]", compact)
    ):
        return False
    if re.search(r",\s*(where|which|and|for)\b", lowered):
        return False
    alpha_tokens = re.findall(r"[A-Za-z]+", compact)
    prose_cue_count = sum(1 for cue in _DISPLAY_EQUATION_CLUSTER_PROSE_CUES if cue in lowered)
    if len(alpha_tokens) >= 6 and re.search(r"\b(is|the|and|of|being|class|labels|probability)\b", lowered):
        return False
    if prose_cue_count >= 1 and len(alpha_tokens) >= 4:
        return False
    if any(cue in lowered for cue in (" denotes ", " token pair ", " model ", " operation ")):
        return False
    if len(compact) < 12 and not re.search(r"[\(\)\[\]\{\}]", compact):
        return False
    if not re.match(
        r"^[A-Za-z][A-Za-z0-9_]*(?:\s*\([^)]*\))?(?:\s*[⋆^]\s*[A-Za-z][A-Za-z0-9_]*)?\s*[=≈≠≤≥]",
        compact,
    ):
        return False

    signal_count = 0
    if re.search(r"[A-Za-z0-9\)\]]\s*[=≈≠≤≥]\s*[A-Za-z0-9\(\[]", compact):
        signal_count += 2
    if re.search(r"[A-Za-z]+\s*[\(\[]", compact):
        signal_count += 1
    if any(ch in compact for ch in "()[]{}⋆^_"):
        signal_count += 1
    if any(ch in compact for ch in "=≈≠≤≥+*/⋆"):
        signal_count += 1
    if _DISPLAY_EQUATION_FUNCTION_RE.search(compact):
        signal_count += 1
    return signal_count >= 3


def _normalized_equation_marker_text(text: str) -> str | None:
    compact = _clean_text(text).strip()
    if not compact:
        return None
    if _DISPLAY_EQUATION_MARKER_RE.fullmatch(compact):
        return compact

    has_left_paren = "(" in compact
    has_right_paren = ")" in compact
    if not (has_left_paren or has_right_paren):
        return None
    if len(compact) > 12:
        return None

    tokens = re.findall(r"\d{1,3}", compact)
    if len(tokens) != 1:
        return None
    return f"({tokens[0]})"


def _looks_like_equation_marker_text(text: str) -> bool:
    return _normalized_equation_marker_text(text) is not None


def _strip_display_equation_terminal_artifacts(text: str) -> str:
    compact = _clean_text(text)
    while compact and compact[-1] in _DISPLAY_EQUATION_TRAILING_TRIM_CHARS:
        compact = compact[:-1].rstrip()
    return compact


def _looks_like_equation_adjacent_fragment(text: str) -> bool:
    compact = _strip_display_equation_terminal_artifacts(text)
    if not compact or len(compact) > 72:
        return False
    if _looks_like_equation_marker_text(compact):
        return False
    if not re.search(r"[A-Za-z]", compact):
        return False
    if compact[-1] in _DISPLAY_EQUATION_TERMINAL_PUNCTUATION:
        return False
    lowered = compact.lower()
    if lowered in _DISPLAY_EQUATION_FRAGMENT_STOPWORDS:
        return False
    if "http" in lowered or "doi" in lowered:
        return False
    if _DISPLAY_EQUATION_FUNCTION_RE.fullmatch(compact):
        return True

    alpha_tokens = re.findall(r"[A-Za-z]+", compact)
    signal_count = 0
    if re.search(r"[=+\-*/\[\]\(\)\{\};,_^]", compact):
        signal_count += 2
    if re.match(r"^[A-Za-z][A-Za-z0-9_]*\s*[\(\[]", compact):
        signal_count += 2
    if re.search(r"(?:[A-Za-z]\d|\d[A-Za-z]|[a-z][A-Z]|[A-Z]{2,})", compact):
        signal_count += 1
    if _DISPLAY_EQUATION_FUNCTION_RE.search(compact):
        signal_count += 1
    if len(alpha_tokens) > 3 and signal_count < 2:
        return False
    return signal_count >= 2


def _looks_like_symbol_only_equation_fragment(text: str) -> bool:
    compact = _strip_display_equation_terminal_artifacts(text)
    if not compact or len(compact) > 24:
        return False
    if compact[-1:] and compact[-1] in _DISPLAY_EQUATION_TERMINAL_PUNCTUATION:
        return False
    if re.search(r"[A-Za-z]{3,}", compact):
        return False
    if re.fullmatch(r"[\s(){}\[\]⎛⎜⎝⎞⎟⎠]+", compact):
        return True
    if any(symbol in compact for symbol in ("⎛", "⎜", "⎝", "⎞", "⎟", "⎠", "(", ")", "[", "]", "{", "}")):
        return True
    return False


def _can_attach_symbol_equation_fragment(
    equation_bbox: tuple[float, float, float, float],
    candidate_block: dict[str, Any],
    *,
    side: str,
) -> bool:
    candidate_bbox = tuple(candidate_block.get("bbox", [0.0, 0.0, 0.0, 0.0]))
    vertical_overlap = max(
        0.0,
        min(equation_bbox[3], candidate_bbox[3]) - max(equation_bbox[1], candidate_bbox[1]),
    )
    min_height = max(
        1.0,
        min(max(0.0, equation_bbox[3] - equation_bbox[1]), max(0.0, candidate_bbox[3] - candidate_bbox[1])),
    )
    center_gap = abs(((equation_bbox[1] + equation_bbox[3]) / 2.0) - ((candidate_bbox[1] + candidate_bbox[3]) / 2.0))
    vertical_gap = min(
        abs(candidate_bbox[1] - equation_bbox[3]),
        abs(equation_bbox[1] - candidate_bbox[3]),
    )
    if vertical_overlap / min_height < 0.45 and center_gap > 14.0 and vertical_gap > 12.0:
        return False
    if side == "right":
        return float(candidate_bbox[0]) <= float(equation_bbox[2]) + 80.0
    return float(candidate_bbox[2]) >= float(equation_bbox[0]) - 80.0


def _build_equation_block_id(base_block_id: str) -> str:
    compact = base_block_id.strip()
    return f"eq_{compact}" if compact else "eq_unknown"


def _normalized_source_block_indices(block: dict[str, Any]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_value in block.get("source_block_indices", []) or []:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    source_block_index = block.get("source_block_index")
    if source_block_index is not None:
        try:
            value = int(source_block_index)
        except (TypeError, ValueError):
            value = None
        if value is not None and value not in seen:
            normalized.append(value)
    return normalized


def _equation_lane(block: dict[str, Any]) -> str:
    lane = str(block.get("layout_lane", "") or "").strip()
    return lane if lane in {"left", "right", "full_width"} else "full_width"


def _looks_like_residual_math_text(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 18:
        return False
    if _looks_like_equation_marker_text(compact):
        return True
    short_tokens = re.findall(r"[A-Za-z]{1,2}", compact)
    if not short_tokens:
        return False
    has_single_letter = any(len(token) == 1 for token in short_tokens)
    has_math_signal = bool(re.search(r"[\(\)\[\]\{\}=+\-*/αβγλσθ]", compact, re.IGNORECASE))
    if has_single_letter and has_math_signal:
        return True
    return has_single_letter and len(short_tokens) >= 2 and len(compact) <= 8


def _drop_residual_equation_text_fragments(projected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    equation_blocks = [
        block
        for block in projected
        if str(block.get("block_type", "text") or "text") == "equation"
    ]
    if not equation_blocks:
        return projected

    equation_sources = [
        {
            "indices": set(_normalized_source_block_indices(block)),
            "bbox": tuple(block.get("bbox", [0.0, 0.0, 0.0, 0.0])),
            "lane": _equation_lane(block),
        }
        for block in equation_blocks
    ]

    filtered: list[dict[str, Any]] = []
    for block in projected:
        if str(block.get("block_type", "text") or "text") == "equation":
            filtered.append(block)
            continue
        text = str(block.get("text", "") or "")
        if not _looks_like_residual_math_text(text):
            filtered.append(block)
            continue
        block_indices = set(_normalized_source_block_indices(block))
        if not block_indices:
            filtered.append(block)
            continue
        block_bbox = tuple(block.get("bbox", [0.0, 0.0, 0.0, 0.0]))
        block_lane = _equation_lane(block)

        is_residual = False
        for source in equation_sources:
            if not (block_indices & source["indices"]):
                continue
            vertical_gap = min(abs(block_bbox[1] - source["bbox"][3]), abs(source["bbox"][1] - block_bbox[3]))
            if vertical_gap <= 28.0:
                is_residual = True
                break
        if is_residual:
            continue
        filtered.append(block)
    return filtered


def _looks_like_equation_continuation_text(text: str) -> bool:
    compact = _clean_text(text)
    lowered = compact.lower()
    if not compact or len(compact) > 48:
        return False
    if _looks_like_equation_marker_text(compact):
        return True
    if re.search(r"[=+\-*/\(\)\[\]\{\}αβγλσθ∑⎧⎨⎩⎪]", compact, re.IGNORECASE):
        return True
    if lowered.startswith("where ") and len(compact) <= 24:
        return True
    if lowered.endswith(" of") and len(compact) <= 24:
        return True
    if compact in {"F v at", "g k is", "k is", "d k is"}:
        return True
    short_tokens = re.findall(r"[A-Za-z]{1,2}", compact)
    return any(len(token) == 1 for token in short_tokens) and len(compact) <= 16


def _attach_equation_continuation_fragments(projected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not projected:
        return projected

    attached_indices: set[int] = set()
    result: list[dict[str, Any]] = []
    for index, block in enumerate(projected):
        if index in attached_indices:
            continue
        if str(block.get("block_type", "text") or "text") != "equation":
            result.append(block)
            continue

        equation_block = dict(block)
        equation_bbox = tuple(equation_block.get("bbox", [0.0, 0.0, 0.0, 0.0]))
        equation_lane = _equation_lane(equation_block)
        leading_symbol_parts: list[str] = []
        continuation_parts: list[str] = []

        previous_index = index - 1
        while previous_index >= 0 and previous_index not in attached_indices:
            candidate = projected[previous_index]
            if str(candidate.get("block_type", "text") or "text") == "equation":
                break
            candidate_text = str(candidate.get("text", "") or "")
            if not _looks_like_symbol_only_equation_fragment(candidate_text):
                break
            candidate_bbox = tuple(candidate.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            candidate_lane = _equation_lane(candidate)
            if candidate_lane != equation_lane:
                break
            if not _can_attach_symbol_equation_fragment(equation_bbox, candidate, side="left"):
                break
            leading_symbol_parts.insert(0, _clean_text(candidate_text))
            equation_bbox = tuple(_bbox_union([candidate_bbox, equation_bbox]))
            attached_indices.add(previous_index)
            previous_index -= 1

        next_index = index + 1
        while next_index < len(projected):
            candidate = projected[next_index]
            if str(candidate.get("block_type", "text") or "text") == "equation":
                break
            candidate_text = str(candidate.get("text", "") or "")
            is_symbol_only_fragment = _looks_like_symbol_only_equation_fragment(candidate_text)
            if not is_symbol_only_fragment and not _looks_like_equation_continuation_text(candidate_text):
                break
            candidate_bbox = tuple(candidate.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            candidate_lane = _equation_lane(candidate)
            if candidate_lane != equation_lane:
                break
            if is_symbol_only_fragment:
                if not _can_attach_symbol_equation_fragment(equation_bbox, candidate, side="right"):
                    break
            else:
                vertical_gap = min(
                    abs(candidate_bbox[1] - equation_bbox[3]),
                    abs(equation_bbox[1] - candidate_bbox[3]),
                )
                if vertical_gap > 28.0:
                    break
            continuation_parts.append(_clean_text(candidate_text))
            equation_bbox = tuple(_bbox_union([equation_bbox, candidate_bbox]))
            attached_indices.add(next_index)
            next_index += 1

        if leading_symbol_parts or continuation_parts:
            equation_block["text"] = _clean_text(
                " ".join([*leading_symbol_parts, str(equation_block.get("text", "")).strip(), *continuation_parts])
            )
            equation_block["bbox"] = list(equation_bbox)
        result.append(equation_block)
    return result


def _equation_visual_row_tolerance(text_blocks: list[dict[str, Any]]) -> float:
    heights = [
        max(1.0, float(block["bbox"][3]) - float(block["bbox"][1]))
        for block in text_blocks
        if len(block.get("bbox", [])) >= 4
    ]
    if not heights:
        return 3.0
    heights.sort()
    median_height = heights[len(heights) // 2]
    return max(2.8, median_height * 0.55)


def _group_equation_blocks_by_visual_rows(text_blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not text_blocks:
        return []

    row_tolerance = _equation_visual_row_tolerance(text_blocks)
    ordered = sorted(
        text_blocks,
        key=lambda item: (
            (float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0,
            float(item["bbox"][0]),
        ),
    )
    rows: list[dict[str, Any]] = []
    for block in ordered:
        bbox = tuple(float(item) for item in block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        center_y = (bbox[1] + bbox[3]) / 2.0
        best_row: dict[str, Any] | None = None
        best_score: tuple[float, float] | None = None
        for row in rows:
            row_bbox = tuple(row["bbox"])
            overlap = _vertical_overlap_ratio(bbox, row_bbox)
            center_diff = abs(center_y - float(row["center_y"]))
            if center_diff > row_tolerance and overlap < 0.55:
                continue
            score = (overlap, -center_diff)
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            rows.append({"bbox": list(bbox), "center_y": center_y, "blocks": [block]})
            continue
        best_row["blocks"].append(block)
        best_row["bbox"] = _bbox_union([tuple(best_row["bbox"]), bbox])
        best_row["center_y"] = (
            float(best_row["center_y"]) * max(1, len(best_row["blocks"]) - 1) + center_y
        ) / float(len(best_row["blocks"]))
    return [sorted(row["blocks"], key=lambda item: float(item["bbox"][0])) for row in rows]


def _vertical_overlap_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    top = max(a_bbox[1], b_bbox[1])
    bottom = min(a_bbox[3], b_bbox[3])
    overlap = max(0.0, bottom - top)
    min_height = max(1.0, min(a_bbox[3] - a_bbox[1], b_bbox[3] - b_bbox[1]))
    return overlap / min_height


def _row_compact_text(blocks: list[dict[str, Any]]) -> str:
    return _clean_text(" ".join(str(block.get("text", "") or "").strip() for block in blocks)).strip()


def _looks_like_display_equation_cluster_block(text: str) -> bool:
    compact = _clean_text(text)
    if not compact:
        return False
    if _looks_like_equation_marker_text(compact):
        return True

    lowered = compact.lower()
    if lowered.startswith(("where ", "with ")):
        return False

    compact_identity = _compact_text(compact)
    alpha_tokens = re.findall(r"[A-Za-z]+", compact)
    math_signal_count = len(re.findall(r"[=∑∈⎧⎨⎩⎪≥≤−+*/\[\]\(\)]", compact))
    prose_cue_count = sum(1 for cue in _DISPLAY_EQUATION_CLUSTER_PROSE_CUES if cue in lowered)

    if prose_cue_count >= 2 and math_signal_count < 4:
        return False
    if len(alpha_tokens) >= 8 and math_signal_count < 4:
        return False

    if any(token in compact_identity for token in ("sw", "sb", "st", "cj", "dw", "jw", "zi", "uj")):
        return True
    if re.match(r"^(?:min|max)\b", lowered) and len(alpha_tokens) <= 4:
        return True
    if math_signal_count >= 4:
        return True
    return math_signal_count >= 2 and len(alpha_tokens) <= 6


def _looks_like_display_equation_cluster_row(blocks: list[dict[str, Any]]) -> bool:
    if not blocks:
        return False
    text = _row_compact_text(blocks)
    if not text:
        return False
    lowered = text.lower()
    if _looks_like_equation_false_positive_text(text):
        return False
    if any(_looks_like_equation_marker_text(str(block.get("text", "") or "").strip()) for block in blocks):
        return True
    if lowered.startswith(("where ", "with ")):
        return False
    return any(_looks_like_display_equation_cluster_block(str(block.get("text", "") or "")) for block in blocks)


def _looks_like_display_equation_support_row(blocks: list[dict[str, Any]]) -> bool:
    text = _row_compact_text(blocks)
    if not text:
        return False
    lowered = text.lower()
    if _looks_like_equation_false_positive_text(text):
        return False
    if lowered in {"and"} or lowered.startswith(("where ", "with ")):
        return False
    alpha_tokens = re.findall(r"[A-Za-z]+", text)
    if len(alpha_tokens) > 4 or len(text) > 40:
        return False
    prose_cue_count = sum(1 for cue in _DISPLAY_EQUATION_CLUSTER_PROSE_CUES if cue in lowered)
    if prose_cue_count:
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))


def _lane_rows_for_equation_detection(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lane_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(_group_equation_blocks_by_visual_rows(text_blocks)):
        by_lane: dict[str, list[dict[str, Any]]] = {}
        for block in row:
            by_lane.setdefault(_equation_lane(block), []).append(block)
        for lane, lane_blocks in by_lane.items():
            lane_rows.append(
                {
                    "row_index": row_index,
                    "lane": lane,
                    "blocks": sorted(lane_blocks, key=lambda item: float(item["bbox"][0])),
                    "bbox": _bbox_union([tuple(block["bbox"]) for block in lane_blocks]),
                }
            )
    return lane_rows


def _row_vertical_gap(previous_row: dict[str, Any], current_row: dict[str, Any]) -> float:
    previous_bbox = previous_row.get("bbox", [0.0, 0.0, 0.0, 0.0])
    current_bbox = current_row.get("bbox", [0.0, 0.0, 0.0, 0.0])
    return float(current_bbox[1]) - float(previous_bbox[3])


def _extract_numeric_marker_from_blocks(blocks: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    if not blocks:
        return None, []

    marker_blocks = [
        block
        for block in sorted(blocks, key=lambda item: float(item["bbox"][0]))
        if re.fullmatch(r"[\s()0-9]+", _clean_text(str(block.get("text", "") or "")))
    ]
    for start_index in range(len(marker_blocks)):
        for width in (3, 2, 1):
            window = marker_blocks[start_index : start_index + width]
            if len(window) != width:
                continue
            gaps = [
                float(window[idx + 1]["bbox"][0]) - float(window[idx]["bbox"][2])
                for idx in range(len(window) - 1)
            ]
            if any(gap > 18.0 for gap in gaps):
                continue
            candidate = "".join(_clean_text(str(block.get("text", "") or "")) for block in window)
            normalized = _normalized_equation_marker_text(candidate)
            if normalized is None:
                continue
            source_block_ids = [
                str(block.get("block_id", "")).strip()
                for block in window
                if str(block.get("block_id", "")).strip()
            ]
            return normalized, source_block_ids
    return None, []


def _find_external_numeric_marker(
    row: dict[str, Any],
    lane_rows: list[dict[str, Any]],
    page_words: list[_Word] | None = None,
) -> tuple[str | None, list[str]]:
    row_index = int(row["row_index"])
    current_bbox = row.get("bbox", [0.0, 0.0, 0.0, 0.0])
    search_rows = [
        sibling
        for sibling in lane_rows
        if abs(int(sibling["row_index"]) - row_index) <= 1 and sibling is not row
    ]
    for sibling in search_rows:
        filtered_blocks = [
            block
            for block in sibling["blocks"]
            if float(block["bbox"][0]) >= float(current_bbox[2]) - 8.0
            and float(block["bbox"][2]) <= float(current_bbox[2]) + 190.0
        ]
        marker_text, marker_source_block_ids = _extract_numeric_marker_from_blocks(filtered_blocks)
        if marker_text is not None:
            return marker_text, marker_source_block_ids
    if page_words:
        row_bbox = row.get("bbox", [0.0, 0.0, 0.0, 0.0])
        marker_words = [
            word
            for word in page_words
            if float(word.x0) >= float(row_bbox[2]) + 20.0
            and float(word.x1) <= float(row_bbox[2]) + 190.0
            and float(word.y0) <= float(row_bbox[3]) + 8.0
            and float(word.y1) >= float(row_bbox[1]) - 8.0
            and re.fullmatch(r"[\s()0-9]+", _clean_text(str(word.text or "")))
        ]
        marker_words = sorted(marker_words, key=lambda item: float(item.x0))
        for start_index in range(len(marker_words)):
            for width in (3, 2, 1):
                window = marker_words[start_index : start_index + width]
                if len(window) != width:
                    continue
                gaps = [
                    float(window[idx + 1].x0) - float(window[idx].x1)
                    for idx in range(len(window) - 1)
                ]
                if any(gap > 18.0 for gap in gaps):
                    continue
                candidate = "".join(_clean_text(str(word.text or "")) for word in window)
                normalized = _normalized_equation_marker_text(candidate)
                if normalized is not None:
                    return normalized, []
    return None, []


def _trim_display_equation_cluster_text(text: str, equation_label: str | None) -> str:
    compact = _strip_display_equation_terminal_artifacts(text)
    lowered = compact.lower()
    if "defined as the following:" in lowered:
        phrase = "defined as the following:"
        index = lowered.rfind(phrase)
        if index >= 0:
            compact = compact[index + len(phrase) :].strip()

    if equation_label in {"(3)", "(4)", "(5)", "(13)"} and ":" in compact:
        candidate = compact.split(":")[-1].strip()
        candidate_lowered = candidate.lower()
        if any(token in candidate_lowered for token in ("s b =", "s w =", "d w", "min ", "c j =", "z i =", "d (")):
            compact = candidate
    return _strip_display_equation_terminal_artifacts(compact)


def _clean_display_equation_text(text: str, equation_label: str | None) -> str:
    compact = _strip_display_equation_terminal_artifacts(text)
    lowered = compact.lower()

    if equation_label in {"(8)", "(9)", "(11)"}:
        if lowered.startswith("n min "):
            compact = compact[2:].lstrip()
            lowered = compact.lower()
        elif " min v " in f" {lowered} " and lowered.startswith("n "):
            compact = compact[2:].lstrip()
            lowered = compact.lower()

    if equation_label == "(5)" and "is d w" in lowered:
        index = lowered.find("d w ( x")
        if index >= 0:
            compact = compact[index:]
            lowered = compact.lower()

    if equation_label == "(13)" and ":" in compact:
        candidate = compact.split(":", 1)[1].strip()
        if candidate:
            compact = candidate
            lowered = compact.lower()
    if equation_label == "(13)" and not lowered.startswith("d "):
        compact = f"d {compact}".strip()
        lowered = compact.lower()

    while compact.endswith(" (") or compact.endswith("("):
        compact = compact[:-1].rstrip()
    return _strip_display_equation_terminal_artifacts(compact)


def _build_numbered_display_equation_cluster(
    cluster_rows: list[dict[str, Any]],
    marker_text: str,
    marker_source_block_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    if not cluster_rows:
        return None

    cluster_blocks = [block for row in cluster_rows for block in row["blocks"]]
    text_rows = [_row_compact_text(row["blocks"]) for row in cluster_rows if _row_compact_text(row["blocks"])]
    original_equation_text = _strip_display_equation_terminal_artifacts(" ".join(text_rows).strip())
    equation_text = _clean_display_equation_text(
        _trim_display_equation_cluster_text(original_equation_text, marker_text),
        marker_text,
    )
    lowered = equation_text.lower()
    if not equation_text or lowered.startswith(("where ", "with ", "set, respectively")):
        return None
    if _looks_like_equation_false_positive_text(equation_text):
        return None
    if not re.search(r"[=∑]", equation_text) and "min" not in lowered:
        return None

    source_block_ids = [
        str(block.get("block_id", "")).strip()
        for block in cluster_blocks
        if str(block.get("block_id", "")).strip()
    ]
    if marker_source_block_ids:
        source_block_ids.extend(marker_source_block_ids)
        source_block_ids = list(dict.fromkeys(source_block_ids))
    if not source_block_ids:
        return None

    first_block_id = source_block_ids[0]
    equation_block_id = _build_equation_block_id(first_block_id)
    preserve_source_text = equation_text != original_equation_text and any(
        cue in original_equation_text.lower()
        for cue in (
            "for the purpose of this paper",
            "defined as the following",
            "within-class distance are defined as",
        )
    )
    return {
        **dict(cluster_blocks[0]),
        "block_id": equation_block_id,
        "equation_id": equation_block_id,
        "block_type": "equation",
        "semantic_role": "display_equation",
        "unit_role": "equation",
        "text": equation_text,
        "bbox": _bbox_union([tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0))) for block in cluster_blocks]),
        "equation_label": marker_text,
        "source_block_ids": source_block_ids,
        "_cluster_first_source_block_id": first_block_id,
        "_preserve_source_text": preserve_source_text,
    }


def _promote_numbered_display_equation_clusters(
    text_blocks: list[dict[str, Any]],
    page_words: list[_Word] | None = None,
) -> list[dict[str, Any]]:
    if not text_blocks:
        return []

    available_blocks = [
        dict(block)
        for block in text_blocks
        if str(block.get("block_type", "text") or "text") != "equation"
    ]
    if not available_blocks:
        return list(text_blocks)

    lane_rows = _lane_rows_for_equation_detection(available_blocks)
    rows_by_lane: dict[str, list[dict[str, Any]]] = {}
    for row in lane_rows:
        rows_by_lane.setdefault(str(row["lane"]), []).append(row)

    promoted_equations: list[dict[str, Any]] = []
    consumed_source_block_ids: set[str] = set()
    consumed_row_keys: set[tuple[str, int]] = set()
    all_lane_rows = list(lane_rows)
    for lane, lane_rows in rows_by_lane.items():
        lane_rows = sorted(lane_rows, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))
        for row_index, row in enumerate(lane_rows):
            row_key = (lane, int(row["row_index"]))
            if row_key in consumed_row_keys:
                continue
            internal_marker_text, internal_marker_source_block_ids = _extract_numeric_marker_from_blocks(row["blocks"])
            external_marker_text, external_marker_source_block_ids = _find_external_numeric_marker(
                row,
                all_lane_rows,
                page_words=page_words,
            )
            marker_text = internal_marker_text or external_marker_text
            marker_source_block_ids = internal_marker_source_block_ids or external_marker_source_block_ids
            if marker_text is None:
                continue
            if not (
                _looks_like_display_equation_cluster_row(row["blocks"])
                or any(
                    _looks_like_equation_marker_text(str(block.get("text", "") or "").strip())
                    for block in row["blocks"]
                )
            ):
                continue

            start_index = row_index
            while start_index - 1 >= 0:
                previous_row = lane_rows[start_index - 1]
                previous_key = (lane, int(previous_row["row_index"]))
                if previous_key in consumed_row_keys:
                    break
                gap = _row_vertical_gap(previous_row, lane_rows[start_index])
                previous_height = _bbox_height(previous_row.get("bbox", []))
                current_height = _bbox_height(lane_rows[start_index].get("bbox", []))
                max_gap = max(18.0, min(previous_height, current_height) * 2.1)
                if gap > max_gap:
                    break
                if not (
                    _looks_like_display_equation_cluster_row(previous_row["blocks"])
                    or _looks_like_display_equation_support_row(previous_row["blocks"])
                ):
                    break
                start_index -= 1

            cluster_rows = lane_rows[start_index : row_index + 1]
            equation_block = _build_numbered_display_equation_cluster(
                cluster_rows,
                marker_text=marker_text,
                marker_source_block_ids=marker_source_block_ids,
            )
            if equation_block is None:
                continue
            promoted_equations.append(equation_block)
            for cluster_row in cluster_rows:
                consumed_row_keys.add((lane, int(cluster_row["row_index"])))
            if not bool(equation_block.get("_preserve_source_text", False)):
                consumed_source_block_ids.update(equation_block.get("source_block_ids", []))

    if not promoted_equations:
        return list(text_blocks)

    equations_by_first_source: dict[str, dict[str, Any]] = {}
    for equation_block in promoted_equations:
        first_source = str(equation_block.pop("_cluster_first_source_block_id", "")).strip()
        equation_block.pop("_preserve_source_text", None)
        if first_source:
            equations_by_first_source[first_source] = equation_block

    projected: list[dict[str, Any]] = []
    for block in text_blocks:
        block_id = str(block.get("block_id", "")).strip()
        equation_block = equations_by_first_source.get(block_id)
        if equation_block is not None:
            projected.append(equation_block)
        if block_id in consumed_source_block_ids:
            continue
        projected.append(dict(block))
    return _consolidate_display_equation_blocks(projected)


def _can_attach_equation_fragment(
    equation_bbox: list[float] | tuple[float, float, float, float],
    fragment_block: dict[str, Any],
    *,
    page_width: float,
    side: str,
) -> bool:
    fragment_bbox = list(fragment_block.get("bbox", []))
    if len(equation_bbox) < 4 or len(fragment_bbox) < 4:
        return False

    equation_height = _bbox_height(equation_bbox)
    fragment_height = _bbox_height(fragment_bbox)
    row_tolerance = max(8.0, max(equation_height, fragment_height) * 1.1)
    if abs(_bbox_center_y(equation_bbox) - _bbox_center_y(fragment_bbox)) > row_tolerance:
        return False

    allowed_overlap = max(6.0, min(equation_height, fragment_height) * 0.45)
    max_gap = max(28.0, page_width * 0.05, min(equation_height, fragment_height) * 2.4)
    if side == "left":
        if float(fragment_bbox[0]) > float(equation_bbox[0]) + 2.0:
            return False
        gap = float(equation_bbox[0]) - float(fragment_bbox[2])
    else:
        if float(fragment_bbox[2]) < float(equation_bbox[2]) - 2.0:
            return False
        gap = float(fragment_bbox[0]) - float(equation_bbox[2])
    return -allowed_overlap <= gap <= max_gap


def _can_attach_equation_marker(
    equation_bbox: list[float] | tuple[float, float, float, float],
    marker_block: dict[str, Any],
    *,
    page_width: float,
) -> bool:
    marker_bbox = list(marker_block.get("bbox", []))
    if len(equation_bbox) < 4 or len(marker_bbox) < 4:
        return False
    if not _looks_like_equation_marker_text(str(marker_block.get("text", "") or "")):
        return False
    equation_height = _bbox_height(equation_bbox)
    marker_height = _bbox_height(marker_bbox)
    row_tolerance = max(10.0, max(equation_height, marker_height) * 1.35)
    vertical_gap = float(marker_bbox[1]) - float(equation_bbox[3])
    if (
        abs(_bbox_center_y(equation_bbox) - _bbox_center_y(marker_bbox)) > row_tolerance
        and vertical_gap > max(8.0, marker_height * 1.1)
    ):
        return False
    equation_width = max(1.0, float(equation_bbox[2]) - float(equation_bbox[0]))
    if float(marker_bbox[0]) < float(equation_bbox[0]) + equation_width * 0.5:
        return False
    if float(marker_bbox[2]) < float(equation_bbox[2]) - max(36.0, equation_width * 0.35):
        return False
    max_gap = max(36.0, page_width * 0.25)
    return (float(marker_bbox[0]) - float(equation_bbox[2])) <= max_gap


def _consolidate_display_equation_blocks(
    text_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not text_blocks:
        return []

    consolidated: list[dict[str, Any]] = []
    for block in text_blocks:
        if (
            consolidated
            and str(block.get("block_type", "text") or "text") == "equation"
            and str(consolidated[-1].get("block_type", "text") or "text") == "equation"
            and int(block.get("page", 0) or 0) == int(consolidated[-1].get("page", 0) or 0)
            and str(block.get("equation_label", "") or "").strip()
            and str(block.get("equation_label", "") or "").strip()
            == str(consolidated[-1].get("equation_label", "") or "").strip()
        ):
            previous = consolidated[-1]
            previous_text = _clean_text(str(previous.get("text", "") or ""))
            current_text = _clean_text(str(block.get("text", "") or ""))
            if current_text and current_text not in previous_text:
                if previous_text and previous_text not in current_text:
                    previous["text"] = f"{previous_text} {current_text}".strip()
                else:
                    previous["text"] = current_text
            previous["bbox"] = _bbox_union(
                [tuple(previous.get("bbox", (0.0, 0.0, 0.0, 0.0))), tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))]
            )
            previous["source_block_ids"] = list(
                dict.fromkeys(
                    list(previous.get("source_block_ids", []) or [])
                    + list(block.get("source_block_ids", []) or [])
                )
            )
            continue
        consolidated.append(dict(block))
    return consolidated


def _promote_display_equation_blocks(
    text_blocks: list[dict[str, Any]],
    *,
    page_width: float,
    page_words: list[_Word] | None = None,
) -> list[dict[str, Any]]:
    if not text_blocks:
        return []

    text_blocks = _promote_numbered_display_equation_clusters(text_blocks, page_words=page_words)

    projected: list[dict[str, Any]] = []
    skip_indices: set[int] = set()

    for idx, text_block in enumerate(text_blocks):
        if idx in skip_indices:
            continue

        if str(text_block.get("block_type", "text") or "text") == "equation":
            projected.append(dict(text_block))
            continue

        text = _clean_text(str(text_block.get("text", "")))
        if not _looks_like_display_equation_text(text):
            projected.append(dict(text_block))
            continue

        equation_block = dict(text_block)
        equation_block_id = _build_equation_block_id(str(text_block.get("block_id", "") or ""))
        source_block_ids = [str(text_block.get("block_id", "")).strip()]
        candidate_source_block_indices = set(_normalized_source_block_indices(text_block))
        equation_label = None
        bbox_sources: list[list[float] | tuple[float, float, float, float]] = [list(text_block.get("bbox", []))]
        candidate_absorbed_left_block_ids: set[str] = set()
        candidate_skip_indices: set[int] = set()
        accepted_as_equation = False

        left_idx = idx - 1
        while left_idx >= 0 and left_idx not in skip_indices:
            left_block = text_blocks[left_idx]
            if str(left_block.get("block_type", "text") or "text") == "equation":
                break
            left_text = _clean_text(str(left_block.get("text", "") or ""))
            if not _looks_like_equation_adjacent_fragment(left_text):
                break
            equation_bbox = _bbox_union(bbox_sources)
            if not _can_attach_equation_fragment(
                equation_bbox,
                left_block,
                page_width=page_width,
                side="left",
            ):
                break
            block_id = str(left_block.get("block_id", "")).strip()
            if block_id:
                candidate_absorbed_left_block_ids.add(block_id)
                source_block_ids.insert(0, block_id)
            candidate_source_block_indices.update(_normalized_source_block_indices(left_block))
            bbox_sources.insert(0, list(left_block.get("bbox", [])))
            text = f"{left_text} {text}"
            left_idx -= 1

        right_idx = idx + 1
        while right_idx < len(text_blocks) and right_idx not in skip_indices:
            right_block = text_blocks[right_idx]
            if str(right_block.get("block_type", "text") or "text") == "equation":
                break
            right_text = _clean_text(str(right_block.get("text", "") or ""))
            if _looks_like_equation_marker_text(right_text):
                equation_bbox = _bbox_union(bbox_sources)
                if not _can_attach_equation_marker(
                    equation_bbox,
                    right_block,
                    page_width=page_width,
                ):
                    break
                text = f"{_strip_display_equation_terminal_artifacts(text)} {right_text}"
                equation_label = _normalized_equation_marker_text(right_text) or right_text
                source_block_ids.append(str(right_block.get("block_id", "")).strip())
                candidate_source_block_indices.update(_normalized_source_block_indices(right_block))
                bbox_sources.append(list(right_block.get("bbox", [])))
                candidate_skip_indices.add(right_idx)
                break

            if not _looks_like_equation_adjacent_fragment(right_text):
                break
            equation_bbox = _bbox_union(bbox_sources)
            if not _can_attach_equation_fragment(
                equation_bbox,
                right_block,
                page_width=page_width,
                side="right",
            ):
                break
            text = f"{text} {right_text}"
            source_block_ids.append(str(right_block.get("block_id", "")).strip())
            candidate_source_block_indices.update(_normalized_source_block_indices(right_block))
            bbox_sources.append(list(right_block.get("bbox", [])))
            candidate_skip_indices.add(right_idx)
            right_idx += 1

        text = _clean_display_equation_text(_strip_display_equation_terminal_artifacts(text), equation_label)
        lowered = text.lower()
        alpha_tokens = re.findall(r"[A-Za-z]+", text)
        prose_cue_count = sum(1 for cue in _DISPLAY_EQUATION_CLUSTER_PROSE_CUES if cue in lowered)
        if equation_label is None and prose_cue_count >= 1 and len(alpha_tokens) >= 4:
            projected.append(dict(text_block))
            continue
        if _looks_like_equation_false_positive_text(text):
            projected.append(dict(text_block))
            continue

        equation_block["block_id"] = equation_block_id
        equation_block["equation_id"] = equation_block_id
        equation_block["block_type"] = "equation"
        equation_block["semantic_role"] = "display_equation"
        equation_block["unit_role"] = "equation"
        equation_block["text"] = text
        equation_block["bbox"] = _bbox_union(bbox_sources)
        equation_block["equation_label"] = equation_label
        equation_block["source_block_ids"] = source_block_ids
        equation_block["source_block_indices"] = sorted(candidate_source_block_indices)
        accepted_as_equation = True

        if accepted_as_equation:
            if candidate_absorbed_left_block_ids:
                projected = [
                    block
                    for block in projected
                    if str(block.get("block_id", "")).strip() not in candidate_absorbed_left_block_ids
                ]
            skip_indices.update(candidate_skip_indices)
        projected.append(equation_block)

    projected = _attach_equation_continuation_fragments(projected)
    return _drop_residual_equation_text_fragments(projected)


def _build_equation_projection(text_block: dict[str, Any]) -> dict[str, Any]:
    equation_id = str(text_block.get("equation_id") or text_block.get("block_id") or "").strip()
    return {
        "equation_id": equation_id,
        "page": int(text_block.get("page", 0) or 0),
        "bbox": list(text_block.get("bbox", [])),
        "text": str(text_block.get("text", "")).strip(),
        "equation_label": str(text_block.get("equation_label", "") or "").strip() or None,
        "source": text_block.get("source", "text-layer"),
        "semantic_role": "display_equation",
        "source_block_ids": list(text_block.get("source_block_ids", []) or []),
    }


def _build_text_content_evidence(text_block: dict[str, Any]) -> dict[str, Any]:
    text = str(text_block.get("text", "")).strip()
    block_id = str(text_block.get("block_id", "")).strip()
    semantic_role, unit_role = _classify_text_block_semantic_role(text_block)
    if semantic_role == "algorithm_pseudocode":
        lines = [
            _clean_text(str(line))
            for line in list(text_block.get("lines", []) or [])
            if _clean_text(str(line))
        ]
        if not lines and text:
            lines = [_clean_text(line) for line in text.splitlines() if _clean_text(line)]
        title = _clean_text(str(text_block.get("title", "") or ""))
        algorithm_ref = str(text_block.get("algorithm_ref", "") or "").strip()
        segments: list[dict[str, Any]] = []
        line_start_index = 0
        if title:
            segments.append({"role": "title", "text": title, "algorithm_ref": algorithm_ref})
            if lines and _clean_text(lines[0]) == title:
                line_start_index = 1
        for line_index, line in enumerate(lines[line_start_index:], start=1):
            segments.append(
                {
                    "role": "step",
                    "text": line,
                    "step_index": line_index,
                    "algorithm_ref": algorithm_ref,
                }
            )
        if not segments and text:
            segments.append({"role": "step", "text": text, "algorithm_ref": algorithm_ref})
        evidence = {
            "evidence_id": f"ce_algorithm_{block_id}",
            "source_type": "algorithm",
            "source_id": block_id,
            "page": int(text_block.get("page", 0) or 0),
            "bbox": list(text_block.get("bbox", [])),
            "semantic_role": semantic_role,
            "content_text": text,
            "segments": segments,
            "source": text_block.get("source", "text-layer"),
            "algorithm_ref": algorithm_ref,
            "title": title,
            "continued_from_previous_page": bool(text_block.get("continued_from_previous_page", False)),
            "continues_to_next_page": bool(text_block.get("continues_to_next_page", False)),
        }
        return evidence
    segment: dict[str, Any] = {"role": unit_role, "text": text}
    if semantic_role == "display_equation":
        equation_label = str(text_block.get("equation_label", "") or "").strip()
        if equation_label:
            segment["equation_label"] = equation_label
        source_block_ids = [str(item).strip() for item in text_block.get("source_block_ids", []) or [] if str(item).strip()]
        if source_block_ids:
            segment["source_block_ids"] = source_block_ids
    elif semantic_role == "reference_entry":
        reference_number = str(text_block.get("reference_number", "") or "").strip()
        if reference_number:
            segment["reference_number"] = reference_number
        reference_entry_index = int(text_block.get("reference_entry_index", 0) or 0)
        if reference_entry_index > 0:
            segment["reference_entry_index"] = reference_entry_index
        if bool(text_block.get("reference_entry_start")):
            segment["entry_start"] = True
        if bool(text_block.get("reference_continuation")):
            segment["continuation"] = True
        if bool(text_block.get("reference_page_continuation")):
            segment["page_continuation"] = True
    return {
        "evidence_id": f"ce_text_{block_id}",
        "source_type": "text",
        "source_id": block_id,
        "page": int(text_block.get("page", 0) or 0),
        "bbox": list(text_block.get("bbox", [])),
        "semantic_role": semantic_role,
        "content_text": text,
        "segments": [segment],
        "source": text_block.get("source", "text-layer"),
    }


def _build_image_content_evidence(image_block: dict[str, Any]) -> dict[str, Any]:
    image_id = str(image_block.get("image_id", "")).strip()
    segments = [dict(segment) for segment in image_block.get("content_segments", []) or []]
    return {
        "evidence_id": f"ce_image_{image_id}",
        "source_type": "image",
        "source_id": image_id,
        "page": int(image_block.get("page", 0) or 0),
        "bbox": list(image_block.get("bbox", [])),
        "semantic_role": image_block.get("image_kind_guess", "image"),
        "content_text": str(image_block.get("content_text", "")).strip(),
        "segments": segments,
        "figure_ref": image_block.get("figure_ref"),
        "title": image_block.get("title", ""),
        "caption_text": image_block.get("caption_text", ""),
        "embedded_text": image_block.get("embedded_text", ""),
        "embedded_text_source": image_block.get("embedded_text_source"),
        "embedded_text_confidence": image_block.get("embedded_text_confidence"),
        "nearby_context_text": image_block.get("nearby_context_text", ""),
        "content_signals": dict(image_block.get("content_signals", {}) or {}),
    }


def _build_table_content_evidence(table_ast: dict[str, Any]) -> dict[str, Any]:
    table_id = str(table_ast.get("table_id", "")).strip()
    title = str(table_ast.get("title", "") or "").strip()
    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in table_ast.get("header", [])
        if str(cell.get("text", "")).strip()
    ]
    row_texts = [
        str(row_text).strip()
        for row_text in (table_ast.get("data_row_texts") or table_ast.get("row_texts") or [])
        if str(row_text).strip()
    ]
    segments: list[dict[str, Any]] = []
    if title:
        segments.append({"role": "title", "text": title})
    if header_texts:
        segments.append({"role": "header", "text": " | ".join(header_texts), "cells": header_texts})
    for row_index, row_text in enumerate(row_texts, start=1):
        segments.append({"role": "row", "row_index": row_index, "text": row_text})
    non_title_segments = [segment for segment in segments if str(segment.get("role", "") or "") != "title"]
    content_text = _join_content_segments(non_title_segments) or _join_content_segments(segments)
    return {
        "evidence_id": f"ce_table_{table_id}",
        "source_type": "table",
        "source_id": table_id,
        "page": int(table_ast.get("page", 0) or 0),
        "bbox": list(table_ast.get("bbox", [])),
        "semantic_role": table_ast.get("semantic_role", "business_table"),
        "content_text": content_text,
        "segments": segments,
        "title": title,
        "header_texts": header_texts,
        "row_count": int(table_ast.get("row_count", 0) or 0),
        "col_count": int(table_ast.get("col_count", 0) or 0),
    }


def _build_toc_content_evidence(toc_block: dict[str, Any]) -> dict[str, Any]:
    toc_id = str(toc_block.get("toc_id", "")).strip()
    title = str(toc_block.get("title") or toc_block.get("toc_title") or "").strip()
    title_inferred = bool(toc_block.get("title_inferred"))
    entries = list(toc_block.get("entries", []) or [])
    segments: list[dict[str, Any]] = []
    if title and not title_inferred:
        segments.append({"role": "title", "text": title})
    for entry in entries:
        outline_index = str(entry.get("outline_index") or "").strip()
        entry_text = str(entry.get("text") or "").strip()
        page_locator = str(entry.get("page_locator") or "").strip()
        line_parts = [part for part in (outline_index, entry_text, page_locator) if part]
        if not line_parts:
            continue
        segments.append(
            {
                "role": "entry",
                "entry_index": int(entry.get("entry_index", 0) or 0),
                "text": " | ".join(line_parts),
                "outline_index": entry.get("outline_index"),
                "page_locator": entry.get("page_locator"),
                "level": entry.get("level"),
            }
        )
    content_text = _join_content_segments(segments)
    return {
        "evidence_id": f"ce_toc_{toc_id}",
        "source_type": "toc",
        "source_id": toc_id,
        "page": int(toc_block.get("page", 0) or 0),
        "bbox": list(toc_block.get("bbox", [])),
        "semantic_role": toc_block.get("semantic_role", "toc_outline"),
        "content_text": content_text,
        "segments": segments,
        "title": title,
        "title_inferred": title_inferred,
        "entry_count": int(toc_block.get("entry_count", 0) or 0),
        "review_required": bool((toc_block.get("toc_diagnostics") or {}).get("review_required", False)),
    }


def _promote_page_text_toc_fragments(page_payload: dict[str, Any]) -> int:
    toc_blocks = list(page_payload.get("toc_blocks", []) or [])
    text_blocks = list(page_payload.get("text_blocks", []) or [])
    if not toc_blocks or not text_blocks:
        return 0

    occupied_bboxes = [
        tuple(float(value) for value in toc_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        for toc_block in toc_blocks
        if len(toc_block.get("bbox", [])) == 4
    ]
    extra_entries = _build_toc_text_block_entries(text_blocks, occupied_bboxes=occupied_bboxes)
    extra_entries.extend(_build_embedded_toc_text_gap_entries(text_blocks, toc_blocks))
    extra_entries.extend(_build_toc_word_gap_entries(page_payload, toc_blocks))
    if not extra_entries:
        return 0

    assigned_entries: dict[str, list[dict[str, Any]]] = {
        str(toc_block.get("toc_id") or ""): []
        for toc_block in toc_blocks
    }
    existing_entry_keys = {
        str(toc_block.get("toc_id") or ""): {
            _normalized_toc_entry_key(entry)
            for entry in toc_block.get("_raw_entries", []) or []
        }
        for toc_block in toc_blocks
    }

    promoted_count = 0
    for extra_entry in extra_entries:
        target_block = _select_target_toc_block(toc_blocks, extra_entry)
        if not target_block:
            continue
        toc_id = str(target_block.get("toc_id") or "")
        entry_key = _normalized_toc_entry_key(extra_entry)
        if not toc_id or entry_key in existing_entry_keys.get(toc_id, set()):
            continue
        existing_entry_keys.setdefault(toc_id, set()).add(entry_key)
        assigned_entries.setdefault(toc_id, []).append(extra_entry)
        promoted_count += 1

    if not promoted_count:
        return 0

    for toc_block in toc_blocks:
        toc_id = str(toc_block.get("toc_id") or "")
        extras = assigned_entries.get(toc_id, [])
        if not extras:
            continue
        combined_raw_entries = list(toc_block.get("_raw_entries", []) or []) + extras
        _populate_toc_block_from_raw_entries(toc_block, combined_raw_entries)
        promoted_block_ids = list(toc_block.get("promoted_text_block_ids", []) or [])
        for extra in extras:
            promoted_block_ids.extend(_toc_entry_source_block_ids(extra))
        toc_block["promoted_text_block_ids"] = list(dict.fromkeys(promoted_block_ids))

    return promoted_count


def _build_embedded_toc_text_gap_entries(
    text_blocks: list[dict[str, Any]],
    toc_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not text_blocks or not toc_blocks:
        return []

    all_text_entries = _build_toc_text_block_entries(text_blocks, allow_embedded_rows=True)
    if not all_text_entries:
        return []

    embedded_entries: list[dict[str, Any]] = []
    for text_entry in all_text_entries:
        if not _is_missing_toc_outline_gap_entry(text_entry, toc_blocks):
            continue
        embedded_entries.append(text_entry)
    return embedded_entries


def _build_toc_word_gap_entries(
    page_payload: dict[str, Any],
    toc_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    page_words = list(page_payload.get("page_words", []) or [])
    if not page_words or not toc_blocks:
        return []

    row_entries: list[dict[str, Any]] = []
    for row_words in _group_page_words_by_visual_rows(page_words):
        candidate_entry = _build_toc_word_row_entry(row_words)
        if not candidate_entry:
            continue
        if _is_missing_toc_outline_gap_entry(candidate_entry, toc_blocks):
            row_entries.append(candidate_entry)

    return row_entries


def _group_page_words_by_visual_rows(page_words: list[_Word]) -> list[list[_Word]]:
    if not page_words:
        return []

    sorted_words = sorted(page_words, key=lambda word: (float(word.y0), float(word.x0)))
    median_height = _median_word_height(sorted_words)
    row_tolerance = max(3.0, min(10.0, median_height * 0.65))
    rows: list[dict[str, Any]] = []
    for word in sorted_words:
        word_center_y = (float(word.y0) + float(word.y1)) / 2
        best_row = None
        best_distance = float("inf")
        for row in rows:
            row_center_y = float(row["center_y"])
            distance = abs(word_center_y - row_center_y)
            if distance > row_tolerance:
                continue
            if distance < best_distance:
                best_distance = distance
                best_row = row
        if best_row is None:
            rows.append({"center_y": word_center_y, "words": [word], "count": 1})
            continue
        best_row["words"].append(word)
        best_row["count"] += 1
        best_row["center_y"] = (
            (float(best_row["center_y"]) * (int(best_row["count"]) - 1)) + word_center_y
        ) / int(best_row["count"])

    return [sorted(row["words"], key=lambda word: float(word.x0)) for row in rows]


def _median_word_height(page_words: list[_Word]) -> float:
    heights = sorted(max(1.0, float(word.y1) - float(word.y0)) for word in page_words)
    if not heights:
        return 10.0
    return heights[len(heights) // 2]


def _build_toc_word_row_entry(row_words: list[_Word]) -> dict[str, Any] | None:
    if len(row_words) < 3:
        return None

    ordered_words = sorted(row_words, key=lambda word: float(word.x0))
    locator_word = ordered_words[-1]
    page_locator = str(locator_word.text or "").strip()
    if not re.fullmatch(r"\d{1,4}|[ivxlcdm]{1,12}", page_locator, re.IGNORECASE):
        return None

    content_parts = [
        str(word.text or "").strip()
        for word in ordered_words[:-1]
        if str(word.text or "").strip()
        and not _looks_like_toc_leader_word(str(word.text or "").strip())
    ]
    if not content_parts:
        return None
    content_text = re.sub(r"\s+", " ", " ".join(content_parts)).strip()
    outline_index, outline_depth, entry_text, _ = _extract_toc_word_outline_prefix(content_text)
    if not outline_index or not entry_text:
        return None

    row_bbox = _bbox_union([(float(word.x0), float(word.y0), float(word.x1), float(word.y1)) for word in ordered_words])
    locator_kind = "arabic" if page_locator.isdigit() else "roman"
    return {
        "source_row_indices": [],
        "source_block_ids": [],
        "source_kind": "word_row",
        "leading_anchor": round(float(row_bbox[0]), 1),
        "outline_index": outline_index,
        "outline_depth": outline_depth,
        "text": entry_text,
        "page_locator": page_locator,
        "page_locator_kind": locator_kind,
        "page_locator_value": int(page_locator) if page_locator.isdigit() else None,
        "sort_y0": float(row_bbox[1]),
        "sort_x0": float(row_bbox[0]),
        "bbox": _bbox_to_list(row_bbox),
    }


def _looks_like_toc_leader_word(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(re.fullmatch(r"[.\u2026·•]{3,}", candidate))


def _extract_toc_word_outline_prefix(text: str) -> tuple[str | None, int, str, str | None]:
    candidate = str(text or "").strip()
    if not candidate:
        return None, 0, "", None

    match = re.match(r"^(\d+(?:\.\d+)+)\b", candidate)
    if match:
        outline_index = match.group(1)
        remainder = candidate[match.end():].lstrip(" .:-")
        return outline_index, len([segment for segment in outline_index.split(".") if segment]), remainder, "numeric"

    match = re.match(r"^(\d+)\.(?:\s+|$)", candidate)
    if match:
        outline_index = f"{match.group(1)}.0"
        remainder = candidate[match.end():].strip()
        return outline_index, 2, remainder, "numeric"

    return None, 0, candidate, None


def _is_missing_toc_outline_gap_entry(
    candidate_entry: dict[str, Any],
    toc_blocks: list[dict[str, Any]],
) -> bool:
    candidate_outline = str(candidate_entry.get("outline_index") or "").strip()
    if not candidate_outline:
        return False
    if _parent_outline_index(candidate_outline):
        return False
    candidate_bbox = list(candidate_entry.get("bbox", []) or [])
    if len(candidate_bbox) != 4:
        return False

    existing_entries: list[dict[str, Any]] = []
    for toc_block in toc_blocks:
        existing_entries.extend(list(toc_block.get("entries", []) or []))
        existing_entries.extend(list(toc_block.get("_raw_entries", []) or []))

    candidate_key = _normalized_toc_entry_key(candidate_entry)
    existing_keys = {_normalized_toc_entry_key(entry) for entry in existing_entries}
    if candidate_key in existing_keys:
        return False

    child_prefix = candidate_outline[:-1] if candidate_outline.endswith(".0") else f"{candidate_outline}."
    has_child = any(
        str(entry.get("outline_index") or "").strip().startswith(child_prefix)
        and str(entry.get("outline_index") or "").strip() != candidate_outline
        for entry in existing_entries
    )
    if not has_child:
        return False

    target_block = _select_target_toc_block(toc_blocks, candidate_entry)
    if not target_block:
        return False

    target_entries = sorted(
        list(target_block.get("entries", []) or []) + list(target_block.get("_raw_entries", []) or []),
        key=_toc_entry_sort_key,
    )
    if not target_entries:
        return False

    candidate_top = float(candidate_bbox[1])
    before_entries = [
        entry
        for entry in target_entries
        if len(entry.get("bbox", []) or []) == 4
        and float(entry["bbox"][3]) <= candidate_top + 1.0
    ]
    after_entries = [
        entry
        for entry in target_entries
        if len(entry.get("bbox", []) or []) == 4
        and float(entry["bbox"][1]) >= candidate_top - 1.0
    ]
    has_previous_sibling_or_parent = any(
        _outline_is_before_gap(candidate_outline, str(entry.get("outline_index") or "").strip())
        for entry in before_entries
    )
    has_following_child = any(
        str(entry.get("outline_index") or "").strip().startswith(child_prefix)
        for entry in after_entries
    )
    return has_previous_sibling_or_parent and has_following_child


def _outline_is_before_gap(candidate_outline: str, previous_outline: str) -> bool:
    candidate_key = _toc_outline_sort_key(candidate_outline)
    previous_key = _toc_outline_sort_key(previous_outline)
    if candidate_key is None or previous_key is None:
        return False
    if candidate_key[0] != previous_key[0]:
        return False
    candidate_tuple = candidate_key[1]
    previous_tuple = previous_key[1]
    if not isinstance(candidate_tuple, tuple) or not isinstance(previous_tuple, tuple):
        return False
    if not candidate_tuple or not previous_tuple:
        return False
    return previous_tuple[0] <= candidate_tuple[0]


def _normalized_toc_entry_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _compact_text(str(entry.get("outline_index") or "")),
        _compact_text(str(entry.get("text") or "")),
        _compact_text(str(entry.get("page_locator") or "")),
    )


def _select_target_toc_block(
    toc_blocks: list[dict[str, Any]],
    candidate_entry: dict[str, Any],
) -> dict[str, Any] | None:
    if not toc_blocks:
        return None
    compatible_blocks = [
        toc_block
        for toc_block in toc_blocks
        if _can_attach_toc_entry_to_block(toc_block, candidate_entry)
    ]
    if not compatible_blocks:
        return None
    if len(compatible_blocks) == 1:
        return compatible_blocks[0]

    candidate_bbox = list(candidate_entry.get("bbox", []) or [])
    if len(candidate_bbox) != 4:
        return compatible_blocks[0]

    candidate_top = float(candidate_bbox[1])
    candidate_bottom = float(candidate_bbox[3])
    best_block = None
    best_distance = float("inf")
    for toc_block in compatible_blocks:
        toc_bbox = list(toc_block.get("bbox", []) or [])
        if len(toc_bbox) != 4:
            continue
        toc_top = float(toc_bbox[1])
        toc_bottom = float(toc_bbox[3])
        if candidate_bottom < toc_top:
            distance = toc_top - candidate_bottom
        elif candidate_top > toc_bottom:
            distance = candidate_top - toc_bottom
        else:
            distance = 0.0
        if distance < best_distance:
            best_distance = distance
            best_block = toc_block
    return best_block or compatible_blocks[0]


def _promote_cross_page_toc_edge_fragments(state: PdfPipelineState) -> int:
    if not state.page_payloads:
        return 0

    toc_index = _next_toc_index(state.toc_nodes)
    toc_blocks_by_page: dict[int, list[dict[str, Any]]] = {}
    for toc_block in state.toc_nodes:
        page_number = int(toc_block.get("page", 0) or 0)
        if page_number <= 0:
            continue
        toc_blocks_by_page.setdefault(page_number, []).append(toc_block)

    promoted_count = 0
    ordered_payloads = sorted(
        state.page_payloads,
        key=lambda payload: int(payload.get("page_number", 0) or 0),
    )
    for page_payload in ordered_payloads:
        page_number = int(page_payload.get("page_number", 0) or 0)
        if page_number <= 1:
            continue

        page_toc_blocks = list(page_payload.get("toc_blocks", []) or [])
        occupied_bboxes = [
            tuple(float(value) for value in toc_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            for toc_block in page_toc_blocks
            if len(toc_block.get("bbox", [])) == 4
        ]
        extra_entries = _build_toc_text_block_entries(
            list(page_payload.get("text_blocks", []) or []),
            occupied_bboxes=occupied_bboxes,
        )
        if not extra_entries:
            continue

        edge_entries = _collect_top_edge_toc_entries(page_payload, page_toc_blocks, extra_entries)
        if not edge_entries:
            continue

        previous_page_blocks = sorted(
            toc_blocks_by_page.get(page_number - 1, []),
            key=_toc_block_sort_key,
            reverse=True,
        )
        if not previous_page_blocks:
            continue

        synthetic_block = None
        for previous_block in previous_page_blocks:
            candidate_block = _build_promoted_toc_block(
                page_number=page_number,
                raw_entries=edge_entries,
                inherited_title=str(previous_block.get("title") or previous_block.get("toc_title") or "").strip(),
            )
            if _can_link_toc_sequence(previous_block, candidate_block):
                synthetic_block = candidate_block
                break

        if synthetic_block is None:
            continue

        toc_index += 1
        synthetic_block["toc_id"] = f"toc_{toc_index:03d}"
        page_payload.setdefault("toc_blocks", []).append(synthetic_block)
        page_payload["toc_blocks"] = sorted(page_payload["toc_blocks"], key=_toc_block_sort_key)
        state.toc_nodes.append(synthetic_block)
        toc_blocks_by_page.setdefault(page_number, []).append(synthetic_block)
        state.counters.toc_block_count += 1
        promoted_count += 1

    return promoted_count


def _collect_top_edge_toc_entries(
    page_payload: dict[str, Any],
    page_toc_blocks: list[dict[str, Any]],
    extra_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not extra_entries:
        return []

    page_height = float(page_payload.get("height", 0.0) or 0.0)
    first_toc_top = min(
        (
            float(toc_block["bbox"][1])
            for toc_block in page_toc_blocks
            if len(toc_block.get("bbox", []) or []) == 4
        ),
        default=float("inf"),
    )
    edge_limit = page_height * 0.22 if page_height > 0 else float("inf")
    if first_toc_top != float("inf"):
        edge_limit = min(edge_limit, first_toc_top)

    unassigned_entries = [
        entry
        for entry in sorted(extra_entries, key=_toc_entry_sort_key)
        if _select_target_toc_block(page_toc_blocks, entry) is None
    ]
    if not unassigned_entries:
        return []

    cluster: list[dict[str, Any]] = []
    last_bottom = None
    for entry in unassigned_entries:
        bbox = list(entry.get("bbox", []) or [])
        if len(bbox) != 4:
            continue
        if float(bbox[1]) > edge_limit:
            break
        if first_toc_top != float("inf") and float(bbox[3]) >= first_toc_top:
            break

        if last_bottom is not None:
            previous_bbox = list(cluster[-1].get("bbox", []) or [])
            previous_height = max(1.0, float(previous_bbox[3]) - float(previous_bbox[1])) if len(previous_bbox) == 4 else 12.0
            current_height = max(1.0, float(bbox[3]) - float(bbox[1]))
            allowed_gap = max(18.0, min(previous_height, current_height) * 1.8)
            if float(bbox[1]) - last_bottom > allowed_gap:
                break

        cluster.append(entry)
        last_bottom = float(bbox[3])

    return cluster


def _build_promoted_toc_block(
    *,
    page_number: int,
    raw_entries: list[dict[str, Any]],
    inherited_title: str,
) -> dict[str, Any]:
    promoted_block_ids: list[str] = []
    bbox_points: list[list[float]] = []
    for raw_entry in raw_entries:
        promoted_block_ids.extend(_toc_entry_source_block_ids(raw_entry))
        bbox = list(raw_entry.get("bbox", []) or [])
        if len(bbox) == 4:
            bbox_points.append(bbox)

    toc_block = {
        "block_type": "toc",
        "page": page_number,
        "bbox": bbox_points[0] if len(bbox_points) == 1 else [0.0, 0.0, 0.0, 0.0],
        "header": [],
        "cells": [],
        "grid": [],
        "display_grid": [],
        "raw_grid": [],
        "semantic_role": "toc_outline",
        "is_business_table": False,
        "toc_title": inherited_title or None,
        "title": inherited_title or None,
        "title_inferred": bool(inherited_title),
        "title_source": "previous_toc_sequence" if inherited_title else None,
        "source_candidate_id": None,
        "source_candidate_diagnostics": None,
        "promoted_text_block_ids": promoted_block_ids,
    }
    _populate_toc_block_from_raw_entries(toc_block, raw_entries)
    return toc_block


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


def _can_attach_toc_entry_to_block(
    toc_block: dict[str, Any],
    candidate_entry: dict[str, Any],
) -> bool:
    toc_bbox = list(toc_block.get("bbox", []) or [])
    candidate_bbox = list(candidate_entry.get("bbox", []) or [])
    if len(toc_bbox) != 4 or len(candidate_bbox) != 4:
        return True

    relation = _toc_entry_block_relation(toc_bbox, candidate_bbox)
    if relation == "overlap":
        return True

    entries = list(toc_block.get("entries", []) or [])
    if not entries:
        return True

    outlined_indices = {
        str(entry.get("outline_index") or "").strip()
        for entry in entries
        if str(entry.get("outline_index") or "").strip()
    }
    candidate_outline_index = str(candidate_entry.get("outline_index") or "").strip()
    candidate_parent_index = str(
        candidate_entry.get("outline_parent_index")
        or _parent_outline_index(candidate_outline_index)
        or ""
    ).strip()
    candidate_outline_key = _toc_outline_sort_key(candidate_outline_index)
    candidate_locator_key = _toc_entry_page_locator_key(candidate_entry)
    first_outline_key = _first_toc_entry_outline_key(entries)
    last_outline_key = _last_toc_entry_outline_key(entries)
    first_locator_key = _first_known_page_locator(entries)
    last_locator_key = _last_known_page_locator(entries)

    if relation == "above":
        if candidate_outline_key and first_outline_key and candidate_outline_key[0] == first_outline_key[0]:
            return candidate_outline_key[1] <= first_outline_key[1]
        if candidate_locator_key and first_locator_key and candidate_locator_key[0] == first_locator_key[0]:
            return candidate_locator_key[1] <= first_locator_key[1]
        return False

    if relation == "below":
        if candidate_parent_index and candidate_parent_index in outlined_indices:
            return True
        if candidate_outline_key and last_outline_key and candidate_outline_key[0] == last_outline_key[0]:
            return candidate_outline_key[1] >= last_outline_key[1]
        if candidate_locator_key and last_locator_key and candidate_locator_key[0] == last_locator_key[0]:
            return candidate_locator_key[1] >= last_locator_key[1]
        return False

    return False


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


def _first_toc_entry_outline_key(
    entries: list[dict[str, Any]],
) -> tuple[str, tuple[int, ...] | str] | None:
    for entry in entries:
        outline_key = _toc_outline_sort_key(str(entry.get("outline_index") or "").strip())
        if outline_key is not None:
            return outline_key
    return None


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


def _merge_same_page_toc_blocks(state: PdfPipelineState) -> int:
    merged_count = 0
    merged_toc_nodes: list[dict[str, Any]] = []

    for page_payload in state.page_payloads:
        page_toc_blocks = sorted(
            list(page_payload.get("toc_blocks", []) or []),
            key=_toc_block_sort_key,
        )
        if len(page_toc_blocks) < 2:
            merged_toc_nodes.extend(page_toc_blocks)
            continue

        merged_page_blocks: list[dict[str, Any]] = []
        current_block = page_toc_blocks[0]
        for next_block in page_toc_blocks[1:]:
            if _can_merge_same_page_toc_blocks(current_block, next_block):
                current_block = _merge_toc_blocks(current_block, next_block)
                merged_count += 1
                continue
            merged_page_blocks.append(current_block)
            current_block = next_block
        merged_page_blocks.append(current_block)

        page_payload["toc_blocks"] = merged_page_blocks
        merged_toc_nodes.extend(merged_page_blocks)

    state.toc_nodes = merged_toc_nodes
    state.counters.toc_block_count = len(merged_toc_nodes)
    return merged_count


def _can_merge_same_page_toc_blocks(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_page = int(previous_block.get("page", 0) or 0)
    current_page = int(current_block.get("page", 0) or 0)
    if previous_page <= 0 or current_page != previous_page:
        return False
    if _toc_titles_conflict(previous_block, current_block):
        return False

    previous_bbox = list(previous_block.get("bbox", []) or [])
    current_bbox = list(current_block.get("bbox", []) or [])
    if len(previous_bbox) != 4 or len(current_bbox) != 4:
        return False

    vertical_gap = float(current_bbox[1]) - float(previous_bbox[3])
    if vertical_gap < -8.0 or vertical_gap > 64.0:
        return False

    horizontal_overlap = _toc_horizontal_overlap_ratio(previous_bbox, current_bbox)
    if horizontal_overlap < 0.72:
        return False

    if _has_same_page_outline_parent_link(previous_block, current_block):
        return True
    if _has_forward_root_outline_progression(previous_block, current_block):
        return True
    if _has_forward_outline_progression(previous_block, current_block):
        return True
    return False


def _merge_toc_blocks(previous_block: dict[str, Any], current_block: dict[str, Any]) -> dict[str, Any]:
    merged_block = dict(previous_block)
    merged_raw_entries = list(previous_block.get("_raw_entries", []) or []) + list(current_block.get("_raw_entries", []) or [])
    merged_promoted_ids = list(previous_block.get("promoted_text_block_ids", []) or []) + list(
        current_block.get("promoted_text_block_ids", []) or []
    )
    merged_block["toc_title"] = previous_block.get("toc_title") or current_block.get("toc_title")
    merged_block["title"] = previous_block.get("title") or current_block.get("title")
    merged_block["title_inferred"] = bool(
        previous_block.get("title_inferred") or current_block.get("title_inferred")
    )
    merged_block["merged_toc_ids"] = list(
        dict.fromkeys(
            list(previous_block.get("merged_toc_ids", []) or [])
            + [previous_block.get("toc_id")]
            + list(current_block.get("merged_toc_ids", []) or [])
            + [current_block.get("toc_id")]
        )
    )
    _populate_toc_block_from_raw_entries(merged_block, merged_raw_entries)
    merged_block["promoted_text_block_ids"] = list(dict.fromkeys(item for item in merged_promoted_ids if item))
    return merged_block


def _stitch_cross_page_toc_sequences(toc_blocks: list[dict[str, Any]]) -> int:
    ordered_blocks = sorted(
        [toc_block for toc_block in toc_blocks if int(toc_block.get("page", 0) or 0) > 0],
        key=lambda toc_block: (
            int(toc_block.get("page", 0) or 0),
            float((toc_block.get("bbox", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[1]),
            float((toc_block.get("bbox", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[0]),
        ),
    )
    if not ordered_blocks:
        return 0

    sequences: list[list[dict[str, Any]]] = []
    current_sequence: list[dict[str, Any]] = []
    for toc_block in ordered_blocks:
        if current_sequence and _can_link_toc_sequence(current_sequence[-1], toc_block):
            current_sequence.append(toc_block)
            continue
        if current_sequence:
            sequences.append(current_sequence)
        current_sequence = [toc_block]
    if current_sequence:
        sequences.append(current_sequence)

    for sequence_index, sequence in enumerate(sequences, start=1):
        sequence_id = f"tocseq_{sequence_index:03d}"
        external_outline_parent_map: dict[str, dict[str, Any]] = {}
        sequence_entry_index = 0

        for position, toc_block in enumerate(sequence, start=1):
            toc_block["toc_sequence_id"] = sequence_id
            toc_block["toc_sequence_length"] = len(sequence)
            toc_block["toc_sequence_page_index"] = position
            toc_block["continued_from_toc_id"] = sequence[position - 2]["toc_id"] if position > 1 else None
            toc_block["continued_to_toc_id"] = sequence[position]["toc_id"] if position < len(sequence) else None
            toc_block["is_toc_continuation"] = position > 1

            annotated_entries: list[dict[str, Any]] = []
            for entry in toc_block.get("entries", []) or []:
                sequence_entry_index += 1
                annotated_entry = dict(entry)
                annotated_entry["page"] = int(toc_block.get("page", 0) or 0)
                annotated_entry["toc_id"] = toc_block.get("toc_id")
                annotated_entry["toc_sequence_id"] = sequence_id
                annotated_entry["sequence_entry_index"] = sequence_entry_index
                annotated_entries.append(annotated_entry)

            reannotated_entries, toc_diagnostics = _annotate_toc_diagnostics(
                annotated_entries,
                external_outline_parent_map=external_outline_parent_map,
            )
            local_sequence_index_by_entry_index = {
                int(entry.get("entry_index", 0) or 0): entry.get("sequence_entry_index")
                for entry in reannotated_entries
                if int(entry.get("entry_index", 0) or 0) > 0
            }
            for entry in reannotated_entries:
                parent_sequence_entry_index = None
                parent_toc_id = None
                if entry.get("parent_entry_index"):
                    parent_sequence_entry_index = local_sequence_index_by_entry_index.get(
                        int(entry.get("parent_entry_index", 0) or 0)
                    )
                    if parent_sequence_entry_index is not None:
                        parent_toc_id = toc_block.get("toc_id")
                if parent_sequence_entry_index is None and entry.get("sequence_parent_entry_index") is not None:
                    parent_sequence_entry_index = entry.get("sequence_parent_entry_index")
                    parent_toc_id = entry.get("sequence_parent_toc_id")
                entry["parent_sequence_entry_index"] = parent_sequence_entry_index
                entry["parent_toc_id"] = parent_toc_id

                section_anchor_sequence_entry_index = None
                section_anchor_toc_id = entry.get("section_anchor_toc_id")
                if section_anchor_toc_id == toc_block.get("toc_id"):
                    section_anchor_sequence_entry_index = local_sequence_index_by_entry_index.get(
                        int(entry.get("section_anchor_entry_index", 0) or 0)
                    )
                else:
                    section_anchor_sequence_entry_index = entry.get("sequence_parent_entry_index")
                entry["section_anchor_sequence_entry_index"] = section_anchor_sequence_entry_index

            toc_block["entries"] = reannotated_entries
            toc_block["toc_diagnostics"] = toc_diagnostics

            first_sequence_entry = reannotated_entries[0]["sequence_entry_index"] if reannotated_entries else None
            last_sequence_entry = reannotated_entries[-1]["sequence_entry_index"] if reannotated_entries else None
            toc_block["sequence_entry_range"] = [first_sequence_entry, last_sequence_entry]

            for entry in reannotated_entries:
                outline_index = str(entry.get("outline_index") or "").strip()
                if outline_index:
                    external_outline_parent_map[outline_index] = dict(entry)

    return len(sequences)


def _can_link_toc_sequence(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_page = int(previous_block.get("page", 0) or 0)
    current_page = int(current_block.get("page", 0) or 0)
    if current_page != previous_page + 1:
        return False
    if _toc_titles_conflict(previous_block, current_block):
        return False
    if _has_cross_page_outline_parent_link(previous_block, current_block):
        return True
    if _has_forward_root_outline_progression(previous_block, current_block):
        return True
    if _can_link_sparse_locator_only_toc(previous_block, current_block):
        return True
    return False


def _toc_titles_conflict(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_title = _compact_text(str(previous_block.get("title") or previous_block.get("toc_title") or ""))
    current_title = _compact_text(str(current_block.get("title") or current_block.get("toc_title") or ""))
    if not previous_title or not current_title:
        return False
    return previous_title != current_title


def _has_cross_page_outline_parent_link(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_outline_indices = {
        str(entry.get("outline_index") or "").strip()
        for entry in previous_block.get("entries", []) or []
        if str(entry.get("outline_index") or "").strip()
    }
    for entry in current_block.get("entries", []) or []:
        outline_parent_index = str(entry.get("outline_parent_index") or "").strip()
        if (
            outline_parent_index
            and not entry.get("parent_entry_index")
            and outline_parent_index in previous_outline_indices
        ):
            return True
    return False


def _has_same_page_outline_parent_link(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_outline_indices = {
        str(entry.get("outline_index") or "").strip()
        for entry in previous_block.get("entries", []) or []
        if str(entry.get("outline_index") or "").strip()
    }
    for entry in current_block.get("entries", []) or []:
        outline_parent_index = str(entry.get("outline_parent_index") or "").strip()
        if outline_parent_index and outline_parent_index in previous_outline_indices:
            return True
    return False


def _has_forward_root_outline_progression(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_root = _last_root_outline_key(previous_block)
    current_root = _first_root_outline_key(current_block)
    if previous_root is None or current_root is None:
        return False
    if previous_root[0] != current_root[0]:
        return False
    return current_root[1] > previous_root[1]


def _has_forward_outline_progression(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_outline = _last_toc_entry_outline_key(list(previous_block.get("entries", []) or []))
    current_outline = _first_toc_entry_outline_key(list(current_block.get("entries", []) or []))
    if previous_outline is None or current_outline is None:
        return False
    if previous_outline[0] != current_outline[0]:
        return False
    return current_outline[1] > previous_outline[1]


def _can_link_sparse_locator_only_toc(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_entries = list(previous_block.get("entries", []) or [])
    current_entries = list(current_block.get("entries", []) or [])
    if not previous_entries or not current_entries:
        return False
    previous_outline_ratio = _toc_outline_coverage_ratio(previous_entries)
    current_outline_ratio = _toc_outline_coverage_ratio(current_entries)
    if max(previous_outline_ratio, current_outline_ratio) > 0.2:
        return False

    previous_last_locator = _last_known_page_locator(previous_entries)
    current_first_locator = _first_known_page_locator(current_entries)
    if previous_last_locator is None or current_first_locator is None:
        return False
    if previous_last_locator[0] != current_first_locator[0]:
        return False
    return current_first_locator[1] >= previous_last_locator[1]


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


def _build_toc_sequences(
    toc_blocks: list[dict[str, Any]],
    page_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    page_payload_by_page = {
        int(page_payload.get("page_number", 0) or 0): page_payload
        for page_payload in page_payloads
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for toc_block in toc_blocks:
        sequence_id = str(toc_block.get("toc_sequence_id") or toc_block.get("toc_id") or "").strip()
        if not sequence_id:
            continue
        groups.setdefault(sequence_id, []).append(toc_block)

    sequences: list[dict[str, Any]] = []
    for sequence_id, blocks in groups.items():
        ordered_blocks = sorted(
            blocks,
            key=lambda toc_block: (
                int(toc_block.get("page", 0) or 0),
                float((toc_block.get("bbox", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[1]),
                float((toc_block.get("bbox", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[0]),
            ),
        )
        sequence_entries: list[dict[str, Any]] = []
        issue_counts: dict[str, int] = {}
        aggregated_issue_counts: dict[str, int] = {}
        review_items: list[dict[str, Any]] = []
        aggregated_review_items: list[dict[str, Any]] = []
        page_locator_kinds = {"arabic": 0, "roman": 0, "unknown": 0}
        missing_page_locator_count = 0
        max_entry_level = 0
        max_outline_depth = 0
        sequence_bbox = [float("inf"), float("inf"), float("-inf"), float("-inf")]

        for toc_block in ordered_blocks:
            bbox = list(toc_block.get("bbox", []) or [])
            if len(bbox) == 4:
                sequence_bbox[0] = min(sequence_bbox[0], float(bbox[0]))
                sequence_bbox[1] = min(sequence_bbox[1], float(bbox[1]))
                sequence_bbox[2] = max(sequence_bbox[2], float(bbox[2]))
                sequence_bbox[3] = max(sequence_bbox[3], float(bbox[3]))

            diagnostics = toc_block.get("toc_diagnostics") or {}
            for key, value in (diagnostics.get("issue_counts") or {}).items():
                issue_counts[key] = issue_counts.get(key, 0) + int(value or 0)
            for key, value in (diagnostics.get("aggregated_issue_counts") or {}).items():
                aggregated_issue_counts[key] = aggregated_issue_counts.get(key, 0) + int(value or 0)

            for item in diagnostics.get("review_items", []) or []:
                review_item = dict(item)
                review_item["page"] = toc_block.get("page")
                review_item["toc_id"] = toc_block.get("toc_id")
                review_items.append(review_item)
            for item in diagnostics.get("aggregated_review_items", []) or []:
                aggregated_item = dict(item)
                aggregated_item["page"] = toc_block.get("page")
                aggregated_item["toc_id"] = toc_block.get("toc_id")
                aggregated_review_items.append(aggregated_item)

            kinds = toc_block.get("page_locator_kinds") or {}
            for key in page_locator_kinds:
                page_locator_kinds[key] += int(kinds.get(key, 0) or 0)
            missing_page_locator_count += int(toc_block.get("missing_page_locator_count", 0) or 0)
            max_entry_level = max(max_entry_level, int(toc_block.get("max_entry_level", 0) or 0))
            max_outline_depth = max(max_outline_depth, int(toc_block.get("max_outline_depth", 0) or 0))

            for entry in toc_block.get("entries", []) or []:
                sequence_entry = dict(entry)
                sequence_entry["toc_id"] = toc_block.get("toc_id")
                sequence_entry["page"] = toc_block.get("page")
                sequence_entries.append(sequence_entry)

        sequence_title = _derive_toc_sequence_title(ordered_blocks, page_payload_by_page)
        if sequence_bbox[0] == float("inf"):
            bbox_value = []
        else:
            bbox_value = sequence_bbox
        root_entry_indices = [
            int(entry.get("sequence_entry_index", 0) or 0)
            for entry in sequence_entries
            if entry.get("parent_sequence_entry_index") is None
        ]
        pages = [int(toc_block.get("page", 0) or 0) for toc_block in ordered_blocks]
        sequence_diagnostics = {
            "review_required": any(bool((toc_block.get("toc_diagnostics") or {}).get("review_required", False)) for toc_block in ordered_blocks),
            "review_item_count": len(review_items),
            "aggregated_review_item_count": len(aggregated_review_items),
            "issue_counts": issue_counts,
            "aggregated_issue_counts": aggregated_issue_counts,
            "mixed_page_numbering": any(bool((toc_block.get("toc_diagnostics") or {}).get("mixed_page_numbering", False)) for toc_block in ordered_blocks),
            "review_items": review_items,
            "aggregated_review_items": aggregated_review_items,
        }
        root_nodes, leaf_entry_indices, max_branching_factor = _build_toc_sequence_tree(sequence_entries)
        navigation_summary = _build_toc_sequence_navigation_summary(sequence_entries, root_nodes)
        sequences.append(
            {
                "toc_sequence_id": sequence_id,
                "semantic_role": "toc_outline_sequence",
                "title": sequence_title,
                "toc_ids": [toc_block.get("toc_id") for toc_block in ordered_blocks],
                "pages": pages,
                "page_count": len(pages),
                "page_span": [min(pages), max(pages)] if pages else [None, None],
                "bbox": bbox_value,
                "entry_count": len(sequence_entries),
                "entries": sequence_entries,
                "root_entry_indices": root_entry_indices,
                "root_entry_count": len(root_entry_indices),
                "leaf_entry_indices": leaf_entry_indices,
                "leaf_entry_count": len(leaf_entry_indices),
                "max_branching_factor": max_branching_factor,
                "root_nodes": root_nodes,
                "navigation_summary": navigation_summary,
                "max_entry_level": max_entry_level,
                "max_outline_depth": max_outline_depth,
                "missing_page_locator_count": missing_page_locator_count,
                "page_locator_kinds": page_locator_kinds,
                "review_required": sequence_diagnostics["review_required"],
                "toc_sequence_diagnostics": sequence_diagnostics,
            }
        )

    sequences.sort(key=lambda sequence: (sequence["page_span"][0] or 0, sequence["toc_sequence_id"]))
    return sequences


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


def _build_toc_sequence_navigation_summary(
    sequence_entries: list[dict[str, Any]],
    root_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_entries = sorted(
        [dict(entry) for entry in sequence_entries],
        key=lambda entry: int(entry.get("sequence_entry_index", 0) or 0),
    )
    entry_by_sequence_index: dict[int, dict[str, Any]] = {}
    page_entries: dict[int, list[dict[str, Any]]] = {}
    outline_index_lookup: dict[str, list[int]] = {}

    for entry in ordered_entries:
        sequence_entry_index = int(entry.get("sequence_entry_index", 0) or 0)
        if sequence_entry_index <= 0:
            continue
        normalized_entry = dict(entry)
        normalized_entry["sequence_entry_index"] = sequence_entry_index
        normalized_entry["page"] = int(entry.get("page", 0) or 0)
        entry_by_sequence_index[sequence_entry_index] = normalized_entry
        page_entries.setdefault(normalized_entry["page"], []).append(normalized_entry)

        outline_index = str(entry.get("outline_index") or "").strip()
        if outline_index:
            outline_index_lookup.setdefault(outline_index, []).append(sequence_entry_index)

    cross_page_parent_link_count = 0
    for entry in entry_by_sequence_index.values():
        parent_page = _resolve_toc_entry_parent_page(entry, entry_by_sequence_index)
        if parent_page is None:
            continue
        if parent_page != int(entry.get("page", 0) or 0):
            cross_page_parent_link_count += 1

    page_entry_spans: list[dict[str, Any]] = []
    for page in sorted(page_entries):
        entries_on_page = sorted(
            page_entries[page],
            key=lambda entry: int(entry.get("sequence_entry_index", 0) or 0),
        )
        first_sequence_entry_index = int(entries_on_page[0].get("sequence_entry_index", 0) or 0)
        last_sequence_entry_index = int(entries_on_page[-1].get("sequence_entry_index", 0) or 0)
        toc_ids: list[str] = []
        root_entry_indices: list[int] = []
        cross_page_parent_entry_count = 0

        for entry in entries_on_page:
            toc_id = str(entry.get("toc_id") or "").strip()
            if toc_id and toc_id not in toc_ids:
                toc_ids.append(toc_id)

            sequence_entry_index = int(entry.get("sequence_entry_index", 0) or 0)
            parent_sequence_entry_index = int(entry.get("parent_sequence_entry_index", 0) or 0)
            if parent_sequence_entry_index <= 0:
                root_entry_indices.append(sequence_entry_index)
                continue

            parent_page = _resolve_toc_entry_parent_page(entry, entry_by_sequence_index)
            if parent_page is not None and parent_page != page:
                cross_page_parent_entry_count += 1

        page_entry_spans.append(
            {
                "page": page,
                "toc_ids": toc_ids,
                "entry_count": len(entries_on_page),
                "first_sequence_entry_index": first_sequence_entry_index,
                "last_sequence_entry_index": last_sequence_entry_index,
                "root_entry_indices": root_entry_indices,
                "cross_page_parent_entry_count": cross_page_parent_entry_count,
            }
        )

    root_sections: list[dict[str, Any]] = []
    cross_page_root_section_count = 0
    for root_node in root_nodes:
        root_section = _summarize_toc_root_section(root_node)
        if root_section["has_cross_page_coverage"]:
            cross_page_root_section_count += 1
        root_sections.append(root_section)

    outline_path_lookup: dict[str, list[int]] = {}
    for sequence_entry_index in sorted(entry_by_sequence_index):
        entry = entry_by_sequence_index[sequence_entry_index]
        outline_path = _build_toc_entry_outline_path(entry, entry_by_sequence_index)
        if not outline_path:
            continue
        outline_path_lookup.setdefault(outline_path, []).append(sequence_entry_index)

    return {
        "page_entry_spans": page_entry_spans,
        "root_sections": root_sections,
        "outline_index_lookup": outline_index_lookup,
        "outline_path_lookup": outline_path_lookup,
        "cross_page_parent_link_count": cross_page_parent_link_count,
        "cross_page_root_section_count": cross_page_root_section_count,
    }


def _summarize_toc_root_section(root_node: dict[str, Any]) -> dict[str, Any]:
    subtree_nodes = sorted(
        _collect_toc_subtree_nodes(root_node),
        key=lambda node: int(node.get("sequence_entry_index", 0) or 0),
    )
    sequence_entry_indices = [
        int(node.get("sequence_entry_index", 0) or 0)
        for node in subtree_nodes
        if int(node.get("sequence_entry_index", 0) or 0) > 0
    ]
    pages = sorted(
        {
            int(node.get("page", 0) or 0)
            for node in subtree_nodes
            if int(node.get("page", 0) or 0) > 0
        }
    )
    first_node = subtree_nodes[0] if subtree_nodes else root_node
    last_node = subtree_nodes[-1] if subtree_nodes else root_node
    direct_children = [
        {
            "sequence_entry_index": int(child.get("sequence_entry_index", 0) or 0),
            "outline_index": child.get("outline_index"),
            "text": child.get("text"),
            "page": int(child.get("page", 0) or 0),
        }
        for child in root_node.get("children", []) or []
    ]
    leaf_count = sum(1 for node in subtree_nodes if not (node.get("children") or []))

    if sequence_entry_indices:
        entry_span = [min(sequence_entry_indices), max(sequence_entry_indices)]
    else:
        entry_span = [None, None]
    if pages:
        page_span = [min(pages), max(pages)]
    else:
        page_span = [None, None]

    return {
        "sequence_entry_index": int(root_node.get("sequence_entry_index", 0) or 0),
        "outline_index": root_node.get("outline_index"),
        "text": root_node.get("text"),
        "page": int(root_node.get("page", 0) or 0),
        "toc_id": root_node.get("toc_id"),
        "entry_span": entry_span,
        "page_span": page_span,
        "pages": pages,
        "page_count": len(pages),
        "entry_count": len(sequence_entry_indices),
        "descendant_count": max(len(sequence_entry_indices) - 1, 0),
        "leaf_count": leaf_count,
        "child_count": int(root_node.get("child_count", 0) or 0),
        "direct_children": direct_children,
        "has_cross_page_coverage": len(pages) > 1,
        "first_page_locator": first_node.get("page_locator"),
        "first_page_locator_kind": first_node.get("page_locator_kind"),
        "first_page_locator_value": first_node.get("page_locator_value"),
        "last_page_locator": last_node.get("page_locator"),
        "last_page_locator_kind": last_node.get("page_locator_kind"),
        "last_page_locator_value": last_node.get("page_locator_value"),
    }


def _collect_toc_subtree_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    subtree_nodes = [node]
    for child in node.get("children", []) or []:
        subtree_nodes.extend(_collect_toc_subtree_nodes(child))
    return subtree_nodes


def _build_toc_entry_outline_path(
    entry: dict[str, Any],
    entry_by_sequence_index: dict[int, dict[str, Any]],
) -> str | None:
    path_segments: list[str] = []
    current_entry: dict[str, Any] | None = entry
    visited: set[int] = set()

    while current_entry:
        sequence_entry_index = int(current_entry.get("sequence_entry_index", 0) or 0)
        if sequence_entry_index <= 0 or sequence_entry_index in visited:
            break
        visited.add(sequence_entry_index)

        outline_index = str(current_entry.get("outline_index") or "").strip()
        if outline_index:
            path_segments.append(outline_index)

        parent_sequence_entry_index = int(current_entry.get("parent_sequence_entry_index", 0) or 0)
        if parent_sequence_entry_index <= 0:
            break
        current_entry = entry_by_sequence_index.get(parent_sequence_entry_index)

    if not path_segments:
        return None
    path_segments.reverse()
    return " > ".join(path_segments)


def _resolve_toc_entry_parent_page(
    entry: dict[str, Any],
    entry_by_sequence_index: dict[int, dict[str, Any]],
) -> int | None:
    parent_sequence_entry_index = int(entry.get("parent_sequence_entry_index", 0) or 0)
    if parent_sequence_entry_index > 0:
        parent_entry = entry_by_sequence_index.get(parent_sequence_entry_index)
        if parent_entry:
            return int(parent_entry.get("page", 0) or 0)

    sequence_parent_page = int(entry.get("sequence_parent_page", 0) or 0)
    if sequence_parent_page > 0:
        return sequence_parent_page
    return None


def _derive_toc_sequence_title(
    ordered_blocks: list[dict[str, Any]],
    page_payload_by_page: dict[int, dict[str, Any]],
) -> str | None:
    explicit_titles = [
        str(toc_block.get("title") or toc_block.get("toc_title") or "").strip()
        for toc_block in ordered_blocks
        if str(toc_block.get("title") or toc_block.get("toc_title") or "").strip()
    ]
    if explicit_titles:
        return explicit_titles[0]

    if not ordered_blocks:
        return None
    first_page = int(ordered_blocks[0].get("page", 0) or 0)
    first_page_payload = page_payload_by_page.get(first_page) or {}
    for block in sorted(first_page_payload.get("text_blocks", []) or [], key=lambda item: (item["bbox"][1], item["bbox"][0])):
        text = _compact_text(str(block.get("text", "") or ""))
        if text in {"目录", "目 录", "tableofcontents", "contents", "toc"}:
            return str(block.get("text", "")).strip() or None
    return None


def _join_content_segments(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def _count_content_evidence_types(content_evidence: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in content_evidence:
        source_type = str(item.get("source_type", "unknown") or "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def _build_content_units(content_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []

    for evidence in content_evidence:
        evidence_id = str(evidence.get("evidence_id", "")).strip()
        source_type = str(evidence.get("source_type", "unknown") or "unknown")
        source_id = str(evidence.get("source_id", "")).strip()
        page = int(evidence.get("page", 0) or 0)
        bbox = list(evidence.get("bbox", []))
        semantic_role = str(evidence.get("semantic_role", "") or "")
        segments = list(evidence.get("segments", []) or [])

        for unit_index, segment in enumerate(segments, start=1):
            unit_role = str(segment.get("role", "content") or "content")
            text = _clean_text(str(segment.get("text", "")))
            if not text:
                continue
            attributes = {
                key: value
                for key, value in segment.items()
                if key not in {"role", "text"}
            }
            units.append(
                {
                    "unit_id": f"cu_{evidence_id}_{unit_index:03d}",
                    "evidence_id": evidence_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "page": page,
                    "bbox": bbox,
                    "semantic_role": semantic_role,
                    "unit_role": unit_role,
                    "unit_index": unit_index,
                    "text": text,
                    "attributes": attributes,
                    "fact_extraction_eligible": _is_fact_extraction_unit(source_type, unit_role),
                    **(
                        {"section_context": deepcopy(evidence.get("section_context", {}) or {})}
                        if dict(evidence.get("section_context", {}) or {})
                        else {}
                    ),
                }
            )

    units.sort(key=_content_unit_sort_key)
    return units


def _content_unit_sort_key(unit: dict[str, Any]) -> tuple[Any, ...]:
    bbox = list(unit.get("bbox", []))
    y0 = float(bbox[1]) if len(bbox) >= 2 else float("inf")
    x0 = float(bbox[0]) if len(bbox) >= 1 else float("inf")
    source_type = str(unit.get("source_type", "unknown") or "unknown")
    source_priority = {
        "text": 0,
        "table": 1,
        "image": 2,
        "toc": 3,
    }.get(source_type, 9)
    return (
        int(unit.get("page", 0) or 0),
        y0,
        x0,
        source_priority,
        int(unit.get("unit_index", 0) or 0),
        str(unit.get("unit_id", "")),
    )


def _is_fact_extraction_unit(source_type: str, unit_role: str) -> bool:
    if source_type == "toc":
        return False
    if source_type == "text":
        return unit_role == "body"
    if source_type == "table":
        return unit_role in {"title", "header", "row"}
    if source_type == "image":
        return unit_role in {"caption", "embedded_text", "nearby_context"}
    return False


def _count_content_unit_types(content_units: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in content_units:
        source_type = str(item.get("source_type", "unknown") or "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def _build_fact_extraction_corpus(content_units: list[dict[str, Any]]) -> tuple[str, int]:
    parts: list[str] = []
    seen_norms: set[str] = set()
    used_unit_count = 0

    for unit in content_units:
        if not unit.get("fact_extraction_eligible", False):
            continue
        text = _clean_text(str(unit.get("text", "")))
        normalized = _compact_text(text)
        if not text or not normalized or normalized in seen_norms:
            continue
        seen_norms.add(normalized)
        parts.append(text)
        used_unit_count += 1

    return "\n".join(parts).strip(), used_unit_count

