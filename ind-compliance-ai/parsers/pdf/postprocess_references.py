from __future__ import annotations

import re
from typing import Any

from .shared import _clean_text, _compact_text


_DISPLAY_EQUATION_HARD_PROSE_CUES = (
    "algorithm",
    "initialization",
    "output",
    "compute",
    "repeat",
    "update",
    "until",
    "end if",
    "proof",
    "theorem",
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
_LITERATURE_AUTHOR_TOKEN_RE = re.compile(
    r"\b[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){0,2}(?:\d|[A-Za-z](?:,[A-Za-z])*)(?:[\u2020\u2021*])?"
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
_PUBLICATION_AUTHOR_NOTE_RE = re.compile(
    r"^(?:(?:[*\u2020\u2021\u00a7\u00b6]|\d{1,3}|[A-Za-z])\s*)?"
    r"(?:corresponding author|correspondence to|present address|"
    r"these authors contributed equally|contributed equally|equal contribution|"
    r"full list of author information)\b",
    re.IGNORECASE,
)
_PUBLICATION_FOOTER_ISSN_RE = re.compile(r"^\d{4}-\d{3}[\dXx]\s*/")


def _looks_like_equation_false_positive_text(text: str) -> bool:
    lowered = _clean_text(text).lower()
    if not lowered:
        return False
    if any(cue in lowered for cue in _DISPLAY_EQUATION_HARD_PROSE_CUES):
        return True
    if re.match(r"^\d+\.\s", lowered):
        return True
    return lowered.startswith(("theorem", "proof", "algorithm", "initialization", "output", "set ", "compute ", "repeat", "update ", "until "))


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


def _looks_like_literature_author_line(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 320:
        return False
    if re.match(r"^(?:appendix|chapter|section)\b", compact, re.IGNORECASE):
        return False
    if compact[:1].islower() or compact.startswith(("(", "[", "{")):
        return False
    if _looks_like_body_prose_with_citation(compact):
        return False
    author_token_count = len(_LITERATURE_AUTHOR_TOKEN_RE.findall(compact))
    marker_count = (
        compact.count(",")
        + compact.count("*")
        + compact.count("\u2020")
        + compact.count("\u2021")
    )
    has_marker_evidence = bool(re.search(r"\b[A-Z][A-Za-z'`.-]+\s+[A-Za-z](?:,[A-Za-z])*\b", compact))
    return author_token_count >= 2 and compact.count(",") >= 2 and (marker_count >= 2 or has_marker_evidence)


def _looks_like_body_prose_with_citation(text: str) -> bool:
    compact = _clean_text(text)
    if not compact:
        return False
    lowered = compact.lower()
    if re.search(r"\bet\s+al\.\s*,\s*\d{4}", lowered):
        return True
    if re.search(r"\([^)]+et\s+al\.\s*,\s*\d{4}[^)]*\)", lowered):
        return True
    if re.search(r"\([A-Z][A-Za-z'`.-]+(?:\s+et\s+al\.)?,\s*\d{4}", compact):
        return True
    prose_markers = {
        "however",
        "such as",
        "these",
        "this",
        "that",
        "which",
        "method",
        "methods",
        "genes",
        "score",
        "ratio",
    }
    return any(marker in lowered for marker in prose_markers) and bool(re.search(r"\(\d{4}\)|,\s*\d{4}", compact))


def _looks_like_author_affiliation_line(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 360:
        return False
    lowered = compact.lower()
    if _looks_like_equation_false_positive_text(compact):
        return False
    if not any(keyword in lowered for keyword in _AFFILIATION_KEYWORDS):
        return False
    if re.match(r"^(?:\d+|[*]|[A-Za-z])\s+", compact):
        return True
    return compact.count(";") >= 1 or compact.count(",") >= 2


def _looks_like_contact_name_line(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 80 or any(ch.isdigit() for ch in compact):
        return False
    return bool(_PERSON_NAME_LINE_RE.fullmatch(compact))


def _looks_like_publication_author_note_text(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 220:
        return False
    if _looks_like_equation_false_positive_text(compact):
        return False
    return bool(_PUBLICATION_AUTHOR_NOTE_RE.search(compact))


def _looks_like_publication_footer_text(text: str, y0: float, page_height: float) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 260:
        return False
    if page_height <= 0 or y0 < page_height * 0.84:
        return False
    lowered = compact.lower()
    if _PUBLICATION_FOOTER_ISSN_RE.search(compact):
        return True
    if "all rights reserved" in lowered:
        return True
    if "copyright" in lowered:
        return True
    return compact.startswith(("\u00a9", "(c)", "漏"))


def _looks_like_license_notice_text(text: str, y0: float = 0.0, page_height: float = 0.0) -> bool:
    compact = _clean_text(text)
    if not compact:
        return False
    lowered = compact.lower()
    normalized = _compact_text(compact)
    if "creativecommons" in normalized or "licensedunder" in normalized:
        return True
    if compact.startswith("\u00a9") or re.match(r"^\s*(?:\(c\)|copyright\s*(?:\u00a9|\(c\))?\s*\d{4})\b", lowered):
        return True
    if "all rights reserved" in lowered and page_height > 0 and y0 >= page_height * 0.84:
        return True
    if "copyright" in lowered:
        if page_height > 0 and y0 >= page_height * 0.84:
            return True
        return bool(
            re.search(r"\bcopyright\s*(?:notice|statement|license|licence)\b", lowered)
            or re.search(r"\b(?:licensed|distributed)\s+under\b", lowered)
        )
    return compact.startswith("漏")


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
