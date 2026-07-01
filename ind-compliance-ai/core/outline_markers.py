from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class OutlineMarker:
    raw_marker: str
    normalized_marker: str
    marker_kind: str
    title: str


@dataclass(frozen=True)
class OutlineHeadingCandidate:
    marker: OutlineMarker
    role: str
    reason: str


_APPENDIX_RE = re.compile(r"^\s*(APPENDIX\s+[A-Z]+)(?:[.):])?\s*(.*)$", re.IGNORECASE)
_MODULE_RE = re.compile(r"^\s*((?:MODULE|M)\s*\d+)(?:[.):])?\s+(\S.*)$", re.IGNORECASE)
_OUTLINE_HEADING_RE = re.compile(
    r"^\s*(?P<marker>"
    r"\d+(?:\.(?:\d+|[A-Za-z]+))*"
    r"|[IVXLCDM]+"
    r"|[A-Z]"
    r")(?:[.):])?\s+(?P<title>\S.*)$",
    re.IGNORECASE,
)


def normalize_outline_marker(value: Any) -> str:
    candidate = re.sub(r"\s+", " ", str(value or "").strip())
    if not candidate:
        return ""

    appendix_match = re.fullmatch(r"APPENDIX\s+([A-Z]+)", candidate, re.IGNORECASE)
    if appendix_match:
        return f"APPENDIX {appendix_match.group(1).upper()}"

    module_match = re.fullmatch(r"(?:MODULE|M)\s*(\d+)", candidate, re.IGNORECASE)
    if module_match:
        return f"MODULE {int(module_match.group(1))}"

    segments = [segment for segment in candidate.split(".") if segment]
    normalized_segments: list[str] = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if segment.isdigit():
            normalized_segments.append(str(int(segment)))
        elif re.fullmatch(r"[A-Za-z]+", segment):
            normalized_segments.append(segment.upper())
        else:
            return candidate.upper() if any(char.isalpha() for char in candidate) else candidate
    while len(normalized_segments) > 1 and normalized_segments[-1] == "0":
        normalized_segments.pop()
    if normalized_segments:
        return ".".join(normalized_segments)
    return candidate.upper() if any(char.isalpha() for char in candidate) else candidate


def classify_outline_marker(value: Any) -> str:
    normalized = normalize_outline_marker(value)
    if not normalized:
        return ""
    if re.fullmatch(r"APPENDIX\s+[A-Z]+", normalized):
        return "appendix"
    if re.fullmatch(r"MODULE\s+\d+", normalized):
        return "module"
    if re.fullmatch(r"\d+(?:\.\d+)*", normalized):
        return "decimal_numeric"
    if re.fullmatch(r"\d+(?:\.(?:\d+|[A-Z]+))*", normalized) and re.search(r"[A-Z]", normalized):
        return "ctd_mixed"
    if re.fullmatch(r"[IVXLCDM]+", normalized):
        return "roman"
    if re.fullmatch(r"[A-Z]", normalized):
        return "alpha"
    return "mixed"


def parse_outline_heading(text: Any) -> OutlineMarker | None:
    candidate = str(text or "").strip()
    if not candidate:
        return None

    appendix_match = _APPENDIX_RE.match(candidate)
    if appendix_match:
        raw_marker = re.sub(r"\s+", " ", appendix_match.group(1).strip()).upper()
        title = str(appendix_match.group(2) or "").strip()
        normalized = normalize_outline_marker(raw_marker)
        if not title:
            return None
        return OutlineMarker(raw_marker, normalized, "appendix", title)

    module_match = _MODULE_RE.match(candidate)
    if module_match:
        raw_marker = re.sub(r"\s+", " ", module_match.group(1).strip())
        title = str(module_match.group(2) or "").strip()
        normalized = normalize_outline_marker(raw_marker)
        return OutlineMarker(raw_marker, normalized, "module", title)

    match = _OUTLINE_HEADING_RE.match(candidate)
    if not match:
        return None

    raw_marker = str(match.group("marker") or "").strip()
    title = str(match.group("title") or "").strip()
    if not raw_marker or not title:
        return None
    normalized = normalize_outline_marker(raw_marker)
    marker_kind = classify_outline_marker(normalized)
    return OutlineMarker(raw_marker, normalized, marker_kind, title)


def classify_outline_heading_candidate(
    text: Any,
    *,
    toc_heading_lookup: dict[str, list[dict[str, Any]]] | None = None,
    active_parent_marker: Any = None,
) -> OutlineHeadingCandidate | None:
    marker = parse_outline_heading(text)
    if marker is None:
        return None

    toc_heading_lookup = toc_heading_lookup or {}
    if _outline_marker_matches_toc(marker, toc_heading_lookup):
        return OutlineHeadingCandidate(marker, "section_heading", "toc_title_match")

    normalized_parent = normalize_outline_marker(active_parent_marker)
    if _is_probable_body_list_marker(marker, normalized_parent):
        return OutlineHeadingCandidate(marker, "body_list_item", "single_level_marker_inside_parent_section")

    if marker.marker_kind == "ctd_mixed":
        return OutlineHeadingCandidate(marker, "section_heading_candidate", "ctd_mixed_marker")
    if marker.marker_kind in {"appendix", "module"}:
        return OutlineHeadingCandidate(marker, "section_heading_candidate", marker.marker_kind)
    if marker.marker_kind == "decimal_numeric" and marker.normalized_marker.count(".") >= 1:
        return OutlineHeadingCandidate(marker, "section_heading_candidate", "decimal_depth")
    if marker.marker_kind in {"roman", "alpha"} and _title_looks_like_outline_heading(marker.title):
        return OutlineHeadingCandidate(marker, "section_heading_candidate", f"{marker.marker_kind}_heading_title")
    return OutlineHeadingCandidate(marker, "weak_candidate", "single_level_unconfirmed_marker")


def _outline_marker_matches_toc(
    marker: OutlineMarker,
    toc_heading_lookup: dict[str, list[dict[str, Any]]],
) -> bool:
    candidates = toc_heading_lookup.get(marker.normalized_marker, [])
    return any(outline_titles_compatible(marker.title, candidate.get("title")) for candidate in candidates)


def _is_probable_body_list_marker(marker: OutlineMarker, normalized_parent_marker: str) -> bool:
    if marker.marker_kind not in {"decimal_numeric", "roman", "alpha"}:
        return False
    if "." in marker.normalized_marker:
        return False
    if not normalized_parent_marker:
        return False
    if _is_single_level_marker_sibling_of_parent(marker.normalized_marker, normalized_parent_marker):
        return False
    return True


def _is_single_level_marker_sibling_of_parent(marker_value: str, normalized_parent_marker: str) -> bool:
    if not re.fullmatch(r"\d+", marker_value):
        return False
    parent_segments = [segment for segment in str(normalized_parent_marker or "").split(".") if segment]
    if len(parent_segments) != 1 or not parent_segments[0].isdigit():
        return False
    return abs(int(marker_value) - int(parent_segments[0])) <= 1


def _title_looks_like_outline_heading(title: Any) -> bool:
    text = re.sub(r"\s+", " ", str(title or "").strip())
    if not text:
        return False
    words = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text)
    if len(words) <= 8:
        return True
    return not bool(re.search(r"[.;:]\s*$|\b(should|must|shall|provide|include|contains?|describes?)\b", text, re.IGNORECASE))


def outline_titles_compatible(body_title: Any, toc_title: Any) -> bool:
    body = _normalize_title_for_match(body_title)
    toc = _normalize_title_for_match(toc_title)
    if not body or not toc:
        return False
    if body == toc:
        return True
    if body.startswith(toc) and len(body) <= len(toc) + 2:
        return True
    return toc.startswith(body) and len(toc) <= len(body) + 2


def _normalize_title_for_match(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[.。:：,，;；()（）\[\]【】<>]", "", text)
    return text.casefold()
