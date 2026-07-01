from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any


class RegionType(str, Enum):
    BODY_TEXT = "body_text"
    SECTION_HEADING = "section_heading"
    SUBSECTION_HEADING = "subsection_heading"
    LIST_ITEM = "list_item"
    TOC = "toc"
    TOC_CONTINUATION = "toc_continuation"
    OUTLINE_TEMPLATE = "outline_template"
    TABLE = "table"
    TABLE_TITLE = "table_title"
    TABLE_NOTE = "table_note"
    TABLE_FOOTNOTE = "table_footnote"
    FIGURE = "figure"
    FIGURE_TITLE = "figure_title"
    FIGURE_LEGEND = "figure_legend"
    FORMULA_DISPLAY = "formula_display"
    FORMULA_INLINE = "formula_inline"
    ALGORITHM_PSEUDOCODE = "algorithm_pseudocode"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    PAGE_NUMBER = "page_number"
    FOOTNOTE = "footnote"
    LOGO = "logo"
    WATERMARK = "watermark"
    STAMP_OR_SEAL = "stamp_or_seal"
    SIDEBAR = "sidebar"
    UNKNOWN_VISUAL = "unknown_visual"
    NOISE_OR_ARTIFACT = "noise_or_artifact"


BBox = tuple[float, float, float, float]


def _bbox_to_list(bbox: BBox) -> list[float]:
    return [float(value) for value in bbox]


@dataclass(slots=True)
class RegionCandidate:
    candidate_id: str
    page: int
    region_type: RegionType
    bbox: BBox
    source: str
    evidence_refs: list[str] = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0
    signals: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "page": self.page,
            "region_type": self.region_type.value,
            "bbox": _bbox_to_list(self.bbox),
            "source": self.source,
            "evidence_refs": list(self.evidence_refs),
            "text": self.text,
            "confidence": round(float(self.confidence), 3),
            "signals": dict(self.signals),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class OwnershipDecision:
    region_id: str
    accepted_candidate_id: str
    region_type: RegionType
    page: int
    bbox: BBox
    owned_evidence_refs: list[str] = field(default_factory=list)
    owned_text_block_ids: list[str] = field(default_factory=list)
    competing_candidate_ids: list[str] = field(default_factory=list)
    decision_factors: list[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "accepted_candidate_id": self.accepted_candidate_id,
            "region_type": self.region_type.value,
            "page": self.page,
            "bbox": _bbox_to_list(self.bbox),
            "owned_evidence_refs": list(self.owned_evidence_refs),
            "owned_text_block_ids": list(self.owned_text_block_ids),
            "competing_candidate_ids": list(self.competing_candidate_ids),
            "decision_factors": list(self.decision_factors),
            "confidence": round(float(self.confidence), 3),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RegionNode:
    region_id: str
    region_type: RegionType
    page: int
    bbox: BBox
    text: str = ""
    children: list[str] = field(default_factory=list)
    links: dict[str, list[str]] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)
    structure_ref: str | None = None
    semantic_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "page": self.page,
            "bbox": _bbox_to_list(self.bbox),
            "text": self.text,
            "children": list(self.children),
            "links": {key: list(value) for key, value in self.links.items()},
            "provenance": list(self.provenance),
            "structure_ref": self.structure_ref,
            "semantic_ref": self.semantic_ref,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RegionOwnershipResult:
    candidates: list[RegionCandidate] = field(default_factory=list)
    decisions: list[OwnershipDecision] = field(default_factory=list)
    nodes: list[RegionNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "nodes": [node.to_dict() for node in self.nodes],
            "summary": {
                "candidate_count": len(self.candidates),
                "decision_count": len(self.decisions),
                "node_count": len(self.nodes),
            },
        }


def arbitrate_region_ownership(candidates: list[RegionCandidate]) -> RegionOwnershipResult:
    accepted: list[RegionCandidate] = []
    decisions: list[OwnershipDecision] = []
    nodes: list[RegionNode] = []
    ordered = sorted(candidates, key=_candidate_sort_key)
    for candidate in ordered:
        accepted_competitors = [
            previous
            for previous in accepted
            if previous.page == candidate.page and _bbox_overlap_ratio(previous.bbox, candidate.bbox) >= 0.35
        ]
        audit_competitors = [
            other
            for other in candidates
            if other.candidate_id != candidate.candidate_id
            and other.page == candidate.page
            and _bbox_overlap_ratio(other.bbox, candidate.bbox) >= 0.35
        ]
        stronger = [previous for previous in accepted_competitors if _candidate_priority(previous) >= _candidate_priority(candidate)]
        if stronger and candidate.region_type == RegionType.BODY_TEXT:
            continue
        accepted.append(candidate)
        region_id = f"region-{len(accepted):04d}-{candidate.region_type.value}"
        competing_ids = [item.candidate_id for item in audit_competitors]
        factors = _decision_factors(candidate, audit_competitors)
        owned_text_ids = _owned_text_block_ids(candidate.evidence_refs)
        decision = OwnershipDecision(
            region_id=region_id,
            accepted_candidate_id=candidate.candidate_id,
            region_type=candidate.region_type,
            page=candidate.page,
            bbox=candidate.bbox,
            owned_evidence_refs=list(candidate.evidence_refs),
            owned_text_block_ids=owned_text_ids,
            competing_candidate_ids=competing_ids,
            decision_factors=factors,
            confidence=candidate.confidence,
            warnings=[],
        )
        node = RegionNode(
            region_id=region_id,
            region_type=candidate.region_type,
            page=candidate.page,
            bbox=candidate.bbox,
            text=candidate.text,
            children=[],
            links={},
            provenance=list(candidate.evidence_refs),
            structure_ref=_structure_ref(candidate),
            semantic_ref=None,
            metadata=dict(candidate.metadata),
        )
        decisions.append(decision)
        nodes.append(node)
    return RegionOwnershipResult(candidates=list(candidates), decisions=decisions, nodes=nodes)


def collect_region_candidates(
    *,
    page_payloads: list[dict[str, Any]],
    table_asts: list[dict[str, Any]],
    toc_nodes: list[dict[str, Any]],
    figure_nodes: list[dict[str, Any]],
    external_candidates: list[RegionCandidate] | None = None,
) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    candidates.extend(list(external_candidates or []))
    candidates.extend(_collect_text_block_candidates(page_payloads))
    candidates.extend(_collect_table_candidates(table_asts))
    candidates.extend(_collect_toc_candidates(toc_nodes))
    figure_candidates = _collect_figure_candidates(figure_nodes, page_payloads)
    candidates.extend(figure_candidates)
    candidates.extend(_collect_text_layer_chart_figure_candidates(page_payloads, figure_candidates))
    return candidates


def _candidate_sort_key(candidate: RegionCandidate) -> tuple[int, float, str]:
    return (-_candidate_priority(candidate), -candidate.confidence, candidate.candidate_id)


def _candidate_priority(candidate: RegionCandidate) -> int:
    priorities = {
        RegionType.PAGE_HEADER: 95,
        RegionType.PAGE_FOOTER: 95,
        RegionType.PAGE_NUMBER: 95,
        RegionType.TABLE: 90,
        RegionType.FIGURE: 88,
        RegionType.TOC: 86,
        RegionType.OUTLINE_TEMPLATE: 84,
        RegionType.FORMULA_DISPLAY: 82,
        RegionType.ALGORITHM_PSEUDOCODE: 82,
        RegionType.TABLE_TITLE: 78,
        RegionType.TABLE_NOTE: 76,
        RegionType.TABLE_FOOTNOTE: 76,
        RegionType.FIGURE_TITLE: 74,
        RegionType.FIGURE_LEGEND: 72,
        RegionType.SECTION_HEADING: 68,
        RegionType.SUBSECTION_HEADING: 66,
        RegionType.LIST_ITEM: 58,
        RegionType.UNKNOWN_VISUAL: 52,
        RegionType.BODY_TEXT: 40,
    }
    return priorities.get(candidate.region_type, 50)


def _decision_factors(candidate: RegionCandidate, competitors: list[RegionCandidate]) -> list[str]:
    factors = [f"accepted {candidate.source} candidate"]
    if candidate.region_type == RegionType.TABLE and any(item.region_type == RegionType.BODY_TEXT for item in competitors):
        factors.append("table evidence outranks overlapping body candidate")
    if candidate.region_type == RegionType.FIGURE:
        factors.append("figure region preserves deferred content analysis")
    if candidate.region_type in {RegionType.PAGE_HEADER, RegionType.PAGE_FOOTER, RegionType.PAGE_NUMBER}:
        factors.append("margin artifact ownership kept outside body text")
    return factors


def _owned_text_block_ids(evidence_refs: list[str]) -> list[str]:
    result: list[str] = []
    for ref in evidence_refs:
        if ref.startswith("text:"):
            result.append(ref.split(":", 1)[1])
    return result


def _structure_ref(candidate: RegionCandidate) -> str | None:
    if candidate.region_type == RegionType.TABLE:
        table_id = candidate.metadata.get("table_id")
        return f"table:{table_id}" if table_id else None
    if candidate.region_type == RegionType.FIGURE:
        figure_id = candidate.metadata.get("figure_id")
        return f"figure:{figure_id}" if figure_id else None
    return None


def _bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    smaller_area = min(max((ax1 - ax0) * (ay1 - ay0), 1.0), max((bx1 - bx0) * (by1 - by0), 1.0))
    return intersection / smaller_area


def _collect_text_block_candidates(page_payloads: list[dict[str, Any]]) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    for payload in page_payloads:
        page = int(payload.get("page") or payload.get("page_number") or 0)
        for index, block in enumerate(payload.get("text_blocks") or []):
            bbox = _coerce_bbox(block.get("bbox"))
            if bbox is None:
                continue
            block_id = str(block.get("block_id") or block.get("id") or f"p{page}:b{index}")
            text = str(block.get("text") or "")
            role = str(block.get("role") or block.get("semantic_role") or "").lower()
            region_type = _region_type_from_text_role(role, text)
            candidates.append(
                RegionCandidate(
                    candidate_id=f"cand-{block_id}-{region_type.value}",
                    page=page,
                    region_type=region_type,
                    bbox=bbox,
                    source="text_blocks",
                    evidence_refs=[f"text:{block_id}"],
                    text=text,
                    confidence=0.55 if region_type == RegionType.BODY_TEXT else 0.72,
                    signals={"text_role": role},
                    metadata={"text_block_id": block_id},
                )
            )
    return candidates


def _region_type_from_text_role(role: str, text: str) -> RegionType:
    lowered = text.strip().lower()
    if "table_note" in role or lowered.startswith(("note:", "notes:", "source:")):
        return RegionType.TABLE_NOTE
    if "table_title" in role or lowered.startswith(("table ", "表 ")):
        return RegionType.TABLE_TITLE
    if "figure_title" in role or lowered.startswith(("fig.", "figure ", "图 ")):
        return RegionType.FIGURE_TITLE
    if "footer" in role:
        return RegionType.PAGE_FOOTER
    if "header" in role:
        return RegionType.PAGE_HEADER
    return RegionType.BODY_TEXT


def _collect_table_candidates(table_asts: list[dict[str, Any]]) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    for index, table in enumerate(table_asts):
        bbox = _coerce_bbox(table.get("bbox") or table.get("region_bbox"))
        if bbox is None:
            continue
        page = int(table.get("page") or table.get("page_number") or 0)
        table_id = str(table.get("table_id") or table.get("id") or f"table-{page}-{index}")
        candidates.append(
            RegionCandidate(
                candidate_id=f"cand-{table_id}-table",
                page=page,
                region_type=RegionType.TABLE,
                bbox=bbox,
                source="table_ast",
                evidence_refs=[f"table:{table_id}"],
                text=str(table.get("title") or table.get("caption") or ""),
                confidence=0.82,
                signals={"has_table_ast": True},
                metadata={"table_id": table_id, "raw_source": table.get("source")},
            )
        )
    return candidates


def _collect_toc_candidates(toc_nodes: list[dict[str, Any]]) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    for index, node in enumerate(toc_nodes):
        bbox = _coerce_bbox(node.get("bbox"))
        if bbox is None:
            continue
        page = int(node.get("page") or node.get("page_number") or 0)
        node_id = str(node.get("node_id") or node.get("id") or f"toc-{page}-{index}")
        candidates.append(
            RegionCandidate(
                candidate_id=f"cand-{node_id}-toc",
                page=page,
                region_type=RegionType.TOC,
                bbox=bbox,
                source="toc_detector",
                evidence_refs=[f"toc:{node_id}"],
                text=str(node.get("title") or node.get("text") or ""),
                confidence=0.8,
                signals={"has_toc_node": True},
                metadata={"toc_node_id": node_id},
            )
        )
    return candidates


def _collect_figure_candidates(
    figure_nodes: list[dict[str, Any]],
    page_payloads: list[dict[str, Any]],
) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    for index, figure in enumerate(figure_nodes):
        bbox = _coerce_bbox(figure.get("bbox"))
        if bbox is None:
            continue
        page = int(figure.get("page") or figure.get("page_number") or 0)
        figure_id = str(figure.get("figure_id") or figure.get("id") or f"figure-{page}-{index}")
        candidates.append(
            RegionCandidate(
                candidate_id=f"cand-{figure_id}-figure",
                page=page,
                region_type=RegionType.FIGURE,
                bbox=bbox,
                source="figure_node",
                evidence_refs=[f"figure:{figure_id}"],
                text=str(figure.get("caption") or figure.get("text") or ""),
                confidence=0.78,
                signals={"has_figure_node": True},
                metadata={
                    "figure_id": figure_id,
                    "content_analysis": (
                        {"status": "evidence_only"}
                        if figure.get("figure_semantics")
                        else {"status": "not_analyzed"}
                    ),
                    **(
                        {"figure_semantics": dict(figure.get("figure_semantics") or {})}
                        if figure.get("figure_semantics")
                        else {}
                    ),
                },
            )
        )
    for payload in page_payloads:
        page = int(payload.get("page") or payload.get("page_number") or 0)
        image_blocks = list(payload.get("images") or []) or list(payload.get("image_blocks") or [])
        for index, image in enumerate(image_blocks):
            bbox = _coerce_bbox(image.get("bbox"))
            if bbox is None:
                continue
            image_id = str(image.get("image_id") or image.get("id") or f"p{page}:i{index}")
            candidates.append(
                RegionCandidate(
                    candidate_id=f"cand-{image_id}-unknown-visual",
                    page=page,
                    region_type=RegionType.UNKNOWN_VISUAL,
                    bbox=bbox,
                    source="image_blocks",
                    evidence_refs=[f"image:{image_id}"],
                    text="",
                    confidence=0.5,
                    signals={"image_kind": image.get("kind")},
                    metadata={"image_id": image_id},
                )
            )
    return candidates


_EXPLICIT_TEXT_LAYER_FIGURE_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?|chart|diagram)\s*[\dIVXLCDM]+(?:[.\-:][\w\d]+)*\b",
    re.IGNORECASE,
)
_TABLE_CAPTION_RE = re.compile(r"^\s*table\s*[\dIVXLCDM]+(?:[.\-:][\w\d]+)*\b", re.IGNORECASE)
_SOURCE_ROW_RE = re.compile(r"^\s*(?:source|sources|note|notes)\s*[:\uff1a]", re.IGNORECASE)
_DATE_OR_TIME_TICK_RE = re.compile(
    r"(?:\b\d{1,2}/\d{4}\b|\b\d{4}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b)",
    re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:%|[xX])?")


def _collect_text_layer_chart_figure_candidates(
    page_payloads: list[dict[str, Any]],
    existing_figure_candidates: list[RegionCandidate],
) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    for payload in page_payloads:
        page = int(payload.get("page") or payload.get("page_number") or 0)
        text_blocks = [
            block
            for block in payload.get("text_blocks") or []
            if _coerce_bbox(block.get("bbox")) is not None and str(block.get("text") or "").strip()
        ]
        ordered = sorted(text_blocks, key=lambda block: ((_coerce_bbox(block.get("bbox")) or (0, 0, 0, 0))[1], (_coerce_bbox(block.get("bbox")) or (0, 0, 0, 0))[0]))
        caption_indexes = [
            index
            for index, block in enumerate(ordered)
            if _is_text_layer_chart_caption(str(block.get("text") or ""), str(block.get("role") or block.get("semantic_role") or ""))
        ]
        for local_index, caption_index in enumerate(caption_indexes, start=1):
            next_caption_index = next((item for item in caption_indexes if item > caption_index), len(ordered))
            cluster_blocks = _collect_chart_cluster_after_caption(
                ordered,
                caption_index=caption_index,
                stop_index=next_caption_index,
            )
            if not _has_chart_region_evidence(cluster_blocks):
                continue
            bboxes = [
                bbox
                for block in cluster_blocks
                if (bbox := _coerce_bbox(block.get("bbox"))) is not None
            ]
            if not bboxes:
                continue
            cluster_bbox = _bbox_union(bboxes)
            if _overlaps_existing_figure_candidate(cluster_bbox, page, existing_figure_candidates):
                continue
            text = _join_block_text(cluster_blocks)
            block_ids = [
                str(block.get("block_id") or block.get("id") or f"p{page}:chart:{idx}")
                for idx, block in enumerate(cluster_blocks)
            ]
            figure_id = f"text-layer-chart-p{page}-{local_index:03d}"
            semantics = _build_text_layer_chart_semantics(cluster_blocks)
            candidates.append(
                RegionCandidate(
                    candidate_id=f"cand-{figure_id}-figure",
                    page=page,
                    region_type=RegionType.FIGURE,
                    bbox=cluster_bbox,
                    source="text_layer_chart_figure",
                    evidence_refs=[f"text:{block_id}" for block_id in block_ids],
                    text=text,
                    confidence=_text_layer_chart_confidence(semantics),
                    signals={
                        "content_kind": "text_layer_chart",
                        "evidence_signals": list(semantics.get("evidence_signals", []) or []),
                    },
                    metadata={
                        "figure_id": figure_id,
                        "figure_semantics": semantics,
                        "content_analysis": {"status": "evidence_only"},
                    },
                )
            )
    return candidates


def _overlaps_existing_figure_candidate(
    bbox: BBox,
    page: int,
    existing_figure_candidates: list[RegionCandidate],
) -> bool:
    for candidate in existing_figure_candidates:
        if candidate.page != page:
            continue
        if candidate.region_type not in {RegionType.FIGURE, RegionType.UNKNOWN_VISUAL}:
            continue
        if _bbox_overlap_ratio(candidate.bbox, bbox) >= 0.18:
            return True
    return False


def _is_text_layer_chart_caption(text: str, role: str) -> bool:
    stripped = " ".join(str(text or "").strip().split())
    if not stripped:
        return False
    if "figure_title" in str(role or "").lower():
        return True
    if _TABLE_CAPTION_RE.match(stripped):
        return False
    return bool(_EXPLICIT_TEXT_LAYER_FIGURE_RE.match(stripped))


def _collect_chart_cluster_after_caption(
    ordered: list[dict[str, Any]],
    *,
    caption_index: int,
    stop_index: int,
) -> list[dict[str, Any]]:
    cluster = [ordered[caption_index]]
    last_bbox = _coerce_bbox(ordered[caption_index].get("bbox"))
    seen_chart_rows = 0
    seen_source = False
    for block in ordered[caption_index + 1 : stop_index]:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        if _TABLE_CAPTION_RE.match(text):
            break
        role = _chart_text_row_role(text)
        bbox = _coerce_bbox(block.get("bbox"))
        if bbox is None:
            continue
        vertical_gap = bbox[1] - (last_bbox[3] if last_bbox is not None else bbox[1])
        if role == "body_text" and seen_chart_rows >= 2:
            break
        if role == "body_text" and vertical_gap > 30.0:
            break
        if role != "body_text":
            seen_chart_rows += 1
        cluster.append(block)
        last_bbox = bbox
        if role == "source":
            seen_source = True
            break
    if seen_source:
        return cluster
    return cluster


def _chart_text_row_role(text: str) -> str:
    stripped = " ".join(str(text or "").strip().split())
    if not stripped:
        return "empty"
    if _SOURCE_ROW_RE.match(stripped):
        return "source"
    numeric_tokens = _NUMERIC_TOKEN_RE.findall(stripped)
    token_count = max(1, len(stripped.split()))
    numeric_ratio = len(numeric_tokens) / token_count
    if len(numeric_tokens) >= 3 and (_DATE_OR_TIME_TICK_RE.search(stripped) or numeric_ratio >= 0.45):
        return "axis_or_tick"
    if len(numeric_tokens) == 1 and len(stripped) <= 12:
        return "axis_or_tick"
    if _DATE_OR_TIME_TICK_RE.search(stripped) and len(stripped) <= 96:
        return "axis_or_tick"
    if len(numeric_tokens) >= 2 and len(stripped) <= 140:
        return "legend_or_series"
    if re.search(r"\b(?:axis|frequency|population|workforce|rate|percent|percentage|count|number)\b", stripped, re.IGNORECASE):
        return "legend_or_series"
    return "body_text"


def _has_chart_region_evidence(blocks: list[dict[str, Any]]) -> bool:
    if len(blocks) < 3:
        return False
    roles = [_chart_text_row_role(str(block.get("text") or "")) for block in blocks[1:]]
    axis_count = roles.count("axis_or_tick")
    source_count = roles.count("source")
    legend_count = roles.count("legend_or_series")
    return axis_count >= 3 or (axis_count >= 2 and (source_count or legend_count))


def _build_text_layer_chart_semantics(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    caption = str(blocks[0].get("text") or "").strip() if blocks else ""
    axis_or_tick_text: list[str] = []
    legend_or_series_text: list[str] = []
    source_text: list[str] = []
    for block in blocks[1:]:
        text = " ".join(str(block.get("text") or "").strip().split())
        if not text:
            continue
        role = _chart_text_row_role(text)
        if role == "source":
            source_text.append(text)
        elif role == "axis_or_tick":
            axis_or_tick_text.append(text)
        elif role == "legend_or_series":
            legend_or_series_text.append(text)
    signals = ["explicit_figure_caption", "text_layer_chart_cluster"]
    if axis_or_tick_text:
        signals.append("axis_or_tick_text")
    if legend_or_series_text:
        signals.append("legend_or_series_text")
    if source_text:
        signals.append("source_text")
    return {
        "semantic_type": "chart_figure",
        "content_kind": "text_layer_chart",
        "chart_type": "unknown_chart",
        "caption_text": caption,
        "axis_or_tick_text": axis_or_tick_text[:16],
        "legend_or_series_text": legend_or_series_text[:12],
        "source_text": source_text[:4],
        "content_analysis": {"status": "evidence_only"},
        "evidence_signals": signals,
    }


def _text_layer_chart_confidence(semantics: dict[str, Any]) -> float:
    confidence = 0.72
    if semantics.get("axis_or_tick_text"):
        confidence += 0.05
    if semantics.get("legend_or_series_text"):
        confidence += 0.03
    if semantics.get("source_text"):
        confidence += 0.05
    return min(confidence, 0.86)


def _join_block_text(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(
        " ".join(str(block.get("text") or "").strip().split())
        for block in blocks
        if str(block.get("text") or "").strip()
    )


def _bbox_union(bboxes: list[BBox]) -> BBox:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _coerce_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 < x0 or y1 < y0:
        return None
    return (x0, y0, x1, y1)
