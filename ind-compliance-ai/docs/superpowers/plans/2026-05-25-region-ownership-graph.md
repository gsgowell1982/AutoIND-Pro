# Region Ownership Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an IND-first, observe-only Region Ownership Graph layer so table, figure, TOC, header/footer, and body regions are auditable before any future table or presentation changes can affect unrelated content.

**Architecture:** This first phase is additive. It defines region candidate/decision/node contracts, collects candidates from existing page payloads and AST objects, runs conservative ownership arbitration in observe-only mode, and exposes metadata without changing public Markdown or table rendering. Later phases can switch suppression/projection to consume ownership decisions after targeted regression gates are in place.

**Tech Stack:** Python dataclasses, existing `parsers/pdf` pipeline dictionaries, stdlib `unittest`, current project venv at `D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe`.

---

## File Structure

- Create `parsers/pdf/region_ownership.py`
  - Owns `RegionType`, `RegionCandidate`, `OwnershipDecision`, `RegionNode`, serialization helpers, candidate collection, and observe-only arbitration.
  - Kept separate from `postprocess.py` and `tables.py` to avoid making large parser files more coupled.

- Modify `parsers/pdf/types.py`
  - Add `region_candidates`, `ownership_decisions`, and `region_nodes` fields to `PdfPipelineState`.
  - Add counters for candidate/decision/node counts.

- Modify `parsers/pdf/pipeline.py`
  - Call the observe-only region ownership pass after page payloads, table ASTs, TOC nodes, and figure nodes exist.
  - Store output in `PdfPipelineState` and exported diagnostics/AST metadata if an existing diagnostics object is available.

- Modify `parsers/pdf_parser.py` or the current PDF AST packaging site if pipeline output is assembled there.
  - Expose `region_ownership` metadata in debug/AST JSON only.
  - Do not change user-facing Markdown text in this phase.

- Create `tests/parser_tests/test_region_ownership_graph.py`
  - Unit tests for serialization, candidate collection, table/title/note ownership, figure/title/legend ownership, and header/footer/body conflicts using synthetic page payloads.

- Modify existing protected regression tests only if needed to assert that production Markdown remains unchanged.

---

### Task 1: Region Ownership Contract

**Files:**
- Create: `parsers/pdf/region_ownership.py`
- Test: `tests/parser_tests/test_region_ownership_graph.py`

- [ ] **Step 1: Write failing serialization tests**

Add this test file:

```python
import unittest

from parsers.pdf.region_ownership import (
    OwnershipDecision,
    RegionCandidate,
    RegionNode,
    RegionType,
)


class RegionOwnershipContractTests(unittest.TestCase):
    def test_region_candidate_serializes_with_audit_fields(self):
        candidate = RegionCandidate(
            candidate_id="cand-p1-table-1",
            page=1,
            region_type=RegionType.TABLE,
            bbox=(10.0, 20.0, 300.0, 180.0),
            source="table_raw_candidate",
            evidence_refs=["text:p1:b3", "line:p1:l2"],
            text="Table X\nDose group",
            confidence=0.92,
            signals={"has_ruling_lines": True, "column_count": 4},
            metadata={"raw_source": "caption_anchored_horizontal_rules"},
        )

        payload = candidate.to_dict()

        self.assertEqual(payload["candidate_id"], "cand-p1-table-1")
        self.assertEqual(payload["region_type"], "table")
        self.assertEqual(payload["bbox"], [10.0, 20.0, 300.0, 180.0])
        self.assertEqual(payload["source"], "table_raw_candidate")
        self.assertEqual(payload["evidence_refs"], ["text:p1:b3", "line:p1:l2"])
        self.assertEqual(payload["signals"]["column_count"], 4)

    def test_ownership_decision_serializes_competing_candidates(self):
        decision = OwnershipDecision(
            region_id="region-p1-table-1",
            accepted_candidate_id="cand-p1-table-1",
            region_type=RegionType.TABLE,
            page=1,
            bbox=(10.0, 20.0, 300.0, 180.0),
            owned_evidence_refs=["text:p1:b3"],
            owned_text_block_ids=["p1:b3"],
            competing_candidate_ids=["cand-p1-body-2"],
            decision_factors=["strong table ruling evidence", "caption adjacency"],
            confidence=0.91,
            warnings=[],
        )

        payload = decision.to_dict()

        self.assertEqual(payload["region_id"], "region-p1-table-1")
        self.assertEqual(payload["region_type"], "table")
        self.assertEqual(payload["competing_candidate_ids"], ["cand-p1-body-2"])
        self.assertIn("caption adjacency", payload["decision_factors"])

    def test_region_node_keeps_children_links_and_provenance(self):
        node = RegionNode(
            region_id="region-p1-figure-1",
            region_type=RegionType.FIGURE,
            page=1,
            bbox=(40.0, 120.0, 420.0, 360.0),
            text="Fig. X Blood pressure",
            children=["region-p1-figure-title-1", "region-p1-figure-legend-1"],
            links={"title": ["region-p1-figure-title-1"], "legend": ["region-p1-figure-legend-1"]},
            provenance=["image:p1:i1"],
            structure_ref=None,
            semantic_ref=None,
            metadata={"content_analysis": {"status": "not_analyzed"}},
        )

        payload = node.to_dict()

        self.assertEqual(payload["region_type"], "figure")
        self.assertEqual(payload["links"]["legend"], ["region-p1-figure-legend-1"])
        self.assertEqual(payload["metadata"]["content_analysis"]["status"], "not_analyzed")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph.RegionOwnershipContractTests
```

Expected: import failure because `parsers.pdf.region_ownership` does not exist.

- [ ] **Step 3: Implement minimal contract**

Create `parsers/pdf/region_ownership.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph.RegionOwnershipContractTests
```

Expected: `Ran 3 tests` and `OK`.

---

### Task 2: Synthetic Candidate Collection

**Files:**
- Modify: `parsers/pdf/region_ownership.py`
- Modify: `tests/parser_tests/test_region_ownership_graph.py`

- [ ] **Step 1: Add failing candidate collection tests**

Append this test class:

```python
from parsers.pdf.region_ownership import collect_region_candidates


class RegionCandidateCollectionTests(unittest.TestCase):
    def test_collects_text_table_figure_and_toc_candidates(self):
        page_payloads = [
            {
                "page": 1,
                "text_blocks": [
                    {"id": "p1:b1", "text": "1 Introduction", "bbox": [40, 50, 200, 70], "role": "body"},
                    {"id": "p1:b2", "text": "Table X Dose groups", "bbox": [40, 100, 220, 120], "role": "table_title"},
                    {"id": "p1:b3", "text": "Note: values are mean.", "bbox": [40, 240, 260, 260], "role": "table_note"},
                    {"id": "p1:b4", "text": "Fig. X Blood pressure", "bbox": [40, 300, 260, 320], "role": "figure_title"},
                ],
                "image_blocks": [
                    {"id": "p1:i1", "bbox": [40, 325, 300, 480], "kind": "image"},
                ],
            }
        ]
        table_asts = [
            {"table_id": "t1", "page": 1, "bbox": [40, 125, 300, 235], "title": "Table X Dose groups"}
        ]
        toc_nodes = [
            {"node_id": "toc1", "page": 1, "bbox": [30, 500, 400, 620], "title": "Contents"}
        ]
        figure_nodes = [
            {"figure_id": "f1", "page": 1, "bbox": [40, 325, 300, 480], "caption": "Fig. X Blood pressure"}
        ]

        candidates = collect_region_candidates(
            page_payloads=page_payloads,
            table_asts=table_asts,
            toc_nodes=toc_nodes,
            figure_nodes=figure_nodes,
        )
        by_type = {candidate.region_type for candidate in candidates}

        self.assertIn(RegionType.BODY_TEXT, by_type)
        self.assertIn(RegionType.TABLE, by_type)
        self.assertIn(RegionType.TABLE_TITLE, by_type)
        self.assertIn(RegionType.TABLE_NOTE, by_type)
        self.assertIn(RegionType.FIGURE, by_type)
        self.assertIn(RegionType.FIGURE_TITLE, by_type)
        self.assertIn(RegionType.TOC, by_type)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph.RegionCandidateCollectionTests
```

Expected: import failure for `collect_region_candidates`.

- [ ] **Step 3: Implement synthetic candidate collection**

Add these helpers to `parsers/pdf/region_ownership.py`:

```python
def collect_region_candidates(
    *,
    page_payloads: list[dict[str, Any]],
    table_asts: list[dict[str, Any]],
    toc_nodes: list[dict[str, Any]],
    figure_nodes: list[dict[str, Any]],
) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    candidates.extend(_collect_text_block_candidates(page_payloads))
    candidates.extend(_collect_table_candidates(table_asts))
    candidates.extend(_collect_toc_candidates(toc_nodes))
    candidates.extend(_collect_figure_candidates(figure_nodes, page_payloads))
    return candidates


def _collect_text_block_candidates(page_payloads: list[dict[str, Any]]) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    for payload in page_payloads:
        page = int(payload.get("page") or payload.get("page_number") or 0)
        for index, block in enumerate(payload.get("text_blocks") or []):
            bbox = _coerce_bbox(block.get("bbox"))
            if bbox is None:
                continue
            block_id = str(block.get("id") or f"p{page}:b{index}")
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
                    "content_analysis": {"status": "not_analyzed"},
                },
            )
        )
    for payload in page_payloads:
        page = int(payload.get("page") or payload.get("page_number") or 0)
        for index, image in enumerate(payload.get("image_blocks") or []):
            bbox = _coerce_bbox(image.get("bbox"))
            if bbox is None:
                continue
            image_id = str(image.get("id") or f"p{page}:i{index}")
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
```

- [ ] **Step 4: Run collection tests**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph
```

Expected: all contract and collection tests pass.

---

### Task 3: Observe-Only Ownership Arbitration

**Files:**
- Modify: `parsers/pdf/region_ownership.py`
- Modify: `tests/parser_tests/test_region_ownership_graph.py`

- [ ] **Step 1: Add failing arbitration tests**

Append this test class:

```python
from parsers.pdf.region_ownership import arbitrate_region_ownership


class RegionOwnershipArbitrationTests(unittest.TestCase):
    def test_table_candidate_owns_overlapping_body_candidate_with_competitor_audit(self):
        candidates = [
            RegionCandidate(
                candidate_id="cand-body",
                page=1,
                region_type=RegionType.BODY_TEXT,
                bbox=(40, 100, 300, 220),
                source="text_blocks",
                evidence_refs=["text:p1:b1"],
                text="Dose group Vehicle X",
                confidence=0.55,
            ),
            RegionCandidate(
                candidate_id="cand-table",
                page=1,
                region_type=RegionType.TABLE,
                bbox=(40, 95, 310, 225),
                source="table_ast",
                evidence_refs=["table:t1", "text:p1:b1"],
                text="Table X",
                confidence=0.86,
                signals={"has_table_ast": True},
            ),
        ]

        result = arbitrate_region_ownership(candidates)

        table_decision = next(item for item in result.decisions if item.region_type == RegionType.TABLE)
        self.assertEqual(table_decision.accepted_candidate_id, "cand-table")
        self.assertIn("cand-body", table_decision.competing_candidate_ids)
        self.assertIn("table evidence outranks overlapping body candidate", table_decision.decision_factors)

    def test_header_footer_candidates_do_not_become_body_nodes(self):
        candidates = [
            RegionCandidate(
                candidate_id="cand-footer",
                page=1,
                region_type=RegionType.PAGE_FOOTER,
                bbox=(40, 760, 300, 780),
                source="text_blocks",
                evidence_refs=["text:p1:f1"],
                text="Confidential 12",
                confidence=0.78,
            )
        ]

        result = arbitrate_region_ownership(candidates)

        self.assertEqual(result.nodes[0].region_type, RegionType.PAGE_FOOTER)
        self.assertEqual(result.decisions[0].owned_text_block_ids, ["p1:f1"])

    def test_figure_node_preserves_deferred_content_analysis(self):
        candidates = [
            RegionCandidate(
                candidate_id="cand-figure",
                page=1,
                region_type=RegionType.FIGURE,
                bbox=(40, 300, 300, 480),
                source="figure_node",
                evidence_refs=["figure:f1"],
                text="Fig. X Blood pressure",
                confidence=0.8,
                metadata={"content_analysis": {"status": "not_analyzed"}},
            )
        ]

        result = arbitrate_region_ownership(candidates)

        self.assertEqual(result.nodes[0].metadata["content_analysis"]["status"], "not_analyzed")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph.RegionOwnershipArbitrationTests
```

Expected: import failure for `arbitrate_region_ownership`.

- [ ] **Step 3: Implement observe-only arbitration**

Add to `parsers/pdf/region_ownership.py`:

```python
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
        competitors = [
            previous
            for previous in accepted
            if previous.page == candidate.page and _bbox_overlap_ratio(previous.bbox, candidate.bbox) >= 0.35
        ]
        stronger = [previous for previous in competitors if _candidate_priority(previous) >= _candidate_priority(candidate)]
        if stronger and candidate.region_type == RegionType.BODY_TEXT:
            continue
        accepted.append(candidate)
        region_id = f"region-{len(accepted):04d}-{candidate.region_type.value}"
        competing_ids = [item.candidate_id for item in competitors]
        factors = _decision_factors(candidate, competitors)
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
```

- [ ] **Step 4: Run graph tests**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph
```

Expected: all region ownership tests pass.

---

### Task 4: Pipeline State Integration in Observe-Only Mode

**Files:**
- Modify: `parsers/pdf/types.py`
- Modify: `parsers/pdf/pipeline.py`
- Test: `tests/parser_tests/test_region_ownership_graph.py`

- [ ] **Step 1: Add failing state integration test**

Append this test class:

```python
from parsers.pdf.types import PdfPipelineState


class RegionOwnershipStateTests(unittest.TestCase):
    def test_pipeline_state_can_store_region_ownership_payloads(self):
        state = PdfPipelineState()

        self.assertEqual(state.region_candidates, [])
        self.assertEqual(state.ownership_decisions, [])
        self.assertEqual(state.region_nodes, [])
        self.assertEqual(state.counters.region_candidate_count, 0)
        self.assertEqual(state.counters.region_node_count, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph.RegionOwnershipStateTests
```

Expected: `AttributeError` for missing state fields.

- [ ] **Step 3: Add state fields**

In `parsers/pdf/types.py`, extend `PdfPipelineCounters`:

```python
    region_candidate_count: int = 0
    region_ownership_decision_count: int = 0
    region_node_count: int = 0
```

Extend `PdfPipelineState`:

```python
    region_candidates: list[dict[str, Any]] = field(default_factory=list)
    ownership_decisions: list[dict[str, Any]] = field(default_factory=list)
    region_nodes: list[dict[str, Any]] = field(default_factory=list)
```

- [ ] **Step 4: Add observe-only pipeline helper**

In `parsers/pdf/pipeline.py`, import the new helpers:

```python
from parsers.pdf.region_ownership import arbitrate_region_ownership, collect_region_candidates
```

Add a local helper near the pipeline orchestration functions:

```python
def _attach_region_ownership_observations(state: PdfPipelineState) -> None:
    candidates = collect_region_candidates(
        page_payloads=state.page_payloads,
        table_asts=state.table_asts,
        toc_nodes=state.toc_nodes,
        figure_nodes=state.figure_nodes,
    )
    result = arbitrate_region_ownership(candidates)
    payload = result.to_dict()
    state.region_candidates = payload["candidates"]
    state.ownership_decisions = payload["decisions"]
    state.region_nodes = payload["nodes"]
    state.counters.region_candidate_count = payload["summary"]["candidate_count"]
    state.counters.region_ownership_decision_count = payload["summary"]["decision_count"]
    state.counters.region_node_count = payload["summary"]["node_count"]
```

Call `_attach_region_ownership_observations(state)` after `state.page_payloads`, `state.table_asts`, `state.toc_nodes`, and `state.figure_nodes` are populated, and before final AST/debug serialization. This call must not alter page text blocks, table ASTs, figure nodes, or Markdown assembly.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph
```

Expected: all tests pass.

---

### Task 5: AST/Debug Metadata Exposure Without Markdown Changes

**Files:**
- Modify: `parsers/pdf_parser.py` or the current AST packaging function that emits `PdfPipelineState` results
- Test: `tests/parser_tests/test_region_ownership_graph.py`

- [ ] **Step 1: Locate AST packaging site**

Run:

```powershell
rg -n "table_asts|toc_nodes|figure_nodes|PdfPipelineState|diagnostics" D:\AutoIND-Pro\ind-compliance-ai\parsers D:\AutoIND-Pro\ind-compliance-ai\api
```

Expected: identify the function that converts `PdfPipelineState` into the returned parser payload.

- [ ] **Step 2: Add a failing packaging test if a direct helper exists**

If the packaging function is directly importable, add a test that constructs a `PdfPipelineState` with one `region_nodes` item and asserts the returned AST/debug payload includes `region_ownership`.

Use this expected payload shape:

```python
{
    "region_ownership": {
        "candidates": [...],
        "decisions": [...],
        "nodes": [...],
    }
}
```

If no direct helper exists, skip this unit test and rely on the parser smoke test in Task 6.

- [ ] **Step 3: Expose region ownership metadata**

At the packaging site, add:

```python
region_ownership_payload = {
    "candidates": state.region_candidates,
    "decisions": state.ownership_decisions,
    "nodes": state.region_nodes,
}
```

Attach it under a debug/AST metadata key that does not change Markdown body output. Prefer an existing diagnostics or `ast` payload location. Do not insert region ownership lines into the user-facing Markdown.

- [ ] **Step 4: Run packaging or smoke test**

Run the direct packaging test if created:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph
```

Expected: pass.

---

### Task 6: Protected Non-Regression Gate

**Files:**
- No production code changes unless tests expose a wiring bug.

- [ ] **Step 1: Run region ownership unit tests**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph
```

Expected: all tests pass.

- [ ] **Step 2: Run deterministic Markdown export gate**

Run:

```powershell
@'
import unittest
suite = unittest.defaultTestLoader.loadTestsFromName('tests.deterministic_tests.test_parse_markdown_export')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
'@ | D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -
```

Expected: existing deterministic Markdown tests pass. If they fail because Markdown text changed, stop and remove the behavior change; this phase is observe-only.

- [ ] **Step 3: Run protected parser regression gate**

Run:

```powershell
$env:IND_A_TST_REGRESSION_PDF='D:\AutoIND-Pro\A-tst.pdf'
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_a_tst_regression tests.parser_tests.test_two_column_literature_regression tests.parser_tests.test_r2_regression tests.parser_tests.test_ectd_regression_sample
```

Expected: protected regressions pass. If any test fails from region ownership metadata only, adjust tests only if they assert complete debug payload equality; do not change Markdown expectations.

- [ ] **Step 4: Run OpenDataLoader diagnostic regression tests**

Run:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_opendataloader_benchmark_regression tests.parser_tests.test_table_semantic_projection_v2 tests.deterministic_tests.test_semantic_table_markdown_export
```

Expected: benchmark/table regression tests pass. Since the phase is observe-only, metric movement should be zero or explainable only by debug metadata if the adapter consumes debug metadata, which it should not.

---

### Task 7: Mirror Sync and Mirror Verification

**Files:**
- Sync only files changed by this plan to `D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai`.

- [ ] **Step 1: Copy changed files to mirror**

Copy the final changed files:

```powershell
$primary='D:\AutoIND-Pro\ind-compliance-ai'
$mirror='D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai'
$files=@(
  'parsers\pdf\region_ownership.py',
  'parsers\pdf\types.py',
  'parsers\pdf\pipeline.py',
  'parsers\pdf_parser.py',
  'tests\parser_tests\test_region_ownership_graph.py',
  'docs\superpowers\specs\2026-05-25-region-ownership-graph-design.md',
  'docs\superpowers\plans\2026-05-25-region-ownership-graph.md'
)
foreach ($file in $files) {
  $source=Join-Path $primary $file
  if (Test-Path $source) {
    $target=Join-Path $mirror $file
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
  }
}
```

If `parsers\pdf_parser.py` was not changed, it can be omitted from final sync.

- [ ] **Step 2: Verify hashes**

Run:

```powershell
$primary='D:\AutoIND-Pro\ind-compliance-ai'
$mirror='D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai'
$files=@(
  'parsers\pdf\region_ownership.py',
  'parsers\pdf\types.py',
  'parsers\pdf\pipeline.py',
  'tests\parser_tests\test_region_ownership_graph.py'
)
foreach ($file in $files) {
  $a=(Get-FileHash -Algorithm SHA256 (Join-Path $primary $file)).Hash
  $b=(Get-FileHash -Algorithm SHA256 (Join-Path $mirror $file)).Hash
  "$file $($a -eq $b)"
}
```

Expected: each line ends with `True`.

- [ ] **Step 3: Run mirror focused tests**

Run from mirror:

```powershell
D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_region_ownership_graph tests.deterministic_tests.test_parse_markdown_export
```

Expected: mirror focused tests pass.

---

## Final Review Checklist

- [ ] Region ownership is observe-only and does not alter Markdown.
- [ ] All new region objects serialize with audit fields.
- [ ] Figure nodes preserve `content_analysis.status=not_analyzed`.
- [ ] Table ownership decisions retain competing body candidates as audit evidence.
- [ ] Header/footer/page-number candidates stay outside body ownership.
- [ ] Protected A-tst, 2-column-tst, r2, and eCTD gates pass.
- [ ] Benchmark-specific projection remains outside production parser behavior.
