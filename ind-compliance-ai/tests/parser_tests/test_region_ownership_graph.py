import unittest
from pathlib import Path

from parsers.pdf.postprocess import build_pdf_parse_result
from parsers.pdf.types import PdfPipelineState
from parsers.pdf.region_ownership import (
    OwnershipDecision,
    RegionCandidate,
    RegionNode,
    RegionType,
    arbitrate_region_ownership,
    collect_region_candidates,
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
                "images": [
                    {"image_id": "img_p1_001", "bbox": [40, 325, 300, 480], "kind": "image"},
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
        self.assertTrue(any(candidate.evidence_refs == ["image:img_p1_001"] for candidate in candidates))

    def test_collects_external_observe_only_region_candidates(self):
        external = RegionCandidate(
            candidate_id="raster-table-p1-001",
            page=1,
            region_type=RegionType.TABLE,
            bbox=(50.0, 80.0, 300.0, 220.0),
            source="raster_table_evidence",
            evidence_refs=["layout:table-detector:1"],
            confidence=0.88,
            metadata={"observe_only": True, "source_type": "raster_image"},
        )

        candidates = collect_region_candidates(
            page_payloads=[],
            table_asts=[],
            toc_nodes=[],
            figure_nodes=[],
            external_candidates=[external],
        )

        self.assertEqual(candidates, [external])


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


class RegionOwnershipStateTests(unittest.TestCase):
    def test_pipeline_state_can_store_region_ownership_payloads(self):
        state = PdfPipelineState()

        self.assertEqual(state.external_region_candidates, [])
        self.assertEqual(state.region_candidates, [])
        self.assertEqual(state.ownership_decisions, [])
        self.assertEqual(state.region_nodes, [])
        self.assertEqual(state.counters.region_candidate_count, 0)
        self.assertEqual(state.counters.region_node_count, 0)

    def test_parse_result_exposes_region_ownership_without_changing_text_payload(self):
        state = PdfPipelineState()
        state.region_candidates = [{"candidate_id": "cand-table", "region_type": "table"}]
        state.ownership_decisions = [{"region_id": "region-table", "region_type": "table"}]
        state.region_nodes = [{"region_id": "region-table", "region_type": "table"}]
        state.counters.region_candidate_count = 1
        state.counters.region_ownership_decision_count = 1
        state.counters.region_node_count = 1

        result = build_pdf_parse_result(Path("synthetic.pdf"), state)

        self.assertEqual(result["text"], "")
        self.assertEqual(result["region_ownership"]["candidates"], state.region_candidates)
        self.assertEqual(result["region_ownership"]["decisions"], state.ownership_decisions)
        self.assertEqual(result["region_ownership"]["nodes"], state.region_nodes)
        self.assertEqual(result["document_ast"]["region_refs"], ["region-table"])
        self.assertEqual(result["metadata"]["region_candidate_count"], 1)
        self.assertEqual(result["metadata"]["region_ownership_decision_count"], 1)
        self.assertEqual(result["metadata"]["region_node_count"], 1)
