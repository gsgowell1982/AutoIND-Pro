from __future__ import annotations

import unittest

from parsers.pdf.raster_table_evidence import (
    CompositeRasterTableEvidenceProvider,
    RasterEvidenceSource,
    RasterTableRegionCandidate,
    TableEvidence,
    VisualRasterTableEvidenceProvider,
    fuse_raster_table_region_candidates,
    raster_table_candidate_to_region_candidate,
)
from parsers.pdf.region_ownership import RegionType


class RasterTableEvidenceContractTests(unittest.TestCase):
    def test_fuses_overlapping_raster_sources_without_losing_provenance(self) -> None:
        candidates = [
            RasterTableRegionCandidate(
                page=1,
                bbox=(100.0, 120.0, 420.0, 300.0),
                sources=[
                    RasterEvidenceSource(
                        source_type="visual_line_grid",
                        confidence=0.82,
                        bbox=(100.0, 120.0, 420.0, 300.0),
                        signals={"horizontal_line_count": 5},
                        provenance=["image:page1.png", "visual:grid"],
                    )
                ],
            ),
            RasterTableRegionCandidate(
                page=1,
                bbox=(108.0, 126.0, 430.0, 306.0),
                sources=[
                    RasterEvidenceSource(
                        source_type="ocr_text_matrix",
                        confidence=0.76,
                        bbox=(108.0, 126.0, 430.0, 306.0),
                        signals={"stable_column_count": 4},
                        provenance=["image:page1.png", "ocr:block-matrix"],
                    )
                ],
            ),
        ]

        fused = fuse_raster_table_region_candidates(candidates)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].bbox, (100.0, 120.0, 430.0, 306.0))
        self.assertEqual(
            [source.source_type for source in fused[0].sources],
            ["visual_line_grid", "ocr_text_matrix"],
        )
        self.assertGreater(fused[0].confidence, 0.82)
        self.assertIn("source_agreement", fused[0].signals)
        self.assertIn("image:page1.png", fused[0].provenance)
        self.assertIn("ocr:block-matrix", fused[0].provenance)

    def test_raster_candidate_converts_to_observe_only_region_candidate(self) -> None:
        candidate = RasterTableRegionCandidate(
            page=3,
            bbox=(50.0, 80.0, 500.0, 260.0),
            sources=[
                RasterEvidenceSource(
                    source_type="layout_model_table_region",
                    confidence=0.91,
                    bbox=(50.0, 80.0, 500.0, 260.0),
                    signals={"provider": "test-layout"},
                    provenance=["layout:test-layout:region-1"],
                )
            ],
            candidate_id="raster-table-p3-001",
        )

        region_candidate = raster_table_candidate_to_region_candidate(candidate)

        self.assertEqual(region_candidate.candidate_id, "raster-table-p3-001")
        self.assertEqual(region_candidate.page, 3)
        self.assertEqual(region_candidate.region_type, RegionType.TABLE)
        self.assertEqual(region_candidate.source, "raster_table_evidence")
        self.assertEqual(region_candidate.evidence_refs, ["layout:test-layout:region-1"])
        self.assertEqual(region_candidate.metadata["observe_only"], True)
        self.assertEqual(region_candidate.metadata["source_type"], "raster_image")
        self.assertEqual(region_candidate.signals["source_types"], ["layout_model_table_region"])

    def test_table_evidence_keeps_region_evidence_separate_from_ast_projection(self) -> None:
        evidence = TableEvidence(
            table_id="raster-table-p1-001",
            page=1,
            bbox=(10.0, 20.0, 300.0, 180.0),
            source_type="raster_image",
            region_confidence=0.88,
            evidence_sources=[
                {"source_type": "layout_model_table_region", "confidence": 0.88},
            ],
            text_tokens=[{"text": "Dose", "bbox": [20, 30, 60, 45]}],
            visual_lines=[],
            cell_candidates=[],
            caption_candidates=[],
            note_candidates=[],
            provenance=["layout:test"],
        )

        payload = evidence.to_dict()

        self.assertEqual(payload["table_id"], "raster-table-p1-001")
        self.assertEqual(payload["source_type"], "raster_image")
        self.assertEqual(payload["region_confidence"], 0.88)
        self.assertEqual(payload["text_tokens"][0]["text"], "Dose")
        self.assertNotIn("display_grid", payload)
        self.assertNotIn("semantic_grid", payload)


class _StaticRasterProvider:
    provider_name = "static_provider"

    def __init__(self, candidates: list[RasterTableRegionCandidate]) -> None:
        self._candidates = candidates

    def detect(self, image_path, *, page: int = 1) -> list[RasterTableRegionCandidate]:
        return self._candidates


class _FailingRasterProvider:
    provider_name = "failing_provider"

    def detect(self, image_path, *, page: int = 1) -> list[RasterTableRegionCandidate]:
        raise RuntimeError("synthetic provider failure")


class RasterTableEvidenceProviderTests(unittest.TestCase):
    def test_composite_provider_fuses_overlapping_provider_outputs(self) -> None:
        line_candidate = RasterTableRegionCandidate(
            page=1,
            bbox=(100.0, 100.0, 400.0, 260.0),
            sources=[
                RasterEvidenceSource(
                    source_type="visual_line_grid",
                    confidence=0.80,
                    bbox=(100.0, 100.0, 400.0, 260.0),
                    provenance=["visual:line-grid"],
                )
            ],
        )
        layout_candidate = RasterTableRegionCandidate(
            page=1,
            bbox=(106.0, 96.0, 410.0, 266.0),
            sources=[
                RasterEvidenceSource(
                    source_type="layout_model_table_region",
                    confidence=0.90,
                    bbox=(106.0, 96.0, 410.0, 266.0),
                    provenance=["layout:model"],
                )
            ],
        )
        provider = CompositeRasterTableEvidenceProvider(
            providers=[
                _StaticRasterProvider([line_candidate]),
                _StaticRasterProvider([layout_candidate]),
            ]
        )

        candidates = provider.detect("page.png", page=1)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_types, ["layout_model_table_region", "visual_line_grid"])
        self.assertEqual(candidates[0].bbox, (100.0, 96.0, 410.0, 266.0))
        self.assertEqual(candidates[0].signals["provider_count"], 2)
        self.assertIn("layout:model", candidates[0].provenance)
        self.assertIn("visual:line-grid", candidates[0].provenance)

    def test_composite_provider_records_provider_failure_without_dropping_other_evidence(self) -> None:
        surviving_candidate = RasterTableRegionCandidate(
            page=2,
            bbox=(20.0, 30.0, 200.0, 160.0),
            sources=[
                RasterEvidenceSource(
                    source_type="ocr_text_matrix",
                    confidence=0.74,
                    bbox=(20.0, 30.0, 200.0, 160.0),
                    provenance=["ocr:tokens"],
                )
            ],
        )
        provider = CompositeRasterTableEvidenceProvider(
            providers=[
                _FailingRasterProvider(),
                _StaticRasterProvider([surviving_candidate]),
            ]
        )

        candidates = provider.detect("page.png", page=2)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_types, ["ocr_text_matrix"])
        self.assertEqual(provider.last_warnings, ["provider_failed:failing_provider:RuntimeError"])
        self.assertIn("provider_failed:failing_provider:RuntimeError", candidates[0].warnings)

    def test_visual_provider_uses_existing_visual_detection_contract(self) -> None:
        provider = VisualRasterTableEvidenceProvider()

        self.assertEqual(provider.provider_name, "visual_raster_table_evidence")
        self.assertTrue(callable(provider.detect))


if __name__ == "__main__":
    unittest.main()
