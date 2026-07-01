from __future__ import annotations

from pathlib import Path
import unittest

from parsers.pdf.postprocess_context import (
    _attach_section_context,
    _infer_module_context,
    _merge_section_context,
    _resolve_section_context_for_bbox,
)


class PostprocessContextTests(unittest.TestCase):
    def test_infer_module_context_detects_module_signal_from_filename(self) -> None:
        context = _infer_module_context(Path("m3-quality-overview.pdf"))

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context["module_label"], "M3")
        self.assertEqual(context["module_signal_hit"], "module_3")
        self.assertEqual(context["module_signal_source"], "filename")
        self.assertEqual(context["anchor_source"], "document_classification")

    def test_merge_section_context_lets_section_fields_override_module_context(self) -> None:
        merged = _merge_section_context(
            {
                "outline_index": "3.2.S.4.1",
                "section_title": "Specification",
                "anchor_source": "heading",
            },
            {
                "module_label": "M3",
                "anchor_source": "document_classification",
                "anchor_confidence": 0.9,
            },
        )

        self.assertEqual(
            merged,
            {
                "module_label": "M3",
                "anchor_source": "heading",
                "anchor_confidence": 0.9,
                "outline_index": "3.2.S.4.1",
                "section_title": "Specification",
            },
        )

    def test_resolve_section_context_uses_latest_anchor_above_bbox_and_module_context(self) -> None:
        context = _resolve_section_context_for_bbox(
            [50.0, 145.0, 400.0, 170.0],
            page_heading_anchors=[
                {"outline_index": "2.6", "section_title": "Overview", "anchor_bbox": [40.0, 80.0, 450.0, 100.0]},
                {"outline_index": "2.6.1", "section_title": "Details", "anchor_bbox": [40.0, 120.0, 450.0, 140.0]},
                {"outline_index": "2.6.2", "section_title": "Later", "anchor_bbox": [40.0, 190.0, 450.0, 210.0]},
            ],
            active_section_context={"outline_index": "2", "section_title": "Module Summary"},
            module_context={"module_label": "M4"},
        )

        self.assertEqual(context["module_label"], "M4")
        self.assertEqual(context["outline_index"], "2.6.1")
        self.assertEqual(context["section_title"], "Details")

    def test_resolve_section_context_falls_back_to_active_context_and_attach_deep_copies(self) -> None:
        section_context = {"outline_index": "1.1", "nested": {"source": "heading"}}
        resolved = _resolve_section_context_for_bbox(
            [50.0, 20.0, 400.0, 40.0],
            page_heading_anchors=[{"outline_index": "1.2", "anchor_bbox": [40.0, 80.0, 450.0, 100.0]}],
            active_section_context=section_context,
            module_context={"module_label": "M1"},
        )
        target: dict[str, object] = {}

        _attach_section_context(target, resolved)
        assert resolved is not None
        resolved["nested"]["source"] = "mutated"

        self.assertEqual(target["section_context"]["module_label"], "M1")
        self.assertEqual(target["section_context"]["outline_index"], "1.1")
        self.assertEqual(target["section_context"]["nested"]["source"], "heading")


if __name__ == "__main__":
    unittest.main()
