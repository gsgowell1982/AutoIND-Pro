from __future__ import annotations

import os
from pathlib import Path
import re
import unittest

from api.main import (
    _build_full_markdown,
    _build_ind_review_heading_like_duplicate_audit,
    _normalize_ind_review_visibility_text,
)
from parsers.pdf_parser import parse_pdf


def _clean_text_for_test(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _business_table_surface_tokens_for_test(table: dict) -> list[str]:
    tokens: list[str] = []
    for row in (
        table.get("semantic_grid")
        or table.get("display_grid")
        or table.get("raw_grid")
        or []
    ):
        if not isinstance(row, list):
            continue
        row_text = _clean_text_for_test(" ".join(str(cell or "") for cell in row))
        if len(row_text) >= 8:
            tokens.append(row_text)
        for cell in row:
            cell_text = _clean_text_for_test(str(cell or ""))
            if len(cell_text) >= 4 and not cell_text.isdigit():
                tokens.append(cell_text)
    return tokens


def _resolve_r2_regression_pdf() -> Path:
    override = os.environ.get("IND_R2_REGRESSION_PDF", "").strip()
    if override:
        return Path(override)
    return Path(r"D:\AutoIND-Pro\r2.pdf")


class R2RegressionTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_path = _resolve_r2_regression_pdf()
        if not cls.sample_path.exists():
            raise unittest.SkipTest(
                "r2 regression sample not found. Set IND_R2_REGRESSION_PDF."
            )
        cls.result = parse_pdf(cls.sample_path)
        cls.metadata = cls.result["metadata"]
        cls.page2_tables = [
            table
            for table in cls.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 2
        ]

    def test_page2_keeps_ruled_one_row_history_tables(self) -> None:
        self.assertGreaterEqual(len(self.page2_tables), 4)

        one_row_tables = [
            table
            for table in self.page2_tables
            if len(table.get("display_grid", []) or table.get("raw_grid", []) or []) == 1
        ]
        self.assertEqual(len(one_row_tables), 2)

        one_row_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for table in one_row_tables
            for row in table.get("display_grid", []) or table.get("raw_grid", [])
        )
        self.assertIn("指导委员会批准进行较小的编辑审校。", one_row_text)
        self.assertIn("2002 年 12月20 日", one_row_text.replace("\n", ""))
        self.assertIn("指导委员会批准新增加的问题。", one_row_text)
        self.assertIn("2003 年11月11 日", one_row_text.replace("\n", ""))

        for table in one_row_tables:
            self.assertEqual(int(table.get("col_count", 0) or 0), 4)
            self.assertGreaterEqual(float(table.get("confidence", 0.0) or 0.0), 0.30)
            self.assertEqual(table.get("semantic_role"), "business_table")

    def test_page2_centered_descriptive_titles_bind_to_one_row_history_tables(self) -> None:
        one_row_tables = [
            table
            for table in self.page2_tables
            if len(table.get("display_grid", []) or table.get("raw_grid", []) or []) == 1
        ]
        self.assertEqual(len(one_row_tables), 2)

        def table_text(table: dict) -> str:
            return "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in table.get("display_grid", []) or table.get("raw_grid", [])
            )

        current_phase_table = next(
            table
            for table in one_row_tables
            if "M4S(R2)" in table_text(table)
        )
        current_qa_table = next(
            table
            for table in one_row_tables
            if "\u95ee\u7b54 (R4)" in table_text(table) or "\u95ee\u7b54(R4)" in table_text(table)
        )

        self.assertEqual(current_phase_table.get("title"), "\u73b0\u884c\u7b2c\u56db\u9636\u6bb5\u7248\u672c")
        self.assertEqual(current_phase_table.get("title_block", {}).get("source"), "descriptive_title_micro_table")
        self.assertEqual(
            current_phase_table.get("context_profile"),
            "descriptive_title_micro_table",
        )
        self.assertEqual(
            current_qa_table.get("title"),
            "\u7f51\u7ad9\u4e0a\u53d1\u5e03\u7684\u73b0\u884cM4S\u95ee\u7b54",
        )
        self.assertEqual(current_qa_table.get("title_block", {}).get("source"), "descriptive_title_micro_table")
        self.assertEqual(current_qa_table.get("context_profile"), "descriptive_title_micro_table")

        phase_notes = " ".join(
            str(segment.get("text") or "")
            for segment in current_phase_table.get("content_segments", []) or []
            if isinstance(segment, dict)
        )
        self.assertNotIn("\u4e3a\u4e86\u4fc3\u8fdbM4S", phase_notes)

    def test_page2_table_history_structured_flow_keeps_explanatory_url_continuation(self) -> None:
        page2 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 2)
        blocks = list(page2.get("blocks", []) or [])

        def block_index_containing(text: str) -> int:
            for index, block in enumerate(blocks):
                block_text = str(block.get("text") or block.get("title") or "")
                if text in block_text:
                    return index
            self.fail(f"Page 2 AST block containing {text!r} not found")

        intro_index = block_index_containing("\u4e3a\u4e86\u4fc3\u8fdbM4S")
        url_index = block_index_containing("\u7f51\u7ad9http://www.ich.org")
        qa_history_index = block_index_containing("M4S \u95ee\u7b54\u5386\u53f2")

        self.assertGreater(url_index, 0)
        self.assertLess(intro_index, url_index)
        self.assertLess(url_index, qa_history_index)

        page2_summary = next(page for page in self.result["pages"] if page["page_number"] == 2)
        diagnostics = dict(page2_summary.get("reading_order_diagnostics", {}) or {})
        self.assertNotEqual(diagnostics.get("strategy"), "zone_columns_left_then_right")
        self.assertIn("structured_table_flow", diagnostics.get("signals", []))

    def test_ind_review_markdown_keeps_page2_m4s_qa_explanatory_url_continuation(self) -> None:
        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")

        intro = "\u4e3a\u4e86\u4fc3\u8fdbM4S \u6307\u5bfc\u539f\u5219\u7684\u5b9e\u65bd"
        url = "\u7f51\u7ad9[http://www.ich.org](http://www.ich.org) \u4e0b\u8f7d\u3002"
        qa_history = "M4S \u95ee\u7b54\u5386\u53f2"

        intro_index = markdown.find(intro)
        url_index = markdown.find(url)
        qa_history_index = markdown.find(qa_history)

        self.assertGreaterEqual(intro_index, 0)
        self.assertGreaterEqual(url_index, 0)
        self.assertGreaterEqual(qa_history_index, 0)
        self.assertLess(intro_index, url_index)
        self.assertLess(url_index, qa_history_index)

    def test_page4_is_promoted_as_continuation_toc_page(self) -> None:
        page4_tocs = [
            toc
            for toc in self.result.get("toc_blocks", [])
            if int(toc.get("page", 0) or 0) == 4
        ]
        self.assertEqual(len(page4_tocs), 1)

        page4_toc = page4_tocs[0]
        self.assertGreaterEqual(int(page4_toc.get("entry_count", 0) or 0), 30)
        self.assertTrue(page4_toc.get("is_toc_continuation"))

        entries_text = "\n".join(
            str(entry.get("text", ""))
            for entry in page4_toc.get("entries", [])
        )
        self.assertIn("药代动力学文字总结", entries_text)
        self.assertIn("非临床列表总结-示例", entries_text)

        sequences = list(self.result.get("toc_sequences") or [])
        self.assertEqual(len(sequences), 1)
        sequence = sequences[0]
        self.assertEqual(sequence.get("pages"), [3, 4])
        self.assertGreaterEqual(int(sequence.get("entry_count", 0) or 0), 50)

    def test_r2_body_outline_markers_align_with_toc_sequences_through_shared_outline_rules(self) -> None:
        sequence = next(seq for seq in self.result.get("toc_sequences", []) or [] if seq.get("pages") == [3, 4])
        navigation_summary = dict(sequence.get("navigation_summary", {}) or {})
        outline_lookup = dict(navigation_summary.get("outline_path_lookup", {}) or {})
        self.assertEqual(outline_lookup.get("2.6 > 2.6.2 > 2.6.2.1"), [9])
        self.assertEqual(outline_lookup.get("2.6 > 2.6.4 > 2.6.4.9"), [26])
        self.assertEqual(outline_lookup.get("4.1"), [42])
        self.assertEqual(navigation_summary.get("cross_page_parent_link_count"), 4)
        self.assertEqual(sequence.get("root_entry_count"), 10)

    def test_toc_aligned_body_headings_promote_module_and_numbered_sections(self) -> None:
        def units_with_text(text: str) -> list[dict]:
            return [
                unit
                for unit in self.result.get("content_units", []) or []
                if str(unit.get("text") or "").strip() == text
            ]

        module2_units = [
            unit
            for unit in units_with_text("\u6a21\u57572\uff1a\u901a\u7528\u6280\u672f\u6587\u6863\u603b\u7ed3")
            if int(unit.get("page", 0) or 0) == 5
        ]
        self.assertEqual(len(module2_units), 1)
        module2 = module2_units[0]
        self.assertEqual(module2.get("semantic_role"), "section_heading")
        self.assertEqual(module2.get("unit_role"), "section_heading")
        self.assertEqual(module2.get("heading_profile"), "toc_aligned_body_heading")
        self.assertEqual(module2.get("section_context", {}).get("section_title"), module2.get("text"))

        heading_expectations = [
            (5, "2.4 \u975e\u4e34\u5e8a\u7efc\u8ff0", "2.4"),
            (7, "2.6 \u975e\u4e34\u5e8a\u6587\u5b57\u603b\u7ed3\u548c\u5217\u8868\u603b\u7ed3", "2.6"),
        ]
        for page, text, outline_index in heading_expectations:
            matches = [
                unit
                for unit in units_with_text(text)
                if int(unit.get("page", 0) or 0) == page
            ]
            self.assertEqual(len(matches), 1)
            unit = matches[0]
            self.assertEqual(unit.get("semantic_role"), "section_heading")
            self.assertEqual(unit.get("unit_role"), "section_heading")
            self.assertEqual(unit.get("section_context", {}).get("outline_index"), outline_index)

    def test_page5_embedded_overview_subheading_and_page7_bullets_do_not_confuse_toc_alignment(self) -> None:
        page5_overview_units = [
            unit
            for unit in self.result.get("content_units", []) or []
            if int(unit.get("page", 0) or 0) == 5
            and str(unit.get("text") or "").strip() == "\u6982\u51b5"
        ]
        self.assertEqual(len(page5_overview_units), 1)
        page5_overview = page5_overview_units[0]
        self.assertEqual(page5_overview.get("semantic_role"), "section_heading")
        self.assertEqual(page5_overview.get("unit_role"), "section_heading")
        self.assertEqual(
            page5_overview.get("heading_profile"),
            "toc_embedded_visual_subheading",
        )
        self.assertEqual(
            page5_overview.get("section_context", {}).get("outline_path"),
            "2.4 > \u6982\u51b5",
        )

        page7_bullet_units = [
            unit
            for unit in self.result.get("content_units", []) or []
            if int(unit.get("page", 0) or 0) == 7
            and str(unit.get("text") or "").strip() == "\uf06c \u5c40\u90e8\u8010\u53d7\u6027"
        ]
        self.assertEqual(len(page7_bullet_units), 1)
        page7_bullet = page7_bullet_units[0]
        self.assertNotEqual(page7_bullet.get("semantic_role"), "section_heading")
        self.assertNotEqual(page7_bullet.get("unit_role"), "section_heading")
        self.assertNotEqual(page7_bullet.get("heading_profile"), "toc_aligned_body_heading")

        page15_real_heading_units = [
            unit
            for unit in self.result.get("content_units", []) or []
            if int(unit.get("page", 0) or 0) == 15
            and str(unit.get("text") or "").strip() == "2.6.6.7 \u5c40\u90e8\u8010\u53d7\u6027"
        ]
        self.assertEqual(len(page15_real_heading_units), 1)
        page15_real_heading = page15_real_heading_units[0]
        self.assertEqual(page15_real_heading.get("semantic_role"), "section_heading")
        self.assertEqual(page15_real_heading.get("unit_role"), "section_heading")
        self.assertEqual(
            page15_real_heading.get("section_context", {}).get("outline_index"),
            "2.6.6.7",
        )

    def test_ind_review_markdown_renders_toc_aligned_body_headings(self) -> None:
        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("### \u6a21\u57572\uff1a\u901a\u7528\u6280\u672f\u6587\u6863\u603b\u7ed3", markdown)
        self.assertIn("#### 2.4 \u975e\u4e34\u5e8a\u7efc\u8ff0", markdown)
        self.assertIn("##### \u6982\u51b5", markdown)
        self.assertIn("#### 2.6 \u975e\u4e34\u5e8a\u6587\u5b57\u603b\u7ed3\u548c\u5217\u8868\u603b\u7ed3", markdown)
        self.assertNotIn("\n2.4 \u975e\u4e34\u5e8a\u7efc\u8ff0\n\n", markdown)
        self.assertNotIn("\n2.6 \u975e\u4e34\u5e8a\u6587\u5b57\u603b\u7ed3\u548c\u5217\u8868\u603b\u7ed3\n\n", markdown)
        self.assertNotIn("\u8d85\u8fc730 \u9875\u3002 \u6982\u51b5", markdown)
        self.assertNotIn("##### \uf06c \u5c40\u90e8\u8010\u53d7\u6027", markdown)
        self.assertIn("###### 2.6.6.7 \u5c40\u90e8\u8010\u53d7\u6027", markdown)

    def test_ind_review_markdown_merges_hanging_indent_bullet_continuations(self) -> None:
        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn(
            "\uf06c \u9057\u4f20\u6bd2\u6027 - "
            "\u5316\u5408\u7269\u7684\u5316\u5b66\u7ed3\u6784\u3001\u4f5c\u7528\u65b9\u5f0f\u3001"
            "\u4e0e\u5df2\u77e5\u9057\u4f20\u6bd2\u6027\u5316\u5408\u7269\u4e4b\u95f4\u7684\u5173\u7cfb",
            markdown,
        )
        self.assertIn(
            "\uf06c \u81f4\u764c\u6027 - "
            "\u5316\u5408\u7269\u7684\u5316\u5b66\u7ed3\u6784\u3001\u4e0e\u5df2\u77e5\u81f4\u764c\u7269\u7684\u5173\u7cfb"
            "\uff0c\u4ee5\u53ca\u9057\u4f20\u6bd2\u6027\u548c\u66b4\u9732\u6570\u636e",
            markdown,
        )
        self.assertNotIn("\n\n\u7684\u5173\u7cfb \uf06c \u81f4\u764c\u6027", markdown)
        self.assertNotIn("\n\n\u9732\u6570\u636e \uf06c \u5bf9\u4eba\u7684\u81f4\u764c\u98ce\u9669", markdown)

    def test_page21_figure_caption_axis_labels_and_legend_are_owned_without_markdown_duplication(self) -> None:
        page21 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 21)
        images = [
            block
            for block in page21.get("blocks", []) or []
            if block.get("block_type") == "image"
        ]
        self.assertEqual(len(images), 1)
        image = images[0]

        self.assertIn("\u56feX", str(image.get("title") or image.get("caption_text") or ""))
        self.assertIn("SHRaX \u957f\u671f\u7ed9\u836f\u7684\u8840\u538b", str(image.get("title") or image.get("caption_text") or ""))

        segments = [segment for segment in image.get("content_segments", []) or [] if isinstance(segment, dict)]
        above_caption = [
            segment
            for segment in segments
            if segment.get("role") == "caption" and segment.get("relation") == "above"
        ]
        below_legend = [
            segment
            for segment in segments
            if segment.get("role") == "legend" and segment.get("relation") == "below"
        ]
        self.assertEqual(len(above_caption), 1)
        self.assertEqual(len(below_legend), 1)
        self.assertIn("SHRaX \u957f\u671f\u7ed9\u836f\u7684\u8840\u538b", above_caption[0].get("text", ""))
        self.assertIn("\u751f\u7406\u76d0\u6c34\u9884\u5904\u7406\u7ec4\u8fbe\u5230\u4e86\u7edf\u8ba1\u5b66\u663e\u8457\u6027", below_legend[0].get("text", ""))

        owned_ids = set(image.get("owned_text_block_ids", []) or [])
        self.assertIn("txt_p21_003", owned_ids)
        self.assertIn("txt_p21_004", owned_ids)
        self.assertIn("txt_p21_005", owned_ids)
        for suffix in range(6, 12):
            self.assertIn(f"txt_p21_{suffix:03d}", owned_ids)

        embedded_text = " ".join(
            str(segment.get("text") or "")
            for segment in segments
            if segment.get("role") in {"embedded_text", "axis_label"}
        )
        self.assertIn("\u5e73\u5747\u8840\u538b\uff08mmHg\uff09", embedded_text)
        self.assertIn("\u65f6\u95f4\uff08\u5206\u949f\uff09", embedded_text)

        composite = image.get("composite_object") or {}
        self.assertEqual(composite.get("object_family"), "figure")
        self.assertEqual(composite.get("ownership_domain"), "figure")
        self.assertEqual(composite.get("reading_flow"), "composite_object")
        self.assertEqual(composite.get("body_flow_policy"), "exclude_owned_text_from_body")
        self.assertEqual(composite.get("presentation_order"), ["title", "body", "notes"])
        self.assertEqual(composite.get("title_policy"), "owned_object_title")
        self.assertEqual(composite.get("note_policy"), "owned_object_notes")
        self.assertTrue(composite.get("has_title"))
        self.assertTrue(composite.get("has_body"))
        self.assertTrue(composite.get("has_notes"))
        self.assertIn("title", composite.get("component_order", []) or [])
        self.assertIn("body", composite.get("component_order", []) or [])
        self.assertIn("notes", composite.get("component_order", []) or [])
        self.assertTrue(
            any(
                boundary.get("note_profile", {}).get("profile_type") == "figure_legend_note"
                and boundary.get("relation") == "below"
                for boundary in composite.get("note_boundaries", []) or []
            ),
            msg=f"missing figure legend note boundary in {composite!r}",
        )
        image_evidence = next(
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "image"
            and evidence.get("source_id") == image.get("image_id")
        )
        evidence_composite = image_evidence.get("composite_object") or {}
        self.assertEqual(evidence_composite.get("object_family"), "figure")
        self.assertEqual(evidence_composite.get("ownership_domain"), "figure")
        self.assertEqual(evidence_composite.get("reading_flow"), "composite_object")
        self.assertEqual(evidence_composite.get("presentation_order"), ["title", "body", "notes"])

        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        caption_index = markdown.find("\u56feX SHRaX \u957f\u671f\u7ed9\u836f\u7684\u8840\u538b")
        image_index = markdown.find("![Figure 1](data:image/png;base64")
        self.assertGreaterEqual(caption_index, 0)
        self.assertGreaterEqual(image_index, 0)
        self.assertLess(caption_index, image_index)
        self.assertEqual(markdown.count("\u751f\u7406\u76d0\u6c34\u9884\u5904\u7406\u7ec4\u8fbe\u5230\u4e86\u7edf\u8ba1\u5b66\u663e\u8457\u6027"), 1)
        self.assertNotIn("\n\n\u5e73\u5747\u8840\u538b\uff08mmHg\uff09\n\n\u65f6\u95f4\uff08\u5206\u949f\uff09\n\n", markdown)
        self.assertNotIn("\u65f6\u95f4\uff08\u5206\u949f\uff09\n\n\u6b21\u53e3\u670d\u751f\u7406\u76d0\u6c341 ml/kg", markdown)

    def test_page28_no_page_outline_template_is_structured_without_polluting_toc(self) -> None:
        page28_tocs = [
            toc
            for toc in self.result.get("toc_blocks", []) or []
            if int(toc.get("page", 0) or 0) == 28
        ]
        self.assertEqual(page28_tocs, [])

        templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 28
        ]
        self.assertEqual(len(templates), 1)
        template = templates[0]
        self.assertEqual(template.get("semantic_role"), "structure_template")
        self.assertEqual(template.get("template_kind"), "no_page_outline")
        self.assertEqual(template.get("page_target_policy"), "no_page_targets")
        self.assertGreaterEqual(int(template.get("entry_count", 0) or 0), 20)
        self.assertEqual(template.get("title"), "\u975e\u4e34\u5e8a\u5217\u8868\u603b\u7ed3-\u6a21\u677f")

        entries = list(template.get("entries", []) or [])
        by_outline = {str(entry.get("outline_index") or ""): entry for entry in entries}
        self.assertIn("2.6.3", by_outline)
        self.assertIn("2.6.5.16", by_outline)
        self.assertIn("2.6.7.3", by_outline)
        self.assertEqual(by_outline["2.6.3"].get("title"), "\u836f\u7406\u5b66")
        self.assertIsNone(by_outline["2.6.3"].get("page_target"))
        self.assertEqual(by_outline["2.6.3"].get("mapping_status"), "template_no_page_target")
        self.assertEqual(by_outline["2.6.5.16"].get("parent_outline_index"), "2.6.5")

        page28 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 28)
        template_nodes = [
            block for block in page28.get("blocks", []) or []
            if block.get("block_type") == "structure_template"
        ]
        self.assertEqual(len(template_nodes), 1)
        self.assertEqual(template_nodes[0].get("structure_template_id"), template.get("structure_template_id"))

    def test_page29_no_page_outline_template_continues_page28_template(self) -> None:
        page29_tocs = [
            toc
            for toc in self.result.get("toc_blocks", []) or []
            if int(toc.get("page", 0) or 0) == 29
        ]
        self.assertEqual(page29_tocs, [])

        page28_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 28
        )
        page29_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 29
        ]
        self.assertEqual(len(page29_templates), 1)
        page29_template = page29_templates[0]
        self.assertEqual(page29_template.get("semantic_role"), "structure_template")
        self.assertEqual(page29_template.get("template_kind"), "no_page_outline")
        self.assertEqual(page29_template.get("page_target_policy"), "no_page_targets")
        self.assertTrue(page29_template.get("is_structure_template_continuation"))
        self.assertEqual(page29_template.get("continued_from_structure_template_id"), page28_template.get("structure_template_id"))
        self.assertEqual(page29_template.get("continued_from_page"), 28)
        self.assertEqual(page29_template.get("continuation_parent_outline_index"), "2.6.7")

        entries = list(page29_template.get("entries", []) or [])
        by_outline = {str(entry.get("outline_index") or ""): entry for entry in entries}
        self.assertIn("2.6.7.4", by_outline)
        self.assertIn("2.6.7.17", by_outline)
        self.assertEqual(by_outline["2.6.7.4"].get("parent_outline_index"), "2.6.7")
        self.assertIsNone(by_outline["2.6.7.17"].get("page_target"))
        self.assertEqual(by_outline["2.6.7.17"].get("mapping_status"), "template_no_page_target")

        page29 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 29)
        template_nodes = [
            block for block in page29.get("blocks", []) or []
            if block.get("block_type") == "structure_template"
        ]
        self.assertEqual(len(template_nodes), 1)
        self.assertEqual(template_nodes[0].get("structure_template_id"), page29_template.get("structure_template_id"))

        page29_text_roles = {
            block.get("text"): block.get("semantic_role")
            for block in page29.get("blocks", []) or []
            if block.get("block_type") == "text"
        }
        self.assertEqual(page29_text_roles.get("2.6.7.4 \u6bd2\u7406\u5b66\uff1a\u539f\u6599\u836f"), "structure_template_entry")
        self.assertEqual(page29_text_roles.get("2.6.7.17 \u5176\u4ed6\u6bd2\u6027\u8bd5\u9a8c"), "structure_template_entry")

        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        self.assertIn("#### \u975e\u4e34\u5e8a\u5217\u8868\u603b\u7ed3-\u6a21\u677f\n\n", markdown)
        self.assertIn("#### \u975e\u4e34\u5e8a\u5217\u8868\u603b\u7ed3-\u6a21\u677f\uff08\u7eed\uff09", markdown)
        self.assertNotIn("\u975e\u4e34\u5e8a\u5217\u8868\u603b\u7ed3-\u6a21\u677f\u7ed3\u6784\u6a21\u677f", markdown)

    def test_page28_redundant_structure_template_heading_is_suppressed_without_mojibake_placeholder(self) -> None:
        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        template_heading = "#### \u975e\u4e34\u5e8a\u5217\u8868\u603b\u7ed3-\u6a21\u677f"
        page28_anchor = markdown.find(template_heading)
        self.assertGreaterEqual(page28_anchor, 0)
        page29_anchor = markdown.find("#### \u975e\u4e34\u5e8a\u5217\u8868\u603b\u7ed3-\u6a21\u677f\uff08\u7eed\uff09", page28_anchor + 1)
        self.assertGreater(page29_anchor, page28_anchor)
        page28_markdown = markdown[page28_anchor:page29_anchor]

        self.assertIn("  - 2.6.3 \u836f\u7406\u5b66", page28_markdown)
        self.assertIn("    - 2.6.3.1 \u836f\u7406\u5b66\uff1a\u6982\u8ff0", page28_markdown)
        self.assertNotIn("#### \u7ed3\u6784\u6a21\u677f", page28_markdown)
        self.assertNotIn("#### \u7f01\u64b4\u702f\u59af\u2103\u6f98", page28_markdown)
        self.assertNotIn("\u7f01\u64b4\u701b", page28_markdown)
        self.assertNotIn("\u7f02\u4f7d\u632b", page28_markdown)
        self.assertNotIn("\u59af\u00b0", page28_markdown)

    def test_pages30_to_40_blank_tabular_templates_do_not_pollute_business_tables(self) -> None:
        blank_tabular_profiles = {"tabular_form_template", "blank_study_summary_template"}
        template_pages = {
            int(template.get("page", 0) or 0)
            for template in self.result.get("structure_templates", []) or []
            if str(template.get("template_profile") or "") in blank_tabular_profiles
        }
        self.assertTrue({30, 33, 38}.issubset(template_pages), msg=f"template pages: {sorted(template_pages)}")

        page38_business_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 38
            and str(table.get("semantic_role") or "business_table") == "business_table"
        ]
        self.assertEqual(page38_business_tables, [])

        page38_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 38
            and str(template.get("template_profile") or "") in blank_tabular_profiles
        ]
        self.assertEqual(len(page38_templates), 1)
        self.assertTrue(all(template.get("data_population") == "blank" for template in page38_templates))
        page38_rows_by_template = [
            "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
            for template in page38_templates
        ]
        self.assertTrue(
            all("\u0043\u0054\u0044 \u4e2d\u7684\u4f4d\u7f6e\uff1a\u5377\u3001\u9875\u7801" in rows for rows in page38_rows_by_template),
            msg=page38_rows_by_template,
        )
        self.assertIn("\u80ce\u76d8\u8f6c\u8fd0", page38_rows_by_template[0])
        self.assertIn("\u4e73\u6c41\u6392\u6cc4", page38_rows_by_template[0])
        self.assertIn("\u65b0\u751f\u80ce\u4ed4\uff1a", page38_rows_by_template[0])

        page38 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 38)
        page38_template_nodes = [
            block for block in page38.get("blocks", []) or []
            if block.get("block_type") == "structure_template"
            and block.get("template_profile") in blank_tabular_profiles
        ]
        self.assertEqual(len(page38_template_nodes), 1)

        page33_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 33
            and str(template.get("template_profile") or "") == "tabular_form_template"
        )
        page33_rows = "\n".join(str(row or "") for row in page33_template.get("row_texts", []) or [])
        self.assertIn("CTD \u4e2d\u7684\u4f4d\u7f6e\uff1a\u5377\u3001\u9875\u7801", page33_rows)
        self.assertIn("\u8bd5\u9a8c\u7f16\u53f7\uff1a", page33_rows)

        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        self.assertIn("\u836f\u4ee3\u52a8\u529b\u5b66\uff1a\u5355\u6b21\u7ed9\u836f\u540e\u7684\u5438\u6536 \u4f9b\u8bd5\u54c1\uff1a(1)", markdown)
        self.assertIn("- CTD\u4e2d\u7684\u4f4d\u7f6e\uff1a\u5377\u3001\u9875\u7801", markdown)
        self.assertIn("- \u8bd5\u9a8c\u7f16\u53f7\uff1a", markdown)

        after_page80_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if 77 <= int(table.get("page", 0) or 0) <= 80
            and str(table.get("semantic_role") or "business_table") == "business_table"
        ]
        self.assertGreaterEqual(len(after_page80_tables), 4)

    def test_page41_blank_tabular_template_absorbs_internal_label_only_table_surface(self) -> None:
        page41_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 41
            and str(template.get("template_profile") or "") == "tabular_form_template"
            and "2.6.5.9" in str(template.get("title") or "")
        )
        self.assertEqual(page41_template.get("data_population"), "blank")
        self.assertEqual(page41_template.get("ownership_domain"), "template_form")
        page41_rows = "\n".join(str(row or "") for row in page41_template.get("row_texts", []) or [])
        self.assertIn("\u6837\u54c1\u4e2d\u7684\u5316\u5408\u7269%", page41_rows)
        self.assertIn("\u0054\u0044 \u4e2d\u7684\u4f4d\u7f6e", page41_rows.replace("C", ""))
        self.assertIn("\u8bd5\u9a8c\u7f16\u53f7", page41_rows)

        page41_business_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 41
            and str(table.get("semantic_role") or "business_table") == "business_table"
        ]
        self.assertEqual(page41_business_tables, [])

        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        heading = "2.6.5.9 \u836f\u4ee3\u52a8\u529b\u5b66\uff1a\u4f53\u5185\u4ee3\u8c22 \u4f9b\u8bd5\u54c1"
        start = markdown.find(heading)
        self.assertGreaterEqual(start, 0)
        end = markdown.find("2.6.5.10 \u836f\u4ee3\u52a8\u529b\u5b66\uff1a\u4f53\u5916\u4ee3\u8c22", start + 1)
        self.assertGreater(end, start)
        page41_markdown = markdown[start:end]

        self.assertIn("- \u6837\u54c1\u4e2d\u7684\u5316\u5408\u7269%", page41_markdown)
        self.assertIn("- \u8bd5\u9a8c\u7f16\u53f7", page41_markdown)
        self.assertNotIn("| \u8840\u6d46 |", page41_markdown)
        self.assertNotIn("| --- |", page41_markdown)
        self.assertIn("\u5907\u6ce8\uff1a\u5982\u6709\u4eba\u4f53\u6570\u636e", page41_markdown)

    def test_page31_sparse_blank_tabular_form_skeleton_links_letter_marker_note_without_business_table_pollution(self) -> None:
        page31_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 31
        ]
        self.assertEqual(len(page31_templates), 1)
        template = page31_templates[0]
        self.assertEqual(template.get("semantic_role"), "structure_template")
        self.assertEqual(template.get("template_kind"), "tabular_form")
        self.assertEqual(template.get("template_profile"), "sparse_tabular_form_skeleton")
        self.assertEqual(template.get("ownership_domain"), "template_form")
        self.assertEqual(template.get("data_population"), "blank")

        rows_text = "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
        self.assertIn("\u8bc4\u4ef7\u7684\u5668\u5b98\u7cfb", rows_text)
        self.assertIn("\u79cd\u5c5e/\u54c1\u7cfb \u7ed9\u836f\u65b9\u6cd5 \u5242\u91cfa", rows_text)
        self.assertIn("\u0047\u004c\u0050 \u4f9d\u4ece\u6027 \u8bd5\u9a8c\u7f16\u53f7(3)", rows_text)

        note_refs = template.get("local_note_refs", []) or []
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and "\u5242\u91cf" in str(ref.get("anchor_text") or "")
                and "\u5355\u6b21\u7ed9\u836f" in str(ref.get("note_text") or "")
                for ref in note_refs
            ),
            msg=f"missing dose-a note ref in {note_refs!r}",
        )

        page31_business_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 31
            and str(table.get("semantic_role") or "business_table") == "business_table"
        ]
        self.assertEqual(page31_business_tables, [])

        page31 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 31)
        page31_template_nodes = [
            block
            for block in page31.get("blocks", []) or []
            if block.get("block_type") == "structure_template"
            and block.get("template_profile") == "sparse_tabular_form_skeleton"
        ]
        self.assertEqual(len(page31_template_nodes), 1)

        for page_number in (34, 39, 40):
            self.assertFalse(
                [
                    template
                    for template in self.result.get("structure_templates", []) or []
                    if int(template.get("page", 0) or 0) == page_number
                    and str(template.get("template_profile") or "") in {
                        "tabular_form_template",
                        "sparse_tabular_form_skeleton",
                    }
                ],
                msg=f"page {page_number} should remain body text/heading, not a tabular form template",
            )

    def test_page42_low_text_ruled_blank_template_is_structure_template_not_body_or_business_table(self) -> None:
        page42_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 42
        ]
        self.assertEqual(len(page42_templates), 1)
        template = page42_templates[0]
        self.assertEqual(template.get("semantic_role"), "structure_template")
        self.assertEqual(template.get("template_kind"), "tabular_form")
        self.assertEqual(template.get("template_profile"), "low_text_ruled_tabular_form_template")
        self.assertEqual(template.get("ownership_domain"), "template_form")
        self.assertEqual(template.get("data_population"), "blank")

        rows_text = "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
        self.assertIn("2.6.5.10 \u836f\u4ee3\u52a8\u529b\u5b66\uff1a\u4f53\u5916\u4ee3\u8c22 \u4f9b\u8bd5\u54c1\uff1a", rows_text)
        self.assertIn("CTD \u4e2d\u7684\u4f4d\u7f6e\uff1a\u5377\u3001\u9875\u7801", rows_text)
        self.assertIn("\u8bd5\u9a8c\u7cfb\u7edf\uff1a", rows_text)
        self.assertIn("\u539f\u5f62\u5316\u5408\u7269", rows_text)
        self.assertIn("\u9644\u52a0\u4fe1\u606f\uff1a", rows_text)
        self.assertGreaterEqual(
            int((template.get("semantic_signals") or {}).get("horizontal_rule_count", 0) or 0),
            2,
        )

        page42_business_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 42
            and str(table.get("semantic_role") or "business_table") == "business_table"
        ]
        self.assertEqual(page42_business_tables, [])

        page42 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 42)
        page42_template_nodes = [
            block
            for block in page42.get("blocks", []) or []
            if block.get("block_type") == "structure_template"
            and block.get("template_profile") == "low_text_ruled_tabular_form_template"
        ]
        self.assertEqual(len(page42_template_nodes), 1)

        for page_number in (43, 46, 51):
            self.assertFalse(
                [
                    template
                    for template in self.result.get("structure_templates", []) or []
                    if int(template.get("page", 0) or 0) == page_number
                    and str(template.get("template_profile") or "") == "low_text_ruled_tabular_form_template"
                ],
                msg=f"page {page_number} is an explanatory body page, not a low-text ruled template",
            )

    def test_pages52_to_62_blank_study_summary_templates_do_not_become_business_tables(self) -> None:
        template_pages = {52, 53, 54, 55, 56, 58, 59, 60, 61}
        explanatory_pages = {57, 62}
        business_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if 52 <= int(table.get("page", 0) or 0) <= 62
            and str(table.get("semantic_role") or "business_table") == "business_table"
        ]
        self.assertEqual(
            business_tables,
            [],
            msg="blank study-summary templates and their instruction notes must not pollute business_table ownership",
        )

        templates_by_page = {
            page: [
                template
                for template in self.result.get("structure_templates", []) or []
                if int(template.get("page", 0) or 0) == page
                and template.get("template_kind") == "tabular_form"
                and template.get("ownership_domain") == "template_form"
            ]
            for page in template_pages | explanatory_pages
        }
        for page in sorted(template_pages):
            self.assertTrue(templates_by_page[page], msg=f"page {page} should be a blank tabular-form template")
            rows_text = "\n".join(
                str(row or "")
                for template in templates_by_page[page]
                for row in template.get("row_texts", []) or []
            )
            self.assertTrue(rows_text.strip(), msg=f"page {page} template should preserve row evidence")
            self.assertTrue(
                any(template.get("data_population") == "blank" for template in templates_by_page[page]),
                msg=f"page {page} template should be blank/sparse, not a populated business table",
            )

        for page in sorted(explanatory_pages):
            self.assertEqual(
                templates_by_page[page],
                [],
                msg=f"page {page} is continuous explanatory note text, not a blank tabular-form template",
            )
            page_ast = next(item for item in self.result["document_ast"]["pages"] if item["page"] == page)
            page_text = "\n".join(str(block.get("text") or "") for block in page_ast.get("blocks", []) or [])
            self.assertIn("\u88682.6.7.", page_text)
            self.assertIn("\u6ce8\u91ca", page_text)

    def test_page58_blank_genotoxicity_template_owns_matrix_labels_and_notes(self) -> None:
        page58_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 58
            and str(template.get("template_profile") or "") == "blank_study_summary_template"
        ]
        self.assertEqual(len(page58_templates), 1)
        template = page58_templates[0]

        row_text = "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
        for expected in (
            "\u6240\u68c0\u6d4b\u7684\u8bf1\u5bfc\u4f5c\u7528\uff1a \u72ec\u7acb\u8bd5\u9a8c\u6b21\u6570\uff1a \u8bd5\u9a8c\u7f16\u53f7\uff1a",
            "\u54c1\u7cfb\uff1a \u5e73\u884c\u57f9\u517b\u7269\u6570\u91cf\uff1a CTD \u4e2d\u7684\u4f4d\u7f6e\uff1a\u5377\u3001\u9875\u7801",
            "\u6d53\u5ea6\u6216\u5242\u91cf\u6c34\u5e73",
            "\u4ee3\u8c22\u6d3b\u5316",
            "\u4f9b\u8bd5\u54c1",
            "\u65e0\u4ee3\u8c22\u6d3b\u5316",
            "\u6709\u4ee3\u8c22\u6d3b\u5316",
        ):
            self.assertIn(expected, row_text)

        note_text = "\n".join(str(note.get("text") or "") for note in template.get("note_blocks", []) or [])
        self.assertIn("\u6ce8\u91ca\uff1a", note_text)
        self.assertIn("\uff081\uff09 \u5e94\u5bf9\u8868\u683c\u8fdb\u884c\u8fde\u7eed\u7f16\u53f7", note_text)
        self.assertIn("\uff085\uff09 \u5e94\u6ce8\u660e\u7edf\u8ba1\u5b66\u5206\u6790\u65b9\u6cd5", note_text)

        page58 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 58)
        body_texts = {
            str(block.get("text") or "")
            for block in page58.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("semantic_role") or "") == "text_block"
        }
        for owned_text in (
            "\u4ee3\u8c22\u6d3b\u5316",
            "\u4f9b\u8bd5\u54c1",
            "(3)",
            "\u65e0\u4ee3\u8c22\u6d3b\u5316",
            "(4)",
            "\u6709\u4ee3\u8c22\u6d3b\u5316",
            "\u6ce8\u91ca\uff1a",
        ):
            self.assertNotIn(owned_text, body_texts)

    def test_ind_review_markdown_renders_page58_blank_template_without_parser_explanation(self) -> None:
        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")

        title = "2.6.7.8 (1)\u9057\u4f20\u6bd2\u6027\uff1a\u4f53\u5916 \u62a5\u544a\u6807\u9898\uff1a \u4f9b\u8bd5\u54c1\uff1a(2)"
        field = "\u6240\u68c0\u6d4b\u7684\u8bf1\u5bfc\u4f5c\u7528\uff1a \u72ec\u7acb\u8bd5\u9a8c\u6b21\u6570\uff1a \u8bd5\u9a8c\u7f16\u53f7\uff1a"
        matrix_label = "\u4ee3\u8c22\u6d3b\u5316"
        note = "\uff081\uff09 \u5e94\u5bf9\u8868\u683c\u8fdb\u884c\u8fde\u7eed\u7f16\u53f7"

        title_index = markdown.find(title)
        field_index = markdown.find(field, title_index)
        matrix_index = markdown.find(matrix_label, title_index)
        note_index = markdown.find(note, title_index)

        self.assertGreaterEqual(title_index, 0)
        self.assertGreater(field_index, title_index)
        self.assertGreater(matrix_index, field_index)
        self.assertGreater(note_index, matrix_index)
        self.assertNotIn("\u6a21\u677f\u77e9\u9635", markdown)
        self.assertNotIn("template matrix", markdown.lower())

    def test_page59_in_vivo_genotoxicity_template_separates_prior_note_and_owns_remarks(self) -> None:
        page58_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 58
            and str(template.get("template_profile") or "") == "blank_study_summary_template"
        )
        page59_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 59
            and str(template.get("template_profile") or "") == "blank_study_summary_template"
        ]
        self.assertEqual(len(page59_templates), 1)
        template = page59_templates[0]
        self.assertIn("2.6.7.9", str(template.get("title") or ""))
        self.assertIn("\u9057\u4f20\u6bd2\u6027\uff1a\u4f53\u5185", str(template.get("title") or ""))
        self.assertNotIn("2.6.7.8", str(template.get("title") or ""))

        row_text = "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
        self.assertNotIn("(5)*-p<0.05 **-p<0.01", row_text)
        for expected in (
            "2.6.7.9 (1)\u9057\u4f20\u6bd2\u6027\uff1a\u4f53\u5185 \u62a5\u544a\u6807\u9898\uff1a \u4f9b\u8bd5\u54c1\uff1a(2)",
            "\u6240\u68c0\u6d4b\u7684\u8bf1\u5bfc\u4f5c\u7528\uff1a \u7ed9\u836f\u65b9\u6848\uff1a \u8bd5\u9a8c\u7f16\u53f7\uff1a",
            "\u79cd\u5c5e/\u54c1\u7cfb\uff1a \u91c7\u6837\u65f6\u95f4\uff1a CTD \u4e2d\u7684\u4f4d\u7f6e\uff1a\u5377\u3001\u9875\u7801",
            "\u8bc4\u4ef7\u7684\u7ec6\u80de\uff1a \u6eb6\u5a92/\u5242\u578b\uff1a GLP \u4f9d\u4ece\u6027\uff1a",
            "\u6bd2\u6027/\u7ec6\u80de\u6bd2\u6027\u4f5c\u7528\uff1a",
            "\u9057\u4f20\u6bd2\u6027\u4f5c\u7528\uff1a",
            "\u66b4\u9732\u7684\u8bc1\u636e\uff1a",
            "\u4f9b\u8bd5\u54c1 \u5242\u91cf(mg/kg) \u52a8\u7269\u6570\u91cf",
        ):
            self.assertIn(expected, row_text)

        page58_notes = "\n".join(str(note.get("text") or "") for note in page58_template.get("note_blocks", []) or [])
        self.assertIn("(5)*-p<0.05 **-p<0.01", page58_notes)

        note_text = "\n".join(str(note.get("text") or "") for note in template.get("note_blocks", []) or [])
        self.assertIn("\u5907\u6ce8\uff1a", note_text)
        self.assertIn("\uff081\uff09 \u5e94\u5bf9\u8868\u683c\u8fdb\u884c\u8fde\u7eed\u7f16\u53f7", note_text)
        self.assertIn("2.6.7.9A", note_text)
        self.assertIn("\uff083\uff09 \u5e94\u6ce8\u660e\u7edf\u8ba1\u5b66\u5206\u6790\u65b9\u6cd5", note_text)

        for composite_template in (page58_template, template):
            composite = composite_template.get("composite_object") or {}
            self.assertEqual(composite.get("object_family"), "structure_template")
            self.assertEqual(composite.get("ownership_domain"), "template_form")
            self.assertEqual(composite.get("reading_flow"), "composite_object")
            self.assertEqual(composite.get("body_flow_policy"), "exclude_owned_text_from_body")
            self.assertEqual(composite.get("presentation_order"), ["title", "body", "notes"])
            self.assertEqual(composite.get("title_policy"), "owned_object_title")
            self.assertEqual(composite.get("note_policy"), "owned_object_notes")
            self.assertTrue(composite.get("has_title"))
            self.assertTrue(composite.get("has_body"))
            self.assertTrue(composite.get("has_notes"))
            self.assertGreaterEqual(int(composite.get("note_block_count", 0) or 0), 1)
            self.assertIn("title", composite.get("component_order", []) or [])
            self.assertIn("body", composite.get("component_order", []) or [])
            self.assertIn("notes", composite.get("component_order", []) or [])
            for note_boundary in composite.get("note_boundaries", []) or []:
                self.assertIn(note_boundary.get("end_boundary"), {"new_object_or_section", "page_boundary", "unknown"})
                profile = note_boundary.get("note_profile") or {}
                self.assertIn(
                    profile.get("profile_type"),
                    {"explicit_note_heading", "numbered_note", "symbol_note", "page_top_continuation_note"},
                )
                self.assertIn(profile.get("owner_context"), {"current_composite_object", "previous_composite_object"})
                self.assertTrue(profile.get("requires_anchor"))

        page58_composite = page58_template.get("composite_object") or {}
        self.assertEqual(
            page58_composite.get("continuation_policy"),
            "accept_page_top_leading_notes_before_next_object",
        )
        self.assertTrue(
            any(
                ref.get("anchor_status") == "previous_note_sequence_anchor"
                and ref.get("note_profile", {}).get("profile_type") == "page_top_continuation_note"
                and "(5)*-p<0.05" in str(ref.get("note_text") or "")
                for ref in page58_composite.get("note_anchor_refs", []) or []
            ),
            msg=f"missing previous-note-sequence anchor in {page58_composite!r}",
        )

        page59 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 59)
        template_nodes = [
            block
            for block in page59.get("blocks", []) or []
            if block.get("block_type") == "structure_template"
            and block.get("template_profile") == "blank_study_summary_template"
        ]
        self.assertEqual(len(template_nodes), 1)
        self.assertIn("2.6.7.9", str(template_nodes[0].get("title") or ""))
        self.assertEqual(
            (template_nodes[0].get("composite_object") or {}).get("presentation_order"),
            ["title", "body", "notes"],
        )
        body_texts = {
            str(block.get("text") or "")
            for block in page59.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("semantic_role") or "") == "text_block"
        }
        self.assertNotIn("(5)*-p<0.05 **-p<0.01", body_texts)
        for owned_text in (
            "\u5907\u6ce8\uff1a",
            "\uff081\uff09 \u5e94\u5bf9\u8868\u683c\u8fdb\u884c\u8fde\u7eed\u7f16\u53f7\uff08\u4f8b\u5982\uff1a2.6.7.9A\u30012.6.7.9B\uff09\u3002",
            "\uff082\uff09 \u56fd\u9645\u975e\u4e13\u5229\u836f\u54c1\u540d\u79f0\uff08INN\uff09\u3002",
            "\uff083\uff09 \u5e94\u6ce8\u660e\u7edf\u8ba1\u5b66\u5206\u6790\u65b9\u6cd5\u3002",
        ):
            self.assertNotIn(owned_text, body_texts)

    def test_ind_review_markdown_renders_page59_template_and_remarks_cleanly(self) -> None:
        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")

        prior_note = "(5)\\*-p<0.05 \\*\\*-p<0.01"
        title = "2.6.7.9 (1)\u9057\u4f20\u6bd2\u6027\uff1a\u4f53\u5185 \u62a5\u544a\u6807\u9898\uff1a \u4f9b\u8bd5\u54c1\uff1a(2)"
        field = "\u6240\u68c0\u6d4b\u7684\u8bf1\u5bfc\u4f5c\u7528\uff1a \u7ed9\u836f\u65b9\u6848\uff1a \u8bd5\u9a8c\u7f16\u53f7\uff1a"
        matrix_label = "\u4f9b\u8bd5\u54c1 \u5242\u91cf(mg/kg) \u52a8\u7269\u6570\u91cf"
        remark = "\uff081\uff09 \u5e94\u5bf9\u8868\u683c\u8fdb\u884c\u8fde\u7eed\u7f16\u53f7\uff08\u4f8b\u5982\uff1a2.6.7.9A\u30012.6.7.9B\uff09\u3002"

        prior_index = markdown.find(prior_note)
        title_index = markdown.find(title)
        field_index = markdown.find(field, title_index)
        matrix_index = markdown.find(matrix_label, title_index)
        remark_index = markdown.find(remark, title_index)

        self.assertGreaterEqual(prior_index, 0)
        self.assertGreater(title_index, prior_index)
        self.assertGreater(field_index, title_index)
        self.assertGreater(matrix_index, field_index)
        self.assertGreater(remark_index, matrix_index)
        self.assertNotIn("\u6a21\u677f\u77e9\u9635", markdown)
        self.assertNotIn("template matrix", markdown.lower())

    def test_page60_carcinogenicity_template_owns_matrix_tail_and_repeated_sex_columns(self) -> None:
        page60_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 60
            and str(template.get("template_profile") or "").startswith("blank_study_summary_template")
        ]
        carcinogenicity = next(
            template
            for template in page60_templates
            if "2.6.7.10" in str(template.get("title") or "")
        )
        carcinogenicity_rows = [
            str(row or "")
            for row in carcinogenicity.get("row_texts", []) or []
        ]
        carcinogenicity_text = "\n".join(carcinogenicity_rows)

        self.assertIn("\u9ad8\u5242\u91cf\u9009\u62e9\u4f9d\u636e\uff1a(3)", carcinogenicity_text)
        self.assertIn("\u65e5\u5242\u91cf(mg/kg)", carcinogenicity_text)
        self.assertIn("0(\u5bf9\u7167)", carcinogenicity_text)
        self.assertIn("\u6bd2\u4ee3\u52a8\u529b\u5b66\uff1aAUC()(4)", carcinogenicity_text)
        self.assertIn("\u52a8\u7269\u6570\u91cf", carcinogenicity_text)
        self.assertIn("\u5b58\u6d3b\u7387(%)", carcinogenicity_text)
        self.assertTrue(
            any(row == "\u6027\u522b M F M F M F M F" for row in carcinogenicity_rows),
            msg=carcinogenicity_rows,
        )

        stale_previous_continuations = [
            template
            for template in page60_templates
            if str(template.get("template_profile") or "") == "blank_study_summary_template_continuation"
            and "2.6.7.9" in str(template.get("title") or "")
            and any("\u65e5\u5242\u91cf(mg/kg)" in str(row or "") for row in template.get("row_texts", []) or [])
        ]
        self.assertEqual(
            stale_previous_continuations,
            [],
            msg="page 60 matrix tail should belong to the local 2.6.7.10 template, not previous 2.6.7.9",
        )

        page59_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 59
            and "2.6.7.9" in str(template.get("title") or "")
        )
        self.assertTrue(
            any(
                "(3)*-p<0.05" in str(note.get("text") or "")
                and str(note.get("relation") or "") == "cross_page_blank_template_note_continuation"
                for note in page59_template.get("note_blocks", []) or []
            )
        )

    def test_page49_toxicology_overview_heading_starts_local_template_region(self) -> None:
        title = "2.6.7.1 \u6bd2\u7406\u5b66\u6982\u8ff0 \u4f9b\u8bd5\u54c1\uff1a\uff081\uff09"
        page49_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 49
        ]
        self.assertEqual(len(page49_templates), 1, msg=page49_templates)
        template = page49_templates[0]

        self.assertEqual(template.get("title"), title)
        self.assertFalse(template.get("continued_from_structure_template_id"))
        self.assertNotIn(title, [str(row or "") for row in template.get("row_texts", []) or []])
        self.assertFalse(
            any(title in str(section.get("text") or "") for section in template.get("sections", []) or []),
            msg=template.get("sections", []),
        )
        self.assertIn("\u4f4d\u7f6e", [str(row or "") for row in template.get("row_texts", []) or []])

        page49 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 49)
        title_blocks = [
            block
            for block in page49.get("blocks", []) or []
            if block.get("block_type") == "text"
            and title in str(block.get("text") or "")
        ]
        self.assertEqual(len(title_blocks), 1, msg=title_blocks)
        self.assertEqual(title_blocks[0].get("semantic_role"), "structure_template_title")

    def test_page38_pregnant_lactating_template_owns_lower_ruled_template_region(self) -> None:
        page38_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 38
        ]
        stale_previous_continuations = [
            template
            for template in page38_templates
            if str(template.get("continued_from_structure_template_id") or "")
        ]
        self.assertEqual(
            stale_previous_continuations,
            [],
            msg="page 38 lower ruled template region belongs to local 2.6.5.7, not previous 2.6.5.6",
        )

        local_template = next(
            template
            for template in page38_templates
            if "2.6.5.7" in str(template.get("title") or "")
            or any("2.6.5.7" in str(row or "") for row in template.get("row_texts", []) or [])
        )
        rows = [str(row or "") for row in local_template.get("row_texts", []) or []]
        self.assertIn("\u4e73\u6c41\u6392\u6cc4", rows)
        self.assertIn("\u6d53\u5ea6\uff1a", rows)
        self.assertIn("\u65b0\u751f\u80ce\u4ed4\uff1a", rows)
        self.assertEqual(rows.count("\u6d53\u5ea6/\u91cf(%\u5242\u91cf)"), 1)
        self.assertEqual(rows.count("\u80ce\u4ed4(3)\uff1a"), 1)
        note_text = "\n".join(str(note.get("text") or "") for note in local_template.get("note_blocks", []) or [])
        self.assertIn("\u9644\u52a0\u4fe1\u606f\uff1a", note_text)
        self.assertTrue(
            any(
                str(note.get("relation") or "") == "terminal_template_additional_info"
                and "\u9644\u52a0\u4fe1\u606f\uff1a" in str(note.get("text") or "")
                for note in local_template.get("note_blocks", []) or []
            ),
            msg=local_template.get("note_blocks", []),
        )

        page38 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 38)
        plain_text = [
            str(block.get("text") or "")
            for block in page38.get("blocks", []) or []
            if block.get("block_type") == "text"
            and block.get("semantic_role") == "text_block"
        ]
        self.assertNotIn("\u65b0\u751f\u80ce\u4ed4\uff1a", plain_text)

    def test_page62_top_result_tail_and_significance_note_continue_page61_template(self) -> None:
        page61_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 61
            and "2.6.7.10" in str(template.get("title") or "")
        )
        page61_notes = [str(note.get("text") or "") for note in page61_template.get("note_blocks", []) or []]
        self.assertTrue(
            any("-\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c\u3002" in text for text in page61_notes),
            msg=page61_notes,
        )
        self.assertTrue(
            any("*-p<0.05 **-p<0.01" in text for text in page61_notes),
            msg=page61_notes,
        )

        page62 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 62)
        plain_top_text = [
            str(block.get("text") or "")
            for block in page62.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("semantic_role") or "") == "text_block"
        ]
        self.assertFalse(
            any("-\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c\u3002" in text for text in plain_top_text),
            msg=plain_top_text,
        )
        self.assertFalse(
            any("*-p<0.05 **-p<0.01" in text for text in plain_top_text),
            msg=plain_top_text,
        )

        instruction_notes = [
            block
            for block in page62.get("blocks", []) or []
            if block.get("block_type") == "text"
            and block.get("semantic_role") == "template_instruction_note"
        ]
        self.assertEqual(len(instruction_notes), 1)
        instruction_text = instruction_notes[0].get("text", "")
        self.assertIn("\u88682.6.7.10 \u7684\u6ce8\u91ca", instruction_text)
        self.assertIn("\uff087\uff09 \u5e94\u9996\u5148\u5217\u51fa\u836f\u7269\u76f8\u5173\u75c5\u53d8", instruction_text)

    def test_pages63_73_74_local_template_notes_attach_to_current_template(self) -> None:
        expectations = [
            (
                63,
                [
                    "\u6ce8\u91ca\uff1a \uff081\uff09\u5e94\u6309\u7167\u4e0eCTD \u76f8\u540c\u7684\u987a\u5e8f",
                    "\u5168\u6027\u8bd5\u9a8c\u300b\uff081997 \u5e7411 \u6708\uff09\u89c4\u5b9a",
                    "\uff082\uff09\u56fd\u9645\u975e\u4e13\u5229\u836f\u54c1\u540d\u79f0\uff08INN\uff09\u3002",
                ],
            ),
            (
                73,
                [
                    "\u6ce8\u91ca\uff1a(1)\u5e94\u5bf9\u6240\u6709\u7684\u5c40\u90e8\u8010\u53d7\u6027\u8bd5\u9a8c\u8fdb\u884c\u603b\u7ed3\u3002",
                    "(2)\u56fd\u9645\u975e\u4e13\u5229\u836f\u54c1\u540d\u79f0(INN)\u3002",
                ],
            ),
            (
                74,
                [
                    "\u5907\u6ce8\uff1a(1)\u5e94\u5bf9\u6240\u6709\u589e\u8865\u7684\u6bd2\u6027\u8bd5\u9a8c\u8fdb\u884c\u603b\u7ed3\u3002",
                    "(2)\u56fd\u9645\u975e\u4e13\u5229\u836f\u54c1\u540d\u79f0(INN)\u3002",
                ],
            ),
        ]

        for page_number, expected_note_fragments in expectations:
            with self.subTest(page=page_number):
                page_templates = [
                    template
                    for template in self.result.get("structure_templates", []) or []
                    if int(template.get("page", 0) or 0) == page_number
                ]
                self.assertEqual(len(page_templates), 1, msg=page_templates)
                template = page_templates[0]
                note_text = "\n".join(str(note.get("text") or "") for note in template.get("note_blocks", []) or [])
                for fragment in expected_note_fragments:
                    self.assertIn(fragment, note_text)

                page = next(page for page in self.result["document_ast"]["pages"] if page["page"] == page_number)
                plain_text = "\n".join(
                    str(block.get("text") or "")
                    for block in page.get("blocks", []) or []
                    if block.get("block_type") == "text"
                    and block.get("semantic_role") == "text_block"
                )
                for fragment in expected_note_fragments:
                    self.assertNotIn(fragment, plain_text)

    def test_pages68_72_new_template_titles_stop_previous_continuation_ownership(self) -> None:
        expectations = [
            (68, "2.6.7.14 (1)\u751f\u6b96\u6bd2\u6027- \u62a5\u544a\u6807\u9898\uff1a \u4f9b\u8bd5\u54c1\uff1a(2)"),
            (72, "2.6.7.16 \u5c40\u90e8\u8010\u53d7\u6027(1) \u4f9b\u8bd5\u54c1\uff1a(2)"),
        ]

        for page_number, title_text in expectations:
            with self.subTest(page=page_number):
                page_templates = [
                    template
                    for template in self.result.get("structure_templates", []) or []
                    if int(template.get("page", 0) or 0) == page_number
                ]
                self.assertEqual(len(page_templates), 1, msg=page_templates)
                continuation = page_templates[0]
                self.assertEqual(
                    continuation.get("template_profile"),
                    "blank_study_summary_template_continuation",
                )
                self.assertFalse(
                    any(title_text in str(row or "") for row in continuation.get("row_texts", []) or []),
                    msg=continuation.get("row_texts", []),
                )

                page = next(page for page in self.result["document_ast"]["pages"] if page["page"] == page_number)
                title_blocks = [
                    block
                    for block in page.get("blocks", []) or []
                    if block.get("block_type") == "text"
                    and title_text in str(block.get("text") or "")
                ]
                self.assertEqual(len(title_blocks), 1, msg=title_blocks)
                self.assertNotEqual(title_blocks[0].get("semantic_role"), "structure_template_entry")

    def test_page54_repeated_dose_template_title_splits_from_previous_note_run(self) -> None:
        title_text = "2.6.7.7 (1)\u91cd\u590d\u7ed9\u836f\u6bd2\u6027(2) \u62a5\u544a\u6807\u9898\uff1a \u4f9b\u8bd5\u54c1\uff1a(3)"
        previous_note_fragment = "\u5e94\u6309\u7167\u4e0eCTD \u76f8\u540c\u7684\u987a\u5e8f\u6765\u603b\u7ed3\u8bf4\u660e\u6240\u6709\u7684\u91cd\u590d\u7ed9\u836f\u6bd2\u6027\u8bd5\u9a8c"

        page54_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 54
        ]
        previous_section_continuation = next(
            (
                template
                for template in page54_templates
                if "2.6.7.6 \u91cd\u590d\u7ed9\u836f\u6bd2\u6027 \u975e\u5173\u952e\u8bd5\u9a8c" in str(template.get("title") or "")
            ),
            None,
        )
        self.assertIsNotNone(previous_section_continuation, msg=page54_templates)
        previous_rows = "\n".join(str(row or "") for row in previous_section_continuation.get("row_texts", []) or [])
        self.assertIn("\u79cd\u5c5e/\u54c1\u7cfb", previous_rows)
        self.assertIn("\u7ed9\u836f\u65b9\u6cd5(\u6eb6", previous_rows)
        self.assertIn("NOAELa", previous_rows)
        self.assertNotIn(title_text, previous_rows)

        repeated_dose_template = next(
            (
                template
                for template in page54_templates
                if title_text in str(template.get("title") or "")
            ),
            None,
        )
        self.assertIsNotNone(repeated_dose_template, msg=page54_templates)

        rows = "\n".join(str(row or "") for row in repeated_dose_template.get("row_texts", []) or [])
        self.assertIn("\u79cd\u5c5e/\u54c1\u7cfb\uff1a", rows)
        self.assertIn("CTD \u4e2d\u7684\u4f4d\u7f6e\uff1a\u5377\u3001\u9875\u7801", rows)
        self.assertNotIn(previous_note_fragment, rows)
        self.assertFalse(
            any(
                title_text in str(row or "")
                for template in page54_templates
                if template is not repeated_dose_template
                for row in template.get("row_texts", []) or []
            ),
            msg=page54_templates,
        )

        page54 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 54)
        title_blocks = [
            block
            for block in page54.get("blocks", []) or []
            if block.get("block_type") == "text"
            and title_text in str(block.get("text") or "")
        ]
        self.assertEqual(len(title_blocks), 1, msg=title_blocks)
        self.assertEqual(title_blocks[0].get("semantic_role"), "structure_template_title")

        plain_texts = [
            str(block.get("text") or "")
            for block in page54.get("blocks", []) or []
            if block.get("block_type") == "text"
            and block.get("semantic_role") == "text_block"
        ]
        self.assertNotIn("\u79cd\u5c5e/\u54c1\u7cfb", plain_texts)
        self.assertNotIn("\u7ed9\u836f\u65b9\u6cd5(\u6eb6", plain_texts)

    def test_template_notes_do_not_absorb_standalone_continuation_marker(self) -> None:
        page64_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 64
        )
        note_texts = [str(note.get("text") or "") for note in page64_template.get("note_blocks", []) or []]
        self.assertFalse(any(text == "(\u7eed)" for text in note_texts), msg=note_texts)

        page64 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 64)
        continuation_marker = [
            block
            for block in page64.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("text") or "") == "(\u7eed)"
        ]
        self.assertEqual(len(continuation_marker), 1, msg=continuation_marker)
        self.assertNotEqual(continuation_marker[0].get("semantic_role"), "structure_template_note")

    def test_pages65_69_adjacent_matrix_headers_attach_to_template_composite(self) -> None:
        page65_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 65
        )
        page65_rows = [str(row or "") for row in page65_template.get("row_texts", []) or []]
        self.assertIn("\u65e5\u5242\u91cf(mg/kg)", page65_rows)
        self.assertIn("0(\u5bf9\u7167)", page65_rows)

        page65 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 65)
        page65_plain = [
            str(block.get("text") or "")
            for block in page65.get("blocks", []) or []
            if block.get("block_type") == "text"
            and block.get("semantic_role") == "text_block"
        ]
        self.assertNotIn("\u65e5\u5242\u91cf(mg/kg)", page65_plain)
        self.assertNotIn("0(\u5bf9\u7167)", page65_plain)

        page69_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 69
        )
        page69_rows = [str(row or "") for row in page69_template.get("row_texts", []) or []]
        self.assertIn("F1\u4ee3\u5e7c\u4ed4\uff1a\u8bc4\u4ef7\u7684\u7a9d\u6570", page69_rows)

        page69 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 69)
        page69_plain = [
            str(block.get("text") or "")
            for block in page69.get("blocks", []) or []
            if block.get("block_type") == "text"
            and block.get("semantic_role") == "text_block"
        ]
        self.assertNotIn("F1\u4ee3\u5e7c\u4ed4\uff1a\u8bc4\u4ef7\u7684\u7a9d\u6570", page69_plain)
        self.assertNotIn("0(\u5bf9\u7167)", page69_plain)

    def test_page66_multi_table_instruction_notes_release_to_body_not_business_table(self) -> None:
        page66_business_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 66
            and str(table.get("semantic_role") or "business_table") == "business_table"
        ]
        self.assertEqual(
            page66_business_tables,
            [],
            msg="multi-table numbered instruction notes should not pollute business_table ownership",
        )

        page66 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 66)
        instruction_notes = [
            block
            for block in page66.get("blocks", []) or []
            if block.get("block_type") == "text"
            and block.get("semantic_role") == "template_instruction_note"
        ]
        self.assertEqual(len(instruction_notes), 1)
        note_text = instruction_notes[0].get("text", "")
        self.assertIn("\u88682.6.7.12\u30012.6.7.13 \u548c2.6.7.14 \u7684\u6ce8\u91ca", note_text)
        self.assertIn("\uff081\uff09 \u5982\u6709\u591a\u9879\u6b64\u7c7b\u8bd5\u9a8c", note_text)
        self.assertIn("\uff088\uff09\u5e94\u6307\u660e\u4ea4\u914d\u65e5\u671f", note_text)

    def test_page17_page18_module4_no_page_outline_template_renders_in_ind_review(self) -> None:
        page17_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 17
        )
        page18_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 18
        )
        self.assertEqual(page17_template.get("template_kind"), "no_page_outline")
        self.assertEqual(page18_template.get("continued_from_structure_template_id"), page17_template.get("structure_template_id"))

        page17_entries = {str(entry.get("outline_index") or ""): entry for entry in page17_template.get("entries", []) or []}
        page18_entries = {str(entry.get("outline_index") or ""): entry for entry in page18_template.get("entries", []) or []}
        self.assertIn("4.2.1", page17_entries)
        self.assertIn("4.2.3.2", page17_entries)
        self.assertIn("4.2.3.7.7", page18_entries)
        self.assertIn("4.3", page18_entries)
        self.assertEqual(page18_entries["4.3"].get("title"), "\u53c2\u8003\u6587\u732e")
        self.assertIn(
            "\u62ec\u4f34\u968f\u6bd2\u4ee3\u52a8\u529b\u5b66\u8bd5\u9a8c",
            str(page17_entries["4.2.3.2"].get("raw_text") or ""),
        )

        page18 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 18)
        page18_template_node = next(
            block for block in page18.get("blocks", []) or []
            if block.get("block_type") == "structure_template"
        )
        self.assertGreaterEqual(len(page18_template_node.get("entries", []) or []), 20)

        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        self.assertIn("#### \u8bd5\u9a8c\u62a5\u544a", markdown)
        self.assertIn("#### \u8bd5\u9a8c\u62a5\u544a\uff08\u7eed\uff09", markdown)
        self.assertNotIn("\u7ed3\u6784\u6a21\u677f", markdown)
        self.assertNotIn("\u8868\u5355\u6a21\u677f", markdown)
        self.assertNotIn("#### \u7ed3\u6784\u6a21\u677f\n\n      - 4.2.3.3.1", markdown)
        self.assertIn("- 4.2.1 \u836f\u7406\u5b66", markdown)
        self.assertIn("  - 4.2.1.1 \u4e3b\u8981\u836f\u6548\u5b66", markdown)
        self.assertIn("  - 4.2.3.7.7 \u5176\u4ed6\u8bd5\u9a8c", markdown)
        self.assertIn("- 4.3 \u53c2\u8003\u6587\u732e", markdown)

    def test_page13_sparse_ruled_table_rebuilds_visual_rows_from_word_geometry(self) -> None:
        page13_tables = [
            table
            for table in self.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 13
        ]
        self.assertEqual(len(page13_tables), 1)

        table = page13_tables[0]
        grid = table.get("display_grid", []) or table.get("raw_grid", [])
        self.assertGreaterEqual(len(grid), 8)
        self.assertEqual(int(table.get("col_count", 0) or 0), 4)

        expected_header = [
            "试验类型和给药期限",
            "给药途径",
            "动物种属",
            "给予的化合物*",
        ]
        self.assertEqual(grid[0], expected_header)
        self.assertEqual([cell["text"] for cell in table.get("header", [])], expected_header)
        self.assertEqual(table.get("header_row_index"), 0)
        self.assertEqual(table.get("data_start_row"), 1)

        note_blocks = table.get("note_blocks", [])
        self.assertEqual(len(note_blocks), 1)
        self.assertEqual(note_blocks[0].get("text"), "*仅在对代谢产物进行研究时才需要这一列。")
        header_note_refs = table.get("header_note_refs", [])
        self.assertTrue(
            any(
                ref.get("marker") == "*"
                and ref.get("header_row") == 0
                and ref.get("header_col") == 3
                and ref.get("header_text") == "给予的化合物*"
                and ref.get("note_text") == "*仅在对代谢产物进行研究时才需要这一列。"
                for ref in header_note_refs
            ),
            msg=f"missing header-note ref in {header_note_refs!r}",
        )
        composite = table.get("composite_object") or {}
        self.assertEqual(composite.get("object_family"), "business_table")
        self.assertEqual(composite.get("ownership_domain"), "table")
        self.assertEqual(composite.get("reading_flow"), "composite_object")
        self.assertEqual(composite.get("body_flow_policy"), "exclude_owned_text_from_body")
        self.assertEqual(composite.get("presentation_order"), ["title", "body", "notes"])
        self.assertEqual(composite.get("title_policy"), "owned_object_title")
        self.assertEqual(composite.get("note_policy"), "owned_object_notes")
        self.assertTrue(composite.get("has_body"))
        self.assertTrue(composite.get("has_notes"))
        self.assertGreaterEqual(int(composite.get("body_component_count", 0) or 0), 1)
        self.assertEqual(int(composite.get("note_block_count", 0) or 0), 1)
        self.assertIn("body", composite.get("component_order", []) or [])
        self.assertIn("notes", composite.get("component_order", []) or [])
        self.assertTrue(
            any(
                ref.get("anchor_status") == "explicit_marker_anchor"
                and ref.get("anchor_type") == "table_header"
                and ref.get("note_profile", {}).get("profile_type") == "symbol_note"
                for ref in composite.get("note_anchor_refs", []) or []
            ),
            msg=f"missing table header note anchor in {composite!r}",
        )
        table_evidence = next(
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "table"
            and evidence.get("source_id") == table.get("table_id")
        )
        evidence_composite = table_evidence.get("composite_object") or {}
        self.assertEqual(evidence_composite.get("object_family"), "business_table")
        self.assertEqual(evidence_composite.get("ownership_domain"), "table")
        self.assertEqual(evidence_composite.get("reading_flow"), "composite_object")
        self.assertEqual(evidence_composite.get("presentation_order"), ["title", "body", "notes"])

        self.assertEqual(
            grid[2],
            ["单次给药毒性", "经口和静脉注射", "大鼠和小鼠", "代谢产物X"],
        )
        self.assertEqual(
            grid[4],
            ["1 个月", "经口", "大鼠和犬", "原形药物"],
        )
        self.assertEqual(
            grid[6],
            ["9 个月", "经口", "犬", "“”"],
        )
        self.assertNotIn("经口和静脉注射 大鼠和小鼠", " | ".join(str(cell or "") for row in grid for cell in row))


    def test_page20_caption_anchored_horizontal_rule_table_is_recovered(self) -> None:
        page20_tables = [
            table
            for table in self.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 20
        ]
        self.assertEqual(len(page20_tables), 1)

        table = page20_tables[0]
        self.assertEqual(table.get("semantic_role"), "business_table")
        self.assertEqual(int(table.get("col_count", 0) or 0), 5)
        self.assertEqual(table.get("detection_method"), "caption_anchored_horizontal_rules")
        expected_title = "表X X及其主要代谢物和对照药物与人X2和X3受体的结合率"
        self.assertEqual(table.get("title"), expected_title)
        self.assertEqual(table.get("title_block", {}).get("source"), "caption_anchored_horizontal_rules")
        self.assertEqual(table.get("title_block", {}).get("text"), expected_title)

        expected_header = [
            "化合物",
            "X2 Ki1(nM)",
            "X2 Ki2(nM)",
            "X3 Ki1(nM)",
            "X3 Ki2(nM)",
        ]
        self.assertEqual([cell["text"] for cell in table.get("header", [])], expected_header)
        self.assertEqual(table.get("header_row_index"), 0)
        self.assertEqual(table.get("data_start_row"), 2)

        grid = table.get("display_grid", []) or table.get("raw_grid", [])
        self.assertGreaterEqual(len(grid), 9)
        self.assertEqual(grid[0], ["化合物", "X2", "X2", "X3", "X3"])
        self.assertEqual(grid[1], [None, "Ki1(nM)", "Ki2(nM)", "Ki1(nM)", "Ki2(nM)"])
        self.assertIn(["1", "538", "2730", "691", "4550"], grid)
        self.assertIn(["7", "3.11", "3.76", "1.94", "1.93"], grid)

        table_evidence = next(
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "table"
            and evidence.get("source_id") == table.get("table_id")
        )
        self.assertEqual(table_evidence.get("title"), expected_title)
        self.assertEqual(table_evidence.get("segments", [])[0].get("role"), "title")
        self.assertEqual(table_evidence.get("segments", [])[0].get("text"), expected_title)

        note_texts = [
            str(note.get("text", ""))
            for note in table.get("note_blocks", [])
        ]
        self.assertEqual(len(note_texts), 1)
        self.assertIn("Ki1 和Ki2 分别代表高亲和力和低亲和力的结合位点", note_texts[0])
        self.assertIn("试验编号", note_texts[0])

    def test_page22_adjacent_caption_anchored_tables_are_split_and_notes_owned(self) -> None:
        page22_tables = [
            table
            for table in self.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 22
        ]
        self.assertEqual(len(page22_tables), 2)

        first, second = sorted(page22_tables, key=lambda table: table.get("bbox", [0, 0, 0, 0])[1])
        self.assertEqual(first.get("detection_method"), "caption_anchored_horizontal_rules")
        self.assertEqual(second.get("detection_method"), "caption_anchored_horizontal_rules")

        expected_first_title_compact = "表X：小鼠单次经口给予X2、10和30mg/kg后的非房室模型药代动力学参数[参考]"
        self.assertEqual(str(first.get("title", "")).replace(" ", ""), expected_first_title_compact)
        self.assertEqual(
            str(first.get("title_block", {}).get("text", "")).replace(" ", ""),
            expected_first_title_compact,
        )
        self.assertIn("[14C]X", second.get("title", ""))
        first_grid_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in first.get("display_grid", []))
        self.assertNotIn("[14C]X", first_grid_text)
        self.assertNotIn("i.v.", first_grid_text)
        first_notes = "\n".join(str(note.get("text", "")) for note in first.get("note_blocks", []))
        self.assertIn("药代动力学参数由每时间点3只动物的混合血浆检测计算得到", first_notes.replace(" ", ""))

        first_grid = first.get("display_grid", []) or first.get("raw_grid", [])
        self.assertEqual(int(first.get("col_count", 0) or 0), 7)
        self.assertEqual(first_grid[0], ["参数(单位)", "参数值", None, None, None, None, None])
        self.assertEqual(first_grid[1], ["性别", "雄性", None, None, "雌性", None, None])
        self.assertIn(["剂量(mg/kg)", "2", "10", "30", "2", "10", "30"], first_grid)
        self.assertIn(["Cmax(ng/ml )", "4.9", "20.4", "30.7", "5.5", "12.9", "28.6"], first_grid)
        self.assertIn(["AUC0-t(ng·h/ml)", "21.6", "80.5", "267", "33.3", "80", "298"], first_grid)
        self.assertIn(["AUC0-inf(ng·h/ml)", "28.3", "112", "297", "40.2", "90", "327"], first_grid)
        first_grid_text_compact = "".join(str(cell or "") for row in first_grid for cell in row)
        self.assertNotIn("剂量210", first_grid_text_compact)
        self.assertFalse(any(row and row[0] == "(mg/kg)" for row in first_grid), msg=first_grid)
        header_groups = first.get("header_column_groups", [])
        self.assertTrue(
            any(group.get("text") == "参数值" and group.get("start_col") == 1 and group.get("end_col") == 6 for group in header_groups),
            msg=f"missing parameter-value span in {header_groups!r}",
        )
        self.assertTrue(
            any(group.get("text") == "雄性" and group.get("start_col") == 1 and group.get("end_col") == 3 for group in header_groups),
            msg=f"missing male span in {header_groups!r}",
        )
        self.assertTrue(
            any(group.get("text") == "雌性" and group.get("start_col") == 4 and group.get("end_col") == 6 for group in header_groups),
            msg=f"missing female span in {header_groups!r}",
        )

        second_grid = second.get("display_grid", []) or second.get("raw_grid", [])
        self.assertGreaterEqual(len(second_grid), 4)
        self.assertEqual(int(second.get("col_count", 0) or 0), 5)
        self.assertEqual(
            [cell["text"] for cell in second.get("header", [])],
            ["剂量(mg/kg)", "给药途径", "尿液*", "粪便", "合计+"],
        )
        self.assertEqual(second_grid[0], ["剂量(mg/kg)", "给药途径", "给药剂量的百分比", None, None])
        self.assertEqual(second_grid[1], [None, None, "尿液*", "粪便", "合计+"])
        second_header_groups = second.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "给药剂量的百分比"
                and group.get("start_col") == 2
                and group.get("end_col") == 4
                and group.get("colspan") == 3
                for group in second_header_groups
            ),
            msg=f"missing dose-percentage span in {second_header_groups!r}",
        )
        second_grid_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in second_grid)
        self.assertIn("mg/kg", second_grid_text)
        self.assertIn("i.v.", second_grid_text)
        self.assertIn("p.o.", second_grid_text)
        self.assertIn("88.1\u00b17.4", second_grid_text)
        self.assertIn("95.3\u00b13.4", second_grid_text)

        second_notes = "\n".join(str(note.get("text", "")) for note in second.get("note_blocks", []))
        self.assertIn("168", second_notes)
        self.assertIn("S.D.", second_notes)
        self.assertIn("22.1%", second_notes)
        self.assertIn("21.7%", second_notes)
        self.assertNotIn("22.1%", first_notes)
        header_note_refs = second.get("header_note_refs", [])
        self.assertTrue(
            any(
                ref.get("marker") == "*"
                and ref.get("header_text") == "尿液*"
                and "22.1%" in str(ref.get("note_text", ""))
                and "21.7%" in str(ref.get("note_text", ""))
                for ref in header_note_refs
            ),
            msg=f"missing urine marker ref in {header_note_refs!r}",
        )
        self.assertTrue(
            any(
                ref.get("marker") == "+"
                and ref.get("header_text") == "合计+"
                and "尸体" in str(ref.get("note_text", ""))
                for ref in header_note_refs
            ),
            msg=f"missing total marker ref in {header_note_refs!r}",
        )


    def test_page23_concentration_table_projects_colspan_and_rowspan_headers(self) -> None:
        page23_tables = [
            table
            for table in self.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 23
        ]
        self.assertEqual(len(page23_tables), 1)

        table = page23_tables[0]
        self.assertEqual(table.get("detection_method"), "caption_anchored_horizontal_rules")
        self.assertEqual(int(table.get("col_count", 0) or 0), 6)

        header = [cell["text"] for cell in table.get("header", [])]
        self.assertEqual(header[1:], ["1 h", "6 h", "24 h", "48 h", "72 h"])
        self.assertNotEqual(header[0], "Column 1")

        display_grid = table.get("display_grid", []) or table.get("raw_grid", [])
        self.assertGreaterEqual(len(display_grid), 3)
        self.assertIsNotNone(display_grid[0][0])
        self.assertEqual(display_grid[1][1:], ["1 h", "6 h", "24 h", "48 h", "72 h"])
        display_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in display_grid[:2])
        self.assertIn("ng", display_text)
        self.assertIn("*/g", display_text)

        header_groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                "ng" in str(group.get("text", ""))
                and "*/g" in str(group.get("text", ""))
                and group.get("start_col") == 1
                and group.get("end_col") == 5
                and group.get("colspan") == 5
                for group in header_groups
            ),
            msg=f"missing concentration span in {header_groups!r}",
        )

        row_groups = table.get("header_row_groups", [])
        self.assertTrue(
            any(
                group.get("col") == 0
                and group.get("start_row") == 0
                and group.get("end_row") == 1
                and group.get("rowspan") == 2
                for group in row_groups
            ),
            msg=f"missing first-column rowspan header in {row_groups!r}",
        )

        header_note_refs = table.get("header_note_refs", [])
        self.assertTrue(
            any(
                ref.get("marker") == "*"
                and "浓度" in str(ref.get("header_text", ""))
                and "ng X 游离碱当量/g" in str(ref.get("note_text", ""))
                for ref in header_note_refs
            ),
            msg=f"missing concentration marker ref in {header_note_refs!r}",
        )


    def test_page24_excretion_table_projects_split_slash_header_and_percentage_span(self) -> None:
        page24_tables = [
            table
            for table in self.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 24
        ]
        self.assertEqual(len(page24_tables), 1)

        table = page24_tables[0]
        self.assertEqual(table.get("detection_method"), "caption_anchored_horizontal_rules")
        self.assertEqual(int(table.get("col_count", 0) or 0), 6)

        header = [cell["text"] for cell in table.get("header", [])]
        self.assertEqual(header, ["剂量(mg/kg)", "给药途径", "尿液", "粪便", "胆汁", "合计"])

        display_grid = table.get("display_grid", []) or table.get("raw_grid", [])
        self.assertGreaterEqual(len(display_grid), 3)
        self.assertEqual(display_grid[0][:3], ["剂量(mg/kg)", "给药途径", "给药剂量的百分比"])
        self.assertEqual(display_grid[1][2:], ["尿液", "粪便", "胆汁", "合计"])

        header_groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "给药剂量的百分比"
                and group.get("start_col") == 2
                and group.get("end_col") == 5
                and group.get("colspan") == 4
                for group in header_groups
            ),
            msg=f"missing percentage span in {header_groups!r}",
        )

    def test_page25_exposure_table_projects_header_and_note_semantics(self) -> None:
        page25_tables = [
            table
            for table in self.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 25
        ]
        self.assertEqual(len(page25_tables), 1)

        table = page25_tables[0]
        self.assertEqual(table.get("detection_method"), "caption_anchored_horizontal_rules")
        self.assertEqual(int(table.get("col_count", 0) or 0), 5)

        header = [cell["text"] for cell in table.get("header", [])]
        self.assertEqual(header, ["种系(剂型)", "剂量(mg/kg/天)", "Cmax(ng/ml)", "AUC(ng·h/ml)#", "参考"])

        display_grid = table.get("display_grid", []) or table.get("raw_grid", [])
        self.assertGreaterEqual(len(display_grid), 5)
        self.assertEqual(display_grid[0], ["种系(剂型)", "剂量(mg/kg/天)", "系统(血浆)暴露量", None, "参考"])
        self.assertEqual(display_grid[1], [None, None, "Cmax(ng/ml)", "AUC(ng·h/ml)#", None])

        header_groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "系统(血浆)暴露量"
                and group.get("start_col") == 2
                and group.get("end_col") == 3
                and group.get("colspan") == 2
                for group in header_groups
            ),
            msg=f"missing exposure span in {header_groups!r}",
        )

        header_row_groups = table.get("header_row_groups", [])
        for expected_col, expected_text in ((0, "种系(剂型)"), (1, "剂量(mg/kg/天)"), (4, "参考")):
            self.assertTrue(
                any(
                    group.get("text") == expected_text
                    and group.get("col") == expected_col
                    and group.get("start_row") == 0
                    and group.get("end_row") == 1
                    and group.get("rowspan") == 2
                    for group in header_row_groups
                ),
                msg=f"missing rowspan header {expected_text!r} in {header_row_groups!r}",
            )

        row_groups = table.get("row_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "小鼠(溶液)"
                and group.get("col") == 0
                and group.get("start_data_row") == 2
                and group.get("end_data_row") == 4
                and group.get("rowspan") == 3
                for group in row_groups
            ),
            msg=f"missing mouse row group in {row_groups!r}",
        )
        self.assertTrue(
            any(
                group.get("text") == "犬(溶液)"
                and group.get("col") == 0
                and group.get("start_data_row") == 6
                and group.get("end_data_row") == 8
                and group.get("rowspan") == 3
                for group in row_groups
            ),
            msg=f"missing dog row group in {row_groups!r}",
        )

        note_text = " ".join(str(note.get("text", "")) for note in table.get("note_blocks", []))
        self.assertIn("显示的数据为", note_text)
        self.assertIn("#-小鼠为AUC0-6", note_text)
        self.assertIn("$-按照人体重", note_text)
        self.assertIn("*-括号中的数字表示", note_text)

        header_note_refs = table.get("header_note_refs", [])
        self.assertTrue(
            any(
                ref.get("marker") == "#"
                and ref.get("header_text") == "AUC(ng·h/ml)#"
                and "AUC0-6" in str(ref.get("note_text", ""))
                for ref in header_note_refs
            ),
            msg=f"missing AUC marker ref in {header_note_refs!r}",
        )
        cell_note_refs = table.get("cell_note_refs", [])
        self.assertTrue(
            any(
                ref.get("marker") == "$"
                and str(ref.get("cell_text", "")).endswith("$")
                and "人体重" in str(ref.get("note_text", ""))
                for ref in cell_note_refs
            ),
            msg=f"missing dose marker ref in {cell_note_refs!r}",
        )
        self.assertTrue(
            any(
                ref.get("marker") == "*"
                and str(ref.get("cell_text", "")).endswith("*")
                and "括号中的数字" in str(ref.get("note_text", ""))
                for ref in cell_note_refs
            ),
            msg=f"missing exposure ratio marker ref in {cell_note_refs!r}",
        )


    def test_page26_lesion_table_projects_header_span_and_marker_note(self) -> None:
        page26_tables = [
            table
            for table in self.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 26
        ]
        self.assertEqual(len(page26_tables), 1)

        table = page26_tables[0]
        self.assertEqual(int(table.get("col_count", 0) or 0), 5)

        header = [cell["text"] for cell in table.get("header", [])]
        self.assertEqual(header, ["损伤", "对照", "3 mg/kg", "30 mg/kg", "100 mg/kg"])

        display_grid = table.get("display_grid", []) or table.get("raw_grid", [])
        self.assertGreaterEqual(len(display_grid), 6)
        self.assertEqual(display_grid[0], ["损伤", "剂量组", None, None, None])
        self.assertEqual(display_grid[1], [None, "对照", "3 mg/kg", "30 mg/kg", "100 mg/kg"])
        self.assertIn(["合计*", "x/50(%)", "x/50(%)", "x/50(%)", "x/50(%)"], display_grid)
        self.assertFalse(
            any("合计* *腺瘤" in " ".join(str(cell or "") for cell in row) for row in display_grid),
            msg=display_grid,
        )

        header_groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "剂量组"
                and group.get("start_col") == 1
                and group.get("end_col") == 4
                and group.get("colspan") == 4
                for group in header_groups
            ),
            msg=f"missing dose-group span in {header_groups!r}",
        )

        header_row_groups = table.get("header_row_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "损伤"
                and group.get("col") == 0
                and group.get("start_row") == 0
                and group.get("end_row") == 1
                and group.get("rowspan") == 2
                for group in header_row_groups
            ),
            msg=f"missing lesion rowspan header in {header_row_groups!r}",
        )

        note_text = " ".join(str(note.get("text", "")) for note in table.get("note_blocks", []))
        self.assertIn("*腺瘤和/或增生", note_text)

        cell_note_refs = table.get("cell_note_refs", [])
        self.assertTrue(
            any(
                ref.get("marker") == "*"
                and ref.get("cell_text") == "合计*"
                and "腺瘤和/或增生" in str(ref.get("note_text", ""))
                for ref in cell_note_refs
            ),
            msg=f"missing total marker ref in {cell_note_refs!r}",
        )


    def test_page77_borderless_overview_table_extends_sparse_rows_and_links_marker_note(self) -> None:
        page77_tables = [
            table
            for table in self.result.get("table_asts", [])
            if int(table.get("page", 0) or 0) == 77
        ]
        self.assertEqual(len(page77_tables), 1)
        table = page77_tables[0]
        self.assertEqual(table.get("semantic_role"), "business_table")
        self.assertEqual(table.get("detection_method"), "word_clustering")
        self.assertGreaterEqual(int(table.get("col_count", 0) or 0), 7)

        grid = table.get("display_grid", []) or table.get("raw_grid", [])
        grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in grid
        )
        self.assertIn("试验类型", grid[0][0])
        self.assertIn("安全药理学", grid_text)
        self.assertIn("对中枢神经系统的影响a", grid_text)
        self.assertIn("95703", grid_text)
        self.assertIn("对心血管系统的影响", grid_text)
        self.assertIn("95706", grid_text)
        self.assertIn("药效学药物相互作用", grid_text)
        self.assertIn("与AZT 抗-HIV 作用的相互作用", grid_text)
        self.assertIn("95425", grid_text)

        note_text = " ".join(str(note.get("text", "")) for note in table.get("note_blocks", []))
        self.assertIn("a-报告中含GLP 依从性声明。", note_text)
        self.assertNotIn("次要药效学 95602 1 抗菌作用", note_text)

        cell_note_refs = table.get("cell_note_refs", [])
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and ref.get("cell_text") == "对中枢神经系统的影响a"
                and ref.get("note_text") == "a-报告中含GLP 依从性声明。"
                for ref in cell_note_refs
            ),
            msg=f"missing CNS marker ref in {cell_note_refs!r}",
        )

    def test_pages78_to_80_text_aligned_borderless_tables_are_recovered(self) -> None:
        tables_by_page = {
            page: [
                table
                for table in self.result.get("table_asts", [])
                if int(table.get("page", 0) or 0) == page
            ]
            for page in (78, 79, 80)
        }
        self.assertGreaterEqual(len(tables_by_page[78]), 1)
        self.assertGreaterEqual(len(tables_by_page[79]), 1)
        self.assertGreaterEqual(len(tables_by_page[80]), 1)

        page78 = max(tables_by_page[78], key=lambda item: int(item.get("col_count", 0) or 0))
        self.assertEqual(page78.get("detection_method"), "text_aligned_borderless_grid")
        self.assertEqual(int(page78.get("col_count", 0) or 0), 8)
        header78 = [cell.get("text") for cell in page78.get("header", [])]
        self.assertEqual(header78[0], "评价的器官系统")
        self.assertIn("剂量a(mg/kg)", header78)
        self.assertIn("GLP依从性", "".join(header78))
        grid78_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in page78.get("display_grid", []) or []
        )
        self.assertIn("CNS", grid78_text)
        self.assertIn("肾脏、GI、CNS 和 凝血功能", grid78_text)
        self.assertIn("心血管", grid78_text)
        self.assertIn("ECG 变化", grid78_text)
        self.assertIn("出量或总外周阻力没有影响", grid78_text)
        page78_ast = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 78)
        page78_body_text = "\n".join(
            str(block.get("text") or "")
            for block in page78_ast.get("blocks", []) or []
            if block.get("block_type") == "text"
        )
        self.assertNotIn("出量或总外周阻力没有影响", page78_body_text)
        note78 = " ".join(str(note.get("text", "")) for note in page78.get("note_blocks", []))
        self.assertIn("a-单次给药，除非另有说明。", note78.replace(" ", ""))
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and ref.get("header_text") == "剂量a(mg/kg)"
                and "单次给药" in str(ref.get("note_text", ""))
                for ref in page78.get("header_note_refs", [])
            ),
            msg=f"missing page78 dose marker ref in {page78.get('header_note_refs', [])!r}",
        )

        page79 = max(tables_by_page[79], key=lambda item: int(item.get("row_count", 0) or 0))
        self.assertEqual(page79.get("detection_method"), "text_aligned_borderless_grid")
        grid79_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in page79.get("display_grid", []) or []
        )
        for expected in ("试验类型", "吸收", "分布", "代谢", "排泄", "药代动力学药物相互作用", "与AZT 的相互作用a"):
            self.assertIn(expected, grid79_text)
        self.assertIn("93302", grid79_text)
        self.assertIn("94051", grid79_text)
        semantic_grid79 = page79.get("semantic_grid") or []
        self.assertGreaterEqual(len(semantic_grid79), 2)
        self.assertEqual(
            semantic_grid79[0],
            ["试验类型", "试验系统", "给药方法", "试验机构", "试验编号", "卷", "部分"],
        )
        self.assertNotIn("Column 5", semantic_grid79[0])
        first_study_row = next(
            row
            for row in semantic_grid79
            if len(row) >= 5 and row[0] == "吸收和排泄" and row[1] == "大鼠"
        )
        self.assertEqual(first_study_row[3], "Sponsor Inc.")
        self.assertEqual(first_study_row[4], "93302")
        self.assertFalse(
            any(
                len(row) >= 5 and row[3] == "Sponsor" and row[4] == "Inc."
                for row in semantic_grid79[1:]
            ),
            msg=semantic_grid79,
        )
        projection79 = (
            page79.get("semantic_projection_v2", {}) or {}
        ).get("overview_inventory_schema_projection", {})
        self.assertEqual(projection79.get("semantic_profile"), "pk_overview_inventory_table")
        self.assertEqual(projection79.get("logical_column_count"), 7)
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        self.assertIn("Sponsor Inc. | 93302", review_markdown)
        self.assertNotIn("Sponsor | Inc. | 93302", review_markdown)
        note79 = " ".join(str(note.get("text", "")) for note in page79.get("note_blocks", []))
        self.assertIn("a-报告中含GLP依从性性声明", note79.replace(" ", ""))
        self.assertNotIn("排泄灌胃", note79)
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and ref.get("cell_text") == "与AZT 的相互作用a"
                and "GLP" in str(ref.get("note_text", ""))
                for ref in page79.get("cell_note_refs", [])
            ),
            msg=f"missing page79 AZT marker ref in {page79.get('cell_note_refs', [])!r}",
        )

        page80 = max(tables_by_page[80], key=lambda item: int(item.get("row_count", 0) or 0))
        self.assertEqual(page80.get("detection_method"), "text_aligned_borderless_grid")
        self.assertEqual(int(page80.get("col_count", 0) or 0), 6)
        header80 = [cell.get("text") for cell in page80.get("header", [])]
        self.assertEqual(header80, ["物种", "小鼠", "大鼠", "犬", "猴", "人"])
        grid80_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in page80.get("display_grid", []) or []
        )
        for expected in ("性别(M/F)/动物数量", "分析物", "PK 参数：", "Tmax (h)", "Cmax(ng/ml 或ng-eq/ml)", "AUC(ng 或ng-eq × h/ml)", "T1/2(h)"):
            self.assertIn(expected, grid80_text)
        grid80 = page80.get("display_grid", []) or []
        self.assertGreaterEqual(len(grid80), 17)
        self.assertEqual(grid80[11][:6], ["Tmax (h)", "4.0", "1.0", "3.3", "1.0", "6.8"])
        self.assertEqual(
            grid80[12][:6],
            ["Cmax(ng/ml 或ng-eq/ml)", "2,260", "609", "172", "72", "8.2"],
        )
        self.assertEqual(
            grid80[13][:6],
            ["AUC(ng 或ng-eq × h/ml)", "15,201", "2,579", "1,923", "582", "135"],
        )
        self.assertEqual(
            grid80[14][:6],
            ["(计算时间-h)", "(0-72)", "(0-24)", "(0.5-48)", "(0-12)", "(0-24)"],
        )
        self.assertEqual(grid80[15][:6], ["T1/2(h)", "10.6", "3.3", "9.2", "3.2", "30.9"])
        self.assertEqual(
            grid80[16][:6],
            ["(计算时间-h)", "(7-48)", "(1-24)", "(24-96)", "(1-12)", "(24-120)"],
        )
        note80 = " ".join(str(note.get("text", "")) for note in page80.get("note_blocks", []))
        self.assertIn("附加信息：", note80)
        self.assertIn("a-总放射性，14C", note80)
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and ref.get("cell_text") == "TRAa"
                and "总放射性" in str(ref.get("note_text", ""))
                for ref in page80.get("cell_note_refs", [])
            ),
            msg=f"missing page80 TRA marker ref in {page80.get('cell_note_refs', [])!r}",
        )

    def test_page81_populated_study_metadata_stays_distinct_from_business_table(self) -> None:
        page81_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 81
            and table.get("semantic_role") == "business_table"
        ]
        self.assertEqual(len(page81_tables), 1)
        table = page81_tables[0]
        self.assertEqual(table.get("detection_method"), "text_aligned_borderless_grid")
        self.assertEqual(int(table.get("col_count", 0) or 0), 7)
        grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in table.get("display_grid", []) or []
        )
        for expected in ("组织/器官", "血液", "血浆", "脑", "肝脏", "nd"):
            self.assertIn(expected, grid_text)

        page80_tables = [
            prior
            for prior in self.result.get("table_asts", []) or []
            if int(prior.get("page", 0) or 0) == 80
            and prior.get("semantic_role") == "business_table"
        ]
        self.assertEqual(len(page80_tables), 1)
        page80_table = page80_tables[0]
        self.assertFalse(table.get("is_continuation"), msg=table)
        self.assertFalse(str(table.get("continued_from") or table.get("continued_from_table_id") or "").strip(), msg=table)
        self.assertNotIn(table.get("table_id"), page80_table.get("continued_to", []) or [])

        note_text = " ".join(str(note.get("text", "")) for note in table.get("note_blocks", []))
        self.assertIn("nd=未检出", note_text)

        page81_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 81
        ]
        self.assertEqual(len(page81_templates), 1)
        metadata = page81_templates[0]
        self.assertEqual(metadata.get("semantic_role"), "structure_template")
        self.assertEqual(metadata.get("template_kind"), "study_metadata")
        self.assertEqual(metadata.get("template_profile"), "populated_study_metadata")
        self.assertEqual(metadata.get("ownership_domain"), "study_context")
        self.assertEqual(metadata.get("data_population"), "populated")
        self.assertNotEqual(metadata.get("template_profile"), "low_text_ruled_tabular_form_template")

        fields = {
            str(field.get("label") or ""): str(field.get("value") or "")
            for field in metadata.get("fields", []) or []
        }
        self.assertEqual(fields.get("试验编号"), "95207")
        self.assertEqual(fields.get("物种"), "大鼠")
        self.assertEqual(fields.get("剂量(mg/kg)"), "10")
        self.assertIn("0.25、0.5、2、6、24、96 和192 h", fields.get("采样时间", ""))

        metadata_rows = "\n".join(str(row or "") for row in metadata.get("row_texts", []) or [])
        self.assertIn("采样时间：0.25、0.5、2、6、24、96 和192 h", metadata_rows)
        self.assertNotIn("nd=未检出", metadata_rows)

    def test_pages82_to_86_study_table_panels_keep_notes_and_metadata_owned(self) -> None:
        page82_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 82
        ]
        self.assertEqual(len(page82_templates), 1)
        page82_metadata_rows = "\n".join(str(row or "") for row in page82_templates[0].get("row_texts", []) or [])
        self.assertNotIn("1)[组织]/[血浆]", page82_metadata_rows)
        page82_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 82
            and table.get("semantic_role") == "business_table"
        )
        page82_notes = " ".join(str(note.get("text", "")) for note in page82_table.get("note_blocks", []) or [])
        self.assertIn("1)[组织]/[血浆]", page82_notes)
        self.assertTrue(
            any(
                ref.get("marker") in {"1", "1)"}
                and "T/P1)" in str(ref.get("header_text", ""))
                and "[组织]/[血浆]" in str(ref.get("note_text", ""))
                for ref in page82_table.get("header_note_refs", []) or []
            ),
            msg=f"missing T/P marker ref in {page82_table.get('header_note_refs', [])!r}",
        )

        page83_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 83
            and table.get("semantic_role") == "business_table"
        )
        page83_header = [cell.get("text") for cell in page83_table.get("header", []) or []]
        self.assertEqual(page83_header, ["物种", "检测浓度", "%结合率", "编号", "卷", "页码"])
        self.assertNotIn("组织/器官", page83_header)
        self.assertFalse(page83_table.get("is_continuation"))
        self.assertFalse(page83_table.get("continued_from"))
        self.assertFalse(page82_table.get("continued_to"))

        page84_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 84
            and template.get("template_kind") == "study_metadata"
        ]
        page84_template_texts = [
            "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
            for template in page84_templates
        ]
        self.assertGreaterEqual(len(page84_template_texts), 2)
        self.assertTrue(any("试验编号：95702" in text and "胎盘转运" in text for text in page84_template_texts))
        self.assertTrue(any("试验编号：95703" in text and "乳汁排泄" in text for text in page84_template_texts))

        page84_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 84
            and table.get("semantic_role") == "business_table"
        ]
        self.assertEqual(len(page84_tables), 2)
        first84, second84 = sorted(page84_tables, key=lambda table: table.get("bbox", [0, 0, 0, 0])[1])
        first84_notes = " ".join(str(note.get("text", "")) for note in first84.get("note_blocks", []) or [])
        second84_notes = " ".join(str(note.get("text", "")) for note in second84.get("note_blocks", []) or [])
        self.assertIn("另外检测了母体血液", first84_notes)
        self.assertNotIn("试验编号：95703", first84_notes)
        self.assertNotEqual(second84_notes.strip(), "新生胎仔：")
        second84_grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in (second84.get("display_grid", []) or second84.get("raw_grid", []) or [])
        )
        self.assertIn("乳汁： | 0.6 | 0.8 | 1.0 | 1.1 | 1.3 | 0.4", second84_grid_text)
        self.assertIn("新生胎仔：", second84_grid_text)

        page85_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 85
            and table.get("semantic_role") == "business_table"
        ]
        self.assertTrue(page85_tables)
        self.assertFalse(
            any(table.get("is_continuation") or table.get("continued_from_table_id") for table in page85_tables),
            msg="2.6.5.9 starts a new study region and must not inherit page 84 lactation tables",
        )

    def test_page84_reclaimed_trailing_label_row_is_not_rendered_again_as_owned_note(self) -> None:
        lactation_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 84
            and table.get("table_id") == "tbl_029"
        )
        label = "\u65b0\u751f\u80ce\u4ed4\uff1a"
        raw_grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in lactation_table.get("raw_grid", []) or []
            if isinstance(row, list)
        )
        self.assertIn(label, raw_grid_text)

        segment_text = "\n".join(
            str(segment.get("text") or "")
            for segment in lactation_table.get("content_segments", []) or []
            if isinstance(segment, dict)
        )
        self.assertNotIn(label, segment_text)
        self.assertNotEqual(str(lactation_table.get("content_text") or "").strip(), label)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        page84_start = review_markdown.index(
            "2.6.5.7 \u836f\u4ee3\u52a8\u529b\u5b66\uff1a\u598a\u5a20\u6216\u54fa\u4e73\u52a8\u7269\u7814\u7a76"
        )
        page85_start = review_markdown.index(
            "2.6.5.9 \u836f\u4ee3\u52a8\u529b\u5b66\uff1a\u4f53\u5185\u4ee3\u8c22",
            page84_start,
        )
        page84_region = review_markdown[page84_start:page85_start]
        self.assertIn(f"| {label} |", page84_region)
        self.assertNotIn(f"\n\n{label}\n\n", page84_region)

    def test_ind_review_markdown_renders_page84_lactation_table_rows_before_page85_metabolism(self) -> None:
        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")

        lactation_heading = (
            "2.6.5.7 "
            "\u836f\u4ee3\u52a8\u529b\u5b66\uff1a\u598a\u5a20\u6216\u54fa\u4e73\u52a8\u7269\u7814\u7a76 "
            "\u4f9b\u8bd5\u54c1\uff1a\u66f2\u9187\u94a0"
        )
        normalized_markdown = _normalize_ind_review_visibility_text(markdown)
        self.assertEqual(
            normalized_markdown.count(_normalize_ind_review_visibility_text(lactation_heading)),
            1,
            msg="split populated study metadata continuation should not repeat the same visible title",
        )
        part2_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if str(template.get("structure_template_id") or "") == "structure_template_047_part2"
        )
        self.assertTrue(part2_template.get("is_structure_template_continuation"), msg=part2_template)

        page84_heading = "2.6.5.7 药代动力学：妊娠或哺乳动物研究 供试品：曲醇钠"
        page85_heading = "2.6.5.9 药代动力学：体内代谢 供试品：曲醇钠"
        page84_index = markdown.index(page84_heading)
        page85_index = markdown.index(page85_heading)
        self.assertLess(page84_index, page85_index)
        page84_region = markdown[page84_index:page85_index]
        page85_region = markdown[page85_index:]

        self.assertIn("试验编号：95703", page84_region)
        self.assertIn("| 乳汁： | 0.6 | 0.8 | 1.0 | 1.1 | 1.3 | 0.4 |", page84_region)
        self.assertIn("| 新生胎仔： |", page84_region)
        self.assertNotIn("| 种属 样品 | 周期 | 量% 原形药物 | M1 |", page84_region)
        self.assertIn(
            "| 种属 | 样品 | 采样时间或 周期 | 占给药剂 量% | 原形药物 | M1 | M2 | 试验编号 | 卷 | 部分 |",
            page85_region,
        )
        self.assertIn(
            "| 大鼠 | 血浆 | 0.5 h | - | 87.2 | 6.1 | 3.4 | 95076 | 26 | 101 |",
            page85_region,
        )
        self.assertNotIn("| 大鼠 血浆 | 0.5 h | - | 87.2 6.1 |", page85_region)

    def test_ind_review_markdown_keeps_template_additional_info_before_below_notes(self) -> None:
        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")

        page37_heading = "2.6.5.9 药代动力学：体内代谢 供试品"
        page38_heading = "2.6.5.10 药代动力学：体外代谢 供试品"
        next_heading = "2.6.5.11 药代动力学：可能的代谢途径"
        remark = "备注：如有人体数据，应列入以便比较。"
        additional_info = "附加信息："

        page37_index = markdown.index(page37_heading)
        page38_index = markdown.index(page38_heading, page37_index)
        page37_region = markdown[page37_index:page38_index]
        self.assertIn("- 附加信息：", page37_region)
        self.assertNotIn("| 附加信息： |", page37_region)
        self.assertIn(remark, page37_region)
        self.assertLess(page37_region.index("- 附加信息："), page37_region.index(remark))

        page38_index = markdown.index(page38_heading, page37_index)
        next_index = markdown.index(next_heading, page38_index)
        page38_region = markdown[page38_index:next_index]
        self.assertIn(additional_info, page38_region)
        self.assertIn(remark, page38_region)
        self.assertLess(page38_region.index(additional_info), page38_region.index(remark))

    def test_structure_template_tail_remarks_are_notes_not_form_rows(self) -> None:
        page45_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 45
        )
        page49_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 49
        )

        page45_notes = "\n".join(str(note.get("text") or "") for note in page45_template.get("note_blocks", []) or [])
        page45_rows = "\n".join(str(row or "") for row in page45_template.get("row_texts", []) or [])
        self.assertIn("附加信息：(2)", page45_notes)
        self.assertIn("备注：(1)国际非专利药品名称(INN)", page45_notes)
        self.assertIn("(2)例如，简要说明结果、种属差异、性别差异、剂量依赖性或特殊备注。", page45_notes)
        self.assertIn("(4)如有其他途径", page45_notes)
        self.assertNotIn("附加信息：(2)", page45_rows)
        self.assertNotIn("备注：(1)国际非专利药品名称(INN)", page45_rows)

        page49_notes = "\n".join(str(note.get("text") or "") for note in page49_template.get("note_blocks", []) or [])
        page49_rows = "\n".join(str(row or "") for row in page49_template.get("row_texts", []) or [])
        self.assertIn("备注：(1)国际非专利药品名称(INN)", page49_notes)
        self.assertIn("(3)应指明技术报告在CTD 中的位置。", page49_notes)
        self.assertIn("a- 除非另有规定", page49_notes)
        self.assertNotIn("备注：(1)国际非专利药品名称(INN)", page49_rows)

        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        page45_heading = "2.6.5.13 药代动力学：排泄 供试品：(1)"
        page49_heading = "2.6.7.1 毒理学概述 供试品：（1）"
        page45_region = markdown[markdown.index(page45_heading):markdown.index("2.6.5.14 药代动力学：胆汁排泄", markdown.index(page45_heading))]
        page49_region = markdown[markdown.index(page49_heading):markdown.index("2.6.7.2 毒代动力学", markdown.index(page49_heading))]
        self.assertIn("\n附加信息：(2)", page45_region)
        self.assertIn("\n备注：(1)国际非专利药品名称(INN)", page45_region)
        self.assertNotIn("\n- 附加信息：(2)", page45_region)
        self.assertNotIn("\n- 备注：(1)国际非专利药品名称(INN)", page45_region)
        self.assertIn("\n备注：(1)国际非专利药品名称(INN)", page49_region)
        self.assertNotIn("\n- 备注：(1)国际非专利药品名称(INN)", page49_region)

    def test_page52_template_numbered_remarks_keep_note_run_order(self) -> None:
        page52_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 52
            and "2.6.7.4" in str(template.get("title") or "")
        )
        ordered_notes = [
            str(note.get("text") or "")
            for note in page52_template.get("note_blocks", []) or []
            if isinstance(note, dict)
        ]
        note_text = "\n".join(ordered_notes)
        note_one = "备注：(1)国际非专利药品名称(INN)"
        note_two = "(2)应按照大体时间顺序列出毒理学试验中使用的所有批次。"
        note_three = "(3)应注明每个批次所使用的毒理学试验。"
        self.assertIn(note_one, note_text)
        self.assertIn(note_two, note_text)
        self.assertIn(note_three, note_text)
        self.assertLess(ordered_notes.index(note_one), ordered_notes.index(note_two))
        self.assertLess(ordered_notes.index(note_two), ordered_notes.index(note_three))

        note_numbers = {
            str(note.get("text") or ""): note.get("note_number")
            for note in page52_template.get("note_blocks", []) or []
            if isinstance(note, dict)
        }
        self.assertEqual(note_numbers.get(note_one), 1)
        self.assertEqual(note_numbers.get(note_two), 2)
        self.assertEqual(note_numbers.get(note_three), 3)

        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        page52_heading = "2.6.7.4 毒理学 原料药 供试品：（1）"
        next_heading = "2.6.7.5"
        start = markdown.index(page52_heading)
        end = markdown.index(next_heading, start + 1)
        region = markdown[start:end]
        self.assertLess(region.index(note_one), region.index(note_two))
        self.assertLess(region.index(note_two), region.index(note_three))

    def test_page54_cross_page_template_tail_notes_attach_to_previous_template(self) -> None:
        page54 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 54)
        page54_text_by_id = {
            str(block.get("block_id") or ""): str(block.get("text") or "")
            for block in page54.get("blocks", []) or []
            if str(block.get("block_type") or "") == "text"
        }
        note_one_start = page54_text_by_id["txt_p54_011"]
        note_one_continuation = page54_text_by_id["txt_p54_012"]
        note_two = page54_text_by_id["txt_p54_013"]
        legend_note = page54_text_by_id["txt_p54_014"]

        page54_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 54
            and "2.6.7.6" in str(template.get("title") or "")
        )
        next_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 54
            and "2.6.7.7" in str(template.get("title") or "")
        )

        note_texts = [
            str(note.get("text") or "")
            for note in page54_template.get("note_blocks", []) or []
            if isinstance(note, dict)
        ]
        note_text = "\n".join(note_texts)
        self.assertIn(note_one_start, note_text)
        self.assertIn(note_one_continuation, note_text)
        self.assertIn(note_two, note_text)
        self.assertIn(legend_note, note_text)
        self.assertLess(note_text.index(note_one_start), note_text.index(note_one_continuation))
        self.assertLess(note_text.index(note_one_continuation), note_text.index(note_two))
        self.assertLess(note_text.index(note_two), note_text.index(legend_note))

        owned_by_next = set(str(block_id) for block_id in next_template.get("owned_text_block_ids", []) or [])
        self.assertFalse({"txt_p54_011", "txt_p54_012", "txt_p54_013", "txt_p54_014"} & owned_by_next)

        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        page54_heading = str(page54_template.get("title") or "")
        next_heading = str(next_template.get("title") or "")
        start = markdown.index(page54_heading)
        end = markdown.index(next_heading, start + 1)
        region = markdown[start:end]
        self.assertIn(note_two, region)
        self.assertIn(legend_note, region)
        self.assertLess(region.index(note_one_start), region.index(note_two))
        self.assertLess(region.index(note_two), region.index(legend_note))

    def test_page55_split_template_matrix_tail_legend_notes_attach_to_template(self) -> None:
        page55 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 55)
        page55_text_by_id = {
            str(block.get("block_id") or ""): str(block.get("text") or "")
            for block in page55.get("blocks", []) or []
            if str(block.get("block_type") or "") == "text"
        }
        legend_note = page55_text_by_id["txt_p55_025"]
        stats_note = page55_text_by_id["txt_p55_026"]
        marker_note = page55_text_by_id["txt_p55_027"]

        page55_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 55
        ]
        repeated_dose_template = next(
            (
                template
                for template in page55_templates
                if "2.6.7.7" in str(template.get("title") or template.get("continued_from_title") or "")
            ),
            None,
        )
        self.assertIsNotNone(repeated_dose_template, msg=page55_templates)

        rows_text = "\n".join(str(row or "") for row in repeated_dose_template.get("row_texts", []) or [])
        notes_text = "\n".join(
            str(note.get("text") or "")
            for note in repeated_dose_template.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertIn("日剂量(mg/kg)", rows_text)
        self.assertIn("毒代动力学：AUC()(4)", rows_text)
        self.assertIn("心电图", rows_text)
        self.assertIn(legend_note, notes_text)
        self.assertIn(stats_note, notes_text)
        self.assertIn(marker_note, notes_text)
        self.assertNotIn(legend_note, rows_text)
        self.assertNotIn(stats_note, rows_text)
        self.assertNotIn(marker_note, rows_text)

        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        section_heading = "2.6.7.7 (1)重复给药毒性(2) 报告标题： 供试品：(3)"
        continuation_heading = "2.6.7.7 (1)重复给药毒性 试验编号(续)"
        section_start = markdown.index(section_heading)
        continuation_start = markdown.index(continuation_heading, section_start + 1)
        region = markdown[section_start:continuation_start]
        self.assertIn(legend_note, region)
        self.assertIn(stats_note.replace("*", "\\*"), region)
        self.assertIn(marker_note, region)
        self.assertNotIn(f"{legend_note} {stats_note} {marker_note}", region)
        self.assertLess(region.index(legend_note), region.index(stats_note.replace("*", "\\*")))
        self.assertLess(region.index(stats_note.replace("*", "\\*")), region.index(marker_note))

    def test_page60_template_title_dedupes_repeated_test_article_label(self) -> None:
        page60_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 60
            and "2.6.7.10" in str(template.get("title") or "")
        )
        title = str(page60_template.get("title") or "")
        self.assertEqual(title.count("供试品：(2)"), 1)
        self.assertNotIn("供试品：(2) 供试品：(2)", title)

        rows_text = "\n".join(str(row or "") for row in page60_template.get("row_texts", []) or [])
        self.assertEqual(rows_text.count("供试品：(2)"), 1)
        self.assertIn("性别 M F M F M F M F", rows_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        heading_marker = "2.6.7.10 (1)致癌性：报告标题： 供试品：(2)"
        heading_index = review_markdown.index(heading_marker)
        heading_line = review_markdown[heading_index : review_markdown.find("\n", heading_index)]
        self.assertEqual(heading_line.count("供试品：(2)"), 1)
        self.assertNotIn("供试品：(2) 供试品：(2)", heading_line)

    def test_page61_numbered_continuation_title_stays_heading_not_template_row(self) -> None:
        title_text = "2.6.7.10 (1)致癌性 试验编号(续)"
        page61 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 61)
        title_blocks = [
            block
            for block in page61.get("blocks", []) or []
            if str(block.get("block_type") or "") == "text"
            and title_text in str(block.get("text") or "")
        ]
        self.assertEqual(len(title_blocks), 1, msg=title_blocks)
        self.assertNotEqual(title_blocks[0].get("semantic_role"), "structure_template_entry")
        self.assertIn(title_blocks[0].get("semantic_role"), {"section_heading", "structure_template_title"})

        page61_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 61
            and str(template.get("template_profile") or "") == "blank_study_summary_template_continuation"
        )
        rows_text = "\n".join(str(row or "") for row in page61_template.get("row_texts", []) or [])
        self.assertNotIn(title_text, rows_text)
        self.assertIn("日剂量(mg/kg)", rows_text)
        self.assertIn("评价数量", rows_text)
        self.assertIn("发生肿瘤病变的动物数量：", rows_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        self.assertIn(f"###### {title_text}", review_markdown)
        heading_index = review_markdown.index(f"###### {title_text}")
        next_heading_index = review_markdown.find("\n###### ", heading_index + 1)
        region = review_markdown[heading_index : next_heading_index if next_heading_index > 0 else len(review_markdown)]
        self.assertNotIn(f"- {title_text}", region)

    def test_page63_multiline_template_notes_keep_physical_order_and_do_not_render_as_rows(self) -> None:
        page63_template = next(
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 63
            and "2.6.7.11" in str(template.get("title") or "")
        )
        note_start = "注释： （1）应按照与CTD 相同的顺序"
        note_continuation = "全性试验》（1997 年11 月）规定的确定性GLP 试验除外）。但应使用更详细的模板来总结探索试验。"
        note_two = "（2）国际非专利药品名称（INN）。"
        rows_text = "\n".join(str(row or "") for row in page63_template.get("row_texts", []) or [])
        notes_text = "\n".join(str(note.get("text") or "") for note in page63_template.get("note_blocks", []) or [])

        self.assertIn(note_start, notes_text)
        self.assertIn(note_continuation, notes_text)
        self.assertIn(note_two, notes_text)
        self.assertNotIn(note_continuation, rows_text)
        self.assertNotIn(note_two, rows_text)

        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        heading = "2.6.7.11 生殖毒性：非关键试验(1) 供试品：(2)"
        next_heading = "2.6.7.12"
        start = markdown.index(heading)
        end = markdown.index(next_heading, start + 1)
        region = markdown[start:end]

        self.assertIn(note_start, region)
        self.assertIn(note_continuation, region)
        self.assertIn(note_two, region)
        self.assertNotIn(f"\n- {note_continuation}", region)
        self.assertNotIn(f"\n- {note_two}", region)
        self.assertLess(region.index(note_start), region.index(note_continuation))
        self.assertLess(region.index(note_continuation), region.index(note_two))

    def test_page66_top_template_instruction_note_renders_before_bottom_section_heading(self) -> None:
        note_heading = "表2.6.7.12、2.6.7.13 和2.6.7.14 的注释"
        section_heading = "2.6.7.13 (1)生殖毒性- 报告标题： 供试品：(2)"
        page66 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 66)
        visible_blocks = [
            block
            for block in page66.get("blocks", []) or []
            if str(block.get("text") or block.get("title") or "").strip()
        ]
        note_index = next(
            index
            for index, block in enumerate(visible_blocks)
            if note_heading in str(block.get("text") or block.get("title") or "")
        )
        section_index = next(
            index
            for index, block in enumerate(visible_blocks)
            if section_heading in str(block.get("text") or block.get("title") or "")
        )
        self.assertLess(note_index, section_index)

        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        note_pos = markdown.index(note_heading)
        section_pos = markdown.index(section_heading)
        self.assertLess(note_pos, section_pos)

    def test_pages64_to_68_template_legends_are_notes_and_hide_internal_structure_heading(self) -> None:
        legend_fragments_by_page = {
            64: [
                "\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c",
                "*-p<0.05 **-p<0.01",
            ],
            65: [
                "\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c",
                "*-p<0.05 **-p<0.01",
            ],
            67: [
                "\u65e0\u663e\u8457\u5f02\u5e38",
                "G=\u598a\u5a20\u65e5",
                "*-p<0.05 **-p<0.01",
            ],
            68: [
                "\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c",
                "*-p<0.05 **-p<0.01",
            ],
        }

        for page_number, fragments in legend_fragments_by_page.items():
            with self.subTest(page=page_number):
                template = next(
                    template
                    for template in self.result.get("structure_templates", []) or []
                    if int(template.get("page", 0) or 0) == page_number
                )
                row_text = "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
                note_text = "\n".join(str(note.get("text") or "") for note in template.get("note_blocks", []) or [])
                for fragment in fragments:
                    self.assertIn(fragment, note_text)
                    self.assertNotIn(fragment, row_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        self.assertNotIn("#### \u8868\u683c\u5f0f\u7ed3\u6784", review_markdown)

        self.assertIn("\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", review_markdown)
        self.assertIn("\u65e0\u663e\u8457\u5f02\u5e38", review_markdown)
        self.assertIn("\\*-p<0.05 \\*\\*-p<0.01", review_markdown)
        self.assertNotIn("\n- -\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", review_markdown)
        self.assertNotIn("\n- - \u65e0\u663e\u8457\u5f02\u5e38", review_markdown)
        self.assertNotIn("\n- \\*-p<0.05 \\*\\*-p<0.01", review_markdown)

    def test_ind_review_markdown_hides_template_type_labels_and_dedupes_template_titles(self) -> None:
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")

        self.assertNotIn("\u7ed3\u6784\u6a21\u677f", review_markdown)
        self.assertNotIn("\u8868\u5355\u6a21\u677f", review_markdown)

        titles = [
            "2.6.3.1 \u836f\u7406\u5b66\u6982\u8ff0 \u4f9b\u8bd5\u54c1\uff1a\uff081\uff09",
            "2.6.5.1 \u836f\u4ee3\u52a8\u529b\u5b66\u6982\u8ff0 \u4f9b\u8bd5\u54c1\uff1a\uff081\uff09",
            "2.6.7.11 \u751f\u6b96\u6bd2\u6027\uff1a\u975e\u5173\u952e\u8bd5\u9a8c(1) \u4f9b\u8bd5\u54c1\uff1a(2)",
            "2.6.7.12 (1)\u751f\u6b96\u6bd2\u6027- \u62a5\u544a\u6807\u9898\uff1a \u4f9b\u8bd5\u54c1\uff1a(2)",
        ]
        for title in titles:
            heading = f"#### {title}"
            self.assertIn(heading, review_markdown)
            heading_index = review_markdown.index(heading)
            next_heading = review_markdown.find("\n#### ", heading_index + 1)
            region = review_markdown[
                heading_index : next_heading if next_heading > 0 else len(review_markdown)
            ]
            self.assertNotIn(f"\n- {title}", region)

    def test_ind_review_markdown_renders_page82_metadata_and_keeps_page83_table_separate(self) -> None:
        document = dict(self.result)
        markdown = _build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("试验编号：95207", markdown)
        self.assertIn("物种：大鼠", markdown)
        self.assertIn("分析物/分析方法(单位)：原形化合物µg/ml)/HPLC", markdown)
        self.assertIn("采样时间：10min、1、4、8、24、48、96 和168 h", markdown)

        page82_heading = "2.6.5.5 药代动力学：器官分布 供试品：曲醇钠"
        page83_heading = "2.6.5.6 药代动力学：血浆蛋白结合率 供试品：曲醇钠"
        page82_index = markdown.index(page82_heading)
        page83_index = markdown.index(page83_heading)
        page82_region = markdown[page82_index:page83_index]

        self.assertIn("| 组织/器官 | 浓度 | T/P1) | 浓度 | T/P1) | 时间 | AUC | t1/2 |", page82_region)
        self.assertNotIn("| 物种 | 检测浓度 | %结合率 | 编号 | 卷 | 页码 |", page82_region)
        self.assertIn("| 物种 | 检测浓度 | %结合率 | 编号 | 卷 | 页码 |", markdown[page83_index:])

    def test_pages85_to_86_study_result_tables_split_from_cross_page_notes(self) -> None:
        page85_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 85
            and table.get("semantic_role") == "business_table"
        ]
        self.assertGreaterEqual(len(page85_tables), 1)
        self.assertFalse(
            any(
                len(table.get("display_grid", []) or table.get("raw_grid", []) or []) >= 20
                and "通用技术文档-安全性" in "\n".join(
                    " | ".join(str(cell or "") for cell in row)
                    for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
                )
                for table in page85_tables
            ),
            msg="page 85 should not collapse page header, metadata, table, and notes into one monster table",
        )
        page85_grid_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            for table in page85_tables
        ]
        self.assertTrue(
            any(
                "样品中的化合物%" in text
                and "原形药物" in text
                and "M1" in text
                and "M2" in text
                for text in page85_grid_texts
            ),
            msg=page85_grid_texts,
        )

        page86_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 86
        ]
        self.assertFalse(
            any(template.get("template_profile") == "blank_study_summary_template" for template in page86_templates),
            msg=page86_templates,
        )
        page86_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 86
            and table.get("semantic_role") == "business_table"
        ]
        self.assertGreaterEqual(len(page86_tables), 1)
        page86_table = max(page86_tables, key=lambda table: int(table.get("col_count", 0) or 0))
        page86_grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in (page86_table.get("display_grid", []) or page86_table.get("raw_grid", []) or [])
        )
        for expected in ("排泄途径(4)", "尿液", "粪便", "合计", "0-24 h", "95102", "95156"):
            self.assertIn(expected, page86_grid_text)
        self.assertNotIn("* - 为了采集胆汁", page86_grid_text)
        self.assertNotIn("n.d.- 未检出", page86_grid_text)

        page85_note_text = " ".join(
            str(note.get("text", ""))
            for table in page85_tables
            for note in table.get("note_blocks", []) or []
        )
        self.assertIn("为了采集胆汁", page85_note_text)
        self.assertIn("n.d.- 未检出", page85_note_text)

    def test_page86_top_cross_page_note_is_owned_by_page85_table(self) -> None:
        page85_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 85
            and table.get("semantic_role") == "business_table"
            and "M1" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            and "M2" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )
        note = next(
            note
            for note in page85_table.get("note_blocks", []) or []
            if "为了采集胆汁" in str(note.get("text", ""))
        )
        page85_table_id = str(page85_table.get("table_id") or page85_table.get("block_id") or "")

        self.assertEqual(note.get("relation"), "cross_page_note_continuation")
        self.assertEqual(note.get("logical_owner_page"), 85)
        self.assertEqual(note.get("physical_page"), 86)
        self.assertEqual(str(note.get("owner_table_id") or ""), page85_table_id)
        self.assertEqual(str(note.get("continued_from_table_id") or ""), page85_table_id)
        self.assertEqual(note.get("note_scope"), "previous_table")
        self.assertEqual(
            note.get("presentation_boundary"),
            "render_with_owner_table_before_next_page_structure",
        )
        ownership = note.get("cross_page_ownership") or {}
        self.assertEqual(ownership.get("owner_type"), "table")
        self.assertEqual(ownership.get("logical_owner_page"), 85)
        self.assertEqual(ownership.get("physical_page"), 86)
        self.assertEqual(str(ownership.get("owner_table_id") or ""), page85_table_id)

        page86_structure_text = "\n".join(
            "\n".join(
                str(item or "")
                for item in [
                    *(template.get("row_texts", []) or []),
                    *(entry.get("text", "") for entry in template.get("fields", []) or []),
                    *(entry.get("text", "") for entry in template.get("sections", []) or []),
                    *(entry.get("text", "") for entry in template.get("entries", []) or []),
                    *(entry.get("text", "") for entry in template.get("note_blocks", []) or []),
                ]
            )
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 86
        )
        self.assertNotIn("为了采集胆汁", page86_structure_text)
        self.assertNotIn("n.d.- 未检出", page86_structure_text)

        markdown = _build_full_markdown([dict(self.result)], markdown_profile="ind-review")
        page85_heading = "2.6.5.9 药代动力学：体内代谢 供试品：曲醇钠"
        page86_heading = "2.6.5.13 药代动力学：排泄 供试品：曲醇钠"
        note_text = "* - 为了采集胆汁，十二指肠给药。 n.d.- 未检出"
        page85_pos = markdown.index(page85_heading)
        page86_pos = markdown.index(page86_heading)
        note_pos = markdown.index(note_text)
        self.assertGreater(note_pos, page85_pos)
        self.assertLess(note_pos, page86_pos)
        next_heading_pos = markdown.find("\n#### ", page86_pos + 1)
        page86_region = markdown[page86_pos:] if next_heading_pos < 0 else markdown[page86_pos:next_heading_pos]
        self.assertNotIn("为了采集胆汁", page86_region)
        self.assertNotIn("n.d.- 未检出", page86_region)

    def test_page85_visual_study_table_releases_header_and_metadata_ownership(self) -> None:
        page85_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 85
            and table.get("semantic_role") == "business_table"
        ]
        self.assertGreaterEqual(len(page85_tables), 1)
        metabolism_table = next(
            table
            for table in page85_tables
            if "M1" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            and "M2" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )
        table_bbox = metabolism_table.get("bbox") or []
        self.assertEqual(len(table_bbox), 4)
        self.assertGreater(
            float(table_bbox[1]),
            250.0,
            msg="page 85 visual study table bbox should start at the data matrix, not at the running header",
        )

        page85_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 85
            and template.get("template_kind") == "study_metadata"
        ]
        template_text = "\n".join(
            "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
            for template in page85_templates
        )
        self.assertIn("2.6.5.9", template_text)
        self.assertIn("药代动力学：体内代谢", template_text)
        self.assertIn("给药方法", template_text)
        self.assertIn("放射性核素", template_text)

        metabolism_template = next(
            template
            for template in page85_templates
            if "2.6.5.9" in "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
        )
        column_projection = (
            metabolism_template.get("semantic_projection_v2", {}) or {}
        ).get("study_metadata_column_projection", {})
        self.assertEqual(column_projection.get("semantic_profile"), "study_metadata_columnar_conditions")
        self.assertEqual(column_projection.get("column_count"), 4)
        projected_rows = column_projection.get("rows") or []
        self.assertIn(
            ["性别(M/F)/动物数量", "大鼠：4M", "犬：3F", "人：8M"],
            projected_rows,
        )
        self.assertIn(
            ["剂量(mg/kg)", "大鼠：5 mg/kg", "犬：5 mg/kg", "人：75 mg"],
            projected_rows,
        )

    def test_page85_complex_borderless_table_projects_multilevel_header_and_row_groups(self) -> None:
        page85_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 85
            and table.get("semantic_role") == "business_table"
        ]
        metabolism_table = next(
            table
            for table in page85_tables
            if "M1" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            and "M2" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )

        projection = metabolism_table.get("semantic_projection_v2") or {}
        grouped_projection = projection.get("grouped_multilevel_borderless_projection") or {}
        self.assertEqual(
            grouped_projection.get("semantic_profile"),
            "grouped_multilevel_borderless_table",
            msg=projection,
        )

        semantic_header_text = " ".join(
            str(item.get("text", ""))
            for item in metabolism_table.get("semantic_header", []) or []
            if isinstance(item, dict)
        ).replace(" ", "")
        for expected in (
            "种属",
            "样品",
            "采样时间或周期",
            "占给药剂量%",
            "原形药物",
            "M1",
            "M2",
            "试验编号",
            "卷",
            "部分",
        ):
            self.assertIn(expected.replace(" ", ""), semantic_header_text)

        semantic_grid = metabolism_table.get("semantic_grid") or []
        self.assertGreaterEqual(len(semantic_grid), 13)
        self.assertEqual(
            [str(cell or "").replace(" ", "") for cell in semantic_grid[0]],
            [
                "种属",
                "样品",
                "采样时间或周期",
                "占给药剂量%",
                "原形药物",
                "M1",
                "M2",
                "试验编号",
                "卷",
                "部分",
            ],
        )
        self.assertEqual(
            semantic_grid[1],
            ["大鼠", "血浆", "0.5 h", "-", "87.2", "6.1", "3.4", "95076", "26", "101"],
        )
        self.assertEqual(
            [str(cell or "") for cell in semantic_grid[2]],
            ["", "尿液", "0-24 h", "2.1", "0.6", "n.d.", "0.2", "", "", ""],
        )
        self.assertEqual(
            semantic_grid[5],
            ["犬", "血浆", "0.5 h", "-", "92.8", "n.d.", "7.2", "95082", "26", "301"],
        )

        header_groups = grouped_projection.get("header_column_groups") or []
        compound_group = next(
            group
            for group in header_groups
            if str(group.get("text", "")).replace(" ", "") == "样品中的化合物%"
        )
        self.assertEqual(
            [str(child).replace(" ", "") for child in compound_group.get("child_headers", [])],
            ["原形药物", "M1", "M2"],
        )
        ctd_group = next(
            group
            for group in header_groups
            if str(group.get("text", "")).replace(" ", "") == "CTD中的位置"
        )
        self.assertEqual(
            [str(child).replace(" ", "") for child in ctd_group.get("child_headers", [])],
            ["卷", "部分"],
        )

        row_groups = grouped_projection.get("row_groups") or metabolism_table.get("row_groups") or []
        for expected in ("大鼠", "犬", "人"):
            group = next(group for group in row_groups if group.get("text") == expected)
            self.assertGreaterEqual(int(group.get("rowspan", 0) or 0), 3)
            self.assertIn("leading_stub", str(group.get("source", "")))

    def test_page86_wide_study_metadata_panel_is_recovered_before_result_matrix(self) -> None:
        page86_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 86
            and template.get("template_kind") == "study_metadata"
        ]
        template_text = "\n".join(
            "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
            for template in page86_templates
        )
        for expected in (
            "2.6.5.13",
            "药代动力学：排泄",
            "性别(M/F)/动物数量",
            "给药方法",
            "分析方法",
        ):
            self.assertIn(expected, template_text)

        page86_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 86
            and table.get("semantic_role") == "business_table"
            and "排泄途径" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )
        table_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in (page86_table.get("display_grid", []) or page86_table.get("raw_grid", []) or [])
        )
        self.assertNotIn("2.6.5.13", table_text)
        self.assertNotIn("给药方法", table_text)
        self.assertIn("排泄途径(4)", table_text)

    def test_page86_excretion_matrix_binds_study_context_and_projects_centered_colspans(self) -> None:
        page86_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 86
            and table.get("semantic_role") == "business_table"
            and "排泄途径" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )

        binding = page86_table.get("semantic_context_binding") or {}
        self.assertEqual(
            binding.get("semantic_profile"),
            "study_context_bound_result_matrix",
            msg=page86_table,
        )
        self.assertEqual(binding.get("group_count"), 4)
        self.assertEqual(binding.get("leaf_count_per_group"), 3)
        self.assertEqual(binding.get("logical_value_col_count"), 12)
        projection = (
            page86_table.get("semantic_projection_v2", {}) or {}
        ).get("study_condition_grouped_result_matrix_projection", {})
        self.assertEqual(
            projection.get("semantic_profile"),
            "study_condition_grouped_result_matrix",
            msg=page86_table.get("semantic_projection_v2"),
        )
        self.assertEqual(projection.get("condition_group_count"), 4)
        self.assertEqual(projection.get("leaf_count_per_group"), 3)
        self.assertEqual(projection.get("logical_leaf_column_count"), 13)
        semantic_grid = page86_table.get("semantic_grid") or []
        expected_leaf_headers = list(binding.get("leaf_headers") or [])
        self.assertEqual(len(semantic_grid[0]), 13)
        self.assertEqual(semantic_grid[0][1:], expected_leaf_headers)
        self.assertEqual(
            semantic_grid[1],
            ["0-24 h", "26", "57", "83", "22", "63", "85", "20", "29", "49", "23", "42", "65"],
        )
        self.assertEqual(
            [group.get("label") for group in binding.get("study_groups", [])],
            ["大鼠", "大鼠", "犬", "犬"],
        )
        for group in binding.get("study_groups", []):
            self.assertEqual(
                [leaf.get("text") for leaf in group.get("leaf_headers", [])],
                ["尿液", "粪便", "合计"],
            )

        trailing_spans = binding.get("trailing_colspan_rows") or []
        trial_spans = next(row for row in trailing_spans if row.get("label") == "试验编号")
        self.assertEqual(
            [(span.get("text"), span.get("start_leaf_col"), span.get("end_leaf_col")) for span in trial_spans.get("spans", [])],
            [("95102", 1, 6), ("95156", 7, 12)],
        )
        ctd_spans = next(row for row in trailing_spans if row.get("label") == "CTD中的位置")
        ctd_texts = [str(span.get("text", "")).replace(" ", "") for span in ctd_spans.get("spans", [])]
        self.assertEqual(ctd_texts, ["第20卷，第75页", "第20卷，第150页"])
        self.assertEqual(
            [(span.get("start_leaf_col"), span.get("end_leaf_col")) for span in ctd_spans.get("spans", [])],
            [(1, 6), (7, 12)],
        )

        page85_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 85
            and table.get("semantic_role") == "business_table"
            and "M1" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )
        continuation_notes = [
            note
            for note in page85_table.get("note_blocks", []) or []
            if "为了采集胆汁" in str(note.get("text", ""))
        ]
        self.assertTrue(continuation_notes)
        self.assertTrue(
            any(note.get("continued_from_page") == 86 or note.get("continuation_page") == 86 for note in continuation_notes),
            msg=continuation_notes,
        )

    def test_page86_excretion_matrix_renders_condition_groups_as_composite_table(self) -> None:
        page86_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 86
            and table.get("semantic_role") == "business_table"
            and "排泄途径" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )
        projection = (
            page86_table.get("semantic_projection_v2", {}) or {}
        ).get("study_condition_grouped_result_matrix_projection", {})
        self.assertEqual(
            projection.get("presentation_boundary"),
            "render_as_single_composite_table",
            msg=projection,
        )
        header_groups = projection.get("header_group_rows") or []
        self.assertEqual(
            [
                (group.get("label"), group.get("start_leaf_col"), group.get("end_leaf_col"), group.get("colspan"))
                for group in header_groups
            ],
            [
                ("大鼠 / 经口给药 / 10 mg/kg", 1, 3, 3),
                ("大鼠 / 静脉注射 / 5 mg/kg", 4, 6, 3),
                ("犬 / 经口给药 / 10 mg/kg", 7, 9, 3),
                ("犬 / 静脉注射 / 5 mg/kg", 10, 12, 3),
            ],
        )
        self.assertEqual(
            [group.get("descriptors", {}).get("溶媒/剂型") for group in header_groups],
            ["溶液 / 水", "溶液 / 生理盐水", "溶液 / 生理盐水", "胶囊"],
        )
        markdown = _build_full_markdown([dict(self.result)], markdown_profile="ind-review")
        heading = "2.6.5.13 药代动力学：排泄 供试品：曲醇钠"
        start = markdown.index(heading, markdown.index("为了采集胆汁"))
        end = markdown.index("2.6.5.14 药代动力学：胆汁排泄", start)
        page86_region = markdown[start:end]

        self.assertIn("大鼠-经口给药-10mgkg 尿液", page86_region)
        self.assertIn("大鼠-静脉注射-5mgkg 粪便", page86_region)
        self.assertIn("犬-经口给药-10mgkg 合计", page86_region)
        self.assertIn("犬-静脉注射-5mgkg 合计", page86_region)
        self.assertNotIn("| 时间 | 尿液 | 粪便 | 合计 | 尿液 | 粪便 | 合计 |", page86_region)
        self.assertNotIn("- 种属 大鼠 大鼠 犬 犬", page86_region)
        self.assertNotIn("- 给药方法 经口给药 静脉注射 经口给药 静脉注射", page86_region)

    def test_page89_toxicology_overview_continuation_links_to_page88_table(self) -> None:
        page88_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 88
            and table.get("semantic_role") == "business_table"
            and "试验类型" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            and "单次给药毒性" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )
        page89_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 89
            and table.get("semantic_role") == "business_table"
        )
        self.assertTrue(page89_table.get("is_continuation"))
        self.assertEqual(page89_table.get("continued_from_table_id"), page88_table.get("table_id"))
        page88_semantic_grid = page88_table.get("semantic_grid") or []
        page89_semantic_grid = page89_table.get("semantic_grid") or []
        self.assertGreaterEqual(len(page88_semantic_grid), 2)
        self.assertGreaterEqual(len(page89_semantic_grid), 2)
        self.assertEqual(len(page88_semantic_grid[0]), 10)
        self.assertEqual(len(page89_semantic_grid[0]), 10)
        self.assertEqual(page89_semantic_grid[0], page88_semantic_grid[0])
        self.assertIn("CD-1", str(page88_semantic_grid[1][1]))
        self.assertEqual(page88_semantic_grid[1][6], "Sponsor Inc.")
        self.assertEqual(page88_semantic_grid[1][7], "96046")
        self.assertIn("CD-1", str(page89_semantic_grid[1][1]))
        self.assertEqual(page89_semantic_grid[1][3], "21 月")
        self.assertEqual(page89_semantic_grid[1][6], "CRO Co.")
        self.assertEqual(page89_semantic_grid[1][7], "95012")
        projection = (
            page89_table.get("semantic_projection_v2", {}) or {}
        ).get("overview_inventory_schema_projection", {})
        self.assertEqual(projection.get("semantic_profile"), "nonclinical_overview_inventory_table")
        self.assertTrue(projection.get("continuation_schema_inherited"))

    def test_page92_auc_data_table_keeps_header_rows_and_does_not_emit_blank_template(self) -> None:
        page92_blank_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 92
            and template.get("template_profile") == "blank_study_summary_template"
        ]
        self.assertEqual(page92_blank_templates, [])

        page92_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 92
            and table.get("semantic_role") == "business_table"
        ]
        self.assertGreaterEqual(len(page92_tables), 1)
        table_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            for table in page92_tables
        ]
        self.assertTrue(
            any(
                "稳态AUC" in text
                and "日剂量" in text
                and "小鼠" in text
                and "大鼠" in text
                and "7000" in text
                for text in table_texts
            ),
            msg=table_texts,
        )

    def test_page92_auc_table_projects_compressed_multilevel_header_spans(self) -> None:
        auc_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 92
            and table.get("semantic_role") == "business_table"
            and "\u7a33\u6001AUC" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )

        header_groups = auc_table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "\u7a33\u6001AUC (\u00b5g-h/ml)"
                and group.get("start_col") == 1
                and group.get("end_col") == 4
                and group.get("colspan") == 4
                for group in header_groups
            ),
            msg=f"missing steady-state AUC span in {header_groups!r}",
        )
        self.assertTrue(
            any(
                group.get("text") == "\u5c0f\u9f20a"
                and group.get("start_col") == 1
                and group.get("end_col") == 2
                and group.get("colspan") == 2
                for group in header_groups
            ),
            msg=f"missing mouse M/F span in {header_groups!r}",
        )
        self.assertTrue(
            any(
                group.get("text") == "\u5927\u9f20b"
                and group.get("start_col") == 3
                and group.get("end_col") == 4
                and group.get("colspan") == 2
                for group in header_groups
            ),
            msg=f"missing rat M/F span in {header_groups!r}",
        )

        logical_cells = auc_table.get("logical_cells", [])
        self.assertTrue(
            any(
                cell.get("text") == "\u7a33\u6001AUC (\u00b5g-h/ml)"
                and cell.get("row") == 0
                and cell.get("col") == 1
                and cell.get("colspan") == 4
                for cell in logical_cells
            ),
            msg=f"missing logical steady-state AUC colspan cell in {logical_cells!r}",
        )
        projection = auc_table.get("semantic_projection_v2", {})
        self.assertEqual(
            projection.get("compressed_multilevel_metric_header_projection", {}).get("semantic_profile"),
            "compressed_metric_multilevel_header",
        )
        grouped_projection = projection.get("study_metric_grouped_matrix_projection", {})
        self.assertEqual(grouped_projection.get("semantic_profile"), "study_metric_grouped_matrix")
        self.assertEqual(grouped_projection.get("logical_column_count"), 8)
        self.assertEqual(
            (auc_table.get("semantic_grid") or [])[0],
            ["日剂量（mg/kg）", "M", "F", "M", "F", "犬c", "雌兔b", "人f"],
        )
        semantic_grid = auc_table.get("semantic_grid") or []
        rows_by_dose = {str(row[0] or ""): row for row in semantic_grid[1:] if isinstance(row, list) and row}
        self.assertEqual(rows_by_dose["5"][5], "3")
        self.assertEqual(rows_by_dose["10"][5], "4")
        self.assertEqual(rows_by_dose["20"][5], "10")
        self.assertNotIn("5 10 20", " ".join(str(cell or "") for row in semantic_grid for cell in row))
        self.assertTrue(
            any(
                group.get("text") == "犬c"
                and group.get("start_col") == 5
                and group.get("end_col") == 5
                for group in auc_table.get("header_column_groups", []) or []
            ),
            msg=auc_table.get("header_column_groups", []),
        )
        note_text = " ".join(str(note.get("text", "")) for note in auc_table.get("note_blocks", []) or [])
        for expected in ("a - 掺食", "b - 灌胃", "c –胶囊", "d – 6 个月毒性试验", "e –致癌性试验", "f –方案147-007"):
            self.assertIn(expected, note_text)
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and "小鼠" in str(ref.get("header_text", ""))
                and "掺食" in str(ref.get("note_text", ""))
                for ref in auc_table.get("header_note_refs", []) or []
            ),
            msg=auc_table.get("header_note_refs", []),
        )
        self.assertTrue(
            any(
                ref.get("marker") == "d"
                and "25d、22e" in str(ref.get("cell_text", ""))
                and "6 个月毒性试验" in str(ref.get("note_text", ""))
                for ref in auc_table.get("cell_note_refs", []) or []
            ),
            msg=auc_table.get("cell_note_refs", []),
        )

        page92_ast = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 92)
        page92_table_block = next(
            block
            for block in page92_ast.get("blocks", []) or []
            if block.get("block_type") == "table"
            and block.get("table_id") == auc_table.get("table_id")
        )
        self.assertIn("2.6.7.3 毒代动力学", str(page92_table_block.get("title", "")))
        body_text = "\n".join(
            str(block.get("text") or "")
            for block in page92_ast.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
        )
        self.assertNotIn("a - 掺食", body_text)
        self.assertNotIn("f –方案147-007", body_text)

    def test_page94_raw_material_table_reclaims_quality_and_batch_header_rows(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 94
            and table.get("semantic_role") == "business_table"
        )
        grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
        )
        for expected in (
            "\u6279\u53f7",
            "\u7eaf\u5ea6(%)",
            "\u7279\u5b9a\u6742\u8d28a",
            "\u8bd5\u9a8c\u7f16\u53f7",
            "\u8bd5\u9a8c\u7c7b\u578b",
        ):
            self.assertIn(expected, grid_text)
        header_text = " ".join(str(item.get("text") or "") for item in table.get("semantic_header", []) if isinstance(item, dict))
        self.assertIn("\u6279\u53f7", header_text)
        self.assertIn("\u7eaf\u5ea6(%)", header_text)
        projection = table.get("semantic_projection_v2", {})
        grouped_projection = projection.get("study_metric_grouped_matrix_projection", {})
        self.assertEqual(grouped_projection.get("semantic_profile"), "study_metric_grouped_matrix")
        self.assertEqual((table.get("semantic_grid") or [])[0], ["批号", "纯度(%)", "A", "B", "C", "试验编号", "试验类型"])
        header_groups = table.get("header_column_groups", []) or []
        self.assertTrue(
            any(
                group.get("text") == "特定杂质a"
                and group.get("start_col") == 2
                and group.get("end_col") == 4
                and group.get("colspan") == 3
                for group in header_groups
            ),
            msg=header_groups,
        )
        self.assertIn(["拟定的质量标准", ">95", "<0.1", "<0.2", "<0.3", "-", "-"], table.get("semantic_grid") or [])
        note_text = " ".join(str(note.get("text", "")) for note in table.get("note_blocks", []) or [])
        self.assertIn("a –面积百分比", note_text)
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and "特定杂质" in str(ref.get("header_text", ""))
                and "面积百分比" in str(ref.get("note_text", ""))
                for ref in table.get("header_note_refs", []) or []
            ),
            msg=table.get("header_note_refs", []),
        )
        page94_ast = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 94)
        body_text = "\n".join(
            str(block.get("text") or "")
            for block in page94_ast.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
        )
        self.assertNotIn("a –面积百分比", body_text)

    def test_page95_single_dose_toxicity_table_reclaims_header_and_keeps_iv_vehicle_row(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 95
            and table.get("semantic_role") == "business_table"
        )
        grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
        )
        for expected in (
            "\u79cd\u5c5e/\u54c1\u7cfb",
            "\u7ed9\u836f\u65b9\u6cd5(\u6eb6\u5a92/\u5242\u578b)",
            "\u5242\u91cf(mg/kg)",
            "\u6027\u522b\u548c\u6570\u91cf",
            "\u8bd5\u9a8c\u7f16\u53f7",
        ):
            self.assertIn(expected, grid_text)
        self.assertIn("\u9759\u8109\u6ce8\u5c04", grid_text)
        self.assertIn("(5%\u8461\u8404\u7cd6)", grid_text)
        projection = (table.get("semantic_projection_v2") or {}).get(
            "toxicology_summary_schema_projection",
            {},
        )
        self.assertEqual(
            projection.get("semantic_profile"),
            "toxicology_summary_schema_table",
            msg=table.get("semantic_projection_v2"),
        )
        self.assertEqual(projection.get("schema_variant"), "single_dose_toxicity_summary")
        self.assertEqual(projection.get("logical_column_count"), 8)
        semantic_grid = table.get("semantic_grid") or []
        self.assertGreaterEqual(len(semantic_grid), 8)
        self.assertEqual(
            semantic_grid[0],
            [
                "种属/品系",
                "给药方法(溶媒/剂型)",
                "剂量(mg/kg)",
                "性别和数量/组",
                "观察到的最大非致死剂量(mg/kg)",
                "近似致死剂量(mg/kg)",
                "值得注意的结果",
                "试验编号",
            ],
        )
        self.assertFalse(
            any(
                isinstance(row, list)
                and len(row) >= 8
                and row[0] in {"给药方法(溶媒/剂型)", "性别和数量/组", "观察到的最大非", "近似致死剂量"}
                and not any(str(cell or "").strip() for cell in row[1:])
                for row in semantic_grid
            ),
            msg=semantic_grid[:10],
        )
        first_mouse_row = next(
            row
            for row in semantic_grid
            if len(row) >= 8 and row[0] == "CD-1 小鼠" and row[1] == "灌胃"
        )
        self.assertEqual(first_mouse_row, ["CD-1 小鼠", "灌胃", "0、1000、2000、", "10M", "≥5000", ">5000", "≥2000：一过性体重下降", "96046"])
        first_mouse_vehicle_row = next(
            row
            for row in semantic_grid
            if len(row) >= 8 and row[1] == "(水)"
        )
        self.assertEqual(first_mouse_vehicle_row, ["", "(水)", "5000", "10F", "≥5000", "", "5000：活动减少、抽搐、衰弱", ""])
        rat_iv_row = next(
            row
            for row in semantic_grid
            if len(row) >= 8 and row[1] == "静脉注射" and row[-1] == "96051"
        )
        self.assertEqual(rat_iv_row[0], "")
        self.assertEqual(rat_iv_row[2], "0、100、250、")
        self.assertEqual(rat_iv_row[3], "5M")
        self.assertEqual(rat_iv_row[4], "250")
        self.assertEqual(rat_iv_row[5], ">250")
        self.assertEqual(rat_iv_row[6], "≥250：雄性体重下降")
        rat_iv_vehicle_row = next(
            row
            for row in semantic_grid
            if len(row) >= 8 and row[1] == "(5%葡萄糖)"
        )
        self.assertEqual(rat_iv_vehicle_row, ["", "(5%葡萄糖)", "500", "5F", "≥500", "<500", "500：3M 死亡", ""])
        rat_oral_row = next(
            row
            for row in semantic_grid
            if len(row) >= 8 and row[0] == "Wistar 大鼠" and row[1] == "灌胃"
        )
        self.assertEqual(rat_oral_row[-1], "96050")
        rat_oral_vehicle_row = next(
            row
            for row in semantic_grid
            if len(row) >= 8 and row[1] == "(CMC 混悬液)"
        )
        self.assertEqual(rat_oral_vehicle_row[2], "5000")
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        self.assertIn("CD-1 小鼠 | 灌胃 | 0、1000、2000、 | 10M | ≥5000 | >5000 | ≥2000：一过性体重下降 | 96046", review_markdown)
        self.assertNotIn("给药方法(溶媒/剂型) |  |  |", review_markdown)
        note_text = " ".join(str(note.get("text") or "") for note in table.get("note_blocks", []) or [] if isinstance(note, dict))
        self.assertNotIn("(5%\u8461\u8404\u7cd6) 500", note_text)
        self.assertEqual(semantic_grid.index(rat_iv_vehicle_row), semantic_grid.index(rat_iv_row) + 1)

    def test_page96_repeated_dose_toxicity_summary_uses_8_column_schema(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 96
            and table.get("semantic_role") == "business_table"
            and "2.6.7.6" in str(table.get("title") or "")
        )
        projection = (table.get("semantic_projection_v2") or {}).get(
            "toxicology_summary_schema_projection",
            {},
        )
        self.assertEqual(
            projection.get("semantic_profile"),
            "toxicology_summary_schema_table",
            msg=table.get("semantic_projection_v2"),
        )
        self.assertEqual(projection.get("schema_variant"), "repeated_dose_toxicity_summary")
        self.assertEqual(projection.get("logical_column_count"), 8)
        semantic_grid = table.get("semantic_grid") or []
        self.assertGreaterEqual(len(semantic_grid), 5)
        self.assertEqual(
            semantic_grid[0],
            [
                "\u79cd\u5c5e/\u54c1\u7cfb",
                "\u7ed9\u836f\u65b9\u6cd5(\u6eb6\u5a92/\u5242\u578b)",
                "\u7ed9\u836f\u671f\u9650",
                "\u5242\u91cf(mg/kg)",
                "\u6027\u522b\u548c\u6570\u91cf/\u7ec4",
                "NOAELa(mg/kg)",
                "\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c",
                "\u8bd5\u9a8c\u7f16\u53f7",
            ],
        )
        semantic_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in semantic_grid
            if isinstance(row, list)
        )
        self.assertNotIn("Column 7", semantic_text)
        self.assertFalse(
            any(isinstance(row, list) and row and row[0] == "\u5a92/\u5242\u578b)" for row in semantic_grid),
            msg=semantic_grid[:4],
        )
        self.assertFalse(
            any(
                isinstance(row, list)
                and row
                and str(row[0] or "") in {
                    "\u7ed9\u836f\u65b9\u6cd5(\u6eb6",
                    "\u5242\u91cf",
                    "Beagle \u72ac \u704c\u80c3(CMC \u6df7",
                    "5 \u5929",
                    "0 \u3001500 \u548c",
                    "\u60ac\u5242)",
                    "1000",
                }
                for row in semantic_grid
            ),
            msg=semantic_grid,
        )

        mouse_row = next(row for row in semantic_grid if row and row[0] == "CD-1 \u5c0f\u9f20")
        self.assertEqual(mouse_row[1], "\u63ba\u98df\u7ed9\u836f")
        self.assertEqual(mouse_row[2], "3 \u6708")
        self.assertIn("62.5", mouse_row[3])
        self.assertIn("7000", mouse_row[3])
        self.assertEqual(mouse_row[4], "10M\u300110F")
        self.assertIn("M\uff1a4000", mouse_row[5])
        self.assertEqual(mouse_row[-1], "94018")

        beagle_row = next(row for row in semantic_grid if row and row[0] == "Beagle \u72ac")
        self.assertEqual(beagle_row[1], "\u704c\u80c3(CMC \u6df7\u60ac\u5242)")
        self.assertEqual(beagle_row[2], "5 \u5929")
        self.assertEqual(beagle_row[3], "0\u3001500\u548c1000")
        self.assertEqual(beagle_row[4], "1M\u30011F")
        self.assertEqual(beagle_row[5], "<500")
        self.assertIn("\u98df\u6b32\u4e0d\u632f", beagle_row[6])
        self.assertEqual(beagle_row[7], "94008")

        note_text = " ".join(
            str(note.get("text") or "")
            for note in table.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertIn("\u672a\u89c1\u4e0d\u826f\u53cd\u5e94\u5242\u91cf", note_text)
        marker_refs = (table.get("header_note_refs", []) or []) + (table.get("cell_note_refs", []) or [])
        self.assertTrue(
            any(ref.get("marker") == "a" and "NOAEL" in str(ref.get("header_text") or ref.get("cell_text") or "") for ref in marker_refs),
            msg=marker_refs,
        )

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        self.assertIn("**2.6.7.6 \u91cd\u590d\u7ed9\u836f\u6bd2\u6027 \u975e\u5173\u952e\u8bd5\u9a8c \u4f9b\u8bd5\u54c1\uff1a\u66f2\u9187\u94a0**", review_markdown)
        self.assertIn(
            "| \u79cd\u5c5e/\u54c1\u7cfb | \u7ed9\u836f\u65b9\u6cd5(\u6eb6\u5a92/\u5242\u578b) | \u7ed9\u836f\u671f\u9650 | \u5242\u91cf(mg/kg) | \u6027\u522b\u548c\u6570\u91cf/\u7ec4 | NOAELa(mg/kg) | \u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c | \u8bd5\u9a8c\u7f16\u53f7 |",
            review_markdown,
        )
        self.assertIn("a \u2013\u672a\u89c1\u4e0d\u826f\u53cd\u5e94\u5242\u91cf", review_markdown)
        self.assertNotIn("Column 7", review_markdown)

    def test_page97_repeated_dose_result_panel_reclaims_title_header_and_notes(self) -> None:
        page97_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 97
            and table.get("semantic_role") == "business_table"
        ]
        candidates = [
            table
            for table in page97_tables
            if (table.get("semantic_projection_v2") or {})
            .get("dose_response_result_panel_projection", {})
            .get("semantic_profile")
            == "dose_response_result_panel"
            and "2.6.7.7A" in str(table.get("title") or "")
        ]
        self.assertTrue(candidates, msg=page97_tables)
        table = candidates[0]
        projection = (table.get("semantic_projection_v2") or {}).get("dose_response_result_panel_projection", {})
        self.assertEqual(projection.get("logical_column_count"), 9)
        semantic_grid = table.get("semantic_grid") or []
        self.assertGreaterEqual(len(semantic_grid), 14)
        self.assertEqual(
            semantic_grid[0],
            [
                "\u65e5\u5242\u91cf(mg/kg)",
                "0 M",
                "0 F",
                "200 M",
                "200 F",
                "600 M",
                "600 F",
                "1800 M",
                "1800 F",
            ],
        )
        grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in semantic_grid
            if isinstance(row, list)
        )
        self.assertIn("\u52a8\u7269\u6570\u91cf | M:30 | F:30 | M:20 | F:20 | M:20 | F:20 | M:30 | F:30", grid_text)
        self.assertIn("\u6709\u8272\u9f3b\u6db2\u6ea2\u3001\u76ae\u6bdb\u7ea2\u67d3\u3001\u5927\u4fbf\u53d1\u767d | - | - | - | - | - | - | ++ | ++", grid_text)
        self.assertIn("\u6b65\u6001\u5446\u677f | - | - | - | - | - | - | - | -", grid_text)
        self.assertNotIn("Column", grid_text)

        title = str(table.get("title") or "")
        self.assertIn("2.6.7.7A", title)
        self.assertIn("MM-180801", title)
        context_text = " ".join(
            str(item.get("text") or item)
            for item in table.get("study_context_blocks", []) or []
        )
        self.assertIn("\u79cd\u5c5e/\u54c1\u7cfb\uff1aWistar \u5927\u9f20", context_text)
        self.assertIn("\u672a\u89c1\u4e0d\u826f\u53cd\u5e94\u5242\u91cf\uff1a200 mg/kg", context_text)

        note_text = " ".join(
            str(note.get("text") or "")
            for note in table.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertIn("-\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", note_text)
        self.assertIn("Dunnett", note_text)
        self.assertIn("*-p<0.05", note_text)
        self.assertIn("++\u4e2d\u5ea6", note_text)

        page97_ast = next(
            page
            for page in self.result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page") or 0) == 97
        )
        body_text = "\n".join(
            str(block.get("text") or "")
            for block in page97_ast.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
        )
        self.assertNotIn("2.6.7.7A \u91cd\u590d\u7ed9\u836f\u6bd2\u6027", body_text)
        self.assertNotIn("\u65e5\u5242\u91cf(mg/kg)", body_text)
        self.assertNotIn("-\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", body_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_index = review_markdown.index("**2.6.7.7A \u91cd\u590d\u7ed9\u836f\u6bd2\u6027")
        next_title_index = review_markdown.index("**2.6.7.7A \u91cd\u590d\u7ed9\u836f\u6bd2\u6027 \u8bd5\u9a8c\u7f16\u53f7", title_index + 1)
        page97_markdown = review_markdown[title_index:next_title_index]
        self.assertIn("-\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", page97_markdown)
        self.assertIn("Dunnett \u6c0f\u68c0\u9a8c\uff1a*-p<0.05 **-p<0.01", page97_markdown)
        self.assertEqual(page97_markdown.count("\u6709\u8272\u9f3b\u6db2\u6ea2\u3001\u76ae\u6bdb\u7ea2\u67d3\u3001\u5927\u4fbf\u53d1\u767d"), 1)
        self.assertEqual(page97_markdown.count("-\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c"), 1)
        self.assertNotIn("| -\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", page97_markdown)
        self.assertNotIn("| **-p<0.01", page97_markdown)

    def test_page100_continuation_table_keeps_top_aligned_dose_and_auc_rows(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 100
            and table.get("semantic_role") == "business_table"
        )
        self.assertFalse(table.get("is_continuation"))
        self.assertFalse(str(table.get("continued_from_table_id") or table.get("continued_from") or "").strip())
        self.assertIn("2.6.7.7B", str(table.get("title") or ""))
        projection = (table.get("semantic_projection_v2") or {}).get("dose_response_result_panel_projection", {})
        self.assertEqual(projection.get("semantic_profile"), "dose_response_result_panel")
        self.assertEqual(projection.get("logical_column_count"), 9)
        grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
        )
        for expected in (
            "\u65e5\u5242\u91cf(mg/kg)",
            "\u52a8\u7269\u6570\u91cf",
            "\u6bd2\u4ee3\u52a8\u529b\u5b66\uff1aAUC(\u00b5g-h/ml)",
            "\u7b2c1\u5929",
            "\u7b2c28\u5929",
            "\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c",
            "\u6b7b\u4ea1\u6216\u5904\u6b7b\u7684\u5782\u6b7b\u52a8\u7269",
        ):
            self.assertIn(expected, grid_text.replace(" ", ""))
        self.assertNotIn("\u672a\u89c1\u4e0d\u826f\u53cd\u5e94\u5242\u91cf", grid_text)

        header_text = " ".join(
            str(cell.get("text") or "")
            for cell in (table.get("semantic_header") or table.get("header") or [])
            if isinstance(cell, dict)
        )
        self.assertIn("\u65e5\u5242\u91cf(mg/kg)", header_text.replace(" ", ""))
        self.assertNotIn("\u52a8\u7269\u6570\u91cf", header_text)
        self.assertNotIn("\u6bd2\u4ee3\u52a8\u529b\u5b66", header_text)

        semantic_rows = table.get("semantic_grid", []) or table.get("display_grid", []) or []
        semantic_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in semantic_rows)
        self.assertIn("\u52a8\u7269\u6570\u91cf", semantic_text)
        self.assertIn(
            "\u52a8\u7269\u6570\u91cf | M:3 | F:3 | M:3 | F:3 | M:3 | F:3 | M:3 | F:3",
            semantic_text,
        )
        self.assertIn("\u6bd2\u4ee3\u52a8\u529b\u5b66\uff1aAUC(\u00b5g-h/ml)", semantic_text)
        self.assertNotIn("M:3 F:3", semantic_text)
        self.assertNotIn("10 12", semantic_text)
        self.assertNotIn("5 8", semantic_text)
        self.assertFalse(
            any(
                "\u65e5\u5242\u91cf" in str(row[0] if row else "")
                and "\u52a8\u7269\u6570\u91cf" in str(row[0] if row else "")
                and "\u6bd2\u4ee3\u52a8\u529b\u5b66" in str(row[0] if row else "")
                for row in semantic_rows
                if isinstance(row, list)
            ),
            msg=semantic_rows[:5],
        )

    def test_page100_new_panel_breaks_page99_continuation_and_keeps_page99_notes(self) -> None:
        page99_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 99
            and "2.6.7.7A" in str(table.get("title") or "")
        )
        page100_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 100
            and "2.6.7.7B" in str(table.get("title") or "")
        )

        self.assertNotIn(
            page100_table.get("table_id"),
            [str(item or "") for item in page99_table.get("continued_to", []) or []],
        )
        self.assertFalse(
            str(page100_table.get("continued_from_table_id") or page100_table.get("continued_from") or "").strip()
        )

        page99_notes = " ".join(
            str(note.get("text") or "")
            for note in page99_table.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertIn("\u7ed9\u836f\u540e\u6062\u590d\u671f\u7ed3\u675f\u65f6", page99_notes)
        self.assertIn("\u7edd\u5bf9\u5668\u5b98\u91cd\u91cf", page99_notes)

        page100_notes = " ".join(
            str(note.get("text") or "")
            for note in page100_table.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertNotIn("\u7ed9\u836f\u540e\u6062\u590d\u671f\u7ed3\u675f\u65f6", page100_notes)
        self.assertNotIn("\u7edd\u5bf9\u5668\u5b98\u91cd\u91cf", page100_notes)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_index = review_markdown.index("**2.6.7.7B \u91cd\u590d\u7ed9\u836f\u6bd2\u6027")
        next_title_index = review_markdown.index("2.6.7.8A", title_index)
        page100_markdown = review_markdown[title_index:next_title_index]
        self.assertIn(
            "| \u65e5\u5242\u91cf(mg/kg) | 0 M | 0 F | 10 M | 10 F | 40 M | 40 F | 100 M | 100 F |",
            page100_markdown,
        )
        self.assertIn(
            "| \u52a8\u7269\u6570\u91cf | M:3 | F:3 | M:3 | F:3 | M:3 | F:3 | M:3 | F:3 |",
            page100_markdown,
        )

    def test_repeated_dose_template_labels_are_owned_not_rendered_as_body(self) -> None:
        label_texts_by_page = {
            97: "\u793a\u4f8b#1",
            98: "\u793a\u4f8b#1",
            99: "\u793a\u4f8b#1",
            100: "\u793a\u4f8b#2",
        }
        for page_number, label in label_texts_by_page.items():
            page_ast = next(
                page
                for page in self.result.get("document_ast", {}).get("pages", []) or []
                if int(page.get("page") or 0) == page_number
            )
            body_text = "\n".join(
                str(block.get("text") or "")
                for block in page_ast.get("blocks", []) or []
                if block.get("block_type") == "text"
                and str(block.get("unit_role") or "body") == "body"
            )
            self.assertNotIn(label, body_text)

            page_tables = [
                table
                for table in self.result.get("table_asts", []) or []
                if int(table.get("page", 0) or 0) == page_number
                and table.get("semantic_role") == "business_table"
                and (table.get("semantic_projection_v2") or {}).get("dose_response_result_panel_projection")
            ]
            self.assertTrue(page_tables, msg=page_number)
            owned_ids = {
                str(block_id)
                for table in page_tables
                for block_id in table.get("owned_text_block_ids", []) or []
            }
            repair_label_ids = {
                str(block_id)
                for table in page_tables
                for repair in table.get("semantic_repairs", []) or []
                if isinstance(repair, dict)
                and repair.get("repair") == "loose_template_label_bound_to_projected_business_table"
                for block_id in repair.get("template_label_block_ids", []) or []
            }
            self.assertTrue(repair_label_ids, msg=(page_number, page_tables))
            self.assertTrue(repair_label_ids <= owned_ids, msg=(page_number, repair_label_ids, owned_ids))

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_index = review_markdown.index("**2.6.7.7A \u91cd\u590d\u7ed9\u836f\u6bd2\u6027")
        next_title_index = review_markdown.index("2.6.7.8A", title_index)
        repeated_dose_markdown = review_markdown[title_index:next_title_index]
        self.assertNotIn("\u793a\u4f8b#1", repeated_dose_markdown)
        self.assertNotIn("\u793a\u4f8b#2", repeated_dose_markdown)

    def test_page101_repeated_dose_continuation_recovered_as_table_not_body_text(self) -> None:
        page100_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 100
            and "2.6.7.7B" in str(table.get("title") or "")
        )
        page101_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 101
            and table.get("semantic_role") == "business_table"
        ]
        candidates = [
            table
            for table in page101_tables
            if (table.get("semantic_projection_v2") or {})
            .get("dose_response_result_panel_projection", {})
            .get("semantic_profile")
            == "dose_response_result_panel"
            and "2.6.7.7B" in str(table.get("title") or "")
        ]
        self.assertTrue(candidates, msg=page101_tables)
        table = candidates[0]
        self.assertTrue(table.get("is_continuation"))
        self.assertEqual(table.get("continued_from_table_id"), page100_table.get("table_id"))
        semantic_grid = table.get("semantic_grid") or []
        self.assertEqual(
            semantic_grid[0],
            [
                "\u65e5\u5242\u91cf(mg/kg)",
                "0 M",
                "0 F",
                "10 M",
                "10 F",
                "40 M",
                "40 F",
                "100 M",
                "100 F",
            ],
        )
        grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in semantic_grid
            if isinstance(row, list)
        )
        self.assertIn("\u5668\u5b98\u91cd\u91cfa(%)", grid_text)
        self.assertIn("\u809d\u810f", grid_text)
        self.assertIn("+17**", grid_text)
        note_text = " ".join(
            str(note.get("text") or "")
            for note in table.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertIn("-\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", note_text)
        self.assertIn("Dunnett", note_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_index = review_markdown.index("**2.6.7.7B")
        next_title_index = review_markdown.index("2.6.7.8A", title_index)
        page101_markdown = review_markdown[title_index:next_title_index]
        self.assertIn("| \u65e5\u5242\u91cf(mg/kg) | 0 M | 0 F | 10 M | 10 F | 40 M | 40 F | 100 M | 100 F |", page101_markdown)
        self.assertNotIn("2.6.7.7B \u91cd\u590d\u7ed9\u836f\u6bd2\u6027 \u8bd5\u9a8c\u7f16\u53f7\uff1a94020(\u7eed) \u65e5\u5242\u91cf", page101_markdown)

    def test_pages102_to_104_genotoxicity_result_matrices_stitch_cross_page_continuations(self) -> None:
        page102_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 102
            and table.get("semantic_role") == "business_table"
        ]
        page102_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            for table in page102_tables
        ]
        parent102 = next(
            (
                table
                for table, text in zip(page102_tables, page102_texts)
                if "TA98" in text
                and "TA100" in text
                and "WP2uvrA" in text
                and "MM-180801" in text
            ),
            None,
        )
        self.assertIsNotNone(parent102, msg=page102_texts)
        projection102 = (parent102.get("semantic_projection_v2") or {}).get("genotoxicity_assay_matrix_projection", {})
        self.assertEqual(projection102.get("semantic_profile"), "genotoxicity_assay_matrix")
        self.assertEqual(projection102.get("assay_kind"), "bacterial_reverse_mutation_matrix")
        semantic102 = parent102.get("semantic_grid") or []
        self.assertEqual(
            semantic102[0],
            [
                "\u4ee3\u8c22\u6d3b\u5316",
                "\u4f9b\u8bd5\u54c1",
                "\u5242\u91cf\u6c34\u5e73",
                "TA98",
                "TA100",
                "TA1535",
                "TA1537",
                "WP2uvrA",
            ],
        )
        semantic102_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in semantic102
            if isinstance(row, list)
        )
        self.assertIn("\u65e0\u4ee3\u8c22\u6d3b\u5316 | MM-180801 | 312.5 | 24\u00b16 | 128\u00b111 | 12\u00b14 | 4\u00b12 | 14\u00b12", semantic102_text)
        self.assertIn("\u6709\u4ee3\u8c22\u6d3b\u5316 | DMSO | 100 \u00b5L/\u76bf | 27\u00b16 | 161\u00b112 | 12\u00b15 | 5\u00b11 | 21\u00b18", semantic102_text)
        self.assertNotIn("MM-180801 | 312.5 |  | 24\u00b16", semantic102_text)

        page103_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 103
            and table.get("semantic_role") == "business_table"
        ]
        page103_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            for table in page103_tables
        ]
        continuation103 = next(
            (
                table
                for table, text in zip(page103_tables, page103_texts)
                if "\u6c28\u57fa\u84bd" in text
                and "1552" in text
                and "366" in text
            ),
            None,
        )
        self.assertIsNotNone(continuation103, msg=page103_texts)
        self.assertTrue(continuation103.get("is_continuation"))
        self.assertEqual(continuation103.get("continued_from_table_id"), parent102.get("table_id"))
        note103 = " ".join(str(note.get("text", "")) for note in continuation103.get("note_blocks", []) or [])
        self.assertIn("\u6c89\u6dc0", note103)

        parent103 = next(
            (
                table
                for table, text in zip(page103_tables, page103_texts)
                if "\u7ec6\u80de\u6bd2\u6027" in text
                and "\u5e73\u5747\u7ec6\u80de\u7578\u53d8\u7387" in text
                and "\u591a\u500d\u4f53\u7ec6\u80de\u603b\u6570" in text
                and "\u4e1d\u88c2\u9709\u7d20" in text
            ),
            None,
        )
        self.assertIsNotNone(parent103, msg=page103_texts)
        self.assertFalse(parent103.get("is_continuation"))

        page104_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 104
            and table.get("semantic_role") == "business_table"
        ]
        page104_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
            for table in page104_tables
        ]
        continuation104 = next(
            (
                table
                for table, text in zip(page104_tables, page104_texts)
                if "\u73af\u78f7\u9170\u80fa" in text
                and "36.5**" in text
                and "0.66" in text
            ),
            None,
        )
        self.assertIsNotNone(continuation104, msg=page104_texts)
        self.assertTrue(continuation104.get("is_continuation"))
        self.assertEqual(continuation104.get("continued_from_table_id"), parent103.get("table_id"))
        note104 = " ".join(str(note.get("text", "")) for note in continuation104.get("note_blocks", []) or [])
        self.assertIn("Dunnett", note104)
        self.assertIn("\u7ec6\u80de\u6709\u4e1d\u5206\u88c2\u6307\u6570", note104)

        new_page104_table = next(
            (
                table
                for table, text in zip(page104_tables, page104_texts)
                if "\u5e73\u5747%PCE" in text
                and "\u5e73\u5747%MN-PCE" in text
                and "MM-180801" in text
            ),
            None,
        )
        self.assertIsNotNone(new_page104_table, msg=page104_texts)
        self.assertFalse(new_page104_table.get("is_continuation"))

    def test_page103_genotoxicity_continuation_title_and_chromosomal_aberration_schema(self) -> None:
        page102_parent = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 102
            and (table.get("semantic_projection_v2") or {})
            .get("genotoxicity_assay_matrix_projection", {})
            .get("assay_kind")
            == "bacterial_reverse_mutation_matrix"
        )
        page103_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 103
            and table.get("semantic_role") == "business_table"
        ]
        continuation = next(
            (
                table
                for table in page103_tables
                if table.get("continued_from_table_id") == page102_parent.get("table_id")
            ),
            None,
        )
        self.assertIsNotNone(continuation, msg=page103_tables)
        self.assertIn(
            continuation.get("table_id"),
            [str(item or "") for item in page102_parent.get("continued_to", []) or []],
        )
        continuation_projection = (continuation.get("semantic_projection_v2") or {}).get(
            "genotoxicity_assay_matrix_projection",
            {},
        )
        self.assertEqual(continuation_projection.get("semantic_profile"), "genotoxicity_assay_matrix")
        self.assertEqual(continuation_projection.get("assay_kind"), "bacterial_reverse_mutation_matrix")
        self.assertTrue(continuation_projection.get("continuation_schema_inherited"))
        page102_context = "\n".join(
            str(item.get("text") or "")
            for item in page102_parent.get("study_context_blocks", []) or []
            if isinstance(item, dict)
        )
        self.assertIn("独立试验次数：2", page102_context)
        page102_body_text = "\n".join(
            str(block.get("text") or "")
            for page in self.result["document_ast"]["pages"]
            if page["page"] == 102
            for block in page.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
        )
        self.assertNotIn("示例#1", page102_body_text)
        self.assertNotIn("独立试验次数：2", page102_body_text)
        self.assertNotIn("实验#1", page102_body_text)
        continuation_grid = continuation.get("semantic_grid") or []
        self.assertEqual(continuation_grid[0], page102_parent.get("semantic_grid", [])[0])
        continuation_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in continuation_grid
            if isinstance(row, list)
        )
        self.assertIn("\u6709\u4ee3\u8c22\u6d3b\u5316 | MM-180801 | 1250 | 33\u00b12 | 153\u00b113 | 13\u00b13 | 8\u00b12 | 18\u00b13", continuation_text)
        self.assertIn("2-\u6c28\u57fa\u84bd", continuation_text)
        self.assertIn("\u6c89\u6dc0", " ".join(str(note.get("text") or "") for note in continuation.get("note_blocks", []) or [] if isinstance(note, dict)))

        chromosomal = next(
            (
                table
                for table in page103_tables
                if (table.get("semantic_projection_v2") or {})
                .get("genotoxicity_assay_matrix_projection", {})
                .get("assay_kind")
                == "chromosomal_aberration_matrix"
            ),
            None,
        )
        self.assertIsNotNone(chromosomal, msg=page103_tables)
        self.assertIn("2.6.7.8B", str(chromosomal.get("title") or ""))
        chromosomal_projection = (chromosomal.get("semantic_projection_v2") or {}).get(
            "genotoxicity_assay_matrix_projection",
            {},
        )
        self.assertEqual(chromosomal_projection.get("logical_column_count"), 7)
        chromosomal_grid = chromosomal.get("semantic_grid") or []
        self.assertEqual(
            chromosomal_grid[0],
            [
                "\u4ee3\u8c22\u6d3b\u5316",
                "\u4f9b\u8bd5\u54c1",
                "\u6d53\u5ea6(\u00b5g/ml)",
                "\u7ec6\u80de\u6bd2\u6027a(%\u5bf9\u7167)",
                "\u5e73\u5747\u7ec6\u80de\u7578\u53d8\u7387%",
                "Abs/\u7ec6\u80de",
                "\u591a\u500d\u4f53\u7ec6\u80de\u603b\u6570",
            ],
        )
        chromosomal_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in chromosomal_grid
            if isinstance(row, list)
        )
        self.assertIn("\u65e0\u4ee3\u8c22\u6d3b\u5316 | DMSO | - | 100 | 2.0 | 0.02 | 4", chromosomal_text)
        self.assertIn("MM-180801 | 10 | 36 | 16.5** | 0.20 | 2", chromosomal_text)

        page103_ast = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 103)
        body_text = "\n".join(
            str(block.get("text") or "")
            for block in page103_ast.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
        )
        self.assertNotIn("2.6.7.8B \u9057\u4f20\u6bd2\u6027\uff1a\u4f53\u5916", body_text)
        self.assertNotIn("MM-180801 312.5", body_text)
        self.assertNotIn("示例#2", body_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title8a_index = review_markdown.index("**2.6.7.8A")
        title_index = review_markdown.index("**2.6.7.8B")
        page102_markdown = review_markdown[title8a_index:title_index]
        self.assertIn("独立试验次数：2", page102_markdown, msg=page102_markdown)
        self.assertNotIn("示例#1", page102_markdown, msg=page102_markdown)
        self.assertNotIn("实验#1", page102_markdown, msg=page102_markdown)
        self.assertNotIn("示例#2", page102_markdown, msg=page102_markdown)
        next_title_index = review_markdown.index("2.6.7.9", title_index)
        page103_markdown = review_markdown[title_index:next_title_index]
        preceding_title_window = review_markdown[max(0, title_index - 1600) : title_index]
        expected_context_fields = [
            "检测的诱导作用：染色体畸变",
            "平行培养物数量：2",
            "试验编号：96668",
            "GLP 依从性：是",
        ]
        table_header = (
            "| 代谢活化 | 供试品 | 浓度(µg/ml) | 细胞毒性a(%对照) | "
            "平均细胞畸变率% | Abs/细胞 | 多倍体细胞总数 |"
        )
        table_header_index = page103_markdown.index(table_header)
        metadata_region = page103_markdown[:table_header_index]
        for field in expected_context_fields:
            self.assertIn(field, metadata_region, msg=page103_markdown)
            if field != "GLP 依从性：是":
                self.assertNotIn(field, preceding_title_window, msg=preceding_title_window)
            self.assertEqual(page103_markdown.count(field), 1, msg=page103_markdown)
        self.assertIn(
            table_header,
            page103_markdown,
        )
        self.assertNotIn("| \u4ee3\u8c22\u6d3b\u5316 | \u4f9b\u8bd5\u54c1 | \u6d53\u5ea6 | \u7ec6\u80de\u6bd2\u6027a | \u5e73\u5747\u7ec6\u80de\u7578\u53d8\u7387 |  |  |  |  |", page103_markdown)

    def test_pages104_to_105_genotoxicity_assay_continuations_inherit_logical_schemas(self) -> None:
        chromosomal_parent = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 103
            and (table.get("semantic_projection_v2") or {})
            .get("genotoxicity_assay_matrix_projection", {})
            .get("assay_kind")
            == "chromosomal_aberration_matrix"
        )
        chromosomal_continuation = next(
            (
                table
                for table in self.result.get("table_asts", []) or []
                if int(table.get("page", 0) or 0) == 104
                and table.get("continued_from_table_id") == chromosomal_parent.get("table_id")
            ),
            None,
        )
        self.assertIsNotNone(chromosomal_continuation)
        chromosomal_projection = (chromosomal_continuation.get("semantic_projection_v2") or {}).get(
            "genotoxicity_assay_matrix_projection",
            {},
        )
        self.assertEqual(chromosomal_projection.get("assay_kind"), "chromosomal_aberration_matrix")
        self.assertTrue(chromosomal_projection.get("continuation_schema_inherited"))
        self.assertEqual(chromosomal_projection.get("logical_column_count"), 7)
        self.assertIn(
            chromosomal_continuation.get("table_id"),
            [str(item or "") for item in chromosomal_parent.get("continued_to", []) or []],
        )
        chromosomal_grid = chromosomal_continuation.get("semantic_grid") or []
        self.assertEqual(chromosomal_grid[0], chromosomal_parent.get("semantic_grid", [])[0])
        chromosomal_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in chromosomal_grid
            if isinstance(row, list)
        )
        self.assertIn("\u6709\u4ee3\u8c22\u6d3b\u5316 | MM-180801 | 200 | 43 | 34.0** | 0.66 | 3", chromosomal_text)
        self.assertIn("\u6709\u4ee3\u8c22\u6d3b\u5316 | \u73af\u78f7\u9170\u80fa | 4 | 68 | 36.5** | 0.63 | 6", chromosomal_text)
        self.assertIn("Dunnett", " ".join(str(note.get("text") or "") for note in chromosomal_continuation.get("note_blocks", []) or [] if isinstance(note, dict)))

        micronucleus_parent = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 104
            and "\u5e73\u5747%PCE" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("semantic_grid") or table.get("display_grid") or [])
                if isinstance(row, list)
            )
            and "MM-180801" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("semantic_grid") or table.get("display_grid") or [])
                if isinstance(row, list)
            )
        )
        micronucleus_projection = (micronucleus_parent.get("semantic_projection_v2") or {}).get(
            "genotoxicity_assay_matrix_projection",
            {},
        )
        self.assertEqual(micronucleus_projection.get("assay_kind"), "micronucleus_matrix")
        self.assertEqual(micronucleus_projection.get("logical_column_count"), 5)
        micronucleus_continuation = next(
            (
                table
                for table in self.result.get("table_asts", []) or []
                if int(table.get("page", 0) or 0) == 105
                and table.get("continued_from_table_id") == micronucleus_parent.get("table_id")
            ),
            None,
        )
        self.assertIsNotNone(micronucleus_continuation)
        micronucleus_cont_projection = (micronucleus_continuation.get("semantic_projection_v2") or {}).get(
            "genotoxicity_assay_matrix_projection",
            {},
        )
        self.assertEqual(micronucleus_cont_projection.get("assay_kind"), "micronucleus_matrix")
        self.assertTrue(micronucleus_cont_projection.get("continuation_schema_inherited"))
        self.assertEqual(micronucleus_cont_projection.get("logical_column_count"), 5)
        self.assertIn(
            micronucleus_continuation.get("table_id"),
            [str(item or "") for item in micronucleus_parent.get("continued_to", []) or []],
        )
        micronucleus_grid = micronucleus_continuation.get("semantic_grid") or []
        self.assertEqual(micronucleus_grid[0], micronucleus_parent.get("semantic_grid", [])[0])
        micronucleus_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in micronucleus_grid
            if isinstance(row, list)
        )
        self.assertIn("\u73af\u78f7\u9170\u80fa | 7 | 5M | 51\u00b12.3 | 2.49\u00b10.30**", micronucleus_text)
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        self.assertIn("| \u4f9b\u8bd5\u54c1 | \u5242\u91cf(mg/kg) | \u52a8\u7269\u6570\u91cf | \u5e73\u5747%PCE (\u00b1SD) | \u5e73\u5747%MN-PCE (\u00b1SD) |", review_markdown)

    def test_page105_genotoxicity_tables_are_single_rendering_owners_not_structure_templates(self) -> None:
        page105_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 105
        ]
        renderable_templates = [
            template
            for template in page105_templates
            if str(template.get("ownership_domain") or "") != "absorbed_by_business_table"
            and not str(template.get("template_profile") or "").startswith("absorbed_")
        ]
        renderable_template_text = "\n".join(
            "\n".join(str(row or "") for row in template.get("row_texts", []) or [])
            for template in renderable_templates
        )
        for duplicate_token in (
            "\u73af\u78f7\u9170\u80fa",
            "51\u00b12.3",
            "2.49\u00b10.30**",
            "Dunnett \u6c0f\u68c0\u9a8c",
            "\u7ec6\u80de\u6838",
            "\u7ec6\u80de\u8d28",
            "NGIR",
        ):
            self.assertNotIn(duplicate_token, renderable_template_text, msg=renderable_template_text)

        absorbed_templates = [
            template
            for template in page105_templates
            if str(template.get("ownership_domain") or "") == "absorbed_by_business_table"
        ]
        self.assertTrue(absorbed_templates, msg=page105_templates)
        absorbed_owner_ids = {
            str(template.get("absorbed_by_table_id") or "").strip()
            for template in absorbed_templates
            if str(template.get("absorbed_by_table_id") or "").strip()
        }
        absorbed_owner_ids.update(
            str(owner_id or "").strip()
            for template in absorbed_templates
            for owner_id in template.get("absorbed_by_table_ids", []) or []
            if str(owner_id or "").strip()
        )
        page105_table_ids = {
            str(table.get("table_id") or "").strip()
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 105
            and (table.get("semantic_projection_v2") or {}).get("genotoxicity_assay_matrix_projection")
        }
        self.assertTrue(page105_table_ids & absorbed_owner_ids, msg=absorbed_templates)

    def test_page105_genotoxicity_markdown_has_single_visible_title_and_table_row_owner(self) -> None:
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_marker = (
            "**2.6.7.9B \u9057\u4f20\u6bd2\u6027\uff1a\u4f53\u5185 "
            "\u62a5\u544a\u6807\u9898\uff1aMM-180801"
        )
        title_index = review_markdown.index(title_marker)
        row_marker = (
            "| \u73af\u78f7\u9170\u80fa | 7 | 5M | 51\u00b12.3 | "
            "2.49\u00b10.30** |"
        )
        row_index = review_markdown.index(row_marker)
        self.assertLess(row_index, title_index)
        next_section_index = review_markdown.index("2.6.7.10", title_index)
        page105_markdown = review_markdown[row_index:next_section_index]

        self.assertEqual(page105_markdown.count("\u62a5\u544a\u6807\u9898\uff1aMM-180801"), 1, msg=page105_markdown)
        self.assertEqual(page105_markdown.count(row_marker), 1, msg=page105_markdown)
        self.assertNotIn("\u7ed3\u6784\u6a21\u677f", page105_markdown)

    def test_page106_dmn_genotoxicity_continuation_renders_inside_page105_logical_table(self) -> None:
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_marker = (
            "**2.6.7.9B \u9057\u4f20\u6bd2\u6027\uff1a\u4f53\u5185 "
            "\u62a5\u544a\u6807\u9898\uff1aMM-180801"
        )
        title_index = review_markdown.index(title_marker)
        next_section_index = review_markdown.index("2.6.7.10", title_index)
        region = review_markdown[title_index:next_section_index]

        header_marker = (
            "| \u4f9b\u8bd5\u54c1 | \u5242\u91cf(mg/kg) | \u52a8\u7269\u6570\u91cf | "
            "\u65f6\u95f4(h) | \u7ec6\u80de\u6838 \u5e73\u5747\u503c\u00b1SD | "
            "\u7ec6\u80de\u8d28 \u5e73\u5747\u503c\u00b1SD | NG \u5e73\u5747\u503c\u00b1SD | "
            "%IR \u5e73\u5747\u503c\u00b1SD | NGIR"
        )
        dmn_row = (
            "| DMN | 10 | 3M | 2 | 10.7\u00b13.0 | 5.8\u00b11.0 | "
            "4.9\u00b12.1 | 41\u00b115 | 11.4\u00b10.4 |"
        )
        last_parent_row = "|  | 2000 | 3M | 16 | 2.7\u00b10.1 | 4.8\u00b10.3 | -2.1\u00b10.3 | 0\u00b10 | - |"

        self.assertEqual(region.count(dmn_row), 1, msg=region)
        self.assertEqual(region.count(header_marker), 1, msg=region)
        self.assertLess(region.index(last_parent_row), region.index(dmn_row), msg=region)
        between_last_parent_row_and_dmn = region[region.index(last_parent_row) : region.index(dmn_row)]
        self.assertNotIn(header_marker, between_last_parent_row_and_dmn, msg=region)

    def test_page105_genotoxicity_absorbed_study_template_fields_render_before_result_matrix(self) -> None:
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_marker = (
            "**2.6.7.9B \u9057\u4f20\u6bd2\u6027\uff1a\u4f53\u5185 "
            "\u62a5\u544a\u6807\u9898\uff1aMM-180801"
        )
        title_index = review_markdown.index(title_marker)
        next_section_index = review_markdown.index("2.6.7.10", title_index)
        region = review_markdown[title_index:next_section_index]
        table_header = "| \u4f9b\u8bd5\u54c1 | \u5242\u91cf(mg/kg) | \u52a8\u7269\u6570\u91cf | \u65f6\u95f4(h) |"

        expected_fields = [
            "\u68c0\u6d4b\u7684\u8bf1\u5bfc\u4f5c\u7528\uff1a\u7a0b\u5e8f\u5916DNA \u5408\u6210",
            "\u8bd5\u9a8c\u7f16\u53f7\uff1a51970",
            "\u79cd\u5c5e/\u54c1\u7cfb\uff1aWistar \u5927\u9f20",
            "CTD \u4e2d\u7684\u4f4d\u7f6e\uff1a\u7b2c11 \u5377\u7b2c502 \u9875",
            "\u6bcf\u53ea\u52a8\u7269\u5206\u6790\u7ec6\u80de\u6570\u91cf\uff1a100",
            "\u66b4\u9732\u7684\u8bc1\u636e\uff1a\u6bd2\u4ee3\u52a8\u529b\u5b66-\u89c1\u8bd5\u9a8c\u7f16\u53f794007",
        ]
        table_index = region.index(table_header)
        metadata_region = region[:table_index]
        for field in expected_fields:
            self.assertIn(field, metadata_region, msg=region)
            self.assertEqual(region.count(field), 1, msg=region)
        self.assertNotIn("\u7ed3\u6784\u6a21\u677f", metadata_region)

    def test_page104_genotoxicity_absorbed_template_title_promotes_to_table_title_once(self) -> None:
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_marker = (
            "**2.6.7.9A 遗传毒性：体内 "
            "报告标题：MM-180801：大鼠经口给药微核试验 "
            "供试品：曲醇钠**"
        )
        self.assertIn(title_marker, review_markdown)
        title_index = review_markdown.index(title_marker)
        next_title_index = review_markdown.index("**2.6.7.9B", title_index)
        region = review_markdown[title_index:next_title_index]
        table_header = "| 供试品 | 剂量(mg/kg) | 动物数量 | 平均%PCE (±SD) | 平均%MN-PCE (±SD) |"
        table_index = region.index(table_header)
        metadata_region = region[:table_index]

        self.assertEqual(region.count("2.6.7.9A 遗传毒性：体内"), 1, msg=region)
        self.assertNotIn("- 2.6.7.9A 遗传毒性：体内", region, msg=region)
        self.assertIn("检测的诱导作用：骨髓微核", metadata_region, msg=region)
        self.assertIn("试验编号：96683", metadata_region, msg=region)
        self.assertIn("| 环磷酰胺 | 7 | 5M | 51±2.3 | 2.49±0.30** |", region, msg=region)
        self.assertNotIn("结构模板", metadata_region, msg=region)

    def test_pages105_to_109_result_matrix_continuations_split_new_study_contexts(self) -> None:
        tables_by_page = {
            page: [
                table
                for table in self.result.get("table_asts", []) or []
                if int(table.get("page", 0) or 0) == page
                and table.get("semantic_role") == "business_table"
            ]
            for page in range(105, 110)
        }

        def table_text(table: dict) -> str:
            return "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
                if isinstance(row, list)
            )

        def semantic_table_text(table: dict) -> str:
            return "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (
                    table.get("semantic_display_grid")
                    or table.get("semantic_grid")
                    or table.get("display_grid")
                    or table.get("raw_grid")
                    or []
                )
                if isinstance(row, list)
            )

        texts_by_page = {
            page: [table_text(table) for table in tables]
            for page, tables in tables_by_page.items()
        }

        page105_dna_table = next(
            (
                table
                for table, text in zip(tables_by_page[105], texts_by_page[105])
                if "\u7ec6\u80de\u6838" in text
                and "\u7ec6\u80de\u8d28" in text
                and "%IR" in text
                and "NGIR" in text
            ),
            None,
        )
        self.assertIsNotNone(page105_dna_table, msg=texts_by_page[105])
        self.assertFalse(page105_dna_table.get("is_continuation"))

        page106_dna_continuation = next(
            (
                table
                for table, text in zip(tables_by_page[106], texts_by_page[106])
                if "DMN" in text
                and "10.7" in text
                and "11.4" in text
            ),
            None,
        )
        self.assertIsNotNone(page106_dna_continuation, msg=texts_by_page[106])
        self.assertTrue(page106_dna_continuation.get("is_continuation"))
        self.assertEqual(
            page106_dna_continuation.get("continued_from_table_id"),
            page105_dna_table.get("table_id"),
        )
        page106_dna_note_text = " ".join(
            str(note.get("text", ""))
            for note in page106_dna_continuation.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        for expected in ("\u7ec6\u80de\u6838=", "\u7ec6\u80de\u8d28=", "NG=", "%IR=", "NGIR="):
            self.assertIn(expected, page106_dna_note_text)
        self.assertNotIn("2.6.7.10", table_text(page106_dna_continuation))

        page106_carcinogenicity = next(
            (
                table
                for table, text in zip(tables_by_page[106], texts_by_page[106])
                if "\u65e5\u5242\u91cf(mg/kg)" in text.replace(" ", "")
                and "\u6027\u522b" in text
                and "\u6444\u98df\u91cf" in text
            ),
            None,
        )
        self.assertIsNotNone(page106_carcinogenicity, msg=texts_by_page[106])
        self.assertFalse(page106_carcinogenicity.get("is_continuation"))
        page106_carc_text = table_text(page106_carcinogenicity)
        self.assertNotIn("DMN", page106_carc_text)
        self.assertNotIn("\u7ec6\u80de\u6838=", page106_carc_text)
        self.assertNotIn("2.6.7.10", page106_carc_text)
        page106_carc_note_text = " ".join(
            str(note.get("text", ""))
            for note in page106_carcinogenicity.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertIn("Dunnett", page106_carc_note_text)
        self.assertIn("\u6765\u6e90\u4e8e\u8bd5\u9a8c\u7f16\u53f795013", page106_carc_note_text)
        self.assertIn("1 \u53ea\u7f3a\u5931\u5c0f\u9f20\u65e0\u6cd5\u8bc4\u4ef7", page106_carc_note_text)

        page107_tumor_table = next(
            (
                table
                for table, text in zip(tables_by_page[107], texts_by_page[107])
                if "\u53d1\u751f\u80bf\u7624\u75c5\u53d8" in text
                and "\u76ae\u80a4" in text
                and "\u80ba\u6ce1" in text
            ),
            None,
        )
        self.assertIsNotNone(page107_tumor_table, msg=texts_by_page[107])
        self.assertTrue(page107_tumor_table.get("is_continuation"))
        self.assertEqual(
            page107_tumor_table.get("continued_from_table_id"),
            page106_carcinogenicity.get("table_id"),
        )
        page107_note_text = " ".join(
            str(note.get("text", ""))
            for note in page107_tumor_table.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertNotIn("Dunnett", page107_note_text)
        self.assertNotIn("\u6765\u6e90\u4e8e\u8bd5\u9a8c\u7f16\u53f795013", page107_note_text)
        self.assertIn("\u8d8b\u52bf\u5206\u6790", page107_note_text)
        self.assertIn(
            "\u817a\u7624+\u764c | 15 | 10 | 11 | 12 | 15 | 9 | 13 | 5",
            semantic_table_text(page107_tumor_table),
        )

        page108_continuation = next(
            (
                table
                for table, text in zip(tables_by_page[108], texts_by_page[108])
                if "\u7eb5\u9694" in text
                and "\u777e\u4e38" in text
            ),
            None,
        )
        self.assertIsNotNone(page108_continuation, msg=texts_by_page[108])
        self.assertTrue(page108_continuation.get("is_continuation"))
        page108_note_text = " ".join(
            str(note.get("text", ""))
            for note in page108_continuation.get("note_blocks", []) or []
            if isinstance(note, dict)
        )
        self.assertNotIn("\u8d8b\u52bf\u5206\u6790", page108_note_text)
        page108_standalone_top_rows = [
            table
            for table, text in zip(tables_by_page[108], texts_by_page[108])
            if text.strip() == "\u817a\u7624+\u764c | 15 | 10 | 11 | 12 | 15 | 9 | 13 | 5"
        ]
        self.assertEqual(page108_standalone_top_rows, [])

        page109_reproductive = next(
            (
                table
                for table, text in zip(tables_by_page[109], texts_by_page[109])
                if "\u79cd\u5c5e/\u54c1\u7cfb" in text
                and "\u7ed9\u836f\u65b9\u6cd5" in text
                and "\u65b0\u897f\u5170\u5154" in text
            ),
            None,
        )
        self.assertIsNotNone(page109_reproductive, msg=texts_by_page[109])
        self.assertFalse(page109_reproductive.get("is_continuation"))
        page109_repro_text = table_text(page109_reproductive)
        self.assertNotIn("\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", page109_repro_text)
        self.assertNotIn("Fisher", page109_repro_text)

        document = {
            **self.result,
            "filename": self.sample_path.name,
            "source_path": str(self.sample_path),
            "source_type": "pdf",
        }
        markdown = _build_full_markdown([document], markdown_profile="ind-review")
        self.assertEqual(markdown.count("a - \u6765\u6e90\u4e8e\u8bd5\u9a8c\u7f16\u53f795013"), 1)
        self.assertNotIn(
            "| \u817a\u7624+\u764c | 15 | 10 | 11 | 12 | 15 | 9 | 13 | 5 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            markdown,
        )

    def test_pages107_to_113_table_owned_statistical_notes_do_not_leak_to_body(self) -> None:
        pages_by_number = {
            int(page.get("page", 0) or 0): page
            for page in self.result.get("document_ast", {}).get("pages", []) or []
        }
        for page_number, token in {
            107: "Dunnett",
            110: "Dunnett",
            111: "Dunnett",
            113: "Fisher",
        }.items():
            page_tables = [
                table
                for table in self.result.get("table_asts", []) or []
                if any(
                    token in str(note.get("text") or "")
                    and int(note.get("page", table.get("page", 0)) or 0) == page_number
                    for note in table.get("note_blocks", []) or []
                    if isinstance(note, dict)
                )
            ]
            self.assertTrue(page_tables, msg=f"page {page_number} has no table-owned {token} note")
            body_text = "\n".join(
                str(block.get("text") or "")
                for block in pages_by_number[page_number].get("blocks", []) or []
                if block.get("block_type") == "text"
                and str(block.get("unit_role") or "body") == "body"
            )
            self.assertNotIn(token, body_text, msg=body_text)
            self.assertNotIn("p<0.05", body_text, msg=body_text)
            self.assertNotIn("\u65e0\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c", body_text, msg=body_text)

        cross_page_fisher_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) in {108, 109}
            and any(
                "Fisher" in str(note.get("text") or "")
                and int(note.get("page", table.get("page", 0)) or 0) == 109
                for note in table.get("note_blocks", []) or []
                if isinstance(note, dict)
            )
        ]
        self.assertTrue(cross_page_fisher_tables)
        page109 = pages_by_number[109]
        page109_body_text = "\n".join(
            str(block.get("text") or "")
            for block in page109.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
        )
        self.assertNotIn("Fisher", page109_body_text, msg=page109_body_text)
        page109_new_study_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 109
            and "\u751f\u6b96\u6bd2\u6027" in str(table.get("title") or "")
            for row in (table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or [])
            if isinstance(row, list)
        )
        self.assertNotIn("Fisher", page109_new_study_text)

    def test_page109_reproductive_summary_visible_table_is_render_addressable(self) -> None:
        pages_by_number = {
            int(page.get("page", 0) or 0): page
            for page in self.result.get("document_ast", {}).get("pages", []) or []
        }
        page109 = pages_by_number[109]
        page109_table_blocks = [
            block
            for block in page109.get("blocks", []) or []
            if block.get("block_type") == "table"
            and "2.6.7.11" in str(block.get("title") or "")
            and str(block.get("table_id") or "").startswith("tbl_reclaimed_template_109_")
        ]
        self.assertTrue(page109_table_blocks, msg=page109.get("blocks", []))

        table_block = page109_table_blocks[0]
        table_id = str(table_block.get("table_id") or "")
        table_ast = next(
            (
                table
                for table in self.result.get("table_asts", []) or []
                if str(table.get("table_id") or "") == table_id
            ),
            None,
        )
        self.assertIsNotNone(table_ast)
        self.assertEqual(table_ast.get("semantic_role"), "business_table")

        page109_body_text = "\n".join(
            str(block.get("text") or "")
            for block in page109.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
            and str(block.get("visible_render_policy") or "") != "metadata_only"
        )
        self.assertNotIn("0\u3001500\u30011000\u3001 8 \u53ea\u598a\u5a20\u96cc\u6027", page109_body_text, msg=page109_body_text)
        self.assertNotIn("0\u30015\u300115\u300145 6 \u53ea\u975e\u598a\u5a20\u96cc\u6027", page109_body_text, msg=page109_body_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        page109_start = review_markdown.index(
            "2.6.7.11 \u751f\u6b96\u6bd2\u6027 \u975e\u5173\u952e\u8bd5\u9a8c \u4f9b\u8bd5\u54c1\uff1a\u66f2\u9187\u94a0"
        )
        page110_start = review_markdown.index("2.6.7.12", page109_start)
        page109_markdown = review_markdown[page109_start:page110_start]
        self.assertIn("2.6.7.11", page109_markdown)
        self.assertIn("Wistar", page109_markdown)
        self.assertIn("94201", page109_markdown)
        self.assertIn("97020", page109_markdown)
        self.assertEqual(page109_markdown.count("2.6.7.11"), 1, msg=page109_markdown)

    def test_page108_carcinogenicity_sparse_male_pathology_row_is_preserved(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 108
            and "肝脏：肝细胞肥大" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
                if isinstance(row, list)
            )
        )

        semantic_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in table.get("semantic_grid", []) or []
            if isinstance(row, list)
        )
        self.assertIn("睾丸：精子生成障碍", semantic_text)
        self.assertIn("睾丸：精子生成障碍 | 1 |  | 2 |  | 15* |  | 30** |", semantic_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        row_marker = "| 睾丸：精子生成障碍 | 1 |  | 2 |  | 15* |  | 30** |  |"
        self.assertIn(row_marker, review_markdown)
        self.assertEqual(review_markdown.count(row_marker), 1, msg=review_markdown)

    def test_ind_late_semantic_projections_record_evidence_coverage_audit(self) -> None:
        projection_keys = (
            "genotoxicity_assay_matrix_projection",
            "dose_response_result_panel_projection",
            "toxicology_summary_schema_projection",
        )
        late_projection_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if any(
                isinstance((table.get("semantic_projection_v2") or {}).get(key), dict)
                for key in projection_keys
            )
        ]
        self.assertTrue(late_projection_tables, msg="r2 should contain IND late semantic projections")

        for table in late_projection_tables:
            evidence_grid = table.get("evidence_grid_before_semantic_projection")
            self.assertIsInstance(evidence_grid, list, msg=table.get("table_id"))
            self.assertTrue(
                any(isinstance(row, list) and any(row) for row in evidence_grid),
                msg=table.get("table_id"),
            )
            projection = table.get("semantic_projection_v2") or {}
            for key in projection_keys:
                payload = projection.get(key)
                if not isinstance(payload, dict):
                    continue
                audit = payload.get("coverage_audit")
                self.assertIsInstance(audit, dict, msg=(table.get("table_id"), key, payload))
                self.assertIn(
                    audit.get("lossless_status"),
                    {"complete", "lossless_with_preserved_rows", "lossy_needs_review"},
                    msg=(table.get("table_id"), key, audit),
                )
                self.assertIn("source_data_row_count", audit, msg=(table.get("table_id"), key, audit))
                self.assertIn("projected_data_row_count", audit, msg=(table.get("table_id"), key, audit))
                self.assertIn("unprojected_data_rows", audit, msg=(table.get("table_id"), key, audit))

    def test_ind_late_semantic_projection_coverage_audit_has_no_unprojected_data_rows(self) -> None:
        projection_keys = (
            "genotoxicity_assay_matrix_projection",
            "dose_response_result_panel_projection",
            "toxicology_summary_schema_projection",
        )
        failures: list[tuple[str, int, str, dict]] = []
        for table in self.result.get("table_asts", []) or []:
            projection = table.get("semantic_projection_v2") or {}
            for key in projection_keys:
                payload = projection.get(key)
                if not isinstance(payload, dict):
                    continue
                audit = payload.get("coverage_audit") or {}
                if audit.get("unprojected_data_rows"):
                    failures.append((
                        str(table.get("table_id") or ""),
                        int(table.get("page", 0) or 0),
                        key,
                        audit,
                    ))
        self.assertEqual(failures, [])

    def test_ind_late_semantic_projection_keeps_display_grid_as_evidence_layer(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 114
            and (table.get("semantic_projection_v2") or {})
            .get("dose_response_result_panel_projection", {})
            .get("semantic_profile")
            == "dose_response_result_panel"
        )
        evidence_grid = table.get("evidence_grid_before_semantic_projection") or []
        display_grid = table.get("display_grid") or []
        semantic_grid = table.get("semantic_grid") or []

        self.assertGreater(len(evidence_grid), len(semantic_grid), msg=table.get("table_id"))
        self.assertEqual(
            len(display_grid),
            len(evidence_grid),
            msg="display_grid should remain the evidence layer after semantic projection",
        )
        display_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in display_grid
            if isinstance(row, list)
        )
        self.assertIn("Dunnett", display_text)
        self.assertIn("Column 2", display_text)

    def test_ind_architecture_converges_ownership_continuation_and_study_render_layers(self) -> None:
        architecture = self.result.get("ind_architecture")
        self.assertIsInstance(architecture, dict)
        self.assertEqual(architecture.get("version"), "ind_architecture_v1")

        ownership = architecture.get("ownership_arbitration")
        self.assertIsInstance(ownership, dict)
        self.assertEqual(ownership.get("exclusive_owner_violation_count"), 0, msg=ownership)
        self.assertEqual(ownership.get("visible_metadata_only_violation_count"), 0, msg=ownership)
        self.assertEqual(ownership.get("naked_template_label_body_count"), 0, msg=ownership)
        self.assertGreater(ownership.get("owned_text_block_count", 0), 0, msg=ownership)

        continuation = architecture.get("continuation_evidence_chains")
        self.assertIsInstance(continuation, dict)
        self.assertGreater(continuation.get("chain_count", 0), 0, msg=continuation)
        chain_profiles = {
            str(chain.get("semantic_profile") or "")
            for chain in continuation.get("chains", []) or []
            if isinstance(chain, dict)
        }
        self.assertIn("dose_response_result_panel", chain_profiles)
        self.assertIn("genotoxicity_assay_matrix", chain_profiles)
        for chain in continuation.get("chains", []) or []:
            self.assertTrue(chain.get("root_table_id"), msg=chain)
            self.assertGreaterEqual(len(chain.get("table_ids", []) or []), 2, msg=chain)
            self.assertGreaterEqual(len(chain.get("source_pages", []) or []), 2, msg=chain)

        study_layer = architecture.get("study_objects")
        self.assertIsInstance(study_layer, dict)
        self.assertGreater(study_layer.get("study_object_count", 0), 0, msg=study_layer)
        study_objects = study_layer.get("objects", []) or []
        self.assertTrue(
            any(
                str(study.get("title") or "").startswith("2.6.7.")
                and study.get("result_table_ids")
                and study.get("render_plan")
                for study in study_objects
                if isinstance(study, dict)
            ),
            msg=study_objects[:5],
        )

    def test_ind_architecture_exposes_normalized_ownership_claims(self) -> None:
        architecture = self.result.get("ind_architecture")
        self.assertIsInstance(architecture, dict)
        ownership = architecture.get("ownership_arbitration")
        self.assertIsInstance(ownership, dict)

        claims = ownership.get("claims")
        self.assertIsInstance(claims, list)
        self.assertGreater(ownership.get("claim_count", 0), 0, msg=ownership)
        self.assertGreater(len(claims), 0, msg=ownership)

        counts = ownership.get("claim_counts_by_type")
        self.assertIsInstance(counts, dict)
        for claim_type in (
            "owned_text_block",
            "object_title",
            "study_context",
            "table_note",
            "template_label",
            "absorbed_structure_template",
        ):
            self.assertGreater(counts.get(claim_type, 0), 0, msg=counts)

        for claim in claims:
            self.assertIsInstance(claim, dict)
            self.assertTrue(str(claim.get("owner_type") or "").strip(), msg=claim)
            self.assertTrue(str(claim.get("owner_id") or "").strip(), msg=claim)
            self.assertTrue(str(claim.get("claim_type") or "").strip(), msg=claim)
            self.assertTrue(
                claim.get("source_block_ids") or claim.get("source_object_ids"),
                msg=claim,
            )
            self.assertTrue(str(claim.get("reason") or "").strip(), msg=claim)

    def test_ind_architecture_exposes_continuation_chain_evidence_claims(self) -> None:
        architecture = self.result.get("ind_architecture")
        self.assertIsInstance(architecture, dict)
        continuation = architecture.get("continuation_evidence_chains")
        self.assertIsInstance(continuation, dict)

        chains = continuation.get("chains")
        self.assertIsInstance(chains, list)
        self.assertGreater(continuation.get("evidence_claim_count", 0), 0, msg=continuation)
        aggregate_counts = continuation.get("claim_counts_by_type")
        self.assertIsInstance(aggregate_counts, dict)
        for claim_type in (
            "continuation_link",
            "inherited_schema",
            "continuation_note",
        ):
            self.assertGreater(aggregate_counts.get(claim_type, 0), 0, msg=aggregate_counts)

        for chain in chains:
            self.assertIsInstance(chain, dict)
            claims = chain.get("evidence_claims")
            self.assertIsInstance(claims, list, msg=chain)
            self.assertGreater(len(claims), 0, msg=chain)
            counts = chain.get("claim_counts_by_type")
            self.assertIsInstance(counts, dict, msg=chain)
            self.assertGreater(counts.get("continuation_link", 0), 0, msg=chain)
            for claim in claims:
                self.assertTrue(str(claim.get("chain_id") or "").strip(), msg=claim)
                self.assertTrue(str(claim.get("claim_type") or "").strip(), msg=claim)
                self.assertTrue(str(claim.get("table_id") or "").strip(), msg=claim)
                self.assertTrue(str(claim.get("reason") or "").strip(), msg=claim)

        dose_chain = next(
            chain
            for chain in chains
            if chain.get("root_table_id") == "tbl_051"
        )
        self.assertEqual(dose_chain.get("source_pages"), [106, 107, 108])
        dose_counts = dose_chain.get("claim_counts_by_type") or {}
        self.assertGreaterEqual(dose_counts.get("continuation_link", 0), 2, msg=dose_chain)
        self.assertGreaterEqual(dose_counts.get("inherited_schema", 0), 2, msg=dose_chain)
        self.assertGreater(dose_counts.get("continuation_note", 0), 0, msg=dose_chain)

    def test_ind_architecture_study_render_plan_has_source_aware_components(self) -> None:
        architecture = self.result.get("ind_architecture")
        self.assertIsInstance(architecture, dict)
        study_layer = architecture.get("study_objects")
        self.assertIsInstance(study_layer, dict)
        objects = study_layer.get("objects")
        self.assertIsInstance(objects, list)

        source_aware_count = 0
        suppressed_source_count = 0
        for study in objects:
            self.assertIsInstance(study, dict)
            render_plan = study.get("render_plan")
            self.assertIsInstance(render_plan, dict, msg=study)
            components = render_plan.get("components")
            self.assertIsInstance(components, list, msg=study)
            self.assertGreater(len(components), 0, msg=study)
            for component in components:
                self.assertTrue(str(component.get("component_id") or "").strip(), msg=component)
                self.assertTrue(str(component.get("type") or "").strip(), msg=component)
                self.assertIn("source_pages", component, msg=component)
                self.assertIn("owner_table_ids", component, msg=component)
                if component.get("source_block_ids") or component.get("source_object_ids"):
                    source_aware_count += 1
            suppressed = [
                component
                for component in components
                if component.get("type") == "suppressed_sources"
            ]
            if suppressed:
                suppressed_source_count += sum(
                    len(component.get("source_object_ids", []) or [])
                    + len(component.get("source_block_ids", []) or [])
                    for component in suppressed
                )

        self.assertGreater(source_aware_count, 0, msg=objects[:3])
        self.assertGreater(suppressed_source_count, 0, msg=objects)

        genotox = next(
            study
            for study in objects
            if "2.6.7.8B" in str(study.get("title") or "")
        )
        components_by_type = {
            str(component.get("type") or ""): component
            for component in genotox.get("render_plan", {}).get("components", []) or []
        }
        self.assertIn("title", components_by_type)
        self.assertIn("study_context", components_by_type)
        self.assertIn("result_tables", components_by_type)
        self.assertIn("notes", components_by_type)
        self.assertIn("suppressed_sources", components_by_type)
        self.assertIn("txt_p103_017", components_by_type["title"].get("source_block_ids", []) or [])
        self.assertIn(
            "continuation_chain::tbl_recovered_103_01",
            components_by_type["result_tables"].get("continuation_chain_ids", []) or [],
        )
        self.assertIn(
            "structure_template_053",
            components_by_type["suppressed_sources"].get("source_object_ids", []) or [],
        )

    def test_ind_architecture_study_titles_bind_to_source_objects_when_text_block_is_absent(self) -> None:
        architecture = self.result.get("ind_architecture")
        self.assertIsInstance(architecture, dict)
        objects = (architecture.get("study_objects") or {}).get("objects")
        self.assertIsInstance(objects, list)

        def title_component(marker: str) -> dict:
            study = next(
                item
                for item in objects
                if marker in str(item.get("title") or "")
            )
            component = next(
                component
                for component in study.get("render_plan", {}).get("components", []) or []
                if component.get("type") == "title"
            )
            self.assertEqual(component.get("source_policy"), "table_title_source_binding")
            self.assertTrue(component.get("source_object_ids"), msg=component)
            return component

        dna_title = title_component("2.6.7.9B")
        self.assertIn("structure_template_054", dna_title.get("source_object_ids", []) or [])
        self.assertIn("preceding_ctd_study_title_bound_to_genotoxicity_table", dna_title.get("source_reasons", []) or [])

        fertility_title = title_component("2.6.7.12")
        self.assertIn("structure_template_057", fertility_title.get("source_object_ids", []) or [])
        self.assertIn("absorbed_structure_template_ids", fertility_title.get("source_reasons", []) or [])

    def test_ind_architecture_exposes_render_consumption_audit(self) -> None:
        architecture = self.result.get("ind_architecture")
        self.assertIsInstance(architecture, dict)
        audit = architecture.get("render_consumption_audit")
        self.assertIsInstance(audit, dict)

        self.assertEqual(audit.get("visible_block_source_duplicate_count"), 0, msg=audit)
        self.assertEqual(audit.get("visible_block_source_duplicate_violations"), [], msg=audit)
        self.assertGreater(audit.get("visible_component_count", 0), 0, msg=audit)
        self.assertGreater(audit.get("suppressed_source_declaration_count", 0), 0, msg=audit)
        self.assertGreater(audit.get("shared_object_source_count", 0), 0, msg=audit)
        self.assertEqual(
            audit.get("block_source_uniqueness_policy"),
            "source_block_ids_must_have_at_most_one_visible_render_consumer",
        )
        self.assertEqual(
            audit.get("object_source_policy"),
            "source_object_ids_are_composite_evidence_and_may_support_multiple_components",
        )

    def test_ind_review_markdown_records_render_plan_visibility_audit(self) -> None:
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        self.assertIn("2.6.7.9B", review_markdown)

        audit = (self.result.get("metadata") or {}).get("ind_review_render_audit")
        self.assertIsInstance(audit, dict)
        self.assertEqual(audit.get("profile"), "ind-review")
        self.assertEqual(audit.get("architecture_visible_block_source_duplicate_count"), 0, msg=audit)
        self.assertEqual(audit.get("study_title_duplicate_count"), 0, msg=audit)
        self.assertGreater(audit.get("study_title_checked_count", 0), 0, msg=audit)
        self.assertGreater(audit.get("study_title_checked_count", 0), 12, msg=audit)
        self.assertEqual(audit.get("study_title_skipped_count", 0), 0, msg=audit)
        skipped = audit.get("study_title_skipped") or []
        self.assertIsInstance(skipped, list, msg=audit)
        self.assertEqual(skipped, [], msg=audit)
        self.assertEqual(audit.get("heading_like_duplicate_unresolved_count"), 0, msg=audit)
        allowed_heading_duplicates = audit.get("heading_like_duplicate_allowed") or []
        self.assertTrue(
            any(
                "2.6.5.5" in str(item.get("title") or "")
                and item.get("reason") in {"structure_template_variant", "populated_study_metadata_variant"}
                for item in allowed_heading_duplicates
                if isinstance(item, dict)
            ),
            msg=audit,
        )
        non_scaffold_duplicates = [
            item
            for item in allowed_heading_duplicates
            if isinstance(item, dict)
            and item.get("reason")
            not in {"document_scaffold", "guidance_template_example_continuation_heading"}
        ]
        self.assertTrue(non_scaffold_duplicates, msg=audit)
        for item in non_scaffold_duplicates:
            self.assertIsInstance(item.get("source_objects"), list, msg=item)
            self.assertGreaterEqual(len(item.get("source_objects") or []), 2, msg=item)
            self.assertIsInstance(item.get("pages"), list, msg=item)
            self.assertIsInstance(item.get("template_profiles"), list, msg=item)
            self.assertIsInstance(item.get("ownership_domains"), list, msg=item)
            self.assertTrue(item.get("template_profiles"), msg=item)
            self.assertTrue(item.get("ownership_domains"), msg=item)

        blank_form_duplicate = next(
            item
            for item in non_scaffold_duplicates
            if item.get("reason") == "structure_template_variant"
            and "2.6.5.5" in str(item.get("title") or "")
        )
        self.assertEqual(blank_form_duplicate.get("pages"), [35, 36], msg=blank_form_duplicate)
        self.assertEqual(blank_form_duplicate.get("template_profiles"), ["tabular_form_template"], msg=blank_form_duplicate)
        self.assertEqual(blank_form_duplicate.get("ownership_domains"), ["template_form"], msg=blank_form_duplicate)

        populated_duplicate = next(
            item
            for item in non_scaffold_duplicates
            if item.get("reason") == "populated_study_metadata_variant"
            and "2.6.5.5" in str(item.get("title") or "")
        )
        self.assertEqual(populated_duplicate.get("pages"), [81, 82], msg=populated_duplicate)
        self.assertEqual(populated_duplicate.get("template_profiles"), ["populated_study_metadata"], msg=populated_duplicate)
        self.assertEqual(populated_duplicate.get("ownership_domains"), ["study_context"], msg=populated_duplicate)
        self.assertTrue(
            any(
                item.get("reason") == "guidance_template_example_continuation_heading"
                and "2.6.7.14" in str(item.get("title") or "")
                and item.get("document_kind") == "guidance_template_example"
                and item.get("allowance_basis") == "guidance_template_example_document"
                for item in allowed_heading_duplicates
                if isinstance(item, dict)
            ),
            msg=audit,
        )
        self.assertEqual(
            _normalize_ind_review_visibility_text(review_markdown).count(
                _normalize_ind_review_visibility_text(
                    "2.6.7.14 "
                    "生殖毒性 "
                    "试验编号：95201(续)"
                )
            ),
            1,
            msg="continuation panel titles should be visible once per repeated local continuation label",
        )
        self.assertEqual(
            audit.get("study_title_eligibility_policy"),
            "only_source_aware_title_components_are_hard_checked",
        )
        self.assertIn(
            "render_plan_title_components_are_visible_at_most_once",
            audit.get("policies", []) or [],
        )

    def test_real_ind_repeated_continuation_heading_remains_audit_risk_without_evidence(self) -> None:
        title = "2.6.7.14 (1)生殖毒性 试验编号(续)"
        markdown = "\n".join(
            [
                "# IND Review Markdown",
                "",
                "## real-ind-submission.pdf",
                "",
                f"###### {title}",
                "",
                "真实申报材料中的第一个续表页。",
                "",
                f"###### {title}",
                "",
                "真实申报材料中的第二个同名续表页。",
            ]
        )
        document = {
            "filename": "real-ind-submission.pdf",
            "metadata": {"parser_hint": "pdf", "document_kind": "regulatory_submission"},
            "structure_templates": [],
            "text": "真实IND申报资料。2.6.7.14 生殖毒性研究报告。",
        }

        audit = _build_ind_review_heading_like_duplicate_audit(document, markdown)

        self.assertEqual(audit.get("heading_like_duplicate_allowed_count"), 0, msg=audit)
        self.assertEqual(audit.get("heading_like_duplicate_unresolved_count"), 1, msg=audit)
        unresolved = audit.get("heading_like_duplicate_unresolved") or []
        self.assertEqual(unresolved[0].get("title"), title, msg=audit)
        self.assertEqual(unresolved[0].get("reason"), "requires_review", msg=audit)

    def test_guidance_template_repeated_continuation_heading_is_allowed_with_document_kind_evidence(self) -> None:
        title = "2.6.7.14 (1)生殖毒性 试验编号(续)"
        markdown = "\n".join(
            [
                "# IND Review Markdown",
                "",
                "## M4S(R2)-guidance-template.pdf",
                "",
                "M4S(R2) 人用药物注册通用技术文档：安全性部分",
                "",
                "附录B 非临床列表总结-模板 示例",
                "",
                "供试品：(1) 供试品：(2) 报告标题： 供试品：",
                "",
                f"###### {title}",
                "",
                "模板说明页中的第一个续表标题。",
                "",
                f"###### {title}",
                "",
                "模板说明页中的第二个续表标题。",
            ]
        )
        document = {
            "filename": "M4S(R2)-guidance-template.pdf",
            "metadata": {"parser_hint": "pdf"},
            "structure_templates": [],
            "text": (
                "ICH M4S(R2) 指导原则 附录B 非临床列表总结-模板 示例 "
                "供试品：(1) 供试品：(2) 报告标题： 供试品："
            ),
        }

        audit = _build_ind_review_heading_like_duplicate_audit(document, markdown)

        self.assertEqual(audit.get("heading_like_duplicate_unresolved_count"), 0, msg=audit)
        allowed = audit.get("heading_like_duplicate_allowed") or []
        continuation = next(
            item
            for item in allowed
            if isinstance(item, dict) and item.get("title") == title
        )
        self.assertEqual(continuation.get("reason"), "guidance_template_example_continuation_heading")
        self.assertEqual(continuation.get("document_kind"), "guidance_template_example")
        self.assertEqual(continuation.get("allowance_basis"), "guidance_template_example_document")
        self.assertIsInstance(continuation.get("continuation_evidence"), dict)

    def test_page110_dose_response_table_renders_owned_study_context_once(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 110
            and "2.6.7.12" in str(table.get("title") or "")
            and (table.get("semantic_projection_v2") or {})
            .get("dose_response_result_panel_projection", {})
            .get("semantic_profile")
            == "dose_response_result_panel"
        )
        context_text = " ".join(
            str(item.get("text") or "")
            for item in table.get("study_context_blocks", []) or []
            if isinstance(item, dict)
        )
        self.assertIn("\u4e0eICH 4.1.1 \u76f8\u4f3c\u7684\u7814\u7a76\u8bbe\u8ba1", context_text)
        self.assertIn("F0 \u96c4\u6027\uff1a100 mg/kg", context_text)

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_marker = "**2.6.7.12 \u751f\u6b96\u6bd2\u6027- \u62a5\u544a\u6807\u9898\uff1aMM-180801"
        title_index = review_markdown.index(title_marker)
        next_table_index = review_markdown.index("| \u65e5\u5242\u91cf(mg/kg)", title_index)
        page110_context_markdown = review_markdown[title_index:next_table_index]
        self.assertIn("\u4e0eICH 4.1.1 \u76f8\u4f3c\u7684\u7814\u7a76\u8bbe\u8ba1", page110_context_markdown)
        self.assertIn("F0 \u96c4\u6027\uff1a100 mg/kg", page110_context_markdown)
        self.assertNotIn("\u7ed3\u6784\u6a21\u677f", page110_context_markdown)
        self.assertEqual(page110_context_markdown.count("F0 \u96c4\u6027\uff1a100 mg/kg"), 1, msg=page110_context_markdown)
        self.assertEqual(review_markdown.count(title_marker), 1, msg=page110_context_markdown)

    def test_page114_perinatal_panel_absorbs_populated_metadata_template(self) -> None:
        page114_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 114
            and (
                "2.6.7.14" in str(template.get("title") or "")
                or str(template.get("absorbed_by_table_id") or "")
            )
        ]
        self.assertTrue(page114_templates)
        absorbed_templates = [
            template
            for template in page114_templates
            if str(template.get("ownership_domain") or "") == "absorbed_by_business_table"
            and str(template.get("absorbed_by_table_id") or "") == "tbl_059"
        ]
        self.assertTrue(absorbed_templates, msg=page114_templates)
        absorbed = absorbed_templates[0]
        self.assertEqual(absorbed.get("visible_render_policy"), "metadata_only")
        self.assertEqual(str(absorbed.get("title") or ""), "")
        self.assertEqual(absorbed.get("row_texts") or [], [])

        page114 = next(
            page
            for page in self.result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page", 0) or 0) == 114
        )
        visible_structure_templates = [
            block
            for block in page114.get("blocks", []) or []
            if block.get("block_type") == "structure_template"
            and str(block.get("visible_render_policy") or "") != "metadata_only"
            and str(block.get("ownership_domain") or "") != "absorbed_by_business_table"
        ]
        self.assertEqual(visible_structure_templates, [])

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        title_marker = "**2.6.7.14 \u751f\u6b96\u6bd2\u6027- \u62a5\u544a\u6807\u9898\uff1aMM-180801"
        title_index = review_markdown.index(title_marker)
        next_title_index = review_markdown.index("**2.6.7.14 \u751f\u6b96\u6bd2\u6027 \u8bd5\u9a8c\u7f16\u53f7", title_index + 1)
        page114_markdown = review_markdown[title_index:next_title_index]
        self.assertNotIn("\u7ed3\u6784\u6a21\u677f", page114_markdown)
        self.assertEqual(page114_markdown.count(title_marker), 1, msg=page114_markdown)
        self.assertEqual(page114_markdown.count("\u56f4\u4ea7\u671f\u6bd2\u6027\uff0c\u5305\u62ec\u6bcd\u4f53\u529f\u80fd"), 1, msg=page114_markdown)
        self.assertEqual(page114_markdown.count("F0 \u96cc\u6027\uff1a7.5 mg/kg"), 1, msg=page114_markdown)
        self.assertIn("| \u65e5\u5242\u91cf(mg/kg) | 0(\u5bf9\u7167) | 7.5 | 75 | 750 |", page114_markdown)
        self.assertIn("F0\u96cc\u6027\uff1a\u6bd2\u4ee3\u52a8\u529b\u5b66\uff1aAUC", page114_markdown)

    def test_page114_coverage_audit_ignores_headers_and_statistical_notes(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 114
            and (table.get("semantic_projection_v2") or {})
            .get("dose_response_result_panel_projection", {})
            .get("semantic_profile")
            == "dose_response_result_panel"
        )
        audit = (
            (table.get("semantic_projection_v2") or {})
            .get("dose_response_result_panel_projection", {})
            .get("coverage_audit", {})
        )
        self.assertNotEqual(audit.get("lossless_status"), "lossy_needs_review", msg=audit)
        unprojected_text = "\n".join(
            str(row.get("text") or "")
            for row in audit.get("unprojected_data_rows", []) or []
            if isinstance(row, dict)
        )
        self.assertNotIn("Dunnett", unprojected_text)
        self.assertNotIn("Kruskal-Wallis", unprojected_text)
        self.assertNotIn("Column 2", unprojected_text)

    def test_page104_genotoxicity_audit_matches_carry_forward_semantic_rows(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 104
            and (table.get("semantic_projection_v2") or {})
            .get("genotoxicity_assay_matrix_projection", {})
            .get("assay_kind")
            == "micronucleus_matrix"
        )
        audit = (
            (table.get("semantic_projection_v2") or {})
            .get("genotoxicity_assay_matrix_projection", {})
            .get("coverage_audit", {})
        )
        unprojected_text = "\n".join(
            str(row.get("text") or "")
            for row in audit.get("unprojected_data_rows", []) or []
            if isinstance(row, dict)
        )
        self.assertNotIn("20 5M 49", unprojected_text, msg=audit)
        self.assertNotIn("200 5M 50", unprojected_text, msg=audit)
        self.assertNotIn("2000 3M 31", unprojected_text, msg=audit)

    def test_bacterial_reverse_mutation_audit_ignores_headers_and_cross_row_merges(self) -> None:
        bacterial_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if (table.get("semantic_projection_v2") or {})
            .get("genotoxicity_assay_matrix_projection", {})
            .get("assay_kind")
            == "bacterial_reverse_mutation_matrix"
        ]
        self.assertTrue(bacterial_tables)

        for table in bacterial_tables:
            audit = (
                (table.get("semantic_projection_v2") or {})
                .get("genotoxicity_assay_matrix_projection", {})
                .get("coverage_audit", {})
            )
            unprojected_text = "\n".join(
                str(row.get("text") or "")
                for row in audit.get("unprojected_data_rows", []) or []
                if isinstance(row, dict)
            )
            self.assertNotEqual(audit.get("lossless_status"), "lossy_needs_review", msg=audit)
            self.assertNotIn("\u5242\u91cf\u6c34\u5e73", unprojected_text, msg=audit)
            self.assertNotIn("2-\u6c28\u57fa", unprojected_text, msg=audit)

    def test_page102_bacterial_reverse_mutation_sparse_positive_controls_are_preserved(self) -> None:
        table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 102
            and (table.get("semantic_projection_v2") or {})
            .get("genotoxicity_assay_matrix_projection", {})
            .get("assay_kind")
            == "bacterial_reverse_mutation_matrix"
        )
        semantic_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in table.get("semantic_grid", []) or []
            if isinstance(row, list)
        )
        self.assertIn("2-\u785d\u57fa\u82b4 | 2 | 696", semantic_text)
        self.assertIn("\u53e0\u6c2e\u5316\u94a0 | 1 |  | 542 | 468", semantic_text)
        self.assertIn("9-\u6c28\u57fa", semantic_text)
        self.assertIn("| 100 |  |  |  | 515", semantic_text)
        self.assertIn("MMS | 2.5", semantic_text)
        self.assertIn("| 573", semantic_text)

        audit = (
            (table.get("semantic_projection_v2") or {})
            .get("genotoxicity_assay_matrix_projection", {})
            .get("coverage_audit", {})
        )
        unprojected_text = "\n".join(
            str(row.get("text") or "")
            for row in audit.get("unprojected_data_rows", []) or []
            if isinstance(row, dict)
        )
        self.assertNotIn("2-\u785d\u57fa\u82b4", unprojected_text, msg=audit)
        self.assertNotIn("\u53e0\u6c2e\u5316\u94a0", unprojected_text, msg=audit)
        self.assertNotIn("9-\u6c28\u57fa\u5429\u5576", unprojected_text, msg=audit)
        self.assertNotIn("MMS", unprojected_text, msg=audit)

    def test_genotoxicity_result_matrix_composites_do_not_leak_cell_text_to_body(self) -> None:
        pages_by_number = {
            int(page.get("page", 0) or 0): page
            for page in self.result.get("document_ast", {}).get("pages", []) or []
        }

        def page_body_text(page_number: int) -> str:
            return "\n".join(
                str(block.get("text") or "")
                for block in pages_by_number[page_number].get("blocks", []) or []
                if block.get("block_type") == "text"
                and str(block.get("unit_role") or "body") == "body"
            )

        for page_number, table_token, leaked_tokens in (
            (102, "TA1537", ("TA98", "24\u00b19", "MM-180801 312.5", "\u5242\u91cf\u6c34\u5e73")),
            (103, "\u5e73\u5747\u7ec6\u80de\u7578\u53d8\u7387", ("\u7ec6\u80de\u6bd2\u6027a", "16.5**", "Abs/\u7ec6\u80de")),
        ):
            page = pages_by_number[page_number]
            blocks = page.get("blocks", []) or []
            matrix_table_index = next(
                index
                for index, block in enumerate(blocks)
                if block.get("block_type") == "table"
                and table_token in "\n".join(
                    " | ".join(str(cell or "") for cell in row)
                    for row in (block.get("semantic_grid") or block.get("display_grid") or block.get("raw_grid") or [])
                    if isinstance(row, list)
                )
            )
            matrix_table = blocks[matrix_table_index]
            study_context_text = "\n".join(
                str(item.get("text") or "")
                for item in matrix_table.get("study_context_blocks", []) or []
                if isinstance(item, dict)
            )
            self.assertIn("\u68c0\u6d4b\u7684\u8bf1\u5bfc\u4f5c\u7528", study_context_text)
            body_text = page_body_text(page_number)
            self.assertNotIn("2.6.7.8", body_text)
            for token in leaked_tokens:
                self.assertNotIn(token, body_text, msg=body_text)

        page101_body = page_body_text(101)
        self.assertNotRegex(page101_body, r"(?m)^-$")
        page108_body = page_body_text(108)
        for token in ("\u817a\u7624+\u764c", "15", "10", "11", "12"):
            self.assertNotRegex(page108_body, rf"(?m)^{re.escape(token)}$")

    def test_ind_review_markdown_renders_page115_116_reproductive_continuation_tables(self) -> None:
        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        page115_heading = "2.6.7.14 \u751f\u6b96\u6bd2\u6027 \u8bd5\u9a8c\u7f16\u53f7\uff1a95201(\u7eed)"
        page117_heading = "2.6.7.17 \u5176\u4ed6\u6bd2\u6027\u8bd5\u9a8c"
        page115_index = review_markdown.index(page115_heading)
        page117_index = review_markdown.index(page117_heading, page115_index)
        region = review_markdown[page115_index:page117_index]

        self.assertEqual(region.count(page115_heading), 1)
        self.assertIn("| \u65e5\u5242\u91cf(mg/kg) | 0(\u5bf9\u7167) | 7.5 | 75 | 750 |", region)
        self.assertIn("| F1\u4ee3\u4ed4\u9f20\uff1a\u8bc4\u4ef7\u7684\u7a9d\u6570 | 23 | 21 | 22 | 15 |", region)
        self.assertIn("| F1 \u96cc\u6027\uff1a\u79bb\u4e73\u540e\u8bc4\u4ef7\u52a8\u7269\u6570 | 23 | 21 | 22 | 23 |", region)
        self.assertIn("| F2\u4ed4\u9f20\uff1a\u5e73\u5747\u6d3b\u80ce\u6570/\u7a9d | 15.0 | 14.9 | 13.6 | 14.4 |", region)
        self.assertNotIn("null | null | null", region)

    def test_page105_dna_repair_matrix_projects_clean_logical_header(self) -> None:
        page105_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 105
            and table.get("semantic_role") == "business_table"
        ]
        dna_table = next(
            (
                table
                for table in page105_tables
                if "%IR" in "\n".join(
                    " | ".join(str(cell or "") for cell in row)
                    for row in (table.get("semantic_grid", []) or table.get("display_grid", []) or [])
                    if isinstance(row, list)
                )
                and "NGIR" in "\n".join(
                    " | ".join(str(cell or "") for cell in row)
                    for row in (table.get("semantic_grid", []) or table.get("display_grid", []) or [])
                    if isinstance(row, list)
                )
            ),
            None,
        )
        self.assertIsNotNone(dna_table, msg=page105_tables)
        projection = (dna_table.get("semantic_projection_v2") or {}).get(
            "genotoxicity_assay_matrix_projection",
            {},
        )
        self.assertEqual(
            projection.get("semantic_profile"),
            "genotoxicity_assay_matrix",
            msg=dna_table.get("semantic_projection_v2"),
        )
        self.assertEqual(projection.get("assay_kind"), "dna_repair_matrix")
        expected_header = [
            "\u4f9b\u8bd5\u54c1",
            "\u5242\u91cf(mg/kg)",
            "\u52a8\u7269\u6570\u91cf",
            "\u65f6\u95f4(h)",
            "\u7ec6\u80de\u6838 \u5e73\u5747\u503c\u00b1SD",
            "\u7ec6\u80de\u8d28 \u5e73\u5747\u503c\u00b1SD",
            "NG \u5e73\u5747\u503c\u00b1SD",
            "%IR \u5e73\u5747\u503c\u00b1SD",
            "NGIR \u5e73\u5747\u503c\u00b1SD",
        ]
        self.assertEqual(dna_table.get("semantic_grid", [])[0], expected_header)
        header_text = " ".join(str(cell or "") for cell in dna_table.get("semantic_grid", [[]])[0])
        self.assertNotIn("\u5242\u91cf \u4f9b\u8bd5\u54c1", header_text)
        self.assertNotIn("\u52a8\u7269\u6570 \u5242\u91cf", header_text)
        title_text = _clean_text_for_test(str(dna_table.get("title") or ""))
        self.assertIn("2.6.7.9B", title_text)
        self.assertIn("DNA", title_text)
        self.assertIn("\u635f\u4f24\u4fee\u590d\u8bd5\u9a8c", title_text)

    def test_pages106_to_116_dose_response_panels_project_logical_schema_and_inherit_continuations(self) -> None:
        def page_tables(page: int) -> list[dict]:
            return [
                table
                for table in self.result.get("table_asts", []) or []
                if int(table.get("page", 0) or 0) == page
                and table.get("semantic_role") == "business_table"
            ]

        def grid_text(table: dict) -> str:
            return "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("semantic_grid", []) or table.get("display_grid", []) or [])
                if isinstance(row, list)
            )

        def dose_projection(table: dict) -> dict:
            return (table.get("semantic_projection_v2") or {}).get(
                "dose_response_result_panel_projection",
                {},
            )

        page106_carcinogenicity = next(
            table
            for table in page_tables(106)
            if dose_projection(table).get("semantic_profile") == "dose_response_result_panel"
            and dose_projection(table).get("logical_column_count") == 9
        )
        projection106 = dose_projection(page106_carcinogenicity)
        self.assertEqual(projection106.get("semantic_profile"), "dose_response_result_panel")
        self.assertEqual(projection106.get("logical_column_count"), 9)
        self.assertEqual(len(page106_carcinogenicity.get("semantic_grid", [])[0]), 9)
        self.assertEqual(page106_carcinogenicity.get("semantic_grid", [])[0][1:], ["0 M", "0 F", "25 M", "25 F", "100 M", "100 F", "400 M", "400 F"])
        self.assertNotIn("Column", grid_text(page106_carcinogenicity))

        for page, expected_cols in (
            (107, 9),
            (108, 9),
            (110, 5),
            (111, 5),
            (112, 5),
            (113, 5),
            (114, 5),
            (115, 5),
            (116, 5),
        ):
            candidates = [
                table
                for table in page_tables(page)
                if dose_projection(table).get("semantic_profile") == "dose_response_result_panel"
            ]
            self.assertTrue(candidates, msg=f"page {page} missing dose-response projection")
            table = candidates[-1]
            self.assertEqual(
                dose_projection(table).get("logical_column_count"),
                expected_cols,
                msg=grid_text(table),
            )
            self.assertEqual(len(table.get("semantic_grid", [])[0]), expected_cols, msg=grid_text(table))
            self.assertFalse(
                any(
                    isinstance(row, list)
                    and len([cell for cell in row[:-1] if str(cell or "").strip()]) == 0
                    and any(token in str(row[-1] or "") for token in ("\u65e5\u5242\u91cf", "\u8bc4\u4ef7\u6570\u91cf", "\u4e34\u5e8a\u89c2\u5bdf"))
                    for row in (table.get("semantic_grid", []) or [])
                ),
                msg=grid_text(table),
            )
            self.assertNotIn("Column", grid_text(table))

            if page == 116:
                page116_grid = table.get("semantic_grid", []) or []
                page116_rows = {
                    str(row[0] or ""): [str(cell or "") for cell in row]
                    for row in page116_grid
                    if isinstance(row, list) and row
                }
                expected_page116_rows = {
                    "F1 雌性：离乳后评价动物数": ["F1 雌性：离乳后评价动物数", "23", "21", "22", "23"],
                    "(离乳后) 死亡或濒死处死动物数": ["(离乳后) 死亡或濒死处死动物数", "0", "1", "0", "0"],
                    "临床观察": ["临床观察", "-", "-", "-", "-"],
                    "尸体解剖观察": ["尸体解剖观察", "-", "-", "-", "-"],
                    "交配前摄食量(%b)": ["交配前摄食量(%b)", "15 g", "0", "0", "-13*"],
                    "感觉功能": ["感觉功能", "-", "-", "-", "-"],
                    "运动活动": ["运动活动", "-", "-", "-", "-"],
                    "学习记忆": ["学习记忆", "-", "-", "-", "-"],
                    "交配前平均天数": ["交配前平均天数", "2.4", "3.3", "3.1", "3.5"],
                    "妊娠雌性数量": ["妊娠雌性数量", "23", "21", "20", "21"],
                    "平均黄体数": ["平均黄体数", "16.4", "16.2", "15.8", "15.5"],
                    "平均着床数": ["平均着床数", "15.8", "15.2", "14.4", "14.9"],
                    "平均吸收胎数": ["平均吸收胎数", "0.8", "0.3", "0.8", "0.5"],
                    "死胎数": ["死胎数", "0", "0", "0", "0"],
                    "胎仔体重 (g)": ["胎仔体重 (g)", "3.69", "3.65", "3.75", "3.81"],
                    "胎仔异常": ["胎仔异常", "-", "-", "-", "-"],
                }
                for row_label, expected_row in expected_page116_rows.items():
                    self.assertEqual(page116_rows.get(row_label), expected_row, msg=grid_text(table))
                page116_labels = [
                    str(row[0] or "")
                    for row in page116_grid
                    if isinstance(row, list) and row and str(row[0] or "").strip()
                ]
                self.assertEqual(page116_labels.count("精子阳性雌性动物数量"), 1, msg=grid_text(table))
                self.assertGreaterEqual(
                    dose_projection(table).get("logical_row_count", 0),
                    26,
                    msg=grid_text(table),
                )
                note_text = " ".join(str(note.get("text") or "") for note in table.get("note_blocks", []) or [])
                self.assertIn("a-从离乳到交配。", note_text)
                self.assertIn("b–离乳后期间", note_text)
                marker_refs = (table.get("cell_note_refs", []) or []) + (table.get("header_note_refs", []) or [])
                self.assertTrue(
                    any(
                        ref.get("marker") == "a"
                        and "交配前体重变化a(g)" in str(ref.get("cell_text") or ref.get("header_text") or "")
                        for ref in marker_refs
                    ),
                    msg=marker_refs,
                )
                self.assertTrue(
                    any(
                        ref.get("marker") == "b"
                        and "摄食量(%b)" in str(ref.get("cell_text") or ref.get("header_text") or "")
                        for ref in marker_refs
                    ),
                    msg=marker_refs,
                )
                page116_ast = next(page_ast for page_ast in self.result["document_ast"]["pages"] if page_ast["page"] == 116)
                page116_body_text = "\n".join(
                    str(block.get("text") or "")
                    for block in page116_ast.get("blocks", []) or []
                    if block.get("block_type") == "text"
                    and str(block.get("unit_role") or "body") == "body"
                )
                self.assertNotIn("a-从离乳到交配。", page116_body_text)

    def test_page112_populated_reproductive_panel_has_single_business_owner(self) -> None:
        page112_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 112
            and table.get("semantic_role") == "business_table"
        ]

        def dose_projection(table: dict) -> dict:
            return (table.get("semantic_projection_v2") or {}).get(
                "dose_response_result_panel_projection",
                {},
            )

        dose_tables = [
            table
            for table in page112_tables
            if dose_projection(table).get("semantic_profile") == "dose_response_result_panel"
        ]
        self.assertEqual(len(dose_tables), 1, msg=page112_tables)
        table = dose_tables[0]
        projection = dose_projection(table)
        self.assertEqual(projection.get("logical_column_count"), 5)
        self.assertTrue(projection.get("has_study_context"), msg=projection)
        title = str(table.get("title") or "")
        self.assertIn("2.6.7.13", title)
        self.assertEqual(title.count("报告标题：MM-180801"), 1, msg=title)
        self.assertEqual(title.count("供试品：曲醇钠"), 1, msg=title)

        owned_text = "\n".join(str(text or "") for text in table.get("owned_texts", []) or [])
        self.assertIn("报告标题：MM-180801", owned_text)
        self.assertIn("胚胎-胎仔发育毒性", owned_text)
        self.assertIn("平均着床前丢失率%", owned_text)

        study_context_text = "\n".join(
            str(block.get("text") or "")
            for block in table.get("study_context_blocks", []) or []
            if isinstance(block, dict)
        )
        self.assertIn("与ICH 4.1.3", study_context_text)
        self.assertIn("剖腹产日", study_context_text)

        page112_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 112
        ]
        duplicated_template_rows = [
            row
            for template in page112_templates
            for row in template.get("row_texts", []) or []
            if any(token in str(row or "") for token in ("报告标题：MM-180801", "平均着床前丢失率%", "日剂量(mg/kg)"))
        ]
        self.assertEqual(duplicated_template_rows, [], msg=page112_templates)

        page112_ast = next(page_ast for page_ast in self.result["document_ast"]["pages"] if page_ast["page"] == 112)
        page112_blocks = page112_ast.get("blocks", []) or []
        rendered_panel_blocks = [
            block
            for block in page112_blocks
            if any(
                token in str(block.get("text") or block.get("title") or "")
                for token in ("报告标题：MM-180801", "平均着床前丢失率%", "日剂量(mg/kg)")
            )
        ]
        self.assertEqual(
            [block.get("block_type") for block in rendered_panel_blocks],
            ["table"],
            msg=rendered_panel_blocks,
        )

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        page112_start = review_markdown.index(f"**{title}**")
        page113_start = review_markdown.index("2.6.7.14", page112_start)
        page112_markdown = review_markdown[page112_start:page113_start]
        self.assertEqual(page112_markdown.count("报告标题：MM-180801"), 1, msg=page112_markdown)
        self.assertEqual(page112_markdown.count("平均着床前丢失率%"), 1, msg=page112_markdown)

    def test_page117_other_toxicity_summary_schema_table_is_recovered(self) -> None:
        page117_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 117
            and table.get("semantic_role") == "business_table"
        ]
        self.assertTrue(page117_tables, msg="page 117 should recover the other-toxicity schema table")
        table = page117_tables[0]
        projection = (table.get("semantic_projection_v2") or {}).get(
            "toxicology_summary_schema_projection",
            {},
        )
        self.assertEqual(
            projection.get("semantic_profile"),
            "toxicology_summary_schema_table",
            msg=table.get("semantic_projection_v2"),
        )
        self.assertEqual(projection.get("logical_column_count"), 7)
        self.assertEqual(
            table.get("semantic_grid", [])[0],
            [
                "\u79cd\u5c5e/\u54c1\u7cfb",
                "\u7ed9\u836f\u65b9\u6cd5",
                "\u7ed9\u836f\u671f\u9650",
                "\u5242\u91cf(mg/kg)",
                "\u6027\u522b\u548c\u6570\u91cf/\u7ec4",
                "\u503c\u5f97\u6ce8\u610f\u7684\u7ed3\u679c",
                "\u8bd5\u9a8c\u7f16\u53f7",
            ],
        )
        semantic_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in table.get("semantic_grid", []) or []
            if isinstance(row, list)
        )
        for expected in ("\u6297\u539f\u6027", "\u6742\u8d28", "97012", "97025"):
            self.assertIn(expected, semantic_text)

    def test_page117_other_toxicity_summary_has_single_business_table_owner(self) -> None:
        page117_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 117
            and table.get("semantic_role") == "business_table"
            and (table.get("semantic_projection_v2") or {})
            .get("toxicology_summary_schema_projection", {})
            .get("schema_variant")
            == "other_toxicity_summary"
        ]
        self.assertEqual(len(page117_tables), 1, msg=page117_tables)
        table = page117_tables[0]
        table_id = str(table.get("table_id") or "").strip()
        self.assertTrue(table_id)

        page117_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 117
        ]
        duplicated_template_rows = [
            row
            for template in page117_templates
            if str(template.get("ownership_domain") or "") != "absorbed_by_business_table"
            for row in template.get("row_texts", []) or []
            if any(token in str(row or "") for token in ("\u6297\u539f\u6027", "\u8c5a\u9f20", "97012", "\u6742\u8d28", "97025"))
        ]
        self.assertEqual(duplicated_template_rows, [], msg=page117_templates)

        absorbed_templates = [
            template
            for template in page117_templates
            if str(template.get("ownership_domain") or "") == "absorbed_by_business_table"
            and (
                str(template.get("absorbed_by_table_id") or "").strip() == table_id
                or table_id in [str(item or "").strip() for item in template.get("absorbed_by_table_ids", []) or []]
            )
        ]
        self.assertTrue(absorbed_templates, msg=page117_templates)

        page117_ast = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 117)
        rendered_blocks = [
            block
            for block in page117_ast.get("blocks", []) or []
            if any(
                token in "\n".join(
                    [
                        str(block.get("text") or ""),
                        str(block.get("title") or ""),
                        "\n".join(str(row or "") for row in block.get("row_texts", []) or []),
                        "\n".join(
                            " | ".join(str(cell or "") for cell in row)
                            for row in (block.get("semantic_grid") or block.get("display_grid") or block.get("raw_grid") or [])
                            if isinstance(row, list)
                        ),
                    ]
                )
                for token in ("97012", "97025")
            )
        ]
        self.assertEqual(
            [block.get("block_type") for block in rendered_blocks],
            ["table"],
            msg=rendered_blocks,
        )
        stale_title_blocks = [
            block
            for block in page117_ast.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("semantic_role") or "") == "structure_template_title"
            and str(block.get("text") or "").strip() == str(table.get("title") or "").strip()
        ]
        self.assertEqual(stale_title_blocks, [])

        review_markdown = _build_full_markdown([self.result], markdown_profile="ind-review")
        page117_start = review_markdown.rfind("2.6.7.17")
        page117_markdown = review_markdown[page117_start:]
        self.assertNotIn("\u7ed3\u6784\u6a21\u677f", page117_markdown)
        self.assertEqual(page117_markdown.count("97012"), 1, msg=page117_markdown)
        self.assertEqual(page117_markdown.count("97025"), 1, msg=page117_markdown)

    def test_absorbed_populated_templates_do_not_render_business_table_surfaces(self) -> None:
        pages_by_number = {
            int(page.get("page", 0) or 0): page
            for page in self.result.get("document_ast", {}).get("pages", []) or []
            if isinstance(page, dict)
        }
        tables_by_id = {
            str(table.get("table_id") or "").strip(): table
            for table in self.result.get("table_asts", []) or []
            if str(table.get("table_id") or "").strip()
        }
        violations = []
        for template in self.result.get("structure_templates", []) or []:
            if str(template.get("ownership_domain") or "") != "absorbed_by_business_table":
                continue
            owner_ids = [
                str(template.get("absorbed_by_table_id") or "").strip(),
                *[
                    str(item or "").strip()
                    for item in template.get("absorbed_by_table_ids", []) or []
                ],
            ]
            owner_tables = [tables_by_id[table_id] for table_id in owner_ids if table_id in tables_by_id]
            if not owner_tables:
                continue
            owner_surface_tokens = [
                token
                for table in owner_tables
                for token in _business_table_surface_tokens_for_test(table)
            ]
            owner_surface_tokens = list(dict.fromkeys(owner_surface_tokens))
            if not owner_surface_tokens:
                continue
            page = pages_by_number.get(int(template.get("page", 0) or 0))
            if not page:
                continue
            for block in page.get("blocks", []) or []:
                if str(block.get("block_type") or "") != "structure_template":
                    continue
                if str(block.get("template_id") or "") != str(template.get("template_id") or ""):
                    continue
                block_text = "\n".join(
                    [
                        str(block.get("text") or ""),
                        str(block.get("title") or ""),
                        "\n".join(str(row or "") for row in block.get("row_texts", []) or []),
                    ]
                )
                matched_tokens = [token for token in owner_surface_tokens if token in block_text]
                if matched_tokens:
                    violations.append(
                        {
                            "template_id": template.get("template_id"),
                            "owner_ids": owner_ids,
                            "matched_tokens": matched_tokens[:5],
                            "block": block,
                        }
                    )
        self.assertEqual(violations, [])

    def test_page87_bile_excretion_study_panel_owns_metadata_and_top_note(self) -> None:
        page87_templates = [
            template
            for template in self.result.get("structure_templates", []) or []
            if int(template.get("page", 0) or 0) == 87
            and template.get("template_kind") == "study_metadata"
        ]
        self.assertEqual(len(page87_templates), 1)
        metadata = page87_templates[0]
        self.assertEqual(metadata.get("template_profile"), "populated_study_metadata")
        self.assertEqual(metadata.get("ownership_domain"), "study_context")
        metadata_rows = "\n".join(str(row or "") for row in metadata.get("row_texts", []) or [])
        for expected in (
            "2.6.5.14 药代动力学：胆汁排泄 供试品：曲醇钠",
            "种属：大鼠",
            "给药方法：经口给药 / 静脉注射",
            "分析物：TRAa",
            "试验编号：95106",
            "CTD 中的位置：第20 卷，第150 页",
        ):
            self.assertIn(expected, metadata_rows)

        page87_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 87
            and table.get("semantic_role") == "business_table"
        )
        grid_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in (page87_table.get("display_grid", []) or page87_table.get("raw_grid", []) or [])
        )
        for expected in ("排泄途径(4)", "胆汁", "尿液", "合计", "0-48 h", "83", "99"):
            self.assertIn(expected, grid_text)
        note_text = " ".join(str(note.get("text", "")) for note in page87_table.get("note_blocks", []) or [])
        self.assertIn("总放射性", note_text)
        self.assertIn("14C", note_text)
        projection = (
            page87_table.get("semantic_projection_v2", {}) or {}
        ).get("study_condition_grouped_result_matrix_projection", {})
        self.assertEqual(
            projection.get("semantic_profile"),
            "study_condition_grouped_result_matrix",
            msg=page87_table.get("semantic_projection_v2"),
        )
        self.assertEqual(projection.get("condition_group_count"), 2)
        self.assertEqual(projection.get("leaf_count_per_group"), 3)
        self.assertEqual(projection.get("logical_leaf_column_count"), 7)
        self.assertEqual(len(projection.get("condition_groups", []) or []), 2)
        header_groups = projection.get("header_group_rows") or []
        self.assertEqual(
            [group.get("descriptors", {}).get("溶媒/剂型") for group in header_groups],
            ["溶液 / 水", "溶液 / 生理盐水"],
        )
        self.assertEqual(
            page87_table.get("semantic_grid", [])[1],
            ["0-2 h", "37", "-", "37", "75", "-", "75"],
        )
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and "TRA" in str(ref.get("cell_text") or ref.get("anchor_text") or "")
                and "总放射性" in str(ref.get("note_text", ""))
                for ref in (page87_table.get("cell_note_refs", []) or []) + (page87_table.get("header_note_refs", []) or [])
            ),
            msg=f"missing page87 TRA marker ref in {page87_table.get('cell_note_refs', [])!r} / {page87_table.get('header_note_refs', [])!r}",
        )

        page87_ast = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 87)
        body_text = "\n".join(
            str(block.get("text") or "")
            for block in page87_ast.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
        )
        self.assertNotIn("2.6.5.14 药代动力学：胆汁排泄", body_text)
        self.assertNotIn("a-总放射性；回收率", body_text)

    def test_page88_toxicology_overview_links_header_note_and_keeps_wrapped_cells_semantic(self) -> None:
        page88_table = next(
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 88
            and table.get("semantic_role") == "business_table"
            and "试验类型" in "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("display_grid", []) or table.get("raw_grid", []) or [])
            )
        )
        note_text = " ".join(str(note.get("text", "")) for note in page88_table.get("note_blocks", []) or [])
        self.assertIn("除非另有说明", note_text)
        self.assertTrue(
            any(
                ref.get("marker") == "a"
                and "剂量" in str(ref.get("header_text", ""))
                and "除非另有说明" in str(ref.get("note_text", ""))
                for ref in page88_table.get("header_note_refs", []) or []
            ),
            msg=f"missing page88 dose header marker ref in {page88_table.get('header_note_refs', [])!r}",
        )

        semantic_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in (page88_table.get("semantic_grid", []) or page88_table.get("display_grid", []) or [])
        )
        semantic_grid = page88_table.get("semantic_grid") or []
        self.assertEqual(len(semantic_grid[0]), 10)
        self.assertEqual(semantic_grid[1][6], "Sponsor Inc.")
        self.assertEqual(semantic_grid[2][6], "CRO Co.")
        projection = (
            page88_table.get("semantic_projection_v2", {}) or {}
        ).get("overview_inventory_schema_projection", {})
        self.assertEqual(projection.get("semantic_profile"), "nonclinical_overview_inventory_table")
        self.assertEqual(projection.get("logical_column_count"), 10)
        self.assertIn("0、62.5、250、1000、 4000、7000", semantic_text)
        self.assertIn("鼠伤寒沙门氏 菌和大肠杆菌", semantic_text)
        self.assertIn("0、500、1000、2500 和/或5000 µg/皿", semantic_text)
        self.assertIn("0、2.5、5、10、20 和 40 µg/皿", semantic_text)

        page88_ast = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 88)
        body_text = "\n".join(
            str(block.get("text") or "")
            for block in page88_ast.get("blocks", []) or []
            if block.get("block_type") == "text"
            and str(block.get("unit_role") or "body") == "body"
        )
        self.assertNotIn("a-总放射性；回收率", body_text)

    def test_after_page80_nonclinical_overview_borderless_tables_are_recovered(self) -> None:
        tables_by_page = {
            page: [
                table
                for table in self.result.get("table_asts", [])
                if int(table.get("page", 0) or 0) == page
                and table.get("semantic_role") == "business_table"
            ]
            for page in (89, 96, 109, 114)
        }

        for page, page_tables in tables_by_page.items():
            self.assertGreaterEqual(len(page_tables), 1, msg=f"page {page} should expose table AST evidence")

        page89 = max(tables_by_page[89], key=lambda item: int(item.get("col_count", 0) or 0))
        self.assertEqual(page89.get("detection_method"), "text_aligned_borderless_grid")
        self.assertGreaterEqual(int(page89.get("col_count", 0) or 0), 10)
        page89_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in page89.get("display_grid", []) or []
        )
        for expected in ("GLP", "95012", "95013", "96208"):
            self.assertIn(expected, page89_text)

        page96 = max(tables_by_page[96], key=lambda item: int(item.get("col_count", 0) or 0))
        self.assertEqual(page96.get("detection_method"), "text_aligned_borderless_grid")
        self.assertGreaterEqual(int(page96.get("col_count", 0) or 0), 7)
        page96_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in page96.get("display_grid", []) or []
        )
        for expected in ("NOAEL", "CD-1", "Wistar", "94018", "94019"):
            self.assertIn(expected, page96_text)

        page109 = max(tables_by_page[109], key=lambda item: int(item.get("col_count", 0) or 0))
        self.assertEqual(page109.get("detection_method"), "text_aligned_borderless_grid")
        self.assertGreaterEqual(int(page109.get("col_count", 0) or 0), 6)
        page109_semantic = page109.get("semantic_grid", []) or []
        self.assertGreaterEqual(len(page109_semantic), 3, msg=page109_semantic)
        self.assertEqual(page109_semantic[1][0], "Wistar 大鼠", msg=page109_semantic)
        self.assertTrue(
            any(isinstance(row, list) and row and row[0] == "新西兰兔" for row in page109_semantic[1:]),
            msg=page109_semantic,
        )
        page109_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in page109.get("display_grid", []) or []
        )
        for expected in ("Wistar", "94201", "97020"):
            self.assertIn(expected, page109_text)

        page114 = max(tables_by_page[114], key=lambda item: int(item.get("row_count", 0) or 0))
        self.assertEqual(page114.get("detection_method"), "text_aligned_borderless_grid")
        self.assertGreaterEqual(int(page114.get("col_count", 0) or 0), 5)
        page114_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in page114.get("display_grid", []) or []
        )
        for expected in ("0(", "7.5", "75", "750"):
            self.assertIn(expected, page114_text)


    def test_page21_image_with_above_title_and_below_legend_is_recovered(self) -> None:
        page21_images = [
            image
            for image in self.result.get("image_blocks", [])
            if int(image.get("page", 0) or 0) == 21
        ]
        self.assertEqual(len(page21_images), 1)

        image = page21_images[0]
        title_text = str(image.get("title") or image.get("caption_text") or "")
        self.assertIn("图X", title_text)
        self.assertIn("SHRaX长期给药的血压", title_text.replace(" ", ""))
        self.assertEqual(image.get("image_kind_guess"), "captioned_figure")

        content_segments = image.get("content_segments", []) or []
        self.assertEqual(content_segments[0].get("role"), "caption")
        self.assertEqual(content_segments[0].get("relation"), "above")
        self.assertTrue(any(segment.get("role") == "legend" for segment in content_segments))

        content_text = str(image.get("content_text") or "")
        self.assertIn("SHRaX长期给药的血压[参考]", content_text.replace(" ", ""))
        self.assertIn("p<0.05", content_text)
        self.assertIn("p<0.01", content_text)
        self.assertIn("aSHR=自发性高血压大鼠", content_text)
        self.assertIn("n=5只/组", content_text.replace(" ", ""))

        image_evidence = next(
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "image"
            and evidence.get("source_id") == image.get("image_id")
        )
        self.assertEqual(image_evidence.get("semantic_role"), "captioned_figure")
        self.assertEqual(image_evidence.get("title"), image.get("title"))
        self.assertTrue(any(segment.get("role") == "legend" for segment in image_evidence.get("segments", [])))


if __name__ == "__main__":
    unittest.main()
