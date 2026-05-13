from __future__ import annotations

import os
from pathlib import Path
import unittest

from core.material_assessment import build_compliance_result_payload
from core.material_review_contract import build_material_review_contract
from parsers.pdf.pipeline import run_pdf_extraction_pipeline
from parsers.pdf_parser import parse_pdf


def _resolve_ectd_implementation_guide_pdf() -> Path:
    override = os.environ.get("IND_ECTD_IMPLEMENTATION_GUIDE_REGRESSION_PDF", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "regulations" / "eCTD实施指南.pdf"


class EctdImplementationGuideRegressionTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_path = _resolve_ectd_implementation_guide_pdf()
        if not cls.sample_path.exists():
            raise unittest.SkipTest(
                "eCTD implementation guide sample not found. "
                "Set IND_ECTD_IMPLEMENTATION_GUIDE_REGRESSION_PDF or place "
                "eCTD实施指南.pdf under data/regulations."
            )

        cls.pipeline_state = run_pdf_extraction_pipeline(cls.sample_path)
        cls.result = parse_pdf(cls.sample_path)
        cls.metadata = cls.result["metadata"]
        cls.pages = {page["page_number"]: page for page in cls.result["pages"]}
        cls.contract = build_material_review_contract(
            [cls.result],
            generated_at="2026-04-10T00:00:00Z",
        )
        cls.pipeline_pages = {
            page_payload["page_number"]: page_payload
            for page_payload in cls.pipeline_state.page_payloads
        }
        cls.toc_sequences = list(cls.result.get("toc_sequences", []) or [])
        cls.payload = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[cls.result],
            consistency_rows=[],
            final_status="completed",
        )
        cls.page14_tables = [
            table
            for table in cls.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 14
        ]

    def test_pages_2_to_4_form_one_toc_sequence(self) -> None:
        self.assertEqual(self.metadata["toc_sequence_count"], 1)
        self.assertEqual(self.metadata["multi_page_toc_sequence_count"], 1)
        self.assertEqual(len(self.toc_sequences), 1)

        toc_sequence = self.toc_sequences[0]
        self.assertEqual(toc_sequence["pages"], [2, 3, 4])
        self.assertEqual(toc_sequence["page_span"], [2, 4])
        self.assertGreaterEqual(toc_sequence["entry_count"], 40)

    def test_pages_2_to_4_keep_one_toc_block_per_page(self) -> None:
        self.assertEqual(self.pages[2]["toc_count"], 1)
        self.assertEqual(self.pages[3]["toc_count"], 1)
        self.assertEqual(self.pages[4]["toc_count"], 1)

    def test_pages_2_to_4_are_consolidated_at_pipeline_stage(self) -> None:
        self.assertEqual(len(self.pipeline_pages[2].get("toc_blocks", []) or []), 1)
        self.assertEqual(len(self.pipeline_pages[3].get("toc_blocks", []) or []), 1)
        self.assertEqual(len(self.pipeline_pages[4].get("toc_blocks", []) or []), 1)

    def test_toc_sequence_does_not_attach_children_to_wrong_previous_root(self) -> None:
        toc_sequence = self.toc_sequences[0]
        entries_by_outline = {
            str(entry.get("outline_index") or "").strip(): entry
            for entry in toc_sequence.get("entries", []) or []
            if str(entry.get("outline_index") or "").strip()
        }

        for outline_index in ("3.1", "3.2", "3.3", "3.4", "3.5"):
            self.assertEqual(entries_by_outline[outline_index].get("outline_parent_index"), "3.0")
            self.assertEqual(entries_by_outline[outline_index].get("parent_sequence_entry_index"), entries_by_outline["3.0"].get("sequence_entry_index"))
        for outline_index in ("5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"):
            self.assertEqual(entries_by_outline[outline_index].get("outline_parent_index"), "5.0")
            self.assertEqual(entries_by_outline[outline_index].get("parent_sequence_entry_index"), entries_by_outline["5.0"].get("sequence_entry_index"))
        for outline_index in ("7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7"):
            self.assertEqual(entries_by_outline[outline_index].get("outline_parent_index"), "7.0")
            self.assertEqual(entries_by_outline[outline_index].get("parent_sequence_entry_index"), entries_by_outline["7.0"].get("sequence_entry_index"))

    def test_toc_sequence_keeps_root_entries_before_child_groups(self) -> None:
        toc_sequence = self.toc_sequences[0]
        entries_by_outline = {
            str(entry.get("outline_index") or "").strip(): entry
            for entry in toc_sequence.get("entries", []) or []
            if str(entry.get("outline_index") or "").strip()
        }

        expected_roots = {
            "3.0": ("eCTD 申报资料中的编号管理", 11),
            "5.0": ("特定类型提交的建议", 17),
            "7.0": ("对eCTD 申报资料文件的要求", 29),
        }
        for outline_index, (title, locator) in expected_roots.items():
            self.assertIn(outline_index, entries_by_outline)
            self.assertEqual(entries_by_outline[outline_index].get("text"), title)
            self.assertEqual(entries_by_outline[outline_index].get("page_locator_value"), locator)

    def test_toc_sequence_attaches_restored_root_children(self) -> None:
        toc_sequence = self.toc_sequences[0]
        root_nodes = {
            str(node.get("outline_index") or "").strip(): node
            for node in toc_sequence.get("root_nodes", []) or []
            if str(node.get("outline_index") or "").strip()
        }

        expected_children = {
            "3.0": ["3.1", "3.2", "3.3", "3.4", "3.5"],
            "5.0": ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"],
            "7.0": ["7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7"],
        }
        for outline_index, child_indices in expected_children.items():
            self.assertIn(outline_index, root_nodes)
            self.assertEqual(
                [str(child.get("outline_index") or "").strip() for child in root_nodes[outline_index].get("children", [])],
                child_indices,
            )

    def test_toc_alignment_no_longer_reports_false_missing_children_or_roots(self) -> None:
        rules_by_id = {item["rule_id"]: item for item in self.payload.get("rules", [])}
        toc_rule = rules_by_id["SR-TOC-001"]
        audit_rows = list((toc_rule.get("details") or {}).get("structure_audit_rows", []) or [])
        self.assertEqual(len(audit_rows), 1)
        audit_row = audit_rows[0]

        missing_roots = set(audit_row.get("missing_body_root_outline_indices", []) or [])
        missing_children = set(audit_row.get("missing_body_direct_child_outline_indices", []) or [])

        self.assertTrue({"7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7"}.isdisjoint(missing_roots))
        self.assertTrue({"1.0"}.isdisjoint(missing_roots))
        self.assertTrue({"3.1", "3.2", "3.3", "3.4", "3.5"}.isdisjoint(missing_children))
        self.assertTrue({"5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"}.isdisjoint(missing_children))

    def test_root_section_2_body_span_matches_toc_page_window(self) -> None:
        rules_by_id = {item["rule_id"]: item for item in self.payload.get("rules", [])}
        toc_rule = rules_by_id["SR-TOC-001"]
        audit_row = list((toc_rule.get("details") or {}).get("structure_audit_rows", []) or [])[0]
        root_row = next(item for item in audit_row.get("root_page_alignment_rows", []) if item.get("outline_index") == "2.0")

        self.assertEqual(root_row.get("body_page_start"), 6)
        self.assertEqual(root_row.get("body_page_end"), 10)
        self.assertEqual(root_row.get("body_anchor_page_start"), 6)
        self.assertEqual(root_row.get("offset"), 0)
        self.assertFalse(root_row.get("span_conflict"))

    def test_body_section_tree_does_not_expand_root_2_with_numbered_list_items(self) -> None:
        root_node = next(
            node
            for node in self.contract.get("section_tree", []) or []
            if str(node.get("outline_index") or "") == "2"
        )
        self.assertEqual(root_node.get("page_span"), [6, 10])

        false_pages = {14, 17, 26, 35, 40}
        for paragraph in self.contract.get("paragraph_index", []) or []:
            if str(paragraph.get("outline_index") or "") != "2":
                continue
            page_start = (paragraph.get("page_span") or [None])[0]
            self.assertNotIn(page_start, false_pages)

    def test_numbered_list_items_do_not_promote_to_section_headings(self) -> None:
        false_heading_prefixes = {
            14: "2. 负责本次提交序列注册事务的联络人信息",
            17: "2 本章所提供示例仅为举例，不包含所有情况。",
            26: "2. 针对当前注册行为前序序列中替换的文件，需替换",
            35: "2. ICH Electronic Common Technical Document",
            40: "2. 负责本次提交序列注册事务的联络人信息",
        }
        for page_number, prefix in false_heading_prefixes.items():
            page = next(
                item
                for item in self.result.get("document_ast", {}).get("pages", []) or []
                if int(item.get("page", 0) or 0) == page_number
            )
            block = next(
                block
                for block in page.get("blocks", []) or []
                if str(block.get("text") or "").strip().startswith(prefix)
            )
            self.assertNotEqual(block.get("semantic_role"), "section_heading")

    def test_year_month_line_does_not_promote_to_root_heading(self) -> None:
        page1 = next(
            item
            for item in self.result.get("document_ast", {}).get("pages", []) or []
            if int(item.get("page", 0) or 0) == 1
        )
        year_block = next(
            block
            for block in page1.get("blocks", []) or []
            if str(block.get("text") or "").strip().startswith("2021")
        )
        self.assertNotEqual(year_block.get("semantic_role"), "section_heading")

    def test_root_section_1_is_restored_from_body_heading(self) -> None:
        root_node = next(
            node
            for node in self.contract.get("section_tree", []) or []
            if str(node.get("outline_index") or "") == "1"
        )
        self.assertEqual(root_node.get("section_title"), "概述")
        self.assertEqual((root_node.get("page_span") or [None])[0], 5)

    def test_page_14_does_not_emit_false_positive_table(self) -> None:
        self.assertEqual(self.pages[14]["table_count"], 0)
        self.assertEqual(self.page14_tables, [])

    def test_sequence_example_tables_preserve_six_logical_columns(self) -> None:
        expected_rows_by_title = {
            "表2. 新适应症和联合用药示例": [
                ["申请编号", "序列号", "申请类型", "注册行为类型", "序列类型", "序列描述"],
                ["l202112345", "0000", "临床试验申请", "首次申请", "首次提交", "xx 临床试验申请"],
                ["l202112345", "0001", "临床试验申请", "新适应症和联合用药", "首次提交", "xx 新适应症申请"],
            ],
            "表3. 新适应症示例一": [
                ["申请编号", "序列号", "申请类型", "注册行为类型", "序列类型", "序列描述"],
                ["x202112345", "0000", "新药申请", "首次申请", "首次提交", "xx 新药上市申请"],
                ["x202112345", "0001", "新药申请", "新适应症", "首次提交", "xx 新适应症申请"],
            ],
            "表4. 新适应症示例二": [
                ["申请编号", "序列号", "申请类型", "注册行为类型", "序列类型", "序列描述"],
                ["x202112345", "0000", "新药申请", "新适应症", "首次提交", "xx 新适应症申请"],
            ],
        }

        tables_by_title = {
            table.get("title"): table
            for table in self.result.get("table_asts", []) or []
            if table.get("title") in expected_rows_by_title
        }

        self.assertEqual(set(tables_by_title), set(expected_rows_by_title))
        for title, expected_display_grid in expected_rows_by_title.items():
            table = tables_by_title[title]
            with self.subTest(title=title):
                self.assertEqual(table.get("col_count"), 6)
                self.assertEqual(table.get("logical_col_count"), 6)
                self.assertEqual(table.get("physical_col_count"), 10)
                self.assertEqual(table.get("display_grid"), expected_display_grid)
                self.assertEqual(table.get("data_grid"), expected_display_grid[1:])
                self.assertEqual(table.get("row_count"), len(expected_display_grid) - 1)
