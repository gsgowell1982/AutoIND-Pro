# Version: v1.0.0
# Optimization Summary:
# - Lock the real eCTD PDF sample in as a parser regression gate.
# - Assert the validated metadata baseline, continuation chains, and page-1 title reconstruction.
# - Allow path override through IND_ECTD_REGRESSION_PDF while defaulting to the workspace sample.
# - Keep the scope explicitly limited to the PDF parsing pipeline, not other formats.

from __future__ import annotations

import os
from pathlib import Path
import unittest

from core.material_assessment import build_compliance_result_payload
from parsers.pdf_parser import parse_pdf


def _resolve_ectd_regression_pdf() -> Path:
    override = os.environ.get("IND_ECTD_REGRESSION_PDF", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "eCTD技术规范.pdf"


class EctdRegressionSampleTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_path = _resolve_ectd_regression_pdf()
        if not cls.sample_path.exists():
            raise unittest.SkipTest(
                "eCTD regression sample not found. "
                "Set IND_ECTD_REGRESSION_PDF or place eCTD技术规范.pdf under the workspace root."
            )
        cls.result = parse_pdf(cls.sample_path)
        cls.metadata = cls.result["metadata"]
        cls.tables = {table["table_id"]: table for table in cls.result["table_asts"]}
        cls.page2 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 2)
        cls.page3 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 3)
        cls.page4 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 4)
        cls.page2_toc = next(toc for toc in cls.result["toc_blocks"] if toc.get("page") == 2)
        cls.page3_toc = next(toc for toc in cls.result["toc_blocks"] if toc.get("page") == 3)
        cls.page4_toc = next(toc for toc in cls.result["toc_blocks"] if toc.get("page") == 4)
        cls.toc_sequences = {sequence["toc_sequence_id"]: sequence for sequence in cls.result.get("toc_sequences", [])}
        cls.main_toc_sequence = next(iter(cls.toc_sequences.values()))
        try:
            from api.main import _build_workbench
        except ModuleNotFoundError:
            cls.workbench = None
        else:
            cls.workbench = _build_workbench(
                [
                    {
                        **cls.result,
                        "file_id": "file_ectd_regression",
                        "filename": cls.sample_path.name,
                    }
                ],
                [{"id": "file_ectd_regression", "filename": cls.sample_path.name}],
                [],
                None,
            )

    def test_metadata_baseline(self) -> None:
        self.assertEqual(self.metadata["table_count"], 12)
        self.assertEqual(self.metadata["image_count"], 16)
        self.assertEqual(self.metadata["figure_count"], 16)
        self.assertEqual(self.metadata["continuation_table_count"], 5)
        self.assertEqual(self.metadata["cross_page_table_links"], 5)
        self.assertEqual(self.metadata["cross_page_boundary_row_merge_count"], 3)
        self.assertEqual(self.metadata["toc_count"], 3)
        self.assertEqual(self.metadata["toc_sequence_count"], 1)
        self.assertEqual(self.metadata["multi_page_toc_sequence_count"], 1)
        self.assertEqual(self.metadata["toc_review_item_count"], 0)
        self.assertEqual(self.metadata["aggregated_toc_review_item_count"], 0)

    def test_page1_title_reconstruction_baseline(self) -> None:
        self.assertEqual(
            self.result["pages"][0]["text"],
            "附件1\neCTD技术规范\n国家药品监督管理局\n2021 年9 月",
        )

    def test_expected_continuation_chains_and_independent_table(self) -> None:
        self.assertEqual(self.tables["tbl_004"].get("continued_from"), "tbl_003")
        self.assertEqual(self.tables["tbl_007"].get("continued_from"), "tbl_006")
        self.assertEqual(self.tables["tbl_008"].get("continued_from"), "tbl_007")
        self.assertEqual(self.tables["tbl_011"].get("continued_from"), "tbl_010")
        self.assertEqual(self.tables["tbl_012"].get("continued_from"), "tbl_011")

        self.assertFalse(self.tables["tbl_002"].get("is_continuation", False))
        self.assertIsNone(self.tables["tbl_002"].get("continued_from"))

    def test_hierarchical_application_table_carries_leading_group_keys_semantically(self) -> None:
        table3 = self.tables["tbl_003"]
        table4 = self.tables["tbl_004"]

        for table in (table3, table4):
            semantic_compaction = table.get("semantic_compaction") or {}
            self.assertTrue(semantic_compaction.get("applied"))
            self.assertEqual(semantic_compaction.get("strategy"), "leading_key_carry_forward")

        self.assertTrue(table3["row_texts"][1].startswith("临床试验申请 | 补充申请 |"))
        self.assertTrue(table3["row_texts"][2].startswith("临床试验申请 | 新适应症和联合用药 |"))
        self.assertTrue(table3["row_texts"][5].startswith("新药申请 | 补充申请 |"))
        self.assertTrue(table4["row_texts"][0].startswith("新药申请 | 新适应症 |"))
        self.assertTrue(table4["row_texts"][1].startswith("新药申请 | 再注册 |"))
        self.assertTrue(table4["row_texts"][4].startswith("仿制药申请 | 补充申请 |"))

        self.assertTrue(table3["raw_row_texts"][2].startswith("null | 补充申请 |"))
        self.assertTrue(table4["raw_row_texts"][1].startswith("null | 新适应症 |"))

    def test_hierarchical_folder_tables_carry_across_same_page_and_continuation_pages(self) -> None:
        table6 = self.tables["tbl_006"]
        table7 = self.tables["tbl_007"]
        table8 = self.tables["tbl_008"]

        for table in (table6, table7, table8):
            semantic_compaction = table.get("semantic_compaction") or {}
            self.assertTrue(semantic_compaction.get("applied"))
            self.assertEqual(semantic_compaction.get("strategy"), "leading_key_carry_forward")

        self.assertTrue(table6["row_texts"][2].startswith("0000 | index.xml |"))
        self.assertTrue(table6["row_texts"][3].startswith("0000 | index-md5.txt |"))
        self.assertTrue(table6["row_texts"][6].startswith("cn | cn-regional.xml |"))

        expected_cn_subfolders = ["00", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        for idx, folder_name in enumerate(expected_cn_subfolders):
            self.assertEqual(table7["raw_grid"][idx][0], folder_name)
            self.assertIsNone(table7["raw_grid"][idx][1])
            self.assertEqual(table7["display_grid"][idx][0], folder_name)
            self.assertIsNone(table7["display_grid"][idx][1])
            self.assertEqual(table7["data_grid"][idx][0], folder_name)
            self.assertIsNone(table7["data_grid"][idx][1])
        self.assertEqual(table7["raw_grid"][0][2], "模块一 1.0 章节内容文件夹")
        self.assertTrue(table7["row_texts"][18].startswith("dtd | cn-regional-1-0.xsd |"))
        self.assertTrue(table7["row_texts"][24].startswith("style | cn-regional-1-0.xsl |"))
        self.assertTrue(all(row.startswith("style | ") for row in table8["row_texts"]))

        self.assertTrue(table6["raw_row_texts"][3].startswith("null | index.xml |"))
        self.assertTrue(table7["raw_row_texts"][0].startswith("00 | null |"))
        self.assertTrue(table8["raw_row_texts"][0].startswith("null | ectd-2-0.xsl |"))

    def test_directory_folder_rows_do_not_absorb_child_filename_lists(self) -> None:
        table7 = self.tables["tbl_007"]
        dtd_row_index = next(
            idx
            for idx, row in enumerate(table7["display_grid"])
            if row and row[0] == "dtd"
        )

        for grid_key in ("raw_grid", "display_grid", "data_grid"):
            grid = table7[grid_key]
            self.assertEqual(grid[dtd_row_index][0], "dtd")
            self.assertIsNone(grid[dtd_row_index][1], msg=f"{grid_key} should keep dtd folder row file cell empty")
            self.assertIn("DTD", grid[dtd_row_index][2])

            child_filenames = [
                row[1]
                for row in grid[dtd_row_index + 1 :]
                if len(row) > 1 and row[1]
            ][:5]
            self.assertEqual(
                child_filenames,
                [
                    "cn-regional-1-0.xsd",
                    "ich-ectd-3-2.dtd",
                    "ich-stf-v2-2.dtd",
                    "xlink.xsd",
                    "xml.xsd",
                ],
            )

    def test_envelope_element_table_carries_level_column_in_semantic_view_only(self) -> None:
        table9 = self.tables["tbl_009"]
        semantic_compaction = table9.get("semantic_compaction") or {}
        self.assertTrue(semantic_compaction.get("applied"))
        self.assertEqual(semantic_compaction.get("strategy"), "leading_key_carry_forward")

        self.assertTrue(table9["row_texts"][1].startswith("申请级别 | application-type |"))
        self.assertTrue(table9["row_texts"][2].startswith("申请级别 | product-type |"))
        self.assertTrue(table9["row_texts"][5].startswith("注册行为"))
        self.assertIn("regulatory-activity-type", table9["row_texts"][5])
        self.assertTrue(table9["row_texts"][7].startswith("序列级别 | sequence-type |"))
        self.assertTrue(table9["row_texts"][11].startswith("序列级别 | sequence-contact>>email |"))

        self.assertTrue(table9["raw_row_texts"][2].startswith("null | application-type |"))
        self.assertTrue(table9["raw_row_texts"][6].startswith("null | regulatory-activity-type |"))

    def test_application_hierarchy_boundary_fragment_merges_into_parent_semantic_row_only(self) -> None:
        table3 = self.tables["tbl_003"]
        table4 = self.tables["tbl_004"]

        self.assertEqual(table3["row_count"], 8)
        self.assertEqual(table3["raw_row_count"], 9)
        self.assertEqual(table4["row_count"], 10)
        self.assertEqual(table4["raw_row_count"], 11)

        self.assertEqual(
            table3["row_texts"][-1].split(" | ", 2)[:2],
            ["\u65b0\u836f\u7533\u8bf7", "\u62a5\u544a"],
        )
        self.assertIn("\u56de\u590d", table3["row_texts"][-1])
        self.assertIn("\u64a4\u56de", table3["row_texts"][-1])
        self.assertEqual(
            table4["row_texts"][0].split(" | ", 2)[:2],
            ["\u65b0\u836f\u7533\u8bf7", "\u65b0\u9002\u5e94\u75c7"],
        )
        self.assertNotIn("\u62a5\u544a |", table4["row_texts"][0])

        self.assertEqual(
            table4["raw_row_texts"][0],
            "null | null | \u56de\u590d\n\u64a4\u56de",
        )
        self.assertTrue(
            table3["raw_row_texts"][-1].startswith(
                "null | \u62a5\u544a | \u9996\u6b21\u63d0\u4ea4"
            )
        )

    def test_sequence_example_tables_keep_wrapped_phrases_continuous(self) -> None:
        table1 = self.tables["tbl_001"]
        table2 = self.tables["tbl_002"]

        table1_text = "\n".join(table1.get("display_row_texts") or table1.get("row_texts") or [])
        table2_text = "\n".join(table2.get("display_row_texts") or table2.get("row_texts") or [])

        self.assertIn("适应症为xx 的临床试验申请", table1_text)
        self.assertIn("新适应症和联合用药", table1_text)
        self.assertIn("研发期间安全性报告", table1_text)
        self.assertIn("研发期间安全性更新报告提交", table1_text)
        self.assertIn("新增适应症为xx 的临床试验申请", table1_text)
        self.assertIn("适应症为xx 的新药上市申请", table2_text)

        self.assertNotIn("临床试 / 验申请", table1_text)
        self.assertNotIn("新适应症和联 / 合用药", table1_text)
        self.assertNotIn("安全 / 性报告", table1_text)
        self.assertNotIn("安全性更新 / 报告提交", table1_text)
        self.assertNotIn("临 / 床试验申请", table1_text)
        self.assertNotIn("新药上 / 市申请", table2_text)

    def test_glossary_continuation_chain_merges_boundary_definition_fragments_semantically(self) -> None:
        table10 = self.tables["tbl_010"]
        table11 = self.tables["tbl_011"]
        table12 = self.tables["tbl_012"]

        self.assertEqual(table10["row_count"], 8)
        self.assertEqual(table10["raw_row_count"], 9)
        self.assertEqual(table11["row_count"], 9)
        self.assertEqual(table11["raw_row_count"], 10)
        self.assertEqual(table12["row_count"], 1)
        self.assertEqual(table12["raw_row_count"], 2)

        self.assertEqual(
            table10["row_texts"][-1].split(" | ", 1)[0],
            "\u5e8f\u5217\u53f7",
        )
        self.assertIn("\u552f\u4e00\u6807\u8bc6", table10["row_texts"][-1])
        self.assertTrue(table11["row_texts"][-1].startswith("DTD |"))
        self.assertIn("\u6570\u636e\u4ea4\u6362", table11["row_texts"][-1])
        self.assertEqual(
            table12["row_texts"][0].split(" | ", 1)[0],
            "\u9a8c\u8bc1",
        )
        self.assertNotIn("DTD |", table12["row_texts"][0])

        self.assertTrue(
            table11["raw_row_texts"][0].startswith(
                "null | \u4ea4\u5e8f\u5217\u7684\u552f\u4e00\u6807\u8bc6"
            )
        )
        self.assertTrue(
            table12["raw_row_texts"][0].startswith(
                "null | \u6570\u636e\u4ea4\u6362\u800c\u5efa\u7acb\u7684\u5173\u4e8e\u6807\u8bb0\u7b26\u7684\u8bed"
            )
        )

    def test_footnotes_are_structured_and_linked_to_inline_markers(self) -> None:
        footnotes = {
            (int(item.get("page", 0) or 0), str(item.get("marker", "")).strip()): item
            for item in self.result.get("footnotes", [])
        }

        page20_note = footnotes.get((20, "1"))
        page39_note = footnotes.get((39, "2"))
        self.assertIsNotNone(page20_note)
        self.assertIsNotNone(page39_note)
        self.assertIn("SAS XPORT", page20_note["text"])
        self.assertIn("V4.0", page20_note["text"])
        self.assertIn("https://www.ich.org", page39_note["text"])
        self.assertIn("https://www.cde.org.cn", page39_note["text"])
        self.assertEqual(
            page20_note["source_block_ids"],
            ["txt_p20_031", "txt_p20_032", "txt_p20_033", "txt_p20_034"],
        )
        self.assertEqual(page39_note["source_block_ids"], ["txt_p39_021", "txt_p39_022"])

        page20 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 20)
        page39 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 39)
        page20_anchor = next(block for block in page20["blocks"] if block["block_id"] == "txt_p20_005")
        page39_anchor = next(block for block in page39["blocks"] if block["block_id"] == "txt_p39_002")
        self.assertIn(page20_note["footnote_id"], page20_anchor.get("linked_footnote_ids", []))
        self.assertIn(page39_note["footnote_id"], page39_anchor.get("linked_footnote_ids", []))

        footnote_units = [
            unit
            for unit in self.result["content_units"]
            if unit.get("semantic_role") == "footnote"
        ]
        self.assertEqual(len(footnote_units), 2)
        self.assertTrue(all(unit.get("unit_role") == "footnote" for unit in footnote_units))
        self.assertTrue(all(not unit.get("fact_extraction_eligible") for unit in footnote_units))
        footnote_source_ids = {
            source_id
            for footnote in (page20_note, page39_note)
            for source_id in footnote.get("footnote_source_block_ids", [])
        }
        self.assertFalse(
            any(
                unit.get("fact_extraction_eligible") and unit.get("source_id") in footnote_source_ids
                for unit in self.result["content_units"]
            )
        )

    def test_document_ast_footnote_blocks_carry_complete_merged_text_for_markdown(self) -> None:
        from api.main import _build_full_markdown

        page20 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 20)
        page39 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 39)
        page20_note = next(block for block in page20["blocks"] if block.get("block_id") == "txt_p20_031")
        page39_note = next(block for block in page39["blocks"] if block.get("block_id") == "txt_p39_021")

        self.assertIn("SAS XPORT", page20_note.get("footnote_text", ""))
        self.assertIn("V4.0", page20_note.get("footnote_text", ""))
        self.assertIn("https://www.ich.org/", page39_note.get("footnote_text", ""))
        self.assertIn("https://www.cde.org.cn/", page39_note.get("footnote_text", ""))

        markdown = _build_full_markdown([self.result])
        page20_definition = markdown[markdown.index("[^1]:") : markdown.index("[^2]:")]
        self.assertIn("SAS XPORT", page20_definition)
        self.assertIn("V4.0", page20_definition)
        self.assertIn("https://www.cde.org.cn", markdown[markdown.index("[^2]:") :])

    def test_workbench_projection_preserves_hierarchical_and_boundary_merged_tables(self) -> None:
        if self.workbench is None:
            self.skipTest("api.main unavailable in this environment")
        pdf_document = self.workbench["pdf_document"]
        self.assertIsNotNone(pdf_document)
        tables = {table["table_id"]: table for table in pdf_document["table_asts"]}
        toc_sequences = {
            sequence["toc_sequence_id"]: sequence
            for sequence in pdf_document.get("toc_sequences", [])
        }

        self.assertEqual(len(tables), 12)
        self.assertEqual(len(pdf_document["toc_blocks"]), 3)
        self.assertEqual(len(toc_sequences), 1)

        table4 = tables["tbl_004"]
        self.assertEqual(table4.get("continued_from"), "tbl_003")
        self.assertEqual((table4.get("semantic_compaction") or {}).get("strategy"), "leading_key_carry_forward")
        self.assertEqual(
            table4["row_texts"][0].split(" | ", 2)[:2],
            ["\u65b0\u836f\u7533\u8bf7", "\u65b0\u9002\u5e94\u75c7"],
        )
        self.assertEqual(table4["raw_row_texts"][0], "null | null | \u56de\u590d\n\u64a4\u56de")

        table11 = tables["tbl_011"]
        self.assertEqual(table11.get("continued_from"), "tbl_010")
        self.assertEqual(table11["row_count"], 9)
        self.assertEqual(table11["raw_row_count"], 10)
        self.assertEqual(table11["row_texts"][-1].split(" | ", 1)[0], "DTD")
        self.assertTrue(
            table11["raw_row_texts"][0].startswith(
                "null | \u4ea4\u5e8f\u5217\u7684\u552f\u4e00\u6807\u8bc6"
            )
        )

        workbench_main_sequence = toc_sequences[self.main_toc_sequence["toc_sequence_id"]]
        self.assertEqual(workbench_main_sequence["pages"], [2, 3, 4])
        self.assertEqual(workbench_main_sequence["entry_count"], 47)
        self.assertEqual(workbench_main_sequence["root_entry_count"], 7)

    def test_context_signals_match_validated_identity_rules(self) -> None:
        self.assertEqual(self.tables["tbl_002"].get("local_context_signal"), "new_table_title")
        self.assertEqual(self.tables["tbl_004"].get("local_context_signal"), "running_header")
        self.assertEqual(self.tables["tbl_007"].get("local_context_signal"), "running_header")
        self.assertEqual(self.tables["tbl_008"].get("local_context_signal"), "running_header")
        self.assertEqual(self.tables["tbl_010"].get("local_context_signal"), "section_heading")
        self.assertEqual(self.tables["tbl_011"].get("local_context_signal"), "running_header")
        self.assertEqual(self.tables["tbl_012"].get("local_context_signal"), "running_header")

    def test_pages_2_to_4_form_one_continuous_toc_sequence(self) -> None:
        self.assertEqual(self.page2_toc.get("toc_sequence_id"), self.page3_toc.get("toc_sequence_id"))
        self.assertEqual(self.page3_toc.get("toc_sequence_id"), self.page4_toc.get("toc_sequence_id"))
        self.assertEqual(self.result["document_ast"]["toc_sequence_refs"], [self.page2_toc.get("toc_sequence_id")])

        self.assertEqual(self.page2_toc.get("toc_sequence_page_index"), 1)
        self.assertEqual(self.page3_toc.get("toc_sequence_page_index"), 2)
        self.assertEqual(self.page4_toc.get("toc_sequence_page_index"), 3)
        self.assertEqual(self.page2_toc.get("toc_sequence_length"), 3)
        self.assertEqual(self.page3_toc.get("toc_sequence_length"), 3)
        self.assertEqual(self.page4_toc.get("toc_sequence_length"), 3)

        self.assertIsNone(self.page2_toc.get("continued_from_toc_id"))
        self.assertEqual(self.page2_toc.get("continued_to_toc_id"), self.page3_toc.get("toc_id"))
        self.assertEqual(self.page3_toc.get("continued_from_toc_id"), self.page2_toc.get("toc_id"))
        self.assertEqual(self.page3_toc.get("continued_to_toc_id"), self.page4_toc.get("toc_id"))
        self.assertEqual(self.page4_toc.get("continued_from_toc_id"), self.page3_toc.get("toc_id"))
        self.assertIsNone(self.page4_toc.get("continued_to_toc_id"))

        self.assertEqual(self.main_toc_sequence["title"], "目 录")
        self.assertEqual(self.main_toc_sequence["pages"], [2, 3, 4])
        self.assertEqual(self.main_toc_sequence["page_span"], [2, 4])
        self.assertEqual(self.main_toc_sequence["entry_count"], 47)
        self.assertEqual(self.main_toc_sequence["root_entry_indices"], [1, 4, 8, 23, 41, 46, 47])
        self.assertEqual(self.main_toc_sequence["root_entry_count"], 7)
        self.assertEqual(self.main_toc_sequence["max_branching_factor"], 11)
        self.assertGreaterEqual(self.main_toc_sequence["leaf_entry_count"], 30)

    def test_pages_2_to_4_toc_blocks_absorb_scattered_outline_lines(self) -> None:
        self.assertEqual(self.page2_toc.get("entry_count"), 19)
        self.assertEqual(self.page3_toc.get("entry_count"), 20)
        self.assertEqual(self.page4_toc.get("entry_count"), 8)
        self.assertEqual(self.page2_toc.get("missing_page_locator_count"), 0)
        self.assertEqual(self.page3_toc.get("missing_page_locator_count"), 0)
        self.assertEqual(self.page4_toc.get("missing_page_locator_count"), 0)

        self.assertGreaterEqual(len(self.page2_toc.get("promoted_text_block_ids", [])), 5)
        self.assertGreaterEqual(len(self.page3_toc.get("promoted_text_block_ids", [])), 3)
        self.assertGreaterEqual(len(self.page4_toc.get("promoted_text_block_ids", [])), 3)

        self.assertEqual(
            [block["block_type"] for block in self.page2["blocks"]],
            ["text", "toc"],
        )
        self.assertEqual(
            [block.get("text") for block in self.page2["blocks"] if block["block_type"] == "text"],
            ["目 录"],
        )
        self.assertEqual([block["block_type"] for block in self.page3["blocks"]], ["toc"])
        self.assertEqual([block["block_type"] for block in self.page4["blocks"]], ["toc"])

    def test_pages_2_to_4_cross_page_outline_parent_resolution_is_clean(self) -> None:
        for toc_block in (self.page2_toc, self.page3_toc, self.page4_toc):
            diagnostics = toc_block.get("toc_diagnostics", {})
            self.assertFalse(diagnostics.get("review_required"))
            self.assertEqual(diagnostics.get("review_item_count"), 0)
            self.assertEqual(
                diagnostics.get("issue_counts"),
                {
                    "missing_page_locator": 0,
                    "unresolved_outline_parent": 0,
                    "parent_locator_regression": 0,
                    "reading_order_page_regression": 0,
                    "empty_heading_text": 0,
                },
            )

        entry_21 = next(entry for entry in self.page2_toc["entries"] if entry.get("outline_index") == "2.1")
        self.assertEqual(entry_21["parent_entry_index"], 8)
        self.assertIsNone(entry_21.get("sequence_parent_outline_index"))

        entry_233 = next(entry for entry in self.page3_toc["entries"] if entry.get("outline_index") == "2.3.3")
        self.assertIsNone(entry_233.get("parent_entry_index"))
        self.assertEqual(entry_233.get("sequence_parent_outline_index"), "2.3")
        self.assertEqual(entry_233.get("sequence_parent_page"), 2)
        self.assertEqual(entry_233.get("section_anchor_outline_index"), "2.3")
        self.assertEqual(entry_233.get("section_anchor_page"), 2)

        entry_311 = next(entry for entry in self.page4_toc["entries"] if entry.get("outline_index") == "3.11")
        self.assertIsNone(entry_311.get("parent_entry_index"))
        self.assertEqual(entry_311.get("sequence_parent_outline_index"), "3.0")
        self.assertEqual(entry_311.get("sequence_parent_page"), 3)

        entry_40 = next(entry for entry in self.page4_toc["entries"] if entry.get("outline_index") == "4.0")
        self.assertIsNone(entry_40.get("parent_entry_index"))
        self.assertEqual(entry_40["text"], "模块一的总体架构")
        self.assertEqual(entry_40.get("sequence_entry_index"), 41)

        entry_41 = next(entry for entry in self.page4_toc["entries"] if entry.get("outline_index") == "4.1")
        self.assertEqual(entry_41.get("parent_entry_index"), 2)
        self.assertIsNone(entry_41.get("sequence_parent_outline_index"))
        self.assertEqual(entry_41.get("parent_sequence_entry_index"), 41)

    def test_page7_numbered_subsections_are_promoted_to_child_section_contexts(self) -> None:
        page7_targets = {
            unit["text"]: unit
            for unit in self.result.get("content_units", [])
            if unit.get("page") == 7 and unit.get("text") in {"1.1 目的", "1.2 范围和应用", "1.3 配套文件"}
        }

        self.assertEqual(
            sorted(page7_targets.keys()),
            ["1.1 目的", "1.2 范围和应用", "1.3 配套文件"],
        )
        self.assertEqual(page7_targets["1.1 目的"]["section_context"]["outline_index"], "1.1")
        self.assertEqual(page7_targets["1.1 目的"]["section_context"]["section_title"], "目的")
        self.assertEqual(page7_targets["1.2 范围和应用"]["section_context"]["outline_index"], "1.2")
        self.assertEqual(page7_targets["1.2 范围和应用"]["section_context"]["section_title"], "范围和应用")
        self.assertEqual(page7_targets["1.3 配套文件"]["section_context"]["outline_index"], "1.3")
        self.assertEqual(page7_targets["1.3 配套文件"]["section_context"]["section_title"], "配套文件")
        self.assertEqual(page7_targets["1.1 目的"]["unit_role"], "section_heading")
        self.assertEqual(page7_targets["1.2 范围和应用"]["unit_role"], "section_heading")
        self.assertEqual(page7_targets["1.3 配套文件"]["unit_role"], "section_heading")

    def test_toc_alignment_no_longer_flags_page7_child_sections_as_missing(self) -> None:
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[self.result],
            consistency_rows=[],
            final_status="completed",
        )
        toc_rule = next(rule for rule in compliance_result["rules"] if rule.get("rule_id") == "SR-TOC-001")
        audit_row = ((toc_rule.get("details") or {}).get("structure_audit_rows") or [None])[0]

        self.assertNotIn("1.1", audit_row.get("missing_body_direct_child_outline_indices", []))
        self.assertNotIn("1.2", audit_row.get("missing_body_direct_child_outline_indices", []))
        self.assertNotIn("1.3", audit_row.get("missing_body_direct_child_outline_indices", []))
        self.assertNotIn("1.1", audit_row.get("missing_body_bounded_subtree_outline_indices", []))
        self.assertNotIn("1.2", audit_row.get("missing_body_bounded_subtree_outline_indices", []))
        self.assertNotIn("1.3", audit_row.get("missing_body_bounded_subtree_outline_indices", []))
        self.assertEqual(audit_row.get("direct_child_coverage_ratio"), 1.0)
        self.assertEqual(audit_row.get("bounded_subtree_coverage_ratio"), 1.0)

    def test_toc_body_alignment_prefers_toc_backed_root_heading_anchor(self) -> None:
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[self.result],
            consistency_rows=[],
            final_status="completed",
        )
        toc_rule = next(rule for rule in compliance_result["rules"] if rule.get("rule_id") == "SR-TOC-001")
        audit_row = ((toc_rule.get("details") or {}).get("structure_audit_rows") or [None])[0]
        root_rows = audit_row.get("root_page_alignment_rows") or []
        chapter_6 = next(row for row in root_rows if row.get("normalized_outline_index") == "6")

        self.assertEqual(chapter_6.get("toc_page_locator_value"), 41)
        self.assertEqual(chapter_6.get("body_page_start"), 41)
        self.assertEqual(chapter_6.get("offset"), 0)
        self.assertEqual(audit_row.get("root_page_offset_values"), [0, 0, 0, 0, 0, 0])
        self.assertEqual(audit_row.get("projected_root_page_values"), [7, 10, 18, 32, 39, 41])
        self.assertTrue(audit_row.get("root_page_offset_ready"))

    def test_toc_body_alignment_path_rows_cover_all_toc_entries(self) -> None:
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[self.result],
            consistency_rows=[],
            final_status="completed",
        )
        toc_rule = next(rule for rule in compliance_result["rules"] if rule.get("rule_id") == "SR-TOC-001")
        audit_row = ((toc_rule.get("details") or {}).get("structure_audit_rows") or [None])[0]
        path_rows = audit_row.get("toc_body_alignment_path_rows") or []
        numbered_rows = {
            row.get("normalized_outline_index"): row
            for row in path_rows
            if row.get("normalized_outline_index")
        }

        self.assertEqual(len(path_rows), self.main_toc_sequence["entry_count"])
        self.assertEqual(
            [
                (row.get("outline_index"), row.get("page_locator_value"), row.get("body_anchor_page"), row.get("body_anchor_kind"))
                for row in path_rows[:3]
            ],
            [(None, 5, 5, "title"), (None, 5, 5, "title"), (None, 6, 6, "title")],
        )
        self.assertEqual(numbered_rows["1.1"].get("body_anchor_page"), 7)
        self.assertEqual(numbered_rows["1.2"].get("body_anchor_page"), 7)
        self.assertEqual(numbered_rows["1.3"].get("body_anchor_page"), 7)
        self.assertEqual(numbered_rows["6"].get("body_anchor_page"), 41)

    def test_main_toc_sequence_exposes_navigation_tree(self) -> None:
        root_nodes = self.main_toc_sequence["root_nodes"]
        self.assertEqual([node["sequence_entry_index"] for node in root_nodes], [1, 4, 8, 23, 41, 46, 47])
        self.assertEqual(root_nodes[0]["text"], "引言")
        self.assertEqual(root_nodes[0]["child_count"], 2)
        self.assertTrue(root_nodes[0]["has_children"])
        self.assertEqual(
            [child["text"] for child in root_nodes[0]["children"]],
            ["电子通用技术文档（eCTD）模型结构", "本文档结构说明"],
        )

        section_2 = next(node for node in root_nodes if node.get("outline_index") == "2.0")
        self.assertEqual(section_2["child_count"], 4)
        section_21 = next(child for child in section_2["children"] if child.get("outline_index") == "2.1")
        self.assertEqual(section_21["child_count"], 4)
        self.assertEqual(section_21["children"][0]["outline_index"], "2.1.1")

        section_3 = next(node for node in root_nodes if node.get("outline_index") == "3.0")
        self.assertEqual(section_3["page"], 3)
        self.assertEqual(section_3["child_count"], 11)


    def test_main_toc_sequence_exposes_navigation_summary(self) -> None:
        navigation_summary = self.main_toc_sequence["navigation_summary"]
        self.assertEqual(
            navigation_summary["page_entry_spans"],
            [
                {
                    "page": 2,
                    "toc_ids": [self.page2_toc["toc_id"]],
                    "entry_count": 19,
                    "first_sequence_entry_index": 1,
                    "last_sequence_entry_index": 19,
                    "root_entry_indices": [1, 4, 8],
                    "cross_page_parent_entry_count": 0,
                },
                {
                    "page": 3,
                    "toc_ids": [self.page3_toc["toc_id"]],
                    "entry_count": 20,
                    "first_sequence_entry_index": 20,
                    "last_sequence_entry_index": 39,
                    "root_entry_indices": [23],
                    "cross_page_parent_entry_count": 3,
                },
                {
                    "page": 4,
                    "toc_ids": [self.page4_toc["toc_id"]],
                    "entry_count": 8,
                    "first_sequence_entry_index": 40,
                    "last_sequence_entry_index": 47,
                    "root_entry_indices": [41, 46, 47],
                    "cross_page_parent_entry_count": 1,
                },
            ],
        )
        self.assertEqual(navigation_summary["cross_page_parent_link_count"], 4)
        self.assertEqual(navigation_summary["cross_page_root_section_count"], 2)
        self.assertEqual(navigation_summary["outline_path_lookup"]["2.0 > 2.3 > 2.3.3"], [20])
        self.assertEqual(navigation_summary["outline_path_lookup"]["3.0 > 3.11"], [40])

        section_2 = next(
            section for section in navigation_summary["root_sections"] if section.get("outline_index") == "2.0"
        )
        self.assertEqual(section_2["entry_span"], [8, 22])
        self.assertEqual(section_2["page_span"], [2, 3])
        self.assertTrue(section_2["has_cross_page_coverage"])
        self.assertEqual([child["outline_index"] for child in section_2["direct_children"]], ["2.1", "2.2", "2.3", "2.4"])

        section_3 = next(
            section for section in navigation_summary["root_sections"] if section.get("outline_index") == "3.0"
        )
        self.assertEqual(section_3["entry_span"], [23, 40])
        self.assertEqual(section_3["page_span"], [3, 4])
        self.assertTrue(section_3["has_cross_page_coverage"])
        self.assertEqual(section_3["last_page_locator"], "30")
        self.assertEqual(section_3["direct_children"][-1]["outline_index"], "3.11")


if __name__ == "__main__":
    unittest.main()
