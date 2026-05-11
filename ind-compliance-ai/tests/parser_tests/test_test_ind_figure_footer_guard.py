# Version: v1.0.1
# Optimization Summary:
# - Lock the test-ind.pdf page-8 figure-caption/footer scenario into a
#   regression test for the PDF parser.
# - Ensure only one Figure 1 text block remains on page 8 and that the footer
#   page number is not merged into the caption text.
# - Preserve numbered top-of-page section headings on pages 13-15 so repeated
#   heading labels are not stripped as running headers.
# - Treat test-ind.pdf as a standing sample regression gate for currently
#   correct figure, table, and TOC recognition alongside the eCTD baseline.

from __future__ import annotations

from pathlib import Path
import unittest

import fitz

from api.main import _build_workbench
from parsers.pdf_parser import parse_pdf
from parsers.pdf.table_modules.assembly import analyze_table_opening_structure
from parsers.pdf.table_modules.raw_objects import (
    extract_raw_evidence_from_pymupdf,
    extract_raw_evidence_from_words,
)
from parsers.pdf.tables import (
    _collect_item_bboxes,
    _exclude_words_in_occupied_regions,
    _extract_words_from_page,
)


def _resolve_test_ind_pdf() -> Path:
    return Path(__file__).resolve().parents[3] / "test-ind.pdf"


class TestIndFigureFooterGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_path = _resolve_test_ind_pdf()
        if not cls.sample_path.exists():
            raise unittest.SkipTest("test-ind.pdf not found under the workspace root.")
        cls.result = parse_pdf(cls.sample_path)
        cls.page1 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 1)
        cls.page2 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 2)
        cls.page8 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 8)
        cls.page13 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 13)
        cls.page14 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 14)
        cls.page15 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 15)
        cls.page19 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 19)
        cls.page22 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 22)
        cls.page23 = next(page for page in cls.result["document_ast"]["pages"] if page["page"] == 23)
        cls.page2_tables = [table for table in cls.result["table_asts"] if table.get("page") == 2]
        cls.page19_tables = [table for table in cls.result["table_asts"] if table.get("page") == 19]
        cls.page22_tables = [table for table in cls.result["table_asts"] if table.get("page") == 22]
        cls.page23_tables = [table for table in cls.result["table_asts"] if table.get("page") == 23]
        cls.page2_toc_blocks = [toc for toc in cls.result.get("toc_blocks", []) if toc.get("page") == 2]
        cls.page22_toc_blocks = [toc for toc in cls.result.get("toc_blocks", []) if toc.get("page") == 22]
        cls.page23_toc_blocks = sorted(
            [toc for toc in cls.result.get("toc_blocks", []) if toc.get("page") == 23],
            key=lambda toc: toc["bbox"][1],
        )
        cls.page23_top_toc = cls.page23_toc_blocks[0]
        cls.page23_lower_toc = cls.page23_toc_blocks[-1]
        cls.toc_sequences = {sequence["toc_sequence_id"]: sequence for sequence in cls.result.get("toc_sequences", [])}
        cls.figure1 = next(figure for figure in cls.result["figures"] if figure.get("page") == 8)
        cls.image1 = next(image for image in cls.result["image_blocks"] if image.get("page") == 8)
        cls.tables = {table["table_id"]: table for table in cls.result["table_asts"]}
        cls.workbench = _build_workbench(
            [
                {
                    **cls.result,
                    "file_id": "file_test_ind_regression",
                    "filename": cls.sample_path.name,
                }
            ],
            [{"id": "file_test_ind_regression", "filename": cls.sample_path.name}],
            [],
            None,
        )

    def test_page1_address_line_bbox_uses_visible_text_geometry(self) -> None:
        page1_text_blocks = [block for block in self.page1["blocks"] if block["block_type"] == "text"]
        office_line = next(block for block in page1_text_blocks if "Office of Communication" in block["text"])
        address_line = next(block for block in page1_text_blocks if block["text"].startswith("1401 Rockville Pike"))

        self.assertEqual(office_line["bbox"], [132.75, 404.06, 482.25, 417.34])
        self.assertEqual(address_line["bbox"], [191.25, 419.06, 420.75, 432.34])
        self.assertGreater(address_line["bbox"][1], office_line["bbox"][3])

    def test_page8_caption_is_not_duplicated_or_footer_contaminated(self) -> None:
        figure_text_blocks = [
            block
            for block in self.page8["blocks"]
            if block["block_type"] == "text" and "Figure 1" in block["text"]
        ]

        self.assertEqual(len(figure_text_blocks), 1)

        figure_text = figure_text_blocks[0]["text"].replace(";", ":")
        self.assertEqual(figure_text, "Figure 1: IND main folder")
        self.assertNotIn("folder 6", figure_text_blocks[0]["text"])

        bbox = figure_text_blocks[0]["bbox"]
        self.assertGreater(bbox[1], 680.0)
        self.assertLess(bbox[3], 705.0)

    def test_page8_footer_page_number_remains_separate_until_footer_filter(self) -> None:
        page8_text = next(page for page in self.result["pages"] if page["page_number"] == 8)["text"]

        self.assertEqual(page8_text.count("Figure 1"), 1)
        self.assertNotIn("Figure 1: IND main folder 6", page8_text)

    def test_page8_figure_ast_preserves_image_content_evidence(self) -> None:
        self.assertEqual(self.image1["caption_text"], "Figure 1: IND main folder")
        self.assertEqual(self.image1["embedded_text"], "Figure 1: IND main")
        self.assertEqual(self.image1["embedded_text_source"], "text-layer")
        self.assertEqual(self.image1["embedded_text_confidence"], 0.98)
        self.assertEqual(self.image1["image_kind_guess"], "captioned_figure")
        self.assertEqual(self.image1["caption_source_block_id"], "txt_p8_021")
        self.assertGreaterEqual(len(self.image1["nearby_context_blocks"]), 1)
        self.assertIn("root directory", self.image1["nearby_context_text"])
        self.assertIn("Figure 1: IND main folder", self.image1["content_text"])
        self.assertIn("Figure 1: IND main", self.image1["content_text"])
        self.assertTrue(self.image1["content_signals"]["has_caption"])
        self.assertTrue(self.image1["content_signals"]["has_embedded_text"])
        self.assertTrue(self.image1["content_signals"]["has_nearby_context"])

        self.assertEqual(self.figure1["caption_text"], self.image1["caption_text"])
        self.assertEqual(self.figure1["embedded_text"], self.image1["embedded_text"])
        self.assertEqual(self.figure1["image_kind_guess"], self.image1["image_kind_guess"])
        self.assertEqual(self.figure1["caption_source_block_id"], self.image1["caption_source_block_id"])
        self.assertEqual(self.figure1["content_text"], self.image1["content_text"])

        image_blocks = [block for block in self.page8["blocks"] if block["block_type"] == "image"]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]["image_kind_guess"], "captioned_figure")
        self.assertIn("Figure 1: IND main folder", image_blocks[0]["content_text"])

    def test_test_ind_metadata_locks_current_sample_baseline(self) -> None:
        metadata = self.result["metadata"]

        self.assertEqual(metadata["table_count"], 9)
        self.assertEqual(metadata["image_count"], 1)
        self.assertEqual(metadata["figure_count"], 1)
        self.assertEqual(metadata["toc_count"], 4)
        self.assertEqual(metadata["toc_sequence_count"], 3)
        self.assertEqual(metadata["multi_page_toc_sequence_count"], 1)
        self.assertEqual(metadata["review_required_table_count"], 0)
        self.assertEqual(metadata["review_required_toc_count"], 0)

    def test_clean_english_ind_sample_does_not_emit_body_ocr_repair_blocks(self) -> None:
        ocr_blocks = [
            block
            for page in self.result["document_ast"]["pages"]
            for block in page.get("blocks", [])
            if block.get("source") == "body-ocr-repair"
        ]
        self.assertEqual(ocr_blocks, [])

    def test_correctly_recognized_figures_tables_and_tocs_form_sample_regression_gate(self) -> None:
        titled_tables = {
            table["table_id"]: (table["page"], table.get("title"))
            for table in self.result["table_asts"]
            if table.get("title")
        }
        self.assertEqual(
            set(titled_tables),
            {"tbl_001", "tbl_002", "tbl_003", "tbl_004", "tbl_005", "tbl_006"},
        )
        self.assertEqual(titled_tables["tbl_001"], (9, "Table 1: Organization of Documents"))
        self.assertEqual(titled_tables["tbl_002"], (10, "Table 1: Organization of Documents"))
        self.assertEqual(titled_tables["tbl_003"], (19, "Electronic Roadmap"))
        self.assertEqual(titled_tables["tbl_004"], (20, "Main IND Table of Contents"))
        self.assertEqual(titled_tables["tbl_005"][0], 21)
        self.assertTrue(titled_tables["tbl_005"][1].startswith("Clinical Table of Contents"))
        self.assertTrue(titled_tables["tbl_005"][1].endswith("Items contained within the Clinical Folder"))
        self.assertEqual(titled_tables["tbl_006"][0], 21)
        self.assertTrue(titled_tables["tbl_006"][1].startswith("Pharmtox Table of Contents"))
        self.assertTrue(titled_tables["tbl_006"][1].endswith("Items contained within the Pharmtox Folder"))

        figure_summaries = [
            (image["page"], image["caption_text"], image["image_kind_guess"])
            for image in self.result.get("image_blocks", [])
        ]
        self.assertEqual(
            figure_summaries,
            [
                (8, "Figure 1: IND main folder", "captioned_figure"),
            ],
        )

        toc_summaries = [
            (
                toc["page"],
                toc["title"],
                toc["entry_count"],
                toc.get("toc_sequence_id"),
            )
            for toc in sorted(
                self.result.get("toc_blocks", []),
                key=lambda item: (item["page"], item["bbox"][1], item["bbox"][0]),
            )
        ]
        self.assertEqual(
            toc_summaries,
            [
                (2, "Table of Contents", 30, "tocseq_001"),
                (22, "TABLE OF CONTENTS", 25, "tocseq_002"),
                (23, "TABLE OF CONTENTS", 1, "tocseq_002"),
                (23, "TABLE OF CONTENTS", 31, "tocseq_003"),
            ],
        )

        toc_sequence_summaries = [
            (
                sequence["toc_sequence_id"],
                sequence["pages"],
                sequence["title"],
                sequence["entry_count"],
            )
            for sequence in self.result.get("toc_sequences", [])
        ]
        self.assertEqual(
            toc_sequence_summaries,
            [
                ("tocseq_001", [2], "Table of Contents", 30),
                ("tocseq_002", [22, 23], "TABLE OF CONTENTS", 26),
                ("tocseq_003", [23], "TABLE OF CONTENTS", 31),
            ],
        )
        return

        titled_tables = [
            (table["table_id"], table["page"], table.get("title"))
            for table in self.result["table_asts"]
            if table.get("title")
        ]
        self.assertEqual(
            titled_tables,
            [
                ("tbl_001", 9, "Table 1: Organization of Documents"),
                ("tbl_002", 10, "Table 1: Organization of Documents"),
                ("tbl_003", 19, "Electronic Roadmap"),
                ("tbl_004", 20, "Main IND Table of Contents"),
                ("tbl_005", 21, "Clinical Table of Contents 每 Items contained within the Clinical Folder"),
                ("tbl_006", 21, "Pharmtox Table of Contents 每 Items contained within the Pharmtox Folder"),
            ],
        )

        figure_summaries = [
            (image["page"], image["caption_text"], image["image_kind_guess"])
            for image in self.result.get("image_blocks", [])
        ]
        self.assertEqual(
            figure_summaries,
            [
                (8, "Figure 1: IND main folder", "captioned_figure"),
            ],
        )

        toc_summaries = [
            (
                toc["page"],
                toc["title"],
                toc["entry_count"],
                toc.get("toc_sequence_id"),
            )
            for toc in sorted(
                self.result.get("toc_blocks", []),
                key=lambda item: (item["page"], item["bbox"][1], item["bbox"][0]),
            )
        ]
        self.assertEqual(
            toc_summaries,
            [
                (2, "Table of Contents", 30, "tocseq_001"),
                (22, "TABLE OF CONTENTS", 25, "tocseq_002"),
                (23, "TABLE OF CONTENTS", 1, "tocseq_002"),
                (23, "TABLE OF CONTENTS", 31, "tocseq_003"),
            ],
        )

        toc_sequence_summaries = [
            (
                sequence["toc_sequence_id"],
                sequence["pages"],
                sequence["title"],
                sequence["entry_count"],
            )
            for sequence in self.result.get("toc_sequences", [])
        ]
        self.assertEqual(
            toc_sequence_summaries,
            [
                ("tocseq_001", [2], "Table of Contents", 30),
                ("tocseq_002", [22, 23], "TABLE OF CONTENTS", 26),
                ("tocseq_003", [23], "TABLE OF CONTENTS", 31),
            ],
        )

    def test_content_evidence_view_unifies_text_table_toc_and_image_content(self) -> None:
        metadata = self.result["metadata"]
        content_evidence = self.result.get("content_evidence", [])
        self.assertEqual(metadata["content_evidence_count"], len(content_evidence))
        self.assertEqual(
            metadata["content_evidence_counts"],
            {
                "text": metadata["content_evidence_counts"]["text"],
                "image": 1,
                "table": metadata["table_count"],
                "toc": 4,
            },
        )
        self.assertEqual(
            len(self.result["document_ast"]["content_evidence_refs"]),
            len(content_evidence),
        )
        self.assertEqual(metadata["toc_sequence_count"], len(self.result.get("toc_sequences", [])))
        self.assertEqual(self.result["document_ast"]["toc_sequence_refs"], list(self.toc_sequences.keys()))

        image_evidence = next(item for item in content_evidence if item["source_type"] == "image")
        self.assertEqual(image_evidence["source_id"], self.image1["image_id"])
        self.assertEqual(image_evidence["semantic_role"], "captioned_figure")
        self.assertEqual(image_evidence["caption_text"], "Figure 1: IND main folder")
        self.assertIn("root directory", image_evidence["nearby_context_text"])

        table_evidence = next(item for item in content_evidence if item["source_type"] == "table" and item["source_id"] == "tbl_003")
        self.assertEqual(table_evidence["title"], "Electronic Roadmap")
        self.assertIn("IND Submission | Submission Date", table_evidence["content_text"])
        self.assertNotIn("Electronic Roadmap", table_evidence["content_text"])
        self.assertEqual(table_evidence["segments"][0]["role"], "title")
        self.assertEqual(table_evidence["segments"][0]["text"], "Electronic Roadmap")
        self.assertGreaterEqual(len(table_evidence["segments"]), 5)

        toc_evidence_page2 = next(
            item
            for item in content_evidence
            if item["source_type"] == "toc" and item["source_id"] == self.page2_toc_blocks[0]["toc_id"]
        )
        self.assertEqual(toc_evidence_page2["semantic_role"], "toc_outline")
        self.assertIn("III | GENERAL ISSUES | 3", toc_evidence_page2["content_text"])
        self.assertIn("APPENDIX B | IND TABLE OF CONTENTS | 18", toc_evidence_page2["content_text"])

        toc_evidence_page22 = next(
            item
            for item in content_evidence
            if item["source_type"] == "toc" and item["source_id"] == self.page22_toc_blocks[0]["toc_id"]
        )
        self.assertEqual(toc_evidence_page22["semantic_role"], "toc_outline")
        self.assertIn("2.4 | Human Clinical Trials | 5", toc_evidence_page22["content_text"])

        toc_evidence_page23_top = next(
            item
            for item in content_evidence
            if item["source_type"] == "toc" and item["source_id"] == self.page23_top_toc["toc_id"]
        )
        self.assertEqual(toc_evidence_page23_top["semantic_role"], "toc_outline")
        self.assertTrue(toc_evidence_page23_top["title_inferred"])
        self.assertEqual(toc_evidence_page23_top["content_text"], "5.5.1 | Safety Outcome Measures | 21")

        toc_evidence_page23_lower = next(
            item
            for item in content_evidence
            if item["source_type"] == "toc" and item["source_id"] == self.page23_lower_toc["toc_id"]
        )
        self.assertEqual(toc_evidence_page23_lower["semantic_role"], "toc_outline")
        self.assertIn(
            "1.1.1 | Construction of Genes Encoding the Light and Heavy Chain Variable Regions | 1",
            toc_evidence_page23_lower["content_text"],
        )

    def test_content_units_flatten_rule_ready_units_without_toc_fact_pollution(self) -> None:
        metadata = self.result["metadata"]
        content_units = self.result.get("content_units", [])
        self.assertEqual(metadata["content_unit_count"], len(content_units))
        self.assertEqual(
            metadata["content_unit_counts"],
            {
                "text": metadata["content_unit_counts"]["text"],
                "image": 5,
                "table": 83,
                "toc": 90,
            },
        )
        self.assertEqual(
            len(self.result["document_ast"]["content_unit_refs"]),
            len(content_units),
        )
        self.assertGreater(metadata["fact_extraction_unit_count"], 0)
        self.assertGreater(metadata["fact_extraction_corpus_length"], 0)

        image_caption_unit = next(
            unit
            for unit in content_units
            if unit["source_type"] == "image" and unit["unit_role"] == "caption"
        )
        self.assertEqual(image_caption_unit["text"], "Figure 1: IND main folder")
        self.assertTrue(image_caption_unit["fact_extraction_eligible"])

        toc_entry_unit = next(
            unit
            for unit in content_units
            if unit["source_type"] == "toc"
            and unit["source_id"] == self.page23_top_toc["toc_id"]
            and unit["unit_role"] == "entry"
        )
        self.assertFalse(toc_entry_unit["fact_extraction_eligible"])
        self.assertEqual(toc_entry_unit["attributes"]["entry_index"], 1)

        table_row_unit = next(
            unit
            for unit in content_units
            if unit["source_type"] == "table"
            and unit["source_id"] == "tbl_003"
            and unit["unit_role"] == "row"
            and "Cover letter" in unit["text"]
        )
        self.assertTrue(table_row_unit["fact_extraction_eligible"])
        self.assertIn("Cover letter", table_row_unit["text"])

    def test_pages_13_to_15_preserve_numbered_top_headings(self) -> None:
        expected = {
            13: "5. Publications",
            14: "2. Publications",
            15: "2. Publications",
        }

        for page in (self.page13, self.page14, self.page15):
            top_text_blocks = [
                block
                for block in page["blocks"]
                if block["block_type"] == "text" and block["bbox"][1] < 120.0
            ]
            self.assertTrue(top_text_blocks)
            self.assertEqual(top_text_blocks[0]["text"], expected[page["page"]])

    def test_pages_13_to_15_page_text_keeps_top_headings(self) -> None:
        expected = {
            13: "5. Publications",
            14: "2. Publications",
            15: "2. Publications",
        }

        for page_number, heading in expected.items():
            page_text = next(page for page in self.result["pages"] if page["page_number"] == page_number)["text"]
            self.assertTrue(page_text.startswith(heading))

    def test_page19_borderless_roadmap_is_recognized_as_table(self) -> None:
        self.assertEqual(len(self.page19_tables), 1)

        table = self.page19_tables[0]
        self.assertEqual(table["detection_method"], "word_clustering")
        self.assertEqual(table["title"], "Electronic Roadmap")
        self.assertEqual(table["local_context_signal"], "new_table_title")
        self.assertGreaterEqual(table["row_count"], 20)
        self.assertEqual(table["col_count"], 5)
        self.assertEqual(table["preceding_text_block"]["text"], "Electronic Roadmap")

        header_texts = [cell["text"] for cell in table["header"]]
        self.assertIn("IND Submission", header_texts)
        self.assertIn("Submission Date", header_texts)
        self.assertIn("Hypertext link Destination", header_texts)

        image_blocks = [block for block in self.page19["blocks"] if block["block_type"] == "image"]
        self.assertEqual(image_blocks, [])

    def test_page19_word_clustering_raw_evidence_builds_full_rows(self) -> None:
        document = fitz.open(self.sample_path)
        try:
            page = document[18]
            words = _extract_words_from_page(page)
            raw = extract_raw_evidence_from_words(
                words=words,
                page_number=19,
                page_height=page.rect.height,
                page_width=page.rect.width,
            )
        finally:
            document.close()

        self.assertIsNotNone(raw)
        assert raw is not None
        self.assertEqual(raw.source, "word_clustering")
        self.assertEqual(len(raw.rows), raw.physical_row_count)
        self.assertGreaterEqual(raw.physical_row_count, 20)
        self.assertEqual(raw.physical_col_count, 5)
        self.assertEqual(raw.raw_data[0][:5], [
            "IND Submission",
            "Submission Date",
            "Submission",
            "CD-ROM",
            "Hypertext link",
        ])

    def test_page19_multiline_header_continuations_merge_into_semantic_header(self) -> None:
        table = self.page19_tables[0]
        self.assertEqual(
            [cell["text"] for cell in table["header"]],
            [
                "IND Submission",
                "Submission Date",
                "Submission Content",
                "CD-ROM",
                "Hypertext link Destination",
            ],
        )
        self.assertEqual(table.get("data_start_row"), 2)
        self.assertNotIn("null | null | Content | null | Destination", table.get("row_texts") or [])
        self.assertEqual(
            (table.get("raw_row_texts") or [None, None])[1],
            "null | null | Content | null | Destination",
        )
        self.assertEqual(
            (table.get("row_texts") or [None])[0],
            "IND 12345.0003 | 04-Jul-2001 | Cover letter | 3.01 | amendtoc.pdf",
        )

    def test_page19_sparse_group_rows_carry_forward_leading_keys_in_semantic_view(self) -> None:
        table = self.page19_tables[0]
        semantic_compaction = table.get("semantic_compaction") or {}
        self.assertTrue(semantic_compaction.get("applied"))
        self.assertEqual(semantic_compaction.get("strategy"), "leading_key_carry_forward")

        row_texts = table.get("row_texts") or []
        self.assertEqual(
            row_texts[:5],
            [
                "IND 12345.0003 | 04-Jul-2001 | Cover letter | 3.01 | amendtoc.pdf",
                "IND 12345.0003 | 04-Jul-2001 | 1571 | 3.01 | null",
                "IND 12345.0003 | 04-Jul-2001 | Protocol 12-345 | 3.01 | null",
                "IND 12345.0003 | 04-Jul-2001 | Investigator | 3.01 | null",
                "IND 12345.0003 | 04-Jul-2001 | Information | null | null",
            ],
        )
        self.assertTrue(all(not row.startswith("null | null |") for row in row_texts[:10]))
        self.assertEqual(
            (table.get("raw_row_texts") or [None] * 4)[3],
            "null | null | 1571 | 3.01 | null",
        )

    def test_page20_and_page21_toc_tables_remain_independent(self) -> None:
        tbl_004 = self.tables["tbl_004"]
        tbl_005 = self.tables["tbl_005"]
        tbl_006 = self.tables["tbl_006"]

        self.assertEqual(tbl_004.get("page"), 20)
        self.assertEqual(tbl_005.get("page"), 21)
        self.assertEqual(tbl_006.get("page"), 21)

        self.assertIsNone(tbl_004.get("continued_from"))
        self.assertIsNone(tbl_005.get("continued_from"))
        self.assertIsNone(tbl_006.get("continued_from"))
        self.assertFalse(tbl_004.get("is_continuation", False))
        self.assertFalse(tbl_005.get("is_continuation", False))
        self.assertFalse(tbl_006.get("is_continuation", False))

        self.assertEqual(tbl_004.get("title"), "Main IND Table of Contents")
        self.assertEqual(
            [cell["text"] for cell in tbl_004["header"]],
            ["Section", "Description", "Electronic folder/filename"],
        )
        self.assertEqual(
            [cell["text"] for cell in tbl_005["header"]],
            ["Item", "Description", "Folder/File"],
        )
        self.assertEqual(
            [cell["text"] for cell in tbl_006["header"]],
            ["Item", "Description", "Folder/File"],
        )

    def test_page21_internal_title_tables_keep_three_distinct_logical_columns(self) -> None:
        for table_id, expected_first_data_row in (
            (
                "tbl_005",
                [
                    "1",
                    "General Investigational Plan",
                    "Clinical\\0000 geninvestplantoc.pdf",
                ],
            ),
            (
                "tbl_006",
                [
                    "1",
                    "Pharmacology and Toxicology Summary",
                    "Pharmtox\\0000 summarytoc.pdf",
                ],
            ),
        ):
            table = self.tables[table_id]

            self.assertEqual(table.get("col_count"), 3)
            self.assertGreaterEqual(len(table.get("grid", [])), 1)
            self.assertGreaterEqual(len(table.get("display_grid", [])), 3)
            self.assertEqual(
                table["display_grid"][1],
                ["Item", "Description", "Folder/File"],
            )
            self.assertEqual(table["grid"][0], expected_first_data_row)
            self.assertEqual(table.get("title_row_index"), 0)
            self.assertEqual(table.get("header_row_index"), 1)
            self.assertEqual(table.get("data_start_row"), 2)
            self.assertEqual(table["display_grid"][2], expected_first_data_row)

            row1_cells = sorted(
                [
                    cell
                    for cell in table.get("cells", [])
                    if int(cell.get("row", 0) or 0) == 1
                ],
                key=lambda cell: int(cell.get("col", 0) or 0),
            )
            self.assertEqual([cell["col"] for cell in row1_cells], [1, 2, 3])
            self.assertEqual(
                [cell.get("text") for cell in row1_cells],
                expected_first_data_row,
            )

    def test_page20_and_page21_opening_structure_detects_internal_title_rows(self) -> None:
        document = fitz.open(self.sample_path)
        try:
            page20 = document[19]
            raw20 = extract_raw_evidence_from_pymupdf(
                page=page20,
                page_number=20,
                page_height=page20.rect.height,
                page_width=page20.rect.width,
                table_index=0,
            )
            page21 = document[20]
            raw21_clinical = extract_raw_evidence_from_pymupdf(
                page=page21,
                page_number=21,
                page_height=page21.rect.height,
                page_width=page21.rect.width,
                table_index=0,
            )
            raw21_pharmtox = extract_raw_evidence_from_pymupdf(
                page=page21,
                page_number=21,
                page_height=page21.rect.height,
                page_width=page21.rect.width,
                table_index=1,
            )
        finally:
            document.close()

        self.assertIsNotNone(raw20)
        self.assertIsNotNone(raw21_clinical)
        self.assertIsNotNone(raw21_pharmtox)
        assert raw20 is not None
        assert raw21_clinical is not None
        assert raw21_pharmtox is not None

        opening20 = analyze_table_opening_structure(raw20.raw_data, raw20.physical_col_count)
        opening21_clinical = analyze_table_opening_structure(
            raw21_clinical.raw_data,
            raw21_clinical.physical_col_count,
        )
        opening21_pharmtox = analyze_table_opening_structure(
            raw21_pharmtox.raw_data,
            raw21_pharmtox.physical_col_count,
        )

        self.assertEqual(opening20.title_text, "Main IND Table of Contents")
        self.assertEqual(
            [cell["text"] for cell in opening20.header_cells],
            ["Section", "Description", "Electronic folder/filename"],
        )
        self.assertEqual(
            opening21_clinical.title_text,
            self.tables["tbl_005"]["title"],
        )
        self.assertEqual(
            [cell["text"] for cell in opening21_clinical.header_cells],
            ["Item", "Description", "Folder/File"],
        )
        self.assertEqual(
            opening21_pharmtox.title_text,
            self.tables["tbl_006"]["title"],
        )
        self.assertEqual(
            [cell["text"] for cell in opening21_pharmtox.header_cells],
            ["Item", "Description", "Folder/File"],
        )

    def test_page2_real_table_of_contents_is_preserved_as_toc_block(self) -> None:
        self.assertEqual(self.page2_tables, [])
        self.assertEqual(len(self.page2_toc_blocks), 1)

        toc_block = self.page2_toc_blocks[0]
        self.assertEqual(toc_block.get("semantic_role"), "toc_outline")
        self.assertFalse(toc_block.get("is_business_table"))
        self.assertEqual(toc_block.get("title"), "Table of Contents")
        self.assertEqual(toc_block.get("entry_count"), 30)
        self.assertEqual(toc_block.get("raw_entry_row_count"), 30)
        self.assertEqual(toc_block.get("wrapped_entry_count"), 0)
        self.assertEqual(toc_block.get("max_entry_level"), 2)
        self.assertEqual(toc_block.get("max_outline_depth"), 1)
        self.assertEqual(toc_block.get("missing_page_locator_count"), 0)
        self.assertEqual(toc_block.get("page_locator_kinds"), {"arabic": 30, "roman": 0, "unknown": 0})

        semantic_signals = toc_block.get("semantic_signals", {})
        self.assertAlmostEqual(float(semantic_signals.get("page_locator_ratio", 0.0)), 1.0, places=3)
        self.assertEqual(int(semantic_signals.get("locator_indent_level_count", 0)), 2)
        self.assertTrue(semantic_signals.get("toc_title_support"))
        self.assertFalse(semantic_signals.get("has_schema_header"))

        toc_diagnostics = toc_block.get("toc_diagnostics", {})
        self.assertFalse(toc_diagnostics.get("review_required"))
        self.assertEqual(toc_diagnostics.get("review_item_count"), 0)
        self.assertAlmostEqual(float(toc_diagnostics.get("locator_coverage_ratio", 0.0)), 1.0, places=3)
        self.assertFalse(toc_diagnostics.get("mixed_page_numbering"))

        first_entry = toc_block["entries"][0]
        self.assertEqual(first_entry["outline_index"], "I")
        self.assertEqual(first_entry["text"], "INTRODUCTION")
        self.assertEqual(first_entry["page_locator"], "1")

        appendix_b = next(entry for entry in toc_block["entries"] if entry.get("outline_index") == "APPENDIX B")
        self.assertEqual(appendix_b["text"], "IND TABLE OF CONTENTS")
        self.assertEqual(appendix_b["page_locator"], "18")
        self.assertEqual(appendix_b["level"], 2)
        self.assertEqual(appendix_b["parent_entry_index"], 27)

        toc_blocks = [block for block in self.page2["blocks"] if block["block_type"] == "toc"]
        self.assertEqual(len(toc_blocks), 1)
        self.assertEqual(toc_blocks[0]["toc_id"], toc_block["toc_id"])
        self.assertEqual(
            [block["text"] for block in self.page2["blocks"] if block["block_type"] == "text"],
            ["Table of Contents"],
        )

        toc_sequence = self.toc_sequences[toc_block["toc_sequence_id"]]
        self.assertEqual(toc_sequence["title"], "Table of Contents")
        self.assertEqual(toc_sequence["pages"], [2])
        self.assertEqual(toc_sequence["entry_count"], 30)
        self.assertEqual(toc_sequence["root_entry_indices"], [1, 2, 3, 14, 21, 27])
        self.assertEqual(toc_sequence["root_entry_count"], 6)
        self.assertEqual(toc_sequence["max_branching_factor"], 10)
        self.assertGreaterEqual(toc_sequence["leaf_entry_count"], 20)

        root_nodes = toc_sequence["root_nodes"]
        self.assertEqual([node["sequence_entry_index"] for node in root_nodes], [1, 2, 3, 14, 21, 27])
        section_iii = next(node for node in root_nodes if node.get("outline_index") == "III")
        self.assertEqual(section_iii["child_count"], 10)
        self.assertEqual(section_iii["children"][0]["outline_index"], "A")
        appendix_root = next(node for node in root_nodes if node.get("outline_index") == "VI")
        self.assertEqual(appendix_root["child_count"], 3)
        self.assertEqual([child["outline_index"] for child in appendix_root["children"]], ["APPENDIX A", "APPENDIX B", "APPENDIX C"])

        navigation_summary = toc_sequence["navigation_summary"]
        self.assertEqual(
            navigation_summary["page_entry_spans"],
            [
                {
                    "page": 2,
                    "toc_ids": [toc_block["toc_id"]],
                    "entry_count": 30,
                    "first_sequence_entry_index": 1,
                    "last_sequence_entry_index": 30,
                    "root_entry_indices": [1, 2, 3, 14, 21, 27],
                    "cross_page_parent_entry_count": 0,
                }
            ],
        )
        self.assertEqual(navigation_summary["cross_page_parent_link_count"], 0)
        self.assertEqual(navigation_summary["cross_page_root_section_count"], 0)
        self.assertEqual(navigation_summary["outline_path_lookup"]["III > A"], [4])
        self.assertEqual(navigation_summary["outline_path_lookup"]["VI > APPENDIX B"], [29])

        section_iii_summary = next(
            section for section in navigation_summary["root_sections"] if section.get("outline_index") == "III"
        )
        self.assertEqual(section_iii_summary["entry_span"], [3, 13])
        self.assertEqual(section_iii_summary["entry_count"], 11)
        self.assertEqual(section_iii_summary["leaf_count"], 10)
        self.assertEqual(section_iii_summary["direct_children"][0]["outline_index"], "A")

        appendix_summary = next(
            section for section in navigation_summary["root_sections"] if section.get("outline_index") == "VI"
        )
        self.assertEqual([child["outline_index"] for child in appendix_summary["direct_children"]], ["APPENDIX A", "APPENDIX B", "APPENDIX C"])
        self.assertEqual(appendix_summary["last_page_locator"], "22")

    def test_page22_lower_outline_is_preserved_as_toc_block(self) -> None:
        self.assertEqual(len(self.page22_tables), 2)
        self.assertEqual(len(self.page22_toc_blocks), 1)

        toc_block = self.page22_toc_blocks[0]
        self.assertEqual(toc_block.get("semantic_role"), "toc_outline")
        self.assertFalse(toc_block.get("is_business_table"))
        self.assertEqual(toc_block.get("title"), "TABLE OF CONTENTS")
        self.assertEqual(toc_block.get("entry_count"), 25)
        self.assertEqual(toc_block.get("raw_entry_row_count"), 26)
        self.assertEqual(toc_block.get("missing_page_locator_count"), 0)
        self.assertEqual(toc_block.get("wrapped_entry_count"), 1)
        self.assertEqual(toc_block.get("max_entry_level"), 3)
        self.assertEqual(toc_block.get("page_locator_kinds"), {"arabic": 23, "roman": 2, "unknown": 0})
        self.assertLess(toc_block["bbox"][3], 722.0)
        self.assertFalse(toc_block["toc_diagnostics"]["review_required"])
        self.assertEqual(toc_block["toc_diagnostics"]["review_item_count"], 0)
        self.assertAlmostEqual(float(toc_block["toc_diagnostics"]["locator_coverage_ratio"]), 1.0, places=3)

        first_entry = toc_block["entries"][0]
        self.assertEqual(first_entry["text"], "SYNOPSIS")
        self.assertEqual(first_entry["page_locator"], "ii")

        entry_25 = next(entry for entry in toc_block["entries"] if entry.get("outline_index") == "2.5")
        self.assertEqual(entry_25["text"], "Rationale for Study Design")
        self.assertEqual(entry_25["page_locator"], "6")

        toc_blocks = [block for block in self.page22["blocks"] if block["block_type"] == "toc"]
        self.assertEqual(len(toc_blocks), 1)
        self.assertEqual(toc_blocks[0]["toc_id"], toc_block["toc_id"])

        text_labels = [
            block["text"]
            for block in self.page22["blocks"]
            if block["block_type"] == "text"
        ]
        self.assertNotIn("TABLE OF CONTENTS", text_labels)

    def test_page22_residual_word_clustering_surfaces_outline_after_excluding_existing_tables(self) -> None:
        document = fitz.open(self.sample_path)
        try:
            page = document[21]
            words = _extract_words_from_page(page)
            residual_words = _exclude_words_in_occupied_regions(
                words,
                _collect_item_bboxes(self.page22_tables),
            )
            raw = extract_raw_evidence_from_words(
                words=residual_words,
                page_number=22,
                page_height=page.rect.height,
                page_width=page.rect.width,
            )
        finally:
            document.close()

        self.assertIsNotNone(raw)
        assert raw is not None
        self.assertGreater(raw.bbox[1], 350.0)
        self.assertEqual(raw.source, "word_clustering")
        self.assertGreaterEqual(raw.physical_row_count, 25)
        self.assertGreaterEqual(raw.physical_col_count, 4)
        self.assertEqual(raw.raw_data[0][3], "TABLE OF CONTENTS")

    def test_page23_outline_is_not_emitted_as_business_table(self) -> None:
        self.assertEqual(self.page23_tables, [])

    def test_page23_top_edge_outline_is_promoted_as_cross_page_toc_continuation(self) -> None:
        self.assertEqual(len(self.page23_toc_blocks), 2)

        toc_block = self.page23_top_toc
        self.assertEqual(toc_block.get("semantic_role"), "toc_outline")
        self.assertFalse(toc_block.get("is_business_table"))
        self.assertEqual(toc_block.get("title"), "TABLE OF CONTENTS")
        self.assertTrue(toc_block.get("title_inferred"))
        self.assertEqual(toc_block.get("entry_count"), 1)
        self.assertEqual(toc_block.get("raw_entry_row_count"), 1)
        self.assertEqual(toc_block.get("wrapped_entry_count"), 0)
        self.assertEqual(toc_block.get("promoted_text_block_ids"), ["txt_p23_001", "txt_p23_002"])
        self.assertFalse(toc_block["toc_diagnostics"]["review_required"])
        self.assertEqual(toc_block["toc_diagnostics"]["review_item_count"], 0)
        self.assertAlmostEqual(float(toc_block["toc_diagnostics"]["locator_coverage_ratio"]), 1.0, places=3)

        only_entry = toc_block["entries"][0]
        self.assertEqual(only_entry["outline_index"], "5.5.1")
        self.assertEqual(only_entry["text"], "Safety Outcome Measures")
        self.assertEqual(only_entry["page_locator"], "21")
        self.assertIsNone(only_entry.get("parent_entry_index"))
        self.assertEqual(only_entry.get("sequence_parent_outline_index"), "5.5")
        self.assertEqual(only_entry.get("parent_sequence_entry_index"), 25)

        toc_blocks = [block for block in self.page23["blocks"] if block["block_type"] == "toc"]
        self.assertEqual(len(toc_blocks), 2)
        self.assertEqual(toc_blocks[0]["toc_id"], toc_block["toc_id"])
        self.assertTrue(toc_blocks[0]["title_inferred"])

        text_labels = [
            block["text"]
            for block in self.page23["blocks"]
            if block["block_type"] == "text"
        ]
        self.assertEqual(
            text_labels,
            ["Figure VI-7: Part of a CMC Table of Contents. This is the third level TOC."],
        )

    def test_page23_lower_outline_is_preserved_as_independent_toc_block(self) -> None:
        self.assertEqual(len(self.page23_toc_blocks), 2)

        toc_block = self.page23_lower_toc
        self.assertEqual(toc_block.get("semantic_role"), "toc_outline")
        self.assertFalse(toc_block.get("is_business_table"))
        self.assertEqual(toc_block.get("title"), "TABLE OF CONTENTS")
        self.assertEqual(toc_block.get("entry_count"), 31)
        self.assertEqual(toc_block.get("raw_entry_row_count"), 38)
        self.assertEqual(toc_block.get("wrapped_entry_count"), 7)
        self.assertEqual(toc_block.get("max_entry_level"), 4)
        self.assertEqual(toc_block.get("max_outline_depth"), 4)
        self.assertEqual(toc_block.get("missing_page_locator_count"), 0)
        self.assertEqual(toc_block["entries"][0]["text"], "LIST OF ABBREVIATIONS")
        self.assertEqual(toc_block["entries"][0]["page_locator"], "vii")
        self.assertEqual(toc_block["entries"][0]["page_locator_kind"], "roman")
        self.assertEqual(toc_block["entries"][0]["page_locator_value"], 7)
        self.assertEqual(toc_block["entries"][1]["outline_index"], "1.0")
        self.assertEqual(toc_block["entries"][1]["text"], "CHEMISTRY")
        self.assertEqual(toc_block["entries"][1]["page_locator_kind"], "arabic")
        self.assertEqual(toc_block["entries"][1]["page_locator_value"], 1)

        semantic_signals = toc_block.get("semantic_signals", {})
        self.assertGreaterEqual(float(semantic_signals.get("page_locator_ratio", 0.0)), 0.75)
        self.assertGreaterEqual(int(semantic_signals.get("locator_indent_level_count", 0)), 3)
        self.assertTrue(semantic_signals.get("toc_title_support"))
        self.assertFalse(semantic_signals.get("has_schema_header"))
        self.assertEqual(toc_block.get("page_locator_kinds"), {"arabic": 30, "roman": 1, "unknown": 0})

        toc_diagnostics = toc_block.get("toc_diagnostics", {})
        self.assertFalse(toc_diagnostics.get("review_required"))
        self.assertEqual(toc_diagnostics.get("review_item_count"), 0)
        self.assertEqual(
            toc_diagnostics.get("issue_counts"),
            {
                "missing_page_locator": 0,
                "unresolved_outline_parent": 0,
                "parent_locator_regression": 0,
                "reading_order_page_regression": 0,
                "empty_heading_text": 0,
            },
        )
        self.assertEqual(toc_diagnostics.get("aggregated_review_item_count"), 0)
        self.assertEqual(toc_diagnostics.get("aggregated_issue_counts"), {})
        self.assertAlmostEqual(float(toc_diagnostics.get("locator_coverage_ratio", 0.0)), 1.0, places=3)
        self.assertTrue(toc_diagnostics.get("mixed_page_numbering"))

        aggregated_review_items = toc_diagnostics.get("aggregated_review_items", [])
        self.assertEqual(aggregated_review_items, [])

        entry_111 = next(entry for entry in toc_block["entries"] if entry.get("outline_index") == "1.1.1")
        self.assertEqual(
            entry_111["text"],
            "Construction of Genes Encoding the Light and Heavy Chain Variable Regions",
        )
        self.assertEqual(entry_111["source_row_indices"], [5, 6])
        self.assertTrue(entry_111["has_wrapped_rows"])
        self.assertEqual(entry_111["parent_entry_index"], 3)
        self.assertEqual(entry_111["outline_parent_index"], "1.1")

        entry_112 = next(entry for entry in toc_block["entries"] if entry.get("outline_index") == "1.1.2")
        self.assertEqual(
            entry_112["text"],
            "Construction of the Light and Heavy Chain Plasmids, p1933 and p1937",
        )
        self.assertEqual(entry_112["source_row_indices"], [7, 8])

        entry_123 = next(entry for entry in toc_block["entries"] if entry.get("outline_index") == "1.2.3")
        self.assertEqual(
            entry_123["text"],
            "Preparation and Characterization of the Clone Master Cell Bank and Working Cell Bank",
        )
        self.assertEqual(entry_123["source_row_indices"], [19, 20])

        entry_11 = next(entry for entry in toc_block["entries"] if entry.get("outline_index") == "1.1")
        self.assertEqual(entry_11["parent_entry_index"], 2)
        self.assertEqual(entry_11["outline_parent_index"], "1.0")

        entry_1141 = next(entry for entry in toc_block["entries"] if entry.get("outline_index") == "1.1.4.1")
        self.assertEqual(entry_1141["level"], 4)
        self.assertEqual(entry_1141["page_locator"], "10")
        self.assertEqual(entry_1141["audit_flags"], [])
        self.assertFalse(entry_1141["review_required"])
        self.assertEqual(entry_1141["section_anchor_entry_index"], 7)
        self.assertEqual(entry_1141["section_anchor_outline_index"], "1.1.4")
        self.assertEqual(entry_1141["section_anchor_text"], "Cloning and Expression of Soluble")

        toc_blocks = [block for block in self.page23["blocks"] if block["block_type"] == "toc"]
        self.assertEqual(len(toc_blocks), 2)
        self.assertEqual(toc_blocks[1]["toc_id"], toc_block["toc_id"])
        self.assertEqual(toc_blocks[1]["title"], "TABLE OF CONTENTS")
        self.assertFalse(toc_blocks[1]["title_inferred"])
        self.assertEqual(toc_blocks[1]["entry_count"], toc_block["entry_count"])
        self.assertEqual(toc_blocks[1]["wrapped_entry_count"], toc_block["wrapped_entry_count"])
        self.assertEqual(toc_blocks[1]["missing_page_locator_count"], toc_block["missing_page_locator_count"])
        self.assertFalse(toc_blocks[1]["review_required"])
        self.assertEqual(toc_blocks[1]["review_item_count"], toc_diagnostics["review_item_count"])
        self.assertEqual(toc_blocks[1]["aggregated_review_item_count"], toc_diagnostics["aggregated_review_item_count"])

        self.assertEqual(self.result["metadata"]["review_required_toc_count"], 0)
        self.assertEqual(self.result["metadata"]["toc_review_item_count"], 0)
        self.assertEqual(self.result["metadata"]["aggregated_toc_review_item_count"], 0)

    def test_document_level_toc_sequences_keep_page22_and_page23_independent(self) -> None:
        self.assertEqual(self.result["metadata"]["toc_sequence_count"], 3)
        self.assertEqual(self.result["metadata"]["multi_page_toc_sequence_count"], 1)

        page22_toc = self.page22_toc_blocks[0]
        page23_top_toc = self.page23_top_toc
        page23_lower_toc = self.page23_lower_toc
        self.assertEqual(page22_toc.get("toc_sequence_id"), page23_top_toc.get("toc_sequence_id"))
        self.assertNotEqual(page23_top_toc.get("toc_sequence_id"), page23_lower_toc.get("toc_sequence_id"))

        page22_sequence = self.toc_sequences[page22_toc["toc_sequence_id"]]
        page23_sequence = self.toc_sequences[page23_lower_toc["toc_sequence_id"]]
        self.assertEqual(page22_sequence["pages"], [22, 23])
        self.assertEqual(page23_sequence["pages"], [23])
        self.assertEqual(page22_sequence["entry_count"], 26)
        self.assertEqual(page23_sequence["entry_count"], 31)
        self.assertEqual(page22_sequence["title"], "TABLE OF CONTENTS")
        self.assertEqual(page23_sequence["title"], "TABLE OF CONTENTS")
        self.assertEqual(page22_sequence["root_entry_count"], 7)
        self.assertEqual(page23_sequence["root_entry_count"], 2)
        self.assertEqual(page22_sequence["navigation_summary"]["cross_page_parent_link_count"], 1)
        self.assertEqual(
            page22_sequence["navigation_summary"]["outline_path_lookup"]["5.0 > 5.5 > 5.5.1"],
            [26],
        )
        self.assertEqual(
            page22_sequence["navigation_summary"]["page_entry_spans"],
            [
                {
                    "page": 22,
                    "toc_ids": [page22_toc["toc_id"]],
                    "entry_count": 25,
                    "first_sequence_entry_index": 1,
                    "last_sequence_entry_index": 25,
                    "root_entry_indices": [1, 2, 3, 4, 10, 11, 14],
                    "cross_page_parent_entry_count": 0,
                },
                {
                    "page": 23,
                    "toc_ids": [page23_top_toc["toc_id"]],
                    "entry_count": 1,
                    "first_sequence_entry_index": 26,
                    "last_sequence_entry_index": 26,
                    "root_entry_indices": [],
                    "cross_page_parent_entry_count": 1,
                },
            ],
        )
        self.assertEqual(page23_sequence["root_nodes"][1]["outline_index"], "1.0")

    def test_workbench_projection_keeps_toc_image_and_semantic_table_objects(self) -> None:
        pdf_document = self.workbench["pdf_document"]
        self.assertIsNotNone(pdf_document)
        self.assertEqual(len(pdf_document["toc_blocks"]), 4)
        self.assertEqual(len(pdf_document["toc_sequences"]), 3)
        self.assertEqual(len(pdf_document["image_blocks"]), 1)
        self.assertEqual(len(pdf_document["table_asts"]), 9)

        image1 = next(image for image in pdf_document["image_blocks"] if image["image_id"] == self.image1["image_id"])
        self.assertEqual(image1["page"], 8)
        self.assertEqual(image1["caption_text"], "Figure 1: IND main folder")

        top_toc = next(toc for toc in pdf_document["toc_blocks"] if toc["toc_id"] == self.page23_top_toc["toc_id"])
        self.assertEqual(top_toc["page"], 23)
        self.assertEqual(top_toc["entry_count"], 1)
        self.assertTrue(top_toc["title_inferred"])

        workbench_sequences = {
            sequence["toc_sequence_id"]: sequence
            for sequence in pdf_document["toc_sequences"]
        }
        page22_sequence = workbench_sequences[self.page22_toc_blocks[0]["toc_sequence_id"]]
        page23_sequence = workbench_sequences[self.page23_lower_toc["toc_sequence_id"]]
        self.assertEqual(page22_sequence["pages"], [22, 23])
        self.assertEqual(page22_sequence["entry_count"], 26)
        self.assertEqual(page22_sequence["title"], "TABLE OF CONTENTS")
        self.assertEqual(page22_sequence["root_entry_count"], 7)
        self.assertEqual(page23_sequence["pages"], [23])
        self.assertEqual(page23_sequence["entry_count"], 31)
        self.assertEqual(page23_sequence["title"], "TABLE OF CONTENTS")
        self.assertEqual(page23_sequence["root_entry_count"], 2)

        roadmap = next(table for table in pdf_document["table_asts"] if table["table_id"] == "tbl_003")
        self.assertEqual(roadmap["page"], 19)
        self.assertEqual(
            (roadmap.get("semantic_compaction") or {}).get("strategy"),
            "leading_key_carry_forward",
        )
        self.assertTrue(
            (roadmap.get("row_texts") or [""])[1].startswith("IND 12345.0003 | 04-Jul-2001 | 1571 |")
        )


if __name__ == "__main__":
    unittest.main()
