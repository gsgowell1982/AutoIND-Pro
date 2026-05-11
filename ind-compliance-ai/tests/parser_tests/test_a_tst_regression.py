from __future__ import annotations

import os
from pathlib import Path
import re
import unittest

from api.main import _build_workbench
from parsers.pdf_parser import parse_pdf


def _resolve_a_tst_regression_pdf() -> Path:
    override = os.environ.get("IND_A_TST_REGRESSION_PDF", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "A-tst.pdf"


def _compact_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").lower())


class ATstRegressionTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_path = _resolve_a_tst_regression_pdf()
        if not cls.sample_path.exists():
            raise unittest.SkipTest(
                "A-tst regression sample not found. "
                "Set IND_A_TST_REGRESSION_PDF or place A-tst.pdf under the workspace root."
            )
        cls.result = parse_pdf(cls.sample_path)
        cls.metadata = cls.result["metadata"]
        cls.page1 = next(page for page in cls.result["pages"] if page["page_number"] == 1)
        cls.page3 = next(page for page in cls.result["pages"] if page["page_number"] == 3)
        cls.page4 = next(page for page in cls.result["pages"] if page["page_number"] == 4)
        cls.page5 = next(page for page in cls.result["pages"] if page["page_number"] == 5)
        cls.page7 = next(page for page in cls.result["pages"] if page["page_number"] == 7)
        cls.page8 = next(page for page in cls.result["pages"] if page["page_number"] == 8)
        cls.page9 = next(page for page in cls.result["pages"] if page["page_number"] == 9)
        cls.page10 = next(page for page in cls.result["pages"] if page["page_number"] == 10)
        cls.workbench = _build_workbench(
            [
                {
                    **cls.result,
                    "file_id": "file_a_tst_regression",
                    "filename": cls.sample_path.name,
                }
            ],
            [{"id": "file_a_tst_regression", "filename": cls.sample_path.name}],
            [],
            None,
        )

    def test_metadata_baseline(self) -> None:
        self.assertEqual(self.metadata["page_count"], 10)
        self.assertEqual(self.metadata["parser_hint"], "pdf-ast-v5")
        self.assertGreaterEqual(self.metadata["table_count"], 4)
        self.assertGreaterEqual(self.metadata["content_evidence_count"], 1)

    def test_page1_publication_front_matter_does_not_survive_as_business_tables_or_figures(self) -> None:
        page1_tables = [table for table in self.result["table_asts"] if table.get("page") == 1]
        page1_images = [image for image in self.result["image_blocks"] if image.get("page") == 1]
        page1_figures = [figure for figure in self.result["figures"] if figure.get("page") == 1]

        self.assertEqual(page1_tables, [])
        self.assertEqual(page1_images, [])
        self.assertEqual(page1_figures, [])

        page1_text = str(self.page1.get("text", ""))
        self.assertIn("Contents lists available at ScienceDirect", page1_text)
        self.assertIn("journal homepage: www.elsevier.com/locate/yjtbi", page1_text)
        self.assertIn("a r t i c l e i n f o", page1_text)
        self.assertIn("a b s t r a c t", page1_text)

    def test_page5_algorithm_and_dataset_page_is_not_promoted_to_toc(self) -> None:
        page5_toc_blocks = [toc for toc in self.result.get("toc_blocks", []) if toc.get("page") == 5]
        self.assertEqual(page5_toc_blocks, [])
        self.assertEqual(self.page5.get("toc_count"), 0)

        page5_tables = [table for table in self.result["table_asts"] if table.get("page") == 5]
        self.assertEqual(len(page5_tables), 1)
        page5_text = str(self.page5.get("text", ""))
        self.assertIn("3.5. Computational complexity", page5_text)
        self.assertIn("Table 1", page5_text)

    def test_page5_keeps_only_right_lane_dataset_table_without_algorithm_pollution(self) -> None:
        page5_tables = [table for table in self.result["table_asts"] if table.get("page") == 5]
        self.assertEqual(len(page5_tables), 1)

        table = page5_tables[0]
        self.assertEqual(
            _compact_text(str(table.get("title", ""))),
            "table1summaryofdatasetsusedintheexperiments",
        )
        self.assertEqual(
            [cell["text"] for cell in table["header"]],
            ["Dataset", "#Instances", "#Features", "#Classes"],
        )

        row_text_blob = "\n".join(table.get("row_texts", []))
        display_row_text_blob = "\n".join(table.get("display_row_texts", []))
        self.assertNotIn("Output: Feature weights W", row_text_blob)
        self.assertNotIn("Update Y so that", row_text_blob)
        self.assertNotIn("Calculate feature weights", row_text_blob)
        self.assertNotIn("Algorithm 1", row_text_blob)
        self.assertNotIn("5. End", row_text_blob)
        self.assertIn("CLL-SUB-111 | 111 | 11340 | 3", display_row_text_blob)
        self.assertIn("Breast | 95 | 4869 | 3", row_text_blob)
        self.assertIn("DLBCL | 77 | 7129 | 2", row_text_blob)

    def test_page7_figure_bbox_does_not_absorb_caption_or_footer_text(self) -> None:
        page7_images = [image for image in self.result.get("image_blocks", []) if image.get("page") == 7]
        self.assertEqual(len(page7_images), 1)
        image = page7_images[0]

        self.assertLess(image["bbox"][3], 682.6)
        self.assertLessEqual(float(image["bbox"][1]), 72.0)
        self.assertEqual(
            image.get("caption_text"),
            "Fig. 1. Relationship between feature dimension and recognition rate (average accuracy and standard deviation) (20 random results of 5-fold CV).",
        )
        embedded_text = str(image.get("embedded_text", ""))
        self.assertNotIn("Fig. 1. Relationship between feature dimension", embedded_text)
        self.assertNotIn("S. Guo et al. / Journal of Theoretical Biology", embedded_text)
        self.assertNotIn("than RLR on most datasets.", embedded_text)

    def test_page7_bottom_tail_rows_remain_two_column_text_not_cross_merged(self) -> None:
        page7_text = str(self.page7.get("text", ""))
        self.assertIn("than RLR on most datasets. The MSVM-RFE method, which uses", page7_text)
        self.assertIn("RLR requires much less computational time than MSVM-RFE on", page7_text)
        self.assertIn("RFE to deal with noise and redundant features, does not perform", page7_text)
        self.assertIn("most datasets. One main issue of F-test is its capability of ad-", page7_text)
        self.assertIn("as well as RLR on four datasets. Moreover, it should be noticed that", page7_text)
        self.assertIn("dressing the noise and outliners. RLR signiﬁcantly outperforms", page7_text)
        self.assertNotIn(
            "than RLR on most datasets. The MSVM-RFE method, which uses RLR requires much less computational time than MSVM-RFE on",
            page7_text,
        )
        self.assertNotIn(
            "RFE to deal with noise and redundant features, does not perform most datasets. One main issue of F-test is its capability of ad-",
            page7_text,
        )

    def test_page3_page4_do_not_emit_trivial_fragmented_equation_blocks(self) -> None:
        trivial_fragment_patterns = [
            re.compile(r"^[ijkmn]\s*=\s*\d+$"),
            re.compile(r"^[ijkmn]\s*=\s*\d+\s*[⎝⎞]?$"),
            re.compile(r"^[ijkmn]\s*=\s*\d+\s+[ijkmn]\s*=\s*\d+$"),
        ]

        for page_number, page in ((3, self.page3), (4, self.page4)):
            equations = [
                str(equation.get("text", "")).strip()
                for equation in self.result.get("equation_blocks", [])
                if equation.get("page") == page_number
            ]
            self.assertTrue(
                all(
                    not any(pattern.fullmatch(text) for pattern in trivial_fragment_patterns)
                    for text in equations
                ),
                msg=f"page {page_number} still contains trivial fragmented equation blocks: {equations}",
            )

            page_text = str(page.get("text", ""))
            self.assertIn("min", page_text)

    def test_page3_top_body_lines_do_not_cross_merge_across_two_columns(self) -> None:
        page3_text = str(self.page3.get("text", ""))
        self.assertNotIn("opti-\ndensity estimation.", page3_text)
        self.assertNotIn(
            "mization problem efﬁciently by using some iterative techniques. the centroid of the class",
            page3_text,
        )
        self.assertNotIn("potential where", page3_text)
        self.assertNotIn("kernel th element of", page3_text)
        self.assertIn("promising results. These methods can solve the L1 regularized opti-", page3_text)
        self.assertIn("density estimation. Instead of calculating the mean of the samples,", page3_text)

    def test_page3_recovers_seven_numbered_display_equations(self) -> None:
        page3_equations = [
            equation for equation in self.result.get("equation_blocks", []) if equation.get("page") == 3
        ]
        self.assertEqual(self.page3.get("equation_count"), 7)
        self.assertEqual(len(page3_equations), 7)

        labels = sorted(
            str(equation.get("equation_label", "") or "").strip()
            for equation in page3_equations
        )
        self.assertEqual(labels, ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)"])

        equation_text_blob = "\n".join(
            _compact_text(str(equation.get("text", "")))
            for equation in page3_equations
        )
        self.assertIn(_compact_text("S w ="), equation_text_blob)
        self.assertIn(_compact_text("c j ="), equation_text_blob)
        self.assertIn(_compact_text("d w"), equation_text_blob)
        self.assertIn(_compact_text("min J w"), equation_text_blob)
        self.assertNotIn(_compact_text("is the corresponding class labels"), equation_text_blob)

    def test_page2_narrative_body_is_not_misclassified_as_table(self) -> None:
        page2 = next(page for page in self.result["pages"] if page["page_number"] == 2)
        page2_tables = [table for table in self.result["table_asts"] if table.get("page") == 2]
        self.assertEqual(page2_tables, [])
        self.assertEqual(page2.get("table_count"), 0)
        page2_text = str(page2.get("text", ""))
        self.assertIn("This section gives a brief review of existing methods", page2_text)
        self.assertIn("Filter methods", page2_text)
        self.assertIn("select features by information of the features", page2_text)

    def test_page2_last_body_line_does_not_cross_merge_across_two_columns(self) -> None:
        page2 = next(page for page in self.result["pages"] if page["page_number"] == 2)
        page2_text = str(page2.get("text", ""))
        self.assertIn("methods can be divided into three types: ﬁlters, wrappers, and", page2_text)
        self.assertIn("classiﬁcation problems for high-dimensional data and have shown", page2_text)
        self.assertNotIn(
            "methods can be divided into three types: ﬁlters, wrappers, and classiﬁcation problems for high-dimensional data and have shown",
            page2_text,
        )

    def test_page3_page4_formula_and_algorithm_regions_do_not_become_business_tables(self) -> None:
        page3_tables = [table for table in self.result["table_asts"] if table.get("page") == 3]
        page4_tables = [table for table in self.result["table_asts"] if table.get("page") == 4]
        self.assertEqual(page3_tables, [])
        self.assertEqual(page4_tables, [])
        self.assertEqual(self.page3.get("table_count"), 0)
        self.assertEqual(self.page4.get("table_count"), 0)
        self.assertIn("The proposed method", str(self.page3.get("text", "")))
        self.assertIn("In order to transform (7) into a convex optimization problem", str(self.page4.get("text", "")))

    def test_page3_formula5_does_not_consume_introductory_right_column_text(self) -> None:
        page3_text = str(self.page3.get("text", ""))
        page3_equations = [
            equation for equation in self.result.get("equation_blocks", []) if equation.get("page") == 3
        ]
        equation5 = next(
            equation
            for equation in page3_equations
            if str(equation.get("equation_label", "") or "").strip() == "(5)"
        )
        equation5_text = str(equation5.get("text", ""))
        self.assertNotIn("is d w", equation5_text)

        self.assertIn("For the purpose of this paper,", page3_text)
        self.assertIn("deﬁned as the following:", page3_text)
        self.assertNotIn("For the purpose of this paper,", equation5_text)
        self.assertNotIn("deﬁned as the following:", equation5_text)

    def test_page3_right_column_inline_math_sentence_keeps_class_label_definition_as_text(self) -> None:
        page3_text = str(self.page3.get("text", ""))
        self.assertIn("Y = [", page3_text)
        self.assertIn("the corresponding class labels", page3_text)
        self.assertIn("d ≫ n", page3_text)
        self.assertIn("centroid of the class is deﬁned as follows:", page3_text)

        page3_equation_text = "\n".join(
            str(equation.get("text", ""))
            for equation in self.result.get("equation_blocks", [])
            if equation.get("page") == 3
        )
        self.assertNotIn("the corresponding class labels", page3_equation_text)
        self.assertNotIn("centroid of the class is deﬁned as follows:", page3_equation_text)

    def test_page3_inline_math_sentence_is_preserved_as_one_text_block_for_y_definition_row(self) -> None:
        page3_text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text" and evidence.get("page") == 3
        ]
        self.assertTrue(
            any(
                "Y = [" in str(evidence.get("content_text", ""))
                and "the corresponding class labels" in str(evidence.get("content_text", ""))
                for evidence in page3_text_evidence
            ),
            msg="page 3 right-column Y-definition row is still split across multiple text blocks",
        )

    def test_page4_recovers_six_numbered_display_equations_and_keeps_theorem_and_algorithm_as_text(self) -> None:
        page4_equations = [
            equation for equation in self.result.get("equation_blocks", []) if equation.get("page") == 4
        ]
        self.assertEqual(self.page4.get("equation_count"), 6)
        self.assertEqual(len(page4_equations), 6)

        labels = sorted(
            str(equation.get("equation_label", "") or "").strip()
            for equation in page4_equations
        )
        self.assertEqual(labels, ["(10)", "(11)", "(12)", "(13)", "(8)", "(9)"])

        equation_text_by_label = {
            str(equation.get("equation_label", "") or "").strip(): _compact_text(str(equation.get("text", "")))
            for equation in page4_equations
        }
        self.assertIn(_compact_text("K w"), equation_text_by_label["(8)"])
        self.assertIn(_compact_text("L w"), equation_text_by_label["(9)"])
        self.assertIn(_compact_text("s t w"), equation_text_by_label["(10)"])
        self.assertIn(_compact_text("F v"), equation_text_by_label["(11)"])
        self.assertIn(_compact_text("i = 1"), equation_text_by_label["(11)"])
        self.assertIn(_compact_text("v ( + ) k 1"), equation_text_by_label["(12)"])
        self.assertIn(_compact_text("d ( )"), equation_text_by_label["(13)"])
        self.assertIn(_compact_text("where β ="), equation_text_by_label["(13)"])
        self.assertIn(_compact_text("g k is the gradient of"), equation_text_by_label["(13)"])
        self.assertFalse(
            str(next(eq for eq in page4_equations if str(eq.get("equation_label") or "").strip() == "(8)").get("text", "")).startswith("n ")
        )
        self.assertFalse(
            str(next(eq for eq in page4_equations if str(eq.get("equation_label") or "").strip() == "(9)").get("text", "")).startswith("n ")
        )
        self.assertFalse(
            str(next(eq for eq in page4_equations if str(eq.get("equation_label") or "").strip() == "(11)").get("text", "")).startswith("n ")
        )
        self.assertNotIn(
            "deﬁned by:",
            str(next(eq for eq in page4_equations if str(eq.get("equation_label") or "").strip() == "(13)").get("text", "")),
        )

        equation_text_blob = "\n".join(str(equation.get("text", "")) for equation in page4_equations)
        self.assertNotIn("Theorem.", equation_text_blob)
        self.assertNotIn("Proof.", equation_text_blob)
        self.assertNotIn("Algorithm 1.", equation_text_blob)
        self.assertNotIn("Initialization:", equation_text_blob)
        self.assertNotIn("Output: Feature weights", equation_text_blob)
        self.assertNotIn("Compute F", equation_text_blob)
        self.assertNotIn("Repeat", equation_text_blob)
        self.assertNotIn("Until", equation_text_blob)

        page4_text = str(self.page4.get("text", ""))
        self.assertIn("Theorem. Let", page4_text)
        self.assertIn("Proof. According to Sun et al. (2010),", page4_text)
        self.assertIn("Algorithm 1. Regularized Logistic Regression for binary-problems", page4_text)
        self.assertIn("Initialization:", page4_text)

    def test_page4_and_page5_promote_algorithm_pseudocode_into_structured_algorithm_blocks(self) -> None:
        algorithm_blocks = list(self.result.get("algorithm_blocks", []) or [])
        page4_algorithms = [block for block in algorithm_blocks if block.get("page") == 4]
        page5_algorithms = [block for block in algorithm_blocks if block.get("page") == 5]

        self.assertEqual(len(page4_algorithms), 2)
        self.assertEqual(len(page5_algorithms), 1)
        self.assertEqual(self.page4.get("algorithm_count"), 2)
        self.assertEqual(self.page5.get("algorithm_count"), 1)

        algorithm1 = next(block for block in page4_algorithms if block.get("algorithm_ref") == "Algorithm 1")
        self.assertEqual(
            algorithm1.get("title"),
            "Algorithm 1. Regularized Logistic Regression for binary-problems",
        )
        self.assertIn("Initialization:", str(algorithm1.get("content_text", "")))
        self.assertIn("Output: Feature weights w", str(algorithm1.get("content_text", "")))
        self.assertIn("11. w i ( k ) = ( v i ( k ) ) 2 , 1 ≤ i ≤ d .", str(algorithm1.get("content_text", "")))
        self.assertNotIn(
            "The pseudo code of our algorithm is presented in Algorithm 1.",
            str(algorithm1.get("content_text", "")),
        )

        algorithm2_page4 = next(block for block in page4_algorithms if block.get("algorithm_ref") == "Algorithm 2")
        self.assertTrue(bool(algorithm2_page4.get("continues_to_next_page")))
        self.assertIn("Initialization: S is the training samples", str(algorithm2_page4.get("content_text", "")))

        algorithm2_page5 = page5_algorithms[0]
        self.assertEqual(algorithm2_page5.get("algorithm_ref"), "Algorithm 2")
        self.assertTrue(bool(algorithm2_page5.get("continued_from_previous_page")))
        self.assertIn("Output: Feature weights W", str(algorithm2_page5.get("content_text", "")))
        self.assertIn("1. For r¼1 to g do", str(algorithm2_page5.get("content_text", "")))
        self.assertIn("W r = Algorithm 1", str(algorithm2_page5.get("content_text", "")))
        self.assertFalse(bool(algorithm2_page5.get("continues_to_next_page")))

        page4_algorithm_nodes = [
            block for block in next(page for page in self.result["document_ast"]["pages"] if page["page"] == 4)["blocks"]
            if block.get("block_type") == "algorithm"
        ]
        page5_algorithm_nodes = [
            block for block in next(page for page in self.result["document_ast"]["pages"] if page["page"] == 5)["blocks"]
            if block.get("block_type") == "algorithm"
        ]
        self.assertEqual(len(page4_algorithm_nodes), 2)
        self.assertEqual(len(page5_algorithm_nodes), 1)

        algorithm_evidence = [
            item for item in self.result.get("content_evidence", [])
            if item.get("source_type") == "algorithm"
        ]
        self.assertEqual(len(algorithm_evidence), 3)
        self.assertTrue(
            any(item.get("source_id") == algorithm2_page5.get("algorithm_id") for item in algorithm_evidence)
        )

        algorithm_units = [
            unit for unit in self.result.get("content_units", [])
            if unit.get("source_type") == "algorithm"
        ]
        self.assertGreaterEqual(len(algorithm_units), 3)
        self.assertTrue(all(not bool(unit.get("fact_extraction_eligible", True)) for unit in algorithm_units))

    def test_page5_content_units_are_anchored_to_local_section_headings(self) -> None:
        heading_42 = next(
            unit
            for unit in self.result.get("content_units", [])
            if unit.get("page") == 5 and unit.get("text") == "4.2. Experimental setup"
        )
        heading_context = dict(heading_42.get("section_context", {}) or {})
        self.assertEqual(heading_context.get("outline_index"), "4.2")
        self.assertEqual(heading_context.get("section_title"), "Experimental setup")
        self.assertEqual(heading_context.get("anchor_source"), "heading")

        body_42 = next(
            unit
            for unit in self.result.get("content_units", [])
            if unit.get("page") == 5
            and unit.get("text") == "The experiments use the classiﬁcation accuracy as the main"
        )
        body_context = dict(body_42.get("section_context", {}) or {})
        self.assertEqual(body_context.get("outline_index"), "4.2")
        self.assertEqual(body_context.get("section_title"), "Experimental setup")
        self.assertEqual(body_context.get("anchor_source"), "heading")

        algorithm2_continuation = next(
            unit
            for unit in self.result.get("content_units", [])
            if unit.get("source_type") == "algorithm"
            and unit.get("page") == 5
            and unit.get("text") == "Algorithm 2. Regularized logistic regression for multiclass problems"
        )
        algorithm_context = dict(algorithm2_continuation.get("section_context", {}) or {})
        self.assertEqual(algorithm_context.get("outline_index"), "3.4")
        self.assertEqual(algorithm_context.get("section_title"), "Feature selection for multiclass problems")

    def test_page4_lower_right_column_does_not_keep_stray_equation_residue_in_multiclass_paragraph(self) -> None:
        page4_text = str(self.page4.get("text", ""))
        self.assertIn("sub-problems by using OVA techniques.", page4_text)
        self.assertNotIn("Fi-\nk is\nnally", page4_text)
        self.assertNotIn("\nd k is\n", page4_text)
        self.assertNotIn("\nk is\n", page4_text)

    def test_page10_references_do_not_become_tables_or_toc(self) -> None:
        page10_tables = [table for table in self.result["table_asts"] if table.get("page") == 10]
        page10_toc_blocks = [toc for toc in self.result.get("toc_blocks", []) if toc.get("page") == 10]
        self.assertEqual(page10_tables, [])
        self.assertEqual(page10_toc_blocks, [])
        page10_text = str(self.page10.get("text", ""))
        self.assertIn("Oh, I.S., Lee, J.S., Moon, B.R., 2004.", page10_text)

    def test_page8_keeps_only_real_table3_with_full_title(self) -> None:
        page8_tables = [table for table in self.result["table_asts"] if table.get("page") == 8]
        self.assertEqual(len(page8_tables), 1)

        table = page8_tables[0]
        self.assertEqual(
            table.get("title"),
            "Table 3 CPU time per run (in seconds) of ﬁve algorithms performed on eight datasets.",
        )
        self.assertEqual(
            [cell["text"] for cell in table["header"]],
            ["Dataset", "RLR", "RFS", "MSVM-RFE", "LLFS", "F-test", "KernelPLS", "mRMR"],
        )
        self.assertEqual(self.page8.get("table_count"), 1)

    def test_page8_vector_figure_is_recovered_from_caption_and_drawings(self) -> None:
        page8_images = [image for image in self.result.get("image_blocks", []) if image.get("page") == 8]
        page8_figures = [figure for figure in self.result.get("figures", []) if figure.get("page") == 8]

        self.assertEqual(len(page8_images), 1)
        self.assertEqual(len(page8_figures), 1)

        image = page8_images[0]
        self.assertEqual(
            image.get("caption_text"),
            "Fig. 2. Sensibility of the regularization parameter λ.",
        )
        self.assertLess(float(image["bbox"][3]), float(image["caption_bbox"][1]))
        self.assertLessEqual(float(image.get("caption_gap", 999.0)), 20.0)
        self.assertLessEqual(float(image["bbox"][0]), 80.0)
        self.assertLessEqual(float(image["bbox"][1]), 260.0)
        self.assertGreaterEqual(float(image["bbox"][2]), 550.0)
        self.assertGreaterEqual(float(image["bbox"][3]), 590.0)
        self.assertGreater(float(image["width"]), 450.0)
        self.assertGreater(float(image["height"]), 300.0)

    def test_page9_table4_title_and_header_are_preserved(self) -> None:
        page9_tables = [table for table in self.result["table_asts"] if table.get("page") == 9]
        self.assertEqual(len(page9_tables), 1)

        table = page9_tables[0]
        self.assertEqual(table.get("title"), "Table 4 Statistical signiﬁcance test.")
        self.assertEqual(
            [cell["text"] for cell in table["header"]],
            [
                "Dataset",
                "RLR vs. RFS",
                "RLR vs. MSVM-RFE",
                "RLR vs. F-test",
                "RLR vs. LLFS",
                "RLR vs. KernelPLS",
                "RLR vs. mRMR",
            ],
        )
        self.assertEqual(self.page9.get("table_count"), 1)

    def test_page9_vector_figure_is_recovered_from_caption_and_drawings(self) -> None:
        page9_images = [image for image in self.result.get("image_blocks", []) if image.get("page") == 9]
        page9_figures = [figure for figure in self.result.get("figures", []) if figure.get("page") == 9]

        self.assertEqual(len(page9_images), 1)
        self.assertEqual(len(page9_figures), 1)

        image = page9_images[0]
        self.assertEqual(
            image.get("caption_text"),
            "Fig. 3. Sensibility of the kernel width σ.",
        )
        self.assertLess(float(image["bbox"][3]), float(image["caption_bbox"][1]))
        self.assertLessEqual(float(image.get("caption_gap", 999.0)), 20.0)
        self.assertLessEqual(float(image["bbox"][0]), 80.0)
        self.assertLessEqual(float(image["bbox"][1]), 90.0)
        self.assertGreaterEqual(float(image["bbox"][2]), 540.0)
        self.assertGreaterEqual(float(image["bbox"][3]), 405.0)
        self.assertGreater(float(image["width"]), 450.0)
        self.assertGreater(float(image["height"]), 300.0)

    def test_page4_equations_11_and_13_absorb_symbol_only_fragments_and_keep_fuller_bboxes(self) -> None:
        page4_equations = [
            equation for equation in self.result.get("equation_blocks", []) if equation.get("page") == 4
        ]
        equation11 = next(
            equation
            for equation in page4_equations
            if str(equation.get("equation_label", "") or "").strip() == "(11)"
        )
        equation13 = next(
            equation
            for equation in page4_equations
            if str(equation.get("equation_label", "") or "").strip() == "(13)"
        )

        self.assertIn("⎝ j ⎠ ⎠", str(equation11.get("text", "")))
        self.assertGreater(float(equation11["bbox"][2]), 240.0)
        self.assertGreater(float(equation13["bbox"][2]), 280.0)
        self.assertTrue(str(equation13.get("text", "")).strip().startswith("( ) d g"))

        page4_text_units = [
            unit
            for unit in self.result.get("content_units", [])
            if unit.get("page") == 4 and unit.get("source_type") == "text"
        ]
        page4_symbol_fragments = {str(unit.get("text", "")).strip() for unit in page4_text_units}
        self.assertNotIn("⎝", page4_symbol_fragments)
        self.assertNotIn("⎝ j ⎠ ⎠", page4_symbol_fragments)

    def test_workbench_projection_keeps_a_tst_page5_non_toc_state(self) -> None:
        pdf_document = self.workbench["pdf_document"]
        self.assertIsNotNone(pdf_document)
        assert pdf_document is not None

        page5_toc_blocks = [toc for toc in pdf_document["toc_blocks"] if toc["page"] == 5]
        page5_tables = [table for table in pdf_document["table_asts"] if table["page"] == 5]
        self.assertEqual(page5_toc_blocks, [])
        self.assertEqual(len(page5_tables), 1)


if __name__ == "__main__":
    unittest.main()
