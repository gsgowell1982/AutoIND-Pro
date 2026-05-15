from __future__ import annotations

import os
from pathlib import Path
import re
import unittest

from api.main import _build_full_markdown, _build_workbench
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

    def test_page1_publication_front_matter_keeps_abstract_before_body(self) -> None:
        markdown = _build_full_markdown([self.result])

        article_info_index = markdown.index("a r t i c l e i n f o")
        abstract_heading_index = markdown.index("a b s t r a c t")
        article_history_index = markdown.index("Article history:")
        abstract_body_index = markdown.index(
            "For classiﬁcation problems based on microarray data"
        )
        body_heading_index = markdown.index("1. Introduction")

        self.assertLess(article_info_index, article_history_index)
        self.assertLess(article_history_index, abstract_heading_index)
        self.assertLess(abstract_heading_index, abstract_body_index)
        self.assertLess(abstract_body_index, body_heading_index)
        self.assertNotIn(
            "Gene selection\n\nstate-of-the-art methods.\n\n1. Introduction",
            markdown,
        )

    def test_page1_publication_author_notes_and_citation_footer_are_metadata(self) -> None:
        page1_units = [
            unit for unit in self.result["content_units"]
            if unit.get("page") == 1
        ]

        corresponding_author = next(
            unit for unit in page1_units
            if "Corresponding author" in str(unit.get("text", ""))
        )
        issn_footer = next(
            unit for unit in page1_units
            if str(unit.get("text", "")).startswith("0022-5193/")
        )

        self.assertEqual(corresponding_author.get("semantic_role"), "author_note")
        self.assertEqual(corresponding_author.get("unit_role"), "publication_metadata")
        self.assertFalse(corresponding_author.get("fact_extraction_eligible"))
        self.assertEqual(issn_footer.get("semantic_role"), "publication_footer")
        self.assertEqual(issn_footer.get("unit_role"), "publication_metadata")
        self.assertFalse(issn_footer.get("fact_extraction_eligible"))

        eligible_text = "\n".join(
            str(unit.get("text", ""))
            for unit in page1_units
            if unit.get("fact_extraction_eligible")
        )
        heading_text = "\n".join(
            str(unit.get("text", ""))
            for unit in page1_units
            if unit.get("unit_role") == "section_heading"
        )
        self.assertIn("1. Introduction", heading_text)
        self.assertNotIn("Corresponding author", eligible_text)
        self.assertNotIn("E-mail address:", eligible_text)
        self.assertNotIn("http://dx.doi.org/", eligible_text)
        self.assertNotIn("0022-5193/&", eligible_text)

    def test_page1_markdown_surfaces_literature_front_matter_metadata_without_fact_eligibility(self) -> None:
        markdown = _build_full_markdown([self.result])

        keyword_index = markdown.index("Keywords:")
        abstract_index = markdown.index("a b s t r a c t")
        keyword_slice = markdown[keyword_index:abstract_index]
        self.assertIn("Class centroid", keyword_slice)
        self.assertIn("Microarray data", keyword_slice)
        self.assertIn("Classi", keyword_slice)
        self.assertIn("L1 regularization", keyword_slice)
        self.assertIn("Gene selection", keyword_slice)
        self.assertIn("Shun Guo", markdown)
        self.assertIn("Donghui Guo", markdown)
        self.assertIn("School of Mathematics and Computer Science", markdown)
        self.assertIn("Corresponding author", markdown)
        self.assertIn("E-mail address: shun.guo@siat.ac.cn", markdown)

        page1_units = [
            unit for unit in self.result["content_units"]
            if unit.get("page") == 1
        ]
        publication_metadata_units = [
            unit for unit in page1_units
            if unit.get("semantic_role") in {
                "author_line",
                "author_affiliation",
                "author_note",
                "contact_email",
                "keyword_metadata",
            }
        ]
        self.assertTrue(publication_metadata_units)
        self.assertTrue(
            all(not bool(unit.get("fact_extraction_eligible", True)) for unit in publication_metadata_units)
        )

        eligible_text = "\n".join(
            str(unit.get("text", ""))
            for unit in page1_units
            if unit.get("fact_extraction_eligible")
        )
        self.assertNotIn("Keywords:", eligible_text)
        self.assertNotIn("Microarray data", eligible_text)
        self.assertNotIn("Gene selection", eligible_text)
        self.assertNotIn("Shun Guo", eligible_text)

    def test_full_markdown_keeps_publication_footer_out_of_body_flow(self) -> None:
        markdown = _build_full_markdown([self.result])

        body_heading_index = markdown.index("1. Introduction")
        related_work_index = markdown.index("2. Related work")
        first_body_slice = markdown[body_heading_index:related_work_index]

        self.assertNotIn("n Corresponding author.", first_body_slice)
        self.assertNotIn("E-mail address: shun.guo@siat.ac.cn", first_body_slice)
        self.assertNotIn("http://dx.doi.org/10.1016/j.jtbi.2016.03.034", first_body_slice)
        self.assertNotIn("0022-5193/& 2016 Elsevier Ltd. All rights reserved.", first_body_slice)
        self.assertNotIn("S. Guo et al. / Journal of Theoretical Biology 400", markdown)
        self.assertIn("method was proposed by Tibshirani, which adds L1 regularized", first_body_slice)

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

    def test_a_tst_recovers_all_thirteen_numbered_display_equations_with_visible_label_regions(self) -> None:
        equations = [
            equation for equation in self.result.get("equation_blocks", []) or []
            if str(equation.get("equation_label", "") or "").strip()
        ]
        self.assertEqual(len(equations), 13)

        labels = sorted(
            (
                str(equation.get("equation_label", "") or "").strip()
                for equation in equations
            ),
            key=lambda label: int(re.search(r"\d+", label).group(0)) if re.search(r"\d+", label) else 0,
        )
        self.assertEqual(labels, [f"({index})" for index in range(1, 14)])

        for equation in equations:
            label = str(equation.get("equation_label", "") or "").strip()
            label_bbox = equation.get("equation_label_bbox")
            self.assertIsInstance(label_bbox, list, msg=f"{label} has no label bbox")
            self.assertEqual(len(label_bbox), 4, msg=f"{label} label bbox is malformed")
            equation_bbox = equation.get("bbox") or []
            self.assertEqual(len(equation_bbox), 4, msg=f"{label} equation bbox is malformed")
            self.assertGreaterEqual(
                float(equation_bbox[2]),
                float(label_bbox[2]) - 0.5,
                msg=f"{label} visible marker is outside the equation evidence bbox",
            )
            self.assertLessEqual(
                float(equation_bbox[0]),
                float(label_bbox[0]) + 0.5,
                msg=f"{label} visible marker is outside the equation evidence bbox",
            )

    def test_display_equations_expose_structured_formula_spans_and_separate_ocr_bbox(self) -> None:
        equation1 = next(
            equation
            for equation in self.result.get("equation_blocks", []) or []
            if str(equation.get("equation_label", "") or "").strip() == "(1)"
        )

        self.assertEqual(equation1.get("layout_label"), "display_formula")
        self.assertEqual(equation1.get("formula_kind"), "display")
        self.assertEqual(equation1.get("evidence_bbox"), equation1.get("bbox"))
        self.assertEqual(equation1.get("formula_number"), "1")
        self.assertEqual(equation1.get("latex_tag"), r"\tag{1}")

        label_bbox = equation1.get("equation_label_bbox") or []
        ocr_bbox = equation1.get("ocr_bbox") or []
        self.assertEqual(len(label_bbox), 4)
        self.assertEqual(len(ocr_bbox), 4)
        self.assertLess(float(ocr_bbox[2]), float(label_bbox[0]))

        spans = equation1.get("formula_spans") or []
        self.assertEqual([span.get("type") for span in spans], ["display_equation", "formula_number"])
        self.assertEqual(spans[0].get("bbox"), ocr_bbox)
        self.assertIn("S w =", str(spans[0].get("text") or ""))
        self.assertEqual(spans[1].get("text"), "(1)")
        self.assertEqual(spans[1].get("latex_tag"), r"\tag{1}")

        page3 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 3)
        equation1_node = next(
            block
            for block in page3["blocks"]
            if block.get("block_type") == "equation"
            and str(block.get("equation_label", "") or "").strip() == "(1)"
        )
        self.assertEqual(equation1_node.get("formula_spans"), spans)
        self.assertEqual(equation1_node.get("ocr_bbox"), ocr_bbox)

    def test_display_formula_ocr_bbox_excludes_embedded_explanatory_text_rows(self) -> None:
        equations_by_label = {
            str(equation.get("equation_label", "") or "").strip(): equation
            for equation in self.result.get("equation_blocks", []) or []
            if equation.get("page") in {3, 4}
        }
        equation3 = equations_by_label["(3)"]
        equation13 = equations_by_label["(13)"]

        equation3_evidence_bbox = equation3.get("evidence_bbox") or equation3.get("bbox") or []
        equation3_ocr_bbox = equation3.get("ocr_bbox") or []
        self.assertEqual(len(equation3_ocr_bbox), 4)
        self.assertLess(float(equation3_evidence_bbox[1]), 350.0)
        self.assertGreater(
            float(equation3_ocr_bbox[1]),
            float(equation3_evidence_bbox[1]) + 10.0,
            msg="OCR crop should trim the prose row ending in 'defined as:' from the formula image evidence",
        )
        self.assertLess(float(equation3_ocr_bbox[1]), float(equation3_evidence_bbox[3]))

        equation13_evidence_bbox = equation13.get("evidence_bbox") or equation13.get("bbox") or []
        equation13_ocr_bbox = equation13.get("ocr_bbox") or []
        self.assertEqual(len(equation13_ocr_bbox), 4)
        self.assertLess(float(equation13_evidence_bbox[1]), 680.0)
        self.assertGreater(
            float(equation13_ocr_bbox[1]),
            float(equation13_evidence_bbox[1]) + 20.0,
            msg="OCR crop should trim preceding natural-language rows from a numbered display formula",
        )
        self.assertLess(float(equation13_ocr_bbox[0]), 49.0)
        self.assertLess(float(equation13_ocr_bbox[2]), float(equation13.get("equation_label_bbox", [999.0])[0]))

    def test_page3_equation1_following_inline_math_definition_is_reconstructed_as_text(self) -> None:
        page3_text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text" and evidence.get("page") == 3
        ]
        self.assertTrue(
            any(
                "set, respectively, i.e." in str(evidence.get("content_text", ""))
                and "and u =" in str(evidence.get("content_text", ""))
                and "x i ." in str(evidence.get("content_text", ""))
                for evidence in page3_text_evidence
            ),
            msg="page 3 inline math definition after equation (1) is still split into small text fragments",
        )

        page3_equation_text = "\n".join(
            str(equation.get("text", ""))
            for equation in self.result.get("equation_blocks", [])
            if equation.get("page") == 3
        )
        self.assertNotIn("set, respectively, i.e.", page3_equation_text)
        self.assertNotIn("and u =", page3_equation_text)

    def test_page3_equation1_following_inline_math_definition_exposes_structured_math_text(self) -> None:
        inline_math_evidence = next(
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
            and evidence.get("page") == 3
            and "set, respectively, i.e." in str(evidence.get("content_text", ""))
        )

        display_text = str(
            inline_math_evidence.get("display_text")
            or inline_math_evidence.get("math_text")
            or inline_math_evidence.get("content_text")
        )
        self.assertIn("u_j", display_text)
        self.assertIn("\\frac{1}{n_j}", display_text)
        self.assertIn("\\sum_{i=1}^{n_j}", display_text)
        self.assertIn("x_i^{(j)}", display_text)
        self.assertIn("and u = \\frac{1}{n}", display_text)
        self.assertIn("\\sum_{i=1}^{n}", display_text)
        self.assertNotIn("u j = n 1 j", display_text)
        self.assertNotIn("∑= n i j 1", display_text)

    def test_page3_equation1_following_inline_math_definition_exposes_inline_formula_span(self) -> None:
        inline_math_evidence = next(
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
            and evidence.get("page") == 3
            and "set, respectively, i.e." in str(evidence.get("content_text", ""))
        )

        inline_spans = inline_math_evidence.get("inline_formula_spans") or []
        self.assertEqual(len(inline_spans), 1)
        inline_span = inline_spans[0]
        self.assertEqual(inline_span.get("type"), "inline_equation")
        self.assertEqual(inline_span.get("source"), "pdf_text_layer_2d_reconstruction")
        self.assertEqual(inline_span.get("render"), "latex_inline")
        self.assertIn("u_j", str(inline_span.get("content") or ""))
        self.assertIn(r"\frac{1}{n_j}", str(inline_span.get("content") or ""))
        self.assertIn(r"\sum_{i=1}^{n}", str(inline_span.get("content") or ""))
        self.assertNotIn("set, respectively", str(inline_span.get("content") or ""))
        self.assertNotIn("i.e.", str(inline_span.get("content") or ""))
        self.assertEqual(len(inline_span.get("bbox") or []), 4)

    def test_page3_non_anchor_inline_formula_sentences_expose_inline_formula_spans(self) -> None:
        page3_text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text" and evidence.get("page") == 3
        ]
        kernel_sentence = next(
            evidence
            for evidence in page3_text_evidence
            if "k ( d )= exp" in str(evidence.get("content_text", ""))
        )
        weighted_norm_sentence = next(
            evidence
            for evidence in page3_text_evidence
            if "with U W =" in str(evidence.get("content_text", ""))
        )

        kernel_spans = kernel_sentence.get("inline_formula_spans") or []
        weighted_norm_spans = weighted_norm_sentence.get("inline_formula_spans") or []

        self.assertTrue(kernel_spans, msg=kernel_sentence)
        self.assertTrue(weighted_norm_spans, msg=weighted_norm_sentence)
        self.assertIn("k", str(kernel_spans[0].get("content") or ""))
        self.assertIn("exp", str(kernel_spans[0].get("content") or ""))
        self.assertNotIn("has been used", str(kernel_spans[0].get("content") or ""))
        self.assertIn("U", str(weighted_norm_spans[0].get("content") or ""))
        self.assertIn("\\sum", str(weighted_norm_spans[0].get("content") or ""))
        self.assertNotIn("with ", str(weighted_norm_spans[0].get("content") or ""))
        self.assertEqual(kernel_spans[0].get("layout_label"), "inline_formula")
        self.assertEqual(weighted_norm_spans[0].get("render"), "latex_inline")

    def test_inline_formula_spans_exclude_following_explanatory_prose(self) -> None:
        inline_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text" and evidence.get("inline_formula_spans")
        ]
        x_definition = next(
            evidence
            for evidence in inline_evidence
            if evidence.get("page") == 3
            and "denotes the i-th sample" in str(evidence.get("content_text", ""))
        )
        theorem_condition = next(
            evidence
            for evidence in inline_evidence
            if evidence.get("page") == 4
            and "is found through" in str(evidence.get("content_text", ""))
        )
        proof_tail = next(
            evidence
            for evidence in inline_evidence
            if evidence.get("page") == 4
            and "The proof is then complete" in str(evidence.get("content_text", ""))
        )

        x_content = str((x_definition.get("inline_formula_spans") or [{}])[0].get("content") or "")
        theorem_content = str((theorem_condition.get("inline_formula_spans") or [{}])[0].get("content") or "")
        proof_content = str((proof_tail.get("inline_formula_spans") or [{}])[0].get("content") or "")

        self.assertIn("x_i^{(j)}", x_content)
        self.assertNotIn("denotes", x_content)
        self.assertNotIn("sample", x_content)
        self.assertNotIn("is found through", theorem_content)
        self.assertNotIn("The proof", proof_content)
        self.assertNotIn("complete", proof_content)

    def test_inline_formula_spans_are_only_emitted_for_formula_core_candidates(self) -> None:
        text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
        ]
        display_equation_evidence = [
            evidence
            for evidence in text_evidence
            if evidence.get("semantic_role") == "display_equation"
        ]
        self.assertTrue(display_equation_evidence)
        self.assertTrue(
            all(not evidence.get("inline_formula_spans") for evidence in display_equation_evidence),
            msg=display_equation_evidence,
        )

        plain_variable_definitions = [
            evidence
            for evidence in text_evidence
            if "where g is the number of classes" in str(evidence.get("content_text", ""))
            or "samples, where xi is the i-th data sample" in str(evidence.get("content_text", ""))
        ]
        self.assertEqual(len(plain_variable_definitions), 2)
        g_definition = next(
            evidence
            for evidence in plain_variable_definitions
            if "where g is the number of classes" in str(evidence.get("content_text", ""))
        )
        x_i_definition = next(
            evidence
            for evidence in plain_variable_definitions
            if "samples, where xi is the i-th data sample" in str(evidence.get("content_text", ""))
        )
        self.assertFalse(g_definition.get("inline_formula_spans"), msg=g_definition)
        x_i_spans = x_i_definition.get("inline_formula_spans") or []
        self.assertTrue(x_i_spans, msg=x_i_definition)
        self.assertEqual(x_i_spans[0].get("content"), "x_i")
        self.assertFalse(x_i_spans[0].get("ocr_candidate"))
        self.assertEqual(x_i_spans[0].get("formula_complexity"), "inline_symbol")

        if_then_definition = next(
            evidence
            for evidence in text_evidence
            if "where y = [ y 1" in str(evidence.get("content_text", ""))
        )
        if_then_spans = if_then_definition.get("inline_formula_spans") or []
        self.assertTrue(if_then_spans, msg=if_then_definition)
        if_then_content = str(if_then_spans[0].get("content") or "")
        self.assertIn("y", if_then_content)
        self.assertNotIn("If", if_then_content)
        self.assertTrue(if_then_spans[0].get("ocr_candidate"))

        initial_point_definition = next(
            evidence
            for evidence in text_evidence
            if "gradient descent with an initial point" in str(evidence.get("content_text", ""))
        )
        initial_point_spans = initial_point_definition.get("inline_formula_spans") or []
        self.assertTrue(initial_point_spans, msg=initial_point_definition)
        initial_point_content = str(initial_point_spans[0].get("content") or "")
        self.assertIn("x_i", initial_point_content)
        self.assertNotIn("then", initial_point_content)
        self.assertNotIn("x *", initial_point_content)
        self.assertTrue(initial_point_spans[0].get("ocr_candidate"))

    def test_page3_equation1_markdown_keeps_image_primary_and_exposes_latex_policy(self) -> None:
        equation1 = next(
            equation
            for equation in self.result.get("equation_blocks", [])
            if str(equation.get("equation_label", "") or "").strip() == "(1)"
        )
        self.assertEqual(equation1.get("latex_render_policy"), "image_primary_latex_enhancement")
        self.assertEqual(equation1.get("latex_confidence"), 0.0)
        self.assertIsNone(equation1.get("latex_text"))
        self.assertEqual(equation1.get("latex_source"), "disabled")
        self.assertFalse(equation1.get("latex_validation", {}).get("accepted", True))
        self.assertIn("no_latex_candidate", equation1.get("latex_validation", {}).get("issues", []))

        page3 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 3)
        equation1_node = next(
            block
            for block in page3["blocks"]
            if block.get("block_type") == "equation"
            and str(block.get("equation_label", "") or "").strip() == "(1)"
        )
        self.assertEqual(equation1_node.get("latex_render_policy"), "image_primary_latex_enhancement")
        self.assertEqual(equation1_node.get("latex_source"), "disabled")
        self.assertFalse(equation1_node.get("latex_validation", {}).get("accepted", True))

        markdown = _build_full_markdown([self.result])
        equation_index = markdown.index("**公式 (1)**")
        inline_index = markdown.index("set, respectively, i.e., u_j")
        equation_slice = markdown[equation_index:inline_index]

        self.assertIn("![公式 (1)](data:image/png;base64,", equation_slice)
        self.assertIn("```text", equation_slice)
        self.assertIn("S w =", equation_slice)
        self.assertIn("LaTeX 重建未达到高置信度，视觉核对以原文截图为准。", equation_slice)
        self.assertNotIn("$$", equation_slice)
        self.assertIn("\\frac{1}{n_j}", markdown[inline_index:inline_index + 300])

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

    def test_two_column_pages_expose_reading_order_diagnostics_for_review(self) -> None:
        for page_number in (2, 3, 7):
            page = next(page for page in self.result["pages"] if page["page_number"] == page_number)
            diagnostics = page.get("reading_order_diagnostics") or {}
            self.assertEqual(
                diagnostics.get("strategy"),
                "zone_columns_left_then_right",
                msg=f"page {page_number} should use column-aware reading order diagnostics",
            )
            self.assertGreaterEqual(
                int(diagnostics.get("column_zone_count", 0) or 0),
                1,
                msg=f"page {page_number} should expose at least one column reading zone",
            )
            self.assertGreater(
                int(diagnostics.get("left_block_count", 0) or 0),
                0,
                msg=f"page {page_number} should expose left-lane text evidence",
            )
            self.assertGreater(
                int(diagnostics.get("right_block_count", 0) or 0),
                0,
                msg=f"page {page_number} should expose right-lane text evidence",
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
        self.assertGreater(float(equation5["bbox"][1]), 490.0)
        self.assertLess(float(equation5["bbox"][3]), 520.0)
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
            equation for equation in self.result.get("equation_blocks", [])
            if equation.get("page") == 4
            and str(equation.get("equation_label", "") or "").strip()
        ]
        self.assertGreaterEqual(self.page4.get("equation_count"), 6)
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
        self.assertNotIn(_compact_text("where β ="), equation_text_by_label["(13)"])
        self.assertNotIn(_compact_text("g k is the gradient of"), equation_text_by_label["(13)"])
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

    def test_page4_theorem_condition_is_promoted_as_unnumbered_display_equation(self) -> None:
        page4_blocks = next(
            page for page in self.result["document_ast"]["pages"] if page["page"] == 4
        )["blocks"]
        unnumbered_equations = [
            block for block in page4_blocks
            if block.get("block_type") == "equation"
            and not str(block.get("equation_label", "") or "").strip()
        ]
        self.assertTrue(
            any("F v v v" in str(block.get("text", "")) for block in unnumbered_equations),
            msg=unnumbered_equations,
        )

        markdown = _build_full_markdown([self.result])
        theorem_index = markdown.index("Theorem. Let F ( v )")
        proof_index = markdown.index("Proof. According to Sun et al.")
        theorem_slice = markdown[theorem_index:proof_index]
        self.assertRegex(theorem_slice, r"\*\*(?:公式|鍏紡)\*\*")
        self.assertIn("F v v v = * 0 =", theorem_slice)
        self.assertNotIn("\n\n( )\n\n∂\n\n∂\n\nF v v v = * 0 =", theorem_slice)

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

    def test_publication_running_headers_and_page_numbers_stay_out_of_body_markdown(self) -> None:
        markdown = _build_full_markdown([self.result])

        self.assertNotIn("S. Guo et al. / Journal of Theoretical Biology 400", markdown)
        for page_number in range(33, 42):
            self.assertNotRegex(markdown, rf"(?m)^{page_number}$")

        metadata_units = [
            unit for unit in self.result.get("content_units", [])
            if unit.get("page") in range(2, 11)
            and (
                "Journal of Theoretical Biology" in str(unit.get("text", ""))
                or str(unit.get("text", "")).strip() in {str(value) for value in range(33, 42)}
            )
        ]
        self.assertTrue(metadata_units)
        self.assertTrue(all(unit.get("unit_role") == "metadata" for unit in metadata_units))
        self.assertTrue(all(not bool(unit.get("fact_extraction_eligible", True)) for unit in metadata_units))

    def test_table_captions_and_header_rows_are_owned_by_tables_not_body_text(self) -> None:
        self.assertEqual(self.metadata["table_count"], 4)
        self.assertGreaterEqual(self.metadata["table_text_suppressed_count"], 20)

        table_text = "\n".join(
            " ".join(
                [
                    str(table.get("title", "")),
                    " ".join(str(cell.get("text", "")) for cell in table.get("header", []) or []),
                ]
            )
            for table in self.result.get("table_asts", [])
        )
        self.assertIn("Table 1", table_text)
        self.assertIn("Table 2", table_text)
        self.assertIn("Table 3", table_text)
        self.assertIn("Table 4", table_text)
        self.assertIn("Dataset", table_text)
        self.assertIn("MSVM-RFE", table_text)

        body_text_by_page = {
            page_number: "\n".join(
                str(evidence.get("content_text", ""))
                for evidence in self.result.get("content_evidence", [])
                if evidence.get("source_type") == "text" and evidence.get("page") == page_number
            )
            for page_number in (5, 6, 8, 9)
        }
        body_lines_by_page = {
            page_number: {line.strip() for line in body_text.splitlines() if line.strip()}
            for page_number, body_text in body_text_by_page.items()
        }

        self.assertNotIn("Table 1 Summary of datasets used in the experiments.", body_lines_by_page[5])
        self.assertNotIn("#Instances", body_text_by_page[5])
        self.assertNotIn(
            "Table 2 Average classiﬁcation accuracy and standard deviations (%) of SVM.",
            body_lines_by_page[6],
        )
        self.assertNotIn(
            "Table 3 CPU time per run (in seconds) of ﬁve algorithms performed on eight datasets.",
            body_lines_by_page[8],
        )
        self.assertNotIn("Table 4 Statistical signiﬁcance test.", body_lines_by_page[9])

        for page_number, body_text in body_text_by_page.items():
            standalone_header_terms = {
                line.strip()
                for line in body_text.splitlines()
                if line.strip() in {"Dataset", "RLR", "RFS", "MSVM-RFE", "LLFS", "F-test", "KernelPLS", "mRMR"}
            }
            self.assertEqual(standalone_header_terms, set(), msg=f"page {page_number} leaked table headers")

    def test_literature_section_headings_do_not_accept_dates_or_grant_numbers(self) -> None:
        heading_units = [
            unit for unit in self.result.get("content_units", [])
            if unit.get("unit_role") == "section_heading"
        ]
        heading_texts = {str(unit.get("text", "")).strip() for unit in heading_units}

        self.assertNotIn("8 March 2016", heading_texts)
        self.assertFalse(
            any(text.startswith("20090121110019") for text in heading_texts),
            msg=sorted(text for text in heading_texts if text.startswith("20090121110019")),
        )

        expected_headings = {
            "1. Introduction",
            "3.1. Class separability measure",
            "4.4.1. Average accuracy of classiﬁcation",
            "4.5. Statistical signiﬁcance test",
            "5. Conclusion",
            "References",
        }
        self.assertTrue(expected_headings.issubset(heading_texts))

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

    def test_page9_table4_keeps_first_data_row_and_normalizes_math_symbol_font_projection(self) -> None:
        table = next(table for table in self.result["table_asts"] if table.get("page") == 9)

        self.assertEqual(table.get("raw_grid", [])[0][0], "CLL-SUB-111")
        self.assertIn("(þ)", str(table.get("raw_grid", [])[0][1]))
        self.assertEqual(table.get("display_grid", [])[0][0], "CLL-SUB-111")
        self.assertEqual(table.get("data_grid", [])[0][0], "CLL-SUB-111")
        self.assertIn(
            [
                "CLL-SUB-111",
                "(+) 0.0335",
                "(+) 0.0478",
                "(+) 1.83e-36",
                "(=)0.9285",
                "(=)0.1384",
                "(+)2.56e-9",
            ],
            table.get("display_grid", []),
        )
        self.assertIn("(-) 2.27e-4", table.get("display_grid", [])[1][1])
        self.assertIn("(=)0.8516", table.get("display_grid", [])[9][1])
        self.assertIn("6(+),3(=),1(-)", table.get("display_grid", [])[10][1])

    def test_page9_table4_markdown_keeps_first_row_and_normalizes_statistical_symbols(self) -> None:
        markdown = _build_full_markdown([self.result])
        table_index = markdown.index("| Dataset | RLR vs. RFS | RLR vs. MSVM-RFE |")
        snippet = markdown[table_index : table_index + 2600]

        self.assertIn(
            "| CLL-SUB-111 | (+) 0.0335 | (+) 0.0478 | (+) 1.83e-36 | (=)0.9285 | (=)0.1384 | (+)2.56e-9 |",
            snippet,
        )
        self.assertIn("| Breast | (-) 2.27e-4 |", snippet)
        self.assertIn("| Lung | (+) 2.07e-8 |", snippet)
        self.assertIn("| DLBCL | (=)0.8516 |", snippet)
        self.assertIn("| Summary | 6(+),3(=),1(-) |", snippet)
        self.assertIn("In the table (+) implies that our method is statistically better", markdown)
        self.assertIn("(=) means that the two methods have no signiﬁcant", markdown)
        self.assertNotIn("(þ)", snippet)
        self.assertNotIn("(¼)", snippet)
        self.assertNotIn("2.07e 8", snippet)
        self.assertNotIn("In the table (þ)", markdown)
        self.assertNotIn("(¼) means", markdown)

    def test_markdown_normalizes_math_symbol_font_artifacts_outside_tables(self) -> None:
        markdown = _build_full_markdown([self.result])
        self.assertIn("1. For r=1 to g do", markdown)
        self.assertIn("k=0 and θ=0.01", markdown)
        self.assertNotIn("1. For r录1 to g do", markdown)
        self.assertNotIn("k录0 and θ=0.01", markdown)
        self.assertNotIn("(镁)", markdown)
        self.assertNotIn("(录)", markdown)

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
            equation for equation in self.result.get("equation_blocks", [])
            if equation.get("page") == 4
            and str(equation.get("equation_label", "") or "").strip()
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
        equation13_text = str(equation13.get("text", "")).strip()
        self.assertTrue(equation13_text.startswith("d") or equation13_text.startswith("( ) d"))
        self.assertIn("g", equation13_text)
        self.assertLess(float(equation13["bbox"][1]), 682.0)
        self.assertLess(float(equation13["bbox"][2]), 300.0)
        self.assertLess(float(equation13["bbox"][3]), 730.0)
        self.assertNotIn("where", equation13_text.lower())
        self.assertNotIn("gradient", equation13_text.lower())
        self.assertNotIn("Algorithm 2", equation13_text)

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
