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
        keyword_index = markdown.index("Keywords:")
        article_history_index = markdown.index("Article history:")
        abstract_body_index = markdown.index(
            "For classiﬁcation problems based on microarray data"
        )
        body_heading_index = markdown.index("1. Introduction")

        self.assertLess(article_info_index, article_history_index)
        self.assertLess(article_history_index, keyword_index)
        self.assertLess(keyword_index, abstract_heading_index)
        self.assertLess(article_history_index, abstract_heading_index)
        self.assertLess(abstract_heading_index, abstract_body_index)
        self.assertLess(abstract_body_index, body_heading_index)
        self.assertNotIn(
            "Gene selection\n\nstate-of-the-art methods.\n\n1. Introduction",
            markdown,
        )
        self.assertNotIn(
            "Gene selection\n\nlysis), such as Fisher Score",
            markdown,
        )

    def test_markdown_keeps_literature_labels_as_standalone_lines(self) -> None:
        markdown = _build_full_markdown([self.result])

        expected_boundaries = (
            ("H I G H L I G H T S", "\x01 A gene selection method is proposed"),
            ("a r t i c l e i n f o", "Article history:"),
            ("a b s t r a c t", "For classiﬁcation problems based on microarray data"),
            ("References", "Argyriou, A., Evgeniou, T., Pontil, M., 2007."),
        )
        for label, following in expected_boundaries:
            self.assertIn(f"{label}\n\n{following}", markdown)
            self.assertNotIn(f"{label} {following}", markdown)

    def test_markdown_reconstructs_scientific_notation_inline_formula_and_float_notes_as_continuous_paragraphs(self) -> None:
        markdown = _build_full_markdown([self.result])

        self.assertIn(r"less than $10^{-5}$", markdown)
        self.assertIn(r"$k(d)=\exp(-d/\sigma)$", markdown)
        self.assertIn(r"and $u$ are the centroid", markdown)
        self.assertIn(r"where $z_i=", markdown)
        self.assertIn(r"and $c_i$, respectively", markdown)
        self.assertIn("The boldfaced values are the highest ones", markdown)
        self.assertIn("The boldfaced values are the best ones", markdown)
        self.assertIn("In the table", markdown)
        self.assertNotIn("$u$sed", markdown)
        self.assertNotIn("10-5", markdown)
        self.assertNotIn("and u is", markdown)

    def test_page1_markdown_keeps_intro_body_out_of_publication_metadata(self) -> None:
        markdown = _build_full_markdown([self.result])

        body_heading_index = markdown.index("1. Introduction")
        lysis_index = markdown.index(
            "lysis), such as Fisher Score (Richard et al., 2001), Laplacian Score"
        )
        trace_ratio_index = markdown.index(
            "(He et al., 2005) and Trace Ratio (Nie et al., 2008). However, these"
        )
        lasso_index = markdown.index(
            "genes (Oh et al., 2004; Bolón-Canedo et al., 2014). The LASSO"
        )
        author_note_index = markdown.index("n Corresponding author.")

        self.assertLess(author_note_index, body_heading_index)
        self.assertGreater(lysis_index, body_heading_index)
        self.assertGreater(trace_ratio_index, body_heading_index)
        self.assertGreater(lasso_index, body_heading_index)

        page1_units = [
            unit for unit in self.result["content_units"]
            if unit.get("page") == 1
        ]
        for expected in (
            "lysis), such as Fisher Score",
            "(He et al., 2005) and Trace Ratio",
            "genes (Oh et al., 2004; Bolón-Canedo",
        ):
            unit = next(unit for unit in page1_units if expected in str(unit.get("text", "")))
            self.assertEqual(unit.get("unit_role"), "body")
            self.assertEqual(unit.get("semantic_role"), "text_block")
            self.assertTrue(unit.get("fact_extraction_eligible"))

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

    def test_display_equation_images_are_preserved_as_evidence_metadata(self) -> None:
        labelled_equations = [
            equation
            for equation in self.result.get("equation_blocks", []) or []
            if str(equation.get("equation_label", "") or "").strip()
        ]

        self.assertEqual(len(labelled_equations), 13)
        for equation in labelled_equations:
            evidence_image = dict(equation.get("evidence_image") or {})
            self.assertEqual(evidence_image.get("source"), "pdf_bbox_crop")
            self.assertEqual(evidence_image.get("render_role"), "visual_evidence")
            self.assertEqual(evidence_image.get("page"), equation.get("page"))
            self.assertEqual(evidence_image.get("bbox"), equation.get("evidence_bbox"))

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
        self.assertGreater(
            float(equation3_evidence_bbox[1]),
            355.0,
            msg="Formula evidence bbox should start at the formula row, not the prose row ending in 'defined as:'",
        )
        self.assertGreater(
            float(equation3_ocr_bbox[1]),
            float(equation3_evidence_bbox[1]) - 2.0,
            msg="OCR crop should trim the prose row ending in 'defined as:' from the formula image evidence",
        )
        self.assertLess(float(equation3_ocr_bbox[1]), float(equation3_evidence_bbox[3]))
        self.assertNotIn("within-class distance are", str(equation3.get("text", "")).lower())
        self.assertNotIn("defined as", str(equation3.get("text", "")).lower())

        page3_body_units = [
            unit
            for unit in self.result.get("content_units", []) or []
            if unit.get("page") == 3
            and unit.get("unit_role") == "body"
            and unit.get("semantic_role") == "text_block"
        ]
        cue_unit = next(
            (
                unit for unit in page3_body_units
                if "within-class distance are de" in str(unit.get("text", ""))
            ),
            None,
        )
        self.assertIsNotNone(cue_unit)
        self.assertTrue(cue_unit.get("fact_extraction_eligible"))

        equation13_evidence_bbox = equation13.get("evidence_bbox") or equation13.get("bbox") or []
        equation13_ocr_bbox = equation13.get("ocr_bbox") or []
        self.assertEqual(len(equation13_ocr_bbox), 4)
        self.assertGreater(float(equation13_evidence_bbox[1]), 700.0)
        self.assertGreater(
            float(equation13_ocr_bbox[1]),
            float(equation13_evidence_bbox[1]) - 2.0,
            msg="OCR crop should trim preceding natural-language rows from a numbered display formula",
        )
        self.assertLess(float(equation13_ocr_bbox[0]), 50.0)
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

    def test_inline_formula_text_layer_reconstruction_handles_common_inline_math_shapes(self) -> None:
        text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text" and evidence.get("inline_formula_spans")
        ]

        class_mean = next(
            evidence
            for evidence in text_evidence
            if evidence.get("page") == 3
            and "Let u j = H 1 j" in str(evidence.get("content_text", ""))
        )
        weighted_norm = next(
            evidence
            for evidence in text_evidence
            if evidence.get("page") == 3
            and "with U W =" in str(evidence.get("content_text", ""))
        )
        squared_weights = next(
            evidence
            for evidence in text_evidence
            if evidence.get("page") == 4
            and "where w j = v j 2" in str(evidence.get("content_text", ""))
        )
        proof_tail = next(
            evidence
            for evidence in text_evidence
            if evidence.get("page") == 4
            and "The proof is then complete" in str(evidence.get("content_text", ""))
        )

        class_mean_content = str((class_mean.get("inline_formula_spans") or [{}])[0].get("content") or "")
        weighted_norm_content = str((weighted_norm.get("inline_formula_spans") or [{}])[0].get("content") or "")
        squared_weights_content = str((squared_weights.get("inline_formula_spans") or [{}])[0].get("content") or "")
        proof_tail_content = str((proof_tail.get("inline_formula_spans") or [{}])[0].get("content") or "")

        self.assertIn("u_j =", class_mean_content)
        self.assertIn(r"\frac{1}{H_j}", class_mean_content)
        self.assertIn(r"\sum_{x_i \in H_j}", class_mean_content)
        self.assertNotIn(r"\sum_{x=i \in }_H_j", class_mean_content)
        self.assertNotIn("Let", class_mean_content)

        self.assertIn("1 \\le i \\le d", weighted_norm_content)
        self.assertIn("w_i \\ge 0", weighted_norm_content)
        self.assertNotIn("≤≤", weighted_norm_content)

        self.assertIn("v_j^2", squared_weights_content)
        self.assertIn("1 \\le j \\le d", squared_weights_content)
        self.assertNotIn("v_j ^2", squared_weights_content)

        self.assertIn("v_i^2", proof_tail_content)
        self.assertIn("1 \\le i \\le d", proof_tail_content)
        self.assertNotIn("≤≤", proof_tail_content)

    def test_text_layer_inline_latex_is_projected_into_markdown_body_text(self) -> None:
        markdown = _build_full_markdown([self.result])

        self.assertIn(r"where $x_i^{(j)}$ denotes the i -th sample", markdown)
        self.assertIn(r"samples, where $x_i$ is the i- th data sample", markdown)
        self.assertIn(
            r"the within-class scatter matrix $S_w$, between-class scatter matrix $S_b$,",
            markdown,
        )
        self.assertIn(r"and Total scatter matrix $S_t$.", markdown)
        self.assertIn(
            r"clude $\operatorname{tr}(S_b)/\operatorname{tr}(S_w)$ and $S_b/S_w$, where $\operatorname{tr}(A)$",
            markdown,
        )
        self.assertIn(
            r"Let $S=[x_1,\ldots,x_n]^T \in R^{n \times d}$ be a training data set containing n",
            markdown,
        )
        self.assertIn(
            r"$Y=[y_1,\ldots,y_n]^T$ is the corresponding class labels",
            markdown,
        )
        self.assertIn(
            r"where $g$ is the number of classes, $x_{ij}$, $c_{ij}$ the j -th element of $x_i$ and",
            markdown,
        )
        self.assertIn(
            r"$c_i$, respectively and $d_w(,)$ is a distance function about w.",
            markdown,
        )
        self.assertIn(r"of distances $d_w(,)$ have been proposed", markdown)
        self.assertIn(
            r"$u_j = \frac{1}{n_j} \sum_{i=1}^{n_j} x_i^{(j)} and u = \frac{1}{n} \sum_{i=1}^{n} x_i.$",
            markdown,
        )
        self.assertIn(r"$u_j = \frac{1}{H_j} \sum_{x_i \in H_j} x_i,$", markdown)
        self.assertIn(r"w_j = v_j^2", markdown)
        self.assertIn(r"1 \le j \le d", markdown)
        self.assertNotIn("**公式项**", markdown)
        self.assertNotIn(r"$^h_x_x =_x * = 0$", markdown)

    def test_body_inline_math_general_patterns_cover_vector_constraints_and_where_clauses(self) -> None:
        markdown = _build_full_markdown([self.result])
        def formula_index(label: str) -> int:
            match = re.search(rf"\\tag\{{{re.escape(label)}\}}", markdown)
            self.assertIsNotNone(match, msg=f"formula {label} not found in markdown")
            return int(match.start())

        weight_index = markdown.index("In order to calculate the weight")
        formula3_index = formula_index("3")
        weight_slice = markdown[weight_index:formula3_index]
        self.assertIn(r"$w=[w_1,w_2,\ldots,w_d]\in R^{1\times d}$ be a weight vector.", weight_slice)
        self.assertNotIn("w = [ w 1 , w 2 , … , w d ]", weight_slice)
        self.assertNotIn("R 1 × d", weight_slice)

        equation9_index = formula_index("9")
        equation10_index = formula_index("10")
        logistic_slice = markdown[equation9_index:equation10_index]
        self.assertIn(r"$z_i$ of (9).", logistic_slice)
        self.assertIn(r"mework to", logistic_slice)
        self.assertIn(r"$z_i$, it has high computational cost", logistic_slice)
        self.assertNotIn("de铿乶e zi", logistic_slice)

        equation11_index = formula_index("11")
        constraint_slice = markdown[equation10_index:equation11_index]
        self.assertIn(r"Two reasons for the constraint $w \ge 0$ are:", constraint_slice)
        self.assertNotIn("Two reasons for the constraint w ≥ 0 are:", constraint_slice)
        self.assertNotIn("\n\n≥\n\n", constraint_slice)

        equation13_index = formula_index("13")
        theorem_index = markdown.index("The following theorem shows")
        formula13_tail = markdown[equation13_index:theorem_index]
        self.assertIn(
            r"where $\beta = \frac{\lVert g^{(k)}\rVert}{\lVert g^{(k-1)}\rVert}$, and",
            formula13_tail,
        )
        self.assertIn(r"$g^{(k)}$ is the gradient of", formula13_tail)
        self.assertIn(r"$F(v)$ at $v^{(k)}$.", formula13_tail)
        self.assertNotIn(r"where $\beta = 1$, and", formula13_tail)
        self.assertNotIn("where β =\n\n1 , and", formula13_tail)

        text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
        ]
        weight_definition = next(
            evidence
            for evidence in text_evidence
            if "w = [ w 1" in str(evidence.get("content_text", ""))
        )
        llfs_definition = next(
            evidence
            for evidence in text_evidence
            if "mework to" in str(evidence.get("content_text", ""))
            and "zi" in str(evidence.get("content_text", ""))
        )
        constraint_definition = next(
            evidence
            for evidence in text_evidence
            if "constraint w" in str(evidence.get("content_text", ""))
        )
        beta_definition = next(
            evidence
            for evidence in text_evidence
            if str(evidence.get("content_text", "")).strip() == "where β ="
        )
        latex_by_evidence = {
            "weight": [span.get("latex_text") for span in weight_definition.get("inline_formula_spans") or []],
            "llfs": [span.get("latex_text") for span in llfs_definition.get("inline_formula_spans") or []],
            "constraint": [span.get("latex_text") for span in constraint_definition.get("inline_formula_spans") or []],
            "beta": [span.get("latex_text") for span in beta_definition.get("inline_formula_spans") or []],
        }
        self.assertIn(r"w=[w_1,w_2,\ldots,w_d]\in R^{1\times d}", latex_by_evidence["weight"])
        self.assertIn(r"z_i", latex_by_evidence["llfs"])
        self.assertIn(r"w \ge 0", latex_by_evidence["constraint"])
        self.assertIn(
            r"\beta = \frac{\lVert g^{(k)}\rVert}{\lVert g^{(k-1)}\rVert}",
            latex_by_evidence["beta"],
        )

    def test_body_inline_math_general_patterns_cover_limits_norms_and_theorem_conditions(self) -> None:
        markdown = _build_full_markdown([self.result])

        note_index = markdown.index("Note that if")
        formula3_index = re.search(r"\\tag\{3\}", markdown)
        self.assertIsNotNone(formula3_index)
        note_slice = markdown[note_index:int(formula3_index.start())]
        self.assertIn(r"Note that if $\sigma \to +\infty$, we have $v_j = u_j$,", note_slice)
        self.assertNotIn("σ→+∞", note_slice)
        self.assertNotIn("v j = u j", note_slice)

        equation13_index_match = re.search(r"\\tag\{13\}", markdown)
        self.assertIsNotNone(equation13_index_match)
        theorem_index = markdown.index("The following theorem shows")
        formula13_tail = markdown[int(equation13_index_match.start()):theorem_index]
        self.assertIn(
            r"where $\beta = \frac{\lVert g^{(k)}\rVert}{\lVert g^{(k-1)}\rVert}$, and",
            formula13_tail,
        )
        self.assertIn(r"$g^{(k)}$ is the gradient of", formula13_tail)
        self.assertIn(r"$F(v)$ at $v^{(k)}$.", formula13_tail)
        self.assertNotIn("where $\\beta = 1$, and", formula13_tail)
        self.assertNotIn("\n\n1 , and\n\n", formula13_tail)
        self.assertNotIn("F v at", formula13_tail)

        theorem_start = markdown.index("Theorem. Let $F(v)$")
        proof_start = markdown.index("Proof. According to Sun et al.")
        theorem_slice = markdown[theorem_start:proof_start]
        self.assertIn(r"If", theorem_slice)
        self.assertIn(r"\frac{\partial F}{\partial v}(v^*) = 0", theorem_slice)
        self.assertNotRegex(theorem_slice, r"\*\*(?:公式|鍏紡|閸忣剙绱?)\*\*")
        self.assertNotIn("![公式]", theorem_slice)
        self.assertNotIn("\n\n∂\n\n∂\n\n", theorem_slice)

        text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
        ]
        limit_definition = next(
            evidence
            for evidence in text_evidence
            if "Note that if" in str(evidence.get("content_text", ""))
        )
        limit_latex = [span.get("latex_text") for span in limit_definition.get("inline_formula_spans") or []]
        self.assertIn(r"\sigma \to +\infty", limit_latex)
        self.assertIn(r"v_j = u_j", limit_latex)

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
        g_spans = g_definition.get("inline_formula_spans") or []
        self.assertEqual(
            [span.get("latex_text") for span in g_spans],
            ["g", "x_{ij}", "c_{ij}", "x_i"],
            msg=g_definition,
        )
        self.assertTrue(all(not span.get("ocr_candidate") for span in g_spans), msg=g_definition)
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

    def test_page4_theorem_math_terms_are_projected_as_readable_latex(self) -> None:
        markdown = _build_full_markdown([self.result])
        theorem_index = markdown.index("Theorem. Let")
        proof_index = markdown.index("Proof. According to Sun et al.")
        theorem_slice = markdown[theorem_index:proof_index]

        self.assertIn(r"Theorem. Let $F(v)$ be a function of v deﬁned in (11). If", theorem_slice)
        self.assertIn(r"\frac{\partial F}{\partial v}(v^*) = 0", theorem_slice)
        self.assertIn(r"If $\frac{\partial F}{\partial v}(v^*) = 0$ then $v^*$ is not a local minimize", theorem_slice)
        self.assertIn(r"point of $F(v)$. Moreover, if", theorem_slice)
        self.assertIn(r"an initial point $v_i^{(0)} \ne 0, 1 \le i \le d$, then $v^*$ is a global minimizer of $F(v)$.", theorem_slice)
        self.assertNotIn("global minimizer of\n\n*", theorem_slice)
        self.assertNotIn("F ( v )", theorem_slice)
        self.assertNotIn("F v v v = * 0 =", theorem_slice)
        self.assertNotIn("v i ( 0 )", theorem_slice)
        self.assertNotRegex(theorem_slice, r"\*\*(?:公式|鍏紡|閸忣剙绱?)\*\*")
        self.assertNotIn("![公式]", theorem_slice)
        self.assertNotIn("≤≤ i d", theorem_slice)

        text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
        ]
        theorem_intro = next(
            evidence
            for evidence in text_evidence
            if "Theorem. Let F ( v )" in str(evidence.get("content_text", ""))
        )
        theorem_initial_point = next(
            evidence
            for evidence in text_evidence
            if "an initial point v i ( 0 )" in str(evidence.get("content_text", ""))
        )
        intro_latex = [span.get("latex_text") for span in theorem_intro.get("inline_formula_spans") or []]
        initial_latex = [span.get("latex_text") for span in theorem_initial_point.get("inline_formula_spans") or []]

        self.assertIn("F(v)", intro_latex)
        self.assertIn(r"v_i^{(0)} \ne 0, 1 \le i \le d", initial_latex)

    def test_page4_proof_math_terms_are_projected_as_readable_latex(self) -> None:
        markdown = _build_full_markdown([self.result])
        proof_index = markdown.index("Proof. According to Sun et al.")
        next_paragraph_index = markdown.index("In the light of Sun et al.", proof_index)
        proof_slice = markdown[proof_index:next_paragraph_index]

        self.assertIn(r"If $f(x)$ is a strictly convex function of $x \in R^d$ and $h(x)=f(y)$,", proof_slice)
        self.assertIn(r"where $y=[y_1,\ldots,y_d]=[x_1^2,\ldots,x_d^2]$. If", proof_slice)
        self.assertIn(r"\frac{\partial h}{\partial x}(x^*) = 0", proof_slice)
        self.assertIn(r"and $x^*$ is found through", proof_slice)
        self.assertIn(r"then $x^*$ is a global minimizer of $h(x)$.", proof_slice)
        self.assertNotIn("h ( x ( )= ( )", proof_slice)
        self.assertNotIn("f y ,", proof_slice)
        self.assertNotIn("] ∂", proof_slice)
        self.assertNotIn("x *", proof_slice)
        self.assertNotIn("global minimizer of h ( x )", proof_slice)

        text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
        ]
        convex_line = next(
            evidence
            for evidence in text_evidence
            if "strictly convex function of x" in str(evidence.get("content_text", ""))
        )
        vector_line = next(
            evidence
            for evidence in text_evidence
            if "where y = [ y 1" in str(evidence.get("content_text", ""))
        )
        minimizer_line = next(
            evidence
            for evidence in text_evidence
            if "global minimizer of h" in str(evidence.get("content_text", ""))
        )
        convex_latex = [span.get("latex_text") for span in convex_line.get("inline_formula_spans") or []]
        vector_latex = [span.get("latex_text") for span in vector_line.get("inline_formula_spans") or []]
        minimizer_latex = [span.get("latex_text") for span in minimizer_line.get("inline_formula_spans") or []]

        self.assertIn("f(x)", convex_latex)
        self.assertIn(r"x \in R^d", convex_latex)
        self.assertIn("h(x)=f(y)", convex_latex)
        self.assertIn(r"y=[y_1,\ldots,y_d]=[x_1^2,\ldots,x_d^2]", vector_latex)
        self.assertIn("h(x)", minimizer_latex)

    def test_page4_indexed_symbol_descriptions_are_projected_as_readable_latex(self) -> None:
        markdown = _build_full_markdown([self.result])
        formula11_index = markdown.index(r"\tag{11}")
        formula11_index = markdown.rfind("$$", 0, formula11_index)
        theorem_index = markdown.index("Theorem. Let", formula11_index)
        formula_slice = markdown[formula11_index:theorem_index]

        self.assertIn(r"where $w_j = v_j^2,1 \le j \le d$ and $z_{ij}$ is the j-th element of $z_i$.", formula_slice)
        self.assertNotIn("and z ij is the j -th element of z i", formula_slice)
        self.assertNotIn("and zij is the j-th element of zi", formula_slice)

        text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
        ]
        z_line = next(
            evidence
            for evidence in text_evidence
            if "zij is the j-th element of zi" in str(evidence.get("content_text", ""))
        )
        z_latex = [span.get("latex_text") for span in z_line.get("inline_formula_spans") or []]
        self.assertIn("z_{ij}", z_latex)
        self.assertIn("z_i", z_latex)

    def test_page5_complexity_terms_are_projected_as_readable_inline_latex(self) -> None:
        markdown = _build_full_markdown([self.result])
        section35_index = markdown.index("3.5. Computational complexity")
        section4_index = markdown.index("4. Experiments and results analysis")
        complexity_slice = markdown[section35_index:section4_index]

        self.assertIn(r"plexities are $O(nd)$ and", complexity_slice)
        self.assertIn(r"$O(Ind)$ respectively.", complexity_slice)
        self.assertIn(r"for binary problems is $O(I(n^2d + nd))$.", complexity_slice)
        self.assertIn(r"g is the number of classes and $g \ge 3$.", complexity_slice)
        self.assertIn(r"complexity of Algorithm 2 is $O((nd + Ind)g)$.", complexity_slice)
        self.assertNotIn("() O Ind", complexity_slice)
        self.assertNotIn("() $O(Ind)$", complexity_slice)
        self.assertNotIn("O I n d", complexity_slice)
        self.assertNotIn("( (2 + nd))", complexity_slice)
        self.assertNotIn("O (( nd + Ind ) g ) . +) )", complexity_slice)
        self.assertNotIn("$O((nd + Ind)g)$. +) )", complexity_slice)

        page5_complexity_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
            and evidence.get("page") == 5
            and "complexity" in str((evidence.get("section_context") or {}).get("section_title") or "").lower()
        ]
        complexity_latex = [
            span.get("latex_text")
            for evidence in page5_complexity_evidence
            for span in evidence.get("inline_formula_spans") or []
        ]
        self.assertIn(r"O(nd)", complexity_latex)
        self.assertIn(r"O(Ind)", complexity_latex)
        self.assertIn(r"O(I(n^2d + nd))", complexity_latex)
        self.assertIn(r"g \ge 3", complexity_latex)
        self.assertIn(r"O((nd + Ind)g)", complexity_latex)

    def test_inline_math_projection_handles_set_membership_vectors_and_formula_residue(self) -> None:
        markdown = _build_full_markdown([self.result])

        definition_index = markdown.index("Deﬁnition 1.")
        equation2_index = markdown.index(r"\tag{2}")
        definition_slice = markdown[definition_index:equation2_index]
        self.assertIn(r"$u_j = \frac{1}{H_j} \sum_{x_i \in H_j} x_i,$", definition_slice)
        self.assertIn(r"where $H_j=\{x_i \mid 1 \le i \le n, y_i=j\}$", definition_slice)
        self.assertIn(r"and $H_j$", definition_slice)
        self.assertNotIn("H j ={ x i | 1 ≤≤ i n y", definition_slice)
        self.assertNotIn(", i = } j and Hj", definition_slice)

        equation5_index = markdown.index(r"\tag{5}")
        equation6_index = markdown.index(r"\tag{6}")
        equation5_slice = markdown[equation5_index:equation6_index]
        equation5_body_after_code = equation5_slice.split("LaTeX 重建未达到高置信度，视觉核对以原文截图为准。", 1)[-1]
        self.assertNotIn("deﬁned as the following:", equation5_slice)
        self.assertNotIn("d w ( x , y ,)= x", equation5_slice)

        equation7_index = markdown.index(r"\tag{7}")
        equation8_index = markdown.index(r"\tag{8}")
        equation7_slice = markdown[equation7_index:equation8_index]
        self.assertIn(
            r"where $z_i=(\lvert x_{i1}-c_{j1}\rvert-\lvert c_{11}-c_{21}\rvert,\ldots,\lvert x_{id}-c_{jd}\rvert-\lvert c_{1d}-c_{2d}\rvert)$,",
            equation7_slice,
        )
        self.assertIn(r"with $x_{ij}$, $c_{ij}$ the j-th element of $x_i$", equation7_slice)
        self.assertIn(r"and $c_i$, respectively, and $x_i \in H_j$.", equation7_slice)
        self.assertNotIn("x_i_1", equation7_slice)
        self.assertNotIn("c_j_1", equation7_slice)
        self.assertNotIn("c_1_d", equation7_slice)

        text_evidence = [
            evidence
            for evidence in self.result.get("content_evidence", [])
            if evidence.get("source_type") == "text"
        ]
        set_definition = next(
            evidence
            for evidence in text_evidence
            if "where H j" in str(evidence.get("content_text", ""))
        )
        z_definition = next(
            evidence
            for evidence in text_evidence
            if "where z i" in str(evidence.get("content_text", ""))
        )
        membership_tail = next(
            evidence
            for evidence in text_evidence
            if "respectively, and x i" in str(evidence.get("content_text", ""))
        )
        set_latex = [span.get("latex_text") for span in set_definition.get("inline_formula_spans") or []]
        z_latex = [span.get("latex_text") for span in z_definition.get("inline_formula_spans") or []]
        membership_latex = [span.get("latex_text") for span in membership_tail.get("inline_formula_spans") or []]
        self.assertIn(r"H_j=\{x_i \mid 1 \le i \le n, y_i=j\}", set_latex)
        self.assertIn(
            r"z_i=(\lvert x_{i1}-c_{j1}\rvert-\lvert c_{11}-c_{21}\rvert,\ldots,\lvert x_{id}-c_{jd}\rvert-\lvert c_{1d}-c_{2d}\rvert)",
            z_latex,
        )
        self.assertIn(r"x_i \in H_j", membership_latex)
        self.assertIn(r"x_i", membership_latex)

    def test_page3_equation1_markdown_keeps_evidence_out_of_user_facing_latex_display(self) -> None:
        equation1 = next(
            equation
            for equation in self.result.get("equation_blocks", [])
            if str(equation.get("equation_label", "") or "").strip() == "(1)"
        )
        self.assertEqual(equation1.get("latex_render_policy"), "latex_primary_with_image_evidence")
        self.assertEqual(dict(equation1.get("evidence_image") or {}).get("render_role"), "visual_evidence")
        self.assertEqual(equation1.get("latex_confidence"), 0.88)
        self.assertIn(r"\begin{cases}", str(equation1.get("latex_text") or ""))
        self.assertIn(r"S_w", str(equation1.get("latex_text") or ""))

        page3 = next(page for page in self.result["document_ast"]["pages"] if page["page"] == 3)
        equation1_node = next(
            block
            for block in page3["blocks"]
            if block.get("block_type") == "equation"
            and str(block.get("equation_label", "") or "").strip() == "(1)"
        )
        self.assertEqual(equation1_node.get("latex_render_policy"), "latex_primary_with_image_evidence")
        self.assertEqual(dict(equation1_node.get("evidence_image") or {}).get("render_role"), "visual_evidence")

        markdown = _build_full_markdown([self.result])
        equation_index = markdown.index(r"\tag{1}")
        inline_index = markdown.index(r"set, respectively, i.e., $u_j")
        equation_slice = markdown[markdown.rfind("$$", 0, equation_index):inline_index]

        self.assertIn("$$", equation_slice)
        self.assertIn(r"\begin{cases}", equation_slice)
        self.assertIn(r"\tag{1}", equation_slice)
        self.assertNotIn("```text", equation_slice)
        self.assertNotIn("PDF", equation_slice)
        self.assertNotIn("LaTeX", equation_slice)
        self.assertIn(r"$u_j = \frac{1}{n_j}", markdown[inline_index:inline_index + 300])

    def test_numbered_display_equations_are_reconstructed_as_latex_for_markdown(self) -> None:
        labelled_equations = {
            str(equation.get("equation_label", "") or "").strip(): equation
            for equation in self.result.get("equation_blocks", []) or []
            if str(equation.get("equation_label", "") or "").strip()
        }

        self.assertEqual(len(labelled_equations), 13)
        for label, equation in labelled_equations.items():
            latex_text = str(equation.get("latex_text") or "")
            self.assertTrue(latex_text, msg=f"{label} has no display LaTeX")
            self.assertGreaterEqual(float(equation.get("latex_confidence") or 0.0), 0.85, msg=label)
            self.assertIn(str(equation.get("latex_tag") or ""), latex_text, msg=label)
            self.assertNotIn("LaTeX", latex_text)
            self.assertNotIn("PDF", latex_text)

        markdown = _build_full_markdown([self.result])
        self.assertNotRegex(markdown, r"\*\*(?:公式|鍏紡)\s*(?:\(\d+\))?\*\*")
        for label in labelled_equations:
            equation_index = markdown.index(str(labelled_equations[label].get("latex_tag") or ""))
            following = markdown[max(0, equation_index - 80):equation_index + 800]
            self.assertIn("$$", following, msg=label)
            self.assertIn(str(labelled_equations[label].get("latex_tag") or ""), following, msg=label)
            self.assertNotIn("![公式", following, msg=label)


    def test_numbered_display_equation_latex_rebuilds_common_math_structures_without_label_residue(self) -> None:
        labelled_equations = {
            str(equation.get("equation_label", "") or "").strip(): str(equation.get("latex_text") or "")
            for equation in self.result.get("equation_blocks", []) or []
            if str(equation.get("equation_label", "") or "").strip()
        }

        equation2 = labelled_equations["(2)"]
        self.assertEqual(equation2.count(r"\sum_{x_i\in H_j}"), 3)
        self.assertIn(
            r"c_j=\sum_{x_i\in H_j}P(x_i=c_j)x_i=\frac{\sum_{x_i\in H_j}k(\lVert x_i-u_j\rVert_2)x_i}{\sum_{x_i\in H_j}k(\lVert x_i-u_j\rVert_2)}",
            equation2,
        )
        self.assertNotIn(r"\lVert x_i-u_j\rVert^2", equation2)

        equation3 = labelled_equations["(3)"]
        self.assertIn(r"S_b=d_w(c_i,c_j)", equation3)
        self.assertIn(r"\sum_{k=1}^{d}", equation3)
        self.assertIn(r"w_k\lvert c_{ik}-c_{jk}\rvert", equation3)
        self.assertNotIn("d_S b", equation3)
        self.assertNotIn("k=1 ( 3 )", equation3)

        equation4 = labelled_equations["(4)"]
        self.assertIn(r"S_w=\sum_{j=1}^{g}\sum_{x_i\in H_j}d_w(x_i,c_j)", equation4)
        self.assertIn(r"\sum_{k=1}^{d}w_k\lvert x_{ik}-c_{jk}\rvert", equation4)
        self.assertNotIn(r"\sum \sum", equation4)
        self.assertNotIn("( 4 )", equation4)

        equation5 = labelled_equations["(5)"]
        self.assertIn(r"d_w(x,y)=\lVert x-y\rVert_w", equation5)
        self.assertNotIn("( 5 )", equation5)
        self.assertNotIn("y_w", equation5)

        equation6 = labelled_equations["(6)"]
        self.assertIn(r"\min_w J(w)=S_w-\gamma S_b", equation6)
        self.assertNotIn("w()", equation6)

        equation11 = labelled_equations["(11)"]
        self.assertIn(r"\min_v F(v)", equation11)
        self.assertIn(r"\sum_{i=1}^{n}", equation11)
        self.assertIn(r"\sum_j v_j^2z_{ij}", equation11)
        self.assertIn(r"\lambda\lVert v\rVert_2^2", equation11)
        self.assertNotIn("v_F", equation11)

        equation12 = labelled_equations["(12)"]
        self.assertIn(r"v^{(k+1)}=v^{(k)}+\alpha^{(k)}d^{(k)}", equation12)
        self.assertNotIn("v ( + )", equation12)

        equation13 = labelled_equations["(13)"]
        self.assertIn(r"d^{(k)}=\begin{cases}", equation13)
        self.assertIn(r"-g^{(k)}", equation13)
        self.assertIn(r"\beta^{(k)}d^{(k-1)}", equation13)
        self.assertNotIn("d - ( k )", equation13)
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

    def test_page4_theorem_condition_stays_in_theorem_prose_not_display_equation(self) -> None:
        page4_blocks = next(
            page for page in self.result["document_ast"]["pages"] if page["page"] == 4
        )["blocks"]
        unnumbered_equations = [
            block for block in page4_blocks
            if block.get("block_type") == "equation"
            and not str(block.get("equation_label", "") or "").strip()
        ]
        self.assertFalse(
            any("F v v v" in str(block.get("text", "")) for block in unnumbered_equations),
            msg=unnumbered_equations,
        )

        markdown = _build_full_markdown([self.result])
        theorem_index = markdown.index("Theorem. Let $F(v)$")
        proof_index = markdown.index("Proof. According to Sun et al.")
        theorem_slice = markdown[theorem_index:proof_index]
        self.assertIn(r"If $\frac{\partial F}{\partial v}(v^*) = 0$ then", theorem_slice)
        self.assertNotIn("![公式]", theorem_slice)
        self.assertNotRegex(theorem_slice, r"\*\*(?:公式|鍏紡|閸忣剙绱?)\*\*")
        self.assertNotIn("F v v v = * 0 =", theorem_slice)
        self.assertNotIn("\n\n( )\n\n∂\n\n∂\n\nF v v v = * 0 =", theorem_slice)

    def test_compact_centroid_subscript_and_theorem_tail_residue_are_repaired_generically(self) -> None:
        markdown = _build_full_markdown([self.result])

        centroid_index = markdown.index("where $x_i^{(j)}$ denotes")
        centroid_slice = markdown[centroid_index:centroid_index + 260]
        self.assertIn("number of classes, $u_j$ and $u$ are the centroid", centroid_slice)
        self.assertNotIn("classes, uj and u", centroid_slice)

        theorem_index = markdown.index("Theorem. Let $F(v)$")
        proof_index = markdown.index("Proof. According to Sun et al.")
        theorem_slice = markdown[theorem_index:proof_index]
        self.assertIn(r"then $v^*$ is a global minimizer of $F(v)$.", theorem_slice)
        self.assertNotIn("\n\n*\n\nProof", markdown[theorem_index:proof_index + 40])

        proof_tail_index = markdown.index(r"$F(v)=L(w)$")
        proof_tail_slice = markdown[proof_tail_index:proof_tail_index + 180]
        self.assertIn(r"$F(v)=L(w)$, where $w_i = v_i^2,1 \le i \le d.$ The proof is then complete.", proof_tail_slice)
        self.assertNotIn("where\n\n)", proof_tail_slice)

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

    def test_algorithm_pseudocode_math_terms_are_projected_as_readable_latex(self) -> None:
        algorithm_blocks = list(self.result.get("algorithm_blocks", []) or [])
        algorithm1 = next(block for block in algorithm_blocks if block.get("algorithm_ref") == "Algorithm 1")
        algorithm2_page5 = next(
            block
            for block in algorithm_blocks
            if block.get("algorithm_ref") == "Algorithm 2" and block.get("page") == 5
        )

        algorithm1_latex = [
            span.get("latex_text")
            for span in algorithm1.get("inline_formula_spans") or []
        ]
        algorithm2_latex = [
            span.get("latex_text")
            for span in algorithm2_page5.get("inline_formula_spans") or []
        ]

        self.assertIn(r"v^{(0)} = w^{(0)} = [1,1,\ldots,1]", algorithm1_latex)
        self.assertIn(r"F^{(0)} = F(v^{(0)})", algorithm1_latex)
        self.assertIn(r"v^{(k+1)} = v^{(k)} + \alpha^{(k)}d^{(k)}", algorithm1_latex)
        self.assertIn(r"v_i^{(k+1)} < 10^{-5}", algorithm1_latex)
        self.assertIn(r"v_i^{(k+1)} = 0", algorithm1_latex)
        self.assertIn(r"F^{(k)} - F^{(k-1)} < \theta", algorithm1_latex)
        self.assertIn(r"w_i^{(k)} = (v_i^{(k)})^2", algorithm1_latex)
        self.assertIn(r"1 \le i \le d", algorithm1_latex)
        self.assertIn(r"c_1,c_2", algorithm1_latex)
        self.assertIn(r"r=1 \text{ to } g", algorithm2_latex)
        self.assertIn(r"W_r = \operatorname{Algorithm 1}(S,Y,\lambda,\sigma,\theta)", algorithm2_latex)
        self.assertIn(r"W = W + W_r", algorithm2_latex)
        self.assertIn(r"W_r", algorithm2_latex)

        algorithm_evidence = [
            item for item in self.result.get("content_evidence", [])
            if item.get("source_type") == "algorithm"
        ]
        algorithm1_evidence = next(item for item in algorithm_evidence if item.get("source_id") == algorithm1.get("algorithm_id"))
        segment_latex = [
            span.get("latex_text")
            for segment in algorithm1_evidence.get("segments") or []
            for span in segment.get("inline_formula_spans") or []
        ]
        self.assertIn(r"v_i^{(k+1)} < 10^{-5}", segment_latex)
        self.assertIn(r"F^{(k)} - F^{(k-1)} < \theta", segment_latex)

        markdown = _build_full_markdown([self.result])
        algorithm1_index = markdown.index("Algorithm 1. Regularized Logistic Regression for binary-problems")
        section35_index = markdown.index("3.5. Computational complexity")
        algorithm_slice = markdown[algorithm1_index:section35_index]

        self.assertIn(r"Set $v^{(0)} = w^{(0)} = [1,1,\ldots,1]$, k=0 and $\theta=0.01$.", algorithm_slice)
        self.assertIn(r"2. Compute $F^{(0)} = F(v^{(0)})$ using Eq. (11);", algorithm_slice)
        self.assertIn(r"1. Compute $c_1,c_2$ using Eq. (2);", algorithm_slice)
        self.assertIn(r"5. Update $v^{(k+1)} = v^{(k)} + \alpha^{(k)}d^{(k)}$, where $\alpha^{(k)}$ is determined via", algorithm_slice)
        self.assertIn(r"6. if $v_i^{(k+1)} < 10^{-5}$, then", algorithm_slice)
        self.assertIn(r"$v_i^{(k+1)} = 0$;", algorithm_slice)
        self.assertIn(r"10. Until $F^{(k)} - F^{(k-1)} < \theta$;", algorithm_slice)
        self.assertIn(r"11. $w_i^{(k)} = (v_i^{(k)})^2$, $1 \le i \le d$.", algorithm_slice)
        self.assertIn(r"1. For $r=1$ to $g$ do", algorithm_slice)
        self.assertIn(r"3. Calculate feature weights $W_r$ by calling", algorithm_slice)
        self.assertIn(r"$W_r = \operatorname{Algorithm 1}(S,Y,\lambda,\sigma,\theta)$", algorithm_slice)
        self.assertIn(r"4. $W = W + W_r$", algorithm_slice)
        self.assertNotIn("cate$g$ories", algorithm_slice)
        self.assertNotIn("1. Compute c c", algorithm_slice)
        self.assertNotIn("feature weights Wr by calling", algorithm_slice)
        self.assertNotIn("v (( + )", algorithm_slice)
        self.assertNotIn("10 − 5", algorithm_slice)
        self.assertNotIn("W r = Algorithm 1 ( S , Y , , , λ σ θ ,)", algorithm_slice)

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

    def test_markdown_merges_wrapped_two_column_body_lines_into_continuous_paragraphs(self) -> None:
        markdown = _build_full_markdown([self.result])

        related_work_index = markdown.index("2. Related work")
        section3_index = markdown.index("3. The proposed method")
        related_work_slice = markdown[related_work_index:section3_index]

        self.assertIn(
            "This section gives a brief review of existing methods that are relevant to our work. "
            "More details can be referred to (Yuan et al., 2012; Chandrashekar and Sahin, 2014).",
            related_work_slice,
        )
        self.assertIn(
            "For example, Laplacian Score (He et al., 2005), Trace Ratio (Nie et al., 2008) "
            "and Similarity Preserving Feature Selection",
            related_work_slice,
        )
        self.assertNotIn(
            "Yuan et al.,\n\n2012; Chandrashekar",
            related_work_slice,
        )
        self.assertNotIn(
            "Trace\n\nRatio",
            related_work_slice,
        )
        self.assertNotIn(
            "Feature Selection\n\n(SPFS)",
            related_work_slice,
        )

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

    def test_markdown_renders_table_titles_and_places_floats_in_reading_order(self) -> None:
        markdown = _build_full_markdown([self.result])

        table_titles = [
            str(table.get("title") or "").strip()
            for table in sorted(
                self.result.get("table_asts", []) or [],
                key=lambda item: str(item.get("table_id") or ""),
            )
            if str(table.get("title") or "").strip()
        ]
        self.assertEqual(len(table_titles), 4)
        for title in table_titles:
            self.assertIn(f"**{title}**", markdown)

        table1_title = next(title for title in table_titles if title.startswith("Table 1"))
        table2_title = next(title for title in table_titles if title.startswith("Table 2"))
        table3_title = next(title for title in table_titles if title.startswith("Table 3"))

        table1_reference = markdown.index("information of these datasets.")
        table1_title_index = markdown.index(f"**{table1_title}**")
        table1_grid_index = markdown.index("| Dataset | #Instances | #Features | #Classes |")
        table2_reference = markdown.index("The experiments use")
        right_column_body_after_table1 = markdown.index("held out for training")
        table2_title_index = markdown.index(f"**{table2_title}**")
        table3_reference = markdown.index("The proposed RLR is computationally")
        table3_title_index = markdown.index(f"**{table3_title}**")

        self.assertGreater(table1_title_index, table1_reference)
        self.assertLess(table1_title_index, table1_grid_index)
        self.assertGreater(table1_title_index, table2_reference)
        self.assertLess(table1_title_index, right_column_body_after_table1)
        self.assertGreater(table2_title_index, table2_reference)
        self.assertGreater(table3_title_index, table3_reference)

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

    def test_table_notes_are_owned_by_adjacent_tables_instead_of_body_text(self) -> None:
        table2 = next(table for table in self.result["table_asts"] if table.get("page") == 6)
        table3 = next(table for table in self.result["table_asts"] if table.get("page") == 8)
        table4 = next(table for table in self.result["table_asts"] if table.get("page") == 9)

        self.assertIn("The boldfaced values are the highest ones", table2.get("content_text", ""))
        self.assertIn("The boldfaced values are the best ones.", table3.get("content_text", ""))
        self.assertIn("In the table", table4.get("content_text", ""))

        for table in (table2, table3, table4):
            note_segments = [
                segment for segment in table.get("content_segments", []) or []
                if segment.get("role") == "note"
            ]
            self.assertTrue(note_segments, msg=table.get("table_id"))
            for segment in note_segments:
                self.assertEqual(segment.get("source_type"), "text_block")
                self.assertEqual(segment.get("relation"), "below")
                self.assertIn("source_block_id", segment)

        body_note_units = [
            unit for unit in self.result.get("content_units", [])
            if unit.get("source_type") == "text"
            and (
                "The boldfaced values are the highest ones" in str(unit.get("text", ""))
                or "The boldfaced values are the best ones." in str(unit.get("text", ""))
                or "In the table" in str(unit.get("text", ""))
            )
        ]
        self.assertEqual(body_note_units, [])

    def test_markdown_attaches_table_notes_and_figure_captions_to_float_blocks(self) -> None:
        markdown = _build_full_markdown([self.result])

        table2_grid = markdown.index("| Dataset | RLR | RFS | MSVM-RFE |")
        table2_note = markdown.index("The boldfaced values are the highest ones")
        self.assertGreater(table2_note, table2_grid)
        self.assertLess(table2_note - table2_grid, 3600)

        table3_grid = markdown.index("| Dataset | RLR | RFS | MSVM-RFE |", table2_grid + 1)
        table3_note = markdown.index("The boldfaced values are the best ones.")
        fig2_image = markdown.index("Fig. 2. Sensibility of the regularization parameter")
        self.assertGreater(table3_note, table3_grid)
        self.assertLess(table3_note, fig2_image)

        table4_grid = markdown.index("| Dataset | RLR vs. RFS | RLR vs. MSVM-RFE |")
        table4_note = markdown.index("In the table (+) implies")
        next_reference = markdown.index("Bol", table4_note)
        self.assertGreater(table4_note, table4_grid)
        self.assertLess(table4_note - table4_grid, 2400)
        self.assertLess(table4_note, next_reference)

        for caption in (
            "Fig. 1. Relationship between feature dimension",
            "Fig. 2. Sensibility of the regularization parameter",
            "Fig. 3. Sensibility of the kernel width",
        ):
            self.assertEqual(markdown.count(caption), 1)

    def test_wide_tables_and_figures_stay_before_later_back_matter_sections(self) -> None:
        markdown = _build_full_markdown([self.result])

        table3_index = markdown.index("**Table 3 CPU time per run")
        fig2_index = markdown.index("Fig. 2. Sensibility of the regularization parameter")
        conclusion_index = markdown.index("5. Conclusion")
        acknowledgments_index = markdown.index("Acknowledgments")
        fig3_index = markdown.index("Fig. 3. Sensibility of the kernel width")
        table4_index = markdown.index("**Table 4 Statistical")
        references_index = markdown.index("References")

        self.assertLess(table3_index, conclusion_index)
        self.assertLess(fig2_index, conclusion_index)
        self.assertLess(fig3_index, acknowledgments_index)
        self.assertLess(table4_index, references_index)

    def test_markdown_normalizes_math_symbol_font_artifacts_outside_tables(self) -> None:
        markdown = _build_full_markdown([self.result])
        self.assertIn(r"1. For $r=1$ to $g$ do", markdown)
        self.assertIn(r"k=0 and $\theta=0.01$", markdown)
        self.assertNotIn("1. For r=1 to g do", markdown)
        self.assertNotIn("k=0 and θ=0.01", markdown)
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

    def test_page4_equations_11_and_13_absorb_symbol_only_fragments_without_prose_cue_rows(self) -> None:
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
        self.assertGreater(float(equation13["bbox"][1]), 700.0)
        self.assertLess(float(equation13["bbox"][2]), 300.0)
        self.assertLess(float(equation13["bbox"][3]), 730.0)
        self.assertNotIn("where", equation13_text.lower())
        self.assertNotIn("defined by", equation13_text.lower())
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
        self.assertTrue(
            any("de" in str(unit.get("text", "")).lower() and "ned by:" in str(unit.get("text", "")).lower() for unit in page4_text_units),
            msg="The prose cue introducing equation (13) should remain body text, not formula content",
        )

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
