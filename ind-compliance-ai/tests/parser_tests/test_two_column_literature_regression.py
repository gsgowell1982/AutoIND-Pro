from __future__ import annotations

import os
from pathlib import Path
import re
import unittest

from api.main import _build_full_markdown, _build_workbench
from parsers.pdf_parser import parse_pdf


def _resolve_two_column_regression_pdf() -> Path:
    override = os.environ.get("IND_TWO_COLUMN_REGRESSION_PDF", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "2-column-tst.pdf"


class TwoColumnLiteratureRegressionTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_path = _resolve_two_column_regression_pdf()
        if not cls.sample_path.exists():
            raise unittest.SkipTest(
                "Two-column regression sample not found. "
                "Set IND_TWO_COLUMN_REGRESSION_PDF or place 2-column-tst.pdf under the workspace root."
            )
        cls.result = parse_pdf(cls.sample_path)
        cls.metadata = cls.result["metadata"]
        cls.page1 = next(page for page in cls.result["pages"] if page["page_number"] == 1)
        cls.page2 = next(page for page in cls.result["pages"] if page["page_number"] == 2)
        cls.page3 = next(page for page in cls.result["pages"] if page["page_number"] == 3)
        cls.page6 = next(page for page in cls.result["pages"] if page["page_number"] == 6)
        cls.page5 = next(page for page in cls.result["pages"] if page["page_number"] == 5)
        cls.page7 = next(page for page in cls.result["pages"] if page["page_number"] == 7)
        cls.page8 = next(page for page in cls.result["pages"] if page["page_number"] == 8)
        cls.page10 = next(page for page in cls.result["pages"] if page["page_number"] == 10)
        cls.page11 = next(page for page in cls.result["pages"] if page["page_number"] == 11)
        cls.page12 = next(page for page in cls.result["pages"] if page["page_number"] == 12)
        cls.workbench = _build_workbench(
            [
                {
                    **cls.result,
                    "file_id": "file_two_column_regression",
                    "filename": cls.sample_path.name,
                }
            ],
            [{"id": "file_two_column_regression", "filename": cls.sample_path.name}],
            [],
            None,
        )
        cls.full_markdown = _build_full_markdown(
            [
                {
                    **cls.result,
                    "file_id": "file_two_column_regression",
                    "filename": cls.sample_path.name,
                    "source_path": str(cls.sample_path),
                }
            ]
        )

    def test_metadata_baseline(self) -> None:
        self.assertEqual(self.metadata["page_count"], 12)
        self.assertEqual(self.metadata["parser_hint"], "pdf-ast-v5")
        self.assertEqual(self.metadata["table_count"], 9)
        self.assertEqual(self.metadata["image_count"], 5)
        self.assertEqual(self.metadata["figure_count"], 5)
        self.assertEqual(self.metadata["toc_count"], 0)
        self.assertEqual(self.metadata["equation_count"], 3)
        self.assertEqual(self.page1.get("image_count"), 0)
        self.assertEqual(self.page10.get("layout_mode"), "mixed")
        self.assertGreater(self.page10.get("layout_confidence", 0.0), 0.5)

    def test_page10_keeps_left_column_before_right_column(self) -> None:
        text = str(self.page10.get("text", ""))
        self.assertIn("Analysis on model efficiency", text)
        self.assertIn("Conclusions", text)
        self.assertIn("The experimental results are shown in Table 9. Due to", text)
        self.assertLess(text.index("Analysis on model efficiency"), text.index("Conclusions"))
        self.assertLess(
            text.index("The experimental results are shown in Table 9. Due to"),
            text.index("Conclusions"),
        )
        self.assertNotIn("evalu-with respect to efficiency", text)
        self.assertNotIn("which in occupation to double", text)

    def test_page10_left_column_table_does_not_absorb_right_column_acknowledgements(self) -> None:
        page10_tables = [table for table in self.result["table_asts"] if table.get("page") == 10]
        self.assertEqual(len(page10_tables), 1)

        table = page10_tables[0]
        self.assertEqual(table["col_count"], 5)
        title = str(table.get("title", ""))
        self.assertTrue("Table 9" in title or "Table\u202f9" in title)
        self.assertLess(float(table.get("bbox", [0.0, 0.0, 0.0, 0.0])[2]), float(self.page10.get("width", 0.0)) * 0.55)
        self.assertEqual([cell["text"] for cell in table["header"]], ["Model", "TT (s)", "IF (ms)", "MO (G)", "F1 (%)"])

        table_text = "\n".join(table.get("raw_row_texts") or []) + "\n" + "\n".join(table.get("data_row_texts") or [])
        self.assertNotIn("Acknowledgements", table_text)
        self.assertNotIn("guidance on medical knowledge", table_text)

        page10_text = str(self.page10.get("text", ""))
        self.assertIn("Acknowledgements", page10_text)
        self.assertIn("We would like to thank the China Academy of Chinese Medical Sciences", page10_text)

    def test_page1_publication_artifacts_do_not_survive_as_figures(self) -> None:
        page1_figures = [figure for figure in self.result["figures"] if figure.get("page") == 1]
        page1_images = [image for image in self.result["image_blocks"] if image.get("page") == 1]
        self.assertEqual(page1_figures, [])
        self.assertEqual(page1_images, [])

    def test_page1_publication_metadata_is_excluded_from_fact_extraction(self) -> None:
        eligible_text = "\n".join(
            str(unit.get("text", ""))
            for unit in self.result["content_units"]
            if unit.get("page") == 1 and unit.get("fact_extraction_eligible")
        )
        self.assertIn("Joint extraction of Chinese medical", eligible_text)
        self.assertIn("Background Most Chinese joint entity and relation extraction tasks", eligible_text)
        self.assertNotIn("bobjzb@163.com", eligible_text)
        self.assertNotIn("snowmanzhao@163.com", eligible_text)
        self.assertNotIn("Open Access", eligible_text)
        self.assertNotIn("BMC Medical Informatics and", eligible_text)
        self.assertNotIn("Decision Making", eligible_text)
        self.assertNotIn("Zhuobin Jiang", eligible_text)
        self.assertNotIn("Yufeng Zhao", eligible_text)
        self.assertNotIn("Keywords Chinese medicine", eligible_text)
        self.assertNotIn("Full list of author information is available at the end of the article", eligible_text)
        self.assertNotIn("https://doi.org/", eligible_text)
        self.assertNotIn("Creative Commons", eligible_text)
        self.assertNotIn("permits use, sharing, adaptation", eligible_text)
        self.assertNotIn("publicdomain/zero/1.0/", eligible_text)
        self.assertIn("ties, overlapping relations, and other challenging extraction issues.", eligible_text)
        self.assertIn("on RoBERTa and single-module global pointer, namely RSGP", eligible_text)

    def test_page2_vector_outline_table_is_recovered_via_ocr(self) -> None:
        page2_tables = [table for table in self.result["table_asts"] if table.get("page") == 2]
        self.assertEqual(len(page2_tables), 1)

        table = page2_tables[0]
        self.assertEqual(table["detection_method"], "vector_ocr")
        self.assertEqual(table["title"], "Table 1 Examples of Normal, SEO and EPO overlapping patterns")
        self.assertEqual(table["col_count"], 3)
        self.assertEqual(table["row_count"], 3)
        self.assertGreater(int(table.get("raw_row_count", 0) or 0), table["row_count"])
        self.assertEqual([cell["text"] for cell in table["header"]], ["Cases", "Texts", "Triples"])

        data_grid = table.get("data_grid") or table.get("grid") or []
        self.assertEqual([row[0] for row in data_grid], ["Normal", "SEO", "EPO"])

        row_text = "\n".join(table.get("row_texts") or [])
        self.assertIn("because mild mood disorders do not require treatment.", row_text)
        self.assertIn("轻度情绪失调不需要治疗。 It is important to distinguish postpartum", row_text)
        self.assertIn("Pancreatic cancer, Examination, Ultrasonography of the upper abdomen)", row_text)
        self.assertIn("(产后抑郁症，诊断，轻度情绪失调) (Postpartum depression, Diagnosis, Mild mood disorders)", row_text)
        self.assertIn("(麻疹，传播途径，呼吸) (Measles, Route of transmission, Respiratory)", row_text)
        self.assertIn("the patient's breathing and coughing.", row_text)
        self.assertNotIn("be- |", row_text)
        self.assertNotIn("Respira- |", row_text)
        self.assertNotIn("\nnull |", row_text)

        semantic_compaction = table.get("semantic_compaction") or {}
        self.assertTrue(semantic_compaction.get("applied"))
        self.assertEqual(semantic_compaction.get("strategy"), "vector_ocr_sparse_anchor_rows")
        self.assertEqual(semantic_compaction.get("compacted_data_row_count"), 3)
        self.assertEqual(self.page2.get("table_count"), 1)

    def test_page2_body_text_repairs_missing_chinese_quote_example(self) -> None:
        text = str(self.page2.get("text", ""))
        self.assertIn("儿童容易得咽喉炎", text)
        self.assertIn("咽喉炎", text)
        self.assertIn("咽喉", text)
        self.assertNotIn("in the sentence “\nChildren are prone", text)

    def test_page3_two_column_body_is_not_misclassified_as_table(self) -> None:
        page3_tables = [table for table in self.result["table_asts"] if table.get("page") == 3]
        self.assertEqual(page3_tables, [])
        self.assertEqual(self.page3.get("table_count"), 0)

    def test_page3_bottom_two_column_lines_do_not_cross_merge_into_full_width_rows(self) -> None:
        text = str(self.page3.get("text", ""))
        self.assertIn("et al. [13] proposed a dependency-driven relation extrac-", text)
        self.assertIn("model for overlapping relation extraction. Subsequently,", text)
        self.assertIn("tion method based on Attentive Graph Convolutional", text)
        self.assertIn("researchers extended the pointer network and proposed", text)
        self.assertNotIn("relation extrac-model for overlapping relation extraction", text)
        self.assertNotIn("Attentive Graph Convolutional researchers extended", text)

    def test_page5_vector_ocr_masking_table_recovers_two_logical_columns(self) -> None:
        page5_tables = [table for table in self.result["table_asts"] if table.get("page") == 5]
        self.assertEqual(len(page5_tables), 1)

        table = page5_tables[0]
        self.assertEqual(table["detection_method"], "vector_ocr")
        self.assertEqual(table["title"], "Table 2 Comparison of masking strategies of BERT and RoBERTa-wwm")
        self.assertEqual(table["col_count"], 2)
        self.assertEqual(self.page5.get("table_count"), 1)

        header = [cell["text"] for cell in table["header"]]
        self.assertEqual(header[0], "Masking Strategy")
        self.assertIn("儿童容易得咽喉炎。", header[1])
        self.assertIn("Childrenare prone to throat infections.", header[1])

        data_grid = table.get("data_grid") or table.get("grid") or []
        self.assertEqual(len(data_grid), 2)
        self.assertEqual([row[0] for row in data_grid], ["BERT", "RoBERTa-wwm"])
        self.assertIn("儿童容易得咽喉[MASK]", data_grid[0][1])
        self.assertIn("pronetothroat", data_grid[0][1])
        self.assertIn("[MASK]", data_grid[0][1])
        self.assertIn("儿童容易得[MASK][MASK][MASK]。", data_grid[1][1])
        self.assertIn("Childrenare prone to", data_grid[1][1])
        self.assertIn("[MASK][MASK].", data_grid[1][1])

        semantic_compaction = table.get("semantic_compaction") or {}
        self.assertTrue(semantic_compaction.get("applied"))
        self.assertEqual(
            semantic_compaction.get("strategy"),
            "vector_ocr_placeholder_tail_column_pairs",
        )
        self.assertEqual(semantic_compaction.get("compacted_data_row_count"), 2)

    def test_page5_display_equations_are_promoted_as_structural_equation_blocks(self) -> None:
        self.assertEqual(self.page5.get("equation_count"), 2)

        equations = [equation for equation in self.result.get("equation_blocks", []) if equation.get("page") == 5]
        self.assertEqual(len(equations), 2)
        self.assertIn("fr(h, t) = rT(h ⋆t) (1)", [equation.get("text") for equation in equations])
        self.assertIn("h ⋆t = ReLU W[h; t]T + b (2)", [equation.get("text") for equation in equations])

        page5_equation_blocks = [
            block
            for block in self.result["document_ast"]["pages"][4]["blocks"]
            if block.get("block_type") == "equation"
        ]
        self.assertEqual(len(page5_equation_blocks), 2)

        equation_units = [
            unit
            for unit in self.result["content_units"]
            if unit.get("page") == 5 and unit.get("semantic_role") == "display_equation"
        ]
        self.assertEqual(len(equation_units), 2)
        self.assertTrue(all(unit.get("unit_role") == "equation" for unit in equation_units))
        self.assertTrue(all(not unit.get("fact_extraction_eligible") for unit in equation_units))

    def test_page5_display_equations_have_validated_latex_projection(self) -> None:
        equations = [equation for equation in self.result.get("equation_blocks", []) if equation.get("page") == 5]
        by_number = {str(equation.get("formula_number")): equation for equation in equations}

        formula1 = str(by_number["1"].get("latex_text") or "")
        formula2 = str(by_number["2"].get("latex_text") or "")

        self.assertIn(r"f_r(h,t)=r^T(h\star t)", formula1)
        self.assertIn(r"\tag{1}", formula1)
        self.assertGreaterEqual(float(by_number["1"].get("latex_confidence") or 0.0), 0.85)

        self.assertIn(r"h\star t=\operatorname{ReLU}", formula2)
        self.assertIn(r"W[h;t]^T+b", formula2)
        self.assertIn(r"\tag{2}", formula2)
        self.assertGreaterEqual(float(by_number["2"].get("latex_confidence") or 0.0), 0.85)

    def test_page5_inline_math_stays_in_text_blocks_instead_of_becoming_equation_blocks(self) -> None:
        page5_text = str(self.page5.get("text", ""))
        self.assertIn("sentence S = {w1, w2, ..., wL} and a set of relations", page5_text)
        self.assertIn("R = {r1, r2, ..., rK}", page5_text)
        self.assertIn("triples T = {(hi, ri, ti)}", page5_text)

        page5_text_blocks = [
            str(block.get("text", ""))
            for block in self.result["document_ast"]["pages"][4]["blocks"]
            if block.get("block_type") == "text"
        ]
        page5_text_blob = "\n".join(page5_text_blocks)
        self.assertIn("sentence S = {w1, w2, ..., wL} and a set of relations", page5_text_blob)
        self.assertIn("R = {r1, r2, ..., rK}", page5_text_blob)
        self.assertIn("triples T = {(hi, ri, ti)}", page5_text_blob)

        page5_equation_texts = [
            str(equation.get("text", ""))
            for equation in self.result.get("equation_blocks", [])
            if equation.get("page") == 5
        ]
        self.assertTrue(all("sentence S =" not in text for text in page5_equation_texts))
        self.assertTrue(all("R = {r1, r2, ..., rK}" not in text for text in page5_equation_texts))
        self.assertTrue(all("triples T = {(hi, ri, ti)}" not in text for text in page5_equation_texts))

    def test_page6_formula_line_is_promoted_as_single_equation_block(self) -> None:
        self.assertEqual(self.page6.get("equation_count"), 1)

        equations = [equation for equation in self.result.get("equation_blocks", []) if equation.get("page") == 6]
        self.assertEqual(len(equations), 1)
        equation_text = str(equations[0].get("text", ""))
        self.assertIn("v(wi, rk, wj)K", equation_text)
        self.assertIn("drop", equation_text)
        self.assertIn("(3)", equation_text)
        self.assertFalse(equation_text.endswith(("“", "”", '"', "'")))

        page6_text_blocks = [
            block.get("text")
            for block in self.result["document_ast"]["pages"][5]["blocks"]
            if block.get("block_type") == "text"
        ]
        self.assertNotIn("v(wi, rk, wj)K", page6_text_blocks)
        self.assertNotIn("(3)", page6_text_blocks)

    def test_page6_tensor_score_equation_latex_keeps_subscripts_superscripts_and_parentheses(self) -> None:
        equation = next(equation for equation in self.result.get("equation_blocks", []) if equation.get("page") == 6)
        latex = str(equation.get("latex_text") or "")

        self.assertIn(r"\{v(w_i,r_k,w_j)\}_{k=1}^{K}", latex)
        self.assertIn(r"R^T", latex)
        self.assertIn(r"\operatorname{ReLU}", latex)
        self.assertIn(r"\operatorname{drop}", latex)
        self.assertIn(r"W[e_i;e_j]^T+b", latex)
        self.assertIn(r"\tag{3}", latex)
        self.assertNotIn("RTReLU drop", latex)
        self.assertNotIn("“", latex)
        self.assertGreaterEqual(float(equation.get("latex_confidence") or 0.0), 0.85)

    def test_page6_where_clauses_project_matrix_dimension_membership_as_inline_latex(self) -> None:
        markdown = self.full_markdown

        self.assertIn(r"$W \in R^{d_e \times 2d}$ and b are trainable weight and bias", markdown)
        self.assertIn(r"$R \in R^{d_e \times 4k}$ is a trainable weight", markdown)
        self.assertNotIn("W ∈Rde×2d and b are trainable", markdown)
        self.assertNotIn("R ∈Rde×4k is a trainable", markdown)

    def test_page6_where_clause_projects_indexed_tuple_range_and_drops_quote_residue(self) -> None:
        markdown = self.full_markdown

        self.assertIn(r"score of $\{(w_i,r_k,w_j)\}_{k=1}^{K}$ for the token pair", markdown)
        self.assertIn(r"drop(·) is a dropout strategy used to prevent over-fitting", markdown)
        self.assertNotIn("score of (wi, rk, wj)K", markdown)
        self.assertNotIn("k=1 for the token pair", markdown)
        self.assertNotIn("\n“\n", markdown)

    def test_page5_same_block_indexed_set_formula_projects_range_tail(self) -> None:
        markdown = self.full_markdown

        self.assertIn(r"triples $T=\{(h_i,r_i,t_i)\}_{i=1}^{n}$, where L denotes", markdown)
        self.assertNotIn(r"$T=\{(h_i,r_i,t_i)\}$$n_i$=1", markdown)
        self.assertNotIn(r"$T=\{(h_i,r_i,t_i)\}$n i=1", markdown)

    def test_page6_tensor_dimension_chain_projects_as_inline_latex(self) -> None:
        markdown = self.full_markdown

        self.assertIn(r"third-order tensor $M^{L \times K \times L}$.", markdown)
        self.assertNotIn("third-order tensor ML×K×L.", markdown)
        self.assertNotIn("third-order tensor MLKL.", markdown)

    def test_page6_body_paragraph_drops_intra_line_duplicate_ocr_residue(self) -> None:
        markdown = self.full_markdown

        self.assertNotIn(
            "resulting in the subject entity WI where etag 2 islocated, resulting in the subject entity",
            markdown,
        )
        self.assertNotIn("object is joined from the column where tag 1 is 3, the object is joined", markdown)
        self.assertNotIn("located to the column where tag 3 is located, resulting in located to the column", markdown)

    def test_page6_figure2_caption_does_not_absorb_preceding_body_sentence(self) -> None:
        fig2_image = next(
            image
            for image in self.result.get("image_blocks", [])
            if "Fig. 2 Single-module global pointer decoding" in str(image.get("content_text", ""))
        )
        caption_text = " ".join(
            str(segment.get("text") or "")
            for segment in fig2_image.get("caption_blocks", []) or []
            if isinstance(segment, dict)
        )

        self.assertIn("Fig. 2 Single-module global pointer decoding", caption_text)
        self.assertNotIn("As shown in Fig. 2", caption_text)
        self.assertNotIn("As shown in Fig. 2, given a sentence, we use a sin- Fig. 2", self.full_markdown)
        self.assertIn(
            "Fig. 2 Single-module global pointer decoding: a the normal case, b the special case",
            self.full_markdown,
        )

    def test_page9_figure4_caption_does_not_absorb_following_body_reference(self) -> None:
        fig4_caption = "Fig. 4 F1-score (%) of extracting triples from sentences with different number"
        fig4_image = next(
            image
            for image in self.result.get("image_blocks", [])
            if fig4_caption in str(image.get("content_text", ""))
        )
        caption_text = " ".join(
            str(segment.get("text") or "")
            for segment in fig4_image.get("caption_blocks", []) or []
            if isinstance(segment, dict)
        )

        self.assertIn(fig4_caption, caption_text)
        self.assertNotIn("Figure 5 shows the F1 of the top three relations among", caption_text)
        self.assertNotIn("Figure 5 shows the F1 of the top three relations among", str(fig4_image.get("caption_text", "")))
        self.assertNotIn("Figure 5 shows the F1 of the top three relations among", str(fig4_image.get("title", "")))
        self.assertNotIn(
            "Figure 5 shows the F1 of the top three relations among Fig. 4 F1-score",
            self.full_markdown,
        )
        self.assertIn(
            "Fig. 4 F1-score (%) of extracting triples from sentences with different number (denotes as N) of triples",
            self.full_markdown,
        )

    def test_page9_chart_axes_are_not_promoted_to_embedded_ocr_table(self) -> None:
        page9_embedded_tables = [
            table
            for table in self.result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 9
            and str(table.get("detection_source") or table.get("detection_method") or "") == "embedded_image_ocr"
        ]

        self.assertEqual(
            page9_embedded_tables,
            [],
            msg="figure chart axes and data labels must remain figure evidence, not embedded OCR table ASTs",
        )

    def test_page6_multilingual_quote_rows_recover_missing_chinese(self) -> None:
        text = str(self.page6.get("text", ""))
        self.assertIn("咽喉炎,发病部位咽喉", text)
        self.assertIn("胃疼 stomachache", text)

        page6_ocr_blocks = [
            block
            for block in self.result["document_ast"]["pages"][5]["blocks"]
            if block.get("source") == "body-ocr-repair"
        ]
        self.assertTrue(any("咽喉炎,发病部位咽喉" in str(block.get("text", "")) for block in page6_ocr_blocks))
        self.assertTrue(any("胃疼 stomachache" in str(block.get("text", "")) for block in page6_ocr_blocks))
        self.assertTrue(any("咽喉炎 throat infections" in str(block.get("text", "")) for block in page6_ocr_blocks))
        self.assertFalse(any("因喉炎 throat infections" in str(block.get("text", "")) for block in page6_ocr_blocks))
        self.assertFalse(any("因候炎 throat infections" in str(block.get("text", "")) for block in page6_ocr_blocks))
        self.assertNotIn("因喉炎 throat infections", self.full_markdown)
        self.assertNotIn("因候炎 throat infections", self.full_markdown)

    def test_page5_page6_body_ocr_repairs_do_not_straddle_the_column_gutter(self) -> None:
        for page_number in (5, 6):
            page = next(item for item in self.result["pages"] if item["page_number"] == page_number)
            column_mid = float(page.get("width", 0.0) or 0.0) / 2.0
            gutter_guard = 4.0
            ocr_blocks = [
                block
                for block in self.result["document_ast"]["pages"][page_number - 1]["blocks"]
                if block.get("source") == "body-ocr-repair"
            ]
            self.assertGreater(len(ocr_blocks), 0)
            for block in ocr_blocks:
                x0, _, x1, _ = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
                self.assertFalse(
                    float(x0) < column_mid - gutter_guard and float(x1) > column_mid + gutter_guard,
                    msg=f"page {page_number} OCR block straddles gutter: {block}",
                )

    def test_page7_four_tables_are_recovered_without_false_toc_promotion(self) -> None:
        page7_tables = [table for table in self.result["table_asts"] if table.get("page") == 7]
        self.assertEqual(self.page7.get("table_count"), 4)
        self.assertEqual(self.page7.get("toc_count"), 0)
        self.assertEqual(len(page7_tables), 4)

        page7_toc_blocks = [
            block
            for block in self.result["document_ast"]["pages"][6]["blocks"]
            if block.get("block_type") == "toc"
        ]
        self.assertEqual(page7_toc_blocks, [])

        titles = [str(table.get("title", "")) for table in page7_tables]
        self.assertTrue(any("Table 3" in title or "Table\u202f3" in title for title in titles))
        self.assertTrue(any("Table 4" in title or "Table\u202f4" in title for title in titles))
        self.assertTrue(any("Table 5" in title or "Table\u202f5" in title for title in titles))
        self.assertTrue(any("Table 6" in title or "Table\u202f6" in title for title in titles))

        table3 = next(
            table
            for table in page7_tables
            if "Table 3" in str(table.get("title", "")) or "Table\u202f3" in str(table.get("title", ""))
        )
        table4 = next(
            table
            for table in page7_tables
            if "Table 4" in str(table.get("title", "")) or "Table\u202f4" in str(table.get("title", ""))
        )
        table5 = next(
            table
            for table in page7_tables
            if "Table 5" in str(table.get("title", "")) or "Table\u202f5" in str(table.get("title", ""))
        )

        self.assertEqual([cell["text"] for cell in table3["header"]], ["Category", "Train", "Validation", "Test"])
        self.assertEqual([cell["text"] for cell in table4["header"]], ["Category", "Train", "Validation", "Test"])
        self.assertEqual([cell["text"] for cell in table5["header"]], ["Category", "Train", "Validation", "Test"])
        self.assertEqual(
            table3.get("data_grid") or table3.get("grid") or [],
            [
                ["Relations", "44", "44", "44"],
                ["Sentences", "17924", "4482", "5602"],
                ["Triples", "54286", "13484", "17512"],
            ],
        )
        self.assertEqual(
            table3.get("row_texts") or [],
            [
                "Relations | 44 | 44 | 44",
                "Sentences | 17924 | 4482 | 5602",
                "Triples | 54286 | 13484 | 17512",
            ],
        )
        self.assertEqual(
            [row[0] for row in (table5.get("data_grid") or table5.get("grid") or [])],
            ["1", "2", "3", "4", "\u22655"],
        )
        self.assertTrue(
            all(
                "Table 5" not in row_text and "Table\u202f5" not in row_text
                for row_text in (table5.get("row_texts") or [])
            )
        )
        self.assertTrue(
            str((table5.get("raw_row_texts") or [""])[0]).startswith("Table 5")
            or str((table5.get("raw_row_texts") or [""])[0]).startswith("Table\u202f5")
        )

        page7_text = str(self.page7.get("text", ""))
        self.assertIn("Experiments and discussion", page7_text)
        self.assertIn("Comparison models", page7_text)

    def test_page7_table_content_evidence_keeps_titles_as_metadata_not_body_text(self) -> None:
        page7_table_evidence = [
            item
            for item in self.result.get("content_evidence", [])
            if item.get("source_type") == "table" and item.get("page") == 7
        ]
        self.assertEqual(len(page7_table_evidence), 4)

        table5_evidence = next(item for item in page7_table_evidence if item.get("source_id") == "tbl_003")
        self.assertEqual(table5_evidence.get("title"), "Table\u202f5 Statistics of different triples in a sentence")
        self.assertEqual(table5_evidence.get("segments", [{}])[0].get("role"), "title")
        self.assertEqual(
            table5_evidence.get("segments", [{}, {}])[1].get("text"),
            "Category | Train | Validation | Test",
        )
        self.assertTrue(str(table5_evidence.get("content_text", "")).startswith("Category | Train | Validation | Test"))
        self.assertNotIn("Table\u202f5 Statistics of different triples in a sentence", str(table5_evidence.get("content_text", "")))

        table6_evidence = next(item for item in page7_table_evidence if item.get("source_id") == "tbl_005")
        self.assertEqual(table6_evidence.get("segments", [{}])[0].get("role"), "title")
        self.assertEqual(table6_evidence.get("segments", [{}, {}])[1].get("text"), "Model | Prec. | Rec. | F1")
        self.assertTrue(str(table6_evidence.get("content_text", "")).startswith("Model | Prec. | Rec. | F1"))
        self.assertNotIn("Table\u202f6 Precision", str(table6_evidence.get("content_text", "")))

    def test_page8_multiline_header_continuation_stays_in_header_not_data(self) -> None:
        page8_tables = [table for table in self.result["table_asts"] if table.get("page") == 8]
        self.assertEqual(len(page8_tables), 2)

        table8 = next(
            table
            for table in page8_tables
            if "Table 8" in str(table.get("title", "")) or "Table\u202f8" in str(table.get("title", ""))
        )

        self.assertEqual(
            [cell["text"] for cell in table8["header"]],
            ["Pre-trained Language Model", "Prec.", "Rec.", "F1"],
        )
        self.assertEqual(table8.get("data_start_row"), 4)
        self.assertEqual(table8.get("row_count"), 3)
        self.assertEqual(
            table8.get("data_grid") or table8.get("grid") or [],
            [
                ["RoBERTa-wwm", "68.32", "58.62", "63.10"],
                ["BERT-wwm", "67.48", "57.82", "62.28"],
                ["ERNIE", "65.22", "56.18", "60.36"],
            ],
        )
        self.assertEqual(
            table8.get("row_texts") or [],
            [
                "RoBERTa-wwm | 68.32 | 58.62 | 63.10",
                "BERT-wwm | 67.48 | 57.82 | 62.28",
                "ERNIE | 65.22 | 56.18 | 60.36",
            ],
        )
        self.assertEqual((table8.get("raw_row_texts") or [None, None, None, None])[3], "Model | null | null | null")

    def test_full_markdown_does_not_repeat_table6_table7_table8_titles(self) -> None:
        markdown = self.full_markdown

        self.assertEqual(markdown.count("Table 6 Precision"), 1)
        self.assertEqual(markdown.count("Table 7 Precision"), 1)
        self.assertEqual(markdown.count("Table 8 Precision"), 1)
        self.assertNotIn("**Table 6 Precision (%), Recall (%) and F1-score (%) of RSGP and**", markdown)
        self.assertNotIn("**Table 7 Precision (%), Recall (%) and F1-score (%) of RSGP and**", markdown)
        self.assertNotIn("**Table 8 Precision (%), Recall (%) and F1-score (%) of different**", markdown)

    def test_full_markdown_projects_display_and_inline_math_as_latex(self) -> None:
        markdown = self.full_markdown

        self.assertIn(r"f_r(h,t)=r^T(h\star t) \tag{1}", markdown)
        self.assertIn(r"h\star t=\operatorname{ReLU}(W[h;t]^T+b) \tag{2}", markdown)
        self.assertIn(r"\{v(w_i,r_k,w_j)\}_{k=1}^{K}", markdown)
        self.assertIn(r"W \in R^{d_e \times 2d}", markdown)
        self.assertIn(r"R \in R^{d_e \times 4k}", markdown)
        self.assertIn(r"S=\{w_1,w_2,\ldots,w_L\}", markdown)
        self.assertIn(r"R=\{r_1,r_2,\ldots,r_K\}", markdown)
        self.assertIn(r"T=\{(h_i,r_i,t_i)\}", markdown)
        self.assertNotIn("RTReLU drop", markdown)

    def test_workbench_projection_keeps_table_compaction_and_equation_blocks(self) -> None:
        pdf_document = self.workbench["pdf_document"]
        self.assertIsNotNone(pdf_document)
        self.assertEqual(len(pdf_document["table_asts"]), 9)
        self.assertEqual(len(pdf_document["equation_blocks"]), 3)
        self.assertEqual(len(pdf_document.get("toc_sequences", [])), 0)

        table1 = next(table for table in pdf_document["table_asts"] if table["table_id"] == "tbl_001")
        self.assertEqual(table1["page"], 2)
        self.assertEqual(table1["detection_method"], "vector_ocr")
        self.assertEqual(table1["row_count"], 3)
        self.assertEqual(table1["raw_row_count"], 25)
        self.assertEqual(
            (table1.get("semantic_compaction") or {}).get("strategy"),
            "vector_ocr_sparse_anchor_rows",
        )

        table9 = next(table for table in pdf_document["table_asts"] if table["table_id"] == "tbl_009")
        self.assertEqual(table9["page"], 10)
        self.assertEqual(table9["col_count"], 5)
        self.assertTrue("Table 9" in str(table9.get("title", "")) or "Table\u202f9" in str(table9.get("title", "")))

        equation_pages = sorted(equation["page"] for equation in pdf_document["equation_blocks"])
        self.assertEqual(equation_pages, [5, 5, 6])

    def test_page11_page12_references_are_structured_as_reference_entries_not_toc(self) -> None:
        self.assertEqual(self.page11.get("toc_count"), 0)
        self.assertEqual(self.page12.get("toc_count"), 0)

        page11_units = [unit for unit in self.result["content_units"] if unit.get("page") == 11]
        page12_units = [unit for unit in self.result["content_units"] if unit.get("page") == 12]
        reference_units = [
            unit
            for unit in self.result["content_units"]
            if unit.get("page") in {11, 12} and unit.get("semantic_role") in {"reference_heading", "reference_entry"}
        ]

        heading_unit = next(unit for unit in page11_units if unit.get("text") == "References")
        self.assertEqual(heading_unit.get("semantic_role"), "reference_heading")
        self.assertEqual(heading_unit.get("unit_role"), "section_heading")
        self.assertFalse(heading_unit.get("fact_extraction_eligible"))

        page11_entry = next(
            unit
            for unit in page11_units
            if unit.get("semantic_role") == "reference_entry"
            and "Grishman R. Information extraction" in str(unit.get("text", ""))
        )
        self.assertEqual(page11_entry.get("unit_role"), "entry")
        self.assertFalse(page11_entry.get("fact_extraction_eligible"))

        page12_entry = next(
            unit
            for unit in page12_units
            if unit.get("semantic_role") == "reference_entry"
            and "Cui Y, Che W, Liu T, Qin B, Yang Z." in str(unit.get("text", ""))
        )
        self.assertEqual(page12_entry.get("unit_role"), "entry")
        self.assertFalse(page12_entry.get("fact_extraction_eligible"))
        self.assertEqual(page12_entry.get("attributes", {}).get("reference_number"), "29")

        self.assertTrue(all(not unit.get("fact_extraction_eligible") for unit in reference_units))
        self.assertFalse(
            any(
                unit.get("semantic_role") == "author_affiliation"
                and (
                    str(unit.get("text", "")).startswith("14. Sahu SK")
                    or "niques on clinical texts. Appl Sci. 2021;11:8319." in str(unit.get("text", ""))
                    or str(unit.get("text", "")).startswith("29. Cui Y, Che W")
                )
                for unit in self.result["content_units"]
            )
        )

        eligible_reference_text = "\n".join(
            str(unit.get("text", ""))
            for unit in self.result["content_units"]
            if unit.get("page") in {11, 12} and unit.get("fact_extraction_eligible")
        )
        self.assertNotIn("Grishman R. Information extraction", eligible_reference_text)
        self.assertNotIn("Cui Y, Che W, Liu T, Qin B, Yang Z.", eligible_reference_text)

        page12_non_reference_units = [
            unit for unit in page12_units if "Publisher" in str(unit.get("text", "")) or "Springer Nature remains neutral" in str(unit.get("text", ""))
        ]
        self.assertTrue(page12_non_reference_units)
        self.assertTrue(all(unit.get("semantic_role") != "reference_entry" for unit in page12_non_reference_units))

    def test_full_markdown_renders_references_as_separate_numbered_entries(self) -> None:
        markdown = self.full_markdown

        self.assertIn("References", markdown)
        self.assertRegex(markdown, r"1\. Grishman R\. Information extraction IEEE Intell Syst\. 2015;30:8[–-]15\.")
        self.assertIn("2. Li D, Zhang Y, Li D, Lin D. Review of entity relation extraction methods.", markdown)
        self.assertIn("3. Zhou B, Cai X, Zhang Y, Yuan X. MTAAL: multi-task adversarial active", markdown)
        self.assertNotRegex(markdown, r"2015;30:8[–-]15\. 2\. Li")
        self.assertNotRegex(markdown, r"2020;57:1424[–-]48\. 3\. Zhou")

    def test_full_markdown_does_not_promote_reference_page_ranges_to_numbered_entries(self) -> None:
        markdown = self.full_markdown
        references_section = markdown.split("\nReferences\n", 1)[1].split("\nPublisher", 1)[0]
        numbered_entries = re.findall(r"(?m)^(\d{1,4})\.\s+\S", references_section)

        self.assertEqual(numbered_entries, [str(number) for number in range(1, 33)])
        self.assertIn("2020. pp. 1476", references_section)
        self.assertIn("2019. pp. 4171", references_section)
        self.assertNotRegex(references_section, r"(?m)^2020\.\s+pp\.\s+1476")
        self.assertNotRegex(references_section, r"(?m)^2019\.\s+pp\.\s+4171")

    def test_full_markdown_drops_adjacent_duplicate_publication_history_residue(self) -> None:
        markdown = self.full_markdown

        self.assertEqual(markdown.count("Received: 25 November 2022"), 1)
        self.assertIn("Received: 25 November 2022 Accepted: 13 June 2024", markdown)
        self.assertNotIn(
            "Received: 25 November 2022 Accepted: 13 June 2024 Received: 25 November 2022 Accepted: 13",
            markdown,
        )

    def test_document_ast_reference_entries_keep_continuity_metadata(self) -> None:
        reference_blocks = [
            block
            for page in self.result.get("document_ast", {}).get("pages", [])
            for block in page.get("blocks", [])
            if block.get("semantic_role") == "reference_entry"
        ]

        self.assertTrue(reference_blocks)
        entry2_blocks = [block for block in reference_blocks if str(block.get("reference_number") or "") == "2"]
        entry3_blocks = [block for block in reference_blocks if str(block.get("reference_number") or "") == "3"]

        self.assertGreaterEqual(len(entry2_blocks), 1)
        self.assertGreaterEqual(len(entry3_blocks), 1)
        self.assertTrue(all(int(block.get("reference_entry_index") or 0) == 2 for block in entry2_blocks))
        self.assertTrue(all(int(block.get("reference_entry_index") or 0) == 3 for block in entry3_blocks))
        self.assertTrue(entry2_blocks[0].get("reference_entry_start"))
        self.assertFalse(entry2_blocks[0].get("reference_continuation"))


    def test_full_markdown_keeps_reference_continuations_inside_same_numbered_entry(self) -> None:
        markdown = self.full_markdown

        self.assertRegex(
            markdown,
            r"2\. Li D, Zhang Y, Li D, Lin D\. Review of entity relation extraction methods\. J Comput Res Dev\. 2020;57:1424[–-]48\.",
        )
        self.assertIn("MTAAL: multi-task adversarial active learning for medical named entity recognition", markdown)
        self.assertNotIn("methods. J\n\nComput Res Dev.", markdown)
        self.assertNotIn("active learn-\n\ning for medical", markdown)

    def test_literature_section_headings_render_as_standalone_markdown_blocks(self) -> None:
        markdown = self.full_markdown

        self.assertIn("\nIntroduction\n\nInformation extraction is a natural language processing", markdown)
        self.assertIn("\nRelated work\n\nTraditional pipeline methods\n\nIn the traditional pipeline methods", markdown)
        self.assertNotIn("Introduction Information extraction is a natural language processing", markdown)
        self.assertNotIn("Related work Traditional pipeline methods In the traditional pipeline methods", markdown)


if __name__ == "__main__":
    unittest.main()
