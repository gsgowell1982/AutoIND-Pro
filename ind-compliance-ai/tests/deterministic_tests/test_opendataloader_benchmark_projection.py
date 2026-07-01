from __future__ import annotations

import importlib
import re
import unittest


class OpenDataLoaderBenchmarkProjectionTests(unittest.TestCase):
    def test_benchmark_projection_strips_product_wrapper_and_base64_images(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        markdown = projection.strip_autoind_product_wrapper(
            "\n".join(
                [
                    "## sample.pdf 解析结果",
                    "",
                    "- Estimated pages: 1",
                    "- Parser strategy: pdf",
                    "",
                    "### Table of Contents",
                    "1. Introduction 1",
                    "",
                    "### Document Body",
                    "",
                    "Visible body text.",
                    "",
                    "![Figure 1](data:image/png;base64,AAAA)",
                    "",
                ]
            ),
            "sample.pdf",
        )

        self.assertEqual(markdown, "Visible body text.\n")

    def test_benchmark_projection_rejects_bottom_footnote_line_as_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "bottom-footnote-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "author_line",
                                "unit_role": "metadata",
                                "text": "25 Wiliam Beckford, An Arabian Tale, from an Unpub-",
                                "bbox": [56, 620, 268, 633],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "S. Hattox, Coffee and Coffeehouses: The Origins of a So-",
                                "bbox": [79, 668, 268, 681],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "25 Wiliam Beckford, An Arabian Tale, from an Unpub-\n\n"
            "S. Hattox, Coffee and Coffeehouses: The Origins of a So-\n",
            document,
        )

        self.assertNotIn("# 25 Wiliam Beckford", markdown)
        self.assertNotIn("# S. Hattox", markdown)
        self.assertIn("25 Wiliam Beckford", markdown)
        self.assertIn("S. Hattox", markdown)

    def test_benchmark_projection_preserves_bottom_numbered_chapter_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "bottom-numbered-chapter-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "page_number",
                                "unit_role": "metadata",
                                "text": "314",
                                "bbox": [300, 31, 318, 44],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Earlier body text establishes the full page vertical span.",
                                "bbox": [62, 86, 350, 102],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "of interest.",
                                "bbox": [62, 462, 108, 478],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "7 Variants of sj Observer Models",
                                "bbox": [62, 503, 230, 519],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "In this chapter, I have presented two variants of a latency-based observer model.",
                                "bbox": [62, 530, 388, 546],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "author_affiliation",
                                "unit_role": "metadata",
                                "text": "18 E.g., <SimultaneityNoisyCriteriaMultistart 225-386>. Note that Matlab has inbuilt func-",
                                "bbox": [62, 587, 388, 600],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "of interest.\n\n"
            "7 Variants of sj Observer Models\n\n"
            "In this chapter, I have presented two variants of a latency-based observer model.\n\n"
            "18 E.g., <SimultaneityNoisyCriteriaMultistart 225-386>. Note that Matlab has inbuilt func-\n",
            document,
        )

        self.assertIn("# 7 Variants of sj Observer Models", markdown)
        self.assertNotIn("# 18 E.g.", markdown)

    def test_benchmark_projection_demotes_explicit_numbered_procedure_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "procedure-steps.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "Procedure:",
                                "bbox": [72, 90, 140, 105],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1. Record the Question that is being investigated in this experiment.",
                                "bbox": [90, 128, 420, 142],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "2. Record a Hypothesis for the question stated above.",
                                "bbox": [90, 146, 370, 160],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3. Predict the results of the experiment based on your hypothesis.",
                                "bbox": [90, 164, 420, 178],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Procedure:\n\n"
            "1. Record the Question that is being investigated in this experiment.\n\n"
            "2. Record a Hypothesis for the question stated above.\n\n"
            "3. Predict the results of the experiment based on your hypothesis.\n",
            document,
        )

        self.assertIn("# Procedure:", markdown)
        self.assertNotIn("### 1. Record the Question", markdown)
        self.assertNotIn("### 2. Record a Hypothesis", markdown)
        self.assertIn("1. Record the Question", markdown)

    def test_benchmark_projection_demotes_existing_numbered_step_heading_with_body_continuation(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "procedure-continuation.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Restriction Enzyme Digest Prep:",
                                "bbox": [72, 90, 250, 105],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "20. Use a micropipette to add 10 uL of tris-EDTA solution (TE) to each tube.",
                                "bbox": [72, 118, 500, 132],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Dissolve the pellets by pipetting in and out.",
                                "bbox": [72, 136, 380, 150],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Restriction Enzyme Digest Prep:\n\n"
            "### 20. Use a micropipette to add 10 uL of tris-EDTA solution (TE) to each tube.\n\n"
            "Dissolve the pellets by pipetting in and out.\n",
            document,
        )

        self.assertNotIn("### 20. Use a micropipette", markdown)
        self.assertIn("20. Use a micropipette", markdown)
        self.assertIn("Dissolve the pellets", markdown)

    def test_benchmark_projection_demotes_existing_numbered_source_note_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "source-note-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Marketed fruit values are shown above.",
                                "bbox": [72, 120, 330, 136],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1. Statistics Canada. Table 32-10-0364-01 Area, production and farm gate",
                                "bbox": [72, 612, 520, 626],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "value of marketed fruits, by crop type and province.",
                                "bbox": [92, 628, 438, 642],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Marketed fruit values are shown above.\n\n"
            "### 1. Statistics Canada. Table 32-10-0364-01 Area, production and farm gate\n\n"
            "value of marketed fruits, by crop type and province.\n",
            document,
        )

        self.assertNotIn("### 1. Statistics Canada", markdown)
        self.assertIn("1. Statistics Canada", markdown)
        self.assertIn("value of marketed fruits", markdown)

    def test_benchmark_projection_demotes_existing_numbered_bibliography_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "bibliography-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The findings were presented in the cited report.",
                                "bbox": [56, 581, 382, 592],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1. Jimes, C., Karaglani, A., Petrides, L., Rios, J., Sebesta, J., & Torre, K. (2019). Open Educational Resources (OER) in Texas Higher Education,",
                                "bbox": [48, 716, 555, 724],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "author_line",
                                "unit_role": "metadata",
                                "text": "2019. Austin, TX: Digital Higher Education Consortium of Texas and Texas Higher Education Coordinating Board; Half Moon Bay,",
                                "bbox": [57, 728, 553, 736],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "CA: Institute for the Study of Knowledge Management in Education.",
                                "bbox": [57, 739, 318, 748],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "The findings were presented in the cited report.\n\n"
            "### 1. Jimes, C., Karaglani, A., Petrides, L., Rios, J., Sebesta, J., & Torre, K. (2019). Open Educational Resources (OER) in Texas Higher Education,\n\n"
            "2019. Austin, TX: Digital Higher Education Consortium of Texas and Texas Higher Education Coordinating Board; Half Moon Bay,\n\n"
            "CA: Institute for the Study of Knowledge Management in Education.\n",
            document,
        )

        self.assertNotIn("### 1. Jimes", markdown)
        self.assertIn("1. Jimes", markdown)
        self.assertIn("Digital Higher Education Consortium", markdown)

    def test_benchmark_projection_demotes_existing_decimal_cross_reference_continuation_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "decimal-cross-reference-continuation.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The data in Figure",
                                "bbox": [72, 220, 180, 234],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3.1.1 and Table 3.1.1 do not reflect those MSMEs who",
                                "bbox": [72, 236, 426, 250],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "were permanently closed during the survey period.",
                                "bbox": [72, 252, 386, 266],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "The data in Figure\n\n"
            "##### 3.1.1 and Table 3.1.1 do not reflect those MSMEs who\n\n"
            "were permanently closed during the survey period.\n",
            document,
        )

        self.assertNotIn("##### 3.1.1 and Table", markdown)
        self.assertIn("3.1.1 and Table 3.1.1", markdown)
        self.assertIn("were permanently closed", markdown)

    def test_benchmark_projection_preserves_real_numbered_section_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "numbered-section.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Earlier paragraph ends here.",
                                "bbox": [72, 180, 240, 194],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "2 Depth Up-Scaling",
                                "bbox": [72, 230, 190, 244],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "To efficiently scale-up LLMs, we use pretrained weights.",
                                "bbox": [72, 265, 390, 279],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Earlier paragraph ends here.\n\n"
            "2 Depth Up-Scaling\n\n"
            "To efficiently scale-up LLMs, we use pretrained weights.\n",
            document,
        )

        self.assertIn("# 2 Depth Up-Scaling", markdown)

    def test_benchmark_projection_rejects_page_number_with_remote_running_header(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "remote-running-header.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "74",
                                "bbox": [56, 31, 66, 47],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Baird",
                                "bbox": [465, 32, 491, 47],
                            },
                            {
                                "block_type": "image",
                                "bbox": [56, 56, 330, 432],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Body text follows below the page image.",
                                "bbox": [56, 448, 268, 464],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "74\n\nBaird\n\nBody text follows below the page image.\n",
            document,
        )

        self.assertNotIn("# 74 Baird", markdown)

    def test_benchmark_projection_rejects_post_media_lowercase_body_line_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "post-media-body-line.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "image",
                                "bbox": [56, 56, 330, 432],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "this list, Richard Walker, apothecary to the Prince",
                                "bbox": [56, 448, 268, 464],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "of Wales, adds Arabic henna, manna, and rhubarb.",
                                "bbox": [56, 462, 268, 478],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "this list, Richard Walker, apothecary to the Prince\n\n"
            "of Wales, adds Arabic henna, manna, and rhubarb.\n",
            document,
        )

        self.assertNotIn("# this list", markdown)

    def test_benchmark_projection_marks_ast_section_heading_without_changing_body_text(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_1",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Opening paragraph.",
                                "bbox": [72, 90, 420, 104],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_2",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1 Background",
                                "bbox": [72, 132, 220, 148],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_3",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The body continues.",
                                "bbox": [72, 160, 420, 174],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("Opening paragraph.", markdown)
        self.assertIn("\n# 1 Background\n", markdown)
        self.assertIn("The body continues.", markdown)
        self.assertNotIn("### Document Body", markdown)

    def test_benchmark_projection_preserves_remainder_after_heading_prefix(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "heading-prefix-remainder.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Overview of OCR Pack",
                                "bbox": [34, 26, 153, 39],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Base Model Performance Evaluation of Upstage OCR Pack",
                                "bbox": [34, 47, 514, 68],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Upstage universal OCR model E2E performance",
                                "bbox": [37, 119, 288, 133],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "evaluation1",
                                "bbox": [37, 141, 94, 155],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Overview of OCR Pack\n\n"
            "Base Model Performance Evaluation of Upstage OCR Pack Upstage universal OCR model E2E performance evaluation1\n",
            document,
        )

        self.assertIn("# Overview of OCR Pack", markdown)
        self.assertIn("# Base Model Performance Evaluation of Upstage OCR Pack", markdown)
        self.assertIn("# Upstage universal OCR model E2E performance evaluation1", markdown)
        self.assertNotIn(
            "# Base Model Performance Evaluation of Upstage OCR Pack Upstage universal OCR model E2E performance evaluation1",
            markdown,
        )

    def test_benchmark_projection_rejects_top_band_paragraph_opening_after_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "top-band-paragraph-after-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 600,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Tycho Brahe's Observatory",
                                "bbox": [56, 58, 210, 73],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Three years after the publication of Copernicus' De Revolutionibus,",
                                "bbox": [56, 96, 339, 107],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Tycho Brahe was born to a family of Danish nobility. He developed",
                                "bbox": [56, 110, 339, 121],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "an early interest in astronomy and, as a young man, made significant",
                                "bbox": [56, 124, 339, 135],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Tycho Brahe's Observatory\n\n"
            "Three years after the publication of Copernicus' De Revolutionibus,\n\n"
            "Tycho Brahe was born to a family of Danish nobility. He developed\n\n"
            "an early interest in astronomy and, as a young man, made significant\n",
            document,
        )

        self.assertIn("# Tycho Brahe's Observatory", markdown)
        self.assertNotIn("# Three years", markdown)
        self.assertNotIn("# Tycho Brahe was born", markdown)

    def test_benchmark_projection_splits_gap_backed_numbered_heading_from_merged_paragraph(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "gap-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_1",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The paragraph ends here.",
                                "bbox": [72, 90, 420, 104],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_2",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "7 Variants of Observer Models",
                                "bbox": [72, 150, 260, 166],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_3",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "In this chapter, models are compared.",
                                "bbox": [72, 182, 430, 196],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "The paragraph ends here. 7 Variants of Observer Models In this chapter, models are compared.\n",
            document,
        )

        self.assertIn("The paragraph ends here.\n\n# 7 Variants of Observer Models\n\nIn this chapter", markdown)

    def test_benchmark_projection_preserves_semantic_html_table(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                                "display_grid": [["Group", "Group"], ["A", "B"], ["1", "2"]],
                                "semantic_grid": [["Group", "Group"], ["A", "B"], ["1", "2"]],
                                "logical_cells": [
                                    {
                                        "row": 0,
                                        "col": 0,
                                        "rowspan": 1,
                                        "colspan": 2,
                                        "text": "Group",
                                        "source": "sparse_header_colspan_projection",
                                    }
                                ],
                                "table_family": "rowspan_grouped_table",
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "display_grid": [["Group", "Group"], ["A", "B"], ["1", "2"]],
                    "semantic_grid": [["Group", "Group"], ["A", "B"], ["1", "2"]],
                    "logical_cells": [
                        {
                            "row": 0,
                            "col": 0,
                            "rowspan": 1,
                            "colspan": 2,
                            "text": "Group",
                            "source": "sparse_header_colspan_projection",
                        }
                    ],
                    "table_family": "rowspan_grouped_table",
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("<table>", markdown)
        self.assertIn('colspan="2"', markdown)
        self.assertIn("<td>1</td>", markdown)

    def test_benchmark_projection_does_not_duplicate_header_when_data_grid_still_contains_header(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "table-header-duplication.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "title": "Table 1. Typed evidence",
                    "header_row_index": 0,
                    "data_start_row": 1,
                    "header": [
                        {"col": 1, "text": "Study"},
                        {"col": 2, "text": "Dose"},
                        {"col": 3, "text": "Result"},
                    ],
                    "display_grid": [
                        ["Study", "Dose", "Result"],
                        ["Rat", "10 mg/kg", "Observed"],
                        ["Dog", "30 mg/kg", "Not observed"],
                    ],
                    "data_grid": [
                        ["Study", "Dose", "Result"],
                        ["Rat", "10 mg/kg", "Observed"],
                        ["Dog", "30 mg/kg", "Not observed"],
                    ],
                    "semantic_projection_v2": {
                        "version": 2,
                        "table_family": "comparison_matrix",
                        "source": "table_semantic_projection_v2",
                    },
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertEqual(markdown.count("| Study | Dose | Result |"), 1)
        self.assertIn("| Rat | 10 mg/kg | Observed |", markdown)
        self.assertIn("| Dog | 30 mg/kg | Not observed |", markdown)

    def test_benchmark_projection_keeps_first_body_row_when_display_grid_starts_with_data(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "table-display-starts-with-data.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "header_row_index": 0,
                    "data_start_row": 1,
                    "header": [
                        {"col": 1, "text": "Region"},
                        {"col": 2, "text": "Low dose"},
                        {"col": 3, "text": "High dose"},
                    ],
                    "display_grid": [
                        ["North", "4", "2"],
                        ["South", "2", "5"],
                    ],
                    "data_grid": [
                        ["North", "4", "2"],
                        ["South", "2", "5"],
                    ],
                    "semantic_projection_v2": {
                        "version": 2,
                        "table_family": "comparison_matrix",
                        "source": "table_semantic_projection_v2",
                    },
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("| Region | Low dose | High dose |", markdown)
        self.assertIn("| North | 4 | 2 |", markdown)
        self.assertIn("| South | 2 | 5 |", markdown)

    def test_benchmark_projection_renders_display_grid_header_spans_without_logical_cells(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "display-grid-header-spans.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "header_row_index": 0,
                    "data_start_row": 2,
                    "header": [
                        {"col": 1, "text": "Species"},
                        {"col": 2, "text": "Low"},
                        {"col": 3, "text": "High"},
                        {"col": 4, "text": "Finding"},
                    ],
                    "display_grid": [
                        ["Species", "Dose group", None, "Finding"],
                        [None, "Low", "High", None],
                        ["Rat", "10", "30", "None"],
                    ],
                    "data_grid": [["Rat", "10", "30", "None"]],
                    "logical_cells": [],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("<table>", markdown)
        self.assertIn('rowspan="2">Species</th>', markdown)
        self.assertIn('colspan="2">Dose group</th>', markdown)
        self.assertIn('rowspan="2">Finding</th>', markdown)
        self.assertIn("<td>Rat</td>", markdown)

    def test_benchmark_projection_preserves_display_header_spans_with_semantic_body_repairs(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "display-header-span-semantic-body.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "title": "Table 1. Multilevel exposure summary",
                    "detection_source": "pymupdf_builtin",
                    "table_family": "comparison_matrix",
                    "header_row_index": 0,
                    "data_start_row": 2,
                    "header": [
                        {"col": 1, "text": "Species"},
                        {"col": 2, "text": "Exposure"},
                        {"col": 3, "text": "Column 3"},
                        {"col": 4, "text": "Reference"},
                    ],
                    "display_grid": [
                        ["Species", "Exposure", None, "Reference"],
                        [None, "Cmax", "AUC", None],
                        ["Rat 12 34 Study A", None, None, None],
                    ],
                    "data_grid": [["Rat 12 34 Study A", None, None, None]],
                    "semantic_grid": [
                        ["Species", "Exposure", "Column 3", "Reference"],
                        [None, "Cmax", "AUC", None],
                        ["Rat", "12", "34", "Study A"],
                    ],
                    "semantic_projection_v2": {
                        "version": 2,
                        "table_family": "comparison_matrix",
                        "source": "table_semantic_projection_v2",
                        "compact_single_cell_body_row_projection": {
                            "source": "compact_single_cell_body_row_projection",
                            "projected_row_count": 1,
                        },
                    },
                    "logical_cells": [],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("<table>", markdown)
        self.assertIn('rowspan="2">Species</th>', markdown)
        self.assertIn('colspan="2">Exposure</th>', markdown)
        self.assertIn('rowspan="2">Reference</th>', markdown)
        self.assertIn("<td>Rat</td>", markdown)
        self.assertIn("<td>12</td>", markdown)
        self.assertIn("<td>34</td>", markdown)
        self.assertIn("<td>Study A</td>", markdown)
        self.assertNotIn("<th>Column 3</th>", markdown)

    def test_benchmark_projection_display_header_spans_do_not_cross_parent_rowspan_slots(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "display-header-span-boundary.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "detection_source": "pymupdf_builtin",
                    "table_family": "comparison_matrix",
                    "header_row_index": 0,
                    "data_start_row": 2,
                    "header": [
                        {"col": 1, "text": "Label"},
                        {"col": 2, "text": "Group"},
                        {"col": 3, "text": "Column 3"},
                        {"col": 4, "text": "Total"},
                    ],
                    "display_grid": [
                        ["Label", "Group", None, "Total"],
                        [None, "A", "B", None],
                        ["Rat", "1", "2", "3"],
                    ],
                    "semantic_grid": [
                        ["Label", "Group", "Column 3", "Total"],
                        [None, "A", "B", None],
                        ["Rat", "1", "2", "3"],
                    ],
                    "semantic_projection_v2": {
                        "version": 2,
                        "table_family": "comparison_matrix",
                        "source": "table_semantic_projection_v2",
                        "compact_single_cell_body_row_projection": {
                            "source": "compact_single_cell_body_row_projection",
                            "projected_row_count": 1,
                        },
                    },
                    "logical_cells": [],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn('rowspan="2">Label</th>', markdown)
        self.assertIn('colspan="2">Group</th>', markdown)
        self.assertIn('rowspan="2">Total</th>', markdown)
        self.assertIn("<th>B</th>", markdown)
        self.assertNotIn('colspan="2">B</th>', markdown)

    def test_benchmark_projection_renders_two_column_section_divider_as_colspan(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "two-column-section-divider.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "header_row_index": 0,
                    "data_start_row": 1,
                    "header": [
                        {"col": 1, "text": "Field"},
                        {"col": 2, "text": "Value"},
                    ],
                    "display_grid": [
                        ["Field", "Value"],
                        ["Competence Area", "#1 THE 3 RS"],
                        ["Learning Outcomes", None],
                        ["Knowledge", "Understand recycling"],
                    ],
                    "semantic_grid": [
                        ["Competence Area", "#1 THE 3 RS"],
                        ["Learning Outcomes", None],
                        ["Knowledge", "Understand recycling"],
                    ],
                    "logical_cells": [
                        {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "Competence Area", "source": "semantic_grid"},
                        {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "text": "#1 THE 3 RS", "source": "semantic_grid"},
                        {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "text": "Learning Outcomes", "source": "semantic_grid"},
                        {"row": 2, "col": 0, "rowspan": 1, "colspan": 1, "text": "Knowledge", "source": "semantic_grid"},
                        {"row": 2, "col": 1, "rowspan": 1, "colspan": 1, "text": "Understand recycling", "source": "semantic_grid"},
                    ],
                    "semantic_projection_v2": {
                        "version": 2,
                        "table_family": "two_column_inventory",
                        "source": "table_semantic_projection_v2",
                        "label_after_value_pair_projection": {
                            "source": "two_column_label_after_value_pair_projection",
                            "projected_pair_count": 2,
                        },
                    },
                    "table_family": "two_column_inventory",
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("<table>", markdown)
        self.assertIn('<td colspan="2">Learning Outcomes</td>', markdown)
        self.assertIn("<td>Knowledge</td>", markdown)
        self.assertIn("<td>Understand recycling</td>", markdown)

    def test_benchmark_projection_preserves_keyed_long_list_without_synthetic_header(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "keyed-long-list.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 1000,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "header_row_index": 0,
                    "data_start_row": 1,
                    "header": [{"col": 1, "text": "Filtered Task Name"}],
                    "display_grid": [
                        ["Filtered Task Name"],
                        ["task228_arc_answer_generation_easy"],
                        ["ai2_arcARCChallenge:1.0.0"],
                        ["ai2_arcARCEasy:1.0.0"],
                    ],
                    "semantic_grid": [
                        ["Field", "Value"],
                        [
                            "Filtered Task Name",
                            "task228_arc_answer_generation_easy ai2_arcARCChallenge:1.0.0 ai2_arcARCEasy:1.0.0",
                        ],
                    ],
                    "semantic_projection_v2": {
                        "version": 2,
                        "table_family": "keyed_long_list",
                        "source": "table_semantic_projection_v2",
                        "single_column_keyed_list_projection": {
                            "source": "single_column_keyed_long_list_projection",
                            "source_row_count": 4,
                            "value_count": 3,
                        },
                    },
                    "table_family": "keyed_long_list",
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("<table>", markdown)
        self.assertIn("<td>Filtered Task Name</td>", markdown)
        self.assertIn("task228_arc_answer_generation_easy", markdown)
        self.assertNotIn("<th>Field</th>", markdown)
        self.assertNotIn("| Field | Value |", markdown)

    def test_benchmark_projection_renders_toc_sequence_entries(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "toc.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {"block_type": "toc", "block_id": "toc_1", "bbox": [50, 100, 400, 500]},
                        ],
                    }
                ]
            },
            "toc_sequences": [
                {
                    "title": "Table of contents",
                    "pages": [1],
                    "entries": [
                        {"text": "Introduction", "page_locator": "7", "page": 1},
                        {"text": "1. Changing Practices", "page_locator": "12", "page": 1},
                    ],
                }
            ],
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Table of contents", markdown)
        self.assertIn("Introduction 7", markdown)
        self.assertIn("1. Changing Practices 12", markdown)

    def test_benchmark_projection_demotes_existing_toc_entry_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "contents-page.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "toc_title",
                                "unit_role": "metadata",
                                "text": "Contents",
                                "bbox": [72, 90, 160, 110],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1. Overview of OCR Pack",
                                "bbox": [72, 130, 260, 144],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "2. Introduction of Product Services and Key Features",
                                "bbox": [72, 154, 430, 168],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3. Product - Detail Specification",
                                "bbox": [72, 178, 330, 192],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "# Contents\n\n"
            "### 1. Overview of OCR Pack\n\n"
            "### 2. Introduction of Product Services and Key Features\n\n"
            "### 3. Product - Detail Specification\n",
            document,
        )

        self.assertIn("# Contents", markdown)
        self.assertIn("1. Overview of OCR Pack", markdown)
        self.assertNotIn("### 1. Overview of OCR Pack", markdown)
        self.assertNotIn("### 2. Introduction", markdown)

    def test_benchmark_projection_keeps_page_header_before_inserted_toc_sequence(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "toc-header.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 792,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_header",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "MOHAVE COMMUNITY COLLEGE BIO181",
                                "bbox": [48, 17, 578, 39],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_toc_title",
                                "semantic_role": "toc_title",
                                "unit_role": "metadata",
                                "text": "Table of Contents",
                                "bbox": [34, 73, 155, 89],
                            },
                            {
                                "block_type": "toc",
                                "block_id": "toc_1",
                                "toc_sequence_id": "tocseq_001",
                                "bbox": [34, 89, 581, 754],
                            },
                        ],
                    }
                ]
            },
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq_001",
                    "title": "Table of Contents",
                    "pages": [1],
                    "bbox": [34, 89, 581, 754],
                    "entries": [
                        {"text": "Measurement Lab worksheet", "page_locator": "3", "page": 1},
                        {"text": "Scientific Method Lab", "page_locator": "6", "page": 1},
                    ],
                }
            ],
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertLess(markdown.index("MOHAVE COMMUNITY COLLEGE BIO181"), markdown.index("# Table of Contents"))
        self.assertEqual(markdown.count("# Table of Contents"), 1)
        self.assertIn("Measurement Lab worksheet 3", markdown)

    def test_benchmark_projection_renders_toc_like_table_as_plain_entries(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "toc-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 612,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                                "title": "Table of Contents",
                                "bbox": [68, 106, 369, 193],
                                "display_grid": [
                                    ["Executive", "Summary", "4"],
                                    ["Legal", "Framework", "6"],
                                    ["Election", "Administration", "11"],
                                ],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_2",
                                "table_id": "tbl_2",
                                "title": "Table of Contents",
                                "bbox": [68, 173, 369, 282],
                                "display_grid": [
                                    ["Political Parties, Candidates Registration and Election", "18"],
                                    ["Campaign", None],
                                    ["Media Freedom and Access to Information", "25"],
                                ],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "title": "Table of Contents",
                    "display_grid": [
                        ["Executive", "Summary", "4"],
                        ["Legal", "Framework", "6"],
                        ["Election", "Administration", "11"],
                    ],
                },
                {
                    "table_id": "tbl_2",
                    "title": "Table of Contents",
                    "display_grid": [
                        ["Political Parties, Candidates Registration and Election", "18"],
                        ["Campaign", None],
                        ["Media Freedom and Access to Information", "25"],
                    ],
                },
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Table of Contents", markdown)
        self.assertIn("Executive Summary 4", markdown)
        self.assertIn("Political Parties, Candidates Registration and Election 18", markdown)
        self.assertNotIn("| Executive | Summary | 4 |", markdown)
        self.assertEqual(markdown.count("# Table of Contents"), 1)

    def test_benchmark_projection_renders_contents_table_as_plain_entries(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        markdown = "\n".join(
            [
                "# CONTENTS",
                "",
                "| Experiment #4: Energy Loss in Pipes | 33 |",
                "| --- | --- |",
                "| Experiment #5: Impact of a Jet | 43 |",
                "| References | 101 |",
                "",
                "Image Credits",
            ]
        )

        projected = projection._project_toc_like_tables_as_plain_entries(markdown)

        self.assertIn("# CONTENTS", projected)
        self.assertIn("Experiment #4: Energy Loss in Pipes 33", projected)
        self.assertIn("Experiment #5: Impact of a Jet 43", projected)
        self.assertIn("References 101", projected)
        self.assertNotIn("| Experiment #4: Energy Loss in Pipes | 33 |", projected)

    def test_benchmark_projection_reconstructs_seedless_contents_page(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "contents-page.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {"block_type": "text", "text": "CONTENTS", "bbox": [85, 115, 157, 129]},
                            {"block_type": "text", "text": "About the Publisher", "bbox": [57, 206, 155, 219]},
                            {"block_type": "text", "text": "About This Project", "bbox": [57, 224, 148, 237]},
                            {"block_type": "text", "text": "LAB MANUAL", "bbox": [57, 269, 114, 280]},
                            {"block_type": "text", "text": "Experiment #1: Hydrostatic Pressure", "bbox": [57, 296, 237, 310]},
                            {"block_type": "text", "text": "vii", "bbox": [544, 206, 557, 219]},
                            {"block_type": "text", "text": "ix", "bbox": [547, 224, 557, 237]},
                            {"block_type": "text", "text": "3", "bbox": [549, 296, 557, 310]},
                            {"block_type": "table", "table_id": "tbl_1", "bbox": [57, 374, 555, 591]},
                            {"block_type": "text", "text": "Image Credits", "bbox": [57, 596, 126, 609]},
                            {"block_type": "text", "text": "104", "bbox": [543, 596, 557, 609]},
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "bbox": [57, 374, 555, 591],
                    "display_grid": [
                        ["Experiment #4: Energy Loss in Pipes", "33"],
                        ["Experiment #5: Impact of a Jet", "43"],
                        ["References", "101"],
                        ["Links by Chapter", "102"],
                    ],
                }
            ],
            "toc_sequences": [],
            "toc_blocks": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# CONTENTS", markdown)
        self.assertIn("About the Publisher vii", markdown)
        self.assertIn("About This Project ix", markdown)
        self.assertIn("LAB MANUAL", markdown)
        self.assertIn("Experiment #1: Hydrostatic Pressure 3", markdown)
        self.assertIn("Experiment #4: Energy Loss in Pipes 33", markdown)
        self.assertIn("Image Credits 104", markdown)
        self.assertNotIn("| Experiment #4: Energy Loss in Pipes | 33 |", markdown)

    def test_benchmark_projection_reconstructs_seedless_contents_group_labels_and_suppresses_sources(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "mixed-contents-page.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {"block_type": "text", "text": "Contents", "bbox": [56, 59, 149, 83]},
                            {"block_type": "text", "text": "Acknowledgment of Country", "bbox": [85, 146, 207, 158]},
                            {"block_type": "text", "text": "Introduction", "bbox": [85, 220, 140, 232]},
                            {"block_type": "text", "text": "Part I. Chapter One - Exploring Your Data", "bbox": [85, 252, 268, 263]},
                            {"block_type": "text", "text": "Section 1.1: Data and Types of Statistical Variables", "bbox": [85, 280, 293, 292]},
                            {
                                "block_type": "text",
                                "text": "Part II. Chapter Two - Test Statistics, p Values, Confidence Intervals and Effect Sizes",
                                "bbox": [85, 412, 446, 423],
                            },
                            {"block_type": "text", "text": "v", "bbox": [534, 146, 541, 158]},
                            {"block_type": "text", "text": "1", "bbox": [535, 220, 541, 232]},
                            {"block_type": "text", "text": "3", "bbox": [533, 280, 541, 292]},
                            {"block_type": "table", "table_id": "tbl_1", "bbox": [85, 424, 538, 694]},
                            {
                                "block_type": "text",
                                "text": "Part IV. Chapter Four - Comparing Associations Between Two Variables",
                                "bbox": [85, 696, 430, 707],
                            },
                            {"block_type": "text", "text": "Section 4.1: Examining Relationships", "bbox": [85, 711, 238, 723]},
                            {"block_type": "text", "text": "Section 4.2: Correlation Assumptions, Interpretation, and Write Up", "bbox": [85, 728, 366, 740]},
                            {"block_type": "text", "text": "29", "bbox": [530, 711, 541, 723]},
                            {"block_type": "text", "text": "31", "bbox": [530, 728, 541, 740]},
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "bbox": [85, 424, 538, 694],
                    "display_grid": [
                        ["Section 2.1: p Values", "12"],
                        ["Section 2.2: Significance", "13"],
                        ["Part III. Chapter Three - Comparing Two Group Means", None],
                        ["Section 3.1: Looking at Group Differences", "20"],
                    ],
                }
            ],
            "toc_sequences": [],
            "toc_blocks": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Contents", markdown)
        self.assertIn("Acknowledgment of Country v", markdown)
        self.assertIn("Introduction 1", markdown)
        self.assertIn("Part I. Chapter One - Exploring Your Data\n", markdown)
        self.assertIn("Section 1.1: Data and Types of Statistical Variables 3", markdown)
        self.assertIn(
            "Part II. Chapter Two - Test Statistics, p Values, Confidence Intervals and Effect Sizes\n",
            markdown,
        )
        self.assertIn("Part III. Chapter Three - Comparing Two Group Means\n", markdown)
        self.assertIn("Section 3.1: Looking at Group Differences 20", markdown)
        self.assertIn("Part IV. Chapter Four - Comparing Associations Between Two Variables\n", markdown)
        self.assertIn("Section 4.1: Examining Relationships 29", markdown)
        self.assertIn("Section 4.2: Correlation Assumptions, Interpretation, and Write Up 31", markdown)
        self.assertNotIn("| Section 2.1: p Values | 12 |", markdown)
        self.assertNotRegex(markdown, r"(?m)^1 3$")

    def test_benchmark_projection_reconstructs_titleless_contents_continuation_page(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "contents-continuation.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "text": "Part V. Chapter Five - Comparing Associations Between Multiple Variables",
                                "bbox": [85, 59, 406, 70],
                            },
                            {"block_type": "table", "table_id": "tbl_1", "bbox": [85, 88, 538, 166]},
                            {
                                "block_type": "text",
                                "text": "Part VI. Chapter Six - Comparing Three or More Group Means",
                                "bbox": [85, 186, 356, 197],
                            },
                            {"block_type": "table", "table_id": "tbl_2", "bbox": [85, 198, 538, 308]},
                            {"block_type": "table", "table_id": "tbl_3", "bbox": [85, 296, 538, 595]},
                            {
                                "block_type": "text",
                                "text": "Part IX. Chapter Nine - Nonparametric Statistics",
                                "bbox": [85, 598, 330, 610],
                            },
                            {"block_type": "table", "table_id": "tbl_4", "bbox": [85, 612, 538, 707]},
                            {"block_type": "text", "text": "References", "bbox": [85, 723, 134, 735]},
                            {"block_type": "text", "text": "101", "bbox": [530, 723, 541, 735]},
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "bbox": [85, 88, 538, 166],
                    "display_grid": [
                        ["Section 5.1: The Linear Model", "35"],
                        ["Section 5.2: Simple Regression Assumptions, Interpretation, and Write Up", "36"],
                    ],
                },
                {
                    "table_id": "tbl_2",
                    "bbox": [85, 198, 538, 308],
                    "display_grid": [
                        ["Section 6.1: Between Versus Within Group Analyses", "49"],
                        ["Section 6.2: One-Way ANOVA Assumptions, Interpretation, and Write Up", "51"],
                    ],
                },
                {
                    "table_id": "tbl_3",
                    "bbox": [85, 296, 538, 595],
                    "display_grid": [
                        ["Part VII. Chapter Seven - Moderation and Mediation", "Analyses"],
                        ["Section 7.1: Mediation and Moderation Models", "64"],
                        ["Part VIII. Chapter Eight - Factor Analysis and Scale", "Reliability"],
                        ["Section 8.1: Factor Analysis Definitions", "75"],
                    ],
                },
                {
                    "table_id": "tbl_4",
                    "bbox": [85, 612, 538, 707],
                    "display_grid": [
                        ["Section 9.1: Nonparametric Definitions", "91"],
                        ["Section 9.2: Choosing Appropriate Tests", "93"],
                    ],
                },
            ],
            "toc_sequences": [],
            "toc_blocks": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("Part V. Chapter Five - Comparing Associations Between Multiple Variables\n", markdown)
        self.assertIn("Section 5.1: The Linear Model 35", markdown)
        self.assertIn("Part VI. Chapter Six - Comparing Three or More Group Means\n", markdown)
        self.assertIn("Section 6.2: One-Way ANOVA Assumptions, Interpretation, and Write Up 51", markdown)
        self.assertIn("Part VII. Chapter Seven - Moderation and Mediation Analyses\n", markdown)
        self.assertIn("Part VIII. Chapter Eight - Factor Analysis and Scale Reliability\n", markdown)
        self.assertIn("Part IX. Chapter Nine - Nonparametric Statistics\n", markdown)
        self.assertIn("References 101", markdown)
        self.assertNotIn("| Section 5.1: The Linear Model | 35 |", markdown)
        self.assertNotIn("# References", markdown)

    def test_benchmark_projection_promotes_page_top_chapter_title_to_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "chapter.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_1",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Chapter 2",
                                "bbox": [190, 92, 253, 107],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_2",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Narratives in Chuj",
                                "bbox": [173, 130, 270, 148],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_3",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Body text follows.",
                                "bbox": [66, 187, 380, 200],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Chapter 2", markdown)
        self.assertIn("# Narratives in Chuj", markdown)
        self.assertIn("Body text follows.", markdown)

    def test_benchmark_projection_promotes_isolated_question_heading_before_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "question-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 760,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 1. Caption text.",
                                "bbox": [70, 416, 531, 429],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "What tool(s) do you typically use in your course?",
                                "bbox": [70, 461, 387, 480],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Ask whether the instructor utilizes your institution's course management system.",
                                "bbox": [70, 494, 543, 507],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "What supporting materials do you utilize for this course?",
                                "bbox": [70, 556, 442, 575],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "If the instructor relies on self-grading homework platforms.",
                                "bbox": [70, 590, 543, 603],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# What tool(s) do you typically use in your course?", markdown)
        self.assertIn("# What supporting materials do you utilize for this course?", markdown)

    def test_benchmark_projection_promotes_centered_figure_title_with_period(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "figure-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 612,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Humanity's Home Base.",
                                "bbox": [131, 58, 265, 73],
                            },
                            {
                                "block_type": "image",
                                "bbox": [56, 95, 339, 236],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Earth and Moon, Drawn to Scale.",
                                "bbox": [105, 468, 291, 483],
                            },
                            {
                                "block_type": "image",
                                "bbox": [56, 505, 339, 532],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Humanity's Home Base.", markdown)
        self.assertIn("# Earth and Moon, Drawn to Scale.", markdown)

    def test_benchmark_projection_promotes_multiline_media_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "multiline-media-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The telescope can observe infrared radiation from space.",
                                "bbox": [56.69, 324.24, 313.64, 335.15],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Observations from the Spitzer Space Telescope",
                                "bbox": [66.32, 369.03, 329.65, 383.45],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "(SST).",
                                "bbox": [180.49, 386.33, 215.45, 400.75],
                            },
                            {
                                "block_type": "image",
                                "bbox": [65.0, 405.0, 335.0, 515.0],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Observations from the Spitzer Space Telescope (SST).", markdown)

    def test_benchmark_projection_prefers_title_after_chapter_label(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "chapter-label.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "chap_label",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Chapter 2",
                                "bbox": [190.74, 92.11, 253.26, 107.13],
                            },
                            {
                                "block_type": "text",
                                "block_id": "chapter_title",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Narratives in Chuj",
                                "bbox": [173.01, 130.78, 270.96, 148.3],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "This collection of narratives demonstrates the broad variety of stories.",
                                "bbox": [98.5, 187.48, 380.1, 200.63],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Introduction to the Texts",
                                "bbox": [162.47, 313.65, 281.55, 329.3],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Two of the stories are ultimately of foreign origin.",
                                "bbox": [66.0, 335.98, 380.09, 349.13],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("# Chapter 2", markdown)
        self.assertIn("# Narratives in Chuj", markdown)
        self.assertIn("# Introduction to the Texts", markdown)

    def test_benchmark_projection_merges_ast_heading_continuation_line(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "wrapped-ast-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3. Perspective of supply and demand balance of wood pellets and cost",
                                "bbox": [85.1, 87.74, 484.46, 100.7],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "structure in Japan",
                                "bbox": [113.42, 105.98, 212.86, 118.94],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "According to a survey, biomass power generation is domestically produced.",
                                "bbox": [85.1, 129.74, 484.4, 140.78],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn(
            "# 3. Perspective of supply and demand balance of wood pellets and cost structure in Japan",
            markdown,
        )
        self.assertNotIn("# structure in Japan", markdown)

    def test_benchmark_projection_rejects_wrapped_chart_metric_heading_from_ast_role(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "chart-metric-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "94.1 4",
                                "bbox": [603.31, 189.71, 614.62, 197.33],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "5",
                                "bbox": [609.0, 198.0, 614.0, 205.0],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "OCR-Precision",
                                "bbox": [387.0, 216.0, 430.0, 224.0],
                            },
                            {
                                "block_type": "image",
                                "bbox": [423.38, 165.75, 625.88, 318.75],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings("94.1 4 5 OCR-Precision\n", document)

        self.assertNotIn("# 94.1 4 5", markdown)

    def test_benchmark_projection_keeps_chapter_label_wrapped_with_true_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "chapter-label-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Chapter 3",
                                "bbox": [90.0, 171.83, 181.21, 192.5],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Numerical differentiation",
                                "bbox": [90.0, 218.98, 381.99, 243.77],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3.1 Introduction",
                                "bbox": [90.0, 284.57, 205.21, 298.92],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Chapter 3 Numerical differentiation", markdown)

    def test_benchmark_projection_promotes_multiline_top_title_without_overusing_card_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "card-headings.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 612,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Acme aims to enrich your business by providing",
                                "bbox": [70, 48, 520, 75],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Easy-to-Apply AI solutions",
                                "bbox": [70, 79, 302, 106],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Our Purpose",
                                "bbox": [70, 165, 165, 185],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Our Mission",
                                "bbox": [250, 165, 342, 185],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "What We Do",
                                "bbox": [430, 165, 525, 185],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "How It Helps",
                                "bbox": [610, 165, 720, 185],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Making AI Beneficial",
                                "bbox": [70, 205, 218, 221],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Easy-to-apply AI, Everywhere",
                                "bbox": [250, 205, 430, 221],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Providing useful AI solutions for everyone.",
                                "bbox": [430, 205, 660, 221],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Reducing manual review time.",
                                "bbox": [610, 205, 780, 221],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Acme aims to enrich your business by providing Easy-to-Apply AI solutions Our Purpose\n\n"
            "Our Mission\n\nMaking AI Beneficial Easy-to-apply AI, Everywhere\n\n"
            "What We Do Providing useful AI solutions for everyone.\n\n"
            "How It Helps Reducing manual review time.\n",
            document,
        )

        self.assertIn("# Acme aims to enrich your business by providing Easy-to-Apply AI solutions", markdown)
        self.assertNotIn("# Our Purpose", markdown)
        self.assertNotIn("# Our Mission", markdown)
        self.assertNotIn("# What We Do", markdown)
        self.assertNotIn("# How It Helps", markdown)

    def test_benchmark_projection_promotes_internal_spanning_table_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "internal-table-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                                "title_row_index": 0,
                                "display_grid": [
                                    ["Key Functions by Main Service Flow", None, None],
                                    ["Stage", "Function", "Benefit"],
                                    ["Creation", "Project setup", "Faster work"],
                                ],
                                "semantic_grid": [
                                    ["Stage", "Function", "Benefit"],
                                    ["Creation", "Project setup", "Faster work"],
                                ],
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "title_row_index": 0,
                    "display_grid": [
                        ["Key Functions by Main Service Flow", None, None],
                        ["Stage", "Function", "Benefit"],
                        ["Creation", "Project setup", "Faster work"],
                    ],
                    "semantic_grid": [
                        ["Stage", "Function", "Benefit"],
                        ["Creation", "Project setup", "Faster work"],
                    ],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Key Functions by Main Service Flow", markdown)
        self.assertIn("| Stage | Function | Benefit |", markdown)

    def test_benchmark_projection_rejects_top_label_when_main_title_follows(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "title-label.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 612,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Semantic Search Pack: Value",
                                "bbox": [70, 34, 230, 48],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "SS Pack allows businesses to access further data more rapidly",
                                "bbox": [70, 65, 618, 92],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The pack can reduce information acquisition time.",
                                "bbox": [70, 124, 540, 140],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Semantic Search Pack: Value\n\n"
            "SS Pack allows businesses to access further data more rapidly The pack can reduce information acquisition time.\n",
            document,
        )

        self.assertNotIn("# Semantic Search Pack: Value", markdown)
        self.assertIn("# SS Pack allows businesses to access further data more rapidly", markdown)

    def test_benchmark_projection_rejects_chart_axis_like_false_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "chart-labels.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 612,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "0.03 0.06",
                                "bbox": [420, 310, 478, 323],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Current Service",
                                "bbox": [410, 330, 510, 344],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Compared to",
                                "bbox": [411, 350, 492, 364],
                            },
                            {
                                "block_type": "image",
                                "bbox": [330, 260, 610, 490],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "0.03 0.06\n\nCurrent Service\n\nCompared to\n\n![Figure 1](#img_p1_001)\n",
            document,
        )

        self.assertNotIn("# 0.03 0.06", markdown)
        self.assertNotIn("# Current Service", markdown)
        self.assertNotIn("# Compared to", markdown)

    def test_benchmark_projection_does_not_promote_wrapped_body_before_figure(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "figure-body.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3.1. Status of Business Operations",
                                "bbox": [94, 182, 249, 194],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "As shown in Figure 3.1.1, the number of MSMEs",
                                "bbox": [94, 206, 323, 218],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "working as usual gradually increased over the",
                                "bbox": [94, 218, 323, 230],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 3.1.1: Status of operations during each survey phase (%)",
                                "bbox": [94, 243, 378, 254],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "3.1. Status of Business Operations\n\n"
            "As shown in Figure 3.1.1, the number of MSMEs working as usual gradually increased over the\n\n"
            "Figure 3.1.1: Status of operations during each survey phase (%)\n",
            document,
        )

        self.assertIn("# 3.1. Status of Business Operations", markdown)
        self.assertNotIn("# As shown in Figure", markdown)

    def test_benchmark_projection_rejects_isolated_body_sentence_as_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "body-sentence.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Ablation Studies",
                                "bbox": [70, 180, 210, 198],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "We present ablation studies for both the instruction and alignment tuning stages.",
                                "bbox": [70, 220, 540, 236],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Ablation on the training datasets follows in the next paragraph.",
                                "bbox": [70, 254, 510, 270],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Ablation Studies\n\n"
            "We present ablation studies for both the instruction and alignment tuning stages.\n\n"
            "Ablation on the training datasets follows in the next paragraph.\n",
            document,
        )

        self.assertIn("# Ablation Studies", markdown)
        self.assertNotIn("# We present ablation studies", markdown)

    def test_benchmark_projection_does_not_treat_sentence_initial_article_as_lettered_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "article-body.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Promotional Materials",
                                "bbox": [70, 55, 238, 77],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "A good promotional strategy should include multiple facets, from physical materials to digital",
                                "bbox": [70, 93, 543, 106],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "communications.",
                                "bbox": [70, 109, 160, 122],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Promotional Materials\n\n"
            "A good promotional strategy should include multiple facets, from physical materials to digital communications.\n",
            document,
        )

        self.assertIn("# Promotional Materials", markdown)
        self.assertNotIn("# A good promotional strategy", markdown)

    def test_benchmark_projection_does_not_treat_lowercase_fragment_as_lettered_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "fragment.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "s in British",
                                "bbox": [290, 219, 335, 229],
                            }
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("s in British", markdown)
        self.assertNotIn("# s in British", markdown)

    def test_benchmark_projection_keeps_short_section_heading_near_body_text(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "short-section.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Annual Events",
                                "bbox": [70, 520, 167, 539],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Creating promotional materials and graphics can make your OER program recognizable.",
                                "bbox": [70, 553, 543, 566],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Annual Events Creating promotional materials and graphics can make your OER program recognizable.\n",
            document,
        )

        self.assertIn("# Annual Events", markdown)

    def test_benchmark_projection_promotes_colon_label_before_grouped_body_items(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "colon-group.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "As a reviewer:",
                                "bbox": [13, 75, 90, 87],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Check source documents beforehand",
                                "bbox": [30, 89, 218, 101],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Confirm all listed appendices",
                                "bbox": [30, 102, 188, 114],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Avoid changing the source scope",
                                "bbox": [30, 116, 224, 128],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "As a reviewer: Check source documents beforehand Confirm all listed appendices Avoid changing the source scope\n",
            document,
        )

        self.assertIn("# As a reviewer:", markdown)
        self.assertIn("Check source documents beforehand", markdown)

    def test_benchmark_projection_promotes_colon_label_before_single_marked_body_item(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "colon-single-marked-item.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "and e-game levels (PR3)",
                                "bbox": [85, 118, 195, 129],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Reference frameworks:",
                                "bbox": [85, 142, 214, 155],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "⮚ GreenComp - The European Sustainability Competence Framework responds to",
                                "bbox": [103, 175, 511, 190],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "and e-game levels (PR3) Reference frameworks: ⮚ GreenComp - The European Sustainability Competence Framework responds to\n",
            document,
        )

        self.assertIn("# Reference frameworks:", markdown)
        self.assertIn("⮚ GreenComp", markdown)

    def test_benchmark_projection_rejects_sentence_colon_lead_in_before_list_items(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "sentence-lead-in.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The contributions of this study are as follows:",
                                "bbox": [72, 112, 340, 126],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Introduction of the model",
                                "bbox": [90, 140, 240, 154],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Superior performance across benchmarks",
                                "bbox": [90, 158, 330, 172],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "The contributions of this study are as follows: Introduction of the model Superior performance across benchmarks\n",
            document,
        )

        self.assertNotIn("# The contributions of this study are as follows:", markdown)
        self.assertIn("The contributions of this study are as follows:", markdown)

    def test_benchmark_projection_rejects_section_like_sentence_continuation(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "continuation.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "2020 and that reduced to 18% in January 2021. Figure",
                                "bbox": [329, 603, 557, 614],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3.1.1 and Table 3.1.1 do not reflect those MSMEs who",
                                "bbox": [329, 615, 557, 626],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "were permanently closed; this was four in July 2020,",
                                "bbox": [329, 627, 557, 638],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "2020 and that reduced to 18% in January 2021. Figure\n\n"
            "3.1.1 and Table 3.1.1 do not reflect those MSMEs who\n\n"
            "were permanently closed; this was four in July 2020,\n",
            document,
        )

        self.assertNotIn("# 3.1.1 and Table", markdown)

    def test_benchmark_projection_rejects_colon_prompt_as_section_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "prompt.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Our Mental Shortcuts",
                                "bbox": [70, 180, 240, 198],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Reflection & Discussion Question 1: Taking Stock of What You Already Know",
                                "bbox": [70, 270, 520, 288],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Already Know",
                                "bbox": [70, 292, 160, 308],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "page_number",
                                "unit_role": "page_number",
                                "text": "98 | Instructor Resources",
                                "bbox": [420, 700, 560, 714],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Our Mental Shortcuts\n\n"
            "Reflection & Discussion Question 1: Taking Stock of What You Already Know\n\n"
            "98 | Instructor Resources\n",
            document,
        )

        self.assertIn("# Our Mental Shortcuts", markdown)
        self.assertNotIn("# Reflection & Discussion Question", markdown)

    def test_benchmark_projection_splits_section_heading_embedded_in_table_title_tail(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "table-title-tail.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                                "title": (
                                    "Table 5: Ablation studies on the different SFT base models used during the "
                                    "direct preference optimization stage. indicate that using OpenOrca results "
                                    "in a model that 4.3.2 Alignment Tuning"
                                ),
                                "display_grid": [["Model", "Score"], ["DPO v2", "73.42"]],
                                "semantic_grid": [["Model", "Score"], ["DPO v2", "73.42"]],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "title": (
                        "Table 5: Ablation studies on the different SFT base models used during the "
                        "direct preference optimization stage. indicate that using OpenOrca results "
                        "in a model that 4.3.2 Alignment Tuning"
                    ),
                    "display_grid": [["Model", "Score"], ["DPO v2", "73.42"]],
                    "semantic_grid": [["Model", "Score"], ["DPO v2", "73.42"]],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "**Table 5: Ablation studies on the different SFT base models used during the "
            "direct preference optimization stage. indicate that using OpenOrca results "
            "in a model that 4.3.2 Alignment Tuning**\n\n"
            "| Model | Score |\n| --- | --- |\n| DPO v2 | 73.42 |\n",
            document,
        )

        self.assertIn("# 4.3.2 Alignment Tuning", markdown)

    def test_benchmark_projection_releases_lowercase_body_tail_from_table_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "table-title-body-tail.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_1",
                                "table_id": "tbl_1",
                                "title": (
                                    "Table 7: Ablation studies on the different merge methods used for obtaining "
                                    "the final model. The best scores for H6 and the individual tasks are shown "
                                    "in bold. tively impacted by adding Synth. Math-Alignment. Thus, alignment "
                                    "is beneficial."
                                ),
                                "display_grid": [["Model", "Score"], ["Merge v1", "74.00"]],
                                "semantic_grid": [["Model", "Score"], ["Merge v1", "74.00"]],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "title": (
                        "Table 7: Ablation studies on the different merge methods used for obtaining "
                        "the final model. The best scores for H6 and the individual tasks are shown "
                        "in bold. tively impacted by adding Synth. Math-Alignment. Thus, alignment "
                        "is beneficial."
                    ),
                    "display_grid": [["Model", "Score"], ["Merge v1", "74.00"]],
                    "semantic_grid": [["Model", "Score"], ["Merge v1", "74.00"]],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn(
            "**Table 7: Ablation studies on the different merge methods used for obtaining "
            "the final model. The best scores for H6 and the individual tasks are shown in bold.**",
            markdown,
        )
        self.assertNotIn("shown in bold. tively impacted", markdown)
        self.assertIn("| Merge v1 | 74.00 |", markdown)
        self.assertLess(markdown.index("| Merge v1 | 74.00 |"), markdown.index("tively impacted by adding Synth. Math-Alignment."))

    def test_benchmark_projection_splits_embedded_colon_year_heading_after_sentence(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "embedded-colon-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Source: Tancangco 1991 as cited in Valte (1992).",
                                "bbox": [70, 420, 360, 436],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Current Situation: 2001-2019",
                                "bbox": [70, 472, 260, 490],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Filipino women are still very much a minority in the formal political sphere.",
                                "bbox": [70, 512, 548, 528],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Source: Tancangco 1991 as cited in Valte (1992). Current Situation: 2001-2019\n\n"
            "Filipino women are still very much a minority in the formal political sphere.\n",
            document,
        )

        self.assertIn("Source: Tancangco 1991 as cited in Valte (1992).\n\n# Current Situation: 2001-2019", markdown)

    def test_benchmark_projection_rejects_multiline_prompt_block_as_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "prompt.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Our Mental Shortcuts",
                                "bbox": [56, 324, 202, 341],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "If you would like to reinforce these ideas, use the video below.",
                                "bbox": [56, 367, 342, 378],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Reflection & Discussion Question 1: Taking Stock of What You",
                                "bbox": [71, 468, 298, 478],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Already Know",
                                "bbox": [71, 481, 124, 491],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Our Mental Shortcuts", markdown)
        self.assertNotIn("# Reflection & Discussion Question", markdown)
        self.assertIn("Reflection & Discussion Question 1: Taking Stock of What You Already Know", markdown)

    def test_benchmark_projection_rejects_numbered_body_instructions_despite_ast_heading_role(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "instructions.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Procedure:",
                                "bbox": [72, 361, 132, 378],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "4. Carefully pour the contents of the test tubes into the correspondingly labeled",
                                "bbox": [90, 389, 479, 405],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "saccharometer, ensuring that the solutions are well mixed.",
                                "bbox": [108, 403, 390, 419],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Procedure:", markdown)
        self.assertNotIn("# 4. Carefully pour", markdown)

    def test_benchmark_projection_rejects_numbered_procedure_step_sequence(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "procedure-sequence.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "17. Carefully pour off the supernatant from both tubes.",
                                "bbox": [72, 100, 410, 112],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "18. Briefly spin the tubes in a balanced configuration in the microcentrifuge to bring any remaining ethanol to",
                                "bbox": [72, 138, 540, 150],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "the bottom of the tube.",
                                "bbox": [72, 152, 210, 164],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "19. Allow the tubes to dry by leaving the tube caps open for 3-5 minutes.",
                                "bbox": [72, 188, 510, 200],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "body_list_item",
                                "unit_role": "body",
                                "text": "II. Set Up the Restriction Digests of the Suspect and Evidence DNA",
                                "bbox": [72, 346, 452, 362],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("# 18. Briefly spin", markdown)
        self.assertNotIn("# 19. Allow", markdown)
        self.assertIn("# II. Set Up the Restriction Digests", markdown)

    def test_benchmark_projection_rejects_numbered_source_note_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "source-note.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The Data Journey",
                                "bbox": [56, 57, 274, 87],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1. Statistics Canada. Table 32-10-0364-01 Area, production and farm gate",
                                "bbox": [50, 494, 337, 504],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "value of marketed fruits. Retrieved January 9th, 2022. DOI: https://doi.org/10.25318/3210036401-eng.",
                                "bbox": [56, 507, 337, 542],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# The Data Journey", markdown)
        self.assertNotIn("# 1. Statistics Canada", markdown)

    def test_benchmark_projection_rejects_page_number_label_before_running_header(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "page-number.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "322",
                                "bbox": [62, 32, 76, 48],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Yarrow",
                                "bbox": [355, 33, 388, 47],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "12 Conclusion",
                                "bbox": [62, 328, 146, 344],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("# 322", markdown)
        self.assertIn("# 12 Conclusion", markdown)

    def test_benchmark_projection_merges_adjacent_heading_continuation_lines(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "split-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Author's Note to the",
                                "bbox": [112, 135, 316, 164],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "2021 Edition",
                                "bbox": [147, 161, 276, 190],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "This book is a minimally amended reprint.",
                                "bbox": [57, 228, 368, 242],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Author's Note to the 2021 Edition", markdown)
        self.assertNotIn("# 2021 Edition", markdown)

    def test_benchmark_projection_does_not_promote_page_top_lowercase_body_tail_group(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "body-tail.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "be used as a good opportunity to learn from each other and increase the capacity of",
                                "bbox": [72, 72, 526, 86],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "human rights institutions in various countries.94",
                                "bbox": [72, 88, 320, 102],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3.2.6. SDGs Dissemination in Social Media",
                                "bbox": [72, 180, 360, 198],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("# be used as a good opportunity", markdown)
        self.assertIn("# 3.2.6. SDGs Dissemination in Social Media", markdown)

    def test_benchmark_projection_does_not_promote_sentence_like_multiline_instruction_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "instruction-body.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "03- Generate Slides with ChatGPT",
                                "bbox": [72, 80, 310, 100],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Provide the summarized content to ChatGPT and instruct it to create a structured outline for Google Slides, including titles,",
                                "bbox": [72, 128, 520, 146],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "04 - Create App Script Code",
                                "bbox": [72, 184, 300, 204],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# 03- Generate Slides with ChatGPT", markdown)
        self.assertNotIn("# Provide the summarized content", markdown)
        self.assertIn("# 04 - Create App Script Code", markdown)

    def test_benchmark_projection_promotes_top_band_subtitle_before_table(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "subtitle-before-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 612,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "AI Pack",
                                "bbox": [70, 38, 110, 54],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Upstage offers 3 AI packs that process unstructured information and data,",
                                "bbox": [70, 55, 736, 82],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "making a tangible impact on your business",
                                "bbox": [70, 86, 449, 113],
                            },
                            {
                                "block_type": "table",
                                "bbox": [49, 153, 913, 482],
                                "text": "",
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn(
            "# Upstage offers 3 AI packs that process unstructured information and data, making a tangible impact on your business",
            markdown,
        )

    def test_benchmark_projection_splits_heading_glued_to_list_item_suffix(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "glued-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Load the Gel",
                                "bbox": [72, 436, 137, 450],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "body_list_item",
                                "unit_role": "body",
                                "text": "1. Use a micropipette to add loading dye.",
                                "bbox": [72, 466, 530, 475],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "• 1-20 uL Micropipette and pipet tips Load the Gel\n\n1. Use a micropipette to add loading dye.\n",
            document,
        )

        self.assertIn("pipet tips\n\n# Load the Gel\n\n1. Use", markdown)

    def test_benchmark_projection_rejects_top_running_header_before_centered_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "header.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "MOHAVE COMMUNITY COLLEGE BIO181",
                                "bbox": [80, 35, 540, 56],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Cellular Replication",
                                "bbox": [244, 72, 371, 92],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Growth and the Creation of Life",
                                "bbox": [72, 364, 303, 384],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("# MOHAVE COMMUNITY COLLEGE BIO181", markdown)
        self.assertIn("# Cellular Replication", markdown)

    def test_benchmark_projection_inserts_missing_internal_table_title_before_table(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "missing-title-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": []},
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "title_row_index": 0,
                    "display_grid": [
                        ["Key Functions by Main Service Flow", None, None],
                        ["Stage", "Function", "Benefit"],
                        ["Creation", "Project setup", "Faster work"],
                    ],
                    "semantic_grid": [
                        ["Stage", "Function", "Benefit"],
                        ["Creation", "Project setup", "Faster work"],
                    ],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection._project_benchmark_table_title_headings(
            "| Stage | Function | Benefit |\n| --- | --- | --- |\n| Creation | Project setup | Faster work |\n",
            document,
        )

        self.assertTrue(markdown.startswith("# Key Functions by Main Service Flow\n\n| Stage | Function | Benefit |"))

    def test_benchmark_projection_promotes_tight_midpage_heading_before_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "tight-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Cellular Replication",
                                "bbox": [244, 72, 371, 92],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Growth and the Creation of Life",
                                "bbox": [72, 364, 303, 384],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "One of the characteristics of living things is the ability",
                                "bbox": [79, 391, 388, 410],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Growth and the Creation of Life", markdown)

    def test_benchmark_projection_promotes_section_title_after_page_label_before_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "page-label-section-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Prologue",
                                "bbox": [60, 35, 98, 45],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "xvii",
                                "bbox": [300, 36, 320, 45],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Functional Abstraction",
                                "bbox": [60, 57, 196, 69],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "But this corrected use of Leibniz notation is ugly.",
                                "bbox": [60, 82, 330, 95],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Prologue xvii Functional Abstraction But this corrected use of Leibniz notation is ugly.\n",
            document,
        )

        self.assertIn("# Functional Abstraction", markdown)
        self.assertIn("But this corrected use", markdown)

    def test_benchmark_projection_promotes_title_below_running_header_separator(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "running-header-separator-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Prologue",
                                "bbox": [60.84, 35.14, 97.94, 45.11],
                            },
                            {
                                "block_type": "image",
                                "bbox": [60.83, 47.95, 371.75, 48.43],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Functional Abstraction",
                                "bbox": [60.84, 57.48, 196.35, 69.44],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "But this corrected use of Leibniz notation is ugly.",
                                "bbox": [60.84, 82.14, 309.61, 93.05],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Prologue Functional Abstraction But this corrected use of Leibniz notation is ugly.\n",
            document,
        )

        self.assertNotIn("# Prologue", markdown)
        self.assertIn("# Functional Abstraction", markdown)
        self.assertIn("But this corrected use", markdown)

    def test_benchmark_projection_promotes_multiline_side_card_heading_not_axis_label(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "side-card.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Cellular Cycle",
                                "bbox": [462, 242, 542, 257],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "and Replication",
                                "bbox": [462, 256, 553, 270],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "A step by step",
                                "bbox": [462, 407, 533, 424],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Cellular Cycle and Replication", markdown)

    def test_benchmark_projection_rejects_bullet_list_and_license_note_groups_as_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "body-groups.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "• Restriction digests from Part II, on ice",
                                "bbox": [90, 340, 255, 351],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "• 10x loading dye, 10 μL",
                                "bbox": [90, 351, 197, 362],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Fact-checking) and is used under a CC BY-SA 3.0 license.",
                                "bbox": [56, 541, 341, 553],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "page_number",
                                "unit_role": "page_number",
                                "text": "48 | Types of Sources",
                                "bbox": [56, 564, 139, 574],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("# • Restriction", markdown)
        self.assertNotIn("# Fact-checking", markdown)

    def test_benchmark_projection_keeps_numbered_section_heading_before_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "numbered-section.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Previous paragraph ends here.",
                                "bbox": [62, 462, 107, 478],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "8 Choosing between Observer Models and Rejecting Participants",
                                "bbox": [62, 503, 230, 519],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "In this chapter, I have presented two variants of a latency-based observer model.",
                                "bbox": [62, 530, 388, 546],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# 8 Choosing between Observer Models and Rejecting Participants", markdown)

    def test_benchmark_projection_keeps_explicit_numbered_section_heading_before_indented_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "explicit-numbered-section.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "4. Entropy",
                                "bbox": [75, 355, 128, 368],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "According to Boltzmann, the total entropy of a macro-state is described.",
                                "bbox": [89, 380, 427, 392],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "4. Entropy\n\nAccording to Boltzmann, the total entropy of a macro-state is described.\n",
            document,
        )

        self.assertIn("# 4. Entropy", markdown)
        self.assertIn("According to Boltzmann", markdown)

    def test_benchmark_projection_reorders_top_section_heading_before_column_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "top-section-heading-order.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Left-column body that was emitted before the title.",
                                "bbox": [40, 122, 265, 134],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "2. General Profile of MSMEs",
                                "bbox": [159, 68, 500, 101],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Right-column body after the title.",
                                "bbox": [275, 123, 503, 135],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertLess(
            markdown.index("# 2. General Profile of MSMEs"),
            markdown.index("Left-column body that was emitted before the title."),
        )

    def test_benchmark_projection_keeps_activity_section_heading_before_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "activity-section.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Activity 1: Determining pH With Indicator Strips (Field Method)",
                                "bbox": [56, 270, 430, 285],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Of the several techniques available for determining pH, one can be used in the field.",
                                "bbox": [56, 308, 558, 319],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Activity 1: Determining pH With Indicator Strips (Field Method)", markdown)

    def test_benchmark_projection_keeps_dotted_activity_section_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "activity-dot.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Activity 4. Determining CEC by replacing adsorbed cations.",
                                "bbox": [56, 270, 401, 285],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "In this activity, the sample is extracted and measured.",
                                "bbox": [56, 300, 401, 315],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Activity 4. Determining CEC by replacing adsorbed cations.", markdown)

    def test_benchmark_projection_recovers_same_left_chapter_number_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "chapter-inline.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "4",
                                "bbox": [60, 53, 75, 78],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Basis Fields",
                                "bbox": [60, 84, 162, 101],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "A vector field may be written as a linear combination.",
                                "bbox": [60, 120, 372, 131],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "4 Basis Fields A vector field may be written as a linear combination.\n",
            document,
        )

        self.assertIn("# 4 Basis Fields", markdown)

    def test_benchmark_projection_recovers_centered_chapter_number_and_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "chapter-centered.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "2",
                                "bbox": [204, 133, 219, 171],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The Lost Homeland",
                                "bbox": [114, 179, 309, 208],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Since the death of my mother, I have been haunted.",
                                "bbox": [57, 246, 368, 260],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# 2", markdown)
        self.assertIn("# The Lost Homeland", markdown)

    def test_benchmark_projection_keeps_short_heading_with_section_spacing_before_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "short-section.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "as our final model.",
                                "bbox": [306, 597, 470, 608],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Conclusion",
                                "bbox": [324, 620, 381, 632],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "We introduce the model and its fine-tuned variant.",
                                "bbox": [306, 641, 526, 652],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Conclusion", markdown)

    def test_benchmark_projection_promotes_short_title_after_paragraph_boundary(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "short-title-after-paragraph.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "4.2 Definitions",
                                "bbox": [90.0, 604.73, 196.45, 619.08],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "In this chapter, various iterative methods will be considered to solve nonlinear equations of the",
                                "bbox": [90.0, 630.74, 512.97, 640.84],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "form f(p) = 0. The point p is called a zero of the function f, or a root of the equation f(x) = 0.",
                                "bbox": [90.0, 642.11, 513.09, 652.72],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "First, some useful definitions and concepts are introduced.",
                                "bbox": [90.0, 654.62, 347.25, 664.72],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Convergence",
                                "bbox": [90.0, 671.66, 149.34, 681.62],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Each numerical method generates a sequence {pn} = p0, p1, p2, . . . which should converge to p:",
                                "bbox": [90.0, 683.15, 513.09, 700.91],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Convergence", markdown)

    def test_benchmark_projection_keeps_page_top_single_word_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "single-word-top.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Stop",
                                "bbox": [56, 58, 90, 75],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Check your emotions. If a claim causes strong emotion, stop.",
                                "bbox": [56, 101, 216, 112],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# Stop", markdown)

    def test_benchmark_projection_keeps_split_numbered_heading_continuation(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "split-numbered-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3. Perspective of supply and demand balance of wood pellets and cost",
                                "bbox": [85, 88, 484, 101],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "structure in Japan",
                                "bbox": [113, 106, 213, 119],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "According to a survey, biomass power generation is domestically produced.",
                                "bbox": [85, 130, 484, 141],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# 3. Perspective of supply and demand balance of wood pellets and cost", markdown)

    def test_benchmark_projection_recovers_infographic_main_title_below_page_label(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "infographic-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Semantic Search Pack: Value",
                                "bbox": [70.98, 37.78, 222.30, 53.66],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "SS Pack allows businesses to access further data more rapidly",
                                "bbox": [70.31, 65.46, 617.72, 92.32],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The SS Pack can reduce the information acquisition time by returning all the information that matches the user's search intent.",
                                "bbox": [48.82, 166.41, 612.59, 179.84],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Semantic Search Pack: Value SS Pack allows businesses to access further data more rapidly The SS Pack can reduce the information acquisition time.",
            document,
        )

        self.assertNotIn("# Semantic Search Pack: Value", markdown)
        self.assertIn("# SS Pack allows businesses to access further data more rapidly", markdown)

    def test_benchmark_projection_recovers_supported_peer_card_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "infographic-cards.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "1.8X",
                                "bbox": [95.25, 274.26, 136.19, 302.35],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Optimal Attempt",
                                "bbox": [370.01, 278.57, 514.57, 304.21],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "SOTA",
                                "bbox": [661.80, 272.58, 712.24, 300.67],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Higher Return of Information",
                                "bbox": [95.25, 305.86, 249.59, 321.74],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Reduced Information Acquisition Time",
                                "bbox": [370.01, 309.22, 573.57, 325.10],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Cutting-Edge Technology",
                                "bbox": [661.80, 304.18, 794.05, 320.06],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Unlike existing search systems that only return",
                                "bbox": [95.25, 343.76, 286.76, 355.97],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "By returning all semantic-based information of the",
                                "bbox": [370.01, 343.76, 577.02, 355.97],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The analysis of user logs saved in real-time allows us",
                                "bbox": [661.80, 343.76, 875.56, 355.97],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            (
                "1.8X Optimal Attempt SOTA Higher Return of Information Reduced Information Acquisition Time "
                "Cutting-Edge Technology Unlike existing search systems that only return"
            ),
            document,
        )

        self.assertIn("# Higher Return of Information", markdown)
        self.assertIn("# Reduced Information Acquisition Time", markdown)
        self.assertIn("# Cutting-Edge Technology", markdown)

    def test_benchmark_projection_orders_infographic_card_deck_by_ownership_lanes(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        def text_block(text: str, bbox: list[float]) -> dict[str, object]:
            return {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": text,
                "bbox": bbox,
            }

        document = {
            "filename": "infographic-card-deck.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            text_block("Semantic Search Pack: Value", [70.98, 37.78, 222.30, 53.66]),
                            text_block(
                                "SS Pack allows businesses to access further data more rapidly",
                                [70.31, 65.46, 617.72, 92.32],
                            ),
                            text_block(
                                "The SS Pack can reduce the information acquisition time by returning all the information that matches the user's search intent.",
                                [48.82, 166.41, 612.59, 179.84],
                            ),
                            text_block("1.8X", [95.25, 274.26, 136.19, 302.35]),
                            text_block("Optimal Attempt", [370.01, 278.57, 514.57, 304.21]),
                            text_block("SOTA", [661.80, 272.58, 712.24, 300.67]),
                            text_block("Higher Return of Information", [95.25, 305.86, 249.59, 321.74]),
                            text_block("Reduced Information Acquisition Time", [370.01, 309.22, 573.57, 325.10]),
                            text_block("Cutting-Edge Technology", [661.80, 304.18, 794.05, 320.06]),
                            text_block("Unlike existing search systems that only return", [95.25, 343.76, 286.76, 355.97]),
                            text_block("By returning all semantic-based information of the", [370.01, 343.76, 577.02, 355.97]),
                            text_block("The analysis of user logs saved in real-time allows us", [661.80, 343.76, 875.56, 355.97]),
                            text_block("information limited to the entered search keywords, SS", [95.25, 359.84, 321.76, 372.05]),
                            text_block("search keywords, the time required for information", [370.01, 359.84, 581.02, 372.05]),
                            text_block("to further optimize the individual search services", [661.80, 359.84, 862.19, 372.05]),
                            text_block("Pack returns all relevant data that meet the user's", [95.25, 376.64, 300.38, 388.85]),
                            text_block("acquisition is reduced drastically compared to that", [370.01, 376.64, 577.02, 388.85]),
                            text_block("over time", [661.80, 376.64, 700.91, 388.85]),
                            text_block("search intent", [95.25, 393.68, 148.60, 405.89]),
                            text_block("of traditional keyword-matching search systems", [370.01, 393.68, 564.67, 405.89]),
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "page": 1,
                                "bbox": [46.32, 479.28, 573.36, 500.64],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "page": 1,
                    "bbox": [46.32, 479.28, 573.36, 500.64],
                    "image_kind_guess": "path_screenshot",
                    "figure_ref": "Figure 1",
                    "caption_text": "",
                    "title": "",
                    "embedded_text": "",
                    "embedded_text_confidence": 0.0,
                    "content_segments": [
                        {"role": "nearby_context", "text": "search intent"},
                    ],
                    "content_signals": {
                        "has_caption": False,
                        "has_embedded_text": False,
                        "has_nearby_context": True,
                        "has_path": True,
                    },
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# 1.8X", markdown)
        self.assertLess(markdown.index("# Higher Return of Information"), markdown.index("# Optimal Attempt"))
        self.assertLess(markdown.index("# Optimal Attempt"), markdown.index("# SOTA"))
        self.assertLess(
            markdown.index("Unlike existing search systems that only return"),
            markdown.index("# Optimal Attempt"),
        )
        self.assertLess(
            markdown.index("of traditional keyword-matching search systems"),
            markdown.index("# SOTA"),
        )
        self.assertNotIn("![Figure 1](#img_p1_001)", markdown)

    def test_benchmark_projection_orders_landscape_panel_page_by_ownership_lanes(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        def text_block(text: str, bbox: list[float]) -> dict[str, object]:
            return {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": text,
                "bbox": bbox,
            }

        document = {
            "filename": "landscape-panel-page.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            text_block("H O W C A N", [54.53, 13.36, 211.16, 43.24]),
                            text_block("FURTHER", [326.44, 21.23, 463.45, 51.23]),
                            text_block("Y O U H E L P ?", [43.64, 40.36, 222.06, 70.24]),
                            text_block("As a boater:", [13.22, 74.68, 78.43, 86.67]),
                            text_block("RESOURCES", [305.71, 55.73, 484.16, 85.73]),
                            text_block("Check tidal conditions beforehand", [30.21, 88.18, 218.4, 100.17]),
                            text_block("Take a safe boating course", [30.21, 169.18, 173.98, 181.17]),
                            text_block("SEAGRASS", [567.76, 69.95, 754.11, 109.8]),
                            text_block("IN SOUTH FLORIDA", [541.51, 104.71, 780.35, 133.51]),
                            text_block("WHY IT IS IMPORTANT", [552.5, 142.5, 769.38, 162.73]),
                            text_block("As a developer:", [13.22, 196.18, 97.39, 208.17]),
                            text_block("Do careful mapping of seagrass in", [30.21, 209.68, 214.23, 221.67]),
                            text_block("As anyone who wants to help:", [13.22, 398.68, 181.63, 410.67]),
                            text_block("Tell your friends and family about the", [30.21, 560.68, 234.24, 572.67]),
                            text_block("importance of this ecosystem", [30.21, 574.18, 193.64, 586.17]),
                            text_block("FLOWCODE", [291.58, 307.71, 500.08, 516.21]),
                            text_block("Scan this QR code and learn", [354.51, 526.99, 496.35, 536.99]),
                            text_block("its restoration!", [391.67, 579.82, 462.02, 590.99]),
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertLess(markdown.index("Y O U H E L P ?"), markdown.index("FURTHER"))
        self.assertLess(markdown.index("Take a safe boating course"), markdown.index("As a developer:"))
        self.assertLess(markdown.index("importance of this ecosystem"), markdown.index("FURTHER"))
        self.assertLess(markdown.index("its restoration!"), markdown.index("SEAGRASS"))

    def test_benchmark_projection_suppresses_caption_only_image_placeholder(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "caption-only-figures.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "page": 1,
                                "bbox": [66, 69, 378, 283],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 1.5. The jacket.",
                                "bbox": [126, 289, 318, 302],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "page": 1,
                    "bbox": [66, 69, 378, 283],
                    "image_kind_guess": "captioned_figure",
                    "caption_text": "Figure 1.5. The jacket.",
                    "title": "Figure 1.5. The jacket.",
                    "embedded_text": "",
                    "content_segments": [
                        {
                            "role": "caption",
                            "text": "Figure 1.5. The jacket.",
                            "source_block_id": "txt_p1_003",
                        }
                    ],
                    "content_signals": {
                        "has_caption": True,
                        "has_embedded_text": False,
                        "caption_owned_text_block_count": 1,
                    },
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotRegex(markdown, r"!\[[^\]]*\]\(#img_p1_001\)")
        self.assertIn("Figure 1.5. The jacket.", markdown)

    def test_benchmark_projection_deduplicates_prefix_figure_caption_paragraphs(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        markdown = projection._deduplicate_prefix_figure_caption_paragraphs(
            "\n".join(
                [
                    "Figure 2.1. Primary caption opening",
                    "",
                    "Figure 2.1. Primary caption opening with owned continuation text.",
                    "",
                    "Figure 2.2. Next figure caption.",
                ]
            )
        )

        self.assertNotIn("Figure 2.1. Primary caption opening\n\nFigure 2.1.", markdown)
        self.assertIn("Figure 2.1. Primary caption opening with owned continuation text.", markdown)
        self.assertIn("Figure 2.2. Next figure caption.", markdown)

    def test_benchmark_projection_splits_embedded_figure_caption_paragraphs(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        markdown = projection._split_embedded_figure_caption_paragraphs(
            "Figure 2.1. First caption continuation. Figure 2.2. Second caption starts here."
        )

        self.assertEqual(
            markdown.rstrip(),
            "Figure 2.1. First caption continuation.\n\nFigure 2.2. Second caption starts here.",
        )

    def test_benchmark_projection_suppresses_embedded_text_image_placeholder(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "embedded-text-figure.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "page": 1,
                                "bbox": [66, 69, 378, 283],
                                "embedded_text": "The HONEY-MOON",
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "page": 1,
                    "bbox": [66, 69, 378, 283],
                    "image_kind_guess": "graphic_image",
                    "caption_text": "",
                    "title": "",
                    "embedded_text": "The HONEY-MOON",
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": "The HONEY-MOON",
                            "source": "ocr",
                            "confidence": 0.91,
                        }
                    ],
                    "content_signals": {
                        "has_caption": False,
                        "has_embedded_text": True,
                        "embedded_text_source": "ocr",
                    },
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotRegex(markdown, r"!\[[^\]]*\]\(#img_p1_001\)")
        self.assertIn("The HONEY-MOON", markdown)

    def test_benchmark_projection_moves_owned_side_caption_after_image_text(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "side-caption-figure.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "74",
                                "bbox": [56, 31, 66, 47],
                            },
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "bbox": [56, 56, 330, 432],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "this list, Richard Walker begins the main paragraph.",
                                "bbox": [56, 449, 269, 465],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "34 Richard Walker, Memoirs of Medicine.",
                                "bbox": [56, 631, 268, 644],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 4.3",
                                "bbox": [341, 379, 382, 392],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The Honey-Moon [graphic]. Mezzotint,",
                                "bbox": [341, 390, 475, 403],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "hand-colored.",
                                "bbox": [341, 401, 389, 414],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "page": 1,
                    "bbox": [56, 56, 330, 432],
                    "image_kind_guess": "graphic_image",
                    "caption_text": "",
                    "title": "",
                    "embedded_text": "The HONEY-MOON",
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": "The HONEY-MOON",
                            "source": "ocr",
                            "confidence": 0.91,
                        }
                    ],
                    "content_signals": {
                        "has_caption": False,
                        "has_embedded_text": True,
                    },
                }
            ],
        }
        markdown = "\n\n".join(
            [
                "74",
                "The HONEY-MOON",
                "this list, Richard Walker begins the main paragraph.",
                "34 Richard Walker, Memoirs of Medicine.",
                "Figure 4.3 The Honey-Moon [graphic]. Mezzotint, hand-colored.",
            ]
        )

        projected = projection._project_owned_figure_captions_after_image_text(markdown, document)

        self.assertLess(projected.index("The HONEY-MOON"), projected.index("Figure 4.3"))
        self.assertLess(projected.index("Figure 4.3"), projected.index("this list, Richard Walker"))
        self.assertEqual(projected.count("Figure 4.3"), 1)

    def test_benchmark_projection_does_not_split_caption_group_with_preceding_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        blocks = [
            {
                "block_type": "image",
                "image_id": "img_p1_001",
                "bbox": [72, 246, 314, 506],
                "embedded_text": "Defensoria del Pueblo @DPNArgentina",
            },
            {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": "DPN Argentina",
                "bbox": [408, 392, 481, 403],
            },
            {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": "Figure 6 Content: World Health",
                "bbox": [344, 403, 516, 419],
            },
            {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": "Day Celebration",
                "bbox": [408, 415, 490, 426],
            },
        ]

        caption_blocks = projection._side_or_below_caption_blocks_for_image(blocks, (72, 246, 314, 506))

        self.assertEqual(caption_blocks, [])

    def test_benchmark_projection_moves_owned_caption_before_bottom_notes(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "caption-before-bottom-notes.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "bbox": [280, 56, 480, 222],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 4.2 William Hogarth, Taste in High Life [graphic].",
                                "bbox": [280, 234, 481, 247],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Print made by Isaac Mills after William Hogarth.",
                                "bbox": [326, 245, 482, 258],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "author_line",
                                "unit_role": "metadata",
                                "text": "25 Wiliam Beckford, An Arabian Tale.",
                                "bbox": [56, 535, 266, 548],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "bbox": [280, 56, 480, 222],
                    "image_kind_guess": "captioned_figure",
                    "caption_text": "Figure 4.2 William Hogarth, Taste in High Life [graphic].",
                    "embedded_text": "",
                    "content_segments": [
                        {
                            "role": "caption",
                            "text": "Figure 4.2 William Hogarth, Taste in High Life [graphic].",
                        },
                        {
                            "role": "nearby_context",
                            "text": "Print made by Isaac Mills after William Hogarth.",
                        },
                    ],
                }
            ],
        }
        markdown = "\n\n".join(
            [
                "Main body text before notes.",
                "25 Wiliam Beckford, An Arabian Tale.",
                "Figure 4.2 William Hogarth, Taste in High Life [graphic].",
                "Print made by Isaac Mills after William Hogarth.",
                "Right column body after figure.",
            ]
        )

        projected = projection._project_owned_figure_captions_before_page_notes(markdown, document)

        self.assertLess(projected.index("Figure 4.2"), projected.index("25 Wiliam Beckford"))
        self.assertLess(projected.index("25 Wiliam Beckford"), projected.index("Right column body"))

    def test_benchmark_projection_keeps_path_screenshot_embedded_text_placeholder(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "table-screenshot.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "page": 1,
                                "bbox": [0, 0, 720, 405],
                                "embedded_text": "Key Functions by Main Service Flow",
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "page": 1,
                    "bbox": [0, 0, 720, 405],
                    "image_kind_guess": "path_screenshot",
                    "caption_text": "",
                    "title": "",
                    "embedded_text": "Key Functions by Main Service Flow",
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": "Key Functions by Main Service Flow",
                            "source": "text-layer",
                            "confidence": 0.98,
                        }
                    ],
                    "content_signals": {
                        "has_caption": False,
                        "has_embedded_text": True,
                        "embedded_text_source": "text-layer",
                        "has_path": True,
                    },
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertRegex(markdown, r"!\[[^\]]*\]\(#img_p1_001\)")
        self.assertIn("Key Functions by Main Service Flow", markdown)

    def test_benchmark_projection_suppresses_path_screenshot_placeholder_when_structured_table_owns_text(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "owned-screenshot-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 405,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "bbox": [0, 0, 720, 405],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_1",
                                "bbox": [34, 47, 677, 355],
                                "title_row_index": 0,
                                "display_grid": [
                                    ["Key Functions by Main Service Flow", None, None, None],
                                    ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                                    ["1. Project creation", "Project creation and management", "Select document type", "Improve work efficiency"],
                                ],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Introduction of product services and key features",
                                "bbox": [34, 26, 294, 40],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "bbox": [34, 47, 677, 355],
                    "title_row_index": 0,
                    "display_grid": [
                        ["Key Functions by Main Service Flow", None, None, None],
                        ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                        ["1. Project creation", "Project creation and management", "Select document type", "Improve work efficiency"],
                    ],
                }
            ],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "bbox": [0, 0, 720, 405],
                    "image_kind_guess": "path_screenshot",
                    "embedded_text": "Key Functions by Main Service Flow Introduction of product services and key features Service Stage Function Name",
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": "Key Functions by Main Service Flow Introduction of product services and key features Service Stage Function Name",
                            "source": "text-layer",
                            "confidence": 0.98,
                        }
                    ],
                    "content_signals": {
                        "has_embedded_text": True,
                        "embedded_text_source": "text-layer",
                        "has_path": True,
                    },
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotRegex(markdown, r"!\[[^\]]*\]\(#img_p1_001\)")
        self.assertLess(
            markdown.index("Introduction of product services and key features"),
            markdown.index("# Key Functions by Main Service Flow"),
        )
        self.assertIn("| Service Stage | Function Name | Explanation | Expected Benefit |", markdown)

    def test_benchmark_projection_projects_figure_owned_pseudo_table_as_plain_evidence(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "chart-owned-pseudo-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "bbox": [56, 56, 520, 289],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_001",
                                "bbox": [56, 56, 520, 289],
                                "detection_source": "embedded_image_ocr",
                                "display_grid": [
                                    ["Getting away from the usual demands", "Column 2", "34%"],
                                    [None, "Being close to nature", "33%"],
                                    ["Enjoying the sounds and smells of nature", None, "32%"],
                                ],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Over time, an angler's motivation may change.",
                                "bbox": [56, 310, 460, 324],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "bbox": [56, 56, 520, 289],
                    "detection_source": "embedded_image_ocr",
                    "detection_method": "embedded_image_ocr",
                    "display_grid": [
                        ["Getting away from the usual demands", "Column 2", "34%"],
                        [None, "Being close to nature", "33%"],
                        ["Enjoying the sounds and smells of nature", None, "32%"],
                    ],
                }
            ],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "bbox": [56, 56, 520, 289],
                    "image_kind_guess": "captioned_textual_figure",
                    "caption_text": "Figure 10.2: Positive attributes reported by recreational anglers.",
                    "title": "Figure 10.2: Positive attributes reported by recreational anglers.",
                    "embedded_text": (
                        "Getting away from the usual demands 34% Being close to nature 33% "
                        "Enjoying the sounds and smells of nature 32%"
                    ),
                    "content_segments": [
                        {
                            "role": "caption",
                            "text": "Figure 10.2: Positive attributes reported by recreational anglers.",
                        },
                        {
                            "role": "embedded_text",
                            "text": (
                                "Getting away from the usual demands 34% Being close to nature 33% "
                                "Enjoying the sounds and smells of nature 32%"
                            ),
                        },
                    ],
                    "content_signals": {
                        "has_caption": True,
                        "has_embedded_text": True,
                    },
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotRegex(markdown, r"!\[[^\]]*\]\(#img_p1_001\)")
        self.assertNotIn("| Getting away from the usual demands | Column 2 | 34% |", markdown)
        self.assertLess(
            markdown.index("Getting away from the usual demands 34%"),
            markdown.index("Figure 10.2: Positive attributes reported by recreational anglers."),
        )
        self.assertIn("Over time, an angler's motivation may change.", markdown)

    def test_benchmark_projection_removes_html_table_for_figure_owned_pseudo_table(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "dense-chart-owned-pseudo-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "bbox": [56, 56, 520, 289],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_001",
                                "bbox": [56, 56, 520, 289],
                                "detection_source": "embedded_image_ocr",
                                "display_grid": [
                                    ["Getting away from the usual demands", "Column 2", "34%"],
                                    [None, "Being close to nature", "33%"],
                                    ["Enjoying the sounds and smells of nature", None, "32%"],
                                    [None, "Catching fish", "31%"],
                                    ["Spending time with family or friends", None, "29%"],
                                    [None, "The scenic beauty 16%", None],
                                    [None, "Experiencing solitude 14%", None],
                                    ["Experiencing excitement/adventure", "14%", None],
                                    ["Reliving my childhood memories of going fishing", "12%", None],
                                    [None, "Catching my own food 12%", None],
                                    [None, "0% 10% 5% 15% 20%", "25% 30% 35% 40%"],
                                ],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Over time, an angler's motivation may change.",
                                "bbox": [56, 310, 460, 324],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "bbox": [56, 56, 520, 289],
                    "detection_source": "embedded_image_ocr",
                    "detection_method": "embedded_image_ocr",
                    "display_grid": [
                        ["Getting away from the usual demands", "Column 2", "34%"],
                        [None, "Being close to nature", "33%"],
                        ["Enjoying the sounds and smells of nature", None, "32%"],
                        [None, "Catching fish", "31%"],
                        ["Spending time with family or friends", None, "29%"],
                        [None, "The scenic beauty 16%", None],
                        [None, "Experiencing solitude 14%", None],
                        ["Experiencing excitement/adventure", "14%", None],
                        ["Reliving my childhood memories of going fishing", "12%", None],
                        [None, "Catching my own food 12%", None],
                        [None, "0% 10% 5% 15% 20%", "25% 30% 35% 40%"],
                    ],
                }
            ],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "bbox": [56, 56, 520, 289],
                    "image_kind_guess": "captioned_textual_figure",
                    "caption_text": "Figure 10.2: Positive attributes reported by recreational anglers.",
                    "title": "Figure 10.2: Positive attributes reported by recreational anglers.",
                    "embedded_text": (
                        "Getting awayfrom the usual demands 34% Being close to nature 33% "
                        "Enjoying the sounds and smells of nature 32% Catching fish 31% "
                        "Spending time with family or friends 29% The scenicbeauty 16% "
                        "Experiencing solitude 14% Experiencing excitement/adventure 14% "
                        "Reliving my childhood memories of going fishing 12% "
                        "Catching my own food 12% 5% 0% 10% 15% 20% 25% 30% 35% 40%"
                    ),
                    "content_segments": [
                        {
                            "role": "caption",
                            "text": "Figure 10.2: Positive attributes reported by recreational anglers.",
                        },
                        {
                            "role": "embedded_text",
                            "text": (
                                "Getting awayfrom the usual demands 34% Being close to nature 33% "
                                "Enjoying the sounds and smells of nature 32% Catching fish 31% "
                                "Spending time with family or friends 29% The scenicbeauty 16% "
                                "Experiencing solitude 14% Experiencing excitement/adventure 14% "
                                "Reliving my childhood memories of going fishing 12% "
                                "Catching my own food 12% 5% 0% 10% 15% 20% 25% 30% 35% 40%"
                            ),
                        },
                    ],
                    "content_signals": {
                        "has_caption": True,
                        "has_embedded_text": True,
                    },
                }
            ],
        }

        markdown = projection._project_figure_owned_pseudo_tables_in_flow(
            """
![Figure 1](#img_p1_001)

Getting awayfrom the usual demands 34% Being close to nature 33% Enjoying the sounds and smells of nature 32% Catching fish 31% Spending time with family or friends 29% The scenicbeauty 16% Experiencing solitude 14% Experiencing excitement/adventure 14% Reliving my childhood memories of going fishing 12% Catching my own food 12% 5% 0% 10% 15% 20% 25% 30% 35% 40%

Figure 10.2: Positive attributes reported by recreational anglers.

<table>
  <tr>
    <th>Getting away from the usual demands</th>
    <th>Column 2</th>
    <th>34%</th>
  </tr>
  <tr>
    <td></td>
    <td>Being close to nature</td>
    <td>33%</td>
  </tr>
</table>

Over time, an angler's motivation may change.
""",
            document,
        )

        self.assertNotRegex(markdown, r"!\[[^\]]*\]\(#img_p1_001\)")
        self.assertNotIn("<table>", markdown)
        self.assertNotIn("<th>Getting away from the usual demands</th>", markdown)
        self.assertLess(
            markdown.index("Getting awayfrom the usual demands 34%"),
            markdown.index("Figure 10.2: Positive attributes reported by recreational anglers."),
        )
        self.assertIn("Over time, an angler's motivation may change.", markdown)

    def test_benchmark_projection_projects_captioned_chart_ocr_table_as_figure_evidence(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "captioned-chart-ocr.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "page": 1,
                                "bbox": [98, 360, 467, 508],
                            },
                            {
                                "block_type": "text",
                                "text": "Figure 4.1. Approved Capacity under the FIT Scheme",
                                "bbox": [164, 340, 405, 351],
                            },
                            {
                                "block_type": "table",
                                "bbox": [98, 360, 467, 508],
                            },
                        ],
                    }
                ]
            },
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "page": 1,
                    "bbox": [98, 360, 467, 508],
                    "image_kind_guess": "captioned_textual_figure",
                    "caption_text": "Figure 4.1. Approved Capacity under the FIT Scheme",
                    "caption_bbox": [164, 340, 405, 351],
                    "embedded_text": "MW 700 Waste materials 600 Biogas 500 Construction wood waste 400",
                    "content_segments": [
                        {"role": "caption", "text": "Figure 4.1. Approved Capacity under the FIT Scheme"},
                        {"role": "embedded_text", "text": "MW 700 Waste materials 600 Biogas 500 Construction wood waste 400"},
                    ],
                    "content_signals": {
                        "has_caption": True,
                        "has_embedded_text": True,
                    },
                }
            ],
            "table_asts": [
                {
                    "bbox": [98, 360, 467, 508],
                    "detection_source": "embedded_image_ocr",
                    "display_grid": [
                        ["MW 700", None],
                        [None, "Waste materials"],
                        ["600", None],
                        ["500", "Biogas"],
                        ["400", "Construction wood waste"],
                        ["300", "General wood"],
                        ["200", "General wood"],
                        ["100", "Unutilised wood"],
                        ["0", "Unutilised wood"],
                    ],
                }
            ],
        }

        markdown = projection._project_figure_owned_pseudo_tables_in_flow(
            """
![Figure 1](#img_p1_001)

Figure 4.1. Approved Capacity under the FIT Scheme

| MW 700 | Column 2 |
| --- | --- |
|  | Waste materials |
| 600 |  |
| 500 | Biogas |
""",
            document,
        )

        self.assertNotRegex(markdown, r"!\[[^\]]*\]\(#img_p1_001\)")
        self.assertNotIn("| MW 700 |", markdown)
        self.assertLess(markdown.index("Figure 4.1"), markdown.index("MW 700"))

    def test_benchmark_projection_keeps_image_ocr_true_table_as_structured_table(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "figure-captioned-true-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_p1_001",
                                "bbox": [56, 350, 555, 628],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_001",
                                "bbox": [56, 350, 555, 628],
                                "detection_source": "embedded_image_ocr",
                                "display_grid": [
                                    [
                                        "Temperature (degree C)",
                                        "Kinematic viscosity v (m² /s)",
                                        "Temperature (degree C)",
                                        "Kinematic viscosity v (m² /s)",
                                    ],
                                    ["0", "1.793E-06", "25", "8.930E-07"],
                                    ["1", "1.732E-06", "26", "8.760E-07"],
                                    ["2", "1.674E-06", "27", "8.540E-07"],
                                ],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "bbox": [56, 350, 555, 628],
                    "detection_source": "embedded_image_ocr",
                    "detection_method": "embedded_image_ocr",
                    "display_grid": [
                        [
                            "Temperature (degree C)",
                            "Kinematic viscosity v (m² /s)",
                            "Temperature (degree C)",
                            "Kinematic viscosity v (m² /s)",
                        ],
                        ["0", "1.793E-06", "25", "8.930E-07"],
                        ["1", "1.732E-06", "26", "8.760E-07"],
                        ["2", "1.674E-06", "27", "8.540E-07"],
                    ],
                }
            ],
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "bbox": [56, 350, 555, 628],
                    "image_kind_guess": "captioned_textual_figure",
                    "caption_text": "Figure 7.2: Kinematic Viscosity of Water at Atmospheric Pressure.",
                    "title": "Figure 7.2: Kinematic Viscosity of Water at Atmospheric Pressure.",
                    "embedded_text": (
                        "Temperature (degree C) Kinematic viscosity v (m²/s) "
                        "Temperature (degree C) Kinematic viscosity v (m2/s) "
                        "0 1.793E-06 25 8.930E-07 1 1.732E-06 26 8.760E-07 "
                        "2 1.674E-06 27 8.540E-07"
                    ),
                    "content_segments": [
                        {
                            "role": "caption",
                            "text": "Figure 7.2: Kinematic Viscosity of Water at Atmospheric Pressure.",
                        },
                        {
                            "role": "embedded_text",
                            "text": (
                                "Temperature (degree C) Kinematic viscosity v (m²/s) "
                                "Temperature (degree C) Kinematic viscosity v (m2/s) "
                                "0 1.793E-06 25 8.930E-07 1 1.732E-06 26 8.760E-07 "
                                "2 1.674E-06 27 8.540E-07"
                            ),
                        },
                    ],
                    "content_signals": {
                        "has_caption": True,
                        "has_embedded_text": True,
                    },
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertRegex(markdown, r"!\[[^\]]*\]\(#img_p1_001\)")
        self.assertIn(
            "| Temperature (degree C) | Kinematic viscosity v (m² /s) | Temperature (degree C) | Kinematic viscosity v (m² /s) |",
            markdown,
        )
        self.assertIn("| 0 | 1.793E-06 | 25 | 8.930E-07 |", markdown)

    def test_benchmark_projection_projects_vector_chart_pseudo_table_as_plain_evidence(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "vector-chart-pseudo-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 1.9b. Deployment of Overseas Foreign Workers by sex, new hires only",
                                "bbox": [113, 108, 528, 123],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "(in thousands)",
                                "bbox": [181, 123, 257, 137],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "400",
                                "bbox": [113, 148, 131, 160],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "374",
                                "bbox": [170, 150, 188, 162],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_001",
                                "bbox": [140, 155, 524, 300],
                                "detection_source": "pymupdf_builtin",
                                "display_grid": [
                                    ["331 335 319", None, None, None],
                                    ["187", None, None, None],
                                    ["128", None, None, None],
                                    [None, "102 102", None, None],
                                    [None, None, "22", "55"],
                                ],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Male",
                                "bbox": [225, 306, 248, 319],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Female",
                                "bbox": [412, 306, 448, 319],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Source: Philippine Statistics Authority (2022)",
                                "bbox": [113, 353, 313, 365],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "bbox": [140, 155, 524, 300],
                    "detection_source": "pymupdf_builtin",
                    "detection_method": "pymupdf_builtin",
                    "display_grid": [
                        ["331 335 319", None, None, None],
                        ["187", None, None, None],
                        ["128", None, None, None],
                        [None, "102 102", None, None],
                        [None, None, "22", "55"],
                    ],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("| 331 335 319 |", markdown)
        self.assertIn("Figure 1.9b. Deployment of Overseas Foreign Workers by sex, new hires only", markdown)
        self.assertIn("331 335 319", markdown)
        self.assertLess(markdown.index("Figure 1.9b."), markdown.index("Source: Philippine Statistics Authority"))

    def test_benchmark_projection_vector_chart_pseudo_table_does_not_cross_section_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "vector-chart-before-section.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 820,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 1.9b. Deployment of Overseas Foreign Workers by sex, new hires only",
                                "bbox": [113, 108, 528, 123],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "400",
                                "bbox": [113, 148, 131, 160],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_001",
                                "bbox": [140, 155, 524, 300],
                                "detection_source": "pymupdf_builtin",
                                "display_grid": [
                                    ["331 335 319", None, None, None],
                                    ["187", None, None, None],
                                    [None, None, "22", "55"],
                                ],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1.5. Migrant Workers More at Risk of COVID-19 Infection",
                                "bbox": [85, 382, 412, 396],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "COVID-19 infection among migrants appears to be higher.",
                                "bbox": [113, 408, 528, 422],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "bbox": [140, 155, 524, 300],
                    "detection_source": "pymupdf_builtin",
                    "detection_method": "pymupdf_builtin",
                    "display_grid": [
                        ["331 335 319", None, None, None],
                        ["187", None, None, None],
                        [None, None, "22", "55"],
                    ],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# 1.5. Migrant Workers More at Risk of COVID-19 Infection", markdown)
        self.assertLess(markdown.index("331 335 319"), markdown.index("# 1.5. Migrant Workers"))
        self.assertLess(markdown.index("# 1.5. Migrant Workers"), markdown.index("COVID-19 infection"))

    def test_benchmark_projection_rejects_chart_legend_as_card_deck(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        def text_block(text: str, bbox: list[float]) -> dict[str, object]:
            return {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": text,
                "bbox": bbox,
            }

        blocks = [
            text_block("Figure 9.4.1: Challenges in importing amongst tourism MSMEs", [54, 48, 420, 62]),
            text_block("100", [64, 91, 82, 102]),
            text_block("80", [64, 122, 76, 133]),
            text_block("July 2020", [130, 244, 178, 255]),
            text_block("Big Challenge", [218.75, 258.13, 260.66, 267.32]),
            text_block("Small Challenge", [304.65, 258.13, 353.67, 267.32]),
            text_block("No Challenge", [398.54, 258.13, 439.33, 267.32]),
            text_block("There were very few tourism MSMEs that exported in each survey round.", [54, 288, 410, 300]),
        ]

        decks = projection._collect_infographic_card_deck_profiles(blocks)

        self.assertEqual([], decks)

    def test_benchmark_projection_rejects_formula_fragments_as_card_deck_lanes(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        def text_block(text: str, bbox: list[float]) -> dict[str, object]:
            return {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": text,
                "bbox": bbox,
            }

        blocks = [
            text_block("Probability, Combinatorics and Control", [75.12, 19.11, 221.81, 29.94]),
            text_block("lim PLL and", [174.33, 53.41, 263.25, 72.23]),
            text_block("PLH + PHL", [191.74, 67.01, 235.14, 87.16]),
            text_block("lim PLL + PHH", [274.90, 53.41, 335.59, 73.55]),
            text_block("(13)", [405.64, 60.31, 423.78, 72.23]),
            text_block("matrices, which make the computations considerably faster.", [75, 105, 310, 118]),
            text_block("However, it is worthwhile at this stage to note their implications.", [75, 123, 360, 136]),
            text_block("Summing up, both limits above can be used to argue in favor of time asymmetry.", [75, 148, 420, 161]),
        ]

        decks = projection._collect_infographic_card_deck_profiles(blocks)

        self.assertEqual([], decks)

    def test_benchmark_projection_rejects_overlapping_chart_axis_labels_as_card_deck_lanes(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        def text_block(text: str, bbox: list[float]) -> dict[str, object]:
            return {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": text,
                "bbox": bbox,
            }

        blocks = [
            text_block("Overview of OCR Pack", [54, 45, 230, 64]),
            text_block("100", [60, 105, 76, 116]),
            text_block("70.23", [80.98, 276.71, 94.59, 282.75]),
            text_block("Company Company", [76.58, 307.84, 131.78, 313.88]),
            text_block("A2 B2", [84.51, 315.29, 121.73, 321.60]),
            text_block("Company A", [658.79, 281.96, 686.36, 288.00]),
            text_block("Compan Compan Company B", [639.75, 282.00, 686.25, 310.50]),
            text_block("68.0", [552.81, 296.21, 563.27, 302.25]),
            text_block("Parsing-F1", [387.92, 296.95, 419.80, 304.21]),
            text_block("65 70 75 80 85 90 95 100", [450, 330, 700, 342]),
        ]

        decks = projection._collect_infographic_card_deck_profiles(blocks)

        self.assertEqual([], decks)

    def test_benchmark_projection_card_deck_support_uses_visual_order_not_source_order(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        def text_block(text: str, bbox: list[float]) -> dict[str, object]:
            return {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": text,
                "bbox": bbox,
            }

        blocks = [
            text_block("Comparison with Beauty Commerce", [60, 148, 240, 163]),
            text_block("Comparison Case of Domestic Subscription", [350, 148, 565, 163]),
            text_block("Education Content Platform PoC Case", [655, 148, 840, 163]),
            text_block("Recommendation Models", [60, 165, 190, 180]),
            text_block("Platform Recommendation Model", [350, 165, 520, 180]),
            text_block("Recommendation model Hit Ratio comparison", [60, 182, 250, 195]),
            text_block("Comparison of quantitative evaluations among", [350, 182, 545, 195]),
            text_block("baseline model and current service values", [60, 202, 252, 214]),
            text_block("personalized content recommendations", [350, 202, 545, 214]),
            text_block("0.03 0.06", [460, 237, 514, 248]),
            text_block("CustomerBERT", [370, 250, 424, 262]),
            text_block("AutoEncoder", [374, 306, 422, 318]),
            text_block("Comparison of prediction rates of correct/incorrect", [655, 165, 867, 178]),
            text_block("answers based on personalized questions", [655, 179, 825, 192]),
            text_block("0.882", [690, 282, 719, 297]),
            text_block("Compared to", [748, 341, 798, 352]),
        ]

        decks = projection._collect_infographic_card_deck_profiles(blocks)

        self.assertEqual(1, len(decks))
        self.assertEqual(
            [
                "Comparison with Beauty Commerce",
                "Comparison Case of Domestic Subscription",
                "Education Content Platform PoC Case",
            ],
            [projection._block_text(block) for block in decks[0]["anchor_blocks"][:3]],
        )

    def test_benchmark_projection_prefers_toc_raw_rows_when_sequence_merged_entries(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "toc-raw-rows.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "toc_blocks": [
                {
                    "toc_id": "toc_001",
                    "block_type": "toc",
                    "_raw_entries": [
                        {
                            "text": "Meeting the Relatives . . . . . . . . . . . . . . . . . . . .",
                            "page_locator": "37",
                            "row_index": 9,
                            "level": 1,
                        },
                        {
                            "text": "6. For the Love of Iran. . . . . . . . . . . . . . . . . . . . . . . . .41",
                            "page_locator": None,
                            "row_index": 10,
                            "level": 1,
                        },
                        {
                            "text": "MENDELIAN GENETICS, PROBABILITY, PEDIGREES AND CHI-SQUARE STATISTICS . 80",
                            "page_locator": None,
                            "row_index": 11,
                            "level": 1,
                        },
                    ],
                }
            ],
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq_001",
                    "title": "Contents",
                    "toc_ids": ["toc_001"],
                    "entries": [
                        {
                            "text": (
                                "Meeting the Relatives . . . . . . . . . . . . . . . . . . . . "
                                "6. For the Love of Iran. . . . . . . . . . . . . . . . . . . . . . . . .41"
                            ),
                            "page_locator": "37",
                        }
                    ],
                }
            ],
            "document_ast": {"pages": []},
            "table_asts": [],
            "image_blocks": [],
        }

        lines = projection._build_toc_projection_lines(document)

        self.assertTrue(any(re.fullmatch(r"Meeting the Relatives \.{10,} 37", line) for line in lines), lines)
        self.assertTrue(any(re.fullmatch(r"6\. For the Love of Iran \.{10,} 41", line) for line in lines), lines)
        self.assertFalse(any(". ." in line for line in lines if "Meeting the Relatives" in line or "For the Love of Iran" in line))
        self.assertIn(
            "MENDELIAN GENETICS, PROBABILITY, PEDIGREES AND CHI-SQUARE STATISTICS . 80",
            lines,
        )
        self.assertFalse(
            any("Meeting the Relatives" in line and "For the Love of Iran" in line for line in lines),
            lines,
        )

    def test_benchmark_projection_restores_toc_prefix_entries_from_text_evidence(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "toc-prefix-evidence.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "text": "\n".join(
                [
                    "Contents",
                    "Author's Note to the 2021 Edition ................................. ix",
                    "Foreword to the 2021 Edition .................................... xi",
                    "Foreword and Acknowledgements ................................. xv",
                    "1. A Fountain in the Square .................................... 1",
                    "2. The Lost Homeland ......................................... 5",
                ]
            ),
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq_001",
                    "title": "Contents",
                    "toc_ids": ["toc_001"],
                    "entries": [
                        {
                            "outline_index": "1.0",
                            "text": "A Fountain in the Square ........................",
                            "page_locator": "1",
                        },
                        {
                            "outline_index": "2.0",
                            "text": "The Lost Homeland ........................",
                            "page_locator": "5",
                        },
                    ],
                }
            ],
            "document_ast": {"pages": []},
            "table_asts": [],
            "image_blocks": [],
        }

        lines = projection._build_toc_projection_lines(document)

        self.assertLess(
            lines.index("Author's Note to the 2021 Edition ................................. ix"),
            lines.index("1. A Fountain in the Square ........................ 1"),
        )
        self.assertIn("Foreword to the 2021 Edition .................................... xi", lines)
        self.assertIn("Foreword and Acknowledgements ................................. xv", lines)

    def test_benchmark_projection_preserves_raw_toc_outline_index_as_entry_prefix(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "toc-outline-prefix.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "toc_blocks": [
                {
                    "toc_id": "toc_001",
                    "block_type": "toc",
                    "_raw_entries": [
                        {
                            "outline_index": "1.0",
                            "text": "A Fountain in the Square . . . . . . . . . . . . . . . . . . .",
                            "page_locator": "1",
                            "row_index": 1,
                            "level": 1,
                        },
                        {
                            "outline_index": "2.1",
                            "text": "Nested Topic . . . . . . . . . . . . . . . . . . .",
                            "page_locator": "5",
                            "row_index": 2,
                            "level": 2,
                        },
                    ],
                }
            ],
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq_001",
                    "title": "Contents",
                    "toc_ids": ["toc_001"],
                    "entries": [{"text": "Merged table of contents", "page_locator": "1"}],
                }
            ],
            "document_ast": {"pages": []},
            "table_asts": [],
            "image_blocks": [],
        }

        lines = projection._build_toc_projection_lines(document)

        self.assertTrue(any(re.fullmatch(r"1\. A Fountain in the Square \.{10,} 1", line) for line in lines), lines)
        self.assertTrue(any(re.fullmatch(r"2\.1 Nested Topic \.{10,} 5", line) for line in lines), lines)
        self.assertFalse(any(". ." in line for line in lines if "Fountain" in line or "Nested Topic" in line))

    def test_benchmark_projection_normalizes_spaced_toc_leaders(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        self.assertEqual(
            "Author's Note to the 2021 Edition ................................. ix",
            projection._normalize_toc_leader_spacing(
                "Author's Note to the 2021 Edition. . . . . . . . . . . . . . . . . ix"
            ),
        )

    def test_benchmark_projection_restores_toc_leaders_for_split_locator_entries(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "split-locator-toc.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "toc_blocks": [
                {
                    "toc_id": "toc_001",
                    "block_type": "toc",
                    "bbox": [30, 90, 580, 740],
                    "_raw_entries": [
                        {
                            "text": "Measurement Lab worksheet",
                            "page_locator": "3",
                            "source_kind": "text_block",
                            "row_index": 1,
                            "bbox": [34, 110, 581, 126],
                            "sort_x0": 34,
                            "sort_y0": 110,
                        },
                        {
                            "text": "How molecules move in a liquid",
                            "page_locator": "12",
                            "source_kind": "text_block",
                            "row_index": 2,
                            "bbox": [45, 132, 581, 148],
                            "sort_x0": 45,
                            "sort_y0": 132,
                        },
                    ],
                }
            ],
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq_001",
                    "title": "Table of Contents",
                    "toc_ids": ["toc_001"],
                    "entries": [
                        {"text": "Measurement Lab worksheet", "page_locator": "3"},
                        {"text": "How molecules move in a liquid", "page_locator": "12"},
                    ],
                }
            ],
            "document_ast": {"pages": []},
            "table_asts": [],
            "image_blocks": [],
        }

        lines = projection._build_toc_projection_lines(document)

        self.assertTrue(
            any(re.fullmatch(r"Measurement Lab worksheet\.{10,}\s+3", line) for line in lines),
            lines,
        )
        self.assertTrue(
            any(re.fullmatch(r"How molecules move in a liquid\.{10,}\s+12", line) for line in lines),
            lines,
        )

    def test_benchmark_projection_does_not_restore_toc_leaders_for_tabular_rows(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "tabular-toc.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "toc_blocks": [
                {
                    "toc_id": "toc_001",
                    "block_type": "toc",
                    "bbox": [56, 140, 386, 580],
                    "_raw_entries": [
                        {
                            "text": "Introduction",
                            "page_locator": "7",
                            "source_kind": "table_row",
                            "row_index": 1,
                            "bbox": [56, 140, 386, 152],
                        },
                        {
                            "text": "Changing Practices, Shifting Sites",
                            "page_locator": "7",
                            "source_kind": "table_row",
                            "row_index": 2,
                            "bbox": [56, 153, 386, 164],
                        },
                        {
                            "text": "Instructor Resources",
                            "page_locator": "97",
                            "source_kind": "text_block",
                            "row_index": 3,
                            "bbox": [85, 180, 342, 192],
                        },
                    ],
                }
            ],
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq_001",
                    "title": "Table of contents",
                    "toc_ids": ["toc_001"],
                    "entries": [
                        {"text": "Introduction", "page_locator": "7"},
                        {"text": "Changing Practices, Shifting Sites", "page_locator": "7"},
                        {"text": "Instructor Resources", "page_locator": "97"},
                    ],
                }
            ],
            "document_ast": {"pages": []},
            "table_asts": [],
            "image_blocks": [],
        }

        lines = projection._build_toc_projection_lines(document)

        self.assertIn("Introduction 7", lines)
        self.assertIn("Changing Practices, Shifting Sites 7", lines)
        self.assertIn("Instructor Resources 97", lines)
        self.assertFalse(any(re.search(r"\.{10,}", line) for line in lines), lines)

    def test_benchmark_projection_does_not_promote_figure_reference_fragment(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "figure-reference-fragment.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The Data Journey",
                                "bbox": [56.69, 57.49, 274.23, 86.75],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "To get started, let's consider the data visualization",
                                "bbox": [56.69, 111.56, 282.86, 122.54],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "1 in Figure 1.1",
                                "bbox": [282.86, 107.75, 341.67, 122.54],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 1.1.",
                                "bbox": [290.78, 142.47, 329.99, 152.23],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# The Data Journey", markdown)
        self.assertNotIn("# 1 in Figure 1.1", markdown)

    def test_benchmark_projection_does_not_promote_mid_paragraph_sentence_fragments(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "paragraph-fragment.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "EFB = empty fruit bunch.",
                                "bbox": [85.1, 86.75, 167.08, 94.79],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Source: Murdiyatmo (2021).",
                                "bbox": [85.1, 97.91, 178.6, 105.95],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "However, the main obstacle with producing second-generation bioethanol is the cost of",
                                "bbox": [85.1, 109.7, 484.42, 120.74],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "enzymes. Murdiyatmo (2021) stated that, at the pilot scale, the cost of enzymes is very",
                                "bbox": [85.1, 125.18, 484.19, 136.22],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "2.1. Diesel and biodiesel use",
                                "bbox": [85.1, 353.33, 225.63, 364.37],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("# 2.1. Diesel and biodiesel use", markdown)
        self.assertNotIn("# However, the main obstacle", markdown)

    def test_benchmark_projection_does_not_promote_standalone_figure_caption_as_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "figure-caption.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 6.1.1: Will they fire more staff in the next 2 months - across survey phases (%)",
                                "bbox": [94, 63, 473, 74],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "100 80 60 40 20 0",
                                "bbox": [94, 94, 210, 108],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "6.2. Expectations for Re-Hiring Employees",
                                "bbox": [94, 414, 340, 426],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Figure 6.1.1: Will they fire more staff in the next 2 months - across survey phases (%)\n\n"
            "100 80 60 40 20 0\n\n"
            "6.2. Expectations for Re-Hiring Employees\n",
            document,
        )

        self.assertNotIn("# Figure 6.1.1", markdown)
        self.assertIn("# 6.2. Expectations for Re-Hiring Employees", markdown)

    def test_benchmark_projection_does_not_promote_body_lead_in_sentence_before_figure(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "lead-in-sentence.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "The Scholarly Publishing Cycle",
                                "bbox": [54, 64, 310, 88],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Having explored the scholarly publishing ecosystem and its primary relationships, we",
                                "bbox": [54, 115, 558, 129],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "can update the cycle as follows:",
                                "bbox": [54, 132, 244, 146],
                            },
                            {
                                "block_type": "image",
                                "bbox": [70, 170, 520, 430],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "The Scholarly Publishing Cycle\n\n"
            "Having explored the scholarly publishing ecosystem and its primary relationships, we can update the cycle as follows:\n\n"
            "![Figure 1](#img_p1_001)\n",
            document,
        )

        self.assertIn("# The Scholarly Publishing Cycle", markdown)
        self.assertNotIn("# Having explored", markdown)

    def test_benchmark_projection_does_not_promote_page_label_or_funding_notice(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "funding-label.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Co-funded by European Union the",
                                "bbox": [406, 22, 554, 53],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3. RECOLLECTION OF NATIONAL INITIATIVES",
                                "bbox": [70, 90, 460, 108],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Partners were also asked to recollect initiatives from their respective countries.",
                                "bbox": [70, 130, 540, 146],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Co-funded by European Union the\n\n"
            "3. RECOLLECTION OF NATIONAL INITIATIVES\n\n"
            "Partners were also asked to recollect initiatives from their respective countries.\n",
            document,
        )

        self.assertNotIn("# Co-funded", markdown)
        self.assertIn("# 3. RECOLLECTION OF NATIONAL INITIATIVES", markdown)

    def test_benchmark_projection_splits_short_all_caps_heading_embedded_before_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "all-caps-embedded-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Figure 7.1: Texas OER landscape survey results show terms used in course schedules",
                                "bbox": [56, 310, 500, 326],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "IMPLEMENTATION",
                                "bbox": [56, 363, 166, 376],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Locally, we implemented a quick and free solution that reflects the constraints.",
                                "bbox": [56, 394, 520, 410],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Figure 7.1: Texas OER landscape survey results show terms used in course schedules\n\n"
            "IMPLEMENTATION Locally, we implemented a quick and free solution that reflects the constraints.\n",
            document,
        )

        self.assertIn("# IMPLEMENTATION", markdown)
        self.assertIn("Locally, we implemented", markdown)

    def test_benchmark_projection_rejects_body_lead_in_before_display_equation(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "equation-lead-in.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "To receive a better approximation the error estimate can be added to the approximation:",
                                "bbox": [90, 166, 476, 176],
                            },
                            {
                                "block_type": "formula",
                                "semantic_role": "display_formula",
                                "unit_role": "formula",
                                "text": "M = Q(h) + E(h)",
                                "bbox": [140, 190, 420, 214],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3.7.3 Formulae of higher accuracy from Richardson's extrapolation *",
                                "bbox": [90, 385, 466, 401],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "To receive a better approximation the error estimate can be added to the approximation:\n\n"
            "M = Q(h) + E(h)\n\n"
            "3.7.3 Formulae of higher accuracy from Richardson's extrapolation *\n",
            document,
        )

        self.assertNotIn("# To receive a better approximation", markdown)
        self.assertIn("# 3.7.3 Formulae of higher accuracy from Richardson's extrapolation *", markdown)

    def test_benchmark_projection_rejects_short_body_lines_around_display_equation(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "formula-context-body-lines.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Chapter 3. Numerical differentiation",
                                "bbox": [90, 73, 240, 83],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Note that the exact error equals",
                                "bbox": [90, 105, 228, 115],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "$M - Q(h) = -0.0342$",
                                "bbox": [210, 126, 393, 144],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "3.7.3 Formulae of higher accuracy from Richardson's extrapolation *",
                                "bbox": [90, 385, 466, 401],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Chapter 3. Numerical differentiation\n\n"
            "Note that the exact error equals\n\n"
            "$M - Q(h) = -0.0342$\n\n"
            "3.7.3 Formulae of higher accuracy from Richardson's extrapolation *\n",
            document,
        )

        self.assertNotIn("# Chapter 3. Numerical differentiation", markdown)
        self.assertNotIn("# Note that the exact error equals", markdown)
        self.assertIn("# 3.7.3 Formulae of higher accuracy from Richardson's extrapolation *", markdown)

    def test_benchmark_projection_splits_short_numbered_heading_from_body_prefix(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "numbered-heading-body-prefix.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "3 Training Details",
                                "bbox": [70, 515, 171, 527],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "After DUS, including continued pretraining, we trained the model.",
                                "bbox": [70, 537, 420, 549],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "4 Results",
                                "bbox": [70, 610, 140, 622],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "3 Training Details After DUS, including continued pretraining, we trained the model.\n\n"
            "4 Results\n",
            document,
        )

        self.assertIn("# 3 Training Details\n\nAfter DUS", markdown)
        self.assertNotIn("# 3 Training Details After DUS", markdown)
        self.assertIn("# 4 Results", markdown)

    def test_benchmark_projection_splits_explicit_short_heading_from_wrapped_body_sentence(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "explicit-heading-body-wrap.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "3 Training Details",
                                "bbox": [70, 515, 171, 527],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "After DUS, including continued pretraining, we",
                                "bbox": [70, 537, 289, 549],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "perform fine-tuning in two stages:",
                                "bbox": [70, 557, 290, 569],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "3 Training Details After DUS, including continued pretraining, we\n\n"
            "perform fine-tuning in two stages:\n",
            document,
        )

        self.assertIn("# 3 Training Details", markdown)
        self.assertNotIn("# 3 Training Details After DUS", markdown)

    def test_benchmark_projection_recovers_short_list_category_heading_midpage(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "midpage-list-category-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The previous paragraph closes with a full sentence.",
                                "bbox": [72, 330, 470, 344],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Trash",
                                "bbox": [72, 357, 99, 371],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "m. Waste Segregation and Segregated Bins. The program separates materials.",
                                "bbox": [72, 389, 480, 403],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "The previous paragraph closes with a full sentence. Trash\n\n"
            "m. Waste Segregation and Segregated Bins. The program separates materials.\n",
            document,
        )

        self.assertIn("# Trash", markdown)
        self.assertIn("m. Waste Segregation", markdown)

    def test_benchmark_projection_recovers_short_category_heading_before_lettered_item_even_after_wrapped_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "wrapped-body-category-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "packaging. Watermarking technology is also being developed so that packaging",
                                "bbox": [126, 310, 494, 323],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "can be more easily recognized by sorters.",
                                "bbox": [126, 326, 314, 339],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Trash",
                                "bbox": [72, 357, 99, 371],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "m. Waste Segregation and Segregated Bins. The program separates materials.",
                                "bbox": [72, 373, 480, 387],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "packaging. Watermarking technology is also being developed so that packaging "
            "can be more easily recognized by sorters. Trash\n\n"
            "m. Waste Segregation and Segregated Bins. The program separates materials.\n",
            document,
        )

        self.assertIn("# Trash", markdown)
        self.assertIn("can be more easily recognized by sorters.", markdown)
        self.assertIn("m. Waste Segregation", markdown)

    def test_benchmark_projection_recovers_short_list_category_heading_from_sentence_tail(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "tail-category-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Trash",
                                "bbox": [72, 357, 99, 371],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "m. Waste Segregation and Segregated Bins. The program separates materials.",
                                "bbox": [72, 373, 480, 387],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Packaging can be more easily recognized by sorters. Trash\n\n"
            "m. Waste Segregation and Segregated Bins. The program separates materials.\n",
            document,
        )

        self.assertIn("# Trash", markdown)
        self.assertIn("Packaging can be more easily recognized by sorters.", markdown)

    def test_benchmark_projection_rejects_bullet_and_numbered_list_items_as_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "list-items-not-headings.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "• Lime is recommended if pH < 5.8",
                                "bbox": [63, 58, 214, 69],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "Activity 5: Evaluating Liming Materials",
                                "bbox": [61, 554, 270, 566],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1. Label four plastic bags",
                                "bbox": [61, 599, 171, 611],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "with the name of one liming agent.",
                                "bbox": [84, 619, 260, 631],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "• Lime is recommended if pH < 5.8\n\n"
            "Activity 5: Evaluating Liming Materials\n\n"
            "1. Label four plastic bags\n\n"
            "with the name of one liming agent.\n",
            document,
        )

        self.assertNotIn("# • Lime is recommended", markdown)
        self.assertIn("# Activity 5: Evaluating Liming Materials", markdown)
        self.assertNotIn("# 1. Label four plastic bags", markdown)

    def test_benchmark_projection_rejects_numbered_step_sequence_despite_ast_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "numbered-step-sequence.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "1. Label four plastic bags",
                                "bbox": [61, 599, 171, 611],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "2. Weigh 20 g of air-dry soil into each plastic bag.",
                                "bbox": [61, 619, 342, 631],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "1. Label four plastic bags\n\n"
            "2. Weigh 20 g of air-dry soil into each plastic bag.\n",
            document,
        )

        self.assertNotIn("# 1. Label four plastic bags", markdown)
        self.assertIn("2. Weigh 20 g", markdown)

    def test_benchmark_projection_splits_running_header_prefix_from_true_title(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "running-header-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "8 Encinas Franco and Laguna",
                                "bbox": [44, 34, 177, 47],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Filipino Women in Electoral Politics",
                                "bbox": [45, 69, 251, 86],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The nature and extent of Filipino women's political participation is described.",
                                "bbox": [45, 100, 372, 114],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "8 Encinas Franco and Laguna Filipino Women in Electoral Politics\n\n"
            "The nature and extent of Filipino women's political participation is described.\n",
            document,
        )

        self.assertIn("# Filipino Women in Electoral Politics", markdown)
        self.assertNotIn("# 8 Encinas Franco and Laguna Filipino Women", markdown)

    def test_benchmark_projection_splits_running_header_and_title_from_merged_opening_paragraph(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "running-header-title-paragraph.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "8 Encinas Franco and Laguna",
                                "bbox": [45, 35, 177, 47],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Filipino Women in Electoral Politics",
                                "bbox": [45, 69, 251, 86],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The nature and extent of Filipino women's political participation is described.",
                                "bbox": [45, 100, 372, 114],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "8 Encinas Franco and Laguna Filipino Women in Electoral Politics "
            "The nature and extent of Filipino women's political participation is described.\n",
            document,
        )

        self.assertIn("# Filipino Women in Electoral Politics", markdown)
        self.assertIn("The nature and extent of Filipino women's political participation is described.", markdown)
        self.assertNotIn("# 8 Encinas Franco and Laguna", markdown)

    def test_benchmark_projection_combines_top_number_and_adjacent_title_when_body_follows(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "number-title-opener.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "2",
                                "bbox": [142, 54, 149, 64],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Fact-Checking",
                                "bbox": [57, 58, 142, 73],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "In this context, we are talking about fact-checking.",
                                "bbox": [80, 128, 260, 140],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "2\n\nFact-Checking\n\nIn this context, we are talking about fact-checking.\n",
            document,
        )

        self.assertIn("# 2 Fact-Checking", markdown)
        self.assertNotIn("\n# 2\n", markdown)
        self.assertNotIn("\n# Fact-Checking\n\n# 2", markdown)
        self.assertIn("In this context, we are talking about fact-checking.", markdown)

    def test_benchmark_projection_recovers_compact_survey_chart_heading_prefixes(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "survey-chart.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "height": None, "blocks": []}]},
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "EducationLevel 122responses Primary LowerSecondary UpperSecondary 76.2% "
            "Non-formal Training Bachelor'sDegreeorHigher Masterdegree Bac+5 18% Ph.D.\n\n"
            "Profession 122responses SocialEntrepreneur 19.7% YouthWorker Educator/Trainer",
            document,
        )

        self.assertIn("# Education Level 122 responses", markdown)
        self.assertIn("Primary LowerSecondary UpperSecondary", markdown)
        self.assertIn("# Profession 122 responses", markdown)
        self.assertNotIn("EducationLevel 122responses", markdown)

    def test_benchmark_projection_rejects_long_lowercase_sentence_heading_continuation(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        markdown = projection._merge_adjacent_heading_continuations(
            "# Recommendation pack shows outstanding performance of 1.7~2.6 times that of\n\n"
            "# competing models even when using commercial service data\n\n"
            "# Comparison with Beauty Commerce Recommendation Models\n"
        )

        self.assertIn("# Recommendation pack shows outstanding performance of 1.7~2.6 times that of", markdown)
        self.assertIn("# competing models even when using commercial service data", markdown)
        self.assertNotIn("that of competing models", markdown)

    def test_benchmark_projection_recovers_media_adjacent_section_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "media-adjacent-headings.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "MOHAVE COMMUNITY COLLEGE BIO181",
                                "bbox": [80.04, 34.74, 540.07, 55.91],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Cellular Replication",
                                "bbox": [244.2, 72.05, 371.4, 91.53],
                            },
                            {"block_type": "image", "image_id": "img_1", "bbox": [38.0, 101.85, 446.0, 354.85]},
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Growth and the Creation of Life",
                                "bbox": [72.0, 364.37, 303.14, 383.85],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "One of the characteristics of living things is the ability",
                                "bbox": [78.96, 390.76, 388.07, 410.3],
                            },
                            {"block_type": "image", "image_id": "img_2", "bbox": [473.0, 488.4, 552.95, 568.35]},
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Mitosis and",
                                "bbox": [462.0, 575.11, 533.52, 589.33],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Meiosis",
                                "bbox": [461.99, 588.07, 506.75, 602.28],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Similiar processes with VERY different results!",
                                "bbox": [462.0, 608.24, 555.3, 625.74],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "MOHAVE COMMUNITY COLLEGE BIO181\n\n"
            "Cellular Replication\n\n"
            "![Figure 1](#img_1)\n\n"
            "Growth and the Creation of Life One of the characteristics of living things is the ability\n\n"
            "![Figure 2](#img_2)\n\n"
            "Mitosis and Meiosis Similiar processes with VERY different results!\n",
            document,
        )

        self.assertIn("# Cellular Replication", markdown)
        self.assertIn("# Growth and the Creation of Life", markdown)
        self.assertIn("# Mitosis and Meiosis", markdown)
        self.assertIn("One of the characteristics of living things is the ability", markdown)
        self.assertIn("Similiar processes with VERY different results!", markdown)

    def test_benchmark_projection_rejects_post_media_chart_legend_cluster_as_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "post-media-chart-labels.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {"block_type": "image", "image_id": "chart", "bbox": [40, 100, 300, 240]},
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Company Company A2 B2",
                                "bbox": [50, 250, 190, 264],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Scene (Photographed document image) Document (Scanned document image)",
                                "bbox": [50, 270, 420, 284],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "![Figure 1](#chart)\n\n"
            "Company Company A2 B2\n\n"
            "Scene (Photographed document image) Document (Scanned document image)\n",
            document,
        )

        self.assertNotIn("# Company Company A2 B2", markdown)
        self.assertIn("Company Company A2 B2", markdown)

    def test_benchmark_projection_rejects_thin_separator_as_post_media_heading_anchor(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "separator-before-running-chapter.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "rule",
                                "bbox": [90.0, 84.0, 513.0, 84.5],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Chapter 3. Numerical differentiation",
                                "bbox": [90.0, 73.2, 240.0, 83.3],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Note that the exact error equals",
                                "bbox": [90.0, 104.8, 228.2, 114.9],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "![Figure 1](#rule)\n\n"
            "Chapter 3. Numerical differentiation Note that the exact error equals\n",
            document,
        )

        self.assertNotIn("# Chapter 3. Numerical differentiation", markdown)
        self.assertIn("Chapter 3. Numerical differentiation Note that the exact error equals", markdown)

    def test_benchmark_projection_rejects_wide_top_page_number_running_header_pair(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "running-header-page-label.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "314",
                                "bbox": [62, 32, 75, 48],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Yarrow",
                                "bbox": [355, 33, 388, 47],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "This construction follows from the observer model.",
                                "bbox": [62, 92, 360, 108],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "7 Variants of sj Observer Models",
                                "bbox": [62, 142, 292, 161],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The model variants differ in their state representation.",
                                "bbox": [62, 178, 382, 194],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "314\n\nYarrow\n\nThis construction follows from the observer model.\n\n"
            "7 Variants of sj Observer Models\n\nThe model variants differ in their state representation.\n",
            document,
        )

        self.assertNotIn("# 314 Yarrow", markdown)
        self.assertNotIn("# 314", markdown)
        self.assertIn("# 7 Variants of sj Observer Models", markdown)

    def test_benchmark_projection_recovers_top_multiline_title_before_author_metadata(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "paper-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective",
                                "bbox": [73.17, 69.74, 522.11, 84.09],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "publication_masthead",
                                "unit_role": "metadata",
                                "text": "Depth Up-Scaling",
                                "bbox": [243.24, 85.68, 352.04, 100.03],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "author_line",
                                "unit_role": "metadata",
                                "text": "Dahyun Kim, Chanjun Park, Sanghoon Kim",
                                "bbox": [97.56, 106.77, 500.7, 120.22],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Upstage AI, South Korea",
                                "bbox": [238.91, 178.22, 359.35, 190.18],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "{kdahyun, chanjun.park,limerobot}@upstage.ai",
                                "bbox": [43.86, 192.4, 554.41, 203.31],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Abstract",
                                "bbox": [157.76, 214.38, 202.24, 226.34],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "We introduce the model and method.",
                                "bbox": [90.82, 237.32, 272.13, 247.18],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective\n\n"
            "Dahyun Kim, Chanjun Park, Sanghoon Kim\n\n"
            "Upstage AI, South Korea\n\n"
            "{kdahyun, chanjun.park,limerobot}@upstage.ai\n\n"
            "Abstract\n",
            document,
        )

        self.assertIn(
            "# SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling",
            markdown,
        )
        self.assertNotIn("# Dahyun Kim", markdown)
        self.assertNotIn("# Upstage AI", markdown)

    def test_benchmark_projection_rejects_affiliation_line_between_authors_and_contact_as_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        blocks = [
            {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": "SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective",
                "bbox": [73.17, 69.74, 522.11, 84.09],
            },
            {
                "block_type": "text",
                "semantic_role": "publication_masthead",
                "unit_role": "metadata",
                "text": "Depth Up-Scaling",
                "bbox": [243.24, 85.68, 352.04, 100.03],
            },
            {
                "block_type": "text",
                "semantic_role": "author_line",
                "unit_role": "metadata",
                "text": "Dahyun Kim, Chanjun Park, Sanghoon Kim",
                "bbox": [97.56, 106.77, 500.7, 120.22],
            },
            {
                "block_type": "text",
                "semantic_role": "author_line",
                "unit_role": "metadata",
                "text": "Mikyoung Cha, Hwalsuk Lee, Sunghun Kim",
                "bbox": [181.67, 148.72, 416.09, 162.17],
            },
            {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": "Upstage AI, South Korea",
                "bbox": [238.91, 178.22, 359.35, 190.18],
            },
            {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": "{kdahyun, chanjun.park,limerobot}@upstage.ai",
                "bbox": [43.86, 192.4, 554.41, 203.31],
            },
            {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": "Abstract",
                "bbox": [157.76, 214.38, 202.24, 226.34],
            },
            {
                "block_type": "text",
                "semantic_role": "text_block",
                "unit_role": "body",
                "text": "We introduce the model and method.",
                "bbox": [90.82, 237.32, 272.13, 247.18],
            },
        ]

        headings = projection._collect_benchmark_heading_texts(
            {"document_ast": {"pages": [{"page": 1, "height": 792, "blocks": blocks}]}}
        )

        self.assertIn("SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling", headings)
        self.assertIn("Abstract", headings)
        self.assertNotIn("Upstage AI, South Korea", headings)

    def test_benchmark_projection_rejects_reference_and_contribution_wrapped_lines_as_headings(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        self.assertTrue(
            projection._looks_like_body_group_false_positive(
                "Khawlah M. Manna, Al-Sadu in Qatar: Traditional Tech- nical Values and Techniques (Doha: Qatar Museums",
                [
                    {"block_type": "text", "text": "Khawlah M. Manna, Al-Sadu in Qatar: Traditional Tech-", "bbox": [303, 629, 490, 642]},
                    {"block_type": "text", "text": "nical Values and Techniques (Doha: Qatar Museums", "bbox": [303, 641, 490, 654]},
                ],
            )
        )
        self.assertTrue(
            projection._looks_like_body_group_false_positive(
                "Hyeonwoo Kim. Chanjun Park led the Data and Evaluation (Data-Centric LLM) part, with Yungi",
                [
                    {"block_type": "text", "text": "Hyeonwoo Kim. Chanjun Park led the Data and", "bbox": [70, 496, 289, 507]},
                    {"block_type": "text", "text": "Evaluation (Data-Centric LLM) part, with Yungi", "bbox": [70, 510, 289, 521]},
                ],
            )
        )

    def test_benchmark_projection_recovers_cover_title_split_before_metadata(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "cover-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "LAW LIBRARY LIBRARY OF CONGRESS",
                                "bbox": [72.0, 72.0, 174.4, 170.23],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Restrictions on Land Ownership",
                                "bbox": [111.06, 266.51, 503.84, 301.8],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "by Foreigners in Selected",
                                "bbox": [154.32, 297.9, 460.55, 333.18],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Jurisdictions",
                                "bbox": [228.36, 329.21, 380.02, 364.5],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "June 2023",
                                "bbox": [274.14, 391.32, 341.61, 411.52],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "LAW LIBRARY LIBRARY OF CONGRESS\n\n"
            "Restrictions on Land Ownership\n\n"
            "by Foreigners in Selected\n\n"
            "Jurisdictions\n\n"
            "June 2023\n",
            document,
        )

        self.assertIn(
            "# Restrictions on Land Ownership by Foreigners in Selected Jurisdictions",
            markdown,
        )
        self.assertNotIn("# LAW LIBRARY", markdown)

    def test_benchmark_projection_recovers_multi_column_card_headings_with_supporting_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "multi-card-headings.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": None,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Our Purpose",
                                "bbox": [69, 328, 136, 344],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Our Mission",
                                "bbox": [313, 328, 377, 344],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Making AI Beneficial",
                                "bbox": [69, 363, 218, 385],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Easy-to-apply AI,",
                                "bbox": [313, 363, 435, 385],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "What We Do",
                                "bbox": [545, 328, 613, 344],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Providing the world's best and easy-to-use AI solutions for everyone",
                                "bbox": [545, 363, 860, 385],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "Our Purpose\n\nOur Mission\n\nMaking AI Beneficial Easy-to-apply AI,\n\n"
            "What We Do Providing the world's best and easy-to-use AI solutions for everyone\n",
            document,
        )

        self.assertIn("# Our Purpose", markdown)
        self.assertIn("# Our Mission", markdown)
        self.assertIn("# What We Do", markdown)
        self.assertIn("Making AI Beneficial", markdown)
        self.assertIn("Providing the world's best", markdown)

    def test_benchmark_projection_merges_adjacent_long_heading_lines_when_ast_confirms_wrap(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "wrapped-long-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "Recommendation pack shows outstanding performance of 1.7~2.6 times that of",
                                "bbox": [65, 80, 620, 98],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "competing models even when using commercial service data",
                                "bbox": [65, 101, 540, 119],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Recommendation model Hit Ratio comparison",
                                "bbox": [65, 150, 340, 166],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }
        markdown = (
            "# Recommendation pack shows outstanding performance of 1.7~2.6 times that of\n\n"
            "# competing models even when using commercial service data\n\n"
            "Comparison with Beauty Commerce Recommendation Models\n"
        )

        projected = projection.project_benchmark_headings(markdown, document)

        self.assertIn(
            "# Recommendation pack shows outstanding performance of 1.7~2.6 times that of competing models even when using commercial service data",
            projected,
        )
        self.assertNotIn("# competing models even", projected)

    def test_benchmark_projection_splits_numbered_heading_embedded_after_lowercase_ocr_body(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "embedded-numbered-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "The model discussion continues.",
                                "bbox": [62, 96, 410, 110],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "11 Dual-Presentation SJ Data",
                                "bbox": [62, 150, 286, 168],
                            },
                            {
                                "block_type": "text",
                                "semantic_role": "text_block",
                                "unit_role": "body",
                                "text": "Several authors have investigated the use of a dual-presentation SJ task.",
                                "bbox": [62, 185, 490, 199],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "the multinomial distribution can provide an exact likelihood. "
            "11 Dual-Presentation sj Data Several authors have investigated the use of a dual-presentation SJ task.\n",
            document,
        )

        self.assertIn("the multinomial distribution can provide an exact likelihood.", markdown)
        self.assertIn("# 11 Dual-Presentation SJ Data", markdown)
        self.assertIn("Several authors have investigated", markdown)

    def test_benchmark_projection_rejects_footnote_boundary_before_single_word_heading(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "single-word-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "TEXAS",
                                "bbox": [72, 160, 130, 178],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "2019. Austin, TX: Digital Higher Education Consortium of Texas and Texas Higher Education Coordinating Board; "
            "Half Moon Bay, CA: Institute.22 TEXAS and Texas Higher Education Coordinating Board.\n",
            document,
        )

        self.assertNotIn("# TEXAS", markdown)
        self.assertIn("Institute.22 TEXAS and Texas", markdown)

    def test_benchmark_projection_rejects_single_word_heading_inside_reference_phrase(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "reference-phrase-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "TEXAS",
                                "bbox": [72, 160, 130, 178],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "2019. Austin, TX: Digital Higher Education Consortium of Texas and Texas Higher Education Coordinating Board; "
            "Half Moon Bay,\n",
            document,
        )

        self.assertNotIn("# TEXAS", markdown)
        self.assertIn("Consortium of Texas and Texas", markdown)

    def test_benchmark_projection_rejects_short_heading_inside_actor_list(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "actor-list-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "semantic_role": "section_heading",
                                "unit_role": "section_heading",
                                "text": "Publishers",
                                "bbox": [170, 120, 240, 138],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
        }

        markdown = projection.project_benchmark_headings(
            "READERS PUBLISHERS AGGREGATORS LIBRARIANS\n",
            document,
        )

        self.assertNotIn("# Publishers", markdown)
        self.assertIn("READERS PUBLISHERS AGGREGATORS LIBRARIANS", markdown)

    def test_benchmark_projection_demotes_chart_owned_diagram_tables_to_plain_figure_text(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "diagram-chart.pdf",
            "source_type": "pdf",
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "image",
                                "image_id": "img_1",
                                "image_kind_guess": "chart_figure",
                                "bbox": [90, 100, 500, 360],
                                "figure_semantics": {"semantic_type": "chart_figure"},
                                "content_segments": [
                                    {
                                        "role": "embedded_text",
                                        "text": "35 31 30 25 23 20 Event Celebration Information Videograph",
                                    }
                                ],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_1",
                                "bbox": [112, 380, 485, 414],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_2",
                                "bbox": [114, 414, 486, 700],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_1",
                    "page": 1,
                    "bbox": [112, 380, 485, 414],
                    "detection_source": "pymupdf_builtin",
                    "display_grid": [["Diagram 5", "Distribution of Komnas HAM's YouTube Content (2019-2020)"]],
                },
                {
                    "table_id": "tbl_2",
                    "page": 1,
                    "bbox": [114, 414, 486, 700],
                    "detection_source": "structured_text_region",
                    "table_family": "two_column_inventory",
                    "display_grid": [
                        ["16 (7%)", "Government"],
                        ["7 (3%)", ""],
                        ["", "Civil Society Organizations"],
                        ["90 (37%)", "Other Institutions"],
                        ["Diagram 6", "Distribution of participating institutions"],
                    ],
                },
            ],
            "image_blocks": [
                {
                    "image_id": "img_1",
                    "page": 1,
                    "bbox": [90, 100, 500, 360],
                    "image_kind_guess": "chart_figure",
                    "figure_semantics": {"semantic_type": "chart_figure"},
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": "35 31 30 25 23 20 Event Celebration Information Videograph",
                        }
                    ],
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("| Diagram 5 |", markdown)
        self.assertNotIn("| 16 (7%) | Government |", markdown)
        self.assertIn("Diagram 5", markdown)
        self.assertIn("Distribution of Komnas HAM's YouTube Content (2019-2020)", markdown)
        self.assertIn("Civil Society Organizations", markdown)

    def test_benchmark_projection_demotes_visual_heading_card_table_to_heading_flow(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "visual-heading-card.pdf",
            "source_type": "pdf",
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "table",
                                "table_id": "tbl_card",
                                "bbox": [34, 26, 690, 155],
                            },
                            {
                                "block_type": "text",
                                "text": "100 95 90 85 80",
                                "bbox": [60, 180, 500, 196],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_card",
                    "page": 1,
                    "bbox": [34, 26, 690, 155],
                    "detection_source": "structured_text_region",
                    "title": "Overview of OCR Pack",
                    "display_grid": [
                        ["Overview of OCR Pack", ""],
                        ["Base Model Performance", "Evaluation of Upstage OCR Pack"],
                        [
                            "Upstage universal OCR model E2E performance",
                            "Upstage universal OCR model performance details: Document",
                        ],
                        ["evaluation1", "criteria"],
                    ],
                    "semantic_grid": [
                        ["Overview of OCR Pack", ""],
                        ["Base Model Performance", "Evaluation of Upstage OCR Pack"],
                        [
                            "Upstage universal OCR model E2E performance",
                            "Upstage universal OCR model performance details: Document",
                        ],
                        ["evaluation1", "criteria"],
                    ],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("<table>", markdown)
        self.assertIn("# Overview of OCR Pack", markdown)
        self.assertIn("# Base Model Performance Evaluation of Upstage OCR Pack", markdown)
        self.assertIn("# Upstage universal OCR model E2E performance evaluation1", markdown)
        self.assertIn("# Upstage universal OCR model performance details: Document criteria", markdown)
        self.assertIn("100 95 90 85 80", markdown)

    def test_benchmark_projection_demotes_top_visual_heading_card_with_title_row_index(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "visual-heading-card-title-row.pdf",
            "source_type": "pdf",
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "table",
                                "table_id": "tbl_card",
                                "bbox": [34, 26, 690, 155],
                                "title": "Overview of OCR Pack",
                                "title_row_index": 0,
                                "display_grid": [
                                    ["Overview of OCR Pack", ""],
                                    ["Base Model Performance", "Evaluation of Upstage OCR Pack"],
                                    [
                                        "Upstage universal OCR model E2E performance",
                                        "Upstage universal OCR model performance details: Document",
                                    ],
                                    ["evaluation1", "criteria"],
                                ],
                            },
                            {
                                "block_type": "image",
                                "image_id": "img_chart",
                                "bbox": [420, 166, 625, 320],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_card",
                    "page": 1,
                    "bbox": [34, 26, 690, 155],
                    "title": "Overview of OCR Pack",
                    "title_row_index": 0,
                    "display_grid": [
                        ["Overview of OCR Pack", ""],
                        ["Base Model Performance", "Evaluation of Upstage OCR Pack"],
                        [
                            "Upstage universal OCR model E2E performance",
                            "Upstage universal OCR model performance details: Document",
                        ],
                        ["evaluation1", "criteria"],
                    ],
                }
            ],
            "image_blocks": [{"image_id": "img_chart", "bbox": [420, 166, 625, 320]}],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("<table>", markdown)
        self.assertIn("# Overview of OCR Pack", markdown)
        self.assertIn("# Base Model Performance Evaluation of Upstage OCR Pack", markdown)
        self.assertIn("# Upstage universal OCR model E2E performance evaluation1", markdown)
        self.assertIn("# Upstage universal OCR model performance details: Document criteria", markdown)

    def test_benchmark_projection_projects_embedded_figure_text_placeholders(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "embedded-figure-text.pdf",
            "source_type": "pdf",
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "text": "Body before the visual.",
                                "bbox": [56, 80, 300, 96],
                            },
                            {
                                "block_type": "image",
                                "image_id": "img_chart",
                                "image_kind_guess": "chart_figure",
                                "bbox": [70, 120, 520, 360],
                            },
                            {
                                "block_type": "text",
                                "text": "Body after the visual.",
                                "bbox": [56, 390, 300, 406],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [
                {
                    "image_id": "img_chart",
                    "page": 1,
                    "bbox": [70, 120, 520, 360],
                    "image_kind_guess": "chart_figure",
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": "Q11 Convenience Reading experience Workflow Habit 20% 40% 60% 80%",
                        }
                    ],
                }
            ],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("![Figure", markdown)
        self.assertIn("Body before the visual.", markdown)
        self.assertIn("Q11 Convenience Reading experience Workflow Habit", markdown)
        self.assertIn("Body after the visual.", markdown)

    def test_benchmark_projection_demotes_embedded_chart_ocr_grid_without_caption_match(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "embedded-chart-grid.pdf",
            "source_type": "pdf",
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "text",
                                "text": "Figure 4.1. Approved Capacity under the FIT Scheme",
                                "bbox": [72, 120, 360, 138],
                            },
                            {
                                "block_type": "image",
                                "image_id": "img_chart",
                                "bbox": [72, 150, 520, 360],
                            },
                            {
                                "block_type": "table",
                                "table_id": "tbl_chart",
                                "bbox": [90, 180, 460, 320],
                                "display_grid": [
                                    ["MW 700", None],
                                    [None, "Waste materials"],
                                    ["600", None],
                                    ["500", "Biogas"],
                                    ["400", "Construction wood waste"],
                                    ["300", "General wood"],
                                    ["200", "General wood (<10MW)"],
                                    ["100", "Unutilised wood"],
                                    ["0", "Unutilised wood (<2MW)"],
                                ],
                            },
                            {
                                "block_type": "text",
                                "text": "FIT = feed-in-tariff.",
                                "bbox": [72, 390, 220, 406],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_chart",
                    "bbox": [90, 180, 460, 320],
                    "detection_source": "embedded_image_ocr",
                    "display_grid": [
                        ["MW 700", None],
                        [None, "Waste materials"],
                        ["600", None],
                        ["500", "Biogas"],
                        ["400", "Construction wood waste"],
                        ["300", "General wood"],
                        ["200", "General wood (<10MW)"],
                        ["100", "Unutilised wood"],
                        ["0", "Unutilised wood (<2MW)"],
                    ],
                }
            ],
            "image_blocks": [{"image_id": "img_chart", "bbox": [72, 150, 520, 360]}],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("| MW 700 |", markdown)
        self.assertIn("Figure 4.1. Approved Capacity under the FIT Scheme", markdown)
        self.assertIn("MW 700", markdown)
        self.assertIn("Waste materials", markdown)
        self.assertIn("FIT = feed-in-tariff.", markdown)

    def test_benchmark_projection_demotes_semantic_visual_heading_card_without_table_invariants(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "semantic-visual-heading-card.pdf",
            "source_type": "pdf",
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "table",
                                "table_id": "tbl_card",
                                "bbox": [34, 26, 690, 155],
                            },
                            {
                                "block_type": "text",
                                "text": "100 95 90 85 80",
                                "bbox": [60, 180, 500, 196],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_card",
                    "page": 1,
                    "bbox": [34, 26, 690, 155],
                    "detection_source": "structured_text_region",
                    "title": "Overview of OCR Pack",
                    "display_grid": [
                        ["Overview of OCR Pack", ""],
                        ["Base Model Performance", "Evaluation of Upstage OCR Pack"],
                        [
                            "Upstage universal OCR model E2E performance",
                            "Upstage universal OCR model performance details: Document",
                        ],
                        ["evaluation1", "criteria"],
                    ],
                    "semantic_projection_v2": {
                        "table_family": "comparison_matrix",
                        "source": "table_semantic_projection_v2",
                    },
                    "logical_cells": [
                        {"row": 0, "col": 0, "rowspan": 1, "colspan": 2, "text": "Overview of OCR Pack"},
                        {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "text": "Base Model Performance"},
                        {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "Evaluation of Upstage OCR Pack"},
                    ],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertNotIn("<table>", markdown)
        self.assertIn("# Overview of OCR Pack", markdown)
        self.assertIn("# Base Model Performance Evaluation of Upstage OCR Pack", markdown)
        self.assertIn("# Upstage universal OCR model E2E performance evaluation1", markdown)

    def test_benchmark_projection_preserves_captioned_two_column_spanning_header_table(self) -> None:
        projection = importlib.import_module("scripts.opendataloader_benchmark_projection")

        document = {
            "filename": "captioned-two-column-table.pdf",
            "source_type": "pdf",
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 720,
                        "blocks": [
                            {
                                "block_type": "table",
                                "table_id": "tbl_species",
                                "bbox": [56, 56, 228, 140],
                            },
                            {
                                "block_type": "text",
                                "text": 'Table 6.1: Four fish species on IUCN Red List "Extinct in the Wild".',
                                "bbox": [56, 154, 366, 164],
                            },
                            {
                                "block_type": "text",
                                "text": "Body paragraph follows the caption.",
                                "bbox": [56, 199, 300, 211],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_species",
                    "page": 1,
                    "bbox": [56, 56, 228, 140],
                    "detection_source": "structured_text_region",
                    "table_family": "two_column_spanning_header_table",
                    "display_grid": [
                        ["Fish species", "on IUCN Red List"],
                        ["Potosi Pupfish", "Cyprinodon alvarezi"],
                        ["La Palma Pupfish", "Cyprinodon longidorsalis"],
                        ["Butterfly Splitfin", "Ameca splendens"],
                        ["Golden Skiffia", "Skiffia francesae"],
                    ],
                    "semantic_grid": [
                        ["Fish species on IUCN Red List", None],
                        ["Potosi Pupfish", "Cyprinodon alvarezi"],
                        ["La Palma Pupfish", "Cyprinodon longidorsalis"],
                        ["Butterfly Splitfin", "Ameca splendens"],
                        ["Golden Skiffia", "Skiffia francesae"],
                    ],
                    "semantic_projection_v2": {
                        "table_family": "two_column_spanning_header_table",
                        "source": "table_semantic_projection_v2",
                    },
                    "logical_cells": [
                        {"row": 0, "col": 0, "rowspan": 1, "colspan": 2, "text": "Fish species on IUCN Red List"},
                        {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "text": "Potosi Pupfish"},
                        {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "Cyprinodon alvarezi"},
                    ],
                }
            ],
            "image_blocks": [],
        }

        markdown = projection.build_opendataloader_benchmark_markdown(document)

        self.assertIn("<table>", markdown)
        self.assertIn("colspan=\"2\">Fish species on IUCN Red List", markdown)
        self.assertNotIn("# Potosi Pupfish Cyprinodon alvarezi", markdown)
        self.assertIn("Table 6.1: Four fish species", markdown)


if __name__ == "__main__":
    unittest.main()
