from __future__ import annotations

import importlib
import unittest


class SemanticTableMarkdownExportTests(unittest.TestCase):
    def test_default_table_export_keeps_pipe_table_for_semantic_spans(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = _semantic_span_table_document()

        markdown = "\n".join(api_main._build_document_body_markdown_sections(document, embed_images=False))

        self.assertIn("| Tissue | 0.5 h | 1 h |", markdown)
        self.assertIn("| Plasma | 1.0 | 2.0 |", markdown)
        self.assertNotIn("<table>", markdown)
        self.assertNotIn("rowspan=", markdown)
        self.assertNotIn("colspan=", markdown)

    def test_semantic_html_table_export_renders_logical_spans(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = _semantic_span_table_document()

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                document,
                embed_images=False,
                table_export_mode="semantic_html",
            )
        )

        self.assertIn("<table>", markdown)
        self.assertIn('<th rowspan="2">Tissue</th>', markdown)
        self.assertIn('<th colspan="2">Concentration (ng eq*/g)</th>', markdown)
        self.assertIn("<th>0.5 h</th>", markdown)
        self.assertIn("<td>Plasma</td>", markdown)
        self.assertIn("<td>2.0</td>", markdown)
        self.assertNotIn("| Tissue | 0.5 h | 1 h |", markdown)

    def test_auto_semantic_table_export_renders_logical_spans_without_affecting_plain_tables(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        span_markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _semantic_span_table_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )
        plain_markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _plain_semantic_table_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("<table>", span_markdown)
        self.assertIn('<th rowspan="2">Tissue</th>', span_markdown)
        self.assertIn('<th colspan="2">Concentration (ng eq*/g)</th>', span_markdown)
        self.assertIn("| A | B |", plain_markdown)
        self.assertNotIn("<table>", plain_markdown)

    def test_auto_semantic_table_export_ignores_body_rowspan_repairs(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _body_rowspan_repair_table_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("| Jurisdiction | Description |", markdown)
        self.assertIn("| Finland | Long description part 1 |", markdown)
        self.assertNotIn("<table>", markdown)
        self.assertNotIn("rowspan=", markdown)

    def test_semantic_html_table_export_ignores_unsafe_body_rowspan_repairs(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _body_rowspan_repair_table_document(),
                embed_images=False,
                table_export_mode="semantic_html",
            )
        )

        self.assertIn("| Jurisdiction | Description |", markdown)
        self.assertIn("| Finland | Long description part 1 |", markdown)
        self.assertIn("|  | Long description part 2 |", markdown)
        self.assertNotIn("<table>", markdown)
        self.assertNotIn("rowspan=", markdown)

    def test_auto_semantic_table_export_keeps_rowspans_from_structured_table_sources(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _structured_source_rowspan_table_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("<table>", markdown)
        self.assertIn('<td rowspan="2">Dose group</td>', markdown)
        self.assertNotIn("| Dose group | Finding |", markdown)

    def test_semantic_html_ignores_stale_body_spans_that_conflict_with_semantic_grid(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _stale_body_span_after_semantic_projection_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("<table>", markdown)
        self.assertIn("<td>2012</td>", markdown)
        self.assertIn("<td>10%</td>", markdown)
        self.assertIn("<td>7%</td>", markdown)
        self.assertNotIn("rowspan=", markdown)
        self.assertNotIn("<td>portfolio</td>", markdown)

    def test_stacked_numeric_schedule_projection_exports_expanded_semantic_grid(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _stacked_numeric_schedule_projection_document(),
                embed_images=False,
            )
        )

        self.assertIn("| 6 |  | 5.76% | 8.93% |", markdown)
        self.assertIn("| 7 |  |  | 8.93% |", markdown)
        self.assertIn("| 8 |  |  | 4.46% |", markdown)
        self.assertNotIn("| 6 7 8 |", markdown)

    def test_semantic_html_keeps_pipe_table_for_non_span_semantic_grid_by_default(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = _plain_semantic_table_document()

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                document,
                embed_images=False,
                table_export_mode="semantic_html",
            )
        )

        self.assertIn("| A | B |", markdown)
        self.assertNotIn("<table>", markdown)

    def test_projected_stub_matrix_semantic_html_renders_without_span_metadata(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = _projected_stub_table_document()

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                document,
                embed_images=False,
                table_export_mode="semantic_html",
            )
        )

        self.assertIn("<table>", markdown)
        self.assertIn("<th></th>", markdown)
        self.assertIn("<th>OCR</th>", markdown)
        self.assertIn("<td>Pack</td>", markdown)
        self.assertIn("<td>Recognizes text</td>", markdown)

    def test_semantic_projection_compaction_exports_semantic_grid_as_pipe_table(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = _parallel_inventory_compacted_document()

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                document,
                embed_images=False,
                table_export_mode="semantic_html",
            )
        )

        self.assertIn("| Reagents | Supplies and Equipment |", markdown)
        self.assertIn("Sample DNA* Evidence A DNA*", markdown)
        self.assertIn("Tube rack Pipet tips Water bath", markdown)
        self.assertNotIn("| Sample DNA* | Pipet tips |", markdown)

    def test_blank_form_semantic_projection_exports_display_columns_not_internal_stub_column(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _blank_form_projection_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("| Added cation | Relative Size & Settling Rates of Floccules |", markdown)
        self.assertIn("| K+ |  |", markdown)
        self.assertNotIn("|  | Added cation | Relative Size & Settling Rates of Floccules |", markdown)

    def test_blank_form_semantic_projection_preserves_visible_stub_column_when_present_in_display_grid(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = _blank_form_projection_document()
        table = document["table_asts"][0]
        table["display_grid"] = [
            [None, "Mitosis", "Meiosis"],
            ["# chromosomes in parent cells", None, None],
            ["# DNA replications", None, None],
        ]
        table["semantic_grid"] = [
            [None, "Mitosis", "Meiosis"],
            ["# chromosomes in parent cells", None, None],
            ["# DNA replications", None, None],
        ]
        table["header_row_groups"] = [
            {"start_row": 0, "end_row": 1, "source": "adjacent_text_block_header_projection"},
        ]

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                document,
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("|  | Mitosis | Meiosis |", markdown)
        self.assertIn("| # chromosomes in parent cells |  |  |", markdown)

    def test_missing_leading_stub_projection_exports_projected_semantic_grid(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _missing_leading_stub_projection_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("|  | Foreign | Domestic |", markdown)
        self.assertIn("| MANILA | 2454 | 6,125 |", markdown)
        self.assertNotIn("| Foreign |  | Domestic |", markdown)

    def test_structural_header_duplicate_in_data_grid_does_not_render_as_body_row(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _structural_header_duplicate_data_grid_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertEqual(markdown.count("| Government | No. of Seats | Aquino | Ramos |"), 1)
        self.assertIn("| Senate | 24 | 8.3 | 16.7 |", markdown)
        self.assertNotIn("| Government | No. of Seats | Aquino | Ramos |\n| Senate |", markdown)

    def test_two_column_spanning_header_table_auto_exports_logical_colspan(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _two_column_spanning_header_document(),
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("<table>", markdown)
        self.assertIn('<th colspan="2">Species on protected list</th>', markdown)
        self.assertIn("<td>Potosi Pupfish</td>", markdown)
        self.assertNotIn("| Species | on protected list |", markdown)

    def test_two_column_inventory_section_divider_keeps_colspan_despite_body_rowspans(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _two_column_inventory_section_divider_document(),
                embed_images=False,
                table_export_mode="semantic_html",
            )
        )

        self.assertIn("<table>", markdown)
        self.assertIn('<td colspan="2">Learning Outcomes</td>', markdown)
        self.assertIn("<td>Knowledge</td>", markdown)
        self.assertNotIn("| Learning Outcomes |  |", markdown)

    def test_leading_boundary_header_semantic_html_keeps_first_compacted_body_row(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _leading_boundary_title_row_document(),
                embed_images=False,
                table_export_mode="semantic_html",
            )
        )

        self.assertIn("<table>", markdown)
        self.assertIn("<td>1. Project creation</td>", markdown)
        self.assertIn("<td>Project creation and management</td>", markdown)

    def test_multiline_header_projection_keeps_display_rows_when_semantic_rows_are_not_compacted(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "wrapped-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_wrapped", "table_id": "tbl_wrapped"}]},
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_wrapped",
                    "table_family": "comparison_matrix",
                    "display_grid": [
                        ["Jurisdiction", "Permitted", "Restrictions"],
                        ["Finland", "Y", "Prior approval for a foreigner’s"],
                        [None, None, "purchase of certain businesses"],
                        ["France", "Y", "None."],
                    ],
                    "semantic_grid": [
                        ["Jurisdiction", "Permitted", "Restrictions"],
                        ["Finland", "Y", "Prior approval for a foreigner’s"],
                        [None, None, "purchase of certain businesses"],
                        ["France", "Y", "None."],
                    ],
                    "semantic_projection_v2": {
                        "version": 2,
                        "table_family": "comparison_matrix",
                        "source": "table_semantic_projection_v2",
                        "multiline_schema_header_projection": {"source": "multiline_schema_header_semantic_projection"},
                    },
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                document,
                embed_images=False,
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("| Finland | Y | Prior approval for a foreigner’s |", markdown)
        self.assertIn("|  |  | purchase of certain businesses |", markdown)

    def test_semantic_html_renders_sparse_header_spans_when_label_is_centered_inside_span(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(
                _sparse_header_span_table_document(),
                embed_images=False,
                table_export_mode="semantic_html",
            )
        )

        self.assertIn("<table>", markdown)
        self.assertIn('<th rowspan="3">Properties</th>', markdown)
        self.assertIn('<th colspan="6">Training Datasets</th>', markdown)
        self.assertIn('<th colspan="3">Instruction</th>', markdown)
        self.assertIn('<th colspan="3">Alignment</th>', markdown)
        self.assertIn("<th>Synth. Math-Instruct</th>", markdown)
        self.assertNotIn("| Properties |", markdown)


def _semantic_span_table_document() -> dict:
    return {
        "filename": "semantic-span-table.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
            ]
        },
        "pages": [{"page_number": 1, "block_count": 1}],
        "table_asts": [
            {
                "table_id": "tbl_001",
                "display_grid": [
                    ["Tissue", "0.5 h", "1 h"],
                    ["Plasma", "1.0", "2.0"],
                ],
                "semantic_grid": [
                    ["Tissue", "Concentration (ng eq*/g)", None],
                    [None, "0.5 h", "1 h"],
                    ["Plasma", "1.0", "2.0"],
                ],
                "logical_cells": [
                    {
                        "row": 0,
                        "col": 0,
                        "rowspan": 2,
                        "colspan": 1,
                        "text": "Tissue",
                        "source": "header_row_group",
                    },
                    {
                        "row": 0,
                        "col": 1,
                        "rowspan": 1,
                        "colspan": 2,
                        "text": "Concentration (ng eq*/g)",
                        "source": "header_column_group",
                    },
                ],
                "semantic_projection_v2": {"version": 2, "table_family": "comparison_matrix"},
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _plain_semantic_table_document() -> dict:
    return {
        "filename": "plain-semantic-table.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_plain", "table_id": "tbl_plain"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_plain",
                "table_family": "comparison_matrix",
                "semantic_grid": [["A", "B"], ["1", "2"]],
                "display_grid": [["A", "B"], ["1", "2"]],
                "logical_cells": [
                    {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "A"},
                    {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "text": "B"},
                ],
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _sparse_header_span_table_document() -> dict:
    return {
        "filename": "sparse-header-span-table.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_sparse_header", "table_id": "tbl_sparse_header"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_sparse_header",
                "table_family": "rowspan_grouped_table",
                "semantic_grid": [
                    [None, None, None, None, "Training Datasets", None, None],
                    ["Properties", None, "Instruction", None, None, "Alignment", None],
                    [None, "Alpaca-GPT4", "OpenOrca", "Synth. Math-Instruct", "Orca DPO Pairs", "Ultrafeedback Cleaned", None],
                    ["Total # Samples", "52K", "2.91M", "126K", "12.9K", "60.8K", "126K"],
                ],
                "display_grid": [
                    [None, None, None, None, "Training Datasets", None, None],
                    ["Properties", None, "Instruction", None, None, "Alignment", None],
                    [None, "Alpaca-GPT4", "OpenOrca", "Synth. Math-Instruct", "Orca DPO Pairs", "Ultrafeedback Cleaned", None],
                    ["Total # Samples", "52K", "2.91M", "126K", "12.9K", "60.8K", "126K"],
                ],
                "logical_cells": [
                    {
                        "row": 0,
                        "col": 0,
                        "rowspan": 3,
                        "colspan": 1,
                        "text": "Properties",
                        "source": "sparse_header_rowspan_projection",
                    },
                    {
                        "row": 0,
                        "col": 1,
                        "rowspan": 1,
                        "colspan": 6,
                        "text": "Training Datasets",
                        "source": "sparse_header_colspan_projection",
                    },
                    {
                        "row": 1,
                        "col": 1,
                        "rowspan": 1,
                        "colspan": 3,
                        "text": "Instruction",
                        "source": "sparse_header_colspan_projection",
                    },
                    {
                        "row": 1,
                        "col": 4,
                        "rowspan": 1,
                        "colspan": 3,
                        "text": "Alignment",
                        "source": "sparse_header_colspan_projection",
                    },
                ],
                "semantic_projection_v2": {"version": 2, "table_family": "rowspan_grouped_table"},
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _body_rowspan_repair_table_document() -> dict:
    return {
        "filename": "body-rowspan-repair-table.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_body_span", "table_id": "tbl_body_span"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_body_span",
                "table_family": "comparison_matrix",
                "semantic_grid": [
                    ["Jurisdiction", "Description"],
                    ["Finland", "Long description part 1"],
                    [None, "Long description part 2"],
                ],
                "display_grid": [
                    ["Jurisdiction", "Description"],
                    ["Finland", "Long description part 1"],
                    [None, "Long description part 2"],
                ],
                "logical_cells": [
                    {
                        "row": 1,
                        "col": 0,
                        "rowspan": 2,
                        "colspan": 1,
                        "text": "Finland",
                        "source": "sparse_body_rowspan_projection",
                    },
                ],
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _structured_source_rowspan_table_document() -> dict:
    return {
        "filename": "structured-source-rowspan-table.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_structured_span", "table_id": "tbl_structured_span"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_structured_span",
                "table_family": "rowspan_grouped_table",
                "detection_source": "caption_anchored_horizontal_rules",
                "detection_method": "caption_anchored_horizontal_rules",
                "semantic_grid": [
                    ["Dose group", "Finding"],
                    ["Dose group", "No lesion"],
                    [None, "Mild lesion"],
                ],
                "display_grid": [
                    ["Dose group", "Finding"],
                    ["Dose group", "No lesion"],
                    [None, "Mild lesion"],
                ],
                "logical_cells": [
                    {
                        "row": 1,
                        "col": 0,
                        "rowspan": 2,
                        "colspan": 1,
                        "text": "Dose group",
                        "source": "sparse_body_rowspan_projection",
                    },
                ],
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _stale_body_span_after_semantic_projection_document() -> dict:
    return {
        "filename": "stale-body-span-after-semantic-projection.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_stale_span", "table_id": "tbl_stale_span"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_stale_span",
                "table_family": "rowspan_grouped_table",
                "detection_source": "caption_anchored_horizontal_rules",
                "detection_method": "caption_anchored_horizontal_rules",
                "semantic_grid": [
                    ["Time t", "Portfolio return", "New investment return"],
                    ["2012", "10%", "7%"],
                    ["2013", "6%", "8%"],
                ],
                "display_grid": [
                    ["Time t", "Portfolio return", "New investment return"],
                    ["2012", "10%", "7%"],
                    ["2013", "6%", "8%"],
                ],
                "logical_cells": [
                    {
                        "row": 1,
                        "col": 0,
                        "rowspan": 2,
                        "colspan": 1,
                        "text": "Time t",
                        "source": "sparse_body_rowspan_projection",
                    },
                    {
                        "row": 2,
                        "col": 1,
                        "rowspan": 2,
                        "colspan": 1,
                        "text": "portfolio",
                        "source": "sparse_body_rowspan_projection",
                    },
                ],
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _stacked_numeric_schedule_projection_document() -> dict:
    return {
        "filename": "stacked-numeric-schedule.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_schedule", "table_id": "tbl_schedule"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_schedule",
                "table_family": "numeric_schedule_table",
                "display_grid": [
                    ["Year", "3-Year", "5-Year", "7-Year"],
                    ["6 7 8", None, "5.76%", "8.93% 8.93% 4.46%"],
                ],
                "semantic_grid": [
                    ["Year", "3-Year", "5-Year", "7-Year"],
                    ["6", None, "5.76%", "8.93%"],
                    ["7", None, None, "8.93%"],
                    ["8", None, None, "4.46%"],
                ],
                "semantic_projection_v2": {
                    "version": 2,
                    "source": "table_semantic_projection_v2",
                    "table_family": "numeric_schedule_table",
                    "stacked_numeric_atom_row_expansion": {"source": "numeric_schedule_table"},
                },
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _projected_stub_table_document() -> dict:
    return {
        "filename": "projected-stub-table.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_stub", "table_id": "tbl_stub"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_stub",
                "table_family": "projected_stub_matrix",
                "semantic_grid": [
                    [None, "OCR", "Recommendation"],
                    ["Pack", "Recognizes text", "Suggests products"],
                ],
                "display_grid": [["OCR", "Recommendation"], ["Recognizes", "Suggests"]],
                "logical_cells": [],
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _parallel_inventory_compacted_document() -> dict:
    return {
        "filename": "parallel-inventory.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_inventory", "table_id": "tbl_inventory"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_inventory",
                "table_family": "two_column_inventory",
                "display_grid": [
                    ["Reagents", "Supplies and Equipment"],
                    ["Sample DNA*", "Pipet tips"],
                    ["Evidence A DNA*", "Water bath"],
                ],
                "data_grid": [
                    ["Sample DNA*", "Pipet tips"],
                    ["Evidence A DNA*", "Water bath"],
                ],
                "semantic_grid": [
                    ["Reagents", "Supplies and Equipment"],
                    ["Sample DNA* Evidence A DNA*", "Tube rack Pipet tips Water bath"],
                ],
                "logical_cells": [],
                "semantic_projection_v2": {
                    "version": 2,
                    "table_family": "two_column_inventory",
                    "source": "table_semantic_projection_v2",
                    "parallel_inventory_list_compaction": {"source": "parallel_inventory_list_compaction"},
                },
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _blank_form_projection_document() -> dict:
    return {
        "filename": "blank-form-projection.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_blank", "table_id": "tbl_blank"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_blank",
                "table_family": "blank_form_comparison_matrix",
                "display_grid": [
                    ["Added cation", "Relative Size & Settling Rates of Floccules"],
                    ["K+", None],
                    ["Na+", None],
                    ["Ca2+", None],
                ],
                "semantic_grid": [
                    [None, "Added cation", "Relative Size & Settling Rates of Floccules"],
                    ["K+", None, None],
                    ["Na+", None, None],
                    ["Ca2+", None, None],
                ],
                "logical_cells": [],
                "semantic_projection_v2": {
                    "version": 2,
                    "table_family": "blank_form_comparison_matrix",
                    "source": "table_semantic_projection_v2",
                    "blank_stub_column_projection": {"source": "blank_form_comparison_projection"},
                },
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _missing_leading_stub_projection_document() -> dict:
    return {
        "filename": "missing-leading-stub.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_stub_header", "table_id": "tbl_stub_header"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_stub_header",
                "table_family": "comparison_matrix",
                "display_grid": [
                    ["Foreign", None, "Domestic"],
                    ["MANILA", "2454", "6,125"],
                    ["CEBU", "1138", "79,500"],
                ],
                "semantic_grid": [
                    [None, "Foreign", "Domestic"],
                    ["MANILA", "2454", "6,125"],
                    ["CEBU", "1138", "79,500"],
                ],
                "logical_cells": [],
                "semantic_projection_v2": {
                    "version": 2,
                    "table_family": "comparison_matrix",
                    "source": "table_semantic_projection_v2",
                    "missing_leading_stub_header_projection": {"source": "missing_leading_stub_header_projection"},
                },
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _structural_header_duplicate_data_grid_document() -> dict:
    return {
        "filename": "structural-header-duplicate-data-grid.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_duplicate_header", "table_id": "tbl_duplicate_header"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_duplicate_header",
                "table_family": "comparison_matrix",
                "header": [
                    {"text": "Government"},
                    {"text": "No. of Seats"},
                    {"text": "Aquino"},
                    {"text": "Ramos"},
                ],
                "display_grid": [
                    ["Government", "No. of Seats", "Aquino", "Ramos"],
                    ["Senate", "24", "8.3", "16.7"],
                    ["House of Representatives", "202", "9.4", "10.4"],
                ],
                "data_grid": [
                    ["Government", "No. of Seats", "Aquino", "Ramos"],
                    ["Senate", "24", "8.3", "16.7"],
                    ["House of Representatives", "202", "9.4", "10.4"],
                ],
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _two_column_spanning_header_document() -> dict:
    return {
        "filename": "two-column-spanning-header.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_span2", "table_id": "tbl_span2"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_span2",
                "table_family": "two_column_spanning_header_table",
                "display_grid": [
                    ["Species", "on protected list"],
                    ["Potosi Pupfish", "Cyprinodon alvarezi"],
                ],
                "semantic_grid": [
                    ["Species on protected list", None],
                    ["Potosi Pupfish", "Cyprinodon alvarezi"],
                ],
                "logical_cells": [
                    {
                        "row": 0,
                        "col": 0,
                        "rowspan": 1,
                        "colspan": 2,
                        "text": "Species on protected list",
                        "source": "two_column_spanning_header_projection",
                    }
                ],
                "semantic_projection_v2": {
                    "version": 2,
                    "table_family": "two_column_spanning_header_table",
                    "source": "table_semantic_projection_v2",
                    "two_column_spanning_header_projection": {"source": "two_column_spanning_header_projection"},
                },
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _two_column_inventory_section_divider_document() -> dict:
    return {
        "filename": "two-column-inventory-section-divider.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_inventory", "table_id": "tbl_inventory"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_inventory",
                "table_family": "two_column_inventory",
                "display_grid": [
                    ["Competence Area", "#1 THE 3 RS"],
                    [None, "Introductory competence text"],
                    ["Competence Statement", None],
                    ["Learning Outcomes", None],
                    ["Knowledge", "Understand reducing and recycling"],
                    ["Skills", "Apply recycling practices"],
                ],
                "semantic_grid": [
                    ["Competence Area", "#1 THE 3 RS"],
                    ["Competence Statement", "Introductory competence text"],
                    ["Learning Outcomes", None],
                    ["Knowledge", "Understand reducing and recycling"],
                    ["Skills", "Apply recycling practices"],
                ],
                "logical_cells": [
                    {"row": 1, "col": 0, "rowspan": 2, "colspan": 1, "text": "Competence Area", "source": "sparse_body_rowspan_projection"},
                    {"row": 2, "col": 0, "rowspan": 2, "colspan": 1, "text": "Learning Outcomes", "source": "sparse_body_rowspan_projection"},
                ],
                "semantic_projection_v2": {
                    "version": 2,
                    "table_family": "two_column_inventory",
                    "source": "table_semantic_projection_v2",
                    "label_after_value_pair_projection": {"source": "two_column_label_after_value_pair_projection"},
                },
            }
        ],
        "image_blocks": [],
        "text": "",
    }


def _leading_boundary_title_row_document() -> dict:
    return {
        "filename": "leading-boundary-title-row.pdf",
        "source_type": "pdf",
        "metadata": {"page_count": 1, "parser_hint": "pdf"},
        "document_ast": {
            "pages": [
                {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_flow", "table_id": "tbl_flow"}]},
            ]
        },
        "table_asts": [
            {
                "table_id": "tbl_flow",
                "table_family": "rowspan_grouped_table",
                "display_grid": [
                    ["Key Functions by Main Service Flow", None, None, None],
                    ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                    ["1. Project creation", "Project creation and", "Select document type", "Proceed quickly"],
                    [None, "management", "with recommended model", "with the process"],
                    ["2. Labeling", "Data storage", "Manage image data", "Manage raw data"],
                ],
                "semantic_grid": [
                    ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                    ["1. Project creation", "Project creation and management", "Select document type with recommended model", "Proceed quickly with the process"],
                    ["2. Labeling", "Data storage", "Manage image data", "Manage raw data"],
                ],
                "logical_cells": [],
                "semantic_projection_v2": {
                    "version": 2,
                    "table_family": "rowspan_grouped_table",
                    "source": "table_semantic_projection_v2",
                    "leading_boundary_header_projection": {
                        "source": "leading_boundary_header_projection",
                        "dropped_leading_row_count": 1,
                        "header_row_index": 1,
                    },
                    "rowspan_key_repetition_compaction": {"source": "rowspan_key_repetition_compaction"},
                    "multicolumn_wrapped_record_compaction": {"source": "multicolumn_wrapped_record_compaction"},
                },
            }
        ],
        "image_blocks": [],
        "text": "",
    }


if __name__ == "__main__":
    unittest.main()
