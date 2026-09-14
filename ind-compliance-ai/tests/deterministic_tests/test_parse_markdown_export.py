from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest


class ParseMarkdownExportTests(unittest.TestCase):
    def test_evidence_markdown_renders_header_only_canonical_spans_as_html(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "block_type": "table",
            "table_id": "tbl_dose_sex_header",
            "semantic_grid": [
                ["日剂量(mg/kg)", "0 M", "0 F", "10 M", "10 F"],
                ["动物数量", "M:3", "F:3", "M:3", "F:3"],
            ],
            "display_grid": [
                ["日剂量(mg/kg)", "0 M", "0 F", "10 M", "10 F"],
                ["动物数量", "M:3", "F:3", "M:3", "F:3"],
            ],
            "cell_spans": [
                {"role": "header", "row": 0, "col": 0, "rowspan": 2, "colspan": 1, "text": "日剂量(mg/kg)"},
                {"role": "header", "row": 0, "col": 1, "rowspan": 1, "colspan": 2, "text": "0"},
                {"role": "header", "row": 0, "col": 3, "rowspan": 1, "colspan": 2, "text": "10"},
            ],
        }

        lines: list[str] = []
        api_main._append_markdown_table(
            lines,
            table,
            table_export_mode="evidence_markdown",
        )
        markdown = "\n".join(lines)

        self.assertIn("<table>", markdown)
        self.assertIn('<th rowspan="2">日剂量(mg/kg)</th>', markdown)
        self.assertIn('<th colspan="2">0</th>', markdown)
        self.assertIn('<th colspan="2">10</th>', markdown)
        self.assertIn("<th>M</th>", markdown)
        self.assertIn("<th>F</th>", markdown)

    def test_canonical_three_level_header_materialization_preserves_leaf_columns(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "block_type": "table",
            "table_id": "tbl_three_level_header",
            "semantic_grid": [
                ["日剂量(mg/kg)", "M", "F", "M", "F", "犬"],
                ["1", "-", "-", "2", "3", "4"],
            ],
            "cell_spans": [
                {"role": "header", "row": 0, "col": 1, "rowspan": 1, "colspan": 4, "text": "稳态AUC"},
                {"role": "header", "row": 1, "col": 1, "rowspan": 1, "colspan": 2, "text": "小鼠"},
                {"role": "header", "row": 1, "col": 3, "rowspan": 1, "colspan": 2, "text": "大鼠"},
            ],
        }

        projected = api_main._project_markdown_visible_semantic_grid(
            table,
            table["semantic_grid"],
        )
        self.assertEqual(
            projected[:3],
            [
                ["", "稳态AUC", "", "", "", ""],
                ["", "小鼠", "", "大鼠", "", ""],
                ["日剂量(mg/kg)", "M", "F", "M", "F", "犬"],
            ],
        )

        rendered_table = api_main._markdown_table_with_projected_presentation_surface(
            table,
            projected,
        )
        html = api_main._build_semantic_html_table(rendered_table)
        self.assertIn('<th colspan="4">稳态AUC</th>', html)
        self.assertIn('<th colspan="2">小鼠</th>', html)
        self.assertIn('<th colspan="2">大鼠</th>', html)
        self.assertIn("<th>犬</th>", html)

    def test_semantic_html_prefers_canonical_cell_spans_over_legacy_spans(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "semantic_grid": [
                ["Group", "Results", "", ""],
                ["", "A", "B", "C"],
                ["X", "1", "2", "3"],
                ["X", "4", "5", "6"],
            ],
            "cell_spans": [
                {
                    "role": "header",
                    "coordinate_space": "semantic_grid",
                    "row": 0,
                    "col": 0,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": "Group",
                },
                {
                    "role": "header",
                    "coordinate_space": "semantic_grid",
                    "row": 0,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": 3,
                    "text": "Results",
                },
                {
                    "role": "body",
                    "coordinate_space": "semantic_grid",
                    "row": 2,
                    "col": 0,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": "X",
                },
            ],
            "logical_cells": [
                {
                    "row": 0,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": 2,
                    "text": "Results",
                    "source": "legacy_wrong_header",
                }
            ],
        }

        html = api_main._build_semantic_html_table(table)

        self.assertIsNotNone(html)
        self.assertIn('<th rowspan="2">Group</th>', html)
        self.assertIn('<th colspan="3">Results</th>', html)
        self.assertNotIn('<th colspan="2">Results</th>', html)
        self.assertIn('<td rowspan="2">X</td>', html)
        self.assertEqual(html.count(">Group</th>"), 1)
        self.assertEqual(html.count(">Results</th>"), 1)
        self.assertEqual(html.count(">X</td>"), 1)

    def test_ind_review_renders_source_backed_body_rowspan_without_mutating_semantic_rows(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        semantic_grid = [
            ["Group", "Value"],
            ["A", "1"],
            ["A", "2"],
        ]
        table = {
            "block_type": "table",
            "block_id": "tbl_grouped",
            "table_id": "tbl_grouped",
            "semantic_role": "business_table",
            "page": 1,
            "data_start_row": 3,
            "semantic_grid": [list(row) for row in semantic_grid],
            "display_grid": [list(row) for row in semantic_grid],
            "raw_grid": [list(row) for row in semantic_grid],
            "presentation_spans": [
                {
                    "row": 1,
                    "col": 0,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": "A",
                    "source": "source_body_row_group_presentation_projection",
                }
            ],
            "semantic_projection_v2": {
                "toxicology_summary_schema_projection": {
                    "semantic_profile": "toxicology_summary_schema_table",
                }
            },
        }
        document = {
            "filename": "source-rowspan.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn('<td rowspan="2">A</td>', markdown)
        self.assertNotIn("| A | 1 |", markdown)
        self.assertEqual(table["semantic_grid"], semantic_grid)

    def test_ind_review_keeps_source_repeated_values_without_presentation_span(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "block_type": "table",
            "block_id": "tbl_repeated",
            "table_id": "tbl_repeated",
            "semantic_role": "business_table",
            "page": 1,
            "semantic_grid": [["Group", "Value"], ["A", "1"], ["A", "2"]],
            "display_grid": [["Group", "Value"], ["A", "1"], ["A", "2"]],
            "raw_grid": [["Group", "Value"], ["A", "1"], ["A", "2"]],
        }
        document = {
            "filename": "source-repeated.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(markdown.count("| A |"), 2)
        self.assertNotIn("rowspan=", markdown)

    def test_ind_review_ignores_stale_data_start_for_study_metric_projection(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        semantic_grid = [
            ["Batch", "Impurity", "Impurity", "Study"],
            ["", "A", "B", ""],
            ["LOT-1", "0.1", "0.2", "S-1"],
            ["LOT-1", "", "", "S-2"],
        ]
        table = {
            "block_type": "table",
            "block_id": "tbl_metric",
            "table_id": "tbl_metric",
            "semantic_role": "business_table",
            "page": 1,
            "data_start_row": 6,
            "semantic_grid": [list(row) for row in semantic_grid],
            "display_grid": [list(row) for row in semantic_grid],
            "raw_grid": [list(row) for row in semantic_grid],
            "presentation_spans": [
                {
                    "row": 2,
                    "col": 0,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": "LOT-1",
                    "source": "source_body_row_group_presentation_projection",
                }
            ],
            "semantic_projection_v2": {
                "study_metric_grouped_matrix_projection": {
                    "semantic_profile": "study_metric_grouped_matrix",
                }
            },
        }
        document = {
            "filename": "study-metric.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn('<td rowspan="2">LOT-1</td>', markdown)
        self.assertNotIn('<th rowspan="2">LOT-1</th>', markdown)

    def test_continued_table_chain_extends_trailing_span_over_inherited_omitted_prefix(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        root = {
            "table_id": "tbl_root",
            "semantic_grid": [["Group", "Value"], ["A", "1"], ["A", "2"]],
            "semantic_row_provenance": [
                {"source_row_refs": ["tbl_root:display_row:1"]},
                {"source_row_refs": ["tbl_root:display_row:2"]},
                {"source_row_refs": ["tbl_root:display_row:3"]},
            ],
            "display_grid": [["Group", "Value"], ["A", "1"], [None, "2"]],
            "presentation_spans": [
                {
                    "row": 1,
                    "col": 0,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": "A",
                    "source": "source_body_row_group_presentation_projection",
                }
            ],
            "semantic_projection_v2": {
                "genotoxicity_assay_matrix_projection": {
                    "semantic_profile": "genotoxicity_assay_matrix",
                    "assay_kind": "example_matrix",
                }
            },
        }
        continuation = {
            "table_id": "tbl_continuation",
            "continued_from_table_id": "tbl_root",
            "semantic_grid": [["Group", "Value"], ["A", "3"], ["A", "4"], ["B", "5"]],
            "semantic_row_provenance": [
                {"source_row_refs": []},
                {"source_row_refs": ["tbl_continuation:display_row:1"]},
                {"source_row_refs": ["tbl_continuation:display_row:2"]},
                {"source_row_refs": ["tbl_continuation:display_row:3"]},
            ],
            "display_grid": [["3"], ["4"], ["B", "5"]],
            "semantic_projection_v2": {
                "genotoxicity_assay_matrix_projection": {
                    "semantic_profile": "genotoxicity_assay_matrix",
                    "assay_kind": "example_matrix",
                    "continuation_schema_inherited": True,
                }
            },
        }

        merged = api_main._merge_continued_table_chain([root, continuation])

        self.assertEqual(
            merged.get("presentation_spans"),
            [
                {
                    "row": 1,
                    "col": 0,
                    "rowspan": 4,
                    "colspan": 1,
                    "text": "A",
                    "source": "source_body_row_group_presentation_projection",
                }
            ],
        )
        self.assertEqual(
            [item.get("source_row_refs") for item in merged.get("semantic_row_provenance", [])],
            [
                ["tbl_root:display_row:1"],
                ["tbl_root:display_row:2"],
                ["tbl_root:display_row:3"],
                ["tbl_continuation:display_row:1"],
                ["tbl_continuation:display_row:2"],
                ["tbl_continuation:display_row:3"],
            ],
        )

    def test_continued_table_chain_extends_canonical_body_span(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        root = {
            "table_id": "tbl_root",
            "page": 102,
            "semantic_grid": [["Group", "Value"], ["A", "1"], ["A", "2"]],
            "display_grid": [["Group", "Value"], ["A", "1"], [None, "2"]],
            "presentation_spans": [
                {
                    "row": 1,
                    "col": 0,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": "A",
                    "source": "source_body_row_group_presentation_projection",
                }
            ],
            "cell_spans": [
                {
                    "span_id": "tbl_root:cell_span:1",
                    "role": "body",
                    "coordinate_space": "semantic_grid",
                    "row": 1,
                    "col": 0,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": "A",
                    "source_pages": [102],
                    "source_table_ids": ["tbl_root"],
                    "source_cell_refs": ["tbl_root:display_row:2", "tbl_root:display_row:3"],
                    "span_group_id": "tbl_root:body_group:0:1:2",
                    "evidence": "source_body_row_group_presentation_projection",
                    "confidence": 0.94,
                }
            ],
            "semantic_projection_v2": {
                "genotoxicity_assay_matrix_projection": {
                    "semantic_profile": "genotoxicity_assay_matrix",
                    "assay_kind": "example_matrix",
                }
            },
        }
        continuation = {
            "table_id": "tbl_continuation",
            "page": 103,
            "continued_from_table_id": "tbl_root",
            "semantic_grid": [["Group", "Value"], ["A", "3"], ["A", "4"], ["B", "5"]],
            "display_grid": [["3"], ["4"], ["B", "5"]],
            "semantic_projection_v2": {
                "genotoxicity_assay_matrix_projection": {
                    "semantic_profile": "genotoxicity_assay_matrix",
                    "assay_kind": "example_matrix",
                    "continuation_schema_inherited": True,
                }
            },
        }

        merged = api_main._merge_continued_table_chain([root, continuation])

        body_spans = [
            span
            for span in merged.get("cell_spans", []) or []
            if isinstance(span, dict) and span.get("role") == "body"
        ]
        self.assertEqual(len(body_spans), 1)
        span = body_spans[0]
        self.assertEqual(
            (span["coordinate_space"], span["row"], span["col"], span["rowspan"], span["colspan"]),
            ("logical_table_chain", 1, 0, 4, 1),
        )
        self.assertEqual(span["source_pages"], [102, 103])
        self.assertEqual(span["source_table_ids"], ["tbl_root", "tbl_continuation"])
        self.assertEqual(span["span_group_id"], "tbl_root:body_group:0:1:2")

    def test_continued_table_chain_drops_complete_inherited_multilevel_header_prefix(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        header = [
            "代谢活化",
            "供试品",
            "浓度(µg/ml)",
            "细胞毒性a(%对照)",
            "平均细胞畸变率%",
            "Abs/细胞",
            "多倍体细胞总数",
        ]
        span = {
            "row": 0,
            "col": 3,
            "rowspan": 1,
            "colspan": 4,
            "text": "细胞毒性a",
            "source": "genotoxicity_multilevel_header_projection",
        }
        root = {
            "table_id": "tbl_root",
            "semantic_grid": [
                header,
                ["无代谢活化", "DMSO", "-", "100", "2.0", "0.02", "4"],
            ],
            "span_header_cells": [span],
            "semantic_projection_v2": {
                "genotoxicity_assay_matrix_projection": {
                    "semantic_profile": "genotoxicity_assay_matrix",
                    "assay_kind": "chromosomal_aberration_matrix",
                    "span_header_cells": [span],
                }
            },
        }
        continuation = {
            "table_id": "tbl_continuation",
            "continued_from_table_id": "tbl_root",
            "semantic_grid": [
                header,
                ["有代谢活化", "环磷酰胺", "4", "68", "36.5**", "0.63", "6"],
            ],
            "span_header_cells": [{**span, "inherited_from_table_id": "tbl_root"}],
            "semantic_projection_v2": {
                "genotoxicity_assay_matrix_projection": {
                    "semantic_profile": "genotoxicity_assay_matrix",
                    "assay_kind": "chromosomal_aberration_matrix",
                    "continuation_schema_inherited": True,
                    "span_header_cells": [{**span, "inherited_from_table_id": "tbl_root"}],
                }
            },
        }

        merged = api_main._merge_continued_table_chain([root, continuation])

        self.assertEqual(
            merged.get("semantic_grid"),
            [
                ["代谢活化", "供试品", "浓度(µg/ml)", "细胞毒性a", "细胞毒性a", "细胞毒性a", "细胞毒性a"],
                header,
                ["无代谢活化", "DMSO", "-", "100", "2.0", "0.02", "4"],
                ["有代谢活化", "环磷酰胺", "4", "68", "36.5**", "0.63", "6"],
            ],
        )

    def test_dose_response_header_projection_preserves_merged_row_source_alignment(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "block_type": "table",
            "block_id": "tbl_001",
            "table_id": "tbl_001",
            "semantic_role": "business_table",
            "page": 1,
            "semantic_grid": [
                ["日剂量(mg/kg)", "0 M", "0 F", "200 M", "200 F"],
                ["动物数量", "M:30", "F:30", "M:20", "F:20"],
                ["附加检查", "-", "-", "-", "-"],
                ["给药后评价：", "", "", "", ""],
                ["评价数量", "10", "10", "0", "0"],
            ],
            "merged_rows": [
                {
                    "row": 4,
                    "kind": "table_note_title",
                    "text": "给药后评价：",
                    "colspan": 5,
                    "source": "single_leading_title_cell_with_following_tabular_rows",
                }
            ],
            "semantic_projection_v2": {
                "source": "table_semantic_projection_v2",
                "dose_response_result_panel_projection": {
                    "semantic_profile": "dose_response_result_panel",
                    "has_sex_leaf_columns": True,
                    "source_has_explicit_sex_header_row": False,
                },
            },
        }
        document = {
            "filename": "dose-response-merged-row.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("| 日剂量(mg/kg) | 0 | 0 | 200 | 200 |", markdown)
        self.assertNotIn("| 性别 | M | F | M | F |", markdown)
        self.assertIn("| 附加检查 | - | - | - | - |", markdown)
        self.assertIn("**给药后评价：**", markdown)
        self.assertNotIn("| 给药后评价： |  |  |  |  |", markdown)

    def test_dose_response_header_projection_keeps_source_backed_sex_row(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        rows = [
            ["日剂量(mg/kg)", "0 M", "0 F", "200 M", "200 F"],
            ["动物数量", "M:30", "F:30", "M:20", "F:20"],
        ]
        block = {
            "semantic_projection_v2": {
                "source": "table_semantic_projection_v2",
                "dose_response_result_panel_projection": {
                    "semantic_profile": "dose_response_result_panel",
                    "has_sex_leaf_columns": True,
                    "source_has_explicit_sex_header_row": True,
                },
            }
        }

        self.assertEqual(
            api_main._project_markdown_visible_semantic_grid(block, rows)[:2],
            [
                ["日剂量(mg/kg)", "0", "0", "200", "200"],
                ["性别", "M", "F", "M", "F"],
            ],
        )

    def test_source_backed_projected_header_rebases_merged_row_coordinates(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "block_type": "table",
            "block_id": "tbl_001",
            "table_id": "tbl_001",
            "semantic_role": "business_table",
            "page": 1,
            "semantic_grid": [
                ["日剂量(mg/kg)", "0 M", "0 F", "200 M", "200 F"],
                ["动物数量", "M:30", "F:30", "M:20", "F:20"],
                ["附加检查", "-", "-", "-", "-"],
                ["给药后评价：", "", "", "", ""],
                ["评价数量", "10", "10", "0", "0"],
            ],
            "merged_rows": [
                {
                    "row": 4,
                    "kind": "table_note_title",
                    "text": "给药后评价：",
                    "colspan": 5,
                    "source": "single_leading_title_cell_with_following_tabular_rows",
                }
            ],
            "semantic_projection_v2": {
                "source": "table_semantic_projection_v2",
                "dose_response_result_panel_projection": {
                    "semantic_profile": "dose_response_result_panel",
                    "has_sex_leaf_columns": True,
                    "source_has_explicit_sex_header_row": True,
                },
            },
        }
        document = {
            "filename": "source-backed-sex-header.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("| 性别 | M | F | M | F |", markdown)
        self.assertIn("| 附加检查 | - | - | - | - |", markdown)
        self.assertIn("**给药后评价：**", markdown)
        self.assertNotIn("| 给药后评价： |  |  |  |  |", markdown)

    def test_source_backed_sex_header_keeps_structural_and_first_data_rows(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "block_type": "table",
            "block_id": "tbl_lineage",
            "table_id": "tbl_lineage",
            "semantic_role": "business_table",
            "page": 1,
            "display_grid": [
                ["Dose", "0", "", "25", ""],
                ["Sex", "M", "F", "M", "F"],
                ["Toxicokinetics:", "", "", "", ""],
                ["Day 28", "AUC", "10", "12", "14"],
                ["Day 180 Css", "0.4", "0.5", "1.7", "0.3"],
            ],
            "semantic_grid": [
                ["Dose", "0 M", "0 F", "25 M", "25 F"],
                ["Toxicokinetics:", "", "", "", ""],
                ["Day 28 AUC", "10", "12", "14", ""],
                ["Day 180 Css", "0.4", "0.5", "1.7", "0.3"],
            ],
            "semantic_row_provenance": [
                {
                    "source_row_refs": [
                        "tbl_lineage:display_row:1",
                        "tbl_lineage:display_row:2",
                    ]
                },
                {"source_row_refs": ["tbl_lineage:display_row:3"]},
                {"source_row_refs": ["tbl_lineage:display_row:4"]},
                {"source_row_refs": ["tbl_lineage:display_row:5"]},
            ],
            "merged_rows": [
                {
                    "row": 3,
                    "source_grid": "display_grid",
                    "source_row_ref": "tbl_lineage:display_row:3",
                    "kind": "table_note_title",
                    "text": "Toxicokinetics:",
                    "colspan": 5,
                }
            ],
            "semantic_projection_v2": {
                "dose_response_result_panel_projection": {
                    "semantic_profile": "dose_response_result_panel",
                    "has_sex_leaf_columns": True,
                    "source_has_explicit_sex_header_row": True,
                }
            },
        }
        document = {
            "filename": "row-lineage.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(markdown.count("Toxicokinetics:"), 1)
        self.assertIn("| Day 28 AUC | 10 | 12 | 14 |  |", markdown)
        self.assertIn("| Day 180 Css | 0.4 | 0.5 | 1.7 | 0.3 |", markdown)

    def test_ambiguous_legacy_structural_signature_preserves_all_rows(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "block_type": "table",
            "block_id": "tbl_legacy",
            "table_id": "tbl_legacy",
            "semantic_role": "business_table",
            "page": 1,
            "semantic_grid": [
                ["Label", "Value"],
                ["Repeated:", ""],
                ["Repeated:", ""],
                ["Result", "10"],
            ],
            "merged_rows": [
                {
                    "row": 2,
                    "kind": "table_note_title",
                    "text": "Repeated:",
                    "colspan": 2,
                }
            ],
            "semantic_projection_v2": {
                "toxicology_summary_schema_projection": {
                    "semantic_profile": "toxicology_summary_schema_table",
                }
            },
        }
        document = {
            "filename": "legacy-ambiguous.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(markdown.count("| Repeated: |  |"), 2)
        self.assertIn("| Result | 10 |", markdown)

    def test_table_notes_keep_physical_order_across_local_and_cross_page_sources(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        local_legend = "-无值得注意的结果 +轻度 ++中度 +++显著"
        local_stats = "Dunnett 氏检验：*-p<0.05 **-p<0.01"
        cross_page_definition = "a-给药结束时。对照组给出组平均值。"
        table = {
            "block_type": "table",
            "block_id": "tbl_001",
            "table_id": "tbl_001",
            "semantic_role": "business_table",
            "page": 97,
            "display_grid": [["日剂量(mg/kg)", "0", "200"], ["体重", "394 g", "-10*"]],
            "note_blocks": [
                {
                    "role": "table_note",
                    "text": local_legend,
                    "source": "dose_response_result_panel_note_row",
                    "page": 97,
                    "bbox": [72, 464, 452, 476],
                },
                {
                    "role": "table_note",
                    "text": local_stats,
                    "source": "dose_response_result_panel_note_row",
                    "page": 97,
                    "bbox": [72, 480, 264, 492],
                },
                {
                    "role": "table_note",
                    "text": cross_page_definition,
                    "source": "cross_page_result_matrix_statistical_note",
                    "page": 98,
                    "bbox": [72, 92, 653, 104],
                },
            ],
        }
        document = {
            "filename": "cross-page-table-notes.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 2, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 97, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        markdown_lines = markdown.splitlines()
        for expected in (local_legend, local_stats, cross_page_definition):
            self.assertIn(expected, markdown_lines, msg=markdown)
        legend_index = markdown_lines.index(local_legend)
        stats_index = markdown_lines.index(local_stats)
        definition_index = markdown_lines.index(cross_page_definition)
        self.assertLess(legend_index, stats_index)
        self.assertLess(stats_index, definition_index)

    def test_study_metric_grouped_matrix_materializes_all_header_levels_in_markdown(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table = {
            "block_type": "table",
            "block_id": "tbl_auc",
            "table_id": "tbl_auc",
            "semantic_role": "business_table",
            "page": 1,
            "semantic_grid": [
                ["日剂量（mg/kg）", "M", "F", "M", "F", "犬c", "雌兔b", "人f"],
                ["25", "10", "12", "6", "8", "", "273", ""],
            ],
            "header_column_groups": [
                {"row": 0, "start_col": 1, "end_col": 4, "colspan": 4, "text": "稳态AUC (µg-h/ml)"},
                {"row": 1, "start_col": 1, "end_col": 2, "colspan": 2, "text": "小鼠a"},
                {"row": 1, "start_col": 3, "end_col": 4, "colspan": 2, "text": "大鼠b"},
                {"row": 1, "start_col": 5, "end_col": 5, "colspan": 1, "text": "犬c"},
                {"row": 1, "start_col": 6, "end_col": 6, "colspan": 1, "text": "雌兔b"},
                {"row": 1, "start_col": 7, "end_col": 7, "colspan": 1, "text": "人f"},
            ],
            "semantic_projection_v2": {
                "source": "table_semantic_projection_v2",
                "study_metric_grouped_matrix_projection": {
                    "semantic_profile": "study_metric_grouped_matrix",
                    "logical_column_count": 8,
                },
            },
        }
        document = {
            "filename": "auc-grouped-header.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn(
            "|  | 稳态AUC (µg-h/ml) | 稳态AUC (µg-h/ml) | 稳态AUC (µg-h/ml) | 稳态AUC (µg-h/ml) |  |  |  |",
            markdown,
        )
        self.assertIn(
            "| 日剂量（mg/kg） | 小鼠a | 小鼠a | 大鼠b | 大鼠b | 犬c | 雌兔b | 人f |",
            markdown,
        )
        self.assertIn("|  | M | F | M | F |  |  |  |", markdown)
        self.assertIn("| 25 | 10 | 12 | 6 | 8 |  | 273 |  |", markdown)

    def test_study_metric_grouped_matrix_does_not_add_header_rows_without_spans(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        rows = [["批号", "纯度(%)", "试验编号"], ["A001", "99.8", "T-1"]]
        block = {
            "header_column_groups": [
                {"row": 0, "start_col": 0, "end_col": 0, "colspan": 1, "text": "批号"},
                {"row": 0, "start_col": 1, "end_col": 1, "colspan": 1, "text": "纯度(%)"},
            ],
            "semantic_projection_v2": {
                "source": "table_semantic_projection_v2",
                "study_metric_grouped_matrix_projection": {
                    "semantic_profile": "study_metric_grouped_matrix",
                    "logical_column_count": 3,
                },
            },
        }

        self.assertEqual(api_main._project_markdown_visible_semantic_grid(block, rows), rows)

    def test_study_panel_note_group_renders_literal_marker_lines_in_source_order(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        group_id = "tbl_001:note_group:1"
        notes = [
            {
                "role": "note",
                "text": "附加信息：",
                "relation": "below",
                "note_group_id": group_id,
                "note_line_index": 0,
                "presentation_mode": "lines",
            },
            {
                "role": "note",
                "text": "* - 为了采集胆汁，十二指肠给药。",
                "marker": "*",
                "relation": "cross_page_note_continuation",
                "note_group_id": group_id,
                "note_line_index": 1,
                "presentation_mode": "lines",
            },
            {
                "role": "note",
                "text": "n.d.- 未检出",
                "marker": "n.d.",
                "relation": "cross_page_note_continuation",
                "note_group_id": group_id,
                "note_line_index": 2,
                "presentation_mode": "lines",
            },
        ]
        table = {
            "block_type": "table",
            "block_id": "tbl_001",
            "table_id": "tbl_001",
            "semantic_role": "business_table",
            "page": 1,
            "display_grid": [["分析物", "结果"], ["M1", "n.d."]],
            "raw_grid": [["分析物", "结果"], ["M1", "n.d."]],
            "note_blocks": notes,
        }
        document = {
            "filename": "study-panel-note-group.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("\nn.d.- 未检出", markdown, msg=markdown)
        label_pos = markdown.index("\n附加信息：\n")
        marker_pos = markdown.index("\n\\* - 为了采集胆汁，十二指肠给药。\n")
        definition_pos = markdown.index("\nn.d.- 未检出")
        self.assertLess(label_pos, marker_pos)
        self.assertLess(marker_pos, definition_pos)
        self.assertNotIn("\n* - 为了采集胆汁", markdown)

    def test_full_markdown_suppresses_table_note_subsegment_even_when_subsegment_has_stronger_source_evidence(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        full_note = (
            "-No notable findings +mild ++moderate +++marked "
            "Dunnett test: *-p<0.05 **-p<0.01 a-After dosing."
        )
        stats_note = "Dunnett test: *-p<0.05 **-p<0.01"
        document = {
            "filename": "table-note-subsegment.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "semantic_role": "business_table",
                                "raw_grid": [["Dose", "0", "10"], ["Body weight", "100 g", "-5*"]],
                                "display_grid": [["Dose", "0", "10"], ["Body weight", "100 g", "-5*"]],
                                "note_blocks": [
                                    {"role": "table_note", "text": full_note},
                                    {
                                        "role": "table_note",
                                        "text": stats_note,
                                        "source_block_id": "txt_stats_note",
                                        "source": "result_matrix_statistical_note_after_table",
                                        "bbox": [72, 220, 260, 232],
                                    },
                                ],
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "block_type": "table",
                    "block_id": "tbl_001",
                    "table_id": "tbl_001",
                    "semantic_role": "business_table",
                    "page": 1,
                    "raw_grid": [["Dose", "0", "10"], ["Body weight", "100 g", "-5*"]],
                    "display_grid": [["Dose", "0", "10"], ["Body weight", "100 g", "-5*"]],
                    "note_blocks": [
                        {"role": "table_note", "text": full_note},
                        {
                            "role": "table_note",
                            "text": stats_note,
                            "source_block_id": "txt_stats_note",
                            "source": "result_matrix_statistical_note_after_table",
                            "bbox": [72, 220, 260, 232],
                        },
                    ],
                }
            ],
        }

        markdown = api_main._build_full_markdown([document])

        self.assertEqual(markdown.count(full_note), 1, msg=markdown)
        self.assertEqual(markdown.count(stats_note), 1, msg=markdown)
        self.assertNotIn(f"\n{stats_note}\n", markdown, msg=markdown)

    def test_markdown_rendering_audit_reports_metadata_only_edge_suppression(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        edge = {
            "source_block_id": "txt_table_note",
            "relation": "table_note",
            "target_object_type": "table",
            "target_object_id": "tbl_001",
            "visible_render_policy": "metadata_only",
        }
        audit = api_main._markdown_rendering_audit_for_block(
            {
                "block_type": "text",
                "block_id": "txt_table_note",
                "text": "Already owned table note.",
            },
            {
                "page_number": 1,
                "page_height": 792.0,
                "metadata_only_source_block_ids": {"txt_table_note"},
                "metadata_only_source_block_edges": {"txt_table_note": [edge]},
            },
        )

        self.assertEqual(
            audit,
            {
                "block_id": "txt_table_note",
                "block_type": "text",
                "page": 1,
                "rendered": False,
                "role": "metadata_only_text",
                "reason": "metadata_reference_edge_metadata_only",
                "source_block_id": "txt_table_note",
                "relation": "table_note",
                "target_object_type": "table",
                "target_object_id": "tbl_001",
            },
        )

    def test_markdown_rendering_audit_reports_absorbed_structure_template_suppression(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        audit = api_main._markdown_rendering_audit_for_block(
            {
                "block_type": "structure_template",
                "block_id": "tpl_absorbed",
                "structure_template_id": "tpl_absorbed",
                "ownership_domain": "absorbed_by_business_table",
                "semantic_role": "absorbed_structure_template_fragment",
            },
            {
                "page_number": 1,
                "page_height": 792.0,
                "metadata_only_source_block_ids": set(),
            },
        )

        self.assertEqual(
            audit,
            {
                "block_id": "tpl_absorbed",
                "block_type": "structure_template",
                "page": 1,
                "rendered": False,
                "role": "absorbed_structure_template",
                "reason": "absorbed_structure_template",
            },
        )

    def test_markdown_page_rendering_audit_collects_suppressed_blocks(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        audits = api_main._collect_markdown_page_rendering_audit(
            [
                {
                    "block_type": "table",
                    "block_id": "tbl_001",
                    "table_id": "tbl_001",
                    "metadata_reference_edges": [
                        {
                            "source_block_id": "txt_table_note",
                            "relation": "table_note",
                            "target_object_type": "table",
                            "target_object_id": "tbl_001",
                            "visible_render_policy": "metadata_only",
                        }
                    ],
                },
                {
                    "block_type": "text",
                    "block_id": "txt_table_note",
                    "text": "Already owned table note.",
                },
                {
                    "block_type": "structure_template",
                    "block_id": "tpl_absorbed",
                    "structure_template_id": "tpl_absorbed",
                    "ownership_domain": "absorbed_by_business_table",
                    "semantic_role": "absorbed_structure_template_fragment",
                },
                {
                    "block_type": "text",
                    "block_id": "txt_body",
                    "text": "Visible body text.",
                },
            ],
            page_number=1,
            page_height=792.0,
        )

        self.assertEqual(
            [(item.get("block_id"), item.get("reason")) for item in audits],
            [
                ("txt_table_note", "metadata_reference_edge_metadata_only"),
                ("tpl_absorbed", "absorbed_structure_template"),
            ],
        )

    def test_markdown_rendering_audit_can_be_attached_to_document_metadata_explicitly(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "sample.pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "page_height": 792.0,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "metadata_reference_edges": [
                                    {
                                        "source_block_id": "txt_table_note",
                                        "relation": "table_note",
                                        "target_object_type": "table",
                                        "target_object_id": "tbl_001",
                                        "visible_render_policy": "metadata_only",
                                    }
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_table_note",
                                "text": "Already owned table note.",
                            },
                        ],
                    }
                ]
            },
        }

        result = api_main._attach_markdown_rendering_audit_metadata(document)

        self.assertIs(result, document)
        self.assertEqual(
            document["metadata"]["markdown_rendering_audit"],
            {
                "page_count": 1,
                "suppressed_block_count": 1,
                "reason_counts": {"metadata_reference_edge_metadata_only": 1},
                "pages": [
                    {
                        "page": 1,
                        "suppressed_block_count": 1,
                        "reason_counts": {"metadata_reference_edge_metadata_only": 1},
                        "suppressed_blocks": [
                            {
                                "block_id": "txt_table_note",
                                "block_type": "text",
                                "page": 1,
                                "rendered": False,
                                "role": "metadata_only_text",
                                "reason": "metadata_reference_edge_metadata_only",
                                "source_block_id": "txt_table_note",
                                "relation": "table_note",
                                "target_object_type": "table",
                                "target_object_id": "tbl_001",
                            }
                        ],
                    }
                ],
            },
        )

    def test_markdown_rendering_audit_metadata_counts_suppression_reasons(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "sample.pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "page_height": 792.0,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "metadata_reference_edges": [
                                    {
                                        "source_block_id": "txt_table_note",
                                        "relation": "table_note",
                                        "target_object_type": "table",
                                        "target_object_id": "tbl_001",
                                        "visible_render_policy": "metadata_only",
                                    }
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_table_note",
                                "text": "Already owned table note.",
                            },
                            {
                                "block_type": "structure_template",
                                "block_id": "tpl_absorbed",
                                "structure_template_id": "tpl_absorbed",
                                "ownership_domain": "absorbed_by_business_table",
                                "semantic_role": "absorbed_structure_template_fragment",
                            },
                        ],
                    }
                ]
            },
        }

        api_main._attach_markdown_rendering_audit_metadata(document)

        audit = document["metadata"]["markdown_rendering_audit"]
        self.assertEqual(
            audit["reason_counts"],
            {
                "absorbed_structure_template": 1,
                "metadata_reference_edge_metadata_only": 1,
            },
        )
        self.assertEqual(
            audit["pages"][0]["reason_counts"],
            {
                "absorbed_structure_template": 1,
                "metadata_reference_edge_metadata_only": 1,
            },
        )

    def test_audit_log_includes_markdown_rendering_audit_summary(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "file_id": "file_001",
            "filename": "sample.pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "page_height": 792.0,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "metadata_reference_edges": [
                                    {
                                        "source_block_id": "txt_table_note",
                                        "relation": "table_note",
                                        "target_object_type": "table",
                                        "target_object_id": "tbl_001",
                                        "visible_render_policy": "metadata_only",
                                    }
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_table_note",
                                "text": "Already owned table note.",
                            },
                            {
                                "block_type": "structure_template",
                                "block_id": "tpl_absorbed",
                                "structure_template_id": "tpl_absorbed",
                                "ownership_domain": "absorbed_by_business_table",
                                "semantic_role": "absorbed_structure_template_fragment",
                            },
                        ],
                    }
                ]
            },
        }

        audit_log = api_main._build_audit_log(
            "run_001",
            "job_001",
            "completed",
            [document],
        )

        self.assertEqual(
            audit_log["markdown_rendering_audit_summary"],
            {
                "document_count": 1,
                "suppressed_block_count": 2,
                "reason_counts": {
                    "absorbed_structure_template": 1,
                    "metadata_reference_edge_metadata_only": 1,
                },
                "documents": [
                    {
                        "file_id": "file_001",
                        "filename": "sample.pdf",
                        "page_count": 1,
                        "suppressed_page_count": 1,
                        "suppressed_block_count": 2,
                        "reason_counts": {
                            "absorbed_structure_template": 1,
                            "metadata_reference_edge_metadata_only": 1,
                        },
                    }
                ],
            },
        )

    def test_full_markdown_does_not_emit_rendering_audit_metadata_by_default(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "sample.pdf",
            "source_type": "pdf",
            "metadata": {
                "page_count": 1,
                "markdown_rendering_audit": {
                    "suppressed_block_count": 1,
                    "pages": [
                        {
                            "page": 1,
                            "suppressed_blocks": [
                                {
                                    "block_id": "txt_table_note",
                                    "reason": "metadata_reference_edge_metadata_only",
                                }
                            ],
                        }
                    ],
                },
            },
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_body",
                                "text": "Visible body text.",
                                "bbox": [72, 80, 240, 96],
                            }
                        ],
                    }
                ]
            },
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("Visible body text.", markdown)
        self.assertNotIn("markdown_rendering_audit", markdown)
        self.assertNotIn("metadata_reference_edge_metadata_only", markdown)

    def test_full_markdown_exports_complete_recursive_toc_tree(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "eCTD实施指南.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 42, "parser_hint": "pdf"},
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq-001",
                    "title": "目 录",
                    "pages": [2, 3, 4],
                    "page_span": [2, 4],
                    "entry_count": 24,
                    "root_nodes": [
                        {
                            "outline_index": "2.0",
                            "text": "基本要求",
                            "page": 2,
                            "page_locator_value": 6,
                            "children": [
                                {
                                    "outline_index": f"2.{index}",
                                    "text": f"第 2.{index} 节",
                                    "page": 2,
                                    "page_locator_value": 6 + index,
                                    "children": [],
                                }
                                for index in range(1, 9)
                            ],
                        },
                        {
                            "outline_index": "3.0",
                            "text": "eCTD 申报资料中的编号管理",
                            "page": 3,
                            "page_locator_value": 11,
                            "children": [
                                {
                                    "outline_index": "3.1",
                                    "text": "原始编号的应用",
                                    "page": 3,
                                    "page_locator_value": 11,
                                    "children": [],
                                }
                            ],
                        },
                        {
                            "outline_index": "4.0",
                            "text": "文件组织结构",
                            "page": 3,
                            "page_locator_value": 13,
                            "children": [
                                {
                                    "outline_index": "4.1",
                                    "text": "模块一：行政文件和药品信息",
                                    "page": 3,
                                    "page_locator_value": 13,
                                    "children": [
                                        {
                                            "outline_index": "4.1.3",
                                            "text": "信封信息的准备",
                                            "page": 3,
                                            "page_locator_value": 14,
                                            "children": [],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "outline_index": "5.0",
                            "text": "特定类型提交的建议",
                            "page": 4,
                            "page_locator_value": 17,
                            "children": [],
                        },
                        {
                            "outline_index": "7.0",
                            "text": "对eCTD 申报资料文件的要求",
                            "page": 4,
                            "page_locator_value": 29,
                            "children": [
                                {
                                    "outline_index": "7.4",
                                    "text": "书签与超文本链接的要求",
                                    "page": 4,
                                    "page_locator_value": 31,
                                    "children": [],
                                }
                            ],
                        },
                    ],
                }
            ],
            "pages": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("### 解析目录结构", markdown)
        for expected in (
            "2.1 第 2.1 节",
            "2.8 第 2.8 节",
            "3.0 eCTD 申报资料中的编号管理",
            "4.1.3 信封信息的准备",
            "5.0 特定类型提交的建议",
            "7.0 对eCTD 申报资料文件的要求",
            "7.4 书签与超文本链接的要求",
        ):
            self.assertIn(expected, markdown)
        self.assertIn("目录页: 2-4", markdown)
        self.assertIn("定位页码: 29", markdown)

    def test_ui_markdown_surfaces_toc_summary_for_single_pdf(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "eCTD实施指南.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 42, "parser_hint": "pdf"},
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq-001",
                    "title": "目 录",
                    "pages": [2, 3, 4],
                    "entry_count": 47,
                    "root_nodes": [
                        {"outline_index": "3.0", "text": "eCTD 申报资料中的编号管理", "children": []},
                        {"outline_index": "5.0", "text": "特定类型提交的建议", "children": []},
                        {"outline_index": "7.0", "text": "对eCTD 申报资料文件的要求", "children": []},
                    ],
                }
            ],
            "pages": [],
            "text": "",
        }

        markdown = api_main._build_ui_markdown([document], [], [])

        self.assertIn("## 解析目录结构", markdown)
        self.assertIn("eCTD实施指南.pdf", markdown)
        self.assertIn("目录项 47 条", markdown)
        self.assertIn("3.0 eCTD 申报资料中的编号管理", markdown)
        self.assertIn("5.0 特定类型提交的建议", markdown)
        self.assertIn("7.0 对eCTD 申报资料文件的要求", markdown)
        self.assertNotIn("请下载完整解析 Markdown 查看", markdown)

    def test_full_markdown_renders_body_blocks_without_page_or_debug_metadata(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "sample.pdf",
            "source_type": "pdf",
            "metadata": {
                "page_count": 2,
                "parser_hint": "pdf",
                "pdf_link_action_page_records": [
                    {
                        "page": 1,
                        "link_annotation_count": 2,
                        "link_action_kinds": ["/URI", "/GoTo"],
                        "link_annotation_xrefs": [31, 32],
                    }
                ],
                "pdf_bookmark_uri_targets": ["https://example.test/guidance"],
            },
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "Before image.",
                                "bbox": [72, 80, 200, 92],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_p1_001",
                                "image_id": "img_p1_001",
                                "title": "Figure 1: Main folder",
                                "caption_text": "Figure 1: Main folder",
                                "bbox": [100, 120, 300, 260],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "Before table.",
                                "bbox": [72, 300, 200, 312],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": "Table 1: Folder mapping",
                                "bbox": [72, 330, 500, 420],
                                "display_grid": [
                                    ["Folder", "Document type"],
                                    ["admin", "Cover letter"],
                                    ["clinical", "Protocol"],
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "After table. See https://example.test/page.",
                                "bbox": [72, 450, 200, 462],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 5}],
            "table_asts": [],
            "image_blocks": [],
            "text": "fallback text should not be needed",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("### 正文结构化内容", markdown)
        self.assertNotIn("### PDF Pages", markdown)
        self.assertNotIn("#### Page 1", markdown)
        self.assertIn("Before image.", markdown)
        self.assertIn("![Figure 1: Main folder](#img_p1_001)", markdown)
        self.assertNotIn("_图片占位", markdown)
        self.assertNotIn("image_id:", markdown)
        self.assertNotIn("bbox:", markdown)
        self.assertIn("**Table 1: Folder mapping**", markdown)
        self.assertIn("| Folder | Document type |", markdown)
        self.assertIn("| admin | Cover letter |", markdown)
        self.assertIn("After table. See [https://example.test/page](https://example.test/page).", markdown)
        self.assertNotIn("### PDF 超链接", markdown)
        self.assertNotIn("Page 1: 2 link annotation(s); actions: /GoTo, /URI; xrefs: 31, 32", markdown)
        self.assertNotIn("https://example.test/guidance", markdown)

        self.assertLess(markdown.index("Before image."), markdown.index("![Figure 1: Main folder](#img_p1_001)"))
        self.assertLess(markdown.index("![Figure 1: Main folder](#img_p1_001)"), markdown.index("Before table."))
        self.assertLess(markdown.index("Before table."), markdown.index("**Table 1: Folder mapping**"))
        self.assertLess(markdown.index("**Table 1: Folder mapping**"), markdown.index("| Folder | Document type |"))

    def test_neutral_image_markdown_projects_embedded_figure_text_without_nearby_body_duplication(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "chart.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "The questionnaire compared print and digital preferences.",
                                "bbox": [54, 84, 540, 96],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_p1_001",
                                "image_id": "img_p1_001",
                                "figure_ref": "Figure 1",
                                "bbox": [90, 120, 520, 360],
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
                    "bbox": [90, 120, 520, 360],
                    "figure_ref": "Figure 1",
                    "caption_text": "",
                    "embedded_text": "Q11 What factors influence your choice of print? Convenience Reading experience 0% 10% 20%",
                    "embedded_text_source": "ocr",
                    "embedded_text_confidence": 0.86,
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": "Q11 What factors influence your choice of print? Convenience Reading experience 0% 10% 20%",
                            "source": "ocr",
                            "confidence": 0.86,
                        },
                        {
                            "role": "nearby_context",
                            "text": "The questionnaire compared print and digital preferences.",
                            "relation": "above",
                            "gap": 18.0,
                            "block_id": "txt_p1_001",
                        },
                    ],
                    "content_text": (
                        "Q11 What factors influence your choice of print? Convenience Reading experience 0% 10% 20%\n"
                        "The questionnaire compared print and digital preferences."
                    ),
                    "image_kind_guess": "textual_image",
                    "content_signals": {
                        "has_caption": False,
                        "has_embedded_text": True,
                        "embedded_text_source": "ocr",
                        "embedded_text_confidence": 0.86,
                        "has_nearby_context": True,
                    },
                }
            ],
            "pages": [{"page_number": 1, "block_count": 2}],
            "text": "",
        }

        markdown = "\n".join(api_main._build_document_body_markdown_sections(document, embed_images=False))

        self.assertIn("![Figure 1](#img_p1_001)", markdown)
        self.assertIn("Q11 What factors influence your choice of print?", markdown)
        self.assertIn("Convenience Reading experience", markdown)
        self.assertEqual(
            markdown.count("The questionnaire compared print and digital preferences."),
            1,
            msg="nearby body context should remain in body flow, not be duplicated as figure semantic text",
        )

    def test_neutral_image_markdown_orders_below_caption_after_embedded_text(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "below-caption.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "image",
                                "block_id": "img_p1_001",
                                "image_id": "img_p1_001",
                                "figure_ref": "Figure 1",
                                "bbox": [80, 80, 520, 300],
                            }
                        ],
                    }
                ]
            },
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "page": 1,
                    "bbox": [80, 80, 520, 300],
                    "figure_ref": "Figure 1",
                    "caption_text": "Figure 1 Chart caption.",
                    "content_segments": [
                        {
                            "role": "caption",
                            "text": "Figure 1 Chart caption.",
                            "relation": "below",
                            "gap": 8.0,
                        },
                        {
                            "role": "embedded_text",
                            "text": "Axis label 0 10 20 Series A",
                            "source": "ocr",
                            "confidence": 0.9,
                        },
                    ],
                    "content_text": "Figure 1 Chart caption.\nAxis label 0 10 20 Series A",
                    "image_kind_guess": "captioned_textual_figure",
                }
            ],
            "pages": [{"page_number": 1, "block_count": 1}],
            "text": "",
        }

        markdown = "\n".join(api_main._build_document_body_markdown_sections(document, embed_images=False))

        content_after_anchor = markdown.split("](#img_p1_001)", 1)[1]
        self.assertLess(
            content_after_anchor.index("Axis label 0 10 20 Series A"),
            content_after_anchor.index("Figure 1 Chart caption."),
        )

    def test_neutral_image_markdown_suppresses_embedded_text_owned_by_structured_table(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "image-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "image",
                                "block_id": "img_p1_001",
                                "image_id": "img_p1_001",
                                "figure_ref": "Figure 1",
                                "bbox": [40, 40, 500, 320],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "bbox": [60, 80, 480, 300],
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
                    "bbox": [40, 40, 500, 320],
                    "figure_ref": "Figure 1",
                    "caption_text": "",
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": "Service Stage Function Name Explanation Expected Benefit Project creation management",
                            "source": "ocr",
                            "confidence": 0.92,
                        }
                    ],
                    "content_text": "Service Stage Function Name Explanation Expected Benefit Project creation management",
                    "image_kind_guess": "path_screenshot",
                }
            ],
            "table_asts": [
                {
                    "block_type": "table",
                    "table_id": "tbl_001",
                    "page": 1,
                    "bbox": [60, 80, 480, 300],
                    "display_grid": [
                        ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                        ["Project creation", "management", "Select type", "Improves workflow"],
                    ],
                    "detection_source": "visual_structure_grid",
                }
            ],
            "pages": [{"page_number": 1, "block_count": 2}],
            "text": "",
        }

        markdown = "\n".join(api_main._build_document_body_markdown_sections(document, embed_images=False))

        self.assertIn("| Service Stage | Function Name | Explanation | Expected Benefit |", markdown)
        self.assertEqual(markdown.count("Service Stage"), 1)
        self.assertNotIn("Project creation management", markdown)

    def test_evidence_only_chart_image_text_is_suppressed_when_structured_table_owns_region(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "image-table-evidence.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "image",
                                "block_id": "img_p1_001",
                                "image_id": "img_p1_001",
                                "figure_ref": "Figure 1",
                                "bbox": [56, 350, 555, 628],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "bbox": [56, 350, 555, 628],
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
                    "bbox": [56, 350, 555, 628],
                    "figure_ref": "Figure 1",
                    "embedded_text": (
                        "Temperature Kinematic viscosity 0 1.793E-06 25 8.930E-07 "
                        "1 1.732E-06 26 8.760E-07"
                    ),
                    "embedded_text_source": "ocr-image-evidence",
                    "text_recovery": {
                        "source": "ocr-image-evidence",
                        "evidence_only": True,
                    },
                    "content_segments": [
                        {
                            "role": "embedded_text",
                            "text": (
                                "Temperature Kinematic viscosity 0 1.793E-06 25 8.930E-07 "
                                "1 1.732E-06 26 8.760E-07"
                            ),
                            "source": "ocr-image-evidence",
                        }
                    ],
                    "image_kind_guess": "chart_figure",
                }
            ],
            "table_asts": [
                {
                    "block_type": "table",
                    "table_id": "tbl_001",
                    "page": 1,
                    "bbox": [56, 350, 555, 628],
                    "display_grid": [
                        ["Temperature", "Kinematic viscosity", "Temperature", "Kinematic viscosity"],
                        ["0", "1.793E-06", "25", "8.930E-07"],
                        ["1", "1.732E-06", "26", "8.760E-07"],
                    ],
                    "detection_source": "embedded_image_ocr",
                }
            ],
            "pages": [{"page_number": 1, "block_count": 2}],
            "text": "",
        }

        markdown = "\n".join(api_main._build_document_body_markdown_sections(document, embed_images=False))

        self.assertIn("| Temperature | Kinematic viscosity | Temperature | Kinematic viscosity |", markdown)
        self.assertEqual(markdown.count("Temperature"), 2)
        self.assertNotIn("Temperature Kinematic viscosity 0 1.793E-06", markdown)

    def test_ind_review_markdown_suppresses_image_embedded_text_but_keeps_caption(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "ectd-tech.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "image",
                                "block_id": "img_p1_001",
                                "image_id": "img_p1_001",
                                "figure_ref": "图11",
                                "caption_text": "图11.信封骨架文件示例",
                                "bbox": [80, 80, 520, 280],
                            }
                        ],
                    }
                ]
            },
            "image_blocks": [
                {
                    "block_type": "image",
                    "image_id": "img_p1_001",
                    "page": 1,
                    "bbox": [80, 80, 520, 280],
                    "figure_ref": "图11",
                    "caption_text": "图11.信封骨架文件示例",
                    "content_segments": [
                        {
                            "role": "embedded_code",
                            "text": "<cn-envelope><application-id>x202112345</application-id></cn-envelope>",
                            "source": "ocr",
                            "confidence": 0.88,
                        },
                        {
                            "role": "caption",
                            "text": "图11.信封骨架文件示例",
                            "relation": "below",
                        },
                    ],
                    "content_text": "<cn-envelope><application-id>x202112345</application-id></cn-envelope>",
                    "image_kind_guess": "code_or_markup_figure",
                    "figure_semantics": {
                        "semantic_type": "code_or_markup_figure",
                        "language": "xml",
                        "content_kind": "embedded_markup",
                    },
                }
            ],
            "pages": [{"page_number": 1, "block_count": 1}],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("![图11.信封骨架文件示例](#img_p1_001)", markdown)
        self.assertIn("图11.信封骨架文件示例", markdown)
        self.assertNotIn("<cn-envelope>", markdown)
        self.assertNotIn("application-id", markdown)

    def test_ind_review_markdown_embeds_pdf_image_crops_without_embedded_ocr_text(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
            fitz = importlib.import_module("fitz")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"required module unavailable in this environment: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "review-image-source.pdf"
            pdf = fitz.open()
            page = pdf.new_page(width=160, height=160)
            page.draw_rect(fitz.Rect(30, 30, 130, 110), color=(0, 0, 1), fill=(0, 0, 1))
            pdf.save(pdf_path)
            pdf.close()

            document = {
                "filename": "review-image-source.pdf",
                "source_type": "pdf",
                "source_path": str(pdf_path),
                "metadata": {"page_count": 1, "parser_hint": "pdf"},
                "document_ast": {
                    "pages": [
                        {
                            "page": 1,
                            "blocks": [
                                {
                                    "block_type": "image",
                                    "block_id": "img_p1_001",
                                    "image_id": "img_p1_001",
                                    "caption_text": "Figure 1 Review crop",
                                    "bbox": [30, 30, 130, 110],
                                }
                            ],
                        }
                    ]
                },
                "image_blocks": [
                    {
                        "block_type": "image",
                        "image_id": "img_p1_001",
                        "page": 1,
                        "bbox": [30, 30, 130, 110],
                        "caption_text": "Figure 1 Review crop",
                        "content_segments": [
                            {
                                "role": "embedded_text",
                                "text": "internal OCR text should stay in AST evidence only",
                                "source": "ocr",
                            },
                            {
                                "role": "caption",
                                "text": "Figure 1 Review crop",
                                "relation": "below",
                            },
                        ],
                    }
                ],
                "pages": [{"page_number": 1, "block_count": 1}],
                "table_asts": [],
                "text": "",
            }

            markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

            self.assertIn("![Figure 1 Review crop](data:image/png;base64,", markdown)
            self.assertNotIn("![Figure 1 Review crop](#img_p1_001)", markdown)
            self.assertIn("Figure 1 Review crop", markdown)
            self.assertNotIn("internal OCR text should stay in AST evidence only", markdown)

    def test_ind_review_markdown_prefers_display_grid_over_semantic_compaction_for_tables(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "ectd-tech.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "table",
                                "block_id": "tbl_006",
                                "table_id": "tbl_006",
                                "title": "表 6. 信封元素",
                                "bbox": [80, 120, 520, 500],
                            }
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "block_type": "table",
                    "table_id": "tbl_006",
                    "title": "表 6. 信封元素",
                    "display_grid": [
                        ["级别", "信封元素属性", "描述", "受控词汇"],
                        ["申请级别", "application-id", "申请编号", None],
                        [None, "application-type", "申请类型", "有"],
                        ["注册行为级别", "related-sequence", "相关序列", None],
                        [None, "regulatory-activity-type", "注册行为类型", "有"],
                    ],
                    "data_grid": [
                        ["申请级别", "application-id", "申请编号", None],
                        ["申请级别", "application-type", "申请类型", "有"],
                        ["注册行为级别", "related-sequence", "相关序列", None],
                        ["注册行为级别", "regulatory-activity-type", "注册行为类型", "有"],
                    ],
                    "semantic_grid": [
                        [
                            "申请级别 注册行为级别",
                            "信封元素属性 application-id application-type related-sequence",
                            "描述 申请编号 申请类型 相关序列",
                            "受控词汇 有",
                        ],
                        [None, "regulatory-activity-type", "注册行为类型", "有"],
                    ],
                    "semantic_projection_v2": {
                        "source": "table_semantic_projection_v2",
                        "table_family": "comparison_matrix",
                        "multiline_schema_header_projection": {},
                        "rowspan_key_repetition_compaction": {},
                        "multicolumn_wrapped_record_compaction": {},
                    },
                }
            ],
            "pages": [{"page_number": 1, "block_count": 1}],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("| 级别 | 信封元素属性 | 描述 | 受控词汇 |", markdown)
        self.assertIn("| 申请级别 | application-id | 申请编号 |  |", markdown)
        self.assertIn("|  | application-type | 申请类型 | 有 |", markdown)
        self.assertNotIn("| 申请级别 注册行为级别 |", markdown)
        self.assertNotIn("信封元素属性 application-id application-type related-sequence", markdown)

    def test_full_markdown_merges_wrapped_body_lines_and_breaks_on_indented_paragraph_starts(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "two-column-body.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "This paragraph starts with an indented first line.",
                                "bbox": [84, 80, 300, 92],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "It continues after a sentence boundary.",
                                "bbox": [72, 94, 300, 106],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "More wrapped content follows in the same paragraph.",
                                "bbox": [72, 108, 300, 120],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_004",
                                "text": "This is a new paragraph because its first line is indented.",
                                "bbox": [84, 136, 300, 148],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_005",
                                "text": "It should remain attached to that new paragraph.",
                                "bbox": [72, 150, 300, 162],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 5}],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(
            "This paragraph starts with an indented first line. "
            "It continues after a sentence boundary. "
            "More wrapped content follows in the same paragraph.",
            markdown,
        )
        self.assertIn(
            "same paragraph.\n\n"
            "This is a new paragraph because its first line is indented. "
            "It should remain attached to that new paragraph.",
            markdown,
        )
        self.assertNotIn(
            "This paragraph starts with an indented first line.\n\n"
            "It continues after a sentence boundary.",
            markdown,
        )

    def test_full_markdown_breaks_between_single_line_indented_body_paragraphs(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "single-line-indented-paragraphs.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "First single-line body paragraph ends here.",
                                "bbox": [84, 80, 320, 92],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "Second single-line body paragraph also starts indented.",
                                "bbox": [84, 112, 340, 124],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 2}],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(
            "First single-line body paragraph ends here.\n\n"
            "Second single-line body paragraph also starts indented.",
            markdown,
        )
        self.assertNotIn(
            "First single-line body paragraph ends here. "
            "Second single-line body paragraph also starts indented.",
            markdown,
        )

    def test_full_markdown_keeps_literature_front_matter_labels_standalone(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "literature-front-matter.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "H I G H L I G H T S",
                                "bbox": [72, 80, 210, 92],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "A compact summary line should not be merged into the label.",
                                "bbox": [72, 96, 360, 108],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "a r t i c l e i n f o",
                                "bbox": [72, 132, 210, 144],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_004",
                                "text": "Article history:",
                                "bbox": [72, 148, 210, 160],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_005",
                                "text": "a b s t r a c t",
                                "bbox": [72, 184, 210, 196],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_006",
                                "text": "This abstract body starts after the standalone label.",
                                "bbox": [72, 200, 360, 212],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_007",
                                "text": "References",
                                "bbox": [72, 236, 150, 248],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_008",
                                "text": "Smith, J. A cited article title.",
                                "bbox": [72, 252, 360, 264],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 8}],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        for label, following in (
            ("H I G H L I G H T S", "A compact summary line"),
            ("a r t i c l e i n f o", "Article history:"),
            ("a b s t r a c t", "This abstract body"),
            ("References", "Smith, J."),
        ):
            self.assertIn(f"{label}\n\n{following}", markdown)
            self.assertNotIn(f"{label} {following}", markdown)

    def test_full_markdown_keeps_literature_section_titles_standalone(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "literature-section-headings.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "Introduction",
                                "bbox": [72, 80, 130, 92],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "Information extraction starts the first paragraph.",
                                "bbox": [72, 96, 330, 108],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "Related work",
                                "bbox": [72, 132, 140, 144],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_004",
                                "text": "Traditional pipeline methods",
                                "bbox": [72, 148, 210, 160],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_005",
                                "text": "In the traditional pipeline methods, relation extraction starts here.",
                                "bbox": [72, 164, 390, 176],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 5}],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("Introduction\n\nInformation extraction starts", markdown)
        self.assertIn("Related work\n\nTraditional pipeline methods\n\nIn the traditional pipeline methods", markdown)
        self.assertNotIn("Introduction Information extraction starts", markdown)
        self.assertNotIn("Related work Traditional pipeline methods", markdown)

    def test_full_markdown_keeps_numbered_reference_continuations_continuous(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "references.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "References",
                                "semantic_role": "reference_heading",
                                "bbox": [72, 80, 150, 92],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "i. Smith J. First reference title. J",
                                "semantic_role": "reference_entry",
                                "bbox": [72, 96, 330, 108],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "Example Med. 2024;1:1-2.",
                                "semantic_role": "reference_entry",
                                "bbox": [90, 110, 330, 122],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_004",
                                "text": "ii. Jones K. Second reference title.",
                                "semantic_role": "reference_entry",
                                "bbox": [72, 124, 330, 136],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 4}],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("i. Smith J. First reference title. J Example Med. 2024;1:1-2.", markdown)
        self.assertIn("ii. Jones K. Second reference title.", markdown)
        self.assertNotIn("title. J\n\nExample Med", markdown)
        self.assertNotIn("1-2. ii. Jones", markdown)

    def test_full_markdown_repairs_soft_hyphenation_inside_merged_reference_entries(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "references.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "References",
                                "semantic_role": "reference_heading",
                                "bbox": [72, 80, 150, 92],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "i. Smith J. Multi-task active learn-",
                                "semantic_role": "reference_entry",
                                "bbox": [72, 96, 330, 108],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "ing for named entity recognition.",
                                "semantic_role": "reference_entry",
                                "bbox": [90, 110, 330, 122],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 3}],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("i. Smith J. Multi-task active learning for named entity recognition.", markdown)
        self.assertNotIn("learn- ing", markdown)

    def test_full_markdown_reconstructs_scientific_notation_inline_formula_and_float_notes_as_continuous_paragraphs(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "math-and-floats.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "are extremely small (e.g., less than 10-5).",
                                "display_text": "are extremely small (e.g., less than 10-5).",
                                "bbox": [72, 80, 380, 92],
                                "inline_formula_spans": [
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [230, 80, 278, 92],
                                        "content": "10-5",
                                        "latex_text": r"10^{-5}",
                                        "latex_confidence": 0.96,
                                        "formula_complexity": "inline_formula",
                                        "ocr_candidate": True,
                                    }
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "k ( d )= exp ( d / σ ) is used in this paper, and u is the centroid mean.",
                                "display_text": "k ( d )= exp ( d / σ ) is used in this paper, and u is the centroid mean.",
                                "bbox": [72, 100, 500, 112],
                                "inline_formula_spans": [
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [72, 100, 240, 112],
                                        "content": "k ( d )= exp ( d / σ )",
                                        "latex_text": r"k(d)=\exp(d/\sigma)",
                                        "latex_confidence": 0.93,
                                        "formula_complexity": "inline_formula",
                                        "ocr_candidate": True,
                                    },
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [310, 100, 324, 112],
                                        "content": "u",
                                        "latex_text": "u",
                                        "latex_confidence": 0.9,
                                        "formula_complexity": "inline_symbol",
                                        "ocr_candidate": False,
                                    },
                                ],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_p1_001",
                                "table_id": "tbl_p1_001",
                                "title": "Table 1: Sample results",
                                "bbox": [72, 140, 480, 220],
                                "display_grid": [
                                    ["A", "B"],
                                    ["1", "2"],
                                ],
                                "content_segments": [
                                    {"role": "note", "text": "Note line 1."},
                                    {"role": "note", "text": "Note line 2 continues the explanation."},
                                ],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_p1_001",
                                "image_id": "img_p1_001",
                                "title": "Figure 1: Sample image",
                                "caption_text": "Figure 1: Sample image",
                                "bbox": [72, 250, 300, 360],
                                "content_segments": [
                                    {"role": "caption", "text": "Legend line 1."},
                                    {"role": "legend", "text": "Legend line 2 continues the explanation."},
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "Body text after the float blocks.",
                                "display_text": "Body text after the float blocks.",
                                "bbox": [72, 390, 300, 402],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 5}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(r"are extremely small (e.g., less than $10^{-5}$).", markdown)
        self.assertNotIn("10-5", markdown)
        self.assertIn(r"$k(d)=\exp(d/\sigma)$ is used in this paper, and $u$ is the centroid mean.", markdown)
        self.assertNotIn("$u$sed", markdown)
        self.assertIn("Note line 1. Note line 2 continues the explanation.", markdown)
        self.assertIn("Legend line 1. Legend line 2 continues the explanation.", markdown)
        self.assertIn("Note line 1. Note line 2 continues the explanation.\n\n![Figure 1: Sample image]", markdown)
        self.assertIn("Legend line 1. Legend line 2 continues the explanation.\n\nBody text after the float blocks.", markdown)
        self.assertNotIn("10-5", markdown)
        self.assertNotIn("$u$sed", markdown)

    def test_full_markdown_uses_unified_block_boundaries_for_body_floats_references_and_toc(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "unified-boundaries.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 2, "parser_hint": "pdf"},
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq_001",
                    "root_nodes": [
                        {"outline_index": "1.0", "text": "Introduction", "children": []},
                    ],
                }
            ],
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "toc",
                                "block_id": "toc_001",
                                "toc_id": "toc_001",
                                "bbox": [72, 60, 480, 130],
                                "entries": [
                                    {"outline_index": "1", "text": "Introduction", "page_locator": "2"},
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "H I G H L I G H T S",
                                "bbox": [72, 150, 220, 162],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "A short highlight should remain below the label.",
                                "bbox": [72, 166, 420, 178],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "1. Introduction",
                                "bbox": [72, 210, 220, 222],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_004",
                                "text": "This paragraph begins after the heading and continues",
                                "bbox": [84, 236, 430, 248],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_005",
                                "text": "on the next physical line in the same paragraph.",
                                "bbox": [72, 250, 430, 262],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": "Table 1 Example table",
                                "bbox": [72, 300, 460, 380],
                                "display_grid": [["A", "B"], ["1", "2"]],
                                "content_segments": [
                                    {"role": "note", "text": "Note line one."},
                                    {"role": "note", "text": "Note line two is the same note."},
                                ],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_001",
                                "image_id": "img_001",
                                "title": "Fig. 1 Example figure",
                                "caption_text": "Fig. 1 Example figure",
                                "bbox": [72, 410, 360, 500],
                                "content_segments": [
                                    {"role": "caption", "text": "Fig. 1 Example figure"},
                                    {"role": "legend", "text": "Legend line two is the same legend."},
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_006",
                                "text": "A new body paragraph after the figure starts here.",
                                "bbox": [84, 530, 430, 542],
                            },
                        ],
                    },
                    {
                        "page": 2,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p2_001",
                                "text": "References",
                                "bbox": [72, 80, 150, 92],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p2_002",
                                "text": "1. Smith J. A reference title. J",
                                "semantic_role": "reference_entry",
                                "reference_entry_index": 1,
                                "reference_number": "1",
                                "reference_entry_start": True,
                                "bbox": [72, 96, 360, 108],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p2_003",
                                "text": "Med. 2024;10:12-18.",
                                "semantic_role": "reference_entry",
                                "reference_entry_index": 1,
                                "reference_number": "1",
                                "reference_continuation": True,
                                "bbox": [90, 110, 360, 122],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p2_004",
                                "text": "2. Jones K. A second reference.",
                                "semantic_role": "reference_entry",
                                "reference_entry_index": 2,
                                "reference_number": "2",
                                "reference_entry_start": True,
                                "bbox": [72, 124, 360, 136],
                            },
                        ],
                    },
                ]
            },
            "pages": [{"page_number": 1, "block_count": 8}, {"page_number": 2, "block_count": 4}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertNotIn("Introduction | 2", markdown)
        self.assertIn("H I G H L I G H T S\n\nA short highlight", markdown)
        self.assertIn("#### 1. Introduction", markdown)
        self.assertIn(
            "This paragraph begins after the heading and continues "
            "on the next physical line in the same paragraph.",
            markdown,
        )
        self.assertIn("Note line one. Note line two is the same note.\n\n![Fig. 1 Example figure]", markdown)
        self.assertIn("Fig. 1 Example figure Legend line two is the same legend.\n\nA new body paragraph", markdown)
        self.assertIn("References\n\n1. Smith J. A reference title. J Med. 2024;10:12-18.", markdown)
        self.assertIn("Med. 2024;10:12-18.\n\n2. Jones K. A second reference.", markdown)
        self.assertNotIn("H I G H L I G H T S A short highlight", markdown)
        self.assertNotIn("Note line one. Note line two is the same note. ![Fig. 1", markdown)
        self.assertNotIn("same legend. A new body paragraph", markdown)

    def test_markdown_text_block_boundary_kind_classifies_common_pdf_roles(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        toc_lookup = {
            "1": [{"title": "Introduction", "depth": 1}],
            "3.2.S.1": [{"title": "General Information", "depth": 3}],
            "III": [{"title": "Study Design", "depth": 1}],
            "A": [{"title": "Inclusion Criteria", "depth": 2}],
            "APPENDIX B": [{"title": "IND Table of Contents", "depth": 2}],
        }

        self.assertEqual(
            api_main._markdown_text_block_boundary_kind(
                {"block_type": "text", "text": "H I G H L I G H T S"},
                toc_lookup,
            ),
            "standalone_label",
        )
        self.assertEqual(
            api_main._markdown_text_block_boundary_kind(
                {"block_type": "text", "text": "1. Introduction"},
                toc_lookup,
            ),
            "standalone_heading",
        )
        for text in (
            "3.2.S.1 General Information",
            "III Study Design",
            "A Inclusion Criteria",
            "APPENDIX B IND Table of Contents",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    api_main._markdown_text_block_boundary_kind(
                        {"block_type": "text", "text": text},
                        toc_lookup,
                    ),
                    "standalone_heading",
                )
        self.assertEqual(
            api_main._markdown_text_block_boundary_kind(
                {"block_type": "text", "semantic_role": "reference_entry", "text": "1. Smith J."},
                toc_lookup,
            ),
            "reference_entry",
        )
        self.assertEqual(
            api_main._markdown_text_block_boundary_kind(
                {"block_type": "text", "semantic_role": "footnote", "text": "1 Note text"},
                toc_lookup,
            ),
            "footnote",
        )
        self.assertEqual(
            api_main._markdown_text_block_boundary_kind(
                {"block_type": "text", "semantic_role": "publication_footer", "text": "ISSN 1234"},
                toc_lookup,
            ),
            "publication_metadata",
        )
        self.assertEqual(
            api_main._markdown_text_block_boundary_kind(
                {"block_type": "text", "text": "A continuous body paragraph"},
                toc_lookup,
            ),
            "body",
        )
        self.assertEqual(
            api_main._markdown_text_block_boundary_kind(
                {
                    "block_type": "text",
                    "text": "1. The applicant should provide a description of the sequence and related submission files.",
                    "section_context": {"outline_index": "4.1", "section_title": "Administrative Information"},
                },
                toc_lookup,
            ),
            "body",
        )

    def test_markdown_page_block_arbitration_covers_common_pdf_page_roles(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        toc_lookup = {"1": [{"title": "Introduction", "depth": 1}]}
        page_context = {
            "page_number": 1,
            "page_height": 800.0,
            "toc_heading_lookup": toc_lookup,
            "float_owned_text_block_ids": {"txt_table_note", "txt_figure_legend"},
            "equation_source_block_ids": {"txt_equation_source"},
        }

        cases = [
            (
                {"block_type": "text", "block_id": "txt_header", "semantic_role": "running_header", "text": "Journal title", "bbox": [72, 18, 280, 30]},
                ("page_header", False, "header"),
            ),
            (
                {"block_type": "image", "block_id": "logo_001", "semantic_role": "logo", "bbox": [32, 18, 58, 42]},
                ("logo", False, "header"),
            ),
            (
                {"block_type": "text", "block_id": "txt_page", "semantic_role": "page_number", "text": "12", "bbox": [300, 770, 315, 782]},
                ("page_number", False, "footer"),
            ),
            (
                {"block_type": "toc", "block_id": "toc_001", "bbox": [72, 90, 520, 170]},
                ("toc", False, "body"),
            ),
            (
                {"block_type": "table", "block_id": "tbl_001", "bbox": [72, 190, 520, 270]},
                ("table", True, "body"),
            ),
            (
                {"block_type": "text", "block_id": "txt_table_note", "semantic_role": "table_note", "text": "Note: Values are mean.", "bbox": [72, 274, 520, 288]},
                ("table_note", False, "body"),
            ),
            (
                {"block_type": "image", "block_id": "img_001", "bbox": [72, 310, 420, 430]},
                ("figure", True, "body"),
            ),
            (
                {"block_type": "text", "block_id": "txt_figure_legend", "semantic_role": "figure_legend", "text": "Fig. 1 Study flow.", "bbox": [72, 434, 420, 448]},
                ("figure_legend", False, "body"),
            ),
            (
                {"block_type": "equation", "block_id": "eq_001", "bbox": [120, 470, 420, 500]},
                ("display_equation", True, "body"),
            ),
            (
                {"block_type": "algorithm", "block_id": "alg_001", "text": "Algorithm 1. Example", "bbox": [72, 510, 520, 560]},
                ("algorithm_pseudocode", True, "body"),
            ),
            (
                {"block_type": "text", "block_id": "txt_equation_source", "text": "x i = y", "bbox": [120, 470, 420, 500]},
                ("equation_source_text", False, "body"),
            ),
            (
                {"block_type": "text", "block_id": "txt_note", "semantic_role": "footnote", "text": "1 Author note.", "bbox": [72, 720, 420, 734]},
                ("footnote", False, "footer"),
            ),
            (
                {"block_type": "text", "block_id": "txt_body", "text": "This is a body paragraph with $x_i$ inline.", "bbox": [72, 540, 520, 554]},
                ("body", True, "body"),
            ),
        ]

        for block, expected in cases:
            with self.subTest(block_id=block["block_id"]):
                decision = api_main._markdown_page_block_role(block, page_context)
                self.assertEqual((decision["role"], decision["render_in_main_flow"], decision["zone"]), expected)

    def test_full_markdown_labels_algorithm_blocks_as_algorithm_pseudocode(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "algorithm-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 792,
                        "blocks": [
                            {
                                "block_type": "algorithm",
                                "block_id": "alg_001",
                                "algorithm_id": "alg_001",
                                "algorithm_ref": "Algorithm 1",
                                "title": "Algorithm 1. Example procedure",
                                "semantic_role": "algorithm_pseudocode",
                                "unit_role": "algorithm",
                                "display_text": "Algorithm 1. Example procedure\n1. Initialize $W$",
                                "text": "Algorithm 1. Example procedure\n1. Initialize W",
                                "bbox": [72, 100, 520, 180],
                            }
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("**Algorithm pseudocode: Algorithm 1. Example procedure**", markdown)
        self.assertIn("1. Initialize $W$", markdown)
        self.assertNotIn("**Algorithm: Algorithm 1. Example procedure**", markdown)

    def test_full_markdown_projects_inline_formula_candidates_with_unicode_minus_variants(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "unicode-minus-formula.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "k ( d )= exp (− d / s ) has been used in many methods.",
                                "display_text": "k ( d )= exp (− d / s ) has been used in many methods.",
                                "bbox": [72, 100, 480, 112],
                                "inline_formula_spans": [
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [72, 100, 240, 112],
                                        "content": "k ( d )= exp ( - d / s )",
                                        "latex_text": r"k(d)=\exp(-d/s)",
                                        "latex_confidence": 0.93,
                                        "formula_complexity": "inline_formula",
                                        "ocr_candidate": True,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(r"$k(d)=\exp(-d/s)$ has been used in many methods.", markdown)
        self.assertNotIn("k ( d )= exp (− d / s ) has been used in many methods.", markdown)

    def test_full_markdown_renders_structured_footnote_refs_without_duplicate_continuations(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "footnote-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf", "footnote_count": 2},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "Allowed characters include underscore1 \"_\".",
                                "bbox": [72, 120, 460, 136],
                                "linked_footnote_ids": ["fn_p1_001"],
                                "footnote_refs": [
                                    {
                                        "marker": "1",
                                        "footnote_id": "fn_p1_001",
                                        "bbox": [320, 116, 325, 126],
                                    }
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_020",
                                "text": "1 Footnote first physical line",
                                "bbox": [72, 720, 460, 732],
                                "semantic_role": "footnote",
                                "footnote_id": "fn_p1_001",
                                "footnote_marker": "1",
                                "footnote_text": "Footnote first physical line continuation line",
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "5. References2",
                                "bbox": [72, 160, 460, 176],
                                "linked_footnote_ids": ["fn_p1_002"],
                                "footnote_refs": [
                                    {
                                        "marker": "2",
                                        "footnote_id": "fn_p1_002",
                                        "bbox": [156, 156, 161, 166],
                                    }
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_022",
                                "text": "2 Reference footnote",
                                "bbox": [72, 748, 460, 760],
                                "semantic_role": "footnote",
                                "footnote_id": "fn_p1_002",
                                "footnote_marker": "2",
                                "footnote_text": "Reference footnote",
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_021",
                                "text": "continuation line",
                                "bbox": [84, 734, 460, 746],
                                "semantic_role": "footnote_continuation",
                                "footnote_id": "fn_p1_001",
                                "footnote_marker": "1",
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 3}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn('Allowed characters include underscore[^1] "_".', markdown)
        self.assertIn("5. References[^2]", markdown)
        self.assertIn("[^1]: Footnote first physical line continuation line", markdown)
        self.assertIn("[^2]: Reference footnote", markdown)
        self.assertNotIn("\ncontinuation line\n", markdown)
        self.assertLess(markdown.index("5. References[^2]"), markdown.index("[^1]:"))

    def test_full_markdown_preserves_footnote_refs_inside_merged_body_paragraphs(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "merged-footnote-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf", "footnote_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "Allowed characters:",
                                "bbox": [72, 120, 240, 136],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "underscore1 \"_\". The path limit is 180 characters.",
                                "bbox": [72, 138, 460, 154],
                                "linked_footnote_ids": ["fn_p1_001"],
                                "footnote_refs": [
                                    {
                                        "marker": "1",
                                        "footnote_id": "fn_p1_001",
                                        "bbox": [146, 134, 151, 144],
                                    }
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_020",
                                "text": "1 Underscore is allowed for this submission package.",
                                "bbox": [72, 720, 460, 732],
                                "semantic_role": "footnote",
                                "footnote_id": "fn_p1_001",
                                "footnote_marker": "1",
                                "footnote_text": "Underscore is allowed for this submission package.",
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 3}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn('Allowed characters: underscore[^1] "_". The path limit is 180 characters.', markdown)
        self.assertIn("[^1]: Underscore is allowed for this submission package.", markdown)
        self.assertNotIn('underscore1 "_". [^1] The path limit', markdown)

    def test_full_markdown_embeds_pdf_image_crop_when_source_file_is_available(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
            fitz = importlib.import_module("fitz")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"required module unavailable in this environment: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "image-source.pdf"
            pdf = fitz.open()
            page = pdf.new_page(width=120, height=120)
            page.draw_rect(fitz.Rect(20, 20, 80, 80), color=(1, 0, 0), fill=(1, 0, 0))
            pdf.save(pdf_path)
            pdf.close()

            document = {
                "filename": "image-source.pdf",
                "source_type": "pdf",
                "source_path": str(pdf_path),
                "metadata": {"page_count": 1, "parser_hint": "pdf"},
                "document_ast": {
                    "pages": [
                        {
                            "page": 1,
                            "blocks": [
                                {
                                    "block_type": "image",
                                    "block_id": "img_p1_001",
                                    "image_id": "img_p1_001",
                                    "caption_text": "Figure 1: red square",
                                    "bbox": [20, 20, 80, 80],
                                }
                            ],
                        }
                    ]
                },
                "pages": [{"page_number": 1, "block_count": 1}],
                "image_blocks": [
                    {
                        "image_id": "img_p1_001",
                        "page": 1,
                        "caption_text": "Figure 1: red square",
                        "bbox": [20, 20, 80, 80],
                    }
                ],
                "table_asts": [],
                "text": "",
            }

            markdown = api_main._build_full_markdown([document])

            self.assertIn("![Figure 1: red square](data:image/png;base64,", markdown)
            self.assertNotIn("_图片占位", markdown)
            self.assertNotIn("image_id:", markdown)
            self.assertNotIn("bbox:", markdown)

    def test_neutral_body_markdown_can_render_image_anchor_without_embedded_crop(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
            fitz = importlib.import_module("fitz")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"required module unavailable in this environment: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "image-source.pdf"
            pdf = fitz.open()
            page = pdf.new_page(width=120, height=120)
            page.draw_rect(fitz.Rect(20, 20, 80, 80), color=(1, 0, 0), fill=(1, 0, 0))
            pdf.save(pdf_path)
            pdf.close()

            document = {
                "filename": "image-source.pdf",
                "source_type": "pdf",
                "source_path": str(pdf_path),
                "metadata": {"page_count": 1, "parser_hint": "pdf"},
                "document_ast": {
                    "pages": [
                        {
                            "page": 1,
                            "blocks": [
                                {
                                    "block_type": "image",
                                    "block_id": "img_p1_001",
                                    "image_id": "img_p1_001",
                                    "caption_text": "Figure 1: red square",
                                    "bbox": [20, 20, 80, 80],
                                }
                            ],
                        }
                    ]
                },
                "pages": [{"page_number": 1, "block_count": 1}],
                "image_blocks": [
                    {
                        "image_id": "img_p1_001",
                        "page": 1,
                        "caption_text": "Figure 1: red square",
                        "bbox": [20, 20, 80, 80],
                    }
                ],
                "table_asts": [],
                "text": "",
            }

            markdown = "\n".join(
                api_main._build_document_body_markdown_sections(document, embed_images=False)
            )

            self.assertIn("![Figure 1: red square](#img_p1_001)", markdown)
            self.assertNotIn("data:image/png;base64,", markdown)

    def test_body_continuation_misclassified_as_author_line_stays_in_reading_order(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "body-list.pdf",
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
                                "block_id": "txt_p1_001",
                                "semantic_role": "body_list_item",
                                "text": "1. Label the tubes with the sample code and either EA or EB for Evidence B.",
                                "bbox": [72, 500, 520, 512],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "semantic_role": "author_line",
                                "text": "All three samples will be digested by the restriction enzymes BamHI and HindIII.",
                                "bbox": [72, 514, 530, 526],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "semantic_role": "text_block",
                                "text": "Use a fresh pipet tip each time you add a reagent to a tube.",
                                "bbox": [72, 540, 520, 552],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = "\n".join(
            api_main._build_document_body_markdown_sections(document, embed_images=False)
        )

        self.assertIn("Label the tubes", markdown)
        self.assertIn("All three samples will be digested", markdown)
        self.assertIn("Use a fresh pipet tip", markdown)
        self.assertLess(markdown.index("Label the tubes"), markdown.index("All three samples will be digested"))
        self.assertLess(markdown.index("All three samples will be digested"), markdown.index("Use a fresh pipet tip"))

    def test_ind_review_markdown_profile_is_separate_from_legacy_full_snapshot(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "review-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "toc_sequences": [
                {
                    "toc_sequence_id": "toc_001",
                    "root_nodes": [
                        {"outline_index": "1", "text": "Study Overview", "children": []},
                    ],
                }
            ],
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 792,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_header",
                                "semantic_role": "running_header",
                                "text": "Company confidential header",
                                "bbox": [72, 18, 360, 30],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "1. Study Overview",
                                "bbox": [72, 80, 240, 94],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "This paragraph starts with an indented first line.",
                                "bbox": [84, 112, 430, 124],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "It continues on the next short physical line.",
                                "bbox": [72, 126, 430, 138],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": "Table 1 Toxicology summary",
                                "bbox": [72, 170, 520, 260],
                                "display_grid": [
                                    ["Endpoint", "Result"],
                                    ["NOAEL", "10 mg/kg"],
                                ],
                                "content_segments": [
                                    {"role": "note", "text": "Note line one."},
                                    {"role": "note", "text": "Note line two."},
                                ],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_001",
                                "image_id": "img_001",
                                "figure_ref": "Figure 1",
                                "caption_text": "Figure 1 Mean blood pressure",
                                "bbox": [72, 300, 460, 470],
                                "content_segments": [
                                    {"role": "caption", "text": "Figure 1 Mean blood pressure"},
                                    {"role": "legend", "text": "Values are mean S.E."},
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_footer",
                                "semantic_role": "page_number",
                                "text": "1",
                                "bbox": [300, 760, 310, 772],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 7}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        legacy_markdown = api_main._build_full_markdown([document])
        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("# IND Parse Snapshot (Full)", legacy_markdown)
        self.assertIn("### ", legacy_markdown)
        self.assertIn("# IND Review Markdown", review_markdown)
        self.assertIn("## review-sample.pdf", review_markdown)
        self.assertNotIn("# IND Parse Snapshot (Full)", review_markdown)
        self.assertNotIn("Company confidential header", review_markdown)
        self.assertNotIn("\n1\n", review_markdown)
        self.assertIn("#### 1. Study Overview", review_markdown)
        self.assertIn(
            "This paragraph starts with an indented first line. "
            "It continues on the next short physical line.",
            review_markdown,
        )
        self.assertIn("**Table 1 Toxicology summary**", review_markdown)
        self.assertIn("| Endpoint | Result |", review_markdown)
        self.assertIn("Note line one. Note line two.", review_markdown)
        self.assertIn("![Figure 1 Mean blood pressure](#img_001)", review_markdown)
        self.assertIn("Figure 1 Mean blood pressure Values are mean S.E.", review_markdown)

    def test_ind_review_markdown_does_not_repeat_table_title_after_same_visible_heading(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        title = "2.6.7.17 Other toxicity study Test article: Example"
        document = {
            "filename": "visible-title-owner.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_title",
                                "semantic_role": "section_heading",
                                "text": title,
                                "section_context": {
                                    "outline_index": "2.6.7.17",
                                    "section_title": "Other toxicity study Test article: Example",
                                    "section_level": 4,
                                },
                                "bbox": [72, 100, 480, 118],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": title,
                                "bbox": [72, 130, 520, 230],
                                "composite_object": {
                                    "object_family": "business_table",
                                    "ownership_domain": "table",
                                    "title_policy": "owned_object_title",
                                    "visible_title_owner": "table",
                                    "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                },
                                "display_grid": [
                                    ["Species", "Result"],
                                    ["Rat", "No finding"],
                                ],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "title": title,
                    "display_grid": [["Species", "Result"], ["Rat", "No finding"]],
                }
            ],
            "pages": [{"page_number": 1, "block_count": 2}],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(review_markdown.count(title), 1, msg=review_markdown)
        self.assertIn("| Species | Result |", review_markdown)
        self.assertIn("| Rat | No finding |", review_markdown)

    def test_ind_review_markdown_merges_split_heading_before_title_owner_arbitration(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        first_line = "2.6.7.9B Genotoxicity: in vivo"
        continuation = "Report title: MM-180801: Rat oral DNA damage repair study"
        full_title = f"{first_line} {continuation}"
        document = {
            "filename": "split-heading-title-owner.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_title_1",
                                "semantic_role": "section_heading",
                                "text": first_line,
                                "section_context": {
                                    "outline_index": "2.6.7.9B",
                                    "section_title": "Genotoxicity: in vivo",
                                    "section_level": 4,
                                },
                                "bbox": [72, 100, 420, 118],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_title_2",
                                "semantic_role": "section_heading_continuation",
                                "text": continuation,
                                "section_context": {
                                    "outline_index": "2.6.7.9B",
                                    "section_title": "Genotoxicity: in vivo",
                                    "section_level": 4,
                                },
                                "bbox": [84, 120, 510, 138],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": full_title,
                                "bbox": [72, 160, 520, 260],
                                "composite_object": {
                                    "object_family": "business_table",
                                    "ownership_domain": "table",
                                    "title_policy": "owned_object_title",
                                    "visible_title_owner": "table",
                                    "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                },
                                "display_grid": [
                                    ["Treatment", "Dose"],
                                    ["Cyclophosphamide", "7.5"],
                                ],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "title": full_title,
                    "display_grid": [["Treatment", "Dose"], ["Cyclophosphamide", "7.5"]],
                }
            ],
            "pages": [{"page_number": 1, "block_count": 3}],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn(f"###### {full_title}", review_markdown)
        self.assertEqual(review_markdown.count(full_title), 1, msg=review_markdown)
        self.assertNotIn(f"\n{continuation}\n", review_markdown)
        self.assertIn("| Cyclophosphamide | 7.5 |", review_markdown)

    def test_ind_review_markdown_merges_ind_metadata_line_heading_when_object_references_full_title(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        first_line = "2.6.7.9B 遗传毒性：体内"
        continuation = "报告标题：MM-180801：大鼠经口给药DNA 损伤修复试验"
        full_title = f"{first_line} {continuation}"
        document = {
            "filename": "ind-metadata-heading-owner.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_title_1",
                                "semantic_role": "section_heading",
                                "text": first_line,
                                "section_context": {
                                    "outline_index": "2.6.7.9B",
                                    "section_title": "遗传毒性：体内",
                                    "section_level": 4,
                                },
                                "bbox": [72, 100, 300, 118],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_title_2",
                                "text": continuation,
                                "section_context": {
                                    "outline_index": "2.6.7.9B",
                                    "section_title": "遗传毒性：体内",
                                    "section_level": 4,
                                },
                                "bbox": [84, 121, 520, 139],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": full_title,
                                "bbox": [72, 166, 520, 266],
                                "composite_object": {
                                    "object_family": "business_table",
                                    "ownership_domain": "table",
                                    "title_policy": "owned_object_title",
                                    "visible_title_owner": "table",
                                    "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                },
                                "display_grid": [
                                    ["处理", "剂量"],
                                    ["环磷酰胺", "7.5"],
                                ],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "title": full_title,
                    "display_grid": [["处理", "剂量"], ["环磷酰胺", "7.5"]],
                }
            ],
            "pages": [{"page_number": 1, "block_count": 3}],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn(f"###### {full_title}", review_markdown)
        self.assertEqual(review_markdown.count(full_title), 1, msg=review_markdown)
        self.assertNotIn(f"\n{continuation}\n", review_markdown)
        self.assertIn("| 环磷酰胺 | 7.5 |", review_markdown)

    def test_ind_review_markdown_merges_multiple_metadata_heading_lines_before_object_title(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        first_line = "2.6.7.9B Genotoxicity: in vivo"
        report_title = "Report title: MM-180801 Rat oral DNA damage repair study"
        test_article = "Test article: Example compound"
        full_title = f"{first_line} {report_title} {test_article}"
        document = {
            "filename": "multi-line-heading-owner.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_title_1",
                                "semantic_role": "section_heading",
                                "text": first_line,
                                "section_context": {
                                    "outline_index": "2.6.7.9B",
                                    "section_title": "Genotoxicity: in vivo",
                                    "section_level": 4,
                                },
                                "bbox": [72, 100, 320, 118],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_title_2",
                                "text": report_title,
                                "bbox": [84, 121, 520, 139],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_title_3",
                                "text": test_article,
                                "bbox": [84, 142, 360, 160],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": full_title,
                                "bbox": [72, 188, 520, 288],
                                "composite_object": {
                                    "object_family": "business_table",
                                    "ownership_domain": "table",
                                    "title_policy": "owned_object_title",
                                    "visible_title_owner": "table",
                                    "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                },
                                "display_grid": [
                                    ["Treatment", "Dose"],
                                    ["Cyclophosphamide", "7.5"],
                                ],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "title": full_title,
                    "display_grid": [["Treatment", "Dose"], ["Cyclophosphamide", "7.5"]],
                }
            ],
            "pages": [{"page_number": 1, "block_count": 4}],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn(f"###### {full_title}", review_markdown)
        self.assertEqual(review_markdown.count(full_title), 1, msg=review_markdown)
        self.assertNotIn(f"\n{report_title}\n", review_markdown)
        self.assertNotIn(f"\n{test_article}\n", review_markdown)
        self.assertIn("| Cyclophosphamide | 7.5 |", review_markdown)

    def test_ind_review_markdown_does_not_repeat_structure_template_title_after_same_visible_heading(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        title = "2.6.7 Study summary template"
        document = {
            "filename": "visible-template-title-owner.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_title",
                                "semantic_role": "section_heading",
                                "text": title,
                                "section_context": {
                                    "outline_index": "2.6.7",
                                    "section_title": "Study summary template",
                                    "section_level": 3,
                                },
                                "bbox": [72, 100, 420, 118],
                            },
                            {
                                "block_type": "structure_template",
                                "block_id": "structure_template_001",
                                "structure_template_id": "structure_template_001",
                                "title": title,
                                "template_profile": "blank_study_summary_template",
                                "ownership_domain": "template_form",
                                "bbox": [72, 130, 520, 230],
                                "entries": [
                                    {
                                        "outline_index": "1",
                                        "title": "Study number",
                                        "text": "Study number",
                                    }
                                ],
                                "row_texts": ["Study number"],
                                "composite_object": {
                                    "object_family": "structure_template",
                                    "ownership_domain": "template_form",
                                    "title_policy": "owned_object_title",
                                    "visible_title_owner": "template_form",
                                    "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                },
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 2}],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(review_markdown.count(title), 1, msg=review_markdown)
        self.assertIn("Study number", review_markdown)

    def test_ind_review_markdown_suppresses_absorbed_structure_template_surface(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "absorbed-template-owner.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "structure_template",
                                "block_id": "structure_template_001",
                                "structure_template_id": "structure_template_001",
                                "title": "2.6.7.9B Genotoxicity: in vivo",
                                "template_profile": "absorbed_populated_study_metadata",
                                "ownership_domain": "absorbed_by_business_table",
                                "absorbed_by_table_id": "tbl_001",
                                "bbox": [72, 100, 520, 160],
                                "entries": [
                                    {
                                        "outline_index": "",
                                        "title": "Cyclophosphamide 7.5M 51+/-2.3 2.49+/-0.30**",
                                        "text": "Cyclophosphamide 7.5M 51+/-2.3 2.49+/-0.30**",
                                    }
                                ],
                                "row_texts": ["Cyclophosphamide 7.5M 51+/-2.3 2.49+/-0.30**"],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": "2.6.7.9B Genotoxicity: in vivo",
                                "bbox": [72, 170, 520, 240],
                                "display_grid": [
                                    ["Article", "Dose", "PCE", "MN-PCE"],
                                    ["Cyclophosphamide", "7.5M", "51+/-2.3", "2.49+/-0.30**"],
                                ],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 2}],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertNotIn("结构模板", review_markdown)
        self.assertNotIn("structure template", review_markdown.lower())
        self.assertEqual(review_markdown.count("Cyclophosphamide"), 1, msg=review_markdown)
        self.assertEqual(review_markdown.count("2.49+/-0.30**"), 1, msg=review_markdown)

    def test_ind_review_markdown_uses_top_level_title_reference_policy_for_title_suppression(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table_title = "2.6.7.17 Other toxicity study Test article: Example"
        template_title = "2.6.7.18 Integrated summary Test article: Example"
        figure_title = "Figure 4 Mean exposure"
        document = {
            "filename": "top-level-title-policy.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_table_title",
                                "semantic_role": "section_heading",
                                "text": table_title,
                                "bbox": [72, 80, 460, 96],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": table_title,
                                "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                "visible_title_owner": "table",
                                "bbox": [72, 110, 520, 180],
                                "display_grid": [["Endpoint", "Result"], ["Antigenicity", "Negative"]],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_template_title",
                                "semantic_role": "section_heading",
                                "text": template_title,
                                "bbox": [72, 210, 460, 226],
                            },
                            {
                                "block_type": "structure_template",
                                "block_id": "structure_template_001",
                                "structure_template_id": "structure_template_001",
                                "title": template_title,
                                "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                "visible_title_owner": "template_form",
                                "template_profile": "blank_study_summary_template",
                                "ownership_domain": "template_form",
                                "bbox": [72, 240, 520, 300],
                                "entries": [{"title": "Study number", "text": "Study number"}],
                                "row_texts": ["Study number"],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_figure_title",
                                "semantic_role": "section_heading",
                                "text": figure_title,
                                "bbox": [72, 330, 460, 346],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_001",
                                "image_id": "img_001",
                                "caption_text": figure_title,
                                "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                "visible_title_owner": "figure",
                                "bbox": [72, 360, 520, 460],
                                "content_segments": [{"role": "legend", "text": "Values are mean."}],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 6}],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(review_markdown.count(table_title), 1, msg=review_markdown)
        self.assertIn("| Antigenicity | Negative |", review_markdown)
        self.assertEqual(review_markdown.count(template_title), 1, msg=review_markdown)
        self.assertIn("Study number", review_markdown)
        self.assertEqual(review_markdown.count(figure_title), 1, msg=review_markdown)
        self.assertIn("Values are mean.", review_markdown)

    def test_ind_review_markdown_suppresses_metadata_only_text_blocks_from_body_flow(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        first_line = "2.6.7.9B Genotoxicity: in vivo"
        metadata_line = "Internal parser ownership reference for table title source"
        edge_only_metadata_line = "Edge-only metadata title source"
        metadata_only_row = "Metadata-only duplicate table surface"
        document = {
            "filename": "metadata-only-visible-policy.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_heading",
                                "semantic_role": "section_heading",
                                "text": first_line,
                                "bbox": [72, 80, 420, 96],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_metadata",
                                "semantic_role": "body",
                                "unit_role": "metadata",
                                "visible_render_policy": "metadata_only",
                                "text": metadata_line,
                                "bbox": [84, 100, 520, 116],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_edge_metadata",
                                "semantic_role": "body",
                                "unit_role": "body",
                                "text": f"{edge_only_metadata_line} would otherwise leak as body text.",
                                "bbox": [84, 120, 520, 136],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": first_line,
                                "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                "visible_title_owner": "table",
                                "bbox": [72, 140, 520, 220],
                                "display_grid": [
                                    ["Article", "Dose"],
                                    ["MM-180801", "2000"],
                                ],
                                "metadata_reference_edges": [
                                    {
                                        "source_block_id": "txt_edge_metadata",
                                        "relation": "title_metadata_continuation",
                                        "target_object_type": "table",
                                        "target_object_id": "tbl_001",
                                        "visible_render_policy": "metadata_only",
                                    }
                                ],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_metadata_only",
                                "table_id": "tbl_metadata_only",
                                "visible_render_policy": "metadata_only",
                                "bbox": [72, 240, 520, 300],
                                "display_grid": [
                                    ["Evidence", "Policy"],
                                    [metadata_only_row, "metadata_only"],
                                ],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 5}],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertNotIn(f"\n{metadata_line}\n", review_markdown)
        self.assertNotIn(edge_only_metadata_line, review_markdown)
        self.assertNotIn(metadata_only_row, review_markdown)
        self.assertIn(f"### {first_line}", review_markdown)
        self.assertIn("| MM-180801 | 2000 |", review_markdown)

    def test_ind_review_markdown_does_not_repeat_image_caption_after_same_visible_heading(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        title = "Figure 1 Mean blood pressure"
        document = {
            "filename": "visible-image-title-owner.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_title",
                                "semantic_role": "section_heading",
                                "text": title,
                                "section_context": {
                                    "section_title": title,
                                    "section_level": 4,
                                },
                                "bbox": [72, 100, 420, 118],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_001",
                                "image_id": "img_001",
                                "figure_ref": "Figure 1",
                                "caption_text": title,
                                "bbox": [72, 130, 460, 300],
                                "content_segments": [
                                    {"role": "caption", "relation": "above", "text": title},
                                    {"role": "legend", "relation": "below", "text": "Values are mean S.E."},
                                ],
                                "composite_object": {
                                    "object_family": "figure",
                                    "ownership_domain": "figure",
                                    "title_policy": "owned_object_title",
                                    "visible_title_owner": "figure",
                                    "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                                },
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 2}],
            "image_blocks": [
                {
                    "block_type": "image",
                    "block_id": "img_001",
                    "image_id": "img_001",
                    "figure_ref": "Figure 1",
                    "caption_text": title,
                    "bbox": [72, 130, 460, 300],
                    "content_segments": [
                        {"role": "caption", "relation": "above", "text": title},
                        {"role": "legend", "relation": "below", "text": "Values are mean S.E."},
                    ],
                    "composite_object": {
                        "object_family": "figure",
                        "ownership_domain": "figure",
                        "title_policy": "owned_object_title",
                        "visible_title_owner": "figure",
                        "metadata_title_reference_policy": "may_reference_without_visible_rendering",
                    },
                }
            ],
            "text": "",
        }

        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(review_markdown.count(title), 1, msg=review_markdown)
        self.assertIn("![Figure 1](#img_001)", review_markdown)
        self.assertIn("Values are mean S.E.", review_markdown)

    def test_full_evidence_snapshot_exposes_visible_evidence_inventory_not_review_content(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "evidence-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 2, "parser_hint": "pdf-ast-v5"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "height": 792,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_001",
                                "text": "1. Evidence overview",
                                "bbox": [72, 80, 260, 96],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": "Table 1 Study evidence",
                                "bbox": [72, 120, 420, 210],
                                "display_grid": [["Item", "Value"], ["Dose", "10 mg/kg"]],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_001",
                                "image_id": "img_001",
                                "caption_text": "Figure 1 XML screenshot",
                                "bbox": [72, 240, 420, 360],
                                "content_text": "<cn-envelope><leaf /></cn-envelope>",
                                "content_segments": [
                                    {"role": "caption", "text": "Figure 1 XML screenshot"},
                                    {"role": "embedded_code", "text": "<cn-envelope><leaf /></cn-envelope>"},
                                ],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "title": "Table 1 Study evidence",
                    "page": 1,
                    "bbox": [72, 120, 420, 210],
                    "display_grid": [["Item", "Value"], ["Dose", "10 mg/kg"]],
                }
            ],
            "image_blocks": [
                {
                    "image_id": "img_001",
                    "caption_text": "Figure 1 XML screenshot",
                    "page": 1,
                    "bbox": [72, 240, 420, 360],
                    "content_text": "<cn-envelope><leaf /></cn-envelope>",
                }
            ],
            "content_evidence": [
                {
                    "evidence_id": "ev_txt_001",
                    "source_type": "text",
                    "source_id": "txt_001",
                    "page": 1,
                    "text": "1. Evidence overview",
                },
                {
                    "evidence_id": "ev_tbl_001",
                    "source_type": "table",
                    "source_id": "tbl_001",
                    "page": 1,
                    "text": "Table 1 Study evidence",
                },
            ],
            "pages": [{"page_number": 1, "block_count": 3}],
            "text": "",
        }

        full_markdown = api_main._build_full_markdown([document])
        review_markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertIn("### Evidence Snapshot Inventory", full_markdown)
        self.assertIn("- Pages: 2", full_markdown)
        self.assertIn("- AST blocks: 3", full_markdown)
        self.assertIn("- Tables: 1", full_markdown)
        self.assertIn("- Images: 1", full_markdown)
        self.assertIn("- Content evidence records: 2", full_markdown)
        self.assertIn("#### Table Evidence Objects", full_markdown)
        self.assertIn("`tbl_001`", full_markdown)
        self.assertIn("#### Image Evidence Objects", full_markdown)
        self.assertIn("`img_001`", full_markdown)
        self.assertIn("embedded evidence", full_markdown)
        self.assertIn("<cn-envelope><leaf /></cn-envelope>", full_markdown)

        self.assertNotIn("### Evidence Snapshot Inventory", review_markdown)
        self.assertNotIn("#### Table Evidence Objects", review_markdown)
        self.assertNotIn("#### Image Evidence Objects", review_markdown)
        self.assertNotIn("<cn-envelope><leaf /></cn-envelope>", review_markdown)

    def test_workbench_exposes_ind_review_markdown_download_url_without_replacing_full_markdown_url(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url="/api/v1/jobs/job_1/markdown/download",
            ind_review_markdown_download_url="/api/v1/jobs/job_1/ind-review/markdown/download",
        )

        self.assertEqual(workbench["full_markdown_download_url"], "/api/v1/jobs/job_1/markdown/download")
        self.assertEqual(
            workbench["ind_review_markdown_download_url"],
            "/api/v1/jobs/job_1/ind-review/markdown/download",
        )

    def test_full_markdown_embeds_pdf_equation_crop_without_user_facing_text_evidence_notes(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
            fitz = importlib.import_module("fitz")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"required module unavailable in this environment: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "equation-source.pdf"
            pdf = fitz.open()
            page = pdf.new_page(width=240, height=120)
            page.insert_text((30, 58), "f(x)=x^2+1   (1)", fontsize=14)
            pdf.save(pdf_path)
            pdf.close()

            document = {
                "filename": "equation-source.pdf",
                "source_type": "pdf",
                "source_path": str(pdf_path),
                "metadata": {"page_count": 1, "parser_hint": "pdf"},
                "document_ast": {
                    "pages": [
                        {
                            "page": 1,
                            "blocks": [
                                {
                                    "block_type": "equation",
                                    "block_id": "eq_p1_001",
                                    "equation_id": "eq_p1_001",
                                    "semantic_role": "display_equation",
                                    "equation_label": "(1)",
                                    "text": "f(x)=x^2+1 (1)",
                                    "bbox": [24, 38, 190, 70],
                                }
                            ],
                        }
                    ]
                },
                "pages": [{"page_number": 1, "block_count": 1}],
                "equation_blocks": [
                    {
                        "equation_id": "eq_p1_001",
                        "page": 1,
                        "equation_label": "(1)",
                        "semantic_role": "display_equation",
                        "text": "f(x)=x^2+1 (1)",
                        "bbox": [24, 38, 190, 70],
                    }
                ],
                "table_asts": [],
                "image_blocks": [],
                "text": "",
            }

            markdown = api_main._build_full_markdown([document])

            self.assertNotIn("**鍏紡 (1)**", markdown)
            self.assertIn("](data:image/png;base64,", markdown)
            self.assertNotIn("```text\nf(x)=x^2+1 (1)\n```", markdown)
            self.assertNotIn("公式文本由 PDF 文本层提取", markdown)
            self.assertNotIn("LaTeX", markdown)

    def test_full_markdown_uses_high_confidence_equation_latex_without_repeating_raw_text(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "equation-latex-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "equation",
                                "block_id": "eq_p1_001",
                                "equation_id": "eq_p1_001",
                                "page": 1,
                                "text": "partial partial partial partial F v v v = * 0 = ,",
                                "bbox": [24, 38, 190, 70],
                                "latex_text": r"\frac{\partial F}{\partial v}(v^*) = 0",
                                "latex_confidence": 0.88,
                                "latex_source": "pdf_text_layer_formula_pattern",
                            }
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "equation_blocks": [],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertNotIn("**鍏紡**", markdown)
        self.assertIn("$$\n\\frac{\\partial F}{\\partial v}(v^*) = 0\n$$", markdown)
        self.assertNotIn("![鍏紡]", markdown)
        self.assertNotIn("```latex", markdown)
        self.assertNotIn("LaTeX", markdown)
        self.assertNotIn("```text\npartial partial partial partial F v v v = * 0 = ,\n```", markdown)
        self.assertEqual(
            document["document_ast"]["pages"][0]["blocks"][0]["text"],
            "partial partial partial partial F v v v = * 0 = ,",
        )

    def test_full_markdown_replaces_high_confidence_inline_formula_spans_in_body_text(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "inline-formula-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "respectively, i.e., u_j = 1/n_j sum_i x_i^(j) and u = 1/n sum_i x_i.",
                                "display_text": "respectively, i.e., u_j = 1/n_j sum_i x_i^(j) and u = 1/n sum_i x_i.",
                                "bbox": [72, 80, 420, 96],
                                "inline_formula_spans": [
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [180, 80, 300, 96],
                                        "content": "u_j = 1/n_j sum_i x_i^(j)",
                                        "latex_text": r"u_j=\frac{1}{n_j}\sum_{i=1}^{n_j}x_i^{(j)}",
                                        "latex_confidence": 0.96,
                                        "formula_complexity": "inline_formula",
                                        "ocr_candidate": True,
                                    },
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [320, 80, 390, 96],
                                        "content": "x_i",
                                        "latex_text": None,
                                        "latex_confidence": 0.0,
                                        "formula_complexity": "inline_symbol",
                                        "ocr_candidate": False,
                                    },
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [300, 80, 410, 96],
                                        "content": "u = 1/n sum_i x_i",
                                        "latex_text": r"u=\frac{1}{n}\sum_{i=1}^{n}x_i",
                                        "latex_confidence": 0.93,
                                        "formula_complexity": "inline_formula",
                                        "ocr_candidate": True,
                                    },
                                ],
                            }
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(
            r"respectively, i.e., $u_j=\frac{1}{n_j}\sum_{i=1}^{n_j}x_i^{(j)}$ and $u=\frac{1}{n}\sum_{i=1}^{n}x_i$.",
            markdown,
        )
        self.assertNotIn("respectively, i.e., u_j = 1/n_j sum_i x_i^(j) and u = 1/n sum_i x_i.", markdown)
        self.assertNotIn("**公式项**", markdown)
        self.assertIn(r"$u_j=\frac{1}{n_j}\sum_{i=1}^{n_j}x_i^{(j)}$", markdown)
        self.assertIn(r"$u=\frac{1}{n}\sum_{i=1}^{n}x_i$", markdown)
        self.assertNotIn("inline_symbol", markdown)

    def test_full_markdown_replaces_high_confidence_inline_symbol_spans_in_body_text(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "inline-symbol-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "where xi (j) denotes the i-th sample and g is the number of classes.",
                                "display_text": "where x_i^{(j)} denotes the i-th sample and g is the number of classes.",
                                "bbox": [72, 80, 420, 96],
                                "inline_formula_spans": [
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [104, 80, 128, 96],
                                        "content": "x_i^{(j)}",
                                        "latex_text": r"x_i^{(j)}",
                                        "latex_confidence": 0.86,
                                        "formula_complexity": "inline_symbol",
                                        "ocr_candidate": False,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(r"where $x_i^{(j)}$ denotes the i-th sample and g is the number of classes.", markdown)
        self.assertNotIn("where x_i^{(j)} denotes", markdown)
        self.assertNotIn("$g$", markdown)

    def test_full_markdown_repairs_generic_inline_binary_operations_after_symbol_projection(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "inline-binary-operation-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "The ratio S_b / S_w and A_i ∕ B_j are used.",
                                "display_text": "The ratio S_b / S_w and A_i ∕ B_j are used.",
                                "bbox": [72, 80, 420, 96],
                            }
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(r"The ratio $S_b/S_w$ and $A_i/B_j$ are used.", markdown)
        self.assertNotIn("S_b / S_w", markdown)
        self.assertNotIn("A_i ∕ B_j", markdown)

    def test_full_markdown_prefers_structured_inline_formula_over_overlapping_2d_reconstruction(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "inline-vector-sample.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "where z i =( x i 1 - c j 1 - c 11 - c 21 , ... , x id - c jd - c 1 d - c 2 d ) , with x ij, c ij the j-",
                                "display_text": "where z_i =( x_i_1 - c_j_1 - c_11 - c_21, ..., x_id - c_jd - c_1_d - c_2_d ), with x ij, c ij the j-",
                                "bbox": [72, 80, 420, 96],
                                "inline_formula_spans": [
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [102, 80, 360, 96],
                                        "content": "z i =( x i 1 - c j 1 - c 11 - c 21 , ... , x id - c jd - c 1 d - c 2 d )",
                                        "latex_text": r"z_i=(\lvert x_{i1}-c_{j1}\rvert-\lvert c_{11}-c_{21}\rvert,\ldots,\lvert x_{id}-c_{jd}\rvert-\lvert c_{1d}-c_{2d}\rvert)",
                                        "latex_confidence": 0.88,
                                        "source": "pdf_text_layer_plain_inline_math_pattern",
                                        "formula_complexity": "inline_formula",
                                        "ocr_candidate": True,
                                    },
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [110, 80, 352, 96],
                                        "content": "z_i =( x_i_1 - c_j_1 - c_11 - c_21, ..., x_id - c_jd - c_1_d - c_2_d ),",
                                        "latex_text": "z_i =( x_i_1 - c_j_1 - c_11 - c_21, ..., x_id - c_jd - c_1_d - c_2_d ),",
                                        "latex_confidence": 0.88,
                                        "source": "pdf_text_layer_2d_reconstruction",
                                        "formula_complexity": "inline_formula",
                                        "ocr_candidate": True,
                                    },
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "th element of xi.",
                                "display_text": "th element of x_i.",
                                "bbox": [72, 98, 180, 114],
                                "inline_formula_spans": [
                                    {
                                        "type": "inline_equation",
                                        "layout_label": "inline_formula",
                                        "bbox": [150, 98, 170, 114],
                                        "content": "x_i",
                                        "latex_text": r"x_i",
                                        "latex_confidence": 0.86,
                                        "source": "pdf_text_layer_plain_inline_math_pattern",
                                        "formula_complexity": "inline_symbol",
                                        "ocr_candidate": False,
                                    }
                                ],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 2}],
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(
            r"where $z_i=(\lvert x_{i1}-c_{j1}\rvert-\lvert c_{11}-c_{21}\rvert,\ldots,\lvert x_{id}-c_{jd}\rvert-\lvert c_{1d}-c_{2d}\rvert)$, with x ij, c ij the j-th element of $x_i$.",
            markdown,
        )
        self.assertNotIn("x_i_1", markdown)
        self.assertNotIn("c_1_d", markdown)

    def test_full_markdown_merges_continued_tables_and_uses_display_grid(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "continued-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 2, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {"block_type": "text", "block_id": "txt_001", "text": "Before table."},
                            {"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"},
                        ],
                    },
                    {
                        "page": 2,
                        "blocks": [
                            {"block_type": "table", "block_id": "tbl_002", "table_id": "tbl_002"},
                            {"block_type": "text", "block_id": "txt_002", "text": "After table."},
                        ],
                    },
                ]
            },
            "pages": [{"page_number": 1, "block_count": 2}, {"page_number": 2, "block_count": 2}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "title": "Table 1: Continued",
                    "display_grid": [
                        ["Display A", "Display B"],
                        ["row\n1", "visible\nvalue"],
                    ],
                    "raw_grid": [
                        ["Raw A", "Raw B"],
                        ["raw row", "raw value"],
                    ],
                    "continued_to": ["tbl_002"],
                },
                {
                    "table_id": "tbl_002",
                    "title": "Table 1: Continued",
                    "display_grid": [
                        ["row 2", "continued value"],
                    ],
                    "raw_grid": [
                        ["raw row 2", "raw value 2"],
                    ],
                    "continued_from": ["tbl_001"],
                },
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("Before table.", markdown)
        self.assertIn("After table.", markdown)
        self.assertIn("| Display A | Display B |", markdown)
        self.assertIn("| row 1 | visible value |", markdown)
        self.assertIn("| row 2 | continued value |", markdown)
        self.assertNotIn("<br>", markdown)
        self.assertNotIn("Table 1: Continued", markdown)
        self.assertNotIn("Raw A", markdown)
        self.assertNotIn("raw row", markdown)
        self.assertEqual(markdown.count("| Display A | Display B |"), 1)
        self.assertNotIn("#### Page 1", markdown)
        self.assertNotIn("#### Page 2", markdown)

    def test_full_markdown_uses_semantic_header_when_sparse_header_continuation_rows_exist(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "roadmap-table.pdf",
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
                    "header": [
                        {"text": "IND Submission", "col": 1},
                        {"text": "Submission Date", "col": 2},
                        {"text": "Submission Content", "col": 3},
                        {"text": "CD-ROM", "col": 4},
                        {"text": "Hypertext link Destination", "col": 5},
                    ],
                    "display_grid": [
                        ["IND Submission", "Submission Date", "Submission", "CD-ROM", "Hypertext link"],
                        [None, None, "Content", None, "Destination"],
                        ["IND 12345.0003", "04-Jul-2001", "Cover letter", "3.01", "amendtoc.pdf"],
                    ],
                    "data_grid": [
                        ["IND 12345.0003", "04-Jul-2001", "Cover letter", "3.01", "amendtoc.pdf"],
                        ["IND 12345.0003", "04-Jul-2001", "1571", "3.01", None],
                    ],
                    "data_start_row": 2,
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(
            "| IND Submission | Submission Date | Submission Content | CD-ROM | Hypertext link Destination |",
            markdown,
        )
        self.assertIn("| IND 12345.0003 | 04-Jul-2001 | Cover letter | 3.01 | amendtoc.pdf |", markdown)
        self.assertIn("| IND 12345.0003 | 04-Jul-2001 | 1571 | 3.01 |  |", markdown)
        self.assertNotIn("| IND Submission | Submission Date | Submission | CD-ROM | Hypertext link |", markdown)
        self.assertNotIn("|  |  | Content |  | Destination |", markdown)

    def test_full_markdown_preserves_internal_table_title_when_using_semantic_header(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "test-ind.pdf",
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
                    "title": "Main IND Table of Contents",
                    "title_row_index": 0,
                    "header_row_index": 1,
                    "data_start_row": 2,
                    "header": [
                        {"text": "Section", "col": 1},
                        {"text": "Description", "col": 2},
                        {"text": "Electronic folder/filename", "col": 3},
                    ],
                    "display_grid": [
                        ["Main IND Table of Contents", None, None],
                        ["Section", "Description", "Electronic folder/filename"],
                        ["-", "Coverletter", "0000_coverletter.pdf"],
                    ],
                    "data_grid": [
                        ["-", "Coverletter", "0000_coverletter.pdf"],
                    ],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("**Main IND Table of Contents**", markdown)
        self.assertIn("| Section | Description | Electronic folder/filename |", markdown)
        self.assertIn("| - | Coverletter | 0000_coverletter.pdf |", markdown)
        self.assertNotIn("| Main IND Table of Contents |  |  |", markdown)

    def test_full_markdown_uses_display_rows_with_semantic_header_to_preserve_visual_empty_cells(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "ectd-technical-spec.pdf",
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
                    "header": [
                        {"text": "文件夹", "col": 1},
                        {"text": "文件", "col": 2},
                        {"text": "命名规则", "col": 3},
                    ],
                    "display_grid": [
                        ["文件夹", "文件", "命名规则"],
                        ["0000", None, "4 位数字组成的序列文件夹"],
                        [None, "index.xml", "符合 ICH 要求的骨架文件"],
                        [None, "index-md5.txt", "符合 ICH 要求的 MD5 校验和文件"],
                    ],
                    "data_grid": [
                        ["0000", None, "4 位数字组成的序列文件夹"],
                        ["0000", "index.xml", "符合 ICH 要求的骨架文件"],
                        ["0000", "index-md5.txt", "符合 ICH 要求的 MD5 校验和文件"],
                    ],
                    "semantic_compaction": {
                        "applied": True,
                        "strategy": "leading_key_carry_forward",
                    },
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("| 文件夹 | 文件 | 命名规则 |", markdown)
        self.assertIn("| 0000 |  | 4 位数字组成的序列文件夹 |", markdown)
        self.assertIn("|  | index.xml | 符合 ICH 要求的骨架文件 |", markdown)
        self.assertIn("|  | index-md5.txt | 符合 ICH 要求的 MD5 校验和文件 |", markdown)
        self.assertNotIn("| 0000 | index.xml | 符合 ICH 要求的骨架文件 |", markdown)

    def test_full_markdown_renders_merged_section_group_rows_as_separators(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "merged-section-table.pdf",
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
                        ["序号", "描述", "说明", "严重程度"],
                        ["4.1 - 基础信息", None, None, None],
                        ["4.1.1", "模块一的区域骨架文件必须存在", "m1/cn", "错误"],
                    ],
                    "merged_rows": [
                        {
                            "row": 2,
                            "kind": "section_group",
                            "text": "4.1 - 基础信息",
                            "colspan": 4,
                            "source": "single_leading_section_group_cell",
                        }
                    ],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("**4.1 - 基础信息**", markdown)
        self.assertIn("| 序号 | 描述 | 说明 | 严重程度 |", markdown)
        self.assertIn("| 4.1.1 | 模块一的区域骨架文件必须存在 | m1/cn | 错误 |", markdown)
        self.assertNotIn("| 4.1 - 基础信息 |  |  |  |", markdown)
    def test_full_markdown_renders_merged_table_note_title_rows_as_separators(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "table-note-title.pdf",
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
                        ["说明:", None, None],
                        ["错误", "必须遵守的关键验证标准", "任何错误信息均会导致申报资料被拒收。"],
                        ["警告", "建议遵守的验证标准", "警告信息可以在说明函中进行解释。"],
                    ],
                    "merged_rows": [
                        {
                            "row": 1,
                            "kind": "table_note_title",
                            "text": "说明:",
                            "colspan": 3,
                            "source": "single_leading_title_cell_with_following_tabular_rows",
                        }
                    ],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("**说明:**", markdown)
        self.assertIn("| 错误 | 必须遵守的关键验证标准 | 任何错误信息均会导致申报资料被拒收。 |", markdown)
        self.assertNotIn("| 说明: |  |  |", markdown)

    def test_full_markdown_merges_continuation_boundary_fragment_into_previous_row(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "continued-fragment-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 2, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
                    {"page": 2, "blocks": [{"block_type": "table", "block_id": "tbl_002", "table_id": "tbl_002"}]},
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}, {"page_number": 2, "block_count": 1}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "display_grid": [
                        ["申请类型", "注册行为类型", "序列类型"],
                        ["新药申请", "报告", "首次提交"],
                    ],
                    "continued_to": ["tbl_002"],
                },
                {
                    "table_id": "tbl_002",
                    "display_grid": [
                        [None, None, "回复\n撤回"],
                        [None, "再注册", "首次提交\n回复\n撤回"],
                    ],
                    "continued_from": ["tbl_001"],
                },
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("| 申请类型 | 注册行为类型 | 序列类型 |", markdown)
        self.assertIn("| 新药申请 | 报告 | 首次提交 / 回复 / 撤回 |", markdown)
        self.assertIn("|  | 再注册 | 首次提交 / 回复 / 撤回 |", markdown)
        self.assertNotIn("|  |  | 回复", markdown)
        self.assertEqual(markdown.count("| 申请类型 | 注册行为类型 | 序列类型 |"), 1)

    def test_full_markdown_formats_table_cell_line_breaks_by_content_shape(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "line-break-table.pdf",
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
                        ["项目", "内容"],
                        ["枚举", "首次提交\n回复\n撤回"],
                        ["表头样式", "注册行为\n类型"],
                        ["连续短语", "临床试验\n申请"],
                        ["软换行", "这是一个较长的说明文本\n因为 PDF 自动折行被拆成两行"],
                    ],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("| 枚举 | 首次提交 / 回复 / 撤回 |", markdown)
        self.assertIn("| 表头样式 | 注册行为类型 |", markdown)
        self.assertIn("| 连续短语 | 临床试验申请 |", markdown)
        self.assertIn("| 软换行 | 这是一个较长的说明文本因为 PDF 自动折行被拆成两行 |", markdown)
        self.assertNotIn("<br>", markdown)
