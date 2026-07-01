from __future__ import annotations

import unittest

from parsers.pdf.table_modules.semantic_projection import (
    apply_table_semantic_projection_v2,
    ensure_table_structure_profile,
    merge_semantic_table_fragments_v2,
)


class TableSemanticProjectionV2Tests(unittest.TestCase):
    def test_structure_profile_separates_border_models_from_semantic_family(self) -> None:
        cases = [
            (
                {
                    "table_id": "tbl_full_grid",
                    "detection_source": "pymupdf_builtin",
                    "col_count": 3,
                    "display_grid": [["A", "B", "C"], ["1", "2", "3"]],
                    "raw_evidence_summary": {"horizontal_line_count": 4, "vertical_line_count": 4},
                    "header_column_groups": [{"text": "Group", "start_col": 1, "end_col": 2, "colspan": 2}],
                },
                "full_grid",
                "ruled_grid",
                "cell_grid_geometry",
            ),
            (
                {
                    "table_id": "tbl_horizontal_rules",
                    "detection_source": "caption_anchored_horizontal_rules",
                    "col_count": 4,
                    "display_grid": [["A", "B", "C", "D"], ["1", "2", "3", "4"]],
                    "raw_evidence_summary": {"horizontal_line_count": 3, "vertical_line_count": 0},
                    "title_block": {"text": "Table 1 Study design", "source": "text-layer"},
                    "note_blocks": [{"text": "* Values are approximate."}],
                },
                "horizontal_rules",
                "horizontal_rule_table",
                "row_rule_with_text_columns",
            ),
            (
                {
                    "table_id": "tbl_borderless",
                    "detection_source": "text_aligned_borderless_grid",
                    "col_count": 6,
                    "display_grid": [["Study", "Species", "Route", "Dose", "GLP", "No."], ["A", "Rat", "po", "10", "Y", "95012"]],
                    "raw_evidence_summary": {"horizontal_line_count": 0, "vertical_line_count": 0},
                },
                "borderless_aligned",
                "aligned_text_grid",
                "text_anchor_grid",
            ),
            (
                {
                    "table_id": "tbl_overview",
                    "detection_source": "word_clustering",
                    "col_count": 7,
                    "display_grid": [["Type", "Study", "Species", "Dose", "Finding", "GLP", "No."], ["1.1 Safety", "CNS", "rat", "10", "No finding observed", "Y", "95703"]],
                    "raw_evidence_summary": {"horizontal_line_count": 0, "vertical_line_count": 0},
                    "cell_note_refs": [{"marker": "a", "cell_text": "CNSa", "note_text": "a-GLP statement."}],
                },
                "borderless_overview",
                "clustered_text_table",
                "word_cluster_grid",
            ),
            (
                {
                    "table_id": "tbl_raster",
                    "detection_source": "embedded_image_ocr",
                    "col_count": 2,
                    "display_grid": [["OCR", "Value"], ["A", "1"]],
                    "raw_evidence_summary": {"horizontal_line_count": 1, "vertical_line_count": 1},
                },
                "raster_or_vector_table",
                "image_or_vector_ocr_table",
                "ocr_or_vector_region",
            ),
        ]

        for table, expected_border_model, expected_family, expected_evidence_model in cases:
            with self.subTest(table=table["table_id"]):
                profile = ensure_table_structure_profile(table)

                self.assertEqual(profile["border_model"], expected_border_model)
                self.assertEqual(profile["structure_family"], expected_family)
                self.assertEqual(profile["evidence_model"], expected_evidence_model)
                self.assertEqual(table["table_structure_profile"], profile)
                self.assertEqual(profile["row_count"], len(table["display_grid"]))
                self.assertEqual(profile["col_count"], table["col_count"])

        full_grid_profile = cases[0][0]["table_structure_profile"]
        self.assertTrue(full_grid_profile["has_merged_cell_evidence"])
        self.assertTrue(cases[1][0]["table_structure_profile"]["has_table_footer"])
        self.assertTrue(cases[3][0]["table_structure_profile"]["has_note_references"])

    def test_structure_profile_is_available_even_when_semantic_family_is_unknown(self) -> None:
        table = {
            "table_id": "tbl_region_only",
            "detection_source": "caption_anchored_horizontal_rules",
            "col_count": 3,
            "display_grid": [["A", "B", "C"]],
            "raw_evidence_summary": {"horizontal_line_count": 2, "vertical_line_count": 0},
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertFalse(changed)
        self.assertNotIn("semantic_projection_v2", table)
        self.assertEqual(table["table_structure_profile"]["border_model"], "horizontal_rules")
        self.assertEqual(table["table_structure_profile"]["structure_family"], "horizontal_rule_table")
        self.assertEqual(table["table_structure_profile"]["evidence_model"], "row_rule_with_text_columns")

    def test_leading_boundary_with_single_cell_header_projects_to_body_columns(self) -> None:
        table = {
            "table_id": "tbl_single_cell_header_after_boundary",
            "detection_source": "visual_structure_grid",
            "col_count": 4,
            "display_grid": [
                ["MOHAVE COMMUNITY COLLEGE", "Column 2", "Column 3", "BIO181"],
                ["Saccharometer DI Water Glucose Solution Yeast Suspension", None, None, None],
                ["2", "24 ml", "0 ml", "4 ml"],
                ["3", "12 ml", "12 ml", "4 ml"],
                ["4", "4 ml", "12 ml", "12 ml"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "comparison_matrix")
        self.assertIn("compact_single_cell_header_projection", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Saccharometer", "DI Water", "Glucose Solution", "Yeast Suspension"],
                ["2", "24 ml", "0 ml", "4 ml"],
                ["3", "12 ml", "12 ml", "4 ml"],
                ["4", "4 ml", "12 ml", "12 ml"],
            ],
        )
        self.assertEqual(
            table["display_grid"][0],
            ["MOHAVE COMMUNITY COLLEGE", "Column 2", "Column 3", "BIO181"],
        )

    def test_dense_record_matrix_with_numeric_stub_compacts_wrapped_row(self) -> None:
        table = {
            "table_id": "tbl_dense_numeric_stub_wrapped_row",
            "detection_source": "visual_structure_grid",
            "col_count": 4,
            "display_grid": [
                ["Version", "Date", "Change", "Affected Sections"],
                ["1.0", "April 30, 2022", "Original", None],
                ["1.0", "June 3,", "Small edits for clarity on Creative", "1. Introduction to Open Educational"],
                [None, "2022", "Commons licensing and attribution.", "Resources"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "comparison_matrix")
        self.assertIn("multicolumn_wrapped_record_compaction", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Version", "Date", "Change", "Affected Sections"],
                ["1.0", "April 30, 2022", "Original", None],
                [
                    "1.0",
                    "June 3, 2022",
                    "Small edits for clarity on Creative Commons licensing and attribution.",
                    "1. Introduction to Open Educational Resources",
                ],
            ],
        )

    def test_dense_record_matrix_compacts_short_description_continuation_fragments(self) -> None:
        table = {
            "table_id": "tbl_dense_record_short_wrapped_fragments",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 4,
            "display_grid": [
                ["Version", "Date", "Change", "Affected Sections"],
                ["1.0", "April 30, 2022", "Original", None],
                ["1.0", "June 3,", "Small edits for clarity on Creative", "1. Introduction to Open Educational"],
                [None, "2022", "Commons licensing and", "Resources"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "comparison_matrix")
        self.assertIn("multicolumn_wrapped_record_compaction", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Version", "Date", "Change", "Affected Sections"],
                ["1.0", "April 30, 2022", "Original", None],
                [
                    "1.0",
                    "June 3, 2022",
                    "Small edits for clarity on Creative Commons licensing and",
                    "1. Introduction to Open Educational Resources",
                ],
            ],
        )

    def test_external_header_is_prepended_when_display_grid_starts_with_body_rows(self) -> None:
        table = {
            "table_id": "tbl_external_header_body_only_grid",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 5,
            "header": [
                {"col": 1, "text": "Year"},
                {"col": 2, "text": "Recovery Rate"},
                {"col": 3, "text": "Unadjusted Basis"},
                {"col": 4, "text": "Depreciation Expense"},
                {"col": 5, "text": "Accumulated Depreciation"},
            ],
            "display_grid": [
                ["1", ".1667", "$100,000", "$16,670", "$16,670"],
                ["2", ".3333", "$100,000", "$33,330", "$50,000"],
                ["3", ".3333", "$100,000", "$33,330", "$88,330"],
                ["4", ".1667", "$100,000", "$16,670", "$100,000"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "comparison_matrix")
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Year", "Recovery Rate", "Unadjusted Basis", "Depreciation Expense", "Accumulated Depreciation"],
                ["1", ".1667", "$100,000", "$16,670", "$16,670"],
                ["2", ".3333", "$100,000", "$33,330", "$50,000"],
                ["3", ".3333", "$100,000", "$33,330", "$88,330"],
                ["4", ".1667", "$100,000", "$16,670", "$100,000"],
            ],
        )
        self.assertIn("external_header_grid_projection", table["semantic_projection_v2"])

    def test_external_header_accepts_body_values_with_footnote_markers_and_units(self) -> None:
        table = {
            "table_id": "tbl_external_header_marked_values",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 4,
            "title": "Table 2. Contents of Saccharometers when testing fermentation with various yeast",
            "header": [
                {"col": 1, "text": "Saccharometer"},
                {"col": 2, "text": "DI Water"},
                {"col": 3, "text": "Glucose Solution"},
                {"col": 4, "text": "Yeast Suspension"},
            ],
            "display_grid": [
                ["1", "*8 ml", "*6 ml", "0 ml"],
                ["2", "*12 ml", "0 ml", "*2 ml"],
                ["3", "*6 ml", "*6 ml", "*2 ml"],
                ["4", "*2 ml", "*6 ml", "*6 ml"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "comparison_matrix")
        self.assertIn("external_header_grid_projection", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Saccharometer", "DI Water", "Glucose Solution", "Yeast Suspension"],
                ["1", "*8 ml", "*6 ml", "0 ml"],
                ["2", "*12 ml", "0 ml", "*2 ml"],
                ["3", "*6 ml", "*6 ml", "*2 ml"],
                ["4", "*2 ml", "*6 ml", "*6 ml"],
            ],
        )

    def test_merged_adjacent_numeric_value_columns_are_split_by_stable_column_evidence(self) -> None:
        table = {
            "table_id": "tbl_merged_adjacent_numeric_values",
            "detection_source": "pymupdf_builtin",
            "col_count": 3,
            "header": [
                {"col": 1, "text": "Port"},
                {"col": 2, "text": "Foreign"},
                {"col": 3, "text": "Domestic"},
            ],
            "display_grid": [
                ["Foreign", None, "Domestic"],
                ["MANILA", "2454 6,125", None],
                ["CEBU", "1138", "79,500"],
                ["BATANGAS", "958 13,196", None],
                ["SUBIC", "313", "136"],
                ["DAVAO", "750", "17,807"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("merged_adjacent_numeric_value_projection", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Port", "Foreign", "Domestic"],
                ["MANILA", "2454", "6,125"],
                ["CEBU", "1138", "79,500"],
                ["BATANGAS", "958", "13,196"],
                ["SUBIC", "313", "136"],
                ["DAVAO", "750", "17,807"],
            ],
        )

    def test_external_header_ignores_metadata_that_is_really_a_data_row(self) -> None:
        table = {
            "table_id": "tbl_spurious_header_metadata",
            "detection_source": "pymupdf_builtin",
            "col_count": 3,
            "header": [
                {"col": 1, "text": "MANILA"},
                {"col": 2, "text": "2454 6,125"},
                {"col": 3, "text": "Domestic"},
            ],
            "display_grid": [
                ["Foreign", None, "Domestic"],
                ["MANILA", "2454 6,125", None],
                ["CEBU", "1138", "79,500"],
                ["BATANGAS", "958 13,196", None],
                ["SUBIC", "313", "136"],
                ["DAVAO", "750", "17,807"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertNotIn("external_header_grid_projection", table["semantic_projection_v2"])
        self.assertIn("missing_leading_stub_header_projection", table["semantic_projection_v2"])
        self.assertEqual(table["semantic_grid"][0], [None, "Foreign", "Domestic"])
        self.assertEqual(table["semantic_grid"][1], ["MANILA", "2454", "6,125"])

    def test_spreadsheet_matrix_restores_missing_corner_cell_for_column_letters_and_row_numbers(self) -> None:
        table = {
            "table_id": "tbl_spreadsheet_grid",
            "detection_source": "pymupdf_builtin",
            "col_count": 6,
            "header": [
                {"col": 1, "text": "1"},
                {"col": 2, "text": "time"},
                {"col": 3, "text": "observed"},
                {"col": 4, "text": "Forecast(observed)"},
                {"col": 5, "text": "Lower Confidence Bound(observed)"},
                {"col": 6, "text": "Upper Confidence Bound(observed)"},
            ],
            "display_grid": [
                ["A", None, "B", "C", "D", "E"],
                ["1", "time", "observed", "Forecast(observed)", "Lower Confidence Bound(observed)", "Upper Confidence Bound(observed)"],
                ["2", "0", "13", None, None, None],
                ["3", "1", "12", None, None, None],
                ["4", "2", "13.5", None, None, None],
            ],
            "data_grid": [
                ["2", "0", "13", None, None, None],
                ["3", "1", "12", None, None, None],
                ["4", "2", "13.5", None, None, None],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "spreadsheet_matrix")
        self.assertEqual(table["semantic_grid"][0], [None, "A", "B", "C", "D", "E"])
        self.assertEqual(
            table["semantic_grid"][1],
            ["1", "time", "observed", "Forecast(observed)", "Lower Confidence Bound(observed)", "Upper Confidence Bound(observed)"],
        )
        self.assertEqual(table["semantic_header"][0]["role"], "projected_row_header")
        self.assertIn("missing_leading_stub_header_projection", table["semantic_projection_v2"])

    def test_sparse_numeric_anchor_columns_project_to_logical_columns_and_release_tail_prose(self) -> None:
        table = {
            "table_id": "tbl_sparse_numeric_anchor_columns",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 8,
            "header": [
                {"col": 1, "text": "Year"},
                {"col": 2, "text": "Recovery Rate"},
                {"col": 3, "text": "Unadjusted"},
                {"col": 4, "text": "Basis"},
                {"col": 5, "text": "Depreciation"},
                {"col": 6, "text": "Expense"},
                {"col": 7, "text": "Accumulated"},
                {"col": 8, "text": "Depreciation"},
            ],
            "display_grid": [
                ["1", ".1667", None, "$100,000", None, "$16,670", None, "$16,670"],
                ["2", ".3333", None, "$100,000", None, "$33,330", None, "$50,000"],
                ["3", ".3333", None, "$100,000", None, "$33,330", None, "$88,330"],
                ["4", ".1667", None, "$100,000", None, "$16,670", None, "$100,000"],
                ["Note", "that the book value or", "basis of", "the asset", "(acquisition", "cost - accumulated", None, "would"],
                ["be $0", "after it has been", "fully depreciated", None, None, None, "the half-year", None],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("sparse_numeric_anchor_column_projection", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Year", "Recovery Rate", "Unadjusted Basis", "Depreciation Expense", "Accumulated Depreciation"],
                ["1", ".1667", "$100,000", "$16,670", "$16,670"],
                ["2", ".3333", "$100,000", "$33,330", "$50,000"],
                ["3", ".3333", "$100,000", "$33,330", "$88,330"],
                ["4", ".1667", "$100,000", "$16,670", "$100,000"],
            ],
        )
        self.assertEqual(table["semantic_projection_v2"]["sparse_numeric_anchor_column_projection"]["released_tail_row_count"], 2)

    def test_sparse_numeric_anchor_columns_use_external_candidates_when_header_metadata_is_data_like(self) -> None:
        table = {
            "table_id": "tbl_sparse_numeric_anchor_external_header",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 9,
            "header": [
                {"col": 1, "text": "2"},
                {"col": 2, "text": ".3333"},
                {"col": 3, "text": ".1667"},
                {"col": 4, "text": "Column 4"},
                {"col": 5, "text": "$100,000"},
                {"col": 6, "text": "$33,330"},
                {"col": 7, "text": "$16,670"},
                {"col": 8, "text": "$50,000"},
                {"col": 9, "text": "$16,670"},
            ],
            "header_candidates": [
                "Year",
                "Recovery Rate",
                "Unadjusted Basis",
                "Depreciation Expense",
                "Accumulated Depreciation",
            ],
            "display_grid": [
                ["1", None, ".1667", None, "$100,000", None, "$16,670", None, "$16,670"],
                ["2", ".3333", None, None, "$100,000", "$33,330", None, "$50,000", None],
                ["3", ".3333", None, None, "$100,000", "$33,330", None, None, "$88,330"],
                ["4", None, ".1667", None, "$100,000", None, "$16,670", "$100,000", None],
                ["Note", "that the", "book value or", "basis of", "the asset", "(acquisition", "cost - accumulated", "depreciation)", "would"],
            ],
            "data_start_row": 2,
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("sparse_numeric_anchor_column_projection", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Year", "Recovery Rate", "Unadjusted Basis", "Depreciation Expense", "Accumulated Depreciation"],
                ["1", ".1667", "$100,000", "$16,670", "$16,670"],
                ["2", ".3333", "$100,000", "$33,330", "$50,000"],
                ["3", ".3333", "$100,000", "$33,330", "$88,330"],
                ["4", ".1667", "$100,000", "$16,670", "$100,000"],
            ],
        )
        self.assertEqual(
            table["semantic_projection_v2"]["sparse_numeric_anchor_column_projection"]["header_source"],
            "header_candidates",
        )

    def test_header_only_fragment_is_dropped_before_following_same_schema_body_table(self) -> None:
        header_fragment = {
            "table_id": "tbl_header_only",
            "page": 1,
            "bbox": [54.0, 478.8, 558.0, 496.8],
            "detection_source": "pymupdf_builtin",
            "col_count": 5,
            "display_grid": [["Year", "Recovery Rate", "Unadjusted Basis", "Depreciation Expense", "Accumulated Depreciation"]],
            "header": [
                {"col": 1, "text": "Year"},
                {"col": 2, "text": "Recovery Rate"},
                {"col": 3, "text": "Unadjusted Basis"},
                {"col": 4, "text": "Depreciation Expense"},
                {"col": 5, "text": "Accumulated Depreciation"},
            ],
        }
        body_table = {
            "table_id": "tbl_body",
            "page": 1,
            "bbox": [58.0, 500.4, 554.0, 565.3],
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 5,
            "header": [
                {"col": 1, "text": "Year"},
                {"col": 2, "text": "Recovery Rate"},
                {"col": 3, "text": "Unadjusted Basis"},
                {"col": 4, "text": "Depreciation Expense"},
                {"col": 5, "text": "Accumulated Depreciation"},
            ],
            "display_grid": [
                ["1", ".3333", "$100,000", "$33,333", "$33,333"],
                ["2", ".4445", "$100,000", "$44,450", "$77,780"],
                ["3", ".1481", "$100,000", "$14,810", "$92,950"],
            ],
        }

        tables = [header_fragment, body_table]
        merged = merge_semantic_table_fragments_v2(tables)

        self.assertEqual(merged, 1)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["table_id"], "tbl_body")
        self.assertIn("header_only_fragment_drop", tables[0]["semantic_fragment_merge_v2"]["merge_types"])

    def test_two_column_spanning_title_row_projects_to_logical_colspan(self) -> None:
        table = {
            "table_id": "tbl_two_column_spanning_title",
            "detection_source": "structured_text_region",
            "col_count": 2,
            "display_grid": [
                ["Species", "on protected list"],
                ["Potosi Pupfish", "Cyprinodon alvarezi"],
                ["La Palma Pupfish", "Cyprinodon longidorsalis"],
                ["Butterfly Splitfin", "Ameca splendens"],
                ["Golden Skiffia", "Skiffia francesae"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "two_column_spanning_header_table")
        self.assertIn("two_column_spanning_header_projection", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Species on protected list", None],
                ["Potosi Pupfish", "Cyprinodon alvarezi"],
                ["La Palma Pupfish", "Cyprinodon longidorsalis"],
                ["Butterfly Splitfin", "Ameca splendens"],
                ["Golden Skiffia", "Skiffia francesae"],
            ],
        )
        self.assertIn(
            {
                "row": 0,
                "col": 0,
                "rowspan": 1,
                "colspan": 2,
                "text": "Species on protected list",
                "source": "two_column_spanning_header_projection",
            },
            table["logical_cells"],
        )

    def test_two_column_table_with_explicit_column_headers_keeps_column_header_semantics(self) -> None:
        table = {
            "table_id": "tbl_two_column_unit_row",
            "detection_source": "caption_anchored_horizontal_rules",
            "col_count": 2,
            "header": [
                {"col": 1, "text": "Mineral or colloid type"},
                {"col": 2, "text": "CEC of pure colloid"},
            ],
            "display_grid": [
                ["Mineral or colloid type", "CEC of pure colloid"],
                [None, "cmolc/kg"],
                ["kaolinite", "10"],
                ["illite", "30"],
                ["montmorillonite/smectite", "100"],
                ["vermiculite", "150"],
                ["humus", "200"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertNotEqual(table["table_family"], "two_column_spanning_header_table")
        self.assertNotIn("two_column_spanning_header_projection", table["semantic_projection_v2"])
        self.assertEqual(table["semantic_grid"][0], ["Mineral or colloid type", "CEC of pure colloid"])

    def test_sparse_multilevel_header_labels_project_to_logical_spans(self) -> None:
        table = {
            "table_id": "tbl_sparse_multilevel_header",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 8,
            "row_groups": [{"row": 1, "col": 0, "rowspan": 2, "text": "Properties", "source": "sparse_body_rowspan_projection"}],
            "display_grid": [
                [None, None, None, None, None, "Training Datasets", None, None],
                ["Properties", None, None, "Instruction", None, None, "Alignment", None],
                [None, None, "Alpaca-GPT4", "OpenOrca", "Synth. Math-Instruct", "Orca DPO Pairs", "Ultrafeedback Cleaned", None],
                ["Total", "# Samples", "52K", "2.91M", "126K", "12.9K", "60.8K", "126K"],
                ["Maximum", "# Samples Used", "52K", "100K", "52K", "12.9K", "60.8K", "20.1K"],
                ["Open", "Source", "O", "O", "X", "O", "O", "X"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("sparse_header_span_projection", table["semantic_projection_v2"])
        logical_cells = table["logical_cells"]
        self.assertIn(
            {
                "row": 0,
                "col": 1,
                "rowspan": 1,
                "colspan": 6,
                "text": "Training Datasets",
                "source": "sparse_header_colspan_projection",
            },
            logical_cells,
        )
        self.assertIn(
            {
                "row": 1,
                "col": 1,
                "rowspan": 1,
                "colspan": 3,
                "text": "Instruction",
                "source": "sparse_header_colspan_projection",
            },
            logical_cells,
        )
        self.assertIn(
            {
                "row": 1,
                "col": 4,
                "rowspan": 1,
                "colspan": 3,
                "text": "Alignment",
                "source": "sparse_header_colspan_projection",
            },
            logical_cells,
        )
        self.assertIn(
            {
                "row": 0,
                "col": 0,
                "rowspan": 3,
                "colspan": 1,
                "text": "Properties",
                "source": "sparse_header_rowspan_projection",
            },
            logical_cells,
        )

    def test_wrapped_record_compaction_preserves_leading_unkeyed_body_row_after_multiline_header(self) -> None:
        table = {
            "table_id": "tbl_multiline_header_with_leading_continuation",
            "detection_source": "word_clustering",
            "col_count": 5,
            "display_grid": [
                [
                    "Jurisdiction",
                    "GATS XVII Reservation",
                    "Foreign Ownership",
                    "Restrictions on",
                    "Foreign Ownership Reporting",
                ],
                [None, "(1994)", "Permitted", "Foreign Ownership", "Requirements"],
                [None, None, None, "right required to acquire desert", None],
                [None, None, None, "lands. No restrictions on lands", None],
                ["Finland", "N", "Y", "Prior approval for a foreigner's", None],
                [None, None, None, "purchase of certain businesses", None],
                [None, None, None, "may be required when it includes land purchase", None],
                ["Greece", "N", "Y", "Ownership is permitted subject to", None],
                [None, None, None, "border-zone approval requirements.", None],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("multiline_schema_header_projection", table["semantic_projection_v2"])
        self.assertIn("multicolumn_wrapped_record_compaction", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                [
                    "Jurisdiction",
                    "GATS XVII Reservation (1994)",
                    "Foreign Ownership Permitted",
                    "Restrictions on Foreign Ownership",
                    "Foreign Ownership Reporting Requirements",
                ],
                [
                    None,
                    None,
                    None,
                    "right required to acquire desert lands. No restrictions on lands",
                    None,
                ],
                [
                    "Finland",
                    "N",
                    "Y",
                    "Prior approval for a foreigner's purchase of certain businesses may be required when it includes land purchase",
                    None,
                ],
                ["Greece", "N", "Y", "Ownership is permitted subject to border-zone approval requirements.", None],
            ],
        )

    def test_multicolumn_wrapped_record_rows_compact_into_logical_records(self) -> None:
        table = {
            "table_id": "tbl_wrapped_service_matrix",
            "detection_source": "word_clustering",
            "col_count": 4,
            "row_groups": [
                {"col": 0, "start_data_row": 1, "end_data_row": 2, "rowspan": 2, "text": "1. Project creation"},
                {"col": 0, "start_data_row": 3, "end_data_row": 8, "rowspan": 6, "text": "2. Data labeling and"},
                {
                    "col": 0,
                    "start_data_row": 9,
                    "end_data_row": 10,
                    "rowspan": 2,
                    "text": "3. Pipeline configuration and Pipeline, Endpoint",
                },
                {"col": 1, "start_data_row": 3, "end_data_row": 4, "rowspan": 2, "text": "Data storage management"},
                {"col": 1, "start_data_row": 6, "end_data_row": 7, "rowspan": 2, "text": "Space"},
            ],
            "display_grid": [
                ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                [
                    "1. Project creation",
                    "Project creation and",
                    "Select document type to automatically run project creation",
                    "The intuitive UI environment allows the person in charge to quickly proceed",
                ],
                [
                    None,
                    "management",
                    "with recommended Modelset and Endpoint deployment",
                    "with the entire process from project creation to deployment",
                ],
                [
                    "2. Data labeling",
                    "Data storage management",
                    None,
                    "Conveniently manage raw data to be used for OCR Pack",
                ],
                [
                    None,
                    None,
                    "Provides convenient functions for uploading raw data and viewer",
                    None,
                ],
                [
                    None,
                    "Model training",
                    "Various basic models for each selected document",
                    "Providing a foundation for customers to implement their own OCR model",
                ],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("multicolumn_wrapped_record_compaction", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                [
                    "1. Project creation",
                    "Project creation and management",
                    "Select document type to automatically run project creation with recommended Modelset and Endpoint deployment",
                    "The intuitive UI environment allows the person in charge to quickly proceed with the entire process from project creation to deployment",
                ],
                [
                    "2. Data labeling",
                    "Data storage management",
                    "Provides convenient functions for uploading raw data and viewer",
                    "Conveniently manage raw data to be used for OCR Pack",
                ],
                [
                    None,
                    "Model training",
                    "Various basic models for each selected document",
                    "Providing a foundation for customers to implement their own OCR model",
                ],
            ],
        )

    def test_wrapped_record_compaction_keeps_new_numbered_stub_record_separate(self) -> None:
        table = {
            "table_id": "tbl_wrapped_grouped_service_records",
            "detection_source": "word_clustering",
            "col_count": 4,
            "row_groups": [
                {"col": 0, "start_data_row": 1, "end_data_row": 2, "rowspan": 2, "text": "1. Project creation"},
                {"col": 0, "start_data_row": 3, "end_data_row": 8, "rowspan": 6, "text": "2. Data labeling and"},
                {
                    "col": 0,
                    "start_data_row": 9,
                    "end_data_row": 10,
                    "rowspan": 2,
                    "text": "3. Pipeline configuration and Pipeline, Endpoint",
                },
                {"col": 1, "start_data_row": 3, "end_data_row": 4, "rowspan": 2, "text": "Data storage management"},
                {"col": 1, "start_data_row": 6, "end_data_row": 7, "rowspan": 2, "text": "Space"},
            ],
            "display_grid": [
                ["Key Functions by Main Service Flow", None, None, None],
                ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                ["1. Project creation", "Project creation and", "Select document type to run project creation", "Fast setup for the team"],
                [None, "management", "with recommended model deployment", "through the entire process"],
                ["2. Data labeling and", "Data storage management", None, "Conveniently manage raw data"],
                [None, None, "Provides convenient functions for uploading raw data", None],
                [None, "Create and manage Labeling", "Creating a Labeling Space to manage annotation", "Labeling work can be outsourced"],
                [None, "Space", "Ontology and data set version management", "supplied from which data sets can be created"],
                [None, "Model training", None, "Providing a foundation for customers"],
                [None, None, "models, training pause, re-training, and cancel functions", "OCR model specialized to customer needs"],
                ["3. Pipeline configuration and Pipeline, Endpoint", None, "Choose Detector or Parser to create a Pipeline", "Reusable OCR model foundation"],
                ["deployment", "Creation and management", "Connect Pipelines to Endpoints and deployment controllers", "Reusable OCR model foundation"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("multicolumn_wrapped_record_compaction", table["semantic_projection_v2"])
        service_stages = [row[0] for row in table["semantic_grid"][1:]]
        self.assertIn("2. Data labeling and", service_stages)
        self.assertIn("3. Pipeline configuration and Pipeline, Endpoint deployment", service_stages)
        merged_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in table["semantic_grid"])
        self.assertNotIn("2. Data labeling and 3. Pipeline", merged_text)
        self.assertNotIn("Data storage management Create Labeling Space", merged_text)

    def test_rowspan_key_repetition_compacts_wrapped_description_rows(self) -> None:
        table = {
            "table_id": "tbl_rowspan_repeated_key_descriptions",
            "detection_source": "word_clustering",
            "col_count": 5,
            "row_groups": [
                {"col": 0, "start_data_row": 2, "end_data_row": 4, "rowspan": 3, "text": "Finland"},
                {"col": 1, "start_data_row": 2, "end_data_row": 4, "rowspan": 3, "text": "N"},
                {"col": 2, "start_data_row": 2, "end_data_row": 4, "rowspan": 3, "text": "Y"},
            ],
            "display_grid": [
                ["Jurisdiction", "GATS XVII", "Foreign", "Restrictions on Foreign Ownership", "Foreign"],
                [None, "(1994)", "Permitted", None, "Reporting Requirements"],
                [None, None, None, "right required to acquire desert", None],
                ["Finland", "N", "Y", "Prior approval for a foreigner’s", None],
                [None, None, None, "purchase of certain businesses", None],
                [None, None, None, "may be required when it includes land purchase.", None],
                ["France", "N", "Y", "None.", None],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("rowspan_key_repetition_compaction", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Jurisdiction", "GATS XVII (1994)", "Foreign Permitted", "Restrictions on Foreign Ownership", "Foreign Reporting Requirements"],
                [None, None, None, "right required to acquire desert", None],
                ["Finland", "N", "Y", "Prior approval for a foreigner’s purchase of certain businesses may be required when it includes land purchase.", None],
                ["France", "N", "Y", "None.", None],
            ],
        )

    def test_single_cell_numeric_body_rows_project_to_schema_columns(self) -> None:
        table = {
            "table_id": "tbl_compact_numeric_rows",
            "detection_source": "pymupdf_builtin",
            "col_count": 7,
            "display_grid": [
                ["AMS", "Average Annual Growth", None, None, None, None, "Remittance inflows"],
                [None, "2000-2004", "2004-2009", "2009-2014", "2014-2019", "2019-2020", None],
                ["Indonesia", "9.4%", "29.5%", "4.7%", "6.4%", "-17.3%", "9,651"],
                ["Lao PDR", "4.0%", "115.7%", "38.0%", "9.5%", "-10.6%", "265"],
                ["Malaysia 18.6% 7.1% 6.9% 0.7% -11.2% 1,454", None, None, None, None, None, None],
                ["Thailand -0.9% 18.6% 11.4% 4.6% -1.2% 8,067", None, None, None, None, None, None],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("compact_single_cell_body_row_projection", table["semantic_projection_v2"])
        self.assertIn(["Malaysia", "18.6%", "7.1%", "6.9%", "0.7%", "-11.2%", "1,454"], table["semantic_grid"])
        self.assertIn(["Thailand", "-0.9%", "18.6%", "11.4%", "4.6%", "-1.2%", "8,067"], table["semantic_grid"])

    def test_single_column_keyed_long_list_projects_to_key_value_record(self) -> None:
        table = {
            "table_id": "tbl_keyed_list",
            "detection_source": "structured_text_region",
            "col_count": 1,
            "display_grid": [
                ["Filtered Task Name"],
                ["task228_arc_answer_generation_easy"],
                ["ai2_arcARCChallenge:1.0.0"],
                ["ai2_arcARCEasy:1.0.0"],
                ["task229_arc_answer_generation_hard"],
                ["hellaswag:1.1.0"],
                ["cot_gsm8k"],
                ["drop:2.0.0"],
                ["winogrande:1.1.0"],
            ],
            "data_grid": [
                ["task228_arc_answer_generation_easy"],
                ["ai2_arcARCChallenge:1.0.0"],
                ["ai2_arcARCEasy:1.0.0"],
                ["task229_arc_answer_generation_hard"],
                ["hellaswag:1.1.0"],
                ["cot_gsm8k"],
                ["drop:2.0.0"],
                ["winogrande:1.1.0"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "keyed_long_list")
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Field", "Value"],
                [
                    "Filtered Task Name",
                    "task228_arc_answer_generation_easy ai2_arcARCChallenge:1.0.0 ai2_arcARCEasy:1.0.0 task229_arc_answer_generation_hard hellaswag:1.1.0 cot_gsm8k drop:2.0.0 winogrande:1.1.0",
                ],
            ],
        )
        self.assertEqual(
            table["semantic_projection_v2"]["single_column_keyed_list_projection"]["source"],
            "single_column_keyed_long_list_projection",
        )

    def test_flowchart_matrix_splits_connector_columns_into_logical_schema(self) -> None:
        table = {
            "table_id": "tbl_flowchart",
            "detection_source": "visual_structure_grid",
            "col_count": 3,
            "header": [
                {"col": 1, "text": "Genes in DNA"},
                {"col": 2, "text": "Protein →"},
                {"col": 3, "text": "Characteristics →"},
            ],
            "display_grid": [
                ["Genes in DNA", "Protein →", "Characteristics →"],
                ["normal hemoglobin genotype", "Normal hemoglobin", "disk-shaped red blood cells"],
                ["sickle hemoglobin genotype", "Sickle cell hemoglobin", "sickle-shaped red blood cells"],
            ],
            "data_grid": [
                ["normal hemoglobin genotype", "Normal hemoglobin", "disk-shaped red blood cells"],
                ["sickle hemoglobin genotype", "Sickle cell hemoglobin", "sickle-shaped red blood cells"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "flowchart_matrix")
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Genes in DNA", "→", "Protein", "→", "Characteristics"],
                [
                    "normal hemoglobin genotype",
                    "→",
                    "Normal hemoglobin",
                    "→",
                    "disk-shaped red blood cells",
                ],
                [
                    "sickle hemoglobin genotype",
                    "→",
                    "Sickle cell hemoglobin",
                    "→",
                    "sickle-shaped red blood cells",
                ],
            ],
        )
        self.assertEqual(
            table["semantic_projection_v2"]["flowchart_connector_projection"]["source"],
            "flowchart_connector_column_projection",
        )

    def test_compressed_image_reagent_matrix_projects_multi_value_rows_to_columns(self) -> None:
        table = {
            "table_id": "tbl_compressed_reagent_matrix",
            "detection_source": "embedded_image_ocr",
            "col_count": 2,
            "display_grid": [
                [None, "For use with CarolinaBLU'm stain:"],
                ["BamHI-Hindlli", "Restriction Evidence H20 Suspect 1 Suspect 2"],
                ["Tube restriction", "Buffer-RNase DNA A or B"],
                ["enzyme mixture", None],
                ["S1", "3 μL 10 μL 2 μL"],
                ["S2", "10 μL 3 μL 2 μL 3 μL"],
                ["EA or EB", "10 μL 3 μL 3 μL 2 μL"],
            ],
            "data_grid": [
                ["BamHI-Hindlli", "Restriction Evidence H20 Suspect 1 Suspect 2"],
                ["Tube restriction", "Buffer-RNase DNA A or B"],
                ["enzyme mixture", None],
                ["S1", "3 μL 10 μL 2 μL"],
                ["S2", "10 μL 3 μL 2 μL 3 μL"],
                ["EA or EB", "10 μL 3 μL 3 μL 2 μL"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "compressed_image_measurement_matrix")
        self.assertEqual(
            table["semantic_grid"],
            [
                [
                    "Tube",
                    "BamHI-Hindlli restriction enzyme mixture",
                    "Restriction Buffer-RNase",
                    "Suspect 1 DNA",
                    "Suspect 2 DNA",
                    "Evidence A or B",
                    "H20",
                ],
                ["S1", "3 μL", "3 μL", "10 μL", None, None, "2 μL"],
                ["S2", "3 μL", "3 μL", None, "10 μL", None, "2 μL"],
                ["EA or EB", "3 μL", "3 μL", None, None, "10 μL", "2 μL"],
            ],
        )
        self.assertEqual(
            table["semantic_projection_v2"]["compressed_measurement_matrix_projection"]["source"],
            "compressed_image_measurement_matrix_projection",
        )

    def test_table_cell_display_projection_repairs_high_confidence_scientific_ocr_atoms(self) -> None:
        from parsers.pdf.table_modules.cell_text_projection import project_table_cell_display_text

        self.assertEqual(project_table_cell_display_text("H20"), "H2O")
        self.assertEqual(project_table_cell_display_text("BamHI-Hindlli restriction enzyme mixture"), "BamHI-HindIII restriction enzyme mixture")

    def test_sparse_multilevel_header_fills_unique_blank_leaf_from_caption_context(self) -> None:
        table = {
            "table_id": "tbl_sparse_header_blank_leaf",
            "detection_source": "text_aligned_borderless_grid",
            "table_family": "rowspan_grouped_table",
            "title": (
                "Training datasets used for the instruction and alignment tuning stages. "
                "For alignment tuning, we employed the Orca DPO Pairs, "
                "Ultrafeedback Cleaned, and Synth. Math-Alignment datasets."
            ),
            "col_count": 7,
            "row_groups": [
                {"col": 0, "start_row": 0, "rowspan": 3, "text": "Properties", "source": "sparse_header_rowspan_projection"}
            ],
            "display_grid": [
                [None, None, None, None, "Training Datasets", None, None],
                ["Properties", None, "Instruction", None, None, "Alignment", None],
                [None, "Alpaca-GPT4", "OpenOrca", "Synth. Math-Instruct", "Orca DPO Pairs", "Ultrafeedback Cleaned", None],
                ["Total # Samples", "52K", "2.91M", "126K", "12.9K", "60.8K", "126K"],
                ["Maximum # Samples Used", "52K", "100K", "52K", "12.9K", "60.8K", "20.1K"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("caption_context_blank_header_leaf_fill", table["semantic_projection_v2"])
        self.assertTrue(any("Synth. Math-Alignment" in row for row in table["semantic_grid"]))

    def test_split_stub_projection_preserves_multilevel_header_leaf_columns(self) -> None:
        table = {
            "table_id": "tbl_split_stub_with_multilevel_leaf_headers",
            "detection_source": "text_aligned_borderless_grid",
            "table_family": "rowspan_grouped_table",
            "title": (
                "Training datasets used for instruction tuning with Alpaca-GPT4, OpenOrca, "
                "and Synth. Math-Instruct, and alignment tuning with Orca DPO Pairs, "
                "Ultrafeedback Cleaned, and Synth. Math-Alignment."
            ),
            "col_count": 8,
            "row_groups": [
                {"col": 0, "start_row": 0, "rowspan": 3, "text": "Properties", "source": "sparse_header_rowspan_projection"}
            ],
            "display_grid": [
                [None, None, None, None, "Training Datasets", None, None, None],
                ["Properties", None, None, "Instruction", None, None, "Alignment", None],
                [None, "Alpaca-GPT4", "OpenOrca", "Synth. Math-Instruct", "Orca DPO Pairs", "Ultrafeedback Cleaned", "Synth. Math-Alignment", None],
                ["Total", "# Samples", "52K", "2.91M", "126K", "12.9K", "60.8K", "126K"],
                ["Maximum", "# Samples Used", "52K", "100K", "52K", "12.9K", "60.8K", "20.1K"],
                ["Open", "Source", "O", "O", "X", "O", "O", "X"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("split_leading_stub_label_column_merge", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                [None, None, None, "Training Datasets", None, None, None],
                ["Properties", None, "Instruction", None, None, "Alignment", None],
                [None, "Alpaca-GPT4", "OpenOrca", "Synth. Math-Instruct", "Orca DPO Pairs", "Ultrafeedback Cleaned", "Synth. Math-Alignment"],
                ["Total # Samples", "52K", "2.91M", "126K", "12.9K", "60.8K", "126K"],
                ["Maximum # Samples Used", "52K", "100K", "52K", "12.9K", "60.8K", "20.1K"],
                ["Open Source", "O", "O", "X", "O", "O", "X"],
            ],
        )
        self.assertNotIn("caption_context_blank_header_leaf_fill", table["semantic_projection_v2"])

    def test_packed_leading_stub_and_marker_column_projects_boolean_matrix(self) -> None:
        table = {
            "table_id": "tbl_packed_stub_marker_matrix",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 11,
            "title": "Ablation studies on datasets used for instruction tuning.",
            "header": [
                {"col": 1, "text": "Column 1"},
                {"col": 2, "text": "Model Dataset A"},
                {"col": 3, "text": "Dataset B Synth."},
                {"col": 4, "text": "Math-Instruct"},
                {"col": 5, "text": "Score"},
                {"col": 6, "text": "Metric 1"},
                {"col": 7, "text": "Metric 2"},
                {"col": 8, "text": "Metric 3"},
                {"col": 9, "text": "Metric 4"},
                {"col": 10, "text": "Metric 5"},
                {"col": 11, "text": "Metric 6"},
            ],
            "display_grid": [
                ["Column 1", "Model Dataset A", "Dataset B Synth.", "Math-Instruct", "Score", "Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5", "Metric 6"],
                [None, "SFT v1 O", "X", "X", "69.15", "67.66", "86.03", "65.88", "60.12", "82.95", "52.24"],
                ["SFT v2", "O", "O", "X", "69.21", "65.36", "85.39", "65.93", "58.47", "82.79", "57.32"],
                ["SFT v3", "O", "O", "O", "70.03", "65.87", "85.55", "65.31", "57.93", "81.37", "64.14"],
                ["SFT v4", "O", "X", "O", "70.88", "67.32", "85.87", "65.87", "58.97", "82.48", "64.75"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("packed_leading_stub_marker_projection", table["semantic_projection_v2"])
        self.assertNotIn("split_leading_stub_label_column_merge", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Model", "Dataset A", "Dataset B", "Synth. Math-Instruct", "Score", "Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5", "Metric 6"],
                ["SFT v1", "O", "X", "X", "69.15", "67.66", "86.03", "65.88", "60.12", "82.95", "52.24"],
                ["SFT v2", "O", "O", "X", "69.21", "65.36", "85.39", "65.93", "58.47", "82.79", "57.32"],
                ["SFT v3", "O", "O", "O", "70.03", "65.87", "85.55", "65.31", "57.93", "81.37", "64.14"],
                ["SFT v4", "O", "X", "O", "70.88", "67.32", "85.87", "65.87", "58.97", "82.48", "64.75"],
            ],
        )

    def test_late_dense_header_ignores_fragmented_caption_rows_before_body_header(self) -> None:
        table = {
            "table_id": "tbl_caption_fragments_before_metric_header",
            "detection_source": "structured_text_region",
            "col_count": 6,
            "header": [
                {"col": 1, "text": "Table"},
                {"col": 2, "text": "8: Task"},
                {"col": 3, "text": "names that"},
                {"col": 4, "text": "we use to"},
                {"col": 5, "text": "filter data for"},
                {"col": 6, "text": "FLAN"},
            ],
            "display_grid": [
                ["winogrande:1.1.0", None, None, None, None, None],
                ["Table", "8: Task", "names that", "we use to", "filter data for", "FLAN"],
                ["derived", "datasets", "such as", "OpenOrca.", None, None],
                ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"],
                ["0.06", "N/A", "0.15", "0.28", "N/A", "0.70"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertIn("leading_boundary_header_projection", table["semantic_projection_v2"])
        self.assertEqual(
            table["semantic_grid"],
            [
                ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"],
                ["0.06", "N/A", "0.15", "0.28", "N/A", "0.70"],
            ],
        )
        self.assertNotIn("Table", table["semantic_grid"][0])

    def test_two_column_inventory_removes_repeated_boundary_rows_in_semantic_grid(self) -> None:
        table = {
            "table_id": "tbl_inventory",
            "detection_source": "structured_text_region",
            "col_count": 2,
            "header": [
                {"col": 1, "text": "Reagents"},
                {"col": 2, "text": "Supplies and Equipment"},
            ],
            "display_grid": [
                ["Reagents", "Supplies and Equipment"],
                ["At each student station:", "Microcentrifuge tube rack"],
                ["Resuspended DNA or ethanol precipitates from Part 1*", "3 1.5-mL microcentrifuge tubes Micropipet, 1- 20 uL"],
                [None, "Micropipet tips"],
                [None, "Micropipet tips"],
                ["To be shared by all groups:", "Beaker or similar container for waste"],
                ["Evidence A DNA*", "Beaker or similar container filled with ice"],
            ],
            "data_grid": [
                ["At each student station:", "Microcentrifuge tube rack"],
                ["Resuspended DNA or ethanol precipitates from Part 1*", "3 1.5-mL microcentrifuge tubes Micropipet, 1- 20 uL"],
                [None, "Micropipet tips"],
                [None, "Micropipet tips"],
                ["To be shared by all groups:", "Beaker or similar container for waste"],
                ["Evidence A DNA*", "Beaker or similar container filled with ice"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "two_column_inventory")
        semantic_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in table["semantic_grid"])
        self.assertIn("To be shared by all groups:", semantic_text)
        self.assertEqual(semantic_text.count("Micropipet tips"), 1)
        self.assertEqual(table["semantic_projection_v2"]["duplicate_boundary_rows_removed"], 1)

    def test_two_column_inventory_parallel_lists_compact_each_column_independently(self) -> None:
        table = {
            "table_id": "tbl_parallel_inventory",
            "detection_source": "visual_structure_grid",
            "col_count": 2,
            "header": [
                {"col": 1, "text": "Reagents"},
                {"col": 2, "text": "Supplies and Equipment"},
            ],
            "display_grid": [
                ["Reagents", "Supplies and Equipment"],
                ["At each station:", "Tube rack"],
                ["Sample DNA*", "Three microcentrifuge tubes"],
                ["To be shared by all groups:", "Micropipet"],
                ["Evidence A DNA*", "Micropipet tips"],
                ["Evidence B DNA*", "Waste beaker"],
                ["Restriction buffer*", "Ice beaker"],
                ["Enzyme mixture*", "Permanent marker"],
                ["Sterile water", "Water bath"],
            ],
            "data_grid": [
                ["At each station:", "Tube rack"],
                ["Sample DNA*", "Three microcentrifuge tubes"],
                ["To be shared by all groups:", "Micropipet"],
                ["Evidence A DNA*", "Micropipet tips"],
                ["Evidence B DNA*", "Waste beaker"],
                ["Restriction buffer*", "Ice beaker"],
                ["Enzyme mixture*", "Permanent marker"],
                ["Sterile water", "Water bath"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "two_column_inventory")
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Reagents", "Supplies and Equipment"],
                [
                    "At each station: Sample DNA* To be shared by all groups: Evidence A DNA* Evidence B DNA* Restriction buffer* Enzyme mixture* Sterile water",
                    "Tube rack Three microcentrifuge tubes Micropipet Micropipet tips Waste beaker Ice beaker Permanent marker Water bath",
                ],
            ],
        )
        self.assertEqual(
            table["semantic_projection_v2"]["parallel_inventory_list_compaction"]["source"],
            "parallel_inventory_list_compaction",
        )

    def test_two_column_grouped_hierarchy_is_not_compacted_as_parallel_inventory_list(self) -> None:
        table = {
            "table_id": "tbl_grouped_hierarchy",
            "detection_source": "visual_structure_grid",
            "col_count": 2,
            "header": [
                {"col": 1, "text": "Area"},
                {"col": 2, "text": "Competence"},
            ],
            "display_grid": [
                ["Area", "Competence"],
                ["1. Embodying sustainability values", "1.1 Valuing sustainability"],
                [None, "1.2 Supporting fairness"],
                [None, "1.3 Promoting nature"],
                ["2. Embracing complexity", "2.1 Systems thinking"],
                [None, "2.2 Critical thinking"],
            ],
            "data_grid": [
                ["1. Embodying sustainability values", "1.1 Valuing sustainability"],
                [None, "1.2 Supporting fairness"],
                [None, "1.3 Promoting nature"],
                ["2. Embracing complexity", "2.1 Systems thinking"],
                [None, "2.2 Critical thinking"],
            ],
            "row_groups": [
                {"col": 0, "start_data_row": 1, "rowspan": 3, "text": "1. Embodying sustainability values"},
                {"col": 0, "start_data_row": 4, "rowspan": 2, "text": "2. Embracing complexity"},
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "rowspan_grouped_table")
        self.assertNotIn("parallel_inventory_list_compaction", table["semantic_projection_v2"])
        self.assertEqual(table["semantic_grid"][1][0], "1. Embodying sustainability values")
        self.assertEqual(table["semantic_grid"][2][1], "1.2 Supporting fairness")
        self.assertTrue(
            [
                cell for cell in table["logical_cells"]
                if cell.get("text") == "1. Embodying sustainability values" and cell.get("rowspan") == 3
            ]
        )

    def test_two_column_grouped_hierarchy_repairs_child_items_shifted_into_parent_column(self) -> None:
        table = {
            "table_id": "tbl_shifted_child_items",
            "detection_source": "pymupdf_builtin",
            "col_count": 2,
            "display_grid": [
                ["Area", "Competence"],
                ["1. Embodying sustainability values", "1.1 Valuing sustainability"],
                ["1.2 Supporting fairness", None],
                [None, "1.3 Promoting nature"],
                ["2. Embracing complexity in sustainability 2.1 Systems thinking", None],
                [None, "2.2 Critical thinking"],
                ["2.3 Problem framing", None],
                ["3. Envisioning sustainable futures", "3.1 Futures literacy"],
                [None, "3.2 Adaptability"],
            ],
            "data_grid": [
                ["1. Embodying sustainability values", "1.1 Valuing sustainability"],
                ["1.2 Supporting fairness", None],
                [None, "1.3 Promoting nature"],
                ["2. Embracing complexity in sustainability 2.1 Systems thinking", None],
                [None, "2.2 Critical thinking"],
                ["2.3 Problem framing", None],
                ["3. Envisioning sustainable futures", "3.1 Futures literacy"],
                [None, "3.2 Adaptability"],
            ],
            "row_groups": [
                {"col": 0, "start_data_row": 1, "rowspan": 3, "text": "1. Embodying sustainability values"},
                {"col": 0, "start_data_row": 4, "rowspan": 3, "text": "2. Embracing complexity in sustainability"},
                {"col": 0, "start_data_row": 7, "rowspan": 2, "text": "3. Envisioning sustainable futures"},
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "rowspan_grouped_table")
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Area", "Competence"],
                ["1. Embodying sustainability values", "1.1 Valuing sustainability"],
                [None, "1.2 Supporting fairness"],
                [None, "1.3 Promoting nature"],
                ["2. Embracing complexity in sustainability", "2.1 Systems thinking"],
                [None, "2.2 Critical thinking"],
                [None, "2.3 Problem framing"],
                ["3. Envisioning sustainable futures", "3.1 Futures literacy"],
                [None, "3.2 Adaptability"],
            ],
        )
        self.assertEqual(
            table["semantic_projection_v2"]["two_column_hierarchy_child_column_projection"]["source"],
            "two_column_hierarchy_child_column_projection",
        )
        self.assertTrue(
            [
                cell for cell in table["logical_cells"]
                if cell.get("text") == "1. Embodying sustainability values" and cell.get("row") == 1 and cell.get("rowspan") == 3
            ]
        )
        self.assertTrue(
            [
                cell for cell in table["logical_cells"]
                if cell.get("text") == "2. Embracing complexity in sustainability" and cell.get("row") == 4 and cell.get("rowspan") == 3
            ]
        )

    def test_two_column_key_value_label_after_value_row_projects_logical_pairs(self) -> None:
        table = {
            "table_id": "tbl_label_after_value",
            "detection_source": "pymupdf_builtin",
            "col_count": 2,
            "display_grid": [
                ["Competence Area", "#1 THE 3 RS: RECYCLE-REUSE-REDUCE"],
                [None, "To know the basics of the 3 Rs and their importance."],
                ["Competence Statement", None],
                ["Learning Outcomes", None],
                [None, "To understand the meaning of reducing, reusing and recycling."],
                ["Knowledge", None],
                [None, "To implement different ways of waste management."],
                ["Skills", None],
                [None, "To acquire a proactive approach."],
                ["Attitudes and Values", None],
            ],
            "data_grid": [
                [None, "To know the basics of the 3 Rs and their importance."],
                ["Competence Statement", None],
                ["Learning Outcomes", None],
                [None, "To understand the meaning of reducing, reusing and recycling."],
                ["Knowledge", None],
                [None, "To implement different ways of waste management."],
                ["Skills", None],
                [None, "To acquire a proactive approach."],
                ["Attitudes and Values", None],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "two_column_inventory")
        self.assertEqual(
            table["semantic_grid"],
            [
                ["Competence Area", "#1 THE 3 RS: RECYCLE-REUSE-REDUCE"],
                ["Competence Statement", "To know the basics of the 3 Rs and their importance."],
                ["Learning Outcomes", None],
                ["Knowledge", "To understand the meaning of reducing, reusing and recycling."],
                ["Skills", "To implement different ways of waste management."],
                ["Attitudes and Values", "To acquire a proactive approach."],
            ],
        )
        self.assertEqual(
            table["semantic_projection_v2"]["label_after_value_pair_projection"]["source"],
            "two_column_label_after_value_pair_projection",
        )

    def test_two_column_blank_form_comparison_projects_missing_stub_column_and_trims_prose_tail(self) -> None:
        table = {
            "table_id": "tbl_blank_form",
            "detection_source": "structured_text_region",
            "col_count": 2,
            "header": [
                {"col": 1, "text": "Mitosis (begins with a single cell)"},
                {"col": 2, "text": "Meiosis (begins with a single cell)"},
            ],
            "display_grid": [
                ["Mitosis", "Meiosis"],
                ["(begins with a single cell)", "(begins with a single cell)"],
                ["# chromosomes in parent", None],
                ["cells", None],
                ["# DNA replications", None],
                ["# nuclear divisions", None],
                ["# daughter cells produced", None],
                ["purpose", None],
                [
                    "5. Using your beads, strings, and magnets recreate the",
                    "process of meiosis. Ensuring you",
                ],
                ["Instructor signature:", None],
            ],
            "data_grid": [
                ["# chromosomes in parent", None],
                ["cells", None],
                ["# DNA replications", None],
                ["# nuclear divisions", None],
                ["# daughter cells produced", None],
                ["purpose", None],
                [
                    "5. Using your beads, strings, and magnets recreate the",
                    "process of meiosis. Ensuring you",
                ],
                ["Instructor signature:", None],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "blank_form_comparison_matrix")
        self.assertEqual(
            table["semantic_grid"],
            [
                [None, "Mitosis (begins with a single cell)", "Meiosis (begins with a single cell)"],
                ["# chromosomes in parent cells", None, None],
                ["# DNA replications", None, None],
                ["# nuclear divisions", None, None],
                ["# daughter cells produced", None, None],
                ["purpose", None, None],
            ],
        )
        self.assertEqual(
            table["semantic_projection_v2"]["blank_stub_column_projection"]["source"],
            "blank_form_comparison_projection",
        )
        self.assertEqual(
            table["semantic_projection_v2"]["blank_stub_column_projection"]["trimmed_row_count"],
            2,
        )

    def test_comparison_matrix_preserves_projected_row_header_role(self) -> None:
        table = {
            "table_id": "tbl_matrix",
            "detection_source": "visual_structure_grid",
            "col_count": 3,
            "header": [
                {"col": 1, "text": "Column 1"},
                {"col": 2, "text": "Mitosis"},
                {"col": 3, "text": "Meiosis"},
            ],
            "display_grid": [
                [None, "Mitosis", "Meiosis"],
                ["# chromosomes in parent cells", "46", "46"],
                ["# DNA replications", "1", "1"],
                ["# nuclear divisions", "1", "2"],
            ],
            "data_grid": [
                ["# chromosomes in parent cells", "46", "46"],
                ["# DNA replications", "1", "1"],
                ["# nuclear divisions", "1", "2"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "comparison_matrix")
        self.assertEqual(table["semantic_header"][0]["role"], "projected_row_header")
        self.assertEqual(table["semantic_header"][1]["text"], "Mitosis")
        self.assertEqual(table["semantic_header"][2]["text"], "Meiosis")

    def test_split_leading_stub_projection_is_not_reprojected_as_missing_corner(self) -> None:
        table = {
            "table_id": "tbl_split_stub_not_missing_corner",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 10,
            "header": [
                {"col": 1, "text": "Model"},
                {"col": 2, "text": "Merge Method"},
                {"col": 4, "text": "H6 (Avg.)"},
                {"col": 5, "text": "ARC"},
                {"col": 6, "text": "HellaSwag"},
                {"col": 7, "text": "MMLU"},
                {"col": 8, "text": "TruthfulQA"},
                {"col": 9, "text": "Winogrande"},
                {"col": 10, "text": "GSM8K"},
            ],
            "display_grid": [
                ["Model", "Merge Method", None, "H6 (Avg.)", "ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"],
                ["Merge v1", "Average (0.5,", "0.5)", "74.00", "71.16", "88.01", "66.14", "71.71", "82.08", "64.90"],
                ["Merge v2", "Average (0.4,", "0.6)", "73.93", "71.08", "88.08", "66.27", "71.89", "81.77", "64.52"],
                ["Merge v3", "Average (0.6,", "0.4)", "74.05", "71.08", "87.88", "66.13", "71.61", "82.08", "65.50"],
                ["Merge v4", "SLERP", None, "73.96", "71.16", "88.03", "66.25", "71.79", "81.93", "64.59"],
            ],
            "data_grid": [
                ["Merge v1", "Average (0.5,", "0.5)", "74.00", "71.16", "88.01", "66.14", "71.71", "82.08", "64.90"],
                ["Merge v2", "Average (0.4,", "0.6)", "73.93", "71.08", "88.08", "66.27", "71.89", "81.77", "64.52"],
                ["Merge v3", "Average (0.6,", "0.4)", "74.05", "71.08", "87.88", "66.13", "71.61", "82.08", "65.50"],
                ["Merge v4", "SLERP", None, "73.96", "71.16", "88.03", "66.25", "71.79", "81.93", "64.59"],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "comparison_matrix")
        self.assertIn("split_leading_stub_label_column_merge", table["semantic_projection_v2"])
        self.assertNotIn("missing_leading_stub_header_projection", table["semantic_projection_v2"])
        self.assertEqual(table["semantic_grid"][1][0], "Merge v1 Average (0.5,")
        self.assertEqual(table["semantic_grid"][1][1], "0.5)")

    def test_comparison_matrix_semantic_boundary_excludes_following_numbered_prose(self) -> None:
        table = {
            "table_id": "tbl_matrix_with_body_tail",
            "detection_source": "text_aligned_borderless_grid",
            "col_count": 3,
            "header": [
                {"col": 1, "text": "Column 1"},
                {"col": 2, "text": "Mitosis"},
                {"col": 3, "text": "Meiosis"},
            ],
            "display_grid": [
                [None, "Mitosis", "Meiosis"],
                ["# chromosomes in parent cells", None, None],
                ["# DNA replications", None, None],
                ["# nuclear divisions", None, None],
                ["# daughter cells produced", None, None],
                ["purpose", None, None],
                [
                    "5. Using your beads, strings, and magnets recreate the",
                    "process of meiosis. Ensuring you",
                    None,
                ],
                [
                    "have two different colored beads, demonstrate the process of",
                    "crossing over. When you",
                    None,
                ],
                ["Instructor signature:", None, None],
            ],
            "data_grid": [
                ["# chromosomes in parent cells", None, None],
                ["# DNA replications", None, None],
                ["# nuclear divisions", None, None],
                ["# daughter cells produced", None, None],
                ["purpose", None, None],
                [
                    "5. Using your beads, strings, and magnets recreate the",
                    "process of meiosis. Ensuring you",
                    None,
                ],
                [
                    "have two different colored beads, demonstrate the process of",
                    "crossing over. When you",
                    None,
                ],
                ["Instructor signature:", None, None],
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "comparison_matrix")
        semantic_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in table["semantic_grid"])
        self.assertIn("# daughter cells produced", semantic_text)
        self.assertNotIn("Using your beads", semantic_text)
        self.assertEqual(
            table["semantic_projection_v2"]["semantic_row_run_boundary_refinement"]["trimmed_row_count"],
            3,
        )

    def test_rowspan_groups_are_materialized_as_logical_cell_spans(self) -> None:
        table = {
            "table_id": "tbl_rowspan",
            "detection_source": "word_clustering",
            "col_count": 4,
            "header": [
                {"col": 1, "text": "Service Stage"},
                {"col": 2, "text": "Function Name"},
                {"col": 3, "text": "Explanation"},
                {"col": 4, "text": "Expected Benefit"},
            ],
            "display_grid": [
                ["Service Stage", "Function Name", "Explanation", "Expected Benefit"],
                ["1. Project creation", "Project creation and", "Select document type", "Fast setup"],
                [None, "management", "Pipeline configuration", "Improve work efficiency"],
                ["2. Data labeling", "Data storage management", None, "Manage raw data"],
                [None, None, "Upload raw data", None],
            ],
            "data_grid": [
                ["1. Project creation", "Project creation and", "Select document type", "Fast setup"],
                [None, "management", "Pipeline configuration", "Improve work efficiency"],
                ["2. Data labeling", "Data storage management", None, "Manage raw data"],
                [None, None, "Upload raw data", None],
            ],
            "row_groups": [
                {
                    "col": 0,
                    "start_data_row": 1,
                    "end_data_row": 2,
                    "rowspan": 2,
                    "text": "1. Project creation",
                    "source": "sparse_body_rowspan_projection",
                }
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "rowspan_grouped_table")
        self.assertTrue(
            [
                cell for cell in table["logical_cells"]
                if cell.get("text") == "1. Project creation" and cell.get("rowspan") == 2
            ]
        )


    def test_projected_stub_matrix_compacts_visual_wrap_rows_into_logical_records(self) -> None:
        table = {
            "table_id": "tbl_projected_stub",
            "detection_source": "visual_structure_grid",
            "col_count": 3,
            "header": [
                {"col": 1, "text": "OCR"},
                {"col": 2, "text": "Recommendation"},
                {"col": 3, "text": "Product semantic search"},
            ],
            "display_grid": [
                ["OCR", "Recommendation", "Product semantic search"],
                ["A solution that recognizes characters in an", "A solution that recommends the best products and", "A solution that enables semantic search, analyzes and"],
                ["image and extracts necessary information", "contents", "organizes key information in unstructured text data"],
                ["Pack", None, None],
                [None, None, "into a standardized form (DB)"],
                ["Applicable to all fields that require text extraction", "Applicable to all fields that use any form of", "Applicable to all fields that deal with various types of"],
                ["from standardized documents, such as receipts,", "recommendation including alternative products,", "unstructured data containing text information that"],
                ["bills, credit cards, ID cards, certificates, and medical Application", "products and contents that are likely to be", "require semantic search and conversion into a DB"],
                ["receipts", "purchased next", None],
            ],
            "data_grid": [
                ["A solution that recognizes characters in an", "A solution that recommends the best products and", "A solution that enables semantic search, analyzes and"],
                ["image and extracts necessary information", "contents", "organizes key information in unstructured text data"],
                ["Pack", None, None],
                [None, None, "into a standardized form (DB)"],
                ["Applicable to all fields that require text extraction", "Applicable to all fields that use any form of", "Applicable to all fields that deal with various types of"],
                ["from standardized documents, such as receipts,", "recommendation including alternative products,", "unstructured data containing text information that"],
                ["bills, credit cards, ID cards, certificates, and medical Application", "products and contents that are likely to be", "require semantic search and conversion into a DB"],
                ["receipts", "purchased next", None],
            ],
            "row_groups": [
                {
                    "col": 0,
                    "start_data_row": 3,
                    "end_data_row": 4,
                    "rowspan": 2,
                    "text": "Pack",
                    "source": "sparse_body_rowspan_projection",
                }
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        self.assertEqual(table["table_family"], "projected_stub_matrix")
        self.assertEqual(table["semantic_grid"][0], [None, "OCR", "Recommendation", "Product semantic search"])
        self.assertEqual(table["semantic_grid"][1][0], "Pack")
        self.assertIn("image and extracts necessary information", table["semantic_grid"][1][1])
        self.assertIn("into a standardized form (DB)", table["semantic_grid"][1][3])
        self.assertEqual(table["semantic_grid"][2][0], "Application")
        self.assertIn("receipts", table["semantic_grid"][2][1])
        self.assertIn("purchased next", table["semantic_grid"][2][2])
        self.assertTrue(
            [
                cell for cell in table["logical_cells"]
                if cell.get("row") == 0 and cell.get("col") == 1 and cell.get("text") == "OCR"
            ]
        )

    def test_projected_stub_matrix_prefers_tail_label_over_preceding_value_phrase(self) -> None:
        table = {
            "table_id": "tbl_projected_stub_tail",
            "detection_source": "visual_structure_grid",
            "col_count": 3,
            "header": [
                {"col": 1, "text": "OCR"},
                {"col": 2, "text": "Recommendation"},
                {"col": 3, "text": "Product semantic search"},
            ],
            "display_grid": [
                ["OCR", "Recommendation", "Product semantic search"],
                ["A solution that recognizes characters in an", "A solution that recommends the best products and", "A solution that enables semantic search, analyzes and"],
                ["image and extracts necessary information", "contents", "organizes key information in unstructured text data"],
                ["Pack", None, None],
                [None, None, "into a standardized form (DB)"],
                ["Applicable to all fields that require text extraction", "Applicable to all fields that use any form of", "Applicable to all fields that deal with various types of"],
                ["from standardized documents, such as receipts,", "recommendation including alternative products,", "unstructured data containing text information that"],
                ["bills, credit cards, ID cards, certificates, and medical Application", "products and contents that are likely to be", "require semantic search and conversion into a DB"],
                ["receipts", "purchased next", None],
                [
                    "Achieved 1st place in the OCR World Competition The team includes specialists who have presented papers in AI conferences Highlight",
                    "Team with specialists and technologies that received Kaggle's Gold Medal recommendation",
                    "World's No.1 in E-commerce subject (Shopee)",
                ],
            ],
            "data_grid": [
                ["A solution that recognizes characters in an", "A solution that recommends the best products and", "A solution that enables semantic search, analyzes and"],
                ["image and extracts necessary information", "contents", "organizes key information in unstructured text data"],
                ["Pack", None, None],
                [None, None, "into a standardized form (DB)"],
                ["Applicable to all fields that require text extraction", "Applicable to all fields that use any form of", "Applicable to all fields that deal with various types of"],
                ["from standardized documents, such as receipts,", "recommendation including alternative products,", "unstructured data containing text information that"],
                ["bills, credit cards, ID cards, certificates, and medical Application", "products and contents that are likely to be", "require semantic search and conversion into a DB"],
                ["receipts", "purchased next", None],
                [
                    "Achieved 1st place in the OCR World Competition The team includes specialists who have presented papers in AI conferences Highlight",
                    "Team with specialists and technologies that received Kaggle's Gold Medal recommendation",
                    "World's No.1 in E-commerce subject (Shopee)",
                ],
            ],
            "row_groups": [
                {
                    "col": 0,
                    "start_data_row": 3,
                    "end_data_row": 4,
                    "rowspan": 2,
                    "text": "Pack",
                    "source": "sparse_body_rowspan_projection",
                }
            ],
        }

        changed = apply_table_semantic_projection_v2(table)

        self.assertTrue(changed)
        labels = [row[0] for row in table["semantic_grid"][1:]]
        self.assertIn("Highlight", labels)
        self.assertNotIn("AI conferences Highlight", labels)
        highlight_row = next(row for row in table["semantic_grid"] if row[0] == "Highlight")
        self.assertIn("AI conferences", highlight_row[1])
        self.assertIn("E-commerce subject (Shopee)", highlight_row[3])

    def test_matrix_right_edge_tail_fragment_merges_into_owner_table(self) -> None:
        main_table = {
            "table_id": "tbl_matrix",
            "page": 1,
            "detection_source": "visual_structure_grid",
            "table_family": "comparison_matrix",
            "bbox": [60.0, 100.0, 540.0, 260.0],
            "col_count": 3,
            "header": [
                {"col": 1, "text": "Genes in DNA"},
                {"col": 2, "text": "Protein"},
                {"col": 3, "text": "Characteristics"},
            ],
            "display_grid": [
                ["Genes in DNA", "Protein", "Characteristics"],
                ["Normal allele", "Normal hemoglobin", "normal health"],
                ["normal code", "same protein", "same shape"],
                ["normal genotype", "soluble protein", "small vessel flow"],
                ["Sickle allele", "Sickle hemoglobin", "sickle-shaped cells"],
                [None, None, "If sickle hemoglobin clumps"],
            ],
            "data_grid": [
                ["Normal allele", "Normal hemoglobin", "normal health"],
                ["normal code", "same protein", "same shape"],
                ["normal genotype", "soluble protein", "small vessel flow"],
                ["Sickle allele", "Sickle hemoglobin", "sickle-shaped cells"],
                [None, None, "If sickle hemoglobin clumps"],
            ],
            "semantic_grid": [
                ["Genes in DNA", "Protein", "Characteristics"],
                ["Normal allele", "Normal hemoglobin", "normal health"],
                ["normal code", "same protein", "same shape"],
                ["normal genotype", "soluble protein", "small vessel flow"],
                ["Sickle allele", "Sickle hemoglobin", "sickle-shaped cells"],
                [None, None, "If sickle hemoglobin clumps"],
            ],
            "semantic_projection_v2": {
                "version": 2,
                "table_family": "comparison_matrix",
                "source": "table_semantic_projection_v2",
            },
        }
        tail_fragment = {
            "table_id": "tbl_tail",
            "page": 1,
            "detection_source": "structured_text_region",
            "title": "If sickle hemoglobin clumps",
            "bbox": [300.0, 235.0, 545.0, 330.0],
            "col_count": 1,
            "display_grid": [
                ["If sickle hemoglobin clumps"],
                ["in long rods"],
                ["sickle-shaped red cells"],
                ["clogged small vessels"],
            ],
            "data_grid": [
                ["If sickle hemoglobin clumps"],
                ["in long rods"],
                ["sickle-shaped red cells"],
                ["clogged small vessels"],
            ],
        }
        tables = [main_table, tail_fragment]

        merged = merge_semantic_table_fragments_v2(tables)

        self.assertEqual(merged, 1)
        self.assertEqual(len(tables), 1)
        merged_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in tables[0]["display_grid"]
        )
        self.assertIn("If sickle hemoglobin clumps in long rods", merged_text)
        self.assertIn("sickle-shaped red cells", merged_text)
        self.assertIn("clogged small vessels", merged_text)
        self.assertIn("matrix_right_edge_tail_fragment", tables[0]["semantic_fragment_merge_v2"]["merge_types"])
        semantic_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in tables[0]["semantic_grid"]
        )
        self.assertIn("If sickle hemoglobin clumps in long rods", semantic_text)

    def test_flowchart_tail_fragment_prefers_owner_cell_continuation_when_spatially_attached(self) -> None:
        main_table = {
            "table_id": "tbl_flowchart",
            "page": 1,
            "detection_source": "visual_structure_grid",
            "table_family": "flowchart_matrix",
            "bbox": [60.0, 100.0, 540.0, 260.0],
            "col_count": 3,
            "header": [
                {"col": 1, "text": "Genes in DNA"},
                {"col": 2, "text": "Protein"},
                {"col": 3, "text": "Characteristics"},
            ],
            "display_grid": [
                ["Genes in DNA", "Protein", "Characteristics"],
                ["Normal allele", "Normal hemoglobin", "normal health"],
                ["Sickle allele", "Sickle hemoglobin", "If sickle hemoglobin clumps"],
            ],
            "data_grid": [
                ["Normal allele", "Normal hemoglobin", "normal health"],
                ["Sickle allele", "Sickle hemoglobin", "If sickle hemoglobin clumps"],
            ],
            "semantic_grid": [
                ["Genes in DNA", "Protein", "Characteristics"],
                ["Normal allele", "Normal hemoglobin", "normal health"],
                ["Sickle allele", "Sickle hemoglobin", "If sickle hemoglobin clumps"],
            ],
            "semantic_projection_v2": {
                "version": 2,
                "table_family": "flowchart_matrix",
                "source": "table_semantic_projection_v2",
            },
        }
        tail_fragment = {
            "table_id": "tbl_tail",
            "page": 1,
            "detection_source": "structured_text_region",
            "title": "If sickle hemoglobin clumps",
            "bbox": [300.0, 235.0, 545.0, 330.0],
            "col_count": 1,
            "display_grid": [
                ["If sickle hemoglobin clumps"],
                ["in long rods"],
                ["sickle-shaped red blood cells"],
                ["clogged small blood vessels"],
            ],
            "data_grid": [
                ["If sickle hemoglobin clumps"],
                ["in long rods"],
                ["sickle-shaped red blood cells"],
                ["clogged small blood vessels"],
            ],
        }
        tables = [main_table, tail_fragment]

        merged = merge_semantic_table_fragments_v2(tables)

        self.assertEqual(merged, 1)
        self.assertEqual(len(tables), 1)
        self.assertIn("matrix_right_edge_tail_fragment", tables[0]["semantic_fragment_merge_v2"]["merge_types"])
        self.assertEqual(len(tables[0]["display_grid"]), 3)
        self.assertIn("If sickle hemoglobin clumps in long rods", tables[0]["display_grid"][2][2])
        self.assertIn("clogged small blood vessels", tables[0]["display_grid"][2][2])


if __name__ == "__main__":
    unittest.main()
