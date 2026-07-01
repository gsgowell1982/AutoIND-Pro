import unittest

from parsers.pdf import postprocess


class HeadingMetadataOwnershipTests(unittest.TestCase):
    def test_ind_metadata_heading_continuation_is_marked_as_title_metadata_not_body(self) -> None:
        heading = {
            "block_type": "text",
            "block_id": "txt_title_1",
            "page": 105,
            "bbox": [72, 100, 300, 118],
            "text": "2.6.7.9B 遗传毒性：体内",
            "semantic_role": "section_heading",
            "unit_role": "section_heading",
            "section_context": {
                "outline_index": "2.6.7.9B",
                "section_title": "遗传毒性：体内",
                "section_level": 4,
            },
        }
        continuation = {
            "block_type": "text",
            "block_id": "txt_title_2",
            "page": 105,
            "bbox": [84, 121, 520, 139],
            "text": "报告标题：MM-180801：大鼠经口给药DNA 损伤修复试验",
            "semantic_role": "",
            "unit_role": "body",
            "section_context": {
                "outline_index": "2.6.7.9B",
                "section_title": "遗传毒性：体内",
                "section_level": 4,
            },
        }
        table = {
            "block_type": "table",
            "block_id": "tbl_001",
            "table_id": "tbl_001",
            "page": 105,
            "bbox": [72, 166, 520, 266],
            "title": (
                "2.6.7.9B 遗传毒性：体内 "
                "报告标题：MM-180801：大鼠经口给药DNA 损伤修复试验"
            ),
            "composite_object": {
                "object_family": "business_table",
                "ownership_domain": "table",
                "visible_title_owner": "table",
                "metadata_title_reference_policy": "may_reference_without_visible_rendering",
            },
        }
        blocks = [heading, continuation, table]
        text_evidence = [
            {
                "evidence_id": "ce_text_txt_title_1",
                "source_type": "text",
                "source_id": "txt_title_1",
                "page": 105,
                "bbox": list(heading["bbox"]),
                "semantic_role": "section_heading",
                "content_text": heading["text"],
                "segments": [{"role": "section_heading", "text": heading["text"]}],
            },
            {
                "evidence_id": "ce_text_txt_title_2",
                "source_type": "text",
                "source_id": "txt_title_2",
                "page": 105,
                "bbox": list(continuation["bbox"]),
                "semantic_role": "body",
                "content_text": continuation["text"],
                "segments": [{"role": "body", "text": continuation["text"]}],
            },
        ]

        postprocess._annotate_heading_metadata_continuation_ownership(blocks, text_evidence)

        self.assertEqual(continuation["semantic_role"], "section_heading_continuation")
        self.assertEqual(continuation["unit_role"], "metadata")
        self.assertEqual(continuation["visible_title_owner"], "txt_title_1")
        self.assertEqual(continuation["visible_render_policy"], "metadata_only")
        self.assertEqual(continuation["metadata_title_reference_policy"], "may_reference_without_visible_rendering")
        self.assertEqual(continuation["title_metadata_owner_type"], "table")
        self.assertEqual(continuation["title_metadata_owner_id"], "tbl_001")

        continuation_evidence = text_evidence[1]
        self.assertEqual(continuation_evidence["semantic_role"], "section_heading_continuation")
        self.assertEqual(continuation_evidence["unit_role"], "metadata")
        self.assertEqual(continuation_evidence["visible_render_policy"], "metadata_only")
        self.assertEqual(continuation_evidence["title_metadata_owner_type"], "table")
        self.assertEqual(continuation_evidence["title_metadata_owner_id"], "tbl_001")
        self.assertEqual(continuation_evidence["segments"][0]["role"], "section_heading_continuation")
        self.assertEqual(continuation_evidence["segments"][0]["unit_role"], "metadata")
        self.assertEqual(continuation_evidence["segments"][0]["visible_render_policy"], "metadata_only")

    def test_ind_metadata_like_body_line_is_not_marked_without_object_title_reference(self) -> None:
        heading = {
            "block_type": "text",
            "block_id": "txt_heading",
            "page": 1,
            "bbox": [72, 100, 280, 118],
            "text": "2.6.7.9B 遗传毒性：体内",
            "semantic_role": "section_heading",
            "unit_role": "section_heading",
        }
        body_line = {
            "block_type": "text",
            "block_id": "txt_body",
            "page": 1,
            "bbox": [84, 121, 520, 139],
            "text": "报告标题：此处说明资料来源，并不是表格标题。",
            "semantic_role": "",
            "unit_role": "body",
        }
        table = {
            "block_type": "table",
            "block_id": "tbl_001",
            "table_id": "tbl_001",
            "page": 1,
            "bbox": [72, 166, 520, 266],
            "title": "Table 1 Results",
        }
        evidence = [
            {
                "evidence_id": "ce_text_txt_body",
                "source_type": "text",
                "source_id": "txt_body",
                "page": 1,
                "semantic_role": "body",
                "content_text": body_line["text"],
                "segments": [{"role": "body", "text": body_line["text"]}],
            }
        ]

        postprocess._annotate_heading_metadata_continuation_ownership([heading, body_line, table], evidence)

        self.assertEqual(body_line.get("semantic_role"), "")
        self.assertEqual(body_line.get("unit_role"), "body")
        self.assertEqual(evidence[0]["semantic_role"], "body")
        self.assertEqual(evidence[0]["segments"][0]["role"], "body")

    def test_multiple_heading_metadata_lines_are_grouped_and_exposed_as_reference_edges(self) -> None:
        heading = {
            "block_type": "text",
            "block_id": "txt_heading",
            "page": 1,
            "bbox": [72, 100, 320, 118],
            "text": "2.6.7.9B Genotoxicity: in vivo",
            "semantic_role": "section_heading",
            "unit_role": "section_heading",
        }
        report_title = {
            "block_type": "text",
            "block_id": "txt_report_title",
            "page": 1,
            "bbox": [84, 121, 520, 139],
            "text": "Report title: MM-180801 Rat oral DNA damage repair study",
            "semantic_role": "",
            "unit_role": "body",
        }
        test_article = {
            "block_type": "text",
            "block_id": "txt_test_article",
            "page": 1,
            "bbox": [84, 142, 360, 160],
            "text": "Test article: Example compound",
            "semantic_role": "",
            "unit_role": "body",
        }
        full_title = " ".join([heading["text"], report_title["text"], test_article["text"]])
        table = {
            "block_type": "table",
            "block_id": "tbl_multi",
            "table_id": "tbl_multi",
            "page": 1,
            "bbox": [72, 188, 520, 288],
            "title": full_title,
            "composite_object": {
                "object_family": "business_table",
                "ownership_domain": "table",
                "visible_title_owner": "table",
                "metadata_title_reference_policy": "may_reference_without_visible_rendering",
            },
        }
        evidence = [
            {
                "evidence_id": "ce_text_txt_report_title",
                "source_type": "text",
                "source_id": "txt_report_title",
                "page": 1,
                "semantic_role": "body",
                "content_text": report_title["text"],
                "segments": [{"role": "body", "text": report_title["text"]}],
            },
            {
                "evidence_id": "ce_text_txt_test_article",
                "source_type": "text",
                "source_id": "txt_test_article",
                "page": 1,
                "semantic_role": "body",
                "content_text": test_article["text"],
                "segments": [{"role": "body", "text": test_article["text"]}],
            },
        ]

        postprocess._annotate_heading_metadata_continuation_ownership(
            [heading, report_title, test_article, table],
            evidence,
        )

        self.assertEqual(report_title["semantic_role"], "section_heading_continuation")
        self.assertEqual(test_article["semantic_role"], "section_heading_continuation")
        self.assertEqual(report_title["visible_render_policy"], "metadata_only")
        self.assertEqual(test_article["visible_render_policy"], "metadata_only")
        self.assertEqual(report_title["title_metadata_group_index"], 1)
        self.assertEqual(test_article["title_metadata_group_index"], 2)
        self.assertEqual(report_title["title_metadata_full_title"], full_title)
        self.assertEqual(test_article["title_metadata_full_title"], full_title)

        edges = table.get("metadata_reference_edges") or []
        self.assertEqual(
            [
                (edge.get("source_block_id"), edge.get("relation"), edge.get("visible_render_policy"))
                for edge in edges
            ],
            [
                ("txt_report_title", "title_metadata_continuation", "metadata_only"),
                ("txt_test_article", "title_metadata_continuation", "metadata_only"),
            ],
        )
        self.assertEqual(table.get("title_metadata_source_block_ids"), ["txt_report_title", "txt_test_article"])
        self.assertEqual(evidence[0]["visible_render_policy"], "metadata_only")
        self.assertEqual(evidence[1]["visible_render_policy"], "metadata_only")
        self.assertEqual(evidence[0]["title_metadata_group_index"], 1)
        self.assertEqual(evidence[1]["title_metadata_group_index"], 2)

        table_evidence = postprocess._build_table_content_evidence(table)
        self.assertEqual(
            [
                (edge.get("source_block_id"), edge.get("target_object_id"), edge.get("visible_render_policy"))
                for edge in table_evidence.get("metadata_reference_edges", []) or []
            ],
            [
                ("txt_report_title", "tbl_multi", "metadata_only"),
                ("txt_test_article", "tbl_multi", "metadata_only"),
            ],
        )
        self.assertEqual(
            table_evidence.get("title_metadata_source_block_ids"),
            ["txt_report_title", "txt_test_article"],
        )

    def test_reference_edges_are_copied_to_source_object_before_content_evidence_build(self) -> None:
        heading = {
            "block_type": "text",
            "block_id": "txt_heading",
            "page": 1,
            "bbox": [72, 100, 320, 118],
            "text": "2.6.7.9B Genotoxicity: in vivo",
            "semantic_role": "section_heading",
            "unit_role": "section_heading",
        }
        metadata_line = {
            "block_type": "text",
            "block_id": "txt_report_title",
            "page": 1,
            "bbox": [84, 121, 520, 139],
            "text": "Report title: MM-180801 Rat oral DNA damage repair study",
            "semantic_role": "",
            "unit_role": "body",
        }
        full_title = f"{heading['text']} {metadata_line['text']}"
        page_table_node = {
            "block_type": "table",
            "block_id": "tbl_source",
            "table_id": "tbl_source",
            "page": 1,
            "bbox": [72, 166, 520, 266],
            "title": full_title,
        }
        source_table = {
            "table_id": "tbl_source",
            "page": 1,
            "bbox": [72, 166, 520, 266],
            "title": full_title,
            "semantic_role": "business_table",
            "row_count": 2,
            "col_count": 2,
            "row_texts": ["Treatment | Dose", "Cyclophosphamide | 7.5"],
        }
        evidence = [
            {
                "evidence_id": "ce_text_txt_report_title",
                "source_type": "text",
                "source_id": "txt_report_title",
                "page": 1,
                "semantic_role": "body",
                "content_text": metadata_line["text"],
                "segments": [{"role": "body", "text": metadata_line["text"]}],
            }
        ]

        postprocess._annotate_heading_metadata_continuation_ownership(
            [heading, metadata_line, page_table_node],
            evidence,
            {("table", "tbl_source"): source_table},
        )

        self.assertEqual(
            source_table.get("title_metadata_source_block_ids"),
            ["txt_report_title"],
        )
        table_evidence = postprocess._build_table_content_evidence(source_table)
        self.assertEqual(
            table_evidence.get("metadata_reference_edges", [])[0].get("source_block_id"),
            "txt_report_title",
        )
        self.assertEqual(
            table_evidence.get("metadata_reference_edges", [])[0].get("target_object_id"),
            "tbl_source",
        )

    def test_table_note_blocks_are_projected_as_metadata_reference_edges(self) -> None:
        table = {
            "table_id": "tbl_notes",
            "page": 1,
            "bbox": [72, 160, 520, 260],
            "title": "Table 1 Results",
            "semantic_role": "business_table",
            "row_count": 2,
            "col_count": 2,
            "row_texts": ["Treatment | Result", "Vehicle | Negative"],
            "owned_text_block_ids": ["txt_note_1"],
            "note_blocks": [
                {
                    "role": "note",
                    "source_block_id": "txt_note_1",
                    "text": "Dunnett test: *-p<0.05",
                }
            ],
        }

        table_evidence = postprocess._build_table_content_evidence(table)

        self.assertEqual(
            [
                (
                    edge.get("source_block_id"),
                    edge.get("relation"),
                    edge.get("target_object_type"),
                    edge.get("target_object_id"),
                    edge.get("visible_render_policy"),
                )
                for edge in table_evidence.get("metadata_reference_edges", []) or []
            ],
            [
                (
                    "txt_note_1",
                    "table_note",
                    "table",
                    "tbl_notes",
                    "metadata_only",
                )
            ],
        )

    def test_table_header_and_cell_note_refs_are_projected_as_metadata_reference_edges(self) -> None:
        table = {
            "table_id": "tbl_note_refs",
            "page": 1,
            "bbox": [72, 160, 520, 260],
            "title": "Table 1 Results",
            "semantic_role": "business_table",
            "row_count": 2,
            "col_count": 2,
            "row_texts": ["Dosea | Result", "Vehicle | Negativeb"],
            "note_blocks": [
                {
                    "role": "note",
                    "source_block_id": "txt_table_note_1",
                    "text": "a Header note; b Cell note.",
                }
            ],
            "header_note_refs": [
                {
                    "marker": "a",
                    "header_row": 0,
                    "header_col": 0,
                    "header_col_1based": 1,
                    "header_text": "Dosea",
                    "note_text": "Header note",
                    "note_source_block_id": "txt_table_note_1",
                }
            ],
            "cell_note_refs": [
                {
                    "marker": "b",
                    "data_row": 1,
                    "data_col": 1,
                    "data_col_1based": 2,
                    "cell_text": "Negativeb",
                    "note_text": "Cell note.",
                    "note_source_block_id": "txt_table_note_1",
                }
            ],
        }

        table_evidence = postprocess._build_table_content_evidence(table)

        self.assertEqual(
            [
                (
                    edge.get("source_block_id"),
                    edge.get("source_role"),
                    edge.get("relation"),
                    edge.get("target_object_type"),
                    edge.get("target_object_id"),
                    edge.get("visible_render_policy"),
                    edge.get("marker"),
                    edge.get("header_text"),
                    edge.get("cell_text"),
                    edge.get("note_text"),
                )
                for edge in table_evidence.get("metadata_reference_edges", []) or []
            ],
            [
                (
                    "txt_table_note_1",
                    "note",
                    "table_note",
                    "table",
                    "tbl_note_refs",
                    "metadata_only",
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    "txt_table_note_1",
                    "table_header_note_ref",
                    "table_header_note_ref",
                    "table",
                    "tbl_note_refs",
                    "metadata_only",
                    "a",
                    "Dosea",
                    None,
                    "Header note",
                ),
                (
                    "txt_table_note_1",
                    "table_cell_note_ref",
                    "table_cell_note_ref",
                    "table",
                    "tbl_note_refs",
                    "metadata_only",
                    "b",
                    None,
                    "Negativeb",
                    "Cell note.",
                ),
            ],
        )

    def test_owned_text_metadata_edge_projection_registry_covers_supported_object_types(self) -> None:
        table = {
            "table_id": "tbl_registry",
            "note_blocks": [
                {
                    "role": "note",
                    "source_block_id": "txt_table_note",
                    "text": "a Header note; b Cell note.",
                }
            ],
            "header_note_refs": [
                {
                    "marker": "a",
                    "header_text": "Dosea",
                    "note_text": "Header note",
                    "note_source_block_id": "txt_table_note",
                }
            ],
            "cell_note_refs": [
                {
                    "marker": "b",
                    "cell_text": "Negativeb",
                    "note_text": "Cell note.",
                    "note_source_block_id": "txt_table_note",
                }
            ],
        }
        image = {
            "image_id": "img_registry",
            "caption_blocks": [
                {
                    "role": "caption",
                    "source_block_id": "txt_caption",
                    "text": "Figure 1 Exposure profile",
                }
            ],
            "content_segments": [
                {
                    "role": "legend",
                    "source_block_id": "txt_legend",
                    "text": "Values are mean.",
                }
            ],
        }
        template = {
            "structure_template_id": "structure_template_registry",
            "note_blocks": [
                {
                    "role": "note",
                    "source_block_id": "txt_template_note",
                    "text": "Template note text.",
                }
            ],
            "local_note_refs": [
                {
                    "marker": "c",
                    "anchor_text": "Study number c",
                    "note_text": "c Local note text.",
                    "note_block_id": "txt_template_local_note",
                }
            ],
        }

        table_evidence = {}
        postprocess._project_owned_text_metadata_reference_edges(
            table_evidence,
            source_object=table,
            target_object_type="table",
            target_object_id="tbl_registry",
        )
        image_evidence = {}
        postprocess._project_owned_text_metadata_reference_edges(
            image_evidence,
            source_object=image,
            target_object_type="image",
            target_object_id="img_registry",
        )
        template_evidence = {}
        postprocess._project_owned_text_metadata_reference_edges(
            template_evidence,
            source_object=template,
            target_object_type="structure_template",
            target_object_id="structure_template_registry",
        )

        self.assertEqual(
            [edge.get("relation") for edge in table_evidence.get("metadata_reference_edges", []) or []],
            ["table_note", "table_header_note_ref", "table_cell_note_ref"],
        )
        self.assertEqual(
            [edge.get("relation") for edge in image_evidence.get("metadata_reference_edges", []) or []],
            ["figure_caption", "figure_legend"],
        )
        self.assertEqual(
            [edge.get("relation") for edge in template_evidence.get("metadata_reference_edges", []) or []],
            ["structure_template_note", "structure_template_local_note_ref"],
        )

    def test_image_caption_and_legend_blocks_are_projected_as_metadata_reference_edges(self) -> None:
        image = {
            "image_id": "img_notes",
            "page": 1,
            "bbox": [72, 160, 520, 360],
            "figure_ref": "Figure 1",
            "caption_text": "Figure 1 Exposure profile",
            "owned_text_block_ids": ["txt_caption_1", "txt_legend_1"],
            "caption_blocks": [
                {
                    "role": "caption",
                    "source_block_id": "txt_caption_1",
                    "text": "Figure 1 Exposure profile",
                }
            ],
            "content_segments": [
                {
                    "role": "legend",
                    "source_block_id": "txt_legend_1",
                    "text": "Values are mean.",
                }
            ],
            "content_text": "Figure 1 Exposure profile\nValues are mean.",
        }

        image_evidence = postprocess._build_image_content_evidence(image)

        self.assertEqual(
            [
                (
                    edge.get("source_block_id"),
                    edge.get("relation"),
                    edge.get("target_object_type"),
                    edge.get("target_object_id"),
                    edge.get("visible_render_policy"),
                )
                for edge in image_evidence.get("metadata_reference_edges", []) or []
            ],
            [
                (
                    "txt_caption_1",
                    "figure_caption",
                    "image",
                    "img_notes",
                    "metadata_only",
                ),
                (
                    "txt_legend_1",
                    "figure_legend",
                    "image",
                    "img_notes",
                    "metadata_only",
                ),
            ],
        )

    def test_structure_template_note_blocks_are_projected_as_metadata_reference_edges(self) -> None:
        template = {
            "structure_template_id": "structure_template_notes",
            "page": 1,
            "bbox": [72, 160, 520, 260],
            "title": "2.6.7 Study summary",
            "template_profile": "blank_study_summary_template",
            "ownership_domain": "template_form",
            "entry_count": 1,
            "entries": [
                {
                    "entry_index": 1,
                    "outline_index": "1",
                    "outline_depth": 1,
                    "title": "Study number",
                    "text": "Study number",
                }
            ],
            "owned_text_block_ids": ["txt_template_note_1"],
            "note_blocks": [
                {
                    "role": "note",
                    "source_block_id": "txt_template_note_1",
                    "note_index": 1,
                    "text": "Template note text.",
                }
            ],
        }

        template_evidence = postprocess._build_structure_template_content_evidence(template)

        self.assertEqual(
            [
                (
                    edge.get("source_block_id"),
                    edge.get("relation"),
                    edge.get("target_object_type"),
                    edge.get("target_object_id"),
                    edge.get("visible_render_policy"),
                )
                for edge in template_evidence.get("metadata_reference_edges", []) or []
            ],
            [
                (
                    "txt_template_note_1",
                    "structure_template_note",
                    "structure_template",
                    "structure_template_notes",
                    "metadata_only",
                )
            ],
        )

    def test_structure_template_local_note_refs_are_projected_as_metadata_reference_edges(self) -> None:
        template = {
            "structure_template_id": "structure_template_local_refs",
            "page": 1,
            "bbox": [72, 160, 520, 260],
            "title": "2.6.7 Study summary",
            "template_profile": "blank_study_summary_template",
            "ownership_domain": "template_form",
            "entry_count": 1,
            "entries": [
                {
                    "entry_index": 1,
                    "outline_index": "1",
                    "outline_depth": 1,
                    "title": "Study number a",
                    "text": "Study number a",
                }
            ],
            "local_note_refs": [
                {
                    "marker": "a",
                    "anchor_text": "Study number a",
                    "anchor_row_index": 1,
                    "note_text": "a Local note text.",
                    "note_block_id": "txt_template_local_note_1",
                    "relation": "local_template_note_marker",
                    "confidence": 0.86,
                }
            ],
        }

        template_evidence = postprocess._build_structure_template_content_evidence(template)

        self.assertEqual(
            [
                (
                    edge.get("source_block_id"),
                    edge.get("source_role"),
                    edge.get("relation"),
                    edge.get("target_object_type"),
                    edge.get("target_object_id"),
                    edge.get("visible_render_policy"),
                    edge.get("marker"),
                    edge.get("anchor_text"),
                    edge.get("note_text"),
                )
                for edge in template_evidence.get("metadata_reference_edges", []) or []
            ],
            [
                (
                    "txt_template_local_note_1",
                    "local_note_ref",
                    "structure_template_local_note_ref",
                    "structure_template",
                    "structure_template_local_refs",
                    "metadata_only",
                    "a",
                    "Study number a",
                    "a Local note text.",
                )
            ],
        )

    def test_ownership_graph_invariant_validator_flags_duplicate_metadata_edges(self) -> None:
        diagnostics = postprocess.validate_ownership_graph_invariants(
            [
                {
                    "block_type": "table",
                    "table_id": "tbl_duplicate_edges",
                    "metadata_reference_edges": [
                        {
                            "source_block_id": "txt_note_duplicate",
                            "relation": "table_note",
                            "target_object_type": "table",
                            "target_object_id": "tbl_duplicate_edges",
                            "visible_render_policy": "metadata_only",
                        },
                        {
                            "source_block_id": "txt_note_duplicate",
                            "relation": "table_note",
                            "target_object_type": "table",
                            "target_object_id": "tbl_duplicate_edges",
                            "visible_render_policy": "metadata_only",
                        },
                    ],
                }
            ]
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].get("invariant"), "duplicate_metadata_reference_edge")
        self.assertEqual(diagnostics[0].get("source_block_id"), "txt_note_duplicate")
        self.assertEqual(diagnostics[0].get("relation"), "table_note")
        self.assertEqual(diagnostics[0].get("target_object_id"), "tbl_duplicate_edges")

    def test_ownership_graph_invariant_validator_flags_metadata_only_visible_source_conflict(self) -> None:
        diagnostics = postprocess.validate_ownership_graph_invariants(
            [
                {
                    "block_type": "table",
                    "table_id": "tbl_metadata_only_owner",
                    "metadata_reference_edges": [
                        {
                            "source_block_id": "txt_note_visible",
                            "relation": "table_note",
                            "target_object_type": "table",
                            "target_object_id": "tbl_metadata_only_owner",
                            "visible_render_policy": "metadata_only",
                        }
                    ],
                },
                {
                    "block_type": "text",
                    "block_id": "txt_note_visible",
                    "text": "This note is already owned by the table.",
                    "semantic_role": "body",
                    "unit_role": "body",
                },
            ]
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].get("invariant"), "metadata_only_source_has_visible_rendering")
        self.assertEqual(diagnostics[0].get("source_block_id"), "txt_note_visible")
        self.assertEqual(diagnostics[0].get("relation"), "table_note")
        self.assertEqual(diagnostics[0].get("target_object_id"), "tbl_metadata_only_owner")

    def test_ownership_graph_invariant_validator_flags_missing_metadata_edge_target_owner(self) -> None:
        diagnostics = postprocess.validate_ownership_graph_invariants(
            [
                {
                    "block_type": "text",
                    "block_id": "txt_orphan_note",
                    "visible_render_policy": "metadata_only",
                },
                {
                    "block_type": "text",
                    "block_id": "txt_edge_carrier",
                    "metadata_reference_edges": [
                        {
                            "source_block_id": "txt_orphan_note",
                            "relation": "table_note",
                            "target_object_type": "table",
                            "target_object_id": "tbl_missing_owner",
                            "visible_render_policy": "metadata_only",
                        }
                    ],
                },
            ]
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].get("invariant"), "metadata_edge_target_owner_missing")
        self.assertEqual(diagnostics[0].get("source_block_id"), "txt_orphan_note")
        self.assertEqual(diagnostics[0].get("relation"), "table_note")
        self.assertEqual(diagnostics[0].get("target_object_id"), "tbl_missing_owner")

    def test_ownership_graph_invariant_validator_flags_title_metadata_edge_without_metadata_only_policy(self) -> None:
        diagnostics = postprocess.validate_ownership_graph_invariants(
            [
                {
                    "block_type": "text",
                    "block_id": "txt_report_title",
                    "semantic_role": "section_heading_continuation",
                    "visible_render_policy": "metadata_only",
                },
                {
                    "block_type": "table",
                    "table_id": "tbl_title_owner",
                    "metadata_reference_edges": [
                        {
                            "source_block_id": "txt_report_title",
                            "relation": "title_metadata_continuation",
                            "target_object_type": "table",
                            "target_object_id": "tbl_title_owner",
                        }
                    ],
                },
            ]
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].get("invariant"), "title_metadata_continuation_not_metadata_only")
        self.assertEqual(diagnostics[0].get("source_block_id"), "txt_report_title")
        self.assertEqual(diagnostics[0].get("target_object_id"), "tbl_title_owner")

    def test_ownership_graph_invariant_validator_flags_multiple_visible_title_owners(self) -> None:
        diagnostics = postprocess.validate_ownership_graph_invariants(
            [
                {
                    "block_type": "table",
                    "table_id": "tbl_title_owner_a",
                    "visible_title_owner": "txt_section_heading",
                    "visible_render_policy": "visible",
                },
                {
                    "block_type": "structure_template",
                    "structure_template_id": "tpl_title_owner_b",
                    "visible_title_owner": "txt_section_heading",
                    "visible_render_policy": "visible",
                },
            ]
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].get("invariant"), "multiple_visible_title_owners")
        self.assertEqual(diagnostics[0].get("visible_title_owner"), "txt_section_heading")
        self.assertEqual(
            diagnostics[0].get("owner_block_ids"),
            ["tbl_title_owner_a", "tpl_title_owner_b"],
        )

    def test_ownership_graph_invariant_validator_flags_absorbed_structure_template_with_visible_table_surface(self) -> None:
        diagnostics = postprocess.validate_ownership_graph_invariants(
            [
                {
                    "block_type": "structure_template",
                    "structure_template_id": "tpl_absorbed_surface",
                    "ownership_domain": "absorbed_by_business_table",
                    "semantic_role": "absorbed_structure_template_fragment",
                    "visible_render_policy": "visible",
                },
                {
                    "block_type": "table",
                    "table_id": "tbl_absorbed_surface",
                    "ownership_domain": "absorbed_by_business_table",
                    "semantic_role": "business_table",
                    "visible_render_policy": "visible",
                    "absorbed_structure_template_ids": ["tpl_absorbed_surface"],
                },
            ]
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].get("invariant"), "absorbed_structure_template_has_visible_table_surface")
        self.assertEqual(diagnostics[0].get("owner_block_id"), "tpl_absorbed_surface")
        self.assertEqual(diagnostics[0].get("related_block_id"), "tbl_absorbed_surface")


if __name__ == "__main__":
    unittest.main()
