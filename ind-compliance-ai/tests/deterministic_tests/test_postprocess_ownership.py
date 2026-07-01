from __future__ import annotations

import unittest

from parsers.pdf.postprocess_ownership import (
    _append_owned_text_metadata_reference_edges,
    _copy_metadata_reference_edges_to_evidence,
    _mark_metadata_only_visibility,
    _metadata_edge_string_list,
    _project_owned_text_metadata_reference_edges,
    validate_ownership_graph_invariants,
)


class PostprocessOwnershipTests(unittest.TestCase):
    def test_copy_metadata_reference_edges_copies_edge_dicts_and_source_ids(self) -> None:
        source = {
            "metadata_reference_edges": [
                {
                    "source_block_id": "txt_note",
                    "relation": "table_note",
                    "target_object_type": "table",
                    "target_object_id": "tbl_001",
                }
            ],
            "title_metadata_source_block_ids": [" txt_title ", "", None, "txt_subtitle"],
        }
        evidence: dict[str, object] = {}

        _copy_metadata_reference_edges_to_evidence(evidence, source)
        source["metadata_reference_edges"][0]["relation"] = "mutated"

        self.assertEqual(evidence["metadata_reference_edges"][0]["relation"], "table_note")
        self.assertEqual(evidence["title_metadata_source_block_ids"], ["txt_title", "None", "txt_subtitle"])

    def test_project_owned_text_metadata_reference_edges_projects_table_note_refs_and_dedupes(self) -> None:
        evidence = {
            "metadata_reference_edges": [
                {
                    "source_block_id": "txt_note_a",
                    "relation": "table_note",
                    "target_object_type": "table",
                    "target_object_id": "tbl_001",
                    "visible_render_policy": "metadata_only",
                }
            ]
        }
        table = {
            "note_blocks": [
                {"role": "table_note", "source_block_id": "txt_note_a", "text": "a Existing note."},
                {"role": "table_note", "source_block_id": "txt_note_b", "text": "b New note."},
            ],
            "header_note_refs": [
                {
                    "marker": "b",
                    "note_block_id": "txt_header_note",
                    "header_text": "Dose b",
                    "note_text": "b Header note.",
                }
            ],
            "cell_note_refs": [
                {
                    "marker": "c",
                    "source_block_ids": ["txt_cell_note", "txt_cell_note"],
                    "cell_text": "10 c",
                    "note_text": "c Cell note.",
                }
            ],
        }

        _project_owned_text_metadata_reference_edges(
            evidence,
            source_object=table,
            target_object_type="table",
            target_object_id="tbl_001",
        )

        edges = evidence["metadata_reference_edges"]
        self.assertEqual(
            [(edge["source_block_id"], edge["relation"]) for edge in edges],
            [
                ("txt_note_a", "table_note"),
                ("txt_note_b", "table_note"),
                ("txt_header_note", "table_header_note_ref"),
                ("txt_cell_note", "table_cell_note_ref"),
            ],
        )
        self.assertEqual(edges[2]["source_role"], "table_header_note_ref")
        self.assertEqual(edges[2]["marker"], "b")
        self.assertEqual(edges[3]["cell_text"], "10 c")
        self.assertTrue(all(edge["visible_render_policy"] == "metadata_only" for edge in edges))

    def test_append_owned_text_metadata_reference_edges_ignores_empty_target_and_copies_extra_fields(self) -> None:
        evidence: dict[str, object] = {}

        _append_owned_text_metadata_reference_edges(
            evidence,
            target_object_type="table",
            target_object_id="",
            relation="table_note",
            segments=[{"source_block_id": "txt_note"}],
        )
        self.assertNotIn("metadata_reference_edges", evidence)

        _append_owned_text_metadata_reference_edges(
            evidence,
            target_object_type="table",
            target_object_id="tbl_001",
            relation="table_note",
            segments=[{"source_block_ids": ["txt_note_1", "txt_note_2"], "payload": {"marker": "a"}}],
            source_id_keys=("source_block_ids",),
            extra_edge_fields=("payload",),
        )
        evidence["metadata_reference_edges"][0]["payload"]["marker"] = "mutated"

        self.assertEqual(
            [edge["source_block_id"] for edge in evidence["metadata_reference_edges"]],
            ["txt_note_1", "txt_note_2"],
        )
        self.assertEqual(evidence["metadata_reference_edges"][1]["payload"]["marker"], "a")

    def test_metadata_edge_string_list_and_mark_metadata_only_visibility(self) -> None:
        self.assertEqual(_metadata_edge_string_list([" a ", "", None, "b"]), ["a", "None", "b"])
        self.assertEqual(_metadata_edge_string_list(" txt_note "), ["txt_note"])
        self.assertEqual(_metadata_edge_string_list(None), [])

        item: dict[str, object] = {}
        _mark_metadata_only_visibility(item)
        self.assertEqual(item["visible_render_policy"], "metadata_only")

    def test_validate_ownership_graph_invariants_reports_core_diagnostics(self) -> None:
        diagnostics = validate_ownership_graph_invariants(
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
                },
                {
                    "block_type": "text",
                    "block_id": "txt_note_duplicate",
                    "semantic_role": "body",
                    "unit_role": "body",
                },
                {
                    "block_type": "structure_template",
                    "structure_template_id": "tpl_absorbed",
                    "ownership_domain": "absorbed_by_business_table",
                    "semantic_role": "absorbed_structure_template_fragment",
                },
                {
                    "block_type": "table",
                    "table_id": "tbl_absorbed",
                    "visible_render_policy": "visible",
                    "absorbed_structure_template_ids": ["tpl_absorbed"],
                },
            ]
        )

        invariants = [item["invariant"] for item in diagnostics]
        self.assertIn("duplicate_metadata_reference_edge", invariants)
        self.assertIn("metadata_only_source_has_visible_rendering", invariants)
        self.assertIn("absorbed_structure_template_has_visible_table_surface", invariants)


if __name__ == "__main__":
    unittest.main()
