from __future__ import annotations

import ast
from pathlib import Path
import unittest

from parsers.pdf.postprocess_toc import (
    _build_toc_sequence_tree,
    _first_known_page_locator,
    _first_root_outline_key,
    _first_toc_entry_outline_key,
    _last_known_page_locator,
    _last_root_outline_key,
    _last_toc_entry_outline_key,
    _next_toc_index,
    _roman_to_int,
    _toc_block_sort_key,
    _toc_entry_block_relation,
    _toc_entry_page_locator_key,
    _toc_entry_sort_key,
    _toc_entry_source_block_ids,
    _toc_horizontal_overlap_ratio,
    _toc_outline_sort_key,
)


class PostprocessTocTests(unittest.TestCase):
    def test_toc_scalar_helpers_preserve_sorting_and_key_contracts(self) -> None:
        self.assertEqual(_toc_block_sort_key({"page": 2, "bbox": [30, 40, 90, 120]}), (2, 40.0, 30.0))
        self.assertEqual(_toc_entry_sort_key({"bbox": [12, 34, 80, 50]}), (34.0, 12.0))
        self.assertEqual(_toc_entry_sort_key({"sort_y0": 7, "sort_x0": 3}), (7.0, 3.0))
        self.assertEqual(_next_toc_index([{"toc_id": "toc_002"}, {"toc_id": "toc_010"}]), 10)
        self.assertEqual(
            _toc_entry_source_block_ids(
                {
                    "source_block_ids": [" text_1 ", "text_2", "text_1", ""],
                    "source_block_id": "text_3",
                }
            ),
            ["text_1", "text_2", "text_3"],
        )
        self.assertEqual(_toc_entry_block_relation([0, 50, 100, 90], [0, 10, 100, 40]), "above")
        self.assertEqual(_toc_entry_block_relation([0, 50, 100, 90], [0, 90, 100, 110]), "below")
        self.assertEqual(_toc_entry_block_relation([0, 50, 100, 90], [0, 70, 100, 95]), "overlap")
        self.assertEqual(_toc_horizontal_overlap_ratio([0, 0, 100, 10], [50, 0, 150, 10]), 0.5)

    def test_toc_outline_and_locator_helpers_preserve_runtime_contracts(self) -> None:
        entries = [
            {"outline_index": "", "page_locator_kind": "unknown"},
            {"outline_index": "II", "page_locator_kind": "roman", "page_locator_value": 4},
            {"outline_index": "2.3", "page_locator_kind": "arabic", "page_locator_value": 12},
        ]
        toc_block = {
            "entries": [
                {"outline_index": "1.2", "level": 2},
                {"outline_index": "2", "level": 1},
                {"outline_index": "III", "level": 1},
            ]
        }

        self.assertEqual(_roman_to_int("xiv"), 14)
        self.assertIsNone(_roman_to_int("ABC"))
        self.assertEqual(_toc_outline_sort_key("2.3"), ("numeric", (2, 3)))
        self.assertEqual(_toc_outline_sort_key("IV"), ("roman", (4,)))
        self.assertEqual(_toc_outline_sort_key("B"), ("alpha", (2,)))
        self.assertEqual(_toc_outline_sort_key("APPENDIX A"), ("appendix", "APPENDIX A"))
        self.assertIsNone(_toc_outline_sort_key(""))
        self.assertEqual(_first_toc_entry_outline_key(entries), ("roman", (2,)))
        self.assertEqual(_last_toc_entry_outline_key(entries), ("numeric", (2, 3)))
        self.assertEqual(_toc_entry_page_locator_key(entries[1]), ("roman", 4))
        self.assertEqual(_first_known_page_locator(entries), ("roman", 4))
        self.assertEqual(_last_known_page_locator(entries), ("arabic", 12))
        self.assertEqual(_first_root_outline_key(toc_block), ("numeric", (2,)))
        self.assertEqual(_last_root_outline_key(toc_block), ("roman", (3,)))

    def test_build_toc_sequence_tree_links_children_and_reports_leaf_summary(self) -> None:
        root, leaf_indices, max_branching = _build_toc_sequence_tree(
            [
                {
                    "sequence_entry_index": 3,
                    "entry_index": 3,
                    "page": 2,
                    "toc_id": "toc_001",
                    "outline_index": "1.2",
                    "outline_depth": 2,
                    "text": "Subsection B",
                    "page_locator": "4",
                    "page_locator_kind": "arabic",
                    "page_locator_value": 4,
                    "level": 2,
                    "parent_sequence_entry_index": 1,
                },
                {
                    "sequence_entry_index": 1,
                    "entry_index": 1,
                    "page": 2,
                    "toc_id": "toc_001",
                    "outline_index": "1",
                    "outline_depth": 1,
                    "text": "Section",
                    "page_locator": "3",
                    "page_locator_kind": "arabic",
                    "page_locator_value": 3,
                    "level": 1,
                    "parent_sequence_entry_index": None,
                },
                {
                    "sequence_entry_index": 2,
                    "entry_index": 2,
                    "page": 2,
                    "toc_id": "toc_001",
                    "outline_index": "1.1",
                    "outline_depth": 2,
                    "text": "Subsection A",
                    "page_locator": "3",
                    "page_locator_kind": "arabic",
                    "page_locator_value": 3,
                    "level": 2,
                    "parent_sequence_entry_index": 1,
                    "section_anchor_sequence_entry_index": 10,
                },
                {
                    "sequence_entry_index": 4,
                    "entry_index": 4,
                    "page": 3,
                    "toc_id": "toc_002",
                    "outline_index": "2",
                    "outline_depth": 1,
                    "text": "Next Section",
                    "page_locator": "5",
                    "page_locator_kind": "arabic",
                    "page_locator_value": 5,
                    "level": 1,
                    "parent_sequence_entry_index": None,
                },
            ]
        )

        self.assertEqual([node["sequence_entry_index"] for node in root], [1, 4])
        self.assertEqual([child["sequence_entry_index"] for child in root[0]["children"]], [2, 3])
        self.assertEqual(root[0]["child_count"], 2)
        self.assertTrue(root[0]["has_children"])
        self.assertFalse(root[1]["has_children"])
        self.assertEqual(leaf_indices, [2, 3, 4])
        self.assertEqual(max_branching, 2)
        self.assertEqual(root[0]["children"][0]["section_anchor_sequence_entry_index"], 10)

    def test_postprocess_keeps_single_first_toc_entry_outline_key_definition(self) -> None:
        postprocess_path = Path(__file__).resolve().parents[2] / "parsers" / "pdf" / "postprocess.py"
        module = ast.parse(postprocess_path.read_text(encoding="utf-8"))

        definitions = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_first_toc_entry_outline_key"
        ]

        self.assertEqual(len(definitions), 1)


if __name__ == "__main__":
    unittest.main()
