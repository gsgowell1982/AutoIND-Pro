from __future__ import annotations

import ast
from pathlib import Path
import unittest

from parsers.pdf.postprocess_geometry import (
    _bbox_center_inside,
    _bbox_center_y,
    _bbox_center_y_value,
    _bbox_height,
    _bbox_overlap_ratio,
    _bbox_overlaps_any,
    _bbox_union,
    _bbox_union_loose,
    _block_vertical_gap,
    _coerce_bbox,
    _node_physical_order_key,
    _valid_block_bbox,
)


class PostprocessGeometryTests(unittest.TestCase):
    def test_postprocess_keeps_single_bbox_overlap_ratio_wrapper(self) -> None:
        postprocess_path = Path(__file__).resolve().parents[2] / "parsers" / "pdf" / "postprocess.py"
        module = ast.parse(postprocess_path.read_text(encoding="utf-8"))
        definitions = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_bbox_overlap_ratio"
        ]

        self.assertEqual(len(definitions), 1)
        body = definitions[0].body
        self.assertEqual(len(body), 1)
        self.assertIsInstance(body[0], ast.Return)
        self.assertIn("_postprocess_geometry._bbox_overlap_ratio", ast.unparse(body[0].value))

    def test_postprocess_keeps_single_bbox_center_y_wrapper(self) -> None:
        postprocess_path = Path(__file__).resolve().parents[2] / "parsers" / "pdf" / "postprocess.py"
        module = ast.parse(postprocess_path.read_text(encoding="utf-8"))
        definitions = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_bbox_center_y"
        ]

        self.assertEqual(len(definitions), 1)
        body = definitions[0].body
        self.assertEqual(len(body), 1)
        self.assertIsInstance(body[0], ast.Return)
        self.assertIn("_postprocess_geometry._bbox_center_y", ast.unparse(body[0].value))

    def test_coerce_bbox_rejects_invalid_or_degenerate_values(self) -> None:
        self.assertEqual(_coerce_bbox(["1", 2, 11.5, 9, 99]), (1.0, 2.0, 11.5, 9.0))
        self.assertIsNone(_coerce_bbox([1, 2, 1, 9]))
        self.assertIsNone(_coerce_bbox([1, 2, 3]))
        self.assertIsNone(_coerce_bbox("1,2,3,4"))

    def test_bbox_union_helpers_ignore_invalid_boxes(self) -> None:
        bboxes = [[10, 20, 30, 40], [0, 5, 12, 15], [9, 9, 8, 10], None]

        self.assertEqual(_bbox_union_loose(bboxes), (0.0, 5.0, 30.0, 40.0))
        self.assertEqual(_bbox_union([[10, 20, 30, 40], [0, 5, 12, 15], [9, 9, 8, 10]]), [0.0, 5.0, 30.0, 40.0])
        self.assertEqual(_bbox_union_loose([None, [1, 2, 1, 4]]), (0.0, 0.0, 0.0, 0.0))

    def test_bbox_center_y_uses_final_runtime_coercion_semantics(self) -> None:
        self.assertEqual(_bbox_center_y([1, 10, 5, 30]), 20.0)
        self.assertEqual(_bbox_center_y(None), 0.0)
        self.assertEqual(_bbox_center_y([1, 10, 1, 30]), 0.0)

    def test_bbox_height_returns_non_negative_height(self) -> None:
        self.assertEqual(_bbox_height([0, 10, 5, 30]), 20.0)
        self.assertEqual(_bbox_height([0, 30, 5, 10]), 0.0)
        self.assertEqual(_bbox_height([0, 30]), 0.0)

    def test_bbox_center_y_value_reads_block_bbox_tolerantly(self) -> None:
        self.assertEqual(_bbox_center_y_value({"bbox": ["0", "10", "5", "30"]}), 20.0)
        self.assertEqual(_bbox_center_y_value({"bbox": ["0", "bad", "5", "30"]}), 0.0)
        self.assertEqual(_bbox_center_y_value({"bbox": []}), 0.0)

    def test_bbox_center_inside_uses_center_and_tolerance(self) -> None:
        container = (10.0, 10.0, 20.0, 20.0)

        self.assertTrue(_bbox_center_inside([12, 12, 16, 16], container))
        self.assertFalse(_bbox_center_inside([20, 20, 30, 30], container))
        self.assertTrue(_bbox_center_inside([20, 20, 30, 30], container, tolerance=5.0))
        self.assertFalse(_bbox_center_inside([1, 2, 1, 4], container))

    def test_block_vertical_gap_uses_coerced_bboxes(self) -> None:
        self.assertEqual(
            _block_vertical_gap({"bbox": [0, 10, 20, 20]}, {"bbox": [0, 27, 20, 40]}),
            7.0,
        )
        self.assertIsNone(_block_vertical_gap({"bbox": [0, 10, 0, 20]}, {"bbox": [0, 27, 20, 40]}))

    def test_bbox_overlap_ratio_uses_first_bbox_area_as_denominator(self) -> None:
        small = (0.0, 0.0, 10.0, 10.0)
        large = (0.0, 0.0, 20.0, 20.0)

        self.assertEqual(_bbox_overlap_ratio(small, large), 1.0)
        self.assertEqual(_bbox_overlap_ratio(large, small), 0.25)

    def test_valid_block_bbox_accepts_positive_four_value_bbox(self) -> None:
        self.assertEqual(_valid_block_bbox(["1", 2, 11.5, 9]), (1.0, 2.0, 11.5, 9.0))
        self.assertIsNone(_valid_block_bbox([1, 2, 1, 9]))
        self.assertIsNone(_valid_block_bbox([1, 2, 3]))

    def test_bbox_overlaps_any_honors_threshold(self) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        occupied = [(5.0, 0.0, 15.0, 10.0)]

        self.assertTrue(_bbox_overlaps_any(bbox, occupied, threshold=0.5))
        self.assertFalse(_bbox_overlaps_any(bbox, occupied, threshold=0.51))

    def test_node_physical_order_key_uses_top_then_left(self) -> None:
        self.assertEqual(_node_physical_order_key({"bbox": [25, 10, 40, 20]}), (10.0, 25.0))
        self.assertEqual(_node_physical_order_key({"bbox": []}), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
