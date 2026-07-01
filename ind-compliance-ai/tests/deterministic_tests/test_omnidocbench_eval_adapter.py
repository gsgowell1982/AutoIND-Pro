from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.omnidocbench_eval_adapter import (
    _collect_low_scores,
    _detect_raster_table_regions,
    _evaluate_table_region_detection,
    _load_samples,
    _merge_aligned_text_matrix_regions,
    _run_table_region_evaluation,
    _write_eval_dataset,
    summarize_omnidocbench_metrics,
)


class OmniDocBenchEvalAdapterTests(unittest.TestCase):
    def test_summarizes_priority_metrics_from_official_result_payload(self) -> None:
        payload = {
            "table": {
                "all": {
                    "TEDS": {"all": 0.8},
                    "TEDS_structure_only": {"all": 0.9},
                }
            },
            "display_formula": {
                "all": {
                    "CDM": {"all": 0.7},
                    "Edit_dist": {"ALL_page_avg": 0.2},
                }
            },
            "reading_order": {
                "all": {
                    "Edit_dist": {"ALL_page_avg": 0.25},
                }
            },
            "text_block": {
                "all": {
                    "Edit_dist": {"ALL_page_avg": 0.1},
                }
            },
        }

        summary = summarize_omnidocbench_metrics(payload)

        self.assertEqual(summary["teds_mean"], 0.8)
        self.assertEqual(summary["teds_s_mean"], 0.9)
        self.assertEqual(summary["formula_cdm_mean"], 0.7)
        self.assertEqual(summary["formula_edit_accuracy"], 0.8)
        self.assertEqual(summary["read_order_accuracy"], 0.75)
        self.assertEqual(summary["core_text_mean"], 0.9)
        self.assertEqual(
            summary["priority_order"],
            ["teds_mean", "teds_s_mean", "formula_cdm_mean", "read_order_accuracy"],
        )

    def test_load_samples_can_select_table_pages_before_counting(self) -> None:
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "OmniDocBench.json"
            dataset.write_text(
                """
[
  {"page_info": {"image_path": "page0.jpg"}, "layout_dets": [{"category_type": "text_block"}]},
  {"page_info": {"image_path": "page1.jpg"}, "layout_dets": [{"category_type": "table"}]},
  {"page_info": {"image_path": "page2.jpg"}, "layout_dets": [{"category_type": "figure"}]},
  {"page_info": {"image_path": "page3.jpg"}, "layout_dets": [{"category_type": "table"}]}
]
""".strip(),
                encoding="utf-8",
            )

            samples = _load_samples(dataset, count=1, category_filter="table")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["page_info"]["image_path"], "page1.jpg")

    def test_write_eval_dataset_persists_only_selected_samples(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected_gt.json"
            selected = [{"page_info": {"image_path": "page1.jpg"}, "layout_dets": []}]

            _write_eval_dataset(path, selected)

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '[\n  {\n    "page_info": {\n      "image_path": "page1.jpg"\n    },\n    "layout_dets": []\n  }\n]',
            )

    def test_collect_low_scores_includes_table_failure_context(self) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            (result_dir / "predictions_quick_match_table_per_table_TEDS.json").write_text(
                """
{
  "page_a.jpg_[0]": {"TEDS": 0.0, "TEDS_structure_only": 0.0},
  "page_b.jpg_[0]": {"TEDS": 0.8, "TEDS_structure_only": 0.9}
}
""".strip(),
                encoding="utf-8",
            )
            (result_dir / "predictions_quick_match_table_result.json").write_text(
                """
[
  {
    "img_id": "page_a.jpg",
    "metric": {"TEDS": 0.0, "TEDS_structure_only": 0.0},
    "pred": "",
    "gt": "<table><tr><td>A</td></tr></table>",
    "gt_attribute": [{"with_span": true, "line": "fewer_line", "include_equation": true}]
  },
  {
    "img_id": "page_b.jpg",
    "metric": {"TEDS": 0.8, "TEDS_structure_only": 0.9},
    "pred": "<table><tr><td>B</td></tr></table>",
    "gt": "<table><tr><td>B</td></tr></table>",
    "gt_attribute": [{"with_span": false, "line": "full_line", "include_equation": false}]
  }
]
""".strip(),
                encoding="utf-8",
            )

            low_scores = _collect_low_scores(result_dir, {})

            self.assertEqual(low_scores["table_prediction_presence"]["total_tables"], 2)
            self.assertEqual(low_scores["table_prediction_presence"]["predicted_tables"], 1)
            self.assertEqual(low_scores["table_prediction_presence"]["missing_pred_tables"], 1)
            self.assertEqual(low_scores["table_prediction_presence"]["predicted_table_rate"], 0.5)
            detail = low_scores["lowest_table_details"][0]
            self.assertEqual(detail["sample"], "page_a.jpg")
            self.assertEqual(detail["teds"], 0.0)
            self.assertEqual(detail["teds_s"], 0.0)
            self.assertFalse(detail["has_pred_table"])
            self.assertEqual(detail["attributes"][0]["line"], "fewer_line")
            self.assertIn("A", detail["gt_preview"])

    def test_evaluate_table_region_detection_matches_polygon_with_iou(self) -> None:
        samples = [
            {
                "page_info": {"image_path": "page1.png"},
                "layout_dets": [
                    {
                        "category_type": "table",
                        "poly": [10, 10, 110, 10, 110, 60, 10, 60],
                    }
                ],
            }
        ]
        predictions = {"page1.png": [{"bbox": [12, 12, 108, 58], "source": "synthetic", "confidence": 0.9}]}

        result = _evaluate_table_region_detection(samples, predictions, iou_threshold=0.5)

        self.assertEqual(result["gt_table_count"], 1)
        self.assertEqual(result["predicted_region_count"], 1)
        self.assertEqual(result["matched_table_count"], 1)
        self.assertEqual(result["missed_table_count"], 0)
        self.assertEqual(result["false_positive_count"], 0)
        self.assertGreater(result["recall"], 0.99)
        self.assertGreater(result["precision"], 0.99)
        self.assertGreater(result["mean_best_iou"], 0.85)

    def test_detect_raster_table_regions_finds_synthetic_ruled_table(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover - optional dependency guard
            self.skipTest("PIL unavailable")

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "ruled_table.png"
            image = Image.new("RGB", (500, 400), "white")
            draw = ImageDraw.Draw(image)
            for x in [80, 180, 280, 420]:
                draw.line((x, 90, x, 230), fill="black", width=2)
            for y in [90, 125, 160, 195, 230]:
                draw.line((80, y, 420, y), fill="black", width=2)
            image.save(image_path)

            regions = _detect_raster_table_regions(image_path)

            self.assertTrue(regions)
            best = max(regions, key=lambda item: item["confidence"])
            self.assertEqual(best["source"], "raster_line_grid")
            self.assertEqual(best["source_type"], "raster_image")
            self.assertTrue(best["observe_only"])
            self.assertEqual(best["signals"]["source_types"], ["visual_line_grid"])
            self.assertLess(abs(best["bbox"][0] - 80), 15)
            self.assertLess(abs(best["bbox"][1] - 90), 15)
            self.assertLess(abs(best["bbox"][2] - 420), 15)
            self.assertLess(abs(best["bbox"][3] - 230), 15)

    def test_detect_raster_table_regions_finds_synthetic_text_matrix_table(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover - optional dependency guard
            self.skipTest("PIL unavailable")

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "text_matrix_table.png"
            image = Image.new("RGB", (600, 420), "white")
            draw = ImageDraw.Draw(image)
            rows = [
                ["Dose", "Route", "AUC", "Cmax"],
                ["2", "po", "14.5", "2.1"],
                ["10", "po", "71.0", "9.8"],
                ["30", "iv", "150.2", "20.4"],
            ]
            xs = [90, 210, 330, 450]
            y = 110
            for row in rows:
                for x, text in zip(xs, row):
                    draw.text((x, y), text, fill="black")
                y += 42
            image.save(image_path)

            regions = _detect_raster_table_regions(image_path)

            self.assertTrue(regions)
            best = max(regions, key=lambda item: item["confidence"])
            self.assertIn(best["source"], {"raster_text_matrix", "raster_fused_table_region"})
            self.assertIn("visual_text_matrix", best["signals"]["source_types"])
            self.assertLess(abs(best["bbox"][0] - 90), 40)
            self.assertLess(abs(best["bbox"][1] - 110), 40)
            self.assertLess(abs(best["bbox"][2] - 500), 60)
            self.assertLess(abs(best["bbox"][3] - 240), 60)

    def test_detect_raster_table_regions_does_not_promote_wrapped_body_text_matrix(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover - optional dependency guard
            self.skipTest("PIL unavailable")

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "wrapped_body_text.png"
            image = Image.new("RGB", (700, 500), "white")
            draw = ImageDraw.Draw(image)
            lines = [
                "This paragraph contains several words with natural gaps between them,",
                "but the words do not form stable aligned columns across visual rows.",
                "It should stay body text evidence rather than a raster table region.",
                "Additional prose continues with irregular phrase lengths and spacing.",
            ]
            y = 120
            for line in lines:
                draw.text((80, y), line, fill="black")
                y += 38
            image.save(image_path)

            regions = _detect_raster_table_regions(image_path)

            self.assertEqual(regions, [])

    def test_detect_raster_table_regions_rejects_page_scale_line_artifact(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover - optional dependency guard
            self.skipTest("PIL unavailable")

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page_scale_artifact.png"
            image = Image.new("RGB", (500, 700), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 60, 460, 650), outline="black", width=2)
            draw.line((40, 350, 460, 350), fill="black", width=2)
            draw.line((250, 60, 250, 650), fill="black", width=2)
            image.save(image_path)

            regions = _detect_raster_table_regions(image_path)

            self.assertEqual(regions, [])

    def test_merge_aligned_text_matrix_regions_combines_split_large_table_sections(self) -> None:
        regions = [
            {
                "bbox": [70.0, 500.0, 500.0, 620.0],
                "source": "raster_text_matrix",
                "confidence": 0.72,
                "signals": {"visual_row_count": 3},
            },
            {
                "bbox": [72.0, 660.0, 505.0, 790.0],
                "source": "raster_text_matrix",
                "confidence": 0.74,
                "signals": {"visual_row_count": 4},
            },
            {
                "bbox": [300.0, 900.0, 560.0, 980.0],
                "source": "raster_text_matrix",
                "confidence": 0.70,
                "signals": {"visual_row_count": 2},
            },
        ]

        merged = _merge_aligned_text_matrix_regions(regions, page_height=1200)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["bbox"], [70.0, 500.0, 505.0, 790.0])
        self.assertEqual(merged[0]["source"], "raster_text_matrix")
        self.assertTrue(merged[0]["observe_only"])
        self.assertEqual(merged[0]["signals"]["merged_region_count"], 2)
        self.assertEqual(merged[0]["signals"]["visual_row_count"], 7)

    def test_run_table_region_evaluation_writes_summary_and_report(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover - optional dependency guard
            self.skipTest("PIL unavailable")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            image_path = image_dir / "page1.png"
            image = Image.new("RGB", (500, 400), "white")
            draw = ImageDraw.Draw(image)
            for x in [80, 180, 280, 420]:
                draw.line((x, 90, x, 230), fill="black", width=2)
            for y in [90, 125, 160, 195, 230]:
                draw.line((80, y, 420, y), fill="black", width=2)
            image.save(image_path)
            dataset = root / "dataset.json"
            dataset.write_text(
                """
[
  {
    "page_info": {"image_path": "page1.png"},
    "layout_dets": [
      {"category_type": "table", "poly": [80, 90, 420, 90, 420, 230, 80, 230]}
    ]
  }
]
""".strip(),
                encoding="utf-8",
            )

            report_path = _run_table_region_evaluation(
                dataset_json=dataset,
                image_dir=image_dir,
                output_dir=root / "out",
                count=1,
                category_filter="table",
            )

            summary = json.loads((report_path.parent / "autoind_omnidocbench_table_region_summary.json").read_text())
            self.assertEqual(summary["metrics"]["gt_table_count"], 1)
            self.assertEqual(summary["metrics"]["matched_table_count"], 1)
            self.assertGreater(summary["metrics"]["mean_best_iou"], 0.85)
            self.assertIn("Raster Table Region", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
