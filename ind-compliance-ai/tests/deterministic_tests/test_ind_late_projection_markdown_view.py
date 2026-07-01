from __future__ import annotations

import unittest

from api import main


class IndLateProjectionMarkdownViewTests(unittest.TestCase):
    def test_ind_late_projection_markdown_uses_semantic_display_grid_not_evidence_grid(self) -> None:
        block = {
            "semantic_projection_v2": {
                "dose_response_result_panel_projection": {
                    "semantic_profile": "dose_response_result_panel",
                }
            },
            "semantic_grid": [
                ["Dose", "0", "10"],
                ["Result", "-", "+"],
            ],
            "semantic_display_grid": [
                ["Dose", "0", "10"],
                ["Result", "-", "+"],
            ],
            "display_grid": [
                ["Dose", "Column 2", "0", "10"],
                ["Result", "", "-", "+"],
                ["Dunnett", "*-p<0.05", "", ""],
            ],
            "data_grid": [
                ["Result", "", "-", "+"],
                ["Dunnett", "*-p<0.05", "", ""],
            ],
        }

        grid = main._normalize_markdown_table_semantic_header_grid(block)

        self.assertEqual(
            grid,
            [
                ["Dose", "0", "10"],
                ["Result", "-", "+"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
