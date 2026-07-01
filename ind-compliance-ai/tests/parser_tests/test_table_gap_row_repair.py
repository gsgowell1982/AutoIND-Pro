from __future__ import annotations

import unittest

from parsers.pdf.tables import _repair_pymupdf_stacked_header_body_gap_rows


class TableGapRowRepairTests(unittest.TestCase):
    def test_recovers_aligned_data_row_between_pymupdf_header_and_body_fragments(self) -> None:
        header = {
            "table_id": "tbl_header",
            "page": 1,
            "detection_source": "pymupdf_builtin",
            "bbox": [100.0, 360.0, 510.0, 403.0],
            "display_grid": [
                ["AMS", "Average Annual Growth", None, None, None, None, "Remittance inflows"],
                [None, "2000-2004", "2004-2009", "2009-2014", "2014-2019", "2019-2020", None],
            ],
            "raw_grid": [
                ["AMS", "Average Annual Growth", None, None, None, None, "Remittance inflows"],
                [None, "2000-2004", "2004-2009", "2009-2014", "2014-2019", "2019-2020", None],
            ],
            "data_grid": [[None, "2000-2004", "2004-2009", "2009-2014", "2014-2019", "2019-2020", None]],
            "col_count": 7,
        }
        body = {
            "table_id": "tbl_body",
            "page": 1,
            "detection_source": "pymupdf_builtin",
            "bbox": [100.0, 420.0, 510.0, 536.0],
            "display_grid": [
                ["Indonesia", "9.4%", "29.5%", "4.7%", "6.4%", "-17.3%", "9,651"],
                ["Lao PDR", "4.0%", "115.7%", "38.0%", "9.5%", "-10.6%", "265"],
            ],
            "raw_grid": [
                ["Indonesia", "9.4%", "29.5%", "4.7%", "6.4%", "-17.3%", "9,651"],
                ["Lao PDR", "4.0%", "115.7%", "38.0%", "9.5%", "-10.6%", "265"],
            ],
            "grid": [
                ["Indonesia", "9.4%", "29.5%", "4.7%", "6.4%", "-17.3%", "9,651"],
                ["Lao PDR", "4.0%", "115.7%", "38.0%", "9.5%", "-10.6%", "265"],
            ],
            "data_grid": [["Lao PDR", "4.0%", "115.7%", "38.0%", "9.5%", "-10.6%", "265"]],
            "data_start_row": 1,
            "col_count": 7,
        }
        words = [
            {"text": "Cambodia", "bbox": [103.0, 406.0, 150.0, 418.0]},
            {"text": "7.5%", "bbox": [174.0, 406.0, 198.0, 418.0]},
            {"text": "-0.7%", "bbox": [225.0, 406.0, 255.0, 418.0]},
            {"text": "50.6%", "bbox": [282.0, 406.0, 315.0, 418.0]},
            {"text": "6.7%", "bbox": [335.0, 406.0, 360.0, 418.0]},
            {"text": "-16.6%", "bbox": [386.0, 406.0, 424.0, 418.0]},
            {"text": "1,272", "bbox": [478.0, 406.0, 505.0, 418.0]},
        ]

        repaired = _repair_pymupdf_stacked_header_body_gap_rows([header, body], words)

        self.assertEqual(repaired, 1)
        self.assertEqual(body["display_grid"][0], ["Cambodia", "7.5%", "-0.7%", "50.6%", "6.7%", "-16.6%", "1,272"])
        self.assertEqual(body["data_grid"][0], ["Cambodia", "7.5%", "-0.7%", "50.6%", "6.7%", "-16.6%", "1,272"])
        self.assertEqual(body["bbox"][1], 406.0)
        self.assertEqual(body["gap_row_repair"]["source"], "pymupdf_stacked_header_body_gap_row")


if __name__ == "__main__":
    unittest.main()
