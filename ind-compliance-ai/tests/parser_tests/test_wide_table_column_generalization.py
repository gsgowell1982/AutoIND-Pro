from __future__ import annotations

import unittest

from parsers.pdf.table_modules.raw_objects import extract_raw_evidence_from_words


def _word(x0: float, y0: float, text: str) -> tuple[float, float, float, float, str]:
    width = max(8.0, len(text) * 4.0)
    return (x0, y0, x0 + width, y0 + 8.0, text)


class WideTableColumnGeneralizationTests(unittest.TestCase):
    def test_word_clustering_accepts_stable_wide_tables_without_fixed_column_cap(self) -> None:
        words = []
        for row_idx in range(5):
            y0 = 100.0 + row_idx * 14.0
            for col_idx in range(13):
                x0 = 30.0 + col_idx * 42.0
                text = f"H{col_idx + 1}" if row_idx == 0 else f"R{row_idx}C{col_idx + 1}"
                words.append(_word(x0, y0, text))

        evidence = extract_raw_evidence_from_words(
            words,
            page_number=1,
            page_height=800.0,
            page_width=620.0,
        )

        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence.physical_col_count, 13)
        self.assertEqual(evidence.physical_row_count, 5)
        self.assertEqual(len(evidence.raw_data), 5)
        self.assertEqual(len(evidence.raw_data[0]), 13)

    def test_word_clustering_rejects_wide_noise_without_stable_row_anchor_evidence(self) -> None:
        words = []
        for row_idx in range(8):
            y0 = 100.0 + row_idx * 14.0
            for local_idx in range(2):
                col_idx = (row_idx * 2 + local_idx) % 14
                x0 = 30.0 + col_idx * 40.0
                words.append(_word(x0, y0, f"N{row_idx}_{local_idx}"))

        evidence = extract_raw_evidence_from_words(
            words,
            page_number=1,
            page_height=800.0,
            page_width=620.0,
        )

        self.assertIsNone(evidence)


if __name__ == "__main__":
    unittest.main()
