# Version: v1.0.0
# Optimization Summary:
# - Cover sparse continuation pages that inherit parent logical columns.
# - Verify observed cell geometry anchors continuation column placement.
# - Ensure parent bbox skeleton fallback follows the same unified mapping rule.

from __future__ import annotations

import inspect
import unittest

import parsers.pdf.table_modules.normalization as normalization_module
from parsers.pdf.table_modules.normalization import normalize_raw_evidence
from parsers.pdf.table_modules.raw_objects import RawCell, RawRow, RawTableEvidence, RawWord


def _make_sparse_row(
    row_idx: int,
    filename: str,
    description: str,
) -> RawRow:
    y0 = 20.0 + row_idx * 18.0
    y1 = y0 + 12.0
    return RawRow(
        physical_row=row_idx,
        cells=[
            RawCell(
                physical_col=0,
                physical_row=row_idx,
                text=filename,
                bbox=(112.0, y0, 188.0, y1),
            ),
            RawCell(
                physical_col=1,
                physical_row=row_idx,
                text=description,
                bbox=(212.0, y0, 292.0, y1),
            ),
        ],
        bbox=(112.0, y0, 292.0, y1),
        y0=y0,
        y1=y1,
    )


def _make_sparse_continuation_raw_table() -> RawTableEvidence:
    return RawTableEvidence(
        page_number=22,
        bbox=(0.0, 0.0, 300.0, 120.0),
        physical_col_count=2,
        physical_row_count=2,
        rows=[
            _make_sparse_row(0, "file-a.xml", "description a"),
            _make_sparse_row(1, "file-b.xml", "description b"),
        ],
        page_height=800.0,
        page_width=600.0,
        near_page_top=True,
        near_page_bottom=False,
    )


class SparseContinuationColumnMappingTests(unittest.TestCase):
    def test_production_normalization_does_not_embed_sample_text_rewrites(self) -> None:
        source = inspect.getsource(normalization_module)

        self.assertNotIn("模块一1.", source)

    def test_map_sparse_continuation_columns_from_observed_geometry(self) -> None:
        normalized = normalize_raw_evidence(
            _make_sparse_continuation_raw_table(),
            parent_col_count=3,
            parent_bbox=(0.0, 0.0, 300.0, 120.0),
            parent_column_boundaries=[0.0, 100.0, 200.0, 300.0],
        )

        self.assertEqual(normalized.column_mapping, [[], [0], [1]])
        self.assertEqual(normalized.grid[0], [None, "file-a.xml", "description a"])
        self.assertEqual(normalized.grid[1], [None, "file-b.xml", "description b"])

    def test_synthesize_parent_skeleton_when_explicit_boundaries_missing(self) -> None:
        normalized = normalize_raw_evidence(
            _make_sparse_continuation_raw_table(),
            parent_col_count=3,
            parent_bbox=(0.0, 0.0, 300.0, 120.0),
            parent_column_boundaries=None,
        )

        self.assertEqual(normalized.column_mapping, [[], [0], [1]])
        self.assertEqual(normalized.grid[0], [None, "file-a.xml", "description a"])

    def test_supplement_sparse_continuation_row_from_same_row_words_with_offset(self) -> None:
        raw_table = RawTableEvidence(
            page_number=23,
            bbox=(0.0, 0.0, 320.0, 80.0),
            physical_col_count=1,
            physical_row_count=1,
            rows=[
                RawRow(
                    physical_row=0,
                    cells=[
                        RawCell(
                            physical_col=0,
                            physical_row=0,
                            text="K-01",
                            bbox=(24.0, 20.0, 48.0, 32.0),
                        )
                    ],
                    bbox=(20.0, 18.0, 292.0, 36.0),
                    y0=18.0,
                    y1=36.0,
                )
            ],
            words=[
                RawWord("K-01", 24.0, 20.0, 48.0, 32.0),
                RawWord("payload", 224.0, 20.0, 270.0, 32.0),
                RawWord("value", 273.0, 20.0, 296.0, 32.0),
            ],
            page_height=800.0,
            page_width=600.0,
            near_page_top=True,
            near_page_bottom=False,
        )

        normalized = normalize_raw_evidence(
            raw_table,
            parent_col_count=3,
            parent_bbox=(0.0, 0.0, 300.0, 120.0),
            parent_column_boundaries=[0.0, 96.0, 198.0, 300.0],
        )

        self.assertEqual(normalized.grid[0], ["K-01", None, "payload value"])

        supplemented = [cell for cell in normalized.rows[0].cells if cell.logical_col == 2][0]
        self.assertTrue(supplemented.supplemented)
        self.assertEqual(supplemented.supplement_reason, "same_row_word_projection")

    def test_supplement_spacing_separates_cjk_label_from_dotted_outline_number(self) -> None:
        raw_table = RawTableEvidence(
            page_number=25,
            bbox=(0.0, 0.0, 320.0, 80.0),
            physical_col_count=1,
            physical_row_count=1,
            rows=[
                RawRow(
                    physical_row=0,
                    cells=[RawCell(physical_col=0, physical_row=0, text="K-01", bbox=(24.0, 20.0, 48.0, 32.0))],
                    bbox=(20.0, 18.0, 292.0, 36.0),
                    y0=18.0,
                    y1=36.0,
                )
            ],
            words=[
                RawWord("K-01", 24.0, 20.0, 48.0, 32.0),
                RawWord("\u6807\u98981.2", 224.0, 20.0, 270.0, 32.0),
                RawWord("\u5185\u5bb9", 273.0, 20.0, 296.0, 32.0),
            ],
            page_height=800.0,
            page_width=600.0,
            near_page_top=True,
            near_page_bottom=False,
        )

        normalized = normalize_raw_evidence(
            raw_table,
            parent_col_count=3,
            parent_bbox=(0.0, 0.0, 300.0, 120.0),
            parent_column_boundaries=[0.0, 96.0, 198.0, 300.0],
        )

        self.assertEqual(normalized.grid[0][2], "\u6807\u9898 1.2 \u5185\u5bb9")

    def test_does_not_supplement_when_multiple_columns_already_have_evidence(self) -> None:
        raw_table = RawTableEvidence(
            page_number=24,
            bbox=(0.0, 0.0, 320.0, 80.0),
            physical_col_count=2,
            physical_row_count=1,
            rows=[
                RawRow(
                    physical_row=0,
                    cells=[
                        RawCell(physical_col=0, physical_row=0, text="K-01", bbox=(24.0, 20.0, 48.0, 32.0)),
                        RawCell(physical_col=1, physical_row=0, text="existing", bbox=(120.0, 20.0, 166.0, 32.0)),
                    ],
                    bbox=(20.0, 18.0, 292.0, 36.0),
                    y0=18.0,
                    y1=36.0,
                )
            ],
            words=[
                RawWord("K-01", 24.0, 20.0, 48.0, 32.0),
                RawWord("existing", 120.0, 20.0, 166.0, 32.0),
                RawWord("payload", 224.0, 20.0, 270.0, 32.0),
            ],
            page_height=800.0,
            page_width=600.0,
            near_page_top=True,
            near_page_bottom=False,
        )

        normalized = normalize_raw_evidence(
            raw_table,
            parent_col_count=3,
            parent_bbox=(0.0, 0.0, 300.0, 120.0),
            parent_column_boundaries=[0.0, 96.0, 198.0, 300.0],
        )

        self.assertEqual(normalized.grid[0], ["K-01", "existing", None])


if __name__ == "__main__":
    unittest.main()
