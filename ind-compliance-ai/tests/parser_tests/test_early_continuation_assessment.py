# Version: v1.0.0
# Optimization Summary:
# - Cover early continuation assessment with current-table local context.
# - Reject local conflicting titles and narrative barriers before inheritance.
# - Preserve valid continuation when the current page repeats the parent title.
# - Cover narrow top-edge template headers so reconstructed page headers do not
#   regress continuation assessment.

from __future__ import annotations

import unittest

from parsers.pdf.table_modules.raw_objects import RawCell, RawRow, RawTableEvidence
from parsers.pdf.tables import (
    _assess_continuation_candidates_v2,
    _build_current_table_context,
    _resolve_parent_context,
)


def _make_raw_row(row_idx: int, texts: list[str], *, top_y: float) -> RawRow:
    row_y0 = top_y + row_idx * 18.0
    row_y1 = row_y0 + 14.0
    cells = [
        RawCell(
            physical_col=col_idx,
            physical_row=row_idx,
            text=text,
            bbox=(50.0 + col_idx * 100.0, row_y0, 140.0 + col_idx * 100.0, row_y1),
        )
        for col_idx, text in enumerate(texts)
    ]
    return RawRow(
        physical_row=row_idx,
        cells=cells,
        bbox=(50.0, row_y0, 550.0, row_y1),
        y0=row_y0,
        y1=row_y1,
    )


def _make_raw_table() -> RawTableEvidence:
    top_y = 80.0
    return RawTableEvidence(
        page_number=2,
        bbox=(50.0, top_y, 550.0, top_y + 140.0),
        physical_col_count=4,
        physical_row_count=3,
        rows=[
            _make_raw_row(0, ["M1", "us", "0001", "leaf.xml"], top_y=top_y),
            _make_raw_row(1, ["M1", "us", "0002", "index.xml"], top_y=top_y),
            _make_raw_row(2, ["M2", "cn", "0003", "study.pdf"], top_y=top_y),
        ],
        page_height=800.0,
        page_width=600.0,
        near_page_top=True,
        near_page_bottom=False,
    )


def _make_parent_table(title: str) -> dict:
    return {
        "table_id": "tbl_001",
        "page": 1,
        "bbox": [50.0, 600.0, 550.0, 780.0],
        "near_page_bottom": True,
        "col_count": 4,
        "row_count": 12,
        "column_signature": [0.125, 0.375, 0.625, 0.875],
        "header": [
            {"col": 1, "text": "Module"},
            {"col": 2, "text": "Region"},
            {"col": 3, "text": "Sequence"},
            {"col": 4, "text": "Filename"},
        ],
        "title": title,
        "section_hint": "",
    }


def _assess(title: str, text_blocks: list[dict[str, object]]) -> object:
    raw_evidence = _make_raw_table()
    prev_table = _make_parent_table(title)
    current_context = _build_current_table_context(
        raw_evidence=raw_evidence,
        text_blocks=text_blocks,
    )
    candidates = _resolve_parent_context(
        raw_evidence=raw_evidence,
        page_number=2,
        prev_tables=[prev_table],
        current_context=current_context,
    )
    return _assess_continuation_candidates_v2(candidates, raw_evidence)


class EarlyContinuationAssessmentTests(unittest.TestCase):
    def test_reject_when_current_table_has_conflicting_local_title(self) -> None:
        assessment = _assess(
            title="表 1. 临床试验申请的相关序列示例",
            text_blocks=[
                {
                    "text": "表 2. 新药申请的相关序列示例",
                    "bbox": [50.0, 12.0, 320.0, 28.0],
                }
            ],
        )
        self.assertFalse(assessment.is_continuation)

    def test_allow_when_current_page_repeats_parent_title(self) -> None:
        assessment = _assess(
            title="表 1. 临床试验申请的相关序列示例",
            text_blocks=[
                {
                    "text": "表 1. 临床试验申请的相关序列示例",
                    "bbox": [50.0, 12.0, 340.0, 28.0],
                }
            ],
        )
        self.assertTrue(assessment.is_continuation)
        self.assertEqual(assessment.selected_parent_id, "tbl_001")

    def test_reject_when_narrative_barrier_exists_before_table(self) -> None:
        assessment = _assess(
            title="表 1. 临床试验申请的相关序列示例",
            text_blocks=[
                {
                    "text": "以下内容说明不同申请类型的差异，下面给出一张用于展示示例数据的表格。",
                    "bbox": [50.0, 42.0, 420.0, 54.0],
                }
            ],
        )
        self.assertFalse(assessment.is_continuation)


    def test_ignore_running_header_when_structure_continues(self) -> None:
        assessment = _assess(
            title="表 1. 临床试验申请的相关序列示例",
            text_blocks=[
                {
                    "text": "eCTD技术规范 V1.0",
                    "bbox": [50.0, 42.0, 420.0, 54.0],
                }
            ],
        )
        self.assertTrue(assessment.is_continuation)

    def test_ignore_narrow_top_right_running_header_when_structure_continues(self) -> None:
        assessment = _assess(
            title="表 1. 临床试验申请的相关序列示例",
            text_blocks=[
                {
                    "text": "eCTD技术规范V1.0",
                    "bbox": [428.71, 43.87, 509.80, 52.87],
                }
            ],
        )
        self.assertTrue(assessment.is_continuation)


if __name__ == "__main__":
    unittest.main()
