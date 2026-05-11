# Version: v1.0.0
# Optimization Summary:
# - Cover semantic-only repair of cross-page boundary row splits.
# - Preserve raw audit rows while merging only validated continuation boundaries.
# - Reject merges when the child keeps its own anchor column or the parent row is complete.

from __future__ import annotations

import unittest

from parsers.pdf.table_modules.postprocess import (
    _refresh_row_texts_from_grid,
    repair_cross_page_boundary_row_splits,
)


def _table(
    table_id: str,
    page: int,
    grid: list[list[str | None]],
    *,
    continued_from: str | None = None,
) -> dict:
    table = {
        "table_id": table_id,
        "page": page,
        "bbox": [50.0, 40.0, 550.0, 220.0],
        "grid": [list(row) for row in grid],
        "raw_grid": [list(row) for row in grid],
        "header": [{"col": 1, "text": "Term"}, {"col": 2, "text": "Definition"}],
        "col_count": 2,
        "is_continuation": bool(continued_from),
    }
    if continued_from:
        table["continued_from"] = continued_from
    _refresh_row_texts_from_grid(table)
    return table


class CrossPageBoundaryRowMergeTests(unittest.TestCase):
    def test_merge_chain_boundary_rows_in_semantic_view_only(self) -> None:
        tbl_010 = _table(
            "tbl_010",
            20,
            [["序列号", "序列号是申请中唯一的4位数字字符串，用于区分同一申请中不同提"]],
        )
        tbl_011 = _table(
            "tbl_011",
            21,
            [
                [None, "交序列的唯一标识。"],
                ["DTD", "文档类型定义是一套为了进行程序间的"],
            ],
            continued_from="tbl_010",
        )
        tbl_012 = _table(
            "tbl_012",
            22,
            [
                [None, "数据交换而建立的规则。"],
                ["XML", "可扩展标记语言。"],
            ],
            continued_from="tbl_011",
        )

        repaired = repair_cross_page_boundary_row_splits([tbl_010, tbl_011, tbl_012])

        self.assertEqual(repaired, 2)
        self.assertEqual(
            tbl_010["row_texts"][-1],
            "序列号 | 序列号是申请中唯一的4位数字字符串，用于区分同一申请中不同提交序列的唯一标识。",
        )
        self.assertEqual(tbl_011["row_texts"][0], "DTD | 文档类型定义是一套为了进行程序间的数据交换而建立的规则。")
        self.assertEqual(tbl_011["row_count"], 1)
        self.assertEqual(tbl_011["raw_row_count"], 2)
        self.assertEqual(tbl_011["raw_row_texts"][0], "null | 交序列的唯一标识。")
        self.assertEqual(tbl_012["row_texts"][0], "XML | 可扩展标记语言。")
        self.assertEqual(tbl_012["row_count"], 1)
        self.assertEqual(tbl_012["raw_row_texts"][0], "null | 数据交换而建立的规则。")

    def test_reject_when_child_first_row_has_own_anchor_value(self) -> None:
        previous = _table(
            "tbl_010",
            20,
            [["序列号", "用于区分不同提交"]],
        )
        current = _table(
            "tbl_011",
            21,
            [["申请号", "四位数字字符串。"]],
            continued_from="tbl_010",
        )

        repaired = repair_cross_page_boundary_row_splits([previous, current])

        self.assertEqual(repaired, 0)
        self.assertEqual(previous["row_texts"][-1], "序列号 | 用于区分不同提交")
        self.assertEqual(current["row_texts"][0], "申请号 | 四位数字字符串。")

    def test_reject_when_parent_boundary_cell_already_ends_with_terminal_punctuation(self) -> None:
        previous = _table(
            "tbl_010",
            20,
            [["序列号", "这是完整定义。"]],
        )
        current = _table(
            "tbl_011",
            21,
            [[None, "下一条定义内容。"]],
            continued_from="tbl_010",
        )

        repaired = repair_cross_page_boundary_row_splits([previous, current])

        self.assertEqual(repaired, 0)
        self.assertEqual(previous["row_texts"][-1], "序列号 | 这是完整定义。")
        self.assertEqual(current["row_texts"][0], "null | 下一条定义内容。")

    def test_preserve_newline_separator_for_list_like_boundary_fragments(self) -> None:
        previous = _table(
            "tbl_003",
            13,
            [[None, "报告", "首次提交"]],
        )
        current = _table(
            "tbl_004",
            14,
            [[None, None, "回复\n撤回"]],
            continued_from="tbl_003",
        )

        repaired = repair_cross_page_boundary_row_splits([previous, current])

        self.assertEqual(repaired, 1)
        self.assertEqual(previous["row_texts"][-1], "null | 报告 | 首次提交\n回复\n撤回")
        self.assertEqual(current["row_count"], 0)
        self.assertEqual(current["raw_row_texts"][0], "null | null | 回复\n撤回")


if __name__ == "__main__":
    unittest.main()
