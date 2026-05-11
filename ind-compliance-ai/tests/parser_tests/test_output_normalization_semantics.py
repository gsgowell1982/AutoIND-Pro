# Version: v1.0.0
# Optimization Summary:
# - Guard continuation-page output normalization from rewriting local data rows
#   with inherited header metadata.
# - Ensure promoted directory filenames are not duplicated in companion
#   description cells.

from __future__ import annotations

import unittest
from types import SimpleNamespace

from parsers.pdf.table_modules.continuum.semantic_repairs import (
    repair_directory_listing_structure,
)
from parsers.pdf.table_modules.postprocess import (
    _align_header_row,
    _join_vector_ocr_group_text,
    _refresh_row_texts_from_grid,
)


def _normalized_row(values: list[str | None]) -> SimpleNamespace:
    cells = [
        SimpleNamespace(
            logical_col=col_idx,
            text=value,
            supplemented=False,
            supplement_reason=None,
        )
        for col_idx, value in enumerate(values)
    ]
    return SimpleNamespace(cells=cells)


class OutputNormalizationSemanticTests(unittest.TestCase):
    def test_vector_ocr_join_inserts_space_between_cjk_sentence_and_english_sentence(self) -> None:
        merged = _join_vector_ocr_group_text(
            "轻度情绪失调不需要治疗。",
            "It is important to distinguish postpartum depression from mild mood disorders.",
        )

        self.assertEqual(
            merged,
            "轻度情绪失调不需要治疗。 It is important to distinguish postpartum depression from mild mood disorders.",
        )

    def test_vector_ocr_join_separates_adjacent_bracketed_groups(self) -> None:
        merged = _join_vector_ocr_group_text(
            "(产后抑郁症，诊断，轻度情绪失调)",
            "(Postpartum depression, Diagnosis, Mild mood disorders)",
        )

        self.assertEqual(
            merged,
            "(产后抑郁症，诊断，轻度情绪失调) (Postpartum depression, Diagnosis, Mild mood disorders)",
        )

    def test_do_not_rewrite_continuation_data_row_with_inherited_header(self) -> None:
        table = {
            "header": [
                {"col": 1, "text": "文件夹"},
                {"col": 2, "text": "文件"},
                {"col": 3, "text": "命名规则"},
            ],
            "header_inherited": True,
            "grid": [
                [None, "00", "模块一 1.0 章节内容文件夹"],
                [None, "02", "模块一 1.2 章节内容文件夹"],
            ],
        }

        _align_header_row(table)
        _refresh_row_texts_from_grid(table)

        self.assertIsNone(table["grid"][0][0])
        self.assertEqual(table["grid"][0][1], "00")
        self.assertEqual(table["row_texts"][0], "null | 00 | 模块一 1.0 章节内容文件夹")

    def test_allow_alignment_when_continuation_page_locally_repeats_header(self) -> None:
        table = {
            "header": [
                {"col": 1, "text": "文件夹"},
                {"col": 2, "text": "文件"},
                {"col": 3, "text": "命名规则"},
            ],
            "header_inherited": True,
            "grid": [[None, "文件", "命名规则"]],
        }

        _align_header_row(table)

        self.assertEqual(table["grid"][0], ["文件夹", "文件", "命名规则"])

    def test_display_and_data_grid_project_table_cell_line_breaks_without_rewriting_raw_grid(self) -> None:
        table = {
            "raw_grid": [
                ["header style", "注册行为\n类型"],
                ["continuous phrase", "临床试验\n申请"],
                ["modifier phrase", "首次\n提交"],
                ["two item enum", "回复\n撤回"],
                ["soft wrap", "这是一个较长的说明文本\n因为 PDF 自动折行被拆成两行"],
                ["enumeration", "首次提交\n回复\n撤回"],
                ["domain enum", "格式转换\n回复\n撤回"],
                ["continuous report", "研发期间安全\n性报告"],
                ["continuous application", "新增适应症为xx 的临\n床试验申请"],
                ["two independent terms", "安全性\n有效性"],
                ["two independent phrases", "新药申请\n仿制药申请"],
                ["split phrase", "新适应症\n和联合用\n药"],
            ],
            "grid": [
                ["header style", "注册行为\n类型"],
                ["continuous phrase", "临床试验\n申请"],
                ["modifier phrase", "首次\n提交"],
                ["two item enum", "回复\n撤回"],
                ["soft wrap", "这是一个较长的说明文本\n因为 PDF 自动折行被拆成两行"],
                ["enumeration", "首次提交\n回复\n撤回"],
                ["domain enum", "格式转换\n回复\n撤回"],
                ["continuous report", "研发期间安全\n性报告"],
                ["continuous application", "新增适应症为xx 的临\n床试验申请"],
                ["two independent terms", "安全性\n有效性"],
                ["two independent phrases", "新药申请\n仿制药申请"],
                ["split phrase", "新适应症\n和联合用\n药"],
            ],
        }

        _refresh_row_texts_from_grid(table)

        self.assertEqual(table["raw_grid"][0][1], "注册行为\n类型")
        self.assertEqual(table["display_grid"][0][1], "注册行为类型")
        self.assertEqual(table["display_grid"][1][1], "临床试验申请")
        self.assertEqual(table["display_grid"][2][1], "首次提交")
        self.assertEqual(table["display_grid"][3][1], "回复 / 撤回")
        self.assertEqual(table["display_grid"][4][1], "这是一个较长的说明文本因为 PDF 自动折行被拆成两行")
        self.assertEqual(table["display_grid"][5][1], "首次提交 / 回复 / 撤回")
        self.assertEqual(table["display_grid"][6][1], "格式转换 / 回复 / 撤回")
        self.assertEqual(table["display_grid"][7][1], "研发期间安全性报告")
        self.assertEqual(table["display_grid"][8][1], "新增适应症为xx 的临床试验申请")
        self.assertEqual(table["display_grid"][9][1], "安全性 / 有效性")
        self.assertEqual(table["display_grid"][10][1], "新药申请 / 仿制药申请")
        self.assertEqual(table["display_grid"][11][1], "新适应症和联合用药")
        self.assertEqual(table["data_grid"], table["display_grid"])

    def test_trim_description_prefix_after_directory_filename_promotion(self) -> None:
        grid = [
            ["dtd\ncn-regional-1-0.xsd\nich-ectd-3-2.dtd", None, "DTD folder"],
            [None, None, "cn-regional-1-0.xsd module schema folder"],
            [None, None, "ich-ectd-3-2.dtd DTD folder"],
            [None, None, "helper folder a"],
            [None, None, "helper folder b"],
            [None, None, "helper folder c"],
        ]
        rows = [_normalized_row(row) for row in grid]
        raw_evidence = SimpleNamespace(
            physical_col_count=5,
            bbox=(0.0, 0.0, 500.0, 600.0),
            rows=[],
            spans=[],
            words=[],
        )

        repaired = repair_directory_listing_structure(
            rows=rows,
            grid=grid,
            raw_evidence=raw_evidence,
            logical_col_count=3,
        )

        self.assertGreaterEqual(repaired, 4)
        self.assertEqual(grid[1][1], "cn-regional-1-0.xsd")
        self.assertEqual(grid[1][2], "module schema folder")
        self.assertEqual(grid[2][1], "ich-ectd-3-2.dtd")
        self.assertEqual(grid[2][2], "DTD folder")
        self.assertEqual(rows[1].cells[2].supplement_reason, "directory_desc_prefix_trim")


if __name__ == "__main__":
    unittest.main()
