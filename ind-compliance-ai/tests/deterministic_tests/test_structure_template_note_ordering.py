from __future__ import annotations

import unittest

from api.main import _structure_template_markdown_ordered_note_blocks


class StructureTemplateNoteOrderingTests(unittest.TestCase):
    def test_numbered_template_note_run_orders_bboxless_first_note_before_positioned_followers(self) -> None:
        template = {
            "structure_template_id": "synthetic_template_note_run",
            "row_texts": [
                "2.6.7.4 毒理学 原料药 供试品：（1）",
                "批号 纯度(%) 特定杂质() 试验编号 试验类型",
                "(2) (3)",
            ],
            "note_blocks": [
                {
                    "note_index": 1,
                    "text": "(2)应按照大体时间顺序列出毒理学试验中使用的所有批次。",
                    "bbox": [77.3, 451.58, 386.57, 463.27],
                },
                {
                    "note_index": 2,
                    "text": "(3)应注明每个批次所使用的毒理学试验。",
                    "bbox": [77.3, 467.18, 302.82, 478.87],
                },
                {
                    "note_index": 3,
                    "text": "备注：(1)国际非专利药品名称(INN)",
                    "bbox": [],
                },
            ],
        }

        ordered = _structure_template_markdown_ordered_note_blocks(template)

        self.assertEqual(
            [note.get("text") for note in ordered],
            [
                "备注：(1)国际非专利药品名称(INN)",
                "(2)应按照大体时间顺序列出毒理学试验中使用的所有批次。",
                "(3)应注明每个批次所使用的毒理学试验。",
            ],
        )

    def test_numbered_template_note_run_does_not_cross_unnumbered_continuation(self) -> None:
        template = {
            "structure_template_id": "synthetic_multiline_note_run",
            "row_texts": [],
            "note_blocks": [
                {
                    "note_index": 1,
                    "text": "注释： （1）应按照与CTD 相同的顺序来总结说明所有的生殖毒性试验",
                    "bbox": [76.0, 429.0, 773.0, 441.0],
                },
                {
                    "note_index": 2,
                    "text": "（2）国际非专利药品名称（INN）。",
                    "bbox": [112.0, 465.0, 280.0, 476.0],
                },
                {
                    "note_index": 3,
                    "text": "全性试验》规定的确定性GLP 试验除外）。但应使用更详细的模板来总结探索试验。",
                    "bbox": [125.0, 446.0, 591.0, 458.0],
                },
            ],
        }

        ordered = _structure_template_markdown_ordered_note_blocks(template)

        self.assertEqual(
            [note.get("text") for note in ordered],
            [
                "注释： （1）应按照与CTD 相同的顺序来总结说明所有的生殖毒性试验",
                "全性试验》规定的确定性GLP 试验除外）。但应使用更详细的模板来总结探索试验。",
                "（2）国际非专利药品名称（INN）。",
            ],
        )


if __name__ == "__main__":
    unittest.main()
