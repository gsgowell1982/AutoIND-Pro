from __future__ import annotations

import unittest

from api.main import (
    _can_merge_markdown_body_paragraph_blocks,
    _markdown_text_starts_bullet_item,
    _merge_markdown_text_fragments,
)


class MarkdownBulletContinuationTests(unittest.TestCase):
    def test_wingdings_bullet_hanging_indent_continuation_merges_as_same_item(self) -> None:
        current_block = {
            "block_type": "text",
            "text": "\uf06c 遗传毒性 - 化合物的化学结构、作用方式、与已知遗传毒性化合物之间",
            "semantic_role": "text_block",
            "bbox": [114.0, 572.0, 505.0, 586.0],
        }
        next_block = {
            "block_type": "text",
            "text": "的关系",
            "semantic_role": "text_block",
            "bbox": [135.0, 596.0, 171.0, 609.0],
        }

        current_text = str(current_block["text"])
        next_text = str(next_block["text"])

        self.assertTrue(_markdown_text_starts_bullet_item(current_text))
        self.assertFalse(_markdown_text_starts_bullet_item(next_text))
        self.assertTrue(
            _can_merge_markdown_body_paragraph_blocks(
                current_block,
                next_block,
                current_text,
                next_text,
                114.0,
            )
        )
        self.assertEqual(
            _merge_markdown_text_fragments(current_text, next_text),
            "\uf06c 遗传毒性 - 化合物的化学结构、作用方式、与已知遗传毒性化合物之间的关系",
        )


if __name__ == "__main__":
    unittest.main()
