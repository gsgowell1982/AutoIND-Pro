from __future__ import annotations

from types import SimpleNamespace
import unittest

from parsers.pdf.postprocess_bilingual import (
    _apply_document_bilingual_ocr_term_corrections,
    _build_document_bilingual_ocr_term_corrections,
    _cjk_edit_distance,
    _iter_bilingual_cjk_english_pairs,
    _iter_state_bilingual_source_texts,
    _normalize_bilingual_english_gloss,
)


class PostprocessBilingualTests(unittest.TestCase):
    def test_iter_bilingual_cjk_english_pairs_expands_suffixes_and_gloss_prefixes(self) -> None:
        pairs = _iter_bilingual_cjk_english_pairs("供试品 test article")

        self.assertIn(("试品", "test"), pairs)
        self.assertIn(("试品", "test article"), pairs)
        self.assertIn(("供试品", "test"), pairs)
        self.assertIn(("供试品", "test article"), pairs)

    def test_normalize_english_gloss_and_cjk_edit_distance_preserve_current_semantics(self) -> None:
        self.assertEqual(_normalize_bilingual_english_gloss(" Test-Article 2024 "), "test-article")
        self.assertEqual(_cjk_edit_distance("供试品", "供试品"), 0)
        self.assertEqual(_cjk_edit_distance("拱试品", "供试品"), 1)
        self.assertEqual(_cjk_edit_distance("", "供试品"), 3)

    def test_iter_state_bilingual_source_texts_collects_text_blocks_tables_headers_and_grid(self) -> None:
        state = SimpleNamespace(
            page_payloads=[
                {
                    "text_blocks": [{"text": "供试品 test article"}],
                    "tables": [
                        {
                            "title": "药代动力学 pharmacokinetics",
                            "header": [{"text": "样品 sample"}],
                            "data_grid": [["动物 species", "剂量 dose"]],
                        }
                    ],
                }
            ]
        )

        texts = _iter_state_bilingual_source_texts(state)

        self.assertEqual(
            texts,
            [
                "供试品 test article",
                "药代动力学 pharmacokinetics",
                "样品 sample",
                "动物 species",
                "剂量 dose",
            ],
        )

    def test_build_document_bilingual_ocr_term_corrections_uses_repeated_canonical_gloss(self) -> None:
        state = SimpleNamespace(
            page_payloads=[
                {
                    "text_blocks": [
                        {"text": "供试品 test article"},
                        {"text": "供试品 test article"},
                        {"text": "拱试品 test article"},
                    ],
                    "tables": [],
                }
            ]
        )

        corrections = _build_document_bilingual_ocr_term_corrections(state)

        self.assertEqual(corrections, {"test article": {"拱试品": "供试品"}})

    def test_apply_document_bilingual_ocr_term_corrections_only_changes_ocr_repair_blocks(self) -> None:
        text_blocks = [
            {"source": "body-ocr-repair", "text": "拱试品 test article"},
            {"source": "text-layer", "text": "拱试品 test article"},
        ]

        _apply_document_bilingual_ocr_term_corrections(
            text_blocks,
            {"test article": {"拱试品": "供试品"}},
        )

        self.assertEqual(text_blocks[0]["text"], "供试品 test article")
        self.assertEqual(
            text_blocks[0]["ocr_term_corrections"],
            [{"from": "拱试品", "to": "供试品", "english_gloss": "test article"}],
        )
        self.assertEqual(text_blocks[1]["text"], "拱试品 test article")
        self.assertNotIn("ocr_term_corrections", text_blocks[1])


if __name__ == "__main__":
    unittest.main()
