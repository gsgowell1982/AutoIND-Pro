from __future__ import annotations

import unittest

from parsers.pdf.postprocess import _classify_text_block_semantic_role


class TextSemanticRoleTests(unittest.TestCase):
    def test_body_copyright_discussion_remains_body_text(self) -> None:
        role, unit_role = _classify_text_block_semantic_role(
            {
                "text": "Copyright protects creative work, so people cannot generally copy it without permission.",
                "bbox": [72.0, 220.0, 420.0, 235.0],
            }
        )

        self.assertEqual(role, "text_block")
        self.assertEqual(unit_role, "body")

    def test_explicit_copyright_footer_remains_license_metadata(self) -> None:
        role, unit_role = _classify_text_block_semantic_role(
            {
                "text": "Copyright 2024 Example Publisher. All rights reserved.",
                "bbox": [72.0, 730.0, 420.0, 744.0],
            }
        )

        self.assertEqual(role, "license_notice")
        self.assertEqual(unit_role, "metadata")

    def test_mid_page_rights_line_does_not_start_license_mode_by_itself(self) -> None:
        role, unit_role = _classify_text_block_semantic_role(
            {
                "text": "& 2016 Elsevier Ltd. All rights reserved.",
                "bbox": [416.86, 506.82, 552.73, 514.07],
            }
        )

        self.assertEqual(role, "text_block")
        self.assertEqual(unit_role, "body")


if __name__ == "__main__":
    unittest.main()
