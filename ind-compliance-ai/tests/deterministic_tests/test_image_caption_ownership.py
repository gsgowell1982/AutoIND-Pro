from __future__ import annotations

import importlib
import unittest


class ImageCaptionOwnershipTests(unittest.TestCase):
    def test_consecutive_figures_keep_body_reference_cues_out_of_caption_ownership(self) -> None:
        postprocess = importlib.import_module("parsers.pdf.postprocess")

        image = {
            "block_type": "image",
            "image_id": "img_p1_002",
            "page": 1,
            "bbox": [90.0, 330.0, 520.0, 430.0],
            "caption_text": "Figure 4 Previous caption 2. Set the language attribute to empty, as shown in Figure 5:",
            "title": "Figure 4 Previous caption 2. Set the language attribute to empty, as shown in Figure 5:",
            "caption_source_block_id": "txt_previous_caption",
        }
        text_nodes = [
            {
                "block_type": "text",
                "block_id": "txt_previous_caption",
                "text": "Figure 4 Previous caption",
                "bbox": [205.0, 268.0, 390.0, 286.0],
            },
            {
                "block_type": "text",
                "block_id": "txt_body_cue",
                "text": "2. Set the language attribute to empty, as shown in Figure 5:",
                "semantic_role": "body_list_item",
                "bbox": [120.0, 300.0, 430.0, 318.0],
            },
            {
                "block_type": "text",
                "block_id": "txt_true_caption",
                "text": "Figure 5 Empty language attribute example",
                "bbox": [210.0, 448.0, 390.0, 466.0],
            },
        ]

        owned = postprocess._attach_image_caption_text_blocks(
            image,
            text_nodes,
            page_height=760.0,
            layout_profile=None,
            already_owned=set(),
        )

        self.assertEqual(image["caption_text"], "Figure 5 Empty language attribute example")
        self.assertEqual(image["title"], "Figure 5 Empty language attribute example")
        self.assertEqual(image["caption_source_block_id"], "txt_true_caption")
        self.assertEqual(owned, {"txt_true_caption"})
        self.assertEqual(image["owned_text_block_ids"], ["txt_true_caption"])
        self.assertEqual(
            [(segment["role"], segment["relation"], segment["source_block_id"]) for segment in image["caption_blocks"]],
            [("caption", "below", "txt_true_caption")],
        )
        self.assertNotIn("txt_body_cue", image["owned_text_block_ids"])
        self.assertNotIn("txt_previous_caption", image["owned_text_block_ids"])
        self.assertNotIn("Figure 4 Previous caption", image["content_text"])
        self.assertNotIn("shown in Figure 5", image["content_text"])

    def test_below_body_reference_cues_are_not_kept_as_previous_image_context(self) -> None:
        postprocess = importlib.import_module("parsers.pdf.postprocess")

        image = {
            "block_type": "image",
            "image_id": "img_p1_001",
            "page": 1,
            "bbox": [90.0, 80.0, 520.0, 190.0],
            "caption_text": "Figure 7 English language attribute example",
            "title": "Figure 7 English language attribute example",
            "content_segments": [
                {
                    "role": "nearby_context",
                    "relation": "below",
                    "text": "2. Set the language attribute to an invalid value, as shown in Figure 8:",
                    "source_block_id": "txt_next_body_cue",
                },
                {
                    "role": "caption",
                    "relation": "below",
                    "text": "Figure 7 English language attribute example",
                    "source_block_id": "txt_true_caption",
                },
            ],
            "nearby_context_blocks": [
                {
                    "block_id": "txt_next_body_cue",
                    "text": "2. Set the language attribute to an invalid value, as shown in Figure 8:",
                    "relation": "below",
                    "bbox": [120.0, 235.0, 460.0, 255.0],
                }
            ],
        }
        text_nodes = [
            {
                "block_type": "text",
                "block_id": "txt_true_caption",
                "text": "Figure 7 English language attribute example",
                "bbox": [205.0, 202.0, 390.0, 222.0],
            }
        ]

        postprocess._attach_image_caption_text_blocks(
            image,
            text_nodes,
            page_height=760.0,
            layout_profile=None,
            already_owned=set(),
        )

        self.assertEqual(image["caption_text"], "Figure 7 English language attribute example")
        self.assertNotIn("shown in Figure 8", image["content_text"])
        self.assertEqual(image["nearby_context_blocks"], [])
        self.assertEqual(
            [(segment["role"], segment.get("source_block_id")) for segment in image["content_segments"]],
            [("caption", "txt_true_caption")],
        )


if __name__ == "__main__":
    unittest.main()
