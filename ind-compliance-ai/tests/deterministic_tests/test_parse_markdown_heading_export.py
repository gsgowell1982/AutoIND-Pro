from __future__ import annotations

import importlib
import unittest


class ParseMarkdownHeadingExportTests(unittest.TestCase):
    def test_toc_matched_numbered_body_heading_is_not_rendered_as_ordered_list_item(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "numbered-heading.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 3, "parser_hint": "pdf"},
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq_001",
                    "root_nodes": [
                        {"outline_index": "5.0", "text": "\u53c2\u8003", "children": []},
                        {"outline_index": "6.0", "text": "\u672f\u8bed\u8868", "children": []},
                    ],
                }
            ],
            "document_ast": {
                "pages": [
                    {
                        "page": 2,
                        "blocks": [
                            {"block_type": "text", "text": "5. \u53c2\u80032"},
                            {"block_type": "text", "text": "1. ICH eCTD Specification"},
                            {"block_type": "text", "text": "2. ICH eCTD Related Files"},
                        ],
                    },
                    {
                        "page": 3,
                        "blocks": [
                            {
                                "block_type": "text",
                                "text": "6. \u672f\u8bed\u8868",
                                "section_context": {
                                    "outline_index": "8",
                                    "section_title": "\u4e0a\u4e00\u4e2a\u5217\u8868\u9879",
                                },
                            },
                            {"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"},
                        ],
                    },
                ]
            },
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "display_grid": [["\u672f\u8bed", "\u5b9a\u4e49"], ["DTD", "\u5b9a\u4e49\u6587\u672c"]],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("#### 5. \u53c2\u80032", markdown)
        self.assertIn("#### 6. \u672f\u8bed\u8868", markdown)
        self.assertIn("\n1. ICH eCTD Specification\n", markdown)
        self.assertIn("\n2. ICH eCTD Related Files\n", markdown)
        self.assertNotIn("\n6. \u672f\u8bed\u8868\n\n|", markdown)
        self.assertNotIn("#### 1. ICH eCTD Specification", markdown)

    def test_visible_urls_are_clickable_without_low_level_pdf_link_summary(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "reference-links.pdf",
            "source_type": "pdf",
            "metadata": {
                "page_count": 2,
                "parser_hint": "pdf",
                "pdf_link_action_page_records": [
                    {
                        "page": 1,
                        "link_annotation_count": 3,
                        "link_action_kinds": ["/GoTo"],
                        "link_annotation_xrefs": [10, 11, 12],
                    },
                    {
                        "page": 2,
                        "link_annotation_count": 1,
                        "link_action_kinds": ["/URI"],
                        "link_annotation_xrefs": [20],
                    },
                ],
                "pdf_bookmark_uri_targets": ["https://bookmark.example/guidance"],
            },
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {"block_type": "toc", "text": "\u76ee\u5f55\u5185\u90e8\u8df3\u8f6c"},
                            {
                                "block_type": "text",
                                "text": "\u53ef\u53c2\u89c1\uff08https://www.ich.org/ \uff09\u3001\uff08https://www.cde.org.cn/\uff09\uff0c\u4ee5\u53ca https://example.test/page.",
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("[https://www.ich.org/](https://www.ich.org/) \uff09", markdown)
        self.assertIn("[https://www.cde.org.cn/](https://www.cde.org.cn/)\uff09", markdown)
        self.assertIn("[https://example.test/page](https://example.test/page).", markdown)
        self.assertNotIn("](https://www.cde.org.cn/\uff09)", markdown)
        self.assertNotIn("### PDF \u8d85\u94fe\u63a5", markdown)
        self.assertNotIn("Page 1: 3 link annotation", markdown)
        self.assertNotIn("https://bookmark.example/guidance", markdown)

    def test_uri_annotation_over_text_block_becomes_clickable_markdown_link(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "reference-annotation-links.pdf",
            "source_type": "pdf",
            "metadata": {
                "page_count": 1,
                "parser_hint": "pdf",
                "pdf_uri_link_annotation_records": [
                    {
                        "page": 1,
                        "xref": 11,
                        "uri": "https://example.test/reference-one",
                        "bbox": [100.0, 96.0, 360.0, 116.0],
                    },
                    {
                        "page": 1,
                        "xref": 12,
                        "uri": "https://example.test/reference-two",
                        "bbox": [100.0, 146.0, 380.0, 166.0],
                    },
                    {
                        "page": 1,
                        "xref": 13,
                        "uri": "https://example.test/background",
                        "bbox": [100.0, 196.0, 380.0, 216.0],
                    },
                ],
            },
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "text": "1. Reference title",
                                "bbox": [100.0, 100.0, 260.0, 112.0],
                            },
                            {
                                "block_type": "text",
                                "text": "Plain description",
                                "bbox": [120.0, 124.0, 260.0, 136.0],
                            },
                            {
                                "block_type": "text",
                                "text": "2. Wrapped reference title",
                                "bbox": [100.0, 150.0, 290.0, 162.0],
                            },
                        ],
                    }
                ]
            },
            "table_asts": [],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("[1. Reference title](https://example.test/reference-one)", markdown)
        self.assertIn("[2. Wrapped reference title](https://example.test/reference-two)", markdown)
        self.assertIn("Plain description", markdown)
        self.assertNotIn("[Plain description]", markdown)
        self.assertNotIn("https://example.test/background", markdown)
