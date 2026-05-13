from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest


class ParseMarkdownExportTests(unittest.TestCase):
    def test_full_markdown_exports_complete_recursive_toc_tree(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "eCTD实施指南.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 42, "parser_hint": "pdf"},
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq-001",
                    "title": "目 录",
                    "pages": [2, 3, 4],
                    "page_span": [2, 4],
                    "entry_count": 24,
                    "root_nodes": [
                        {
                            "outline_index": "2.0",
                            "text": "基本要求",
                            "page": 2,
                            "page_locator_value": 6,
                            "children": [
                                {
                                    "outline_index": f"2.{index}",
                                    "text": f"第 2.{index} 节",
                                    "page": 2,
                                    "page_locator_value": 6 + index,
                                    "children": [],
                                }
                                for index in range(1, 9)
                            ],
                        },
                        {
                            "outline_index": "3.0",
                            "text": "eCTD 申报资料中的编号管理",
                            "page": 3,
                            "page_locator_value": 11,
                            "children": [
                                {
                                    "outline_index": "3.1",
                                    "text": "原始编号的应用",
                                    "page": 3,
                                    "page_locator_value": 11,
                                    "children": [],
                                }
                            ],
                        },
                        {
                            "outline_index": "4.0",
                            "text": "文件组织结构",
                            "page": 3,
                            "page_locator_value": 13,
                            "children": [
                                {
                                    "outline_index": "4.1",
                                    "text": "模块一：行政文件和药品信息",
                                    "page": 3,
                                    "page_locator_value": 13,
                                    "children": [
                                        {
                                            "outline_index": "4.1.3",
                                            "text": "信封信息的准备",
                                            "page": 3,
                                            "page_locator_value": 14,
                                            "children": [],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "outline_index": "5.0",
                            "text": "特定类型提交的建议",
                            "page": 4,
                            "page_locator_value": 17,
                            "children": [],
                        },
                        {
                            "outline_index": "7.0",
                            "text": "对eCTD 申报资料文件的要求",
                            "page": 4,
                            "page_locator_value": 29,
                            "children": [
                                {
                                    "outline_index": "7.4",
                                    "text": "书签与超文本链接的要求",
                                    "page": 4,
                                    "page_locator_value": 31,
                                    "children": [],
                                }
                            ],
                        },
                    ],
                }
            ],
            "pages": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("### 解析目录结构", markdown)
        for expected in (
            "2.1 第 2.1 节",
            "2.8 第 2.8 节",
            "3.0 eCTD 申报资料中的编号管理",
            "4.1.3 信封信息的准备",
            "5.0 特定类型提交的建议",
            "7.0 对eCTD 申报资料文件的要求",
            "7.4 书签与超文本链接的要求",
        ):
            self.assertIn(expected, markdown)
        self.assertIn("目录页: 2-4", markdown)
        self.assertIn("定位页码: 29", markdown)

    def test_ui_markdown_surfaces_toc_summary_for_single_pdf(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "eCTD实施指南.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 42, "parser_hint": "pdf"},
            "toc_sequences": [
                {
                    "toc_sequence_id": "tocseq-001",
                    "title": "目 录",
                    "pages": [2, 3, 4],
                    "entry_count": 47,
                    "root_nodes": [
                        {"outline_index": "3.0", "text": "eCTD 申报资料中的编号管理", "children": []},
                        {"outline_index": "5.0", "text": "特定类型提交的建议", "children": []},
                        {"outline_index": "7.0", "text": "对eCTD 申报资料文件的要求", "children": []},
                    ],
                }
            ],
            "pages": [],
            "text": "",
        }

        markdown = api_main._build_ui_markdown([document], [], [])

        self.assertIn("## 解析目录结构", markdown)
        self.assertIn("eCTD实施指南.pdf", markdown)
        self.assertIn("目录项 47 条", markdown)
        self.assertIn("3.0 eCTD 申报资料中的编号管理", markdown)
        self.assertIn("5.0 特定类型提交的建议", markdown)
        self.assertIn("7.0 对eCTD 申报资料文件的要求", markdown)
        self.assertNotIn("请下载完整解析 Markdown 查看", markdown)

    def test_full_markdown_renders_body_blocks_without_page_or_debug_metadata(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "sample.pdf",
            "source_type": "pdf",
            "metadata": {
                "page_count": 2,
                "parser_hint": "pdf",
                "pdf_link_action_page_records": [
                    {
                        "page": 1,
                        "link_annotation_count": 2,
                        "link_action_kinds": ["/URI", "/GoTo"],
                        "link_annotation_xrefs": [31, 32],
                    }
                ],
                "pdf_bookmark_uri_targets": ["https://example.test/guidance"],
            },
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_001",
                                "text": "Before image.",
                                "bbox": [72, 80, 200, 92],
                            },
                            {
                                "block_type": "image",
                                "block_id": "img_p1_001",
                                "image_id": "img_p1_001",
                                "title": "Figure 1: Main folder",
                                "caption_text": "Figure 1: Main folder",
                                "bbox": [100, 120, 300, 260],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_002",
                                "text": "Before table.",
                                "bbox": [72, 300, 200, 312],
                            },
                            {
                                "block_type": "table",
                                "block_id": "tbl_001",
                                "table_id": "tbl_001",
                                "title": "Table 1: Folder mapping",
                                "bbox": [72, 330, 500, 420],
                                "display_grid": [
                                    ["Folder", "Document type"],
                                    ["admin", "Cover letter"],
                                    ["clinical", "Protocol"],
                                ],
                            },
                            {
                                "block_type": "text",
                                "block_id": "txt_p1_003",
                                "text": "After table. See https://example.test/page.",
                                "bbox": [72, 450, 200, 462],
                            },
                        ],
                    }
                ]
            },
            "pages": [{"page_number": 1, "block_count": 5}],
            "table_asts": [],
            "image_blocks": [],
            "text": "fallback text should not be needed",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("### 正文结构化内容", markdown)
        self.assertNotIn("### PDF Pages", markdown)
        self.assertNotIn("#### Page 1", markdown)
        self.assertIn("Before image.", markdown)
        self.assertIn("![Figure 1: Main folder](#img_p1_001)", markdown)
        self.assertNotIn("_图片占位", markdown)
        self.assertNotIn("image_id:", markdown)
        self.assertNotIn("bbox:", markdown)
        self.assertNotIn("**Table 1: Folder mapping**", markdown)
        self.assertIn("| Folder | Document type |", markdown)
        self.assertIn("| admin | Cover letter |", markdown)
        self.assertIn("After table. See [https://example.test/page](https://example.test/page).", markdown)
        self.assertNotIn("### PDF 超链接", markdown)
        self.assertNotIn("Page 1: 2 link annotation(s); actions: /GoTo, /URI; xrefs: 31, 32", markdown)
        self.assertNotIn("https://example.test/guidance", markdown)

        self.assertLess(markdown.index("Before image."), markdown.index("![Figure 1: Main folder](#img_p1_001)"))
        self.assertLess(markdown.index("![Figure 1: Main folder](#img_p1_001)"), markdown.index("Before table."))
        self.assertLess(markdown.index("Before table."), markdown.index("| Folder | Document type |"))
        self.assertLess(markdown.index("| admin | Cover letter |"), markdown.index("After table."))

    def test_full_markdown_embeds_pdf_image_crop_when_source_file_is_available(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
            fitz = importlib.import_module("fitz")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"required module unavailable in this environment: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "image-source.pdf"
            pdf = fitz.open()
            page = pdf.new_page(width=120, height=120)
            page.draw_rect(fitz.Rect(20, 20, 80, 80), color=(1, 0, 0), fill=(1, 0, 0))
            pdf.save(pdf_path)
            pdf.close()

            document = {
                "filename": "image-source.pdf",
                "source_type": "pdf",
                "source_path": str(pdf_path),
                "metadata": {"page_count": 1, "parser_hint": "pdf"},
                "document_ast": {
                    "pages": [
                        {
                            "page": 1,
                            "blocks": [
                                {
                                    "block_type": "image",
                                    "block_id": "img_p1_001",
                                    "image_id": "img_p1_001",
                                    "caption_text": "Figure 1: red square",
                                    "bbox": [20, 20, 80, 80],
                                }
                            ],
                        }
                    ]
                },
                "pages": [{"page_number": 1, "block_count": 1}],
                "image_blocks": [
                    {
                        "image_id": "img_p1_001",
                        "page": 1,
                        "caption_text": "Figure 1: red square",
                        "bbox": [20, 20, 80, 80],
                    }
                ],
                "table_asts": [],
                "text": "",
            }

            markdown = api_main._build_full_markdown([document])

            self.assertIn("![Figure 1: red square](data:image/png;base64,", markdown)
            self.assertNotIn("_图片占位", markdown)
            self.assertNotIn("image_id:", markdown)
            self.assertNotIn("bbox:", markdown)

    def test_full_markdown_merges_continued_tables_and_uses_display_grid(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "continued-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 2, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {"block_type": "text", "block_id": "txt_001", "text": "Before table."},
                            {"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"},
                        ],
                    },
                    {
                        "page": 2,
                        "blocks": [
                            {"block_type": "table", "block_id": "tbl_002", "table_id": "tbl_002"},
                            {"block_type": "text", "block_id": "txt_002", "text": "After table."},
                        ],
                    },
                ]
            },
            "pages": [{"page_number": 1, "block_count": 2}, {"page_number": 2, "block_count": 2}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "title": "Table 1: Continued",
                    "display_grid": [
                        ["Display A", "Display B"],
                        ["row\n1", "visible\nvalue"],
                    ],
                    "raw_grid": [
                        ["Raw A", "Raw B"],
                        ["raw row", "raw value"],
                    ],
                    "continued_to": ["tbl_002"],
                },
                {
                    "table_id": "tbl_002",
                    "title": "Table 1: Continued",
                    "display_grid": [
                        ["row 2", "continued value"],
                    ],
                    "raw_grid": [
                        ["raw row 2", "raw value 2"],
                    ],
                    "continued_from": ["tbl_001"],
                },
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("Before table.", markdown)
        self.assertIn("After table.", markdown)
        self.assertIn("| Display A | Display B |", markdown)
        self.assertIn("| row 1 | visible value |", markdown)
        self.assertIn("| row 2 | continued value |", markdown)
        self.assertNotIn("<br>", markdown)
        self.assertNotIn("Table 1: Continued", markdown)
        self.assertNotIn("Raw A", markdown)
        self.assertNotIn("raw row", markdown)
        self.assertEqual(markdown.count("| Display A | Display B |"), 1)
        self.assertNotIn("#### Page 1", markdown)
        self.assertNotIn("#### Page 2", markdown)

    def test_full_markdown_uses_semantic_header_when_sparse_header_continuation_rows_exist(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "roadmap-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "header": [
                        {"text": "IND Submission", "col": 1},
                        {"text": "Submission Date", "col": 2},
                        {"text": "Submission Content", "col": 3},
                        {"text": "CD-ROM", "col": 4},
                        {"text": "Hypertext link Destination", "col": 5},
                    ],
                    "display_grid": [
                        ["IND Submission", "Submission Date", "Submission", "CD-ROM", "Hypertext link"],
                        [None, None, "Content", None, "Destination"],
                        ["IND 12345.0003", "04-Jul-2001", "Cover letter", "3.01", "amendtoc.pdf"],
                    ],
                    "data_grid": [
                        ["IND 12345.0003", "04-Jul-2001", "Cover letter", "3.01", "amendtoc.pdf"],
                        ["IND 12345.0003", "04-Jul-2001", "1571", "3.01", None],
                    ],
                    "data_start_row": 2,
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn(
            "| IND Submission | Submission Date | Submission Content | CD-ROM | Hypertext link Destination |",
            markdown,
        )
        self.assertIn("| IND 12345.0003 | 04-Jul-2001 | Cover letter | 3.01 | amendtoc.pdf |", markdown)
        self.assertIn("| IND 12345.0003 | 04-Jul-2001 | 1571 | 3.01 |  |", markdown)
        self.assertNotIn("| IND Submission | Submission Date | Submission | CD-ROM | Hypertext link |", markdown)
        self.assertNotIn("|  |  | Content |  | Destination |", markdown)

    def test_full_markdown_preserves_internal_table_title_when_using_semantic_header(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "test-ind.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "title": "Main IND Table of Contents",
                    "title_row_index": 0,
                    "header_row_index": 1,
                    "data_start_row": 2,
                    "header": [
                        {"text": "Section", "col": 1},
                        {"text": "Description", "col": 2},
                        {"text": "Electronic folder/filename", "col": 3},
                    ],
                    "display_grid": [
                        ["Main IND Table of Contents", None, None],
                        ["Section", "Description", "Electronic folder/filename"],
                        ["-", "Coverletter", "0000_coverletter.pdf"],
                    ],
                    "data_grid": [
                        ["-", "Coverletter", "0000_coverletter.pdf"],
                    ],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("**Main IND Table of Contents**", markdown)
        self.assertIn("| Section | Description | Electronic folder/filename |", markdown)
        self.assertIn("| - | Coverletter | 0000_coverletter.pdf |", markdown)
        self.assertNotIn("| Main IND Table of Contents |  |  |", markdown)

    def test_full_markdown_uses_display_rows_with_semantic_header_to_preserve_visual_empty_cells(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "ectd-technical-spec.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "header": [
                        {"text": "文件夹", "col": 1},
                        {"text": "文件", "col": 2},
                        {"text": "命名规则", "col": 3},
                    ],
                    "display_grid": [
                        ["文件夹", "文件", "命名规则"],
                        ["0000", None, "4 位数字组成的序列文件夹"],
                        [None, "index.xml", "符合 ICH 要求的骨架文件"],
                        [None, "index-md5.txt", "符合 ICH 要求的 MD5 校验和文件"],
                    ],
                    "data_grid": [
                        ["0000", None, "4 位数字组成的序列文件夹"],
                        ["0000", "index.xml", "符合 ICH 要求的骨架文件"],
                        ["0000", "index-md5.txt", "符合 ICH 要求的 MD5 校验和文件"],
                    ],
                    "semantic_compaction": {
                        "applied": True,
                        "strategy": "leading_key_carry_forward",
                    },
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("| 文件夹 | 文件 | 命名规则 |", markdown)
        self.assertIn("| 0000 |  | 4 位数字组成的序列文件夹 |", markdown)
        self.assertIn("|  | index.xml | 符合 ICH 要求的骨架文件 |", markdown)
        self.assertIn("|  | index-md5.txt | 符合 ICH 要求的 MD5 校验和文件 |", markdown)
        self.assertNotIn("| 0000 | index.xml | 符合 ICH 要求的骨架文件 |", markdown)

    def test_full_markdown_renders_merged_section_group_rows_as_separators(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "merged-section-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "display_grid": [
                        ["序号", "描述", "说明", "严重程度"],
                        ["4.1 - 基础信息", None, None, None],
                        ["4.1.1", "模块一的区域骨架文件必须存在", "m1/cn", "错误"],
                    ],
                    "merged_rows": [
                        {
                            "row": 2,
                            "kind": "section_group",
                            "text": "4.1 - 基础信息",
                            "colspan": 4,
                            "source": "single_leading_section_group_cell",
                        }
                    ],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("**4.1 - 基础信息**", markdown)
        self.assertIn("| 序号 | 描述 | 说明 | 严重程度 |", markdown)
        self.assertIn("| 4.1.1 | 模块一的区域骨架文件必须存在 | m1/cn | 错误 |", markdown)
        self.assertNotIn("| 4.1 - 基础信息 |  |  |  |", markdown)
    def test_full_markdown_renders_merged_table_note_title_rows_as_separators(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "table-note-title.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "display_grid": [
                        ["说明:", None, None],
                        ["错误", "必须遵守的关键验证标准", "任何错误信息均会导致申报资料被拒收。"],
                        ["警告", "建议遵守的验证标准", "警告信息可以在说明函中进行解释。"],
                    ],
                    "merged_rows": [
                        {
                            "row": 1,
                            "kind": "table_note_title",
                            "text": "说明:",
                            "colspan": 3,
                            "source": "single_leading_title_cell_with_following_tabular_rows",
                        }
                    ],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("**说明:**", markdown)
        self.assertIn("| 错误 | 必须遵守的关键验证标准 | 任何错误信息均会导致申报资料被拒收。 |", markdown)
        self.assertNotIn("| 说明: |  |  |", markdown)

    def test_full_markdown_merges_continuation_boundary_fragment_into_previous_row(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "continued-fragment-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 2, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
                    {"page": 2, "blocks": [{"block_type": "table", "block_id": "tbl_002", "table_id": "tbl_002"}]},
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}, {"page_number": 2, "block_count": 1}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "display_grid": [
                        ["申请类型", "注册行为类型", "序列类型"],
                        ["新药申请", "报告", "首次提交"],
                    ],
                    "continued_to": ["tbl_002"],
                },
                {
                    "table_id": "tbl_002",
                    "display_grid": [
                        [None, None, "回复\n撤回"],
                        [None, "再注册", "首次提交\n回复\n撤回"],
                    ],
                    "continued_from": ["tbl_001"],
                },
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("| 申请类型 | 注册行为类型 | 序列类型 |", markdown)
        self.assertIn("| 新药申请 | 报告 | 首次提交 / 回复 / 撤回 |", markdown)
        self.assertIn("|  | 再注册 | 首次提交 / 回复 / 撤回 |", markdown)
        self.assertNotIn("|  |  | 回复", markdown)
        self.assertEqual(markdown.count("| 申请类型 | 注册行为类型 | 序列类型 |"), 1)

    def test_full_markdown_formats_table_cell_line_breaks_by_content_shape(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        document = {
            "filename": "line-break-table.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {
                "pages": [
                    {"page": 1, "blocks": [{"block_type": "table", "block_id": "tbl_001", "table_id": "tbl_001"}]},
                ]
            },
            "pages": [{"page_number": 1, "block_count": 1}],
            "table_asts": [
                {
                    "table_id": "tbl_001",
                    "display_grid": [
                        ["项目", "内容"],
                        ["枚举", "首次提交\n回复\n撤回"],
                        ["表头样式", "注册行为\n类型"],
                        ["连续短语", "临床试验\n申请"],
                        ["软换行", "这是一个较长的说明文本\n因为 PDF 自动折行被拆成两行"],
                    ],
                }
            ],
            "image_blocks": [],
            "text": "",
        }

        markdown = api_main._build_full_markdown([document])

        self.assertIn("| 枚举 | 首次提交 / 回复 / 撤回 |", markdown)
        self.assertIn("| 表头样式 | 注册行为类型 |", markdown)
        self.assertIn("| 连续短语 | 临床试验申请 |", markdown)
        self.assertIn("| 软换行 | 这是一个较长的说明文本因为 PDF 自动折行被拆成两行 |", markdown)
        self.assertNotIn("<br>", markdown)
