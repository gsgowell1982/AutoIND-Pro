from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ectd_validation_standard_review_export import (
    build_validation_standard_review_markdown,
    write_validation_standard_review_markdown,
)


class EctdValidationStandardReviewExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.regulations_root = cls.repo_root / "data" / "regulations"
        cls.normalized_root = cls.regulations_root / "normalized"
        cls.markdown = build_validation_standard_review_markdown(
            cls.regulations_root,
            cls.normalized_root,
        )

    def test_review_markdown_keeps_utf8_labels(self) -> None:
        self.assertIn("# eCTD验证标准 提取校对稿", self.markdown)
        self.assertIn("- 来源文件:", self.markdown)
        self.assertIn("- 提取验证项数: `149`", self.markdown)
        self.assertIn("- `严重程度`: `提示信息`", self.markdown)

    def test_review_markdown_preserves_wrapped_detail_lines(self) -> None:
        self.assertIn("```text", self.markdown)
        self.assertIn("util文件夹必须存在，并检查该文件夹中的下列文件：", self.markdown)
        self.assertIn("- ich-ectd-3-2.dtd (checksum必须符合ICH发布的值)", self.markdown)
        self.assertIn("请注意，只有在当前序列使用到STF时才需要检查ich-stf-v2-2.dtd、stf样式表和valid-values.xml。", self.markdown)

    def test_review_markdown_splits_inline_detail_for_622_and_structures_font_list_for_626(self) -> None:
        self.assertIn("### 6.22 PDF应该设置启用“快速Web访问（Fast Web Access）”", self.markdown)
        self.assertIn("不能提交未启用“快速Web访问（Fast Web Access）”情况下创建的PDF文件。参考文献（2.7.5章", self.markdown)
        self.assertNotIn("### 6.22 PDF应该设置启用“快速Web访问（Fast Web Access）” 不能提交", self.markdown)
        self.assertIn("- `结构化列表项`:", self.markdown)
        self.assertIn("Times New Roman Bold Italic", self.markdown)
        self.assertIn("- `结构化单行比对串`: 宋体 | Times New Roman | Times New Roman Italic", self.markdown)
        self.assertNotIn("必须遵守的关键验证标准", self.markdown)

    def test_write_review_markdown_emits_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = write_validation_standard_review_markdown(
                self.regulations_root,
                self.normalized_root,
                output_path=Path(temp_dir) / "review.md",
            )
            self.assertTrue(output_path.exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("## 1. 基础识别", content)
            self.assertIn("### 6.26 如使用非标准字体，需嵌入在PDF文件中", content)


if __name__ == "__main__":
    unittest.main()
