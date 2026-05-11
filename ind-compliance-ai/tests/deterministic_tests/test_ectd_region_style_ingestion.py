from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ectd_region_style_ingestion import (
    ECTD_REGION_STYLE_BUNDLE_ID,
    ECTD_REGION_STYLE_BUNDLE_VERSION,
    build_ectd_region_style_bundle,
    write_ectd_region_style_bundle,
)


def _resolve_attachment_1_3_dir() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [path for path in regulations_root.iterdir() if path.is_dir() and "1-3" in path.name]
    if not matches:
        raise unittest.SkipTest("Attachment 1-3 directory not found under data/regulations.")
    return matches[0]


class EctdRegionStyleIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_dir = _resolve_attachment_1_3_dir()
        cls.payload = build_ectd_region_style_bundle(cls.source_dir)

    def test_build_region_style_bundle_extracts_basic_stylesheet_structure(self) -> None:
        self.assertEqual(self.payload["schema_version"], ECTD_REGION_STYLE_BUNDLE_VERSION)
        self.assertEqual(self.payload["bundle_id"], ECTD_REGION_STYLE_BUNDLE_ID)
        self.assertEqual(self.payload["stylesheet_count"], 1)

        stylesheet = self.payload["stylesheets"][0]
        self.assertEqual(stylesheet["filename"], "cn-regional-1-0.xsl")
        self.assertEqual(stylesheet["stylesheet_version"], "1.0")
        self.assertEqual(stylesheet["output_method"], "html")
        self.assertEqual(stylesheet["output_encoding"], "UTF-8")
        self.assertGreaterEqual(stylesheet["template_count"], 8)

    def test_build_region_style_bundle_extracts_display_mappings_and_static_texts(self) -> None:
        stylesheet = self.payload["stylesheets"][0]
        mappings = stylesheet["controlled_display_mappings"]
        self.assertEqual(mappings["application-type"]["cnapt2"], "新药申请")
        self.assertEqual(mappings["product-type"]["cnprt2"], "生物制品")
        self.assertEqual(mappings["regulatory-activity-type"]["cnrat9"], "基线")
        self.assertEqual(mappings["sequence-type"]["cnsqt4"], "格式转换")

        static_texts = stylesheet["static_texts"]
        self.assertEqual(static_texts["main_heading"], "模块一")
        self.assertEqual(static_texts["style_version_label"], "样式文件版本 1.0")

    def test_write_region_style_bundle_emits_json_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = write_ectd_region_style_bundle(self.source_dir, output_root=Path(temp_dir))
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["bundle_id"], ECTD_REGION_STYLE_BUNDLE_ID)
            self.assertEqual(payload["stylesheets"][0]["filename"], "cn-regional-1-0.xsl")


if __name__ == "__main__":
    unittest.main()
