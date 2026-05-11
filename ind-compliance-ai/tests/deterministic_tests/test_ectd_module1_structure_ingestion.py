from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ectd_module1_structure_ingestion import (
    ECTD_MODULE1_STRUCTURE_BUNDLE_VERSION,
    build_ectd_module1_structure_bundle,
    write_ectd_module1_structure_bundle,
)


def _resolve_module1_structure_pdf() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [
        path
        for path in regulations_root.rglob("*.pdf")
        if "1-4" in path.name or "1-4" in str(path.parent)
    ]
    if not matches:
        raise unittest.SkipTest("Attachment 1-4 module-1 structure pdf not found under data/regulations.")
    return matches[0]


class EctdModule1StructureIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = _resolve_module1_structure_pdf()
        cls.bundle = build_ectd_module1_structure_bundle(cls.source_path)

    def test_build_bundle_extracts_expected_record_count_and_core_rows(self) -> None:
        bundle = self.bundle

        self.assertEqual(bundle["schema_version"], ECTD_MODULE1_STRUCTURE_BUNDLE_VERSION)
        self.assertEqual(bundle["bundle_id"], "cn_ectd_attachment_1_4")
        self.assertEqual(bundle["source_path"], str(self.source_path))
        self.assertEqual(bundle["record_count"], 70)

        records_by_ordinal = {
            int(record["ordinal"]): record for record in bundle["records"]
        }

        self.assertEqual(records_by_ordinal[2]["section_no"], "1.0")
        self.assertEqual(records_by_ordinal[2]["title"], "说明函")
        self.assertEqual(records_by_ordinal[2]["element"], "cn-1-0")
        self.assertEqual(records_by_ordinal[2]["entry_kind"], "文件")
        self.assertEqual(records_by_ordinal[2]["path"], "m1/cn/00/cover-letter.pdf")

        self.assertEqual(records_by_ordinal[6]["section_no"], "1.3.1.1")
        self.assertEqual(
            records_by_ordinal[6]["title"],
            "研究药物说明书及修订说明（适用于临床试验申请）",
        )
        self.assertEqual(records_by_ordinal[6]["element"], "cn-1-3-1-1")
        self.assertEqual(records_by_ordinal[6]["path"], "m1/cn/03/pi-ind-drug.pdf")
        self.assertEqual(records_by_ordinal[6]["page_span"], [2, 3])

        self.assertEqual(records_by_ordinal[58]["raw_section_no"], "1. 8.1.6")
        self.assertEqual(records_by_ordinal[58]["section_no"], "1.8.1.6")
        self.assertEqual(records_by_ordinal[58]["element"], "cn-1-8-1-6")

        self.assertEqual(
            records_by_ordinal[36]["title"],
            "申请撤回尚未批准的药物临床试验申请、上市注册许可申请、补充申请或再注册申请",
        )
        self.assertEqual(
            records_by_ordinal[37]["title"],
            "申请上市注册审评期间变更仅包括申请人更名、变更注册地址名称等不涉及技术审评内容的变更",
        )

        self.assertEqual(records_by_ordinal[60]["section_no"], "1.8.3")
        self.assertEqual(records_by_ordinal[60]["path"], "m1/cn/08/rmp.pdf")
        self.assertEqual(records_by_ordinal[60]["page_span"], [11, 12])
        self.assertIn("药物警戒活动计划", records_by_ordinal[60]["description"])
        self.assertNotIn("药物警戒活动计 划", records_by_ordinal[60]["description"])

        self.assertEqual(records_by_ordinal[70]["section_no"], "1.12")
        self.assertEqual(records_by_ordinal[70]["path"], "m1/cn/12/cert-docs-for-smallmicro.pdf")
        self.assertEqual(records_by_ordinal[70]["page_span"], [13, 14])
        self.assertEqual(records_by_ordinal[70]["description"], "小微企业证明文件（如适用）")

    def test_build_bundle_reports_page_and_kind_summary(self) -> None:
        bundle = self.bundle

        self.assertEqual(bundle["page_count"], 14)
        self.assertEqual(bundle["entry_kind_counts"]["文件"], 54)
        self.assertEqual(bundle["entry_kind_counts"]["目录"], 16)
        self.assertEqual(bundle["cross_page_record_count"], 10)

    def test_write_bundle_emits_normalized_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            artifact_path = write_ectd_module1_structure_bundle(
                self.source_path,
                output_root=output_root,
            )

            self.assertTrue(artifact_path.exists())
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["bundle_id"], "cn_ectd_attachment_1_4")
            self.assertEqual(payload["record_count"], 70)


if __name__ == "__main__":
    unittest.main()
