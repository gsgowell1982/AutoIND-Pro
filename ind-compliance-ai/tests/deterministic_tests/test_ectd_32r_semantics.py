from __future__ import annotations

import json
import unittest
from pathlib import Path
from xml.etree import ElementTree

from core.ectd_32r_semantics import (
    build_ectd_32r_semantic_contract,
    parse_figure2_skeleton,
    validate_ectd_32r_records,
)


class Ectd32RSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_ectd_32r_semantic_contract()

    def test_contract_uses_controlled_vocabulary_table4_titles(self) -> None:
        titles = [item["title"] for item in self.contract["extension_nodes"]]
        self.assertEqual(
            titles,
            [
                "3.2.R.1工艺验证",
                "3.2.R.2批记录",
                "3.2.R.3分析方法验证报告",
                "3.2.R.4稳定性图谱",
                "3.2.R.5可比性方案",
                "3.2.R.6其他",
            ],
        )
        self.assertEqual(
            self.contract["sources"]["table4"],
            "data/regulations/附件1-2：受控词汇文件包/node-extension-property_CN.xml",
        )

    def test_contract_is_valid_json_schema_instance(self) -> None:
        schema_path = Path("schemas/ectd/ectd_32r_node_extension_contract.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        try:
            import jsonschema
        except ImportError as exc:  # pragma: no cover - dependency is part of the test environment
            self.skipTest(f"jsonschema unavailable: {exc}")
        jsonschema.validate(self.contract, schema)

    def test_figure2_skeleton_parses_expected_parent_and_leaf_attributes(self) -> None:
        result = parse_figure2_skeleton()
        self.assertEqual(result["parent_path"], ["m3-quality", "m3-2-body-of-data", "m3-2-r-regional-information"])
        self.assertEqual(result["extension_count"], 2)
        self.assertEqual(result["extension_titles"], ["3.2.R.1工艺验证", "3.2.R.2批记录"])
        self.assertEqual(result["leaf_attributes"], ["ID", "operation", "xlink:type", "xlink:href", "checksum", "checksum-type"])

    def test_biologic_records_require_extensions_and_validate_parent_titles_and_paths(self) -> None:
        records = [
            {
                "extension_title": "3.2.R.1工艺验证",
                "parent_local_tag": "m3-2-r-regional-information",
                "leaf_hrefs": ["m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf"],
            }
        ]
        result = validate_ectd_32r_records(records, product_type="biologic", regional_information_present=True)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["review_required"])

    def test_non_biologic_section_does_not_require_biologic_extensions(self) -> None:
        result = validate_ectd_32r_records([], product_type="small-molecule", regional_information_present=True)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["review_required"])

    def test_biologic_section_without_extensions_requires_review(self) -> None:
        result = validate_ectd_32r_records([], product_type="biologic", regional_information_present=True)
        self.assertEqual(result["status"], "review")
        self.assertIn("biologic_32r_requires_node_extension", result["issue_codes"])

    def test_wrong_parent_title_and_href_are_reported(self) -> None:
        result = validate_ectd_32r_records(
            [
                {
                    "extension_title": "3.2.R.X错误",
                    "parent_local_tag": "m3-2-s-drug-substance",
                    "leaf_hrefs": ["m1/01-admin/wrong.pdf"],
                }
            ],
            product_type="biologic",
            regional_information_present=True,
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("invalid_extension_title", result["issue_codes"])
        self.assertIn("invalid_parent", result["issue_codes"])
        self.assertIn("invalid_leaf_href", result["issue_codes"])


if __name__ == "__main__":
    unittest.main()
