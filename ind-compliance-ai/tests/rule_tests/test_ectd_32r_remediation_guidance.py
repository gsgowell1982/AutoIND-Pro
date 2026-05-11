from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.material_assessment import build_compliance_result_payload
from parsers.parser_registry import parse_file


class Ectd32RRemediationGuidanceTests(unittest.TestCase):
    def _build_result(self, cn_regional_payload: str, index_payload: str) -> dict:
        with TemporaryDirectory() as temp_dir:
            application_root = Path(temp_dir) / "x202112345"
            sequence_root = application_root / "0000"
            cn_regional_path = sequence_root / "m1" / "cn" / "cn-regional.xml"
            cn_regional_path.parent.mkdir(parents=True, exist_ok=True)
            cn_regional_path.write_text(cn_regional_payload, encoding="utf-8")
            index_path = sequence_root / "index.xml"
            index_path.write_text(index_payload, encoding="utf-8")

            parsed_cn_regional = parse_file(cn_regional_path)
            parsed_index = parse_file(index_path)

            return build_compliance_result_payload(
                submission_profile="FIH",
                parsed_documents=[parsed_cn_regional, parsed_index],
                consistency_rows=[],
                final_status="completed",
            )

    def test_pass_case_keeps_remediation_guidance_empty(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="biologic"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
  <m3-quality>
    <m3-2-body-of-data>
      <m3-2-r-regional-information>
        <node-extension>
          <title>3.2.R.1工艺验证</title>
          <leaf checksum-type="MD5" checksum="abc123" xlink:href="m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf">
            <title>工艺验证</title>
          </leaf>
        </node-extension>
      </m3-2-r-regional-information>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}

        self.assertEqual(rules_by_id["SR-ECTD-007"]["status"], "pass")
        self.assertEqual(rules_by_id["SR-ECTD-007"]["details"]["remediation_guidance"], [])

    def test_invalid_titles_emit_title_fix_guidance(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="biologic"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
  <m3-quality>
    <m3-2-body-of-data>
      <m3-2-r-regional-information>
        <node-extension>
          <title>3.2.R.7原辅料说明</title>
          <leaf checksum-type="MD5" checksum="abc123" xlink:href="m3/32-body-data/32r-reg-info/cn32r7/material-note.pdf">
            <title>原辅料说明</title>
          </leaf>
        </node-extension>
      </m3-2-r-regional-information>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}
        details = rules_by_id["SR-ECTD-007"]["details"]

        self.assertEqual(rules_by_id["SR-ECTD-007"]["status"], "warn")
        self.assertEqual(
            [item["guidance_code"] for item in details["remediation_guidance"]],
            ["fix_32r_extension_titles"],
        )
        invalid_title_targets = [
            detail
            for detail in details["remediation_guidance"][0]["guidance_target_details"]
            if detail["target_type"] == "node_extension_title"
        ]
        self.assertTrue(any(detail.get("source_leaf_titles") for detail in invalid_title_targets))
        allowed_titles = [
            detail["label"]
            for detail in details["remediation_guidance"][0]["guidance_target_details"]
            if detail["target_type"] == "allowed_title"
        ]
        self.assertEqual(len(allowed_titles), 6)
        self.assertTrue(any(str(label).startswith("3.2.R.6") for label in allowed_titles))
        self.assertIn(
            "cn-regional.xml",
            [
                detail["label"]
                for detail in details["remediation_guidance"][0]["guidance_target_details"]
                if detail["target_type"] == "document_file"
            ],
        )
        self.assertEqual(len(details["extension_issue_bundles"]), 1)
        self.assertIn("invalid_extension_title", details["extension_issue_bundles"][0]["issue_codes"])
        self.assertEqual(
            details["extension_issue_bundles"][0]["primary_issue_code"],
            "invalid_extension_title",
        )
        invalid_title_diff_rows = [
            row
            for row in details["extension_issue_bundles"][0]["issue_diff_rows"]
            if row["issue_code"] == "invalid_extension_title"
        ]
        self.assertEqual(len(invalid_title_diff_rows), 1)
        self.assertEqual(invalid_title_diff_rows[0]["field_path"], "node-extension/title")
        self.assertTrue(invalid_title_diff_rows[0]["recommended_target_value"].startswith("3.2.R.6"))
        self.assertEqual(len(invalid_title_diff_rows[0]["target_value_candidates"]), 6)
        self.assertTrue(any(str(item).startswith("3.2.R.1") for item in invalid_title_diff_rows[0]["target_value_candidates"]))
        self.assertTrue(any(str(item).startswith("3.2.R.6") for item in invalid_title_diff_rows[0]["target_value_candidates"]))
        self.assertTrue(invalid_title_diff_rows[0]["suggested_snippet"].startswith("<title>3.2.R.6"))
        self.assertTrue(invalid_title_diff_rows[0]["suggested_snippet"].endswith("</title>"))
        self.assertEqual(
            invalid_title_diff_rows[0]["suggested_action_title"],
            "将 node-extension 标题改为表4允许标题",
        )
        self.assertIn(
            "3.2.R.1",
            " ".join(invalid_title_diff_rows[0]["suggested_action_steps"]),
        )
        self.assertIn(
            "3.2.R.6",
            " ".join(invalid_title_diff_rows[0]["suggested_action_steps"]),
        )
        self.assertIn(
            "node-extension/title 已改为表4允许标题",
            " ".join(invalid_title_diff_rows[0]["verification_checks"]),
        )
        self.assertIn(
            "leaf title",
            " ".join(invalid_title_diff_rows[0]["verification_checks"]),
        )

    def test_invalid_title_with_known_slot_prefix_narrows_title_candidates(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="biologic"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
  <m3-quality>
    <m3-2-body-of-data>
      <m3-2-r-regional-information>
        <node-extension>
          <title>3.2.R.1错误标题</title>
          <leaf checksum-type="MD5" checksum="abc123" xlink:href="m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf">
            <title>工艺验证</title>
          </leaf>
        </node-extension>
      </m3-2-r-regional-information>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}
        details = rules_by_id["SR-ECTD-007"]["details"]

        invalid_title_diff_rows = [
            row
            for row in details["extension_issue_bundles"][0]["issue_diff_rows"]
            if row["issue_code"] == "invalid_extension_title"
        ]
        self.assertEqual(len(invalid_title_diff_rows), 1)
        self.assertTrue(invalid_title_diff_rows[0]["recommended_target_value"].startswith("3.2.R.1"))
        self.assertEqual(len(invalid_title_diff_rows[0]["target_value_candidates"]), 1)
        self.assertTrue(invalid_title_diff_rows[0]["target_value_candidates"][0].startswith("3.2.R.1"))
        self.assertIn(
            "3.2.R.1",
            invalid_title_diff_rows[0]["recommendation_basis"],
        )

    def test_invalid_title_with_unique_leaf_title_semantics_narrows_title_candidates(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="biologic"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
  <m3-quality>
    <m3-2-body-of-data>
      <m3-2-r-regional-information>
        <node-extension>
          <title>区域性药学信息</title>
          <leaf checksum-type="MD5" checksum="abc123" xlink:href="m3/32-body-data/32r-reg-info/cn32r3/method-validation.pdf">
            <title>分析方法验证报告</title>
          </leaf>
        </node-extension>
      </m3-2-r-regional-information>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}
        details = rules_by_id["SR-ECTD-007"]["details"]

        invalid_title_diff_rows = [
            row
            for row in details["extension_issue_bundles"][0]["issue_diff_rows"]
            if row["issue_code"] == "invalid_extension_title"
        ]
        self.assertEqual(len(invalid_title_diff_rows), 1)
        self.assertTrue(invalid_title_diff_rows[0]["recommended_target_value"].startswith("3.2.R.3"))
        self.assertEqual(len(invalid_title_diff_rows[0]["target_value_candidates"]), 1)
        self.assertTrue(invalid_title_diff_rows[0]["target_value_candidates"][0].startswith("3.2.R.3"))
        self.assertIn(
            "leaf title",
            invalid_title_diff_rows[0]["recommendation_basis"],
        )

    def test_invalid_title_surfaces_conflict_when_slot_and_leaf_semantics_disagree(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="biologic"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
  <m3-quality>
    <m3-2-body-of-data>
      <m3-2-r-regional-information>
        <node-extension>
          <title>3.2.R.1错误标题</title>
          <leaf checksum-type="MD5" checksum="abc123" xlink:href="m3/32-body-data/32r-reg-info/cn32r3/method-validation.pdf">
            <title>分析方法验证报告</title>
          </leaf>
        </node-extension>
      </m3-2-r-regional-information>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}
        details = rules_by_id["SR-ECTD-007"]["details"]

        invalid_title_diff_rows = [
            row
            for row in details["extension_issue_bundles"][0]["issue_diff_rows"]
            if row["issue_code"] == "invalid_extension_title"
        ]
        self.assertEqual(len(invalid_title_diff_rows), 1)
        self.assertFalse(bool(invalid_title_diff_rows[0].get("recommended_target_value")))
        self.assertEqual(len(invalid_title_diff_rows[0]["target_value_candidates"]), 2)
        self.assertTrue(any(str(item).startswith("3.2.R.1") for item in invalid_title_diff_rows[0]["target_value_candidates"]))
        self.assertTrue(any(str(item).startswith("3.2.R.3") for item in invalid_title_diff_rows[0]["target_value_candidates"]))
        self.assertTrue(invalid_title_diff_rows[0]["has_recommendation_conflict"])
        self.assertIn(
            "3.2.R.1",
            invalid_title_diff_rows[0]["recommendation_conflict_summary"],
        )
        self.assertIn(
            "3.2.R.3",
            invalid_title_diff_rows[0]["recommendation_conflict_summary"],
        )
        self.assertEqual(
            invalid_title_diff_rows[0]["tie_break_guidance_title"],
            "冲突时请按资料语义优先复核最终 Table-4 标题",
        )
        self.assertIn(
            "leaf title",
            " ".join(invalid_title_diff_rows[0]["tie_break_guidance_steps"]),
        )
        self.assertIn(
            "xlink:href",
            " ".join(invalid_title_diff_rows[0]["tie_break_guidance_steps"]),
        )
        focus_items = invalid_title_diff_rows[0]["tie_break_focus_items"]
        self.assertEqual(len(focus_items), 5)
        self.assertIn("cn-regional.xml", [item["value"] for item in focus_items])
        self.assertIn("分析方法验证报告", [item["value"] for item in focus_items])
        self.assertIn(
            "m3/32-body-data/32r-reg-info/cn32r3/method-validation.pdf",
            [item["value"] for item in focus_items],
        )
        self.assertIn("method-validation.pdf", [item["value"] for item in focus_items])
        self.assertIn("cn32r3", [item["value"] for item in focus_items])
        candidate_comparisons = invalid_title_diff_rows[0]["tie_break_candidate_comparisons"]
        self.assertEqual(len(candidate_comparisons), 2)
        self.assertTrue(any(str(item["candidate_title"]).startswith("3.2.R.1") for item in candidate_comparisons))
        self.assertTrue(any(str(item["candidate_title"]).startswith("3.2.R.3") for item in candidate_comparisons))
        self.assertTrue(
            any("标题前缀" in " ".join(item.get("supports", [])) for item in candidate_comparisons),
        )
        self.assertTrue(
            any("leaf title" in " ".join(item.get("supports", [])) for item in candidate_comparisons),
        )
        slot_candidate_comparison = next(
            item for item in candidate_comparisons if str(item["candidate_title"]).startswith("3.2.R.1")
        )
        semantic_candidate_comparison = next(
            item for item in candidate_comparisons if str(item["candidate_title"]).startswith("3.2.R.3")
        )
        self.assertIn("3.2.R.1", " ".join(slot_candidate_comparison["supports"]))
        self.assertIn("3.2.R.3", " ".join(slot_candidate_comparison["concerns"]))
        self.assertIn("cn32r3", " ".join(slot_candidate_comparison["concerns"]))
        self.assertIn("method-validation.pdf", " ".join(slot_candidate_comparison["concerns"]))
        self.assertIn("3.2.R.3", " ".join(semantic_candidate_comparison["supports"]))
        self.assertIn("cn32r3", " ".join(semantic_candidate_comparison["supports"]))
        self.assertIn("method-validation.pdf", " ".join(semantic_candidate_comparison["supports"]))
        self.assertIn("3.2.R.1", " ".join(semantic_candidate_comparison["concerns"]))

    def test_invalid_leaf_paths_emit_path_fix_guidance(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="biologic"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
  <m3-quality>
    <m3-2-body-of-data>
      <m3-2-r-regional-information>
        <node-extension>
          <title>3.2.R.1工艺验证</title>
          <leaf checksum-type="MD5" checksum="abc123" xlink:href="m1/cover-letter.pdf">
            <title>工艺验证</title>
          </leaf>
        </node-extension>
      </m3-2-r-regional-information>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}
        details = rules_by_id["SR-ECTD-007"]["details"]

        self.assertEqual(rules_by_id["SR-ECTD-007"]["status"], "warn")
        self.assertEqual(
            [item["guidance_code"] for item in details["remediation_guidance"]],
            ["fix_32r_leaf_paths"],
        )
        leaf_targets = [
            detail
            for detail in details["remediation_guidance"][0]["guidance_target_details"]
            if detail["target_type"] == "leaf_href"
        ]
        self.assertTrue(any(detail.get("source_extension_title") for detail in leaf_targets))
        self.assertIn(
            "cn-regional.xml",
            [
                detail["label"]
                for detail in details["remediation_guidance"][0]["guidance_target_details"]
                if detail["target_type"] == "document_file"
            ],
        )
        self.assertIn(
            "m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf",
            details["remediation_guidance"][0]["guidance_steps"][2],
        )
        self.assertEqual(len(details["extension_issue_bundles"]), 1)
        self.assertIn("invalid_leaf_href", details["extension_issue_bundles"][0]["issue_codes"])
        self.assertEqual(
            details["extension_issue_bundles"][0]["primary_issue_code"],
            "invalid_leaf_href",
        )
        invalid_href_diff_rows = [
            row
            for row in details["extension_issue_bundles"][0]["issue_diff_rows"]
            if row["issue_code"] == "invalid_leaf_href"
        ]
        self.assertEqual(len(invalid_href_diff_rows), 1)
        self.assertEqual(invalid_href_diff_rows[0]["field_path"], "node-extension/leaf/@xlink:href")
        self.assertEqual(
            invalid_href_diff_rows[0]["recommended_target_value"],
            "m3/32-body-data/32r-reg-info/cn32r1/cover-letter.pdf",
        )
        self.assertEqual(
            invalid_href_diff_rows[0]["suggested_snippet"],
            'xlink:href="m3/32-body-data/32r-reg-info/cn32r1/cover-letter.pdf"',
        )
        self.assertEqual(
            invalid_href_diff_rows[0]["suggested_action_title"],
            "将 3.2.R leaf 文件迁回正确相对路径",
        )
        self.assertIn(
            "保持目录结构上传",
            " ".join(invalid_href_diff_rows[0]["suggested_action_steps"]),
        )
        self.assertIn(
            "cn32r1/cover-letter.pdf",
            " ".join(invalid_href_diff_rows[0]["suggested_action_steps"]),
        )
        self.assertIn(
            "xlink:href 与包内实际相对路径一致",
            " ".join(invalid_href_diff_rows[0]["verification_checks"]),
        )
        self.assertIn(
            "32r-reg-info",
            " ".join(invalid_href_diff_rows[0]["verification_checks"]),
        )

    def test_misplaced_extensions_emit_relocation_guidance(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="biologic"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
  <m3-quality>
    <m3-2-body-of-data>
      <node-extension>
        <title>3.2.R.1工艺验证</title>
        <leaf checksum-type="MD5" checksum="abc123" xlink:href="m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf">
          <title>工艺验证</title>
        </leaf>
      </node-extension>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}
        details = rules_by_id["SR-ECTD-007"]["details"]

        self.assertEqual(rules_by_id["SR-ECTD-007"]["status"], "warn")
        self.assertEqual(
            [item["guidance_code"] for item in details["remediation_guidance"]],
            ["relocate_32r_node_extensions"],
        )
        relocation_targets = [
            detail
            for detail in details["remediation_guidance"][0]["guidance_target_details"]
            if detail["target_type"] == "node_extension_title"
        ]
        self.assertTrue(any(detail.get("source_parent_pointer") for detail in relocation_targets))
        expected_parent_labels = [
            detail["label"]
            for detail in details["remediation_guidance"][0]["guidance_target_details"]
            if detail["target_type"] == "xml_parent"
        ]
        self.assertIn(
            "cn-regional.xml",
            [
                detail["label"]
                for detail in details["remediation_guidance"][0]["guidance_target_details"]
                if detail["target_type"] == "document_file"
            ],
        )
        self.assertIn(
            "m3-quality > m3-2-body-of-data > m3-2-r-regional-information",
            expected_parent_labels,
        )
        self.assertEqual(len(details["extension_issue_bundles"]), 1)
        self.assertIn("misplaced_extension", details["extension_issue_bundles"][0]["issue_codes"])
        self.assertEqual(
            details["extension_issue_bundles"][0]["primary_issue_code"],
            "misplaced_extension",
        )
        misplaced_diff_rows = [
            row
            for row in details["extension_issue_bundles"][0]["issue_diff_rows"]
            if row["issue_code"] == "misplaced_extension"
        ]
        self.assertEqual(len(misplaced_diff_rows), 1)
        self.assertEqual(misplaced_diff_rows[0]["field_path"], "node-extension (parent placement)")
        self.assertIn(
            "<m3-2-r-regional-information>",
            misplaced_diff_rows[0]["suggested_snippet"],
        )
        self.assertIn(
            "<title>3.2.R.1",
            misplaced_diff_rows[0]["suggested_snippet"],
        )
        self.assertIn(
            'xlink:href="m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf"',
            misplaced_diff_rows[0]["suggested_snippet"],
        )
        self.assertEqual(
            misplaced_diff_rows[0]["suggested_action_title"],
            "将 node-extension 整体迁回 3.2.R 区域性管理信息节点下",
        )
        self.assertIn(
            "m3-2-r-regional-information",
            " ".join(misplaced_diff_rows[0]["suggested_action_steps"]),
        )
        self.assertIn(
            "标题",
            " ".join(misplaced_diff_rows[0]["suggested_action_steps"]),
        )
        self.assertIn(
            "m3-2-r-regional-information",
            " ".join(misplaced_diff_rows[0]["verification_checks"]),
        )
        self.assertIn(
            "标题和 href",
            " ".join(misplaced_diff_rows[0]["verification_checks"]),
        )

    def test_same_extension_can_aggregate_multiple_issue_codes_in_one_bundle(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="biologic"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
  <m3-quality>
    <m3-2-body-of-data>
      <node-extension>
        <title>3.2.R.7原辅料说明</title>
        <leaf checksum-type="MD5" checksum="abc123" xlink:href="m1/cover-letter.pdf">
          <title>原辅料说明</title>
        </leaf>
      </node-extension>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}
        details = rules_by_id["SR-ECTD-007"]["details"]

        self.assertEqual(rules_by_id["SR-ECTD-007"]["status"], "warn")
        self.assertEqual(len(details["extension_issue_bundles"]), 1)
        self.assertEqual(
            set(details["extension_issue_bundles"][0]["issue_codes"]),
            {"invalid_extension_title", "invalid_leaf_href", "misplaced_extension"},
        )
        self.assertEqual(
            details["extension_issue_bundles"][0]["recommended_fix_sequence"],
            ["misplaced_extension", "invalid_extension_title", "invalid_leaf_href"],
        )
        self.assertEqual(
            [row["field_path"] for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [
                "node-extension (parent placement)",
                "node-extension/title",
                "node-extension/leaf/@xlink:href",
            ],
        )
        self.assertEqual(
            [bool(row.get("suggested_snippet")) for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [True, True, True],
        )
        self.assertEqual(
            [bool(row.get("suggested_action_title")) for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [True, True, True],
        )
        self.assertEqual(
            [bool(row.get("suggested_action_steps")) for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [True, True, True],
        )
        self.assertEqual(
            [bool(row.get("verification_checks")) for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [True, True, True],
        )
        self.assertEqual(
            [bool(row.get("tie_break_guidance_title")) for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [False, False, False],
        )
        self.assertEqual(
            [bool(row.get("tie_break_focus_items")) for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [False, False, False],
        )
        self.assertEqual(
            [bool(row.get("tie_break_candidate_comparisons")) for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [False, False, False],
        )
        self.assertEqual(
            [bool(row.get("recommended_target_value")) for row in details["extension_issue_bundles"][0]["issue_diff_rows"]],
            [False, True, True],
        )

    def test_na_case_keeps_remediation_guidance_empty(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    application-type="new-drug-application"
    product-type="chemical"
    sequence-number="0000"
    related-sequence="0000"
    sequence-description="initial submission"
  />
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0000" related-sequence="0000" />
</ectd:ectd>
"""

        result = self._build_result(cn_regional_payload, index_payload)
        rules_by_id = {item["rule_id"]: item for item in result["rules"]}

        self.assertEqual(rules_by_id["SR-ECTD-007"]["status"], "na")
        self.assertEqual(rules_by_id["SR-ECTD-007"]["details"]["extension_issue_bundles"], [])
        self.assertEqual(rules_by_id["SR-ECTD-007"]["details"]["remediation_guidance"], [])


if __name__ == "__main__":
    unittest.main()
