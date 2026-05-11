from __future__ import annotations

import unittest

from core.scope_transition_projection import build_scope_transition_overview


class ScopeTransitionProjectionTests(unittest.TestCase):
    def test_build_scope_transition_overview_confirms_stable_document_scope(self) -> None:
        overview = build_scope_transition_overview(
            {
                "upload_mode": "single_document",
                "upload_mode_label": "单文件上传",
                "likely_scopes": ["document"],
                "likely_scope_labels": ["文档"],
            },
            {
                "upload_mode": "single_document",
                "upload_mode_label": "单文件上传",
                "available_scopes": ["document"],
                "available_scope_labels": ["文档"],
            },
        )

        self.assertEqual(overview["transition_status"], "confirmed")
        self.assertEqual(overview["transition_status_label"], "预判确认")
        self.assertEqual(overview["added_scopes"], [])
        self.assertEqual(overview["removed_scopes"], [])
        self.assertEqual(overview["primary_reason_code"], "stable_scope_match")
        self.assertEqual(overview["recommended_action_code"], "proceed_with_current_scope")
        self.assertIn("上传预判与解析后作用域一致", overview["transition_notice"])

    def test_build_scope_transition_overview_marks_scope_expansion(self) -> None:
        overview = build_scope_transition_overview(
            {
                "upload_mode": "ectd_sequence_package",
                "upload_mode_label": "eCTD 序列包",
                "likely_scopes": ["document", "sequence"],
                "likely_scope_labels": ["文档", "序列"],
            },
            {
                "upload_mode": "ectd_application_project",
                "upload_mode_label": "eCTD 申请项目",
                "available_scopes": ["document", "sequence", "activity", "application"],
                "available_scope_labels": ["文档", "序列", "注册行为", "申请项目"],
            },
        )

        self.assertEqual(overview["transition_status"], "expanded")
        self.assertEqual(overview["added_scopes"], ["activity", "application"])
        self.assertEqual(overview["removed_scopes"], [])
        self.assertEqual(overview["primary_reason_code"], "multi_sequence_confirmed")
        self.assertIn("多序列", overview["primary_reason_label"])
        self.assertEqual(overview["recommended_action_code"], "review_activity_application_rules")
        self.assertEqual(overview["guidance_priority"], "attention")
        self.assertIn("activity", overview["guidance_rule_groups"])
        self.assertIn("application", overview["guidance_rule_groups"])
        self.assertIn(
            {"target_type": "rule_scope", "label": "activity", "description": "优先查看 activity 级规则组。"},
            overview["guidance_target_details"],
        )
        self.assertIn(
            {"target_type": "rule_scope", "label": "application", "description": "继续查看 application 级规则组。"},
            overview["guidance_target_details"],
        )
        self.assertGreaterEqual(len(overview["guidance_steps"]), 2)
        self.assertIn("解析后确认了更高层级作用域", overview["transition_notice"])

    def test_build_scope_transition_overview_marks_scope_narrowing(self) -> None:
        overview = build_scope_transition_overview(
            {
                "upload_mode": "ectd_sequence_package",
                "upload_mode_label": "eCTD 序列包",
                "likely_scopes": ["document", "sequence"],
                "likely_scope_labels": ["文档", "序列"],
            },
            {
                "upload_mode": "single_document",
                "upload_mode_label": "单文件上传",
                "available_scopes": ["document"],
                "available_scope_labels": ["文档"],
            },
        )

        self.assertEqual(overview["transition_status"], "narrowed")
        self.assertEqual(overview["added_scopes"], [])
        self.assertEqual(overview["removed_scopes"], ["sequence"])
        self.assertEqual(overview["primary_reason_code"], "metadata_insufficient")
        self.assertEqual(overview["recommended_action_code"], "verify_ectd_metadata_files")
        self.assertEqual(overview["guidance_priority"], "priority")
        self.assertIn("index.xml", overview["guidance_targets"])
        self.assertIn("cn-regional.xml", overview["guidance_targets"])
        self.assertIn(
            {"target_type": "file", "label": "index.xml", "description": "确认主索引文件存在且与当前提交包一致。"},
            overview["guidance_target_details"],
        )
        self.assertIn(
            {"target_type": "file", "label": "cn-regional.xml", "description": "确认区域信封文件存在且能够被系统正常解析。"},
            overview["guidance_target_details"],
        )
        self.assertIn(
            {
                "target_type": "metadata_field",
                "label": "application-number",
                "description": "核对申请号是否存在、格式正确且与当前提交包一致。",
            },
            overview["guidance_target_details"],
        )
        self.assertIn(
            {
                "target_type": "metadata_field",
                "label": "sequence-number",
                "description": "核对序列号是否存在、格式正确且与目录序列一致。",
            },
            overview["guidance_target_details"],
        )
        self.assertIn(
            {
                "target_type": "metadata_field",
                "label": "regulatory-activity-type",
                "description": "核对注册行为类型是否完整且与当前提交意图一致。",
            },
            overview["guidance_target_details"],
        )
        self.assertIn(
            {
                "target_type": "metadata_field",
                "label": "sequence-type",
                "description": "核对序列类型是否完整且与注册行为/申请类型组合一致。",
            },
            overview["guidance_target_details"],
        )
        self.assertGreaterEqual(len(overview["guidance_steps"]), 3)
        self.assertIn("解析后未能确认部分预判作用域", overview["transition_notice"])

    def test_build_scope_transition_overview_marks_missing_path_context_as_narrowing_reason(self) -> None:
        overview = build_scope_transition_overview(
            {
                "upload_mode": "ectd_sequence_candidate",
                "upload_mode_label": "eCTD 序列候选",
                "likely_scopes": ["document", "sequence"],
                "likely_scope_labels": ["文档", "序列"],
                "detection_basis": "filename_only",
                "signal_filenames": ["cn-regional.xml", "index.xml"],
            },
            {
                "upload_mode": "single_document",
                "upload_mode_label": "单文件上传",
                "available_scopes": ["document"],
                "available_scope_labels": ["文档"],
            },
        )

        self.assertEqual(overview["transition_status"], "narrowed")
        self.assertEqual(overview["primary_reason_code"], "missing_path_context")
        self.assertIn("路径上下文不足", overview["primary_reason_label"])
        self.assertEqual(overview["recommended_action_code"], "preserve_relative_paths")
        self.assertEqual(overview["guidance_priority"], "priority")
        self.assertIn("relative paths", overview["guidance_targets"])
        self.assertIn(
            {
                "target_type": "upload_requirement",
                "label": "relative paths",
                "description": "不要只上传平铺文件，需保留原始相对路径关系。",
            },
            overview["guidance_target_details"],
        )
        self.assertGreaterEqual(len(overview["guidance_steps"]), 3)


if __name__ == "__main__":
    unittest.main()
