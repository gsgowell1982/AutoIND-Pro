from __future__ import annotations

import unittest

from core.submission_scope_projection import build_submission_scope_overview


class SubmissionScopeProjectionTests(unittest.TestCase):
    def test_build_submission_scope_overview_for_single_document(self) -> None:
        overview = build_submission_scope_overview(
            {
                "upload_mode": "single_document",
                "available_scopes": ["document"],
                "ectd_project_context": {
                    "sequence_package_count": 1,
                    "regulatory_activity_count": 1,
                    "application_project_count": 1,
                },
            }
        )

        self.assertEqual(overview["upload_mode"], "single_document")
        self.assertEqual(overview["upload_mode_label"], "单文件上传")
        self.assertEqual(overview["available_scopes"], ["document"])
        self.assertEqual(overview["available_scope_labels"], ["文档"])
        self.assertEqual(overview["blocked_scopes"], ["sequence", "activity", "application"])
        self.assertEqual(overview["blocked_scope_labels"], ["序列", "注册行为", "申请项目"])
        self.assertIn("仅执行 document 级规则", overview["scope_notice"])
        self.assertEqual(overview["sequence_package_count"], 1)
        self.assertEqual(overview["regulatory_activity_count"], 1)
        self.assertEqual(overview["application_project_count"], 1)

    def test_build_submission_scope_overview_for_application_project(self) -> None:
        overview = build_submission_scope_overview(
            {
                "upload_mode": "ectd_application_project",
                "available_scopes": ["document", "sequence", "activity", "application"],
                "ectd_project_context": {
                    "sequence_package_count": 3,
                    "regulatory_activity_count": 2,
                    "application_project_count": 1,
                },
            }
        )

        self.assertEqual(overview["upload_mode_label"], "eCTD 申请项目")
        self.assertEqual(overview["available_scope_labels"], ["文档", "序列", "注册行为", "申请项目"])
        self.assertEqual(overview["blocked_scopes"], [])
        self.assertIn("已具备 document/sequence/activity/application 级规则执行条件", overview["scope_notice"])
        self.assertEqual(overview["sequence_package_count"], 3)
        self.assertEqual(overview["regulatory_activity_count"], 2)
        self.assertEqual(overview["application_project_count"], 1)


if __name__ == "__main__":
    unittest.main()
