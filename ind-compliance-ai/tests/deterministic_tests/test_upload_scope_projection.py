from __future__ import annotations

import unittest

from core.upload_scope_projection import build_upload_scope_overview


class UploadScopeProjectionTests(unittest.TestCase):
    def test_build_upload_scope_overview_for_single_document(self) -> None:
        overview = build_upload_scope_overview(
            [
                {
                    "filename": "cn-regional.xml",
                    "relative_path": "cn-regional.xml",
                }
            ]
        )

        self.assertEqual(overview["upload_mode"], "single_document")
        self.assertEqual(overview["upload_mode_label"], "单文件上传")
        self.assertEqual(overview["likely_scopes"], ["document"])
        self.assertEqual(overview["sequence_root_count"], 0)
        self.assertEqual(overview["application_root_count"], 0)
        self.assertEqual(overview["signal_filenames"], ["cn-regional.xml"])

    def test_build_upload_scope_overview_for_ectd_sequence_package(self) -> None:
        overview = build_upload_scope_overview(
            [
                {
                    "filename": "x202112345/0000/index.xml",
                    "relative_path": "x202112345/0000/index.xml",
                },
                {
                    "filename": "x202112345/0000/m1/cn/cn-regional.xml",
                    "relative_path": "x202112345/0000/m1/cn/cn-regional.xml",
                },
                {
                    "filename": "x202112345/0000/m5/study.pdf",
                    "relative_path": "x202112345/0000/m5/study.pdf",
                },
            ]
        )

        self.assertEqual(overview["upload_mode"], "ectd_sequence_package")
        self.assertEqual(overview["upload_mode_label"], "eCTD 序列包")
        self.assertEqual(overview["likely_scopes"], ["document", "sequence"])
        self.assertEqual(overview["sequence_root_count"], 1)
        self.assertEqual(overview["application_root_count"], 1)
        self.assertEqual(overview["detection_basis"], "relative_path")
        self.assertEqual(overview["signal_filenames"], ["cn-regional.xml", "index.xml"])

    def test_build_upload_scope_overview_for_ectd_application_project(self) -> None:
        overview = build_upload_scope_overview(
            [
                {
                    "filename": "x202112345/0000/index.xml",
                    "relative_path": "x202112345/0000/index.xml",
                },
                {
                    "filename": "x202112345/0000/m1/cn/cn-regional.xml",
                    "relative_path": "x202112345/0000/m1/cn/cn-regional.xml",
                },
                {
                    "filename": "x202112345/0001/index.xml",
                    "relative_path": "x202112345/0001/index.xml",
                },
                {
                    "filename": "x202112345/0001/m1/cn/cn-regional.xml",
                    "relative_path": "x202112345/0001/m1/cn/cn-regional.xml",
                },
            ]
        )

        self.assertEqual(overview["upload_mode"], "ectd_application_project")
        self.assertEqual(overview["upload_mode_label"], "eCTD 申请项目")
        self.assertEqual(overview["likely_scopes"], ["document", "sequence", "application"])
        self.assertEqual(overview["sequence_root_count"], 2)
        self.assertEqual(overview["application_root_count"], 1)
        self.assertIn("上传批次已呈现多序列结构", overview["scope_notice"])


if __name__ == "__main__":
    unittest.main()
