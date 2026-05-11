from __future__ import annotations

import unittest

from core.structure_audit_projection import (
    build_structure_audit_export_payload,
    build_structure_audit_markdown_report,
    build_structure_audit_navigation_targets,
)


class StructureAuditProjectionTests(unittest.TestCase):
    def test_builds_toc_and_body_navigation_targets_with_outline_indices(self) -> None:
        record = {
            "projected_root_page_values": [5, 7, 6],
            "missing_body_root_outline_indices": ["1", "2"],
            "missing_body_direct_child_outline_indices": ["1.1"],
            "missing_body_bounded_subtree_outline_indices": ["1.1.1.2"],
        }
        toc_sequences = [
            {
                "toc_sequence_id": "toc-seq-001",
                "pages": [2, 3],
                "page_span": [2, 3],
            }
        ]

        self.assertEqual(
            build_structure_audit_navigation_targets(record, toc_sequences),
            [
                {
                    "target_kind": "toc",
                    "label": "目录起始页",
                    "page": 2,
                    "toc_sequence_id": "toc-seq-001",
                    "outline_indices": ["1", "2", "1.1", "1.1.1.2"],
                },
                {
                    "target_kind": "body_root",
                    "label": "正文根章节起始页",
                    "page": 5,
                    "toc_sequence_id": None,
                    "outline_indices": ["1", "2", "1.1", "1.1.1.2"],
                },
            ],
        )

    def test_returns_only_body_target_when_no_toc_sequence_exists(self) -> None:
        record = {
            "projected_root_page_values": [9],
            "missing_body_root_outline_indices": ["3"],
        }

        self.assertEqual(
            build_structure_audit_navigation_targets(record, []),
            [
                {
                    "target_kind": "body_root",
                    "label": "正文根章节起始页",
                    "page": 9,
                    "toc_sequence_id": None,
                    "outline_indices": ["3"],
                }
            ],
        )

    def test_returns_empty_list_when_no_navigation_anchor_can_be_resolved(self) -> None:
        self.assertEqual(build_structure_audit_navigation_targets({}, []), [])

    def test_builds_export_payload_with_summary_counts_and_records(self) -> None:
        records = [
            {
                "filename": "toc-gap-a.pdf",
                "alignment_ready": False,
                "missing_body_direct_child_path_rows": [
                    {"outline_path": "1.0 > 1.2"},
                    {"outline_path": "2.0 > 2.3"},
                ],
                "missing_body_bounded_subtree_path_rows": [
                    {"outline_path": "1.0 > 1.1 > 1.1.2"},
                ],
                "root_page_alignment_rows": [
                    {"order_conflict": True, "offset_conflict": False, "span_conflict": False},
                    {"order_conflict": False, "offset_conflict": False, "span_conflict": False},
                ],
            },
            {
                "filename": "toc-gap-b.pdf",
                "alignment_ready": True,
                "missing_body_direct_child_path_rows": [],
                "missing_body_bounded_subtree_path_rows": [],
                "root_page_alignment_rows": [],
            },
        ]

        payload = build_structure_audit_export_payload(records, generated_at="2026-04-09T12:00:00Z")

        self.assertEqual(payload["schema_version"], "structure-audit-export-v1")
        self.assertEqual(payload["generated_at"], "2026-04-09T12:00:00Z")
        self.assertEqual(payload["summary"]["record_count"], 2)
        self.assertEqual(payload["summary"]["warning_record_count"], 1)
        self.assertEqual(payload["summary"]["passing_record_count"], 1)
        self.assertEqual(payload["summary"]["document_count"], 2)
        self.assertEqual(payload["summary"]["documents_with_path_gaps"], ["toc-gap-a.pdf"])
        self.assertEqual(payload["summary"]["missing_direct_child_path_count"], 2)
        self.assertEqual(payload["summary"]["missing_bounded_subtree_path_count"], 1)
        self.assertEqual(payload["summary"]["root_page_conflict_count"], 1)
        self.assertEqual(payload["records"], records)

    def test_builds_markdown_report_with_summary_and_path_details(self) -> None:
        records = [
            {
                "filename": "toc-gap-a.pdf",
                "alignment_ready": False,
                "missing_body_direct_child_path_rows": [
                    {
                        "text_path": "Overview > Composition",
                        "outline_path": "1.0 > 1.2",
                        "nearest_body_anchor_outline_index": "1",
                        "nearest_body_anchor_page": 3,
                    }
                ],
                "missing_body_bounded_subtree_path_rows": [
                    {
                        "text_path": "Overview > Scope > Dosage > Administration",
                        "outline_path": "1.0 > 1.1 > 1.1.1 > 1.1.1.2",
                        "nearest_body_anchor_outline_index": "1.1.1",
                        "nearest_body_anchor_page": 5,
                    }
                ],
                "root_page_alignment_rows": [
                    {
                        "outline_index": "2.0",
                        "toc_navigation_page": 2,
                        "toc_page_locator_value": 12,
                        "body_page_start": 18,
                        "body_page_end": 24,
                        "projected_page": 18,
                        "offset": 6,
                        "order_conflict": False,
                        "offset_conflict": True,
                        "span_conflict": False,
                    }
                ],
            }
        ]

        report = build_structure_audit_markdown_report(records, generated_at="2026-04-09T12:00:00Z")

        self.assertIn("# 结构审计报告", report)
        self.assertIn("生成时间：2026-04-09T12:00:00Z", report)
        self.assertIn("- 文档数：1", report)
        self.assertIn("- 缺失直接子级路径：1", report)
        self.assertIn("- 缺失子树路径：1", report)
        self.assertIn("## 预警文档", report)
        self.assertIn("### toc-gap-a.pdf", report)
        self.assertIn("### 缺失直接子级路径", report)
        self.assertIn("Overview > Composition", report)
        self.assertIn("最近正文锚点：1（第 3 页）", report)
        self.assertIn("### 缺失子树路径", report)
        self.assertIn("Overview > Scope > Dosage > Administration", report)
        self.assertIn("### 根章节页码冲突", report)
        self.assertIn("TOC页 2 / 标注页 12 / 正文页区间 18-24", report)

    def test_markdown_report_groups_warning_records_before_passing_records_with_severity_labels(self) -> None:
        records = [
            {
                "filename": "passing-doc.pdf",
                "alignment_ready": True,
                "missing_body_direct_child_path_rows": [],
                "missing_body_bounded_subtree_path_rows": [],
                "root_page_alignment_rows": [],
            },
            {
                "filename": "warning-doc.pdf",
                "alignment_ready": False,
                "missing_body_direct_child_path_rows": [
                    {
                        "text_path": "Overview > Composition",
                        "outline_path": "1.0 > 1.2",
                    }
                ],
                "missing_body_bounded_subtree_path_rows": [
                    {
                        "text_path": "Overview > Scope > Dosage > Administration",
                        "outline_path": "1.0 > 1.1 > 1.1.1 > 1.1.1.2",
                    }
                ],
                "root_page_alignment_rows": [
                    {
                        "outline_index": "2.0",
                        "toc_navigation_page": 2,
                        "toc_page_locator_value": 12,
                        "body_page_start": 18,
                        "body_page_end": 24,
                        "order_conflict": False,
                        "offset_conflict": True,
                        "span_conflict": False,
                    }
                ],
            },
        ]

        report = build_structure_audit_markdown_report(records, generated_at="2026-04-09T12:00:00Z")

        warning_group_index = report.index("## 预警文档")
        passing_group_index = report.index("## 通过文档")
        warning_doc_index = report.index("### warning-doc.pdf")
        passing_doc_index = report.index("### passing-doc.pdf")

        self.assertLess(warning_group_index, passing_group_index)
        self.assertLess(warning_doc_index, passing_doc_index)
        self.assertIn("- 严重级别：高", report)
        self.assertIn("#### 高严重级别问题", report)
        self.assertIn("#### 中严重级别问题", report)
        self.assertIn("#### 低严重级别问题", report)
        self.assertLess(report.index("### 缺失直接子级路径"), report.index("### 缺失子树路径"))
        self.assertLess(report.index("### 缺失子树路径"), report.index("### 根章节页码冲突"))


if __name__ == "__main__":
    unittest.main()
