from __future__ import annotations

from pathlib import Path
import unittest

from scripts.pdf_regression_report import (
    _aggregate,
    _compare_markdown_rendering_audit_baseline,
    _doc_record,
    _render_markdown,
)


class PdfRegressionReportContractTests(unittest.TestCase):
    maxDiff = None

    def test_doc_record_includes_table_contract_summary(self) -> None:
        sample_path = Path(__file__).resolve().parents[3] / "eCTD技术规范.pdf"
        if not sample_path.exists():
            raise unittest.SkipTest("eCTD technical specification sample not found")

        record = _doc_record(sample_path)

        self.assertIn("table_contract", record)
        contract = record["table_contract"]
        self.assertEqual(contract["table_count"], record["table_count"])
        self.assertGreater(contract["continued_table_count"], 0)
        self.assertEqual(len(contract["tables"]), record["table_count"])
        first_table = contract["tables"][0]
        self.assertIn("views", first_table)
        self.assertIn("fingerprints", first_table)
        self.assertIn("raw_grid", first_table["views"])

    def test_markdown_report_renders_table_contract_summary(self) -> None:
        report = {
            "generated_at": "2026-05-11T00:00:00",
            "input": "sample.pdf",
            "glob": "*.pdf",
            "summary": {
                "documents": 1,
                "total_pages": 1,
                "total_tables": 1,
                "total_raw_candidates": 1,
                "total_accepted_candidates": 1,
                "total_rejected_candidates": 0,
                "total_cross_page_links": 0,
                "total_low_confidence_tables": 0,
                "total_possible_missing_content_tables": 0,
                "total_possible_missing_content_candidates": 0,
                "acceptance_rate": 1.0,
            },
            "documents": [
                {
                    "file": "sample.pdf",
                    "page_count": 1,
                    "table_count": 1,
                    "raw_table_candidates": 1,
                    "accepted_table_candidates": 1,
                    "rejected_table_candidates": 0,
                    "cross_page_table_links": 0,
                    "low_confidence_table_count": 0,
                    "possible_missing_content_table_count": 0,
                    "possible_missing_content_candidate_count": 0,
                    "continuation_similarity_avg": None,
                    "table_contract": {
                        "table_count": 1,
                        "continued_table_count": 0,
                        "review_required_table_count": 0,
                        "tables": [
                            {
                                "table_id": "tbl_001",
                                "fingerprints": {"data_grid": "abc123"},
                            }
                        ],
                    },
                }
            ],
            "errors": [],
        }

        markdown = _render_markdown(report)

        self.assertIn("## Table Contract Summary", markdown)
        self.assertIn("tbl_001:abc123", markdown)

    def test_aggregate_includes_markdown_rendering_audit_summary(self) -> None:
        summary = _aggregate(
            [
                {
                    "page_count": 1,
                    "table_count": 1,
                    "raw_table_candidates": 1,
                    "accepted_table_candidates": 1,
                    "rejected_table_candidates": 0,
                    "cross_page_table_links": 0,
                    "review_required_table_count": 0,
                    "low_confidence_table_count": 0,
                    "possible_missing_content_table_count": 0,
                    "possible_missing_content_candidate_count": 0,
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 2,
                        "reason_counts": {
                            "absorbed_structure_template": 1,
                            "metadata_reference_edge_metadata_only": 1,
                        },
                    },
                },
                {
                    "page_count": 1,
                    "table_count": 0,
                    "raw_table_candidates": 0,
                    "accepted_table_candidates": 0,
                    "rejected_table_candidates": 0,
                    "cross_page_table_links": 0,
                    "review_required_table_count": 0,
                    "low_confidence_table_count": 0,
                    "possible_missing_content_table_count": 0,
                    "possible_missing_content_candidate_count": 0,
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 1,
                        "reason_counts": {"metadata_reference_edge_metadata_only": 1},
                    },
                },
            ]
        )

        self.assertEqual(summary["total_markdown_suppressed_blocks"], 3)
        self.assertEqual(
            summary["markdown_suppression_reason_counts"],
            {
                "absorbed_structure_template": 1,
                "metadata_reference_edge_metadata_only": 2,
            },
        )

    def test_markdown_report_renders_markdown_rendering_audit_summary(self) -> None:
        report = {
            "generated_at": "2026-05-11T00:00:00",
            "input": "sample.pdf",
            "glob": "*.pdf",
            "summary": {
                "documents": 1,
                "total_pages": 1,
                "total_tables": 1,
                "total_raw_candidates": 1,
                "total_accepted_candidates": 1,
                "total_rejected_candidates": 0,
                "total_cross_page_links": 0,
                "total_review_required_tables": 0,
                "total_low_confidence_tables": 0,
                "total_possible_missing_content_tables": 0,
                "total_possible_missing_content_candidates": 0,
                "total_markdown_suppressed_blocks": 2,
                "markdown_suppression_reason_counts": {
                    "absorbed_structure_template": 1,
                    "metadata_reference_edge_metadata_only": 1,
                },
                "acceptance_rate": 1.0,
            },
            "documents": [
                {
                    "file": "sample.pdf",
                    "page_count": 1,
                    "table_count": 1,
                    "raw_table_candidates": 1,
                    "accepted_table_candidates": 1,
                    "rejected_table_candidates": 0,
                    "cross_page_table_links": 0,
                    "low_confidence_table_count": 0,
                    "possible_missing_content_table_count": 0,
                    "possible_missing_content_candidate_count": 0,
                    "continuation_similarity_avg": None,
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 2,
                        "reason_counts": {
                            "absorbed_structure_template": 1,
                            "metadata_reference_edge_metadata_only": 1,
                        },
                    },
                    "table_contract": {
                        "table_count": 1,
                        "continued_table_count": 0,
                        "review_required_table_count": 0,
                        "tables": [],
                    },
                }
            ],
            "errors": [],
        }

        markdown = _render_markdown(report)

        self.assertIn("- Markdown Suppressed Blocks: `2`", markdown)
        self.assertIn(
            "- Markdown Suppression Reasons: `absorbed_structure_template=1, metadata_reference_edge_metadata_only=1`",
            markdown,
        )
        self.assertIn(
            "| sample.pdf | 2 | absorbed_structure_template=1, metadata_reference_edge_metadata_only=1 |",
            markdown,
        )

    def test_markdown_rendering_audit_baseline_fails_when_suppression_drops(self) -> None:
        current = {
            "summary": {
                "total_markdown_suppressed_blocks": 1,
                "markdown_suppression_reason_counts": {
                    "metadata_reference_edge_metadata_only": 1,
                },
            }
        }
        baseline = {
            "baseline_id": "r2_rendering_ownership_baseline",
            "summary": {
                "total_markdown_suppressed_blocks": 3,
                "markdown_suppression_reason_counts": {
                    "absorbed_structure_template": 1,
                    "metadata_reference_edge_metadata_only": 2,
                },
            },
        }

        result = _compare_markdown_rendering_audit_baseline(current, baseline)

        self.assertFalse(result["passed"])
        self.assertEqual(result["baseline_id"], "r2_rendering_ownership_baseline")
        self.assertEqual(
            result["category_counts"],
            {
                "global_reason_drop": 2,
                "global_total_drop": 1,
            },
        )
        self.assertEqual(
            result["severity_counts"],
            {
                "high": 1,
                "medium": 2,
            },
        )
        self.assertEqual(
            result["failures"],
            [
                {
                    "metric": "total_markdown_suppressed_blocks",
                    "category": "global_total_drop",
                    "severity": "high",
                    "baseline": 3,
                    "current": 1,
                    "delta": -2,
                },
                {
                    "metric": "markdown_suppression_reason_counts.absorbed_structure_template",
                    "category": "global_reason_drop",
                    "severity": "medium",
                    "baseline": 1,
                    "current": 0,
                    "delta": -1,
                },
                {
                    "metric": "markdown_suppression_reason_counts.metadata_reference_edge_metadata_only",
                    "category": "global_reason_drop",
                    "severity": "medium",
                    "baseline": 2,
                    "current": 1,
                    "delta": -1,
                },
            ],
        )

    def test_markdown_rendering_audit_baseline_allows_equal_or_higher_suppression(self) -> None:
        current = {
            "summary": {
                "total_markdown_suppressed_blocks": 4,
                "markdown_suppression_reason_counts": {
                    "absorbed_structure_template": 1,
                    "metadata_reference_edge_metadata_only": 3,
                },
            }
        }
        baseline = {
            "summary": {
                "total_markdown_suppressed_blocks": 3,
                "markdown_suppression_reason_counts": {
                    "absorbed_structure_template": 1,
                    "metadata_reference_edge_metadata_only": 2,
                },
            },
        }

        result = _compare_markdown_rendering_audit_baseline(current, baseline)

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_markdown_rendering_audit_baseline_reports_document_level_drops(self) -> None:
        current = {
            "summary": {
                "total_markdown_suppressed_blocks": 2,
                "markdown_suppression_reason_counts": {
                    "absorbed_structure_template": 1,
                    "metadata_reference_edge_metadata_only": 1,
                },
            },
            "documents": [
                {
                    "file": "r2.pdf",
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 1,
                        "reason_counts": {"metadata_reference_edge_metadata_only": 1},
                    },
                },
                {
                    "file": "ectd.pdf",
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 1,
                        "reason_counts": {"absorbed_structure_template": 1},
                    },
                },
            ],
        }
        baseline = {
            "summary": {
                "total_markdown_suppressed_blocks": 3,
                "markdown_suppression_reason_counts": {
                    "absorbed_structure_template": 1,
                    "metadata_reference_edge_metadata_only": 2,
                },
            },
            "documents": [
                {
                    "file": "r2.pdf",
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 2,
                        "reason_counts": {
                            "absorbed_structure_template": 1,
                            "metadata_reference_edge_metadata_only": 1,
                        },
                    },
                },
                {
                    "file": "ectd.pdf",
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 1,
                        "reason_counts": {"absorbed_structure_template": 1},
                    },
                },
            ],
        }

        result = _compare_markdown_rendering_audit_baseline(current, baseline)

        self.assertFalse(result["passed"])
        self.assertIn(
            {
                "metric": "documents.markdown_suppressed_blocks",
                "category": "document_total_drop",
                "severity": "high",
                "file": "r2.pdf",
                "baseline": 2,
                "current": 1,
                "delta": -1,
            },
            result["failures"],
        )
        self.assertIn(
            {
                "metric": "documents.markdown_suppression_reason_counts.absorbed_structure_template",
                "category": "document_reason_drop",
                "severity": "medium",
                "file": "r2.pdf",
                "baseline": 1,
                "current": 0,
                "delta": -1,
            },
            result["failures"],
        )

    def test_markdown_rendering_audit_baseline_orders_high_severity_failures_first(self) -> None:
        current = {
            "summary": {
                "total_markdown_suppressed_blocks": 3,
                "markdown_suppression_reason_counts": {"metadata_reference_edge_metadata_only": 1},
            },
            "documents": [
                {
                    "file": "r2.pdf",
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 1,
                        "reason_counts": {"metadata_reference_edge_metadata_only": 1},
                    },
                }
            ],
        }
        baseline = {
            "summary": {
                "total_markdown_suppressed_blocks": 3,
                "markdown_suppression_reason_counts": {
                    "absorbed_structure_template": 1,
                    "metadata_reference_edge_metadata_only": 1,
                },
            },
            "documents": [
                {
                    "file": "r2.pdf",
                    "markdown_rendering_audit_summary": {
                        "suppressed_block_count": 2,
                        "reason_counts": {"metadata_reference_edge_metadata_only": 1},
                    },
                }
            ],
        }

        result = _compare_markdown_rendering_audit_baseline(current, baseline)

        self.assertEqual(
            [
                (item.get("severity"), item.get("category"), item.get("file"), item.get("metric"))
                for item in result["failures"]
            ],
            [
                (
                    "high",
                    "document_total_drop",
                    "r2.pdf",
                    "documents.markdown_suppressed_blocks",
                ),
                (
                    "medium",
                    "global_reason_drop",
                    None,
                    "markdown_suppression_reason_counts.absorbed_structure_template",
                ),
            ],
        )
        self.assertEqual(
            result["top_failures"],
            result["failures"][:2],
        )

    def test_markdown_report_renders_markdown_rendering_audit_baseline_failures(self) -> None:
        report = {
            "generated_at": "2026-05-11T00:00:00",
            "input": "sample.pdf",
            "glob": "*.pdf",
            "summary": {
                "documents": 1,
                "total_pages": 1,
                "total_tables": 0,
                "total_raw_candidates": 0,
                "total_accepted_candidates": 0,
                "total_rejected_candidates": 0,
                "total_cross_page_links": 0,
                "total_review_required_tables": 0,
                "total_low_confidence_tables": 0,
                "total_possible_missing_content_tables": 0,
                "total_possible_missing_content_candidates": 0,
                "total_markdown_suppressed_blocks": 1,
                "markdown_suppression_reason_counts": {"metadata_reference_edge_metadata_only": 1},
                "acceptance_rate": None,
            },
            "documents": [],
            "markdown_rendering_audit_regression": {
                "baseline_id": "r2_rendering_ownership_baseline",
                "passed": False,
                "failures": [
                    {
                        "metric": "total_markdown_suppressed_blocks",
                        "category": "global_total_drop",
                        "severity": "high",
                        "baseline": 3,
                        "current": 1,
                        "delta": -2,
                    },
                    {
                        "metric": "documents.markdown_suppressed_blocks",
                        "category": "document_total_drop",
                        "severity": "high",
                        "file": "r2.pdf",
                        "baseline": 2,
                        "current": 1,
                        "delta": -1,
                    }
                ],
            },
            "errors": [],
        }

        markdown = _render_markdown(report)

        self.assertIn("## Markdown Rendering Audit Regression Guard", markdown)
        self.assertIn("- Baseline: `r2_rendering_ownership_baseline`", markdown)
        self.assertIn("- Passed: `False`", markdown)
        self.assertIn("- Failure Categories: `document_total_drop=1, global_total_drop=1`", markdown)
        self.assertIn("- Failure Severities: `high=2`", markdown)
        self.assertIn("### Top Failures", markdown)
        self.assertIn("| File | Category | Severity | Metric | Delta |", markdown)
        self.assertIn("|  | global_total_drop | high | total_markdown_suppressed_blocks | -2 |", markdown)
        self.assertIn("| r2.pdf | document_total_drop | high | documents.markdown_suppressed_blocks | -1 |", markdown)
        self.assertIn("| File | Category | Severity | Metric | Baseline | Current | Delta |", markdown)
        self.assertIn("|  | global_total_drop | high | total_markdown_suppressed_blocks | 3 | 1 | -2 |", markdown)
        self.assertIn("| r2.pdf | document_total_drop | high | documents.markdown_suppressed_blocks | 2 | 1 | -1 |", markdown)


if __name__ == "__main__":
    unittest.main()
