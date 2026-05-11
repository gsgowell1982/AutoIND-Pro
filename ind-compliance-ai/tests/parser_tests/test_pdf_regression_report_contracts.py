from __future__ import annotations

from pathlib import Path
import unittest

from scripts.pdf_regression_report import _doc_record, _render_markdown


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


if __name__ == "__main__":
    unittest.main()
