from __future__ import annotations

from pathlib import Path
import unittest

from core.material_assessment import build_compliance_result_payload
from core.material_review_contract import build_material_review_contract
from parsers.pdf_parser import parse_pdf
from tests.parser_tests.test_ectd_regression_sample import _resolve_ectd_regression_pdf


def _resolve_gate_samples() -> list[Path]:
    roots = []
    for candidate in (Path(__file__).resolve().parents[3], Path(r"D:\AutoIND-Pro")):
        if candidate not in roots:
            roots.append(candidate)

    def resolve_named_sample(filename: str) -> Path | None:
        for root in roots:
            candidate = root / filename
            if candidate.exists():
                return candidate
        return None

    def resolve_glob_sample(pattern: str) -> Path | None:
        for root in roots:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
        return None

    samples = [
        resolve_named_sample("test-ind.pdf"),
        resolve_named_sample("2-column-tst.pdf"),
        _resolve_ectd_regression_pdf(),
        resolve_named_sample("A-tst.pdf"),
    ]
    missing = [
        name
        for name, path in zip(
            ["test-ind.pdf", "2-column-tst.pdf", "eCTD regression sample", "A-tst.pdf"],
            samples,
            strict=True,
        )
        if path is None
    ]
    if missing:
        raise unittest.SkipTest("Missing real PDF gate sample(s): " + ", ".join(missing))
    return [path for path in samples if path is not None]


class RealPdfContractReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_paths = _resolve_gate_samples()
        cls.parsed_documents = [parse_pdf(path) for path in cls.sample_paths]
        cls.contract = build_material_review_contract(
            cls.parsed_documents,
            generated_at="2026-04-02T00:00:00Z",
        )
        cls.payload = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=cls.parsed_documents,
            consistency_rows=[],
            final_status="completed",
        )

    def test_real_pdf_parse_results_preserve_document_identity_fields(self) -> None:
        for path, parsed_document in zip(self.sample_paths, self.parsed_documents, strict=True):
            self.assertEqual(parsed_document.get("filename"), path.name)
            self.assertEqual(parsed_document.get("source_path"), str(path))
            self.assertEqual(parsed_document.get("source_type"), "pdf")

    def test_real_pdf_contract_preserves_document_identity_and_readiness_counts(self) -> None:
        documents = self.contract["documents"]
        self.assertEqual(self.contract["document_count"], 4)
        self.assertEqual(len(documents), 4)
        self.assertEqual(
            [document.get("filename") for document in documents],
            [path.name for path in self.sample_paths],
        )
        self.assertEqual(
            [document.get("source_path") for document in documents],
            [str(path) for path in self.sample_paths],
        )
        self.assertEqual(self.contract["index_summary"]["navigation_sequence_count"], 4)
        self.assertEqual(self.contract["index_summary"]["diagnostic_count"], 0)

        for document in documents:
            summary = document["summary"]
            self.assertGreater(summary["content_evidence_count"], 0)
            self.assertGreater(summary["content_unit_count"], 0)

    def test_real_pdf_gate_samples_stay_rule_ready_downstream(self) -> None:
        rules_by_id = {item["rule_id"]: item["status"] for item in self.payload["rules"]}

        self.assertEqual(self.payload["summary"]["hard_failures"], 0)
        self.assertEqual(self.payload["summary"]["parsed_documents"], 4)
        self.assertEqual(self.payload["artifacts"]["non_text_asset_gap_count"], 0)
        expected_core_statuses = {
            "HR-PARSE-001": "pass",
            "HR-PARSE-002": "pass",
            "HR-NAV-001": "pass",
            "SR-CTD-002": "pass",
            "SR-ECTD-001": "pass",
            "SR-EVID-001": "pass",
            "SR-STRUCT-001": "pass",
        }
        for rule_id, expected_status in expected_core_statuses.items():
            self.assertEqual(rules_by_id.get(rule_id), expected_status)


if __name__ == "__main__":
    unittest.main()
