from __future__ import annotations

import unittest

from core.ectd_pdf_presentation_contract import build_ectd_pdf_presentation_contract
from core.material_assessment import _build_requirement_details


class EctdPdfPresentationContractTests(unittest.TestCase):
    def test_contract_traces_chinese_section_3_4_and_ich_format_specification(self) -> None:
        contract = build_ectd_pdf_presentation_contract()

        self.assertEqual(contract["schema_version"], "ectd-pdf-presentation-contract-v1")
        self.assertEqual(contract["source_references"]["cn_technical_specification"]["section"], "3.4")
        self.assertEqual(contract["source_references"]["cn_technical_specification"]["pdf_page"], 23)
        self.assertEqual(
            contract["source_references"]["ich_submission_formats"]["sections"],
            [f"2.{index}" for index in range(1, 19)],
        )

    def test_contract_exposes_chinese_thresholds_exceptions_and_manual_boundary(self) -> None:
        regional = build_ectd_pdf_presentation_contract()["chinese_regional_requirements"]

        self.assertEqual(regional["long_document_navigation"]["page_threshold_exclusive"], 5)
        self.assertEqual(
            set(regional["long_document_navigation"]["exceptions"]),
            {"foreign_reference_material", "literature_reference", "application_form"},
        )
        self.assertEqual(regional["font_sizes_pt"]["narrative_min"], 12)
        self.assertEqual(regional["font_sizes_pt"]["table_min"], 10.5)
        self.assertEqual(regional["font_sizes_pt"]["toc_min"], 12)
        self.assertEqual(regional["font_sizes_pt"]["footnote_min"], 10.5)
        self.assertTrue(regional["cross_application_hyperlinks"]["prohibited"])
        self.assertEqual(
            regional["font_sizes_pt"]["automation_boundary"],
            "manual_review_when_semantic_role_or_font_evidence_is_unavailable",
        )

    def test_pdf_requirement_details_carry_the_presentation_contract(self) -> None:
        details = _build_requirement_details(
            {
                "requirement_id": "cn_ectd_technical_specification:req_long_pdf_navigation_aids",
                "citation_anchor": "cn_ectd_technical_specification#sec_3_4",
            },
            fallback_requirement_id="cn_ectd_technical_specification:req_long_pdf_navigation_aids",
            fallback_citation_anchor="cn_ectd_technical_specification#sec_3_4",
            match_strength="test",
            matched_documents=[],
        )
        self.assertEqual(
            details["pdf_presentation_contract"]["schema_version"],
            "ectd-pdf-presentation-contract-v1",
        )


if __name__ == "__main__":
    unittest.main()
