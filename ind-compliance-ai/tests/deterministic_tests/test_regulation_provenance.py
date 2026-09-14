from __future__ import annotations

import unittest

from core.regulation_provenance import build_requirement_provenance


class RegulationProvenanceTests(unittest.TestCase):
    def test_projects_requirement_to_exact_ectd_clause(self) -> None:
        provenance = build_requirement_provenance(
            {
                "requirement_id": "cn_ectd_technical_specification:req_file_name_character_constraints",
            }
        )
        self.assertEqual(provenance["regulation_id"], "cn_ectd_technical_specification")
        self.assertEqual(provenance["section"], "3.3.2")
        self.assertEqual(provenance["clause_id"], "cn_ectd_technical_specification:sec_3_3_2")
        self.assertEqual(provenance["source_filename"], "eCTD技术规范.pdf")
        self.assertTrue(provenance["rule_description"])
        self.assertTrue(provenance["source_excerpt"])
        self.assertEqual(provenance["source_page"], 20)
        self.assertEqual(provenance["traceability_status"], "exact_clause")

    def test_unknown_requirement_keeps_explicit_fallback_status(self) -> None:
        provenance = build_requirement_provenance(
            {},
            fallback_requirement_id="HR-ECTD-999",
            fallback_citation_anchor="internal#unknown",
        )
        self.assertEqual(provenance["traceability_status"], "fallback_requirement_id_only")
        self.assertEqual(provenance["requirement_id"], "HR-ECTD-999")
        self.assertEqual(provenance["citation_anchor"], "internal#unknown")


if __name__ == "__main__":
    unittest.main()
