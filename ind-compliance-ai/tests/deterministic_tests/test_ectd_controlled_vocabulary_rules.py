from __future__ import annotations

import unittest

from core.ectd_controlled_vocabulary_rules import (
    build_ectd_vocabulary_rule_contract,
    validate_ectd_envelope_vocabulary,
)


class EctdControlledVocabularyRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_ectd_vocabulary_rule_contract()

    def test_contract_contains_attachment_1_2_vocabularies_and_field_bindings(self) -> None:
        self.assertEqual(self.contract["bundle_id"], "cn_ectd_attachment_1_2")
        self.assertEqual(self.contract["vocabularies"]["application-type"]["code_count"], 3)
        self.assertEqual(self.contract["vocabularies"]["product-type"]["code_count"], 2)
        self.assertEqual(self.contract["vocabularies"]["regulatory-activity-type"]["code_count"], 9)
        self.assertEqual(
            self.contract["field_bindings"]["application-type"]["vocabulary_name"],
            "cv-application-type",
        )
        self.assertEqual(
            self.contract["application_prefix_mapping"]["x"]["application_type_code"],
            "cnapt2",
        )

    def test_valid_envelope_values_and_dependency_triplet_pass(self) -> None:
        result = validate_ectd_envelope_vocabulary(
            {
                "application-type": "cnapt2",
                "product-type": "cnprt1",
                "regulatory-activity-type": "cnrat1",
                "sequence-type": "cnsqt1",
            },
            contract=self.contract,
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["invalid_fields"], [])
        self.assertTrue(result["type_compatibility"]["valid"])

    def test_invalid_value_and_incompatible_triplet_are_separate_findings(self) -> None:
        result = validate_ectd_envelope_vocabulary(
            {
                "application-type": "cnapt9",
                "product-type": "cnprt1",
                "regulatory-activity-type": "cnrat7",
                "sequence-type": "cnsqt1",
            },
            contract=self.contract,
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("application-type", result["invalid_fields"])
        self.assertFalse(result["type_compatibility"]["valid"])
        self.assertEqual(result["type_compatibility"]["issue_code"], "incompatible_type_triplet")

    def test_contract_records_upload_evidence_locations_for_all_fields(self) -> None:
        bindings = self.contract["field_bindings"]
        self.assertEqual(
            bindings["application-type"]["xml_locations"],
            ["cn-regional.xml/@application-type", "index.xml//application-type"],
        )
        self.assertIn("parsed cn-regional.xml envelope", bindings["product-type"]["evidence_sources"])
        self.assertIn("parsed index.xml envelope", bindings["regulatory-activity-type"]["evidence_sources"])


if __name__ == "__main__":
    unittest.main()
