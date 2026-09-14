from __future__ import annotations

import unittest

from core.ectd_sequence_semantics import (
    build_ectd_sequence_semantic_contract,
    validate_ectd_sequence_semantics,
)


class EctdSequenceSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_ectd_sequence_semantic_contract()

    def test_clinical_trial_example_contains_rows_0000_to_0009(self) -> None:
        scenario = self.contract["scenarios"]["clinical_trial_application"]
        self.assertEqual([row["sequence_number"] for row in scenario["rows"]], [f"{i:04d}" for i in range(10)])
        self.assertEqual(scenario["rows"][0]["related_sequence"], "0000")
        self.assertEqual(scenario["rows"][1]["related_sequence"], "0000")
        self.assertEqual(scenario["rows"][2]["regulatory_activity_type_code"], "cnrat2")
        self.assertEqual(scenario["rows"][4]["regulatory_activity_type_code"], "cnrat5")
        self.assertEqual(scenario["rows"][8]["regulatory_activity_type_code"], "cnrat7")

    def test_new_drug_example_contains_table2_rows_0000_to_0008(self) -> None:
        scenario = self.contract["scenarios"]["new_drug_application"]
        self.assertEqual([row["sequence_number"] for row in scenario["rows"]], [f"{i:04d}" for i in range(9)])
        self.assertEqual(scenario["rows"][2]["related_sequence"], "0000")
        self.assertEqual(scenario["rows"][3]["regulatory_activity_type_code"], "cnrat2")
        self.assertEqual(scenario["rows"][6]["regulatory_activity_type_code"], "cnrat6")
        self.assertEqual(scenario["rows"][8]["regulatory_activity_type_code"], "cnrat8")

    def test_new_drug_sequence_outside_table2_range_requires_manual_review(self) -> None:
        result = validate_ectd_sequence_semantics(
            "cnapt2",
            [{"sequence_number": f"{i:04d}"} for i in range(10)],
            contract=self.contract,
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["review_required"])
        self.assertIn("sequence_outside_table_example_range", {item["issue_code"] for item in result["review_items"]})
        self.assertEqual(result["scenario"], "table2_new_drug_application")

    def test_sequence_quality_policy_exposes_description_and_contact_requirements(self) -> None:
        policy = self.contract["sequence_quality_policy"]
        self.assertEqual(policy["description_max_characters"], 120)
        self.assertEqual(policy["required_contact_fields"], ["name", "phone", "email"])
        self.assertTrue(policy["prohibited_description_substitution_review"])

    def test_table3_compatible_combination_is_available_for_generic_application(self) -> None:
        result = validate_ectd_sequence_semantics(
            "cnapt3",
            [{
                "sequence_number": "0000",
                "related_sequence": "0000",
                "regulatory_activity_type": "cnrat1",
                "sequence_type": "cnsqt1",
            }],
            contract=self.contract,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["scenario"], "table3_relationship_examples")

    def test_description_length_and_contact_completeness_are_reported(self) -> None:
        result = validate_ectd_sequence_semantics(
            "cnapt2",
            [{
                "sequence_number": "0000",
                "related_sequence": "0000",
                "regulatory_activity_type": "cnrat1",
                "sequence_type": "cnsqt1",
                "sequence_description": "x" * 121,
                "sequence_contact_name": "Applicant",
                "sequence_contact_phone": "",
                "sequence_contact_email": "bad",
            }],
            contract=self.contract,
        )
        issue_codes = {item["issue_code"] for item in result["findings"]}
        self.assertIn("sequence_description_too_long", issue_codes)
        self.assertIn("sequence_contact_incomplete", issue_codes)
        self.assertIn("sequence_contact_email_invalid", issue_codes)

    def test_clinical_trial_sequence_chain_validates(self) -> None:
        sequences = [
            {"sequence_number": "0000", "related_sequence": "0000", "regulatory_activity_type": "cnrat1", "sequence_type": "cnsqt1", "sequence_description": "Clinical trial application for indication xx"},
            {"sequence_number": "0001", "related_sequence": "0000", "regulatory_activity_type": "cnrat1", "sequence_type": "cnsqt2", "sequence_description": "Response to sequence 0000"},
            {"sequence_number": "0002", "related_sequence": "0002", "regulatory_activity_type": "cnrat2", "sequence_type": "cnsqt1", "sequence_description": "Supplement"},
            {"sequence_number": "0003", "related_sequence": "0002", "regulatory_activity_type": "cnrat2", "sequence_type": "cnsqt2", "sequence_description": "Response to sequence 0002"},
            {"sequence_number": "0004", "related_sequence": "0004", "regulatory_activity_type": "cnrat5", "sequence_type": "cnsqt1", "sequence_description": "New indication and drug combination for xx"},
            {"sequence_number": "0005", "related_sequence": "0005", "regulatory_activity_type": "cnrat2", "sequence_type": "cnsqt1", "sequence_description": "Supplement"},
            {"sequence_number": "0006", "related_sequence": "0005", "regulatory_activity_type": "cnrat2", "sequence_type": "cnsqt2", "sequence_description": "Response to sequence 0005"},
            {"sequence_number": "0007", "related_sequence": "0004", "regulatory_activity_type": "cnrat5", "sequence_type": "cnsqt2", "sequence_description": "Response to sequence 0004"},
            {"sequence_number": "0008", "related_sequence": "0008", "regulatory_activity_type": "cnrat7", "sequence_type": "cnsqt1", "sequence_description": "Development safety update report"},
            {"sequence_number": "0009", "related_sequence": "0009", "regulatory_activity_type": "cnrat7", "sequence_type": "cnsqt1", "sequence_description": "Potential serious safety risk information"},
        ]

        result = validate_ectd_sequence_semantics("cnapt1", sequences, contract=self.contract)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["findings"], [])

    def test_gap_and_related_sequence_mismatch_are_reported(self) -> None:
        result = validate_ectd_sequence_semantics(
            "cnapt1",
            [
                {"sequence_number": "0000", "related_sequence": "0000", "regulatory_activity_type": "cnrat1", "sequence_type": "cnsqt1"},
                {"sequence_number": "0002", "related_sequence": "0000", "regulatory_activity_type": "cnrat2", "sequence_type": "cnsqt1"},
            ],
            contract=self.contract,
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("sequence_history_gap", {item["issue_code"] for item in result["findings"]})
        self.assertIn("related_sequence_mismatch", {item["issue_code"] for item in result["findings"]})

    def test_sequence_above_table_example_is_not_rejected_as_maximum(self) -> None:
        result = validate_ectd_sequence_semantics(
            "cnapt1",
            [
                *[
                    {
                        "sequence_number": row["sequence_number"],
                        "related_sequence": row["related_sequence"],
                        "regulatory_activity_type": row["regulatory_activity_type_code"],
                        "sequence_type": row["sequence_type_code"],
                    }
                    for row in self.contract["scenarios"]["clinical_trial_application"]["rows"]
                ],
                {"sequence_number": "0010", "related_sequence": "0010", "regulatory_activity_type": "cnrat2", "sequence_type": "cnsqt1"},
            ],
            contract=self.contract,
        )

        self.assertNotEqual(result["status"], "fail")
        self.assertTrue(result["example_range_not_exhaustive"])

    def test_description_intent_is_evidence_and_ambiguous_text_requires_review(self) -> None:
        result = validate_ectd_sequence_semantics(
            "cnapt1",
            [
                {
                    "sequence_number": "0000",
                    "related_sequence": "0000",
                    "regulatory_activity_type": "cnrat1",
                    "sequence_type": "cnsqt1",
                    "sequence_description": "Clinical trial application for indication xx",
                },
                {
                    "sequence_number": "0001",
                    "related_sequence": "0000",
                    "regulatory_activity_type": "cnrat1",
                    "sequence_type": "cnsqt2",
                    "sequence_description": "Submission package update",
                },
            ],
            contract=self.contract,
        )

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["review_required"])
        self.assertIn("sequence_description_intent_unconfirmed", {item["issue_code"] for item in result["review_items"]})


if __name__ == "__main__":
    unittest.main()
