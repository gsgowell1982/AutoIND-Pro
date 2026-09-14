from __future__ import annotations

import unittest

from core.ectd_application_identity import assess_application_identity


class EctdApplicationIdentityTests(unittest.TestCase):
    def test_envelope_application_type_supports_application_root_prefix(self) -> None:
        result = assess_application_identity(
            "x202112345",
            sequence_packages=[
                {
                    "application_number": "x202112345",
                    "application_type": "cnapt2",
                    "matched_documents": [{"filename": "index.xml"}],
                }
            ],
        )

        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["application_category"], "new_drug_application")
        self.assertEqual(result["evidence_summary"]["strong_count"], 1)
        self.assertEqual(result["evidence"][0]["evidence_type"], "envelope_application_type")

    def test_explicit_content_type_conflict_requires_review(self) -> None:
        result = assess_application_identity(
            "x202112345",
            sequence_packages=[
                {
                    "application_number": "x202112345",
                    "application_type": "cnapt2",
                }
            ],
            content_signals=[
                {
                    "source_path": "x202112345/0000/m1/cn/00/application-form.pdf",
                    "evidence_type": "application_form_field",
                    "observed_application_type": "cnapt3",
                    "observed_value": "仿制药申请",
                }
            ],
        )

        self.assertEqual(result["status"], "conflict")
        self.assertTrue(result["review_required"])
        self.assertIn("application_type_conflict", result["issue_codes"])

    def test_weak_title_signal_does_not_claim_supported_type(self) -> None:
        result = assess_application_identity(
            "x202112345",
            content_signals=[
                {
                    "source_path": "x202112345/0000/m1/cn/02/title.txt",
                    "evidence_type": "module1_title_signal",
                    "observed_value": "新药申请资料目录",
                }
            ],
        )

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(result["review_required"])
        self.assertEqual(result["evidence_summary"]["weak_count"], 1)

    def test_missing_evidence_is_insufficient(self) -> None:
        result = assess_application_identity("x202112345")

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["evidence"], [])
        self.assertTrue(result["review_required"])

    def test_workbench_projection_keeps_identity_separate_from_structure_findings(self) -> None:
        from api.main import _build_application_identity_projection

        projection = _build_application_identity_projection(
            {
                "application_roots": [
                    {
                        "name": "x202112345",
                        "relative_path": "x202112345",
                        "sequences": [{"name": "0000", "relative_path": "x202112345/0000"}],
                    }
                ]
            },
            {
                "ectd_project_context": {
                    "sequence_packages": [
                        {
                            "application_root_name": "x202112345",
                            "application_number": "x202112345",
                            "application_type": "cnapt2",
                            "sequence_root": "x202112345/0000",
                        }
                    ]
                }
            },
        )

        self.assertTrue(projection["enabled"])
        self.assertEqual(projection["summary"]["supported_count"], 1)
        self.assertEqual(projection["applications"][0]["status"], "supported")

    def test_workbench_projection_accepts_explicit_parsed_content_evidence(self) -> None:
        from api.main import _build_application_identity_projection

        projection = _build_application_identity_projection(
            {"application_roots": [{"name": "x202112345", "relative_path": "x202112345", "sequences": []}]},
            {"ectd_project_context": {"sequence_packages": []}},
            parsed_documents=[
                {
                    "filename": "application-form.pdf",
                    "source_path": "x202112345/0000/m1/cn/00/application-form.pdf",
                    "metadata": {
                        "ectd_application_type_evidence": [
                            {
                                "evidence_type": "application_form_field",
                                "observed_application_type": "cnapt2",
                                "observed_value": "new drug application",
                            }
                        ]
                    },
                }
            ],
        )

        self.assertEqual(projection["applications"][0]["status"], "supported")
        self.assertEqual(projection["applications"][0]["evidence_summary"]["strong_count"], 1)

    def test_workbench_projection_includes_clinical_trial_sequence_semantics(self) -> None:
        from api.main import _build_application_identity_projection

        projection = _build_application_identity_projection(
            {"application_roots": [{"name": "l202112345", "relative_path": "l202112345", "sequences": []}]},
            {
                "ectd_project_context": {
                    "sequence_packages": [
                        {
                            "application_root_name": "l202112345",
                            "application_number": "l202112345",
                            "application_type": "cnapt1",
                            "sequence_number": "0000",
                            "related_sequence_number": "0000",
                            "regulatory_activity_type": "cnrat1",
                            "sequence_type": "cnsqt1",
                        },
                        {
                            "application_root_name": "l202112345",
                            "application_number": "l202112345",
                            "application_type": "cnapt1",
                            "sequence_number": "0001",
                            "related_sequence_number": "0000",
                            "regulatory_activity_type": "cnrat1",
                            "sequence_type": "cnsqt2",
                        },
                    ]
                }
            },
        )

        validation = projection["applications"][0]["sequence_semantic_validation"]
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["covered_example_sequences"], [f"{i:04d}" for i in range(10)])

    def test_workbench_projection_surfaces_ambiguous_sequence_description_for_review(self) -> None:
        from api.main import _build_application_identity_projection

        projection = _build_application_identity_projection(
            {"application_roots": [{"name": "l202112345", "relative_path": "l202112345", "sequences": []}]},
            {
                "ectd_project_context": {
                    "sequence_packages": [
                        {
                            "application_root_name": "l202112345",
                            "application_type": "cnapt1",
                            "sequence_number": "0000",
                            "related_sequence_number": "0000",
                            "regulatory_activity_type": "cnrat1",
                            "sequence_type": "cnsqt1",
                            "sequence_description": "Submission package",
                        }
                    ]
                }
            },
        )

        validation = projection["applications"][0]["sequence_semantic_validation"]
        self.assertTrue(validation["review_required"])
        self.assertTrue(any(item["status"] == "manual_review" for item in projection["findings"]))


if __name__ == "__main__":
    unittest.main()
