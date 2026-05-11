from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ectd_controlled_vocabulary_ingestion import (
    ECTD_CONTROLLED_VOCABULARY_BUNDLE_VERSION,
    build_ectd_controlled_vocabulary_bundle,
    write_ectd_controlled_vocabulary_bundle,
)


def _resolve_cv_attachment_dir() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [path for path in regulations_root.iterdir() if path.is_dir() and "1-2" in path.name]
    if not matches:
        raise unittest.SkipTest("Attachment 1-2 controlled vocabulary directory not found under data/regulations.")
    return matches[0]


class EctdControlledVocabularyIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_dir = _resolve_cv_attachment_dir()
        cls.bundle = build_ectd_controlled_vocabulary_bundle(cls.source_dir)

    def test_build_bundle_extracts_all_required_vocabularies_and_dependency_rows(self) -> None:
        bundle = self.bundle

        self.assertEqual(bundle["schema_version"], ECTD_CONTROLLED_VOCABULARY_BUNDLE_VERSION)
        self.assertEqual(bundle["bundle_id"], "cn_ectd_attachment_1_2")
        self.assertEqual(bundle["source_directory"], str(self.source_dir))
        self.assertEqual(bundle["controlled_vocabulary_count"], 4)

        vocabularies_by_name = {
            item["controlled_vocabulary_name"]: item for item in bundle["controlled_vocabularies"]
        }
        self.assertEqual(
            sorted(vocabularies_by_name),
            [
                "cv-application-type",
                "cv-product-type",
                "cv-regulatory-activity-type",
                "cv-sequence-type",
            ],
        )
        self.assertEqual(vocabularies_by_name["cv-application-type"]["code_count"], 3)
        self.assertEqual(
            vocabularies_by_name["cv-application-type"]["values"],
            ["cnapt1", "cnapt2", "cnapt3"],
        )
        self.assertEqual(vocabularies_by_name["cv-product-type"]["code_count"], 2)
        self.assertEqual(vocabularies_by_name["cv-regulatory-activity-type"]["code_count"], 9)
        self.assertEqual(vocabularies_by_name["cv-sequence-type"]["code_count"], 4)

        dependency_matrix = bundle["dependency_matrix"]
        self.assertEqual(dependency_matrix["matrix_name"], "depend-apt-rat-sqt")
        self.assertEqual(dependency_matrix["row_count"], 54)
        self.assertIn(
            {
                "application_type": "cnapt2",
                "regulatory_activity_type": "cnrat9",
                "sequence_type": "cnsqt4",
            },
            dependency_matrix["rows"],
        )

    def test_build_bundle_surfaces_schema_dependency_truth_without_overclaiming_closure(self) -> None:
        bundle = self.bundle

        self.assertEqual(bundle["schema_status_summary"]["status"], "partial_dependency_closure")
        self.assertIn("xml.xsd", bundle["schema_status_summary"]["missing_dependencies"])

        vocabularies_by_name = {
            item["controlled_vocabulary_name"]: item for item in bundle["controlled_vocabularies"]
        }
        self.assertEqual(
            vocabularies_by_name["cv-application-type"]["schema_dependency_status"],
            "missing_dependencies",
        )
        self.assertIn(
            "xml.xsd",
            vocabularies_by_name["cv-application-type"]["schema_missing_dependencies"],
        )
        self.assertEqual(bundle["dependency_matrix"]["schema_dependency_status"], "ready")
        self.assertEqual(bundle["dependency_matrix"]["schema_missing_dependencies"], [])

    def test_write_bundle_emits_normalized_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            artifact_path = write_ectd_controlled_vocabulary_bundle(
                self.source_dir,
                output_root=output_root,
            )

            self.assertTrue(artifact_path.exists())
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["bundle_id"], "cn_ectd_attachment_1_2")
            self.assertEqual(payload["controlled_vocabulary_count"], 4)
            self.assertEqual(payload["dependency_matrix"]["row_count"], 54)


if __name__ == "__main__":
    unittest.main()
