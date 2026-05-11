from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ectd_stf_valid_values_ingestion import (
    ECTD_STF_VALID_VALUES_BUNDLE_ID,
    ECTD_STF_VALID_VALUES_BUNDLE_VERSION,
    build_ectd_stf_valid_values_bundle,
    write_ectd_stf_valid_values_bundle,
)


def _resolve_attachment_2_6_dir() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [path for path in regulations_root.iterdir() if path.is_dir() and "2-6" in path.name]
    if not matches:
        raise unittest.SkipTest("Attachment 2-6 directory not found under data/regulations.")
    return matches[0]


class EctdStfValidValuesIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_dir = _resolve_attachment_2_6_dir()
        cls.payload = build_ectd_stf_valid_values_bundle(cls.source_dir)

    def test_build_stf_valid_values_bundle_extracts_groups_and_version(self) -> None:
        self.assertEqual(self.payload["schema_version"], ECTD_STF_VALID_VALUES_BUNDLE_VERSION)
        self.assertEqual(self.payload["bundle_id"], ECTD_STF_VALID_VALUES_BUNDLE_ID)
        self.assertEqual(self.payload["valid_values_file"]["filename"], "valid-values.xml")
        self.assertEqual(self.payload["valid_values_file"]["dtd_version"], "2.2")
        self.assertEqual(self.payload["group_count"], 6)
        self.assertGreaterEqual(self.payload["total_value_count"], 50)

    def test_build_stf_valid_values_bundle_extracts_expected_group_values(self) -> None:
        groups_by_key = {
            (item["element"], item["name"]): item
            for item in self.payload["groups"]
        }
        self.assertEqual(groups_by_key[("category", "species")]["value_count"], 9)
        self.assertEqual(groups_by_key[("category", "route-of-admin")]["value_count"], 8)
        self.assertEqual(groups_by_key[("property", "")]["value_count"], 1)
        self.assertIn(
            {"realm": "ich", "value": "legacy-clinical-study-report"},
            groups_by_key[("file-tag", "")]["values"],
        )

    def test_write_stf_valid_values_bundle_emits_json_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = write_ectd_stf_valid_values_bundle(self.source_dir, output_root=Path(temp_dir))
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["bundle_id"], ECTD_STF_VALID_VALUES_BUNDLE_ID)
            self.assertEqual(payload["valid_values_file"]["filename"], "valid-values.xml")


if __name__ == "__main__":
    unittest.main()
