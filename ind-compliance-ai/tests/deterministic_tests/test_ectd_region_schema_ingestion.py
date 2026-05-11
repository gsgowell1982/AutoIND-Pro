from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ectd_region_schema_ingestion import (
    ECTD_REGION_SCHEMA_BUNDLE_VERSION,
    build_ectd_region_schema_bundle,
    write_ectd_region_schema_bundle,
)


def _resolve_schema_attachment_dir() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [path for path in regulations_root.iterdir() if path.is_dir() and "1-1" in path.name]
    if not matches:
        raise unittest.SkipTest("Attachment 1-1 regional schema directory not found under data/regulations.")
    return matches[0]


class EctdRegionSchemaIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_dir = _resolve_schema_attachment_dir()
        cls.bundle = build_ectd_region_schema_bundle(cls.source_dir)

    def test_build_bundle_extracts_root_envelope_and_content_structure(self) -> None:
        bundle = self.bundle

        self.assertEqual(bundle["schema_version"], ECTD_REGION_SCHEMA_BUNDLE_VERSION)
        self.assertEqual(bundle["bundle_id"], "cn_ectd_attachment_1_1")
        self.assertEqual(bundle["source_directory"], str(self.source_dir))

        schema = bundle["region_schema"]
        self.assertEqual(schema["filename"], "cn-regional-1-0.xsd")
        self.assertEqual(schema["schema_name"], "cn-regional-1-0")
        self.assertEqual(schema["target_namespace"], "cn_ectd")
        self.assertEqual(schema["root_element"]["name"], "cn_ectd")
        self.assertEqual(
            schema["root_element"]["children"],
            [
                {
                    "name": "cn-envelope",
                    "type": "cn-envelope",
                    "min_occurs": "1",
                    "max_occurs": "1",
                },
                {
                    "name": "cn-content",
                    "type": "cn-content",
                    "min_occurs": "1",
                    "max_occurs": "1",
                },
            ],
        )
        self.assertEqual(
            schema["root_attributes"],
            [
                {
                    "name": "schema-version",
                    "type": "inline-restriction",
                    "use": "required",
                    "allowed_values": ["1.0"],
                }
            ],
        )
        self.assertEqual(len(schema["envelope_fields"]), 10)
        self.assertEqual(
            schema["envelope_fields"][:4],
            [
                {"name": "application-id", "type": "xs:string", "min_occurs": "1", "max_occurs": "1"},
                {"name": "application-type", "type": "cn-type", "min_occurs": "1", "max_occurs": "1"},
                {"name": "product-type", "type": "cn-type", "min_occurs": "1", "max_occurs": "1"},
                {"name": "product-number", "type": "xs:string", "min_occurs": "1", "max_occurs": "1"},
            ],
        )
        self.assertEqual(len(schema["content_top_level_elements"]), 13)
        self.assertEqual(schema["content_top_level_elements"][0]["name"], "cn-1-0")
        self.assertEqual(schema["content_top_level_elements"][0]["min_occurs"], "1")
        self.assertEqual(schema["content_top_level_elements"][-1]["name"], "cn-1-12")
        self.assertEqual(
            schema["sequence_contact_fields"],
            [
                {"name": "name", "type": "xs:string", "min_occurs": "1", "max_occurs": "1"},
                {"name": "phone", "type": "xs:string", "min_occurs": "1", "max_occurs": "1"},
                {"name": "email", "type": "xs:string", "min_occurs": "1", "max_occurs": "1"},
            ],
        )

    def test_build_bundle_surfaces_schema_dependency_truth(self) -> None:
        bundle = self.bundle

        schema = bundle["region_schema"]
        self.assertEqual(schema["import_count"], 2)
        self.assertEqual(
            schema["imports"],
            [
                {
                    "namespace": "http://www.w3.org/1999/xlink",
                    "schema_location": "xlink.xsd",
                },
                {
                    "namespace": "http://www.w3.org/XML/1998/namespace",
                    "schema_location": "xml.xsd",
                },
            ],
        )
        self.assertEqual(schema["schema_dependency_status"], "ready")
        self.assertEqual(schema["schema_missing_dependencies"], [])
        self.assertEqual(bundle["schema_status_summary"]["status"], "ready")

    def test_write_bundle_emits_normalized_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            artifact_path = write_ectd_region_schema_bundle(
                self.source_dir,
                output_root=output_root,
            )

            self.assertTrue(artifact_path.exists())
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["bundle_id"], "cn_ectd_attachment_1_1")
            self.assertEqual(payload["region_schema"]["schema_name"], "cn-regional-1-0")
            self.assertEqual(len(payload["region_schema"]["root_attributes"]), 1)
            self.assertEqual(len(payload["region_schema"]["envelope_fields"]), 10)
            self.assertEqual(len(payload["region_schema"]["sequence_contact_fields"]), 3)


if __name__ == "__main__":
    unittest.main()
