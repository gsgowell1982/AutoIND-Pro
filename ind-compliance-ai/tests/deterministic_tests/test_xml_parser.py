from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from parsers.parser_registry import parse_file


def _resolve_cv_attachment_dir() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [path for path in regulations_root.iterdir() if path.is_dir() and "1-2" in path.name]
    if not matches:
        raise unittest.SkipTest("Attachment 1-2 controlled vocabulary directory not found under data/regulations.")
    return matches[0]


def _resolve_schema_attachment_dir() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [path for path in regulations_root.iterdir() if path.is_dir() and "1-1" in path.name]
    if not matches:
        raise unittest.SkipTest("Attachment 1-1 regional schema directory not found under data/regulations.")
    return matches[0]


def _find_regulation_file(filename: str) -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = sorted(regulations_root.rglob(filename))
    if not matches:
        raise unittest.SkipTest(f"{filename} not found under data/regulations.")
    return matches[0]


def _copy_cn_regional_schema_set(destination_dir: Path, *, include_imports: bool = True) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_find_regulation_file("cn-regional-1-0.xsd"), destination_dir / "cn-regional-1-0.xsd")
    if include_imports:
        shutil.copyfile(_find_regulation_file("xlink.xsd"), destination_dir / "xlink.xsd")
        shutil.copyfile(_find_regulation_file("xml.xsd"), destination_dir / "xml.xsd")


def _minimal_valid_cn_regional_payload() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns="cn_ectd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xlink="http://www.w3.org/1999/xlink" schema-version="1.0" xsi:schemaLocation="cn_ectd util/dtd/cn-regional-1-0.xsd">
  <cn-envelope>
    <application-id>x202112345</application-id>
    <application-type code="cnapt1"/>
    <product-type code="cnpt1"/>
    <product-number>1</product-number>
    <related-sequence>0000</related-sequence>
    <regulatory-activity-type code="cnrat1"/>
    <sequence-number>0001</sequence-number>
    <sequence-type code="cnsqt1"/>
    <sequence-description>initial</sequence-description>
    <sequence-contact><name>A</name><phone>1</phone><email>a@example.com</email></sequence-contact>
  </cn-envelope>
  <cn-content>
    <cn-1-0><leaf ID="l1" operation="new" xlink:href="m1/cn/doc.pdf" checksum-type="MD5" checksum="abc"><title>Doc</title></leaf></cn-1-0>
  </cn-content>
</cn_ectd>
"""


class XmlParserTests(unittest.TestCase):
    def test_parse_xml_extracts_ectd_controlled_vocabulary_values(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<vocabulary>
  <item code="new-drug-application" />
  <item code="chemical" />
</vocabulary>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cv-application-type.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        self.assertEqual(parsed.get("filename"), "cv-application-type.xml")
        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_controlled_vocabulary_name"),
            "cv-application-type",
        )
        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_controlled_vocabulary_values"),
            ["new-drug-application", "chemical"],
        )

    def test_parse_real_official_cv_application_type_xml_extracts_precise_entries_and_schema_audit(self) -> None:
        cv_dir = _resolve_cv_attachment_dir()
        xml_path = cv_dir / "cv-application-type.xml"

        parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("ectd_controlled_vocabulary_name"), "cv-application-type")
        self.assertEqual(
            metadata.get("ectd_controlled_vocabulary_values"),
            ["cnapt1", "cnapt2", "cnapt3"],
        )
        self.assertEqual(metadata.get("ectd_controlled_vocabulary_version"), "1.0")
        self.assertEqual(metadata.get("ectd_controlled_vocabulary_valid_from"), "2021-9-1")
        self.assertEqual(metadata.get("ectd_controlled_vocabulary_code_count"), 3)
        self.assertEqual(
            metadata.get("ectd_controlled_vocabulary_entries"),
            [
                {
                    "code": "cnapt1",
                    "descriptions": {
                        "zh": "临床试验申请",
                        "en": "Investigational New Drug",
                    },
                },
                {
                    "code": "cnapt2",
                    "descriptions": {
                        "zh": "新药申请",
                        "en": "New Drug Application",
                    },
                },
                {
                    "code": "cnapt3",
                    "descriptions": {
                        "zh": "仿制药申请",
                        "en": "Abbreviated New Drug Application",
                    },
                },
            ],
        )
        self.assertEqual(metadata.get("xml_schema_primary_path"), str(cv_dir / "cn-cv.xsd"))
        self.assertEqual(metadata.get("xml_schema_dependency_status"), "ready")
        self.assertEqual(metadata.get("xml_schema_missing_dependencies"), [])
        dependency_files = metadata.get("xml_schema_dependency_files") or []
        self.assertIn(str(cv_dir / "cn-cv.xsd"), dependency_files)
        self.assertTrue(any(path.endswith("附件3-2：w3c标准xml命名规范定义文件\\xml.xsd") for path in dependency_files))

    def test_parse_xml_extracts_controlled_vocabulary_values_from_nested_text_nodes(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<vocabulary>
  <entry><code>supplemental-information</code></entry>
  <entry><code>replace</code></entry>
</vocabulary>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cv-sequence-type.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        self.assertEqual(parsed.get("filename"), "cv-sequence-type.xml")
        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_controlled_vocabulary_name"),
            "cv-sequence-type",
        )
        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_controlled_vocabulary_values"),
            ["supplemental-information", "replace"],
        )

    def test_parse_xml_extracts_ectd_envelope_and_checksum_metadata(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    sequence-number="0003"
    related-sequence="0002"
    sequence-description="适应症为xx的新药上市申请"
  />
  <m1>
    <leaf checksum-type="MD5" checksum="abc123" xlink:href="m1/file-a.pdf" />
    <leaf checksum-type="md5" checksum="def456" xlink:href="m1/file-b.pdf" />
  </m1>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        self.assertEqual(parsed.get("filename"), "cn-regional.xml")
        self.assertEqual(parsed.get("source_type"), "xml")
        self.assertEqual((parsed.get("metadata") or {}).get("parser_hint"), "xml-tree-v1")
        self.assertEqual((parsed.get("metadata") or {}).get("ectd_application_number"), "x202112345")
        self.assertEqual((parsed.get("metadata") or {}).get("ectd_sequence_number"), "0003")
        self.assertEqual((parsed.get("metadata") or {}).get("ectd_related_sequence_number"), "0002")
        self.assertEqual((parsed.get("metadata") or {}).get("ectd_previous_sequence_number"), "0002")
        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_sequence_description"),
            "适应症为xx的新药上市申请",
        )
        self.assertEqual((parsed.get("metadata") or {}).get("ectd_envelope_count"), 1)
        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_envelope_attributes"),
            {
                "application-number": "x202112345",
                "sequence-number": "0003",
                "related-sequence": "0002",
                "sequence-description": "适应症为xx的新药上市申请",
            },
        )
        self.assertEqual((parsed.get("metadata") or {}).get("ectd_checksum_types"), ["MD5", "md5"])
        self.assertEqual((parsed.get("metadata") or {}).get("ectd_leaf_count"), 2)
        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_leaf_hrefs"),
            ["m1/file-a.pdf", "m1/file-b.pdf"],
        )
        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_leaf_records"),
            [
                {
                    "href": "m1/file-a.pdf",
                    "checksum_type": "MD5",
                    "checksum": "abc123",
                    "operation": "",
                    "operation_present": False,
                    "xml_lang": "",
                    "xml_lang_present": False,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m1[1]/leaf[0]",
                    "parent_pointer": "/cn_ectd[0]/m1[1]",
                    "parent_local_tag": "m1",
                },
                {
                    "href": "m1/file-b.pdf",
                    "checksum_type": "md5",
                    "checksum": "def456",
                    "operation": "",
                    "operation_present": False,
                    "xml_lang": "",
                    "xml_lang_present": False,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m1[1]/leaf[1]",
                    "parent_pointer": "/cn_ectd[0]/m1[1]",
                    "parent_local_tag": "m1",
                },
            ],
        )

    def test_parse_xml_extracts_instance_schema_location_records_for_cn_regional(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd
  xmlns="cn_ectd"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="cn_ectd util/dtd/cn-regional-1-0.xsd">
  <cn-envelope>
    <application-id>x202112345</application-id>
  </cn-envelope>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            xml_path = package_root / "cn-regional.xml"
            schema_path = package_root / "util" / "dtd" / "cn-regional-1-0.xsd"
            schema_path.parent.mkdir(parents=True)
            schema_path.write_text("<xs:schema xmlns:xs=\"http://www.w3.org/2001/XMLSchema\"/>", encoding="utf-8")
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("xml_schema_location_raw"), "cn_ectd util/dtd/cn-regional-1-0.xsd")
        self.assertEqual(metadata.get("xml_schema_location_count"), 1)
        self.assertEqual(
            metadata.get("xml_schema_location_records"),
            [
                {
                    "attribute_name": "schemaLocation",
                    "attribute_namespace": "http://www.w3.org/2001/XMLSchema-instance",
                    "namespace": "cn_ectd",
                    "schema_location": "util/dtd/cn-regional-1-0.xsd",
                    "schema_location_normalized": "util/dtd/cn-regional-1-0.xsd",
                    "schema_resolved_path": str(schema_path.resolve(strict=False)),
                    "schema_resolved_path_exists": True,
                    "schema_resolved_filename": "cn-regional-1-0.xsd",
                    "schema_location_is_local_path": True,
                    "schema_location_points_to_util": True,
                }
            ],
        )
        self.assertTrue(metadata.get("xml_schema_locations_all_local"))
        self.assertTrue(metadata.get("xml_schema_locations_all_resolve"))
        self.assertTrue(metadata.get("xml_schema_locations_all_point_to_util"))

    def test_parse_xml_reports_schema_location_that_does_not_point_to_util(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd
  xmlns="cn_ectd"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="cn_ectd cn-regional-1-0.xsd">
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        records = metadata.get("xml_schema_location_records") or []
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["schema_location"], "cn-regional-1-0.xsd")
        self.assertFalse(records[0]["schema_location_points_to_util"])
        self.assertFalse(metadata.get("xml_schema_locations_all_point_to_util"))
        self.assertFalse(metadata.get("xml_schema_locations_all_resolve"))

    def test_parse_xml_validates_cn_regional_against_resolved_local_schema(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cn_regional_path = Path(temp_dir) / "x202112345" / "0001" / "m1" / "cn" / "cn-regional.xml"
            _copy_cn_regional_schema_set(cn_regional_path.parent / "util" / "dtd")
            cn_regional_path.write_text(_minimal_valid_cn_regional_payload(), encoding="utf-8")

            parsed = parse_file(cn_regional_path)

        metadata = parsed.get("metadata") or {}
        self.assertTrue(metadata.get("xml_schema_validation_attempted"))
        self.assertTrue(metadata.get("xml_schema_is_valid"))
        self.assertEqual(metadata.get("xml_schema_validation_error_count"), 0)
        self.assertEqual(metadata.get("xml_schema_validation_errors"), [])
        self.assertEqual(metadata.get("xml_schema_validation_prerequisite_missing"), "")
        self.assertTrue(str(metadata.get("xml_schema_validation_schema_path") or "").endswith("cn-regional-1-0.xsd"))

    def test_parse_xml_reports_cn_regional_schema_validation_errors(self) -> None:
        xml_payload = _minimal_valid_cn_regional_payload().replace("<cn-content>", "<cn-content><unknown/>")
        with TemporaryDirectory() as temp_dir:
            cn_regional_path = Path(temp_dir) / "x202112345" / "0001" / "m1" / "cn" / "cn-regional.xml"
            _copy_cn_regional_schema_set(cn_regional_path.parent / "util" / "dtd")
            cn_regional_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(cn_regional_path)

        metadata = parsed.get("metadata") or {}
        self.assertTrue(metadata.get("xml_schema_validation_attempted"))
        self.assertFalse(metadata.get("xml_schema_is_valid"))
        self.assertGreaterEqual(metadata.get("xml_schema_validation_error_count"), 1)
        self.assertIn("unknown", metadata.get("xml_schema_validation_errors")[0]["message"])

    def test_parse_xml_reports_schema_validation_prerequisite_when_imports_are_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cn_regional_path = Path(temp_dir) / "x202112345" / "0001" / "m1" / "cn" / "cn-regional.xml"
            _copy_cn_regional_schema_set(cn_regional_path.parent / "util" / "dtd", include_imports=False)
            cn_regional_path.write_text(_minimal_valid_cn_regional_payload(), encoding="utf-8")

            parsed = parse_file(cn_regional_path)

        metadata = parsed.get("metadata") or {}
        self.assertFalse(metadata.get("xml_schema_validation_attempted"))
        self.assertEqual(metadata.get("xml_schema_validation_prerequisite_missing"), "schema_compile_exception")
        self.assertIn("xlink", metadata.get("xml_schema_validation_error"))

    def test_parse_xml_extracts_all_leaf_title_records_independent_of_href(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <cn-content>
    <cn-1-0>
      <leaf checksum-type="MD5" checksum="a1" xlink:href="m1/cn/00/cover-letter.pdf">
        <title>Cover Letter</title>
      </leaf>
      <leaf checksum-type="MD5" checksum="a2">
        <title>  No href title  </title>
      </leaf>
      <leaf checksum-type="MD5" checksum="a3" xlink:href="m1/cn/00/blank-title.pdf">
        <title>   </title>
      </leaf>
      <leaf checksum-type="MD5" checksum="a4" xlink:href="m1/cn/00/missing-title.pdf" />
    </cn-1-0>
  </cn-content>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("ectd_leaf_count"), 4)
        self.assertEqual(
            metadata.get("ectd_leaf_hrefs"),
            [
                "m1/cn/00/cover-letter.pdf",
                "m1/cn/00/blank-title.pdf",
                "m1/cn/00/missing-title.pdf",
            ],
        )
        self.assertEqual(
            metadata.get("ectd_leaf_title_records"),
            [
                {
                    "href": "m1/cn/00/cover-letter.pdf",
                    "title_present": True,
                    "raw_title_text": "Cover Letter",
                    "normalized_title_text": "Cover Letter",
                    "leaf_pointer": "/cn_ectd[0]/cn-content[0]/cn-1-0[0]/leaf[0]",
                    "parent_pointer": "/cn_ectd[0]/cn-content[0]/cn-1-0[0]",
                    "parent_local_tag": "cn-1-0",
                },
                {
                    "href": "",
                    "title_present": True,
                    "raw_title_text": "  No href title  ",
                    "normalized_title_text": "No href title",
                    "leaf_pointer": "/cn_ectd[0]/cn-content[0]/cn-1-0[0]/leaf[1]",
                    "parent_pointer": "/cn_ectd[0]/cn-content[0]/cn-1-0[0]",
                    "parent_local_tag": "cn-1-0",
                },
                {
                    "href": "m1/cn/00/blank-title.pdf",
                    "title_present": True,
                    "raw_title_text": "   ",
                    "normalized_title_text": "",
                    "leaf_pointer": "/cn_ectd[0]/cn-content[0]/cn-1-0[0]/leaf[2]",
                    "parent_pointer": "/cn_ectd[0]/cn-content[0]/cn-1-0[0]",
                    "parent_local_tag": "cn-1-0",
                },
                {
                    "href": "m1/cn/00/missing-title.pdf",
                    "title_present": False,
                    "raw_title_text": "",
                    "normalized_title_text": "",
                    "leaf_pointer": "/cn_ectd[0]/cn-content[0]/cn-1-0[0]/leaf[3]",
                    "parent_pointer": "/cn_ectd[0]/cn-content[0]/cn-1-0[0]",
                    "parent_local_tag": "cn-1-0",
                },
            ],
        )

    def test_parse_xml_extracts_cn_envelope_child_field_metadata(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <cn-envelope>
    <application-id>x202112345</application-id>
    <application-type code="cnapt2" version="1.0" />
    <product-type code="cnprt1" version="1.0" />
    <related-sequence>0000</related-sequence>
    <regulatory-activity-type code="cnrat1" version="1.0" />
    <sequence-number>0001</sequence-number>
    <sequence-type code="cnsqt2" version="1.0" />
    <sequence-description>补充提交</sequence-description>
    <sequence-contact>
      <name>张三</name>
      <phone>13600000000</phone>
      <email>a@example.com</email>
    </sequence-contact>
  </cn-envelope>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("ectd_application_number"), "x202112345")
        self.assertEqual(metadata.get("ectd_sequence_number"), "0001")
        self.assertEqual(metadata.get("ectd_related_sequence_number"), "0000")
        self.assertEqual(metadata.get("ectd_previous_sequence_number"), "0000")
        self.assertEqual(metadata.get("ectd_sequence_description"), "补充提交")
        self.assertEqual(
            metadata.get("ectd_envelope_attributes"),
            {
                "application-number": "x202112345",
                "application-type": "cnapt2",
                "product-type": "cnprt1",
                "related-sequence": "0000",
                "regulatory-activity-type": "cnrat1",
                "sequence-number": "0001",
                "sequence-type": "cnsqt2",
                "sequence-description": "补充提交",
            },
        )

    def test_parse_xml_extracts_cn_regional_root_schema_version(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink" schema-version="1.2">
  <cn-envelope>
    <application-id>x202112345</application-id>
    <sequence-number>0002</sequence-number>
  </cn-envelope>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "0002" / "m1" / "cn" / "cn-regional.xml"
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("ectd_schema_version"), "1.2")

    def test_parse_xml_extracts_sequence_directory_number_from_source_path(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    sequence-number="0003"
    related-sequence="0000"
    sequence-description="琛ュ厖鎻愪氦"
  />
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "0003" / "m1" / "cn" / "cn-regional.xml"
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        self.assertEqual((parsed.get("metadata") or {}).get("ectd_sequence_directory_number"), "0003")

    def test_parse_xml_extracts_dependency_matrix_rows(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<dependencies>
  <row application-type="clinical-trial-application" regulatory-activity-type="initial-application" sequence-type="initial-submission" />
  <row application-type="clinical-trial-application" regulatory-activity-type="initial-application" sequence-type="response" />
</dependencies>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "depend-apt-rat-sqt.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("ectd_dependency_matrix_name"), "depend-apt-rat-sqt")
        self.assertEqual(
            metadata.get("ectd_dependency_matrix_rows"),
            [
                {
                    "application_type": "clinical-trial-application",
                    "regulatory_activity_type": "initial-application",
                    "sequence_type": "initial-submission",
                },
                {
                    "application_type": "clinical-trial-application",
                    "regulatory_activity_type": "initial-application",
                    "sequence_type": "response",
                },
            ],
        )

    def test_parse_real_official_dependency_matrix_xml_extracts_all_triplets_and_schema_readiness(self) -> None:
        cv_dir = _resolve_cv_attachment_dir()
        xml_path = cv_dir / "depend-apt-rat-sqt.xml"

        parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        rows = metadata.get("ectd_dependency_matrix_rows") or []
        self.assertEqual(metadata.get("ectd_dependency_matrix_name"), "depend-apt-rat-sqt")
        self.assertEqual(metadata.get("ectd_dependency_matrix_version"), "1.0")
        self.assertEqual(metadata.get("ectd_dependency_matrix_valid_from"), "2021-9-1")
        self.assertEqual(len(rows), 54)
        self.assertIn(
            {
                "application_type": "cnapt1",
                "regulatory_activity_type": "cnrat1",
                "sequence_type": "cnsqt1",
            },
            rows,
        )
        self.assertIn(
            {
                "application_type": "cnapt2",
                "regulatory_activity_type": "cnrat9",
                "sequence_type": "cnsqt4",
            },
            rows,
        )
        self.assertIn(
            {
                "application_type": "cnapt3",
                "regulatory_activity_type": "cnrat8",
                "sequence_type": "cnsqt3",
            },
            rows,
        )
        self.assertEqual(metadata.get("xml_schema_primary_path"), str(cv_dir / "cn-dependency.xsd"))
        self.assertEqual(metadata.get("xml_schema_dependency_status"), "ready")
        self.assertEqual(metadata.get("xml_schema_missing_dependencies"), [])

    def test_parse_real_official_regional_schema_xsd_extracts_structure_and_dependency_audit(self) -> None:
        schema_dir = _resolve_schema_attachment_dir()
        xsd_path = schema_dir / "cn-regional-1-0.xsd"

        parsed = parse_file(xsd_path)

        self.assertEqual(parsed.get("filename"), "cn-regional-1-0.xsd")
        self.assertEqual(parsed.get("source_type"), "xml")
        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("xml_schema_name"), "cn-regional-1-0")
        self.assertEqual(metadata.get("xml_schema_target_namespace"), "cn_ectd")
        self.assertEqual(metadata.get("xml_schema_import_count"), 2)
        self.assertEqual(metadata.get("xml_schema_element_count"), 94)
        self.assertEqual(metadata.get("xml_schema_complex_type_count"), 25)
        self.assertEqual(
            metadata.get("xml_schema_imports"),
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
        self.assertIn("cn_ectd", metadata.get("xml_schema_named_elements") or [])
        self.assertIn("cn-envelope", metadata.get("xml_schema_named_elements") or [])
        self.assertIn("cn-content", metadata.get("xml_schema_named_elements") or [])
        self.assertIn("application-type", metadata.get("xml_schema_named_elements") or [])
        self.assertEqual(metadata.get("xml_schema_primary_path"), str(xsd_path))
        self.assertEqual(metadata.get("xml_schema_dependency_status"), "ready")
        self.assertEqual(metadata.get("xml_schema_missing_dependencies"), [])
        dependency_files = metadata.get("xml_schema_dependency_files") or []
        self.assertIn(str(xsd_path), dependency_files)
        self.assertTrue(any(path.endswith("附件3-1：w3c标准xlink结构定义文件\\xlink.xsd") for path in dependency_files))
        self.assertTrue(any(path.endswith("附件3-2：w3c标准xml命名规范定义文件\\xml.xsd") for path in dependency_files))

    def test_parse_xml_extracts_sequence_contact_metadata(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope
    application-number="x202112345"
    sequence-number="0003"
    related-sequence="0002"
    sequence-description="补充提交"
  >
    <sequence-contact>
      <contact-name>Jane Doe</contact-name>
      <telephone>010-12345678</telephone>
      <email>jane@example.com</email>
    </sequence-contact>
  </envelope>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("ectd_sequence_contact_name"), "Jane Doe")
        self.assertEqual(metadata.get("ectd_sequence_contact_phone"), "010-12345678")
        self.assertEqual(metadata.get("ectd_sequence_contact_email"), "jane@example.com")
        self.assertEqual(
            metadata.get("ectd_sequence_contact"),
            {
                "name": "Jane Doe",
                "phone": "010-12345678",
                "email": "jane@example.com",
            },
        )

    def test_parse_xml_extracts_32r_node_extension_metadata(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <m3-quality>
    <m3-2-body-of-data>
      <m3-2-r-regional-information>
        <node-extension>
          <title>3.2.R.1工艺验证</title>
          <leaf xlink:href="m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf">
            <title>工艺验证</title>
          </leaf>
        </node-extension>
        <node-extension>
          <title>3.2.R.2批记录</title>
          <leaf xlink:href="m3/32-body-data/32r-reg-info/cn32r2/batch-record.pdf">
            <title>批记录</title>
          </leaf>
        </node-extension>
      </m3-2-r-regional-information>
    </m3-2-body-of-data>
  </m3-quality>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertEqual(metadata.get("ectd_32r_extension_count"), 2)
        self.assertEqual(
            metadata.get("ectd_32r_extension_titles"),
            ["3.2.R.1工艺验证", "3.2.R.2批记录"],
        )
        self.assertEqual(
            metadata.get("ectd_32r_extension_leaf_hrefs"),
            [
                "m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf",
                "m3/32-body-data/32r-reg-info/cn32r2/batch-record.pdf",
            ],
        )
        self.assertEqual(
            metadata.get("ectd_node_extension_records"),
            [
                {
                    "extension_title": "3.2.R.1工艺验证",
                    "parent_local_tag": "m3-2-r-regional-information",
                    "parent_pointer": "/cn_ectd[0]/m3-quality[0]/m3-2-body-of-data[0]/m3-2-r-regional-information[0]",
                    "leaf_hrefs": ["m3/32-body-data/32r-reg-info/cn32r1/pro-val.pdf"],
                    "leaf_titles": ["工艺验证"],
                    "is_within_32r_scope": True,
                },
                {
                    "extension_title": "3.2.R.2批记录",
                    "parent_local_tag": "m3-2-r-regional-information",
                    "parent_pointer": "/cn_ectd[0]/m3-quality[0]/m3-2-body-of-data[0]/m3-2-r-regional-information[0]",
                    "leaf_hrefs": ["m3/32-body-data/32r-reg-info/cn32r2/batch-record.pdf"],
                    "leaf_titles": ["批记录"],
                    "is_within_32r_scope": True,
                },
            ],
        )

    def test_parse_xml_extracts_ectd_attribute_records_with_section_context(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <m2-7-3-summary-of-clinical-efficacy indication=" Lung cancer ">
    <leaf checksum-type="MD5" checksum="a1" xlink:href="m2/273/efficacy.pdf" />
  </m2-7-3-summary-of-clinical-efficacy>
  <m3-2-s-drug-substance manufacturer="" substance="API-1">
    <leaf checksum-type="MD5" checksum="a2" xlink:href="m3/32s/api.pdf" />
  </m3-2-s-drug-substance>
</ectd:ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "index.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        element_records = metadata.get("ectd_element_records") or []
        section_record = next(
            record
            for record in element_records
            if record["element_local_tag"] == "m2-7-3-summary-of-clinical-efficacy"
        )
        self.assertEqual(section_record["attribute_names"], ["indication"])
        self.assertIn("m273summaryofclinicalefficacy", section_record["ancestor_normalized_names"])

        attribute_records = metadata.get("ectd_attribute_records") or []
        indication_record = next(
            record
            for record in attribute_records
            if record["attribute_name"] == "indication"
        )
        self.assertEqual(indication_record["raw_value"], " Lung cancer ")
        self.assertEqual(indication_record["normalized_value"], "Lung cancer")
        self.assertTrue(indication_record["has_edge_whitespace"])
        self.assertIn("m273summaryofclinicalefficacy", indication_record["ancestor_normalized_names"])

        manufacturer_record = next(
            record
            for record in attribute_records
            if record["attribute_name"] == "manufacturer"
        )
        self.assertEqual(manufacturer_record["normalized_value"], "")
        self.assertEqual(manufacturer_record["element_local_tag"], "m3-2-s-drug-substance")
        self.assertIn("m32sdrugsubstance", manufacturer_record["ancestor_normalized_names"])

    def test_parse_xml_preserves_index_doctype_and_dtd_validation_status(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE ectd:ectd SYSTEM "util/dtd/ich-ectd-3-2.dtd">
<ectd:ectd xmlns:ectd="http://www.ich.org/ectd" xmlns:xlink="http://www.w3c.org/1999/xlink" dtd-version="3.2" />
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "index.xml"
            dtd_path = Path(temp_dir) / "util" / "dtd" / "ich-ectd-3-2.dtd"
            dtd_path.parent.mkdir(parents=True)
            dtd_path.write_bytes(
                next(
                    (Path(__file__).resolve().parents[2] / "data" / "regulations").rglob("ich-ectd-3-2.dtd")
                ).read_bytes()
            )
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertTrue(metadata.get("xml_is_well_formed"))
        self.assertTrue(metadata.get("xml_doctype_present"))
        self.assertEqual(metadata.get("xml_doctype_name"), "ectd:ectd")
        self.assertEqual(metadata.get("xml_doctype_system_id"), "util/dtd/ich-ectd-3-2.dtd")
        self.assertTrue(metadata.get("xml_dtd_resolved_path_exists"))
        self.assertTrue(metadata.get("xml_dtd_validation_attempted"))
        self.assertTrue(metadata.get("xml_dtd_is_valid"))
        self.assertEqual(metadata.get("xml_dtd_validation_error_count"), 0)

    def test_parse_xml_reports_not_well_formed_payload(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE ectd:ectd SYSTEM "util/dtd/ich-ectd-3-2.dtd">
<ectd:ectd xmlns:ectd="http://www.ich.org/ectd">
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "index.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        metadata = parsed.get("metadata") or {}
        self.assertFalse(metadata.get("xml_is_well_formed"))
        self.assertTrue(metadata.get("xml_doctype_present"))
        self.assertEqual(metadata.get("xml_doctype_system_id"), "util/dtd/ich-ectd-3-2.dtd")
        self.assertIn("no element found", metadata.get("xml_parse_error", ""))

    def test_parse_xml_extracts_leaf_xml_lang_metadata_and_presence(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <m2>
    <leaf checksum-type="MD5" checksum="a1" xlink:href="m2/intro-zh.pdf" xml:lang="zh" />
    <leaf checksum-type="MD5" checksum="a2" xlink:href="m2/intro-empty.pdf" xml:lang="" />
    <leaf checksum-type="MD5" checksum="a3" xlink:href="m2/intro-missing.pdf" />
    <leaf checksum-type="MD5" checksum="a4" xlink:href="m2/intro-en.pdf" xml:lang="en" />
    <leaf checksum-type="MD5" checksum="a5" xlink:href="m2/intro-invalid.pdf" xml:lang="xyz" />
  </m2>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_leaf_records"),
            [
                {
                    "href": "m2/intro-zh.pdf",
                    "checksum_type": "MD5",
                    "checksum": "a1",
                    "operation": "",
                    "operation_present": False,
                    "xml_lang": "zh",
                    "xml_lang_present": True,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m2[0]/leaf[0]",
                    "parent_pointer": "/cn_ectd[0]/m2[0]",
                    "parent_local_tag": "m2",
                },
                {
                    "href": "m2/intro-empty.pdf",
                    "checksum_type": "MD5",
                    "checksum": "a2",
                    "operation": "",
                    "operation_present": False,
                    "xml_lang": "",
                    "xml_lang_present": True,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m2[0]/leaf[1]",
                    "parent_pointer": "/cn_ectd[0]/m2[0]",
                    "parent_local_tag": "m2",
                },
                {
                    "href": "m2/intro-missing.pdf",
                    "checksum_type": "MD5",
                    "checksum": "a3",
                    "operation": "",
                    "operation_present": False,
                    "xml_lang": "",
                    "xml_lang_present": False,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m2[0]/leaf[2]",
                    "parent_pointer": "/cn_ectd[0]/m2[0]",
                    "parent_local_tag": "m2",
                },
                {
                    "href": "m2/intro-en.pdf",
                    "checksum_type": "MD5",
                    "checksum": "a4",
                    "operation": "",
                    "operation_present": False,
                    "xml_lang": "en",
                    "xml_lang_present": True,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m2[0]/leaf[3]",
                    "parent_pointer": "/cn_ectd[0]/m2[0]",
                    "parent_local_tag": "m2",
                },
                {
                    "href": "m2/intro-invalid.pdf",
                    "checksum_type": "MD5",
                    "checksum": "a5",
                    "operation": "",
                    "operation_present": False,
                    "xml_lang": "xyz",
                    "xml_lang_present": True,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m2[0]/leaf[4]",
                    "parent_pointer": "/cn_ectd[0]/m2[0]",
                    "parent_local_tag": "m2",
                },
            ],
        )

    def test_parse_xml_extracts_leaf_operation_metadata_and_presence(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <m2>
    <leaf checksum-type="MD5" checksum="a1" xlink:href="m2/intro-new.pdf" operation="new" />
    <leaf checksum-type="MD5" checksum="a2" xlink:href="m2/intro-replace.pdf" operation="replace" />
    <leaf checksum-type="MD5" checksum="a3" xlink:href="m2/intro-missing.pdf" />
  </m2>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_leaf_records"),
            [
                {
                    "href": "m2/intro-new.pdf",
                    "checksum_type": "MD5",
                    "checksum": "a1",
                    "operation": "new",
                    "operation_present": True,
                    "xml_lang": "",
                    "xml_lang_present": False,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m2[0]/leaf[0]",
                    "parent_pointer": "/cn_ectd[0]/m2[0]",
                    "parent_local_tag": "m2",
                },
                {
                    "href": "m2/intro-replace.pdf",
                    "checksum_type": "MD5",
                    "checksum": "a2",
                    "operation": "replace",
                    "operation_present": True,
                    "xml_lang": "",
                    "xml_lang_present": False,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m2[0]/leaf[1]",
                    "parent_pointer": "/cn_ectd[0]/m2[0]",
                    "parent_local_tag": "m2",
                },
                {
                    "href": "m2/intro-missing.pdf",
                    "checksum_type": "MD5",
                    "checksum": "a3",
                    "operation": "",
                    "operation_present": False,
                    "xml_lang": "",
                    "xml_lang_present": False,
                    "leaf_title": "",
                    "leaf_pointer": "/cn_ectd[0]/m2[0]/leaf[2]",
                    "parent_pointer": "/cn_ectd[0]/m2[0]",
                    "parent_local_tag": "m2",
                },
            ],
        )

    def test_parse_xml_extracts_leaf_lifecycle_records_including_modified_file_and_hrefless_leafs(self) -> None:
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <m1>
    <leaf checksum-type="MD5" checksum="a1" xlink:href="m1/cn/00/new.pdf" operation="new">
      <title>New Leaf</title>
    </leaf>
    <leaf checksum-type="MD5" checksum="a2" xlink:href="m1/cn/00/replace-v2.pdf" operation="replace">
      <title>Replace Leaf</title>
      <modified-file xlink:href="../0000/m1/cn/00/replace-v1.pdf" />
    </leaf>
    <leaf checksum-type="MD5" checksum="a3" operation="delete">
      <title>Delete Leaf</title>
      <modified-file xlink:href="../0000/m1/cn/00/delete-v1.pdf" />
    </leaf>
    <leaf checksum-type="MD5" checksum="a4" operation="append" modified-file="../0000/m1/cn/00/append-v1.pdf">
      <title>Append Leaf</title>
    </leaf>
  </m1>
</cn_ectd>
"""
        with TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "cn-regional.xml"
            xml_path.write_text(xml_payload, encoding="utf-8")

            parsed = parse_file(xml_path)

        self.assertEqual(
            (parsed.get("metadata") or {}).get("ectd_leaf_lifecycle_records"),
            [
                {
                    "href": "m1/cn/00/new.pdf",
                    "href_present": True,
                    "operation": "new",
                    "operation_present": True,
                    "modified_file_href": "",
                    "modified_file_present": False,
                    "modified_file_pointer": "",
                    "leaf_title": "New Leaf",
                    "leaf_pointer": "/cn_ectd[0]/m1[0]/leaf[0]",
                    "parent_pointer": "/cn_ectd[0]/m1[0]",
                    "parent_local_tag": "m1",
                },
                {
                    "href": "m1/cn/00/replace-v2.pdf",
                    "href_present": True,
                    "operation": "replace",
                    "operation_present": True,
                    "modified_file_href": "../0000/m1/cn/00/replace-v1.pdf",
                    "modified_file_present": True,
                    "modified_file_pointer": "/cn_ectd[0]/m1[0]/leaf[1]/modified-file[1]",
                    "leaf_title": "Replace Leaf",
                    "leaf_pointer": "/cn_ectd[0]/m1[0]/leaf[1]",
                    "parent_pointer": "/cn_ectd[0]/m1[0]",
                    "parent_local_tag": "m1",
                },
                {
                    "href": "",
                    "href_present": False,
                    "operation": "delete",
                    "operation_present": True,
                    "modified_file_href": "../0000/m1/cn/00/delete-v1.pdf",
                    "modified_file_present": True,
                    "modified_file_pointer": "/cn_ectd[0]/m1[0]/leaf[2]/modified-file[1]",
                    "leaf_title": "Delete Leaf",
                    "leaf_pointer": "/cn_ectd[0]/m1[0]/leaf[2]",
                    "parent_pointer": "/cn_ectd[0]/m1[0]",
                    "parent_local_tag": "m1",
                },
                {
                    "href": "",
                    "href_present": False,
                    "operation": "append",
                    "operation_present": True,
                    "modified_file_href": "../0000/m1/cn/00/append-v1.pdf",
                    "modified_file_present": True,
                    "modified_file_pointer": "",
                    "leaf_title": "Append Leaf",
                    "leaf_pointer": "/cn_ectd[0]/m1[0]/leaf[3]",
                    "parent_pointer": "/cn_ectd[0]/m1[0]",
                    "parent_local_tag": "m1",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
