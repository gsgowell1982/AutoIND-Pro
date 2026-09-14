from __future__ import annotations

import unittest

from core.ectd_naming_validation import build_ectd_path_naming_contract, parse_application_number, validate_package_naming
from core.ectd_structure_validation import build_ectd_content_file_format_contract, validate_package_structure


VALID = {
    "application_roots": [{"name": "x202112345", "relative_path": "x202112345", "sequences": [{"name": "0000", "relative_path": "x202112345/0000"}]}],
    "directory_paths": ["x202112345", "x202112345/0000", "x202112345/0000/m1", "x202112345/0000/m1/cn", "x202112345/0000/m1/cn/00", "x202112345/0000/m2", "x202112345/0000/m3", "x202112345/0000/m4", "x202112345/0000/m5", "x202112345/0000/util", "x202112345/0000/util/dtd", "x202112345/0000/util/style"],
    "file_paths": ["x202112345/0000/index.xml", "x202112345/0000/index-md5.txt", "x202112345/0000/m1/cn/cn-regional.xml"],
}


class EctdPackageAuditAdapterTests(unittest.TestCase):
    def test_accepts_minimal_application_sequence_shape(self) -> None:
        self.assertEqual(validate_package_structure(VALID), [])
        self.assertEqual(validate_package_naming(VALID), [])

    def test_reports_wrong_sequence_name_and_missing_module(self) -> None:
        inventory = {**VALID, "application_roots": [{"name": "x202112345", "relative_path": "x202112345", "sequences": [{"name": "001", "relative_path": "x202112345/001"}]}], "directory_paths": [p for p in VALID["directory_paths"] if not p.endswith("/m5")], "file_paths": [p.replace("/0000/", "/001/") for p in VALID["file_paths"]]}
        findings = validate_package_structure(inventory) + validate_package_naming(inventory)
        self.assertTrue(any(item["rule_id"] == "HR-ECTD-015" for item in findings))
        self.assertTrue(any(item["rule_id"] == "HR-ECTD-001" for item in findings))

    def test_parses_application_number_category_year_and_serial(self) -> None:
        parsed = parse_application_number("x202112345")
        self.assertEqual(parsed["prefix"], "x")
        self.assertEqual(parsed["category"], "new_drug_application")
        self.assertEqual(parsed["year"], 2021)
        self.assertEqual(parsed["serial"], "12345")
        self.assertTrue(parsed["format_valid"])

    def test_reports_application_number_errors_by_component(self) -> None:
        inventory = {
            **VALID,
            "application_roots": [
                {"name": "z20A11234", "relative_path": "z20A11234", "sequences": VALID["application_roots"][0]["sequences"]}
            ],
        }
        findings = validate_package_naming(inventory)
        issue_codes = {item.get("details", {}).get("issue_code") for item in findings}
        self.assertIn("application_prefix", issue_codes)
        self.assertIn("application_year", issue_codes)
        self.assertIn("application_serial", issue_codes)
        self.assertIn("application_length", issue_codes)
        self.assertTrue(all(item["rule_id"] == "HR-ECTD-002" for item in findings))

    def test_historical_application_year_is_not_required_to_equal_current_year(self) -> None:
        inventory = {**VALID}
        findings = validate_package_naming(inventory)
        self.assertFalse(any(item.get("details", {}).get("issue_code") == "application_year_current" for item in findings))

    def test_future_application_year_requires_manual_review_instead_of_hard_failure(self) -> None:
        from datetime import datetime

        future_year = datetime.now().year + 1
        inventory = {**VALID, "application_roots": [{"name": f"x{future_year}12345", "relative_path": f"x{future_year}12345", "sequences": VALID["application_roots"][0]["sequences"]}]}
        findings = validate_package_naming(inventory)
        future = next(item for item in findings if item.get("details", {}).get("issue_code") == "application_year_future")
        self.assertEqual(future["status"], "human_review")
        self.assertEqual(future["severity"], "warning")
        self.assertFalse(future["blocking"])

    def test_reports_unknown_extension_in_project_inventory(self) -> None:
        inventory = {
            **VALID,
            "file_paths": [*VALID["file_paths"], "x202112345/0000/m3/32-body-data/32p3-manuf/unsupported.bin"],
        }
        findings = validate_package_structure(inventory)
        format_findings = [item for item in findings if item["rule_id"] == "HR-ECTD-019"]
        self.assertEqual(len(format_findings), 1)
        self.assertEqual(format_findings[0]["relative_path"], "x202112345/0000/m3/32-body-data/32p3-manuf/unsupported.bin")
        self.assertEqual(format_findings[0]["details"]["detected_extension"], ".bin")
        provenance = format_findings[0]["details"]["regulatory_provenance"]
        self.assertEqual(provenance["section"], "3.3.1")
        self.assertIn("eCTD", provenance["rule_description"])
        self.assertEqual(provenance["source_filename"], "eCTD技术规范.pdf")

    def test_allows_ectd_support_extensions_under_util(self) -> None:
        inventory = {
            **VALID,
            "file_paths": [
                *VALID["file_paths"],
                "x202112345/0000/util/dtd/ich-ectd-3-2.dtd",
                "x202112345/0000/util/dtd/xlink.xsd",
                "x202112345/0000/util/style/ectd-2-0.xsl",
            ],
        }
        self.assertFalse(any(item["rule_id"] == "HR-ECTD-019" for item in validate_package_structure(inventory)))

    def test_content_format_contract_declares_five_content_types_and_util_support_types(self) -> None:
        contract = build_ectd_content_file_format_contract()
        self.assertEqual(contract["allowed_content_extensions"], [".pdf", ".xml", ".xpt", ".txt", ".xsl"])
        self.assertEqual(contract["util_support_extensions"], [".dtd", ".xsd", ".xml", ".xsl", ".txt"])
        self.assertEqual(contract["source"]["clause"], "3.3.1")

    def test_reports_invalid_character_in_directory_or_file_name_from_inventory(self) -> None:
        inventory = {
            **VALID,
            "directory_paths": [*VALID["directory_paths"], "x202112345/0000/m3/Bad-Dir"],
            "file_paths": [*VALID["file_paths"], "x202112345/0000/m3/Bad-Dir/说明.pdf"],
        }
        findings = validate_package_naming(inventory)
        invalid = [item for item in findings if item["rule_id"] == "HR-ECTD-005"]
        self.assertTrue(all(item["details"]["regulatory_provenance"]["section"] == "3.3.2" for item in invalid))
        self.assertTrue(any(item["relative_path"].endswith("Bad-Dir") for item in invalid))
        self.assertTrue(any(item["relative_path"].endswith("说明.pdf") for item in invalid))

    def test_reports_sequence_relative_path_and_segment_length_limits(self) -> None:
        long_segment = "a" * 65
        long_path = "x202112345/0000/m3/" + ("a" * 185) + ".pdf"
        inventory = {
            **VALID,
            "file_paths": [*VALID["file_paths"], f"x202112345/0000/m3/{long_segment}.pdf", long_path],
        }
        findings = validate_package_naming(inventory)
        details = [item["details"] for item in findings if item["rule_id"] == "HR-ECTD-005"]
        issue_codes = {code for detail in details for code in detail.get("issue_codes", [])}
        self.assertIn("segment_too_long", issue_codes)
        self.assertIn("path_too_long", issue_codes)

    def test_path_naming_contract_records_table5_limits_and_xml_reference_requirement(self) -> None:
        contract = build_ectd_path_naming_contract()
        self.assertEqual(contract["source"]["clause"], "3.3.2")
        self.assertEqual(contract["source"]["table"], "5")
        self.assertEqual(contract["max_sequence_relative_path_length"], 180)
        self.assertEqual(contract["max_name_segment_length"], 64)
        self.assertTrue(contract["xml_skeleton_reference_required"])

    def test_reports_non_scaffold_empty_directory_from_inventory(self) -> None:
        inventory = {
            **VALID,
            "directory_paths": [*VALID["directory_paths"], "x202112345/0000/m3/32-body-data/32p3-manuf/empty-subsection"],
        }
        findings = validate_package_structure(inventory)
        empty = [item for item in findings if item["rule_id"] == "HR-ECTD-021"]
        self.assertEqual(len(empty), 1)
        self.assertEqual(empty[0]["relative_path"], "x202112345/0000/m3/32-body-data/32p3-manuf/empty-subsection")
        self.assertEqual(empty[0]["details"]["regulatory_provenance"]["section"], "3.3.3")

    def test_reports_zero_byte_submitted_file_as_placeholder_candidate(self) -> None:
        inventory = {
            **VALID,
            "files": [
                {"relative_path": "x202112345/0000/m3/32-body-data/32p3-manuf/empty.pdf", "size": 0}
            ],
            "file_paths": [*VALID["file_paths"], "x202112345/0000/m3/32-body-data/32p3-manuf/empty.pdf"],
        }
        findings = validate_package_structure(inventory)
        placeholders = [item for item in findings if item["rule_id"] == "HR-ECTD-022"]
        self.assertEqual(len(placeholders), 1)
        self.assertEqual(placeholders[0]["details"]["reason"], "zero_byte_file")
        self.assertEqual(placeholders[0]["details"]["regulatory_provenance"]["section"], "3.3.3")


if __name__ == "__main__":
    unittest.main()
