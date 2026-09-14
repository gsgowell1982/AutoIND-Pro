from __future__ import annotations

import unittest

from core.ectd_file_reuse_contract import build_ectd_file_reuse_contract


class EctdFileReuseContractTests(unittest.TestCase):
    def test_contract_covers_chinese_reuse_policy_and_ich_appendix_6_operations(self) -> None:
        contract = build_ectd_file_reuse_contract()
        self.assertEqual(contract["schema_version"], "ectd-file-reuse-contract-v1")
        self.assertEqual(contract["china_regional_policy"]["section"], "3.3.4")
        self.assertTrue(contract["china_regional_policy"]["cross_application_reference"]["prohibited"])
        self.assertTrue(contract["reuse_modes"]["same_sequence"]["multiple_leaf_references_allowed"])
        self.assertTrue(contract["reuse_modes"]["prior_sequence_same_application"]["allowed"])
        self.assertEqual(contract["source_references"]["ich_appendix_6_file_reuse"]["pdf_page"], 103)
        self.assertEqual(set(contract["lifecycle_operations"]), {"new", "replace", "append", "delete"})

    def test_contract_makes_modified_file_and_delete_checksum_rules_explicit(self) -> None:
        contract = build_ectd_file_reuse_contract()
        operations = contract["operation_constraints"]
        self.assertTrue(operations["replace"]["modified_file_required"])
        self.assertTrue(operations["append"]["modified_file_required"])
        self.assertTrue(operations["delete"]["modified_file_required"])
        self.assertTrue(operations["delete"]["checksum_must_be_empty"])
        self.assertTrue(operations["replace"]["target_must_be_current_leaf"])


if __name__ == "__main__":
    unittest.main()
