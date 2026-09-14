from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.ectd_package_inventory import build_package_inventory


class EctdPackageInventoryTests(unittest.TestCase):
    def test_builds_application_sequence_and_file_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "x202112345" / "0000"
            (root / "m1" / "cn" / "00").mkdir(parents=True)
            (root / "util" / "dtd").mkdir(parents=True)
            (root / "index.xml").write_text("<index />", encoding="utf-8")
            (root / "m1" / "cn" / "cn-regional.xml").write_text("<regional />", encoding="utf-8")
            inventory = build_package_inventory(Path(tmp))

        self.assertEqual(inventory["application_roots"][0]["name"], "x202112345")
        self.assertEqual(inventory["application_roots"][0]["application_category"], "new_drug_application")
        self.assertEqual(inventory["application_roots"][0]["application_year"], 2021)
        self.assertEqual(inventory["application_roots"][0]["application_serial"], "12345")
        self.assertEqual(inventory["application_roots"][0]["sequences"][0]["name"], "0000")
        self.assertIn("x202112345/0000/index.xml", inventory["file_paths"])
        self.assertIn("x202112345/0000/m1/cn/00", inventory["directory_paths"])
        index_file = next(item for item in inventory["files"] if item["relative_path"].endswith("index.xml"))
        self.assertEqual(len(index_file["sha256"]), 64)
        self.assertEqual(len(index_file["md5"]), 32)

    def test_rejects_duplicate_normalized_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                build_package_inventory(
                    root,
                    file_records=[
                        {"relative_path": "x202112345/0000/index.xml", "content": b"a"},
                        {"relative_path": r"x202112345\\0000\\index.xml", "content": b"b"},
                    ],
                )

    def test_includes_explicit_empty_directories(self) -> None:
        inventory = build_package_inventory(
            Path("<zip>"),
            file_records=[{"relative_path": "x202112345/0000/index.xml", "content": b"<index />"}],
            explicit_directory_paths=["x202112345/0000/m5"],
        )
        self.assertIn("x202112345/0000/m5", inventory["directory_paths"])


if __name__ == "__main__":
    unittest.main()
