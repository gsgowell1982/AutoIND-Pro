from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path

from core.ectd_package_intake import inventory_from_zip_bytes
from api.upload_controller import is_directory_upload_path, validate_upload


class EctdPackageIntakeTests(unittest.TestCase):
    def _zip(self, entries: list[tuple[str, bytes]]) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, content in entries:
                archive.writestr(name, content)
        return output.getvalue()

    def test_extracts_safe_zip_into_inventory(self) -> None:
        inventory = inventory_from_zip_bytes(self._zip([
            ("x202112345/0000/index.xml", b"<index />"),
            ("x202112345/0000/m1/cn/cn-regional.xml", b"<regional />"),
        ]))
        self.assertEqual(inventory["application_roots"][0]["name"], "x202112345")
        self.assertEqual(inventory["source_kind"], "zip")
        self.assertIn("x202112345/0000/index.xml", inventory["file_paths"])

    def test_rejects_zip_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            inventory_from_zip_bytes(self._zip([("../escape.txt", b"bad")]))

    def test_rejects_duplicate_normalized_zip_paths(self) -> None:
        with self.assertRaises(ValueError):
            inventory_from_zip_bytes(self._zip([
                ("x202112345/0000/index.xml", b"a"),
                (r"x202112345\\0000\\index.xml", b"b"),
            ]))

    def test_preserves_explicit_empty_directories_from_zip_inventory(self) -> None:
        payload = self._zip([
            ("x202112345/0000/index.xml", b"<index />"),
            ("x202112345/0000/m5/", b""),
        ])
        inventory = inventory_from_zip_bytes(payload)
        self.assertIn("x202112345/0000/m5", inventory["directory_paths"])

    def test_folder_upload_can_carry_unknown_extension_but_single_file_remains_strict(self) -> None:
        self.assertTrue(is_directory_upload_path("x202112345/0000/readme.bin", "readme.bin"))
        validate_upload(Path("readme.bin"), relative_path="x202112345/0000/readme.bin")
        with self.assertRaises(ValueError):
            validate_upload(Path("readme.bin"), relative_path="readme.bin")

if __name__ == "__main__":
    unittest.main()
