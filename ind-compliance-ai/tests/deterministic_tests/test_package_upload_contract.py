from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.main import create_app


class PackageUploadContractTests(unittest.TestCase):
    def test_folder_upload_returns_complete_relative_path_inventory(self) -> None:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/uploads",
            files=[
                ("files", ("index.xml", "<index />", "application/xml")),
                ("files", ("readme.bin", "raw", "application/octet-stream")),
            ],
            data={
                "relative_paths": [
                    "x202112345/0000/index.xml",
                    "x202112345/0000/readme.bin",
                ]
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        inventory = payload["package_inventory"]
        self.assertEqual(inventory["source_kind"], "folder")
        self.assertEqual(
            inventory["file_paths"],
            ["x202112345/0000/index.xml", "x202112345/0000/readme.bin"],
        )
        self.assertEqual(payload["files"][1]["relative_path"], "x202112345/0000/readme.bin")

    def test_folder_upload_preserves_explicit_empty_directories(self) -> None:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/uploads",
            files=[
                ("files", ("index.xml", "<index />", "application/xml")),
            ],
            data={
                "relative_paths": ["x202112345/0000/index.xml"],
                "directory_paths": [
                    "x202112345",
                    "x202112345/0000",
                    "x202112345/0000/m2",
                    "x202112345/0000/m1/cn/00",
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        inventory = response.json()["package_inventory"]
        self.assertIn("x202112345/0000/m2", inventory["directory_paths"])
        self.assertIn("x202112345/0000/m1/cn/00", inventory["directory_paths"])


if __name__ == "__main__":
    unittest.main()
