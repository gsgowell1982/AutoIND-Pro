from __future__ import annotations

import unittest

from api.main import is_package_support_file


class PackageSupportFileTests(unittest.TestCase):
    def test_ectd_support_files_are_not_document_parse_failures(self) -> None:
        for filename in ("index-md5.txt", "ich-ectd-3-2.dtd", "ectd-2-0.xsl", "schema.xsd"):
            self.assertTrue(is_package_support_file(filename))

    def test_content_documents_are_not_support_files(self) -> None:
        self.assertFalse(is_package_support_file("m5/study.pdf"))
        self.assertFalse(is_package_support_file("m1/cn/cn-regional.xml"))


if __name__ == "__main__":
    unittest.main()
