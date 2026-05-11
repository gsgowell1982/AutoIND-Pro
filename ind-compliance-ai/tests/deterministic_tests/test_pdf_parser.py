from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]

from parsers.parser_registry import parse_file


@unittest.skipIf(pymupdf is None, "PyMuPDF is required for PDF parser tests.")
class PdfParserTests(unittest.TestCase):
    def test_parse_pdf_extracts_embedded_outline_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "bookmarked.pdf"
            document = pymupdf.open()
            for index in range(3):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.set_toc(
                [
                    [1, "Overview", 1],
                    [2, "Details", 2],
                    [1, "Appendix", 3],
                ]
            )
            document.save(pdf_path)
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "bookmarked.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("embedded_outline_count"), 3)
        self.assertEqual(metadata.get("embedded_outline_depth"), 2)

    def test_parse_pdf_extracts_bookmark_multiple_action_count_without_counting_outline_siblings(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "bookmark-actions-base.pdf"
            pdf_path = Path(temp_dir) / "bookmark-actions.pdf"
            document = pymupdf.open()
            for index in range(2):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.set_toc([[1, "Overview", 1], [1, "Details", 2]])
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            outline_xref = int(mutated.get_toc(simple=False)[0][3]["xref"])
            mutated.xref_set_key(
                outline_xref,
                "A",
                "<</S/GoTo/D[4 0 R/XYZ 72 806 0]/Next<</S/Launch/F(run.exe)>>>>",
            )
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "bookmark-actions.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("embedded_outline_count"), 2)
        self.assertTrue(metadata.get("pdf_bookmark_multiple_action_evidence_available"))
        self.assertEqual(metadata.get("pdf_bookmark_multiple_action_count"), 1)
        self.assertEqual(metadata.get("pdf_bookmark_multiple_action_xrefs"), [outline_xref])

    def test_parse_pdf_extracts_invalid_bookmark_without_action_or_destination(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "bookmark-invalid-base.pdf"
            pdf_path = Path(temp_dir) / "bookmark-invalid.pdf"
            document = pymupdf.open()
            document.new_page().insert_text((72, 72), "Page 1")
            document.set_toc([[1, "Unassigned bookmark", 1]])
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            outline_xref = int(mutated.get_toc(simple=False)[0][3]["xref"])
            mutated.xref_set_key(outline_xref, "A", "null")
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "bookmark-invalid.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertTrue(metadata.get("pdf_bookmark_validity_evidence_available"))
        self.assertEqual(metadata.get("pdf_invalid_bookmark_count"), 1)
        self.assertEqual(metadata.get("pdf_invalid_bookmark_xrefs"), [outline_xref])

    def test_parse_pdf_extracts_damaged_bookmark_internal_destinations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "bookmark-damaged-base.pdf"
            pdf_path = Path(temp_dir) / "bookmark-damaged.pdf"
            document = pymupdf.open()
            for index in range(3):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.set_toc(
                [
                    [1, "Valid internal target", 1],
                    [1, "Missing page object target", 2],
                    [1, "Empty destination array", 3],
                    [1, "Damaged direct destination", 1],
                ]
            )
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            outline_rows = mutated.get_toc(simple=False)
            page_xref = mutated.page_xref(0)
            valid_xref = int(outline_rows[0][3]["xref"])
            missing_page_xref = int(outline_rows[1][3]["xref"])
            empty_destination_xref = int(outline_rows[2][3]["xref"])
            direct_destination_xref = int(outline_rows[3][3]["xref"])
            mutated.xref_set_key(valid_xref, "A", f"<</S/GoTo/D[{page_xref} 0 R/XYZ 72 806 0]>>")
            mutated.xref_set_key(missing_page_xref, "A", "<</S/GoTo/D[9999 0 R/XYZ 72 806 0]>>")
            mutated.xref_set_key(empty_destination_xref, "A", "<</S/GoTo/D[]>>")
            mutated.xref_set_key(direct_destination_xref, "A", "null")
            mutated.xref_set_key(direct_destination_xref, "Dest", "[9998 0 R/Fit]")
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "bookmark-damaged.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertTrue(metadata.get("pdf_bookmark_target_integrity_evidence_available"))
        self.assertEqual(metadata.get("pdf_damaged_bookmark_target_count"), 3)
        self.assertEqual(
            metadata.get("pdf_damaged_bookmark_target_xrefs"),
            [missing_page_xref, empty_destination_xref, direct_destination_xref],
        )
        self.assertEqual(
            [item.get("reason") for item in metadata.get("pdf_damaged_bookmark_target_details", [])],
            ["destination_page_object_missing", "destination_array_empty", "destination_page_object_missing"],
        )

    def test_parse_pdf_extracts_non_inherit_zoom_destinations_for_bookmarks_and_links(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "inherit-zoom-base.pdf"
            pdf_path = Path(temp_dir) / "inherit-zoom.pdf"
            document = pymupdf.open()
            for index in range(2):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.set_toc(
                [
                    [1, "Bookmark inherit zero", 1],
                    [1, "Bookmark inherit null", 1],
                    [1, "Bookmark fit", 1],
                    [1, "Bookmark explicit zoom", 1],
                ]
            )
            first_page = document[0]
            first_page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTO,
                    "from": pymupdf.Rect(72, 100, 220, 120),
                    "page": 1,
                    "to": pymupdf.Point(72, 72),
                }
            )
            first_page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTO,
                    "from": pymupdf.Rect(72, 130, 220, 150),
                    "page": 1,
                    "to": pymupdf.Point(72, 72),
                }
            )
            first_page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTO,
                    "from": pymupdf.Rect(72, 160, 220, 180),
                    "page": 1,
                    "to": pymupdf.Point(72, 72),
                }
            )
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            page_xref = mutated.page_xref(1)
            outline_rows = mutated.get_toc(simple=False)
            mutated.xref_set_key(int(outline_rows[0][3]["xref"]), "A", f"<</S/GoTo/D[{page_xref} 0 R/XYZ 72 720 0]>>")
            mutated.xref_set_key(
                int(outline_rows[1][3]["xref"]),
                "A",
                f"<</S/GoTo/D[{page_xref} 0 R/XYZ 72 720 null]>>",
            )
            fit_bookmark_xref = int(outline_rows[2][3]["xref"])
            zoom_bookmark_xref = int(outline_rows[3][3]["xref"])
            mutated.xref_set_key(fit_bookmark_xref, "A", f"<</S/GoTo/D[{page_xref} 0 R/Fit]>>")
            mutated.xref_set_key(zoom_bookmark_xref, "A", f"<</S/GoTo/D[{page_xref} 0 R/XYZ 72 720 1.25]>>")
            links = mutated[0].get_links()
            mutated.xref_set_key(int(links[0]["xref"]), "A", f"<</S/GoTo/D[{page_xref} 0 R/XYZ 72 720 0]>>")
            fit_link_xref = int(links[1]["xref"])
            zoom_link_xref = int(links[2]["xref"])
            mutated.xref_set_key(fit_link_xref, "A", f"<</S/GoTo/D[{page_xref} 0 R/FitH 720]>>")
            mutated.xref_set_key(zoom_link_xref, "A", f"<</S/GoTo/D[{page_xref} 0 R/XYZ 72 720 2]>>")
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertTrue(metadata.get("pdf_bookmark_inherit_zoom_evidence_available"))
        self.assertEqual(metadata.get("pdf_bookmark_destination_count"), 4)
        self.assertEqual(metadata.get("pdf_bookmark_non_inherit_zoom_count"), 2)
        self.assertEqual(metadata.get("pdf_bookmark_non_inherit_zoom_xrefs"), [fit_bookmark_xref, zoom_bookmark_xref])
        self.assertEqual(
            [item.get("reason") for item in metadata.get("pdf_bookmark_non_inherit_zoom_details", [])],
            ["destination_not_xyz", "xyz_zoom_not_inherit"],
        )
        self.assertTrue(metadata.get("pdf_link_inherit_zoom_evidence_available"))
        self.assertEqual(metadata.get("pdf_link_destination_count"), 3)
        self.assertEqual(metadata.get("pdf_link_non_inherit_zoom_count"), 2)
        self.assertEqual(metadata.get("pdf_link_non_inherit_zoom_xrefs"), [fit_link_xref, zoom_link_xref])
        self.assertEqual(
            [item.get("reason") for item in metadata.get("pdf_link_non_inherit_zoom_details", [])],
            ["destination_not_xyz", "xyz_zoom_not_inherit"],
        )

    def test_parse_pdf_extracts_bookmark_action_kinds(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "bookmark-action-kinds-base.pdf"
            pdf_path = Path(temp_dir) / "bookmark-action-kinds.pdf"
            document = pymupdf.open()
            for index in range(4):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.set_toc(
                [
                    [1, "GoTo", 1],
                    [1, "GoToR", 2],
                    [1, "Launch", 3],
                    [1, "Named", 4],
                ]
            )
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            outline_rows = mutated.get_toc(simple=False)
            action_payloads = [
                "<</S/GoTo/D[4 0 R/XYZ 72 806 0]>>",
                "<</S/GoToR/F(relative-target.pdf)/D[0/Fit]>>",
                "<</S/Launch/F(run.exe)>>",
                "<</S/Named/N/NextPage>>",
            ]
            for row, action_payload in zip(outline_rows, action_payloads):
                mutated.xref_set_key(int(row[3]["xref"]), "A", action_payload)
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "bookmark-action-kinds.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertTrue(metadata.get("pdf_bookmark_action_evidence_available"))
        self.assertEqual(
            metadata.get("pdf_bookmark_action_kinds"),
            ["/GoTo", "/GoToR", "/Launch", "/Named"],
        )

    def test_parse_pdf_extracts_bookmark_external_file_targets(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "bookmark-file-targets-base.pdf"
            pdf_path = Path(temp_dir) / "bookmark-file-targets.pdf"
            document = pymupdf.open()
            for index in range(3):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.set_toc(
                [
                    [1, "Relative GoToR", 1],
                    [1, "Absolute GoToR", 2],
                    [1, "Launch", 3],
                ]
            )
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            outline_rows = mutated.get_toc(simple=False)
            action_payloads = [
                "<</S/GoToR/F(../0001/m2/target.pdf)/D[0/Fit]>>",
                "<</S/GoToR/F(C:/submission/x202112345/0001/m2/target.pdf)/D[0/Fit]>>",
                "<</S/Launch/F(run.exe)>>",
            ]
            for row, action_payload in zip(outline_rows, action_payloads):
                mutated.xref_set_key(int(row[3]["xref"]), "A", action_payload)
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "bookmark-file-targets.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertTrue(metadata.get("pdf_bookmark_external_file_target_evidence_available"))
        self.assertEqual(metadata.get("pdf_bookmark_external_file_target_count"), 3)
        self.assertEqual(
            metadata.get("pdf_bookmark_external_file_targets"),
            [
                "../0001/m2/target.pdf",
                "C:/submission/x202112345/0001/m2/target.pdf",
                "run.exe",
            ],
        )

    def test_parse_pdf_extracts_bookmark_uri_targets(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "bookmark-uri-targets-base.pdf"
            pdf_path = Path(temp_dir) / "bookmark-uri-targets.pdf"
            document = pymupdf.open()
            for index in range(3):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.set_toc(
                [
                    [1, "Web", 1],
                    [1, "Email", 2],
                    [1, "Internal", 3],
                ]
            )
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            outline_rows = mutated.get_toc(simple=False)
            action_payloads = [
                "<</S/URI/URI(https://example.com/submission)>>",
                "<</S/URI/URI(mailto:regulatory@example.com)>>",
                "<</S/GoTo/D[4 0 R/XYZ 72 806 0]>>",
            ]
            for row, action_payload in zip(outline_rows, action_payloads):
                mutated.xref_set_key(int(row[3]["xref"]), "A", action_payload)
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "bookmark-uri-targets.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertTrue(metadata.get("pdf_bookmark_uri_target_evidence_available"))
        self.assertEqual(metadata.get("pdf_bookmark_uri_target_count"), 2)
        self.assertEqual(
            metadata.get("pdf_bookmark_uri_targets"),
            [
                "https://example.com/submission",
                "mailto:regulatory@example.com",
            ],
        )

    def test_parse_pdf_extracts_link_annotation_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "linked.pdf"
            document = pymupdf.open()
            document.new_page()
            document.new_page()
            first_page = document[0]
            first_page.insert_text((72, 72), "Jump to page 2")
            first_page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTO,
                    "from": pymupdf.Rect(72, 60, 180, 80),
                    "page": 1,
                    "to": pymupdf.Point(72, 72),
                }
            )
            first_page.insert_text((72, 120), "Jump to target.pdf")
            first_page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTOR,
                    "from": pymupdf.Rect(72, 108, 210, 128),
                    "file": "target.pdf",
                    "page": 0,
                }
            )
            first_page.insert_text((72, 168), "Open example.com")
            first_page.insert_link(
                {
                    "kind": pymupdf.LINK_URI,
                    "from": pymupdf.Rect(72, 156, 210, 176),
                    "uri": "https://example.com",
                }
            )
            document.save(pdf_path)
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "linked.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("link_annotation_count"), 3)
        self.assertEqual(metadata.get("navigational_link_count"), 2)
        self.assertEqual(metadata.get("internal_link_count"), 1)
        self.assertEqual(metadata.get("external_file_link_count"), 1)
        self.assertEqual(metadata.get("external_uri_link_count"), 1)
        self.assertEqual(metadata.get("external_file_link_targets"), ["target.pdf"])

    def test_parse_pdf_extracts_non_link_annotation_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "annotated.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Reviewer note")
            page.add_text_annot((72, 96), "Check this area")
            document.save(pdf_path)
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "annotated.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("non_link_annotation_count"), 1)
        self.assertEqual(metadata.get("non_link_annotation_types"), ["Text"])

    def test_parse_pdf_extracts_embedded_file_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "attached.pdf"
            document = pymupdf.open()
            document.new_page()
            document.embfile_add(
                "payload.txt",
                b"hello world",
                filename="payload.txt",
                ufilename="payload.txt",
                desc="demo",
            )
            document.save(pdf_path)
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "attached.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("embedded_file_count"), 1)
        self.assertEqual(metadata.get("embedded_file_names"), ["payload.txt"])

    def test_parse_pdf_extracts_security_metadata_for_openable_restricted_pdf(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "owner-only.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Openable but restricted")
            document.save(
                pdf_path,
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner123",
                user_pw="",
                permissions=int(pymupdf.PDF_PERM_PRINT),
            )
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "owner-only.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("pdf_format_version"), "PDF 1.7")
        self.assertFalse(metadata.get("pdf_needs_password"))
        self.assertTrue(metadata.get("pdf_openable_without_password"))
        self.assertTrue(metadata.get("pdf_has_security_settings"))
        self.assertTrue(metadata.get("pdf_encryption_scheme"))
        self.assertNotEqual(metadata.get("pdf_security_permissions"), -4)

    def test_parse_pdf_extracts_pdf_format_version_from_header(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "base.pdf"
            pdf_path = Path(temp_dir) / "legacy-version.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Legacy version")
            document.save(base_path)
            document.close()

            data = base_path.read_bytes()
            self.assertTrue(data.startswith(b"%PDF-1.7"))
            pdf_path.write_bytes(data.replace(b"%PDF-1.7", b"%PDF-1.3", 1))

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "legacy-version.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("pdf_format_version"), "PDF 1.3")

    def test_parse_pdf_extracts_linearized_fast_web_access_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            regular_path = Path(temp_dir) / "regular.pdf"
            linearized_path = Path(temp_dir) / "linearized.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Fast Web Access metadata")
            document.save(regular_path)
            document.close()

            regular_bytes = regular_path.read_bytes()
            first_object_marker = regular_bytes.index(b"<<", regular_bytes.index(b" obj"))
            linearized_path.write_bytes(
                regular_bytes[: first_object_marker + 2]
                + b"\n/Linearized 1\n"
                + regular_bytes[first_object_marker + 2 :]
            )

            regular = parse_file(regular_path)
            linearized = parse_file(linearized_path)

        regular_metadata = dict(regular.get("metadata", {}) or {})
        linearized_metadata = dict(linearized.get("metadata", {}) or {})
        self.assertFalse(regular_metadata.get("pdf_is_linearized"))
        self.assertTrue(linearized_metadata.get("pdf_is_linearized"))

    def test_parse_pdf_extracts_font_embedding_metadata(self) -> None:
        font_path = Path(r"C:\Windows\Fonts\arial.ttf")
        if not font_path.exists():
            self.skipTest("Windows Arial font fixture is unavailable.")

        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "font-embedding.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Standard Helvetica text", fontname="helv")
            page.insert_font(fontname="CustomArial", fontfile=str(font_path))
            page.insert_text((72, 100), "Embedded Arial text", fontname="CustomArial")
            document.save(pdf_path)
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        font_records = list(metadata.get("pdf_font_records", []) or [])
        self.assertIsNotNone(metadata.get("pdf_font_count"))
        self.assertGreaterEqual(metadata.get("pdf_font_count"), 2)
        self.assertTrue(metadata.get("pdf_font_embedding_evidence_available"))
        standard_record = next(
            record for record in font_records if record.get("base_font") == "Helvetica"
        )
        embedded_record = next(
            record for record in font_records if "Arial" in str(record.get("base_font") or "")
        )
        self.assertFalse(standard_record.get("is_embedded"))
        self.assertTrue(embedded_record.get("is_embedded"))
        self.assertEqual(embedded_record.get("font_file_kind"), "FontFile2")

    def test_parse_pdf_extracts_initial_view_catalog_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "initial-view.pdf"
            document = pymupdf.open()
            for index in range(2):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.set_toc([[1, "Overview", 1], [1, "Details", 2]])
            catalog = document.pdf_catalog()
            document.xref_set_key(catalog, "PageMode", "/UseOutlines")
            document.xref_set_key(catalog, "PageLayout", "/SinglePage")
            document.xref_set_key(catalog, "OpenAction", "[4 0 R /Fit]")
            document.save(pdf_path)
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "initial-view.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("pdf_initial_view_page_mode"), "/UseOutlines")
        self.assertEqual(metadata.get("pdf_initial_view_page_layout"), "/SinglePage")
        self.assertTrue(metadata.get("pdf_initial_view_open_action_present"))
        self.assertEqual(metadata.get("pdf_initial_view_open_action_kind"), "array")

    def test_parse_pdf_extracts_disallowed_content_markers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "restricted-content.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Restricted content")
            catalog = document.pdf_catalog()
            document.xref_set_key(
                catalog,
                "Names",
                "<</JavaScript<</Names[(DocJS)<</S/JavaScript/JS(alert)>>]>>/RichMedia<</Names[(Media)(Clip)]>>/ThreeD<</Names[(Scene)(Mesh)]>>>>",
            )
            document.save(pdf_path)
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "restricted-content.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(
            metadata.get("pdf_disallowed_content_markers"),
            ["3d", "javascript", "richmedia"],
        )

    def test_parse_pdf_extracts_link_action_kinds(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "link-actions-base.pdf"
            pdf_path = Path(temp_dir) / "link-actions.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "GoTo")
            page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTO,
                    "from": pymupdf.Rect(72, 60, 120, 80),
                    "page": 0,
                    "to": pymupdf.Point(72, 72),
                }
            )
            page.insert_text((72, 120), "GoToR")
            page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTOR,
                    "from": pymupdf.Rect(72, 108, 140, 128),
                    "file": "target.pdf",
                    "page": 0,
                }
            )
            page.insert_text((72, 168), "URI")
            page.insert_link(
                {
                    "kind": pymupdf.LINK_URI,
                    "from": pymupdf.Rect(72, 156, 120, 176),
                    "uri": "https://example.com",
                }
            )
            page.insert_text((72, 216), "Launch")
            page.insert_link(
                {
                    "kind": pymupdf.LINK_LAUNCH,
                    "from": pymupdf.Rect(72, 204, 140, 224),
                    "file": "run.exe",
                }
            )
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            links = mutated[0].get_links()
            mutated.xref_set_key(int(links[0]["xref"]), "A", "<</S/Named/N/NextPage>>")
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "link-actions.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(
            metadata.get("pdf_link_action_kinds"),
            ["/GoToR", "/Launch", "/Named", "/URI"],
        )
        self.assertEqual(metadata.get("pdf_link_action_pages"), [1])
        self.assertEqual(metadata.get("pdf_link_action_page_records")[0]["page"], 1)
        self.assertIn("/URI", metadata.get("pdf_link_action_page_records")[0]["link_action_kinds"])
        uri_records = metadata.get("pdf_uri_link_annotation_records")
        self.assertEqual(len(uri_records), 1)
        self.assertEqual(uri_records[0]["page"], 1)
        self.assertEqual(uri_records[0]["xref"], int(links[2]["xref"]))
        self.assertEqual(uri_records[0]["uri"], "https://example.com")
        self.assertEqual(uri_records[0]["bbox"], [72.0, 156.0, 120.0, 176.0])

    def test_parse_pdf_extracts_broken_link_annotation_count(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "broken-link-base.pdf"
            pdf_path = Path(temp_dir) / "broken-link.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Broken link")
            page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTO,
                    "from": pymupdf.Rect(72, 60, 140, 80),
                    "page": 0,
                    "to": pymupdf.Point(72, 72),
                }
            )
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            links = mutated[0].get_links()
            mutated.xref_set_key(int(links[0]["xref"]), "A", "null")
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "broken-link.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("pdf_broken_link_annotation_count"), 1)

    def test_parse_pdf_extracts_link_multiple_action_count(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "multiple-action-link-base.pdf"
            pdf_path = Path(temp_dir) / "multiple-action-link.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Multiple action link")
            page.insert_link(
                {
                    "kind": pymupdf.LINK_URI,
                    "from": pymupdf.Rect(72, 60, 180, 80),
                    "uri": "https://example.com",
                }
            )
            document.save(base_path)
            document.close()

            mutated = pymupdf.open(base_path)
            links = mutated[0].get_links()
            link_xref = int(links[0]["xref"])
            mutated.xref_set_key(
                link_xref,
                "A",
                "<</S/URI/URI(https://example.com)/Next<</S/Launch/F(run.exe)>>>>",
            )
            mutated.save(pdf_path)
            mutated.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "multiple-action-link.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(metadata.get("pdf_link_multiple_action_count"), 1)
        self.assertEqual(metadata.get("pdf_link_multiple_action_xrefs"), [link_xref])

    def test_parse_pdf_returns_minimal_result_for_password_protected_pdf(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "password-protected.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Password protected")
            document.save(
                pdf_path,
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner123",
                user_pw="user123",
                permissions=int(pymupdf.PDF_PERM_PRINT),
            )
            document.close()

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "password-protected.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(parsed.get("text"), "")
        self.assertEqual(parsed.get("content_evidence"), [])
        self.assertEqual(parsed.get("content_units"), [])
        self.assertTrue(metadata.get("pdf_is_encrypted"))
        self.assertTrue(metadata.get("pdf_needs_password"))
        self.assertFalse(metadata.get("pdf_openable_without_password"))
        self.assertTrue(metadata.get("pdf_has_security_settings"))
        self.assertEqual(metadata.get("parser_hint"), "pdf-password-protected")

    def test_parse_pdf_returns_minimal_result_for_unreadable_pdf(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "broken.pdf"
            pdf_path.write_bytes(b"not-a-pdf")

            parsed = parse_file(pdf_path)

        metadata = dict(parsed.get("metadata", {}) or {})
        self.assertEqual(parsed.get("filename"), "broken.pdf")
        self.assertEqual(parsed.get("source_type"), "pdf")
        self.assertEqual(parsed.get("text"), "")
        self.assertEqual(parsed.get("content_evidence"), [])
        self.assertEqual(parsed.get("content_units"), [])
        self.assertFalse(metadata.get("pdf_is_readable"))
        self.assertEqual(metadata.get("pdf_readability_issue"), "open_failed")
        self.assertEqual(metadata.get("parser_hint"), "pdf-unreadable")
