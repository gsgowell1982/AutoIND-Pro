from __future__ import annotations

import unittest

from parsers.pdf.postprocess_references import (
    _is_literature_front_page,
    _is_reference_heading_text,
    _looks_like_equation_false_positive_text,
    _looks_like_author_affiliation_line,
    _looks_like_body_prose_with_citation,
    _looks_like_contact_name_line,
    _looks_like_license_notice_text,
    _looks_like_literature_author_line,
    _looks_like_publication_author_note_text,
    _looks_like_publication_footer_text,
    _looks_like_reference_boundary_heading,
    _looks_like_reference_entry_continuation,
    _looks_like_reference_entry_seed,
    _looks_like_reference_page_boilerplate,
    _mark_reference_entry,
    _mark_reference_heading,
    _match_reference_entry_start,
    _set_text_evidence_role,
)


class PostprocessReferenceTests(unittest.TestCase):
    def test_reference_heading_and_entry_detection_contracts(self) -> None:
        self.assertTrue(_is_reference_heading_text("References"))
        self.assertFalse(_is_reference_heading_text("References and notes for model setup"))
        self.assertEqual(_match_reference_entry_start("[12]. Smith J. Example paper."), ("12", True))
        self.assertEqual(_match_reference_entry_start("29."), ("29", False))
        self.assertTrue(_looks_like_reference_boundary_heading("Publisher's Note"))
        self.assertFalse(_looks_like_reference_boundary_heading("2020. pp. 1476"))
        self.assertTrue(_looks_like_reference_entry_seed("Cui Y, Che W, Liu T, Qin B, Yang Z. A journal article. 2020."))
        self.assertFalse(_looks_like_reference_entry_seed("Supplementary Information"))
        self.assertTrue(_looks_like_reference_entry_continuation("Proceedings of ACL. pp. 1476-1486.", [70, 90, 520, 102], 792.0))
        self.assertFalse(_looks_like_reference_entry_continuation("Page 12 of 14", [70, 20, 160, 30], 792.0))

    def test_reference_markers_mutate_text_blocks_consistently(self) -> None:
        heading = {
            "semantic_role": "reference_entry",
            "reference_number": "3",
            "reference_entry_index": 3,
            "reference_entry_start": True,
            "reference_continuation": False,
            "reference_page_continuation": True,
        }
        entry: dict[str, object] = {}

        _mark_reference_heading(heading)
        _mark_reference_entry(entry, "7", 2, entry_start=False, page_continuation=True)

        self.assertEqual(heading["semantic_role"], "reference_heading")
        self.assertEqual(heading["unit_role"], "section_heading")
        self.assertNotIn("reference_number", heading)
        self.assertEqual(entry["semantic_role"], "reference_entry")
        self.assertEqual(entry["unit_role"], "entry")
        self.assertEqual(entry["reference_number"], "7")
        self.assertEqual(entry["reference_entry_index"], 2)
        self.assertFalse(entry["reference_entry_start"])
        self.assertTrue(entry["reference_continuation"])
        self.assertTrue(entry["reference_page_continuation"])

    def test_literature_front_matter_detection_contracts(self) -> None:
        self.assertTrue(_looks_like_literature_author_line("Smith J, Doe A, Lee K, Chen P"))
        self.assertTrue(_looks_like_body_prose_with_citation("These methods were described previously (Smith et al., 2020)."))
        self.assertTrue(_looks_like_author_affiliation_line("1 Department of Biology, Example University, Example City"))
        self.assertTrue(_looks_like_contact_name_line("Jane Doe"))
        self.assertTrue(_looks_like_publication_author_note_text("*Corresponding author"))
        self.assertTrue(_looks_like_publication_footer_text("1234-567X / Copyright 2024", 750.0, 792.0))
        self.assertTrue(_looks_like_license_notice_text("This article is licensed under Creative Commons"))

    def test_text_evidence_role_and_front_page_contracts(self) -> None:
        evidence = {"content_text": "  Example metadata line  "}
        _set_text_evidence_role(evidence, "author_note", "publication_metadata")

        self.assertEqual(evidence["semantic_role"], "author_note")
        self.assertEqual(evidence["segments"], [{"role": "publication_metadata", "text": "Example metadata line"}])
        self.assertTrue(
            _is_literature_front_page(
                [
                    {"semantic_role": "author_line", "content_text": "Smith J, Doe A, Lee K"},
                    {"semantic_role": "author_affiliation", "content_text": "Department"},
                    {"semantic_role": "text_block", "content_text": "Abstract"},
                ]
            )
        )

    def test_reference_page_boilerplate_filters_top_page_citations(self) -> None:
        self.assertTrue(_looks_like_reference_page_boilerplate("Journal Name (2024) 12:34", [70, 20, 300, 32], 792.0))
        self.assertFalse(_looks_like_reference_page_boilerplate("where x is the model score", [70, 20, 300, 32], 792.0))

    def test_equation_false_positive_filter_preserves_proof_prose_cues(self) -> None:
        self.assertTrue(
            _looks_like_equation_false_positive_text(
                "gradient descent with an initial point x i ( 0 ) != 0, then x * is a global minimizer"
            )
        )


if __name__ == "__main__":
    unittest.main()
