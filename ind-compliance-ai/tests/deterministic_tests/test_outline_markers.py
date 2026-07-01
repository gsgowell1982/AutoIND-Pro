from __future__ import annotations

import unittest


class OutlineMarkerTests(unittest.TestCase):
    def test_parses_ind_outline_heading_markers(self) -> None:
        from core.outline_markers import parse_outline_heading

        cases = [
            ("2.6.2.1 Pharmacokinetic Summary", "2.6.2.1", "2.6.2.1", "decimal_numeric", "Pharmacokinetic Summary"),
            ("3.2.S.1 General Information", "3.2.S.1", "3.2.S.1", "ctd_mixed", "General Information"),
            ("III STUDY DESIGN", "III", "III", "roman", "STUDY DESIGN"),
            ("A Inclusion Criteria", "A", "A", "alpha", "Inclusion Criteria"),
            ("APPENDIX B IND TABLE OF CONTENTS", "APPENDIX B", "APPENDIX B", "appendix", "IND TABLE OF CONTENTS"),
            ("Module 1 Administrative Information", "Module 1", "MODULE 1", "module", "Administrative Information"),
        ]

        for text, raw, normalized, marker_kind, title in cases:
            with self.subTest(text=text):
                marker = parse_outline_heading(text)
                self.assertIsNotNone(marker)
                assert marker is not None
                self.assertEqual(marker.raw_marker, raw)
                self.assertEqual(marker.normalized_marker, normalized)
                self.assertEqual(marker.marker_kind, marker_kind)
                self.assertEqual(marker.title, title)

    def test_rejects_weak_list_like_marker_without_title(self) -> None:
        from core.outline_markers import parse_outline_heading

        self.assertIsNone(parse_outline_heading("A"))
        self.assertIsNone(parse_outline_heading("III"))
        self.assertIsNone(parse_outline_heading("1)"))

    def test_classifies_single_level_numbered_body_points_as_body_list_items(self) -> None:
        from core.outline_markers import classify_outline_heading_candidate

        body_point = classify_outline_heading_candidate(
            "1. The applicant should provide a description of the sequence and related submission files.",
            active_parent_marker="4.1",
        )
        self.assertEqual(body_point.role, "body_list_item")
        self.assertEqual(body_point.reason, "single_level_marker_inside_parent_section")

        confirmed_heading = classify_outline_heading_candidate(
            "1. Introduction",
            toc_heading_lookup={"1": [{"title": "Introduction", "depth": 1}]},
            active_parent_marker="4.1",
        )
        self.assertEqual(confirmed_heading.role, "section_heading")
        self.assertEqual(confirmed_heading.reason, "toc_title_match")

    def test_classifies_deep_and_ctd_markers_as_stronger_section_candidates(self) -> None:
        from core.outline_markers import classify_outline_heading_candidate

        for text in ("4.1 Administrative Information", "3.2.S.1 General Information"):
            with self.subTest(text=text):
                candidate = classify_outline_heading_candidate(text, active_parent_marker="4")
                self.assertEqual(candidate.role, "section_heading_candidate")
                self.assertIn(candidate.reason, {"decimal_depth", "ctd_mixed_marker"})

    def test_normalizes_outline_keys_consistently(self) -> None:
        from core.outline_markers import normalize_outline_marker

        self.assertEqual(normalize_outline_marker("01.02.00"), "1.2")
        self.assertEqual(normalize_outline_marker("3.2.s.1"), "3.2.S.1")
        self.assertEqual(normalize_outline_marker("appendix b"), "APPENDIX B")
        self.assertEqual(normalize_outline_marker("module 1"), "MODULE 1")


if __name__ == "__main__":
    unittest.main()
