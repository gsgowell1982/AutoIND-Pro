from __future__ import annotations

import unittest

from parsers.pdf.postprocess_algorithms import (
    _algorithm_inline_formula_candidates,
    _algorithm_line_is_open_ended,
    _algorithm_line_is_opener,
    _algorithm_multiline_formula_candidates,
    _algorithm_same_lane,
    _algorithm_vertical_gap,
    _build_algorithm_markdown_display_text,
    _build_algorithm_projection,
    _dedupe_algorithm_formula_candidates,
    _extract_algorithm_ref,
    _looks_like_algorithm_body_line,
    _looks_like_algorithm_boundary_text,
    _looks_like_algorithm_continuation_seed,
    _looks_like_algorithm_section_heading,
    _looks_like_algorithm_title,
)


class PostprocessAlgorithmTests(unittest.TestCase):
    def test_algorithm_text_classifiers_preserve_title_body_and_boundary_contracts(self) -> None:
        self.assertEqual(
            _extract_algorithm_ref("Algorithm 2. Regularized logistic regression"),
            "Algorithm 2",
        )
        self.assertTrue(_looks_like_algorithm_title("Algorithm 1: Example"))
        self.assertTrue(_looks_like_algorithm_section_heading("3.4. Feature selection"))
        self.assertTrue(_looks_like_algorithm_body_line("5. Update v ( k ) = v ( k - 1 )"))
        self.assertTrue(_looks_like_algorithm_continuation_seed("Output: Feature weights W"))
        self.assertTrue(_looks_like_algorithm_boundary_text("3.4. Feature selection"))
        self.assertTrue(_algorithm_line_is_opener("Initialization: set W"))
        self.assertTrue(_algorithm_line_is_open_ended("Set W ="))
        self.assertFalse(_algorithm_line_is_open_ended("Return W."))

    def test_algorithm_geometry_helpers_preserve_lane_and_gap_contracts(self) -> None:
        seed = {"layout_lane": "left", "bbox": [40, 100, 200, 120]}
        same_lane = {"layout_lane": "left", "bbox": [40, 124, 200, 140]}
        other_lane = {"layout_lane": "right", "bbox": [300, 124, 500, 140]}

        self.assertTrue(_algorithm_same_lane(seed, same_lane))
        self.assertFalse(_algorithm_same_lane(seed, other_lane))
        self.assertEqual(_algorithm_vertical_gap(seed, same_lane), 4.0)
        self.assertEqual(_algorithm_vertical_gap({}, same_lane), 999.0)

    def test_algorithm_projection_and_markdown_display_contracts(self) -> None:
        block = {
            "algorithm_id": "alg_001",
            "page": 4,
            "bbox": [40, 100, 300, 200],
            "algorithm_ref": "Algorithm 1",
            "title": "Algorithm 1. Example",
            "content_text": "Algorithm 1. Example\nSet v ( 0 ) = w ( 0 ) =[ 1,1, … ,1 ] , k¼0 and θ=0.01.",
            "lines": [
                "Algorithm 1. Example",
                "Set v ( 0 ) = w ( 0 ) =[ 1,1, … ,1 ] , k¼0 and θ=0.01.",
            ],
            "line_count": 2,
            "inline_formula_spans": [{"latex_text": r"\theta=0.01"}],
            "continued_from_previous_page": False,
            "continues_to_next_page": True,
        }

        projection = _build_algorithm_projection(block)
        markdown_text = _build_algorithm_markdown_display_text(block)

        self.assertEqual(projection["algorithm_id"], "alg_001")
        self.assertEqual(projection["semantic_role"], "algorithm_pseudocode")
        self.assertTrue(projection["continues_to_next_page"])
        self.assertIn(r"$v^{(0)} = w^{(0)} = [1,1,\ldots,1]$", markdown_text)
        self.assertIn(r"$\theta=0.01$", markdown_text)

    def test_algorithm_formula_candidate_helpers_preserve_latex_contracts(self) -> None:
        candidates = _algorithm_inline_formula_candidates(
            "W r = Algorithm 1 ( S , Y , , , lambda sigma theta ,)"
        )
        multiline = _algorithm_multiline_formula_candidates(
            "5. Update v (( + )\n( k + 1 ) = v ( k ) + α ( k ) d ( k )"
        )
        deduped = _dedupe_algorithm_formula_candidates(
            [("short", "x"), ("longer-source", "y"), ("short", "x")]
        )

        self.assertIn(
            (r"W r = Algorithm 1 ( S , Y , , , λ σ θ ,)", r"W_r = \operatorname{Algorithm 1}(S,Y,\lambda,\sigma,\theta)"),
            candidates,
        )
        self.assertIn(
            (
                "v (( + )\n( k + 1 ) = v ( k ) + α ( k ) d ( k )",
                r"v^{(k+1)} = v^{(k)} + \alpha^{(k)}d^{(k)}",
                0,
            ),
            multiline,
        )
        self.assertEqual(deduped, [("longer-source", "y"), ("short", "x")])


if __name__ == "__main__":
    unittest.main()
