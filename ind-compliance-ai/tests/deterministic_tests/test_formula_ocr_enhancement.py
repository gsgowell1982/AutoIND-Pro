from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from parsers.pdf.formula_ocr import (
    FormulaOcrCandidate,
    build_formula_latex_enhancement,
    enhance_equation_blocks_with_formula_ocr,
    enhance_inline_formula_spans_with_formula_ocr,
    _extract_paddle_formula_latex,
    _extract_texteller_latex,
    _recognize_with_paddle_formula,
    _recognize_with_texteller_cli,
)
from parsers.pdf.postprocess import _clean_display_equation_text
from parsers.pdf.postprocess import (
    _apply_complexity_text_projection,
    _build_plain_text_inline_formula_spans,
    _looks_like_duplicate_equation_text_block,
    _merge_preferred_inline_formula_spans,
    _suppress_inline_formula_fragment_spans_for_text,
)


class FormulaOcrEnhancementTests(unittest.TestCase):
    def test_duplicate_equation_text_block_accepts_short_identical_formula_with_bbox_overlap(self) -> None:
        text_block = {
            "text": "d w ( x , y ,)= x - y w ,",
            "bbox": [301.91, 500.39, 370.08, 510.25],
        }
        equation_source = {
            "text": "d w ( x , y ,)= x - y w , ( 5 )",
            "bbox": [301.91, 500.39, 553.4, 510.25],
        }

        self.assertTrue(_looks_like_duplicate_equation_text_block(text_block, equation_source))

    def test_complexity_projection_uses_formula_shape_and_semantic_context_not_section_number(self) -> None:
        text_block = {
            "text": "The runtime complexity is O (( nd + Ind ) g ) . +) )",
            "bbox": [10.0, 20.0, 200.0, 30.0],
            "section_context": {
                "outline_index": "5.2",
                "section_title": "Runtime analysis",
            },
        }

        _apply_complexity_text_projection(text_block)
        spans = _build_plain_text_inline_formula_spans(text_block)

        self.assertEqual(
            text_block.get("display_text"),
            "The runtime complexity is $O((nd + Ind)g)$.",
        )
        self.assertEqual(text_block.get("text_projection"), "complexity_inline_math")
        self.assertIn("O((nd + Ind)g)", [span.get("latex_text") for span in spans])

    def test_complexity_projection_keeps_linear_iteration_term_distinct_from_quadratic_shape(self) -> None:
        text_block = {
            "text": "() O Ind respectively. Here, n is number of",
            "bbox": [10.0, 20.0, 200.0, 30.0],
            "section_context": {
                "outline_index": "9.1",
                "section_title": "Runtime analysis",
            },
        }

        _apply_complexity_text_projection(text_block)
        spans = _build_plain_text_inline_formula_spans(text_block)

        self.assertEqual(
            text_block.get("display_text"),
            "$O(Ind)$ respectively. Here, n is number of",
        )
        self.assertIn("O(Ind)", [span.get("latex_text") for span in spans])
        self.assertNotIn("O(I(n^2d + nd))", [span.get("latex_text") for span in spans])

    def test_plain_text_inline_math_spans_cover_uppercase_vector_assignments(self) -> None:
        s_block = {
            "text": "Let S = [ x 1 , ..., x n ] T ∈ R n × d be a training data set containing n",
            "bbox": [10.0, 20.0, 200.0, 30.0],
        }
        y_block = {
            "text": "Y = [ y 1 , ..., y n ] T is the corresponding class labels, and d ≫ n.",
            "bbox": [10.0, 20.0, 200.0, 30.0],
        }

        s_spans = _build_plain_text_inline_formula_spans(s_block)
        y_spans = _build_plain_text_inline_formula_spans(y_block)

        self.assertIn(r"S=[x_1,\ldots,x_n]^T \in R^{n \times d}", [span.get("latex_text") for span in s_spans])
        self.assertIn(r"Y=[y_1,\ldots,y_n]^T", [span.get("latex_text") for span in y_spans])
        self.assertTrue(any(span.get("formula_complexity") == "inline_formula" for span in s_spans))
        self.assertTrue(any(span.get("formula_complexity") == "inline_formula" for span in y_spans))

    def test_plain_text_inline_math_spans_do_not_convert_normal_word_fragments_to_subscripts(self) -> None:
        word_block = {
            "text": "The method compared with the six state-of-the-art methods and Kullback-Leibler di-",
            "bbox": [10.0, 20.0, 300.0, 30.0],
        }
        math_block = {
            "text": "where g is the number of classes, x ij, c ij the j-th element of x i",
            "bbox": [10.0, 40.0, 300.0, 50.0],
        }

        word_spans = _build_plain_text_inline_formula_spans(word_block)
        math_spans = _build_plain_text_inline_formula_spans(math_block)

        self.assertNotIn("s_{ix}", [span.get("latex_text") for span in word_spans])
        self.assertNotIn("d_i", [span.get("latex_text") for span in word_spans])
        self.assertIn("x_{ij}", [span.get("latex_text") for span in math_spans])
        self.assertIn("c_{ij}", [span.get("latex_text") for span in math_spans])
        self.assertIn("x_i", [span.get("latex_text") for span in math_spans])

    def test_merge_keeps_same_symbol_when_two_candidates_come_from_distinct_source_ranges(self) -> None:
        spans = [
            {
                "content": "xi",
                "latex_text": "x_i",
                "formula_complexity": "inline_symbol",
                "bbox": [100.0, 10.0, 120.0, 20.0],
                "_source_start": 18,
                "_source_end": 20,
            },
            {
                "content": "x i",
                "latex_text": "x_i",
                "formula_complexity": "inline_symbol",
                "bbox": [100.0, 10.0, 120.0, 20.0],
                "_source_start": 41,
                "_source_end": 44,
            },
        ]

        merged = _merge_preferred_inline_formula_spans(spans)

        self.assertEqual(["x_i", "x_i"], [span.get("latex_text") for span in merged])
        self.assertEqual(["xi", "x i"], [span.get("content") for span in merged])

    def test_merge_prefers_positioned_symbol_over_duplicate_without_source_range(self) -> None:
        spans = [
            {
                "content": "xi",
                "latex_text": "x_i",
                "formula_complexity": "inline_symbol",
                "bbox": [100.0, 10.0, 120.0, 20.0],
                "_source_start": 18,
                "_source_end": 20,
            },
            {
                "content": "x i",
                "latex_text": "x_i",
                "formula_complexity": "inline_symbol",
                "bbox": [100.0, 10.0, 120.0, 20.0],
            },
        ]

        merged = _merge_preferred_inline_formula_spans(spans)

        self.assertEqual(["xi"], [span.get("content") for span in merged])
        self.assertEqual([(18, 20)], [(span.get("_source_start"), span.get("_source_end")) for span in merged])

    def test_display_equation_text_trims_explanatory_prose_transitions(self) -> None:
        cleaned = _clean_display_equation_text(
            "d ( k ) = - g ( k ) , k = 0, where beta = 1 and g k is the gradient of F v",
            "(13)",
        )

        self.assertIn("d ( k )", cleaned)
        self.assertIn("k = 0", cleaned)
        self.assertNotIn("where", cleaned.lower())
        self.assertNotIn("gradient", cleaned.lower())

    def test_accepts_valid_latex_candidate_as_high_confidence_enhancement(self) -> None:
        equation = {
            "text": "S w = n 1 ∑∑ j = 1 i = 1 x i",
            "equation_label": "(1)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\begin{cases}"
                r"S_w=\frac{1}{n}\sum_{j=1}^{g}\sum_{i=1}^{n_j}"
                r"(x_i^{(j)}-u_j)(x_i^{(j)}-u_j)^T,"
                r"\end{cases}"
            ),
            confidence=0.93,
            source="test_formula_ocr",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertEqual(enhancement["latex_render_policy"], "image_primary_latex_enhancement")
        self.assertEqual(enhancement["latex_source"], "test_formula_ocr")
        self.assertGreaterEqual(enhancement["latex_confidence"], 0.85)
        self.assertIn(r"\frac{1}{n}", str(enhancement["latex_text"]))
        self.assertIn(r"\sum", str(enhancement["latex_text"]))
        self.assertIn("syntax_valid", enhancement["latex_validation"]["signals"])
        self.assertIn("symbol_overlap", enhancement["latex_validation"]["signals"])

    def test_rejects_plain_flattened_text_candidate_and_preserves_image_primary_fallback(self) -> None:
        equation = {
            "text": "S w = n 1 ∑∑ j = 1 i = 1 x i",
            "equation_label": "(1)",
        }
        candidate = FormulaOcrCandidate(
            latex="S w = n 1 j = 1 i = 1 x i",
            confidence=0.98,
            source="test_formula_ocr",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertEqual(enhancement["latex_render_policy"], "image_primary_latex_enhancement")
        self.assertIsNone(enhancement["latex_text"])
        self.assertLess(enhancement["latex_confidence"], 0.85)
        self.assertIn("insufficient_math_structure", enhancement["latex_validation"]["issues"])

    def test_disabled_formula_ocr_preserves_high_confidence_text_layer_latex(self) -> None:
        equation_blocks = [
            {
                "text": "partial partial partial partial F v v v = * 0 = ,",
                "latex_text": r"\frac{\partial F}{\partial v}(v^*) = 0",
                "latex_confidence": 0.88,
                "latex_source": "pdf_text_layer_formula_pattern",
                "latex_candidate_text": "partial partial partial partial F v v v = * 0 = ,",
                "latex_validation": {
                    "accepted": True,
                    "score": 0.88,
                    "signals": ["text_layer_derivative_stationarity_pattern"],
                    "issues": [],
                },
            }
        ]

        with patch.dict("os.environ", {"IND_FORMULA_OCR_ENABLED": ""}, clear=False):
            enhance_equation_blocks_with_formula_ocr(Path("unused.pdf"), equation_blocks)

        equation = equation_blocks[0]
        self.assertEqual(equation["latex_text"], r"\frac{\partial F}{\partial v}(v^*) = 0")
        self.assertEqual(equation["latex_confidence"], 0.88)
        self.assertEqual(equation["latex_source"], "pdf_text_layer_formula_pattern")
        self.assertTrue(equation["latex_validation"]["accepted"])
        self.assertEqual(equation["formula_ocr_enhancement"]["latex_source"], "disabled")
        self.assertFalse(equation["formula_ocr_enhancement"]["latex_validation"]["accepted"])

    def test_rejects_repetitive_hallucinated_candidate_without_high_confidence_flag(self) -> None:
        equation = {
            "text": "S w = n 1 sum j = 1 g sum i = 1 n j x i u j",
            "equation_label": "(1)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\left\{S_w=\frac{1}{n}\sum_{j=1}^{g}\sum_{i=1}^{n_j}"
                r"(x_i^{(j)}-u_j)(x_i^{(j)}-u_j)^T"
                + (r"\sqrt{\sqrt{\sqrt{x}}}" * 12)
            ),
            confidence=1.0,
            source="rapid_latex_ocr",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertIsNone(enhancement["latex_text"])
        self.assertEqual(enhancement["latex_confidence"], 0.0)
        self.assertGreater(enhancement["latex_validation"]["score"], 0.0)
        self.assertIn("repetitive_latex_candidate", enhancement["latex_validation"]["issues"])
        self.assertFalse(enhancement["latex_validation"]["accepted"])

    def test_accepts_project_owned_latex_with_escaped_literal_braces(self) -> None:
        equation = {
            "text": "S w = n 1 sum j = 1 g sum i = 1 n j x i u j",
            "equation_label": "(1)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\left\{ S_w=\frac{1}{n}\sum_{j=1}^{g}\sum_{i=1}^{n_j}"
                r"(x_i^{(j)}-u_j)(x_i^{(j)}-u_j)^T \right. \tag{1}"
            ),
            confidence=0.95,
            source="rapid_latex_ocr",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertEqual(enhancement["latex_source"], "rapid_latex_ocr")
        self.assertGreaterEqual(enhancement["latex_confidence"], 0.85)
        self.assertIn("syntax_valid", enhancement["latex_validation"]["signals"])
        self.assertNotIn("latex_syntax_weak", enhancement["latex_validation"]["issues"])
        self.assertIn(r"\left\{", str(enhancement["latex_text"]))

    def test_normalizes_repeated_latex_spacing_without_rejecting_formula_body(self) -> None:
        equation = {
            "text": "min J w = S w - gamma S b",
            "equation_label": "(6)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\operatorname*{min}_{w}J(w)=S_{w}-\gamma S_{b},"
                + (r"\qquad" * 16)
                + r"(6)"
            ),
            confidence=1.45,
            source="rapid_latex_ocr",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertEqual(enhancement["latex_source"], "rapid_latex_ocr")
        self.assertGreaterEqual(enhancement["latex_confidence"], 0.85)
        latex_text = str(enhancement["latex_text"])
        self.assertIn(r"\operatorname*", latex_text)
        self.assertIn(r"\tag{6}", latex_text)
        self.assertLessEqual(latex_text.count(r"\qquad"), 2)
        self.assertLessEqual(enhancement["latex_validation"]["ocr_confidence"], 1.0)

    def test_trims_repetitive_mfr_tail_and_accepts_remaining_formula_body(self) -> None:
        equation = {
            "text": "min v F v = sum log 1 exp sum v z lambda v 2",
            "equation_label": "(11)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\operatorname*{m i n}_{\nu}F(\nu)="
                r"\sum_{i=1}^{n}\log\left(1+\exp\left(-\sum_j\nu_j^2z_{ij}\right)\right)"
                r"+\lambda\|\nu\|_2^2,"
                + (r"\quad\mathrm{~o~n~}" * 24)
            ),
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertEqual(enhancement["latex_source"], "paddleocr_formula:PP-FormulaNet_plus-M")
        self.assertTrue(enhancement["latex_validation"]["accepted"])
        self.assertGreaterEqual(enhancement["latex_confidence"], 0.85)
        latex_text = str(enhancement["latex_text"])
        self.assertIn(r"\operatorname*", latex_text)
        self.assertIn(r"\operatorname*{min}", latex_text)
        self.assertIn(r"\sum_{i=1}^{n}", latex_text)
        self.assertIn(r"\lambda", latex_text)
        self.assertNotIn(r"\mathrm{~o~n~}", latex_text)
        self.assertIn("trimmed_repetitive_tail", enhancement["latex_validation"]["signals"])

    def test_trims_repetitive_mfr_tail_even_when_later_tail_is_malformed(self) -> None:
        equation = {
            "text": "min v F v = sum log 1 exp sum v z lambda v 2",
            "equation_label": "(11)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\operatorname*{min}_{\boldsymbol{\nu}}F(\boldsymbol{\nu})="
                r"\sum_{i=1}^{n}\log\left(1+\exp\left(-\sum_{j}\nu_{j}^{2}z_{i j}\right)\right)"
                r"+\lambda||\boldsymbol{\nu}||_{2}^{2},"
                + (r"\quad\mathrm{~o~n~}" * 8)
                + r"\quad\mathrm{}\mathrm{~o~n~~}\quad\mathrm{}"
                + r"\mathrm{\mathrm{~o~n~}\quad\mathrm{~"
            ),
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertTrue(enhancement["latex_validation"]["accepted"])
        latex_text = str(enhancement["latex_text"])
        self.assertIn(r"\boldsymbol{\nu}", latex_text)
        self.assertIn(r"\sum_{i=1}^{n}", latex_text)
        self.assertNotIn(r"\mathrm{~o~n~}", latex_text)
        self.assertNotIn("repetitive_latex_candidate", enhancement["latex_validation"]["issues"])

    def test_rejects_short_formula_candidate_with_unexplained_new_symbol_word(self) -> None:
        equation = {
            "text": "partial partial F v v v = * 0 =",
            "equation_label": "",
        }
        candidate = FormulaOcrCandidate(
            latex=r"\begin{array}{r}{\left.\frac{\partial F}{\partial v}\right|_{y=y_{addlo}^{*}}}\end{array}",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertIsNone(enhancement["latex_text"])
        self.assertFalse(enhancement["latex_validation"]["accepted"])
        self.assertIn("unexplained_symbol_tokens", enhancement["latex_validation"]["issues"])

    def test_rejects_short_formula_candidate_with_spaced_unexplained_symbol_word(self) -> None:
        equation = {
            "text": "partial partial F v v v = * 0 =",
            "equation_label": "",
        }
        candidate = FormulaOcrCandidate(
            latex=r"\begin{array}{r}{\left.\frac{\partial F}{\partial v}\right|_{y=y_{a d d l o}^{*}}}\end{array}",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertIsNone(enhancement["latex_text"])
        self.assertFalse(enhancement["latex_validation"]["accepted"])
        self.assertIn("unexplained_symbol_tokens", enhancement["latex_validation"]["issues"])

    def test_normalizes_mfr_formula_tag_to_detected_equation_label(self) -> None:
        equation = {
            "text": "v k + 1 = v k + alpha k d k",
            "equation_label": "(12)",
        }
        candidate = FormulaOcrCandidate(
            latex=r"\nu^{(k+1)}=\nu^{(k)}+\alpha^{(k)}d^{(k)},k=0,1,\ldots,\tag{1}",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertTrue(enhancement["latex_validation"]["accepted"])
        self.assertIn(r"\tag{12}", str(enhancement["latex_text"]))
        self.assertNotIn(r"\tag{1}", str(enhancement["latex_text"]))
        self.assertIn("normalized_formula_tag", enhancement["latex_validation"]["signals"])

    def test_extracts_texteller_cli_latex_from_fenced_stdout(self) -> None:
        stdout = "Predicted LaTeX: ```\n\\[c_j=\\frac{1}{n_j}\\sum_i x_i\\]\n```"

        latex = _extract_texteller_latex(stdout)

        self.assertEqual(latex, r"\[c_j=\frac{1}{n_j}\sum_i x_i\]")

    def test_texteller_cli_backend_returns_candidate_when_command_succeeds(self) -> None:
        completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": "Predicted LaTeX: ```\n\\[x_i=y_i\\]\n```",
                "stderr": "",
            },
        )()
        with patch("parsers.pdf.formula_ocr._texteller_cli_path", return_value="texteller.exe"), patch(
            "parsers.pdf.formula_ocr.subprocess.run",
            return_value=completed,
        ) as run_mock:
            candidate = _recognize_with_texteller_cli(b"fake-png")

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.source, "texteller_cli")
        self.assertEqual(candidate.confidence, 0.95)
        self.assertEqual(candidate.latex, r"\[x_i=y_i\]")
        self.assertIn("inference", run_mock.call_args.args[0])

    def test_extracts_paddle_formula_latex_from_nested_result(self) -> None:
        latex = _extract_paddle_formula_latex(
            [
                {
                    "res": {
                        "rec_formula": r"d_w(x,y)=\left\|x-y\right\|_w,\tag{5}",
                    }
                }
            ]
        )

        self.assertEqual(latex, r"d_w(x,y)=\left\|x-y\right\|_w,\tag{5}")

    def test_paddle_formula_backend_returns_candidate_when_model_predicts_formula(self) -> None:
        class FakeModel:
            def predict(self, *, input, batch_size):
                self.input = input
                self.batch_size = batch_size
                return [{"res": {"rec_formula": r"S_w=\frac{1}{n}\sum_i x_i"}}]

        fake_model = FakeModel()
        with patch("parsers.pdf.formula_ocr._load_paddle_formula_model", return_value=fake_model):
            candidate = _recognize_with_paddle_formula(b"fake-png")

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.source, "paddleocr_formula:PP-FormulaNet_plus-M")
        self.assertEqual(candidate.confidence, 0.96)
        self.assertEqual(candidate.latex, r"S_w=\frac{1}{n}\sum_i x_i")
        self.assertEqual(fake_model.batch_size, 1)

    def test_accepts_concise_texteller_formula_when_validation_is_just_below_default_threshold(self) -> None:
        equation = {
            "text": "d w ( x , y ,)= x - y w , ( 5 )",
            "equation_label": "(5)",
        }
        candidate = FormulaOcrCandidate(
            latex=r"d_w(x,y)=\left\|x-y\right\|_w,\tag{5}",
            confidence=0.95,
            source="texteller_cli",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertEqual(enhancement["latex_source"], "texteller_cli")
        self.assertGreaterEqual(enhancement["latex_confidence"], 0.8)
        self.assertIn(r"d_w", str(enhancement["latex_text"]))
        self.assertIn(r"\tag{5}", str(enhancement["latex_text"]))
        self.assertTrue(enhancement["latex_validation"]["accepted"])

    def test_rejects_paddle_formula_candidate_polluted_by_explanatory_prose(self) -> None:
        equation = {
            "text": "d k = - g k k = 0",
            "equation_label": "(13)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\begin{aligned}&obtained by carrying out some line search. "
                r"The direction d^{(k)} is,\\&defined by:\\"
                r"&d^{(k)}=-g^{(k)}, k=0\tag{13}\end{aligned}"
            ),
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertIsNone(enhancement["latex_text"])
        self.assertFalse(enhancement["latex_validation"]["accepted"])
        self.assertIn("explanatory_prose_in_formula_candidate", enhancement["latex_validation"]["issues"])

    def test_rejects_paddle_formula_candidate_with_mathrm_wrapped_prose(self) -> None:
        equation = {
            "text": "d k = - g k k = 0",
            "equation_label": "(13)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\begin{array}{rl}&{\mathrm{obtained~by~carrying~out~some~line~search.}}\\"
                r"&{\mathrm{The~direction~}d^{(k)}}\\"
                r"&d^{(k)}=-g^{(k)},k=0\tag{13}\end{array}"
            ),
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertIsNone(enhancement["latex_text"])
        self.assertIn("explanatory_prose_in_formula_candidate", enhancement["latex_validation"]["issues"])

    def test_rejects_paddle_formula_candidate_with_spaced_letter_prose(self) -> None:
        equation = {
            "text": "d k = - g k k = 0",
            "equation_label": "(13)",
        }
        candidate = FormulaOcrCandidate(
            latex=(
                r"\begin{array}{rl}&{\mathrm{p t a i n e d~b y~c a r r y i n g~o u t~"
                r"s o m e~l i n e~s e a r c h.~T h e~d i r e c t i o n~}d^{(k)}}\\"
                r"&{\mathrm{e f i n e d~b y:}}\\"
                r"&d^{(k)}=-g^{(k)},k=0\tag{13}\end{array}"
            ),
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertIsNone(enhancement["latex_text"])
        self.assertIn("explanatory_prose_in_formula_candidate", enhancement["latex_validation"]["issues"])

    def test_accepts_short_paddle_formula_with_eqno_number_after_normalization(self) -> None:
        equation = {
            "text": "d w x y x y w",
            "equation_label": "(5)",
        }
        candidate = FormulaOcrCandidate(
            latex=r"d_{w}(x,y){=}||x-y||_{w},\eqno(5)",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertTrue(enhancement["latex_validation"]["accepted"])
        self.assertIn(r"\tag{5}", str(enhancement["latex_text"]))
        self.assertIn("syntax_valid", enhancement["latex_validation"]["signals"])

    def test_accepts_short_paddle_formula_without_visible_number_when_symbol_structure_is_clear(self) -> None:
        equation = {
            "text": "d w x y x y w",
            "equation_label": "(5)",
        }
        candidate = FormulaOcrCandidate(
            latex=r"d_{w}(x,y){=}||x-y||_{w},",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        enhancement = build_formula_latex_enhancement(equation, candidate)

        self.assertTrue(enhancement["latex_validation"]["accepted"])
        self.assertIn(r"d_{w}", str(enhancement["latex_text"]))

    def test_inline_formula_ocr_enhances_only_ocr_candidate_spans_without_overwriting_text(self) -> None:
        evidence_items = [
            {
                "evidence_id": "ce_text_inline_001",
                "source_type": "text",
                "page": 3,
                "content_text": "set, respectively, i.e., u j = n 1 j sum x i and u = n 1 sum x i .",
                "display_text": r"set, respectively, i.e., u_j = \frac{1}{n_j}\sum_i x_i and u = \frac{1}{n}\sum_i x_i.",
                "inline_formula_spans": [
                    {
                        "type": "inline_equation",
                        "bbox": [100.0, 200.0, 260.0, 214.0],
                        "content": r"u_j = \frac{1}{n_j}\sum_i x_i and u = \frac{1}{n}\sum_i x_i.",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    }
                ],
                "segments": [
                    {
                        "role": "body",
                        "text": "set, respectively, i.e.",
                        "inline_formula_spans": [
                            {
                                "type": "inline_equation",
                                "bbox": [100.0, 200.0, 260.0, 214.0],
                                "content": r"u_j = \frac{1}{n_j}\sum_i x_i and u = \frac{1}{n}\sum_i x_i.",
                                "formula_complexity": "inline_formula",
                                "ocr_candidate": True,
                            },
                            {
                                "type": "inline_equation",
                                "bbox": [300.0, 200.0, 312.0, 214.0],
                                "content": "x_i",
                                "formula_complexity": "inline_symbol",
                                "ocr_candidate": False,
                            },
                        ],
                    }
                ],
            }
        ]
        candidate = FormulaOcrCandidate(
            latex=r"u_j=\frac{1}{n_j}\sum_i x_i,\quad u=\frac{1}{n}\sum_i x_i",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        with patch("parsers.pdf.formula_ocr.formula_ocr_enabled", return_value=True), patch(
            "parsers.pdf.formula_ocr._recognize_inline_formula_latex",
            return_value=candidate,
        ) as recognize_mock:
            enhance_inline_formula_spans_with_formula_ocr("sample.pdf", evidence_items)

        self.assertEqual(recognize_mock.call_count, 1)
        enhanced_span = evidence_items[0]["inline_formula_spans"][0]
        self.assertEqual(
            enhanced_span["content"],
            r"u_j = \frac{1}{n_j}\sum_i x_i and u = \frac{1}{n}\sum_i x_i.",
        )
        self.assertEqual(
            evidence_items[0]["display_text"],
            r"set, respectively, i.e., u_j = \frac{1}{n_j}\sum_i x_i and u = \frac{1}{n}\sum_i x_i.",
        )
        self.assertEqual(enhanced_span["latex_source"], "paddleocr_formula:PP-FormulaNet_plus-M")
        self.assertEqual(enhanced_span["latex_render_policy"], "text_primary_latex_enhancement")
        self.assertIn(r"\frac{1}{n_j}", str(enhanced_span["latex_text"]))
        self.assertTrue(enhanced_span["latex_validation"]["accepted"])

        segment_spans = evidence_items[0]["segments"][0]["inline_formula_spans"]
        self.assertIn(r"\frac{1}{n_j}", str(segment_spans[0]["latex_text"]))
        self.assertNotIn("latex_text", segment_spans[1])
        self.assertEqual(segment_spans[1]["content"], "x_i")

    def test_inline_formula_ocr_rejected_candidate_is_metadata_only(self) -> None:
        evidence_items = [
            {
                "evidence_id": "ce_text_inline_002",
                "source_type": "text",
                "page": 4,
                "content_text": "Theorem. Let F ( v ) be a function of v defined in (11). If F v v v = * 0 =",
                "inline_formula_spans": [
                    {
                        "type": "inline_equation",
                        "bbox": [90.0, 650.0, 210.0, 682.0],
                        "content": "F v v v = * 0 =",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    }
                ],
            }
        ]
        candidate = FormulaOcrCandidate(
            latex=r"\begin{array}{r}{\left.\frac{\partial F}{\partial v}\right|_{y=y_{addlo}^{*}}}\end{array}",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        with patch("parsers.pdf.formula_ocr.formula_ocr_enabled", return_value=True), patch(
            "parsers.pdf.formula_ocr._recognize_inline_formula_latex",
            return_value=candidate,
        ):
            enhance_inline_formula_spans_with_formula_ocr("sample.pdf", evidence_items)

        span = evidence_items[0]["inline_formula_spans"][0]
        self.assertEqual(span["content"], "F v v v = * 0 =")
        self.assertIsNone(span["latex_text"])
        self.assertEqual(span["latex_confidence"], 0.0)
        self.assertIn("latex_candidate_text", span)
        self.assertFalse(span["latex_validation"]["accepted"])
        self.assertIn("unexplained_symbol_tokens", span["latex_validation"]["issues"])

    def test_inline_formula_ocr_respects_document_budget_and_marks_skipped_spans(self) -> None:
        evidence_items = [
            {
                "evidence_id": "ce_text_inline_budget_001",
                "source_type": "text",
                "page": 3,
                "inline_formula_spans": [
                    {
                        "type": "inline_equation",
                        "bbox": [100.0, 200.0, 260.0, 214.0],
                        "content": r"u_j = \frac{1}{n_j}\sum_i x_i",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    },
                    {
                        "type": "inline_equation",
                        "bbox": [100.0, 230.0, 260.0, 244.0],
                        "content": r"v_j = \frac{1}{n_j}\sum_i y_i",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    },
                    {
                        "type": "inline_equation",
                        "bbox": [100.0, 260.0, 260.0, 274.0],
                        "content": r"w_j = \frac{1}{n_j}\sum_i z_i",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    },
                ],
            }
        ]
        candidate = FormulaOcrCandidate(
            latex=r"u_j=\frac{1}{n_j}\sum_i x_i",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        with patch("parsers.pdf.formula_ocr.formula_ocr_enabled", return_value=True), patch.dict(
            "os.environ",
            {"IND_FORMULA_OCR_MAX_INLINE_SPANS": "1"},
            clear=False,
        ), patch(
            "parsers.pdf.formula_ocr._recognize_inline_formula_latex",
            return_value=candidate,
        ) as recognize_mock:
            enhance_inline_formula_spans_with_formula_ocr("sample.pdf", evidence_items)

        self.assertEqual(recognize_mock.call_count, 1)
        spans = evidence_items[0]["inline_formula_spans"]
        self.assertIn(r"\frac{1}{n_j}", str(spans[0].get("latex_text")))
        self.assertNotIn("latex_text", spans[1])
        self.assertNotIn("latex_text", spans[2])
        self.assertEqual(spans[1].get("latex_skip_reason"), "inline_formula_ocr_document_budget_exceeded")
        self.assertEqual(spans[2].get("latex_skip_reason"), "inline_formula_ocr_document_budget_exceeded")
        self.assertEqual(spans[1].get("latex_render_policy"), "text_primary_latex_enhancement")

    def test_inline_formula_ocr_respects_page_and_block_budgets(self) -> None:
        evidence_items = [
            {
                "evidence_id": "ce_text_inline_budget_002",
                "source_type": "text",
                "page": 3,
                "inline_formula_spans": [
                    {
                        "type": "inline_equation",
                        "bbox": [100.0, 200.0, 260.0, 214.0],
                        "content": r"u_j = \frac{1}{n_j}\sum_i x_i",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    },
                    {
                        "type": "inline_equation",
                        "bbox": [100.0, 230.0, 260.0, 244.0],
                        "content": r"v_j = \frac{1}{n_j}\sum_i y_i",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    },
                ],
            },
            {
                "evidence_id": "ce_text_inline_budget_003",
                "source_type": "text",
                "page": 4,
                "inline_formula_spans": [
                    {
                        "type": "inline_equation",
                        "bbox": [100.0, 200.0, 260.0, 214.0],
                        "content": r"w_j = \frac{1}{n_j}\sum_i z_i",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    }
                ],
            },
        ]
        candidate = FormulaOcrCandidate(
            latex=r"u_j=\frac{1}{n_j}\sum_i x_i",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        with patch("parsers.pdf.formula_ocr.formula_ocr_enabled", return_value=True), patch.dict(
            "os.environ",
            {
                "IND_FORMULA_OCR_MAX_INLINE_SPANS": "10",
                "IND_FORMULA_OCR_MAX_INLINE_SPANS_PER_PAGE": "1",
                "IND_FORMULA_OCR_MAX_INLINE_SPANS_PER_BLOCK": "1",
            },
            clear=False,
        ), patch(
            "parsers.pdf.formula_ocr._recognize_inline_formula_latex",
            return_value=candidate,
        ) as recognize_mock:
            enhance_inline_formula_spans_with_formula_ocr("sample.pdf", evidence_items)

        self.assertEqual(recognize_mock.call_count, 2)
        page3_spans = evidence_items[0]["inline_formula_spans"]
        page4_spans = evidence_items[1]["inline_formula_spans"]
        self.assertIn(r"\frac{1}{n_j}", str(page3_spans[0].get("latex_text")))
        self.assertEqual(page3_spans[1].get("latex_skip_reason"), "inline_formula_ocr_page_budget_exceeded")
        self.assertIn(r"\frac{1}{n_j}", str(page4_spans[0].get("latex_text")))

    def test_inline_formula_ocr_cached_duplicate_does_not_consume_budget_twice(self) -> None:
        shared_span = {
            "type": "inline_equation",
            "bbox": [100.0, 200.0, 260.0, 214.0],
            "content": r"u_j = \frac{1}{n_j}\sum_i x_i",
            "formula_complexity": "inline_formula",
            "ocr_candidate": True,
        }
        evidence_items = [
            {
                "evidence_id": "ce_text_inline_budget_004",
                "source_type": "text",
                "page": 3,
                "inline_formula_spans": [
                    dict(shared_span),
                    dict(shared_span),
                    {
                        "type": "inline_equation",
                        "bbox": [100.0, 230.0, 260.0, 244.0],
                        "content": r"v_j = \frac{1}{n_j}\sum_i y_i",
                        "formula_complexity": "inline_formula",
                        "ocr_candidate": True,
                    },
                ],
            }
        ]
        candidate = FormulaOcrCandidate(
            latex=r"u_j=\frac{1}{n_j}\sum_i x_i",
            confidence=0.96,
            source="paddleocr_formula:PP-FormulaNet_plus-M",
        )

        with patch("parsers.pdf.formula_ocr.formula_ocr_enabled", return_value=True), patch.dict(
            "os.environ",
            {"IND_FORMULA_OCR_MAX_INLINE_SPANS": "1"},
            clear=False,
        ), patch(
            "parsers.pdf.formula_ocr._recognize_inline_formula_latex",
            return_value=candidate,
        ) as recognize_mock:
            enhance_inline_formula_spans_with_formula_ocr("sample.pdf", evidence_items)

        self.assertEqual(recognize_mock.call_count, 1)
        spans = evidence_items[0]["inline_formula_spans"]
        self.assertIn(r"\frac{1}{n_j}", str(spans[0].get("latex_text")))
        self.assertIn(r"\frac{1}{n_j}", str(spans[1].get("latex_text")))
        self.assertEqual(spans[2].get("latex_skip_reason"), "inline_formula_ocr_document_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
