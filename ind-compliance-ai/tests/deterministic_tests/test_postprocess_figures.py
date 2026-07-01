from __future__ import annotations

import unittest

from parsers.pdf.postprocess_figures import (
    _figure_reference_token_count,
    _looks_like_contaminated_figure_caption_text,
    _looks_like_explicit_figure_caption_text,
    _looks_like_figure_body_reference_cue,
    _looks_like_figure_legend_text,
)


class PostprocessFigureTests(unittest.TestCase):
    def test_explicit_figure_caption_accepts_caption_labels_but_rejects_body_reference_labels(self) -> None:
        self.assertTrue(_looks_like_explicit_figure_caption_text("Figure 2. Dose response curve"))
        self.assertTrue(_looks_like_explicit_figure_caption_text("图 3：血药浓度曲线"))
        self.assertFalse(_looks_like_explicit_figure_caption_text("Figure 2 shows the observed dose response."))
        self.assertFalse(_looks_like_explicit_figure_caption_text("The observed dose response is shown in Figure 2."))

    def test_figure_body_reference_cue_detects_body_reference_phrasing(self) -> None:
        self.assertTrue(_looks_like_figure_body_reference_cue("Figure 2 shows the observed dose response."))
        self.assertTrue(_looks_like_figure_body_reference_cue("The observed dose response is shown in Fig. 2."))
        self.assertTrue(_looks_like_figure_body_reference_cue("结果如图 2 所示"))
        self.assertFalse(_looks_like_figure_body_reference_cue("Figure 2. Dose response curve"))

    def test_figure_reference_token_count_counts_english_and_cjk_reference_tokens(self) -> None:
        self.assertEqual(_figure_reference_token_count("See Fig. 1 and Figure 2, plus 图 3."), 3)
        self.assertEqual(_figure_reference_token_count("No referenced figure here."), 0)

    def test_contaminated_caption_requires_body_reference_or_multiple_reference_tokens(self) -> None:
        self.assertTrue(_looks_like_contaminated_figure_caption_text("Figure 2 shows the result in Figure 3."))
        self.assertTrue(_looks_like_contaminated_figure_caption_text("The result is shown in Fig. 2."))
        self.assertFalse(_looks_like_contaminated_figure_caption_text("Figure 2. Dose response curve"))
        self.assertFalse(_looks_like_contaminated_figure_caption_text(""))

    def test_figure_legend_accepts_caption_labels_and_statistical_legend_cues(self) -> None:
        self.assertTrue(_looks_like_figure_legend_text("Figure 2. Dose response curve"))
        self.assertTrue(_looks_like_figure_legend_text("Mean ± standard error, n = 6."))
        self.assertTrue(_looks_like_figure_legend_text("统计学显著，p < 0.05"))
        self.assertFalse(_looks_like_figure_legend_text("This is ordinary nearby context."))


if __name__ == "__main__":
    unittest.main()
