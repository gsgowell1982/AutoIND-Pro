from __future__ import annotations

import os
import unittest

from scripts.run_pdf_sample_regression import GATE_TEST_MODULES, configure_default_pdf_samples


ECTD_PDF_REGRESSION_MODULES = {
    "tests.parser_tests.test_ectd_regression_sample",
    "tests.parser_tests.test_ectd_implementation_guide_regression",
    "tests.parser_tests.test_ectd_validation_standard_regression",
}


class PdfRegressionGateConfigTests(unittest.TestCase):
    def test_fast_gate_contains_core_ind_text_layer_samples(self) -> None:
        modules = set(GATE_TEST_MODULES["fast"])

        self.assertIn("tests.parser_tests.test_a_tst_regression", modules)
        self.assertIn("tests.parser_tests.test_two_column_literature_regression", modules)
        self.assertIn("tests.parser_tests.test_r2_regression", modules)
        self.assertTrue(ECTD_PDF_REGRESSION_MODULES.issubset(modules))
        self.assertIn("tests.deterministic_tests.test_opendataloader_goal_metrics", modules)

    def test_ectd_gate_is_the_complete_three_pdf_parser_baseline(self) -> None:
        modules = set(GATE_TEST_MODULES["ectd"])

        self.assertEqual(modules, ECTD_PDF_REGRESSION_MODULES)

    def test_protected_gate_extends_fast_gate_with_confirmed_pdf_contracts(self) -> None:
        fast_modules = set(GATE_TEST_MODULES["fast"])
        protected_modules = set(GATE_TEST_MODULES["protected"])

        self.assertTrue(fast_modules.issubset(protected_modules))
        self.assertTrue(ECTD_PDF_REGRESSION_MODULES.issubset(protected_modules))
        self.assertIn("tests.parser_tests.test_test_ind_figure_footer_guard", protected_modules)
        self.assertIn("tests.parser_tests.test_pdf_sample_batch_regression", protected_modules)
        self.assertIn("tests.deterministic_tests.test_real_pdf_contract_readiness", protected_modules)
        self.assertIn("tests.parser_tests.test_opendataloader_benchmark_regression", protected_modules)

    def test_r2_after_80_gate_is_explicitly_separate_from_general_protected_gate(self) -> None:
        modules = GATE_TEST_MODULES["r2-after-80"]

        self.assertEqual(modules, ["tests.parser_tests.test_r2_regression"])

    def test_default_pdf_sample_configuration_includes_all_ectd_parser_regression_pdfs(self) -> None:
        env_keys = [
            "IND_ECTD_REGRESSION_PDF",
            "IND_ECTD_IMPLEMENTATION_GUIDE_REGRESSION_PDF",
            "IND_ECTD_VALIDATION_STANDARD_REGRESSION_PDF",
        ]
        previous_values = {key: os.environ.get(key) for key in env_keys}
        try:
            for key in env_keys:
                os.environ.pop(key, None)

            configure_default_pdf_samples()

            for key in env_keys:
                with self.subTest(env_key=key):
                    self.assertIn(key, os.environ)
                    self.assertTrue(os.path.exists(os.environ[key]))
        finally:
            for key, value in previous_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
