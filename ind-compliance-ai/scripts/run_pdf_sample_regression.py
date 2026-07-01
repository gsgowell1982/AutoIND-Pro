from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GATE_TEST_MODULES = {
    "ectd": [
        "tests.parser_tests.test_ectd_regression_sample",
        "tests.parser_tests.test_ectd_implementation_guide_regression",
        "tests.parser_tests.test_ectd_validation_standard_regression",
    ],
    "fast": [
        "tests.parser_tests.test_a_tst_regression",
        "tests.parser_tests.test_two_column_literature_regression",
        "tests.parser_tests.test_r2_regression",
        "tests.parser_tests.test_ectd_regression_sample",
        "tests.parser_tests.test_ectd_implementation_guide_regression",
        "tests.parser_tests.test_ectd_validation_standard_regression",
        "tests.deterministic_tests.test_opendataloader_goal_metrics",
    ],
    "protected": [
        "tests.parser_tests.test_a_tst_regression",
        "tests.parser_tests.test_two_column_literature_regression",
        "tests.parser_tests.test_r2_regression",
        "tests.parser_tests.test_ectd_regression_sample",
        "tests.parser_tests.test_ectd_implementation_guide_regression",
        "tests.deterministic_tests.test_opendataloader_goal_metrics",
        "tests.parser_tests.test_ectd_validation_standard_regression",
        "tests.parser_tests.test_test_ind_figure_footer_guard",
        "tests.parser_tests.test_pdf_sample_batch_regression",
        "tests.deterministic_tests.test_real_pdf_contract_readiness",
        "tests.parser_tests.test_opendataloader_benchmark_regression",
        "tests.parser_tests.test_text_block_visual_reconstruction",
    ],
    "r2-after-80": [
        "tests.parser_tests.test_r2_regression",
    ],
}


DEFAULT_TEST_MODULES = [
    "tests.parser_tests.test_text_block_visual_reconstruction",
    "tests.parser_tests.test_two_column_literature_regression",
    "tests.parser_tests.test_ectd_regression_sample",
    "tests.parser_tests.test_test_ind_figure_footer_guard",
    "tests.parser_tests.test_a_tst_regression",
    "tests.deterministic_tests.test_real_pdf_contract_readiness",
]


def _configure_optional_a_tst_sample() -> None:
    if os.environ.get("IND_A_TST_REGRESSION_PDF", "").strip():
        return

    candidate_paths = [
        PROJECT_ROOT.parent / "A-tst.pdf",
        Path(r"D:\AutoIND-Pro\A-tst.pdf"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            os.environ["IND_A_TST_REGRESSION_PDF"] = str(candidate)
            break


def _configure_optional_pdf_sample(env_key: str, candidate_paths: list[Path]) -> None:
    if os.environ.get(env_key, "").strip():
        return

    for candidate in candidate_paths:
        if candidate.exists():
            os.environ[env_key] = str(candidate)
            break


def configure_default_pdf_samples() -> None:
    _configure_optional_a_tst_sample()
    _configure_optional_pdf_sample(
        "IND_ECTD_REGRESSION_PDF",
        [
            PROJECT_ROOT.parent / "eCTD技术规范.pdf",
            Path(r"D:\AutoIND-Pro\eCTD技术规范.pdf"),
        ],
    )
    _configure_optional_pdf_sample(
        "IND_ECTD_IMPLEMENTATION_GUIDE_REGRESSION_PDF",
        [
            PROJECT_ROOT.parent / "eCTD实施指南.pdf",
            Path(r"D:\AutoIND-Pro\eCTD实施指南.pdf"),
        ],
    )
    _configure_optional_pdf_sample(
        "IND_ECTD_VALIDATION_STANDARD_REGRESSION_PDF",
        [
            PROJECT_ROOT.parent / "eCTD验证标准.pdf",
            Path(r"D:\AutoIND-Pro\eCTD验证标准.pdf"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the standing PDF parser regression gate for the core enterprise samples."
    )
    parser.add_argument(
        "--gate",
        choices=sorted(GATE_TEST_MODULES),
        default="protected",
        help="Regression gate to run: ectd, fast, protected, or r2-after-80 (default: protected)",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="unittest verbosity level (default: 2)",
    )
    args = parser.parse_args()

    configure_default_pdf_samples()

    suite = unittest.defaultTestLoader.loadTestsFromNames(GATE_TEST_MODULES[args.gate])
    runner = unittest.TextTestRunner(verbosity=max(1, args.verbosity))
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
