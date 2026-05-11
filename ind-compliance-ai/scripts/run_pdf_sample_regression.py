from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the standing PDF parser regression gate for the core enterprise samples."
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="unittest verbosity level (default: 2)",
    )
    args = parser.parse_args()

    _configure_optional_a_tst_sample()

    suite = unittest.defaultTestLoader.loadTestsFromNames(DEFAULT_TEST_MODULES)
    runner = unittest.TextTestRunner(verbosity=max(1, args.verbosity))
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
