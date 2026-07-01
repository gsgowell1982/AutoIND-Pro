from __future__ import annotations

import unittest

import scripts.opendataloader_bench_smoke as bench_smoke
from scripts.run_opendataloader_regression_guard import (
    evaluate_guard_payload,
    get_guard_profile,
)
import scripts.run_opendataloader_regression_guard as regression_guard


def _payload(
    *,
    overall: float,
    nid: float,
    teds: float,
    teds_s: float,
    mhs: float,
    missing_predictions: int = 0,
) -> dict[str, object]:
    return {
        "metrics": {
            "score": {
                "overall_mean": overall,
                "nid_mean": nid,
                "teds_mean": teds,
                "teds_s_mean": teds_s,
                "mhs_mean": mhs,
            },
            "missing_predictions": missing_predictions,
        },
        "documents": [],
    }


class OpenDataLoaderRegressionGuardTests(unittest.TestCase):
    def test_active_s394_accepts_current_guard_metrics(self) -> None:
        result = evaluate_guard_payload(
            _payload(
                overall=0.9216329477548946,
                nid=0.9412561630828833,
                teds=0.974097593011289,
                teds_s=0.9950675783133767,
                mhs=0.844626147148826,
            ),
            get_guard_profile("active-s394"),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.failures, [])

    def test_active_s394_teds_drop_is_hard_failure(self) -> None:
        result = evaluate_guard_payload(
            _payload(
                overall=0.95,
                nid=0.95,
                teds=0.9700,
                teds_s=0.9950675783133767,
                mhs=0.90,
            ),
            get_guard_profile("active-s394"),
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("teds_mean" in failure for failure in result.failures))

    def test_active_s394_nid_drop_is_soft_warning_by_default(self) -> None:
        result = evaluate_guard_payload(
            _payload(
                overall=0.9215,
                nid=0.9300,
                teds=0.974097593011289,
                teds_s=0.9950675783133767,
                mhs=0.8445,
            ),
            get_guard_profile("active-s394"),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.failures, [])
        self.assertTrue(any("nid_mean" in warning for warning in result.warnings))

    def test_public_s358_is_reference_profile(self) -> None:
        profile = get_guard_profile("public-s358")

        self.assertEqual(profile.name, "public-s358")
        self.assertEqual(profile.intent, "public_reference")
        self.assertAlmostEqual(profile.reference_metrics["overall_mean"], 0.9232562090486595)

    def test_direct_guard_rerun_smoke_adapter_is_available(self) -> None:
        smoke_script = regression_guard.Path(regression_guard.__file__).with_name(
            "opendataloader_bench_smoke.py"
        )

        self.assertTrue(
            smoke_script.exists(),
            msg=f"Direct OpenDataLoader guard reruns require the smoke adapter: {smoke_script}",
        )

    def test_direct_guard_defaults_output_under_selected_benchmark_root(self) -> None:
        profile = get_guard_profile("active-s394")
        benchmark_root = regression_guard.Path("D:/selected/opendataloader-bench")
        args = regression_guard.argparse.Namespace(
            benchmark_root=benchmark_root,
            output_dir=None,
            guard_report=None,
        )

        resolved = regression_guard.resolve_guard_run_paths(args, profile)

        expected_output = benchmark_root / "active-s394-regression-guard"
        self.assertEqual(resolved.benchmark_root, benchmark_root)
        self.assertEqual(resolved.output_dir, expected_output)
        self.assertEqual(
            resolved.guard_report,
            expected_output / "regression_guard_report.md",
        )

    def test_direct_guard_preserves_explicit_output_and_report_paths(self) -> None:
        profile = get_guard_profile("active-s394")
        benchmark_root = regression_guard.Path("D:/selected/opendataloader-bench")
        output_dir = regression_guard.Path("D:/custom/output")
        guard_report = regression_guard.Path("D:/custom/report.md")
        args = regression_guard.argparse.Namespace(
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            guard_report=guard_report,
        )

        resolved = regression_guard.resolve_guard_run_paths(args, profile)

        self.assertEqual(resolved.benchmark_root, benchmark_root)
        self.assertEqual(resolved.output_dir, output_dir)
        self.assertEqual(resolved.guard_report, guard_report)

    def test_smoke_adapter_requires_output_under_benchmark_root(self) -> None:
        benchmark_root = bench_smoke.Path("D:/selected/opendataloader-bench")
        outside_output = bench_smoke.Path("D:/external/adapter-output")

        with self.assertRaisesRegex(ValueError, "under --benchmark-root"):
            bench_smoke.validate_output_dir_under_benchmark_root(
                benchmark_root,
                outside_output,
            )


if __name__ == "__main__":
    unittest.main()
