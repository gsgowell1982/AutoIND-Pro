from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCHMARK_ENGINE_NAME = "autoind-pro-benchmark-neutral"


@dataclass(frozen=True)
class GuardProfile:
    name: str
    intent: str
    description: str
    reference_run: str
    reference_metrics: dict[str, float]
    hard_minimums: dict[str, float]
    soft_minimums: dict[str, float]


@dataclass(frozen=True)
class GuardResult:
    profile: str
    passed: bool
    failures: list[str]
    warnings: list[str]
    observed: dict[str, float | int | None]
    report_markdown: str


@dataclass(frozen=True)
class GuardRunPaths:
    benchmark_root: Path
    output_dir: Path
    guard_report: Path


def get_guard_profile(name: str) -> GuardProfile:
    profiles = {
        "active-s394": GuardProfile(
            name="active-s394",
            intent="active_regression_guard",
            description=(
                "Current-code OpenDataLoader 200-sample regression guard for the benchmark-specific "
                "projection layer. This is not production IND review Markdown."
            ),
            reference_run="s394-chart-inventory-full",
            reference_metrics={
                "overall_mean": 0.9216329477548946,
                "nid_mean": 0.9412561630828833,
                "nid_s_mean": 0.9394511758123503,
                "teds_mean": 0.974097593011289,
                "teds_s_mean": 0.9950675783133767,
                "mhs_mean": 0.844626147148826,
                "mhs_s_mean": 0.9305339122040028,
            },
            hard_minimums={
                "missing_predictions": 0,
                "teds_mean": 0.974097593011289,
                "teds_s_mean": 0.9950675783133767,
            },
            soft_minimums={
                "overall_mean": 0.9210,
                "nid_mean": 0.9410,
                "mhs_mean": 0.8440,
            },
        ),
        "public-s358": GuardProfile(
            name="public-s358",
            intent="public_reference",
            description=(
                "Public promotional OpenDataLoader reference. Use archived s358 artifacts or the s358 "
                "snapshot/tag for exact reproduction; current-code regression protection should normally "
                "use active-s394."
            ),
            reference_run="s358-table-title-body-tail-boundary-evalonly",
            reference_metrics={
                "overall_mean": 0.9232562090486595,
                "nid_mean": 0.9428950593899802,
                "nid_s_mean": 0.9434146896369913,
                "teds_mean": 0.9733554130557177,
                "teds_s_mean": 0.9932701243991438,
                "mhs_mean": 0.8468248963171249,
                "mhs_s_mean": 0.9351301735647529,
            },
            hard_minimums={
                "missing_predictions": 0,
            },
            soft_minimums={
                "overall_mean": 0.9230,
                "nid_mean": 0.9425,
                "mhs_mean": 0.8460,
                "teds_mean": 0.9730,
                "teds_s_mean": 0.9930,
            },
        ),
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"Unknown OpenDataLoader guard profile: {name}") from exc


def evaluate_guard_payload(payload: dict[str, Any], profile: GuardProfile) -> GuardResult:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    score = metrics.get("score") if isinstance(metrics.get("score"), dict) else {}
    observed: dict[str, float | int | None] = {
        "missing_predictions": _number_or_none(metrics.get("missing_predictions")),
    }
    for key in (
        "overall_mean",
        "nid_mean",
        "nid_s_mean",
        "teds_mean",
        "teds_s_mean",
        "mhs_mean",
        "mhs_s_mean",
    ):
        observed[key] = _number_or_none(score.get(key))

    failures: list[str] = []
    warnings: list[str] = []
    for key, minimum in profile.hard_minimums.items():
        value = observed.get(key)
        if value is None:
            failures.append(f"{key} is missing; expected >= {minimum}")
        elif key == "missing_predictions":
            if value != minimum:
                failures.append(f"{key}={value}; expected {minimum}")
        elif float(value) + 1e-12 < minimum:
            failures.append(f"{key}={value}; hard minimum {minimum}")
    for key, minimum in profile.soft_minimums.items():
        value = observed.get(key)
        if value is None:
            warnings.append(f"{key} is missing; soft minimum {minimum}")
        elif float(value) + 1e-12 < minimum:
            warnings.append(f"{key}={value}; soft minimum {minimum}")

    report = _build_guard_report(profile, observed, failures, warnings)
    return GuardResult(
        profile=profile.name,
        passed=not failures,
        failures=failures,
        warnings=warnings,
        observed=observed,
        report_markdown=report,
    )


def _number_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    return None


def _build_guard_report(
    profile: GuardProfile,
    observed: dict[str, float | int | None],
    failures: list[str],
    warnings: list[str],
) -> str:
    rows = [
        "| Metric | Observed | Reference | Hard Min | Soft Min |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    keys = [
        "overall_mean",
        "nid_mean",
        "nid_s_mean",
        "teds_mean",
        "teds_s_mean",
        "mhs_mean",
        "mhs_s_mean",
        "missing_predictions",
    ]
    for key in keys:
        rows.append(
            "| "
            + " | ".join(
                [
                    key,
                    _fmt(observed.get(key)),
                    _fmt(profile.reference_metrics.get(key)),
                    _fmt(profile.hard_minimums.get(key)),
                    _fmt(profile.soft_minimums.get(key)),
                ]
            )
            + " |"
        )
    status = "PASS" if not failures else "FAIL"
    failure_text = "None" if not failures else "\n".join(f"- {item}" for item in failures)
    warning_text = "None" if not warnings else "\n".join(f"- {item}" for item in warnings)
    return f"""# OpenDataLoader Regression Guard - {profile.name}

Generated: `{time.strftime("%Y-%m-%d %H:%M:%S")}`

Status: **{status}**

Profile intent: `{profile.intent}`

Reference run: `{profile.reference_run}`

Projection boundary: this guard uses AutoIND's benchmark-specific projection layer. It is independent from production IND review Markdown.

{profile.description}

## Metrics

{chr(10).join(rows)}

## Hard Failures

{failure_text}

## Soft Warnings

{warning_text}
"""


def _fmt(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.12g}"
    return "null"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_benchmark_root() -> Path:
    return _repo_root().parent / "external_benchmarks" / "opendataloader-bench"


def resolve_guard_run_paths(args: argparse.Namespace, profile: GuardProfile) -> GuardRunPaths:
    benchmark_root = args.benchmark_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else benchmark_root / f"{profile.name}-regression-guard"
    )
    guard_report = (
        args.guard_report.resolve()
        if args.guard_report
        else output_dir / "regression_guard_report.md"
    )
    return GuardRunPaths(
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        guard_report=guard_report,
    )


def run_guard(args: argparse.Namespace) -> GuardResult:
    profile = get_guard_profile(args.profile)
    paths = resolve_guard_run_paths(args, profile)
    eval_path: Path
    if args.evaluation_json:
        eval_path = args.evaluation_json.resolve()
    else:
        smoke_script = Path(__file__).with_name("opendataloader_bench_smoke.py")
        command = [
            sys.executable,
            str(smoke_script),
            "--benchmark-root",
            str(paths.benchmark_root),
            "--count",
            str(args.count),
            "--projection",
            "benchmark-neutral",
            "--output-dir",
            str(paths.output_dir),
        ]
        subprocess.run(command, cwd=_repo_root(), check=True)
        eval_path = (
            paths.output_dir
            / "prediction-smoke"
            / BENCHMARK_ENGINE_NAME
            / "evaluation.smoke.json"
        )
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    result = evaluate_guard_payload(payload, profile)
    report_path = paths.guard_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.report_markdown, encoding="utf-8")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or validate the OpenDataLoader 200-sample regression guard. "
            "This uses the benchmark-specific projection layer, not production IND review Markdown."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("active-s394", "public-s358"),
        default="active-s394",
        help="Guard profile. active-s394 is the current-code regression guard; public-s358 is the promotional reference.",
    )
    parser.add_argument("--benchmark-root", type=Path, default=_default_benchmark_root())
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Smoke output directory. Defaults to <benchmark-root>/<profile>-regression-guard.",
    )
    parser.add_argument(
        "--evaluation-json",
        type=Path,
        default=None,
        help="Validate an existing OpenDataLoader evaluation JSON instead of running the parser.",
    )
    parser.add_argument(
        "--guard-report",
        type=Path,
        default=None,
        help="Guard report path. Defaults to <output-dir>/regression_guard_report.md.",
    )
    return parser.parse_args()


def main() -> None:
    result = run_guard(_parse_args())
    print(result.report_markdown)
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
