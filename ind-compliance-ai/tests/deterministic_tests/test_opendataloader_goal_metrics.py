from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.opendataloader_goal_metrics import (
    compare_table_regression_baseline,
    compute_goal_metrics,
    normalize_markdown_for_goal_text,
    _documents_are_toc_projection_equivalent,
)


class OpenDataLoaderGoalMetricsTests(unittest.TestCase):
    def test_normalized_goal_text_ignores_markdown_format_and_toc_blocks(self) -> None:
        ground_truth = """# Table of Contents

Introduction ........ 1
Methods ........ 2

# Introduction

The **primary** endpoint was measured in cohort A.
"""
        prediction = """### Table of Contents

- Introduction
- Methods

## Introduction

The primary endpoint was measured in cohort A.
"""

        self.assertEqual(
            normalize_markdown_for_goal_text(ground_truth),
            normalize_markdown_for_goal_text(prediction),
        )

    def test_normalized_goal_text_ignores_autoind_snapshot_wrapper(self) -> None:
        ground_truth = "Figure 1.6. Alien temporary work permits, Thailand\n\nSource: Department of Employment\n"
        prediction = """# IND Parse Snapshot (Full)

Generated at: 2026-05-26T11:27:07+00:00

## 01030000000076.pdf (PDF)

- Estimated pages: 1
- Parser strategy: pdf-ast-v5

### 正文结构化内容

Figure 1.6. Alien temporary work permits, Thailand

Source: Department of Employment
"""

        self.assertEqual(
            normalize_markdown_for_goal_text(ground_truth),
            normalize_markdown_for_goal_text(prediction),
        )

    def test_goal_metrics_can_reach_one_when_only_format_and_toc_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "doc1.md").write_text(
                "# Table of Contents\n\nA .... 1\n\n# A\n\nValue **A** is present.\n",
                encoding="utf-8",
            )
            (pred_dir / "doc1.md").write_text(
                "### Table of Contents\n\n- A\n\n## A\n\nValue A is present.\n",
                encoding="utf-8",
            )
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "score": {
                                "overall_mean": 0.25,
                                "nid_mean": 0.5,
                                "teds_mean": None,
                                "mhs_mean": 0.0,
                            },
                            "missing_predictions": 0,
                        },
                        "documents": [
                            {
                                "document_id": "doc1",
                                "scores": {
                                    "overall": 0.25,
                                    "nid": 0.5,
                                    "teds": None,
                                    "teds_s": None,
                                    "mhs": 0.0,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        score = payload["metrics"]["goal_score"]
        self.assertEqual(score["core_text_mean"], 1.0)
        self.assertEqual(score["comprehensive_mean"], 1.0)
        self.assertEqual(payload["weak_samples"], [])

    def test_table_presence_and_quality_are_still_core_goal_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "doc_table.md").write_text(
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                encoding="utf-8",
            )
            (pred_dir / "doc_table.md").write_text(
                "A B 1 2\n",
                encoding="utf-8",
            )
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "score": {
                                "overall_mean": 0.4,
                                "nid_mean": 0.8,
                                "teds_mean": 0.0,
                                "mhs_mean": None,
                            },
                            "missing_predictions": 0,
                        },
                        "documents": [
                            {
                                "document_id": "doc_table",
                                "scores": {
                                    "overall": 0.4,
                                    "nid": 0.8,
                                    "teds": 0.0,
                                    "teds_s": 0.0,
                                    "mhs": None,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        self.assertEqual(payload["metrics"]["table_presence"]["recall"], 0.0)
        self.assertLess(payload["metrics"]["goal_score"]["comprehensive_mean"], 1.0)
        self.assertEqual(payload["weak_samples"][0]["primary_gap"], "table_region_discovery")

    def test_goal_text_normalization_treats_html_and_markdown_tables_as_format_variants(self) -> None:
        ground_truth = """Report title

<table>
<tr>
<td rowspan="2">
No.
</td>
<td>
Political party
</td>
<td>
Number of candidates
</td>
</tr>
<tr>
<td>
Total
</td>
<td>
86,092
</td>
</tr>
</table>

24
"""
        prediction = """Report title

| No. | Political party | Number of candidates |
| --- | --- | --- |
|  | Total | 86,092 |

24
"""

        normalized_gt = normalize_markdown_for_goal_text(ground_truth)
        normalized_pred = normalize_markdown_for_goal_text(prediction)

        self.assertNotIn("table", normalized_gt)
        self.assertNotIn("tr", normalized_gt)
        self.assertNotIn("td", normalized_gt)
        self.assertIn("political party", normalized_gt)
        self.assertIn("86,092", normalized_gt)
        self.assertEqual(normalized_gt, normalized_pred)

    def test_table_goal_metrics_report_teds_and_teds_s_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "doc_table.md").write_text(
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                encoding="utf-8",
            )
            (pred_dir / "doc_table.md").write_text(
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                encoding="utf-8",
            )
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "score": {
                                "overall_mean": 0.6,
                                "nid_mean": 1.0,
                                "teds_mean": 0.5,
                                "teds_s_mean": 0.75,
                                "mhs_mean": None,
                            },
                            "missing_predictions": 0,
                        },
                        "documents": [
                            {
                                "document_id": "doc_table",
                                "scores": {
                                    "overall": 0.6,
                                    "nid": 1.0,
                                    "teds": 0.5,
                                    "teds_s": 0.75,
                                    "mhs": None,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        score = payload["metrics"]["goal_score"]
        official = payload["metrics"]["official_reference"]
        self.assertEqual(score["table_teds_mean"], 0.5)
        self.assertEqual(score["table_teds_s_mean"], 0.75)
        self.assertEqual(score["table_quality_mean"], 0.75)
        self.assertEqual(official["teds_mean"], 0.5)
        self.assertEqual(official["teds_s_mean"], 0.75)

    def test_heading_markdown_score_is_diagnostic_not_comprehensive_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "doc_heading.md").write_text(
                "# Section A\n\nThe recovered body content is identical.\n",
                encoding="utf-8",
            )
            (pred_dir / "doc_heading.md").write_text(
                "Section A\n\nThe recovered body content is identical.\n",
                encoding="utf-8",
            )
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "score": {
                                "overall_mean": 0.5,
                                "nid_mean": 1.0,
                                "teds_mean": None,
                                "mhs_mean": 0.0,
                            },
                            "missing_predictions": 0,
                        },
                        "documents": [
                            {
                                "document_id": "doc_heading",
                                "scores": {
                                    "overall": 0.5,
                                    "nid": 1.0,
                                    "teds": None,
                                    "teds_s": None,
                                    "mhs": 0.0,
                                    "mhs_s": 0.0,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        score = payload["metrics"]["goal_score"]
        self.assertEqual(score["core_text_mean"], 1.0)
        self.assertEqual(score["comprehensive_mean"], 1.0)
        self.assertEqual(payload["weak_samples"], [])

    def test_autoind_report_wrappers_and_structured_toc_are_ignored(self) -> None:
        ground_truth = """# Contents

1. Front Matter 1
2. Introduction 3

# Front Matter

The submitted content starts here.
"""
        prediction = """### 解析目录结构

#### sample.pdf

- Contents（目录页: 1；目录项 2 条）
  - 1.0 Front Matter（目录页: 1；定位页码: 1）
  - 2.0 Introduction（目录页: 1；定位页码: 3）

### 正文结构化内容

Front Matter

The submitted content starts here.
"""

        self.assertEqual(
            normalize_markdown_for_goal_text(ground_truth),
            normalize_markdown_for_goal_text(prediction),
        )

    def test_toc_only_documents_do_not_reduce_core_goal_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "toc_only.md").write_text(
                "# Table of contents\n\nIntroduction 7\nMethods 12\n\n5\n",
                encoding="utf-8",
            )
            (pred_dir / "toc_only.md").write_text(
                "### 解析目录结构\n\n#### toc_only.pdf\n\n- Table of contents（目录页: 1；目录项 2 条）\n"
                "  - Introduction（目录页: 1；定位页码: 7）\n"
                "  - Methods（目录页: 1；定位页码: 12）\n\n"
                "### 正文结构化内容\n",
                encoding="utf-8",
            )
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {"score": {"overall_mean": 0.0}, "missing_predictions": 0},
                        "documents": [
                            {
                                "document_id": "toc_only",
                                "scores": {
                                    "overall": 0.0,
                                    "nid": 0.0,
                                    "teds": None,
                                    "teds_s": None,
                                    "mhs": 0.0,
                                    "mhs_s": 0.0,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        score = payload["metrics"]["goal_score"]
        self.assertEqual(score["core_text_mean"], 1.0)
        self.assertEqual(score["comprehensive_mean"], 1.0)
        self.assertEqual(payload["weak_samples"], [])

    def test_toc_page_residual_with_recovered_entries_is_scored_as_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "toc_page.md").write_text(
                """MOHAVE COMMUNITY COLLEGE

BIO181

Genetics Lab - Blood Disorders .............................................................................. 94
Human Traits Governed by Mendelian Genetics................................................... 97
1. Record your phenotype and genotype for the following Mendelian traits:.. 97
Human Traits not Governed by Mendelian Genetics ............................................ 98
Human Genetics Problems ................................................................................... 100
Pedigree Analysis ................................................................................................. 102
Practice Problems................................................................................................. 102
Lab Materials......................................................................................................... 104
Contributors and Attributions .............................................................................. 104
From Gene to Protein via Transcription and Translation.................................... 105

2
""",
                encoding="utf-8",
            )
            (pred_dir / "toc_page.md").write_text(
                """### ???????????

MOHAVE COMMUNITY COLLEGE BIO181 Genetics Lab - Blood Disorders .............................................................................. 94 Human Traits Governed by Mendelian Genetics................................................... 97

1. Record your phenotype and genotype for the following Mendelian traits: .. 97

Human Traits not Governed by Mendelian Genetics ............................................ 98 Human Genetics Problems ................................................................................... 100 Pedigree Analysis ................................................................................................. 102 Practice Problems ................................................................................................. 102 Lab Materials......................................................................................................... 104 Contributors and Attributions .............................................................................. 104 From Gene to Protein via Transcription and Translation .................................... 105
""",
                encoding="utf-8",
            )
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {"score": {"overall_mean": 0.0}, "missing_predictions": 0},
                        "documents": [
                            {
                                "document_id": "toc_page",
                                "scores": {
                                    "overall": 0.0,
                                    "nid": 0.0,
                                    "teds": None,
                                    "teds_s": None,
                                    "mhs": 0.0,
                                    "mhs_s": 0.0,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        score = payload["metrics"]["goal_score"]
        self.assertEqual(score["core_text_mean"], 1.0)
        self.assertEqual(score["comprehensive_mean"], 1.0)
        self.assertEqual(payload["weak_samples"], [])

    def test_plain_toc_page_and_autoind_structured_toc_projection_are_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "toc_structured.md").write_text(
                """MOHAVE COMMUNITY COLLEGE

BIO181

# Table of Contents

Measurement Lab worksheet...................................................................................... 3
Scientific Method Lab.................................................................................................. 6
Chemistry of the Cell ~ But this is biology!........................................... 9
Biological Macromolecules and Their Indicators............................. 10
Worksheet for Chemistry of the Cell ....................................................... 12
How molecules move in a liquid............................................................................. 12

1
""",
                encoding="utf-8",
            )
            (pred_dir / "toc_structured.md").write_text(
                """### 解析目录结构

#### toc_structured.pdf

- Table of Contents（目录页: 1；目录项 6 条）
  - Measurement Lab worksheet（目录页: 1；定位页码: 3）
  - Scientific Method Lab（目录页: 1；定位页码: 6）
  - Chemistry of the Cell ~ But this is biology!（目录页: 1；定位页码: 9）
  - Biological Macromolecules and Their Indicators（目录页: 1；定位页码: 10）
  - Worksheet for Chemistry of the Cell（目录页: 1；定位页码: 12）
  - How molecules move in a liquid（目录页: 1；定位页码: 12）

### 正文结构化内容
""",
                encoding="utf-8",
            )
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {"score": {"overall_mean": 0.0}, "missing_predictions": 0},
                        "documents": [
                            {
                                "document_id": "toc_structured",
                                "scores": {
                                    "overall": 0.0,
                                    "nid": 0.0,
                                    "teds": None,
                                    "teds_s": None,
                                    "mhs": 0.0,
                                    "mhs_s": 0.0,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        score = payload["metrics"]["goal_score"]
        self.assertEqual(score["core_text_mean"], 1.0)
        self.assertEqual(score["comprehensive_mean"], 1.0)
        self.assertEqual(payload["weak_samples"], [])

    def test_plain_toc_page_and_table_projected_toc_are_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "toc_table_projected.md").write_text(
                """# Table of Contents

Executive Summary 4
Legal Framework 6
Election Administration 11
Civil Society Engagement 15
Political Parties, Candidates Registration and Election 18
Campaign
Media Freedom and Access to Information 25
Voter Education and Awareness 29
Participation of Marginalized Sectors 31
Recommendations 39
""",
                encoding="utf-8",
            )
            (pred_dir / "toc_table_projected.md").write_text(
                """**Table of Contents**

| Executive | Summary | 4 |
| --- | --- | --- |
| Legal | Framework | 6 |
| Election | Administration | 11 |
| Civil Society | Engagement | 15 |

**Table of Contents**

| Political Parties, Candidates Registration and Election | Column 2 | 18 |
| --- | --- | --- |
| Campaign |  |  |
| Media | Freedom and Access to Information | 25 |
| Voter | Education and Awareness | 29 |
| Participation of | Marginalized Sectors | 31 |

39

Recommendations
""",
                encoding="utf-8",
            )
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {"score": {"overall_mean": 0.0}, "missing_predictions": 0},
                        "documents": [
                            {
                                "document_id": "toc_table_projected",
                                "scores": {
                                    "overall": 0.0,
                                    "nid": 0.0,
                                    "teds": None,
                                    "teds_s": None,
                                    "mhs": 0.0,
                                    "mhs_s": 0.0,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        score = payload["metrics"]["goal_score"]
        self.assertEqual(score["core_text_mean"], 1.0)
        self.assertEqual(score["comprehensive_mean"], 1.0)
        self.assertEqual(payload["weak_samples"], [])

    def test_numbered_infographic_is_not_toc_projection_equivalent(self) -> None:
        ground_truth = """# 10 THINGS YOU SHOULD KNOW ABOUT

# COPYRIGHT

1

Creative work belongs to its author.

2

Copyright protects creative work.

3

Fair use has limits.
"""
        prediction = """10 THINGS YOU SHOULD KNOW ABOUT

Creative work belongs to its author.

2 Copyright protects creative work. 3 Fair use has limits.
"""

        self.assertNotEqual(
            normalize_markdown_for_goal_text(ground_truth),
            normalize_markdown_for_goal_text(prediction),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_dir = root / "ground-truth"
            pred_dir = root / "prediction"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "infographic.md").write_text(ground_truth, encoding="utf-8")
            (pred_dir / "infographic.md").write_text(prediction, encoding="utf-8")
            eval_path = root / "evaluation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "metrics": {"score": {"overall_mean": 0.0}, "missing_predictions": 0},
                        "documents": [
                            {
                                "document_id": "infographic",
                                "scores": {
                                    "overall": 0.0,
                                    "nid": 0.0,
                                    "teds": None,
                                    "teds_s": None,
                                    "mhs": 0.0,
                                    "mhs_s": 0.0,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compute_goal_metrics(
                evaluation_json=eval_path,
                ground_truth_dir=gt_dir,
                prediction_dir=pred_dir,
            )

        self.assertLess(payload["metrics"]["goal_score"]["core_text_mean"], 1.0)

    def test_running_numbered_body_text_is_not_plain_toc_equivalent(self) -> None:
        ground_truth = (
            "As creators, we take photos, write songs, make videos, etc. 2 "
            "Copyright protects creative work, so people cannot generally copy work. "
            "BUT COPYRIGHT DOESNT COVER EVERYTHING 6 Copyright gives protection but has limitations. "
            "which allows us to re-use copyrighted work in fair ways. "
            "Works in the public domain are free to re-use and share however you want. 10 "
            "Some creators are happy to share their creative work."
        )
        prediction = (
            "As creators, we are US Government documents, like NASA photos and reports by federal agencies. "
            "generally copy or share or perform other work 4 "
            "Works in the public protects all of us and domain are free to share. 5 "
            "Downloading music, movies, ebooks, or games creative work. They use a licensing system."
        )

        self.assertFalse(_documents_are_toc_projection_equivalent(ground_truth, prediction))

    def test_table_regression_baseline_fails_when_s226_table_metrics_drop(self) -> None:
        baseline = {
            "metrics": {
                "goal_score": {
                    "core_text_mean": 0.947789,
                    "table_teds_mean": 0.945294,
                    "table_teds_s_mean": 0.969348,
                }
            }
        }
        current = {
            "metrics": {
                "goal_score": {
                    "core_text_mean": 0.947789,
                    "table_teds_mean": 0.944,
                    "table_teds_s_mean": 0.969348,
                }
            }
        }

        result = compare_table_regression_baseline(current, baseline)

        self.assertFalse(result["passed"])
        self.assertEqual(result["baseline_id"], "opendataloader_200_s226_table_convergence")
        self.assertEqual(result["failures"][0]["metric"], "table_teds_mean")
        self.assertEqual(result["failures"][0]["baseline"], 0.945294)
        self.assertEqual(result["failures"][0]["current"], 0.944)

    def test_table_regression_baseline_allows_equal_or_better_s226_table_metrics(self) -> None:
        baseline = {
            "metrics": {
                "goal_score": {
                    "core_text_mean": 0.947789,
                    "table_teds_mean": 0.945294,
                    "table_teds_s_mean": 0.969348,
                }
            }
        }
        current = {
            "metrics": {
                "goal_score": {
                    "core_text_mean": 0.947789,
                    "table_teds_mean": 0.945294,
                    "table_teds_s_mean": 0.970,
                }
            }
        }

        result = compare_table_regression_baseline(current, baseline)

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_table_regression_baseline_fails_when_s230_core_text_drops(self) -> None:
        baseline = {
            "metrics": {
                "goal_score": {
                    "core_text_mean": 0.947789,
                    "table_teds_mean": 0.945294,
                    "table_teds_s_mean": 0.969348,
                }
            }
        }
        current = {
            "metrics": {
                "goal_score": {
                    "core_text_mean": 0.947,
                    "table_teds_mean": 0.945294,
                    "table_teds_s_mean": 0.969348,
                }
            }
        }

        result = compare_table_regression_baseline(current, baseline)

        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["metric"], "core_text_mean")
        self.assertEqual(result["failures"][0]["baseline"], 0.947789)
        self.assertEqual(result["failures"][0]["current"], 0.947)


if __name__ == "__main__":
    unittest.main()
