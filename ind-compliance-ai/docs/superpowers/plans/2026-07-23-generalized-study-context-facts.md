# Generalized Study Context Facts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Make study-context parsing and absorbed-template transfer atomic, source-aware, and generalized so inline multi-field rows cannot create trailing duplicate bullets in IND tables.

**Architecture:** Reuse the existing context-fact record model, but strengthen its shared label-boundary parser and make template transfer compare atomic facts with source precedence. Preserve raw rows for audit, attach provenance to every transferred fact, and render only source-ordered visible facts.

**Tech Stack:** Python 3, PyMuPDF parser pipeline, `unittest`, Markdown export helpers.

---

### Task 1: Add failing unit tests for generalized inline-field parsing

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Write the failing tests**

Add tests that call the real private parser helpers:

```python
def test_study_context_parser_splits_inline_ind_fields(self):
    facts = postprocess._study_context_fact_records(
        "种属/品系：新西兰兔 剖腹产日：G29 CTD 中的位置：第6 卷，第200 页",
        source_ref="unit:row:1",
        source_owner="unit",
        page=112,
    )
    self.assertEqual(
        [(fact["label"], fact["value"]) for fact in facts],
        [
            ("种属/品系", "新西兰兔"),
            ("剖腹产日", "G29"),
            ("CTD 中的位置", "第6 卷，第200 页"),
        ],
    )

def test_absorbed_template_composite_field_uses_atomic_facts(self):
    facts = postprocess._absorbed_template_field_fact_records(
        {"label": "种属/品系", "value": "新西兰兔 剖腹产日：G29", "field_index": 1},
        row="种属/品系：新西兰兔 剖腹产日：G29",
        source_template_id="structure_template_unit",
        page=112,
        fallback_bbox=[0, 0, 10, 10],
    )
    self.assertEqual(
        [(fact["label"], fact["value"]) for fact in facts],
        [("种属/品系", "新西兰兔"), ("剖腹产日", "G29")],
    )
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression.GeneralizedStudyContextFactTests -v
```

Expected: the new parser test fails because `剖腹产日` is currently absorbed into the preceding value and the template helper returns a composite fact.

### Task 2: Implement shared generalized field-boundary recognition

**Files:**
- Modify: `parsers/pdf/postprocess.py:10837-10855, 10860-10920, 34958-34995`

- [ ] **Step 1: Extend canonical context-label recognition**

Add a maintained label family for lifecycle/event-day fields (including `剖腹产日`, `受孕日`, `终止妊娠日`, `采样日`, `给药起始日`, and `恢复期`) and use it from `_study_context_explicit_field_label`. Keep the existing schema labels and scoring; do not add page-specific branches.

- [ ] **Step 2: Make template fallback always prefer parsed atomic facts**

In `_absorbed_template_field_fact_records`, return all parsed atomic records whenever at least one reliable boundary is found and only use the structured field fallback when no atomic record can be parsed. This prevents a partially parsed inline row from becoming a long composite value.

- [ ] **Step 3: Run the unit tests to verify GREEN**

Run the Task 1 command and expect both tests to pass.

### Task 3: Add failing transfer and source-order regression tests

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add a transfer test with partial atomic coverage**

Construct a minimal table with visible facts for `种属/品系=新西兰兔`, `剖腹产日=G29`, and `F1 仔畜=5 mg/kg`, then call `_transfer_absorbed_template_study_context_to_business_table` with the composite template field. Assert no new visible block is transferred and the audit reports all candidate atomic facts covered.

- [ ] **Step 2: Add the page-112 end-to-end invariant**

Extend `test_page112_populated_reproductive_panel_has_single_business_owner` to assert:

```python
context_lines = [
    str(block.get("text") or "")
    for block in table.get("study_context_blocks", [])
    if str(block.get("text") or "").strip()
]
self.assertEqual(sum("种属/品系：新西兰兔" in line for line in context_lines), 1)
self.assertEqual(sum("剖腹产日：G29" in line for line in context_lines), 1)
self.assertEqual(context_lines[-1], "F1 仔畜：5 mg/kg")
```

- [ ] **Step 3: Run the new tests to verify RED**

Run the transfer test and the page-112 test. The transfer test should expose the composite-key mismatch; the page-112 test should fail because the synthetic row is currently last.

### Task 4: Implement conservative atomic transfer and provenance

**Files:**
- Modify: `parsers/pdf/postprocess.py:34996-35155`
- Modify: `api/main.py:3180-3225` only if ordering metadata needs a renderer guard

- [ ] **Step 1: Compare atomic fact keys and preserve source precedence**

Use the parsed fact list returned by `_absorbed_template_field_fact_records`. Transfer only missing atomic facts. If a candidate has the same canonical label as an existing populated fact but a different value, record a conflict in `study_context_transfer_audits` and do not append it as visible context.

- [ ] **Step 2: Keep ambiguous unsplit rows audit-only**

When no reliable atomic facts are parsed, keep the candidate in `ambiguous_rows` and do not create a visible `study_context` block.

- [ ] **Step 3: Preserve physical order for transferred facts**

Carry source row/bbox metadata into transferred blocks and insert them by the same source-order helper used by other positioned context segments. Do not append a no-position block after all visible context.

- [ ] **Step 4: Run transfer and page-112 tests to verify GREEN**

Run the targeted test class and the page-112 regression. Expected: no synthetic trailing duplicate; `F1 仔畜：5 mg/kg` remains last.

### Task 5: Broaden regression coverage and verify the full baseline

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`
- Modify: `tests/deterministic_tests/test_parse_markdown_export.py` only if the shared renderer needs a deterministic assertion

- [ ] **Step 1: Add parameterized-style table cases for punctuation and field order**

Cover full-width/half-width colons, semicolon-separated fields, reordered lifecycle labels, empty values, and unknown labels. Assert atomic facts, raw row preservation, and no unsafe transfer.

- [ ] **Step 2: Run focused suites**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression.GeneralizedStudyContextFactTests -v
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_page112_populated_reproductive_panel_has_single_business_owner -v
```

- [ ] **Step 3: Run the complete r2 regression suite**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression -v
```

Expected: all existing tests and new generalized context tests pass with no new warnings or duplicate visible facts.

- [ ] **Step 4: Review diff and record verification**

Run `git diff --check` and inspect only the touched parser, API, and test files. Do not revert unrelated user changes.
