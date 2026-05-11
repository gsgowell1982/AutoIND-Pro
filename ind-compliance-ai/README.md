# IND Compliance AI

IND Compliance AI is an engineering platform for IND submission compliance support.
It focuses on structured parsing, auditable artifacts, and explainable risk signals.

> Current release: **v1.0.1**

## Non-approval statement

This project does **not** replace regulatory judgment or approval decisions.
It provides compliance support signals, traceable evidence, and risk explanations only.

## v1.0.1 updates

### 1) PDF table pipeline hardening (enterprise IND scenarios)

- Introduced configurable parser policy at `config/pdf_parser.toml` via `parsers/pdf/settings.py`.
- Added non-destructive table-content policy and semantic rule-engine policy switches.
- Added table-detection policy for supplemental candidate merge and two-column guard.

### 2) Continuum modularization and maintainability

- Added semantic rule-engine and semantic repair modules:
  - `parsers/pdf/table_modules/continuum/rules_engine.py`
  - `parsers/pdf/table_modules/continuum/semantic_orchestration.py`
  - `parsers/pdf/table_modules/continuum/semantic_repairs.py`
- Added layout/projection modules to reduce `normalization.py` complexity:
  - `parsers/pdf/table_modules/continuum/column_layout.py`
  - `parsers/pdf/table_modules/continuum/row_projection.py`
  - `parsers/pdf/table_modules/continuum/legacy_supplement.py`
- Kept extraction non-destructive by default while preserving auditable diagnostics.

### 3) Borderless and two-column handling strategy

- Supplemental word-clustering candidate path now runs conservatively in parallel.
- Strict overlap de-dup protects already-correct PyMuPDF table detections.
- Added two-column layout guard and tabular-strength scoring to reduce false positives in literature-style PDFs.

## v1.0.0 implemented capabilities

### 1) Upload and job orchestration

- Multi-file upload with per-file status and task progress.
- Asynchronous background processing (FastAPI + BackgroundTasks).
- Job status APIs with parse outputs and run identifiers.

### 2) Multi-format parsing

- Supported formats: `pdf`, `doc/docx`, `ppt/pptx`, `xml`.
- Normalized parser outputs include:
  - `text`
  - `atomic_facts`
  - `metadata`
  - format-specific structural fields (`pages/slides/document_ast/table_asts/image_blocks`, etc.)

### 3) PDF parsing (modularized architecture)

PDF parsing is now decoupled into dedicated modules under `parsers/pdf/`:

- `shared.py`: shared text/bbox utilities and word model.
- `layout.py`: word extraction and drawing/path helpers.
- `text_blocks.py`: text block extraction, semantic merge, dedup, header/footer filtering.
- `image_blocks.py`: image block handling, OCR fallback, figure-title assignment.
- `tables.py`: table row/column clustering, table AST build, validation, merge, cross-page stitching.
- `pipeline.py`: page-by-page parse pipeline orchestration.
- `postprocess.py`: output assembly, markdown-facing payload shaping.
- `types.py`: pipeline state/counters and parser constants.

`parsers/pdf_parser.py` remains the stable external entrypoint.

#### PDF table recognition highlights

- Table AST with explicit row/col/cell relationships.
- Single-column and multi-column table support.
- Multi-line cell merge, row/column span handling.
- Same-page fragment merge + same-title merge.
- Cross-page table stitching via structure similarity + context heuristics.
- Continuation metadata:
  - `continued_from`, `continued_to`
  - `continuation_hint`
  - `continuation_source` (source table, strategy, inherited fields, similarity)
- Sequential table id renumbering (`tbl_001`, `tbl_002`, ...).

#### PDF image recognition highlights

- Image blocks are preserved as first-class nodes with `page + bbox`.
- Figure reference generation (`Figure X -> image_id`).
- OCR/text-layer/path-based image-vs-text correction.

### 4) Audit Workbench (frontend)

- PDF viewer with bounding-box highlighting.
- Structured markdown executive summary.
- Full markdown export/download for complete review content.
- Cross-document consistency panel for atomic-fact alignment.

> The old "Content Alignment Preview (Plain Text, Up to 500 Chars)" block has been removed to avoid partial-content bias.

### 5) Runs evidence package (audit & reproducibility)

Each run creates an immutable folder under `runs/`, e.g.
`runs/run_YYYY-MM-DD_HHMMSS_xxxxxxxx/`, including:

- `manifest.json`
- `artifacts/ast/*`
- `artifacts/tables/*.json`
- `artifacts/images/*.json`
- `artifacts/normalized/material_normalized.json`
- `artifacts/atomic_facts/atomic_facts.json`
- `output/compliance_result.json`
- `output/audit_log.json`
- `logs/pipeline.log`

## Project structure (current)

```text
ind-compliance-ai/
鈹溾攢鈹€ api/                         # FastAPI service and job orchestration
鈹溾攢鈹€ core/
鈹?  鈹斺攢鈹€ run_manager.py           # run evidence package lifecycle
鈹溾攢鈹€ parsers/
鈹?  鈹溾攢鈹€ pdf_parser.py            # stable PDF parser entrypoint
鈹?  鈹溾攢鈹€ pdf/                     # modular PDF implementation
鈹?  鈹?  鈹溾攢鈹€ pipeline.py
鈹?  鈹?  鈹溾攢鈹€ postprocess.py
鈹?  鈹?  鈹溾攢鈹€ tables.py
鈹?  鈹?  鈹溾攢鈹€ image_blocks.py
鈹?  鈹?  鈹溾攢鈹€ text_blocks.py
鈹?  鈹?  鈹溾攢鈹€ layout.py
鈹?  鈹?  鈹溾攢鈹€ shared.py
鈹?  鈹?  鈹斺攢鈹€ types.py
鈹?  鈹溾攢鈹€ docx_parser.py
鈹?  鈹溾攢鈹€ pptx_parser.py
鈹?  鈹溾攢鈹€ xml_parser.py
鈹?  鈹斺攢鈹€ common/
鈹溾攢鈹€ ui/frontend/                 # React + Vite workbench UI
鈹溾攢鈹€ runs/                        # generated run artifacts (git ignored)
鈹溾攢鈹€ output/parsed_markdown/      # generated markdown outputs
鈹斺攢鈹€ main.py                      # dev bootstrap / API mode entry
```

## Quick start

```bash
cp .env.example .env
poetry install
python3 main.py
```

### Windows bootstrap without Poetry

If you are running this project from `D:\AutoIND-Pro` on Windows and do not have Poetry installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python .\scripts\validate_environment.py
.\.venv\Scripts\python .\main.py
```

For API-only validation without frontend tooling:

```powershell
.\.venv\Scripts\python .\scripts\validate_environment.py --skip-frontend
.\.venv\Scripts\python .\main.py --mode api
```

Or use the repository-level launcher:

```powershell
..\scripts\run-ind-compliance-ai.ps1
```

Default dev endpoints:

- UI: `http://localhost:5173`
- API: `http://localhost:8000`

### PDF regression gate

Before merging parser changes that affect PDF layout, reading order, footer filtering,
TOC extraction, or table detection, run the standing PDF regression gate:

```powershell
.\.venv\Scripts\python .\scripts\run_pdf_sample_regression.py
```

This gate covers the current enterprise anchor samples:

- `A-tst.pdf` for journal-style two-column parsing, numbered display equations, vector-drawn figures, and algorithm/pseudocode structure regression
- `2-column-tst.pdf` for literature-style double-column parsing
- `test-ind.pdf` for figure/footer/TOC protection
- `eCTD鎶€鏈鑼?pdf` for eCTD TOC and cross-page structure regression

The current `A-tst.pdf` protected expectations include:

- page-3 two-column top-body integrity plus 7 numbered display equations `(1)` to `(7)`
- page-3 inline-math prose preservation for the `Y = [...] ... d ≫ n` sentence
- page-4 six numbered display equations `(8)` to `(13)` with continuation cleanup, symbol-fragment absorption for `(11)` / `(13)`, and no lower-right equation residue
- page-4/page-5 algorithm pseudocode emitted as `algorithm_blocks` and kept out of TOC/table false positives
- page-7 figure bbox isolation from caption/footer/body-tail text while still covering the real top portion of the chart
- page-8/page-9 vector-drawn `Fig. 2` / `Fig. 3` figure recovery as full multi-row panel figures, even without raw raster image blocks

## API-only mode

```bash
python3 main.py --mode api
```

## Frontend dependency sync options

If local startup appears stuck at `Syncing frontend dependencies (npm install) ...`:

- Skip auto install for this run:
  ```bash
  python3 main.py --skip-frontend-install
  ```
- Increase install timeout:
  ```bash
  python3 main.py --frontend-install-timeout 1800
  ```

## Environment notes

- Ensure `node`/`npm` are in PATH for local frontend startup.
- Use `Python 3.11` or `Python 3.12`. `Python 3.14` is not supported by this project.
- If parser dependencies are missing:
  ```bash
  pip install -r requirements.txt
  ```
- For legacy `.doc`, convert to `.docx` for best quality when possible.

## Processing flow

Upload -> Parse -> Normalize -> Atomic Facts -> Consistency Checks -> Output

See `docs/architecture.md` and `docs/phase1_scope.md` for broader context.

