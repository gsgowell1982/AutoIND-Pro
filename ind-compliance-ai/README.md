# IND Compliance AI

IND Compliance AI is an engineering platform for IND submission compliance support.
It focuses on structured parsing, auditable artifacts, and explainable risk signals.

> Current release: **v1.0.0**

## Non-approval statement

This project does **not** replace regulatory judgment or approval decisions.
It provides compliance support signals, traceable evidence, and risk explanations only.

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
├── api/                         # FastAPI service and job orchestration
├── core/
│   └── run_manager.py           # run evidence package lifecycle
├── parsers/
│   ├── pdf_parser.py            # stable PDF parser entrypoint
│   ├── pdf/                     # modular PDF implementation
│   │   ├── pipeline.py
│   │   ├── postprocess.py
│   │   ├── tables.py
│   │   ├── image_blocks.py
│   │   ├── text_blocks.py
│   │   ├── layout.py
│   │   ├── shared.py
│   │   └── types.py
│   ├── docx_parser.py
│   ├── pptx_parser.py
│   ├── xml_parser.py
│   └── common/
├── ui/frontend/                 # React + Vite workbench UI
├── runs/                        # generated run artifacts (git ignored)
├── output/parsed_markdown/      # generated markdown outputs
└── main.py                      # dev bootstrap / API mode entry
```

## Quick start

```bash
cp .env.example .env
poetry install
python3 main.py
```

Default dev endpoints:

- UI: `http://localhost:5173`
- API: `http://localhost:8000`

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
- If parser dependencies are missing:
  ```bash
  pip install python-docx python-pptx pymupdf fastapi uvicorn python-multipart
  ```
- For legacy `.doc`, convert to `.docx` for best quality when possible.

## Processing flow

Upload -> Parse -> Normalize -> Atomic Facts -> Consistency Checks -> Output

See `docs/architecture.md` and `docs/phase1_scope.md` for broader context.
