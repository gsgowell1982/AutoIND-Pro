# IND Compliance AI (Phase 1 Prototype)

IND Compliance AI is a structured engineering foundation for a regulatory support platform focused on China IND submissions first, with future expansion to US and other regions.

## Phase 1 goals

- Build a deterministic project structure and control points.
- Define parsing, rule, and risk-analysis module boundaries.
- Establish configuration, schema, and audit-ready output contracts.
- Keep decision authority with human regulatory professionals.

## Non-approval statement

This project does **not** replace regulatory judgment or approval decisions.
It provides compliance support signals, traceable evidence, and risk explanations only.

## Phase 1 implemented modules

1. **文件上传与预处理区**
   - 支持上传 `pdf/doc/docx/ppt/pptx`
   - 展示任务级与文件级解析进度
2. **智能审核工作台 (Audit Workbench)**
   - 左区：PDF 阅读 + Bounding Box 高亮展示
   - 中区：标准化 Markdown 视图（企业级量化指标总结 + 第一页预览）
   - 提供完整解析 Markdown 下载（全量内容用于审阅留档）
   - 右区：AI 规则检查列表占位（Phase 1 仅保留 UI 位置）
3. **一致性检查看板**
   - 展示跨模块原子事实比对
   - 示例：M3 与 M5 的批号一致性

## Quick start (recommended)

```bash
cp .env.example .env
poetry install
python3 main.py
```

After startup in default `dev` mode:

- UI: `http://localhost:5173`
- API: `http://localhost:8000`

## API-only mode

```bash
python3 main.py --mode api
```

## Windows + PyCharm + Conda notes

If `python main.py` fails with `FileNotFoundError` when launching frontend,
the Python environment cannot find `npm`.

Install Node.js into the same conda environment (or ensure npm is in PATH):

```bash
conda install -c conda-forge nodejs=22
```

Then verify:

```bash
node --version
npm --version
```

If parser dependencies are missing in the selected interpreter, install:

```bash
pip install python-docx python-pptx pymupdf fastapi uvicorn python-multipart
```

Legacy `.doc` notes:

- Best quality: upload `.docx`.
- For `.doc`, the parser tries Windows Word COM / LibreOffice / antiword when available.
- If none is available, the API will return a clear error asking to convert `.doc` to `.docx`.

## Core processing flow

Upload -> Parse -> Rule Evaluation -> Cross-Module Checks -> Risk Output

See `docs/architecture.md` and `docs/phase1_scope.md` for details.
