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
   - 中区：结构化 Markdown 视图（由解析 JSON 转化）
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

## Core processing flow

Upload -> Parse -> Rule Evaluation -> Cross-Module Checks -> Risk Output

See `docs/architecture.md` and `docs/phase1_scope.md` for details.
