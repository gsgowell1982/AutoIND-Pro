# IND Compliance AI (Phase 1 Scaffold)

IND Compliance AI is a structured engineering foundation for a regulatory support platform focused on China IND submissions first, with future expansion to US and other regions.

## Phase 1 goals

- Build a deterministic project structure and control points.
- Define parsing, rule, and risk-analysis module boundaries.
- Establish configuration, schema, and audit-ready output contracts.
- Keep decision authority with human regulatory professionals.

## Non-approval statement

This project does **not** replace regulatory judgment or approval decisions.
It provides compliance support signals, traceable evidence, and risk explanations only.

## Quick start

```bash
cp .env.example .env
python3 scripts/validate_environment.py
```

## Core processing flow

Upload -> Parse -> Rule Evaluation -> Cross-Module Checks -> Risk Output

See `docs/architecture.md` and `docs/phase1_scope.md` for details.
