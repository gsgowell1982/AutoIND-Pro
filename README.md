# AutoIND-Pro
AutoIND-Pro is an enterprise-grade AI platform designed to accelerate IND applications. It automates regulatory compliance audits, assesses submission risks, and generates high-fidelity drug dossiers using domain-specific LLMs. Streamline your journey from R&amp;D data to submission-ready documents with precision and regulatory intelligence.

## Run `ind-compliance-ai`

The working application currently lives in `ind-compliance-ai/`.

From `D:\AutoIND-Pro`, start it with:

```powershell
.\scripts\run-ind-compliance-ai.ps1
```

This script:

- uses `Python 3.12`
- creates `ind-compliance-ai/.venv` if needed
- installs backend dependencies from `ind-compliance-ai/requirements.txt`
- validates Python packages and frontend tooling
- starts the backend and frontend through `ind-compliance-ai/main.py`

Useful variants:

```powershell
.\scripts\run-ind-compliance-ai.ps1 -Mode api
.\scripts\run-ind-compliance-ai.ps1 -SkipFrontendInstall
```
