try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - scaffold fallback
    FastAPI = None  # type: ignore[assignment]

from api.analysis_controller import run_analysis


def create_app() -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed; install dependencies before running API")

    app = FastAPI(title="IND Compliance AI", version="0.1.0")

    @app.get('/health')
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post('/analyze')
    def analyze(file_path: str, submission_profile: str) -> dict[str, object]:
        return run_analysis(file_path=file_path, submission_profile=submission_profile)

    return app


app = create_app() if FastAPI is not None else None
