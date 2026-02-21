from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "ui" / "frontend"


def _resolve_npm_executable() -> str | None:
    """Resolve npm command across POSIX and Windows environments."""
    if os.name == "nt":
        for candidate in ("npm.cmd", "npm.exe", "npm"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None
    return shutil.which("npm")


def run_api_server(host: str, port: int) -> int:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.call(command, cwd=PROJECT_ROOT)


def run_dev_stack(api_host: str, api_port: int, ui_port: int) -> int:
    if not (FRONTEND_DIR / "package.json").exists():
        print("Frontend package.json is missing in ui/frontend.", file=sys.stderr)
        return 1
    npm_executable = _resolve_npm_executable()
    if npm_executable is None:
        print(
            "npm executable was not found. Install Node.js and ensure npm is in PATH.\n"
            "Conda example: `conda install -c conda-forge nodejs=22`",
            file=sys.stderr,
        )
        return 1

    if not (FRONTEND_DIR / "node_modules").exists():
        print("Installing frontend dependencies in ui/frontend ...")
        install_code = subprocess.call([npm_executable, "install"], cwd=FRONTEND_DIR)
        if install_code != 0:
            return install_code

    env = os.environ.copy()
    api_client_host = "localhost" if api_host == "0.0.0.0" else api_host
    env["VITE_API_BASE_URL"] = f"http://{api_client_host}:{api_port}"

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        api_host,
        "--port",
        str(api_port),
        "--reload",
    ]
    frontend_cmd = [
        npm_executable,
        "run",
        "dev",
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        str(ui_port),
    ]

    backend = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT)
    frontend = subprocess.Popen(frontend_cmd, cwd=FRONTEND_DIR, env=env)

    print(f"API running at:   http://{api_client_host}:{api_port}")
    print(f"UI running at:    http://localhost:{ui_port}")
    print("Press Ctrl+C to stop both services.")

    try:
        while True:
            backend_code = backend.poll()
            frontend_code = frontend.poll()
            if backend_code is not None:
                return backend_code
            if frontend_code is not None:
                return frontend_code
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in (frontend, backend):
            if process.poll() is None:
                process.terminate()
        for process in (frontend, backend):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="IND Compliance AI entrypoint")
    parser.add_argument(
        "--mode",
        choices=["dev", "api"],
        default="dev",
        help="dev: run backend + React UI; api: run only FastAPI.",
    )
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=5173)
    args = parser.parse_args()

    if args.mode == "api":
        return run_api_server(host=args.api_host, port=args.api_port)
    return run_dev_stack(api_host=args.api_host, api_port=args.api_port, ui_port=args.ui_port)


if __name__ == "__main__":
    raise SystemExit(main())
