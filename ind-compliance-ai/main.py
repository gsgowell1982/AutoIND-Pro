from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "ui" / "frontend"
FRONTEND_DEPS_FINGERPRINT_FILE = ".deps_fingerprint"
DEFAULT_FRONTEND_INSTALL_TIMEOUT = 900
VECTOR_TABLE_OCR_RUNTIME_MODULES = ("numpy", "rapidocr_onnxruntime", "onnxruntime")
DEFAULT_PORT_SEARCH_LIMIT = 50


def _build_workspace_label(project_root: Path) -> str:
    normalized = str(project_root).lower()
    if "\\new code\\" in normalized:
        return f"mirror workspace: {project_root}"
    if normalized.startswith("d:\\autoind-pro"):
        return f"primary workspace: {project_root}"
    return f"workspace: {project_root}"


def _api_client_host(api_host: str) -> str:
    return "localhost" if api_host == "0.0.0.0" else api_host


def _port_probe_host(host: str) -> str:
    return "localhost" if host in {"0.0.0.0", "::"} else host


def _port_is_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _resolve_available_port(
    *,
    host: str,
    requested_port: int,
    auto_port: bool,
    label: str,
) -> int:
    if not auto_port:
        return requested_port

    probe_host = _port_probe_host(host)
    for port in range(requested_port, requested_port + DEFAULT_PORT_SEARCH_LIMIT):
        if not _port_is_reachable(probe_host, port):
            if port != requested_port:
                print(
                    f"{label} port {requested_port} is already reachable on {probe_host}; using {port} instead."
                )
            return port

    raise RuntimeError(
        f"No available {label} port found from {requested_port} to "
        f"{requested_port + DEFAULT_PORT_SEARCH_LIMIT - 1}."
    )


def _resolve_startup_ports(
    *,
    api_host: str,
    requested_api_port: int,
    requested_ui_port: int,
    auto_port: bool,
) -> tuple[int, int]:
    api_port = _resolve_available_port(
        host=api_host,
        requested_port=requested_api_port,
        auto_port=auto_port,
        label="API",
    )
    ui_port = _resolve_available_port(
        host="localhost",
        requested_port=requested_ui_port,
        auto_port=auto_port,
        label="UI",
    )
    return api_port, ui_port


def _probe_import(module_name: str) -> str | None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive startup surface
        return f"{module_name}: {type(exc).__name__}: {exc}"
    return None


def _collect_runtime_capability_warnings() -> tuple[list[str], list[str]]:
    failures = [
        failure
        for failure in (_probe_import(module_name) for module_name in VECTOR_TABLE_OCR_RUNTIME_MODULES)
        if failure
    ]
    if not failures:
        return [], []

    warnings = [
        (
            "PDF vector-table OCR fallback is unavailable in the current Python environment. "
            "Wide borderless tables such as 2-column-tst.pdf page 2 Table 1 may be missed. "
            "Install requirements.txt in the active environment."
        )
    ]
    return warnings, failures


def _print_runtime_capability_warnings(warnings: list[str], failures: list[str]) -> None:
    if not warnings:
        return
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)


def _resolve_npm_executable() -> str | None:
    """Resolve npm command across POSIX and Windows environments."""
    if os.name == "nt":
        for candidate in ("npm.cmd", "npm.exe", "npm"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None
    return shutil.which("npm")


def _frontend_needs_install(frontend_dir: Path) -> bool:
    """Install frontend deps only when dependency fingerprint changes."""
    node_modules_dir = frontend_dir / "node_modules"
    if not node_modules_dir.exists():
        return True

    fingerprint = _dependency_fingerprint(frontend_dir)
    if not fingerprint:
        return False

    installed_fingerprint = _read_installed_fingerprint(frontend_dir)
    if not installed_fingerprint:
        # Legacy workspace: avoid forcing an unnecessary reinstall.
        _write_installed_fingerprint(frontend_dir, fingerprint)
        return False
    return installed_fingerprint != fingerprint


def _dependency_fingerprint(frontend_dir: Path) -> str:
    hash_builder = hashlib.sha256()
    fingerprint_targets = [frontend_dir / "package-lock.json", frontend_dir / "package.json"]
    has_content = False
    for target in fingerprint_targets:
        if not target.exists():
            continue
        has_content = True
        hash_builder.update(target.name.encode("utf-8"))
        hash_builder.update(b"\0")
        hash_builder.update(target.read_bytes())
        hash_builder.update(b"\0")
    if not has_content:
        return ""
    return hash_builder.hexdigest()


def _read_installed_fingerprint(frontend_dir: Path) -> str:
    fingerprint_file = frontend_dir / FRONTEND_DEPS_FINGERPRINT_FILE
    if not fingerprint_file.exists():
        return ""
    return fingerprint_file.read_text(encoding="utf-8").strip()


def _write_installed_fingerprint(frontend_dir: Path, fingerprint: str) -> None:
    if not fingerprint:
        return
    fingerprint_file = frontend_dir / FRONTEND_DEPS_FINGERPRINT_FILE
    fingerprint_file.write_text(fingerprint, encoding="utf-8")


def _sync_frontend_dependencies(
    frontend_dir: Path,
    npm_executable: str,
    timeout_seconds: int,
) -> int:
    print("Syncing frontend dependencies (npm install --no-audit --no-fund) ...")
    install_cmd = [npm_executable, "install", "--no-audit", "--no-fund", "--progress=false"]
    try:
        result = subprocess.run(
            install_cmd,
            cwd=frontend_dir,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(
            (
                f"npm install timed out after {timeout_seconds}s.\n"
                "Try one of these:\n"
                "1) Run manually in ui/frontend: npm install --no-audit --no-fund\n"
                "2) Start without auto install: python main.py --skip-frontend-install\n"
                "3) If you are in mainland China, configure npm registry mirror first."
            ),
            file=sys.stderr,
        )
        return 1
    if result.returncode != 0:
        return result.returncode

    _write_installed_fingerprint(frontend_dir, _dependency_fingerprint(frontend_dir))
    return 0


def run_api_server(host: str, port: int) -> int:
    runtime_warnings, runtime_failures = _collect_runtime_capability_warnings()
    _print_runtime_capability_warnings(runtime_warnings, runtime_failures)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--no-access-log",
    ]
    return subprocess.call(command, cwd=PROJECT_ROOT)


def run_dev_stack(
    api_host: str,
    api_port: int,
    ui_port: int,
    skip_frontend_install: bool = False,
    frontend_install_timeout: int = DEFAULT_FRONTEND_INSTALL_TIMEOUT,
    auto_port: bool = True,
) -> int:
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

    if skip_frontend_install:
        print("Skipping frontend dependency sync (--skip-frontend-install enabled).")
    elif _frontend_needs_install(FRONTEND_DIR):
        install_code = _sync_frontend_dependencies(
            frontend_dir=FRONTEND_DIR,
            npm_executable=npm_executable,
            timeout_seconds=max(60, frontend_install_timeout),
        )
        if install_code != 0:
            return install_code

    runtime_warnings, runtime_failures = _collect_runtime_capability_warnings()
    _print_runtime_capability_warnings(runtime_warnings, runtime_failures)

    api_port, ui_port = _resolve_startup_ports(
        api_host=api_host,
        requested_api_port=api_port,
        requested_ui_port=ui_port,
        auto_port=auto_port,
    )

    env = os.environ.copy()
    api_client_host = _api_client_host(api_host)
    env["VITE_API_BASE_URL"] = f"http://{api_client_host}:{api_port}"
    env["VITE_WORKSPACE_LABEL"] = _build_workspace_label(PROJECT_ROOT)
    env["VITE_RUNTIME_WARNING"] = runtime_warnings[0] if runtime_warnings else ""

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
        "--no-access-log",
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
    print(f"Workspace:        {_build_workspace_label(PROJECT_ROOT)}")
    if auto_port:
        print("Auto-port mode:   enabled; occupied default ports are avoided.")
    if runtime_warnings:
        print("Runtime capability warnings are active in this session.", file=sys.stderr)
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
    parser.add_argument(
        "--no-auto-port",
        action="store_true",
        help="Disable automatic port fallback when the default API/UI ports are already occupied.",
    )
    parser.add_argument(
        "--skip-frontend-install",
        action="store_true",
        help="Skip npm install auto-sync before starting the dev stack.",
    )
    parser.add_argument(
        "--frontend-install-timeout",
        type=int,
        default=DEFAULT_FRONTEND_INSTALL_TIMEOUT,
        help="Timeout in seconds for automatic npm install.",
    )
    args = parser.parse_args()

    if args.mode == "api":
        return run_api_server(host=args.api_host, port=args.api_port)
    return run_dev_stack(
        api_host=args.api_host,
        api_port=args.api_port,
        ui_port=args.ui_port,
        skip_frontend_install=args.skip_frontend_install,
        frontend_install_timeout=args.frontend_install_timeout,
        auto_port=not args.no_auto_port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
