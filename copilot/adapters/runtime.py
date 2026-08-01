from __future__ import annotations

import os
from pathlib import Path


DEFAULT_YFINANCE_PROXY_URL = "http://127.0.0.1:7890"

_ENV_LOADED = False


def load_project_env_files() -> None:
    """Load project env files without overriding values already in the process."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    package_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()
    env_paths = _unique_paths(
        [
            cwd / ".env",
            package_root / ".env",
        ]
    )

    try:
        from dotenv import load_dotenv
    except Exception:
        for path in env_paths:
            _load_env_file_fallback(path)
    else:
        for path in env_paths:
            if path.exists():
                load_dotenv(path, override=False)

    _ENV_LOADED = True


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique.append(path)
        seen.add(resolved)
    return unique


def ensure_yfinance_proxy() -> None:
    """Configure yfinance proxy defaults for Copilot adapter calls."""
    load_project_env_files()
    proxy_url = os.getenv("YFINANCE_PROXY_URL", "").strip() or DEFAULT_YFINANCE_PROXY_URL
    for key in ("HTTP_PROXY", "HTTPS_PROXY"):
        if not os.getenv(key):
            os.environ[key] = proxy_url


def _load_env_file_fallback(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
