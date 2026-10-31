"""Path configuration and resolution utilities for local and cloud execution.

Provides dynamic resolution for DATA_ROOT, CACHE_ROOT, and CKPT_ROOT via environment
variables, ensuring identical execution across local macOS and Google Colab environments.
"""

import os
from pathlib import Path

# Project root: directory containing src/ and pyproject.toml
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    """Return configured data root path, defaulting to <PROJECT_ROOT>/data."""
    custom = os.environ.get("DATA_ROOT")
    if custom:
        return Path(custom).expanduser().resolve()
    return PROJECT_ROOT / "data"


def get_cache_root() -> Path:
    """Return configured cache root path, defaulting to <PROJECT_ROOT>/cache."""
    custom = os.environ.get("CACHE_ROOT")
    if custom:
        return Path(custom).expanduser().resolve()
    return PROJECT_ROOT / "cache"


def get_ckpt_root() -> Path:
    """Return configured checkpoint root path, defaulting to <PROJECT_ROOT>/checkpoints."""
    custom = os.environ.get("CKPT_ROOT")
    if custom:
        return Path(custom).expanduser().resolve()
    return PROJECT_ROOT / "checkpoints"


def resolve_path(relative_or_absolute: str | Path, base_root: Path | None = None) -> Path:
    """Resolve a path against a base root if not already absolute."""
    path = Path(relative_or_absolute).expanduser()
    if path.is_absolute():
        return path
    root = base_root if base_root is not None else PROJECT_ROOT
    return (root / path).resolve()

