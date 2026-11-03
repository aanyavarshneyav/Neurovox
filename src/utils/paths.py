"""Path configuration and resolution utilities for local and cloud execution.

Provides dynamic resolution for DATA_ROOT, CACHE_ROOT, CKPT_ROOT, and RESULTS_ROOT
via environment variables, ensuring identical execution across local macOS and
Google Colab environments without hardcoded user directories.
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
    return (PROJECT_ROOT / "data").resolve()


def get_cache_root() -> Path:
    """Return configured cache root path, defaulting to <PROJECT_ROOT>/cache."""
    custom = os.environ.get("CACHE_ROOT")
    if custom:
        return Path(custom).expanduser().resolve()
    return (PROJECT_ROOT / "cache").resolve()


def get_ckpt_root() -> Path:
    """Return configured checkpoint root path, defaulting to <PROJECT_ROOT>/checkpoints."""
    custom = os.environ.get("CKPT_ROOT")
    if custom:
        return Path(custom).expanduser().resolve()
    return (PROJECT_ROOT / "checkpoints").resolve()


def get_results_root() -> Path:
    """Return configured results root path, defaulting to <PROJECT_ROOT>/results."""
    custom = os.environ.get("RESULTS_ROOT")
    if custom:
        return Path(custom).expanduser().resolve()
    return (PROJECT_ROOT / "results").resolve()


def resolve_path(relative_or_absolute: str | Path, base_root: Path | None = None) -> Path:
    """Resolve a relative or absolute path against configured root directories.
    
    If relative_or_absolute is already absolute, it is returned directly.
    If base_root is explicitly provided, resolution is performed relative to base_root.
    Otherwise, if the first path segment matches a managed root folder ("data", "cache",
    "checkpoints", "results"), it is dynamically routed to the respective environment-aware
    root directory (DATA_ROOT, CACHE_ROOT, CKPT_ROOT, or RESULTS_ROOT). All other relative
    paths resolve against PROJECT_ROOT.
    """
    path = Path(relative_or_absolute).expanduser()
    if path.is_absolute():
        return path

    if base_root is not None:
        return (base_root / path).resolve()

    parts = path.parts
    if parts:
        first = parts[0]
        subpath = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        if first == "data":
            return (get_data_root() / subpath).resolve()
        if first == "cache":
            return (get_cache_root() / subpath).resolve()
        if first == "checkpoints":
            return (get_ckpt_root() / subpath).resolve()
        if first == "results":
            return (get_results_root() / subpath).resolve()

    return (PROJECT_ROOT / path).resolve()


__all__ = [
    "PROJECT_ROOT",
    "get_data_root",
    "get_cache_root",
    "get_ckpt_root",
    "get_results_root",
    "resolve_path",
]
