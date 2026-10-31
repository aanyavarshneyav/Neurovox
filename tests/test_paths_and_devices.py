"""Unit tests for path resolution, accelerator device selection, and notebook integrity."""

import json
import os
import re
from pathlib import Path
import pytest
import torch

from src.utils.paths import (
    PROJECT_ROOT,
    get_cache_root,
    get_ckpt_root,
    get_data_root,
    resolve_path,
)
from scripts.check_env import get_accelerator_device


def test_default_paths():
    """Verify default paths resolve relative to project root when env vars are unset."""
    orig_data = os.environ.pop("DATA_ROOT", None)
    orig_cache = os.environ.pop("CACHE_ROOT", None)
    orig_ckpt = os.environ.pop("CKPT_ROOT", None)

    try:
        assert get_data_root() == PROJECT_ROOT / "data"
        assert get_cache_root() == PROJECT_ROOT / "cache"
        assert get_ckpt_root() == PROJECT_ROOT / "checkpoints"
    finally:
        if orig_data:
            os.environ["DATA_ROOT"] = orig_data
        if orig_cache:
            os.environ["CACHE_ROOT"] = orig_cache
        if orig_ckpt:
            os.environ["CKPT_ROOT"] = orig_ckpt


def test_env_override_paths(tmp_path):
    """Verify environment variables dynamically override default paths."""
    custom_data = tmp_path / "custom_data"
    custom_cache = tmp_path / "custom_cache"
    custom_ckpt = tmp_path / "custom_ckpt"

    os.environ["DATA_ROOT"] = str(custom_data)
    os.environ["CACHE_ROOT"] = str(custom_cache)
    os.environ["CKPT_ROOT"] = str(custom_ckpt)

    try:
        assert get_data_root() == custom_data.resolve()
        assert get_cache_root() == custom_cache.resolve()
        assert get_ckpt_root() == custom_ckpt.resolve()
    finally:
        del os.environ["DATA_ROOT"]
        del os.environ["CACHE_ROOT"]
        del os.environ["CKPT_ROOT"]


def test_resolve_path():
    """Verify resolve_path handles relative and absolute paths properly."""
    rel = "configs/t0_ecapa.yaml"
    assert resolve_path(rel) == PROJECT_ROOT / rel

    abs_path = Path("/tmp/absolute_test.txt")
    assert resolve_path(abs_path) == abs_path


def test_accelerator_device_selection():
    """Verify get_accelerator_device returns a valid device and metadata dict."""
    device, details = get_accelerator_device()
    assert isinstance(device, torch.device)
    assert details["device_type"] in ["cuda", "mps", "cpu"]
    assert "device_name" in details


def test_colab_runner_notebook_validity():
    """Verify notebooks/colab_runner.ipynb exists, is valid JSON, and adheres to nbformat 4."""
    nb_path = PROJECT_ROOT / "notebooks" / "colab_runner.ipynb"
    assert nb_path.exists(), "colab_runner.ipynb does not exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4
    assert "cells" in nb
    assert len(nb["cells"]) >= 5


def test_no_hardcoded_user_paths_in_code():
    """Verify no hardcoded '/Users' or '/content' paths exist in src/ or scripts/ python files."""
    for folder in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
        for py_file in folder.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            # Exclude comments or docstrings that explicitly discuss the rule
            lines = text.splitlines()
            for idx, line in enumerate(lines, start=1):
                # Ignore docstring/comment references
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("*"):
                    continue
                assert "/Users/" not in line, f"Found hardcoded /Users/ in {py_file}:{idx}: {line}"
                assert "/content/" not in line, f"Found hardcoded /content/ in {py_file}:{idx}: {line}"

