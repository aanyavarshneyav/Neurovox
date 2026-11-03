"""Pytest fixtures and environment configuration."""

import pytest


@pytest.fixture(autouse=True)
def redirect_roots(tmp_path, monkeypatch):
    """Redirect CKPT_ROOT and RESULTS_ROOT to tmp_path so tests never write to real results/ or checkpoints/."""
    monkeypatch.setenv("CKPT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path / "results"))


@pytest.fixture
def dummy_tensor_shape():
    """Provides standard acoustic tensor input shape (B, 1, F, T)."""
    return (2, 1, 80, 180)
