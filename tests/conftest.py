"""Pytest fixtures and environment configuration."""

import pytest


@pytest.fixture
def dummy_tensor_shape():
    """Provides standard acoustic tensor input shape (B, 1, F, T)."""
    return (2, 1, 80, 180)
