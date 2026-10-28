"""Initial scaffold smoke tests."""

import src


def test_package_scaffold():
    """Verify package import and version existence."""
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


def test_environment_fixture(dummy_tensor_shape):
    """Verify fixture injection for tensor shapes."""
    batch_size, channels, freq_bins, time_frames = dummy_tensor_shape
    assert batch_size == 2
    assert channels == 1
    assert freq_bins == 80
    assert time_frames == 180
