"""Evaluation protocols: cosine scoring, EER, dual minDCF, bootstrap CI, and embedding extraction."""

from src.eval.extractor import EmbeddingExtractor
from src.eval.metrics import (
    compute_det_curve,
    compute_eer,
    compute_mindcf,
    speaker_bootstrap_ci,
    speaker_bootstrap_two_sample_difference,
)

__all__ = [
    "EmbeddingExtractor",
    "compute_det_curve",
    "compute_eer",
    "compute_mindcf",
    "speaker_bootstrap_ci",
    "speaker_bootstrap_two_sample_difference",
]
