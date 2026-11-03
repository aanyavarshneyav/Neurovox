"""Unit tests for full-utterance embedding extractor and batching invariance."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.dataset import LibriSpeechDataset
from src.eval.extractor import EmbeddingExtractor
from src.models.small_tdnn_se import small_tdnn_se


def test_extractor_l2_normalization():
    """Verify extracted embeddings are strictly unit-norm (norm = 1.0 +- 1e-5)."""
    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    extractor = EmbeddingExtractor(model, device=torch.device("cpu"))

    # Random waveform (1.5 seconds at 16 kHz = 24,000 samples)
    wav = torch.randn(24000)
    emb = extractor.extract_from_waveform(wav)

    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_extractor_batching_invariance():
    """Verify extracting an utterance alone vs in a batch of identical-length utts matches within 1e-4."""
    torch.manual_seed(42)
    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    extractor = EmbeddingExtractor(model, device=torch.device("cpu"))

    # 4 synthetic waveforms of identical length (e.g. 32,000 samples = 2.0s)
    T_samples = 32000
    w0 = torch.randn(1, T_samples)
    w1 = torch.randn(1, T_samples)
    w2 = torch.randn(1, T_samples)
    w3 = torch.randn(1, T_samples)

    # Extract w0 alone (batch size 1)
    emb_alone = extractor.extract_from_waveform(w0)

    # Extract all 4 together in a single batch (B=4, 1, T)
    batch = torch.stack([w0, w1, w2, w3], dim=0)  # (4, 1, T)
    batch_embs = extractor.extract_batch_identical_length(batch)
    emb_in_batch = batch_embs[0]

    max_diff = np.max(np.abs(emb_alone - emb_in_batch))
    assert max_diff < 1e-4, f"Batching altered embedding! Max diff: {max_diff:.2e} (must be < 1e-4)"


def test_extractor_bucketed_batch_matches_batch_size_1():
    """Verify extract_batch_bucketed matches individual batch-size-1 extraction within 1e-4."""
    torch.manual_seed(123)
    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    extractor = EmbeddingExtractor(model, device=torch.device("cpu"))

    # 6 waveforms with 3 distinct lengths (2 of length 24000, 2 of length 32000, 2 of length 16000)
    wavs = [
        torch.randn(24000),
        torch.randn(32000),
        torch.randn(16000),
        torch.randn(24000),
        torch.randn(16000),
        torch.randn(32000),
    ]

    # Batch-size-1 extraction for each
    embs_single = [extractor.extract_from_waveform(w) for w in wavs]

    # Bucketed batch extraction
    embs_bucketed = extractor.extract_batch_bucketed(wavs)

    for i, (e_single, e_bucket) in enumerate(zip(embs_single, embs_bucketed)):
        diff = np.max(np.abs(e_single - e_bucket))
        assert diff < 1e-4, f"Waveform {i}: bucketed extraction deviates from batch-size-1 by {diff:.2e}"


def test_extractor_enforces_eval_mode():
    """Verify extractor enforces model.eval() even if model was in train() mode."""
    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    model.train()
    assert model.training is True

    extractor = EmbeddingExtractor(model, device=torch.device("cpu"))
    assert model.training is False, "Extractor must set model.eval()"

    # Put it back to train and extract; should still extract with eval
    model.train()
    wav = torch.randn(24000)
    _ = extractor.extract_from_waveform(wav)
    assert model.training is False, "Extractor forward must preserve model.eval()"


def test_extractor_extracts_only_trial_utterances(tmp_path):
    """Verify extractor loads only unique utterances mentioned in the trial list."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create 3 dummy audio files
    paths = []
    for i in range(3):
        p = audio_dir / f"utt_{i}.flac"
        sf.write(str(p), np.random.randn(16000).astype(np.float32), 16000)
        paths.append(str(p))

    # Trial list only uses utt_0 and utt_1, omits utt_2
    trial_file = tmp_path / "trials.txt"
    with open(trial_file, "w", encoding="utf-8") as f:
        f.write(f"1 {paths[0]} {paths[1]}\n")

    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    extractor = EmbeddingExtractor(model, device=torch.device("cpu"))

    cached_embs = extractor.extract_for_trial_file(trial_file)

    assert paths[0] in cached_embs
    assert paths[1] in cached_embs
    assert paths[2] not in cached_embs, "Unreferenced utterance was loaded into cache!"
    assert len(cached_embs) == 2


def test_extractor_dataset_feature_parity(tmp_path):
    """Verify LibriSpeechDataset.feature_extractor and EmbeddingExtractor.feature_extractor match within 1e-5.

    Also asserts exact (80, 180) shape for a 29,040-sample crop.
    """
    # Create minimal mock manifest
    manifest_csv = tmp_path / "manifest.csv"
    with open(manifest_csv, "w") as f:
        f.write("utt_id,spk_id,chapter_id,gender,path,duration,split,label\n")
        f.write("u1,1,c1,M,dummy.flac,2.0,train,0\n")

    dataset = LibriSpeechDataset(
        manifest_path=manifest_csv,
        is_train=False,
        return_features=True,
        enable_specaug=False,
    )

    model = small_tdnn_se(in_features=80, channels=128, emb_dim=192)
    extractor = EmbeddingExtractor(model, device=torch.device("cpu"))

    target_samples = 180 * 160 + 240  # 29,040 samples
    torch.manual_seed(42)
    wav = torch.randn(1, target_samples)

    feat_ds = dataset.feature_extractor(wav)
    feat_ex = extractor.feature_extractor(wav)

    # Shape assertion: (1, 80, 180)
    assert feat_ds.shape == (1, 80, 180), f"Expected shape (1, 80, 180), got {feat_ds.shape}"
    assert feat_ex.shape == (1, 80, 180), f"Expected shape (1, 80, 180), got {feat_ex.shape}"

    # Tolerance assertion: < 1e-5
    max_diff = (feat_ds - feat_ex).abs().max().item()
    assert max_diff < 1e-5, f"Feature extraction mismatch between dataset and extractor: {max_diff:.2e}"
