"""Unit tests for dataset loading, transforms, and speaker-balanced batch sampler.

Verifies:
- crop_or_pad_waveform handles exact length, short clips (wrap padding), and long clips (cropping).
- Waveform augmentations are individually toggleable and deterministically seeded.
- SpeakerBalancedBatchSampler guarantees B distinct speakers per batch and 100% speaker coverage per epoch.
- LibriSpeechDataset integrates cleanly with PyTorch DataLoader and produces (B, 80, 180) batches.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from src.data.cache import AudioMemmapReader, AudioMemmapWriter
from src.data.dataset import LibriSpeechDataset
from src.data.sampler import SpeakerBalancedBatchSampler
from src.data.transforms import (
    BabbleNoiseAugmentation,
    ColoredNoiseAugmentation,
    RIRReverberation,
    SpeedPerturbation,
    crop_or_pad_waveform,
)


def test_crop_or_pad_exact():
    """Verify exact length input is returned unchanged."""
    x = torch.randn(29040)
    out = crop_or_pad_waveform(x, target_samples=29040)
    assert out.shape == (29040,)
    assert torch.equal(out, x)


def test_crop_or_pad_short_wrap_padding():
    """Verify short clips are wrap-padded to exact target length."""
    short = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out = crop_or_pad_waveform(short, target_samples=12)
    assert out.shape == (12,)
    # Circular repetition of [1, 2, 3, 4, 5]
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0])
    assert torch.equal(out, expected)


def test_crop_or_pad_long_random_and_center():
    """Verify long clips are cropped to target length (random vs center)."""
    long_clip = torch.arange(100.0)
    rng = np.random.default_rng(123)

    # Train mode: random offset
    out_train = crop_or_pad_waveform(long_clip, target_samples=20, rng=rng, is_train=True)
    assert out_train.shape == (20,)

    # Eval mode: center crop
    out_eval = crop_or_pad_waveform(long_clip, target_samples=20, is_train=False)
    assert out_eval.shape == (20,)
    # Center of 100 with length 20: start at (100 - 20) // 2 = 40
    expected_eval = torch.arange(40.0, 60.0)
    assert torch.equal(out_eval, expected_eval)


def test_speed_perturbation_toggleable_and_deterministic():
    """Verify SpeedPerturbation is toggleable, respects probability and seeds."""
    x = torch.randn(29040)
    sp_off = SpeedPerturbation(p=0.0)
    out_off = sp_off(x)
    assert torch.equal(out_off, x)

    sp_on = SpeedPerturbation(min_factor=1.05, max_factor=1.05, p=1.0)
    rng1 = np.random.default_rng(42)
    out1 = sp_on(x, rng=rng1)
    rng2 = np.random.default_rng(42)
    out2 = sp_on(x, rng=rng2)

    assert not torch.equal(out1, x)
    assert torch.equal(out1, out2)


def test_colored_noise_white_and_pink():
    """Verify ColoredNoiseAugmentation modifies signal at target SNR."""
    x = torch.randn(29040) * 0.1
    noise_aug = ColoredNoiseAugmentation(snr_min_db=10.0, snr_max_db=10.0, p=1.0)

    rng = np.random.default_rng(42)
    noisy = noise_aug(x, rng=rng)

    assert noisy.shape == x.shape
    assert not torch.equal(noisy, x)
    # Energy should have increased due to additive noise
    assert torch.mean(noisy**2) > torch.mean(x**2)


def test_babble_noise():
    """Verify BabbleNoiseAugmentation adds speech babble."""
    x = torch.randn(29040) * 0.2
    babble_aug = BabbleNoiseAugmentation(p=1.0)

    rng = np.random.default_rng(99)
    out = babble_aug(x, rng=rng)

    assert out.shape == x.shape
    assert not torch.equal(out, x)


def test_rir_reverberation_bypass_and_execution(tmp_path: Path):
    """Verify RIRReverberation safely bypasses if missing, and convolves if present."""
    x = torch.randn(29040)

    # Missing RIR directory -> bypassed
    rir_empty = RIRReverberation(rir_dir=None, p=1.0)
    assert not rir_empty.is_available
    assert torch.equal(rir_empty(x), x)

    # Create dummy RIR wav
    rir_dir = tmp_path / "rir"
    rir_dir.mkdir()
    dummy_rir = np.zeros(1600, dtype=np.float32)
    dummy_rir[0] = 1.0  # Impulse
    dummy_rir[200] = 0.5  # Reflection
    sf.write(str(rir_dir / "test_rir.wav"), dummy_rir, 16000)

    rir_active = RIRReverberation(rir_dir=rir_dir, p=1.0)
    assert rir_active.is_available

    rng = np.random.default_rng(42)
    out = rir_active(x, rng=rng)
    assert out.shape == x.shape
    assert not torch.equal(out, x)


def test_speaker_balanced_sampler_distinct_and_coverage():
    """Test that every batch has B distinct speakers and every speaker is visited per epoch."""
    # 251 speakers (matching train-clean-100), 10 samples each
    num_speakers = 251
    batch_size = 64
    batches_per_epoch = 10  # 10 batches * 64 = 640 draws (minimum needed is ceil(251/64) = 4)

    speaker_labels = []
    for spk in range(num_speakers):
        speaker_labels.extend([spk] * 10)  # 2510 samples total

    sampler = SpeakerBalancedBatchSampler(
        speaker_labels=speaker_labels,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        seed=42,
    )

    assert len(sampler) == batches_per_epoch

    all_speakers_set = set(range(num_speakers))

    # Evaluate across 2 epochs
    for epoch in range(2):
        sampler.set_epoch(epoch)
        visited_speakers = set()

        for batch in sampler:
            # Assertion 1: Batch size is exactly B
            assert len(batch) == batch_size

            batch_speakers = [speaker_labels[idx] for idx in batch]

            # Assertion 2: All B speakers in this batch are strictly distinct
            assert len(set(batch_speakers)) == batch_size, "Batch contains duplicate speakers!"

            visited_speakers.update(batch_speakers)

        # Assertion 3: Every single speaker is visited in the epoch
        assert visited_speakers == all_speakers_set, (
            f"Epoch {epoch} missed {len(all_speakers_set - visited_speakers)} speakers!"
        )


def test_audio_memmap_writer_and_reader(tmp_path: Path):
    """Test building and reading audio memmap cache."""
    # Create 3 synthetic wav/flac files
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    manifest_csv = tmp_path / "manifest.csv"

    entries = []
    for i in range(3):
        flac_path = audio_dir / f"test_{i}.flac"
        data = np.sin(np.linspace(0, 10, 32000)).astype(np.float32) * 0.5
        sf.write(str(flac_path), data, 16000, format="FLAC")
        entries.append(
            f"utt_{i},{i},ch_1,M,{flac_path},2.0,train,{i}\n"
        )

    with open(manifest_csv, "w") as f:
        f.write("utt_id,spk_id,chapter_id,gender,path,duration,split,label\n")
        f.writelines(entries)

    cache_dir = tmp_path / "cache"
    writer = AudioMemmapWriter(output_dir=cache_dir)
    summary = writer.write_from_manifest(manifest_csv, min_free_disk_gb=0.1)

    assert summary["num_utterances"] == 3
    assert Path(summary["bin_path"]).exists()
    assert Path(summary["index_path"]).exists()

    reader = AudioMemmapReader(Path(summary["index_path"]))
    assert len(reader) == 3
    read_audio = reader.read_audio("utt_0")
    assert read_audio.shape == (32000,)
    assert read_audio.dtype == np.float32
    # Verify close correlation with original signal
    orig_data, _ = sf.read(str(audio_dir / "test_0.flac"), dtype="float32")
    assert np.allclose(read_audio, orig_data, atol=1e-3)


def test_librispeech_dataset_with_dataloader(tmp_path: Path):
    """Test full LibriSpeechDataset pipeline with DataLoader batching."""
    # Create small mock manifest
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    manifest_csv = tmp_path / "manifest.csv"

    entries = []
    # 8 speakers, 2 utterances each
    for spk in range(8):
        for u in range(2):
            utt_id = f"spk{spk}_utt{u}"
            flac_path = audio_dir / f"{utt_id}.flac"
            data = np.random.randn(30000).astype(np.float32) * 0.1
            sf.write(str(flac_path), data, 16000, format="FLAC")
            entries.append(
                f"{utt_id},{spk},ch_1,M,{flac_path},1.875,train,{spk}\n"
            )

    with open(manifest_csv, "w") as f:
        f.write("utt_id,spk_id,chapter_id,gender,path,duration,split,label\n")
        f.writelines(entries)

    dataset = LibriSpeechDataset(
        manifest_path=manifest_csv,
        is_train=True,
        return_features=True,
        enable_specaug=True,
    )

    assert len(dataset) == 16
    feat, label = dataset[(0, 0, 0)]
    assert feat.shape == (80, 180)
    assert isinstance(label, int)

    sampler = SpeakerBalancedBatchSampler(
        speaker_labels=dataset.labels,
        batch_size=4,
        batches_per_epoch=2,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
    )

    for batch_feats, batch_labels in loader:
        assert batch_feats.shape == (4, 80, 180)
        assert batch_feats.dtype == torch.float32
        assert batch_labels.shape == (4,)
        # All 4 speakers in each batch must be distinct
        assert len(set(batch_labels.tolist())) == 4


def test_dataset_train_mode_plain_int_index_raises(tmp_path: Path):
    """Verify __getitem__ with a plain int index raises TypeError in train mode (F2)."""
    audio_dir = tmp_path / "audio_f2"
    audio_dir.mkdir()
    manifest_csv = tmp_path / "manifest_f2.csv"
    flac_path = audio_dir / "sample.flac"
    sf.write(str(flac_path), np.random.randn(30000).astype(np.float32) * 0.1, 16000, format="FLAC")
    with open(manifest_csv, "w") as f:
        f.write("utt_id,spk_id,chapter_id,gender,path,duration,split,label\n")
        f.write(f"sample,0,ch_1,M,{flac_path},1.875,train,0\n")

    dataset = LibriSpeechDataset(
        manifest_path=manifest_csv,
        is_train=True,
    )
    with pytest.raises(TypeError, match="tuple"):
        _ = dataset[0]


def test_same_idx_drawn_twice_in_epoch_gets_different_crop_and_aug(tmp_path: Path):
    """Verify same idx drawn with different draw_id gets distinct crops/augmentations (F1)."""
    audio_dir = tmp_path / "audio_draw"
    audio_dir.mkdir()
    manifest_csv = tmp_path / "manifest_draw.csv"
    flac_path = audio_dir / "long_sample.flac"
    # Long audio (4 seconds = 64000 samples) so crops differ
    sf.write(str(flac_path), np.random.randn(64000).astype(np.float32) * 0.1, 16000, format="FLAC")
    with open(manifest_csv, "w") as f:
        f.write("utt_id,spk_id,chapter_id,gender,path,duration,split,label\n")
        f.write(f"long_sample,0,ch_1,M,{flac_path},4.0,train,0\n")

    dataset = LibriSpeechDataset(
        manifest_path=manifest_csv,
        is_train=True,
        return_features=False,  # return waveform
        enable_speed_aug=True,
        enable_noise_aug=True,
        base_seed=42,
    )

    epoch = 1
    idx = 0
    draw_0_wave, _ = dataset[(idx, epoch, 0)]
    draw_1_wave, _ = dataset[(idx, epoch, 1)]

    assert not torch.equal(draw_0_wave, draw_1_wave), "Two draws of same idx in one epoch must differ"


def test_deterministic_seeding_across_workers(tmp_path: Path):
    """Verify same (seed, epoch, idx, draw_id) reproduces across workers with babble+RIR and set_epoch changes (F1)."""
    audio_dir = tmp_path / "audio_det"
    audio_dir.mkdir()
    manifest_csv = tmp_path / "manifest_det.csv"

    # Create dummy RIR
    rir_dir = tmp_path / "rirs"
    rir_dir.mkdir()
    dummy_rir = np.zeros(1600, dtype=np.float32)
    dummy_rir[0] = 1.0
    dummy_rir[150] = 0.4
    sf.write(str(rir_dir / "test_rir.wav"), dummy_rir, 16000)

    entries = []
    # 8 speakers, 2 utterances each
    for spk in range(8):
        for u in range(2):
            utt_id = f"spk{spk}_utt{u}"
            flac_path = audio_dir / f"{utt_id}.flac"
            data = np.random.randn(32000).astype(np.float32) * 0.1
            sf.write(str(flac_path), data, 16000, format="FLAC")
            entries.append(
                f"{utt_id},{spk},ch_1,M,{flac_path},2.0,train,{spk}\n"
            )

    with open(manifest_csv, "w") as f:
        f.write("utt_id,spk_id,chapter_id,gender,path,duration,split,label\n")
        f.writelines(entries)

    def create_pipeline(num_workers: int):
        ds = LibriSpeechDataset(
            manifest_path=manifest_csv,
            is_train=True,
            return_features=True,
            enable_speed_aug=True,
            enable_noise_aug=True,
            enable_babble_aug=True,
            enable_rir_aug=True,
            rir_dir=rir_dir,
            enable_specaug=True,
            base_seed=42,
        )
        sampler = SpeakerBalancedBatchSampler(
            speaker_labels=ds.labels,
            batch_size=4,
            batches_per_epoch=2,
            seed=42,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
        return sampler, loader

    sampler_w0, loader_w0 = create_pipeline(0)
    sampler_w2, loader_w2 = create_pipeline(2)

    # Test epoch 1
    sampler_w0.set_epoch(1)
    sampler_w2.set_epoch(1)
    batches_w0_ep1 = list(loader_w0)
    batches_w2_ep1 = list(loader_w2)

    for (f0, l0), (f2, l2) in zip(batches_w0_ep1, batches_w2_ep1):
        diff = (f0 - f2).abs().max().item()
        assert diff < 1e-5, f"Epoch 1 mismatch: max diff = {diff:.2e}"
        assert torch.equal(l0, l2)

    # Test epoch 2 with persistent workers
    sampler_w0.set_epoch(2)
    sampler_w2.set_epoch(2)
    batches_w0_ep2 = list(loader_w0)
    batches_w2_ep2 = list(loader_w2)

    for (f0, l0), (f2, l2) in zip(batches_w0_ep2, batches_w2_ep2):
        diff = (f0 - f2).abs().max().item()
        assert diff < 1e-5, f"Epoch 2 mismatch: max diff = {diff:.2e}"
        assert torch.equal(l0, l2)



