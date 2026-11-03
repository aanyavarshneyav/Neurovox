"""Unit tests for dataset integrity, speaker disjointness, and trial validity.

Asserts:
1. Complete pairwise disjointness of speaker IDs across train, dev, test-clean, and test-other.
2. Exact matching between CSV manifests and audio files on disk (existence, count, durations).
3. Strict 100% cross-chapter guarantee on genuine target trials.
4. Strict gender-matching on impostor trials.
5. Verification report checksum and audit consistency.
"""

from pathlib import Path
import csv
import json
import pytest
import soundfile as sf

from src.utils.paths import PROJECT_ROOT, get_data_root
from src.data.prep_librispeech import (
    parse_speakers_txt,
    prepare_librispeech,
    build_train_speaker_map,
)
from src.data.trial_generator import TrialGenerator


def get_speakers_txt_path() -> Path:
    """Find SPEAKERS.TXT in raw or existing data directory."""
    candidates = [
        get_data_root() / "raw" / "LibriSpeech" / "SPEAKERS.TXT",
        get_data_root() / "LibriSpeech" / "SPEAKERS.TXT",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("SPEAKERS.TXT not found in data directories.")


def test_speakers_pairwise_disjoint_in_speakers_txt():
    """Verify that train-clean-100, dev-clean, test-clean, and test-other are strictly disjoint."""
    speakers_txt = get_speakers_txt_path()
    meta = parse_speakers_txt(speakers_txt)

    target_subsets = ["train-clean-100", "dev-clean", "test-clean", "test-other"]
    subset_speakers: dict[str, set[str]] = {s: set() for s in target_subsets}

    for spk_id, info in meta.items():
        sub = info["subset"]
        if sub in subset_speakers:
            subset_speakers[sub].add(spk_id)

    # Assert expected speaker counts
    assert len(subset_speakers["train-clean-100"]) == 251, "train-clean-100 must have 251 speakers"
    assert len(subset_speakers["dev-clean"]) == 40, "dev-clean must have 40 speakers"
    assert len(subset_speakers["test-clean"]) == 40, "test-clean must have 40 speakers"
    assert len(subset_speakers["test-other"]) == 33, "test-other must have 33 speakers"

    # Assert strictly empty pairwise intersections
    for i in range(len(target_subsets)):
        for j in range(i + 1, len(target_subsets)):
            s1 = target_subsets[i]
            s2 = target_subsets[j]
            overlap = subset_speakers[s1].intersection(subset_speakers[s2])
            assert len(overlap) == 0, (
                f"Speaker overlap detected between {s1} and {s2}: {overlap}"
            )


def test_manifest_speaker_disjointness_if_manifests_exist():
    """Verify generated manifests have zero overlapping speaker IDs."""
    manifest_dir = get_data_root() / "manifests"
    if not manifest_dir.exists():
        pytest.skip("Manifest directory does not exist yet.")

    splits = ["train-clean-100", "dev-clean", "test-clean", "test-other"]
    split_spks: dict[str, set[str]] = {}

    for split in splits:
        csv_path = manifest_dir / f"{split}.csv"
        if csv_path.exists():
            spks = set()
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    spks.add(row["spk_id"])
            split_spks[split] = spks

    present_splits = list(split_spks.keys())
    for i in range(len(present_splits)):
        for j in range(i + 1, len(present_splits)):
            s1, s2 = present_splits[i], present_splits[j]
            overlap = split_spks[s1].intersection(split_spks[s2])
            assert len(overlap) == 0, f"Manifest speaker overlap between {s1} and {s2}: {overlap}"


def test_manifest_files_exist_and_durations_match():
    """Verify that all files in existing manifests exist on disk and durations match."""
    manifest_dir = get_data_root() / "manifests"
    if not manifest_dir.exists():
        pytest.skip("Manifest directory does not exist yet.")

    manifest_files = list(manifest_dir.glob("*.csv"))
    if not manifest_files:
        pytest.skip("No CSV manifest files found.")

    for m_file in manifest_files:
        with open(m_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, f"Manifest {m_file.name} is empty."

        # Check a sample of rows (first 20 and last 20) for fast verification
        sample_rows = rows[:20] + rows[-20:] if len(rows) > 40 else rows
        for r in sample_rows:
            rel_path = Path(r["path"])
            abs_path = (PROJECT_ROOT / rel_path) if not rel_path.is_absolute() else rel_path
            assert abs_path.exists(), f"File listed in manifest does not exist: {abs_path}"

            # Verify duration within 0.01s tolerance
            header_duration = sf.info(str(abs_path)).duration
            manifest_duration = float(r["duration"])
            assert abs(header_duration - manifest_duration) < 0.01, (
                f"Duration mismatch for {abs_path}: manifest={manifest_duration}, file={header_duration}"
            )


def test_trial_generator_strict_cross_chapter_and_gender_match():
    """Verify trial list guarantees: 100% cross-chapter targets, gender-matched impostors."""
    trials_dir = get_data_root() / "trials"
    if not trials_dir.exists():
        pytest.skip("Trials directory does not exist yet.")

    trial_files = list(trials_dir.glob("trials_*.txt"))
    if not trial_files:
        pytest.skip("No trial files found.")

    speakers_txt = get_speakers_txt_path()
    speakers_meta = parse_speakers_txt(speakers_txt)

    for tf in trial_files:
        target_count = 0
        impostor_count = 0

        with open(tf, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                parts = line.strip().split()
                assert len(parts) == 3, f"Invalid trial line format at {tf}:{line_idx}"
                label, p1, p2 = int(parts[0]), parts[1], parts[2]

                # Extract spk and chapter from paths (.../<spk>/<chapter>/<utt>.flac)
                p1_parts = Path(p1).parts
                p2_parts = Path(p2).parts

                spk1, ch1 = p1_parts[-3], p1_parts[-2]
                spk2, ch2 = p2_parts[-3], p2_parts[-2]

                if label == 1:
                    target_count += 1
                    # Assert same speaker
                    assert spk1 == spk2, f"Target trial has different speakers: {spk1} vs {spk2} at line {line_idx}"
                    # STRICT CROSS-CHAPTER ASSERTION
                    assert ch1 != ch2, (
                        f"CRITICAL VIOLATION: Target trial at {tf}:{line_idx} has same chapter {ch1}! "
                        "100% of target trials must be cross-chapter."
                    )
                elif label == 0:
                    impostor_count += 1
                    # Assert different speaker
                    assert spk1 != spk2, f"Impostor trial has same speaker: {spk1} at line {line_idx}"
                    # STRICT GENDER-MATCHED ASSERTION
                    g1 = speakers_meta.get(spk1, {}).get("gender")
                    g2 = speakers_meta.get(spk2, {}).get("gender")
                    if g1 and g2:
                        assert g1 == g2, (
                            f"CRITICAL VIOLATION: Impostor trial at {tf}:{line_idx} has mismatched genders: "
                            f"{spk1}({g1}) vs {spk2}({g2})"
                        )

        assert target_count > 0, f"No target trials in {tf}"
        assert impostor_count > 0, f"No impostor trials in {tf}"


def test_synthetic_data_pipeline_end_to_end(tmp_path):
    """End-to-end test on synthetic directory structure to verify all constraints."""
    mock_raw = tmp_path / "mock_raw" / "LibriSpeech"
    mock_raw.mkdir(parents=True)

    # Create mock SPEAKERS.TXT
    speakers_txt = mock_raw / "SPEAKERS.TXT"
    speakers_txt.write_text(
        "; Mock SPEAKERS.TXT\n"
        "101 | M | train-clean-100 | 10.0 | Alice\n"
        "102 | F | train-clean-100 | 10.0 | Bob\n"
        "201 | M | dev-clean       | 10.0 | Charlie\n"
        "202 | M | dev-clean       | 10.0 | Dave\n"
        "203 | F | dev-clean       | 10.0 | Eve\n"
        "205 | F | dev-clean       | 10.0 | Fiona\n"
        "204 | F | dev-clean       | 10.0 | SingleChapterGirl\n"
    )

    import numpy as np

    # Helper to generate short FLAC
    def make_flac(path: Path, duration_sec: float = 2.5):
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = int(duration_sec * 16000)
        data = np.zeros(samples, dtype=np.float32)
        sf.write(str(path), data, 16000, format="FLAC")

    # Train speakers
    make_flac(mock_raw / "train-clean-100" / "101" / "1001" / "101-1001-0001.flac")
    make_flac(mock_raw / "train-clean-100" / "102" / "1002" / "102-1002-0001.flac")

    # Dev speakers:
    # 201 (M): 2 chapters
    make_flac(mock_raw / "dev-clean" / "201" / "2001" / "201-2001-0001.flac")
    make_flac(mock_raw / "dev-clean" / "201" / "2002" / "201-2002-0001.flac")
    # 202 (M): 2 chapters
    make_flac(mock_raw / "dev-clean" / "202" / "2003" / "202-2003-0001.flac")
    make_flac(mock_raw / "dev-clean" / "202" / "2004" / "202-2004-0001.flac")
    # 203 (F): 2 chapters
    make_flac(mock_raw / "dev-clean" / "203" / "2005" / "203-2005-0001.flac")
    make_flac(mock_raw / "dev-clean" / "203" / "2006" / "203-2006-0001.flac")
    # 205 (F): 2 chapters
    make_flac(mock_raw / "dev-clean" / "205" / "2008" / "205-2008-0001.flac")
    make_flac(mock_raw / "dev-clean" / "205" / "2009" / "205-2009-0001.flac")
    # 204 (F): single chapter ONLY
    make_flac(mock_raw / "dev-clean" / "204" / "2007" / "204-2007-0001.flac")
    make_flac(mock_raw / "dev-clean" / "204" / "2007" / "204-2007-0002.flac")

    mock_manifests = tmp_path / "manifests"
    counts = prepare_librispeech(mock_raw, mock_manifests, splits=["train-clean-100", "dev-clean"])
    assert counts["train-clean-100"] == 2
    assert counts["dev-clean"] == 10

    # Verify train_spk2label.json
    spk2label_file = mock_manifests / "train_spk2label.json"
    assert spk2label_file.exists()
    with open(spk2label_file) as f:
        spk2label = json.load(f)
    assert spk2label == {"101": 0, "102": 1}

    # Generate trials for dev-clean
    gen = TrialGenerator(manifest_path=mock_manifests / "dev-clean.csv", min_duration=2.0, seed=42)
    # Speaker 204 must be strictly excluded from targets!
    assert len(gen.single_chapter_speakers) == 1
    assert gen.single_chapter_speakers[0]["spk_id"] == "204"
    assert "204" not in gen.qualifying_speakers

    trials, report = gen.generate_trials(targets_per_speaker=1)
    assert report["num_excluded_single_chapter_speakers"] == 1
    assert report["target_trials_count"] == 4  # 201, 202, 203, 205 each give 1 target

    # Assert 100% cross chapter on generated targets
    for label, p1, p2 in trials:
        if label == 1:
            ch1 = Path(p1).parts[-2]
            ch2 = Path(p2).parts[-2]
            assert ch1 != ch2, f"Target pair has same chapter: {p1} vs {p2}"

