#!/usr/bin/env python3
"""LibriSpeech manifest preparation and metadata extraction.

Parses SPEAKERS.TXT to extract gender and speaker metadata, computes exact FLAC
audio durations using soundfile, and writes standardized CSV manifests:
- utt_id: Unique utterance identifier (e.g. '19-198-0001')
- spk_id: Speaker identifier (e.g. '19')
- chapter_id: Book chapter identifier (e.g. '198')
- gender: Speaker biological gender ('M' or 'F') from SPEAKERS.TXT
- path: Path to FLAC file relative to project root
- duration: Audio duration in seconds
- split: Corpus partition name (e.g. 'train-clean-100', 'dev-clean')

For the training partition ('train-clean-100'), creates a contiguous zero-indexed
speaker-to-label mapping (0 to 250 for 251 speakers) saved to train_spk2label.json
and included in the manifest.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import soundfile as sf

# Project root path resolution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import PROJECT_ROOT, get_data_root, resolve_path


def parse_speakers_txt(speakers_txt_path: Path) -> dict[str, dict[str, str]]:
    """Parse LibriSpeech SPEAKERS.TXT metadata table.

    File schema:
        ; Comment lines
        READER_ID | SEX | SUBSET | MINUTES | NAME

    Args:
        speakers_txt_path: Path to SPEAKERS.TXT.

    Returns:
        Dictionary mapping reader_id (as str) to dict with keys:
            'gender', 'subset', 'minutes', 'name'.
    """
    speakers: dict[str, dict[str, str]] = {}
    if not speakers_txt_path.exists():
        raise FileNotFoundError(f"SPEAKERS.TXT not found at {speakers_txt_path}")

    with open(speakers_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                reader_id = parts[0]
                gender = parts[1].upper()
                subset = parts[2]
                minutes = parts[3]
                name = parts[4]
                speakers[reader_id] = {
                    "gender": gender,
                    "subset": subset,
                    "minutes": minutes,
                    "name": name,
                }
    return speakers


def scan_split_utterances(
    split_dir: Path,
    split_name: str,
    speakers_meta: dict[str, dict[str, str]],
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Scan all FLAC files in a partition directory and extract metadata.

    Directory structure:
        <split_dir>/<spk_id>/<chapter_id>/<utt_id>.flac

    Args:
        split_dir: Directory containing speaker subdirectories.
        split_name: Partition name (e.g. 'train-clean-100').
        speakers_meta: Dictionary of speaker metadata from parse_speakers_txt.
        project_root: Repository root for relative path formatting.

    Returns:
        List of utterance metadata records.
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Partition directory does not exist: {split_dir}")

    records: list[dict[str, Any]] = []
    # Collect all .flac files
    flac_files = sorted(split_dir.rglob("*.flac"))

    for flac_path in flac_files:
        # Expected path structure: .../<spk>/<chapter>/<utt_id>.flac
        utt_id = flac_path.stem
        chapter_id = flac_path.parent.name
        spk_id = flac_path.parent.parent.name

        spk_info = speakers_meta.get(spk_id, {})
        gender = spk_info.get("gender", "UNKNOWN")

        # Fast header reading for duration via soundfile
        info = sf.info(str(flac_path))
        duration = round(float(info.duration), 4)

        try:
            rel_path = str(flac_path.relative_to(project_root))
        except ValueError:
            rel_path = str(flac_path)

        records.append(
            {
                "utt_id": utt_id,
                "spk_id": spk_id,
                "chapter_id": chapter_id,
                "gender": gender,
                "path": rel_path,
                "duration": duration,
                "split": split_name,
            }
        )

    return records


def build_train_speaker_map(train_records: list[dict[str, Any]]) -> dict[str, int]:
    """Build a contiguous zero-indexed mapping from speaker_id to integer label.

    Speakers are sorted numerically if integer-convertible, else lexicographically.

    Args:
        train_records: List of records for the training partition.

    Returns:
        Dictionary mapping spk_id (str) to contiguous int in [0, N_speakers - 1].
    """
    unique_spks = sorted(
        list(set(r["spk_id"] for r in train_records)),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    return {spk: idx for idx, spk in enumerate(unique_spks)}


def write_manifest_csv(
    records: list[dict[str, Any]],
    output_path: Path,
    spk_to_label: dict[str, int] | None = None,
) -> None:
    """Write utterance records to a CSV file.

    Args:
        records: List of metadata dictionaries.
        output_path: Destination path for the CSV.
        spk_to_label: Optional dictionary mapping spk_id to contiguous int label.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "utt_id",
        "spk_id",
        "chapter_id",
        "gender",
        "path",
        "duration",
        "split",
    ]
    if spk_to_label is not None:
        fieldnames.append("label")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = dict(r)
            if spk_to_label is not None:
                row["label"] = spk_to_label[r["spk_id"]]
            writer.writerow(row)


def prepare_librispeech(
    raw_dir: Path,
    output_dir: Path,
    splits: list[str] | None = None,
) -> dict[str, int]:
    """Prepare all LibriSpeech manifests and training speaker label mapping.

    Args:
        raw_dir: Root directory of raw LibriSpeech corpus (contains LibriSpeech/ or split dirs).
        output_dir: Directory where CSV manifests will be saved.
        splits: List of splits to process (default: train-clean-100, dev-clean, test-clean, test-other).

    Returns:
        Dictionary mapping split name to utterance count.
    """
    if splits is None:
        splits = ["train-clean-100", "dev-clean", "test-clean", "test-other"]

    # Locate LibriSpeech root
    libri_root = raw_dir / "LibriSpeech" if (raw_dir / "LibriSpeech").exists() else raw_dir
    speakers_txt = libri_root / "SPEAKERS.TXT"
    if not speakers_txt.exists():
        # Check fallback in data root
        speakers_txt = get_data_root() / "LibriSpeech" / "SPEAKERS.TXT"

    speakers_meta = parse_speakers_txt(speakers_txt)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    train_records: list[dict[str, Any]] = []

    for split in splits:
        split_dir = libri_root / split
        if not split_dir.exists():
            print(f"[Warning] Split directory {split_dir} does not exist. Skipping.")
            continue

        records = scan_split_utterances(split_dir, split, speakers_meta)
        counts[split] = len(records)
        print(f"Processed {split}: {len(records)} utterances.")

        if split == "train-clean-100":
            train_records = records

    # Build contiguous speaker label mapping for train
    if train_records:
        spk_to_label = build_train_speaker_map(train_records)
        spk2label_path = output_dir / "train_spk2label.json"
        with open(spk2label_path, "w", encoding="utf-8") as f:
            json.dump(spk_to_label, f, indent=2)
        print(f"Saved contiguous speaker-to-label map for {len(spk_to_label)} speakers to {spk2label_path}.")

        # Write train CSV with 'label' column
        write_manifest_csv(
            train_records,
            output_dir / "train-clean-100.csv",
            spk_to_label=spk_to_label,
        )

    # Write evaluation splits without label column
    for split in splits:
        if split == "train-clean-100":
            continue
        split_dir = libri_root / split
        if not split_dir.exists():
            continue
        records = scan_split_utterances(split_dir, split, speakers_meta)
        write_manifest_csv(records, output_dir / f"{split}.csv")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech CSV manifests.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(get_data_root() / "raw"),
        help="Path to raw LibriSpeech directory (default: data/raw).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(get_data_root() / "manifests"),
        help="Path to output manifests directory (default: data/manifests).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100", "dev-clean", "test-clean", "test-other"],
        help="Partitions to process.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    prepare_librispeech(raw_dir, out_dir, splits=args.splits)


if __name__ == "__main__":
    main()

