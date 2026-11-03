#!/usr/bin/env python3
"""Build int16 disk memmap cache for training audio.

Stores audio from LibriSpeech manifests as int16 in a contiguous binary memmap
accompanied by a JSON index file per docs/DATA_PROTOCOL.md and docs/BENCHMARKS.md.

Note on caching policy:
    Empirical measurements in docs/BENCHMARKS.md show that on-the-fly FLAC decoding
    via soundfile with num_workers=4 achieves 26.36 ms/batch (37.94 batches/sec),
    which is 4.5x to 15.6x faster than model forward+backward step times
    (116.51 ms/step for lower bound, 411.52 ms/step for full SKA-HNN).
    Because asynchronous background pre-fetching completely hides I/O latency,
    and available disk space on the Mac M3 Pro is ~27.5 GB (requiring a >= 8 GB free buffer),
    on-the-fly decoding is the optimal default strategy and full memmap creation
    is skipped during standard training. This script provides verified caching
    capabilities if a disk-cache setup is explicitly required.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

# Project root path resolution
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.cache import AudioMemmapWriter
from src.utils.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build int16 binary memmap audio cache for LibriSpeech training split."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/train-clean-100.csv",
        help="Path to manifest CSV file (default: data/manifests/train-clean-100.csv)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Target cache directory for binary memmap and index (default: data/cache)",
    )
    parser.add_argument(
        "--bin-name",
        type=str,
        default="train-clean-100.bin",
        help="Filename for binary memmap file (default: train-clean-100.bin)",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="train-clean-100_index.json",
        help="Filename for index JSON file (default: train-clean-100_index.json)",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Optional maximum number of utterances to cache (useful for testing or subsets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect manifest, compute sizes and verify disk headroom without writing files",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=8.0,
        help="Safety buffer for free disk space in GB (default: 8.0 GB per AGENTS.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    cache_dir = resolve_path(args.cache_dir)

    print(f"=== Audio Cache Builder ===")
    print(f"Manifest: {manifest_path}")
    print(f"Target Cache Dir: {cache_dir}")
    print(f"Dry-run Mode: {args.dry_run}")

    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        sys.exit(1)

    # Inspect disk space
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(cache_dir)
    free_gb = stat.free / (1024**3)
    print(f"Available free disk space: {free_gb:.2f} GB")

    writer = AudioMemmapWriter(
        output_dir=cache_dir,
        bin_name=args.bin_name,
        index_name=args.index_name,
    )

    if args.dry_run:
        import csv
        entries = []
        with open(manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
                if args.max_utterances and len(entries) >= args.max_utterances:
                    break

        total_sec = sum(float(r.get("duration", 0.0)) for r in entries)
        est_samples = int(total_sec * 16000)
        est_gb = (est_samples * 2) / (1024**3)

        print("\n--- Dry Run Analysis ---")
        print(f"Utterances scanned: {len(entries)}")
        print(f"Total audio duration: {total_sec / 3600:.2f} hours ({total_sec:.1f} s)")
        print(f"Estimated int16 samples: {est_samples:,}")
        print(f"Estimated cache file size: {est_gb:.2f} GB")
        print(f"Free disk space after write: {free_gb - est_gb:.2f} GB")

        if (free_gb - est_gb) < args.min_free_disk_gb:
            print(
                f"WARNING: Writing cache would violate the minimum free disk buffer "
                f"({args.min_free_disk_gb:.1f} GB)."
            )
        else:
            print(f"Status: Disk space check passed (leaves >= {args.min_free_disk_gb:.1f} GB free).")

        print("\n[Caching Decision]")
        print("docs/BENCHMARKS.md confirms on-the-fly decoding achieves 26.36 ms/batch,")
        print("which is 15.6x faster than full SKA compute (411.52 ms/step).")
        print("Asynchronous pre-fetching completely hides I/O latency without disk cache.")
        return

    print("\nStarting cache creation...")
    summary = writer.write_from_manifest(
        manifest_csv_path=manifest_path,
        max_utterances=args.max_utterances,
        min_free_disk_gb=args.min_free_disk_gb,
    )
    print("Cache creation successful!")
    print(f"Utterances cached: {summary['num_utterances']}")
    print(f"Total samples: {summary['total_samples']:,}")
    print(f"Size: {summary['file_size_mb']} MB")
    print(f"Binary: {summary['bin_path']}")
    print(f"Index: {summary['index_path']}")


if __name__ == "__main__":
    main()

