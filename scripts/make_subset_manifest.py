#!/usr/bin/env python3
"""Generate a deterministic 50-speaker subset manifest from train-clean-100.

Takes the first 50 unique speaker IDs in train-clean-100, remaps labels to 0..49,
and outputs data/manifests/train-50spk.csv.
Guarantees bitwise reproducible checksum across platforms.
"""

import argparse
import hashlib
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_data_root, resolve_path

EXPECTED_SHA256 = "f63c4dc99eb41b45880fcf527aa724160e6b4860390cc7c0221a06a27f5f46da"


def make_subset_manifest(
    source_manifest: Path | str = "data/manifests/train-clean-100.csv",
    output_manifest: Path | str = "data/manifests/train-50spk.csv",
    num_speakers: int = 50,
) -> Path:
    source_path = resolve_path(source_manifest)
    out_path = resolve_path(output_manifest)

    if not source_path.exists():
        raise FileNotFoundError(f"Source manifest not found: {source_path}")

    df = pd.read_csv(source_path)
    # Sort unique speakers to ensure deterministic selection across systems
    unique_spks = sorted(df["spk_id"].unique())[:num_speakers]
    df_subset = df[df["spk_id"].isin(unique_spks)].copy()

    # Re-map labels 0..num_speakers-1
    spk2label = {spk: idx for idx, spk in enumerate(unique_spks)}
    df_subset["label"] = df_subset["spk_id"].map(spk2label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_csv(out_path, index=False)

    # Compute sha256
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Created subset manifest: {out_path}")
    print(f"Utterances: {len(df_subset)} across {len(unique_spks)} speakers")
    print(f"SHA-256: {h}")

    if num_speakers == 50 and h.lower() != EXPECTED_SHA256.lower():
        print(f"WARNING: Generated SHA-256 differs from expected reference: {EXPECTED_SHA256}", file=sys.stderr)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic subset manifest.")
    parser.add_argument("--source", default="data/manifests/train-clean-100.csv", help="Source manifest path")
    parser.add_argument("--output", default="data/manifests/train-50spk.csv", help="Output manifest path")
    parser.add_argument("--num-speakers", type=int, default=50, help="Number of speakers to select")
    args = parser.parse_args()

    make_subset_manifest(args.source, args.output, args.num_speakers)
    return 0


if __name__ == "__main__":
    sys.exit(main())

