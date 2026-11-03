#!/usr/bin/env python3
"""Generate deterministic train-probe trial list from train-clean-100."""

from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.utils.paths import get_data_root


def main() -> int:
    data_root = get_data_root()
    manifest = data_root / "manifests" / "train-clean-100.csv"
    out_dir = data_root / "trials"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    subprocess.check_call([
        sys.executable,
        str(REPO_ROOT / "src/data/trial_generator.py"),
        "--manifest", str(manifest),
        "--targets-per-speaker", "14",
        "--seed", "42",
        "--out-dir", str(out_dir),
    ])

    src_txt = out_dir / "trials_train-clean-100.txt"
    src_meta = out_dir / "trials_train-clean-100_meta.csv"
    dst_txt = out_dir / "trials_train-probe.txt"
    dst_meta = out_dir / "trials_train-probe_meta.csv"

    shutil.copy(src_txt, dst_txt)
    shutil.copy(src_meta, dst_meta)
    print(f"Generated train-probe trials: {dst_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

