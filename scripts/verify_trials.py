#!/usr/bin/env python3
"""Cross-machine verification of generated evaluation trial lists against committed v2 checksums.

Ensures bitwise cross-machine reproducibility between local development (Mac M3 Pro)
and cloud environments (Google Colab, fresh VMs).

If any regenerated trial list or metadata file deviates from the cryptographically
committed SHA-256 checksums established in ADR-017, this script raises a loud,
unambiguous error and terminates execution immediately.
"""

import argparse
import hashlib
from pathlib import Path
import sys

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_data_root, resolve_path

# Cryptographically committed v2 trial list and metadata checksums (ADR-017 + Phase 2A train-probe)
COMMITTED_V2_CHECKSUMS: dict[str, str] = {
    "trials_test-clean.txt": "5b47db5e847990a75b35edfc5d906944dd069a59144df7c6b5f5126388ee6aa1",
    "trials_test-other.txt": "92e651bccc3af6253b6c97437d3197812cf384490cbe16cd812253779651a9a6",
    "trials_dev-clean.txt": "2025424ac8116efa2ab5a008cb112d4598a7c2e8b28f30abb6be81680312d32d",
    "trials_test-clean_meta.csv": "2b7ea81443af82a272e7bf0eff5a5e57ebf38e372f07efa421cc8415ef75d2bf",
    "trials_test-other_meta.csv": "7284487deb0ecea8b50cf090802222b8e789ae7287b0a68ab2eb1e44b466e0a4",
    "trials_dev-clean_meta.csv": "f11c9755fb4f2a464f79a6c0fec6a8e8597b832e2a93262a750b07a355271cc1",
    "trials_train-probe.txt": "923d22e5533ecb442787295901ea98dc301948a550ae4cd1440c57da53aa8b7f",
    "trials_train-probe_meta.csv": "cc56e15c96852f601ecb8fd1aa1d6d04329a3d4b770440ef246c562d96af20f8",
}

# Cryptographically committed manifest checksums
COMMITTED_MANIFEST_CHECKSUMS: dict[str, str] = {
    "train-50spk.csv": "f63c4dc99eb41b45880fcf527aa724160e6b4860390cc7c0221a06a27f5f46da",
    "train-clean-100.csv": "212d53386080f4a73941a1893b46a1852c41013a8b8de0669b47bb75b7ad1862",
    "dev-clean.csv": "d906995bf50db3c29aa45734ad088d6f82d082782cc437c8ceb169f5a769cea4",
    "test-clean.csv": "832fa18e8e8a03fb3b2e24a1fe2e7d0f555366972a91bce916ff1fbc66092379",
    "test-other.csv": "fef5b8b7a09a9f461c6e174dc363f1544026879704072205bf43df410bf9bffe",
    "train_spk2label.json": "15ab0fbdd43ed559439c2f91b9cb84881645879d5664a8142cfabfb0ae046c49",
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_trials(
    trials_dir: Path | str,
    manifests_dir: Path | str | None = None,
    strict: bool = True,
) -> dict[str, dict[str, str | bool]]:
    """Verify trial list, companion metadata, and subset manifest SHA-256 hashes against committed values.

    Args:
        trials_dir: Directory containing generated trials (e.g. data/trials).
        manifests_dir: Optional directory containing manifests (e.g. data/manifests).
        strict: If True, raises RuntimeError immediately upon the first checksum mismatch.

    Returns:
        Dictionary mapping filename to verification status details.

    Raises:
        FileNotFoundError: If a required trial list or metadata file is missing.
        RuntimeError: If strict=True and a computed checksum differs from the committed hash.
    """
    directory = resolve_path(trials_dir)
    results: dict[str, dict[str, str | bool]] = {}
    mismatches: list[str] = []

    print("=" * 80)
    print("CROSS-MACHINE TRIAL & DATA REPRODUCIBILITY AUDIT (COMMITTED CHECKSUMS)")
    print(f"Audit directory: {directory.resolve()}")
    print("=" * 80)

    for filename, expected_hash in COMMITTED_V2_CHECKSUMS.items():
        file_path = directory / filename
        if not file_path.exists():
            msg = f"MISSING FILE: Required trial file does not exist: {file_path}"
            print(f"[-] {filename:<30} ERROR: File not found")
            if strict:
                raise FileNotFoundError(msg)
            results[filename] = {"status": False, "error": "FileNotFound"}
            continue

        actual_hash = compute_sha256(file_path)
        matches = actual_hash.lower() == expected_hash.lower()

        results[filename] = {
            "status": matches,
            "expected_sha256": expected_hash,
            "computed_sha256": actual_hash,
        }

        if matches:
            print(f"[PASS] {filename:<28} SHA-256: {actual_hash[:16]}... (matches committed reference)")
        else:
            print(f"[FAIL] {filename:<28} MISMATCH!")
            print(f"       Expected: {expected_hash}")
            print(f"       Actual:   {actual_hash}")
            mismatches.append(
                f"Checksum mismatch for '{filename}':\n"
                f"  Expected: {expected_hash}\n"
                f"  Computed: {actual_hash}"
            )

    # Check subset manifests if manifests_dir is provided or present in data root
    target_manifests_dir = None
    if manifests_dir is not None:
        target_manifests_dir = resolve_path(manifests_dir)
    elif (directory.parent / "manifests").exists():
        target_manifests_dir = directory.parent / "manifests"

    if target_manifests_dir and target_manifests_dir.exists():
        print("-" * 80)
        print(f"Auditing manifests in: {target_manifests_dir.resolve()}")
        for m_filename, m_expected_hash in COMMITTED_MANIFEST_CHECKSUMS.items():
            m_path = target_manifests_dir / m_filename
            if not m_path.exists():
                if manifests_dir is not None:
                    msg = f"MISSING MANIFEST: Required manifest does not exist: {m_path}"
                    if strict:
                        raise FileNotFoundError(msg)
                continue

            m_actual = compute_sha256(m_path)
            m_matches = m_actual.lower() == m_expected_hash.lower()
            results[m_filename] = {
                "status": m_matches,
                "expected_sha256": m_expected_hash,
                "computed_sha256": m_actual,
            }
            if m_matches:
                print(f"[PASS] {m_filename:<28} SHA-256: {m_actual[:16]}... (matches committed reference)")
            else:
                print(f"[FAIL] {m_filename:<28} MISMATCH!")
                mismatches.append(
                    f"Checksum mismatch for manifest '{m_filename}':\n"
                    f"  Expected: {m_expected_hash}\n"
                    f"  Computed: {m_actual}"
                )

    print("=" * 80)
    if mismatches:
        error_msg = (
            "CRITICAL: Cross-machine trial reproducibility check FAILED!\n"
            + "\n".join(mismatches)
            + "\nStopping execution. Ensure data prep and trial generation ran deterministically."
        )
        if strict:
            raise RuntimeError(error_msg)
        print("STATUS: FAILED")
        return results

    print("STATUS: PASS (All trial lists bitwise identical to committed v2)")
    print("=" * 80)
    return results


def verify_data_integrity(
    manifests_dir: Path | str | None = None,
    trials_dir: Path | str | None = None,
    check_audio: bool = True,
) -> bool:
    """Verify partition counts, speaker disjointness, and cross-chapter target integrity.

    Verifies:
      1. Speaker counts: train-clean-100 (251), dev-clean (40), test-clean (40), test-other (33).
      2. Utterance counts: train-clean-100 (28,539), dev-clean (2,703), test-clean (2,620), test-other (2,939).
      3. Strict pairwise speaker disjointness across all partitions.
      4. 100% cross-chapter guarantee on genuine target trials.
      5. Audio file existence on disk (sample check).
    """
    import csv

    m_dir = resolve_path(manifests_dir) if manifests_dir else get_data_root() / "manifests"
    t_dir = resolve_path(trials_dir) if trials_dir else get_data_root() / "trials"

    expected_specs = {
        "train-clean-100": {"speakers": 251, "utterances": 28539},
        "dev-clean": {"speakers": 40, "utterances": 2703},
        "test-clean": {"speakers": 40, "utterances": 2620},
        "test-other": {"speakers": 33, "utterances": 2939},
    }

    print("\n" + "=" * 80)
    print("LIBRISPEECH DATA PROTOCOL & INTEGRITY AUDIT")
    print(f"Manifests directory: {m_dir.resolve()}")
    print("=" * 80)

    partition_speakers: dict[str, set[str]] = {}
    all_passed = True

    for split, spec in expected_specs.items():
        csv_path = m_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[-] {split:<18} ERROR: Manifest not found: {csv_path}")
            all_passed = False
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        spks = set(row["spk_id"] for row in reader)
        partition_speakers[split] = spks
        n_spk = len(spks)
        n_utt = len(reader)

        spk_ok = n_spk == spec["speakers"]
        utt_ok = n_utt == spec["utterances"]

        status = "[PASS]" if (spk_ok and utt_ok) else "[FAIL]"
        print(
            f"{status} {split:<18} Speakers: {n_spk:>3}/{spec['speakers']:<3} | "
            f"Utterances: {n_utt:>5}/{spec['utterances']:<5}"
        )
        if not (spk_ok and utt_ok):
            all_passed = False

        # Optional audio existence check on sample of rows
        if check_audio and len(reader) > 0:
            sample = reader[:5] + reader[-5:]
            existing = sum(1 for r in sample if resolve_path(r["path"]).exists())
            if existing == len(sample):
                print(f"       -> Audio files verified on disk for {split} (sample: {existing}/{len(sample)} found)")
            else:
                print(f"       -> [NOTICE] Sample audio files not yet extracted on disk ({existing}/{len(sample)} found)")

    # Assert pairwise speaker disjointness
    print("-" * 80)
    print("Auditing speaker disjointness across partitions...")
    splits = list(partition_speakers.keys())
    disjoint_ok = True
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = partition_speakers[s1].intersection(partition_speakers[s2])
            if overlap:
                print(f"[FAIL] Speaker leakage detected between {s1} and {s2}: {overlap}")
                disjoint_ok = False
                all_passed = False
    if disjoint_ok:
        print("[PASS] 100% Disjoint: Zero speaker overlap across train, dev, test-clean, and test-other.")

    # Verify cross-chapter target guarantee in trial files
    print("-" * 80)
    print("Auditing 100% cross-chapter target trials guarantee...")
    for tf_name in ["trials_dev-clean.txt", "trials_test-clean.txt", "trials_test-other.txt"]:
        tf_path = t_dir / tf_name
        if not tf_path.exists():
            continue
        total_targets = 0
        cross_chapter = 0
        with open(tf_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] == "1":
                    total_targets += 1
                    ch1 = Path(parts[1]).parent.name
                    ch2 = Path(parts[2]).parent.name
                    if ch1 != ch2:
                        cross_chapter += 1
        pct = (cross_chapter / total_targets * 100) if total_targets else 0
        cc_ok = cross_chapter == total_targets
        status = "[PASS]" if cc_ok else "[FAIL]"
        print(f"{status} {tf_name:<24} Cross-chapter: {cross_chapter}/{total_targets} ({pct:.1f}%)")
        if not cc_ok:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("STATUS: PASS (Dataset protocol and trial structure strictly verified)")
    else:
        print("STATUS: FAIL (Discrepancies found in data or trial structure)")
    print("=" * 80)
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify trial lists against committed v2 SHA-256 checksums.")
    parser.add_argument(
        "--trials-dir",
        type=str,
        default=str(get_data_root() / "trials"),
        help="Path to directory containing trial lists (default: DATA_ROOT/trials)",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Also verify partition manifests, speaker counts, disjointness, and cross-chapter targets.",
    )
    args = parser.parse_args()

    try:
        verify_trials(args.trials_dir, strict=True)
        if args.check_data:
            data_ok = verify_data_integrity(trials_dir=args.trials_dir)
            if not data_ok:
                return 1
        return 0
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

