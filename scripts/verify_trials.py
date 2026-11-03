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
    "trials_train-probe.txt": "381b6f38c398d487a141305a9c3d10937364d85016f3d07db97cacb85840d8c2",
    "trials_train-probe_meta.csv": "a4c44572c5c4f576e5673d2fab444d5f587c93979d8840097dfe4c543c204c17",
}

# Cryptographically committed manifest checksums
COMMITTED_MANIFEST_CHECKSUMS: dict[str, str] = {
    "train-50spk.csv": "f63c4dc99eb41b45880fcf527aa724160e6b4860390cc7c0221a06a27f5f46da",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify trial lists against committed v2 SHA-256 checksums.")
    parser.add_argument(
        "--trials-dir",
        type=str,
        default=str(get_data_root() / "trials"),
        help="Path to directory containing trial lists (default: DATA_ROOT/trials)",
    )
    args = parser.parse_args()

    try:
        verify_trials(args.trials_dir, strict=True)
        return 0
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

