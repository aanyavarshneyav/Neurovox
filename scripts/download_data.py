#!/usr/bin/env python3
"""Download and extract LibriSpeech partitions with resume support and checksum verification.

Downloads:
- train-clean-100.tar.gz (~6.38 GB)
- dev-clean.tar.gz (~338 MB)
- test-clean.tar.gz (~347 MB)
- test-other.tar.gz (~329 MB)
From OpenSLR mirrors.

Features:
- Robust HTTP Range resume support across mirrors
- Automatic reconnection on socket timeouts / IncompleteRead
- MD5 checksum verification against official OpenSLR md5sum.txt
- Sequential download -> verify -> extract -> delete tarball to maintain >= 8 GB free disk buffer
- Extraction target: data/raw/
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.request
import urllib.error

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_data_root, resolve_path

OPENSLR_BASE_URLS = [
    "https://openslr.magicdatatech.com/resources/12/",
    "https://openslr.elda.org/resources/12/",
    "https://openslr.trmal.net/resources/12/",
    "https://www.openslr.org/resources/12/",
]

# Official OpenSLR 12 MD5 checksums and sizes (bytes)
LIBRISPEECH_ARCHIVES: dict[str, dict[str, str | int]] = {
    "dev-clean.tar.gz": {
        "md5": "42e2234ba48799c1f50f24a7926300a1",
        "size_bytes": 337926286,
        "split": "dev-clean",
    },
    "test-clean.tar.gz": {
        "md5": "32fa31d27d2e1cad72775fee3f4849a9",
        "size_bytes": 346663984,
        "split": "test-clean",
    },
    "test-other.tar.gz": {
        "md5": "fb5a50374b501bb3bac4815ee91d3135",
        "size_bytes": 328757843,
        "split": "test-other",
    },
    "train-clean-100.tar.gz": {
        "md5": "2a93770f6d5c6c964bc36631d331a522",
        "size_bytes": 6387309499,
        "split": "train-clean-100",
    },
}

METADATA_FILES = [
    "SPEAKERS.TXT",
    "CHAPTERS.TXT",
    "BOOKS.TXT",
    "README.TXT",
    "LICENSE.TXT",
]


def compute_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 hash of a local file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def get_available_disk_gb(path: Path) -> float:
    """Return available disk space in gigabytes (GB) for given path."""
    stat = shutil.disk_usage(path)
    return stat.free / (1024**3)


def download_file_with_resume(
    filename: str,
    dest_path: Path,
    expected_size: int,
    base_urls: list[str] = OPENSLR_BASE_URLS,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 100,
) -> None:
    """Download a file with HTTP Range resume support across mirrors with auto-reconnection."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    mirror_idx = 0
    retries = 0

    while True:
        existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0
        if existing_bytes == expected_size:
            print(f"  [Complete] {filename} is fully downloaded ({existing_bytes:,} bytes).")
            return
        elif existing_bytes > expected_size:
            print(f"  [Warning] Existing file larger than expected. Truncating {filename}.")
            dest_path.unlink()
            existing_bytes = 0

        base_url = base_urls[mirror_idx % len(base_urls)]
        url = base_url + filename
        headers = {"User-Agent": "Neurovox-Downloader/1.0"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            print(f"  [Resume] Resuming {filename} from {existing_bytes:,} / {expected_size:,} bytes via {base_url}...")
        else:
            print(f"  [Download] Starting {filename} ({expected_size:,} bytes) via {base_url}...")

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                is_partial = resp.status == 206
                mode = "ab" if (existing_bytes > 0 and is_partial) else "wb"
                downloaded = existing_bytes if (existing_bytes > 0 and is_partial) else 0

                start_time = time.time()
                last_log = start_time

                with open(dest_path, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_log > 5.0 or downloaded == expected_size:
                            pct = (downloaded / expected_size) * 100.0 if expected_size > 0 else 0
                            elapsed = max(0.1, now - start_time)
                            rate_mb = (downloaded - existing_bytes) / (1024 * 1024 * elapsed)
                            print(
                                f"  [{filename}] {downloaded:,} / {expected_size:,} bytes "
                                f"({pct:.1f}%) @ {rate_mb:.2f} MB/s"
                            )
                            last_log = now

            if dest_path.exists() and dest_path.stat().st_size == expected_size:
                print(f"  [Success] Downloaded {filename} ({expected_size:,} bytes).")
                return
            else:
                curr_size = dest_path.stat().st_size if dest_path.exists() else 0
                print(f"  [Incomplete] Connection closed at {curr_size:,} / {expected_size:,} bytes. Resuming...")
                mirror_idx += 1
                retries += 1
                if retries > max_retries:
                    raise RuntimeError(f"Exceeded max retries for {filename}.")
                time.sleep(2.0)

        except Exception as e:
            curr_size = dest_path.stat().st_size if dest_path.exists() else 0
            print(f"  [Connection drop] {url}: {e} (saved {curr_size:,} bytes). Retrying in 2s with next mirror...")
            mirror_idx += 1
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Exceeded maximum retries ({max_retries}) for {filename}. Last error: {e}")
            time.sleep(2.0)


def extract_tarball(archive_path: Path, extract_dir: Path) -> None:
    """Extract a tarball into extract_dir."""
    print(f"  Extracting {archive_path.name} to {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print(f"  Extraction of {archive_path.name} complete.")


def copy_or_download_metadata(extract_dir: Path, base_urls: list[str] = OPENSLR_BASE_URLS) -> None:
    """Ensure root metadata files (SPEAKERS.TXT, etc.) exist in raw/LibriSpeech."""
    libri_dir = extract_dir / "LibriSpeech"
    libri_dir.mkdir(parents=True, exist_ok=True)

    alt_dir = get_data_root() / "LibriSpeech"

    for meta in METADATA_FILES:
        target = libri_dir / meta
        if target.exists():
            continue
        if (alt_dir / meta).exists():
            print(f"  Copying {meta} from {alt_dir}...")
            shutil.copy2(alt_dir / meta, target)
            continue

        for base_url in base_urls:
            url = base_url + meta
            try:
                print(f"  Downloading metadata {meta} from {url}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Neurovox/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    target.write_bytes(resp.read())
                break
            except Exception as e:
                print(f"  Failed to fetch {meta} from {url}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract LibriSpeech partitions.")
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=str(get_data_root() / "raw"),
        help="Destination directory for extraction (default: data/raw).",
    )
    parser.add_argument(
        "--check-size",
        action="store_true",
        help="Display download and extraction sizes without downloading.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm and proceed with download without interactive prompt.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100", "dev-clean", "test-clean", "test-other"],
        help="Splits to download (default: train-clean-100 dev-clean test-clean test-other).",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir).resolve()
    temp_download_dir = dest_dir / ".download_tmp"

    requested_archives = {
        name: meta
        for name, meta in LIBRISPEECH_ARCHIVES.items()
        if meta["split"] in args.splits
    }

    total_download_bytes = sum(int(m["size_bytes"]) for m in requested_archives.values())
    total_download_gb = total_download_bytes / (1024**3)
    estimated_extracted_gb = total_download_gb * 1.3
    current_free_gb = get_available_disk_gb(dest_dir.parent if dest_dir.exists() else dest_dir.parent.parent)

    print("==================================================================")
    print("LibriSpeech Download & Extraction Plan")
    print("==================================================================")
    print(f"Target raw directory:     {dest_dir}")
    print(f"Requested splits:         {', '.join(args.splits)}")
    print(f"Total download size:      {total_download_bytes:,} bytes ({total_download_gb:.2f} GiB / ~{total_download_bytes/(1000**3):.2f} GB)")
    print(f"Est. extracted size:      ~{estimated_extracted_gb:.2f} GiB")
    print(f"Current free disk space:  {current_free_gb:.2f} GiB")
    print("Archive retention policy: Delete tarballs immediately after extraction")
    print("Free space buffer target: >= 8.00 GB free disk space preserved")
    print("==================================================================")

    for name, meta in requested_archives.items():
        size_mb = int(meta["size_bytes"]) / (1024**2)
        print(f"  - {name:<24} {size_mb:>8.1f} MiB  (MD5: {meta['md5']})")
    print("==================================================================")

    if args.check_size:
        return

    if not args.confirm:
        print("\n[CONFIRMATION REQUIRED]")
        print(f"Downloading {total_download_gb:.2f} GiB requires explicit confirmation.")
        print("Please rerun with --confirm or confirm when prompted.")
        resp = input("Proceed with download and extraction? (y/N): ").strip().lower()
        if resp not in ["y", "yes"]:
            print("Download cancelled by user.")
            sys.exit(0)

    largest_archive_bytes = max(int(m["size_bytes"]) for m in requested_archives.values())
    peak_needed_gb = (largest_archive_bytes / (1024**3)) * 2.3
    if current_free_gb - peak_needed_gb < 8.0:
        print(
            f"[WARNING] Free disk buffer warning: Available ({current_free_gb:.1f} GiB) - "
            f"Peak Needed ({peak_needed_gb:.1f} GiB) < 8.0 GiB buffer!"
        )

    temp_download_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copy_or_download_metadata(dest_dir)

    for name, meta in requested_archives.items():
        split = str(meta["split"])
        libri_split_dir = dest_dir / "LibriSpeech" / split
        if libri_split_dir.exists() and any(libri_split_dir.iterdir()):
            print(f"\n[Skip] Split {split} already extracted at {libri_split_dir}.")
            continue

        archive_tmp_path = temp_download_dir / name
        expected_size = int(meta["size_bytes"])
        expected_md5 = str(meta["md5"])

        print(f"\n---> Processing {split} ({name})")
        # 1. Download with resume & auto-reconnect
        download_file_with_resume(name, archive_tmp_path, expected_size)

        # 2. Checksum verification
        print(f"  Verifying MD5 checksum for {name}...")
        actual_md5 = compute_md5(archive_tmp_path)
        if actual_md5 != expected_md5:
            archive_tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {name}! Expected {expected_md5}, computed {actual_md5}. "
                "Corrupted file removed."
            )
        print(f"  Checksum verified: {actual_md5} matches.")

        # 3. Extract to dest_dir
        extract_tarball(archive_tmp_path, dest_dir)

        # 4. Immediate tarball cleanup to preserve disk buffer
        print(f"  Deleting temporary archive {archive_tmp_path.name} to preserve disk space...")
        archive_tmp_path.unlink(missing_ok=True)

        free_now = get_available_disk_gb(dest_dir)
        print(f"  [Disk Status] Free disk space: {free_now:.2f} GiB")

    shutil.rmtree(temp_download_dir, ignore_errors=True)
    print("\n==================================================================")
    print("LibriSpeech download and extraction successfully completed.")
    print(f"Extracted directory: {dest_dir / 'LibriSpeech'}")
    print(f"Remaining free disk space: {get_available_disk_gb(dest_dir):.2f} GiB")
    print("==================================================================")


if __name__ == "__main__":
    main()
