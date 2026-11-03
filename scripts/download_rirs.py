#!/usr/bin/env python3
"""Download and extract OpenSLR 28 (RIRS_NOISES) with resume and integrity checks.

Downloads official rirs_noises.zip (1.22 GiB / ~1.31 GB) from OpenSLR mirrors,
verifies file size/integrity, extracts simulated RIRs per docs/DATA_PROTOCOL.md Section 5.1,
and deletes the temporary zip archive immediately to preserve the >= 8 GB free disk buffer.
"""

import argparse
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

# Project root path resolution
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_data_root, resolve_path

OPENSLR_RIR_URLS = [
    "https://openslr.elda.org/resources/28/rirs_noises.zip",
    "https://www.openslr.org/resources/28/rirs_noises.zip",
    "https://openslr.trmal.net/resources/28/rirs_noises.zip",
]

EXPECTED_ZIP_SIZE = 1311166223  # 1,311,166,223 bytes (~1.22 GiB)


def get_free_disk_gb(path: Path) -> float:
    """Return available disk space in GB."""
    stat = shutil.disk_usage(path if path.exists() else path.parent)
    return stat.free / (1024**3)


def download_rirs_zip(dest_path: Path, max_retries: int = 50) -> None:
    """Download rirs_noises.zip with HTTP Range resume support across mirrors."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    mirror_idx = 0
    retries = 0

    while True:
        existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0
        if existing_bytes == EXPECTED_ZIP_SIZE:
            print(f"  [Complete] rirs_noises.zip is fully downloaded ({existing_bytes:,} bytes).")
            return
        elif existing_bytes > EXPECTED_ZIP_SIZE:
            print(f"  [Warning] Existing file larger than expected. Restarting download.")
            dest_path.unlink()
            existing_bytes = 0

        url = OPENSLR_RIR_URLS[mirror_idx % len(OPENSLR_RIR_URLS)]
        headers = {"User-Agent": "Neurovox-Downloader/1.0"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            print(f"  [Resume] Resuming from {existing_bytes:,} / {EXPECTED_ZIP_SIZE:,} bytes via {url}...")
        else:
            print(f"  [Download] Starting download ({EXPECTED_ZIP_SIZE:,} bytes) via {url}...")

        req = urllib.request.Request(url, headers=headers)
        chunk_size = 1024 * 1024  # 1 MB chunk

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
                        if now - last_log > 5.0 or downloaded == EXPECTED_ZIP_SIZE:
                            pct = (downloaded / EXPECTED_ZIP_SIZE) * 100.0
                            elapsed = max(0.1, now - start_time)
                            rate_mb = (downloaded - existing_bytes) / (1024 * 1024 * elapsed)
                            print(
                                f"  [rirs_noises.zip] {downloaded:,} / {EXPECTED_ZIP_SIZE:,} bytes "
                                f"({pct:.1f}%) @ {rate_mb:.2f} MB/s"
                            )
                            last_log = now

            if dest_path.exists() and dest_path.stat().st_size == EXPECTED_ZIP_SIZE:
                print(f"  [Success] Successfully downloaded rirs_noises.zip ({EXPECTED_ZIP_SIZE:,} bytes).")
                return
            else:
                curr = dest_path.stat().st_size if dest_path.exists() else 0
                print(f"  [Incomplete] Download ended at {curr:,} bytes. Resuming...")
                mirror_idx += 1
                retries += 1
                if retries > max_retries:
                    raise RuntimeError("Exceeded maximum retries downloading rirs_noises.zip.")
                time.sleep(2.0)

        except Exception as e:
            curr = dest_path.stat().st_size if dest_path.exists() else 0
            print(f"  [Connection error] {url}: {e} (saved {curr:,} bytes). Retrying in 2s...")
            mirror_idx += 1
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Exceeded maximum retries ({max_retries}). Last error: {e}")
            time.sleep(2.0)


def extract_rirs(
    zip_path: Path,
    target_dir: Path,
    subset: str = "simulated-small-medium",
) -> int:
    """Extract selected RIR files from zip archive to target directory.

    Args:
        zip_path: Path to rirs_noises.zip.
        target_dir: Directory where files will be extracted (e.g. data/raw/RIRS_NOISES).
        subset: Extraction filter ('simulated-small-medium', 'simulated-all', or 'all').

    Returns:
        Number of files extracted.
    """
    print(f"\n--> Extracting subset '{subset}' to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_members = len(members)

        for idx, info in enumerate(members):
            filename = info.filename
            if info.is_dir():
                continue

            # Apply subset filter
            should_extract = False
            if subset == "all":
                should_extract = True
            elif subset == "simulated-all":
                should_extract = "simulated_rirs" in filename
            elif subset == "simulated-small-medium":
                # Matches docs/DATA_PROTOCOL.md Section 5.1 specification
                should_extract = (
                    "simulated_rirs/smallroom" in filename
                    or "simulated_rirs/mediumroom" in filename
                )

            if should_extract:
                zf.extract(info, path=target_dir)
                extracted_count += 1
                if extracted_count % 5000 == 0:
                    print(f"  Extracted {extracted_count:,} files...")

    print(f"  [Complete] Extracted {extracted_count:,} RIR files to {target_dir}.")
    return extracted_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract OpenSLR 28 (RIRS_NOISES).")
    parser.add_argument(
        "--dest-zip",
        type=str,
        default=str(get_data_root() / "raw" / "rirs_noises.zip"),
        help="Path where downloaded zip file is saved (default: data/raw/rirs_noises.zip)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        default=True,
        help="Only download the zip archive without extracting it (default: True).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the archive after downloading.",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=str(get_data_root() / "raw" / "RIRS_NOISES"),
        help="Target extraction directory if extracting (default: data/raw/RIRS_NOISES)",
    )
    parser.add_argument(
        "--subset",
        choices=["simulated-small-medium", "simulated-all", "all"],
        default="simulated-small-medium",
        help=(
            "Extraction subset: 'simulated-small-medium' (default, ~0.9 GB extracted, preserving >=8 GB disk buffer), "
            "'simulated-all' (~2.1 GB extracted), or 'all' (~3.5 GB extracted)."
        ),
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep rirs_noises.zip after extraction (default: False, per Tarball Cleanup Policy).",
    )
    args = parser.parse_args()

    dest_zip_path = resolve_path(args.dest_zip)
    dest_dir = resolve_path(args.dest_dir)
    do_extract = args.extract and not args.no_extract

    free_gb_start = get_free_disk_gb(dest_zip_path.parent)
    print("==================================================================")
    print(" OpenSLR 28 (RIRS_NOISES) Downloader")
    print("==================================================================")
    print(f"Target Zip File:     {dest_zip_path}")
    print(f"Extract Archive:     {do_extract}")
    print(f"Download Size:       {EXPECTED_ZIP_SIZE:,} bytes (~1.22 GiB)")
    print(f"Current Free Disk:   {free_gb_start:.2f} GiB")
    print("==================================================================")

    # Download archive directly to dest_zip_path with resume support
    download_rirs_zip(dest_zip_path)

    if not do_extract:
        free_gb_end = get_free_disk_gb(dest_zip_path.parent)
        print("\n==================================================================")
        print(" RIRS_NOISES Download Complete (Extraction Skipped)")
        print("==================================================================")
        print(f"Downloaded Zip:      {dest_zip_path} ({dest_zip_path.stat().st_size:,} bytes)")
        print(f"Final Free Disk:     {free_gb_end:.2f} GiB (Buffer preserved: {free_gb_end >= 8.0})")
        print("==================================================================")
        return

    # Extract if requested
    extracted_count = extract_rirs(dest_zip_path, dest_dir, subset=args.subset)

    # Cleanup zip per standing policy if extracting and not keeping zip
    if not args.keep_zip:
        print(f"\n--> Cleaning up temporary zip {dest_zip_path.name} to preserve disk buffer...")
        dest_zip_path.unlink(missing_ok=True)

    free_gb_end = get_free_disk_gb(dest_dir.parent)
    print("\n==================================================================")
    print(" RIRS_NOISES Preparation Complete")
    print("==================================================================")
    print(f"Files Extracted:     {extracted_count:,}")
    print(f"Location:            {dest_dir}")
    print(f"Final Free Disk:     {free_gb_end:.2f} GiB (Buffer preserved: {free_gb_end >= 8.0})")
    print("==================================================================")


if __name__ == "__main__":
    main()

