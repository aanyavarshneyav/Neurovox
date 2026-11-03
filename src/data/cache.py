"""Disk memmap audio cache reader and builder for high-throughput I/O.

Implements int16 binary memmap storage and structured indexing for audio datasets,
per docs/DATA_PROTOCOL.md Section 4 and docs/BENCHMARKS.md.

Note on caching policy:
    Empirical measurements in docs/BENCHMARKS.md show that on-the-fly FLAC decoding
    via soundfile with num_workers=4 achieves 26.36 ms/batch (37.94 batches/sec),
    which is 4.5x to 15.6x faster than model forward+backward step times
    (116.51 ms/step for lower-bound baseline, 411.52 ms/step for full SKA-HNN).
    Because asynchronous background pre-fetching completely hides I/O latency,
    and available disk space on the Mac M3 Pro is ~27.5 GB (requiring a >= 8 GB free buffer),
    the dataset defaults to on-the-fly FLAC decoding. This module provides complete,
    verified memmap caching capabilities whenever disk-cache execution is enabled.
"""

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Optional
import numpy as np
import soundfile as sf
import torch

from src.utils.paths import PROJECT_ROOT, resolve_path


class AudioMemmapWriter:
    """Writes raw audio to an int16 contiguous binary memmap and builds an index."""

    def __init__(self, output_dir: Path, bin_name: str = "audio_memmap.bin", index_name: str = "audio_index.json") -> None:
        """Initialize writer with output directory and filenames.

        Args:
            output_dir: Directory where binary memmap and index are written.
            bin_name: Name of binary file (default 'audio_memmap.bin').
            index_name: Name of index JSON file (default 'audio_index.json').
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bin_path = self.output_dir / bin_name
        self.index_path = self.output_dir / index_name

    def write_from_manifest(
        self,
        manifest_csv_path: Path,
        max_utterances: Optional[int] = None,
        min_free_disk_gb: float = 8.0,
    ) -> dict[str, Any]:
        """Convert audio files from manifest into int16 memmap and write index.

        Args:
            manifest_csv_path: Path to manifest CSV (e.g. train-clean-100.csv).
            max_utterances: Optional limit on number of utterances to process.
            min_free_disk_gb: Minimum free disk buffer required in GB.

        Returns:
            Dictionary summary of built cache.
        """
        manifest_csv_path = resolve_path(manifest_csv_path)
        if not manifest_csv_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_csv_path}")

        # Read manifest entries
        entries: list[dict[str, str]] = []
        with open(manifest_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
                if max_utterances is not None and len(entries) >= max_utterances:
                    break

        if not entries:
            raise ValueError(f"Manifest {manifest_csv_path} contains 0 entries.")

        # Estimate disk space required
        total_duration_sec = sum(float(r.get("duration", 0.0)) for r in entries)
        est_samples = int(total_duration_sec * 16000)
        est_bytes = est_samples * 2  # 2 bytes per int16 sample
        est_gb = est_bytes / (1024**3)

        # Check available free disk space
        stat = shutil.disk_usage(self.output_dir)
        free_gb = stat.free / (1024**3)
        if (free_gb - est_gb) < min_free_disk_gb:
            raise RuntimeError(
                f"Cannot build cache: requires {est_gb:.2f} GB, but writing would leave "
                f"{(free_gb - est_gb):.2f} GB free, which violates the {min_free_disk_gb:.1f} GB buffer."
            )

        # Sequentially write int16 samples and construct index
        index_data: dict[str, Any] = {
            "version": "1.0",
            "dtype": "int16",
            "sample_rate": 16000,
            "bin_file": self.bin_path.name,
            "total_samples": 0,
            "utterances": {},
        }

        current_sample_offset = 0
        with open(self.bin_path, mode="wb") as bin_file:
            for entry in entries:
                rel_path = entry["path"]
                full_path = resolve_path(rel_path)
                audio, sr = sf.read(str(full_path), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Rescale to int16
                int16_audio = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
                num_samples = len(int16_audio)

                bin_file.write(int16_audio.tobytes())

                utt_id = entry["utt_id"]
                index_data["utterances"][utt_id] = {
                    "offset": current_sample_offset,
                    "length": num_samples,
                    "duration": float(entry.get("duration", num_samples / sr)),
                    "spk_id": entry.get("spk_id", ""),
                    "label": int(entry["label"]) if "label" in entry and entry["label"] != "" else -1,
                }
                current_sample_offset += num_samples

        index_data["total_samples"] = current_sample_offset

        # Save index JSON
        with open(self.index_path, mode="w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        return {
            "num_utterances": len(entries),
            "total_samples": current_sample_offset,
            "file_size_mb": round((current_sample_offset * 2) / (1024**2), 2),
            "bin_path": str(self.bin_path),
            "index_path": str(self.index_path),
        }


class AudioMemmapReader:
    """Reads int16 audio slices directly from disk memmap."""

    def __init__(self, index_path: Path) -> None:
        """Initialize reader from index JSON file.

        Args:
            index_path: Path to audio_index.json.
        """
        self.index_path = resolve_path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        with open(self.index_path, mode="r", encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.bin_path = self.index_path.parent / self.metadata["bin_file"]
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary cache file not found: {self.bin_path}")

        self.sample_rate = int(self.metadata.get("sample_rate", 16000))
        self.total_samples = int(self.metadata["total_samples"])
        self.utterances: dict[str, dict[str, Any]] = self.metadata["utterances"]
        self.utt_keys: list[str] = list(self.utterances.keys())

        # Open read-only memmap
        self.memmap = np.memmap(
            self.bin_path,
            dtype=np.int16,
            mode="r",
            shape=(self.total_samples,),
        )

    def __len__(self) -> int:
        return len(self.utt_keys)

    def read_audio(self, utt_id: str) -> np.ndarray:
        """Read audio for an utterance by ID, normalized to float32 [-1.0, 1.0].

        Args:
            utt_id: Utterance identifier.

        Returns:
            1D float32 numpy array.
        """
        meta = self.utterances[utt_id]
        offset = meta["offset"]
        length = meta["length"]
        raw_int16 = self.memmap[offset : offset + length]
        return (raw_int16.astype(np.float32) / 32767.0)

    def get_metadata(self, utt_id: str) -> dict[str, Any]:
        """Return utterance metadata dictionary."""
        return self.utterances[utt_id]

