#!/usr/bin/env python3
"""LibriSpeech verification trial list generator.

Generates evaluation trial lists adhering to strict biometric verification criteria:
1. Cross-Chapter Target Trials: Genuine pairs (label=1) must strictly originate from
   different book chapters (chapter_1 != chapter_2) to eliminate acoustic room tone
   and recording condition shortcuts.
2. Single-Chapter Speaker Exclusion: Speakers appearing in only one chapter cannot
   produce valid cross-chapter target pairs and are STRICTLY EXCLUDED from targets
   (never falling back to same-chapter trials).
3. Gender-Matched Impostor Trials: Impostor pairs (label=0) are sampled strictly
   from distinct speakers of the SAME gender (gender_1 == gender_2, spk_1 != spk_2).
4. Uniform Target Sampling: Targets are sampled uniformly across qualifying speakers.
5. Reuse & Volume Transparency: Computes empirical reuse statistics and writes
   trials_report.json with SHA-256 integrity checksums.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import pandas as pd

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import PROJECT_ROOT, get_data_root, resolve_path


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class TrialGenerator:
    """Generates cross-chapter target and gender-matched impostor trial lists."""

    def __init__(
        self,
        manifest_path: Path,
        min_duration: float = 2.0,
        seed: int = 42,
    ) -> None:
        """Initialize trial generator.

        Args:
            manifest_path: Path to CSV manifest created by prep_librispeech.py.
            min_duration: Minimum utterance duration in seconds (default 2.0 s).
            seed: Random seed for deterministic reproducibility (default 42).
        """
        self.manifest_path = Path(manifest_path).resolve()
        self.min_duration = min_duration
        self.seed = seed
        self.rng = random.Random(seed)

        self._load_and_filter_data()

    def _load_and_filter_data(self) -> None:
        """Load manifest and filter by duration."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        self.total_raw_utterances = len(df)
        self.raw_speakers = sorted(df["spk_id"].astype(str).unique())

        # Filter by minimum duration
        df_filtered = df[df["duration"] >= self.min_duration].copy()
        self.total_filtered_utterances = len(df_filtered)

        # Structure by speaker and chapter
        # spk -> chapter -> list of {utt_id, path, gender, duration}
        self.spk_data: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self.spk_gender: dict[str, str] = {}
        self.utt_to_record: dict[str, dict[str, Any]] = {}

        for _, row in df_filtered.iterrows():
            spk = str(row["spk_id"])
            ch = str(row["chapter_id"])
            gender = str(row["gender"]).upper()
            utt_id = str(row["utt_id"])
            rec = {
                "utt_id": utt_id,
                "spk_id": spk,
                "chapter_id": ch,
                "gender": gender,
                "path": str(row["path"]),
                "duration": float(row["duration"]),
            }
            self.spk_data[spk][ch].append(rec)
            self.spk_gender[spk] = gender
            self.utt_to_record[utt_id] = rec

        # Segregate single-chapter vs multi-chapter speakers
        self.single_chapter_speakers: list[dict[str, Any]] = []
        self.qualifying_speakers: list[str] = []

        for spk in sorted(self.spk_data.keys()):
            chapters = list(self.spk_data[spk].keys())
            if len(chapters) == 1:
                self.single_chapter_speakers.append(
                    {
                        "spk_id": spk,
                        "chapter_id": chapters[0],
                        "gender": self.spk_gender[spk],
                        "utterances_count": len(self.spk_data[spk][chapters[0]]),
                        "reason": "single_chapter",
                    }
                )
            else:
                self.qualifying_speakers.append(spk)

    def generate_trials(
        self,
        targets_per_speaker: int | None = None,
        max_total_trials: int | None = None,
    ) -> tuple[list[tuple[int, str, str]], dict[str, Any]]:
        """Generate balanced target and impostor trial pairs.

        Args:
            targets_per_speaker: Number of cross-chapter targets to sample per qualifying speaker.
                If None, automatically computed from what the dataset supports.
            max_total_trials: Optional cap on total trial count.

        Returns:
            tuple of (trials_list, report_dict):
                trials_list: List of tuples (label, path1, path2)
                report_dict: Comprehensive audit metadata for trials_report.json
        """
        # Determine available cross-chapter pairs per qualifying speaker
        spk_cross_pairs: dict[str, list[tuple[str, str]]] = {}
        for spk in self.qualifying_speakers:
            chs = list(self.spk_data[spk].keys())
            pairs: list[tuple[str, str]] = []
            for i in range(len(chs)):
                for j in range(i + 1, len(chs)):
                    ch_a, ch_b = chs[i], chs[j]
                    utts_a = self.spk_data[spk][ch_a]
                    utts_b = self.spk_data[spk][ch_b]
                    for ua in utts_a:
                        for ub in utts_b:
                            # Order deterministically
                            p1, p2 = (ua["path"], ub["path"]) if ua["path"] < ub["path"] else (ub["path"], ua["path"])
                            pairs.append((p1, p2))
            # Shuffle pairs reproducibly
            self.rng.shuffle(pairs)
            spk_cross_pairs[spk] = pairs

        # Determine target count per speaker
        if targets_per_speaker is None:
            # Sized based on data: choose a count that keeps utterance reuse modest (~2-4x)
            # while ensuring fair representation across speakers
            median_pairs = (
                int(pd.Series([len(p) for p in spk_cross_pairs.values()]).median())
                if spk_cross_pairs
                else 0
            )
            # Target count: min across speakers or capped at 100 per speaker
            min_pairs = min(len(p) for p in spk_cross_pairs.values()) if spk_cross_pairs else 0
            targets_per_speaker = min(100, max(20, min_pairs))

        target_trials: list[tuple[int, str, str]] = []
        per_spk_target_counts: dict[str, int] = {}

        for spk in self.qualifying_speakers:
            available = spk_cross_pairs[spk]
            k = min(targets_per_speaker, len(available))
            selected = available[:k]
            for p1, p2 in selected:
                target_trials.append((1, p1, p2))
            per_spk_target_counts[spk] = k

        num_targets = len(target_trials)

        # Generate gender-matched impostors
        # Pool all speakers (including single-chapter speakers) by gender
        male_spks = [s for s, g in self.spk_gender.items() if g == "M"]
        female_spks = [s for s, g in self.spk_gender.items() if g == "F"]

        impostor_trials: list[tuple[int, str, str]] = []
        per_spk_impostor_counts: dict[str, int] = Counter()
        impostor_pairs_seen: set[tuple[str, str]] = set()

        # Split target count evenly between male and female impostor pairs
        target_impostors = num_targets
        impostors_per_gender = target_impostors // 2
        extra = target_impostors % 2

        for gender, spk_list, needed in [
            ("M", male_spks, impostors_per_gender + extra),
            ("F", female_spks, impostors_per_gender),
        ]:
            if len(spk_list) < 2:
                raise ValueError(f"Need at least 2 speakers of gender {gender} for impostor trials.")

            gender_pairs: list[tuple[str, str]] = []
            attempts = 0
            max_attempts = needed * 50

            while len(gender_pairs) < needed and attempts < max_attempts:
                attempts += 1
                spk1, spk2 = self.rng.sample(spk_list, 2)
                # Sample one utterance from each speaker
                ch1 = self.rng.choice(list(self.spk_data[spk1].keys()))
                ch2 = self.rng.choice(list(self.spk_data[spk2].keys()))
                u1 = self.rng.choice(self.spk_data[spk1][ch1])
                u2 = self.rng.choice(self.spk_data[spk2][ch2])

                p1, p2 = (u1["path"], u2["path"]) if u1["path"] < u2["path"] else (u2["path"], u1["path"])
                pair_key = (p1, p2)
                if pair_key not in impostor_pairs_seen:
                    impostor_pairs_seen.add(pair_key)
                    gender_pairs.append(pair_key)
                    per_spk_impostor_counts[spk1] += 1
                    per_spk_impostor_counts[spk2] += 1

            for p1, p2 in gender_pairs:
                impostor_trials.append((0, p1, p2))

        # Combine all trials and shuffle
        all_trials = target_trials + impostor_trials
        self.rng.shuffle(all_trials)

        if max_total_trials is not None and len(all_trials) > max_total_trials:
            all_trials = all_trials[:max_total_trials]

        # Calculate utterance reuse statistics
        utterance_counter: Counter[str] = Counter()
        for _, p1, p2 in all_trials:
            utterance_counter[p1] += 1
            utterance_counter[p2] += 1

        reuse_values = list(utterance_counter.values())
        mean_reuse = float(pd.Series(reuse_values).mean()) if reuse_values else 0.0
        median_reuse = float(pd.Series(reuse_values).median()) if reuse_values else 0.0
        max_reuse = int(max(reuse_values)) if reuse_values else 0

        # Reuse distribution buckets
        reuse_dist: dict[str, int] = {
            "1": sum(1 for v in reuse_values if v == 1),
            "2": sum(1 for v in reuse_values if v == 2),
            "3_to_5": sum(1 for v in reuse_values if 3 <= v <= 5),
            "6_to_10": sum(1 for v in reuse_values if 6 <= v <= 10),
            "over_10": sum(1 for v in reuse_values if v > 10),
        }

        total_unique_spks = len(self.spk_data)
        num_single_ch = len(self.single_chapter_speakers)

        report: dict[str, Any] = {
            "manifest_file": str(self.manifest_path.name),
            "random_seed": self.seed,
            "min_duration_sec": self.min_duration,
            "total_raw_utterances": self.total_raw_utterances,
            "total_filtered_utterances": self.total_filtered_utterances,
            "total_speakers_in_manifest": len(self.raw_speakers),
            "total_speakers_post_filtering": total_unique_spks,
            "num_excluded_single_chapter_speakers": num_single_ch,
            "fraction_excluded_speakers": round(num_single_ch / total_unique_spks, 4) if total_unique_spks > 0 else 0.0,
            "single_chapter_policy": "Excluded from targets; retained for gender-matched impostors",
            "excluded_speakers": self.single_chapter_speakers,
            "num_qualifying_speakers": len(self.qualifying_speakers),
            "target_trials_count": len(target_trials),
            "impostor_trials_count": len(impostor_trials),
            "total_trials_count": len(all_trials),
            "target_to_impostor_ratio": round(len(target_trials) / max(1, len(impostor_trials)), 4),
            "per_speaker_target_counts": per_spk_target_counts,
            "per_speaker_impostor_counts": dict(sorted(per_spk_impostor_counts.items())),
            "utterance_reuse_statistics": {
                "unique_utterances_in_corpus": self.total_filtered_utterances,
                "unique_utterances_used": len(utterance_counter),
                "utterance_utilization_rate": round(len(utterance_counter) / self.total_filtered_utterances, 4) if self.total_filtered_utterances > 0 else 0.0,
                "mean_reuse_count": round(mean_reuse, 2),
                "median_reuse_count": round(median_reuse, 2),
                "max_reuse_count": max_reuse,
                "reuse_distribution_histogram": reuse_dist,
            },
        }

        return all_trials, report


def save_trials_and_report(
    trials: list[tuple[int, str, str]],
    report: dict[str, Any],
    out_trials_path: Path,
    out_report_path: Path,
) -> None:
    """Save trial pairs to text file and audit report to JSON with SHA-256."""
    out_trials_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_trials_path, "w", encoding="utf-8") as f:
        for label, p1, p2 in trials:
            f.write(f"{label} {p1} {p2}\n")

    # Compute checksum of written trials file
    trials_sha256 = compute_sha256(out_trials_path)
    report["trial_file_sha256"] = trials_sha256
    report["trial_file_name"] = out_trials_path.name

    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved {len(trials)} trials to {out_trials_path} (SHA-256: {trials_sha256[:16]}...)")
    print(f"Saved verification audit report to {out_report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LibriSpeech verification trial lists.")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to partition CSV manifest (e.g. data/manifests/test-clean.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(get_data_root() / "trials"),
        help="Directory to store trial files and reports.",
    )
    parser.add_argument(
        "--targets-per-speaker",
        type=int,
        default=None,
        help="Number of target trials per qualifying speaker (default: data-supported volume).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum duration in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    split_name = manifest_path.stem

    generator = TrialGenerator(
        manifest_path=manifest_path,
        min_duration=args.min_duration,
        seed=args.seed,
    )
    trials, report = generator.generate_trials(
        targets_per_speaker=args.targets_per_speaker,
    )

    out_trials_path = out_dir / f"trials_{split_name}.txt"
    out_report_path = out_dir / f"trials_report_{split_name}.json"

    save_trials_and_report(trials, report, out_trials_path, out_report_path)


if __name__ == "__main__":
    main()

