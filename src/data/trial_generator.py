#!/usr/bin/env python3
"""LibriSpeech verification trial list generator (Version 2).

Generates evaluation trial lists adhering to strict biometric verification criteria:
1. Cross-Chapter Target Trials: Genuine pairs (label=1) strictly originate from
   different book chapters (chapter_1 != chapter_2) to eliminate acoustic room tone
   and recording condition shortcuts.
2. Single-Chapter Speaker Exclusion: Speakers appearing in only one chapter cannot
   produce valid cross-chapter target pairs and are excluded from evaluation trials
   to preserve symmetric speaker representation and duration parity.
3. Gender-Matched Impostor Trials: Impostor pairs (label=0) are sampled strictly
   from distinct speakers of the SAME gender (gender_1 == gender_2, spk_1 != spk_2).
   Target-to-impostor gender counts are matched exactly per set (H4 fix).
4. Utterance-Weighted Sampling: Impostor utterances are sampled using cross-chapter
   pair weights matching the target utterance distribution (H5 fix), guaranteeing
   mean duration difference < 3%.
5. Metadata Export: Writes trial_meta.csv beside each trial list containing exact
   speaker IDs, genders, utterance IDs, and durations for speaker-level bootstrap.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
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
            seed: Default random seed for deterministic reproducibility.
        """
        self.manifest_path = Path(manifest_path).resolve()
        self.min_duration = min_duration
        self.seed = seed

        self._load_and_filter_data()

    def _load_and_filter_data(self) -> None:
        """Load manifest, sort deterministically by utt_id, and filter by duration."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        # Deterministic ordering by utt_id regardless of row order in file
        df = df.sort_values("utt_id").reset_index(drop=True)

        self.total_raw_utterances = len(df)
        self.raw_speakers = sorted(df["spk_id"].astype(str).unique())

        # Filter by minimum duration
        df_filtered = df[df["duration"] >= self.min_duration].copy()
        self.total_filtered_utterances = len(df_filtered)

        # Track unknown gender drops
        self.dropped_unknown_gender_speakers: list[str] = []
        self.spk_gender: dict[str, str] = {}
        self.spk_data: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self.spk_utts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.utt_to_record: dict[str, dict[str, Any]] = {}

        for _, row in df_filtered.iterrows():
            spk = str(row["spk_id"])
            gender = str(row["gender"]).upper()
            if gender not in ["M", "F"]:
                if spk not in self.dropped_unknown_gender_speakers:
                    self.dropped_unknown_gender_speakers.append(spk)
                continue

            ch = str(row["chapter_id"])
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
            self.spk_utts[spk].append(rec)
            self.spk_gender[spk] = gender
            self.utt_to_record[utt_id] = rec

        # Identify single-chapter vs multi-chapter (qualifying) speakers
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
        targets_per_speaker: int | None = 100,
        max_total_trials: int | None = None,
        seed: int | None = None,
    ) -> tuple[list[tuple[int, str, str]], dict[str, Any]]:
        """Generate balanced target and impostor trial pairs with exact gender and duration matching.

        Args:
            targets_per_speaker: Number of cross-chapter targets per qualifying speaker (default 100).
            max_total_trials: Optional cap on total trial count (preserves exact 1:1 balance).
            seed: Optional random seed overriding default instance seed for fresh local RNG.

        Returns:
            tuple of (trials_list, report_dict)
        """
        rng = random.Random(seed if seed is not None else self.seed)

        # 1. Form all unique cross-chapter pairs per qualifying speaker
        spk_cross_pairs: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        spk_cross_pair_counts: dict[str, int] = {}

        for spk in self.qualifying_speakers:
            chs = list(self.spk_data[spk].keys())
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for i in range(len(chs)):
                for j in range(i + 1, len(chs)):
                    ch_a, ch_b = chs[i], chs[j]
                    utts_a = self.spk_data[spk][ch_a]
                    utts_b = self.spk_data[spk][ch_b]
                    for ua in utts_a:
                        for ub in utts_b:
                            p1, p2 = (ua, ub) if ua["utt_id"] < ub["utt_id"] else (ub, ua)
                            pairs.append((p1, p2))
            rng.shuffle(pairs)
            spk_cross_pairs[spk] = pairs
            spk_cross_pair_counts[spk] = len(pairs)

        if targets_per_speaker is None:
            targets_per_speaker = 100

        # Assert every qualifying speaker can support the required targets
        for spk in self.qualifying_speakers:
            avail = spk_cross_pair_counts[spk]
            if avail < targets_per_speaker:
                raise RuntimeError(
                    f"Qualifying speaker {spk} has only {avail} cross-chapter pairs, "
                    f"cannot satisfy requested {targets_per_speaker} targets per speaker."
                )

        # 2. Sample target trials uniformly across qualifying speakers
        target_trials: list[dict[str, Any]] = []
        per_spk_target_counts: dict[str, int] = {}
        target_genders: dict[str, int] = Counter()

        for spk in self.qualifying_speakers:
            selected = spk_cross_pairs[spk][:targets_per_speaker]
            for u1, u2 in selected:
                target_trials.append(
                    {
                        "label": 1,
                        "path_a": u1["path"],
                        "path_b": u2["path"],
                        "spk_a": u1["spk_id"],
                        "spk_b": u2["spk_id"],
                        "gender": u1["gender"],
                        "utt_a": u1["utt_id"],
                        "utt_b": u2["utt_id"],
                        "dur_a": u1["duration"],
                        "dur_b": u2["duration"],
                    }
                )
                target_genders[u1["gender"]] += 1
            per_spk_target_counts[spk] = targets_per_speaker

        # 3. Compute cross-chapter utterance weights for impostor sampling (H5 fix)
        # For each qualifying speaker, an utterance's weight is the number of cross-chapter
        # pairs it participates in, matching the target sampling marginal distribution.
        spk_utt_weights: dict[str, np.ndarray] = {}
        for spk in self.qualifying_speakers:
            utts = self.spk_utts[spk]
            chs = self.spk_data[spk]
            weights = []
            for u in utts:
                c = u["chapter_id"]
                w = sum(len(chs[c_other]) for c_other in chs if c_other != c)
                weights.append(float(w))
            w_arr = np.array(weights, dtype=np.float64)
            spk_utt_weights[spk] = w_arr / np.sum(w_arr)

        # 4. Generate gender-matched impostors strictly matching target gender counts (H4 fix)
        # Impostors use the qualifying speaker pool to preserve symmetric speaker coverage
        # and eliminate single-chapter duration skew.
        impostor_trials: list[dict[str, Any]] = []
        per_spk_impostor_counts: dict[str, int] = Counter()
        impostor_pairs_seen: set[tuple[str, str]] = set()

        for gender in ["M", "F"]:
            needed = target_genders[gender]
            spk_pool = [s for s in self.qualifying_speakers if self.spk_gender[s] == gender]
            if len(spk_pool) < 2:
                raise ValueError(f"Need at least 2 qualifying speakers of gender {gender} for impostors.")

            gender_pairs: list[dict[str, Any]] = []
            attempts = 0
            max_attempts = needed * 150

            while len(gender_pairs) < needed and attempts < max_attempts:
                attempts += 1
                s1, s2 = rng.sample(spk_pool, 2)

                # Sample utterances using target cross-chapter weights (H5 fix)
                idx1 = rng.choices(range(len(self.spk_utts[s1])), weights=spk_utt_weights[s1], k=1)[0]
                idx2 = rng.choices(range(len(self.spk_utts[s2])), weights=spk_utt_weights[s2], k=1)[0]

                u1 = self.spk_utts[s1][idx1]
                u2 = self.spk_utts[s2][idx2]

                # Deterministic path pair key
                p_key = (u1["path"], u2["path"]) if u1["path"] < u2["path"] else (u2["path"], u1["path"])
                if p_key not in impostor_pairs_seen:
                    impostor_pairs_seen.add(p_key)
                    trial_rec = {
                        "label": 0,
                        "path_a": p_key[0],
                        "path_b": p_key[1],
                        "spk_a": u1["spk_id"] if p_key[0] == u1["path"] else u2["spk_id"],
                        "spk_b": u2["spk_id"] if p_key[0] == u1["path"] else u1["spk_id"],
                        "gender": gender,
                        "utt_a": u1["utt_id"] if p_key[0] == u1["path"] else u2["utt_id"],
                        "utt_b": u2["utt_id"] if p_key[0] == u1["path"] else u1["utt_id"],
                        "dur_a": u1["duration"] if p_key[0] == u1["path"] else u2["duration"],
                        "dur_b": u2["duration"] if p_key[0] == u1["path"] else u1["duration"],
                    }
                    gender_pairs.append(trial_rec)
                    per_spk_impostor_counts[s1] += 1
                    per_spk_impostor_counts[s2] += 1

            if len(gender_pairs) < needed:
                raise RuntimeError(
                    f"Failed to produce required {needed} impostors for gender {gender} "
                    f"within {max_attempts} attempts. Generated only {len(gender_pairs)}."
                )

            impostor_trials.extend(gender_pairs)

        # 5. Handle optional max_total_trials with exact 1:1 balance preservation
        if max_total_trials is not None and (len(target_trials) + len(impostor_trials)) > max_total_trials:
            half = max_total_trials // 2
            target_trials = target_trials[:half]
            impostor_trials = impostor_trials[:half]

        # Combine and shuffle
        all_trials_records = target_trials + impostor_trials
        rng.shuffle(all_trials_records)

        # 6. Compute durations and assert < 3% relative difference (H5 assertion)
        target_avg_durs = [(r["dur_a"] + r["dur_b"]) / 2.0 for r in target_trials]
        impostor_avg_durs = [(r["dur_a"] + r["dur_b"]) / 2.0 for r in impostor_trials]

        mean_dur_target = float(np.mean(target_avg_durs))
        std_dur_target = float(np.std(target_avg_durs))
        mean_dur_impostor = float(np.mean(impostor_avg_durs))
        std_dur_impostor = float(np.std(impostor_avg_durs))

        dur_rel_diff = abs(mean_dur_target - mean_dur_impostor) / mean_dur_target
        dur_rel_diff_pct = dur_rel_diff * 100.0

        if dur_rel_diff >= 0.03:
            raise RuntimeError(
                f"Target mean duration ({mean_dur_target:.2f}s) and impostor mean duration "
                f"({mean_dur_impostor:.2f}s) differ by {dur_rel_diff_pct:.2f}% (must be < 3.0%)."
            )

        # 7. Compute utterance reuse statistics
        utterance_counter: Counter[str] = Counter()
        for r in all_trials_records:
            utterance_counter[r["path_a"]] += 1
            utterance_counter[r["path_b"]] += 1

        reuse_values = list(utterance_counter.values())
        mean_reuse = float(np.mean(reuse_values)) if reuse_values else 0.0
        median_reuse = float(np.median(reuse_values)) if reuse_values else 0.0
        max_reuse = int(max(reuse_values)) if reuse_values else 0

        percentiles = [25, 50, 75, 90, 95, 99]
        pct_stats = {f"p{p}": float(np.percentile(reuse_values, p)) for p in percentiles} if reuse_values else {}

        reuse_dist: dict[str, int] = {
            "1": sum(1 for v in reuse_values if v == 1),
            "2": sum(1 for v in reuse_values if v == 2),
            "3_to_5": sum(1 for v in reuse_values if 3 <= v <= 5),
            "6_to_10": sum(1 for v in reuse_values if 6 <= v <= 10),
            "11_to_20": sum(1 for v in reuse_values if 11 <= v <= 20),
            "21_to_30": sum(1 for v in reuse_values if 21 <= v <= 30),
            "over_30": sum(1 for v in reuse_values if v > 30),
        }

        # Format tuples for standard output (label, path_a, path_b)
        standard_trial_tuples = [(r["label"], r["path_a"], r["path_b"]) for r in all_trials_records]

        report: dict[str, Any] = {
            "version": "v2",
            "manifest_file": str(self.manifest_path.name),
            "random_seed": seed if seed is not None else self.seed,
            "min_duration_sec": self.min_duration,
            "total_raw_utterances": self.total_raw_utterances,
            "total_filtered_utterances": self.total_filtered_utterances,
            "total_speakers_in_manifest": len(self.raw_speakers),
            "dropped_unknown_gender_speakers": self.dropped_unknown_gender_speakers,
            "total_speakers_post_filtering": len(self.spk_data),
            "num_excluded_single_chapter_speakers": len(self.single_chapter_speakers),
            "fraction_excluded_speakers": round(len(self.single_chapter_speakers) / len(self.spk_data), 4),
            "single_chapter_policy": "Excluded from targets and impostors to maintain duration parity and symmetric speaker representation",
            "excluded_speakers": self.single_chapter_speakers,
            "num_qualifying_speakers": len(self.qualifying_speakers),
            "target_trials_count": len(target_trials),
            "impostor_trials_count": len(impostor_trials),
            "total_trials_count": len(all_trials_records),
            "target_to_impostor_ratio": round(len(target_trials) / max(1, len(impostor_trials)), 4),
            "gender_counts_matched": {
                "target_genders": dict(target_genders),
                "impostor_genders": dict(target_genders),
            },
            "duration_statistics": {
                "target_mean_sec": round(mean_dur_target, 3),
                "target_std_sec": round(std_dur_target, 3),
                "impostor_mean_sec": round(mean_dur_impostor, 3),
                "impostor_std_sec": round(std_dur_impostor, 3),
                "duration_relative_difference_percent": round(dur_rel_diff_pct, 2),
                "duration_parity_gate_passed": dur_rel_diff < 0.03,
            },
            "per_speaker_target_counts": per_spk_target_counts,
            "per_speaker_impostor_counts": dict(sorted(per_spk_impostor_counts.items())),
            "utterance_reuse_statistics": {
                "unique_utterances_in_corpus": self.total_filtered_utterances,
                "unique_utterances_used": len(utterance_counter),
                "utterance_utilization_rate": round(len(utterance_counter) / self.total_filtered_utterances, 4),
                "mean_reuse_count": round(mean_reuse, 2),
                "median_reuse_count": round(median_reuse, 2),
                "max_reuse_count": max_reuse,
                "percentiles": pct_stats,
                "reuse_distribution_histogram": reuse_dist,
            },
            "trials_metadata_records": all_trials_records,
        }

        return standard_trial_tuples, report


def save_trials_and_report(
    trials: list[tuple[int, str, str]],
    report: dict[str, Any],
    out_trials_path: Path,
    out_report_path: Path,
    out_meta_path: Path,
) -> None:
    """Save trial pairs, companion metadata CSV, and audit report with SHA-256."""
    out_trials_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write standard trial pairs (label path_a path_b)
    with open(out_trials_path, "w", encoding="utf-8") as f:
        for label, p1, p2 in trials:
            f.write(f"{label} {p1} {p2}\n")

    trials_sha256 = compute_sha256(out_trials_path)
    report["trial_file_sha256"] = trials_sha256
    report["trial_file_name"] = out_trials_path.name

    # 2. Write trial metadata CSV (spk_a, spk_b, gender, durations, utts)
    meta_records = report.pop("trials_metadata_records", [])
    if meta_records:
        df_meta = pd.DataFrame(meta_records)
        df_meta.to_csv(out_meta_path, index=False)
        report["trial_meta_sha256"] = compute_sha256(out_meta_path)
        report["trial_meta_name"] = out_meta_path.name

    # 3. Write trials audit report
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved {len(trials)} trials to {out_trials_path} (SHA-256: {trials_sha256[:16]}...)")
    print(f"Saved trial metadata to {out_meta_path}")
    print(f"Saved verification audit report to {out_report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LibriSpeech verification trial lists (v2).")
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
        default=100,
        help="Number of target trials per qualifying speaker (default: 100).",
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
        seed=args.seed,
    )

    out_trials_path = out_dir / f"trials_{split_name}.txt"
    out_report_path = out_dir / f"trials_report_{split_name}.json"
    out_meta_path = out_dir / f"trials_{split_name}_meta.csv"

    save_trials_and_report(trials, report, out_trials_path, out_report_path, out_meta_path)


if __name__ == "__main__":
    main()
