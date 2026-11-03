"""Speaker-balanced mini-batch sampler for text-independent speaker verification.

Implements the batch formation protocol specified in:
    J. Alam, A. Fathan, M. S. Alam, "A Hybrid Neural Approach to Speaker Verification
    with an Improved Additive Angular Margin Loss", IJCNN 2025, Section IV-A:
    "mini-batch of size 64 containing speech segments from 64 randomly selected speakers"
    and docs/PROJECT_SPEC.md Table 3.

Constraints:
1. Every batch of size B contains speech segments from exactly B pairwise distinct speakers.
2. Across an entire training epoch, every speaker in the dataset is guaranteed to be visited.
3. Completely deterministic across epochs given a random seed, with epoch progression via set_epoch().
"""

import math
from typing import Any, Iterator, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Sampler


class SpeakerBalancedBatchSampler(Sampler[list[int]]):
    """Mini-batch sampler ensuring B distinct speakers per batch and full speaker coverage per epoch.

    For each batch of size B, samples B distinct speakers without replacement.
    Ensures all S speakers in the pool are visited before any speaker is re-visited
    in the unvisited queue, guaranteeing 100% speaker coverage per epoch.
    """

    def __init__(
        self,
        speaker_labels: Sequence[Any],
        batch_size: int = 64,
        batches_per_epoch: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """Initialize SpeakerBalancedBatchSampler.

        Args:
            speaker_labels: Sequence of speaker identifiers (strings or ints) for all dataset samples.
            batch_size: Number of distinct speakers per batch (default 64 per paper).
            batches_per_epoch: Number of batches to yield per epoch. If None, defaults to
                ceil(len(speaker_labels) / batch_size). Must be at least ceil(num_speakers / batch_size).
            seed: Base random seed for reproducible multi-epoch shuffling.
        """
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Build mapping from speaker ID to list of dataset indices
        self.spk2indices: dict[Any, list[int]] = {}
        for idx, spk in enumerate(speaker_labels):
            self.spk2indices.setdefault(spk, []).append(idx)

        self.unique_speakers: list[Any] = sorted(list(self.spk2indices.keys()), key=str)
        self.num_speakers = len(self.unique_speakers)

        if self.num_speakers < self.batch_size:
            raise ValueError(
                f"Cannot form batches of {self.batch_size} distinct speakers: dataset only "
                f"contains {self.num_speakers} unique speakers."
            )

        min_batches_needed = math.ceil(self.num_speakers / self.batch_size)
        if batches_per_epoch is None:
            # Default to full dataset coverage
            self.batches_per_epoch = max(min_batches_needed, math.ceil(len(speaker_labels) / self.batch_size))
        else:
            if batches_per_epoch < min_batches_needed:
                raise ValueError(
                    f"batches_per_epoch ({batches_per_epoch}) is less than minimum needed "
                    f"({min_batches_needed}) to visit all {self.num_speakers} speakers."
                )
            self.batches_per_epoch = batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Set epoch index for deterministic pseudo-random shuffling across epochs."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Return the number of batches generated per epoch."""
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of sample indices.

        Each yielded list has length batch_size and corresponds to batch_size
        mutually distinct speakers.
        """
        # Seeded local RNG for this epoch
        rng = np.random.default_rng(self.seed + self.epoch * 10007)

        # Shuffle unique speakers to begin the unvisited queue
        unvisited = list(self.unique_speakers)
        rng.shuffle(unvisited)

        for _ in range(self.batches_per_epoch):
            batch_speakers: list[Any] = []

            if len(unvisited) >= self.batch_size:
                # Take batch_size distinct speakers from the unvisited queue
                batch_speakers = unvisited[: self.batch_size]
                unvisited = unvisited[self.batch_size :]
            elif len(unvisited) > 0:
                # Take all remaining unvisited speakers
                k = len(unvisited)
                batch_speakers = list(unvisited)
                unvisited = []

                # Replenish queue with all speakers and pick (batch_size - k) distinct speakers
                # that are not already in batch_speakers
                eligible_speakers = [s for s in self.unique_speakers if s not in batch_speakers]
                additional = rng.choice(eligible_speakers, size=self.batch_size - k, replace=False).tolist()
                batch_speakers.extend(additional)

                # Reset unvisited queue with a fresh shuffle of all speakers
                unvisited = list(self.unique_speakers)
                rng.shuffle(unvisited)
            else:
                # Queue empty: sample batch_size distinct speakers from all speakers
                batch_speakers = rng.choice(self.unique_speakers, size=self.batch_size, replace=False).tolist()

            # For each distinct speaker in the batch, sample one utterance index
            batch_indices: list[int] = []
            for spk in batch_speakers:
                indices = self.spk2indices[spk]
                chosen_idx = int(rng.choice(indices))
                batch_indices.append(chosen_idx)

            yield batch_indices

