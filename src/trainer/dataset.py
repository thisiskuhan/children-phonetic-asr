"""
SFT dataset — lightweight manifest reader for length-grouped sampling.
======================================================================

Reads a JSONL manifest (``sft_train.jsonl`` or ``sft_val.jsonl``) into
memory.  Each item is a lightweight dict — **audio is NOT loaded here**.
Loading, resampling, mono-downmix, and RMS normalisation are handled by
the :class:`SFTCollator` in ``data_collator.py``.

This split keeps the dataset fast for ``LengthGroupedSampler`` (which
only needs ``input_lengths``) and avoids holding raw waveforms in RAM.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from utils import loads, resolve_audio_path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Required JSONL fields — assertion fires on first bad row
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: tuple[str, ...] = (
    "utterance_id",
    "audio_path",
    "audio_duration_sec",
    "age_bucket",
    "phonetic_text",
    "n_phonemes",
    "dataset",
)


# ---------------------------------------------------------------------------
# SFTDataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Map-style dataset backed by a JSONL manifest.

    Parameters
    ----------
    manifest_path : str | Path
        Absolute path to a split JSONL file (``sft_train.jsonl`` etc.).
    audio_dirs : dict[int, str]
        ``{dataset_key: abs_audio_dir}`` — from ``cfg["paths"]["audio_dirs"]``.
    split : str
        Human-readable split label used in log tags (``"train"`` / ``"val"``).
    ds_oversample : dict[int | str, int] | None
        Optional: oversample specific datasets by factor. E.g., ``{1: 5}``
        means DS1 samples appear 5× more often. Each oversampled copy
        gets different augmentation at runtime (speed perturb, SpecAug).
        Only applied for train split.

    Attributes
    ----------
    input_lengths : list[float]
        Duration in seconds per row, in manifest order.  Used by
        ``LengthGroupedSampler`` for length-grouped batching.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        audio_dirs: dict[int, str],
        *,
        split: str = "train",
        ds_oversample: dict[int | str, int] | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        assert manifest_path.is_file(), (
            f"FATAL: manifest not found: {manifest_path}"
        )

        self._audio_dirs = audio_dirs
        self._split = split

        # ------------------------------------------------------------------
        # Load manifest — one dict per row, resolve audio paths up front
        # ------------------------------------------------------------------
        rows: list[dict[str, Any]] = []
        durations: list[float] = []

        with open(manifest_path, encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                row = loads(raw_line)

                # Validate required fields
                for field in _REQUIRED_FIELDS:
                    assert field in row, (
                        f"FATAL: missing field '{field}' at "
                        f"{manifest_path.name}:{line_no}"
                    )

                # Resolve relative audio path → absolute
                row["audio_path"] = resolve_audio_path(
                    row["audio_path"],
                    row["dataset"],
                    self._audio_dirs,
                )

                rows.append(row)
                durations.append(float(row["audio_duration_sec"]))

        # ------------------------------------------------------------------
        # Apply oversampling for specific datasets (train split only)
        # ------------------------------------------------------------------
        if ds_oversample and split == "train":
            oversample_rows: list[dict[str, Any]] = []
            oversample_durations: list[float] = []

            for row, dur in zip(rows, durations):
                ds_key = row["dataset"]
                # Normalize key to string for comparison
                ds_key_str = str(ds_key)
                ds_key_int = int(ds_key) if isinstance(ds_key, (int, str)) and str(ds_key).isdigit() else None

                factor = 1
                if ds_key_str in ds_oversample:
                    factor = ds_oversample[ds_key_str]
                elif ds_key_int is not None and ds_key_int in ds_oversample:
                    factor = ds_oversample[ds_key_int]

                for _ in range(factor):
                    oversample_rows.append(row)
                    oversample_durations.append(dur)

            # Shuffle to mix oversampled data
            combined = list(zip(oversample_rows, oversample_durations))
            _rng = random.Random(1507)
            _rng.shuffle(combined)
            rows = [r for r, _ in combined]
            durations = [d for _, d in combined]

            # Log oversampling stats
            from collections import Counter
            ds_counts = Counter(str(r["dataset"]) for r in rows)
            total = len(rows)
            stats = ", ".join(f"DS{ds}={cnt} ({100*cnt/total:.1f}%)" for ds, cnt in sorted(ds_counts.items()))
            log.info("[SFT] Oversampling applied: %s", stats)

        self._rows = rows
        self.input_lengths: list[float] = durations

        log.info(
            "[SFT] %s dataset loaded — %s rows from %s",
            split, f"{len(self._rows):,}", manifest_path.name,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single manifest row (no audio — collator loads it).

        Returns
        -------
        dict
            Keys: ``utterance_id``, ``audio_path`` (absolute),
            ``audio_duration_sec``, ``age_bucket``, ``phonetic_text``,
            ``n_phonemes``, ``dataset``, ``child_id`` (if present).
        """
        return self._rows[idx]
