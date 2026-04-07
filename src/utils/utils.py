"""
Shared constants and helpers used by multiple pipeline modules.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Fast JSON — orjson with stdlib fallback (3-10× faster JSONL parsing)
# ---------------------------------------------------------------------------

try:
    import orjson as _orjson

    loads = _orjson.loads

    def dumps(obj: object, *, ensure_ascii: bool = False, indent: int = 0) -> str:
        """orjson.dumps → bytes; decode to str for file.write() compat."""
        opts = _orjson.OPT_NON_STR_KEYS
        if indent:
            opts |= _orjson.OPT_INDENT_2
        return _orjson.dumps(obj, option=opts).decode("utf-8")

    def dumps_line(obj: object) -> str:
        """Single JSONL line — no indent, trailing newline."""
        return _orjson.dumps(obj, option=_orjson.OPT_NON_STR_KEYS).decode("utf-8") + "\n"

    _JSON_BACKEND = "orjson"

except ImportError:
    import json as _json

    loads = _json.loads

    def dumps(obj: object, *, ensure_ascii: bool = False, indent: int = 0) -> str:
        return _json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent or None)

    def dumps_line(obj: object) -> str:
        return _json.dumps(obj, ensure_ascii=False) + "\n"

    _JSON_BACKEND = "json"


# ---------------------------------------------------------------------------
# IPA suprasegmentals — not phonemes, only length / stress markers.
#
# Used to split TPS (all IPA tokens / sec) into:
#   PPS  = segments / sec          (excludes suprasegmentals)
#   SPS  = suprasegmentals / sec   (length marks only)
#
# Currently only ː is present in the corpus vocab.
# Extend this frozenset if future data introduces ˑ ˈ ˌ etc.
# ---------------------------------------------------------------------------
SUPRASEGMENTALS: frozenset[str] = frozenset({"ː"})


# ---------------------------------------------------------------------------
# Warning suppression — single source of truth for all known non-actionable
# third-party warnings.  Call once in the main process (pipeline.py) **and**
# once per multiprocessing worker (warning filters are not inherited).
# ---------------------------------------------------------------------------

def configure_warnings() -> None:
    """Suppress known non-actionable third-party warnings."""
    # -- torchaudio 2.8→2.9 migration notices (torchcodec transition) --------
    warnings.filterwarnings(
        "ignore",
        message=r"In 2\.9, this function's implementation will be changed to use torchaudio\.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torio\.io\._streaming_media_(encoder|decoder)\.StreamingMedia",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", module="torchaudio._backend")
    warnings.filterwarnings("ignore", module="torchaudio._backend.ffmpeg")

    # -- WavLM attention mask mismatch (bool key_padding + float attn_mask) --
    warnings.filterwarnings(
        "ignore",
        message="Support for mismatched key_padding_mask and attn_mask",
    )

    # -- torch.compile internals (dynamo / inductor / sympy) -----------------
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logging.getLogger("torch._inductor").setLevel(logging.ERROR)
    logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=r".*pow_by_natural.*")
    warnings.filterwarnings("ignore", message=r".*CUDAGraph supports dynamic shapes.*")


def nearest_rank_pctl(
    vals: list[float],
    p: float,
    *,
    decimals: int = 6,
    presorted: bool = False,
) -> float:
    """Nearest-rank percentile.

    Parameters
    ----------
    vals : list[float]
        Values.  Will be sorted internally unless *presorted* is True.
    p : float
        Percentile in [0, 100].
    decimals : int
        Decimal places for rounding (default 6).
    presorted : bool
        If True, skip the internal sort (caller guarantees ascending order).

    Returns
    -------
    float
        The value at the nearest rank, rounded to *decimals* places.
    """
    if not vals:
        return 0.0
    sv = vals if presorted else sorted(vals)
    n = len(sv)
    idx = min(round((n - 1) * p / 100), n - 1)
    return round(sv[idx], decimals)


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Shared audio loading — used by audio EDA and model-selection EDA
# ---------------------------------------------------------------------------

def load_audio_mono(path: str) -> tuple[torch.Tensor, int, int]:
    """Load audio file → mono float32 tensor with DC offset removed.

    Parameters
    ----------
    path : str
        Absolute path to an audio file (FLAC, WAV, etc.).

    Returns
    -------
    (wav, sr, num_channels) : tuple[Tensor, int, int]
        *wav* is a 1-D float32 CPU tensor (channel-averaged mono, DC-removed).
        *sr* is the native sample rate.
        *num_channels* is the original channel count before mono conversion.

    Raises
    ------
    RuntimeError / FileNotFoundError
        On any soundfile/FileNotFoundError load failure — callers should catch and classify.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    num_channels = data.shape[1]
    if num_channels > 1:
        data = data.mean(axis=1)     # mono — channel average
    else:
        data = data[:, 0]
    wav = torch.from_numpy(data)
    wav = wav - wav.mean()           # DC offset removal
    return wav, sr, num_channels


def init_torch_worker() -> None:
    """Restrict PyTorch to one thread — call once per worker process.

    Prevents N-workers × M-threads CPU contention in multiprocessing pools.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def resolve_audio_path(
    audio_path_field: str,
    dataset_key: int,
    audio_dirs: dict[int, str],
) -> str:
    """Resolve a relative ``audio_path`` from a manifest row to an absolute path.

    Parameters
    ----------
    audio_path_field : str
        The ``audio_path`` value stored in the JSONL row (e.g. ``audio/U_xxx.flac``).
    dataset_key : int
        The ``dataset`` field from the same row.
    audio_dirs : dict[int, str]
        Mapping ``dataset_key → absolute directory`` (from ``cfg["paths"]["audio_dirs"]``).

    Returns
    -------
    str
        Absolute path to the audio file on disk.
    """
    fname = Path(audio_path_field).name
    return str(Path(audio_dirs[dataset_key]) / fname)


def sampler_weights_from_hours(
    age_hours: dict[str, float],
) -> dict[str, float]:
    """Compute inverse-proportional, mean-normalised sampler weights by age bucket.

    Parameters
    ----------
    age_hours : dict[str, float]
        Bucket label → total hours.

    Returns
    -------
    dict[str, float]
        Bucket label → weight (mean weight ≈ 1.0).  Sorted by key.
        Empty dict if input is empty or all-zero.
    """
    total = sum(age_hours.values())
    n = len(age_hours)
    if total <= 0 or n == 0:
        return {}
    raw: dict[str, float] = {}
    for bucket, hrs in age_hours.items():
        share = hrs / total
        raw[bucket] = (1.0 / share) if share > 0 else 0.0

    # Cap "unknown" at max of the real age buckets so noisy metadata
    # never dominates the sampler.  The data is still included — just
    # not amplified beyond the rarest real cohort.
    real = {b: w for b, w in raw.items() if b != "unknown"}
    if "unknown" in raw and real:
        cap = max(real.values())
        raw["unknown"] = min(raw["unknown"], cap)

    w_mean = sum(raw.values()) / n
    if w_mean <= 0:
        return {}
    return {b: round(w / w_mean, 4) for b, w in sorted(raw.items())}
