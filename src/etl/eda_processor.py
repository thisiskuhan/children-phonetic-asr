"""
EDA Processor — Text Sanitisation + Audio Control Calibration
==============================================================

Two phases:
    Phase 0 (``sanitize``):  Text EDA — NFC normalise, fix affricates/ASCII-r,
        drop bad rows, flag drills, health check.  Produces cleaned JSONL
        manifests in ``data/processed/``.

    Audio EDA (``audio_eda``):  Streaming multiprocessed analysis
        of raw audio.  Produces ``eda_controls.json`` in ``data/reports/``
        that drives every downstream training decision.

Design constraints (Audio EDA):
    • Stream — never load the full dataset into RAM.
    • Multiprocessing — ``mp.Pool`` over manifest rows.
    • Deterministic — fixed seed for spectral subset.
    • torchaudio only for audio I/O.
"""

from __future__ import annotations

import logging
import multiprocessing as mp_pool
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torchaudio
from tqdm import tqdm

from utils import (SUPRASEGMENTALS, nearest_rank_pctl, sampler_weights_from_hours,
                  loads, dumps, dumps_line, load_audio_mono, init_torch_worker)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  TEXT EDA — constants & helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Normalisation fix map — derived from full Unicode audit of both raw datasets
# (Feb 2026, 153,067 rows).  Do NOT add entries here without re-running the
# audit on new data first.
#
#   Confirmed anomalies in corpus:
#     ASCII r  (U+0072) :  2 occurrences in DS2  → ɹ (U+0279)
#     decomposed tʃ     :  1 occurrence  in DS2  → ʧ (U+02A7)
#     decomposed dʒ     :  0 occurrences         → ʤ (U+02A4)  [defensive]
#
#   Confirmed NOT anomalies (valid IPA — do not remap):
#     θ (U+03B8) 9,545  — voiceless dental fricative
#     x (U+0078)   154  — voiceless velar fricative
#     c (U+0063)   176  — voiceless palatal plosive
#     χ (U+03C7)    18  — voiceless uvular fricative
# ---------------------------------------------------------------------------
TEXT_FIXES: dict[str, str] = {
    "tʃ": "ʧ",   # decomposed affricate → ligature
    "dʒ": "ʤ",   # decomposed affricate → ligature (defensive; 0 found in corpus)
    "r":  "ɹ",   # ASCII r → IPA alveolar approximant
}

# Display constants — not configurable
_DRILL_SAMPLE_LIMIT = 5
_DRILL_PRINT_WIDTH  = 85


def _is_drill(text: str, min_words: int, dup_ratio: float) -> bool:
    """True if ≥min_words tokens and unique/total < dup_ratio (diadochokinetic drills)."""
    words = text.split()
    if len(words) < min_words:
        return False
    return len(set(words)) / len(words) < dup_ratio


def _health_check(path: Path, expected_rows: int, eda_cfg: dict) -> None:
    """Stream-verify every invariant on the output file."""
    max_dur   = eda_cfg["max_duration"]
    min_dur   = eda_cfg["min_duration"]
    max_runon = eda_cfg["max_runon_length"]
    tps_min   = eda_cfg["tps_min"]
    tps_max   = eda_cfg["tps_max"]
    min_phon  = eda_cfg.get("min_phonemes", 1)

    violations: dict[str, int] = {}
    n = 0

    log.info("")
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = loads(line)
            t = r["phonetic_text"]
            dur = r["audio_duration_sec"]
            tps = len(t.replace(" ", "")) / dur if dur > 0 else 0

            checks = {
                "ASCII 'r'":      "r" in t,
                "affricate tʃ":   "tʃ" in t,
                "affricate dʒ":   "dʒ" in t,
                "empty label":    not t,
                "over duration":  dur > max_dur,
                "mic click":      dur < min_dur,
                "run-on":         " " not in t and len(t) > max_runon,
                "short_phonemes": len(t.replace(" ", "")) < min_phon,
                "physics":        not (tps_min <= tps <= tps_max),
            }
            for name, bad in checks.items():
                if bad:
                    violations[name] = violations.get(name, 0) + 1
            n += 1

    log.info(f"[HEALTH] {path.name}")
    rc = "✓" if n == expected_rows else "✗"
    log.info(f"[HEALTH]   {rc}  row count     {n:,} / {expected_rows:,}")
    if violations:
        for name, cnt in violations.items():
            log.warning(f"[HEALTH]   ✗  {name:<16s}  {cnt:,} rows")
    else:
        for name in ("ASCII 'r'", "affricate tʃ", "affricate dʒ", "empty label",
                     "over duration", "mic click", "run-on", "short_phonemes", "physics"):
            log.info(f"[HEALTH]   ✓  {name}")


# ---------------------------------------------------------------------------
# Text EDA core
# ---------------------------------------------------------------------------

def _sanity_fix(cfg: dict) -> dict[int, dict]:
    """
    Stream-process each JSONL dataset using config values.

    Normalisation (applied in order):
      1. NFC-normalise text
      2. Apply TEXT_FIXES map (affricates + ASCII r typo)
      3. Squash whitespace

    Drop conditions (all thresholds from config):
      - Run-on label  : no space and len > max_runon_length
      - Empty label
      - Duration > max_duration
      - Mic click     : duration < min_duration
      - Physics       : tokens/sec outside [tps_min, tps_max]

    Adds is_drill (bool) to each kept row.
    Saves to <processed>/<key>_transcript.jsonl.
    """
    # Derive dataset set from cfg["datasets"] — the single canonical list
    # (same source as audio_eda, so Phase 0 and Audio EDA always cover identical datasets)
    datasets    = {k: cfg["paths"]["datasets"][k] for k in cfg["datasets"]}
    eda_cfg     = cfg["eda"]
    out         = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)

    max_dur     = eda_cfg["max_duration"]
    min_dur     = eda_cfg["min_duration"]
    max_runon   = eda_cfg["max_runon_length"]
    tps_min     = eda_cfg["tps_min"]
    tps_max     = eda_cfg["tps_max"]
    min_phon    = eda_cfg.get("min_phonemes", 1)  # drop if n_phonemes < this
    drill_words = eda_cfg["drill_min_words"]
    drill_ratio = eda_cfg["drill_duplicate_ratio"]

    all_stats: dict[int, dict] = {}
    ds_char_counts: dict[int, Counter] = {}   # per-dataset char inventory for policy audit
    all_valid_speaker_ids: set[str] = set()   # union across all datasets

    for key, path in datasets.items():
        dropped = {"run_on": 0, "empty": 0, "duration": 0, "mic_click": 0, "short_phonemes": 0, "physics": 0}
        n_raw = n_clean = n_drill = 0
        total_duration_sec: float = 0.0
        long_buckets: Counter = Counter()     # 1-sec buckets for files ≥15s (raw, pre-drop)
        age_hours: dict[str, float] = {}
        tps_vals: list[float] = []
        pps_vals: list[float] = []
        sps_vals: list[float] = []
        drill_samples: list[str] = []
        valid_speaker_ids: set[str] = set()    # non-unknown child_ids in clean rows
        ds_chars: Counter = Counter()          # char inventory for this dataset

        out_path = out / f"{key}_transcript.jsonl"

        with open(path, encoding="utf-8") as fin, \
             out_path.open("w", encoding="utf-8") as fout:

            for line in fin:
                r = loads(line)
                n_raw += 1
                raw_dur = r["audio_duration_sec"]
                if raw_dur >= 15:
                    long_buckets[int(raw_dur)] += 1
                t: str = r["phonetic_text"]

                # --- Normalise ---
                t = unicodedata.normalize("NFC", t)
                for src, tgt in TEXT_FIXES.items():
                    t = t.replace(src, tgt)
                t = " ".join(t.split())

                # --- Drop conditions ---
                if " " not in t and len(t) > max_runon:
                    dropped["run_on"] += 1
                    continue
                if not t:
                    dropped["empty"] += 1
                    continue

                dur = r["audio_duration_sec"]
                if dur > max_dur:
                    dropped["duration"] += 1
                    continue
                if dur < min_dur:
                    dropped["mic_click"] += 1
                    continue

                n_tokens = len(t.replace(" ", ""))
                if n_tokens < min_phon:
                    dropped["short_phonemes"] += 1
                    continue
                tps = n_tokens / dur
                if not (tps_min <= tps <= tps_max):
                    dropped["physics"] += 1
                    continue

                # --- Keep ---
                r["phonetic_text"] = t
                r["n_phonemes"] = n_tokens
                drill = _is_drill(t, drill_words, drill_ratio)
                r["is_drill"] = drill
                fout.write(dumps_line(r))
                n_clean += 1
                total_duration_sec += dur
                cid = r.get("child_id")
                if cid and cid.strip() and cid.strip().lower() != "unknown":
                    valid_speaker_ids.add(cid)
                # Accumulate age hours + rate stats for post-cleaning distribution log
                bucket = r.get("age_bucket", "unknown")
                age_hours[bucket] = age_hours.get(bucket, 0) + dur / 3600
                tps_vals.append(tps)
                n_supra = sum(1 for ch in t if ch in SUPRASEGMENTALS)
                n_seg   = n_tokens - n_supra
                pps_vals.append(n_seg / dur)
                sps_vals.append(n_supra / dur)
                for ch in t:
                    if ch != " ":
                        ds_chars[ch] += 1
                if drill:
                    n_drill += 1
                    if len(drill_samples) < _DRILL_SAMPLE_LIMIT:
                        drill_samples.append(t)

        stats = {
            "n_raw":   n_raw,
            "n_clean": n_clean,
            "dropped": dropped,
            "n_drill": n_drill,
            "total_hours": round(total_duration_sec / 3600, 3),
            "n_valid_speakers": len(valid_speaker_ids),
            "long_buckets": dict(long_buckets),
            "out":     str(out_path),
        }
        all_stats[key] = stats
        ds_char_counts[key] = ds_chars
        all_valid_speaker_ids.update(valid_speaker_ids)

        total_dropped = sum(dropped.values())
        pct = 100 * n_clean / n_raw if n_raw else 0

        log.info("")
        log.info(f"--- SANITY DS{key} START ---")
        log.info(f"[SANITY] DS{key}  {path}")
        log.info(f"[SANITY] raw {n_raw:,}  →  clean {n_clean:,}  ({pct:.1f}% kept)")
        log.info(f"[DROP]   run-on    {dropped['run_on']:>6,}")
        log.info(f"[DROP]   empty     {dropped['empty']:>6,}")
        log.info(f"[DROP]   duration  {dropped['duration']:>6,}   (>{max_dur}s)")
        log.info(f"[DROP]   mic_click {dropped['mic_click']:>6,}   (<{min_dur}s)")
        log.info(f"[DROP]   physics   {dropped['physics']:>6,}   (tps <{tps_min} or >{tps_max})")
        log.info(f"[DROP]   TOTAL     {total_dropped:>6,}")
        log.info(f"[DRILL]  flagged {n_drill:>5,}")
        for i, s in enumerate(drill_samples, 1):
            log.info(f"[DRILL]  {i}. {s[:_DRILL_PRINT_WIDTH]}")
        total_hours = total_duration_sec / 3600
        log.info(f"[SANITY] total duration  {total_hours:>10.2f} hrs  (clean rows)")
        log.info(f"[SANITY] valid speakers  {len(valid_speaker_ids):>10,}")
        # ---- Age-bucket hours (post-cleaning) ----
        if age_hours:
            log.info("[SANITY] hours by age (post-clean):")
            for bucket in sorted(age_hours):
                hrs = age_hours[bucket]
                pct = 100 * hrs / total_hours if total_hours > 0 else 0
                log.info(f"[SANITY]   {bucket:<8s}  {hrs:>8.2f} hrs  ({pct:>5.1f}%)")
        # ---- TPS / PPS / SPS distributions (post-cleaning) ----
        if tps_vals:
            tps_s = sorted(tps_vals)
            pps_s = sorted(pps_vals)
            sps_s = sorted(sps_vals)
            log.info(f"[SANITY] TPS  p01={nearest_rank_pctl(tps_s, 1, presorted=True, decimals=3):.1f}  "
                     f"p50={nearest_rank_pctl(tps_s, 50, presorted=True, decimals=3):.1f}  "
                     f"p99={nearest_rank_pctl(tps_s, 99, presorted=True, decimals=3):.1f}  "
                     f"min={tps_s[0]:.1f}  max={tps_s[-1]:.1f}")
            log.info(f"[SANITY] PPS  p50={nearest_rank_pctl(pps_s, 50, presorted=True, decimals=3):.1f}   "
                     f"SPS  p50={nearest_rank_pctl(sps_s, 50, presorted=True, decimals=3):.2f}  "
                     f"(segments vs suprasegmentals — advisory)")
        # ---- Long-file distribution (≥15s, raw, pre-drop) ----
        if long_buckets:
            log.info(f"[SANITY] files ≥15s (raw):")
            for sec in sorted(long_buckets):
                log.info(f"[SANITY]   {sec}–{sec+1}s  {long_buckets[sec]:>5,}")
        _health_check(out_path, n_clean, eda_cfg)
        log.info(f"--- SANITY DS{key} END ---")

    # ---- Overall summary across all datasets ----
    grand_total_hrs = sum(s["total_hours"] for s in all_stats.values())
    grand_clean     = sum(s["n_clean"] for s in all_stats.values())
    grand_raw       = sum(s["n_raw"] for s in all_stats.values())
    # ---- Overall long-file distribution (≥15s, raw) ----
    overall_long: Counter = Counter()
    for s in all_stats.values():
        overall_long.update(s.get("long_buckets", {}))
    if overall_long:
        log.info("[SANITY] files ≥15s overall (raw):")
        for sec in sorted(overall_long):
            log.info(f"[SANITY]   {sec}–{sec+1}s  {overall_long[sec]:>5,}")
    log.info("")
    log.info("--- SANITY SUMMARY ---")
    for k, s in all_stats.items():
        log.info(f"[SANITY] DS{k}  {s['n_clean']:>8,} rows  {s['total_hours']:>8.2f} hrs")
    log.info(f"[SANITY] ALL  {grand_clean:>8,} rows  {grand_total_hrs:>8.2f} hrs  (from {grand_raw:,} raw)")
    log.info(f"[SANITY] valid speakers  {len(all_valid_speaker_ids):>10,}")
    log.info("--- SANITY SUMMARY END ---")

    # ------------------------------------------------------------------
    # LABEL POLICY AUDIT — cross-dataset transcription philosophy check
    # ------------------------------------------------------------------
    # Evidence-based: compares post-normalisation character inventories.
    # Flags exclusive characters, combining diacritics, modifier letters
    # — any of which signal narrow-vs-broad philosophy mismatch.
    # ------------------------------------------------------------------
    if len(ds_char_counts) >= 2:
        log.info("")
        log.info("--- LABEL POLICY AUDIT ---")

        inventories = {k: set(c.keys()) for k, c in ds_char_counts.items()}
        shared  = set.intersection(*inventories.values())
        union   = set.union(*inventories.values())

        log.info(f"[POLICY] shared inventory      {len(shared)} chars")
        log.info(f"[POLICY] union  inventory      {len(union)} chars")

        # ---- Exclusive characters per dataset ----
        has_exclusive = False
        for k in sorted(inventories):
            exclusive = sorted(inventories[k] - shared, key=ord)
            if exclusive:
                has_exclusive = True
                for ch in exclusive:
                    cp   = f"U+{ord(ch):04X}"
                    name = unicodedata.name(ch, "?")
                    cnt  = ds_char_counts[k][ch]
                    log.info(f"[POLICY]   DS{k} only  {cp}  '{ch}'  "
                             f"count={cnt:>6,}  ({name})")
        if not has_exclusive:
            log.info("[POLICY]   (no exclusive chars — identical inventories)")

        # ---- Narrow-transcription indicators per dataset ----
        # combining(ch) > 0  → Unicode Mn/Mc (diacritics: ̃ ̚ ̥ ̪ etc.)
        # category == 'Lm'   → modifier letters  (ʰ ʷ ʲ ˠ ˤ etc.)
        # SUPRASEGMENTALS excluded — they are expected broad-level markers.
        ds_densities: list[tuple[int, int, int, float]] = []
        for k in sorted(ds_char_counts):
            cc = ds_char_counts[k]
            total_tokens = sum(cc.values())
            n_combining  = sum(v for ch, v in cc.items()
                               if unicodedata.combining(ch) > 0)
            n_modifier   = sum(v for ch, v in cc.items()
                               if unicodedata.category(ch) == "Lm"
                               and ch not in SUPRASEGMENTALS)
            density = ((n_combining + n_modifier) / total_tokens * 1000
                       if total_tokens else 0)
            log.info(f"[POLICY] DS{k}  tokens={total_tokens:>10,}  "
                     f"combining={n_combining:>6,}  "
                     f"modifier={n_modifier:>6,}  "
                     f"narrow_density={density:.2f}/1k")
            ds_densities.append((k, n_combining, n_modifier, density))

        # ---- Verdict ----
        densities = [d for _, _, _, d in ds_densities]
        spread = max(densities) - min(densities) if densities else 0
        if has_exclusive or spread > 0.5:
            log.warning(f"[POLICY] verdict: REVIEW NEEDED  "
                        f"(exclusive_chars={has_exclusive}  "
                        f"density_spread={spread:.2f}/1k)")
        else:
            log.info(f"[POLICY] verdict: CONSISTENT  "
                     f"(density_spread={spread:.2f}/1k, "
                     f"no exclusive chars)")

        log.info("--- LABEL POLICY AUDIT END ---")

    return all_stats


# ---------------------------------------------------------------------------
#  AUDIO EDA PHASE 1 — Control Calibration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-file result dataclass — returned by each worker
# ---------------------------------------------------------------------------

@dataclass
class _FileMetrics:
    utterance_id: str
    duration_sec: float
    child_id: str
    age_bucket: str
    sample_rate: int
    num_channels: int
    rms: float
    clipped_ratio: float
    lead_silence_ratio: float
    trail_silence_ratio: float
    spectral_centroid: float | None = None
    duration_mismatch: bool = False  # |manifest_dur - actual_dur| > _DUR_TOLERANCE_S


# ---------------------------------------------------------------------------
# Accumulator — collects per-file results into corpus-level summaries
# ---------------------------------------------------------------------------

@dataclass
class _Accumulator:
    rms_vals: list[float] = field(default_factory=list)
    clipped_ratios: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    speaker_hours: dict[str, float] = field(default_factory=dict)
    age_hours: dict[str, float] = field(default_factory=dict)
    sample_rates: list[int] = field(default_factory=list)
    channel_counts: list[int] = field(default_factory=list)
    lead_silence_ratios: list[float] = field(default_factory=list)
    trail_silence_ratios: list[float] = field(default_factory=list)
    spectral_centroids: list[float] = field(default_factory=list)
    n_duration_mismatches: int = 0
    duration_mismatch_ids: list[str] = field(default_factory=list)  # utterance_ids with duration mismatch

    def ingest(self, m: _FileMetrics) -> None:
        if m.duration_mismatch:
            self.n_duration_mismatches += 1
            self.duration_mismatch_ids.append(m.utterance_id)
        self.rms_vals.append(m.rms)
        self.clipped_ratios.append(m.clipped_ratio)
        self.durations.append(m.duration_sec)
        self.sample_rates.append(m.sample_rate)
        self.channel_counts.append(m.num_channels)
        self.lead_silence_ratios.append(m.lead_silence_ratio)
        self.trail_silence_ratios.append(m.trail_silence_ratio)

        hrs = m.duration_sec / 3600
        self.speaker_hours[m.child_id] = self.speaker_hours.get(m.child_id, 0) + hrs
        self.age_hours[m.age_bucket] = self.age_hours.get(m.age_bucket, 0) + hrs

        if m.spectral_centroid is not None:
            self.spectral_centroids.append(m.spectral_centroid)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _pctl(vals: list[float], p: float) -> float:
    """Thin wrapper — delegates to the shared utility."""
    return nearest_rank_pctl(vals, p, decimals=6)


# ---------------------------------------------------------------------------
# Per-file worker function  (runs in separate process)
# ---------------------------------------------------------------------------

# Module-level config — set once via initializer in each worker process
_worker_cfg: dict[str, Any] = {}


def _worker_init(cfg: dict[str, Any]) -> None:
    """Called once per worker process — stash config in module globals."""
    from utils import configure_warnings
    global _worker_cfg
    _worker_cfg = cfg

    # Warning filters are NOT inherited by worker processes — apply centrally.
    configure_warnings()

    # ---- Keep PyTorch single-threaded inside each worker ----
    init_torch_worker()

    # Pre-create Hann window once per worker — avoids re-allocation per file.
    # Placed on CPU explicitly; moved to wav.device inside _analyse_file.
    _worker_cfg["_hann_window"] = torch.hann_window(1024)

    # Pre-create freqs tensor for spectral centroid (n_fft=1024 → 513 bins).
    # Reused across all files — avoids reallocating per file.
    expected_sr = cfg["format"]["expected_sr"]
    _worker_cfg["_freqs"] = torch.linspace(0, expected_sr / 2, 513)


# Error sentinel tags — returned instead of None so the main loop can classify
_ERR_LOAD  = "__err_load__"
_ERR_EMPTY = "__err_empty__"
_ERR_NAN   = "__err_nan__"

# Manifest duration sanity tolerance — flag if |manifest_sec - actual_sec| exceeds this
_DUR_TOLERANCE_S = 0.1


def _analyse_file(args: tuple[str, str, str, float, str, bool]) -> _FileMetrics | str:
    """
    Analyse a single audio file.

    Parameters (packed as tuple for Pool.imap):
        audio_path       — absolute path to .flac
        utterance_id     — row ID
        child_id         — speaker
        duration_sec     — from manifest (for age/speaker accounting)
        age_bucket       — age group
        compute_centroid — whether to compute spectral centroid

    Returns _FileMetrics on success, or an error sentinel string.
    """
    audio_path, utterance_id, child_id, duration_sec, age_bucket, compute_centroid = args
    cfg = _worker_cfg

    try:
        wav, sr, num_channels = load_audio_mono(audio_path)
    except Exception:
        return _ERR_LOAD

    n_samples = wav.numel()
    if n_samples == 0:
        return _ERR_EMPTY

    # ---- RMS ----
    rms = torch.sqrt(torch.mean(wav ** 2)).item()
    if rms != rms:  # NaN guard
        return _ERR_NAN

    # ---- Clipping ----
    clip_thresh = cfg["clipping"]["sample_threshold"]
    n_clipped = (wav.abs() >= clip_thresh).sum().item()
    clipped_ratio = n_clipped / n_samples

    # ---- Leading / trailing silence ----
    window_samples = int(cfg["silence"]["window_sec"] * sr)
    window_samples = min(window_samples, n_samples)

    lead_rms = torch.sqrt(torch.mean(wav[:window_samples] ** 2)).item()
    trail_rms = torch.sqrt(torch.mean(wav[-window_samples:] ** 2)).item()

    lead_ratio = lead_rms / rms if rms > 1e-10 else 0.0
    trail_ratio = trail_rms / rms if rms > 1e-10 else 0.0

    # ---- Spectral centroid via STFT (subset only) ----
    #   Short-time magnitude → per-frame centroid → average over time.
    centroid_val: float | None = None
    if compute_centroid:
        n_fft = 1024
        hop = 512
        window = cfg["_hann_window"].to(wav.device)  # cached in worker init
        stft = torch.stft(wav, n_fft=n_fft, hop_length=hop,
                          window=window, return_complex=True)
        mag = stft.abs()                          # (freq_bins, time_frames)
        freqs = cfg["_freqs"].to(wav.device)    # cached in worker init
        # Per-frame centroid: sum(freq * mag) / sum(mag)
        frame_mags = mag.sum(dim=0)                    # (time_frames,)
        valid = frame_mags > 1e-10
        if valid.any():
            # (freq_bins,1) * (freq_bins, time_frames) → sum over freq → (time_frames,)
            weighted = (freqs.unsqueeze(1) * mag).sum(dim=0)
            centroids = weighted[valid] / frame_mags[valid]
            centroid_val = centroids.mean().item()

    actual_dur   = n_samples / sr
    dur_mismatch = abs(duration_sec - actual_dur) > _DUR_TOLERANCE_S

    return _FileMetrics(
        utterance_id=utterance_id,
        duration_sec=actual_dur,   # use actual decoded duration, not manifest
        child_id=child_id,
        age_bucket=age_bucket,
        sample_rate=sr,
        num_channels=num_channels,
        rms=rms,
        clipped_ratio=clipped_ratio,
        lead_silence_ratio=lead_ratio,
        trail_silence_ratio=trail_ratio,
        spectral_centroid=centroid_val,
        duration_mismatch=dur_mismatch,
    )


# ---------------------------------------------------------------------------
# Manifest reader — yields worker args without loading all rows into memory
# ---------------------------------------------------------------------------

def _iter_manifest(
    manifest_paths: list[str],
    audio_dirs: list[str],
    centroid_indices: set[int],
) -> Iterator[tuple[str, str, str, float, str, bool]]:
    """
    Stream processed JSONL manifests → yield one worker tuple per row.

    manifest_paths and audio_dirs must be parallel lists (same order).
    Never materialises the full list — feeds Pool.imap directly.
    """
    idx = 0

    for manifest, audio_dir_str in zip(manifest_paths, audio_dirs):
        audio_dir = Path(audio_dir_str)

        with open(manifest, encoding="utf-8") as f:
            for line in f:
                row = loads(line)
                fname = Path(row["audio_path"]).name
                fpath = str(audio_dir / fname)
                yield (
                    fpath,
                    row["utterance_id"],
                    row.get("child_id", "unknown"),
                    row["audio_duration_sec"],
                    row["age_bucket"],
                    idx in centroid_indices,
                )
                idx += 1


# ---------------------------------------------------------------------------
# Audio EDA core
# ---------------------------------------------------------------------------

def _run_audio_eda(
    manifest_paths: list[str],
    audio_dirs: list[str],
    output_dir: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute Audio EDA Phase 1 and write eda_controls.json.

    Parameters:
        manifest_paths — processed JSONL paths (1_transcript.jsonl, …)
        audio_dirs     — parallel list of audio directories (same order)
        output_dir     — where to write eda_controls.json
        cfg            — audio_eda section of the project config
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Single pre-pass: count rows + extract utterance_ids (for error attribution) ----
    # Previously two separate passes; merging saves ~151 k redundant json.loads calls.
    utterance_ids: list[str] = []
    for mp_ in manifest_paths:
        with open(mp_, encoding="utf-8") as f:
            for line in f:
                utterance_ids.append(loads(line)["utterance_id"])
    n_total = len(utterance_ids)

    # ---- Deterministic centroid subset ----
    rng = random.Random(cfg["spectral"]["seed"])
    n_centroid = max(1, int(n_total * cfg["spectral"]["subset_fraction"]))
    centroid_indices = set(rng.sample(range(n_total), min(n_centroid, n_total)))

    log.info("")
    log.info("--- AUDIO EDA START ---")
    log.info(f"[EDA] manifests       {manifest_paths}")
    log.info(f"[EDA] total files     {n_total:,}")
    log.info(f"[EDA] centroid subset {len(centroid_indices):,}  (seed={cfg['spectral']['seed']})")
    log.info(f"[EDA] workers         {cfg['num_workers']}")

    # ---- Multiprocessed analysis (streaming — generator feeds imap) ----
    acc = _Accumulator()
    err_counts: Counter[str] = Counter()
    err_load_ids: list[str] = []  # utterance_ids for _ERR_LOAD

    with mp_pool.Pool(
        processes=cfg["num_workers"],
        initializer=_worker_init,
        initargs=(cfg,),
    ) as pool:
        gen = _iter_manifest(manifest_paths, audio_dirs, centroid_indices)
        with tqdm(pool.imap(_analyse_file, gen, chunksize=256),
                  total=n_total, desc="[EDA] analysing", unit="file",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            for idx, result in enumerate(pbar):
                if isinstance(result, str):      # error sentinel
                    err_counts[result] += 1
                    if result == _ERR_LOAD:
                        err_load_ids.append(utterance_ids[idx])
                else:
                    acc.ingest(result)

    n_ok = len(acc.rms_vals)
    n_errors = sum(err_counts.values())
    log.info(f"[EDA] analysed    {n_ok:>10,}")
    if n_errors:
        log.warning(f"[EDA] errors      {n_errors:>10,}")
        for tag, cnt in err_counts.most_common():
            label = tag.strip("_").replace("err_", "")
            log.warning(f"[EDA]   ✗  {label:<16s} {cnt:>8,}")
    if acc.n_duration_mismatches:
        log.warning(
            f"[EDA] dur_mismatch {acc.n_duration_mismatches:>9,}  "
            f"(|manifest-actual|>{_DUR_TOLERANCE_S}s — possible corrupt manifests)"
        )

    if n_ok == 0:
        raise RuntimeError(
            "No valid audio analysed — 0 files loaded successfully. "
            "Check manifest paths, audio directories, and torchaudio installation."
        )

    # ====================================================================
    # Compute corpus-level metrics
    # ====================================================================

    # ---- 1. RMS (clamp p01 floor to avoid infinite spread on near-silence) ----
    rms_p01 = _pctl(acc.rms_vals, 1)
    rms_p50 = _pctl(acc.rms_vals, 50)
    rms_p99 = _pctl(acc.rms_vals, 99)
    rms_min = round(min(acc.rms_vals), 6) if acc.rms_vals else 0
    rms_max = round(max(acc.rms_vals), 6) if acc.rms_vals else 0
    rms_p01_clamped = max(rms_p01, 1e-4)
    rms_spread = rms_p99 / rms_p01_clamped
    recommend_rms = rms_spread > cfg["rms"]["spread_ratio_threshold"]

    rms_block = {
        "p01": rms_p01, "p50": rms_p50, "p99": rms_p99,
        "min": rms_min, "max": rms_max,
        "p01_clamped": round(rms_p01_clamped, 6),
        "spread_ratio": round(rms_spread, 2),
        "recommend_normalization": recommend_rms,
    }

    # ---- 2. Clipping ----
    file_pct_thresh = cfg["clipping"]["file_percent_threshold"] / 100
    n_clipped_files = sum(1 for r in acc.clipped_ratios if r >= file_pct_thresh)
    pct_clipped = round(100 * n_clipped_files / n_ok, 2) if n_ok else 0
    recommend_drop_clipped = pct_clipped > cfg["clipping"]["corpus_percent_threshold"]

    clipping_block = {
        "sample_threshold": cfg["clipping"]["sample_threshold"],
        "file_percent_threshold": cfg["clipping"]["file_percent_threshold"],
        "n_files_heavily_clipped": n_clipped_files,
        "percent_files_above_threshold": pct_clipped,
        "recommend_drop_clipped": recommend_drop_clipped,
    }

    # ---- 3. Speaker dominance ----
    # Source of truth for total hours = sum of actual decoded durations,
    # NOT speaker_hours (which would miss files with missing speaker metadata).
    total_hrs = sum(acc.durations) / 3600
    age_total = sum(acc.age_hours.values())
    n_buckets = len(acc.age_hours)
    top_k = cfg["speaker"]["top_k"]
    top_speakers = sorted(acc.speaker_hours.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_hrs = sum(h for _, h in top_speakers)
    top_pct = round(100 * top_hrs / total_hrs, 2) if total_hrs > 0 else 0
    recommend_sampler = top_pct > cfg["speaker"]["dominance_threshold"]

    # Sampler weights by age — inversely proportional to duration share,
    # normalised so mean weight = 1.0.
    sampler_weights = sampler_weights_from_hours(acc.age_hours)

    speaker_block = {
        "n_speakers": len(acc.speaker_hours),
        "total_hours": round(total_hrs, 2),
        "top_k": top_k,
        "top_speakers": [
            {"child_id": cid, "hours": round(h, 3), "percent": round(100 * h / total_hrs, 2)}
            for cid, h in top_speakers
        ],
        "top5_speaker_hour_percent": top_pct,
        "recommend_weighted_sampler": recommend_sampler,
        "sampler_weights_by_age_bucket": sampler_weights,
    }

    # ---- 4. Age distribution ----
    age_block = {
        "hours_per_age_bucket": {k: round(v, 3) for k, v in sorted(acc.age_hours.items())},
        "percent_per_age_bucket": {
            k: round(100 * v / age_total, 2) for k, v in sorted(acc.age_hours.items())
        } if age_total > 0 else {},
    }

    # ---- 5. Duration profiling ----
    dur_p50 = _pctl(acc.durations, 50)
    dur_p90 = _pctl(acc.durations, 90)
    dur_p95 = _pctl(acc.durations, 95)
    dur_p96 = _pctl(acc.durations, 96)
    dur_p97 = _pctl(acc.durations, 97)
    dur_p98 = _pctl(acc.durations, 98)
    dur_p99 = _pctl(acc.durations, 99)
    dur_max = round(max(acc.durations), 3) if acc.durations else 0

    # Tail analysis — files above p99 (for research inspection)
    n_above_p99 = sum(1 for d in acc.durations if d > dur_p99)
    pct_above_p99 = round(100 * n_above_p99 / n_ok, 2) if n_ok else 0

    target_vram = cfg["duration"]["target_vram_seconds"]
    # Theoretical batch size — assumes no feature extraction overhead,
    # no fp16 scaling, no grad checkpointing.  Must be validated empirically.
    theoretical_batch = max(1, int(target_vram / dur_p95)) if dur_p95 > 0 else 1

    duration_block = {
        "p50": dur_p50, "p90": dur_p90, "p95": dur_p95,
        "p96": dur_p96, "p97": dur_p97, "p98": dur_p98, "p99": dur_p99,
        "max": dur_max,
        "n_above_p99": n_above_p99,
        "percent_above_p99": pct_above_p99,
        "p50_audio_seconds_per_utterance": dur_p50,
        "target_vram_seconds": target_vram,
        "theoretical_batch_size_at_p95": theoretical_batch,
        "n_duration_manifest_mismatches": acc.n_duration_mismatches,
    }

    # ---- 6. Silence ----
    silence_thresh = cfg["silence"]["ratio_threshold"]
    n_lead_silent = sum(1 for r in acc.lead_silence_ratios if r < silence_thresh)
    n_trail_silent = sum(1 for r in acc.trail_silence_ratios if r < silence_thresh)
    n_either = sum(
        1 for l, t in zip(acc.lead_silence_ratios, acc.trail_silence_ratios)
        if l < silence_thresh or t < silence_thresh
    )
    pct_silent = round(100 * n_either / n_ok, 2) if n_ok else 0
    recommend_vad = pct_silent > cfg["silence"]["corpus_percent_threshold"]

    silence_block = {
        "window_sec": cfg["silence"]["window_sec"],
        "ratio_threshold": silence_thresh,
        "n_lead_silent": n_lead_silent,
        "n_trail_silent": n_trail_silent,
        "n_either": n_either,
        "percent_high_silence_files": pct_silent,
        "lead_ratio_p50":  _pctl(acc.lead_silence_ratios, 50),
        "lead_ratio_p95":  _pctl(acc.lead_silence_ratios, 95),
        "trail_ratio_p50": _pctl(acc.trail_silence_ratios, 50),
        "trail_ratio_p95": _pctl(acc.trail_silence_ratios, 95),
        "recommend_vad_trimming": recommend_vad,
    }

    # ---- 7. Spectral centroid ----
    sc_mean = round(sum(acc.spectral_centroids) / len(acc.spectral_centroids), 1) \
        if acc.spectral_centroids else 0.0

    # ---- 8. Sample rate / channel check ----
    expected_sr = cfg["format"]["expected_sr"]
    expected_ch = cfg["format"]["expected_channels"]
    n_not_16k = sum(1 for sr in acc.sample_rates if sr != expected_sr)
    n_not_mono = sum(1 for ch in acc.channel_counts if ch != expected_ch)
    pct_not_16k = round(100 * n_not_16k / n_ok, 2) if n_ok else 0
    pct_not_mono = round(100 * n_not_mono / n_ok, 2) if n_ok else 0
    require_resample = n_not_16k > 0
    require_mono_downmix = n_not_mono > 0

    format_block = {
        "expected_sr": expected_sr,
        "expected_channels": expected_ch,
        "n_not_16k": n_not_16k,
        "n_not_mono": n_not_mono,
        "percent_not_16k": pct_not_16k,
        "percent_not_mono": pct_not_mono,
        "require_resampling": require_resample,
        "require_mono_downmix": require_mono_downmix,
    }

    # ====================================================================
    # Assemble output
    # ====================================================================

    decisions = {
        "apply_rms_normalization": recommend_rms,
        "drop_heavily_clipped": recommend_drop_clipped,
        "use_weighted_sampler": recommend_sampler,
        "apply_vad_trimming": recommend_vad,
        "require_resampling": require_resample,
        "require_mono_downmix": require_mono_downmix,
    }

    controls: dict[str, Any] = {
        "total_audio_hours_analysed": round(total_hrs, 3),
        "rms": rms_block,
        "clipping": clipping_block,
        "speaker_dominance": speaker_block,
        "age_distribution": age_block,
        "duration_profile": duration_block,
        "silence": silence_block,
        "spectral_centroid_mean": sc_mean,
        "sample_rate_check": format_block,
        "decisions": decisions,
    }

    # ---- Write output ----
    out_path = out / "eda_controls.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(dumps(controls, indent=2))

    # ---- Write diagnostic JSONs (research auditability) ----
    if err_load_ids:
        load_fail_path = out / "eda_failed_loads.json"
        with open(load_fail_path, "w", encoding="utf-8") as f:
            f.write(dumps({"utterance_ids": err_load_ids, "count": len(err_load_ids)}, indent=2))
        log.info(f"[EDA] saved {len(err_load_ids):,} load failures → {load_fail_path}")

    if acc.duration_mismatch_ids:
        mismatch_path = out / "eda_duration_mismatch.json"
        with open(mismatch_path, "w", encoding="utf-8") as f:
            f.write(dumps({"utterance_ids": acc.duration_mismatch_ids, "count": len(acc.duration_mismatch_ids)}, indent=2))
        log.info(f"[EDA] saved {len(acc.duration_mismatch_ids):,} duration mismatches → {mismatch_path}")

    # ====================================================================
    # Print summary
    # ====================================================================

    log.info("")
    log.info("[EDA] --- RMS ---")
    log.info(f"[EDA]   p01={rms_p01:.6f}  p50={rms_p50:.6f}  p99={rms_p99:.6f}")
    log.info(f"[EDA]   spread ratio       {rms_spread:>10.2f}")
    rc = "✓" if not recommend_rms else "!"
    log.info(f"[EDA]   {rc}  normalisation   {'RECOMMENDED' if recommend_rms else 'not needed'}")

    log.info("[EDA] --- CLIPPING ---")
    rc = "✓" if not recommend_drop_clipped else "✗"
    log.info(f"[EDA]   {rc}  heavily clipped {n_clipped_files:>10,}  ({pct_clipped}% of files)")

    log.info("[EDA] --- SPEAKERS ---")
    log.info(f"[EDA]   unique speakers    {len(acc.speaker_hours):>10,}")
    log.info(f"[EDA]   top-{top_k} hour share  {top_pct:>9.1f}%")
    rc = "!" if recommend_sampler else "✓"
    log.info(f"[EDA]   {rc}  weighted sampler {'RECOMMENDED' if recommend_sampler else 'not needed'}")

    log.info("[EDA] --- AGE ---")
    for bucket in sorted(acc.age_hours):
        hrs = acc.age_hours[bucket]
        pct = 100 * hrs / age_total if age_total > 0 else 0
        log.info(f"[EDA]   {bucket:<8s}  {hrs:>8.2f} hrs  ({pct:>5.1f}%)")

    log.info("[EDA] --- DURATION ---")
    log.info(f"[EDA]   p50={dur_p50:.2f}s  p90={dur_p90:.2f}s  p95={dur_p95:.2f}s  max={dur_max:.2f}s")
    log.info(f"[EDA]   p96={dur_p96:.2f}s  p97={dur_p97:.2f}s  p98={dur_p98:.2f}s  p99={dur_p99:.2f}s")
    log.info(f"[EDA]   above p99         {n_above_p99:>10,}  ({pct_above_p99}% of corpus)")
    log.info(f"[EDA]   theoretical batch {theoretical_batch:>10,}  (at p95, {target_vram}s budget)")

    log.info("[EDA] --- SILENCE ---")
    rc = "!" if recommend_vad else "✓"
    log.info(f"[EDA]   lead silent files  {n_lead_silent:>10,}")
    log.info(f"[EDA]   trail silent files {n_trail_silent:>10,}")
    log.info(f"[EDA]   {rc}  VAD trimming    {'RECOMMENDED' if recommend_vad else 'not needed'}  ({pct_silent}% affected)")

    log.info("[EDA] --- SPECTRAL ---")
    log.info(f"[EDA]   centroid mean      {sc_mean:>10.1f} Hz  (advisory)")

    log.info("[EDA] --- FORMAT ---")
    rc = "✗" if require_resample else "✓"
    log.info(f"[EDA]   {rc}  not {expected_sr} Hz     {n_not_16k:>10,}")
    rc2 = "✗" if n_not_mono > 0 else "✓"
    log.info(f"[EDA]   {rc2}  not mono          {n_not_mono:>10,}")
    if require_mono_downmix:
        log.info("[EDA]   !  mono downmix     REQUIRED")

    log.info("")
    log.info("[EDA] decisions:")
    for k, v in decisions.items():
        tag = "!" if v else "✓"
        log.info(f"[EDA]   {tag}  {k:<28s} {v}")

    log.info(f"[EDA] saved → {out_path}")

    # ====================================================================
    # Rewrite processed transcripts — remove failed loads + dur mismatches
    # ====================================================================
    bad_ids: set[str] = set(err_load_ids) | set(acc.duration_mismatch_ids)
    if bad_ids:
        log.info("")
        log.info("--- TRANSCRIPT CLEANUP ---")
        for mp_ in manifest_paths:
            kept_lines: list[str] = []
            removed_ids: list[str] = []
            with open(mp_, encoding="utf-8") as f:
                for line in f:
                    row = loads(line)
                    uid = row["utterance_id"]
                    if uid in bad_ids:
                        reason = "load_failure" if uid in err_load_ids else "duration_mismatch"
                        log.info(f"[TRANSCRIPT] remove  {uid}  ({reason})  from {Path(mp_).name}")
                        removed_ids.append(uid)
                    else:
                        kept_lines.append(line)
            if removed_ids:
                with open(mp_, "w", encoding="utf-8") as f:
                    f.writelines(kept_lines)
                log.info(f"[TRANSCRIPT] {Path(mp_).name}  {len(removed_ids)} row(s) removed  →  {len(kept_lines):,} kept")
        log.info("--- TRANSCRIPT CLEANUP END ---")

    # ====================================================================
    # Transcript status — final state of all processed transcript files
    # ====================================================================
    log.info("")
    log.info("--- TRANSCRIPT STATUS ---")
    total_rows_all = 0
    total_hrs_all  = 0.0
    for mp_ in manifest_paths:
        n_rows  = 0
        dur_sum = 0.0
        with open(mp_, encoding="utf-8") as f:
            for line in f:
                row = loads(line)
                n_rows  += 1
                dur_sum += row["audio_duration_sec"]
        hrs = dur_sum / 3600
        log.info(f"[TRANSCRIPT] {Path(mp_).name:<30s}  {n_rows:>8,} rows  {hrs:>8.2f} hrs")
        total_rows_all += n_rows
        total_hrs_all  += hrs
    log.info(f"[TRANSCRIPT] {'ALL':<30s}  {total_rows_all:>8,} rows  {total_hrs_all:>8.2f} hrs")
    log.info("--- TRANSCRIPT STATUS END ---")

    log.info("--- AUDIO EDA END ---")

    return controls


# ---------------------------------------------------------------------------
#  Orchestrator
# ---------------------------------------------------------------------------

class EDAProcessor:
    """Unified EDA — text sanitisation (Phase 0) + audio calibration."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def sanitize(self) -> dict[int, dict]:
        """Phase 0: text EDA — clean transcripts, flag drills."""
        return _sanity_fix(self.cfg)

    def audio_eda(self) -> dict[str, Any]:
        """Audio EDA — streaming analysis → eda_controls.json."""
        paths     = self.cfg["paths"]
        keys      = self.cfg["datasets"]
        processed = paths["processed"]

        manifest_paths = [f"{processed}/{k}_transcript.jsonl" for k in keys]
        audio_dirs     = [paths["audio_dirs"][k] for k in keys]
        output_dir     = paths["reports"]
        eda_cfg        = self.cfg["audio_eda"]

        return _run_audio_eda(manifest_paths, audio_dirs, output_dir, eda_cfg)

