#!/usr/bin/env python3
"""Per-dataset audio quality diagnostic.

Computes per-dataset breakdowns of:
  - SNR (10th-percentile frame energy as noise floor)
  - RMS distribution (p01, p10, p50, p90, p99)
  - Clipping rate (% samples with |x| >= 0.999)
  - F0 median (energy-gated pitch detection)
  - Original sample rate distribution
  - Duration distribution (p10, p50, p90, p99, mean)
  - Leading/trailing silence ratio

Reads 100% of DS1 (~12k files) and a matched random sample of DS2.
Results saved to data/reports/per_dataset_audio_diagnostic.json.

Usage:
    python scripts/per_dataset_audio_diagnostic.py [--data-root data] [--workers 8] [--ds2-sample 12000]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

import soundfile as sf
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
_TARGET_SR = 16_000
_F0_MIN_HZ = 50.0
_F0_MAX_HZ = 600.0
_EPS = 1e-10
_CLIP_THRESHOLD = 0.999


# ---------------------------------------------------------------------------
#  Per-file result
# ---------------------------------------------------------------------------
@dataclass
class FileResult:
    utterance_id: str
    dataset: str
    age_bucket: str
    child_id: str
    duration_sec: float       # actual decoded duration
    original_sr: int          # sample rate before resampling
    rms: float
    snr_db: float
    clipping_ratio: float     # fraction of samples with |x| >= threshold
    f0_median: float          # 0.0 if no voiced frames
    silence_lead_ratio: float # RMS(first 0.5s) / RMS(full)
    silence_trail_ratio: float


# ---------------------------------------------------------------------------
#  Worker state
# ---------------------------------------------------------------------------
_w_resamplers: dict[int, torchaudio.transforms.Resample] = {}


def _worker_init() -> None:
    """Silence torchaudio warnings and limit threads per worker."""
    import warnings as _w
    _w.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    torch.set_num_threads(1)


def _analyse_file(args: tuple[str, str, str, str, str]) -> FileResult | None:
    """Analyse one audio file. Returns None on error."""
    audio_path, utterance_id, dataset, age_bucket, child_id = args
    global _w_resamplers

    try:
        info = sf.info(audio_path)
        original_sr = info.samplerate
    except Exception:
        return None

    try:
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)  # (channels, samples)
    except Exception:
        return None

    if wav.numel() == 0:
        return None

    # Mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # (N,)

    # Resample to 16kHz
    if sr != _TARGET_SR:
        if sr not in _w_resamplers:
            _w_resamplers[sr] = torchaudio.transforms.Resample(sr, _TARGET_SR)
        wav = _w_resamplers[sr](wav)

    n_samples = wav.numel()
    duration_sec = n_samples / _TARGET_SR

    # ---- RMS ----
    rms = torch.sqrt(torch.mean(wav ** 2)).item()
    if rms != rms:
        rms = 0.0

    # ---- Clipping ----
    clipping_ratio = (wav.abs() >= _CLIP_THRESHOLD).float().mean().item()

    # ---- SNR (10th-percentile frame energy as noise floor) ----
    frame_size = min(int(0.025 * _TARGET_SR), n_samples)  # 25ms
    hop = max(frame_size // 2, 1)

    if n_samples >= frame_size:
        frames = wav.unfold(0, frame_size, hop)
        frame_energies = frames.pow(2).mean(dim=1).sqrt()
    else:
        frame_energies = torch.sqrt(torch.mean(wav ** 2)).unsqueeze(0)

    sorted_e, _ = frame_energies.sort()
    noise_idx = max(0, int(sorted_e.numel() * 0.10) - 1)
    noise_rms = max(sorted_e[noise_idx].item(), _EPS)
    snr_db = float(20.0 * math.log10(max(rms, _EPS) / noise_rms))

    # ---- Silence ratios (leading/trailing 0.5s) ----
    window_samples = int(0.5 * _TARGET_SR)
    if n_samples > window_samples * 2:
        lead_rms = torch.sqrt(torch.mean(wav[:window_samples] ** 2)).item()
        trail_rms = torch.sqrt(torch.mean(wav[-window_samples:] ** 2)).item()
        silence_lead = lead_rms / max(rms, _EPS)
        silence_trail = trail_rms / max(rms, _EPS)
    else:
        silence_lead = 1.0
        silence_trail = 1.0

    # ---- F0 (energy-gated pitch detection) ----
    f0_median = 0.0
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(
            wav.unsqueeze(0), _TARGET_SR,
            freq_low=int(_F0_MIN_HZ),
            freq_high=int(_F0_MAX_HZ),
        )
        pitch_vals = pitch[0]
        energy_gate = rms * 0.3
        pitch_frames = pitch_vals.numel()
        pitch_hop = max(1, n_samples // max(pitch_frames, 1))

        if n_samples >= pitch_hop and pitch_frames > 0:
            usable = pitch_frames * pitch_hop
            fe = wav[:usable].reshape(pitch_frames, pitch_hop).pow(2).mean(dim=1).sqrt()
        else:
            fe = torch.full((pitch_frames,), rms)

        valid = (
            (pitch_vals >= _F0_MIN_HZ)
            & (pitch_vals <= _F0_MAX_HZ)
            & (fe > energy_gate)
        )
        valid_pitch = pitch_vals[valid]
        if valid_pitch.numel() > 0:
            f0_median = float(valid_pitch.median().item())
    except Exception:
        pass

    return FileResult(
        utterance_id=utterance_id,
        dataset=dataset,
        age_bucket=age_bucket,
        child_id=child_id,
        duration_sec=duration_sec,
        original_sr=original_sr,
        rms=rms,
        snr_db=snr_db,
        clipping_ratio=clipping_ratio,
        f0_median=f0_median,
        silence_lead_ratio=silence_lead,
        silence_trail_ratio=silence_trail,
    )


# ---------------------------------------------------------------------------
#  Load manifest and build worker args
# ---------------------------------------------------------------------------
def _load_manifest(
    jsonl_path: Path,
    audio_dir: Path,
    dataset_label: str,
    max_rows: int | None = None,
    seed: int = 1507,
) -> list[tuple[str, str, str, str, str]]:
    """Read JSONL, resolve audio paths, optionally sample."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            # audio_path in JSONL is relative like "audio/U_xxx.flac"
            # Resolve against the dataset's audio directory
            fname = Path(row["audio_path"]).name
            full_path = str(audio_dir / fname)
            rows.append((
                full_path,
                row["utterance_id"],
                dataset_label,
                row.get("age_bucket", "unknown"),
                row.get("child_id", "unknown"),
            ))

    if max_rows and len(rows) > max_rows:
        total = len(rows)
        rng = random.Random(seed)
        rows = rng.sample(rows, max_rows)
        log.info("  Sampled %d / %d rows from %s", max_rows, total, dataset_label)

    return rows


# ---------------------------------------------------------------------------
#  Aggregation
# ---------------------------------------------------------------------------
def _percentile(vals: list[float], p: float) -> float:
    """Nearest-rank percentile (matches numpy default)."""
    if not vals:
        return 0.0
    vals_s = sorted(vals)
    # Nearest-rank: idx = ceil(p/100 * n) - 1, clamped
    idx = max(0, min(math.ceil(p / 100.0 * len(vals_s)) - 1, len(vals_s) - 1))
    return vals_s[idx]


def _aggregate(results: list[FileResult], label: str) -> dict[str, Any]:
    """Compute summary statistics for one dataset."""
    n = len(results)
    if n == 0:
        return {"n": 0, "label": label}

    rms_vals = [r.rms for r in results]
    snr_vals = [r.snr_db for r in results]
    dur_vals = [r.duration_sec for r in results]
    clip_vals = [r.clipping_ratio for r in results]
    f0_vals = [r.f0_median for r in results if r.f0_median > 0]
    lead_vals = [r.silence_lead_ratio for r in results]
    trail_vals = [r.silence_trail_ratio for r in results]

    sr_counter = Counter(r.original_sr for r in results)
    age_counter = Counter(r.age_bucket for r in results)
    speaker_count = len(set(r.child_id for r in results))

    # Per-age F0
    age_f0: dict[str, list[float]] = {}
    for r in results:
        if r.f0_median > 0:
            age_f0.setdefault(r.age_bucket, []).append(r.f0_median)

    # Per-age SNR
    age_snr: dict[str, list[float]] = {}
    for r in results:
        age_snr.setdefault(r.age_bucket, []).append(r.snr_db)

    clipped_files = sum(1 for c in clip_vals if c > 0.0001)

    # Per-age RMS
    age_rms: dict[str, list[float]] = {}
    for r in results:
        age_rms.setdefault(r.age_bucket, []).append(r.rms)

    # Per-age duration
    age_dur: dict[str, list[float]] = {}
    for r in results:
        age_dur.setdefault(r.age_bucket, []).append(r.duration_sec)

    # Per-age clipping
    age_clip: dict[str, list[float]] = {}
    for r in results:
        age_clip.setdefault(r.age_bucket, []).append(r.clipping_ratio)

    return {
        "label": label,
        "n_files": n,
        "n_speakers": speaker_count,
        "age_distribution": dict(sorted(age_counter.items())),
        "sample_rate_distribution": {str(k): v for k, v in sorted(sr_counter.items())},
        "duration": {
            "mean": sum(dur_vals) / n,
            "p10": _percentile(dur_vals, 10),
            "p50": _percentile(dur_vals, 50),
            "p90": _percentile(dur_vals, 90),
            "p99": _percentile(dur_vals, 99),
            "total_hours": sum(dur_vals) / 3600,
        },
        "rms": {
            "mean": sum(rms_vals) / n,
            "p01": _percentile(rms_vals, 1),
            "p10": _percentile(rms_vals, 10),
            "p50": _percentile(rms_vals, 50),
            "p90": _percentile(rms_vals, 90),
            "p99": _percentile(rms_vals, 99),
            "spread_p99_p01": _percentile(rms_vals, 99) / max(_percentile(rms_vals, 1), _EPS),
        },
        "snr_db": {
            "mean": sum(snr_vals) / n,
            "p01": _percentile(snr_vals, 1),
            "p10": _percentile(snr_vals, 10),
            "p50": _percentile(snr_vals, 50),
            "p90": _percentile(snr_vals, 90),
            "p99": _percentile(snr_vals, 99),
        },
        "clipping": {
            "files_with_clipping": clipped_files,
            "pct_files_clipped": 100.0 * clipped_files / n,
            "mean_clip_ratio": sum(clip_vals) / n,
            "max_clip_ratio": max(clip_vals),
        },
        "f0_hz": {
            "n_voiced": len(f0_vals),
            "pct_voiced": 100.0 * len(f0_vals) / n,
            "mean": sum(f0_vals) / max(len(f0_vals), 1),
            "p10": _percentile(f0_vals, 10),
            "p50": _percentile(f0_vals, 50),
            "p90": _percentile(f0_vals, 90),
            "per_age": {
                age: {
                    "p10": _percentile(vals, 10),
                    "p50": _percentile(vals, 50),
                    "p90": _percentile(vals, 90),
                    "n": len(vals),
                }
                for age, vals in sorted(age_f0.items())
            },
        },
        "snr_per_age": {
            age: {
                "mean": sum(vals) / len(vals),
                "p10": _percentile(vals, 10),
                "p50": _percentile(vals, 50),
                "p90": _percentile(vals, 90),
                "n": len(vals),
            }
            for age, vals in sorted(age_snr.items())
        },
        "rms_per_age": {
            age: {
                "mean": sum(vals) / len(vals),
                "p50": _percentile(vals, 50),
                "n": len(vals),
            }
            for age, vals in sorted(age_rms.items())
        },
        "duration_per_age": {
            age: {
                "mean": sum(vals) / len(vals),
                "p50": _percentile(vals, 50),
                "total_hours": sum(vals) / 3600,
                "n": len(vals),
            }
            for age, vals in sorted(age_dur.items())
        },
        "clipping_per_age": {
            age: {
                "pct_files_clipped": 100.0 * sum(1 for v in vals if v > 0.0001) / max(len(vals), 1),
                "n": len(vals),
            }
            for age, vals in sorted(age_clip.items())
        },
        "silence": {
            "lead_ratio_mean": sum(lead_vals) / n,
            "lead_ratio_p10": _percentile(lead_vals, 10),
            "trail_ratio_mean": sum(trail_vals) / n,
            "trail_ratio_p10": _percentile(trail_vals, 10),
            "files_with_quiet_lead": sum(1 for v in lead_vals if v < 0.1),
            "files_with_quiet_trail": sum(1 for v in trail_vals if v < 0.1),
        },
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Per-dataset audio quality diagnostic")
    parser.add_argument("--data-root", default="data", help="Path to data/ directory")
    parser.add_argument("--workers", type=int, default=8, help="Multiprocessing pool size")
    parser.add_argument("--ds2-sample", type=int, default=12000,
                        help="Random sample size for DS2 (0 = all)")
    parser.add_argument("--seed", type=int, default=1507)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    raw_dir = data_root / "raw"
    reports_dir = data_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ds1_jsonl = raw_dir / "1_train_phon_transcripts.jsonl"
    ds2_jsonl = raw_dir / "2_train_phon_transcripts.jsonl"
    ds1_audio = raw_dir / "1_audio"
    ds2_audio = raw_dir / "2_audio"

    for p in [ds1_jsonl, ds2_jsonl, ds1_audio, ds2_audio]:
        if not p.exists():
            log.error("Missing: %s", p)
            sys.exit(1)

    # ---- Load manifests ----
    log.info("Loading DS1 manifest (100%%) ...")
    ds1_args = _load_manifest(ds1_jsonl, ds1_audio, "DS1")
    log.info("  DS1: %d files", len(ds1_args))

    ds2_max = args.ds2_sample if args.ds2_sample > 0 else None
    log.info("Loading DS2 manifest (%s) ...", f"sample {ds2_max}" if ds2_max else "100%")
    ds2_args = _load_manifest(ds2_jsonl, ds2_audio, "DS2", max_rows=ds2_max, seed=args.seed)
    log.info("  DS2: %d files", len(ds2_args))

    all_args = ds1_args + ds2_args
    log.info("Total: %d files to analyse with %d workers", len(all_args), args.workers)

    # ---- Process ----
    t0 = time.time()
    results: list[FileResult] = []
    errors = 0

    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        for i, result in enumerate(pool.imap_unordered(_analyse_file, all_args, chunksize=64)):
            if result is None:
                errors += 1
            else:
                results.append(result)
            if (i + 1) % 2000 == 0 or (i + 1) == len(all_args):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                log.info("  %d / %d  (%.0f files/s, %d errors)", i + 1, len(all_args), rate, errors)

    elapsed = time.time() - t0
    log.info("Analysis complete: %d files in %.1fs (%.0f files/s), %d errors",
             len(results), elapsed, len(results) / max(elapsed, 0.1), errors)

    # ---- Split by dataset ----
    ds1_results = [r for r in results if r.dataset == "DS1"]
    ds2_results = [r for r in results if r.dataset == "DS2"]

    ds1_summary = _aggregate(ds1_results, "DS1")
    ds2_summary = _aggregate(ds2_results, "DS2")

    # ---- Comparison ----
    comparison = {}
    if ds1_summary["n_files"] > 0 and ds2_summary["n_files"] > 0:
        comparison = {
            "snr_diff_db": ds1_summary["snr_db"]["p50"] - ds2_summary["snr_db"]["p50"],
            "rms_diff": ds1_summary["rms"]["p50"] - ds2_summary["rms"]["p50"],
            "f0_diff_hz": ds1_summary["f0_hz"]["p50"] - ds2_summary["f0_hz"]["p50"],
            "duration_diff_sec": ds1_summary["duration"]["p50"] - ds2_summary["duration"]["p50"],
            "ds1_pct_non_16k": 100.0 * (1 - ds1_summary["sample_rate_distribution"].get("16000", 0) / ds1_summary["n_files"]),
            "ds2_pct_non_16k": 100.0 * (1 - ds2_summary["sample_rate_distribution"].get("16000", 0) / ds2_summary["n_files"]),
        }

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ds1_files_analysed": len(ds1_results),
        "ds2_files_analysed": len(ds2_results),
        "ds2_sampling": f"random {args.ds2_sample}" if args.ds2_sample > 0 else "all",
        "errors": errors,
        "elapsed_sec": round(elapsed, 1),
        "DS1": ds1_summary,
        "DS2": ds2_summary,
        "comparison": comparison,
    }

    # ---- Save ----
    out_path = reports_dir / "per_dataset_audio_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved to %s", out_path)

    # ---- Pretty-print summary ----
    print("\n" + "=" * 72)
    print("  PER-DATASET AUDIO QUALITY DIAGNOSTIC")
    print("=" * 72)

    for label, s in [("DS1", ds1_summary), ("DS2", ds2_summary)]:
        print(f"\n  {label}  ({s['n_files']} files, {s.get('n_speakers', '?')} speakers)")
        print(f"  {'─' * 60}")
        d = s["duration"]
        print(f"  Duration      p50={d['p50']:.2f}s  p90={d['p90']:.2f}s  p99={d['p99']:.2f}s  total={d['total_hours']:.1f}h")
        r = s["rms"]
        print(f"  RMS           p01={r['p01']:.4f}  p50={r['p50']:.4f}  p99={r['p99']:.4f}  spread={r['spread_p99_p01']:.1f}×")
        sn = s["snr_db"]
        print(f"  SNR (dB)      p10={sn['p10']:.1f}  p50={sn['p50']:.1f}  p90={sn['p90']:.1f}  mean={sn['mean']:.1f}")
        c = s["clipping"]
        print(f"  Clipping      {c['files_with_clipping']} files ({c['pct_files_clipped']:.1f}%)  max_ratio={c['max_clip_ratio']:.4f}")
        f0 = s["f0_hz"]
        print(f"  F0 (Hz)       p10={f0['p10']:.0f}  p50={f0['p50']:.0f}  p90={f0['p90']:.0f}  voiced={f0['pct_voiced']:.0f}%")
        if f0.get("per_age"):
            ages = "  ".join(f"{a}={v['p50']:.0f}Hz(n={v['n']})" for a, v in f0["per_age"].items())
            print(f"  F0 by age     {ages}")
        if s.get("rms_per_age"):
            age_rms = "  ".join(f"{a}={v['p50']:.4f}(n={v['n']})" for a, v in s["rms_per_age"].items())
            print(f"  RMS by age    {age_rms}")
        if s.get("duration_per_age"):
            age_dur = "  ".join(f"{a}={v['p50']:.1f}s/{v['total_hours']:.1f}h(n={v['n']})" for a, v in s["duration_per_age"].items())
            print(f"  Dur by age    {age_dur}")
        sr_dist = s["sample_rate_distribution"]
        sr_str = "  ".join(f"{k}Hz={v}" for k, v in sr_dist.items())
        print(f"  Sample rates  {sr_str}")
        si = s["silence"]
        print(f"  Silence       lead_quiet={si['files_with_quiet_lead']}  trail_quiet={si['files_with_quiet_trail']}")
        if s.get("snr_per_age"):
            age_snr = "  ".join(f"{a}={v['p50']:.1f}dB(n={v['n']})" for a, v in s["snr_per_age"].items())
            print(f"  SNR by age    {age_snr}")
        print(f"  Ages          {s['age_distribution']}")

    if comparison:
        print(f"\n  COMPARISON (DS1 − DS2)")
        print(f"  {'─' * 60}")
        print(f"  SNR diff:       {comparison['snr_diff_db']:+.1f} dB  (negative = DS1 noisier)")
        print(f"  RMS diff:       {comparison['rms_diff']:+.4f}")
        print(f"  F0 diff:        {comparison['f0_diff_hz']:+.0f} Hz  (positive = DS1 higher pitch)")
        print(f"  Duration diff:  {comparison['duration_diff_sec']:+.2f} s")
        print(f"  Non-16kHz:      DS1={comparison['ds1_pct_non_16k']:.1f}%  DS2={comparison['ds2_pct_non_16k']:.1f}%")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
