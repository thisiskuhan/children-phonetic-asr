"""
Model-Selection Acoustic Diagnostics
=====================================

Produces research-grade acoustic signals that allow evidence-based
selection between **Wav2Vec2**, **HuBERT**, and **WavLM** based purely
on corpus characteristics.

Runs AFTER the data split.  Primary analysis = TRAIN SET ONLY.
A lightweight TRAIN vs VAL comparison is stored for observability
but MUST NOT influence model recommendation scores.

Output:  ``data/reports/model_selection_eda.json``

Design constraints:
    • torchaudio only (no librosa).  soundfile allowed.
    • Streaming + memory-safe for 150 k+ files.
    • Multiprocessing via mp.Pool — identical pattern to audio EDA.
    • Deterministic (seeded).
    • float32 accumulation.  ε-guarded log / division.
    • Reuses ``load_audio_mono`` and ``init_torch_worker`` from utils.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from tqdm import tqdm

from utils import (
    loads, dumps, nearest_rank_pctl, load_audio_mono, init_torch_worker,
    resolve_audio_path,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_EPS = 1e-10                      # numerical guard for log / division
_N_MFCC = 13                     # standard MFCC dimensionality
_N_FFT = 1024
_HOP = 512
_FRAME_LEN = _N_FFT
_TARGET_SR = 16000                # resample to this if needed
_F0_MIN_HZ = 50.0                # child-speech safe range
_F0_MAX_HZ = 600.0
_SPEAKER_DISTANCE_MAX = 5_000    # if speakers > this → random sample

# ---------------------------------------------------------------------------
#  Per-file result returned by each worker
# ---------------------------------------------------------------------------

@dataclass
class _MSFileResult:
    utterance_id: str
    child_id: str
    age_bucket: str
    duration_sec: float
    phonemes_per_sec: float
    rms: float
    snr_db: float
    reverb_slope: float
    spectral_flatness: float
    zcr_variance: float
    harmonic_peak_count: float
    mfcc_mean: list[float]
    spectral_centroid: float
    f0_median: float         # 0.0 if no voiced frames detected


# ---------------------------------------------------------------------------
#  Worker state — module-level, set once per process
# ---------------------------------------------------------------------------

_w_cfg: dict[str, Any] = {}


def _ms_worker_init(cfg: dict[str, Any]) -> None:
    """Initialise each pool worker with config + cached transforms."""
    global _w_cfg
    _w_cfg = cfg
    init_torch_worker()

    # Pre-create reusable transforms (one per worker, avoids re-allocation)
    _w_cfg["_hann"] = torch.hann_window(_N_FFT)
    _w_cfg["_mfcc_transform"] = torchaudio.transforms.MFCC(
        sample_rate=_TARGET_SR,
        n_mfcc=_N_MFCC,
        melkwargs={"n_fft": _N_FFT, "hop_length": _HOP, "n_mels": 40},
    )
    # Cached freq axis for spectral centroid (always same after resample)
    n_freq_bins = _N_FFT // 2 + 1
    _w_cfg["_freqs"] = torch.linspace(0, _TARGET_SR / 2, n_freq_bins)
    # Lazy cache for Resample transforms keyed by source SR
    _w_cfg["_resamplers"] = {}


# ---------------------------------------------------------------------------
#  Error sentinels
# ---------------------------------------------------------------------------

_ERR_LOAD  = "__ms_err_load__"
_ERR_EMPTY = "__ms_err_empty__"


# ---------------------------------------------------------------------------
#  Per-file analysis function (runs in worker process)
# ---------------------------------------------------------------------------

def _ms_analyse_file(
    args: tuple[str, str, str, str, float, float],
) -> _MSFileResult | str:
    """
    Analyse one audio file for all model-selection metrics.

    Parameters (packed tuple for Pool.imap):
        audio_path       — absolute path
        utterance_id     — row ID
        child_id         — speaker
        age_bucket       — age group
        duration_sec     — from manifest
        phonemes_per_sec — pre-computed from metadata
    """
    audio_path, utterance_id, child_id, age_bucket, duration_sec, pps = args

    try:
        wav, sr, _ = load_audio_mono(audio_path)
    except Exception:
        return _ERR_LOAD

    if wav.numel() == 0:
        return _ERR_EMPTY

    # ---- Resample to target SR if needed (cached transform) ----
    if sr != _TARGET_SR:
        resamplers = _w_cfg["_resamplers"]
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, _TARGET_SR)
        wav = resamplers[sr](wav)
        sr = _TARGET_SR

    n_samples = wav.numel()
    actual_dur = n_samples / sr

    # ==================================================================
    # 1. RMS + SNR
    # ==================================================================
    rms = torch.sqrt(torch.mean(wav ** 2)).item()
    if rms != rms:  # NaN guard
        rms = 0.0

    # Noise floor: 10th percentile of short-time frame energy
    frame_size = min(int(0.025 * sr), n_samples)   # 25ms frames
    hop_frames = max(frame_size // 2, 1)

    if n_samples >= frame_size:
        frames = wav.unfold(0, frame_size, hop_frames)          # (n_frames, frame_size)
        frame_energies_t = frames.pow(2).mean(dim=1).sqrt()     # (n_frames,)
    else:
        frame_energies_t = torch.sqrt(torch.mean(wav ** 2)).unsqueeze(0)

    frame_energies_sorted, _ = frame_energies_t.sort()
    noise_idx = max(0, int(frame_energies_sorted.numel() * 0.10) - 1)
    noise_rms = max(frame_energies_sorted[noise_idx].item(), _EPS)
    snr_db = float(20.0 * math.log10(max(rms, _EPS) / noise_rms))

    # ==================================================================
    # 2. Reverb proxy — energy decay slope
    # ==================================================================
    # Log energy per frame → detect decay regions → linear fit slope
    log_energies = torch.log(frame_energies_t.clamp(min=_EPS))
    # Use the last 30% of the signal's frames as the "tail"
    tail_start = max(0, int(log_energies.numel() * 0.7))
    tail = log_energies[tail_start:]
    reverb_slope = 0.0
    if tail.numel() >= 3:
        x = torch.arange(tail.numel(), dtype=torch.float32)
        x_mean = x.mean()
        y_mean = tail.mean()
        num = ((x - x_mean) * (tail - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        reverb_slope = float(num / den.clamp(min=_EPS))

    # ==================================================================
    # 3. Overlap proxy — spectral flatness, ZCR variance, harmonic peaks
    # ==================================================================
    hann = _w_cfg["_hann"]
    stft = torch.stft(wav, n_fft=_N_FFT, hop_length=_HOP,
                       window=hann, return_complex=True)
    mag = stft.abs()           # (freq_bins, time_frames)

    # Spectral flatness: geometric mean / arithmetic mean of magnitude spectrum
    log_mag = torch.log(mag + _EPS)
    geo_mean = torch.exp(log_mag.mean(dim=0))     # (time_frames,)
    arith_mean = mag.mean(dim=0) + _EPS
    flatness_per_frame = geo_mean / arith_mean
    spectral_flatness = flatness_per_frame.mean().item()

    # ZCR variance
    signs = torch.sign(wav)
    if signs.numel() > 1:
        zcr_per_sample = (signs[1:] != signs[:-1]).float()
        n_zcr = zcr_per_sample.numel()
        if n_zcr >= frame_size:
            zcr_frames_t = zcr_per_sample.unfold(0, frame_size, hop_frames).mean(dim=1)
            zcr_variance = float(zcr_frames_t.var().item()) if zcr_frames_t.numel() > 1 else 0.0
        else:
            zcr_variance = 0.0
    else:
        zcr_variance = 0.0

    # Harmonic peak count — count prominent peaks in the averaged magnitude spectrum
    avg_spectrum = mag.mean(dim=1)   # (freq_bins,)
    threshold = avg_spectrum.mean().item() * 1.5
    local_max = (avg_spectrum[1:-1] > avg_spectrum[:-2]) & (avg_spectrum[1:-1] > avg_spectrum[2:])
    prominent = local_max & (avg_spectrum[1:-1] > threshold)
    harmonic_peak_count = float(prominent.sum().item())

    # ==================================================================
    # 4. MFCC mean (13-dim)
    # ==================================================================
    mfcc_transform = _w_cfg["_mfcc_transform"]
    mfcc = mfcc_transform(wav.unsqueeze(0))   # (1, n_mfcc, time)
    mfcc_mean = mfcc[0].mean(dim=1).tolist()  # (n_mfcc,)

    # ==================================================================
    # 5. Spectral centroid
    # ==================================================================
    freqs = _w_cfg["_freqs"]
    frame_mags = mag.sum(dim=0)
    valid_frames = frame_mags > _EPS
    centroid = 0.0
    if valid_frames.any():
        weighted = (freqs.unsqueeze(1) * mag).sum(dim=0)
        centroids = weighted[valid_frames] / frame_mags[valid_frames]
        centroid = centroids.mean().item()

    # ==================================================================
    # 6. F0 — robust pitch (median of voiced frames)
    # ==================================================================
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(
            wav.unsqueeze(0), sr,
            freq_low=int(_F0_MIN_HZ),
            freq_high=int(_F0_MAX_HZ),
        )
        pitch_vals = pitch[0]
        # Energy gate — only trust frames with sufficient energy
        energy_gate_threshold = rms * 0.3
        pitch_frames = pitch_vals.numel()
        pitch_hop = max(1, n_samples // max(pitch_frames, 1))
        # Build frame energies aligned with pitch frames
        if n_samples >= pitch_hop and pitch_frames > 0:
            # Trim wav to exact multiple of pitch_hop for clean reshape
            usable = pitch_frames * pitch_hop
            fe_t = wav[:usable].reshape(pitch_frames, pitch_hop).pow(2).mean(dim=1).sqrt()
        else:
            fe_t = torch.full((pitch_frames,), rms)

        valid_mask = (
            (pitch_vals >= _F0_MIN_HZ)
            & (pitch_vals <= _F0_MAX_HZ)
            & (fe_t > energy_gate_threshold)
        )
        valid_pitch = pitch_vals[valid_mask]
        f0_median = float(valid_pitch.median().item()) if valid_pitch.numel() > 0 else 0.0
    except Exception:
        f0_median = 0.0

    return _MSFileResult(
        utterance_id=utterance_id,
        child_id=child_id,
        age_bucket=age_bucket,
        duration_sec=actual_dur,
        phonemes_per_sec=pps,
        rms=rms,
        snr_db=snr_db,
        reverb_slope=reverb_slope,
        spectral_flatness=spectral_flatness,
        zcr_variance=zcr_variance,
        harmonic_peak_count=harmonic_peak_count,
        mfcc_mean=mfcc_mean,
        spectral_centroid=centroid,
        f0_median=f0_median,
    )


# ---------------------------------------------------------------------------
#  Manifest streaming → worker args
# ---------------------------------------------------------------------------

def _iter_sft_manifest(
    path: str,
    audio_dirs: dict[int, str],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, str, float, float]]]:
    """Read SFT JSONL, return (rows_metadata, worker_args).

    Returns two parallel lists so that we can feed worker_args to the pool
    while keeping lightweight metadata for the corpus-level aggregation
    without storing raw waveforms.
    """
    rows: list[dict[str, Any]] = []
    worker_args: list[tuple[str, str, str, str, float, float]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = loads(line)
            uid = r["utterance_id"]
            cid = r["child_id"]
            age = r.get("age_bucket", "unknown")
            dur = r["audio_duration_sec"]
            text = r.get("phonetic_text", "")
            ds = r["dataset"]
            n_phonemes = len(text.replace(" ", ""))
            pps = n_phonemes / max(dur, _EPS)

            abs_path = resolve_audio_path(r["audio_path"], ds, audio_dirs)

            rows.append({
                "utterance_id": uid,
                "child_id": cid,
                "age_bucket": age,
                "duration_sec": dur,
                "phonemes_per_sec": pps,
            })
            worker_args.append((abs_path, uid, cid, age, dur, pps))

    return rows, worker_args


# ---------------------------------------------------------------------------
#  Percentile helper (thin wrapper — matches existing pattern)
# ---------------------------------------------------------------------------

def _pctl(vals: list[float], p: float, decimals: int = 6) -> float:
    return nearest_rank_pctl(vals, p, decimals=decimals)


# ---------------------------------------------------------------------------
#  Cosine distance helpers
# ---------------------------------------------------------------------------

def _cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine_similarity.  Returns 0.0 on degenerate inputs."""
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    na = torch.norm(ta)
    nb = torch.norm(tb)
    if na < _EPS or nb < _EPS:
        return 0.0
    sim = torch.dot(ta, tb) / (na * nb)
    return float(1.0 - sim.clamp(-1.0, 1.0))


def _cosine_distance_matrix(vectors: list[list[float]]) -> list[float]:
    """Return flattened upper-triangle cosine distances (fully vectorised)."""
    n = len(vectors)
    if n < 2:
        return [0.0]
    t = torch.tensor(vectors, dtype=torch.float32)
    norms = torch.norm(t, dim=1, keepdim=True).clamp(min=_EPS)
    t_normed = t / norms
    sim_matrix = t_normed @ t_normed.T
    # Extract upper triangle without Python loop
    rows, cols = torch.triu_indices(n, n, offset=1)
    dists = (1.0 - sim_matrix[rows, cols].clamp(-1.0, 1.0)).tolist()
    return dists


def _logistic_severity(x: float, x0: float, k: float) -> float:
    """Logistic severity ∈ [0, 1].  Higher *x* → higher severity.

    Parameters
    ----------
    x  : raw corpus difficulty metric (higher = more challenging).
    x0 : midpoint — severity = 0.5 when x == x0.
         Corpus-anchored: set to p50 of the difficulty metric.
    k  : steepness.  Corpus-anchored: k = 4 / (p90 − p10) so that
         the logistic maps the corpus spread to ~[0.12, 0.88].
    """
    z = k * (x - x0)
    z = max(-30.0, min(30.0, z))          # prevent overflow
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
#  Plot generation helpers
# ---------------------------------------------------------------------------

def _save_plots(report: dict[str, Any], plot_dir: Path) -> list[str]:
    """Generate diagnostic plots. Returns list of saved filenames."""
    saved: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("[MODEL-SEL] matplotlib not available — skipping plots")
        return saved

    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. SNR distribution
    snr = report.get("snr", {})
    if snr:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = [snr.get(f"p{p:02d}", 0) for p in range(1, 100)]
        ax.plot(range(1, 100), vals, color="#2196F3", linewidth=1.5)
        ax.set_xlabel("Percentile")
        ax.set_ylabel("SNR (dB)")
        ax.set_title("SNR Distribution (Train Set)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = "ms_snr_distribution.png"
        fig.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)
        saved.append(fname)

    # 2. Model scores bar chart
    signals = report.get("model_recommendation_signals", {})
    scores = {
        "Wav2Vec2": signals.get("wav2vec2_score", 0),
        "HuBERT": signals.get("hubert_score", 0),
        "WavLM": signals.get("wavlm_score", 0),
    }
    if any(v > 0 for v in scores.values()):
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#FF9800", "#4CAF50", "#2196F3"]
        bars = ax.bar(scores.keys(), scores.values(), color=colors, width=0.5)
        ax.set_ylabel("Score (0-10)")
        ax.set_title("Model Recommendation Scores")
        ax.set_ylim(0, 10.5)
        for bar, val in zip(bars, scores.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fname = "ms_model_scores.png"
        fig.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)
        saved.append(fname)

    # 3. Speaker acoustic distance heatmap (if available)
    spk_dist = report.get("speaker_acoustic_distance", {})
    if spk_dist and "p50" in spk_dist:
        fig, ax = plt.subplots(figsize=(6, 4))
        dist_vals = [spk_dist.get("p50", 0), spk_dist.get("p95", 0), spk_dist.get("max", 0)]
        labels = ["p50", "p95", "max"]
        ax.barh(labels, dist_vals, color=["#66BB6A", "#FFA726", "#EF5350"])
        ax.set_xlabel("Cosine Distance")
        ax.set_title("Cross-Speaker Acoustic Distance")
        ax.set_xlim(0, max(dist_vals) * 1.15 + 0.01)
        fig.tight_layout()
        fname = "ms_speaker_distance.png"
        fig.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)
        saved.append(fname)

    # 4. Age acoustic mismatch heatmap
    age_mm = report.get("age_acoustic_mismatch", {})
    dist_matrix = age_mm.get("cosine_distance_matrix", {})
    if dist_matrix:
        buckets = sorted(dist_matrix.keys())
        n = len(buckets)
        if n >= 2:
            mat = [[0.0] * n for _ in range(n)]
            for i, b1 in enumerate(buckets):
                for j, b2 in enumerate(buckets):
                    mat[i][j] = dist_matrix.get(b1, {}).get(b2, 0.0)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(n))
            ax.set_xticklabels(buckets, rotation=45, ha="right")
            ax.set_yticks(range(n))
            ax.set_yticklabels(buckets)
            ax.set_title("Age Bucket Acoustic Distance")
            fig.colorbar(im, ax=ax, label="Cosine Distance")
            fig.tight_layout()
            fname = "ms_age_acoustic_mismatch.png"
            fig.savefig(plot_dir / fname, dpi=150)
            plt.close(fig)
            saved.append(fname)

    return saved


# ---------------------------------------------------------------------------
#  Core analysis pipeline
# ---------------------------------------------------------------------------

def run_model_selection_eda(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Execute model-selection acoustic diagnostics on the train set.

    Parameters
    ----------
    cfg : dict
        Full project config (as returned by ``load_config``).

    Returns
    -------
    dict
        The model_selection_eda report (also written to JSON).
    """
    paths      = cfg["paths"]
    audio_dirs = {int(k): v for k, v in paths["audio_dirs"].items()}
    reports    = Path(paths["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    processed  = paths["processed"]
    eda_cfg    = cfg.get("audio_eda", {})
    n_workers  = eda_cfg.get("num_workers", 8)
    seed       = cfg.get("split", {}).get("seed", 1507)

    sft_train_path = f"{processed}/sft_train.jsonl"
    sft_val_path   = f"{processed}/sft_val.jsonl"

    log.info("")
    log.info("--- MODEL SELECTION EDA START ---")

    # ==================================================================
    # Load metadata + build worker args (lightweight — text only)
    # ==================================================================
    train_rows_meta, train_worker_args = _iter_sft_manifest(sft_train_path, audio_dirs)
    n_train = len(train_rows_meta)
    log.info(f"[MODEL-SEL] train files          {n_train:>10,}")

    # ==================================================================
    # Multiprocessed per-file analysis (TRAIN)
    # ==================================================================
    train_results: list[_MSFileResult] = []
    err_count = 0

    log.info(f"[MODEL-SEL] workers              {n_workers:>10}")
    log.info(f"[MODEL-SEL] analysing train set ...")

    with mp.Pool(
        processes=n_workers,
        initializer=_ms_worker_init,
        initargs=(eda_cfg,),
    ) as pool:
        with tqdm(pool.imap(_ms_analyse_file, train_worker_args, chunksize=128),
                  total=n_train, desc="[MODEL-SEL] train", unit="file",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            for idx, result in enumerate(pbar):
                if isinstance(result, str):
                    err_count += 1
                    if err_count <= 10:
                        log.warning(f"[MODEL-SEL] load error  file #{idx}")
                else:
                    train_results.append(result)

    n_ok = len(train_results)
    log.info(f"[MODEL-SEL] analysed             {n_ok:>10,}")
    if err_count:
        log.warning(f"[MODEL-SEL] load errors          {err_count:>10,}")

    if n_ok == 0:
        raise RuntimeError("Model-selection EDA: 0 files analysed successfully.")

    # ==================================================================
    # 1. SNR statistics
    # ==================================================================
    snr_vals = sorted([r.snr_db for r in train_results])
    snr_var = float(torch.tensor(snr_vals).var().item()) if len(snr_vals) > 1 else 0.0
    snr_block: dict[str, Any] = {
        "p01": _pctl(snr_vals, 1, 2),
        "p10": _pctl(snr_vals, 10, 2),
        "p25": _pctl(snr_vals, 25, 2),
        "p50": _pctl(snr_vals, 50, 2),
        "p75": _pctl(snr_vals, 75, 2),
        "p90": _pctl(snr_vals, 90, 2),
        "p99": _pctl(snr_vals, 99, 2),
        "variance": round(snr_var, 4),
        "n_files": n_ok,
    }
    # Store full percentile curve for plot
    for p in range(1, 100):
        snr_block[f"p{p:02d}"] = _pctl(snr_vals, p, 2)

    log.info("")
    log.info("[MODEL-SEL] --- SNR ---")
    log.info(f"[SNR]   p01={snr_block['p01']:.1f}  p50={snr_block['p50']:.1f}  "
             f"p99={snr_block['p99']:.1f}  var={snr_var:.2f}")

    # ==================================================================
    # 2. Noise block (same data, different framing)
    # ==================================================================
    rms_vals = sorted([r.rms for r in train_results])
    noise_block: dict[str, Any] = {
        "rms_p01": _pctl(rms_vals, 1),
        "rms_p50": _pctl(rms_vals, 50),
        "rms_p99": _pctl(rms_vals, 99),
        "snr_p01": snr_block["p01"],
        "snr_p50": snr_block["p50"],
        "snr_p99": snr_block["p99"],
        "snr_variance": snr_block["variance"],
    }

    # ==================================================================
    # 3. Reverb proxy
    # ==================================================================
    reverb_vals = sorted([r.reverb_slope for r in train_results])
    reverb_block: dict[str, Any] = {
        "slope_p01": _pctl(reverb_vals, 1),
        "slope_p10": _pctl(reverb_vals, 10),
        "slope_p25": _pctl(reverb_vals, 25),
        "slope_p50": _pctl(reverb_vals, 50),
        "slope_p75": _pctl(reverb_vals, 75),
        "slope_p90": _pctl(reverb_vals, 90),
        "slope_p99": _pctl(reverb_vals, 99),
    }
    log.info("[MODEL-SEL] --- REVERB ---")
    log.info(f"[REVERB]   slope p50={reverb_block['slope_p50']:.4f}  "
             f"p01={reverb_block['slope_p01']:.4f}  p99={reverb_block['slope_p99']:.4f}")

    # ==================================================================
    # 4. Overlap proxy
    # ==================================================================
    flatness_vals = sorted([r.spectral_flatness for r in train_results])
    zcr_var_vals = sorted([r.zcr_variance for r in train_results])
    peak_vals = sorted([r.harmonic_peak_count for r in train_results])

    # Data-adaptive thresholds: a file is "overlap-flagged" if ALL three
    # metrics exceed their corpus 75th percentile.
    flat_p75 = _pctl(flatness_vals, 75)
    zcr_p75 = _pctl(zcr_var_vals, 75)
    peak_p75 = _pctl(peak_vals, 75)

    n_overlap = 0
    for r in train_results:
        if (r.spectral_flatness > flat_p75
                and r.zcr_variance > zcr_p75
                and r.harmonic_peak_count > peak_p75):
            n_overlap += 1
    pct_overlap = round(100.0 * n_overlap / n_ok, 2)

    overlap_block: dict[str, Any] = {
        "spectral_flatness_p75_threshold": round(flat_p75, 6),
        "zcr_variance_p75_threshold": round(zcr_p75, 6),
        "harmonic_peaks_p75_threshold": round(peak_p75, 2),
        "n_flagged": n_overlap,
        "percent_flagged": pct_overlap,
    }
    log.info("[MODEL-SEL] --- OVERLAP ---")
    log.info(f"[OVERLAP]   flagged {n_overlap:,}  ({pct_overlap}% of train)")

    # ==================================================================
    # 5. Speaker mixing (intra-speaker MFCC variance)
    # ==================================================================
    log.info("[MODEL-SEL] --- SPEAKER MIXING ---")
    speaker_mfccs: dict[str, list[list[float]]] = defaultdict(list)
    for r in train_results:
        speaker_mfccs[r.child_id].append(r.mfcc_mean)

    intra_speaker_dists: dict[str, list[float]] = {}
    speaker_variances: list[float] = []
    for cid, mfcc_list in speaker_mfccs.items():
        if len(mfcc_list) < 2:
            continue
        dists = _cosine_distance_matrix(mfcc_list)
        intra_speaker_dists[cid] = dists
        var = float(torch.tensor(dists).var().item()) if len(dists) > 1 else 0.0
        speaker_variances.append(var)

    all_intra_dists: list[float] = []
    for dists in intra_speaker_dists.values():
        all_intra_dists.extend(dists)
    all_intra_dists.sort()

    speaker_mixing_block: dict[str, Any] = {
        "n_speakers_with_multiple_utts": len(speaker_variances),
        "speaker_variance_p50": _pctl(sorted(speaker_variances), 50),
        "speaker_variance_p95": _pctl(sorted(speaker_variances), 95),
        "global_intra_dist_p50": _pctl(all_intra_dists, 50),
        "global_intra_dist_p95": _pctl(all_intra_dists, 95),
        "global_intra_dist_max": round(max(all_intra_dists), 6) if all_intra_dists else 0.0,
    }
    log.info(f"[SPEAKER MIX]   intra-dist p50={speaker_mixing_block['global_intra_dist_p50']:.4f}  "
             f"p95={speaker_mixing_block['global_intra_dist_p95']:.4f}")

    # ==================================================================
    # 6. Speaking rate variance
    # ==================================================================
    speaker_rates: dict[str, list[float]] = defaultdict(list)
    for r in train_results:
        speaker_rates[r.child_id].append(r.phonemes_per_sec)

    rate_variances: list[float] = []
    for cid, rates in speaker_rates.items():
        if len(rates) >= 2:
            v = float(torch.tensor(rates).var().item())
            rate_variances.append(v)
    rate_variances.sort()

    rate_block: dict[str, Any] = {
        "n_speakers_with_multiple_utts": len(rate_variances),
        "variance_p50": _pctl(rate_variances, 50, 4),
        "variance_p95": _pctl(rate_variances, 95, 4),
    }
    log.info("[MODEL-SEL] --- SPEAKING RATE ---")
    log.info(f"[SPEAKING RATE]   var p50={rate_block['variance_p50']:.4f}  "
             f"p95={rate_block['variance_p95']:.4f}")

    # ==================================================================
    # 7. Duration extremes
    # ==================================================================
    durations = [r.duration_sec for r in train_results]
    n_ultra_short = sum(1 for d in durations if d < 0.5)
    n_short = sum(1 for d in durations if d < 0.7)
    n_long = sum(1 for d in durations if d > 8.0)
    dur_block: dict[str, Any] = {
        "pct_below_0.5s": round(100.0 * n_ultra_short / n_ok, 2),
        "pct_below_0.7s": round(100.0 * n_short / n_ok, 2),
        "pct_above_8s": round(100.0 * n_long / n_ok, 2),
        "n_below_0.5s": n_ultra_short,
        "n_below_0.7s": n_short,
        "n_above_8s": n_long,
    }
    log.info("[MODEL-SEL] --- DURATION EXTREMES ---")
    log.info(f"[DURATION]   <0.5s: {n_ultra_short:,} ({dur_block['pct_below_0.5s']}%)  "
             f"<0.7s: {n_short:,} ({dur_block['pct_below_0.7s']}%)  "
             f">8s: {n_long:,} ({dur_block['pct_above_8s']}%)")

    # ==================================================================
    # 8. Loudness variance per speaker
    # ==================================================================
    speaker_rms: dict[str, list[float]] = defaultdict(list)
    for r in train_results:
        speaker_rms[r.child_id].append(r.rms)

    loudness_variances: list[float] = []
    for cid, rms_list in speaker_rms.items():
        if len(rms_list) >= 2:
            v = float(torch.tensor(rms_list).var().item())
            loudness_variances.append(v)
    loudness_variances.sort()

    loudness_block: dict[str, Any] = {
        "n_speakers": len(loudness_variances),
        "variance_p50": _pctl(loudness_variances, 50),
        "variance_p95": _pctl(loudness_variances, 95),
        "variance_max": round(max(loudness_variances), 6) if loudness_variances else 0.0,
    }
    log.info("[MODEL-SEL] --- LOUDNESS PER SPEAKER ---")
    log.info(f"[LOUDNESS]   var p50={loudness_block['variance_p50']:.6f}  "
             f"p95={loudness_block['variance_p95']:.6f}")

    # ==================================================================
    # 9. Cross-speaker acoustic distance
    # ==================================================================
    log.info("[MODEL-SEL] --- SPEAKER DISTANCE ---")
    speaker_mfcc_means: dict[str, list[float]] = {}
    for cid, mfcc_list in speaker_mfccs.items():
        # Mean MFCC per speaker = element-wise average of all utterance MFCCs
        t = torch.tensor(mfcc_list, dtype=torch.float32)
        speaker_mfcc_means[cid] = t.mean(dim=0).tolist()

    spk_ids = sorted(speaker_mfcc_means.keys())
    n_spk = len(spk_ids)

    # Downsample if too many speakers
    rng = random.Random(seed)
    if n_spk > _SPEAKER_DISTANCE_MAX:
        log.info(f"[SPEAKER DIST]   downsampling {n_spk:,} → {_SPEAKER_DISTANCE_MAX:,} speakers")
        spk_ids = sorted(rng.sample(spk_ids, _SPEAKER_DISTANCE_MAX))

    spk_vectors = [speaker_mfcc_means[cid] for cid in spk_ids]
    cross_dists = _cosine_distance_matrix(spk_vectors)
    cross_dists.sort()

    spk_dist_block: dict[str, Any] = {
        "n_speakers": len(spk_ids),
        "n_pairs": len(cross_dists),
        "p50": _pctl(cross_dists, 50, 4),
        "p95": _pctl(cross_dists, 95, 4),
        "max": round(max(cross_dists), 4) if cross_dists else 0.0,
    }
    log.info(f"[SPEAKER DIST]   p50={spk_dist_block['p50']:.4f}  "
             f"p95={spk_dist_block['p95']:.4f}  max={spk_dist_block['max']:.4f}")

    # ==================================================================
    # 10. Age acoustic mismatch
    # ==================================================================
    log.info("[MODEL-SEL] --- AGE ACOUSTIC MISMATCH ---")
    age_centroids: dict[str, list[float]] = defaultdict(list)
    age_f0s: dict[str, list[float]] = defaultdict(list)
    for r in train_results:
        age_centroids[r.age_bucket].append(r.spectral_centroid)
        if r.f0_median > 0:
            age_f0s[r.age_bucket].append(r.f0_median)

    age_vectors: dict[str, list[float]] = {}
    for bucket in sorted(age_centroids):
        cent_mean = sum(age_centroids[bucket]) / max(len(age_centroids[bucket]), 1)
        f0_mean = sum(age_f0s.get(bucket, [0.0])) / max(len(age_f0s.get(bucket, [0.0])), 1)
        age_vectors[bucket] = [cent_mean, f0_mean]

    # Cosine distance matrix between age buckets
    age_buckets = sorted(age_vectors.keys())
    age_dist_matrix: dict[str, dict[str, float]] = {}
    for b1 in age_buckets:
        age_dist_matrix[b1] = {}
        for b2 in age_buckets:
            age_dist_matrix[b1][b2] = round(
                _cosine_distance(age_vectors[b1], age_vectors[b2]), 4
            )

    age_mismatch_block: dict[str, Any] = {
        "buckets": age_buckets,
        "per_bucket": {
            b: {
                "spectral_centroid_mean": round(
                    sum(age_centroids[b]) / max(len(age_centroids[b]), 1), 2
                ),
                "f0_mean": round(
                    sum(age_f0s.get(b, [0.0])) / max(len(age_f0s.get(b, [0.0])), 1), 2
                ),
                "n_files": len(age_centroids[b]),
            }
            for b in age_buckets
        },
        "cosine_distance_matrix": age_dist_matrix,
    }
    for b in age_buckets:
        info = age_mismatch_block["per_bucket"][b]
        log.info(f"[AGE MISMATCH]   {b:<8s}  centroid={info['spectral_centroid_mean']:.1f} Hz  "
                 f"F0={info['f0_mean']:.1f} Hz  n={info['n_files']:,}")

    # ==================================================================
    # 11. Train vs Val shift (OBSERVABILITY ONLY)
    # ==================================================================
    log.info("[MODEL-SEL] --- TRAIN vs VAL SHIFT ---")
    train_snr_p50 = snr_block["p50"]
    train_rms_p50 = _pctl(rms_vals, 50)
    train_centroid_vals = [r.spectral_centroid for r in train_results]
    train_centroid_mean = sum(train_centroid_vals) / max(len(train_centroid_vals), 1)
    train_f0_vals = [r.f0_median for r in train_results if r.f0_median > 0]
    train_f0_mean = sum(train_f0_vals) / max(len(train_f0_vals), 1)

    # Lightweight val analysis — stream with same pool but fewer files
    val_rows_meta, val_worker_args = _iter_sft_manifest(sft_val_path, audio_dirs)
    n_val = len(val_rows_meta)
    log.info(f"[TRAIN-VAL]   val files            {n_val:>10,}")
    log.info(f"[TRAIN-VAL]   analysing val set ...")

    val_results: list[_MSFileResult] = []
    val_errs = 0
    with mp.Pool(
        processes=n_workers,
        initializer=_ms_worker_init,
        initargs=(eda_cfg,),
    ) as pool:
        with tqdm(pool.imap(_ms_analyse_file, val_worker_args, chunksize=128),
                  total=n_val, desc="[MODEL-SEL] val", unit="file",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            for idx, result in enumerate(pbar):
                if isinstance(result, str):
                    val_errs += 1
                else:
                    val_results.append(result)

    val_snr_vals = sorted([r.snr_db for r in val_results])
    val_rms_vals = sorted([r.rms for r in val_results])
    val_centroid_vals = [r.spectral_centroid for r in val_results]
    val_f0_vals = [r.f0_median for r in val_results if r.f0_median > 0]

    val_snr_p50 = _pctl(val_snr_vals, 50, 2) if val_snr_vals else 0.0
    val_rms_p50 = _pctl(val_rms_vals, 50) if val_rms_vals else 0.0
    val_centroid_mean = sum(val_centroid_vals) / max(len(val_centroid_vals), 1)
    val_f0_mean = sum(val_f0_vals) / max(len(val_f0_vals), 1)

    shift_block: dict[str, Any] = {
        "train": {
            "snr_p50": round(train_snr_p50, 2),
            "rms_p50": round(train_rms_p50, 6),
            "spectral_centroid_mean": round(train_centroid_mean, 2),
            "f0_mean": round(train_f0_mean, 2),
        },
        "val": {
            "snr_p50": round(val_snr_p50, 2),
            "rms_p50": round(val_rms_p50, 6),
            "spectral_centroid_mean": round(val_centroid_mean, 2),
            "f0_mean": round(val_f0_mean, 2),
        },
        "delta": {
            "snr_p50": round(train_snr_p50 - val_snr_p50, 2),
            "rms_p50": round(train_rms_p50 - val_rms_p50, 6),
            "spectral_centroid_mean": round(train_centroid_mean - val_centroid_mean, 2),
            "f0_mean": round(train_f0_mean - val_f0_mean, 2),
        },
        "val_files_analysed": len(val_results),
        "val_load_errors": val_errs,
    }
    log.info(f"[TRAIN-VAL]   SNR  Δ={shift_block['delta']['snr_p50']:+.2f} dB")
    log.info(f"[TRAIN-VAL]   RMS  Δ={shift_block['delta']['rms_p50']:+.6f}")
    log.info(f"[TRAIN-VAL]   Centroid Δ={shift_block['delta']['spectral_centroid_mean']:+.1f} Hz")
    log.info(f"[TRAIN-VAL]   F0   Δ={shift_block['delta']['f0_mean']:+.1f} Hz")

    # ==================================================================
    # 12. WEBS — Weighted Evidence-Based Scoring
    # ==================================================================
    #
    # Research-grade model fitness using:
    #   (a) Continuous corpus-adaptive severity per challenge dimension
    #   (b) Literature-derived robustness matrix  r_{m,d} = 1 - D_m/max(D)
    #   (c) Overlap × reverb interference interaction
    #   (d) Phonetic resolution dimension (CTC-specific)
    #   (e) Severity-normalised importance weights
    #   (f) Confidence estimate from score margin
    #   (g) Sensitivity analysis (±0.1 robustness perturbation)
    #
    # NOTE: Scores are RELATIVE FITNESS for this corpus only.
    #       They do NOT generalise to other datasets.
    log.info("[MODEL-SEL] --- WEBS SCORING ---")

    # ---- Robustness matrix (rank-based) ----
    # r_{m,d} ∈ {0.2, 0.5, 0.8}:  ordinal ranking per dimension.
    # Rank order is stable across papers; absolute WER deltas are not
    # (they depend on dataset, LM, decoding — not the SSL encoder).
    # Mapping:  best → 0.8,  middle → 0.5,  worst → 0.2.
    #
    # Ranking sources (order only — no absolute numbers used):
    #   noise:          WavLM > HuBERT > wav2vec2   (SUPERB, Chen et al. 2022)
    #   reverb:         WavLM > HuBERT > wav2vec2   (REVERB challenge cross-eval)
    #   interference:   WavLM > HuBERT > wav2vec2   (LibriCSS, CHiME-6)
    #   speaker_var:    HuBERT > WavLM > wav2vec2   (SUPERB SID/ASV)
    #   temporal:       WavLM > HuBERT > wav2vec2   (SUPERB PR, discrete-unit alignment)
    #   phonetic:       WavLM > HuBERT > wav2vec2   (SUPERB PR, Hsu et al. 2021)
    robustness: dict[str, dict[str, float]] = {
        "noise":                {"wav2vec2": 0.2, "hubert": 0.5, "wavlm": 0.8},
        "reverb":               {"wav2vec2": 0.2, "hubert": 0.5, "wavlm": 0.8},
        "interference":         {"wav2vec2": 0.2, "hubert": 0.5, "wavlm": 0.8},
        "speaker_variability":  {"wav2vec2": 0.2, "hubert": 0.8, "wavlm": 0.5},
        "temporal":             {"wav2vec2": 0.2, "hubert": 0.5, "wavlm": 0.8},
        "phonetic":             {"wav2vec2": 0.2, "hubert": 0.5, "wavlm": 0.8},
    }

    models = ["wav2vec2", "hubert", "wavlm"]
    dimensions = list(robustness.keys())

    # ---- Phase 1: Corpus-anchored severity ----
    # s_d ∈ [0,1].  Logistic curve with DATA-DRIVEN midpoints:
    #   x0 = corpus p50 of the difficulty metric
    #   k  = 4 / (p90 − p10)   → maps corpus spread to ~[0.12, 0.88]
    # No fake literature constants — fully reproducible from data.

    severities: dict[str, float] = {}

    # 1a. Noise
    # Metric: inverted SNR (higher = noisier).  40 − SNR so axis points up.
    # No IQR shift — that's a corpus constant which distorts per-file
    # spread and therefore the logistic anchors.  If variability matters
    # it should be a separate dimension, not a constant addend.
    snr_p50_val = snr_block["p50"]
    snr_difficulties = sorted([40.0 - r.snr_db for r in train_results])
    noise_difficulty = 40.0 - snr_p50_val
    noise_x0 = _pctl(snr_difficulties, 50)
    noise_range = _pctl(snr_difficulties, 90) - _pctl(snr_difficulties, 10)
    noise_k = 4.0 / max(noise_range, _EPS)
    severities["noise"] = round(_logistic_severity(noise_difficulty, x0=noise_x0, k=noise_k), 4)

    # 1b. Reverb
    # Metric: slope IQR normalised by |slope median|.
    reverb_iqr = abs(reverb_block["slope_p75"] - reverb_block["slope_p25"])
    reverb_med_abs = abs(reverb_block["slope_p50"])
    reverb_difficulty = reverb_iqr / max(reverb_med_abs, _EPS)
    # Per-file reverb difficulty for anchoring
    reverb_diffs = sorted([
        abs(r.reverb_slope - reverb_block["slope_p50"]) / max(reverb_med_abs, _EPS)
        for r in train_results
    ])
    reverb_x0 = _pctl(reverb_diffs, 50)
    reverb_range = _pctl(reverb_diffs, 90) - _pctl(reverb_diffs, 10)
    reverb_k = 4.0 / max(reverb_range, _EPS)
    severities["reverb"] = round(_logistic_severity(reverb_difficulty, x0=reverb_x0, k=reverb_k), 4)

    # 1c. Overlap (raw — before interference interaction)
    # Metric: percentage of files flagged as overlapping.
    # Corpus-anchored: build a per-file binary overlap vector and derive
    # x0 and k from its distribution — no hand-set constants.
    overlap_per_file = sorted([
        1.0 if (r.spectral_flatness > flat_p75
                and r.zcr_variance > zcr_p75
                and r.harmonic_peak_count > peak_p75)
        else 0.0
        for r in train_results
    ])
    overlap_x0 = _pctl(overlap_per_file, 50)
    overlap_range = _pctl(overlap_per_file, 90) - _pctl(overlap_per_file, 10)
    overlap_k = 4.0 / max(overlap_range, _EPS) if overlap_range > _EPS else 0.4
    s_overlap_raw = _logistic_severity(pct_overlap, x0=overlap_x0 * 100.0, k=overlap_k)

    # 1d. Interference — joint overlap × reverb interaction
    # s_interference = 1 − (1 − s_overlap)(1 − s_reverb)
    # Captures the compounding effect: reverb + overlap is worse than either alone.
    severities["interference"] = round(
        1.0 - (1.0 - s_overlap_raw) * (1.0 - severities["reverb"]), 4
    )

    # 1e. Speaker variability
    # Metric: cross-speaker cosine distance p95.
    # Anchor k from the FULL cross_dists distribution (already sorted).
    spk_dist_p95 = spk_dist_block["p95"]
    spk_x0 = _pctl(cross_dists, 50, 4)
    spk_p90 = _pctl(cross_dists, 90, 4)
    spk_p10 = _pctl(cross_dists, 10, 4)
    spk_range = spk_p90 - spk_p10
    spk_k = 4.0 / max(spk_range, _EPS)
    severities["speaker_variability"] = round(
        _logistic_severity(spk_dist_p95, x0=spk_x0, k=spk_k), 4
    )

    # 1f. Temporal (duration extremes — split ultra-short bucket)
    # CTC failure is steepest < 0.5 s; 0.5–0.7 s is mild.
    # Weighted: ultra-short × 1.0 + mild-short × 0.3 + long × 0.5
    pct_ultra = dur_block["pct_below_0.5s"]
    pct_mild  = dur_block["pct_below_0.7s"] - dur_block["pct_below_0.5s"]
    pct_long  = dur_block["pct_above_8s"]
    pct_extreme = 1.0 * pct_ultra + 0.3 * pct_mild + 0.5 * pct_long
    # Anchor at corpus data: build per-file bucket indicator for percentiles
    dur_per_file = []
    for d in durations:
        if d < 0.5:
            dur_per_file.append(1.0)
        elif d < 0.7:
            dur_per_file.append(0.3)
        elif d > 8.0:
            dur_per_file.append(0.5)
        else:
            dur_per_file.append(0.0)
    dur_per_file.sort()
    temporal_x0 = _pctl(dur_per_file, 50)
    temporal_range = _pctl(dur_per_file, 90) - _pctl(dur_per_file, 10)
    temporal_k = 4.0 / max(temporal_range, _EPS) if temporal_range > _EPS else 0.20
    severities["temporal"] = round(_logistic_severity(pct_extreme, x0=temporal_x0, k=temporal_k), 4)

    # 1g. Phonetic resolution (CTC-specific)
    # Components: phoneme-frequency entropy + rare-tail mass + rate variance.
    # Inventory size is NOT included — CTC cares about token frequency
    # distribution, not raw inventory count.
    tc_path = reports / "training_controls.json"
    phon_total_tokens = 1
    phon_rare_tail_mass = 0.0
    phon_entropy = 0.0           # bits
    if tc_path.exists():
        try:
            tc_raw = tc_path.read_text(encoding="utf-8")
            tc_data = loads(tc_raw)
            pf = tc_data.get("phoneme_frequency", {})
            phon_total_tokens = max(pf.get("total_tokens", 1), 1)
            # Entropy of the token frequency distribution (bits)
            freq_dict = pf.get("frequency", {})
            if freq_dict:
                phon_entropy = 0.0
                for _pinfo in freq_dict.values():
                    p = _pinfo.get("count", 0) / phon_total_tokens
                    if p > 0:
                        phon_entropy -= p * math.log2(p)
            # Rare tail: sum of tokens from phonemes with < 0.5% frequency.
            rare_count = 0
            for _pinfo in freq_dict.values():
                if _pinfo.get("percent", 100.0) < 0.5:
                    rare_count += _pinfo.get("count", 0)
            phon_rare_tail_mass = rare_count / phon_total_tokens
        except Exception:
            log.warning("[WEBS] Could not read training_controls.json — using defaults")

    # Normalised components (0→easy, 1→challenging at moderate level)
    # Entropy: max for uniform = log2(N); higher entropy → harder for CTC.
    # Normalise by log2(80) ≈ 6.32 (generous upper bound for phoneme sets).
    phon_entropy_norm = phon_entropy / 6.32
    phon_rate_norm = rate_block["variance_p95"] / 10.0             # 10→1
    phon_rare_norm = phon_rare_tail_mass / 0.02                    # 2%→1
    # Composite:  α=0.45 (entropy), β=0.30 (rate variance), γ=0.25 (rare tail)
    phon_composite = 0.45 * phon_entropy_norm + 0.30 * phon_rate_norm + 0.25 * phon_rare_norm
    # Corpus-anchored midpoint: build per-file phonetic difficulty from the
    # one component that varies per file (rate variance).  Entropy and rare-
    # tail are corpus constants and correctly stay constant.
    per_file_rate = sorted([r.f0_median for r in train_results if r.f0_median > 0])
    if len(per_file_rate) >= 10:
        # Use per-speaker rate variance as proxy — already computed in rate_variances
        rate_sorted = sorted(rate_variances) if rate_variances else [0.0]
        rate_p50 = _pctl(rate_sorted, 50)
        rate_p90 = _pctl(rate_sorted, 90)
        rate_p10 = _pctl(rate_sorted, 10)
        # Map the same way as phon_rate_norm
        phon_x0 = 0.45 * phon_entropy_norm + 0.30 * (rate_p50 / 10.0) + 0.25 * phon_rare_norm
        phon_range = 0.30 * ((rate_p90 - rate_p10) / 10.0)   # only rate varies
        phon_k = 4.0 / max(phon_range, _EPS) if phon_range > _EPS else 4.0
    else:
        phon_x0 = 0.5 * phon_composite
        phon_k = 4.0
    severities["phonetic"] = round(_logistic_severity(phon_composite, x0=phon_x0, k=phon_k), 4)

    # Log the severity profile
    log.info("")
    for dim in dimensions:
        bar_len = int(severities[dim] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        log.info(f"[SEVERITY]   {dim:<25s} {severities[dim]:.4f}  {bar}")

    # ---- Phase 2: Blended weights ----
    # 50% base importance + 50% severity-weighted.
    # Pure severity-normalisation (w_d·s_d) can suppress real model
    # strengths when a dimension has low corpus severity but the best
    # model's advantage is precisely on that dimension.  The 50/50
    # blend ensures base importance never vanishes.
    base_weights: dict[str, float] = {
        "noise": 0.25,
        "reverb": 0.10,
        "interference": 0.15,
        "speaker_variability": 0.20,
        "temporal": 0.10,
        "phonetic": 0.20,
    }
    sev_ws = {d: base_weights[d] * severities[d] for d in dimensions}
    sev_sum = sum(sev_ws.values())
    if sev_sum > _EPS:
        sev_norm = {d: v / sev_sum for d, v in sev_ws.items()}
    else:
        sev_norm = {d: 1.0 / len(dimensions) for d in dimensions}
    # Blend: 50% base (always present) + 50% severity-adjusted
    blended = {d: 0.5 * base_weights[d] + 0.5 * sev_norm[d] for d in dimensions}
    bl_sum = sum(blended.values())
    norm_weights = {d: round(v / bl_sum, 4) for d, v in blended.items()}

    log.info("")
    log.info("[WEIGHTS]   base → normalised")
    for d in dimensions:
        log.info(f"[WEIGHTS]   {d:<25s}  {base_weights[d]:.2f} → {norm_weights[d]:.4f}")

    # ---- Phase 3: Fitness scores ----
    # fitness_m = Σ_d  w'_d · [1 − s_d · (1 − r_{m,d})]
    # Penalty for model m on dimension d = w'_d · s_d · (1 − r_m,d)
    fitness: dict[str, float] = {}
    penalty_breakdown: dict[str, dict[str, float]] = {}

    for m in models:
        total = 0.0
        penalties: dict[str, float] = {}
        for d in dimensions:
            r = robustness[d][m]
            s = severities[d]
            w = norm_weights[d]
            total += w * (1.0 - s * (1.0 - r))
            penalties[d] = round(w * s * (1.0 - r), 4)
        fitness[m] = total
        penalty_breakdown[m] = penalties

    # Map fitness ∈ [0,1] → score ∈ [0, 10]
    scores: dict[str, float] = {
        m: round(max(0.0, min(10.0, fitness[m] * 10.0)), 1)
        for m in models
    }
    recommended = max(scores, key=scores.get)

    # ---- Phase 4: Confidence (Issue 7) ----
    sorted_sc = sorted(scores.values(), reverse=True)
    margin = sorted_sc[0] - sorted_sc[1] if len(sorted_sc) >= 2 else 0.0
    sc_mean = sum(sorted_sc) / max(len(sorted_sc), 1)
    sc_std = (sum((s - sc_mean) ** 2 for s in sorted_sc) / max(len(sorted_sc), 1)) ** 0.5
    conf_ratio = margin / max(sc_std, _EPS)
    if conf_ratio > 2.0:
        conf_label = "HIGH"
    elif conf_ratio > 1.0:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    # ---- Phase 5: Sensitivity analysis ----
    # Perturb each r_{m,d} by ±0.1.  If the top-ranked model changes
    # under ANY single perturbation, the ranking is FRAGILE.
    ranking_stable = True
    for d in dimensions:
        for m in models:
            for delta in (-0.1, 0.1):
                perturbed_fitness: dict[str, float] = {}
                for m2 in models:
                    f2 = 0.0
                    for d2 in dimensions:
                        r2 = robustness[d2][m2]
                        if d2 == d and m2 == m:
                            r2 = max(0.0, min(1.0, r2 + delta))
                        f2 += norm_weights[d2] * (1.0 - severities[d2] * (1.0 - r2))
                    perturbed_fitness[m2] = f2
                perturbed_top = max(perturbed_fitness, key=perturbed_fitness.get)
                if perturbed_top != recommended:
                    ranking_stable = False

    # ---- Logging ----
    log.info("")
    for m in models:
        log.info(f"[SCORES]   {m:<12s}  {scores[m]:>5.1f} / 10")
    log.info(f"[SCORES]   → recommended: {recommended.upper()}")
    log.info(f"[SCORES]   confidence:   {conf_label}  "
             f"(margin={margin:.1f}, ratio={conf_ratio:.2f})")
    log.info(f"[SCORES]   ranking:      {'STABLE' if ranking_stable else 'FRAGILE'} "
             f"under ±0.1 robustness perturbation")

    log.info("")
    log.info("[MODEL-SEL] --- PENALTY BREAKDOWN ---")
    for m in models:
        parts = "  ".join(f"{d}={penalty_breakdown[m][d]:.4f}" for d in dimensions)
        log.info(f"[PENALTY]   {m:<12s}  {parts}")

    # Explanations — sorted by severity, skip negligible dimensions (< 0.05)
    explanations: list[str] = []
    for d in sorted(dimensions, key=lambda x: severities[x], reverse=True):
        if severities[d] < 0.05:
            continue
        best_m = max(models, key=lambda m, _d=d: robustness[_d][m])
        worst_m = min(models, key=lambda m, _d=d: robustness[_d][m])
        explanations.append(
            f"{d} (severity={severities[d]:.2f}, weight={norm_weights[d]:.4f}): "
            f"{best_m} most robust (r={robustness[d][best_m]:.3f}), "
            f"{worst_m} most vulnerable (r={robustness[d][worst_m]:.3f})."
        )

    log.info("")
    for i, exp in enumerate(explanations, 1):
        log.info(f"[SCORES]   {i}. {exp}")

    log.info("")
    log.info("[SCORES]   [NOTE] Scores are relative fitness for THIS corpus only. "
             "They do NOT generalise to other datasets.")

    # ---- Assemble signals block ----
    severity_details: dict[str, Any] = {
        "noise": {
            "snr_p50": snr_p50_val,
            "difficulty_metric": round(noise_difficulty, 2),
        },
        "reverb": {
            "slope_iqr": round(reverb_iqr, 4),
            "slope_median_abs": round(reverb_med_abs, 4),
            "difficulty_metric": round(reverb_difficulty, 4),
        },
        "interference": {
            "pct_overlap": round(pct_overlap, 2),
            "s_overlap_raw": round(s_overlap_raw, 4),
            "s_reverb": severities["reverb"],
            "interaction": "1 - (1 - s_overlap)(1 - s_reverb)",
        },
        "speaker_variability": {
            "cross_spk_dist_p95": spk_dist_p95,
        },
        "temporal": {
            "pct_below_0.5s": dur_block["pct_below_0.5s"],
            "pct_below_0.7s": dur_block["pct_below_0.7s"],
            "pct_above_8s": dur_block["pct_above_8s"],
            "pct_extreme_weighted": round(pct_extreme, 2),
            "weights": "ultra(<0.5s)×1.0 + mild(0.5-0.7s)×0.3 + long(>8s)×0.5",
        },
        "phonetic": {
            "entropy_bits": round(phon_entropy, 4),
            "entropy_normalised": round(phon_entropy_norm, 4),
            "rare_tail_mass": round(phon_rare_tail_mass, 4),
            "rate_variance_p95": rate_block["variance_p95"],
            "composite": round(phon_composite, 4),
        },
    }

    model_signals_block: dict[str, Any] = {
        "method": "WEBS",
        "method_full": "Weighted Evidence-Based Scoring",
        "calibration": "rank-based robustness, corpus-anchored severity, 50/50 weight blend",
        "note": (
            "Scores are relative fitness for this corpus only. "
            "They do NOT generalise to other datasets."
        ),
        "severities": {d: severities[d] for d in dimensions},
        "severity_details": severity_details,
        "robustness_matrix": robustness,
        "base_weights": base_weights,
        "normalized_weights": norm_weights,
        "penalty_breakdown": penalty_breakdown,
        "fitness_raw": {m: round(fitness[m], 4) for m in models},
        "wav2vec2_score": scores["wav2vec2"],
        "hubert_score": scores["hubert"],
        "wavlm_score": scores["wavlm"],
        "recommended_model": recommended,
        "confidence": {
            "label": conf_label,
            "margin": round(margin, 2),
            "ratio": round(conf_ratio, 2),
        },
        "sensitivity": {
            "ranking_stable": ranking_stable,
            "perturbation_delta": 0.1,
        },
        "explanations": explanations,
    }

    # ==================================================================
    # Assemble full report
    # ==================================================================
    report: dict[str, Any] = {
        "noise": noise_block,
        "reverb": reverb_block,
        "overlap": overlap_block,
        "speaker_mixing": speaker_mixing_block,
        "snr": snr_block,
        "speaking_rate_variance": rate_block,
        "duration_extremes": dur_block,
        "loudness_per_speaker": loudness_block,
        "speaker_acoustic_distance": spk_dist_block,
        "age_acoustic_mismatch": age_mismatch_block,
        "train_val_shift": shift_block,
        "model_recommendation_signals": model_signals_block,
    }

    # ---- Write JSON ----
    out_path = reports / "model_selection_eda.json"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(dumps(report, indent=2))
    log.info(f"[MODEL-SEL] saved → {out_path}")

    # ---- Generate plots ----
    plot_dir = Path(cfg["data"]) / "plots"
    saved_plots = _save_plots(report, plot_dir)
    if saved_plots:
        log.info(f"[MODEL-SEL] plots → {plot_dir}  ({len(saved_plots)} files)")
        for fname in saved_plots:
            log.info(f"[MODEL-SEL]   {fname}")

    log.info("--- MODEL SELECTION EDA END ---")

    return report
