"""
DataLoader + Collator throughput benchmark — CPU/RAM profiling.
===============================================================

Measures **samples/sec** and **batches/sec** for the full collator
pipeline (load → mono → resample → RMS normalise → tokenise → pad)
with configurable worker count and batch size.

Usage (standalone)
------------------
    cd src
    python -m utils.bench_dataloader [--batches N] [--workers W] [-a]

Usage (via pipeline)
--------------------
    python pipeline.py --benchmark [--workers W] [--batch-size N] [-a]

Output: structured log lines with timing, throughput, memory stats,
and per-worker averages.  Results are also written to
``data/logs/bench_dataloader_<timestamp>.log``.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader, SequentialSampler

# ---------------------------------------------------------------------------
# Project imports — resolve src/ on sys.path
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config.config import load_config          # noqa: E402
from trainer.data_collator import SFTCollator   # noqa: E402
from trainer.dataset import SFTDataset          # noqa: E402
from transformers import Wav2Vec2CTCTokenizer   # noqa: E402

log = logging.getLogger("bench_dataloader")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Resident Set Size of current process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def _setup_bench_log(log_dir: str) -> Path | None:
    """Add a file handler for the benchmark log — only in standalone mode.

    If the root logger already has a FileHandler (i.e. pipeline.py set
    up logging), we skip creating a second file — everything goes to
    the pipeline log instead.  Returns the log file path, or None when
    piggybacking on existing handlers.
    """
    root = logging.getLogger()
    has_file_handler = any(
        isinstance(h, logging.FileHandler) for h in root.handlers
    )
    if has_file_handler:
        # Pipeline already logging to file — no duplicate
        return None

    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = d / f"bench_dataloader_{stamp}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    bench = logging.getLogger("bench_dataloader")
    bench.addHandler(fh)

    return log_file


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def run_benchmark(
    cfg: dict,
    *,
    split: str = "train",
    num_batches: int | None = 100,
    batch_size: int = 4,
    num_workers: int = 1,
    all_data: bool = False,
) -> dict[str, float]:
    """Time the DataLoader + Collator pipeline.

    Parameters
    ----------
    cfg : dict
        Global config (from ``load_config``).
    split : str
        ``"train"`` or ``"val"``.
    num_batches : int | None
        How many batches to iterate.  Ignored when *all_data* is True.
    batch_size : int
        Physical batch size.
    num_workers : int
        DataLoader worker processes.
    all_data : bool
        If True, iterate the **entire** dataset (overrides *num_batches*).

    Returns
    -------
    dict[str, float]
        Timing and throughput statistics.
    """
    sft = cfg["sft"]
    paths = cfg["paths"]

    # ---- Benchmark log file ----
    bench_log = _setup_bench_log(paths["logs"])
    if bench_log:
        log.info("Benchmark log: %s", bench_log)
    else:
        log.info("Benchmark output → pipeline log (no separate file)")

    # ---- Dataset ----
    manifest = f"{paths['processed']}/sft_{split}.jsonl"
    audio_dirs = paths["audio_dirs"]
    ds = SFTDataset(manifest, audio_dirs, split=split)
    log.info("Dataset: %s rows (%s split)", f"{len(ds):,}", split)

    # ---- Tokenizer + Collator ----
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(paths["tokenizer"])
    collator = SFTCollator(
        tokenizer,
        target_sr=16_000,
        
    )

    # ---- DataLoader ----
    kw: dict = {}
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=SequentialSampler(ds),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        **kw,
    )

    max_batches = math.ceil(len(ds) / batch_size)
    if all_data:
        total_batches = max_batches
    else:
        total_batches = min(num_batches or 100, max_batches)

    total_samples = min(total_batches * batch_size, len(ds))

    log.info(
        "Config: %d batches × %d batch_size = %d samples, "
        "%d worker(s)%s",
        total_batches, batch_size, total_samples,
        num_workers, " (FULL DATASET)" if all_data else "",
    )

    # ---- Warmup (2 batches — populate resampler cache) ----
    warmup_n = min(2, total_batches)
    log.info("Warming up (%d batches)...", warmup_n)
    it = iter(loader)
    for _ in range(warmup_n):
        _ = next(it)
    del it
    gc.collect()

    # ---- Timed run ----
    log.info("Starting timed run...")
    rss_start = _rss_mb()

    batch_times: list[float] = []
    waveform_lengths: list[int] = []
    actual_samples = 0
    report_interval = max(total_batches // 4, 1)

    loader_iter = iter(loader)
    for i in range(total_batches):
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        t1 = time.perf_counter()

        batch_times.append(t1 - t0)
        B = batch["input_values"].size(0)
        actual_samples += B
        waveform_lengths.extend(batch["input_lengths"].tolist())

        # ---- Output validation (first batch only) ----
        if i == 0:
            iv = batch["input_values"]
            am = batch["attention_mask"]
            il = batch["input_lengths"]

            assert iv.ndim == 2, (
                f"input_values not 2-D (B,T): shape={iv.shape}"
            )
            assert iv.dtype == torch.float32, (
                f"input_values dtype wrong: {iv.dtype}"
            )
            assert am.dtype == torch.long, (
                f"attention_mask dtype not long: {am.dtype}"
            )
            # Verify 16kHz: frame count should be ~duration × 16000
            for s in range(B):
                frames = il[s].item()
                dur_sec = frames / 16_000
                assert 0.01 < dur_sec < 60, (
                    f"Sample {s}: {frames} frames = {dur_sec:.2f}s "
                    f"— unlikely for 16kHz mono"
                )

            log.info(
                "  [VALIDATE] First batch OK — shape=%s, dtype=%s, "
                "mask_dtype=%s, frames=[%d..%d]",
                list(iv.shape), iv.dtype, am.dtype,
                il.min().item(), il.max().item(),
            )

        if (i + 1) % report_interval == 0:
            running_sps = actual_samples / sum(batch_times)
            log.info(
                "  batch %d/%d — %.1f samples/sec (running)",
                i + 1, total_batches, running_sps,
            )

    rss_end = _rss_mb()
    total_time = sum(batch_times)
    n_batches_done = len(batch_times)

    # ---- Compute stats ----
    samples_per_sec = actual_samples / total_time
    batches_per_sec = n_batches_done / total_time
    avg_batch_ms = (total_time / n_batches_done) * 1000
    avg_sample_ms = (total_time / actual_samples) * 1000
    per_worker_sps = samples_per_sec / max(num_workers, 1)

    avg_wav_frames = sum(waveform_lengths) / len(waveform_lengths)
    avg_wav_sec = avg_wav_frames / 16_000

    results = {
        "samples_per_sec": samples_per_sec,
        "batches_per_sec": batches_per_sec,
        "avg_batch_ms": avg_batch_ms,
        "avg_sample_ms": avg_sample_ms,
        "per_worker_sps": per_worker_sps,
        "rss_start_mb": rss_start,
        "rss_end_mb": rss_end,
        "total_time_sec": total_time,
        "total_samples": actual_samples,
        "total_batches": n_batches_done,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "avg_waveform_sec": avg_wav_sec,
    }

    # ---- Report ----
    log.info("=" * 64)
    log.info("BENCHMARK RESULTS  (%d worker(s), batch_size=%d)", num_workers, batch_size)
    log.info("=" * 64)
    log.info("  Total time      : %.2f sec (%d batches, %d samples)",
             total_time, n_batches_done, actual_samples)
    log.info("  Samples/sec     : %.2f", samples_per_sec)
    log.info("  Batches/sec     : %.2f", batches_per_sec)
    log.info("  Avg batch       : %.1f ms", avg_batch_ms)
    log.info("  Avg sample      : %.1f ms", avg_sample_ms)
    log.info("  Per-worker avg  : %.2f samples/sec", per_worker_sps)
    log.info("  Avg waveform    : %.2f sec (%.0f frames @ 16kHz)",
             avg_wav_sec, avg_wav_frames)
    log.info("  RSS memory      : %.0f MB → %.0f MB (Δ %.0f MB)",
             rss_start, rss_end, rss_end - rss_start)
    log.info("-" * 64)

    train_workers = sft["num_workers"]
    extrapolated = per_worker_sps * train_workers
    log.info(
        "  EXTRAPOLATION   : %d training workers × %.1f sps/worker "
        "≈ %.0f samples/sec",
        train_workers, per_worker_sps, extrapolated,
    )

    # GPU step comparison
    gpu_step_ms = 300  # conservative estimate for fwd+bwd+optim
    gpu_needs_sps = batch_size / (gpu_step_ms / 1000)
    bottleneck = "GPU" if extrapolated > gpu_needs_sps else "DATA"
    log.info(
        "  GPU needs       : ~%.0f samples/sec (batch=%d, ~%dms/step)",
        gpu_needs_sps, batch_size, gpu_step_ms,
    )
    log.info("  Bottleneck      : %s", bottleneck)
    log.info("=" * 64)
    if bench_log:
        log.info("Benchmark log saved: %s", bench_log)

    return results


# ---------------------------------------------------------------------------
# CLI (standalone execution)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark DataLoader + Collator throughput",
    )
    parser.add_argument(
        "--batches", type=int, default=100,
        help="Number of batches to time (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Physical batch size (default: 4)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of DataLoader workers (default: 1)",
    )
    parser.add_argument(
        "-a", "--all", action="store_true", dest="all_data",
        help="Benchmark the entire dataset (overrides --batches)",
    )
    parser.add_argument(
        "--split", choices=["train", "val"], default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (auto-detected if omitted)",
    )
    args = parser.parse_args()

    # ---- Locate config ----
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    if not config_path.is_file():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    # ---- Console logging (standalone mode only) ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Config: %s", config_path)
    cfg = load_config(config_path)

    run_benchmark(
        cfg,
        split=args.split,
        num_batches=args.batches,
        batch_size=args.batch_size,
        num_workers=args.workers,
        all_data=args.all_data,
    )


if __name__ == "__main__":
    main()
