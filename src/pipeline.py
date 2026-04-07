from __future__ import annotations

import argparse
import importlib
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import os

from dotenv import load_dotenv

# Load .env from project root (silently no-op if missing)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from config import load_config
from etl import EDAProcessor, AudioChecker, DataSplitter, run_model_selection_eda
from tokenizer import Tokenizer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root — one directory above src/
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC  = Path(__file__).resolve().parent


def _configure_warnings() -> None:
    """Suppress known non-actionable third-party warnings in logs."""
    from utils import configure_warnings
    configure_warnings()


# ---------------------------------------------------------------------------
# Logging — one file per run, shared by every module
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str) -> Path:
    """Configure root logger: file + console, single format."""
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = d / f"309_log_{stamp}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)

    return log_file


# ---------------------------------------------------------------------------
# Config logging — dump active config values to log for reproducibility
# ---------------------------------------------------------------------------

_LOG_SECTIONS = ("eda", "hf_sft", "nst")


def _log_config(cfg: dict) -> None:
    """Log key config sections so every run is fully traceable in logs."""
    log.info("--- ACTIVE CONFIG ---")
    log.info(f"[CONFIG] data = {cfg.get('data')}")
    log.info(f"[CONFIG] datasets = {cfg.get('datasets')}")
    for section in _LOG_SECTIONS:
        sec = cfg.get(section)
        if sec is None:
            continue
        log.info(f"[CONFIG] [{section}]")
        for k, v in sec.items():
            if isinstance(v, dict):
                log.info(f"[CONFIG]   {k}:")
                for sk, sv in v.items():
                    log.info(f"[CONFIG]     {sk}: {sv}")
            else:
                log.info(f"[CONFIG]   {k}: {v}")
    log.info("--- ACTIVE CONFIG END ---")


# ---------------------------------------------------------------------------
# Health check — verify environment before running anything
# ---------------------------------------------------------------------------

_REQUIRED_PACKAGES = [
    "torch", "torchaudio", "transformers", "yaml", "tqdm",
]


def health_check(cfg: dict) -> None:
    """Verify dependencies, raw data, and config before pipeline runs."""
    ok = True

    # ---- Python packages ----
    log.info("--- HEALTH CHECK ---")
    log.info("[HEALTH] checking dependencies ...")
    for pkg in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            log.info(f"[HEALTH]   ✓  {pkg}")
        except ImportError:
            log.warning(f"[HEALTH]   ✗  {pkg}  NOT INSTALLED")
            ok = False

    # torchcodec — may be needed as torchaudio backend
    try:
        importlib.import_module("torchcodec")
        log.info("[HEALTH]   ✓  torchcodec")
    except (ImportError, RuntimeError):
        log.info("[HEALTH]   ⚠  torchcodec  not available (optional backend — install FFmpeg or ignore)")

    # orjson — fast JSON backend
    from utils import _JSON_BACKEND
    tag = "✓" if _JSON_BACKEND == "orjson" else "⚠"
    log.info(f"[HEALTH]   {tag}  json backend: {_JSON_BACKEND}")

    # ---- Config sanity ----
    log.info("[HEALTH] checking config ...")
    config_path = SRC / "config" / "config.yaml"
    if config_path.exists():
        log.info(f"[HEALTH]   ✓  {config_path.name}")
    else:
        log.warning(f"[HEALTH]   ✗  {config_path.name}  NOT FOUND")
        ok = False

    # ---- Raw data directories ----
    log.info("[HEALTH] checking raw data ...")
    paths = cfg["paths"]
    raw = Path(paths["raw"])
    if raw.is_dir():
        log.info(f"[HEALTH]   ✓  {raw}")
    else:
        log.warning(f"[HEALTH]   ✗  {raw}  NOT FOUND")
        ok = False

    # ---- Per-dataset: transcript + audio dir ----
    for key in cfg["datasets"]:
        ts = Path(paths["datasets"][key])
        ad = Path(paths["audio_dirs"][key])

        if ts.is_file():
            # Count rows
            n = sum(1 for _ in open(ts, encoding="utf-8"))
            log.info(f"[HEALTH]   ✓  DS{key} transcript  {n:,} rows")
        else:
            log.warning(f"[HEALTH]   ✗  DS{key} transcript  {ts}  NOT FOUND")
            ok = False

        if ad.is_dir():
            n_flac = sum(1 for _ in ad.glob("*.flac"))
            log.info(f"[HEALTH]   ✓  DS{key} audio dir   {n_flac:,} .flac files")
        else:
            log.warning(f"[HEALTH]   ✗  DS{key} audio dir   {ad}  NOT FOUND")
            ok = False

    if ok:
        log.info("[HEALTH] ✓  all checks passed")
    else:
        log.warning("[HEALTH] ✗  some checks FAILED — see above")
    log.info("--- HEALTH CHECK END ---")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def clear_checkpoints(cfg: dict) -> None:
    """Delete the sft_checkpoints/ directory to free disk space."""
    ckpt_dir = Path(cfg["paths"]["models"]) / "sft_checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        log.info("sft_checkpoints/  deleted")
    else:
        log.info("sft_checkpoints/  (not found, nothing to delete)")


def delete_generated(cfg: dict) -> None:
    """Remove all generated dirs (processed, models, reports, logs, plots) and __pycache__ trees."""
    paths = cfg["paths"]
    data_root = Path(cfg["data"])

    dirs = [
        Path(paths["processed"]),
        Path(paths["models"]),
        Path(paths["reports"]),
        Path(paths["logs"]),
        Path(paths["plots"]),
    ]
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
            log.info(f"{d.relative_to(data_root)}/  deleted")
        else:
            log.info(f"{d.relative_to(data_root)}/  (not found, skipping)")

    removed = 0
    for pattern in ("__pycache__", ".pytest_cache"):
        for cache in ROOT.rglob(pattern):
            if "venv" in cache.parts or ".git" in cache.parts:
                continue
            shutil.rmtree(cache)
            removed += 1
    log.info(f"__pycache__ + .pytest_cache  ({removed} dirs removed)")


def clear_cache() -> None:
    """Remove all __pycache__ and .pytest_cache trees (excluding venv and .git)."""
    removed = 0
    for pattern in ("__pycache__", ".pytest_cache"):
        for cache in ROOT.rglob(pattern):
            if "venv" in cache.parts or ".git" in cache.parts:
                continue
            shutil.rmtree(cache)
            removed += 1
    log.info(f"__pycache__ + .pytest_cache  ({removed} dirs removed)")


def run_etl(cfg: dict) -> None:
    """Run all ETL stages: audio check → sanitise → audio EDA → tokenizer → split."""
    ac_results = {}
    for key in cfg["datasets"]:
        ac_results[key] = AudioChecker(cfg, ds_key=key).run()
    grand_hrs = sum(r["total_hours"] for r in ac_results.values())
    log.info("--- AUDIO CHECK SUMMARY ---")
    for k, r in ac_results.items():
        log.info(f"DS{k}  {r['n_ok']:>10,} OK  {r['total_hours']:>8.2f} hrs")
    log.info(f"ALL  {sum(r['n_ok'] for r in ac_results.values()):>10,} OK  {grand_hrs:>8.2f} hrs")
    log.info("--- AUDIO CHECK SUMMARY END ---")

    EDAProcessor(cfg).sanitize()
    EDAProcessor(cfg).audio_eda()
    Tokenizer(cfg).run()
    DataSplitter(cfg).run()


# ---------------------------------------------------------------------------
# Service initialisation — all third-party auth/setup in one place
# ---------------------------------------------------------------------------

def _init_services(cfg: dict) -> None:
    """Initialise all external services before the pipeline runs.

    Called once at startup from ``__main__`` after config and logging are ready.
    Any service that requires authentication or global setup belongs here —
    not scattered inside individual modules.

    Current services
    ----------------
    - **W&B** — authenticate via ``WANDB_API_KEY`` env var (set in .env or
      cloud pod settings).  Skipped silently if wandb is not installed or
      ``sft.wandb.enabled`` is false.
    """
    # ---- Weights & Biases login ----
    wb_cfg = cfg.get("hf_sft", cfg.get("sft", {})).get("wandb", {})
    if wb_cfg.get("enabled"):
        try:
            import wandb
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if not wandb_api_key:
                log.warning(
                    "[INIT] WANDB_API_KEY not set — W&B disabled. "
                    "Set it in .env or export WANDB_API_KEY=..."
                )
            else:
                wandb.login(key=wandb_api_key, relogin=True)
                log.info("[INIT] W&B authenticated")
        except ModuleNotFoundError:
            log.warning("[INIT] wandb.enabled=true but wandb is not installed — pip install wandb")


def run_sft(cfg: dict) -> None:
    """Run single-stage HF Trainer SFT on WavLM Base+."""
    from trainer import HFSFTTrainer

    trainer = HFSFTTrainer(cfg)
    best_metrics = trainer.train()
    log.info("--- SFT SUMMARY ---")
    for k, v in best_metrics.items():
        log.info(f"  {k}: {v}")
    log.info("--- SFT SUMMARY END ---")


def run_avg(cfg: dict, top_n: int = 5) -> None:
    """Average top-N checkpoints, save to avg/, and evaluate."""
    from trainer import HFSFTTrainer

    trainer = HFSFTTrainer(cfg)
    avg_metrics = trainer.avg_only(top_n=top_n)
    if avg_metrics:
        log.info("--- AVG SUMMARY ---")
        for k, v in avg_metrics.items():
            log.info(f"  {k}: {v}")
        log.info("--- AVG SUMMARY END ---")
    else:
        log.error("Averaging failed — check logs above.")


def run_nst(cfg: dict) -> None:
    """Run full NST: teacher inference → filter → backup → re-split.

    Flow:
      1. teacher_infer.py  — GPU inference on orphan audio
      2. filter_pseudo_labels.py — confidence filtering, writes:
            pseudo_labelled.jsonl  (SFT-ready)
            3_transcript.jsonl     (DataSplitter input)
      3. Backup original splits:
            sft_train.jsonl → sft_train_original.jsonl
            sft_val.jsonl   → sft_val_original.jsonl
      4. DataSplitter(datasets=[1,2,3]) — exact same ETL logic;
         pseudo rows (empty child_id) → forced to train,
         val stays gold-speaker-only.
    """
    import subprocess

    nst_dir = SRC / "nst"
    env = {**os.environ, "PYTHONPATH": str(SRC)}

    # ── Step 1: Teacher inference ─────────────────────────────────
    log.info("--- NST: TEACHER INFERENCE ---")
    subprocess.run(
        [sys.executable, str(nst_dir / "teacher_infer.py")],
        check=True,
        env=env,
    )

    # ── Step 2: Confidence filter ─────────────────────────────────
    log.info("--- NST: FILTER + SANITY ---")
    subprocess.run(
        [sys.executable, str(nst_dir / "filter_pseudo_labels.py")],
        check=True,
        env=env,
    )

    # ── Step 3: Backup originals ──────────────────────────────────
    processed = Path(cfg["paths"]["processed"])
    pseudo_transcript = processed / "3_transcript.jsonl"

    if not pseudo_transcript.exists():
        log.warning("[NST] 3_transcript.jsonl not found — filter produced 0 rows?")
        log.info("--- NST COMPLETE (no re-split) ---")
        return

    sft_train = processed / "sft_train.jsonl"
    sft_val   = processed / "sft_val.jsonl"
    backup_train = processed / "sft_train_original.jsonl"

    # First run: save original gold-only train as backup
    if sft_train.exists() and not backup_train.exists():
        shutil.copy2(sft_train, backup_train)
        log.info("[NST] backed up sft_train.jsonl → sft_train_original.jsonl")

    # Re-run safety: always restore from backup before appending
    # (prevents duplicate pseudo-labels if --nst is run multiple times)
    if backup_train.exists():
        shutil.copy2(backup_train, sft_train)
        log.info("[NST] restored sft_train.jsonl from backup (clean gold-only)")

    # ── Step 4: Append pseudo-labels to existing train split ─────
    #  DO NOT re-split — re-splitting reshuffles speakers between
    #  train/val, leaking Run N's train speakers into Run N+1's val.
    #  Instead: keep sft_train.jsonl and sft_val.jsonl intact,
    #  just append the pseudo-labelled rows (3_transcript.jsonl) to train.
    import json

    n_pseudo = 0
    with open(pseudo_transcript) as src, open(sft_train, "a") as dst:
        for line in src:
            row = json.loads(line)
            row["dataset"] = 3
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_pseudo += 1

    log.info(
        "--- NST COMPLETE: appended %d pseudo-labels to sft_train.jsonl "
        "(val unchanged, %d rows) ---",
        n_pseudo,
        sum(1 for _ in open(sft_val)),
    )
    log.info("--- NST COMPLETE ---")


def run_benchmark(
    cfg: dict,
    *,
    workers: int = 1,
    batch_size: int = 4,
    batches: int = 100,
    all_data: bool = False,
) -> None:
    """Run DataLoader + Collator throughput benchmark."""
    from utils.bench_dataloader import run_benchmark as _bench

    _bench(
        cfg,
        split="train",
        num_batches=batches,
        batch_size=batch_size,
        num_workers=workers,
        all_data=all_data,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _configure_warnings()

    parser = argparse.ArgumentParser(description="309 — Phoneme-level ASR pipeline")
    parser.add_argument("--health",     action="store_true",
                        help="Check dependencies, raw data, and config")
    parser.add_argument("--etl",        action="store_true",
                        help="Run full ETL: audio check → sanitise → audio EDA → tokenizer → split")
    parser.add_argument("--model-sel",  action="store_true",
                        help="Model-selection acoustic diagnostics (insights only — run after --etl)")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run --etl then --model-sel in one shot")
    parser.add_argument("--del",        dest="delete", action="store_true",
                        help="Delete all generated data (processed/, models/, reports/, logs/, plots/) and __pycache__")
    parser.add_argument("--cc",         action="store_true",
                        help="Clear __pycache__ and .pytest_cache directories")
    parser.add_argument("--clearckpt",  action="store_true",
                        help="Delete sft_checkpoints/ folder (frees ~1.6 GB)")
    parser.add_argument("--sft",        action="store_true",
                        help="Run supervised fine-tuning (SFT) on WavLM Base+")
    parser.add_argument("--nst",        action="store_true",
                        help="Run NST: teacher inference → confidence filter → sanity check → pseudo_labelled.jsonl")
    parser.add_argument("--avg",        action="store_true",
                        help="Average top-N checkpoints, save to avg/, and evaluate")
    parser.add_argument("--avg-top-n",  type=int, default=5, dest="avg_top_n",
                        help="Number of best checkpoints to average (default: 5)")
    parser.add_argument("--benchmark",  action="store_true",
                        help="Benchmark DataLoader + Collator throughput")
    parser.add_argument("--workers",    type=int, default=1,
                        help="DataLoader workers for --benchmark (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4, dest="bench_batch_size",
                        help="Batch size for --benchmark (default: 4)")
    parser.add_argument("--batches",    type=int, default=100,
                        help="Number of batches for --benchmark (default: 100)")
    parser.add_argument("-a",           action="store_true", dest="all_data",
                        help="Benchmark entire dataset (overrides --batches)")
    args = parser.parse_args()

    cfg = load_config(SRC / "config" / "config.yaml")
    log_file = _setup_logging(cfg["paths"]["logs"])
    log.info(f"pipeline started  log={log_file}")
    _log_config(cfg)

    _init_services(cfg)

    if args.cc:
        clear_cache()
    elif args.clearckpt:
        clear_checkpoints(cfg)
    elif args.delete:
        delete_generated(cfg)
    elif args.health:
        health_check(cfg)
    elif args.preprocess:
        run_etl(cfg)
        run_model_selection_eda(cfg)
    elif args.etl:
        run_etl(cfg)
    elif args.model_sel:
        run_model_selection_eda(cfg)
    elif args.sft:
        run_sft(cfg)
    elif args.avg:
        run_avg(cfg, top_n=args.avg_top_n)
    elif args.nst:
        run_nst(cfg)
    elif args.benchmark:
        run_benchmark(
            cfg,
            workers=args.workers,
            batch_size=args.bench_batch_size,
            batches=args.batches,
            all_data=args.all_data,
        )
    else:
        parser.print_help()

    log.info("pipeline finished")
    clear_cache()
    log.info("Work by adolf & claude.")  # do not remove this footer note.
