"""
Trainer package — WavLM Base+ children's phonemic speech.
==========================================================

Modules
-------
- ``tracking``      — Shared W&B experiment tracking
- ``metrics``       — CTC decoding, PER, alignment diagnostics
- ``dataset``       — Lightweight JSONL manifest reader (SFTDataset)
- ``data_collator`` — Audio loading, preprocessing, dynamic batching (SFTCollator)
- ``model``         — Model construction, staged freeze/unfreeze, param groups
- ``sft_trainer``   — HF Trainer SFT with LLRD

Quick start
-----------
::

    from config import load_config
    from trainer import HFSFTTrainer

    cfg = load_config("config/config.yaml")
    trainer = HFSFTTrainer(cfg)
    best_metrics = trainer.train()
"""

from __future__ import annotations

# ---- Tracking (shared infrastructure) ----
from utils.tracking import WandbTracker

# ---- Trainers ----
from trainer.sft_trainer_hf import HFSFTTrainer

# ---- Model ----
from trainer.model import (
    build_model,
    freeze_for_stage,
    get_parameter_groups,
    get_stage_transition_groups,
    grad_norms,
)

# ---- Data ----
from trainer.dataset import SFTDataset
from trainer.data_collator import SFTCollator

# ---- Metrics ----
from trainer.metrics import (
    EMATracker,
    blank_ratio,
    compute_per,
    compute_per_batch,
    ctc_greedy_decode,
    format_alignment,
    mean_argmax_run_length,
    per_phoneme_recall,
)

__all__ = [
    # Tracking
    "WandbTracker",
    # Trainers
    "HFSFTTrainer",
    # Model
    "build_model",
    "freeze_for_stage",
    "get_parameter_groups",
    "get_stage_transition_groups",
    "grad_norms",
    # Data
    "SFTDataset",
    "SFTCollator",
    # Metrics
    "EMATracker",
    "blank_ratio",
    "compute_per",
    "compute_per_batch",
    "ctc_greedy_decode",
    "format_alignment",
    "mean_argmax_run_length",
    "per_phoneme_recall",
]
