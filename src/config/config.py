"""
Unified project configuration — single load, validated, defaults filled.
========================================================================

Called once from ``pipeline.py``.  The returned dict is passed to every module.

Required keys (must be present in config.yaml — no defaults):
    data   — root directory containing raw/, processed/, reports/, models/

All other keys have sensible defaults.  Override only what you need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Defaults — every optional key has a sensible fallback
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "datasets": [1, 2],

    # ---- Top-level shared settings (single source of truth) ----
    "seed": 1507,
    "num_workers": 8,
    "max_duration": 20.0,
    "min_duration": 0.3,

    "eda": {
        "max_duration": 25.0,
        "min_duration": 0.1,
        "max_runon_length": 15,
        "tps_min": 1.0,
        "tps_max": 25.0,
        "drill_min_words": 4,
        "drill_duplicate_ratio": 0.5,
    },

    "audio_check": {
        "check_md5": True,
        "fail_on_error": False,
        "audit_orphan_duration": False,  # always off by default — enable manually
    },

    "audio_eda": {
        "num_workers": 8,
        "rms": {"spread_ratio_threshold": 10.0},
        "clipping": {
            "sample_threshold": 0.999,
            "file_percent_threshold": 5.0,
            "corpus_percent_threshold": 1.0,
        },
        "speaker": {"top_k": 5, "dominance_threshold": 30.0},
        "duration": {"target_vram_seconds": 120.0},
        "silence": {
            "window_sec": 0.5,
            "ratio_threshold": 0.1,
            "corpus_percent_threshold": 20.0,
        },
        "spectral": {"subset_fraction": 0.05, "seed": 1507},
        "format": {"expected_sr": 16000, "expected_channels": 1},
    },

    "split": {
        "val_ratio": 0.05,
        "test_ratio": 0.0,
        "seed": 1507,
        "max_retries": 50,
        "min_phoneme_count_warn": 5,
    },

    "hf_sft": {
        "model_name": "microsoft/wavlm-base-plus",
        "vocab_size": 53,
        "blank_id": 0,
        "max_epochs": 50,
        "physical_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "fp16": False,
        "bf16": True,
        "tf32": True,
        "head_lr": 3e-4,
        "encoder_lr": 1e-4,
        "cnn_lr": 3e-7,
        "weight_decay": 0.01,
        "warmup_steps": 1500,
        "max_grad_norm": 1.0,
        "llrd_decay": 0.85,
        "lr_min_ratio": 0.01,
        "mask_time_prob": 0.10,
        "mask_time_length": 10,
        "mask_feature_prob": 0.01,
        "mask_feature_length": 10,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        "feat_proj_dropout": 0.0,
        "final_dropout": 0.1,
        "layerdrop": 0.0,
        "speed_perturb": False,
        "speed_perturb_range": [0.9, 1.1],
        "speed_augment": True,
        "noise_augment": True,
        "pitch_augment": True,
        "save_top_k": 3,
        "early_stopping_patience": 8,
        "checkpoint_averaging": False,
        "checkpoint_avg_top_n": 5,
        "min_duration_sec": 0.5,
        "prefetch_factor": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "unfreeze_cnn": False,
        "resume_from": None,
        "wandb": {
            "enabled": False,
            "project": "309-hf-sft",
            "entity": None,
            "run_name": None,
            "tags": [],
        },
    },
}

# Keys that MUST be present and non-empty in config.yaml
_REQUIRED = ("data",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _resolve_paths(cfg: dict, project_root: Path) -> None:
    """Derive all standard directory paths from ``cfg["data"]``."""
    data = Path(cfg["data"])
    if not data.is_absolute():
        data = (project_root / data).resolve()
    cfg["data"] = str(data)

    keys = cfg["datasets"]
    ssl_root = data / "ssl"

    # Resolve hf_sft noise_dir / rir_dir relative to data/ssl/
    hf_sft_cfg = cfg.get("hf_sft", {})
    for key in ("noise_dir", "rir_dir"):
        val = hf_sft_cfg.get(key)
        if val is not None:
            p = Path(val)
            if not p.is_absolute():
                p = ssl_root / p
            hf_sft_cfg[key] = str(p)

    audio_dirs = {k: str(data / "raw" / f"{k}_audio") for k in keys}

    # NST pseudo-labels (dataset 3) reuse dataset 2's audio directory.
    # DataSplitter writes dataset: 3 in sft_train.jsonl after re-split;
    # SFTDataset needs audio_dirs[3] to resolve those rows.
    processed = data / "processed"
    if (processed / "3_transcript.jsonl").exists() and 2 in keys:
        audio_dirs[3] = audio_dirs[2]

    cfg["paths"] = {
        "raw":        str(data / "raw"),
        "processed":  str(processed),
        "reports":    str(data / "reports"),
        "models":     str(data / "models"),
        "logs":       str(data / "logs"),
        "plots":      str(data / "plots"),
        "tokenizer":  str(data / "models" / "tokenizer"),
        "datasets":   {k: str(data / "raw" / f"{k}_train_phon_transcripts.jsonl") for k in keys},
        "audio_dirs": audio_dirs,
        "word_track_transcript": str(data / "raw" / "train_word_transcripts.jsonl"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load config.yaml, validate required fields, fill defaults, resolve paths.

    Raises ``ValueError`` if a required key is missing or empty.
    """
    import yaml  # lazy — only dependency on pyyaml

    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # ---- Validate required keys ----
    for key in _REQUIRED:
        if key not in raw or not raw[key]:
            raise ValueError(
                f"config.yaml: '{key}' is REQUIRED but missing or empty.  "
                f"Set it before running any pipeline command."
            )

    # ---- Merge user values over defaults ----
    cfg = _deep_merge(_DEFAULTS, raw)

    # ---- Propagate top-level shared settings into sections ----
    _PROPAGATION_MAP = {
        "seed":         ("split", "hf_sft"),
        "num_workers":  ("audio_eda", "hf_sft", "nst"),
        "max_duration": ("eda", "hf_sft", "nst"),
        "min_duration": ("eda", "nst"),
    }
    for key, sections in _PROPAGATION_MAP.items():
        top_val = cfg.get(key)
        if top_val is not None:
            for sec in sections:
                if sec in cfg and isinstance(cfg[sec], dict):
                    cfg[sec][key] = top_val

    # ---- Resolve paths (relative to project root = src/../) ----
    project_root = config_path.resolve().parent.parent.parent
    _resolve_paths(cfg, project_root)

    return cfg
