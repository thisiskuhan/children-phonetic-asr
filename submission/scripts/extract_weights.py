#!/usr/bin/env python3
"""Extract model_state_dict from training checkpoint.

Usage:
    python submission/scripts/extract_weights.py [checkpoint_path] [output_path]

Default checkpoint: data/models/sft_checkpoints/best.pt
Default output:     submission/src/model/best_model.pt
"""

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "data" / "models" / "sft_checkpoints" / "best.pt"
DEFAULT_OUT = ROOT / "submission" / "src" / "model" / "best_model.pt"

EXPECTED_KEYS = 250
EXPECTED_PARAMS = 94_422_693
EXPECTED_VOCAB_DIM = 53  # lm_head.weight shape[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Source: {ckpt_path} ({ckpt_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ── Load ─────────────────────────────────────────────────────
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    if "model_state_dict" not in ckpt:
        print(f"ERROR: checkpoint missing 'model_state_dict' key")
        print(f"  Available keys: {list(ckpt.keys())}")
        sys.exit(1)

    state = ckpt["model_state_dict"]

    # ── Validate structure ───────────────────────────────────────
    n_keys = len(state)
    n_params = sum(v.numel() for v in state.values())

    if n_keys != EXPECTED_KEYS:
        print(f"ERROR: expected {EXPECTED_KEYS} keys, got {n_keys}")
        sys.exit(1)

    if n_params != EXPECTED_PARAMS:
        print(f"ERROR: expected {EXPECTED_PARAMS:,} params, got {n_params:,}")
        sys.exit(1)

    # ── Validate critical layers ─────────────────────────────────
    if "lm_head.weight" not in state:
        print("ERROR: missing lm_head.weight — CTC head not in checkpoint")
        sys.exit(1)

    vocab_dim = state["lm_head.weight"].shape[0]
    if vocab_dim != EXPECTED_VOCAB_DIM:
        print(f"ERROR: lm_head vocab_size={vocab_dim}, expected {EXPECTED_VOCAB_DIM}")
        sys.exit(1)

    if "lm_head.bias" not in state:
        print("ERROR: missing lm_head.bias")
        sys.exit(1)

    # Check for NaN/Inf in a sample of critical tensors
    for name in ["lm_head.weight", "lm_head.bias",
                 "wavlm.encoder.layers.0.attention.q_proj.weight",
                 "wavlm.feature_projection.projection.weight"]:
        if name not in state:
            print(f"ERROR: missing key '{name}'")
            sys.exit(1)
        t = state[name]
        if torch.isnan(t).any():
            print(f"ERROR: NaN detected in {name}")
            sys.exit(1)
        if torch.isinf(t).any():
            print(f"ERROR: Inf detected in {name}")
            sys.exit(1)

    # ── Save ─────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(out_path))

    # ── Verify round-trip ────────────────────────────────────────
    reloaded = torch.load(str(out_path), map_location="cpu", weights_only=True)
    if len(reloaded) != n_keys:
        print(f"ERROR: round-trip key count mismatch: {len(reloaded)} vs {n_keys}")
        sys.exit(1)

    for k in state:
        if k not in reloaded:
            print(f"ERROR: round-trip missing key '{k}'")
            sys.exit(1)
        if not torch.equal(state[k], reloaded[k]):
            print(f"ERROR: round-trip mismatch on '{k}'")
            sys.exit(1)

    size_mb = out_path.stat().st_size / 1024 / 1024
    out_hash = sha256(out_path)

    print(f"Keys:   {n_keys}")
    print(f"Params: {n_params:,}")
    print(f"Vocab:  {vocab_dim}")
    print(f"NaN/Inf: clean")
    print(f"Round-trip: verified (all {n_keys} tensors identical)")
    print(f"Saved → {out_path} ({size_mb:.1f} MB, sha256={out_hash})")


if __name__ == "__main__":
    main()
