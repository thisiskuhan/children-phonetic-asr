#!/usr/bin/env python3
"""Average multiple HF checkpoints into a single model.

Uniform weight averaging of the top-N checkpoints from HF Trainer.
Typically gives 1-3% relative PER improvement by smoothing noise
in individual checkpoints.

Usage:
    python submission/scripts/avg_checkpoints.py \
        --checkpoints ckpt1/ ckpt2/ ckpt3/ \
        --output submission/src/model/

    # Average top-5 from a training run:
    python submission/scripts/avg_checkpoints.py \
        --checkpoints runs/run_18/checkpoint-{1000,2000,3000,4000,5000}/ \
        --output submission/src/model/
"""

import argparse
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def average_checkpoints(checkpoint_dirs: list[Path]) -> OrderedDict:
    """Load and uniformly average state dicts from N checkpoints."""
    N = len(checkpoint_dirs)
    avg_state: OrderedDict | None = None
    keys = None

    for i, ckpt_dir in enumerate(checkpoint_dirs):
        st_path = ckpt_dir / "model.safetensors"
        bin_path = ckpt_dir / "pytorch_model.bin"
        if st_path.exists():
            from safetensors.torch import load_file
            sd = load_file(str(st_path), device="cpu")
        elif bin_path.exists():
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            print(f"ERROR: no model weights in {ckpt_dir}", file=sys.stderr)
            sys.exit(1)

        n_params = sum(v.numel() for v in sd.values())
        print(f"  [{i+1}/{N}] {ckpt_dir.name}: {len(sd)} keys, {n_params:,} params")

        if avg_state is None:
            keys = list(sd.keys())
            avg_state = OrderedDict()
            for k in keys:
                avg_state[k] = sd[k].float()
        else:
            if set(sd.keys()) != set(keys):
                missing = set(keys) - set(sd.keys())
                extra = set(sd.keys()) - set(keys)
                print(f"ERROR: key mismatch in {ckpt_dir.name}: "
                      f"missing={missing}, extra={extra}", file=sys.stderr)
                sys.exit(1)
            for k in keys:
                avg_state[k] += sd[k].float()
        del sd

    # Divide by N
    for k in keys:
        avg_state[k] /= N

    return avg_state


def main():
    parser = argparse.ArgumentParser(
        description="Average N HF checkpoints (uniform weight averaging)")
    parser.add_argument(
        "--checkpoints", nargs="+", type=Path, required=True,
        help="Paths to HF checkpoint directories")
    parser.add_argument(
        "--output", type=Path, default=Path("submission/src/model"),
        help="Output directory for averaged model (default: submission/src/model/)")
    args = parser.parse_args()

    # Validate
    for ckpt in args.checkpoints:
        if not ckpt.is_dir():
            print(f"ERROR: not a directory: {ckpt}", file=sys.stderr)
            sys.exit(1)

    N = len(args.checkpoints)
    print(f"Averaging {N} checkpoints:")

    avg_state = average_checkpoints(args.checkpoints)

    # Convert back to original dtype (bf16/fp16 if saved that way)
    ref_path = args.checkpoints[0] / "model.safetensors"
    if ref_path.exists():
        from safetensors.torch import load_file
        ref_sd = load_file(str(ref_path), device="cpu")
        for k in avg_state:
            avg_state[k] = avg_state[k].to(ref_sd[k].dtype)
        del ref_sd

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        out_path = args.output / "model.safetensors"
        save_file(avg_state, str(out_path))
        print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    except ImportError:
        out_path = args.output / "pytorch_model.bin"
        torch.save(avg_state, str(out_path))
        print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Copy config.json from first checkpoint
    config_src = args.checkpoints[0] / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, args.output / "config.json")
        print(f"Copied: config.json from {args.checkpoints[0].name}")

    n_params = sum(v.numel() for v in avg_state.values())
    print(f"Total params: {n_params:,}")
    print(f"Averaged {N} checkpoints → {args.output}")


if __name__ == "__main__":
    main()
