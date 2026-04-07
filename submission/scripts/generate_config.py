#!/usr/bin/env python3
"""Generate model/config.json from pretrained WavLM with training overrides.

Requires network (downloads pretrained config once).
Must match build_model() in src/trainer/model.py exactly.

Usage:
    python submission/scripts/generate_config.py
"""

from pathlib import Path
from transformers import WavLMForCTC

OUT = Path(__file__).resolve().parents[1] / "src" / "model"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    model = WavLMForCTC.from_pretrained(
        "microsoft/wavlm-base-plus",
        use_safetensors=True,
        vocab_size=53,
        pad_token_id=0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=5,
        mask_feature_prob=0.004,
        mask_feature_length=10,
        layerdrop=0.0,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        final_dropout=0.1,
    )
    model.config.save_pretrained(str(OUT))

    n = sum(p.numel() for p in model.parameters())
    print(f"config.json → {OUT / 'config.json'}")
    print(f"Model: {n:,} params, vocab_size={model.config.vocab_size}")


if __name__ == "__main__":
    main()
