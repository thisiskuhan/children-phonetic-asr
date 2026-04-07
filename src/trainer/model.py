"""Model construction — WavLMForCTC for SFT.
=============================================

- ``build_model``                    — load → configure → Stage 1 freeze
- ``freeze_for_stage``               — set ``requires_grad`` for a stage
- ``get_parameter_groups``           — all trainable groups for a stage
- ``get_stage_transition_groups``    — only groups newly unfrozen at a stage
- ``grad_norms``                     — per-component gradient L₂ norms
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import WavLMConfig, WavLMForCTC

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NO_DECAY_SUFFIXES: tuple[str, ...] = (
    "bias",
    "LayerNorm.weight",
    "LayerNorm.bias",
    "layer_norm.weight",
    "layer_norm.bias",
)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def build_model(cfg: dict[str, Any]) -> WavLMForCTC:
    """Load WavLM Base+ and apply full SFT configuration (guide §2).

    Handles two checkpoint formats:
    1. **Native WavLM** — standard ``WavLMForCTC.from_pretrained`` path.
    2. **Wav2Vec2ForPreTraining** (SSL checkpoint) — keys are prefixed
       ``wav2vec2.*`` instead of ``wavlm.*``.  We detect this via
       ``config.json``, remap keys, and load into a fresh ``WavLMForCTC``.

    The CTC head (``lm_head``) is reinitialised only when the checkpoint's
    vocab size differs from the target (e.g. loading base WavLM with
    vocab_size 32 → 53).  When resuming from our own SFT checkpoint
    (same vocab), the learned head weights are preserved.
    Stage 1 freeze is applied (only ``lm_head`` trainable).

    Parameters
    ----------
    cfg : dict
        Global config with an ``sft`` sub-dict.

    Returns
    -------
    WavLMForCTC
        Configured model ready for Stage 1 training.
    """
    sft = cfg["sft"]

    model_path = Path(sft["model_name"])
    is_wav2vec2_ckpt = _is_wav2vec2_checkpoint(model_path)

    # Config overrides applied regardless of loading path.
    config_overrides = dict(
        vocab_size=sft["vocab_size"],
        pad_token_id=sft["blank_id"],
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        apply_spec_augment=True,
        mask_time_prob=sft["mask_time_prob"],
        mask_time_length=sft["mask_time_length"],
        mask_feature_prob=sft["mask_feature_prob"],
        mask_feature_length=sft["mask_feature_length"],
        layerdrop=sft["layerdrop"],
        attention_dropout=sft["attention_dropout"],
        hidden_dropout=sft["hidden_dropout"],
        feat_proj_dropout=sft["feat_proj_dropout"],
        final_dropout=sft["final_dropout"],
    )

    if is_wav2vec2_ckpt:
        model = _load_wav2vec2_into_wavlm(model_path, config_overrides)
    else:
        model = WavLMForCTC.from_pretrained(
            str(model_path),
            use_safetensors=True,
            **config_overrides,
        )

    # Reinitialise CTC head — guarantees fresh random weights when loading
    # a pretrained base model.  Skip when resuming from our own checkpoint
    # (same vocab size means the head is already correctly shaped).
    if model.lm_head.out_features != sft["vocab_size"]:
        log.info("[MODEL] CTC head vocab mismatch (%d→%d) — reinitialising",
                 model.lm_head.out_features, sft["vocab_size"])
        model.lm_head = nn.Linear(
            model.config.hidden_size, sft["vocab_size"], bias=True,
        )
    else:
        log.info("[MODEL] CTC head matches vocab_size=%d — keeping learned weights",
                 sft["vocab_size"])

    # Gradient checkpointing — OFF for Stage 1 (encoder is frozen → pure
    # inference, no recomputation needed).  Enabled in Stages 2/3 via
    # _transition_to_stage() when encoder layers become trainable.
    # model.gradient_checkpointing_enable()    # §9 — deferred to stage 2+

    # Stage 1 freeze — only lm_head trainable.
    freeze_for_stage(model, stage=1)

    # ---- Post-load assertions (guide §17 checklist) ----
    assert model.config.pad_token_id == sft["blank_id"]
    assert model.config.vocab_size == sft["vocab_size"]
    assert model.config.ctc_loss_reduction == "mean"
    assert model.config.ctc_zero_infinity is True
    assert model.config.layerdrop == sft["layerdrop"]

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(model.wavlm.encoder.layers)
    hidden = model.config.hidden_size
    log.info(
        "[MODEL] WavLM loaded (%d layers, hidden=%d) — %s params, %s trainable (Stage 1)",
        num_layers, hidden, f"{total:,}", f"{trainable:,}",
    )

    return model


def _is_wav2vec2_checkpoint(model_path: Path) -> bool:
    """Return True if the checkpoint was saved by Wav2Vec2ForPreTraining."""
    config_file = model_path / "config.json"
    if not config_file.exists():
        return False
    with open(config_file) as f:
        ckpt_cfg = json.load(f)
    return ckpt_cfg.get("model_type") == "wav2vec2"


def _load_wav2vec2_into_wavlm(
    model_path: Path,
    config_overrides: dict[str, Any],
) -> WavLMForCTC:
    """Load a Wav2Vec2ForPreTraining checkpoint into WavLMForCTC.

    Remaps ``wav2vec2.*`` state-dict keys to ``wavlm.*`` and drops
    SSL-only heads (quantizer, project_hid, project_q).
    """
    from safetensors.torch import load_file

    # Build WavLM config from the checkpoint config + overrides
    wavlm_config = WavLMConfig.from_pretrained(str(model_path))
    for k, v in config_overrides.items():
        setattr(wavlm_config, k, v)

    # Create fresh WavLMForCTC (random weights)
    model = WavLMForCTC(wavlm_config)

    # Load Wav2Vec2 state dict and remap keys
    w2v_sd = load_file(str(model_path / "model.safetensors"))
    remapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, tensor in w2v_sd.items():
        if key.startswith("wav2vec2."):
            new_key = "wavlm." + key[len("wav2vec2."):]
            remapped[new_key] = tensor
        else:
            # SSL-only: quantizer.*, project_hid.*, project_q.*
            skipped.append(key)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    del w2v_sd, remapped  # free ~400 MB state dict copies
    # lm_head is expected to be missing (reinitialised anyway)
    real_missing = [k for k in missing if not k.startswith("lm_head")]
    if real_missing:
        log.warning(
            "[MODEL] %d keys missing after wav2vec2→wavlm remap: %s",
            len(real_missing), real_missing[:10],
        )
    log.info(
        "[MODEL] Loaded Wav2Vec2ForPreTraining checkpoint → WavLMForCTC: "
        "%d keys remapped, %d SSL-only keys skipped, %d missing (lm_head expected)",
        len(remapped), len(skipped), len(missing),
    )

    return model


# ---------------------------------------------------------------------------
# freeze_for_stage
# ---------------------------------------------------------------------------

def freeze_for_stage(
    model: WavLMForCTC,
    stage: int,
    *,
    unfreeze_cnn: bool = True,
) -> None:
    """Set ``requires_grad`` flags for the given training stage.

    Layer split is dynamic — works for any encoder depth (Base+ 12,
    Large 24, etc.).  The upper/lower boundary is ``num_layers // 2``.

    Stage 1
        Only ``lm_head`` trainable.
    Stage 2
        ``lm_head`` + upper half of transformer layers.
    Stage 3
        Everything — full encoder, feature projection.
        CNN is unfrozen only when *unfreeze_cnn* is ``True``.
        Set to ``False`` when fine-tuning an SSL-pretrained model
        (CNN features are already well-trained).

    Parameters
    ----------
    model : WavLMForCTC
    stage : {1, 2, 3}
    unfreeze_cnn : bool
        Whether to unfreeze the CNN feature extractor in Stage 3.
        Default ``True`` (unfreeze).  Set ``False`` for SSL-pretrained models.
    """
    assert stage in (1, 2, 3), f"Invalid stage: {stage}"

    num_layers = len(model.wavlm.encoder.layers)
    mid = num_layers // 2  # Base+=6, Large=12

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad_(False)

    # Disable the HuggingFace-internal _requires_grad flag on the CNN
    # feature extractor.  Without this, the CNN forward always sets
    # hidden_states.requires_grad = True, forcing autograd to build a
    # full computation graph through all 7 frozen CNN layers on the
    # raw 16 kHz waveform — resulting in massive VRAM waste.
    model.wavlm.feature_extractor._freeze_parameters()

    # CTC head — always trainable.
    for p in model.lm_head.parameters():
        p.requires_grad_(True)

    if stage >= 2:
        # Upper half of transformer layers.
        for layer in model.wavlm.encoder.layers[mid:]:
            for p in layer.parameters():
                p.requires_grad_(True)

    if stage >= 3:
        # Lower half of transformer layers.
        for layer in model.wavlm.encoder.layers[:mid]:
            for p in layer.parameters():
                p.requires_grad_(True)

        # Encoder infrastructure: pos_conv_embed, layer_norm, masked_spec_embed.
        for p in model.wavlm.encoder.pos_conv_embed.parameters():
            p.requires_grad_(True)
        for p in model.wavlm.encoder.layer_norm.parameters():
            p.requires_grad_(True)
        model.wavlm.masked_spec_embed.requires_grad_(True)

        # Feature projection (Linear → hidden_size + LayerNorm).
        for p in model.wavlm.feature_projection.parameters():
            p.requires_grad_(True)

        # CNN feature extractor (7 conv layers) — conditional.
        if unfreeze_cnn:
            # Re-enable the _requires_grad flag so the CNN forward
            # properly tracks gradients when CNN is trainable.
            model.wavlm.feature_extractor._requires_grad = True
            for p in model.wavlm.feature_extractor.parameters():
                p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "[MODEL] Stage %d freeze (%d layers, mid=%d) — %s / %s trainable (%.1f%%)",
        stage, num_layers, mid,
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )


# ---------------------------------------------------------------------------
# Parameter groups — internal helpers
# ---------------------------------------------------------------------------

def _split_decay(
    named_params: list[tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float,
    group_name: str,
) -> list[dict[str, Any]]:
    """Split named params into *decay* and *no-decay* groups.

    Only includes parameters with ``requires_grad == True``.
    Returns up to two dicts suitable for ``torch.optim.AdamW``.
    Each dict carries an extra ``"name"`` key for logging.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if any(nd in name for nd in _NO_DECAY_SUFFIXES):
            no_decay.append(param)
        else:
            decay.append(param)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({
            "params": decay,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": f"{group_name}_decay",
        })
    if no_decay:
        groups.append({
            "params": no_decay,
            "lr": lr,
            "weight_decay": 0.0,
            "name": f"{group_name}_no_decay",
        })
    return groups


def _head_groups(
    model: WavLMForCTC, cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """CTC head (``lm_head``) — highest LR (3e-4)."""
    sft = cfg["sft"]
    return _split_decay(
        list(model.lm_head.named_parameters()),
        lr=sft["head_lr"],
        weight_decay=sft["weight_decay"],
        group_name="head",
    )


def _upper_encoder_groups(
    model: WavLMForCTC, cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Upper half of transformer layers with per-layer LLRD.

    Dynamic split: layers[mid:] where mid = num_layers // 2.
    Base+ (12 layers) → layers 6–11.  Large (24 layers) → layers 12–23.

    When ``llrd_decay < 1.0``, each layer gets its own optimizer group:
        lr(layer_i) = encoder_lr × decay^(num_layers - 1 - i)
    Top layer gets full ``encoder_lr``; lowest upper layer gets most decay.
    When ``llrd_decay == 1.0``, all layers share ``encoder_lr`` (no LLRD).
    """
    sft = cfg["sft"]
    base_lr = sft["encoder_lr"]
    decay = sft.get("llrd_decay", 1.0)
    wd = sft["weight_decay"]
    num_layers = len(model.wavlm.encoder.layers)
    mid = num_layers // 2

    groups: list[dict[str, Any]] = []
    for i, layer in enumerate(model.wavlm.encoder.layers[mid:], start=mid):
        depth = num_layers - 1 - i
        lr = base_lr * (decay ** depth)
        params = [(f"encoder.layers.{i}.{name}", p)
                  for name, p in layer.named_parameters()]
        groups += _split_decay(params, lr=lr, weight_decay=wd, group_name=f"enc_L{i}")
    return groups


def _lower_encoder_groups(
    model: WavLMForCTC, cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Lower half of layers with per-layer LLRD + encoder infrastructure.

    Dynamic split: layers[:mid] where mid = num_layers // 2.
    Base+ (12 layers) → layers 0–5.  Large (24 layers) → layers 0–11.

    Transformer layers get per-layer LLRD rates.  Infrastructure params
    (feature_projection, pos_conv_embed, layer_norm, masked_spec_embed)
    are grouped at layer-0's LR — the deepest / most decayed rate — since
    they are closest to the CNN / raw acoustics and should move least.
    """
    sft = cfg["sft"]
    base_lr = sft["encoder_lr"]
    decay = sft.get("llrd_decay", 1.0)
    wd = sft["weight_decay"]
    num_layers = len(model.wavlm.encoder.layers)
    mid = num_layers // 2

    groups: list[dict[str, Any]] = []

    # Per-layer groups for lower half.
    for i, layer in enumerate(model.wavlm.encoder.layers[:mid]):
        depth = num_layers - 1 - i
        lr = base_lr * (decay ** depth)
        params = [(f"encoder.layers.{i}.{name}", p)
                  for name, p in layer.named_parameters()]
        groups += _split_decay(params, lr=lr, weight_decay=wd,
                               group_name=f"enc_L{i}")

    # Infrastructure at layer-0 LR (deepest decay).
    infra_lr = base_lr * (decay ** (num_layers - 1))
    infra_params: list[tuple[str, nn.Parameter]] = []

    for name, p in model.wavlm.feature_projection.named_parameters():
        infra_params.append((f"feature_projection.{name}", p))
    for name, p in model.wavlm.encoder.pos_conv_embed.named_parameters():
        infra_params.append((f"encoder.pos_conv_embed.{name}", p))
    for name, p in model.wavlm.encoder.layer_norm.named_parameters():
        infra_params.append((f"encoder.layer_norm.{name}", p))
    infra_params.append(("masked_spec_embed", model.wavlm.masked_spec_embed))

    groups += _split_decay(infra_params, lr=infra_lr, weight_decay=wd,
                           group_name="enc_infra")
    return groups


def _cnn_groups(
    model: WavLMForCTC, cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """CNN feature extractor — lowest LR (1e-6, encoder/10)."""
    sft = cfg["sft"]
    return _split_decay(
        list(model.wavlm.feature_extractor.named_parameters()),
        lr=sft["cnn_lr"],
        weight_decay=sft["weight_decay"],
        group_name="cnn",
    )


# ---------------------------------------------------------------------------
# Parameter groups — public API
# ---------------------------------------------------------------------------

def get_parameter_groups(
    model: WavLMForCTC,
    stage: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return ALL trainable parameter groups for *stage*.

    Used to create the optimizer at Stage 1.  For stage transitions
    (2 → 3), prefer :func:`get_stage_transition_groups` so that
    existing optimizer momentum is preserved.

    A runtime assertion verifies every ``requires_grad`` parameter
    appears in exactly one group — no orphans, no duplicates.
    """
    sft = cfg["sft"]
    groups = _head_groups(model, cfg)
    if stage >= 2:
        groups += _upper_encoder_groups(model, cfg)
    if stage >= 3:
        groups += _lower_encoder_groups(model, cfg)
        if sft.get("unfreeze_cnn", True):
            groups += _cnn_groups(model, cfg)

    _verify_groups(model, groups)
    return groups


def get_stage_transition_groups(
    model: WavLMForCTC,
    new_stage: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return ONLY the parameter groups newly trainable in *new_stage*.

    Used with ``optimizer.add_param_group()`` at stage transitions so
    that already-training parameters keep their momentum and variance.
    """
    sft = cfg["sft"]
    if new_stage == 2:
        return _upper_encoder_groups(model, cfg)
    if new_stage == 3:
        groups = _lower_encoder_groups(model, cfg)
        if sft.get("unfreeze_cnn", True):
            groups += _cnn_groups(model, cfg)
        return groups
    raise ValueError(f"No transition into stage {new_stage}")


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def _verify_groups(
    model: WavLMForCTC,
    groups: list[dict[str, Any]],
) -> None:
    """Assert every requires_grad param is in exactly one group."""
    grouped_ids: set[int] = set()
    for g in groups:
        for p in g["params"]:
            pid = id(p)
            assert pid not in grouped_ids, (
                f"Duplicate param in groups (group={g.get('name', '?')})"
            )
            grouped_ids.add(pid)

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert id(p) in grouped_ids, (
                f"Trainable param not in any optimizer group: {name}"
            )


# ---------------------------------------------------------------------------
# Gradient diagnostics (SFT guide §10)
# ---------------------------------------------------------------------------

def grad_norms(model: WavLMForCTC) -> dict[str, float]:
    """Compute per-component gradient L₂ norms.

    Dynamic split: upper = layers[mid:], lower = layers[:mid] where
    mid = num_layers // 2.  Works for Base+ (12) and Large (24).

    Returns
    -------
    dict[str, float]
        Keys: ``head``, ``upper_encoder``, ``lower_encoder``, ``cnn``.
        Value is ``0.0`` when a component is frozen or has no
        accumulated gradients.
    """
    def _norm(params: list[nn.Parameter]) -> float:
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        # vector_norm(dtype=float32) accumulates in fp32 registers
        # without materialising a full fp32 copy — saves ~380 MB
        # peak VRAM vs .float() at full-model scale.
        return sum(
            torch.linalg.vector_norm(g, dtype=torch.float32).square()
            for g in grads
        ).sqrt().item()

    num_layers = len(model.wavlm.encoder.layers)
    mid = num_layers // 2

    # ---- Head ----
    head = _norm(list(model.lm_head.parameters()))

    # ---- Upper encoder (layers mid..N-1) ----
    upper: list[nn.Parameter] = []
    for layer in model.wavlm.encoder.layers[mid:]:
        upper.extend(layer.parameters())
    upper_norm = _norm(upper)

    # ---- Lower encoder (layers 0..mid-1 + projection + infra) ----
    lower: list[nn.Parameter] = []
    for layer in model.wavlm.encoder.layers[:mid]:
        lower.extend(layer.parameters())
    lower.extend(model.wavlm.feature_projection.parameters())
    lower.extend(model.wavlm.encoder.pos_conv_embed.parameters())
    lower.extend(model.wavlm.encoder.layer_norm.parameters())
    lower.append(model.wavlm.masked_spec_embed)
    lower_norm = _norm(lower)

    # ---- CNN ----
    cnn = _norm(list(model.wavlm.feature_extractor.parameters()))

    return {
        "head": head,
        "upper_encoder": upper_norm,
        "lower_encoder": lower_norm,
        "cnn": cnn,
    }

