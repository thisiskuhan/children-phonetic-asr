"""
HF Trainer — single-stage WavLM CTC fine-tuning via HuggingFace Trainer.
========================================================================

Replaces the original 3-stage ``SFTTrainer`` (sft_trainer.py, 2 369 lines).

RETROSPECTIVE — why the 3-stage trainer was over-engineered
-----------------------------------------------------------
The custom loop in ``sft_trainer.py`` implemented:

  What was GOOD (kept here):
  - LLRD (layer-wise learning-rate decay)            → real ~1-3 % PER gain
  - Discriminative LR (head vs encoder)              → standard, effective
  - Top-k checkpointing by val PER                   → essential
  - Early stopping on val PER plateau                 → prevents wasted GPU
  - SpecAugment / speed perturbation                  → solid augmentation

  What was BAD / unnecessary:
  - 3-stage signal-driven unfreezing (head → top-6 → full)
        Added ~800 lines of stage logic, EMA signal trackers, per-group
        warmup at transitions, and resume-checkpoint stage replay — all
        for no measurable PER improvement over simply unfreezing everything
        from step 0 with LLRD.
  - Age-weighted CTC loss (inverse-frequency by speaker age bucket)
        Complex to compute, required separate EDA report, and made loss
        scale unstable.  Standard mean CTC loss works just as well.
  - EMA-smoothed stage transition signals
        Whole subsystem of scalar trackers deciding when to advance stages.
        Fragile, hard to tune, and the stages themselves didn't help.
  - Custom AMP / GradScaler management
        HF Trainer handles this correctly out of the box.
  - Manual gradient clipping + grad-ratio band monitor
        Nice for debugging but HF Trainer's max_grad_norm does the job.
  - Manual W&B logging, throughput tracking, alignment logging
        All replaced by HF Trainer callbacks + compute_metrics.

  Result: this 634-line HF Trainer wrapper matched the 2 369-line custom
  loop at run 4 (leaderboard PER 0.336) with far less code and faster
  iteration.

Public API
----------
- ``HFSFTTrainer(cfg)``  — build model, datasets, collators from config
- ``HFSFTTrainer.train()`` — run full training loop to completion

Reuses existing modules:
- ``trainer.model.build_model`` — loads WavLM Base+, reinits CTC head
- ``trainer.model.get_parameter_groups`` — LLRD-aware param groups
- ``trainer.dataset.SFTDataset`` — JSONL manifest reader
- ``trainer.data_collator.SFTCollator`` — audio loading + preprocessing
- ``trainer.metrics`` — CTC decoding, PER, CER, blank ratio, etc.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import shutil
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any

import types

import jiwer
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    EarlyStoppingCallback,
)
from transformers.models.wavlm.modeling_wavlm import _compute_mask_indices
from transformers.trainer_pt_utils import LengthGroupedSampler

from trainer.data_collator import SFTCollator
from trainer.dataset import SFTDataset
from trainer.metrics import compute_per_batch, compute_per_and_recall
from trainer.email_callback import EmailNotificationCallback
from trainer.model import (
    build_model,
    freeze_for_stage,
    get_parameter_groups,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kill torch.compile / inductor EARLY — before any torch import triggers
# inductor's 32-thread compile-worker pool (~11 GB wasted RAM).
# Must happen at module-import time, not in train().
# ---------------------------------------------------------------------------
import torch._dynamo
torch._dynamo.config.disable = True
try:
    import torch._inductor.config  # noqa: E402
    torch._inductor.config.compile_threads = 0
except (ImportError, AttributeError):
    pass


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=False + benchmark=True set later in train();
    # here we only seed PRNGs for statistical reproducibility.


# ---------------------------------------------------------------------------
# torch.compile-safe SpecAugment — no in-place indexed assignments
# ---------------------------------------------------------------------------

def _patch_wavlm_mask_hidden_states(model) -> None:
    """Replace WavLM._mask_hidden_states with a graph-break-free version.

    The stock implementation uses two anti-patterns that break torch.compile:

    1. ``torch.tensor(numpy_array, ...)`` — during dynamo tracing the numpy
       output of ``_compute_mask_indices`` becomes a FakeTensor/proxy, so
       torch.tensor() emits the "copy construct from tensor" UserWarning
       and may cause graph breaks.

    2. ``hidden_states[bool_mask] = value`` — in-place boolean index-put on
       a gradient-tracked tensor triggers IndexPutBackward0 and forces dynamo
       to restart the graph at every call.

    Replacements:
    - ``torch.from_numpy(arr).to(device, dtype)`` — zero-copy, no proxy issue.
    - ``torch.where(mask, embed, hidden_states)`` — out-of-place, autograd-safe.
    - ``hidden_states.masked_fill(mask, 0.0)``    — out-of-place, compile-safe.
    """

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices=None,
        attention_mask=None,
    ) -> torch.FloatTensor:
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        batch_size, sequence_length, hidden_size = hidden_states.size()

        # ---- Time masking ----
        if mask_time_indices is not None:
            # Externally supplied mask (e.g. pre-training step)
            time_mask = mask_time_indices.bool()
        elif self.config.mask_time_prob > 0 and self.training:
            np_mask = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            # torch.from_numpy: zero-copy view — avoids the
            # "copy construct from tensor" dynamo warning.
            time_mask = torch.from_numpy(np_mask).to(
                device=hidden_states.device, dtype=torch.bool,
            )
        else:
            time_mask = None

        if time_mask is not None:
            # torch.where: out-of-place — no IndexPutBackward0 warning.
            mask_3d = time_mask.unsqueeze(-1).expand_as(hidden_states)
            embed = self.masked_spec_embed.to(hidden_states.dtype).expand_as(hidden_states)
            hidden_states = torch.where(mask_3d, embed, hidden_states)

        # ---- Feature masking ----
        if self.config.mask_feature_prob > 0 and self.training:
            np_feat_mask = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            feat_mask = torch.from_numpy(np_feat_mask).to(
                device=hidden_states.device, dtype=torch.bool,
            )
            feat_mask = feat_mask[:, None, :].expand(-1, sequence_length, -1)
            # masked_fill: out-of-place, compile-friendly.
            hidden_states = hidden_states.masked_fill(feat_mask, 0.0)

        return hidden_states

    # Bind as an instance method so self refers to model.wavlm
    model.wavlm._mask_hidden_states = types.MethodType(_mask_hidden_states, model.wavlm)
    log.info("[PATCH] WavLM._mask_hidden_states replaced with compile-safe version")


# ---------------------------------------------------------------------------
# Custom CTC Trainer — pops collator-extra keys, delegates CTC to model
# ---------------------------------------------------------------------------

class _CTCTrainer(Trainer):
    """HF Trainer subclass with CTC loss, optional age weighting, and SR-CTC.

    Fast path (no age weights, no SR-CTC): delegates to the model's
    built-in CTC loss — zero overhead.

    Slow path (age weights and/or SR-CTC active): pops labels, runs
    forward without labels, computes CTC + optional SR-CTC manually
    in fp32.
    """

    # Collators are set via __init__ kwargs and stored here so the
    # dataloader overrides can access them without fragile post-hoc
    # assignment.  See _LRGroupedCTCTrainer.__init__.
    _train_collator: SFTCollator | None = None
    _val_collator: SFTCollator | None = None
    _age_loss_weights: dict[str, float] | None = None
    _sr_ctc_beta: float = 0.0
    _ds1_mask_time_prob: float | None = None
    _default_mask_time_prob: float = 0.30
    _oom_count: int = 0

    # Per-eval buffers: collected by prediction_step, consumed by
    # compute_metrics.  Cleared at the start of each evaluate() call.
    _eval_output_lengths: list[np.ndarray]
    _eval_age_buckets: list[list[str]]
    _eval_datasets: list[list[str]]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        age_buckets = inputs.pop("age_buckets", None)
        inputs.pop("input_lengths", None)
        ds_ids = inputs.pop("datasets", None)

        # ---- Dynamic SpecAug: reduce mask_time_prob for DS1-heavy batches ----
        # DS1 is already noisy (~12 dB SNR); stacking full SpecAug on top
        # double-corrupts the signal.  Interpolate mask_time_prob based on
        # DS1 fraction in the batch: all-DS2 → 0.30, all-DS1 → ds1_mask_time_prob.
        _specaug_adjusted = False
        if self._ds1_mask_time_prob is not None and ds_ids is not None and self.model.training:
            ds1_count = sum(1 for d in ds_ids if d == "1")
            ds1_frac = ds1_count / len(ds_ids)
            if ds1_frac > 0:
                effective_prob = (
                    self._default_mask_time_prob
                    - (self._default_mask_time_prob - self._ds1_mask_time_prob) * ds1_frac
                )
                model.wavlm.config.mask_time_prob = effective_prob
                _specaug_adjusted = True

        has_age_weights = bool(self._age_loss_weights)
        sr_beta = self._sr_ctc_beta

        if not has_age_weights and sr_beta <= 0:
            # Fast path — delegate to model's built-in CTC loss (zero overhead)
            outputs = model(**inputs)
            loss = outputs.loss
            if _specaug_adjusted:
                model.wavlm.config.mask_time_prob = self._default_mask_time_prob
            return (loss, outputs) if return_outputs else loss

        # ---- Need manual CTC and/or SR-CTC ----
        # Forward WITHOUT labels to get logits directly
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs["input_values"].shape[:2],
                dtype=torch.long, device=inputs["input_values"].device,
            )
        input_lengths = model._get_feat_extract_output_lengths(
            attention_mask.sum(-1),
        ).to(torch.long)

        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        log_probs = torch.nn.functional.log_softmax(
            logits, dim=-1, dtype=torch.float32,
        )

        # ---- CTC loss ----
        with torch.backends.cudnn.flags(enabled=False):
            if has_age_weights:
                per_sample_loss = torch.nn.functional.ctc_loss(
                    log_probs.transpose(0, 1), flattened_targets,
                    input_lengths, target_lengths,
                    blank=model.config.pad_token_id,
                    reduction="none",
                    zero_infinity=True,
                )
                length_norm_loss = per_sample_loss / target_lengths.float().clamp(min=1)
                device = per_sample_loss.device
                weights = torch.ones(len(length_norm_loss), device=device)
                if age_buckets is not None:
                    w = self._age_loss_weights
                    for i, bucket in enumerate(age_buckets):
                        if bucket in w:
                            weights[i] = w[bucket]
                ctc_loss = (length_norm_loss * weights).sum() / weights.sum()
            else:
                ctc_loss = torch.nn.functional.ctc_loss(
                    log_probs.transpose(0, 1), flattened_targets,
                    input_lengths, target_lengths,
                    blank=model.config.pad_token_id,
                    reduction="mean",
                    zero_infinity=True,
                )

        # ---- SR-CTC regularization (Yao et al., ICLR 2025) ----
        # Smooth CTC distribution along time with kernel [0.25, 0.5, 0.25],
        # then minimize KL(stop_grad(smoothed) || original).
        # Attacks blank-peaky distributions: our model emits 85% blank frames,
        # causing 28% deletion errors — especially on short/noisy DS1 3-4yo.
        if sr_beta > 0:
            with torch.no_grad():
                probs = log_probs.exp()                          # (B, T, V)
                probs_t = probs.permute(0, 2, 1)                 # (B, V, T)
                B, V, T = probs_t.shape
                probs_flat = probs_t.reshape(B * V, 1, T)
                kernel = torch.tensor(
                    [0.25, 0.5, 0.25],
                    device=probs.device, dtype=probs.dtype,
                ).view(1, 1, 3)
                # Replicate padding preserves sum-to-1 at boundaries
                probs_padded = torch.nn.functional.pad(
                    probs_flat, (1, 1), mode="replicate",
                )
                probs_smooth = torch.nn.functional.conv1d(probs_padded, kernel)
                probs_smooth = probs_smooth.reshape(B, V, T).permute(0, 2, 1)

            # Mask to real frames only (exclude padding)
            max_T = log_probs.shape[1]
            frame_mask = (
                torch.arange(max_T, device=log_probs.device).unsqueeze(0)
                < input_lengths.unsqueeze(1)
            )  # (B, T)

            # KL(p_smooth || p_original) — gradient flows through log_probs only
            kl_elements = probs_smooth * (probs_smooth.clamp(min=1e-7).log() - log_probs)
            kl_per_frame = kl_elements.sum(dim=-1)               # (B, T)
            kl_per_frame = kl_per_frame * frame_mask.float()
            sr_loss = kl_per_frame.sum() / frame_mask.float().sum()

            loss = ctc_loss + sr_beta * sr_loss
        else:
            loss = ctc_loss

        outputs.loss = loss
        if _specaug_adjusted:
            model.wavlm.config.mask_time_prob = self._default_mask_time_prob
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Guard against None batches and CUDA OOM.

        None batch: collator dropped all samples → return zero loss.
        CUDA OOM: free failed computation graph, log diagnostics,
        return zero loss so training continues on next batch.
        """
        if inputs is None:
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)
        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            import gc
            # Log diagnostics before cleanup
            vram_mb = (
                torch.cuda.max_memory_allocated() / 1e6
                if torch.cuda.is_available() else 0
            )
            batch_info = ""
            if isinstance(inputs, dict):
                iv = inputs.get("input_values")
                if iv is not None:
                    batch_info = f", batch_shape={tuple(iv.shape)}"
            self._oom_count += 1
            log.warning(
                "[OOM] CUDA out of memory — skipping batch #%d "
                "(peak_vram=%.0f MB%s). Freeing graph and continuing.",
                self._oom_count, vram_mb, batch_info,
            )
            # Release the failed computation graph
            del inputs
            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Save output_lengths + metadata before they're lost, then delegate."""
        # attention_mask.sum(-1) gives raw audio sample counts per item;
        # _get_feat_extract_output_lengths maps to post-CNN frame counts.
        attn = inputs.get("attention_mask")
        if attn is not None:
            with torch.no_grad():
                raw_lengths = attn.sum(-1)
                out_lengths = model._get_feat_extract_output_lengths(
                    raw_lengths,
                ).cpu().numpy()
            self._eval_output_lengths.append(out_lengths)

        # Buffer metadata for per-group metrics (popped before forward)
        age = inputs.pop("age_buckets", None)
        ds = inputs.pop("datasets", None)
        if age is not None:
            self._eval_age_buckets.append(age)
        if ds is not None:
            self._eval_datasets.append(ds)

        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys,
        )

    def evaluate(self, *args, **kwargs):
        """Wrap evaluate to clear per-eval buffers."""
        self._eval_output_lengths = []
        self._eval_age_buckets = []
        self._eval_datasets = []
        return super().evaluate(*args, **kwargs)


# ---------------------------------------------------------------------------
# HFSFTTrainer — public API
# ---------------------------------------------------------------------------

class HFSFTTrainer:
    """Single-stage HF Trainer for WavLM CTC fine-tuning.

    Parameters
    ----------
    cfg : dict
        Global config with ``hf_sft`` sub-dict and ``paths`` sub-dict.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._sft = cfg["hf_sft"]
        # model.py reads cfg["sft"] — build a canonical view once
        self._model_cfg: dict[str, Any] = {**cfg, "sft": cfg["hf_sft"]}

        _seed_everything(self._sft["seed"])

        # ---- Tokenizer ----
        self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            cfg["paths"]["tokenizer"],
        )

        # ---- Collators ----
        # ---- Max-duration config drift guard ----
        _sft_max_dur = self._sft.get("max_duration")
        _eda_max_dur = cfg.get("eda", {}).get("max_duration")
        if _sft_max_dur and _eda_max_dur and float(_sft_max_dur) != float(_eda_max_dur):
            raise ValueError(
                f"Config drift: hf_sft.max_duration={_sft_max_dur} != "
                f"eda.max_duration={_eda_max_dur}"
            )
        collator_max_dur = _sft_max_dur or _eda_max_dur

        # Augmentation toggles — when False, zero out the corresponding probs
        _speed_on = self._sft.get("speed_augment", True)
        _noise_on = self._sft.get("noise_augment", True)
        _pitch_on = self._sft.get("pitch_augment", True)

        self._train_collator = SFTCollator(
            self._tokenizer,
            target_sr=16_000,
            min_duration_sec=self._sft.get("min_duration_sec", 1.0),
            max_duration_sec=float(collator_max_dur) if collator_max_dur else None,
            speed_perturb=self._sft.get("speed_perturb", False) and _speed_on,
            speed_perturb_range=tuple(self._sft.get("speed_perturb_range", [0.9, 1.1])),
            noise_dir=self._sft.get("noise_dir") if _noise_on else None,
            noise_prob=self._sft.get("noise_prob", 0.0) if _noise_on else 0.0,
            rir_dir=self._sft.get("rir_dir") if _noise_on else None,
            rir_prob=self._sft.get("rir_prob", 0.0) if _noise_on else 0.0,
            pitch_prob=self._sft.get("pitch_prob", 0.0) if _pitch_on else 0.0,
            pitch_semitones=self._sft.get("pitch_semitones", 2.0),
            noise_datasets=self._sft.get("noise_augment_datasets"),
            pitch_datasets=self._sft.get("pitch_augment_datasets"),
            pitch_semitones_per_dataset=self._sft.get("pitch_semitones_per_dataset"),
            silence_trim=self._sft.get("silence_trim", False),
            silence_trim_db=self._sft.get("silence_trim_db", -40.0),
            silence_trim_abs_floor=self._sft.get("silence_trim_abs_floor", 0.0),
        )
        self._val_collator = SFTCollator(
            self._tokenizer,
            target_sr=16_000,
            min_duration_sec=self._sft.get("min_duration_sec", 1.0),
            max_duration_sec=float(collator_max_dur) if collator_max_dur else None,
            silence_trim=self._sft.get("silence_trim", False),
            silence_trim_db=self._sft.get("silence_trim_db", -40.0),
            silence_trim_abs_floor=self._sft.get("silence_trim_abs_floor", 0.0),
        )

        # ---- Datasets ----
        processed = cfg["paths"]["processed"]
        audio_dirs = cfg["paths"]["audio_dirs"]
        ds_oversample = self._sft.get("ds_oversample", None)
        self._train_ds = SFTDataset(
            f"{processed}/sft_train.jsonl", audio_dirs, split="train",
            ds_oversample=ds_oversample,
        )
        self._val_ds = SFTDataset(
            f"{processed}/sft_val.jsonl", audio_dirs, split="val",
        )

        # ---- Model — build then set initial freeze state ----
        self._model = build_model(self._model_cfg)
        _head_warmup = self._sft.get("warmup_head_only_steps", 0)
        if _head_warmup > 0:
            # Stage 1: only lm_head trainable for first N steps.
            # Encoder unfreezes via _HeadWarmupCallback.on_step_begin.
            freeze_for_stage(self._model, stage=1)
            log.info(
                "[HF-TRAINER] Head warmup: %d steps head-only (Stage 1), "
                "then full unfreeze (Stage 3)",
                _head_warmup,
            )
        else:
            # No warmup — unfreeze everything from step 0
            freeze_for_stage(
                self._model, stage=3,
                unfreeze_cnn=self._sft.get("unfreeze_cnn", False),
            )
        # Enable gradient checkpointing for VRAM savings
        self._model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        # Patch WavLM masking to be torch.compile-safe (out-of-place ops)
        _patch_wavlm_mask_hidden_states(self._model)

        log.info(
            "[HF-TRAINER] Ready — train=%d, val=%d, model=%s",
            len(self._train_ds), len(self._val_ds),
            self._sft["model_name"],
        )

    def train(self) -> dict[str, Any]:
        """Run single-stage HF training loop with LLRD + LR floor.

        Returns
        -------
        dict[str, Any]
            Best evaluation metrics.
        """
        sft = self._sft
        model = self._model

        # ---- Force-disable torch.compile / dynamo / inductor ----
        # CTC models with variable-length inputs are incompatible with
        # torch.compile (inductor crashes on dynamic shapes).  Disable
        # at every level to be absolutely sure.
        import torch._dynamo
        torch._dynamo.config.disable = True
        try:
            import torch._inductor.config
            torch._inductor.config.compile_threads = 0
        except (ImportError, AttributeError):
            pass

        # ---- Hardware ----
        torch.backends.cuda.matmul.allow_tf32 = sft["tf32"]
        torch.backends.cudnn.allow_tf32 = sft["tf32"]
        # Disable cuDNN deterministic for speed (~5–15% uplift).
        # Seeds still ensure statistical reproducibility.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # ---- Output directory ----
        ckpt_dir = Path(self._cfg["paths"]["models"]) / "hf_sft_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ---- Compute total steps ----
        pbs = sft["physical_batch_size"]
        accum = sft["gradient_accumulation_steps"]
        eff_batch = pbs * accum
        steps_per_epoch = math.ceil(len(self._train_ds) / eff_batch)
        total_steps = sft["max_epochs"] * steps_per_epoch
        warmup_steps = sft["warmup_steps"]

        # ---- AMP ----
        use_bf16 = sft.get("bf16", False)
        use_fp16 = sft.get("fp16", False) and not use_bf16

        # ---- W&B ----
        wb_cfg = sft.get("wandb", {})
        report_to = "wandb" if wb_cfg.get("enabled") else "none"

        # ---- LLRD + LR floor config ----
        llrd_decay = sft.get("llrd_decay", 1.0)
        lr_min_ratio = sft.get("lr_min_ratio", 0.01)

        # ---- TrainingArguments ----
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            overwrite_output_dir=True,

            # Training
            num_train_epochs=sft["max_epochs"],
            per_device_train_batch_size=pbs,
            per_device_eval_batch_size=pbs,
            gradient_accumulation_steps=accum,
            learning_rate=sft["encoder_lr"],
            weight_decay=sft["weight_decay"],
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=sft.get("max_grad_norm", 1.0),

            # LR schedule — cosine with floor (overridden in create_scheduler)
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",

            # Precision
            fp16=use_fp16,
            bf16=use_bf16,
            tf32=sft["tf32"],

            # Eval & save — once per epoch
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=sft.get("save_top_k", 3),
            load_best_model_at_end=True,
            metric_for_best_model="per",
            greater_is_better=False,

            # Logging
            logging_strategy="steps",
            logging_steps=50,
            logging_first_step=True,
            report_to=report_to,
            run_name=wb_cfg.get("run_name"),

            # DataLoader
            dataloader_num_workers=sft["num_workers"],
            dataloader_pin_memory=sft.get("pin_memory", True),
            dataloader_persistent_workers=sft.get("persistent_workers", True),
            dataloader_prefetch_factor=sft.get("prefetch_factor", 4),
            dataloader_drop_last=True,
            group_by_length=False,   # we override get_train_dataloader with LengthGroupedSampler

            # Misc
            seed=sft["seed"],
            remove_unused_columns=False,
            label_names=["labels"],

            # torch.compile — disabled for CTC (variable-length inputs
            # cause constant recompilation or CUDA graph capture failures).
            # When compile is off, omit backend to prevent inductor from
            # initialising its 32-thread compile-worker pool (~11 GB wasted).
            torch_compile=sft.get("torch_compile", False),
            **({
                "torch_compile_mode": sft.get("torch_compile_mode"),
                "torch_compile_backend": sft.get("torch_compile_backend", "inductor"),
            } if sft.get("torch_compile", False) else {}),
        )

        # ---- Build LLRD parameter groups via model.py ----
        # Must build groups with all params visible (Stage 3), then re-freeze
        # if head warmup is active.  get_parameter_groups filters by
        # requires_grad, so params frozen in Stage 1 would be excluded.
        _head_warmup_steps = sft.get("warmup_head_only_steps", 0)
        if _head_warmup_steps > 0:
            freeze_for_stage(model, stage=3, unfreeze_cnn=sft.get("unfreeze_cnn", False))
        param_groups = get_parameter_groups(model, stage=3, cfg=self._model_cfg)
        if _head_warmup_steps > 0:
            freeze_for_stage(model, stage=1)

        # Capture for the nested class
        _param_groups = param_groups
        _lr_min_ratio = lr_min_ratio
        _warmup_steps = warmup_steps

        class _LRGroupedCTCTrainer(_CTCTrainer):
            """LLRD + LR floor + custom DataLoaders."""

            def __init__(self, *args, train_collator=None, val_collator=None,
                         age_loss_weights=None, sr_ctc_beta=0.0,
                         ds1_mask_time_prob=None, default_mask_time_prob=0.30,
                         **kwargs):
                super().__init__(*args, **kwargs)
                self._train_collator = train_collator
                self._val_collator = val_collator
                self._age_loss_weights = age_loss_weights
                self._sr_ctc_beta = float(sr_ctc_beta)
                self._ds1_mask_time_prob = ds1_mask_time_prob
                self._default_mask_time_prob = default_mask_time_prob

            def create_optimizer(self):
                if self.optimizer is not None:
                    return self.optimizer

                self.optimizer = torch.optim.AdamW(
                    _param_groups,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )

                # Log LLRD per-layer LR table
                layer_lrs = [(g["name"], g["lr"]) for g in _param_groups
                             if g["name"].startswith("enc_L")]
                if layer_lrs:
                    parts = [f"{name}={lr:.2e}" for name, lr in layer_lrs]
                    log.info("[LLRD] Per-layer LR: %s", ", ".join(parts))

                # Log param groups
                for g in _param_groups:
                    n_params = sum(p.numel() for p in g["params"])
                    log.info(
                        "[HF-TRAINER] Optimizer group '%s': %s params, lr=%.2e, wd=%.4f",
                        g["name"], f"{n_params:,}", g["lr"],
                        g.get("weight_decay", 0.0),
                    )

                return self.optimizer

            def create_scheduler(self, num_training_steps, optimizer=None):
                """Cosine schedule with LR floor — each group decays from its own LR.

                lr(t) = lr_min + (base_lr - lr_min)/2 * (1 + cos(π * progress))

                Where lr_min = base_lr * lr_min_ratio.
                """
                opt = optimizer or self.optimizer

                # Pin initial_lr so cosine decays from each group's own LR
                for g in opt.param_groups:
                    g.setdefault("initial_lr", g["lr"])

                warmup = _warmup_steps
                total = num_training_steps
                min_ratio = _lr_min_ratio

                def lr_lambda(current_step: int) -> float:
                    # Linear warmup (start from 1/warmup, not zero)
                    if current_step < warmup:
                        return (current_step + 1) / warmup
                    # Cosine decay with floor
                    decay_steps = max(total - warmup, 1)
                    progress = min((current_step - warmup) / decay_steps, 1.0)
                    cosine_f = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return min_ratio + (1.0 - min_ratio) * cosine_f

                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt, lr_lambda, last_epoch=-1,
                )
                return self.lr_scheduler

            def get_train_dataloader(self) -> DataLoader:
                """Use LengthGroupedSampler + our custom collator."""
                sampler = LengthGroupedSampler(
                    batch_size=self.args.per_device_train_batch_size,
                    lengths=self.train_dataset.input_lengths,
                )
                kw: dict[str, Any] = {}
                n_workers = self.args.dataloader_num_workers
                if n_workers > 0:
                    kw["prefetch_factor"] = self.args.dataloader_prefetch_factor
                    kw["persistent_workers"] = self.args.dataloader_persistent_workers
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    sampler=sampler,
                    collate_fn=self._train_collator,
                    num_workers=n_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    drop_last=True,
                    **kw,
                )

            def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
                """Length-grouped eval DataLoader — minimises padding waste.

                Without length grouping, every batch is padded to the
                longest sample (~20 s = 320 K samples → 1000 WavLM
                frames).  Eager self-attention is O(T²), so padding
                inflates eval cost ~4-6× vs length-grouped batches.
                """
                ds = eval_dataset if eval_dataset is not None else self.eval_dataset
                kw: dict[str, Any] = {}
                n_workers = self.args.dataloader_num_workers
                if n_workers > 0:
                    kw["prefetch_factor"] = self.args.dataloader_prefetch_factor
                    kw["persistent_workers"] = False  # eval runs once/epoch — no need to keep workers alive
                sampler = LengthGroupedSampler(
                    batch_size=self.args.per_device_eval_batch_size,
                    lengths=ds.input_lengths,
                )
                return DataLoader(
                    ds,
                    batch_size=self.args.per_device_eval_batch_size,
                    sampler=sampler,
                    collate_fn=self._val_collator,
                    num_workers=n_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    drop_last=False,
                    **kw,
                )

        # ---- Compute metrics function ----
        tokenizer = self._tokenizer
        blank_id = sft["blank_id"]
        unk_id = tokenizer.unk_token_id  # 1 for Wav2Vec2CTCTokenizer

        # ---- Preprocess logits to argmax on GPU (saves massive RAM) ----
        def preprocess_logits_for_metrics(logits, labels):
            logits = logits.clone()
            logits[:, :, unk_id] = -1e9  # suppress UNK before argmax — matches submission
            return logits.argmax(dim=-1)  # (B, T)

        # IPA post-processing filter — must match submission exactly
        _VALID = set(
            " bcdefghijklmnoprstuvwxz"
            "\u00e6\u00e7\u00f0\u014b\u0250\u0251\u0254\u0259\u025a\u025b\u025f\u026a\u026b\u026c\u0279\u027e\u0281\u0283\u028a\u028c\u0292\u0294\u029d\u02a4\u02a7\u02d0\u03b8\u03c7"
        )
        _SPACE_RE = re.compile(r"\s+")

        def compute_metrics(eval_pred):
            pred_ids = eval_pred.predictions  # numpy (B, T_padded)
            labels = eval_pred.label_ids      # numpy (B, L)

            # ---- Trim predictions to real output lengths ----
            # prediction_step() saved per-batch output_lengths in the
            # _eval_output_lengths buffer.  Concatenate and use them to
            # avoid decoding padding frames (which can inject spurious
            # tokens from cross-batch -100 padding or intra-batch blank
            # ambiguity).
            output_lengths = None
            if hasattr(trainer, "_eval_output_lengths") and trainer._eval_output_lengths:
                output_lengths = np.concatenate(trainer._eval_output_lengths)

            # ---- CTC diagnostics (computed on real frames only) ----
            total_real_frames = 0
            blank_frames = 0

            # CTC greedy decode: collapse repeats → remove blanks (vectorised)
            hyps: list[list[int]] = []
            for i in range(pred_ids.shape[0]):
                T = int(output_lengths[i]) if output_lengths is not None else pred_ids.shape[1]
                seq = pred_ids[i, :T]
                total_real_frames += T
                blank_frames += int(np.count_nonzero(seq == blank_id))

                if T == 0:
                    hyps.append([])
                    continue
                change = np.empty(T, dtype=np.bool_)
                change[0] = True
                change[1:] = seq[1:] != seq[:-1]
                collapsed = seq[change]
                filtered = collapsed[(collapsed != blank_id) & (collapsed != unk_id)]
                hyps.append(filtered.tolist())

            blank_ratio = blank_frames / max(total_real_frames, 1)

            # Extract refs: strip -100 padding using numpy masking
            refs: list[list[int]] = []
            for i in range(labels.shape[0]):
                row = labels[i]
                refs.append(row[row != -100].tolist())

            per, recall, error_counts, top_confusions = compute_per_and_recall(hyps, refs)

            # Decoded sequence lengths (after CTC collapse)
            mean_hyp_len = sum(len(h) for h in hyps) / max(len(hyps), 1)
            mean_ref_len = sum(len(r) for r in refs) / max(len(refs), 1)

            # Mean argmax run length (collapse quality proxy)
            total_runs = 0
            total_run_frames = 0
            for i in range(pred_ids.shape[0]):
                T = int(output_lengths[i]) if output_lengths is not None else pred_ids.shape[1]
                if T == 0:
                    continue
                seq = pred_ids[i, :T]
                changes = int(np.count_nonzero(seq[1:] != seq[:-1]))
                total_runs += changes + 1
                total_run_frames += T
            mean_run_len = total_run_frames / max(total_runs, 1)

            # CER via jiwer (same as leaderboard) — decode + IPA filter matching submission
            def _decode_and_clean(ids):
                raw = tokenizer.decode(ids, group_tokens=False)
                safe = "".join(c for c in raw if c in _VALID)
                return _SPACE_RE.sub(" ", safe).strip()

            hyp_strs = [_decode_and_clean(h) for h in hyps]
            ref_strs = [_decode_and_clean(r) for r in refs]
            valid_refs, valid_hyps = [], []
            for r, h in zip(ref_strs, hyp_strs):
                if r.strip():
                    valid_refs.append(r)
                    valid_hyps.append(h)
            cer = jiwer.cer(valid_refs, valid_hyps) if valid_refs else 0.0

            metrics: dict[str, float] = {
                "per": per,
                "cer": cer,
                "blank_ratio": blank_ratio,
                "mean_hyp_len": mean_hyp_len,
                "mean_ref_len": mean_ref_len,
                "mean_run_len": mean_run_len,
                "n_del": float(error_counts["del"]),
                "n_ins": float(error_counts["ins"]),
                "n_sub": float(error_counts["sub"]),
            }

            # ---- Single-pass grouping for per-age / per-dataset / cross / CER / length ----
            age_buckets = None
            if hasattr(trainer, "_eval_age_buckets") and trainer._eval_age_buckets:
                age_buckets = [b for batch in trainer._eval_age_buckets for b in batch]
            datasets = None
            if hasattr(trainer, "_eval_datasets") and trainer._eval_datasets:
                datasets = [d for batch in trainer._eval_datasets for d in batch]

            has_age = age_buckets is not None and len(age_buckets) == len(hyps)
            has_ds = datasets is not None and len(datasets) == len(hyps)
            has_cross = has_age and has_ds

            # Pre-allocate all group dicts (filled in one loop)
            age_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
            ds_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
            cross_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
            age_cer_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
            len_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

            for i in range(len(hyps)):
                h, r = hyps[i], refs[i]

                # Length bucket
                n = len(r)
                if n <= 5:
                    lbucket = "short_le5"
                elif n <= 15:
                    lbucket = "mid_6to15"
                else:
                    lbucket = "long_gt15"
                len_groups[lbucket][0].append(h)
                len_groups[lbucket][1].append(r)

                if has_age:
                    a = age_buckets[i]
                    age_groups[a][0].append(h)
                    age_groups[a][1].append(r)
                    # Per-age CER (only valid refs)
                    if i < len(ref_strs) and ref_strs[i].strip():
                        age_cer_groups[a][0].append(hyp_strs[i])
                        age_cer_groups[a][1].append(ref_strs[i])

                if has_ds:
                    d = datasets[i]
                    ds_groups[d][0].append(h)
                    ds_groups[d][1].append(r)

                if has_cross:
                    key = f"{datasets[i]}_{age_buckets[i]}"
                    cross_groups[key][0].append(h)
                    cross_groups[key][1].append(r)

            # Emit per-age PER
            if has_age:
                for age, (a_hyps, a_refs) in sorted(age_groups.items()):
                    metrics[f"per_age/{age}"] = compute_per_batch(a_hyps, a_refs)
                    metrics[f"hyp_len_age/{age}"] = sum(len(h) for h in a_hyps) / max(len(a_hyps), 1)
                    metrics[f"ref_len_age/{age}"] = sum(len(r) for r in a_refs) / max(len(a_refs), 1)

            # Emit per-dataset PER
            if has_ds:
                for ds, (d_hyps, d_refs) in sorted(ds_groups.items()):
                    metrics[f"per_ds/{ds}"] = compute_per_batch(d_hyps, d_refs)

            # Emit age×dataset cross PER
            if has_cross:
                cross_parts = []
                for key, (c_hyps, c_refs) in sorted(cross_groups.items()):
                    p = compute_per_batch(c_hyps, c_refs)
                    metrics[f"per_ds_age/{key}"] = p
                    cross_parts.append(f"DS{key}={p:.4f}(n={len(c_refs)})")
                log.info("[EVAL] Age×DS PER: %s", "  ".join(cross_parts))

            # Emit per-age CER
            if has_age:
                for age, (a_h, a_r) in sorted(age_cer_groups.items()):
                    if a_r:
                        metrics[f"cer_age/{age}"] = jiwer.cer(a_r, a_h)

            # Emit length-bucketed PER
            for bucket, (l_hyps, l_refs) in sorted(len_groups.items()):
                metrics[f"per_len/{bucket}"] = compute_per_batch(l_hyps, l_refs)

            # ---- Per-phoneme recall + dead phonemes ----
            n_dead = 0
            worst_k = 10
            worst = sorted(recall.values(), key=lambda x: x["recall"])[:worst_k]
            for info in recall.values():
                if info["total"] > 0 and info["recall"] == 0.0:
                    n_dead += 1
            metrics["dead_phonemes"] = float(n_dead)
            if worst:
                metrics["worst_phoneme_recall"] = worst[0]["recall"]
            # Log worst-K phonemes for visibility
            if recall:
                worst_items = sorted(recall.items(), key=lambda x: x[1]["recall"])[:worst_k]
                parts = [
                    f"{tokenizer.convert_ids_to_tokens(tid)}({tid})={info['recall']:.2f}"
                    f"({info['hits']}/{info['total']})"
                    for tid, info in worst_items
                ]
                log.info("[EVAL] Worst %d phonemes: %s", len(parts), ", ".join(parts))

            # ---- Alignment logging (5 random REF/HYP pairs) ----
            n_align = min(5, len(hyps))
            indices = random.sample(range(len(hyps)), n_align)
            for idx in indices:
                ref_str = ref_strs[idx] if idx < len(ref_strs) else "?"
                hyp_str = hyp_strs[idx] if idx < len(hyp_strs) else "?"
                sample_per = compute_per_batch([hyps[idx]], [refs[idx]])
                log.info(
                    "[ALIGN] sample %d (PER=%.3f)\n  REF: %s\n  HYP: %s",
                    idx, sample_per, ref_str, hyp_str,
                )

            # ---- Error breakdown + confusion pairs ----
            total_errors = error_counts["del"] + error_counts["ins"] + error_counts["sub"]
            if total_errors > 0:
                log.info(
                    "[EVAL] Errors: %d del (%.1f%%), %d ins (%.1f%%), %d sub (%.1f%%)",
                    error_counts["del"], 100 * error_counts["del"] / total_errors,
                    error_counts["ins"], 100 * error_counts["ins"] / total_errors,
                    error_counts["sub"], 100 * error_counts["sub"] / total_errors,
                )
            if top_confusions:
                parts = [
                    f"{tokenizer.convert_ids_to_tokens(r)}→{tokenizer.convert_ids_to_tokens(h)}:{c}"
                    for (r, h), c in top_confusions
                ]
                log.info("[EVAL] Top confusions: %s", ", ".join(parts))

            return metrics

        # ---- Callbacks ----
        # First-batch CTC constraint check: output_len >= target_len
        # (CTC requires output_len >= target_len; violation = silent NaN loss)
        class _CTCConstraintCheck(TrainerCallback):
            """Run once before the first step to verify CTC feasibility."""

            def __init__(self):
                super().__init__()
                self._checked = False

            def on_train_begin(self, args, state, control, model=None, **kwargs):
                if self._checked:
                    return
                self._checked = True
                # Grab a single batch from the train dataloader
                try:
                    import time
                    t0 = time.perf_counter()
                    log.info("[CTC-CHECK] Preflight with val collator (no augmentation)...")

                    # Use val collator (no augmentation) + num_workers=0 for
                    # fast preflight.  CTC constraint (output_len >= target_len)
                    # depends only on audio length vs label length, not augmentation.
                    dl = DataLoader(
                        trainer.train_dataset,
                        batch_size=trainer.args.per_device_train_batch_size,
                        sampler=SequentialSampler(trainer.train_dataset),
                        collate_fn=trainer._val_collator,
                        num_workers=0,
                        drop_last=True,
                    )
                    batch = next(iter(dl))
                    log.info("[CTC-CHECK] Preflight batch fetched in %.1fs",
                             time.perf_counter() - t0)
                    if batch is None:
                        return
                    attn = batch.get("attention_mask")
                    labels = batch.get("labels")
                    if attn is None or labels is None:
                        return
                    with torch.no_grad():
                        raw_lens = attn.sum(-1)
                        out_lens = model._get_feat_extract_output_lengths(raw_lens)
                    target_lens = (labels != -100).sum(-1)
                    violations = (out_lens < target_lens).sum().item()
                    total = labels.size(0)
                    if violations > 0:
                        log.warning(
                            "[CTC-CHECK] %d/%d samples VIOLATE output_len >= target_len! "
                            "Min output_len=%d, max target_len=%d. "
                            "Check min_duration / max label length.",
                            violations, total,
                            out_lens.min().item(), target_lens.max().item(),
                        )
                    else:
                        log.info(
                            "[CTC-CHECK] OK — all %d samples satisfy "
                            "output_len >= target_len (min_out=%d, max_tgt=%d)",
                            total, out_lens.min().item(), target_lens.max().item(),
                        )
                    log.info(
                        "[CTC-CHECK] Preflight total time: %.1fs",
                        time.perf_counter() - t0,
                    )
                except Exception as e:
                    log.warning("[CTC-CHECK] Skipped — %s", e)

        # ---- Overfitting Alert Callback ----
        class _OverfittingAlertCallback(TrainerCallback):
            """
            Monitor training health at each evaluation and emit alerts for:
            1. Train/eval loss divergence (overfitting signal)
            2. DS1 PER plateau or regression
            3. DS1-DS2 gap widening (distribution-level overfitting)
            4. Overall PER plateau
            """
            def __init__(self):
                self.history: list[dict] = []
                self.best_per = float('inf')
                self.best_ds1_per = float('inf')
                self.epochs_without_improvement = 0

            def on_log(self, args, state, control, logs=None, **kwargs):
                # Capture training loss from logs
                if logs and "loss" in logs:
                    self._last_train_loss = logs["loss"]

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if not metrics:
                    return

                epoch = state.epoch or 0
                eval_loss = metrics.get("eval_loss", 0)
                eval_per = metrics.get("eval_per", 1.0)
                ds1_per = metrics.get("eval_per_ds/1", None)
                ds2_per = metrics.get("eval_per_ds/2", None)
                train_loss = getattr(self, "_last_train_loss", None)

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "eval_per": eval_per,
                    "ds1_per": ds1_per,
                    "ds2_per": ds2_per,
                }
                self.history.append(record)

                alerts = []
                warnings = []

                # ---- Alert 1: Train/Eval loss divergence ----
                if train_loss is not None and len(self.history) >= 2:
                    gap = eval_loss - train_loss
                    prev_gap = (self.history[-2].get("eval_loss", 0) -
                                (self.history[-2].get("train_loss") or 0))
                    if gap > 0.15:
                        alerts.append(f"OVERFITTING: eval_loss - train_loss = {gap:.3f} (>0.15)")
                    elif gap > 0.08:
                        warnings.append(f"Mild divergence: eval-train gap = {gap:.3f}")

                # ---- Alert 2: DS1 PER tracking ----
                if ds1_per is not None:
                    if ds1_per < self.best_ds1_per:
                        improvement = self.best_ds1_per - ds1_per
                        self.best_ds1_per = ds1_per
                        log.info(
                            "[HEALTH] ✓ DS1 improving: %.4f (↓%.4f from best)",
                            ds1_per, improvement
                        )
                    else:
                        regression = ds1_per - self.best_ds1_per
                        if regression > 0.02:
                            alerts.append(f"DS1 REGRESSION: {ds1_per:.4f} vs best {self.best_ds1_per:.4f} (+{regression:.4f})")
                        elif regression > 0.01:
                            warnings.append(f"DS1 stagnant: {ds1_per:.4f} vs best {self.best_ds1_per:.4f}")

                # ---- Alert 3: DS1-DS2 gap widening ----
                if ds1_per is not None and ds2_per is not None:
                    gap = ds1_per - ds2_per
                    if len(self.history) >= 2:
                        prev = self.history[-2]
                        if prev.get("ds1_per") and prev.get("ds2_per"):
                            prev_gap = prev["ds1_per"] - prev["ds2_per"]
                            gap_delta = gap - prev_gap
                            if gap_delta > 0.015:
                                alerts.append(
                                    f"GAP WIDENING: DS1-DS2 gap increased by {gap_delta:.4f} "
                                    f"({prev_gap:.3f} → {gap:.3f})"
                                )

                # ---- Alert 4: Overall PER plateau ----
                if eval_per < self.best_per - 0.001:
                    self.best_per = eval_per
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= 3:
                        warnings.append(
                            f"PER plateau: {self.epochs_without_improvement} epochs without improvement"
                        )

                # ---- Emit alerts ----
                if alerts:
                    log.warning(
                        "[HEALTH] ⚠️  EPOCH %.1f ALERTS:\n  • %s",
                        epoch, "\n  • ".join(alerts)
                    )
                if warnings:
                    log.info(
                        "[HEALTH] ⚡ Epoch %.1f warnings:\n  • %s",
                        epoch, "\n  • ".join(warnings)
                    )

                # ---- Summary line ----
                ds1_str = f"DS1={ds1_per:.4f}" if ds1_per else "DS1=N/A"
                ds2_str = f"DS2={ds2_per:.4f}" if ds2_per else "DS2=N/A"
                gap_str = f"gap={ds1_per-ds2_per:.4f}" if (ds1_per and ds2_per) else ""
                log.info(
                    "[HEALTH] Epoch %.1f: PER=%.4f | %s | %s %s | best_PER=%.4f | best_DS1=%.4f",
                    epoch, eval_per, ds1_str, ds2_str, gap_str, self.best_per, self.best_ds1_per
                )

        # ---- Startup Logging Callback ----
        class _StartupLoggingCallback(TrainerCallback):
            """Log phase transitions during the silent startup gap."""
            def __init__(self):
                self._first_step_done = False
                self._train_start_time = None

            def on_train_begin(self, args, state, control, **kwargs):
                import time
                self._train_start_time = time.time()
                log.info("[STARTUP] DataLoader workers forking (%d workers)...",
                         args.dataloader_num_workers)

            def on_step_begin(self, args, state, control, **kwargs):
                if not self._first_step_done:
                    import time
                    elapsed = time.time() - self._train_start_time if self._train_start_time else 0
                    log.info("[STARTUP] First batch ready (%.1fs). "
                             "Running forward + backward...", elapsed)

            def on_step_end(self, args, state, control, **kwargs):
                if not self._first_step_done:
                    import time
                    elapsed = time.time() - self._train_start_time if self._train_start_time else 0
                    log.info("[STARTUP] First step complete (%.1fs). "
                             "Training is running normally.", elapsed)
                    self._first_step_done = True

        # ---- Head Warmup Callback ----
        _warmup_head_steps = sft.get("warmup_head_only_steps", 0)
        _unfreeze_cnn = sft.get("unfreeze_cnn", False)

        class _HeadWarmupCallback(TrainerCallback):
            """CTC head warmup: train only lm_head for N steps, then unfreeze.

            Medin et al. 2024 protocol: 1000 steps head-only before
            unfreezing the encoder.  Prevents random CTC head gradients
            from corrupting encoder features during early training.
            """

            def __init__(self):
                super().__init__()
                self._unfrozen = False

            def on_step_begin(self, args, state, control, model=None, **kwargs):
                if self._unfrozen or _warmup_head_steps <= 0:
                    return
                if state.global_step >= _warmup_head_steps:
                    freeze_for_stage(model, stage=3, unfreeze_cnn=_unfreeze_cnn)
                    self._unfrozen = True
                    n_train = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                    log.info(
                        "[HEAD-WARMUP] Step %d: unfroze encoder (Stage 3). "
                        "Trainable params: %s",
                        state.global_step, f"{n_train:,}",
                    )

        # ---- VRAM Defragmentation Callback ----
        class _VRAMCleanupCallback(TrainerCallback):
            """Defragment CUDA memory at epoch boundaries.

            Variable-length dynamic padding creates tensors of different
            sizes every batch.  After thousands of alloc/free cycles,
            PyTorch's caching allocator fragments, reducing usable VRAM.
            gc.collect() + empty_cache() returns cached blocks to the
            driver, defragmenting the pool.

            Also runs after evaluation (which allocates large temporary
            prediction buffers that can fragment the pool).
            """

            def on_epoch_end(self, args, state, control, **kwargs):
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    before = torch.cuda.memory_allocated() / 1e6
                    torch.cuda.empty_cache()
                    after = torch.cuda.memory_allocated() / 1e6
                    log.info(
                        "[VRAM] Epoch %d cleanup: %.0f MB → %.0f MB allocated",
                        int(state.epoch), before, after,
                    )

            def on_train_end(self, args, state, control, **kwargs):
                # Access _oom_count from the trainer we close over
                oom = getattr(trainer_ref[0], "_oom_count", 0)
                total = state.global_step or 1
                pct = oom / total * 100
                if oom > 0:
                    log.warning(
                        "[OOM] Training finished with %d OOM skips out of "
                        "%d steps (%.2f%%). Batches >1%% = biased training.",
                        oom, total, pct,
                    )
                else:
                    log.info("[OOM] Training finished with 0 OOM skips.")

            def on_evaluate(self, args, state, control, **kwargs):
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # trainer_ref: mutable container so _VRAMCleanupCallback.on_train_end
        # can access the trainer's _oom_count via closure.
        trainer_ref: list = [None]

        # ---- Email notification callback ----
        _email_cfg = sft.get("email", {})
        _email_cb = None
        if _email_cfg.get("enabled"):
            _email_cb = EmailNotificationCallback(_email_cfg)
            log.info("[EMAIL] Notifications enabled → %s", _email_cfg["recipient"])

        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=sft.get("early_stopping_patience", 8),
            ),
            _StartupLoggingCallback(),
            _CTCConstraintCheck(),
            _HeadWarmupCallback(),
            _OverfittingAlertCallback(),
            _VRAMCleanupCallback(),
        ]
        if _email_cb:
            callbacks.append(_email_cb)

        # ---- Build trainer ----
        # ---- Age-based loss weighting ----
        _age_weights_raw = sft.get("age_loss_weights")
        _age_loss_weights = (
            {str(k): float(v) for k, v in _age_weights_raw.items()}
            if _age_weights_raw else None
        )
        if _age_loss_weights:
            log.info("[HF-TRAINER] Age loss weights: %s", _age_loss_weights)

        # ---- SR-CTC regularization ----
        _sr_ctc_beta = float(sft.get("sr_ctc_beta", 0.0))
        if _sr_ctc_beta > 0:
            log.info("[HF-TRAINER] SR-CTC enabled: beta=%.3f", _sr_ctc_beta)

        # ---- DS1-specific SpecAug reduction ----
        _ds1_mask_time_prob_raw = sft.get("ds1_mask_time_prob")
        _ds1_mask_time_prob = float(_ds1_mask_time_prob_raw) if _ds1_mask_time_prob_raw is not None else None
        _default_mask_time_prob = float(sft.get("mask_time_prob", 0.30))
        if _ds1_mask_time_prob is not None:
            log.info(
                "[HF-TRAINER] DS1 SpecAug reduction: mask_time_prob %.2f → %.2f for DS1 batches",
                _default_mask_time_prob, _ds1_mask_time_prob,
            )

        trainer = _LRGroupedCTCTrainer(
            model=model,
            args=training_args,
            train_dataset=self._train_ds,
            eval_dataset=self._val_ds,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=callbacks,
            train_collator=self._train_collator,
            val_collator=self._val_collator,
            age_loss_weights=_age_loss_weights,
            sr_ctc_beta=_sr_ctc_beta,
            ds1_mask_time_prob=_ds1_mask_time_prob,
            default_mask_time_prob=_default_mask_time_prob,
        )
        trainer_ref[0] = trainer

        # ---- Persist full config alongside checkpoints ----
        config_path = ckpt_dir / "training_config.json"
        config_path.write_text(json.dumps(self._cfg, indent=2, default=str))
        log.info("[HF-TRAINER] Config saved to %s", config_path)

        # ---- Train ----
        llrd_info = f"llrd={llrd_decay}" if llrd_decay < 1.0 else "no-llrd"
        log.info(
            "[HF-TRAINER] Starting — epochs=%d, pbs=%d, accum=%d, eff=%d, "
            "encoder_lr=%.2e, head_lr=%.2e, warmup=%d, total_steps~%d, "
            "%s, lr_floor=%.0e",
            sft["max_epochs"], pbs, accum, eff_batch,
            sft["encoder_lr"], sft["head_lr"], warmup_steps, total_steps,
            llrd_info,
            sft["encoder_lr"] * lr_min_ratio,
        )
        log.info(
            "[STARTUP] Entering trainer.train(); next log may take 1-3 minutes "
            "while first training batch is prepared by DataLoader workers."
        )

        try:
            result = trainer.train(
                resume_from_checkpoint=sft.get("resume_from"),
            )
        except Exception as exc:
            if _email_cb:
                try:
                    _email_cb.send_failure_alert(exc)
                except Exception:
                    log.warning("[EMAIL] Could not send failure alert")
            raise

        # ---- Save best model ----
        best_ckpt = ckpt_dir / "best"
        trainer.save_model(str(best_ckpt))
        log.info("[HF-TRAINER] Best model saved to %s", best_ckpt)

        # ---- Final eval (best single checkpoint) ----
        final_metrics = trainer.evaluate()
        best_per = final_metrics.get("eval_per", float("inf"))
        log.info("[HF-TRAINER] Final eval (best checkpoint): %s", final_metrics)

        # ---- Post-training checkpoint averaging (log-only) ----
        if sft.get("checkpoint_averaging", False):
            self._average_and_evaluate(
                trainer, ckpt_dir, best_per,
                top_n=sft.get("checkpoint_avg_top_n", 5),
            )

        return {
            "train_result": {
                k: v for k, v in result._asdict().items()
                if isinstance(v, (int, float, str))
            },
            **final_metrics,
        }

    # ------------------------------------------------------------------
    # Standalone checkpoint averaging (--avg entry point)
    # ------------------------------------------------------------------

    def avg_only(self, *, top_n: int = 5) -> dict[str, Any]:
        """Average top-N checkpoints, save to ``avg/``, and evaluate.

        Designed to be called mid-run (after killing training) or
        post-training via ``python pipeline.py --avg``.

        Returns
        -------
        dict[str, Any]
            Eval metrics of the averaged model.
        """
        import gc

        sft = self._sft
        ckpt_dir = Path(self._cfg["paths"]["models"]) / "hf_sft_checkpoints"

        # ---- Rank & average ----
        ranked = self._rank_checkpoints(ckpt_dir)
        if len(ranked) < 2:
            log.error("[AVG] Need >= 2 checkpoints in %s (have %d)", ckpt_dir, len(ranked))
            return {}

        use_n = min(top_n, len(ranked))
        selected = ranked[:use_n]
        log.info("[AVG] Averaging %d checkpoints (by eval PER):", use_n)
        for path, per in selected:
            log.info("[AVG]   %s  PER=%.4f", path.name, per)

        avg_state = self._load_and_average([p for p, _ in selected])

        # ---- Cast back to original dtype ----
        ref_path = selected[0][0] / "model.safetensors"
        if ref_path.exists():
            from safetensors.torch import load_file
            ref_sd = load_file(str(ref_path), device="cpu")
            for k in avg_state:
                avg_state[k] = avg_state[k].to(ref_sd[k].dtype)
            del ref_sd

        # ---- Save ----
        avg_dir = ckpt_dir / "avg"
        avg_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file
        out_path = avg_dir / "model.safetensors"
        save_file(avg_state, str(out_path))

        config_src = selected[0][0] / "config.json"
        if config_src.exists():
            shutil.copy2(config_src, avg_dir / "config.json")

        meta = {
            "source_checkpoints": [
                {"path": str(p.name), "eval_per": per} for p, per in selected
            ],
            "top_n_requested": top_n,
            "top_n_used": use_n,
            "best_single_per": selected[0][1],
        }
        (avg_dir / "averaging_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )
        log.info(
            "[AVG] Saved averaged model → %s (%.1f MB)",
            avg_dir, out_path.stat().st_size / 1024 / 1024,
        )

        # ---- Load averaged weights into model ----
        self._model.load_state_dict(avg_state, strict=True)
        del avg_state
        gc.collect()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()

        # ---- Build a minimal Trainer for evaluation ----
        torch.backends.cuda.matmul.allow_tf32 = sft.get("tf32", True)
        torch.backends.cudnn.allow_tf32 = sft.get("tf32", True)

        pbs = sft["physical_batch_size"]
        eval_args = TrainingArguments(
            output_dir=str(ckpt_dir / "_avg_eval"),
            per_device_eval_batch_size=pbs,
            bf16=sft.get("bf16", False),
            tf32=sft.get("tf32", True),
            dataloader_num_workers=sft.get("num_workers", 4),
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to="none",
        )

        tokenizer = self._tokenizer
        blank_id = sft["blank_id"]
        unk_id = tokenizer.unk_token_id

        def preprocess_logits_for_metrics(logits, labels):
            logits = logits.clone()
            logits[:, :, unk_id] = -1e9
            return logits.argmax(dim=-1)

        _VALID = set(
            " bcdefghijklmnoprstuvwxz"
            "\u00e6\u00e7\u00f0\u014b\u0250\u0251\u0254\u0259\u025a\u025b\u025f\u026a\u026b\u026c\u0279\u027e\u0281\u0283\u028a\u028c\u0292\u0294\u029d\u02a4\u02a7\u02d0\u03b8\u03c7"
        )
        _SPACE_RE = re.compile(r"\s+")

        def compute_metrics(eval_pred):
            pred_ids = eval_pred.predictions
            labels = eval_pred.label_ids

            output_lengths = None
            if hasattr(eval_trainer, "_eval_output_lengths") and eval_trainer._eval_output_lengths:
                output_lengths = np.concatenate(eval_trainer._eval_output_lengths)

            total_real_frames = 0
            blank_frames = 0
            hyps: list[list[int]] = []
            for i in range(pred_ids.shape[0]):
                T = int(output_lengths[i]) if output_lengths is not None else pred_ids.shape[1]
                seq = pred_ids[i, :T]
                total_real_frames += T
                blank_frames += int(np.count_nonzero(seq == blank_id))
                if T == 0:
                    hyps.append([])
                    continue
                change = np.empty(T, dtype=np.bool_)
                change[0] = True
                change[1:] = seq[1:] != seq[:-1]
                collapsed = seq[change]
                filtered = collapsed[(collapsed != blank_id) & (collapsed != unk_id)]
                hyps.append(filtered.tolist())

            refs: list[list[int]] = []
            for i in range(labels.shape[0]):
                row = labels[i]
                refs.append(row[row != -100].tolist())

            per, recall, error_counts, top_confusions = compute_per_and_recall(hyps, refs)

            def _decode_and_clean(ids):
                raw = tokenizer.decode(ids, group_tokens=False)
                safe = "".join(c for c in raw if c in _VALID)
                return _SPACE_RE.sub(" ", safe).strip()

            hyp_strs = [_decode_and_clean(h) for h in hyps]
            ref_strs = [_decode_and_clean(r) for r in refs]
            valid_refs, valid_hyps = [], []
            for r, h in zip(ref_strs, hyp_strs):
                if r.strip():
                    valid_refs.append(r)
                    valid_hyps.append(h)
            cer = jiwer.cer(valid_refs, valid_hyps) if valid_refs else 0.0

            # Per-dataset PER
            ds_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
            datasets = None
            if hasattr(eval_trainer, "_eval_datasets") and eval_trainer._eval_datasets:
                datasets = [d for batch in eval_trainer._eval_datasets for d in batch]
            if datasets and len(datasets) == len(hyps):
                for i in range(len(hyps)):
                    ds_groups[datasets[i]][0].append(hyps[i])
                    ds_groups[datasets[i]][1].append(refs[i])

            metrics = {
                "per": per,
                "cer": cer,
                "n_del": float(error_counts["del"]),
                "n_ins": float(error_counts["ins"]),
                "n_sub": float(error_counts["sub"]),
            }
            for ds_key, (d_h, d_r) in sorted(ds_groups.items()):
                metrics[f"per_ds/{ds_key}"] = compute_per_batch(d_h, d_r)

            # Log summary
            log.info(
                "[AVG-EVAL] PER=%.4f | CER=%.4f | del=%d ins=%d sub=%d",
                per, cer, error_counts["del"], error_counts["ins"], error_counts["sub"],
            )
            if ds_groups:
                parts = [f"DS{k}={compute_per_batch(h, r):.4f}" for k, (h, r) in sorted(ds_groups.items())]
                log.info("[AVG-EVAL] %s", " | ".join(parts))

            # Top confusions
            if top_confusions:
                top10 = top_confusions[:10]
                parts = [f"{tokenizer.convert_ids_to_tokens(s)}→{tokenizer.convert_ids_to_tokens(t)}:{c}" for (s, t), c in top10]
                log.info("[AVG-EVAL] Top confusions: %s", ", ".join(parts))

            return metrics

        eval_trainer = _CTCTrainer(
            model=self._model,
            args=eval_args,
            eval_dataset=self._val_ds,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # Attach val collator
        eval_trainer._val_collator = self._val_collator

        # Override get_eval_dataloader for length-grouped eval
        def _get_eval_dl(self_t, eval_dataset=None):
            ds = eval_dataset if eval_dataset is not None else self_t.eval_dataset
            sampler = LengthGroupedSampler(
                batch_size=self_t.args.per_device_eval_batch_size,
                lengths=ds.input_lengths,
            )
            return DataLoader(
                ds,
                batch_size=self_t.args.per_device_eval_batch_size,
                sampler=sampler,
                collate_fn=self_t._val_collator,
                num_workers=self_t.args.dataloader_num_workers,
                pin_memory=True,
                drop_last=False,
            )
        eval_trainer.get_eval_dataloader = types.MethodType(_get_eval_dl, eval_trainer)

        log.info("[AVG] Running evaluation on val set (%d samples)...", len(self._val_ds))
        avg_metrics = eval_trainer.evaluate()

        avg_per = avg_metrics.get("eval_per", float("inf"))
        best_per = selected[0][1]
        delta = best_per - avg_per
        log.info(
            "[AVG] RESULT: avg PER=%.4f | best-single PER=%.4f | delta=%.4f (%s)",
            avg_per, best_per, abs(delta),
            "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME",
        )

        # Cleanup
        del eval_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_metrics

    # ------------------------------------------------------------------
    # Post-training checkpoint averaging
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_checkpoints(ckpt_dir: Path) -> list[tuple[Path, float]]:
        """Return ``[(checkpoint_path, eval_per), ...]`` sorted best-first.

        Reads ``trainer_state.json`` from each ``checkpoint-*`` directory.
        HF Trainer writes one ``trainer_state.json`` per checkpoint containing
        the full ``log_history`` up to that point.  We extract the *last*
        ``eval_per`` entry for that checkpoint's own epoch.
        """
        ranked: list[tuple[Path, float]] = []
        for d in ckpt_dir.glob("checkpoint-*"):
            if not d.is_dir():
                continue
            ts = d / "trainer_state.json"
            if not ts.exists():
                continue
            # Must have weights
            if not (d / "model.safetensors").exists() and not (d / "pytorch_model.bin").exists():
                continue
            try:
                state = json.loads(ts.read_text())
                # log_history is a list of dicts; eval entries have "eval_per"
                per = None
                for entry in reversed(state.get("log_history", [])):
                    if "eval_per" in entry:
                        per = entry["eval_per"]
                        break
                if per is not None:
                    ranked.append((d, per))
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning("[CKPT-AVG] Skipping %s — corrupt trainer_state.json", d.name)
        ranked.sort(key=lambda x: x[1])
        return ranked

    @staticmethod
    def _load_and_average(checkpoint_dirs: list[Path]) -> OrderedDict:
        """Uniform weight averaging of state dicts (fp32 accumulation)."""
        N = len(checkpoint_dirs)
        avg_state: OrderedDict | None = None
        keys: list[str] | None = None

        for i, d in enumerate(checkpoint_dirs):
            st_path = d / "model.safetensors"
            bin_path = d / "pytorch_model.bin"
            if st_path.exists():
                from safetensors.torch import load_file
                sd = load_file(str(st_path), device="cpu")
            else:
                sd = torch.load(bin_path, map_location="cpu", weights_only=True)

            n_params = sum(v.numel() for v in sd.values())
            log.info(
                "[CKPT-AVG]   [%d/%d] %s: %d keys, %s params",
                i + 1, N, d.name, len(sd), f"{n_params:,}",
            )

            if avg_state is None:
                keys = list(sd.keys())
                avg_state = OrderedDict()
                for k in keys:
                    avg_state[k] = sd[k].float()
            else:
                if set(sd.keys()) != set(keys):
                    missing = set(keys) - set(sd.keys())
                    extra = set(sd.keys()) - set(keys)
                    raise RuntimeError(
                        f"Key mismatch in {d.name}: missing={missing}, extra={extra}"
                    )
                for k in keys:
                    avg_state[k] += sd[k].float()
            del sd

        for k in keys:
            avg_state[k] /= N

        return avg_state

    def _average_and_evaluate(
        self,
        trainer: Trainer,
        ckpt_dir: Path,
        best_single_per: float,
        *,
        top_n: int = 5,
    ) -> None:
        """Average top-N checkpoints, save to ``avg/``, evaluate, and compare.

        Results are log-only. The trainer's model is left with avg weights
        loaded (caller returns immediately after, so the in-memory state
        doesn't matter).

        Parameters
        ----------
        trainer
            The HF Trainer (needed for evaluate).
        ckpt_dir
            Directory containing ``checkpoint-*`` sub-directories.
        best_single_per
            PER of the best single checkpoint (for comparison logging).
        top_n
            Number of best checkpoints to average.
        """
        log.info("[CKPT-AVG] === Post-Training Checkpoint Averaging ===")

        # ---- Rank all saved checkpoints by eval PER ----
        ranked = self._rank_checkpoints(ckpt_dir)
        if not ranked:
            log.warning("[CKPT-AVG] No valid checkpoints found — skipping averaging.")
            return

        available = len(ranked)
        use_n = min(top_n, available)
        if available < top_n:
            log.warning(
                "[CKPT-AVG] Requested top-%d but only %d checkpoints available. "
                "Using all %d.",
                top_n, available, available,
            )
        if use_n < 2:
            log.warning(
                "[CKPT-AVG] Need >= 2 checkpoints to average (have %d) — skipping.",
                use_n,
            )
            return

        selected = ranked[:use_n]
        log.info("[CKPT-AVG] Selected %d checkpoints (by eval PER):", use_n)
        for path, per in selected:
            log.info("[CKPT-AVG]   %s  PER=%.4f", path.name, per)

        # ---- Average weights ----
        try:
            avg_state = self._load_and_average([p for p, _ in selected])
        except RuntimeError as e:
            log.error("[CKPT-AVG] Averaging failed: %s", e)
            return

        # ---- Cast back to original dtype ----
        ref_path = selected[0][0] / "model.safetensors"
        if ref_path.exists():
            from safetensors.torch import load_file
            ref_sd = load_file(str(ref_path), device="cpu")
            for k in avg_state:
                avg_state[k] = avg_state[k].to(ref_sd[k].dtype)
            del ref_sd

        # ---- Save averaged model ----
        avg_dir = ckpt_dir / "avg"
        avg_dir.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file
            out_path = avg_dir / "model.safetensors"
            save_file(avg_state, str(out_path))
        except ImportError:
            out_path = avg_dir / "pytorch_model.bin"
            torch.save(avg_state, str(out_path))

        # Copy config.json so the HF model can be loaded later
        config_src = selected[0][0] / "config.json"
        if config_src.exists():
            shutil.copy2(config_src, avg_dir / "config.json")

        # Save averaging metadata
        meta = {
            "source_checkpoints": [
                {"path": str(p.name), "eval_per": per} for p, per in selected
            ],
            "top_n_requested": top_n,
            "top_n_used": use_n,
            "best_single_per": best_single_per,
        }
        (avg_dir / "averaging_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )

        n_params = sum(v.numel() for v in avg_state.values())
        log.info(
            "[CKPT-AVG] Saved averaged model → %s (%s params, %.1f MB)",
            avg_dir, f"{n_params:,}",
            out_path.stat().st_size / 1024 / 1024,
        )
        del avg_state

        # ---- Load averaged weights into the trainer's model & evaluate ----
        log.info("[CKPT-AVG] Loading averaged weights for evaluation...")
        device = trainer.args.device
        if (avg_dir / "model.safetensors").exists():
            from safetensors.torch import load_file
            avg_sd = load_file(str(avg_dir / "model.safetensors"), device="cpu")
        else:
            avg_sd = torch.load(
                avg_dir / "pytorch_model.bin", map_location="cpu", weights_only=True,
            )
        trainer.model.load_state_dict(avg_sd, strict=True)
        trainer.model.to(device)
        del avg_sd

        # Run full eval — same compute_metrics, same val set, same logging
        avg_metrics = trainer.evaluate()

        avg_per = avg_metrics.get("eval_per", float("inf"))
        delta = best_single_per - avg_per
        log.info(
            "[CKPT-AVG] Averaged eval: PER=%.4f (best-single=%.4f, delta=%.4f %s)",
            avg_per, best_single_per, abs(delta),
            "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME",
        )
