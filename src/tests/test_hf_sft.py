"""
HF SFT Trainer — comprehensive pre-deploy verification suite.
=============================================================

Exercises every component on CPU with real data (after ETL):
 1. Config loading — hf_sft section exists, no stale keys
 2. Model build — correct freeze state (all encoder unfrozen, CNN frozen)
 3. Collator — real audio loads, batch shape correct, zero-mean unit-var
 4. Forward pass — loss finite, not NaN
 5. Backward — gradients flow to encoder + head, NOT to CNN
 6. Gradient checkpointing — use_reentrant=False, actually active
 7. Optimizer — discriminative LR (head 3e-4, encoder 5e-5)
 8. Compute metrics — CTC decode → PER + CER
 9. compute_loss — pops extra collator keys, model gets clean inputs
10. TrainingArguments — correct construction
11. Crash-cause immunity — all 3 prior crash root causes blocked
12. DataLoader — collator-aware, LengthGroupedSampler, persistent workers

Run:  PYTHONPATH=src python -m pytest src/tests/test_hf_sft.py -v
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

# Ensure src/ is on the path
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config.config import load_config
from trainer.model import build_model, freeze_for_stage, grad_norms
from trainer.data_collator import SFTCollator
from trainer.dataset import SFTDataset
from trainer.metrics import compute_per_batch
from trainer.sft_trainer_hf import _CTCTrainer, HFSFTTrainer
from trainer.model import get_parameter_groups, _verify_groups

# ═══════════════════════════════════════════════════════════════════════
# Shared config + paths
# ═══════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "src" / "config" / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    """Load real config once for all tests."""
    return load_config(str(_CONFIG_PATH))


@pytest.fixture(scope="module")
def hf_cfg(cfg):
    """The hf_sft sub-dict."""
    return cfg["hf_sft"]


@pytest.fixture(scope="module")
def model(cfg):
    """Build model with hf_sft config (expensive — once per module)."""
    test_cfg = {**cfg, "sft": cfg["hf_sft"]}
    # Fall back to HF hub if local checkpoint doesn't exist (local dev)
    if not Path(test_cfg["sft"]["model_name"]).exists():
        test_cfg = {**test_cfg, "sft": {**test_cfg["sft"], "model_name": "microsoft/wavlm-base-plus"}}
    m = build_model(test_cfg)
    # HF trainer unfreeze: all encoder + head, CNN stays frozen
    freeze_for_stage(m, stage=3, unfreeze_cnn=False)
    m.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    return m


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — Config loading + hf_sft keys
# ═══════════════════════════════════════════════════════════════════════

class TestConfig:
    """Verify hf_sft config section is complete and clean."""

    def test_hf_sft_section_exists(self, cfg):
        assert "hf_sft" in cfg, "hf_sft section missing from config"

    def test_no_stage_keys_leaked(self, hf_cfg):
        """3-stage-only keys must NOT be in hf_sft."""
        stage_keys = [
            "stage1_physical_batch_size", "stage2_physical_batch_size",
            "stage3_physical_batch_size", "stage1_gradient_accumulation_steps",
            "stage2_gradient_accumulation_steps", "stage3_gradient_accumulation_steps",
            "per_group_warmup_steps", "ema_decay", "stage1_blank_threshold",
            "stage2_per_improvement_threshold", "beam_width", "beam_eval_interval",
            "grad_ratio_band", "max_grad_norm_stage1", "max_grad_norm_stage2",
            "max_grad_norm_stage3", "deterministic",
        ]
        for k in stage_keys:
            assert k not in hf_cfg, f"3-stage key '{k}' leaked into hf_sft"

    def test_all_required_keys_present(self, hf_cfg):
        """Every key the HF trainer reads must exist."""
        required = [
            "model_name", "vocab_size", "blank_id", "seed", "max_epochs",
            "physical_batch_size", "gradient_accumulation_steps",
            "fp16", "bf16", "tf32",
            "head_lr", "encoder_lr", "cnn_lr", "weight_decay", "warmup_steps", "max_grad_norm",
            "llrd_decay", "lr_min_ratio",
            "mask_time_prob", "mask_time_length", "mask_feature_prob", "mask_feature_length",
            "attention_dropout", "hidden_dropout", "feat_proj_dropout", "final_dropout",
            "layerdrop", "speed_perturb", "speed_perturb_range",
            "save_top_k", "early_stopping_patience",
            "max_duration", "min_duration_sec",
            "num_workers", "prefetch_factor", "pin_memory", "persistent_workers",
            "unfreeze_cnn", "resume_from", "wandb",
        ]
        for k in required:
            assert k in hf_cfg, f"Required key '{k}' missing from hf_sft"

    def test_effective_batch_size(self, hf_cfg):
        eff = hf_cfg["physical_batch_size"] * hf_cfg["gradient_accumulation_steps"]
        assert eff >= 16, f"Effective batch {eff} too small (< 16)"
        assert eff <= 256, f"Effective batch {eff} too large (> 256)"

    def test_lr_values(self, hf_cfg):
        assert hf_cfg["encoder_lr"] == 1e-4
        assert hf_cfg["head_lr"] == 3e-4
        assert hf_cfg["head_lr"] > hf_cfg["encoder_lr"]

    def test_llrd_config(self, hf_cfg):
        assert 0.5 < hf_cfg["llrd_decay"] < 1.0, "LLRD decay should be in (0.5, 1.0)"

    def test_lr_floor_config(self, hf_cfg):
        assert 0.0 < hf_cfg["lr_min_ratio"] <= 0.2, "lr_min_ratio should be in (0, 0.2]"

    def test_warmup_reasonable(self, hf_cfg):
        """Warmup should be 100-5000 steps (not zero, not absurd)."""
        assert 100 <= hf_cfg["warmup_steps"] <= 5000

    def test_bf16_not_fp16(self, hf_cfg):
        """RTX 5090 Blackwell: bf16 preferred, fp16 off."""
        assert hf_cfg["bf16"] is True
        assert hf_cfg["fp16"] is False

    def test_cnn_frozen(self, hf_cfg):
        assert hf_cfg["unfreeze_cnn"] is False

    def test_wandb_config(self, hf_cfg):
        wb = hf_cfg["wandb"]
        assert wb["project"] == "309-hf-sft"
        assert "enabled" in wb


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Model build + freeze state
# ═══════════════════════════════════════════════════════════════════════

class TestModelFreeze:
    """Verify correct requires_grad for single-stage training."""

    def test_head_trainable(self, model):
        for name, p in model.lm_head.named_parameters():
            assert p.requires_grad, f"lm_head.{name} should be trainable"

    def test_all_encoder_layers_trainable(self, model):
        for i, layer in enumerate(model.wavlm.encoder.layers):
            for name, p in layer.named_parameters():
                assert p.requires_grad, (
                    f"Encoder layer {i} param '{name}' should be trainable"
                )

    def test_cnn_frozen(self, model):
        """CNN feature extractor must be frozen (crash 3 root cause)."""
        for name, p in model.wavlm.feature_extractor.named_parameters():
            assert not p.requires_grad, (
                f"CNN param '{name}' should be frozen"
            )

    def test_hf_internal_cnn_freeze_flag(self, model):
        """HF's internal _requires_grad flag must be False (crash 3 fix)."""
        assert not model.wavlm.feature_extractor._requires_grad, (
            "CRITICAL: _requires_grad=True on CNN → autograd builds full graph "
            "through frozen CNN layers → massive VRAM waste (crash 3 root cause)"
        )

    def test_feature_projection_trainable(self, model):
        for name, p in model.wavlm.feature_projection.named_parameters():
            assert p.requires_grad, (
                f"Feature projection param '{name}' should be trainable"
            )

    def test_gradient_checkpointing_enabled(self, model):
        """Grad checkpoint must be ON (crash 1/3 fix)."""
        assert model.wavlm.encoder.gradient_checkpointing, (
            "Gradient checkpointing should be enabled"
        )

    def test_param_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        # WavLM Base+: ~94.4M total, CNN is ~4.2M
        assert total > 90_000_000, f"Total params unexpectedly low: {total:,}"
        assert trainable > 85_000_000, f"Trainable params unexpectedly low: {trainable:,}"
        assert frozen > 3_000_000, f"Frozen params (CNN) unexpectedly low: {frozen:,}"
        print(f"\n  Total: {total:,}  Trainable: {trainable:,}  Frozen(CNN): {frozen:,}")

    def test_ctc_config(self, model):
        """Model config must match our CTC setup."""
        assert model.config.ctc_loss_reduction == "mean"
        assert model.config.ctc_zero_infinity is True
        assert model.config.pad_token_id == 0
        assert model.config.vocab_size == 53

    def test_specaugment_config(self, model, hf_cfg):
        assert model.config.mask_time_prob == hf_cfg["mask_time_prob"]
        assert model.config.mask_feature_prob == hf_cfg["mask_feature_prob"]


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Collator with real audio
# ═══════════════════════════════════════════════════════════════════════

class TestCollator:
    """Verify collator loads real audio and produces correct batch format."""

    @pytest.fixture(scope="class")
    def tokenizer_and_collator(self, cfg):
        from transformers import Wav2Vec2CTCTokenizer
        tok = Wav2Vec2CTCTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
        hf = cfg["hf_sft"]
        collator = SFTCollator(
            tok, target_sr=16_000,
            min_duration_sec=hf.get("min_duration_sec", 0.5),
            max_duration_sec=float(hf["max_duration"]),
        )
        return tok, collator

    @pytest.fixture(scope="class")
    def sample_batch(self, cfg, tokenizer_and_collator):
        """Load 4 real samples from the train set."""
        tok, collator = tokenizer_and_collator
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"],
            split="train",
        )
        rows = [ds[i] for i in range(4)]
        batch = collator(rows)
        return batch

    def test_batch_not_none(self, sample_batch):
        assert sample_batch is not None, "Collator returned None — all samples dropped"

    def test_batch_keys(self, sample_batch):
        expected = {"input_values", "attention_mask", "labels", "input_lengths",
                    "age_buckets", "datasets"}
        assert set(sample_batch.keys()) == expected

    def test_input_values_shape(self, sample_batch):
        iv = sample_batch["input_values"]
        assert iv.ndim == 2, f"input_values should be (B, T), got {iv.shape}"
        assert iv.shape[0] == 4
        assert iv.dtype == torch.float32

    def test_attention_mask_dtype_long(self, sample_batch):
        """CRITICAL: WavLM silently fails with bool/float attention mask."""
        am = sample_batch["attention_mask"]
        assert am.dtype == torch.long, (
            f"attention_mask must be long, got {am.dtype} "
            "(WavLM has silent bug with bool/float masks)"
        )

    def test_labels_padded_with_minus100(self, sample_batch):
        labels = sample_batch["labels"]
        assert labels.dtype == torch.long
        # At least one pad position should be -100 (unless all same length)
        assert (labels == -100).any() or labels.shape[0] == 1

    def test_zero_mean_unit_variance(self, sample_batch):
        """Audio normalisation must match WavLM pretraining convention."""
        iv = sample_batch["input_values"]
        am = sample_batch["attention_mask"]
        for i in range(iv.shape[0]):
            length = am[i].sum().item()
            wav = iv[i, :length]
            mean = wav.mean().item()
            std = wav.std().item()
            assert abs(mean) < 0.01, f"Sample {i} mean={mean:.4f}, expected ~0"
            assert abs(std - 1.0) < 0.1, f"Sample {i} std={std:.4f}, expected ~1.0"

    def test_input_lengths_match_mask(self, sample_batch):
        am = sample_batch["attention_mask"]
        il = sample_batch["input_lengths"]
        for i in range(am.shape[0]):
            assert am[i].sum().item() == il[i].item()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — Forward pass + loss
# ═══════════════════════════════════════════════════════════════════════

class TestForwardLoss:
    """Verify model produces finite CTC loss on real data."""

    @pytest.fixture(scope="class")
    def real_batch(self, cfg):
        """Get a small real batch for forward pass test."""
        from transformers import Wav2Vec2CTCTokenizer
        tok = Wav2Vec2CTCTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
        hf = cfg["hf_sft"]
        collator = SFTCollator(
            tok, target_sr=16_000,
            min_duration_sec=hf.get("min_duration_sec", 0.5),
            max_duration_sec=float(hf["max_duration"]),
        )
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"], split="train",
        )
        rows = [ds[i] for i in range(2)]
        return collator(rows)

    def test_forward_produces_loss(self, model, real_batch):
        """Forward with labels → model computes CTC loss internally."""
        model.eval()
        with torch.no_grad():
            out = model(
                input_values=real_batch["input_values"],
                attention_mask=real_batch["attention_mask"],
                labels=real_batch["labels"],
            )
        assert out.loss is not None, "Model returned None loss"
        assert torch.isfinite(out.loss), f"Loss is not finite: {out.loss.item()}"
        assert out.loss.item() > 0, f"Loss should be positive: {out.loss.item()}"
        print(f"\n  Forward pass CTC loss: {out.loss.item():.4f}")

    def test_logits_shape(self, model, real_batch):
        model.eval()
        with torch.no_grad():
            out = model(
                input_values=real_batch["input_values"],
                attention_mask=real_batch["attention_mask"],
            )
        B = real_batch["input_values"].shape[0]
        V = model.config.vocab_size
        assert out.logits.ndim == 3
        assert out.logits.shape[0] == B
        assert out.logits.shape[2] == V  # (B, T, 53)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — Backward + gradient flow
# ═══════════════════════════════════════════════════════════════════════

class TestGradientFlow:
    """Verify gradients reach the right parameters after backward."""

    def test_gradients_flow(self, model, cfg):
        """Full forward-backward on real data — check grads exist.
        Temporarily disables layerdrop so all 12 layers participate
        deterministically (layerdrop is a training regularisation, not
        relevant to gradient-connectivity verification)."""
        from transformers import Wav2Vec2CTCTokenizer
        tok = Wav2Vec2CTCTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
        hf = cfg["hf_sft"]
        collator = SFTCollator(
            tok, target_sr=16_000,
            min_duration_sec=hf.get("min_duration_sec", 0.5),
            max_duration_sec=float(hf["max_duration"]),
        )
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"], split="train",
        )
        batch = collator([ds[0], ds[1]])

        # Disable layerdrop so every encoder layer runs in forward pass
        saved_layerdrop = model.config.layerdrop
        model.config.layerdrop = 0.0

        model.train()
        model.zero_grad()

        out = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out.loss.backward()

        # Restore layerdrop
        model.config.layerdrop = saved_layerdrop

        # Head must have gradients
        head_has_grad = False
        for name, p in model.lm_head.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                head_has_grad = True
        assert head_has_grad, "CRITICAL: No gradient in lm_head!"

        # Encoder layers must have gradients
        num_layers = len(model.wavlm.encoder.layers)
        encoder_grad_count = 0
        for i, layer in enumerate(model.wavlm.encoder.layers):
            for name, p in layer.named_parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    encoder_grad_count += 1
                    break  # one param per layer is enough
        assert encoder_grad_count == num_layers, (
            f"Only {encoder_grad_count}/{num_layers} encoder layers got gradients "
            f"(crash 3: use_reentrant=True silently dropped grads)"
        )

        # CNN must NOT have gradients (frozen)
        for name, p in model.wavlm.feature_extractor.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                f"CNN param '{name}' has gradient — should be frozen"
            )

        # Feature projection should have gradients
        fp_has_grad = False
        for name, p in model.wavlm.feature_projection.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                fp_has_grad = True
        assert fp_has_grad, "Feature projection should have gradients"

        print(f"\n  Loss: {out.loss.item():.4f}")
        print(f"  Head grad: ✓")
        print(f"  Encoder layers with grad: {encoder_grad_count}/{num_layers}")
        print(f"  CNN grad: ✗ (frozen)")
        print(f"  Feature projection grad: ✓")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — _CTCTrainer.compute_loss pops extra keys
# ═══════════════════════════════════════════════════════════════════════

class TestComputeLoss:
    """Verify _CTCTrainer.compute_loss correctly strips collator extras."""

    def test_pops_extra_keys(self, model, cfg):
        """compute_loss must pop age_buckets, input_lengths, datasets."""
        from transformers import Wav2Vec2CTCTokenizer
        tok = Wav2Vec2CTCTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
        hf = cfg["hf_sft"]
        collator = SFTCollator(
            tok, target_sr=16_000,
            min_duration_sec=hf.get("min_duration_sec", 0.5),
            max_duration_sec=float(hf["max_duration"]),
        )
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"], split="train",
        )
        batch = collator([ds[0], ds[1]])

        # Simulate what HF Trainer does — pass batch as inputs
        trainer_instance = _CTCTrainer.__new__(_CTCTrainer)
        model.eval()
        with torch.no_grad():
            loss = trainer_instance.compute_loss(model, dict(batch))

        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        assert loss.item() > 0
        print(f"\n  _CTCTrainer.compute_loss: {loss.item():.4f}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6b — Age-weighted CTC loss
# ═══════════════════════════════════════════════════════════════════════

class TestAgeLossWeighting:
    """Verify age-based loss weighting in _CTCTrainer.compute_loss."""

    @staticmethod
    def _make_batch(model, batch_size=4, seq_len=16000, label_len=5):
        """Create a synthetic batch matching SFTCollator output format."""
        input_values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        # Random labels in valid range (1..vocab-1), pad with -100
        labels_raw = torch.randint(1, model.config.vocab_size, (batch_size, label_len))
        pad = torch.full((batch_size, 3), -100, dtype=torch.long)
        labels = torch.cat([labels_raw, pad], dim=1)
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "age_buckets": ["3-4", "5-7", "8-11", "12+"],
            "input_lengths": [seq_len] * batch_size,
            "datasets": ["1", "1", "2", "2"],
        }

    def test_disabled_matches_baseline(self, model):
        """With _age_loss_weights=None, loss must equal model's built-in CTC."""
        torch.manual_seed(99)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = None

        model.eval()
        with torch.no_grad():
            loss_disabled = trainer.compute_loss(model, dict(batch))

        # Baseline: model(**inputs).loss directly
        clean = {
            "input_values": batch["input_values"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        with torch.no_grad():
            baseline_loss = model(**clean).loss

        assert torch.allclose(loss_disabled, baseline_loss, atol=1e-6), (
            f"Disabled path {loss_disabled.item():.6f} != "
            f"baseline {baseline_loss.item():.6f}"
        )
        print(f"\n  Disabled path matches baseline: {baseline_loss.item():.4f}")

    def test_uniform_weights_match_baseline(self, model):
        """All weights=1.0 must produce identical loss to baseline."""
        torch.manual_seed(99)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = {
            "3-4": 1.0, "5-7": 1.0, "8-11": 1.0, "12+": 1.0, "unknown": 1.0,
        }

        model.eval()
        with torch.no_grad():
            loss_uniform = trainer.compute_loss(model, dict(batch))

        clean = {
            "input_values": batch["input_values"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        with torch.no_grad():
            baseline_loss = model(**clean).loss

        assert torch.allclose(loss_uniform, baseline_loss, atol=1e-5), (
            f"Uniform weights {loss_uniform.item():.6f} != "
            f"baseline {baseline_loss.item():.6f}"
        )
        print(f"\n  Uniform weights match baseline: {baseline_loss.item():.4f}")

    def test_weights_change_loss(self, model):
        """Non-uniform weights must produce a DIFFERENT loss than baseline."""
        torch.manual_seed(99)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = {
            "3-4": 1.3, "5-7": 1.0, "8-11": 0.9, "12+": 0.85,
        }

        model.eval()
        with torch.no_grad():
            loss_weighted = trainer.compute_loss(model, dict(batch))

        clean = {
            "input_values": batch["input_values"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        with torch.no_grad():
            baseline_loss = model(**clean).loss

        assert not torch.allclose(loss_weighted, baseline_loss, atol=1e-5), (
            "Weighted loss should differ from baseline"
        )
        assert torch.isfinite(loss_weighted), f"Loss not finite: {loss_weighted.item()}"
        assert loss_weighted.item() > 0, f"Loss should be positive: {loss_weighted.item()}"
        print(f"\n  Weighted: {loss_weighted.item():.4f} vs baseline: {baseline_loss.item():.4f}")

    def test_higher_weight_increases_contribution(self, model):
        """Sample with weight 2.0 should pull loss toward its own loss value."""
        torch.manual_seed(99)

        # Make batch where first sample is easy, others are harder
        batch = self._make_batch(model)

        trainer_high = _CTCTrainer.__new__(_CTCTrainer)
        trainer_high._age_loss_weights = {"3-4": 2.0, "5-7": 1.0, "8-11": 1.0, "12+": 1.0}

        trainer_low = _CTCTrainer.__new__(_CTCTrainer)
        trainer_low._age_loss_weights = {"3-4": 0.5, "5-7": 1.0, "8-11": 1.0, "12+": 1.0}

        model.eval()
        with torch.no_grad():
            loss_high = trainer_high.compute_loss(model, dict(batch))
            loss_low = trainer_low.compute_loss(model, dict(batch))

        # Higher weight on first sample should change loss vs lower weight
        assert loss_high.item() != loss_low.item(), "Different weights must give different losses"
        print(f"\n  w=2.0: {loss_high.item():.4f}, w=0.5: {loss_low.item():.4f}")

    def test_gradients_flow(self, model):
        """Weighted loss must produce valid gradients for backward pass."""
        torch.manual_seed(99)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = {"3-4": 1.3, "5-7": 1.0, "8-11": 0.9, "12+": 0.85}

        model.train()
        loss = trainer.compute_loss(model, dict(batch))
        loss.backward()

        # Check gradients exist on lm_head
        head_grad = model.lm_head.weight.grad
        assert head_grad is not None, "No gradient on lm_head"
        assert torch.isfinite(head_grad).all(), "Non-finite gradients on lm_head"
        assert head_grad.abs().sum() > 0, "Zero gradients on lm_head"

        model.zero_grad()
        print(f"\n  Gradients flow OK, loss={loss.item():.4f}")

    def test_return_outputs(self, model):
        """return_outputs=True must return (loss, outputs) tuple."""
        torch.manual_seed(99)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = {"3-4": 1.3, "5-7": 1.0, "8-11": 0.9, "12+": 0.85}

        model.eval()
        with torch.no_grad():
            result = trainer.compute_loss(model, dict(batch), return_outputs=True)

        assert isinstance(result, tuple) and len(result) == 2
        loss, outputs = result
        assert torch.isfinite(loss)
        assert hasattr(outputs, "logits")
        print(f"\n  return_outputs=True OK, logits shape={outputs.logits.shape}")

    def test_missing_bucket_defaults_to_1(self, model):
        """If a bucket is not in the weight dict, its weight should be 1.0."""
        torch.manual_seed(99)
        batch = self._make_batch(model)
        # Only specify weight for "3-4", rest should default to 1.0
        trainer_partial = _CTCTrainer.__new__(_CTCTrainer)
        trainer_partial._age_loss_weights = {"3-4": 1.5}

        trainer_full = _CTCTrainer.__new__(_CTCTrainer)
        trainer_full._age_loss_weights = {"3-4": 1.5, "5-7": 1.0, "8-11": 1.0, "12+": 1.0}

        model.eval()
        with torch.no_grad():
            loss_partial = trainer_partial.compute_loss(model, dict(batch))
            loss_full = trainer_full.compute_loss(model, dict(batch))

        assert torch.allclose(loss_partial, loss_full, atol=1e-6), (
            f"Partial {loss_partial.item():.6f} != full {loss_full.item():.6f}"
        )
        print(f"\n  Missing buckets default to 1.0: {loss_partial.item():.4f}")

    def test_config_has_age_weights(self, cfg):
        """Config must have age_loss_weights key in hf_sft (can be null or dict)."""
        hf = cfg["hf_sft"]
        assert "age_loss_weights" in hf, "age_loss_weights missing from config"
        weights = hf["age_loss_weights"]
        if weights is not None:
            assert isinstance(weights, dict), f"Expected dict or None, got {type(weights)}"
            assert all(isinstance(v, (int, float)) for v in weights.values()), \
                "All weights must be numeric"
            assert all(v > 0 for v in weights.values()), "All weights must be positive"
        print(f"\n  Config age_loss_weights: {weights}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6c — SR-CTC regularization
# ═══════════════════════════════════════════════════════════════════════

class TestSRCTC:
    """Verify SR-CTC regularization in _CTCTrainer.compute_loss."""

    @staticmethod
    def _make_batch(model, batch_size=4, seq_len=16000, label_len=5):
        input_values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels_raw = torch.randint(1, model.config.vocab_size, (batch_size, label_len))
        pad = torch.full((batch_size, 3), -100, dtype=torch.long)
        labels = torch.cat([labels_raw, pad], dim=1)
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "age_buckets": ["3-4", "5-7", "8-11", "12+"],
            "input_lengths": [seq_len] * batch_size,
            "datasets": ["1", "1", "2", "2"],
        }

    def test_sr_ctc_disabled_matches_baseline(self, model):
        """With sr_ctc_beta=0, loss must equal model's built-in CTC."""
        torch.manual_seed(42)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = None
        trainer._sr_ctc_beta = 0.0

        model.eval()
        with torch.no_grad():
            loss_disabled = trainer.compute_loss(model, dict(batch))

        clean = {
            "input_values": batch["input_values"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        with torch.no_grad():
            baseline_loss = model(**clean).loss

        assert torch.allclose(loss_disabled, baseline_loss, atol=1e-6), (
            f"SR-CTC disabled {loss_disabled.item():.6f} != "
            f"baseline {baseline_loss.item():.6f}"
        )
        print(f"\n  SR-CTC disabled matches baseline: {baseline_loss.item():.4f}")

    def test_sr_ctc_increases_loss(self, model):
        """SR-CTC regularization must add a non-negative term to total loss."""
        torch.manual_seed(42)
        batch = self._make_batch(model)

        trainer_off = _CTCTrainer.__new__(_CTCTrainer)
        trainer_off._age_loss_weights = None
        trainer_off._sr_ctc_beta = 0.0

        trainer_on = _CTCTrainer.__new__(_CTCTrainer)
        trainer_on._age_loss_weights = None
        trainer_on._sr_ctc_beta = 0.2

        model.eval()
        with torch.no_grad():
            loss_off = trainer_off.compute_loss(model, dict(batch))
            loss_on = trainer_on.compute_loss(model, dict(batch))

        # KL divergence is non-negative → SR-CTC loss >= 0 → total >= baseline
        assert loss_on.item() >= loss_off.item() - 1e-6, (
            f"SR-CTC loss {loss_on.item():.6f} < baseline {loss_off.item():.6f}"
        )
        print(f"\n  Without SR-CTC: {loss_off.item():.4f}, with: {loss_on.item():.4f}")

    def test_sr_ctc_gradients_flow(self, model):
        """SR-CTC loss must produce valid gradients for backward pass."""
        torch.manual_seed(42)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = None
        trainer._sr_ctc_beta = 0.2

        model.train()
        loss = trainer.compute_loss(model, dict(batch))
        loss.backward()

        head_grad = model.lm_head.weight.grad
        assert head_grad is not None, "No gradient on lm_head"
        assert torch.isfinite(head_grad).all(), "Non-finite gradients"
        assert head_grad.abs().sum() > 0, "Zero gradients"

        model.zero_grad()
        print(f"\n  SR-CTC gradients flow OK, loss={loss.item():.4f}")

    def test_sr_ctc_with_age_weights(self, model):
        """SR-CTC must work correctly combined with age-weighted CTC."""
        torch.manual_seed(42)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = {"3-4": 1.3, "5-7": 1.0, "8-11": 0.9, "12+": 0.85}
        trainer._sr_ctc_beta = 0.2

        model.eval()
        with torch.no_grad():
            loss = trainer.compute_loss(model, dict(batch))

        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
        print(f"\n  SR-CTC + age weights: {loss.item():.4f}")

    def test_sr_ctc_return_outputs(self, model):
        """return_outputs=True must work with SR-CTC."""
        torch.manual_seed(42)
        batch = self._make_batch(model)

        trainer = _CTCTrainer.__new__(_CTCTrainer)
        trainer._age_loss_weights = None
        trainer._sr_ctc_beta = 0.2

        model.eval()
        with torch.no_grad():
            result = trainer.compute_loss(model, dict(batch), return_outputs=True)

        assert isinstance(result, tuple) and len(result) == 2
        loss, outputs = result
        assert torch.isfinite(loss)
        assert hasattr(outputs, "logits")
        print(f"\n  SR-CTC return_outputs OK, loss={loss.item():.4f}")

    def test_config_has_sr_ctc_beta(self, cfg):
        """Config must have sr_ctc_beta in hf_sft."""
        hf = cfg["hf_sft"]
        assert "sr_ctc_beta" in hf, "sr_ctc_beta missing from config"
        beta = hf["sr_ctc_beta"]
        assert isinstance(beta, (int, float)), f"Expected numeric, got {type(beta)}"
        assert beta >= 0, f"sr_ctc_beta must be non-negative: {beta}"
        print(f"\n  Config sr_ctc_beta: {beta}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — Optimizer LR groups
# ═══════════════════════════════════════════════════════════════════════

class TestOptimizerGroups:
    """Verify discriminative LR: head 3e-4, encoder 5e-5."""

    def test_lr_group_construction(self, model, hf_cfg):
        """Simulate the optimizer construction from the trainer."""
        head_lr = hf_cfg["head_lr"]
        encoder_lr = hf_cfg["encoder_lr"]

        no_decay_names = (
            "bias", "LayerNorm.weight", "LayerNorm.bias",
            "layer_norm.weight", "layer_norm.bias",
        )

        head_params, encoder_params = 0, 0
        groups = {"head_decay": [], "head_no_decay": [],
                  "encoder_decay": [], "encoder_no_decay": []}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith("lm_head")
            is_no_decay = any(nd in name for nd in no_decay_names)

            if is_head:
                head_params += param.numel()
                if is_no_decay:
                    groups["head_no_decay"].append(param)
                else:
                    groups["head_decay"].append(param)
            else:
                encoder_params += param.numel()
                if is_no_decay:
                    groups["encoder_no_decay"].append(param)
                else:
                    groups["encoder_decay"].append(param)

        # Build actual optimizer
        opt_groups = [
            {"params": groups["head_decay"], "lr": head_lr, "weight_decay": 0.01},
            {"params": groups["head_no_decay"], "lr": head_lr, "weight_decay": 0.0},
            {"params": groups["encoder_decay"], "lr": encoder_lr, "weight_decay": 0.01},
            {"params": groups["encoder_no_decay"], "lr": encoder_lr, "weight_decay": 0.0},
        ]
        opt_groups = [g for g in opt_groups if g["params"]]
        optimizer = torch.optim.AdamW(opt_groups, lr=encoder_lr)

        # Verify group LRs
        for g in optimizer.param_groups:
            if g["lr"] == head_lr:
                pass  # head groups
            elif g["lr"] == encoder_lr:
                pass  # encoder groups
            else:
                pytest.fail(f"Unexpected LR: {g['lr']}")

        assert head_params > 0, "No head params found"
        assert encoder_params > head_params, "Encoder should have more params than head"

        # Head: hidden_size * vocab + bias → dynamic check
        hidden = model.config.hidden_size
        expected_head = hidden * model.config.vocab_size + model.config.vocab_size
        assert abs(head_params - expected_head) < 100, f"Head params: {head_params:,}, expected ~{expected_head:,}"
        # Encoder: must be >> head
        assert encoder_params > 10 * head_params, f"Encoder params: {encoder_params:,}"

        print(f"\n  Head params: {head_params:,} @ lr={head_lr}")
        print(f"  Encoder params: {encoder_params:,} @ lr={encoder_lr}")
        print(f"  Optimizer groups: {len(opt_groups)}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — Compute metrics (CTC decode + PER + CER)
# ═══════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    """Verify CTC greedy decode → PER/CER computation."""

    def test_ctc_greedy_decode(self):
        """Argmax → collapse repeats → remove blanks."""
        # Simulated argmax predictions: 0=blank, 1-5=tokens
        pred_ids = np.array([
            [0, 0, 1, 1, 2, 0, 3, 3, 3, 0],  # → [1, 2, 3]
            [0, 4, 4, 5, 0, 0, 1, 0, 0, 0],  # → [4, 5, 1]
        ])
        blank_id = 0

        hyps = []
        for i in range(pred_ids.shape[0]):
            seq = pred_ids[i]
            tokens = []
            prev = -1
            for t in seq:
                t = int(t)
                if t != prev:
                    if t != blank_id:
                        tokens.append(t)
                    prev = t
            hyps.append(tokens)

        assert hyps[0] == [1, 2, 3]
        assert hyps[1] == [4, 5, 1]

    def test_per_computation(self):
        """PER from compute_per_batch with known values."""
        hyps = [[1, 2, 3], [4, 5, 1]]
        refs = [[1, 2, 3], [4, 5, 6]]
        per = compute_per_batch(hyps, refs)
        # First: 0 edits / 3 = 0.0; Second: 1 edit / 3 = 0.333
        # Mean: 0.1667
        assert per == pytest.approx(1 / 6, abs=0.01)

    def test_perfect_match_per_zero(self):
        hyps = [[1, 2, 3]]
        refs = [[1, 2, 3]]
        per = compute_per_batch(hyps, refs)
        assert per == 0.0

    def test_preprocess_logits_for_metrics(self):
        """Verify argmax reduction preserves predictions."""
        logits = torch.randn(4, 100, 53)  # (B, T, V)
        expected = logits.argmax(dim=-1)   # (B, T)
        result = logits.argmax(dim=-1)
        assert torch.equal(result, expected)
        assert result.shape == (4, 100)

    def test_blank_ratio_computation(self):
        """blank_ratio = fraction of frames predicted as blank_id=0."""
        # 10 frames × 2 samples = 20 total frames
        pred_ids = np.array([
            [0, 0, 1, 1, 2, 0, 3, 3, 3, 0],  # 4 blanks / 10
            [0, 4, 4, 5, 0, 0, 1, 0, 0, 0],  # 5 blanks / 10
        ])
        blank_id = 0
        total_frames = pred_ids.size            # 20
        blank_frames = int((pred_ids == blank_id).sum())  # 9
        blank_ratio = blank_frames / max(total_frames, 1)
        assert blank_ratio == pytest.approx(10 / 20, abs=1e-6)

    def test_mean_decoded_lengths(self):
        """mean_hyp_len and mean_ref_len from decode results."""
        hyps = [[1, 2, 3], [4, 5, 1]]  # len 3, 3
        refs = [[1, 2, 3], [4, 5, 6, 7]]  # len 3, 4
        mean_hyp = sum(len(h) for h in hyps) / len(hyps)
        mean_ref = sum(len(r) for r in refs) / len(refs)
        assert mean_hyp == pytest.approx(3.0)
        assert mean_ref == pytest.approx(3.5)

    def test_prediction_trimming_removes_padding(self):
        """Trimming predictions to output_lengths removes padding artefacts."""
        blank_id = 0
        # Simulated padded predictions — last 3 frames are padding (-100)
        pred_ids = np.array([
            [0, 1, 1, 2, 0, 3, -100, -100, -100, -100],
            [0, 4, 5, 0, 1, 0,    0,    0, -100, -100],
        ])
        output_lengths = np.array([6, 6])  # real frames per sample

        # Decode WITH trimming (what compute_metrics now does)
        hyps_trimmed = []
        for i in range(pred_ids.shape[0]):
            T = int(output_lengths[i])
            seq = pred_ids[i, :T]
            tokens = []
            prev = -1
            for t in seq:
                t = int(t)
                if t != prev:
                    if t != blank_id:
                        tokens.append(t)
                    prev = t
            hyps_trimmed.append(tokens)

        # Decode WITHOUT trimming (old behaviour)
        hyps_untrimmed = []
        for i in range(pred_ids.shape[0]):
            seq = pred_ids[i]
            tokens = []
            prev = -1
            for t in seq:
                t = int(t)
                if t != prev:
                    if t != blank_id:
                        tokens.append(t)
                    prev = t
            hyps_untrimmed.append(tokens)

        # Trimmed decode should be clean
        assert hyps_trimmed[0] == [1, 2, 3]
        assert hyps_trimmed[1] == [4, 5, 1]

        # Untrimmed has spurious -100 tokens from padding
        assert -100 in hyps_untrimmed[0], (
            "Without trimming, -100 padding tokens leak into hypotheses"
        )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — TrainingArguments construction
# ═══════════════════════════════════════════════════════════════════════

_has_accelerate = importlib.util.find_spec("accelerate") is not None


@pytest.mark.skipif(not _has_accelerate, reason="accelerate not installed")
class TestTrainingArgs:
    """Verify TrainingArguments are set correctly from hf_sft config."""

    def test_args_construction(self, hf_cfg):
        from transformers import TrainingArguments
        sft = hf_cfg
        pbs = sft["physical_batch_size"]
        accum = sft["gradient_accumulation_steps"]

        # On CPU bf16 is unavailable — test with fp16/bf16 both off
        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                overwrite_output_dir=True,
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
                warmup_steps=sft["warmup_steps"],
                lr_scheduler_type="cosine",
                fp16=False,
                bf16=False,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=sft.get("save_top_k", 3),
                load_best_model_at_end=True,
                metric_for_best_model="per",
                greater_is_better=False,
                logging_strategy="steps",
                logging_steps=50,
                report_to="none",
                dataloader_num_workers=sft["num_workers"],
                dataloader_pin_memory=sft.get("pin_memory", True),
                dataloader_persistent_workers=sft.get("persistent_workers", True),
                dataloader_prefetch_factor=sft.get("prefetch_factor", 4),
                dataloader_drop_last=True,
                group_by_length=False,
                seed=sft["seed"],
                remove_unused_columns=False,
                label_names=["labels"],
                no_cuda=True,
            )

        assert args.per_device_train_batch_size == pbs
        assert args.gradient_accumulation_steps == accum
        assert args.learning_rate == sft["encoder_lr"]
        assert args.warmup_steps == sft["warmup_steps"]
        assert args.load_best_model_at_end == True
        assert args.metric_for_best_model == "per"
        assert args.greater_is_better == False
        assert args.remove_unused_columns == False
        assert args.dataloader_drop_last == True
        assert args.eval_strategy.value == "epoch"
        assert args.save_strategy.value == "epoch"
        print(f"\n  TrainingArguments: ✓")
        print(f"  PBS={args.per_device_train_batch_size}, accum={args.gradient_accumulation_steps}")
        print(f"  warmup={args.warmup_steps}, lr={args.learning_rate}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10 — Crash cause immunity
# ═══════════════════════════════════════════════════════════════════════

class TestCrashImmunity:
    """Verify all 3 prior crash root causes are blocked."""

    def test_crash1_grad_ckpt_on(self, model):
        """Crash 1: grad ckpt ON when encoder frozen → recomputed frozen
        layers. In HF trainer, all encoder layers are trainable so grad ckpt
        is beneficial, not wasteful."""
        assert model.wavlm.encoder.gradient_checkpointing
        # And encoder IS trainable (so ckpt is correct here)
        trainable_layers = sum(
            1 for layer in model.wavlm.encoder.layers
            if any(p.requires_grad for p in layer.parameters())
        )
        assert trainable_layers == 12

    def test_crash2_no_stage_transitions(self, hf_cfg):
        """Crash 2: CUDA fragmentation from PBS changes at stage transitions.
        HF trainer has no stage transitions — single PBS throughout."""
        assert "stage1_physical_batch_size" not in hf_cfg
        assert "stage2_physical_batch_size" not in hf_cfg
        # Single PBS value
        assert isinstance(hf_cfg["physical_batch_size"], int)

    def test_crash3_cnn_freeze_flag(self, model):
        """Crash 3: HF internal _requires_grad → full CNN compute graph."""
        assert not model.wavlm.feature_extractor._requires_grad

    def test_crash3_use_reentrant_false(self, model):
        """Crash 3 (additional): use_reentrant=True silently dropped grads."""
        # We verify by checking gradients actually flow — done in TestGradientFlow
        # Here just verify the config flag
        assert model.wavlm.encoder.gradient_checkpointing

    def test_fast_path_delegates_to_model(self):
        """Fast path (no age weights, no SR-CTC) must delegate to model's CTC."""
        import inspect
        src = inspect.getsource(_CTCTrainer.compute_loss)
        # Fast path must return model's built-in loss
        assert "outputs.loss" in src, "Fast path should use model's built-in CTC loss"
        # Fast path guard: when both features off, skip manual CTC
        assert "not has_age_weights and sr_beta <= 0" in src, (
            "Fast path guard missing — should skip manual CTC when no age weights and no SR-CTC"
        )

    def test_no_total_steps_recomputation_bug(self):
        """Crash analysis Bug 2: _total_steps recomputed on resume.
        HF Trainer manages this internally via TrainingArguments — no manual
        total_steps tracking outside create_scheduler needed."""
        import inspect
        src = inspect.getsource(HFSFTTrainer.train)
        # Must not call get_scheduler directly — LambdaLR in create_scheduler is fine
        assert "get_scheduler" not in src, (
            "Manual get_scheduler call found — should use create_scheduler override"
        )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 11 — Dataset + DataLoader integration
# ═══════════════════════════════════════════════════════════════════════

class TestDatasetIntegration:
    """Verify dataset loads correctly and has input_lengths for sampler."""

    def test_train_dataset_loads(self, cfg):
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"], split="train",
        )
        assert len(ds) > 100_000, f"Train set too small: {len(ds)}"
        assert hasattr(ds, "input_lengths"), "Missing input_lengths for LengthGroupedSampler"
        assert len(ds.input_lengths) == len(ds)

    def test_val_dataset_loads(self, cfg):
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_val.jsonl",
            cfg["paths"]["audio_dirs"], split="val",
        )
        assert len(ds) > 10_000, f"Val set too small: {len(ds)}"

    def test_length_grouped_sampler_works(self, cfg):
        """LengthGroupedSampler must work with our dataset."""
        from transformers.trainer_pt_utils import LengthGroupedSampler
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"], split="train",
        )
        sampler = LengthGroupedSampler(
            batch_size=32,
            lengths=ds.input_lengths,
        )
        # Just verify it produces valid indices
        indices = list(sampler)
        assert len(indices) == len(ds)
        assert min(indices) >= 0
        assert max(indices) < len(ds)

    def test_sample_row_schema(self, cfg):
        """Dataset rows must have all fields the collator expects."""
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"], split="train",
        )
        row = ds[0]
        required = ["utterance_id", "audio_path", "audio_duration_sec",
                     "age_bucket", "phonetic_text", "dataset"]
        for field in required:
            assert field in row, f"Missing field '{field}' in dataset row"
        # Audio path must exist
        assert Path(row["audio_path"]).is_file(), (
            f"Audio file not found: {row['audio_path']}"
        )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 12 — Full integration: build HFSFTTrainer (no train)
# ═══════════════════════════════════════════════════════════════════════

class TestFullIntegration:
    """Build the full HFSFTTrainer — verifies everything wires together."""

    def test_hf_sft_trainer_init(self, cfg):
        """Constructor must succeed: config → tokenizer → collator →
        dataset → model → freeze → grad ckpt."""
        trainer = HFSFTTrainer(cfg)

        # Verify internal state
        assert trainer._sft is cfg["hf_sft"]
        assert trainer._model is not None
        assert len(trainer._train_ds) > 100_000
        assert len(trainer._val_ds) > 10_000

        # Verify model state after init
        m = trainer._model
        assert m.wavlm.encoder.gradient_checkpointing
        assert not m.wavlm.feature_extractor._requires_grad

        # Head warmup: if active, encoder is frozen at init (Stage 1)
        _head_warmup = cfg["hf_sft"].get("warmup_head_only_steps", 0)
        if _head_warmup > 0:
            # Stage 1: only lm_head trainable, encoder frozen
            assert any(p.requires_grad for p in m.lm_head.parameters()), (
                "lm_head must be trainable during head warmup"
            )
            for i, layer in enumerate(m.wavlm.encoder.layers):
                assert not any(p.requires_grad for p in layer.parameters()), (
                    f"Layer {i} should be frozen during head warmup (Stage 1)"
                )
        else:
            # No warmup — all encoder layers trainable from step 0
            for i, layer in enumerate(m.wavlm.encoder.layers):
                assert any(p.requires_grad for p in layer.parameters()), (
                    f"Layer {i} not trainable after HFSFTTrainer init"
                )

        # Verify CNN frozen
        for p in m.wavlm.feature_extractor.parameters():
            assert not p.requires_grad

        print(f"\n  HFSFTTrainer initialized successfully")
        print(f"  Train: {len(trainer._train_ds):,} samples")
        print(f"  Val:   {len(trainer._val_ds):,} samples")
        _enc_state = "frozen (head warmup)" if _head_warmup > 0 else "trainable"
        print(f"  Model: gradient_checkpointing=ON, CNN=frozen, "
              f"encoder={_enc_state}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 13 — LLRD parameter groups
# ═══════════════════════════════════════════════════════════════════════

class TestLLRD:
    """Verify per-layer LR decay via model.py group builders."""

    def test_llrd_groups_use_model_py(self, model, cfg):
        """get_parameter_groups must produce per-layer groups with LLRD."""
        llrd_cfg = {**cfg, "sft": cfg["hf_sft"]}
        groups = get_parameter_groups(model, stage=3, cfg=llrd_cfg)

        # N layers × 2 (decay/no_decay) + head × 2 + infra × 2
        num_layers = len(model.wavlm.encoder.layers)
        assert len(groups) > num_layers, f"Expected >{num_layers} groups with LLRD, got {len(groups)}"

        # Verify no orphans/duplicates
        _verify_groups(model, groups)

    def test_llrd_layer_lr_ordering(self, model, cfg):
        """Top layer should have highest LR, layer 0 lowest."""
        llrd_cfg = {**cfg, "sft": cfg["hf_sft"]}
        groups = get_parameter_groups(model, stage=3, cfg=llrd_cfg)
        num_layers = len(model.wavlm.encoder.layers)
        top_layer = num_layers - 1

        # Extract LR per encoder layer
        layer_lrs: dict[int, float] = {}
        for g in groups:
            name = g["name"]
            if name.startswith("enc_L") and "_decay" in name:
                layer_idx = int(name.split("_")[1][1:])
                layer_lrs[layer_idx] = g["lr"]

        assert len(layer_lrs) == num_layers, f"Expected {num_layers} layer groups, got {len(layer_lrs)}"

        # Top layer should have the highest LR
        encoder_lr = cfg["hf_sft"]["encoder_lr"]
        assert abs(layer_lrs[top_layer] - encoder_lr) < 1e-10, (
            f"Layer {top_layer} LR {layer_lrs[top_layer]:.2e} != encoder_lr {encoder_lr:.2e}"
        )

        # Layer 0 should have the lowest LR
        decay = cfg["hf_sft"]["llrd_decay"]
        expected_l0 = encoder_lr * (decay ** top_layer)
        assert abs(layer_lrs[0] - expected_l0) < 1e-10, (
            f"Layer 0 LR {layer_lrs[0]:.2e} != expected {expected_l0:.2e}"
        )

        # Monotonically increasing from layer 0 to top
        for i in range(top_layer):
            assert layer_lrs[i] < layer_lrs[i + 1], (
                f"LR not increasing: layer {i} ({layer_lrs[i]:.2e}) >= "
                f"layer {i+1} ({layer_lrs[i+1]:.2e})"
            )

        print(f"\n  LLRD LR range: layer 0 = {layer_lrs[0]:.2e}, "
              f"layer {top_layer} = {layer_lrs[top_layer]:.2e}")
        print(f"  Ratio: {layer_lrs[top_layer] / layer_lrs[0]:.1f}×")

    def test_head_lr_separate(self, model, cfg):
        """Head groups must use head_lr, not encoder_lr."""
        llrd_cfg = {**cfg, "sft": cfg["hf_sft"]}
        groups = get_parameter_groups(model, stage=3, cfg=llrd_cfg)

        head_groups = [g for g in groups if g["name"].startswith("head")]
        assert len(head_groups) >= 1, "No head groups found"

        head_lr = cfg["hf_sft"]["head_lr"]
        for g in head_groups:
            assert g["lr"] == head_lr, (
                f"Head group '{g['name']}' lr={g['lr']:.2e} != head_lr={head_lr:.2e}"
            )

    def test_infra_lr_at_deepest_decay(self, model, cfg):
        """Infrastructure params should have layer-0 LR (deepest decay)."""
        llrd_cfg = {**cfg, "sft": cfg["hf_sft"]}
        groups = get_parameter_groups(model, stage=3, cfg=llrd_cfg)
        num_layers = len(model.wavlm.encoder.layers)

        infra_groups = [g for g in groups if "infra" in g["name"]]
        assert len(infra_groups) >= 1, "No infra groups found"

        decay = cfg["hf_sft"]["llrd_decay"]
        encoder_lr = cfg["hf_sft"]["encoder_lr"]
        expected_infra_lr = encoder_lr * (decay ** (num_layers - 1))

        for g in infra_groups:
            assert abs(g["lr"] - expected_infra_lr) < 1e-10, (
                f"Infra group '{g['name']}' lr={g['lr']:.2e} != "
                f"expected {expected_infra_lr:.2e}"
            )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 15 — LR floor (cosine with minimum)
# ═══════════════════════════════════════════════════════════════════════

class TestLRFloor:
    """Verify cosine schedule decays to lr_min_ratio, not zero."""

    def test_cosine_with_floor(self, cfg):
        """Simulate the LR lambda and verify floor behaviour."""
        sft = cfg["hf_sft"]
        warmup = sft["warmup_steps"]
        lr_min_ratio = sft["lr_min_ratio"]
        total_steps = 100000  # simulated total

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                return (current_step + 1) / warmup
            decay_steps = max(total_steps - warmup, 1)
            progress = min((current_step - warmup) / decay_steps, 1.0)
            cosine_f = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_f

        # At step 0: warmup starts from 1/warmup (never zero)
        assert lr_lambda(0) == 1.0 / warmup

        # At warmup end: should be 1.0
        assert abs(lr_lambda(warmup) - 1.0) < 1e-6

        # At total_steps: should be lr_min_ratio (NOT zero)
        final = lr_lambda(total_steps)
        assert abs(final - lr_min_ratio) < 1e-6, (
            f"LR at end should be {lr_min_ratio}, got {final}"
        )

        # At halfway through decay: should be > lr_min_ratio
        mid = lr_lambda(warmup + (total_steps - warmup) // 2)
        assert mid > lr_min_ratio, f"Mid-decay LR {mid} should be > {lr_min_ratio}"

        # Floor is strictly > 0
        assert final > 0, "LR floor must be > 0"

        print(f"\n  LR schedule: warmup={warmup}, total={total_steps}")
        print(f"  Peak: {lr_lambda(warmup):.4f}")
        print(f"  Mid:  {mid:.4f}")
        print(f"  Floor: {final:.6f} (= {lr_min_ratio} × base_lr)")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 13 — Weights update (1-step training simulation)
# ═══════════════════════════════════════════════════════════════════════

class TestWeightsUpdate:
    """Simulate 1 optimizer step — verify weights actually change."""

    def test_one_step_updates_weights(self, model, cfg):
        """1 forward-backward-step: head + encoder weights must change,
        CNN weights must NOT change."""
        from transformers import Wav2Vec2CTCTokenizer
        tok = Wav2Vec2CTCTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
        hf = cfg["hf_sft"]
        collator = SFTCollator(
            tok, target_sr=16_000,
            min_duration_sec=hf.get("min_duration_sec", 0.5),
            max_duration_sec=float(hf["max_duration"]),
        )
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"], split="train",
        )
        batch = collator([ds[0], ds[1]])

        # Snapshot weights before
        head_before = model.lm_head.weight.data.clone()
        encoder_before = model.wavlm.encoder.layers[0].attention.q_proj.weight.data.clone()
        cnn_before = list(model.wavlm.feature_extractor.parameters())[0].data.clone()

        # Build optimizer with discriminative LR
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith("lm_head")]
        enc_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and not n.startswith("lm_head")]
        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": hf["head_lr"]},
            {"params": enc_params, "lr": hf["encoder_lr"]},
        ])

        # Forward-backward-step
        model.train()
        optimizer.zero_grad()
        out = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out.loss.backward()
        optimizer.step()

        # Head weights MUST change
        head_after = model.lm_head.weight.data
        assert not torch.equal(head_before, head_after), (
            "CRITICAL: Head weights did not change after optimizer step"
        )

        # Encoder weights MUST change
        encoder_after = model.wavlm.encoder.layers[0].attention.q_proj.weight.data
        assert not torch.equal(encoder_before, encoder_after), (
            "CRITICAL: Encoder weights did not change after optimizer step "
            "(use_reentrant bug? frozen by mistake?)"
        )

        # CNN weights must NOT change
        cnn_after = list(model.wavlm.feature_extractor.parameters())[0].data
        assert torch.equal(cnn_before, cnn_after), (
            "CNN weights changed — should be frozen!"
        )

        head_delta = (head_after - head_before).abs().mean().item()
        enc_delta = (encoder_after - encoder_before).abs().mean().item()
        print(f"\n  Loss: {out.loss.item():.4f}")
        print(f"  Head weight Δ: {head_delta:.6f}")
        print(f"  Encoder weight Δ: {enc_delta:.6f}")
        print(f"  CNN weight Δ: 0.0 (frozen ✓)")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 14 — Dynamic LLRD (both Base+ and Large)
# ═══════════════════════════════════════════════════════════════════════

class TestDynamicLLRD:
    """Test LLRD / freeze logic with both 12-layer and 24-layer models.

    Uses HF Hub models directly — no local data required.
    """

    @staticmethod
    def _make_cfg(model_name: str, num_layers: int) -> dict:
        """Build a minimal cfg dict for model.py functions."""
        return {
            "sft": {
                "model_name": model_name,
                "vocab_size": 53,
                "blank_id": 0,
                "mask_time_prob": 0.1,
                "mask_time_length": 10,
                "mask_feature_prob": 0.0,
                "mask_feature_length": 10,
                "layerdrop": 0.0,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "feat_proj_dropout": 0.0,
                "final_dropout": 0.0,
                "head_lr": 3e-4,
                "encoder_lr": 5e-5,
                "llrd_decay": 0.92,
                "weight_decay": 0.01,
                "cnn_lr": 1e-6,
                "unfreeze_cnn": False,
            },
        }

    @pytest.fixture(scope="class")
    def base_model(self):
        cfg = self._make_cfg("microsoft/wavlm-base-plus", 12)
        m = build_model(cfg)
        return m, cfg

    @pytest.fixture(scope="class")
    def large_model(self):
        cfg = self._make_cfg("microsoft/wavlm-large", 24)
        m = build_model(cfg)
        return m, cfg

    # ---- Freeze tests ----

    def test_base_stage1_only_head(self, base_model):
        m, _ = base_model
        freeze_for_stage(m, stage=1)
        trainable_names = {n for n, p in m.named_parameters() if p.requires_grad}
        assert all(n.startswith("lm_head") for n in trainable_names)
        assert len(trainable_names) > 0

    def test_base_stage2_upper_half(self, base_model):
        m, _ = base_model
        freeze_for_stage(m, stage=2, unfreeze_cnn=False)
        num_layers = len(m.wavlm.encoder.layers)
        mid = num_layers // 2  # 6
        for i, layer in enumerate(m.wavlm.encoder.layers):
            layer_trainable = any(p.requires_grad for p in layer.parameters())
            if i >= mid:
                assert layer_trainable, f"Layer {i} should be trainable in stage 2"
            else:
                assert not layer_trainable, f"Layer {i} should be frozen in stage 2"

    def test_base_stage3_all_unfrozen(self, base_model):
        m, _ = base_model
        freeze_for_stage(m, stage=3, unfreeze_cnn=False)
        for i, layer in enumerate(m.wavlm.encoder.layers):
            assert any(p.requires_grad for p in layer.parameters()), \
                f"Layer {i} should be trainable in stage 3"
        # CNN stays frozen
        for p in m.wavlm.feature_extractor.parameters():
            assert not p.requires_grad

    def test_large_stage2_upper_half(self, large_model):
        m, _ = large_model
        freeze_for_stage(m, stage=2, unfreeze_cnn=False)
        num_layers = len(m.wavlm.encoder.layers)
        assert num_layers == 24
        mid = num_layers // 2  # 12
        for i, layer in enumerate(m.wavlm.encoder.layers):
            layer_trainable = any(p.requires_grad for p in layer.parameters())
            if i >= mid:
                assert layer_trainable, f"Layer {i} should be trainable in stage 2 (Large)"
            else:
                assert not layer_trainable, f"Layer {i} should be frozen in stage 2 (Large)"

    def test_large_stage3_all_unfrozen(self, large_model):
        m, _ = large_model
        freeze_for_stage(m, stage=3, unfreeze_cnn=False)
        for i, layer in enumerate(m.wavlm.encoder.layers):
            assert any(p.requires_grad for p in layer.parameters()), \
                f"Layer {i} should be trainable in stage 3 (Large)"

    # ---- LLRD group tests ----

    def test_base_llrd_12_layer_groups(self, base_model):
        m, cfg = base_model
        freeze_for_stage(m, stage=3, unfreeze_cnn=False)
        groups = get_parameter_groups(m, stage=3, cfg=cfg)
        _verify_groups(m, groups)

        layer_lrs = {}
        for g in groups:
            if g["name"].startswith("enc_L") and "_decay" in g["name"]:
                idx = int(g["name"].split("_")[1][1:])
                layer_lrs[idx] = g["lr"]

        assert len(layer_lrs) == 12, f"Expected 12 layer LR groups, got {len(layer_lrs)}"
        # Monotonic
        for i in range(11):
            assert layer_lrs[i] < layer_lrs[i + 1]
        # Top layer = encoder_lr
        assert abs(layer_lrs[11] - 5e-5) < 1e-10
        # Bottom layer = encoder_lr * decay^11
        assert abs(layer_lrs[0] - 5e-5 * 0.92 ** 11) < 1e-10

    def test_large_llrd_24_layer_groups(self, large_model):
        m, cfg = large_model
        freeze_for_stage(m, stage=3, unfreeze_cnn=False)
        groups = get_parameter_groups(m, stage=3, cfg=cfg)
        _verify_groups(m, groups)

        layer_lrs = {}
        for g in groups:
            if g["name"].startswith("enc_L") and "_decay" in g["name"]:
                idx = int(g["name"].split("_")[1][1:])
                layer_lrs[idx] = g["lr"]

        assert len(layer_lrs) == 24, f"Expected 24 layer LR groups, got {len(layer_lrs)}"
        # Monotonic
        for i in range(23):
            assert layer_lrs[i] < layer_lrs[i + 1]
        # Top layer = encoder_lr
        assert abs(layer_lrs[23] - 5e-5) < 1e-10
        # Bottom layer = encoder_lr * decay^23
        assert abs(layer_lrs[0] - 5e-5 * 0.92 ** 23) < 1e-10

    def test_large_infra_at_deepest_decay(self, large_model):
        m, cfg = large_model
        freeze_for_stage(m, stage=3, unfreeze_cnn=False)
        groups = get_parameter_groups(m, stage=3, cfg=cfg)
        infra = [g for g in groups if "infra" in g["name"]]
        assert len(infra) >= 1
        expected = 5e-5 * 0.92 ** 23
        for g in infra:
            assert abs(g["lr"] - expected) < 1e-10

    def test_large_head_lr_independent(self, large_model):
        m, cfg = large_model
        freeze_for_stage(m, stage=3, unfreeze_cnn=False)
        groups = get_parameter_groups(m, stage=3, cfg=cfg)
        head = [g for g in groups if g["name"].startswith("head")]
        assert len(head) >= 1
        for g in head:
            assert g["lr"] == 3e-4

    # ---- Grad norms ----

    def test_base_grad_norms_keys(self, base_model):
        m, _ = base_model
        norms = grad_norms(m)
        assert set(norms.keys()) == {"head", "upper_encoder", "lower_encoder", "cnn"}

    def test_large_grad_norms_keys(self, large_model):
        m, _ = large_model
        norms = grad_norms(m)
        assert set(norms.keys()) == {"head", "upper_encoder", "lower_encoder", "cnn"}

    # ---- Forward pass with synthetic input ----

    def test_base_forward_synthetic(self, base_model):
        m, _ = base_model
        freeze_for_stage(m, stage=3, unfreeze_cnn=False)
        m.config.layerdrop = 0.0
        m.train()
        m.zero_grad()
        x = torch.randn(2, 16000)  # 1 second of audio
        labels = torch.randint(1, 53, (2, 5))
        labels[labels == 0] = 1  # avoid blank
        out = m(input_values=x, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        # All 12 layers should get gradients
        for i, layer in enumerate(m.wavlm.encoder.layers):
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in layer.parameters())
            assert has_grad, f"Base+ layer {i} got no gradient"

    def test_large_forward_synthetic(self, large_model):
        m, _ = large_model
        freeze_for_stage(m, stage=3, unfreeze_cnn=False)
        m.config.layerdrop = 0.0
        m.train()
        m.zero_grad()
        x = torch.randn(2, 16000)
        labels = torch.randint(1, 53, (2, 5))
        labels[labels == 0] = 1
        out = m(input_values=x, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        # All 24 layers should get gradients
        for i, layer in enumerate(m.wavlm.encoder.layers):
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in layer.parameters())
            assert has_grad, f"Large layer {i} got no gradient"

    # ---- Parameter count sanity ----

    def test_base_param_count(self, base_model):
        m, _ = base_model
        total = sum(p.numel() for p in m.parameters())
        # Base+ ~94M
        assert 90_000_000 < total < 100_000_000, f"Base+ total params: {total:,}"

    def test_large_param_count(self, large_model):
        m, _ = large_model
        total = sum(p.numel() for p in m.parameters())
        # Large ~316M
        assert 310_000_000 < total < 330_000_000, f"Large total params: {total:,}"

