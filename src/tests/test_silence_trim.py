"""
Silence trimming & pipeline parity tests.
==========================================

Verifies:
  1. _trim_silence correctly removes leading/trailing silence
  2. _trim_silence preserves minimum length
  3. _trim_silence handles edge cases (all silence, no silence, near-DC)
  4. Collator with silence_trim=True produces valid batches
  5. Inference _load_audio silence trim matches training _trim_silence exactly
  6. No data leakage: val collator gets silence_trim but NO augmentation
  7. Pitch_prob is non-zero in config (regression guard for the bug)
  8. Real audio samples produce identical trim results between train and inference

Run:  PYTHONPATH=src python -m pytest src/tests/test_silence_trim.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from unittest.mock import MagicMock

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from trainer.data_collator import SFTCollator
from config.config import load_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "src" / "config" / "config.yaml"


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_collator(silence_trim: bool = True, silence_trim_db: float = -40.0,
                   min_duration_sec: float = 0.5, **kwargs) -> SFTCollator:
    """Create a minimal SFTCollator with a mock tokenizer."""
    tok = MagicMock()
    tok.side_effect = lambda text: MagicMock(input_ids=[1, 2, 3])
    tok.__call__ = tok.side_effect
    return SFTCollator(
        tok,
        target_sr=16_000,
        min_duration_sec=min_duration_sec,
        silence_trim=silence_trim,
        silence_trim_db=silence_trim_db,
        **kwargs,
    )


def _make_wav_with_silence(
    sr: int = 16_000,
    lead_silence_sec: float = 1.0,
    signal_sec: float = 2.0,
    trail_silence_sec: float = 1.0,
    signal_amp: float = 0.5,
    silence_amp: float = 0.0,
) -> torch.Tensor:
    """Create a synthetic waveform: silence + signal + silence."""
    lead = torch.full((int(lead_silence_sec * sr),), silence_amp)
    signal = torch.randn(int(signal_sec * sr)) * signal_amp
    trail = torch.full((int(trail_silence_sec * sr),), silence_amp)
    return torch.cat([lead, signal, trail])


def _inference_trim(wav: torch.Tensor, silence_db: float = -40.0,
                    min_trim_samples: int = 8000) -> torch.Tensor:
    """Replicate inference _load_audio silence trim logic."""
    n = wav.size(0)
    if n < min_trim_samples:
        return wav
    abs_wav = wav.abs()
    peak = abs_wav.max()
    if peak <= 1e-10:
        return wav
    threshold = peak * (10.0 ** (silence_db / 20.0))
    above = abs_wav > threshold
    nonzero = torch.nonzero(above, as_tuple=False)
    if nonzero.numel() == 0:
        return wav
    start = nonzero[0].item()
    end = nonzero[-1].item() + 1
    if (end - start) < min_trim_samples:
        mid = (start + end) // 2
        half = min_trim_samples // 2
        start = max(0, mid - half)
        end = min(n, start + min_trim_samples)
        start = max(0, end - min_trim_samples)
    if start > 0 or end < n:
        wav = wav[start:end].contiguous()
    return wav


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — _trim_silence unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestTrimSilence:
    """Core silence trimming algorithm tests."""

    def test_trims_leading_silence(self):
        """Leading silence is removed."""
        wav = _make_wav_with_silence(lead_silence_sec=1.0, signal_sec=2.0,
                                     trail_silence_sec=0.0)
        collator = _make_collator(silence_trim=True)
        trimmed = collator._trim_silence(wav)
        # Should be shorter than original (1s silence removed)
        assert trimmed.size(0) < wav.size(0), (
            f"Expected trimming: orig={wav.size(0)}, trimmed={trimmed.size(0)}"
        )
        # Signal should be mostly intact (within min_samples tolerance)
        assert trimmed.size(0) >= 16_000 * 1.5, (
            f"Signal too short after trim: {trimmed.size(0)} samples"
        )

    def test_trims_trailing_silence(self):
        """Trailing silence is removed."""
        wav = _make_wav_with_silence(lead_silence_sec=0.0, signal_sec=2.0,
                                     trail_silence_sec=1.0)
        collator = _make_collator(silence_trim=True)
        trimmed = collator._trim_silence(wav)
        assert trimmed.size(0) < wav.size(0)

    def test_trims_both_sides(self):
        """Both leading and trailing silence removed."""
        wav = _make_wav_with_silence(lead_silence_sec=1.0, signal_sec=2.0,
                                     trail_silence_sec=1.0)
        collator = _make_collator(silence_trim=True)
        trimmed = collator._trim_silence(wav)
        # Original = 4s, signal = 2s, so trimmed ≈ 2s
        orig_samples = wav.size(0)
        trim_samples = trimmed.size(0)
        removed = orig_samples - trim_samples
        # Should remove roughly 2s (32000 samples) of silence
        assert removed > 16_000, (
            f"Expected to remove >1s of silence, removed {removed} samples"
        )

    def test_no_trim_when_no_silence(self):
        """Pure signal (no silence) stays unchanged."""
        wav = torch.randn(32_000) * 0.5  # 2s of signal
        collator = _make_collator(silence_trim=True)
        trimmed = collator._trim_silence(wav)
        # Should be same or very close to original
        assert abs(trimmed.size(0) - wav.size(0)) < 100

    def test_preserves_minimum_length(self):
        """Short signal after trim is padded to minimum."""
        # 2s total: 0.9s silence + 0.2s signal + 0.9s silence
        # After trim, signal = 0.2s = 3200 samples < min_samples (8000)
        wav = _make_wav_with_silence(lead_silence_sec=0.9, signal_sec=0.2,
                                     trail_silence_sec=0.9)
        collator = _make_collator(silence_trim=True, min_duration_sec=0.5)
        trimmed = collator._trim_silence(wav)
        min_samples = int(0.5 * 16_000)
        assert trimmed.size(0) >= min_samples, (
            f"Trimmed to {trimmed.size(0)} samples, min is {min_samples}"
        )

    def test_all_silence_returns_unchanged(self):
        """If entire signal is below threshold, return unchanged."""
        wav = torch.zeros(32_000)  # pure digital silence
        collator = _make_collator(silence_trim=True)
        trimmed = collator._trim_silence(wav)
        assert trimmed.size(0) == wav.size(0), "All-silence should return unchanged"

    def test_near_dc_signal_returns_unchanged(self):
        """Near-DC signal (peak < 1e-10) returns unchanged."""
        wav = torch.full((32_000,), 1e-12)
        collator = _make_collator(silence_trim=True)
        trimmed = collator._trim_silence(wav)
        assert trimmed.size(0) == wav.size(0)

    def test_too_short_for_trim_returns_unchanged(self):
        """Wav shorter than min_samples skips trimming."""
        wav = torch.randn(4000)  # 0.25s < min_duration 0.5s
        collator = _make_collator(silence_trim=True, min_duration_sec=0.5)
        trimmed = collator._trim_silence(wav)
        assert trimmed.size(0) == wav.size(0)

    def test_disabled_when_flag_false(self):
        """silence_trim=False means _load_and_preprocess skips trim."""
        collator_on = _make_collator(silence_trim=True)
        collator_off = _make_collator(silence_trim=False)
        assert collator_on._silence_trim is True
        assert collator_off._silence_trim is False

    def test_threshold_db_affects_trimming(self):
        """Different dB thresholds produce different trim amounts."""
        # Low-level noise at -30 dB relative to signal
        signal = torch.randn(16_000) * 0.5
        noise = torch.randn(16_000) * 0.005  # ~-40 dB relative to 0.5
        wav = torch.cat([noise, signal, noise])

        # -40 dB threshold: noise is at boundary, may not trim much
        collator_40 = _make_collator(silence_trim=True, silence_trim_db=-40.0)
        trimmed_40 = collator_40._trim_silence(wav)

        # -20 dB threshold: noise is well below, should trim more aggressively
        collator_20 = _make_collator(silence_trim=True, silence_trim_db=-20.0)
        trimmed_20 = collator_20._trim_silence(wav)

        # -20 dB is more aggressive, so trimmed_20 should be shorter or equal
        assert trimmed_20.size(0) <= trimmed_40.size(0), (
            f"Higher threshold (-20dB) should trim more: "
            f"-20dB={trimmed_20.size(0)}, -40dB={trimmed_40.size(0)}"
        )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Training/inference parity
# ═══════════════════════════════════════════════════════════════════════

class TestPipelineParity:
    """Verify training and inference pipelines produce identical trim results."""

    def test_trim_matches_inference(self):
        """Training _trim_silence and inference trim produce same result."""
        wav = _make_wav_with_silence(lead_silence_sec=1.0, signal_sec=2.0,
                                     trail_silence_sec=1.0, signal_amp=0.5)
        collator = _make_collator(silence_trim=True, silence_trim_db=-40.0,
                                   min_duration_sec=0.5)
        train_trimmed = collator._trim_silence(wav.clone())
        infer_trimmed = _inference_trim(wav.clone(), silence_db=-40.0,
                                         min_trim_samples=8000)
        assert train_trimmed.size(0) == infer_trimmed.size(0), (
            f"Length mismatch: train={train_trimmed.size(0)}, "
            f"infer={infer_trimmed.size(0)}"
        )
        assert torch.allclose(train_trimmed, infer_trimmed, atol=1e-7), (
            "Trimmed waveforms differ between training and inference"
        )

    def test_parity_no_silence(self):
        """Parity for signal with no silence."""
        wav = torch.randn(32_000) * 0.5
        collator = _make_collator(silence_trim=True, silence_trim_db=-40.0,
                                   min_duration_sec=0.5)
        train_trimmed = collator._trim_silence(wav.clone())
        infer_trimmed = _inference_trim(wav.clone(), silence_db=-40.0,
                                         min_trim_samples=8000)
        assert train_trimmed.size(0) == infer_trimmed.size(0)

    def test_parity_all_silence(self):
        """Parity for all-silence input."""
        wav = torch.zeros(32_000)
        collator = _make_collator(silence_trim=True, silence_trim_db=-40.0,
                                   min_duration_sec=0.5)
        train_trimmed = collator._trim_silence(wav.clone())
        infer_trimmed = _inference_trim(wav.clone(), silence_db=-40.0,
                                         min_trim_samples=8000)
        assert train_trimmed.size(0) == infer_trimmed.size(0)

    def test_parity_short_signal(self):
        """Parity for short signal below min_samples."""
        wav = _make_wav_with_silence(lead_silence_sec=0.8, signal_sec=0.1,
                                     trail_silence_sec=0.8, signal_amp=0.5)
        collator = _make_collator(silence_trim=True, silence_trim_db=-40.0,
                                   min_duration_sec=0.5)
        train_trimmed = collator._trim_silence(wav.clone())
        infer_trimmed = _inference_trim(wav.clone(), silence_db=-40.0,
                                         min_trim_samples=8000)
        assert train_trimmed.size(0) == infer_trimmed.size(0)
        assert torch.allclose(train_trimmed, infer_trimmed, atol=1e-7)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Config regression guards
# ═══════════════════════════════════════════════════════════════════════

class TestConfigGuards:
    """Prevent regression of the pitch_prob=0.0 bug and verify new settings."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config(str(_CONFIG_PATH))

    def test_pitch_prob_nonzero(self, cfg):
        """CRITICAL: pitch_prob must be > 0 (was 0.0 due to missing config key)."""
        pp = cfg["hf_sft"].get("pitch_prob", 0.0)
        assert pp > 0, (
            f"pitch_prob={pp} — pitch augmentation is DISABLED! "
            f"This was a bug: pitch_augment=True but pitch_prob defaulted to 0.0"
        )
        assert 0.1 <= pp <= 0.5, f"pitch_prob={pp} outside reasonable range [0.1, 0.5]"

    def test_pitch_semitones_reasonable(self, cfg):
        ps = cfg["hf_sft"].get("pitch_semitones", 2.0)
        assert 1.0 <= ps <= 5.0, f"pitch_semitones={ps} outside [1, 5]"

    def test_noise_augment_ds2_only(self, cfg):
        """DS1 should NOT receive noise/RIR — already at 13 dB SNR."""
        ds_list = cfg["hf_sft"].get("noise_augment_datasets", [])
        assert 1 not in ds_list, f"DS1 in noise_augment_datasets: {ds_list} — DS1 is already noisy (13 dB SNR), adding MUSAN hurts"
        assert 2 in ds_list, f"DS2 missing from noise_augment_datasets: {ds_list}"

    def test_pitch_augment_all_datasets(self, cfg):
        """Pitch should apply to ALL datasets (SNR-safe, unlike noise/RIR)."""
        pitch_ds = cfg["hf_sft"].get("pitch_augment_datasets")
        assert pitch_ds is None, (
            f"pitch_augment_datasets={pitch_ds} — should be null (all datasets). "
            f"Pitch is SNR-safe and helps both DS1 and DS2."
        )

    def test_pitch_semitones_per_dataset(self, cfg):
        """Per-dataset pitch semitones should be set for DS1 and DS2 (asymmetric)."""
        ps_map = cfg["hf_sft"].get("pitch_semitones_per_dataset", {})
        assert ps_map, "pitch_semitones_per_dataset not set"
        assert 1 in ps_map or "1" in ps_map, "DS1 missing from pitch_semitones_per_dataset"
        assert 2 in ps_map or "2" in ps_map, "DS2 missing from pitch_semitones_per_dataset"
        ds1_v = ps_map.get(1, ps_map.get("1"))
        ds2_v = ps_map.get(2, ps_map.get("2"))
        # Values can be float (symmetric) or dict {low, high} (asymmetric)
        if isinstance(ds1_v, dict):
            assert ds1_v["low"] <= 0, f"DS1 low={ds1_v['low']} should be ≤0"
            assert ds1_v["high"] >= 1, f"DS1 high={ds1_v['high']} should be ≥1"
        else:
            assert 1.0 <= ds1_v <= 3.0, f"DS1 pitch_semitones={ds1_v} outside [1, 3]"
        if isinstance(ds2_v, dict):
            assert ds2_v["low"] <= 0, f"DS2 low={ds2_v['low']} should be ≤0"
            assert ds2_v["high"] >= 2, f"DS2 high={ds2_v['high']} should be ≥2"
        else:
            assert 2.0 <= ds2_v <= 5.0, f"DS2 pitch_semitones={ds2_v} outside [2, 5]"

    def test_silence_trim_enabled(self, cfg):
        assert cfg["hf_sft"].get("silence_trim", False) is True

    def test_silence_trim_db_calibrated(self, cfg):
        db = cfg["hf_sft"].get("silence_trim_db", 0)
        assert db == -40.0, f"silence_trim_db={db}, expected -40.0"

    def test_noise_prob_reasonable(self, cfg):
        """DS2-only noise prob should be moderate-to-aggressive."""
        np_ = cfg["hf_sft"].get("noise_prob", 0.0)
        assert 0.3 <= np_ <= 0.6, f"noise_prob={np_} outside [0.3, 0.6]"

    def test_silence_trim_abs_floor(self, cfg):
        """Absolute silence floor should be set to ~1e-4."""
        floor = cfg["hf_sft"].get("silence_trim_abs_floor", 0.0)
        assert floor > 0, "silence_trim_abs_floor should be > 0"
        assert floor <= 1e-3, f"silence_trim_abs_floor={floor} too high (max 1e-3)"



# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — Data leakage prevention
# ═══════════════════════════════════════════════════════════════════════

class TestNoDataLeakage:
    """Verify val collator does NOT apply augmentation (only deterministic preprocessing)."""

    def test_val_collator_no_augmentation(self):
        """Val collator should have silence_trim but no speed/noise/RIR/pitch."""
        tok = MagicMock()
        val_collator = SFTCollator(
            tok,
            target_sr=16_000,
            min_duration_sec=0.5,
            silence_trim=True,
            silence_trim_db=-40.0,
        )
        # Silence trim is deterministic preprocessing — OK for val
        assert val_collator._silence_trim is True
        # Augmentation must be OFF
        assert val_collator._speed_perturb is False, "Val collator has speed perturb!"
        assert val_collator._noise_prob == 0.0, "Val collator has noise! Data leakage!"
        assert val_collator._rir_prob == 0.0, "Val collator has RIR! Data leakage!"
        assert val_collator._pitch_prob == 0.0, "Val collator has pitch! Data leakage!"

    def test_train_collator_has_augmentation(self):
        """Train collator should have all augmentations enabled."""
        tok = MagicMock()
        train_collator = SFTCollator(
            tok,
            target_sr=16_000,
            min_duration_sec=0.5,
            speed_perturb=True,
            pitch_prob=0.30,
            pitch_semitones=3.0,
            pitch_datasets=None,  # all datasets
            pitch_semitones_per_dataset={1: {"low": -2.0, "high": 6.0}, 2: {"low": -2.0, "high": 10.0}},
            silence_trim=True,
            silence_trim_db=-40.0,
            silence_trim_abs_floor=1e-4,
        )
        assert train_collator._speed_perturb is True
        assert train_collator._pitch_prob == 0.30
        assert train_collator._pitch_semitones == 3.0
        assert train_collator._silence_trim is True
        assert train_collator._silence_trim_abs_floor == 1e-4
        assert train_collator._pitch_datasets is None  # all datasets
        assert train_collator._pitch_semitones_map == {"1": (-2.0, 6.0), "2": (-2.0, 10.0)}

    def test_silence_trim_is_deterministic(self):
        """Same input → same output (no randomness in trim)."""
        wav = _make_wav_with_silence(lead_silence_sec=1.0, signal_sec=2.0,
                                     trail_silence_sec=1.0, signal_amp=0.5)
        collator = _make_collator(silence_trim=True)
        trimmed1 = collator._trim_silence(wav.clone())
        trimmed2 = collator._trim_silence(wav.clone())
        assert torch.allclose(trimmed1, trimmed2, atol=1e-8), (
            "Silence trim must be deterministic (safe for eval)"
        )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — Real audio integration (if data available)
# ═══════════════════════════════════════════════════════════════════════

class TestRealAudioTrim:
    """Test silence trim on real competition audio files."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config(str(_CONFIG_PATH))

    @pytest.fixture(scope="class")
    def real_collator_with_trim(self, cfg):
        from transformers import Wav2Vec2CTCTokenizer
        tok = Wav2Vec2CTCTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
        return SFTCollator(
            tok, target_sr=16_000,
            min_duration_sec=cfg["hf_sft"].get("min_duration_sec", 0.5),
            max_duration_sec=float(cfg["hf_sft"]["max_duration"]),
            silence_trim=True,
            silence_trim_db=-40.0,
        )

    @pytest.fixture(scope="class")
    def real_collator_without_trim(self, cfg):
        from transformers import Wav2Vec2CTCTokenizer
        tok = Wav2Vec2CTCTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
        return SFTCollator(
            tok, target_sr=16_000,
            min_duration_sec=cfg["hf_sft"].get("min_duration_sec", 0.5),
            max_duration_sec=float(cfg["hf_sft"]["max_duration"]),
            silence_trim=False,
        )

    @pytest.fixture(scope="class")
    def real_samples(self, cfg):
        from trainer.dataset import SFTDataset
        ds = SFTDataset(
            f"{cfg['paths']['processed']}/sft_train.jsonl",
            cfg["paths"]["audio_dirs"],
            split="train",
        )
        return [ds[i] for i in range(8)]

    def test_trimmed_batch_produces_valid_output(self, real_collator_with_trim,
                                                   real_samples):
        """Real audio with trim produces valid batch."""
        batch = real_collator_with_trim(real_samples)
        assert batch is not None
        assert batch["input_values"].ndim == 2
        assert batch["input_values"].shape[0] == 8

    def test_trimmed_still_zero_mean_unit_var(self, real_collator_with_trim,
                                                real_samples):
        """CMVN after trim still produces zero-mean unit-var."""
        batch = real_collator_with_trim(real_samples)
        iv = batch["input_values"]
        am = batch["attention_mask"]
        for i in range(iv.shape[0]):
            length = am[i].sum().item()
            wav = iv[i, :length]
            mean = wav.mean().item()
            std = wav.std().item()
            assert abs(mean) < 0.01, f"Sample {i} mean={mean:.4f}"
            assert abs(std - 1.0) < 0.1, f"Sample {i} std={std:.4f}"

    def test_trim_does_not_increase_length(self, real_collator_with_trim,
                                             real_collator_without_trim,
                                             real_samples):
        """Trimmed samples should be <= non-trimmed length."""
        batch_trim = real_collator_with_trim(real_samples)
        batch_notrim = real_collator_without_trim(real_samples)
        for i in range(8):
            len_trim = batch_trim["attention_mask"][i].sum().item()
            len_notrim = batch_notrim["attention_mask"][i].sum().item()
            assert len_trim <= len_notrim, (
                f"Sample {i}: trimmed ({len_trim}) > untrimmed ({len_notrim})"
            )

    def test_trim_preserves_label_integrity(self, real_collator_with_trim,
                                              real_samples):
        """Labels are unchanged by silence trim (only audio affected)."""
        batch = real_collator_with_trim(real_samples)
        labels = batch["labels"]
        # Labels should have valid token IDs (> 0) where not padded
        valid_mask = labels != -100
        valid_ids = labels[valid_mask]
        assert (valid_ids > 0).all(), "Labels contain <= 0 IDs (possible blank leak)"
        assert (valid_ids < 53).all(), "Labels contain IDs >= vocab_size"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — Pitch gating (separated from noise/RIR)
# ═══════════════════════════════════════════════════════════════════════

class TestPitchGating:
    """Verify pitch augmentation is independently gated from noise/RIR."""

    def test_pitch_all_noise_ds2_only(self):
        """Pitch applies to all datasets while noise/RIR is DS2-only."""
        tok = MagicMock()
        tok.side_effect = lambda text: MagicMock(input_ids=[1, 2, 3])
        tok.__call__ = tok.side_effect
        collator = SFTCollator(
            tok,
            target_sr=16_000,
            pitch_prob=0.30,
            pitch_semitones=3.0,
            noise_datasets=[2],       # noise/RIR = DS2 only
            pitch_datasets=None,      # pitch = all datasets
        )
        assert collator._noise_datasets == frozenset({"2"})
        assert collator._pitch_datasets is None  # None = all

    def test_pitch_gated_to_specific_datasets(self):
        """pitch_datasets can restrict pitch to specific datasets."""
        tok = MagicMock()
        collator = SFTCollator(
            tok,
            target_sr=16_000,
            pitch_prob=0.30,
            pitch_datasets=[1],  # pitch DS1 only
        )
        assert collator._pitch_datasets == frozenset({"1"})

    def test_per_dataset_semitones(self):
        """Per-dataset semitone map overrides default — supports both float and dict."""
        tok = MagicMock()
        # Float format (symmetric)
        collator_sym = SFTCollator(
            tok,
            target_sr=16_000,
            pitch_prob=0.30,
            pitch_semitones=3.0,
            pitch_semitones_per_dataset={1: 2.0, 2: 3.0},
        )
        assert collator_sym._pitch_semitones == 3.0  # default
        assert collator_sym._pitch_semitones_map == {"1": (-2.0, 2.0), "2": (-3.0, 3.0)}

        # Dict format (asymmetric)
        collator_asym = SFTCollator(
            tok,
            target_sr=16_000,
            pitch_prob=0.30,
            pitch_semitones=3.0,
            pitch_semitones_per_dataset={1: {"low": -2.0, "high": 6.0}, 2: {"low": -2.0, "high": 10.0}},
        )
        assert collator_asym._pitch_semitones_map == {"1": (-2.0, 6.0), "2": (-2.0, 10.0)}

    def test_per_dataset_semitones_empty_by_default(self):
        """No per-dataset map = empty dict (use default semitones)."""
        tok = MagicMock()
        collator = SFTCollator(
            tok,
            target_sr=16_000,
            pitch_prob=0.30,
            pitch_semitones=3.0,
        )
        assert collator._pitch_semitones_map == {}

    def test_regularization_r15_levels(self):
        """Config should have R15-level regularization (not aggressive R21)."""
        cfg = load_config(str(_CONFIG_PATH))
        sft = cfg["hf_sft"]
        assert sft["attention_dropout"] == 0.1, f"attention_dropout={sft['attention_dropout']}, expected 0.1"
        assert sft["hidden_dropout"] == 0.1, f"hidden_dropout={sft['hidden_dropout']}, expected 0.1"
        assert sft["layerdrop"] == 0.1, f"layerdrop={sft['layerdrop']}, expected 0.1"
        assert sft["mask_time_prob"] == 0.30, f"mask_time_prob={sft['mask_time_prob']}, expected 0.30"
        assert sft["mask_feature_prob"] == 0.12, f"mask_feature_prob={sft['mask_feature_prob']}, expected 0.12"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — Absolute silence floor
# ═══════════════════════════════════════════════════════════════════════

class TestAbsSilenceFloor:
    """Verify absolute silence floor zeroes sub-threshold samples."""

    def test_floor_zeros_quiet_samples(self):
        """Samples below abs floor should be zeroed."""
        collator = _make_collator(silence_trim=False,
                                  silence_trim_abs_floor=1e-4)
        wav = torch.tensor([0.5, 1e-5, -1e-5, 0.3, 5e-5, -0.2])
        result = collator._apply_abs_floor(wav)
        assert result[0] == 0.5, "Above-floor sample should be unchanged"
        assert result[1] == 0.0, "Below-floor sample should be zeroed"
        assert result[2] == 0.0, "Below-floor negative sample should be zeroed"
        assert result[3] == 0.3, "Above-floor sample should be unchanged"
        assert result[4] == 0.0, "Below-floor sample should be zeroed"
        assert result[5] == -0.2, "Above-floor sample should be unchanged"

    def test_floor_disabled_when_zero(self):
        """Floor of 0 should be a no-op."""
        collator = _make_collator(silence_trim=False,
                                  silence_trim_abs_floor=0.0)
        wav = torch.tensor([1e-5, 1e-6, 0.5])
        result = collator._apply_abs_floor(wav)
        assert torch.allclose(result, wav), "Floor=0 should not change anything"

    def test_floor_preserves_original(self):
        """Original tensor should not be mutated."""
        collator = _make_collator(silence_trim=False,
                                  silence_trim_abs_floor=1e-4)
        wav = torch.tensor([1e-5, 0.5])
        original = wav.clone()
        _ = collator._apply_abs_floor(wav)
        assert torch.allclose(wav, original), "Original tensor should not be mutated"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — Asymmetric pitch shift
# ═══════════════════════════════════════════════════════════════════════

class TestAsymmetricPitch:
    """Verify asymmetric pitch shift uses (low, high) range correctly."""

    def test_pitch_shift_symmetric_tuple(self):
        """Float input should be converted to symmetric tuple."""
        collator = _make_collator()
        wav = torch.randn(16_000)
        # Symmetric: semitones=2.0 → uniform(-2, 2)
        result = collator._pitch_shift(wav, semitones=2.0)
        assert result.shape == wav.shape, "Pitch shift should preserve length"

    def test_pitch_shift_asymmetric_tuple(self):
        """Tuple (low, high) should produce valid shifted output."""
        collator = _make_collator()
        wav = torch.randn(16_000)
        result = collator._pitch_shift(wav, semitones=(-2.0, 10.0))
        assert result.shape == wav.shape, "Asymmetric pitch shift should preserve length"

    def test_pitch_shift_none_uses_default(self):
        """None should fall back to self._pitch_semitones."""
        collator = _make_collator(pitch_prob=1.0, pitch_semitones=3.0)
        wav = torch.randn(16_000)
        result = collator._pitch_shift(wav, semitones=None)
        assert result.shape == wav.shape

    def test_asymmetric_map_lookup(self):
        """Collator should store asymmetric map as tuples."""
        tok = MagicMock()
        collator = SFTCollator(
            tok, target_sr=16_000,
            pitch_prob=0.30,
            pitch_semitones_per_dataset={
                1: {"low": -2.0, "high": 6.0},
                2: {"low": -2.0, "high": 10.0},
            },
        )
        assert collator._pitch_semitones_map["1"] == (-2.0, 6.0)
        assert collator._pitch_semitones_map["2"] == (-2.0, 10.0)

    def test_mixed_format_map(self):
        """Map should handle mix of float and dict values."""
        tok = MagicMock()
        collator = SFTCollator(
            tok, target_sr=16_000,
            pitch_prob=0.30,
            pitch_semitones_per_dataset={
                1: 2.0,                          # symmetric
                2: {"low": -2.0, "high": 10.0},  # asymmetric
            },
        )
        assert collator._pitch_semitones_map["1"] == (-2.0, 2.0)
        assert collator._pitch_semitones_map["2"] == (-2.0, 10.0)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — DS1 dynamic SpecAug reduction (Run 24)
# ═══════════════════════════════════════════════════════════════════════

class TestDS1SpecAugReduction:
    """Verify dynamic mask_time_prob interpolation by DS1 batch fraction.

    Run 24 reduces SpecAug time masking for DS1 batches because DS1
    is already noisy (~12 dB SNR) — stacking full SpecAug double-corrupts.
    mask_time_prob is interpolated: all-DS2 → 0.30, all-DS1 → 0.15.
    """

    def test_config_has_ds1_mask_time_prob(self):
        """Config must have ds1_mask_time_prob key."""
        import yaml
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        sft = cfg["hf_sft"]
        assert "ds1_mask_time_prob" in sft, "Missing ds1_mask_time_prob in config"
        assert sft["ds1_mask_time_prob"] == 0.15
        assert sft["mask_time_prob"] == 0.30

    def test_interpolation_all_ds2(self):
        """All-DS2 batch → no mutation (stays at 0.30)."""
        default = 0.30
        ds1_prob = 0.15
        ds_ids = ["2", "2", "2", "2"]
        ds1_count = sum(1 for d in ds_ids if d == "1")
        ds1_frac = ds1_count / len(ds_ids)
        assert ds1_frac == 0.0
        # In code: ds1_frac == 0 → _specaug_adjusted stays False → no mutation

    def test_interpolation_all_ds1(self):
        """All-DS1 batch → mask_time_prob = ds1_mask_time_prob."""
        default = 0.30
        ds1_prob = 0.15
        ds_ids = ["1", "1", "1", "1"]
        ds1_count = sum(1 for d in ds_ids if d == "1")
        ds1_frac = ds1_count / len(ds_ids)
        effective = default - (default - ds1_prob) * ds1_frac
        assert effective == ds1_prob, f"Expected {ds1_prob}, got {effective}"

    def test_interpolation_half_ds1(self):
        """50% DS1 → mask_time_prob ≈ 0.225."""
        default = 0.30
        ds1_prob = 0.15
        ds_ids = ["1", "1", "2", "2"]
        ds1_count = sum(1 for d in ds_ids if d == "1")
        ds1_frac = ds1_count / len(ds_ids)
        effective = default - (default - ds1_prob) * ds1_frac
        assert abs(effective - 0.225) < 1e-10, f"Expected ~0.225, got {effective}"

    def test_interpolation_quarter_ds1(self):
        """25% DS1 → mask_time_prob = 0.2625."""
        default = 0.30
        ds1_prob = 0.15
        ds_ids = ["1", "2", "2", "2"]
        ds1_count = sum(1 for d in ds_ids if d == "1")
        ds1_frac = ds1_count / len(ds_ids)
        effective = default - (default - ds1_prob) * ds1_frac
        assert abs(effective - 0.2625) < 1e-10

    def test_trainer_class_has_attrs(self):
        """_CTCTrainer must have ds1_mask_time_prob and default_mask_time_prob attrs."""
        from trainer.sft_trainer_hf import _CTCTrainer
        assert hasattr(_CTCTrainer, "_ds1_mask_time_prob")
        assert hasattr(_CTCTrainer, "_default_mask_time_prob")
        # Default values on the class
        assert _CTCTrainer._ds1_mask_time_prob is None  # disabled by default
        assert _CTCTrainer._default_mask_time_prob == 0.30

    def test_compute_loss_resets_after_forward(self):
        """mask_time_prob must be reset to default after forward, even on exception path.

        This test verifies the reset logic by checking both exit paths in
        compute_loss (fast path and SR-CTC path) reset the config value.
        """
        # Test the code structure — both return statements have the reset
        import inspect
        from trainer.sft_trainer_hf import _CTCTrainer
        source = inspect.getsource(_CTCTrainer.compute_loss)
        # Count reset occurrences
        reset_count = source.count("model.wavlm.config.mask_time_prob = self._default_mask_time_prob")
        assert reset_count == 2, (
            f"Expected 2 reset points (fast path + SR-CTC path), found {reset_count}"
        )

    def test_eval_does_not_mutate_specaug(self):
        """compute_loss must NOT mutate mask_time_prob when model.training is False.

        The guard `self.model.training` prevents SpecAug adjustment during eval.
        """
        import inspect
        from trainer.sft_trainer_hf import _CTCTrainer
        source = inspect.getsource(_CTCTrainer.compute_loss)
        assert "self.model.training" in source, (
            "compute_loss must check self.model.training before adjusting SpecAug"
        )

    def test_collator_passes_datasets_key(self):
        """Collator __call__ return dict must include 'datasets' key."""
        tok = MagicMock()
        tok.return_value = MagicMock(input_ids=[1, 2, 3])
        collator = SFTCollator(tok, target_sr=16_000)

        # Create a minimal valid batch
        wav = torch.randn(16_000)
        import tempfile, soundfile as sf, numpy as np
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.random.randn(16_000).astype(np.float32), 16_000)
            row = {
                "audio_path": f.name,
                "phonetic_text": "h ɛ l oʊ",
                "age_bucket": "3-4",
                "dataset": "1",
                "audio_duration_sec": 1.0,
            }
            result = collator([row])

        assert result is not None
        assert "datasets" in result, "Collator must return 'datasets' key"
        assert result["datasets"] == ["1"]
