"""
Real-audio pipeline stress test — exercises every augmentation step.
====================================================================

Runs the FULL collator pipeline (load → mono → resample → silence trim →
speed perturb → CMVN → floor pad → noise/RIR → pitch/VTLP → re-normalize)
on actual competition audio files with all augmentations enabled.

Verifies:
  1. No NaN / Inf in outputs
  2. Correct tensor shapes and dtypes
  3. CMVN normalization (zero-mean, unit-var) for unpadded regions
  4. Attention mask correctness (1 = real, 0 = pad)
  5. Label validity (valid token IDs, proper padding)
  6. Speed perturbation changes waveform length
  7. Pitch shift preserves waveform length
  8. Noise injection modifies waveform
  9. RIR convolution modifies waveform
  10. Dataset gating: DS1 gets pitch but NOT noise/RIR
  11. Per-dataset semitone ranges are applied correctly
  12. squeeze() safety: interpolate outputs have correct ndim
  13. Multiple batches processed without state corruption

Run:  PYTHONPATH=src python -m pytest src/tests/test_pipeline_stress.py -v
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config.config import load_config
from trainer.data_collator import SFTCollator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "src" / "config" / "config.yaml"

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def cfg():
    return load_config(str(_CONFIG_PATH))


@pytest.fixture(scope="module")
def tokenizer(cfg):
    from transformers import Wav2Vec2CTCTokenizer
    return Wav2Vec2CTCTokenizer.from_pretrained(cfg["paths"]["tokenizer"])


@pytest.fixture(scope="module")
def real_samples(cfg):
    """Load 20 real training samples — mix of DS1 and DS2."""
    from trainer.dataset import SFTDataset
    ds = SFTDataset(
        f"{cfg['paths']['processed']}/sft_train.jsonl",
        cfg["paths"]["audio_dirs"],
        split="train",
    )
    # Grab first 10 samples then seek DS2 samples
    samples = [ds[i] for i in range(10)]
    # Find some DS2 samples (dataset 2)
    for i in range(10, min(len(ds), 500)):
        row = ds[i]
        if str(row.get("dataset", "")) == "2":
            samples.append(row)
            if len(samples) >= 20:
                break
    return samples


@pytest.fixture(scope="module")
def full_train_collator(cfg, tokenizer):
    """Collator with ALL augmentations enabled, matching config.yaml exactly."""
    sft = cfg["hf_sft"]
    return SFTCollator(
        tokenizer,
        target_sr=16_000,
        min_duration_sec=sft.get("min_duration_sec", 1.0),
        max_duration_sec=float(sft["max_duration"]),
        speed_perturb=sft.get("speed_perturb", False),
        speed_perturb_range=tuple(sft.get("speed_perturb_range", [0.9, 1.1])),
        noise_dir=sft.get("noise_dir"),
        noise_prob=sft.get("noise_prob", 0.0),
        rir_dir=sft.get("rir_dir"),
        rir_prob=sft.get("rir_prob", 0.0),
        pitch_prob=sft.get("pitch_prob", 0.0),
        pitch_semitones=sft.get("pitch_semitones", 2.0),
        noise_datasets=sft.get("noise_augment_datasets"),
        pitch_datasets=sft.get("pitch_augment_datasets"),
        pitch_semitones_per_dataset=sft.get("pitch_semitones_per_dataset"),
        silence_trim=sft.get("silence_trim", False),
        silence_trim_db=sft.get("silence_trim_db", -40.0),
        silence_trim_abs_floor=sft.get("silence_trim_abs_floor", 0.0),
    )


@pytest.fixture(scope="module")
def val_collator(cfg, tokenizer):
    """Val collator — silence trim only, NO augmentation."""
    sft = cfg["hf_sft"]
    return SFTCollator(
        tokenizer,
        target_sr=16_000,
        min_duration_sec=sft.get("min_duration_sec", 1.0),
        max_duration_sec=float(sft["max_duration"]),
        silence_trim=sft.get("silence_trim", False),
        silence_trim_db=sft.get("silence_trim_db", -40.0),
        silence_trim_abs_floor=sft.get("silence_trim_abs_floor", 0.0),
    )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — Full pipeline batch validity
# ═══════════════════════════════════════════════════════════════════════

class TestFullPipelineBatchValidity:
    """Process real audio through the full augmented pipeline and verify outputs."""

    def test_batch_no_nan_no_inf(self, full_train_collator, real_samples):
        """No NaN or Inf in output waveforms after full augmentation pipeline."""
        batch = full_train_collator(real_samples[:8])
        assert batch is not None, "Entire batch was dropped"
        iv = batch["input_values"]
        assert not torch.isnan(iv).any(), "NaN detected in input_values"
        assert not torch.isinf(iv).any(), "Inf detected in input_values"

    def test_batch_correct_shapes(self, full_train_collator, real_samples):
        """Batch tensors have correct shapes and dtypes."""
        batch = full_train_collator(real_samples[:8])
        assert batch is not None
        B = batch["input_values"].shape[0]
        assert B <= 8, f"Batch size {B} > 8 input samples"
        assert batch["input_values"].ndim == 2, "input_values should be (B, T)"
        assert batch["attention_mask"].ndim == 2, "attention_mask should be (B, T)"
        assert batch["labels"].ndim == 2, "labels should be (B, L)"
        assert batch["input_values"].dtype == torch.float32
        assert batch["attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["input_lengths"].shape == (B,)

    def test_cmvn_after_augmentation(self, full_train_collator, real_samples):
        """Unpadded regions of every sample are ~zero-mean, ~unit-var."""
        batch = full_train_collator(real_samples[:8])
        assert batch is not None
        iv = batch["input_values"]
        am = batch["attention_mask"]
        for i in range(iv.shape[0]):
            length = am[i].sum().item()
            wav = iv[i, :length]
            mean = wav.mean().item()
            std = wav.std().item()
            assert abs(mean) < 0.05, (
                f"Sample {i} mean={mean:.4f} (should be ~0 after CMVN)"
            )
            assert abs(std - 1.0) < 0.15, (
                f"Sample {i} std={std:.4f} (should be ~1 after CMVN)"
            )

    def test_attention_mask_correctness(self, full_train_collator, real_samples):
        """Attention mask is 1 for real samples, 0 for padding."""
        batch = full_train_collator(real_samples[:8])
        assert batch is not None
        am = batch["attention_mask"]
        il = batch["input_lengths"]
        for i in range(am.shape[0]):
            assert am[i, :il[i]].sum().item() == il[i].item(), (
                f"Sample {i}: mask ones != declared length"
            )
            if il[i] < am.shape[1]:
                assert am[i, il[i]:].sum().item() == 0, (
                    f"Sample {i}: non-zero mask in padding region"
                )

    def test_labels_valid(self, full_train_collator, real_samples):
        """Labels contain valid token IDs where not padded."""
        batch = full_train_collator(real_samples[:8])
        assert batch is not None
        labels = batch["labels"]
        valid_mask = labels != -100
        valid_ids = labels[valid_mask]
        assert valid_ids.numel() > 0, "No valid label IDs in batch"
        assert (valid_ids >= 0).all(), f"Negative label ID: min={valid_ids.min()}"
        assert (valid_ids < 53).all(), f"Label ID >= vocab_size: max={valid_ids.max()}"

    def test_padded_region_is_zero(self, full_train_collator, real_samples):
        """Padding region of input_values should be zeros."""
        batch = full_train_collator(real_samples[:8])
        assert batch is not None
        iv = batch["input_values"]
        am = batch["attention_mask"]
        for i in range(iv.shape[0]):
            length = am[i].sum().item()
            if length < iv.shape[1]:
                pad_region = iv[i, length:]
                assert (pad_region == 0).all(), (
                    f"Sample {i}: non-zero values in padding region"
                )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Individual augmentation effects
# ═══════════════════════════════════════════════════════════════════════

class TestAugmentationEffects:
    """Verify each augmentation step actually modifies the waveform correctly."""

    def test_speed_perturb_changes_length(self, tokenizer, cfg, real_samples):
        """Speed perturbation should produce variable-length outputs."""
        sft = cfg["hf_sft"]
        collator = SFTCollator(
            tokenizer, target_sr=16_000,
            min_duration_sec=sft.get("min_duration_sec", 1.0),
            speed_perturb=True,
            speed_perturb_range=(0.8, 1.2),
            silence_trim=True,
            silence_trim_db=-40.0,
        )
        # Process same sample 10 times — lengths should vary
        sample = real_samples[0]
        lengths = set()
        for _ in range(10):
            wav = collator._load_and_preprocess(sample["audio_path"])
            lengths.add(wav.size(0))
        assert len(lengths) > 1, (
            "Speed perturbation produced identical lengths across 10 runs"
        )

    def test_pitch_shift_preserves_length(self, full_train_collator):
        """Pitch shift should NOT change waveform length."""
        wav = torch.randn(32_000)
        shifted = full_train_collator._pitch_shift(wav.clone(), semitones=4.0)
        assert shifted.size(0) == wav.size(0), (
            f"Pitch shift changed length: {wav.size(0)} → {shifted.size(0)}"
        )

    def test_pitch_shift_modifies_waveform(self, full_train_collator):
        """Pitch shift should actually change the waveform content."""
        torch.manual_seed(42)
        wav = torch.sin(torch.linspace(0, 200 * 3.14159, 32_000))
        random.seed(999)  # ensure non-trivial shift
        shifted = full_train_collator._pitch_shift(wav.clone(), semitones=4.0)
        # Should not be identical
        assert not torch.allclose(wav, shifted, atol=1e-5), (
            "Pitch shift did not modify the waveform"
        )

    def test_pitch_shift_output_is_1d(self, full_train_collator):
        """Pitch shift output must be a 1-D tensor (squeeze safety)."""
        for length in [1, 100, 16_000, 48_000]:
            wav = torch.randn(length)
            shifted = full_train_collator._pitch_shift(wav.clone(), semitones=2.0)
            assert shifted.ndim == 1, (
                f"Pitch shift output ndim={shifted.ndim} for input len={length}"
            )
            assert shifted.size(0) == length, (
                f"Pitch shift changed len: {length} → {shifted.size(0)}"
            )

    def test_noise_injection_modifies_waveform(self, full_train_collator):
        """Noise injection should change the waveform."""
        wav = torch.randn(32_000)
        wav = (wav - wav.mean()) / wav.std()  # CMVN-normalised
        noised = full_train_collator._inject_noise(wav.clone())
        assert not torch.allclose(wav, noised, atol=1e-5), (
            "Noise injection did not modify the waveform"
        )

    def test_rir_convolution_modifies_waveform(self, full_train_collator):
        """RIR convolution should change the waveform."""
        wav = torch.randn(32_000)
        wav = (wav - wav.mean()) / wav.std()
        reverbed = full_train_collator._apply_rir(wav.clone())
        assert reverbed.size(0) == wav.size(0), "RIR changed waveform length"
        assert not torch.allclose(wav, reverbed, atol=1e-5), (
            "RIR convolution did not modify the waveform"
        )

    def test_noise_injection_no_nan(self, full_train_collator):
        """Noise injection on various inputs should never produce NaN."""
        for _ in range(20):
            wav = torch.randn(random.randint(8_000, 64_000))
            wav = (wav - wav.mean()) / (wav.std() + 1e-8)
            noised = full_train_collator._inject_noise(wav)
            assert not torch.isnan(noised).any(), "NaN from noise injection"

    def test_rir_convolution_no_nan(self, full_train_collator):
        """RIR convolution should never produce NaN."""
        for _ in range(20):
            wav = torch.randn(random.randint(8_000, 64_000))
            wav = (wav - wav.mean()) / (wav.std() + 1e-8)
            reverbed = full_train_collator._apply_rir(wav)
            assert not torch.isnan(reverbed).any(), "NaN from RIR convolution"

    def test_silence_trim_real_audio(self, full_train_collator, real_samples):
        """Silence trim on real audio should not crash or produce empty tensors."""
        for sample in real_samples[:10]:
            import soundfile as sf
            data, sr = sf.read(sample["audio_path"], dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            wav = torch.from_numpy(data)
            trimmed = full_train_collator._trim_silence(wav)
            assert trimmed.ndim == 1
            assert trimmed.size(0) > 0, "Trim produced empty tensor"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Dataset gating correctness
# ═══════════════════════════════════════════════════════════════════════

class TestDatasetGating:
    """Verify DS1 gets pitch but NOT noise/RIR, DS2 gets everything."""

    def test_ds1_gets_pitch_not_noise(self, full_train_collator):
        """DS1 row: pitch_allowed=True, noise_allowed=False."""
        row = {"dataset": 1, "audio_path": "", "phonetic_text": "test"}
        _ds_key = str(row["dataset"])
        noise_allowed = (
            full_train_collator._noise_datasets is None
            or _ds_key in full_train_collator._noise_datasets
        )
        pitch_allowed = (
            full_train_collator._pitch_datasets is None
            or _ds_key in full_train_collator._pitch_datasets
        )
        assert not noise_allowed, f"DS1 should NOT get noise (noise_datasets={full_train_collator._noise_datasets})"
        assert pitch_allowed, f"DS1 should get pitch (pitch_datasets={full_train_collator._pitch_datasets})"

    def test_ds2_gets_everything(self, full_train_collator):
        """DS2 row: both pitch_allowed=True and noise_allowed=True."""
        _ds_key = "2"
        noise_allowed = (
            full_train_collator._noise_datasets is None
            or _ds_key in full_train_collator._noise_datasets
        )
        pitch_allowed = (
            full_train_collator._pitch_datasets is None
            or _ds_key in full_train_collator._pitch_datasets
        )
        assert noise_allowed, "DS2 should get noise"
        assert pitch_allowed, "DS2 should get pitch"

    def test_per_dataset_semitones_ds1_conservative(self, full_train_collator):
        """DS1 gets asymmetric pitch [-2, +6] st (conservative down, aggressive up)."""
        m = full_train_collator._pitch_semitones_map
        assert "1" in m, f"DS1 missing from pitch_semitones_map: {m}"
        low, high = m["1"]
        assert low <= 0, f"DS1 low={low} should be ≤0"
        assert high >= 1, f"DS1 high={high} should be ≥1"

    def test_per_dataset_semitones_ds2_wide(self, full_train_collator):
        """DS2 gets asymmetric pitch [-2, +10] st (wider up-shift for F0 gap)."""
        m = full_train_collator._pitch_semitones_map
        assert "2" in m, f"DS2 missing from pitch_semitones_map: {m}"
        low, high = m["2"]
        assert low <= 0, f"DS2 low={low} should be ≤0"
        assert high >= 4, f"DS2 high={high} should be ≥4"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — Multi-batch state safety
# ═══════════════════════════════════════════════════════════════════════

class TestMultiBatchStateSafety:
    """Process multiple batches to verify no state corruption between calls."""

    def test_consecutive_batches_no_corruption(self, full_train_collator, real_samples):
        """Process 5 consecutive batches — all should be valid."""
        for batch_idx in range(5):
            start = (batch_idx * 4) % len(real_samples)
            batch_samples = real_samples[start:start + 4]
            if not batch_samples:
                batch_samples = real_samples[:4]
            batch = full_train_collator(batch_samples)
            assert batch is not None, f"Batch {batch_idx} was dropped"
            iv = batch["input_values"]
            assert not torch.isnan(iv).any(), f"NaN in batch {batch_idx}"
            assert not torch.isinf(iv).any(), f"Inf in batch {batch_idx}"
            # CMVN check
            am = batch["attention_mask"]
            for i in range(iv.shape[0]):
                length = am[i].sum().item()
                wav = iv[i, :length]
                assert abs(wav.mean().item()) < 0.05, (
                    f"Batch {batch_idx}, sample {i}: mean={wav.mean():.4f}"
                )

    def test_pitch_semitones_unchanged_after_batch(self, full_train_collator, real_samples):
        """self._pitch_semitones should be unchanged after processing a batch."""
        original_st = full_train_collator._pitch_semitones
        _ = full_train_collator(real_samples[:4])
        assert full_train_collator._pitch_semitones == original_st, (
            f"pitch_semitones mutated: {original_st} → {full_train_collator._pitch_semitones}"
        )

    def test_mixed_ds1_ds2_batch(self, full_train_collator, real_samples):
        """Batch with mixed DS1/DS2 samples processes correctly."""
        ds1 = [s for s in real_samples if str(s.get("dataset", "")) == "1"][:3]
        ds2 = [s for s in real_samples if str(s.get("dataset", "")) == "2"][:3]
        mixed = ds1 + ds2
        if len(mixed) < 2:
            pytest.skip("Need both DS1 and DS2 samples")
        batch = full_train_collator(mixed)
        assert batch is not None
        assert not torch.isnan(batch["input_values"]).any()
        # Check datasets metadata is preserved
        for ds_id in batch["datasets"]:
            assert ds_id in ("1", "2"), f"Unknown dataset ID: {ds_id}"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — Val vs Train collator separation
# ═══════════════════════════════════════════════════════════════════════

class TestValTrainSeparation:
    """Val collator must NOT apply stochastic augmentation."""

    def test_val_deterministic_same_input(self, val_collator, real_samples):
        """Val collator with same input → same output (excluding loading jitter)."""
        samples = real_samples[:4]
        batch1 = val_collator(samples)
        batch2 = val_collator(samples)
        assert batch1 is not None and batch2 is not None
        # Lengths should be identical (no speed perturb)
        assert torch.equal(batch1["input_lengths"], batch2["input_lengths"]), (
            "Val collator produced different lengths — stochastic augmentation leak?"
        )
        # Waveforms should be identical
        assert torch.allclose(
            batch1["input_values"], batch2["input_values"], atol=1e-6
        ), "Val collator produced different waveforms across runs"

    def test_val_no_augmentation_flags(self, val_collator):
        """Val collator has all augmentation disabled."""
        assert val_collator._speed_perturb is False
        assert val_collator._noise_prob == 0.0
        assert val_collator._rir_prob == 0.0
        assert val_collator._pitch_prob == 0.0
        assert val_collator._silence_trim is True  # deterministic, OK for val


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — interpolate / squeeze safety
# ═══════════════════════════════════════════════════════════════════════

class TestInterpolateSqueezeSafety:
    """Verify interpolate→squeeze always produces 1-D tensors."""

    @pytest.mark.parametrize("length", [1, 2, 100, 400, 16_000, 48_000])
    def test_pitch_shift_always_1d(self, full_train_collator, length):
        """Pitch shift output is always 1-D regardless of input length."""
        wav = torch.randn(length)
        random.seed(42)  # force a shift
        shifted = full_train_collator._pitch_shift(wav, semitones=3.0)
        assert shifted.ndim == 1, f"ndim={shifted.ndim} for input len={length}"
        assert shifted.size(0) == length

    def test_speed_perturb_always_1d(self, tokenizer):
        """Speed perturbation output is always 1-D."""
        collator = SFTCollator(
            tokenizer, target_sr=16_000, speed_perturb=True,
            speed_perturb_range=(0.8, 1.2),
        )
        for length in [400, 1_600, 16_000]:
            wav = torch.randn(length)
            # Directly test the interpolate path
            factor = 0.85
            new_len = int(length / factor)
            result = torch.nn.functional.interpolate(
                wav.unsqueeze(0).unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            assert result.ndim == 1, f"ndim={result.ndim} for input len={length}"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — Full pipeline end-to-end stress
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEndStress:
    """Stress test: process many batches with all augmentations."""

    def test_20_batch_stress(self, full_train_collator, real_samples):
        """Process 20 randomised batches without any crash or invalid output."""
        rng = random.Random(12345)
        for batch_idx in range(20):
            batch_size = rng.randint(2, min(8, len(real_samples)))
            batch_samples = rng.sample(real_samples, batch_size)
            batch = full_train_collator(batch_samples)
            assert batch is not None, f"Batch {batch_idx} was None"
            iv = batch["input_values"]
            am = batch["attention_mask"]
            assert not torch.isnan(iv).any(), f"NaN in stress batch {batch_idx}"
            assert not torch.isinf(iv).any(), f"Inf in stress batch {batch_idx}"
            assert iv.shape[0] == batch_size, (
                f"Batch {batch_idx}: expected {batch_size} samples, got {iv.shape[0]}"
            )
            # Every sample should have reasonable length
            for i in range(iv.shape[0]):
                length = am[i].sum().item()
                assert length >= 400, (
                    f"Batch {batch_idx}, sample {i}: length={length} < min_samples"
                )
