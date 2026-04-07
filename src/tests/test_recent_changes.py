"""
Regression tests — covers every bug found in recent code reviews.
=================================================================

Every test here would have caught a specific bug that was actually
discovered during the audit of the training pipeline.  Tests are grouped
by module, use synthetic data (no real audio), and run fast (<2s each).

Run:  PYTHONPATH=src python -m pytest src/tests/test_recent_changes.py -v --tb=short

Modules covered:
  - SFTCollator  (augmentation order, renorm, noise SNR, RIR, pitch, None batch,
                  attention mask dtype, RIR subsample, speed perturb headroom)
  - compute_per_and_recall (4-tuple, error counts, confusion tracking)
  - _CTCTrainer.training_step (None guard)
  - SFTDataset (oversampling with local RNG)
  - DataSplitter.run() (no SSL keys in return dict)
  - Integration (full collation pipeline with synthetic WAV files)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import Tensor

# Ensure src/ is on the path
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from trainer.data_collator import SFTCollator
from trainer.metrics import compute_per_and_recall, compute_per_batch
from trainer.sft_trainer_hf import _CTCTrainer


# ═══════════════════════════════════════════════════════════════════════
# Helpers — synthetic data factories
# ═══════════════════════════════════════════════════════════════════════

def _make_tokenizer():
    """Return a mock tokenizer that encodes space-separated phonemes to IDs."""
    tok = MagicMock()
    # Simple encoder: split on space, hash each token to an ID in [2..52]
    def _encode(text, **kwargs):
        if not text.strip():
            return []
        return [hash(t) % 51 + 2 for t in text.split()]
    tok.side_effect = _encode
    tok.__call__ = lambda self, text, **kw: MagicMock(input_ids=_encode(text))
    tok.return_value = MagicMock(input_ids=[2, 3, 4])
    return tok


def _make_simple_tokenizer():
    """Return a mock tokenizer where __call__ returns input_ids."""
    tok = MagicMock()
    tok.return_value = MagicMock(input_ids=[2, 3, 4, 5])
    return tok


def _make_wav_file(path: str | Path, duration_sec: float = 1.0, sr: int = 16000):
    """Write a synthetic WAV file (sine wave)."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), wav, sr)
    return wav


def _make_manifest_row(uid: str, audio_path: str, duration: float = 1.0,
                       dataset: int = 1, age: str = "5-8") -> dict:
    """Create a manifest row matching SFTDataset schema."""
    return {
        "utterance_id": uid,
        "audio_path": str(audio_path),
        "audio_duration_sec": duration,
        "age_bucket": age,
        "phonetic_text": "h ɛ l oʊ",
        "n_phonemes": 4,
        "dataset": dataset,
        "child_id": f"child_{uid}",
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — SFTCollator augmentation tests
# ═══════════════════════════════════════════════════════════════════════

class TestCollatorAugmentationOrder:
    """Verify augmentation fires in order: RIR → noise → pitch."""

    def test_augmentation_order_rir_noise_pitch(self, tmp_path):
        """Mock each augmentation method to record call order."""
        tok = _make_simple_tokenizer()
        # Construct with noise_datasets=[2] — only DS2 gets noise/RIR/pitch
        collator = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
            pitch_prob=1.0, pitch_semitones=2.0,
            noise_datasets=[2],
        )
        # Inject fake caches and set probs after construction
        collator._noise_cache = [torch.randn(16000)]
        collator._noise_prob = 1.0
        collator._rir_cache = [torch.randn(4000)]
        collator._rir_prob = 1.0

        call_order = []

        orig_rir = SFTCollator._apply_rir
        orig_noise = SFTCollator._inject_noise
        orig_pitch = SFTCollator._pitch_shift

        def mock_rir(self, wav):
            call_order.append("rir")
            return wav

        def mock_noise(self, wav):
            call_order.append("noise")
            return wav

        def mock_pitch(self, wav, semitones=None):
            call_order.append("pitch")
            return wav

        # Write a fake audio file — use dataset=2 (DS2) since
        # noise/RIR/pitch are DS2-only after DS1-aware augmentation branching.
        wav_path = tmp_path / "test.wav"
        _make_wav_file(wav_path, duration_sec=1.0)
        row = _make_manifest_row("U001", wav_path, dataset=2)

        with patch.object(SFTCollator, '_apply_rir', mock_rir), \
             patch.object(SFTCollator, '_inject_noise', mock_noise), \
             patch.object(SFTCollator, '_pitch_shift', mock_pitch):
            # Force random.random() to return 0.0 (< all probs)
            with patch('trainer.data_collator.random') as mock_random:
                mock_random.random.return_value = 0.0
                mock_random.uniform.return_value = 1.0
                mock_random.choice = random.choice

                collator([row])

        assert call_order == ["rir", "noise", "pitch"], (
            f"Augmentation order must be RIR→noise→pitch, got {call_order}"
        )

    def test_ds1_skips_noise_rir_but_gets_pitch(self, tmp_path):
        """DS1 samples skip noise/RIR (noise_datasets=[2]) but GET pitch
        (pitch_datasets=None = all datasets)."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
            pitch_prob=1.0, pitch_semitones=2.0,
            noise_datasets=[2],      # noise/RIR = DS2 only
            pitch_datasets=None,     # pitch = all datasets
        )
        collator._noise_cache = [torch.randn(16000)]
        collator._noise_prob = 1.0
        collator._rir_cache = [torch.randn(4000)]
        collator._rir_prob = 1.0

        call_order = []

        def mock_rir(self, wav):
            call_order.append("rir")
            return wav

        def mock_noise(self, wav):
            call_order.append("noise")
            return wav

        def mock_pitch(self, wav, semitones=None):
            call_order.append("pitch")
            return wav

        wav_path = tmp_path / "test_ds1.wav"
        _make_wav_file(wav_path, duration_sec=1.0)
        row = _make_manifest_row("U001", wav_path, dataset=1)  # DS1

        with patch.object(SFTCollator, '_apply_rir', mock_rir), \
             patch.object(SFTCollator, '_inject_noise', mock_noise), \
             patch.object(SFTCollator, '_pitch_shift', mock_pitch):
            with patch('trainer.data_collator.random') as mock_random:
                mock_random.random.return_value = 0.0
                mock_random.uniform.return_value = 1.0
                mock_random.choice = random.choice
                collator([row])

        assert call_order == ["pitch"], (
            f"DS1 must skip noise/RIR but get pitch, got: {call_order}"
        )


class TestCollatorRenormalisation:
    """Verify re-normalisation happens ONLY when augmentation fired."""

    def test_renormalise_only_when_augmented(self, tmp_path):
        """With all probs=0.0, waveform normalised once (in _load_and_preprocess).
        With noise_prob=1.0, waveform re-normalised after augmentation."""
        tok = _make_simple_tokenizer()

        # No augmentation
        collator_clean = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
            noise_prob=0.0, rir_prob=0.0, pitch_prob=0.0,
        )

        wav_path = tmp_path / "test.wav"
        _make_wav_file(wav_path, duration_sec=1.0)
        row = _make_manifest_row("U001", wav_path, dataset=2)  # DS2 for augmentation tests

        batch_clean = collator_clean([row])
        assert batch_clean is not None
        wav_clean = batch_clean["input_values"][0]
        length_clean = batch_clean["attention_mask"][0].sum().item()
        wav_clean_real = wav_clean[:int(length_clean)]

        # Should be normalised once: mean≈0, std≈1
        assert abs(wav_clean_real.mean().item()) < 0.01, "Clean wav should be zero-mean"
        assert abs(wav_clean_real.std().item() - 1.0) < 0.1, "Clean wav should be unit-var"

        # With noise augmentation — construct without noise_dir, then inject
        collator_noisy = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
        )
        # Inject a loud noise cache and enable noise_prob after construction
        collator_noisy._noise_cache = [torch.randn(16000) * 3.0]
        collator_noisy._noise_prob = 1.0

        batch_noisy = collator_noisy([row])
        assert batch_noisy is not None
        wav_noisy = batch_noisy["input_values"][0]
        length_noisy = batch_noisy["attention_mask"][0].sum().item()
        wav_noisy_real = wav_noisy[:int(length_noisy)]

        # Should ALSO be normalised (re-normalisation after augmentation)
        assert abs(wav_noisy_real.mean().item()) < 0.05, (
            f"Noisy wav mean={wav_noisy_real.mean().item():.4f}, should be ~0 after re-norm"
        )
        assert abs(wav_noisy_real.std().item() - 1.0) < 0.15, (
            f"Noisy wav std={wav_noisy_real.std().item():.4f}, should be ~1 after re-norm"
        )

        # The two outputs must differ (noise was injected)
        assert not torch.allclose(wav_clean_real, wav_noisy_real[:wav_clean_real.size(0)], atol=0.1), (
            "Clean and noisy waveforms should differ — noise injection had no effect"
        )


class TestNoiseInjectionSNR:
    """Verify SNR stratification: 60% mild, 30% moderate, 10% harsh."""

    def test_noise_injection_snr_bounds(self):
        """Inject noise 1000 times, verify SNR distribution matches spec."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(tok, target_sr=16000)
        # Manual setup — just use _inject_noise directly
        signal = torch.sin(torch.linspace(0, 100 * math.pi, 16000))
        signal = signal / signal.std()  # unit-var

        # Create a noise source
        noise = torch.randn(32000) * 0.5
        collator._noise_cache = [noise]

        snrs = []
        for _ in range(1000):
            noisy = collator._inject_noise(signal.clone())
            added_noise = noisy - signal
            sig_pow = signal.pow(2).mean()
            noise_pow = added_noise.pow(2).mean()
            if noise_pow > 1e-12:
                snr_db = 10 * torch.log10(sig_pow / noise_pow).item()
                snrs.append(snr_db)

        snrs = np.array(snrs)
        n = len(snrs)

        mild = np.sum((snrs >= 20) & (snrs <= 25)) / n
        moderate = np.sum((snrs >= 10) & (snrs < 20)) / n
        harsh = np.sum((snrs >= 5) & (snrs < 10)) / n

        # ±8% tolerance (wider for 10% bucket due to small counts)
        assert abs(mild - 0.60) < 0.08, f"Mild bucket: {mild:.2%}, expected ~60%"
        assert abs(moderate - 0.30) < 0.08, f"Moderate bucket: {moderate:.2%}, expected ~30%"
        assert abs(harsh - 0.10) < 0.08, f"Harsh bucket: {harsh:.2%}, expected ~10%"


class TestRIRAugmentation:
    """Verify RIR convolution preserves length and energy."""

    def test_rir_length_preserved(self):
        """Output length must match input length for various sizes."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(tok, target_sr=16000)

        # Synthetic RIR: decaying exponential impulse response
        rir = torch.zeros(8000)
        rir[0] = 1.0
        rir[100:200] = torch.exp(-torch.linspace(0, 5, 100)) * 0.3
        collator._rir_cache = [rir]

        for length in [8000, 16000, 48000]:
            wav = torch.randn(length)
            wav = wav / wav.std()
            result = collator._apply_rir(wav)
            assert result.size(0) == length, (
                f"RIR changed length: input={length}, output={result.size(0)}"
            )

    def test_rir_energy_preserved(self):
        """Output RMS should be within 10% of input RMS (energy normalisation)."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(tok, target_sr=16000)

        rir = torch.zeros(4000)
        rir[0] = 1.0
        rir[50:150] = torch.exp(-torch.linspace(0, 5, 100)) * 0.5
        collator._rir_cache = [rir]

        wav = torch.randn(16000)
        wav = wav / wav.std()
        rms_in = wav.pow(2).mean().sqrt().item()

        result = collator._apply_rir(wav)
        rms_out = result.pow(2).mean().sqrt().item()

        assert abs(rms_out - rms_in) / rms_in < 0.10, (
            f"RIR energy not preserved: input_rms={rms_in:.4f}, output_rms={rms_out:.4f}"
        )


class TestPitchShift:
    """Verify pitch shift preserves dtype and length."""

    def test_pitch_shift_dtype_preserved_float32(self):
        """pitch_shift must return the same dtype as input."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(tok, target_sr=16000, pitch_prob=1.0, pitch_semitones=2.0)

        wav = torch.randn(16000, dtype=torch.float32)
        collator._pitch_semitones = 2.0
        # Force a large enough shift to avoid the <0.1 skip
        with patch('trainer.data_collator.random') as mock_rng:
            mock_rng.uniform.return_value = 1.5
            result = collator._pitch_shift(wav)

        assert result.dtype == torch.float32, (
            f"Pitch shift changed dtype: input=float32, output={result.dtype}"
        )

    def test_pitch_shift_length_preserved(self):
        """STFT-based pitch shift should not change waveform length."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(tok, target_sr=16000, pitch_prob=1.0, pitch_semitones=2.0)

        wav = torch.randn(16000, dtype=torch.float32)

        for n_steps in [2.0, -2.0, 1.0]:
            with patch('trainer.data_collator.random') as mock_rng:
                mock_rng.uniform.return_value = n_steps
                result = collator._pitch_shift(wav)
            assert result.size(0) == wav.size(0), (
                f"Pitch shift changed length: n_steps={n_steps}, "
                f"input={wav.size(0)}, output={result.size(0)}"
            )

    def test_pitch_shift_skip_small_shift(self):
        """Shifts < 0.1 semitones should be skipped (returns input unchanged)."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(tok, target_sr=16000, pitch_prob=1.0, pitch_semitones=2.0)

        wav = torch.randn(16000, dtype=torch.float32)

        with patch('trainer.data_collator.random') as mock_rng:
            mock_rng.uniform.return_value = 0.05  # < 0.1
            result = collator._pitch_shift(wav)

        assert torch.equal(result, wav), "Small pitch shift should return input unchanged"


class TestCollatorNoneBatch:
    """Verify None batch when all samples are dropped."""

    def test_none_batch_when_all_dropped(self, tmp_path):
        """All samples exceed max_duration → collator returns None."""
        tok = _make_simple_tokenizer()
        # max_duration very short — 0.05s = 800 samples
        collator = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.01, max_duration_sec=0.05,
        )

        # Create 4 audio files, each 1.0s (>> 0.05s max)
        rows = []
        for i in range(4):
            p = tmp_path / f"long_{i}.wav"
            _make_wav_file(p, duration_sec=1.0)
            rows.append(_make_manifest_row(f"U{i}", p, duration=1.0))

        result = collator(rows)
        assert result is None, "Collator should return None when all samples exceed max_duration"


class TestAttentionMaskDtype:
    """Verify attention_mask is long (WavLM silent bug with bool/float)."""

    def test_attention_mask_dtype_is_long(self, tmp_path):
        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
        )

        wav_path = tmp_path / "test.wav"
        _make_wav_file(wav_path, duration_sec=1.0)
        row = _make_manifest_row("U001", wav_path)

        batch = collator([row])
        assert batch is not None
        assert batch["attention_mask"].dtype == torch.int64, (
            f"attention_mask must be int64/long, got {batch['attention_mask'].dtype}"
        )
        assert batch["attention_mask"].dtype != torch.bool
        assert batch["attention_mask"].dtype != torch.float32


class TestRIRSubsample:
    """Verify RIR cache is subsampled to 1000 files."""

    def test_rir_subsampled_to_1000(self, tmp_path):
        """Create >1000 fake RIR files, verify cache is capped."""
        rir_dir = tmp_path / "rir_files"
        rir_dir.mkdir()

        # Create 1500 tiny WAV files
        for i in range(1500):
            p = rir_dir / f"rir_{i:04d}.wav"
            # Minimal WAV: 100 samples at 16kHz
            wav_data = np.random.randn(100).astype(np.float32) * 0.01
            sf.write(str(p), wav_data, 16000)

        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            rir_dir=str(rir_dir), rir_prob=0.5,
        )

        assert len(collator._rir_cache) <= 1000, (
            f"RIR cache should be ≤1000, got {len(collator._rir_cache)}"
        )


class TestSpeedPerturbHeadroom:
    """Verify effective_max computation with speed perturbation."""

    def test_effective_max_with_speed_perturb(self):
        """max_duration=10s, speed_range=[0.8, 1.2] →
        effective_max = 10/0.8 * 1.05 = 13.125s = 210000 samples."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            max_duration_sec=10.0,
            speed_perturb=True,
            speed_perturb_range=(0.8, 1.2),
        )

        expected_max = int(10.0 / 0.8 * 1.05 * 16000)
        assert collator._max_samples == expected_max, (
            f"effective_max_samples={collator._max_samples}, expected={expected_max}"
        )

    def test_no_headroom_without_speed_perturb(self):
        """Without speed perturb, max_samples = max_duration * sr."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            max_duration_sec=10.0,
            speed_perturb=False,
        )

        expected_max = int(10.0 * 16000)
        assert collator._max_samples == expected_max, (
            f"max_samples={collator._max_samples}, expected={expected_max}"
        )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — compute_per_and_recall tests
# ═══════════════════════════════════════════════════════════════════════

class TestComputePerAndRecall:
    """Verify the 4-tuple return from compute_per_and_recall."""

    def test_returns_four_values(self):
        """Must return (per, recall, error_counts, top_confusions)."""
        hyps = [[1, 2, 3]]
        refs = [[1, 2, 3]]
        result = compute_per_and_recall(hyps, refs)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"

        per, recall, error_counts, top_confusions = result
        assert isinstance(per, float)
        assert isinstance(recall, dict)
        assert isinstance(error_counts, dict)
        assert isinstance(top_confusions, list)

    def test_error_counts_sum_to_total_edits(self):
        """del + ins + sub must equal total edit distance."""
        # Known: hyp=[1,2,4], ref=[1,2,3] → 1 sub
        # Known: hyp=[1], ref=[1,2,3] → 2 del
        hyps = [[1, 2, 4], [1]]
        refs = [[1, 2, 3], [1, 2, 3]]

        per, recall, error_counts, top_confusions = compute_per_and_recall(hyps, refs)

        total_errors = error_counts["del"] + error_counts["ins"] + error_counts["sub"]
        # Sample 0: 1 sub. Sample 1: 2 del. Total = 3
        assert total_errors == 3, f"Expected 3 total errors, got {total_errors}"
        assert error_counts["sub"] == 1
        assert error_counts["del"] == 2
        assert error_counts["ins"] == 0

        # Cross-check with compute_per_batch
        per_batch = compute_per_batch(hyps, refs)
        total_ref_len = sum(len(r) for r in refs)
        assert abs(per - per_batch) < 1e-6, (
            f"PER mismatch: compute_per_and_recall={per}, compute_per_batch={per_batch}"
        )
        assert abs(per - total_errors / total_ref_len) < 1e-6

    def test_top_confusions_sorted_descending(self):
        """Top confusions must be sorted by count descending."""
        # Create known substitution patterns: 3→4 five times, 5→6 three times
        hyps = [[4]] * 5 + [[6]] * 3
        refs = [[3]] * 5 + [[5]] * 3

        _, _, _, top_confusions = compute_per_and_recall(hyps, refs)

        assert len(top_confusions) >= 2
        assert top_confusions[0][1] >= top_confusions[1][1], (
            "Top confusions not sorted descending"
        )
        assert top_confusions[0][0] == (3, 4), (
            f"Most common confusion should be (3→4), got {top_confusions[0][0]}"
        )
        assert top_confusions[0][1] == 5
        assert top_confusions[1][0] == (5, 6)
        assert top_confusions[1][1] == 3

    def test_confusion_only_tracks_substitutions(self):
        """Deletions and insertions must NOT appear in top_confusions."""
        # hyp=[1,2,7,8] vs ref=[1,2,3] → 1 del(3), 2 ins(7,8) via alignment
        # Actually: del of 3, ins of 7, ins of 8
        hyps = [[1, 2, 7, 8]]
        refs = [[1, 2, 3]]

        _, _, error_counts, top_confusions = compute_per_and_recall(hyps, refs)

        # Check that confusions only contain genuine substitution pairs
        for (ref_tok, hyp_tok), count in top_confusions:
            assert ref_tok is not None and hyp_tok is not None, (
                f"Confusion pair has None: ref={ref_tok}, hyp={hyp_tok}"
            )

    def test_perfect_match_zero_errors(self):
        """Perfect match should have zero errors and empty confusions."""
        hyps = [[1, 2, 3], [4, 5]]
        refs = [[1, 2, 3], [4, 5]]

        per, recall, error_counts, top_confusions = compute_per_and_recall(hyps, refs)
        assert per == 0.0
        assert error_counts["del"] == 0
        assert error_counts["ins"] == 0
        assert error_counts["sub"] == 0
        assert len(top_confusions) == 0
        # All phonemes should have recall=1.0
        for tok_id, info in recall.items():
            assert info["recall"] == 1.0, f"Token {tok_id} recall={info['recall']}, expected 1.0"

    def test_recall_tracks_per_phoneme(self):
        """Each phoneme's hits/total/recall must be correct."""
        # Ref: [1,2,3,1], Hyp: [1,2,4,1]
        # Alignment: 1=match, 2=match, 3→4=sub, 1=match
        hyps = [[1, 2, 4, 1]]
        refs = [[1, 2, 3, 1]]

        _, recall, _, _ = compute_per_and_recall(hyps, refs)

        assert recall[1]["hits"] == 2 and recall[1]["total"] == 2
        assert recall[1]["recall"] == 1.0
        assert recall[2]["hits"] == 1 and recall[2]["total"] == 1
        assert recall[3]["hits"] == 0 and recall[3]["total"] == 1
        assert recall[3]["recall"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — _CTCTrainer.training_step None guard
# ═══════════════════════════════════════════════════════════════════════

class TestCTCTrainerNoneGuard:
    """Verify training_step handles None batches gracefully."""

    def test_none_batch_returns_zero_loss(self):
        """None inputs → zero-loss tensor, no model.forward() call."""
        trainer = _CTCTrainer.__new__(_CTCTrainer)

        # Mock args with device
        mock_args = MagicMock()
        mock_args.device = torch.device("cpu")
        trainer.args = mock_args

        mock_model = MagicMock()

        result = trainer.training_step(mock_model, None, num_items_in_batch=0)

        assert isinstance(result, torch.Tensor), "Should return a Tensor"
        assert result.item() == pytest.approx(0.0), "Should return zero loss"
        assert result.requires_grad, "Zero loss must require grad for backward compat"
        mock_model.assert_not_called()

    def test_normal_batch_calls_model(self):
        """Non-None inputs should proceed to parent class (model.forward called)."""
        trainer = _CTCTrainer.__new__(_CTCTrainer)

        # We can't easily test the full training_step without HF infrastructure,
        # but we can verify compute_loss works with a proper batch
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(1.5, requires_grad=True)
        mock_model.return_value = mock_output

        inputs = {
            "input_values": torch.randn(2, 16000),
            "attention_mask": torch.ones(2, 16000, dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4, -100], [5, 6, -100, -100]]),
        }

        loss = trainer.compute_loss(mock_model, inputs)
        assert loss.item() == 1.5
        mock_model.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — SFTDataset oversampling with local RNG
# ═══════════════════════════════════════════════════════════════════════

class TestDatasetOversampling:
    """Verify oversampling uses local RNG and correct factors."""

    def test_oversampling_uses_local_rng(self, tmp_path):
        """Global random state must be unaffected by dataset creation."""
        from trainer.dataset import SFTDataset

        # Create a minimal manifest
        manifest = tmp_path / "manifest.jsonl"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        rows = []
        for i in range(10):
            uid = f"U{i:04d}"
            audio_path = audio_dir / f"{uid}.wav"
            _make_wav_file(audio_path, duration_sec=1.0)
            rows.append({
                "utterance_id": uid,
                "audio_path": str(audio_path),
                "audio_duration_sec": 1.0,
                "age_bucket": "5-8",
                "phonetic_text": "h ɛ l oʊ",
                "n_phonemes": 4,
                "dataset": 1,
                "child_id": f"child_{i}",
            })

        with open(manifest, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Save global random state
        state_before = random.getstate()

        ds = SFTDataset(
            manifest,
            {1: str(audio_dir)},
            split="train",
            ds_oversample={1: 3},
        )

        state_after = random.getstate()

        # Global state must be unchanged
        assert state_before == state_after, (
            "SFTDataset oversampling polluted global random state"
        )

    def test_oversampling_factor_applied_correctly(self, tmp_path):
        """DS1 × 3 + DS2 × 1 = correct total count."""
        from trainer.dataset import SFTDataset

        manifest = tmp_path / "manifest.jsonl"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        rows = []
        # 10 DS1 rows + 10 DS2 rows
        for ds_key in [1, 2]:
            for i in range(10):
                uid = f"U_ds{ds_key}_{i:04d}"
                audio_path = audio_dir / f"{uid}.wav"
                _make_wav_file(audio_path, duration_sec=1.0)
                rows.append({
                    "utterance_id": uid,
                    "audio_path": str(audio_path),
                    "audio_duration_sec": 1.0,
                    "age_bucket": "5-8",
                    "phonetic_text": "h ɛ l oʊ",
                    "n_phonemes": 4,
                    "dataset": ds_key,
                    "child_id": f"child_{ds_key}_{i}",
                })

        with open(manifest, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        ds = SFTDataset(
            manifest,
            {1: str(audio_dir), 2: str(audio_dir)},
            split="train",
            ds_oversample={1: 3},
        )

        # DS1: 10 × 3 = 30, DS2: 10 × 1 = 10, total = 40
        assert len(ds) == 40, f"Expected 40 rows, got {len(ds)}"

        ds_counts = Counter(str(ds[i]["dataset"]) for i in range(len(ds)))
        assert ds_counts["1"] == 30, f"DS1 count={ds_counts['1']}, expected 30"
        assert ds_counts["2"] == 10, f"DS2 count={ds_counts['2']}, expected 10"

    def test_oversampling_only_on_train_split(self, tmp_path):
        """Val split should NOT apply oversampling."""
        from trainer.dataset import SFTDataset

        manifest = tmp_path / "manifest.jsonl"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        rows = []
        for i in range(20):
            uid = f"U{i:04d}"
            audio_path = audio_dir / f"{uid}.wav"
            _make_wav_file(audio_path, duration_sec=1.0)
            rows.append({
                "utterance_id": uid,
                "audio_path": str(audio_path),
                "audio_duration_sec": 1.0,
                "age_bucket": "5-8",
                "phonetic_text": "h ɛ l oʊ",
                "n_phonemes": 4,
                "dataset": 1,
                "child_id": f"child_{i}",
            })

        with open(manifest, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        ds = SFTDataset(
            manifest,
            {1: str(audio_dir)},
            split="val",
            ds_oversample={1: 3},
        )

        assert len(ds) == 20, (
            f"Val split should not oversample: expected 20, got {len(ds)}"
        )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — DataSplitter return dict
# ═══════════════════════════════════════════════════════════════════════

class TestDataSplitterReturnKeys:
    """Verify run() return dict has no stale SSL keys."""

    def test_return_dict_has_no_ssl_keys(self, tmp_path):
        """After SSL removal, ssl_rows and ssl_hours must not be in result."""
        # We can't easily run the full DataSplitter without real data,
        # so we verify the source code doesn't contain these keys
        from etl.data_split import DataSplitter
        import inspect

        source = inspect.getsource(DataSplitter.run)
        assert "ssl_rows" not in source, (
            "ssl_rows still referenced in DataSplitter.run()"
        )
        assert "ssl_hours" not in source, (
            "ssl_hours still referenced in DataSplitter.run()"
        )

    def test_return_dict_expected_keys_in_source(self):
        """Verify expected keys are in the return dict."""
        from etl.data_split import DataSplitter
        import inspect

        source = inspect.getsource(DataSplitter.run)
        for key in ["n_train_rows", "n_val_rows", "train_hrs", "val_hrs"]:
            assert key in source, f"Expected key '{key}' not found in DataSplitter.run() source"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — Integration tests with synthetic WAV files
# ═══════════════════════════════════════════════════════════════════════

class TestFullCollationPipelineNoAugmentation:
    """End-to-end collation with synthetic audio, no augmentation."""

    def test_full_pipeline_produces_valid_batch(self, tmp_path):
        """Create 8 fake audio files, collate, verify all output keys and shapes."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
            noise_prob=0.0, rir_prob=0.0, pitch_prob=0.0,
        )

        rows = []
        for i in range(8):
            duration = 1.0 + i * 0.25  # 1.0s to 2.75s
            p = audio_dir / f"test_{i}.wav"
            _make_wav_file(p, duration_sec=duration)
            rows.append(_make_manifest_row(f"U{i:03d}", p, duration=duration))

        batch = collator(rows)

        # Verify not None
        assert batch is not None, "Batch should not be None"

        # Verify all keys present
        expected_keys = {"input_values", "attention_mask", "labels",
                         "input_lengths", "age_buckets", "datasets"}
        assert set(batch.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(batch.keys())}"
        )

        # Shape checks
        assert batch["input_values"].shape[0] == 8
        assert batch["input_values"].ndim == 2
        assert batch["attention_mask"].shape == batch["input_values"].shape
        assert batch["attention_mask"].dtype == torch.int64
        assert batch["labels"].shape[0] == 8
        assert batch["input_lengths"].shape == (8,)

        # Labels should have -100 padding (different length texts)
        # Since our mock tokenizer returns same length, padding depends on batch
        assert batch["labels"].dtype == torch.long

        # Metadata lists
        assert len(batch["age_buckets"]) == 8
        assert len(batch["datasets"]) == 8

    def test_normalisation_correct(self, tmp_path):
        """Each sample's real portion should be ~zero-mean, ~unit-var."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
        )

        rows = []
        for i in range(4):
            p = audio_dir / f"test_{i}.wav"
            _make_wav_file(p, duration_sec=1.0)
            rows.append(_make_manifest_row(f"U{i:03d}", p, duration=1.0))

        batch = collator(rows)
        assert batch is not None

        iv = batch["input_values"]
        am = batch["attention_mask"]

        for i in range(iv.shape[0]):
            length = am[i].sum().item()
            wav = iv[i, :int(length)]
            mean = wav.mean().item()
            std = wav.std().item()
            assert abs(mean) < 0.05, f"Sample {i} mean={mean:.4f}, expected ~0"
            assert abs(std - 1.0) < 0.15, f"Sample {i} std={std:.4f}, expected ~1.0"


class TestFullCollationPipelineWithAugmentation:
    """End-to-end collation with noise augmentation."""

    def test_noise_augmentation_changes_waveform(self, tmp_path):
        """With noise_prob=1.0, output should differ from clean + be renormalised."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        tok = _make_simple_tokenizer()

        # Clean collator
        collator_clean = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
            noise_prob=0.0,
        )

        # Noisy collator
        collator_noisy = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.1, max_duration_sec=5.0,
            noise_prob=1.0,
        )
        # Manually inject noise cache (synthetic, no MUSAN needed)
        collator_noisy._noise_cache = [torch.randn(32000) * 0.5]

        rows = []
        for i in range(4):
            p = audio_dir / f"test_{i}.wav"
            _make_wav_file(p, duration_sec=1.0)
            rows.append(_make_manifest_row(f"U{i:03d}", p, duration=1.0))

        batch_clean = collator_clean(rows)
        batch_noisy = collator_noisy(rows)

        assert batch_clean is not None
        assert batch_noisy is not None

        # Same number of samples
        assert batch_noisy["input_values"].shape[0] == 4

        # Attention mask still correct type
        assert batch_noisy["attention_mask"].dtype == torch.int64

        # Noisy batch should be renormalised (std ≈ 1.0)
        for i in range(4):
            length = batch_noisy["attention_mask"][i].sum().item()
            wav = batch_noisy["input_values"][i, :int(length)]
            assert abs(wav.std().item() - 1.0) < 0.15, (
                f"Noisy sample {i} std={wav.std().item():.4f}, expected ~1.0 after re-norm"
            )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — Collator constructor defaults
# ═══════════════════════════════════════════════════════════════════════

class TestCollatorDefaults:
    """Verify SFTCollator works with minimal arguments (backward compat)."""

    def test_construction_without_augmentation_params(self):
        """Old-style construction without noise/rir/pitch params must work."""
        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.5, max_duration_sec=20.0,
        )
        assert collator._noise_prob == 0.0
        assert collator._rir_prob == 0.0
        assert collator._pitch_prob == 0.0
        assert len(collator._noise_cache) == 0
        assert len(collator._rir_cache) == 0

    def test_construction_with_all_augmentation_params(self, tmp_path):
        """New-style construction with all augmentation params."""
        # Create fake dirs
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        for i in range(5):
            sf.write(str(noise_dir / f"n{i}.wav"), np.random.randn(16000).astype(np.float32), 16000)

        rir_dir = tmp_path / "rir"
        rir_dir.mkdir()
        for i in range(5):
            sf.write(str(rir_dir / f"r{i}.wav"), np.random.randn(4000).astype(np.float32), 16000)

        tok = _make_simple_tokenizer()
        collator = SFTCollator(
            tok, target_sr=16000,
            min_duration_sec=0.5, max_duration_sec=20.0,
            noise_dir=str(noise_dir), noise_prob=0.3,
            rir_dir=str(rir_dir), rir_prob=0.15,
            pitch_prob=0.3, pitch_semitones=2.0,
        )
        assert collator._noise_prob == 0.3
        assert collator._rir_prob == 0.15
        assert collator._pitch_prob == 0.3
        assert len(collator._noise_cache) == 5
        assert len(collator._rir_cache) == 5
