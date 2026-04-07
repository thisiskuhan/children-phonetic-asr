"""
Tests for checkpoint averaging logic used by --avg flag.

Tests:
  1. _load_and_average — uniform averaging math is correct (fp32 accumulation)
  2. _rank_checkpoints — reads trainer_state.json, sorts by eval_per ascending
  3. Edge cases — single checkpoint, mismatched keys, missing files
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

# The methods under test are static, so we can import them directly
# without instantiating the full trainer.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trainer.sft_trainer_hf import HFSFTTrainer


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    """Create a temp dir with N mock checkpoints for averaging tests."""
    return tmp_path / "hf_sft_checkpoints"


def _make_checkpoint(
    ckpt_dir: Path,
    step: int,
    eval_per: float,
    weights: dict[str, torch.Tensor],
    use_safetensors: bool = True,
) -> Path:
    """Create a mock checkpoint-{step}/ directory with weights + trainer_state."""
    d = ckpt_dir / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)

    # Write weights
    if use_safetensors:
        from safetensors.torch import save_file

        save_file(weights, str(d / "model.safetensors"))
    else:
        torch.save(weights, d / "pytorch_model.bin")

    # Write trainer_state.json with eval_per in log_history
    state = {
        "log_history": [
            {"loss": 1.0, "step": step},
            {"eval_loss": 0.5, "eval_per": eval_per, "step": step},
        ]
    }
    (d / "trainer_state.json").write_text(json.dumps(state))

    # Write config.json (for avg_only to copy)
    (d / "config.json").write_text(json.dumps({"model_type": "wavlm"}))

    return d


# ── Test _load_and_average ────────────────────────────────────────────


class TestLoadAndAverage:
    """Verify the averaging math is correct."""

    def test_two_checkpoints_uniform_average(self, tmp_ckpt_dir):
        """Average of [2, 4] should be [3]."""
        w1 = {"layer.weight": torch.tensor([2.0, 4.0, 6.0])}
        w2 = {"layer.weight": torch.tensor([4.0, 8.0, 12.0])}

        d1 = _make_checkpoint(tmp_ckpt_dir, 100, 0.30, w1)
        d2 = _make_checkpoint(tmp_ckpt_dir, 200, 0.25, w2)

        avg = HFSFTTrainer._load_and_average([d1, d2])

        expected = torch.tensor([3.0, 6.0, 9.0])
        assert torch.allclose(avg["layer.weight"], expected, atol=1e-6)

    def test_three_checkpoints_uniform_average(self, tmp_ckpt_dir):
        """Average of [1, 2, 3] = [2]."""
        w1 = {"w": torch.tensor([1.0, 10.0])}
        w2 = {"w": torch.tensor([2.0, 20.0])}
        w3 = {"w": torch.tensor([3.0, 30.0])}

        d1 = _make_checkpoint(tmp_ckpt_dir, 100, 0.30, w1)
        d2 = _make_checkpoint(tmp_ckpt_dir, 200, 0.25, w2)
        d3 = _make_checkpoint(tmp_ckpt_dir, 300, 0.20, w3)

        avg = HFSFTTrainer._load_and_average([d1, d2, d3])

        expected = torch.tensor([2.0, 20.0])
        assert torch.allclose(avg["w"], expected, atol=1e-6)

    def test_fp32_accumulation(self, tmp_ckpt_dir):
        """Averaging must happen in fp32 even if weights are fp16/bf16."""
        # Use bf16 weights — small values that would lose precision
        w1 = {"w": torch.tensor([1.0, 0.0001]).bfloat16()}
        w2 = {"w": torch.tensor([1.0, 0.0003]).bfloat16()}

        d1 = _make_checkpoint(tmp_ckpt_dir, 100, 0.30, w1)
        d2 = _make_checkpoint(tmp_ckpt_dir, 200, 0.25, w2)

        avg = HFSFTTrainer._load_and_average([d1, d2])

        # Result should be fp32
        assert avg["w"].dtype == torch.float32
        # Average of [0.0001, 0.0003] = 0.0002 (approx, within bf16 input error)
        assert avg["w"][0].item() == pytest.approx(1.0, abs=0.01)
        assert avg["w"][1].item() == pytest.approx(0.0002, abs=0.001)

    def test_multiple_keys(self, tmp_ckpt_dir):
        """All keys in the state dict should be averaged independently."""
        w1 = {"a": torch.tensor([1.0]), "b": torch.tensor([10.0]), "c": torch.tensor([100.0])}
        w2 = {"a": torch.tensor([3.0]), "b": torch.tensor([30.0]), "c": torch.tensor([300.0])}

        d1 = _make_checkpoint(tmp_ckpt_dir, 100, 0.30, w1)
        d2 = _make_checkpoint(tmp_ckpt_dir, 200, 0.25, w2)

        avg = HFSFTTrainer._load_and_average([d1, d2])

        assert torch.allclose(avg["a"], torch.tensor([2.0]))
        assert torch.allclose(avg["b"], torch.tensor([20.0]))
        assert torch.allclose(avg["c"], torch.tensor([200.0]))

    def test_key_mismatch_raises(self, tmp_ckpt_dir):
        """Mismatched keys between checkpoints should raise RuntimeError."""
        w1 = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        w2 = {"a": torch.tensor([3.0]), "c": torch.tensor([4.0])}  # different key

        d1 = _make_checkpoint(tmp_ckpt_dir, 100, 0.30, w1)
        d2 = _make_checkpoint(tmp_ckpt_dir, 200, 0.25, w2)

        with pytest.raises(RuntimeError, match="Key mismatch"):
            HFSFTTrainer._load_and_average([d1, d2])

    def test_single_checkpoint(self, tmp_ckpt_dir):
        """Single checkpoint should return itself (divided by 1 = same)."""
        w1 = {"w": torch.tensor([42.0, 7.0])}
        d1 = _make_checkpoint(tmp_ckpt_dir, 100, 0.30, w1)

        avg = HFSFTTrainer._load_and_average([d1])
        assert torch.allclose(avg["w"], torch.tensor([42.0, 7.0]))

    def test_pytorch_bin_fallback(self, tmp_ckpt_dir):
        """Should load pytorch_model.bin when safetensors not available."""
        w1 = {"w": torch.tensor([2.0])}
        w2 = {"w": torch.tensor([4.0])}

        d1 = _make_checkpoint(tmp_ckpt_dir, 100, 0.30, w1, use_safetensors=False)
        d2 = _make_checkpoint(tmp_ckpt_dir, 200, 0.25, w2, use_safetensors=False)

        avg = HFSFTTrainer._load_and_average([d1, d2])
        assert torch.allclose(avg["w"], torch.tensor([3.0]))

    def test_result_is_ordered_dict(self, tmp_ckpt_dir):
        """Result should be an OrderedDict for model.load_state_dict compat."""
        w1 = {"w": torch.tensor([1.0])}
        d1 = _make_checkpoint(tmp_ckpt_dir, 100, 0.30, w1)

        avg = HFSFTTrainer._load_and_average([d1])
        assert isinstance(avg, OrderedDict)

    def test_five_checkpoint_average(self, tmp_ckpt_dir):
        """Realistic scenario: average 5 checkpoints with 2D tensors."""
        weights = []
        for i in range(5):
            w = {"encoder.weight": torch.randn(64, 32)}
            weights.append(w)

        dirs = []
        for i, w in enumerate(weights):
            d = _make_checkpoint(tmp_ckpt_dir, (i + 1) * 100, 0.30 - i * 0.01, w)
            dirs.append(d)

        avg = HFSFTTrainer._load_and_average(dirs)

        # Manually compute expected average
        expected = sum(w["encoder.weight"].float() for w in weights) / 5.0
        assert torch.allclose(avg["encoder.weight"], expected, atol=1e-5)


# ── Test _rank_checkpoints ────────────────────────────────────────────


class TestRankCheckpoints:
    """Verify checkpoint ranking by eval_per."""

    def test_sorts_ascending(self, tmp_ckpt_dir):
        """Best PER (lowest) should be first."""
        _make_checkpoint(tmp_ckpt_dir, 100, 0.30, {"w": torch.tensor([1.0])})
        _make_checkpoint(tmp_ckpt_dir, 200, 0.20, {"w": torch.tensor([1.0])})
        _make_checkpoint(tmp_ckpt_dir, 300, 0.25, {"w": torch.tensor([1.0])})

        ranked = HFSFTTrainer._rank_checkpoints(tmp_ckpt_dir)

        assert len(ranked) == 3
        assert ranked[0][1] == pytest.approx(0.20)  # best
        assert ranked[1][1] == pytest.approx(0.25)
        assert ranked[2][1] == pytest.approx(0.30)  # worst

    def test_skips_dir_without_trainer_state(self, tmp_ckpt_dir):
        """Checkpoints without trainer_state.json should be skipped."""
        _make_checkpoint(tmp_ckpt_dir, 100, 0.30, {"w": torch.tensor([1.0])})

        # Create a directory without trainer_state.json
        bad = tmp_ckpt_dir / "checkpoint-999"
        bad.mkdir(parents=True)
        from safetensors.torch import save_file

        save_file({"w": torch.tensor([1.0])}, str(bad / "model.safetensors"))

        ranked = HFSFTTrainer._rank_checkpoints(tmp_ckpt_dir)
        assert len(ranked) == 1

    def test_skips_dir_without_weights(self, tmp_ckpt_dir):
        """Checkpoints without model weights should be skipped."""
        _make_checkpoint(tmp_ckpt_dir, 100, 0.30, {"w": torch.tensor([1.0])})

        # Create directory with trainer_state but no weights
        bad = tmp_ckpt_dir / "checkpoint-999"
        bad.mkdir(parents=True)
        state = {"log_history": [{"eval_per": 0.10, "step": 999}]}
        (bad / "trainer_state.json").write_text(json.dumps(state))

        ranked = HFSFTTrainer._rank_checkpoints(tmp_ckpt_dir)
        assert len(ranked) == 1

    def test_empty_dir(self, tmp_ckpt_dir):
        """Empty directory should return empty list."""
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
        ranked = HFSFTTrainer._rank_checkpoints(tmp_ckpt_dir)
        assert ranked == []

    def test_reads_last_eval_per(self, tmp_ckpt_dir):
        """Should pick the LAST eval_per from log_history (not first)."""
        d = tmp_ckpt_dir / "checkpoint-100"
        d.mkdir(parents=True)
        from safetensors.torch import save_file

        save_file({"w": torch.tensor([1.0])}, str(d / "model.safetensors"))

        # log_history has multiple eval entries — should use the LAST one
        state = {
            "log_history": [
                {"eval_per": 0.50, "step": 50},
                {"eval_per": 0.40, "step": 75},
                {"eval_per": 0.30, "step": 100},  # last = this one
            ]
        }
        (d / "trainer_state.json").write_text(json.dumps(state))

        ranked = HFSFTTrainer._rank_checkpoints(tmp_ckpt_dir)
        assert len(ranked) == 1
        assert ranked[0][1] == pytest.approx(0.30)

    def test_returns_path_objects(self, tmp_ckpt_dir):
        """Ranked entries should contain Path objects."""
        _make_checkpoint(tmp_ckpt_dir, 100, 0.30, {"w": torch.tensor([1.0])})

        ranked = HFSFTTrainer._rank_checkpoints(tmp_ckpt_dir)
        assert isinstance(ranked[0][0], Path)
        assert ranked[0][0].name == "checkpoint-100"
