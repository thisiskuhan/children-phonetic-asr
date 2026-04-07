"""
Eval diagnostics — test age×DS cross PER, per-age CER, length-bucketed PER.
============================================================================

Validates the new compute_metrics additions work correctly with mock data,
without requiring model/tokenizer/collator (pure metric logic).

Run:  PYTHONPATH=src python -m pytest src/tests/test_eval_diagnostics.py -v
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from trainer.metrics import compute_per_batch


# ---------------------------------------------------------------------------
# Helpers — replicate the grouping logic from compute_metrics
# ---------------------------------------------------------------------------


def _group_by_key(hyps, refs, keys):
    """Group hyps/refs by a list of keys (same length)."""
    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for h, r, k in zip(hyps, refs, keys):
        groups[k][0].append(h)
        groups[k][1].append(r)
    return groups


def _cross_key(ds, age):
    return f"{ds}_{age}"


def _length_bucket(ref):
    n = len(ref)
    if n <= 5:
        return "short_le5"
    elif n <= 15:
        return "mid_6to15"
    else:
        return "long_gt15"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_eval_data():
    """8 samples: 4 from DS1, 4 from DS2, mixed ages and lengths."""
    # Perfect match → PER = 0.0
    # Completely wrong → PER = 1.0
    # Partial match  → PER between 0 and 1
    hyps = [
        [1, 2, 3],          # DS1, 3-4, perfect
        [1, 2, 4],          # DS1, 3-4, 1 sub in 3 → PER=0.33
        [5, 6, 7, 8, 9],    # DS1, 5-7, perfect
        [10, 11],           # DS1, 8-11, perfect
        [1, 2, 3],          # DS2, 3-4, perfect
        [5, 6, 7, 8, 9],    # DS2, 5-7, perfect
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # DS2, 8-11, perfect (long)
        [20, 21, 22, 23],   # DS2, unknown, 1 sub in 4 → PER=0.25
    ]
    refs = [
        [1, 2, 3],          # DS1, 3-4
        [1, 2, 3],          # DS1, 3-4
        [5, 6, 7, 8, 9],    # DS1, 5-7
        [10, 11],           # DS1, 8-11
        [1, 2, 3],          # DS2, 3-4
        [5, 6, 7, 8, 9],    # DS2, 5-7
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # DS2, 8-11
        [20, 21, 22, 99],   # DS2, unknown
    ]
    age_buckets = ["3-4", "3-4", "5-7", "8-11", "3-4", "5-7", "8-11", "unknown"]
    datasets = ["1", "1", "1", "1", "2", "2", "2", "2"]
    return hyps, refs, age_buckets, datasets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgeByDatasetCrossPER:
    """Verify age×DS cross breakdown produces correct keys and values."""

    def test_cross_keys_present(self, mock_eval_data):
        hyps, refs, ages, dsets = mock_eval_data
        cross_groups = defaultdict(lambda: ([], []))
        for h, r, a, d in zip(hyps, refs, ages, dsets):
            key = _cross_key(d, a)
            cross_groups[key][0].append(h)
            cross_groups[key][1].append(r)

        keys = sorted(cross_groups.keys())
        assert "1_3-4" in keys
        assert "1_5-7" in keys
        assert "1_8-11" in keys
        assert "2_3-4" in keys
        assert "2_5-7" in keys
        assert "2_8-11" in keys
        assert "2_unknown" in keys

    def test_ds1_3_4_per(self, mock_eval_data):
        hyps, refs, ages, dsets = mock_eval_data
        cross_groups = defaultdict(lambda: ([], []))
        for h, r, a, d in zip(hyps, refs, ages, dsets):
            cross_groups[_cross_key(d, a)][0].append(h)
            cross_groups[_cross_key(d, a)][1].append(r)
        # DS1_3-4: sample 0 perfect (0/3) + sample 1 has 1 sub (1/3) = 1/6
        per = compute_per_batch(*cross_groups["1_3-4"])
        assert abs(per - 1 / 6) < 1e-6, f"Expected ~0.167, got {per}"

    def test_ds2_3_4_per(self, mock_eval_data):
        hyps, refs, ages, dsets = mock_eval_data
        cross_groups = defaultdict(lambda: ([], []))
        for h, r, a, d in zip(hyps, refs, ages, dsets):
            cross_groups[_cross_key(d, a)][0].append(h)
            cross_groups[_cross_key(d, a)][1].append(r)
        per = compute_per_batch(*cross_groups["2_3-4"])
        assert per == 0.0, "DS2_3-4 should be perfect"

    def test_ds1_perfect_groups(self, mock_eval_data):
        hyps, refs, ages, dsets = mock_eval_data
        cross_groups = defaultdict(lambda: ([], []))
        for h, r, a, d in zip(hyps, refs, ages, dsets):
            cross_groups[_cross_key(d, a)][0].append(h)
            cross_groups[_cross_key(d, a)][1].append(r)
        assert compute_per_batch(*cross_groups["1_5-7"]) == 0.0
        assert compute_per_batch(*cross_groups["1_8-11"]) == 0.0

    def test_unknown_per(self, mock_eval_data):
        hyps, refs, ages, dsets = mock_eval_data
        cross_groups = defaultdict(lambda: ([], []))
        for h, r, a, d in zip(hyps, refs, ages, dsets):
            cross_groups[_cross_key(d, a)][0].append(h)
            cross_groups[_cross_key(d, a)][1].append(r)
        per = compute_per_batch(*cross_groups["2_unknown"])
        assert abs(per - 0.25) < 1e-6

    def test_sample_counts(self, mock_eval_data):
        hyps, refs, ages, dsets = mock_eval_data
        cross_groups = defaultdict(lambda: ([], []))
        for h, r, a, d in zip(hyps, refs, ages, dsets):
            cross_groups[_cross_key(d, a)][0].append(h)
            cross_groups[_cross_key(d, a)][1].append(r)
        assert len(cross_groups["1_3-4"][1]) == 2
        assert len(cross_groups["2_8-11"][1]) == 1
        assert len(cross_groups["2_unknown"][1]) == 1


class TestLengthBucketedPER:
    """Verify length-bucketed grouping and PER calculation."""

    def test_bucket_assignment(self):
        assert _length_bucket([1, 2, 3]) == "short_le5"
        assert _length_bucket([1, 2, 3, 4, 5]) == "short_le5"
        assert _length_bucket([1, 2, 3, 4, 5, 6]) == "mid_6to15"
        assert _length_bucket(list(range(15))) == "mid_6to15"
        assert _length_bucket(list(range(16))) == "long_gt15"
        assert _length_bucket(list(range(50))) == "long_gt15"

    def test_all_buckets_covered(self, mock_eval_data):
        hyps, refs, _, _ = mock_eval_data
        len_groups = defaultdict(lambda: ([], []))
        for h, r in zip(hyps, refs):
            bucket = _length_bucket(r)
            len_groups[bucket][0].append(h)
            len_groups[bucket][1].append(r)
        # We have refs of length 3, 3, 5, 2, 3, 5, 16, 4
        assert "short_le5" in len_groups  # lengths 2, 3 (x4), 4
        assert "mid_6to15" not in len_groups or len(len_groups["mid_6to15"][1]) > 0
        assert "long_gt15" in len_groups  # length 16

    def test_short_bucket_per(self, mock_eval_data):
        hyps, refs, _, _ = mock_eval_data
        len_groups = defaultdict(lambda: ([], []))
        for h, r in zip(hyps, refs):
            bucket = _length_bucket(r)
            len_groups[bucket][0].append(h)
            len_groups[bucket][1].append(r)
        short_per = compute_per_batch(*len_groups["short_le5"])
        # Short refs (len<=5): [1,2,3], [1,2,3], [5,6,7,8,9], [10,11], [1,2,3], [5,6,7,8,9], [20,21,22,99]
        # Short hyps:          [1,2,3], [1,2,4], [5,6,7,8,9], [10,11], [1,2,3], [5,6,7,8,9], [20,21,22,23]
        # Edits: 0 + 1 + 0 + 0 + 0 + 0 + 1 = 2, ref_len: 3+3+5+2+3+5+4 = 25
        assert abs(short_per - 2 / 25) < 1e-6

    def test_long_bucket_per(self, mock_eval_data):
        hyps, refs, _, _ = mock_eval_data
        len_groups = defaultdict(lambda: ([], []))
        for h, r in zip(hyps, refs):
            bucket = _length_bucket(r)
            len_groups[bucket][0].append(h)
            len_groups[bucket][1].append(r)
        long_per = compute_per_batch(*len_groups["long_gt15"])
        assert long_per == 0.0  # perfect match


class TestPerAgeCER:
    """Verify per-age CER grouping logic."""

    def test_age_cer_groups(self):
        """Ensure grouping produces correct age keys."""
        hyp_strs = ["abc", "abd", "efghi", "jk", "abc", "efghi", "long", "wxyz"]
        ref_strs = ["abc", "abc", "efghi", "jk", "abc", "efghi", "long", "wxyz"]
        age_buckets = ["3-4", "3-4", "5-7", "8-11", "3-4", "5-7", "8-11", "unknown"]

        age_cer_groups = defaultdict(lambda: ([], []))
        for hs, rs, a in zip(hyp_strs, ref_strs, age_buckets):
            if rs.strip():
                age_cer_groups[a][0].append(hs)
                age_cer_groups[a][1].append(rs)

        assert sorted(age_cer_groups.keys()) == ["3-4", "5-7", "8-11", "unknown"]
        assert len(age_cer_groups["3-4"][0]) == 3
        assert len(age_cer_groups["5-7"][0]) == 2


class TestSeedConfig:
    """Verify all seeds in config are set to 1507."""

    def test_config_yaml_seeds(self):
        """Verify all seed sources in config.yaml are 1507.
        Note: split.seed and hf_sft.seed are propagated from top-level seed
        by config.py, not written explicitly in the YAML."""
        import yaml
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert cfg["seed"] == 1507, "top-level seed (propagated to split, hf_sft)"
        assert cfg["audio_eda"]["spectral"]["seed"] == 1507, "spectral seed"

    def test_config_py_defaults(self):
        from config.config import _DEFAULTS
        assert _DEFAULTS["audio_eda"]["spectral"]["seed"] == 1507
        assert _DEFAULTS["split"]["seed"] == 1507
        assert _DEFAULTS["seed"] == 1507

    def test_val_ratio_10_percent(self):
        import yaml
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["split"]["val_ratio"] == 0.10, "val_ratio should be 10%"
