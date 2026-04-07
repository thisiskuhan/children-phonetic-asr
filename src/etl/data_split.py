"""
Data Splitter — Deterministic, CTC-safe, speaker-disjoint train/val split
=========================================================================

Hard guarantees:
    • Speaker-disjoint: all rows of a child_id → same split
    • CTC phoneme coverage: train inventory == full inventory (retry rarely needed)
    • Age × dataset stratified: val distribution ≈ global distribution
    • Drills excluded from val (and from test)
    • Unknown speakers quarantined to train
    • Deterministic: seeded RNG, retry uses seed + attempt

Upgrades over naive speaker split:
    • Two-pass architecture   — speaker stats in pass 1 (RAM ∝ speakers, ~1 k);
                                pass 2 materialises all SFT row dicts in memory
                                (RAM ∝ rows, ~150 k dicts at current scale) for
                                CTC monitoring + train controls.
    • Duration-targeted val   — val ≈ val_ratio × total_hours, not just
                                val_ratio × n_speakers; balances CTC training cost.
    • Stratify by age×dataset — prevents silent domain skew when DS1/DS2 differ
                                acoustically.
    • Per-speaker age = mode  — avoids annotation noise from multi-session children.
    • Rare-phoneme protection  — speakers whose phonemes are corpus-rare (below a
                                proportional threshold) are pinned to train before
                                any split attempt, eliminating most retries.
    • Val fill rebalancing    — after per-bucket allocation, if val_hours < target,
                                top-up via ascending-duration fill (overshoot-safe).
    • Split SHA-256 fingerprint — sha256(sft_train.jsonl) and sha256(sft_val.jsonl)
                                  logged for paper-grade reproducibility.
    • Duration-accumulator    — in duration-targeted mode each age×dataset bucket
                                fills its proportional hours share (add-only-if-fits),
                                then a global top-up fills the residual; guaranteed
                                overshoot ≈ 0.  Count-mode (test carve) uses
                                floor + largest-remainder allocation.
"""

from __future__ import annotations

import logging
import math
import random as _random_mod  # instance-based RNG via _random_mod.Random(seed)
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils import nearest_rank_pctl, sampler_weights_from_hours, sha256_file, loads, dumps, dumps_line

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  SFT output schema
# ---------------------------------------------------------------------------

# Columns written to sft_train.jsonl / sft_val.jsonl / sft_test.jsonl.
# Everything else (session_id, md5_hash, filesize_bytes, is_drill) is a
# preprocessing artefact — not needed by the training loop.
_SFT_FIELDS = (
    "utterance_id",
    "child_id",
    "audio_path",
    "audio_duration_sec",
    "age_bucket",
    "phonetic_text",
    "n_phonemes",
    "dataset",          # injected from ds_labels — last, so pop is cheap
)


# ---------------------------------------------------------------------------
#  Rare-phoneme threshold (corpus-size-aware)
# ---------------------------------------------------------------------------

# A phoneme is "rare" when its corpus-wide occurrence count falls below:
#     max(_RARE_PHONEME_THRESHOLD_MIN, round(total_tokens × rare_phoneme_fraction))
# The fraction is read from config["split"]["rare_phoneme_fraction"] so it can be
# varied per experiment without touching source code.
_RARE_PHONEME_THRESHOLD_MIN = 10     # floor — always catch truly scarce chars

# ---------------------------------------------------------------------------
#  DECISION: Rare-phoneme JSONL oversampling — REJECTED
# ---------------------------------------------------------------------------
# Considered duplicating training rows containing rare phonemes (ɟ, ç, ɬ, x,
# ʁ, χ, c) at 4× to boost gradient signal.  This would be zero trainer changes
# — just JSONL row duplication — preserving LengthGroupedSampler padding
# efficiency.
#
# Rejected because these are non-English phonemes (voiced palatal stop, palatal
# fricative, lateral fricative, uvular fricative, etc.) from multilingual
# children or transcriber conventions.  The competition test set is English-only,
# so these phonemes won't appear in reference transcripts.  Oversampling them
# would waste model capacity on patterns irrelevant to scoring.
#
# ʒ (156 tokens, English "measure"/"vision") is the only borderline case but
# 156 × 20 epochs ≈ 3,120 exposures is sufficient without duplication.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Speaker:
    """Per-speaker aggregate computed in pass 1 (streaming)."""
    child_id:        str
    age_bucket:      str            # mode of all rows for this speaker
    dataset:         int            # majority dataset (for age×dataset key)
    n_rows:          int   = 0
    duration:        float = 0.0
    phoneme_counter: Counter = field(default_factory=Counter)
    has_drill:       bool  = False
    # ---- accumulated for mode computation ----
    _age_counts:     Counter = field(default_factory=Counter, repr=False)
    _ds_counts:      Counter = field(default_factory=Counter, repr=False)

    def finalise(self) -> None:
        """Set age_bucket and dataset to their mode values."""
        if self._age_counts:
            self.age_bucket = self._age_counts.most_common(1)[0][0]
        if self._ds_counts:
            self.dataset = self._ds_counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _is_unknown_speaker(child_id: str | None) -> bool:
    """True if child_id is null, empty, or the literal string 'unknown'."""
    if child_id is None:
        return True
    s = child_id.strip()
    return s == "" or s.lower() == "unknown"


def _char_inventory(text: str) -> set[str]:
    """Unique non-space characters in a phonetic_text string."""
    return {ch for ch in text if ch != " "}


# ---------------------------------------------------------------------------
#  DataSplitter
# ---------------------------------------------------------------------------

class DataSplitter:
    """Deterministic, CTC-safe, speaker-disjoint train/val/test splitter."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._split_cfg: dict[str, Any] = cfg["split"]
        self._processed = Path(cfg["paths"]["processed"])
        self._datasets:  list[int] = cfg["datasets"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full split pipeline. Returns summary dict."""

        val_ratio:      float = self._split_cfg["val_ratio"]
        test_ratio:     float = self._split_cfg.get("test_ratio", 0.0)
        seed:           int   = self._split_cfg["seed"]
        max_retries:    int   = self._split_cfg["max_retries"]
        min_count_warn: int   = self._split_cfg["min_phoneme_count_warn"]
        rare_fraction:  float = self._split_cfg.get("rare_phoneme_fraction", 3e-5)

        log.info("")
        log.info("--- SPLIT START ---")

        # ================================================================
        # PASS 1 — stream manifests, build speaker aggregates only
        # ================================================================
        (
            speakers,           # child_id → _Speaker
            unknown_uids,       # set of utterance_ids with unknown child_id
            unknown_phonemes,   # phoneme inventory of unknown rows (cached)
            corpus_phonemes,    # Counter: char → total corpus occurrences
            total_rows,
            total_duration,
            ds_labels,          # utterance_id → dataset key (int)
        ) = self._pass1_build_speakers()

        valid_speakers = list(speakers.values())
        n_valid   = len(valid_speakers)
        total_hrs = total_duration / 3600

        log.info(f"[SPLIT] total rows            {total_rows:>10,}")
        log.info(f"[SPLIT] total hours           {total_hrs:>10.2f}")
        log.info(f"[SPLIT] valid speakers        {n_valid:>10,}")
        log.info(f"[SPLIT] unknown-speaker rows  {len(unknown_uids):>10,}  "
                 f"(forced to train)")

        # ================================================================
        # Full corpus phoneme inventory
        # ================================================================
        full_inventory: set[str] = set(corpus_phonemes.keys())
        log.info(f"[SPLIT] corpus inventory      {len(full_inventory):>10} chars")

        # ================================================================
        # Rare-phoneme speaker protection
        # ================================================================
        # Proportional threshold: scales with corpus size so the rule
        # remains meaningful on both small and large datasets.
        total_phoneme_tokens = sum(corpus_phonemes.values())
        rare_threshold = max(
            _RARE_PHONEME_THRESHOLD_MIN,
            round(total_phoneme_tokens * rare_fraction),
        )
        rare_chars: set[str] = {
            ch for ch, cnt in corpus_phonemes.items()
            if cnt < rare_threshold
        }

        # Speakers whose phoneme inventory intersects rare_chars → pinned to train.
        # NOTE: pinning removes these speakers from the stratification pool,
        # causing a minor age×dataset skew toward train.  With n_rare ≈ 10
        # out of ~1 000 speakers the effect is < 1 % — acceptable trade-off
        # for guaranteed CTC coverage without retries.
        rare_train_spk: list[_Speaker] = []
        pool_speakers:  list[_Speaker] = []
        for sp in valid_speakers:
            if sp.phoneme_counter.keys() & rare_chars:
                rare_train_spk.append(sp)
            else:
                pool_speakers.append(sp)

        n_rare_forced = len(rare_train_spk)
        if n_rare_forced:
            log.info(f"[SPLIT] rare-phoneme spkrs → train  "
                     f"{n_rare_forced:>5,}  "
                     f"(threshold < {rare_threshold} / {total_phoneme_tokens:,} tokens)")

        # ================================================================
        # Duration targets
        # ================================================================
        target_val_hrs    = total_hrs * val_ratio
        n_test_target_spk = round(test_ratio * n_valid) if test_ratio > 0.0 else 0

        log.info(f"[SPLIT] target val hours      {target_val_hrs:>10.2f}")
        log.info(f"[SPLIT] test target           {n_test_target_spk:>10} speakers  "
                 f"({test_ratio:.0%} of {n_valid})")

        # ================================================================
        # Carve test speakers first (seed−1 avoids collision with val seeds)
        # ================================================================
        test_spk:   list[_Speaker] = []
        split_pool = pool_speakers   # speakers available for train/val

        if n_test_target_spk > 0:
            remaining_spk, test_spk = self._split_once_duration(
                pool_speakers, n_test_target_spk, seed - 1,
            )
            split_pool = remaining_spk
            log.info(f"[SPLIT] test speakers carved  {len(test_spk):>10,}")

        # ================================================================
        # Retry loop — duration-targeted, age×dataset stratified, rare-safe
        # ================================================================
        train_spk: list[_Speaker] = []
        val_spk:   list[_Speaker] = []
        attempt    = 0

        for attempt in range(max_retries):
            current_seed = seed + attempt
            train_spk_cand, val_spk_cand = self._split_once_duration(
                split_pool, 0, current_seed,
                target_val_hrs=target_val_hrs,
            )

            # CTC coverage: rare-train speakers are always in train side;
            # use cached unknown_phonemes to avoid re-streaming on every retry.
            effective_train_inv = (
                self._speaker_inventory(train_spk_cand + rare_train_spk)
                | unknown_phonemes
            )
            if effective_train_inv >= full_inventory:
                train_spk = train_spk_cand
                val_spk   = val_spk_cand
                break

            missing = full_inventory - effective_train_inv
            log.info(f"[SPLIT] retry {attempt + 1:>2d}  "
                     f"missing {len(missing)} phoneme(s): "
                     f"{' '.join(sorted(missing, key=ord))}")
        else:
            raise RuntimeError(
                f"CTC phoneme coverage FAILED after {max_retries} retries.  "
                f"Missing: {sorted(full_inventory - self._speaker_inventory(train_spk + rare_train_spk), key=ord)}"
            )

        # Merge rare-pinned speakers into final train set
        train_spk = train_spk + rare_train_spk

        retry_count = attempt
        log.info(f"[SPLIT] retries used          {retry_count:>10}")
        log.info(f"[SPLIT] phoneme coverage      {'FULL':>10}")

        # ================================================================
        # Build speaker-ID sets
        # ================================================================
        train_ids = {s.child_id for s in train_spk}
        val_ids   = {s.child_id for s in val_spk}
        test_ids  = {s.child_id for s in test_spk}

        # ================================================================
        # Leakage guard — assert pairwise disjointness
        # ================================================================
        assert train_ids.isdisjoint(val_ids), (
            f"LEAK: {len(train_ids & val_ids)} speaker(s) in both train and val"
        )
        assert train_ids.isdisjoint(test_ids), (
            f"LEAK: {len(train_ids & test_ids)} speaker(s) in both train and test"
        )
        assert val_ids.isdisjoint(test_ids), (
            f"LEAK: {len(val_ids & test_ids)} speaker(s) in both val and test"
        )

        # ================================================================
        # PASS 2 — stream manifests again, route rows to splits
        # ================================================================
        (
            train_rows,
            val_rows,
            test_rows,
            n_drill_excluded_val,
            n_drill_excluded_test,
            train_phoneme_counts,
        ) = self._pass2_route_rows(train_ids, val_ids, test_ids, unknown_uids)

        # ================================================================
        # Min-phoneme-count monitoring (computed inside pass 2 — no extra I/O)
        # ================================================================
        low_phonemes: list[tuple[str, int]] = [
            (ch, train_phoneme_counts.get(ch, 0))
            for ch in sorted(full_inventory, key=ord)
            if train_phoneme_counts.get(ch, 0) < min_count_warn
        ]

        # ================================================================
        # Summary logging
        # ================================================================
        train_hrs = sum(r["audio_duration_sec"] for r in train_rows) / 3600
        val_hrs   = sum(r["audio_duration_sec"] for r in val_rows)   / 3600
        test_hrs  = sum(r["audio_duration_sec"] for r in test_rows)  / 3600
        val_pct   = 100.0 * val_hrs / total_hrs if total_hrs > 0 else 0.0
        test_pct  = 100.0 * test_hrs / total_hrs if total_hrs > 0 else 0.0

        # Duration percentiles per split
        train_durs = sorted(r["audio_duration_sec"] for r in train_rows)
        val_durs   = sorted(r["audio_duration_sec"] for r in val_rows)
        train_p95 = nearest_rank_pctl(train_durs, 95, presorted=True)
        train_p99 = nearest_rank_pctl(train_durs, 99, presorted=True)
        val_p95   = nearest_rank_pctl(val_durs,   95, presorted=True)
        val_p99   = nearest_rank_pctl(val_durs,   99, presorted=True)

        log.info(f"[SPLIT] stratification key    age \u00d7 dataset")
        log.info(f"[SPLIT] train speakers        {len(train_ids):>10,}")
        log.info(f"[SPLIT] val   speakers        {len(val_ids):>10,}")
        if n_test_target_spk > 0:
            log.info(f"[SPLIT] test  speakers        {len(test_ids):>10,}")
        log.info(f"[SPLIT] train rows            {len(train_rows):>10,}")
        log.info(f"[SPLIT] val   rows            {len(val_rows):>10,}")
        if n_test_target_spk > 0:
            log.info(f"[SPLIT] test  rows            {len(test_rows):>10,}")
        log.info(f"[SPLIT] train hours           {train_hrs:>10.2f}")
        log.info(f"[SPLIT] train dur p95/p99     {train_p95:>7.3f}s / {train_p99:.3f}s")
        log.info(f"[SPLIT] target val hours      {target_val_hrs:>10.2f}")
        val_delta = val_hrs - target_val_hrs
        log.info(f"[SPLIT] achieved val hours    {val_hrs:>10.2f}  "
                 f"({val_pct:.1f}%  \u0394 {val_delta:+.2f} h)")
        log.info(f"[SPLIT] val   dur p95/p99     {val_p95:>7.3f}s / {val_p99:.3f}s")
        if n_test_target_spk > 0:
            log.info(f"[SPLIT] test  hours           {test_hrs:>10.2f}  ({test_pct:.1f}%)")
        log.info(f"[SPLIT] drills excl. from val {n_drill_excluded_val:>10,}")
        if n_test_target_spk > 0:
            log.info(f"[SPLIT] drills excl. from tst {n_drill_excluded_test:>10,}")

        self._log_age_dist("train", train_rows, train_hrs)
        self._log_age_dist("val",   val_rows,   val_hrs)
        if n_test_target_spk > 0:
            self._log_age_dist("test",  test_rows,  test_hrs)

        if low_phonemes:
            for ch, cnt in low_phonemes:
                cp = f"U+{ord(ch):04X}"
                log.warning(f"[SPLIT] low train count  {cp}  '{ch}'  "
                             f"count={cnt:>6,}  (< {min_count_warn})")
        else:
            log.info(f"[SPLIT] min phoneme count     all \u2265 {min_count_warn}")

        if train_phoneme_counts:
            min_ch = min(train_phoneme_counts, key=train_phoneme_counts.get)
            log.info(f"[SPLIT] rarest in train       '{min_ch}' "
                     f"count={train_phoneme_counts[min_ch]:,}")

        # ================================================================
        # Write outputs
        # ================================================================
        sft_train_path = self._processed / "sft_train.jsonl"
        sft_val_path   = self._processed / "sft_val.jsonl"
        sft_test_path  = self._processed / "sft_test.jsonl"

        self._write_jsonl(sft_train_path, train_rows, ds_labels)
        self._write_jsonl(sft_val_path,   val_rows,   ds_labels)

        log.info(f"[SPLIT] wrote  {sft_train_path.name:25s}  {len(train_rows):>10,} rows  "
                 f"{train_hrs:>8.2f} hrs")
        log.info(f"[SPLIT] wrote  {sft_val_path.name:25s}  {len(val_rows):>10,} rows  "
                 f"{val_hrs:>8.2f} hrs")

        if n_test_target_spk > 0:
            self._write_jsonl(sft_test_path, test_rows, ds_labels)
            log.info(f"[SPLIT] wrote  {sft_test_path.name:25s}  "
                     f"{len(test_rows):>10,} rows")
        elif sft_test_path.exists():
            sft_test_path.unlink()
            log.info(f"[SPLIT] removed stale        {sft_test_path.name}")

        # ================================================================
        # SHA-256 fingerprints (paper-grade reproducibility)
        # ================================================================
        train_sha = sha256_file(sft_train_path)
        val_sha   = sha256_file(sft_val_path)
        log.info(f"[SPLIT] train_sha256          {train_sha}")
        log.info(f"[SPLIT] val_sha256            {val_sha}")

        # ================================================================
        # data_fingerprint.json — single-file reproducibility manifest
        # ================================================================
        tokenizer_dir = Path(self._cfg["paths"]["tokenizer"])
        vocab_path    = tokenizer_dir / "vocab.json"
        vocab_sha     = sha256_file(vocab_path) if vocab_path.exists() else "N/A"

        tkn_cfg_path = tokenizer_dir / "tokenizer_config.json"
        tkn_cfg_sha  = sha256_file(tkn_cfg_path) if tkn_cfg_path.exists() else "N/A"

        fingerprint: dict[str, Any] = {
            "train_sha256":      train_sha,
            "val_sha256":        val_sha,
            "vocab_sha256":      vocab_sha,
            "tokenizer_config_sha256": tkn_cfg_sha,
            "datasets":          sorted(self._datasets),
            "config": {
                "val_ratio":             val_ratio,
                "test_ratio":            test_ratio,
                "seed":                  seed,
                "max_retries":           max_retries,
                "rare_phoneme_fraction": rare_fraction,
                "rare_threshold_resolved": rare_threshold,
            },
            "split_summary": {
                "n_train_rows":     len(train_rows),
                "n_val_rows":       len(val_rows),
                "n_test_rows":      len(test_rows),
                "n_train_speakers": len(train_ids),
                "n_val_speakers":   len(val_ids),
                "n_test_speakers":  len(test_ids),
                "train_hrs":        round(train_hrs, 4),
                "val_hrs":          round(val_hrs, 4),
                "target_val_hrs":   round(target_val_hrs, 4),
                "retry_count":      retry_count,
                "phoneme_coverage": True,
            },
        }

        reports_dir = Path(self._cfg["paths"]["reports"])
        reports_dir.mkdir(parents=True, exist_ok=True)

        fp_path = reports_dir / "data_fingerprint.json"
        with fp_path.open("w", encoding="utf-8") as f:
            f.write(dumps(fingerprint, indent=2))
        log.info(f"[SPLIT] wrote  {fp_path.name}")

        # ================================================================
        # Training controls (training_controls.json)
        # ================================================================
        # Computed from the in-memory train_rows — no extra I/O pass.
        # The trainer uses these; the corpus-wide eda_controls.json
        # from audio EDA stays as the design-decision record.
        tc = self._compute_train_controls(train_rows, train_phoneme_counts)
        tc_path = reports_dir / "training_controls.json"
        with tc_path.open("w", encoding="utf-8") as f:
            f.write(dumps(tc, indent=2))
        log.info(f"[SPLIT] wrote  {tc_path.name}")

        log.info("--- SPLIT END ---")

        return {
            "n_train_rows":     len(train_rows),
            "n_val_rows":       len(val_rows),
            "n_test_rows":      len(test_rows),
            "n_train_speakers": len(train_ids),
            "n_val_speakers":   len(val_ids),
            "n_test_speakers":  len(test_ids),
            "train_hrs":        round(train_hrs, 4),
            "val_hrs":          round(val_hrs, 4),
            "target_val_hrs":   round(target_val_hrs, 4),
            "retry_count":      retry_count,
            "phoneme_coverage": True,
            "train_sha256":     train_sha,
            "val_sha256":       val_sha,
        }

    # ------------------------------------------------------------------
    # Pass 1 — stream manifests, build speaker aggregates
    # ------------------------------------------------------------------

    def _pass1_build_speakers(self) -> tuple[
        dict[str, _Speaker],    # child_id → _Speaker
        set[str],               # unknown_uids
        set[str],               # unknown_phonemes (cached — avoids re-stream on retry)
        Counter,                # corpus phoneme counts
        int,                    # total rows
        float,                  # total duration (seconds)
        dict[str, int],         # utterance_id → dataset key
    ]:
        """
        Single streaming pass over all JSONL manifests.

        Builds per-speaker aggregates (age mode, dataset mode, duration,
        phoneme counts) without keeping all rows in RAM.
        Unknown-speaker phonemes are collected here so the retry loop
        never needs to re-stream the corpus to check CTC coverage.
        """
        speakers:         dict[str, _Speaker] = {}
        unknown_uids:     set[str]            = set()
        unknown_phonemes: set[str]            = set()
        corpus_phonemes:  Counter             = Counter()
        ds_labels:        dict[str, int]      = {}
        total_rows    = 0
        total_duration = 0.0

        for key in sorted(self._datasets):
            path = self._processed / f"{key}_transcript.jsonl"
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    r = loads(line)
                    uid = r["utterance_id"]
                    ds_labels[uid] = key
                    total_rows    += 1
                    total_duration += r["audio_duration_sec"]

                    for ch in r["phonetic_text"]:
                        if ch != " ":
                            corpus_phonemes[ch] += 1

                    cid: str | None = r.get("child_id")
                    if _is_unknown_speaker(cid):
                        unknown_uids.add(uid)
                        unknown_phonemes.update(_char_inventory(r["phonetic_text"]))
                        continue

                    assert cid is not None  # type narrowing — filtered above

                    if cid not in speakers:
                        speakers[cid] = _Speaker(
                            child_id=cid,
                            age_bucket=r.get("age_bucket", "unknown"),
                            dataset=key,
                        )
                    sp = speakers[cid]
                    sp.n_rows    += 1
                    sp.duration  += r["audio_duration_sec"]
                    sp._age_counts[r.get("age_bucket", "unknown")] += 1
                    sp._ds_counts[key] += 1
                    for ch in r["phonetic_text"]:
                        if ch != " ":
                            sp.phoneme_counter[ch] += 1
                    if r.get("is_drill", False):
                        sp.has_drill = True

        # Finalise age_bucket and dataset to their per-speaker mode
        for sp in speakers.values():
            sp.finalise()

        return (
            speakers, unknown_uids, unknown_phonemes, corpus_phonemes,
            total_rows, total_duration, ds_labels,
        )

    # ------------------------------------------------------------------
    # Pass 2 — stream manifests again to route rows
    # ------------------------------------------------------------------

    def _pass2_route_rows(
        self,
        train_ids:    set[str],
        val_ids:      set[str],
        test_ids:     set[str],
        unknown_uids: set[str],
    ) -> tuple[
        list[dict],   # train_rows
        list[dict],   # val_rows
        list[dict],   # test_rows
        int,          # n_drill_excluded_val
        int,          # n_drill_excluded_test
        Counter,      # train_phoneme_counts (computed in same pass)
    ]:
        """
        Second streaming pass: route each row to its split bucket.

        Also accumulates train phoneme counts — eliminates a separate
        third pass that previously re-read all manifests.

        Drills from val/test speakers are redirected to train.
        Unknown-speaker rows go to train unconditionally.
        """
        train_rows: list[dict] = []
        val_rows:   list[dict] = []
        test_rows:  list[dict] = []
        n_drill_excl_val  = 0
        n_drill_excl_test = 0
        train_phoneme_counts: Counter = Counter()

        for key in sorted(self._datasets):
            path = self._processed / f"{key}_transcript.jsonl"
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    r = loads(line)
                    cid = r.get("child_id")
                    uid = r["utterance_id"]
                    is_train = False

                    if uid in unknown_uids or _is_unknown_speaker(cid):
                        train_rows.append(r)
                        is_train = True
                    elif cid in test_ids:
                        if r.get("is_drill", False):
                            n_drill_excl_test += 1
                            # DROP drill — do NOT redirect to train (speaker leak)
                        else:
                            test_rows.append(r)
                    elif cid in val_ids:
                        if r.get("is_drill", False):
                            n_drill_excl_val += 1
                            # DROP drill — do NOT redirect to train (speaker leak)
                        else:
                            val_rows.append(r)
                    else:
                        train_rows.append(r)
                        is_train = True

                    if is_train:
                        for ch in r["phonetic_text"]:
                            if ch != " ":
                                train_phoneme_counts[ch] += 1

        return (
            train_rows, val_rows, test_rows,
            n_drill_excl_val, n_drill_excl_test,
            train_phoneme_counts,
        )

    # ------------------------------------------------------------------
    # Duration-targeted, age×dataset stratified split
    # ------------------------------------------------------------------

    def _split_once_duration(
        self,
        speakers:         list[_Speaker],
        n_val_target_spk: int,
        seed:             int,
        *,
        target_val_hrs:   float = 0.0,
    ) -> tuple[list[_Speaker], list[_Speaker]]:
        """
        One attempt at an age×dataset-stratified speaker split.

        Duration-targeted mode (target_val_hrs > 0):
            Each (age_bucket, dataset) bucket fills its proportional share of
            target_val_hrs using a greedy duration-accumulator over shuffled
            speakers (add only if fits ⇒ per-bucket overshoot = 0).  A global
            ascending-duration top-up then fills any remaining deficit.

        Count-targeted mode (target_val_hrs == 0):
            Allocates exactly n_val_target_spk speakers using floor +
            largest-remainder across buckets.  Used for the test carve.
        """
        rng = _random_mod.Random(seed)

        # ---- group by (age_bucket, dataset) ----
        buckets: dict[tuple[str, int], list[_Speaker]] = {}
        for sp in speakers:
            bk = (sp.age_bucket, sp.dataset)
            buckets.setdefault(bk, []).append(sp)

        total_spk  = len(speakers)
        sorted_keys = sorted(buckets.keys())

        # ---- total pool hours (proportional bucket targets in duration mode) ----
        total_pool_hrs = (
            sum(sp.duration for sp in speakers) / 3600
            if total_spk > 0 else 0.0
        )

        # ---- count-mode: floor + largest-remainder (test carve only) ----
        alloc: dict[tuple[str, int], int] = {}
        if target_val_hrs <= 0.0:
            fractions: list[tuple[float, tuple[str, int]]] = []
            for bk in sorted_keys:
                n_bk  = len(buckets[bk])
                ideal = n_val_target_spk * n_bk / total_spk if total_spk > 0 else 0.0
                fl    = math.floor(ideal)
                fl    = max(0, min(fl, n_bk - 1))   # keep ≥1 per bucket for train
                alloc[bk] = fl
                fractions.append((ideal - math.floor(ideal), bk))
            allocated = sum(alloc.values())
            remainder = n_val_target_spk - allocated
            fractions.sort(key=lambda x: -x[0])
            for i in range(max(0, remainder)):
                bk = fractions[i % len(fractions)][1]
                if alloc[bk] < len(buckets[bk]) - 1:
                    alloc[bk] += 1

        # ---- sample ----
        val_spk:   list[_Speaker] = []
        train_spk: list[_Speaker] = []
        for bk in sorted_keys:
            bk_list = list(buckets[bk])
            rng.shuffle(bk_list)
            if target_val_hrs > 0.0 and total_pool_hrs > 0.0:
                # Duration-accumulator mode: fill this bucket's proportional
                # share of the hours target.  Only add a speaker when their
                # duration fits ⇒ per-bucket overshoot is always zero.
                # Track selected speakers explicitly; a counter + slice is
                # wrong because non-contiguous speakers may be selected.
                bk_total_hrs  = sum(sp.duration for sp in bk_list) / 3600
                bk_target_hrs = target_val_hrs * (bk_total_hrs / total_pool_hrs)
                bk_val_hrs    = 0.0
                val_sel:   list[_Speaker] = []
                train_sel: list[_Speaker] = []
                for j, sp in enumerate(bk_list):
                    is_last = (j == len(bk_list) - 1)
                    sp_hrs  = sp.duration / 3600
                    if (not is_last) and (bk_val_hrs + sp_hrs <= bk_target_hrs):
                        bk_val_hrs += sp_hrs
                        val_sel.append(sp)
                    else:
                        train_sel.append(sp)
                val_spk.extend(val_sel)
                train_spk.extend(train_sel)
            else:
                n_val = max(0, min(alloc.get(bk, 0), len(bk_list) - 1))
                val_spk.extend(bk_list[:n_val])
                train_spk.extend(bk_list[n_val:])

        # ---- duration top-up (val fill rebalancing — overshoot-safe) ----
        # Sort train speakers by duration ascending so we fill the deficit
        # tightly: a speaker is moved to val only if their full duration fits
        # within the remaining deficit.  Worst-case overshoot is bounded to
        # < 1 speaker duration (effectively zero for reasonable speaker sizes).
        if target_val_hrs > 0.0:
            achieved_hrs = sum(sp.duration for sp in val_spk) / 3600
            deficit_hrs  = target_val_hrs - achieved_hrs
            if deficit_hrs > 0:
                train_spk.sort(key=lambda sp: sp.duration)   # ascending
                moved_idx: list[int] = []
                for i, sp in enumerate(train_spk):
                    if deficit_hrs <= 0:
                        break
                    sp_hrs = sp.duration / 3600
                    if sp_hrs <= deficit_hrs:   # only move if it fits
                        val_spk.append(sp)
                        moved_idx.append(i)
                        deficit_hrs -= sp_hrs
                for i in reversed(moved_idx):
                    train_spk.pop(i)

        return train_spk, val_spk

    # ------------------------------------------------------------------
    # Phoneme inventory helpers
    # ------------------------------------------------------------------

    def _speaker_inventory(self, speakers: list[_Speaker]) -> set[str]:
        """Combined phoneme inventory of a speaker list."""
        inv: set[str] = set()
        for sp in speakers:
            inv.update(sp.phoneme_counter.keys())
        return inv

    # ------------------------------------------------------------------
    # Output writing
    # ------------------------------------------------------------------

    def _write_jsonl(
        self,
        path:      Path,
        rows:      list[dict],
        ds_labels: dict[str, int],
    ) -> None:
        """Write SFT JSONL — only _SFT_FIELDS columns, with injected 'dataset'."""
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                ds  = ds_labels.get(r["utterance_id"], 0)
                out = {k: r[k] for k in _SFT_FIELDS[:-1]}  # all except 'dataset'
                out["dataset"] = ds
                f.write(dumps_line(out))
        tmp.rename(path)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_age_dist(
        self,
        label:     str,
        rows:      list[dict],
        total_hrs: float,
    ) -> None:
        """Print age-bucket hour breakdown for a split."""
        age_hours: dict[str, float] = {}
        for r in rows:
            bk = r.get("age_bucket", "unknown")
            age_hours[bk] = age_hours.get(bk, 0.0) + r["audio_duration_sec"] / 3600
        if not age_hours:
            log.info(f"[SPLIT] {label:5s} age dist  (empty)")
            return
        for bk in sorted(age_hours):
            hrs = age_hours[bk]
            pct = 100.0 * hrs / total_hrs if total_hrs > 0 else 0.0
            log.info(f"[SPLIT] {label:5s}  {bk:<8s}  "
                     f"{hrs:>8.2f} hrs  ({pct:>5.1f}%)")

    # ------------------------------------------------------------------
    # Train-only controls
    # ------------------------------------------------------------------

    def _compute_train_controls(
        self,
        train_rows:          list[dict],
        train_phoneme_counts: Counter,
    ) -> dict[str, Any]:
        """Derive train-only statistics that the trainer needs.

        Computed entirely from the in-memory ``train_rows`` produced by
        pass-2 — no extra file I/O.  These stats replace the corpus-wide
        values in ``eda_controls.json`` for anything that depends on
        data distribution (RMS, durations, sampler weights, phoneme freq).

        Design-time decisions (tokenizer, sample rate, VAD, clipping) are
        NOT repeated here — those remain in ``eda_controls.json``.
        """
        n = len(train_rows)
        if n == 0:
            return {"scope": "train_only", "n_rows": 0}

        # ---- Duration stats ----
        durations = sorted(r["audio_duration_sec"] for r in train_rows)
        total_hrs = sum(durations) / 3600

        eda_cfg  = self._cfg.get("audio_eda", {})
        vram_sec = eda_cfg.get("duration", {}).get("target_vram_seconds", 120.0)
        dur_p95  = nearest_rank_pctl(durations, 95, presorted=True)
        batch_at_p95 = max(1, int(vram_sec / dur_p95)) if dur_p95 > 0 else 1

        dur_p99  = nearest_rank_pctl(durations, 99, presorted=True)
        n_above_p99 = sum(1 for d in durations if d > dur_p99)

        duration_block = {
            "p50": nearest_rank_pctl(durations, 50, presorted=True),
            "p90": nearest_rank_pctl(durations, 90, presorted=True),
            "p95": dur_p95,
            "p96": nearest_rank_pctl(durations, 96, presorted=True),
            "p97": nearest_rank_pctl(durations, 97, presorted=True),
            "p98": nearest_rank_pctl(durations, 98, presorted=True),
            "p99": dur_p99,
            "max": round(durations[-1], 3),
            "n_rows_above_p99": n_above_p99,
            "recommended_max_duration_sec": dur_p99,
            "target_vram_seconds": vram_sec,
            "theoretical_batch_size_at_p95": batch_at_p95,
            "batch_size_note": (
                "Upper bound — ignores feature-extraction latency, "
                "padding overhead, and gradient memory.  "
                "Validate empirically with a profiling run."
            ),
            "collator_policy_note": (
                "Set collator max_length from recommended_max_duration_sec.  "
                "Rows above this threshold can be truncated or filtered "
                "at collator time — not a split concern."
            ),
        }

        # ---- Age distribution + sampler weights ----
        age_hours: dict[str, float] = {}
        for r in train_rows:
            bk = r.get("age_bucket", "unknown")
            age_hours[bk] = age_hours.get(bk, 0.0) + r["audio_duration_sec"] / 3600

        age_total = sum(age_hours.values())
        weights   = sampler_weights_from_hours(age_hours)

        age_block = {
            "hours_per_age_bucket": {
                k: round(v, 3) for k, v in sorted(age_hours.items())
            },
            "percent_per_age_bucket": {
                k: round(100 * v / age_total, 2)
                for k, v in sorted(age_hours.items())
            } if age_total > 0 else {},
            "sampler_weights_by_age_bucket": weights,
        }

        # ---- Phoneme frequency (already computed during CTC monitoring) ----
        total_tokens = sum(train_phoneme_counts.values())
        phoneme_freq: dict[str, dict[str, Any]] = {}
        for ch in sorted(train_phoneme_counts, key=lambda c: -train_phoneme_counts[c]):
            cnt = train_phoneme_counts[ch]
            phoneme_freq[ch] = {
                "count": cnt,
                "percent": round(100 * cnt / total_tokens, 4) if total_tokens > 0 else 0.0,
            }

        phoneme_block = {
            "total_tokens": total_tokens,
            "inventory_size": len(train_phoneme_counts),
            "frequency": phoneme_freq,
        }

        # ---- Speaker hour stats ----
        speaker_hours: dict[str, float] = {}
        for r in train_rows:
            cid = r.get("child_id", "unknown")
            speaker_hours[cid] = speaker_hours.get(cid, 0.0) + r["audio_duration_sec"] / 3600

        spk_hrs = sorted(speaker_hours.values())
        spk_p50 = nearest_rank_pctl(spk_hrs, 50, presorted=True)
        spk_p95 = nearest_rank_pctl(spk_hrs, 95, presorted=True)
        spk_p99 = nearest_rank_pctl(spk_hrs, 99, presorted=True)
        spk_max = round(spk_hrs[-1], 3) if spk_hrs else 0.0
        skew_ratio = round(spk_max / spk_p50, 2) if spk_p50 > 0 else 0.0

        speaker_block = {
            "n_speakers": len(speaker_hours),
            "total_hours": round(total_hrs, 3),
            "speaker_hours_p50": spk_p50,
            "speaker_hours_p95": spk_p95,
            "speaker_hours_p99": spk_p99,
            "speaker_hours_max": spk_max,
            "speaker_hours_skew_ratio": skew_ratio,
            "recommend_speaker_balanced_sampling": skew_ratio > 10.0,
            "speaker_epoch_cap_hours": spk_p95,
            "speaker_epoch_cap_note": (
                "If recommend_speaker_balanced_sampling is true, limit each "
                "speaker to speaker_epoch_cap_hours per epoch.  Excess rows "
                "are deferred to subsequent epochs."
            ),
        }

        # ---- Log summary ----
        log.info(f"[SPLIT] train controls  hours={total_hrs:.2f}  "
                 f"dur_p95={dur_p95:.3f}s  batch@p95={batch_at_p95}  "
                 f"phoneme_tokens={total_tokens:,}")
        for bk in sorted(age_hours):
            hrs = age_hours[bk]
            pct = 100 * hrs / age_total if age_total > 0 else 0
            wt  = weights.get(bk, 0)
            log.info(f"[SPLIT] tc  {bk:<8s}  {hrs:>8.2f} h  "
                     f"({pct:>5.1f}%)  weight={wt:.4f}")

        return {
            "scope": "train_only",
            "source": "sft_train.jsonl",
            "n_rows": n,
            "total_hours": round(total_hrs, 3),
            "duration_profile": duration_block,
            "age_distribution": age_block,
            "phoneme_frequency": phoneme_block,
            "speaker_stats": speaker_block,
        }
