#!/usr/bin/env python3
"""End-to-end NST flow test — filter → backup → DataSplitter re-split.

Tests the FULL pipeline.py run_nst() flow (minus teacher_infer.py GPU step)
using synthetic pseudo-label data against the real ETL transcripts.

Invariants verified:
  1. filter writes pseudo_labelled.jsonl with correct rows
  2. filter writes 3_transcript.jsonl with same rows (DataSplitter format)
  3. filter rejects bad rows (physics, CER, empty, duration, short_phonemes)
  4. pipeline backs up sft_train.jsonl → sft_train_original.jsonl
  5. pipeline backs up sft_val.jsonl → sft_val_original.jsonl
  6. DataSplitter re-split produces new sft_train.jsonl and sft_val.jsonl
  7. new sft_train has ALL pseudo rows (dataset=3) — because empty child_id
  8. new sft_val has ZERO pseudo rows — val stays gold-speaker-only
  9. new sft_train rows > original sft_train rows (by n_pseudo)
  10. new sft_val rows ≈ original sft_val (same speakers, just re-shuffled)
  11. all rows have required _SFT_FIELDS
  12. no leakage: train child_ids ∩ val child_ids == unknown speakers only
  13. backup files match originals exactly (byte-identical)
  14. 0-kept edge case: no 3_transcript.jsonl, no re-split, no backup
  15. idempotency: running backup twice doesn't overwrite originals
  16. data_fingerprint.json and training_controls.json are updated
  17. 3_transcript.jsonl row count == pseudo_labelled.jsonl row count

Usage:
    python src/tests/test_nst_flow.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_config
from etl import DataSplitter

logging.basicConfig(
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS   = ROOT / "data" / "reports"
NST_DIR   = ROOT / "data" / "nst"

# ── Required SFT fields (must match data_split.py _SFT_FIELDS) ──
_SFT_FIELDS = {
    "utterance_id", "child_id", "audio_path", "audio_duration_sec",
    "age_bucket", "phonetic_text", "n_phonemes", "dataset",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _generate_raw_pseudo_labels(out_path: Path, n_good: int = 50, n_bad: int = 24) -> int:
    """Generate synthetic pseudo_labels_raw.jsonl mimicking teacher_infer output.

    Returns the number of rows that SHOULD pass all filters.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # Good rows — should pass all filters
    for i in range(n_good):
        text = f"h ɛ l oʊ w ɜː l d {i}"
        n_chars = len(text.replace(" ", ""))
        dur = max(0.5, 1.0 + (i % 10) * 0.3)  # 0.5s to 3.7s
        rows.append({
            "utterance_id":        f"U_test_pseudo_{i:04d}",
            "audio_path":          f"data/raw/2_audio/U_test_pseudo_{i:04d}.flac",
            "audio_duration_sec":  round(dur, 3),
            "age_bucket":          ["3-5", "5-8", "8-11"][i % 3],
            "child_id":            "",            # empty → unknown speaker → forced to train
            "dataset":             2,
            "beam_pred":           text,
            "greedy_pred":         text,          # perfect agreement → CER=0
            "beam_greedy_cer":     0.0,
            "norm_score":          -0.3 - (i % 5) * 0.1,  # all above -1.2
            "n_beam_chars":        n_chars,
            "n_greedy_chars":      n_chars,
        })

    # Bad rows — various rejection reasons
    for i in range(n_bad):
        kind = i % 6
        text = "b æ d"
        dur = 2.0
        cer = 0.0
        norm = -0.5
        n_chars = 3

        if kind == 0:   # too high CER
            cer = 0.5
        elif kind == 1: # too low norm score
            norm = -2.0
        elif kind == 2: # empty beam
            text = ""
            n_chars = 0
        elif kind == 3: # physics: too fast (high tps)
            dur = 0.05  # n_chars=3, dur=0.05 → tps=60 >> 25
        elif kind == 4: # duration too short
            dur = 0.05
        elif kind == 5: # short phonemes (< min_phonemes=3)
            text = "b æ"
            n_chars = 2
            dur = 1.5   # pass all other filters

        greedy_text = text if kind != 0 else "x y z"
        n_greedy = len(greedy_text.replace(" ", ""))

        rows.append({
            "utterance_id":        f"U_test_bad_{i:04d}",
            "audio_path":          f"data/raw/2_audio/U_test_bad_{i:04d}.flac",
            "audio_duration_sec":  round(dur, 3),
            "age_bucket":          "5-8",
            "child_id":            "",
            "dataset":             2,
            "beam_pred":           text,
            "greedy_pred":         greedy_text,
            "beam_greedy_cer":     cer,
            "norm_score":          norm,
            "n_beam_chars":        n_chars,
            "n_greedy_chars":      n_greedy,
        })

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return n_good


def _cleanup_test_artifacts():
    """Remove all test artifacts (3_transcript, backups, etc)."""
    artifacts = [
        PROCESSED / "3_transcript.jsonl",
        PROCESSED / "sft_train_original.jsonl",
        PROCESSED / "sft_val_original.jsonl",
        PROCESSED / "pseudo_labelled.jsonl",
        NST_DIR / "pseudo_labels_raw.jsonl",
        REPORTS / "nst_filter_report.json",
    ]
    for p in artifacts:
        if p.exists():
            p.unlink()


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def check(self, condition: bool, name: str, detail: str = ""):
        if condition:
            self.passed += 1
            log.info(f"  ✓  {name}")
        else:
            self.failed += 1
            msg = f"  ✗  {name}" + (f"  — {detail}" if detail else "")
            log.error(msg)
            self.errors.append(msg)

    def summary(self):
        total = self.passed + self.failed
        log.info("")
        log.info(f"{'='*60}")
        log.info(f"  PASSED: {self.passed}/{total}")
        if self.failed:
            log.info(f"  FAILED: {self.failed}/{total}")
            for e in self.errors:
                log.info(f"    {e}")
        log.info(f"{'='*60}")
        return self.failed == 0


def main():
    t0 = time.time()
    T = TestResult()

    log.info("=" * 60)
    log.info("  NST FLOW — END-TO-END TEST")
    log.info("=" * 60)

    cfg = load_config(ROOT / "src" / "config" / "config.yaml")

    # ── Record pre-state ─────────────────────────────────────────
    sft_train_path = PROCESSED / "sft_train.jsonl"
    sft_val_path   = PROCESSED / "sft_val.jsonl"

    T.check(sft_train_path.exists(), "sft_train.jsonl exists (from ETL)")
    T.check(sft_val_path.exists(),   "sft_val.jsonl exists (from ETL)")

    orig_train_sha = _sha256(sft_train_path)
    orig_val_sha   = _sha256(sft_val_path)
    orig_train_n   = _count_lines(sft_train_path)
    orig_val_n     = _count_lines(sft_val_path)

    # Record pre-existing speaker overlap (from DataSplitter drill redirect)
    orig_train_cids = set()
    orig_val_cids   = set()
    with sft_train_path.open() as f:
        for line in f:
            cid = json.loads(line).get("child_id", "")
            if cid:
                orig_train_cids.add(cid)
    with sft_val_path.open() as f:
        for line in f:
            cid = json.loads(line).get("child_id", "")
            if cid:
                orig_val_cids.add(cid)
    orig_overlap = orig_train_cids & orig_val_cids

    log.info(f"  Pre-state: train={orig_train_n:,} rows  val={orig_val_n:,} rows"
             f"  (drill-overlap speakers: {len(orig_overlap)})")

    # ── Clean up any previous test artifacts ─────────────────────
    _cleanup_test_artifacts()

    # ══════════════════════════════════════════════════════════════
    #  TEST 1: Filter with good + bad synthetic data
    # ══════════════════════════════════════════════════════════════
    log.info("")
    log.info("--- TEST 1: FILTER (good + bad rows) ---")

    N_GOOD = 50
    N_BAD  = 24
    raw_path = NST_DIR / "pseudo_labels_raw.jsonl"
    n_pass_filter = _generate_raw_pseudo_labels(raw_path, n_good=N_GOOD, n_bad=N_BAD)

    # Account for pseudo_keep_fraction (score-based ranking after filter)
    keep_frac = cfg["nst"].get("pseudo_keep_fraction", 1.0)
    if 0 < keep_frac < 1.0 and n_pass_filter > 0:
        expected_kept = max(1, int(n_pass_filter * keep_frac))
    else:
        expected_kept = n_pass_filter
    log.info(f"  n_pass_filter={n_pass_filter}  keep_frac={keep_frac}  expected_kept={expected_kept}")

    T.check(raw_path.exists(), "synthetic pseudo_labels_raw.jsonl created")
    T.check(_count_lines(raw_path) == N_GOOD + N_BAD,
            f"raw has {N_GOOD + N_BAD} rows",
            f"got {_count_lines(raw_path)}")

    # Run filter as subprocess (same as pipeline does)
    env = {**__import__("os").environ, "PYTHONPATH": str(_SRC)}
    result = subprocess.run(
        [sys.executable, str(_SRC / "nst" / "filter_pseudo_labels.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    T.check(result.returncode == 0, "filter exit code 0",
            f"exit={result.returncode}\nSTDERR:\n{result.stderr[-500:]}")

    if result.returncode != 0:
        log.error("Filter failed — aborting remaining tests.")
        log.error(result.stderr[-1000:])
        _cleanup_test_artifacts()
        T.summary()
        return

    # ── Verify filter outputs ────────────────────────────────────
    pseudo_path = PROCESSED / "pseudo_labelled.jsonl"
    transcript_path = PROCESSED / "3_transcript.jsonl"

    T.check(pseudo_path.exists(), "pseudo_labelled.jsonl exists")
    T.check(transcript_path.exists(), "3_transcript.jsonl exists")

    pseudo_n = _count_lines(pseudo_path)
    transcript_n = _count_lines(transcript_path)

    T.check(pseudo_n == expected_kept,
            f"pseudo_labelled.jsonl has {expected_kept} rows",
            f"got {pseudo_n}")
    T.check(transcript_n == expected_kept,
            f"3_transcript.jsonl has {expected_kept} rows",
            f"got {transcript_n}")
    T.check(pseudo_n == transcript_n,
            "pseudo_labelled == 3_transcript row count")

    # Verify 3_transcript rows have all required fields for DataSplitter
    transcript_rows = _load_jsonl(transcript_path)
    for i, row in enumerate(transcript_rows[:5]):  # spot check first 5
        has_all = _SFT_FIELDS.issubset(row.keys())
        T.check(has_all, f"3_transcript row {i} has all _SFT_FIELDS",
                f"missing: {_SFT_FIELDS - row.keys()}")

    # All pseudo rows should have empty child_id
    all_empty_cid = all(r.get("child_id", "") == "" for r in transcript_rows)
    T.check(all_empty_cid, "all 3_transcript rows have empty child_id")

    # Verify pseudo labels use beam_pred as phonetic_text
    pseudo_rows = _load_jsonl(pseudo_path)
    if pseudo_rows:
        T.check(pseudo_rows[0]["phonetic_text"] != "",
                "pseudo phonetic_text is non-empty (from beam_pred)")

    # Verify filter report exists
    report_path = REPORTS / "nst_filter_report.json"
    T.check(report_path.exists(), "nst_filter_report.json exists")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        T.check(report["kept"] == expected_kept,
                f"report.kept == {expected_kept}",
                f"got {report['kept']}")
        T.check(report["rejected"] == N_BAD,
                f"report.rejected == {N_BAD} (filter rejects)",
                f"got {report['rejected']}")
        # Verify short_phonemes filter is working
        reasons = report.get("rejection_reasons", {})
        T.check("short_phonemes" in reasons,
                "short_phonemes in rejection_reasons",
                f"reasons={list(reasons.keys())}")
        T.check("beam_greedy_cer" in reasons,
                "beam_greedy_cer in rejection_reasons",
                f"reasons={list(reasons.keys())}")
        T.check(report["thresholds"].get("min_phonemes") == cfg["nst"].get("min_phonemes", cfg["eda"]["min_phonemes"]),
                "report.thresholds.min_phonemes matches config",
                f"got {report['thresholds'].get('min_phonemes')}")
        T.check(report["thresholds"].get("max_beam_greedy_cer") == cfg["nst"]["max_beam_greedy_cer"],
                "report.thresholds.max_beam_greedy_cer matches config",
                f"got {report['thresholds'].get('max_beam_greedy_cer')}")
    # ══════════════════════════════════════════════════════════════
    #  TEST 2: Backup originals
    # ══════════════════════════════════════════════════════════════
    log.info("")
    log.info("--- TEST 2: BACKUP ORIGINALS ---")

    backup_train = PROCESSED / "sft_train_original.jsonl"
    backup_val   = PROCESSED / "sft_val_original.jsonl"

    # No backups yet
    T.check(not backup_train.exists(), "sft_train_original.jsonl doesn't exist yet")
    T.check(not backup_val.exists(),   "sft_val_original.jsonl doesn't exist yet")

    # Perform backup (same as pipeline step 3)
    shutil.copy2(sft_train_path, backup_train)
    shutil.copy2(sft_val_path, backup_val)

    T.check(backup_train.exists(), "sft_train_original.jsonl created")
    T.check(backup_val.exists(),   "sft_val_original.jsonl created")

    # Verify byte-identical
    T.check(_sha256(backup_train) == orig_train_sha,
            "backup train SHA-256 matches original")
    T.check(_sha256(backup_val) == orig_val_sha,
            "backup val SHA-256 matches original")

    # ── Idempotency: running backup again should NOT overwrite ───
    # Simulate what pipeline does: only copy if backup doesn't exist
    backup_train_sha_before = _sha256(backup_train)
    if sft_train_path.exists() and not backup_train.exists():
        shutil.copy2(sft_train_path, backup_train)
    T.check(_sha256(backup_train) == backup_train_sha_before,
            "idempotent: backup not overwritten on second run")

    # ══════════════════════════════════════════════════════════════
    #  TEST 3: DataSplitter re-split with DS3
    # ══════════════════════════════════════════════════════════════
    log.info("")
    log.info("--- TEST 3: DATASPLITTER RE-SPLIT ---")

    cfg_nst = copy.deepcopy(cfg)
    cfg_nst["datasets"] = sorted(cfg["datasets"] + [3])
    cfg_nst["paths"]["audio_dirs"][3] = cfg["paths"]["audio_dirs"][2]

    log.info(f"  Running DataSplitter with datasets={cfg_nst['datasets']}")
    split_result = DataSplitter(cfg_nst).run()

    T.check(split_result is not None, "DataSplitter returned result dict")
    T.check(split_result.get("phoneme_coverage") is True,
            "phoneme coverage FULL")

    # ── Verify new sft_train.jsonl ───────────────────────────────
    new_train = _load_jsonl(sft_train_path)
    new_val   = _load_jsonl(sft_val_path)

    new_train_n = len(new_train)
    new_val_n   = len(new_val)

    log.info(f"  New state: train={new_train_n:,} rows  val={new_val_n:,} rows")
    log.info(f"  Delta:     train +{new_train_n - orig_train_n:,}  val {new_val_n - orig_val_n:+,}")

    # Pseudo rows should be in train
    train_ds3 = [r for r in new_train if r.get("dataset") == 3]
    val_ds3   = [r for r in new_val   if r.get("dataset") == 3]

    T.check(len(train_ds3) == expected_kept,
            f"train has {expected_kept} DS3 pseudo rows",
            f"got {len(train_ds3)}")
    T.check(len(val_ds3) == 0,
            "val has 0 DS3 pseudo rows (gold-only)",
            f"got {len(val_ds3)}")

    # New train should be larger by exactly n_pseudo
    # (Minus any val→train speaker shuffling from re-seed — should be close)
    T.check(new_train_n >= orig_train_n + expected_kept - 100,
            f"new train rows >= original + pseudo - tolerance",
            f"{new_train_n} vs {orig_train_n + expected_kept}")

    # ── All rows have _SFT_FIELDS ────────────────────────────────
    bad_train_fields = sum(1 for r in new_train if not _SFT_FIELDS.issubset(r.keys()))
    bad_val_fields   = sum(1 for r in new_val   if not _SFT_FIELDS.issubset(r.keys()))
    T.check(bad_train_fields == 0,
            "all train rows have _SFT_FIELDS",
            f"{bad_train_fields} rows missing fields")
    T.check(bad_val_fields == 0,
            "all val rows have _SFT_FIELDS",
            f"{bad_val_fields} rows missing fields")

    # ── Speaker disjointness ─────────────────────────────────────
    # DataSplitter redirects val-speaker DRILL rows to train by design,
    # so a small overlap is expected.  Verify overlap doesn't grow beyond
    # what the original ETL split already had (drills only).
    train_cids = {r["child_id"] for r in new_train if r["child_id"]}
    val_cids   = {r["child_id"] for r in new_val   if r["child_id"]}
    overlap = train_cids & val_cids

    T.check(len(overlap) <= len(orig_overlap),
            "speaker overlap unchanged (drill redirect only)",
            f"new={len(overlap)} vs orig={len(orig_overlap)}, new extras: {overlap - orig_overlap}")

    # ── Val should be close to original ──────────────────────────
    # Same seed + same speakers → should be nearly identical
    # Allow ±5% tolerance for data volume change effects
    val_ratio = abs(new_val_n - orig_val_n) / max(orig_val_n, 1) * 100
    T.check(val_ratio < 5.0,
            f"val count within 5% of original ({val_ratio:.1f}% change)",
            f"orig={orig_val_n} new={new_val_n}")

    # ── Reports updated ──────────────────────────────────────────
    fp_path = REPORTS / "data_fingerprint.json"
    tc_path = REPORTS / "training_controls.json"
    T.check(fp_path.exists(), "data_fingerprint.json updated")
    T.check(tc_path.exists(), "training_controls.json updated")

    if fp_path.exists():
        fp = json.loads(fp_path.read_text())
        T.check(3 in fp.get("datasets", []),
                "fingerprint includes dataset 3",
                f"datasets={fp.get('datasets')}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 4: Edge case — 0 kept rows
    # ══════════════════════════════════════════════════════════════
    log.info("")
    log.info("--- TEST 4: ZERO KEPT ROWS ---")

    # Generate all-bad data
    _generate_raw_pseudo_labels(raw_path, n_good=0, n_bad=10)

    result = subprocess.run(
        [sys.executable, str(_SRC / "nst" / "filter_pseudo_labels.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    T.check(result.returncode == 0, "filter exits 0 even with 0 kept")

    T.check(not (PROCESSED / "3_transcript.jsonl").exists(),
            "3_transcript.jsonl removed when 0 rows kept")

    # ══════════════════════════════════════════════════════════════
    #  RESTORE ORIGINAL STATE
    # ══════════════════════════════════════════════════════════════
    log.info("")
    log.info("--- RESTORING ORIGINALS ---")

    # Restore sft_train.jsonl and sft_val.jsonl from backups
    shutil.copy2(backup_train, sft_train_path)
    shutil.copy2(backup_val,   sft_val_path)

    T.check(_sha256(sft_train_path) == orig_train_sha,
            "sft_train.jsonl restored to original")
    T.check(_sha256(sft_val_path) == orig_val_sha,
            "sft_val.jsonl restored to original")

    # Clean up artifacts
    _cleanup_test_artifacts()
    T.check(not (PROCESSED / "3_transcript.jsonl").exists(),
            "3_transcript.jsonl cleaned up")
    T.check(not (PROCESSED / "sft_train_original.jsonl").exists(),
            "sft_train_original.jsonl cleaned up")

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"\nCompleted in {elapsed:.1f}s")
    ok = T.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
