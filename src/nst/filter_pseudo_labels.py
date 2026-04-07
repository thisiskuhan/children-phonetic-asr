#!/usr/bin/env python3
"""NST pseudo-label filtering — strict quality gate before SFT training.

Reads raw pseudo-labels from teacher_infer.py, applies multi-signal
confidence filtering, runs EDA-style sanity validation, and writes
SFT-ready output.  DataSplitter re-split is handled by pipeline.py.

Pipeline (single --nst call):
  1. Load raw pseudo-labels from teacher_infer.py output
  2. Apply confidence filters (beam-greedy CER, norm score, tps physics, etc.)
  3. Build SFT-compatible rows (exact same schema as sft_train.jsonl)
  4. Sanity validation — mirrors EDA sanitiser checks
  5. Write pseudo_labelled.jsonl
  6. Write pseudo rows as 3_transcript.jsonl (DataSplitter input format)

Usage:
    python src/nst/filter_pseudo_labels.py

Output:
    data/processed/pseudo_labelled.jsonl  — SFT-compatible pseudo-label rows
    data/processed/3_transcript.jsonl     — pseudo transcript for DataSplitter
    data/reports/nst_filter_report.json   — detailed report
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Project imports ────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_config
from utils import loads, dumps, dumps_line, nearest_rank_pctl

log = logging.getLogger(__name__)

# ── Required fields — must match trainer/dataset.py exactly ──────
_REQUIRED_FIELDS = (
    "utterance_id",
    "audio_path",
    "audio_duration_sec",
    "age_bucket",
    "phonetic_text",
    "n_phonemes",
    "dataset",
)


def _sanity_check(rows: list[dict], tps_min: float, tps_max: float,
                   min_dur: float, max_dur: float,
                   min_phon: int = 1) -> dict:
    """EDA-style sanity validation on the final pseudo-label set.

    Returns a dict of check results for logging and the report.
    """
    violations: dict[str, int] = defaultdict(int)
    n = len(rows)

    for row in rows:
        t = row["phonetic_text"]
        dur = row["audio_duration_sec"]
        n_phon = row["n_phonemes"]
        n_chars = len(t.replace(" ", ""))

        # Field presence
        for f in _REQUIRED_FIELDS:
            if f not in row:
                violations["missing_field"] += 1

        # n_phonemes consistency
        if n_phon != n_chars:
            violations["n_phonemes_mismatch"] += 1

        # Empty label
        if not t.strip():
            violations["empty_label"] += 1

        # Duration bounds
        if dur > max_dur:
            violations["over_duration"] += 1
        if dur < min_dur:
            violations["under_duration"] += 1

        # Physics (tps)
        if dur > 0 and n_chars > 0:
            tps = n_chars / dur
            if tps < tps_min:
                violations["tps_below_floor"] += 1
            if tps > tps_max:
                violations["tps_above_ceiling"] += 1

        # Run-on (no spaces, long label — likely garbage)
        if " " not in t and len(t) > 15:
            violations["run_on"] += 1

        # Short phoneme count
        if n_chars < min_phon:
            violations["short_phonemes"] += 1

    return {"n_rows": n, "violations": dict(violations)}


def _dataset_stats(rows: list[dict]) -> dict:
    """Compute EDA-style summary stats for a JSONL dataset (parsed dicts).

    Returns a flat dict suitable for printing and JSON serialisation.
    """
    n = len(rows)
    if n == 0:
        return {"n_rows": 0}

    durations = [r["audio_duration_sec"] for r in rows]
    total_hrs = sum(durations) / 3600

    # phonemes / tps
    tps_vals = []
    n_phonemes_all = []
    for r in rows:
        d = r["audio_duration_sec"]
        np_ = r.get("n_phonemes", len(r.get("phonetic_text", "").replace(" ", "")))
        n_phonemes_all.append(np_)
        if d > 0 and np_ > 0:
            tps_vals.append(np_ / d)
    tps_vals.sort()
    durations.sort()
    n_phonemes_all.sort()

    # age buckets
    by_age: dict[str, int] = defaultdict(int)
    for r in rows:
        by_age[r.get("age_bucket", "unknown")] += 1

    # dataset split
    by_ds: dict[int, int] = defaultdict(int)
    for r in rows:
        by_ds[r.get("dataset", 0)] += 1

    return {
        "n_rows":       n,
        "total_hours":  round(total_hrs, 3),
        "dur_p05":      nearest_rank_pctl(durations, 5, presorted=True, decimals=3),
        "dur_p50":      nearest_rank_pctl(durations, 50, presorted=True, decimals=3),
        "dur_p95":      nearest_rank_pctl(durations, 95, presorted=True, decimals=3),
        "phon_p50":     nearest_rank_pctl(n_phonemes_all, 50, presorted=True, decimals=0),
        "tps_p05":      nearest_rank_pctl(tps_vals, 5, presorted=True, decimals=3),
        "tps_p50":      nearest_rank_pctl(tps_vals, 50, presorted=True, decimals=3),
        "tps_p95":      nearest_rank_pctl(tps_vals, 95, presorted=True, decimals=3),
        "by_age":       dict(by_age),
        "by_dataset":   {str(k): v for k, v in sorted(by_ds.items())},
    }


def _log_dataset_stats(label: str, stats: dict) -> None:
    """Log dataset stats block via log.info — mirrors EDA style."""
    log.info("[FILTER] --- %s ---", label)
    log.info("[FILTER]   rows:       %10s", f"{stats['n_rows']:,}")
    if stats["n_rows"] == 0:
        return
    log.info("[FILTER]   hours:      %10.2f", stats['total_hours'])
    log.info("[FILTER]   duration:   p05=%.2f  p50=%.2f  p95=%.2f",
             stats['dur_p05'], stats['dur_p50'], stats['dur_p95'])
    log.info("[FILTER]   n_phonemes: p50=%s", stats['phon_p50'])
    log.info("[FILTER]   tps:        p05=%.2f  p50=%.2f  p95=%.2f",
             stats['tps_p05'], stats['tps_p50'], stats['tps_p95'])
    log.info("[FILTER]   age buckets:")
    for age in sorted(stats["by_age"]):
        log.info("[FILTER]     %-10s  %8s", age, f"{stats['by_age'][age]:,}")
    if stats.get("by_dataset"):
        log.info("[FILTER]   datasets:")
        for ds, cnt in sorted(stats["by_dataset"].items()):
            log.info("[FILTER]     ds%-5s    %8s", ds, f"{cnt:,}")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    t0 = time.time()
    root = Path(__file__).resolve().parents[2]

    # ── Load config ──────────────────────────────────────────────
    cfg = load_config(root / "src" / "config" / "config.yaml")
    nst = cfg["nst"]
    eda = cfg["eda"]

    max_bg_cer    = nst["max_beam_greedy_cer"]
    min_norm      = nst["min_norm_score"]
    min_chars     = nst["min_pred_chars"]
    min_phon      = nst.get("min_phonemes", eda.get("min_phonemes", 1))
    tps_min       = eda["tps_min"]           # physics floor — same as EDA sanitiser
    tps_max       = eda["tps_max"]           # physics ceiling — same as EDA sanitiser
    min_dur       = nst["min_duration"]
    max_dur       = nst["max_duration"]
    raw_path      = root / nst["output_raw"]
    out_path      = root / nst["output_filtered"]
    processed_dir = root / "data" / "processed"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load raw pseudo-labels ───────────────────────────────────
    log.info("[FILTER] loading raw pseudo-labels from %s …", raw_path.name)
    raw_rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            raw_rows.append(loads(line))
    n_raw = len(raw_rows)
    log.info("[FILTER] raw rows: %s", f"{n_raw:,}")

    # ══════════════════════════════════════════════════════════════
    #  Confidence filtering (ALL must pass — AND logic)
    # ══════════════════════════════════════════════════════════════
    kept = []
    reasons: dict[str, int] = defaultdict(int)

    for row in raw_rows:
        reject = []

        # Filter 0: beam-greedy agreement (strongest signal)
        if row["beam_greedy_cer"] > max_bg_cer:
            reject.append("beam_greedy_cer")

        # Filter 1: CTC confidence (norm score)
        if row["norm_score"] < min_norm:
            reject.append("norm_score")

        # Filter 2: minimum prediction length
        if row["n_greedy_chars"] < min_chars:
            reject.append("min_chars")

        # Filter 3: greedy pred must be non-empty
        if not row["greedy_pred"].strip():
            reject.append("empty_pred")

        # Filter 4: physics — phonemes/sec must be in plausible range
        dur = row["audio_duration_sec"]
        if dur > 0 and row["n_greedy_chars"] > 0:
            tps = row["n_greedy_chars"] / dur
            if not (tps_min <= tps <= tps_max):
                reject.append("physics_tps")
        elif dur <= 0:
            reject.append("zero_duration")

        # Filter 5: duration bounds
        if dur > max_dur or dur < min_dur:
            reject.append("duration_bounds")

        # Filter 6: minimum phoneme count (mirrors eda.min_phonemes)
        if row["n_greedy_chars"] < min_phon:
            reject.append("short_phonemes")

        if reject:
            for r in reject:
                reasons[r] += 1
        else:
            kept.append(row)

    n_kept = len(kept)
    n_rejected = n_raw - n_kept

    # ══════════════════════════════════════════════════════════════
    #  Score-based ranking — keep only top pseudo_keep_fraction
    # ══════════════════════════════════════════════════════════════
    keep_frac = nst.get("pseudo_keep_fraction", 1.0)
    if 0 < keep_frac < 1.0 and n_kept > 0:
        # Rank by norm_score (greedy-only: higher = more confident)
        kept.sort(key=lambda r: r["norm_score"], reverse=True)
        n_keep_ranked = max(1, int(n_kept * keep_frac))
        n_cut = n_kept - n_keep_ranked
        kept = kept[:n_keep_ranked]
        log.info("[FILTER] score ranking: kept top %.0f%% = %s rows "
                 "(cut %s low-confidence)",
                 keep_frac * 100, f"{n_keep_ranked:,}", f"{n_cut:,}")
        n_kept = len(kept)

    # ══════════════════════════════════════════════════════════════
    #  Build SFT-compatible rows (exact same schema as sft_train.jsonl)
    # ══════════════════════════════════════════════════════════════
    log.info("[FILTER] building SFT-compatible rows …")
    sft_rows = []
    for row in kept:
        # Use beam_pred as label (higher quality than greedy)
        label_text = row["beam_pred"].strip()
        n_phonemes = len(label_text.replace(" ", ""))

        sft_rows.append({
            "utterance_id":      row["utterance_id"],
            "audio_path":        row["audio_path"],
            "audio_duration_sec": row["audio_duration_sec"],
            "age_bucket":        row["age_bucket"],
            "phonetic_text":     label_text,
            "n_phonemes":        n_phonemes,
            "dataset":           row["dataset"],
            # Force empty child_id so DataSplitter routes ALL pseudo-labels
            # to train set only.  Real child_ids leak pseudo rows into val,
            # contaminating evaluation (caused Run 16 test inflation).
            "child_id":          "",
            # Extra metadata (SFTDataset ignores extra keys)
            "pseudo_label":      True,
            "norm_score":        row["norm_score"],
        })

    # ══════════════════════════════════════════════════════════════
    #  EDA-style sanity validation on filtered output
    # ══════════════════════════════════════════════════════════════
    log.info("[FILTER] running sanity validation …")
    sanity = _sanity_check(sft_rows, tps_min, tps_max, min_dur, max_dur, min_phon)

    if sanity["violations"]:
        log.warning("[FILTER] SANITY VIOLATIONS FOUND:")
        for name, cnt in sorted(sanity["violations"].items(), key=lambda x: -x[1]):
            log.warning("[FILTER]   %-25s %8s", name, f"{cnt:,}")
    else:
        log.info("[FILTER] ✓  all %s rows pass sanity checks", f"{n_kept:,}")

    # ══════════════════════════════════════════════════════════════
    #  Write output
    # ══════════════════════════════════════════════════════════════
    log.info("[FILTER] writing %s rows → %s", f"{n_kept:,}", out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(dumps_line(row))

    # ══════════════════════════════════════════════════════════════
    #  Write pseudo rows as 3_transcript.jsonl (DataSplitter input)
    # ══════════════════════════════════════════════════════════════
    pseudo_transcript = processed_dir / "3_transcript.jsonl"
    if n_kept > 0:
        log.info("[FILTER] writing %s pseudo rows → %s",
                 f"{n_kept:,}", pseudo_transcript.name)
        with open(pseudo_transcript, "w", encoding="utf-8") as f:
            for row in sft_rows:
                f.write(dumps_line(row))
    else:
        # Remove stale file so pipeline.py skips re-split
        if pseudo_transcript.exists():
            pseudo_transcript.unlink()
        log.warning("[FILTER] 0 rows kept — no 3_transcript.jsonl written")

    # ══════════════════════════════════════════════════════════════
    #  REPORT — two sections: pseudo-filtered + combined
    # ══════════════════════════════════════════════════════════════

    # ── Section 1: Pseudo-filtered data stats ────────────────────
    pseudo_stats = _dataset_stats(sft_rows)

    # Confidence distributions (pseudo-specific)
    norm_scores = sorted([r["norm_score"] for r in kept], reverse=True)

    log.info("--- NST FILTER REPORT START ---")
    log.info("[FILTER] raw total:      %10s", f"{n_raw:,}")
    log.info("[FILTER] kept:           %10s  (%.1f%%)", f"{n_kept:,}",
             n_kept / max(n_raw, 1) * 100)
    log.info("[FILTER] rejected:       %10s  (%.1f%%)", f"{n_rejected:,}",
             n_rejected / max(n_raw, 1) * 100)
    log.info("[FILTER] sanity clean:   %s",
             "YES" if not sanity["violations"] else "NO — see above")
    log.info("[FILTER] rejection reasons (a row can hit multiple):")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        log.info("[FILTER]   %-20s %8s", reason, f"{count:,}")

    log.info("[FILTER] confidence distributions (pseudo only):")
    log.info("[FILTER]   norm_score:  p50=%.4f  p10=%.4f",
             nearest_rank_pctl(norm_scores, 50, presorted=True),
             nearest_rank_pctl(norm_scores, 90, presorted=True))

    # norm_score threshold sweep table
    log.info("[FILTER] norm_score threshold sweep (min_chars >= %s):",
             min_chars)
    log.info("[FILTER]   %10s  %8s  %8s", "norm_score >", "kept", "% of raw")
    for ns_t in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]:
        ct = sum(1 for r in raw_rows
                 if r["norm_score"] >= ns_t
                 and r["n_greedy_chars"] >= min_chars
                 and r["greedy_pred"].strip())
        log.info("[FILTER]   %10.1f  %8s  %7.1f%%", ns_t, f"{ct:,}",
                 ct / max(n_raw, 1) * 100)

    _log_dataset_stats("PSEUDO-LABELLED DATA", pseudo_stats)

    elapsed = time.time() - t0
    log.info("[FILTER] time: %.1fs", elapsed)
    log.info("[FILTER] output: %s", out_path)
    log.info("[FILTER] 3_transcript.jsonl: %s", pseudo_transcript)
    log.info("--- NST FILTER REPORT END ---")

    # ── Save filter report as JSON ───────────────────────────────
    report_path = root / "data" / "reports" / "nst_filter_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_total": n_raw,
        "kept": n_kept,
        "rejected": n_rejected,
        "keep_rate": round(n_kept / max(n_raw, 1), 4),
        "thresholds": {
            "max_beam_greedy_cer": max_bg_cer,
            "min_norm_score": min_norm,
            "min_pred_chars": min_chars,
            "min_phonemes": min_phon,
            "tps_min": tps_min,
            "tps_max": tps_max,
            "min_duration": min_dur,
            "max_duration": max_dur,
        },
        "rejection_reasons": dict(reasons),
        "sanity_violations": sanity["violations"],
        "pseudo_stats": pseudo_stats,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(dumps(report, indent=2))
    log.info("[FILTER] report: %s", report_path)


if __name__ == "__main__":
    main()
