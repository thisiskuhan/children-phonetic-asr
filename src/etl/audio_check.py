import hashlib
import logging
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from utils import loads, dumps_line

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio validation — checks presence, filesize, and optional MD5
# against the raw transcript JSONL.
#
#   Efficient design:
#     - File presence + size : O(1) per file (stat syscall, no file read)
#     - MD5                  : reads each file once (~2-3 min for 20GB)
#     - Orphan detection     : set subtraction after transcript pass
#
#   audio_dir must be the folder containing U_*.flac files directly.
#   The transcript audio_path field is "audio/U_....flac" — filename is
#   extracted and resolved against audio_dir.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5(path: str) -> str:
    """Stream-compute MD5 in 1 MB chunks — avoids loading full file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_audio_check(
    transcript: str,
    audio_dir: str,
    report_dir: str,
    processed_dir: str,
    check_md5: bool,
    fail_on_error: bool,
    ds_label: str,
    audit_orphan_duration: bool = False,
    word_track_transcript: str | None = None,
) -> dict:
    """
    Validate every audio file referenced in the transcript JSONL.

    Steps:
      1. For every transcript row: single stat() → existence + filesize
      2. Optional MD5 verify  (check_md5=True)
      3. Orphan detection — files in audio_dir not in transcript
      4. Failures → <report_dir>/{n}_audiocheck_failures.jsonl
      5. Orphans  → <processed_dir>/{n}_orphans.jsonl
      6. If word_track_transcript given, enrich orphans with metadata
         (child_id, age_bucket, audio_duration_sec, session_id, orthographic_text)
    """
    audio_path = Path(audio_dir)
    out_dir    = Path(report_dir)
    proc_dir   = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info(f"--- AUDIO DS{ds_label} START ---")
    log.info(f"[AUDIO] transcript   {transcript}")
    log.info(f"[AUDIO] audio dir    {audio_dir}")
    log.info(f"[AUDIO] md5 check    {'ON' if check_md5 else 'OFF'}")

    seen_files: set[str] = set()
    failures:   list[dict] = []
    total_duration_sec: float = 0.0

    # Fast line count for progress bar
    with open(transcript, encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)

    n_total = n_missing = n_size_mismatch = n_md5_fail = 0

    with open(transcript, encoding="utf-8") as f:
        with tqdm(f, total=n_lines, desc=f"[AUDIO] DS{ds_label}", unit="file",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            for line in pbar:
                row = loads(line)
                n_total += 1

                # Accumulate duration for total-hours reporting
                total_duration_sec += row.get("audio_duration_sec", 0.0)

                # transcript stores "audio/U_....flac" — take just the filename
                fname = Path(row["audio_path"]).name
                fpath = audio_path / fname
                seen_files.add(fname)
                fail: dict = {}

                # Single stat() syscall — covers existence AND filesize together
                try:
                    st = fpath.stat()
                except FileNotFoundError:
                    n_missing += 1
                    fail["missing"] = True
                    failures.append({**row, "failures": fail})
                    continue

                actual_size = st.st_size
                if actual_size != row["filesize_bytes"]:
                    n_size_mismatch += 1
                    fail["size_expected"] = row["filesize_bytes"]
                    fail["size_actual"]   = actual_size

                # MD5 (optional — reads full file bytes)
                if check_md5:
                    actual_md5 = _md5(str(fpath))
                    if actual_md5 != row["md5_hash"]:
                        n_md5_fail += 1
                        fail["md5_expected"] = row["md5_hash"]
                        fail["md5_actual"]   = actual_md5

                if fail:
                    failures.append({**row, "failures": fail})

    # ---- Orphan detection — files in folder not referenced by transcript ----
    # Case-insensitive glob: covers .flac and .FLAC (real-world datasets vary)
    # Single iterdir pass (replaces two separate .glob("*.flac") + .glob("*.FLAC") scans)
    all_files_map = {
        p.name: p for p in audio_path.iterdir()
        if p.is_file() and p.suffix.lower() == ".flac"
    }
    orphan_names  = sorted(set(all_files_map) - seen_files)
    n_orphans     = len(orphan_names)

    # ---- Save failures only if any exist ----
    prefix    = f"{ds_label}_" if ds_label else ""
    fail_path = out_dir / f"{prefix}audiocheck_failures.jsonl"
    if failures:
        with open(fail_path, "w", encoding="utf-8") as f:
            for row in failures:
                f.write(dumps_line(row))

    # ---- Save orphans → processed dir (enriched from word-track if available) ----
    orphan_path = proc_dir / f"{prefix}orphans.jsonl"

    # Load word-track transcript into a lookup keyed by utterance_id.
    # Single-pass read — same pattern as the transcript loop above.
    word_track_lookup: dict[str, dict] = {}
    if word_track_transcript and Path(word_track_transcript).is_file():
        log.info(f"[AUDIO] loading word-track transcript for orphan enrichment …")
        with open(word_track_transcript, encoding="utf-8") as wf:
            for wline in wf:
                wrow = loads(wline)
                word_track_lookup[wrow["utterance_id"]] = wrow
        log.info(f"[AUDIO]   word-track entries: {len(word_track_lookup):,}")

    n_enriched = 0
    n_enrich_size_mismatch = 0

    # Only add the flag column when coverage is partial (not all orphans in word track).
    partial_coverage = (word_track_lookup
                        and any(fn.replace(".flac", "") not in word_track_lookup
                                for fn in orphan_names))

    if orphan_names:
        with open(orphan_path, "w", encoding="utf-8") as f:
            for fname in orphan_names:
                uid = fname.replace(".flac", "")
                actual_size = all_files_map[fname].stat().st_size
                row_out: dict = {
                    "utterance_id": uid,
                    "filesize_bytes": actual_size,
                }

                wt = word_track_lookup.get(uid)

                if partial_coverage:
                    row_out["word_track_matched"] = wt is not None

                if wt:
                    # Validate filesize consistency between word track and disk
                    wt_size = wt.get("filesize_bytes")
                    if wt_size is not None and wt_size != actual_size:
                        n_enrich_size_mismatch += 1

                    # Enrich with word-track metadata
                    for key in ("child_id", "session_id", "audio_path",
                                "audio_duration_sec", "age_bucket",
                                "md5_hash", "orthographic_text"):
                        if key in wt:
                            row_out[key] = wt[key]

                    row_out["dataset"] = int(ds_label) if ds_label.isdigit() else ds_label
                    n_enriched += 1

                f.write(dumps_line(row_out))

    if word_track_lookup:
        log.info(f"[AUDIO]   orphans enriched   {n_enriched:>10,} / {n_orphans:,}")
        if n_enrich_size_mismatch:
            log.warning(f"[AUDIO]   ✗  size mismatch   {n_enrich_size_mismatch:>10,}  "
                        f"(word-track filesize ≠ disk stat)")

    # ---- Orphan duration audit (off by default — header-only read via torchaudio.info) ----
    orphan_hours: float | None = None
    if audit_orphan_duration and orphan_names:
        log.info(f"[AUDIO] scanning {n_orphans:,} orphan files for duration (soundfile.info) ...")
        orphan_dur_sec = 0.0
        n_orphan_err   = 0
        for fname in orphan_names:
            fpath_o = all_files_map[fname]
            try:
                info = sf.info(str(fpath_o))
                orphan_dur_sec += info.frames / info.samplerate
            except Exception:
                n_orphan_err += 1
        orphan_hours = orphan_dur_sec / 3600
        log.info(f"[AUDIO]   orphan duration  {orphan_hours:>10.2f} hrs  "
                 f"({n_orphans - n_orphan_err:,} ok, {n_orphan_err:,} unreadable)")

    n_ok   = n_total - len(failures)
    total_hours = total_duration_sec / 3600
    rc_ok  = "✓" if n_missing == 0 and n_size_mismatch == 0 else "✗"
    rc_md5 = "✓" if n_md5_fail == 0 else "✗"

    log.info("")
    log.info(f"[AUDIO] rows in transcript  {n_total:>10,}")
    log.info(f"[AUDIO] files in audio dir  {len(all_files_map):>10,}")
    log.info(f"[AUDIO]   {rc_ok}  OK              {n_ok:>10,}")
    log.info(f"[AUDIO]   {'✗' if n_missing else '✓'}  missing         {n_missing:>10,}")
    log.info(f"[AUDIO]   {'✗' if n_size_mismatch else '✓'}  size mismatch   {n_size_mismatch:>10,}")
    if check_md5:
        log.info(f"[AUDIO]   {rc_md5}  md5 mismatch    {n_md5_fail:>10,}")
    if n_orphans:
        log.info(f"[AUDIO]   !  orphans         {n_orphans:>10,}  → {orphan_path}")
        if orphan_hours is not None:
            log.info(f"[AUDIO]   !  orphan hours     {orphan_hours:>10.2f} hrs")
    else:
        log.info(f"[AUDIO]   ✓  orphans         {n_orphans:>10,}")
    if failures:
        log.info(f"[AUDIO] failures → {fail_path}")
    log.info(f"[AUDIO] total duration  {total_hours:>10.2f} hrs")
    log.info(f"--- AUDIO DS{ds_label} END ---")

    # ---- Hard-fail — training on partial audio silently = catastrophic ----
    if fail_on_error and (n_missing > 0 or n_size_mismatch > 0 or n_md5_fail > 0):
        raise RuntimeError(
            f"Audio integrity FAILED for DS{ds_label}: "
            f"{n_missing} missing, {n_size_mismatch} size mismatch, "
            f"{n_md5_fail} md5 mismatch.  "
            f"Set audio_check.fail_on_error=false to continue despite errors."
        )

    return {
        "n_total":         n_total,
        "n_ok":            n_ok,
        "n_missing":       n_missing,
        "n_size_mismatch": n_size_mismatch,
        "n_md5_fail":      n_md5_fail if check_md5 else None,
        "n_orphans":       n_orphans,
        "n_enriched":      n_enriched if n_enriched < n_orphans else None,
        "orphan_hours":    round(orphan_hours, 3) if orphan_hours is not None else None,
        "total_hours":     round(total_hours, 3),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AudioChecker:
    def __init__(self, cfg: dict, ds_key: int):
        paths  = cfg["paths"]
        ac_cfg = cfg["audio_check"]

        self.transcript        = paths["datasets"][ds_key]
        self.audio_dir         = paths["audio_dirs"][ds_key]
        self.report_dir        = paths["reports"]
        self.processed_dir     = paths["processed"]
        self.check_md5             = ac_cfg["check_md5"]
        self.fail_on_error         = ac_cfg.get("fail_on_error", False)
        self.ds_label              = str(ds_key)
        self.audit_orphan_duration = ac_cfg.get("audit_orphan_duration", False)
        self.word_track_transcript = paths.get("word_track_transcript")

    def run(self) -> dict:
        return run_audio_check(
            self.transcript,
            self.audio_dir,
            self.report_dir,
            self.processed_dir,
            self.check_md5,
            self.fail_on_error,
            self.ds_label,
            self.audit_orphan_duration,
            self.word_track_transcript,
        )
