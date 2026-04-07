# Audio Check — File Integrity Validation (Stage 0)

> **Note:** Numbers in this document reflect an earlier pipeline run. The final Run 24 numbers (DS1 = 6.4 hrs, DS2 = 78.6 hrs, total = 85.1 hrs) are in [README.md](../README.md). Run 24 artifacts: [eda_controls.json](../study/runs/run_24/reports/eda_controls.json), [eda_failed_loads.json](../study/runs/run_24/reports/eda_failed_loads.json), [ETL log](../study/runs/run_24/logs/309_log_20260329_190324.log).

> **File:** `src/etl/audio_check.py`
> **Runs via:** `python pipeline.py --etl` (first stage) or standalone via `AudioChecker(cfg, ds_key=key).run()`
> **Input:** Raw JSONL (`data/raw/*_train_phon_transcripts.jsonl`) + raw audio (`data/raw/*_audio/`)
> **Output:** `data/reports/{key}_audiocheck_failures.jsonl` (if failures) + `data/processed/{key}_orphans.jsonl` (if orphans)
> **Config:** `config.yaml` → `audio_check:` section

---

## Rationale — Why This Stage Exists

Every downstream stage assumes audio files exist, are complete, and match their manifest metadata. If this assumption fails silently:

- **Sanity** writes rows pointing to missing/corrupt audio → downstream loaders crash at training time
- **Audio EDA** calls `torchaudio.load()` on broken files → worker exceptions, skewed corpus statistics
- **Training** encounters truncated waveforms mid-epoch → `loss = NaN`, wasted GPU hours

Audio Check is the **zero-cost insurance gate**. It runs before any audio is decoded, using only filesystem metadata (`stat()`) and optional MD5, so it finishes in minutes even on 150K+ files. Every failure it catches here saves hours of debugging later.

## How It Helps the Overall Pipeline

| Downstream stage | What Audio Check prevents |
|-----------------|---------------------------|
| Sanity (`--etl`) | Rows referencing missing/truncated files passing through to cleaned manifests |
| Audio EDA | `torchaudio.load()` crashes on corrupt files inflating error sentinel counts |
| Tokenizer | Vocab built from rows whose audio will never load — wasted logit capacity |
| Data Split | Speaker duration accounting skewed by phantom files |
| Training | Mid-epoch crashes, `NaN` loss, wasted GPU budget |
| SSL Pre-training | Orphan detection reveals the **unlabeled audio pool** — 114K orphans = 80+ hours of children's audio available for self-supervised continued pre-training |

---

## What It Does

Validates every audio file referenced in the raw transcript JSONL — existence, filesize, optional MD5 hash — then detects orphan files (audio files not referenced by any transcript row).

**Design:**

- **File presence + size:** O(1) per file (single `stat()` syscall, no file read)
- **MD5:** Optional full-file read (~2–3 min for 20GB)
- **Orphan detection:** `set(dir_listing) - set(transcript_references)` after the full pass

---

## Step-by-Step

### 1. Existence + Filesize

**What:** For each transcript row, `stat()` the audio file referenced by `audio_path`.

**Why:** Files can go missing during download, extraction, or storage corruption. A missing file at training time causes a crash. A size mismatch means the file was truncated (partial download, disk full) — the waveform will be shorter than expected, causing duration mismatches and CTC alignment failures.

**How:** Single `stat()` syscall returns both existence and size. If `FileNotFoundError` → count as missing. If `st_size != row["filesize_bytes"]` → count as size mismatch.

**Total hours:** Duration is accumulated from `audio_duration_sec` in the transcript and printed per dataset + overall summary.

---

### 2. MD5 Verification (Optional)

**What:** Stream-compute MD5 in 1MB chunks, compare to `row["md5_hash"]`.

**Why:** A file can exist with the correct size but have corrupted content (bit-flip, partial overwrite). MD5 catches this. Slow but thorough — reads every byte.

**How:** Enabled by `audio_check.check_md5: true` in config. Reads file in 1MB chunks to avoid memory spike on large files.

---

### 3. Orphan Detection

**What:** After processing all transcript rows, glob `*.flac` **and** `*.FLAC` in the audio directory and subtract the set of filenames seen in the transcript.

**Why:** Orphan files waste storage and can cause confusion. In competition datasets, orphans often indicate dataset version mismatches (files from a different split leaked into the directory). The orphan count for DS2 (~114K) confirmed a known dataset packaging issue. The case-insensitive glob (`*.flac` + `*.FLAC`) handles real-world datasets where file extensions may vary.

**Output:** `data/processed/{key}_orphans.jsonl` — each row has `utterance_id` and `filesize_bytes`.

---

### 4. Orphan Duration Audit (Optional)

**What:** When `audit_orphan_duration: true`, calls `sf.info()` (soundfile) on every orphan file to read its actual duration from the FLAC header.

**Why:** Orphans have no transcript rows, so their duration is unknown. For the DS2 114K orphan pool (potential SSL pretraining data), total hours matters — it determines whether self-supervised pretraining is viable. Without this flag, the orphan count is known but the audio volume is not.

**How:** `sf.info()` reads only the file header (no decode) — returns `frames` and `samplerate`. Duration = `frames / samplerate`. On 114K files this reads ~10 MB of headers instead of ~22 GB of audio. Expect ~2–3 minutes.

**Flag:** `audio_check.audit_orphan_duration: false` — always off by default. Set to `true` manually when you need the number, then set it back. Has no effect on any other pipeline output.

---

### 5. Failure Reporting

**What:** Any file that fails existence, size, or MD5 check gets written to `data/reports/{key}_audiocheck_failures.jsonl` with the original row data plus a `failures` dict detailing what went wrong.

---

## Output Summary

Prints per-dataset:

```
[AUDIO] rows in transcript  141,024
[AUDIO] files in audio dir  255,073
[AUDIO]   ✓  OK              141,024
[AUDIO]   ✓  missing               0
[AUDIO]   ✓  size mismatch         0
[AUDIO]   ✓  md5 mismatch          0
[AUDIO]   !  orphans         114,022  → data/processed/2_orphans.jsonl
[AUDIO]   !  orphan hours      ??.??  hrs          ← only when audit_orphan_duration: true
[AUDIO] total duration        78.64 hrs
```

Cross-dataset summary (printed by `pipeline.py`):

```
--- AUDIO CHECK SUMMARY ---
[AUDIO] DS1    12,043 OK      3.25 hrs
[AUDIO] DS2   141,024 OK     78.56 hrs
[AUDIO] ALL   153,067 OK     81.81 hrs
--- AUDIO CHECK SUMMARY END ---
```

---

## Config

```yaml
audio_check:
  check_md5: true                 # verify MD5 hashes (slow but thorough)
  audit_orphan_duration: false    # scan orphan files for actual duration — slow, enable manually
```

> Code-level defaults (not in config.yaml): `fail_on_error=false`, `progress_interval=10000`.

**`check_md5`:** When `true`, verifies stored MD5 hashes against recomputed hashes. Slow but thorough — catches bit-rot and partial downloads.

**`audit_orphan_duration`:** When `true`, reads the FLAC header of every orphan file via `sf.info()` (soundfile — header-only, no decode) and logs total orphan hours. Always `false` by default — flip it manually when needed, then flip it back. Has zero effect on transcript validation, MD5 checks, or any other output.

---

## When to Run

Run `--audiocheck` once after downloading/extracting the dataset. It doesn't need to be re-run unless you re-download or move files. All other pipeline stages (`--sanity`, `--tkn`, `--eda`) assume the audio is intact.

---

## Decision Evolution — Mar 2026 Update

This section preserves the original design while documenting what changed in the live pipeline.

| Earlier choice | Current state | Why changed / what we learned |
|---|---|---|
| Audio check is a hard gate before ETL | Still true; no policy change | This stayed stable because it prevents downstream silent failures.
| Use EDA to detect load + duration issues | Confirmed by current reports: `eda_failed_loads.json` count=1, `eda_duration_mismatch.json` count=3 | Even after sanity cleaning, a small number of audio-level failures remain and must be removed post-load analysis.
| Treat orphan handling as optional analytics | Still optional (`audit_orphan_duration` default false) | We keep it optional to avoid slowing routine runs; enable only for SSL planning.

**Current rationale:** we kept the original gate and added stronger evidence that the second line of defense (Audio EDA anomaly removal) is necessary in practice.
