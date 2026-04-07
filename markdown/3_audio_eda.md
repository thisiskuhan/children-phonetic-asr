# Audio EDA — Control Calibration (Stage 2)

> **Note:** Numbers below reflect an earlier pipeline run. The final Run 24 analysed 76.8 hrs across 134,434 rows. See [README.md](../README.md). Run 24 artifacts: [eda_controls.json](../study/runs/run_24/reports/eda_controls.json), [eda_duration_mismatch.json](../study/runs/run_24/reports/eda_duration_mismatch.json), [eda_failed_loads.json](../study/runs/run_24/reports/eda_failed_loads.json).

> **File:** `src/etl/eda_processor.py` → `_run_audio_eda()`
> **Runs via:** `python pipeline.py --etl` (third stage, after sanity)
> **Prerequisite:** Sanity must have run first (reads from `data/processed/`)
> **Input:** Cleaned JSONL (`data/processed/*_transcript.jsonl`) + raw audio (`data/raw/*_audio/`)
> **Output:** `data/reports/eda_controls.json`
> **Config:** `config.yaml` → `audio_eda:` section
> **Datasets:** Derived from `cfg["datasets"]` — same canonical list as Sanity and tokenizer

---

## Rationale — Why This Stage Exists

Training a CTC model requires dozens of design decisions: should we normalise volume? Trim silence? Resample? Use a weighted sampler? Cap batch duration? Each decision depends on **corpus-level acoustic statistics** that no human can eyeball from 150K+ audio files.

Audio EDA replaces guesswork with measurement. It computes every acoustic metric that drives a training decision, writes the result to a single JSON (`eda_controls.json`), and prints a human-readable recommendation for each. The trainer reads this file and configures itself accordingly — no manual knob-turning.

Without this stage:

- **Volume mismatch**: If RMS spread is 100×, the model learns to rely on loudness rather than spectral content. You'd only discover this after training when PER won't converge.
- **Silent padding**: 20%+ files with leading/trailing silence waste sequence positions on CTC blanks. VAD trimming is the fix, but you need evidence to justify it.
- **Batch OOM**: A single 60-second file forces batch-size-1, wasting 95% of GPU memory. Duration profiling gives the collator its `max_length`.
- **Age bias**: Without sampler weights, the model sees 4× more 8–11yr data than toddler data and underfits young children.

## How It Helps the Overall Pipeline

| Downstream stage | What Audio EDA provides |
|-----------------|-------------------------|
| Tokenizer | Removes audio-broken rows from processed manifests → tokenizer reads only loadable files |
| Data Split | Clean row counts for accurate duration-targeted val sizing |
| Training | `eda_controls.json` drives: RMS normalisation flag, VAD flag, batch sizing, sampler weights, resampling decision |
| `training_controls.json` | Split stage recomputes train-only stats; EDA provides corpus-wide baseline for comparison |
| Model Selection | Reuses the same audio loading pattern; EDA's transcript cleanup ensures model-sel analyses only valid files |

---

## What It Does

Streams cleaned manifests → multiprocessed audio analysis → writes `eda_controls.json` that drives every downstream training decision (normalization, clipping, sampling, batching, VAD, resampling).

**Design constraints:**

- **Streaming** — never loads the full dataset into RAM; generator feeds `Pool.imap` directly
- **Multiprocessing** — `mp.Pool` over manifest rows with configurable worker count
- **Deterministic** — fixed seed for spectral centroid subset selection
- **torchaudio only** — single audio I/O backend, no librosa

---

## Architecture

```
cleaned JSONL ──→ _iter_manifest() ──→ generator ──→ Pool.imap(_analyse_file, ...) ──→ _Accumulator ──→ eda_controls.json
                    (streaming)           │               (per-file worker)              (corpus stats)
                                          │
                                     centroid_indices
                                     (deterministic subset)
```

### Worker Design

Each worker receives a tuple `(audio_path, utterance_id, child_id, duration_sec, age_bucket, compute_centroid)`. Config is stashed in a module-level global via `_worker_init` (called once per process, avoids serializing config per task).

**Error handling:** Workers return error sentinel strings (`_ERR_LOAD`, `_ERR_EMPTY`, `_ERR_NAN`) instead of raising exceptions. The main loop classifies errors by type and counts them. This prevents a single corrupt file from killing the entire pool.

### Zero-Data Guard

If `n_ok == 0` after the pool completes, a `RuntimeError` is raised immediately — prevents silent division-by-zero in all downstream metric computations.

---

## Step-by-Step: What Each Metric Does

### 1. RMS — Volume Distribution

**What:** Per-file root-mean-square power: `sqrt(mean(waveform²))`, computed **after DC offset removal** (`wav = wav - wav.mean()`).

**Why DC removal:** Child recordings from clinical settings often have microphone DC bias — a constant offset in the waveform. Without removal, RMS is inflated by the DC component, making quiet files look louder than they are. This also affects the silence ratio (lead/trail), since the DC bias lifts the floor.

**Why RMS:** If some files are whisper-quiet and others are clipping-loud, the model sees wildly different feature magnitudes for the same phoneme. CTC alignment becomes noisy — the model learns to rely on volume patterns rather than spectral content.

**How it drives a decision:**

- `p01_clamped = max(rms_p01, 1e-4)` — prevents division by near-zero for near-silence files
- `spread_ratio = rms_p99 / rms_p01_clamped`
- If `spread_ratio > 10.0` → recommend RMS normalization

**Config:** `audio_eda.rms.spread_ratio_threshold`

---

### 2. Clipping — Saturation Detection

**What:** Per-file: fraction of samples where `|x| ≥ 0.999`.

**Why:** Clipped samples are information-destroying — the true waveform peak is lost. Heavy clipping introduces harmonic distortion that confuses the feature extractor, creating phantom spectral content at odd harmonics.

**How it drives a decision:**

- A file is "heavily clipped" if >5% of its samples clip
- If >1% of corpus files are heavily clipped → recommend dropping them

**Config:** `audio_eda.clipping.sample_threshold`, `file_percent_threshold`, `corpus_percent_threshold`

**Units:** `file_percent_threshold` is in percent (e.g., `5.0` = 5%), divided by 100 in code for comparison against `clipped_ratio` (a fraction). `corpus_percent_threshold` stays in percent for comparison against `pct_clipped` (also percent). Units are consistent throughout.

---

### 3. Speaker Dominance

**What:** Total audio hours per `child_id`. Reports top-K speakers by hour share.

**Why:** If 5 speakers produce 40% of all audio hours, the model overfits to their vocal characteristics (pitch, prosody, formant structure). CTC will converge faster on familiar voices and underfit rare speakers — your competition PER tanks on the diverse test set.

**How it drives a decision:**

- If `top_k_speaker_hour_percent > 30%` → recommend weighted sampler

**Sampler weights by age bucket:** Inversely proportional to duration share, mean-normalised to 1.0. Overrepresented groups get weight < 1, underrepresented get > 1. The `unknown` bucket is capped at the max of the real age buckets so noisy metadata never dominates the sampler. This balances effective exposure without discarding data.

**Config:** `audio_eda.speaker.top_k`, `dominance_threshold`

---

### 4. Age Distribution

**What:** Hours and percentage per `age_bucket`.

**Why:** Child speech varies dramatically with age. Toddler formants are physically different from pre-teen formants (shorter vocal tract → higher F1/F2). If the model sees 40% 8-11yr but only 15% 3-4yr, it will underfit toddler phonetics.

**Output:** `hours_per_age_bucket` and `percent_per_age_bucket` — feeds directly into sampler weight computation.

---

### 5. Duration Profiling

**What:** Percentiles (p50, p90, p95, p96, p97, p98, p99, max) of utterance duration.

**Why:** Duration drives batch size. CTC requires padding all sequences in a batch to the longest one. If p95 is 6.5s but max is 60s, a single outlier forces massive padding waste. The theoretical batch size at p95 tells the trainer what to expect. The p96–p99 tail shows how steeply the distribution rises in the last 5% — a steep rise means a small number of long files will dominate padding cost.

**How it drives a decision:**

- `theoretical_batch_size_at_p95 = target_vram_seconds / dur_p95`
- Also reports `p50_audio_seconds_per_utterance` for throughput estimation
- `n_duration_manifest_mismatches` — count of files where `|manifest_dur - actual_dur| > 0.1s`, catching corrupt manifests
- `n_above_p99` / `percent_above_p99` — tail count for research inspection (not filtered)
- **VRAM planning uses actual decoded duration** (`n_samples / sr`), not manifest duration — ensures batch size estimates are based on what the GPU will actually see

**Config:** `audio_eda.duration.target_vram_seconds`

---

### 6. Silence Detection

**What:** Leading/trailing silence ratio: `rms(first/last N sec) / rms(full file)`.

**Why:** Many recordings start with a "click" followed by silence, or end with ambient noise after the child stops speaking. This wasted audio occupies sequence positions that CTC fills with blank tokens — the model learns to predict blanks for silence instead of learning phoneme timing.

**How it drives a decision:**

- A file has "significant silence" if lead or trail ratio < 0.1 (window is <10% of overall energy)
- If >20% of files have significant silence → recommend VAD trimming

**Config:** `audio_eda.silence.window_sec`, `ratio_threshold`, `corpus_percent_threshold`

**Direction:** Low ratio = quiet window relative to the file's overall energy. This is correct — a ratio of 0.01 means the lead/trail is 100x quieter than the average.

---

### 7. Spectral Centroid (Advisory)

**What:** Mean spectral centroid across a random subset of files.

**Why:** Spectral centroid is a proxy for the "brightness" of the audio. A corpus-wide mean tells you whether the data skews toward high-frequency content (telephone speech, bright mic) or low-frequency (muffled recordings). Advisory only — doesn't drive an automated decision.

**How:** STFT (1024 FFT, 512 hop, Hann window) → per-frame `sum(freq * mag) / sum(mag)` → averaged across valid frames → averaged across subset. The Hann window is **cached in worker init** (created once per process via `_worker_init`) and moved to `wav.device` at use time — avoids re-allocation per file and is device-safe for future GPU migration. The frequency axis tensor is also created on `wav.device` for consistency.

**Subset selection:** Fraction of corpus (default 5%), selected via deterministic `random.Random(seed=42)`. Avoids computing STFT on 150K files while providing a stable estimate.

**Config:** `audio_eda.spectral.subset_fraction`, `seed`

---

### 8. Format Check (Sample Rate & Channels)

**What:** Count files not at expected sample rate (16kHz) or expected channels (mono).

**Why:** WavLM expects 16kHz mono input. Files at 44.1kHz or stereo need preprocessing before training. This check quantifies the preprocessing burden — if 0% need resampling, you skip that pipeline step entirely.

**How it drives a decision:**

- Any file not 16kHz → `require_resampling = True`
- Any file not mono → `require_mono_downmix = True`

**Config:** `audio_eda.format.expected_sr`, `expected_channels`

---

### 9. Duration Mismatch Tracking

**What:** For every file, computes `actual_dur = n_samples / sr` and compares to `manifest_dur`. Flags when `|actual - manifest| > 0.1s`.

**Why:** Corrupt or truncated files can have a manifest that claims 5.0s but the actual waveform is 2.3s. If you use manifest duration for batch padding and the actual data is shorter, you get silent padding that confuses the model. If you use manifest duration for statistics and it's wrong, your p95 is unreliable.

**Tolerance:** 0.1s accounts for normal encoding/decoding rounding at the codec level.

**Output:** `n_duration_manifest_mismatches` in `duration_profile`. All `utterance_id`s with a mismatch are collected and saved to `data/reports/eda_duration_mismatch.json`. Their rows are then **removed from the processed transcripts** (see §10).

---

### 10. Audio Load Failures

**What:** Every file that raises an exception during `torchaudio.load()` returns the sentinel `_ERR_LOAD`. The corresponding `utterance_id` is recorded.

**Why:** A load failure means the file is unreadable — corrupt FLAC, missing codec, or filesystem error. Including it in training would cause a crash or silent skip depending on the dataloader. Better to know now and remove explicitly.

**Output:** All failed `utterance_id`s are saved to `data/reports/eda_failed_loads.json`. Their rows are then **removed from the processed transcripts** (see §11).

---

### 11. Transcript Cleanup (Post-EDA)

**What:** After the pool completes, the union of load-failure IDs and duration-mismatch IDs is removed from all processed JSONL files (`data/processed/*_transcript.jsonl`) in-place.

**Why:** Downstream training and splitting stages read the processed transcripts directly. Any reference to an unloadable or corrupt file must be gone before those stages run — otherwise training crashes or silently skips rows. **Note:** The tokenizer separately reads the EDA removal reports and raw transcripts to include the removed labels' phonemes in the vocab (see `tokenizer.md`), ensuring no coverage gap from audio-broken rows.

**How:** Each transcript file is streamed line-by-line. Lines whose `utterance_id` is in `bad_ids` are dropped; the rest are written back. Each removed row is logged individually with its reason (`load_failure` or `duration_mismatch`).

**Idempotent:** `bad_ids` is recomputed from the pool results on every run. If the bad rows were already removed on a previous run, the pool won't encounter those files again, so `bad_ids` will be empty and nothing is written.

---

### 12. Transcript Status

**What:** After cleanup, every processed transcript file is re-read and its final row count and total hours are logged.

**Why:** Closes the loop — confirms exactly how many rows are in the files that every downstream stage will consume. One ground-truth number per file, printed once.

**Example output:**

```
--- TRANSCRIPT STATUS ---
[TRANSCRIPT] 1_transcript.jsonl              11,918 rows      6.29 hrs
[TRANSCRIPT] 2_transcript.jsonl             139,787 rows     76.77 hrs
[TRANSCRIPT] ALL                            151,705 rows     83.06 hrs
--- TRANSCRIPT STATUS END ---
```

---

## Output: `eda_controls.json`

Top-level structure:

```json
{
  "total_audio_hours_analysed": 83.056,
  "rms": { "p01": ..., "p50": ..., "p99": ..., "spread_ratio": ..., "recommend_normalization": false },
  "clipping": { "n_files_heavily_clipped": ..., "recommend_drop_clipped": false },
  "speaker_dominance": { "n_speakers": ..., "top_speakers": [...], "sampler_weights_by_age_bucket": {...} },
  "age_distribution": { "hours_per_age_bucket": {...}, "percent_per_age_bucket": {...} },
  "duration_profile": { "p50": ..., "p90": ..., "p95": ..., "p96": ..., "p97": ..., "p98": ..., "p99": ..., "max": ...,
                        "n_above_p99": ..., "percent_above_p99": ...,
                        "theoretical_batch_size_at_p95": ..., "n_duration_manifest_mismatches": 0 },
  "silence": { "n_lead_silent": ..., "recommend_vad_trimming": false },
  "spectral_centroid_mean": 2847.3,
  "sample_rate_check": { "require_resampling": false, "require_mono_downmix": false },
  "decisions": {
    "apply_rms_normalization": false,
    "drop_heavily_clipped": false,
    "use_weighted_sampler": true,
    "apply_vad_trimming": false,
    "require_resampling": false,
    "require_mono_downmix": false
  }
}
```

**Source of truth:** `total_audio_hours_analysed` is computed from `sum(acc.durations) / 3600` (actual decoded durations), not from speaker hours. This guarantees correct totals even if some files have missing speaker metadata.

---

## Percentile Implementation

`eda_processor.py`, `data_split.py`, and `tokenizer.py` all import `nearest_rank_pctl()` from `src/utils/utils.py` — single source of truth:

```python
# src/utils/utils.py
def nearest_rank_pctl(
    vals: list[float], p: float, *, decimals: int = 6, presorted: bool = False
) -> float:
    sv = vals if presorted else sorted(vals)
    n = len(sv)
    idx = min(round((n - 1) * p / 100), n - 1)
    return round(sv[idx], decimals)
```

`sampler_weights_from_hours()` is also in `src/utils/utils.py` — shared between EDA and split.

This is unbiased for production-scale lists (150K+ elements). Python's banker's rounding causes ±1 index deviation only on lists shorter than ~10 elements — irrelevant at corpus scale.

The `SUPRASEGMENTALS` constant (currently `frozenset({"ː"})`) is also defined in `src/utils/utils.py` and imported by both modules — ensures TPS/PPS/SPS splits are computed identically everywhere.

---

## Multiprocessing Portability

All worker objects are module-level functions and primitive-only dataclasses → pickle-safe. Config is passed via `initializer`/`initargs` (set once per process), not per-task serialization. The entry script (`pipeline.py`) has `if __name__ == "__main__":` — safe for RunPod, Lambda, and any `fork`/`spawn` backend.

---

## Diagnostic Outputs

In addition to `eda_controls.json`, two research-audit files are written to `data/reports/` if any issues were found:

| File | Contents |
|------|----------|
| `eda_failed_loads.json` | `utterance_id`s that failed `torchaudio.load()` |
| `eda_duration_mismatch.json` | `utterance_id`s where `\|manifest_dur - actual_dur\| > 0.1s` |

Both files have the format `{"utterance_ids": [...], "count": N}`. The rows they reference are removed from the processed transcripts immediately after the pool finishes.

---

## Next Step

`eda_controls.json` is the corpus-wide design-decision record. After `--split`, train-only statistics are written to `training_controls.json` (duration profile, sampler weights, phoneme frequency, speaker stats) — that file drives the trainer. `data_fingerprint.json` provides SHA-256 hashes for paper-grade reproducibility.

---

## Decision Evolution — Mar 2026 Update

The original logic remains, but current control outputs changed as data cleanup matured.

| Earlier expectation | Current value (`data/reports/eda_controls.json`) | Why it changed |
|---|---|---|
| Duration tail discussed around ~12s+ context | `p99=10.847`, `max=14.99`, `batch@p95=19` | Final processed manifests are cleaner (post-EDA anomaly removals), reducing tail heaviness.
| Weighted sampler might be needed for speaker dominance | `top5_speaker_hour_percent=4.84`, `use_weighted_sampler=false` | Real dominance is low; weighted sampler would add complexity with little gain.
| RMS normalisation as a conditional recommendation | `spread_ratio=124.89`, `apply_rms_normalization=true` | The corpus loudness spread is very high; normalization is now a hard practical requirement.
| VAD trimming considered | `percent_high_silence_files=4.61`, `apply_vad_trimming=false` | Silence prevalence is below threshold; trimming would add risk/complexity for small payoff.

**Current rationale:** we kept the decision framework, but moved from assumptions to measured controls that now directly justify each training-time toggle.
