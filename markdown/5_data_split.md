# DataSplitter — Deterministic, CTC-Safe, Speaker-Disjoint Train/Val Split (Stage 4)

> **Note:** Numbers below reflect an earlier pipeline run. The final Run 24 split: train 120,699 (869 speakers, 69.1 hrs) / val 13,718 (134 speakers, 7.6 hrs), seed 1507. See [README.md](../README.md). Run 24 artifacts: [data_fingerprint.json](../study/runs/run_24/reports/data_fingerprint.json), [training_controls.json](../study/runs/run_24/reports/training_controls.json).

> **File:** `src/etl/data_split.py`
> **Runs via:** `python pipeline.py --etl` (fifth and final ETL stage)
> **Prerequisite:** Sanity + Audio EDA + Tokenizer must have run first (reads from `data/processed/`)
> **Input:** Cleaned JSONL (`data/processed/*_transcript.jsonl`)
> **Output:** `data/processed/sft_train.jsonl`, `sft_val.jsonl` + `data/ssl/processed/ssl_train.jsonl` + `data/reports/training_controls.json`, `data_fingerprint.json`
> **Config:** `config.yaml` → `split:` section
> **Datasets:** Derived from `cfg["datasets"]` — same canonical list as all other stages

---

## Rationale — Why This Stage Exists

A naive random split on a children's speech dataset is **guaranteed to leak data**. The same child appears in multiple utterances — if some go to train and others to val, the model memorises that child's voice (pitch, prosody, formant structure) and reports inflated PER that collapses on unseen speakers.

Beyond speaker leakage, CTC has a unique failure mode: if any phoneme character exists only in val and not in train, the model's output head has zero training signal for that logit. CTC loss for utterances containing that phoneme produces garbage gradients. This is unrecoverable.

The DataSplitter solves both problems simultaneously:

- **Speaker-disjoint split** → no voice leakage
- **CTC phoneme coverage guarantee** → every character in the vocab appears in train
- **Age × dataset stratification** → val represents the same demographic mix as the corpus
- **Duration-targeted sizing** → val is sized by hours (not speaker count) for stable epoch-level metrics
- **Drill quarantine** → repetitive exercises excluded from val to prevent inflated metrics
- **Rare-phoneme pinning** → speakers carrying rare phonemes forced to train before any split attempt

It also produces two critical outputs for training:

1. **`training_controls.json`** — train-only statistics (duration profile, sampler weights, phoneme frequency, speaker stats) that the trainer consumes directly. No more guessing batch sizes or sampler configs.
2. **`data_fingerprint.json`** — SHA-256 hashes of all split files + resolved config. If any bit changes upstream, the hash changes. Paper-grade reproducibility.

## How It Helps the Overall Pipeline

| Downstream stage | What DataSplitter provides |
|-----------------|----------------------------|
| SSL Pre-training | `ssl_train.jsonl` — all 151K+ rows (train + val), audio paths only, no labels. Used for domain-adaptive continued pre-training on children's audio |
| SFT Training | `sft_train.jsonl` — labeled train rows with `dataset` field for domain-aware batching |
| SFT Validation | `sft_val.jsonl` — speaker-disjoint, drill-free, age-stratified val set |
| Trainer config | `training_controls.json` → `recommended_max_duration_sec` (collator), `theoretical_batch_size_at_p95` (batch size), `sampler_weights_by_age_bucket` (weighted sampler), `speaker_epoch_cap_hours` (speaker balancing) |
| Reproducibility | `data_fingerprint.json` → SHA-256 hashes of train/val/vocab/config. Deterministic with `seed=42` |

---

## What It Does

Two-pass architecture: speaker stats in pass 1 (RAM ∝ speakers, ~1 k); pass 2 materialises all SFT row dicts in memory (RAM ∝ rows, ~150 k dicts) for CTC monitoring + train controls; SSL manifest streams directly to disk.

After splitting: writes SFT + SSL manifests, computes train-only controls (`training_controls.json`), and produces a SHA-256 fingerprint (`data_fingerprint.json`) for paper-grade reproducibility.

**Current numbers (Feb 2026):**

| Metric | Value |
|--------|-------|
| Source rows | 151,705 |
| Total hours | 83.06 |
| Valid speakers | 1,003 |
| Unknown-speaker rows | 0 |
| Rare-phoneme speakers pinned | 10 |
| Train speakers | 892 |
| Val speakers | 111 |
| Train rows | 136,080 |
| Val rows | 15,625 |
| Train hours | 74.77 |
| Target val hours | 8.31 |
| Achieved val hours | 8.29 (Δ −0.02 h) |
| Drills excluded from val | 12 |
| Phoneme coverage | FULL (50/50) |
| Rarest phoneme in train | `χ` (count=18) |
| Retries needed | 0 |

---

## Hard Guarantees

### 1. Speaker-Disjoint

Split is by `child_id`, not by row. All rows of a speaker go to the same split. This prevents data leakage — if a child appears in both train and val, the model memorises that child's voice rather than learning phoneme patterns.

**Exception:** Val speakers who have drill rows — those drill rows are moved to train (see §3). The speaker's non-drill rows remain in val. This is correct: the drill utterances are repetitive exercises, not representative of the child's natural speech.

### 2. Unknown Speaker Quarantine

Any row where `child_id` is null, empty, or literal `"unknown"` is forced to train, never val. You can't guarantee speaker disjointness for rows with no speaker ID.

### 3. Drill Handling

Rows with `is_drill == True` are excluded from val and moved to train. Drills are diadochokinetic exercises ("ba ba ba ba") — they inflate val metrics because the model only needs to learn one phoneme pattern repeated. Keeping them in train is fine (real phonemes, just repetitive).

### 4. Age × Dataset Stratification

Val age distribution ≈ global age distribution. Speakers are grouped by `(age_bucket, dataset)`, and val speakers are sampled proportionally per bucket. This prevents age and domain bias — without stratification, val could accidentally be all toddlers from one dataset.

### 5. Rare-Phoneme Protection

Before any split attempt, speakers whose phoneme inventory contains corpus-rare characters (count < `max(10, total_tokens × rare_phoneme_fraction)`) are pinned to train. This guarantees CTC coverage without retries in most cases. Currently 10 speakers are pinned (threshold < 45 / 1,492,975 tokens).

**Note:** Pinning removes these speakers from the stratification pool, causing a minor age×dataset skew toward train. With ~10 out of ~1,000 speakers the effect is < 1 % — acceptable trade-off for guaranteed CTC coverage.

### 6. Duration-Targeted Val

Val is sized by hours, not by speaker count. Each `(age_bucket, dataset)` cell fills its proportional hours share using a per-bucket duration accumulator (add-only-if-fits). A global ascending-duration top-up fills the residual. Guaranteed overshoot ≈ 0 (current: Δ −0.02 h).

### 7. CTC Phoneme Coverage Retry Loop

After each split attempt, the train phoneme inventory is checked against the full corpus inventory. If any phoneme exists only in val (not in train), the model can never learn to predict it — CTC loss for that phoneme is undefined. The splitter reshuffles with `seed + attempt` and retries up to `max_retries` times. If still failing, `RuntimeError` with diagnostics.

### 8. Minimum Phoneme Count Warning

After a successful split, every phoneme's count in train is checked against `min_phoneme_count_warn`. Below-threshold phonemes are logged as warnings — they're in train but may not have enough examples for the model to learn them reliably.

### 9. Determinism

All randomness comes from `random.Random(seed)`. Each retry uses `seed + attempt`. Same config → same split, every time, every machine.

---

## Step-by-Step: What, Why, How

### 1. Load Manifests

**What:** Reads all `*_transcript.jsonl` files from `data/processed/`, sorted by dataset key.

**Why:** Sorted input order guarantees deterministic row ordering regardless of filesystem quirks.

**How:** Streams each JSONL, collects rows in memory (needed for speaker grouping), and tracks `utterance_id → dataset_key` mapping for the `"dataset"` field injection.

---

### 2. Group Speakers

**What:** Groups rows by `child_id` into `_Speaker` dataclasses. Rows with unknown/null child_id are separated into `unknown_rows`.

**Why:** Speaker-level split requires knowing which rows belong to which speaker. The `_Speaker` dataclass pre-computes per-speaker phoneme inventory (as a `Counter`), total duration, row count, and whether any drill rows exist.

**How:** Single pass over all rows. `_is_unknown_speaker()` checks for null, empty, or literal "unknown" (case-insensitive).

---

### 3. Age×Dataset Stratified Split

**What:** Groups valid speakers (minus rare-phoneme-pinned) by `(age_bucket, dataset)`, computes proportional val targets per cell using a duration accumulator, fills each cell with speakers in shuffled order (add-only-if-fits), then runs a global ascending-duration top-up to fill the residual.

**Why:** Proportional duration-targeted sampling ensures val represents the same age×domain mix and the same hours ratio as the corpus. The alternative — speaker-count allocation — produces unpredictable val hours because speaker durations vary widely.

**How:** `_split_once_duration()` uses `random.Random(seed)`. Each `(age, dataset)` cell gets `proportional_share × target_val_hrs`. Speakers are added only if adding them keeps cell hours ≤ target. After all cells are filled, remaining deficit is filled by scanning leftover speakers in ascending-duration order.

---

### 4. Phoneme Coverage Check

**What:** After splitting, verifies `train_inventory ⊇ full_inventory`.

**Why:** CTC requires every label character to appear in training data. A phoneme that only exists in val means the model has no training signal for that logit — during training, the CTC loss for utterances containing that phoneme produces garbage gradients.

**How:** `_speaker_inventory()` unions all `phoneme_counter` keys from train speakers + unknown rows. Compared against `_corpus_inventory()` (all rows). Missing phonemes trigger a retry.

---

### 5. Row Partitioning

**What:** After speaker assignment, iterates all rows and routes them to train or val. Unknown rows → train. Val speakers' drill rows → train. Everything else follows speaker assignment.

**Why:** Separating the split decision (speaker-level) from the row routing (row-level with drill override) keeps the logic clean and auditable.

---

### 6. Output Writing

**What:** Writes 3 JSONL files + 2 JSON reports:

| File | Contents |
|------|----------|
| `sft_train.jsonl` | All train rows + `"dataset"` field |
| `sft_val.jsonl` | All val rows + `"dataset"` field |
| `ssl_train.jsonl` | All rows, minimal fields: `utterance_id`, `audio_path`, `dataset` |
| `training_controls.json` | Train-only statistics: duration profile, sampler weights, phoneme frequency, speaker stats |
| `data_fingerprint.json` | SHA-256 hashes of split files + vocab + tokenizer config, resolved config snapshot |

**Why:** SFT manifests drive supervised fine-tuning (Wav2Vec2 CTC). SSL manifest drives self-supervised pre-training (wav2vec2 base) where labels aren't used — only audio paths matter. `training_controls.json` gives the trainer everything it needs to configure batching, sampling, and curriculum. `data_fingerprint.json` provides paper-grade reproducibility.

---

## Config

```yaml
split:
  val_ratio: 0.10                 # fraction of total hours → val (duration-targeted)
  test_ratio: 0.0                 # fraction of speakers → test (0 = no test file)
  seed: 42                        # base seed for deterministic split
  max_retries: 50                 # CTC phoneme-coverage retry limit
  min_phoneme_count_warn: 5       # warn if any phoneme < this in train
  rare_phoneme_fraction: 3.0e-5   # speakers with a rare-phoneme are pinned to train;
                                  # rare = count < max(10, total_tokens * this fraction)
```

---

## Output Logging

```
--- SPLIT START ---
[SPLIT] total rows               151,705
[SPLIT] total hours                83.06
[SPLIT] valid speakers             1,003
[SPLIT] unknown-speaker rows           0  (forced to train)
[SPLIT] corpus inventory              50 chars
[SPLIT] rare-phoneme spkrs → train    10  (threshold < 45 / 1,492,975 tokens)
[SPLIT] target val hours            8.31
[SPLIT] test target                    0 speakers  (0% of 1003)
[SPLIT] retries used                   0
[SPLIT] phoneme coverage            FULL
[SPLIT] stratification key    age × dataset
[SPLIT] train speakers               892
[SPLIT] val   speakers               111
[SPLIT] train rows               136,080
[SPLIT] val   rows                15,625
[SPLIT] train hours                74.77
[SPLIT] target val hours            8.31
[SPLIT] achieved val hours          8.29  (10.0%  Δ -0.02 h)
[SPLIT] drills excl. from val         12
[SPLIT] train  12+           6.28 hrs  (  8.4%)
[SPLIT] train  3-4          22.89 hrs  ( 30.6%)
[SPLIT] train  5-7          22.39 hrs  ( 29.9%)
[SPLIT] train  8-11         20.14 hrs  ( 26.9%)
[SPLIT] train  unknown       3.07 hrs  (  4.1%)
[SPLIT] val    12+           0.66 hrs  (  7.9%)
[SPLIT] val    3-4           2.82 hrs  ( 34.0%)
[SPLIT] val    5-7           2.23 hrs  ( 26.9%)
[SPLIT] val    8-11          2.24 hrs  ( 27.0%)
[SPLIT] val    unknown       0.34 hrs  (  4.2%)
[SPLIT] min phoneme count     all ≥ 5
[SPLIT] rarest in train       'χ' count=18
[SPLIT] wrote  sft_train.jsonl               136,080 rows
[SPLIT] wrote  sft_val.jsonl                  15,625 rows
[SPLIT] wrote  ssl_train.jsonl               151,705 rows
[SPLIT] train_sha256          4e97049...c941d6
[SPLIT] val_sha256            40ca2ed...5ae39
[SPLIT] wrote  data_fingerprint.json
[SPLIT] train controls  hours=74.77  dur_p95=6.500s  batch@p95=18  phoneme_tokens=1,345,226
[SPLIT] tc  12+           6.28 h  (  8.4%)  weight=1.7441
[SPLIT] tc  3-4          22.89 h  ( 30.6%)  weight=0.4786
[SPLIT] tc  5-7          22.39 h  ( 29.9%)  weight=0.4893
[SPLIT] tc  8-11         20.14 h  ( 26.9%)  weight=0.5438
[SPLIT] tc  unknown       3.07 h  (  4.1%)  weight=1.7441
[SPLIT] wrote  training_controls.json
--- SPLIT END ---
```

---

## Output: `training_controls.json`

Train-only statistics derived from `sft_train.jsonl` rows already in memory — no extra I/O pass. The trainer consumes this directly; the corpus-wide `eda_controls.json` from Audio EDA stays as the design-decision record.

```json
{
  "scope": "train_only",
  "source": "sft_train.jsonl",
  "n_rows": 136080,
  "total_hours": 74.769,
  "duration_profile": {
    "p50": 1.119, "p90": 4.416, "p95": 6.5,
    "p96": 7.22, "p97": 8.197, "p98": 9.739, "p99": 12.574,
    "max": 24.903,
    "n_rows_above_p99": 1361,
    "recommended_max_duration_sec": 12.574,
    "target_vram_seconds": 120.0,
    "theoretical_batch_size_at_p95": 18,
    "batch_size_note": "Upper bound — ignores feature-extraction latency, ...",
    "collator_policy_note": "Set collator max_length from recommended_max_duration_sec. ..."
  },
  "age_distribution": {
    "hours_per_age_bucket": { "12+": 6.281, "3-4": 22.888, "5-7": 22.389, "8-11": 20.143, "unknown": 3.069 },
    "percent_per_age_bucket": { "12+": 8.4, "3-4": 30.61, "5-7": 29.94, "8-11": 26.94, "unknown": 4.1 },
    "sampler_weights_by_age_bucket": { "12+": 1.7441, "3-4": 0.4786, "5-7": 0.4893, "8-11": 0.5438, "unknown": 1.7441 }
  },
  "phoneme_frequency": { "...": { "count": "...", "percent": "..." } },
  "speaker_stats": {
    "n_speakers": 899,
    "total_hours": 74.769,
    "speaker_hours_p50": 0.045,
    "speaker_hours_p95": 0.3,
    "speaker_hours_p99": 0.492,
    "speaker_hours_max": 0.899,
    "speaker_hours_skew_ratio": 19.88,
    "recommend_speaker_balanced_sampling": true,
    "speaker_epoch_cap_hours": 0.3,
    "speaker_epoch_cap_note": "If recommend_speaker_balanced_sampling is true, limit each speaker to speaker_epoch_cap_hours per epoch."
  }
}
```

**Key design decisions:**

- **Unknown weight capped:** `sampler_weights_from_hours()` in `utils.py` caps the `unknown` bucket at the max of the real age buckets, so noisy metadata never dominates the sampler. Currently `unknown = 1.7441` (capped at `12+`), down from 2.615 uncapped.
- **Speaker skew acted on:** `speaker_hours_skew_ratio` = max/p50 = 19.88. Because this exceeds 10×, `recommend_speaker_balanced_sampling` is `true`. The `speaker_epoch_cap_hours` (= p95 = 0.30 h) tells the trainer how to cap per-speaker contribution.
- **Duration enforcement deferred to collator:** `recommended_max_duration_sec` (= p99 = 12.574 s) and `n_rows_above_p99` (1,361) give the collator what it needs. The collator now uses `sft.max_duration` (= `eda.max_duration` = 15 s) from `config.yaml` as its safety threshold — **not** the p99 recommendation. This keeps all ≤15 s utterances in the dataset. The p99/n_rows_above_p99 values remain useful for batch-size planning but no longer drive filtering. Any sample exceeding the 15 s cap is **dropped** (never truncated) with a warning log. OOM from occasional long batches is caught by a `try/except RuntimeError` in the training loop (see `sft.md`).

---

## Output: `data_fingerprint.json`

```json
{
  "train_sha256": "4e9704...",
  "val_sha256": "40ca2e...",
  "vocab_sha256": "8bb369...",
  "tokenizer_config_sha256": "ff5797...",
  "datasets": [1, 2],
  "config": {
    "val_ratio": 0.1,
    "test_ratio": 0.0,
    "seed": 42,
    "max_retries": 50,
    "rare_phoneme_fraction": 3e-05,
    "rare_threshold_resolved": 45
  },
  "split_summary": {
    "n_train_rows": 136080,
    "n_val_rows": 15625,
    "n_test_rows": 0,
    "n_train_speakers": 892,
    "n_val_speakers": 111,
    "train_hrs": 74.7695,
    "val_hrs": 8.2904,
    "target_val_hrs": 8.306,
    "retry_count": 0,
    "phoneme_coverage": true
  }
}
```

---

## Edge Cases Handled

| Scenario | Behaviour |
|----------|-----------|
| Single-speaker age bucket | Clamped: speaker stays in train, 0 val from that bucket |
| All rare phonemes in one speaker | Retry loop moves that speaker to train |
| Empty val after drill removal | Would trigger retry (all val rows were drills) |
| Unknown speakers | Quarantined to train, never split |
| No unknown speakers | 0 logged, no special handling needed |
| Unicode phonemes | No ASCII assumptions — `set()` operations on actual characters |

---

## Return Value

```python
{
    "n_train_rows":     136_080,
    "n_val_rows":       15_625,
    "n_test_rows":      0,
    "n_train_speakers": 892,
    "n_val_speakers":   111,
    "n_test_speakers":  0,
    "train_hrs":        74.7695,
    "val_hrs":          8.2904,
    "target_val_hrs":   8.306,
    "retry_count":      0,
    "phoneme_coverage": True,
    "train_sha256":     "4e9704...",
    "val_sha256":       "40ca2e...",
}
```

---

## Decision Evolution — Mar 2026 Update

Original split logic is still the same; output numbers changed with refreshed processed manifests.

| Earlier baseline in this doc | Current files now | Why this moved |
|---|---|---|
| Total rows ~151,705 | Current processed total: `150,928` | Upstream EDA removed 4 additional bad-audio rows after initial sanity counts.
| Train/val rows `136,080 / 15,625` | Current split: `135,247 / 15,681` (`training_controls.json` + manifests) | Re-splitting after refreshed manifests changed bucket-level allocation while preserving policy constraints.
| Train hours ~74.77 / val ~8.29 | Current train hours `71.315`, val hours `7.909` | Total corpus hours dropped after stricter cleaned input; val target follows ratio policy.
| Batch@p95 examples around 18 in older run | Current `theoretical_batch_size_at_p95=19` | Updated duration profile (`p95=6.208`) yields slightly higher theoretical batch.

**Current rationale:** we did not alter split safeguards (speaker disjointness, rare-phoneme protection, drill handling); only the underlying cleaned corpus changed, so outputs were recomputed accordingly.

| `recommended_max_duration_sec` (12.574 s) used as collator ceiling | Collator now uses `sft.max_duration` (15 s) from `config.yaml`, asserted equal to `eda.max_duration` | All ≤15 s utterances kept; p99 recommendation retained for batch-size planning only. OOM from rare long batches caught by `try/except` in training loop. No truncation — only drop + log. |
