# SFT Trainer — WavLM CTC Fine-Tuning via HuggingFace Trainer

> **Note:** The definitive Run 24 config and training log are in [study/runs/run_24/](../study/runs/run_24/) — see [training_config.json](../study/runs/run_24/training_config.json) and [training log](../study/runs/run_24/logs/309_log_20260329_190622.log).

> **Package:** `src/trainer/` (7 modules) + `src/utils/` (3 modules) — ~5,200 lines total
> **Entry point:** `HFSFTTrainer(cfg).train()` or `HFSFTTrainer(cfg).avg_only(top_n=5)`
> **Prerequisite:** Full ETL pipeline (Sanity → Audio EDA → Tokenizer → Data Split) must have run
> **Model:** WavLM Base+ (`microsoft/wavlm-base-plus`, 94.7M params) → `WavLMForCTC`
> **Output:** Top-K checkpoints + `best/` + optional `avg/` in `data/models/hf_sft_checkpoints/`
> **Config:** `config.yaml` → `hf_sft:` section

---

## Architecture

The trainer wraps HuggingFace `Trainer` via a two-class pattern:

```
HFSFTTrainer            — public API: build model, datasets, collators from config
  └── _LRGroupedCTCTrainer(_CTCTrainer(Trainer))
        ├── create_optimizer()     — LLRD-aware AdamW with per-layer groups
        ├── create_scheduler()     — cosine decay with LR floor
        ├── compute_loss()         — CTC + optional age weights + SR-CTC
        ├── training_step()        — OOM recovery guard
        ├── prediction_step()      — buffers output lengths + metadata
        ├── get_train_dataloader() — LengthGroupedSampler + SFTCollator
        └── get_eval_dataloader()  — length-grouped eval with val collator
```

**Why HF Trainer, not a custom loop:** The original codebase had a 2,370-line custom training loop (`sft_trainer.py`) with 3-stage signal-driven unfreezing, EMA signal trackers, per-group warmup ramps, and manual AMP/GradScaler management. None of this produced measurable PER improvement over simply unfreezing everything from step 0 with LLRD. The HF Trainer wrapper matched the old loop's results at ~1,900 lines with far less complexity and faster iteration.

**What was kept from the old loop:** LLRD, discriminative LR, SR-CTC, top-K checkpointing by PER, early stopping, OOM recovery, length-grouped sampling. **What was dropped:** 3-stage transitions, EMA model weights, age-weighted CTC loss, manual AMP, custom W&B tracking.

---

## How It Fits the Pipeline

| Upstream | What SFT consumes |
|----------|--------------------|
| Sanity | Clean, NFC-normalised phonetic labels |
| Audio EDA | Removal of broken audio (load failures, duration mismatches) |
| Tokenizer | `vocab.json` mapping 50 IPA phonemes + 3 specials → IDs 0–52 |
| Data Split | `sft_train.jsonl` (136K rows, 74.8h) + `sft_val.jsonl` (15.6K rows, 8.3h) |

| SFT produces | Used by |
|---------------|---------|
| Best checkpoint (`.safetensors`) | Submission inference |
| Averaged checkpoint (`avg/`) | Alternative submission candidate |
| Eval metrics (PER, CER, per-age, per-dataset) | Model selection |

---

## Package Structure

```
src/trainer/
├── __init__.py            (82 lines)   Public API + re-exports
├── sft_trainer_hf.py    (1922 lines)   HF Trainer wrapper — full training orchestration
├── model.py              (577 lines)   Model build, staged freeze, LLRD param groups
├── data_collator.py      (725 lines)   Audio loading, preprocessing, augmentation, padding
├── dataset.py            (176 lines)   JSONL manifest reader
├── metrics.py            (598 lines)   CTC decode, PER, CER, phoneme recall, error analysis
└── email_callback.py     (256 lines)   Email notification on completion/failure

src/utils/
├── __init__.py            (14 lines)   Re-exports
├── utils.py              (249 lines)   JSON, audio path resolution, mono loader
├── tracking.py           (202 lines)   WandbTracker — background daemon with bounded queue
└── bench_dataloader.py   (393 lines)   DataLoader throughput benchmark
```

---

## Module-by-Module

### 1. `dataset.py` — JSONL Manifest Reader

Map-style `torch.utils.data.Dataset` that loads a JSONL manifest into memory. Returns one row-dict per `__getitem__`. Supports optional dataset-level oversampling via `ds_oversample` (e.g., `{1: 3}` repeats DS1 rows 3×).

**Audio is NOT loaded here.** The dataset stores only metadata (paths, durations, age buckets, text). Waveforms are loaded by the collator at batch time — keeping RAM at ~50 MB instead of ~180 GB. The `.input_lengths` attribute (list of durations) is exposed for `LengthGroupedSampler`.

---

### 2. `data_collator.py` — Audio Loading & Preprocessing

Takes a list of manifest-row dicts, produces a padded batch dict.

#### Per-Sample Pipeline

| Step | What | Why |
|------|------|-----|
| `torchaudio.load` | Load raw audio | Defers I/O to batch time |
| Mono downmix | `mean(dim=0)` if multi-channel | WavLM expects mono |
| Resample to 16 kHz | Cached `Resample` objects per source SR | Kernel computed once, reused ~100K times/epoch |
| Optional silence trim | Trim leading/trailing silence at -40 dB | Removes dead signal; matches submission preprocessing |
| RMS normalisation | Scale to `target_rms` (0.05) | Corpus has 125× RMS spread |
| Min-duration floor pad | Zero-pad to 1s if shorter | Guarantees ≥50 WavLM output frames for CTC |
| Optional speed perturb | Resample at [0.9, 1.1]× | Data augmentation, cached per-speed resampler |
| Optional noise/RIR | MUSAN additive / RIR convolution | Environmental augmentation |
| Optional pitch shift | Semitone-based pitch perturbation | Per-dataset configurable |
| Tokenise | `tokenizer(phonetic_text).input_ids` | IPA text → integer labels for CTC |

#### Batch-Level

| Step | What | Why |
|------|------|-----|
| Dynamic padding | Pad waveforms to max length in batch | Length-grouped sampling keeps waste ~5% |
| Attention mask (`long` dtype) | `1 = real, 0 = pad` | WavLM silently mishandles `bool` masks |
| Max-duration drop | Drop (never truncate) samples exceeding `max_duration_sec` | Secondary safety after ETL filtering |
| Label padding with `-100` | CTC ignores `-100` positions | PyTorch convention for ignore-index |

**What's NOT here:** SpecAugment. Applied inside `WavLMForCTC.forward()` during `model.train()` — fresh masks each forward pass.

---

### 3. `model.py` — Model Construction & Freeze Logic

#### `build_model(cfg)`

Loads WavLM Base+ with all config applied at construction (vocab_size=53, dropout rates, SpecAugment params). The CTC head (`lm_head`) is explicitly replaced with a fresh `nn.Linear(768, 53)` to guarantee random init regardless of the pretrained checkpoint's head shape.

Gradient checkpointing is enabled with `use_reentrant=False`. `torch.compile` is fully disabled (CTC variable-length inputs break inductor).

#### `freeze_for_stage(model, stage)`

Sets `requires_grad_(False)` on everything first, then selectively enables:

- **Stage 1:** `lm_head` only (40K params, 0.04%)
- **Stage 3:** Full model — all encoder layers + feature projection + CNN (optional) (94.7M, 100%)

Stage 2 (partial encoder) exists in code but is not used by the current trainer. The HF trainer uses either Stage 1 (during `warmup_head_only_steps`, if >0) or Stage 3 (everything else).

#### Parameter Groups & LLRD

**Discriminative LR:** Head gets `head_lr` (3e-4), encoder gets `encoder_lr` (1e-4), CNN gets `cnn_lr` (frozen by default).

**LLRD (Layer-wise LR Decay):** With `llrd_decay: 0.85`, each of the 12 encoder layers gets its own optimizer group: `lr(layer_i) = encoder_lr × 0.85^(11 - i)`. Top layers (task-specific) get full LR; bottom layers (general acoustics) get gently reduced LR.

```
enc_L11 → 1.0e-4  (full encoder_lr)
enc_L6  → 4.4e-5  (× 0.85^5)
enc_L0  → 1.7e-5  (× 0.85^11)
```

Each group is split into decay/no-decay sub-groups (LayerNorm + bias get `weight_decay=0.0`). All `requires_grad` parameters are verified to appear in exactly one group.

---

### 4. `metrics.py` — Evaluation Functions

All metrics are stateless pure functions.

| Metric | What | Why |
|--------|------|-----|
| **CTC greedy decode** | argmax → collapse repeats → remove blanks | Standard CTC post-processing |
| **PER** (corpus-level) | `total_edits / total_ref_tokens` | Micro-average — weights longer utterances proportionally |
| **CER** (via jiwer) | Character error rate on decoded IPA strings | Matches leaderboard metric |
| **Blank ratio** | Fraction of argmax frames = blank (ID 0) | Primary CTC health indicator (>0.95 = collapse) |
| **Mean run length** | Average consecutive identical-token run | Short = healthy alignment, long = stuck |
| **Per-phoneme recall** | Levenshtein alignment → per-token hits/total | Identifies which phonemes the model hasn't learned |
| **Error breakdown** | del / ins / sub counts + top confusion pairs | Guides training decisions (high del → SR-CTC, high sub → more data) |

---

### 5. `sft_trainer_hf.py` — Training Orchestration

#### CTC Loss (`compute_loss`)

Two paths:

- **Fast path** (no age weights, no SR-CTC): Delegates to model's built-in CTC — zero overhead.
- **Slow path** (age weights or SR-CTC active): Forward without labels, compute CTC manually in fp32 with per-sample length normalisation.

Length normalisation: `loss = loss / target_lengths.float().clamp(min=1)`. Without this, long utterances produce ~16× more gradient than short ones. PyTorch's `reduction="mean"` does this internally; we replicate it when using `reduction="none"` for per-sample weighting.

#### SR-CTC Regularisation (Yao et al., ICLR 2025)

Attacks blank-peaky CTC distributions. The model emits ~85% blank frames, causing deletion errors — especially on short/noisy DS1 (3–4 year olds).

1. Softmax probabilities from logits
2. 1D convolution with kernel `[0.25, 0.5, 0.25]` along time (replicate padding)
3. `KL(stop_grad(smoothed) || original)` — gradient flows through original only
4. Mask to real frames (exclude padding)
5. `loss = ctc_loss + β × sr_loss` (β = 0.1)

Stop-gradient on the smoothed target prevents collapse to uniform distribution.

#### Dynamic SpecAugment for DS1

DS1 is already noisy (~12 dB SNR). Full SpecAugment on top double-corrupts. `mask_time_prob` is interpolated based on DS1 fraction in the batch: all-DS2 → 0.30, all-DS1 → 0.15 (configurable via `ds1_mask_time_prob`).

#### LR Schedule

```
Linear warmup:  step 0 → warmup_steps     →  LR ramps 0 → base_lr
Cosine decay:   warmup → total_steps      →  lr_min + (base - lr_min)/2 × (1 + cos(π·progress))
```

Each LLRD group decays from its own `initial_lr`. The floor `lr_min_ratio` (0.10) prevents dead steps at end of training — earlier runs lost epochs when LR decayed to zero.

#### Optional Head Warmup

When `warmup_head_only_steps > 0`, the model starts in Stage 1 (head-only) and unfreezes to Stage 3 via `_HeadWarmupCallback.on_step_begin`. Currently disabled (`warmup_head_only_steps: 0`) — no measurable benefit over full unfreeze with LLRD.

#### OOM Recovery

`training_step()` catches `RuntimeError("out of memory")`, logs diagnostics (peak VRAM, batch shape), clears the failed computation graph (`zero_grad(set_to_none=True)` + `gc.collect` + `empty_cache`), and returns zero loss. Training continues on the next batch. Self-healing: same files appear in different batch compositions next epoch.

#### Evaluation

Full val pass computing: `val_loss`, `PER`, `CER`, `blank_ratio`, `mean_run_length`, per-age PER, per-dataset PER, age×dataset cross PER, per-age CER, length-bucketed PER, per-phoneme recall, worst-K phonemes, error breakdown (del/ins/sub), top confusion pairs, 5 random REF/HYP alignment samples.

UNK token suppressed before argmax (matching submission). IPA post-processing filter applied for CER computation (matching leaderboard).

#### Callbacks

| Callback | Purpose |
|----------|---------|
| `EarlyStoppingCallback` | Stop when `eval_per` doesn't improve for `patience` epochs |
| `_CTCConstraintCheck` | Preflight check: verify `output_len ≥ target_len` on first batch |
| `_HeadWarmupCallback` | Unfreeze encoder after `warmup_head_only_steps` |
| `_OverfittingAlertCallback` | Monitor train/eval divergence, DS1 regression, gap widening |
| `_StartupLoggingCallback` | Log phase transitions during silent DataLoader startup |
| `_VRAMCleanupCallback` | `gc.collect` + `empty_cache` at epoch boundaries + OOM count summary |
| `EmailNotificationCallback` | Email alerts on training completion/failure |

#### Checkpoint Averaging

Post-training: ranks all `checkpoint-*` directories by `eval_per` from `trainer_state.json`, loads top-N state dicts, averages weights in fp32, saves to `avg/` with metadata, evaluates, and reports delta vs best single checkpoint.

Also available standalone via `HFSFTTrainer(cfg).avg_only(top_n=5)` for mid-run or post-training averaging via `python pipeline.py --avg`.

---

## Checkpointing

HuggingFace Trainer handles save/load. Strategy: save every epoch, keep top-K by `eval_per`, load best model at end. Each checkpoint directory contains `model.safetensors`, `config.json`, `trainer_state.json`, `optimizer.pt`, `scheduler.pt`.

Resume via `hf_sft.resume_from: path/to/checkpoint-NNNN` in config.

---

## Key Design Decisions

### Why `zero_infinity=True` in CTC Loss
CTC loss is `-inf` when alignment is impossible. Without `zero_infinity`, one bad sample → `nan` loss → all params become `nan`. With it, impossible alignments contribute 0 and are counted as an instability indicator.

### Why fp32 Log-Softmax
CTC requires `log_softmax(logits)`. In fp16 with a small vocab (53), `exp(x - max)` rounds to 0 for many entries → `-inf` in the log. Explicit `.float().log_softmax()` prevents this.

### Why Dynamic Padding (Not Fixed Max Length)
With `LengthGroupedSampler`, batches contain similar-length utterances, so padding to `max(batch)` instead of `max(dataset)` saves ~30% compute on short-utterance batches.

### Why Label Padding = -100 (Not 0)
CTC blank = PAD = ID 0. Padding labels with 0 would inject false blank targets. `-100` is PyTorch's ignore-index convention.

### Why `attention_mask` dtype = `long`
WavLM silently misinterprets boolean masks. Must be `int64`. Discovered empirically, undocumented in HuggingFace.

### Why CNN Gets Lower LR (or is Frozen)
WavLM's CNN (7 conv layers, 4.7M params) was trained on 94K hours. Aggressive fine-tuning destroys learned spectral decomposition. Default: frozen (`unfreeze_cnn: false`).

### Why `torch.compile` is Disabled
CTC models produce variable-length outputs. torch.compile / inductor constantly recompiles on dynamic shapes and wastes ~11 GB RAM spawning 32 compile-worker threads. Disabled at module-import time to prevent even the thread pool spawning.

---

## Config Reference (`hf_sft:`)

| Parameter | Current Value | Purpose |
|-----------|---------------|---------|
| `model_name` | `microsoft/wavlm-base-plus` | Pretrained model |
| `vocab_size` | 53 | 50 phonemes + 3 specials |
| `blank_id` | 0 | CTC blank = PAD |
| `max_epochs` | 25 | Training budget (early stopping usually fires first) |
| `physical_batch_size` | 32 | Micro-batch per forward |
| `gradient_accumulation_steps` | 2 | Effective batch = 64 |
| `head_lr` / `encoder_lr` | 3e-4 / 1e-4 | Discriminative LR |
| `llrd_decay` | 0.85 | Layer-wise LR decay factor |
| `lr_min_ratio` | 0.10 | LR floor (10% of base) |
| `warmup_steps` | 1500 | Linear warmup duration |
| `warmup_head_only_steps` | 0 | Head-only steps (disabled) |
| `sr_ctc_beta` | 0.1 | SR-CTC regularisation weight |
| `weight_decay` | 0.015 | AdamW L2 |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `early_stopping_patience` | 5 | Epochs without PER improvement |
| `save_top_k` | 5 | Best checkpoints to keep |
| `bf16` / `tf32` | true / true | Mixed precision |
| `mask_time_prob` | 0.30 | SpecAugment time masking |
| `ds1_mask_time_prob` | 0.15 | Reduced SpecAug for DS1 batches |
| `ds_oversample` | `{1: 3}` | DS1 3× oversampling |
| `checkpoint_averaging` | configurable | Post-training top-N averaging |
