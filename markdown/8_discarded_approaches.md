# Discarded Approaches — SSL Pre-Training & 3-Stage SFT

> Things we built, tried (or planned to try), and ultimately moved away from.
> Documented here for honesty — and because maybe these ideas weren't bad,
> just badly timed or not tuned well enough. Worth revisiting if the ceiling
> is hit with the current approach.

---

## 1. Three-Stage Signal-Driven SFT Trainer

**File:** `src/trainer/sft_trainer.py` (2,370 lines)
**Active:** Feb 25 – Mar 13, 2026 (Runs 1–10)
**Replaced by:** `sft_trainer_hf.py` (HuggingFace Trainer wrapper)
**Git:** Renamed to `deprecated_sft_trainer.py` at commit `868bd3f`, then removed

### What It Was

A fully custom training loop with 3-stage progressive unfreezing, driven by EMA-smoothed signals rather than fixed epoch counts:

| Stage | Trainable | Trigger | What Happens |
|-------|-----------|---------|--------------|
| 1 — Head only | `lm_head` (40K params) | Start of training | Only the CTC head learns. Encoder is frozen. Wait for blank ratio to drop below `stage1_blank_threshold` (0.945). |
| 2 — Upper encoder | Head + layers 6–11 (42.6M) | Blank threshold met | Encoder layers unfreeze top-down. Per-group warmup ramps new params over 1,000 steps. EMA tracks PER improvement rate. Transition to Stage 3 when `consecutive_slowing ≥ 3` and relative PER improvement < 1% for 3 consecutive evals. |
| 3 — Full model | All params + CNN (94.7M) | Diminishing returns in S2 | Everything trainable including CNN and feature projection. Gradient checkpointing enabled. |

Additional machinery the custom loop had:
- **EMA signal trackers** — smoothed PER, loss, and blank ratio to detect stalls
- **Per-group warmup** — newly unfrozen parameters had their own LR ramp (1,000 steps) at each stage transition
- **Age-weighted CTC loss** — inverse-frequency weighting per age bucket (3–4, 5–7, 8–11, 12+)
- **Manual AMP** — bf16/fp16 with GradScaler, manual gradient accumulation, custom LR schedule
- **Custom W&B tracking** — background daemon with bounded queue
- **EMA model weights** — exponential moving average of parameters for evaluation
- **Beam search eval** — optional beam decode during validation

### Results with the 3-Stage Trainer

| Run | Epochs | Best Val PER | Notes |
|-----|--------|--------------|-------|
| Run 1 (Run 3 internally) | 27 | 0.4141 | First complete run. LR hit zero early. GradScaler collapsed. |
| Run 4–10 | varied | ~0.33–0.35 | Incremental fixes (cosine LR floor, bf16, SpecAug tuning) |

### Why We Moved Away

**The short answer:** Stage 2 delivered most of the gains. Stage 3 added diminishing returns. The stage transitions themselves added no measurable benefit over simply starting with everything unfrozen and using LLRD.

Run 1 breakdown:
- Stage 1 (E1–4): PER 0.91 → 0.77 — blank initialisation, necessary
- Stage 2 (E5–18): PER 0.77 → 0.48 — **bulk of real learning** (−38% relative)
- Stage 3 (E19–27): PER 0.48 → 0.41 — diminishing (−13% relative)

When we built the HF Trainer wrapper (`sft_trainer_hf.py`), we tested head-only warmup vs full unfreeze from step 0 with LLRD. Full unfreeze matched or beat 3-stage at the same epoch budget — with 500 fewer lines and no signal tracking bugs.

The custom machinery — EMA smoothing, transition gates, per-group warmup — added complexity without adding PER improvement. The 2,370-line custom loop had bugs we kept finding (LR schedule restoration on resume, GradScaler instability in Stage 3, grad-ratio band warnings that were cosmetic). The HF Trainer wrapper was correct by construction.

### Honest Caveat

The 3-stage approach wasn't necessarily wrong. It was inspired by gradual unfreezing literature (ULMFiT, etc.) and could theoretically prevent catastrophic forgetting of lower layers. But:

1. **We didn't tune the transition signals well.** The EMA smoothing windows and threshold values were somewhat arbitrary. Better signal design might have helped.
2. **We didn't try enough runs.** Only ~6 complete runs with the old trainer before switching. The HF wrapper got 10+ runs with more systematic hyperparameter variation.
3. **LLRD might be doing the same job.** Layer-wise LR decay (0.85×) already gives lower layers less gradient. That might be sufficient protective gating without explicit staging.

If someone revisits this: try 2-stage (head-only warmup → full with LLRD) instead of 3-stage. The current `warmup_head_only_steps` in `sft_trainer_hf.py` supports this — it's just set to 0 because we saw no benefit.

---

## 2. SSL Continued Pre-Training

**File:** `src/trainer/ssl_trainer.py` (759 lines)
**Active:** Mar 11–16, 2026 (built and tested, never used for a production run)
**Git:** Last seen at `fd8450f`, code subsequently removed from the codebase
**Plan docs:** `study/ssl_pretraining_plan.md`, `study/ssl_implementation_plan.md`

### What It Was

Self-supervised continued pre-training of WavLM Base+ on children's audio before any CTC fine-tuning. The idea: WavLM was pre-trained on 94K hours of adult speech. Children violate adult coarticulation, prosody, and phoneme boundaries. SSL would re-tune the encoder's internal representations on our ~163 hours of children's data (labelled + orphan audio combined) using the same masked prediction objective WavLM was originally trained with.

**Architecture:** `Wav2Vec2ForPreTraining` — mask spans of the CNN feature output, then predict the masked frames from context. No labels needed.

**Full pipeline we built:**
- `SSLDataset` — JSONL manifest reader for labelled + orphan audio
- `SSLCollator` — audio loading, MUSAN noise + RIR augmentation, masking
- `build_ssl_model()` — load pretrained + freeze CNN + set mask config
- `ssl_trainer.py` — custom training loop with evaluation on masked accuracy
- `compute_masked_lm_accuracy()` — validation metric
- `src/tests/test_ssl.py` — 61 tests covering model, collator, training step, checkpoint, integration
- `pipeline.py --ssl` entry point

**Config planned:** 10–20 epochs, LR 5e-5, frozen CNN, mask 65% of frames, cosine schedule.

### Why It Was Never Used

1. **SFT kept improving.** The original decision rule was: "run SSL if Run 4 val PER > 0.36". Run 4 beat that, and every subsequent run kept improving. SSL was always "next if we plateau" — and we never plateaued hard enough to justify the detour.

2. **Time constraint.** SSL pre-training on 163 hours would take ~6–10 hours on our GPU budget. That's an entire training run we could spend on SFT hyperparameter tuning instead. Given the competition timeline, SFT iteration had better expected ROI.

3. **Domain mismatch might be smaller than expected.** WavLM Base+ transferred surprisingly well to children's speech. By Run 17 (PER 0.2586), the encoder was already producing good phoneme boundaries without domain-specific pre-training. The gap between adult and child acoustics may not be as large as the literature suggests for this age range (3–12 year olds, not toddlers).

### Honest Caveat

SSL continued pre-training is a well-established technique that works in published papers (e.g., Wav2Vec2 for low-resource languages). We didn't give it a fair shot:

1. **Never ran it once with real data.** All testing was on synthetic/demo data. The implementation was complete and passing 61 tests, but we never committed GPU hours to an actual pre-training run.
2. **The orphan data is free information.** We have 114K unlabelled audio files (~89 hours) that SFT can't use at all. SSL is the only way to extract signal from them. That's untapped potential.
3. **The DS1 problem might benefit from SSL.** DS1 (3–4 year olds, noisy, ~12 dB SNR) is structurally the hardest subset. SSL pre-training on DS1 audio might help the encoder learn child-specific spectral patterns that SFT augmentation alone can't teach.

If the PER ceiling is hit at ~0.22–0.23, SSL is the first thing worth revisiting. The code was battle-tested (61 tests), the config was designed, and the data pipeline was ready. It just needed GPU time.

---

## Timeline

```
Feb 25     First SFT trainer commit (3-stage custom loop)
Mar 2      W&B integration, GPU optimization
Mar 3      Stage gate logic modifications, OOM fixes
Mar 5      Run 1 complete — PER 0.4141 (0.3528 on leaderboard)
Mar 9      Run 3 — PER 0.4141 (27 epochs, 8.9h on RTX 5090)
Mar 10     LLRD + EMA weights added to old trainer
Mar 11     SSL trainer built (ssl_trainer.py, 759 lines)
Mar 13     HF Trainer wrapper built (sft_trainer_hf.py)
           Old trainer renamed to deprecated_sft_trainer.py
           SSL trainer enhanced
Mar 16     SSL trainer reaches working state (all tests passing)
Mar 18     Run 11 — PER 0.282 (first run on HF Trainer)
Mar 22     Run 17 — PER 0.2586 (full augmentation)
Mar 24     Run 20 — PER 0.2627 (SR-CTC, LR floor)
Mar 28     Run 24 — PER 0.2314 (DS1 SpecAug reduction) ← current best
```

The switch from custom loop to HF Trainer happened at commit `868bd3f` (Mar 13). Every run from Run 11 onwards used the HF wrapper. The old trainer was kept as `deprecated_sft_trainer.py` for reference but never used again.
