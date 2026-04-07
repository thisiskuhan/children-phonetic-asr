# Submission Inference — Beam Search + KenLM

> **Package:** `submission/src/` (main.py + decode modules + model weights)
> **Entry point:** `python src/main.py` (invoked by `entrypoint.sh` in the container)
> **Runtime:** DrivenData A100 80 GB, 24 vCPU, 220 GB RAM, 2-hour limit, no network
> **Model:** WavLM Base+ CTC (`model.safetensors`, 94.7M params)
> **Decode:** Beam search + KenLM 3-gram LM primary, greedy CTC fallback
> **Build:** `bash zip_submission.sh [name]` → `submission/<name>.zip`

---

## Pipeline Overview

```
Audio files → ThreadPool load + resample + silence trim + CMVN
           → Length-sorted, adaptive batch sizing
           → GPU forward (WavLMForCTC, fp16 autocast)
           → UNK suppression (logits[:, :, 1] = -1e9)
           → Beam search + KenLM 3-gram LM decode
              └── on failure → greedy CTC fallback
                  └── on OOM → retry at reduced batch size
           → IPA post-processing (strip invalid chars, normalise spaces)
           → submission.jsonl
```

---

## Decode Modes

### Beam Search (primary)

Uses `pyctcdecode.build_ctcdecoder` with a KenLM word-level 3-gram language model trained on gold phonetic transcripts.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `beam_width` | 100 | Search breadth |
| `alpha` | 0.575 | LM weight |
| `beta` | 3.0 | Word insertion bonus |
| `beam_prune_logp` | -10.0 | Prune beams with log-prob below this |
| `token_min_logp` | -5.0 | Minimum token log-prob to consider |

Decode is parallelised via `multiprocessing.Pool` (up to 16 workers). Each batch's log-softmax output is transferred to CPU as numpy arrays for pyctcdecode.

If any batch fails during beam decode, that batch falls back to greedy silently and continues.

### Greedy (fallback)

Pure CTC greedy: argmax → mask padding → collapse repeats → remove blank & UNK. Implemented as tensor ops with zero Python loops over frames. Used when:

- Beam search initialisation fails (missing LM/vocab)
- Beam batch decode raises an exception
- `DECODE_MODE="greedy"` is set explicitly

If greedy also fails (OOM), the batch size is halved (128 → 16) and retried.

---

## Audio Preprocessing

Matches training pipeline (`data_collator.py`) exactly:

| Step | Detail |
|------|--------|
| Load | `soundfile.read` → float32 numpy |
| Mono | `mean(dim=-1)` if multi-channel |
| Resample | `torchaudio.functional.resample` to 16 kHz |
| Silence trim | Leading/trailing silence below -40 dB (re peak), min 0.5s after trim |
| Absolute floor | Samples below 1e-4 (~-80 dBFS) zeroed — removes sub-audible residual noise |
| CMVN | Zero-mean unit-variance per utterance — matches WavLM pretraining |

Audio loading is threaded (`ThreadPoolExecutor`, 20 workers) for CPU-bound I/O overlap.

Samples shorter than 0.1s after preprocessing are dropped (empty prediction).

---

## Adaptive Batching

Utterances are sorted longest-first. Each batch is sized by:

```python
bs = min(BATCH_SIZE, max(1, MAX_SAMPLES // max_len_in_batch))
```

This caps total samples per batch at ~8s × 128 = 16.4M samples, preventing OOM on long utterances while maximising throughput on short ones. Waveforms within a batch are padded to the nearest 1-second boundary (`BUCKET_STEP = 16000`).

---

## Source Layout

```
submission/src/
├── main.py                 (534 lines)  Inference pipeline
├── decode_beam_lm.py       (184 lines)  pyctcdecode + KenLM wrapper
├── decode_greedy.py         (33 lines)  Tensor-op greedy decoder
├── tokenizer/                           Wav2Vec2CTCTokenizer files
│   └── vocab.json                       53-token IPA vocab
├── model/                               Model weights
│   ├── model.safetensors                Averaged checkpoint (~360 MB)
│   └── config.json                      WavLM config with project overrides
├── kenlm_word3gram.bin                  KenLM binary LM (or .arpa)
└── unigrams.json                        Unigram list for pyctcdecode
```

---

## Build Scripts (`submission/scripts/`)

| Script | Purpose |
|--------|---------|
| `avg_checkpoints.py` | Average top-N HF checkpoint state dicts in fp32, save to submission model dir |
| `build_arpa_lm.py` | Build KenLM ARPA 3-gram LM from gold phonetic transcripts (Witten-Bell smoothing) |
| `build_lm.py` | Build flat binary trigram LM (V×V×V log-probs) for C beam search (alternative) |
| `extract_weights.py` | Extract `model_state_dict` from training `.pt` checkpoint with integrity validation |
| `generate_config.py` | Download pretrained WavLM config and apply project-specific overrides |

---

## Build & Submit

```bash
# 1. Average top-5 checkpoints
python submission/scripts/avg_checkpoints.py \
    data/models/hf_sft_checkpoints/checkpoint-{A,B,C,D,E} \
    --output submission/src/model/

# 2. Build KenLM LM
python submission/scripts/build_arpa_lm.py

# 3. Package
bash zip_submission.sh my_submission   # → submission/my_submission.zip
```

`zip_submission.sh` zips `submission/src/` excluding `__pycache__` and `.pyc` files.

---

## Key Design Decisions

### Why `torch.compile` is Disabled
CTC models produce variable-length outputs depending on input duration. `torch.compile` / inductor constantly recompiles on dynamic shapes, wastes ~11 GB RAM spawning 32 compile-worker threads, and provides no speedup for variable-length inference. Disabled at import time:

```python
torch._dynamo.config.disable = True  # (set in Dockerfile/entrypoint)
```

### Why UNK is Suppressed in Logits
UNK (ID 1) should never appear in predictions. Setting `logits[:, :, UNK] = -1e9` before softmax/argmax eliminates it from beam search candidates and greedy decisions. Applied both during inference and training evaluation.

### Why Beam Width = 100
Tuned on held-out val set. Width 100 provides diminishing returns past 50 but remains within the 2-hour time budget. The parallel decode pool (16 workers) keeps wall-clock cost manageable.

### Why CMVN Instead of RMS
Training uses RMS normalisation. However, CMVN (zero-mean, unit-variance) matches WavLM's pre-training distribution more closely and gave slightly better results at inference. The model tolerates this mismatch because the encoder's learned representations are robust to normalisation scheme differences.

### Why No Ensemble
Single-model inference stays within the 2-hour budget with beam search. Multi-model ensembling was tested but the latency cost exceeded the PER improvement, especially with beam width 100.

---

## IPA Scoring Reference

### Metric

$$\text{IPA-CER} = \frac{S + D + I}{N}$$

S = substitutions, D = deletions, I = insertions, N = reference chars. Lower is better. Computed by `jiwer.cer()` after normalisation.

### Normalisation Pipeline (`metric/score.py → normalize_ipa`)

Both prediction and reference pass through these steps before CER:

1. NFC Unicode normalisation
2. Nasal vowel decomposition — `ẽ→e`, `ĩ→i`, `õ→o`, `ũ→u`
3. Delete tie bars & stress — `U+035C`, `U+0361`, `ˈ`, `ˌ`, combining tilde
4. Rhotic normalisation — `ɝ→ɚ`
5. Remove ASCII punctuation — all `string.punctuation` deleted
6. Digraph → ligature — `tʃ→ʧ`, `dʒ→ʤ`
7. Collapse whitespace — multiple spaces → single space, strip

### Valid IPA Characters (50 + space)

**Consonants:** b c d f g h j k l m n p r s t v w x z
**Vowels:** e i o u
**IPA vowels:** ɑ æ ɐ ɔ ə ɚ ɛ ɪ ʊ ʌ
**IPA consonants:** ç ð ŋ ɟ ɫ ɬ ɹ ɾ ʁ ʃ ʒ ʔ ʝ θ χ
**Ligatures:** ʧ ʤ | **Length:** ː | **Delimiter:** space

Our 53-token vocab (50 IPA + PAD + UNK + `|`) aligns exactly — `|` maps to space, PAD/UNK are filtered before decode.

### Test Set Sizes

| Variant | Lines | Purpose |
|---------|-------|---------|
| Smoke test | 3,000 | Training subset for debugging (not scored for prizes) |
| Full test | 77,011 | Scored with IPA-CER, must complete within 2 hours |
