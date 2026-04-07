# CTC Tokenizer — Vocab & Metadata Pipeline (Stage 3)

> **Note:** The definitive Run 24 tokenizer output is in [study/runs/run_24/tokenizer/](../study/runs/run_24/tokenizer/) — includes [vocab.json](../study/runs/run_24/tokenizer/vocab.json), [data_card.json](../study/runs/run_24/tokenizer/data_card.json), [phoneme_freq.tsv](../study/runs/run_24/tokenizer/phoneme_freq.tsv), and [dataset_fingerprint.json](../study/runs/run_24/tokenizer/dataset_fingerprint.json).

> **File:** `src/tokenizer/tokenizer.py`
> **Runs via:** `python pipeline.py --etl` (fourth stage, after audio EDA)
> **Prerequisite:** Sanity + Audio EDA must have run first (reads from `data/processed/` + `data/reports/`)
> **Input:** Cleaned JSONL (`data/processed/*_transcript.jsonl`) + EDA removal reports (`data/reports/eda_*.json`) + raw transcripts (for vocab inclusion of audio-broken labels)
> **Output:** `data/models/tokenizer/` (vocab.json, tokenizer_config.json, special_tokens_map.json, inventory.txt, dataset_fingerprint.json, phoneme_freq.tsv, data_card.json)
> **Config:** Output directory derived from `cfg["paths"]["tokenizer"]`; dataset keys from `cfg["datasets"]`
> **Datasets:** Same canonical list as Sanity and Audio EDA

---

## Rationale — Why This Stage Exists

CTC (Connectionist Temporal Classification) requires a fixed, deterministic mapping from phoneme characters to integer IDs. This is not a general-purpose text tokenizer — it's a **character-level vocabulary** where every IPA symbol gets exactly one logit in the model's output head.

A broken tokenizer is the #1 silent failure mode in CTC pipelines:

- **Duplicate entries** (e.g., composed vs decomposed `ʃ` + combining char): model wastes logit capacity, CTC alignment is ambiguous, PER silently degrades.
- **Missing characters**: `[UNK]` absorbs rare phonemes. Training loss decreases (model predicts UNK for unknowns) but PER on those phonemes stays terrible.
- **Wrong blank ID**: HuggingFace `Wav2Vec2ForCTC` hardcodes `blank_id = pad_token_id`. If PAD isn't ID 0, CTC loss computes **wrong gradients** silently.
- **Non-deterministic ID assignment**: Python `set()` has hash-randomized iteration. Without sorted vocab, the same code on different machines assigns different IDs → checkpoints are **not portable**.

The tokenizer stage eliminates all of these by construction: NFC-normalised input, sorted vocab, anchored special tokens, zero-OOV assertion, IPA validation, and no BOS/EOS.

## How It Helps the Overall Pipeline

| Downstream stage | What the Tokenizer provides |
|-----------------|-----------------------------|
| Data Split | Phoneme frequency counts for CTC coverage checking and rare-phoneme speaker pinning |
| `training_controls.json` | `phoneme_frequency` block with per-phoneme count/percent → drives weighted CTC loss, curriculum learning |
| Training (SFT) | `vocab.json` → `Wav2Vec2CTCTokenizer` encodes labels to integer sequences for CTC loss |
| Training (SSL) | Not used — SSL pre-training has no labels |
| Evaluation | `phoneme_freq.tsv` → weighted PER computation, per-phoneme error heatmaps |
| Reproducibility | `dataset_fingerprint.json` + `data_card.json` = paper-ready corpus documentation |

---

## What It Does

Reads cleaned JSONL → mines character inventory → builds a deterministic CTC-compatible HuggingFace tokenizer → verifies zero OOV → saves tokenizer + research metadata.

**Final numbers (Feb 2026):**

| Metric | Value |
|--------|-------|
| Vocab size | 53 (50 chars + 3 specials) |
| Specials | `[PAD]=0`, `[UNK]=1`, `\|=2` |
| BOS/EOS | None (suppressed) |
| Blank ID | 0 (`pad_token_id`) |
| OOV rate | 0.000% (guaranteed by assertion) |
| Training rows | 151,705 (83.06 hours) |
| Vocab built from | 151,709 labels (151,705 trainable + 4 audio-broken) |

---

## Step-by-Step: What, Why, How

### 1. Stream + Collect Character Counts

**What:** Iterate every character in every `phonetic_text` field, count occurrences. Skip spaces.

**Why:** The vocab must come from **post-sanity-normalised** data, not raw. If you built it from raw pre-sanity data, you'd include characters that were dropped or normalized away (e.g., ASCII `r` which was replaced by `ɹ`). The result would be a vocab with dead tokens the model wastes logit capacity on.

**Exception — EDA-removed labels:** Rows that Audio EDA removes (load failures, duration mismatches) have broken *audio* but valid *phonemic labels*. The competition host intended those labels to be in the dataset. After streaming the cleaned training corpus, the tokenizer reads the EDA removal reports (`eda_failed_loads.json`, `eda_duration_mismatch.json`), looks up the removed rows' labels from the raw transcripts, and merges their characters into the vocab. This ensures the model can recognise the full intended phoneme space, even for phonemes that only appeared in audio-broken rows. If any exclusive characters are found (present only in removed rows), they are logged as `RESCUED`.

**How:** Single-pass streaming — no need to load all data into memory. Also collects corpus-wide stats (durations, TPS/PPS/SPS, phoneme counts) for the metadata files, avoiding a second pass.

**Guards after collection:**

- `assert durations` — catches empty corpus (upstream wipe)
- `assert has_space` — catches label corruption (no word boundaries)
- `assert "|" not in unique_chars` — catches delimiter collision in raw labels
- `assert tps_values` — catches all-zero-duration edge case
- `assert n_phonemes_list` — catches empty phoneme collection

**IPA consistency guard:** If both ASCII `g` (U+0067) and IPA `ɡ` (U+0261) are present in the corpus, the build **fails with an assertion** — two different tokens for the same phoneme corrupts CTC targets. Must be normalised upstream in `TEXT_FIXES` before re-running.

**Source ordering:** Input paths are sorted (`sorted(clean_jsonl_paths)`) before iteration — ensures the fingerprint is deterministic even if config list order changes.

---

### 2. Sorted Character Set

**What:** `vocab_list = sorted(unique_chars)`

**Why:** Python `set()` has no guaranteed order — iteration order depends on hash randomization, which varies across Python invocations and machines. Without sorting, the same code on a different machine could assign different integer IDs to the same character. Your model checkpoint would be **incompatible across machines** — fine-tuning on machine A, evaluating on machine B would silently produce garbage because the model's output head maps logit index 5 to `d` on one machine and `e` on another.

**How:** `sorted()` uses Unicode codepoint order → deterministic, reproducible IDs everywhere.

---

### 3. Special Token Anchoring

**What:** `{"[PAD]": 0, "[UNK]": 1, "|": 2}`

**Why each one:**

#### `[PAD] = 0` — CTC Blank

CTC loss uses a "blank" token to represent silence/repetition between phoneme emissions. HuggingFace's `Wav2Vec2ForCTC` hardcodes `blank_id = config.pad_token_id`. If PAD isn't ID 0, or if `model.config.pad_token_id` doesn't match, CTC loss **silently computes wrong gradients** — your model trains but converges to a worse optimum. This is the single most common silent bug in CTC pipelines.

**Enforcement:** The tokenizer build asserts `tokenizer.pad_token_id == 0` immediately after instantiation — catches any HF version that silently reorders specials.

#### `[UNK] = 1` — Unknown Token

Required by the HuggingFace tokenizer contract. After our zero-OOV check (step 7), this token should **never** be emitted during training. If it does, something has changed in the data since the tokenizer was built.

#### `| = 2` — Word Delimiter

HuggingFace's `replace_word_delimiter_char=" "` automatically swaps spaces in labels to `|` during encoding. So `"ʔə ʔæpɫ"` becomes `["ʔ", "ə", "|", "ʔ", "æ", "p", "ɫ"]` internally. This is how CTC knows where word boundaries are during decoding.

**How:** Hardcoded as the first 3 entries in `vocab.json` before appending the corpus characters starting at ID 3.

---

### 4. `bos_token=None, eos_token=None`

**What:** Explicitly suppress begin-of-sequence and end-of-sequence tokens.

**Why:** CTC is **not** a sequence-to-sequence model. It doesn't use begin/end markers — there's no autoregressive decoding step where the model needs to know when to start or stop. If you leave HuggingFace defaults, it auto-creates `<s>=53` and `</s>=54` in an `added_tokens.json` file. This:

- Wastes 2 logit dimensions (the linear output head has 55 outputs instead of 53)
- Can break `from_pretrained()` on some HF versions that serialize `None` differently
- Creates confusion when `len(tokenizer)` returns 55 but your vocab.json has 53 entries

**How:** Pass `bos_token=None, eos_token=None` to the constructor. HF doesn't register them → no `added_tokens.json` → vocab locked at exactly 53.

---

### 5. Per-Character OOV Check

**What:** For every character in the vocabulary, assert it doesn't map to `[UNK]`.

**Why:** If any character in your labels maps to `[UNK]` at training time, CTC tries to align audio to the UNK token. The model can never learn those phonemes correctly — it treats all unknown phonemes as one category. This is a **silent** failure: training loss decreases (the model learns to predict UNK for rare sounds) but PER on those phonemes stays terrible. You wouldn't notice until you do a per-phoneme error analysis after training.

**How:** `tokenizer.convert_tokens_to_ids(ch) != tokenizer.unk_token_id` — uses the direct ID lookup instead of `tokenizer(ch).input_ids[0]`. The `__call__()` method applies normalization that could mask the issue (e.g., lowercasing, NFKC). Direct lookup tests what actually happens at encode time.

---

### 6. Linguistic Audit (Step 9 in code)

**What:** After building the tokenizer, print the corpus frequency of 7 "suspicious" Latin letters: `c e j o w x z`.

**Why:** In a phonetic dataset, seeing plain Latin letters could mean orthographic text leaked through (someone typed English instead of IPA). In standard IPA, `c` is the voiceless palatal plosive (not the English letter C), and `x` is the voiceless velar fricative (not the English letter X). By printing their counts on every build, you catch data corruption early — if `c` suddenly jumps from 176 to 50,000, a labeller started using English orthography.

**Confirmed valid for this dataset:**

| Token | IPA Value | Count | Explanation |
|-------|-----------|------:|-------------|
| `c` | Voiceless palatal plosive | 176 | Fronted /k/, toddler speech pattern |
| `e` | Close-mid front vowel | 30,202 | Pure monophthong, un-diphthongized |
| `j` | Palatal approximant | 18,156 | Standard IPA for "yes" sound |
| `o` | Close-mid back vowel | 33,200 | Pure monophthong, un-diphthongized |
| `w` | Labial-velar approximant | 49,185 | Standard IPA |
| `x` | Voiceless velar fricative | 153 | Guttural /k/ failure in child speech |
| `z` | Voiced alveolar fricative | 23,849 | Standard IPA |

---

### 6b. Rare Phoneme Audit

**What:** After writing `phoneme_freq.tsv`, any phoneme with frequency < 0.1% of total characters triggers an audit block. Up to 5 example transcripts containing that phoneme are printed.

**Why:** Low-frequency phonemes are the most likely to be labelling errors, Unicode confusables, or non-native speech patterns. Seeing actual transcripts immediately tells you whether `ç` at 0.007% is real palatalized speech or a keyboard accident. No filtering happens — this is inspection only.

**How:** During the single-pass stream (step 1), up to 5 example `phonetic_text` strings are stored per character. After `phoneme_freq.tsv` is written, characters with `pct < 0.1` are reported with their Unicode codepoint, count, frequency, and examples.

**Example output (Feb 2026):**

```
--- RARE PHONEME AUDIT (<0.1% frequency) ---
[RARE] U+0063  'c'  count=   176  (0.0118%)
[RARE]   1. æmɛcθ
[RARE]   2. ɪts oɚnʤ wɪf lɑ oɚnc ɑn ɪ
[RARE] U+03C7  'χ'  count=    18  (0.0012%)
[RARE]   1. χɹut
[RARE]   2. χɹi
--- RARE PHONEME AUDIT END ---
```

**Threshold:** 0.1% of total characters. At 1.49M total phoneme tokens (Feb 2026), this means any phoneme with fewer than ~1,490 occurrences is audited.

---

### 7. IPA Validation (Step 10 in code)

**What:** Every character in the vocabulary is checked against a canonical IPA inventory (`_IPA_CHARS` frozenset — 100+ valid IPA symbols including pulmonic/non-pulmonic consonants, all vowel positions, suprasegmentals, clinical extensions).

**Why:** Catches non-IPA characters that slipped through upstream processing — could be Unicode confusables, encoding artifacts, or labelling errors. A non-IPA token in the vocab means the model learns a logit for a phoneme that doesn't exist.

**How:** `[ch for ch in vocab_list if ch not in _IPA_CHARS]` — any failures are printed with their Unicode codepoint, then the build **fails with an assertion** (`assert not non_ipa`). This is fatal, not a warning — non-IPA symbols in a phoneme vocab means either grapheme leak or a missing entry in `_IPA_CHARS`. Fix upstream labels or update the inventory before re-running.

---

### 8. Metadata Files (Step 11 in code)

Three files persisted for reproducibility and downstream analysis:

#### `dataset_fingerprint.json`

**What:** Rows kept, total duration, percentile distributions (duration, TPS, PPS, SPS, transcript length), full phoneme inventory, special token mapping.

**Why:** If you re-run the pipeline next month and the numbers change, you know the data changed. This is a free reproducibility section for a paper — anyone can verify your exact corpus statistics. The `special_tokens` block (`pad`, `unk`, `word_delimiter`) lets inference assert tokenizer compatibility automatically.

**TPS vs PPS vs SPS:** TPS (tokens per second) counts all IPA characters and drives the physics filter. PPS (phonemes per second) excludes suprasegmentals — scientifically correct for papers. SPS (suprasegmentals per second) captures length marks (`ː`). All three are in the fingerprint; TPS has full percentiles, PPS/SPS have p50 + p99.

**Percentile formula:** Nearest-rank `round((N-1) * p / 100)` via `nearest_rank_pctl()` from `src/utils/utils.py` — single source of truth shared with `eda_processor.py`. Unbiased at corpus scale (150K+ elements).

**Key stats:**

- Duration: p50=1.1s, p99=12.5s
- TPS: p50=4.5, p99=12.2
- Transcript length: p50=5 phonemes, p99=59

#### `phoneme_freq.tsv`

**What:** Tab-separated file with phoneme, absolute count, and percentage for all 50 characters.

**Why:** Class imbalance analysis. The most frequent phoneme (`ə`, schwa) appears ~200K times; the rarest (`x`, velar fricative) appears 153 times — a 1000:1 ratio. This data feeds into:

- Weighted PER (don't let rare phonemes dominate your error metric)
- Error heatmaps (which phonemes does the model confuse?)
- Curriculum learning (train on common phonemes first, introduce rare ones later)
- Weighted CTC loss (upweight rare phoneme targets)

#### `data_card.json`

**What:** Machine-readable documentation of every design decision.

**Why:** A methods section in JSON format. Documents:

- Label scheme: `narrow_phonemic`
- Affricate convention: `single_token (ʧ ʤ)`
- R-symbol: `ɹ`
- Text fixes applied
- Filter thresholds (derived from actual data, not hardcoded)
- `rate_metric_note` — explains that TPS counts all IPA characters (including `ː`), PPS excludes suprasegmentals, and that true phonemic rate (handling ties, affricates-as-one-segment, coarticulation) requires separate linguistic analysis

Anyone reading this file knows exactly what preprocessing was applied without reading the code.

---

## Full Phoneme Inventory (50 characters)

```text
b c d e f g h i j k l m n o p s t u v w x z
æ ç ð ŋ ɐ ɑ ɔ ə ɚ ɛ ɟ ɪ ɫ ɬ ɹ ɾ ʁ ʃ ʊ ʌ ʒ ʔ ʝ ʤ ʧ ː θ χ
```

**Coverage:** Standard English phoneme set + clinical extensions for child speech errors (fronted stops, velar fricatives, palatal variants). Research-grade inventory, not toy ASR.

---

## Critical Reminders for Training

When creating the model, you **must** set:

```python
model.config.pad_token_id = tokenizer.pad_token_id  # = 0
model.config.vocab_size = len(tokenizer)             # = 53
```

If you forget this, CTC loss will use the wrong blank ID and your model will not converge properly.

---

## Output Directory Contents

```text
data/models/tokenizer/
├── vocab.json                  # Character → ID mapping (53 entries)
├── tokenizer_config.json       # HuggingFace tokenizer config
├── special_tokens_map.json     # Special token definitions
├── inventory.txt               # Plain-text phoneme list (for debugging)
├── dataset_fingerprint.json    # Corpus statistics + reproducibility
├── phoneme_freq.tsv            # Per-phoneme frequency table
└── data_card.json              # Design decisions documentation
```

---

## Decision Evolution — Mar 2026 Update

The tokenizer design is unchanged; the corpus snapshot evolved.

| Earlier baseline in this doc | Current files now | Why this moved |
|---|---|---|
| Rows_total around 151K+ context | `rows_total=150,928` (`dataset_fingerprint.json`) | Additional audio-level removals after EDA reduced trainable rows.
| Keep labels from EDA-removed audio in vocab | Still active: `eda_removed_labels_in_vocab=4` | This policy prevents phoneme coverage holes caused by audio corruption.
| Fixed vocab size 53 with anchored specials | Still exactly true (`vocab_size=53`) | This remains the core CTC safety invariant for model portability and loss correctness.
| Duration/TPS stats from earlier run | Current fingerprint: `duration p99=10.848`, `TPS p99=12.162` | Updated corpus snapshot after full ETL re-run and anomaly removals.

**Current rationale:** we intentionally did not change tokenizer policy; we only updated evidence and corpus statistics that the same policy now operates on.
