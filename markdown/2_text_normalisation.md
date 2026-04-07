# Sanity — Text Normalisation & Cleaning Pipeline (Stage 1)

> **Note:** Numbers below reflect an earlier pipeline config (max_duration=15s, mic_click=0.1s). The final Run 24 used max_duration=16s and min_duration=0.5s — dropping more rows (153,067 → 134,434). See [README.md](../README.md).

> **File:** `src/etl/eda_processor.py` → `_sanity_fix()`
> **Runs via:** `python pipeline.py --etl` (second stage, after audio check)
> **Input:** Raw JSONL (`data/raw/*_train_phon_transcripts.jsonl`)
> **Output:** Cleaned JSONL (`data/processed/*_transcript.jsonl`)
> **Config:** `config.yaml` → `eda:` section (all thresholds config-driven)
> **Datasets:** Derived from `cfg["datasets"]` — single canonical list shared with Audio EDA and tokenizer

---

## Rationale — Why This Stage Exists

Raw competition JSONL comes from multiple labellers with inconsistent conventions. Without normalisation:

- **Unicode ghosts**: The same phoneme stored as different byte sequences creates duplicate vocab entries. The model wastes logit capacity learning two representations of one sound, and CTC alignment becomes ambiguous.
- **Bad rows**: Run-on labels (missing spaces), mic clicks (<0.1s), and physics-impossible transcripts (TPS outside [1, 25]) poison CTC training. The model tries to learn impossible alignment paths — producing `loss = inf` or silently degraded convergence.
- **Whitespace inconsistency**: Double spaces create empty words (`||` tokens) that confuse CTC's blank-token logic.

Sanity is the **single point of truth** for label quality. Every downstream module (Audio EDA, tokenizer, split, trainer) reads from `data/processed/*_transcript.jsonl` and trusts that every row is normalised, valid, and trainable.

## How It Helps the Overall Pipeline

| Downstream stage | What Sanity prevents |
|-----------------|----------------------|
| Audio EDA | Computing spectral features on rows that will never train (saves ~1% compute) |
| Tokenizer | Duplicate vocab entries from unnormalised Unicode; OOV crashes from stale characters |
| Data Split | Speaker duration accounting skewed by phantom/corrupt rows |
| Training | CTC `inf` loss from impossible alignments; wasted gradient steps on garbage rows |
| Evaluation | Inflated PER from inconsistent label conventions between datasets |

---

## What It Does

Streams raw JSONL → normalizes text → drops bad rows → saves clean JSONL → runs an independent health check on the output.

**Final numbers (Feb 2026):**

| Dataset | Raw | Clean | Kept | Dropped |
|---------|----:|------:|-----:|--------:|
| DS1 | 12,043 | 11,919 | 99.0% | 124 |
| DS2 | 141,024 | 139,790 | 99.1% | 1,234 |
| **Total** | **153,067** | **151,709** | **99.1%** | **1,358** |

---

## Step-by-Step: What, Why, How

### 1. NFC Unicode Normalization

**What:** `unicodedata.normalize("NFC", text)`

**Why:** The same IPA character can be stored as 1 codepoint or 2+ codepoints (composed vs decomposed). Without NFC, `ʧ` and `t+ʃ` look identical to humans but are different bytes. The tokenizer would create duplicate vocab entries for the same sound — the model wastes capacity learning two representations of one phoneme.

**How:** NFC (Canonical Decomposition followed by Canonical Composition) collapses decomposed forms into single codepoints. Applied first, before any other text processing.

---

### 2. TEXT_FIXES Map

**What:** Three targeted string replacements:

| Find | Replace | Reason | Count in corpus |
|------|---------|--------|----------------:|
| `tʃ` | `ʧ` | Decomposed affricate → ligature | 1 |
| `dʒ` | `ʤ` | Decomposed affricate → ligature | 0 (defensive) |
| `r` | `ɹ` | ASCII r typo → IPA alveolar approximant | 2 |

**Why:** The corpus audit found 2 ASCII `r` characters in 153K rows (labeller typo) and 1 decomposed `tʃ` (labeller's keyboard produced 2 chars instead of the single ligature `ʧ`). If left unfixed, the tokenizer sees them as different phonemes → the model learns a split vocabulary for the same sound. The `dʒ→ʤ` entry is defensive: 0 found in current data, but it could appear in future data from the same labelling pipeline.

**How:** Simple `str.replace()` applied after NFC normalization. Order doesn't matter because the three patterns don't overlap.

---

### 3. Whitespace Squash

**What:** `" ".join(t.split())`

**Why:** Some labels had double spaces or leading/trailing whitespace. Without this, `"ʔə  ʔæpɫ"` and `"ʔə ʔæpɫ"` produce different token sequences. The tokenizer's word delimiter (`|`) replaces single spaces — double spaces would create empty words (adjacent `||` tokens), confusing CTC alignment.

**How:** `.split()` splits on any whitespace (spaces, tabs, newlines) and removes empties; rejoining with single space guarantees uniform formatting.

---

### 4. Drop: Run-on Labels

**What:** If `" " not in text and len(text) > 15` → drop.

**Why:** Some labellers accidentally stripped all spaces from long transcripts. A label like `"ʔəʔæpɫɪzɹɛd"` has no word boundaries — CTC can't align it properly and the model learns garbage alignment paths. These are not single-word utterances (which would be short); they're multi-word transcripts with missing spaces.

**How:** The threshold of 15 characters was chosen because the longest legitimate single-word IPA transcription in child speech is around 12–14 characters. Anything longer without a space is almost certainly a run-on.

**Count:** DS1=23, DS2=350.

---

### 5. Drop: Duration > max_duration (15s)

**What:** If `audio_duration_sec > 15.0` → drop.

**Why:** This is a GPU policy filter, not linguistic filtering. Very long recordings are usually recording artifacts (mic left on), not single child utterances. WavLM self-attention is O(T²) — a 25s utterance produces 1,250 frames and 37.5 MB attention per layer. Reducing to 15s (750 frames) cuts attention VRAM by 2.8×. Additionally, CTC alignment becomes unreliable over long sequences because the number of valid alignment paths grows exponentially with sequence length.

**How:** Simple threshold comparison on the `audio_duration_sec` field from the raw JSONL. 99.48% of utterances are ≤ 15s (only 709 rows / 0.52% dropped, losing 3.50h / 4.7% of hours). The cutoff was chosen from duration distribution analysis: p99 = 12.5s, p95 = 6.4s.

**Count:** DS1=0, DS2=69.

---

### 6. Drop: Mic Click (< 0.1s)

**What:** If `audio_duration_sec < 0.1` → drop.

**Why:** A 0.1s audio file contains ~1,600 samples at 16kHz — barely enough for a single phoneme. Below that threshold, there isn't enough acoustic information for any phoneme. These are mic activation/deactivation artifacts or CNN segmentation failures that produced near-empty clips.

**How:** Simple lower-bound threshold. 0.1s was chosen because the shortest legitimate phoneme burst (a plosive release like /p/) is approximately 50–80ms.

**Count:** DS1=0, DS2=0.

---

### 7. Drop: Physics Filter (TPS < 1 or > 25)

**What:** `tps = n_tokens / duration`. If not in [1.0, 25.0] → drop.

**Why:** This is the most important filter. A human physically cannot articulate more than ~25 IPA tokens per second (even rapid adult speech peaks around 20 TPS). Below 1 TPS means a 5-second recording with fewer than 5 tokens — almost certainly a labelling error (label assigned to wrong audio, or a silence-heavy recording with a brief utterance). Without this filter, the model tries to learn impossible alignments — CTC stretches 3 phonemes over 5 seconds of audio, filling the gaps with blanks and learning nothing useful.

**Note:** The metric counts all IPA characters including suprasegmentals like `ː` (length mark), so it's "tokens per second", not strictly "phonemes per second". This doesn't affect the filter's validity — the thresholds are calibrated to the token-level count. True PPS (excluding suprasegmentals) and SPS (suprasegmentals only) are logged as advisory metrics alongside TPS.

**On true phonemic rate:** Even PPS is not a true phonemic rate — it still counts each Unicode character as one unit. Proper phonemic rate analysis must handle ties (e.g., `t͡ʃ` as one phoneme), affricates-as-one-segment, coarticulation, and language-specific phonological rules. That is a separate linguistic analysis step, not a preprocessing filter. The `data_card.json` output by the tokenizer includes a `rate_metric_note` documenting this distinction.

**How:** `n_tokens` is computed **after** all text normalization (NFC + TEXT_FIXES + whitespace squash), ensuring the count reflects the actual cleaned label. This is critical — if you computed it before normalization, the `tʃ→ʧ` fix would change the character count and the TPS would drift.

**Count:** DS1=101, DS2=815.

---

### 8. Drill Detection

**What:** Flag rows where ≥4 tokens and >50% are duplicates. Sets `is_drill=True`.

**Why:** Speech therapy datasets contain diadochokinetic drills (`"ba ba ba ba"`, `"kwæk kwæk kwæk kwæk"`). These are repetitive articulation exercises, not natural speech. We **flag** them but **keep** them — they're valid training data (the child is still producing real phonemes), but you might want to exclude them later for evaluation, weight them differently in curriculum learning, or analyze them separately.

**How:** `len(set(words)) / len(words) < 0.5` — if more than half the words are duplicates, it's a drill. The ≥4 word minimum prevents false positives on short natural repetitions ("no no" is only 2 words).

**Count:** DS1=61, DS2=115.

---

### 8a. Total Hours Reporting

**What:** Accumulates `audio_duration_sec` for every kept row and prints total hours per dataset + a cross-dataset summary.

**Why:** Gives immediate visibility into data size after cleaning — you need to know if cleaning reduced your corpus from 83 hours to 30 hours. Printed both per-DS and as the `SANITY SUMMARY` block.

**Example output:**

```
--- SANITY SUMMARY ---
[SANITY] DS1     11,919 rows      3.21 hrs
[SANITY] DS2    139,790 rows     78.02 hrs
[SANITY] ALL    151,709 rows     81.23 hrs  (from 153,067 raw)
--- SANITY SUMMARY END ---
```

---

### 8b. Hours by Age Bucket (Post-Clean)

**What:** After cleaning each dataset, prints the duration breakdown per `age_bucket`.

**Why:** Lets you compare **raw imbalance vs usable imbalance** — if cleaning disproportionately drops toddler data, you'll see it here before it silently degrades model performance on young children.

---

### 8c. TPS / PPS / SPS Distributions (Post-Clean)

**What:** Prints TPS (tokens per second) as the primary metric with p01, p50, p99, min, max. Then prints PPS (segments per second, excluding suprasegmentals) and SPS (suprasegmentals per second) as advisory metadata on a second line.

**Why:** TPS drives the physics filter — its distribution shows you how close kept rows are to the [1, 25] boundaries. PPS and SPS split this into segments vs length markers (`ː`) — useful for papers or linguistic analysis. Extremely low TPS = silence-heavy clips. Extremely high TPS = possible mis-alignment or truncated labels. Logging per dataset catches systematic labelling issues early — before they poison CTC alignment.

---

### 9. Health Check (Independent Re-read)

**What:** After writing the output file, re-opens it from disk and re-checks every invariant.

**Why:** The most insidious bugs are silent. You write 139K rows correctly according to your in-memory logic, but maybe the physics filter had an off-by-one, or the JSON serialization dropped a character. The health check opens the **output** file fresh, re-computes every drop condition independently, and prints pass/fail for each. If any violation exists in the output, you know immediately — not 3 days into training when your PER won't converge.

**How:** Streams the output JSONL, checks 8 invariants per row:

- No ASCII `r` in text
- No decomposed `tʃ` in text
- No decomposed `dʒ` in text
- No empty labels
- No duration over max
- No mic clicks
- No run-on labels
- No physics violations

Plus a row count match: the number of rows in the file must equal the count from the write pass.

**Result:** 18/18 checks green (9 per dataset × 2 datasets).

---

### 10. Label Policy Audit (Cross-Dataset)

**What:** After all datasets are cleaned, compares their post-normalisation character inventories side-by-side. Reports shared vs exclusive characters, counts combining diacritics (Unicode `Mn`/`Mc`) and modifier letters (category `Lm`, excluding known suprasegmentals), and computes a "narrow transcription density" per dataset.

**Why:** Even when every dataset passes the same text fixes and filters, they might come from labellers using different transcription philosophies. DS1 might use broad IPA (`k æ t`) while DS2 uses narrow IPA with diacritics (`kʰ æ t̚`). Both are valid IPA — the vocab builds fine, the health check passes — but CTC learns inconsistent targets. One dataset teaches "always predict `k`", the other teaches "`k` + aspiration modifier" for the same sound. The model can never reconcile this and wastes capacity.

**How:**

1. During cleaning, a per-dataset `Counter` accumulates every non-space character from kept rows.
2. After the SANITY SUMMARY, an audit block runs if ≥2 datasets exist:
   - **Shared vs union inventory** — how many characters appear in ALL datasets vs any dataset.
   - **Exclusive characters** — characters that appear in one dataset only. Logged with Unicode codepoint, name, and count. A `ʰ` (aspirated modifier) appearing only in DS2 is a hard signal.
   - **Narrow-transcription density** — for each dataset: `(combining_marks + modifier_letters) / total_tokens × 1000`. Combining marks (`unicodedata.combining(ch) > 0`) catch diacritics like nasalisation `̃`, unreleased `̚`, dental `̪`. Modifier letters (category `Lm`, excluding `ː`) catch aspiration `ʰ`, labialisation `ʷ`, palatalisation `ʲ`.
   - **Verdict**: `CONSISTENT` if no exclusive chars and density spread ≤ 0.5/1k tokens. `REVIEW NEEDED` otherwise.

**Example output:**

```
--- LABEL POLICY AUDIT ---
[POLICY] shared inventory      50 chars
[POLICY] union  inventory      50 chars
[POLICY]   (no exclusive chars — identical inventories)
[POLICY] DS1  tokens=   118,942  combining=     0  modifier=     0  narrow_density=0.00/1k
[POLICY] DS2  tokens= 1,372,218  combining=     0  modifier=     0  narrow_density=0.00/1k
[POLICY] verdict: CONSISTENT  (density_spread=0.00/1k, no exclusive chars)
--- LABEL POLICY AUDIT END ---
```

**Not a filter:** This is audit-only — no rows are dropped. It produces evidence that both datasets use the same label philosophy (broad transcription, no combining diacritics, no modifier letters). If a future dataset introduces narrow transcription, the audit will flag it before it poisons training.

---

## Output Schema

Each row in the output JSONL contains:

```json
{
  "utterance_id": "U_...",
  "child_id": "C_...",
  "session_id": "S_...",
  "audio_path": "audio/U_....flac",
  "audio_duration_sec": 1.435,
  "age_bucket": "3-4",
  "md5_hash": "...",
  "filesize_bytes": 121365,
  "phonetic_text": "ʔə ʔæpɫ",
  "n_phonemes": 6,
  "is_drill": false
}
```

**Key fields added by EDA:**

- `phonetic_text` — normalized (NFC + TEXT_FIXES + whitespace squash)
- `n_phonemes` — character count excluding spaces (computed after normalization)
- `is_drill` — boolean flag for diadochokinetic drills

---

## Decision Evolution — Mar 2026 Update

Historical decisions above remain valid. Current files show the following evolution after full ETL reruns.

| Earlier baseline in this doc | Current files now | Why this moved |
|---|---|---|
| Cleaned total after sanity: 151,709 rows | Processed transcripts now: 150,928 rows (`11,910` + `139,018`) | Post-sanity audio-level removals from EDA (1 load failure + 3 duration mismatches) reduced final usable rows.
| Duration profile references older p99 (~12.5s pre-final cleanup context) | Current tokenizer fingerprint p99 = `10.848s` | After final cleaning + anomaly removals, long-tail shrank; this changed downstream batch planning.
| Sanity as text-first cleaning stage | Still unchanged | This was the correct separation of concerns: text validity first, audio integrity refinement later.

**Current rationale:** we did not weaken sanity rules; we added downstream evidence that a two-layer filter (text + audio) yields cleaner manifests for training stability.
