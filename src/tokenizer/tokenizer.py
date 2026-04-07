import logging
import unicodedata
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer

from utils import SUPRASEGMENTALS, nearest_rank_pctl, loads, dumps

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical IPA inventory — used to validate vocab at build time.
# Source: IPA chart 2015 + IPA Extensions block + clinical extensions.
# ʧ (tʃ) and ʤ (dʒ) are single-codepoint affricates (U+02A7, U+02A4).
# ---------------------------------------------------------------------------
_IPA_CHARS: frozenset[str] = frozenset(
    # ---- Pulmonic consonants ----
    "p b t d ʈ ɖ c ɟ k ɡ q ɢ ʔ "    # stops  (incl. script-g ɡ)
    "m ɱ n ɳ ɲ ŋ ɴ "              # nasals
    "ʙ r ʀ "                          # trills
    "ɾ ɽ "                          # taps / flaps
    "ɸ β f v θ ð s z ʃ ʒ "         # fricatives (part 1)
    "ʂ ʐ ç ʝ x ɣ χ ʁ ħ ʕ h ɦ "   # fricatives (part 2)
    "ɬ ɮ "                          # lateral fricatives
    "ʋ ɹ ɻ j ɰ "                  # central approximants
    "l ɭ ʎ ʟ "                      # lateral approximants
    # ---- Non-pulmonic consonants ----
    "ʘ ǀ ǁ ǂ ǃ "               # clicks
    "ɓ ɗ ʄ ɠ ʛ "               # voiced implosives
    # ---- Vowels ----
    "i y ɨ ʉ ɯ u "                  # close
    "ɪ ʏ ʊ "                        # near-close
    "e ø ɘ ɵ ɤ o "                  # close-mid
    "ə "                             # mid (schwa)
    "ɛ œ ɜ ɞ ʌ ɔ "              # open-mid
    "æ ɐ "                          # near-open
    "a ɶ ɑ ɒ "                      # open
    # ---- Extended / clinical IPA ----
    "ɚ ɝ "                          # r-coloured schwa / open-mid
    "ɫ "                             # velarized l
    "ʧ ʤ "                          # single-token affricates tʃ dʒ
    # ---- Suprasegmentals ----
    "ː ˑ ˈ ˌ "                    # length markers, stress
    # ---- Diacritics used as spacing chars in some transcriptions ----
    "ʰ ʷ ʲ "                        # aspiration, labialization, palatalization
    # ASCII letters that ARE in the IPA chart as primary symbols
    "g w "                           # ASCII g (alongside script-g ɡ); w = labial-velar approximant
    .split()
)

# Normalisation fixes — must match eda_processor.TEXT_FIXES exactly.
# Duplicated here (3 entries) to avoid importing the heavy EDA module.
_LABEL_FIXES: dict[str, str] = {"tʃ": "ʧ", "dʒ": "ʤ", "r": "ɹ"}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_ctc_tokenizer(
    clean_jsonl_paths: list[str],
    output_dir: str,
    *,
    cfg: dict | None = None,
) -> Wav2Vec2CTCTokenizer:
    """
    Build and persist a deterministic CTC tokenizer from cleaned JSONL data.

    Steps:
      1. Stream all files → collect chars, counts, and corpus stats
      2. Sort chars for reproducible IDs across machines/runs
      3. Anchor special tokens: [PAD]=0  [UNK]=1  |=2
      4. Map corpus chars from ID 3 onward
      5. Save vocab.json
      6. Instantiate Wav2Vec2CTCTokenizer (no BOS/EOS)
      7. Per-char OOV assertion — zero silent failures
      8. save_pretrained + inventory.txt
      9. Linguistic audit — flag suspicious Latin letters
     10. IPA validation — assert all chars are in canonical IPA inventory
     11. Persist metadata: fingerprint.json, phoneme_freq.tsv, data_card.json

    Notes:
      - Space is NOT a vocab token; | is the word delimiter (swapped internally).
      - BOS/EOS set to None — prevents HF from auto-appending <s>/<s> as added
        tokens (IDs 53-54). Keeps vocab locked at exactly 53.
      - After training: model.config.pad_token_id = tokenizer.pad_token_id
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Deterministic source ordering — sorted ensures fingerprint stability
    # even if config list order changes.
    clean_jsonl_paths = sorted(clean_jsonl_paths)

    # 1. Stream processed JSONL — collect chars, counts, and corpus stats
    char_counts: dict[str, int] = {}
    char_examples: dict[str, list[str]] = {}  # up to 5 example transcripts per char
    _EXAMPLE_LIMIT = 5
    has_space = False
    durations: list[float] = []
    tps_values: list[float] = []
    pps_values: list[float] = []
    sps_values: list[float] = []
    n_phonemes_list: list[int] = []
    rows_by_source: dict[str, int] = {}
    drills_by_source: dict[str, int] = {}

    for path in clean_jsonl_paths:
        src = Path(path).name
        rows_by_source[src] = 0
        drills_by_source[src] = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = loads(line)
                text = row["phonetic_text"]
                dur = row["audio_duration_sec"]
                n_ph = row["n_phonemes"]

                rows_by_source[src] += 1
                if row.get("is_drill", False):
                    drills_by_source[src] += 1

                durations.append(dur)
                n_phonemes_list.append(n_ph)
                if dur > 0:
                    tps_values.append(n_ph / dur)
                    n_supra = sum(1 for ch in text if ch in SUPRASEGMENTALS)
                    pps_values.append((n_ph - n_supra) / dur)
                    sps_values.append(n_supra / dur)

                if not has_space and " " in text:
                    has_space = True
                for ch in text:
                    if ch != " ":
                        char_counts[ch] = char_counts.get(ch, 0) + 1
                        # Collect example transcripts (up to limit)
                        if ch not in char_examples:
                            char_examples[ch] = []
                        if len(char_examples[ch]) < _EXAMPLE_LIMIT:
                            char_examples[ch].append(text)
    unique_chars = set(char_counts)

    # ---- Include labels from EDA-removed rows in vocab ----
    # Competition host intended all rows.  Rows with broken audio still have
    # valid phonemic labels — vocab must cover the full intended inventory.
    eda_n_removed = 0
    eda_exclusive: list[str] = []
    eda_parts: list[str] = []
    if cfg is not None:
        eda_chars, eda_n_removed, eda_n_load, eda_n_dur = _collect_eda_removed_labels(cfg)
        if eda_chars:
            eda_exclusive = sorted(
                [ch for ch in eda_chars if ch not in unique_chars], key=ord
            )
            for ch, cnt in eda_chars.items():
                char_counts[ch] = char_counts.get(ch, 0) + cnt
                unique_chars.add(ch)
            if eda_n_load:
                eda_parts.append(f"{eda_n_load} load fail")
            if eda_n_dur:
                eda_parts.append(f"{eda_n_dur} dur mismatch")

    # Empty-corpus guard — catch upstream wipe before building a useless vocab
    assert durations, "FATAL: no rows survived cleaning — check EDA processor output"

    # Pre-build guards — catch label corruption before poisoning the vocab
    assert has_space, "FATAL: no spaces found in labels — word_delimiter_token won't work"
    assert "|" not in unique_chars, "FATAL: '|' present in raw labels — would collide with delimiter"
    assert tps_values, "FATAL: no valid TPS values — every row had dur==0 (check upstream filter)"
    assert n_phonemes_list, "FATAL: no phoneme counts collected — empty corpus"

    # IPA consistency guard — ASCII g (U+0067) vs script-g ɡ (U+0261)
    # Both mapping to the same phoneme corrupts training targets — must be fatal.
    assert not ("g" in unique_chars and "\u0261" in unique_chars), (
        "FATAL: both ASCII 'g' (U+0067) and IPA 'ɡ' (U+0261) in corpus — "
        "same phoneme with two symbols corrupts CTC targets.  "
        "Normalise upstream in TEXT_FIXES before re-running."
    )

    # 2. Sorted → deterministic token IDs
    vocab_list = sorted(unique_chars)
    assert max(len(ch) for ch in vocab_list) == 1, "FATAL: multi-character token slipped into vocab"

    # 3 & 4. Build vocab dict with anchored specials
    vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "|": 2}
    for i, ch in enumerate(vocab_list, start=3):
        vocab[ch] = i

    # 5. Save vocab.json
    vocab_file = out / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        f.write(dumps(vocab, indent=2))

    # 6. Instantiate tokenizer — bos/eos=None prevents HF from appending <s>/<s>
    #    as added_tokens (IDs 53-54), keeping vocab locked at 53.
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",  # spaces in text → | IDs internally
        bos_token=None,
        eos_token=None,
    )

    # 7. Per-character OOV check — convert_tokens_to_ids avoids __call__ normalization
    for ch in vocab_list:
        assert tokenizer.convert_tokens_to_ids(ch) != tokenizer.unk_token_id, \
            f"FATAL: character '{ch}' mapped to [UNK]"

    # 8. Persist full HF config + plain-text inventory for debugging
    tokenizer.save_pretrained(output_dir)
    (out / "inventory.txt").write_text("\n".join(vocab_list), encoding="utf-8")

    log.info("")
    log.info("--- TOKENIZER START ---")
    log.info(f"[TOK] sources     {clean_jsonl_paths}")
    n_training = sum(rows_by_source.values())
    if eda_n_removed:
        log.info(f"[TOK] training rows  {n_training:,}")
        log.info(f"[TOK] +EDA labels    {eda_n_removed}  "
                 f"({', '.join(eda_parts)} — audio broken, labels valid)")
        log.info(f"[TOK] vocab from     {n_training + eda_n_removed:,} labels")
        if eda_exclusive:
            log.warning(f"[TOK] RESCUED     {len(eda_exclusive)} exclusive char(s): "
                        f"{' '.join(eda_exclusive)}")
        else:
            log.info(f"[TOK] coverage    all removed labels' chars in training corpus")
    else:
        log.info(f"[TOK] rows        {n_training:,}")
    log.info(f"[TOK] vocab size  {len(vocab)}   ({len(vocab_list)} chars + 3 specials)")
    log.info(f"[TOK] specials    {tokenizer.all_special_tokens}")
    log.info(f"[TOK] blank id    {tokenizer.pad_token_id}")

    # CTC blank must be ID 0 — HF default can shift if specials are reordered
    assert tokenizer.pad_token_id == 0, (
        f"FATAL: pad_token_id={tokenizer.pad_token_id}, expected 0.  "
        f"CTC blank *must* be ID 0 in Wav2Vec2.  Check vocab.json ordering."
    )

    log.info(f"[TOK] OOV check   PASSED  ({len(vocab_list)} chars verified)")
    log.info(f"[TOK] saved to    {output_dir}/")

    # 9. Linguistic audit — flag Latin letters that could be grapheme leaks
    #    Expected: all are valid narrow IPA in child-speech clinical data
    LATIN_WATCH = list("cejowxz")
    found = {ch: char_counts.get(ch, 0) for ch in LATIN_WATCH if ch in char_counts}
    if found:
        log.info("")
        log.info("--- LINGUISTIC AUDIT ---")
        for ch, cnt in sorted(found.items()):
            log.info(f"[AUDIT] '{ch}'  count={cnt:>8,}  (verify: IPA, not grapheme)")
        missing = [ch for ch in LATIN_WATCH if ch not in char_counts]
        if missing:
            log.info(f"[AUDIT] absent   {missing}")
        log.info("--- LINGUISTIC AUDIT END ---")

    # 10. IPA validation — every vocab char must be in the canonical IPA inventory
    #     (note: step 11 = metadata persistence below)
    non_ipa = [ch for ch in vocab_list if ch not in _IPA_CHARS]
    log.info("")
    log.info("--- IPA CHECK ---")
    if non_ipa:
        for ch in non_ipa:
            u = ord(ch)
            log.info(f"[IPA]   ✗  U+{u:04X}  '{ch}'  NOT in IPA inventory")
        # Fatal — non-IPA symbols in a phoneme vocab means either grapheme leak
        # or missing entry in _IPA_CHARS.  Do not silently train on bad labels.
        assert False, (
            f"FATAL: {len(non_ipa)} non-IPA token(s) in vocab: "
            f"{non_ipa!r}.  Fix upstream labels or update _IPA_CHARS."
        )
    else:   
        log.info(f"[IPA]   ✓  all {len(vocab_list)} chars confirmed IPA")
    log.info("--- IPA CHECK END ---")

    log.info("--- TOKENIZER END ---")

    # ---------------------------------------------------------------------------
    # 11. Persist metadata — reproducibility + downstream analysis
    # ---------------------------------------------------------------------------

    # ---- dataset_fingerprint.json ----
    total_rows = sum(rows_by_source.values())
    total_hrs = sum(durations) / 3600
    # In-place sorts — avoids allocating 5 duplicate 151 k-element lists
    durations.sort();        dur_sorted = durations
    tps_values.sort();       tps_sorted = tps_values
    pps_values.sort();       pps_sorted = pps_values
    sps_values.sort();       sps_sorted = sps_values
    n_phonemes_list.sort();  nph_sorted = n_phonemes_list

    fingerprint = {
        "generated_by": "tokenizer.py",
        "sources": clean_jsonl_paths,
        "rows_kept": rows_by_source,
        "rows_total": total_rows,
        "eda_removed_labels_in_vocab": eda_n_removed,
        "drills_kept": drills_by_source,
        "total_duration_hours": round(total_hrs, 2),
        "duration_stats": {
            "min": dur_sorted[0],
            "p01": nearest_rank_pctl(dur_sorted, 1),
            "p50": nearest_rank_pctl(dur_sorted, 50),
            "p99": nearest_rank_pctl(dur_sorted, 99),
            "max": dur_sorted[-1],
        },
        "tps_stats": {
            "min": round(tps_sorted[0], 2),
            "p01": nearest_rank_pctl(tps_sorted, 1),
            "p50": nearest_rank_pctl(tps_sorted, 50),
            "p99": nearest_rank_pctl(tps_sorted, 99),
            "max": round(tps_sorted[-1], 2),
        },
        "pps_stats": {
            "note": "segments only (excludes suprasegmentals)",
            "p50": nearest_rank_pctl(pps_sorted, 50),
            "p99": nearest_rank_pctl(pps_sorted, 99),
        },
        "sps_stats": {
            "note": "suprasegmentals only (ː)",
            "p50": nearest_rank_pctl(sps_sorted, 50),
            "p99": nearest_rank_pctl(sps_sorted, 99),
        },
        "transcript_length_stats": {
            "min": nph_sorted[0],
            "p01": nearest_rank_pctl(nph_sorted, 1),
            "p50": nearest_rank_pctl(nph_sorted, 50),
            "p99": nearest_rank_pctl(nph_sorted, 99),
            "max": nph_sorted[-1],
        },
        "phoneme_inventory": vocab_list,
        "vocab_size": len(vocab),
        "special_tokens": {
            "pad": "[PAD]",
            "unk": "[UNK]",
            "word_delimiter": "|",
        },
    }
    fp_path = out / "dataset_fingerprint.json"
    with open(fp_path, "w", encoding="utf-8") as f:
        f.write(dumps(fingerprint, indent=2))

    # ---- phoneme_freq.tsv ----
    freq_path = out / "phoneme_freq.tsv"
    rare_phonemes: list[tuple[str, int, float]] = []  # (char, count, pct)
    with open(freq_path, "w", encoding="utf-8") as f:
        f.write("phoneme\tcount\tpercent\n")
        total_chars = sum(char_counts.values())
        for ch in vocab_list:                               # sorted order
            cnt = char_counts[ch]
            pct = round(cnt / total_chars * 100, 4)
            f.write(f"{ch}\t{cnt}\t{pct}\n")
            if pct < 0.1:
                rare_phonemes.append((ch, cnt, pct))

    # ---- Rare phoneme audit (research inspection) ----
    if rare_phonemes:
        log.info("")
        log.info("--- RARE PHONEME AUDIT (<0.1% frequency) ---")
        for ch, cnt, pct in rare_phonemes:
            u = ord(ch)
            log.info(f"[RARE] U+{u:04X}  '{ch}'  count={cnt:>6,}  ({pct:.4f}%)")
            examples = char_examples.get(ch, [])
            for i, ex in enumerate(examples[:5], 1):
                # Truncate long transcripts for display
                display = ex if len(ex) <= 60 else ex[:57] + "..."
                log.info(f"[RARE]   {i}. {display}")
        log.info("--- RARE PHONEME AUDIT END ---")

    # ---- data_card.json ----
    data_card = {
        "label_scheme": "narrow_phonemic",
        "affricates": "single_token (ʧ ʤ)",
        "r_symbol": "ɹ",
        "space_delimiter": True,
        "word_delimiter_token": "|",
        "ctc_blank": "[PAD]=0",
        "unicode_normalization": "NFC",
        "text_fixes_applied": {"tʃ": "ʧ", "dʒ": "ʤ", "r": "ɹ"},
        "duration_filter_sec": f"{dur_sorted[0]}–{dur_sorted[-1]}",
        "tps_filter": f"{round(tps_sorted[0], 1)}–{round(tps_sorted[-1], 1)}",
        "rate_metric_note": (
            "TPS counts all IPA characters (including ː). "
            "PPS excludes suprasegmentals. "
            "True phonemic rate (handling ties, affricates-as-one-segment, "
            "coarticulation) requires separate linguistic analysis."
        ),
    }
    dc_path = out / "data_card.json"
    with open(dc_path, "w", encoding="utf-8") as f:
        f.write(dumps(data_card, indent=2))

    return tokenizer


# ---------------------------------------------------------------------------
# EDA removal audit
# ---------------------------------------------------------------------------

def _collect_eda_removed_labels(
    cfg: dict,
) -> tuple[dict[str, int], int, int, int]:
    """
    Collect phoneme chars from rows that Audio EDA removed (bad audio).

    The competition host intended all rows.  Broken audio files still carry
    valid phonemic labels whose characters belong in the vocab so the model
    can recognise the full intended phoneme space.

    Returns
    -------
    char_counts : dict[str, int]
        Character → count across the removed rows' labels.
    n_removed : int
        Total removed rows.
    n_load : int
        Rows removed due to load failure.
    n_dur : int
        Rows removed due to duration mismatch.
    """
    reports = Path(cfg["paths"]["reports"])

    removed_ids: set[str] = set()
    n_load = n_dur = 0
    for name, tag in (("eda_failed_loads.json", "load"),
                      ("eda_duration_mismatch.json", "dur")):
        p = reports / name
        if not p.exists():
            continue
        ids = loads(p.read_bytes())["utterance_ids"]
        removed_ids.update(ids)
        if tag == "load":
            n_load = len(ids)
        else:
            n_dur = len(ids)

    if not removed_ids:
        return {}, 0, 0, 0

    # Scan raw transcripts for the removed rows' labels
    char_counts: dict[str, int] = {}
    found = 0
    for key in cfg["datasets"]:
        raw_path = Path(cfg["paths"]["datasets"][key])
        if not raw_path.exists():
            continue
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                r = loads(line)
                if r["utterance_id"] not in removed_ids:
                    continue
                t = unicodedata.normalize("NFC", r["phonetic_text"])
                for src, tgt in _LABEL_FIXES.items():
                    t = t.replace(src, tgt)
                for ch in t:
                    if ch != " ":
                        char_counts[ch] = char_counts.get(ch, 0) + 1
                found += 1
                if found == len(removed_ids):
                    break
        if found == len(removed_ids):
            break

    return char_counts, len(removed_ids), n_load, n_dur


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        paths = cfg["paths"]
        processed = paths["processed"]
        self.paths = [f"{processed}/{k}_transcript.jsonl" for k in cfg["datasets"]]
        self.output_dir = paths["tokenizer"]

    def run(self) -> Wav2Vec2CTCTokenizer:
        return build_ctc_tokenizer(self.paths, self.output_dir, cfg=self._cfg)

