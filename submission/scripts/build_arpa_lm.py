#!/usr/bin/env python3
"""Build a KenLM-compatible ARPA n-gram LM from gold training phonetic transcripts.

Outputs a proper ARPA format file that can be loaded by kenlm.Model() and used
with pyctcdecode for CTC beam search.

Each IPA character is treated as a separate "word" for the LM.
Space (word boundary) → "|" token (matching our CTC vocab).

Uses Kneser-Ney-inspired back-off with Witten-Bell smoothing.

Usage:
    python build_arpa_lm.py --data data/raw --vocab submission/src/tokenizer/vocab.json \
                            --output submission/src/kenlm_5gram.arpa --order 5
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_transcripts(data_dir: Path) -> list[str]:
    """Load phonetic transcripts from training JSONL files."""
    texts = []
    for f in sorted(data_dir.glob("*_train_phon_transcripts.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t = rec.get("phonetic_text", "").strip()
                if t:
                    texts.append(t)
    print(f"Loaded {len(texts)} transcripts from {data_dir}")
    return texts


def text_to_tokens(text: str, char2id: dict) -> list[str]:
    """Convert phonetic text to list of token strings (characters)."""
    tokens = []
    for ch in text:
        if ch == " ":
            tokens.append("|")
        elif ch in char2id:
            tokens.append(ch)
    return tokens


def build_ngram_counts(texts: list[str], char2id: dict, order: int):
    """Count n-grams up to the given order.

    Returns:
        counts: list of Counters, counts[n] has n-gram counts (n=1..order)
        total_tokens: total number of tokens seen
    """
    counts = [None]  # index 0 unused
    for n in range(1, order + 1):
        counts.append(Counter())

    total_tokens = 0

    for text in texts:
        tokens = text_to_tokens(text, char2id)
        if not tokens:
            continue

        # Prepend <s> and append </s>
        seq = ["<s>"] + tokens + ["</s>"]
        total_tokens += len(tokens)

        for n in range(1, order + 1):
            for i in range(len(seq) - n + 1):
                ngram = tuple(seq[i : i + n])
                counts[n][ngram] += 1

    print(f"Total tokens: {total_tokens:,}")
    for n in range(1, order + 1):
        print(f"  {n}-grams: {len(counts[n]):,} unique")

    return counts, total_tokens


def compute_arpa_model(counts, order: int, total_tokens: int):
    """Compute log10 probabilities and back-off weights using Witten-Bell smoothing.

    Returns:
        ngram_probs: list of dicts, ngram_probs[n] = {ngram: (log10_prob, log10_bow)}
    """
    ngram_probs = [None]  # index 0 unused

    # --- Unigrams ---
    uni = counts[1]
    total = sum(uni.values())
    n_types = len(uni)

    # Witten-Bell: P(w) = C(w) / (N + T), where T = number of types
    # Back-off mass = T / (N + T)
    uni_probs = {}
    for ngram, cnt in uni.items():
        p = cnt / (total + n_types)
        uni_probs[ngram] = (math.log10(max(p, 1e-20)), 0.0)  # no back-off for unigrams

    ngram_probs.append(uni_probs)

    # --- Higher-order n-grams ---
    for n in range(2, order + 1):
        cur_counts = counts[n]
        prev_counts = counts[n - 1]

        # Group by context (prefix)
        context_total = defaultdict(int)     # context -> total count
        context_types = defaultdict(int)     # context -> number of unique continuations
        context_words = defaultdict(dict)    # context -> {word: count}

        for ngram, cnt in cur_counts.items():
            context = ngram[:-1]
            word = ngram[-1]
            context_total[context] += cnt
            context_words[context][word] = cnt

        for context in context_words:
            context_types[context] = len(context_words[context])

        # Compute probs and back-off weights
        cur_probs = {}
        for ngram, cnt in cur_counts.items():
            context = ngram[:-1]
            ct = context_total[context]
            nt = context_types[context]
            # Witten-Bell: P(w|ctx) = C(ctx w) / (C(ctx) + T(ctx))
            p = cnt / (ct + nt)
            log_p = math.log10(max(p, 1e-20))
            cur_probs[ngram] = (log_p, 0.0)

        # Compute back-off weights for previous order
        # bow(ctx) = (1 - sum_observed P(w|ctx)) / (1 - sum_observed P_lower(w))
        if n - 1 >= 1:
            prev = ngram_probs[n - 1]
            new_prev = {}
            for prev_ngram, (prev_lp, _) in prev.items():
                ctx = prev_ngram  # this n-1 gram IS a context for order n
                if ctx in context_words:
                    ct = context_total[ctx]
                    nt = context_types[ctx]
                    # Mass assigned to observed = ct / (ct + nt)
                    p_observed = ct / (ct + nt)
                    # Back-off mass
                    bow_mass = 1.0 - p_observed

                    # sum of lower-order probs for observed words
                    sum_lower = 0.0
                    for word in context_words[ctx]:
                        lower_ngram = ctx[1:] + (word,) if n - 1 > 1 else (word,)
                        if lower_ngram in prev:
                            sum_lower += 10 ** prev[lower_ngram][0]
                        else:
                            # try even lower
                            sum_lower += 1e-10

                    denom = max(1.0 - sum_lower, 1e-20)
                    bow = bow_mass / denom
                    log_bow = math.log10(max(bow, 1e-20))
                else:
                    log_bow = 0.0  # no back-off needed

                new_prev[prev_ngram] = (prev_lp, log_bow)
            ngram_probs[n - 1] = new_prev

        ngram_probs.append(cur_probs)

    return ngram_probs


def write_arpa(ngram_probs, order: int, output: Path):
    """Write ARPA format file."""
    with open(output, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        for n in range(1, order + 1):
            f.write(f"ngram {n}={len(ngram_probs[n])}\n")
        f.write("\n")

        for n in range(1, order + 1):
            f.write(f"\\{n}-grams:\n")

            # Sort for reproducibility
            items = sorted(ngram_probs[n].items(), key=lambda x: x[0])

            for ngram, (log_p, log_bow) in items:
                ngram_str = " ".join(ngram)
                if n < order and log_bow != 0.0:
                    f.write(f"{log_p:.6f}\t{ngram_str}\t{log_bow:.6f}\n")
                else:
                    f.write(f"{log_p:.6f}\t{ngram_str}\n")

            f.write("\n")

        f.write("\\end\\\n")


def main():
    p = argparse.ArgumentParser(description="Build KenLM ARPA character LM")
    p.add_argument("--data", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--vocab", type=Path, default=Path("submission/src/tokenizer/vocab.json")
    )
    p.add_argument(
        "--output", type=Path, default=Path("submission/src/kenlm_5gram.arpa")
    )
    p.add_argument("--order", type=int, default=5, help="N-gram order (default: 5)")
    args = p.parse_args()

    # Load vocab
    with open(args.vocab) as f:
        vocab = json.load(f)
    char2id = {k: v for k, v in vocab.items() if v >= 2}  # skip PAD, UNK
    print(f"Vocab: {len(char2id)} tokens (excluding PAD/UNK)")

    # Load transcripts
    texts = load_transcripts(args.data)
    if not texts:
        print("ERROR: no transcripts found", file=sys.stderr)
        sys.exit(1)

    # Build n-gram counts
    counts, total_tokens = build_ngram_counts(texts, char2id, args.order)

    # Compute ARPA probabilities
    print("Computing ARPA probabilities...")
    ngram_probs = compute_arpa_model(counts, args.order, total_tokens)

    # Write ARPA
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_arpa(ngram_probs, args.order, args.output)

    size_kb = args.output.stat().st_size / 1024
    print(f"\nSaved {args.output} ({size_kb:.0f} KB)")

    # Validate with kenlm
    try:
        import kenlm

        model = kenlm.Model(str(args.output))
        print(f"KenLM loaded OK: order={model.order}")

        # Test a few sequences
        test_seqs = ["ð ə", "h ɛ l oʊ", "k æ t"]
        for seq in test_seqs:
            score = model.score(seq, bos=True, eos=True)
            print(f"  score('{seq}') = {score:.2f}")
    except Exception as e:
        print(f"WARNING: KenLM validation failed: {e}")
        print("The ARPA file may need manual inspection.")


if __name__ == "__main__":
    main()
