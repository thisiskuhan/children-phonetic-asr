#!/usr/bin/env python3
"""Build character-level trigram LM from gold training transcripts.

Outputs a flat binary file (V×V×V float32 log-probs) for use in
the C beam search decoder.  V=53, so the table is ~581 KB.

BOS (beginning-of-sequence) context uses token id 0 (PAD) since
PAD is never emitted by the model.

Smoothing: Jelinek-Mercer interpolation of trigram/bigram/unigram
with add-alpha smoothing at each level.

Usage:
    python build_lm.py --data data/raw --vocab submission/src/tokenizer/vocab.json \
                       --output submission/src/lm_trigram.bin
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_transcripts(data_dir: Path) -> list[str]:
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


def build_trigram_table(
    texts: list[str],
    char2id: dict[str, int],
    V: int,
    alpha: float = 0.001,
    lam3: float = 0.7,
    lam2: float = 0.2,
    lam1: float = 0.1,
) -> np.ndarray:
    """Build interpolated trigram log-prob table.

    P(c|a,b) = λ3·P_α(c|a,b) + λ2·P_α(c|b) + λ1·P_α(c)

    BOS context uses id=0 (PAD). UNK (id=1) gets -100 log-prob.
    """
    BOS = 0
    UNK = 1
    word_sep = char2id.get("|", -1)

    # -- Count n-grams --
    uni: Counter = Counter()
    bi: Counter = Counter()
    tri: Counter = Counter()

    for text in texts:
        ids = []
        for ch in text:
            if ch == " ":
                if word_sep >= 0:
                    ids.append(word_sep)
            elif ch in char2id:
                ids.append(char2id[ch])
            # skip unknown chars

        if not ids:
            continue

        seq = [BOS, BOS] + ids
        for i in range(2, len(seq)):
            c, b, a = seq[i], seq[i - 1], seq[i - 2]
            uni[c] += 1
            bi[(b, c)] += 1
            tri[(a, b, c)] += 1

    total = sum(uni.values())
    print(
        f"Corpus: {total:,} tokens, "
        f"{len(uni)} unigrams, {len(bi):,} bigrams, {len(tri):,} trigrams"
    )

    # -- Context totals --
    bi_ctx: Counter = Counter()
    for (b, _c), cnt in bi.items():
        bi_ctx[b] += cnt

    tri_ctx: Counter = Counter()
    for (a, b, _c), cnt in tri.items():
        tri_ctx[(a, b)] += cnt

    # -- Build table --
    table = np.full((V, V, V), -20.0, dtype=np.float32)

    for a in range(V):
        for b in range(V):
            for c in range(V):
                if c == 0 or c == UNK:
                    table[a, b, c] = -100.0
                    continue

                # Unigram
                p1 = (uni.get(c, 0) + alpha) / (total + alpha * V)

                # Bigram
                bt = bi_ctx.get(b, 0)
                p2 = (bi.get((b, c), 0) + alpha) / (bt + alpha * V) if bt > 0 else p1

                # Trigram
                tt = tri_ctx.get((a, b), 0)
                p3 = (tri.get((a, b, c), 0) + alpha) / (tt + alpha * V) if tt > 0 else p2

                p = lam3 * p3 + lam2 * p2 + lam1 * p1
                table[a, b, c] = math.log(max(p, 1e-30))

    return table


def main():
    p = argparse.ArgumentParser(description="Build character trigram LM")
    p.add_argument("--data", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--vocab", type=Path, default=Path("submission/src/tokenizer/vocab.json")
    )
    p.add_argument("--output", type=Path, default=Path("submission/src/lm_trigram.bin"))
    p.add_argument("--alpha", type=float, default=0.001)
    args = p.parse_args()

    with open(args.vocab) as f:
        vocab = json.load(f)
    V = max(vocab.values()) + 1
    print(f"Vocab size: {V}")

    texts = load_transcripts(args.data)
    if not texts:
        print("ERROR: no transcripts found", file=sys.stderr)
        sys.exit(1)

    table = build_trigram_table(texts, vocab, V, alpha=args.alpha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.tofile(str(args.output))

    size_kb = args.output.stat().st_size / 1024
    print(f"Saved {args.output} ({size_kb:.0f} KB)")

    # Sanity: show BOS→BOS distribution
    id2char = {v: k for k, v in vocab.items()}
    print(f"\nTop-10 P(token | BOS, BOS):")
    bos_probs = [(c, table[0, 0, c]) for c in range(V) if c >= 2]
    bos_probs.sort(key=lambda x: -x[1])
    for c, lp in bos_probs[:10]:
        print(f"  P({id2char.get(c, '?'):3s}) = {math.exp(lp):.4f}  (log={lp:.2f})")


if __name__ == "__main__":
    main()
