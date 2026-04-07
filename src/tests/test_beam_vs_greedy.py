#!/usr/bin/env python3
"""Compare greedy vs beam search PER on sft_val.jsonl.

Usage:
    python src/tests/test_beam_vs_greedy.py [--n 200] [--beam 25] [--topk 15]
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from transformers import WavLMConfig, WavLMForCTC, Wav2Vec2CTCTokenizer

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLANK = 0
NEG_INF = float("-inf")


# ── PER (Levenshtein) ────────────────────────────────────────────
def _edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[lb]


def corpus_per(hyps, refs):
    edits = sum(_edit_distance(h, r) for h, r in zip(hyps, refs))
    total = sum(len(r) for r in refs)
    return edits / total if total else 0.0


# ── Audio ─────────────────────────────────────────────────────────
def load_audio(path):
    data, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    std = wav.std()
    if std > 1e-8:
        wav = (wav - wav.mean()) / std
    return wav


# ── Greedy decode ─────────────────────────────────────────────────
def greedy_decode(logits, output_len):
    """Single-sample greedy CTC decode."""
    ids = logits[:output_len].argmax(dim=-1)
    prev = -1
    result = []
    for tok in ids.tolist():
        if tok != 0 and tok != 1 and tok != prev:
            result.append(tok)
        prev = tok
    return result


# ── Beam search (copied from main_beam.py) ───────────────────────
def _logaddexp(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_beam_search(log_probs, beam_width=25, top_k=15, blank=0):
    T, V = log_probs.shape
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        frame = log_probs[t]
        if 0 < top_k < V:
            top_indices = np.argpartition(frame, -top_k)[-top_k:]
            if blank not in top_indices:
                top_indices = np.append(top_indices, blank)
        else:
            top_indices = np.arange(V)

        new_beams = defaultdict(lambda: (NEG_INF, NEG_INF))
        blank_score = float(frame[blank])

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _logaddexp(p_b, p_nb)
            new_p_b = p_total + blank_score
            old_b, old_nb = new_beams[prefix]
            new_beams[prefix] = (_logaddexp(old_b, new_p_b), old_nb)

            last_tok = prefix[-1] if prefix else -1
            for v_idx in top_indices:
                v = int(v_idx)
                if v == blank:
                    continue
                tok_score = float(frame[v])
                new_prefix = prefix + (v,)

                if v == last_tok:
                    new_p_nb_ext = p_b + tok_score
                    old_b2, old_nb2 = new_beams[prefix]
                    new_beams[prefix] = (old_b2, _logaddexp(old_nb2, p_nb + tok_score))
                else:
                    new_p_nb_ext = p_total + tok_score

                old_b3, old_nb3 = new_beams[new_prefix]
                new_beams[new_prefix] = (old_b3, _logaddexp(old_nb3, new_p_nb_ext))

        scored = [
            (pref, pb, pnb, _logaddexp(pb, pnb))
            for pref, (pb, pnb) in new_beams.items()
        ]
        scored.sort(key=lambda x: x[3], reverse=True)
        beams = {pref: (pb, pnb) for pref, pb, pnb, _ in scored[:beam_width]}

    best_prefix = max(beams, key=lambda p: _logaddexp(beams[p][0], beams[p][1]))
    return list(best_prefix)


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of val samples")
    parser.add_argument("--beam", type=int, default=25, help="Beam width")
    parser.add_argument("--topk", type=int, default=15, help="Top-K per frame")
    args = parser.parse_args()

    # Load model
    model_dir = ROOT / "submission" / "src" / "model"
    tok_dir = ROOT / "submission" / "src" / "tokenizer"
    print(f"Loading model from {model_dir} ...")

    config = WavLMConfig.from_json_file(str(model_dir / "config.json"))
    config.apply_spec_augment = False
    model = WavLMForCTC(config)
    state = torch.load(model_dir / "best_model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(DEVICE).eval()

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(tok_dir))

    # Load val data
    val_path = ROOT / "data" / "processed" / "sft_val.jsonl"
    with open(val_path) as f:
        items = [json.loads(line) for line in f]
    items = items[: args.n]
    print(f"Evaluating {len(items)} samples | beam={args.beam} top_k={args.topk}")

    # Resolve audio dirs
    audio_dirs = [ROOT / "data" / "raw" / "1_audio", ROOT / "data" / "raw" / "2_audio"]

    greedy_hyps, beam_hyps, refs = [], [], []
    t_greedy, t_beam = 0.0, 0.0

    with torch.inference_mode(), torch.amp.autocast(device_type=DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16):
        for i, item in enumerate(items):
            uid = item["utterance_id"]
            ref_text = item["phonetic_text"]

            # Encode reference
            ref_ids = tokenizer.encode(ref_text)

            # Find audio file
            audio_rel = item.get("audio_path", "")
            wav = None
            if audio_rel:
                for d in audio_dirs:
                    p = d / Path(audio_rel).name
                    if p.exists():
                        wav = load_audio(p)
                        break
            if wav is None:
                # Try by utterance ID
                for d in audio_dirs:
                    for ext in (".flac", ".wav"):
                        p = d / f"{uid}{ext}"
                        if p.exists():
                            wav = load_audio(p)
                            break
                    if wav is not None:
                        break
            if wav is None:
                continue

            # Forward
            inp = wav.unsqueeze(0).to(DEVICE)
            mask = torch.ones(1, wav.shape[0], dtype=torch.long, device=DEVICE)
            logits = model(input_values=inp, attention_mask=mask).logits  # (1, T, V)
            olen = model._get_feat_extract_output_lengths(
                torch.tensor([wav.shape[0]], dtype=torch.long, device=DEVICE)
            ).item()

            # Greedy
            t0 = time.perf_counter()
            g_ids = greedy_decode(logits[0], olen)
            t_greedy += time.perf_counter() - t0

            # Beam
            t0 = time.perf_counter()
            lp = F.log_softmax(logits[0, :olen], dim=-1).float().cpu().numpy()
            b_ids = ctc_beam_search(lp, beam_width=args.beam, top_k=args.topk, blank=BLANK)
            t_beam += time.perf_counter() - t0

            greedy_hyps.append(g_ids)
            beam_hyps.append(b_ids)
            refs.append(ref_ids)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(items)} done ...")

    n = len(refs)
    per_greedy = corpus_per(greedy_hyps, refs)
    per_beam = corpus_per(beam_hyps, refs)

    print(f"\n{'='*50}")
    print(f"Samples evaluated: {n}")
    print(f"Greedy PER:  {per_greedy:.4f}  ({t_greedy*1000:.1f}ms total, {t_greedy/n*1000:.2f}ms/utt)")
    print(f"Beam PER:    {per_beam:.4f}  ({t_beam*1000:.1f}ms total, {t_beam/n*1000:.2f}ms/utt)")
    print(f"Δ PER:       {per_beam - per_greedy:+.4f}  ({'BETTER' if per_beam < per_greedy else 'WORSE' if per_beam > per_greedy else 'SAME'})")
    print(f"Relative:    {(per_beam - per_greedy) / per_greedy * 100:+.2f}%")
    print(f"{'='*50}")

    # Show a few examples
    print("\nSample comparisons (first 5):")
    for i in range(min(5, n)):
        ref_str = tokenizer.decode(refs[i])
        g_str = tokenizer.decode(greedy_hyps[i])
        b_str = tokenizer.decode(beam_hyps[i])
        match = "✓" if b_str == g_str else "≠"
        print(f"  REF:    {ref_str}")
        print(f"  GREEDY: {g_str}")
        print(f"  BEAM:   {b_str}  {match}")
        print()


if __name__ == "__main__":
    main()
