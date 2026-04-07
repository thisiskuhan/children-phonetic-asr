#!/usr/bin/env python3
"""Sweep LM_WEIGHT for beam search on the validation set.

Steps:
  1. GPU forward pass on val set → cache log-probs to disk (one-time)
  2. For each LM_WEIGHT in [0.0, 0.05, 0.10, ..., 1.0]:
     - Beam-decode cached log-probs with parallel C decoder
     - Compute corpus PER against val references
  3. Save results to JSON

Usage (on RunPod after Run 16 finishes):
    cd /workspace/309
    source venv/bin/activate
    python scripts/sweep_lm_weight.py

Outputs:
    data/reports/lm_weight_sweep.json
    data/reports/lm_weight_sweep.txt
"""

import ctypes
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from transformers import WavLMForCTC, WavLMConfig, Wav2Vec2CTCTokenizer

# ── Project root ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config

# ── Constants ───────────────────────────────────────────────────
SAMPLE_RATE = 16_000
MIN_SAMPLES = SAMPLE_RATE // 10
BUCKET_STEP = SAMPLE_RATE
BLANK = 0
UNK = 1
BATCH_SIZE = 64
BEAM_WIDTH = 50
LENGTH_ALPHA = 0.0
BEAM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16

# LM weights to sweep — fine grid around likely optimum
LM_WEIGHTS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
              0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]


def ts():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ── Audio loading ───────────────────────────────────────────────
def _load_audio(path: str):
    try:
        data, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim > 1:
            wav = wav.mean(dim=-1)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        std = wav.std()
        if std > 1e-8:
            wav = (wav - wav.mean()) / std
        if wav.shape[0] < MIN_SAMPLES:
            return None
        return wav
    except Exception:
        return None


def bucket_pad_len(n: int) -> int:
    return ((n + BUCKET_STEP - 1) // BUCKET_STEP) * BUCKET_STEP


# ── Edit distance ──────────────────────────────────────────────
def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def corpus_per(hyps, refs):
    total_edits = 0
    total_ref = 0
    for h, r in zip(hyps, refs):
        total_edits += edit_distance(h, r)
        total_ref += len(r)
    return total_edits / total_ref if total_ref > 0 else 0.0


# ── Parallel beam decoder (same as submission) ─────────────────
class ParallelBeamDecoder:
    def __init__(self, so_path: str, n_workers: int,
                 lm_table=None, lm_weight=0.0, unk_id=1):
        self._so_path = so_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._idx = 0
        self._so_copies = []
        self._pool = ThreadPoolExecutor(max_workers=n_workers)
        self._lm_table = lm_table
        self._lm_weight = lm_weight
        self._unk_id = unk_id
        self._use_lm = lm_table is not None and lm_weight > 0.0

    def set_lm_weight(self, w):
        self._lm_weight = w
        self._use_lm = self._lm_table is not None and w > 0.0

    def _get_lib(self):
        if hasattr(self._local, 'lib'):
            return self._local.lib, self._local.has_lm, \
                   self._local.out_buf, self._local.out_len
        with self._lock:
            idx = self._idx
            self._idx += 1
        tmp = f'/tmp/_beam_sweep_{idx}.so'
        shutil.copy2(self._so_path, tmp)
        self._so_copies.append(tmp)
        lib = ctypes.CDLL(tmp)
        lib.ctc_beam_search.restype = ctypes.c_int
        lib.ctc_beam_search.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        has_lm = False
        try:
            lib.ctc_beam_search_lm.restype = ctypes.c_int
            lib.ctc_beam_search_lm.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_double, ctypes.c_double,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_double, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ]
            has_lm = True
        except AttributeError:
            pass
        out_buf = (ctypes.c_int * 512)()
        out_len = ctypes.c_int(0)
        self._local.lib = lib
        self._local.has_lm = has_lm
        self._local.out_buf = out_buf
        self._local.out_len = out_len
        return lib, has_lm, out_buf, out_len

    def _decode_one(self, args):
        lp, beam_width, blank, length_alpha = args
        lib, has_lm, out_buf, out_len = self._get_lib()
        T, V = lp.shape
        lp_c = np.ascontiguousarray(lp, dtype=np.float32)
        out_len.value = 0

        if self._use_lm and has_lm:
            lm = np.ascontiguousarray(self._lm_table, dtype=np.float32)
            lib.ctc_beam_search_lm(
                lp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                T, V, beam_width, blank,
                length_alpha, 1e9,
                lm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self._lm_weight, self._unk_id,
                out_buf, ctypes.byref(out_len),
            )
        else:
            lib.ctc_beam_search(
                lp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                T, V, beam_width, blank,
                length_alpha, 1e9,
                out_buf, ctypes.byref(out_len),
            )
        return list(out_buf[:out_len.value])

    def decode_batch(self, lp_slices, beam_width, blank, length_alpha):
        args = [(lp, beam_width, blank, length_alpha) for lp in lp_slices]
        futures = [self._pool.submit(self._decode_one, a) for a in args]
        return [f.result() for f in futures]

    def shutdown(self):
        self._pool.shutdown(wait=False)
        for p in self._so_copies:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def main():
    t_start = time.time()

    # ── Load config ──────────────────────────────────────────────
    cfg = load_config(ROOT / "src" / "config" / "config.yaml")
    audio_dirs = cfg["paths"]["audio_dirs"]

    # ── Paths ────────────────────────────────────────────────────
    val_manifest = Path(cfg["paths"]["processed"]) / "sft_val.jsonl"
    tokenizer_dir = Path(cfg["paths"]["tokenizer"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Find best checkpoint
    ckpt_dir = Path(cfg["paths"]["models"]) / "hf_sft_checkpoints" / "best"
    if not ckpt_dir.exists():
        # Try current training checkpoints
        ckpt_base = Path(cfg["paths"]["models"]) / "hf_sft_checkpoints"
        candidates = sorted(ckpt_base.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[1]))
        if candidates:
            ckpt_dir = candidates[-1]
            log(f"No 'best' dir found, using latest: {ckpt_dir.name}")
        else:
            log("FATAL: No checkpoint found")
            sys.exit(1)

    log(f"Checkpoint: {ckpt_dir}")
    log(f"Val manifest: {val_manifest}")

    # ── Load model + tokenizer ───────────────────────────────────
    log("Loading model...")
    config = WavLMConfig.from_json_file(str(ckpt_dir / "config.json"))
    config.apply_spec_augment = False
    model = WavLMForCTC(config)

    st_path = ckpt_dir / "model.safetensors"
    bin_path = ckpt_dir / "pytorch_model.bin"
    if st_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(st_path), device=DEVICE)
    elif bin_path.exists():
        state = torch.load(bin_path, map_location=DEVICE, weights_only=True)
    else:
        log(f"FATAL: No weights in {ckpt_dir}")
        sys.exit(1)

    model.load_state_dict(state, strict=True)
    del state
    model = model.to(DEVICE).eval()
    raw_model = model
    log(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(tokenizer_dir))

    # ── Load val data ────────────────────────────────────────────
    log("Loading val manifest (excluding DS3 pseudo-labels)...")
    val_items = []
    ds3_skipped = 0
    with open(val_manifest) as f:
        for line in f:
            row = json.loads(line)
            if row["dataset"] == 3:
                ds3_skipped += 1
                continue
            val_items.append(row)
    log(f"Val samples: {len(val_items)} (skipped {ds3_skipped} DS3 pseudo-labels)")

    # Encode references
    log("Encoding references...")
    ref_ids_all = []
    uids = []
    for row in val_items:
        text = row["phonetic_text"]
        ids = tokenizer(text).input_ids
        ref_ids_all.append(ids)
        uids.append(row["utterance_id"])

    # Load audio
    log("Loading audio...")
    t_load = time.time()
    wavs = {}
    failed = set()
    for row in val_items:
        uid = row["utterance_id"]
        ds = row["dataset"]
        fname = Path(row["audio_path"]).name
        abs_path = str(Path(audio_dirs[ds]) / fname)
        wav = _load_audio(abs_path)
        if wav is None:
            failed.add(uid)
        else:
            wavs[uid] = wav
    log(f"Audio loaded: {len(wavs)} ok, {len(failed)} fail ({time.time()-t_load:.1f}s)")

    # Sort by length (longest first, matches training eval order)
    valid_indices = [i for i, uid in enumerate(uids) if uid not in failed]
    valid_indices.sort(key=lambda i: wavs[uids[i]].shape[0], reverse=True)

    # ── GPU forward pass → cache log-probs ───────────────────────
    log(f"Running GPU forward pass on {len(valid_indices)} val samples...")
    cache_path = reports_dir / "lm_sweep_logprobs.npz"

    if cache_path.exists():
        log(f"Found cached log-probs at {cache_path}, loading...")
        cached = np.load(str(cache_path), allow_pickle=True)
        all_lp = list(cached["log_probs"])
        all_uids = list(cached["uids"])
        all_ref_ids = [list(r) for r in cached["ref_ids"]]
        log(f"Loaded {len(all_lp)} cached log-probs")
    else:
        all_lp = []
        all_uids = []
        all_ref_ids = []

        # Warmup
        if DEVICE == "cuda":
            with torch.inference_mode(), torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE):
                d = torch.zeros(1, SAMPLE_RATE, device=DEVICE)
                m = torch.ones(1, SAMPLE_RATE, dtype=torch.long, device=DEVICE)
                model(input_values=d, attention_mask=m).logits
                del d, m
            torch.cuda.synchronize()

        t_fwd = time.time()
        with torch.inference_mode(), \
             torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE):
            for b_start in range(0, len(valid_indices), BATCH_SIZE):
                batch_idx = valid_indices[b_start:b_start + BATCH_SIZE]
                batch_uids = [uids[i] for i in batch_idx]
                batch_wavs = [wavs[uids[i]] for i in batch_idx]
                batch_refs = [ref_ids_all[i] for i in batch_idx]
                lengths = [w.shape[0] for w in batch_wavs]
                max_len = bucket_pad_len(max(lengths))

                padded = torch.zeros(len(batch_wavs), max_len, dtype=torch.float32)
                attn_mask = torch.zeros(len(batch_wavs), max_len, dtype=torch.long)
                for i, w in enumerate(batch_wavs):
                    padded[i, :lengths[i]] = w
                    attn_mask[i, :lengths[i]] = 1
                padded = padded.to(DEVICE, non_blocking=True)
                attn_mask = attn_mask.to(DEVICE, non_blocking=True)

                logits = model(input_values=padded, attention_mask=attn_mask).logits
                input_lens = torch.tensor(lengths, dtype=torch.long, device=DEVICE)
                output_lens = raw_model._get_feat_extract_output_lengths(input_lens)

                log_probs = F.log_softmax(logits, dim=-1).float()
                log_probs[:, :, UNK] = -1e9
                log_probs = log_probs.cpu().numpy()
                olens = output_lens.cpu().tolist()

                for b in range(logits.size(0)):
                    all_lp.append(log_probs[b, :olens[b], :])
                    all_uids.append(batch_uids[b])
                    all_ref_ids.append(batch_refs[b])

                del padded, attn_mask, logits

                done = min(b_start + BATCH_SIZE, len(valid_indices))
                if b_start == 0 or done % 2000 < BATCH_SIZE:
                    log(f"  Forward: {done}/{len(valid_indices)}")

        fwd_time = time.time() - t_fwd
        log(f"Forward pass done: {len(all_lp)} samples in {fwd_time:.1f}s "
            f"({len(all_lp)/fwd_time:.0f} utt/s)")

        # Save cache
        log(f"Saving log-probs cache to {cache_path}...")
        np.savez(str(cache_path),
                 log_probs=np.array(all_lp, dtype=object),
                 uids=np.array(all_uids),
                 ref_ids=np.array(all_ref_ids, dtype=object))
        cache_mb = cache_path.stat().st_size / 1024 / 1024
        log(f"Cache saved: {cache_mb:.0f} MB")

    # Free GPU memory
    del model, raw_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Load LM table ────────────────────────────────────────────
    lm_path = ROOT / "submission" / "src" / "lm_trigram.bin"
    lm_table = None
    if lm_path.exists():
        raw = np.fromfile(str(lm_path), dtype=np.float32)
        V = round(len(raw) ** (1.0 / 3.0))
        if V * V * V == len(raw):
            lm_table = raw.reshape(V, V, V)
            log(f"LM loaded: V={V}, {lm_path.stat().st_size / 1024:.0f} KB")
        else:
            log(f"WARN: LM file size mismatch, skipping LM")
    else:
        log(f"WARN: LM not found at {lm_path}, will sweep without LM (greedy baseline only)")

    # ── Init parallel beam decoder ───────────────────────────────
    so_path = ROOT / "submission" / "src" / "beam_decode.so"
    if not so_path.exists():
        # Try to compile
        c_path = ROOT / "submission" / "src" / "beam_decode.c"
        if c_path.exists():
            import subprocess
            log("Compiling beam_decode.so...")
            subprocess.check_call(
                ["gcc", "-O3", "-shared", "-fPIC", "-o", str(so_path), str(c_path), "-lm"],
                timeout=30,
            )
            log("Compiled OK")
        else:
            log("FATAL: beam_decode.so and beam_decode.c not found")
            sys.exit(1)

    decoder = ParallelBeamDecoder(
        str(so_path), BEAM_WORKERS,
        lm_table=lm_table, lm_weight=0.0, unk_id=UNK)

    # ── Also compute greedy baseline ─────────────────────────────
    log("Computing greedy baseline...")
    t_greedy = time.time()
    greedy_hyps = []
    for lp in all_lp:
        ids = np.argmax(lp, axis=-1)
        # CTC collapse: remove blanks and consecutive duplicates
        collapsed = []
        prev = -1
        for tok in ids:
            if tok != BLANK and tok != UNK and tok != prev:
                collapsed.append(int(tok))
            prev = tok
        greedy_hyps.append(collapsed)
    greedy_per = corpus_per(greedy_hyps, all_ref_ids)
    log(f"Greedy PER: {greedy_per:.6f} ({time.time()-t_greedy:.1f}s)")

    # ── Sweep LM weights ────────────────────────────────────────
    results = [{"lm_weight": "greedy", "per": greedy_per, "time_sec": 0}]

    for lm_w in LM_WEIGHTS:
        decoder.set_lm_weight(lm_w)
        log(f"Sweeping LM_WEIGHT={lm_w:.2f}...")
        t_sw = time.time()

        hyps_ids = []
        for b_start in range(0, len(all_lp), BATCH_SIZE):
            batch_lp = all_lp[b_start:b_start + BATCH_SIZE]
            ids_list = decoder.decode_batch(batch_lp, BEAM_WIDTH, BLANK, LENGTH_ALPHA)
            hyps_ids.extend(ids_list)

        sweep_per = corpus_per(hyps_ids, all_ref_ids)
        elapsed = time.time() - t_sw
        results.append({
            "lm_weight": lm_w,
            "per": sweep_per,
            "time_sec": round(elapsed, 1),
        })
        log(f"  LM_WEIGHT={lm_w:.2f} → PER={sweep_per:.6f} ({elapsed:.1f}s)")

    decoder.shutdown()

    # ── Find best ────────────────────────────────────────────────
    beam_results = [r for r in results if r["lm_weight"] != "greedy"]
    best = min(beam_results, key=lambda r: r["per"])
    greedy_r = results[0]

    # ── Save results ─────────────────────────────────────────────
    output = {
        "sweep_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(ckpt_dir),
        "val_samples": len(all_lp),
        "beam_width": BEAM_WIDTH,
        "length_alpha": LENGTH_ALPHA,
        "beam_workers": BEAM_WORKERS,
        "lm_path": str(lm_path),
        "greedy_per": greedy_r["per"],
        "best_lm_weight": best["lm_weight"],
        "best_beam_per": best["per"],
        "improvement_over_greedy": greedy_r["per"] - best["per"],
        "total_time_sec": round(time.time() - t_start, 1),
        "results": results,
    }

    json_path = reports_dir / "lm_weight_sweep.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results saved to {json_path}")

    # Pretty text report
    txt_path = reports_dir / "lm_weight_sweep.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  LM WEIGHT SWEEP RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {ckpt_dir}\n")
        f.write(f"Val samples: {len(all_lp)}\n")
        f.write(f"Beam width: {BEAM_WIDTH}\n")
        f.write(f"Length alpha: {LENGTH_ALPHA}\n\n")
        f.write(f"{'LM Weight':>12} | {'PER':>10} | {'Time':>8} | {'vs Greedy':>10}\n")
        f.write("-" * 50 + "\n")
        for r in results:
            w = r["lm_weight"]
            w_str = f"{w}" if isinstance(w, str) else f"{w:.2f}"
            diff = greedy_r["per"] - r["per"]
            diff_str = f"{diff:+.6f}" if w != "greedy" else "baseline"
            t_str = f"{r['time_sec']:.1f}s" if r["time_sec"] > 0 else "-"
            marker = " ★" if r is best else ""
            f.write(f"{w_str:>12} | {r['per']:.6f}   | {t_str:>8} | {diff_str:>10}{marker}\n")
        f.write("-" * 50 + "\n")
        f.write(f"\n★ BEST: LM_WEIGHT={best['lm_weight']:.2f} → PER={best['per']:.6f}\n")
        f.write(f"  Greedy PER: {greedy_r['per']:.6f}\n")
        f.write(f"  Improvement: {greedy_r['per'] - best['per']:+.6f}\n")
        f.write(f"  Total sweep time: {time.time() - t_start:.0f}s\n")
    log(f"Text report saved to {txt_path}")

    # Print summary
    log("")
    log("=" * 50)
    log("  SWEEP SUMMARY")
    log("=" * 50)
    log(f"  Greedy PER:     {greedy_r['per']:.6f}")
    log(f"  Best beam PER:  {best['per']:.6f} (LM_WEIGHT={best['lm_weight']:.2f})")
    log(f"  Improvement:    {greedy_r['per'] - best['per']:+.6f}")
    log(f"  Total time:     {time.time() - t_start:.0f}s")
    log("=" * 50)


if __name__ == "__main__":
    main()
