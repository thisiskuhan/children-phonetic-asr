#!/usr/bin/env python3
"""Phonetic track inference — WavLM-CTC with beam search + LM.

Competition : On Top of Pasketti (Children's Speech Recognition)
Track       : Phonetic (IPA-CER)
Model       : WavLM Base+ fine-tuned CTC head, vocab=53
Runtime     : A100 80 GB, 24 vCPU, 220 GB RAM, 2-hour limit, no network

Pipeline:
    1. Threaded audio prefetch (load + resample + silence trim + CMVN on CPU)
    2. Length-sorted batching with adaptive batch size
    3. CTC decode: DECODE_MODE="beam" → pyctcdecode + KenLM 3-gram LM
                   DECODE_MODE="greedy" → argmax, collapse repeats
       (beam always falls back to greedy on failure)
    4. Post-process (strip invalid characters) + write submission.jsonl
"""

import gc
import importlib
import json
import logging
import os
import platform
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMForCTC, Wav2Vec2CTCTokenizer

# Suppress framework noise
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*pow_by_natural.*")
warnings.filterwarnings("ignore", message=".*CUDAGraph supports dynamic shapes.*")
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════
DECODE_MODE   = "beam"       # "beam" = beam search + LM, "greedy" = greedy CTC
BATCH_SIZE    = 128
MAX_SAMPLES   = 128 * 16_000 * 8     # cap total samples per batch (~8s avg)
SAMPLE_RATE   = 16_000
BUCKET_STEP   = SAMPLE_RATE          # 1-s buckets for padding
MIN_SAMPLES   = SAMPLE_RATE // 10    # ignore < 0.1s clips
NUM_WORKERS   = min(20, os.cpu_count() or 8)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE     = torch.float16 if DEVICE == "cuda" else torch.bfloat16
BLANK         = 0
UNK           = 1

# Beam search hyper-parameters (tuned on held-out val set)
BEAM_WIDTH      = 100
BEAM_ALPHA      = 0.575   # LM weight
BEAM_BETA       = 3.0     # word insertion bonus
BEAM_PRUNE_LOGP = -10.0
TOKEN_MIN_LOGP  = -5.0

# Audio pre-processing at inference (matches training pipeline)
SILENCE_DB      = -40.0              # trim leading/trailing silence below this (dB re peak)
MIN_TRIM_SAMPLES = int(0.5 * 16_000) # min samples after trim (0.5s) to avoid empty wavs
ABS_FLOOR       = 1e-4               # absolute silence floor (~-80 dBFS) — zero sub-audible noise

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Valid output characters (IPA vocab + space).  Anything else is stripped.
_VALID = set(
    " bcdefghijklmnoprstuvwxz"
    "\u00e6\u00e7\u00f0\u014b\u0250\u0251\u0254\u0259\u025a\u025b\u025f"
    "\u026a\u026b\u026c\u0279\u027e\u0281\u0283\u028a\u028c\u0292\u0294"
    "\u029d\u02a4\u02a7\u02d0\u03b8\u03c7"
)
_SPACE_RE = re.compile(r"\s+")


# ════════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════════
def _ts():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{_ts()}] {msg}", flush=True)


def warn(msg):
    print(f"[{_ts()}] WARN {msg}", file=sys.stderr, flush=True)


# ════════════════════════════════════════════════════════════════════
#  RUNTIME DIAGNOSTICS (removable — nothing depends on this)
# ════════════════════════════════════════════════════════════════════
def print_runtime_info():
    """Print system specs and available packages.  Safe to delete entirely."""
    cpu_name = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    log(f"Python {sys.version.split()[0]} | {platform.machine()} | "
        f"CPU {os.cpu_count()} | {cpu_name}")
    if DEVICE == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            mem_gib = getattr(props, "total_memory", 0) / 1024**3
            log(f"GPU: {torch.cuda.get_device_name(0)} ({mem_gib:.0f} GiB)")
        except Exception as e:
            log(f"GPU: {torch.cuda.get_device_name(0)} (mem query failed: {e})")
    else:
        log("GPU: none (CPU mode)")

    pkgs = [
        "torch", "torchaudio", "transformers", "numpy", "soundfile",
        "pyctcdecode", "kenlm", "flashlight.lib.text",
        "ctcdecode", "nemo_toolkit", "jiwer", "orjson",
    ]
    available, missing = [], []
    for name in pkgs:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            available.append(f"{name}=={ver}")
        except ImportError:
            missing.append(name)
    log(f"Packages: {', '.join(available)}")
    if missing:
        log(f"Not installed: {', '.join(missing)}")


# ════════════════════════════════════════════════════════════════════
#  MODEL & TOKENIZER
# ════════════════════════════════════════════════════════════════════
def load_model(model_dir: Path):
    model = WavLMForCTC.from_pretrained(str(model_dir), apply_spec_augment=False)
    return model.to(DEVICE).eval()


def load_tokenizer(src_root: Path):
    return Wav2Vec2CTCTokenizer.from_pretrained(str(src_root / "tokenizer"))


# ════════════════════════════════════════════════════════════════════
#  AUDIO LOADING
# ════════════════════════════════════════════════════════════════════
def _load_audio(path):
    data, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # Silence trim — remove leading/trailing silence below threshold.
    # Matches the training pipeline (data_collator.py _trim_silence).
    n = wav.size(0)
    if n >= MIN_TRIM_SAMPLES:
        abs_wav = wav.abs()
        peak = abs_wav.max()
        if peak > 1e-10:
            threshold = peak * (10.0 ** (SILENCE_DB / 20.0))
            above = abs_wav > threshold
            nonzero = torch.nonzero(above, as_tuple=False)
            if nonzero.numel() > 0:
                start = nonzero[0].item()
                end = nonzero[-1].item() + 1
                if (end - start) < MIN_TRIM_SAMPLES:
                    mid = (start + end) // 2
                    half = MIN_TRIM_SAMPLES // 2
                    start = max(0, mid - half)
                    end = min(n, start + MIN_TRIM_SAMPLES)
                    start = max(0, end - MIN_TRIM_SAMPLES)
                if start > 0 or end < n:
                    wav = wav[start:end].contiguous()

    # Absolute silence floor — zero sub-audible residual noise
    # Matches training pipeline (data_collator.py _apply_abs_floor)
    mask = wav.abs() < ABS_FLOOR
    if mask.any():
        wav = wav.clone()
        wav[mask] = 0.0

    # Zero-mean unit-variance (CMVN) — matches WavLM pretraining
    std = wav.std()
    if std > 1e-8:
        wav = (wav - wav.mean()) / std
    return wav


def _load_item(args):
    uid, path = args
    try:
        wav = _load_audio(path)
        if wav.shape[0] < MIN_SAMPLES:
            return (uid, None)
        return (uid, wav)
    except Exception:
        return (uid, None)


# ════════════════════════════════════════════════════════════════════
#  BATCHING & FORWARD
# ════════════════════════════════════════════════════════════════════
def _bucket_pad(n: int) -> int:
    return ((n + BUCKET_STEP - 1) // BUCKET_STEP) * BUCKET_STEP


def _make_batches(order, wavs):
    batches, i = [], 0
    while i < len(order):
        max_len = wavs[order[i]].shape[0]
        bs = min(BATCH_SIZE, max(1, MAX_SAMPLES // max_len))
        batches.append(order[i : i + bs])
        i += bs
    return batches


def _forward_batch_single(model, raw_model, batch_wavs):
    """Run batched GPU forward at original speed, return (logits, output_lens)."""
    lengths = [w.shape[0] for w in batch_wavs]
    max_len = _bucket_pad(max(lengths))

    padded = torch.zeros(len(batch_wavs), max_len, dtype=torch.float32)
    attn = torch.zeros(len(batch_wavs), max_len, dtype=torch.long)
    for i, w in enumerate(batch_wavs):
        padded[i, :lengths[i]] = w
        attn[i, :lengths[i]] = 1
    padded = padded.to(DEVICE, non_blocking=True)
    attn = attn.to(DEVICE, non_blocking=True)

    logits = model(input_values=padded, attention_mask=attn).logits
    in_lens = torch.tensor(lengths, dtype=torch.long, device=DEVICE)
    out_lens = raw_model._get_feat_extract_output_lengths(in_lens)

    lp = logits.float()
    lp[:, :, UNK] = -1e9
    return lp, out_lens


def _forward_batch(model, raw_model, batch_wavs):
    """Run batched GPU forward, return (logits, output_lens) on device."""
    return _forward_batch_single(model, raw_model, batch_wavs)



# ════════════════════════════════════════════════════════════════════
#  GREEDY DECODE
# ════════════════════════════════════════════════════════════════════
def _greedy_batch(logits, output_lengths):
    ids = logits.argmax(dim=-1)
    frame_idx = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
    ids = ids.masked_fill(frame_idx >= output_lengths.unsqueeze(1), 0)
    shifted = F.pad(ids[:, :-1], (1, 0), value=-1)
    mask = (ids != shifted) & (ids != BLANK) & (ids != UNK)
    ids_cpu = ids.cpu()
    mask_cpu = mask.cpu()
    return [ids_cpu[b][mask_cpu[b]].tolist() for b in range(ids.size(0))]


# ════════════════════════════════════════════════════════════════════
#  BEAM SEARCH DECODE (pyctcdecode + KenLM)
# ════════════════════════════════════════════════════════════════════
def _run_beam(model, raw_model, tokenizer, wavs, order, failed):
    """Primary decoder: parallel beam search with word-level LM."""
    from decode_beam_lm import init_decoder, decode_batch_beam_parallel

    src_root = Path(__file__).parent.resolve()
    if not init_decoder(
        vocab_path=src_root / "tokenizer" / "vocab.json",
        unigrams_path=src_root / "unigrams.json",
        alpha=BEAM_ALPHA,
        beta=BEAM_BETA,
    ):
        raise RuntimeError("Beam decoder init failed")

    batches = _make_batches(order, wavs)
    log(f"Beam (w={BEAM_WIDTH}, α={BEAM_ALPHA}, β={BEAM_BETA}): "
        f"{len(batches)} batches")

    num_pool = min(os.cpu_count() or 4, 16)
    decode_pool = Pool(num_pool)
    log(f"Beam decode pool: {num_pool} workers")

    predictions: dict[str, str] = {uid: "" for uid in failed}
    greedy_fallback = 0
    t0 = time.time()
    done = 0
    n = len(order) + len(failed)

    try:
        with torch.inference_mode(), \
             torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE):
            for batch_uids in batches:
                batch_wavs = [wavs[u] for u in batch_uids]
                lp, out_lens = _forward_batch(model, raw_model, batch_wavs)
                # Pass raw logits (not log_softmax) — pyctcdecode applies
                # its own _log_softmax internally.  Applying F.log_softmax
                # here would cause double log_softmax, compressing the
                # distribution and degrading beam search quality.
                lp_cpu = lp.cpu().numpy()
                ol_cpu = out_lens.cpu().tolist()
                lp_list = [lp_cpu[i, :ol_cpu[i], :]
                           for i in range(len(batch_uids))]
                try:
                    texts = decode_batch_beam_parallel(
                        lp_list,
                        beam_width=BEAM_WIDTH,
                        beam_prune_logp=BEAM_PRUNE_LOGP,
                        token_min_logp=TOKEN_MIN_LOGP,
                        pool=decode_pool,
                    )
                    for uid, text in zip(batch_uids, texts):
                        predictions[uid] = text
                except Exception as e:
                    warn(f"Beam batch failed ({e}), "
                         f"greedy fallback for {len(batch_uids)} utts")
                    ids_list = _greedy_batch(lp, out_lens)
                    for uid, ids in zip(batch_uids, ids_list):
                        predictions[uid] = tokenizer.decode(
                            ids, group_tokens=False)
                    greedy_fallback += len(batch_uids)
                del lp

                done += len(batch_uids)
                if (done % max(1, n // 10) < BATCH_SIZE
                        or done == len(order)):
                    rate = done / max(time.time() - t0, 1e-6)
                    log(f"BEAM {done}/{n} ({100*done/n:.0f}%) "
                        f"{rate:.0f} utt/s")
    finally:
        decode_pool.close()
        decode_pool.join()

    if greedy_fallback:
        warn(f"Greedy fallback used for {greedy_fallback}/{n} utterances")
    return predictions


# ════════════════════════════════════════════════════════════════════
#  GREEDY DECODE (full pipeline — fallback only)
# ════════════════════════════════════════════════════════════════════
def _run_greedy(model, raw_model, tokenizer, wavs, order, failed):
    """Fallback decoder: pure greedy CTC."""
    batches = _make_batches(order, wavs)
    log(f"Greedy: {len(batches)} batches")

    predictions: dict[str, str] = {uid: "" for uid in failed}
    t0 = time.time()
    done = 0
    n = len(order) + len(failed)

    with torch.inference_mode(), \
         torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE):
        for batch_uids in batches:
            batch_wavs = [wavs[u] for u in batch_uids]
            lp, out_lens = _forward_batch(model, raw_model, batch_wavs)

            ids_list = _greedy_batch(lp, out_lens)
            for uid, ids in zip(batch_uids, ids_list):
                predictions[uid] = tokenizer.decode(ids, group_tokens=False)

            del lp
            done += len(batch_uids)
            if (done % max(1, n // 10) < BATCH_SIZE
                    or done == len(order)):
                rate = done / max(time.time() - t0, 1e-6)
                log(f"GREEDY {done}/{n} ({100*done/n:.0f}%) "
                    f"{rate:.0f} utt/s")

    return predictions


# ════════════════════════════════════════════════════════════════════
#  POST-PROCESSING
# ════════════════════════════════════════════════════════════════════
def _postprocess(predictions: dict[str, str]) -> dict[str, str]:
    """Strip characters not in the IPA vocab and normalise whitespace."""
    cleaned = 0
    for uid in predictions:
        raw = predictions[uid]
        safe = "".join(c for c in raw if c in _VALID)
        safe = _SPACE_RE.sub(" ", safe).strip()
        if safe != raw:
            cleaned += 1
            predictions[uid] = safe
    if cleaned:
        log(f"Post-process: cleaned {cleaned} predictions")
    return predictions


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    src_root = Path(__file__).parent.resolve()
    data_dir = Path("data")
    out_path = Path("submission") / "submission.jsonl"

    # ── Runtime info ─────────────────────────────────────────────
    try:
        print_runtime_info()
    except Exception as e:
        warn(f"Runtime info failed (non-fatal): {e}")

    # ── Validate paths ───────────────────────────────────────────
    required = [
        data_dir / "utterance_metadata.jsonl",
        data_dir / "submission_format.jsonl",
        src_root / "model" / "config.json",
    ]
    for p in required:
        if not p.exists():
            log(f"FATAL: required file missing: {p}")
            sys.exit(1)

    # ── Load model ───────────────────────────────────────────────
    log("Loading model ...")
    model = load_model(src_root / "model")
    raw_model = model
    tokenizer = load_tokenizer(src_root)
    n_params = sum(p.numel() for p in raw_model.parameters())
    log(f"Model: {n_params:,} params on {DEVICE}")

    # ── Warmup (CUDA graphs / JIT) ───────────────────────────────
    if DEVICE == "cuda":
        log("Warming up ...")
        with torch.inference_mode(), \
             torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE):
            d = torch.zeros(1, SAMPLE_RATE, device=DEVICE)
            m = torch.ones(1, SAMPLE_RATE, dtype=torch.long, device=DEVICE)
            model(input_values=d, attention_mask=m).logits
            del d, m
        torch.cuda.synchronize()
        log("Warmup done")

    # ── Load audio (threaded) ────────────────────────────────────
    with open(data_dir / "utterance_metadata.jsonl") as f:
        items = [json.loads(line) for line in f]
    log(f"Utterances: {len(items)}")

    t_load = time.time()
    wavs: dict[str, torch.Tensor] = {}
    failed: set[str] = set()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        tasks = [(it["utterance_id"], data_dir / it["audio_path"])
                 for it in items]
        for uid, wav in pool.map(_load_item, tasks):
            if wav is None:
                failed.add(uid)
            else:
                wavs[uid] = wav
    log(f"Audio: {len(wavs)} loaded, {len(failed)} failed "
        f"({time.time()-t_load:.1f}s)")

    order = sorted(wavs, key=lambda u: wavs[u].shape[0], reverse=True)

    # ── Decode ───────────────────────────────────────────────────
    log(f"Decode mode: {DECODE_MODE}")
    predictions = None

    if DECODE_MODE == "beam":
        try:
            log("Attempting beam search + LM decode ...")
            predictions = _run_beam(
                model, raw_model, tokenizer, wavs, order, failed)
            log("Beam search decode complete")
        except Exception as e:
            warn(f"Beam search failed: {e}")
            predictions = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # Greedy: either chosen mode or fallback from failed beam
    if predictions is None:
        log("Running greedy decode ...")
        try:
            predictions = _run_greedy(
                model, raw_model, tokenizer, wavs, order, failed)
        except Exception as e:
            warn(f"Greedy failed: {e}")
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            # Strategy 3: greedy with reduced batch (OOM recovery)
            global BATCH_SIZE, MAX_SAMPLES
            BATCH_SIZE = 16
            MAX_SAMPLES = 16 * 16_000 * 4
            log(f"Retrying greedy with BATCH_SIZE={BATCH_SIZE}")
            try:
                predictions = _run_greedy(
                    model, raw_model, tokenizer, wavs, order, failed)
            except Exception as e2:
                warn(f"All decode strategies failed: {e2}")
                predictions = {uid: "" for uid in wavs}
                predictions.update({uid: "" for uid in failed})

    # ── Post-process + write ─────────────────────────────────────
    predictions = _postprocess(predictions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "submission_format.jsonl") as fr, \
         open(out_path, "w") as fw:
        for line in fr:
            rec = json.loads(line)
            rec["phonetic_text"] = predictions.get(
                rec["utterance_id"], "")
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start
    log(f"Done. {len(predictions)} predictions, {len(failed)} failed, "
        f"{elapsed:.0f}s total")


if __name__ == "__main__":
    main()
