#!/usr/bin/env python3
"""NST teacher inference — greedy + beam decode on orphan audio.

Loads the teacher model (best SFT checkpoint), runs batched GPU inference on
all orphan audio files, decodes with both greedy and beam search, and writes
raw pseudo-labels with confidence signals including beam-greedy CER.

Beam search uses the C-native CTC decoder from submission/src/beam_decode.so
(O(1) prefix extension, ~1-2ms per utterance at beam_width=50).

Usage:
    python src/nst/teacher_infer.py                 # uses config.yaml defaults
    python src/nst/teacher_infer.py --device cpu     # force CPU (slow)

Output:
    data/nst/pseudo_labels_raw.jsonl  — one row per orphan, fields:
        utterance_id, greedy_pred, greedy_score, norm_score,
        n_greedy_chars, beam_pred, n_beam_chars, beam_greedy_cer,
        audio_duration_sec, age_bucket, child_id, dataset
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import shutil
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import WavLMConfig, WavLMForCTC, Wav2Vec2CTCTokenizer

# ── Project imports (add src/ to path) ───────────────────────────
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_config
from utils import loads, dumps_line, nearest_rank_pctl

log = logging.getLogger(__name__)

# Silence noisy PyTorch warnings during inference
warnings.filterwarnings("ignore", message=".*mismatched key_padding_mask.*")
warnings.filterwarnings("ignore", message=".*pow_by_natural.*")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# ── Constants ────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
MIN_SAMPLES = SAMPLE_RATE // 10
BLANK       = 0


def _char_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings — O(min(m,n)) space."""
    la, lb = len(a), len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    if lb == 0:
        return la
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


def _resolve_checkpoint(nst_cfg: dict, root: Path) -> Path:
    """Resolve teacher checkpoint path.

    - ``"auto"`` → ``data/models/hf_sft_checkpoints/best/``  (SFT output)
    - Explicit path → relative to project root
    """
    ckpt = nst_cfg["teacher_checkpoint"]
    if ckpt == "auto":
        best = root / "data" / "models" / "hf_sft_checkpoints" / "best"
        if best.is_dir() and (best / "config.json").exists():
            return best
        raise FileNotFoundError(
            f"NST teacher_checkpoint='auto' but no best checkpoint found at "
            f"{best}.  Run --sft first."
        )
    path = root / ckpt
    if not path.is_dir():
        raise FileNotFoundError(f"teacher_checkpoint not found: {path}")
    return path


# ════════════════════════════════════════════════════════════════════
#  Model + tokenizer loading
# ════════════════════════════════════════════════════════════════════
def _load_model(checkpoint_dir: Path, device: str):
    """Load WavLM-CTC from a HF Trainer checkpoint directory."""
    config = WavLMConfig.from_json_file(str(checkpoint_dir / "config.json"))
    config.apply_spec_augment = False
    model = WavLMForCTC(config)

    # HF Trainer saves as model.safetensors (preferred) or pytorch_model.bin
    st_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if st_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(st_path), device=device)
    elif bin_path.exists():
        state = torch.load(bin_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(f"No model weights in {checkpoint_dir}")

    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device).eval()
    return model


def _load_tokenizer(tokenizer_dir: Path):
    return Wav2Vec2CTCTokenizer.from_pretrained(str(tokenizer_dir))


# ════════════════════════════════════════════════════════════════════
#  Audio loading (same as submission — resample + RMS norm)
# ════════════════════════════════════════════════════════════════════
def _load_audio(path: str) -> torch.Tensor | None:
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


def _load_item(args):
    uid, path = args
    wav = _load_audio(path)
    return (uid, wav)


def _bucket_pad_len(n: int, step: int = SAMPLE_RATE) -> int:
    return ((n + step - 1) // step) * step


# ════════════════════════════════════════════════════════════════════
#  Greedy decode + confidence scoring (fully GPU-vectorized)
# ════════════════════════════════════════════════════════════════════
def _greedy_decode_and_score(
    logits: torch.Tensor,
    output_lengths: torch.Tensor,
    log_probs: torch.Tensor | None = None,
) -> tuple[list[list[int]], list[float], torch.Tensor]:
    """Batch greedy CTC decode + confidence score — all on GPU.

    Returns:
        token_ids_list: list of token-id lists per utterance
        scores_list:    list of sum-log-prob scores per utterance
        log_probs:      [B, T, V] log-softmax (reused by beam decode)
    """
    # Greedy token IDs
    ids = logits.argmax(dim=-1)                        # [B, T]
    frame_idx = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
    valid = frame_idx < output_lengths.unsqueeze(1)    # [B, T]
    ids = ids.masked_fill(~valid, 0)

    # CTC collapse mask: non-blank, non-UNK, non-repeat, within valid length
    shifted = F.pad(ids[:, :-1], (1, 0), value=-1)
    kept = (ids != shifted) & (ids != BLANK) & (ids != 1) & valid  # [B, T]

    # Log-probs at greedy positions (on GPU)
    if log_probs is None:
        log_probs = F.log_softmax(logits.float(), dim=-1)  # [B, T, V]
    greedy_lps = log_probs.gather(2, ids.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # Sum log-probs only at kept (collapsed CTC output) positions
    masked_lps = greedy_lps * kept.float()             # [B, T]
    scores = masked_lps.sum(dim=-1)                    # [B]

    # Move to CPU once for extraction
    ids_cpu = ids.cpu()
    kept_cpu = kept.cpu()
    scores_cpu = scores.cpu().tolist()

    token_ids_list = [ids_cpu[b][kept_cpu[b]].tolist() for b in range(ids.size(0))]
    return token_ids_list, scores_cpu, log_probs


# ════════════════════════════════════════════════════════════════════
#  Parallel beam decoder — per-thread .so copies for true parallelism
# ════════════════════════════════════════════════════════════════════
class _ParallelBeamDecoder:
    """Thread-safe parallel CTC beam decoder.

    Each worker thread loads its own copy of beam_decode.so, giving it
    separate static buffers.  ctypes calls release the GIL → true parallelism.
    """

    def __init__(self, so_path: str, n_workers: int):
        self._so_path = so_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._idx = 0
        self._so_copies: list[str] = []
        self._pool = ThreadPoolExecutor(max_workers=n_workers)
        log.info("[NST] parallel beam decoder: %d workers", n_workers)

    def _get_lib(self):
        if hasattr(self._local, 'lib'):
            return self._local.lib, self._local.out_buf, self._local.out_len
        with self._lock:
            idx = self._idx
            self._idx += 1
        tmp = f'/tmp/_beam_mt_{idx}.so'
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
        out_buf = (ctypes.c_int * 512)()
        out_len = ctypes.c_int(0)
        self._local.lib = lib
        self._local.out_buf = out_buf
        self._local.out_len = out_len
        return lib, out_buf, out_len

    def _decode_one(self, args):
        lp, beam_width, blank, length_alpha = args
        lib, out_buf, out_len = self._get_lib()
        T, V = lp.shape
        lp_c = np.ascontiguousarray(lp, dtype=np.float32)
        out_len.value = 0
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


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="NST teacher inference on orphans")
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu)")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    t_start = time.time()
    root = Path(__file__).resolve().parents[2]

    # ── Load config ──────────────────────────────────────────────
    cfg = load_config(root / "src" / "config" / "config.yaml")
    nst = cfg["nst"]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if nst.get("amp_dtype", "float16") == "float16" else torch.bfloat16

    batch_size   = nst["batch_size"]
    blank_bias   = nst["blank_bias"]
    temperature  = nst["temperature"]
    min_dur      = nst["min_duration"]
    max_dur      = nst["max_duration"]
    beam_width   = nst["beam_width"]
    length_alpha = nst["length_alpha"]

    checkpoint_dir = _resolve_checkpoint(nst, root)
    tokenizer_dir  = root / nst["tokenizer_dir"]
    orphans_path   = root / nst["orphans_jsonl"]
    output_path    = root / nst["output_raw"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[NST] device=%s  batch=%d  beam_width=%d  blank_bias=%.2f  temp=%.2f",
              device, batch_size, beam_width, blank_bias, temperature)

    # ── Load model + tokenizer ───────────────────────────────────
    log.info("[NST] loading model from %s …", checkpoint_dir.name)
    model = _load_model(checkpoint_dir, device)
    raw_model = model  # need _get_feat_extract_output_lengths
    tokenizer = _load_tokenizer(tokenizer_dir)

    try:
        import logging as _logging
        _logging.getLogger("torch.utils._sympy.interp").setLevel(_logging.ERROR)
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        log.info("[NST] torch.compile OK")
    except Exception:
        log.info("[NST] torch.compile skipped")

    n_params = sum(p.numel() for p in raw_model.parameters())
    log.info("[NST] model: %s params", f"{n_params:,}")

    # ── Load C beam search decoder ───────────────────────────────
    _beam_src = str(root / "submission" / "src")
    sys.path.insert(0, _beam_src)
    try:
        import decode_beam as _db
        _c_beam = _db.init_c_backend()
        if _c_beam:
            log.info("[NST] C beam decoder loaded OK (beam_width=%d)", beam_width)
        else:
            log.warning("[NST] C beam decoder unavailable — using Python fallback")
    finally:
        if _beam_src in sys.path:
            sys.path.remove(_beam_src)

    # ── Set up parallel beam decoder ─────────────────────────────
    parallel_beam = None
    beam_so = Path(_beam_src) / "beam_decode.so"
    n_beam_workers = nst.get("beam_workers", 8)
    if _c_beam and beam_so.exists() and n_beam_workers > 1:
        parallel_beam = _ParallelBeamDecoder(str(beam_so), n_beam_workers)

    # ── Load orphans ─────────────────────────────────────────────
    log.info("[NST] loading orphans from %s …", orphans_path.name)
    orphans = []
    with open(orphans_path, encoding="utf-8") as f:
        for line in f:
            row = loads(line)
            dur = row.get("audio_duration_sec", 0)
            if min_dur <= dur <= max_dur:
                orphans.append(row)
    log.info("[NST] orphans loaded: %s  (duration filtered to %s–%ss)",
             f"{len(orphans):,}", min_dur, max_dur)

    # ── Resolve audio paths + sort by duration (no loading yet) ─
    audio_dirs = cfg["paths"]["audio_dirs"]
    items = []  # (uid, abs_path, metadata_row)
    for row in orphans:
        ds = row.get("dataset", 2)
        fname = Path(row["audio_path"]).name
        abs_path = str(Path(audio_dirs[ds]) / fname)
        items.append((row["utterance_id"], abs_path, row))

    # Sort longest-first by metadata duration (minimises padding waste)
    items.sort(key=lambda x: x[2].get("audio_duration_sec", 0), reverse=True)
    log.info("[NST] %s items sorted by duration (no audio loaded yet)", f"{len(items):,}")

    num_workers = nst.get("num_workers", 8)

    # ── Warmup (single pass — torch.compile dynamic=True handles varied shapes) ──
    if device == "cuda":
        log.info("[NST] warming up …")
        with torch.inference_mode(), torch.amp.autocast(device_type=device, dtype=amp_dtype):
            L = _bucket_pad_len(SAMPLE_RATE * 10)
            d = torch.zeros(batch_size, L, device=device)
            m = torch.ones(batch_size, L, dtype=torch.long, device=device)
            model(input_values=d, attention_mask=m).logits
            del d, m
        torch.cuda.synchronize()
        log.info("[NST] warmup done")

    # ════════════════════════════════════════════════════════════════
    #  Inference: load audio on-the-fly per batch (threaded prefetch)
    # ════════════════════════════════════════════════════════════════
    n = len(items)
    log.info("[NST] running inference on %s utterances …", f"{n:,}")
    results: list[dict] = []
    n_fail = 0
    t_infer = time.time()

    # Build batches (indices into items)
    batches = [items[i : i + batch_size] for i in range(0, n, batch_size)]

    io_pool = ThreadPoolExecutor(max_workers=num_workers)

    def _load_batch(batch):
        """Load audio for a batch using shared thread pool."""
        load_args = [(uid, path) for uid, path, _meta in batch]
        loaded = dict(io_pool.map(_load_item, load_args))
        uids, wavs, metas = [], [], []
        for uid, _path, meta in batch:
            wav = loaded[uid]
            if wav is not None:
                uids.append(uid)
                wavs.append(wav)
                metas.append(meta)
        return uids, wavs, metas

    # Prefetch 2 batches ahead so I/O never stalls the GPU
    prefetch_pool = ThreadPoolExecutor(max_workers=2)
    futures = {}
    for i in range(min(2, len(batches))):
        futures[i] = prefetch_pool.submit(_load_batch, batches[i])

    with torch.inference_mode(), \
         torch.amp.autocast(device_type=device, dtype=amp_dtype), \
         tqdm(total=n, desc="[NST] inference", unit="utt",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:

        for b_idx in range(len(batches)):
            # Get current batch (already prefetched)
            batch_uids, batch_wavs, batch_metas = futures.pop(b_idx).result()

            # Submit next prefetch while GPU runs
            nxt = b_idx + 2
            if nxt < len(batches):
                futures[nxt] = prefetch_pool.submit(_load_batch, batches[nxt])

            n_fail += len(batches[b_idx]) - len(batch_uids)
            if not batch_wavs:
                pbar.update(len(batches[b_idx]))
                continue

            lengths = [w.shape[0] for w in batch_wavs]
            max_len = _bucket_pad_len(max(lengths))

            # Pad + mask
            padded = torch.zeros(len(batch_wavs), max_len, dtype=torch.float32)
            attn_mask = torch.zeros(len(batch_wavs), max_len, dtype=torch.long)
            for i, w in enumerate(batch_wavs):
                padded[i, :lengths[i]] = w
                attn_mask[i, :lengths[i]] = 1
            padded = padded.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            # Forward
            logits = model(input_values=padded, attention_mask=attn_mask).logits
            input_lens = torch.tensor(lengths, dtype=torch.long, device=device)
            output_lens = raw_model._get_feat_extract_output_lengths(input_lens)

            # Apply blank bias + temperature (same as submission)
            if blank_bias != 0.0:
                logits[..., BLANK] -= blank_bias
            if temperature != 1.0:
                logits = logits / temperature

            # ── Greedy decode + scoring (GPU-vectorized) ─────────
            log_probs = F.log_softmax(logits.float(), dim=-1)
            greedy_ids_list, g_scores, _ = _greedy_decode_and_score(
                logits, output_lens, log_probs
            )

            # ── Beam decode (parallel C backend) ────────────────
            lp_np = log_probs.cpu().numpy()
            olens = output_lens.cpu().tolist()

            lp_slices = [lp_np[i, :olens[i], :].copy()
                         for i in range(len(batch_uids))]
            if parallel_beam is not None:
                beam_ids_list = parallel_beam.decode_batch(
                    lp_slices, beam_width, BLANK, length_alpha)
            else:
                beam_ids_list = [
                    _db.decode_one(lp, beam_width=beam_width,
                                   blank=BLANK, length_alpha=length_alpha)
                    if _c_beam else
                    _db._beam_search_py(lp, beam_width=beam_width,
                                        blank=BLANK, length_alpha=length_alpha)
                    for lp in lp_slices
                ]

            # ── Per-utterance results ────────────────────────────
            for i, uid in enumerate(batch_uids):
                greedy_text = tokenizer.decode(greedy_ids_list[i])

                g_score = g_scores[i]
                n_chars = max(len(greedy_text.replace(" ", "")), 1)
                n_score = g_score / n_chars

                beam_text = tokenizer.decode(beam_ids_list[i])

                # Beam-greedy CER (character-level agreement)
                g_chars = greedy_text.replace(" ", "")
                b_chars = beam_text.replace(" ", "")
                denom = max(len(g_chars), len(b_chars), 1)
                bg_cer = _char_edit_distance(g_chars, b_chars) / denom

                meta = batch_metas[i]
                results.append({
                    "utterance_id":      uid,
                    "greedy_pred":       greedy_text,
                    "greedy_score":      round(g_score, 4),
                    "norm_score":        round(n_score, 4),
                    "n_greedy_chars":    len(greedy_text.replace(" ", "")),
                    "beam_pred":         beam_text,
                    "n_beam_chars":      len(b_chars),
                    "beam_greedy_cer":   round(bg_cer, 4),
                    "audio_duration_sec": meta.get("audio_duration_sec", 0),
                    "age_bucket":        meta.get("age_bucket", "unknown"),
                    "child_id":          meta.get("child_id", ""),
                    "audio_path":        meta.get("audio_path", ""),
                    "dataset":           meta.get("dataset", 2),
                })

            del padded, attn_mask, logits, log_probs, lp_np
            pbar.update(len(batches[b_idx]))

    prefetch_pool.shutdown(wait=False)
    io_pool.shutdown(wait=False)
    if parallel_beam is not None:
        parallel_beam.shutdown()

    # ── Free GPU memory ──────────────────────────────────────────
    del model, raw_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════
    #  Write raw pseudo-labels
    # ════════════════════════════════════════════════════════════════
    log.info("[NST] writing %s rows → %s", f"{len(results):,}", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(dumps_line(row))

    # ── Summary stats ────────────────────────────────────────────
    n_scores_s = sorted((r["norm_score"] for r in results), reverse=True)
    n_res = len(results)

    bg_cers = sorted((r["beam_greedy_cer"] for r in results))

    log.info("--- NST TEACHER INFERENCE SUMMARY ---")
    log.info("[NST] utterances:        %10s", f"{n_res:,}")
    log.info("[NST] audio load fails:  %10d", n_fail)
    log.info("[NST] norm_score (per-char log-prob):")
    log.info("[NST]   p50=%.4f  p10=%.4f (worst 10%%)",
             nearest_rank_pctl(n_scores_s, 50, presorted=True),
             nearest_rank_pctl(n_scores_s, 90, presorted=True))
    log.info("[NST] beam_greedy_cer:")
    log.info("[NST]   p50=%.4f  p90=%.4f  p99=%.4f  exact_match=%.1f%%",
             nearest_rank_pctl(bg_cers, 50, presorted=True),
             nearest_rank_pctl(bg_cers, 90, presorted=True),
             nearest_rank_pctl(bg_cers, 99, presorted=True),
             sum(1 for c in bg_cers if c == 0.0) / max(n_res, 1) * 100)

    elapsed = time.time() - t_start
    log.info("[NST] total time: %.0fs", elapsed)
    log.info("[NST] output: %s", output_path)
    log.info("--- NST TEACHER INFERENCE END ---")


if __name__ == "__main__":
    main()
