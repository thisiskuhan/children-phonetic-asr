"""CTC beam search decoder — pyctcdecode + KenLM word-level 3-gram LM.

Provides:
    init_decoder()                  → initialise from vocab / LM / unigrams
    decode_batch_beam()             → sequential beam search
    decode_batch_beam_parallel()    → pool-based parallel beam search
    is_available()                  → readiness check
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ── Logging ──────────────────────────────────────────────────────
def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _warn(msg):
    print(f"[{time.strftime('%H:%M:%S')}] WARN {msg}",
          file=sys.stderr, flush=True)


# ── Module state ─────────────────────────────────────────────────
_decoder = None
_decoder_no_lm = None
_initialized = False


def init_decoder(
    vocab_path: Optional[Path] = None,
    lm_path: Optional[Path] = None,
    unigrams_path: Optional[Path] = None,
    alpha: float = 0.5,
    beta: float = 1.5,
) -> bool:
    """Initialise pyctcdecode beam search decoder.

    Returns True on success.  Safe to call multiple times.
    """
    global _decoder, _decoder_no_lm, _initialized

    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        _warn("pyctcdecode not installed — beam search unavailable")
        return False

    src_dir = Path(__file__).parent
    if vocab_path is None:
        vocab_path = src_dir / "tokenizer" / "vocab.json"
    if lm_path is None:
        binary = src_dir / "kenlm_word3gram.bin"
        arpa = src_dir / "kenlm_word3gram.arpa"
        lm_path = binary if binary.exists() else arpa
    if unigrams_path is None:
        unigrams_path = src_dir / "unigrams.json"

    # ── Vocabulary → labels ──────────────────────────────────────
    try:
        with open(vocab_path) as f:
            vocab = json.load(f)
    except Exception as e:
        _warn(f"Failed to load vocab: {e}")
        return False

    id2char = {v: k for k, v in vocab.items()}
    V = max(id2char.keys()) + 1

    # idx 0 = [PAD] → '' (CTC blank)
    # idx 1 = [UNK] → '⁂' (placeholder, suppressed via logits)
    # idx 2 = |     → ' ' (word boundary for word-level LM)
    # idx 3… = IPA characters
    labels = []
    for i in range(V):
        ch = id2char.get(i, "")
        if ch == "[PAD]":
            labels.append("")
        elif ch == "[UNK]":
            labels.append("\u2042")
        elif ch == "|":
            labels.append(" ")
        else:
            labels.append(ch)

    # ── Unigrams ─────────────────────────────────────────────────
    unigrams = None
    if unigrams_path.exists():
        try:
            with open(unigrams_path) as f:
                unigrams = json.load(f)
            _log(f"Loaded {len(unigrams)} unigrams")
        except Exception as e:
            _warn(f"Failed to load unigrams: {e}")

    # ── Decoder with LM ─────────────────────────────────────────
    if lm_path.exists():
        try:
            _decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=str(lm_path),
                unigrams=unigrams,
                alpha=alpha,
                beta=beta,
            )
            _log(f"Beam decoder ready: LM={lm_path.name}, "
                 f"α={alpha}, β={beta}")
        except Exception as e:
            _warn(f"LM decoder build failed: {e}")
            _decoder = None
    else:
        _warn(f"LM not found: {lm_path}")

    # ── Decoder without LM (intermediate fallback) ──────────────
    try:
        _decoder_no_lm = build_ctcdecoder(labels=labels)
    except Exception as e:
        _warn(f"No-LM decoder build failed: {e}")

    _initialized = _decoder is not None or _decoder_no_lm is not None
    return _initialized


# ── Decode ───────────────────────────────────────────────────────
def decode_batch_beam(
    log_probs_list: list[np.ndarray],
    beam_width: int = 30,
    beam_prune_logp: float = -10.0,
    token_min_logp: float = -5.0,
    use_lm: bool = True,
) -> list[str]:
    """Decode a batch of (T, V) log-probability arrays sequentially."""
    decoder = _decoder if (use_lm and _decoder) else _decoder_no_lm
    if decoder is None:
        raise RuntimeError("Decoder not initialised — call init_decoder()")
    return [
        decoder.decode(
            lp,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
        )
        for lp in log_probs_list
    ]


def decode_batch_beam_parallel(
    log_probs_list: list[np.ndarray],
    beam_width: int = 30,
    beam_prune_logp: float = -10.0,
    token_min_logp: float = -5.0,
    use_lm: bool = True,
    pool=None,
) -> list[str]:
    """Decode using multiprocessing.Pool via pyctcdecode.decode_batch."""
    decoder = _decoder if (use_lm and _decoder) else _decoder_no_lm
    if decoder is None:
        raise RuntimeError("Decoder not initialised — call init_decoder()")
    try:
        return decoder.decode_batch(
            pool=pool,
            logits_list=log_probs_list,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
        )
    except Exception:
        return decode_batch_beam(
            log_probs_list,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            use_lm=use_lm,
        )


def is_available() -> bool:
    """True if at least one decoder backend is ready."""
    return _initialized and (_decoder is not None or _decoder_no_lm is not None)
