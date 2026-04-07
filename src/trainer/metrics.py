"""
Training metrics — SFT (CTC decoding, PER).
============================================

Pure functions with no model or dataset dependencies.
Used by ``trainer.HFSFTTrainer`` during training.

Metrics
-------
**SFT**
- **CTC greedy decode** — argmax → collapse repeats → remove blanks
- **PER** (Phoneme Error Rate) — Levenshtein distance / reference length
- **Blank ratio** — fraction of argmax frames emitting the CTC blank
- **Mean argmax run length** — alignment quality proxy (PhD-level CTC monitoring)
- **Per-phoneme recall** — hit / miss per phoneme via Levenshtein alignment
- **Alignment formatting** — REF / HYP display for qualitative log inspection
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CTC greedy decode (for training)
# ---------------------------------------------------------------------------

def ctc_greedy_decode(
    logits: Tensor,
    input_lengths: Tensor | None = None,
    *,
    blank_id: int = 0,
) -> list[list[int]]:
    """Greedy CTC decode: argmax → collapse consecutive repeats → remove blanks.

    Parameters
    ----------
    logits : Tensor
        Shape ``(B, T, V)`` — raw logits from the CTC head.
    input_lengths : Tensor | None
        Shape ``(B,)`` — number of valid *output* frames per sample.
        If provided, only the first ``input_lengths[i]`` frames of each
        sequence are decoded.  Padding frames are ignored.
    blank_id : int
        CTC blank token ID (default 0 = ``[PAD]``).

    Returns
    -------
    list[list[int]]
        Decoded token-ID sequence per batch element.
    """
    import numpy as np

    preds_np = logits.argmax(dim=-1).cpu().numpy()      # (B, T)
    batch_size = preds_np.shape[0]
    lengths_np = (
        input_lengths.cpu().numpy() if input_lengths is not None
        else None
    )
    decoded: list[list[int]] = []

    for i in range(batch_size):
        T = int(lengths_np[i]) if lengths_np is not None else preds_np.shape[1]
        seq = preds_np[i, :T]
        if T == 0:
            decoded.append([])
            continue
        # Vectorised collapse: keep positions where token changes
        change = np.empty(T, dtype=np.bool_)
        change[0] = True
        change[1:] = seq[1:] != seq[:-1]
        collapsed = seq[change]
        decoded.append(collapsed[collapsed != blank_id].tolist())

    return decoded


# ---------------------------------------------------------------------------
# Edit distance — Levenshtein
# ---------------------------------------------------------------------------

def _edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein distance between two integer sequences — O(min(m,n)) space."""
    # Keep the shorter sequence as the "column" to minimise memory.
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


def _align(
    ref: list[int],
    hyp: list[int],
) -> list[tuple[str, int | None, int | None]]:
    """Full Levenshtein alignment with backtrace.

    Parameters
    ----------
    ref : list[int]
        Reference token-ID sequence.
    hyp : list[int]
        Hypothesis token-ID sequence.

    Returns
    -------
    list[tuple[str, int | None, int | None]]
        Sequence of ``(op, ref_token, hyp_token)`` where *op* is one of
        ``"match"``, ``"sub"``, ``"del"``, ``"ins"``.
    """
    m, n = len(ref), len(hyp)

    # Full DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # delete
                    dp[i][j - 1],       # insert
                    dp[i - 1][j - 1],   # substitute
                )

    # Backtrace — prefer match > del > ins > sub for stable alignment
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("match", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
        else:
            # substitution
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1

    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# Phoneme Error Rate (PER)
# ---------------------------------------------------------------------------

def compute_per(hyp: list[int], ref: list[int]) -> float:
    """PER for a single hypothesis / reference pair.

    Parameters
    ----------
    hyp : list[int]
        Decoded hypothesis token-ID sequence.
    ref : list[int]
        Reference token-ID sequence.

    Returns
    -------
    float
        ``edit_distance(hyp, ref) / len(ref)``.  Returns 0.0 for empty *ref*.
    """
    if not ref:
        return 0.0
    return _edit_distance(hyp, ref) / len(ref)


def compute_per_batch(
    hyps: list[list[int]],
    refs: list[list[int]],
) -> float:
    """Corpus-level PER — total edits / total reference tokens.

    Parameters
    ----------
    hyps : list[list[int]]
        Hypothesis sequences.
    refs : list[list[int]]
        Reference sequences (same length as *hyps*).

    Returns
    -------
    float
        Corpus PER.  Returns 0.0 if total reference length is zero.
    """
    assert len(hyps) == len(refs), (
        f"FATAL: hyps/refs length mismatch: {len(hyps)} vs {len(refs)}"
    )
    total_edits = 0
    total_ref_len = 0
    for h, r in zip(hyps, refs):
        total_edits += _edit_distance(h, r)
        total_ref_len += len(r)
    if total_ref_len == 0:
        return 0.0
    return total_edits / total_ref_len


# ---------------------------------------------------------------------------
# Blank ratio
# ---------------------------------------------------------------------------

def blank_ratio(
    logits: Tensor,
    attention_mask: Tensor | None = None,
    *,
    blank_id: int = 0,
) -> float:
    """Fraction of argmax frames emitting the CTC blank.

    Parameters
    ----------
    logits : Tensor
        Shape ``(B, T, V)``.
    attention_mask : Tensor | None
        Shape ``(B, T)``.  If provided, padding frames are excluded
        from the count.
    blank_id : int
        CTC blank token ID.

    Returns
    -------
    float
        Blank fraction in ``[0, 1]``.
    """
    preds = logits.argmax(dim=-1)                       # (B, T)
    is_blank = preds == blank_id
    if attention_mask is not None:
        mask = attention_mask.bool()
        if mask.sum() == 0:
            return 0.0
        return is_blank[mask].float().mean().item()
    return is_blank.float().mean().item()


# ---------------------------------------------------------------------------
# Mean argmax run length — alignment quality proxy
# ---------------------------------------------------------------------------

def mean_argmax_run_length(
    logits: Tensor,
    attention_mask: Tensor | None = None,
) -> float:
    """Average length of consecutive same-token runs in the argmax sequence.

    Short runs → healthy alignment (model transitions between tokens).
    Long runs  → stuck / collapsing (model repeats one token indefinitely).

    Parameters
    ----------
    logits : Tensor
        Shape ``(B, T, V)``.
    attention_mask : Tensor | None
        Shape ``(B, T)``.

    Returns
    -------
    float
        Mean run length across all sequences in the batch.
    """
    preds = logits.argmax(dim=-1)                       # (B, T)
    total_runs = 0
    total_frames = 0

    for i in range(preds.size(0)):
        seq = preds[i]
        if attention_mask is not None:
            length = int(attention_mask[i].sum().item())
            seq = seq[:length]
        else:
            length = seq.size(0)
        if length == 0:
            continue

        changes = int((seq[1:] != seq[:-1]).sum().item())
        total_runs += changes + 1
        total_frames += length

    if total_runs == 0:
        return 0.0
    return total_frames / total_runs


# ---------------------------------------------------------------------------
# Combined PER + per-phoneme recall (single alignment pass)
# ---------------------------------------------------------------------------

def compute_per_and_recall(
    hyps: list[list[int]],
    refs: list[list[int]],
) -> tuple[float, dict[int, dict[str, int | float]], dict[str, int], list[tuple[tuple[int, int], int]]]:
    """Corpus PER, per-phoneme recall, error breakdown, and confusions.

    Equivalent to calling ``compute_per_batch`` + ``per_phoneme_recall``
    separately, but runs ``_align`` only once per sample instead of twice.

    Parameters
    ----------
    hyps : list[list[int]]
        Hypothesis sequences (decoded token IDs).
    refs : list[list[int]]
        Reference sequences (same length as *hyps*).

    Returns
    -------
    (per, recall, error_counts, top_confusions)
        *per* is the corpus-level PER (total edits / total ref tokens).
        *recall* maps ``token_id → {"hits", "total", "recall"}``.
        *error_counts* is ``{"del": int, "ins": int, "sub": int}``.
        *top_confusions* is a list of ``((ref_tok, hyp_tok), count)``
        sorted descending by count (top 20 substitution pairs).
    """
    assert len(hyps) == len(refs), (
        f"FATAL: hyps/refs length mismatch: {len(hyps)} vs {len(refs)}"
    )
    total_edits = 0
    total_ref_len = 0
    hits: dict[int, int] = {}
    totals: dict[int, int] = {}
    n_del = 0
    n_ins = 0
    n_sub = 0
    confusion_counts: dict[tuple[int, int], int] = {}

    for h, r in zip(hyps, refs):
        total_ref_len += len(r)
        if not r and not h:
            continue
        ops = _align(r, h)
        for op, ref_tok, hyp_tok in ops:
            if op != "match":
                total_edits += 1
            if op == "del":
                n_del += 1
            elif op == "ins":
                n_ins += 1
            elif op == "sub":
                n_sub += 1
                pair = (ref_tok, hyp_tok)
                confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
            if ref_tok is not None:
                totals[ref_tok] = totals.get(ref_tok, 0) + 1
                if op == "match":
                    hits[ref_tok] = hits.get(ref_tok, 0) + 1

    per = total_edits / total_ref_len if total_ref_len > 0 else 0.0

    recall: dict[int, dict[str, int | float]] = {}
    for tok_id in sorted(totals):
        h = hits.get(tok_id, 0)
        t = totals[tok_id]
        recall[tok_id] = {
            "hits": h,
            "total": t,
            "recall": h / t if t > 0 else 0.0,
        }

    error_counts = {"del": n_del, "ins": n_ins, "sub": n_sub}
    top_confusions = sorted(confusion_counts.items(), key=lambda x: -x[1])[:20]

    return per, recall, error_counts, top_confusions


# ---------------------------------------------------------------------------
# Per-phoneme recall
# ---------------------------------------------------------------------------

def per_phoneme_recall(
    hyps: list[list[int]],
    refs: list[list[int]],
) -> dict[int, dict[str, int | float]]:
    """Per-phoneme recall via Levenshtein alignment.

    For each phoneme that appears in *refs*, counts how many reference
    occurrences were correctly matched (``"match"``) vs. missed
    (``"sub"`` or ``"del"``).

    Parameters
    ----------
    hyps : list[list[int]]
        Hypothesis sequences (decoded token IDs).
    refs : list[list[int]]
        Reference sequences.

    Returns
    -------
    dict[int, dict[str, int | float]]
        ``{token_id: {"hits": int, "total": int, "recall": float}}``.
        Only phonemes present in *refs* are included, sorted by token ID.
    """
    hits: dict[int, int] = {}
    totals: dict[int, int] = {}

    for h, r in zip(hyps, refs):
        ops = _align(r, h)
        for op, ref_tok, _hyp_tok in ops:
            if ref_tok is None:
                continue                                # insertion — no ref
            totals[ref_tok] = totals.get(ref_tok, 0) + 1
            if op == "match":
                hits[ref_tok] = hits.get(ref_tok, 0) + 1

    result: dict[int, dict[str, int | float]] = {}
    for tok_id in sorted(totals):
        h = hits.get(tok_id, 0)
        t = totals[tok_id]
        result[tok_id] = {
            "hits": h,
            "total": t,
            "recall": h / t if t > 0 else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Alignment formatting — qualitative log inspection (SFT guide §10)
# ---------------------------------------------------------------------------

def format_alignment(
    ref_tokens: list[str],
    hyp_tokens: list[str],
) -> str:
    """Render a REF / HYP alignment with error markers for logging.

    Expects *string* tokens (phoneme symbols, not IDs).
    The caller should convert IDs → strings via the tokenizer before calling.

    Parameters
    ----------
    ref_tokens : list[str]
        Reference phoneme strings.
    hyp_tokens : list[str]
        Hypothesis phoneme strings.

    Returns
    -------
    str
        Multi-line alignment string, e.g.::

            REF:  h ɛ l oʊ | w ɚ l d
            HYP:  h ɛ l oʊ | w ɝ l d
                                ↑ sub
    """
    # Convert to int-indexed lists for alignment (use hash for uniqueness)
    all_tokens = sorted(set(ref_tokens) | set(hyp_tokens))
    tok2id = {t: i for i, t in enumerate(all_tokens)}
    id2tok = {i: t for t, i in tok2id.items()}

    ref_ids = [tok2id[t] for t in ref_tokens]
    hyp_ids = [tok2id[t] for t in hyp_tokens]
    ops = _align(ref_ids, hyp_ids)

    ref_cells: list[str] = []
    hyp_cells: list[str] = []
    err_cells: list[str] = []

    for op, r_id, h_id in ops:
        r_str = id2tok[r_id] if r_id is not None else ""
        h_str = id2tok[h_id] if h_id is not None else ""
        width = max(len(r_str), len(h_str), 1)

        if op == "match":
            ref_cells.append(r_str.ljust(width))
            hyp_cells.append(h_str.ljust(width))
            err_cells.append(" " * width)
        elif op == "sub":
            ref_cells.append(r_str.ljust(width))
            hyp_cells.append(h_str.ljust(width))
            err_cells.append(("↑" + " sub").ljust(width)[:width])
        elif op == "del":
            ref_cells.append(r_str.ljust(width))
            hyp_cells.append("*".ljust(width))
            err_cells.append(("↑" + " del").ljust(width)[:width])
        elif op == "ins":
            ref_cells.append("*".ljust(width))
            hyp_cells.append(h_str.ljust(width))
            err_cells.append(("↑" + " ins").ljust(width)[:width])

    ref_line  = "REF:  " + " ".join(ref_cells)
    hyp_line  = "HYP:  " + " ".join(hyp_cells)
    err_line  = "      " + " ".join(err_cells)

    # Only include error line if there are errors
    if any(c.strip() for c in err_cells):
        return f"{ref_line}\n{hyp_line}\n{err_line}"
    return f"{ref_line}\n{hyp_line}"


# ---------------------------------------------------------------------------
# EMA tracker — used for smoothing grad norms and PER (SFT guide §3)
# ---------------------------------------------------------------------------

class EMATracker:
    """Exponential moving average for scalar signals.

    Parameters
    ----------
    decay : float
        Smoothing factor in ``(0, 1)``.  Higher = more smoothing.
        ``EMA(PER, decay=0.6)`` and ``EMA(grad_norm, decay=0.6)`` per SFT guide.
    """

    __slots__ = ("decay", "_value", "_initialised")

    def __init__(self, decay: float = 0.6) -> None:
        assert 0.0 < decay < 1.0, f"FATAL: EMA decay must be in (0, 1), got {decay}"
        self.decay = decay
        self._value: float = 0.0
        self._initialised: bool = False

    def update(self, x: float) -> float:
        """Feed a new observation and return the smoothed value.

        Parameters
        ----------
        x : float
            Raw observed value.

        Returns
        -------
        float
            Smoothed EMA value after incorporating *x*.
        """
        if not self._initialised:
            self._value = x
            self._initialised = True
        else:
            self._value = self.decay * self._value + (1.0 - self.decay) * x
        return self._value

    @property
    def value(self) -> float:
        """Current smoothed value."""
        return self._value

    # ------------------------------------------------------------------
    # Serialisation — needed for checkpoint resume
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialise internal state for checkpointing."""
        return {
            "decay": self.decay,
            "value": self._value,
            "initialised": self._initialised,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore internal state from a checkpoint dict."""
        self.decay = d["decay"]
        self._value = d["value"]
        self._initialised = d["initialised"]

    def has_plateaued(self, x: float, *, tol: float = 0.01) -> bool:
        """Check if the signal has plateaued (relative change < *tol*).

        Parameters
        ----------
        x : float
            New observation.
        tol : float
            Relative change threshold.  Default 1%.

        Returns
        -------
        bool
            ``True`` if ``|new_ema - old_ema| / max(|old_ema|, 1e-8) < tol``.
        """
        old = self._value
        new = self.update(x)
        return abs(new - old) / max(abs(old), 1e-8) < tol
