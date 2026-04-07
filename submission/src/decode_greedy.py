"""Greedy CTC decoder — pure tensor ops, zero Python loops over frames.

Called by main.py as fallback if beam search fails or times out.
Also used for the GPU-side decode (fastest possible path).
"""

import torch
import torch.nn.functional as F

BLANK = 0
UNK   = 1


def greedy_batch(
    logits: torch.Tensor,
    output_lengths: torch.Tensor,
) -> list[list[int]]:
    """Batch greedy CTC decode on valid frames only.

    logits         : (B, T, V) — raw logits (pre-softmax OK)
    output_lengths : (B,) — valid frame count per sample

    Returns list of token-id lists (blanks + UNK removed, repeats collapsed).
    """
    ids = logits.argmax(dim=-1)                                    # (B, T)
    frame_idx = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
    ids = ids.masked_fill(frame_idx >= output_lengths.unsqueeze(1), 0)

    shifted = F.pad(ids[:, :-1], (1, 0), value=-1)
    mask = (ids != shifted) & (ids != BLANK) & (ids != UNK)
    ids_cpu = ids.cpu()
    mask_cpu = mask.cpu()
    return [ids_cpu[b][mask_cpu[b]].tolist() for b in range(ids.size(0))]
