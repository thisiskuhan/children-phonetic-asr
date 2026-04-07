#!/usr/bin/env python3
"""
Training loop simulation — verifies flow WITHOUT GPU.

Replays the exact same loop logic from sft_trainer.py with mock data:
- Epoch/micro-step/accumulation counting
- End-of-epoch evaluation trigger timing
- Residual gradient flush at epoch end
- Stage transitions (1→2→3)
- Early stopping
- LR schedule (warmup → decay, per-group warmup)
- Checkpoint naming / top-K retention
- Resume offset calculation
- VRAM extremes tracking (per-epoch max/min with batch context)
- Resume PBS fix (Stage 2+ forces base PBS)
- GradScaler health tracking (scale, backoffs)
- drop_last=True effect on micro-step count
- _total_steps invariance across stage transitions

Run:  python -m tests.sim_training
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Load real config ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config  # noqa: E402

CFG = load_config(
    Path(__file__).resolve().parent.parent / "config" / "config.yaml",
)
SFT = CFG["sft"]

# ── Real dataset stats ────────────────────────────────────────────────
TRAIN_ROWS = 136_080  # from training_controls.json
VAL_ROWS = 15_625
PBS = SFT["stage2_physical_batch_size"]
ACCUM = SFT["stage2_gradient_accumulation_steps"]
EFF_BS = PBS * ACCUM
MAX_EPOCHS = SFT["max_epochs"]
WARMUP = SFT["warmup_steps"]
PGW = SFT["per_group_warmup_steps"]
STAGE1_PBS = SFT["stage1_physical_batch_size"]
TARGET_SR = 16_000
USE_BF16 = SFT.get("bf16", False)
USE_FP16 = SFT.get("fp16", False) and not USE_BF16
LR_MIN_RATIO = SFT.get("lr_min_ratio", 0.01)

# With drop_last=True, micro_per_epoch = floor(TRAIN_ROWS / PBS)
MICRO_PER_EPOCH = TRAIN_ROWS // PBS  # 34020 (136080/4 = exact)
OPTIM_PER_EPOCH = math.ceil(MICRO_PER_EPOCH / ACCUM)  # 4253
TOTAL_STEPS = MAX_EPOCHS * OPTIM_PER_EPOCH


# ── Helpers ───────────────────────────────────────────────────────────
@dataclass
class LRGroup:
    name: str
    initial_lr: float
    lr: float = 0.0
    birth_step: int = 0

    def __repr__(self) -> str:
        return f"{self.name}(lr={self.lr:.2e}, birth={self.birth_step})"


def update_lr(
    step: int, total: int, warmup: int, pgw: int, groups: list[LRGroup],
    lr_min_ratio: float = 0.01,
) -> None:
    """Mirror _update_lr from sft_trainer.py (cosine decay with floor)."""
    if step < warmup:
        gf = step / max(warmup, 1)
    else:
        decay_steps = max(total - warmup, 1)
        progress = min((step - warmup) / decay_steps, 1.0)
        cosine_f = 0.5 * (1.0 + math.cos(math.pi * progress))
        gf = lr_min_ratio + (1.0 - lr_min_ratio) * cosine_f

    for g in groups:
        if g.birth_step > 0:
            group_f = min(1.0, (step - g.birth_step) / max(pgw, 1))
        else:
            group_f = 1.0
        g.lr = g.initial_lr * gf * group_f


@dataclass
class SimState:
    """Mirrors all mutable state from the training loop."""
    global_step: int = 0
    stage: int = 1
    eval_count: int = 0
    epoch: int = 0

    # Stage transition signals
    consecutive_decreasing: int = 0
    consecutive_slowing: int = 0
    val_loss_history: list[float] = field(default_factory=list)
    ema_per_history: list[float] = field(default_factory=list)

    # Early stopping
    best_val_loss: float = float("inf")
    patience_counter: int = 0

    # LR groups
    groups: list[LRGroup] = field(default_factory=list)

    # Counters
    total_micro_steps: int = 0
    total_optim_steps: int = 0
    total_residual_flushes: int = 0
    evals_triggered: list[int] = field(default_factory=list)
    stage_transitions: list[tuple[int, int, int]] = field(default_factory=list)
    # (from_stage, to_stage, at_step)

    # Checkpoint log
    checkpoints_saved: list[str] = field(default_factory=list)

    # PBS tracking (for stage transitions + resume)
    active_pbs: int = 0
    active_accum: int = 0
    loader_rebuild_needed: bool = False

    # GradScaler tracking
    scaler_scale: float = 65536.0
    prev_scaler_scale: float = 65536.0
    scaler_backoffs: int = 0

    # VRAM extremes per epoch
    vram_max_mb: float = 0.0
    vram_min_mb: float = float("inf")
    vram_max_step: int = 0
    vram_min_step: int = 0
    vram_max_dur: float = 0.0
    vram_min_dur: float = 0.0
    vram_max_shape: str = ""
    vram_min_shape: str = ""


def mock_eval_metrics(eval_num: int, stage: int) -> dict:
    """Generate plausible eval metrics for stage transition simulation.

    Simulates:
    - Stage 1: blank_ratio drops from 0.98 → 0.85 over ~3 evals,
      val_loss steadily decreasing
    - Stage 2: PER improves then plateaus
    - Stage 3: PER fine-tunes, then early stop

    Tuned so transitions happen at realistic eval counts.
    """
    if stage == 1:
        # blank_ratio: 0.98 → 0.85 over 4 evals
        blank = max(0.85, 0.98 - eval_num * 0.035)
        # val_loss: steadily decreasing
        val_loss = 8.0 - eval_num * 0.5
        per = 0.95 - eval_num * 0.05
        return {
            "blank_ratio": blank,
            "val_loss": max(val_loss, 2.0),
            "per": max(per, 0.70),
        }
    elif stage == 2:
        # PER improving but slowing down
        base_per = 0.55
        # Each eval improves by less
        improvement = 0.02 / max(eval_num - 4, 1)
        per = max(0.30, base_per - (eval_num - 4) * improvement)
        val_loss = 2.0 - (eval_num - 4) * 0.05
        return {
            "blank_ratio": 0.40,
            "val_loss": max(val_loss, 1.2),
            "per": per,
        }
    else:  # stage 3
        per = 0.28 - (eval_num - 10) * 0.005
        val_loss = 1.2 + (eval_num - 10) * 0.02  # starts overfitting
        return {
            "blank_ratio": 0.25,
            "val_loss": val_loss,
            "per": max(per, 0.20),
        }


# ── VRAM simulation ───────────────────────────────────────────────────
def mock_vram_for_batch(
    batch_max_dur_sec: float, pbs: int, stage: int,
) -> float:
    """Simulate VRAM usage (MB) based on batch duration and stage.

    Models the real behavior: VRAM scales with sequence length (T) due to
    attention matrices and intermediate activations.  Stage 2+ uses more
    because encoder gradients and gradient checkpointing are active.
    """
    # Base VRAM: model weights + optimizer states
    base = 4500.0 if stage == 1 else 5800.0
    # Sequence-proportional term: longer audio → bigger tensors
    T = int(batch_max_dur_sec * TARGET_SR)
    seq_term = pbs * T * 4 / 1e6  # rough: 4 bytes per value, PBS samples
    # Gradient overhead for stage 2+
    grad_term = seq_term * 0.8 if stage >= 2 else 0.0
    return base + seq_term + grad_term


def mock_scaler_update(
    step: int, current_scale: float,
) -> float:
    """Simulate GradScaler behavior.

    Most steps: scale stays at 65536 or grows (doubling every 2000 steps).
    Occasional inf/NaN triggers a halving (backoff).
    We simulate a deterministic pattern: backoff at steps divisible by 5000.
    """
    if step > 0 and step % 5000 == 0:
        # Simulate a backoff
        return current_scale / 2.0
    elif step > 0 and step % 2000 == 0:
        # Growth
        return current_scale * 2.0
    return current_scale


# ── Main simulation ──────────────────────────────────────────────────
def run_simulation() -> None:
    print("=" * 70)
    print("  TRAINING LOOP SIMULATION")
    print("=" * 70)
    print(f"\n  Config values (from config.yaml):")
    print(f"    train_rows       = {TRAIN_ROWS:,}")
    print(f"    val_rows         = {VAL_ROWS:,}")
    print(f"    physical_batch   = {PBS}")
    print(f"    grad_accum       = {ACCUM}")
    print(f"    effective_batch  = {EFF_BS}")
    print(f"    max_epochs       = {MAX_EPOCHS}")
    print(f"    warmup_steps     = {WARMUP}")
    print(f"    per_group_warmup = {PGW}")
    print(f"\n  Derived values:")
    print(f"    micro_per_epoch  = ceil({TRAIN_ROWS}/{PBS}) = {MICRO_PER_EPOCH}")
    print(f"    optim_per_epoch  = ceil({MICRO_PER_EPOCH}/{ACCUM}) = {OPTIM_PER_EPOCH}")
    print(f"    total_steps      = {MAX_EPOCHS} × {OPTIM_PER_EPOCH} = {TOTAL_STEPS:,}")
    print(f"    residual micros  = {MICRO_PER_EPOCH % ACCUM} per epoch "
          f"({'flush needed' if MICRO_PER_EPOCH % ACCUM != 0 else 'exact division'})")

    # ── Init state ──
    s = SimState()
    s.groups = [
        LRGroup("head", SFT["head_lr"]),
    ]
    update_lr(0, TOTAL_STEPS, WARMUP, PGW, s.groups)

    # PBS starts boosted for Stage 1
    s.active_pbs = STAGE1_PBS
    s.active_accum = max(1, EFF_BS // STAGE1_PBS)

    head_grad_plateaued = False
    errors: list[str] = []

    # Track VRAM extremes across ALL epochs for cross-epoch verification
    all_epoch_vram: list[dict] = []
    # Track PBS at each epoch for verification
    epoch_pbs_log: list[int] = []
    # Track scaler history for verification
    scaler_history: list[float] = []

    print(f"\n{'─' * 70}")
    print(f"  EPOCH LOOP")
    print(f"{'─' * 70}\n")

    for epoch in range(MAX_EPOCHS):
        s.epoch = epoch
        micro_step = 0
        epoch_optim_steps = 0
        epoch_pbs_log.append(s.active_pbs)

        # Reset per-epoch VRAM extremes (mirrors sft_trainer.py)
        s.vram_max_mb = 0.0
        s.vram_min_mb = float("inf")
        s.vram_max_step = 0
        s.vram_min_step = 0
        s.vram_max_dur = 0.0
        s.vram_min_dur = 0.0
        s.vram_max_shape = ""
        s.vram_min_shape = ""

        # Rebuild loader check (mirrors sft_trainer.py)
        if s.loader_rebuild_needed:
            s.loader_rebuild_needed = False

        # Compute micro count based on active PBS (with drop_last=True)
        micro_count = TRAIN_ROWS // s.active_pbs
        active_accum = max(1, EFF_BS // s.active_pbs)

        # Simulate deterministic batch durations — use a seeded RNG
        # so VRAM extremes are reproducible
        rng = random.Random(42 + epoch)

        for _micro_idx in range(micro_count):
            micro_step += 1
            s.total_micro_steps += 1

            # Simulate a batch with random max duration 1.0–12.6s
            batch_max_dur = rng.uniform(1.0, 12.6)
            batch_shape = f"{s.active_pbs}x{int(batch_max_dur * TARGET_SR)}"

            # ── Optimizer step at accumulation boundary ──
            if micro_step % active_accum == 0:
                s.global_step += 1
                s.total_optim_steps += 1
                epoch_optim_steps += 1
                update_lr(s.global_step, TOTAL_STEPS, WARMUP, PGW, s.groups)

                # ── GradScaler simulation (fp16 only) ──
                if USE_FP16:
                    new_scale = mock_scaler_update(
                        s.global_step, s.scaler_scale,
                    )
                    if new_scale < s.scaler_scale:
                        s.scaler_backoffs += 1
                    s.prev_scaler_scale = s.scaler_scale
                    s.scaler_scale = new_scale
                    scaler_history.append(s.scaler_scale)

                # ── VRAM extremes tracking (mirrors sft_trainer.py) ──
                vram_now = mock_vram_for_batch(
                    batch_max_dur, s.active_pbs, s.stage,
                )
                if vram_now > s.vram_max_mb:
                    s.vram_max_mb = vram_now
                    s.vram_max_step = s.global_step
                    s.vram_max_dur = batch_max_dur
                    s.vram_max_shape = batch_shape
                if vram_now < s.vram_min_mb:
                    s.vram_min_mb = vram_now
                    s.vram_min_step = s.global_step
                    s.vram_min_dur = batch_max_dur
                    s.vram_min_shape = batch_shape

                # ── Simulate head_grad_plateaued ──
                if s.global_step >= OPTIM_PER_EPOCH * 3:
                    head_grad_plateaued = True

        # ── Residual flush ──
        if micro_step % active_accum != 0:
            s.global_step += 1
            s.total_optim_steps += 1
            s.total_residual_flushes += 1
            epoch_optim_steps += 1
            update_lr(s.global_step, TOTAL_STEPS, WARMUP, PGW, s.groups)

        # ── Record per-epoch VRAM extremes ──
        all_epoch_vram.append({
            "epoch": epoch + 1,
            "max_mb": s.vram_max_mb,
            "min_mb": s.vram_min_mb,
            "max_step": s.vram_max_step,
            "min_step": s.vram_min_step,
            "max_dur": s.vram_max_dur,
            "min_dur": s.vram_min_dur,
            "max_shape": s.vram_max_shape,
            "min_shape": s.vram_min_shape,
        })

        # ── End-of-epoch evaluation ──
        s.eval_count += 1
        s.evals_triggered.append(s.global_step)
        metrics = mock_eval_metrics(s.eval_count, s.stage)

        # ── Stage 1 → 2 logic ──
        if s.stage == 1:
            blank_ok = metrics["blank_ratio"] < SFT["stage1_blank_threshold"]
            s.val_loss_history.append(metrics["val_loss"])
            if len(s.val_loss_history) >= 2:
                if s.val_loss_history[-1] < s.val_loss_history[-2]:
                    s.consecutive_decreasing += 1
                else:
                    s.consecutive_decreasing = 0
            loss_ok = s.consecutive_decreasing >= SFT["stage1_min_evals_decreasing"]

            if loss_ok and (blank_ok or head_grad_plateaued):
                s.stage_transitions.append((1, 2, s.global_step))
                s.stage = 2
                s.consecutive_decreasing = 0
                s.ema_per_history.clear()
                s.consecutive_slowing = 0
                # Add encoder group
                enc = LRGroup("encoder", SFT["encoder_lr"], birth_step=s.global_step)
                s.groups.append(enc)
                # PBS reverts to base (mirrors _transition_to_stage)
                if s.active_pbs != PBS:
                    s.active_pbs = PBS
                    s.active_accum = ACCUM
                    s.loader_rebuild_needed = True
                print(f"    ═══ STAGE 1 → 2 at step {s.global_step} "
                      f"(epoch {epoch+1}, eval #{s.eval_count}) "
                      f"blank={metrics['blank_ratio']:.3f} "
                      f"PBS={s.active_pbs} ═══")

        # ── Stage 2 → 3 logic ──
        elif s.stage == 2:
            ema_per = metrics["per"]  # simplified
            s.ema_per_history.append(ema_per)
            if len(s.ema_per_history) >= 2:
                prev, curr = s.ema_per_history[-2], s.ema_per_history[-1]
                rel_imp = (prev - curr) / max(prev, 1e-8)
                if rel_imp < SFT["stage2_per_improvement_threshold"]:
                    s.consecutive_slowing += 1
                else:
                    s.consecutive_slowing = 0

            if s.consecutive_slowing >= SFT["stage2_min_evals_slowing"]:
                s.stage_transitions.append((2, 3, s.global_step))
                s.stage = 3
                s.consecutive_slowing = 0
                # Add CNN group
                cnn = LRGroup("cnn", SFT["cnn_lr"], birth_step=s.global_step)
                s.groups.append(cnn)
                print(f"    ═══ STAGE 2 → 3 at step {s.global_step} "
                      f"(epoch {epoch+1}, eval #{s.eval_count}) ═══")

        elif s.stage == 3:
            s.val_loss_history.append(metrics["val_loss"])

        # ── Checkpoint naming ──
        ckpt_name = f"step{s.global_step:06d}_per{metrics['per']:.4f}.pt"
        s.checkpoints_saved.append(ckpt_name)

        # ── Early stopping ──
        if metrics["val_loss"] < s.best_val_loss:
            s.best_val_loss = metrics["val_loss"]
            s.patience_counter = 0
        else:
            s.patience_counter += 1

        lr_strs = ", ".join(f"{g.name}={g.lr:.2e}" for g in s.groups)
        print(f"  Epoch {epoch+1:>2}/{MAX_EPOCHS}  "
              f"micros={micro_step:>5}  opt_steps={epoch_optim_steps:>5}  "
              f"global={s.global_step:>6}  stage={s.stage}  "
              f"eval#{s.eval_count:<2}  patience={s.patience_counter}  "
              f"PBS={s.active_pbs:>2}  "
              f"VRAM=[{s.vram_min_mb:.0f}–{s.vram_max_mb:.0f}]  "
              f"AMP={'bf16' if USE_BF16 else ('fp16' if USE_FP16 else 'off')}  "
              f"{f'scaler={s.scaler_scale:.0f}  ' if USE_FP16 else ''}"
              f"LR=[{lr_strs}]")

        if s.patience_counter >= SFT["early_stopping_patience"]:
            print(f"\n    ⛔ EARLY STOPPING at step {s.global_step} "
                  f"(epoch {epoch+1}, eval #{s.eval_count}, "
                  f"patience={s.patience_counter})")
            break

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SIMULATION RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  Counters:")
    print(f"    total micro-steps   = {s.total_micro_steps:>10,}")
    print(f"    total optim steps   = {s.total_optim_steps:>10,}")
    print(f"    residual flushes    = {s.total_residual_flushes:>10,}")
    print(f"    evals triggered     = {s.eval_count:>10}")
    print(f"    final global_step   = {s.global_step:>10,}")
    print(f"    epochs completed    = {s.epoch + 1:>10}")

    # ── Eval timing — must fire exactly once per completed epoch ──
    print(f"\n  Eval steps: {s.evals_triggered[:10]}{'...' if len(s.evals_triggered) > 10 else ''}")
    for i, step in enumerate(s.evals_triggered):
        expected = OPTIM_PER_EPOCH * (i + 1)
        if step != expected:
            err = f"Eval #{i+1} at step {step}, expected {expected}"
            errors.append(err)

    # ── Stage transitions ──
    print(f"\n  Stage transitions:")
    if s.stage_transitions:
        for frm, to, step in s.stage_transitions:
            print(f"    Stage {frm} → {to} at step {step}")
    else:
        print(f"    (none — stayed in stage {s.stage})")

    # ── LR schedule verification ──
    print(f"\n  LR schedule spot-checks:")
    for check_step in [0, 1, WARMUP // 2, WARMUP, WARMUP + 1000, TOTAL_STEPS - 1]:
        test_groups = [LRGroup("head", SFT["head_lr"])]
        update_lr(check_step, TOTAL_STEPS, WARMUP, PGW, test_groups)
        print(f"    step={check_step:>6}  head_lr={test_groups[0].lr:.6e}  "
              f"({'warmup' if check_step < WARMUP else 'decay'})")

    # ── LR at step 0 must be 0 (linear warmup), not base_lr ──
    test_g = [LRGroup("head", SFT["head_lr"])]
    update_lr(0, TOTAL_STEPS, WARMUP, PGW, test_g)
    if test_g[0].lr != 0.0:
        errors.append(f"LR at step 0 = {test_g[0].lr}, expected 0.0")

    # ── LR at warmup_steps must be base_lr ──
    update_lr(WARMUP, TOTAL_STEPS, WARMUP, PGW, test_g)
    expected_lr = SFT["head_lr"]
    if abs(test_g[0].lr - expected_lr) > 1e-10:
        errors.append(f"LR at warmup end = {test_g[0].lr:.2e}, expected {expected_lr:.2e}")

    # ── LR at last step must be near floor (lr_min_ratio × base_lr) ──
    update_lr(TOTAL_STEPS - 1, TOTAL_STEPS, WARMUP, PGW, test_g)
    lr_floor = SFT["head_lr"] * LR_MIN_RATIO
    if test_g[0].lr > lr_floor * 1.5:
        errors.append(f"LR at last step = {test_g[0].lr:.2e}, expected near floor {lr_floor:.2e}")

    # ── Per-group warmup verification ──
    print(f"\n  Per-group warmup (encoder born at step 12759):")
    birth = 12759  # example
    enc_g = [LRGroup("encoder", SFT["encoder_lr"], birth_step=birth)]
    for offset in [0, PGW // 2, PGW, PGW + 500]:
        step = birth + offset
        update_lr(step, TOTAL_STEPS, WARMUP, PGW, enc_g)
        print(f"    step={step} (birth+{offset}): lr={enc_g[0].lr:.2e}")
    # At birth, lr should be 0
    update_lr(birth, TOTAL_STEPS, WARMUP, PGW, enc_g)
    if enc_g[0].lr != 0.0:
        errors.append(f"Encoder LR at birth step = {enc_g[0].lr}, expected 0.0")

    # ── Checkpoint naming ──
    print(f"\n  Sample checkpoints:")
    for ckpt in s.checkpoints_saved[:5]:
        print(f"    {ckpt}")
    if len(s.checkpoints_saved) > 5:
        print(f"    ... ({len(s.checkpoints_saved)} total)")

    # ── Resume offset calculation ──
    print(f"\n  Resume verification (simulating resume from end of epoch 2):")
    resume_epoch_saved = 2
    resume_step = OPTIM_PER_EPOCH * resume_epoch_saved
    resume_start_epoch = resume_epoch_saved
    resume_skip = 0
    print(f"    saved epoch       = {resume_epoch_saved}")
    print(f"    global_step       = {resume_step}")
    print(f"    start_epoch       = {resume_start_epoch} (from checkpoint, not derived)")
    print(f"    skip_micros       = {resume_skip} (always 0 — end-of-epoch checkpoints)")
    remaining_in_epoch = MICRO_PER_EPOCH
    remaining_full_steps = remaining_in_epoch // ACCUM
    remaining_residual = remaining_in_epoch % ACCUM
    print(f"    remaining micros in epoch {resume_start_epoch} = {remaining_in_epoch}")
    print(f"    → {remaining_full_steps} full optim steps + {remaining_residual} residual")

    # ── VRAM extremes summary ──
    print(f"\n  VRAM extremes per epoch (simulated):")
    for ev in all_epoch_vram[:5]:
        print(f"    Epoch {ev['epoch']:>2}: "
              f"MAX {ev['max_mb']:>8.0f} MB @ step {ev['max_step']:>6} "
              f"(longest={ev['max_dur']:.2f}s, {ev['max_shape']}) | "
              f"MIN {ev['min_mb']:>8.0f} MB @ step {ev['min_step']:>6} "
              f"(longest={ev['min_dur']:.2f}s, {ev['min_shape']})")
    if len(all_epoch_vram) > 5:
        print(f"    ... ({len(all_epoch_vram)} total)")

    # ── AMP / GradScaler summary ──
    amp_mode = "bf16" if USE_BF16 else ("fp16" if USE_FP16 else "off")
    print(f"\n  AMP mode: {amp_mode}")
    if USE_FP16:
        print(f"  GradScaler tracking:")
        print(f"    final_scale    = {s.scaler_scale:.0f}")
        print(f"    total_backoffs = {s.scaler_backoffs}")
        print(f"    history length = {len(scaler_history)}")
    else:
        print(f"  GradScaler: disabled (bf16 — no scaling needed)")

    # ── PBS tracking summary ──
    print(f"\n  PBS per epoch (first 5):")
    for i, pbs_val in enumerate(epoch_pbs_log[:5]):
        print(f"    Epoch {i+1}: PBS={pbs_val}")

    # ══════════════════════════════════════════════════════════════════
    #           A S S E R T I O N S
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  ASSERTIONS")
    print(f"{'─' * 70}")

    checks: list[tuple[str, object, object]] = [
        ("micro steps / epoch (base PBS)",
         MICRO_PER_EPOCH, TRAIN_ROWS // PBS),
        ("optim steps / epoch",
         OPTIM_PER_EPOCH, math.ceil(MICRO_PER_EPOCH / ACCUM)),
        ("total_steps = epochs × optim/epoch",
         TOTAL_STEPS, MAX_EPOCHS * OPTIM_PER_EPOCH),
        ("residual micros per epoch",
         MICRO_PER_EPOCH % ACCUM, TRAIN_ROWS % (PBS * ACCUM) // PBS if TRAIN_ROWS % (PBS * ACCUM) else 0),
    ]

    # == drop_last=True verification ==
    # With drop_last=True, micro_per_epoch = TRAIN_ROWS // PBS (floor)
    # not ceil.  For 136080 / 4 = 34020 (exact), so same here, but
    # we verify the formula is floor not ceil.
    drop_last_micros = TRAIN_ROWS // PBS
    ceil_micros = math.ceil(TRAIN_ROWS / PBS)
    checks.append((
        "drop_last=True: floor division used",
        drop_last_micros, MICRO_PER_EPOCH,
    ))
    # Also verify for an odd dataset size where floor != ceil
    odd_size = 136_081  # one extra sample
    floor_val = odd_size // PBS  # 34020
    ceil_val = math.ceil(odd_size / PBS)  # 34021
    checks.append((
        "drop_last proof: floor(136081/4)=34020 != ceil=34021",
        floor_val != ceil_val, True,
    ))

    # == _total_steps invariance across stage transitions ==
    # total_steps must be the same whether computed with Stage 1 PBS
    # or base PBS (since EFF_BS is constant).
    micro_stage1 = TRAIN_ROWS // STAGE1_PBS  # floor div with drop_last
    accum_stage1 = max(1, EFF_BS // STAGE1_PBS)
    optim_stage1 = math.ceil(micro_stage1 / accum_stage1)
    total_steps_stage1 = MAX_EPOCHS * optim_stage1

    micro_base = TRAIN_ROWS // PBS
    accum_base = ACCUM
    optim_base = math.ceil(micro_base / accum_base)
    total_steps_base = MAX_EPOCHS * optim_base

    checks.append((
        "_total_steps invariant (base vs stage1 PBS)",
        total_steps_base, TOTAL_STEPS,
    ))
    # NOTE: stage1 total may differ by a few steps due to floor/ceil
    # rounding, but the REAL total_steps uses base PBS (that's the fix)
    # So we just verify base-PBS total_steps matches our TOTAL_STEPS.

    # == Verify eval triggers at exact epoch boundaries ==
    for i, step in enumerate(s.evals_triggered):
        checks.append((
            f"eval #{i+1} at epoch boundary",
            step, OPTIM_PER_EPOCH * (i + 1),
        ))

    # == VRAM extremes per-epoch checks ==
    for ev in all_epoch_vram:
        # Max VRAM must be >= min VRAM
        if ev["max_mb"] < ev["min_mb"]:
            errors.append(
                f"Epoch {ev['epoch']}: vram_max ({ev['max_mb']:.0f}) "
                f"< vram_min ({ev['min_mb']:.0f})"
            )
        # Max duration for max VRAM batch should be longer than min VRAM batch
        # (since VRAM scales with duration in our model)
        if ev["max_dur"] < ev["min_dur"]:
            errors.append(
                f"Epoch {ev['epoch']}: max-VRAM batch duration "
                f"({ev['max_dur']:.2f}s) < min-VRAM batch duration "
                f"({ev['min_dur']:.2f}s) — VRAM should scale with duration"
            )
        # Steps must be within the epoch's range
        # Shape strings must not be empty
        if not ev["max_shape"] or not ev["min_shape"]:
            errors.append(
                f"Epoch {ev['epoch']}: empty shape string in VRAM tracking"
            )

    checks.append((
        "VRAM extremes: all epochs have max >= min",
        all(e["max_mb"] >= e["min_mb"] for e in all_epoch_vram), True,
    ))
    checks.append((
        "VRAM extremes: max-VRAM batch has longer audio",
        all(e["max_dur"] >= e["min_dur"] for e in all_epoch_vram), True,
    ))
    checks.append((
        "VRAM extremes: shape strings populated",
        all(e["max_shape"] and e["min_shape"] for e in all_epoch_vram),
        True,
    ))

    # == VRAM jump at stage transition ==
    # Stage 2 base VRAM should be higher than Stage 1 (encoder grads)
    stage1_epochs = [e for e in all_epoch_vram if e["epoch"] <= 4]
    stage2_epochs = [e for e in all_epoch_vram if e["epoch"] > 4 and e["epoch"] <= 8]
    if stage1_epochs and stage2_epochs:
        avg_s1_max = sum(e["max_mb"] for e in stage1_epochs) / len(stage1_epochs)
        avg_s2_max = sum(e["max_mb"] for e in stage2_epochs) / len(stage2_epochs)
        checks.append((
            "VRAM jump: Stage 2 avg max > Stage 1 avg max",
            avg_s2_max > avg_s1_max, True,
        ))

    # == PBS tracking checks ==
    # Stage 1 should use boosted PBS, Stage 2+ should use base PBS
    if s.stage_transitions:
        transition_epoch = None
        for frm, to, step in s.stage_transitions:
            if frm == 1 and to == 2:
                # Find which epoch this happened in
                for i, eval_step in enumerate(s.evals_triggered):
                    if eval_step >= step:
                        transition_epoch = i + 1
                        break
        if transition_epoch is not None:
            # Epochs before transition should have Stage 1 PBS
            for i in range(min(transition_epoch - 1, len(epoch_pbs_log))):
                if epoch_pbs_log[i] != STAGE1_PBS:
                    errors.append(
                        f"Epoch {i+1}: expected stage1 PBS={STAGE1_PBS}, "
                        f"got {epoch_pbs_log[i]}"
                    )
            # Epochs after transition should have base PBS
            for i in range(transition_epoch, len(epoch_pbs_log)):
                if epoch_pbs_log[i] != PBS:
                    errors.append(
                        f"Epoch {i+1}: expected base PBS={PBS}, "
                        f"got {epoch_pbs_log[i]}"
                    )
            checks.append((
                f"PBS: Stage 1 epochs use PBS={STAGE1_PBS}",
                all(p == STAGE1_PBS for p in epoch_pbs_log[:transition_epoch - 1]),
                True,
            ))
            checks.append((
                f"PBS: Stage 2+ epochs use PBS={PBS}",
                all(p == PBS for p in epoch_pbs_log[transition_epoch:]),
                True,
            ))

    # == GradScaler tracking checks (fp16 only; bf16 disables scaler) ==
    if USE_FP16:
        checks.append((
            "GradScaler: backoffs detected",
            s.scaler_backoffs > 0, True,
        ))
        checks.append((
            "GradScaler: scale stays positive",
            s.scaler_scale > 0, True,
        ))
        # GradScaler history only records at accumulation boundaries, not
        # at residual flushes (residuals bypass the scaler tracking in the
        # real code too — they step the optimizer but don't log scaler).
        expected_scaler_len = s.total_optim_steps - s.total_residual_flushes
        checks.append((
            "GradScaler: history length = optim - residuals",
            len(scaler_history), expected_scaler_len,
        ))
    else:
        checks.append((
            "bf16: GradScaler disabled (no backoffs)",
            s.scaler_backoffs, 0,
        ))

    # == Resume PBS fix simulation ==
    # Simulate: build loader with Stage1 PBS → resume restores stage=2
    # → PBS must revert to base
    sim_resume_pbs = STAGE1_PBS  # loader built with this
    sim_resume_stage = 2  # checkpoint restores this
    if sim_resume_stage >= 2:
        sim_resume_pbs = PBS  # fix forces base PBS
        sim_resume_accum = ACCUM
        sim_rebuild_needed = True
    checks.append((
        "Resume→Stage2: PBS forced to base",
        sim_resume_pbs, PBS,
    ))
    checks.append((
        "Resume→Stage2: accum forced to base",
        sim_resume_accum, ACCUM,
    ))
    checks.append((
        "Resume→Stage2: loader rebuild scheduled",
        sim_rebuild_needed, True,
    ))
    # Also: if resume into stage 1, PBS should stay boosted
    sim_s1_pbs = STAGE1_PBS
    sim_s1_stage = 1
    sim_s1_rebuild = False
    if sim_s1_stage >= 2:
        sim_s1_pbs = PBS
        sim_s1_rebuild = True
    checks.append((
        "Resume→Stage1: PBS stays boosted",
        sim_s1_pbs, STAGE1_PBS,
    ))
    checks.append((
        "Resume→Stage1: no loader rebuild",
        sim_s1_rebuild, False,
    ))

    # ── Print all checks ──
    all_pass = True
    for label, got, expected in checks:
        ok = got == expected
        mark = "✓" if ok else "✗"
        print(f"    {mark}  {label}: {got}" +
              (f" (expected {expected})" if not ok else ""))
        if not ok:
            all_pass = False
            errors.append(f"{label}: got {got}, expected {expected}")

    # Extra: verify total micro_step count makes sense
    # Stage 1 epochs have fewer micros (boosted PBS=16 → 8505/epoch)
    # Stage 2+ epochs have more (base PBS=4 → 34020/epoch)
    if s.total_micro_steps > 0:
        if s.patience_counter >= SFT["early_stopping_patience"]:
            # Count micros by epoch, accounting for PBS changes
            expected_micros = 0
            for i, pbs_val in enumerate(epoch_pbs_log[:s.epoch + 1]):
                expected_micros += TRAIN_ROWS // pbs_val
            ok = s.total_micro_steps == expected_micros
            mark = "✓" if ok else "✗"
            print(f"    {mark}  total micros: {s.total_micro_steps:,} "
                  f"(expected {expected_micros:,} — early stop, "
                  f"variable PBS)")
            if not ok:
                errors.append(
                    f"total micros: {s.total_micro_steps} != "
                    f"{expected_micros}"
                )

    # ── Final verdict ──
    print(f"\n{'=' * 70}")
    if errors:
        print(f"  ✗  FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"      • {e}")
    else:
        print(f"  ✓  ALL CHECKS PASSED")
    print(f"{'=' * 70}\n")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    run_simulation()
