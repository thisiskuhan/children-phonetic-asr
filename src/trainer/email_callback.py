"""
Email notification callback for HF Trainer.
============================================

Sends a professional summary email after each evaluation epoch
and on training failure. Uses Gmail SMTP with App Password.

Config (in config.yaml under hf_sft.email):
    enabled: true
    run_name: "run24"

Secrets (in .env):
    EMAIL_SENDER, EMAIL_RECIPIENT, EMAIL_APP_PASSWORD
"""

from __future__ import annotations

import logging
import os
import smtplib
import socket
import traceback
from email.mime.text import MIMEText
from typing import Any

from transformers import TrainerCallback

log = logging.getLogger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587
_TIMEOUT = 15


class EmailNotificationCallback(TrainerCallback):
    """Send email after each eval epoch and on failure."""

    def __init__(self, email_cfg: dict[str, Any]):
        super().__init__()
        self._sender = os.environ["EMAIL_SENDER"]
        self._recipient = os.environ["EMAIL_RECIPIENT"]
        self._password = os.environ["EMAIL_APP_PASSWORD"]
        self._run_name = email_cfg.get("run_name", "training")
        self._best_per = float("inf")
        self._last_train_loss = None
        self._last_grad_norm = None
        self._last_lr = None

    # -- Capture training metrics from logs --
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self._last_train_loss = logs["loss"]
        if "grad_norm" in logs:
            self._last_grad_norm = logs["grad_norm"]
        if "learning_rate" in logs:
            self._last_lr = logs["learning_rate"]

    # -- Send eval summary after each epoch --
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        try:
            self._send_eval_email(state, metrics)
        except Exception as e:
            log.warning("[EMAIL] Failed to send eval email: %s", e)

    # -- Send alert on training failure --
    def on_train_end(self, args, state, control, **kwargs):
        # Check if training ended abnormally (no best metric logged)
        pass

    def _send_eval_email(self, state, metrics: dict) -> None:
        epoch = int(state.epoch) if state.epoch else 0
        step = state.global_step

        # Core metrics
        per = metrics.get("eval_per", None)
        cer = metrics.get("eval_cer", None)
        eval_loss = metrics.get("eval_loss", None)
        blank_ratio = metrics.get("eval_blank_ratio", None)

        # Track best
        is_new_best = False
        if per is not None and per < self._best_per:
            self._best_per = per
            is_new_best = True

        # Error breakdown
        n_del = metrics.get("eval_n_del", None)
        n_ins = metrics.get("eval_n_ins", None)
        n_sub = metrics.get("eval_n_sub", None)

        # Sequence lengths
        mean_hyp = metrics.get("eval_mean_hyp_len", None)
        mean_ref = metrics.get("eval_mean_ref_len", None)
        mean_run = metrics.get("eval_mean_run_len", None)

        # Per-dataset PER
        ds1_per = metrics.get("eval_per_ds/1", None)
        ds2_per = metrics.get("eval_per_ds/2", None)

        # Per-age PER
        age_keys = sorted(k for k in metrics if k.startswith("eval_per_age/"))
        age_per = {k.split("/")[1]: metrics[k] for k in age_keys}

        # Length-bucket PER
        len_keys = sorted(k for k in metrics if k.startswith("eval_per_len/"))
        len_per = {k.split("/")[1]: metrics[k] for k in len_keys}

        # Phoneme health
        dead = metrics.get("eval_dead_phonemes", None)
        worst_recall = metrics.get("eval_worst_phoneme_recall", None)

        # Build subject
        best_tag = " ** NEW BEST **" if is_new_best else ""
        subject = f"[{self._run_name}] Epoch {epoch} — PER {per:.4f}{best_tag}"

        # Build body
        lines = [
            f"Run: {self._run_name}",
            f"Epoch: {epoch}  |  Step: {step:,}",
            "",
            "=" * 50,
            "  EVALUATION RESULTS",
            "=" * 50,
            "",
            f"  PER:          {_fmt(per, '.4f')}{'  ** NEW BEST **' if is_new_best else ''}",
            f"  Best PER:     {_fmt(self._best_per, '.4f')}",
            f"  CER:          {_fmt(cer, '.4f')}",
            f"  Eval Loss:    {_fmt(eval_loss, '.4f')}",
            f"  Blank Ratio:  {_fmt(blank_ratio, '.2%')}",
            "",
            "-" * 50,
            "  TRAINING STATE",
            "-" * 50,
            "",
            f"  Train Loss:   {_fmt(self._last_train_loss, '.4f')}",
            f"  Grad Norm:    {_fmt(self._last_grad_norm, '.4f')}",
            f"  Learning Rate:{_fmt(self._last_lr, '.2e')}",
            "",
            "-" * 50,
            "  ERROR BREAKDOWN",
            "-" * 50,
            "",
        ]

        total_err = (n_del or 0) + (n_ins or 0) + (n_sub or 0)
        if total_err > 0:
            lines.append(f"  Deletions:    {_fmt(n_del, '.0f')}  ({100*n_del/total_err:.1f}%)")
            lines.append(f"  Insertions:   {_fmt(n_ins, '.0f')}  ({100*n_ins/total_err:.1f}%)")
            lines.append(f"  Substitutions:{_fmt(n_sub, '.0f')}  ({100*n_sub/total_err:.1f}%)")
        else:
            lines.append("  No error data available")

        lines += [
            "",
            "-" * 50,
            "  SEQUENCE STATS",
            "-" * 50,
            "",
            f"  Mean Hyp Len: {_fmt(mean_hyp, '.1f')}",
            f"  Mean Ref Len: {_fmt(mean_ref, '.1f')}",
            f"  Mean Run Len: {_fmt(mean_run, '.2f')}",
            "",
        ]

        if ds1_per is not None or ds2_per is not None:
            lines += [
                "-" * 50,
                "  PER BY DATASET",
                "-" * 50,
                "",
                f"  DS1:          {_fmt(ds1_per, '.4f')}",
                f"  DS2:          {_fmt(ds2_per, '.4f')}",
                "",
            ]

        if age_per:
            lines += [
                "-" * 50,
                "  PER BY AGE",
                "-" * 50,
                "",
            ]
            for age, p in sorted(age_per.items()):
                lines.append(f"  {age:12s}  {p:.4f}")
            lines.append("")

        if len_per:
            lines += [
                "-" * 50,
                "  PER BY LENGTH",
                "-" * 50,
                "",
            ]
            for bucket, p in sorted(len_per.items()):
                lines.append(f"  {bucket:12s}  {p:.4f}")
            lines.append("")

        if dead is not None or worst_recall is not None:
            lines += [
                "-" * 50,
                "  PHONEME HEALTH",
                "-" * 50,
                "",
                f"  Dead Phonemes:     {_fmt(dead, '.0f')}",
                f"  Worst Recall:      {_fmt(worst_recall, '.2f')}",
                "",
            ]

        lines += [
            "=" * 50,
            f"  Host: {socket.gethostname()}",
            "=" * 50,
        ]

        body = "\n".join(lines)
        self._send(subject, body)

    def send_failure_alert(self, error: Exception) -> None:
        """Send an alert email when training crashes."""
        subject = f"[{self._run_name}] TRAINING FAILED"
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        body = (
            f"Run: {self._run_name}\n"
            f"Host: {socket.gethostname()}\n\n"
            f"{'=' * 50}\n"
            f"  TRAINING FAILURE ALERT\n"
            f"{'=' * 50}\n\n"
            f"  Error: {error}\n\n"
            f"  Traceback:\n{''.join(tb)}\n"
        )
        self._send(subject, body)

    def _send(self, subject: str, body: str) -> None:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self._sender
        msg["To"] = self._recipient

        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=_TIMEOUT) as srv:
            srv.ehlo()
            srv.starttls()
            srv.ehlo()
            srv.login(self._sender, self._password)
            srv.sendmail(self._sender, [self._recipient], msg.as_string())

        log.info("[EMAIL] Sent: %s", subject)


def _fmt(val, fmt_spec: str) -> str:
    """Format a value, returning '-' if None."""
    if val is None:
        return "-"
    return format(val, fmt_spec)
