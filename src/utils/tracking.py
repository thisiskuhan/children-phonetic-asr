"""
Experiment tracking — shared W&B integration for SFT trainer.
=============================================================

Provides a fully non-blocking, fault-isolated W&B logger that can be
used by any trainer in the package.  All wandb I/O runs on a daemon
background thread — the training loop never waits.

Usage
-----
::

    tracker = WandbTracker(sft_config_dict)
    tracker.init()                          # once, before training
    tracker.log({"train/loss": 0.5}, step=1)  # non-blocking
    tracker.finish()                        # after training

If ``wandb`` is not installed or init fails, every method is a silent
no-op — training is never interrupted.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

# ---------------------------------------------------------------------------
# Optional: Weights & Biases (graceful no-op when not installed)
# ---------------------------------------------------------------------------
try:
    import wandb as _wandb
except ModuleNotFoundError:  # pragma: no cover
    _wandb = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# Maximum queued payloads before dropping.  Prevents silent RAM growth
# if wandb is slow or the network stalls.
_MAX_QUEUE_SIZE: int = 1024


class WandbTracker:
    """Non-blocking, fault-isolated Weights & Biases logger.

    Parameters
    ----------
    config : dict[str, Any]
        The trainer's config dict (e.g. ``cfg["sft"]``).  The ``wandb``
        sub-dict controls project/entity/tags/etc.
    config_section : str
        Name of the config section — used as default project name
        (e.g. ``"sft"`` → project ``"309-sft"``).
    """

    _SENTINEL = object()  # signals the bg thread to stop

    def __init__(
        self,
        config: dict[str, Any],
        *,
        config_section: str = "sft",
    ) -> None:
        self._config = config
        self._section = config_section
        self._enabled = False
        self._queue: queue.Queue[
            tuple[dict[str, Any], int | None] | object
        ] = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._drops: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether W&B tracking is active."""
        return self._enabled

    def init(self, *, metadata: dict[str, Any] | None = None) -> None:
        """Start a W&B run + background logging thread.

        Parameters
        ----------
        metadata : dict, optional
            Extra key-value pairs merged into ``wandb.config`` after init.
            Use for param counts, dataset sizes, hardware info, git hash, etc.

        Synchronous (runs once before training).  If init fails,
        ``enabled`` stays False and all subsequent calls are no-ops.
        """
        wb_cfg = self._config.get("wandb", {})
        self._enabled = (
            bool(wb_cfg.get("enabled"))
            and _wandb is not None
        )
        if not self._enabled:
            if wb_cfg.get("enabled") and _wandb is None:
                log.warning(
                    "[WANDB] wandb.enabled=true but wandb is not installed — "
                    "pip install wandb",
                )
            return

        try:
            default_project = f"309-{self._section}"
            _wandb.init(
                project=wb_cfg.get("project", default_project),
                entity=wb_cfg.get("entity"),
                name=wb_cfg.get("run_name"),
                tags=wb_cfg.get("tags", []),
                config=self._config,
                resume="allow",
            )
            log.info("[WANDB] Run initialised: %s", _wandb.run.url)
            if metadata:
                _wandb.config.update(metadata, allow_val_change=True)
        except Exception:
            log.warning(
                "[WANDB] init failed — training continues without W&B",
                exc_info=True,
            )
            self._enabled = False
            return

        # Start background logging thread (daemon — dies with main process)
        self._thread = threading.Thread(
            target=self._worker,
            name="wandb-logger",
            daemon=True,
        )
        self._thread.start()

    def log(
        self,
        payload: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Enqueue metrics for background W&B logging.

        Returns **immediately** — zero overhead on the training thread.
        Total no-op when wandb is off.
        """
        if not self._enabled:
            return
        try:
            self._queue.put_nowait((payload, step))
        except Exception:
            self._drops += 1
            if self._drops in (1, 10, 100, 1000):
                log.warning(
                    "[WANDB] Dropped %d metric payload(s) — queue full",
                    self._drops,
                )

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        """Signal the background thread to stop, then close the W&B run.

        Parameters
        ----------
        summary : dict, optional
            Final summary metrics (e.g. best_per, best_loss) written to
            ``wandb.run.summary`` before closing.

        Gives the thread up to 30 s to drain remaining metrics.
        Swallows all errors — training result is already secured.
        """
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(self._SENTINEL)
            if self._thread is not None:
                self._thread.join(timeout=30)
            if summary:
                for k, v in summary.items():
                    _wandb.run.summary[k] = v  # type: ignore[union-attr]
            _wandb.finish()  # type: ignore[union-attr]
        except Exception:
            log.debug("[WANDB] finish failed (non-fatal)", exc_info=True)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Drain the queue and call wandb.log() off the training thread.

        Runs until it receives ``_SENTINEL``.  Every wandb call is
        individually wrapped so a single failure doesn't kill the thread.
        """
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            try:
                payload, step = item  # type: ignore[misc]
                _wandb.log(payload, step=step)  # type: ignore[union-attr]
            except Exception:
                log.debug("[WANDB] log failed (non-fatal)", exc_info=True)
