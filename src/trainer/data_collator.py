"""
Data collator — audio loading, preprocessing, and dynamic batching.
===================================================================

**SFTCollator** — supervised fine-tuning (CTC):
  Loads audio, preprocesses, tokenises phonetic text, pads labels with -100.
  Optional augmentation: speed perturbation, MUSAN noise, RIR reverb, pitch shift.

Per-sample preprocessing:
  1. Load raw audio via ``soundfile``
  2. Resample to 16 kHz if native SR differs  (85 % of corpus)
  3. Mono downmix if multi-channel            (23 % of corpus)
  4. Silence trim — remove leading/trailing silence below threshold
  5. Absolute silence floor — zero sub-audible residual noise (<1e-4)
  6. Optional speed perturbation
  7. Zero-mean unit-variance normalisation (matches WavLM pretraining)
  8. Floor-pad short clips

Batch-level:
  9. Dynamic padding — pad waveforms to longest in batch
 10. Attention mask — ``long`` dtype (critical — WavLM silent bug if bool/float)

DS1/DS2-aware augmentation (Run 22+):
  Noise/RIR gated by ``noise_datasets`` (e.g. ``[2]`` = DS2 only).
  Pitch gated **separately** by ``pitch_datasets`` (e.g. ``None`` = all).
  Per-dataset semitone range via ``pitch_semitones_per_dataset`` dict.
  Speed perturbation applies to ALL datasets regardless.

**NOT here:** SpecAugment — applied inside the model's ``forward()`` pass.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import soundfile as sf

import torch
import torchaudio
import torchaudio.functional as F
from torch import Tensor
from transformers import Wav2Vec2CTCTokenizer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SFTCollator
# ---------------------------------------------------------------------------

class SFTCollator:
    """Collate function for SFT training and evaluation.

    Parameters
    ----------
    tokenizer : Wav2Vec2CTCTokenizer
        Pre-built tokenizer loaded from ``data/models/tokenizer/``.
    target_sr : int
        Target sample rate (always 16 000 for WavLM).
    label_pad : int
        Padding value for labels — CTC loss ignores these (default ``-100``).
    """

    __slots__ = (
        "_tokenizer", "_target_sr", "_label_pad",
        "_resamplers", "_min_samples", "_max_samples",
        "_max_duration_sec",
        "_speed_perturb", "_speed_low", "_speed_high",
        "_noise_cache", "_noise_prob",
        "_rir_cache", "_rir_prob",
        "_pitch_prob", "_pitch_semitones",
        "_noise_datasets",
        "_pitch_datasets", "_pitch_semitones_map",
        "_silence_trim", "_silence_trim_db", "_silence_trim_abs_floor",
    )

    def __init__(
        self,
        tokenizer: Wav2Vec2CTCTokenizer,
        *,
        target_sr: int = 16_000,
        label_pad: int = -100,
        min_duration_sec: float = 1.0,
        max_duration_sec: float | None = None,
        speed_perturb: bool = False,
        speed_perturb_range: tuple[float, float] = (0.9, 1.1),
        noise_dir: str | Path | None = None,
        noise_prob: float = 0.0,
        rir_dir: str | Path | None = None,
        rir_prob: float = 0.0,
        pitch_prob: float = 0.0,
        pitch_semitones: float = 2.0,
        noise_datasets: list[int] | None = None,
        pitch_datasets: list[int] | None = None,
        pitch_semitones_per_dataset: dict[str, float] | None = None,
        silence_trim: bool = False,
        silence_trim_db: float = -40.0,
        silence_trim_abs_floor: float = 0.0,
    ) -> None:
        self._tokenizer = tokenizer
        self._target_sr = target_sr
        self._label_pad = label_pad
        self._speed_perturb = speed_perturb
        self._speed_low, self._speed_high = speed_perturb_range
        self._pitch_prob = pitch_prob
        self._pitch_semitones = pitch_semitones
        self._silence_trim = silence_trim
        self._silence_trim_db = silence_trim_db
        self._silence_trim_abs_floor = silence_trim_abs_floor
        # Which datasets receive noise/RIR augmentation.
        # None = all datasets.  E.g. [2] = DS2 only.
        self._noise_datasets: frozenset[str] | None = (
            frozenset(str(d) for d in noise_datasets)
            if noise_datasets is not None else None
        )
        # Which datasets receive pitch augmentation (separate from noise/RIR).
        # None = all datasets.  Pitch is SNR-safe so can apply to all.
        self._pitch_datasets: frozenset[str] | None = (
            frozenset(str(d) for d in pitch_datasets)
            if pitch_datasets is not None else None
        )
        # Per-dataset pitch semitone range — overrides _pitch_semitones
        # for specific datasets.  Values can be:
        #   float → symmetric ±N semitones
        #   dict {low, high} → asymmetric range (e.g. {low: -2, high: 10})
        self._pitch_semitones_map: dict[str, tuple[float, float]] = {}
        if pitch_semitones_per_dataset:
            for k, v in pitch_semitones_per_dataset.items():
                if isinstance(v, dict):
                    self._pitch_semitones_map[str(k)] = (float(v["low"]), float(v["high"]))
                else:
                    self._pitch_semitones_map[str(k)] = (-float(v), float(v))
        # Minimum waveform length in samples.  Short files are zero-padded
        # to this floor so WavLM's conv feature extractor (320× downsample)
        # always produces enough output frames for CTC loss.
        self._min_samples = int(min_duration_sec * target_sr)
        # Maximum waveform length in samples.  Samples exceeding this are
        # DROPPED (never truncated) — serves as secondary safety check in
        # case ETL missed any long files or manifest durations were wrong.
        # When speed perturbation is active, scale by 1/speed_low + 5%
        # headroom so that a file originally at max_duration slowed to
        # minimum speed still passes with margin for resampling rounding
        # (e.g. 13s / 0.9 * 1.05 ≈ 15.17s).
        self._max_duration_sec = max_duration_sec
        if max_duration_sec:
            effective_max = (
                max_duration_sec / speed_perturb_range[0] * 1.05
                if speed_perturb else max_duration_sec
            )
            self._max_samples = int(effective_max * target_sr)
            log.info(
                "[COLLATOR] max_duration=%.2fs  effective_max=%.2fs (%d samples)%s",
                max_duration_sec, effective_max, self._max_samples,
                f"  (speed_perturb [{speed_perturb_range[0]:.2f}–{speed_perturb_range[1]:.2f}] + 5%% headroom)"
                if speed_perturb else "",
            )
        else:
            self._max_samples = None
        # Cache Resample transforms keyed by source SR.  Avoids
        # recomputing the sinc-interpolation filter kernel (85% of
        # corpus needs resampling to 16 kHz).
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        # -- MUSAN noise files pre-loaded into RAM (~1-2 GB for 930 files) --
        self._noise_cache: list[Tensor] = []
        self._noise_prob = noise_prob
        if noise_dir is not None and noise_prob > 0:
            noise_root = Path(noise_dir)
            assert noise_root.is_dir(), (
                f"FATAL: noise_dir not found: {noise_root}"
            )
            noise_paths = sorted(str(p) for p in noise_root.rglob("*.wav"))
            assert len(noise_paths) > 0, (
                f"FATAL: no .wav files found in noise_dir: {noise_root}"
            )
            if len(noise_paths) < 500:
                log.warning(
                    "[COLLATOR] Only %d noise files in %s — "
                    "MUSAN /noise/ should have ~930.  Wrong path?",
                    len(noise_paths), noise_root,
                )
            for p in noise_paths:
                data_n, sr_n = sf.read(p, dtype="float32", always_2d=True)
                if data_n.shape[1] > 1:
                    data_n = data_n.mean(axis=1, keepdims=True)
                wav_n = torch.from_numpy(data_n.T)  # (channels, samples)
                if sr_n != target_sr:
                    if sr_n not in self._resamplers:
                        self._resamplers[sr_n] = torchaudio.transforms.Resample(
                            orig_freq=sr_n, new_freq=target_sr,
                        )
                    wav_n = self._resamplers[sr_n](wav_n)
                self._noise_cache.append(wav_n.squeeze(0))
            log.info(
                "[COLLATOR] MUSAN noise cached — %d files in RAM from %s  "
                "(noise_prob=%.2f)",
                len(self._noise_cache), noise_root, noise_prob,
            )
        elif noise_prob > 0:
            log.warning(
                "[COLLATOR] noise_prob=%.2f but noise_dir is None — "
                "noise augmentation DISABLED",
                noise_prob,
            )
            self._noise_prob = 0.0

        # -- RIR files pre-loaded into RAM --
        self._rir_cache: list[Tensor] = []
        self._rir_prob = rir_prob
        if rir_dir is not None and rir_prob > 0:
            rir_root = Path(rir_dir)
            assert rir_root.is_dir(), (
                f"FATAL: rir_dir not found: {rir_root}"
            )
            rir_paths = sorted(str(p) for p in rir_root.rglob("*.wav"))
            assert len(rir_paths) > 0, (
                f"FATAL: no .wav files found in rir_dir: {rir_root}"
            )
            # Subsample to cap RAM — 60k RIRs × 8 workers via fork = ~48 GB.
            # 1000 diverse RIRs gives equivalent augmentation quality.
            _MAX_RIR = 1000
            _n_total_rir = len(rir_paths)
            if _n_total_rir > _MAX_RIR:
                _rir_rng = random.Random(42)
                rir_paths = _rir_rng.sample(rir_paths, _MAX_RIR)
                log.info(
                    "[COLLATOR] RIR subsampled %d → %d to limit worker RAM",
                    _n_total_rir, _MAX_RIR,
                )
            failed = 0
            for p in rir_paths:
                try:
                    data, rir_sr = sf.read(p, dtype="float32")
                except Exception:
                    failed += 1
                    continue
                rir_wav = torch.from_numpy(data)
                if rir_wav.ndim == 2:
                    rir_wav = rir_wav.mean(dim=1)
                if rir_sr != target_sr:
                    if rir_sr not in self._resamplers:
                        self._resamplers[rir_sr] = torchaudio.transforms.Resample(
                            orig_freq=rir_sr, new_freq=target_sr,
                        )
                    rir_wav = self._resamplers[rir_sr](rir_wav.unsqueeze(0)).squeeze(0)
                # Truncate RIR to 0.5s — captures early reflections and
                # room decay; tail beyond this is near-silence that only
                # inflates FFT size (320k+16k → 512k vs 320k+8k → 344k).
                _MAX_RIR_SAMPLES = target_sr // 2  # 8000 @ 16kHz
                if rir_wav.size(0) > _MAX_RIR_SAMPLES:
                    rir_wav = rir_wav[:_MAX_RIR_SAMPLES]
                # Normalise RIR so peak = 1
                rir_max = rir_wav.abs().max()
                if rir_max > 1e-8:
                    rir_wav = rir_wav / rir_max
                self._rir_cache.append(rir_wav)
            if failed:
                log.warning(
                    "[COLLATOR] %d / %d RIR files failed to load",
                    failed, len(rir_paths),
                )
            log.info(
                "[COLLATOR] RIR cached — %d files in RAM from %s  "
                "(rir_prob=%.2f)",
                len(self._rir_cache), rir_root, rir_prob,
            )
        elif rir_prob > 0:
            log.warning(
                "[COLLATOR] rir_prob=%.2f but rir_dir is None — "
                "RIR augmentation DISABLED",
                rir_prob,
            )
            self._rir_prob = 0.0

    # ------------------------------------------------------------------
    # Silence trimming
    # ------------------------------------------------------------------

    def _trim_silence(self, wav: Tensor) -> Tensor:
        """Remove leading/trailing silence below ``_silence_trim_db``.

        Uses energy-based detection: frames below the dB threshold
        relative to peak amplitude are considered silence.  Preserves
        a minimum of ``_min_samples`` to avoid producing empty tensors.

        Parameters
        ----------
        wav : Tensor
            1-D float32 waveform.

        Returns
        -------
        Tensor
            Trimmed 1-D waveform (may be same object if no trim needed).
        """
        n = wav.size(0)
        if n < self._min_samples:
            return wav

        # Compute per-sample energy in dB relative to peak
        abs_wav = wav.abs()
        peak = abs_wav.max()
        if peak < 1e-10:
            return wav  # near-digital-silence — nothing to trim

        # Threshold in linear amplitude
        threshold = peak * (10.0 ** (self._silence_trim_db / 20.0))

        # Find first and last sample above threshold
        above = abs_wav > threshold
        nonzero = torch.nonzero(above, as_tuple=False)
        if nonzero.numel() == 0:
            return wav  # entire signal below threshold — keep as-is

        start = nonzero[0].item()
        end = nonzero[-1].item() + 1  # exclusive

        # Ensure minimum length after trim
        if (end - start) < self._min_samples:
            # Centre the minimum window around the signal midpoint
            mid = (start + end) // 2
            half = self._min_samples // 2
            start = max(0, mid - half)
            end = min(n, start + self._min_samples)
            start = max(0, end - self._min_samples)

        if start == 0 and end == n:
            return wav  # no trim needed

        return wav[start:end].contiguous()

    def _apply_abs_floor(self, wav: Tensor) -> Tensor:
        """Clamp samples below an absolute amplitude floor to zero.

        Kills sub-audible residual noise that relative-dB trim leaves
        behind.  53% of test files are >50% silent, so the model must
        learn to handle near-zero regions cleanly.

        Parameters
        ----------
        wav : Tensor
            1-D float32 waveform.

        Returns
        -------
        Tensor
            Waveform with sub-floor samples zeroed.
        """
        if self._silence_trim_abs_floor <= 0.0:
            return wav
        mask = wav.abs() < self._silence_trim_abs_floor
        if mask.any():
            wav = wav.clone()
            wav[mask] = 0.0
        return wav

    # ------------------------------------------------------------------
    # Per-sample audio preprocessing
    # ------------------------------------------------------------------

    def _load_and_preprocess(self, audio_path: str) -> Tensor:
        """Load → mono → resample → zero-mean unit-variance normalise.

        Parameters
        ----------
        audio_path : str
            Absolute path to audio file (FLAC, WAV, etc.).

        Returns
        -------
        Tensor
            1-D float32 tensor — preprocessed waveform.
        """
        # soundfile is faster than torchaudio for FLAC/WAV decoding
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)

        # Resample (cached transform avoids recomputing sinc kernel)
        if sr != self._target_sr:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self._target_sr,
                )
            waveform = self._resamplers[sr](waveform)

        wav = waveform.squeeze(0)                       # (N,) — 1-D

        # Silence trim — remove leading/trailing silence below threshold.
        # Applied before speed perturbation and CMVN so the model never
        # sees long silence tails that shift the mean/variance.
        # Controlled by constructor flag (training=True, validation=False
        # to match inference pipeline).
        if self._silence_trim:
            wav = self._trim_silence(wav)

        # Absolute silence floor — zero out sub-audible residual noise
        # that relative-dB trim leaves behind.  Applied after trim so
        # the floor acts on the trimmed segment, not the full recording.
        wav = self._apply_abs_floor(wav)

        # Speed perturbation (training only — controlled by constructor flag).
        # Uses torch.nn.functional.interpolate with linear mode — ~20× faster
        # than sinc resampling and perceptually identical for small tempo
        # changes (0.85–1.15×).  factor > 1 = faster speech = fewer samples.
        if self._speed_perturb:
            factor = random.uniform(self._speed_low, self._speed_high)
            if abs(factor - 1.0) > 1e-3:
                new_len = int(wav.size(0) / factor)
                wav = torch.nn.functional.interpolate(
                    wav.unsqueeze(0).unsqueeze(0),  # (1, 1, N)
                    size=new_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)  # (N,)

        # Zero-mean unit-variance normalisation
        # Matches WavLM / wav2vec2 pretraining (Wav2Vec2FeatureExtractor do_normalize=True)
        std = wav.std()
        if std > 1e-8:
            wav = (wav - wav.mean()) / std

        # Floor-pad short waveforms to minimum length (default 1 s = 16 000
        # samples).  Avoids CTC loss crash when WavLM's 320× downsample
        # produces fewer output frames than the label length.
        if wav.size(0) < self._min_samples:
            padded = torch.zeros(self._min_samples, dtype=wav.dtype)
            padded[: wav.size(0)] = wav
            wav = padded

        return wav

    # ------------------------------------------------------------------
    # MUSAN noise injection (ported from SSLCollator)
    # ------------------------------------------------------------------

    def _inject_noise(self, wav: Tensor) -> Tensor:
        """Add MUSAN noise at a stratified random SNR.

        SNR mixture:
          60 %  →  20–25 dB  (mild — barely noticeable)
          30 %  →  10–20 dB  (moderate — clearly present)
          10 %  →   5–10 dB  (harsh — robustness training)
        """
        noise_full = random.choice(self._noise_cache)
        n_wav = wav.size(0)
        n_noise = noise_full.size(0)

        if n_noise > n_wav:
            offset = random.randint(0, n_noise - n_wav)
            noise = noise_full[offset: offset + n_wav]
        elif n_noise < n_wav:
            offset = random.randint(0, n_noise - 1)
            noise = torch.roll(noise_full, -offset)
            repeats = (n_wav + n_noise - 1) // n_noise
            noise = noise.repeat(repeats)[:n_wav]
        else:
            noise = noise_full

        roll = random.random()
        if roll < 0.6:
            snr_db = random.uniform(20.0, 25.0)
        elif roll < 0.9:
            snr_db = random.uniform(10.0, 20.0)
        else:
            snr_db = random.uniform(5.0, 10.0)

        signal_rms = wav.pow(2).mean().sqrt()
        if signal_rms < 1e-6:
            return wav
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-9)
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / noise_rms)

        return wav + noise

    # ------------------------------------------------------------------
    # RIR convolution (ported from SSLCollator)
    # ------------------------------------------------------------------

    def _apply_rir(self, wav: Tensor) -> Tensor:
        """Convolve waveform with a random room impulse response.

        Output is trimmed to original length and energy-normalised.
        Uses direct torch.fft for speed (avoids torchaudio overhead).
        """
        rir = random.choice(self._rir_cache)

        original_len = wav.size(0)
        # Direct FFT convolution — faster than torchaudio.functional.fftconvolve
        # which adds shape checks and extra copies.
        fft_len = 1
        while fft_len < wav.size(0) + rir.size(0) - 1:
            fft_len <<= 1
        wav_f = torch.fft.rfft(wav, n=fft_len)
        rir_f = torch.fft.rfft(rir, n=fft_len)
        wav_rev = torch.fft.irfft(wav_f * rir_f, n=fft_len)
        wav_rev = wav_rev[:original_len].contiguous()

        rms_orig = wav.pow(2).mean().sqrt()
        rms_rev = wav_rev.pow(2).mean().sqrt()
        if rms_rev > 1e-8:
            wav_rev = wav_rev * (rms_orig / rms_rev)

        return wav_rev

    # ------------------------------------------------------------------
    # Pitch shift
    # ------------------------------------------------------------------

    def _pitch_shift(self, wav: Tensor, semitones: float | tuple[float, float] | None = None) -> Tensor:
        """Apply random pitch shift within a semitone range.

        Uses resample-based approach: resample to shift pitch, then
        interpolate back to original length to preserve duration.
        ~20-50× faster than STFT-based pitch_shift for small shifts.

        Parameters
        ----------
        wav : Tensor
            1-D float32 waveform.
        semitones : float | tuple[float, float] | None
            If float: symmetric ±N semitones.
            If (low, high) tuple: asymmetric range.
            Falls back to ``self._pitch_semitones`` when *None*.
        """
        if semitones is None:
            low, high = -self._pitch_semitones, self._pitch_semitones
        elif isinstance(semitones, tuple):
            low, high = semitones
        else:
            low, high = -semitones, semitones
        n_steps = random.uniform(low, high)
        if abs(n_steps) < 0.1:
            return wav
        # Pitch ratio: positive semitones = higher pitch = shorter period
        ratio = 2.0 ** (n_steps / 12.0)
        orig_len = wav.size(0)
        # Resample to shift pitch (changes length proportionally)
        resampled_len = int(round(orig_len / ratio))
        if resampled_len < 1:
            return wav
        # Linear interpolation — fast and sufficient for ±2 semitones
        shifted = torch.nn.functional.interpolate(
            wav.unsqueeze(0).unsqueeze(0),  # (1, 1, N)
            size=resampled_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (resampled_len,)
        # Interpolate back to original length to preserve duration
        shifted = torch.nn.functional.interpolate(
            shifted.unsqueeze(0).unsqueeze(0),
            size=orig_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (orig_len,)
        return shifted.to(wav.dtype)

    # ------------------------------------------------------------------
    # __call__ — collate a list of dataset rows into a batch
    # ------------------------------------------------------------------

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Collate a list of manifest rows into a padded batch.

        Parameters
        ----------
        batch : list[dict]
            Each dict has keys from :class:`SFTDataset.__getitem__`:
            ``audio_path``, ``phonetic_text``, ``age_bucket``, etc.

        Returns
        -------
        dict[str, Any] | None
            ``None`` if all samples were dropped (exceeded max_duration).
            Otherwise a dict with:
            ``input_values``  — ``(B, T_max)`` float32, zero-padded waveforms
            ``attention_mask`` — ``(B, T_max)`` long, 1 = real, 0 = pad
            ``labels``        — ``(B, L_max)`` long, -100 = pad
            ``input_lengths`` — ``(B,)`` long, real frame counts per sample
            ``age_buckets``   — ``list[str]``, one per sample
            ``datasets``      — ``list[str]``, dataset ID per sample
        """
        waveforms: list[Tensor] = []
        label_ids: list[list[int]] = []
        age_buckets: list[str] = []
        datasets: list[str] = []

        dropped_ids: list[str] = []

        for row in batch:
            # Audio
            wav = self._load_and_preprocess(row["audio_path"])

            # ---- Max-duration safety drop (NEVER truncate) ----
            # Secondary check: ETL should have removed these, but
            # manifest durations can disagree with actual decoded
            # waveform length.  If a sample exceeds the configured cap
            # after full preprocessing (resample + mono), drop it from
            # the batch and log a warning so it's always visible.
            if self._max_samples is not None and wav.size(0) > self._max_samples:
                actual_dur = wav.size(0) / self._target_sr
                uid = row.get("utterance_id", "?")
                dropped_ids.append(uid)
                log.warning(
                    "[COLLATOR] DROPPED — exceeds max_duration "
                    "(%.2fs > %.2fs): %s  "
                    "manifest_dur=%.2fs  actual_samples=%d",
                    actual_dur, self._max_duration_sec,
                    uid, row["audio_duration_sec"], wav.size(0),
                )
                continue

            # ---- Augmentation (dataset-gated) ----
            # noise_datasets controls noise/RIR.  E.g. [2] = DS2 only.
            # pitch_datasets controls pitch separately (SNR-safe, can differ).
            _ds_key = str(row.get("dataset", ""))
            _noise_allowed = (
                self._noise_datasets is None or _ds_key in self._noise_datasets
            )
            _pitch_allowed = (
                self._pitch_datasets is None or _ds_key in self._pitch_datasets
            )
            _augmented = False

            if _noise_allowed:
                if self._rir_cache and random.random() < self._rir_prob:
                    wav = self._apply_rir(wav)
                    _augmented = True
                if self._noise_cache and random.random() < self._noise_prob:
                    wav = self._inject_noise(wav)
                    _augmented = True

            if _pitch_allowed:
                if self._pitch_prob > 0 and random.random() < self._pitch_prob:
                    # Per-dataset semitone range if configured
                    _st = self._pitch_semitones_map.get(_ds_key) if self._pitch_semitones_map else None
                    wav = self._pitch_shift(wav, semitones=_st)
                    _augmented = True

            # Re-normalise after augmentation — noise/RIR/pitch shift the
            # waveform away from zero-mean unit-var.  WavLM expects normalised.
            if _augmented:
                std = wav.std()
                if std > 1e-8:
                    wav = (wav - wav.mean()) / std

            waveforms.append(wav)

            # Sanity: frame count vs declared duration (±0.5 s tolerance).
            # Floor-padded files will exceed declared duration — use the
            # padded minimum as the lower bound to avoid false alarms.
            # Skip when speed perturbation or silence trimming is active:
            # both legitimately change waveform length vs manifest duration.
            if not self._speed_perturb and not self._silence_trim:
                expected_frames = row["audio_duration_sec"] * self._target_sr
                effective_expected = max(expected_frames, self._min_samples)
                actual_frames = wav.size(0)
                if abs(actual_frames - effective_expected) > self._target_sr * 0.5:
                    log.warning(
                        "[COLLATOR] Frame count mismatch — %s: "
                        "expected ~%.0f frames (%.2fs @ %dHz), got %d",
                        row.get("utterance_id", "?"),
                        expected_frames, row["audio_duration_sec"],
                        self._target_sr, actual_frames,
                    )

            # Labels — tokenise phonetic text
            encoded = self._tokenizer(row["phonetic_text"]).input_ids
            label_ids.append(encoded)

            # Metadata
            age_buckets.append(row["age_bucket"])
            datasets.append(str(row["dataset"]))

        # If ALL samples in the batch were dropped, return None so the
        # training loop can skip this micro-step cleanly.
        if not waveforms:
            log.warning(
                "[COLLATOR] Entire batch dropped (%d samples exceeded "
                "max_duration %.2fs): %s",
                len(dropped_ids),
                self._max_duration_sec,
                dropped_ids,
            )
            return None

        # ------------------------------------------------------------------
        # Pad waveforms — dynamic to longest in batch (NOT fixed max length)
        # ------------------------------------------------------------------
        input_values = torch.nn.utils.rnn.pad_sequence(
            waveforms, batch_first=True, padding_value=0.0,
        )
        input_lengths = torch.tensor(
            [w.size(0) for w in waveforms], dtype=torch.long,
        )
        # Vectorised attention mask: arange < lengths (no Python loop)
        attention_mask = (
            torch.arange(input_values.size(1)).unsqueeze(0) < input_lengths.unsqueeze(1)
        ).long()

        # ------------------------------------------------------------------
        # Pad labels with -100 (CTC loss ignores these positions)
        # ------------------------------------------------------------------
        label_tensors = [torch.tensor(ids, dtype=torch.long) for ids in label_ids]
        labels = torch.nn.utils.rnn.pad_sequence(
            label_tensors, batch_first=True, padding_value=self._label_pad,
        )

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_lengths": input_lengths,
            "age_buckets": age_buckets,
            "datasets": datasets,
        }

