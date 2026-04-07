# Project Documentation

> Code-level documentation for each phase of the 309 Children's Speech Recognition pipeline.
> Each file documents the *why* behind the code — design decisions, trade-offs, and rationale.

---

| File | Phase | What It Covers |
|------|-------|----------------|
| [1_audio_check.md](1_audio_check.md) | ETL — Stage 0 | Audio file integrity (existence, filesize, MD5, orphan detection) |
| [2_text_normalisation.md](2_text_normalisation.md) | ETL — Stage 1 | Text normalisation pipeline (NFC, whitespace, invalid character removal) |
| [3_audio_eda.md](3_audio_eda.md) | ETL — Stage 2 | Audio EDA control calibration (RMS, peak, clipping, SNR, duration) |
| [4_tokenizer.md](4_tokenizer.md) | ETL — Stage 3 | CTC tokenizer vocab construction (50 IPA phonemes + 3 specials) |
| [5_data_split.md](5_data_split.md) | ETL — Stage 4 | Deterministic speaker-disjoint train/val split |
| [6_sft_trainer.md](6_sft_trainer.md) | Training | WavLM CTC fine-tuning via HuggingFace Trainer — full trainer architecture |
| [7_submission.md](7_submission.md) | Inference | Beam search + KenLM submission pipeline, build scripts, decode modes |
| [8_discarded_approaches.md](8_discarded_approaches.md) | Reference | SSL pre-training and 3-stage SFT — what we tried and why we moved on |
