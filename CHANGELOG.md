# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-10-04

First public release.

### Added

- Nine detectors: `advantage_collapse`, `entropy_collapse`, `reward_hacking`,
  `length_pathology`, `clip_saturation`, `kl_drift`, `gradient_pathology`,
  `reward_composition`, `plateau`. Each emits evidence, a prescription and references.
- Ingest from W&B runs, TensorBoard directories, JSONL, JSON and CSV, plus
  `run_from_records` for any training loop that can hand over a list of dicts.
- Three-pass alias resolution (exact, suffix, token-set) across verl and TRL metric
  names, with unresolved keys surfaced rather than dropped.
- Robust statistics: Theil–Sen regression, Mann–Kendall with tie correction, Spearman,
  detrended CUSUM changepoints, and `project_threshold` model-selecting extrapolation.
- Cost model converting waste into GPU-hours and dollars, combining per-detector waste
  fractions as independent survival probabilities rather than summing them.
- Terminal (with ASCII fallback), Markdown, JSON and self-contained HTML reports.
- `LiveMonitor` and `RLDoctorCallback` for in-training diagnosis.
- Twelve simulated pathology scenarios, doubling as the test suite's detection matrix
  and as `rldoctor demo` / `rldoctor selftest`.
- Native `trainer_state.json` ingestion (every HuggingFace checkpoint ships one) and
  verl's `file` logger format, including `val-core/<source>/<metric>/mean@N` keys.
- Exactly-zero gradient steps reported as dead updates, and counted as wasted rollouts.
- `validation/`: four public GRPO runs with pinned verdicts, re-checked in CI.

### Fixed before release, found on real logs

- `nan` padding from eval rows logged on a different cadence was counted as non-finite
  gradient norms -- a false "numerically dead" verdict on a healthy run.
- Statistical outliers only 2x the median were reported as gradient spikes.
- Truncation was counted as wasted rollout compute, producing an indefensible "99%".
- TRL's `eval_` prefix was not recognised, so held-out metrics leaked into training series.
- A KL of exactly 0.0 at every step (a run with no KL term) was read as a pinned policy.
- A plateau was declared on a run less than half way to `max_steps`.
