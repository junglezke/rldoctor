# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] — unreleased

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
