# Contributing to rldoctor

## The most useful thing you can do

**Tell us about a metric key we do not recognise.** Run:

```bash
rldoctor diagnose <your run> --verbose
```

Anything listed under "unrecognised log keys" is a metric we dropped. If one of them is
something rldoctor should understand, open an issue with the key name, the framework and
version, and what it measures. That is usually a one-line addition to
`src/rldoctor/aliases.py` and it makes the tool work for everyone on that stack.

**Second most useful: a run that produced a wrong verdict.** A false positive is a bug of
the highest severity in this project — a diagnostic that cries wolf gets uninstalled, and
then it catches nothing. If rldoctor warned about a run that was fine, or stayed quiet on
a run that was not, please open an issue with the metric log (a CSV of the relevant series
is plenty) and what actually happened.

## Adding a detector

Three files:

1. **`src/rldoctor/detectors/your_check.py`** — subclass `Detector`, set `name`, `title`,
   `requires`, `references`, and implement `run(self, run) -> Finding`. The base class
   validates `requires` before calling you, so the body can assume its inputs exist.
2. **`src/rldoctor/detectors/__init__.py`** — add the class to `BUILTIN_DETECTORS`.
3. **`src/rldoctor/simulate.py`** — add a scenario that reproduces the failure, and record
   the expected detection in `EXPECTED_DETECTIONS`.

That third step is not optional busywork. It is what makes the check trustworthy: it
proves the detector fires on the failure it claims, and the shared `healthy` scenario
proves it stays quiet otherwise. `pytest tests/test_detectors.py` runs the whole matrix.

### What a good detector does

- **Returns `SKIPPED` with a named missing field** rather than guessing from absent data.
- **Puts the numbers it fired on into `evidence`.** A reader must be able to disagree with
  your threshold rather than with the tool.
- **Always prescribes.** A finding with no suggested change is a complaint. This is
  enforced by `test_findings_are_actionable`.
- **Cites its source.** Thresholds pulled from a paper are defensible; thresholds pulled
  from nowhere are not.
- **Checks a second signal before accusing.** Entropy decay means nothing without the
  reward trend; length growth means nothing without a held-out score. Conditional firing
  is the single biggest lever on the false-positive rate.
- **Uses the robust helpers in `stats.py`.** Curves are noisy and heavy-tailed; means and
  OLS slopes over-react to the handful of catastrophic steps every real run contains.

## Changing a threshold

Run the real-run validation before and after:

```bash
python validation/fetch.py
python validation/run.py --check
```

The simulated scenarios tell you a detector *can* fire. The five public runs in
`validation/runs.json` tell you whether it fires on runs that exist. If a verdict moves,
either the new verdict is right -- update `runs.json` and say why in the commit -- or the
change broke something real. Adding a public run you know the ground truth for is one of
the most useful contributions there is.

## Development

```bash
pip install -e ".[dev]"
pytest
rldoctor selftest
ruff check src tests
```

Please keep the core dependency-free beyond `numpy`. Anything heavier goes behind an
optional extra with a lazy import, because this tool has to install cleanly next to
whatever a user's cluster image already pins.

## Code style

Match the surrounding code. Comments explain *why* a threshold or an approach was chosen,
not what the line does — the constants in this codebase are opinions, and an opinion
without a reason cannot be argued with or improved.
