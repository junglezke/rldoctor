"""Live monitoring: catch the failure at step 200, not in the post-mortem.

A post-hoc report tells you that you wasted 40 GPU-hours. A live monitor stops
you from wasting them. :class:`LiveMonitor` takes the metric dict your training
loop already builds, and every ``check_every`` steps re-runs the detectors over
the history so far.

    from rldoctor.live import LiveMonitor

    monitor = LiveMonitor(config=cfg, check_every=25)
    for step, metrics in training_loop():
        monitor.log(metrics, step=step)   # returns new findings, or []

Deliberate design choices:

* **Never raises into your training loop.** A monitoring bug must not kill a
  job that is otherwise fine, so :meth:`log` swallows its own exceptions and
  reports them once.
* **Only speaks when something changes.** Re-reporting the same warning every
  25 steps trains people to filter it out. A finding is announced when it
  appears or when it gets worse.
* **Warm-up.** Trend statistics on 20 steps are noise. Nothing fires before
  ``min_steps``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .detectors.base import Finding, Severity
from .diagnosis import Diagnosis, diagnose
from .ingest.base import run_from_records

_LOG = logging.getLogger("rldoctor")

#: Below this many steps, trend tests have no power and would only produce noise.
DEFAULT_MIN_STEPS = 60


class LiveMonitor:
    """Incremental diagnosis over a run that is still in progress."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        check_every: int = 25,
        min_steps: int = DEFAULT_MIN_STEPS,
        alert_at: Severity = Severity.WARNING,
        on_finding: Optional[Callable[[Finding], None]] = None,
        only: Sequence[str] = (),
        skip: Sequence[str] = (),
        name: str = "live",
    ) -> None:
        self.config = dict(config) if config else None
        self.check_every = max(int(check_every), 1)
        self.min_steps = max(int(min_steps), 8)
        self.alert_at = alert_at
        self.on_finding = on_finding or _default_alert
        self.only = tuple(only)
        self.skip = tuple(skip)
        self.name = name

        self.records: List[Dict[str, Any]] = []
        self.latest: Optional[Diagnosis] = None
        self._announced: Dict[str, Severity] = {}
        self._errored = False

    # -- ingestion ---------------------------------------------------------

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> List[Finding]:
        """Record one step's metrics and, on a check boundary, re-diagnose.

        Returns findings that are new or newly worse. Safe to call every step.
        """
        record = dict(metrics)
        if step is not None:
            record.setdefault("step", step)
        self.records.append(record)

        if len(self.records) < self.min_steps:
            return []
        if len(self.records) % self.check_every != 0:
            return []
        return self.check()

    def check(self) -> List[Finding]:
        """Force a diagnosis now, regardless of the check interval."""
        try:
            run = run_from_records(self.records, config=self.config, name=self.name, source="live")
            self.latest = diagnose(run, only=self.only, skip=self.skip)
        except Exception as exc:  # pragma: no cover - defensive
            if not self._errored:
                self._errored = True
                _LOG.warning("rldoctor live monitoring disabled after an internal error: %s", exc)
            return []

        fresh = []
        for finding in self.latest.problems:
            if finding.severity < self.alert_at:
                continue
            previous = self._announced.get(finding.detector)
            if previous is not None and finding.severity <= previous:
                continue  # already said this, and it has not got worse
            self._announced[finding.detector] = finding.severity
            fresh.append(finding)

        for finding in fresh:
            try:
                self.on_finding(finding)
            except Exception:  # pragma: no cover - a bad callback is the user's
                _LOG.exception("rldoctor on_finding callback raised")
        return fresh

    # -- output ------------------------------------------------------------

    def report(self, fmt: str = "terminal") -> str:
        """Render the most recent diagnosis. Runs one if none exists yet."""
        from .report import render

        if self.latest is None:
            self.check()
        if self.latest is None:
            return "rldoctor: not enough steps logged to diagnose yet."
        return render(self.latest, fmt)

    def save(self, path: str, fmt: str = "html") -> str:
        text = self.report(fmt)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    @property
    def worst(self) -> Severity:
        return self.latest.worst if self.latest else Severity.SKIPPED


def _default_alert(finding: Finding) -> None:
    """Log a finding at a level matching its severity."""
    level = logging.ERROR if finding.severity >= Severity.CRITICAL else logging.WARNING
    lines = [f"[{finding.severity.label}] {finding.title}: {finding.summary}"]
    if finding.prescription:
        lines.append(f"  -> {finding.prescription[0]}")
    _LOG.log(level, "\n".join(lines))
