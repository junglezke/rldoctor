"""Detector contract: every check is a small, independently testable unit.

A detector answers one question about one run and returns a :class:`Finding`.
Three rules keep the report trustworthy:

1. **Never guess from missing data.** If the required series is absent, return
   ``Severity.SKIPPED`` and say which key would have enabled the check.
2. **Always show the arithmetic.** Every finding carries the numbers it fired
   on, so a user can disagree with the threshold rather than the tool.
3. **Always prescribe.** A diagnosis with no suggested change is a complaint.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

from ..schema import Run


class Severity(enum.IntEnum):
    """Ordered so that ``max()`` over findings gives the run's worst state."""

    SKIPPED = 0
    OK = 1
    INFO = 2
    WARNING = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return {
            Severity.SKIPPED: "SKIP",
            Severity.OK: "OK",
            Severity.INFO: "INFO",
            Severity.WARNING: "WARN",
            Severity.CRITICAL: "CRIT",
        }[self]


@dataclass
class Finding:
    """One detector's verdict on one run."""

    detector: str
    title: str
    severity: Severity
    summary: str
    evidence: List[str] = field(default_factory=list)
    prescription: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    #: Rollout compute the finding accounts for, as a fraction in [0, 1].
    #: Aggregated by the cost model into GPU-hours and dollars.
    wasted_fraction: Optional[float] = None
    missing_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "title": self.title,
            "severity": self.severity.name,
            "summary": self.summary,
            "evidence": self.evidence,
            "prescription": self.prescription,
            "references": self.references,
            "metrics": self.metrics,
            "wasted_fraction": self.wasted_fraction,
            "missing_fields": self.missing_fields,
        }


class Detector:
    """Base class for all checks.

    Subclasses set the class attributes and implement :meth:`run`.
    ``requires`` is checked before :meth:`run` is called, so detector bodies can
    assume their inputs exist.
    """

    name: ClassVar[str] = "detector"
    title: ClassVar[str] = "Detector"
    #: Canonical fields without which the check cannot run at all.
    requires: ClassVar[List[str]] = []
    #: Fields that sharpen the diagnosis but are not required.
    optional: ClassVar[List[str]] = []
    references: ClassVar[List[str]] = []
    #: Algorithms this check applies to; empty means "any".
    algorithms: ClassVar[List[str]] = []

    def applicable(self, run: Run) -> bool:
        if self.algorithms and run.config.algorithm:
            return run.config.algorithm.lower() in self.algorithms
        return True

    def check(self, run: Run) -> Finding:
        """Validate inputs, then delegate to :meth:`run`."""
        if not self.applicable(run):
            return self.skip(f"not applicable to algorithm={run.config.algorithm}")
        missing = [f for f in self.requires if not run.has(f)]
        if missing:
            return self.skip(
                "needs " + ", ".join(f"`{m}`" for m in missing) + " which this log does not contain",
                missing_fields=missing,
            )
        return self.run(run)

    def run(self, run: Run) -> Finding:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers for subclasses -------------------------------------------

    def finding(
        self,
        severity: Severity,
        summary: str,
        evidence: Optional[List[str]] = None,
        prescription: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        wasted_fraction: Optional[float] = None,
    ) -> Finding:
        return Finding(
            detector=self.name,
            title=self.title,
            severity=severity,
            summary=summary,
            evidence=evidence or [],
            prescription=prescription or [],
            references=list(self.references),
            metrics=metrics or {},
            wasted_fraction=wasted_fraction,
        )

    def skip(self, reason: str, missing_fields: Optional[List[str]] = None) -> Finding:
        return Finding(
            detector=self.name,
            title=self.title,
            severity=Severity.SKIPPED,
            summary=reason,
            references=list(self.references),
            missing_fields=missing_fields or [],
        )

    def ok(self, summary: str, metrics: Optional[Dict[str, Any]] = None) -> Finding:
        return self.finding(Severity.OK, summary, metrics=metrics)
