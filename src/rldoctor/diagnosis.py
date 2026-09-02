"""The top-level ``diagnose`` entry point and its result object."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import detectors as _detectors
from . import economics
from .detectors.base import Finding, Severity
from .schema import Run

__all__ = ["Diagnosis", "diagnose"]


@dataclass
class Diagnosis:
    """Everything ``rldoctor`` concluded about one run."""

    run: Run
    findings: List[Finding]
    cost: economics.CostModel
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    version: str = "0.1.0"

    # -- views -------------------------------------------------------------

    @property
    def worst(self) -> Severity:
        actionable = [f.severity for f in self.findings if f.severity > Severity.SKIPPED]
        return max(actionable) if actionable else Severity.SKIPPED

    @property
    def problems(self) -> List[Finding]:
        """Findings worth a human's attention, worst first."""
        return sorted(
            [f for f in self.findings if f.severity >= Severity.INFO],
            key=lambda f: (-int(f.severity), f.detector),
        )

    @property
    def passed(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.OK]

    @property
    def skipped(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.SKIPPED]

    def by_name(self, detector: str) -> Optional[Finding]:
        for finding in self.findings:
            if finding.detector == detector:
                return finding
        return None

    @property
    def headline(self) -> str:
        """One sentence for a Slack message or a CI summary line."""
        problems = self.problems
        if not problems:
            checked = len(self.passed)
            return f"No problems found across {checked} checks on {self.run.name}."
        worst = problems[0]
        extra = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
        return f"[{worst.severity.label}] {worst.title}: {worst.summary}{extra}"

    def exit_code(self, fail_on: Severity = Severity.CRITICAL) -> int:
        """0 when nothing at or above ``fail_on`` fired -- for CI gating."""
        return 1 if self.worst >= fail_on else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "run": {
                "name": self.run.name,
                "source": self.run.source,
                "n_steps": self.run.n_steps,
                "fields": self.run.available_fields,
                "unmapped_keys": self.run.unmapped_keys,
                "config": {
                    k: v
                    for k, v in vars(self.run.config).items()
                    if k != "raw" and v is not None
                },
            },
            "worst_severity": self.worst.name,
            "headline": self.headline,
            "cost": {
                "total_gpu_hours": self.cost.total_gpu_hours,
                "wasted_gpu_hours": self.cost.wasted_gpu_hours,
                "wasted_fraction": self.cost.wasted_fraction,
                "usd_per_gpu_hour": self.cost.usd_per_gpu_hour,
                "wasted_usd": self.cost.wasted_usd,
                "contributions": self.cost.contributions,
            },
            "findings": [f.to_dict() for f in self.findings],
        }


def diagnose(
    run: Run,
    only: Sequence[str] = (),
    skip: Sequence[str] = (),
    usd_per_gpu_hour: Optional[float] = None,
    num_gpus: Optional[int] = None,
    extra_detectors: Iterable[_detectors.Detector] = (),
) -> Diagnosis:
    """Run every applicable detector over ``run`` and price the damage.

    A detector that raises is reported as a skipped check rather than being
    allowed to take down the whole report: one broken check should never cost
    you the other eight.
    """
    checks = list(_detectors.build(only=only, skip=skip)) + list(extra_detectors)
    findings: List[Finding] = []
    for detector in checks:
        try:
            findings.append(detector.check(run))
        except Exception as exc:  # pragma: no cover - defensive
            findings.append(
                Finding(
                    detector=detector.name,
                    title=detector.title,
                    severity=Severity.SKIPPED,
                    summary=f"detector raised {type(exc).__name__}: {exc}",
                )
            )

    cost = economics.estimate(
        run, findings, usd_per_gpu_hour=usd_per_gpu_hour, num_gpus=num_gpus
    )
    return Diagnosis(run=run, findings=findings, cost=cost)
