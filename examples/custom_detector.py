"""Add a project-specific check without forking rldoctor.

Run: python examples/custom_detector.py
"""

import numpy as np

from rldoctor import Severity, diagnose, register, stats
from rldoctor.detectors.base import Detector
from rldoctor.schema import THROUGHPUT, Run
from rldoctor.simulate import simulate_run


@register
class ThroughputRegression(Detector):
    """Flag a cluster that is quietly getting slower.

    A worked example of the detector contract: declare what you need, and the
    base class guarantees it exists before `run` is called.
    """

    name = "throughput_regression"
    title = "Throughput regression"
    requires = [THROUGHPUT]

    def run(self, run: Run):
        steps, tokens_per_s = run.finite(THROUGHPUT)
        start = float(np.median(stats.head(tokens_per_s)))
        now = float(np.median(stats.tail(tokens_per_s, frac=0.1)))
        drop = 1.0 - now / start if start > 0 else 0.0
        trend = stats.trend(steps, tokens_per_s)

        if drop < 0.15:
            return self.ok(f"Throughput is stable ({start:,.0f} -> {now:,.0f} tok/s).")
        return self.finding(
            Severity.WARNING if drop < 0.35 else Severity.CRITICAL,
            f"Throughput has fallen {drop:.0%} since the start of the run.",
            evidence=[
                f"{start:,.0f} -> {now:,.0f} tokens/s "
                f"(trend {trend.slope:+.1f}/step, p{stats.fmt_p(trend.p_value)})"
            ],
            prescription=[
                "Check for a straggler rank: a single slow node throttles the whole "
                "collective, and the symptom is a slow, monotone throughput decline.",
                "Rule out growing response length before blaming the hardware -- longer "
                "generations lower tokens/s per step for entirely benign reasons.",
            ],
        )


result = diagnose(simulate_run("healthy", n_steps=300, seed=0))
finding = result.by_name("throughput_regression")
print(f"[{finding.severity.label}] {finding.title}: {finding.summary}")
print("\nregistered detectors:")
from rldoctor.detectors import available  # noqa: E402

for name in available():
    print(" ", name)
