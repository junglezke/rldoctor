"""Translate findings into GPU-hours and dollars.

A warning that says "47% of your groups are degenerate" gets nodded at and
ignored. The same warning that says "that is 91 H100-hours, about $270 on this
run alone" gets acted on. Same number, different units.

Prices are on-demand list rates and are deliberately conservative -- reserved
capacity and spot are cheaper, so the figures here are an *upper* bound on the
per-hour rate and a *lower* bound on the argument for fixing the problem.
Override with ``--gpu-hour-cost`` when you know your real rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import stats
from .detectors.base import Finding, Severity
from .schema import TIME_PER_STEP, Run

#: Indicative on-demand USD/GPU-hour. Update via ``--gpu-hour-cost``.
GPU_HOURLY_USD: Dict[str, float] = {
    "h200": 3.90,
    "h100": 2.99,
    "a100": 1.79,
    "a100-80g": 1.99,
    "l40s": 1.10,
    "a10g": 0.75,
    "v100": 0.55,
    "mi300x": 2.50,
}
DEFAULT_GPU_HOURLY_USD = 2.99

#: Share of a GRPO step's wall-clock spent generating rollouts. Degenerate
#: groups waste generation, not the optimiser pass, so the run-level waste is
#: scaled by this rather than applied to total GPU-hours. Measured splits for
#: 7B-class models with long generations typically land in 0.6-0.85; 0.7 is a
#: deliberately conservative default. Override with ``rollout_share``.
DEFAULT_ROLLOUT_SHARE = 0.70


@dataclass
class CostModel:
    """What the run cost, and how much of it was wasted."""

    total_gpu_hours: Optional[float]
    wasted_gpu_hours: Optional[float]
    wasted_fraction: float
    usd_per_gpu_hour: float
    contributions: Dict[str, float]
    rollout_share: float = DEFAULT_ROLLOUT_SHARE

    @property
    def total_usd(self) -> Optional[float]:
        return None if self.total_gpu_hours is None else self.total_gpu_hours * self.usd_per_gpu_hour

    @property
    def wasted_usd(self) -> Optional[float]:
        return None if self.wasted_gpu_hours is None else self.wasted_gpu_hours * self.usd_per_gpu_hour

    def headline(self) -> Optional[str]:
        if self.wasted_gpu_hours is None or self.wasted_fraction <= 0.01:
            return None
        parts = [
            f"~{self.wasted_fraction:.0%} of this run's rollout compute produced no learning signal"
        ]
        parts.append(f"{self.wasted_gpu_hours:,.0f} GPU-hours")
        usd = self.wasted_usd
        if usd is not None:
            parts.append(f"about ${usd:,.0f} at ${self.usd_per_gpu_hour:.2f}/GPU-hour")
        return " = ".join(parts)


def gpu_hourly_cost(gpu_type: Optional[str], override: Optional[float] = None) -> float:
    if override is not None:
        return float(override)
    if gpu_type:
        return GPU_HOURLY_USD.get(gpu_type.strip().lower(), DEFAULT_GPU_HOURLY_USD)
    return DEFAULT_GPU_HOURLY_USD


def estimate(
    run: Run,
    findings: List[Finding],
    usd_per_gpu_hour: Optional[float] = None,
    num_gpus: Optional[int] = None,
    rollout_share: float = DEFAULT_ROLLOUT_SHARE,
) -> CostModel:
    """Combine per-finding waste fractions into a single run-level estimate.

    Waste fractions are combined as independent survival probabilities rather
    than summed. Summing them double-counts -- a rollout that is both truncated
    and in a degenerate group is one wasted rollout, not two -- and can exceed
    100%, which destroys the credibility of the whole number.
    """
    contributions = {
        f.detector: float(f.wasted_fraction)
        for f in findings
        if f.wasted_fraction is not None and f.wasted_fraction > 0 and f.severity >= Severity.INFO
    }

    survival = 1.0
    for value in contributions.values():
        survival *= 1.0 - min(max(value, 0.0), 0.99)
    wasted_fraction = 1.0 - survival

    total = total_gpu_hours(run, num_gpus=num_gpus)
    rate = gpu_hourly_cost(run.config.gpu_type, usd_per_gpu_hour)
    return CostModel(
        total_gpu_hours=total,
        # Scale by the generation share: a degenerate group wastes the rollouts
        # it produced, not the backward pass that ignored them.
        wasted_gpu_hours=None if total is None else total * wasted_fraction * rollout_share,
        wasted_fraction=wasted_fraction,
        usd_per_gpu_hour=rate,
        contributions=contributions,
        rollout_share=rollout_share,
    )


def total_gpu_hours(run: Run, num_gpus: Optional[int] = None) -> Optional[float]:
    """Wall-clock GPU-hours consumed, if the log carries enough to say."""
    gpus = num_gpus if num_gpus is not None else run.config.num_gpus
    if not run.has(TIME_PER_STEP):
        return None
    _, seconds = run.finite(TIME_PER_STEP)
    if seconds.size == 0:
        return None
    # Median x n_steps rather than a raw sum: logs routinely contain a handful
    # of enormous step times from checkpointing or preemption.
    median = stats.nanmedian(seconds)
    if not np.isfinite(median) or median <= 0:
        return None
    hours = median * run.n_steps / 3600.0
    return hours * gpus if gpus else hours
