"""Zero-variance GRPO groups -- the most expensive silent failure in RLVR.

GRPO centres rewards inside each group of ``G`` rollouts sampled from the same
prompt.  When every rollout in a group earns the same reward, the centred
advantage is exactly zero for all ``G`` of them, so the policy-gradient
contribution of that whole group is exactly zero.  You paid for ``G``
generations and bought no gradient.

Nothing about this is visible in the loss curve.  Loss looks fine, reward looks
fine, and 70% of your rollout budget goes to /dev/null.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import stats
from ..schema import (
    GROUP_SIZE,
    REWARD_MAX,
    REWARD_MEAN,
    REWARD_MIN,
    ZERO_VAR_GROUP_FRAC,
    Run,
)
from .base import Detector, Finding, Severity

# Thresholds are stated as "how many rollouts you pay for per effective
# rollout", which is the number a practitioner actually cares about:
#   f = 0.50 -> 2.0x cost per effective sample
#   f = 0.75 -> 4.0x
#   f = 0.90 -> 10.0x
_INFO = 0.30
_WARN = 0.50
_CRIT = 0.75


class AdvantageCollapse(Detector):
    name = "advantage_collapse"
    title = "Zero-variance groups (wasted rollouts)"
    requires: list = []
    optional = [ZERO_VAR_GROUP_FRAC, REWARD_MEAN, GROUP_SIZE]
    algorithms = ["grpo", "rloo", "reinforce", "dr_grpo", "dapo", "gspo"]
    references = [
        "Yu et al., DAPO: An Open-Source LLM Reinforcement Learning System at Scale (arXiv:2503.14476) -- dynamic sampling",
        "Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation (arXiv:2605.21125)",
        "TRL logs this directly as `frac_reward_zero_std`",
    ]

    def run(self, run: Run) -> Finding:
        measured = run.has(ZERO_VAR_GROUP_FRAC)
        if measured:
            steps, frac = run.finite(ZERO_VAR_GROUP_FRAC)
            basis = "measured"
        else:
            estimate = _estimate_zero_var_frac(run)
            if estimate is None:
                return self.skip(
                    "needs `zero_var_group_frac` (TRL: `frac_reward_zero_std`) or a "
                    "bounded reward series to estimate it from",
                    missing_fields=[ZERO_VAR_GROUP_FRAC],
                )
            steps, frac = estimate
            basis = "estimated"

        if frac.size == 0:
            return self.skip("no finite observations of the group-variance series")

        recent = stats.tail(frac, frac=0.25)
        f = float(np.median(recent))
        tr = stats.trend(steps, frac)

        multiplier = 1.0 / max(1.0 - f, 1e-6)
        severity = (
            Severity.CRITICAL
            if f >= _CRIT
            else Severity.WARNING
            if f >= _WARN
            else Severity.INFO
            if f >= _INFO
            else Severity.OK
        )

        regime, regime_note = _classify_regime(run)

        evidence = [
            f"{basis} zero-variance group fraction over the last quarter of the run: "
            f"{f:.1%} (median of {recent.size} steps)",
            f"effective sample efficiency: {1 - f:.1%} -- you pay for "
            f"{multiplier:.1f} rollouts per rollout that produces gradient",
        ]
        if basis == "estimated":
            evidence.append(
                "estimate assumes binary reward and independent rollouts: "
                "P(all-same) = p^G + (1-p)^G. Log `frac_reward_zero_std` for the exact value."
            )
        if tr.significant:
            direction = "rising" if tr.slope > 0 else "falling"
            evidence.append(
                f"trend is {direction} ({tr.slope:+.2e}/step, Mann-Kendall p{stats.fmt_p(tr.p_value)}), "
                f"so the waste is {'getting worse' if tr.slope > 0 else 'self-correcting'}"
            )
        if regime_note:
            evidence.append(regime_note)

        prescription = _prescribe(regime, f, run)

        if severity is Severity.OK:
            return self.finding(
                Severity.OK,
                f"{f:.1%} of groups are degenerate -- normal for a healthy GRPO run.",
                evidence=evidence,
                metrics={"zero_var_frac": f, "effective_efficiency": 1 - f, "basis": basis},
                wasted_fraction=f,
            )

        headline = {
            "saturated": "The task is too easy: most groups are all-correct and contribute no gradient.",
            "too_hard": "The task is too hard: most groups are all-wrong and contribute no gradient.",
            "mixed": "Most groups are degenerate and contribute no gradient.",
        }[regime]

        return self.finding(
            severity,
            f"{headline} {f:.0%} of your rollout budget produces exactly zero policy gradient.",
            evidence=evidence,
            prescription=prescription,
            metrics={
                "zero_var_frac": f,
                "effective_efficiency": 1 - f,
                "cost_multiplier": multiplier,
                "regime": regime,
                "trend_slope": tr.slope,
                "trend_p": tr.p_value,
                "basis": basis,
            },
            wasted_fraction=f,
        )


def _estimate_zero_var_frac(run: Run) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Estimate the degenerate-group rate from a bounded reward series.

    Under a binomial model with per-rollout success probability ``p`` and group
    size ``G``, a group is degenerate with probability ``p^G + (1-p)^G``.  This
    is a lower bound in practice, because rollouts from one prompt are
    positively correlated -- so if the estimate already looks bad, reality is
    worse.
    """
    if not run.has(REWARD_MEAN):
        return None
    group_size = run.config.group_size
    if group_size is None and run.has(GROUP_SIZE, min_points=1):
        group_size = int(stats.nanmedian(run.series[GROUP_SIZE], default=float("nan")))
    if not group_size or group_size < 2 or not np.isfinite(group_size):
        return None

    steps, reward = run.finite(REWARD_MEAN)
    lo, hi = _reward_bounds(run, reward)
    if hi <= lo:
        return None
    p = np.clip((reward - lo) / (hi - lo), 0.0, 1.0)
    frac = p**group_size + (1.0 - p) ** group_size
    return steps, frac


def _reward_bounds(run: Run, reward: np.ndarray) -> tuple[float, float]:
    """Best available guess at the reward range, for normalising to a pass rate."""
    if run.has(REWARD_MIN) and run.has(REWARD_MAX):
        return (
            float(np.nanmin(run.series[REWARD_MIN])),
            float(np.nanmax(run.series[REWARD_MAX])),
        )
    lo, hi = float(np.nanmin(reward)), float(np.nanmax(reward))
    # A reward that already lives in [0, 1] is almost always a pass rate.
    if lo >= 0.0 and hi <= 1.0:
        return 0.0, 1.0
    return lo, hi


def _classify_regime(run: Run) -> tuple[str, str]:
    """Distinguish 'all groups correct' from 'all groups wrong'.

    Same symptom, opposite fix -- which is why reporting the bare fraction is
    not enough to act on.
    """
    if not run.has(REWARD_MEAN):
        return "mixed", ""
    _, reward = run.finite(REWARD_MEAN)
    recent = stats.tail(reward, frac=0.25)
    lo, hi = _reward_bounds(run, reward)
    if hi <= lo:
        return "mixed", ""
    p = float((np.median(recent) - lo) / (hi - lo))
    if p >= 0.85:
        return "saturated", f"mean pass rate is {p:.0%} -- the model has outgrown this data"
    if p <= 0.15:
        return "too_hard", f"mean pass rate is {p:.0%} -- the model almost never succeeds"
    return "mixed", f"mean pass rate is {p:.0%}, so the degeneracy is not explained by difficulty alone"


def _prescribe(regime: str, f: float, run: Run) -> list:
    group_size = run.config.group_size
    shared = [
        "Enable dynamic sampling (DAPO): keep resampling prompts until each group has "
        "non-zero reward variance. verl: `algorithm.filter_groups.enable=True`; "
        "TRL: see the dynamic-sampling issue tracker for the current flag.",
    ]
    if regime == "saturated":
        return [
            "Raise task difficulty -- filter out prompts the current policy already solves "
            "with pass rate > 0.9 and refresh the training pool.",
            "Adopt a curriculum keyed on measured pass rate; target the 0.3-0.7 band where "
            "group variance, and therefore gradient, is maximal.",
            *shared,
        ]
    if regime == "too_hard":
        prescription = [
            "Lower task difficulty or add an intermediate curriculum stage -- the policy "
            "cannot bootstrap from an all-zero reward signal.",
            "Check the verifier before blaming the model: an over-strict or crashing grader "
            "produces the exact same all-wrong signature. Manually grade 20 rollouts.",
        ]
        if group_size and group_size < 16:
            prescription.append(
                f"Raise group size from {group_size} to at least 16 so rare successes can "
                "appear inside a group at all."
            )
        return prescription + shared
    prescription = list(shared)
    if group_size and group_size <= 8:
        prescription.insert(
            0,
            f"Increase group size (currently {group_size}). Degeneracy probability falls "
            "roughly geometrically in G, so 8 -> 16 typically halves the waste.",
        )
    prescription.append(
        "If you would rather not pay for resampling, keep the collapsed groups but change "
        "the normalisation (e.g. Dr. GRPO / AVSPO-style variance handling) so they still "
        "contribute signal."
    )
    return prescription
