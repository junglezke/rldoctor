"""Reward-component imbalance: which term is actually driving the policy?

Most RLVR recipes sum several reward functions -- a correctness reward, a format
reward, maybe a length or language-consistency term. The policy does not
optimise the sum; it optimises whichever component has *variance it can move*.
A format reward that is easy to saturate can dominate the gradient early and
then contribute nothing, while a correctness reward that is almost always zero
contributes nothing at any point.

The useful quantity is each component's share of total reward variance, not its
mean. A component with a large mean and no variance is a constant, and a
constant is invisible to a policy gradient.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .. import stats
from ..schema import Run
from .base import Detector, Finding, Severity

_DOMINANCE_WARN = 0.70
_DOMINANCE_CRIT = 0.90
_DEAD_SHARE = 0.02
#: A component whose recent variance share has fallen well below its early share
#: is saturating rather than taking over. Reported as context, not as a gate --
#: measured across scenarios the decline ratios of a benign transient and a real
#: takeover overlap too much to separate them reliably.
_SATURATION_DECAY = 0.85

#: Minimum observations before this check will say anything above INFO.
#:
#: Variance shares during warm-up are dominated by whichever term is still
#: ramping, and a format reward saturates within the first few dozen steps in
#: every recipe that uses one. With 150 points the recent half is entirely
#: post-warm-up and the shares mean what they claim to mean; below that they do
#: not, and guessing would produce exactly the kind of early-run false alarm
#: that gets a monitor switched off.
_MIN_OBS_FOR_VERDICT = 150

#: Substrings that mark a component as auxiliary rather than the actual task.
_AUX_HINTS = ("format", "tag", "xml", "json", "length", "lang", "style", "think")


class RewardComposition(Detector):
    name = "reward_composition"
    title = "Reward component imbalance"
    requires: list = []
    references = [
        "Guo et al., DeepSeek-R1 (arXiv:2501.12948) -- accuracy plus format reward composition",
    ]

    def run(self, run: Run) -> Finding:
        components = {k: v for k, v in run.reward_components.items() if np.isfinite(v).sum() >= 4}
        if len(components) < 2:
            return self.skip(
                "needs at least two per-function reward series "
                "(TRL logs these as `rewards/<func_name>/mean`)"
            )

        # Judge on the recent window: the question is which term is driving the
        # policy *now*, not which one dominated during warm-up.
        recent = {k: stats.tail(v[np.isfinite(v)], frac=0.5) for k, v in components.items()}
        early = {k: stats.head(v[np.isfinite(v)], frac=0.5, maximum=10**9) for k, v in components.items()}
        shares = _variance_shares(recent)
        early_shares = _variance_shares(early)
        ordered = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        top_name, top_share = ordered[0]
        dead = [name for name, share in ordered if share < _DEAD_SHARE]

        # A component whose share is collapsing is saturating, not dominating.
        # Format rewards do this in every healthy run: they carry most of the
        # variance for the first few dozen steps and then become a constant.
        saturating = top_share < early_shares.get(top_name, 1.0) * _SATURATION_DECAY

        evidence = [
            "share of total reward variance by component: "
            + ", ".join(f"{name} {share:.0%}" for name, share in ordered)
        ]
        for name, series in components.items():
            steps, values = run.steps[np.isfinite(series)], series[np.isfinite(series)]
            tr = stats.trend(steps, values)
            evidence.append(
                f"  {name}: mean {np.median(values):.3g}, spread {stats.mad(values):.3g}, "
                f"trend {tr.slope:+.2e}/step ({tr.direction})"
            )

        metrics: Dict[str, object] = {
            "variance_shares": shares,
            "early_variance_shares": early_shares,
            "dominant": top_name,
            "saturating": saturating,
        }
        severity = Severity.OK
        summaries = []
        prescription = []

        is_aux = any(hint in top_name.lower() for hint in _AUX_HINTS)
        if saturating:
            evidence.append(
                f"'{top_name}' carried {early_shares.get(top_name, 0):.0%} of the variance early on "
                f"and {top_share:.0%} recently -- it is saturating, which is expected behaviour "
                "for an auxiliary term, not a takeover"
            )
        if top_share >= _DOMINANCE_CRIT and is_aux:
            severity = Severity.CRITICAL
            summaries.append(
                f"'{top_name}' accounts for {top_share:.0%} of all reward variance. Your policy is "
                "being trained almost entirely on an auxiliary term, not on the task."
            )
        elif top_share >= _DOMINANCE_WARN and is_aux:
            severity = Severity.WARNING
            summaries.append(
                f"An auxiliary term ('{top_name}') carries {top_share:.0%} of the reward variance."
            )
        elif top_share >= _DOMINANCE_CRIT:
            severity = Severity.INFO
            summaries.append(
                f"'{top_name}' carries {top_share:.0%} of the reward variance; the other components "
                "are effectively decorative."
            )

        if dead:
            # A format reward that every rollout now earns is a saturated
            # auxiliary term: worth knowing, not worth alarming about. A *task*
            # reward with no variance is a different matter -- that is the run
            # having no learning signal at all.
            dead_task = [d for d in dead if not any(h in d.lower() for h in _AUX_HINTS)]
            severity = max(severity, Severity.WARNING if dead_task else Severity.INFO)
            summaries.append(
                f"{', '.join(repr(d) for d in dead)} contribute <2% of reward variance -- they are "
                "constants as far as the gradient is concerned"
                + (
                    ", and that includes what looks like your task reward."
                    if dead_task
                    else " (expected for a saturated format reward)."
                )
            )
            prescription.append(
                "A component with no variance cannot teach anything. Either rescale it so it "
                "actually separates rollouts, or delete it and save the compute spent computing it."
            )

        n_obs = min(int(np.isfinite(v).sum()) for v in components.values())
        metrics["n_observations"] = n_obs
        if severity > Severity.INFO and n_obs < _MIN_OBS_FOR_VERDICT:
            severity = Severity.INFO
            summaries.append(
                f"Only {n_obs} logged points, so this is reported as information rather than a "
                f"verdict: below ~{_MIN_OBS_FOR_VERDICT} the shares still reflect warm-up."
            )

        if severity is not Severity.OK:
            prescription += [
                "Weight components by the variance you want them to contribute, not by the mean "
                "you want them to have. Doubling a coefficient on a saturated term changes nothing.",
                "Plot each component separately against the held-out score. A component that does "
                "not correlate with held-out performance is at best noise and at worst an exploit.",
                "If a format reward has saturated (every rollout gets it), drop it -- it is now a "
                "constant offset and its only remaining effect is to shrink the relative variance "
                "of the term you care about.",
            ]
            return self.finding(
                severity, " ".join(summaries), evidence=evidence, prescription=prescription, metrics=metrics
            )

        return self.finding(
            Severity.OK,
            f"Reward variance is reasonably spread across {len(ordered)} components "
            f"(largest: '{top_name}' at {top_share:.0%}).",
            evidence=evidence,
            metrics=metrics,
        )


def _variance_shares(components: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Robust variance share per component.

    Uses MAD rather than std so that a single catastrophic batch does not make
    one component look dominant for the whole run.
    """
    spreads = {name: stats.mad(values) ** 2 for name, values in components.items()}
    total = sum(spreads.values())
    if total <= 0:
        n = len(components)
        return {name: 1.0 / n for name in components}
    return {name: value / total for name, value in spreads.items()}
