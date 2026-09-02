"""Entropy collapse -- exploration dies, and the run keeps logging as if it were alive.

Policy entropy in RLVR decays monotonically by default.  Some decay is healthy:
the policy is committing to solutions it has learned.  Collapse is when entropy
keeps falling *after* reward has stopped improving -- at that point the model is
no longer exploring, every group returns the same rollout, and the remaining
GPU-hours buy nothing.

The two cases look identical on an entropy plot, which is why this detector
conditions on the reward trend before it fires, and why it reports a projected
step at which exploration runs out rather than a bare "entropy is low" warning.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import stats
from ..schema import ENTROPY, EVAL_SCORE, REWARD_MEAN, TIME_PER_STEP, Run
from .base import Detector, Finding, Severity

#: Entropy below this fraction of its initial value is treated as no exploration
#: budget left. Chosen to fire before the policy fully saturates, not after.
_FLOOR_FRAC = 0.10


class EntropyCollapse(Detector):
    name = "entropy_collapse"
    title = "Entropy collapse (exploration exhausted)"
    requires = [ENTROPY]
    optional = [REWARD_MEAN, EVAL_SCORE, TIME_PER_STEP]
    references = [
        "Cui et al., The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models (Clip-Cov / KL-Cov)",
        "Yu et al., DAPO (arXiv:2503.14476) -- clip-higher as an entropy remedy",
        "Understanding and Preventing Entropy Collapse in RLVR with On-Policy Regularisation (arXiv:2605.11491)",
    ]

    def run(self, run: Run) -> Finding:
        steps, entropy = run.finite(ENTROPY)
        if steps.size < 8:
            return self.skip(f"only {steps.size} entropy observations; need at least 8")

        h0 = float(np.median(stats.head(entropy)))
        h_now = float(np.median(stats.tail(entropy, frac=0.1, minimum=3)))
        if h0 <= 0:
            return self.skip("initial entropy is non-positive; series looks unusable")

        decay = 1.0 - (h_now / h0)
        floor = _FLOOR_FRAC * h0

        # Trend over the recent half only: early-run decay is expected and would
        # bias a whole-run slope towards a false alarm.
        half = max(len(steps) // 2, 8)
        tr = stats.trend(steps[-half:], entropy[-half:])
        cliff = stats.cusum_changepoint(steps, entropy, min_sigma=3.0)

        # Entropy decays multiplicatively, so the projection picks between a
        # linear and an exponential model rather than assuming either.
        crossing, eta_model = stats.project_threshold(steps[-half:], entropy[-half:], floor)
        eta_steps = int(crossing - steps[-1]) if crossing is not None else None
        if eta_steps is not None and eta_steps < 0:
            eta_steps = None

        reward_trend = _reward_trend(run)
        reward_still_improving = reward_trend is not None and reward_trend.direction == "up"

        evidence = [
            f"entropy fell from {h0:.4f} to {h_now:.4f} over {int(steps[-1] - steps[0])} steps "
            f"({decay:.0%} of the initial exploration budget spent)",
        ]
        if tr.significant:
            evidence.append(
                f"still falling in the recent half at {tr.slope:+.3e}/step "
                f"(Mann-Kendall p{stats.fmt_p(tr.p_value)}, tau={tr.tau:+.2f})"
            )
        else:
            evidence.append(
                f"recent-half trend is not statistically significant (p{stats.fmt_p(tr.p_value)}); "
                "entropy has stabilised"
            )
        if eta_steps is not None and eta_steps > 0:
            evidence.append(_eta_sentence(run, eta_steps, floor, eta_model))
        if cliff is not None and cliff.is_drop:
            evidence.append(
                f"abrupt drop detected at step {cliff.step:.0f}: "
                f"{cliff.before:.4f} -> {cliff.after:.4f} ({cliff.magnitude:.1f} robust sigma). "
                "A cliff rather than a slope usually means a config change, a resume from "
                "checkpoint, or a reward-function bug landing mid-run."
            )
        if reward_trend is not None:
            verdict = (
                "reward is still improving, so this decay is at least buying something"
                if reward_still_improving
                else "reward has stopped improving, so this decay is buying nothing"
            )
            evidence.append(
                f"{_reward_label(run)} trend over the same window: {reward_trend.slope:+.3e}/step "
                f"(p{stats.fmt_p(reward_trend.p_value)}) -- {verdict}"
            )

        metrics = {
            "entropy_initial": h0,
            "entropy_current": h_now,
            "relative_decay": decay,
            "floor": floor,
            "recent_slope": tr.slope,
            "recent_p": tr.p_value,
            "eta_steps_to_floor": eta_steps,
            "eta_model": eta_model,
            "reward_still_improving": reward_still_improving,
        }

        run_length = float(steps[-1] - steps[0]) or 1.0
        severity = self._severity(h_now, floor, eta_steps, run_length, reward_still_improving)
        if severity is Severity.OK:
            return self.finding(
                Severity.OK,
                f"Entropy has decayed {decay:.0%} but is stable and exploration budget remains.",
                evidence=evidence,
                metrics=metrics,
            )

        if h_now <= floor:
            summary = (
                f"Exploration is exhausted: entropy is at {h_now:.4f}, below the "
                f"{_FLOOR_FRAC:.0%}-of-initial floor ({floor:.4f}). Rollouts within a group are "
                "near-identical, so GRPO has almost nothing left to rank."
            )
        elif eta_steps is not None:
            summary = (
                f"Entropy is collapsing and will hit the exploration floor in roughly "
                f"{eta_steps} steps. {'Reward has already plateaued.' if not reward_still_improving else ''}"
            ).strip()
        else:
            summary = f"Entropy has fallen {decay:.0%} from its starting value and is still declining."

        return self.finding(
            severity,
            summary,
            evidence=evidence,
            prescription=_prescribe(run, reward_still_improving, cliff is not None),
            metrics=metrics,
        )

    @staticmethod
    def _severity(
        h_now: float,
        floor: float,
        eta_steps: Optional[int],
        run_length: float,
        reward_improving: bool,
    ) -> Severity:
        """Severity is driven by *how soon* exploration runs out, not by how much
        entropy has already been spent.

        Entropy always decays, so a threshold on cumulative decay fires on every
        healthy run eventually. What distinguishes collapse is the arrival time:
        entropy heading for the floor within the horizon you still intend to
        train for is a problem; entropy that will get there in ten times the run
        length is just convergence.
        """
        if h_now <= floor:
            # Below the floor with reward still climbing is late-stage
            # convergence: a real risk, but not an emergency.
            return Severity.WARNING if reward_improving else Severity.CRITICAL
        if eta_steps is None or eta_steps <= 0:
            return Severity.OK
        if eta_steps <= 0.25 * run_length:
            return Severity.WARNING if reward_improving else Severity.CRITICAL
        if eta_steps <= 1.0 * run_length:
            return Severity.INFO if reward_improving else Severity.WARNING
        return Severity.OK


def _reward_trend(run: Run):
    """Trend of the most trustworthy progress signal available."""
    for field_name in (EVAL_SCORE, REWARD_MEAN):
        if run.has(field_name, min_points=8):
            steps, values = run.finite(field_name)
            half = max(len(steps) // 2, 8)
            return stats.trend(steps[-half:], values[-half:])
    return None


def _reward_label(run: Run) -> str:
    return "held-out eval" if run.has(EVAL_SCORE, min_points=8) else "training reward"


def _eta_sentence(run: Run, eta_steps: int, floor: float, model: str) -> str:
    base = (
        f"extrapolating the {model} fit, entropy reaches the floor ({floor:.4f}) in about "
        f"{eta_steps} steps"
    )
    if run.has(TIME_PER_STEP):
        sec = stats.nanmedian(stats.tail(run.series[TIME_PER_STEP]))
        if np.isfinite(sec) and sec > 0:
            hours = eta_steps * sec / 3600.0
            return base + f" -- roughly {hours:.1f} GPU-wall-clock hours from now"
    return base


def _prescribe(run: Run, reward_improving: bool, had_cliff: bool) -> list:
    prescription = []
    if had_cliff:
        prescription.append(
            "Investigate the abrupt drop first. Diff your config around that step and check "
            "for a resumed checkpoint or a changed reward function before tuning anything."
        )
    eps_high = run.config.clip_eps_high
    if eps_high is None or eps_high <= 0.2:
        prescription.append(
            "Decouple the clip bounds (DAPO 'clip-higher'): raise `clip_eps_high` to ~0.28 while "
            "leaving `clip_eps_low` at 0.2. Low-probability tokens are currently clipped away "
            "before they can be reinforced, which is a primary driver of entropy collapse."
        )
    coef = run.config.entropy_coef
    if not coef:
        prescription.append(
            "Add a small entropy bonus (start at 1e-3 and tune by an order of magnitude). "
            "Treat it as a floor-holding device, not a cure -- a large bonus trades collapse "
            "for a different instability."
        )
    else:
        prescription.append(
            f"Entropy coefficient is {coef:g} and clearly not holding. Prefer covariance-targeted "
            "interventions (Clip-Cov / KL-Cov) over simply raising it."
        )
    if not reward_improving:
        prescription.append(
            "Reward has plateaued while entropy keeps falling: this run is finished learning. "
            "Stop it, keep the best checkpoint, and spend the remaining budget on harder data."
        )
    prescription.append(
        "Raise rollout temperature slightly, or sample with a higher top-p, to restore group "
        "diversity without touching the loss."
    )
    return prescription
