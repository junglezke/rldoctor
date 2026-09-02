"""Reward-eval divergence -- the signature of a verifier being gamed.

The training reward is a proxy.  The held-out score is the thing you actually
want.  When the proxy climbs while the held-out score stalls or falls, the
policy has found something the verifier rewards and the task does not: a
degenerate output format, a test-suite loophole, a rubric keyword.

Recent audits put the scale of the underlying problem bluntly: 28.5% of
SWE-bench Verified tasks have test suites weak enough to accept a
Docker-verified *incorrect* patch. If your verifier is that permeable, reward
going up is not evidence of anything.

This detector measures the *hacking gap*: how far the normalised training
reward has pulled away from the normalised held-out score.
"""

from __future__ import annotations

import numpy as np

from .. import stats
from ..schema import EVAL_SCORE, REWARD_MEAN, Run
from .base import Detector, Finding, Severity

_GAP_WARN = 0.15
_GAP_CRIT = 0.30


class RewardHacking(Detector):
    name = "reward_hacking"
    title = "Reward-eval divergence (verifier gaming)"
    requires = [REWARD_MEAN, EVAL_SCORE]
    references = [
        "Auditing Reward Hackability in Code RL Training Environments (arXiv:2606.16062)",
        "LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking (arXiv:2604.15149)",
        "Reward Hacking in the Era of Large Models: Mechanisms and Emergent Behaviours (arXiv:2604.13602)",
    ]

    def run(self, run: Run) -> Finding:
        r_steps, reward = run.finite(REWARD_MEAN)
        e_steps, evals = run.finite(EVAL_SCORE)
        if e_steps.size < 4:
            return self.skip(
                f"only {e_steps.size} held-out evaluations; run eval at least 4 times "
                "to make divergence measurable"
            )

        # Evals are logged less often than training metrics, so compare on the
        # eval grid and interpolate the training reward onto it.
        reward_on_eval = np.interp(e_steps, r_steps, reward)

        r_norm = _normalise(reward_on_eval)
        e_norm = _normalise(evals)
        gap = r_norm - e_norm

        gap_now = float(np.median(stats.tail(gap, frac=0.34, minimum=2)))
        gap_trend = stats.trend(e_steps, gap)
        reward_trend = stats.trend(e_steps, reward_on_eval)
        eval_trend = stats.trend(e_steps, evals)
        rank_corr = stats.spearman(reward_on_eval, evals)

        evidence = [
            f"normalised training reward has moved {r_norm[-1]:+.2f} from its start; "
            f"held-out score has moved {e_norm[-1]:+.2f} -- hacking gap {gap_now:+.2f}",
            f"rank correlation between training reward and held-out score: rho={rank_corr:+.2f} "
            f"over {e_steps.size} evaluations",
        ]
        if gap_trend.significant:
            evidence.append(
                f"the gap is {'widening' if gap_trend.slope > 0 else 'closing'} "
                f"({gap_trend.slope:+.2e}/step, p{stats.fmt_p(gap_trend.p_value)})"
            )

        diverging = reward_trend.direction == "up" and eval_trend.direction != "up"
        regressing = reward_trend.direction == "up" and eval_trend.direction == "down"

        if regressing or diverging and gap_now >= _GAP_CRIT:
            severity = Severity.CRITICAL
        elif diverging and gap_now >= _GAP_WARN or rank_corr < 0.0 and e_steps.size >= 6:
            severity = Severity.WARNING
        elif gap_now >= _GAP_WARN:
            severity = Severity.INFO
        else:
            severity = Severity.OK

        metrics = {
            "hacking_gap": gap_now,
            "rank_correlation": rank_corr,
            "reward_slope": reward_trend.slope,
            "eval_slope": eval_trend.slope,
            "gap_slope": gap_trend.slope,
            "n_evals": int(e_steps.size),
        }

        if severity is Severity.OK:
            return self.finding(
                Severity.OK,
                f"Training reward and held-out score are moving together (rho={rank_corr:+.2f}).",
                evidence=evidence,
                metrics=metrics,
            )

        if regressing:
            summary = (
                "Training reward is going up while the held-out score goes *down*. "
                "This is the textbook signature of reward hacking, not of learning."
            )
        elif diverging:
            summary = (
                "Training reward is climbing while the held-out score is flat. The policy is "
                "optimising something your verifier rewards and your task does not."
            )
        else:
            summary = (
                f"Training reward and held-out score have drifted apart (gap {gap_now:+.2f}, "
                f"rho={rank_corr:+.2f}). Worth an eyeball before you trust the reward curve."
            )

        return self.finding(
            severity,
            summary,
            evidence=evidence,
            prescription=[
                "Read 20 high-reward rollouts end to end. Reward hacks are almost always "
                "obvious on inspection and almost never visible in aggregate metrics.",
                "Audit the verifier, not the model. For code tasks, check whether your tests "
                "accept known-wrong patches; for rubric rewards, check for keyword stuffing.",
                "Hold out a second verifier that grades the same task differently (isomorphic "
                "verification). A gap that exists against one grader and not another localises "
                "the exploit to the grader.",
                "Add an explicit anti-hack penalty for the specific exploit once you have "
                "identified it, and re-measure the gap rather than assuming it closed.",
                "Roll back to the checkpoint before the gap opened -- later checkpoints have "
                "already been shaped by the exploit.",
            ],
            metrics=metrics,
        )


def _normalise(values: np.ndarray) -> np.ndarray:
    """Express a series as change from its own start, in units of its own spread.

    Training reward and held-out accuracy live on different scales, so a raw
    difference is meaningless. Anchoring both to their starting level and
    dividing by a robust spread makes 'how far has this moved' comparable.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    baseline = float(np.median(values[: max(values.size // 4, 1)]))
    scale = stats.mad(values)
    if scale <= 0:
        scale = float(np.max(np.abs(values - baseline))) or 1.0
    return (values - baseline) / scale
