"""Response-length pathologies: length hacking and truncation poisoning.

Two distinct failures share the response-length series.

**Length hacking.** Reward rises together with response length while accuracy
does not. The policy has learned that longer output scores better -- because a
length-correlated reward component exists, because the grader is an LLM that
likes verbosity, or because more tokens means more chances to emit the string
the regex is looking for.

**Truncation poisoning.** Responses hit ``max_response_length`` and get cut off.
A truncated rollout is usually scored as a failure, so the reward is measuring
the length limit rather than the policy -- and every truncated sample injects a
systematically wrong advantage into the update.
"""

from __future__ import annotations

import numpy as np

from .. import stats
from ..schema import (
    EVAL_SCORE,
    RESPONSE_LEN_CLIP_RATIO,
    RESPONSE_LEN_MEAN,
    REWARD_MEAN,
    Run,
)
from .base import Detector, Finding, Severity

_TRUNC_WARN = 0.05
_TRUNC_CRIT = 0.15
_GROWTH_WARN = 0.50  # +50% mean length over the run
_CORR_WARN = 0.60  # rank correlation between length and reward
#: Held-out gain must be at least this multiple of the length gain to count
#: as "earned". 0.25 means a 100% longer answer should buy at least 25% more
#: held-out score before we accept the extra tokens as reasoning.
_EARNED_RATIO = 0.25


class LengthPathology(Detector):
    name = "length_pathology"
    title = "Length hacking / truncation"
    requires = [RESPONSE_LEN_MEAN]
    optional = [REWARD_MEAN, EVAL_SCORE, RESPONSE_LEN_CLIP_RATIO]
    references = [
        "Yu et al., DAPO (arXiv:2503.14476) -- overlong filtering and length-aware penalties",
        "Singhal et al., A Long Way to Go: Investigating Length Correlations in RLHF (arXiv:2310.03716)",
    ]

    def run(self, run: Run) -> Finding:
        steps, length = run.finite(RESPONSE_LEN_MEAN)
        if steps.size < 8:
            return self.skip(f"only {steps.size} length observations; need at least 8")

        l0 = float(np.median(stats.head(length)))
        l_now = float(np.median(stats.tail(length, frac=0.1, minimum=3)))
        growth = (l_now - l0) / l0 if l0 > 0 else 0.0
        length_trend = stats.trend(steps, length)

        evidence = [
            f"mean response length moved {l0:.0f} -> {l_now:.0f} tokens ({growth:+.0%})",
        ]
        metrics = {
            "length_initial": l0,
            "length_current": l_now,
            "length_growth": growth,
            "length_slope": length_trend.slope,
        }
        severity = Severity.OK
        summaries = []
        prescription = []

        # -- truncation ----------------------------------------------------
        trunc_now = None
        if run.has(RESPONSE_LEN_CLIP_RATIO):
            _, trunc = run.finite(RESPONSE_LEN_CLIP_RATIO)
            trunc_now = float(np.median(stats.tail(trunc, frac=0.25)))
            metrics["truncation_rate"] = trunc_now
            evidence.append(f"{trunc_now:.1%} of responses are hitting the length limit")
            if trunc_now >= _TRUNC_CRIT:
                severity = max(severity, Severity.CRITICAL)
                summaries.append(
                    f"{trunc_now:.0%} of rollouts are truncated -- their rewards measure your "
                    "length cap, not your policy."
                )
            elif trunc_now >= _TRUNC_WARN:
                severity = max(severity, Severity.WARNING)
                summaries.append(f"{trunc_now:.0%} of rollouts are truncated.")
            if trunc_now >= _TRUNC_WARN:
                prescription += [
                    "Apply overlong filtering (DAPO): mask truncated rollouts out of the loss "
                    "instead of scoring them as failures. Scoring a cut-off answer as wrong "
                    "teaches the model that its correct prefix was bad.",
                    "Alternatively raise `max_response_length`, but re-check throughput first -- "
                    "generation cost is roughly linear in this cap.",
                ]

        # -- length hacking ------------------------------------------------
        if run.has(REWARD_MEAN, min_points=8):
            r_steps, r_vals = run.finite(REWARD_MEAN)
            reward_aligned = np.interp(steps, r_steps, r_vals)
            corr = stats.spearman(length, reward_aligned)
            metrics["length_reward_corr"] = corr
            evidence.append(f"length-reward rank correlation: rho={corr:+.2f}")

            # `unearned` is a three-state signal: True when held-out gain does
            # not justify the extra tokens, False when it does, None when there
            # is no held-out score to ask.
            unearned = None
            if run.has(EVAL_SCORE, min_points=4):
                e_steps, e_vals = run.finite(EVAL_SCORE)
                eval_trend = stats.trend(e_steps, e_vals)
                eval_gain = _relative_gain(e_vals)
                metrics["eval_slope"] = eval_trend.slope
                metrics["eval_relative_gain"] = eval_gain
                # A binary "did the eval improve?" test is too blunt: real length
                # hacking usually comes with a *small* genuine gain alongside a
                # large one bought with tokens. What matters is proportionality --
                # did capability grow anything like as fast as output length?
                unearned = eval_trend.direction != "up" or eval_gain < _EARNED_RATIO * growth
                evidence.append(
                    f"held-out score trend: {eval_trend.slope:+.3e}/step "
                    f"({eval_trend.direction}, p{stats.fmt_p(eval_trend.p_value)}); it gained "
                    f"{eval_gain:+.0%} while output length gained {growth:+.0%}"
                )

            hacking = growth >= _GROWTH_WARN and corr >= _CORR_WARN
            # Length growth on its own is not a pathology. Reasoning models are
            # *supposed* to think for longer as they improve. The finding only
            # escalates when a held-out score says the extra tokens bought
            # nothing -- and when there is no held-out score, this detector says
            # so rather than guessing.
            if hacking and unearned is True:
                severity = max(severity, Severity.CRITICAL)
                summaries.append(
                    f"Length grew {growth:+.0%} in lockstep with reward (rho={corr:+.2f}) while "
                    "the held-out score did not improve. The policy is being paid by the token."
                )
            elif hacking and unearned is None:
                severity = max(severity, Severity.INFO)
                summaries.append(
                    f"Length grew {growth:+.0%} and tracks reward closely (rho={corr:+.2f}), with "
                    "no held-out eval logged to tell earned reasoning from padding."
                )
                prescription.append(
                    "Log a held-out score. Without one, length-driven reward gain and genuine "
                    "capability gain are indistinguishable from the training curve alone."
                )
            elif growth >= _GROWTH_WARN and unearned is not False:
                severity = max(severity, Severity.INFO)
                summaries.append(f"Response length grew {growth:+.0%} over the run.")

            if hacking and unearned is True:
                prescription += [
                    "Regress reward on length across your rollouts. If length explains most of "
                    "the reward variance, the reward function is the bug.",
                    "Add a length penalty or normalise the reward by length, then confirm the "
                    "held-out score still moves -- a penalty that also kills accuracy means the "
                    "extra tokens were doing real work.",
                    "If you grade with an LLM judge, re-run the judge with position- and "
                    "length-debiasing; verbosity bias in judges is well documented.",
                ]

        if severity is Severity.OK:
            return self.finding(
                Severity.OK,
                f"Response length is stable ({growth:+.0%} over the run) with no truncation problem.",
                evidence=evidence,
                metrics=metrics,
            )

        return self.finding(
            severity,
            " ".join(summaries),
            evidence=evidence,
            prescription=prescription,
            metrics=metrics,
            wasted_fraction=trunc_now if trunc_now and trunc_now >= _TRUNC_WARN else None,
        )


def _relative_gain(values: np.ndarray) -> float:
    """Fractional change from the start of a series to its end, robustly measured."""
    if values.size < 2:
        return 0.0
    start = float(np.median(values[: max(values.size // 4, 1)]))
    end = float(np.median(stats.tail(values, frac=0.25, minimum=1)))
    if abs(start) < 1e-9:
        return 0.0 if abs(end) < 1e-9 else float("inf")
    return (end - start) / abs(start)
