"""Plateau detection: the run finished learning a while ago and nobody noticed.

RL runs are usually stopped on a step budget rather than on evidence. This
detector asks the only question that matters at the end of a run: has the
held-out score (or, failing that, the training reward) moved outside its own
noise band recently?

It deliberately uses a non-parametric trend test rather than "did the last value
beat the best value". Best-so-far comparisons on a noisy curve declare progress
roughly half the time by construction.
"""

from __future__ import annotations

import numpy as np

from .. import stats
from ..schema import EVAL_SCORE, REWARD_MEAN, TIME_PER_STEP, Run
from .base import Detector, Finding, Severity

#: Fraction of the run treated as "recent" when asking whether it is still moving.
_WINDOW_FRAC = 0.35
_MIN_WINDOW = 12


class Plateau(Detector):
    name = "plateau"
    title = "Learning plateau"
    requires: list = []
    optional = [EVAL_SCORE, REWARD_MEAN, TIME_PER_STEP]
    references = [
        "Mann, Nonparametric tests against trend (Econometrica, 1945)",
    ]

    def run(self, run: Run) -> Finding:
        field_name, label = _pick_signal(run)
        if field_name is None:
            return self.skip(
                "needs `eval_score` or `reward_mean` to judge whether the run is still learning",
                missing_fields=[EVAL_SCORE, REWARD_MEAN],
            )

        steps, values = run.finite(field_name)
        window = max(int(steps.size * _WINDOW_FRAC), _MIN_WINDOW)
        if steps.size < _MIN_WINDOW:
            return self.skip(f"only {steps.size} {label} observations; need at least {_MIN_WINDOW}")
        window = min(window, steps.size)

        recent_steps, recent_values = steps[-window:], values[-window:]
        tr = stats.trend(recent_steps, recent_values)
        overall = stats.trend(steps, values)

        noise = stats.mad(recent_values)
        drift = abs(tr.slope) * (recent_steps[-1] - recent_steps[0])
        snr = drift / noise if noise > 0 else float("inf")

        best = float(np.max(values))
        best_step = float(steps[int(np.argmax(values))])
        steps_since_best = float(steps[-1] - best_step)
        frac_since_best = steps_since_best / max(steps[-1] - steps[0], 1)

        evidence = [
            f"{label} over the last {window} logged points: slope {tr.slope:+.3e}/step "
            f"(Mann-Kendall p{stats.fmt_p(tr.p_value)}, tau={tr.tau:+.2f})",
            f"that is a total move of {drift:.3g} against a noise floor of {noise:.3g} "
            f"(signal-to-noise {snr:.2f})",
            f"best {label} was {best:.4g} at step {best_step:.0f}, "
            f"{steps_since_best:.0f} steps ago ({frac_since_best:.0%} of the run)",
            f"whole-run trend for reference: {overall.slope:+.3e}/step ({overall.direction})",
        ]

        metrics = {
            "signal": field_name,
            "recent_slope": tr.slope,
            "recent_p": tr.p_value,
            "snr": snr,
            "best": best,
            "best_step": best_step,
            "steps_since_best": steps_since_best,
        }

        wasted_steps = steps_since_best if frac_since_best >= 0.25 else 0.0
        cost_line = _cost_sentence(run, wasted_steps)
        if cost_line:
            evidence.append(cost_line)

        plateaued = tr.direction != "up" and snr < 1.0
        declining = tr.direction == "down"

        if declining and frac_since_best >= 0.3:
            severity = Severity.WARNING
            summary = (
                f"{label.capitalize()} has been *declining* for the last {window} points and the "
                f"best checkpoint is {steps_since_best:.0f} steps behind you."
            )
        elif plateaued and frac_since_best >= 0.3:
            severity = Severity.WARNING
            summary = (
                f"{label.capitalize()} stopped improving {steps_since_best:.0f} steps ago "
                f"({frac_since_best:.0%} of the run). Everything since has been noise."
            )
        elif plateaued:
            severity = Severity.INFO
            summary = f"{label.capitalize()} is flat over the recent window (SNR {snr:.2f})."
        else:
            severity = Severity.OK
            summary = (
                f"Still learning: {label} is improving at {tr.slope:+.3e}/step "
                f"(p{stats.fmt_p(tr.p_value)})."
            )

        if severity is Severity.OK:
            return self.finding(Severity.OK, summary, evidence=evidence, metrics=metrics)

        prescription = [
            f"Roll back to the step-{best_step:.0f} checkpoint. Later checkpoints are not better, "
            "they are just later.",
            "Stop on evidence, not on a step budget: add an early-stopping rule keyed on a "
            "held-out trend test over a fixed window.",
        ]
        if run.has(EVAL_SCORE):
            prescription.append(
                "If you want more out of this setup, the binding constraint is data, not steps -- "
                "harder prompts, a broader task mix, or a stronger verifier."
            )
        else:
            prescription.append(
                "You are judging progress by training reward alone. Add a held-out eval before "
                "spending more compute; a flat training reward and a flat held-out score are very "
                "different situations."
            )
        return self.finding(
            severity, summary, evidence=evidence, prescription=prescription, metrics=metrics
        )


def _pick_signal(run: Run):
    if run.has(EVAL_SCORE, min_points=_MIN_WINDOW):
        return EVAL_SCORE, "held-out score"
    if run.has(REWARD_MEAN, min_points=_MIN_WINDOW):
        return REWARD_MEAN, "training reward"
    return None, ""


def _cost_sentence(run: Run, wasted_steps: float) -> str:
    if wasted_steps <= 0 or not run.has(TIME_PER_STEP):
        return ""
    sec = stats.nanmedian(stats.tail(run.series[TIME_PER_STEP]))
    if not np.isfinite(sec) or sec <= 0:
        return ""
    hours = wasted_steps * sec / 3600.0
    gpus = run.config.num_gpus
    if gpus:
        return (
            f"those {wasted_steps:.0f} post-peak steps cost {hours:.1f} wall-clock hours "
            f"on {gpus} GPUs = {hours * gpus:.0f} GPU-hours"
        )
    return f"those {wasted_steps:.0f} post-peak steps cost {hours:.1f} wall-clock hours"
