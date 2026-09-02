"""KL drift from the reference policy.

KL(policy || reference) is the leash. Two failure modes matter:

* **Runaway drift** -- KL grows without bound while the held-out score does not.
  The policy is walking away from its initialisation and paying for it in
  fluency and general capability, buying nothing in return. This is where
  "the model got better at math and forgot how to write" comes from.
* **A strangled policy** -- KL is pinned near zero because the coefficient is
  too large. The run is safe, stable, and learning nothing.

The interesting quantity is not KL itself but KL *per unit of held-out
progress*: how much capability drift you are spending per point of score.
"""

from __future__ import annotations

import numpy as np

from .. import stats
from ..schema import EVAL_SCORE, KL, REWARD_MEAN, Run
from .base import Detector, Finding, Severity

_KL_FLAT = 1e-4  # below this, the leash is doing all the work
_KL_HIGH = 0.5
_KL_EXTREME = 2.0


class KLDrift(Detector):
    name = "kl_drift"
    title = "KL drift from reference policy"
    requires = [KL]
    optional = [EVAL_SCORE, REWARD_MEAN]
    references = [
        "Stiennon et al., Learning to Summarize from Human Feedback (arXiv:2009.01325) -- KL as the RLHF budget",
        "Gao et al., Scaling Laws for Reward Model Overoptimization (arXiv:2210.10760)",
    ]

    def run(self, run: Run) -> Finding:
        steps, kl = run.finite(KL)
        if steps.size < 8:
            return self.skip(f"only {steps.size} KL observations; need at least 8")

        kl_now = float(np.median(stats.tail(kl, frac=0.1, minimum=3)))
        kl_max = float(np.nanmax(kl))
        tr = stats.trend(steps, kl)

        # Curvature: is the drift accelerating? Compare the slope of the second
        # half against the first. Accelerating KL is the dangerous shape.
        mid = steps.size // 2
        slope_early = stats.trend(steps[:mid], kl[:mid]).slope if mid >= 8 else 0.0
        slope_late = stats.trend(steps[mid:], kl[mid:]).slope if steps.size - mid >= 8 else tr.slope
        accelerating = slope_late > max(slope_early, 0) * 1.5 and slope_late > 0

        evidence = [
            f"KL is {kl_now:.4g} now (peak {kl_max:.4g}), trending {tr.slope:+.2e}/step "
            f"(p{stats.fmt_p(tr.p_value)})",
        ]
        metrics = {
            "kl_current": kl_now,
            "kl_max": kl_max,
            "kl_slope": tr.slope,
            "kl_slope_early": slope_early,
            "kl_slope_late": slope_late,
            "accelerating": accelerating,
        }

        progress = _progress_trend(run)
        if progress is not None:
            label, ptrend = progress
            metrics["progress_slope"] = ptrend.slope
            metrics["progress_field"] = label
            evidence.append(
                f"{label} is moving {ptrend.slope:+.3e}/step ({ptrend.direction}, "
                f"p{stats.fmt_p(ptrend.p_value)}) over the same window"
            )
            if tr.slope > 0 and ptrend.slope > 0:
                efficiency = tr.slope / ptrend.slope
                metrics["kl_per_progress"] = efficiency
                evidence.append(
                    f"you are spending {efficiency:.3g} nats of KL per unit of {label} gained"
                )

        stalled = progress is not None and progress[1].direction != "up"

        if kl_now >= _KL_EXTREME and stalled:
            severity = Severity.CRITICAL
            summary = (
                f"KL has reached {kl_now:.3g} and the model has stopped improving. You are "
                "paying for capability drift and receiving nothing."
            )
        elif kl_now >= _KL_EXTREME or (accelerating and kl_now >= _KL_HIGH):
            severity = Severity.WARNING
            summary = (
                f"KL is {kl_now:.3g} and {'accelerating' if accelerating else 'high'}. Expect "
                "off-task regressions (formatting, instruction-following, general chat) even if "
                "your target metric looks fine."
            )
        elif kl_now >= _KL_HIGH:
            severity = Severity.INFO
            summary = f"KL is elevated ({kl_now:.3g}). Worth a general-capability spot check."
        elif kl_max <= _KL_FLAT:
            severity = Severity.WARNING
            summary = (
                f"KL never exceeded {kl_max:.2g}: the policy is effectively pinned to the "
                "reference. Your KL penalty is doing the training, and it is not training."
            )
            return self.finding(
                severity,
                summary,
                evidence=evidence,
                prescription=[
                    "Lower `kl_coef` by an order of magnitude, or drop the KL term entirely -- "
                    "several strong RLVR recipes (DAPO among them) run with no KL penalty at all "
                    "and rely on clipping for stability.",
                    "Confirm the reference model is actually the SFT init and not a stale copy of "
                    "the policy; a self-referencing KL term is always ~0 and always useless.",
                ],
                metrics=metrics,
            )
        else:
            severity = Severity.OK
            summary = f"KL is well behaved (now {kl_now:.4g}, peak {kl_max:.4g})."

        if severity is Severity.OK:
            return self.finding(Severity.OK, summary, evidence=evidence, metrics=metrics)

        return self.finding(
            severity,
            summary,
            evidence=evidence,
            prescription=[
                "Raise `kl_coef` (typical range 1e-3 to 1e-2) or switch to an adaptive KL "
                "controller that targets a fixed KL budget instead of a fixed coefficient.",
                "Evaluate on a general-capability suite, not just the RL task. KL drift is "
                "cheap to detect here and expensive to discover after release.",
                "If the drift is accelerating, cap it: hard-stop the run when KL crosses a "
                "budget, and keep the last checkpoint below it."
                if accelerating
                else "Track KL per unit of held-out gain across runs and pick the checkpoint on "
                "the knee of that curve rather than on peak reward.",
            ],
            metrics=metrics,
        )


def _progress_trend(run: Run):
    for field_name, label in ((EVAL_SCORE, "held-out score"), (REWARD_MEAN, "training reward")):
        if run.has(field_name, min_points=6):
            steps, values = run.finite(field_name)
            return label, stats.trend(steps, values)
    return None
