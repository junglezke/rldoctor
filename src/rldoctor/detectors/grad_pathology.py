"""Gradient-norm pathologies: spikes, vanishing updates and NaNs.

The gradient norm is the cheapest early-warning signal in the whole run, and
almost nobody reads it until after a crash. Three things worth catching:

* **NaN / inf** -- the run is already dead; every step after the first NaN is
  wasted wall-clock.
* **Exactly-zero updates** -- steps whose gradient norm is not small but
  literally ``0.0``. In GRPO that means every rollout in every group scored the
  same, so the centred advantage was zero. You generated and paid for those
  rollouts and moved the policy not at all. On one real public run this was 88%
  of steps, rising from 40% at the start to 93% at the end.
* **Vanishing gradient** -- the norm decays towards zero while the loss does
  not. In GRPO this is usually downstream of zero-variance groups: no advantage
  spread means no gradient, no matter how many tokens you generate.
* **Spikes** -- isolated updates orders of magnitude larger than the median.
  With gradient clipping these are survivable but they still corrupt the
  optimiser's second-moment estimates for many steps afterwards.
"""

from __future__ import annotations

import numpy as np

from .. import stats
from ..schema import GRAD_NORM, POLICY_LOSS, ZERO_VAR_GROUP_FRAC, Run
from .base import Detector, Finding, Severity

_SPIKE_SIGMA = 8.0
#: A spike must also be materially larger than the typical step, not merely a
#: statistical outlier. On a tightly clustered series the MAD is tiny, so 8
#: robust sigmas can be reached by a value only twice the median -- which is not
#: a spike by any useful definition. Found on a real run where every "spike" was
#: 2x the median.
_SPIKE_RATIO = 3.0
_VANISH_RATIO = 0.05  # current norm below 5% of the early-run norm
#: Fraction of steps with a gradient norm of exactly zero that is worth
#: reporting. Exact zeros are discrete dead updates, not a decay, and in GRPO
#: they are the fingerprint of degenerate groups.
_ZERO_STEP_INFO = 0.20
_ZERO_STEP_WARN = 0.40
_ZERO_STEP_CRIT = 0.70


class GradientPathology(Detector):
    name = "gradient_pathology"
    title = "Gradient norm pathologies"
    requires = [GRAD_NORM]
    optional = [POLICY_LOSS, ZERO_VAR_GROUP_FRAC]
    references = [
        "Pascanu et al., On the difficulty of training recurrent neural networks (arXiv:1211.5063)",
    ]

    def run(self, run: Run) -> Finding:
        # Only values the log actually carried as non-finite count. A `nan` in
        # the aligned series usually means the metric was not logged at that
        # step -- every run that evaluates on a different cadence from training
        # is full of them -- and counting those reports numerical death on a
        # perfectly healthy job.
        nonfinite_at = run.nonfinite_steps.get(GRAD_NORM, [])
        n_nonfinite = len(nonfinite_at)
        steps, grad = run.finite(GRAD_NORM)
        if steps.size < 8:
            return self.skip(f"only {steps.size} finite grad-norm observations; need at least 8")

        metrics = {"n_nonfinite": n_nonfinite}
        evidence = []
        summaries = []
        prescription = []
        severity = Severity.OK

        # -- NaN / inf -----------------------------------------------------
        if n_nonfinite:
            first_bad = min(nonfinite_at)
            severity = max(severity, Severity.CRITICAL)
            summaries.append(
                f"{n_nonfinite} non-finite gradient norms, first at step "
                f"{first_bad:.0f}. Training is numerically dead from that point."
            )
            evidence.append(
                f"the log records a non-finite gradient norm at {n_nonfinite} step(s) "
                f"(first: step {first_bad:.0f})"
            )
            prescription += [
                "Resume from the last checkpoint before the first NaN -- everything after it is "
                "garbage even if the loss curve kept plotting.",
                "Switch the loss and logits to bf16 or fp32 if you are on fp16; overflow in the "
                "log-prob ratio is the usual culprit in RLVR.",
                "Clamp the importance ratio and the reward before the loss, and assert finiteness "
                "on the advantage tensor each step so you fail loudly instead of silently.",
            ]

        # -- exactly-zero updates ------------------------------------------
        # Distinct from a decay: these steps did not move the policy at all.
        # The rollouts were still generated and paid for.
        zero_frac = float(np.mean(grad == 0.0))
        metrics["zero_grad_step_frac"] = zero_frac
        if zero_frac >= _ZERO_STEP_INFO:
            nonzero = grad[grad > 0]
            typical = float(np.median(nonzero)) if nonzero.size else 0.0
            early = float(np.mean(grad[: max(grad.size // 4, 1)] == 0.0))
            late = float(np.mean(grad[-max(grad.size // 4, 1) :] == 0.0))
            evidence.append(
                f"{zero_frac:.0%} of steps have a gradient norm of exactly zero "
                f"({early:.0%} in the first quarter, {late:.0%} in the last); when the "
                f"update does fire its norm is typically {typical:.3g}"
            )
            severity = max(
                severity,
                Severity.CRITICAL
                if zero_frac >= _ZERO_STEP_CRIT
                else Severity.WARNING
                if zero_frac >= _ZERO_STEP_WARN
                else Severity.INFO,
            )
            summaries.append(
                f"{zero_frac:.0%} of optimiser steps produced exactly zero gradient. Those "
                "rollouts were generated and paid for and moved the policy not at all."
            )
            prescription += [
                "In GRPO an exactly-zero update almost always means every rollout in every "
                "group scored the same, so the centred advantage was zero. Log "
                "`frac_reward_zero_std` to confirm, then fix the group variance rather than "
                "the optimiser.",
                "If the zero fraction is rising through the run, your data has become too "
                "easy (or too hard) for the current policy -- filter by measured pass rate "
                "and refresh the pool.",
            ]
            zero_waste = zero_frac
        else:
            zero_waste = None

        # -- vanishing -----------------------------------------------------
        g0 = float(np.median(stats.head(grad)))
        g_now = float(np.median(stats.tail(grad, frac=0.1, minimum=3)))
        metrics.update({"grad_initial": g0, "grad_current": g_now})
        evidence.append(f"grad norm moved {g0:.3g} -> {g_now:.3g}")

        if g0 > 0 and g_now / g0 <= _VANISH_RATIO:
            explained = ""
            if run.has(ZERO_VAR_GROUP_FRAC):
                _, zvf = run.finite(ZERO_VAR_GROUP_FRAC)
                frac = float(np.median(stats.tail(zvf, frac=0.25)))
                if frac >= 0.5:
                    explained = (
                        f" This is explained by the group-variance collapse ({frac:.0%} degenerate "
                        "groups) -- fix that first; the gradient is a symptom, not the disease."
                    )
                metrics["zero_var_frac"] = frac
            severity = max(severity, Severity.WARNING)
            summaries.append(
                f"Gradient norm has fallen to {g_now / g0:.1%} of its early-run value: the "
                f"optimiser is barely moving.{explained}"
            )
            if not explained:
                prescription += [
                    "Check whether the advantage tensor is near-zero. In GRPO a vanishing "
                    "gradient is almost always an advantage problem, not an optimiser problem.",
                    "Verify the learning-rate schedule has not decayed to ~0 earlier than you "
                    "intended.",
                ]

        # -- spikes --------------------------------------------------------
        z = stats.robust_z(grad)
        median = float(np.median(grad))
        big_enough = grad > max(median, 1e-12) * _SPIKE_RATIO
        spikes = np.where((z > _SPIKE_SIGMA) & big_enough)[0]
        metrics["n_spikes"] = int(spikes.size)
        if spikes.size:
            worst = int(spikes[np.argmax(grad[spikes])])
            evidence.append(
                f"{spikes.size} spike(s) beyond {_SPIKE_SIGMA:.0f} robust sigma; worst is "
                f"{grad[worst]:.3g} at step {steps[worst]:.0f} "
                f"({grad[worst] / max(np.median(grad), 1e-12):.0f}x the median)"
            )
            if spikes.size >= 3:
                severity = max(severity, Severity.WARNING)
                summaries.append(f"{spikes.size} large gradient spikes across the run.")
                prescription.append(
                    "Tighten `max_grad_norm` and look at the rollouts from the spiking steps -- "
                    "recurring spikes usually trace to a handful of pathological prompts or to a "
                    "reward function that occasionally returns an extreme value."
                )
            else:
                severity = max(severity, Severity.INFO)
                summaries.append(f"{spikes.size} isolated gradient spike(s).")

        if severity is Severity.OK:
            return self.finding(
                Severity.OK,
                f"Gradient norm is healthy ({g0:.3g} -> {g_now:.3g}, no spikes or NaNs).",
                evidence=evidence,
                metrics=metrics,
            )
        return self.finding(
            severity,
            " ".join(summaries),
            evidence=evidence,
            prescription=prescription,
            metrics=metrics,
            wasted_fraction=zero_waste,
        )
