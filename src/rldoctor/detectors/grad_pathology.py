"""Gradient-norm pathologies: spikes, vanishing updates and NaNs.

The gradient norm is the cheapest early-warning signal in the whole run, and
almost nobody reads it until after a crash. Three things worth catching:

* **NaN / inf** -- the run is already dead; every step after the first NaN is
  wasted wall-clock.
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
_VANISH_RATIO = 0.05  # current norm below 5% of the early-run norm


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
        spikes = np.where(z > _SPIKE_SIGMA)[0]
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
            severity, " ".join(summaries), evidence=evidence, prescription=prescription, metrics=metrics
        )
