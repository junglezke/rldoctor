"""PPO clip saturation and asymmetric clipping.

The PPO/GRPO surrogate clips the importance ratio into ``[1-eps_low,
1+eps_high]``. A clipped token contributes no gradient. So the clip fraction is
the fraction of your update that was thrown away, and nobody watches it.

Two shapes matter:

* **Saturation.** A high overall clip fraction means the policy is moving far
  faster per step than the trust region allows. Your effective learning rate is
  much smaller than the one in your config, and the run is being held together
  by the clip rather than by the optimiser.
* **Asymmetry.** When lower-bound clipping dominates upper-bound clipping, the
  update is systematically better at suppressing tokens than at promoting them.
  Low-probability tokens -- exactly the ones exploration needs -- get clipped
  away before they can ever be reinforced. This is the mechanism DAPO's
  "clip-higher" was introduced to fix, and it feeds directly into entropy
  collapse.
"""

from __future__ import annotations

import numpy as np

from .. import stats
from ..schema import ENTROPY, PG_CLIPFRAC, PG_CLIPFRAC_HIGH, PG_CLIPFRAC_LOW, Run
from .base import Detector, Finding, Severity

_CLIP_INFO = 0.10
_CLIP_WARN = 0.20
_CLIP_CRIT = 0.40
_ASYM_RATIO = 3.0


class ClipSaturation(Detector):
    name = "clip_saturation"
    title = "PPO clip saturation / asymmetry"
    requires: list = []
    optional = [PG_CLIPFRAC, PG_CLIPFRAC_LOW, PG_CLIPFRAC_HIGH, ENTROPY]
    references = [
        "Schulman et al., Proximal Policy Optimization Algorithms (arXiv:1707.06347)",
        "Yu et al., DAPO (arXiv:2503.14476) -- decoupled clip bounds ('clip-higher')",
    ]

    def run(self, run: Run) -> Finding:
        has_total = run.has(PG_CLIPFRAC)
        has_sides = run.has(PG_CLIPFRAC_LOW) and run.has(PG_CLIPFRAC_HIGH)
        if not has_total and not has_sides:
            return self.skip(
                "needs `pg_clipfrac` (verl: `actor/pg_clipfrac`, TRL: `clip_ratio/region_mean`) "
                "or both one-sided clip fractions",
                missing_fields=[PG_CLIPFRAC],
            )

        evidence = []
        metrics = {}
        severity = Severity.OK
        summaries = []
        prescription = []

        if has_total:
            steps, clip = run.finite(PG_CLIPFRAC)
            clip_now = float(np.median(stats.tail(clip, frac=0.25)))
            tr = stats.trend(steps, clip)
            metrics["clipfrac"] = clip_now
            metrics["clipfrac_slope"] = tr.slope
            evidence.append(
                f"{clip_now:.1%} of token updates are being clipped away "
                f"(trend {tr.slope:+.2e}/step, p{stats.fmt_p(tr.p_value)})"
            )
            if clip_now >= _CLIP_CRIT:
                severity = max(severity, Severity.CRITICAL)
                summaries.append(
                    f"{clip_now:.0%} of the policy update is discarded by clipping -- your "
                    "effective learning rate bears no relation to your configured one."
                )
            elif clip_now >= _CLIP_WARN:
                severity = max(severity, Severity.WARNING)
                summaries.append(f"{clip_now:.0%} of token updates are clipped.")
            elif clip_now >= _CLIP_INFO:
                severity = max(severity, Severity.INFO)
                summaries.append(f"Clip fraction is {clip_now:.0%}, on the high side.")

            if clip_now >= _CLIP_WARN:
                prescription += [
                    "Lower the learning rate, or reduce the number of inner PPO epochs per batch. "
                    "Heavy clipping means the policy moved too far before the ratio was measured.",
                    "Check for a stale-rollout problem: with async generation, the sampling policy "
                    "can lag the training policy by enough steps that the importance ratio is "
                    "large before the optimiser does anything at all.",
                ]

        if has_sides:
            _, low = run.finite(PG_CLIPFRAC_LOW)
            _, high = run.finite(PG_CLIPFRAC_HIGH)
            low_now = float(np.median(stats.tail(low, frac=0.25)))
            high_now = float(np.median(stats.tail(high, frac=0.25)))
            metrics["clipfrac_low"] = low_now
            metrics["clipfrac_high"] = high_now
            ratio = low_now / max(high_now, 1e-9)
            metrics["clip_asymmetry"] = ratio
            evidence.append(
                f"one-sided clipping: lower bound {low_now:.2%}, upper bound {high_now:.2%} "
                f"(ratio {ratio:.1f}x)"
            )
            if ratio >= _ASYM_RATIO and low_now >= 0.02:
                severity = max(severity, Severity.WARNING)
                summaries.append(
                    f"Lower-bound clipping dominates by {ratio:.0f}x: the update suppresses tokens "
                    "far more readily than it promotes them, which starves exploration."
                )
                prescription.append(
                    "Decouple the clip bounds and raise `clip_eps_high` to ~0.28 while keeping "
                    "`clip_eps_low` at 0.2 (DAPO clip-higher). This gives low-probability tokens "
                    "room to be reinforced before they are clipped out of existence."
                )
                if run.has(ENTROPY):
                    ent_tr = stats.trend(*run.finite(ENTROPY))
                    if ent_tr.direction == "down":
                        evidence.append(
                            "entropy is falling at the same time, which is consistent with "
                            "asymmetric clipping driving the collapse rather than merely "
                            "co-occurring with it"
                        )

        if severity is Severity.OK:
            return self.finding(
                Severity.OK,
                "Clipping is within the range where the trust region helps rather than dominates.",
                evidence=evidence,
                metrics=metrics,
            )

        return self.finding(
            severity, " ".join(summaries), evidence=evidence, prescription=prescription, metrics=metrics
        )
