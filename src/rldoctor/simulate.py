"""Synthetic RLVR runs with known, injected pathologies.

This module exists for three reasons:

1. **Tests need ground truth.** A detector is only trustworthy if we can show it
   fires on the failure it claims to detect and stays quiet on a healthy run.
   Every scenario here has a known label, so the test suite asserts both.
2. **The demo needs no GPUs.** ``rldoctor demo`` generates a failing run and
   diagnoses it in under a second, so anyone can evaluate the tool before
   deciding whether to point it at a real job.
3. **False positives are the real risk.** A diagnostic that cries wolf gets
   uninstalled. The ``healthy`` scenario is the regression test that matters
   most.

The generator is a phenomenological model of curve *shape*, not a simulator of
LLM training. It reproduces the signatures practitioners actually see; it makes
no claim to reproduce the underlying dynamics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

SCENARIOS = (
    "healthy",
    "entropy_collapse",
    "saturated_groups",
    "too_hard",
    "reward_hacking",
    "length_hacking",
    "truncation",
    "kl_blowup",
    "clip_saturation",
    "grad_spikes",
    "plateau",
    "format_domination",
)

#: What each scenario is expected to trigger, used by the test suite and by
#: ``rldoctor selftest``.
EXPECTED_DETECTIONS: Dict[str, Tuple[str, ...]] = {
    "healthy": (),
    "entropy_collapse": ("entropy_collapse",),
    "saturated_groups": ("advantage_collapse",),
    "too_hard": ("advantage_collapse",),
    "reward_hacking": ("reward_hacking",),
    "length_hacking": ("length_pathology",),
    "truncation": ("length_pathology",),
    "kl_blowup": ("kl_drift",),
    "clip_saturation": ("clip_saturation",),
    "grad_spikes": ("gradient_pathology",),
    "plateau": ("plateau",),
    "format_domination": ("reward_composition",),
}


def _logistic(t: np.ndarray, ceiling: float, midpoint: float, rate: float, floor: float) -> np.ndarray:
    return floor + (ceiling - floor) / (1.0 + np.exp(-rate * (t - midpoint)))


def simulate(
    scenario: str = "healthy",
    n_steps: int = 400,
    seed: int = 0,
    eval_every: int = 25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate ``(records, config)`` for one scenario.

    Returns the same shape as a real training log, so it flows through the
    normal ingest path rather than a test-only shortcut.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")

    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float)
    frac = t / max(n_steps - 1, 1)

    group_size = 8
    config: Dict[str, Any] = {
        "algorithm": "grpo",
        "num_generations": group_size,
        "train_batch_size": 512,
        "num_gpus": 8,
        "gpu_type": "H100",
        "max_response_length": 2048,
        "learning_rate": 1e-6,
        "beta": 0.001,
        "epsilon": 0.2,
        "epsilon_high": 0.2,
        "model_name": "Qwen2.5-7B-Instruct",
    }

    # -- healthy baseline ---------------------------------------------------
    # A genuinely healthy run: reward still climbing at the end, entropy decaying
    # but with budget left, response length growing modestly because the policy
    # is learning to reason for longer rather than to pad.
    pass_rate = _logistic(t, ceiling=0.66, midpoint=n_steps * 0.55, rate=6.0 / n_steps, floor=0.18)
    entropy = 0.62 * np.exp(-1.1 * frac) + 0.22
    kl = 0.02 + 0.35 * frac**1.3
    length = 380 + 130 * frac
    grad = 0.55 * np.exp(-0.6 * frac)
    clip_low = 0.035 + 0.02 * frac
    clip_high = 0.030 + 0.02 * frac
    truncation = np.full(n_steps, 0.004)
    eval_gap = 0.04  # honest generalisation gap
    correctness_scale = 1.0
    format_scale = 1.0
    time_per_step = 41.0

    # -- pathology injection ------------------------------------------------
    if scenario == "entropy_collapse":
        entropy = 0.62 * np.exp(-6.5 * frac) + 0.006
        pass_rate = _logistic(t, ceiling=0.45, midpoint=n_steps * 0.2, rate=14.0 / n_steps, floor=0.18)
        clip_low = 0.06 + 0.10 * frac
        clip_high = 0.02 + 0.005 * frac

    elif scenario == "saturated_groups":
        pass_rate = _logistic(t, ceiling=0.965, midpoint=n_steps * 0.25, rate=12.0 / n_steps, floor=0.30)
        entropy = 0.62 * np.exp(-2.6 * frac) + 0.10

    elif scenario == "too_hard":
        pass_rate = np.full(n_steps, 0.035) + rng.normal(0, 0.008, n_steps)
        entropy = 0.62 * np.exp(-0.4 * frac) + 0.20
        grad = 0.08 * np.exp(-2.5 * frac) + 0.002

    elif scenario == "reward_hacking":
        pass_rate = _logistic(t, ceiling=0.93, midpoint=n_steps * 0.4, rate=9.0 / n_steps, floor=0.20)
        # Held-out score peaks early then degrades as the exploit takes over.
        eval_gap = None  # handled explicitly below

    elif scenario == "length_hacking":
        length = 380 * (1.0 + 2.6 * frac**1.4)
        pass_rate = _logistic(t, ceiling=0.88, midpoint=n_steps * 0.4, rate=9.0 / n_steps, floor=0.20)
        eval_gap = None

    elif scenario == "truncation":
        length = 380 + 1500 * frac
        truncation = np.clip(0.005 + 0.42 * frac**2, 0, 0.6)

    elif scenario == "kl_blowup":
        kl = 0.02 + 4.5 * frac**2.4
        pass_rate = _logistic(t, ceiling=0.48, midpoint=n_steps * 0.25, rate=12.0 / n_steps, floor=0.18)

    elif scenario == "clip_saturation":
        clip_low = 0.12 + 0.42 * frac
        clip_high = 0.03 + 0.02 * frac
        entropy = 0.62 * np.exp(-4.5 * frac) + 0.02

    elif scenario == "grad_spikes":
        grad = 0.55 * np.exp(-0.6 * frac)
        for idx in rng.choice(np.arange(int(n_steps * 0.1), n_steps), size=7, replace=False):
            grad[int(idx)] *= rng.uniform(40, 160)

    elif scenario == "plateau":
        pass_rate = _logistic(t, ceiling=0.55, midpoint=n_steps * 0.12, rate=25.0 / n_steps, floor=0.18)

    elif scenario == "format_domination":
        correctness_scale = 0.05
        format_scale = 1.0

    pass_rate = np.clip(pass_rate, 0.001, 0.999)

    # -- assemble records ---------------------------------------------------
    records: List[Dict[str, Any]] = []
    for i in range(n_steps):
        p = float(np.clip(pass_rate[i] + rng.normal(0, 0.018), 0.0, 1.0))
        zero_var = float(np.clip(p**group_size + (1 - p) ** group_size + rng.normal(0, 0.012), 0, 1))

        correctness = correctness_scale * p
        # Format reward saturates fast: high mean, near-zero variance late.
        format_reward = format_scale * float(np.clip(_logistic(
            np.array([t[i]]), ceiling=0.98, midpoint=n_steps * 0.06, rate=40.0 / n_steps, floor=0.35
        )[0] + rng.normal(0, 0.06 if scenario == "format_domination" else 0.01), 0, 1))

        record: Dict[str, Any] = {
            "step": i,
            "reward": round(correctness + 0.1 * format_reward, 6),
            "reward_std": round(float(np.sqrt(max(p * (1 - p), 1e-6))) + rng.normal(0, 0.01), 6),
            "frac_reward_zero_std": round(zero_var, 6),
            "entropy": round(float(max(entropy[i] + rng.normal(0, 0.006), 1e-4)), 6),
            "kl": round(float(max(kl[i] + rng.normal(0, 0.004), 0.0)), 6),
            "clip_ratio/low_mean": round(float(max(clip_low[i] + rng.normal(0, 0.004), 0)), 6),
            "clip_ratio/high_mean": round(float(max(clip_high[i] + rng.normal(0, 0.004), 0)), 6),
            "clip_ratio/region_mean": round(
                float(max(clip_low[i] + clip_high[i] + rng.normal(0, 0.005), 0)), 6
            ),
            "grad_norm": round(float(max(grad[i] * float(np.exp(rng.normal(0, 0.12))), 1e-6)), 6),
            "completions/mean_length": round(float(max(length[i] + rng.normal(0, 12), 1)), 2),
            "completions/clipped_ratio": round(float(np.clip(truncation[i] + rng.normal(0, 0.004), 0, 1)), 6),
            "learning_rate": 1e-6,
            "step_time": round(time_per_step + float(rng.normal(0, 1.5)), 3),
            # Tokens/s across the cluster: batch x group x response length, over
            # step time. Logged by verl as `perf/throughput`.
            "perf/throughput": round(
                512 * group_size * float(length[i]) / max(time_per_step, 1e-6)
                * float(np.exp(rng.normal(0, 0.03))),
                1,
            ),
            "rewards/correctness/mean": round(correctness + rng.normal(0, 0.01), 6),
            "rewards/format/mean": round(format_reward, 6),
        }

        if i % eval_every == 0 or i == n_steps - 1:
            record["eval/accuracy"] = round(_eval_score(scenario, p, frac[i], eval_gap, rng), 6)

        records.append(record)

    return records, config


def _eval_score(scenario: str, p: float, frac: float, eval_gap, rng) -> float:
    """Held-out score, which is where hacking becomes visible."""
    noise = float(rng.normal(0, 0.012))
    if scenario == "reward_hacking":
        # Rises with the policy for the first third, then decays as the exploit
        # displaces genuine capability.
        genuine = 0.20 + 0.22 * min(frac / 0.33, 1.0)
        decay = 0.16 * max(frac - 0.33, 0.0) / 0.67
        return float(np.clip(genuine - decay + noise, 0.0, 1.0))
    if scenario == "length_hacking":
        return float(np.clip(0.21 + 0.03 * frac + noise, 0.0, 1.0))
    if scenario == "plateau":
        return float(np.clip(min(p, 0.52) - 0.04 + noise, 0.0, 1.0))
    gap = 0.04 if eval_gap is None else eval_gap
    return float(np.clip(p - gap + noise, 0.0, 1.0))


def simulate_run(scenario: str = "healthy", n_steps: int = 400, seed: int = 0, **kwargs):
    """Convenience wrapper returning a ready :class:`~rldoctor.schema.Run`."""
    from .ingest.base import run_from_records

    records, config = simulate(scenario, n_steps=n_steps, seed=seed, **kwargs)
    return run_from_records(records, config=config, name=f"sim:{scenario}", source="simulated")
