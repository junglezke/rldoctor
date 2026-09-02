"""Regressions from running against a real public GRPO training log.

Every test here corresponds to a bug that the simulated corpus did not catch and
a real ``trainer_state.json`` did. The fixtures reproduce the *shape* of that log
rather than shipping someone else's data.

The bug that motivated this file was the expensive kind: a false positive.
``rldoctor`` reported "4 non-finite gradient norms -- training is numerically
dead from step 305" on a run whose gradient norms were entirely finite. The
run logged eval metrics on a different cadence from training metrics, so the
aligned series was padded with `nan` at every eval-only row, and the detector
counted the padding. Every real run with a held-out eval has that shape.
"""

from __future__ import annotations

import json

import pytest

from rldoctor import Severity, aliases, diagnose, load_run, run_from_records
from rldoctor import schema as S


def _trl_style_records(n: int = 400, eval_every: int = 50, nan_at=()):
    """Training rows plus separate eval-only rows, the way TRL actually logs."""
    records = []
    for step in range(1, n + 1):
        grad = float("nan") if step in nan_at else 0.03 + 0.00001 * step
        records.append(
            {
                "step": step,
                "epoch": step / n,
                "loss": 0.01,
                "grad_norm": grad,
                "learning_rate": 1e-6,
                "reward": 2.0 + step * 0.001,
                "reward_std": 0.4,
                "frac_reward_zero_std": 0.1,
                "entropy": 0.5,
                "completions/mean_length": 1200.0,
                "completions/clipped_ratio": 0.01,
                "clip_ratio/region_mean": 0.02,
                "rewards/accuracy_reward/mean": 0.6,
                "rewards/format_reward/mean": 0.95,
            }
        )
        if step % eval_every == 0:
            # An eval-only row: no grad_norm, no training reward.
            records.append(
                {
                    "step": step,
                    "eval_loss": 0.02,
                    "eval_reward": 1.9,
                    "eval_entropy": 0.51,
                    "eval_completions/mean_length": 1150.0,
                    "eval_completions/clipped_ratio": 0.02,
                    "eval_frac_reward_zero_std": 0.12,
                    "eval_rewards/accuracy_reward/mean": 0.58,
                }
            )
    return records


# -- the false positive -----------------------------------------------------


def test_sparse_logging_is_not_reported_as_numerical_death():
    """`nan` padding means 'not logged here', not 'the model produced NaN'."""
    run = run_from_records(_trl_style_records())
    finding = diagnose(run).by_name("gradient_pathology")
    assert finding.severity <= Severity.INFO, finding.summary
    assert "non-finite" not in finding.summary


def test_a_genuinely_logged_nan_still_fires():
    """The fix must not buy quiet by going blind."""
    run = run_from_records(_trl_style_records(nan_at={305, 306}))
    finding = diagnose(run).by_name("gradient_pathology")
    assert finding.severity is Severity.CRITICAL
    assert "non-finite" in finding.summary
    assert "305" in finding.summary


def test_nonfinite_steps_records_only_logged_values():
    run = run_from_records(_trl_style_records(nan_at={100}))
    assert run.nonfinite_steps.get(S.GRAD_NORM) == [100.0]
    plain = run_from_records(_trl_style_records())
    assert not plain.nonfinite_steps


# -- eval-prefixed keys must not contaminate the training series ------------


def test_trl_underscore_eval_prefix_is_recognised():
    """TRL writes `eval_reward`; verl writes `val/test_score`. Handling only one
    separator merges the other's metrics into the training series."""
    assert aliases.resolve("eval_reward") == S.EVAL_SCORE
    assert aliases.resolve("eval_rewards/accuracy_reward/mean") == S.EVAL_SCORE
    assert aliases.resolve("val/test_score") == S.EVAL_SCORE


@pytest.mark.parametrize(
    "key",
    [
        "eval_entropy",
        "eval_loss",
        "eval_completions/mean_length",
        "eval_completions/clipped_ratio",
        "eval_frac_reward_zero_std",
        "eval_clip_ratio/region_mean",
    ],
)
def test_eval_copies_of_training_metrics_are_dropped(key):
    """An eval-split entropy is not the training entropy, and mixing them
    silently corrupts every trend the detectors compute."""
    assert aliases.resolve(key) is None


def test_generic_tails_do_not_match():
    """`mean` as a suffix is meaningless: `rewards/x/mean` and
    `response_length/mean` share it. Matching on it mapped a held-out accuracy
    series onto the training reward."""
    assert aliases.resolve("some_unknown_thing/mean") is None
    assert aliases.resolve("whatever/std") is None
    # ...while a specific tail still resolves.
    assert aliases.resolve("some_new_worker/entropy") == S.ENTROPY


def test_training_and_eval_series_stay_separate_end_to_end():
    run = run_from_records(_trl_style_records())
    _, lengths = run.finite(S.RESPONSE_LEN_MEAN)
    assert set(lengths) == {1200.0}, "eval-split lengths leaked into the training series"
    _, entropy = run.finite(S.ENTROPY)
    assert set(entropy) == {0.5}
    assert run.has(S.EVAL_SCORE)


# -- trainer_state.json ingestion -------------------------------------------


def test_trainer_state_json_is_read_directly(tmp_path):
    """Every HuggingFace checkpoint ships one, which makes it the most widely
    available real training log there is."""
    state = {
        "global_step": 400,
        "max_steps": 400,
        "num_train_epochs": 1,
        "log_history": _trl_style_records(120, eval_every=40),
    }
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    run = load_run(str(path))
    assert run.n_steps > 100
    assert run.has(S.REWARD_MEAN) and run.has(S.ENTROPY) and run.has(S.EVAL_SCORE)
    assert set(run.reward_components) == {"accuracy_reward", "format_reward"}
    assert run.source.startswith("trainer_state:")


def test_truncation_is_not_counted_as_wasted_rollout_compute():
    """A truncated rollout's advantage is distorted, not zero. Folding it into
    the waste estimate produced a '99% of compute wasted' claim we could not
    defend."""
    records = _trl_style_records()
    for record in records:
        if "completions/clipped_ratio" in record:
            record["completions/clipped_ratio"] = 1.0
    result = diagnose(run_from_records(records))
    assert result.by_name("length_pathology").severity is Severity.CRITICAL
    assert result.cost.wasted_fraction == 0.0
    assert result.cost.headline() is None


# -- gradient shapes only real runs produce ---------------------------------


def _with_grad(values):
    return [
        {"step": i + 1, "grad_norm": float(v), "reward": 1.0, "entropy": 0.5}
        for i, v in enumerate(values)
    ]


def test_a_tightly_clustered_series_has_no_spikes():
    """8 robust sigmas is reachable at 2x the median when the MAD is tiny, and
    2x the median is not a spike by any useful definition. A real 80-step run
    was flagged for six 'spikes' whose largest value was twice the typical one.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    values = list(0.11 + rng.normal(0, 0.002, 80))
    values[10] = values[40] = values[70] = 0.21  # 2x the median
    finding = diagnose(run_from_records(_with_grad(values))).by_name("gradient_pathology")
    assert finding.metrics.get("n_spikes", 0) == 0
    assert finding.severity is Severity.OK


def test_a_real_spike_still_fires():
    import numpy as np

    rng = np.random.default_rng(1)
    values = list(0.5 + rng.normal(0, 0.02, 200))
    for idx in (50, 100, 150):
        values[idx] = 12.0  # 24x the median, as seen on a real run
    finding = diagnose(run_from_records(_with_grad(values))).by_name("gradient_pathology")
    assert finding.metrics["n_spikes"] == 3
    assert finding.severity >= Severity.WARNING


def test_exactly_zero_updates_are_reported_as_dead_steps():
    """88% of steps on one real public run had a gradient norm of exactly zero,
    rising from 40% to 93% across training. That is not a decay -- those steps
    did not move the policy at all, and the rollouts were still paid for."""
    values = [0.0 if i % 10 else 13.4 for i in range(500)]
    finding = diagnose(run_from_records(_with_grad(values))).by_name("gradient_pathology")
    assert finding.severity is Severity.CRITICAL
    assert "exactly zero gradient" in finding.summary
    assert finding.metrics["zero_grad_step_frac"] > 0.85
    # These are genuinely wasted rollouts, unlike truncated ones.
    assert finding.wasted_fraction and finding.wasted_fraction > 0.85


def test_occasional_zeros_are_not_alarming():
    values = [0.0 if i % 50 == 0 else 0.5 for i in range(500)]
    finding = diagnose(run_from_records(_with_grad(values))).by_name("gradient_pathology")
    assert finding.severity is Severity.OK
