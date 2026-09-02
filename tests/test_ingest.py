"""Ingest has to be forgiving: it meets logs it did not design."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rldoctor import aliases, load_csv, load_jsonl, load_run, run_from_records
from rldoctor import schema as S


def test_trl_keys_resolve():
    assert aliases.resolve("frac_reward_zero_std") == S.ZERO_VAR_GROUP_FRAC
    assert aliases.resolve("completions/mean_length") == S.RESPONSE_LEN_MEAN
    assert aliases.resolve("completions/clipped_ratio") == S.RESPONSE_LEN_CLIP_RATIO
    assert aliases.resolve("clip_ratio/low_mean") == S.PG_CLIPFRAC_LOW
    assert aliases.resolve("reward") == S.REWARD_MEAN


def test_verl_keys_resolve():
    assert aliases.resolve("actor/entropy") == S.ENTROPY
    assert aliases.resolve("critic/score/mean") == S.REWARD_MEAN
    assert aliases.resolve("response_length/mean") == S.RESPONSE_LEN_MEAN
    assert aliases.resolve("response_length/clip_ratio") == S.RESPONSE_LEN_CLIP_RATIO
    assert aliases.resolve("actor/pg_clipfrac") == S.PG_CLIPFRAC
    assert aliases.resolve("perf/time_per_step") == S.TIME_PER_STEP


def test_unknown_prefix_still_resolves_by_suffix():
    """Framework releases reshuffle prefixes; the suffix pass absorbs that."""
    assert aliases.resolve("some_new_worker/entropy") == S.ENTROPY
    assert aliases.resolve("policy/grad_norm") == S.GRAD_NORM


def test_a_training_split_prefix_is_stripped_but_eval_is_not():
    """Confusing a held-out score with a training reward would break the single
    most important check in the tool."""
    assert aliases.resolve("train/reward") == S.REWARD_MEAN
    assert aliases.resolve("eval/accuracy") == S.EVAL_SCORE
    assert aliases.resolve("val/test_score") == S.EVAL_SCORE
    assert aliases.resolve("eval/reward") == S.EVAL_SCORE


def test_reward_components_are_kept_separate_from_the_total():
    assert aliases.resolve_reward_component("rewards/format/mean") == "format"
    assert aliases.resolve_reward_component("rewards/mean") is None
    assert aliases.resolve_reward_component("reward") is None


def test_sparse_series_stay_aligned_with_steps():
    records = [{"step": i, "reward": 0.5} for i in range(10)]
    records[0]["eval/accuracy"] = 0.4
    records[5]["eval/accuracy"] = 0.6
    run = run_from_records(records)
    assert run.series[S.EVAL_SCORE].shape == run.steps.shape
    steps, values = run.finite(S.EVAL_SCORE)
    assert list(steps) == [0.0, 5.0]
    assert list(values) == [0.4, 0.6]


def test_records_are_sorted_by_step():
    run = run_from_records([{"step": 5, "reward": 1.0}, {"step": 1, "reward": 0.0}])
    assert list(run.steps) == [1.0, 5.0]
    assert list(run.series[S.REWARD_MEAN]) == [0.0, 1.0]


def test_missing_step_column_falls_back_to_record_order():
    run = run_from_records([{"reward": float(i)} for i in range(5)])
    assert list(run.steps) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_unmapped_keys_are_surfaced_not_swallowed():
    run = run_from_records([{"step": 0, "my_bespoke_metric": 1.0, "reward": 0.5}])
    assert "my_bespoke_metric" in run.unmapped_keys


def test_non_numeric_values_are_ignored():
    run = run_from_records([{"step": 0, "reward": 0.5, "note": "restarted here"}])
    assert S.REWARD_MEAN in run.series
    assert "note" not in run.unmapped_keys  # not numeric, so not a candidate metric


def test_string_numbers_are_parsed():
    """CSV gives everything to us as strings."""
    run = run_from_records([{"step": "0", "reward": "0.5"}, {"step": "1", "reward": "0.7"}])
    assert list(run.series[S.REWARD_MEAN]) == [0.5, 0.7]


def test_booleans_are_not_treated_as_metrics():
    run = run_from_records([{"step": 0, "reward": 0.5, "is_resumed": True}])
    assert S.REWARD_MEAN in run.series
    assert not any(np.isfinite(v).all() and k == "is_resumed" for k, v in run.series.items())


def test_nested_config_is_flattened():
    run = run_from_records(
        [{"step": 0, "reward": 0.1}],
        config={"algorithm": {"adv_estimator": "grpo"}, "rollout": {"n": 16}},
    )
    assert run.config.algorithm == "grpo"
    assert run.config.group_size == 16


def test_series_must_stay_aligned():
    with pytest.raises(ValueError, match="stay aligned"):
        S.Run(steps=np.arange(3.0), series={"reward_mean": np.arange(5.0)})


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "log.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"config": {"num_generations": 8}}) + "\n")
        for i in range(20):
            handle.write(json.dumps({"step": i, "reward": i / 20, "entropy": 0.5}) + "\n")
    run = load_jsonl(str(path))
    assert run.n_steps == 20
    assert run.config.group_size == 8
    assert load_run(str(path)).n_steps == 20


def test_csv_roundtrip(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("step,reward\n0,0.1\n1,0.2\n", encoding="utf-8")
    run = load_csv(str(path))
    assert list(run.series[S.REWARD_MEAN]) == [0.1, 0.2]


def test_load_run_rejects_unknown_formats(tmp_path):
    path = tmp_path / "log.parquet"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="cannot infer format"):
        load_run(str(path))


def test_load_run_reports_a_missing_file():
    with pytest.raises(FileNotFoundError):
        load_run("definitely_not_here.jsonl")


def test_wandb_uri_parsing():
    from rldoctor.ingest.wandb_source import parse_uri

    assert parse_uri("wandb://team/proj/abc123") == {
        "entity": "team",
        "project": "proj",
        "run_id": "abc123",
    }
    assert parse_uri("https://wandb.ai/team/proj/runs/abc123")["run_id"] == "abc123"
    with pytest.raises(ValueError):
        parse_uri("wandb://not-enough-parts")
