"""Turn arbitrary training logs into a canonical :class:`~rldoctor.schema.Run`.

The design constraint is that ``rldoctor`` must be useful on a log someone
already has, with no instrumentation added. So the ingest layer accepts:

* a list of per-step dicts (what every training loop already has in hand),
* JSONL / CSV files,
* a W&B run URI,
* a TensorBoard event directory,

and normalises all of them through :mod:`rldoctor.aliases`.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .. import aliases
from ..schema import CANONICAL_FIELDS, Run, RunConfig

#: Keys that identify the x-axis, most specific first.
_STEP_KEYS = (
    "step",
    "global_step",
    "_step",
    "iteration",
    "iter",
    "training_step",
    "trainer/global_step",
    "epoch",
)

_CONFIG_MAP = {
    "algorithm": ("algorithm", "algo", "adv_estimator", "algorithm.adv_estimator", "loss_type"),
    "group_size": ("num_generations", "group_size", "rollout.n", "n_samples_per_prompt", "num_rollouts"),
    "batch_size": ("train_batch_size", "batch_size", "per_device_train_batch_size"),
    "num_gpus": ("num_gpus", "n_gpus", "world_size", "trainer.n_gpus_per_node"),
    "gpu_type": ("gpu_type", "gpu", "device_name"),
    "max_response_length": ("max_response_length", "max_completion_length", "max_new_tokens"),
    "kl_coef": ("kl_coef", "beta", "kl_loss_coef", "init_kl_coef"),
    "entropy_coef": ("entropy_coef", "entropy_coeff", "entropy_bonus"),
    "clip_eps_low": ("clip_eps_low", "epsilon", "epsilon_low", "clip_ratio_low", "cliprange"),
    "clip_eps_high": ("clip_eps_high", "epsilon_high", "clip_ratio_high"),
    "learning_rate": ("learning_rate", "lr", "actor_lr"),
    "model_name": ("model_name", "model", "model_name_or_path", "model.path"),
}


def run_from_records(
    records: Sequence[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]] = None,
    name: str = "run",
    source: str = "records",
) -> Run:
    """Build a :class:`Run` from per-step metric dicts.

    Records need not all carry the same keys -- sparse series (a held-out eval
    every 50 steps, say) are represented as ``nan`` at the steps where they are
    absent, which keeps every series index-aligned with ``steps``.
    """
    records = [_unwrap(r) for r in records if isinstance(r, Mapping)]
    if not records:
        raise ValueError("no records to ingest")

    steps = _extract_steps(records)
    order = np.argsort(steps, kind="mergesort")
    steps = steps[order]
    records = [records[i] for i in order]

    n = len(records)
    series: Dict[str, np.ndarray] = {}
    components: Dict[str, np.ndarray] = {}
    unmapped: set = set()
    resolved_map: Dict[str, str] = {}
    nonfinite: Dict[str, List[float]] = {}

    for idx, record in enumerate(records):
        for key, value in record.items():
            if key in _STEP_KEYS or key.startswith("_"):
                continue
            number = _as_float(value)
            if number is None:
                continue

            component = aliases.resolve_reward_component(key)
            if component is not None:
                components.setdefault(component, np.full(n, np.nan))[idx] = number
                continue

            canonical = aliases.resolve(key)
            if canonical is None:
                unmapped.add(key)
                continue
            resolved_map[key] = canonical
            if not math.isfinite(number):
                # Logged, and not a number. Distinct from "not logged".
                nonfinite.setdefault(canonical, []).append(float(steps[idx]))
                continue
            target = series.setdefault(canonical, np.full(n, np.nan))
            # First writer wins per step: the curated alias order puts the more
            # trustworthy key first (e.g. verl's task score before the
            # penalty-adjusted reward), so we must not let a later key clobber it.
            if math.isnan(target[idx]):
                target[idx] = number

    run_config = _config_from(config, records)
    # A single-valued canonical series like group_size doubles as config.
    if run_config.group_size is None and "group_size" in series:
        median = np.nanmedian(series["group_size"])
        if np.isfinite(median):
            run_config.group_size = int(median)

    return Run(
        steps=steps,
        series={k: v for k, v in series.items() if k in CANONICAL_FIELDS},
        config=run_config,
        reward_components=components,
        name=name,
        source=source,
        unmapped_keys=sorted(unmapped),
        nonfinite_steps=nonfinite,
    )


#: Keys under which some loggers nest the metric dict. verl's ``FileLogger``
#: (``trainer.logger=['file']``) writes ``{"step": N, "data": {...}}`` -- read
#: naively, every metric is one level too deep and the run comes back empty.
_NESTED_METRIC_KEYS = ("data", "metrics", "scalars", "logs")


def _unwrap(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Lift a nested metric dict to the top level, keeping the step key."""
    for key in _NESTED_METRIC_KEYS:
        nested = record.get(key)
        if isinstance(nested, Mapping):
            flat = {k: v for k, v in record.items() if k != key}
            for inner_key, value in nested.items():
                flat.setdefault(inner_key, value)
            return flat
    return record


def _extract_steps(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    for key in _STEP_KEYS:
        if any(key in r for r in records):
            values = [_as_float(r.get(key)) for r in records]
            if all(v is not None for v in values):
                return np.array(values, dtype=float)
    # No step column: fall back to record order. Trend slopes are then per
    # logged point rather than per training step, which the report notes.
    return np.arange(len(records), dtype=float)


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if not isinstance(value, bool) else None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    # numpy scalars / 0-d arrays / torch tensors with .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (TypeError, ValueError):
            return None
    return None


def _flatten(config: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
            # Also expose the leaf name so `rollout.n` is findable as `n`.
            # setdefault keeps the shallowest definition on collision.
            flat.setdefault(key, value)
    return flat


def _config_from(
    config: Optional[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> RunConfig:
    flat = _flatten(dict(config)) if config else {}
    resolved: Dict[str, Any] = {}
    for attr, candidates in _CONFIG_MAP.items():
        for candidate in candidates:
            if candidate in flat and flat[candidate] is not None:
                resolved[attr] = flat[candidate]
                break

    def as_int(key: str) -> Optional[int]:
        value = _as_float(resolved.get(key))
        return int(value) if value is not None and np.isfinite(value) else None

    def as_float(key: str) -> Optional[float]:
        return _as_float(resolved.get(key))

    algorithm = resolved.get("algorithm")
    if isinstance(algorithm, str):
        algorithm = algorithm.strip().lower()
        # verl spells GRPO as an advantage estimator.
        algorithm = {"grpo_passk": "grpo", "gae": "ppo"}.get(algorithm, algorithm)
    else:
        algorithm = None

    return RunConfig(
        algorithm=algorithm,
        group_size=as_int("group_size"),
        batch_size=as_int("batch_size"),
        num_gpus=as_int("num_gpus"),
        gpu_type=resolved.get("gpu_type") if isinstance(resolved.get("gpu_type"), str) else None,
        max_response_length=as_int("max_response_length"),
        kl_coef=as_float("kl_coef"),
        entropy_coef=as_float("entropy_coef"),
        clip_eps_low=as_float("clip_eps_low"),
        clip_eps_high=as_float("clip_eps_high"),
        learning_rate=as_float("learning_rate"),
        model_name=resolved.get("model_name") if isinstance(resolved.get("model_name"), str) else None,
        raw=dict(flat),
    )


# -- file / URI loading -----------------------------------------------------


def load_jsonl(path: str) -> Run:
    records: List[Dict[str, Any]] = []
    config: Optional[Dict[str, Any]] = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            # A leading `{"config": {...}}` line is a common convention.
            if set(obj) == {"config"} and isinstance(obj["config"], dict):
                config = obj["config"]
                continue
            records.append(obj)
    return run_from_records(records, config=config, name=os.path.basename(path), source=f"jsonl:{path}")


def load_csv(path: str) -> Run:
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        records = [dict(row) for row in csv.DictReader(handle)]
    return run_from_records(records, name=os.path.basename(path), source=f"csv:{path}")


def load_json(path: str) -> Run:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return run_from_records(payload, name=os.path.basename(path), source=f"json:{path}")
    if isinstance(payload, dict):
        # HuggingFace Trainer checkpoints: every `checkpoint-N/trainer_state.json`
        # carries the full metric history. It is the most widely available real
        # training log there is, and it needs no instrumentation at all.
        if isinstance(payload.get("log_history"), list):
            config = {
                k: v
                for k, v in payload.items()
                if k != "log_history" and isinstance(v, (int, float, str, bool))
            }
            return run_from_records(
                payload["log_history"],
                config=config,
                name=os.path.basename(os.path.dirname(os.path.abspath(path))) or "trainer_state",
                source=f"trainer_state:{path}",
            )
        history = payload.get("history") or payload.get("records") or payload.get("log")
        if isinstance(history, list):
            return run_from_records(
                history,
                config=payload.get("config"),
                name=payload.get("name", os.path.basename(path)),
                source=f"json:{path}",
            )
    raise ValueError(f"{path}: expected a list of records or an object with a 'history' list")


def load_run(uri: str, **kwargs: Any) -> Run:
    """Load a run from a path or URI, dispatching on shape.

    Supported: ``*.jsonl``, ``*.json`` (including a HuggingFace
    ``trainer_state.json``), ``*.csv``, a TensorBoard event directory, and
    ``wandb://entity/project/run_id``.
    """
    if uri.startswith("wandb://") or uri.startswith("https://wandb.ai/"):
        from .wandb_source import load_wandb

        return load_wandb(uri, **kwargs)

    if os.path.isdir(uri):
        from .tensorboard_source import load_tensorboard

        return load_tensorboard(uri, **kwargs)

    if not os.path.exists(uri):
        raise FileNotFoundError(f"no such run: {uri}")

    lowered = uri.lower()
    if lowered.endswith(".jsonl") or lowered.endswith(".ndjson"):
        return load_jsonl(uri)
    if lowered.endswith(".csv") or lowered.endswith(".tsv"):
        return load_csv(uri)
    if lowered.endswith(".json"):
        return load_json(uri)
    raise ValueError(
        f"cannot infer format for {uri!r}. Supported: .jsonl, .json, .csv, "
        "a TensorBoard directory, or wandb://entity/project/run_id"
    )


def iter_unmapped(run: Run) -> Iterable[str]:
    return run.unmapped_keys
