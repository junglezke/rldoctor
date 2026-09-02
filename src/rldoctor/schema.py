"""Canonical metric schema shared by every ingester and detector.

Different RL frameworks name the same quantity differently (``actor/entropy`` in
verl, ``entropy`` in TRL, ``policy/entropy`` elsewhere).  Detectors are written
against the canonical names defined here, and :mod:`rldoctor.aliases` maps
framework-specific keys onto them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Canonical field names.
#
# Keep this list flat and boring: it is the contract between ingesters and
# detectors, and every entry must be something at least one mainstream
# framework actually logs.
# --------------------------------------------------------------------------

STEP = "step"

# Reward / objective
REWARD_MEAN = "reward_mean"
REWARD_STD = "reward_std"
REWARD_MAX = "reward_max"
REWARD_MIN = "reward_min"
EVAL_SCORE = "eval_score"

# GRPO group statistics
ZERO_VAR_GROUP_FRAC = "zero_var_group_frac"
GROUP_SIZE = "group_size"
ADVANTAGE_MEAN = "advantage_mean"
ADVANTAGE_STD = "advantage_std"

# Policy health
ENTROPY = "entropy"
KL = "kl"
PG_CLIPFRAC = "pg_clipfrac"
PG_CLIPFRAC_LOW = "pg_clipfrac_low"
PG_CLIPFRAC_HIGH = "pg_clipfrac_high"
GRAD_NORM = "grad_norm"
LR = "lr"
POLICY_LOSS = "policy_loss"

# Generation shape
RESPONSE_LEN_MEAN = "response_len_mean"
RESPONSE_LEN_MAX = "response_len_max"
RESPONSE_LEN_CLIP_RATIO = "response_len_clip_ratio"
PROMPT_LEN_MEAN = "prompt_len_mean"

# Critic (PPO only)
VALUE_EXPLAINED_VAR = "value_explained_var"

# Throughput / cost
THROUGHPUT = "throughput"
TIME_PER_STEP = "time_per_step"
NUM_GPUS = "num_gpus"

CANONICAL_FIELDS: List[str] = [
    REWARD_MEAN,
    REWARD_STD,
    REWARD_MAX,
    REWARD_MIN,
    EVAL_SCORE,
    ZERO_VAR_GROUP_FRAC,
    GROUP_SIZE,
    ADVANTAGE_MEAN,
    ADVANTAGE_STD,
    ENTROPY,
    KL,
    PG_CLIPFRAC,
    PG_CLIPFRAC_LOW,
    PG_CLIPFRAC_HIGH,
    GRAD_NORM,
    LR,
    POLICY_LOSS,
    RESPONSE_LEN_MEAN,
    RESPONSE_LEN_MAX,
    RESPONSE_LEN_CLIP_RATIO,
    PROMPT_LEN_MEAN,
    VALUE_EXPLAINED_VAR,
    THROUGHPUT,
    TIME_PER_STEP,
    NUM_GPUS,
]

#: Human-readable descriptions, used by ``rldoctor fields`` and HTML reports.
FIELD_DOCS: Dict[str, str] = {
    REWARD_MEAN: "Mean training reward (post reward-shaping) per step.",
    REWARD_STD: "Std-dev of training reward within the batch.",
    EVAL_SCORE: "Held-out evaluation score. The ground truth reward cannot hack.",
    ZERO_VAR_GROUP_FRAC: "Fraction of GRPO groups whose rollouts all got the same reward.",
    GROUP_SIZE: "Rollouts per prompt (GRPO 'num_generations' / 'rollout.n').",
    ENTROPY: "Mean token-level policy entropy. The exploration budget.",
    KL: "KL(policy || reference). Drift away from the SFT init.",
    PG_CLIPFRAC: "Fraction of tokens whose importance ratio hit the PPO clip bound.",
    PG_CLIPFRAC_LOW: "Clip fraction on the lower bound (1 - eps_low).",
    PG_CLIPFRAC_HIGH: "Clip fraction on the upper bound (1 + eps_high).",
    GRAD_NORM: "Global gradient norm before clipping.",
    RESPONSE_LEN_MEAN: "Mean generated response length in tokens.",
    RESPONSE_LEN_CLIP_RATIO: "Fraction of responses truncated at max_response_length.",
    VALUE_EXPLAINED_VAR: "Explained variance of the value head (PPO only).",
    THROUGHPUT: "Tokens per second across the cluster.",
    TIME_PER_STEP: "Wall-clock seconds per training step.",
}


@dataclass
class RunConfig:
    """Configuration facts about a run that detectors can use for context.

    Every field is optional: ``rldoctor`` must produce a useful report from a
    bare metric log with no config attached.
    """

    algorithm: Optional[str] = None  # "grpo" | "ppo" | "rloo" | ...
    group_size: Optional[int] = None
    batch_size: Optional[int] = None
    num_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    max_response_length: Optional[int] = None
    kl_coef: Optional[float] = None
    entropy_coef: Optional[float] = None
    clip_eps_low: Optional[float] = None
    clip_eps_high: Optional[float] = None
    learning_rate: Optional[float] = None
    model_name: Optional[str] = None
    raw: Dict[str, object] = field(default_factory=dict)


@dataclass
class Run:
    """A training run reduced to canonical time series.

    ``series`` maps a canonical field name to a float array aligned with
    ``steps``.  Missing observations are ``nan`` rather than dropped, so that
    every series stays index-aligned with ``steps``.
    """

    steps: np.ndarray
    series: Dict[str, np.ndarray]
    config: RunConfig = field(default_factory=RunConfig)
    reward_components: Dict[str, np.ndarray] = field(default_factory=dict)
    name: str = "run"
    source: str = "unknown"
    #: Framework keys that were present in the log but not mapped to a
    #: canonical field. Surfaced by ``rldoctor diagnose --verbose`` so users can
    #: tell us about naming drift instead of silently getting a worse report.
    unmapped_keys: List[str] = field(default_factory=list)
    #: Steps at which a field was logged with a non-finite value, per field.
    #:
    #: This has to be tracked separately because `nan` in a series is ambiguous:
    #: it means "not logged at this step" far more often than it means "the
    #: model produced NaN". Real runs log eval metrics on a different cadence
    #: from training metrics, so every such run has thousands of padding nans,
    #: and a detector that counts them reports numerical death on a healthy job.
    nonfinite_steps: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.steps = np.asarray(self.steps, dtype=float)
        for key, value in list(self.series.items()):
            arr = np.asarray(value, dtype=float)
            if arr.shape != self.steps.shape:
                raise ValueError(
                    f"series {key!r} has shape {arr.shape}, expected {self.steps.shape} "
                    "to stay aligned with `steps`"
                )
            self.series[key] = arr

    # -- accessors ---------------------------------------------------------

    def has(self, field_name: str, min_points: int = 2) -> bool:
        """True when ``field_name`` exists with at least ``min_points`` finite values."""
        arr = self.series.get(field_name)
        if arr is None:
            return False
        return int(np.isfinite(arr).sum()) >= min_points

    def get(self, field_name: str) -> Optional[np.ndarray]:
        return self.series.get(field_name)

    def finite(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(steps, values)`` restricted to finite observations."""
        arr = self.series.get(field_name)
        if arr is None:
            return np.array([]), np.array([])
        mask = np.isfinite(arr)
        return self.steps[mask], arr[mask]

    @property
    def n_steps(self) -> int:
        return int(self.steps.size)

    @property
    def available_fields(self) -> List[str]:
        return sorted(k for k in self.series if self.has(k, min_points=1))

    def summary(self) -> str:
        span = f"{self.steps[0]:.0f}-{self.steps[-1]:.0f}" if self.n_steps else "empty"
        return (
            f"{self.name}: {self.n_steps} steps ({span}), "
            f"{len(self.available_fields)} canonical fields, source={self.source}"
        )
