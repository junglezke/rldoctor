"""Map framework-specific metric keys onto :mod:`rldoctor.schema` canonical names.

Design note
-----------
Framework metric names drift between releases, so this module never assumes an
exact key.  Resolution runs in three passes, most-trusted first:

1. **Exact alias match** against the curated table below.
2. **Suffix match** -- ``anything/entropy`` resolves to ``entropy``, which
   survives most prefix reshuffles (``actor/`` -> ``policy/`` and friends).
3. **Token match** -- normalised token-set equality, which catches
   ``response_length_mean`` vs ``response_length/mean``.

Anything still unresolved is recorded on ``Run.unmapped_keys`` instead of being
dropped silently, so a user can see what we missed and open an issue.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import schema as S

#: Curated aliases, canonical name -> framework keys observed in the wild.
#:
#: Sources: TRL ``GRPOTrainer`` logged-metric table, verl
#: ``verl/trainer/ppo/metric_utils.py`` and actor workers, OpenRLHF PPO trainer
#: logging, and common hand-rolled loops.
ALIASES: Dict[str, List[str]] = {
    S.REWARD_MEAN: [
        "reward",  # TRL
        "critic/score/mean",  # verl (pre-penalty task score)
        "critic/rewards/mean",  # verl (post-penalty)
        "train/reward",
        "reward/mean",
        "rewards/mean",
        "episode_reward_mean",
        "objective/rlhf_reward",  # TRL PPO
    ],
    S.REWARD_STD: [
        "reward_std",  # TRL
        "reward/std",
        "rewards/std",
        "critic/score/std",
    ],
    S.REWARD_MAX: ["critic/score/max", "reward/max", "reward_max"],
    S.REWARD_MIN: ["critic/score/min", "reward/min", "reward_min"],
    S.EVAL_SCORE: [
        "eval_score",
        "val/test_score",  # verl validation
        "val-core/acc",
        "eval/reward",
        "eval/accuracy",
        "eval/score",
        "test/accuracy",
        "validation/score",
    ],
    S.ZERO_VAR_GROUP_FRAC: [
        "frac_reward_zero_std",  # TRL -- exactly this quantity
        "zero_var_group_frac",
        "grpo/zero_std_frac",
        "actor/zero_advantage_frac",
    ],
    S.GROUP_SIZE: ["group_size", "num_generations", "rollout/n", "n_samples_per_prompt"],
    S.ADVANTAGE_MEAN: ["critic/advantages/mean", "advantage/mean", "advantages_mean"],
    S.ADVANTAGE_STD: ["critic/advantages/std", "advantage/std", "advantages_std"],
    S.ENTROPY: [
        "entropy",  # TRL
        "actor/entropy",  # verl
        "actor/entropy_loss",
        "policy/entropy",
        "policy_entropy",
        "train/entropy",
    ],
    S.KL: [
        "kl",  # TRL
        "actor/kl_loss",  # verl
        "actor/ppo_kl",
        "critic/kl",
        "policy/approxkl",
        "objective/kl",
        "train/kl",
    ],
    S.PG_CLIPFRAC: [
        "clip_ratio/region_mean",  # TRL
        "clip_ratio",
        "actor/pg_clipfrac",  # verl
        "policy/clipfrac",
        "policy/clipfrac_avg",
        "clipfrac",
    ],
    S.PG_CLIPFRAC_LOW: [
        "clip_ratio/low_mean",  # TRL
        "actor/pg_clipfrac_lower",  # verl
    ],
    S.PG_CLIPFRAC_HIGH: ["clip_ratio/high_mean", "actor/pg_clipfrac_higher"],
    S.GRAD_NORM: ["grad_norm", "actor/grad_norm", "train/grad_norm", "policy/grad_norm"],
    S.LR: ["learning_rate", "lr", "actor/lr", "actor_lr", "train/learning_rate"],
    S.POLICY_LOSS: ["policy_loss", "actor/pg_loss", "loss", "train/policy_loss"],
    S.RESPONSE_LEN_MEAN: [
        "completions/mean_length",  # TRL
        "response_length/mean",  # verl
        "response_length",
        "completion_length",
        "train/response_length",
    ],
    S.RESPONSE_LEN_MAX: ["completions/max_length", "response_length/max"],
    S.RESPONSE_LEN_CLIP_RATIO: [
        "completions/clipped_ratio",  # TRL
        "response_length/clip_ratio",  # verl
        "truncation_rate",
    ],
    S.PROMPT_LEN_MEAN: ["prompt_length/mean", "prompt_length", "completions/mean_prompt_length"],
    S.VALUE_EXPLAINED_VAR: ["critic/vf_explained_var", "value/explained_var", "explained_variance"],
    S.THROUGHPUT: ["perf/throughput", "throughput", "tokens_per_second", "perf/tokens_per_sec"],
    S.TIME_PER_STEP: ["perf/time_per_step", "step_time", "time/step", "timing_s/step"],
}

#: Per-reward-function series look like ``rewards/<name>/mean`` in TRL.  We keep
#: them separately so the reward-composition detector can reason about them.
REWARD_COMPONENT_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"^rewards?/(?P<name>[^/]+)/mean$"),
    re.compile(r"^reward/(?P<name>[^/]+)$"),
    re.compile(r"^critic/score/(?P<name>[^/]+)/mean$"),
)

#: Keys that look like reward components but are aggregates, not components.
_COMPONENT_BLOCKLIST = {"mean", "std", "max", "min", "total", "sum", "overall"}

_EXACT: Dict[str, str] = {}
_SUFFIX: Dict[str, str] = {}
for _canon, _keys in ALIASES.items():
    for _k in _keys:
        _EXACT.setdefault(_k.lower(), _canon)
        _SUFFIX.setdefault(_k.lower().rsplit("/", 1)[-1], _canon)


def _tokens(key: str) -> Tuple[str, ...]:
    return tuple(t for t in re.split(r"[^a-z0-9]+", key.lower()) if t)


_TOKENSET: Dict[frozenset, str] = {}
for _canon, _keys in ALIASES.items():
    for _k in _keys:
        _TOKENSET.setdefault(frozenset(_tokens(_k)), _canon)
    _TOKENSET.setdefault(frozenset(_tokens(_canon)), _canon)


def _strip_split_prefix(key: str) -> str:
    """Drop a leading ``train/``/``training/`` split prefix.

    We deliberately do *not* strip ``eval/``, ``val/`` or ``test/`` -- for this
    tool the difference between a training reward and a held-out score is the
    entire point.
    """
    lowered = key.lower()
    for prefix in ("train/", "training/"):
        if lowered.startswith(prefix):
            return lowered[len(prefix) :]
    return lowered


def resolve(key: str) -> Optional[str]:
    """Resolve one framework metric key to a canonical field, or ``None``."""
    lowered = key.lower().strip()
    if not lowered:
        return None

    if lowered in _EXACT:
        return _EXACT[lowered]

    stripped = _strip_split_prefix(lowered)
    if stripped in _EXACT:
        return _EXACT[stripped]

    # Never let a held-out metric masquerade as a training metric.
    is_eval = any(stripped.startswith(p) for p in ("eval/", "val/", "test/", "validation/"))
    if is_eval:
        tail = stripped.split("/", 1)[1] if "/" in stripped else stripped
        if any(w in tail for w in ("acc", "score", "reward", "pass", "solve")):
            return S.EVAL_SCORE
        return None

    tail = stripped.rsplit("/", 1)[-1]
    if tail in _SUFFIX:
        return _SUFFIX[tail]

    token_key = frozenset(_tokens(stripped))
    if token_key in _TOKENSET:
        return _TOKENSET[token_key]

    return None


def resolve_reward_component(key: str) -> Optional[str]:
    """Return the reward-function name if ``key`` is a per-component reward series."""
    lowered = key.lower().strip()
    for pattern in REWARD_COMPONENT_PATTERNS:
        match = pattern.match(lowered)
        if match:
            name = match.group("name")
            if name not in _COMPONENT_BLOCKLIST:
                return name
    return None


def describe_coverage(resolved: Dict[str, str]) -> str:
    """One-line summary of which canonical fields a log provided."""
    found = sorted(set(resolved.values()))
    return f"{len(found)}/{len(S.CANONICAL_FIELDS)} canonical fields resolved: {', '.join(found)}"
