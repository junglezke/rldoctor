"""Drop-in callback for TRL / any HuggingFace ``Trainer``.

    from trl import GRPOTrainer
    from rldoctor.integrations.trl import RLDoctorCallback

    trainer = GRPOTrainer(..., callbacks=[RLDoctorCallback()])

That is the whole integration. The callback reads the metrics TRL already logs
-- including ``frac_reward_zero_std``, which is exactly the degenerate-group
rate -- so nothing extra needs instrumenting.

``transformers`` is imported lazily so that ``rldoctor`` stays installable in
an environment that has never heard of it.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..detectors.base import Severity
from ..live import DEFAULT_MIN_STEPS, LiveMonitor


try:  # pragma: no cover - depends on the user's environment
    from transformers import TrainerCallback as _TrainerCallback

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAVE_TRANSFORMERS = False

    class _TrainerCallback:  # type: ignore[no-redef]
        """Stand-in so this module imports without `transformers` installed.

        Subclassing the real base when it exists keeps any `isinstance` check
        inside the Trainer happy; falling back to a stub keeps `rldoctor`
        importable in environments that have never installed transformers.
        """


class RLDoctorCallback(_TrainerCallback):
    """Live rldoctor diagnosis attached to a HuggingFace training loop.

    Args:
        check_every: how many logged steps between diagnoses.
        min_steps: warm-up before any check runs.
        alert_at: minimum severity worth interrupting a human for.
        report_path: if set, an HTML report is written here at the end of training.
        halt_on: if set, training is stopped when a finding at or above this
            severity fires. Off by default -- a monitoring tool should not kill
            your job unless you explicitly ask it to.
    """

    def __init__(
        self,
        check_every: int = 25,
        min_steps: int = DEFAULT_MIN_STEPS,
        alert_at: Severity = Severity.WARNING,
        report_path: Optional[str] = None,
        halt_on: Optional[Severity] = None,
        only: Sequence[str] = (),
        skip: Sequence[str] = (),
    ) -> None:
        self.monitor = LiveMonitor(
            check_every=check_every,
            min_steps=min_steps,
            alert_at=alert_at,
            only=only,
            skip=skip,
            name="trl",
        )
        self.report_path = report_path
        self.halt_on = halt_on
        self._configured = False
        if not HAVE_TRANSFORMERS:
            raise ImportError(
                "RLDoctorCallback needs `transformers`. Install it, or drive "
                "rldoctor.live.LiveMonitor directly from your own training loop."
            )

    # -- TrainerCallback hooks --------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: D102
        if not self._configured:
            self.monitor.config = _config_from_args(args, kwargs.get("model"))
            self._configured = True
        return control

    def on_log(self, args, state, control, logs: Optional[Mapping[str, Any]] = None, **kwargs):
        if not logs:
            return control
        findings = self.monitor.log(logs, step=int(getattr(state, "global_step", 0) or 0))
        if self.halt_on is not None and any(f.severity >= self.halt_on for f in findings):
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.report_path:
            self.monitor.save(self.report_path, fmt="html")
        return control


def _config_from_args(args: Any, model: Any = None) -> dict:
    """Pull the handful of config values the detectors can use for context."""
    keys = (
        "num_generations",
        "per_device_train_batch_size",
        "learning_rate",
        "beta",
        "epsilon",
        "epsilon_high",
        "max_completion_length",
        "entropy_coef",
        "loss_type",
    )
    config = {key: getattr(args, key) for key in keys if hasattr(args, key)}
    config.setdefault("algorithm", "grpo")
    n_gpu = getattr(args, "world_size", None) or getattr(args, "n_gpu", None)
    if n_gpu:
        config["num_gpus"] = int(n_gpu)
    name = getattr(getattr(model, "config", None), "_name_or_path", None)
    if name:
        config["model_name"] = name
    return config
