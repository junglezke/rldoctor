"""Load a run from TensorBoard event files.

Point ``rldoctor`` at the directory containing ``events.out.tfevents.*``:

    rldoctor diagnose ./outputs/my-grpo-run/tensorboard
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Any, Dict

from .base import run_from_records


def load_tensorboard(path: str, **_: Any):
    """Read scalar summaries from a TensorBoard log directory."""
    if not _has_event_files(path):
        raise ValueError(
            f"{path} contains no TensorBoard event files "
            "(looked for **/events.out.tfevents.*)"
        )

    scalars = _read_scalars(path)
    if not scalars:
        raise ValueError(f"{path}: no scalar summaries found in the event files")

    # Pivot (tag, step, value) into one record per step.
    by_step: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for tag, points in scalars.items():
        for step, value in points:
            by_step[int(step)][tag] = float(value)

    records = [{"step": step, **values} for step, values in sorted(by_step.items())]
    return run_from_records(records, name=os.path.basename(os.path.abspath(path)), source=f"tensorboard:{path}")


def _has_event_files(path: str) -> bool:
    return bool(glob.glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True))


def _read_scalars(path: str) -> Dict[str, list]:
    """Prefer tbparse, fall back to TensorBoard's own accumulator."""
    try:
        from tbparse import SummaryReader
    except ImportError:
        return _read_scalars_via_accumulator(path)

    reader = SummaryReader(path, pivot=False, extra_columns={"dir_name"})
    frame = reader.scalars
    if frame is None or len(frame) == 0:
        return {}
    out: Dict[str, list] = defaultdict(list)
    for tag, step, value in zip(frame["tag"], frame["step"], frame["value"], strict=True):
        out[str(tag)].append((int(step), float(value)))
    return dict(out)


def _read_scalars_via_accumulator(path: str) -> Dict[str, list]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "reading TensorBoard logs needs tbparse or tensorboard: "
            "pip install 'rldoctor[tensorboard]'"
        ) from exc

    out: Dict[str, list] = defaultdict(list)
    for event_file in sorted(glob.glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True)):
        accumulator = EventAccumulator(event_file, size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                out[tag].append((int(event.step), float(event.value)))
    return dict(out)
