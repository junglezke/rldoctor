"""rldoctor -- diagnose why your GRPO/RLVR run is burning GPU-hours without learning.

Quick start::

    from rldoctor import diagnose, load_run

    diagnosis = diagnose(load_run("wandb://team/project/run_id"))
    print(diagnosis.headline)

Or from a plain list of the metric dicts your loop already produces::

    from rldoctor import run_from_records, diagnose

    diagnose(run_from_records(my_log_rows)).print()
"""

from .detectors import Detector, Finding, Severity, register
from .diagnosis import Diagnosis, diagnose
from .ingest import load_csv, load_json, load_jsonl, load_run, run_from_records
from .schema import Run, RunConfig

__version__ = "0.1.0"

__all__ = [
    "Detector",
    "Diagnosis",
    "Finding",
    "Run",
    "RunConfig",
    "Severity",
    "__version__",
    "diagnose",
    "load_csv",
    "load_json",
    "load_jsonl",
    "load_run",
    "register",
    "run_from_records",
]


def _print(self) -> None:  # pragma: no cover - thin delegation
    """Render this diagnosis to stdout. Attached here to keep report imports lazy."""
    from .report import emit
    from .report.terminal import render

    emit(render(self))


Diagnosis.print = _print  # type: ignore[attr-defined]
