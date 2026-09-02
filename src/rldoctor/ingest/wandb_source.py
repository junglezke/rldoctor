"""Load a run straight from Weights & Biases.

This is the zero-instrumentation path: point ``rldoctor`` at a run you already
logged and get a diagnosis without touching your training code.

    rldoctor diagnose wandb://my-team/my-project/3xk91abc
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import run_from_records

_URI = re.compile(
    r"^(?:wandb://|https?://wandb\.ai/)(?P<entity>[^/]+)/(?P<project>[^/]+)/"
    r"(?:runs/)?(?P<run_id>[^/?#]+)"
)


def parse_uri(uri: str) -> Dict[str, str]:
    match = _URI.match(uri.strip())
    if not match:
        raise ValueError(
            f"cannot parse {uri!r}. Expected wandb://entity/project/run_id or a wandb.ai run URL."
        )
    return match.groupdict()


def load_wandb(uri: str, samples: int = 10_000, api_key: Optional[str] = None):
    """Fetch a W&B run's history and config as a :class:`~rldoctor.schema.Run`."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "reading W&B runs needs the wandb client: pip install 'rldoctor[wandb]'"
        ) from exc

    parts = parse_uri(uri)
    api = wandb.Api(api_key=api_key) if api_key else wandb.Api()
    run = api.run(f"{parts['entity']}/{parts['project']}/{parts['run_id']}")

    records: List[Dict[str, Any]] = []
    # scan_history streams every logged row; samples= would silently subsample
    # and quietly change every slope we compute.
    for row in run.scan_history(page_size=1000):
        if isinstance(row, dict):
            records.append({k: v for k, v in row.items() if v is not None})
        if len(records) >= samples:
            break

    if not records:
        raise ValueError(f"W&B run {uri} has no logged history")

    config = {k: v for k, v in dict(run.config).items() if not k.startswith("_")}
    return run_from_records(
        records,
        config=config,
        name=run.name or parts["run_id"],
        source=f"wandb:{parts['entity']}/{parts['project']}/{parts['run_id']}",
    )
