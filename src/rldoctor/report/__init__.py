"""Report renderers. Each takes a :class:`~rldoctor.diagnosis.Diagnosis` and returns text."""

from __future__ import annotations

import json
from typing import Callable, Dict

from ..diagnosis import Diagnosis


def render_json(diagnosis: Diagnosis) -> str:
    return json.dumps(diagnosis.to_dict(), indent=2, default=str)


def _terminal(diagnosis: Diagnosis) -> str:
    from .terminal import render

    return render(diagnosis)


def _markdown(diagnosis: Diagnosis) -> str:
    from .markdown import render

    return render(diagnosis)


def _html(diagnosis: Diagnosis) -> str:
    from .html import render

    return render(diagnosis)


RENDERERS: Dict[str, Callable[[Diagnosis], str]] = {
    "terminal": _terminal,
    "text": _terminal,
    "markdown": _markdown,
    "md": _markdown,
    "json": render_json,
    "html": _html,
}


def render(diagnosis: Diagnosis, fmt: str = "terminal") -> str:
    try:
        return RENDERERS[fmt](diagnosis)
    except KeyError:
        raise KeyError(
            f"unknown format {fmt!r}; choose from {', '.join(sorted(set(RENDERERS)))}"
        ) from None


__all__ = ["RENDERERS", "render", "render_json"]
