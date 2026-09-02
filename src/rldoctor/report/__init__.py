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


def emit(text: str) -> None:
    """Print text a terminal may not be able to encode.

    Reports can carry codepoints a legacy console code page cannot represent.
    Losing a glyph is acceptable; raising UnicodeEncodeError instead of showing
    the report is not.
    """
    import sys

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(text)


__all__ = ["RENDERERS", "emit", "render", "render_json"]
