"""ANSI terminal report.

Deliberately dependency-free: no rich, no colorama. A diagnostic tool that
drags a UI stack into a training image will not get installed on the cluster
where it is actually needed.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import List, Optional

from ..detectors.base import Finding, Severity
from ..diagnosis import Diagnosis

_RESET = "\033[0m"
_STYLES = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "bright_red": "\033[91m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
}

_SEVERITY_STYLE = {
    Severity.CRITICAL: "bright_red",
    Severity.WARNING: "yellow",
    Severity.INFO: "blue",
    Severity.OK: "green",
    Severity.SKIPPED: "dim",
}

_SEVERITY_GLYPH = {
    Severity.CRITICAL: "CRIT",
    Severity.WARNING: "WARN",
    Severity.INFO: "INFO",
    Severity.OK: " OK ",
    Severity.SKIPPED: "SKIP",
}


#: Box-drawing glyphs, with an ASCII fallback for terminals whose encoding
#: cannot represent them (Windows consoles on a legacy code page, most CI logs
#: with a C locale). Printing mojibake is worse than printing dashes.
_GLYPHS = {
    "unicode": {"rule": "─", "dot": "·", "bar": "▍"},
    "ascii": {"rule": "-", "dot": "*", "bar": ">"},
}


def _supports_unicode(stream=None) -> bool:
    if os.environ.get("RLDOCTOR_ASCII"):
        return False
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or ""
    try:
        "─·▍".encode(encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _supports_color(stream=None) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


class _Painter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        prefix = "".join(_STYLES.get(s, "") for s in styles)
        return f"{prefix}{text}{_RESET}"


def render(
    diagnosis: Diagnosis,
    width: Optional[int] = None,
    color: Optional[bool] = None,
    unicode: Optional[bool] = None,
) -> str:
    """Render a full report as a string."""
    width = width or min(shutil.get_terminal_size((100, 24)).columns, 100)
    paint = _Painter(_supports_color() if color is None else color)
    glyph = _GLYPHS["unicode" if (_supports_unicode() if unicode is None else unicode) else "ascii"]
    out: List[str] = []

    out += _header(diagnosis, width, paint, glyph)
    out += _cost_banner(diagnosis, width, paint, glyph)

    problems = diagnosis.problems
    if problems:
        out.append(_rule("FINDINGS", width, paint, glyph))
        out.append("")
        for finding in problems:
            out += _finding_block(finding, width, paint, glyph)
    else:
        out.append(_rule("FINDINGS", width, paint, glyph))
        out.append("")
        out.append(paint("  Nothing to report. Every applicable check passed.", "green"))
        out.append("")

    out += _footer(diagnosis, width, paint, glyph)
    return "\n".join(out)


def _header(diagnosis: Diagnosis, width: int, paint: _Painter, glyph: dict) -> List[str]:
    run = diagnosis.run
    title = paint("rldoctor", "bold", "cyan") + paint(f" {diagnosis.version}", "dim")
    name = paint(run.name, "bold")
    pad = width - _visible_len(title) - _visible_len(name) - 4
    lines = [
        "",
        "  " + title + " " * max(pad, 1) + name,
        "  " + paint(glyph["rule"] * (width - 4), "dim"),
    ]

    facts = [f"{run.n_steps} steps", f"{len(run.available_fields)} metrics"]
    config = run.config
    if config.algorithm:
        facts.append(config.algorithm.upper())
    if config.model_name:
        facts.append(config.model_name)
    if config.group_size:
        facts.append(f"G={config.group_size}")
    if config.num_gpus:
        facts.append(f"{config.num_gpus}x {config.gpu_type or 'GPU'}")
    lines.append("  " + paint(f" {glyph['dot']} ".join(facts), "dim"))
    lines.append("")
    return lines


def _cost_banner(diagnosis: Diagnosis, width: int, paint: _Painter, glyph: dict) -> List[str]:
    headline = diagnosis.cost.headline()
    if not headline:
        return []
    body = textwrap.wrap(headline, width - 8)
    lines = ["  " + paint(glyph["bar"] + " ", "bright_red") + paint(body[0], "bold")]
    lines += ["    " + paint(chunk, "dim") for chunk in body[1:]]
    lines.append("")
    return lines


def _finding_block(finding: Finding, width: int, paint: _Painter, glyph: dict) -> List[str]:
    style = _SEVERITY_STYLE[finding.severity]
    badge = paint(f" {_SEVERITY_GLYPH[finding.severity]} ", "bold", style)
    lines = [f"  {badge} {paint(finding.title, 'bold')}"]

    body_width = width - 8
    for chunk in textwrap.wrap(finding.summary, body_width):
        lines.append("        " + chunk)
    lines.append("")

    if finding.evidence:
        lines.append("        " + paint("evidence", "dim", "bold"))
        for item in finding.evidence:
            lines += _bullet(item, body_width - 4, paint, marker=glyph["dot"], indent=10)
        lines.append("")

    if finding.prescription:
        lines.append("        " + paint("do this", "dim", "bold"))
        for index, item in enumerate(finding.prescription, start=1):
            lines += _bullet(item, body_width - 5, paint, marker=f"{index}.", indent=10)
        lines.append("")

    if finding.references:
        lines.append("        " + paint("refs", "dim", "bold"))
        for item in finding.references:
            lines += _bullet(item, body_width - 4, paint, marker=glyph["dot"], indent=10, dim=True)
        lines.append("")

    return lines


def _bullet(
    text: str, width: int, paint: _Painter, marker: str, indent: int, dim: bool = False
) -> List[str]:
    prefix = " " * indent + paint(marker, "dim") + " "
    hanging = " " * (indent + len(marker) + 1)
    chunks = textwrap.wrap(text, max(width, 20)) or [""]
    styles = ("dim",) if dim else ()
    return [prefix + paint(chunks[0], *styles)] + [
        hanging + paint(chunk, *styles) for chunk in chunks[1:]
    ]


def _footer(diagnosis: Diagnosis, width: int, paint: _Painter, glyph: dict) -> List[str]:
    lines = [_rule("CHECKS", width, paint, glyph), ""]

    passed = diagnosis.passed
    if passed:
        names = ", ".join(f.detector for f in passed)
        for chunk in textwrap.wrap(f"passed: {names}", width - 6):
            lines.append("  " + paint(chunk, "green"))

    skipped = diagnosis.skipped
    if skipped:
        lines.append("")
        lines.append("  " + paint("skipped (not enough data):", "dim"))
        for finding in skipped:
            for chunk in _bullet(
                f"{finding.detector} -- {finding.summary}", width - 12, paint, glyph["dot"], 4, dim=True
            ):
                lines.append(chunk)

    unmapped = diagnosis.run.unmapped_keys
    if unmapped:
        lines.append("")
        preview = ", ".join(unmapped[:6]) + (" ..." if len(unmapped) > 6 else "")
        for chunk in textwrap.wrap(
            f"{len(unmapped)} log keys were not recognised: {preview}", width - 6
        ):
            lines.append("  " + paint(chunk, "dim"))
        lines.append(
            "  "
            + paint(
                "If one of those is a metric rldoctor should understand, please open an issue.",
                "dim",
            )
        )

    lines.append("")
    return lines


def _rule(label: str, width: int, paint: _Painter, glyph: dict) -> str:
    rule = glyph["rule"]
    text = f"{rule * 2} {label} "
    return "  " + paint(text + rule * max(width - len(text) - 4, 0), "dim")


def _visible_len(text: str) -> int:
    """Length ignoring ANSI escape sequences."""
    out, in_escape = 0, False
    for char in text:
        if in_escape:
            if char == "m":
                in_escape = False
            continue
        if char == "\033":
            in_escape = True
            continue
        out += 1
    return out
