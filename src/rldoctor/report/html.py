"""Self-contained HTML report.

One file, no network calls, no CDN, no JavaScript framework. It has to open
from a scratch directory on a cluster login node and from a Slack attachment on
someone's phone, so everything -- CSS and the sparkline SVGs -- is inlined.
"""

from __future__ import annotations

import html as _html
from typing import List, Optional, Sequence

import numpy as np

from .. import schema as S
from ..detectors.base import Finding, Severity
from ..diagnosis import Diagnosis

_SEVERITY_CLASS = {
    Severity.CRITICAL: "crit",
    Severity.WARNING: "warn",
    Severity.INFO: "info",
    Severity.OK: "ok",
    Severity.SKIPPED: "skip",
}

#: Series worth plotting, in the order a practitioner reads them.
_CHARTS: Sequence = (
    (S.REWARD_MEAN, "training reward"),
    (S.EVAL_SCORE, "held-out score"),
    (S.ENTROPY, "entropy"),
    (S.ZERO_VAR_GROUP_FRAC, "zero-variance groups"),
    (S.KL, "KL to reference"),
    (S.RESPONSE_LEN_MEAN, "response length"),
    (S.PG_CLIPFRAC, "clip fraction"),
    (S.GRAD_NORM, "grad norm"),
)

_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a1a;--muted:#6b6b6b;--line:#e3e3e0;--card:#fff;
--crit:#c0392b;--warn:#b8860b;--info:#2b6cb0;--ok:#2f855a;--skip:#9a9a9a;}
@media (prefers-color-scheme:dark){:root{--bg:#151515;--fg:#ececec;--muted:#9a9a9a;
--line:#2c2c2c;--card:#1d1d1d;--crit:#ff6b5b;--warn:#e2b13c;--info:#6aa9e9;--ok:#5ec98a;--skip:#777;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 ui-sans-serif,-apple-system,
"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:920px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
h1 .v{color:var(--muted);font-weight:400;font-size:14px}
.facts{color:var(--muted);font-size:13px;margin-bottom:28px}
.banner{border-left:3px solid var(--crit);background:var(--card);padding:14px 18px;
border-radius:0 6px 6px 0;margin-bottom:32px;font-weight:600}
.banner .sub{font-weight:400;color:var(--muted);font-size:13px;margin-top:4px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
border-bottom:1px solid var(--line);padding-bottom:8px;margin:40px 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:18px 20px;margin-bottom:16px}
.card.crit{border-left:3px solid var(--crit)}
.card.warn{border-left:3px solid var(--warn)}
.card.info{border-left:3px solid var(--info)}
.badge{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.07em;
padding:2px 7px;border-radius:3px;color:#fff;vertical-align:2px}
.badge.crit{background:var(--crit)}.badge.warn{background:var(--warn)}
.badge.info{background:var(--info)}.badge.ok{background:var(--ok)}.badge.skip{background:var(--skip)}
.title{font-weight:650;margin-left:9px}
.summary{margin:12px 0 4px}
details{margin-top:10px}
summary{cursor:pointer;color:var(--muted);font-size:13px;user-select:none}
summary:hover{color:var(--fg)}
ul,ol{margin:10px 0;padding-left:22px}
li{margin:5px 0}
.evidence li{color:var(--muted);font-size:13.5px}
.refs li{color:var(--muted);font-size:12.5px}
code{background:rgba(128,128,128,.14);padding:1px 5px;border-radius:3px;font-size:12.5px}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}
.chart{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.chart .lbl{font-size:11.5px;color:var(--muted);letter-spacing:.03em}
.chart .val{font-size:15px;font-weight:650;margin:2px 0 6px}
.pill{display:inline-block;font-size:12px;color:var(--muted);border:1px solid var(--line);
border-radius:99px;padding:2px 10px;margin:0 6px 6px 0}
footer{margin-top:48px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);
padding-top:16px}
a{color:inherit}
"""


def render(diagnosis: Diagnosis) -> str:
    run = diagnosis.run
    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>rldoctor — {_esc(run.name)}</title>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>rldoctor <span class='v'>{diagnosis.version}</span> — {_esc(run.name)}</h1>",
        f"<div class='facts'>{_esc(_facts(diagnosis))}</div>",
    ]

    headline = diagnosis.cost.headline()
    if headline:
        parts.append(
            f"<div class='banner'>{_esc(headline)}"
            f"<div class='sub'>Estimated from logged step time, assuming "
            f"{diagnosis.cost.rollout_share:.0%} of step wall-clock is rollout generation. "
            "Override the rate with <code>--gpu-hour-cost</code>.</div></div>"
        )

    charts = _charts(run)
    if charts:
        parts += ["<h2>Run at a glance</h2>", f"<div class='charts'>{charts}</div>"]

    parts.append("<h2>Findings</h2>")
    problems = diagnosis.problems
    if not problems:
        parts.append("<div class='card ok'>Every applicable check passed.</div>")
    else:
        parts += [_card(f) for f in problems]

    parts.append("<h2>Checks</h2>")
    if diagnosis.passed:
        parts.append(
            "".join(f"<span class='pill'>✓ {_esc(f.detector)}</span>" for f in diagnosis.passed)
        )
    if diagnosis.skipped:
        items = "".join(
            f"<li><code>{_esc(f.detector)}</code> — {_esc(f.summary)}</li>"
            for f in diagnosis.skipped
        )
        parts.append(
            "<details><summary>"
            f"{len(diagnosis.skipped)} check(s) skipped for lack of data"
            f"</summary><ul class='evidence'>{items}</ul></details>"
        )

    parts.append(
        "<footer>Generated by "
        "<a href='https://github.com/junglezke/rldoctor'>rldoctor</a> "
        f"v{diagnosis.version} at {_esc(diagnosis.created_at)}. "
        "Thresholds and their rationale are documented in "
        "<code>docs/detectors.md</code>.</footer>"
    )
    parts.append("</div></body></html>")
    return "\n".join(parts)


def _facts(diagnosis: Diagnosis) -> str:
    run = diagnosis.run
    facts = [f"{run.n_steps} steps", f"{len(run.available_fields)} metrics"]
    config = run.config
    if config.algorithm:
        facts.append(config.algorithm.upper())
    if config.model_name:
        facts.append(config.model_name)
    if config.group_size:
        facts.append(f"group size {config.group_size}")
    if config.num_gpus:
        facts.append(f"{config.num_gpus}× {config.gpu_type or 'GPU'}")
    facts.append(f"source: {run.source}")
    return " · ".join(facts)


def _card(finding: Finding) -> str:
    cls = _SEVERITY_CLASS[finding.severity]
    out = [
        f"<div class='card {cls}'>",
        f"<span class='badge {cls}'>{finding.severity.label}</span>",
        f"<span class='title'>{_esc(finding.title)}</span>",
        f"<div class='summary'>{_esc(finding.summary)}</div>",
    ]
    if finding.prescription:
        items = "".join(f"<li>{_esc(p)}</li>" for p in finding.prescription)
        out.append(f"<ol>{items}</ol>")
    if finding.evidence:
        items = "".join(f"<li>{_esc(e)}</li>" for e in finding.evidence)
        out.append(f"<details open><summary>Evidence</summary><ul class='evidence'>{items}</ul></details>")
    if finding.references:
        items = "".join(f"<li>{_esc(r)}</li>" for r in finding.references)
        out.append(f"<details><summary>References</summary><ul class='refs'>{items}</ul></details>")
    out.append("</div>")
    return "".join(out)


def _charts(run) -> str:
    cards = []
    for field_name, label in _CHARTS:
        if not run.has(field_name, min_points=4):
            continue
        steps, values = run.finite(field_name)
        spark = _sparkline(steps, values)
        current = values[-1]
        cards.append(
            f"<div class='chart'><div class='lbl'>{_esc(label)}</div>"
            f"<div class='val'>{_fmt(current)}</div>{spark}</div>"
        )
    return "".join(cards)


def _sparkline(x: np.ndarray, y: np.ndarray, width: int = 190, height: int = 40) -> str:
    """Inline SVG sparkline. No library, no runtime, no external request."""
    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo
    if span <= 0:
        span = 1.0
        lo -= 0.5
    x_lo, x_hi = float(x[0]), float(x[-1])
    x_span = (x_hi - x_lo) or 1.0

    points = " ".join(
        f"{(xi - x_lo) / x_span * width:.2f},{height - (yi - lo) / span * height:.2f}"
        for xi, yi in zip(x, y, strict=True)
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' "
        "preserveAspectRatio='none' aria-hidden='true'>"
        f"<polyline points='{points}' fill='none' stroke='currentColor' "
        "stroke-width='1.4' stroke-linejoin='round' opacity='.75'/></svg>"
    )


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:.3g}"
    return f"{value:.4g}"


def _esc(text: Optional[str]) -> str:
    return _html.escape(str(text if text is not None else ""))
