"""Regenerate docs/assets/report.svg, the banner in the README.

    python tools/make_banner.py

Kept in the repo so the README image is reproducible rather than a screenshot
nobody can regenerate after the report format changes.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ansi2svg import render  # noqa: E402

from rldoctor import diagnose  # noqa: E402
from rldoctor.report.terminal import render as render_report  # noqa: E402
from rldoctor.simulate import simulate_run  # noqa: E402

_ANSI = re.compile(r"\033\[[0-9;]*m")
_SECOND_ITEM = re.compile(r"^\s+2\.\s")


def main() -> None:
    lines = render_report(
        diagnose(simulate_run("saturated_groups", n_steps=400, seed=0)),
        width=84,
        color=True,
        unicode=True,
    ).split("\n")

    # Header through the first prescription item: enough to make the argument,
    # short enough to read above the fold, and it ends on a whole thought.
    first_action = next(i for i, line in enumerate(lines) if "do this" in line)
    stop = next(
        i
        for i in range(first_action + 1, len(lines))
        if _SECOND_ITEM.match(_ANSI.sub("", lines[i]))
    )

    svg = render(
        "\n".join(lines[1:stop]).rstrip(),
        "rldoctor diagnose wandb://my-team/my-project/3xk91abc",
    )
    out = pathlib.Path(__file__).parent.parent / "docs" / "assets" / "report.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
