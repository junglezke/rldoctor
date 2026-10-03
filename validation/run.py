"""Diagnose every cached public run and compare against the pinned verdicts.

    python validation/fetch.py
    python validation/run.py            # print the table
    python validation/run.py --check    # exit 1 if any verdict changed

Run this before changing a threshold. The simulated scenarios tell you a
detector can fire; these tell you whether it fires on runs that exist.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from rldoctor import Severity, diagnose, load_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 on any verdict drift")
    args = parser.parse_args()

    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))["runs"]
    drift = 0
    print(f"{'run':<28}{'steps':>7}{'secs':>6}  verdicts at WARNING or above")
    print("-" * 96)
    for run in runs:
        path = HERE / ".cache" / f"{run['id']}.json"
        if not path.exists():
            print(f"{run['id']:<28}  not cached -- run validation/fetch.py first")
            drift += 1
            continue
        started = time.perf_counter()
        result = diagnose(load_run(str(path)))
        elapsed = time.perf_counter() - started
        got = {
            f.detector: f.severity.name for f in result.problems if f.severity >= Severity.WARNING
        }
        status = "ok" if got == run["expect"] else "DRIFT"
        drift += status != "ok"
        shown = ", ".join(f"{k}:{v}" for k, v in sorted(got.items())) or "none"
        print(f"{run['id']:<28}{result.run.n_steps:>7}{elapsed:>6.1f}  [{status}] {shown}")
        if status != "ok":
            print(f"{'':<43}expected: {run['expect'] or 'none'}")
    print()
    if drift:
        print(f"{drift} run(s) changed verdict. If the new verdict is right, update runs.json "
              "and say why in the commit; if not, the change broke something real.")
    return 1 if (args.check and drift) else 0


if __name__ == "__main__":
    raise SystemExit(main())
