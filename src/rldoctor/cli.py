"""Command line interface.

    rldoctor diagnose wandb://team/project/run_id
    rldoctor diagnose ./logs/train.jsonl --format markdown -o report.md
    rldoctor demo entropy_collapse
    rldoctor selftest
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .detectors.base import Severity

_SEVERITY_CHOICES = ["info", "warning", "critical", "never"]


def _severity_from(name: str) -> Severity:
    """Map a --fail-on choice to a threshold. 'never' is handled by the caller."""
    return {
        "info": Severity.INFO,
        "warning": Severity.WARNING,
        "critical": Severity.CRITICAL,
    }[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rldoctor",
        description="Diagnose why your GRPO/RLVR run is burning GPU-hours without learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  rldoctor diagnose wandb://my-team/my-project/3xk91abc\n"
            "  rldoctor diagnose ./outputs/run/tensorboard\n"
            "  rldoctor diagnose train_log.jsonl --format markdown -o report.md\n"
            "  rldoctor demo reward_hacking\n"
            "  rldoctor selftest\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"rldoctor {__version__}")
    sub = parser.add_subparsers(dest="command")

    diagnose = sub.add_parser("diagnose", help="diagnose a training run")
    diagnose.add_argument(
        "uri",
        help="path to a .jsonl/.json/.csv log, a TensorBoard directory, "
        "or wandb://entity/project/run_id",
    )
    _add_common(diagnose)

    demo = sub.add_parser("demo", help="diagnose a simulated run with a known pathology")
    demo.add_argument("scenario", nargs="?", default="entropy_collapse")
    demo.add_argument("--steps", type=int, default=400)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--list", action="store_true", help="list available scenarios and exit")
    _add_common(demo)

    sub.add_parser("selftest", help="check every detector against every simulated scenario")
    sub.add_parser("detectors", help="list available detectors")
    sub.add_parser("fields", help="list canonical metric fields and their framework aliases")

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f", "--format", default="terminal", choices=["terminal", "markdown", "json", "html"]
    )
    parser.add_argument("-o", "--output", help="write the report to a file instead of stdout")
    parser.add_argument("--only", nargs="+", default=[], metavar="DETECTOR")
    parser.add_argument("--skip", nargs="+", default=[], metavar="DETECTOR")
    parser.add_argument(
        "--fail-on",
        default="never",
        choices=_SEVERITY_CHOICES,
        help="exit non-zero when a finding at or above this severity fires (default: never)",
    )
    parser.add_argument("--gpu-hour-cost", type=float, help="USD per GPU-hour for the cost estimate")
    parser.add_argument("--num-gpus", type=int, help="override the GPU count from the run config")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="show unrecognised log keys")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "detectors":
        return _cmd_detectors()
    if args.command == "fields":
        return _cmd_fields()
    if args.command == "selftest":
        return _cmd_selftest()
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "diagnose":
        return _cmd_diagnose(args)
    parser.print_help()
    return 0


# -- commands ---------------------------------------------------------------


def _cmd_diagnose(args) -> int:
    from .ingest import load_run

    try:
        run = load_run(args.uri)
    except Exception as exc:
        print(f"rldoctor: could not load {args.uri}: {exc}", file=sys.stderr)
        return 2
    return _run_and_emit(run, args)


def _cmd_demo(args) -> int:
    from .simulate import SCENARIOS, simulate_run

    if args.list:
        print("available demo scenarios:\n")
        for name in SCENARIOS:
            print(f"  {name}")
        return 0
    if args.scenario not in SCENARIOS:
        print(
            f"rldoctor: unknown scenario {args.scenario!r}.\n"
            f"available: {', '.join(SCENARIOS)}",
            file=sys.stderr,
        )
        return 2
    run = simulate_run(args.scenario, n_steps=args.steps, seed=args.seed)
    if args.format == "terminal":
        print(
            f"\n  (simulated run: '{args.scenario}'. "
            "Everything below is computed from synthetic metrics.)"
        )
    return _run_and_emit(run, args)


def _run_and_emit(run, args) -> int:
    from .diagnosis import diagnose
    from .report import render

    try:
        result = diagnose(
            run,
            only=args.only,
            skip=args.skip,
            usd_per_gpu_hour=args.gpu_hour_cost,
            num_gpus=args.num_gpus,
        )
    except KeyError as exc:
        print(f"rldoctor: {exc}", file=sys.stderr)
        return 2

    if args.format == "terminal" and args.no_color:
        from .report.terminal import render as render_terminal

        text = render_terminal(result, color=False)
    else:
        text = render(result, args.format)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.output}")
    else:
        print(text)

    if args.verbose and run.unmapped_keys:
        print("\nunrecognised log keys:", file=sys.stderr)
        for key in run.unmapped_keys:
            print(f"  {key}", file=sys.stderr)

    if args.fail_on == "never":
        return 0
    return result.exit_code(_severity_from(args.fail_on))


def _cmd_detectors() -> int:
    from . import detectors

    print("available detectors:\n")
    for cls in detectors.BUILTIN_DETECTORS:
        requires = ", ".join(cls.requires) or "(adaptive)"
        print(f"  {cls.name:<22} {cls.title}")
        print(f"  {'':<22} requires: {requires}")
        if cls.algorithms:
            print(f"  {'':<22} algorithms: {', '.join(cls.algorithms)}")
        print()
    return 0


def _cmd_fields() -> int:
    from . import aliases, schema

    print("canonical fields and the framework keys that map onto them:\n")
    for field_name in schema.CANONICAL_FIELDS:
        keys = aliases.ALIASES.get(field_name, [])
        doc = schema.FIELD_DOCS.get(field_name, "")
        print(f"  {field_name}")
        if doc:
            print(f"    {doc}")
        if keys:
            print(f"    aliases: {', '.join(keys)}")
        print()
    return 0


def _cmd_selftest() -> int:
    """Assert every scenario trips the detector it is supposed to trip."""
    from .diagnosis import diagnose
    from .simulate import EXPECTED_DETECTIONS, simulate_run

    failures = 0
    print(f"\nrldoctor {__version__} selftest — {len(EXPECTED_DETECTIONS)} scenarios\n")
    for scenario, expected in EXPECTED_DETECTIONS.items():
        run = simulate_run(scenario, n_steps=400, seed=0)
        result = diagnose(run)
        fired = {f.detector for f in result.problems if f.severity >= Severity.WARNING}

        missing = [name for name in expected if name not in fired]
        if scenario == "healthy":
            ok = not fired
            detail = "no false positives" if ok else f"false positives: {', '.join(sorted(fired))}"
        else:
            ok = not missing
            detail = (
                f"caught {', '.join(sorted(fired))}" if ok else f"MISSED {', '.join(missing)}"
            )

        status = "pass" if ok else "FAIL"
        failures += 0 if ok else 1
        print(f"  [{status}] {scenario:<20} {detail}")

    print()
    if failures:
        print(f"{failures} scenario(s) failed.\n")
        return 1
    print("all scenarios behaved as expected.\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
