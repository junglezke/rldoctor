"""Reports and CLI: whatever else breaks, the tool must not crash on output."""

from __future__ import annotations

import json

import pytest

from rldoctor import Severity, diagnose
from rldoctor.cli import main
from rldoctor.report import RENDERERS, render
from rldoctor.simulate import SCENARIOS, simulate_run


@pytest.fixture(scope="module")
def result():
    return diagnose(simulate_run("entropy_collapse", n_steps=300, seed=0))


@pytest.mark.parametrize("fmt", sorted(set(RENDERERS)))
def test_every_renderer_produces_output(result, fmt):
    text = render(result, fmt)
    assert isinstance(text, str)
    assert len(text) > 200


def test_json_report_is_valid_and_complete(result):
    payload = json.loads(render(result, "json"))
    assert payload["worst_severity"] == "CRITICAL"
    assert payload["run"]["n_steps"] == 300
    assert len(payload["findings"]) >= 8
    assert all({"detector", "severity", "summary"} <= set(f) for f in payload["findings"])


def test_html_report_is_self_contained(result):
    html = render(result, "html")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "<script src=" not in html, "the report must not phone home for assets"
    assert "<link rel=\"stylesheet\" href=\"http" not in html


def test_terminal_report_has_an_ascii_mode(result):
    from rldoctor.report.terminal import render as render_terminal

    ascii_text = render_terminal(result, color=False, unicode=False)
    ascii_text.encode("ascii")  # must not raise


def test_unknown_format_is_rejected(result):
    with pytest.raises(KeyError, match="unknown format"):
        render(result, "pdf")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenarios_render_without_crashing(scenario):
    result = diagnose(simulate_run(scenario, n_steps=200, seed=1))
    for fmt in ("terminal", "markdown", "json", "html"):
        assert render(result, fmt)


def test_headline_is_one_line(result):
    assert "\n" not in result.headline
    assert result.headline


def test_exit_code_gates_on_severity(result):
    assert result.exit_code(Severity.CRITICAL) == 1
    healthy = diagnose(simulate_run("healthy", seed=0))
    assert healthy.exit_code(Severity.CRITICAL) == 0
    assert healthy.exit_code(Severity.WARNING) == 0


# -- CLI --------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["detectors"], ["fields"], ["selftest"], []])
def test_informational_commands_exit_zero(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out


def test_demo_runs_and_defaults_to_not_failing(capsys):
    assert main(["demo", "healthy", "--steps", "150"]) == 0
    assert "rldoctor" in capsys.readouterr().out


def test_demo_rejects_an_unknown_scenario(capsys):
    assert main(["demo", "nonsense"]) == 2
    assert "unknown scenario" in capsys.readouterr().err


def test_fail_on_gates_ci(capsys):
    assert main(["demo", "entropy_collapse", "--fail-on", "critical", "-f", "json"]) == 1
    assert main(["demo", "healthy", "--fail-on", "critical", "-f", "json"]) == 0


def test_output_file_is_written(tmp_path, capsys):
    target = tmp_path / "report.md"
    assert main(["demo", "plateau", "-f", "markdown", "-o", str(target)]) == 0
    assert target.read_text(encoding="utf-8").startswith("## rldoctor report")


def test_diagnose_reports_a_bad_path_without_a_traceback(capsys):
    assert main(["diagnose", "nope.jsonl"]) == 2
    assert "could not load" in capsys.readouterr().err


def test_diagnose_reads_a_jsonl_file(tmp_path, capsys):
    from rldoctor.simulate import simulate

    records, config = simulate("kl_blowup", n_steps=200, seed=0)
    path = tmp_path / "run.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"config": config}) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")

    assert main(["diagnose", str(path), "-f", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    kl = next(f for f in payload["findings"] if f["detector"] == "kl_drift")
    assert kl["severity"] in {"WARNING", "CRITICAL"}
