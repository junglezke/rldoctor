"""Live monitoring must be useful, quiet, and impossible to crash a job with."""

from __future__ import annotations

import numpy as np
import pytest

from rldoctor import Severity, stats
from rldoctor.live import LiveMonitor
from rldoctor.simulate import simulate


def _feed(monitor: LiveMonitor, scenario: str, n_steps: int = 400, seed: int = 0):
    records, _ = simulate(scenario, n_steps=n_steps, seed=seed)
    flagged = []
    for step, record in enumerate(records):
        for finding in monitor.log(record, step=step):
            flagged.append((step, finding))
    return flagged


def test_it_catches_collapse_long_before_the_run_ends():
    """The whole point: flag it at step N, not in the post-mortem."""
    records, config = simulate("entropy_collapse", n_steps=400, seed=0)
    monitor = LiveMonitor(config=config, check_every=25)
    first_step = None
    for step, record in enumerate(records):
        for finding in monitor.log(record, step=step):
            if finding.detector == "entropy_collapse" and first_step is None:
                first_step = step
    assert first_step is not None
    assert first_step < 200, f"flagged too late (step {first_step} of 400)"


def test_a_healthy_run_is_never_interrupted():
    records, config = simulate("healthy", n_steps=400, seed=0)
    monitor = LiveMonitor(config=config, check_every=25)
    flagged = [f for step, record in enumerate(records) for f in monitor.log(record, step=step)]
    assert not flagged, f"interrupted a healthy run with {[f.detector for f in flagged]}"


def test_findings_are_announced_once_unless_they_get_worse():
    records, config = simulate("entropy_collapse", n_steps=400, seed=0)
    monitor = LiveMonitor(config=config, check_every=25)
    seen = [
        (f.detector, f.severity)
        for step, record in enumerate(records)
        for f in monitor.log(record, step=step)
    ]
    entropy_alerts = [sev for name, sev in seen if name == "entropy_collapse"]
    # Repeats are allowed only when severity strictly increases.
    assert entropy_alerts == sorted(set(entropy_alerts)), entropy_alerts


def test_nothing_fires_during_warm_up():
    records, config = simulate("entropy_collapse", n_steps=400, seed=0)
    monitor = LiveMonitor(config=config, check_every=5, min_steps=60)
    early = [f for step, record in enumerate(records[:59]) for f in monitor.log(record, step=step)]
    assert not early


def test_a_broken_record_cannot_kill_training():
    """A monitoring bug must never take down an otherwise healthy job."""
    monitor = LiveMonitor(check_every=1, min_steps=8)
    for step in range(12):
        assert monitor.log({"reward": float(step)}, step=step) is not None
    # Feed something structurally hostile.
    assert monitor.log({"reward": object()}, step=99) == []


def test_callback_exceptions_are_contained():
    def explode(finding):
        raise RuntimeError("user callback is broken")

    records, config = simulate("entropy_collapse", n_steps=200, seed=0)
    monitor = LiveMonitor(config=config, check_every=25, on_finding=explode)
    for step, record in enumerate(records):
        monitor.log(record, step=step)  # must not raise
    assert monitor.worst >= Severity.WARNING


def test_report_works_before_any_check_has_run():
    monitor = LiveMonitor()
    assert "not enough steps" in monitor.report() or monitor.latest is not None


def test_saves_an_html_report(tmp_path):
    records, config = simulate("plateau", n_steps=200, seed=0)
    monitor = LiveMonitor(config=config, check_every=25)
    for step, record in enumerate(records):
        monitor.log(record, step=step)
    target = tmp_path / "report.html"
    monitor.save(str(target))
    assert target.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# -- the projection model that drives every ETA we print ---------------------


def test_projection_picks_the_exponential_model_for_decay():
    """A linear extrapolation of exponential decay overstates urgency by an
    order of magnitude. Getting this wrong once costs the tool its credibility."""
    x = np.arange(100, dtype=float)  # ends at 0.081, still above the threshold
    y = 0.6 * np.exp(-0.02 * x)  # crosses 0.05 at x = ln(12)/0.02 = 124.2
    crossing, model = stats.project_threshold(x, y, 0.05)
    assert model == "exponential"
    assert crossing == pytest.approx(124.2, rel=0.05)


def test_projection_uses_a_linear_model_when_that_fits_better():
    x = np.arange(100, dtype=float)  # ends at 5.05, still above the threshold
    y = 10.0 - 0.05 * x  # crosses 1.0 at x = 180
    crossing, model = stats.project_threshold(x, y, 1.0)
    assert model == "linear"
    assert crossing == pytest.approx(180.0, rel=0.02)


def test_projection_refuses_to_forecast_a_series_moving_away():
    x = np.arange(100, dtype=float)
    y = 1.0 + 0.1 * x
    assert stats.project_threshold(x, y, 0.5) == (None, "none")


def test_projection_reports_nothing_for_a_threshold_already_crossed():
    """"When will it cross?" has no forward answer once it already has."""
    x = np.arange(200, dtype=float)
    y = 10.0 - 0.05 * x  # ends at 0.05, long past 1.0
    assert stats.project_threshold(x, y, 1.0) == (None, "none")


def test_projection_declines_on_noise():
    rng = np.random.default_rng(0)
    x = np.arange(100, dtype=float)
    crossing, model = stats.project_threshold(x, 5.0 + rng.normal(0, 1, 100), 0.1)
    assert model == "none" and crossing is None


def test_head_window_is_capped_so_a_baseline_cannot_drift():
    long_series = np.arange(1000, dtype=float)
    assert stats.head(long_series).size == 25
    assert stats.head(np.arange(10, dtype=float)).size == 5
