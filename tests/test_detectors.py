"""The detection matrix: every scenario must trip its own detector, and the
healthy run must trip nothing.

The second half of that sentence is the one that matters. A diagnostic that
fires on healthy runs gets uninstalled after the first false alarm, so the
healthy-run tests are deliberately strict and run across multiple seeds.
"""

from __future__ import annotations

import pytest

from rldoctor import Severity, diagnose
from rldoctor.detectors import BUILTIN_DETECTORS, available, build
from rldoctor.simulate import EXPECTED_DETECTIONS, SCENARIOS, simulate_run


@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if s != "healthy"])
def test_scenario_trips_its_detector(scenario):
    result = diagnose(simulate_run(scenario, n_steps=400, seed=0))
    fired = {f.detector for f in result.problems if f.severity >= Severity.WARNING}
    for expected in EXPECTED_DETECTIONS[scenario]:
        assert expected in fired, (
            f"{scenario} should trip {expected}; fired instead: {sorted(fired) or 'nothing'}"
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_healthy_run_raises_no_warnings(seed):
    result = diagnose(simulate_run("healthy", n_steps=400, seed=seed))
    fired = {f.detector for f in result.problems if f.severity >= Severity.WARNING}
    assert not fired, f"false positives on a healthy run (seed={seed}): {sorted(fired)}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_findings_are_actionable(scenario):
    """Anything at WARNING or above must carry evidence and a prescription.

    A finding without a suggested change is a complaint, not a diagnosis.
    """
    result = diagnose(simulate_run(scenario, n_steps=400, seed=0))
    for finding in result.problems:
        if finding.severity >= Severity.WARNING:
            assert finding.evidence, f"{finding.detector} fired with no evidence"
            assert finding.prescription, f"{finding.detector} fired with no prescription"
            assert finding.summary.strip()


def test_every_detector_reports_something_on_a_full_log():
    """With a complete log, no detector should silently skip."""
    result = diagnose(simulate_run("healthy", n_steps=400, seed=0))
    skipped = {f.detector for f in result.skipped}
    assert not skipped, f"detectors skipped despite a complete log: {sorted(skipped)}"


def test_detectors_skip_gracefully_on_a_sparse_log():
    from rldoctor import run_from_records

    run = run_from_records([{"step": i, "reward": 0.5} for i in range(30)])
    result = diagnose(run)
    # Nothing should crash, and the checks that cannot run must say why.
    assert all(f.summary for f in result.skipped)
    assert len(result.findings) == len(BUILTIN_DETECTORS)


def test_a_broken_detector_does_not_sink_the_report():
    from rldoctor.detectors.base import Detector

    class Exploding(Detector):
        name = "exploding"
        title = "Always raises"

        def run(self, run):
            raise RuntimeError("boom")

    result = diagnose(simulate_run("healthy", seed=0), extra_detectors=[Exploding()])
    broken = result.by_name("exploding")
    assert broken is not None
    assert broken.severity is Severity.SKIPPED
    assert "RuntimeError" in broken.summary
    # And the rest of the report survived.
    assert len(result.findings) == len(BUILTIN_DETECTORS) + 1


def test_only_and_skip_select_detectors():
    assert [d.name for d in build(only=["plateau"])] == ["plateau"]
    names = [d.name for d in build(skip=["plateau"])]
    assert "plateau" not in names
    assert len(names) == len(BUILTIN_DETECTORS) - 1


def test_unknown_detector_name_is_rejected_with_a_helpful_message():
    with pytest.raises(KeyError, match="unknown detector"):
        build(only=["no_such_detector"])


def test_registry_names_are_unique_and_stable():
    names = available()
    assert len(names) == len(set(names))
    assert set(names) == {cls.name for cls in BUILTIN_DETECTORS}


def test_advantage_collapse_distinguishes_too_easy_from_too_hard():
    """Same symptom, opposite fix. Reporting the bare fraction is not enough."""
    easy = diagnose(simulate_run("saturated_groups", seed=0)).by_name("advantage_collapse")
    hard = diagnose(simulate_run("too_hard", seed=0)).by_name("advantage_collapse")
    assert easy.metrics["regime"] == "saturated"
    assert hard.metrics["regime"] == "too_hard"
    assert "too easy" in easy.summary
    assert "too hard" in hard.summary


def test_reward_hacking_needs_a_held_out_signal():
    """Without an eval series the check must skip, not guess."""
    from rldoctor import run_from_records

    run = run_from_records(
        [{"step": i, "reward": i / 100.0, "entropy": 0.5} for i in range(100)]
    )
    finding = diagnose(run).by_name("reward_hacking")
    assert finding.severity is Severity.SKIPPED
    assert "eval_score" in finding.missing_fields


def test_length_growth_alone_is_not_an_accusation():
    """Reasoning models legitimately get longer. Without a held-out score the
    detector must stay at INFO rather than assert hacking."""
    result = diagnose(simulate_run("length_hacking", seed=0, eval_every=10_000))
    finding = result.by_name("length_pathology")
    assert finding.severity <= Severity.INFO
