import math

import numpy as np
import pytest

from lerobot_teleoperator_yam_gello.gravity_sysid_fit import (
    apply_offset_delta,
    balance_and_friction,
    fit_sine,
    suggest_corrections,
    wrap_angle,
)


def _sine(theta, amplitude, phase, intercept):
    return [amplitude * math.sin(t + phase) + intercept for t in theta]


def test_fit_recovers_amplitude_phase_intercept():
    theta = np.linspace(-0.5, 1.8, 7)
    fit = fit_sine(theta, _sine(theta, amplitude=120.0, phase=0.4, intercept=-8.0))

    assert fit.amplitude == pytest.approx(120.0, abs=1e-6)
    assert fit.phase_rad == pytest.approx(0.4, abs=1e-6)
    assert fit.intercept == pytest.approx(-8.0, abs=1e-6)
    assert fit.rms_residual == pytest.approx(0.0, abs=1e-6)


def test_suggested_offset_delta_recovers_known_phase_shift():
    """If reality is the model rotated by +0.785 rad, the delta must say so."""
    theta = np.linspace(-0.3, 1.6, 6)
    model_fit = fit_sine(theta, _sine(theta, 100.0, 0.1, 0.0))
    measured_fit = fit_sine(theta, _sine(theta, 100.0, 0.1 + 0.785, 0.0))

    corrections = suggest_corrections(measured_fit, model_fit)

    assert corrections.offset_delta_rad == pytest.approx(0.785, abs=1e-6)
    assert corrections.amplitude_ratio == pytest.approx(1.0, abs=1e-6)
    assert corrections.warnings == ()


def test_amplitude_ratio_reflects_distal_mass_error():
    theta = np.linspace(0.0, 1.5, 5)
    model_fit = fit_sine(theta, _sine(theta, 80.0, 0.0, 0.0))
    measured_fit = fit_sine(theta, _sine(theta, 128.0, 0.0, 0.0))

    corrections = suggest_corrections(measured_fit, model_fit)

    assert corrections.amplitude_ratio == pytest.approx(1.6, abs=1e-6)


def test_clustered_poses_are_flagged_as_ill_conditioned():
    theta = [1.000, 1.001, 1.002, 1.003]
    measured_fit = fit_sine(theta, _sine(theta, 100.0, 0.3, 0.0))
    model_fit = fit_sine(theta, _sine(theta, 100.0, 0.0, 0.0))

    corrections = suggest_corrections(measured_fit, model_fit)

    assert any("ill-conditioned" in w for w in corrections.warnings)


def test_tiny_amplitude_flags_unreliable_phase():
    theta = np.linspace(-0.5, 1.5, 5)
    measured_fit = fit_sine(theta, _sine(theta, 5.0, 0.3, 0.0))
    model_fit = fit_sine(theta, _sine(theta, 6.0, 0.0, 0.0))

    corrections = suggest_corrections(measured_fit, model_fit)

    assert any("friction floor" in w for w in corrections.warnings)


def test_balance_and_friction_from_edges():
    balance, friction = balance_and_friction(-40.0, 120.0)
    assert balance == pytest.approx(40.0)
    assert friction == pytest.approx(80.0)
    with pytest.raises(ValueError):
        balance_and_friction(120.0, -40.0)


def test_apply_offset_delta_touches_only_the_target_joint_and_wraps():
    offsets = (0.0, 0.0, 0.785, -0.785, 0.0, 0.0)
    updated = apply_offset_delta(offsets, 3, -3.0)

    assert updated[2] == pytest.approx(0.785)
    assert updated[3] == pytest.approx(wrap_angle(-0.785 - 3.0))
    assert updated[0] == 0.0
    assert len(updated) == 6


def test_fit_requires_three_poses():
    with pytest.raises(ValueError, match="3 poses"):
        fit_sine([0.0, 1.0], [1.0, 2.0])


def test_edge_event_fit_recovers_curve_and_friction_despite_drift():
    """Per-event fit tolerates the joint drifting between probes: every edge
    carries its own angle, unlike a per-pose balance midpoint."""
    from lerobot_teleoperator_yam_gello.gravity_sysid_fit import fit_edge_events

    amplitude, phase, intercept, friction = 110.0, 0.35, -6.0, 42.0
    theta, currents, directions = [], [], []
    for t in np.linspace(-0.9, 1.4, 6):
        for d in (-1, 1):
            # Edges observed at slightly different angles (drift during probing).
            t_edge = t + (0.03 if d > 0 else -0.02)
            theta.append(t_edge)
            currents.append(
                amplitude * math.sin(t_edge + phase) + intercept + friction * d
            )
            directions.append(d)

    fit = fit_edge_events(theta, currents, directions)

    assert fit.sine.amplitude == pytest.approx(amplitude, abs=1e-6)
    assert fit.sine.phase_rad == pytest.approx(phase, abs=1e-6)
    assert fit.sine.intercept == pytest.approx(intercept, abs=1e-6)
    assert fit.friction_ma == pytest.approx(friction, abs=1e-6)


def test_edge_event_fit_requires_both_directions():
    from lerobot_teleoperator_yam_gello.gravity_sysid_fit import fit_edge_events

    with pytest.raises(ValueError, match="both directions"):
        fit_edge_events([0.0, 0.5, 1.0, 1.5], [10, 20, 30, 40], [1, 1, 1, 1])
