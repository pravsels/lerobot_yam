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
    assert corrections.reliable is True


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
    assert corrections.reliable is False


def test_implausible_amplitude_marks_fit_unreliable():
    """Regression from real bench data: a clustered-angle fit extrapolated to a
    25 A 'amplitude'; such a suggestion must be withheld, not printed."""
    theta = np.linspace(1.0, 1.05, 4)
    measured_fit = fit_sine(theta, _sine(theta, 25378.0, -2.44, -24528.0))
    model_fit = fit_sine(theta, _sine(theta, 137.0, -2.2, 1243.0))

    corrections = suggest_corrections(measured_fit, model_fit)

    assert corrections.reliable is False
    assert any("XL330 could balance" in w for w in corrections.warnings)


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


def _chain_locked_torque(model, signs, offsets, joint, offset, q):
    """Model torque with the joint's offset set and the next joint co-varied,
    replicating the chain-locked scan of solve_apex_offset."""
    offs = list(offsets)
    delta = offset - offs[joint]
    offs[joint] = offset
    if joint + 1 < len(offs):
        offs[joint + 1] -= delta
    q_urdf = np.asarray(signs, float) * np.asarray(q, float) + np.asarray(offs)
    return float(model.gravity_torques(q_urdf)[joint])


def test_apex_capture_places_torque_zero_with_unstable_slope():
    from lerobot_teleoperator_yam_gello.gravity_assist import GelloGravityModel
    from lerobot_teleoperator_yam_gello.gravity_sysid_fit import solve_apex_offset

    model = GelloGravityModel()
    signs = (1, -1, -1, -1, 1, 1)
    offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    q_yam = np.array([0.1, 0.6, 1.2, 0.05, 0.0, -0.3])
    joint = 2  # elbow_flex

    solution = solve_apex_offset(model, q_yam, signs, offsets, joint)

    def torque_at(offset, q):
        return _chain_locked_torque(model, signs, offsets, joint, offset, q)

    # Zero torque at the captured pose for both candidate offsets...
    assert torque_at(solution.offset_apex_rad, q_yam) == pytest.approx(0.0, abs=1e-3)
    assert torque_at(solution.offset_hanging_rad, q_yam) == pytest.approx(0.0, abs=1e-3)
    # ...but only the apex offset gives a motor current with negative slope
    # (unstable equilibrium: the commanded current flips sign across the apex).
    eps = 1e-3
    plus, minus = q_yam.copy(), q_yam.copy()
    plus[joint] += eps
    minus[joint] -= eps
    apex_slope = signs[joint] * (
        torque_at(solution.offset_apex_rad, plus) - torque_at(solution.offset_apex_rad, minus)
    )
    hanging_slope = signs[joint] * (
        torque_at(solution.offset_hanging_rad, plus)
        - torque_at(solution.offset_hanging_rad, minus)
    )
    assert apex_slope < 0
    assert hanging_slope > 0
    assert solution.amplitude_nm > 0.05


def test_chain_locked_solve_pins_distal_angles():
    """Regression from the shoulder-lift fiasco: scanning a proximal offset
    must not rotate the already-calibrated distal chain. With calibrated
    distal offsets in the config, the solve zeroes the torque of the
    co-varied (shape-preserving) configuration."""
    from lerobot_teleoperator_yam_gello.gravity_assist import GelloGravityModel
    from lerobot_teleoperator_yam_gello.gravity_sysid_fit import solve_apex_offset

    model = GelloGravityModel()
    signs = (1, 1, -1, -1, 1, 1)
    offsets = [0.0, 0.0, 2.70, 0.07, 0.0, 0.0]
    q_yam = np.array([0.1, 0.63, 1.0, 0.88, -0.04, -0.01])
    joint = 1  # shoulder_lift

    solution = solve_apex_offset(model, q_yam, signs, offsets, joint)

    locked_zero = _chain_locked_torque(
        model, signs, offsets, joint, solution.offset_apex_rad, q_yam
    )
    assert locked_zero == pytest.approx(0.0, abs=1e-3)
    # The pinned distal torque is reported so the operator can sanity-check
    # that the joint's own authority can actually balance it.
    assert abs(solution.distal_torque_nm) <= solution.amplitude_nm
    # An unlocked scan would generally land somewhere else entirely.
    unlocked = solve_apex_offset(model, q_yam, signs, offsets, joint, lock_distal=False)
    assert unlocked.offset_apex_rad != pytest.approx(solution.offset_apex_rad, abs=0.05)


def test_apex_capture_rejects_weightless_distal_chain():
    from lerobot_teleoperator_yam_gello.gravity_assist import GelloGravityModel
    from lerobot_teleoperator_yam_gello.gravity_sysid_fit import solve_apex_offset

    model = GelloGravityModel(
        link_masses_kg=[0.142, 0.09, 0.12, 0.0, 0.0, 0.0, 0.0]
    )

    with pytest.raises(ValueError, match="no gravity signal"):
        solve_apex_offset(
            model,
            np.zeros(6),
            (1, -1, -1, -1, 1, 1),
            [0.0] * 6,
            2,
        )
