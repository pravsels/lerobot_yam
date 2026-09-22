"""Fit math for the GELLO gravity sysid CLI (`gello_gravity_sysid`).

With the rest of the arm locked rigid, the torque needed to hold one joint at
rest is a pure pendulum curve in that joint's angle:

    tau_hold(theta) = A * sin(theta + phi)

and in motor units the balance current is I(theta) = A~ * sin(theta + phi) + c
(the intercept c absorbs cable bias / asymmetric friction). Measuring the
balance current at several poses and fitting (A~, phi, c) for both the
*measured* data and the *model-predicted* currents at the same poses gives the
two corrections our gravity-assist config exposes:

- phase gap  -> `gravity_joint_offsets_rad` delta for this joint
    model evaluated at (theta + delta) matches reality when
    delta = phi_measured - phi_model
- amplitude ratio -> scale for the masses distal to this joint

Balance and friction come from the two breakaway edges of the holding
interval (methodology adapted from
`experimental/so101_pwm_control/torque_identification.py` on the
`feat/pwm-control-so101` branch):

    lower edge I_lo:  largest current at which the joint still falls
    upper edge I_hi:  smallest current at which the joint drives upward
    balance  = (I_lo + I_hi) / 2        friction = (I_hi - I_lo) / 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SineFit:
    """`values ~= amplitude * sin(theta + phase) + intercept`."""

    amplitude: float
    phase_rad: float
    intercept: float
    rms_residual: float
    condition_number: float
    n_points: int

    def describe(self, unit: str = "mA") -> str:
        return (
            f"A={self.amplitude:.1f}{unit} phase={self.phase_rad:+.3f}rad "
            f"intercept={self.intercept:+.1f}{unit} "
            f"rms={self.rms_residual:.1f}{unit} cond={self.condition_number:.1f} "
            f"n={self.n_points}"
        )


@dataclass(frozen=True)
class Corrections:
    """Suggested config changes derived from measured vs model fits.

    `reliable=False` means the numbers must not be applied: the fit was
    degenerate (clustered angles) or physically implausible (amplitude beyond
    what an XL330 could ever balance).
    """

    offset_delta_rad: float
    amplitude_ratio: float
    warnings: tuple[str, ...]
    reliable: bool = True


def balance_and_friction(edge_low_ma: float, edge_high_ma: float) -> tuple[float, float]:
    """Balance current and friction half-width from the two holding-interval edges."""
    if edge_high_ma < edge_low_ma:
        raise ValueError("upper edge must be >= lower edge")
    return (edge_low_ma + edge_high_ma) / 2.0, (edge_high_ma - edge_low_ma) / 2.0


def fit_sine(theta_rad: Sequence[float], values: Sequence[float]) -> SineFit:
    """Least-squares fit of `a*sin(theta) + b*cos(theta) + c`."""
    theta = np.asarray(theta_rad, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if theta.shape != y.shape or theta.ndim != 1:
        raise ValueError("theta and values must be equal-length 1D sequences")
    if theta.size < 3:
        raise ValueError("need at least 3 poses to fit amplitude, phase, and intercept")
    design = np.column_stack([np.sin(theta), np.cos(theta), np.ones_like(theta)])
    coeffs, _, _, singular = np.linalg.lstsq(design, y, rcond=None)
    a, b, c = (float(v) for v in coeffs)
    residual = y - design @ coeffs
    rms = float(np.sqrt(np.mean(residual**2)))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    # a*sin(t) + b*cos(t) = A*sin(t + phi), A >= 0
    amplitude = math.hypot(a, b)
    phase = math.atan2(b, a)
    return SineFit(
        amplitude=amplitude,
        phase_rad=phase,
        intercept=c,
        rms_residual=rms,
        condition_number=condition,
        n_points=int(theta.size),
    )


@dataclass(frozen=True)
class EdgeFit:
    """Sine fit plus stiction fitted directly from breakaway-edge events.

    Each event is one edge of the holding interval observed at its *own* angle
    (`I_edge = A*sin(theta+phi) + c + friction*d`, d = -1 lower / +1 upper), so
    the joint is allowed to drift between probes — the same per-event trick as
    `torque_identification.py` on `feat/pwm-control-so101`. This matters on the
    GELLO: every fall/drive probe moves the joint, so a fixed-pose balance
    midpoint chases a moving target while edge events stay exact.
    """

    sine: SineFit
    friction_ma: float


def fit_edge_events(
    theta_rad: Sequence[float],
    edge_current_ma: Sequence[float],
    directions: Sequence[int],
) -> EdgeFit:
    theta = np.asarray(theta_rad, dtype=np.float64)
    current = np.asarray(edge_current_ma, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    if not (theta.shape == current.shape == d.shape) or theta.ndim != 1:
        raise ValueError("theta, currents, and directions must be equal-length 1D")
    if theta.size < 4:
        raise ValueError("need at least 4 edge events to fit A, phase, intercept, friction")
    if not (np.any(d > 0) and np.any(d < 0)):
        raise ValueError("need edge events from both directions to separate friction")
    design = np.column_stack([np.sin(theta), np.cos(theta), np.ones_like(theta), d])
    coeffs, _, _, singular = np.linalg.lstsq(design, current, rcond=None)
    a, b, c, friction = (float(v) for v in coeffs)
    residual = current - design @ coeffs
    rms = float(np.sqrt(np.mean(residual**2)))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    sine = SineFit(
        amplitude=math.hypot(a, b),
        phase_rad=math.atan2(b, a),
        intercept=c,
        rms_residual=rms,
        condition_number=condition,
        n_points=int(theta.size),
    )
    return EdgeFit(sine=sine, friction_ma=friction)


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class ApexSolution:
    """Joint offset solved from a physically captured over-center (apex) pose.

    At the apex the joint's gravity torque is exactly zero and the *holding
    current has negative slope* (unstable equilibrium: past the apex, gravity
    pulls away, so the required current flips sign). A joint's torque is a pure
    sinusoid in its own offset, so two model evaluations give both zeros
    analytically; the slope test picks the apex zero over the hanging
    (stable, COM-down) zero half a turn away.
    """

    offset_apex_rad: float
    offset_hanging_rad: float
    amplitude_nm: float
    distal_torque_nm: float = 0.0


def solve_apex_offset(
    model,
    q_yam: Sequence[float],
    signs: Sequence[int],
    offsets_rad: Sequence[float],
    joint_index: int,
    lock_distal: bool = True,
) -> ApexSolution:
    """Offset placing the model's torque zero at the captured apex pose.

    `q_yam` is the full six-joint pose (follower radians) captured while the
    operator balances the joint under test at its physical over-center pose.

    Offsets compound down the kinematic chain, so naively scanning this
    joint's offset would rotate every distal joint too — including ones whose
    orientation is already calibrated — and the solve would zero the torque on
    a wrongly-shaped arm. With ``lock_distal`` (default) the next joint's
    offset is co-varied by the opposite amount so all distal absolute angles
    stay pinned: the scan then only moves this joint's own contribution and
    one capture yields its true offset, with the pinned distal torque as a
    constant. The caller must then keep the composed sums fixed (subtract the
    found offset delta from the next joint's offset), which
    ``gello_gravity_sysid --capture-apex`` does automatically.
    """
    q = np.asarray(q_yam, dtype=np.float64)
    signs_arr = np.asarray(signs, dtype=np.float64)
    base = np.asarray(offsets_rad, dtype=np.float64)

    def torque_with(offset: float, q_pose: np.ndarray = q) -> float:
        offs = base.copy()
        offs[joint_index] = offset
        if lock_distal and joint_index + 1 < offs.shape[0]:
            offs[joint_index + 1] = base[joint_index + 1] - (
                offset - base[joint_index]
            )
        q_urdf = signs_arr * q_pose + offs
        return float(model.gravity_torques(q_urdf)[joint_index])

    # tau(o) = a*cos(o) + b*sin(o) + d  (d = the pinned distal-chain torque).
    tau_0 = torque_with(0.0)
    tau_90 = torque_with(math.pi / 2.0)
    tau_180 = torque_with(math.pi)
    a = (tau_0 - tau_180) / 2.0
    d = (tau_0 + tau_180) / 2.0
    b = tau_90 - d
    amplitude = math.hypot(a, b)
    if amplitude < 1e-4:
        raise ValueError(
            "no gravity signal for this joint at the captured pose; "
            "the distal chain is (modeled as) weightless or aligned with the axis"
        )
    if abs(d) > amplitude:
        raise ValueError(
            f"the pinned distal-chain torque ({d:+.4f} Nm) exceeds this "
            f"joint's own gravity authority ({amplitude:.4f} Nm) at the "
            "captured pose, so no offset can balance it — the distal offsets "
            "are likely wrong; recapture joints tip-to-base first"
        )
    # a*cos(o) + b*sin(o) = A*sin(o + phi), phi = atan2(a, b); solve = -d.
    phi = math.atan2(a, b)
    base_angle = math.asin(max(-1.0, min(1.0, -d / amplitude)))
    candidates = (
        wrap_angle(base_angle - phi),
        wrap_angle(math.pi - base_angle - phi),
    )

    def is_apex(offset: float) -> bool:
        eps = 1e-3
        plus, minus = q.copy(), q.copy()
        plus[joint_index] += eps
        minus[joint_index] -= eps
        dtau_dq = (torque_with(offset, plus) - torque_with(offset, minus)) / (2 * eps)
        # Motor-frame holding current is sign * tau; apex requires its slope
        # over the motor angle to be negative.
        return float(signs[joint_index]) * dtau_dq < 0.0

    apex, hanging = (
        (candidates[0], candidates[1])
        if is_apex(candidates[0])
        else (candidates[1], candidates[0])
    )
    return ApexSolution(
        offset_apex_rad=apex,
        offset_hanging_rad=hanging,
        amplitude_nm=amplitude,
        distal_torque_nm=d,
    )


def suggest_corrections(
    measured: SineFit,
    model: SineFit,
    *,
    max_condition: float = 50.0,
    min_amplitude_ma: float = 15.0,
) -> Corrections:
    """Offset delta and mass scale that make the model match the measurements.

    The model matches reality when it is evaluated at `theta + delta`:
        A_m*sin(theta + delta + phi_m) == A_s*sin(theta + phi_s)
    so `delta = phi_s - phi_m` and the distal masses scale by `A_s / A_m`.
    """
    warnings: list[str] = []
    reliable = True
    if measured.condition_number > max_condition:
        reliable = False
        warnings.append(
            f"measured fit is ill-conditioned ({measured.condition_number:.0f}); "
            "the poses did not vary this joint's angle enough — spread them out"
        )
    if measured.amplitude > 2500.0:
        # The XL330 stalls at ~1470 mA; a larger fitted balance amplitude is a
        # degenerate-fit artifact, not physics.
        reliable = False
        warnings.append(
            f"fitted amplitude {measured.amplitude:.0f} mA exceeds anything an "
            "XL330 could balance; the fit is an extrapolation artifact"
        )
    if measured.amplitude < min_amplitude_ma:
        warnings.append(
            f"measured amplitude {measured.amplitude:.1f} mA is close to the "
            "friction floor; phase (offset) estimate is unreliable for this joint"
        )
    if model.amplitude <= 1e-9:
        warnings.append(
            "model predicts ~zero gravity torque for this joint at these poses; "
            "amplitude ratio is meaningless"
        )
        ratio = float("nan")
    else:
        ratio = measured.amplitude / model.amplitude
    if measured.rms_residual > 0.35 * max(measured.amplitude, 1.0):
        warnings.append(
            "measured balance currents deviate strongly from a sine "
            f"(rms {measured.rms_residual:.1f} mA vs amplitude "
            f"{measured.amplitude:.1f} mA); geometry may differ by more than a "
            "rotation (translation offsets), or the poses were not quasistatic"
        )
    return Corrections(
        offset_delta_rad=wrap_angle(measured.phase_rad - model.phase_rad),
        amplitude_ratio=ratio,
        warnings=tuple(warnings),
        reliable=reliable,
    )


def apply_offset_delta(
    offsets_rad: Sequence[float], joint_index: int, delta_rad: float
) -> tuple[float, ...]:
    """New `gravity_joint_offsets_rad` tuple with `delta` added to one joint."""
    updated = list(float(v) for v in offsets_rad)
    updated[joint_index] = wrap_angle(updated[joint_index] + delta_rad)
    return tuple(updated)
