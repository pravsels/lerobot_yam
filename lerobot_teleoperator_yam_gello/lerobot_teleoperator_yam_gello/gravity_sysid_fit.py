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
    """Suggested config changes derived from measured vs model fits."""

    offset_delta_rad: float
    amplitude_ratio: float
    warnings: tuple[str, ...]


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


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


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
    if measured.condition_number > max_condition:
        warnings.append(
            f"measured fit is ill-conditioned ({measured.condition_number:.0f}); "
            "the poses did not vary this joint's angle enough — spread them out"
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
    )


def apply_offset_delta(
    offsets_rad: Sequence[float], joint_index: int, delta_rad: float
) -> tuple[float, ...]:
    """New `gravity_joint_offsets_rad` tuple with `delta` added to one joint."""
    updated = list(float(v) for v in offsets_rad)
    updated[joint_index] = wrap_angle(updated[joint_index] + delta_rad)
    return tuple(updated)
