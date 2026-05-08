"""6DOF rigid-body dynamics in the ECEF rotating frame.

State (13-vector):
    [0:3]   r_ecef  position, m
    [3:6]   v_ecef  velocity in ECEF (rotating) frame, m/s
    [6:10]  q_b2e   body -> ECEF quaternion (w, x, y, z, Hamilton)
    [10:13] omega_b body angular velocity in body frame, rad/s

Quaternion convention is body-to-ECEF: a body-frame vector v_b is rotated to
ECEF by v_e = R(q) @ v_b.
"""

from __future__ import annotations

import numpy as np

from . import aero
from .atmosphere import atmosphere
from ..common.packet import Telemetry
from .earth import (
    OMEGA_VEC,
    ecef_to_lla,
    geodetic_altitude,
    gravity_ecef,
    ned_basis,
)
from .rounds import RoundSpec


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (xx + zz), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (xx + yy)],
    ])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_derivative(q: np.ndarray, omega_b: np.ndarray) -> np.ndarray:
    """dq/dt for body-rate omega_b expressed in body frame."""
    w_quat = np.array([0.0, omega_b[0], omega_b[1], omega_b[2]])
    return 0.5 * quat_mul(q, w_quat)


def state_derivative(t: float, x: np.ndarray, spec: RoundSpec) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]
    q = x[6:10]
    omega_b = x[10:13]

    R_b2e = quat_to_matrix(q)
    R_e2b = R_b2e.T

    # Air-relative velocity. In the ECEF rotating frame, the atmosphere is
    # stationary, so v_rel = v.
    v_rel_e = v
    v_rel_mag = float(np.linalg.norm(v_rel_e))
    v_rel_b = R_e2b @ v_rel_e

    # Atmosphere at current altitude
    alt = geodetic_altitude(r)
    rho, _p, _T, a_sound = atmosphere(alt)
    mach = v_rel_mag / a_sound if a_sound > 0 else 0.0

    # --- Forces (all in ECEF) ---
    # Drag opposing relative wind
    cd = aero.cd_lookup(mach, spec.cd_table)
    A = spec.reference_area
    f_drag_e = -0.5 * rho * v_rel_mag * cd * A * v_rel_e

    # Thrust along body x-axis
    thrust = spec.thrust_at(t)
    f_thrust_e = R_b2e @ np.array([thrust, 0.0, 0.0]) if thrust > 0.0 else np.zeros(3)

    # Gravity
    g_e = gravity_ecef(r)

    # Pseudo-forces (ECEF rotating frame)
    coriolis = -2.0 * np.cross(OMEGA_VEC, v)
    centrifugal = -np.cross(OMEGA_VEC, np.cross(OMEGA_VEC, r))

    mass = spec.mass_at(t)
    a_e = (f_drag_e + f_thrust_e) / mass + g_e + coriolis + centrifugal

    # --- Moments (body frame), standard aerodynamic formulation ---
    # qbar = 0.5 rho V^2 (dynamic pressure), S = ref. area, d = caliber.
    # Restoring weathercock: M = qbar * S * d * Cm_alpha * (v_hat_b x e_x_b)
    #   v_hat x e_x = (0, vz, -vy)/V, so for Cm_alpha < 0 a +alpha (vz>0)
    #   produces a nose-down (negative My) restoring moment.
    # Pitch damping: qbar * S * d * Cmq * (d/2V) * (0, q, r)  (negative Cmq)
    # Roll damping:  qbar * S * d * Clp * (d/2V) * (p, 0, 0)
    qbar_p = 0.5 * rho * v_rel_mag * v_rel_mag
    S = spec.reference_area
    d = spec.diameter
    if v_rel_mag > 1e-3:
        weather = qbar_p * S * d * spec.cm_alpha * np.array(
            [0.0, v_rel_b[2] / v_rel_mag, -v_rel_b[1] / v_rel_mag]
        )
        damp_factor = qbar_p * S * d * (d / (2.0 * v_rel_mag))
    else:
        weather = np.zeros(3)
        damp_factor = 0.0
    pitch_damp = damp_factor * spec.cmq * np.array([0.0, omega_b[1], omega_b[2]])
    roll_damp = damp_factor * spec.clp * np.array([omega_b[0], 0.0, 0.0])

    M_b = weather + pitch_damp + roll_damp

    # Euler's equation, axisymmetric inertia
    ixx, iyy = spec.inertia(t)
    I_b = np.diag([ixx, iyy, iyy])
    Iw = I_b @ omega_b
    omega_dot = np.linalg.solve(I_b, M_b - np.cross(omega_b, Iw))

    q_dot = quat_derivative(q, omega_b)

    dx = np.empty(13)
    dx[0:3] = v
    dx[3:6] = a_e
    dx[6:10] = q_dot
    dx[10:13] = omega_dot
    return dx


def state_to_telemetry(t: float, x: np.ndarray, spec: RoundSpec, phase: int) -> Telemetry:
    """Convert raw state to a Telemetry record (lat/lon/alt + NED quaternion)."""
    r = x[0:3]
    v = x[3:6]
    q = quat_normalize(x[6:10])
    omega_b = x[10:13]

    lat, lon, alt = ecef_to_lla(r)
    R_e2n = ned_basis(lat, lon).T
    v_n = R_e2n @ v

    # body -> NED rotation: R_b2n = R_e2n @ R_b2e
    R_b2e = quat_to_matrix(q)
    R_b2n = R_e2n @ R_b2e
    q_b2n = matrix_to_quat(R_b2n)

    return Telemetry(
        t_sim=float(t),
        lat=float(lat), lon=float(lon), alt=float(alt),
        q_w=float(q_b2n[0]), q_x=float(q_b2n[1]),
        q_y=float(q_b2n[2]), q_z=float(q_b2n[3]),
        v_n=float(v_n[0]), v_e=float(v_n[1]), v_d=float(v_n[2]),
        p=float(omega_b[0]), q=float(omega_b[1]), r=float(omega_b[2]),
        phase=int(phase),
        round_id=int(spec.round_id),
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> Hamilton quaternion (w, x, y, z). Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return quat_normalize(q)
