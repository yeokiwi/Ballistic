"""Simulator command-line entry point.

Example:
    python -m ballistic.sim.main --round artillery \
        --lat 1.3521 --lon 103.8198 --alt 50 \
        --azimuth 90 --elevation 45 --roll 0
"""

from __future__ import annotations

import argparse

import numpy as np

from . import rounds
from .dynamics import matrix_to_quat
from .earth import lla_to_ecef, ned_basis
from .integrator import SimConfig, run
from .telemetry import UdpTelemetry


def attitude_to_R_b2n(azimuth_deg: float, elevation_deg: float, roll_deg: float) -> np.ndarray:
    """3-2-1 (yaw-pitch-roll) Euler -> body->NED rotation matrix.

    azimuth: heading from North, increasing toward East (degrees)
    elevation: nose-up angle above the local horizon (degrees)
    roll: bank about body x-axis (degrees)
    """
    psi = np.deg2rad(azimuth_deg)
    theta = np.deg2rad(elevation_deg)
    phi = np.deg2rad(roll_deg)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cphi, sphi = np.cos(phi), np.sin(phi)
    Rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[ctheta, 0.0, stheta], [0.0, 1.0, 0.0], [-stheta, 0.0, ctheta]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cphi, -sphi], [0.0, sphi, cphi]])
    return Rz @ Ry @ Rx


def build_initial_state(spec, lat: float, lon: float, alt: float,
                        az: float, el: float, roll: float) -> np.ndarray:
    r_e = lla_to_ecef(lat, lon, alt)
    R_b2n = attitude_to_R_b2n(az, el, roll)
    R_n2e = ned_basis(lat, lon)
    R_b2e = R_n2e @ R_b2n
    q_b2e = matrix_to_quat(R_b2e)
    v_b = np.array([spec.v0, 0.0, 0.0])
    v_e = R_b2e @ v_b
    omega_b = np.array([spec.spin0, 0.0, 0.0])
    x = np.empty(13)
    x[0:3] = r_e
    x[3:6] = v_e
    x[6:10] = q_b2e
    x[10:13] = omega_b
    return x


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="6DOF ballistic round simulator")
    p.add_argument("--round", required=True, choices=sorted(rounds.REGISTRY.keys()))
    p.add_argument("--lat", type=float, required=True, help="launch latitude, deg")
    p.add_argument("--lon", type=float, required=True, help="launch longitude, deg")
    p.add_argument("--alt", type=float, default=0.0, help="launch altitude above ellipsoid, m")
    p.add_argument("--azimuth", type=float, required=True, help="heading from N, deg")
    p.add_argument("--elevation", type=float, required=True, help="elevation above horizon, deg")
    p.add_argument("--roll", type=float, default=0.0, help="roll about body x, deg")
    p.add_argument("--host", default="127.0.0.1", help="GCS UDP host")
    p.add_argument("--port", type=int, default=51000, help="GCS UDP port")
    p.add_argument("--rate", type=float, default=50.0, help="telemetry Hz")
    p.add_argument("--dt", type=float, default=1e-3, help="integration step, s")
    p.add_argument("--speed", type=float, default=1.0, help="real-time multiplier")
    p.add_argument("--max-time", type=float, default=600.0, help="safety cap, s")
    p.add_argument("--ground-alt", type=float, default=None,
                   help="impact altitude (default = launch alt)")
    p.add_argument("--quiet", action="store_true", help="suppress per-packet stdout")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = rounds.get(args.round)
    x0 = build_initial_state(spec, args.lat, args.lon, args.alt,
                             args.azimuth, args.elevation, args.roll)
    cfg = SimConfig(
        dt=args.dt,
        telemetry_hz=args.rate,
        speed=args.speed,
        max_time=args.max_time,
        ground_alt=args.alt if args.ground_alt is None else args.ground_alt,
    )
    sender = UdpTelemetry(args.host, args.port)
    print(f"[sim] {spec.name}: launching from "
          f"({args.lat:.5f}, {args.lon:.5f}, {args.alt:.1f} m) "
          f"az={args.azimuth} el={args.elevation} roll={args.roll}")
    print(f"[sim] sending UDP -> {args.host}:{args.port} @ {args.rate} Hz")
    last_print = 0.0
    try:
        for pkt in run(x0, spec, cfg, sender.send):
            if args.quiet:
                continue
            if pkt.t_sim - last_print >= 0.5 or pkt.phase == 3:
                last_print = pkt.t_sim
                v_mag = (pkt.v_n ** 2 + pkt.v_e ** 2 + pkt.v_d ** 2) ** 0.5
                print(f"  t={pkt.t_sim:6.2f}s  alt={pkt.alt:8.1f}m  "
                      f"v={v_mag:6.1f}m/s  phase={pkt.phase}")
    except KeyboardInterrupt:
        print("[sim] interrupted")
        return 130
    finally:
        sender.close()
    print("[sim] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
