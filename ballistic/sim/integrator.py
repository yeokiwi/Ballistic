"""Fixed-step RK4 integrator with wall-clock pacing and telemetry emission."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from ..common.packet import (
    PHASE_BALLISTIC,
    PHASE_BOOST,
    PHASE_IMPACT,
    PHASE_PRELAUNCH,
    Telemetry,
)
from .dynamics import (
    quat_normalize,
    state_derivative,
    state_to_telemetry,
)
from .earth import ecef_to_lla
from .rounds import RoundSpec


@dataclass
class SimConfig:
    dt: float = 1e-3              # integration step, s
    telemetry_hz: float = 50.0    # telemetry packet rate
    speed: float = 1.0            # wall-clock multiplier (1.0 = real-time)
    max_time: float = 600.0       # safety cap, s
    ground_alt: float = 0.0       # impact when alt drops below this, m


def rk4_step(t: float, x: np.ndarray, dt: float, spec: RoundSpec) -> np.ndarray:
    k1 = state_derivative(t, x, spec)
    k2 = state_derivative(t + 0.5 * dt, x + 0.5 * dt * k1, spec)
    k3 = state_derivative(t + 0.5 * dt, x + 0.5 * dt * k2, spec)
    k4 = state_derivative(t + dt, x + dt * k3, spec)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[6:10] = quat_normalize(x_next[6:10])
    return x_next


def _phase(t: float, spec: RoundSpec, ascending: bool) -> int:
    if spec.thrust > 0.0 and t < spec.burn_time:
        return PHASE_BOOST
    return PHASE_BALLISTIC


def run(
    x0: np.ndarray,
    spec: RoundSpec,
    cfg: SimConfig,
    on_telemetry: Callable[[Telemetry], None],
) -> Iterator[Telemetry]:
    """Step the simulation in real time, calling on_telemetry per packet.

    Yields each Telemetry sample as it is produced. Terminates when the
    projectile hits the ground or max_time elapses.
    """
    x = x0.copy()
    t = 0.0
    step_per_packet = max(1, int(round(1.0 / (cfg.telemetry_hz * cfg.dt))))
    wall_start = time.perf_counter()
    sample_idx = 0

    # Emit initial state
    pkt = state_to_telemetry(t, x, spec, PHASE_PRELAUNCH)
    on_telemetry(pkt)
    yield pkt

    prev_alt = pkt.alt
    while t < cfg.max_time:
        for _ in range(step_per_packet):
            x = rk4_step(t, x, cfg.dt, spec)
            t += cfg.dt
        sample_idx += 1

        _, _, alt = ecef_to_lla(x[0:3])
        ascending = alt > prev_alt
        prev_alt = alt

        if alt <= cfg.ground_alt and t > 0.1:
            pkt = state_to_telemetry(t, x, spec, PHASE_IMPACT)
            on_telemetry(pkt)
            yield pkt
            return

        phase = _phase(t, spec, ascending)
        pkt = state_to_telemetry(t, x, spec, phase)
        on_telemetry(pkt)
        yield pkt

        if cfg.speed > 0:
            target_wall = wall_start + (t / cfg.speed)
            sleep_for = target_wall - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
