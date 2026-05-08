import numpy as np
import pytest

from ballistic.sim import rounds
from ballistic.sim.earth import lla_to_ecef
from ballistic.sim.integrator import SimConfig, run
from ballistic.sim.main import build_initial_state


def _fly(spec, lat=0.0, lon=0.0, alt=0.0, az=90.0, el=45.0, roll=0.0):
    x0 = build_initial_state(spec, lat, lon, alt, az, el, roll)
    cfg = SimConfig(dt=2e-3, telemetry_hz=10.0, speed=0.0,
                    max_time=300.0, ground_alt=alt)
    last = None
    apex = 0.0
    for pkt in run(x0, spec, cfg, lambda p: None):
        last = pkt
        apex = max(apex, pkt.alt)
    r0 = lla_to_ecef(lat, lon, alt)
    r1 = lla_to_ecef(last.lat, last.lon, last.alt)
    range_m = float(np.linalg.norm(r1 - r0))
    return last, apex, range_m


@pytest.mark.parametrize("name,range_min,range_max,tof_min,tof_max", [
    ("mortar",     3000.0, 8000.0,  20.0, 60.0),
    ("artillery", 10000.0, 22000.0, 40.0, 90.0),
    ("rocket",    15000.0, 40000.0, 50.0, 130.0),
])
def test_round_range_envelope(name, range_min, range_max, tof_min, tof_max):
    spec = rounds.get(name)
    last, apex, range_m = _fly(spec)
    assert range_min < range_m < range_max, (
        f"{name}: range {range_m:.0f}m outside [{range_min},{range_max}]")
    assert tof_min < last.t_sim < tof_max, (
        f"{name}: TOF {last.t_sim:.1f}s outside [{tof_min},{tof_max}]")
    assert apex > 500.0
    assert last.phase == 3  # impact
