import numpy as np

from ballistic.sim.earth import (
    A_WGS84,
    ecef_to_lla,
    gravity_ecef,
    lla_to_ecef,
    ned_basis,
)


def test_lla_ecef_round_trip():
    cases = [
        (0.0, 0.0, 0.0),
        (1.3521, 103.8198, 50.0),
        (40.7128, -74.0060, 100.0),
        (-33.8688, 151.2093, 0.0),
        (89.5, 12.0, 1234.5),
    ]
    for lat, lon, alt in cases:
        r = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(r)
        assert abs(lat2 - lat) < 1e-6
        assert abs(lon2 - lon) < 1e-6
        assert abs(alt2 - alt) < 1e-3


def test_gravity_magnitude_at_surface():
    r = lla_to_ecef(0.0, 0.0, 0.0)
    g = np.linalg.norm(gravity_ecef(r))
    # Equatorial surface: ~9.8 m/s^2 (note: this is gravitational only, no
    # centrifugal -- the "weight" felt is g - centrifugal ~ 9.78 m/s^2).
    assert 9.7 < g < 9.9


def test_ned_basis_orthonormal():
    R = ned_basis(45.0, 30.0)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    # Down vector should point inward (towards earth centre)
    r = lla_to_ecef(45.0, 30.0, 0.0)
    down = R[:, 2]
    assert np.dot(down, r) < 0.0
