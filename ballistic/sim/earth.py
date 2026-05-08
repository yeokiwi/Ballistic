"""WGS84 geodesy + simple Earth gravity model.

All vectors are numpy arrays in metres, radians, or m/s as appropriate.
ECEF = Earth-Centred Earth-Fixed (rotates with Earth).
ENU/NED bases are right-handed local tangent frames at a given LLA.
"""

from __future__ import annotations

import numpy as np

# WGS84 ellipsoid
A_WGS84 = 6378137.0                 # semi-major axis, m
F_WGS84 = 1.0 / 298.257223563        # flattening
B_WGS84 = A_WGS84 * (1.0 - F_WGS84)  # semi-minor axis, m
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84) # first eccentricity squared
EP2_WGS84 = E2_WGS84 / (1.0 - E2_WGS84)

# Gravity / rotation
GM_EARTH = 3.986004418e14            # m^3/s^2
J2_EARTH = 1.08263e-3
OMEGA_EARTH = 7.2921150e-5           # rad/s
OMEGA_VEC = np.array([0.0, 0.0, OMEGA_EARTH])


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Geodetic latitude/longitude/height -> ECEF position (m)."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * np.cos(lon)
    y = (n + alt_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - E2_WGS84) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_lla(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """ECEF (m) -> (lat_deg, lon_deg, alt_m). Closed-form Bowring."""
    x, y, z = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    if p < 1e-9:
        # On the spin axis
        lat = np.pi / 2.0 if z >= 0 else -np.pi / 2.0
        alt = abs(z) - B_WGS84
        return np.rad2deg(lat), np.rad2deg(lon), alt
    theta = np.arctan2(z * A_WGS84, p * B_WGS84)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    lat = np.arctan2(z + EP2_WGS84 * B_WGS84 * sin_t ** 3,
                     p - E2_WGS84 * A_WGS84 * cos_t ** 3)
    sin_lat = np.sin(lat)
    n = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sin_lat * sin_lat)
    alt = p / np.cos(lat) - n
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def ned_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return 3x3 matrix whose columns are N, E, D unit vectors expressed in ECEF.

    NED is the local geographic frame: x=North, y=East, z=Down.
    Multiplying by [n,e,d] yields the ECEF vector. Inverse (transpose for
    orthonormal basis) converts an ECEF vector to NED components.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    # North: tangent along increasing latitude
    n = np.array([-sl * co, -sl * so, cl])
    # East: tangent along increasing longitude
    e = np.array([-so, co, 0.0])
    # Down: -outward normal
    d = np.array([-cl * co, -cl * so, -sl])
    return np.column_stack((n, e, d))


def ecef_to_ned(vec_ecef: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Project an ECEF vector into the local NED basis at (lat, lon)."""
    return ned_basis(lat_deg, lon_deg).T @ vec_ecef


def ned_to_ecef(vec_ned: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    return ned_basis(lat_deg, lon_deg) @ vec_ned


def gravity_ecef(r_ecef: np.ndarray) -> np.ndarray:
    """Gravitational acceleration (m/s^2) in ECEF, point-mass + J2 zonal.

    Pure gravitational attraction only -- no centrifugal term, since this is
    used inside an ECEF rotating-frame integrator that adds Coriolis +
    centrifugal pseudo-forces explicitly.
    """
    x, y, z = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    if r < 1.0:
        return np.zeros(3)
    mu_r3 = GM_EARTH / (r2 * r)
    # Point-mass term
    g = -mu_r3 * np.array([x, y, z])
    # J2 perturbation
    factor = 1.5 * J2_EARTH * (A_WGS84 / r) ** 2
    z2_r2 = (z * z) / r2
    gx = -mu_r3 * x * factor * (1.0 - 5.0 * z2_r2)
    gy = -mu_r3 * y * factor * (1.0 - 5.0 * z2_r2)
    gz = -mu_r3 * z * factor * (3.0 - 5.0 * z2_r2)
    return g + np.array([gx, gy, gz])


def geodetic_altitude(r_ecef: np.ndarray) -> float:
    return ecef_to_lla(r_ecef)[2]
