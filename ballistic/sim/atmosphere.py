"""Simplified ISA (International Standard Atmosphere) up to ~86 km.

Returns density (kg/m^3), pressure (Pa), temperature (K), and speed of sound
(m/s) as a function of geometric altitude above the WGS84 ellipsoid. We treat
geometric and geopotential altitude as equal -- accurate to better than 0.5%
within the troposphere/stratosphere range that ballistic rounds occupy.
"""

from __future__ import annotations

import numpy as np

R_AIR = 287.05287       # J/(kg*K)
GAMMA = 1.4
G0 = 9.80665            # m/s^2 (ISA reference)

# (h_base, T_base, p_base, lapse) per ISA layer
_LAYERS = [
    (0.0,     288.15,    101325.0,    -6.5e-3),
    (11000.0, 216.65,    22632.06,     0.0),
    (20000.0, 216.65,    5474.889,     1.0e-3),
    (32000.0, 228.65,    868.0187,     2.8e-3),
    (47000.0, 270.65,    110.9063,     0.0),
    (51000.0, 270.65,    66.93887,    -2.8e-3),
    (71000.0, 214.65,    3.956420,    -2.0e-3),
]


def _layer(h: float) -> tuple[float, float, float, float]:
    for i in range(len(_LAYERS) - 1, -1, -1):
        if h >= _LAYERS[i][0]:
            return _LAYERS[i]
    return _LAYERS[0]


def atmosphere(alt_m: float) -> tuple[float, float, float, float]:
    """Return (rho, pressure, temperature, speed_of_sound) at geometric altitude."""
    if alt_m < 0.0:
        alt_m = 0.0
    if alt_m > 86000.0:
        alt_m = 86000.0
    h_b, t_b, p_b, l_b = _layer(alt_m)
    if l_b == 0.0:
        t = t_b
        p = p_b * np.exp(-G0 * (alt_m - h_b) / (R_AIR * t_b))
    else:
        t = t_b + l_b * (alt_m - h_b)
        p = p_b * (t / t_b) ** (-G0 / (R_AIR * l_b))
    rho = p / (R_AIR * t)
    a = np.sqrt(GAMMA * R_AIR * t)
    return float(rho), float(p), float(t), float(a)
