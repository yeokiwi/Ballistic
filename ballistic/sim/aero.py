"""Simple aerodynamic coefficient lookup.

Each round carries a Mach -> Cd table; we linearly interpolate, clamping at
the endpoints. Values are not from a ballistics reference; they are tuned to
sit in the right magnitude band for engineering-grade visualisations.
"""

from __future__ import annotations

import numpy as np


def cd_lookup(mach: float, table: np.ndarray) -> float:
    """Interpolate Cd from a 2-column [mach, cd] table. Clamps at endpoints."""
    m = float(mach)
    machs = table[:, 0]
    cds = table[:, 1]
    if m <= machs[0]:
        return float(cds[0])
    if m >= machs[-1]:
        return float(cds[-1])
    return float(np.interp(m, machs, cds))


# Mach, Cd. Fin-stabilised projectile with low subsonic drag, transonic spike,
# tapering supersonic drag.
CD_FIN = np.array([
    [0.0, 0.18],
    [0.5, 0.18],
    [0.8, 0.20],
    [1.0, 0.40],
    [1.2, 0.42],
    [2.0, 0.32],
    [3.0, 0.28],
    [5.0, 0.26],
])

# Spin-stabilised artillery shell -- slightly higher base drag.
CD_SPIN = np.array([
    [0.0, 0.20],
    [0.5, 0.21],
    [0.8, 0.23],
    [1.0, 0.46],
    [1.2, 0.48],
    [2.0, 0.36],
    [3.0, 0.31],
    [5.0, 0.29],
])
