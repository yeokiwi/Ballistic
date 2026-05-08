"""Round parameters for the three supported munitions.

Values are open-source approximations -- intended to reproduce realistic
flight times and ranges, not to match ballistic firing tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import aero
from ..common.packet import ROUND_ARTILLERY, ROUND_MORTAR, ROUND_ROCKET


@dataclass(frozen=True)
class RoundSpec:
    name: str
    round_id: int
    mass: float                 # initial mass, kg
    mass_burnout: float         # mass after motor burn (=mass for unpowered), kg
    diameter: float             # m
    length: float               # m
    v0: float                   # initial speed along body x-axis, m/s
    spin0: float                # initial axial spin, rad/s
    cd_table: np.ndarray        # Mach->Cd, shape (n,2)
    thrust: float = 0.0         # N during burn
    burn_time: float = 0.0      # s
    cm_alpha: float = -2.0      # pitching-moment slope per rad (negative = restoring)
    cmq: float = -8.0           # pitch damping derivative (negative)
    clp: float = -0.01          # roll damping derivative (negative)

    @property
    def reference_area(self) -> float:
        r = 0.5 * self.diameter
        return np.pi * r * r

    def mass_at(self, t: float) -> float:
        if self.thrust <= 0.0 or self.burn_time <= 0.0 or t >= self.burn_time:
            return self.mass_burnout if self.thrust > 0.0 else self.mass
        frac = t / self.burn_time
        return self.mass + (self.mass_burnout - self.mass) * frac

    def thrust_at(self, t: float) -> float:
        if self.thrust > 0.0 and t < self.burn_time:
            return self.thrust
        return 0.0

    def inertia(self, t: float) -> tuple[float, float]:
        """Return (I_xx, I_yy) for an axisymmetric solid cylinder approximation."""
        m = self.mass_at(t)
        r = 0.5 * self.diameter
        ixx = 0.5 * m * r * r
        iyy = (1.0 / 12.0) * m * (3.0 * r * r + self.length * self.length)
        return ixx, iyy


MORTAR_120 = RoundSpec(
    name="120mm mortar",
    round_id=ROUND_MORTAR,
    mass=13.0,
    mass_burnout=13.0,
    diameter=0.120,
    length=0.70,
    v0=318.0,
    spin0=30.0,
    cd_table=aero.CD_FIN,
    cm_alpha=-3.0,
    cmq=-12.0,
    clp=-0.02,
)

ROCKET_127 = RoundSpec(
    name="127mm rocket",
    round_id=ROUND_ROCKET,
    mass=50.0,
    mass_burnout=30.0,
    diameter=0.127,
    length=1.85,
    v0=50.0,
    spin0=30.0,
    cd_table=aero.CD_FIN,
    thrust=25000.0,
    burn_time=1.5,
    cm_alpha=-4.0,
    cmq=-15.0,
    clp=-0.02,
)

ARTILLERY_155 = RoundSpec(
    name="155mm artillery",
    round_id=ROUND_ARTILLERY,
    mass=43.2,
    mass_burnout=43.2,
    diameter=0.155,
    length=0.60,
    v0=684.0,
    spin0=270.0,
    cd_table=aero.CD_SPIN,
    cm_alpha=-2.0,
    cmq=-8.0,
    clp=-0.01,
)


REGISTRY: dict[str, RoundSpec] = {
    "mortar": MORTAR_120,
    "rocket": ROCKET_127,
    "artillery": ARTILLERY_155,
}


def get(name: str) -> RoundSpec:
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown round '{name}'; choose from {sorted(REGISTRY)}")
    return REGISTRY[key]
