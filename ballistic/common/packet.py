"""UDP telemetry wire format.

A single fixed-size datagram per sample. Little-endian. Layout:

    version  : u8           (=1)
    t_sim    : f64          seconds since launch
    lat      : f64          WGS84 latitude, deg
    lon      : f64          WGS84 longitude, deg
    alt      : f64          height above ellipsoid, m
    q_w/x/y/z: f32 x4       body -> local NED quaternion (Hamilton, w first)
    v_n/e/d  : f32 x3       NED velocity, m/s
    p/q/r    : f32 x3       body angular velocity, rad/s
    phase    : u8           0=prelaunch 1=boost 2=ballistic 3=impact
    round_id : u8           0=mortar 1=rocket 2=artillery
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PHASE_PRELAUNCH = 0
PHASE_BOOST = 1
PHASE_BALLISTIC = 2
PHASE_IMPACT = 3

ROUND_MORTAR = 0
ROUND_ROCKET = 1
ROUND_ARTILLERY = 2

_FMT = "<B d ddd ffff fff fff B B"
_STRUCT = struct.Struct(_FMT)
PACKET_SIZE = _STRUCT.size  # 78 bytes
VERSION = 1


@dataclass
class Telemetry:
    t_sim: float
    lat: float
    lon: float
    alt: float
    q_w: float
    q_x: float
    q_y: float
    q_z: float
    v_n: float
    v_e: float
    v_d: float
    p: float
    q: float
    r: float
    phase: int
    round_id: int

    def pack(self) -> bytes:
        return _STRUCT.pack(
            VERSION,
            self.t_sim,
            self.lat, self.lon, self.alt,
            self.q_w, self.q_x, self.q_y, self.q_z,
            self.v_n, self.v_e, self.v_d,
            self.p, self.q, self.r,
            int(self.phase), int(self.round_id),
        )

    @classmethod
    def unpack(cls, data: bytes) -> "Telemetry":
        if len(data) != PACKET_SIZE:
            raise ValueError(f"expected {PACKET_SIZE} bytes, got {len(data)}")
        fields = _STRUCT.unpack(data)
        version = fields[0]
        if version != VERSION:
            raise ValueError(f"unsupported packet version {version}")
        return cls(*fields[1:])
