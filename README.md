# Ballistic Round Simulator + GCS

A Python 6DOF ballistic-round simulator paired with a PySide6 desktop ground
control station (GCS). The simulator integrates rigid-body equations of
motion in the ECEF rotating frame and streams real-time position/attitude
telemetry over UDP. The GCS draws the trajectory in 3D (ENU around the
launch site, with a ground grid) and shows the round body's attitude in a
second 3D view.

Three rounds are built in:

| key | round | mass | v0 | notes |
|---|---|---|---|---|
| `mortar` | 120 mm fin-stabilised mortar | 13.0 kg | 318 m/s | unpowered |
| `rocket` | 127 mm fin-stabilised rocket | 50.0 kg | 50 m/s + thrust | 25 kN x 1.5 s boost |
| `artillery` | 155 mm spin-stabilised shell | 43.2 kg | 684 m/s | 270 rad/s spin |

Aerodynamic coefficients and round parameters are open-source approximations
intended to reproduce realistic flight times and ranges, not to match
ballistic firing tables.

## Install

```
pip install -r requirements.txt
```

PySide6 needs the usual desktop OpenGL / Qt platform libs. On Debian/Ubuntu:

```
sudo apt install libegl1 libgl1 libxkbcommon0 libdbus-1-3 libfontconfig1 \
    libxcb-cursor0 libnss3 libxcomposite1 libxdamage1 libxrandr2
```

## Run

Terminal 1 -- start the GCS:

```
python -m ballistic.gcs.main --port 51000
```

Terminal 2 -- launch a round:

```
python -m ballistic.sim.main \
    --round artillery \
    --lat 1.3521 --lon 103.8198 --alt 50 \
    --azimuth 90 --elevation 45 --roll 0 \
    --host 127.0.0.1 --port 51000
```

The left pane shows the trajectory in 3D in a local ENU frame anchored at
the launch site: the launch is a yellow dot at the origin, the round is
red, the trail is cyan, and an orange drop-line tracks straight down to
the ground grid. Mouse: left-drag orbits the camera, mid-drag (or
shift-drag) pans, the wheel zooms. The right pane shows the round body's
attitude in the same ENU convention (green = East, red = North,
blue = Up). The status bar shows TOF, altitude, ground offset, speed and
phase.

### CLI options (simulator)

| flag | default | meaning |
|---|---|---|
| `--round` | required | `mortar` / `rocket` / `artillery` |
| `--lat --lon` | required | launch geodetic latitude/longitude (deg) |
| `--alt` | 0 | launch altitude above WGS84 ellipsoid (m) |
| `--azimuth` | required | heading from North toward East (deg) |
| `--elevation` | required | nose-up angle above local horizon (deg) |
| `--roll` | 0 | bank about body x (deg) |
| `--host --port` | 127.0.0.1:51000 | GCS UDP destination |
| `--rate` | 50 | telemetry Hz |
| `--dt` | 1e-3 | RK4 step (s) |
| `--speed` | 1.0 | wall-clock multiplier (>1 = faster than real time, 0 = no pacing) |
| `--max-time` | 600 | safety cap (s) |
| `--ground-alt` | =launch alt | impact altitude (m) |

### Examples

120 mm mortar at QE 80 deg from Singapore, fired north:
```
python -m ballistic.sim.main --round mortar \
    --lat 1.3521 --lon 103.8198 --alt 30 \
    --azimuth 0 --elevation 80
```

127 mm rocket fired east at QE 35 deg, fast-forward 5x:
```
python -m ballistic.sim.main --round rocket \
    --lat 1.3521 --lon 103.8198 --alt 30 \
    --azimuth 90 --elevation 35 --speed 5
```

## Architecture

```
ballistic/
  common/packet.py     fixed-size UDP wire format (struct, 78 bytes)
  sim/
    earth.py           WGS84 LLA<->ECEF, NED basis, point-mass+J2 gravity
    atmosphere.py      ISA up to 86 km
    aero.py            Mach->Cd interpolation
    rounds.py          RoundSpec + 3 munitions
    dynamics.py        6DOF state derivative in ECEF (Coriolis+centrifugal)
    integrator.py      RK4 + telemetry + wall-clock pacing
    telemetry.py       UDP sender
    main.py            simulator CLI
  gcs/
    udp_listener.py    QThread receiving telemetry
    trajectory_view.py pyqtgraph.opengl 3D trail in local ENU + ground grid
    attitude_view.py   pyqtgraph.opengl 3D body view (ENU axes)
    main.py            MainWindow (trajectory | attitude | status)
tests/                 pytest suite (earth/aero/dynamics/packet)
```

### State and frames

The integrator state is `[r_ecef, v_ecef, q_b2e, omega_b]` (13 vector).
Quaternions use the Hamilton convention with `(w, x, y, z)` order; `q_b2e`
maps body axes to ECEF. Pseudo-forces (Coriolis + centrifugal) are added
explicitly because the integrator runs in the ECEF rotating frame. Telemetry
re-projects attitude into the local NED basis at the round's current LLA so
the GCS body view rotates relative to the local horizon.

### Wire format

Single fixed-size UDP datagram per sample. See `ballistic/common/packet.py`
for the exact layout.

## Tests

```
python -m pytest tests/
```

Covers WGS84 round trip, gravity magnitude, Cd table behaviour, UDP packet
round trip, and a per-round trajectory envelope (range + time-of-flight).
