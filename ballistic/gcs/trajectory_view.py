"""Live 3D trajectory view using pyqtgraph.opengl.

Coordinates are local East-North-Up metres around the launch site. Axes:
green = +x East, red = +y North, blue = +z Up. The ground grid lies in
the XY plane at z=0. Mouse: left-drag orbits, mid-drag (or shift-drag)
pans, wheel zooms.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl

from ..sim.earth import lla_to_ecef, ned_basis


class TrajectoryView(gl.GLViewWidget):
    GROUND_HALF_SIZE = 20_000.0   # metres
    GROUND_SPACING = 2_000.0      # metres per minor grid cell
    AXIS_LEN = 2_000.0            # metres

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor((20, 20, 28))
        self.setCameraPosition(distance=20_000.0, azimuth=-60, elevation=25)

        grid = gl.GLGridItem()
        grid.setSize(self.GROUND_HALF_SIZE * 2, self.GROUND_HALF_SIZE * 2)
        grid.setSpacing(self.GROUND_SPACING, self.GROUND_SPACING)
        grid.setColor((130, 130, 160, 160))
        self.addItem(grid)

        # ENU axis triad. +x East (green), +y North (red), +z Up (blue).
        self._add_axis((0, 0, 0), (self.AXIS_LEN, 0, 0), (0.2, 1.0, 0.2, 1.0))
        self._add_axis((0, 0, 0), (0, self.AXIS_LEN, 0), (1.0, 0.2, 0.2, 1.0))
        self._add_axis((0, 0, 0), (0, 0, self.AXIS_LEN), (0.3, 0.5, 1.0, 1.0))

        # Launch marker (yellow), current position (red).
        self._launch_marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(1.0, 1.0, 0.3, 1.0), size=14.0)
        self.addItem(self._launch_marker)

        self._round_marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(1.0, 0.2, 0.2, 1.0), size=10.0)
        self.addItem(self._round_marker)

        # Cyan trail polyline.
        self._trail = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)),
            color=(0.2, 1.0, 1.0, 1.0), width=2.0, antialias=True)
        self.addItem(self._trail)

        # Drop-line from current position straight down to the ground plane.
        self._dropline = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            color=(1.0, 0.6, 0.2, 0.5), width=1.0, antialias=True)
        self.addItem(self._dropline)

        self._launch_ecef: np.ndarray | None = None
        self._R_e2n: np.ndarray | None = None
        self._points: list[tuple[float, float, float]] = []

    def _add_axis(self, p0, p1, color) -> None:
        line = gl.GLLinePlotItem(
            pos=np.array([p0, p1], dtype=float),
            color=color, width=2.0, antialias=True)
        self.addItem(line)

    def add_fix(self, lat: float, lon: float, alt: float) -> None:
        if self._launch_ecef is None:
            self._launch_ecef = lla_to_ecef(lat, lon, alt)
            self._R_e2n = ned_basis(lat, lon).T
            point = (0.0, 0.0, 0.0)
        else:
            r_ecef = lla_to_ecef(lat, lon, alt)
            ned = self._R_e2n @ (r_ecef - self._launch_ecef)
            point = (float(ned[1]), float(ned[0]), float(-ned[2]))  # E, N, U
        self._points.append(point)

        pts = np.array(self._points, dtype=float)
        self._trail.setData(pos=pts)
        cur = pts[-1]
        self._round_marker.setData(pos=cur.reshape(1, 3))
        ground = np.array([cur[0], cur[1], 0.0])
        self._dropline.setData(pos=np.vstack([ground, cur]))

    def last_enu(self) -> tuple[float, float, float]:
        """Return the most recent (East, North, Up) offset in metres."""
        return self._points[-1] if self._points else (0.0, 0.0, 0.0)

    def clear_trail(self) -> None:
        """Clear the trail and forget the launch reference frame."""
        self._launch_ecef = None
        self._R_e2n = None
        self._points.clear()
        empty = np.zeros((1, 3))
        self._trail.setData(pos=empty)
        self._round_marker.setData(pos=empty)
        self._launch_marker.setData(pos=empty)
        self._dropline.setData(pos=np.zeros((2, 3)))
