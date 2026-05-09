"""Live 3D attitude display using pyqtgraph.opengl.

Renders a NED axis triad and a round-shaped body whose orientation is
driven by a body->NED quaternion supplied per telemetry packet.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QMatrix4x4

from ..common.packet import ROUND_ARTILLERY, ROUND_MORTAR, ROUND_ROCKET


def _cylinder_mesh(radius: float, height: float, n: int = 24) -> gl.MeshData:
    """Cylinder along +x with caps at x=0 and x=height."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cy = radius * np.cos(angles)
    cz = radius * np.sin(angles)
    verts = []
    # ring at base, ring at top, plus 2 cap centres
    for x in (0.0, height):
        for i in range(n):
            verts.append([x, cy[i], cz[i]])
    base_centre = len(verts); verts.append([0.0, 0.0, 0.0])
    top_centre = len(verts);  verts.append([height, 0.0, 0.0])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        a, b = i, j
        c, d = n + i, n + j
        faces.append([a, b, c])
        faces.append([b, d, c])
        faces.append([base_centre, b, a])
        faces.append([top_centre, c, d])
    return gl.MeshData(vertexes=np.array(verts, dtype=float),
                       faces=np.array(faces, dtype=int))


def _cone_mesh(radius: float, height: float, base_x: float, n: int = 24) -> gl.MeshData:
    """Cone with base at base_x, apex at base_x+height."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cy = radius * np.cos(angles)
    cz = radius * np.sin(angles)
    verts = [[base_x, cy[i], cz[i]] for i in range(n)]
    apex = len(verts); verts.append([base_x + height, 0.0, 0.0])
    centre = len(verts); verts.append([base_x, 0.0, 0.0])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, apex])
        faces.append([centre, j, i])
    return gl.MeshData(vertexes=np.array(verts, dtype=float),
                       faces=np.array(faces, dtype=int))


def _fin_mesh(span: float, chord: float, x_offset: float, angle_deg: float) -> gl.MeshData:
    """Flat triangular fin in body x-(rotated y) plane, around tail."""
    a = np.deg2rad(angle_deg)
    cy, cz = np.cos(a), np.sin(a)
    base1 = np.array([x_offset, 0.0, 0.0])
    base2 = np.array([x_offset + chord, 0.0, 0.0])
    tip = np.array([x_offset + chord * 0.4, span * cy, span * cz])
    verts = np.array([base1, base2, tip])
    faces = np.array([[0, 1, 2], [0, 2, 1]])
    return gl.MeshData(vertexes=verts, faces=faces)


def _build_round(round_id: int) -> list[gl.GLMeshItem]:
    """Build a list of GLMeshItems forming the body. Body x = forward.

    Body is centred so that x in [-L/2, +L/2]; we'll model it from -L/2 (tail)
    to +L/2 (nose) by translating each piece.
    """
    items: list[gl.GLMeshItem] = []
    if round_id == ROUND_MORTAR:
        L, r = 0.7, 0.060
        body_color = (0.6, 0.6, 0.65, 1.0)
        nose = _cone_mesh(r, 0.10, base_x=L / 2 - 0.10)
        body = _cylinder_mesh(r, L - 0.10)
        # translate body so it ends at L/2 - 0.10
        body_xform = QMatrix4x4(); body_xform.translate(-L / 2, 0, 0)
        nose_item = gl.GLMeshItem(meshdata=nose, smooth=True, color=body_color, shader='shaded')
        body_item = gl.GLMeshItem(meshdata=body, smooth=True, color=body_color, shader='shaded')
        nose_item.setTransform(QMatrix4x4())
        body_item.setTransform(body_xform)
        items += [body_item, nose_item]
        # tail fins
        for i in range(4):
            fin = _fin_mesh(span=0.05, chord=0.10, x_offset=-L / 2, angle_deg=i * 90)
            fi = gl.GLMeshItem(meshdata=fin, smooth=False, color=(0.3, 0.3, 0.35, 1), shader='shaded')
            items.append(fi)
    elif round_id == ROUND_ROCKET:
        L, r = 1.85, 0.0635
        body_color = (0.7, 0.4, 0.3, 1.0)
        nose = _cone_mesh(r, 0.20, base_x=L / 2 - 0.20)
        body = _cylinder_mesh(r, L - 0.20)
        body_xform = QMatrix4x4(); body_xform.translate(-L / 2, 0, 0)
        items.append(gl.GLMeshItem(meshdata=body, smooth=True, color=body_color, shader='shaded'))
        items[-1].setTransform(body_xform)
        items.append(gl.GLMeshItem(meshdata=nose, smooth=True, color=(0.85, 0.85, 0.85, 1), shader='shaded'))
        for i in range(4):
            fin = _fin_mesh(span=0.10, chord=0.20, x_offset=-L / 2, angle_deg=i * 90)
            items.append(gl.GLMeshItem(meshdata=fin, smooth=False, color=(0.4, 0.2, 0.15, 1), shader='shaded'))
    else:  # artillery
        L, r = 0.60, 0.0775
        body_color = (0.55, 0.55, 0.45, 1.0)
        nose = _cone_mesh(r, 0.18, base_x=L / 2 - 0.18)
        body = _cylinder_mesh(r, L - 0.18)
        body_xform = QMatrix4x4(); body_xform.translate(-L / 2, 0, 0)
        items.append(gl.GLMeshItem(meshdata=body, smooth=True, color=body_color, shader='shaded'))
        items[-1].setTransform(body_xform)
        items.append(gl.GLMeshItem(meshdata=nose, smooth=True, color=(0.7, 0.6, 0.3, 1), shader='shaded'))
    return items


def quat_to_matrix4_enu(q_w: float, q_x: float, q_y: float, q_z: float) -> QMatrix4x4:
    """body->NED quaternion -> body->ENU rotation as a QMatrix4x4.

    The scene uses a right-handed ENU frame so that +z is visually "up".
    The mapping NED (N, E, D) -> ENU (E, N, U) is the rotation
        M = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]   (det = +1)
    so we return M @ R(q) where R(q) is the body->NED rotation.
    """
    n = (q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z) ** 0.5
    if n < 1e-9:
        return QMatrix4x4()
    w, x, y, z = q_w / n, q_x / n, q_y / n, q_z / n
    xx, yy, zz = x * x, y * y, z * z
    # body->NED rotation rows
    r00 = 1.0 - 2.0 * (yy + zz); r01 = 2.0 * (x * y - z * w); r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w); r11 = 1.0 - 2.0 * (xx + zz); r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w); r21 = 2.0 * (y * z + x * w); r22 = 1.0 - 2.0 * (xx + yy)
    # ENU rows: row0 = NED row1 (East), row1 = NED row0 (North), row2 = -NED row2 (Up)
    return QMatrix4x4(
        r10,  r11,  r12,  0.0,
        r00,  r01,  r02,  0.0,
        -r20, -r21, -r22, 0.0,
        0.0,  0.0,  0.0,  1.0,
    )


# Back-compat alias used by tests/external callers.
quat_to_matrix4 = quat_to_matrix4_enu


class AttitudeView(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=4.0, azimuth=30, elevation=20)
        self.setBackgroundColor((20, 20, 28))

        # ENU axis triad (so visual "up" is genuinely up):
        #   +x  East  (green)
        #   +y  North (red)
        #   +z  Up    (blue)
        self._add_axis([0, 0, 0], [1, 0, 0], (0.2, 1.0, 0.2, 1))
        self._add_axis([0, 0, 0], [0, 1, 0], (1.0, 0.2, 0.2, 1))
        self._add_axis([0, 0, 0], [0, 0, 1], (0.3, 0.5, 1.0, 1))

        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.5, 0.5)
        self.addItem(grid)

        self._round_items: list[gl.GLMeshItem] = []
        self._round_id: int | None = None

    def _add_axis(self, p0, p1, color):
        pts = np.array([p0, p1], dtype=float)
        line = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
        self.addItem(line)

    def _set_round(self, round_id: int) -> None:
        if self._round_id == round_id:
            return
        for it in self._round_items:
            self.removeItem(it)
        self._round_items = _build_round(round_id)
        for it in self._round_items:
            self.addItem(it)
        self._round_id = round_id

    def update_attitude(self, round_id: int, q_w: float, q_x: float,
                        q_y: float, q_z: float) -> None:
        self._set_round(round_id)
        m = quat_to_matrix4_enu(q_w, q_x, q_y, q_z)
        for it in self._round_items:
            # Build per-item transform: rotation * existing local translation
            local = QMatrix4x4(it.transform())
            # The local transform was set once at build time; replace by rot @ local.
            # We stored the original local transform in setTransform; here we
            # recompose: rebuild local based on round geometry.
            # Simpler: keep cached locals.
            it.setTransform(m * _local_for(it, round_id))


# Cache the original local transform per item (assigned at build time). We
# tag each item by its index in the build order.

_LOCAL_CACHE: dict[int, QMatrix4x4] = {}


def _local_for(item: gl.GLMeshItem, round_id: int) -> QMatrix4x4:
    key = id(item)
    cached = _LOCAL_CACHE.get(key)
    if cached is None:
        cached = QMatrix4x4(item.transform())
        _LOCAL_CACHE[key] = cached
    return cached
