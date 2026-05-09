"""GCS main window: 3D trajectory + 3D attitude + status bar.

Run with:
    python -m ballistic.gcs.main --port 51000
"""

from __future__ import annotations

import argparse
import os
import sys

# pyqtgraph.opengl uses QOpenGLWidget. Force the OpenGL RHI backend before
# QApplication is constructed so all GL widgets share a single context.
os.environ.setdefault("QSG_RHI_BACKEND", "opengl")

from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface  # noqa: E402

QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QSplitter,
    QStatusBar,
)

from ..common.packet import (
    PHASE_BALLISTIC,
    PHASE_BOOST,
    PHASE_IMPACT,
    PHASE_PRELAUNCH,
    ROUND_ARTILLERY,
    ROUND_MORTAR,
    ROUND_ROCKET,
    Telemetry,
)
from .attitude_view import AttitudeView
from .trajectory_view import TrajectoryView
from .udp_listener import UdpListener


_PHASE_NAMES = {
    PHASE_PRELAUNCH: "PRELAUNCH",
    PHASE_BOOST: "BOOST",
    PHASE_BALLISTIC: "BALLISTIC",
    PHASE_IMPACT: "IMPACT",
}
_ROUND_NAMES = {
    ROUND_MORTAR: "120mm mortar",
    ROUND_ROCKET: "127mm rocket",
    ROUND_ARTILLERY: "155mm artillery",
}


class MainWindow(QMainWindow):
    def __init__(self, port: int):
        super().__init__()
        self.setWindowTitle(f"Ballistic GCS (UDP :{port})")
        self.resize(1280, 720)

        self.trajectory_view = TrajectoryView()
        self.attitude_view = AttitudeView()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.trajectory_view)
        splitter.addWidget(self.attitude_view)
        splitter.setSizes([800, 480])
        self.setCentralWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._round_label = QLabel("round: -")
        self._phase_label = QLabel("phase: -")
        self._tof_label = QLabel("t: 0.0 s")
        self._pos_label = QLabel("E: 0.0 m  N: 0.0 m")
        self._alt_label = QLabel("alt: 0.0 m")
        self._vel_label = QLabel("v: 0.0 m/s")
        for w in (self._round_label, self._phase_label, self._tof_label,
                  self._pos_label, self._alt_label, self._vel_label):
            self.status.addPermanentWidget(w)
            w.setMinimumWidth(140)

        self.listener = UdpListener(port)
        self.listener.packet_received.connect(self._on_packet)
        self.listener.start()
        self._last_round: int | None = None

    def closeEvent(self, event):
        self.listener.stop()
        self.listener.wait(1000)
        super().closeEvent(event)

    def _on_packet(self, pkt: Telemetry) -> None:
        if pkt.round_id != self._last_round:
            self._last_round = pkt.round_id
            self.trajectory_view.clear_trail()
        self.trajectory_view.add_fix(pkt.lat, pkt.lon, pkt.alt)
        self.attitude_view.update_attitude(pkt.round_id, pkt.q_w, pkt.q_x, pkt.q_y, pkt.q_z)

        v_mag = (pkt.v_n ** 2 + pkt.v_e ** 2 + pkt.v_d ** 2) ** 0.5
        cur = self.trajectory_view.last_enu()
        self._round_label.setText(f"round: {_ROUND_NAMES.get(pkt.round_id, '?')}")
        self._phase_label.setText(f"phase: {_PHASE_NAMES.get(pkt.phase, '?')}")
        self._tof_label.setText(f"t: {pkt.t_sim:6.2f} s")
        self._pos_label.setText(f"E: {cur[0]:7.0f} m  N: {cur[1]:7.0f} m")
        self._alt_label.setText(f"alt: {pkt.alt:7.1f} m")
        self._vel_label.setText(f"v: {v_mag:6.1f} m/s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ballistic Ground Control Station")
    p.add_argument("--port", type=int, default=51000, help="UDP listen port")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv)
    win = MainWindow(args.port)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
