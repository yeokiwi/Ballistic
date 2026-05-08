"""Background QThread that receives telemetry packets and emits Qt signals."""

from __future__ import annotations

import socket

from PySide6.QtCore import QThread, Signal

from ..common.packet import PACKET_SIZE, Telemetry


class UdpListener(QThread):
    packet_received = Signal(object)  # Telemetry

    def __init__(self, port: int, host: str = "0.0.0.0", parent=None):
        super().__init__(parent)
        self.port = port
        self.host = host
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.25)
        try:
            while not self._stop:
                try:
                    data, _addr = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                if len(data) != PACKET_SIZE:
                    continue
                try:
                    pkt = Telemetry.unpack(data)
                except Exception:
                    continue
                self.packet_received.emit(pkt)
        finally:
            sock.close()
