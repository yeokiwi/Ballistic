"""UDP telemetry sender."""

from __future__ import annotations

import socket

from ..common.packet import Telemetry


class UdpTelemetry:
    def __init__(self, host: str = "127.0.0.1", port: int = 51000):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, t: Telemetry) -> None:
        self.sock.sendto(t.pack(), self.addr)

    def close(self) -> None:
        self.sock.close()
