"""Leaflet map embedded in a QWebEngineView with a QWebChannel bridge."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView


class _Bridge(QObject):
    fix = Signal(float, float, float, float)         # lat, lon, alt, t
    reset_signal = Signal()

    @Slot()
    def reset(self) -> None:
        self.reset_signal.emit()


class MapView(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = _Bridge()
        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.page().setWebChannel(self.channel)
        html = Path(__file__).with_name("leaflet.html")
        self.load(QUrl.fromLocalFile(str(html.resolve())))

    def push_fix(self, lat: float, lon: float, alt: float, t: float) -> None:
        self.bridge.fix.emit(lat, lon, alt, t)

    def reset(self) -> None:
        self.bridge.reset_signal.emit()
