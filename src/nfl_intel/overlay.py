"""Transparent, click-through overlay drawn on top of the video player.

The overlay window sits exactly over the captured screen region, stays on top, ignores
the mouse, and asks macOS to exclude it from screen capture so the pipeline doesn't see
its own drawings. A worker thread runs capture + tracking and posts results to the window.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from .capture import Region
from .draw import minimap, team_color
from .field import draw_field
from .schema import FrameResult


def exclude_from_capture(widget: QWidget) -> bool:
    """Hide this window from screen recording and let it float over full-screen apps."""
    try:
        import objc
        from AppKit import (
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowSharingNone,
        )

        view = objc.objc_object(c_void_p=int(widget.winId()))
        window = view.window()
        window.setSharingType_(NSWindowSharingNone)
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        return True
    except Exception as e:  # pragma: no cover - depends on the host OS
        print(f"warning: could not exclude overlay from capture ({e}); use --mirror if boxes stack up", file=sys.stderr)
        return False


def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class OverlayWindow(QWidget):
    def __init__(self, region: Region, show_minimap: bool = True):
        super().__init__()
        self.region = region
        self.scale = 1.0                 # captured pixels per screen point
        self.result: FrameResult | None = None
        self.show_minimap = show_minimap
        self._field = draw_field(4.0)
        self._minimap: QImage | None = None
        self._fps = 0.0
        self._last = time.monotonic()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(region.left, region.top, region.width, region.height)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        exclude_from_capture(self)

    @Slot(object, float)
    def set_result(self, result: FrameResult, scale: float) -> None:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 / dt if self._fps else 1 / dt
        self.result = result
        self.scale = scale
        mm = minimap(result, 4.0, self._field) if self.show_minimap else None
        self._minimap = bgr_to_qimage(mm) if mm is not None else None
        self.update()

    def paintEvent(self, event) -> None:
        if self.result is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Menlo", 11))
        s = 1.0 / self.scale
        for p in self.result.players:
            b, g, r = team_color(p.team)
            pen = QPen(QColor(r, g, b, 230), 2)
            painter.setPen(pen)
            x1, y1, x2, y2 = (v * s for v in p.box)
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            label = str(p.track_id)
            if p.field_yd is not None:
                label += f" {p.field_yd[0]:.0f},{p.field_yd[1]:.0f}"
            painter.drawText(int(x1), int(y1) - 3, label)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
        painter.drawText(8, 16, f"{len(self.result.players)} players  {self._fps:4.1f} fps")
        if self._minimap is not None:
            target_w = int(self.width() * 0.3)
            img = self._minimap.scaledToWidth(target_w, Qt.TransformationMode.SmoothTransformation)
            painter.setOpacity(0.85)
            painter.drawImage(self.width() - img.width() - 8, self.height() - img.height() - 8, img)
        painter.end()


class TrackingWorker(QThread):
    result_ready = Signal(object, float)
    finished_all = Signal()

    def __init__(self, tracker, source, sink=None):
        super().__init__()
        self.tracker = tracker
        self.source = source
        self.sink = sink            # optional callable(FrameResult, frame)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        for idx, t, frame in self.source:
            if self._stop:
                break
            res = self.tracker.process(frame, idx, t)
            if self.sink is not None:
                self.sink(res, frame)
            self.result_ready.emit(res, getattr(self.source, "scale", 1.0))
        self.finished_all.emit()


def run_overlay(region: Region, tracker, source: Iterable, sink=None, show_minimap: bool = True) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = OverlayWindow(region, show_minimap=show_minimap)
    worker = TrackingWorker(tracker, source, sink)
    worker.result_ready.connect(window.set_result)
    worker.finished_all.connect(app.quit)
    app.aboutToQuit.connect(worker.stop)
    window.show()
    worker.start()
    code = app.exec()
    worker.stop()
    worker.wait(2000)
    return code
