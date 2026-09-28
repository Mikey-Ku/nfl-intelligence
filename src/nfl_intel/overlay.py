"""Transparent, click-through overlay on top of the video player, plus the stats side panel.

The overlay window sits exactly over the captured window, stays on top and ignores the
mouse. A worker thread runs capture + tracking + stats and posts results to the windows.
"""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Iterable

import numpy as np
from PySide6.QtCore import QPoint, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from .capture import Region, ScreenSource
from .draw import minimap, player_label, team_color
from .field import draw_field
from .panel import render_panel
from .schema import FrameResult
from .stats import LiveStats, StatsSnapshot

WINDOW_FLAGS = (
    Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.WindowTransparentForInput
    | Qt.WindowType.WindowDoesNotAcceptFocus
)


def float_over_fullscreen(widget: QWidget, hide_from_capture: bool = False) -> bool:
    """Let the window float over full-screen apps on every Space.

    With `hide_from_capture` the window is also left out of screen recordings, which the
    region-capture source needs so the pipeline doesn't see its own boxes. Window-ID capture
    never sees the overlay anyway, so there it stays visible to screenshots and recordings.
    """
    try:
        import objc
        from AppKit import (
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowSharingNone,
        )

        view = objc.objc_object(c_void_p=int(widget.winId()))
        window = view.window()
        if hide_from_capture:
            window.setSharingType_(NSWindowSharingNone)
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        return True
    except Exception as e:  # pragma: no cover - depends on the host OS
        print(f"warning: could not configure the overlay window ({e}); use --mirror if it misbehaves", file=sys.stderr)
        return False


def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class OverlayWindow(QWidget):
    def __init__(self, region: Region, show_minimap: bool = True, hide_from_capture: bool = False):
        super().__init__()
        self.region = region
        self.hide_from_capture = hide_from_capture
        self.scale = 1.0                 # captured pixels per screen point
        self.result: FrameResult | None = None
        self.show_minimap = show_minimap
        self._field = draw_field(4.0)
        self._minimap: QImage | None = None

        self.setWindowFlags(WINDOW_FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(region.left, region.top, region.width, region.height)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        float_over_fullscreen(self, hide_from_capture=self.hide_from_capture)

    def move_to(self, region: Region) -> None:
        if region != self.region:
            self.region = region
            self.setGeometry(region.left, region.top, region.width, region.height)

    @Slot(object, float, object)
    def set_result(self, result: FrameResult, scale: float, _snapshot=None) -> None:
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
            painter.setPen(QPen(QColor(r, g, b, 230), 2))
            x1, y1, x2, y2 = (v * s for v in p.box)
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            painter.drawText(int(x1), int(y1) - 3, player_label(p))
        if self._minimap is not None:
            target_w = int(self.width() * 0.3)
            img = self._minimap.scaledToWidth(target_w, Qt.TransformationMode.SmoothTransformation)
            painter.setOpacity(0.85)
            painter.drawImage(self.width() - img.width() - 8, self.height() - img.height() - 8, img)
        painter.end()


class StatsPanelWindow(QWidget):
    """The live stats panel, docked beside the video inside or next to the captured window."""

    WIDTH = 300
    GAP = 10

    def __init__(self, side: str = "left", hide_from_capture: bool = False):
        super().__init__()
        self.side = side
        self.hide_from_capture = hide_from_capture
        self._img: QImage | None = None
        self._last_render = 0.0
        self.setWindowFlags(WINDOW_FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(self.WIDTH, 560)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        float_over_fullscreen(self, hide_from_capture=self.hide_from_capture)

    def place(self, region: Region, scale: float, field_bbox) -> None:
        """Dock beside the video: left of it by default (NFL Pro's filter column is dead space)."""
        if field_bbox is not None:
            vx1 = region.left + field_bbox[0] / scale
            vx2 = region.left + field_bbox[2] / scale
        else:
            vx1, vx2 = region.left, region.left + region.width
        height = int(min(640, max(440, region.height * 0.7)))
        x = vx1 - self.WIDTH - self.GAP if self.side == "left" else vx2 + self.GAP
        y = region.top + (region.height - height) / 2
        screen = QGuiApplication.screenAt(QPoint(int(region.left + region.width / 2), int(region.top + region.height / 2)))
        geo = (screen or QGuiApplication.primaryScreen()).geometry()
        x = min(max(x, geo.left() + 4), geo.right() - self.WIDTH - 4)
        y = min(max(y, geo.top() + 4), geo.bottom() - height - 4)
        if (int(x), int(y), self.WIDTH, height) != (self.x(), self.y(), self.width(), self.height()):
            self.setGeometry(int(x), int(y), self.WIDTH, height)

    def set_snapshot(self, snapshot: StatsSnapshot, region: Region, scale: float) -> None:
        now = time.monotonic()
        if now - self._last_render < 0.1:            # 10 Hz is plenty for numbers
            return
        self._last_render = now
        self.place(region, scale, snapshot.field_bbox)
        self._img = render_panel(snapshot, self.width(), self.height(), dpr=self.devicePixelRatioF())
        self.update()

    def paintEvent(self, event) -> None:
        if self._img is None:
            return
        painter = QPainter(self)
        painter.drawImage(0, 0, self._img)
        painter.end()


class TrackingWorker(QThread):
    result_ready = Signal(object, float, object)
    finished_all = Signal()

    def __init__(self, tracker, source, sink=None, stats: LiveStats | None = None):
        super().__init__()
        self.tracker = tracker
        self.source = source
        self.sink = sink            # optional callable(FrameResult, frame, StatsSnapshot | None)
        self.stats = stats
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        for idx, t, frame in self.source:
            if self._stop:
                break
            res = self.tracker.process(frame, idx, t)
            snap = None
            if self.stats is not None:
                field = getattr(self.tracker, "field", None)
                snap = self.stats.update(res, field.bbox if field is not None else None)
            if self.sink is not None:
                self.sink(res, frame, snap)
            self.result_ready.emit(res, getattr(self.source, "scale", 1.0), snap)
        self.finished_all.emit()


def run_overlay(
    region: Region,
    tracker,
    source: Iterable,
    sink=None,
    show_minimap: bool = True,
    seconds: float | None = None,
    stats: LiveStats | None = None,
    panel_side: str | None = "left",
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    hide = isinstance(source, ScreenSource)
    window = OverlayWindow(region, show_minimap=show_minimap, hide_from_capture=hide)
    panel = StatsPanelWindow(panel_side, hide_from_capture=hide) if (stats is not None and panel_side) else None
    worker = TrackingWorker(tracker, source, sink, stats)
    worker.result_ready.connect(window.set_result)
    worker.finished_all.connect(app.quit)
    app.aboutToQuit.connect(worker.stop)

    def on_result(_res, scale, snap):
        current = getattr(source, "region", region)
        window.move_to(current)
        if panel is not None and snap is not None:
            panel.set_snapshot(snap, current, scale)

    worker.result_ready.connect(on_result)
    if seconds:
        QTimer.singleShot(int(seconds * 1000), app.quit)
    # Ctrl-C / kill -TERM end the session cleanly so the CSV and recording still get written.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.quit())
    keepalive = QTimer()               # lets Python see the signal while Qt's loop is blocking
    keepalive.timeout.connect(lambda: None)
    keepalive.start(200)

    window.show()
    if panel is not None:
        panel.place(region, 1.0, None)
        panel.show()
    worker.start()
    code = app.exec()
    worker.stop()
    worker.wait(2000)
    return code
