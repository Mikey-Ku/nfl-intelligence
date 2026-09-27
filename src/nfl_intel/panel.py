"""The side panel: renders a StatsSnapshot to an image with QPainter.

One renderer feeds the on-screen panel window, the saved test shots and the tests. Sizes are
in points; `dpr` is the device pixel ratio (2 on the HiDPI Dell), so text stays crisp.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen

from .draw import team_color
from .stats import StatsSnapshot

BG = QColor(14, 16, 20, 238)
CARD = QColor(255, 255, 255, 14)
TEXT = QColor(240, 240, 242)
MUTED = QColor(150, 156, 168)
GOOD = QColor(88, 204, 130)
BAD = QColor(238, 96, 96)
LINE = QColor(255, 255, 255, 40)
FONT = "Helvetica Neue"
TEAM_NAMES = ("Dark jerseys", "Light jerseys", "Unassigned")


def _team_qcolor(team: int | None) -> QColor:
    b, g, r = team_color(team)
    return QColor(r, g, b)


def _font(size: float, bold: bool = False) -> QFont:
    f = QFont(FONT)
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


def _label(p: QPainter, x: float, y: float, text: str) -> None:
    p.setFont(_font(8.5, bold=True))
    p.setPen(MUTED)
    p.drawText(QPointF(x, y), text.upper())


def _sparkline(p: QPainter, rect: QRectF, points: list[tuple[float, float]], lo: float, hi: float, color: QColor) -> None:
    p.setPen(QPen(LINE, 1))
    p.drawLine(QPointF(rect.left(), rect.bottom()), QPointF(rect.right(), rect.bottom()))
    if len(points) < 2 or hi <= lo:
        return
    xs = [x for x, _ in points]
    x0, x1 = min(xs), max(xs)
    if x1 <= x0:
        return
    path = QPainterPath()
    for i, (x, y) in enumerate(points):
        px = rect.left() + (x - x0) / (x1 - x0) * rect.width()
        py = rect.bottom() - (min(max(y, lo), hi) - lo) / (hi - lo) * rect.height()
        path.moveTo(px, py) if i == 0 else path.lineTo(px, py)
    p.setPen(QPen(color, 1.6))
    p.drawPath(path)


def render_panel(s: StatsSnapshot, width: int = 300, height: int = 560, dpr: float = 2.0) -> QImage:
    img = QImage(int(width * dpr), int(height * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(BG)
    p.drawRoundedRect(QRectF(0, 0, width, height), 10, 10)

    m = 16.0                       # margin
    w = width - 2 * m
    y = 28.0

    # header
    p.setFont(_font(13, bold=True))
    p.setPen(TEXT)
    p.drawText(QPointF(m, y), f"Play {s.play}" if s.play else "Waiting for a play")
    p.setFont(_font(9))
    p.setPen(MUTED)
    p.drawText(QRectF(m, y - 14, w, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
               f"{s.play_clock:4.1f}s   {s.fps:4.1f} fps")
    y += 22

    # play value / plus-minus
    _label(p, m, y, "Play value")
    y += 8
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(CARD)
    p.drawRoundedRect(QRectF(m, y, w, 96), 8, 8)
    if s.value is None:
        p.setPen(MUTED)
        p.setFont(_font(9.5))
        why = "Needs the value model" if s.calibrated else "Needs field calibration (yards)"
        p.drawText(QRectF(m + 12, y, w - 24, 96), Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
                   f"Plus/minus is off.\n{why}.")
    else:
        p.setPen(TEXT)
        p.setFont(_font(26, bold=True))
        p.drawText(QPointF(m + 12, y + 40), f"{s.value * 100:.0f}%")
        d = s.value_delta or 0.0
        p.setPen(GOOD if d >= 0 else BAD)
        p.setFont(_font(14, bold=True))
        p.drawText(QRectF(m, y + 14, w - 12, 30), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{d * 100:+.0f}")
        _sparkline(p, QRectF(m + 12, y + 52, w - 24, 34), s.value_history, 0.0, 1.0, GOOD if d >= 0 else BAD)
    y += 96 + 24

    # swings
    if s.swings:
        _label(p, m, y, "Biggest swings")
        y += 16
        for sw in s.swings:
            p.setFont(_font(10, bold=True))
            p.setPen(GOOD if sw.delta >= 0 else BAD)
            p.drawText(QPointF(m, y), f"{sw.delta * 100:+.0f}")
            p.setFont(_font(9.5))
            p.setPen(TEXT)
            p.drawText(QPointF(m + 38, y), f"{sw.t:.1f}s  {sw.label}".rstrip())
            y += 17
        y += 10

    # on field
    _label(p, m, y, "On the field")
    y += 30
    p.setFont(_font(24, bold=True))
    p.setPen(TEXT)
    p.drawText(QPointF(m, y), str(s.on_field))
    p.setFont(_font(9))
    p.setPen(MUTED)
    p.drawText(QPointF(m + 44, y), "of 22 players + officials")
    y += 14
    total = max(1, sum(s.team_counts))
    x = m
    for i, n in enumerate(s.team_counts):
        seg = w * n / total
        if seg > 0:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_team_qcolor(i if i < 2 else None))
            p.drawRoundedRect(QRectF(x, y, max(2.0, seg - 2), 6), 3, 3)
            x += seg
    y += 22
    for i, n in enumerate(s.team_counts):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_team_qcolor(i if i < 2 else None))
        p.drawEllipse(QPointF(m + 4, y - 4), 4, 4)
        p.setFont(_font(10))
        p.setPen(TEXT)
        p.drawText(QPointF(m + 16, y), TEAM_NAMES[i])
        p.drawText(QRectF(m, y - 13, w, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(n))
        y += 18
    y += 12

    # extra metrics (yards-based, once calibrated)
    if s.rows:
        _label(p, m, y, "This play")
        y += 18
        for name, value in s.rows:
            p.setFont(_font(10))
            p.setPen(MUTED)
            p.drawText(QPointF(m, y), name)
            p.setPen(TEXT)
            p.drawText(QRectF(m, y - 13, w, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, value)
            y += 18
        y += 12

    # tracking health
    _label(p, m, y, "Tracking")
    y += 18
    for name, value, bad in (
        ("Players tracked", str(s.active_ids), False),
        ("Picked up late", str(s.reacquired), s.reacquired > 3),
        ("Lost", str(s.lost), s.lost > 3),
    ):
        p.setFont(_font(10))
        p.setPen(MUTED)
        p.drawText(QPointF(m, y), name)
        p.setPen(BAD if bad else TEXT)
        p.drawText(QRectF(m, y - 13, w, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, value)
        y += 18
    y += 6
    if height - y > 50:
        spark = QRectF(m, y, w, min(44.0, height - y - 26))
        _sparkline(p, spark, [(-ago, float(n)) for ago, n in s.history], 0.0, 26.0, QColor(120, 170, 255))
        p.setFont(_font(8))
        p.setPen(MUTED)
        p.drawText(QPointF(m, spark.bottom() + 13), "players found, last 10 s")
    p.end()
    return img


def qimage_to_bgr(img: QImage, background: tuple[int, int, int] = (20, 16, 14)) -> np.ndarray:
    """QImage -> BGR array, flattened onto a solid background."""
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = img.height(), img.width()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w].astype(np.float32)
    alpha = arr[..., 3:4] / 255.0
    rgb = arr[..., :3] * alpha + np.array(background[::-1], dtype=np.float32) * (1 - alpha)
    return np.ascontiguousarray(rgb[..., ::-1].astype(np.uint8))
