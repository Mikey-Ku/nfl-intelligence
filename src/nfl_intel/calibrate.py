"""Click field landmarks on a frame to build a pixel→yard homography.

Click a point in the window, then type its landmark name in the terminal (for example
`L40-H1` = left 40, near hash). Keys: `u` undo, `s` save (needs 4+ points), `q` quit.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .field import FieldCalibration, landmark_xy

WINDOW = "nfl calibrate"


def _draw(frame: np.ndarray, points: list[tuple[int, int]], names: list[str]) -> np.ndarray:
    img = frame.copy()
    for (x, y), name in zip(points, names):
        cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{len(points)} points  click=add  u=undo  s=save  q=quit", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def calibrate_interactive(frame: np.ndarray, out_path: str | Path, meta: dict | None = None) -> FieldCalibration | None:
    points: list[tuple[int, int]] = []
    names: list[str] = []
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    calib = None
    while True:
        cv2.imshow(WINDOW, _draw(frame, points, names))
        key = cv2.waitKey(30) & 0xFF
        if clicked:
            x, y = clicked.pop()
            name = input(f"landmark for ({x},{y}) e.g. L40-H1, 50-S2, REZ-S1 (blank to cancel): ").strip().upper()
            if name:
                try:
                    landmark_xy(name)
                except ValueError as e:
                    print(e)
                    continue
                points.append((x, y))
                names.append(name)
                if len(points) >= 4:
                    calib = FieldCalibration.from_landmarks(points, names, meta)
                    yd = [landmark_xy(n) for n in names]
                    print(f"  fit with {len(points)} points, mean error {calib.reprojection_error(points, yd):.2f} yd")
        elif key == ord("u") and points:
            points.pop()
            names.pop()
        elif key == ord("s"):
            if len(points) < 4:
                print("need at least 4 points")
                continue
            calib = FieldCalibration.from_landmarks(points, names, {**(meta or {}), "points": points, "names": names})
            calib.save(out_path)
            print(f"saved {out_path}")
            break
        elif key == ord("q"):
            break
    cv2.destroyWindow(WINDOW)
    return calib
