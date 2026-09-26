"""Frame sources: a region of the screen (where NFL Pro is playing) or a video file.

Both yield `(frame_index, seconds, frame_bgr)`. Screen capture uses mss; on a Retina
display the grabbed image is larger than the region in points, and `ScreenSource.scale`
says by how much so overlays can map pixels back to points.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    @classmethod
    def parse(cls, text: str) -> Region:
        parts = [int(p) for p in text.replace(" ", "").split(",")]
        if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
            raise ValueError("region must be 'left,top,width,height' with positive size")
        return cls(*parts)

    def as_mss(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class WindowInfo:
    id: int
    owner: str
    title: str
    region: Region

    def __str__(self) -> str:
        r = self.region
        return f"[{self.id}] {self.owner} — {self.title or '(untitled)'}  {r.width}x{r.height} at ({r.left},{r.top})"


def list_windows(min_size: int = 200) -> list[WindowInfo]:
    """On-screen windows, biggest first. Titles need Screen Recording permission."""
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    infos = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID
    )
    out = []
    for w in infos or []:
        if w.get("kCGWindowLayer", 0) != 0:
            continue
        b = w.get("kCGWindowBounds") or {}
        region = Region(int(b.get("X", 0)), int(b.get("Y", 0)), int(b.get("Width", 0)), int(b.get("Height", 0)))
        if region.width < min_size or region.height < min_size:
            continue
        out.append(WindowInfo(
            id=int(w.get("kCGWindowNumber", 0)),
            owner=str(w.get("kCGWindowOwnerName", "")),
            title=str(w.get("kCGWindowName", "")),
            region=region,
        ))
    out.sort(key=lambda w: w.region.width * w.region.height, reverse=True)
    return out


def find_window(query: str) -> WindowInfo | None:
    """Largest on-screen window whose owner or title contains `query` (case-insensitive)."""
    q = query.lower()
    for w in list_windows():
        if q in w.owner.lower() or q in w.title.lower():
            return w
    return None


def looks_blank(frame: np.ndarray) -> bool:
    """True for an (almost) uniform frame: nothing is being captured."""
    return frame.size == 0 or float(frame.std()) < 2.0


class ScreenSource:
    def __init__(self, region: Region, fps_cap: float | None = None, max_frames: int | None = None):
        self.region = region
        self.fps_cap = fps_cap
        self.max_frames = max_frames
        self.scale = 1.0

    def grab(self) -> np.ndarray:
        import mss

        with mss.mss() as sct:
            shot = sct.grab(self.region.as_mss())
        frame = np.asarray(shot)[:, :, :3]          # BGRA -> BGR
        self.scale = frame.shape[1] / self.region.width
        return np.ascontiguousarray(frame)

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import mss

        start = time.monotonic()
        idx = 0
        min_dt = 1.0 / self.fps_cap if self.fps_cap else 0.0
        with mss.mss() as sct:
            region = self.region.as_mss()
            while self.max_frames is None or idx < self.max_frames:
                t0 = time.monotonic()
                shot = sct.grab(region)
                frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])
                self.scale = frame.shape[1] / self.region.width
                yield idx, t0 - start, frame
                idx += 1
                if min_dt:
                    wait = min_dt - (time.monotonic() - t0)
                    if wait > 0:
                        time.sleep(wait)


class VideoSource:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"could not open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scale = 1.0

    def read_frame(self, index: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"could not read frame {index} of {self.path}")
        return frame

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, idx / self.fps, frame
            idx += 1

    def release(self) -> None:
        self.cap.release()
