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
    on_screen: bool = True

    def __str__(self) -> str:
        r = self.region
        where = "" if self.on_screen else "  (off screen)"
        return f"[{self.id}] {self.owner} — {self.title or '(untitled)'}  {r.width}x{r.height} at ({r.left},{r.top}){where}"


def list_windows(min_size: int = 200, on_screen_only: bool = False) -> list[WindowInfo]:
    """Normal app windows, on-screen ones first then biggest first. Titles need Screen Recording permission."""
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionAll,
        kCGWindowListOptionOnScreenOnly,
    )

    option = kCGWindowListOptionOnScreenOnly if on_screen_only else kCGWindowListOptionAll
    infos = CGWindowListCopyWindowInfo(option | kCGWindowListExcludeDesktopElements, kCGNullWindowID)
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
            on_screen=bool(w.get("kCGWindowIsOnscreen", False)),
        ))
    out.sort(key=lambda w: (not w.on_screen, -w.region.width * w.region.height))
    return out


def find_window(query: str) -> WindowInfo | None:
    """Best window whose owner or title contains `query` (case-insensitive): on-screen and largest first."""
    q = query.lower()
    for w in list_windows():
        if q in w.owner.lower() or q in w.title.lower():
            return w
    return None


def window_by_id(window_id: int) -> WindowInfo | None:
    for w in list_windows(min_size=1):
        if w.id == window_id:
            return w
    return None


def grab_window(window_id: int) -> np.ndarray | None:
    """Capture one window's contents by ID, even when it is covered or on another Space.

    Nothing drawn over the window (like our own overlay) ends up in the image, so this is
    the preferred live source. Returns None if the window is gone.
    """
    import Quartz

    img = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
        Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution,
    )
    if img is None:
        return None
    w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    if w == 0 or h == 0:
        return None
    bpr = Quartz.CGImageGetBytesPerRow(img)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    arr = np.frombuffer(data, dtype=np.uint8)[: bpr * h].reshape(h, bpr // 4, 4)[:, :w, :3]  # BGRA -> BGR
    return np.ascontiguousarray(arr)


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


class WindowSource:
    """Frames from one window by ID. `region` tracks where the window is on screen."""

    def __init__(self, window: WindowInfo, fps_cap: float | None = None, max_frames: int | None = None):
        self.window = window
        self.region = window.region
        self.fps_cap = fps_cap
        self.max_frames = max_frames
        self.scale = 1.0

    def refresh(self) -> Region | None:
        """Re-read the window's position (it may have moved or resized). None if it closed."""
        info = window_by_id(self.window.id)
        if info is None:
            return None
        self.window = info
        self.region = info.region
        return self.region

    def grab(self) -> np.ndarray:
        frame = grab_window(self.window.id)
        if frame is None:
            raise RuntimeError(f"window {self.window.id} is gone")
        self.scale = frame.shape[1] / max(1, self.region.width)
        return frame

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        start = time.monotonic()
        idx = 0
        min_dt = 1.0 / self.fps_cap if self.fps_cap else 0.0
        while self.max_frames is None or idx < self.max_frames:
            t0 = time.monotonic()
            frame = grab_window(self.window.id)
            if frame is None:
                break
            if idx % 30 == 0 and self.refresh() is None:
                break
            self.scale = frame.shape[1] / max(1, self.region.width)
            yield idx, t0 - start, frame
            idx += 1
            if min_dt:
                wait = min_dt - (time.monotonic() - t0)
                if wait > 0:
                    time.sleep(wait)


class VideoSource:
    def __init__(self, path: str, start: float = 0.0, stop: float | None = None):
        """`start` and `stop` are seconds into the clip (its real capture times when it has them)."""
        self.path = path
        self.start, self.stop = start, stop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"could not open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scale = 1.0
        self.times = self._capture_times(path)
        if self.times is not None and len(self.times) > 1:
            self.fps = (len(self.times) - 1) / (self.times[-1] - self.times[0])       # the rate it was really captured at

    @staticmethod
    def _capture_times(path: str) -> list[float] | None:
        """`nfl capture` writes when each frame was really grabbed next to the clip; the file
        itself plays at a fixed rate, which would stretch or squeeze every speed."""
        from pathlib import Path

        side = Path(path).with_suffix(".times.csv")
        if not side.exists():
            return None
        try:
            times = [float(line.split(",")[1]) for line in side.read_text().splitlines()[1:] if line.strip()]
        except (ValueError, IndexError):
            return None
        return [t - times[0] for t in times] if times else None

    def read_frame(self, index: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"could not read frame {index} of {self.path}")
        return frame

    def _time(self, idx: int) -> float:
        return self.times[idx] if self.times is not None and idx < len(self.times) else idx / self.fps

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        idx = 0
        if self.start > 0:
            while idx < self.frame_count - 1 and self._time(idx) < self.start - 1e-6:
                idx += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            t = self._time(idx)
            if self.stop is not None and t > self.stop + 1e-6:
                break
            yield idx, t, frame
            idx += 1

    def release(self) -> None:
        self.cap.release()
