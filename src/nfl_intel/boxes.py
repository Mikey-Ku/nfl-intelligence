"""Correct the detector's boxes on the pre-labelled frames, so `nfl train` has something to learn.

`nfl prelabel` saved keyframes to `data/labels/images/` and the detector's own boxes beside
them in `data/labels/labels/` (YOLO text: class, centre x, centre y, width, height, all as
fractions of the frame). Training on those as they are would only teach the detector its own
mistakes. This window is for fixing them, frame by frame:

    uv run nfl boxes data/labels

Mouse: drag on empty ground to draw a box; click inside a box to select it (the smallest
box under the cursor); drag a selected box's corner to resize it, its middle to move it.
Keys: `d` delete the selected box, `r` switch it between player and referee, `u` undo the
last change on this frame, `z` zoom 2x around the cursor (again to zoom out), `n` / `p`
next / previous frame (saved on every turn), `g` go to a frame number, `q` save and quit.

What to fix on each frame: a referee boxed as a player (`r`); two players in one box
(delete it, draw one per man, head to feet); a player with no box (the quarterback under
centre, linemen in a pile); boxes on the crowd or the bench (delete). Frames you have not
touched keep the detector's boxes, which are mostly right, so stop whenever.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WINDOW = "nfl boxes"
CLASSES = ["player", "referee"]
COLOUR = [(60, 200, 60), (40, 160, 255)]
HANDLE = 10                                            # pixels on screen that count as grabbing a corner


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def norm(self) -> Box:
        return Box(self.cls, min(self.x1, self.x2), min(self.y1, self.y2), max(self.x1, self.x2), max(self.y1, self.y2))

    def area(self) -> float:
        b = self.norm()
        return (b.x2 - b.x1) * (b.y2 - b.y1)

    def contains(self, x: float, y: float) -> bool:
        b = self.norm()
        return b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2


def read_yolo(path: Path, w: int, h: int) -> list[Box]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:5])
        out.append(Box(c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def write_yolo(path: Path, boxes: list[Box], w: int, h: int) -> None:
    lines = []
    for b in (b.norm() for b in boxes):
        if b.area() < 16:                                   # a click that never became a box
            continue
        cx, cy = (b.x1 + b.x2) / 2 / w, (b.y1 + b.y2) / 2 / h
        bw, bh = (b.x2 - b.x1) / w, (b.y2 - b.y1) / h
        lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def hit(boxes: list[Box], x: float, y: float) -> int | None:
    """The smallest box under the point, or None."""
    inside = [(b.area(), i) for i, b in enumerate(boxes) if b.contains(x, y)]
    return min(inside)[1] if inside else None


def corner(b: Box, x: float, y: float, reach: float) -> str | None:
    """Which corner of `b` the point is within `reach` of: 'tl', 'tr', 'bl', 'br', or None."""
    b = b.norm()
    for name, (cx, cy) in (("tl", (b.x1, b.y1)), ("tr", (b.x2, b.y1)), ("bl", (b.x1, b.y2)), ("br", (b.x2, b.y2))):
        if abs(x - cx) <= reach and abs(y - cy) <= reach:
            return name
    return None


def resized(b: Box, which: str, x: float, y: float) -> Box:
    b = b.norm()
    if "l" in which:
        b.x1 = x
    else:
        b.x2 = x
    if "t" in which:
        b.y1 = y
    else:
        b.y2 = y
    return b


@dataclass
class View:
    """What part of the frame is on screen: a window `scale` times smaller than the frame, zoomed `zoom` times around (ox, oy)."""
    w: int
    h: int
    scale: float
    zoom: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    def to_frame(self, sx: float, sy: float) -> tuple[float, float]:
        return self.ox + sx / (self.scale * self.zoom), self.oy + sy / (self.scale * self.zoom)

    def to_screen(self, fx: float, fy: float) -> tuple[int, int]:
        return int(round((fx - self.ox) * self.scale * self.zoom)), int(round((fy - self.oy) * self.scale * self.zoom))

    def zoom_at(self, fx: float, fy: float, zoom: float) -> None:
        self.zoom = zoom
        vw, vh = self.w / zoom, self.h / zoom
        self.ox = float(np.clip(fx - vw / 2, 0, self.w - vw))
        self.oy = float(np.clip(fy - vh / 2, 0, self.h - vh))

    def render(self, frame: np.ndarray) -> np.ndarray:
        x0, y0 = int(self.ox), int(self.oy)
        x1, y1 = int(round(self.ox + self.w / self.zoom)), int(round(self.oy + self.h / self.zoom))
        crop = frame[y0:y1, x0:x1]
        size = (int(round(self.w * self.scale)), int(round(self.h * self.scale)))
        return cv2.resize(crop, size, interpolation=cv2.INTER_AREA if self.zoom * self.scale < 1 else cv2.INTER_LINEAR)


def frames_in(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "images").glob("*.jpg")) + sorted((data_dir / "images").glob("*.png"))


def label_path(data_dir: Path, image: Path) -> Path:
    return data_dir / "labels" / (image.stem + ".txt")


def _draw(frame: np.ndarray, boxes: list[Box], selected: int | None, view: View, title: str) -> np.ndarray:
    img = view.render(frame)
    for i, b in enumerate(boxes):
        b = b.norm()
        p1, p2 = view.to_screen(b.x1, b.y1), view.to_screen(b.x2, b.y2)
        colour = (0, 255, 255) if i == selected else COLOUR[b.cls % len(COLOUR)]
        cv2.rectangle(img, p1, p2, colour, 2 if i == selected else 1)
        if b.cls == 1:
            cv2.putText(img, "ref", (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (24, 24, 24), -1)
    cv2.putText(img, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def edit_interactive(data_dir: str | Path, scale: float = 0.5, start: int = 0) -> int:
    """The editing loop. Returns the number of frames whose boxes were changed."""
    data_dir = Path(data_dir)
    images = frames_in(data_dir)
    if not images:
        raise SystemExit(f"no frames in {data_dir / 'images'}: run `nfl prelabel` first")
    k = max(0, min(start, len(images) - 1))
    mouse: dict = {"pos": (0, 0), "down": None, "drag": None}

    def on_mouse(event, x, y, flags, param):
        mouse["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["down"] is not None:
            mouse["drag"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["up"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    changed: set[int] = set()

    def load(i):
        frame = cv2.imread(str(images[i]))
        h, w = frame.shape[:2]
        return frame, read_yolo(label_path(data_dir, images[i]), w, h), View(w, h, scale)

    frame, boxes, view = load(k)
    history: list[list[Box]] = []
    selected: int | None = None
    grab: tuple | None = None                               # (index, corner-or-"move", last frame point, new box?)

    def finish_grab(ux: float, uy: float) -> None:
        """The button came up (or a key ended the drag): keep the box, or drop a new one that was only a click."""
        nonlocal grab, selected
        if grab is None:
            return
        i, _, _, new = grab
        grab = None
        if i >= len(boxes):                                     # the box went away under the drag (undo, delete)
            return
        boxes[i] = boxes[i].norm()
        if new and boxes[i].area() < 16:
            boxes.pop(i)
            history.pop()
            selected = hit(boxes, *view.to_frame(ux, uy))
        else:
            changed.add(k)

    try:
        while True:
            h, w = frame.shape[:2]
            title = (f"{k + 1}/{len(images)}  {images[k].name}  {len(boxes)} boxes ({sum(b.cls == 1 for b in boxes)} ref)  "
                     f"drag=draw  click=select  d=delete  r=player/ref  u=undo  z=zoom  n/p=frames  g=go to  q=save")
            cv2.imshow(WINDOW, _draw(frame, boxes, selected, view, title))
            key = cv2.waitKey(20) & 0xFF
            if mouse["down"] is not None and grab is None and mouse.get("drag") is not None:
                fx, fy = view.to_frame(*mouse["down"])
                reach = HANDLE / (view.scale * view.zoom)
                if selected is not None and selected < len(boxes) and corner(boxes[selected], fx, fy, reach):
                    history.append([Box(**vars(b)) for b in boxes])
                    grab = (selected, corner(boxes[selected], fx, fy, reach), (fx, fy), False)
                elif selected is not None and selected < len(boxes) and boxes[selected].contains(fx, fy):
                    history.append([Box(**vars(b)) for b in boxes])
                    grab = (selected, "move", (fx, fy), False)
                else:
                    history.append([Box(**vars(b)) for b in boxes])
                    boxes.append(Box(0, fx, fy, fx, fy))
                    selected = len(boxes) - 1
                    grab = (selected, "br", (fx, fy), True)
            if grab is not None and mouse.get("drag") is not None:
                i, which, last, new = grab
                if i >= len(boxes):
                    grab = None
                else:
                    fx, fy = view.to_frame(*mouse["drag"])
                    if which == "move":
                        b = boxes[i]
                        dx, dy = fx - last[0], fy - last[1]
                        boxes[i] = Box(b.cls, b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy)
                        grab = (i, which, (fx, fy), new)
                    else:
                        boxes[i] = resized(boxes[i], which, fx, fy)
                mouse["drag"] = None
            if "up" in mouse:
                ux, uy = mouse.pop("up")
                if grab is not None:
                    finish_grab(ux, uy)
                else:
                    selected = hit(boxes, *view.to_frame(ux, uy))
                mouse["down"] = mouse["drag"] = None
            if key != 255 and grab is not None:                     # a key while the button is still down ends the drag
                finish_grab(*mouse["pos"])
                mouse["down"] = mouse["drag"] = None
            if selected is not None and selected >= len(boxes):
                selected = None
            if key == ord("d") and selected is not None:
                history.append([Box(**vars(b)) for b in boxes])
                boxes.pop(selected)
                selected = None
                changed.add(k)
            elif key == ord("r") and selected is not None:
                history.append([Box(**vars(b)) for b in boxes])
                boxes[selected].cls = 1 - boxes[selected].cls
                changed.add(k)
            elif key == ord("u") and history:
                boxes = history.pop()
                selected = None
                changed.add(k)
            elif key == ord("z"):
                if view.zoom == 1.0:
                    view.zoom_at(*view.to_frame(*mouse["pos"]), 2.0)
                else:
                    view.zoom_at(0, 0, 1.0)
            elif key in (ord("n"), ord("p"), ord("g"), ord("q")):
                write_yolo(label_path(data_dir, images[k]), boxes, w, h)
                if key == ord("q"):
                    break
                if key == ord("g"):
                    try:
                        k = max(0, min(int(input(f"frame number (1-{len(images)}): ")) - 1, len(images) - 1))
                    except ValueError:
                        print("not a number")
                else:
                    k = min(k + 1, len(images) - 1) if key == ord("n") else max(k - 1, 0)
                frame, boxes, view = load(k)
                history, selected, grab = [], None, None
    finally:
        write_yolo(label_path(data_dir, images[k]), boxes, w, h)    # whatever happens, the frame on screen is saved
        cv2.destroyWindow(WINDOW)
    print(f"{len(changed)} frames changed this session; labels in {data_dir / 'labels'}")
    return len(changed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Correct the detector's boxes on the pre-labelled frames.")
    ap.add_argument("data", nargs="?", default="data/labels", help="folder with images/ and labels/")
    ap.add_argument("--scale", type=float, default=0.5, help="display scale (4K frames at 0.5 fit a laptop screen)")
    ap.add_argument("--start", type=int, default=1, help="frame number to begin at")
    a = ap.parse_args(argv)
    edit_interactive(a.data, a.scale, a.start - 1)


if __name__ == "__main__":
    main()
