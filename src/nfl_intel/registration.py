"""Field registration: from the painted lines in a frame to a pixel -> yards homography.

How it works, for one frame:
  1. Find the video rectangle and the paint: bright, colourless, thin structures on the turf,
     with the players' boxes blanked out.
  2. Find the long lines and keep the biggest family of near-parallel ones: the yard lines.
     (Whatever the camera angle, yard lines are the many long parallel lines.)
  3. Number them. Equally spaced lines seen in perspective follow t(k) = (a k + b) / (c k + 1)
     along any line that crosses them; a small search finds the integer numbering that fits.
  4. Find the hash marks: short dashes lying between yard lines, in two rows. Four dashes
     between neighbouring yard lines confirms they are 5 yards apart.
  5. Fit a camera (camera.py): level, fixed position, with pan, tilt and zoom. Every point on
     yard line k must land on x = 5 k and every dash of the hash row on that row's y. Because
     the camera model has only 6 unknowns, one hash row is enough; players usually hide the
     other.

After the first lock the camera is followed frame to frame: the paint near where each line
should be says where it is now, and only pan, tilt and zoom are refit.

The result is metric but relative: x is measured from an arbitrary yard line, and which hash
row was seen is a guess when only one is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Camera, fit_camera, looks_across
from .field import FIELD_WIDTH, HASH_INSET

HASH_NEAR = HASH_INSET                       # 23.58 yd from the near sideline
HASH_FAR = FIELD_WIDTH - HASH_INSET          # 29.75
CANON_WIDTH = 1280                           # cameras are expressed in a 1280-wide version of the video
WORK_WIDTH = 1280                            # first lock: full detail
REFINE_WIDTH = 960                           # following: cheaper


@dataclass
class Line:
    a: float
    b: float
    c: float                                 # a*u + b*v + c = 0 with a^2 + b^2 = 1 (working-image pixels)
    support: float = 0.0                     # painted pixels along it
    index: int | None = None                 # yard-line number (each step is 5 yards)

    @property
    def angle(self) -> float:
        """Direction of the line in degrees, 0..180."""
        return float(np.degrees(np.arctan2(self.a, -self.b)) % 180.0)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        return self.a * pts[:, 0] + self.b * pts[:, 1] + self.c

    def intersect(self, other: Line) -> np.ndarray | None:
        d = self.a * other.b - other.a * self.b
        if abs(d) < 1e-9:
            return None
        return np.array([(self.b * other.c - other.b * self.c) / d, (other.a * self.c - self.a * other.c) / d])

    def clip(self, w: int, h: int) -> np.ndarray | None:
        """The two points where the line enters and leaves a w x h image, or None if it misses."""
        pts = []
        if abs(self.b) > 1e-9:
            for u in (0.0, w - 1.0):
                v = -(self.a * u + self.c) / self.b
                if -0.5 <= v <= h - 0.5:
                    pts.append((u, v))
        if abs(self.a) > 1e-9:
            for v in (0.0, h - 1.0):
                u = -(self.b * v + self.c) / self.a
                if -0.5 <= u <= w - 0.5:
                    pts.append((u, v))
        if len(pts) < 2:
            return None
        pts = np.array(pts)
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        if d[i, j] < 1.0:
            return None
        return np.array([pts[i], pts[j]])

    def points(self, w: int, h: int, n: int = 40) -> np.ndarray:
        """`n` points evenly spaced along the part of the line inside a w x h image."""
        ends = self.clip(w, h)
        if ends is None:
            return np.zeros((0, 2))
        s = np.linspace(0.0, 1.0, n)[:, None]
        pts = ends[0] * (1 - s) + ends[1] * s
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts


@dataclass
class Registration:
    H: np.ndarray                            # full-frame pixels -> field yards
    camera: Camera                           # in canonical video pixels (CANON_WIDTH wide)
    rect: tuple[int, int, int, int]          # the video's rectangle in the full frame (x1, y1, x2, y2)
    residual: float                          # mean error of the constraints, yards
    yard_lines: list[Line] = field(default_factory=list)
    hash_rows: list[Line] = field(default_factory=list)
    ticks_between: float = 0.0               # hash marks between neighbouring yard lines (4 when right)
    notes: list[str] = field(default_factory=list)

    def to_field(self, pts_px) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_pixels(self, pts_yd) -> np.ndarray:
        pts = np.asarray(pts_yd, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)


def full_to_canon(rect) -> np.ndarray:
    """Full-frame pixels -> canonical video pixels."""
    x1, y1, x2, _ = rect
    s_ = CANON_WIDTH / float(x2 - x1)
    return np.array([[s_, 0, -x1 * s_], [0, s_, -y1 * s_], [0, 0, 1.0]])


def homography_for(camera: Camera, rect) -> np.ndarray:
    H = camera.pixel_to_field() @ full_to_canon(rect)
    return H / H[2, 2]


def canon_size(rect) -> tuple[int, int]:
    x1, y1, x2, y2 = rect
    return CANON_WIDTH, int(round((y2 - y1) * CANON_WIDTH / float(x2 - x1)))


# ------------------------------------------------------------------ paint

def video_rect(frame: np.ndarray, band: tuple[int, int], grass: np.ndarray) -> tuple[int, int, int, int]:
    """The video's rectangle inside the captured window.

    Left and right: the run of columns that are mostly grass within the turf rows. Top and
    bottom: grow from the turf rows until a row of the player's dark surround. The camera's
    optical centre is the middle of this rectangle, so it has to be the whole picture, crowd
    included, not just the grass.
    """
    top, bottom = band
    cols = grass[top:bottom].mean(axis=0)
    ok = (cols > 0.25).astype(np.uint8)
    bridge = max(3, int(grass.shape[1] * 0.012) | 1)       # a yard line seen head-on, a cable
    ok = cv2.morphologyEx(ok.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8)).ravel() > 0
    best, start = (0, 0), None
    for i, v in enumerate(np.append(ok, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    x1, x2 = best
    if x2 - x1 < 50:
        return int(x1), int(top), int(x2), int(bottom)
    step = 4                                                       # a quarter of the pixels is plenty
    strip = frame[::step, x1 + (x2 - x1) // 20 : x2 - (x2 - x1) // 20 : step]
    dark = (strip.max(axis=2) < 40).mean(axis=1) > 0.92          # rows that are almost all surround
    r1 = top // step
    while r1 > 0 and not dark[r1 - 1]:
        r1 -= 1
    r2 = min(len(dark), bottom // step + 1)
    while r2 < len(dark) and not dark[r2]:
        r2 += 1
    return int(x1), int(r1 * step), int(x2), int(min(frame.shape[0], r2 * step))


@dataclass
class View:
    """A frame prepared for registration: the video, resized, with its paint found."""
    img: np.ndarray
    mask: np.ndarray                         # painted field lines, turf rows only
    rect: tuple[int, int, int, int]
    scale: float                             # working pixels per full-frame pixel
    to_canon: float                          # canonical pixels per working pixel
    boxes: np.ndarray | None                 # players' boxes in working pixels


def prepare(frame: np.ndarray, field_info, boxes: np.ndarray | None, width: int, rect=None, img: np.ndarray | None = None) -> View | None:
    rect = rect or video_rect(frame, field_info.band, field_info.grass)
    x1, y1, x2, y2 = rect
    if x2 - x1 < 200 or y2 - y1 < 100:
        return None
    scale = width / float(x2 - x1)
    if img is None:
        img = cv2.resize(frame[y1:y2, x1:x2], (width, max(8, int(round((y2 - y1) * scale)))), interpolation=cv2.INTER_AREA)
    wb = None
    if boxes is not None and len(boxes):
        wb = (np.asarray(boxes, dtype=np.float64) - np.array([x1, y1, x1, y1])) * scale
    mask = paint_mask(img, wb, field_info.grass_lab)
    top = int((field_info.band[0] - y1) * scale)
    bottom = int((field_info.band[1] - y1) * scale)
    mask[: max(0, top)] = 0
    mask[max(0, bottom):] = 0
    return View(img, mask, rect, scale, CANON_WIDTH / float(width), wb)


def paint_mask(img: np.ndarray, boxes: np.ndarray | None = None, grass_lab: np.ndarray | None = None) -> np.ndarray:
    """Painted field lines: brighter than the grass, colourless and thin.

    Removed: the players' boxes, and everything in or near clutter. Clutter is decoration
    (end-zone lettering, logos: painted colour that is neither grass nor white) and thick
    white (the border around the field, big letters). Their edges run every which way and
    would be mistaken for field lines, so only open-field paint is kept.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    gL, ga, gb = (110, 105, 160) if grass_lab is None else [int(v) for v in grass_lab]
    w = img.shape[1]
    k = max(9, int(w * 0.012) | 1)
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_TOPHAT, ell)
    chroma = np.abs(a - 128) + np.abs(b - 128)
    white = (L > gL + 22) & (chroma < 45)
    paint = white & (tophat > 14)
    dist_grass = np.sqrt((L - gL) ** 2 + (a - ga) ** 2 + (b - gb) ** 2)
    decor = ((dist_grass > 38) & (chroma >= 45) & (L > gL - 10)).astype(np.uint8)
    decor = cv2.morphologyEx(decor, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    thick = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, ell)
    clutter = cv2.bitwise_or(decor, thick)
    if boxes is not None and len(boxes):                     # white jerseys are thick white too: not clutter
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64):
            cv2.rectangle(clutter, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
    grow = max(7, int(w * 0.03) | 1)
    clutter = cv2.dilate(clutter, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) > 0
    mask = (paint & ~clutter).astype(np.uint8) * 255
    if boxes is not None and len(boxes):
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64):
            mx, my = 0.15 * (x2 - x1), 0.06 * (y2 - y1)
            cv2.rectangle(mask, (int(x1 - mx), int(y1 - my)), (int(x2 + mx), int(y2 + my)), 0, -1)
    return mask


# ------------------------------------------------------------------ lines

def _line_from_segment(x1, y1, x2, y2) -> Line:
    a, b = y2 - y1, x1 - x2
    n = np.hypot(a, b)
    a, b = a / n, b / n
    return Line(float(a), float(b), float(-(a * x1 + b * y1)))


def _refit(line: Line, ys: np.ndarray, xs: np.ndarray, tol: float) -> Line | None:
    """Total-least-squares line through the paint pixels within `tol` of the line."""
    d = np.abs(line.a * xs + line.b * ys + line.c)
    m = d < tol
    if m.sum() < 30:
        return None
    pts = np.column_stack([xs[m], ys[m]]).astype(np.float64)
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    direction = vt[0]
    a, b = -direction[1], direction[0]
    return Line(float(a), float(b), float(-(a * mean[0] + b * mean[1])), support=float(m.sum()))


def _same_line(a: Line, b: Line, w: int, h: int, tol: float = 9.0) -> bool:
    ends = a.clip(w, h)
    if ends is None:
        return False
    return bool(np.abs(b.distance(ends)).max() < tol)


def long_lines(mask: np.ndarray, min_support_frac: float = 0.15, min_painted: float = 0.3, max_lines: int = 40) -> list[Line]:
    """Long painted lines. The accumulating Hough transform doesn't care that players break a
    line into pieces; each candidate is then refit on the paint around it."""
    h, w = mask.shape
    votes = int(min(w, h) * min_support_frac)
    found = cv2.HoughLines(mask, 1, np.pi / 360, votes)
    if found is None:
        return []
    ys, xs = np.nonzero(mask)
    fat = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    lines: list[Line] = []
    for rho, theta in found.reshape(-1, 2)[:400]:
        cand = Line(float(np.cos(theta)), float(np.sin(theta)), float(-rho))
        if any(_same_line(cand, other, w, h, 14.0) for other in lines):
            continue
        ln = _refit(cand, ys, xs, tol=4.0)
        if ln is None:
            continue
        span = ln.points(w, h, 200)
        if len(span) < 20:
            continue
        pi = span.astype(int)
        painted = float((fat[pi[:, 1], pi[:, 0]] > 0).mean())
        length = float(np.linalg.norm(span[-1] - span[0]))
        if painted < min_painted or painted * length < votes:
            continue
        ln.support = painted * length
        if any(_same_line(ln, other, w, h) for other in lines):
            continue
        lines.append(ln)
        if len(lines) >= max_lines:
            break
    return lines


def family_angle(lines: list[Line]) -> float:
    """Average direction of a set of lines in degrees, 0..180. Line angles wrap at 180 (1 and 179
    degrees are nearly the same line), so the average is taken on doubled angles."""
    a = np.radians([ln.angle for ln in lines]) * 2.0
    wts = np.array([max(ln.support, 1.0) for ln in lines])
    return float(np.degrees(np.arctan2((np.sin(a) * wts).sum(), (np.cos(a) * wts).sum()) / 2.0) % 180.0)


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _vanishing_point(a: Line, b: Line) -> np.ndarray:
    """Homogeneous intersection of two lines (third coordinate ~0 when they are parallel)."""
    return np.cross([a.a, a.b, a.c], [b.a, b.b, b.c])


def _passes_through(ln: Line, vp: np.ndarray, centre: np.ndarray, tol_deg: float) -> bool:
    """Does the line point at the vanishing point, as seen from where it crosses the image?"""
    foot = centre - (ln.a * centre[0] + ln.b * centre[1] + ln.c) * np.array([ln.a, ln.b])
    if abs(vp[2]) < 1e-9 * max(1.0, np.abs(vp[:2]).max()):
        to_vp = vp[:2]
    else:
        to_vp = vp[:2] / vp[2] - foot
    n = np.linalg.norm(to_vp)
    if n < 1e-9:
        return False
    along = np.array([-ln.b, ln.a])
    cosang = abs(along @ to_vp) / n
    return cosang >= np.cos(np.radians(tol_deg))


def yard_line_family(lines: list[Line], w: int = WORK_WIDTH, h: int = 720, tol_deg: float = 1.5) -> tuple[list[Line], list[Line]]:
    """The biggest set of lines that meet at one vanishing point: the yard lines.

    Parallel lines on the field converge to a single point in the picture. Lettering edges
    and borders that merely look parallel usually miss it.
    """
    if len(lines) < 2:
        return list(lines), []
    centre = np.array([w / 2.0, h / 2.0])
    best: list[Line] = []
    best_score = -1.0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if angle_diff(lines[i].angle, lines[j].angle) > 25.0:
                continue
            vp = _vanishing_point(lines[i], lines[j])
            fam = [ln for ln in lines if _passes_through(ln, vp, centre, tol_deg)]
            score = sum(ln.support for ln in fam)
            if score > best_score:
                best, best_score = fam, score
    rest = [ln for ln in lines if all(ln is not f for f in best)]
    return merge_family(best, w, h), rest


def merge_family(family: list[Line], w: int, h: int, min_gap_frac: float = 0.02) -> list[Line]:
    """One line per painted line: fits that cross the middle of the picture within a few pixels
    of each other are the same line, and the best-supported one is kept."""
    if len(family) < 2:
        return list(family)
    origin, direction = _transversal(family, w, h)
    t = _positions(family, origin, direction)
    order = np.argsort(t)
    gap = max(w, h) * min_gap_frac
    out: list[Line] = []
    last_t = None
    for i in order:
        if not np.isfinite(t[i]):
            continue
        if last_t is not None and t[i] - last_t < gap:
            if family[i].support > out[-1].support:
                out[-1] = family[i]
            last_t = t[i]
            continue
        out.append(family[i])
        last_t = t[i]
    return out


# ------------------------------------------------------------------ numbering

def _transversal(family: list[Line], w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """A line across the family through the image centre: origin and unit direction."""
    mean_angle = np.radians(family_angle(family))
    direction = np.array([-np.sin(mean_angle), np.cos(mean_angle)])       # perpendicular to the family
    return np.array([w / 2.0, h / 2.0]), direction


def _positions(family: list[Line], origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Where each line crosses the transversal, as a signed distance from the origin."""
    t = []
    for ln in family:
        denom = ln.a * direction[0] + ln.b * direction[1]
        t.append(-(ln.a * origin[0] + ln.b * origin[1] + ln.c) / denom if abs(denom) > 1e-9 else np.nan)
    return np.array(t)


def _fit_projective(k: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    """t = (a k + b) / (c k + 1): solve a, b, c by least squares."""
    A = np.column_stack([k, np.ones_like(k), -k * t])
    try:
        sol, *_ = np.linalg.lstsq(A, t, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return sol


def _index_of(t: np.ndarray, abc: np.ndarray) -> np.ndarray:
    a, b, c = abc
    denom = a - c * t
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denom) > 1e-9, (t - b) / denom, np.nan)


def numberings(family: list[Line], w: int, h: int, max_step: int = 3, tol: float = 0.1) -> list[list[Line]]:
    """Every way of numbering the yard lines so that equal steps in number are equal steps on
    the field (best first). Lines that don't fit a numbering are left out of it."""
    if len(family) < 3:
        return []
    origin, direction = _transversal(family, w, h)
    t = _positions(family, origin, direction)
    order = np.argsort(t)
    fam = [family[i] for i in order]
    t = t[order]
    n = len(fam)
    seen: dict[tuple, float] = {}
    for i in range(n - 2):
        for j in range(i + 1, min(n - 1, i + 4)):
            for l in range(j + 1, min(n, j + 4)):
                for s1 in range(1, max_step + 1):
                    for s2 in range(1, max_step + 1):
                        abc = _fit_projective(np.array([0.0, s1, s1 + s2]), t[[i, j, l]])
                        if abc is None:
                            continue
                        idx = _index_of(t, abc)
                        if np.any(~np.isfinite(idx)):
                            continue
                        near = np.round(idx)
                        good = np.abs(idx - near) < tol
                        gi = near[good]
                        if len(gi) < 3 or np.any(np.diff(gi) <= 0):
                            continue
                        key = tuple((int(m), int(near[m] - gi[0])) for m in np.nonzero(good)[0])
                        span = gi[-1] - gi[0]
                        # image distance per 5 yards between neighbours: a real camera is far away,
                        # so it changes gently; a wrong numbering needs wild foreshortening to fit
                        per_step = np.abs(np.diff(t[good])) / np.diff(gi)
                        wild = float(np.log(per_step.max() / max(per_step.min(), 1e-6)))
                        score = good.sum() * 10.0 - (span - (good.sum() - 1)) * 2.0 - 4.0 * wild
                        seen[key] = max(seen.get(key, -1e9), score)
    out = []
    for key, _ in sorted(seen.items(), key=lambda kv: -kv[1]):
        lines = []
        for m, index in key:
            ln = fam[m]
            lines.append(Line(ln.a, ln.b, ln.c, ln.support, index))
        out.append(lines)
    return out


# ------------------------------------------------------------------ hash marks

def hash_ticks(mask: np.ndarray, yard_lines: list[Line], long: list[Line], angle: float | None = None) -> np.ndarray:
    """Centres of the short dashes: small paint blobs, elongated along the yard-line direction,
    not part of any long line. `angle` overrides the direction taken from `yard_lines`."""
    h, w = mask.shape
    clean = mask.copy()
    for ln in long:
        ends = ln.clip(w, h)
        if ends is not None:
            cv2.line(clean, tuple(ends[0].astype(int)), tuple(ends[1].astype(int)), 0, 9)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if angle is None:
        if not yard_lines:
            return np.zeros((0, 2))
        angle = family_angle(yard_lines)
    ang = np.radians(angle)
    along = np.array([np.cos(ang), np.sin(ang)])
    out = []
    max_len = w * 0.065                       # dashes near the camera are several times longer than far ones
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 6 or max(bw, bh) > max_len or area > max_len * 8:
            continue
        ys, xs = np.nonzero(labels[stats[i, 1]:stats[i, 1] + bh, stats[i, 0]:stats[i, 0] + bw] == i)
        if len(xs) < 6:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        pts -= pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts, full_matrices=False)
        if sv[1] > 0 and sv[0] / sv[1] < 1.8:
            continue                                    # round blob, not a dash
        if abs(vt[0] @ along) < 0.85:
            continue                                    # a dash, but pointing the wrong way
        length = float(np.ptp(pts @ vt[0]))
        out.append((cents[i][0], cents[i][1], length))
    return np.array(out).reshape(-1, 3)


def tick_tolerance(ticks: np.ndarray, floor: float = 5.0) -> float:
    """How far a dash's centre may sit from its row. A dash is painted across the row, and
    only part of a faint one may be seen, so its centre wanders by up to half its length."""
    if ticks.shape[1] < 3 or len(ticks) == 0:
        return floor
    return float(max(floor, 0.55 * np.median(ticks[:, 2])))


def _fit_through(pts: np.ndarray) -> Line:
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    a, b = -vt[0][1], vt[0][0]
    return Line(float(a), float(b), float(-(a * mean[0] + b * mean[1])), support=float(len(pts)))


def row_candidates(ticks: np.ndarray, yard_lines: list[Line], w: int, h: int, tol: float | None = None, n: int = 6) -> list[Line]:
    """Lines across the yard lines that pass through several dashes, best-supported first."""
    if len(ticks) < 5 or len(yard_lines) < 2:
        return []
    tol = tick_tolerance(ticks) if tol is None else tol
    ticks = ticks[:, :2]
    rng = np.random.default_rng(0)
    yard_angle = family_angle(yard_lines)
    rows: list[Line] = []
    remaining = ticks.copy()
    for _ in range(n):
        if len(remaining) < 4:
            break
        best, best_in = None, None
        for _ in range(300):
            i, j = rng.choice(len(remaining), 2, replace=False)
            if np.linalg.norm(remaining[i] - remaining[j]) < w * 0.06:
                continue
            ln = _line_from_segment(*remaining[i], *remaining[j])
            if angle_diff(ln.angle, yard_angle) < 25.0:
                continue                                # rows cross the yard lines, they don't follow them
            inl = np.abs(ln.distance(remaining)) < tol
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = ln, inl
        if best is None or best_in.sum() < 4:
            break
        row = _fit_through(remaining[best_in])
        rows.append(row)
        remaining = remaining[np.abs(row.distance(remaining)) > max(tol * 3, w * 0.02)]
    return rows


def row_quality(row: Line, ticks: np.ndarray, yard_lines: list[Line], tol: float | None = None) -> tuple[int, int]:
    """How many of the dashes on this row sit on whole yards, out of how many are on it.

    Hash marks are one yard apart, so between two yard lines they fall at 1/5, 2/5, 3/5, 4/5 of
    the way (in the field's own measure). Blobs that merely line up, or a cable hanging in
    front of the camera, don't.
    """
    lines = [ln for ln in yard_lines if ln.index is not None]
    tol = tick_tolerance(ticks) if tol is None else tol
    ticks = ticks[:, :2]
    on = ticks[np.abs(row.distance(ticks)) < tol]
    if len(lines) < 2 or len(on) < 3:
        return 0, len(on)
    direction = np.array([-row.b, row.a])
    origin = -row.c * np.array([row.a, row.b])
    t_ref, x_ref = [], []
    for ln in lines:
        p = row.intersect(ln)
        if p is not None:
            t_ref.append((p - origin) @ direction)
            x_ref.append(5.0 * ln.index)
    if len(t_ref) < 2:
        return 0, len(on)
    t_ref, x_ref = np.array(t_ref), np.array(x_ref)
    t = (on - origin) @ direction
    if len(t_ref) >= 3:
        abc = _fit_projective(t_ref, x_ref)             # x = (a t + b) / (c t + 1)
        if abc is None:
            return 0, len(on)
        x = (abc[0] * t + abc[1]) / (abc[2] * t + 1.0)
    else:
        x = np.interp(t, np.sort(t_ref), x_ref[np.argsort(t_ref)])
    inside = (x > x_ref.min() - 6) & (x < x_ref.max() + 6)
    off = np.abs(x - np.round(x))
    return int(((off < 0.17) & inside).sum()), int(inside.sum())


def ticks_between(ticks: np.ndarray, row: Line, yard_lines: list[Line], tol: float | None = None) -> float:
    """Median number of hash marks between neighbouring numbered yard lines along a row."""
    tol = tick_tolerance(ticks) if tol is None else tol
    ticks = ticks[:, :2]
    on = ticks[np.abs(row.distance(ticks)) < tol]
    if len(on) < 3:
        return 0.0
    lines = sorted([ln for ln in yard_lines if ln.index is not None], key=lambda ln: ln.index)
    counts = []
    for a, b in zip(lines, lines[1:]):
        if b.index - a.index != 1:
            continue
        da, db = a.distance(on), b.distance(on)
        counts.append(int(((da * db) < 0).sum()))       # on opposite sides of the two lines = between them
    return float(np.median(counts)) if counts else 0.0


# ------------------------------------------------------------------ first lock

def _line_constraints(lines: list[Line], w: int, h: int, mask: np.ndarray, to_canon: float, n: int = 40) -> np.ndarray:
    """Points on each numbered yard line where there is paint, as (u, v, x, weight) in canonical pixels."""
    fat = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    out = []
    for ln in lines:
        if ln.index is None:
            continue
        pts = ln.points(w, h, n)
        if len(pts) == 0:
            continue
        pi = np.round(pts).astype(int)
        on = fat[pi[:, 1], pi[:, 0]] > 0
        out.extend((p[0] * to_canon, p[1] * to_canon, 5.0 * ln.index, 1.0) for p in pts[on])
    return np.array(out).reshape(-1, 4)


def _row_constraints(row: Line, ticks: np.ndarray, y: float, to_canon: float) -> np.ndarray:
    tol = tick_tolerance(ticks)
    on = ticks[np.abs(row.distance(ticks[:, :2])) < tol, :2]
    # a dash is painted across the row: use where the row crosses it, not its wandering centre
    foot = on - row.distance(on)[:, None] * np.array([row.a, row.b])
    return np.array([(p[0] * to_canon, p[1] * to_canon, y, 1.0) for p in foot]).reshape(-1, 4)


def plausible(H: np.ndarray, rect, boxes: np.ndarray | None, field_info=None, margin: float = 3.0) -> bool:
    """A mapping is only believable if the people standing on the turf land inside the field
    and the picture doesn't show more grass than a field has."""
    x1, y1, x2, y2 = rect
    probe = np.array([[x1, y2], [x2, y2], [(x1 + x2) / 2, (y1 + 2 * y2) / 3]], dtype=np.float64)
    corners = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.all(np.isfinite(corners)) or np.ptp(corners[:, 1]) > FIELD_WIDTH + 25 or np.ptp(corners[:, 0]) > 110:
        return False
    if boxes is None or len(boxes) < 6:
        return True
    b = np.asarray(boxes, dtype=np.float64)
    feet = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
    if field_info is not None:
        top, bottom = field_info.band
        on_turf = np.array([field_info.contains_foot(x, y) and y > top + 0.2 * (bottom - top) for x, y in feet])
        feet = feet[on_turf]
        if len(feet) < 6:
            return True
    yd = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H).reshape(-1, 2)
    inside = (yd[:, 1] > -margin) & (yd[:, 1] < FIELD_WIDTH + margin)
    return bool(inside.mean() >= 0.9)


def _rescale_across(cam: Camera, err: float, both: bool, option: list[Line], ticks: np.ndarray, view: View, cw: int, ch: int):
    """Check the scale across the field against the hash marks themselves.

    Under the fitted camera the dashes should fall in two rows 6.17 yards apart. With a long
    lens looking down the field, one row and a camera model leave that scale loose; if the
    dashes form two clear rows at some other spacing, refit with both rows pinned.
    """
    t_canon = ticks[:, :2] * view.to_canon
    t_yd = cv2.perspectiveTransform(t_canon.reshape(-1, 1, 2), cam.pixel_to_field()).reshape(-1, 2)
    x_lo = 5.0 * min(ln.index for ln in option) - 3.0
    x_hi = 5.0 * max(ln.index for ln in option) + 3.0
    use = (np.abs(t_yd[:, 0] - np.round(t_yd[:, 0])) < 0.25) & (t_yd[:, 0] > x_lo) & (t_yd[:, 0] < x_hi)
    if use.sum() < 8:
        return cam, err, both
    y = t_yd[use, 1]
    pts = t_canon[use]
    # two densest rows: 1-d clustering by histogram peaks
    edges = np.arange(y.min() - 1.0, y.max() + 2.0, 1.0)
    hist, _ = np.histogram(y, edges)
    smooth = np.convolve(hist, [0.5, 1.0, 0.5], mode="same")
    first = int(np.argmax(smooth))
    masked = smooth.copy()
    masked[max(0, first - 2): first + 3] = 0
    second = int(np.argmax(masked))
    if masked[second] < 3 or smooth[first] < 4:
        return cam, err, both
    c1, c2 = sorted([edges[first] + 0.5, edges[second] + 0.5])
    in1, in2 = np.abs(y - c1) < 1.3, np.abs(y - c2) < 1.3
    if in1.sum() < 4 or in2.sum() < 4:
        return cam, err, both
    spacing = float(np.median(y[in2]) - np.median(y[in1]))
    if abs(spacing - (HASH_FAR - HASH_NEAR)) < 0.5:
        return cam, err, both                       # already right
    cons_x = _line_constraints(option, view.mask.shape[1], view.mask.shape[0], view.mask, view.to_canon)
    cons_y = np.vstack([
        np.column_stack([pts[in1], np.full(in1.sum(), HASH_NEAR), np.ones(in1.sum())]),
        np.column_stack([pts[in2], np.full(in2.sum(), HASH_FAR), np.ones(in2.sum())]),
    ])
    fit = fit_camera(cons_x, cons_y, cw, ch)
    if fit is None or fit[1] > max(0.25, err * 2.5):
        return cam, err, both
    return fit[0], fit[1], True


def register(frame: np.ndarray, field_info, boxes: np.ndarray | None = None, max_error: float = 0.35) -> Registration | None:
    """First lock from a single frame. `field_info` is fieldmask.FieldInfo; boxes are full-frame xyxy."""
    view = prepare(frame, field_info, boxes, WORK_WIDTH)
    if view is None:
        return None
    h, w = view.mask.shape
    lines = long_lines(view.mask)
    family, _ = yard_line_family(lines, w, h)
    if len(family) < 3:
        return None
    ticks = hash_ticks(view.mask, family, family)
    if len(ticks) < 5:
        return None
    cw, ch = canon_size(view.rect)
    best = None
    for option in numberings(family, w, h)[:4]:
        ranked = []
        for row in row_candidates(ticks, option, w, h):
            good, total = row_quality(row, ticks, option)
            if good >= 4:
                ranked.append((good, good / max(total, 1), row))
        strong = [r for r in ranked if r[0] >= 5 and r[1] >= 0.6]
        if not strong:
            continue
        strong.sort(key=lambda r: -r[0])
        row = strong[0][2]
        cons_x = _line_constraints(option, w, h, view.mask, view.to_canon)
        centre = np.array([w / 2.0, h / 2.0])
        # a second row, if one can be seen: near-parallel to the first and clear of it
        second = None
        for _, _, other in sorted(ranked, key=lambda r: -r[0]):
            if other is row or angle_diff(other.angle, row.angle) > 14.0:
                continue
            p_mid = row.points(w, h, 3)
            if len(p_mid) == 3 and abs(other.distance(p_mid[1:2])[0]) > w * 0.05:
                second = other
                break
        trials = []
        if second is not None:
            for y1_, y2_ in ((HASH_NEAR, HASH_FAR), (HASH_FAR, HASH_NEAR)):
                cons_y = np.vstack([_row_constraints(row, ticks, y1_, view.to_canon),
                                    _row_constraints(second, ticks, y2_, view.to_canon)])
                trials.append((cons_y, [row, second], True))
        trials.append((_row_constraints(row, ticks, HASH_NEAR, view.to_canon), [row], False))
        for cons_y, used, both in trials:
            if not both and best is not None and best[6] and best[2] < 0.15:
                continue                                        # two rows already fit well
            fit = fit_camera(cons_x, cons_y, cw, ch)
            if fit is None:
                continue
            cam, err = fit
            H_try = homography_for(cam, view.rect)
            if not plausible(H_try, view.rect, boxes, field_info):
                continue
            score = err - (0.05 if both else 0.0)               # two rows beat one when they fit about as well
            if best is None or score < best[0]:
                counts = [c for c in [ticks_between(ticks, row, option)] if c > 0]
                best = (score, cam, err, option, used, counts[0] if counts else 0.0, both)
        if best is not None and best[2] < 0.12:
            break
    if best is None or best[2] > max_error:
        return None
    _, cam, err, option, rows, between, both = best
    notes = []
    cam, err, both = _rescale_across(cam, err, both, option, ticks, view, cw, ch)
    if not both:
        # which hash row was that? look for the other one a hash-width away on either side
        Hc = cam.pixel_to_field()
        t_canon = ticks[:, :2] * view.to_canon
        t_yd = cv2.perspectiveTransform(t_canon.reshape(-1, 1, 2), Hc).reshape(-1, 2)
        whole = np.abs(t_yd[:, 0] - np.round(t_yd[:, 0])) < 0.2
        gap = HASH_FAR - HASH_NEAR
        beyond = int((whole & (np.abs(t_yd[:, 1] - (HASH_NEAR + gap)) < 1.5)).sum())
        before = int((whole & (np.abs(t_yd[:, 1] - (HASH_NEAR - gap)) < 1.5)).sum())
        if before > beyond and before >= 3:
            cam.C = cam.C + np.array([0.0, gap, 0.0])              # the row we used was the far one
        elif max(before, beyond) < 3:
            notes.append("only one hash row visible: positions across the field may be off by 6 yards")
    if not looks_across(cam):
        return None        # end-zone angle: the across-field scale isn't reliable yet, so no mapping beats a wrong one
    H = homography_for(cam, view.rect)
    if not plausible(H, view.rect, boxes, field_info):
        return None
    reg = Registration(H=H, camera=cam, rect=view.rect, residual=err, yard_lines=option, hash_rows=rows,
                       ticks_between=between, notes=notes)
    return reg


# ------------------------------------------------------------------ following

def _search_along_normals(mask: np.ndarray, pts: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """For points along a predicted line, find the nearest paint across the line.
    Returns (matched points, which input points found paint)."""
    h, w = mask.shape
    if len(pts) < 3:
        return np.zeros((0, 2)), np.zeros(len(pts), bool)
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    offs = np.arange(-radius, radius + 1)
    q = pts[:, None, :] + offs[None, :, None] * normal[:, None, :]
    qi = np.round(q).astype(int)
    inside = (qi[..., 0] >= 0) & (qi[..., 0] < w) & (qi[..., 1] >= 0) & (qi[..., 1] < h)
    hit = np.zeros(inside.shape, bool)
    hit[inside] = mask[qi[..., 1][inside], qi[..., 0][inside]] > 0
    cost = np.where(hit, np.abs(offs)[None, :], 10 ** 6)
    best = cost.argmin(axis=1)
    found = cost[np.arange(len(pts)), best] < 10 ** 6
    if found.sum() >= 4:                                   # a line moves as a whole: drop stray hits
        d = offs[best]
        med = np.median(d[found])
        found &= np.abs(d - med) <= max(3, radius // 3)
    return q[np.arange(len(pts)), best][found], found


def refine(camera: Camera, frame: np.ndarray, field_info, rect, boxes: np.ndarray | None = None,
           guide: np.ndarray | None = None, radius_frac: float = 0.014, img: np.ndarray | None = None) -> tuple[Camera, dict] | None:
    """Follow the camera: find the paint near where each line should be and refit pan, tilt
    and zoom. `guide` is a full-frame pixel -> yards mapping used to predict where the lines
    are (the last mapping carried along by the camera's motion); default is the camera itself.
    """
    view = prepare(frame, field_info, boxes, REFINE_WIDTH, rect, img)
    if view is None:
        return None
    h, w = view.mask.shape
    fat = cv2.dilate(view.mask, np.ones((3, 3), np.uint8))
    radius = max(5, int(w * radius_frac))
    H_full = guide if guide is not None else homography_for(camera, rect)
    x1, y1 = rect[0], rect[1]
    to_work = np.array([[view.scale, 0, -x1 * view.scale], [0, view.scale, -y1 * view.scale], [0, 0, 1.0]])
    Hw = H_full @ np.linalg.inv(to_work)                    # working pixels -> yards
    Hw_inv = np.linalg.inv(Hw)
    rows_on = np.nonzero(view.mask.any(axis=1))[0]
    if len(rows_on) < 10:
        return None
    top, bottom = int(rows_on[0]), int(rows_on[-1])

    def project(pts_yd):
        q = cv2.perspectiveTransform(np.asarray(pts_yd, dtype=np.float64).reshape(-1, 1, 2), Hw_inv).reshape(-1, 2)
        ok = np.all(np.isfinite(q), axis=1) & (q[:, 0] >= 2) & (q[:, 0] < w - 2) & (q[:, 1] >= top) & (q[:, 1] <= bottom)
        return q, ok

    probe = np.array([[0, top], [w, top], [w, bottom], [0, bottom]], dtype=np.float64)
    corners = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), Hw).reshape(-1, 2)
    if not np.all(np.isfinite(corners)) or np.ptp(corners[:, 0]) > 150 or np.ptp(corners[:, 1]) > 150:
        return None
    lo, hi = int(np.floor(corners[:, 0].min() / 5)) - 1, int(np.ceil(corners[:, 0].max() / 5)) + 1
    ys = np.arange(-1.0, FIELD_WIDTH + 1.0, 1.0)
    cons_x, cons_y, used_lines, erase = [], [], 0, []
    for k in range(lo, hi + 1):
        q, ok = project(np.column_stack([np.full_like(ys, 5.0 * k), ys]))
        if ok.sum() < 8:
            continue
        erase.append(q[ok])
        m, found = _search_along_normals(fat, q[ok], radius)
        if found.sum() < max(6, 0.3 * ok.sum()):
            continue
        used_lines += 1
        cons_x.extend((p[0], p[1], 5.0 * k, 1.0) for p in m)
    xs = np.arange(5.0 * lo, 5.0 * hi + 0.1, 1.0)
    used_side = 0
    for y in (0.0, FIELD_WIDTH):
        q, ok = project(np.column_stack([xs, np.full_like(xs, y)]))
        if ok.sum() < 8:
            continue
        m, found = _search_along_normals(fat, q[ok], radius)
        if found.sum() < max(6, 0.5 * ok.sum()):
            continue
        used_side += 1
        cons_y.extend((p[0], p[1], y, 1.0) for p in m)
    clean = view.mask.copy()
    for pts in erase:
        cv2.polylines(clean, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 0, 7)
    mid = Hw_inv @ np.array([corners[:, 0].mean(), FIELD_WIDTH / 2, 1.0])
    mid2 = Hw_inv @ np.array([corners[:, 0].mean(), FIELD_WIDTH / 2 + 5.0, 1.0])
    d = mid2[:2] / mid2[2] - mid[:2] / mid[2]
    yard_angle = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    ticks = hash_ticks(clean, [], [], angle=yard_angle)
    used_ticks = 0
    if len(ticks):
        centres = np.ascontiguousarray(ticks[:, :2])
        tick_yd = cv2.perspectiveTransform(centres.reshape(-1, 1, 2), Hw).reshape(-1, 2)
        whole = np.abs(tick_yd[:, 0] - np.round(tick_yd[:, 0])) < 0.3
        for row in (HASH_NEAR, HASH_FAR):
            near = whole & (np.abs(tick_yd[:, 1] - row) < 1.5)
            used_ticks += int(near.sum())
            cons_y.extend((p[0], p[1], row, 1.0) for p in centres[near])
    if used_lines < 2 or len(cons_x) < 12:
        return None
    if len(cons_y) < 3:
        # nothing says where we are across the field this frame: hold it with the guide's own opinion
        gx = np.linspace(0.2 * w, 0.8 * w, 4)
        gpts = np.column_stack([gx, np.full_like(gx, (top + bottom) / 2.0)])
        gy = cv2.perspectiveTransform(gpts.reshape(-1, 1, 2), Hw).reshape(-1, 2)[:, 1]
        cons_y.extend((p[0], p[1], y_, 0.5) for p, y_ in zip(gpts, gy))
    cons_x, cons_y = np.array(cons_x), np.array(cons_y)
    cons_x[:, :2] *= view.to_canon
    cons_y[:, :2] *= view.to_canon
    cw, ch = canon_size(rect)
    fit = fit_camera(cons_x, cons_y, cw, ch, start=camera, fix_position=True)
    if fit is None:
        return None
    cam, err = fit
    if err > 0.4 or abs(np.log(cam.f / camera.f)) > 0.25 or abs(cam.pan - camera.pan) > 0.15 or abs(cam.tilt - camera.tilt) > 0.1:
        return None
    return cam, {"lines": used_lines, "sidelines": used_side, "ticks": used_ticks, "residual": err}


def camera_motion(prev_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
    """Homography taking points in the previous frame to the current one, from how the
    background moved. Players move on their own, so they are masked out and RANSAC ignores
    whatever still disagrees."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=mask)
    if pts is None or len(pts) < 12:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    if ok.sum() < 12:
        return None
    M, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
    if M is None or inl is None or inl.sum() < 10:
        return None
    return M


class FieldRegistrar:
    """Keeps a pixel -> yards mapping through a shot.

    First lock: a full registration from the painted lines (slow; needs a clear look at the
    field; retried every few frames until it works). After that the camera is followed: its
    motion carries the mapping to the next frame, and the paint refits pan, tilt and zoom.
    If the paint can't be found for `max_misses` frames in a row the mapping is dropped
    instead of being left to drift.
    """

    def __init__(self, retry_every: int = 8, motion_width: int = 640, max_misses: int = 25, refine_every: int = 2):
        self.retry_every = retry_every
        self.motion_width = motion_width
        self.max_misses = max_misses
        self.refine_every = refine_every
        self.reset()

    def reset(self) -> None:
        self.H: np.ndarray | None = None
        self.camera: Camera | None = None
        self.last: Registration | None = None
        self.rect: tuple[int, int, int, int] | None = None
        self.age = 0                       # frames since the paint last confirmed the mapping
        self.frames = 0
        self.full_ok = 0
        self.full_tried = 0
        self.refined = 0
        self.info: dict = {}
        self._since_try = 10 ** 6
        self._prev_gray: np.ndarray | None = None
        self._scale = 1.0
        self._field_seen = None
        self._retry = False
        self._wait = self.retry_every

    @property
    def locked(self) -> bool:
        return self.H is not None

    def _working(self, frame: np.ndarray, rect) -> tuple[np.ndarray, np.ndarray]:
        """One resize of the video per frame: the working image, and a small grey one for motion."""
        x1, y1, x2, y2 = rect
        scale = REFINE_WIDTH / float(x2 - x1)
        img = cv2.resize(frame[y1:y2, x1:x2], (REFINE_WIDTH, max(8, int(round((y2 - y1) * scale)))), interpolation=cv2.INTER_AREA)
        self._scale = self.motion_width / float(x2 - x1)
        small = cv2.resize(img, (self.motion_width, max(8, int(round((y2 - y1) * self._scale)))), interpolation=cv2.INTER_AREA)
        return img, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _player_mask(self, shape, rect, boxes) -> np.ndarray:
        mask = np.full(shape, 255, np.uint8)
        if boxes is not None and len(boxes):
            x1, y1, _, _ = rect
            for bx1, by1, bx2, by2 in np.asarray(boxes, dtype=np.float64):
                p1 = (int((bx1 - x1) * self._scale) - 3, int((by1 - y1) * self._scale) - 3)
                p2 = (int((bx2 - x1) * self._scale) + 3, int((by2 - y1) * self._scale) + 3)
                cv2.rectangle(mask, p1, p2, 0, -1)
        return mask

    def update(self, frame: np.ndarray, field_info, boxes: np.ndarray | None = None) -> np.ndarray | None:
        """Returns the current full-frame pixel -> yards homography, or None when there is no lock."""
        self.frames += 1
        if field_info is None:
            return self.H
        if field_info is not self._field_seen or self.rect is None:     # the video's rectangle rarely moves
            self._field_seen = field_info
            rect = video_rect(frame, field_info.band, field_info.grass)
            if rect[2] - rect[0] < 200 or rect[3] - rect[1] < 100:
                return self.H
            if self.rect is None or max(abs(a - b) for a, b in zip(rect, self.rect)) > 12:
                if self.rect is not None and self.H is not None:
                    self.H, self.camera = None, None          # the player window moved: start over
                self.rect = rect
                self._prev_gray = None
        img, gray = self._working(frame, self.rect)

        if self.camera is not None and self.H is not None:
            guide = self.H
            if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
                M = camera_motion(self._prev_gray, gray, self._player_mask(gray.shape, self.rect, boxes))
                if M is not None:
                    x1, y1 = self.rect[0], self.rect[1]
                    s_ = self._scale
                    to_small = np.array([[s_, 0, -x1 * s_], [0, s_, -y1 * s_], [0, 0, 1.0]])
                    M_full = np.linalg.inv(to_small) @ M @ to_small      # previous full px -> current full px
                    guide = self.H @ np.linalg.inv(M_full)
                    guide /= guide[2, 2]
            self.H = guide
            self.age += 1
            if self.frames % self.refine_every == 0 or self._retry:
                out = refine(self.camera, frame, field_info, self.rect, boxes, guide=guide, img=img)
                self._retry = out is None                       # a miss is retried on the very next frame
                if out is not None:
                    self.camera, self.info = out
                    self.H = homography_for(self.camera, self.rect)
                    self.age = 0
                    self.refined += 1
            if self.age > self.max_misses:
                self.H, self.camera, self.last, self.info = None, None, None, {}
                self._since_try = 0
        self._prev_gray = gray

        if self.H is None:
            self._since_try += 1
            if self._since_try >= self._wait:
                self._since_try = 0
                self.full_tried += 1
                reg = register(frame, field_info, boxes)
                if reg is not None:
                    self.H, self.camera, self.last = reg.H, reg.camera, reg
                    self.age = 0
                    self.full_ok += 1
                    self._wait = self.retry_every
                else:
                    self._wait = min(self._wait * 2, 64)       # a look that failed is tried again, but less and less often
        return self.H

    def to_field(self, pts_px) -> np.ndarray | None:
        if self.H is None:
            return None
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)


# ------------------------------------------------------------------ drawing

def draw_grid(frame: np.ndarray, H: np.ndarray, rect, color=(0, 255, 255)) -> np.ndarray:
    """Draw the field model back onto the frame through the mapping: yard lines every 5 yards,
    the hash rows, both sidelines. If the mapping is right they sit on the paint."""
    out = frame.copy()
    Hinv = np.linalg.inv(H)
    x1, y1, x2, y2 = rect

    def to_px(pts):
        return cv2.perspectiveTransform(np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2), Hinv).reshape(-1, 2)

    probe = np.array([[x1, y2], [x2, y2], [x1, (y1 + y2) / 2], [x2, (y1 + y2) / 2]], dtype=np.float64)
    corners = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), H).reshape(-1, 2)
    lo = max(-30, int(np.floor(np.nanmin(corners[:, 0]) / 5.0)) - 2)
    hi = min(30, int(np.ceil(np.nanmax(corners[:, 0]) / 5.0)) + 2)

    def poly(pts, col, thick):
        px = to_px(pts)
        ok = np.all(np.isfinite(px), axis=1) & (np.abs(px) < 1e5).all(axis=1)
        px = px[ok].astype(np.int32)
        if len(px) >= 2:
            cv2.polylines(out, [px.reshape(-1, 1, 2)], False, col, thick, cv2.LINE_AA)

    ys = np.linspace(0.0, FIELD_WIDTH, 30)
    for k in range(lo, hi + 1):
        poly([(5.0 * k, y) for y in ys], color, 2)
        mid = to_px([(5.0 * k, FIELD_WIDTH / 2)])[0]
        if np.all(np.isfinite(mid)) and x1 <= mid[0] < x2 and y1 <= mid[1] < y2:
            cv2.putText(out, str(5 * k), (int(mid[0]) + 6, int(mid[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    xs = np.linspace(5.0 * lo, 5.0 * hi, 60)
    for y, col in ((0.0, (0, 140, 255)), (FIELD_WIDTH, (0, 140, 255)), (HASH_NEAR, (255, 200, 0)), (HASH_FAR, (255, 200, 0))):
        poly([(x, y) for x in xs], col, 2)
    inside = np.zeros(out.shape[:2], np.uint8)
    cv2.rectangle(inside, (x1, y1), (x2, y2), 255, -1)
    return np.where(inside[..., None] > 0, out, frame)
