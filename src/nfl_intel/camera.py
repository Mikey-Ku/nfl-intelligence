"""A physical camera model for field registration.

A free homography has 8 degrees of freedom and needs evidence across the whole field to pin
down. An All-22 camera is a real camera: level (no roll), square pixels, looking at a flat
field from a fixed spot, with pan, tilt and zoom. That is 6 numbers, so yard lines plus a
single row of hash marks are enough, and during a shot only pan, tilt and zoom change.

Coordinates: the field is the plane Z = 0 with X along its length and Y across it, in yards.
Pixels are measured from the image centre, u right and v down.

Constraints use the same format everywhere: rows of (u, v, value, weight), where `value` is
the field X (for `cons_x`) or the field Y (for `cons_y`) that pixel must map to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Camera:
    f: float                 # focal length in pixels
    pan: float               # radians, direction the camera faces in the field plane (0 = along +X)
    tilt: float              # radians below the horizon
    C: np.ndarray            # camera position (X, Y, Z) in yards
    cx: float
    cy: float

    def rotation(self) -> np.ndarray:
        return rotation(self.pan, self.tilt)

    def field_to_pixel(self) -> np.ndarray:
        """3x3 homography taking field (X, Y, 1) to pixels."""
        R = self.rotation()
        K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])
        H = K @ np.column_stack([R[:, 0], R[:, 1], -R @ self.C])
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    def pixel_to_field(self) -> np.ndarray:
        H = np.linalg.inv(self.field_to_pixel())
        return H / H[2, 2]

    def params(self) -> np.ndarray:
        return np.array([np.log(self.f), self.pan, self.tilt, *self.C])

    @classmethod
    def from_params(cls, p, cx: float, cy: float) -> Camera:
        return cls(float(np.exp(p[0])), float(p[1]), float(p[2]), np.array(p[3:6], dtype=np.float64), cx, cy)


def realistic(cam: Camera, w: int) -> bool:
    """Stadium cameras sit 10-90 yards up, look down at 7-70 degrees, and zoom within what a
    broadcast lens can do. A fit outside that is the maths explaining noise, not a camera."""
    return bool(
        np.isfinite(cam.f) and 0.5 * w <= cam.f <= 13.0 * w
        and np.radians(7) <= cam.tilt <= np.radians(70)
        and 8.0 <= cam.C[2] <= 90.0
        and abs(cam.C[0]) < 300 and abs(cam.C[1]) < 300
    )


def looks_across(cam: Camera, min_sin: float = 0.5) -> bool:
    """True for a sideline camera (looking across the field), False for one looking down its
    length from behind an end zone. The second kind sees the field through a long lens at a
    shallow angle, where the scale across the field is poorly pinned down."""
    return bool(abs(np.sin(cam.pan)) >= min_sin)


def rotation(pan: float, tilt: float) -> np.ndarray:
    """World -> camera rotation for a level camera. Rows are the camera's right, down and
    forward directions in field coordinates."""
    cp, sp, ct, st = np.cos(pan), np.sin(pan), np.cos(tilt), np.sin(tilt)
    fwd = np.array([cp * ct, sp * ct, -st])
    right = np.array([sp, -cp, 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _rays(cons: np.ndarray, f: float, R: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """World directions of the rays through the constraint pixels."""
    d_cam = np.column_stack([(cons[:, 0] - cx) / f, (cons[:, 1] - cy) / f, np.ones(len(cons))])
    return d_cam @ R                       # R^T applied to each row


def _solve_position(cons_x, cons_y, f, R, cx, cy) -> tuple[np.ndarray, float] | None:
    """With the camera's orientation and zoom fixed, its position follows from a linear
    least-squares problem: a ray from C hits the ground at X = Cx - Cz dx/dz, Y = Cy - Cz dy/dz."""
    dx = _rays(cons_x, f, R, cx, cy)
    dy = _rays(cons_y, f, R, cx, cy)
    if np.any(dx[:, 2] > -1e-6) or np.any(dy[:, 2] > -1e-6):
        return None                        # some pixel looks at or above the horizon
    wx, wy = cons_x[:, 3], cons_y[:, 3]
    A = np.vstack([
        np.column_stack([np.ones(len(dx)), np.zeros(len(dx)), -dx[:, 0] / dx[:, 2]]) * wx[:, None],
        np.column_stack([np.zeros(len(dy)), np.ones(len(dy)), -dy[:, 1] / dy[:, 2]]) * wy[:, None],
    ])
    b = np.concatenate([cons_x[:, 2] * wx, cons_y[:, 2] * wy])
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    if C[2] <= 1.0:
        return None                        # the camera is above the field, not in it
    err = np.abs(A @ C - b)
    return C, float(err.sum() / (wx.sum() + wy.sum()))


def residuals(p, cons_x, cons_y, cx, cy) -> np.ndarray:
    cam = Camera.from_params(p, cx, cy)
    R = cam.rotation()
    out = []
    for cons, axis in ((cons_x, 0), (cons_y, 1)):
        d = _rays(cons, cam.f, R, cx, cy)
        dz = np.minimum(d[:, 2], -1e-6)
        ground = cam.C[axis] - cam.C[2] * d[:, axis] / dz
        out.append((ground - cons[:, 2]) * cons[:, 3])
    return np.concatenate(out)


def error(cam: Camera, cons_x, cons_y) -> float:
    """Weighted mean error in yards."""
    r = np.abs(residuals(cam.params(), cons_x, cons_y, cam.cx, cam.cy))
    return float(r.sum() / (cons_x[:, 3].sum() + cons_y[:, 3].sum()))


def fit_camera(cons_x: np.ndarray, cons_y: np.ndarray, w: int, h: int, start: Camera | None = None,
               fix_position: bool = False) -> tuple[Camera, float] | None:
    """Find the camera that best explains the constraints.

    With `start`, refine from there (and with `fix_position`, move only pan, tilt and zoom:
    the camera is bolted down). Without it, search zoom, tilt and pan on a grid, solve the
    position in closed form at each, and refine the best few.
    """
    cons_x, cons_y = np.asarray(cons_x, dtype=np.float64), np.asarray(cons_y, dtype=np.float64)
    if len(cons_x) < 8 or len(cons_y) < 3:
        return None
    cx, cy = w / 2.0, h / 2.0

    def refine(cam: Camera, free_position: bool) -> tuple[Camera, float] | None:
        p0 = cam.params()
        if free_position:
            fun = lambda p: residuals(p, cons_x, cons_y, cx, cy)      # noqa: E731
            x0 = p0
        else:
            fun = lambda p: residuals(np.concatenate([p, p0[3:]]), cons_x, cons_y, cx, cy)   # noqa: E731
            x0 = p0[:3]
        try:
            sol = least_squares(fun, x0, loss="soft_l1", f_scale=0.5, max_nfev=60)
        except Exception:
            return None
        p = sol.x if free_position else np.concatenate([sol.x, p0[3:]])
        out = Camera.from_params(p, cx, cy)
        if not realistic(out, w):
            return None
        return out, error(out, cons_x, cons_y)

    if start is not None:
        return refine(start, free_position=not fix_position)

    candidates = grid_search(cons_x, cons_y, w, cx, cy)
    best = None
    for f, pan, tilt, C in candidates[:4]:
        out = refine(Camera(float(f), float(pan), float(tilt), C, cx, cy), free_position=True)
        if out is not None and (best is None or out[1] < best[1]):
            best = out
        if best is not None and best[1] < 0.06:
            break                                             # as good as the paint allows
    return best


def grid_search(cons_x: np.ndarray, cons_y: np.ndarray, w: int, cx: float, cy: float, keep: int = 6):
    """Every combination of zoom, tilt and pan on a grid, all at once.

    With orientation and zoom fixed, the best camera position has a closed form: along each
    axis it is the weighted mean of what the constraints ask for, and the height is a ratio
    of two sums. So the whole grid is a few array operations instead of thousands of solves.
    Returns the best `keep` as (f, pan, tilt, C), best first.
    """
    # the grid only has to land near the answer, so a thinned set of constraints will do
    if len(cons_x) > 80:
        cons_x = cons_x[np.linspace(0, len(cons_x) - 1, 80).astype(int)]
    if len(cons_y) > 50:
        cons_y = cons_y[np.linspace(0, len(cons_y) - 1, 50).astype(int)]
    pans = np.radians(np.arange(0.0, 360.0, 6.0))
    tilts = np.radians(np.arange(8.0, 68.0, 4.0))
    P, T = np.meshgrid(pans, tilts, indexing="ij")
    P, T = P.ravel(), T.ravel()
    cp, sp, ct, st = np.cos(P), np.sin(P), np.cos(T), np.sin(T)
    fwd = np.stack([cp * ct, sp * ct, -st], axis=1)
    right = np.stack([sp, -cp, np.zeros_like(cp)], axis=1)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=1)                 # (G, 3, 3), rows right/down/forward
    wx, wy = cons_x[:, 3], cons_y[:, 3]
    X, Y = cons_x[:, 2], cons_y[:, 2]
    found = []
    for f in np.geomspace(0.7 * w, 12.0 * w, 13):
        dx = np.column_stack([(cons_x[:, 0] - cx) / f, (cons_x[:, 1] - cy) / f, np.ones(len(cons_x))])
        dy = np.column_stack([(cons_y[:, 0] - cx) / f, (cons_y[:, 1] - cy) / f, np.ones(len(cons_y))])
        rx = np.einsum("nc,gcw->gnw", dx, R)                  # world rays, (G, N, 3)
        ry = np.einsum("nc,gcw->gnw", dy, R)
        ok = (rx[..., 2] < -1e-6).all(axis=1) & (ry[..., 2] < -1e-6).all(axis=1)
        if not ok.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            a = rx[..., 0] / rx[..., 2]                       # X = Cx - Cz a
            b = ry[..., 1] / ry[..., 2]                       # Y = Cy - Cz b
        a_bar = (a * wx).sum(axis=1) / wx.sum()
        b_bar = (b * wy).sum(axis=1) / wy.sum()
        x_bar = (X * wx).sum() / wx.sum()
        y_bar = (Y * wy).sum() / wy.sum()
        da, db = a - a_bar[:, None], b - b_bar[:, None]
        num = (wx * (X - x_bar) * da).sum(axis=1) + (wy * (Y - y_bar) * db).sum(axis=1)
        den = (wx * da ** 2).sum(axis=1) + (wy * db ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Cz = -num / den
        Cx = x_bar + Cz * a_bar
        Cy = y_bar + Cz * b_bar
        ex = np.abs(Cx[:, None] - Cz[:, None] * a - X) * wx
        ey = np.abs(Cy[:, None] - Cz[:, None] * b - Y) * wy
        err = (ex.sum(axis=1) + ey.sum(axis=1)) / (wx.sum() + wy.sum())
        ok &= np.isfinite(err) & (Cz >= 8.0) & (Cz <= 90.0)
        for g in np.nonzero(ok)[0]:
            found.append((float(err[g]), float(f), float(P[g]), float(T[g]), np.array([Cx[g], Cy[g], Cz[g]])))
    found.sort(key=lambda c: c[0])
    return [(f, pan, tilt, C) for _, f, pan, tilt, C in found[:keep]]
