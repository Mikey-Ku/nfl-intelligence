"""The camera model recovers a known camera from yard lines and one row of hash marks."""

import numpy as np
import pytest

from nfl_intel.camera import Camera, error, fit_camera, rotation

W, H = 1280, 720
HASH_NEAR = 70.75 / 3


def _project(cam: Camera, pts):
    Hm = cam.field_to_pixel()
    q = (Hm @ np.column_stack([pts, np.ones(len(pts))]).T).T
    return q[:, :2] / q[:, 2:3]


def _constraints(cam: Camera, lines=(0, 1, 2, 3, 4), x0=30.0, row_y=HASH_NEAR, noise=0.0, seed=0):
    """Pixels on a few yard lines (x = x0 + 5k) and on one hash row, as the detector would give."""
    rng = np.random.default_rng(seed)
    cx, cy = [], []
    for k in lines:
        pts = np.column_stack([np.full(30, x0 + 5.0 * k), np.linspace(2.0, 51.0, 30)])
        px = _project(cam, pts) + rng.normal(0, noise, (30, 2))
        ok = (px[:, 0] > 0) & (px[:, 0] < W) & (px[:, 1] > 0) & (px[:, 1] < H)
        cx.extend((p[0], p[1], 5.0 * k, 1.0) for p in px[ok])
    ticks = np.column_stack([np.arange(x0 - 5, x0 + 26, 1.0), np.full(31, row_y)])
    px = _project(cam, ticks) + rng.normal(0, noise, (31, 2))
    ok = (px[:, 0] > 0) & (px[:, 0] < W) & (px[:, 1] > 0) & (px[:, 1] < H)
    cy.extend((p[0], p[1], row_y, 1.0) for p in px[ok])
    return np.array(cx), np.array(cy)


def _field_error(true: Camera, est: Camera, x0: float) -> float:
    """Mean distance in yards between where the two cameras put a grid of pixels."""
    gu, gv = np.meshgrid(np.linspace(100, W - 100, 8), np.linspace(250, H - 40, 5))
    px = np.column_stack([gu.ravel(), gv.ravel(), np.ones(gu.size)])
    a = (true.pixel_to_field() @ px.T).T
    b = (est.pixel_to_field() @ px.T).T
    a, b = a[:, :2] / a[:, 2:3], b[:, :2] / b[:, 2:3]
    a[:, 0] -= x0                           # the estimate measures x from its own line 0
    return float(np.linalg.norm(a - b, axis=1).mean())


def test_rotation_is_orthonormal_and_level():
    R = rotation(0.7, 0.4)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert R[0, 2] == 0.0                   # the camera's right vector stays horizontal: no roll


@pytest.mark.parametrize("name,cam", [
    ("high sideline", Camera(2600.0, np.radians(118), np.radians(24), np.array([62.0, -38.0, 32.0]), W / 2, H / 2)),
    ("behind the offense", Camera(3800.0, np.radians(4), np.radians(14), np.array([-95.0, 26.0, 30.0]), W / 2, H / 2)),
])
def test_recovers_camera_from_yard_lines_and_one_hash_row(name, cam):
    cons_x, cons_y = _constraints(cam, noise=0.4)
    assert len(cons_x) > 60 and len(cons_y) >= 8, name
    fit = fit_camera(cons_x, cons_y, W, H)
    assert fit is not None, name
    est, err = fit
    assert err < 0.2, (name, err)
    assert _field_error(cam, est, 30.0) < 0.75, name
    assert est.C[2] > 5 and 0.05 < est.tilt < 1.2


def test_following_a_pan_with_the_position_fixed():
    cam = Camera(2600.0, np.radians(118), np.radians(24), np.array([62.0, -38.0, 32.0]), W / 2, H / 2)
    start, _ = fit_camera(*_constraints(cam), W, H)
    moved = Camera(2900.0, cam.pan + np.radians(1.5), cam.tilt + np.radians(0.6), cam.C, cam.cx, cam.cy)
    cons_x, cons_y = _constraints(moved, noise=0.4, seed=1)
    fit = fit_camera(cons_x, cons_y, W, H, start=start, fix_position=True)
    assert fit is not None
    est, err = fit
    assert err < 0.2
    assert np.allclose(est.C, start.C)                      # the camera didn't move, only turned and zoomed
    assert _field_error(moved, est, 30.0) < 0.75
    assert error(start, cons_x, cons_y) > err               # and the refit beats not refitting


def test_too_few_constraints():
    assert fit_camera(np.zeros((3, 4)), np.zeros((2, 4)), W, H) is None
