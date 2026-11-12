"""Field registration on a rendered field, where the true camera is known."""

import cv2
import numpy as np
import pytest

from nfl_intel.camera import Camera
from nfl_intel.fieldmask import find_field
from nfl_intel.registration import (
    CANON_WIDTH,
    FieldRegistrar,
    Line,
    angle_diff,
    family_angle,
    merge_family,
    numberings,
    register,
    video_rect,
)

FIELD_W = 160 / 3
HASH = 70.75 / 3
GRASS = (52, 128, 58)            # BGR
SURROUND = (14, 14, 16)
VIDEO = (120, 90, 1400, 810)     # x1, y1, x2, y2 of the video inside the captured window


def render(cam: Camera, size=(900, 1520)) -> np.ndarray:
    """A window with a dark surround and a video of a football field seen by `cam`
    (the camera is in canonical video pixels: CANON_WIDTH wide)."""
    x1, y1, x2, y2 = VIDEO
    vw, vh = x2 - x1, y2 - y1
    s = CANON_WIDTH / vw
    u, v = np.meshgrid(np.arange(vw) * s, np.arange(vh) * s)
    Hm = cam.pixel_to_field()
    q = np.stack([u, v, np.ones_like(u)], axis=-1) @ Hm.T
    with np.errstate(divide="ignore", invalid="ignore"):
        X, Y = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    ground = q[..., 2] * Hm[2, 2] > 0                     # below the horizon
    img = np.full((vh, vw, 3), (70, 70, 70), dtype=np.uint8)
    on_field = ground & (X > -2) & (X < 122) & (Y > -2) & (Y < FIELD_W + 2)
    img[on_field] = GRASS
    rng = np.random.default_rng(0)
    img[on_field] = np.clip(img[on_field] + rng.normal(0, 4, (int(on_field.sum()), 3)), 0, 255).astype(np.uint8)
    inside = ground & (Y >= 0) & (Y <= FIELD_W) & (X >= 10) & (X <= 110)
    white = np.zeros_like(ground)
    white |= inside & (np.abs(X - 5 * np.round(X / 5)) < 0.09)                       # yard lines
    whole = np.abs(X - np.round(X)) < 0.07
    white |= inside & whole & (Y > HASH - 2 / 3) & (Y < HASH)                        # hash marks: 2 ft long, painted
    white |= inside & whole & (Y > FIELD_W - HASH) & (Y < FIELD_W - HASH + 2 / 3)    # outward from the inbounds line
    white |= ground & (X > 8) & (X < 112) & (((Y > -2) & (Y < 0)) | ((Y > FIELD_W) & (Y < FIELD_W + 2)))   # border
    img[white] = (235, 235, 235)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    frame = np.full((*size, 3), SURROUND, dtype=np.uint8)
    frame[y1:y2, x1:x2] = img
    return frame


def to_truth(cam: Camera, H_est: np.ndarray):
    """Median distance in yards between the estimate and the truth over the field in view,
    allowing the estimate its arbitrary 5-yard shift and mirror."""
    x1, y1, x2, y2 = VIDEO
    gu, gv = np.meshgrid(np.linspace(x1 + 150, x2 - 150, 7), np.linspace(y1 + 330, y2 - 60, 4))
    px = np.column_stack([gu.ravel(), gv.ravel()])
    s = CANON_WIDTH / (x2 - x1)
    canon = (px - np.array([x1, y1])) * s
    true = cv2.perspectiveTransform(canon.reshape(-1, 1, 2), cam.pixel_to_field()).reshape(-1, 2)
    est = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H_est).reshape(-1, 2)
    best = np.inf
    for sx in (1, -1):
        for flip in (False, True):
            x = sx * est[:, 0]
            y = FIELD_W - est[:, 1] if flip else est[:, 1]
            shift = 5 * np.round(np.median(true[:, 0] - x) / 5)
            best = min(best, float(np.median(np.hypot(true[:, 0] - (x + shift), true[:, 1] - y))))
    return best


SIDELINE_CAM = Camera(2300.0, np.radians(100), np.radians(27), np.array([52.0, -42.0, 36.0]), CANON_WIDTH / 2, 360.0)


def test_helpers():
    a, b = Line(0.0, 1.0, -100.0, 50), Line(0.0, 1.0, -104.0, 80)
    far = Line(0.0, 1.0, -300.0, 60)
    assert angle_diff(1.0, 179.0) == pytest.approx(2.0)
    flat = [Line(np.sin(np.radians(t)), -np.cos(np.radians(t)), 0.0, 1.0) for t in (1.0, 179.0, 0.5, 179.5)]
    assert min(family_angle(flat), 180 - family_angle(flat)) < 1.0          # not 90: angles wrap at 180
    merged = merge_family([a, b, far], 1280, 720)
    assert len(merged) == 2 and merged[0].support == 80
    ends = Line(0.6, 0.8, -500.0).clip(1280, 720)
    assert ends is not None and abs(Line(0.6, 0.8, -500.0).distance(ends)).max() < 1e-6


def test_numbering_skips_a_hidden_line():
    # equally spaced lines seen in perspective; the one at k = 1 is hidden
    t = lambda k: (90.0 * k - 300.0) / (0.04 * k + 1.0)                       # noqa: E731
    fam = [Line(0.0, 1.0, -(360.0 + t(k)), 100.0) for k in (0, 2, 3, 4)]
    best = numberings(fam, 1280, 720)[0]
    assert [ln.index for ln in best] == [0, 2, 3, 4]


def test_video_rect_finds_the_video():
    frame = render(SIDELINE_CAM)
    field = find_field(frame)
    assert field is not None
    x1, y1, x2, y2 = video_rect(frame, field.band, field.grass)
    assert abs(x1 - VIDEO[0]) <= 12 and abs(x2 - VIDEO[2]) <= 12
    assert abs(y1 - VIDEO[1]) <= 8 and abs(y2 - VIDEO[3]) <= 8


def test_first_lock_on_a_rendered_field():
    frame = render(SIDELINE_CAM)
    field = find_field(frame)
    reg = register(frame, field)
    assert reg is not None
    assert reg.residual < 0.3
    assert len(reg.yard_lines) >= 4 and 3 <= reg.ticks_between <= 5
    assert to_truth(SIDELINE_CAM, reg.H) < 1.0
    assert 20 < reg.camera.C[2] < 60                                          # height, yards


def test_following_a_pan_and_zoom():
    cam = SIDELINE_CAM
    regr = FieldRegistrar(retry_every=1, refine_every=1)
    frame = render(cam)
    field = find_field(frame)
    assert regr.update(frame, field) is not None
    for step in range(1, 6):
        moved = Camera(cam.f * (1 + 0.012 * step), cam.pan + np.radians(0.35 * step), cam.tilt + np.radians(0.1 * step),
                       cam.C, cam.cx, cam.cy)
        H = regr.update(render(moved), field)
        assert H is not None
    assert regr.age == 0 and regr.refined >= 4                                # the paint confirmed it
    assert to_truth(moved, H) < 1.0
    regr.reset()
    assert regr.H is None and not regr.locked


def test_the_sidelines_pin_the_camera():
    """The first lock knows the yard lines and the hash rows. Following it, the border's inner
    edge is the sideline: the camera's position is solved once more with it, and the border
    then sits on the field's edge in the mapping."""
    from nfl_intel.fieldmask import border_edges

    regr = FieldRegistrar(retry_every=1, refine_every=1)
    frame = render(SIDELINE_CAM)
    field = find_field(frame)
    assert regr.update(frame, field) is not None
    moved = 0
    for _ in range(4):
        H = regr.update(frame, field)
        moved += int(bool(regr.info.get("moved")))
    assert moved == 1 and regr.info["sidelines"] >= 1                         # solved again once, then held
    edges = border_edges(frame, H, regr.rect)
    assert any(e is not None for e in edges)
    for edge, true in zip(edges, (0.0, FIELD_W)):
        assert edge is None or abs(edge - true) < 0.35
    assert to_truth(SIDELINE_CAM, H) < 1.0


def test_light_green_lines_under_floodlights_are_paint():
    """A thin line on vivid turf comes out light green, as colourful as the grass. It is
    still a line; gold lettering of the same brightness is not."""
    import cv2

    from nfl_intel.registration import paint_mask

    grass_lab = np.array([138, 101, 171], dtype=np.uint8)
    line_lab = np.array([185, 106, 160], dtype=np.uint8)           # the grass, lighter: on the way to grey
    gold_lab = np.array([185, 128, 190], dtype=np.uint8)           # as bright, another colour
    lab = np.tile(grass_lab, (300, 600, 1))
    for x in (100, 250, 400):
        lab[:, x : x + 4] = line_lab
    lab[120:180, 480:560] = gold_lab
    lab[140:160, 500:504] = line_lab                               # a stroke inside the lettering
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    assert not (paint_mask(img, None, grass_lab.astype(float)) > 0)[:, 95:410].any()      # white lines only: nothing
    mask = paint_mask(img, None, grass_lab.astype(float), soft=True) > 0
    for x in (100, 250, 400):
        assert mask[:, x : x + 4].mean() > 0.8
    assert not mask[100:200, 460:580].any()                        # the lettering and what is near it
    assert mask[:, :90].mean() < 0.01
