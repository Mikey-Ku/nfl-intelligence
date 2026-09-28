import cv2
import numpy as np
import pytest

from nfl_intel.imaging import shrink


def _dots(w, h, pts):
    img = np.zeros((h, w, 3), np.uint8)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 9, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def _centre(img):
    g = img[..., 0].astype(np.float64)
    ys, xs = np.mgrid[: g.shape[0], : g.shape[1]]
    return (xs * g).sum() / g.sum(), (ys * g).sum() / g.sum()


@pytest.mark.parametrize("src,dst", [((3024, 1898), (960, 603)), ((1624, 920), (960, 544)), ((3024, 1898), (378, 237)), ((1000, 700), (640, 448))])
def test_shrink_puts_points_where_a_plain_resize_does(src, dst):
    for pt in [(src[0] * 0.1, src[1] * 0.15), (src[0] * 0.93, src[1] * 0.9), (src[0] * 0.5, src[1] * 0.5)]:
        img = _dots(*src, [pt])
        a = _centre(cv2.resize(img, dst, interpolation=cv2.INTER_AREA))
        b = _centre(shrink(img, dst))
        assert b == pytest.approx(a, abs=0.06)                     # pixels of the small image
        assert shrink(img, dst).shape == (dst[1], dst[0], 3)


def test_shrink_keeps_brightness_and_handles_grey_and_enlarging():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (1898, 3024), dtype=np.uint8)
    assert shrink(img, (96, 54)).mean() == pytest.approx(img.mean(), abs=1.5)
    assert shrink(img[:100, :100], (200, 200)).shape == (200, 200)
