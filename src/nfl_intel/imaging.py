"""Small image helpers shared by the per-frame path."""

from __future__ import annotations

import cv2
import numpy as np


def shrink(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to `size` (width, height), fast, keeping the geometry of cv2.resize.

    OpenCV's area resize is the right filter for shrinking but takes 10-20 ms on a 4K grab
    when the ratio isn't a whole number. Here whole-number blocks are averaged first (the fast
    path) and the rest of the way is a linear resample, placed so that a point lands exactly
    where a one-step resize would put it. Enlarging falls through to a plain linear resize.
    """
    w, h = int(size[0]), int(size[1])
    sh, sw = img.shape[:2]
    k = int(min(sw // max(1, w), sh // max(1, h)))
    if k < 2:
        interp = cv2.INTER_LINEAR if (sw >= w and sh >= h) or (sw < w) else cv2.INTER_AREA
        return cv2.resize(img, (w, h), interpolation=interp)
    whole = img[: sh - sh % k, : sw - sw % k]
    blocks = cv2.resize(whole, (whole.shape[1] // k, whole.shape[0] // k), interpolation=cv2.INTER_AREA)
    a, b = w * k / sw, h * k / sh
    M = np.array([[a, 0.0, 0.5 * a - 0.5], [0.0, b, 0.5 * b - 0.5]])
    return cv2.warpAffine(blocks, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
