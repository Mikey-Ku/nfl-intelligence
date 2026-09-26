import numpy as np
import pytest

from nfl_intel.field import (
    FIELD_WIDTH,
    HASH_INSET,
    FieldCalibration,
    column_x,
    draw_field,
    field_to_image,
    landmark_xy,
)


def test_column_positions():
    assert column_x("LEZ") == 0
    assert column_x("L0") == 10          # left goal line
    assert column_x("L40") == 50
    assert column_x("50") == 60
    assert column_x("R40") == 70
    assert column_x("R0") == 110
    assert column_x("REZ") == 120


def test_landmark_rows():
    assert landmark_xy("50-S1") == (60, 0)
    x, y = landmark_xy("l20-h2")
    assert x == 30 and y == pytest.approx(FIELD_WIDTH - HASH_INSET)
    with pytest.raises(ValueError):
        landmark_xy("L60-S1")
    with pytest.raises(ValueError):
        landmark_xy("L40-X1")


def test_homography_round_trip():
    # a synthetic camera: scale + shear + perspective
    yd = [(20, 0), (20, FIELD_WIDTH), (80, 0), (80, FIELD_WIDTH), (50, HASH_INSET)]
    px = [(100, 700), (420, 120), (1500, 700), (1200, 120), (820, 380)]
    cal = FieldCalibration.from_correspondences(px[:4], yd[:4])
    back = cal.to_field(px[:4])
    assert np.allclose(back, yd[:4], atol=1e-3)
    assert np.allclose(cal.to_pixels(yd[:4]), px[:4], atol=1e-2)
    assert cal.reprojection_error(px[:4], yd[:4]) < 1e-3


def test_from_landmarks_and_save_load(tmp_path):
    names = ["L10-S1", "L10-S2", "R10-S1", "R10-S2"]
    px = [(0, 500), (200, 0), (1000, 500), (800, 0)]
    cal = FieldCalibration.from_landmarks(px, names, {"angle": "sideline"})
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = FieldCalibration.load(path)
    assert loaded.meta["angle"] == "sideline"
    assert np.allclose(loaded.to_field([(500, 250)]), cal.to_field([(500, 250)]))


def test_too_few_points():
    with pytest.raises(ValueError):
        FieldCalibration.from_correspondences([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def test_draw_field_and_mapping():
    img = draw_field(scale=4.0, pad=2.0)
    assert img.shape == (int((FIELD_WIDTH + 4) * 4), 124 * 4, 3)
    assert np.allclose(field_to_image([(0, 0)], 4.0, 2.0), [[8, 8]])
