import numpy as np

from nfl_intel.fieldmask import expand_bbox, find_field, grass_mask

GRASS = (40, 160, 60)      # BGR turf
STANDS = (90, 90, 90)


def _stadium(size=(720, 1280)):
    """Grey frame with a green field in the middle, like a browser window around a player."""
    frame = np.full((*size, 3), STANDS, dtype=np.uint8)
    frame[200:650, 300:1100] = GRASS
    return frame


def test_grass_mask_hits_turf_only():
    frame = _stadium()
    m = grass_mask(frame)
    assert m[400, 700] and not m[50, 50]


def test_find_field_bbox_and_membership():
    frame = _stadium()
    field = find_field(frame)
    assert field is not None
    x1, y1, x2, y2 = field.bbox
    # bbox wraps the green area, allowing for the dilation margin
    assert 250 <= x1 <= 310 and 1090 <= x2 <= 1150 and 150 <= y1 <= 205 and 645 <= y2 <= 700
    assert field.contains_foot(700, 640)          # on the grass
    assert field.foot_on_field(700, 640)
    assert not field.contains_foot(700, 100)      # in the stands
    assert not field.foot_on_field(700, 100)
    assert not field.contains_foot(-5, 400)       # off frame
    assert 0.3 < field.fraction < 0.5
    # grass colour is the turf, in OpenCV Lab
    assert field.grass_lab[1] < 128 < field.grass_lab[2]


def test_players_standing_on_the_field_are_inside_it():
    frame = _stadium()
    frame[400:520, 600:660] = (240, 240, 240)   # a player covers the grass under their own feet
    field = find_field(frame)
    assert field.contains_foot(630, 519)


def test_find_field_none_without_grass():
    frame = np.full((360, 640, 3), STANDS, dtype=np.uint8)
    assert find_field(frame) is None


def test_expand_bbox_clamps():
    assert expand_bbox((0, 0, 100, 100), (120, 120), margin=0.5) == (0, 0, 120, 120)
    assert expand_bbox((100, 100, 200, 200), (400, 400), margin=0.1) == (90, 90, 210, 210)
