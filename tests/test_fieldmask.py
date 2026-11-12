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


def _bordered_field(squeeze=1.0):
    """A field 53.33 yd wide at 10 px a yard with a 2 yd white border, and the mapping to yards.
    `squeeze` under 1 gives a mapping that puts both sidelines too far in, as a real fit can."""
    from nfl_intel.field import FIELD_WIDTH

    frame = np.full((700, 1000, 3), STANDS, dtype=np.uint8)
    top, bottom = 60, 60 + int(round(FIELD_WIDTH * 10))
    frame[top - 20 : bottom + 20] = (245, 245, 245)
    frame[top:bottom] = GRASS
    mid = FIELD_WIDTH / 2
    H = np.array([[0.1, 0.0, 0.0], [0.0, 0.1 * squeeze, mid - squeeze * (top / 10.0 + mid)], [0.0, 0.0, 1.0]])
    return frame, H, FIELD_WIDTH


def test_the_border_is_found_where_the_mapping_puts_it():
    from nfl_intel.fieldmask import border_edges

    frame, H, width = _bordered_field()
    low, high = border_edges(frame, H, (0, 0, 1000, 700))
    assert abs(low) < 0.3 and abs(high - width) < 0.3
    frame, H, width = _bordered_field(squeeze=0.92)           # the mapping is 2 yd short at each sideline
    low, high = border_edges(frame, H, (0, 0, 1000, 700))
    assert 1.8 < low < 2.5 and 1.8 < width - high < 2.5


def test_no_border_in_the_picture_is_none():
    from nfl_intel.fieldmask import border_edges

    frame = np.full((700, 1000, 3), GRASS, dtype=np.uint8)
    H = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, -6.0], [0.0, 0.0, 1.0]])
    assert border_edges(frame, H, (0, 0, 1000, 700)) == (None, None)
