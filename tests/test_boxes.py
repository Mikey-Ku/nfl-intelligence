"""The box editor's parts: YOLO files in and out, picking a box, corners, the zoomed view."""

import numpy as np

from nfl_intel.boxes import Box, View, corner, frames_in, hit, label_path, read_yolo, resized, write_yolo


def test_yolo_round_trip(tmp_path):
    boxes = [Box(0, 100, 200, 160, 320), Box(1, 10.5, 20.5, 30.5, 60.5)]
    p = tmp_path / "f.txt"
    write_yolo(p, boxes, 1000, 500)
    lines = p.read_text().splitlines()
    assert lines[0] == "0 0.130000 0.520000 0.060000 0.240000"
    back = read_yolo(p, 1000, 500)
    assert [b.cls for b in back] == [0, 1]
    assert np.allclose([back[0].x1, back[0].y1, back[0].x2, back[0].y2], [100, 200, 160, 320])
    assert read_yolo(tmp_path / "missing.txt", 10, 10) == []
    write_yolo(p, [], 10, 10)
    assert p.read_text() == ""


def test_hit_picks_the_smallest_box_under_the_point():
    big, small = Box(0, 0, 0, 100, 100), Box(0, 40, 40, 60, 60)
    assert hit([big, small], 50, 50) == 1
    assert hit([big, small], 10, 10) == 0
    assert hit([big, small], 200, 200) is None
    assert Box(0, 60, 60, 40, 40).contains(50, 50)                # drawn backwards still counts


def test_corner_and_resize():
    b = Box(0, 10, 10, 50, 90)
    assert corner(b, 12, 11, reach=5) == "tl" and corner(b, 49, 88, reach=5) == "br"
    assert corner(b, 30, 50, reach=5) is None
    r = resized(b, "tl", 0, 0)
    assert (r.x1, r.y1, r.x2, r.y2) == (0, 0, 50, 90)
    r = resized(b, "br", 70, 100)
    assert (r.x1, r.y1, r.x2, r.y2) == (10, 10, 70, 100)


def test_view_maps_screen_to_frame_and_zooms_around_a_point():
    v = View(w=2000, h=1000, scale=0.5)
    assert v.to_frame(100, 50) == (200, 100) and v.to_screen(200, 100) == (100, 50)
    v.zoom_at(1500, 500, 2.0)                                    # a 1000x500 window on the frame, centred there
    assert (v.ox, v.oy) == (1000, 250)
    assert v.to_frame(0, 0) == (1000, 250) and v.to_screen(1500, 500) == (500, 250)
    v.zoom_at(1990, 990, 2.0)                                    # clipped to the frame's edge
    assert (v.ox, v.oy) == (1000, 500)
    img = v.render(np.zeros((1000, 2000, 3), dtype=np.uint8))
    assert img.shape == (500, 1000, 3)


def test_frames_and_label_paths(tmp_path):
    (tmp_path / "images").mkdir()
    for name in ("b.jpg", "a.jpg", "c.png"):
        (tmp_path / "images" / name).write_bytes(b"")
    assert [p.name for p in frames_in(tmp_path)] == ["a.jpg", "b.jpg", "c.png"]
    assert label_path(tmp_path, tmp_path / "images" / "a.jpg") == tmp_path / "labels" / "a.txt"
