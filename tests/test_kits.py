import numpy as np

from nfl_intel.fieldtrack import FieldTracker
from nfl_intel.kits import KitReader, kit_feature

BROWN, ORANGE, WHITE, BLUE, GRASS = (30, 50, 90), (40, 110, 240), (245, 245, 245), (230, 170, 90), (60, 140, 70)     # BGR


def _picture(players):
    """A field with a man drawn at each (x, jersey colour, pants colour): a helmet, a jersey, a pair of pants."""
    img = np.full((240, 80 * len(players), 3), GRASS, np.uint8)
    boxes = []
    for k, (jersey, pants) in enumerate(players):
        x = 80 * k + 20
        img[40:70, x + 8 : x + 32] = pants          # helmets here are the pants' colour
        img[70:130, x : x + 40] = jersey
        img[130:200, x + 5 : x + 35] = pants
        img[85:115, x + 12 : x + 28] = WHITE        # a big white number on every jersey, dark or light
        boxes.append((x, 40, x + 40, 200))
    return img, boxes


def test_the_reader_learns_two_kits_from_the_line_up_and_reads_them_afterwards():
    rng = np.random.default_rng(0)
    feats, sides = [], []
    for _ in range(60):
        img, boxes = _picture([(BROWN, ORANGE), (WHITE, BLUE)])
        img = np.clip(img.astype(int) + rng.integers(-12, 13, img.shape), 0, 255).astype(np.uint8)
        for side, box in enumerate(boxes):
            feats.append(kit_feature(img, box)); sides.append(side)
    reader = KitReader()
    assert not reader.fitted and reader.read([feats[0]]) == [None]            # before the lesson it says nothing
    assert not reader.fit(feats[:10], [0] * 10)                               # one side alone teaches nothing
    assert reader.fit(feats, sides) and reader.fitted
    img, boxes = _picture([(WHITE, BLUE), (BROWN, ORANGE), (WHITE, BLUE)])
    assert reader.read([kit_feature(img, b) for b in boxes]) == [1, 0, 1]
    half = _picture([(BROWN, BLUE)])                                          # one man's jersey over another's legs: no reading
    assert reader.read([kit_feature(half[0], half[1][0]), None]) == [None, None]
    assert kit_feature(img, (5, 5, 7, 9)) is None                             # too small to say


def _settle(tracker, xy, tracks, frames=8):
    for k in range(frames):
        slots = tracker.update(0.04 * k, np.array(xy, float), [None] * len(xy), tracks)
    return 0.04 * frames, [s.id for s in slots]


def test_a_player_without_a_box_does_not_push_his_neighbour_off_his_own():
    tracker = FieldTracker()
    t, (a, b) = _settle(tracker, [(0.0, 0.0), (1.0, 0.0)], [1, 2])
    # a's box is gone; b's unbroken track is where it was; a stray box turns up two yards past b, out of a's reach
    slots = tracker.update(t, np.array([(1.0, 0.0), (3.2, 0.0)]), [None, None], [2, 9])
    assert slots[0].id == b and slots[1].id not in (a, b)                     # b keeps his box and the stray is somebody new
    old = FieldTracker(unmatched=None)                                        # with no price on going without a box...
    t, (a, b) = _settle(old, [(0.0, 0.0), (1.0, 0.0)], [1, 2])
    slots = old.update(t, np.array([(1.0, 0.0), (3.2, 0.0)]), [None, None], [2, 9])
    assert slots[0].id == a and slots[1].id == b                              # ...a takes b's box and b is pushed onto the stray


def test_a_box_that_slides_to_the_other_team_goes_to_the_man_it_shows():
    tracker = FieldTracker()
    t, (blocker, rusher) = _settle(tracker, [(0.0, 0.0), (1.0, 0.0)], [1, 2])
    by_id = {s.id: s for s in tracker.slots}
    by_id[blocker].kit, by_id[rusher].kit = 0, 1                              # they lined up on opposite sides
    # the two lock up: one box is left, on the blocker's track, and it shows the rusher's kit
    slots = tracker.update(t, np.array([(0.8, 0.0)]), [None], [1], kits=[1])
    assert slots[0].id == rusher
    slots = tracker.update(t + 0.04, np.array([(0.8, 0.0)]), [None], [1], kits=[1])
    assert slots[0].id == rusher and any(s.id == blocker for s in tracker.slots)       # the blocker waits; he is not gone
    slots = tracker.update(t + 0.08, np.array([(0.1, 0.0), (0.8, 0.0)]), [None, None], [7, 1], kits=[0, 1])
    assert [s.id for s in slots] == [blocker, rusher]                         # and takes his own box when it shows again


def test_a_player_keeps_a_box_in_the_wrong_kit_when_nobody_else_wants_it():
    tracker = FieldTracker()
    t, (a,) = _settle(tracker, [(0.0, 0.0)], [1])
    tracker.slots[0].kit = 0
    slots = tracker.update(t, np.array([(0.1, 0.0)]), [None], [1], kits=[1])  # a misread, or an arm across him: still his box
    assert slots[0].id == a
    slots = tracker.update(t + 0.04, np.array([(2.0, 0.0)]), [None], [5], kits=[1])    # but not a wrong-kit box two yards off
    assert slots[0].id != a


def test_a_stranger_learns_his_kit_from_his_own_boxes():
    tracker = FieldTracker()
    for k in range(8):
        tracker.update(0.04 * k, np.array([(0.0, 0.0)]), [None], [1], kits=[1])
    assert tracker.slots[0].kit is None and tracker.slots[0].wears == 1      # nobody told us; his boxes did
