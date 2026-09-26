import numpy as np

from nfl_intel.teams import TeamAssigner, TeamClassifier, TeamMemory, jersey_feature, kmeans, torso_crop


def _player_frame(colors, size=(720, 1280)):
    """Green frame with a coloured jersey rectangle per player; returns (frame, boxes)."""
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = (40, 160, 60)  # grass in BGR
    boxes = []
    for i, c in enumerate(colors):
        x1, y1 = 60 + i * 90, 300
        x2, y2 = x1 + 50, y1 + 120
        frame[y1:y2, x1:x2] = c
        boxes.append((x1, y1, x2, y2))
    return frame, boxes


def test_kmeans_separates_two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal((0, 0, 0), 0.5, (40, 3))
    b = rng.normal((10, 10, 10), 0.5, (40, 3))
    centres, labels = kmeans(np.vstack([a, b]), 2)
    assert centres.shape == (2, 3)
    assert len(set(labels[:40])) == 1 and len(set(labels[40:])) == 1
    assert labels[0] != labels[40]


def test_torso_crop_clips_to_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert torso_crop(frame, (-50, -50, 10, 10)).shape[0] >= 0
    assert torso_crop(frame, (0, 0, 300, 300)).size > 0
    assert torso_crop(frame, (200, 200, 300, 300)).size == 0
    assert jersey_feature(frame[0:0, 0:0]) is None


def test_classifier_orders_dark_team_first():
    clf = TeamClassifier()
    clf.fit(np.array([[200, 0, 0], [205, 0, 0], [20, 0, 0], [25, 0, 0]], dtype=float))
    assert clf.predict(np.array([[22, 0, 0]]))[0] == 0
    assert clf.predict(np.array([[202, 0, 0]]))[0] == 1


def test_memory_sticks_after_votes():
    mem = TeamMemory()
    assert mem.current(7) is None
    for _ in range(3):
        mem.update(7, 1)
    assert mem.update(7, 0) == 1  # one bad frame doesn't flip it


def test_assigner_warmup_then_labels():
    white, red = (240, 240, 240), (30, 30, 200)
    frame, boxes = _player_frame([white, red, white, red, white, red])
    ids = list(range(6))
    assigner = TeamAssigner(warmup_frames=3, min_samples=6)
    assert assigner.assign(frame, boxes, ids) == [None] * 6
    assert assigner.assign(frame, boxes, ids) == [None] * 6
    labels = assigner.assign(frame, boxes, ids)
    assert None not in labels
    assert labels[0] == labels[2] == labels[4]
    assert labels[1] == labels[3] == labels[5]
    assert labels[0] != labels[1]
    # red jersey is darker in L than white, so it is team 0
    assert labels[1] == 0
