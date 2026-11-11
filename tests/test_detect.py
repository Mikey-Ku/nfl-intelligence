from nfl_intel.detect import keep_classes


def test_coco_weights_keep_person():
    assert keep_classes({0: "person", 1: "bicycle", 32: "sports ball"}) == [0]


def test_fine_tuned_weights_keep_players_and_drop_referees():
    assert keep_classes({0: "player", 1: "referee"}) == [0]


def test_other_weights_keep_every_class():
    assert keep_classes({0: "helmet", 1: "ball"}) is None
