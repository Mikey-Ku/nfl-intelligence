import numpy as np
import supervision as sv

from nfl_intel.train import CLASSES, prelabel, write_dataset_yaml, yolo_line


def test_yolo_line_normalises():
    line = yolo_line((100, 200, 300, 400), 1000, 800, cls=1)
    cls, cx, cy, w, h = line.split()
    assert cls == "1"
    assert float(cx) == 0.2 and float(cy) == 0.375
    assert float(w) == 0.2 and float(h) == 0.25


class FakeDetector:
    def __call__(self, frame):
        h, w = frame.shape[:2]
        boxes = np.array([[w * 0.1, h * 0.2, w * 0.15, h * 0.4]])
        return sv.Detections(xyxy=boxes, confidence=np.array([0.9]), class_id=np.array([0]))


def _write_clip(path, frames=10, size=(64, 96)):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size[1], size[0]))
    for i in range(frames):
        img = np.full((*size, 3), i * 20, dtype=np.uint8)
        writer.write(img)
    writer.release()


def test_prelabel_and_dataset_yaml(tmp_path):
    clip = tmp_path / "play.mp4"
    _write_clip(clip)
    out = tmp_path / "labels"
    n = prelabel([str(clip)], out, FakeDetector(), every=4)
    assert n == 3  # frames 0, 4, 8
    images = sorted((out / "images").glob("*.jpg"))
    labels = sorted((out / "labels").glob("*.txt"))
    assert [p.stem for p in images] == [p.stem for p in labels] == ["play_00000", "play_00004", "play_00008"]
    assert labels[0].read_text().startswith("0 0.125000 0.300000")

    yaml_path = write_dataset_yaml(out, val_fraction=0.34)
    text = yaml_path.read_text()
    assert "names:" in text and all(c in text for c in CLASSES)
    assert len((out / "val.txt").read_text().split()) == 1
    assert len((out / "train.txt").read_text().split()) == 2
