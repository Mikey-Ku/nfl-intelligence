"""Read text off a frame with Apple's Vision framework (on-device, nothing leaves the Mac).

Used to read NFL Pro's play header ("Q3 03:41 / 3RD & 1 / ATL ON GB 7") and the play
description under the video, which identify the play in the public play-by-play.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TextBox:
    text: str
    confidence: float
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 in pixels of the image given


def read_text(image_bgr: np.ndarray, accurate: bool = True) -> list[TextBox]:
    """All text Vision finds in the image, top to bottom."""
    import Quartz
    import Vision

    h, w = image_bgr.shape[:2]
    ok, png = cv2.imencode(".png", image_bgr)
    if not ok:
        return []
    data = Quartz.CFDataCreate(None, png.tobytes(), len(png))
    source = Quartz.CGImageSourceCreateWithData(data, None)
    cg = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg is None:
        return []
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelAccurate if accurate else Vision.VNRequestTextRecognitionLevelFast
    )
    request.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    success, _ = handler.performRequests_error_([request], None)
    if not success:
        return []
    out = []
    for obs in request.results() or []:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        bb = obs.boundingBox()                  # normalised, origin bottom-left
        x1, y2 = bb.origin.x * w, (1.0 - bb.origin.y) * h
        x2, y1 = x1 + bb.size.width * w, y2 - bb.size.height * h
        out.append(TextBox(str(cands[0].string()), float(cands[0].confidence()), (x1, y1, x2, y2)))
    out.sort(key=lambda t: (round(t.box[1] / 10), t.box[0]))
    return out
