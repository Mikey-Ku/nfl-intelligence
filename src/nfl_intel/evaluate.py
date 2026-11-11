"""Score a tracking CSV against hand-labeled ground truth in the same schema.

Per frame, predicted and true players are matched one-to-one by distance (Hungarian
assignment) within a threshold. From the matches we report:
  position_error   mean distance of matched pairs (yards if both sides are calibrated, else pixels)
  offset           the steady part of that error: the median of (run minus truth) over all pairs.
                   A run that puts every player at the bottom of his box and a labeller who
                   clicks his ankles differ by most of a yard, the same way every time.
  spread           the error left once the offset is taken out
  recall           matched / true players   (are we finding everyone?)
  precision        matched / predicted players   (are we inventing people?)
  id_switches      times a true player's matched identity changed (the brief's "identity swaps").
                   The identity is the run's `slot` where it has one (the number the analysis
                   keeps a player under), else the image tracker's id.
  mota             1 - (misses + false positives + switches) / true players, the standard MOT summary

Players are paired after the offset is taken out. Linemen stand a yard apart; with a yard of
steady offset left in, a man is as near his neighbour's truth as his own and the pairing,
and with it the count of identity changes, is partly chance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import FrameResult, read_csv


@dataclass
class EvalResult:
    frames: int
    unit: str                 # "yd" or "px"
    position_error: float
    recall: float
    precision: float
    id_switches: int
    mota: float
    matched: int
    misses: int
    false_positives: int
    offset: tuple[float, float] = (0.0, 0.0)
    spread: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        steady = ""
        if np.hypot(*self.offset) >= 0.05 * (1.0 if self.unit == "yd" else 20.0):
            steady = f" ({self.spread:.2f} around a steady offset of {self.offset[0]:+.2f}, {self.offset[1]:+.2f})"
        return (
            f"{self.frames} frames  error {self.position_error:.2f} {self.unit}{steady}  "
            f"recall {self.recall:.2%}  precision {self.precision:.2%}  "
            f"id switches {self.id_switches}  MOTA {self.mota:.3f}"
        )


def _positions(res: FrameResult, use_field: bool, by_slot: bool = False) -> tuple[np.ndarray, list[int]]:
    pts, ids = [], []
    for p in res.players:
        xy = p.field_yd if use_field else p.foot_px
        if xy is None:
            continue
        pts.append(xy)
        ids.append(p.slot if by_slot and p.slot is not None else p.track_id)
    return np.array(pts, dtype=float).reshape(-1, 2), ids


def evaluate(pred: list[FrameResult], truth: list[FrameResult], threshold: float | None = None) -> EvalResult:
    use_field = all(p.field_yd is not None for r in truth for p in r.players) and any(
        p.field_yd is not None for r in pred for p in r.players
    )
    unit = "yd" if use_field else "px"
    if threshold is None:
        threshold = 3.0 if use_field else 60.0

    pred_by_frame = {r.frame: r for r in pred}
    by_slot = any(p.slot is not None for r in pred for p in r.players)    # the identity the analysis keeps, when the run has it

    def pair_up(shift: np.ndarray):
        """Per frame: (true points, true ids, run points, run ids, [(i, j)] within the threshold of each other once shifted)."""
        out = []
        for t in truth:
            gt_pts, gt_ids = _positions(t, use_field)
            p = pred_by_frame.get(t.frame)
            pr_pts, pr_ids = _positions(p, use_field, by_slot) if p is not None else (np.zeros((0, 2)), [])
            pairs = []
            if len(gt_pts) and len(pr_pts):
                cost = np.linalg.norm((gt_pts + shift)[:, None, :] - pr_pts[None, :, :], axis=2)
                rows, cols = linear_sum_assignment(cost)
                pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] <= threshold]
            out.append((gt_pts, gt_ids, pr_pts, pr_ids, pairs))
        return out

    offset = np.zeros(2)
    first = [pr[j] - gt[i] for gt, _, pr, _, pairs in pair_up(offset) for i, j in pairs]
    if len(first) >= 20:                                    # too few pairs and the "steady" part is just noise
        offset = np.median(np.array(first), axis=0)
    matched = misses = false_pos = switches = 0
    errors: list[float] = []
    spread: list[float] = []
    last_match: dict[int, int] = {}          # true id -> last matched predicted id
    frames = 0
    for gt_pts, gt_ids, pr_pts, pr_ids, pairs in pair_up(offset):
        frames += 1
        for i, j in pairs:
            errors.append(float(np.linalg.norm(gt_pts[i] - pr_pts[j])))
            spread.append(float(np.linalg.norm(gt_pts[i] + offset - pr_pts[j])))
            gid, pid = gt_ids[i], pr_ids[j]
            if gid in last_match and last_match[gid] != pid:
                switches += 1
            last_match[gid] = pid
        matched += len(pairs)
        misses += len(gt_pts) - len(pairs)
        false_pos += len(pr_pts) - len(pairs)
    total_gt = matched + misses
    return EvalResult(
        frames=frames,
        unit=unit,
        position_error=float(np.mean(errors)) if errors else float("nan"),
        recall=matched / total_gt if total_gt else 0.0,
        precision=matched / (matched + false_pos) if (matched + false_pos) else 0.0,
        id_switches=switches,
        mota=1.0 - (misses + false_pos + switches) / total_gt if total_gt else 0.0,
        matched=matched,
        misses=misses,
        false_positives=false_pos,
        offset=(float(offset[0]), float(offset[1])),
        spread=float(np.mean(spread)) if spread else float("nan"),
    )


def evaluate_files(pred_csv: str | Path, truth_csv: str | Path, threshold: float | None = None) -> EvalResult:
    return evaluate(read_csv(pred_csv), read_csv(truth_csv), threshold)
