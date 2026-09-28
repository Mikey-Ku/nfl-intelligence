"""Coverage recognition: man or zone, and which shell, from how the defenders line up and move.

Trained on Big Data Bowl tracking with the coverage charted per play by NFL Next Gen Stats
(published in nflverse participation data). Features use only positions: where the coverage
defenders stand at the snap relative to the line of scrimmage, where they are a moment later,
and how closely each one shadows the receiver nearest to him (the tell for man coverage).

The model reads the play at any moment from the snap to 2 s after it; the longer it watches,
the better the read.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bdb import DEFAULT_OUTCOMES, DEFAULT_ROOT, HZ, Play, iter_plays, load_outcomes
from .field import FIELD_WIDTH

DEFAULT_MODEL = Path("models/coverage.pkl")
SHELLS = ["COVER_0", "COVER_1", "2_MAN", "COVER_2", "COVER_3", "COVER_4", "COVER_6"]
SHELL_NAMES = {"COVER_0": "Cover 0", "COVER_1": "Cover 1", "2_MAN": "2-Man", "COVER_2": "Cover 2",
               "COVER_3": "Cover 3", "COVER_4": "Cover 4", "COVER_6": "Cover 6"}
SLOTS = 7              # defenders described, deepest first
READ_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0)


def _pad(values, n=SLOTS) -> np.ndarray:
    out = np.full(n, np.nan)
    v = np.asarray(values, dtype=np.float64)[:n]
    out[: len(v)] = v
    return out


def coverage_features(xy: np.ndarray, defenders: list[int], receivers: list[int], los: float, t: int,
                      seconds: float | None = None) -> np.ndarray | None:
    """xy is (frames, players, 2) from the snap; `t` is the frame to read the defence at.
    `seconds` is the time since the snap if the frames aren't 10 per second."""
    if len(defenders) < 5 or t >= len(xy):
        return None
    a, b = xy[0, defenders], xy[t, defenders]
    order = np.argsort(-a[:, 0])[:SLOTS]                       # deepest defender first
    depth0, depth1 = a[order, 0] - los, b[order, 0] - los
    wide0, wide1 = np.abs(a[order, 1] - FIELD_WIDTH / 2), np.abs(b[order, 1] - FIELD_WIDTH / 2)
    moved = np.linalg.norm(b[order] - a[order], axis=1)
    back = b[order, 0] - a[order, 0]                           # dropping (+) or coming up (-)
    shadow = []
    if receivers:
        r0, r1 = xy[0, receivers], xy[t, receivers]
        for i in order:
            k = int(np.argmin(np.linalg.norm(r0 - a[i], axis=1)))
            shadow.append(float(np.linalg.norm((b[i] - a[i]) - (r1[k] - r0[k]))))
    summary = [
        t / HZ if seconds is None else seconds, len(defenders),
        float((depth0 > 10).sum()), float((depth1 > 10).sum()),
        float(np.nanmean(shadow)) if shadow else np.nan,
        float(np.nanmin(shadow)) if shadow else np.nan,
    ]
    return np.concatenate([_pad(depth0), _pad(depth1), _pad(wide0), _pad(wide1), _pad(moved), _pad(back), _pad(shadow), summary])


@dataclass
class CoverageRead:
    man: float                     # probability of man coverage
    shell: str                     # most likely shell, e.g. COVER_3
    shell_confidence: float

    @property
    def text(self) -> str:
        kind = "Man" if self.man >= 0.5 else "Zone"
        sure = self.man if self.man >= 0.5 else 1 - self.man
        return f"{kind} {sure:.0%}, {SHELL_NAMES.get(self.shell, self.shell)} {self.shell_confidence:.0%}"


class CoverageModel:
    def __init__(self, man_zone, shell, meta: dict | None = None):
        self.man_zone = man_zone
        self.shell = shell
        self.meta = meta or {}

    def save(self, path: str | Path = DEFAULT_MODEL) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"man_zone": self.man_zone, "shell": self.shell, "meta": self.meta}, f)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> CoverageModel:
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        return cls(d["man_zone"], d["shell"], d.get("meta"))

    def read_many(self, X: np.ndarray) -> list[CoverageRead]:
        X = np.atleast_2d(X)
        man = self.man_zone.predict_proba(X)[:, 1]
        shells = self.shell.predict_proba(X)
        classes = self.shell.classes_
        best = shells.argmax(axis=1)
        return [CoverageRead(float(man[i]), SHELLS[int(classes[best[i]])], float(shells[i, best[i]])) for i in range(len(X))]

    def read(self, xy: np.ndarray, defenders, receivers, los: float, t: int, seconds: float | None = None) -> CoverageRead | None:
        if seconds is not None:
            seconds = min(seconds, READ_TIMES[-1])
        else:
            t = min(t, int(READ_TIMES[-1] * HZ))
        f = coverage_features(xy, defenders, receivers, los, t, seconds)
        return None if f is None else self.read_many(f)[0]


def rows_for(weeks, root=DEFAULT_ROOT, outcomes_path=DEFAULT_OUTCOMES, times=READ_TIMES):
    outcomes = load_outcomes(outcomes_path)
    X, man, shell, when = [], [], [], []
    for week in weeks:
        for play in iter_plays(week, root, outcomes):
            if play.man_zone is None:
                continue
            for sec in times:
                t = int(sec * HZ)
                f = coverage_features(play.xy, play.defenders, play.receivers, play.los, t)
                if f is None:
                    continue
                X.append(f)
                man.append(int(play.man_zone == "man"))
                shell.append(SHELLS.index(play.coverage) if play.coverage in SHELLS else -1)
                when.append(sec)
    return np.array(X), np.array(man), np.array(shell), np.array(when)


def train(root=DEFAULT_ROOT, outcomes_path=DEFAULT_OUTCOMES, out: str | Path = DEFAULT_MODEL, verbose: bool = True) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    kw = dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=40,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0)
    X, man, shell, _ = rows_for(range(1, 15), root, outcomes_path)
    Xt, mant, shellt, when = rows_for(range(15, 19), root, outcomes_path)
    mz = HistGradientBoostingClassifier(**kw).fit(X, man)
    known, knownt = shell >= 0, shellt >= 0
    sh = HistGradientBoostingClassifier(**kw).fit(X[known], shell[known])
    model = CoverageModel(mz, sh)
    report = {"train_rows": int(len(X)), "by_time": []}
    p = mz.predict_proba(Xt)[:, 1]
    ps = sh.predict(Xt)
    for sec in READ_TIMES:
        m = when == sec
        mk = m & knownt
        report["by_time"].append({
            "seconds": sec, "plays": int(m.sum()),
            "man_zone_accuracy": float(accuracy_score(mant[m], p[m] > 0.5)),
            "man_zone_auc": float(roc_auc_score(mant[m], p[m])),
            "man_zone_baseline": float(max(mant[m].mean(), 1 - mant[m].mean())),
            "shell_accuracy": float(accuracy_score(shellt[mk], ps[mk])),
            "shell_baseline": float(np.bincount(shellt[mk]).max() / mk.sum()),
        })
    model.meta = {"report": report, "shells": SHELLS}
    model.save(out)
    if verbose:
        print(f"trained on {report['train_rows']} reads (weeks 1-14), tested on weeks 15-18 -> {out}")
        print("  watching for   man/zone right (always-zone)   AUC     shell right (always-most-common)")
        for r in report["by_time"]:
            print(f"  {r['seconds']:>4.1f} s        {r['man_zone_accuracy']:>6.1%}  ({r['man_zone_baseline']:.1%})           {r['man_zone_auc']:.3f}   "
                  f"{r['shell_accuracy']:>6.1%}  ({r['shell_baseline']:.1%})")
    return report
