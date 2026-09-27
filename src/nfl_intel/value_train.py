"""Train and evaluate the pass value models on Big Data Bowl throws."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bdb import DEFAULT_OUTCOMES, DEFAULT_ROOT, HZ, iter_plays, load_outcomes
from .value import DEFAULT_MODEL, FEATURES, PassValueModel, receiver_features, velocities

TRAIN_WEEKS = range(1, 15)
TEST_WEEKS = range(15, 19)


def throw_rows(weeks, root=DEFAULT_ROOT, outcomes_path=DEFAULT_OUTCOMES, noise: float = 0.0, seed: int = 0):
    """One row per play: features of the real target at the throw, and what happened.

    `noise` adds Gaussian position error (yards) to every player, to mimic tracking from video.
    """
    rng = np.random.default_rng(seed)
    outcomes = load_outcomes(outcomes_path)
    X, caught, yards, meta = [], [], [], []
    for week in weeks:
        for play in iter_plays(week, root, outcomes):
            if play.qb < 0 or play.target < 0 or play.complete is None or play.yards is None or len(play.xy) < 3:
                continue
            xy = play.xy + rng.normal(0.0, noise, play.xy.shape) if noise > 0 else play.xy
            v = velocities(xy, HZ)
            t = len(xy) - 1
            X.append(receiver_features(xy[t], v[t], play.target, play.qb, play.defenders, play.los, t / HZ))
            caught.append(play.complete)
            yards.append(play.yards)
            meta.append((play.game_id, play.play_id, week))
    return np.array(X), np.array(caught, dtype=int), np.array(yards, dtype=float), meta


def fit(X, caught, yards, seed: int = 0) -> PassValueModel:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    common = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=seed)
    catch = HistGradientBoostingClassifier(**common).fit(X, caught)
    done = caught == 1
    yd = HistGradientBoostingRegressor(loss="absolute_error", **common).fit(X[done], yards[done])
    return PassValueModel(catch, yd)


def evaluate(model: PassValueModel, X, caught, yards) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    p, y = model.score(X)
    base = float(caught.mean())
    done = caught == 1
    bins = np.quantile(p, np.linspace(0, 1, 6))
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            calib.append((float(p[m].mean()), float(caught[m].mean()), int(m.sum())))
    return {
        "plays": int(len(X)),
        "catch_rate": base,
        "auc": float(roc_auc_score(caught, p)),
        "log_loss": float(log_loss(caught, p)),
        "log_loss_baseline": float(log_loss(caught, np.full(len(caught), base))),
        "brier": float(brier_score_loss(caught, p)),
        "brier_baseline": float(base * (1 - base)),
        "yards_mae": float(np.abs(y[done] - yards[done]).mean()),
        "yards_mae_baseline": float(np.abs(np.median(yards[done]) - yards[done]).mean()),
        "expected_yards_mean": float((p * y).mean()),
        "actual_yards_mean": float(np.where(done, yards, 0.0).mean()),
        "calibration": calib,
    }


def train(root=DEFAULT_ROOT, outcomes_path=DEFAULT_OUTCOMES, out: str | Path = DEFAULT_MODEL, noise: float = 0.3, verbose: bool = True) -> dict:
    """Fit on weeks 1-14 (clean + noisy copies), test on weeks 15-18 clean and noisy."""
    Xc, cc, yc, _ = throw_rows(TRAIN_WEEKS, root, outcomes_path)
    Xn, cn, yn, _ = throw_rows(TRAIN_WEEKS, root, outcomes_path, noise=noise, seed=1)
    model = fit(np.vstack([Xc, Xn]), np.concatenate([cc, cn]), np.concatenate([yc, yn]))
    Xt, ct, yt, _ = throw_rows(TEST_WEEKS, root, outcomes_path)
    Xtn, ctn, ytn, _ = throw_rows(TEST_WEEKS, root, outcomes_path, noise=1.0, seed=2)
    report = {
        "train_plays": int(len(Xc)),
        "test_clean": evaluate(model, Xt, ct, yt),
        "test_noisy_1yd": evaluate(model, Xtn, ctn, ytn),
    }
    try:
        from sklearn.inspection import permutation_importance

        imp = permutation_importance(model.catch, Xt, ct, n_repeats=5, random_state=0, scoring="neg_log_loss")
        report["importance"] = sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1])
    except Exception:                                   # pragma: no cover
        report["importance"] = []
    baseline = typical_value_curve(model, TRAIN_WEEKS, root, outcomes_path)
    report["baseline"] = baseline
    model.meta = {"train_weeks": list(TRAIN_WEEKS), "test_weeks": list(TEST_WEEKS), "noise": noise,
                  "baseline": baseline,
                  "test": {k: v for k, v in report["test_clean"].items() if k != "calibration"}}
    model.save(out)
    if verbose:
        print_report(report, out)
    return report


def typical_value_curve(model: PassValueModel, weeks, root=DEFAULT_ROOT, outcomes_path=DEFAULT_OUTCOMES, seconds: float = 6.0) -> list[float]:
    """Average play value at each tenth of a second after the snap, over plays still going."""
    from .demo import score_plays

    outcomes = load_outcomes(outcomes_path)
    n = int(seconds * HZ) + 1
    total, count = np.zeros(n), np.zeros(n)
    for week in weeks:
        plays = [p for p in iter_plays(week, root, outcomes) if p.qb >= 0 and p.receivers and len(p.xy) >= 3]
        for sp in score_plays(plays, model):
            k = min(n, len(sp.values))
            total[:k] += sp.values[:k]
            count[:k] += 1
    curve = np.where(count >= 50, total / np.maximum(count, 1), np.nan)
    last = float(curve[~np.isnan(curve)][-1])
    return [float(v) if not np.isnan(v) else last for v in curve]


def print_report(report: dict, out) -> None:
    print(f"trained on {report['train_plays']} throws (weeks 1-14), tested on weeks 15-18 -> {out}")
    for name in ("test_clean", "test_noisy_1yd"):
        r = report[name]
        print(f"\n{name}: {r['plays']} plays, catch rate {r['catch_rate']:.1%}")
        print(f"  catch model   AUC {r['auc']:.3f}   log loss {r['log_loss']:.3f} (always-average {r['log_loss_baseline']:.3f})"
              f"   Brier {r['brier']:.3f} (baseline {r['brier_baseline']:.3f})")
        print(f"  yards model   off by {r['yards_mae']:.1f} yd on average (always-median {r['yards_mae_baseline']:.1f})")
        print(f"  expected yards per throw {r['expected_yards_mean']:.2f} vs actual {r['actual_yards_mean']:.2f}")
        print("  calibration (predicted -> actual, plays): " + "  ".join(f"{a:.0%}->{b:.0%} ({n})" for a, b, n in r["calibration"]))
    if report.get("baseline"):
        b = report["baseline"]
        print("\ntypical play value by time since the snap: " + "  ".join(f"{t:.0f}s {b[int(t * HZ)]:.1f}" for t in (0, 1, 2, 3, 4, 5) if int(t * HZ) < len(b)))
    if report.get("importance"):
        print("\nwhat the catch model leans on: " + ", ".join(f"{k} {v:.3f}" for k, v in report["importance"][:8]))
