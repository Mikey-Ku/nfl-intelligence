"""Mark play logs against the play-by-play and the official charting.

A play log (`<clip>.plays.json`) records what the film analysis decided about each play;
FTN's charting records what a person watching the same play wrote down. This puts the two
side by side: where the quarterback lined up, how many backs, how many in the box, how many
rushed. Run the film with `--no-hints` first when marking the formation reads themselves,
otherwise the analysis has already been told some of the answers.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def _look(play: dict) -> dict:
    """What the log says, as numbers."""
    look, defense = play.get("offense_look") or "", play.get("defense_look") or ""
    backs = re.search(r"(\d) back", look)
    box = re.search(r"(\d+)\+ in the box", defense)
    return {
        "shotgun": look.startswith("Shotgun") if look.startswith(("Shotgun", "Under")) else None,
        "backs": int(backs.group(1)) if backs else (0 if "empty" in look else None),
        "box": int(box.group(1)) if box else None,
        "rushers": (play.get("found") or {}).get("DL"),
    }


def mark(plays: list[dict]) -> Counter:
    c: Counter = Counter()
    for p in plays:
        if p.get("play_type") not in ("pass", "run"):
            continue
        ours, ch, found = _look(p), p.get("charting") or {}, p.get("found") or {}
        c["plays"] += 1
        c["quarterback named"] += bool(found.get("QB"))
        if p["play_type"] == "pass":
            c["passes"] += 1
            c["passes valued"] += bool(p.get("value"))
            c["passes graded"] += bool(p.get("throw"))
            c["grades firm"] += bool((p.get("throw") or {}).get("firm"))
        truth = {"S": True, "P": True, "U": False}.get(ch.get("qb_location"), p.get("shotgun"))
        if truth is not None:
            c["alignment marked"] += 1
            c["alignment right"] += ours["shotgun"] is not None and ours["shotgun"] == truth
        if ours["backs"] is not None and ch.get("n_offense_backfield") is not None:
            # the charting counts backs only, not the quarterback (GB-ATL 2026: shotgun with one back is charted 1)
            charted = ch["n_offense_backfield"]
            c["backs marked"] += 1
            c["backs right"] += ours["backs"] == charted
            c["backs within one"] += abs(ours["backs"] - charted) <= 1
        if ours["box"] is not None and ch.get("n_defense_box") is not None:
            c["box marked"] += 1
            c["box right"] += ours["box"] == ch["n_defense_box"]
            c["box within one"] += abs(ours["box"] - ch["n_defense_box"]) <= 1
        if p["play_type"] == "pass" and ours["rushers"] is not None and ch.get("n_pass_rushers"):
            c["rush marked"] += 1
            c["rush right"] += ours["rushers"] == ch["n_pass_rushers"]
            c["rush within one"] += abs(ours["rushers"] - ch["n_pass_rushers"]) <= 1
    return c


def report(paths: list[str | Path]) -> Counter:
    plays = [p for path in paths for p in json.loads(Path(path).read_text())]
    c = mark(plays)

    def row(name: str, hit: str, of: str) -> None:
        if c[of]:
            print(f"  {name:44s} {c[hit]:3d} of {c[of]:3d}  ({c[hit] / c[of]:.0%})")

    print(f"{c['plays']} runs and passes in {len(paths)} log{'s' if len(paths) != 1 else ''}")
    row("quarterback named", "quarterback named", "plays")
    row("passes given a value", "passes valued", "passes")
    row("passes graded (throw found)", "passes graded", "passes")
    row("   of those, firm either side of the moment", "grades firm", "passes graded")
    row("shotgun or under centre read right", "alignment right", "alignment marked")
    row("backs in the backfield, exact", "backs right", "backs marked")
    row("backs in the backfield, within one", "backs within one", "backs marked")
    row("defenders in the box, exact", "box right", "box marked")
    row("defenders in the box, within one", "box within one", "box marked")
    row("pass rushers, exact", "rush right", "rush marked")
    row("pass rushers, within one", "rush within one", "rush marked")
    return c
