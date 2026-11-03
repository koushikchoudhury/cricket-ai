"""Generate plausible Cricsheet-format T20 matches for tests.

Not a data source for modelling - just structurally faithful JSON so the
pipeline can be exercised end to end without network access.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

OUTCOMES = [0, 1, 2, 3, 4, 6, "W", "wd", "nb", "lb"]
WEIGHTS = [34, 33, 8, 1, 11, 5, 5, 2, 0.6, 1.4]


def _innings(rng, batting, bowling_xi, batting_xi, target=None, overs=20):
    order = list(batting_xi)
    striker, non_striker, nxt = order[0], order[1], 2
    runs = wkts = 0
    out = {"team": batting, "overs": []}
    if target:
        out["target"] = {"overs": overs, "runs": target}
    for o in range(overs):
        bowler = bowling_xi[6 + (o % 5)]
        dels, legal = [], 0
        while legal < 6:
            r = rng.choices(OUTCOMES, WEIGHTS)[0]
            d = {"batter": striker, "bowler": bowler, "non_striker": non_striker}
            if r == "wd":
                d |= {"extras": {"wides": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}}
            elif r == "nb":
                d |= {"extras": {"noballs": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}}
            elif r == "lb":
                d |= {"extras": {"legbyes": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}}
                legal += 1
            elif r == "W":
                kind = rng.choice(["caught", "bowled", "lbw", "run out"])
                w = {"player_out": striker, "kind": kind}
                if kind in ("caught", "run out"):
                    w["fielders"] = [{"name": rng.choice(bowling_xi)}]
                d |= {"runs": {"batter": 0, "extras": 0, "total": 0}, "wickets": [w]}
                legal += 1
            else:
                d |= {"runs": {"batter": r, "extras": 0, "total": r}}
                legal += 1
            dels.append(d)
            runs += d["runs"]["total"]
            if r == "W":
                wkts += 1
                if wkts == 10:
                    break
                striker, nxt = order[nxt], nxt + 1
            elif isinstance(r, int) and r % 2 == 1:
                striker, non_striker = non_striker, striker
            if target and runs >= target:
                break
        out["overs"].append({"over": o, "deliveries": dels})
        striker, non_striker = non_striker, striker
        if wkts == 10 or (target and runs >= target):
            break
    return out, runs, wkts


def generate_match(seed: int, teams=("Kolkata", "Mumbai"), start=date(2020, 4, 1)):
    rng = random.Random(seed)
    t1, t2 = rng.sample(list(teams), 2)
    xi = {t: [f"{t[:3]}_{i}" for i in range(11)] for t in teams}
    inn1, r1, _ = _innings(rng, t1, xi[t2], xi[t1])
    inn2, r2, w2 = _innings(rng, t2, xi[t1], xi[t2], target=r1 + 1)
    if r2 > r1:
        outcome = {"winner": t2, "by": {"wickets": 10 - w2}}
    elif r2 == r1:
        outcome = {"result": "tie"}
    else:
        outcome = {"winner": t1, "by": {"runs": r1 - r2}}
    day = start + timedelta(days=seed)
    return {
        "meta": {"data_version": "1.1.0", "created": str(day), "revision": 1},
        "info": {
            "balls_per_over": 6,
            "dates": [str(day)],
            "event": {"name": "Synthetic League", "match_number": seed},
            "gender": "male",
            "match_type": "T20",
            "outcome": outcome,
            "overs": 20,
            "players": xi,
            "registry": {"people": {p: p.lower() for ps in xi.values() for p in ps}},
            "season": day.year,
            "team_type": "club",
            "teams": [t1, t2],
            "toss": {"winner": t1, "decision": "bat"},
            "venue": rng.choice(["Eden Gardens", "Wankhede Stadium"]),
        },
        "innings": [inn1, inn2],
    }
