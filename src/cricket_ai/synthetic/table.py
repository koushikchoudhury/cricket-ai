"""The real table that module 2 learns to imitate: one row per over of a T20 innings.

Each row is the state at the start of the over (innings, over number, score, wickets)
and what happened in it (runs, wickets, extras, boundaries, dots, legal balls). The mix
of small-count discrete columns, skewed continuous columns and hard cricket constraints
is what makes it a useful test for tabular generative models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DISCRETE = [
    "innings",
    "over",
    "wickets_at_start",
    "wickets",
    "extras",
    "fours",
    "sixes",
    "dots",
    "legal_balls",
]
CONTINUOUS = ["runs_at_start", "runs"]
COLUMNS = [
    "innings",
    "over",
    "runs_at_start",
    "wickets_at_start",
    "runs",
    "wickets",
    "extras",
    "fours",
    "sixes",
    "dots",
    "legal_balls",
]

# Hard rules every real over satisfies. A synthetic row that breaks one is impossible cricket.
VALIDITY_RULES = {
    "non_negative": lambda t: (t[COLUMNS] >= 0).all(axis=1),
    "wickets_total_le_10": lambda t: t.wickets_at_start + t.wickets <= 10,
    "runs_cover_boundaries_and_extras": lambda t: t.runs >= 4 * t.fours + 6 * t.sixes + t.extras,
    "dots_le_legal_balls": lambda t: t.dots <= t.legal_balls,
    "scoring_balls_fit_in_over": lambda t: t.fours + t.sixes + t.dots <= t.legal_balls + t.extras,
    "legal_balls_le_7": lambda t: t.legal_balls <= 7,
    "first_over_starts_at_zero": lambda t: (
        (t.over != 0) | ((t.runs_at_start == 0) & (t.wickets_at_start == 0))
    ),
    "start_score_reachable": lambda t: t.runs_at_start <= 40 * t.over,
}


def build_overs(
    matches: pd.DataFrame, deliveries: pd.DataFrame, max_overs: int = 20
) -> pd.DataFrame:
    """Aggregate deliveries of scheduled ``max_overs``-over T20 matches into over rows."""
    m = matches[(matches.overs == max_overs) & matches.match_type.isin(["T20", "IT20"])]
    d = deliveries[
        deliveries.match_id.isin(m.match_id)
        & ~deliveries.is_super_over
        & deliveries.innings.isin([1, 2])
        & (deliveries.over < max_overs)
    ]
    d = d.assign(
        four=(d.runs_batter == 4).astype(np.int16),
        six=(d.runs_batter == 6).astype(np.int16),
        dot=(d.runs_total == 0).astype(np.int16),
        legal=d.is_legal.astype(np.int16),
    )
    o = (
        d.groupby(["match_id", "innings", "over"], sort=True, observed=True)
        .agg(
            runs=("runs_total", "sum"),
            extras=("runs_extras", "sum"),
            fours=("four", "sum"),
            sixes=("six", "sum"),
            dots=("dot", "sum"),
            legal_balls=("legal", "sum"),
            wickets_end=("team_wickets", "max"),
            runs_end=("team_runs", "max"),
        )
        .reset_index()
    )
    by_innings = o.groupby(["match_id", "innings"], sort=False)
    o["runs_at_start"] = o.runs_end - o.runs
    o["wickets_at_start"] = by_innings.wickets_end.shift(1).fillna(0)
    o["wickets"] = o.wickets_end - o.wickets_at_start

    season = m.set_index("match_id").season
    o["season"] = o.match_id.map(season)
    out = o[["match_id", "season", *COLUMNS]].copy()
    for c in COLUMNS:
        out[c] = out[c].astype(np.int64)
    return out.reset_index(drop=True)


def validity(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of rows passing each rule, plus ``all_rules``."""
    passed = pd.DataFrame({name: rule(df) for name, rule in VALIDITY_RULES.items()})
    rates = {name: round(float(passed[name].mean()), 4) for name in passed}
    rates["all_rules"] = round(float(passed.all(axis=1).mean()), 4)
    return rates


def split_by_match(
    overs: pd.DataFrame, holdout_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/holdout split with whole matches on one side.

    Generators only ever see the train part; the holdout is the reference for
    utility and privacy evaluation.
    """
    ids = np.array(sorted(overs.match_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    hold = set(ids[: max(1, int(len(ids) * holdout_frac))])
    mask = overs.match_id.isin(hold)
    return (
        overs.loc[~mask, COLUMNS].reset_index(drop=True),
        overs.loc[mask, COLUMNS].reset_index(drop=True),
    )
