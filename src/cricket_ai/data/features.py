"""Feature tables for modelling.

``chase_states`` turns second-innings deliveries into one training row per
match state (after each delivery) for the win-probability model in module 1.
Label: did the chasing side go on to win?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cricket_ai.data.parse import overs_to_balls

CHASE_FEATURES = [
    "runs_needed",
    "balls_left",
    "wickets_in_hand",
    "target",
    "current_run_rate",
    "required_run_rate",
    "runs_last_12_balls",
    "wickets_last_12_balls",
]
RRR_CAP = 36.0  # required run rate is unbounded near the end; cap it


def chase_states(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    include_dl: bool = False,
) -> pd.DataFrame:
    """One row per second-innings delivery state for decided limited-overs matches."""
    m = matches[
        matches.result.eq("win")
        & matches.target_runs.notna()
        & matches.match_type.isin(["T20", "IT20", "ODI", "ODM"])
    ].copy()
    if not include_dl:
        m = m[m.method.isna()]
    m["max_balls"] = [
        overs_to_balls(o, int(b)) for o, b in zip(m.target_overs, m.balls_per_over, strict=True)
    ]
    cols = ["match_id", "match_type", "start_date", "season", "venue", "winner",
            "target_runs", "max_balls"]

    d = deliveries[(deliveries.innings == 2) & ~deliveries.is_super_over]
    d = d.merge(m[cols], on="match_id", how="inner")
    d = d.sort_values(["match_id", "delivery_seq"])

    out = pd.DataFrame(
        {
            "match_id": d.match_id.values,
            "match_type": d.match_type.values,
            "start_date": d.start_date.values,
            "season": d.season.values,
            "venue": d.venue.values,
            "batting_team": d.batting_team.astype(str).values,
            "bowling_team": d.bowling_team.astype(str).values,
            "delivery_seq": d.delivery_seq.values,
            "target": d.target_runs.astype(int).values,
            "runs_needed": (d.target_runs - d.team_runs).astype(int).values,
            "balls_left": (d.max_balls - d.legal_balls).astype(int).values,
            "wickets_in_hand": (10 - d.team_wickets).astype(int).values,
            "balls_bowled": d.legal_balls.astype(int).values,
            "runs_scored": d.team_runs.astype(int).values,
        }
    )
    out["current_run_rate"] = np.where(
        out.balls_bowled > 0, out.runs_scored * 6.0 / out.balls_bowled.clip(lower=1), 0.0
    )
    out["required_run_rate"] = np.where(
        out.balls_left > 0,
        out.runs_needed * 6.0 / out.balls_left.clip(lower=1),
        RRR_CAP,
    ).clip(max=RRR_CAP)

    # Momentum: runs and wickets over the last 12 deliveries (rolling within each match).
    g = d.groupby("match_id", sort=False)
    out["runs_last_12_balls"] = (
        g.runs_total.rolling(12, min_periods=1).sum().reset_index(level=0, drop=True).reindex(d.index).values
    )
    out["wickets_last_12_balls"] = (
        g.wickets_on_ball.rolling(12, min_periods=1).sum().reset_index(level=0, drop=True).reindex(d.index).values
    )

    out["chasing_team_won"] = (d.winner.values == out.batting_team.values).astype(np.int8)

    # Keep only live states: game not yet decided.
    live = (out.runs_needed > 0) & (out.balls_left > 0) & (out.wickets_in_hand > 0)
    return out[live].reset_index(drop=True)


def split_by_date(df: pd.DataFrame, test_from: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: no match leaks between train and test."""
    cutoff = pd.Timestamp(test_from)
    return df[df.start_date < cutoff], df[df.start_date >= cutoff]
