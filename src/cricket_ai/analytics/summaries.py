"""Batting, bowling, phase and team summaries.

Conventions follow standard scoring:
* balls faced excludes wides; strike rate = runs / balls faced * 100
* bowler runs conceded = batter runs + wides + no-balls (byes/leg-byes not charged)
* bowler balls = legal deliveries; economy = runs conceded / overs
* bowler wickets exclude run outs, retirements, obstructing the field, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _regular(deliveries: pd.DataFrame) -> pd.DataFrame:
    return deliveries[~deliveries.is_super_over]


def batting_summary(deliveries: pd.DataFrame, min_balls: int = 0) -> pd.DataFrame:
    d = _regular(deliveries)
    faced = d[d.wides == 0]
    agg = faced.groupby("batter", observed=True).agg(
        innings=("match_id", lambda s: s.nunique()),
        runs=("runs_batter", "sum"),
        balls=("runs_batter", "size"),
        fours=("runs_batter", lambda s: int((s == 4).sum())),
        sixes=("runs_batter", lambda s: int((s == 6).sum())),
        dots=("runs_total", lambda s: int((s == 0).sum())),
    )
    outs = (
        d[d.player_out.notna() & ~d.wicket_kind.isin(["retired hurt", "retired not out"])]
        .groupby("player_out")
        .size()
        .rename("dismissals")
    )
    agg = agg.join(outs, how="left").fillna({"dismissals": 0})
    agg["dismissals"] = agg.dismissals.astype(int)
    agg["strike_rate"] = (agg.runs / agg.balls * 100).round(2)
    agg["average"] = np.where(agg.dismissals > 0, agg.runs / agg.dismissals.clip(lower=1), np.nan)
    agg["average"] = agg.average.round(2)
    agg["boundary_pct"] = ((agg.fours + agg.sixes) / agg.balls * 100).round(2)
    return agg[agg.balls >= min_balls].sort_values("runs", ascending=False)


def bowling_summary(deliveries: pd.DataFrame, min_balls: int = 0) -> pd.DataFrame:
    d = _regular(deliveries).assign(
        conceded=lambda x: x.runs_batter + x.wides + x.noballs,
        legal=lambda x: x.is_legal.astype(int),
    )
    agg = d.groupby("bowler", observed=True).agg(
        matches=("match_id", lambda s: s.nunique()),
        balls=("legal", "sum"),
        runs_conceded=("conceded", "sum"),
        wickets=("bowler_wickets", "sum"),
        wides=("wides", lambda s: int((s > 0).sum())),
        noballs=("noballs", lambda s: int((s > 0).sum())),
        dots=("runs_total", lambda s: int((s == 0).sum())),
    )
    overs = agg.balls / 6
    agg["economy"] = (agg.runs_conceded / overs.replace(0, np.nan)).round(2)
    agg["average"] = (agg.runs_conceded / agg.wickets.replace(0, np.nan)).round(2)
    agg["strike_rate"] = (agg.balls / agg.wickets.replace(0, np.nan)).round(2)
    agg["dot_pct"] = (agg.dots / agg.balls.replace(0, np.nan) * 100).round(2)
    return agg[agg.balls >= min_balls].sort_values("wickets", ascending=False)


def assign_phase(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    """Powerplay / middle / death, using T20 or ODI boundaries depending on scheduled overs."""
    overs = deliveries.match_id.map(matches.set_index("match_id").overs).fillna(20)
    t20 = overs <= 20
    o = deliveries.over
    phase = np.where(
        t20,
        np.select([o <= 5, o <= 14], ["powerplay", "middle"], "death"),
        np.select([o <= 9, o <= 39], ["powerplay", "middle"], "death"),
    )
    return pd.Series(phase, index=deliveries.index, name="phase")


def phase_summary(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    d = _regular(deliveries)
    d = d.assign(phase=assign_phase(d, matches), legal=d.is_legal.astype(int))
    agg = d.groupby(["phase", "innings"]).agg(
        runs=("runs_total", "sum"),
        balls=("legal", "sum"),
        wickets=("wickets_on_ball", "sum"),
    )
    agg["run_rate"] = (agg.runs / agg.balls.replace(0, np.nan) * 6).round(2)
    agg["balls_per_wicket"] = (agg.balls / agg.wickets.replace(0, np.nan)).round(1)
    order = {"powerplay": 0, "middle": 1, "death": 2}
    return agg.reset_index().sort_values(
        ["innings", "phase"], key=lambda s: s.map(order) if s.name == "phase" else s
    )


def team_season_summary(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in matches.iterrows():
        for team in (m.team1, m.team2):
            if team is None:
                continue
            rows.append(
                {
                    "season": m.season,
                    "team": team,
                    "played": 1,
                    "won": int(m.winner == team),
                    "no_result": int(m.result == "no result"),
                    "toss_won": int(m.toss_winner == team),
                    "toss_and_match_won": int(m.toss_winner == team and m.winner == team),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby(["season", "team"]).sum()
    decided = (df.played - df.no_result).replace(0, np.nan)
    df["win_pct"] = (df.won / decided * 100).round(1)
    return df.reset_index().sort_values(["season", "win_pct"], ascending=[True, False])
