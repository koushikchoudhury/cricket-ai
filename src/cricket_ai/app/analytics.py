"""Analytics page: batting, bowling, phases and teams, filtered by season."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from cricket_ai.analytics import (
    batting_summary,
    bowling_summary,
    phase_summary,
    team_season_summary,
)
from cricket_ai.app import state

ALL = "All seasons"


@st.cache_data(show_spinner="Crunching the numbers ...")
def summaries(processed: str, _mtime: float, season: str, min_balls: int) -> dict:
    matches = pd.read_parquet(f"{processed}/matches.parquet")
    deliveries = pd.read_parquet(f"{processed}/deliveries.parquet")
    if season != ALL:
        matches = matches[matches.season == season]
        deliveries = deliveries[deliveries.match_id.isin(matches.match_id)]
    teams = team_season_summary(matches)
    if season == ALL and len(teams):
        teams = teams.groupby("team", as_index=False)[
            ["played", "won", "no_result", "toss_won", "toss_and_match_won"]
        ].sum()
        decided = (teams.played - teams.no_result).where(lambda s: s > 0)
        teams["win_pct"] = (teams.won / decided * 100).round(1)
    return {
        "batting": batting_summary(deliveries, min_balls=min_balls).reset_index(),
        "bowling": bowling_summary(deliveries, min_balls=min_balls).reset_index(),
        "phases": phase_summary(deliveries, matches),
        "teams": teams.sort_values("win_pct", ascending=False) if len(teams) else teams,
        "n_matches": len(matches),
    }


def top_bar(df: pd.DataFrame, name: str, value: str, title: str, n: int = 15) -> alt.Chart:
    top = df.nlargest(n, value)
    return (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title=title),
            y=alt.Y(f"{name}:N", sort="-x", title=None),
            tooltip=list(top.columns),
        )
        .properties(height=24 * len(top) + 40)
    )


def render(p: state.Paths) -> None:
    st.title("Analytics")
    matches = state.parquet(p.processed / "matches.parquet")
    if matches is None:
        state.need(
            "The match data",
            f"cricket-ai data download {p.dataset}\ncricket-ai data build {p.dataset}",
        )
        return

    seasons = sorted(matches.season.dropna().unique(), key=state.season_key, reverse=True)
    c1, c2 = st.columns([1, 2])
    season = c1.selectbox("Season", [ALL, *seasons])
    default = 300 if season == ALL else 60
    min_balls = c2.slider(
        "Minimum balls faced / bowled",
        0,
        1000,
        default,
        step=10,
        help="Hides players with too few balls for rates such as strike rate to mean much.",
    )
    data = summaries(
        str(p.processed), state.mtime(p.processed / "deliveries.parquet"), season, min_balls
    )
    st.caption(f"{data['n_matches']:,} matches in {season.lower()}.")

    bat_tab, bowl_tab, phase_tab, team_tab = st.tabs(["Batting", "Bowling", "Phases", "Teams"])
    with bat_tab:
        st.altair_chart(top_bar(data["batting"], "batter", "runs", "Runs"))
        st.dataframe(data["batting"], hide_index=True)
    with bowl_tab:
        st.altair_chart(top_bar(data["bowling"], "bowler", "wickets", "Wickets"))
        st.dataframe(data["bowling"], hide_index=True)
    with phase_tab:
        phases = data["phases"].assign(innings=lambda d: "Innings " + d.innings.astype(str))
        st.altair_chart(
            alt.Chart(phases)
            .mark_bar()
            .encode(
                x=alt.X(
                    "phase:N",
                    sort=["powerplay", "middle", "death"],
                    title=None,
                    axis=alt.Axis(labelAngle=0),
                ),
                xOffset="innings:N",
                y=alt.Y("run_rate:Q", title="Runs per over"),
                color=alt.Color("innings:N", title=None, legend=alt.Legend(orient="top")),
                tooltip=["phase", "innings", "runs", "balls", "wickets", "run_rate"],
            )
            .properties(height=320)
        )
        st.dataframe(phases, hide_index=True)
    with team_tab:
        st.dataframe(data["teams"], hide_index=True)
