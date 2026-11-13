"""Win-probability page: chase calculator, match curves and the model report."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from cricket_ai.app import state
from cricket_ai.winprob.predict import chase_probability


def calculator(model, par_rpo: float | None) -> None:
    st.caption("Describe a T20 chase and see the chasing side's chance of winning.")
    c1, c2, c3, c4, c5 = st.columns(5)
    target = c1.number_input("Target", 1, 350, 180)
    runs = c2.number_input("Runs scored", 0, 350, 100)
    overs = c3.number_input("Overs completed", 0, 19, 12)
    balls = c4.number_input("Balls into the over", 0, 5, 0)
    wickets = c5.number_input("Wickets lost", 0, 9, 2)

    bowled = int(overs) * 6 + int(balls)
    p = chase_probability(model, int(target), int(runs), bowled, int(wickets), par_rpo=par_rpo)
    needed, left = max(int(target) - int(runs), 0), 120 - bowled
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win probability", f"{p:.1%}")
    m2.metric("Runs needed", needed)
    m3.metric("Balls left", left)
    m4.metric("Required run rate", f"{needed * 6 / left:.2f}" if left else "-")
    st.progress(min(max(p, 0.0), 1.0))

    what_if = pd.DataFrame(
        {
            "wickets_lost": range(10),
            "win_probability": [
                chase_probability(model, int(target), int(runs), bowled, w, par_rpo=par_rpo) * 100
                for w in range(10)
            ],
        }
    )
    what_if["now"] = what_if.wickets_lost == int(wickets)
    st.subheader("What if more wickets had fallen?")
    st.altair_chart(
        alt.Chart(what_if)
        .mark_bar()
        .encode(
            x=alt.X(
                "wickets_lost:O",
                title="Wickets lost at this score",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "win_probability:Q", title="Win probability (%)", scale=alt.Scale(domain=[0, 100])
            ),
            color=alt.condition("datum.now", alt.value("#d62728"), alt.value("#9ecae1")),
            tooltip=["wickets_lost", alt.Tooltip("win_probability:Q", format=".1f")],
        )
        .properties(height=280)
    )
    if par_rpo:
        st.caption(
            f"Target difficulty is judged against a par score of {par_rpo * 20:.0f}, the "
            "average first-innings scoring in the most recent matches."
        )


@st.cache_data(show_spinner="Computing the win-probability curve ...")
def curve_for(processed: str, model_path: str, _mtime: float, match_id: str) -> pd.DataFrame:
    from cricket_ai.winprob.curve import match_curve
    from cricket_ai.winprob.model import load

    states = pd.read_parquet(f"{processed}/chase_states.parquet")
    deliveries = pd.read_parquet(f"{processed}/deliveries.parquet")
    curve = match_curve(load(model_path), states, deliveries, match_id)
    cols = [
        "balls_bowled",
        "win_prob",
        "swing",
        "over",
        "ball_in_over",
        "batter",
        "bowler",
        "runs_total",
        "is_wicket",
        "player_out",
        "runs_scored",
        "wickets_in_hand",
        "batting_team",
        "bowling_team",
        "target",
    ]
    return curve[cols]


def curve_chart(curve: pd.DataFrame) -> alt.LayerChart:
    data = curve.assign(
        overs=curve.balls_bowled / 6,
        win_pct=curve.win_prob * 100,
        score=curve.runs_scored.astype(int).astype(str)
        + "/"
        + (10 - curve.wickets_in_hand).astype(int).astype(str),
        is_wicket=curve.is_wicket.eq(True),
    )
    base = alt.Chart(data).encode(
        x=alt.X("overs:Q", title="Overs", scale=alt.Scale(domain=[0, 20])),
        y=alt.Y("win_pct:Q", title="Chasing side win %", scale=alt.Scale(domain=[0, 100])),
    )
    line = base.mark_line().encode(
        tooltip=[
            alt.Tooltip("overs:Q", format=".1f"),
            "score",
            alt.Tooltip("win_pct:Q", format=".1f", title="win %"),
        ]
    )
    wickets = (
        base.transform_filter("datum.is_wicket")
        .mark_point(color="#d62728", size=70, filled=True)
        .encode(
            tooltip=[
                "player_out",
                "bowler",
                "score",
                alt.Tooltip("win_pct:Q", format=".1f", title="win %"),
            ]
        )
    )
    even = (
        alt.Chart(pd.DataFrame({"y": [50]}))
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(y="y:Q")
    )
    return (even + line + wickets).properties(height=360)


def match_curves(p: state.Paths) -> None:
    states = state.parquet(p.processed / "chase_states.parquet")
    matches = state.parquet(p.processed / "matches.parquet")
    chases = matches[matches.match_id.isin(states.match_id.unique())].copy()
    seasons = sorted(chases.season.unique(), key=state.season_key, reverse=True)
    c1, c2 = st.columns([1, 3])
    season = c1.selectbox("Season", seasons)
    pool = chases[chases.season == season].sort_values("start_date", ascending=False)

    def label(match_id: str) -> str:
        m = pool.set_index("match_id").loc[match_id]
        stage = m.get("event_stage")
        extra = f" ({stage})" if isinstance(stage, str) and stage else ""
        return f"{m.start_date:%d %b %Y}: {m.team1} v {m.team2}{extra}"

    match_id = c2.selectbox("Match", pool.match_id.tolist(), format_func=label)
    curve = curve_for(
        str(p.processed), str(p.winprob_model), state.mtime(p.winprob_model), match_id
    )
    m = pool.set_index("match_id").loc[match_id]
    st.subheader(
        f"{curve.batting_team.iloc[0]} chasing {int(curve.target.iloc[0])} "
        f"against {curve.bowling_team.iloc[0]}"
    )
    st.caption(f"Winner: {m.winner}. Red dots are wickets.")
    st.altair_chart(curve_chart(curve))

    st.subheader("Biggest swings")
    top = curve.reindex(curve.swing.abs().sort_values(ascending=False).index).head(6)
    st.dataframe(
        pd.DataFrame(
            {
                "Ball": top.over.astype("Int64").astype(str)
                + "."
                + top.ball_in_over.astype("Int64").astype(str),
                "Batter": top.batter,
                "Bowler": top.bowler,
                "Runs": top.runs_total,
                "Wicket": top.player_out.fillna(""),
                "Win % after": (top.win_prob * 100).round(1),
                "Swing (points)": (top.swing * 100).round(1),
            }
        ),
        hide_index=True,
    )


def report(p: state.Paths) -> None:
    metrics = state.read_json(p.reports / "winprob" / "metrics.json")
    if not metrics:
        state.need("The model report", f"cricket-ai winprob train {p.dataset}")
        return
    st.caption(f"Split: {metrics['split']}. Lower is better except AUC.")
    st.dataframe(
        pd.DataFrame(metrics["test_metrics"]).T.rename_axis("model").reset_index(), hide_index=True
    )
    st.subheader("The hand-written MLP by stage of the chase")
    st.dataframe(pd.DataFrame(metrics["mlp_by_stage"]), hide_index=True)
    for name, caption in (
        ("reliability.png", "Calibration on held-out seasons"),
        ("training_curve.png", "Training curve"),
    ):
        image = p.reports / "winprob" / name
        if image.exists():
            st.image(str(image), caption=caption, width=560)


def render(p: state.Paths) -> None:
    st.title("Win probability")
    if not state.has("torch"):
        st.info('This page needs PyTorch: `pip install -e ".[winprob]"`.')
        return
    model = state.winprob_model(p)
    if model is None:
        state.need(
            "The win-probability model",
            f"cricket-ai data features {p.dataset}\ncricket-ai winprob train {p.dataset}",
        )
        return
    calc_tab, curve_tab, report_tab = st.tabs(["Chase calculator", "Match curve", "Model report"])
    with calc_tab:
        calculator(model, state.latest_par_rpo(p))
    with curve_tab:
        match_curves(p)
    with report_tab:
        report(p)
