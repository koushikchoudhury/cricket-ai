"""Synthetic-data page: generate overs with the GAN and compare them with real overs."""

from __future__ import annotations

import time

import altair as alt
import pandas as pd
import streamlit as st

from cricket_ai.app import state
from cricket_ai.synthetic.table import COLUMNS, validity

RUN_CAP = 24  # overs above this are grouped into one bar


def evaluation(p: state.Paths) -> None:
    metrics = state.read_json(p.reports / "synthetic" / "metrics.json")
    if not metrics:
        state.need("The evaluation report", f"cricket-ai synth evaluate {p.dataset}")
        return
    cols = [
        "model",
        "mean_tvd",
        "corr_diff",
        "validity",
        "runs_mae",
        "wicket_auc",
        "dcr_ratio",
        "membership_auc",
    ]
    names = {
        "model": "Model",
        "mean_tvd": "Column distance (TVD)",
        "corr_diff": "Correlation error",
        "validity": "Valid cricket",
        "runs_mae": "Runs MAE",
        "wicket_auc": "Wicket AUC",
        "dcr_ratio": "Distance ratio",
        "membership_auc": "Membership AUC",
    }
    table = pd.DataFrame(metrics["summary"]).reindex(columns=cols).rename(columns=names)
    numeric = [c for c in table.columns if c != "Model"]
    values = table[numeric].apply(pd.to_numeric, errors="coerce")
    table[numeric] = values.map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    st.dataframe(table, hide_index=True)
    st.caption(
        "Lower is better for TVD, correlation error and runs MAE; higher for validity and "
        "wicket AUC. Membership AUC near 0.5 means the training data cannot be picked out. "
        "The real (TRTR) row is a model trained on real data, the target to match."
    )


def compare_runs(real: pd.DataFrame, synth: pd.DataFrame) -> alt.Chart:
    def share(df: pd.DataFrame, label: str) -> pd.DataFrame:
        s = df.runs.clip(upper=RUN_CAP).value_counts(normalize=True).sort_index()
        return pd.DataFrame({"runs": s.index, "share": s.values * 100, "data": label})

    both = pd.concat([share(real, "Real"), share(synth, "Synthetic")])
    return (
        alt.Chart(both)
        .mark_bar()
        .encode(
            x=alt.X(
                "runs:O",
                title=f"Runs in the over ({RUN_CAP} = {RUN_CAP} or more)",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset="data:N",
            y=alt.Y("share:Q", title="Share of overs (%)"),
            color=alt.Color("data:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=["data", "runs", alt.Tooltip("share:Q", format=".1f")],
        )
        .properties(height=300)
    )


def compare_by_over(real: pd.DataFrame, synth: pd.DataFrame) -> alt.Chart:
    def mean(df: pd.DataFrame, label: str) -> pd.DataFrame:
        s = df.groupby("over").runs.mean()
        return pd.DataFrame({"over": s.index + 1, "runs": s.values, "data": label})

    both = pd.concat([mean(real, "Real"), mean(synth, "Synthetic")])
    return (
        alt.Chart(both)
        .mark_line(point=True)
        .encode(
            x=alt.X("over:O", title="Over of the innings", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("runs:Q", title="Average runs", scale=alt.Scale(zero=False)),
            color=alt.Color("data:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=["data", "over", alt.Tooltip("runs:Q", format=".2f")],
        )
        .properties(height=300)
    )


def render(p: state.Paths) -> None:
    st.title("Synthetic data")
    st.write(
        "A conditional GAN, written from scratch in PyTorch, learned to generate overs of T20 "
        "cricket: the state at the start of the over and what happened in it."
    )
    real = state.parquet(p.processed / "overs.parquet")
    if real is None:
        state.need("The over-level table", f"cricket-ai synth table {p.dataset}")
        return

    st.subheader("How good is it?")
    evaluation(p)

    st.subheader("Generate overs")
    model = state.gan(p)
    saved = p.processed / "synthetic" / "gan.parquet"
    if model is not None:
        c1, c2 = st.columns([3, 1])
        n = c1.slider("How many overs", 200, 5000, 1000, step=100)
        if c2.button("Generate", type="primary"):
            start = time.perf_counter()
            st.session_state["synthetic"] = model.sample(int(n), seed=int(time.time()))
            st.session_state["synthetic_note"] = (
                f"Generated {n:,} overs in {time.perf_counter() - start:.1f}s."
            )
    elif saved.exists():
        st.caption("PyTorch or the trained GAN is not available here; showing the saved sample.")
        st.session_state.setdefault("synthetic", state.parquet(saved))
        st.session_state.setdefault("synthetic_note", f"Loaded {saved.name}.")
    else:
        state.need("The trained GAN", f"cricket-ai synth train {p.dataset} --model gan")
        return

    synth = st.session_state.get("synthetic")
    if synth is None:
        st.caption("Press Generate to sample new overs from the GAN.")
        return

    st.caption(st.session_state.get("synthetic_note", ""))
    rules = validity(synth)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Valid cricket",
        f"{rules['all_rules']:.1%}",
        help="Share of overs obeying all 8 cricket rules. Real overs: 100%.",
    )
    c2.metric(
        "Average runs per over",
        f"{synth.runs.mean():.2f}",
        delta=f"{synth.runs.mean() - real.runs.mean():+.2f} vs real",
        delta_color="off",
    )
    c3.metric(
        "Overs with a wicket",
        f"{(synth.wickets > 0).mean():.1%}",
        delta=f"{((synth.wickets > 0).mean() - (real.wickets > 0).mean()) * 100:+.1f} pts vs real",
        delta_color="off",
    )
    c4.metric(
        "Boundaries per over",
        f"{(synth.fours + synth.sixes).mean():.2f}",
        delta=f"{(synth.fours + synth.sixes).mean() - (real.fours + real.sixes).mean():+.2f}"
        " vs real",
        delta_color="off",
    )

    left, right = st.columns(2)
    left.altair_chart(compare_runs(real, synth))
    right.altair_chart(compare_by_over(real, synth))

    with st.expander("Which rules do the invalid overs break?"):
        broken = pd.DataFrame(
            [
                {"rule": k, "overs passing (%)": round(v * 100, 1)}
                for k, v in rules.items()
                if k != "all_rules"
            ]
        )
        st.dataframe(broken, hide_index=True)
    with st.expander("Sample of generated overs"):
        st.dataframe(synth[COLUMNS].head(100), hide_index=True)
    st.download_button(
        "Download as CSV",
        synth[COLUMNS].to_csv(index=False),
        file_name="synthetic_overs.csv",
        mime="text/csv",
    )
