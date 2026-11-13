"""Overview page: what the project is and what is built on this machine."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cricket_ai.app import state


def module_status(p: state.Paths) -> pd.DataFrame:
    ds = p.dataset
    checks = [
        (
            "1. Data pipeline",
            (p.processed / "deliveries.parquet").exists(),
            f"cricket-ai data download {ds} && cricket-ai data build {ds}",
        ),
        (
            "1. Win-probability model",
            p.winprob_model.exists(),
            f"cricket-ai data features {ds} && cricket-ai winprob train {ds}",
        ),
        (
            "2. Synthetic data (GAN)",
            (p.gan / "generator.pt").exists(),
            f"cricket-ai synth table {ds} && cricket-ai synth train {ds} --model gan",
        ),
        (
            "4. Search index (RAG)",
            (p.rag_index / "index.json").exists(),
            f"cricket-ai rag index {ds} --embedder st",
        ),
        (
            "5. Agent",
            (p.rag_index / "index.json").exists(),
            "needs the search index and an LLM (see the sidebar)",
        ),
    ]
    return pd.DataFrame(
        [
            {"Module": n, "Status": "Ready" if ok else "Not built", "Build with": "" if ok else cmd}
            for n, ok, cmd in checks
        ]
    )


def render(p: state.Paths) -> None:
    st.title("Cricket AI")
    st.write(
        "An end-to-end applied AI project on ball-by-ball cricket data: analytics and win "
        "probability, synthetic data from a hand-written GAN, question answering with "
        "retrieval and SQL, and an agent that ties them together."
    )

    matches = state.parquet(p.processed / "matches.parquet")
    deliveries = state.parquet(p.processed / "deliveries.parquet")
    index = state.read_json(p.rag_index / "index.json")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(matches):,}" if matches is not None else "-")
    c2.metric("Deliveries", f"{len(deliveries):,}" if deliveries is not None else "-")
    c3.metric("Seasons", matches.season.nunique() if matches is not None else "-")
    c4.metric("Searchable documents", f"{index['n_docs']:,}" if index else "-")

    st.subheader("What is built on this machine")
    st.dataframe(module_status(p), hide_index=True)

    st.subheader("Pages")
    st.markdown(
        "- **Analytics**: batting, bowling, phase and team tables by season.\n"
        "- **Win probability**: a chase calculator and the win-probability curve of any match.\n"
        "- **Synthetic data**: generate overs with the GAN and compare them with real ones.\n"
        "- **Ask**: questions answered from retrieved match reports, with SQL for numbers.\n"
        "- **Agent**: a chat that chooses and calls the tools above."
    )
