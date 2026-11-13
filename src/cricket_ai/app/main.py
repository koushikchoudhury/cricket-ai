"""Entry point of the Streamlit app.

Run with ``cricket-ai app`` (or ``streamlit run src/cricket_ai/app/main.py``).
"""

from __future__ import annotations

import streamlit as st

from cricket_ai.app import agent, analytics, ask, overview, state, synthetic, winprob
from cricket_ai.llm import DEFAULT_MODELS, PROVIDERS, llm_settings

PAGES = ["Overview", "Analytics", "Win probability", "Synthetic data", "Ask", "Agent"]


def main() -> None:
    st.set_page_config(page_title="Cricket AI", page_icon=":cricket_bat_and_ball:", layout="wide")

    with st.sidebar:
        st.header("Cricket AI")
        page = st.radio("Page", PAGES, label_visibility="collapsed")
        available = state.datasets()
        dataset = st.selectbox(
            "Dataset",
            available or ["ipl"],
            index=(available.index("ipl") if "ipl" in available else 0),
        )
        with st.expander("Language model", expanded=page in ("Ask", "Agent")):
            default_provider, default_model = llm_settings()
            provider = st.selectbox(
                "Provider",
                PROVIDERS,
                index=PROVIDERS.index(default_provider),
                help="Used by the Ask and Agent pages.",
            )
            model = None
            if provider != "none":
                fallback = default_model if provider == default_provider else None
                model = st.text_input("Model", fallback or DEFAULT_MODELS[provider]).strip() or None

    p = state.paths(dataset)
    if not available:
        st.title("Cricket AI")
        state.need("The match data", "cricket-ai data download ipl\ncricket-ai data build ipl")
        return

    if page == "Overview":
        overview.render(p)
    elif page == "Analytics":
        analytics.render(p)
    elif page == "Win probability":
        winprob.render(p)
    elif page == "Synthetic data":
        synthetic.render(p)
    elif page == "Ask":
        ask.render(p, provider, model)
    else:
        agent.render(p, provider, model)


main()
