"""Entry point of the Streamlit app.

Run with ``cricket-ai app`` (or ``streamlit run src/cricket_ai/app/main.py``).
"""

from __future__ import annotations

import os

import streamlit as st

from cricket_ai.app import agent, analytics, ask, cloud, overview, state, synthetic, winprob
from cricket_ai.llm import DEFAULT_MODELS, llm_settings

PAGES = ["Overview", "Analytics", "Win probability", "Synthetic data", "Ask", "Agent"]


def choose_model() -> tuple[str, str | None]:
    """Offer only the language models that can be used right now."""
    found = state.providers()
    options = list(found)
    wanted, wanted_model = llm_settings()
    if os.environ.get("CRICKET_AI_LLM") and wanted in found:
        default = wanted  # an explicit choice wins
    else:
        default = next((name for name in options if name != "none"), "none")
    provider = st.selectbox(
        "Provider", options, index=options.index(default), help="Used by the Ask and Agent pages."
    )
    if len(options) == 1:
        st.caption(
            "No language model found. Start Ollama and pull a model "
            "(`ollama pull qwen2.5:3b`), or set ANTHROPIC_API_KEY / OPENAI_API_KEY."
        )
    if provider == "none":
        return provider, None
    models = found[provider]
    if provider == "ollama":
        preferred = wanted_model if wanted_model in models else DEFAULT_MODELS["ollama"]
        index = models.index(preferred) if preferred in models else 0
        return provider, st.selectbox("Model", models, index=index, help="Installed in Ollama.")
    fallback = wanted_model if provider == wanted else None
    return provider, st.text_input("Model", fallback or models[0]).strip() or None


def main() -> None:
    st.set_page_config(page_title="Cricket AI", page_icon="🏏", layout="wide")
    cloud.apply_secrets()
    try:
        cloud.ensure_data()  # hosted: download the data bundle on first start
    except Exception as exc:  # noqa: BLE001 - show the problem on the page
        st.error(f"Could not download the data bundle: {type(exc).__name__}: {exc}")

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
            provider, model = choose_model()
        if page in ("Ask", "Agent") and provider != "none":
            cloud.sidebar_budget()
        st.caption(
            "Data: [Cricsheet](https://cricsheet.org), Open Data Commons Attribution License. "
            "Code: [GitHub](https://github.com/koushikchoudhury/cricket-ai)."
        )

    p = state.paths(dataset)
    if not available:
        st.title("Cricket AI")
        state.need("The match data", "cricket-ai data download ipl\ncricket-ai data build ipl")
        return

    if page in ("Ask", "Agent"):
        state.warm_up(provider, model)  # background: the page renders without waiting

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
