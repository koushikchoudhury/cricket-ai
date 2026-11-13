"""Headless tests of the Streamlit app: every page renders and the main flows work."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("duckdb")
pytest.importorskip("langgraph")

import streamlit as st  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from matchgen import generate_match  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from cricket_ai import config  # noqa: E402
from cricket_ai.app import state  # noqa: E402
from cricket_ai.data.build import build_tables, write_tables  # noqa: E402
from cricket_ai.data.features import chase_states  # noqa: E402
from cricket_ai.rag.corpus import build_corpus  # noqa: E402
from cricket_ai.rag.retrieval import HashingEmbedder, HybridRetriever  # noqa: E402
from cricket_ai.synthetic.baselines import IndependentMarginals  # noqa: E402
from cricket_ai.synthetic.evaluate import evaluate_all  # noqa: E402
from cricket_ai.synthetic.table import DISCRETE, build_overs, split_by_match  # noqa: E402

MAIN = str(Path(state.__file__).parent / "main.py")
PAGES = ["Overview", "Analytics", "Win probability", "Synthetic data", "Ask", "Agent"]
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A small but complete project: tables, features, overs, index, reports, models."""
    root = tmp_path_factory.mktemp("project")
    raw = root / "raw"
    raw.mkdir()
    for seed in range(60):
        (raw / f"g{seed:03d}.json").write_text(json.dumps(generate_match(seed, day_step=9)))
    processed = root / "data" / "processed" / "ipl"
    built = build_tables(raw, workers=1)
    write_tables(built, processed)
    states = chase_states(built.matches, built.deliveries)
    states.to_parquet(processed / "chase_states.parquet", index=False)
    overs = build_overs(built.matches, built.deliveries)
    overs.to_parquet(processed / "overs.parquet", index=False)
    HybridRetriever(build_corpus(built.matches, built.deliveries), HashingEmbedder()).save(
        processed / "rag"
    )
    train, holdout = split_by_match(overs)
    sample = IndependentMarginals().fit(train, DISCRETE).sample(len(train))
    (processed / "synthetic").mkdir()
    sample.to_parquet(processed / "synthetic" / "gan.parquet", index=False)
    evaluate_all(train, holdout, {"gan": sample}, root / "reports" / "ipl" / "synthetic")

    if HAS_TORCH:
        from cricket_ai.synthetic.gan import GANConfig, TabularGAN
        from cricket_ai.winprob import model as mlp
        from cricket_ai.winprob.data import date_split
        from cricket_ai.winprob.evaluate import run_experiment

        models = root / "models" / "ipl"
        cfg = mlp.TrainConfig(max_epochs=3, device="cpu")
        run_experiment(
            states, root / "reports" / "ipl" / "winprob", models, cfg, split=date_split(states)
        )
        gan = TabularGAN(GANConfig(epochs=2, batch_size=200, device="cpu")).fit(train, DISCRETE)
        gan.save(models / "synth_gan")
    return root


@pytest.fixture
def app(project, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", project / "data")
    monkeypatch.setattr(config, "PROCESSED_DIR", project / "data" / "processed")
    monkeypatch.setattr(config, "REPORTS_DIR", project / "reports")
    monkeypatch.setattr(config, "MODELS_DIR", project / "models")
    monkeypatch.delenv("CRICKET_AI_LLM", raising=False)
    monkeypatch.setattr(state, "providers", lambda: {"none": []})
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(MAIN, default_timeout=180)


OLLAMA = {"none": [], "ollama": ["llama3.2:1b", "qwen2.5:3b"]}


def open_page(app: AppTest, page: str, provider: str | None = None) -> AppTest:
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    if provider:
        next(s for s in app.sidebar.selectbox if s.label == "Provider").set_value(provider).run()
    return app


def button(app: AppTest, label: str):
    return next(b for b in app.button if b.label == label)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    at = open_page(app, page)
    assert not at.exception, at.exception
    assert at.title[0].value == ("Cricket AI" if page == "Overview" else page)


def test_overview_reports_what_is_built(app):
    at = open_page(app, "Overview")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Matches"] == "60" and metrics["Seasons"] == "2"
    status = at.dataframe[0].value.set_index("Module").Status
    assert status["1. Data pipeline"] == "Ready" and status["4. Search index (RAG)"] == "Ready"


def test_app_without_data_explains_how_to_build(app, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "nothing")
    at = app.run()
    assert not at.exception
    assert "cricket-ai data build ipl" in at.info[0].value


def test_analytics_filters_by_season(app):
    at = open_page(app, "Analytics")
    all_runs = at.dataframe[0].value.runs.sum()
    season = next(s for s in at.selectbox if s.label == "Season")
    season.set_value(season.options[1]).run()
    assert not at.exception
    assert 0 < at.dataframe[0].value.runs.sum() < all_runs


def test_ask_returns_sources_without_llm(app):
    at = open_page(app, "Ask")
    at.text_input(key="ask_question").set_value("Who won the match on 1 April 2020?")
    button(at, "Ask").click().run()
    assert not at.exception
    assert any("Retrieved sources" in i.value for i in at.info)
    assert len(at.expander) >= 5  # language-model expander + 5 sources


def test_ask_with_llm_shows_answer_and_sql(app, monkeypatch):
    replies = iter(
        [
            AIMessage("```sql\nSELECT COUNT(*) AS n FROM matches\n```"),
            AIMessage("There are 60 matches, according to the database."),
        ]
    )
    monkeypatch.setattr(
        state, "chat_model", lambda provider, model: GenericFakeChatModel(messages=replies)
    )
    monkeypatch.setattr(state, "providers", lambda: OLLAMA)
    at = open_page(app, "Ask", provider="ollama")
    at.text_input(key="ask_question").set_value("How many matches are there?")
    button(at, "Ask").click().run()
    assert not at.exception
    assert any("There are 60 matches" in m.value for m in at.markdown)
    assert "SELECT COUNT(*)" in at.code[0].value


def test_agent_needs_a_model_then_chats(app, monkeypatch):
    at = open_page(app, "Agent")
    assert "needs a language model" in at.info[0].value

    replies = iter(
        [
            AIMessage(
                "", tool_calls=[{"name": "leaderboard", "args": {"stat": "runs"}, "id": "c1"}]
            ),
            AIMessage("Kol_1 has scored the most runs (leaderboard)."),
        ]
    )
    monkeypatch.setattr(state, "chat_model", lambda provider, model: Scripted(messages=replies))
    monkeypatch.setattr(state, "providers", lambda: OLLAMA)
    at = open_page(at, "Agent", provider="ollama")
    at.chat_input[0].set_value("Who has the most runs?").run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    said = [m.value for m in at.chat_message[1].markdown]
    assert any("Kol_1 has scored the most runs" in text for text in said)
    assert any("Tools used: leaderboard" in e.label for e in at.expander)


def test_llm_failure_is_shown_not_raised(app, monkeypatch):
    class Broken(Scripted):
        def invoke(self, *args, **kwargs):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(state, "chat_model", lambda p, m: Broken(messages=iter([])))
    monkeypatch.setattr(state, "providers", lambda: OLLAMA)
    at = open_page(app, "Agent", provider="ollama")
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert "Is Ollama running" in at.error[0].value


@pytest.mark.skipif(not HAS_TORCH, reason="needs PyTorch")
def test_chase_calculator_responds_to_inputs(app):
    at = open_page(app, "Win probability")

    def probability() -> float:
        value = next(m.value for m in at.metric if m.label == "Win probability")
        return float(value.rstrip("%"))

    def set_input(label: str, value: int) -> None:
        next(n for n in at.number_input if n.label == label).set_value(value).run()

    start = probability()
    set_input("Runs scored", 170)  # 10 needed instead of 80
    easier = probability()
    set_input("Wickets lost", 9)
    assert not at.exception
    assert easier > start and probability() <= easier
    assert {m.label: m.value for m in at.metric}["Runs needed"] == "10"
    assert any("chasing" in s.value for s in at.subheader)  # match curve rendered


@pytest.mark.skipif(not HAS_TORCH, reason="needs PyTorch")
def test_gan_generates_on_demand(app):
    at = open_page(app, "Synthetic data")
    assert not any(m.label == "Valid cricket" for m in at.metric)
    button(at, "Generate").click().run()
    assert not at.exception
    assert any(m.label == "Valid cricket" for m in at.metric)
    assert any("Generated" in c.value for c in at.caption)


def test_page_icon_is_an_emoji_not_a_shortcode():
    # ':cricket_bat_and_ball:' was not a known shortcode, so Streamlit served it as a file
    # path, and on Windows a path with colons raises OSError on every page load.
    import re

    source = Path(MAIN).read_text(encoding="utf-8")
    icon = re.search(r'page_icon="([^"]*)"', source).group(1)
    assert len(icon) <= 2 and ":" not in icon


def sidebar_select(at: AppTest, label: str):
    return next((s for s in at.sidebar.selectbox if s.label == label), None)


def test_sidebar_offers_only_available_models(app, monkeypatch):
    at = open_page(app, "Agent")
    assert sidebar_select(at, "Provider").options == ["none"]
    assert any("No language model found" in c.value for c in at.sidebar.caption)
    assert sidebar_select(at, "Model") is None

    # Ollama running with two models pulled: a new session offers and selects it
    monkeypatch.setattr(state, "providers", lambda: OLLAMA)
    at = open_page(AppTest.from_file(MAIN, default_timeout=180), "Agent")
    provider = sidebar_select(at, "Provider")
    assert provider.options == ["none", "ollama"]  # no anthropic / openai without a key
    assert provider.value == "ollama"
    model = sidebar_select(at, "Model")
    assert model.options == ["llama3.2:1b", "qwen2.5:3b"] and model.value == "qwen2.5:3b"


def test_agent_shows_progress_and_timing(app, monkeypatch):
    usage = {
        "load_duration": 9_500_000_000,
        "prompt_eval_count": 1200,
        "prompt_eval_duration": 2_000_000_000,
        "eval_count": 40,
        "eval_duration": 1_500_000_000,
    }
    replies = iter(
        [
            AIMessage(
                "",
                tool_calls=[{"name": "leaderboard", "args": {"stat": "runs"}, "id": "c1"}],
                response_metadata=usage,
            ),
            AIMessage("Kol_1 leads.", response_metadata={**usage, "load_duration": 0}),
        ]
    )
    monkeypatch.setattr(state, "chat_model", lambda provider, model: Scripted(messages=replies))
    monkeypatch.setattr(state, "providers", lambda: OLLAMA)
    at = open_page(app, "Agent", provider="ollama")
    at.chat_input[0].set_value("Who has the most runs?").run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "loading 9.5s" in captions and "reading 2,400 tokens 4.0s" in captions
    assert "writing 80 tokens 3.0s" in captions
    assert "was Ollama loading the model into memory" in captions
    assert any("Calling **leaderboard**" in m.value for m in at.markdown)


def test_ollama_is_warmed_up_once_in_the_background(app, monkeypatch):
    from cricket_ai import llm

    monkeypatch.setattr(state, "providers", lambda: {"none": [], "ollama": ["qwen2.5:3b"]})
    calls = []
    monkeypatch.setattr(llm, "warm_up_ollama", lambda model, timeout=120.0: calls.append(model))
    at = open_page(app, "Agent")
    at = open_page(at, "Ask")
    assert not at.exception
    state._warm_up(next(iter(["qwen2.5:3b"]))).join(timeout=5)
    assert calls == ["qwen2.5:3b"]  # once per model, however many pages and reruns
    assert at.title[0].value == "Ask"  # the page still renders


@pytest.mark.parametrize("page", ["Overview", "Analytics", "Ask"])
def test_each_page_renders_only_itself(app, page):
    at = open_page(app, page)
    assert [t.value for t in at.title] == ["Cricket AI" if page == "Overview" else page]
