"""Shared, cached access to the data, models and indexes the pages need.

Loaders are keyed by file path and modification time, so rebuilding a table or retraining a
model is picked up on the next rerun without restarting the app.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from cricket_ai import config


@dataclass(frozen=True)
class Paths:
    dataset: str
    processed: Path
    models: Path
    reports: Path

    @property
    def winprob_model(self) -> Path:
        return self.models / "winprob_mlp.pt"

    @property
    def gan(self) -> Path:
        return self.models / "synth_gan"

    @property
    def rag_index(self) -> Path:
        return self.processed / "rag"


def paths(dataset: str) -> Paths:
    return Paths(
        dataset,
        config.processed_dir(dataset),
        config.MODELS_DIR / dataset,
        config.REPORTS_DIR / dataset,
    )


def datasets() -> list[str]:
    root = config.PROCESSED_DIR
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "matches.parquet").exists())


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def season_key(season: str) -> int:
    return int(str(season)[:4])


def need(what: str, command: str) -> None:
    """A friendly 'not built yet' message with the command that builds it."""
    st.info(f"{what} is not available yet. Build it with:\n\n```bash\n{command}\n```")


# ----- cached loaders -----
@st.cache_data(show_spinner="Loading data ...")
def _parquet(path: str, _mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def parquet(path: Path) -> pd.DataFrame | None:
    return _parquet(str(path), mtime(path)) if path.exists() else None


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource(show_spinner="Loading win-probability model ...")
def _winprob(path: str, _mtime: float):
    from cricket_ai.winprob.model import load

    return load(Path(path))


def winprob_model(p: Paths):
    if not (p.winprob_model.exists() and has("torch")):
        return None
    return _winprob(str(p.winprob_model), mtime(p.winprob_model))


def latest_par_rpo(p: Paths) -> float | None:
    states = parquet(p.processed / "chase_states.parquet")
    if states is None or "par_score" not in states:
        return None
    latest = states[states.start_date == states.start_date.max()]
    return float((latest.par_score / 20).mean())


@st.cache_resource(show_spinner="Loading GAN ...")
def _gan(directory: str, _mtime: float):
    from cricket_ai.synthetic.gan import TabularGAN

    return TabularGAN.load(Path(directory))


def gan(p: Paths):
    weights = p.gan / "generator.pt"
    if not (weights.exists() and has("torch")):
        return None
    return _gan(str(p.gan), mtime(weights))


@st.cache_resource(show_spinner="Loading search index ...")
def _retriever(directory: str, _mtime: float):
    from cricket_ai.rag.retrieval import HybridRetriever

    return HybridRetriever.load(Path(directory))


def retriever(p: Paths):
    meta = p.rag_index / "index.json"
    return _retriever(str(p.rag_index), mtime(meta)) if meta.exists() else None


@st.cache_resource(show_spinner="Loading SQL tables ...")
def _sql(processed: str, _mtime: float):
    from cricket_ai.rag.sql import CricketSQL

    return CricketSQL.from_dir(Path(processed))


def sql(p: Paths):
    matches = p.processed / "matches.parquet"
    return _sql(str(p.processed), mtime(matches)) if matches.exists() else None


@st.cache_resource(show_spinner="Connecting to the language model ...")
def chat_model(provider: str, model: str | None):
    from cricket_ai.llm import get_chat_model

    return get_chat_model(provider, model)


def llm_error(exc: Exception, provider: str) -> None:
    hint = {
        "ollama": "Is Ollama running, and is the model pulled (`ollama pull <model>`)?",
        "anthropic": "Is ANTHROPIC_API_KEY set and `langchain-anthropic` installed?",
        "openai": "Is OPENAI_API_KEY set and `langchain-openai` installed?",
    }.get(provider, "")
    st.error(f"The language model call failed: {type(exc).__name__}: {exc}\n\n{hint}")
