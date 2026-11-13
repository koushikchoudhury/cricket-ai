"""Settings for running the app on a host such as Streamlit Community Cloud.

* Secrets (``.streamlit/secrets.toml`` locally, or the host's secrets box) are copied into
  environment variables, so the rest of the code reads one place.
* The data bundle is downloaded on first start when the data is not on disk.
* A question budget protects a paid API key: a shared limit per day across all visitors and
  a limit per visitor session. With no limits set (e.g. on your laptop) nothing is counted.
"""

from __future__ import annotations

import datetime as dt
import os
import threading

import streamlit as st

SECRET_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "CRICKET_AI_LLM",
    "CRICKET_AI_LLM_MODEL",
    "CRICKET_AI_BUNDLE_URL",
    "CRICKET_AI_DAILY_LIMIT",
    "CRICKET_AI_SESSION_LIMIT",
)
SESSION_KEY = "llm_questions_used"


def apply_secrets() -> None:
    """Copy known Streamlit secrets into the environment (existing variables win)."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file is the normal local case
        return
    for key in SECRET_KEYS:
        if key in secrets and not os.environ.get(key):
            os.environ[key] = str(secrets[key])


@st.cache_resource(show_spinner="Downloading the data and models (first start only) ...")
def ensure_data() -> str:
    from cricket_ai.deploy import ensure_bundle

    return ensure_bundle()


# ----- question budget -----
def _limit(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value.isdigit() else None


@st.cache_resource
def _ledger() -> dict:
    """Shared by every visitor of this app process."""
    return {"lock": threading.Lock(), "day": dt.date.today(), "used": 0}


def _today(ledger: dict) -> None:
    if ledger["day"] != dt.date.today():
        ledger["day"], ledger["used"] = dt.date.today(), 0


def budget_status() -> dict:
    """Limits and what is left; ``None`` means unlimited."""
    daily, per_session = _limit("CRICKET_AI_DAILY_LIMIT"), _limit("CRICKET_AI_SESSION_LIMIT")
    ledger = _ledger()
    with ledger["lock"]:
        _today(ledger)
        used_today = ledger["used"]
    used_here = st.session_state.get(SESSION_KEY, 0)
    return {
        "daily_left": None if daily is None else max(daily - used_today, 0),
        "session_left": None if per_session is None else max(per_session - used_here, 0),
    }


def take_question() -> str | None:
    """Reserve one language-model question. Returns why not, or None if allowed."""
    daily, per_session = _limit("CRICKET_AI_DAILY_LIMIT"), _limit("CRICKET_AI_SESSION_LIMIT")
    used_here = st.session_state.get(SESSION_KEY, 0)
    if per_session is not None and used_here >= per_session:
        return (
            f"You have used your {per_session} questions for this visit. Answers now show the "
            "retrieved sources only."
        )
    ledger = _ledger()
    with ledger["lock"]:
        _today(ledger)
        if daily is not None and ledger["used"] >= daily:
            return (
                f"Today's limit of {daily} questions for this demo is used up. Answers show the "
                "retrieved sources only until tomorrow."
            )
        ledger["used"] += 1
    st.session_state[SESSION_KEY] = used_here + 1
    return None


def sidebar_budget() -> None:
    status = budget_status()
    parts = []
    if status["session_left"] is not None:
        parts.append(f"{status['session_left']} for you")
    if status["daily_left"] is not None:
        parts.append(f"{status['daily_left']} today for everyone")
    if parts:
        st.sidebar.caption("Questions left: " + ", ".join(parts))


def is_hosted() -> bool:
    """True when running with a bundle URL or limits, i.e. as a shared demo."""
    return bool(
        os.environ.get("CRICKET_AI_BUNDLE_URL")
        or _limit("CRICKET_AI_DAILY_LIMIT") is not None
        or _limit("CRICKET_AI_SESSION_LIMIT") is not None
    )
