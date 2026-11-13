"""Agent page: chat with the LangGraph agent and see which tools it used."""

from __future__ import annotations

import streamlit as st

from cricket_ai.app import cloud, state

EXAMPLES = [
    "Who scored the most runs in the 2016 season?",
    "A team chasing 180 is 100/2 after 12 overs. What is their chance of winning?",
    "How did JJ Bumrah bowl in the 2020 season?",
]


@st.cache_resource(show_spinner="Starting the agent ...")
def build_agent(dataset: str, provider: str, model: str | None, _stamp: float):
    from cricket_ai.agent.graph import CricketAgent
    from cricket_ai.agent.tools import CricketContext

    p = state.paths(dataset)
    ctx = CricketContext(
        retriever=state.retriever(p),
        sql=state.sql(p),
        winprob_model=state.winprob_model(p),
        latest_par_rpo=state.latest_par_rpo(p),
        gan=state.gan(p),
    )
    return CricketAgent(state.chat_model(provider, model), ctx)


def show_tools(calls: list[dict]) -> None:
    if not calls:
        return
    with st.expander(f"Tools used: {', '.join(c['name'] for c in calls)}"):
        for call in calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            status = "" if call["ok"] else " (failed)"
            st.markdown(f"**{call['name']}**({args}) · {call['seconds']}s{status}")
            st.code(call["output"], language=None)


def show_timing(turn: dict) -> None:
    if not turn.get("timing"):
        return
    st.caption(f"{turn['seconds']}s: {turn['timing']}")
    loading = sum(c.get("load_s") or 0 for c in turn.get("llm_calls", []))
    if loading >= 2:
        st.caption(
            f"{loading:.1f}s of that was Ollama loading the model into memory. It now stays "
            "loaded for 30 minutes, so the next answers skip this."
        )


def describe(event: dict) -> str | None:
    if event["type"] == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
        return f"Calling **{event['name']}**({args})"
    if event["type"] == "writing":
        return "Writing the answer ..."
    return None


def render(p: state.Paths, provider: str, model: str | None) -> None:
    st.title("Agent")
    st.write(
        "The agent decides which part of the project answers your question: search, player "
        "statistics, leaderboards, SQL, the win-probability model or the GAN."
    )
    if provider == "none":
        st.info(
            "The agent needs a language model. Choose one in the sidebar, for example "
            "**ollama** with `qwen2.5:3b` (run `ollama pull qwen2.5:3b` once)."
        )
        return
    if not state.has("langgraph"):
        st.info('The agent needs LangGraph: `pip install -e ".[agent]"`.')
        return
    if state.retriever(p) is None and state.sql(p) is None:
        state.need(
            "The data and search index",
            f"cricket-ai data build {p.dataset}\ncricket-ai rag index {p.dataset}",
        )
        return

    try:
        agent = build_agent(p.dataset, provider, model, state.mtime(p.rag_index / "index.json"))
    except Exception as exc:  # noqa: BLE001
        state.llm_error(exc, provider)
        return

    allow_gan = st.sidebar.checkbox(
        "Let the agent run the GAN",
        value=True,
        help="Generating synthetic overs is the slow tool; untick to make the agent ask "
        "without it.",
    )
    agent.approver = lambda name, args: allow_gan
    st.caption("Tools: " + ", ".join(t.name for t in agent.tools))

    history = st.session_state.setdefault("agent_history", [])
    if st.sidebar.button("Clear chat"):
        history.clear()

    if not history:
        st.caption("Try one:")
        for example in EXAMPLES:
            if st.button(example):
                st.session_state["agent_pending"] = example

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            show_tools(turn.get("tools", []))
            show_timing(turn)

    question = st.chat_input("Ask about matches, players, chases or synthetic overs")
    question = question or st.session_state.pop("agent_pending", None)
    if not question:
        return

    refused = cloud.take_question()
    if refused:
        st.info(refused + " The Ask page still answers from the sources.")
        return

    earlier = [{"role": t["role"], "content": t["content"]} for t in history]
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.status("Choosing a tool ...", expanded=True) as status:

                def progress(event: dict) -> None:
                    line = describe(event)
                    if line:
                        status.write(line)
                        status.update(label=line.replace("**", ""))

                result = agent.ask(question, history=earlier, on_event=progress)
                status.update(label=f"Done in {result.seconds}s", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            state.llm_error(exc, provider)
            history.pop()
            return
        turn = {
            "role": "assistant",
            "content": result.answer,
            "tools": result.tool_calls,
            "seconds": result.seconds,
            "timing": result.timing,
            "llm_calls": result.llm_calls,
        }
        st.markdown(result.answer)
        show_tools(result.tool_calls)
        show_timing(turn)
    history.append(turn)
