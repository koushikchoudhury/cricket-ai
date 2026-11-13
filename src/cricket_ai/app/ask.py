"""Ask page: a chat over retrieved match reports, with SQL for numbers.

Follow-ups work: "what about Kohli?" after a question about Rohit Sharma in 2016 is
rewritten into a standalone question before searching, and the chat shows how it was
understood.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cricket_ai.app import state

EXAMPLES = [
    "Who won the IPL final in 2016?",
    "How many runs did V Kohli score in the 2016 season?",
    "Who was player of the match when Mumbai Indians played Chennai Super Kings on 26 April 2019?",
]
HISTORY_KEY = "ask_history"


def show_details(turn: dict, retriever) -> None:
    """Under an answer: how the question was understood, the SQL and the sources."""
    if turn.get("standalone") and turn["standalone"] != turn.get("asked"):
        st.caption(f"Understood as: *{turn['standalone']}*")
    if turn.get("sql"):
        with st.expander("SQL" + (" (failed)" if turn.get("error") else "")):
            st.code(turn["sql"], language="sql")
            if turn.get("error"):
                st.warning(turn["error"])
            elif turn.get("rows") is not None:
                st.dataframe(pd.DataFrame(turn["rows"]), hide_index=True)
    sources = turn.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                doc = retriever.docs[retriever.index[source["doc_id"]]]
                st.markdown(f"**[{source['n']}] {source['title']}**")
                st.caption(doc.text)
    st.caption(f"Route: {turn['route']} · {turn['seconds']}s")


def render(p: state.Paths, provider: str, model: str | None) -> None:
    st.title("Ask")
    st.write(
        "Ask about matches, players and seasons. Answers come from retrieved match reports and "
        "season summaries; anything that needs counting goes to SQL on a read-only database. "
        "Follow-up questions keep the context."
    )
    retriever = state.retriever(p)
    if retriever is None:
        state.need("The search index", f"cricket-ai rag index {p.dataset} --embedder st")
        return

    from cricket_ai.rag.pipeline import CricketRAG

    with st.sidebar.expander("Answer settings"):
        route = st.selectbox(
            "Route", ["auto", "search", "sql"], help="auto sends counting questions to SQL."
        )
        k = st.slider("Sources to retrieve", 1, 10, 5)
    history = st.session_state.setdefault(HISTORY_KEY, [])
    if st.sidebar.button("Clear chat", key="ask_clear"):
        history.clear()

    if provider == "none":
        st.info(
            "No language model is selected, so you will get the retrieved sources without a "
            "written answer, and follow-ups cannot be understood. Pick one under "
            "**Language model** in the sidebar."
        )

    if not history:
        st.caption("Try one:")
        for example in EXAMPLES:
            if st.button(example, key=f"ask_example_{example}"):
                st.session_state["ask_pending"] = example

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                show_details(turn, retriever)

    question = st.chat_input("Ask about matches, players or seasons")
    question = question or st.session_state.pop("ask_pending", None)
    if not question:
        return

    earlier = [{"role": t["role"], "content": t["content"]} for t in history]
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            rag = CricketRAG(retriever, state.sql(p), state.chat_model(provider, model), k=k)
            with st.spinner("Thinking ..."):
                answer = rag.ask(
                    question, route=None if route == "auto" else route, history=earlier
                )
        except Exception as exc:  # noqa: BLE001 - show the failure instead of a traceback
            history.pop()
            state.llm_error(exc, provider)
            return
        if answer.text.startswith("No LLM configured"):
            text = "Here are the most relevant sources (no language model selected)."
        else:
            text = answer.text
        turn = {
            "role": "assistant",
            "content": text,
            "asked": question,
            "standalone": answer.standalone,
            "route": answer.route,
            "seconds": answer.seconds,
            "sql": answer.sql,
            "rows": answer.rows,
            "error": answer.error,
            "sources": answer.sources,
        }
        st.markdown(text)
        show_details(turn, retriever)
    history.append(turn)
