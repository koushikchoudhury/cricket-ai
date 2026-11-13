"""Ask page: retrieval-augmented question answering with sources and SQL."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cricket_ai.app import state

EXAMPLES = [
    "Who won the IPL final in 2016?",
    "How many runs did V Kohli score in the 2016 season?",
    "Who was player of the match when Mumbai Indians played Chennai Super Kings on 26 April 2019?",
]


def render(p: state.Paths, provider: str, model: str | None) -> None:
    st.title("Ask")
    st.write(
        "Questions are answered from retrieved match reports and season summaries. Anything "
        "that needs counting goes to SQL the model writes, run on a read-only database."
    )
    retriever = state.retriever(p)
    if retriever is None:
        state.need("The search index", f"cricket-ai rag index {p.dataset} --embedder st")
        return

    from cricket_ai.rag.pipeline import CricketRAG

    if provider == "none":
        st.caption(
            "No language model selected in the sidebar: you will get the retrieved "
            "sources without a written answer."
        )

    st.caption("Try one:")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state["ask_question"] = example

    with st.form("ask"):
        question = st.text_input("Your question", key="ask_question")
        c1, c2 = st.columns([1, 3])
        route = c1.selectbox(
            "Route", ["auto", "search", "sql"], help="auto sends counting questions to SQL."
        )
        k = c2.slider("Sources to retrieve", 1, 10, 5)
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        try:
            rag = CricketRAG(retriever, state.sql(p), state.chat_model(provider, model), k=k)
            with st.spinner("Thinking ..."):
                st.session_state["ask_answer"] = rag.ask(
                    question.strip(), route=None if route == "auto" else route
                )
        except Exception as exc:  # noqa: BLE001 - show the failure instead of a traceback
            st.session_state.pop("ask_answer", None)
            state.llm_error(exc, provider)

    answer = st.session_state.get("ask_answer")
    if answer is None:
        return

    st.subheader("Answer")
    if provider == "none" or answer.text.startswith("No LLM configured"):
        st.info("Retrieved sources are listed below.")
    else:
        st.markdown(answer.text)
    st.caption(f"Route: {answer.route} · {answer.seconds}s")

    if answer.sql:
        st.subheader("SQL")
        st.code(answer.sql, language="sql")
        if answer.error:
            st.warning(f"The query failed: {answer.error}")
        elif answer.rows is not None:
            st.dataframe(pd.DataFrame(answer.rows), hide_index=True)

    st.subheader("Sources")
    for source in answer.sources:
        doc = retriever.docs[retriever.index[source["doc_id"]]]
        with st.expander(f"[{source['n']}] {source['title']}"):
            st.write(doc.text)
            st.caption(source["doc_id"])
