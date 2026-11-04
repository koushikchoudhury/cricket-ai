"""Question answering: route, retrieve, optionally run SQL, then answer with citations."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

import pandas as pd

from cricket_ai.llm import message_text
from cricket_ai.rag.retrieval import Hit, HybridRetriever
from cricket_ai.rag.sql import SCHEMA, CricketSQL

NUMERIC = re.compile(
    r"\b(how many|how much|most|least|fewest|highest|lowest|average|total|top|best|worst|"
    r"strike rate|economy|number of|count|record|more than|fewer than|per (?:match|innings))\b",
    re.IGNORECASE,
)

SQL_SYSTEM = (
    "You write a single DuckDB SQL SELECT query that answers a cricket question.\n"
    + SCHEMA
    + "\nReturn only the query inside a ```sql code block. Keep results small "
    "(aggregate, ORDER BY, LIMIT 10). Never modify data."
)

ANSWER_SYSTEM = (
    "You answer questions about cricket using only the context provided.\n"
    "- Numbered sources [1], [2], ... are retrieved documents; cite the ones you use.\n"
    "- If an SQL result is given, trust it for numbers and say it came from the database.\n"
    "- If the context does not contain the answer, say so plainly. Do not guess.\n"
    "- Be concise: one to three sentences."
)


@dataclass
class Answer:
    question: str
    route: str
    text: str
    sources: list[dict] = field(default_factory=list)
    sql: str | None = None
    rows: list[dict] | None = None
    error: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def route_question(question: str) -> str:
    return "sql" if NUMERIC.search(question) else "search"


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.doc.title}\n{h.doc.text}" for i, h in enumerate(hits, 1))


def format_rows(rows: pd.DataFrame, max_rows: int = 20) -> str:
    if rows is None or rows.empty:
        return "(no rows)"
    return rows.head(max_rows).to_markdown(index=False)


class CricketRAG:
    def __init__(
        self,
        retriever: HybridRetriever,
        sql: CricketSQL | None = None,
        llm=None,
        k: int = 5,
        method: str = "hybrid",
    ):
        self.retriever, self.sql, self.llm = retriever, sql, llm
        self.k, self.method = k, method

    def write_sql(
        self, question: str, error: str | None = None, previous: str | None = None
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = f"Question: {question}"
        if error:
            prompt += f"\n\nYour previous query:\n{previous}\nfailed with:\n{error}\nFix it."
        reply = self.llm.invoke([SystemMessage(SQL_SYSTEM), HumanMessage(prompt)])
        return extract_sql(message_text(reply))

    def run_sql(self, question: str) -> tuple[str | None, pd.DataFrame | None, str | None]:
        sql_text, rows, err = None, None, None
        for _ in range(2):  # one retry with the error message
            sql_text = self.write_sql(question, err, sql_text)
            try:
                return sql_text, self.sql.run(sql_text), None
            except Exception as exc:  # noqa: BLE001 - errors are fed back to the model
                err = f"{type(exc).__name__}: {exc}"
        return sql_text, rows, err

    def ask(self, question: str, route: str | None = None) -> Answer:
        start = time.perf_counter()
        route = route if route in ("sql", "search") else route_question(question)
        hits = self.retriever.search(question, k=self.k, method=self.method)
        sources = [
            {"n": i, "doc_id": h.doc.doc_id, "title": h.doc.title} for i, h in enumerate(hits, 1)
        ]
        answer = Answer(question=question, route=route, text="", sources=sources)

        rows = None
        if route == "sql" and self.llm is not None and self.sql is not None:
            answer.sql, rows, answer.error = self.run_sql(question)
            if rows is not None:
                answer.rows = rows.head(50).to_dict(orient="records")

        if self.llm is None:
            answer.text = (
                "No LLM configured (use --llm ollama|anthropic|openai for a written answer). "
                "Most relevant sources:\n"
                + "\n".join(
                    f"[{s['n']}] {h.doc.text[:300]}" for s, h in zip(sources, hits, strict=True)
                )
            )
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            parts = [f"Question: {question}", "Sources:\n" + format_context(hits)]
            if answer.sql:
                parts.append(f"SQL query:\n{answer.sql}\nSQL result:\n{format_rows(rows)}")
                if answer.error:
                    parts.append(f"The SQL query failed: {answer.error}")
            reply = self.llm.invoke(
                [SystemMessage(ANSWER_SYSTEM), HumanMessage("\n\n".join(parts))]
            )
            answer.text = message_text(reply).strip()
        answer.seconds = round(time.perf_counter() - start, 2)
        return answer
