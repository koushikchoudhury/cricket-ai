"""Evaluation with questions whose answers are known.

Questions are generated from the tables themselves, so every one has a ground-truth answer
and a ground-truth source document. Five types:

* ``player_of_match``  - who was player of the match in a given game       (search)
* ``match_winner``     - who won a given game                               (search)
* ``phase_runs``       - runs a team scored in a phase of a given innings  (sql)
* ``season_runs``      - a player's runs in a season                        (sql)
* ``season_wickets``   - a player's wickets in a season                     (sql)

Retrieval is scored with recall@k and MRR for BM25, dense and hybrid search. With an LLM
configured, answers are also scored for correctness (the expected name or number appears).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cricket_ai.rag.corpus import PHASE_LABEL, Document, fmt_date
from cricket_ai.rag.retrieval import HybridRetriever

QUESTION_TYPES = ["player_of_match", "match_winner", "phase_runs", "season_runs", "season_wickets"]


def generate_questions(docs: list[Document], per_type: int = 20, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_kind: dict[str, list[Document]] = {}
    for d in docs:
        by_kind.setdefault(d.kind, []).append(d)

    def pick(pool: list[Document]) -> list[Document]:
        if not pool:
            return []
        idx = rng.choice(len(pool), min(per_type, len(pool)), replace=False)
        return [pool[i] for i in sorted(idx)]

    def when(m: dict) -> str:
        return fmt_date(m["date"])

    matches = by_kind.get("match", [])
    phases = by_kind.get("innings_phase", [])
    players = by_kind.get("player_season", [])
    qs = []
    for d in pick(
        [
            d
            for d in matches
            if d.metadata.get("player_of_match") and "," not in d.metadata["player_of_match"]
        ]
    ):
        t1, t2 = d.metadata["teams"]
        qs.append(
            {
                "type": "player_of_match",
                "expected": d.metadata["player_of_match"],
                "gold": d.doc_id,
                "question": f"Who was the player of the match when {t1} played {t2} "
                f"on {when(d.metadata)}?",
            }
        )
    for d in pick([d for d in matches if d.metadata.get("winner")]):
        t1, t2 = d.metadata["teams"]
        qs.append(
            {
                "type": "match_winner",
                "expected": d.metadata["winner"],
                "gold": d.doc_id,
                "question": f"Which team won the match between {t1} and {t2} "
                f"on {when(d.metadata)}?",
            }
        )
    for d in pick(phases):
        m = d.metadata
        qs.append(
            {
                "type": "phase_runs",
                "expected": str(m["runs"]),
                "gold": d.doc_id,
                "question": f"How many runs did {m['team']} score in the {PHASE_LABEL[m['phase']]} "
                f"against {m['opponent']} on {when(m)}?",
            }
        )
    for d in pick([d for d in players if d.metadata.get("runs", 0) >= 100]):
        m = d.metadata
        qs.append(
            {
                "type": "season_runs",
                "expected": str(m["runs"]),
                "gold": d.doc_id,
                "question": f"How many runs did {m['player']} score in the {m['season']} season?",
            }
        )
    for d in pick([d for d in players if d.metadata.get("wickets", 0) >= 5]):
        m = d.metadata
        qs.append(
            {
                "type": "season_wickets",
                "expected": str(m["wickets"]),
                "gold": d.doc_id,
                "question": f"How many wickets did {m['player']} take in the {m['season']} season?",
            }
        )
    return qs


def evaluate_retrieval(
    retriever: HybridRetriever,
    questions: list[dict],
    methods: tuple[str, ...] = ("bm25", "dense", "hybrid"),
    ks: tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    rows = []
    for method in methods:
        for q in questions:
            hits = retriever.search(q["question"], k=max(ks), method=method)
            ids = [h.doc.doc_id for h in hits]
            rank = ids.index(q["gold"]) + 1 if q["gold"] in ids else None
            row = {"method": method, "type": q["type"], "rr": 1 / rank if rank else 0.0}
            row.update({f"recall@{k}": float(rank is not None and rank <= k) for k in ks})
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_retrieval(per_question: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [c for c in per_question.columns if c.startswith("recall@")] + ["rr"]
    overall = per_question.groupby("method")[metrics].mean().round(3).rename(columns={"rr": "mrr"})
    by_type = (
        per_question.groupby(["type", "method"])["recall@5"]
        .mean()
        .unstack()
        .round(3)
        .reindex(QUESTION_TYPES)
        .dropna(how="all")
    )
    return overall, by_type


def answer_correct(expected: str, text: str) -> bool:
    text = text.lower()
    if expected.isdigit():
        return re.search(rf"(?<![\d.]){expected}(?![\d.])", text) is not None
    # names: accept the surname (e.g. 'Kohli' for 'V Kohli')
    tokens = [t for t in re.findall(r"\w+", expected.lower()) if len(t) > 1]
    return bool(tokens) and tokens[-1] in text


def evaluate_answers(rag, questions: list[dict]) -> pd.DataFrame:
    rows = []
    for q in questions:
        a = rag.ask(q["question"])
        rows.append(
            {
                "type": q["type"],
                "route": a.route,
                "correct": answer_correct(q["expected"], a.text),
                "expected": q["expected"],
                "answer": a.text,
                "sql": a.sql,
                "error": a.error,
                "seconds": a.seconds,
                "question": q["question"],
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    overall: pd.DataFrame,
    by_type: pd.DataFrame,
    answers: pd.DataFrame | None,
    meta: dict,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    md = [
        "# RAG evaluation\n",
        f"{meta['n_questions']} generated questions with known answers; corpus of "
        f"{meta['n_docs']} documents; embedder: {meta['embedder']}.\n",
        "## Retrieval (is the gold document in the top k?)\n",
        overall.to_markdown(),
        "\n\n### Recall@5 by question type\n",
        by_type.to_markdown(),
        "\n",
    ]
    result = {
        "meta": meta,
        "retrieval": overall.to_dict(orient="index"),
        "retrieval_by_type": by_type.to_dict(orient="index"),
    }
    if answers is not None:
        acc = answers.groupby("type").correct.mean().round(3).reindex(QUESTION_TYPES).dropna()
        md += [
            f"\n## Answers (LLM: {meta.get('llm')})\n",
            f"Overall accuracy: **{answers.correct.mean():.1%}**; median latency "
            f"{answers.seconds.median():.1f}s.\n\n",
            acc.to_frame("accuracy").to_markdown(),
            "\n",
        ]
        result["answer_accuracy"] = float(answers.correct.mean())
        result["answer_accuracy_by_type"] = acc.to_dict()
        answers.to_csv(out / "answers.csv", index=False)
    (out / "report.md").write_text("\n".join(md))
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
