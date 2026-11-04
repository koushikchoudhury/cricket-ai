"""Agent evaluation: does it pick the right tool and reach the right answer?

Tasks are generated from the data, so each has an expected tool and an expected answer:

* ``season_runs``    - a player's runs in a season            -> player_stats or run_sql
* ``top_scorer``     - who scored most runs in a season        -> leaderboard or run_sql
* ``match_winner``   - who won a given match                   -> search_cricket
* ``chase``          - win probability for a chase situation   -> win_probability
* ``synthetic``      - summarise GAN-generated overs           -> generate_synthetic_overs

Reported: tool-choice accuracy, numeric faithfulness (every number in the answer appears in
the question or a tool output), answer accuracy, steps and latency per task type. An answer
counts as correct only if it is also faithful.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cricket_ai.agent.tools import CricketContext
from cricket_ai.faithfulness import unsupported_numbers
from cricket_ai.rag.corpus import fmt_date
from cricket_ai.rag.evaluate import answer_correct

EXPECTED_TOOLS = {
    "season_runs": {"player_stats", "run_sql"},
    "top_scorer": {"leaderboard", "run_sql"},
    "match_winner": {"search_cricket"},
    "chase": {"win_probability"},
    "synthetic": {"generate_synthetic_overs"},
}
PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")


def generate_tasks(ctx: CricketContext, per_type: int = 4, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    tasks: list[dict] = []
    if ctx.sql is not None:
        seasons = ctx.sql.con.execute(
            "SELECT player, season, SUM(runs) AS runs FROM batting GROUP BY ALL HAVING SUM(runs) >= 150"
        ).df()
        for i in rng.choice(len(seasons), min(per_type, len(seasons)), replace=False):
            r = seasons.iloc[int(i)]
            tasks.append(
                {
                    "type": "season_runs",
                    "expected": str(int(r.runs)),
                    "question": f"How many runs did {r.player} score in the {r.season} season?",
                }
            )
        tops = ctx.sql.con.execute(
            """SELECT season, arg_max(player, runs) AS player FROM
               (SELECT season, player, SUM(runs) AS runs FROM batting GROUP BY ALL) GROUP BY season"""
        ).df()
        for i in rng.choice(len(tops), min(per_type, len(tops)), replace=False):
            r = tops.iloc[int(i)]
            tasks.append(
                {
                    "type": "top_scorer",
                    "expected": r.player,
                    "question": f"Who scored the most runs in the {r.season} season?",
                }
            )
    if ctx.retriever is not None:
        matches = [d for d in ctx.retriever.docs if d.kind == "match" and d.metadata.get("winner")]
        for i in rng.choice(len(matches), min(per_type, len(matches)), replace=False):
            m = matches[int(i)].metadata
            t1, t2 = m["teams"]
            tasks.append(
                {
                    "type": "match_winner",
                    "expected": m["winner"],
                    "question": f"Who won the match between {t1} and {t2} on {fmt_date(m['date'])}?",
                }
            )
    if ctx.winprob_model is not None:
        for target, runs, overs, wkts in [
            (180, 100, 12, 2),
            (200, 60, 10, 6),
            (150, 140, 18, 3),
            (170, 40, 8, 1),
        ][:per_type]:
            tasks.append(
                {
                    "type": "chase",
                    "expected": None,
                    "question": f"A team chasing {target} is {runs}/{wkts} after {overs} overs. "
                    "What is their chance of winning?",
                }
            )
    if ctx.gan is not None:
        tasks.append(
            {
                "type": "synthetic",
                "expected": None,
                "question": "Generate 500 synthetic overs from the GAN. What are the average runs per over?",
            }
        )
    return tasks


def score_task(
    task: dict, answer: str, tools_used: list[str], tool_outputs: list[str] | None = None
) -> dict:
    right_tool = bool(EXPECTED_TOOLS[task["type"]] & set(tools_used))
    invented = unsupported_numbers(answer, task["question"], *(tool_outputs or []))
    if task["type"] == "chase":
        correct = right_tool and bool(PERCENT.search(answer))
    elif task["type"] == "synthetic":
        correct = right_tool and bool(re.search(r"\d", answer))
    else:
        correct = answer_correct(task["expected"], answer)
    return {
        "right_tool": right_tool,
        "faithful": not invented,
        "invented_numbers": ",".join(invented),
        "correct": bool(correct and not invented),
    }


def evaluate_agent(agent, tasks: list[dict]) -> pd.DataFrame:
    rows = []
    for t in tasks:
        try:
            r = agent.ask(t["question"])
            answer, used, outputs = r.answer, r.tools_used, [c["output"] for c in r.tool_calls]
            steps, secs, limit, error = r.steps, r.seconds, r.stopped_by_limit, None
        except Exception as exc:  # noqa: BLE001 - one failing task should not stop the run
            answer, used, outputs, steps, secs, limit, error = "", [], [], 0, 0.0, False, str(exc)
        rows.append(
            {
                "type": t["type"],
                "question": t["question"],
                "expected": t["expected"],
                "answer": answer,
                "tools": ",".join(used),
                "steps": steps,
                "seconds": secs,
                "step_limit": limit,
                "error": error,
                **score_task(t, answer, used, outputs),
            }
        )
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame, out: Path, llm: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    by_type = (
        results.groupby("type")
        .agg(
            tasks=("question", "size"),
            right_tool=("right_tool", "mean"),
            faithful=("faithful", "mean"),
            correct=("correct", "mean"),
            steps=("steps", "mean"),
            seconds=("seconds", "median"),
        )
        .round(3)
    )
    summary = {
        "llm": llm,
        "tasks": len(results),
        "tool_accuracy": round(float(results.right_tool.mean()), 3),
        "faithfulness": round(float(results.faithful.mean()), 3),
        "answer_accuracy": round(float(results.correct.mean()), 3),
        "mean_steps": round(float(results.steps.mean()), 2),
        "median_seconds": round(float(results.seconds.median()), 2),
        "step_limit_hits": int(results.step_limit.sum()),
        "errors": int(results.error.notna().sum()),
    }
    results.to_csv(out / "results.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({**summary, "by_type": by_type.to_dict(orient="index")}, indent=2, default=str)
    )
    bad = results[~results.faithful][["type", "question", "invented_numbers", "answer"]]
    (out / "report.md").write_text(
        f"# Agent evaluation ({llm})\n\n"
        f"{summary['tasks']} tasks. Tool choice {summary['tool_accuracy']:.0%}, faithful "
        f"{summary['faithfulness']:.0%}, correct {summary['answer_accuracy']:.0%} "
        f"(correct requires faithful), {summary['mean_steps']} tool rounds on average, median "
        f"{summary['median_seconds']}s per task, {summary['step_limit_hits']} step-limit stops, "
        f"{summary['errors']} errors.\n\n{by_type.to_markdown()}\n\n"
        "## Answers with numbers not found in the question or tool outputs\n\n"
        + (bad.to_markdown(index=False) if len(bad) else "None.")
        + "\n"
    )
    return summary
