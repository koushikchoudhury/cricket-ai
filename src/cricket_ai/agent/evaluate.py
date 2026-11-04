"""Agent evaluation: does it pick the right tool and reach the right answer?

Tasks are generated from the data, so each has an expected tool and an expected answer:

* ``season_runs``    - a player's runs in a season            -> player_stats or run_sql
* ``top_scorer``     - who scored most runs in a season        -> leaderboard or run_sql
* ``match_winner``   - who won a given match                   -> search_cricket
* ``chase``          - win probability for a chase situation   -> win_probability
* ``synthetic``      - summarise GAN-generated overs           -> generate_synthetic_overs

With ``repeats`` > 1 every task is run several times. Local models are not fully
deterministic even at temperature 0 (three identical runs gave 82%, 88% and 100%), so a
single run of a few tasks is noise; the report gives the mean, the range across runs and how
consistent each task is.

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
            from cricket_ai.agent.tools import win_probability

            # Ground truth: the tool's answer when called with the correct arguments.
            truth = win_probability(ctx, target, runs, overs, wkts)
            tasks.append(
                {
                    "type": "chase",
                    "expected": PERCENT.findall(truth)[-1].replace(" ", ""),
                    "expected_args": {
                        "target": target,
                        "runs_scored": runs,
                        "overs": overs,
                        "wickets_lost": wkts,
                    },
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


def args_match(expected: dict, calls: list[dict]) -> bool:
    """Did any call pass the expected arguments (numbers compared as numbers)?"""

    def same(a, b):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return str(a).strip().lower() == str(b).strip().lower()

    return any(all(k in c and same(c[k], v) for k, v in expected.items()) for c in calls)


def percent_matches(expected: str, answer: str) -> bool:
    """The answer states the expected probability (rounding to whole percent is fine)."""
    want = float(expected.rstrip("%"))
    for found in PERCENT.findall(answer):
        got = float(found.rstrip("% "))
        decimals = len(found.rstrip("% ").split(".")[1]) if "." in found else 0
        if abs(round(want, decimals) - got) < 1e-9:
            return True
    return False


def score_task(
    task: dict,
    answer: str,
    tools_used: list[str],
    tool_outputs: list[str] | None = None,
    tool_args: list[dict] | None = None,
) -> dict:
    right_tool = bool(EXPECTED_TOOLS[task["type"]] & set(tools_used))
    # Faithful to the evidence. Note this cannot catch a tool called with wrong arguments,
    # because the answer then faithfully repeats a wrong tool output; args_ok covers that.
    invented = unsupported_numbers(answer, task["question"], *(tool_outputs or []))
    args_ok = None
    if task.get("expected_args"):
        args_ok = args_match(task["expected_args"], tool_args or [])
    if task["type"] == "chase":
        correct = right_tool and bool(args_ok) and percent_matches(task["expected"], answer)
    elif task["type"] == "synthetic":
        correct = right_tool and bool(re.search(r"\d", answer))
    else:
        correct = answer_correct(task["expected"], answer)
    return {
        "right_tool": right_tool,
        "faithful": not invented,
        "invented_numbers": ",".join(invented),
        "args_ok": args_ok,
        "correct": bool(correct and not invented),
    }


def evaluate_agent(agent, tasks: list[dict], repeats: int = 1) -> pd.DataFrame:
    rows = []
    for run, t in ((r, t) for r in range(1, max(1, repeats) + 1) for t in tasks):
        try:
            r = agent.ask(t["question"])
            answer, used, outputs = r.answer, r.tools_used, [c["output"] for c in r.tool_calls]
            args = [c["args"] for c in r.tool_calls]
            steps, secs, limit, error = r.steps, r.seconds, r.stopped_by_limit, None
        except Exception as exc:  # noqa: BLE001 - one failing task should not stop the run
            answer, used, outputs, steps, secs, limit, error = "", [], [], 0, 0.0, False, str(exc)
            args = []
        rows.append(
            {
                "run": run,
                "type": t["type"],
                "question": t["question"],
                "expected": t["expected"],
                "answer": answer,
                "tools": ",".join(used),
                "tool_args": json.dumps(args, default=str),
                "steps": steps,
                "seconds": secs,
                "step_limit": limit,
                "error": error,
                **score_task(t, answer, used, outputs, args),
            }
        )
    return pd.DataFrame(rows)


def _rate(series: pd.Series) -> float | None:
    s = series.dropna()
    return round(float(s.astype(bool).mean()), 3) if len(s) else None


def write_report(results: pd.DataFrame, out: Path, llm: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if "run" not in results:
        results = results.assign(run=1)
    runs = int(results.run.nunique())
    per_run = (
        results.groupby("run")
        .agg(
            right_tool=("right_tool", "mean"),
            faithful=("faithful", "mean"),
            correct=("correct", "mean"),
            seconds=("seconds", "median"),
        )
        .round(3)
    )
    per_run["args_ok"] = results.groupby("run").args_ok.apply(_rate)

    # by type: mean over runs, plus the lowest and highest run
    type_run = results.groupby(["type", "run"]).correct.mean()
    by_type = results.groupby("type").agg(
        tasks_per_run=("question", lambda q: q.nunique()),
        right_tool=("right_tool", "mean"),
        faithful=("faithful", "mean"),
        correct=("correct", "mean"),
        steps=("steps", "mean"),
        seconds=("seconds", "median"),
    )
    by_type["args_ok"] = results.groupby("type").args_ok.apply(_rate)
    if runs > 1:
        by_type["correct_min"] = type_run.groupby("type").min()
        by_type["correct_max"] = type_run.groupby("type").max()
    by_type = by_type.round(3)

    # consistency: share of tasks that got the same verdict in every run
    consistency = round(float((results.groupby("question").correct.nunique() == 1).mean()), 3)
    summary = {
        "llm": llm,
        "tasks": int(results.question.nunique()),
        "runs": runs,
        "tool_accuracy": round(float(results.right_tool.mean()), 3),
        "faithfulness": round(float(results.faithful.mean()), 3),
        "argument_accuracy": _rate(results.args_ok),
        "answer_accuracy": round(float(per_run.correct.mean()), 3),
        "answer_accuracy_min": float(per_run.correct.min()),
        "answer_accuracy_max": float(per_run.correct.max()),
        "consistency": consistency,
        "mean_steps": round(float(results.steps.mean()), 2),
        "median_seconds": round(float(results.seconds.median()), 2),
        "step_limit_hits": int(results.step_limit.sum()),
        "errors": int(results.error.notna().sum()),
    }
    results.to_csv(out / "results.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                **summary,
                "by_type": by_type.to_dict(orient="index"),
                "per_run": per_run.to_dict(orient="index"),
            },
            indent=2,
            default=str,
        )
    )

    spread = (
        f" (range {summary['answer_accuracy_min']:.0%}-{summary['answer_accuracy_max']:.0%} "
        f"over {runs} runs; {consistency:.0%} of tasks got the same verdict every run)"
        if runs > 1
        else ""
    )
    args = (
        f", tool arguments {summary['argument_accuracy']:.0%}"
        if summary["argument_accuracy"] is not None
        else ""
    )
    bad = results[~results.faithful][["run", "type", "question", "invented_numbers", "answer"]]
    flaky = results.groupby("question").filter(lambda g: g.correct.nunique() > 1)
    flaky = (
        flaky.groupby(["type", "question"])
        .correct.mean()
        .round(2)
        .reset_index(name="share_correct")
    )
    md = [
        f"# Agent evaluation ({llm})\n",
        f"{summary['tasks']} tasks x {runs} run(s). Correct **{summary['answer_accuracy']:.0%}**"
        f"{spread} (correct requires faithful). Tool choice {summary['tool_accuracy']:.0%}"
        f"{args}, faithful {summary['faithfulness']:.0%}, {summary['mean_steps']} tool rounds on "
        f"average, median {summary['median_seconds']}s per task, {summary['step_limit_hits']} "
        f"step-limit stops, {summary['errors']} errors.\n",
        "## By task type\n",
        by_type.to_markdown(),
        "\n",
    ]
    if runs > 1:
        md += [
            "\n## Per run\n",
            per_run.to_markdown(),
            "\n",
            "\n## Tasks that changed verdict between runs\n",
            flaky.to_markdown(index=False) if len(flaky) else "None.",
            "\n",
        ]
    md += [
        "\n## Answers with numbers not found in the question or tool outputs\n",
        bad.to_markdown(index=False) if len(bad) else "None.",
        "\n",
    ]
    (out / "report.md").write_text("\n".join(md))
    return summary
