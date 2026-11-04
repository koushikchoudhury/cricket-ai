"""The agent's tools: each one wraps a module of the project.

=========================== ======== ===================================================
tool                        module   what it does
=========================== ======== ===================================================
search_cricket              4        hybrid search over match reports, phases, seasons
player_stats                1/4      a player's batting and bowling, optionally by season
run_sql                     4        read-only SQL for anything player_stats cannot answer
win_probability             1        chasing side's win chance from a match situation
generate_synthetic_overs    2        sample overs from the GAN and summarise them
=========================== ======== ===================================================

Tools are only offered when what they need exists (an index, a trained model), and every
call is timed and recorded for the trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from langchain_core.tools import StructuredTool

from cricket_ai.rag.retrieval import HybridRetriever
from cricket_ai.rag.sql import SCHEMA, CricketSQL

EXPENSIVE_TOOLS = {"generate_synthetic_overs"}
MAX_TOOL_CHARS = 4000


@dataclass
class ToolCall:
    name: str
    args: dict
    seconds: float
    ok: bool
    output_preview: str


@dataclass
class CricketContext:
    retriever: HybridRetriever | None = None
    sql: CricketSQL | None = None
    winprob_model: object | None = None
    latest_par_rpo: float | None = None
    gan: object | None = None
    calls: list[ToolCall] = field(default_factory=list)

    @classmethod
    def from_dataset(
        cls, processed: Path, models: Path, device: str | None = None
    ) -> CricketContext:
        ctx = cls()
        if (processed / "matches.parquet").exists():
            ctx.sql = CricketSQL.from_dir(processed)
        if (processed / "rag" / "index.json").exists():
            ctx.retriever = HybridRetriever.load(processed / "rag", device=device)
        try:
            from cricket_ai.winprob.model import load

            if (models / "winprob_mlp.pt").exists():
                ctx.winprob_model = load(models / "winprob_mlp.pt")
                states = pd.read_parquet(processed / "chase_states.parquet")
                latest = states[states.start_date == states.start_date.max()]
                ctx.latest_par_rpo = float((latest.par_score / 20).mean())
        except ImportError:  # torch not installed
            pass
        try:
            from cricket_ai.synthetic.gan import TabularGAN

            if (models / "synth_gan" / "generator.pt").exists():
                ctx.gan = TabularGAN.load(models / "synth_gan")
        except ImportError:
            pass
        return ctx


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + "\n...(truncated)"


def _table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "(no rows)"
    return df.head(max_rows).to_markdown(index=False)


# ----- tool implementations (plain functions, easy to test) -----
def search_cricket(ctx: CricketContext, query: str, k: int = 5) -> str:
    hits = ctx.retriever.search(query, k=k)
    return "\n\n".join(f"[{h.doc.doc_id}] {h.doc.title}\n{h.doc.text}" for h in hits)


def run_sql(ctx: CricketContext, query: str) -> str:
    return _table(ctx.sql.run(query))


def player_stats(ctx: CricketContext, player: str, season: str | None = None) -> str:
    like = f"%{player.strip()}%"
    where = "player ILIKE ?" + (" AND season = ?" if season else "")
    params = [like, str(season)] if season else [like]
    bat = ctx.sql.con.execute(
        f"""SELECT player, {"season," if season else ""} COUNT(*) AS innings, SUM(runs) AS runs,
                   SUM(balls) AS balls, ROUND(100.0 * SUM(runs) / NULLIF(SUM(balls), 0), 1) AS strike_rate,
                   ROUND(SUM(runs) / NULLIF(SUM(CASE WHEN NOT not_out THEN 1 ELSE 0 END), 0), 2) AS average,
                   MAX(runs) AS highest
            FROM batting WHERE {where} GROUP BY ALL ORDER BY runs DESC LIMIT 5""",
        params,
    ).df()
    bowl = ctx.sql.con.execute(
        f"""SELECT player, {"season," if season else ""} COUNT(DISTINCT match_id) AS matches,
                   SUM(wickets) AS wickets, SUM(runs_conceded) AS runs_conceded,
                   ROUND(6.0 * SUM(runs_conceded) / NULLIF(SUM(balls), 0), 2) AS economy
            FROM bowling WHERE {where} GROUP BY ALL ORDER BY wickets DESC LIMIT 5""",
        params,
    ).df()
    if bat.empty and bowl.empty:
        return f"No player matching '{player}'" + (f" in season {season}" if season else "") + "."
    return f"Batting:\n{_table(bat)}\n\nBowling:\n{_table(bowl)}"


def win_probability(
    ctx: CricketContext,
    target: int,
    runs_scored: int,
    balls_bowled: int,
    wickets_lost: int,
    total_balls: int = 120,
    runs_last_12_balls: int | None = None,
    wickets_last_12_balls: int = 0,
) -> str:
    from cricket_ai.winprob.model import predict_proba

    runs_needed = target - runs_scored
    balls_left = total_balls - balls_bowled
    wickets_in_hand = 10 - wickets_lost
    if runs_needed <= 0:
        return "The chasing side has already reached the target: win probability 100%."
    if balls_left <= 0 or wickets_in_hand <= 0:
        return "The chase is over and the target was not reached: win probability 0%."
    crr = runs_scored * 6 / balls_bowled if balls_bowled else 0.0
    rrr = min(runs_needed * 6 / balls_left, 36.0)
    par = (ctx.latest_par_rpo or 8.0) * total_balls / 6
    recent = runs_last_12_balls if runs_last_12_balls is not None else round(crr * 2)
    state = pd.DataFrame(
        [
            {
                "runs_needed": runs_needed,
                "balls_left": balls_left,
                "wickets_in_hand": wickets_in_hand,
                "target": target,
                "current_run_rate": crr,
                "required_run_rate": rrr,
                "runs_last_12_balls": recent,
                "wickets_last_12_balls": wickets_last_12_balls,
                "par_score": par,
                "target_vs_par": target - par,
                "balls_bowled": balls_bowled,
            }
        ]
    )
    p = float(predict_proba(ctx.winprob_model, state)[0])
    return (
        f"Chasing {target}: {runs_scored}/{wickets_lost} after {balls_bowled // 6}.{balls_bowled % 6} "
        f"overs needs {runs_needed} off {balls_left} balls (required rate {rrr:.2f}). "
        f"Win probability for the chasing side: {p:.1%} (era par score {par:.0f})."
    )


def generate_synthetic_overs(
    ctx: CricketContext, n: int = 1000, over: int | None = None, innings: int | None = None
) -> str:
    from cricket_ai.synthetic.table import validity

    n = int(min(max(n, 50), 20_000))
    synth = ctx.gan.sample(
        n * 4 if over is not None or innings is not None else n, seed=int(time.time())
    )
    if over is not None:
        synth = synth[synth.over == int(over) - 1]  # users count overs from 1
    if innings is not None:
        synth = synth[synth.innings == int(innings)]
    synth = synth.head(n)
    if synth.empty:
        return "The generator produced no rows matching that filter."
    summary = pd.DataFrame(
        {
            "overs generated": [len(synth)],
            "mean runs": [round(synth.runs.mean(), 2)],
            "wicket in over %": [round(100 * (synth.wickets > 0).mean(), 1)],
            "boundaries per over": [round((synth.fours + synth.sixes).mean(), 2)],
            "valid cricket %": [round(100 * validity(synth)["all_rules"], 1)],
        }
    )
    return f"Synthetic overs from the GAN (module 2):\n{_table(summary)}"


# ----- LangChain tool objects -----
def _traced(ctx: CricketContext, name: str, fn):
    def run(**kwargs):
        start = time.perf_counter()
        try:
            out, ok = _clip(fn(ctx, **kwargs)), True
        except Exception as exc:  # noqa: BLE001 - the model sees the error and can recover
            out, ok = f"Error: {type(exc).__name__}: {exc}", False
        ctx.calls.append(
            ToolCall(name, kwargs, round(time.perf_counter() - start, 3), ok, out[:300])
        )
        return out

    return run


def build_tools(ctx: CricketContext) -> list[StructuredTool]:
    specs = []
    if ctx.retriever is not None:
        specs.append(
            (
                search_cricket,
                "search_cricket",
                "Search match reports, innings phases (powerplay/middle/death) and player "
                "season summaries. Use for who won, player of the match, how a game went. "
                "Include names and dates in the query.",
            )
        )
    if ctx.sql is not None:
        specs.append(
            (
                player_stats,
                "player_stats",
                "Batting and bowling totals for a player, optionally for one season "
                "(e.g. season='2016'). Partial names work, e.g. 'Kohli'.",
            )
        )
        specs.append(
            (
                run_sql,
                "run_sql",
                "Run one read-only DuckDB SELECT for questions player_stats cannot answer "
                "(rankings, records, team totals).\n" + SCHEMA,
            )
        )
    if ctx.winprob_model is not None:
        specs.append(
            (
                win_probability,
                "win_probability",
                "Chasing side's chance of winning a T20 from the situation: target, runs "
                "scored so far, legal balls bowled (overs x 6) and wickets lost.",
            )
        )
    if ctx.gan is not None:
        specs.append(
            (
                generate_synthetic_overs,
                "generate_synthetic_overs",
                "Generate n synthetic overs with the GAN and summarise them; optionally "
                "filter by over number (1-20) or innings (1/2). Slow; ask only if needed.",
            )
        )

    tools = []
    for fn, name, description in specs:
        tools.append(
            StructuredTool.from_function(
                func=_traced(ctx, name, fn),
                name=name,
                description=description,
                args_schema=_schema_for(fn),
            )
        )
    return tools


def _schema_for(fn):
    """A pydantic model for the function's arguments (everything after ``ctx``)."""
    import inspect
    import typing

    from pydantic import create_model

    hints = typing.get_type_hints(fn)
    fields = {}
    for name, p in list(inspect.signature(fn).parameters.items())[1:]:
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[name] = (hints.get(name, str), default)
    return create_model(f"{fn.__name__}_args", **fields)


def summarise_calls(calls: list[ToolCall]) -> pd.DataFrame:
    return (
        pd.DataFrame([vars(c) for c in calls])
        if calls
        else pd.DataFrame(columns=["name", "args", "seconds", "ok", "output_preview"])
    )


__all__ = ["CricketContext", "EXPENSIVE_TOOLS", "build_tools", "summarise_calls"]
