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
from cricket_ai.rag.sql import CricketSQL
from cricket_ai.winprob.predict import chase_probability, overs_to_balls

EXPENSIVE_TOOLS = {"generate_synthetic_overs"}
# The agent sends every tool description with every model call, so this is the short form
# of rag.sql.SCHEMA (which the RAG pipeline still uses in full when it writes SQL).
AGENT_SQL_HINT = (
    "Tables: matches(match_id, season TEXT e.g. '2016', start_date, team1, team2, venue, "
    "winner, event_stage, player_of_match); batting(match_id, season, batting_team, player, "
    "runs, balls, not_out), one row per batter per innings; bowling(match_id, season, "
    "bowling_team, player, balls, runs_conceded, wickets); deliveries(match_id, innings, "
    "over 0-based, batter, bowler, runs_batter, runs_total, is_wicket, is_super_over). "
    "Names have initials: use player ILIKE '%Kohli%'."
)
MAX_TOOL_CHARS = 4000


@dataclass
class ToolCall:
    name: str
    args: dict
    seconds: float
    ok: bool
    output: str


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
        cls, processed: Path, models: Path, device: str | None = "cpu"
    ) -> CricketContext:
        ctx = cls()
        if (processed / "matches.parquet").exists():
            ctx.sql = CricketSQL.from_dir(processed)
        if (processed / "rag" / "index.json").exists():
            # Embedding one question is quick on the CPU, and it leaves the GPU's memory to
            # the language model (on a 4 GB card they otherwise compete).
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


def normalise_season(season) -> str | None:
    """Accept 2016, '2016', '2016 season', 'IPL 2016'. Returns None for no season."""
    if season is None:
        return None
    s = str(season).lower().replace("season", "").replace("ipl", "").strip()
    return s or None


def _season_clause(season: str | None) -> tuple[str, list]:
    """SQL filter for a season; '2020' also matches split seasons stored as '2020/21'."""
    if not season:
        return "", []
    if "/" in season:
        return " AND season = ?", [season]
    return " AND (season = ? OR season LIKE ?)", [season, f"{season}/%"]


def _num(x, fmt="{:.0f}") -> str:
    return "n/a" if x is None or pd.isna(x) else fmt.format(x)


def player_stats(ctx: CricketContext, player: str, season: str | int | None = None) -> str:
    season = normalise_season(season)
    bat, bowl = _player_rows(ctx, f"%{player.strip()}%", season)
    note = ""
    if bat.empty and bowl.empty:
        # Names are stored with initials ('DA Warner'); models often expand them
        # ('David Warner'). Fall back to the surname and say so.
        surname = player.strip().split()[-1] if player.strip() else ""
        if len(surname) >= 3:
            bat, bowl = _player_rows(ctx, f"%{surname}%", season)
            if not (bat.empty and bowl.empty):
                note = f"No exact match for '{player}'; matched by surname '{surname}'. "
    when = f"the {season} season" if season else "all seasons"
    bat_i, bowl_i = bat.set_index("player"), bowl.set_index("player")
    names = sorted(set(bat.player) | set(bowl.player))
    if not names:
        return f"No player matching '{player}' in {when}."

    query = player.strip().lower()

    def rank(name: str) -> tuple:
        # exact name first, then same surname, then whoever played most
        played = bat_i.innings.get(name, 0) + bowl_i.matches.get(name, 0)
        return (name.lower() != query, name.lower().split()[-1] != query.split()[-1], -played)

    names.sort(key=rank)
    lines = [note.strip()] if note else []
    text = _player_lines(names[:1], bat_i, bowl_i, when, lines)
    if len(names) > 1:
        # One player in full keeps the reply short and unambiguous; the rest are named only.
        others = ", ".join(names[1:6]) + (" and more" if len(names) > 6 else "")
        text += f"\nOther players matching '{player}': {others}. Ask by full name for those."
    return text


def _player_rows(ctx: CricketContext, like: str, season: str | None):
    clause, sparams = _season_clause(season)
    params = [like, *sparams]
    bat = ctx.sql.con.execute(
        f"""SELECT player, COUNT(*) AS innings, SUM(runs) AS runs, SUM(balls) AS balls,
                   SUM(CASE WHEN NOT not_out THEN 1 ELSE 0 END) AS outs, MAX(runs) AS highest
            FROM batting WHERE player ILIKE ?{clause} GROUP BY player""",
        params,
    ).df()
    bowl = ctx.sql.con.execute(
        f"""SELECT player, COUNT(DISTINCT match_id) AS matches, SUM(balls) AS balls_bowled,
                   SUM(wickets) AS wickets, SUM(runs_conceded) AS runs_conceded
            FROM bowling WHERE player ILIKE ?{clause} GROUP BY player""",
        params,
    ).df()
    return bat, bowl


def _player_lines(names, bat_i, bowl_i, when: str, lines: list[str]) -> str:
    for name in names[:3]:
        # Batting and bowling are reported separately: runs conceded as a bowler are never
        # added to, or worded as, runs scored.
        if name in bat_i.index:
            b = bat_i.loc[name]
            sr = 100 * b.runs / b.balls if b.balls else None
            avg = _num(b.runs / b.outs, "{:.2f}") if b.outs else "n/a (never dismissed)"
            batting = (
                f"BATTING: scored {_num(b.runs)} runs in {_num(b.innings)} innings off "
                f"{_num(b.balls)} balls, strike rate {_num(sr, '{:.1f}')}, average {avg}, "
                f"highest score {_num(b.highest)}."
            )
        else:
            batting = "BATTING: did not bat."
        if name in bowl_i.index and bowl_i.loc[name].balls_bowled:
            w = bowl_i.loc[name]
            econ = 6 * w.runs_conceded / w.balls_bowled
            overs = f"{int(w.balls_bowled) // 6}.{int(w.balls_bowled) % 6}"
            bowling = (
                f"BOWLING: took {_num(w.wickets)} wickets in {_num(w.matches)} matches, "
                f"bowled {overs} overs and conceded {_num(w.runs_conceded)} runs "
                f"(economy {econ:.2f})."
            )
        else:
            bowling = "BOWLING: did not bowl."
        lines.append(f"{name}, {when}. {batting} {bowling}")
    return "\n".join(lines)


LEADERBOARD_STATS = {
    # stat: (table, value expression, order, unit, minimum balls for rate stats)
    "runs": ("batting", "SUM(runs)", "DESC", "runs scored", 0),
    "wickets": ("bowling", "SUM(wickets)", "DESC", "wickets taken", 0),
    "strike_rate": (
        "batting",
        "ROUND(100.0 * SUM(runs) / SUM(balls), 1)",
        "DESC",
        "strike rate",
        100,
    ),
    "economy": (
        "bowling",
        "ROUND(6.0 * SUM(runs_conceded) / SUM(balls), 2)",
        "ASC",
        "economy (runs conceded per over)",
        120,
    ),
    "sixes": (
        "deliveries",
        "SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END)",
        "DESC",
        "sixes hit",
        0,
    ),
    "fours": (
        "deliveries",
        "SUM(CASE WHEN runs_batter = 4 THEN 1 ELSE 0 END)",
        "DESC",
        "fours hit",
        0,
    ),
}


def leaderboard(
    ctx: CricketContext, stat: str = "runs", season: str | int | None = None, n: int = 5
) -> str:
    stat = stat.lower().strip().replace(" ", "_")
    if stat not in LEADERBOARD_STATS:
        return f"Unknown stat {stat!r}. Choose from: {', '.join(LEADERBOARD_STATS)}."
    table, expr, order, unit, min_balls = LEADERBOARD_STATS[stat]
    season = normalise_season(season)
    clause, params = _season_clause(season)
    n = int(min(max(int(n), 1), 20))
    if table == "deliveries":
        sql = f"""SELECT d.batter AS player, {expr} AS value FROM deliveries d
                  JOIN matches m USING (match_id)
                  WHERE NOT d.is_super_over{clause.replace("season", "m.season")}
                  GROUP BY 1 ORDER BY value {order} LIMIT {n}"""
    else:
        if not season:
            min_balls *= 5
        having = f"HAVING SUM(balls) >= {min_balls}" if min_balls else ""
        sql = f"""SELECT player, {expr} AS value FROM {table} WHERE TRUE{clause}
                  GROUP BY player {having} ORDER BY value {order} LIMIT {n}"""
    rows = ctx.sql.con.execute(sql, params).df()
    when = f"the {season} season" if season else "all seasons"
    if rows.empty:
        return f"No data for {unit} in {when}."
    note = f" (minimum {min_balls} balls)" if min_balls else ""
    first = rows.iloc[0]
    lines = [
        f"Leader: {first.player}, {first.value:g} {unit} in {when}.",
        f"Top {len(rows)} by {unit} in {when}{note}:",
    ]
    lines += [f"{i}. {r.player}: {r.value:g}" for i, r in enumerate(rows.itertuples(), 1)]
    return "\n".join(lines)


def win_probability(
    ctx: CricketContext,
    target: int,
    runs_scored: int,
    overs: float | str,
    wickets_lost: int,
    total_overs: int = 20,
    runs_last_12_balls: int | None = None,
    wickets_last_12_balls: int = 0,
) -> str:
    if not 0 <= float(overs) <= total_overs:
        return (
            f"Error: overs={overs} is outside 0-{total_overs}. Pass overs bowled as in the "
            "question (e.g. 12 or 12.3), not balls."
        )
    balls_bowled = overs_to_balls(overs)
    total_balls = total_overs * 6
    runs_needed = target - runs_scored
    balls_left = total_balls - balls_bowled
    situation = f"Situation: chasing {target}, {runs_scored}/{wickets_lost} after {overs} overs."
    if runs_needed <= 0:
        return f"{situation} The target is already reached: win probability 100%."
    if balls_left <= 0 or wickets_lost >= 10:
        return f"{situation} The chase is over without reaching the target: win probability 0%."
    p = chase_probability(
        ctx.winprob_model,
        target,
        runs_scored,
        balls_bowled,
        wickets_lost,
        total_balls,
        ctx.latest_par_rpo,
        runs_last_12_balls=runs_last_12_balls,
        wickets_last_12_balls=wickets_last_12_balls,
    )
    rrr = min(runs_needed * 6 / balls_left, 36.0)
    return (
        f"{situation} Needs {runs_needed} runs off {balls_left} balls "
        f"(required rate {rrr:.2f}). Win probability for the chasing side: {p:.1%}."
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
        ctx.calls.append(ToolCall(name, kwargs, round(time.perf_counter() - start, 3), ok, out))
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
                "One player's batting and bowling, optionally for one season "
                "(e.g. season='2016'). Partial names work, e.g. 'Kohli'. Not for rankings.",
            )
        )
        specs.append(
            (
                leaderboard,
                "leaderboard",
                "Rank players: who scored the most runs, took the most wickets, hit the most "
                "sixes or fours, or had the best strike rate or economy, overall or in one "
                f"season. stat is one of: {', '.join(LEADERBOARD_STATS)}.",
            )
        )
        specs.append(
            (
                run_sql,
                "run_sql",
                "Run one read-only DuckDB SELECT for anything the other tools cannot answer "
                "(team totals, records, custom filters). " + AGENT_SQL_HINT,
            )
        )
    if ctx.winprob_model is not None:
        specs.append(
            (
                win_probability,
                "win_probability",
                "Chasing side's chance of winning a T20. Pass the numbers exactly as the "
                "question states them: target, runs_scored, overs bowled (e.g. 12 or 12.3) "
                "and wickets_lost. For '100/2 after 12 overs': runs_scored=100, "
                "wickets_lost=2, overs=12.",
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
        else pd.DataFrame(columns=["name", "args", "seconds", "ok", "output"])
    )


__all__ = ["CricketContext", "EXPENSIVE_TOOLS", "build_tools", "summarise_calls"]
