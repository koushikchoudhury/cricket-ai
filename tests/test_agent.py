import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("duckdb")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from matchgen import generate_match  # noqa: E402

from cricket_ai.agent.evaluate import (  # noqa: E402
    evaluate_agent,
    generate_tasks,
    score_task,
    write_report,
)
from cricket_ai.agent.graph import CricketAgent, write_trace  # noqa: E402
from cricket_ai.agent.tools import CricketContext, build_tools  # noqa: E402
from cricket_ai.data.build import build_tables  # noqa: E402
from cricket_ai.rag.corpus import build_corpus  # noqa: E402
from cricket_ai.rag.retrieval import HashingEmbedder, HybridRetriever  # noqa: E402
from cricket_ai.rag.sql import CricketSQL  # noqa: E402


class Scripted(GenericFakeChatModel):
    """A fake chat model that plays back scripted replies, including tool calls."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*replies):
    msgs = []
    for r in replies:
        if isinstance(r, str):
            msgs.append(AIMessage(r))
        else:  # list of (tool name, args)
            msgs.append(
                AIMessage(
                    "",
                    tool_calls=[
                        {"name": n, "args": a, "id": f"call{i}"} for i, (n, a) in enumerate(r)
                    ],
                )
            )
    return Scripted(messages=iter(msgs))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    for seed in range(30):
        (raw / f"g{seed:03d}.json").write_text(json.dumps(generate_match(seed, day_step=9)))
    (raw / "tiny_match.json").write_text(
        (Path(__file__).parent / "fixtures" / "tiny_match.json").read_text()
    )
    return build_tables(raw, workers=1)


@pytest.fixture
def ctx(built):
    return CricketContext(
        retriever=HybridRetriever(build_corpus(built.matches, built.deliveries), HashingEmbedder()),
        sql=CricketSQL(built.matches, built.deliveries, built.players),
    )


def test_tools_offered_only_when_available(ctx):
    names = [t.name for t in build_tools(ctx)]
    assert names == ["search_cricket", "player_stats", "leaderboard", "run_sql"]
    assert build_tools(CricketContext()) == []


def test_player_stats_keeps_batting_and_bowling_apart(ctx):
    tool = {t.name: t for t in build_tools(ctx)}["player_stats"]
    b1 = tool.invoke({"player": "B1"})
    # B1 scored 16 and, separately, conceded 15 as a bowler; nothing is added together
    assert "B1, all seasons. BATTING: scored 16 runs in 1 innings off 5 balls" in b1
    assert "BOWLING: took 1 wickets in 1 matches, bowled 1.0 overs and conceded 15 runs" in b1
    assert "31" not in b1  # 16 + 15
    assert "BOWLING: did not bowl." in tool.invoke({"player": "A2"})
    assert "No player matching" in tool.invoke({"player": "Nobody Atall"})


@pytest.mark.parametrize("season", ["2024", 2024, "2024 season", "IPL 2024"])
def test_player_stats_accepts_season_formats(ctx, season):
    tool = {t.name: t for t in build_tools(ctx)}["player_stats"]
    assert "B1, the 2024 season. BATTING: scored 16 runs" in tool.invoke(
        {"player": "B1", "season": season}
    )


def test_split_seasons_match_their_first_year():
    from cricket_ai.agent.tools import _season_clause, normalise_season

    assert normalise_season(2020) == "2020" and normalise_season(None) is None
    clause, params = _season_clause("2020")
    assert params == ["2020", "2020/%"] and "LIKE" in clause
    assert _season_clause("2007/08") == (" AND season = ?", ["2007/08"])


def test_leaderboard(ctx):
    tool = {t.name: t for t in build_tools(ctx)}["leaderboard"]
    out = tool.invoke({"stat": "runs", "season": "2024", "n": 3})
    assert out.startswith("Top 3 by runs scored in the 2024 season")
    assert "1. B1: 16" in out
    top = ctx.sql.con.execute(
        "SELECT player FROM batting GROUP BY player ORDER BY SUM(runs) DESC LIMIT 1"
    ).fetchone()[0]
    assert f"1. {top}:" in tool.invoke({"stat": "runs"})
    assert "Unknown stat" in tool.invoke({"stat": "catches"})
    assert "economy" in tool.invoke({"stat": "economy", "season": 2024})


def test_tool_errors_are_returned_not_raised(ctx):
    sql_tool = {t.name: t for t in build_tools(ctx)}["run_sql"]
    out = sql_tool.invoke({"query": "DROP TABLE matches"})
    assert out.startswith("Error: UnsafeQueryError")
    assert ctx.calls[-1].ok is False


def test_agent_answers_with_a_tool(ctx):
    llm = scripted([("player_stats", {"player": "B1"})], "B1 scored 16 runs (player_stats).")
    r = CricketAgent(llm, ctx).ask("How many runs did B1 score?")
    assert r.answer.startswith("B1 scored 16") and r.tools_used == ["player_stats"]
    assert r.steps == 1 and r.tool_calls[0]["ok"] and not r.stopped_by_limit


def test_agent_recovers_from_unknown_tool_and_bad_sql(ctx):
    llm = scripted(
        [("make_tea", {}), ("run_sql", {"query": "SELECT nope FROM batting"})],
        [("run_sql", {"query": "SELECT COUNT(*) AS n FROM matches"})],
        "There are 31 matches (run_sql).",
    )
    r = CricketAgent(llm, ctx).ask("How many matches?")
    assert r.steps == 2 and r.answer.startswith("There are 31")
    assert [c["ok"] for c in r.tool_calls] == [False, True]


def test_step_limit_stops_loops(ctx):
    llm = scripted(*[[("search_cricket", {"query": "Team A"})]] * 5)
    r = CricketAgent(llm, ctx, max_steps=2).ask("loop forever")
    assert r.stopped_by_limit and r.steps == 2


def test_human_approval_can_decline(ctx, monkeypatch):
    from cricket_ai.agent import graph, tools

    monkeypatch.setattr(graph, "EXPENSIVE_TOOLS", {"search_cricket"})
    monkeypatch.setattr(tools, "EXPENSIVE_TOOLS", {"search_cricket"})
    asked = []
    llm = scripted([("search_cricket", {"query": "Team A"})], "OK, without searching.")
    agent = CricketAgent(llm, ctx, approver=lambda n, a: asked.append(n) or False)
    r = agent.ask("Search for Team A")
    assert asked == ["search_cricket"] and r.tool_calls == []  # declined, so never ran
    assert r.answer == "OK, without searching."


def test_agent_needs_llm_and_tools(ctx):
    with pytest.raises(ValueError):
        CricketAgent(None, ctx)
    with pytest.raises(ValueError):
        CricketAgent(scripted("x"), CricketContext())


def test_graph_diagram(ctx):
    mermaid = CricketAgent(scripted("x"), ctx).mermaid()
    assert "agent" in mermaid and "tools" in mermaid and "limit" in mermaid


def test_task_generation_and_scoring(ctx, tmp_path):
    tasks = generate_tasks(ctx, per_type=2)
    assert {t["type"] for t in tasks} == {"season_runs", "top_scorer", "match_winner"}
    first = next(t for t in tasks if t["type"] == "season_runs")
    good = score_task(
        first,
        f"They scored {first['expected']} runs",
        ["player_stats"],
        [f"BATTING: scored {first['expected']} runs"],
    )
    assert good["right_tool"] and good["faithful"] and good["correct"]
    top = next(t for t in tasks if t["type"] == "top_scorer")
    assert not score_task(top, "No idea", ["player_stats"])["right_tool"]  # cannot rank players
    chase = {
        "type": "chase",
        "expected": None,
        "question": "A team chasing 180 is 100/2 after 12 overs. What is their chance?",
    }
    tool_out = "Chasing 180: 100/2 after 12.0 overs needs 80 off 48 balls. Win probability: 83.9%"
    assert score_task(chase, "They have an 83.9% chance.", ["win_probability"], [tool_out])[
        "correct"
    ]
    # the run captured on 2026-11-04: right tool, right %, but misquoted the score as 120/2
    misquote = score_task(
        chase, "They are 120/2 with an 83.9% chance.", ["win_probability"], [tool_out]
    )
    assert (
        not misquote["faithful"]
        and not misquote["correct"]
        and misquote["invented_numbers"] == "120"
    )
    assert not score_task(chase, "likely", ["win_probability"], [tool_out])["correct"]

    task = [t for t in tasks if t["type"] == "season_runs"][:1]
    llm = scripted(
        [("run_sql", {"query": f"SELECT {task[0]['expected']} AS runs"})],
        f"{task[0]['expected']} runs, from run_sql.",
    )
    results = evaluate_agent(CricketAgent(llm, ctx), task)
    summary = write_report(results, tmp_path, "scripted")
    assert summary["tool_accuracy"] == 1.0 and summary["answer_accuracy"] == 1.0
    assert summary["faithfulness"] == 1.0
    assert (tmp_path / "report.md").exists()


def test_trace_file(ctx, tmp_path):
    llm = scripted([("player_stats", {"player": "B1"})], "16 runs.")
    r = CricketAgent(llm, ctx).ask("B1 runs?")
    path = write_trace(r, tmp_path)
    line = json.loads(path.read_text().splitlines()[-1])
    assert line["tool_calls"][0]["name"] == "player_stats" and line["answer"] == "16 runs."


def test_win_probability_tool(ctx):
    pytest.importorskip("torch")
    from cricket_ai.agent.tools import win_probability
    from cricket_ai.winprob import model as mlp
    from cricket_ai.winprob.data import date_split

    states = __import__("cricket_ai.data.features", fromlist=["chase_states"]).chase_states(
        ctx.sql.con.execute("SELECT * FROM matches").df(),
        ctx.sql.con.execute("SELECT * FROM deliveries").df(),
    )
    split = date_split(states)
    model, _ = mlp.train_mlp(split.train, split.val, mlp.TrainConfig(max_epochs=3, device="cpu"))
    ctx.winprob_model, ctx.latest_par_rpo = model, 8.0
    easy = win_probability(ctx, target=150, runs_scored=145, balls_bowled=100, wickets_lost=1)
    hard = win_probability(ctx, target=220, runs_scored=60, balls_bowled=90, wickets_lost=7)
    pct = [float(s.split("side: ")[1].split("%")[0]) for s in (easy, hard)]
    assert pct[0] > pct[1]
    assert "100%" in win_probability(ctx, 150, 150, 100, 1)
    assert "win_probability" in [t.name for t in build_tools(ctx)]
