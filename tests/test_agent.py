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
    assert "Top 3 by runs scored in the 2024 season" in out
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
        "expected": "83.9%",
        "expected_args": {"target": 180, "runs_scored": 100, "overs": 12, "wickets_lost": 2},
        "question": "A team chasing 180 is 100/2 after 12 overs. What is their chance?",
    }
    good_args = [{"target": 180, "runs_scored": 100, "overs": 12, "wickets_lost": 2}]
    tool_out = "Situation: chasing 180, 100/2 after 12 overs. Needs 80 runs off 48 balls. 83.9%."
    ok = score_task(chase, "They have an 83.9% chance.", ["win_probability"], [tool_out], good_args)
    assert ok["correct"] and ok["args_ok"]
    assert score_task(chase, "About 84%.", ["win_probability"], [tool_out], good_args)["correct"]
    # run 1, 2026-11-04: right tool, right %, but misquoted the score as 120/2
    misquote = score_task(
        chase, "They are 120/2 with an 83.9% chance.", ["win_probability"], [tool_out], good_args
    )
    assert not misquote["faithful"] and not misquote["correct"]
    assert misquote["invented_numbers"] == "120"
    # run 2, 2026-11-04: the model called the tool with runs_scored=60. The answer faithfully
    # repeats the (wrong) tool output, so only the argument check and ground truth catch it.
    wrong_out = "Situation: chasing 180, 60/2 after 12 overs. Needs 120 runs off 48 balls. 30.3%."
    wrong = score_task(
        chase,
        "The team is at 60/2 after 12 overs, needs 120 off 48 balls: 30.3%.",
        ["win_probability"],
        [wrong_out],
        [{"target": 180, "runs_scored": 60, "overs": 12, "wickets_lost": 2}],
    )
    assert wrong["faithful"] and not wrong["args_ok"] and not wrong["correct"]
    assert not score_task(chase, "likely", ["win_probability"], [tool_out], good_args)["correct"]

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
    easy = win_probability(ctx, target=150, runs_scored=145, overs=16.4, wickets_lost=1)
    hard = win_probability(ctx, target=220, runs_scored=60, overs=15, wickets_lost=7)
    pct = [float(s.split("side: ")[1].split("%")[0]) for s in (easy, hard)]
    assert pct[0] > pct[1]
    assert "100%" in win_probability(ctx, 150, 150, 16.4, 1)
    # run 2 passed 18 (meaning overs) where balls were expected; overs are now the input,
    # and a ball count is rejected instead of silently giving "102 balls left"
    assert "Needs 10 runs off 12 balls" in win_probability(ctx, 150, 140, 18, 3)
    assert win_probability(ctx, 150, 140, 108, 3).startswith("Error: overs=108")
    assert "win_probability" in [t.name for t in build_tools(ctx)]


def test_overs_notation():
    from cricket_ai.agent.tools import overs_to_balls

    assert overs_to_balls(12) == 72 and overs_to_balls("12.3") == 75 and overs_to_balls(0.5) == 5
    with pytest.raises(ValueError):
        overs_to_balls("12.7")


def test_player_name_falls_back_to_surname(ctx):
    # run 2 (DA Warner): models expand initials into full names; data stores initials
    tool = {t.name: t for t in build_tools(ctx)}["player_stats"]
    out = tool.invoke({"player": "Someone Kol_1"})
    assert out.startswith("No exact match for 'Someone Kol_1'; matched by surname 'Kol_1'.")
    assert "Kol_1, all seasons. BATTING" in out
    assert tool.invoke({"player": "Zz"}).startswith("No player matching")


def test_leaderboard_leads_with_the_answer(ctx):
    tool = {t.name: t for t in build_tools(ctx)}["leaderboard"]
    out = tool.invoke({"stat": "runs", "season": "2024"})
    assert out.splitlines()[0] == "Leader: B1, 16 runs scored in the 2024 season."


def test_repeats_report_mean_range_and_flaky_tasks(ctx, tmp_path):
    task = [t for t in generate_tasks(ctx, per_type=1) if t["type"] == "season_runs"]
    runs_expected = task[0]["expected"]
    call = [("player_stats", {"player": task[0]["question"].split(" did ")[1].split(" score")[0]})]
    # same task twice: right in run 1, wrong in run 2 (as local models do)
    llm = scripted(call, f"{runs_expected} runs.", call, "No idea.")
    results = evaluate_agent(CricketAgent(llm, ctx), task, repeats=2)
    assert list(results.run) == [1, 2] and list(results.correct) == [True, False]
    summary = write_report(results, tmp_path, "scripted")
    assert summary["runs"] == 2 and summary["tasks"] == 1
    assert summary["answer_accuracy"] == 0.5
    assert (summary["answer_accuracy_min"], summary["answer_accuracy_max"]) == (0.0, 1.0)
    assert summary["consistency"] == 0.0
    report = (tmp_path / "report.md").read_text()
    assert "range 0%-100% over 2 runs" in report and "changed verdict" in report


class Recording(Scripted):
    """Scripted model that remembers what it was sent."""

    seen: list = []

    def invoke(self, messages, *args, **kwargs):
        type(self).seen.append([(m.type, m.content) for m in messages])
        return super().invoke(messages, *args, **kwargs)


def test_follow_up_questions_get_the_conversation(ctx):
    Recording.seen = []
    llm = Recording(messages=iter([AIMessage("Kohli did well in 2020.")]))
    history = [
        {"role": "user", "content": "How did JJ Bumrah bowl in the 2020 season?"},
        {"role": "assistant", "content": "He took 27 wickets."},
    ]
    CricketAgent(llm, ctx).ask("how about kohli?", history=history)
    sent = Recording.seen[0]
    assert [kind for kind, _ in sent] == ["system", "human", "ai", "human"]
    assert "2020 season" in sent[1][1] and sent[3][1] == "how about kohli?"
    assert "keeps the same season" in sent[0][1]


def test_progress_events_and_timing(ctx):
    events = []
    llm = scripted([("player_stats", {"player": "B1"})], "16 runs.")
    r = CricketAgent(llm, ctx).ask("B1 runs?", on_event=events.append)
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "writing"]
    assert events[0]["name"] == "player_stats" and events[0]["args"] == {"player": "B1"}
    assert len(r.llm_calls) == 2 and r.steps == 1
    assert r.timing.startswith("model ") and "in 2 call(s)" in r.timing and "tools" in r.timing


def test_player_stats_shows_one_player_and_names_the_rest(ctx):
    tool = {t.name: t for t in build_tools(ctx)}["player_stats"]
    out = tool.invoke({"player": "Kol"})
    assert out.count("BATTING:") == 1  # one player in full, not three
    assert "Other players matching 'Kol':" in out and "Ask by full name" in out
    exact = tool.invoke({"player": "Kol_1"})
    assert exact.startswith("Kol_1, all seasons.")  # exact name wins over Kol_10, Kol_11


def test_agent_prompt_is_compact(ctx):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from cricket_ai.agent.graph import SYSTEM_PROMPT

    size = len(SYSTEM_PROMPT) + sum(
        len(json.dumps(convert_to_openai_tool(t))) for t in build_tools(ctx)
    )
    assert size < 4000  # sent with every model call; was ~5,300 before the SQL hint was cut
