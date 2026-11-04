import json

import numpy as np
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from matchgen import generate_match  # noqa: E402

from cricket_ai.data.build import build_tables  # noqa: E402
from cricket_ai.llm import get_chat_model, llm_settings, message_text  # noqa: E402
from cricket_ai.rag.corpus import (  # noqa: E402
    build_corpus,
    fmt_overs,
    load_documents,
    save_documents,
)
from cricket_ai.rag.evaluate import (  # noqa: E402
    answer_correct,
    evaluate_answers,
    evaluate_retrieval,
    generate_questions,
    summarise_retrieval,
    write_report,
)
from cricket_ai.rag.images import ImageIndex  # noqa: E402
from cricket_ai.rag.pipeline import CricketRAG, extract_sql, route_question  # noqa: E402
from cricket_ai.rag.retrieval import BM25, HashingEmbedder, HybridRetriever, rrf  # noqa: E402
from cricket_ai.rag.sql import CricketSQL, UnsafeQueryError, check_query  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    for seed in range(40):
        (raw / f"g{seed:03d}.json").write_text(json.dumps(generate_match(seed, day_step=9)))
    fixture = __import__("pathlib").Path(__file__).parent / "fixtures" / "tiny_match.json"
    (raw / "tiny_match.json").write_text(fixture.read_text())
    return build_tables(raw, workers=1)


@pytest.fixture(scope="module")
def docs(built):
    return build_corpus(built.matches, built.deliveries)


@pytest.fixture(scope="module")
def retriever(docs):
    return HybridRetriever(docs, HashingEmbedder())


@pytest.fixture(scope="module")
def sql(built):
    return CricketSQL(built.matches, built.deliveries, built.players)


def doc(docs, doc_id):
    return next(d for d in docs if d.doc_id == doc_id)


# ----- corpus -----
def test_match_report_facts(docs):
    text = doc(docs, "match:tiny_match").text
    assert "Team A 24/2 (2.0 overs); Team B 25/1 (1.4 overs)" in text
    assert "Team B won by 9 wickets" in text
    assert "Player of the match: B2" in text
    assert "B1 16 (5)" in text  # top scorer
    assert "B1 1/15 (1.0 overs)" in text  # leg-bye not charged, run out not credited


def test_playoff_stage_is_searchable(built):
    from cricket_ai.data.parse import parse_match_dict
    from cricket_ai.rag.corpus import match_documents

    fixture = __import__("pathlib").Path(__file__).parent / "fixtures" / "tiny_match.json"
    data = json.loads(fixture.read_text())
    data["info"]["event"] = {"name": "Fixture League", "stage": "Final"}
    parsed = parse_match_dict(data, "final_match")
    assert parsed.match["event_stage"] == "Final"

    import pandas as pd

    m = pd.DataFrame([parsed.match])
    m["start_date"] = pd.to_datetime(m.start_date)
    d = pd.DataFrame(parsed.deliveries)
    final = match_documents(m, d)[0]
    assert "Fixture League 2024, Final: Team A v Team B" in final.text
    assert final.title.endswith("(Final)") and final.metadata["stage"] == "Final"

    docs = build_corpus(built.matches, built.deliveries) + [final]
    hits = HybridRetriever(docs, HashingEmbedder()).search("Who won the final in 2024?", k=1)
    assert hits[0].doc.doc_id == "match:final_match"


def test_phase_and_player_documents(docs):
    phase = doc(docs, "phase:tiny_match:1:powerplay")
    assert "24 runs for 2 wickets off 12 balls" in phase.text
    assert "A1 caught off B1" in phase.text and "A2 run out" in phase.text
    player = doc(docs, "player:B1:2024")
    assert "16 runs in 1 innings" in player.text and "1 wickets" in player.text
    assert player.metadata["runs"] == 16


def test_document_roundtrip(docs, tmp_path):
    save_documents(docs[:5], tmp_path / "d.jsonl")
    assert [d.doc_id for d in load_documents(tmp_path / "d.jsonl")] == [d.doc_id for d in docs[:5]]
    assert fmt_overs(10) == "1.4"


# ----- retrieval -----
def test_bm25_prefers_exact_terms():
    bm = BM25().fit(["kohli scored a hundred", "bumrah took five wickets", "rain stopped play"])
    s = bm.scores("bumrah wickets")
    assert s.argmax() == 1 and s[2] == 0


def test_exact_dates_count_in_keyword_search():
    from cricket_ai.rag.retrieval import with_date_tokens

    assert with_date_tokens("on 2 April 2020.").endswith("d_2_april_2020")
    assert with_date_tokens("2nd april, 2020").endswith("d_2_april_2020")
    texts = ["Kolkata v Mumbai on 27 April 2020, 117/2", "Kolkata v Mumbai on 2 April 2020"]
    bm = BM25().fit([with_date_tokens(t) for t in texts])
    assert bm.scores(with_date_tokens("match on 2 April 2020")).argmax() == 1


def test_query_intent():
    from cricket_ai.rag.retrieval import intent_kinds

    assert intent_kinds("Who won the final in 2016?") == ["match"]
    assert intent_kinds("Runs in the death overs") == ["innings_phase"]
    assert intent_kinds("Kohli's strike rate in the 2016 season") == ["player_season"]
    assert intent_kinds("Tell me about Eden Gardens") == []


def test_rrf_rewards_agreement():
    fused = rrf([np.array([0, 1, 2]), np.array([1, 0, 3])])
    assert fused[0] == fused[1] > fused[2] > 0


def test_hybrid_search_finds_the_right_match(retriever):
    q = "Who was player of the match at Eden Gardens on 10 April 2024?"
    for method in ("bm25", "dense", "hybrid"):
        assert retriever.search(q, k=3, method=method)[0].doc.doc_id == "match:tiny_match"
    only_players = retriever.search("B2", k=5, kinds=["player_season"])
    assert {h.doc.kind for h in only_players} == {"player_season"}


def test_retriever_save_load(retriever, tmp_path):
    retriever.save(tmp_path / "idx")
    loaded = HybridRetriever.load(tmp_path / "idx")
    q = "Team A powerplay"
    assert [h.doc.doc_id for h in loaded.search(q)] == [h.doc.doc_id for h in retriever.search(q)]


# ----- SQL -----
def test_sql_answers_numbers(sql):
    df = sql.run("SELECT player, SUM(runs) AS runs FROM batting WHERE player = 'B1' GROUP BY 1")
    assert int(df.runs.iloc[0]) == 16
    wk = sql.run("SELECT SUM(wickets) AS w FROM bowling WHERE player = 'B2'")
    assert int(wk.w.iloc[0]) == 0  # the run out is not B2's wicket
    assert len(sql.run("SELECT * FROM deliveries", limit=7)) == 7


@pytest.mark.parametrize(
    "bad",
    [
        "DROP TABLE matches",
        "SELECT 1; DROP TABLE matches",
        "DELETE FROM matches",
        "WITH x AS (SELECT 1) INSERT INTO matches SELECT * FROM x",
        "COPY matches TO 'out.csv'",
        "",
    ],
)
def test_sql_guard_blocks_writes(bad):
    with pytest.raises(UnsafeQueryError):
        check_query(bad)


def test_sql_cannot_read_files(sql):
    with pytest.raises(Exception, match="(?i)permission|disabled|access"):
        sql.run("SELECT * FROM read_csv('/etc/hostname')")


def test_sql_guard_allows_keywords_inside_strings(sql):
    assert check_query("SELECT 'drop' AS x -- comment\n;") == "SELECT 'drop' AS x"
    assert sql.run("SELECT 'load' AS x").x.iloc[0] == "load"


# ----- pipeline -----
def fake_llm(*replies):
    return GenericFakeChatModel(messages=iter([AIMessage(r) for r in replies]))


def test_routing_and_sql_extraction():
    assert route_question("How many runs did Kohli score in 2016?") == "sql"
    assert route_question("Who was the top scorer?") == "sql"
    assert route_question("Who won the final?") == "search"
    assert extract_sql("Here:\n```sql\nSELECT 1\n```") == "SELECT 1"


def test_pipeline_sql_route(retriever, sql):
    llm = fake_llm(
        "```sql\nSELECT SUM(runs) AS runs FROM batting WHERE player = 'B1'\n```",
        "B1 scored 16 runs, according to the database.",
    )
    a = CricketRAG(retriever, sql, llm).ask("How many runs did B1 score?")
    assert a.route == "sql" and a.error is None
    assert int(a.rows[0]["runs"]) == 16
    assert "16" in a.text and a.sources


def test_pipeline_retries_bad_sql(retriever, sql):
    llm = fake_llm(
        "```sql\nSELECT nope FROM batting\n```",
        "```sql\nSELECT COUNT(*) AS n FROM matches\n```",
        "There are 41 matches.",
    )
    a = CricketRAG(retriever, sql, llm).ask("How many matches are there?")
    assert a.error is None and a.rows[0]["n"] == 41


def test_pipeline_without_llm(retriever, sql):
    a = CricketRAG(retriever, sql, None).ask("player of the match Eden Gardens 10 April 2024")
    assert a.route == "search" and "No LLM configured" in a.text
    assert a.sources[0]["doc_id"] == "match:tiny_match"


# ----- evaluation -----
def test_generated_questions_and_retrieval_eval(docs, retriever, tmp_path):
    qs = generate_questions(docs, per_type=10)
    types = {q["type"] for q in qs}
    assert {"match_winner", "phase_runs", "season_runs", "season_wickets"} <= types
    assert all(q["gold"] in {d.doc_id for d in docs} for q in qs)
    overall, by_type = summarise_retrieval(evaluate_retrieval(retriever, qs))
    assert set(overall.index) == {"bm25", "dense", "hybrid"}
    assert overall.loc["hybrid", "recall@10"] >= 0.8
    write_report(
        tmp_path,
        overall,
        by_type,
        None,
        {"n_questions": len(qs), "n_docs": len(docs), "embedder": "hashing"},
    )
    assert (tmp_path / "report.md").exists()


def test_answer_scoring(retriever, sql):
    assert answer_correct("V Kohli", "Virat Kohli was player of the match")
    assert answer_correct("52", "They scored 52 runs.")
    assert not answer_correct("52", "They scored 152 runs.")
    q = [
        {
            "type": "season_runs",
            "expected": "16",
            "gold": "player:B1:2024",
            "question": "How many runs did B1 score in the 2024 season?",
        }
    ]
    llm = fake_llm("```sql\nSELECT 16 AS runs\n```", "B1 scored 16 runs.")
    out = evaluate_answers(CricketRAG(retriever, sql, llm), q)
    assert out.correct.all()


# ----- LLM switch and images -----
def test_llm_settings(monkeypatch):
    monkeypatch.delenv("CRICKET_AI_LLM", raising=False)
    monkeypatch.delenv("CRICKET_AI_LLM_MODEL", raising=False)
    assert llm_settings() == ("none", None)
    assert get_chat_model() is None
    monkeypatch.setenv("CRICKET_AI_LLM", "ollama")
    assert llm_settings() == ("ollama", "qwen2.5:3b")
    with pytest.raises(ValueError):
        llm_settings("gemini")
    assert message_text(AIMessage([{"type": "text", "text": "hi"}])) == "hi"


class FakeClip:
    """Maps the word 'six' and 'wicket' to two directions, like a tiny CLIP."""

    def _vec(self, s):
        return np.array([s.count("six"), s.count("wicket"), 0.1], dtype=np.float32)

    def embed_images(self, paths):
        v = np.stack([self._vec(p.stem) for p in paths])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_texts(self, texts):
        v = np.stack([self._vec(t) for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_image_index(tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for name in ("big_six", "wicket_falls", "crowd"):
        (folder / f"{name}.jpg").write_bytes(b"not-a-real-image")
    (folder / "captions.json").write_text(json.dumps({"big_six.jpg": "Six over long-on"}))
    index = ImageIndex.build(folder, FakeClip())
    top = index.search("a huge six", k=1)[0]
    assert top.path.endswith("big_six.jpg") and top.caption == "Six over long-on"
    index.save(tmp_path / "idx")
    again = ImageIndex.load(tmp_path / "idx", FakeClip())
    assert again.search("wicket", k=1)[0].path.endswith("wicket_falls.jpg")
