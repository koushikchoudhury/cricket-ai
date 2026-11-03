import pandas as pd

from cricket_ai.data.parse import overs_to_balls, parse_match


def test_match_metadata(tiny_match_path):
    m = parse_match(tiny_match_path).match
    assert m["match_id"] == "tiny_match"
    assert m["winner"] == "Team B" and m["result"] == "win" and m["win_by_wickets"] == 9
    assert m["target_runs"] == 25 and m["target_overs"] == 2
    assert m["season"] == "2024" and m["venue"] == "Eden Gardens"
    assert m["toss_winner"] == "Team B" and m["toss_decision"] == "field"


def test_first_innings_totals(tiny_match_path):
    d = pd.DataFrame(parse_match(tiny_match_path).deliveries)
    inn1 = d[d.innings == 1]
    assert len(inn1) == 14
    assert inn1.is_legal.sum() == 12
    assert inn1.runs_total.sum() == 24
    assert inn1.iloc[-1].team_runs == 24
    assert inn1.iloc[-1].team_wickets == 2
    assert inn1.iloc[-1].legal_balls == 12
    assert (inn1.wides.sum(), inn1.noballs.sum(), inn1.legbyes.sum()) == (1, 1, 1)


def test_ball_numbering_skips_illegal(tiny_match_path):
    d = pd.DataFrame(parse_match(tiny_match_path).deliveries)
    over0 = d[(d.innings == 1) & (d.over == 0)]
    # wide is delivery 2: same ball_in_over as the ball before it
    assert over0.ball_in_over.tolist() == [1, 1, 2, 3, 4, 5, 5, 6]


def test_wicket_credit(tiny_match_path):
    d = pd.DataFrame(parse_match(tiny_match_path).deliveries)
    caught = d[d.player_out == "A1"].iloc[0]
    assert caught.wicket_kind == "caught" and caught.fielders == "B2" and caught.bowler_wickets == 1
    run_out = d[d.player_out == "A2"].iloc[0]
    assert run_out.wicket_kind == "run out" and run_out.bowler_wickets == 0


def test_players_registry(tiny_match_path):
    players = pd.DataFrame(parse_match(tiny_match_path).players)
    assert len(players) == 10
    assert players.set_index("player").loc["B2", "player_id"] == "b2"


def test_overs_to_balls():
    assert overs_to_balls(20, 6) == 120
    assert overs_to_balls(18.4, 6) == 112
    assert overs_to_balls(None, 6) is None
