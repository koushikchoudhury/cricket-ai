import io
import zipfile

import pandas as pd
import pytest

from cricket_ai.analytics import (
    batting_summary,
    bowling_summary,
    phase_summary,
    team_season_summary,
)
from cricket_ai.data.build import build_tables, load_tables, write_tables
from cricket_ai.data.download import dataset_url, safe_extract
from cricket_ai.data.features import CHASE_FEATURES, chase_states, split_by_date


@pytest.fixture
def built(raw_dir):
    return build_tables(raw_dir, workers=1)


def test_build_records_broken_files_and_validates(built):
    assert len(built.matches) == 41
    assert built.report.errors["unparseable_files"] == 1
    other = {k: v for k, v in built.report.errors.items() if k != "unparseable_files"}
    assert not any(other.values()), other


def test_roundtrip_parquet(built, tmp_path):
    paths = write_tables(built, tmp_path / "processed")
    assert paths["quality_report"].exists()
    matches, deliveries, players = load_tables(tmp_path / "processed")
    assert len(deliveries) == len(built.deliveries)
    assert deliveries["innings"].dtype == "int16"


def test_chase_states_tiny(built):
    m = built.matches[built.matches.match_id == "tiny_match"]
    d = built.deliveries[built.deliveries.match_id == "tiny_match"]
    s = chase_states(m, d)
    # 10 second-innings deliveries; the winning ball is a terminal state and is dropped
    assert len(s) == 9
    first = s.iloc[0]
    assert (first.runs_needed, first.balls_left, first.wickets_in_hand) == (19, 11, 10)
    assert first.target == 25
    assert s.chasing_team_won.eq(1).all()
    assert s.iloc[4].wickets_in_hand == 9  # after the bowled dismissal


def test_chase_states_full(built):
    s = chase_states(built.matches, built.deliveries)
    assert set(CHASE_FEATURES).issubset(s.columns)
    assert s[CHASE_FEATURES].notna().all().all()
    assert (s.runs_needed > 0).all() and (s.balls_left > 0).all()
    assert s.required_run_rate.le(36).all()
    # ties are excluded; both labels should be present across 40 generated matches
    assert set(s.chasing_team_won.unique()) == {0, 1}
    train, test = split_by_date(s, "2020-04-30")
    assert set(train.match_id).isdisjoint(test.match_id)


def test_analytics_tiny(built):
    d = built.deliveries[built.deliveries.match_id == "tiny_match"]
    m = built.matches[built.matches.match_id == "tiny_match"]
    bowl = bowling_summary(d)
    # B1: batter runs 13 + 1 wide + 1 no-ball; leg-bye not charged
    assert bowl.loc["B1", "runs_conceded"] == 15
    assert bowl.loc["B1", "balls"] == 6 and bowl.loc["B1", "wickets"] == 1
    assert bowl.loc["B2", "wickets"] == 0  # run out not credited
    bat = batting_summary(d)
    assert bat.loc["A2", "balls"] == 3  # the wide does not count as a ball faced
    assert bat.loc["A2", "runs"] == 5 and bat.loc["A2", "dismissals"] == 1
    phases = phase_summary(d, m)
    assert phases.runs.sum() == 24 + 25
    seasons = team_season_summary(m)
    assert seasons.set_index("team").loc["Team B", "won"] == 1


def test_safe_extract_blocks_zip_slip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ok.json", "{}")
        zf.writestr("readme.txt", "x")
    p = tmp_path / "a.zip"
    p.write_bytes(buf.getvalue())
    out = safe_extract(p, tmp_path / "out")
    assert [f.name for f in out] == ["ok.json"]


def test_dataset_url_validation():
    assert dataset_url("ipl").endswith("/ipl_json.zip")
    with pytest.raises(ValueError):
        dataset_url("../etc")


def test_season_summary_handles_no_result():
    m = pd.DataFrame(
        [
            {
                "season": "2024",
                "team1": "X",
                "team2": "Y",
                "winner": None,
                "result": "no result",
                "toss_winner": "X",
            }
        ]
    )
    s = team_season_summary(m).set_index("team")
    assert s.loc["X", "no_result"] == 1 and pd.isna(s.loc["X", "win_pct"])
