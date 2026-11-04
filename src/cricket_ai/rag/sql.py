"""Read-only SQL over the cricket tables, for questions whose answer is a number.

LLMs are unreliable at arithmetic over retrieved text, so "how many / most / average"
questions are answered by SQL the LLM writes and DuckDB executes. Safety:

* tables are loaded into an in-memory database, then file and network access is switched
  off and the configuration locked, so a query cannot read or write anything else;
* only a single SELECT / WITH statement is accepted, with a denylist for anything else;
* every query is wrapped in a row limit.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from cricket_ai.rag.corpus import batting_innings, bowling_innings

SCHEMA = """\
Tables (DuckDB):

matches(match_id, season TEXT e.g. '2016' or '2007/08', start_date DATE, event_name, match_number,
        event_stage ('Final', 'Qualifier 1', 'Qualifier 2', 'Eliminator', or NULL for league games),
        city, venue, team1, team2, toss_winner, toss_decision ('bat'|'field'), winner,
        result ('win'|'tie'|'no result'|...), win_by_runs, win_by_wickets, method (e.g. 'D/L'),
        target_runs, player_of_match)

deliveries(match_id, innings (1 or 2; 3+ only in super overs), is_super_over BOOLEAN,
           batting_team, bowling_team, over (0-based: 0 is the first over), ball_in_over,
           batter, non_striker, bowler, runs_batter, runs_extras, runs_total, wides, noballs,
           byes, legbyes, is_legal BOOLEAN, is_wicket BOOLEAN, player_out, wicket_kind,
           bowler_wickets (wickets credited to the bowler), team_runs, team_wickets)

batting(match_id, season, start_date, event_name, innings, batting_team, player, runs, balls,
        wicket_kind, not_out BOOLEAN)          -- one row per batter per innings

bowling(match_id, season, start_date, event_name, innings, bowling_team, player, balls,
        runs_conceded, wickets)                -- one row per bowler per innings

Notes:
- Player names are stored with initials, e.g. 'V Kohli', 'JJ Bumrah'. Match names with
  ILIKE, e.g. player ILIKE '%Kohli%'.
- Phases by 0-based over: powerplay over <= 5, middle 6-14, death over >= 15 (T20).
- Exclude super overs from totals: WHERE NOT is_super_over.
- Strike rate = 100.0 * SUM(runs) / SUM(balls); economy = 6.0 * SUM(runs_conceded) / SUM(balls).
"""

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|"
    r"export|import|call|set|reset|checkpoint|vacuum|grant|revoke|truncate|replace)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    pass


def check_query(sql: str) -> str:
    """Return the cleaned query, or raise UnsafeQueryError."""
    q = re.sub(r"--[^\n]*", " ", sql)
    q = re.sub(r"/\*.*?\*/", " ", q, flags=re.DOTALL).strip().rstrip(";").strip()
    if not q:
        raise UnsafeQueryError("Empty query")
    if ";" in q:
        raise UnsafeQueryError("Only one statement is allowed")
    if not re.match(r"(?is)^(select|with)\b", q):
        raise UnsafeQueryError("Only SELECT queries are allowed")
    # Ignore string literals when looking for forbidden keywords (e.g. ILIKE '%Load%').
    bare = re.sub(r"'(?:[^']|'')*'", "''", q)
    if FORBIDDEN.search(bare):
        raise UnsafeQueryError(f"Forbidden keyword: {FORBIDDEN.search(bare).group(0)}")
    return q


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(str)
    return out


class CricketSQL:
    def __init__(self, matches: pd.DataFrame, deliveries: pd.DataFrame, players: pd.DataFrame):
        con = duckdb.connect(":memory:")
        meta = matches[["match_id", "season", "start_date", "event_name"]]
        bat = (
            batting_innings(deliveries)
            .rename(columns={"batter": "player"})
            .merge(meta, on="match_id")
        )
        bowl = (
            bowling_innings(deliveries)
            .rename(columns={"bowler": "player", "runs": "runs_conceded"})
            .merge(meta, on="match_id")
        )
        tables = {
            "matches": matches,
            "deliveries": deliveries,
            "players": players,
            "batting": bat[
                [
                    "match_id",
                    "season",
                    "start_date",
                    "event_name",
                    "innings",
                    "batting_team",
                    "player",
                    "runs",
                    "balls",
                    "wicket_kind",
                    "not_out",
                ]
            ],
            "bowling": bowl[
                [
                    "match_id",
                    "season",
                    "start_date",
                    "event_name",
                    "innings",
                    "bowling_team",
                    "player",
                    "balls",
                    "runs_conceded",
                    "wickets",
                ]
            ],
        }
        for name, df in tables.items():
            con.register(f"_{name}", _plain(df))
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
            con.unregister(f"_{name}")
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        self.con = con

    @classmethod
    def from_dir(cls, processed: Path) -> CricketSQL:
        return cls(
            pd.read_parquet(processed / "matches.parquet"),
            pd.read_parquet(processed / "deliveries.parquet"),
            pd.read_parquet(processed / "players.parquet"),
        )

    def run(self, sql: str, limit: int = 200) -> pd.DataFrame:
        q = check_query(sql)
        return self.con.execute(f"SELECT * FROM ({q}) AS _q LIMIT {int(limit)}").df()
