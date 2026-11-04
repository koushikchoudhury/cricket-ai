"""Parse one Cricsheet JSON match into flat rows.

Produces three record sets per match:

* ``match``      - one row of match metadata and outcome
* ``deliveries`` - one row per delivery (legal or not), with running innings totals
* ``players``    - one row per player in each team's XI, with the Cricsheet registry id
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXTRA_KEYS = ("wides", "noballs", "byes", "legbyes", "penalty")

# Dismissals that are not credited to the bowler.
NON_BOWLER_DISMISSALS = frozenset(
    {
        "run out",
        "retired hurt",
        "retired out",
        "retired not out",
        "obstructing the field",
        "timed out",
        "handled the ball",
    }
)
# Dismissals that do not cost the batting side a wicket.
NOT_A_TEAM_WICKET = frozenset({"retired hurt", "retired not out"})


@dataclass
class ParsedMatch:
    match: dict[str, Any]
    deliveries: list[dict[str, Any]] = field(default_factory=list)
    players: list[dict[str, Any]] = field(default_factory=list)


def overs_to_balls(overs: float | int | None, balls_per_over: int) -> int | None:
    """Convert cricket overs notation (e.g. 18.4) to balls."""
    if overs is None:
        return None
    whole = int(overs)
    part = round((float(overs) - whole) * 10)
    return whole * balls_per_over + part


def parse_match(path: str | Path) -> ParsedMatch:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_match_dict(data, match_id=path.stem)


def _outcome(info: dict[str, Any]) -> dict[str, Any]:
    outcome = info.get("outcome", {}) or {}
    by = outcome.get("by", {}) or {}
    winner = outcome.get("winner") or outcome.get("eliminator")
    result = outcome.get("result")
    if result is None:
        result = "win" if winner else "unknown"
    if outcome.get("eliminator"):
        result = "tie_super_over"
    return {
        "winner": winner,
        "result": result,
        "win_by_runs": by.get("runs"),
        "win_by_wickets": by.get("wickets"),
        "win_by_innings": by.get("innings"),
        "method": outcome.get("method"),
    }


def parse_match_dict(data: dict[str, Any], match_id: str) -> ParsedMatch:
    info = data["info"]
    teams: list[str] = list(info.get("teams", []))
    balls_per_over = int(info.get("balls_per_over", 6))
    event = info.get("event", {}) or {}
    toss = info.get("toss", {}) or {}
    dates = info.get("dates", []) or []
    innings_list = data.get("innings", []) or []

    target = next((inn["target"] for inn in innings_list if "target" in inn), None) or {}
    pom = info.get("player_of_match") or []

    match: dict[str, Any] = {
        "match_id": match_id,
        "data_version": data.get("meta", {}).get("data_version"),
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
        "team_type": info.get("team_type"),
        "event_name": event.get("name"),
        "match_number": event.get("match_number"),
        # Playoff stage, e.g. "Final", "Qualifier 2", "Eliminator"; None for league games.
        "event_stage": event.get("stage"),
        "season": str(info["season"]) if "season" in info else None,
        "start_date": dates[0] if dates else None,
        "city": info.get("city"),
        "venue": info.get("venue"),
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "overs": info.get("overs"),
        "balls_per_over": balls_per_over,
        "target_runs": target.get("runs"),
        "target_overs": target.get("overs"),
        "player_of_match": ",".join(pom) if pom else None,
        **_outcome(info),
    }

    parsed = ParsedMatch(match=match)

    registry = (info.get("registry", {}) or {}).get("people", {}) or {}
    for team, names in (info.get("players", {}) or {}).items():
        for name in names:
            parsed.players.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "player": name,
                    "player_id": registry.get(name),
                }
            )

    for innings_no, inn in enumerate(innings_list, start=1):
        batting = inn["team"]
        bowling = next((t for t in teams if t != batting), None)
        super_over = bool(inn.get("super_over", False))
        team_runs = 0
        team_wickets = 0
        legal_balls = 0
        seq = 0
        for over in inn.get("overs", []):
            over_no = int(over["over"])
            ball_in_over = 0
            for d in over.get("deliveries", []):
                seq += 1
                extras = d.get("extras", {}) or {}
                runs = d["runs"]
                is_legal = "wides" not in extras and "noballs" not in extras
                if is_legal:
                    ball_in_over += 1
                    legal_balls += 1

                wickets = d.get("wickets", []) or []
                kinds = [w.get("kind") for w in wickets]
                team_wkts_this_ball = sum(k not in NOT_A_TEAM_WICKET for k in kinds)
                bowler_wkts_this_ball = sum(k not in NON_BOWLER_DISMISSALS for k in kinds)
                team_runs += int(runs["total"])
                team_wickets += team_wkts_this_ball

                first = wickets[0] if wickets else {}
                fielders = [f.get("name") for f in first.get("fielders", []) if f.get("name")]

                row = {
                    "match_id": match_id,
                    "innings": innings_no,
                    "is_super_over": super_over,
                    "batting_team": batting,
                    "bowling_team": bowling,
                    "over": over_no,
                    "ball_in_over": ball_in_over,
                    "delivery_seq": seq,
                    "batter": d.get("batter"),
                    "non_striker": d.get("non_striker"),
                    "bowler": d.get("bowler"),
                    "runs_batter": int(runs.get("batter", 0)),
                    "runs_extras": int(runs.get("extras", 0)),
                    "runs_total": int(runs.get("total", 0)),
                    **{k: int(extras.get(k, 0)) for k in EXTRA_KEYS},
                    "is_legal": is_legal,
                    "is_wicket": bool(wickets),
                    "wickets_on_ball": len(wickets),
                    "bowler_wickets": bowler_wkts_this_ball,
                    "player_out": first.get("player_out"),
                    "wicket_kind": first.get("kind"),
                    "fielders": ",".join(fielders) if fielders else None,
                    "team_runs": team_runs,
                    "team_wickets": team_wickets,
                    "legal_balls": legal_balls,
                }
                parsed.deliveries.append(row)
    return parsed
