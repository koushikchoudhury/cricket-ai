"""Turn the ball-by-ball tables into a searchable text corpus.

Three kinds of document, each small enough to retrieve precisely:

* ``match``          - a match report: teams, venue, date, toss, scores, result,
                       player of the match, top scorers and best bowling.
* ``innings_phase``  - one innings split into powerplay, middle and death overs.
* ``player_season``  - a player's batting and bowling for one season.

Every document carries metadata (match id, season, teams, player) so answers can cite it
and the evaluation can check that the right document was retrieved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from cricket_ai.analytics.summaries import assign_phase
from cricket_ai.data.parse import NON_BOWLER_DISMISSALS

PHASE_LABEL = {"powerplay": "powerplay", "middle": "middle overs", "death": "death overs"}
PHASE_ORDER = ["powerplay", "middle", "death"]


@dataclass
class Document:
    doc_id: str
    kind: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def _s(x) -> str:
    """String value, or '' for None/NaN."""
    return x if isinstance(x, str) else ""


def fmt_date(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {ts:%B %Y}" if pd.notna(ts) else "unknown date"


def fmt_overs(legal_balls: int, balls_per_over: int = 6) -> str:
    return f"{legal_balls // balls_per_over}.{legal_balls % balls_per_over}"


def _regular(deliveries: pd.DataFrame) -> pd.DataFrame:
    return deliveries[~deliveries.is_super_over]


def batting_innings(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Runs, balls and dismissal for each batter in each innings."""
    d = _regular(deliveries)
    faced = d[d.wides == 0]
    bat = (
        faced.groupby(["match_id", "innings", "batter"], observed=True)
        .agg(runs=("runs_batter", "sum"), balls=("runs_batter", "size"))
        .reset_index()
    )
    # batters dismissed without facing a legal ball still appear via player_out
    outs = d[d.player_out.notna()][["match_id", "innings", "player_out", "wicket_kind"]]
    outs = outs.rename(columns={"player_out": "batter"}).drop_duplicates(
        ["match_id", "innings", "batter"]
    )
    bat = bat.merge(outs, on=["match_id", "innings", "batter"], how="left")
    bat["not_out"] = bat.wicket_kind.isna() | bat.wicket_kind.isin(
        ["retired hurt", "retired not out"]
    )
    team = d.drop_duplicates(["match_id", "innings"])[["match_id", "innings", "batting_team"]]
    return bat.merge(team, on=["match_id", "innings"], how="left")


def bowling_innings(deliveries: pd.DataFrame) -> pd.DataFrame:
    d = _regular(deliveries).assign(
        conceded=lambda x: x.runs_batter + x.wides + x.noballs,
        legal=lambda x: x.is_legal.astype(int),
    )
    bowl = (
        d.groupby(["match_id", "innings", "bowler", "bowling_team"], observed=True)
        .agg(balls=("legal", "sum"), runs=("conceded", "sum"), wickets=("bowler_wickets", "sum"))
        .reset_index()
    )
    return bowl[bowl.balls > 0]


def innings_totals(deliveries: pd.DataFrame) -> pd.DataFrame:
    d = _regular(deliveries)
    return (
        d.groupby(["match_id", "innings"], observed=True)
        .agg(
            team=("batting_team", "first"),
            opponent=("bowling_team", "first"),
            runs=("runs_total", "sum"),
            wickets=("team_wickets", "max"),
            balls=("legal_balls", "max"),
        )
        .reset_index()
    )


def _dismissal(row) -> str:
    if row.wicket_kind in NON_BOWLER_DISMISSALS:
        return f"{row.player_out} {row.wicket_kind}"
    return f"{row.player_out} {row.wicket_kind} off {row.bowler}"


def _bat_line(row) -> str:
    star = "*" if row.not_out else ""
    return f"{row.batter} {row.runs}{star} ({row.balls})"


def _bowl_line(row) -> str:
    return f"{row.bowler} {row.wickets}/{row.runs} ({fmt_overs(row.balls)} overs)"


def _result(m) -> str:
    if m.result == "no result":
        return "No result."
    if m.result in ("tie", "tie_super_over") and not _s(m.winner):
        return "The match was tied."
    if m.result == "tie_super_over":
        return f"Tied; {m.winner} won the super over."
    if pd.notna(m.win_by_runs):
        margin = f"by {int(m.win_by_runs)} runs"
    elif pd.notna(m.win_by_wickets):
        margin = f"by {int(m.win_by_wickets)} wickets"
    else:
        margin = ""
    method = f" ({m.method} method)" if _s(m.method) else ""
    return f"{m.winner} won {margin}{method}.".replace("  ", " ")


def match_documents(
    matches: pd.DataFrame, deliveries: pd.DataFrame, top_n: int = 3
) -> list[Document]:
    totals = innings_totals(deliveries)
    bat = batting_innings(deliveries)
    bowl = bowling_innings(deliveries)
    docs = []
    for m in matches.itertuples(index=False):
        t = totals[(totals.match_id == m.match_id) & (totals.innings <= 2)].sort_values("innings")
        scores = "; ".join(
            f"{r.team} {r.runs}/{r.wickets} ({fmt_overs(int(r.balls))} overs)"
            for r in t.itertuples()
        )
        b = bat[bat.match_id == m.match_id].sort_values(["runs", "balls"], ascending=[False, True])
        w = bowl[bowl.match_id == m.match_id].sort_values(
            ["wickets", "runs"], ascending=[False, True]
        )
        event = _s(m.event_name) or "Match"
        stage = _s(getattr(m, "event_stage", None))
        if stage:
            number = f", {stage}"
        elif pd.notna(m.match_number):
            number = f", match {m.match_number}"
        else:
            number = ""
        city = _s(m.city)
        venue = _s(m.venue) + (f", {city}" if city and city not in _s(m.venue) else "")
        pom = _s(m.player_of_match)
        parts = [
            f"{event} {m.season}{number}: {m.team1} v {m.team2} at {venue} "
            f"on {fmt_date(m.start_date)}.",
            f"Toss: {m.toss_winner} won the toss and chose to {m.toss_decision}."
            if _s(m.toss_winner)
            else "",
            f"Scores: {scores}." if scores else "",
            f"Result: {_result(m)}",
            f"Player of the match: {pom.replace(',', ', ')}." if pom else "",
            f"Top scorers: {', '.join(_bat_line(r) for r in b.head(top_n).itertuples())}."
            if len(b)
            else "",
            f"Best bowling: {', '.join(_bowl_line(r) for r in w.head(top_n).itertuples())}."
            if len(w)
            else "",
        ]
        docs.append(
            Document(
                doc_id=f"match:{m.match_id}",
                kind="match",
                title=f"{m.team1} v {m.team2}, {fmt_date(m.start_date)}"
                + (f" ({stage})" if stage else ""),
                text=" ".join(p for p in parts if p),
                metadata={
                    "match_id": m.match_id,
                    "season": m.season,
                    "date": str(pd.Timestamp(m.start_date).date()),
                    "teams": [m.team1, m.team2],
                    "venue": _s(m.venue),
                    "stage": stage or None,
                    "event": event,
                    "winner": _s(m.winner) or None,
                    "player_of_match": pom or None,
                },
            )
        )
    return docs


def phase_documents(matches: pd.DataFrame, deliveries: pd.DataFrame) -> list[Document]:
    d = _regular(deliveries)
    d = d[d.innings <= 2].assign(phase=assign_phase(d, matches[["match_id", "overs"]]))
    meta = matches.set_index("match_id")
    agg = (
        d.assign(legal=d.is_legal.astype(int))
        .groupby(["match_id", "innings", "phase"], observed=True)
        .agg(
            team=("batting_team", "first"),
            opponent=("bowling_team", "first"),
            runs=("runs_total", "sum"),
            wickets=("wickets_on_ball", "sum"),
            balls=("legal", "sum"),
            first_over=("over", "min"),
            last_over=("over", "max"),
        )
        .reset_index()
    )
    faced = d[d.wides == 0]
    top_bat = (
        faced.groupby(["match_id", "innings", "phase", "batter"], observed=True)
        .runs_batter.agg(["sum", "size"])
        .reset_index()
        .sort_values("sum", ascending=False)
    )
    wkts = d[d.player_out.notna()][
        ["match_id", "innings", "phase", "bowler", "player_out", "wicket_kind"]
    ]

    docs = []
    for r in agg.itertuples(index=False):
        m = meta.loc[r.match_id]
        key = (top_bat.match_id == r.match_id) & (top_bat.innings == r.innings)
        tb = top_bat[key & (top_bat.phase == r.phase)].head(3)
        wk = wkts[
            (wkts.match_id == r.match_id) & (wkts.innings == r.innings) & (wkts.phase == r.phase)
        ]
        rr = r.runs * 6 / r.balls if r.balls else 0.0
        bats = ", ".join(f"{x.batter} {x['sum']} ({x['size']})" for _, x in tb.iterrows())
        falls = ", ".join(_dismissal(x) for x in wk.itertuples())
        text = (
            f"{r.team} batting against {r.opponent}, {PHASE_LABEL[r.phase]} "
            f"(overs {r.first_over + 1}-{r.last_over + 1}), {_s(m.event_name)} {m.season}, "
            f"{fmt_date(m.start_date)}, innings {r.innings}: {r.runs} runs for {r.wickets} wickets "
            f"off {r.balls} balls (run rate {rr:.2f})."
            + (f" Top batters: {bats}." if bats else "")
            + (f" Wickets: {falls}." if falls else " No wickets fell.")
        )
        docs.append(
            Document(
                doc_id=f"phase:{r.match_id}:{r.innings}:{r.phase}",
                kind="innings_phase",
                title=f"{r.team} {PHASE_LABEL[r.phase]} v {r.opponent}, {fmt_date(m.start_date)}",
                text=text,
                metadata={
                    "match_id": r.match_id,
                    "season": m.season,
                    "date": str(pd.Timestamp(m.start_date).date()),
                    "team": r.team,
                    "opponent": r.opponent,
                    "phase": r.phase,
                    "innings": int(r.innings),
                    "runs": int(r.runs),
                    "wickets": int(r.wickets),
                },
            )
        )
    return docs


def player_season_documents(matches: pd.DataFrame, deliveries: pd.DataFrame) -> list[Document]:
    season = matches.set_index("match_id").season
    bat = batting_innings(deliveries).assign(season=lambda x: x.match_id.map(season))
    bowl = bowling_innings(deliveries).assign(season=lambda x: x.match_id.map(season))
    events = matches.groupby("season").event_name.first()

    bs = (
        bat.groupby(["batter", "season"])
        .agg(
            innings=("match_id", "size"),
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            outs=("not_out", lambda s: int((~s).sum())),
            high=("runs", "max"),
            fifties=("runs", lambda s: int(((s >= 50) & (s < 100)).sum())),
            hundreds=("runs", lambda s: int((s >= 100).sum())),
            team=("batting_team", lambda s: ", ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .rename(columns={"batter": "player"})
    )
    ws = (
        bowl.groupby(["bowler", "season"])
        .agg(
            matches=("match_id", "nunique"),
            balls=("balls", "sum"),
            runs=("runs", "sum"),
            wickets=("wickets", "sum"),
            team=("bowling_team", lambda s: ", ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )
    merged = bs.merge(ws, on=["player", "season"], how="outer", suffixes=("_bat", "_bowl"))
    docs = []
    for r in merged.itertuples(index=False):
        ev = _s(events.get(r.season, ""))
        team = r.team_bat if isinstance(r.team_bat, str) else r.team_bowl
        parts = [f"{r.player} ({team}) in the {ev} {r.season} season."]
        meta = {"player": r.player, "season": r.season, "team": team}
        if pd.notna(r.runs_bat):
            sr = r.runs_bat / r.balls_bat * 100 if r.balls_bat else 0.0
            avg = f"average {r.runs_bat / r.outs:.2f}" if r.outs else "never dismissed"
            parts.append(
                f"Batting: {int(r.runs_bat)} runs in {int(r.innings)} innings off "
                f"{int(r.balls_bat)} balls, strike rate {sr:.1f}, {avg}, highest score "
                f"{int(r.high)}, {int(r.fifties)} fifties and {int(r.hundreds)} hundreds."
            )
            meta["runs"] = int(r.runs_bat)
        if pd.notna(r.wickets) and r.balls_bowl:
            econ = r.runs_bowl / (r.balls_bowl / 6)
            parts.append(
                f"Bowling: {int(r.wickets)} wickets in {int(r.matches)} matches, "
                f"{fmt_overs(int(r.balls_bowl))} overs, {int(r.runs_bowl)} runs conceded, "
                f"economy {econ:.2f}."
            )
            meta["wickets"] = int(r.wickets)
        docs.append(
            Document(
                doc_id=f"player:{r.player}:{r.season}",
                kind="player_season",
                title=f"{r.player}, {r.season}",
                text=" ".join(parts),
                metadata=meta,
            )
        )
    return docs


def build_corpus(matches: pd.DataFrame, deliveries: pd.DataFrame) -> list[Document]:
    return (
        match_documents(matches, deliveries)
        + phase_documents(matches, deliveries)
        + player_season_documents(matches, deliveries)
    )


def save_documents(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), default=str) + "\n")


def load_documents(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as fh:
        return [Document(**json.loads(line)) for line in fh]
