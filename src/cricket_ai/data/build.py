"""Build validated Parquet tables from a folder of Cricsheet JSON files."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cricket_ai.data.parse import ParsedMatch, parse_match
from cricket_ai.data.validate import QualityReport, validate

log = logging.getLogger(__name__)

DELIVERY_DTYPES = {
    "innings": "int16",
    "over": "int16",
    "ball_in_over": "int16",
    "delivery_seq": "int32",
    "runs_batter": "int16",
    "runs_extras": "int16",
    "runs_total": "int16",
    "wides": "int16",
    "noballs": "int16",
    "byes": "int16",
    "legbyes": "int16",
    "penalty": "int16",
    "wickets_on_ball": "int8",
    "bowler_wickets": "int8",
    "team_runs": "int32",
    "team_wickets": "int16",
    "legal_balls": "int32",
}
CATEGORY_COLS = ["batting_team", "bowling_team", "wicket_kind"]


@dataclass
class BuildResult:
    matches: pd.DataFrame
    deliveries: pd.DataFrame
    players: pd.DataFrame
    report: QualityReport
    failed_files: dict[str, str]


def _safe_parse(path: Path) -> tuple[str, ParsedMatch | None, str | None]:
    try:
        return str(path), parse_match(path), None
    except Exception as exc:  # noqa: BLE001 - we record and continue on malformed files
        return str(path), None, f"{type(exc).__name__}: {exc}"


def build_tables(
    raw_dir: Path, workers: int | None = None, limit: int | None = None
) -> BuildResult:
    files = sorted(Path(raw_dir).glob("*.json"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .json match files in {raw_dir}")

    match_rows, delivery_rows, player_rows, failed = [], [], [], {}
    if workers == 1 or len(files) < 50:
        results = map(_safe_parse, files)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_safe_parse, files, chunksize=32)
    for path, parsed, err in results:
        if err:
            failed[path] = err
            continue
        match_rows.append(parsed.match)
        delivery_rows.extend(parsed.deliveries)
        player_rows.extend(parsed.players)

    matches = pd.DataFrame(match_rows)
    matches["start_date"] = pd.to_datetime(matches["start_date"], errors="coerce")
    deliveries = pd.DataFrame(delivery_rows).astype(DELIVERY_DTYPES)
    for col in CATEGORY_COLS:
        deliveries[col] = deliveries[col].astype("category")
    players = pd.DataFrame(player_rows, columns=["match_id", "team", "player", "player_id"])

    report = validate(matches, deliveries)
    if failed:
        report.errors["unparseable_files"] = len(failed)
        report.examples["unparseable_files"] = [
            {"file": k, "error": v} for k, v in list(failed.items())[:5]
        ]
    log.info(
        "Parsed %d matches, %d deliveries (%d files failed)",
        len(matches),
        len(deliveries),
        len(failed),
    )
    return BuildResult(matches, deliveries, players, report, failed)


def write_tables(result: BuildResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("matches", "deliveries", "players"):
        p = out_dir / f"{name}.parquet"
        getattr(result, name).to_parquet(p, index=False)
        paths[name] = p
    qp = out_dir / "quality_report.json"
    qp.write_text(json.dumps(result.report.to_dict(), indent=2, default=str))
    paths["quality_report"] = qp
    return paths


def load_tables(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(processed / "matches.parquet"),
        pd.read_parquet(processed / "deliveries.parquet"),
        pd.read_parquet(processed / "players.parquet"),
    )
