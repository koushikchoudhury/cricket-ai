"""Command-line entry point: ``cricket-ai <group> <command>``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from cricket_ai import config

app = typer.Typer(help="Cricket AI portfolio toolkit.", no_args_is_help=True)
data_app = typer.Typer(help="Module 1: data pipeline.", no_args_is_help=True)
app.add_typer(data_app, name="data")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@data_app.command("download")
def download(dataset: str = typer.Argument("ipl", help="Cricsheet dataset, e.g. ipl, t20s, odis")):
    """Download a Cricsheet JSON archive into data/raw/<dataset>."""
    from cricket_ai.data.download import download_dataset

    files = download_dataset(dataset, config.raw_dir(dataset))
    typer.echo(f"Extracted {len(files)} match files to {config.raw_dir(dataset)}")


@data_app.command("build")
def build(
    dataset: str = typer.Argument("ipl"),
    raw: Path = typer.Option(None, help="Folder of match JSON files (default data/raw/<dataset>)"),
    workers: int = typer.Option(None, help="Parser processes (default: all cores)"),
    limit: int = typer.Option(None, help="Only parse the first N files (for quick runs)"),
    strict: bool = typer.Option(False, help="Exit non-zero if any quality error is found"),
):
    """Parse JSON into validated Parquet tables in data/processed/<dataset>."""
    from cricket_ai.data.build import build_tables, write_tables

    result = build_tables(raw or config.raw_dir(dataset), workers=workers, limit=limit)
    paths = write_tables(result, config.processed_dir(dataset))
    rep = result.report
    typer.echo(f"{rep.n_matches} matches, {rep.n_deliveries} deliveries")
    typer.echo(f"errors: {rep.errors}\nwarnings: {rep.warnings}")
    for name, p in paths.items():
        typer.echo(f"  {name}: {p}")
    if strict and not rep.ok:
        raise typer.Exit(code=1)


@data_app.command("features")
def features(dataset: str = typer.Argument("ipl"), include_dl: bool = False):
    """Build the win-probability chase-state table (chase_states.parquet)."""
    from cricket_ai.data.build import load_tables
    from cricket_ai.data.features import chase_states

    out_dir = config.processed_dir(dataset)
    matches, deliveries, _ = load_tables(out_dir)
    states = chase_states(matches, deliveries, include_dl=include_dl)
    path = out_dir / "chase_states.parquet"
    states.to_parquet(path, index=False)
    chase_win_rate = states.groupby("match_id").chasing_team_won.first().mean()
    typer.echo(
        f"{len(states)} states from {states.match_id.nunique()} matches "
        f"(chasing win rate {chase_win_rate:.3f}) -> {path}"
    )


@data_app.command("report")
def report(dataset: str = typer.Argument("ipl"), min_balls: int = 300, top: int = 15):
    """Write batting/bowling/phase/team summaries to reports/<dataset>/."""
    from cricket_ai.analytics import (
        batting_summary,
        bowling_summary,
        phase_summary,
        team_season_summary,
    )
    from cricket_ai.data.build import load_tables

    matches, deliveries, _ = load_tables(config.processed_dir(dataset))
    out = config.REPORTS_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "batting": batting_summary(deliveries, min_balls=min_balls),
        "bowling": bowling_summary(deliveries, min_balls=min_balls),
        "phases": phase_summary(deliveries, matches),
        "team_season": team_season_summary(matches),
    }
    md = [f"# {dataset} summary\n", f"{len(matches)} matches, {len(deliveries)} deliveries\n"]
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv")
        md.append(f"\n## {name}\n\n{df.head(top).to_markdown()}\n")
    (out / "summary.md").write_text("".join(md))
    typer.echo(f"Reports written to {out}")


@app.command("version")
def version():
    from cricket_ai import __version__

    typer.echo(json.dumps({"cricket_ai": __version__}))


if __name__ == "__main__":
    app()
