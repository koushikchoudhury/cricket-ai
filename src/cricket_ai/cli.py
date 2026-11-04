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
winprob_app = typer.Typer(help="Module 1: win-probability model.", no_args_is_help=True)
app.add_typer(winprob_app, name="winprob")

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


@winprob_app.command("train")
def winprob_train(
    dataset: str = typer.Argument("ipl"),
    max_epochs: int = typer.Option(60, help="Upper bound; early stopping usually ends sooner"),
    device: str = typer.Option("auto", help="auto, cpu, cuda or mps"),
    test_seasons: int = typer.Option(2, help="Most recent seasons held out for testing"),
):
    """Fit baselines and the PyTorch MLP, calibrate, evaluate on held-out seasons."""
    import pandas as pd

    from cricket_ai.winprob.data import date_split, season_split
    from cricket_ai.winprob.evaluate import run_experiment
    from cricket_ai.winprob.model import TrainConfig

    states = pd.read_parquet(config.processed_dir(dataset) / "chase_states.parquet")
    try:
        split = season_split(states, test_seasons=test_seasons)
    except ValueError:
        split = date_split(states)  # too few seasons: fall back to a chronological split
    cfg = TrainConfig(max_epochs=max_epochs, device=device)
    result = run_experiment(
        states,
        report_dir=config.REPORTS_DIR / dataset / "winprob",
        model_dir=config.MODELS_DIR / dataset,
        cfg=cfg,
        split=split,
    )
    typer.echo(f"Split: {result['split']}")
    typer.echo(pd.DataFrame(result["test_metrics"]).T.to_markdown())
    typer.echo(f"Report: {config.REPORTS_DIR / dataset / 'winprob' / 'report.md'}")


@winprob_app.command("curve")
def winprob_curve(
    match_id: str = typer.Argument(
        ..., help="Cricsheet match id, e.g. 1473505 (file name without .json)"
    ),
    dataset: str = typer.Option("ipl"),
    moments: int = typer.Option(5, help="How many key moments to list"),
):
    """Plot the win-probability curve of one chase and list its biggest swings."""
    import pandas as pd

    from cricket_ai.winprob.curve import key_moments, match_curve, plot_curve
    from cricket_ai.winprob.model import load

    proc = config.processed_dir(dataset)
    states = pd.read_parquet(proc / "chase_states.parquet")
    deliveries = pd.read_parquet(proc / "deliveries.parquet")
    matches = pd.read_parquet(proc / "matches.parquet").set_index("match_id")
    model = load(config.MODELS_DIR / dataset / "winprob_mlp.pt")
    if match_id not in set(states.match_id):
        recent = (
            matches.loc[matches.index.isin(states.match_id)]
            .sort_values("start_date", ascending=False)
            .head(8)
        )
        typer.echo(f"No chase found for match id '{match_id}'.", err=True)
        typer.echo("Match ids are Cricsheet file names without .json. Recent chases:", err=True)
        for mid, r in recent.iterrows():
            typer.echo(f"  {mid}  {r.start_date:%Y-%m-%d}  {r.team1} v {r.team2}", err=True)
        raise typer.Exit(code=1)
    curve = match_curve(model, states, deliveries, match_id)
    m = matches.loc[match_id]
    title = (
        f"{curve.batting_team.iloc[0]} chasing {int(curve.target.iloc[0])} v "
        f"{curve.bowling_team.iloc[0]}, {m.start_date:%d %b %Y}"
    )
    out = config.REPORTS_DIR / dataset / "winprob" / "curves" / f"{match_id}.png"
    plot_curve(curve, title, out)
    typer.echo(title)
    typer.echo(f"Winner: {m.winner}")
    typer.echo(key_moments(curve, moments).to_markdown(index=False, floatfmt=".3f"))
    typer.echo(f"Plot: {out}")


@app.command("version")
def version():
    from cricket_ai import __version__

    typer.echo(json.dumps({"cricket_ai": __version__}))


if __name__ == "__main__":
    app()
