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
synth_app = typer.Typer(help="Module 2: synthetic data.", no_args_is_help=True)
app.add_typer(synth_app, name="synth")
rag_app = typer.Typer(help="Module 4: question answering (RAG + SQL).", no_args_is_help=True)
app.add_typer(rag_app, name="rag")
agent_app = typer.Typer(help="Module 5: LangGraph agent.", no_args_is_help=True)
app.add_typer(agent_app, name="agent")

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
    half_life: float = typer.Option(4.0, help="Recency weighting half-life in seasons; 0 = off"),
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
    cfg = TrainConfig(max_epochs=max_epochs, device=device, recency_half_life=half_life or None)
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


@winprob_app.command("ablate")
def winprob_ablate(
    dataset: str = typer.Argument("ipl"),
    max_epochs: int = typer.Option(60),
    device: str = typer.Option("auto"),
    test_seasons: int = typer.Option(2),
):
    """Compare feature sets and recency weighting (6 variants, logistic + MLP each)."""
    import pandas as pd

    from cricket_ai.winprob.data import date_split, season_split
    from cricket_ai.winprob.evaluate import run_ablation
    from cricket_ai.winprob.model import TrainConfig

    states = pd.read_parquet(config.processed_dir(dataset) / "chase_states.parquet")
    try:
        split = season_split(states, test_seasons=test_seasons)
    except ValueError:
        split = date_split(states)
    out = config.REPORTS_DIR / dataset / "winprob"
    table = run_ablation(
        states, out, TrainConfig(max_epochs=max_epochs, device=device), split=split
    )
    typer.echo(f"Split: {split.description}")
    typer.echo(table.to_markdown(index=False))
    typer.echo(f"Saved: {out / 'ablation.md'}")


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


def _overs(dataset: str):
    import pandas as pd

    path = config.processed_dir(dataset) / "overs.parquet"
    if not path.exists():
        typer.echo(f"{path} not found. Run `cricket-ai synth table {dataset}` first.", err=True)
        raise typer.Exit(code=1)
    return pd.read_parquet(path)


@synth_app.command("table")
def synth_table(dataset: str = typer.Argument("ipl")):
    """Build the over-level real table (data/processed/<dataset>/overs.parquet)."""
    from cricket_ai.data.build import load_tables
    from cricket_ai.synthetic.table import build_overs, split_by_match, validity

    out_dir = config.processed_dir(dataset)
    matches, deliveries, _ = load_tables(out_dir)
    overs = build_overs(matches, deliveries)
    overs.to_parquet(out_dir / "overs.parquet", index=False)
    train, holdout = split_by_match(overs)
    typer.echo(
        f"{len(overs)} overs from {overs.match_id.nunique()} matches "
        f"(train {len(train)}, holdout {len(holdout)})"
    )
    typer.echo(f"Real data validity: {validity(overs)}")


@synth_app.command("train")
def synth_train(
    dataset: str = typer.Argument("ipl"),
    model: str = typer.Option(
        "gan", help="gan (hand-written), marginals, or ctgan (pip install ctgan)"
    ),
    epochs: int = typer.Option(300, help="Training epochs for gan / ctgan"),
    rows: int = typer.Option(0, help="Rows to sample (default: same as the real train set)"),
    device: str = typer.Option("auto", help="auto, cuda, mps or cpu"),
    batch_size: int = typer.Option(500, help="Must be a multiple of pac (10)"),
):
    """Fit a generator on the real train split and save a synthetic table."""
    from cricket_ai.synthetic.table import DISCRETE, split_by_match, validity

    train, _ = split_by_match(_overs(dataset))
    n = rows or len(train)
    model_dir = config.MODELS_DIR / dataset / f"synth_{model}"
    if model == "gan":
        from cricket_ai.synthetic.gan import GANConfig, TabularGAN

        gen = TabularGAN(GANConfig(epochs=epochs, device=device, batch_size=batch_size))
        gen.fit(train, DISCRETE)
        gen.save(model_dir)
        typer.echo(f"Trained on {gen.history.device} in {gen.history.seconds}s -> {model_dir}")
    elif model == "marginals":
        from cricket_ai.synthetic.baselines import IndependentMarginals

        gen = IndependentMarginals().fit(train, DISCRETE)
    elif model == "ctgan":
        from cricket_ai.synthetic.baselines import CTGANBaseline

        gen = CTGANBaseline(epochs=epochs, device=device).fit(train, DISCRETE)
    else:
        typer.echo(f"Unknown model {model!r}", err=True)
        raise typer.Exit(code=1)

    synth = gen.sample(n)
    out = config.processed_dir(dataset) / "synthetic" / f"{model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    synth.to_parquet(out, index=False)
    typer.echo(f"Sampled {len(synth)} rows -> {out}")
    typer.echo(
        f"Validity (share of rows obeying every cricket rule): {validity(synth)['all_rules']}"
    )


@synth_app.command("evaluate")
def synth_evaluate(dataset: str = typer.Argument("ipl")):
    """Compare every synthetic table with the real holdout; write reports/<dataset>/synthetic/."""
    import pandas as pd

    from cricket_ai.synthetic.evaluate import evaluate_all
    from cricket_ai.synthetic.table import split_by_match

    train, holdout = split_by_match(_overs(dataset))
    synth_dir = config.processed_dir(dataset) / "synthetic"
    synthetic = {p.stem: pd.read_parquet(p) for p in sorted(synth_dir.glob("*.parquet"))}
    if not synthetic:
        typer.echo(
            f"No synthetic tables in {synth_dir}. Run `cricket-ai synth train` first.", err=True
        )
        raise typer.Exit(code=1)
    out = config.REPORTS_DIR / dataset / "synthetic"
    result = evaluate_all(train, holdout, synthetic, out)
    cols = [
        "mean_tvd",
        "corr_diff",
        "validity",
        "runs_mae",
        "wicket_auc",
        "dcr_ratio",
        "membership_auc",
    ]
    table = pd.DataFrame(result["summary"]).set_index("model")
    typer.echo(table.reindex(columns=cols).to_markdown())
    typer.echo(f"Report: {out / 'report.md'}")


def _rag_dir(dataset: str):
    return config.processed_dir(dataset) / "rag"


def _load_rag(dataset: str, llm: str | None, model: str | None, k: int, device: str | None):
    from cricket_ai.llm import get_chat_model
    from cricket_ai.rag.pipeline import CricketRAG
    from cricket_ai.rag.retrieval import HybridRetriever
    from cricket_ai.rag.sql import CricketSQL

    index = _rag_dir(dataset)
    if not (index / "index.json").exists():
        typer.echo(f"No index in {index}. Run `cricket-ai rag index {dataset}` first.", err=True)
        raise typer.Exit(code=1)
    retriever = HybridRetriever.load(index, device=device)
    sql = CricketSQL.from_dir(config.processed_dir(dataset))
    return CricketRAG(retriever, sql, get_chat_model(llm, model), k=k)


@rag_app.command("index")
def rag_index(
    dataset: str = typer.Argument("ipl"),
    embedder: str = typer.Option(
        "hashing", help="hashing (no download) or st (sentence-transformers)"
    ),
    model: str = typer.Option(None, help="Embedding model, default BAAI/bge-small-en-v1.5"),
    device: str = typer.Option("auto", help="auto, cuda or cpu (sentence-transformers only)"),
):
    """Build the document corpus and search index (data/processed/<dataset>/rag/)."""
    import time
    from collections import Counter

    from cricket_ai.data.build import load_tables
    from cricket_ai.rag.corpus import build_corpus
    from cricket_ai.rag.retrieval import HybridRetriever, make_embedder

    matches, deliveries, _ = load_tables(config.processed_dir(dataset))
    t0 = time.perf_counter()
    docs = build_corpus(matches, deliveries)
    typer.echo(f"{len(docs)} documents: {dict(Counter(d.kind for d in docs))}")
    retriever = HybridRetriever(docs, make_embedder(embedder, model, device))
    retriever.save(_rag_dir(dataset))
    typer.echo(f"Indexed in {time.perf_counter() - t0:.0f}s -> {_rag_dir(dataset)}")


@rag_app.command("ask")
def rag_ask(
    question: str = typer.Argument(..., help="Your question, in quotes"),
    dataset: str = typer.Option("ipl"),
    llm: str = typer.Option(None, help="none, ollama, anthropic or openai (or set CRICKET_AI_LLM)"),
    model: str = typer.Option(None, help="Model name for the provider"),
    route: str = typer.Option("auto", help="auto, sql or search"),
    k: int = typer.Option(5, help="Documents to retrieve"),
    device: str = typer.Option(None),
):
    """Answer a question with retrieved sources, using SQL for numbers."""
    rag = _load_rag(dataset, llm, model, k, device)
    a = rag.ask(question, route=None if route == "auto" else route)
    typer.echo(f"\n{a.text}\n")
    if a.sql:
        typer.echo(f"SQL ({'failed: ' + a.error if a.error else 'ok'}):\n{a.sql}\n")
    typer.echo("Sources:")
    for s in a.sources:
        typer.echo(f"  [{s['n']}] {s['title']}  ({s['doc_id']})")
    images = _rag_dir(dataset) / "images"
    if (images / "images.json").exists():
        from cricket_ai.rag.images import ClipEmbedder, ImageIndex

        typer.echo("Related images:")
        for h in ImageIndex.load(images, ClipEmbedder()).search(question, 3):
            typer.echo(f"  {h.path}  {h.caption}  ({h.score:.2f})")
    typer.echo(f"[route={a.route}, {a.seconds}s]")


@rag_app.command("evaluate")
def rag_evaluate(
    dataset: str = typer.Argument("ipl"),
    per_type: int = typer.Option(20, help="Questions per type (5 types)"),
    llm: str = typer.Option(None, help="Also score written answers with this LLM"),
    model: str = typer.Option(None),
    answers: int = typer.Option(0, help="How many questions to answer with the LLM (0 = all)"),
    device: str = typer.Option(None),
):
    """Score retrieval (and, with --llm, answers) on generated questions with known answers."""
    from cricket_ai.llm import llm_settings
    from cricket_ai.rag.evaluate import (
        evaluate_answers,
        evaluate_retrieval,
        generate_questions,
        summarise_retrieval,
        write_report,
    )

    provider, model_name = llm_settings(llm, model)
    rag = _load_rag(dataset, provider, model_name, 5, device)
    questions = generate_questions(rag.retriever.docs, per_type=per_type)
    overall, by_type = summarise_retrieval(evaluate_retrieval(rag.retriever, questions))
    typer.echo(f"{len(questions)} questions\n\nRetrieval:\n{overall.to_markdown()}\n")
    typer.echo(f"Recall@5 by type:\n{by_type.to_markdown()}\n")
    answered = None
    if rag.llm is not None:
        subset = questions[:answers] if answers else questions
        typer.echo(f"Answering {len(subset)} questions with {provider}:{model_name} ...")
        answered = evaluate_answers(rag, subset)
        typer.echo(f"Answer accuracy: {answered.correct.mean():.1%}")
        typer.echo(answered.groupby("type").correct.mean().round(3).to_markdown())
    out = config.REPORTS_DIR / dataset / "rag"
    meta = {
        "n_questions": len(questions),
        "n_docs": len(rag.retriever.docs),
        "embedder": rag.retriever.embedder.config(),
        "llm": f"{provider}:{model_name}" if rag.llm is not None else None,
    }
    write_report(out, overall, by_type, answered, meta)
    typer.echo(f"Report: {out / 'report.md'}")


@rag_app.command("images")
def rag_images(
    folder: Path = typer.Argument(..., help="Folder of images (optional captions.json inside)"),
    dataset: str = typer.Option("ipl"),
    device: str = typer.Option(None),
):
    """Index images with CLIP so `rag ask` also returns related pictures."""
    from cricket_ai.rag.images import ClipEmbedder, ImageIndex

    index = ImageIndex.build(folder, ClipEmbedder(device=device))
    index.save(_rag_dir(dataset) / "images")
    typer.echo(f"Indexed {len(index.paths)} images -> {_rag_dir(dataset) / 'images'}")


def _make_agent(dataset: str, llm: str | None, model: str | None, yes: bool, max_steps: int):
    from cricket_ai.agent.graph import CricketAgent
    from cricket_ai.agent.tools import CricketContext
    from cricket_ai.llm import get_chat_model, llm_settings

    provider, model_name = llm_settings(llm, model)
    chat = get_chat_model(provider, model_name)
    if chat is None:
        typer.echo("The agent needs an LLM: add --llm ollama (or anthropic / openai).", err=True)
        raise typer.Exit(code=1)
    ctx = CricketContext.from_dataset(config.processed_dir(dataset), config.MODELS_DIR / dataset)

    def approve(name: str, args: dict) -> bool:
        return yes or typer.confirm(f"Agent wants to run {name}({args}). Allow?", default=True)

    agent = CricketAgent(chat, ctx, max_steps=max_steps, approver=approve)
    typer.echo(
        f"[{provider}:{model_name}] tools: {', '.join(t.name for t in agent.tools)}", err=True
    )
    return agent, f"{provider}:{model_name}"


@agent_app.command("ask")
def agent_ask(
    question: str = typer.Argument(...),
    dataset: str = typer.Option("ipl"),
    llm: str = typer.Option(None, help="ollama, anthropic or openai (or set CRICKET_AI_LLM)"),
    model: str = typer.Option(None),
    yes: bool = typer.Option(False, "--yes", "-y", help="Approve expensive tools without asking"),
    max_steps: int = typer.Option(6, help="Maximum tool rounds"),
):
    """Ask the agent one question; it chooses and calls the tools."""
    from cricket_ai.agent.graph import write_trace

    agent, _ = _make_agent(dataset, llm, model, yes, max_steps)
    result = agent.ask(question)
    typer.echo(f"\n{result.answer}\n")
    for c in result.tool_calls:
        typer.echo(f"  -> {c['name']}({c['args']})  {c['seconds']}s{'' if c['ok'] else '  ERROR'}")
    trace = write_trace(result, config.REPORTS_DIR / dataset / "agent")
    typer.echo(f"[{result.steps} tool rounds, {result.seconds}s; trace: {trace}]")


@agent_app.command("chat")
def agent_chat(
    dataset: str = typer.Option("ipl"),
    llm: str = typer.Option(None),
    model: str = typer.Option(None),
    yes: bool = typer.Option(False, "--yes", "-y"),
    max_steps: int = typer.Option(6),
):
    """Interactive chat in the terminal (each question is answered independently)."""
    from cricket_ai.agent.graph import write_trace

    agent, _ = _make_agent(dataset, llm, model, yes, max_steps)
    typer.echo("Ask a question (empty line to quit).")
    while True:
        question = typer.prompt("you", default="", show_default=False).strip()
        if not question:
            break
        result = agent.ask(question)
        typer.echo(f"agent: {result.answer}\n  [tools: {', '.join(result.tools_used) or 'none'}]")
        write_trace(result, config.REPORTS_DIR / dataset / "agent")


@agent_app.command("evaluate")
def agent_evaluate(
    dataset: str = typer.Argument("ipl"),
    llm: str = typer.Option(None),
    model: str = typer.Option(None),
    per_type: int = typer.Option(4, help="Tasks per type"),
    max_steps: int = typer.Option(6),
):
    """Score tool choice and answers on tasks generated from the data."""
    from cricket_ai.agent.evaluate import evaluate_agent, generate_tasks, write_report

    agent, name = _make_agent(dataset, llm, model, True, max_steps)
    tasks = generate_tasks(agent.ctx, per_type=per_type)
    typer.echo(f"Running {len(tasks)} tasks ...")
    results = evaluate_agent(agent, tasks)
    out = config.REPORTS_DIR / dataset / "agent"
    summary = write_report(results, out, name)
    typer.echo(json.dumps(summary, indent=2))
    typer.echo((out / "report.md").read_text())


@agent_app.command("graph")
def agent_graph(dataset: str = typer.Option("ipl")):
    """Print the agent graph as Mermaid (paste into mermaid.live or a README)."""
    from cricket_ai.agent.graph import CricketAgent
    from cricket_ai.agent.tools import CricketContext

    class _NoLLM:
        def bind_tools(self, tools):
            return self

    ctx = CricketContext.from_dataset(config.processed_dir(dataset), config.MODELS_DIR / dataset)
    typer.echo(CricketAgent(_NoLLM(), ctx).mermaid())


@agent_app.command("ui")
def agent_ui(
    dataset: str = typer.Option("ipl"),
    llm: str = typer.Option(None),
    model: str = typer.Option(None),
    port: int = typer.Option(7860),
):
    """Chat with the agent in the browser (pip install gradio)."""
    from cricket_ai.agent.ui import launch

    agent, _ = _make_agent(dataset, llm, model, True, 6)
    launch(agent, port=port)


@app.command("version")
def version():
    from cricket_ai import __version__

    typer.echo(json.dumps({"cricket_ai": __version__}))


if __name__ == "__main__":
    app()
