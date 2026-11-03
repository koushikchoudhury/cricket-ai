# cricket-ai

An end-to-end applied AI portfolio built on cricket: ball-by-ball analytics and win probability,
synthetic data generation, computer vision on match footage, multimodal RAG and an agent that
orchestrates all of it, each module containerised and deployable to Kubernetes on AWS.

Core models (the GAN, the shot classifier and the win-probability model) are hand-written in
plain PyTorch; libraries such as Ultralytics, Hugging Face and CTGAN are used as baselines and
building blocks around them.

## Modules

| # | Module | What it shows | Status |
|---|--------|---------------|--------|
| 1 | Data pipeline, analytics, win probability | Data engineering, validation, tabular modelling, calibration | 🟡 Pipeline done, model next |
| 2 | Synthetic data (GAN, CTGAN) | Generative modelling, fidelity / utility / privacy evaluation | ⚪ Planned |
| 3 | Video and image analysis | YOLO detection, ball tracking, pose-based shot classification | ⚪ Planned |
| 4 | Multimodal RAG | Retrieval over scorecards, commentary, images and clips, with evals | ⚪ Planned |
| 5 | LangGraph agent | Tool orchestration across modules 1–4 | ⚪ Planned |
| 6 | Docker, Kubernetes, AWS | kind locally, short EKS deployment, S3/ECR | 🟡 Data image done |

Full task breakdown: [TRACKER.md](TRACKER.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cricket-ai data download ipl      # Cricsheet IPL archive -> data/raw/ipl
cricket-ai data build ipl         # validated Parquet tables -> data/processed/ipl
cricket-ai data features ipl      # win-probability chase states
cricket-ai data report ipl        # batting / bowling / phase / team summaries -> reports/ipl
```

Or with Docker:

```bash
docker build -f docker/data.Dockerfile -t cricket-ai-data .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" cricket-ai-data data download ipl
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" cricket-ai-data data build ipl
```

`make help` lists the shortcuts.

## Module 1: data pipeline

```
Cricsheet JSON ──parse──▶ matches / deliveries / players ──validate──▶ Parquet
                                         │
                                         ├──▶ analytics (batting, bowling, phases, teams)
                                         └──▶ chase_states (features for win probability)
```

**Tables**

- `matches`: one row per match: teams, venue, date, toss, result, margin, target, method (D/L).
- `deliveries`: one row per delivery including wides and no-balls, with legal-ball numbering,
  the extras breakdown, wicket details, bowler-credited wickets and running innings totals.
- `players`: each XI with Cricsheet registry ids.

**Scoring conventions** follow the Laws: wides don't count as balls faced, byes and leg-byes
aren't charged to the bowler, run outs and retirements aren't bowler wickets, retired hurt
isn't a team wicket.

**Data quality.** Every build writes `quality_report.json`. Errors (duplicate keys, run
arithmetic that doesn't add up, unparseable files) are counted and `--strict` fails the build;
warnings record real-world quirks such as umpire-miscounted seven-ball overs.

**Win-probability features.** `chase_states` produces one row per live second-innings state
(runs needed, balls left, wickets in hand, required and current run rate, last-12-ball
momentum) labelled with whether the chasing side won. Ties, no-results and D/L-adjusted matches
are excluded by default, and `split_by_date` gives a time-based train/test split so no match
leaks across it.

## Repository layout

```
src/cricket_ai/
  data/        download, parse, validate, build, features
  analytics/   summary tables
  winprob/     module 1 model (next)
  synthetic/   module 2
  vision/      module 3
  rag/         module 4
  agent/       module 5
  cli.py       `cricket-ai` command
docker/        one Dockerfile per module
infra/k8s/     kind and EKS manifests (module 6)
infra/aws/     S3 / ECR / EKS setup (module 6)
notebooks/     Colab / Kaggle training notebooks
tests/         unit and end-to-end tests with a hand-scored fixture match
```

## Compute

GPU training runs on Colab or Kaggle notebooks in `notebooks/`. AWS resources are created
with the owner's own credentials; nothing in this repo stores keys.

## Data and licence

Ball-by-ball data comes from [Cricsheet](https://cricsheet.org), made available under the
Open Data Commons Attribution License. Raw and processed data are not committed. Code is MIT.
