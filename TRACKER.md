# Tracker

Status: ✅ done · 🟡 in progress · ⚪ not started. Update this file in the same commit as the work.

| Module | Status | Next step |
|---|---|---|
| 1. Data, analytics, win probability | 🟡 | Train on real IPL data and review the report; first-innings model |
| 2. Synthetic data | ⚪ | Decide the synthetic target table (match-level vs over-level sequences) |
| 3. Video and image analysis | ⚪ | Source and label a small broadcast-frame dataset |
| 4. Multimodal RAG | ⚪ | Build the scorecard and commentary corpus |
| 5. LangGraph agent | ⚪ | Wrap modules 1–4 as tools |
| 6. Docker, Kubernetes, AWS | 🟡 | kind cluster running the data job |

---

## 1. Data pipeline, analytics, win probability

**Deliverables:** Parquet tables, quality report, analytics reports, a calibrated win-probability
model with an evaluation write-up, and a win-probability curve for any match.

- [x] Repo scaffold, packaging, CLI, CI, tests
- [x] Cricsheet download with safe zip extraction
- [x] Parser: matches, deliveries (legal-ball numbering, extras, wickets, running totals), players
- [x] Validation and `quality_report.json`
- [x] Analytics: batting, bowling, phase, team-season summaries
- [x] Chase-state feature table and time-based split
- [x] Run on the full IPL archive: 1,243 matches, 0 quality errors, leaders check out
- [ ] Run on the T20I archive
- [ ] Normalise franchise renames (e.g. Kings XI Punjab → Punjab Kings) for team-level analysis
- [ ] DuckDB query layer for ad-hoc SQL (also used by the agent in module 5)
- [x] Season-based train/val/test split with no shared matches
- [x] Baselines: base rate, logistic regression, monotone gradient boosting
- [x] **Hand-written PyTorch MLP** win-probability model (training loop, early stopping, checkpoint)
- [ ] First-innings model (projected score → win probability) so curves cover the whole match
- [x] Calibration: reliability diagram, Brier score, log loss, ECE; temperature scaling
- [x] Evaluation code for held-out seasons vs baselines, including metrics by chase stage
- [x] Run 1 on real IPL data (results below): era drift found
- [x] Era fixes: par score from earlier matches, recency weighting, Platt calibration
- [ ] Run 2 with the era fixes; ablation with `--half-life 0`
- [x] Win-probability curve plot and "key moments" (largest swings) per match
- [ ] Notebook: EDA and model report
- [ ] Model card

**Metrics:** log loss, Brier score, ECE, AUC on a held-out season split.

### Results

**Run 1** (2026-11-04). Train 2008–2023 (984 matches), val 2024 (71), test 2025–26 (140 matches, 15,921 states).

| model | log loss | Brier | ECE | AUC |
|---|---|---|---|---|
| MLP (PyTorch) | **0.403** | **0.129** | 0.085 | 0.913 |
| Logistic | 0.405 | 0.132 | 0.099 | **0.914** |
| GBM, monotone | 0.526 | 0.165 | 0.129 | 0.879 |
| Base rate | 0.689 | 0.248 | 0.036 | 0.500 |

MLP by chase stage: ECE 0.152 early, 0.080 middle, 0.047 late.

Findings:
- The MLP only edges logistic regression: most signal is near-linear in runs, balls and wickets.
- **Era drift.** Every model under-predicts the chasing side on 2025–26 (reliability curves sit above the diagonal), worst early in the chase, when difficulty depends on the scoring environment. Consistent with higher scoring after the 2023 Impact Player rule.
- Gradient boosting suffers most: trees cannot extrapolate to targets above the training range.
- Temperature scaling (0.93) cannot correct a shift, only sharpen or soften.

## 2. Synthetic data (GAN, CTGAN)

**Deliverables:** a hand-written GAN, a CTGAN baseline and an evaluation report comparing
synthetic and real data.

- [ ] Choose the target: match-level summary rows and/or over-by-over sequences
- [ ] Preprocessing: mode-specific normalisation for continuous columns, one-hot for categoricals
- [ ] **Hand-written PyTorch GAN** (WGAN-GP with a conditional vector), trained on Colab/Kaggle
- [ ] CTGAN baseline (SDV)
- [ ] Fidelity: per-column distributions, correlation difference, KS and TV distances
- [ ] Domain validity: run arithmetic, wickets ≤ 10, balls ≤ overs × 6
- [ ] Utility: train on synthetic, test on real (TSTR) with the module 1 win-probability model
- [ ] Privacy: distance to closest record, membership-inference check
- [ ] Report and notebook

**Metrics:** KS / TV distances, correlation difference, TSTR vs TRTR gap, DCR.

## 3. Video and image analysis

**Deliverables:** detection and tracking pipeline on clips, a shot classifier, and an annotated
output video.

- [ ] Dataset: collect clips you have rights to use; label frames (batter, bowler, ball, stumps, bat)
- [ ] YOLO (Ultralytics) fine-tune for players, bat, stumps and ball; mAP report
- [ ] Ball tracking: small-object detection + Kalman filter / ByteTrack; trajectory plot
- [ ] Pose extraction (YOLO-pose or MediaPipe) over the batting stroke
- [ ] **Hand-written PyTorch shot classifier** on pose sequences (1D-CNN or GRU): drive, pull, cut, sweep, defence, …
- [ ] Shot classifier evaluation: confusion matrix, macro F1, per-class errors
- [ ] Inference script producing an annotated video and JSON events
- [ ] Export to ONNX for the serving image

**Metrics:** mAP@0.5 and mAP@0.5:0.95 for detection; tracking continuity; macro F1 for shots.

## 4. Multimodal RAG

**Deliverables:** a retrieval service answering questions over scorecards, commentary, images
and clips, with an evaluation set and scores.

- [ ] Corpus: scorecards from module 1 tables, commentary text, frames and clip captions from module 3
- [ ] Chunking and metadata (match, innings, over, players) for filtered retrieval
- [ ] Text embeddings (Hugging Face) and image embeddings (CLIP / SigLIP)
- [ ] Vector store (Qdrant or pgvector) with hybrid search (BM25 + dense) and a reranker
- [ ] Structured route: questions with numbers go to SQL over the Parquet tables
- [ ] Generation with citations to source chunks
- [ ] Eval set of ~100 questions with answers and source ids
- [ ] Metrics: recall@k, MRR, faithfulness and answer correctness (Ragas or LLM-judge with spot checks)
- [ ] FastAPI service

## 5. LangGraph agent

**Deliverables:** an agent that answers cricket questions by calling the other modules, with
traces and an evaluation.

- [ ] Tools: SQL analytics, win-probability lookup, synthetic-data generator, vision on a clip, RAG search
- [ ] LangGraph state graph with planner, tool nodes and an answer node; retries and guardrails
- [ ] Human-in-the-loop step for expensive tools (vision on long clips)
- [ ] Tracing (LangSmith or OpenTelemetry)
- [ ] Task-success eval set and report
- [ ] Simple UI (Streamlit or Gradio)

## 6. Docker, Kubernetes, AWS

**Deliverables:** an image per module, manifests that run on kind, a short EKS deployment
with teardown, and CI that builds images.

- [x] `docker/data.Dockerfile`
- [ ] Dockerfiles: winprob API, vision (GPU and CPU variants), RAG API, agent
- [ ] `docker-compose.yml` for local integration
- [ ] kind cluster: data pipeline as a Job, APIs as Deployments + Services, ConfigMaps/Secrets
- [ ] Helm chart or Kustomize overlays (kind vs EKS)
- [ ] AWS: S3 for data and models, ECR for images (with your credentials)
- [ ] EKS: short-lived cluster via eksctl, deploy, record a demo, tear down; cost notes
- [ ] GitHub Actions: lint, test, build and push images on tag
- [ ] Architecture diagram in the README

---

## Log

- 2026-11-04: Scaffold, data pipeline, analytics, chase-state features, tests, data Dockerfile, CI.
- 2026-11-04: Pipeline verified on real IPL data. Win-probability baselines, PyTorch MLP, calibration, evaluation and curves (PR #1).
- 2026-11-04: Run 1 found era drift; added par score, recency weighting and Platt calibration.
