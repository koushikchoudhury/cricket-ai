# Tracker

Status: ✅ done · 🟡 in progress · ⚪ not started. Update this file in the same commit as the work.

| Module | Status | Next step |
|---|---|---|
| 1. Data, analytics, win probability | 🟡 | Train on real IPL data and review the report; first-innings model |
| 2. Synthetic data | ✅ | Optional: improve wicket utility; innings-level sequence generator |
| 3. Video and image analysis | ⚪ | Source and label a small broadcast-frame dataset |
| 4. Multimodal RAG | 🟡 | Index IPL with a real embedding model; run the eval with an LLM |
| 5. LangGraph agent | 🟡 | Run `agent evaluate` with Ollama or an API and record the results |
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
- [x] Run 2 with the era fixes: calibration better, discrimination worse (below)
- [x] Configurable feature sets; `cricket-ai winprob ablate` (6 variants)
- [ ] Run the ablation and pick the final configuration
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

**Run 2** (2026-11-04): + par_score, target_vs_par, recency half-life 4, Platt calibration.

| model | log loss | Brier | ECE | AUC |
|---|---|---|---|---|
| MLP (PyTorch) | **0.414** | **0.132** | 0.048 | 0.894 |
| Logistic | 0.425 | 0.139 | 0.122 | **0.914** |
| GBM, monotone | 0.575 | 0.203 | 0.228 | 0.910 |
| Base rate | 0.690 | 0.249 | 0.045 | 0.500 |

Findings:
- Platt calibration worked: MLP ECE 0.085 → 0.048.
- But log loss and AUC got worse for every model, so a feature change hurt the model itself.
- MLP validation loss was best after epoch 1 and then rose (0.448 → 0.551) while training loss fell:
  the model learned something that does not transfer to the next season.
- Hypothesis: raw `par_score` is near-constant within a season, so it acts as an era label, and
  2024–26 values lie outside the training range. `target_vs_par` should carry the useful signal.
  The ablation tests this; the default feature set now leaves raw `par_score` out.

## 2. Synthetic data (GAN, CTGAN)

**Deliverables:** a hand-written GAN, a CTGAN baseline and an evaluation report comparing
synthetic and real data.

- [x] Target: over-level table (start-of-over state + what happened), 11 mixed-type columns
- [x] Eight cricket validity rules; real data passes all of them
- [x] Match-based train/holdout split (generators never see holdout matches)
- [x] Preprocessing: mode-specific normalisation (Bayesian GMM) for continuous columns, one-hot for discrete
- [x] Conditional vector and training-by-sampling (log-frequency) for rare categories
- [x] **Hand-written PyTorch GAN**: residual generator, PacGAN critic, WGAN-GP, Gumbel-softmax heads
- [x] Baselines: independent marginals (floor) and CTGAN (`pip install ctgan`)
- [x] Fidelity: TVD / KS per column, correlation difference, distribution and correlation plots
- [x] Utility: TSTR vs TRTR on runs-in-over (MAE) and wicket-in-over (AUC)
- [x] Privacy: distance to closest record vs holdout, membership-inference AUC
- [x] CLI: `cricket-ai synth table / train / evaluate`
- [x] Diagnostics: hand-written GAN matches the CTGAN library; critic dropout off learns ~3x faster
- [x] Train the GAN and CTGAN on IPL (GTX 1650) and record the results
- [ ] Improve utility on rare events (wicket AUC gap)
- [ ] Optional: innings-level sequence generator so synthetic chases can feed the win-probability model

**Metrics:** KS / TV distances, correlation difference, validity rate, TSTR vs TRTR gap, DCR ratio, membership AUC.

### Results on IPL (2026-11-04)

Trained on 38,185 real overs (whole matches held out for evaluation). Hand-written GAN:
300 epochs on a GTX 1650 in 572 s.

| model | mean TVD | corr error | validity | runs MAE | wicket AUC | DCR ratio | membership AUC |
|---|---|---|---|---|---|---|---|
| real (TRTR) | – | – | 1.000 | 3.492 | 0.605 | – | – |
| **GAN (hand-written)** | 0.080 | **0.049** | **0.867** | **3.623** | 0.555 | 2.36 | 0.486 |
| CTGAN library | 0.091 | 0.079 | 0.779 | 3.801 | **0.570** | 3.39 | 0.492 |
| independent marginals | 0.010 | 0.166 | 0.625 | 3.670 | 0.488 | 6.64 | 0.491 |

Findings:
- The hand-written GAN beats the CTGAN library on fidelity, correlations, validity and runs utility.
- Marginals win TVD by construction but break relationships (3x the GAN's correlation error) and
  produce impossible cricket in 37% of rows.
- No memorisation: membership AUC ≈ 0.49 for every model; DCR ratio > 1 means synthetic rows are no
  closer to training rows than unseen real rows are.
- Weak spot: rare events. TSTR wicket AUC recovers about half the real signal (0.555 vs 0.605);
  CTGAN is slightly better there.

### Diagnostics (2026-11-04, CPU in CI, 4,615 simulated overs, batch 100)

The yardstick is the correlation between over number and runs at the start of the over
(0.95 in real data), plus the mean correlation-matrix error (independent marginals: 0.135).

| setting | time | corr error | r(over, runs at start) |
|---|---|---|---|
| hand-written, CTGAN defaults, 100 epochs | 35s | 0.113 | 0.07 |
| CTGAN library, 100 epochs | 45s | 0.116 | 0.03 |
| hand-written, 300 epochs | 104s | 0.076 | 0.58 |
| **hand-written, critic dropout 0, 100 epochs** | 35s | 0.085 | 0.42 |
| critic steps 5 + dropout 0, 60 epochs | 57s | 0.075 | 0.49 |
| learning rate 1e-3, 100 epochs | 36s | 0.134 | 0.00 |

Findings:
- The hand-written GAN performs on par with the reference CTGAN implementation (no bug).
- Both learn strong discrete-to-continuous links slowly; quality keeps improving with steps.
- Removing critic dropout (CTGAN default 0.5) gave the best quality per unit of time, so it is
  the new default. A higher learning rate hurt. A full IPL run (~20k steps) exceeds the best run here.

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

- [x] Corpus from module 1 tables: match reports, innings-phase summaries, player-season summaries
- [x] Metadata on every document (match, season, teams, phase, player) and kind filters
- [x] Hand-written BM25 + embeddings (hashing for CI; sentence-transformers, default bge-small)
- [x] Hybrid search with reciprocal rank fusion; date tokens for exact-date matching
- [x] Structured route: number questions go to LLM-written SQL on DuckDB (read-only, file access off)
- [x] Generation with numbered citations; one retry when the SQL fails
- [x] LLM switch: none / Ollama / Anthropic / OpenAI (`--llm`, `CRICKET_AI_LLM`)
- [x] Image search with CLIP (`cricket-ai rag images <folder>`), ready for frames from module 3
- [x] Eval set generated from the tables: 5 question types with known answers and source ids
- [x] Metrics: recall@1/5/10 and MRR for BM25, dense and hybrid; answer accuracy with an LLM
- [ ] Index IPL with bge-small on the GPU and record retrieval results
- [x] Answer eval run 1 (qwen2.5:3b): retrieval hybrid recall@5 1.00, MRR 0.948; answers 100% but only
      match questions were sampled (eval bug, now stratified across types) and the final's answer mixed
      details from two matches (now caught by the faithfulness check)
- [ ] Answer eval run 2 across all question types, with faithfulness
- [ ] Commentary text (Cricsheet has none; needs another source)
- [ ] Reranker (cross-encoder) and an LLM-judge for faithfulness
- [ ] FastAPI service (module 6)

## 5. LangGraph agent

**Deliverables:** an agent that answers cricket questions by calling the other modules, with
traces and an evaluation.

- [x] Tools: search_cricket (module 4), player_stats and run_sql (modules 1/4), win_probability (module 1),
      generate_synthetic_overs (module 2); offered only when their index or model exists
- [x] LangGraph state graph: agent -> tools -> agent, with a step limit node
- [x] Guardrails: read-only SQL, tool errors returned to the model, unknown tools handled, step limit
- [x] Human-in-the-loop approval for expensive tools (yes/no prompt; `--yes` to skip)
- [x] Tracing: every tool call timed and logged to `reports/<dataset>/agent/traces.jsonl`
      (LangSmith also works by setting `LANGSMITH_TRACING=true` and an API key)
- [x] Task eval generated from the data: tool-choice accuracy, answer accuracy, steps, latency
- [x] CLI (`agent ask / chat / evaluate / graph`) and a Gradio chat UI (`agent ui`)
- [x] Tests with a scripted fake model: tool use, error recovery, step limit, approval, scoring
- [x] Run 1 with Ollama qwen2.5:3b (17 tasks): tool choice 100%, answers 71%; exposed eval and tool flaws (below)
- [x] Fixes: leaderboard tool, labelled batting/bowling output, tolerant season args, numeric faithfulness
- [x] Run 2 with the fixes (below): 15/17 scored correct, but 2 chase answers were wrong yet passed
- [x] Fixes: win_probability takes overs; chase tasks scored against ground truth and expected arguments;
      tool arguments in results.csv; surname fallback for names; leaderboard leads with the leader
- [x] Variance check: three identical runs (same model, code, tasks) scored 88%, 82% and 100%
      (season_runs 0.75 / 0.50 / 1.00). Added `--repeats N`: mean, range across runs, per-task consistency
- [ ] Run 3 with `--repeats 5` and record mean and range
- [ ] Vision tool once module 3 exists

### Run 1 (2026-11-04, Ollama qwen2.5:3b on GTX 1650)

| task | tasks | right tool | correct | median s |
|---|---|---|---|---|
| chase | 4 | 1.00 | 1.00* | 3.3 |
| match_winner | 4 | 1.00 | 1.00 | 7.6 |
| season_runs | 4 | 1.00 | 0.75 | 2.9 |
| synthetic | 1 | 1.00 | 1.00 | 3.2 |
| top_scorer | 4 | 1.00* | 0.00 | 2.2 |

\* overstated, found by reading the answers:
- **top_scorer**: the model used player_stats, which cannot rank players, and the eval counted it
  as a right tool. Added a `leaderboard` tool and fixed the expected tools.
- **chase**: right tool and probability, but two answers misquoted the situation ("120/2" for 100/2,
  "102 balls" for 12). Scoring only checked for a %. Added numeric faithfulness: every number in an
  answer must appear in the question or a tool output, and correct now requires faithful.
- **season_runs miss** (Shubman Gill 2026): the model passed season as a number, the tool schema
  rejected it, and the model reported "no such player". Seasons now accept 2026, "2026 season",
  and "2020" matches "2020/21".
- **Wording**: the 3B model described a bowler's runs conceded as "runs in bowling". player_stats
  now returns labelled sentences (BATTING: scored ... / BOWLING: conceded ...; "did not bowl").

### Run 2 (2026-11-04, same model, after the fixes)

Scored 15/17 correct (run 1: 12/17); tool choice 17/17; top_scorer 0/4 -> 3/4 with the new
leaderboard tool. Reading the answers showed the score was still too high:

- **chase, 2 of 4 wrong but scored correct.** The model called win_probability with
  runs_scored=60 for "100/2" (30.3% instead of ~84%), and with balls_bowled=18 for "18 overs"
  ("10 off 102 balls"). The answers faithfully repeated the wrong tool output, so the
  faithfulness check passed them. Lesson: faithfulness to evidence does not catch wrong
  tool *inputs*. Fixes: the tool now takes overs as people say them, rejects ball counts,
  and chase tasks are scored against the true probability and the expected arguments.
- **season_runs, DA Warner 2019**: "no player named DA Warner", most likely the model expanded
  the name. Names now fall back to the surname, and results.csv records tool arguments.
- **top_scorer 2021**: the leaderboard was right (Gaikwad 635, du Plessis 633) but the model
  read the second line. The tool now starts with "Leader: RD Gaikwad, 635 runs scored".

## App (Streamlit)

**Deliverable:** one app that demonstrates every module from the data and models on disk.

- [x] Pages: Overview, Analytics, Win probability (calculator + match curve + report),
      Synthetic data (generate, compare, download), Ask (sources + SQL), Agent (chat + tool calls)
- [x] Cached loaders keyed by file time, so rebuilt data and retrained models are picked up
- [x] Friendly "not built yet" messages with the command to run; LLM errors shown, not raised
- [x] `cricket-ai app`; headless tests of every page and the main flows (Streamlit AppTest)
- [x] Only language models usable right now are offered (Ollama running with models pulled,
      or an API key and package present); Ollama models picked from the installed list
- [x] Speed (first real use: 18.6s for a one-tool question): Ollama keeps the model loaded for
      30 minutes and is warmed up in the background when Ask or Agent opens; the search embedder
      runs on the CPU so it does not compete with the LLM for 4 GB of GPU memory; the agent sends a
      short SQL hint instead of the full schema with every call; output capped at 512 tokens
- [x] Each answer shows where its time went (model loading / reading / writing, tools) and live
      progress while the agent works
- [x] Follow-up questions keep context (last three question/answer pairs); "how about kohli?"
      after a 2020 question now means Kohli in 2020
- [x] Ask page is a chat; follow-ups are rewritten into standalone questions before search
      (question condensing), and the chat shows how each follow-up was understood. Found when
      "what about kohli ?" after a 2016 question retrieved random Kohli seasons
- [ ] Page for module 3 when it exists
- [ ] Docker image and deployment (module 6)

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
- 2026-11-04: Run 2 mixed (better calibration, worse discrimination); added ablation command.
- 2026-11-04: Module 2 built: over-level table, hand-written GAN, baselines, full evaluation.
- 2026-11-04: Module 2 run on IPL: hand-written GAN beats the CTGAN library on 4 of 5 quality metrics.
- 2026-11-04: Module 4 built: corpus, hybrid search, SQL route, LLM switch, image search, evaluation.
- 2026-11-04: Module 5 built: LangGraph agent with five tools, approval, tracing, evaluation and UI.
- 2026-11-04: First real RAG answer exposed a gap: playoff stage ("Final") was not parsed, so the final was unretrievable. Added the stage and a query-intent ranking vote.
- 2026-11-14: Streamlit app with a page per module, sharing one win-probability helper with the agent.
