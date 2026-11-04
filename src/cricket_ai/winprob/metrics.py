"""Probabilistic metrics: log loss, Brier score, expected calibration error, AUC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "y": y, "p": p})
    t = df.groupby("bin").agg(count=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    t["lower"] = t.index / n_bins
    return t.reset_index(drop=True)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins)
    return float((t["count"] * (t.predicted - t.observed).abs()).sum() / len(y))


def all_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"log_loss": log_loss(y, p), "brier": brier(y, p), "ece": ece(y, p)}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {k: round(v, 4) for k, v in out.items()}


def metrics_by_stage(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Metrics for early, middle and late chase states (by fraction of balls left)."""
    frac = df.balls_left / (df.balls_left + df.balls_bowled).clip(lower=1)
    stage = np.select([frac > 2 / 3, frac > 1 / 3], ["early", "middle"], "late")
    rows = []
    for s in ("early", "middle", "late"):
        mask = stage == s
        if mask.sum():
            rows.append(
                {
                    "stage": s,
                    "states": int(mask.sum()),
                    **all_metrics(df.chasing_team_won.to_numpy()[mask], p[mask]),
                }
            )
    return pd.DataFrame(rows)
