"""Baselines the MLP has to beat: base rate, logistic regression and monotone gradient boosting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cricket_ai.winprob.data import FEATURES, LABEL, MONOTONIC, feature_matrix, recency_weights


class BaseRate:
    """Predicts the training-set chasing win rate for every state."""

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> BaseRate:
        self.rate_ = float(np.average(y, weights=sample_weight))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


def make_baselines(seed: int = 42) -> dict[str, object]:
    return {
        "base_rate": BaseRate(),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "gbm_monotone": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            monotonic_cst=[MONOTONIC[f] for f in FEATURES],
            # Early stopping uses the season-based validation split passed to fit(), not a
            # random row sample: rows from one match are correlated, so a random holdout leaks.
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        ),
    }


def fit_baselines(
    train: pd.DataFrame,
    val: pd.DataFrame,
    seed: int = 42,
    half_life: float | None = None,
) -> dict[str, object]:
    X, y = feature_matrix(train), train[LABEL].to_numpy()
    X_val, y_val = feature_matrix(val), val[LABEL].to_numpy()
    w = recency_weights(train, half_life)
    models = make_baselines(seed)
    for name, model in models.items():
        if name == "gbm_monotone":
            model.fit(X, y, sample_weight=w, X_val=X_val, y_val=y_val)
        elif name == "logistic":
            model.fit(X, y, logisticregression__sample_weight=w)
        else:
            model.fit(X, y, sample_weight=w)
    return models


def predict(model: object, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(feature_matrix(df))[:, 1]
