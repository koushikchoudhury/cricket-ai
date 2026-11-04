"""Feature matrix and leakage-free splits for the win-probability model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "runs_needed",
    "balls_left",
    "wickets_in_hand",
    "target",
    "current_run_rate",
    "required_run_rate",
    "runs_last_12_balls",
    "wickets_last_12_balls",
    "par_score",
    "target_vs_par",
]
DERIVED_FEATURES = ["rate_gap", "runs_per_wicket_needed", "frac_balls_left"]
FEATURES = BASE_FEATURES + DERIVED_FEATURES

# Direction each feature should push win probability (+1 up, -1 down, 0 free).
# Used as monotonic constraints for gradient boosting and as sanity checks for the MLP.
MONOTONIC = {
    "runs_needed": -1,
    "balls_left": 1,
    "wickets_in_hand": 1,
    "target": 0,
    "current_run_rate": 0,
    "required_run_rate": -1,
    "runs_last_12_balls": 0,
    "wickets_last_12_balls": 0,
    "par_score": 0,
    "target_vs_par": -1,
    "rate_gap": -1,
    "runs_per_wicket_needed": -1,
    "frac_balls_left": 0,
}
LABEL = "chasing_team_won"


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rate_gap"] = out.required_run_rate - out.current_run_rate
    out["runs_per_wicket_needed"] = out.runs_needed / out.wickets_in_hand.clip(lower=1)
    total_balls = (out.balls_left + out.balls_bowled).clip(lower=1)
    out["frac_balls_left"] = out.balls_left / total_balls
    return out


def recency_weights(df: pd.DataFrame, half_life: float | None) -> np.ndarray:
    """Sample weights that halve every ``half_life`` seasons back from the latest season.

    Recent seasons describe today's game better; older seasons still add volume.
    ``None`` or 0 gives equal weights.
    """
    if not half_life:
        return np.ones(len(df), dtype=np.float32)
    years = df.start_date.dt.year.to_numpy()
    age = years.max() - years
    w = 0.5 ** (age / half_life)
    return (w / w.mean()).astype(np.float32)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    if not set(DERIVED_FEATURES).issubset(df.columns):
        df = add_derived(df)
    return df[FEATURES].to_numpy(dtype=np.float32)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    description: str

    def sizes(self) -> dict[str, dict[str, int]]:
        return {
            name: {"states": len(df), "matches": int(df.match_id.nunique())}
            for name, df in (("train", self.train), ("val", self.val), ("test", self.test))
        }


def season_split(df: pd.DataFrame, test_seasons: int = 2, val_seasons: int = 1) -> Split:
    """Train on early seasons, tune on the next, test on the most recent. No match is shared."""
    years = df.start_date.dt.year
    seasons = sorted(years.unique())
    if len(seasons) < test_seasons + val_seasons + 1:
        raise ValueError(f"Need more than {test_seasons + val_seasons} seasons, got {seasons}")
    test_set = seasons[-test_seasons:]
    val_set = seasons[-(test_seasons + val_seasons) : -test_seasons]
    train_set = seasons[: -(test_seasons + val_seasons)]
    desc = (
        f"train {train_set[0]}-{train_set[-1]}, val {val_set[0]}-{val_set[-1]}, "
        f"test {test_set[0]}-{test_set[-1]}"
    )
    return Split(
        add_derived(df[years.isin(train_set)]),
        add_derived(df[years.isin(val_set)]),
        add_derived(df[years.isin(test_set)]),
        desc,
    )


def date_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.2) -> Split:
    """Chronological split by match for datasets with too few seasons."""
    order = df.groupby("match_id").start_date.first().sort_values(kind="stable").index.to_numpy()
    n = len(order)
    n_test, n_val = max(1, int(n * test_frac)), max(1, int(n * val_frac))
    test_ids, val_ids = set(order[n - n_test :]), set(order[n - n_test - n_val : n - n_test])
    is_test, is_val = df.match_id.isin(test_ids), df.match_id.isin(val_ids)
    return Split(
        add_derived(df[~is_test & ~is_val]),
        add_derived(df[is_val]),
        add_derived(df[is_test]),
        f"chronological by match: {n - n_test - n_val}/{n_val}/{n_test} matches",
    )


def auto_split(df: pd.DataFrame) -> Split:
    try:
        return season_split(df)
    except ValueError:
        return date_split(df)
