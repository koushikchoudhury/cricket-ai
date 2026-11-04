"""Win-probability curve and key moments for one chase."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cricket_ai.data.features import RRR_CAP  # noqa: E402
from cricket_ai.winprob.model import WinProbMLP, predict_proba  # noqa: E402


def _start_state(first: pd.Series) -> pd.DataFrame:
    """The state before the first ball of the chase."""
    total = int(first.balls_left + first.balls_bowled)
    return pd.DataFrame(
        [
            {
                **first.to_dict(),
                "delivery_seq": 0,
                "runs_needed": int(first.target),
                "balls_left": total,
                "balls_bowled": 0,
                "runs_scored": 0,
                "wickets_in_hand": 10,
                "current_run_rate": 0.0,
                "required_run_rate": min(first.target * 6.0 / max(total, 1), RRR_CAP),
                "runs_last_12_balls": 0,
                "wickets_last_12_balls": 0,
            }
        ]
    )


def match_curve(
    model: WinProbMLP, states: pd.DataFrame, deliveries: pd.DataFrame, match_id: str
) -> pd.DataFrame:
    s = states[states.match_id == match_id].sort_values("delivery_seq")
    if s.empty:
        raise KeyError(f"No live chase states for match {match_id}")
    s = pd.concat([_start_state(s.iloc[0]), s], ignore_index=True)
    s["win_prob"] = predict_proba(model, s)
    d = deliveries[(deliveries.match_id == match_id) & (deliveries.innings == 2)]
    cols = [
        "delivery_seq",
        "over",
        "ball_in_over",
        "batter",
        "bowler",
        "runs_total",
        "is_wicket",
        "player_out",
        "wicket_kind",
    ]
    s = s.merge(d[cols], on="delivery_seq", how="left")
    s["swing"] = s.win_prob.diff().fillna(0.0)
    return s


def key_moments(curve: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    top = curve.reindex(curve.swing.abs().sort_values(ascending=False).index).head(k)
    top = top.assign(
        ball=lambda t: (
            t.over.astype("Int64").astype(str) + "." + t.ball_in_over.astype("Int64").astype(str)
        )
    )
    return top[["ball", "batter", "bowler", "runs_total", "player_out", "win_prob", "swing"]]


def plot_curve(curve: pd.DataFrame, title: str, path: Path) -> None:
    x = curve.balls_bowled.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, curve.win_prob * 100, lw=2)
    ax.axhline(50, c="grey", lw=0.8, ls=":")
    w = curve[curve.is_wicket.fillna(False).astype(bool)]
    ax.scatter(w.balls_bowled, w.win_prob * 100, c="crimson", zorder=3, s=28, label="wicket")
    ticks = np.arange(0, x.max() + 1, 12)
    ax.set_xticks(ticks, [str(t // 6) for t in ticks])
    ax.set(xlabel="overs", ylabel="chasing side win %", ylim=(0, 100), title=title)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
