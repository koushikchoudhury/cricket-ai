"""Win probability for a chase described in plain numbers.

One place that turns "chasing 180, 100/2 after 12 overs" into the model's features, used by
the agent tool and the Streamlit app so both always agree.
"""

from __future__ import annotations

import pandas as pd

from cricket_ai.data.features import RRR_CAP

DEFAULT_PAR_RPO = 8.0


def overs_to_balls(overs: float | str) -> int:
    """Cricket notation: 12 -> 72 balls, 12.3 -> 75 balls (the digit after the point is balls)."""
    text = str(overs).strip()
    whole, _, part = text.partition(".")
    balls = int(part[:1] or 0)
    if balls > 5:
        raise ValueError(f"'{overs}' is not valid overs notation (balls after the point are 0-5)")
    return int(whole or 0) * 6 + balls


def chase_state(
    target: int,
    runs_scored: int,
    balls_bowled: int,
    wickets_lost: int,
    total_balls: int = 120,
    par_rpo: float | None = None,
    runs_last_12_balls: int | None = None,
    wickets_last_12_balls: int = 0,
) -> pd.DataFrame:
    """One-row feature frame for a live chase situation."""
    runs_needed = target - runs_scored
    balls_left = total_balls - balls_bowled
    crr = runs_scored * 6 / balls_bowled if balls_bowled else 0.0
    par = (par_rpo or DEFAULT_PAR_RPO) * total_balls / 6
    return pd.DataFrame(
        [
            {
                "runs_needed": runs_needed,
                "balls_left": balls_left,
                "wickets_in_hand": 10 - wickets_lost,
                "target": target,
                "current_run_rate": crr,
                "required_run_rate": min(runs_needed * 6 / max(balls_left, 1), RRR_CAP),
                "runs_last_12_balls": (
                    runs_last_12_balls if runs_last_12_balls is not None else round(crr * 2)
                ),
                "wickets_last_12_balls": wickets_last_12_balls,
                "par_score": par,
                "target_vs_par": target - par,
                "balls_bowled": balls_bowled,
            }
        ]
    )


def chase_probability(
    model,
    target: int,
    runs_scored: int,
    balls_bowled: int,
    wickets_lost: int,
    total_balls: int = 120,
    par_rpo: float | None = None,
    **momentum,
) -> float:
    """Chasing side's win probability; 1.0 / 0.0 once the chase is decided."""
    from cricket_ai.winprob.model import predict_proba

    if target - runs_scored <= 0:
        return 1.0
    if total_balls - balls_bowled <= 0 or wickets_lost >= 10:
        return 0.0
    state = chase_state(
        target, runs_scored, balls_bowled, wickets_lost, total_balls, par_rpo, **momentum
    )
    return float(predict_proba(model, state)[0])
