import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from matchgen import generate_match  # noqa: E402

from cricket_ai.data.build import build_tables  # noqa: E402
from cricket_ai.data.features import chase_states  # noqa: E402
from cricket_ai.winprob import model as mlp  # noqa: E402
from cricket_ai.winprob.curve import key_moments, match_curve  # noqa: E402
from cricket_ai.winprob.data import (  # noqa: E402
    FEATURES,
    add_derived,
    date_split,
    recency_weights,
    season_split,
)
from cricket_ai.winprob.evaluate import run_experiment  # noqa: E402
from cricket_ai.winprob.metrics import all_metrics, ece  # noqa: E402


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    # 160 matches spread across four seasons (one match every 9 days)
    for seed in range(160):
        (raw / f"g{seed:03d}.json").write_text(json.dumps(generate_match(seed, day_step=9)))
    built = build_tables(raw, workers=1)
    return built.matches, built.deliveries, chase_states(built.matches, built.deliveries)


FAST = mlp.TrainConfig(max_epochs=25, patience=4, batch_size=512, device="cpu")


def test_season_split_has_no_leakage(tables):
    _, _, states = tables
    split = season_split(states, test_seasons=1, val_seasons=1)
    ids = [set(d.match_id) for d in (split.train, split.val, split.test)]
    assert all(a.isdisjoint(b) for i, a in enumerate(ids) for b in ids[i + 1 :])
    assert split.train.start_date.max() < split.val.start_date.min()
    assert split.val.start_date.max() < split.test.start_date.min()


def test_date_split_and_derived(tables):
    _, _, states = tables
    split = date_split(states)
    assert set(FEATURES).issubset(split.train.columns)
    assert "par_score" not in FEATURES and "target_vs_par" in FEATURES
    assert split.train.start_date.max() <= split.test.start_date.min()
    d = add_derived(states.head(50))
    assert np.allclose(d.rate_gap, d.required_run_rate - d.current_run_rate)


def test_metrics_reference_values():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.3])
    m = all_metrics(y, p)
    assert m["auc"] == 1.0
    assert m["brier"] == pytest.approx((0.01 + 0.01 + 0.04 + 0.09) / 4, abs=1e-4)
    assert ece(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.0)


def test_mlp_learns_and_is_sensible(tables):
    _, _, states = tables
    split = season_split(states, test_seasons=1, val_seasons=1)
    model, hist = mlp.train_mlp(split.train, split.val, FAST)
    assert hist.val_loss[hist.best_epoch - 1] < 0.6  # far better than a coin flip (0.693)
    assert 0.25 <= hist.calib_scale <= 4.0 and -3 <= hist.calib_bias <= 3

    base = split.test.iloc[[0]].copy()
    base[["balls_left", "balls_bowled", "wickets_in_hand", "target"]] = [30, 90, 7, 170]

    def p(**kw):
        row = base.assign(**kw)
        row["current_run_rate"] = (row.target - row.runs_needed) * 6 / row.balls_bowled
        row["required_run_rate"] = row.runs_needed * 6 / row.balls_left
        return float(mlp.predict_proba(model, row)[0])

    assert p(runs_needed=10) > p(runs_needed=40) > p(runs_needed=80)
    assert p(runs_needed=40, wickets_in_hand=8) > p(runs_needed=40, wickets_in_hand=2)


def test_checkpoint_roundtrip(tables, tmp_path):
    _, _, states = tables
    split = date_split(states)
    cfg = mlp.TrainConfig(max_epochs=3, device="cpu")
    model, hist = mlp.train_mlp(split.train, split.val, cfg)
    mlp.save(model, cfg, hist, tmp_path / "m.pt")
    loaded = mlp.load(tmp_path / "m.pt")
    assert np.allclose(mlp.predict_proba(model, split.test), mlp.predict_proba(loaded, split.test))
    assert float(loaded.calib_scale) == pytest.approx(hist.calib_scale)
    assert float(loaded.calib_bias) == pytest.approx(hist.calib_bias)


def test_experiment_and_curve(tables, tmp_path):
    matches, deliveries, states = tables
    result = run_experiment(
        states,
        tmp_path / "rep",
        tmp_path / "models",
        FAST,
        split=season_split(states, test_seasons=1, val_seasons=1),
    )
    metrics = pd.DataFrame(result["test_metrics"]).T
    assert {"base_rate", "logistic", "gbm_monotone", "mlp_pytorch"} == set(metrics.index)
    assert metrics.loc["mlp_pytorch", "log_loss"] < metrics.loc["base_rate", "log_loss"]
    for f in ("report.md", "metrics.json", "reliability.png", "training_curve.png"):
        assert (tmp_path / "rep" / f).exists()

    model = mlp.load(tmp_path / "models" / "winprob_mlp.pt")
    match_id = states.match_id.iloc[0]
    curve = match_curve(model, states, deliveries, match_id)
    assert curve.delivery_seq.iloc[0] == 0 and curve.balls_bowled.iloc[0] == 0
    assert curve.win_prob.between(0, 1).all()
    assert len(key_moments(curve, 3)) == 3


def test_recency_weights(tables):
    _, _, states = tables
    w = recency_weights(states, half_life=2)
    years = states.start_date.dt.year.to_numpy()
    assert w.mean() == pytest.approx(1.0, rel=1e-4)
    latest, oldest = w[years == years.max()][0], w[years == years.min()][0]
    assert latest / oldest == pytest.approx(2 ** ((years.max() - years.min()) / 2), rel=1e-4)
    assert (recency_weights(states, None) == 1).all()


def test_platt_calibration_fixes_a_shift(tables):
    _, _, states = tables
    split = season_split(states, test_seasons=1, val_seasons=1)
    model, _ = mlp.train_mlp(split.train, split.val, FAST)
    x = torch.from_numpy(mlp.feature_matrix(split.val))
    y = torch.from_numpy(split.val.chasing_team_won.to_numpy(dtype=np.float32))
    # Flip a third of the losses to wins: the validation era is now easier to chase in.
    flip = (y == 0) & (torch.arange(len(y)) % 3 == 0)
    y_shift = torch.where(flip, torch.ones_like(y), y)
    _, bias = mlp.calibrate(model, x, y_shift)
    assert bias > 0  # calibration shifts probabilities up, which temperature alone cannot do


def test_ablation_and_feature_sets(tables, tmp_path):
    from cricket_ai.winprob.evaluate import run_ablation

    _, _, states = tables
    split = season_split(states, test_seasons=1, val_seasons=1)
    cfg = mlp.TrainConfig(max_epochs=2, device="cpu")
    table = run_ablation(
        states,
        tmp_path,
        cfg,
        split=split,
        variants=[("core", None), ("core+par+target_vs_par", 2.0)],
    )
    assert len(table) == 4 and set(table.model) == {"mlp", "logistic"}
    assert (tmp_path / "ablation.md").exists()


def test_checkpoint_keeps_its_feature_set(tables, tmp_path):
    from cricket_ai.winprob.data import FEATURE_SETS

    _, _, states = tables
    split = date_split(states)
    cfg = mlp.TrainConfig(max_epochs=2, device="cpu", features=tuple(FEATURE_SETS["core"]))
    model, hist = mlp.train_mlp(split.train, split.val, cfg)
    mlp.save(model, cfg, hist, tmp_path / "core.pt")
    loaded = mlp.load(tmp_path / "core.pt")
    assert loaded.features == FEATURE_SETS["core"]
    assert np.allclose(mlp.predict_proba(model, split.test), mlp.predict_proba(loaded, split.test))
