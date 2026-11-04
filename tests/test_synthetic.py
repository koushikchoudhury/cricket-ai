import json

import numpy as np
import pandas as pd
import pytest
from matchgen import generate_match

from cricket_ai.data.build import build_tables
from cricket_ai.synthetic.baselines import IndependentMarginals
from cricket_ai.synthetic.evaluate import (
    REAL_REFERENCE,
    column_fidelity,
    correlation_difference,
    evaluate_all,
    privacy,
)
from cricket_ai.synthetic.sampler import ConditionalSampler
from cricket_ai.synthetic.table import (
    COLUMNS,
    DISCRETE,
    build_overs,
    split_by_match,
    validity,
)
from cricket_ai.synthetic.transformer import DataTransformer


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    for seed in range(60):
        (raw / f"g{seed:03d}.json").write_text(json.dumps(generate_match(seed, day_step=9)))
    return build_tables(raw, workers=1)


@pytest.fixture(scope="module")
def overs(built):
    return build_overs(built.matches, built.deliveries)


@pytest.fixture(scope="module")
def split(overs):
    return split_by_match(overs)


def test_overs_agree_with_deliveries(built, overs):
    d = built.deliveries
    innings_runs = d.groupby(["match_id", "innings"]).runs_total.sum()
    over_runs = overs.groupby(["match_id", "innings"]).runs.sum()
    assert (innings_runs.loc[over_runs.index] == over_runs).all()
    # runs/wickets at the start of an over equal the running totals of earlier overs
    o = overs.sort_values(["match_id", "innings", "over"])
    g = o.groupby(["match_id", "innings"])
    assert (o.runs_at_start == g.runs.cumsum() - o.runs).all()
    assert (o.wickets_at_start == g.wickets.cumsum() - o.wickets).all()
    assert set(overs.columns) == {"match_id", "season", *COLUMNS}


def test_real_data_obeys_every_rule(overs):
    rates = validity(overs)
    assert all(v == 1.0 for v in rates.values()), rates


def test_validity_catches_impossible_rows(overs):
    bad = overs.head(4)[COLUMNS].copy()
    bad.loc[bad.index[0], "wickets_at_start"] = 9
    bad.loc[bad.index[0], "wickets"] = 3  # 12 wickets
    bad.loc[bad.index[1], "sixes"] = 5
    bad.loc[bad.index[1], "runs"] = 4  # five sixes for four runs
    rates = validity(bad)
    assert rates["wickets_total_le_10"] == 0.75
    assert rates["runs_cover_boundaries_and_extras"] <= 0.75
    assert rates["all_rules"] <= 0.5


def test_split_keeps_matches_whole(overs):
    train, holdout = split_by_match(overs)
    assert len(train) + len(holdout) == len(overs)
    assert 0.1 < len(holdout) / len(overs) < 0.3
    # same seed -> same split
    t2, h2 = split_by_match(overs)
    assert t2.equals(train) and h2.equals(holdout)


def test_transformer_roundtrip(split):
    train, _ = split
    t = DataTransformer(seed=0).fit(train, DISCRETE)
    x = t.transform(train)
    assert x.shape == (len(train), t.output_dim)
    assert x.dtype == np.float32
    back = t.inverse_transform(x)
    for c in COLUMNS:
        assert (back[c].to_numpy() == train[c].to_numpy()).mean() > 0.99, c


def test_conditional_sampler(split):
    train, _ = split
    t = DataTransformer(seed=0).fit(train, DISCRETE)
    x = t.transform(train)
    s = ConditionalSampler(x, t)
    rng = np.random.default_rng(0)
    cond, mask, col, cat = s.sample_condvec(400, rng)
    assert cond.shape == (400, s.cond_dim) and (cond.sum(1) == 1).all()
    assert (mask.sum(1) == 1).all() and s.n_columns == len(DISCRETE)
    idx = s.sample_rows(col, cat, rng)
    starts = np.array([b[0] for b in s.blocks])
    assert (x[idx, starts[col] + cat] == 1).all()
    assert s.sample_original_condvec(7, rng).shape == (7, s.cond_dim)


def test_fidelity_metrics_on_identical_data(split):
    train, _ = split
    fid = column_fidelity(train, train)
    assert (fid.value == 0).all()
    assert correlation_difference(train, train) == 0


def test_privacy_flags_a_copy(split):
    train, holdout = split
    copy = privacy(train, holdout, train.copy())
    assert copy["dcr_median_synthetic"] == 0
    assert copy["membership_auc"] > 0.8
    marg = IndependentMarginals().fit(train).sample(len(train))
    fresh = privacy(train, holdout, marg)
    assert fresh["membership_auc"] < 0.65


def test_evaluate_all_with_marginals(split, tmp_path):
    train, holdout = split
    marg = IndependentMarginals().fit(train).sample(len(train), seed=1)
    result = evaluate_all(train, holdout, {"marginals": marg}, tmp_path)
    table = pd.DataFrame(result["summary"]).set_index("model")
    assert {REAL_REFERENCE, "marginals"} == set(table.index)
    assert table.loc["marginals", "validity"] < 1.0  # independent columns break cricket rules
    for f in ("report.md", "metrics.json", "marginals.png", "correlations.png"):
        assert (tmp_path / f).exists()


def test_gan_trains_samples_and_reloads(split, tmp_path):
    pytest.importorskip("torch")
    from cricket_ai.synthetic.gan import GANConfig, TabularGAN

    train, _ = split
    cfg = GANConfig(epochs=3, batch_size=200, device="cpu", log_every=1)
    gan = TabularGAN(cfg).fit(train, DISCRETE)
    assert len(gan.history.generator_loss) == 3
    assert all(np.isfinite(gan.history.critic_loss))

    synth = gan.sample(350, seed=7)
    assert list(synth.columns) == COLUMNS and len(synth) == 350
    for c in DISCRETE:
        assert set(synth[c]).issubset(set(train[c])), c
    assert synth.runs.between(train.runs.min(), train.runs.max()).all()

    gan.save(tmp_path / "gan")
    loaded = TabularGAN.load(tmp_path / "gan")
    assert loaded.sample(350, seed=7).equals(synth)


def test_gan_learns_structure(split):
    """A short run should already beat independent columns on correlations."""
    pytest.importorskip("torch")
    from cricket_ai.synthetic.gan import GANConfig, TabularGAN

    train, holdout = split
    gan = TabularGAN(GANConfig(epochs=40, batch_size=200, device="cpu")).fit(train, DISCRETE)
    synth = gan.sample(len(train))
    marg = IndependentMarginals().fit(train).sample(len(train))
    assert correlation_difference(holdout, synth) < correlation_difference(holdout, marg)
