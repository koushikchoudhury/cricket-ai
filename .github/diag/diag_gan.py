"""Temporary diagnostic: hand-written GAN vs the CTGAN library on the same data."""

import json, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, "tests")
import numpy as np, torch
from matchgen import generate_match
from cricket_ai.data.build import build_tables
from cricket_ai.synthetic.table import build_overs, split_by_match, validity, DISCRETE
from cricket_ai.synthetic.gan import GANConfig, TabularGAN
from cricket_ai.synthetic.evaluate import correlation_difference

tmp = Path(tempfile.mkdtemp())
for s in range(150):
    (tmp / f"g{s:03d}.json").write_text(json.dumps(generate_match(s, day_step=9)))
b = build_tables(tmp, workers=1)
train, hold = split_by_match(build_overs(b.matches, b.deliveries))
print("train rows", len(train), flush=True)


def report(tag, s, secs):
    v = validity(s)
    print(
        f"{tag:28s} ({secs:4.0f}s) corr {correlation_difference(hold, s):.4f} valid {v['all_rules']:.3f} "
        f"first_over {v['first_over_starts_at_zero']:.3f} r(over,ras) {np.corrcoef(s.over, s.runs_at_start)[0, 1]:.3f} "
        f"r(runs,fours) {np.corrcoef(s.runs, s.fours)[0, 1]:.3f}",
        flush=True,
    )


report("REAL holdout", hold, 0)
variants = [
    ("baseline e100", dict(epochs=100)),
    ("baseline e300", dict(epochs=300)),
    ("critic_steps5 e60", dict(epochs=60, critic_steps=5)),
    ("lr1e-3 e100", dict(epochs=100, lr_generator=1e-3, lr_critic=1e-3)),
    ("no-dropout e100", dict(epochs=100, critic_dropout=0.0)),
    ("cs5+lr1e-3 e60", dict(epochs=60, critic_steps=5, lr_generator=1e-3, lr_critic=1e-3)),
    ("cs5+nodrop e60", dict(epochs=60, critic_steps=5, critic_dropout=0.0)),
]
for tag, kw in variants:
    t0 = time.time()
    g = TabularGAN(GANConfig(batch_size=100, device="cpu", log_every=10**6, **kw)).fit(
        train, DISCRETE
    )
    report(tag, g.sample(len(train)), time.time() - t0)
