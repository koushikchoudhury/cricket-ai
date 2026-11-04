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
    print(f"{tag:28s} ({secs:4.0f}s) corr {correlation_difference(hold, s):.4f} valid {v['all_rules']:.3f} "
          f"first_over {v['first_over_starts_at_zero']:.3f} r(over,ras) {np.corrcoef(s.over, s.runs_at_start)[0,1]:.3f} "
          f"r(runs,fours) {np.corrcoef(s.runs, s.fours)[0,1]:.3f}", flush=True)

report("REAL holdout", hold, 0)
for epochs in (30, 100):
    t0 = time.time()
    g = TabularGAN(GANConfig(epochs=epochs, batch_size=100, device="cpu", log_every=10**6)).fit(train, DISCRETE)
    secs = time.time() - t0
    report(f"mine e{epochs} bn-eval", g.sample(len(train)), secs)
    g.generator.train(); g.generator.eval = lambda: g.generator  # sample with batch statistics
    report(f"mine e{epochs} bn-train", g.sample(len(train)), secs)

from ctgan import CTGAN
for epochs in (30, 100):
    t0 = time.time(); torch.manual_seed(0)
    c = CTGAN(epochs=epochs, batch_size=100, verbose=False)
    c.fit(train, discrete_columns=DISCRETE)
    report(f"ctgan e{epochs}", c.sample(len(train)), time.time() - t0)
