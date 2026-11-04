"""Temporary diagnostic: does the GAN improve with more training steps?"""
import json, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, "tests")
import numpy as np
from matchgen import generate_match
from cricket_ai.data.build import build_tables
from cricket_ai.synthetic.table import build_overs, split_by_match, validity, DISCRETE
from cricket_ai.synthetic.gan import GANConfig, TabularGAN
from cricket_ai.synthetic.baselines import IndependentMarginals
from cricket_ai.synthetic.evaluate import correlation_difference, column_fidelity

tmp = Path(tempfile.mkdtemp())
for s in range(150):
    (tmp / f"g{s:03d}.json").write_text(json.dumps(generate_match(s, day_step=9)))
b = build_tables(tmp, workers=1)
train, hold = split_by_match(build_overs(b.matches, b.deliveries))
print("train rows", len(train), flush=True)
marg = IndependentMarginals().fit(train).sample(len(train))
print(f"marginals: corr {correlation_difference(hold, marg):.4f} validity {validity(marg)['all_rules']}", flush=True)
print(f"real r(over, runs_at_start) {np.corrcoef(hold.over, hold.runs_at_start)[0,1]:.3f}", flush=True)
for epochs in (5, 30, 100):
    t0 = time.time()
    g = TabularGAN(GANConfig(epochs=epochs, batch_size=100, device="cpu", log_every=10**6)).fit(train, DISCRETE)
    s = g.sample(len(train))
    fid = column_fidelity(hold, s)
    steps = epochs * (len(train) // 100)
    print(f"epochs {epochs:3d} steps {steps:5d} ({time.time()-t0:.0f}s): corr {correlation_difference(hold, s):.4f} "
          f"tvd {fid[fid.metric=='tvd'].value.mean():.4f} ks {fid[fid.metric=='ks'].value.mean():.4f} "
          f"validity {validity(s)['all_rules']} r(over,runs_at_start) {np.corrcoef(s.over, s.runs_at_start)[0,1]:.3f} "
          f"G {g.history.generator_loss[-1]:.3f} C {g.history.critic_loss[-1]:.3f}", flush=True)
