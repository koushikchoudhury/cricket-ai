"""Baselines for the hand-written GAN.

* ``IndependentMarginals``: samples every column independently from its real
  distribution. Perfect column fidelity, no relationships between columns. Any model
  worth using has to beat it on correlations, validity and utility.
* ``CTGANBaseline``: the reference CTGAN implementation (``pip install ctgan``), to show
  how the hand-written GAN compares with the library it re-implements.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


class IndependentMarginals:
    name = "marginals"

    def fit(self, df: pd.DataFrame, discrete_columns: list[str] | None = None):
        self.columns = list(df.columns)
        self.values = {c: df[c].to_numpy() for c in self.columns}
        return self

    def sample(self, n: int, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({c: rng.choice(v, n, replace=True) for c, v in self.values.items()})


class CTGANBaseline:
    name = "ctgan"

    def __init__(self, epochs: int = 300, device: str = "auto", seed: int = 42):
        self.epochs = epochs
        self.device = device
        self.seed = seed

    def fit(self, df: pd.DataFrame, discrete_columns: list[str]):
        try:
            from ctgan import CTGAN
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("CTGAN baseline needs `pip install ctgan`") from exc
        import torch

        use_gpu = torch.cuda.is_available() if self.device == "auto" else self.device == "cuda"
        params = inspect.signature(CTGAN).parameters
        gpu_arg = {"enable_gpu": use_gpu} if "enable_gpu" in params else {"cuda": use_gpu}
        torch.manual_seed(self.seed)
        self.model = CTGAN(epochs=self.epochs, verbose=True, **gpu_arg)
        self.model.fit(df, discrete_columns=discrete_columns)
        return self

    def sample(self, n: int, seed: int = 42) -> pd.DataFrame:
        import torch

        torch.manual_seed(seed)
        return self.model.sample(n)
