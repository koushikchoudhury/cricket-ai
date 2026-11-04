"""Reversible encoding of a mixed-type table into the matrix the GAN works on.

* Discrete columns become one-hot blocks (decoded with softmax).
* Continuous columns use mode-specific normalisation (Xu et al., 2019): a Bayesian
  Gaussian mixture finds the modes of the column, each value is encoded as the mode it
  belongs to (one-hot) plus its offset within that mode scaled to [-1, 1] (decoded with
  tanh). Multi-modal, skewed columns such as "runs at start of over" are far easier to
  learn this way than with plain standardisation.

The mixture is fitted with scikit-learn; everything the network does is plain PyTorch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture


@dataclass
class Span:
    dim: int
    activation: str  # "tanh" or "softmax"


@dataclass
class ColumnInfo:
    name: str
    kind: str  # "continuous" or "discrete"
    spans: list[Span]
    categories: np.ndarray | None = None
    gmm: BayesianGaussianMixture | None = None
    valid_modes: np.ndarray | None = None
    means: np.ndarray | None = None
    stds: np.ndarray | None = None
    lo: float = field(default=-np.inf)
    hi: float = field(default=np.inf)
    integer: bool = False

    @property
    def output_dim(self) -> int:
        return sum(s.dim for s in self.spans)


class DataTransformer:
    def __init__(
        self,
        max_modes: int = 10,
        weight_threshold: float = 0.005,
        max_fit_rows: int = 50_000,
        seed: int = 0,
    ):
        self.max_modes = max_modes
        self.weight_threshold = weight_threshold
        self.max_fit_rows = max_fit_rows
        self.seed = seed
        self.columns: list[str] = []
        self.info: list[ColumnInfo] = []

    # ----- fitting -----
    def fit(self, df: pd.DataFrame, discrete_columns: list[str]) -> DataTransformer:
        self.columns = list(df.columns)
        self.info = []
        for col in self.columns:
            values = df[col].to_numpy()
            if col in discrete_columns:
                cats = np.sort(pd.unique(values))
                self.info.append(
                    ColumnInfo(col, "discrete", [Span(len(cats), "softmax")], categories=cats)
                )
            else:
                self.info.append(self._fit_continuous(col, values.astype(float)))
        return self

    def _fit_continuous(self, col: str, x: np.ndarray) -> ColumnInfo:
        rng = np.random.default_rng(self.seed)
        fit_x = x if len(x) <= self.max_fit_rows else rng.choice(x, self.max_fit_rows, False)
        with warnings.catch_warnings():
            # Integer-valued columns rarely reach the tolerance; the fitted modes are still fine.
            warnings.simplefilter("ignore", ConvergenceWarning)
            gmm = BayesianGaussianMixture(
                n_components=self.max_modes,
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=0.001,
                max_iter=200,
                random_state=self.seed,
            ).fit(fit_x.reshape(-1, 1))
        valid = gmm.weights_ > self.weight_threshold
        if not valid.any():
            valid[np.argmax(gmm.weights_)] = True
        return ColumnInfo(
            col,
            "continuous",
            [Span(1, "tanh"), Span(int(valid.sum()), "softmax")],
            gmm=gmm,
            valid_modes=valid,
            means=gmm.means_.ravel()[valid],
            stds=np.sqrt(gmm.covariances_.ravel()[valid]).clip(min=1e-6),
            lo=float(x.min()),
            hi=float(x.max()),
            integer=bool(np.allclose(x, np.round(x))),
        )

    # ----- encode / decode -----
    @property
    def output_info(self) -> list[Span]:
        return [s for info in self.info for s in info.spans]

    @property
    def output_dim(self) -> int:
        return sum(info.output_dim for info in self.info)

    def transform(self, df: pd.DataFrame, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(self.seed)
        blocks = []
        for info in self.info:
            values = df[info.name].to_numpy()
            if info.kind == "discrete":
                index = {c: i for i, c in enumerate(info.categories)}
                idx = np.array([index[v] for v in values])
                block = np.zeros((len(values), len(info.categories)), dtype=np.float32)
                block[np.arange(len(values)), idx] = 1.0
                blocks.append(block)
            else:
                blocks.append(self._transform_continuous(info, values.astype(float), rng))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def _transform_continuous(
        self, info: ColumnInfo, x: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        probs = info.gmm.predict_proba(x.reshape(-1, 1))[:, info.valid_modes] + 1e-6
        probs = probs / probs.sum(axis=1, keepdims=True)
        # Sample the mode from the posterior (as in CTGAN) rather than taking the argmax.
        u = rng.random(len(x))[:, None]
        mode = (u > probs.cumsum(axis=1)).sum(axis=1).clip(max=probs.shape[1] - 1)
        alpha = (x - info.means[mode]) / (4 * info.stds[mode])
        block = np.zeros((len(x), 1 + len(info.means)), dtype=np.float32)
        block[:, 0] = alpha.clip(-0.99, 0.99)
        block[np.arange(len(x)), 1 + mode] = 1.0
        return block

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        out: dict[str, np.ndarray] = {}
        st = 0
        for info in self.info:
            if info.kind == "discrete":
                n = len(info.categories)
                out[info.name] = info.categories[np.argmax(data[:, st : st + n], axis=1)]
                st += n
            else:
                n = len(info.means)
                alpha = np.clip(data[:, st], -1, 1)
                mode = np.argmax(data[:, st + 1 : st + 1 + n], axis=1)
                x = alpha * 4 * info.stds[mode] + info.means[mode]
                # Keep values inside the observed range, as SDV does by default.
                x = np.clip(x, info.lo, info.hi)
                out[info.name] = np.round(x).astype(np.int64) if info.integer else x
                st += 1 + n
        return pd.DataFrame(out)[self.columns]
