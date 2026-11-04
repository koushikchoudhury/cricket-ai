"""Conditional vectors and training-by-sampling (Xu et al., 2019).

Each training step picks one discrete column at random and a category of it, with
probability proportional to log(frequency). The generator is asked for rows with that
category and the critic is shown real rows that have it. Rare categories (a six-wicket
collapse, a 30-run over) are therefore seen far more often than their raw frequency,
which counters mode collapse on imbalanced columns.
"""

from __future__ import annotations

import numpy as np

from cricket_ai.synthetic.transformer import DataTransformer


class ConditionalSampler:
    def __init__(self, data: np.ndarray, transformer: DataTransformer, log_frequency=True):
        self.blocks: list[tuple[int, int]] = []  # (start in data matrix, n categories)
        st = 0
        for info in transformer.info:
            if info.kind == "discrete":
                self.blocks.append((st, len(info.categories)))
            st += info.output_dim

        self.n_columns = len(self.blocks)
        self.cond_dim = sum(n for _, n in self.blocks)
        self.cond_offsets = np.cumsum([0] + [n for _, n in self.blocks[:-1]]).astype(int)
        max_n = max((n for _, n in self.blocks), default=0)

        self.train_cum = np.ones((self.n_columns, max_n))
        self.original_cum = np.ones((self.n_columns, max_n))
        self.rows: list[list[np.ndarray]] = []
        for c, (start, n) in enumerate(self.blocks):
            onehot = data[:, start : start + n]
            freq = onehot.sum(axis=0)
            train_p = np.log(freq + 1) if log_frequency else freq.astype(float)
            self.train_cum[c, :n] = np.cumsum(train_p / train_p.sum())
            self.original_cum[c, :n] = np.cumsum(freq / freq.sum())
            self.rows.append([np.flatnonzero(onehot[:, k]) for k in range(n)])
        self.n_rows = len(data)

    def _draw(self, cum: np.ndarray, col: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(len(col))[:, None]
        cat = (u > cum[col]).sum(axis=1)
        limits = np.array([n for _, n in self.blocks])[col] - 1
        return np.minimum(cat, limits)

    def _onehot(self, col: np.ndarray, cat: np.ndarray) -> np.ndarray:
        cond = np.zeros((len(col), self.cond_dim), dtype=np.float32)
        cond[np.arange(len(col)), self.cond_offsets[col] + cat] = 1.0
        return cond

    def sample_condvec(self, batch: int, rng: np.random.Generator):
        """Training conditions: returns (cond, mask, col, cat)."""
        if self.n_columns == 0:
            return None, None, None, None
        col = rng.integers(0, self.n_columns, batch)
        cat = self._draw(self.train_cum, col, rng)
        mask = np.zeros((batch, self.n_columns), dtype=np.float32)
        mask[np.arange(batch), col] = 1.0
        return self._onehot(col, cat), mask, col, cat

    def sample_original_condvec(self, batch: int, rng: np.random.Generator) -> np.ndarray | None:
        """Generation conditions: categories drawn at their real frequency."""
        if self.n_columns == 0:
            return None
        col = rng.integers(0, self.n_columns, batch)
        return self._onehot(col, self._draw(self.original_cum, col, rng))

    def sample_rows(
        self, col: np.ndarray | None, cat: np.ndarray | None, rng: np.random.Generator
    ) -> np.ndarray:
        """Indices of real rows that have the requested category in the requested column."""
        if col is None:
            return rng.integers(0, self.n_rows, len(cat) if cat is not None else 1)
        idx = np.empty(len(col), dtype=np.int64)
        pairs = col.astype(np.int64) * 10_000 + cat
        for p in np.unique(pairs):
            where = np.flatnonzero(pairs == p)
            pool = self.rows[p // 10_000][p % 10_000]
            idx[where] = pool[rng.integers(0, len(pool), len(where))]
        return idx
