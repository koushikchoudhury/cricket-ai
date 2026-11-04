"""Hybrid retrieval: hand-written BM25 + embedding search, fused with reciprocal rank fusion.

* **BM25** is strong on exact tokens: player names, dates, venues.
* **Embeddings** are strong on paraphrase ("best bowling" vs "most wickets").
* **Reciprocal rank fusion** (Cormack et al., 2009) merges the two rankings without having
  to calibrate their scores against each other.

Embedders:

* ``hashing`` - a sparse hashed bag of words and bigrams. Needs no download, so it runs in
  CI and offline; it is lexical, so treat it as a stand-in.
* ``sentence-transformers`` - a real embedding model (default ``BAAI/bge-small-en-v1.5``,
  ~130 MB, runs on the GTX 1650).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

from cricket_ai.rag.corpus import Document, load_documents, save_documents

TOKEN = r"(?u)\b\w+\b"
RRF_K = 60
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? ({MONTHS}),? (\d{{4}})\b", re.IGNORECASE)


def with_date_tokens(text: str) -> str:
    """Append one token per date ('2 April 2020' -> 'd_2_april_2020').

    Lexical search otherwise treats '2' as a common token (it appears in '117/2'), so an
    exact date barely counts and '27 April' can outrank '2 April'.
    """
    tokens = [f"d_{int(d)}_{m.lower()}_{y}" for d, m, y in DATE.findall(text)]
    return f"{text} {' '.join(tokens)}" if tokens else text


def tokenize(text: str) -> list[str]:
    return re.findall(TOKEN, text.lower())


class BM25:
    """Okapi BM25 over a sparse term matrix (k1=1.5, b=0.75)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts: list[str]) -> BM25:
        self.vectorizer = CountVectorizer(token_pattern=TOKEN, lowercase=True)
        tf = self.vectorizer.fit_transform(texts).tocsc().astype(np.float32)
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg = doc_len.mean() if n_docs else 1.0
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # Precompute the BM25 weight of every (doc, term) entry.
        tf = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * doc_len[tf.row] / avg)
        weights = tf.data * (self.k1 + 1) / (tf.data + norm) * self.idf[tf.col]
        self.matrix = sparse.csr_matrix((weights, (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        q.data[:] = 1.0  # each query term counts once
        return np.asarray(self.matrix @ q.T.toarray()).ravel()


class HashingEmbedder:
    name = "hashing"

    def __init__(self, n_features: int = 2**18):
        self.vec = HashingVectorizer(
            n_features=n_features,
            ngram_range=(1, 2),
            token_pattern=TOKEN,
            alternate_sign=False,
            norm="l2",
        )

    def embed_documents(self, texts: list[str]):
        return self.vec.transform(texts)

    def embed_query(self, text: str):
        return self.vec.transform([text])

    def config(self) -> dict:
        return {"name": self.name, "n_features": self.vec.n_features}


class SentenceTransformerEmbedder:
    name = "sentence-transformers"
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model, device=device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self.BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.model.encode([prefix + text], normalize_embeddings=True).astype(np.float32)

    def config(self) -> dict:
        return {"name": self.name, "model": self.model_name}


def make_embedder(name: str = "hashing", model: str | None = None, device: str | None = None):
    if name == "hashing":
        return HashingEmbedder()
    if name in ("sentence-transformers", "st"):
        kwargs = {"device": device} if device and device != "auto" else {}
        return SentenceTransformerEmbedder(model or "BAAI/bge-small-en-v1.5", **kwargs)
    raise ValueError(f"Unknown embedder {name!r}")


@dataclass
class Hit:
    doc: Document
    score: float
    rank: int
    method: str


def _dot(matrix, query) -> np.ndarray:
    out = matrix @ query.T
    out = out.toarray() if sparse.issparse(out) else np.asarray(out)
    return out.ravel()


def rrf(rankings: list[np.ndarray], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, docs: list[Document], embedder, embeddings=None):
        self.docs = docs
        self.embedder = embedder
        texts = [f"{d.title}. {d.text}" for d in docs]
        self.bm25 = BM25().fit([with_date_tokens(t) for t in texts])
        self.embeddings = embeddings if embeddings is not None else embedder.embed_documents(texts)
        self.index = {d.doc_id: i for i, d in enumerate(docs)}

    def _top(self, scores: np.ndarray, n: int, mask: np.ndarray | None) -> np.ndarray:
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return top[np.isfinite(scores[top])]

    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
        kinds: list[str] | None = None,
        candidates: int = 50,
    ) -> list[Hit]:
        mask = None
        if kinds:
            mask = np.array([d.kind in kinds for d in self.docs])
        if method == "bm25":
            s = self.bm25.scores(with_date_tokens(query))
            return [
                Hit(self.docs[i], float(s[i]), r, method)
                for r, i in enumerate(self._top(s, k, mask))
            ]
        dense = _dot(self.embeddings, self.embedder.embed_query(query))
        if method == "dense":
            return [
                Hit(self.docs[i], float(dense[i]), r, method)
                for r, i in enumerate(self._top(dense, k, mask))
            ]
        if method != "hybrid":
            raise ValueError(f"Unknown method {method!r}")
        lexical = self.bm25.scores(with_date_tokens(query))
        fused = rrf([self._top(lexical, candidates, mask), self._top(dense, candidates, mask)])
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(self.docs[i], s, r, method) for r, (i, s) in enumerate(ranked)]

    # ----- persistence -----
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_documents(self.docs, directory / "docs.jsonl")
        if sparse.issparse(self.embeddings):
            sparse.save_npz(directory / "embeddings.npz", sparse.csr_matrix(self.embeddings))
        else:
            np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "index.json").write_text(
            json.dumps({"embedder": self.embedder.config(), "n_docs": len(self.docs)}, indent=2)
        )

    @classmethod
    def load(cls, directory: Path, device: str | None = None) -> HybridRetriever:
        meta = json.loads((directory / "index.json").read_text())
        cfg = meta["embedder"]
        embedder = make_embedder(cfg["name"], cfg.get("model"), device)
        docs = load_documents(directory / "docs.jsonl")
        if (directory / "embeddings.npz").exists():
            emb = sparse.load_npz(directory / "embeddings.npz")
        else:
            emb = np.load(directory / "embeddings.npy")
        return cls(docs, embedder, embeddings=emb)
