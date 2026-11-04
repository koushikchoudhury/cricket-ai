"""Image retrieval: find match photos or video frames from a text question.

Images (and video frames extracted later in module 3) are embedded with CLIP, which maps
pictures and text into the same space, so "Kohli celebrating a hundred" can be matched
against pictures directly. Optional captions in ``captions.json`` ({"file.jpg": "text"})
are shown with the results and also used for keyword matching.

The embedder is injectable: ``ClipEmbedder`` on a real machine, a fake one in tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ClipEmbedder:
    def __init__(self, model: str = "clip-ViT-B-32", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model, device=device)

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        return self.model.encode(images, batch_size=32, normalize_embeddings=True).astype(
            np.float32
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)


@dataclass
class ImageHit:
    path: str
    caption: str
    score: float


class ImageIndex:
    def __init__(
        self, paths: list[Path], captions: dict[str, str], embeddings: np.ndarray, embedder
    ):
        self.paths, self.captions = paths, captions
        self.embeddings, self.embedder = embeddings, embedder

    @classmethod
    def build(cls, folder: Path, embedder) -> ImageIndex:
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise FileNotFoundError(f"No images in {folder}")
        cap_file = folder / "captions.json"
        captions = json.loads(cap_file.read_text()) if cap_file.exists() else {}
        return cls(paths, captions, embedder.embed_images(paths), embedder)

    def search(self, query: str, k: int = 3) -> list[ImageHit]:
        q = self.embedder.embed_texts([query])[0]
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [
            ImageHit(
                str(self.paths[i]), self.captions.get(self.paths[i].name, ""), float(scores[i])
            )
            for i in top
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "image_embeddings.npy", self.embeddings)
        (directory / "images.json").write_text(
            json.dumps({"paths": [str(p) for p in self.paths], "captions": self.captions}, indent=2)
        )

    @classmethod
    def load(cls, directory: Path, embedder) -> ImageIndex:
        meta = json.loads((directory / "images.json").read_text())
        return cls(
            [Path(p) for p in meta["paths"]],
            meta["captions"],
            np.load(directory / "image_embeddings.npy"),
            embedder,
        )
