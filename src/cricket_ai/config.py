"""Project paths. Override the data root with the CRICKET_AI_DATA_DIR environment variable."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("CRICKET_AI_DATA_DIR", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path(os.environ.get("CRICKET_AI_REPORTS_DIR", "reports")).resolve()
MODELS_DIR = Path(os.environ.get("CRICKET_AI_MODELS_DIR", "models")).resolve()


def raw_dir(dataset: str) -> Path:
    return RAW_DIR / dataset


def processed_dir(dataset: str) -> Path:
    return PROCESSED_DIR / dataset
