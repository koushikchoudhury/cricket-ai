"""Download and extract Cricsheet JSON archives.

Cricsheet (https://cricsheet.org) publishes ball-by-ball data under the
Open Data Commons Attribution License. Archive names follow
``https://cricsheet.org/downloads/<dataset>_json.zip`` (for example ``ipl``,
``t20s``, ``odis``, ``tests``, ``bbl``, ``wpl``).
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

CRICSHEET_URL = "https://cricsheet.org/downloads/{dataset}_json.zip"
_DATASET_RE = re.compile(r"^[a-z0-9_]+$")


def dataset_url(dataset: str) -> str:
    if not _DATASET_RE.match(dataset):
        raise ValueError(f"Invalid dataset name: {dataset!r}")
    return CRICSHEET_URL.format(dataset=dataset)


def safe_extract(zip_path: Path, dest: Path) -> list[Path]:
    """Extract only .json files, rejecting any path that escapes ``dest`` (zip-slip)."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.endswith(".json"):
                continue
            target = (dest / Path(member.filename).name).resolve()
            if target.parent != dest:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(target)
    return extracted


def download_dataset(dataset: str, dest: Path, timeout: int = 120) -> list[Path]:
    """Download a Cricsheet archive and extract its match JSON files into ``dest``."""
    url = dataset_url(dataset)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{dataset}.zip"
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return safe_extract(zip_path, dest)
