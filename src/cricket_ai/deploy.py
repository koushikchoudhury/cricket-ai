"""Package the built data and models so the app can run somewhere else.

The tables, search index and trained models are deliberately not in Git. For a hosted app,
``cricket-ai bundle ipl`` zips exactly what the app reads, with a manifest of checksums.
Upload the zip somewhere you control (a GitHub Release asset works), set
``CRICKET_AI_BUNDLE_URL`` on the host, and the app downloads, verifies and unpacks it the
first time it starts.

Only load a bundle you built: the model files are PyTorch checkpoints, which can run code
when loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from cricket_ai import __version__, config

BUNDLE_VERSION = 1
MANIFEST = "manifest.json"
MARKER = ".bundle.json"

# (area, relative paths or folders within that area's <dataset> folder)
CONTENTS: dict[str, list[str]] = {
    "processed": [
        "matches.parquet",
        "deliveries.parquet",
        "players.parquet",
        "chase_states.parquet",
        "overs.parquet",
        "quality_report.json",
        "rag",
        "synthetic/gan.parquet",
    ],
    "models": ["winprob_mlp.pt", "synth_gan"],
    "reports": ["winprob", "synthetic"],
}
REQUIRED = {
    "processed/matches.parquet",
    "processed/deliveries.parquet",
    "processed/players.parquet",
}


def _roots(dataset: str) -> dict[str, Path]:
    return {
        "processed": config.processed_dir(dataset),
        "models": config.MODELS_DIR / dataset,
        "reports": config.REPORTS_DIR / dataset,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class BundleReport:
    path: Path
    files: int
    bytes: int
    missing: list[str]


def create_bundle(dataset: str, out: Path) -> BundleReport:
    """Zip what the app needs for ``dataset``. Missing optional pieces are reported."""
    roots = _roots(dataset)
    files: dict[str, Path] = {}
    missing = []
    for area, entries in CONTENTS.items():
        for entry in entries:
            source = roots[area] / entry
            key = f"{area}/{entry}"
            if source.is_dir():
                for f in sorted(p for p in source.rglob("*") if p.is_file()):
                    files[f"{area}/{f.relative_to(roots[area]).as_posix()}"] = f
            elif source.is_file():
                files[key] = source
            else:
                missing.append(key)
    absent = REQUIRED - set(files)
    if absent:
        raise FileNotFoundError(
            f"The bundle needs {sorted(absent)}. Run `cricket-ai data build {dataset}` first."
        )

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "cricket_ai_version": __version__,
        "dataset": dataset,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": {k: {"sha256": _sha256(p), "bytes": p.stat().st_size} for k, p in files.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST, json.dumps(manifest, indent=2))
        for key, path in files.items():
            zf.write(path, key)
    return BundleReport(out, len(files), out.stat().st_size, missing)


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"Unsafe path in bundle: {relative}")
    return target


def install_bundle(zip_path: Path) -> dict:
    """Verify every file against the manifest, then unpack into the data/model/report folders."""
    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read(MANIFEST))
        if manifest.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError(f"Unsupported bundle version {manifest.get('bundle_version')}")
        dataset = manifest["dataset"]
        roots = _roots(dataset)
        names = set(zf.namelist()) - {MANIFEST}
        if names != set(manifest["files"]):
            raise ValueError("Bundle contents do not match its manifest")
        with tempfile.TemporaryDirectory() as tmp:
            staged: list[tuple[Path, Path]] = []
            for name, meta in manifest["files"].items():
                area, _, relative = name.partition("/")
                if area not in roots:
                    raise ValueError(f"Unexpected area in bundle: {name}")
                staged_path = _safe_target(Path(tmp), name)
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(staged_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if _sha256(staged_path) != meta["sha256"]:
                    raise ValueError(f"Checksum mismatch for {name}")
                staged.append((staged_path, _safe_target(roots[area], relative)))
            for source, target in staged:  # only after everything verified
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), target)
    marker = {k: manifest[k] for k in ("dataset", "created", "cricket_ai_version")}
    (roots["processed"] / MARKER).write_text(json.dumps(marker, indent=2))
    return manifest


def download(url: str, dest: Path, timeout: int = 300) -> Path:
    import requests

    with requests.get(url, stream=True, timeout=timeout) as reply:
        reply.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in reply.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def ensure_bundle(url: str | None = None, dataset: str = "ipl") -> str:
    """Install the bundle from ``url`` (or CRICKET_AI_BUNDLE_URL) unless data is already here.

    Returns what happened: 'present', 'installed' or 'no-url'.
    """
    if (config.processed_dir(dataset) / "matches.parquet").exists():
        return "present"
    url = url or os.environ.get("CRICKET_AI_BUNDLE_URL")
    if not url:
        return "no-url"
    with tempfile.TemporaryDirectory() as tmp:
        install_bundle(download(url, Path(tmp) / "bundle.zip"))
    return "installed"
