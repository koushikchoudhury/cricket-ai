"""Bundle, cloud settings and the question budget."""

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from cricket_ai import config
from cricket_ai.deploy import create_bundle, ensure_bundle, install_bundle


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point the config at a temporary data/models/reports tree."""

    def use(root: Path):
        monkeypatch.setattr(config, "PROCESSED_DIR", root / "data" / "processed")
        monkeypatch.setattr(config, "MODELS_DIR", root / "models")
        monkeypatch.setattr(config, "REPORTS_DIR", root / "reports")

    src = tmp_path / "laptop"
    processed = src / "data" / "processed" / "ipl"
    (processed / "rag").mkdir(parents=True)
    for name in ("matches.parquet", "deliveries.parquet", "players.parquet"):
        (processed / name).write_bytes(name.encode() * 10)
    (processed / "rag" / "index.json").write_text("{}")
    (src / "models" / "ipl" / "synth_gan").mkdir(parents=True)
    (src / "models" / "ipl" / "winprob_mlp.pt").write_bytes(b"weights")
    (src / "models" / "ipl" / "synth_gan" / "generator.pt").write_bytes(b"gan")
    use(src)
    return {"src": src, "use": use, "tmp": tmp_path}


def test_bundle_roundtrip(project):
    report = create_bundle("ipl", project["tmp"] / "b.zip")
    assert report.files == 6
    assert "processed/chase_states.parquet" in report.missing  # optional, reported

    host = project["tmp"] / "host"
    project["use"](host)
    manifest = install_bundle(report.path)
    assert manifest["dataset"] == "ipl"
    assert (host / "data/processed/ipl/matches.parquet").read_bytes() == b"matches.parquet" * 10
    assert (host / "models/ipl/synth_gan/generator.pt").read_bytes() == b"gan"
    assert json.loads((host / "data/processed/ipl/.bundle.json").read_text())["dataset"] == "ipl"


def test_bundle_needs_the_core_tables(project):
    (project["src"] / "data/processed/ipl/deliveries.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="data build"):
        create_bundle("ipl", project["tmp"] / "b.zip")


def tamper(src: Path, dst: Path, edit) -> Path:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            name, data = edit(item.filename, data)
            if name:
                zout.writestr(name, data)
    return dst


def test_tampered_bundle_is_rejected_before_anything_is_written(project):
    good = create_bundle("ipl", project["tmp"] / "b.zip").path
    bad = tamper(
        good,
        project["tmp"] / "bad.zip",
        lambda n, d: (n, b"evil" if n.endswith("winprob_mlp.pt") else d),
    )
    host = project["tmp"] / "host"
    project["use"](host)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        install_bundle(bad)
    assert not (host / "data").exists()  # nothing unpacked


def test_bundle_with_extra_or_escaping_files_is_rejected(project):
    good = create_bundle("ipl", project["tmp"] / "b.zip").path

    def add_escape(name, data):
        return name, data

    extra = tamper(good, project["tmp"] / "extra.zip", add_escape)
    with zipfile.ZipFile(extra, "a") as zf:
        zf.writestr("processed/../../../evil.txt", "x")
    project["use"](project["tmp"] / "host")
    with pytest.raises(ValueError, match="do not match"):
        install_bundle(extra)


def test_ensure_bundle_downloads_once(project, monkeypatch):
    zip_path = create_bundle("ipl", project["tmp"] / "b.zip").path
    host = project["tmp"] / "host"
    project["use"](host)
    calls = []

    def fake_download(url, dest, timeout=300):
        calls.append(url)
        shutil.copy(zip_path, dest)
        return dest

    monkeypatch.setattr("cricket_ai.deploy.download", fake_download)
    monkeypatch.delenv("CRICKET_AI_BUNDLE_URL", raising=False)
    assert ensure_bundle() == "no-url"
    monkeypatch.setenv("CRICKET_AI_BUNDLE_URL", "https://example.com/b.zip")
    assert ensure_bundle() == "installed"
    assert ensure_bundle() == "present"
    assert calls == ["https://example.com/b.zip"]
