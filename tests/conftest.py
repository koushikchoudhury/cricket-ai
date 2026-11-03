import json
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(Path(__file__).parent))

from matchgen import generate_match  # noqa: E402


@pytest.fixture
def tiny_match_path() -> Path:
    return FIXTURES / "tiny_match.json"


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A folder with the hand-scored match plus 40 generated T20s and one broken file."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "tiny_match.json").write_text((FIXTURES / "tiny_match.json").read_text())
    for seed in range(40):
        (raw / f"gen_{seed:03d}.json").write_text(json.dumps(generate_match(seed)))
    (raw / "broken.json").write_text("{not json")
    return raw
