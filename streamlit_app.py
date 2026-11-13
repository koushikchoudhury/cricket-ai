"""Entry point for Streamlit Community Cloud (and `streamlit run streamlit_app.py`).

Locally, `cricket-ai app` does the same thing.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # works even if the package is not installed
runpy.run_path(str(ROOT / "src" / "cricket_ai" / "app" / "main.py"), run_name="__main__")
