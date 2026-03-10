"""Streamlit Cloud entrypoint.

This lightweight wrapper runs the dashboard app from `dashboard/app.py`
so Streamlit Cloud can use a root-level `streamlit_app.py`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


APP_FILE = Path(__file__).resolve().parent / "dashboard" / "app.py"
runpy.run_path(str(APP_FILE), run_name="__main__")
