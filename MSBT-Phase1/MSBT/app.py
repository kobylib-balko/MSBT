"""Streamlit entrypoint (local + Community Cloud).

Cloud main file path: MSBT/app.py (Phase 2)
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from msbt.streamlit_app.app import main

    main()
except Exception:
    st.set_page_config(page_title="MSBT error", layout="wide")
    st.error("App failed to start. Details:")
    st.code(traceback.format_exc())
