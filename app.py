"""DataSense AI - aplikasi analisis data dan pendukung keputusan (Streamlit)."""

from __future__ import annotations

import streamlit as st

from common.config import DATASENSE_NAME, ensure_runtime_dirs
from ui.datasense_app import render_datasense
from ui.shared import inject_css

st.set_page_config(page_title=DATASENSE_NAME, layout="wide", initial_sidebar_state="expanded")
ensure_runtime_dirs()
inject_css()

with st.sidebar:
    st.markdown(f"## {DATASENSE_NAME}")
    st.caption("Make Sense of Data with Analytics and AI")
    st.markdown("**Catatan**\n\n- Data diproses secara lokal.\n- Hasil memiliki keterbatasan yang ditampilkan di aplikasi.")

render_datasense()
