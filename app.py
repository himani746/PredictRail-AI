"""
app.py — RailPulse AI entry point
Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="RailPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bootstrap
from core import session
from ui import styles
from ui import sidebar

session.init()
styles.inject()

# ── Router ─────────────────────────────────────────────────────────────────────
if not session.get("logged_in"):
    from ui.pages import login, register
    p = session.get("page")
    if p == "Register":
        register.render()
    else:
        login.render()
else:
    sidebar.render()
    p = session.get("page")

    if   p == "Dashboard":         from ui.pages import dashboard;  dashboard.render()
    elif p == "Smart Predictor":   from ui.pages import predictor;  predictor.render()
    elif p == "PNR Tracker":       from ui.pages import pnr;        pnr.render()
    elif p == "Route Intelligence": from ui.pages import heatmap;   heatmap.render()
    elif p == "Admin Console":     from ui.pages import admin;      admin.render()
    elif p == "Profile":           from ui.pages import profile;    profile.render()
    else:                          from ui.pages import dashboard;  dashboard.render()
