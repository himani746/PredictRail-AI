"""
core/session.py
Centralised Streamlit session-state management.
Call init() once at the top of app.py before any page renders.
"""

import random
import streamlit as st
from core.config import SEED


def init() -> None:
    """Seed all session-state keys. Idempotent — safe on every rerun."""
    _defaults()
    _seed_dashboard()


def _defaults() -> None:
    defaults = {
        "logged_in":   False,
        "username":    "",
        "page":        "Dashboard",
        "pred_result": None,
        "show_alts":   False,
        "alert_set":   False,
        "pnr_queried": "6230185420",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "users_db" not in st.session_state:
        st.session_state["users_db"] = {
            "demo": {
                "password": "demo123",
                "name":     "Sarvesh Kumar",
                "email":    "sarvesh@railpulse.ai",
                "phone":    "+91 98765 43210",
                "joined":   "Jan 2026",
                "bookings": 14,
                "saved":    3,
            },
        }


def _seed_dashboard() -> None:
    """Generate all 'random' dashboard numbers once — stable on refresh."""
    if "dash" in st.session_state:
        return
    rng = random.Random(SEED)
    st.session_state["dash"] = {
        "bookings_today":  rng.randint(4500, 4900),
        "wl_confirmed":    rng.randint(1050, 1150),
        "active_wl":       rng.randint(370, 400),
        "model_conf":      84.3,
        "bookings_trend":  [rng.randint(280, 640) for _ in range(14)],
        "wl_trend":        [rng.randint(40,  130) for _ in range(14)],
        "conf_rates":      [68, 72, 61, 89, 83, 57],
        "model_acc":       [74.2, 76.8, 78.1, 79.4, 81.0, 82.7, 84.3],
        "health":          [97, 99, 88, 94, 91],
        "heatmap_demand":  [rng.randint(40, 100) for _ in range(15)],
        "heatmap_wl":      [rng.randint(2,  35)  for _ in range(15)],
        "revenue_slices":  [rng.randint(8, 15)   for _ in range(7)],
    }


# ── Accessors ─────────────────────────────────────────────────────────────────
def get(key: str, default=None):
    return st.session_state.get(key, default)

def set_(key: str, value) -> None:
    st.session_state[key] = value

def navigate(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()
