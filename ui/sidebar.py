"""ui/sidebar.py — persistent left navigation."""
import streamlit as st
from core import session
from ml.predictor import get_predictor

NAV = [
    ("🏠", "Dashboard"),
    ("⚡", "Smart Predictor"),
    ("🔍", "PNR Tracker"),
    ("🗺", "Route Intelligence"),
    ("📊", "Admin Console"),
    ("👤", "Profile"),
]


def render():
    with st.sidebar:
        user     = session.get("users_db")[session.get("username")]
        initials = "".join(w[0] for w in user["name"].split()[:2]).upper()
        conf     = session.get("dash", {}).get("model_conf", 84.3)
        cur      = session.get("page", "Dashboard")

        # Brand
        st.markdown(
            '<div class="sb-brand"><span class="sb-pulse"></span> RailPulse AI</div>',
            unsafe_allow_html=True,
        )

        # User chip
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;
             background:rgba(57,255,133,0.06);border:1px solid rgba(57,255,133,0.14);
             border-radius:10px;padding:10px 12px;margin-bottom:1.5rem;">
            <div style="width:34px;height:34px;border-radius:50%;
                 background:linear-gradient(135deg,#39ff85,#00d4ff);
                 display:flex;align-items:center;justify-content:center;
                 font-size:12px;font-weight:800;color:#000;flex-shrink:0;">{initials}</div>
            <div>
                <div style="font-size:13px;font-weight:600;color:#f0f0f8;">{user['name']}</div>
                <div style="font-size:11px;color:#8888aa;">{user['email']}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Nav buttons
        for icon, key in NAV:
            if key == "Smart Predictor":
                st.markdown(
                    '<div class="nav-hint">⚡ start here</div>',
                    unsafe_allow_html=True,
                )
            if st.button(f"{icon}  {key}", key=f"nav_{key}", use_container_width=True):
                session.navigate(key)

        # Active-item highlight via nth-of-type
        idx = next((i for i, (_, k) in enumerate(NAV) if k == cur), 0) + 1
        st.markdown(f"""
        <style>
        section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type({idx}) button {{
            background: rgba(57,255,133,0.10) !important;
            border-color: rgba(57,255,133,0.30) !important;
            color: #39ff85 !important;
            font-weight: 600 !important;
        }}
        </style>""", unsafe_allow_html=True)

        st.markdown('<div class="nd"></div>', unsafe_allow_html=True)

        # Model confidence
        predictor  = get_predictor()
        model_name = predictor.model_name
        st.markdown(f"""
        <div style="background:rgba(57,255,133,0.05);border:1px solid rgba(57,255,133,0.15);
             border-radius:10px;padding:12px;margin-bottom:1rem;">
            <div style="font-size:10px;color:#55556a;letter-spacing:1px;
                 text-transform:uppercase;margin-bottom:6px;">AI Confidence</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:26px;
                 font-weight:800;color:#39ff85;animation:pulse 3s infinite;">{conf}%</div>
            <div style="height:4px;background:#22222f;border-radius:2px;margin-top:8px;overflow:hidden;">
                <div style="height:100%;width:{conf}%;background:linear-gradient(90deg,#39ff85,#00d4ff);
                     border-radius:2px;"></div>
            </div>
            <div style="font-size:10px;color:#55556a;margin-top:5px;">
                ↑ 1.6% vs last week · {model_name}
            </div>
        </div>""", unsafe_allow_html=True)

        # Quick PNR widget
        st.markdown(
            '<div style="font-size:11px;color:#55556a;text-transform:uppercase;'
            'letter-spacing:.8px;margin-bottom:6px;">Quick PNR Check</div>',
            unsafe_allow_html=True,
        )
        qpnr = st.text_input(
            "", placeholder="Enter PNR…", key="quick_pnr",
            label_visibility="collapsed", max_chars=10,
        )
        if st.button("Track →", key="quick_pnr_go", use_container_width=True):
            if qpnr.strip():
                session.set_("pnr_queried", qpnr.strip())
                session.navigate("PNR Tracker")

        st.markdown('<div class="nd"></div>', unsafe_allow_html=True)

        if st.button("Sign Out", key="signout", use_container_width=True):
            session.set_("logged_in", False)
            session.set_("username",  "")
            session.navigate("Dashboard")
