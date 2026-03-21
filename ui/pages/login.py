"""ui/pages/login.py"""
import streamlit as st
from core import session
from ui import components as C


def render():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, cm, _ = st.columns([1, 1.1, 1])
    with cm:
        st.markdown("""
        <div class="auth-wrap">
            <div class="auth-logo"><span style="color:#39ff85;">⚡</span> RailPulse AI</div>
            <div class="auth-tag">
                India's first AI-powered waitlist prediction engine.<br>
                Know your confirmation probability <em>before</em> you book.
            </div>
            <div class="nd"></div>
            <div class="auth-title">Sign in</div>
        </div>""", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div style="max-width:420px;margin:0 auto;">', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            c1, c2   = st.columns(2)
            with c1:
                if st.button("Sign In", use_container_width=True):
                    db = session.get("users_db")
                    if username in db and db[username]["password"] == password:
                        session.set_("logged_in", True)
                        session.set_("username",  username)
                        session.navigate("Dashboard")
                    else:
                        st.error("Invalid credentials")
            with c2:
                if st.button("Register →", use_container_width=True):
                    session.navigate("Register")
            st.markdown("""
            <div style="margin-top:1rem;padding:10px 14px;
                 background:rgba(57,255,133,0.06);border:1px solid rgba(57,255,133,0.15);
                 border-radius:8px;font-size:12px;color:#8888aa;">
                Demo: <code style="color:#39ff85;">demo</code> /
                      <code style="color:#39ff85;">demo123</code>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
