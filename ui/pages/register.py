"""ui/pages/register.py"""
import time
import streamlit as st
from datetime import datetime
from core import session


def render():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, cm, _ = st.columns([1, 1.1, 1])
    with cm:
        st.markdown("""
        <div style="max-width:420px;margin:0 auto;background:#13131a;
             border:1px solid rgba(255,255,255,0.07);border-radius:20px;
             padding:2.5rem;box-shadow:0 24px 64px rgba(0,0,0,.6);">
            <div class="auth-logo"><span style="color:#39ff85;">⚡</span> RailPulse AI</div>
            <div class="auth-tag">Create your account</div>
            <div class="nd"></div>
        </div>""", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div style="max-width:420px;margin:0 auto;">', unsafe_allow_html=True)
            r_name  = st.text_input("Full Name",        placeholder="Sarvesh Kumar")
            r_email = st.text_input("Email",            placeholder="you@example.com")
            r_phone = st.text_input("Phone",            placeholder="+91 XXXXX XXXXX")
            r_user  = st.text_input("Username",         placeholder="Choose a username")
            r_pass  = st.text_input("Password",         type="password", placeholder="Min. 6 characters")
            r_pass2 = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
            c1, c2  = st.columns(2)
            with c1:
                if st.button("Create Account", use_container_width=True):
                    if not all([r_name, r_email, r_user, r_pass]):
                        st.error("All fields required.")
                    elif r_pass != r_pass2:
                        st.error("Passwords don't match.")
                    elif len(r_pass) < 6:
                        st.error("Password too short.")
                    elif r_user in session.get("users_db"):
                        st.error("Username taken.")
                    else:
                        db = session.get("users_db")
                        db[r_user] = {
                            "password": r_pass, "name": r_name,
                            "email": r_email, "phone": r_phone or "Not provided",
                            "joined": datetime.now().strftime("%b %Y"),
                            "bookings": 0, "saved": 0,
                        }
                        session.set_("users_db", db)
                        st.success("Account created! Signing you in…")
                        time.sleep(0.8)
                        session.navigate("Login")
            with c2:
                if st.button("← Back", use_container_width=True):
                    session.navigate("Login")
            st.markdown('</div>', unsafe_allow_html=True)
