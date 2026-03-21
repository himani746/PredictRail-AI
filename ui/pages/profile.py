"""ui/pages/profile.py"""
import streamlit as st
from core import session
from ui import components as C


def render():
    user     = session.get("users_db")[session.get("username")]
    initials = "".join(w[0] for w in user["name"].split()[:2]).upper()

    C.page_header("My", "Profile",
                  "Account details · Booking history · Preferences")

    pp1, pp2 = st.columns([1, 2])

    with pp1:
        C.card_open("neon")
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="pf-avatar">{initials}</div>
            <div class="pf-name">{user['name']}</div>
            <div class="pf-sub">{user['email']}</div>
            <div style="margin:1rem 0;"><span class="badge b-n">Verified Member</span></div>
            <div class="nd"></div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div class="pf-stat">
                    <div class="pf-sv">{user['bookings']}</div>
                    <div class="pf-sl">Bookings</div>
                </div>
                <div class="pf-stat">
                    <div class="pf-sv">{user['saved']}</div>
                    <div class="pf-sl">Saved Routes</div>
                </div>
            </div>
            <div style="margin-top:1rem;padding:10px;background:rgba(57,255,133,0.05);
                 border-radius:8px;border:1px solid rgba(57,255,133,0.12);">
                <div style="font-size:10px;color:#55556a;text-transform:uppercase;
                     letter-spacing:.6px;">Member Since</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
                     font-weight:700;color:#f0f0f8;margin-top:4px;">{user['joined']}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        C.card_close()

    with pp2:
        # Account details
        C.card_open()
        C.section_header("Account Details")
        for label, val in [
            ("Full Name",  user["name"]),
            ("Email",      user["email"]),
            ("Phone",      user.get("phone","Not provided")),
            ("Joined",     user["joined"]),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:10px 0;
                 border-bottom:1px solid rgba(255,255,255,0.04);">
                <div style="font-size:11px;color:#55556a;text-transform:uppercase;
                     letter-spacing:.5px;">{label}</div>
                <div style="font-size:13px;font-weight:500;color:#f0f0f8;">{val}</div>
            </div>""", unsafe_allow_html=True)

        C.neon_divider()
        C.section_header("Edit Profile")

        new_name  = st.text_input("Full Name",  value=user["name"],         key="ep_name")
        new_phone = st.text_input("Phone",      value=user.get("phone",""), key="ep_phone")
        new_email = st.text_input("Email",      value=user["email"],        key="ep_email")

        if st.button("💾  Save Changes", use_container_width=True, key="save_p"):
            db = session.get("users_db")
            db[session.get("username")].update(
                {"name": new_name, "phone": new_phone, "email": new_email}
            )
            session.set_("users_db", db)
            st.success("✅ Profile updated.")
        C.card_close()

        # Recent bookings
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        C.card_open()
        C.section_header("Recent Bookings")
        recent = [
            ("6230185420","Deccan Express #11007","Pune → Mumbai","18 Mar 2026","₹760","CNF","b-n"),
            ("5119274831","Mumbai Shatabdi #12027","Mumbai → Pune","12 Mar 2026","₹635","CNF","b-n"),
            ("4208163720","Deccan Queen #12123",  "Pune → Mumbai","05 Mar 2026","₹255","CAN","b-r"),
        ]
        for pnr_no, train, route, dt, fare, status, bc in recent:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                 padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                <div>
                    <div style="font-size:13px;font-weight:600;color:#f0f0f8;">{train}</div>
                    <div style="font-size:11px;color:#8888aa;margin-top:2px;">
                        {route} · {dt} ·
                        <span style="font-family:'JetBrains Mono',monospace;color:#55556a;">
                            {pnr_no}</span>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <span class="badge {bc}">{status}</span>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
                         font-weight:700;color:#f0f0f8;">{fare}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Smart alert banner
        if session.get("alert_set"):
            st.markdown("""
            <div style="margin-top:1rem;padding:10px 14px;background:rgba(57,255,133,0.06);
                 border:1px solid rgba(57,255,133,0.2);border-radius:8px;font-size:12px;color:#8888aa;">
                🔔 <strong style="color:#39ff85;">Smart Alert Active</strong> —
                You'll be notified if your WL confidence drops below 50%
            </div>""", unsafe_allow_html=True)
        C.card_close()
