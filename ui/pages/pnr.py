"""ui/pages/pnr.py"""
import streamlit as st
from core import session
from core.config import PNR_DEMO
from ui import components as C


def render():
    C.page_header("PNR", "Tracker",
                  "Complete journey timeline from booking to arrival")

    # Search bar
    C.card_open()
    pc1, pc2 = st.columns([5, 1])
    with pc1:
        pnr_in = st.text_input(
            "", placeholder="Enter 10-digit PNR  (try: 6230185420  or  5119274831)",
            value=session.get("pnr_queried", "6230185420"),
            label_visibility="collapsed", max_chars=10, key="pnr_field",
        )
    with pc2:
        if st.button("🔍  Track", use_container_width=True, key="pnr_track"):
            if pnr_in.strip():
                session.set_("pnr_queried", pnr_in.strip())
    C.card_close()

    pnr = PNR_DEMO.get(session.get("pnr_queried", "6230185420"))
    if not pnr:
        st.markdown("""
        <div style="background:rgba(255,71,87,0.08);border:1px solid rgba(255,71,87,0.2);
             border-radius:12px;padding:1rem;font-size:13px;color:#ff4757;">
            PNR not found. Try <code>6230185420</code> or <code>5119274831</code>.
        </div>""", unsafe_allow_html=True)
        return

    # Status header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.25rem;">
        <div style="font-size:13px;color:#8888aa;">PNR:
            <span style="font-family:'JetBrains Mono',monospace;color:#f0f0f8;">
                {session.get('pnr_queried')}
            </span>
        </div>
        <span class="badge b-g">DEMO</span>
        <span class="badge b-n">CNF</span>
    </div>""", unsafe_allow_html=True)

    pr1, pr2 = st.columns([3, 1])

    with pr1:
        C.card_open("neon")

        # Journey summary
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
             gap:1rem;margin-bottom:1.25rem;">
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
                     font-weight:700;color:#f0f0f8;">{pnr['train']}</div>
                <div style="font-size:12px;color:#8888aa;margin-top:3px;">
                    {pnr['route']} · {pnr['date']}
                </div>
            </div>
            <div style="display:flex;gap:8px;">
                <span style="background:rgba(59,130,246,.15);color:#60a5fa;font-size:11px;
                     font-weight:500;padding:3px 10px;border-radius:6px;">{pnr['cls']}</span>
                <span style="background:#22222f;color:#8888aa;font-size:11px;
                     font-weight:500;padding:3px 10px;border-radius:6px;">
                     Coach {pnr['coach']}</span>
                <span style="background:#22222f;color:#8888aa;font-size:11px;
                     font-weight:500;padding:3px 10px;border-radius:6px;">
                     Berth {pnr['berth']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # WL position sparkline
        st.markdown('<div style="font-size:12px;font-weight:500;color:#8888aa;'
                    'margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.6px;">'
                    'Waitlist Position Journey</div>', unsafe_allow_html=True)
        C.wl_sparkline(pnr["wl_history"])

        C.cyber_divider()
        st.markdown('<div style="font-size:13px;font-weight:600;color:#f0f0f8;'
                    'margin-bottom:1rem;">Journey Timeline</div>', unsafe_allow_html=True)
        C.timeline(pnr["timeline"])
        C.card_close()

    with pr2:
        C.card_open()
        init = "".join(w[0] for w in pnr["passenger"].split()[:2]).upper()
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:1rem;">
            <div style="width:54px;height:54px;border-radius:50%;
                 background:linear-gradient(135deg,#39ff85,#00d4ff);
                 display:flex;align-items:center;justify-content:center;
                 font-size:17px;font-weight:800;color:#000;margin:0 auto 8px;">{init}</div>
            <div style="font-size:14px;font-weight:600;color:#f0f0f8;">{pnr['passenger']}</div>
            <div style="font-size:12px;color:#8888aa;margin-top:3px;">{pnr['fare']}</div>
            <span class="badge b-n" style="margin-top:8px;">CNF</span>
        </div>""", unsafe_allow_html=True)
        C.neon_divider()
        st.markdown('<div style="font-size:12px;font-weight:600;color:#8888aa;'
                    'margin-bottom:.75rem;">Quick Actions</div>', unsafe_allow_html=True)
        st.button("📥 Download e-Ticket", use_container_width=True, key="dl_t")
        st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
        st.button("🔄 Find Alternatives",  use_container_width=True, key="fa_t")
        st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
        st.button("❌ Cancel & Refund",    use_container_width=True, key="cr_t")
        C.card_close()
