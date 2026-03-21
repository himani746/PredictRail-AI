"""ui/pages/dashboard.py"""
import streamlit as st
from core import session
from core.config import HEALTH_SVCS
from ui import components as C
from ui import charts


def render():
    d = session.get("dash")

    C.page_header("Executive", "Dashboard",
                  "System-wide overview · Stable metrics · AI performance · 18 Mar 2026")

    C.insight(
        '💡 <strong>Insight:</strong> Waitlist confirmations are up <strong>12% this week</strong> '
        'due to weekend travel demand on the Mumbai–Pune corridor. '
        'The allocation engine cleared <strong>847 WL positions</strong> in the last 24 hours.'
    )

    # KPIs
    C.kpi_grid(
        ("Total Bookings Today", f"{d['bookings_today']:,}", "↑ 12.4% vs yesterday", "up"),
        ("WL Confirmed Today",   f"{d['wl_confirmed']:,}",  "↑ 8.1% confirmation rate", "up"),
        ("Active Waitlists",     str(d["active_wl"]),       "↓ 23 in last hour", "down"),
        ("AI Confidence",        f"{d['model_conf']}%",     "↑ 1.6% this week", "up"),
    )

    # Charts row
    c1, c2 = st.columns([3, 2])
    with c1:
        C.card_open("cyber")
        C.section_header("Booking Trend (14 days)", "Total bookings vs waitlisted")
        st.plotly_chart(charts.booking_trend(), use_container_width=True)
        C.card_close()
    with c2:
        C.card_open("neon")
        C.section_header("Confirmation Rate by Class", "WL/1–15 historical clearance")
        st.plotly_chart(charts.class_rates(), use_container_width=True)
        C.card_close()

    # System health + model accuracy
    h1, h2 = st.columns(2)
    with h1:
        C.card_open()
        C.section_header("System Health")
        for i, svc in enumerate(HEALTH_SVCS):
            C.health_bar(svc, d["health"][i])
        C.card_close()
    with h2:
        C.card_open("neon")
        C.section_header("AI Accuracy Over Time",
                          "Test-set performance · 240K validation records")
        st.plotly_chart(charts.model_accuracy(), use_container_width=True)
        C.badge_row(
            C.badge("1.2M Past Bookings", "n"),
            C.badge("18 Months of Data",  "c"),
            C.badge("Updated Weekly",      "g"),
        )
        C.card_close()

    # Live allocation feed
    C.card_open()
    C.section_header("Live Allocation Feed")
    alerts = [
        ("🟢","11007 Deccan Express","WL/12 auto-upgraded to RAC · Dynamic Seat Allocation triggered","2 min ago"),
        ("🟡","12301 Rajdhani Exp",  "High demand — 42 bookings in 15 mins · Model flagging pressure","8 min ago"),
        ("🔵","12027 Shatabdi",      "Chart prepared · All RAC passengers confirmed to berths","14 min ago"),
        ("🔴","11301 Indrayani Exp", "Prediction confidence below 60% for WL/18+ positions","31 min ago"),
    ]
    for icon, train, msg, t in alerts:
        C.alert_row(icon, train, msg, t)
    C.card_close()
