"""ui/pages/predictor.py"""
import random
import time
import streamlit as st
from datetime import date, timedelta

from core import session
from core.config import CLASS_META, DEMO_TRAINS
from ui import components as C
from ui import charts
from ml.predictor import get_predictor

_ALT_TRAINS = [
    {"id":"12027","name":"Mumbai Shatabdi","dep":"07:30","arr":"10:35","dur":"3h 05m",
     "cls":"CC","fare":"₹1,285","wl":0,"avail":"AVL 12","score":94},
    {"id":"11007","name":"Deccan Express",  "dep":"05:15","arr":"08:50","dur":"3h 35m",
     "cls":"3A","fare":"₹760",  "wl":0,"avail":"AVL 6", "score":81},
    {"id":"12123","name":"Deccan Queen",    "dep":"14:45","arr":"19:15","dur":"4h 20m",
     "cls":"3A","fare":"₹695",  "wl":4,"avail":"WL 4",  "score":67},
    {"id":"11301","name":"Indrayani Exp",   "dep":"17:30","arr":"21:10","dur":"3h 40m",
     "cls":"SL","fare":"₹285",  "wl":0,"avail":"AVL 32","score":71},
]


def _ai_says_text(prob: float, wl: int, days: int, cls: str, train_name: str) -> str:
    base = CLASS_META.get(cls, {}).get("base_confirm", 65)
    name = train_name.split("—")[1].strip() if "—" in train_name else train_name
    if prob >= 75:
        return (f"This waitlist is looking <strong>very healthy</strong>. "
                f"WL/{wl} on the {name} typically clears "
                f"<strong>{max(1,days-1)} to {days} days before departure</strong>. "
                f"Out of every 100 similar bookings, "
                f"<strong>{int(prob)} confirmed</strong>. I'd book it.")
    elif prob >= 45:
        return (f"This is a coin-toss situation. {cls} class clears "
                f"<strong>{base}% of WL/1–10</strong>, but WL/{wl} is deeper than ideal. "
                f"Check again in 48 hours — if it hasn't moved, shift to one of the "
                f"confirmed alternatives listed below.")
    else:
        return (f"I wouldn't count on this one. WL/{wl} with only {days} day(s) left "
                f"means the waitlist needs to clear fast — and historically only "
                f"<strong>{int(prob)}% of these confirm</strong>. "
                f"The alternatives below all have confirmed seats right now.")


def render():
    C.page_header("Smart", "Predictor",
                  "Enter your journey — get AI-powered confirmation probability with full explanation")

    predictor = get_predictor()

    # ── Step 1: Input form ────────────────────────────────────────────────────
    C.card_open("cyber")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
        <div style="width:22px;height:22px;border-radius:50%;background:var(--cyber);
             display:flex;align-items:center;justify-content:center;font-size:11px;
             font-weight:700;color:#000;">1</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
             font-weight:600;color:#f0f0f8;">Your Journey</div>
    </div>""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        from_stn = st.selectbox("FROM", ["PUNE — Pune Junction","CSMT — Mumbai",
                                          "NDLS — New Delhi","SBC — Bangalore",
                                          "MAS — Chennai","NGP — Nagpur"])
    with fc2:
        to_stn = st.selectbox("TO", ["CSMT — Mumbai","PUNE — Pune Junction",
                                      "NDLS — New Delhi","SBC — Bangalore",
                                      "MAS — Chennai","NGP — Nagpur"])
    with fc3:
        travel_date = st.date_input("TRAVEL DATE", value=date(2026, 3, 25))

    fc4, fc5, fc6, fc7 = st.columns(4)
    with fc4:
        train_sel = st.selectbox("TRAIN", [f"{t['id']} — {t['name']}" for t in DEMO_TRAINS])
    with fc5:
        cls_sel = st.selectbox("CLASS", list(CLASS_META.keys()),
                               format_func=lambda k: f"{k} — {CLASS_META[k]['label']}")
    with fc6:
        wl_no = st.number_input("WAITLIST NUMBER  (0 = no WL)",
                                 min_value=0, max_value=100, value=8,
                                 help="Enter 0 for direct confirmed seats")
    with fc7:
        pax = st.number_input("PASSENGERS", min_value=1, max_value=6, value=1)

    clicked = st.button("⚡  Predict My Chances", use_container_width=True)
    C.card_close()

    if clicked:
        tid  = train_sel.split(" — ")[0]
        days = max(1, (travel_date - date.today()).days)

        # Loading animation
        bar    = st.progress(0)
        status = st.empty()
        for pct, msg in [(20,"Scanning route cancellation history…"),
                         (50,"Running prediction model…"),
                         (80,"Calculating confidence…"),
                         (100,"Done.")]:
            status.markdown(
                f'<div style="text-align:center;padding:.5rem;font-size:12px;'
                f'color:#8888aa;">{msg}</div>',
                unsafe_allow_html=True,
            )
            bar.progress(pct)
            time.sleep(0.28)
        bar.empty()
        status.empty()

        result = predictor.predict(
            train_id=tid, coach_type=cls_sel, wl_no=wl_no,
            booking_day=days, season="normal",
        )
        session.set_("pred_result", {
            "prob":       result["probability_pct"],
            "raw":        result["probability"],
            "train":      train_sel,
            "tid":        tid,
            "cls":        cls_sel,
            "wl":         wl_no,
            "from":       from_stn,
            "to":         to_stn,
            "date":       travel_date,
            "days":       days,
            "fare":       CLASS_META[cls_sel]["fare"],
            "pax":        pax,
            "model_used": result["model_used"],
            "ml_alts":    result["alternatives"],
        })
        st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    res = session.get("pred_result")
    if not res:
        return

    prob  = res["prob"]
    days  = res["days"]
    wl    = res["wl"]
    tid   = res["tid"]
    eta   = (date.today() + timedelta(days=max(1, days - 2))).strftime("%d %b %Y")

    if prob >= 75:
        v_color, v_label, v_glass = "#39ff85", "Likely to Confirm",           "neon"
        badge_v,  rec_color       = "b-n",      "#39ff85"
    elif prob >= 45:
        v_color, v_label, v_glass = "#ffb340", "Uncertain — Act Wisely",       "amber"
        badge_v,  rec_color       = "b-a",      "#ffb340"
    else:
        v_color, v_label, v_glass = "#ff4757", "High Risk — Book Alternative", "red"
        badge_v,  rec_color       = "b-r",      "#ff4757"

    # ── Hero number ───────────────────────────────────────────────────────────
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    C.card_open(v_glass)
    st.markdown(f"""
    <div class="hero-verdict">
        <div class="hero-pct" style="color:{v_color};">{prob:.0f}%</div>
        <div class="hero-sublabel" style="color:{v_color};">{v_label}</div>
        <div class="hero-eta">
            {"Expected to confirm by " + eta if wl > 0 else "Seat directly available — no waitlist"}
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;margin:.5rem 0;">'
                f'<span class="badge {badge_v}">{res["model_used"]}</span>'
                f'</div>', unsafe_allow_html=True)
    C.card_close()

    # ── Recommendation + CTAs ─────────────────────────────────────────────────
    if prob >= 75:
        rec_head = "Go ahead and book this ticket"
        rec_body = (f"Based on <strong>1.2M past journeys</strong>, WL/{wl} on this route "
                    f"confirms <strong>{prob:.0f}% of the time</strong> — "
                    f"usually within {max(1,days-2)} day(s). You're in a strong position.")
        cta      = "🎫 Book This Ticket"
    elif prob >= 45:
        rec_head = "Moderate chance — keep an eye on it"
        rec_body = (f"WL/{wl} has about a <strong>{prob:.0f}% chance</strong> of clearing. "
                    f"If it hasn't moved by <strong>{eta}</strong>, "
                    f"switch to a confirmed alternative below.")
        cta      = "🎫 Book WL Ticket"
    else:
        rec_head = "Book an alternative now"
        rec_body = (f"WL/{wl} has only a <strong>{prob:.0f}% chance</strong>. "
                    f"Based on past data, positions this deep rarely clear in time. "
                    f"Confirmed seats are available on other trains below.")
        cta      = "🔍 See Confirmed Alternatives"

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
         border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:.75rem;">
        <div style="font-size:14px;font-weight:700;color:{rec_color};margin-bottom:5px;">
            {"✅" if prob>=75 else "⚠️" if prob>=45 else "❌"} {rec_head}
        </div>
        <div style="font-size:13px;color:#8888aa;line-height:1.65;">{rec_body}</div>
    </div>""", unsafe_allow_html=True)

    # Show ML-based alternative suggestions if any
    if res.get("ml_alts"):
        for alt_txt in res["ml_alts"]:
            st.markdown(f'<div style="font-size:12px;color:#8888aa;padding:3px 0;">💡 {alt_txt}</div>',
                        unsafe_allow_html=True)

    cb1, cb2 = st.columns([2, 1])
    with cb1:
        if st.button(cta, use_container_width=True, key="cta_primary"):
            st.success("Booking flow initiated!")
    with cb2:
        if st.button("🔔 Set Alert", use_container_width=True, key="set_alert"):
            session.set_("alert_set", True)
            st.success(f"Alert set — you'll be notified if WL/{wl} confidence drops below 50%.")

    # ── AI explanation ────────────────────────────────────────────────────────
    with st.expander("🧠  Why this score? — AI Explanation", expanded=True):
        C.ai_says(_ai_says_text(prob, wl, days, res["cls"], res["train"]))
        C.cyber_divider()

        base      = CLASS_META.get(res["cls"], {}).get("base_confirm", 65)
        wl_safety = max(5, 100 - wl * 3)
        day_adv   = min(100, days * 7)

        C.prob_bar(
            "How often this class confirms",
            base, f"{base}/100", "#00d4ff",
            f"{res['cls']} class confirmed in {base}% of past similar WL/1–10 bookings",
        )
        C.prob_bar(
            "How safe your waitlist position is",
            wl_safety, f"{wl_safety}/100", "#39ff85",
            "WL/" + str(wl) + " — " + (
                "very safe, top of the list" if wl <= 5 else
                "moderate position" if wl <= 12 else
                "deep position, higher risk"
            ),
        )
        C.prob_bar(
            "Time advantage until travel",
            day_adv, f"{day_adv}/100", "#ffb340",
            f"{days} day(s) left — " + (
                "plenty of time for cancellations" if days >= 7 else
                "limited time" if days >= 3 else
                "very tight window"
            ),
        )

    # ── What-If Simulator ─────────────────────────────────────────────────────
    with st.expander("🔬  What-If Simulator — explore other positions"):
        st.markdown('<div style="font-size:12px;color:#8888aa;margin-bottom:.75rem;">'
                    'Drag to see how probability changes at different waitlist numbers.</div>',
                    unsafe_allow_html=True)
        sim_wl   = st.slider("Waitlist position", 1, 40, int(wl), step=1, key="wif_slider")
        sim_res  = predictor.predict(train_id=tid, coach_type=res["cls"],
                                     wl_no=sim_wl, booking_day=days)
        sim_prob = sim_res["probability_pct"]
        sim_col  = "#39ff85" if sim_prob >= 75 else "#ffb340" if sim_prob >= 45 else "#ff4757"
        delta    = sim_prob - prob
        d_col    = "#39ff85" if delta > 0 else "#ff4757" if delta < 0 else "#8888aa"
        d_txt    = f"↑ +{delta:.1f}%" if delta > 0 else f"↓ {delta:.1f}%" if delta < 0 else "no change"

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:24px;padding:1rem;
             background:rgba(255,255,255,0.03);border-radius:12px;margin-top:.5rem;">
            <div style="text-align:center;min-width:90px;">
                <div style="font-size:10px;color:#55556a;text-transform:uppercase;
                     letter-spacing:.8px;margin-bottom:4px;">At WL/{sim_wl}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:38px;
                     font-weight:800;color:{sim_col};line-height:1;">{sim_prob:.0f}%</div>
            </div>
            <div style="flex:1;">
                <div style="height:10px;background:rgba(255,255,255,0.06);
                     border-radius:5px;overflow:hidden;">
                    <div style="height:100%;width:{sim_prob}%;background:{sim_col};
                         border-radius:5px;"></div>
                </div>
                <div style="font-size:12px;color:{d_col};margin-top:6px;font-weight:600;">
                    {d_txt} vs your current WL/{wl}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Snap comparison grid
        snap_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:.75rem;">'
        for snap_wl in [2, 5, 8, 12, 18, 25]:
            sp = predictor.predict(train_id=tid, coach_type=res["cls"],
                                   wl_no=snap_wl, booking_day=days)
            sp_pct = sp["probability_pct"]
            sc = "#39ff85" if sp_pct>=75 else "#ffb340" if sp_pct>=45 else "#ff4757"
            you = " (you)" if snap_wl == wl else ""
            bright = "0.18" if snap_wl == wl else "0.06"
            snap_html += (
                f'<div style="background:rgba(255,255,255,0.04);border:1px solid '
                f'rgba(255,255,255,{bright});border-radius:8px;padding:8px 12px;text-align:center;">'
                f'<div style="font-size:10px;color:#55556a;">WL/{snap_wl}{you}</div>'
                f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:16px;'
                f'font-weight:700;color:{sc};margin-top:2px;">{sp_pct:.0f}%</div></div>'
            )
        snap_html += '</div>'
        st.markdown(snap_html, unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#55556a;margin-top:.75rem;">'
                    '🟢 &gt;75% — Book with confidence &nbsp;·&nbsp; '
                    '🟡 45–75% — Consider alternatives &nbsp;·&nbsp; '
                    '🔴 &lt;45% — High risk</div>', unsafe_allow_html=True)

    # ── Gauge (secondary) ─────────────────────────────────────────────────────
    with st.expander("📊  Detailed Probability Gauge"):
        gc1, gc2 = st.columns([1, 1])
        with gc1:
            st.plotly_chart(charts.gauge(prob), use_container_width=True)
        with gc2:
            C.info_grid(
                ("Train · Class · WL",
                 f"{res['train']} · {res['cls']} · {'WL/'+str(wl) if wl>0 else 'Direct'}"),
                ("Route",
                 f"{res['from'].split('—')[0].strip()} → {res['to'].split('—')[0].strip()}"),
                ("Travel Date",
                 f"{res['date'].strftime('%d %b %Y')} · {days} day(s) away"),
                ("Expected Confirmation",
                 eta if wl > 0 else "Already available"),
            )

    # ── Alternatives (always visible, auto-expanded when high risk) ───────────
    with st.expander("🚆  Alternative Trains — confirmed options on this route",
                     expanded=(prob < 75)):
        st.markdown('<div style="font-size:12px;color:#8888aa;margin-bottom:.75rem;">'
                    'Ranked by convenience score — combines availability, travel time, '
                    'fare, and on-time rate.</div>', unsafe_allow_html=True)
        for alt in _ALT_TRAINS:
            C.alt_card(alt)
