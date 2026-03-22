"""
RailPulse AI — Single-file Streamlit app (all modules merged, all bugs fixed)
Run: streamlit run app.py

BUGS FIXED:
  1. dashboard.py called charts.booking_trend/class_rates/model_accuracy with NO args → now passes session data
  2. heatmap.py called charts.heatmap_treemap() → renamed correctly to charts.heatmap()
  3. heatmap.py called charts.demand_scatter() → renamed correctly to charts.scatter_demand()
  4. admin.py called charts.revenue_donut() with no args → now passes session data
  5. _pipeline.py saved PNGs as "feature_importance.png" but admin.py looked for "feature_importance_real.png" → unified to _real suffix
  6. ml/predict.py imported FEATURE_COLS from core.config where it doesn't exist → removed dead import
  7. Predictor._ml_predict used wrong dict key "train_name" → added safe .get fallback
  8. Model payload stacking path: imputer/scaler correctly applied; base Pipeline models skip preprocessing (handled internally)
"""

# ─────────────────────────────────────────────────────────────────────────────
# std-lib / third-party
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, json, random, time, warnings
from pathlib import Path
from datetime import date, timedelta, datetime
from typing import Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="RailPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG (was core/config.py)
# ═════════════════════════════════════════════════════════════════════════════
ROOT    = Path(__file__).parent
DATA    = ROOT / "data"
OUTPUTS = ROOT / "outputs"

STATIONS_PATH  = DATA  / "stations.json"
TRAINS_PATH    = DATA  / "trains.json"
SCHEDULES_PATH = DATA  / "schedules.json"
MODEL_PATH     = OUTPUTS / "best_model_real.joblib"

OUTPUTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

SEED       = 42
N_BOOKINGS = 30_000
TEST_SIZE  = 0.20
CV_FOLDS   = 3

COACH_TYPES  = ["1A", "2A", "3A", "SL", "CC", "2S"]
SEASONS      = ["peak", "normal", "off-peak"]
COACH_PROBS  = [0.05, 0.10, 0.25, 0.38, 0.12, 0.10]

COACH_BONUS  = {"1A": 0.30, "2A": 0.22, "CC": 0.12, "3A": 0.05, "SL": 0.00, "2S": -0.08}
SEASON_PEN   = {"off-peak": 0.20, "normal": 0.00, "peak": -0.25}
TYPE_BONUS   = {
    "Raj": 0.20, "Drnt": 0.18, "JShtb": 0.15, "Shtb": 0.12, "SKr": 0.12,
    "SF":  0.08, "Mail": 0.06, "Exp":   0.03,  "Pass": 0.00,
    "MEMU": -0.03, "DEMU": -0.03, "GR": -0.05, "Toy": -0.08,
    "Hyd": 0.05, "Del": 0.10, "Klkt": 0.12, "Unknown": 0.0, "": 0.0,
}

CLASS_META = {
    "SL": {"label": "Sleeper",     "base_confirm": 72, "fare": 285},
    "3A": {"label": "3 Tier AC",   "base_confirm": 65, "fare": 760},
    "2A": {"label": "2 Tier AC",   "base_confirm": 58, "fare": 1145},
    "1A": {"label": "First AC",    "base_confirm": 85, "fare": 2120},
    "CC": {"label": "Chair Car",   "base_confirm": 78, "fare": 635},
    "2S": {"label": "2nd Sitting", "base_confirm": 55, "fare": 140},
    "EC": {"label": "Exec Chair",  "base_confirm": 60, "fare": 1285},
}

DEMO_TRAINS = [
    {"id": "11007", "name": "Deccan Express",   "type": "EXPRESS",
     "dep": "05:15", "arr": "08:50", "dur": "3h 35m", "dep_stn": "PUNE", "arr_stn": "CSMT",
     "on_time": 94, "cancel_rate": 18, "occupancy": 87, "price_tier": 2, "comfort": 3},
    {"id": "12027", "name": "Mumbai Shatabdi",  "type": "SHATABDI",
     "dep": "07:30", "arr": "10:35", "dur": "3h 05m", "dep_stn": "PUNE", "arr_stn": "CSMT",
     "on_time": 97, "cancel_rate": 8,  "occupancy": 92, "price_tier": 4, "comfort": 5},
    {"id": "12123", "name": "Deccan Queen",     "type": "SUPERFAST",
     "dep": "14:45", "arr": "19:15", "dur": "4h 20m", "dep_stn": "PUNE", "arr_stn": "CSMT",
     "on_time": 81, "cancel_rate": 14, "occupancy": 78, "price_tier": 2, "comfort": 3},
    {"id": "11301", "name": "Indrayani Express","type": "EXPRESS",
     "dep": "17:30", "arr": "21:10", "dur": "3h 40m", "dep_stn": "PUNE", "arr_stn": "CSMT",
     "on_time": 76, "cancel_rate": 22, "occupancy": 65, "price_tier": 1, "comfort": 2},
    {"id": "12301", "name": "Rajdhani Express", "type": "RAJDHANI",
     "dep": "16:35", "arr": "08:15", "dur": "15h 40m","dep_stn": "PUNE", "arr_stn": "NDLS",
     "on_time": 91, "cancel_rate": 11, "occupancy": 95, "price_tier": 5, "comfort": 5},
]

PNR_DEMO = {
    "6230185420": {
        "train": "Deccan Express #11007", "route": "Pune → Mumbai",
        "date": "18 Mar 2026", "cls": "3A", "coach": "B4", "berth": "32 LB",
        "passenger": "S. Kumar", "fare": "₹760",
        "timeline": [
            {"label": "Booking Initiated",  "time": "14 Mar 2026, 10:38", "state": "done",   "note": "Payment received · PNR generated"},
            {"label": "Waitlisted (WL/14)", "time": "14 Mar 2026, 10:39", "state": "done",   "note": "Initial position: WL/14"},
            {"label": "Moved to WL/8",      "time": "15 Mar 2026, 06:12", "state": "done",   "note": "6 cancellations processed by allocation engine"},
            {"label": "Upgraded to RAC/4",  "time": "16 Mar 2026, 20:44", "state": "done",   "note": "Dynamic Seat Allocation triggered"},
            {"label": "Confirmed — CNF",    "time": "17 Mar 2026, 22:14", "state": "done",   "note": "Berth B4/32 LB assigned"},
            {"label": "Chart Prepared",     "time": "17 Mar 2026, 23:59", "state": "active", "note": "Final chart locked · Boarding confirmed"},
            {"label": "Train Departed",     "time": "18 Mar 2026, 05:15", "state": "pending","note": "Platform 2 · On time"},
            {"label": "Journey Complete",   "time": "18 Mar 2026, 08:50", "state": "pending","note": "CSMT Mumbai"},
        ],
        "wl_history": [14, 14, 12, 10, 8, 8, 6, 4, "RAC", "CNF"],
    },
    "5119274831": {
        "train": "Mumbai Shatabdi #12027", "route": "Mumbai → Pune",
        "date": "12 Mar 2026", "cls": "CC", "coach": "C2", "berth": "45",
        "passenger": "R. Sharma", "fare": "₹635",
        "timeline": [
            {"label": "Booking Confirmed", "time": "08 Mar 2026, 14:20", "state": "done", "note": "Direct confirmation — no waitlist"},
            {"label": "Chart Prepared",    "time": "11 Mar 2026, 22:00", "state": "done", "note": "Seat C2/45 confirmed"},
            {"label": "Train Departed",    "time": "12 Mar 2026, 06:25", "state": "done", "note": "Departed on time"},
            {"label": "Journey Complete",  "time": "12 Mar 2026, 09:30", "state": "done", "note": "Arrived — journey complete ✓"},
        ],
        "wl_history": ["CNF", "CNF", "CNF", "CNF"],
    },
}

ROUTES = [
    ("Mumbai","Delhi"),     ("Mumbai","Pune"),      ("Delhi","Kolkata"),
    ("Mumbai","Ahmedabad"), ("Bangalore","Chennai"), ("Delhi","Jaipur"),
    ("Hyderabad","Chennai"),("Kolkata","Patna"),     ("Delhi","Lucknow"),
    ("Mumbai","Goa"),       ("Chennai","Bangalore"), ("Pune","Nashik"),
    ("Agra","Delhi"),       ("Nagpur","Mumbai"),     ("Bhopal","Delhi"),
]

HEALTH_SVCS = [
    "Prediction Engine","Booking API","Seat Alloc. Engine",
    "PNR Sync Service","Database Cluster",
]

MODEL_PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":       "#55A868",
    "Gradient Boosting":   "#C44E52",
    "Stacking Ensemble":   "#FF8C00",
}

APP_TITLE     = "RailPulse AI"
APP_ICON      = "⚡"
MODEL_VERSION = "v2.0"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CSS (was ui/styles.py)
# ═════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');
:root{
  --bg:#080810;--bg2:#0e0e1a;--bg3:#141420;--bg4:#1c1c2e;--bg5:#242438;
  --bd:rgba(255,255,255,0.06);--bd2:rgba(255,255,255,0.12);--bd3:rgba(255,255,255,0.18);
  --neon:#00ff88;--ndim:rgba(0,255,136,0.08);--nglow:rgba(0,255,136,0.25);--nbright:rgba(0,255,136,0.5);
  --cyber:#38bdf8;--cdim:rgba(56,189,248,0.08);--cglow:rgba(56,189,248,0.25);
  --amber:#f59e0b;--adim:rgba(245,158,11,0.08);--aglow:rgba(245,158,11,0.25);
  --rose:#f43f5e;--rdim:rgba(244,63,94,0.08);--rglow:rgba(244,63,94,0.25);
  --violet:#a78bfa;--vdim:rgba(167,139,250,0.08);
  --tp:#f1f5f9;--ts:#94a3b8;--th:#475569;
  --font-display:'Sora',sans-serif;
  --font-body:'Outfit',sans-serif;
  --font-mono:'IBM Plex Mono',monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:var(--font-body);background:var(--bg)!important;color:var(--tp);}
.stApp{background:var(--bg)!important;}
#MainMenu,footer,header{visibility:hidden!important;}
.block-container{padding:1.75rem 2rem 3rem!important;max-width:100%!important;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]>div{
  background:var(--bg2)!important;
  border-right:1px solid var(--bd);
  padding:1.25rem .875rem 1rem!important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button{
  background:transparent!important;border:1px solid transparent!important;
  color:var(--ts)!important;text-align:left!important;font-family:var(--font-body)!important;
  font-size:13px!important;font-weight:500!important;border-radius:10px!important;
  padding:9px 13px!important;margin-bottom:1px!important;width:100%!important;
  transition:all .18s!important;letter-spacing:.1px!important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover{
  background:rgba(255,255,255,0.04)!important;color:var(--tp)!important;
  border-color:var(--bd)!important;
}

/* ── Cards — no empty box flash ── */
.rp-card{
  background:var(--bg3);border:1px solid var(--bd);border-radius:14px;
  padding:1.25rem 1.4rem;margin-bottom:1rem;
  transition:border-color .2s;
}
.rp-card:hover{border-color:var(--bd2);}
.rp-card-neon{border-color:rgba(0,255,136,0.18);background:linear-gradient(145deg,rgba(0,255,136,0.03),var(--bg3));}
.rp-card-cyber{border-color:rgba(56,189,248,0.18);background:linear-gradient(145deg,rgba(56,189,248,0.03),var(--bg3));}
.rp-card-amber{border-color:rgba(245,158,11,0.22);background:linear-gradient(145deg,rgba(245,158,11,0.04),var(--bg3));}
.rp-card-rose{border-color:rgba(244,63,94,0.22);background:linear-gradient(145deg,rgba(244,63,94,0.04),var(--bg3));}

/* ── KPI tiles ── */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:1.5rem;}
.kpi{
  background:var(--bg3);border:1px solid var(--bd);border-radius:14px;
  padding:1.1rem 1.3rem;position:relative;overflow:hidden;
  transition:transform .18s,border-color .18s;
}
.kpi:hover{transform:translateY(-2px);border-color:var(--bd2);}
.kpi::after{
  content:'';position:absolute;top:0;right:0;width:60px;height:60px;
  border-radius:50%;filter:blur(28px);opacity:.35;
}
.kpi-n::after{background:var(--neon);}.kpi-n{border-top:2px solid var(--neon);}
.kpi-c::after{background:var(--cyber);}.kpi-c{border-top:2px solid var(--cyber);}
.kpi-a::after{background:var(--amber);}.kpi-a{border-top:2px solid var(--amber);}
.kpi-r::after{background:var(--rose);} .kpi-r{border-top:2px solid var(--rose);}
.kpi-lbl{font-family:var(--font-body);font-size:10.5px;font-weight:600;letter-spacing:1.1px;text-transform:uppercase;color:var(--th);margin-bottom:10px;}
.kpi-val{font-family:var(--font-display);font-size:28px;font-weight:700;color:var(--tp);line-height:1;letter-spacing:-0.5px;}
.kpi-d{font-size:11.5px;margin-top:8px;font-weight:500;}
.up{color:var(--neon);}.down{color:var(--rose);}.flat{color:var(--ts);}

/* ── Page headers ── */
.ph{font-family:var(--font-display);font-size:26px;font-weight:700;color:var(--tp);margin-bottom:3px;letter-spacing:-0.5px;}
.ph .acc{
  background:linear-gradient(90deg,var(--neon),var(--cyber));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.ps{font-size:13px;color:var(--ts);margin-bottom:1.75rem;font-weight:400;}

/* ── Section headers ── */
.sl{font-family:var(--font-display);font-size:14px;font-weight:600;color:var(--tp);margin-bottom:3px;}
.ss{font-size:11.5px;color:var(--ts);margin-bottom:.875rem;}

/* ── Badges ── */
.badge{
  display:inline-block;font-family:var(--font-body);font-size:10.5px;
  font-weight:600;padding:2.5px 9px;border-radius:20px;letter-spacing:.3px;
}
.b-n{background:var(--ndim);color:var(--neon);border:1px solid var(--nbright);}
.b-c{background:var(--cdim);color:var(--cyber);border:1px solid var(--cglow);}
.b-a{background:var(--adim);color:var(--amber);border:1px solid var(--aglow);}
.b-r{background:var(--rdim);color:var(--rose);border:1px solid var(--rglow);}
.b-g{background:rgba(255,255,255,.05);color:var(--ts);border:1px solid var(--bd);}
.b-v{background:var(--vdim);color:var(--violet);border:1px solid rgba(167,139,250,.3);}

/* ── Dividers ── */
.nd{height:1px;background:linear-gradient(90deg,transparent,var(--neon) 40%,transparent);margin:1.1rem 0;opacity:.25;}
.cd{height:1px;background:linear-gradient(90deg,transparent,var(--cyber) 40%,transparent);margin:1.1rem 0;opacity:.25;}

/* ── Prob bars ── */
.pb-wrap{margin:.5rem 0;}
.pb-row{display:flex;justify-content:space-between;font-size:12px;color:var(--ts);margin-bottom:5px;}
.pb-track{height:7px;background:var(--bg5);border-radius:4px;overflow:hidden;}
.pb-fill{height:100%;border-radius:4px;transition:width .6s ease;}

/* ── Timeline ── */
.tl{position:relative;padding-left:26px;}
.tl::before{content:'';position:absolute;left:5px;top:8px;bottom:8px;width:1.5px;background:var(--bg5);border-radius:2px;}
.tl-s{position:relative;margin-bottom:20px;}
.tl-d{position:absolute;left:-26px;top:2px;width:12px;height:12px;border-radius:50%;border:2px solid var(--bg5);background:var(--bg2);}
.tl-d.done{background:var(--neon);border-color:var(--neon);box-shadow:0 0 8px var(--nglow);}
.tl-d.active{background:var(--cyber);border-color:var(--cyber);box-shadow:0 0 8px var(--cglow);animation:rp-pulse 1.8s ease-in-out infinite;}
.tl-d.pending{background:var(--bg5);border-color:var(--bd2);}
@keyframes rp-pulse{0%,100%{box-shadow:0 0 6px var(--cglow);}50%{box-shadow:0 0 18px var(--cglow);}}
@keyframes sb-glow{0%,100%{box-shadow:0 0 6px var(--nglow);}50%{box-shadow:0 0 14px var(--nglow);}}
.tl-lbl{font-size:13px;font-weight:600;color:var(--tp);}
.tl-meta{font-size:11px;color:var(--ts);margin-top:2px;line-height:1.5;}

/* ── Hero result ── */
.hero-pct{font-family:var(--font-display);font-size:88px;font-weight:800;line-height:1;letter-spacing:-3px;text-align:center;}
.hero-sublabel{font-size:12px;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;margin-top:8px;text-align:center;}
.hero-eta{font-size:13px;color:var(--ts);margin-top:5px;text-align:center;}

/* ── AI says ── */
.ai-says{
  background:linear-gradient(135deg,rgba(56,189,248,0.06),rgba(0,255,136,0.03));
  border:1px solid rgba(56,189,248,0.18);border-radius:12px;
  padding:.875rem 1.1rem;margin-bottom:.875rem;display:flex;gap:10px;align-items:flex-start;
}
.ai-says-icon{font-size:20px;flex-shrink:0;margin-top:1px;}
.ai-says-title{font-size:10.5px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--cyber);margin-bottom:4px;}
.ai-says-text{font-size:13px;color:var(--ts);line-height:1.6;}
.ai-says-text strong{color:var(--tp);font-weight:600;}

/* ── Health bars ── */
.hb-row{display:flex;align-items:center;gap:10px;margin-bottom:9px;}
.hb-nm{font-size:12px;color:var(--ts);width:145px;flex-shrink:0;}
.hb-trk{flex:1;height:4px;background:var(--bg5);border-radius:3px;overflow:hidden;}
.hb-fil{height:100%;border-radius:3px;}
.hb-val{font-size:11.5px;font-weight:600;color:var(--tp);width:34px;text-align:right;}

/* ── Profile ── */
.pf-avatar{
  width:72px;height:72px;border-radius:50%;
  background:linear-gradient(135deg,var(--neon),var(--cyber));
  display:flex;align-items:center;justify-content:center;
  font-family:var(--font-display);font-size:22px;font-weight:700;color:#000;
  margin:0 auto .875rem;
}
.pf-name{font-family:var(--font-display);font-size:18px;font-weight:700;text-align:center;color:var(--tp);}
.pf-sub{font-size:12px;color:var(--ts);text-align:center;margin-top:3px;}
.pf-stat{background:var(--bg4);border-radius:10px;padding:10px;text-align:center;}
.pf-sv{font-family:var(--font-display);font-size:20px;font-weight:700;color:var(--tp);}
.pf-sl{font-size:10.5px;color:var(--ts);margin-top:2px;}

/* ── Auth ── */
.auth-wrap{
  max-width:400px;margin:0 auto;background:var(--bg2);
  border:1px solid var(--bd);border-radius:18px;padding:2.25rem;
  box-shadow:0 20px 60px rgba(0,0,0,.7);
}
.auth-logo{font-family:var(--font-display);font-size:26px;font-weight:800;color:var(--tp);text-align:center;margin-bottom:5px;}
.auth-tag{font-size:13px;color:var(--ts);text-align:center;margin-bottom:1.5rem;}
.auth-title{font-family:var(--font-display);font-size:17px;font-weight:700;color:var(--tp);margin-bottom:1.1rem;}

/* ── Sidebar brand ── */
.sb-brand{
  font-family:var(--font-display);font-size:16px;font-weight:700;color:var(--tp);
  display:flex;align-items:center;gap:8px;margin-bottom:1.25rem;letter-spacing:-.3px;
}
.sb-pulse{
  width:8px;height:8px;border-radius:50%;background:var(--neon);
  box-shadow:0 0 8px var(--nglow);animation:sb-glow 2s infinite;
}
.nav-hint{font-size:9.5px;color:var(--neon);letter-spacing:.8px;text-transform:uppercase;margin-bottom:2px;margin-left:2px;opacity:.6;}

/* ── Insight callout ── */
.insight{
  background:rgba(56,189,248,0.05);border-left:2px solid var(--cyber);
  border-radius:0 10px 10px 0;padding:9px 14px;margin-bottom:1.25rem;
  font-size:12.5px;color:var(--ts);line-height:1.6;
}
.insight strong{color:var(--cyber);}

/* ── Form inputs ── */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"]>div>div,
div[data-testid="stDateInput"] input,
div[data-testid="stNumberInput"] input{
  background:var(--bg4)!important;border:1px solid var(--bd2)!important;
  color:var(--tp)!important;border-radius:10px!important;
  font-family:var(--font-body)!important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stNumberInput"] label{
  color:var(--th)!important;font-size:10.5px!important;font-weight:600!important;
  letter-spacing:.9px!important;text-transform:uppercase!important;
  font-family:var(--font-body)!important;
}

/* ── Buttons ── */
div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,var(--neon),#00d4aa)!important;color:#000!important;
  font-weight:700!important;border:none!important;border-radius:10px!important;
  font-family:var(--font-display)!important;font-size:13.5px!important;
  letter-spacing:.2px!important;transition:opacity .2s,transform .15s,box-shadow .2s!important;
}
div[data-testid="stButton"]>button:hover{
  opacity:.88!important;transform:translateY(-1px)!important;
  box-shadow:0 6px 20px var(--nglow)!important;
}

/* ── Slider ── */
[data-testid="stSlider"]>div>div>div{background:var(--neon)!important;}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid var(--bd)!important;}
[data-testid="stTabs"] [role="tab"]{color:var(--ts)!important;font-size:13px!important;font-family:var(--font-body)!important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--neon)!important;border-bottom:2px solid var(--neon)!important;}

/* ── Expanders ── */
[data-testid="stExpander"]{
  background:var(--bg3)!important;border:1px solid var(--bd)!important;
  border-radius:12px!important;margin-bottom:.75rem!important;
}
[data-testid="stExpander"] summary{
  font-family:var(--font-body)!important;font-size:13px!important;
  font-weight:500!important;color:var(--ts)!important;padding:.75rem 1rem!important;
}
[data-testid="stExpander"] summary:hover{color:var(--tp)!important;}

/* ── Dataframes ── */
[data-testid="stDataFrame"]{border-radius:10px!important;overflow:hidden;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--bg5);border-radius:3px;}

/* ── Alt-train cards ── */
.alt-card{
  background:var(--bg4);border:1px solid var(--bd);border-radius:12px;
  padding:.875rem 1.1rem;margin-bottom:7px;display:flex;align-items:center;
  justify-content:space-between;transition:border-color .18s,transform .15s;cursor:pointer;
}
.alt-card:hover{border-color:var(--nglow);transform:translateX(2px);}
.score-h{color:var(--neon);font-family:var(--font-display);font-size:20px;font-weight:700;}
.score-m{color:var(--amber);font-family:var(--font-display);font-size:20px;font-weight:700;}
.score-l{color:var(--rose);font-family:var(--font-display);font-size:20px;font-weight:700;}
</style>
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SESSION (was core/session.py)
# ═════════════════════════════════════════════════════════════════════════════
def _session_defaults() -> None:
    defaults = {
        "logged_in":   False,
        "username":    "",
        "page":        "Dashboard",
        "pred_result": None,
        "show_alts":   False,
        "alert_set":   False,
        "pnr_queried": "6230185420",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
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


def _compute_model_dashboard_stats():
    """
    Derive real dashboard stats from the trained model payload.
    Returns dict with model_source='real_model' on success,
    or model_source='seed' with load_error string on failure.
    """
    import joblib
    rng = random.Random(SEED)

    fallback = {
        "bookings_today": rng.randint(4500, 4900),
        "wl_confirmed":   rng.randint(1050, 1150),
        "active_wl":      rng.randint(370, 400),
        "model_conf":     84.3,
        "bookings_trend": [rng.randint(280, 640) for _ in range(14)],
        "wl_trend":       [rng.randint(40, 130)  for _ in range(14)],
        "conf_rates":     [68, 72, 61, 89, 83, 57],
        "model_acc":      [74.2, 76.8, 78.1, 79.4, 81.0, 82.7, 84.3],
        "health":         [97, 99, 88, 94, 91],
        "heatmap_demand": [rng.randint(40, 100) for _ in range(15)],
        "heatmap_wl":     [rng.randint(2, 35)   for _ in range(15)],
        "revenue_slices": [rng.randint(8, 15)   for _ in range(7)],
        "model_source":   "seed",
        "best_model_name":"—",
        "n_trains":        0,
        "load_error":      "",
    }

    # -- Resolve the model path explicitly from app.py location ---------------
    app_dir    = Path(__file__).resolve().parent
    model_path = app_dir / "outputs" / "best_model_real.joblib"

    if not model_path.exists():
        fallback["load_error"] = f"File not found: {model_path}"
        return fallback

    try:
        pl      = joblib.load(model_path)
        metrics = pl.get("metrics", {})
        tf_     = pl.get("train_data")
        best    = pl.get("best_model_name", "ML Model")

        # Real accuracy from saved metrics
        # The metrics dict structure is {metric_name: {model_name: value}}
        # e.g. {"Accuracy": {"Gradient Boosting": 0.9407, ...}}
        acc_val = 84.3
        if metrics:
            try:
                df_metrics = pd.DataFrame(metrics).T
                # Find the accuracy column — handle capitalisation variations
                acc_col = next(
                    (c for c in df_metrics.columns
                     if c.lower() in ("accuracy", "acc")),
                    None
                )
                if acc_col:
                    acc_series = df_metrics[acc_col].astype(float)
                    raw = float(acc_series.get(best, acc_series.max()))
                    # Values stored as 0-1 fractions → multiply by 100
                    acc_val = round(raw * 100 if raw <= 1.0 else raw, 1)
            except Exception:
                pass

        mdl      = pl["model"]
        imp_     = pl["imputer"]
        sca_     = pl["scaler"]
        le_coach = pl["le_coach"]
        le_season= pl["le_season"]
        need_pre = pl.get("need_preprocess", False)

        # Detect feature count from saved payload
        n_feat = 31
        fc = pl.get("feature_cols", [])
        if fc:
            n_feat = len(fc)
        elif hasattr(imp_, "n_features_in_"):
            n_feat = int(imp_.n_features_in_)

        t_mean = tf_.mean(numeric_only=True).to_dict() if tf_ is not None and len(tf_) > 0 else {}

        def _quick_prob(coach, wl=5, days=30):
            try:
                X = _single_row_array(wl, coach, days, "normal",
                                      0.15, 0, 5, 0,
                                      t_mean, le_coach, le_season,
                                      n_features=n_feat)
                if need_pre:
                    X = sca_.transform(imp_.transform(X))
                return round(float(mdl.predict_proba(X)[0, 1]) * 100, 1)
            except Exception:
                return 65.0

        conf_rates = [int(_quick_prob(c)) for c in ["SL","3A","2A","1A","CC","2S"]]
        model_acc  = [74.2, 76.8, 78.1, 79.4, 81.0, 82.7, round(acc_val, 1)]

        n_trains = len(tf_) if tf_ is not None else 5208
        rng2     = random.Random(SEED + 1)
        btd_base = int(4600 * max(1, n_trains / 5000))
        wl_base  = int(90   * max(1, n_trains / 5000))

        return {
            "bookings_today":  int(btd_base * rng2.uniform(0.95, 1.05)),
            "wl_confirmed":    int(btd_base * 0.22 * rng2.uniform(0.9, 1.1)),
            "active_wl":       int(btd_base * 0.08),
            "model_conf":      acc_val,
            "bookings_trend":  [int(btd_base * rng2.uniform(0.7, 1.3)) for _ in range(14)],
            "wl_trend":        [int(wl_base  * rng2.uniform(0.5, 1.5)) for _ in range(14)],
            "conf_rates":      conf_rates,
            "model_acc":       model_acc,
            "health":          [97, 99, 88, 94, 91],
            "heatmap_demand":  [rng2.randint(40, 100) for _ in range(15)],
            "heatmap_wl":      [rng2.randint(2, 35)   for _ in range(15)],
            "revenue_slices":  [rng2.randint(8, 15)   for _ in range(7)],
            "model_source":    "real_model",
            "best_model_name": best,
            "n_trains":        n_trains,
            "load_error":      "",
        }

    except Exception as e:
        import traceback
        fallback["load_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return fallback


def _seed_dashboard() -> None:
    """Load model stats once per session into session_state['dash']."""
    if st.session_state.get("dash_computed"):
        return
    result = _compute_model_dashboard_stats()
    st.session_state["dash"]          = result
    st.session_state["dash_computed"] = True


def ss(key, default=None):
    return st.session_state.get(key, default)

def ss_set(key, value):
    st.session_state[key] = value

def navigate(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ML FEATURES (was ml/features.py)
# ═════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "waitlist_position", "coach_type_enc",  "booking_day",     "season_enc",
    "cancellation_rate", "day_of_week",     "total_stops",     "duration_hours",
    "distance_km",       "train_type_enc",  "total_seats",     "speed_kmph",
    "seats_per_stop",    "quota_available", "is_weekend",      "is_holiday_week",
    "wl_x_cancel",       "wl_x_booking",    "coach_x_season",  "quota_x_wl",
    "dist_per_stop",     "has_ac",          "has_sleeper",     "premium_score",
    "zone_enc",
]

FEATURE_NAMES = [
    "Waitlist Position",  "Coach Type",      "Booking Day",    "Season",
    "Cancellation Rate",  "Day of Week",     "Total Stops",    "Duration (hrs)",
    "Distance (km)",      "Train Type",      "Total Seats",    "Speed (km/h)",
    "Seats/Stop",         "Quota Available", "Is Weekend",     "Is Holiday",
    "WL × Cancel",        "WL / BookDay",    "Coach × Season", "Quota / WL",
    "Dist/Stop",          "Has AC",          "Has Sleeper",    "Premium Score",
    "Zone",
]


def _make_encoders():
    from sklearn.preprocessing import LabelEncoder
    le_coach  = LabelEncoder().fit(COACH_TYPES)
    le_season = LabelEncoder().fit(SEASONS)
    return le_coach, le_season


def _add_train_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration_hours"] = df["duration_h"] + df["duration_m"] / 60
    df["total_seats"] = (
        df["first_ac"]    * 24 + df["second_ac"]   * 46 +
        df["third_ac"]    * 64 + df["sleeper"]     * 72 +
        df["chair_car"]   * 78 + df["first_class"] * 18
    )
    df["has_ac"]        = ((df["first_ac"] + df["second_ac"] + df["third_ac"]) > 0).astype(int)
    df["has_sleeper"]   = (df["sleeper"] > 0).astype(int)
    df["speed_kmph"]    = (df["distance_km"] / df["duration_hours"].replace(0, np.nan)).fillna(0)
    df["premium_score"] = (df["first_ac"] * 3 + df["second_ac"] * 2 + df["third_ac"] * 1 + df["chair_car"] * 1)
    return df


def _add_stop_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_stops"]    = df["total_stops"].fillna(5).astype(float)
    df["seats_per_stop"] = df["total_seats"] / df["total_stops"].replace(0, 1)
    df["dist_per_stop"]  = df["distance_km"] / df["total_stops"].replace(0, 1)
    return df


def _build_interaction_features(bk: pd.DataFrame) -> pd.DataFrame:
    bk = bk.copy()
    bk["wl_x_cancel"]    = bk["waitlist_position"] * bk["cancellation_rate"]
    bk["wl_x_booking"]   = bk["waitlist_position"] / (bk["booking_day"] + 1)
    bk["coach_x_season"] = bk["coach_type_enc"]    * bk["season_enc"]
    bk["quota_x_wl"]     = bk["quota_available"]   / (bk["waitlist_position"] + 1)
    bk["is_weekend"]     = (bk["day_of_week"] >= 5).astype(int)
    return bk


def _single_row_array(waitlist_position, coach_type, booking_day, season,
                      cancellation_rate, day_of_week, quota_available,
                      is_holiday_week, train_row, le_coach, le_season,
                      n_features=31) -> np.ndarray:
    """
    Build a feature vector matching whatever the saved model expects.
    n_features=25 → old model (25 base features)
    n_features=31 → improved model (25 base + 6 new engineered features)
    """
    coach_enc  = int(le_coach.transform([coach_type])[0]) if coach_type in le_coach.classes_ else 0
    season_enc = int(le_season.transform([season])[0])    if season in le_season.classes_ else 1
    is_weekend = int(day_of_week >= 5)
    wl_x_cancel  = waitlist_position * cancellation_rate
    wl_x_booking = waitlist_position / (booking_day + 1)
    coach_x_seas = coach_enc * season_enc
    quota_x_wl   = quota_available / (waitlist_position + 1)
    t = train_row

    total_seats = float(t.get("total_seats", 100))

    # 25 base features (same as original model)
    base = [
        waitlist_position,                      # 0
        coach_enc,                              # 1
        booking_day,                            # 2
        season_enc,                             # 3
        cancellation_rate,                      # 4
        day_of_week,                            # 5
        float(t.get("total_stops",     10)),    # 6
        float(t.get("duration_hours",   5)),    # 7
        float(t.get("distance_km",    500)),    # 8
        float(t.get("train_type_enc",   0)),    # 9
        total_seats,                            # 10
        float(t.get("speed_kmph",      60)),    # 11
        float(t.get("seats_per_stop",  10)),    # 12
        quota_available,                        # 13
        is_weekend,                             # 14
        is_holiday_week,                        # 15
        wl_x_cancel,                            # 16
        wl_x_booking,                           # 17
        coach_x_seas,                           # 18
        quota_x_wl,                             # 19
        float(t.get("dist_per_stop",   50)),    # 20
        float(t.get("has_ac",           0)),    # 21
        float(t.get("has_sleeper",      0)),    # 22
        float(t.get("premium_score",    0)),    # 23
        float(t.get("zone_enc",         0)),    # 24
    ]

    if n_features <= 25:
        return np.array([base], dtype=float)

    # 6 additional engineered features (improved model, n_features=31)
    extra = [
        waitlist_position / max(total_seats, 1),        # 25 wl_per_seat
        waitlist_position / (quota_available + 1),      # 26 booking_pressure
        float(min(int(booking_day / 15), 4)),           # 27 days_bucket (0-4)
        total_seats * cancellation_rate,                 # 28 route_popularity
        float(waitlist_position ** 2),                  # 29 wl_squared
        quota_available / max(total_seats, 1),          # 30 quota_ratio
    ]
    return np.array([base + extra], dtype=float)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PREDICTOR (was ml/predictor.py — cleaned & fixed)
# ═════════════════════════════════════════════════════════════════════════════
class Predictor:
    """Loads trained joblib model if present; falls back to heuristic."""

    def __init__(self):
        self._payload   = None
        self._le_coach  = None
        self._le_season = None
        self._loaded    = False
        self._try_load()

    def _try_load(self):
        # Resolve path explicitly from app.py location — same as _compute_model_dashboard_stats
        model_path = Path(__file__).resolve().parent / "outputs" / "best_model_real.joblib"
        if model_path.exists():
            try:
                import joblib
                self._payload   = joblib.load(model_path)
                self._le_coach  = self._payload["le_coach"]
                self._le_season = self._payload["le_season"]
                self._loaded    = True
            except Exception as e:
                import traceback
                st.warning(f"Model load failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                self._loaded = False
        if not self._loaded:
            self._le_coach, self._le_season = _make_encoders()

    def predict(self, train_id, coach_type, wl_no, booking_day,
                season="normal", day_of_week=0, cancellation_rate=0.15,
                quota_available=5, is_holiday_week=0) -> dict:
        if self._loaded and self._payload is not None:
            return self._ml_predict(train_id, coach_type, wl_no, booking_day,
                                    season, day_of_week, cancellation_rate,
                                    quota_available, is_holiday_week)
        return self._heuristic_predict(train_id, coach_type, wl_no,
                                       booking_day, season, day_of_week)

    @property
    def model_name(self) -> str:
        if self._loaded and self._payload:
            return self._payload.get("best_model_name", "ML Model")
        return "Heuristic"

    @property
    def metrics(self) -> Optional[dict]:
        return self._payload.get("metrics") if self._payload else None

    def _ml_predict(self, train_id, coach_type, wl_no, booking_day,
                    season, day_of_week, cancellation_rate,
                    quota_available, is_holiday_week) -> dict:
        pl   = self._payload
        mdl  = pl["model"]
        _imp = pl["imputer"]
        _sca = pl["scaler"]
        tf_  = pl["train_data"]

        # Detect how many features the saved model expects
        # Check imputer (always present), fall back to feature_cols list
        n_features = 31  # default: improved model
        try:
            fc = pl.get("feature_cols", [])
            if fc:
                n_features = len(fc)
            elif hasattr(_imp, "n_features_in_"):
                n_features = int(_imp.n_features_in_)
        except Exception:
            pass

        row    = tf_[tf_["train_number"] == str(train_id)]
        t_dict = row.iloc[0].to_dict() if not row.empty else {}
        if not row.empty:
            r = row.iloc[0]
            train_info = {
                "name":     str(r.get("train_name", "")),
                "type":     str(r.get("train_type", "")),
                "distance": float(r.get("distance_km", 0)),
                "from":     str(r.get("from_station_name", "")),
                "to":       str(r.get("to_station_name", "")),
            }
        else:
            t_dict     = tf_.mean(numeric_only=True).to_dict()
            train_info = {}

        X_in = _single_row_array(
            wl_no, coach_type, booking_day, season,
            cancellation_rate, day_of_week, quota_available,
            is_holiday_week, t_dict, self._le_coach, self._le_season,
            n_features=n_features,
        )
        if pl.get("need_preprocess", False):
            X_in = _sca.transform(_imp.transform(X_in))

        prob = float(mdl.predict_proba(X_in)[0, 1])
        return self._package(prob, train_id, coach_type, wl_no,
                             booking_day, season, train_info, self.model_name)

    def _heuristic_predict(self, train_id, coach_type, wl_no,
                           booking_day, season, day_of_week) -> dict:
        base        = CLASS_META.get(coach_type, {}).get("base_confirm", 65)
        wl_penalty  = min(wl_no * 3.8, 58)
        day_bonus   = min(booking_day * 0.9, 22)
        train_bonus = {"12027":9,"11007":5,"12123":-4,"12301":12,"11301":-6}.get(str(train_id), 0)
        season_adj  = {"off-peak":8, "normal":0, "peak":-10}.get(season, 0)
        prob_pct    = max(4, min(97, base - wl_penalty + day_bonus + train_bonus + season_adj))
        prob        = prob_pct / 100
        train_info  = {}
        for t in DEMO_TRAINS:
            if t["id"] == str(train_id):
                train_info = {"name": t["name"], "type": t["type"],
                              "from": t["dep_stn"], "to": t["arr_stn"]}
                break
        return self._package(prob, train_id, coach_type, wl_no,
                             booking_day, season, train_info, "Heuristic")

    @staticmethod
    def _package(prob, train_id, coach_type, wl_no,
                 booking_day, season, train_info, model_used) -> dict:
        band = ("High (>75%)"     if prob >= 0.75 else
                "Medium (50–75%)" if prob >= 0.50 else
                "Low (25–50%)"    if prob >= 0.25 else
                "Very Low (<25%)")
        alts = []
        if prob < 0.50:
            for c in [x for x in list(CLASS_META.keys()) if x != coach_type][:2]:
                alts.append(f"Try {c} class — may have better quota availability")
            if booking_day < 30:
                alts.append("Booking 30+ days in advance improves WL clearance significantly")
            if season == "peak":
                alts.append("Off-peak travel improves confirmation probability by ~20%")
        return {
            "probability":      round(prob, 4),
            "probability_pct":  round(prob * 100, 1),
            "confirmed":        prob >= 0.50,
            "confidence_band":  band,
            "model_used":       model_used,
            "train_info":       train_info,
            "alternatives":     alts,
            "input": {
                "train_id": train_id, "coach_type": coach_type,
                "wl_no": wl_no, "booking_day": booking_day, "season": season,
            },
        }


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data(show_spinner=False)
def _load_real_stations():
    """
    Load all 8,990 stations from data/stations.json.
    Returns (display_list, code_map) where display_list is sorted
    "CODE — Name" strings and code_map maps display → code.
    Falls back to a small hardcoded list if the file is missing.
    """
    path = ROOT / "data" / "stations.json"
    if not path.exists():
        fallback = ["PUNE — Pune Junction","CSMT — Mumbai CST",
                    "NDLS — New Delhi",    "SBC — Bangalore City",
                    "MAS — Chennai Central","NGP — Nagpur",
                    "HWH — Howrah",        "ADI — Ahmedabad"]
        return fallback, {s: s.split(" — ")[0] for s in fallback}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    seen = {}
    for feat in raw.get("features", []):
        p    = feat.get("properties", {})
        code = p.get("code", "").strip()
        name = p.get("name", "").strip()
        if code and name and code not in seen:
            seen[code] = name

    sorted_rows  = sorted(seen.items(), key=lambda x: x[1])
    display_list = [f"{c} — {n}" for c, n in sorted_rows]
    code_map     = {f"{c} — {n}": c for c, n in sorted_rows}
    return display_list, code_map


@st.cache_data(show_spinner=False)
def _load_real_trains():
    """
    Load all trains from data/trains.json.
    Returns (display_list, train_map, route_index) where:
      - display_list: sorted "NUMBER — Name" strings
      - train_map: display_str → train_number
      - route_index: dict {from_code: {to_code: [train_numbers]}}
        also includes intermediate stops if schedules available
    """
    path = ROOT / "data" / "trains.json"
    if not path.exists():
        display_list = [f"{t['id']} — {t['name']}" for t in DEMO_TRAINS]
        train_map    = {f"{t['id']} — {t['name']}": t["id"] for t in DEMO_TRAINS}
        return display_list, train_map, {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    seen        = {}   # number → name
    route_index = {}   # from_code → set of train numbers (trains that originate there)
    # Also build: number → (from_code, to_code) for direct route matching
    train_routes = {}  # number → {"from": code, "to": code, "name": name}

    for feat in raw.get("features", []):
        p      = feat.get("properties", {})
        number = str(p.get("number", "")).strip()
        name   = p.get("name",   "").strip()
        from_c = str(p.get("from_station_code", "")).strip().upper()
        to_c   = str(p.get("to_station_code",   "")).strip().upper()
        if not number or not name:
            continue
        if number not in seen:
            seen[number] = name
        train_routes[number] = {"from": from_c, "to": to_c, "name": name}

        # Index by origin station
        if from_c:
            if from_c not in route_index:
                route_index[from_c] = {}
            if to_c not in route_index[from_c]:
                route_index[from_c][to_c] = []
            if number not in route_index[from_c][to_c]:
                route_index[from_c][to_c].append(number)

    sorted_rows  = sorted(seen.items(), key=lambda x: x[0].zfill(6))
    display_list = [f"{num} — {name}" for num, name in sorted_rows]
    train_map    = {f"{num} — {name}": num for num, name in sorted_rows}

    # Store train_routes inside a wrapper so callers can access it
    train_map["__routes__"] = train_routes  # type: ignore
    return display_list, train_map, route_index


def _get_trains_for_route(from_code: str, to_code: str,
                          train_map: dict, route_index: dict) -> list:
    """
    Return filtered list of "NUMBER — Name" display strings for trains
    that run from from_code to to_code (direct origin→destination match).
    Falls back to all trains if none found.
    """
    train_routes = train_map.get("__routes__", {})

    # Direct match: train starts at from_code and ends at to_code
    direct = route_index.get(from_code, {}).get(to_code, [])

    # Partial match: train starts at from_code (goes anywhere via to_code)
    via = []
    if from_code in route_index:
        for dest, nums in route_index[from_code].items():
            if dest != to_code:
                via.extend(nums)

    # Combine: direct first, then via
    combined = list(dict.fromkeys(direct + via))  # preserve order, dedupe

    if not combined:
        # No match — return full list so user is never stuck
        return [k for k in train_map.keys() if k != "__routes__"]

    result = []
    for num in combined:
        info = train_routes.get(num, {})
        name = info.get("name", "")
        if name:
            result.append(f"{num} — {name}")

    return result if result else [k for k in train_map.keys() if k != "__routes__"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CHARTS (was ui/charts.py — all bugs fixed)
# ═════════════════════════════════════════════════════════════════════════════
_BL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,32,0.6)",
    font=dict(family="Outfit", color="#94a3b8", size=11),
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.05)",
               tickfont=dict(size=11, color="#475569")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.05)",
               tickfont=dict(size=11, color="#475569")),
)


def chart_booking_trend(bookings_trend, wl_trend):
    dates = [date.today() - timedelta(days=i) for i in range(13, -1, -1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=bookings_trend, name="Bookings",
        fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        line=dict(color="#38bdf8", width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=dates, y=wl_trend, name="Waitlisted",
        fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
        line=dict(color="#f59e0b", width=2, dash="dot"), mode="lines"))
    fig.update_layout(**_BL, height=200, showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), x=0.01, y=0.99))
    return fig


def chart_class_rates(conf_rates):
    classes = ["SL","3A","2A","1A","CC","EC"]
    colors  = ["#00ff88" if r>=75 else "#38bdf8" if r>=60 else "#f59e0b" for r in conf_rates]
    fig = go.Figure(go.Bar(x=classes, y=conf_rates, marker_color=colors,
        text=[f"{r}%" for r in conf_rates], textposition="outside",
        textfont=dict(color="#f1f5f9", size=11)))
    fig.add_hline(y=75, line_dash="dash", line_color="rgba(0,255,136,0.35)",
        annotation_text="Target", annotation_font_color="#00ff88", annotation_font_size=10)
    yax = {**_BL["yaxis"], "range":[0,105]}
    bl  = {k:v for k,v in _BL.items() if k!="yaxis"}
    fig.update_layout(**bl, yaxis=yax, height=200, showlegend=False)
    return fig


def chart_model_accuracy(model_acc):
    months = ["Sep","Oct","Nov","Dec","Jan","Feb","Mar"]
    fig = go.Figure(go.Scatter(x=months, y=model_acc, mode="lines+markers",
        line=dict(color="#00ff88", width=3),
        marker=dict(color="#00ff88", size=7, line=dict(color="#080810", width=2)),
        fill="tozeroy", fillcolor="rgba(0,255,136,0.05)"))
    yax = {**_BL["yaxis"], "range":[70,90]}
    bl  = {k:v for k,v in _BL.items() if k!="yaxis"}
    fig.update_layout(**bl, yaxis=yax, height=190, showlegend=False)
    return fig


def chart_gauge(value):
    color = "#00ff88" if value >= 75 else "#f59e0b" if value >= 45 else "#f43f5e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix":"%","font":{"size":48,"color":"#f1f5f9","family":"Sora"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"rgba(255,255,255,0.1)","tickfont":{"color":"#475569","size":10}},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"rgba(34,34,47,0.8)","borderwidth":0,
            "steps":[{"range":[0,45],"color":"rgba(244,63,94,0.10)"},
                     {"range":[45,75],"color":"rgba(245,158,11,0.10)"},
                     {"range":[75,100],"color":"rgba(0,255,136,0.10)"}],
            "threshold":{"line":{"color":color,"width":3},"thickness":0.85,"value":value},
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=210, margin=dict(l=20,r=20,t=30,b=10))
    return fig


def chart_heatmap(heatmap_demand, heatmap_wl, routes):
    rows = [{"source":s,"dest":t,"demand":heatmap_demand[i],"wl_avg":heatmap_wl[i],
             "route":f"{s} → {t}"} for i,(s,t) in enumerate(routes)]
    df = pd.DataFrame(rows)
    fig = px.treemap(df, path=["source","dest"], values="demand", color="wl_avg",
        color_continuous_scale=["#080810","#38bdf8","#00ff88"], color_continuous_midpoint=18)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=360,
        coloraxis_colorbar=dict(title="WL",tickfont=dict(color="#94a3b8"),title_font=dict(color="#94a3b8")))
    fig.update_traces(textfont=dict(color="#f1f5f9", size=12))
    return fig, df


def chart_scatter_demand(df):
    fig = px.scatter(df, x="demand", y="wl_avg", text="route", size="demand",
        color="wl_avg", color_continuous_scale=["#00ff88","#f59e0b","#f43f5e"],
        hover_name="route", size_max=28)
    fig.update_traces(textposition="top center", textfont=dict(color="#94a3b8", size=9))
    fig.update_layout(**_BL, height=320, showlegend=False,
        xaxis_title="Demand Index", yaxis_title="Avg Waitlist", coloraxis_showscale=False)
    return fig


def chart_revenue_donut(revenue_slices):
    classes = ["SL","3A","2A","1A","CC","EC","TQ"]
    total   = sum(revenue_slices)
    fig = go.Figure(go.Pie(labels=classes, values=revenue_slices, hole=0.55,
        marker=dict(colors=["#38bdf8","#00ff88","#f59e0b","#f43f5e","#b46eff","#ff6bba","#73ffd8"],
                    line=dict(color="rgba(0,0,0,0.4)",width=1.5)),
        textfont=dict(color="#f1f5f9", size=11)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280,
        font=dict(color="#94a3b8"),
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0,r=0,t=10,b=0),
        annotations=[dict(text=f"₹{total}L", x=0.5, y=0.5,
            font=dict(size=20, color="#f1f5f9", family="Sora"), showarrow=False)])
    return fig


def chart_train_booking_trend(train_id: str, train_name: str, predictor):
    """
    14-day simulated booking trend for a specific train.
    Uses the model to score WL/1–20 for different booking days → demand proxy.
    """
    dates  = [date.today() - timedelta(days=i) for i in range(13, -1, -1)]
    rng    = random.Random(hash(train_id) % 9999)

    bookings, wl_counts = [], []
    for i, d in enumerate(dates):
        days_ahead = 14 - i + rng.randint(1, 7)
        result     = predictor.predict(train_id=train_id, coach_type="SL",
                                       wl_no=rng.randint(3, 15),
                                       booking_day=days_ahead, season="normal")
        prob = result["probability"]
        # Scale bookings by probability — high-confirm trains → more bookings
        base_bookings = int(300 + prob * 400 + rng.uniform(-50, 50))
        base_wl       = int(30  + (1 - prob) * 80 + rng.uniform(-10, 10))
        bookings.append(max(50, base_bookings))
        wl_counts.append(max(5,  base_wl))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=bookings, name="Bookings",
        fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        line=dict(color="#38bdf8", width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=dates, y=wl_counts, name="Waitlisted",
        fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
        line=dict(color="#f59e0b", width=2, dash="dot"), mode="lines"))
    fig.update_layout(**_BL, height=220, showlegend=True,
        title=dict(text=f"<b>{train_name}</b> — Booking Trend",
                   font=dict(color="#f1f5f9", size=12), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11),
                    x=0.01, y=0.99))
    return fig, bookings, wl_counts


def chart_train_class_rates(train_id: str, predictor):
    """
    Per-class confirmation rates for a specific train using the real model.
    Runs WL/5, 30 days ahead for each class.
    """
    classes = ["SL","3A","2A","1A","CC","2S"]
    rates   = []
    for cls in classes:
        result = predictor.predict(train_id=train_id, coach_type=cls,
                                   wl_no=5, booking_day=30, season="normal")
        rates.append(int(result["probability_pct"]))

    colors = ["#00ff88" if r>=75 else "#38bdf8" if r>=50 else "#f59e0b" for r in rates]
    fig = go.Figure(go.Bar(x=classes, y=rates, marker_color=colors,
        text=[f"{r}%" for r in rates], textposition="outside",
        textfont=dict(color="#f1f5f9", size=11)))
    fig.add_hline(y=75, line_dash="dash", line_color="rgba(0,255,136,0.35)",
        annotation_text="Target", annotation_font_color="#00ff88",
        annotation_font_size=10)
    yax = {**_BL["yaxis"], "range":[0,110]}
    bl  = {k:v for k,v in _BL.items() if k!="yaxis"}
    fig.update_layout(**bl, yaxis=yax, height=220, showlegend=False)
    return fig, rates


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — UI COMPONENTS (was ui/components.py)
# ═════════════════════════════════════════════════════════════════════════════
def card_open(accent=""):
    cls = f"rp-card rp-card-{accent}" if accent else "rp-card"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)

def card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def section_header(title, subtitle=""):
    st.markdown(f'<div class="sl">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ss">{subtitle}</div>', unsafe_allow_html=True)

def page_header(title_plain, title_accent, subtitle):
    st.markdown(
        f'<div class="ph">{title_plain} <span class="acc">{title_accent}</span></div>'
        f'<div class="ps">{subtitle}</div>',
        unsafe_allow_html=True,
    )

def kpi_grid(*items):
    accents = ["n","c","a","r"]
    cells = ""
    for i, (label, value, delta, direction) in enumerate(items):
        acc = accents[i % 4]
        cells += (f'<div class="kpi kpi-{acc}">'
                  f'<div class="kpi-lbl">{label}</div>'
                  f'<div class="kpi-val">{value}</div>'
                  f'<div class="kpi-d {direction}">{delta}</div>'
                  f'</div>')
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)

def badge(text, variant="g"):
    return f'<span class="badge b-{variant}">{text}</span>'

def badge_row(*badges):
    st.markdown('<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">'
                + "".join(badges) + '</div>', unsafe_allow_html=True)

def neon_divider():
    st.markdown('<div class="nd"></div>', unsafe_allow_html=True)

def cyber_divider():
    st.markdown('<div class="cd"></div>', unsafe_allow_html=True)

def health_bar(name, pct):
    color = "#00ff88" if pct >= 95 else "#38bdf8" if pct >= 85 else "#f59e0b"
    st.markdown(f"""
    <div class="hb-row">
        <div class="hb-nm">{name}</div>
        <div class="hb-trk"><div class="hb-fil" style="width:{pct}%;background:{color};opacity:.85;"></div></div>
        <div class="hb-val">{pct}%</div>
    </div>""", unsafe_allow_html=True)

def prob_bar(label, value, display, color, note=""):
    note_html = f'<div style="font-size:11px;color:#475569;margin-top:3px;">{note}</div>' if note else ""
    st.markdown(f"""
    <div class="pb-wrap">
        <div class="pb-row"><span>{label}</span><span style="color:{color};font-weight:600;">{display}</span></div>
        <div class="pb-track"><div class="pb-fill" style="width:{min(value,100):.0f}%;background:{color};opacity:.8;"></div></div>
        {note_html}
    </div>""", unsafe_allow_html=True)

def timeline(steps):
    st.markdown('<div class="tl">', unsafe_allow_html=True)
    for step in steps:
        st.markdown(f"""
        <div class="tl-s">
            <div class="tl-d {step['state']}"></div>
            <div class="tl-lbl">{step['label']}</div>
            <div class="tl-meta">{step['time']} · {step['note']}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def wl_sparkline(wl_history):
    colors = {"CNF": "#00ff88", "RAC": "#38bdf8"}
    spans  = ""
    for i, v in enumerate(wl_history):
        if str(v) in colors:
            col = colors[str(v)]
        elif isinstance(v, int):
            col = ("#00ff88" if v <= 4 else "#38bdf8" if v <= 8 else "#f59e0b" if v <= 15 else "#f43f5e")
        else:
            col = "#94a3b8"
        label = str(v) if not isinstance(v, int) else f"WL/{v}"
        arrow = " →" if i < len(wl_history) - 1 else ""
        spans += (f'<span style="color:{col};font-family:var(--font-mono),'
                  f'monospace;font-size:12px;">{label}{arrow}</span> ')
    st.markdown(
        f'<div style="padding:10px 12px;background:rgba(255,255,255,0.03);'
        f'border-radius:8px;margin-bottom:1.25rem;line-height:2;">{spans}</div>',
        unsafe_allow_html=True,
    )

def insight(html_body):
    st.markdown(f'<div class="insight">{html_body}</div>', unsafe_allow_html=True)

def info_grid(*items):
    cells = "".join([
        f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:11px;">'
        f'<div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.8px;">{lbl}</div>'
        f'<div style="font-size:14px;font-weight:600;color:#f1f5f9;margin-top:4px;">{val}</div></div>'
        for lbl, val in items
    ])
    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:1.25rem;">{cells}</div>',
        unsafe_allow_html=True,
    )

def ai_says(text):
    st.markdown(f"""
    <div class="ai-says">
        <div class="ai-says-icon">⚡</div>
        <div>
            <div class="ai-says-title">RailPulse AI says</div>
            <div class="ai-says-text">{text}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def alt_card(alt):
    sc_cls = "score-h" if alt["score"] >= 80 else "score-m" if alt["score"] >= 65 else "score-l"
    ab_cls = "b-n" if alt["wl"] == 0 else "b-a"
    st.markdown(f"""
    <div class="alt-card">
        <div style="flex:1;">
            <div style="font-family:var(--font-display);font-size:14px;font-weight:600;color:#f1f5f9;margin-bottom:3px;">
                {alt['name']} <span style="font-size:11px;color:#475569;">#{alt['id']}</span>
            </div>
            <div style="font-size:12px;color:#94a3b8;">{alt['dep']} → {alt['arr']} &nbsp;·&nbsp; {alt['dur']} &nbsp;·&nbsp; {alt['cls']}</div>
        </div>
        <div style="display:flex;align-items:center;gap:14px;flex-shrink:0;">
            <span class="badge {ab_cls}">{alt['avail']}</span>
            <div style="text-align:right;">
                <div style="font-family:var(--font-display);font-size:14px;font-weight:700;color:#f1f5f9;">{alt['fare']}</div>
                <div style="font-size:10px;color:#475569;">per person</div>
            </div>
            <div style="text-align:center;min-width:52px;">
                <div class="{sc_cls}">{alt['score']}</div>
                <div style="font-size:10px;color:#475569;">score</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def alert_row(icon, train, msg, time_str):
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
        <span style="font-size:14px;margin-top:1px;">{icon}</span>
        <div style="flex:1;">
            <div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:2px;">{train}</div>
            <div style="font-size:12px;color:#94a3b8;">{msg}</div>
        </div>
        <div style="font-size:11px;color:#475569;white-space:nowrap;">{time_str}</div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PAGE: LOGIN
# ═════════════════════════════════════════════════════════════════════════════
def page_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, cm, _ = st.columns([1, 1.1, 1])
    with cm:
        st.markdown("""
        <div class="auth-wrap">
            <div class="auth-logo"><span style="color:#00ff88;">⚡</span> RailPulse AI</div>
            <div class="auth-tag">India's first AI-powered waitlist prediction engine.<br>
                Know your confirmation probability <em>before</em> you book.</div>
            <div class="nd"></div>
            <div class="auth-title">Sign in</div>
        </div>""", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="max-width:420px;margin:0 auto;">', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Sign In", use_container_width=True):
                    db = ss("users_db")
                    if username in db and db[username]["password"] == password:
                        ss_set("logged_in", True)
                        ss_set("username", username)
                        navigate("Dashboard")
                    else:
                        st.error("Invalid credentials")
            with c2:
                if st.button("Register →", use_container_width=True):
                    navigate("Register")
            st.markdown("""
            <div style="margin-top:1rem;padding:10px 14px;background:rgba(0,255,136,0.06);
                 border:1px solid rgba(0,255,136,0.15);border-radius:8px;font-size:12px;color:#94a3b8;">
                Demo: <code style="color:#00ff88;">demo</code> /
                      <code style="color:#00ff88;">demo123</code>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PAGE: REGISTER
# ═════════════════════════════════════════════════════════════════════════════
def page_register():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, cm, _ = st.columns([1, 1.1, 1])
    with cm:
        st.markdown("""
        <div style="max-width:420px;margin:0 auto;background:#0e0e1a;border:1px solid rgba(255,255,255,0.07);
             border-radius:20px;padding:2.5rem;box-shadow:0 24px 64px rgba(0,0,0,.6);">
            <div class="auth-logo"><span style="color:#00ff88;">⚡</span> RailPulse AI</div>
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
                    elif r_user in ss("users_db"):
                        st.error("Username taken.")
                    else:
                        db = ss("users_db")
                        db[r_user] = {
                            "password": r_pass, "name": r_name,
                            "email": r_email, "phone": r_phone or "Not provided",
                            "joined": datetime.now().strftime("%b %Y"),
                            "bookings": 0, "saved": 0,
                        }
                        ss_set("users_db", db)
                        st.success("Account created! Signing you in…")
                        time.sleep(0.8)
                        navigate("Login")
            with c2:
                if st.button("← Back", use_container_width=True):
                    navigate("Login")
            st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
NAV = [
    ("🏠", "Dashboard"),
    ("⚡", "Smart Predictor"),
    ("🔍", "PNR Tracker"),
    ("🗺", "Route Intelligence"),
    ("👤", "Profile"),
]

def render_sidebar():
    with st.sidebar:
        user     = ss("users_db")[ss("username")]
        initials = "".join(w[0] for w in user["name"].split()[:2]).upper()
        conf     = ss("dash", {}).get("model_conf", 84.3)
        cur      = ss("page", "Dashboard")
        predictor = get_predictor()

        st.markdown('<div class="sb-brand"><span class="sb-pulse"></span> RailPulse AI</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;background:rgba(0,255,136,0.06);
             border:1px solid rgba(0,255,136,0.14);border-radius:10px;padding:10px 12px;margin-bottom:1.5rem;">
            <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#00ff88,#38bdf8);
                 display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#000;flex-shrink:0;">{initials}</div>
            <div>
                <div style="font-size:13px;font-weight:600;color:#f1f5f9;">{user['name']}</div>
                <div style="font-size:11px;color:#94a3b8;">{user['email']}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        for icon, key in NAV:
            if key == "Smart Predictor":
                st.markdown('<div class="nav-hint">⚡ start here</div>', unsafe_allow_html=True)
            if st.button(f"{icon}  {key}", key=f"nav_{key}", use_container_width=True):
                navigate(key)

        idx = next((i for i, (_, k) in enumerate(NAV) if k == cur), 0) + 1
        st.markdown(f"""<style>
        section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type({idx}) button {{
            background: rgba(0,255,136,0.10) !important;
            border-color: rgba(0,255,136,0.30) !important;
            color: #00ff88 !important;
            font-weight: 600 !important;
        }}
        </style>""", unsafe_allow_html=True)

        st.markdown('<div class="nd"></div>', unsafe_allow_html=True)

        model_name = predictor.model_name
        src_label  = "Live Model" if model_name != "Heuristic" else "Heuristic"
        bar_color  = "#00ff88" if conf >= 85 else "#f59e0b" if conf >= 70 else "#f43f5e"
        st.markdown(f"""
        <div style="background:rgba(0,255,136,0.04);border:1px solid rgba(0,255,136,0.14);
             border-radius:11px;padding:11px 13px;margin-bottom:1rem;">
            <div style="font-size:9.5px;color:var(--th);letter-spacing:1.1px;
                 text-transform:uppercase;margin-bottom:7px;font-family:var(--font-body);">
                Model Accuracy
            </div>
            <div style="font-family:var(--font-display);font-size:24px;font-weight:700;
                 color:{bar_color};letter-spacing:-0.5px;">{conf}%</div>
            <div style="height:3px;background:var(--bg5);border-radius:2px;
                 margin-top:9px;overflow:hidden;">
                <div style="height:100%;width:{conf}%;
                     background:linear-gradient(90deg,{bar_color},var(--cyber));
                     border-radius:2px;"></div>
            </div>
            <div style="font-size:10px;color:var(--th);margin-top:5px;">
                {model_name} · {src_label}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-size:10px;color:var(--th);text-transform:uppercase;'
                    'letter-spacing:.9px;margin-bottom:5px;">Quick PNR Check</div>',
                    unsafe_allow_html=True)
        qpnr = st.text_input("", placeholder="Enter PNR…", key="quick_pnr",
                             label_visibility="collapsed", max_chars=10)
        if st.button("Track →", key="quick_pnr_go", use_container_width=True):
            if qpnr.strip():
                ss_set("pnr_queried", qpnr.strip())
                navigate("PNR Tracker")

        st.markdown('<div class="nd"></div>', unsafe_allow_html=True)
        if st.button("Sign Out", key="signout", use_container_width=True):
            ss_set("logged_in", False)
            ss_set("username", "")
            navigate("Dashboard")


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — PAGE: DASHBOARD (real model stats + per-train analysis)
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    d         = ss("dash")
    predictor = get_predictor()

    today_str = date.today().strftime("%d %b %Y")
    src_badge = ("🤖 Live Model" if d.get("model_source") == "real_model"
                 else "⚙️ Seed Data")
    best_mdl  = d.get("best_model_name", "—")

    page_header("Executive", "Dashboard",
                f"System-wide overview · AI performance · {today_str}")

    # Model source indicator
    if d.get("model_source") == "real_model":
        insight(
            f'✅ <strong>Model Active:</strong> Dashboard stats derived from '
            f'<strong>{best_mdl}</strong> · Accuracy: <strong>{d["model_conf"]}%</strong> · '
            f'{d.get("n_trains", 0):,} trains in training data. '
            f'Confirmation rates computed live by the model for WL/5, 30 days ahead.'
        )
    else:
        err = d.get("load_error", "")
        if err:
            st.error(f"**Model load failed** — {err}", icon="🔴")
        else:
            insight(
                '⚙️ <strong>Seed Data:</strong> Model file not found at '
                f'<code>outputs/best_model_real.joblib</code>. '
                'Run <code>python ml/_pipeline.py</code> from the project root, then restart.'
            )

    # ── KPIs — values from model, deltas relative to model stats ────────────
    n_trains = d.get("n_trains", 0)
    wl_rate  = round(d["wl_confirmed"] / max(d["bookings_today"], 1) * 100, 1)
    kpi_grid(
        ("Total Bookings Today", f"{d['bookings_today']:,}",
         f"Based on {n_trains:,} trained trains", "up"),
        ("WL Confirmed Today",   f"{d['wl_confirmed']:,}",
         f"{wl_rate}% confirmation rate", "up"),
        ("Active Waitlists",     str(d["active_wl"]),
         f"{round(d['active_wl']/max(d['bookings_today'],1)*100,1)}% of bookings", "down"),
        ("Model Accuracy",       f"{d['model_conf']}%",
         f"Best: {best_mdl}", "up"),
    )

    # ── Train selector ────────────────────────────────────────────────────────
    card_open("cyber")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:.75rem;">
        <span style="font-size:16px;">🚆</span>
        <div style="font-family:var(--font-display);font-size:14px;font-weight:600;color:#f1f5f9;">
            Train-Specific Analysis
        </div>
        <span class="badge b-c">Live Model</span>
    </div>
    <div style="font-size:12px;color:#94a3b8;margin-bottom:.75rem;">
        Select any train — the model will compute its 14-day booking trend and
        per-class confirmation rates in real time.
    </div>""", unsafe_allow_html=True)

    # Load real train list for the selector
    _, train_map, _ = _load_real_trains()
    # Build reverse map: number → display string
    num_to_display = {v: k for k, v in train_map.items() if k != "__routes__"}

    # Curated popular trains as defaults, then full list
    popular_ids = ["11007","12027","12123","11301","12301","12951","12259","12002","12004","22691"]
    popular     = [num_to_display[i] for i in popular_ids if i in num_to_display]
    all_trains  = sorted(k for k in train_map.keys() if k != "__routes__")
    train_opts  = popular + ["─── All trains ───"] + \
                  [t for t in all_trains if t not in popular]

    sel_train = st.selectbox(
        "Select train for analysis",
        train_opts,
        index=0,
        label_visibility="collapsed",
        key="dash_train_sel",
    )
    card_close()

    # ── Per-train charts (only if a real train selected) ─────────────────────
    if sel_train and "───" not in sel_train:
        sel_id   = train_map.get(sel_train, sel_train.split(" — ")[0])
        sel_name = sel_train.split(" — ")[1] if " — " in sel_train else sel_train

        with st.spinner(f"Computing model predictions for {sel_name}…"):
            fig_trend, bookings, wl_counts = chart_train_booking_trend(
                sel_id, sel_name, predictor)
            fig_cls, cls_rates = chart_train_class_rates(sel_id, predictor)

        tc1, tc2 = st.columns([3, 2])
        with tc1:
            card_open("cyber")
            section_header(f"Booking Trend — {sel_name}",
                           "14-day model-estimated demand · probability-scaled")
            st.plotly_chart(fig_trend, use_container_width=True)
            # Mini stats row
            avg_b = int(sum(bookings) / 14)
            avg_w = int(sum(wl_counts) / 14)
            peak  = max(bookings)
            st.markdown(
                f'<div style="display:flex;gap:20px;margin-top:.5rem;">'
                f'<div style="font-size:12px;color:#94a3b8;">Avg daily bookings: '
                f'<span style="color:#38bdf8;font-weight:600;">{avg_b:,}</span></div>'
                f'<div style="font-size:12px;color:#94a3b8;">Avg WL: '
                f'<span style="color:#f59e0b;font-weight:600;">{avg_w}</span></div>'
                f'<div style="font-size:12px;color:#94a3b8;">Peak: '
                f'<span style="color:#f1f5f9;font-weight:600;">{peak:,}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            card_close()
        with tc2:
            card_open("neon")
            section_header(f"Confirmation Rate by Class",
                           f"{sel_name} · WL/5 · 30 days ahead · live model")
            st.plotly_chart(fig_cls, use_container_width=True)
            best_cls = ["SL","3A","2A","1A","CC","2S"][cls_rates.index(max(cls_rates))]
            worst_cls= ["SL","3A","2A","1A","CC","2S"][cls_rates.index(min(cls_rates))]
            st.markdown(
                f'<div style="font-size:12px;color:#94a3b8;margin-top:.25rem;">'
                f'Best class: <span style="color:#00ff88;font-weight:600;">{best_cls} ({max(cls_rates)}%)</span>'
                f' &nbsp;·&nbsp; '
                f'Lowest: <span style="color:#f59e0b;font-weight:600;">{worst_cls} ({min(cls_rates)}%)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            card_close()
    else:
        # No train selected — show system-wide charts
        c1, c2 = st.columns([3, 2])
        with c1:
            card_open("cyber")
            section_header("Booking Trend (14 days)", "System-wide · Total bookings vs waitlisted")
            st.plotly_chart(chart_booking_trend(d["bookings_trend"], d["wl_trend"]),
                            use_container_width=True)
            card_close()
        with c2:
            card_open("neon")
            section_header("Confirmation Rate by Class",
                           "System-wide · WL/5 · 30 days · model estimates" if d.get("model_source")=="real_model"
                           else "WL/1–15 historical clearance")
            st.plotly_chart(chart_class_rates(d["conf_rates"]), use_container_width=True)
            card_close()

    # ── System health + model accuracy ───────────────────────────────────────
    h1, h2 = st.columns(2)
    with h1:
        card_open()
        section_header("System Components")
        for i, svc in enumerate(HEALTH_SVCS):
            health_bar(svc, d["health"][i])
        card_close()
    with h2:
        card_open("neon")
        section_header("Model Accuracy Progression",
                       f"{best_mdl} · {d.get('n_trains',0):,} training records")
        st.plotly_chart(chart_model_accuracy(d["model_acc"]), use_container_width=True)
        badge_row(
            badge(src_badge, "n" if d.get("model_source")=="real_model" else "a"),
            badge(f"{d.get('n_trains',0):,} trains", "c"),
            badge(f"Acc: {d['model_conf']}%", "n"),
        )
        card_close()

    # ── Top routes from model ─────────────────────────────────────────────────
    # Show top 5 high-demand routes based on model predictions
    card_open()
    section_header("High-Demand Routes", "Model-estimated · WL/10 at 14 days ahead")
    rng_r = random.Random(SEED + 7)
    for src, dst in ROUTES[:5]:
        r_res = predictor.predict(train_id="11007", coach_type="SL",
                                  wl_no=10, booking_day=14, season="peak")
        prob_pct = r_res["probability_pct"]
        col = "#00ff88" if prob_pct >= 60 else "#f59e0b" if prob_pct >= 40 else "#f43f5e"
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<div style="font-size:13px;color:var(--ts);">{src} → {dst}</div>'
            f'<div style="font-family:var(--font-display);font-size:14px;font-weight:700;color:{col};">'
            f'{prob_pct:.0f}% confirm</div></div>',
            unsafe_allow_html=True,
        )
    card_close()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — PAGE: PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
_ALT_TRAINS = [
    {"id":"12027","name":"Mumbai Shatabdi","dep":"07:30","arr":"10:35","dur":"3h 05m","cls":"CC","fare":"₹1,285","wl":0,"avail":"AVL 12","score":94},
    {"id":"11007","name":"Deccan Express",  "dep":"05:15","arr":"08:50","dur":"3h 35m","cls":"3A","fare":"₹760",  "wl":0,"avail":"AVL 6", "score":81},
    {"id":"12123","name":"Deccan Queen",    "dep":"14:45","arr":"19:15","dur":"4h 20m","cls":"3A","fare":"₹695",  "wl":4,"avail":"WL 4",  "score":67},
    {"id":"11301","name":"Indrayani Exp",   "dep":"17:30","arr":"21:10","dur":"3h 40m","cls":"SL","fare":"₹285",  "wl":0,"avail":"AVL 32","score":71},
]


def _ai_says_text(prob, wl, days, cls, train_name):
    base = CLASS_META.get(cls, {}).get("base_confirm", 65)
    name = train_name.split("—")[1].strip() if "—" in train_name else train_name
    if prob >= 75:
        return (f"This waitlist is looking <strong>very healthy</strong>. WL/{wl} on the {name} "
                f"typically clears <strong>{max(1,days-1)} to {days} days before departure</strong>. "
                f"Out of every 100 similar bookings, <strong>{int(prob)} confirmed</strong>. I'd book it.")
    elif prob >= 45:
        return (f"This is a coin-toss situation. {cls} class clears "
                f"<strong>{base}% of WL/1–10</strong>, but WL/{wl} is deeper than ideal. "
                f"Check again in 48 hours — if it hasn't moved, shift to a confirmed alternative.")
    else:
        return (f"I wouldn't count on this one. WL/{wl} with only {days} day(s) left means the "
                f"waitlist needs to clear fast — and historically only "
                f"<strong>{int(prob)}% of these confirm</strong>. Alternatives below have confirmed seats.")


def page_predictor():
    page_header("Smart", "Predictor",
                "Enter your journey — get AI-powered confirmation probability with full explanation")
    predictor = get_predictor()

    # Show which model is active
    model_badge = predictor.model_name
    badge_col   = "n" if model_badge != "Heuristic" else "a"
    st.markdown(
        f'<div style="margin-bottom:1rem;">'
        f'<span class="badge b-{badge_col}">🤖 {model_badge}</span>'
        f'<span style="font-size:12px;color:#475569;margin-left:8px;">'
        f'{"Real ML model active — predictions use your trained Gradient Boosting pipeline" if model_badge != "Heuristic" else "Heuristic fallback — run ml/_pipeline.py to load real model"}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # ── Load real data ────────────────────────────────────────────────────────
    with st.spinner("Loading station & train data…"):
        station_list, station_map     = _load_real_stations()
        train_list,   train_map, route_index = _load_real_trains()

    # ── Step 1: FROM / TO with search ────────────────────────────────────────
    card_open("cyber")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
        <div style="width:22px;height:22px;border-radius:50%;background:var(--cyber);
             display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#000;">1</div>
        <div style="font-family:var(--font-display);font-size:14px;font-weight:600;color:#f1f5f9;">
            Your Journey
        </div>
        <span style="font-size:11px;color:#475569;">— type to search stations</span>
    </div>""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown('<div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;">FROM</div>',
                    unsafe_allow_html=True)
        from_search = st.text_input("from_search", placeholder="Search station… e.g. PUNE or Mumbai",
                                    label_visibility="collapsed", key="from_search_box")
        from_filtered = ([s for s in station_list
                          if from_search.upper() in s.upper()]
                         if from_search.strip() else station_list)
        if not from_filtered:
            from_filtered = station_list
        default_from = next((i for i, s in enumerate(from_filtered) if s.startswith("PUNE")), 0)
        from_stn = st.selectbox("FROM station", from_filtered,
                                index=min(default_from, len(from_filtered)-1),
                                label_visibility="collapsed", key="pred_from")

    with fc2:
        st.markdown('<div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;">TO</div>',
                    unsafe_allow_html=True)
        to_search = st.text_input("to_search", placeholder="Search station… e.g. CSMT or Delhi",
                                  label_visibility="collapsed", key="to_search_box")
        to_filtered = ([s for s in station_list
                        if to_search.upper() in s.upper()]
                       if to_search.strip() else station_list)
        if not to_filtered:
            to_filtered = station_list
        default_to = next((i for i, s in enumerate(to_filtered)
                           if s.startswith("CSMT") or s.startswith("BCT")), 0)
        to_stn = st.selectbox("TO station", to_filtered,
                              index=min(default_to, len(to_filtered)-1),
                              label_visibility="collapsed", key="pred_to")

    with fc3:
        travel_date = st.date_input("TRAVEL DATE", value=date.today() + timedelta(days=7))

    # ── Step 2: Filter trains by selected route ───────────────────────────────
    from_code = station_map.get(from_stn, from_stn.split(" — ")[0] if " — " in from_stn else from_stn)
    to_code   = station_map.get(to_stn,   to_stn.split(" — ")[0]   if " — " in to_stn   else to_stn)

    route_trains = _get_trains_for_route(from_code, to_code, train_map, route_index)
    n_route      = len(route_trains)
    is_filtered  = n_route < len([k for k in train_map if k != "__routes__"])

    st.markdown(
        f'<div style="font-size:12px;margin:6px 0 10px;color:#94a3b8;">'
        f'{"🚆 " + str(n_route) + " trains found for " + from_code + " → " + to_code if is_filtered else "🚆 Showing all trains — select FROM and TO to filter"}'
        f'</div>',
        unsafe_allow_html=True,
    )

    fc4, fc5, fc6, fc7 = st.columns(4)
    with fc4:
        st.markdown('<div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;">TRAIN</div>',
                    unsafe_allow_html=True)
        # Search within the route-filtered trains
        train_search = st.text_input("train_search",
                                     placeholder="Search by name or number…",
                                     label_visibility="collapsed", key="train_search_box")
        train_filtered = ([t for t in route_trains
                           if train_search.upper() in t.upper()]
                          if train_search.strip() else route_trains)
        if not train_filtered:
            train_filtered = route_trains

        default_train = next((i for i, t in enumerate(train_filtered) if "11007" in t), 0)
        train_sel = st.selectbox("TRAIN", train_filtered,
                                 index=min(default_train, len(train_filtered)-1),
                                 label_visibility="collapsed", key="pred_train")

    with fc5:
        cls_sel = st.selectbox("CLASS", list(CLASS_META.keys()),
                               format_func=lambda k: f"{k} — {CLASS_META[k]['label']}")
    with fc6:
        wl_no = st.number_input("WAITLIST NUMBER  (0 = no WL)",
                                min_value=0, max_value=100, value=8)
    with fc7:
        pax = st.number_input("PASSENGERS", min_value=1, max_value=6, value=1)

    # ── Step 3: Extra model inputs (season, day) ──────────────────────────────
    with st.expander("⚙️  Advanced options — season & travel day"):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            season = st.selectbox("SEASON", ["normal","peak","off-peak"],
                                  format_func=lambda s: {"normal":"Normal","peak":"Peak (holidays/festivals)","off-peak":"Off-peak"}[s],
                                  key="pred_season")
        with ac2:
            day_of_week = st.selectbox("DAY OF WEEK",
                                       list(range(7)),
                                       format_func=lambda d: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][d],
                                       index=travel_date.weekday(),
                                       key="pred_dow")
        with ac3:
            cancel_rate = st.slider("HIST. CANCELLATION RATE", 0.03, 0.40, 0.15, 0.01,
                                    key="pred_cancel",
                                    help="Historical cancellation rate for this route (0.15 = 15%)")

    clicked = st.button("⚡  Predict My Chances", use_container_width=True)
    card_close()

    if clicked:
        tid  = train_map.get(train_sel, train_sel.split(" — ")[0] if " — " in train_sel else train_sel)
        days = max(1, (travel_date - date.today()).days)
        bar  = st.progress(0)
        stat = st.empty()
        for pct, msg in [(20,"Scanning route cancellation history…"),
                         (50,"Running prediction model…"),
                         (80,"Calculating confidence…"),
                         (100,"Done.")]:
            stat.markdown(f'<div style="text-align:center;padding:.5rem;font-size:12px;color:#94a3b8;">{msg}</div>',
                          unsafe_allow_html=True)
            bar.progress(pct)
            time.sleep(0.28)
        bar.empty(); stat.empty()

        # ── Call model with all real inputs ───────────────────────────────────
        result = predictor.predict(
            train_id         = tid,
            coach_type       = cls_sel,
            wl_no            = wl_no,
            booking_day      = days,
            season           = season,
            day_of_week      = day_of_week,
            cancellation_rate= cancel_rate,
            quota_available  = max(0, 30 - wl_no),   # estimate: quota shrinks as WL grows
            is_holiday_week  = 1 if season == "peak" else 0,
        )
        ss_set("pred_result", {
            "prob":        result["probability_pct"],
            "raw":         result["probability"],
            "train":       train_sel,
            "tid":         tid,
            "cls":         cls_sel,
            "wl":          wl_no,
            "from":        from_stn,
            "to":          to_stn,
            "from_code":   from_code,
            "to_code":     to_code,
            "date":        travel_date,
            "days":        days,
            "season":      season,
            "fare":        CLASS_META[cls_sel]["fare"],
            "pax":         pax,
            "model_used":  result["model_used"],
            "ml_alts":     result["alternatives"],
            "train_info":  result.get("train_info", {}),
        })
        st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    res = ss("pred_result")
    if not res:
        return

    prob = res["prob"]; days = res["days"]; wl = res["wl"]; tid = res["tid"]
    eta  = (date.today() + timedelta(days=max(1, days - 2))).strftime("%d %b %Y")

    if prob >= 75:
        v_color, v_label, v_glass = "#00ff88", "Likely to Confirm",           "neon"
        badge_v, rec_color        = "b-n",      "#00ff88"
    elif prob >= 45:
        v_color, v_label, v_glass = "#f59e0b", "Uncertain — Act Wisely",       "amber"
        badge_v, rec_color        = "b-a",      "#f59e0b"
    else:
        v_color, v_label, v_glass = "#f43f5e", "High Risk — Book Alternative", "red"
        badge_v, rec_color        = "b-r",      "#f43f5e"

    # Hero result card
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    card_open(v_glass)

    # Show train info from model if available
    tinfo = res.get("train_info", {})
    if tinfo.get("name"):
        st.markdown(
            f'<div style="font-size:12px;color:#94a3b8;text-align:center;margin-bottom:.5rem;">'
            f'🚆 {tinfo["name"]} · {tinfo.get("type","")} · '
            f'{int(tinfo.get("distance",0)):,} km · '
            f'{res["from"].split("—")[0].strip()} → {res["to"].split("—")[0].strip()}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"""
    <div class="hero-verdict">
        <div class="hero-pct" style="color:{v_color};">{prob:.0f}%</div>
        <div class="hero-sublabel" style="color:{v_color};">{v_label}</div>
        <div class="hero-eta">{"Expected to confirm by " + eta if wl > 0 else "Seat directly available — no waitlist"}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;margin:.5rem 0;">'
        f'<span class="badge {badge_v}">{res["model_used"]}</span>'
        f'<span style="font-size:11px;color:#475569;margin-left:8px;">'
        f'Season: {res.get("season","normal")} · {days} day(s) ahead · WL/{wl}'
        f'</span></div>',
        unsafe_allow_html=True,
    )
    card_close()

    # Recommendation
    if prob >= 75:
        rec_head = "Go ahead and book this ticket"
        rec_body = (f"Based on historical patterns, WL/{wl} on this route confirms "
                    f"<strong>{prob:.0f}% of the time</strong> with {days} day(s) to departure.")
        cta = "🎫 Book This Ticket"
    elif prob >= 45:
        rec_head = "Moderate chance — keep an eye on it"
        rec_body = (f"WL/{wl} has about a <strong>{prob:.0f}% chance</strong> of clearing. "
                    f"If it hasn't moved by <strong>{eta}</strong>, switch to a confirmed alternative.")
        cta = "🎫 Book WL Ticket"
    else:
        rec_head = "Book an alternative now"
        rec_body = (f"WL/{wl} has only a <strong>{prob:.0f}% chance</strong>. Confirmed seats available on other trains.")
        cta = "🔍 See Confirmed Alternatives"

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
         border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:.75rem;">
        <div style="font-size:14px;font-weight:700;color:{rec_color};margin-bottom:5px;">
            {"✅" if prob>=75 else "⚠️" if prob>=45 else "❌"} {rec_head}
        </div>
        <div style="font-size:13px;color:#94a3b8;line-height:1.65;">{rec_body}</div>
    </div>""", unsafe_allow_html=True)

    if res.get("ml_alts"):
        for alt_txt in res["ml_alts"]:
            st.markdown(f'<div style="font-size:12px;color:#94a3b8;padding:3px 0;">💡 {alt_txt}</div>',
                        unsafe_allow_html=True)

    cb1, cb2 = st.columns([2, 1])
    with cb1:
        if st.button(cta, use_container_width=True, key="cta_primary"):
            st.success("Booking flow initiated!")
    with cb2:
        if st.button("🔔 Set Alert", use_container_width=True, key="set_alert"):
            ss_set("alert_set", True)
            st.success(f"Alert set — you'll be notified if WL/{wl} confidence drops below 50%.")

    # AI Explanation
    with st.expander("🧠  Why this score? — AI Explanation", expanded=True):
        ai_says(_ai_says_text(prob, wl, days, res["cls"], res["train"]))
        cyber_divider()
        base      = CLASS_META.get(res["cls"], {}).get("base_confirm", 65)
        wl_safety = max(5, 100 - wl * 3)
        day_adv   = min(100, days * 7)
        prob_bar("How often this class confirms", base, f"{base}/100", "#38bdf8",
                 f"{res['cls']} class confirmed in {base}% of past similar WL/1–10 bookings")
        prob_bar("How safe your waitlist position is", wl_safety, f"{wl_safety}/100", "#00ff88",
                 "WL/" + str(wl) + " — " + ("very safe, top of the list" if wl<=5 else
                                              "moderate position" if wl<=12 else "deep position, higher risk"))
        prob_bar("Time advantage until travel", day_adv, f"{day_adv}/100", "#f59e0b",
                 f"{days} day(s) left — " + ("plenty of time for cancellations" if days>=7 else
                                               "limited time" if days>=3 else "very tight window"))

    # What-If Simulator
    with st.expander("🔬  What-If Simulator — explore other positions"):
        sim_wl  = st.slider("Waitlist position", 1, 40, int(wl), step=1, key="wif_slider")
        sim_res = predictor.predict(
            train_id=tid, coach_type=res["cls"], wl_no=sim_wl,
            booking_day=days, season=res.get("season","normal"),
            quota_available=max(0, 30 - sim_wl),
        )
        sim_prob = sim_res["probability_pct"]
        sim_col  = "#00ff88" if sim_prob>=75 else "#f59e0b" if sim_prob>=45 else "#f43f5e"
        delta    = sim_prob - prob
        d_col    = "#00ff88" if delta>0 else "#f43f5e" if delta<0 else "#94a3b8"
        d_txt    = f"↑ +{delta:.1f}%" if delta>0 else f"↓ {delta:.1f}%" if delta<0 else "no change"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:24px;padding:1rem;
             background:rgba(255,255,255,0.03);border-radius:12px;margin-top:.5rem;">
            <div style="text-align:center;min-width:90px;">
                <div style="font-size:10px;color:#475569;text-transform:uppercase;
                     letter-spacing:.8px;margin-bottom:4px;">At WL/{sim_wl}</div>
                <div style="font-family:var(--font-display);font-size:38px;
                     font-weight:800;color:{sim_col};line-height:1;">{sim_prob:.0f}%</div>
            </div>
            <div style="flex:1;">
                <div style="height:10px;background:rgba(255,255,255,0.06);
                     border-radius:5px;overflow:hidden;">
                    <div style="height:100%;width:{sim_prob}%;background:{sim_col};border-radius:5px;"></div>
                </div>
                <div style="font-size:12px;color:{d_col};margin-top:6px;font-weight:600;">
                    {d_txt} vs your current WL/{wl}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        snap_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:.75rem;">'
        for snap_wl in [2, 5, 8, 12, 18, 25]:
            sp     = predictor.predict(train_id=tid, coach_type=res["cls"],
                                       wl_no=snap_wl, booking_day=days,
                                       season=res.get("season","normal"),
                                       quota_available=max(0, 30-snap_wl))
            sp_pct = sp["probability_pct"]
            sc     = "#00ff88" if sp_pct>=75 else "#f59e0b" if sp_pct>=45 else "#f43f5e"
            you    = " (you)" if snap_wl == wl else ""
            bright = "0.18" if snap_wl == wl else "0.06"
            snap_html += (
                f'<div style="background:rgba(255,255,255,0.04);border:1px solid '
                f'rgba(255,255,255,{bright});border-radius:8px;padding:8px 12px;text-align:center;">'
                f'<div style="font-size:10px;color:#475569;">WL/{snap_wl}{you}</div>'
                f'<div style="font-family:var(--font-display);font-size:16px;'
                f'font-weight:700;color:{sc};margin-top:2px;">{sp_pct:.0f}%</div></div>'
            )
        snap_html += '</div>'
        st.markdown(snap_html, unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#475569;margin-top:.75rem;">'
                    '🟢 &gt;75% — Book with confidence &nbsp;·&nbsp; '
                    '🟡 45–75% — Consider alternatives &nbsp;·&nbsp; '
                    '🔴 &lt;45% — High risk</div>', unsafe_allow_html=True)

    # Gauge
    with st.expander("📊  Detailed Probability Gauge"):
        gc1, gc2 = st.columns([1, 1])
        with gc1:
            st.plotly_chart(chart_gauge(prob), use_container_width=True)
        with gc2:
            info_grid(
                ("Train · Class · WL",
                 f"{res['train'][:45]} · {res['cls']} · {'WL/'+str(wl) if wl>0 else 'Direct'}"),
                ("Route",
                 f"{res['from'].split('—')[0].strip()} → {res['to'].split('—')[0].strip()}"),
                ("Travel Date",
                 f"{res['date'].strftime('%d %b %Y')} · {days} day(s) away"),
                ("Season · Day",
                 f"{res.get('season','normal').title()} · "
                 f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][res['date'].weekday()]}"),
            )

    # Alternatives — filtered to same route
    with st.expander("🚆  Alternative Trains on this route",
                     expanded=(prob < 75)):
        if res.get("from_code") and res.get("to_code"):
            alt_route = _get_trains_for_route(
                res["from_code"], res["to_code"], train_map, route_index)
            # Show top 6 alternatives (excluding selected train)
            alts_to_show = [t for t in alt_route if tid not in t][:6]
            if alts_to_show:
                st.markdown(
                    f'<div style="font-size:12px;color:#94a3b8;margin-bottom:.75rem;">'
                    f'Showing {len(alts_to_show)} other trains on '
                    f'{res["from_code"]} → {res["to_code"]} · '
                    f'ranked by route order</div>',
                    unsafe_allow_html=True,
                )
                for alt_display in alts_to_show:
                    alt_num  = train_map.get(alt_display, "")
                    alt_name = alt_display.split(" — ")[1] if " — " in alt_display else alt_display
                    alt_res  = predictor.predict(
                        train_id=alt_num, coach_type=res["cls"],
                        wl_no=wl, booking_day=days,
                        season=res.get("season","normal"),
                        quota_available=max(0, 30-wl),
                    )
                    alt_prob = alt_res["probability_pct"]
                    alt_col  = "#00ff88" if alt_prob>=75 else "#f59e0b" if alt_prob>=45 else "#f43f5e"
                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;'
                        f'padding:10px 14px;background:rgba(255,255,255,0.03);'
                        f'border:1px solid rgba(255,255,255,0.06);border-radius:10px;margin-bottom:6px;">'
                        f'<div>'
                        f'<div style="font-size:13px;font-weight:600;color:#f1f5f9;">{alt_name}</div>'
                        f'<div style="font-size:11px;color:#475569;">#{alt_num} · {res["cls"]}</div>'
                        f'</div>'
                        f'<div style="font-family:var(--font-display);font-size:20px;'
                        f'font-weight:800;color:{alt_col};">{alt_prob:.0f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                for alt in _ALT_TRAINS:
                    alt_card(alt)
        else:
            for alt in _ALT_TRAINS:
                alt_card(alt)
    base = CLASS_META.get(cls, {}).get("base_confirm", 65)
    name = train_name.split("—")[1].strip() if "—" in train_name else train_name
    if prob >= 75:
        return (f"This waitlist is looking <strong>very healthy</strong>. WL/{wl} on the {name} "
                f"typically clears <strong>{max(1,days-1)} to {days} days before departure</strong>. "
                f"Out of every 100 similar bookings, <strong>{int(prob)} confirmed</strong>. I'd book it.")
    elif prob >= 45:
        return (f"This is a coin-toss situation. {cls} class clears "
                f"<strong>{base}% of WL/1–10</strong>, but WL/{wl} is deeper than ideal. "
                f"Check again in 48 hours — if it hasn't moved, shift to a confirmed alternative.")
    else:
        return (f"I wouldn't count on this one. WL/{wl} with only {days} day(s) left means the "
                f"waitlist needs to clear fast — and historically only "
                f"<strong>{int(prob)}% of these confirm</strong>. Alternatives below have confirmed seats.")


def page_predictor():
    page_header("Smart", "Predictor",
                "Enter your journey — get AI-powered confirmation probability with full explanation")
    predictor = get_predictor()

    # ── Load real stations & trains from data/ ────────────────────────────────
    with st.spinner("Loading station & train data…"):
        station_list, station_map = _load_real_stations()
        train_list,   train_map, route_index = _load_real_trains()

    # ── Step 1: Input form ────────────────────────────────────────────────────
    card_open("cyber")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
        <div style="width:22px;height:22px;border-radius:50%;background:var(--cyber);
             display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#000;">1</div>
        <div style="font-family:var(--font-display);font-size:14px;font-weight:600;color:#f1f5f9;">Your Journey</div>
    </div>""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        default_from = next((i for i, s in enumerate(station_list) if s.startswith("PUNE")), 0)
        from_stn = st.selectbox("FROM", station_list, index=default_from, key="pred_from")
    with fc2:
        default_to = next((i for i, s in enumerate(station_list)
                           if s.startswith("CSMT") or s.startswith("BCT")), 0)
        to_stn = st.selectbox("TO", station_list, index=default_to, key="pred_to")
    with fc3:
        travel_date = st.date_input("TRAVEL DATE", value=date(2026, 3, 25))

    fc4, fc5, fc6, fc7 = st.columns(4)
    with fc4:
        default_train = next((i for i, t in enumerate(train_list) if "11007" in t), 0)
        train_sel = st.selectbox("TRAIN", train_list, index=default_train, key="pred_train")
    with fc5:
        cls_sel = st.selectbox("CLASS", list(CLASS_META.keys()),
                               format_func=lambda k: f"{k} — {CLASS_META[k]['label']}")
    with fc6:
        wl_no = st.number_input("WAITLIST NUMBER  (0 = no WL)", min_value=0, max_value=100, value=8)
    with fc7:
        pax = st.number_input("PASSENGERS", min_value=1, max_value=6, value=1)

    clicked = st.button("⚡  Predict My Chances", use_container_width=True)
    card_close()

    if clicked:
        tid  = train_map.get(train_sel, train_sel.split(" — ")[0])
        days = max(1, (travel_date - date.today()).days)
        bar  = st.progress(0)
        stat = st.empty()
        for pct, msg in [(20,"Scanning route cancellation history…"),
                         (50,"Running prediction model…"),
                         (80,"Calculating confidence…"),
                         (100,"Done.")]:
            stat.markdown(f'<div style="text-align:center;padding:.5rem;font-size:12px;color:#94a3b8;">{msg}</div>',
                          unsafe_allow_html=True)
            bar.progress(pct)
            time.sleep(0.28)
        bar.empty(); stat.empty()

        result = predictor.predict(train_id=tid, coach_type=cls_sel,
                                   wl_no=wl_no, booking_day=days, season="normal")
        ss_set("pred_result", {
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
    res = ss("pred_result")
    if not res:
        return

    prob = res["prob"]; days = res["days"]; wl = res["wl"]; tid = res["tid"]
    eta  = (date.today() + timedelta(days=max(1, days - 2))).strftime("%d %b %Y")

    if prob >= 75:
        v_color, v_label, v_glass = "#00ff88", "Likely to Confirm",           "neon"
        badge_v, rec_color        = "b-n",      "#00ff88"
    elif prob >= 45:
        v_color, v_label, v_glass = "#f59e0b", "Uncertain — Act Wisely",       "amber"
        badge_v, rec_color        = "b-a",      "#f59e0b"
    else:
        v_color, v_label, v_glass = "#f43f5e", "High Risk — Book Alternative", "red"
        badge_v, rec_color        = "b-r",      "#f43f5e"

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    card_open(v_glass)
    st.markdown(f"""
    <div class="hero-verdict">
        <div class="hero-pct" style="color:{v_color};">{prob:.0f}%</div>
        <div class="hero-sublabel" style="color:{v_color};">{v_label}</div>
        <div class="hero-eta">{"Expected to confirm by " + eta if wl > 0 else "Seat directly available — no waitlist"}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;margin:.5rem 0;"><span class="badge {badge_v}">{res["model_used"]}</span></div>',
                unsafe_allow_html=True)
    card_close()

    if prob >= 75:
        rec_head = "Go ahead and book this ticket"
        rec_body = (f"Based on your trained <strong>{predictor.model_name}</strong> model, "
                    f"WL/{wl} on this route confirms "
                    f"<strong>{prob:.0f}% of the time</strong> with {days} day(s) to departure.")
        cta = "🎫 Book This Ticket"
    elif prob >= 45:
        rec_head = "Moderate chance — keep an eye on it"
        rec_body = (f"WL/{wl} has about a <strong>{prob:.0f}% chance</strong> of clearing. "
                    f"If it hasn't moved by <strong>{eta}</strong>, switch to a confirmed alternative.")
        cta = "🎫 Book WL Ticket"
    else:
        rec_head = "Book an alternative now"
        rec_body = (f"WL/{wl} has only a <strong>{prob:.0f}% chance</strong>. Confirmed seats are available on other trains.")
        cta = "🔍 See Confirmed Alternatives"

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
         border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:.75rem;">
        <div style="font-size:14px;font-weight:700;color:{rec_color};margin-bottom:5px;">
            {"✅" if prob>=75 else "⚠️" if prob>=45 else "❌"} {rec_head}
        </div>
        <div style="font-size:13px;color:#94a3b8;line-height:1.65;">{rec_body}</div>
    </div>""", unsafe_allow_html=True)

    if res.get("ml_alts"):
        for alt_txt in res["ml_alts"]:
            st.markdown(f'<div style="font-size:12px;color:#94a3b8;padding:3px 0;">💡 {alt_txt}</div>',
                        unsafe_allow_html=True)

    cb1, cb2 = st.columns([2, 1])
    with cb1:
        if st.button(cta, use_container_width=True, key="cta_primary"):
            st.success("Booking flow initiated!")
    with cb2:
        if st.button("🔔 Set Alert", use_container_width=True, key="set_alert"):
            ss_set("alert_set", True)
            st.success(f"Alert set — you'll be notified if WL/{wl} confidence drops below 50%.")

    with st.expander("🧠  Why this score? — AI Explanation", expanded=True):
        ai_says(_ai_says_text(prob, wl, days, res["cls"], res["train"]))
        cyber_divider()
        base      = CLASS_META.get(res["cls"], {}).get("base_confirm", 65)
        wl_safety = max(5, 100 - wl * 3)
        day_adv   = min(100, days * 7)
        prob_bar("How often this class confirms", base, f"{base}/100", "#38bdf8",
                 f"{res['cls']} class confirmed in {base}% of past similar WL/1–10 bookings")
        prob_bar("How safe your waitlist position is", wl_safety, f"{wl_safety}/100", "#00ff88",
                 "WL/" + str(wl) + " — " + ("very safe, top of the list" if wl<=5 else
                                              "moderate position" if wl<=12 else "deep position, higher risk"))
        prob_bar("Time advantage until travel", day_adv, f"{day_adv}/100", "#f59e0b",
                 f"{days} day(s) left — " + ("plenty of time for cancellations" if days>=7 else
                                               "limited time" if days>=3 else "very tight window"))

    with st.expander("🔬  What-If Simulator — explore other positions"):
        sim_wl   = st.slider("Waitlist position", 1, 40, int(wl), step=1, key="wif_slider")
        sim_res  = predictor.predict(train_id=tid, coach_type=res["cls"],
                                     wl_no=sim_wl, booking_day=days)
        sim_prob = sim_res["probability_pct"]
        sim_col  = "#00ff88" if sim_prob>=75 else "#f59e0b" if sim_prob>=45 else "#f43f5e"
        delta    = sim_prob - prob
        d_col    = "#00ff88" if delta>0 else "#f43f5e" if delta<0 else "#94a3b8"
        d_txt    = f"↑ +{delta:.1f}%" if delta>0 else f"↓ {delta:.1f}%" if delta<0 else "no change"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:24px;padding:1rem;background:rgba(255,255,255,0.03);border-radius:12px;margin-top:.5rem;">
            <div style="text-align:center;min-width:90px;">
                <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;">At WL/{sim_wl}</div>
                <div style="font-family:var(--font-display);font-size:38px;font-weight:800;color:{sim_col};line-height:1;">{sim_prob:.0f}%</div>
            </div>
            <div style="flex:1;">
                <div style="height:10px;background:rgba(255,255,255,0.06);border-radius:5px;overflow:hidden;">
                    <div style="height:100%;width:{sim_prob}%;background:{sim_col};border-radius:5px;"></div>
                </div>
                <div style="font-size:12px;color:{d_col};margin-top:6px;font-weight:600;">{d_txt} vs your current WL/{wl}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        snap_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:.75rem;">'
        for snap_wl in [2, 5, 8, 12, 18, 25]:
            sp     = predictor.predict(train_id=tid, coach_type=res["cls"],
                                       wl_no=snap_wl, booking_day=days)
            sp_pct = sp["probability_pct"]
            sc     = "#00ff88" if sp_pct>=75 else "#f59e0b" if sp_pct>=45 else "#f43f5e"
            you    = " (you)" if snap_wl == wl else ""
            bright = "0.18" if snap_wl == wl else "0.06"
            snap_html += (
                f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,{bright});'
                f'border-radius:8px;padding:8px 12px;text-align:center;">'
                f'<div style="font-size:10px;color:#475569;">WL/{snap_wl}{you}</div>'
                f'<div style="font-family:var(--font-display);font-size:16px;font-weight:700;color:{sc};margin-top:2px;">{sp_pct:.0f}%</div></div>'
            )
        snap_html += '</div>'
        st.markdown(snap_html, unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#475569;margin-top:.75rem;">'
                    '🟢 &gt;75% — Book with confidence &nbsp;·&nbsp; '
                    '🟡 45–75% — Consider alternatives &nbsp;·&nbsp; '
                    '🔴 &lt;45% — High risk</div>', unsafe_allow_html=True)

    with st.expander("📊  Detailed Probability Gauge"):
        gc1, gc2 = st.columns([1, 1])
        with gc1:
            st.plotly_chart(chart_gauge(prob), use_container_width=True)
        with gc2:
            info_grid(
                ("Train · Class · WL",
                 f"{res['train'][:45]} · {res['cls']} · {'WL/'+str(wl) if wl>0 else 'Direct'}"),
                ("Route",
                 f"{res['from'].split('—')[0].strip()} → {res['to'].split('—')[0].strip()}"),
                ("Travel Date",
                 f"{res['date'].strftime('%d %b %Y')} · {days} day(s) away"),
                ("Expected Confirmation",
                 eta if wl > 0 else "Already available"),
            )

    with st.expander("🚆  Alternative Trains — confirmed options on this route",
                     expanded=(prob < 75)):
        for alt in _ALT_TRAINS:
            alt_card(alt)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13 — PAGE: PNR TRACKER
# ═════════════════════════════════════════════════════════════════════════════
def page_pnr():
    page_header("PNR", "Tracker", "Complete journey timeline from booking to arrival")

    card_open()
    pc1, pc2 = st.columns([5, 1])
    with pc1:
        pnr_in = st.text_input("", placeholder="Enter 10-digit PNR  (try: 6230185420  or  5119274831)",
                               value=ss("pnr_queried", "6230185420"),
                               label_visibility="collapsed", max_chars=10, key="pnr_field")
    with pc2:
        if st.button("🔍  Track", use_container_width=True, key="pnr_track"):
            if pnr_in.strip():
                ss_set("pnr_queried", pnr_in.strip())
    card_close()

    pnr = PNR_DEMO.get(ss("pnr_queried", "6230185420"))
    if not pnr:
        st.markdown("""
        <div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.2);
             border-radius:12px;padding:1rem;font-size:13px;color:#f43f5e;">
            PNR not found. Try <code>6230185420</code> or <code>5119274831</code>.
        </div>""", unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.25rem;">
        <div style="font-size:13px;color:#94a3b8;">PNR:
            <span style="font-family:var(--font-mono);color:#f1f5f9;">{ss('pnr_queried')}</span>
        </div>
        <span class="badge b-g">DEMO</span><span class="badge b-n">CNF</span>
    </div>""", unsafe_allow_html=True)

    pr1, pr2 = st.columns([3, 1])
    with pr1:
        card_open("neon")
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:1.25rem;">
            <div>
                <div style="font-family:var(--font-display);font-size:16px;font-weight:700;color:#f1f5f9;">{pnr['train']}</div>
                <div style="font-size:12px;color:#94a3b8;margin-top:3px;">{pnr['route']} · {pnr['date']}</div>
            </div>
            <div style="display:flex;gap:8px;">
                <span style="background:rgba(59,130,246,.15);color:#60a5fa;font-size:11px;font-weight:500;padding:3px 10px;border-radius:6px;">{pnr['cls']}</span>
                <span style="background:#1c1c2e;color:#94a3b8;font-size:11px;font-weight:500;padding:3px 10px;border-radius:6px;">Coach {pnr['coach']}</span>
                <span style="background:#1c1c2e;color:#94a3b8;font-size:11px;font-weight:500;padding:3px 10px;border-radius:6px;">Berth {pnr['berth']}</span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;font-weight:500;color:#94a3b8;margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.6px;">Waitlist Position Journey</div>', unsafe_allow_html=True)
        wl_sparkline(pnr["wl_history"])
        cyber_divider()
        st.markdown('<div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:1rem;">Journey Timeline</div>', unsafe_allow_html=True)
        timeline(pnr["timeline"])
        card_close()

    with pr2:
        card_open()
        init = "".join(w[0] for w in pnr["passenger"].split()[:2]).upper()
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:1rem;">
            <div style="width:54px;height:54px;border-radius:50%;background:linear-gradient(135deg,#00ff88,#38bdf8);display:flex;align-items:center;justify-content:center;font-size:17px;font-weight:800;color:#000;margin:0 auto 8px;">{init}</div>
            <div style="font-size:14px;font-weight:600;color:#f1f5f9;">{pnr['passenger']}</div>
            <div style="font-size:12px;color:#94a3b8;margin-top:3px;">{pnr['fare']}</div>
            <span class="badge b-n" style="margin-top:8px;">CNF</span>
        </div>""", unsafe_allow_html=True)
        neon_divider()
        st.button("📥 Download e-Ticket", use_container_width=True, key="dl_t")
        st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
        st.button("🔄 Find Alternatives",  use_container_width=True, key="fa_t")
        st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
        st.button("❌ Cancel & Refund",    use_container_width=True, key="cr_t")
        card_close()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 14 — PAGE: ROUTE INTELLIGENCE (real model data, no fake API)
# ═════════════════════════════════════════════════════════════════════════════
def page_heatmap():
    d         = ss("dash")
    predictor = get_predictor()

    page_header("Route", "Intelligence",
                "Model-computed demand · WL pressure by corridor · based on your training data")

    # ── Compute real per-route stats from the model ───────────────────────────
    # For each route in ROUTES, run predictor at WL/10, 14 days, normal season
    # → demand proxy = confirmation probability (higher prob = higher demand train)
    @st.cache_data(show_spinner=False)
    def _route_stats_from_model():
        rows = []
        for src, dst in ROUTES:
            # Find a train on this route from the model's train_data
            pl = getattr(get_predictor(), "_payload", None)
            train_id = "11007"  # fallback
            if pl is not None:
                tf_ = pl.get("train_data")
                if tf_ is not None:
                    mask = (
                        tf_["from_station_name"].str.contains(src,  case=False, na=False) &
                        tf_["to_station_name"].str.contains(dst,    case=False, na=False)
                    )
                    if mask.any():
                        train_id = str(tf_[mask].iloc[0]["train_number"])

            res_high = predictor.predict(train_id=train_id, coach_type="SL",
                                         wl_no=10, booking_day=14, season="peak")
            res_low  = predictor.predict(train_id=train_id, coach_type="SL",
                                         wl_no=3,  booking_day=30, season="normal")
            # demand = inverse of confirmation at high WL → harder to confirm = more demand
            demand  = int((1 - res_high["probability"]) * 100)
            wl_avg  = int(10 + (1 - res_low["probability"]) * 25)
            rows.append({
                "source":  src,  "dest":   dst,
                "demand":  max(demand, 5),
                "wl_avg":  max(wl_avg, 2),
                "route":   f"{src} → {dst}",
                "model_prob_high_wl": round(res_high["probability_pct"], 1),
                "model_prob_low_wl":  round(res_low["probability_pct"],  1),
            })
        return pd.DataFrame(rows)

    with st.spinner("Computing route demand from model…"):
        df = _route_stats_from_model()

    # KPIs derived from real model output
    top_route  = df.loc[df["demand"].idxmax()]
    peak_wl    = df["wl_avg"].max()
    n_routes   = len(df)
    low_conf   = df[df["model_prob_high_wl"] < 50]

    kpi_grid(
        ("Highest Demand Route",  f"{top_route['source']} → {top_route['dest']}",
         f"Demand index: {top_route['demand']}", "up"),
        ("Peak Avg Waitlist",     f"WL / {peak_wl}",
         df.loc[df["wl_avg"].idxmax(), "route"], "flat"),
        ("Routes Analysed",       str(n_routes),
         "From your trains.json data", "up"),
        ("Low Confirm Routes",    str(len(low_conf)),
         "< 50% confirm at WL/10", "down" if len(low_conf) > 5 else "flat"),
    )

    fig_heat, _ = chart_heatmap(df["demand"].tolist(), df["wl_avg"].tolist(), ROUTES)

    hc1, hc2 = st.columns([3, 2])
    with hc1:
        card_open("cyber")
        section_header("Demand Treemap",
                       "Block size = demand index · Color = avg WL pressure · computed by model")
        st.plotly_chart(fig_heat, use_container_width=True)
        card_close()
    with hc2:
        card_open()
        section_header("Demand vs Waitlist",
                       "Each bubble = one route · model-estimated values")
        st.plotly_chart(chart_scatter_demand(df), use_container_width=True)
        card_close()

    # Full table with model probabilities
    card_open()
    section_header("All Routes — Model Analysis",
                   "Ranked by demand · confirmation probabilities at WL/10 and WL/3")
    display_df = df[["route","demand","wl_avg","model_prob_high_wl","model_prob_low_wl"]].copy()
    display_df = display_df.sort_values("demand", ascending=False).reset_index(drop=True)
    display_df.index += 1
    display_df.columns = ["Route","Demand Index","Avg WL","Confirm % at WL/10","Confirm % at WL/3"]
    st.dataframe(
        display_df.style
            .background_gradient(subset=["Demand Index"],        cmap="YlOrRd")
            .background_gradient(subset=["Confirm % at WL/10"],  cmap="RdYlGn")
            .background_gradient(subset=["Confirm % at WL/3"],   cmap="YlGn"),
        use_container_width=True,
    )
    st.markdown(
        '<div style="font-size:11px;color:#475569;margin-top:.5rem;">'
        '⚡ All values computed live by your trained model · '
        f'Best model: {predictor.model_name}'
        '</div>',
        unsafe_allow_html=True,
    )
    card_close()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 15 — PAGE: PROFILE
# ═════════════════════════════════════════════════════════════════════════════
def page_profile():
    user     = ss("users_db")[ss("username")]
    initials = "".join(w[0] for w in user["name"].split()[:2]).upper()
    page_header("My", "Profile", "Account details · Booking history · Preferences")

    pp1, pp2 = st.columns([1, 2])
    with pp1:
        card_open("neon")
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="pf-avatar">{initials}</div>
            <div class="pf-name">{user['name']}</div>
            <div class="pf-sub">{user['email']}</div>
            <div style="margin:1rem 0;"><span class="badge b-n">Verified Member</span></div>
            <div class="nd"></div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div class="pf-stat"><div class="pf-sv">{user['bookings']}</div><div class="pf-sl">Bookings</div></div>
                <div class="pf-stat"><div class="pf-sv">{user['saved']}</div><div class="pf-sl">Saved Routes</div></div>
            </div>
            <div style="margin-top:1rem;padding:10px;background:rgba(0,255,136,0.05);border-radius:8px;border:1px solid rgba(0,255,136,0.12);">
                <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.6px;">Member Since</div>
                <div style="font-family:var(--font-display);font-size:16px;font-weight:700;color:#f1f5f9;margin-top:4px;">{user['joined']}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        card_close()

    with pp2:
        card_open()
        section_header("Account Details")
        for label, val in [("Full Name", user["name"]), ("Email", user["email"]),
                           ("Phone", user.get("phone","Not provided")), ("Joined", user["joined"])]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
                <div style="font-size:13px;font-weight:500;color:#f1f5f9;">{val}</div>
            </div>""", unsafe_allow_html=True)

        neon_divider()
        section_header("Edit Profile")
        new_name  = st.text_input("Full Name", value=user["name"],         key="ep_name")
        new_phone = st.text_input("Phone",     value=user.get("phone",""), key="ep_phone")
        new_email = st.text_input("Email",     value=user["email"],        key="ep_email")
        if st.button("💾  Save Changes", use_container_width=True, key="save_p"):
            db = ss("users_db")
            db[ss("username")].update({"name": new_name, "phone": new_phone, "email": new_email})
            ss_set("users_db", db)
            st.success("✅ Profile updated.")
        card_close()

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        card_open()
        section_header("Recent Bookings")
        for pnr_no, train, route, dt, fare, status, bc in [
            ("6230185420","Deccan Express #11007","Pune → Mumbai","18 Mar 2026","₹760","CNF","b-n"),
            ("5119274831","Mumbai Shatabdi #12027","Mumbai → Pune","12 Mar 2026","₹635","CNF","b-n"),
            ("4208163720","Deccan Queen #12123",  "Pune → Mumbai","05 Mar 2026","₹255","CAN","b-r"),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                <div>
                    <div style="font-size:13px;font-weight:600;color:#f1f5f9;">{train}</div>
                    <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{route} · {dt} · <span style="font-family:var(--font-mono);color:#475569;">{pnr_no}</span></div>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <span class="badge {bc}">{status}</span>
                    <div style="font-family:var(--font-display);font-size:14px;font-weight:700;color:#f1f5f9;">{fare}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        if ss("alert_set"):
            st.markdown("""
            <div style="margin-top:1rem;padding:10px 14px;background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.2);border-radius:8px;font-size:12px;color:#94a3b8;">
                🔔 <strong style="color:#00ff88;">Smart Alert Active</strong> — You'll be notified if your WL confidence drops below 50%
            </div>""", unsafe_allow_html=True)
        card_close()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 17 — MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    _session_defaults()
    _seed_dashboard()
    st.markdown(CSS, unsafe_allow_html=True)

    if not ss("logged_in"):
        p = ss("page")
        if p == "Register":
            page_register()
        else:
            page_login()
        return

    render_sidebar()
    p = ss("page")

    if   p == "Dashboard":          page_dashboard()
    elif p == "Smart Predictor":    page_predictor()
    elif p == "PNR Tracker":        page_pnr()
    elif p == "Route Intelligence": page_heatmap()
    elif p == "Profile":            page_profile()
    else:                           page_dashboard()


if __name__ == "__main__":
    main()
