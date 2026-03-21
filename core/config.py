"""
core/config.py
Single source of truth for ALL constants, paths, and static data.
Import from here — never hardcode values elsewhere.
"""

import os
from pathlib import Path

# ── Paths (pathlib + os.path for compatibility) ───────────────────────────────
ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# Legacy string paths kept for any code that uses os.path style
BASE_DIR   = str(ROOT)
DATA_DIR   = str(DATA)
OUTPUT_DIR = str(OUTPUTS)

STATIONS_PATH  = DATA / "stations.json"
TRAINS_PATH    = DATA / "trains.json"
SCHEDULES_PATH = DATA / "schedules.json"
MODEL_PATH     = OUTPUTS / "best_model_real.joblib"

OUTPUTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# ── Training hyperparameters ──────────────────────────────────────────────────
SEED       = 42
N_BOOKINGS = 30_000
TEST_SIZE  = 0.20
CV_FOLDS   = 3

# ── Domain constants ──────────────────────────────────────────────────────────
COACH_TYPES = ["1A", "2A", "3A", "SL", "CC", "2S"]
SEASONS     = ["peak", "normal", "off-peak"]

COACH_PROBS = [0.05, 0.10, 0.25, 0.38, 0.12, 0.10]

COACH_BONUS = {"1A": 0.30, "2A": 0.22, "CC": 0.12,
               "3A": 0.05, "SL": 0.00, "2S": -0.08}
SEASON_PEN  = {"off-peak": 0.20, "normal": 0.00, "peak": -0.25}
TYPE_BONUS  = {
    "Raj": 0.20, "Drnt": 0.18, "JShtb": 0.15, "Shtb": 0.12, "SKr": 0.12,
    "SF":  0.08, "Mail": 0.06, "Exp":   0.03,  "Pass": 0.00,
    "MEMU": -0.03, "DEMU": -0.03, "GR": -0.05, "Toy": -0.08,
    "Hyd": 0.05, "Del": 0.10, "Klkt": 0.12, "Unknown": 0.0, "": 0.0,
}

COACH_CAPACITY = {
    "first_ac": 24, "second_ac": 46, "third_ac": 64,
    "sleeper":  72, "chair_car": 78, "first_class": 18,
}

# ── UI class metadata ─────────────────────────────────────────────────────────
CLASS_META = {
    "SL": {"label": "Sleeper",     "base_confirm": 72, "fare": 285},
    "3A": {"label": "3 Tier AC",   "base_confirm": 65, "fare": 760},
    "2A": {"label": "2 Tier AC",   "base_confirm": 58, "fare": 1145},
    "1A": {"label": "First AC",    "base_confirm": 85, "fare": 2120},
    "CC": {"label": "Chair Car",   "base_confirm": 78, "fare": 635},
    "2S": {"label": "2nd Sitting", "base_confirm": 55, "fare": 140},
    "EC": {"label": "Exec Chair",  "base_confirm": 60, "fare": 1285},
}

# ── Demo train table (used when no real data file exists) ─────────────────────
DEMO_TRAINS = [
    {"id": "11007", "name": "Deccan Express",    "type": "EXPRESS",
     "dep": "05:15", "arr": "08:50", "dur": "3h 35m",
     "dep_stn": "PUNE", "arr_stn": "CSMT",
     "on_time": 94, "cancel_rate": 18, "occupancy": 87, "price_tier": 2, "comfort": 3},
    {"id": "12027", "name": "Mumbai Shatabdi",   "type": "SHATABDI",
     "dep": "07:30", "arr": "10:35", "dur": "3h 05m",
     "dep_stn": "PUNE", "arr_stn": "CSMT",
     "on_time": 97, "cancel_rate": 8,  "occupancy": 92, "price_tier": 4, "comfort": 5},
    {"id": "12123", "name": "Deccan Queen",      "type": "SUPERFAST",
     "dep": "14:45", "arr": "19:15", "dur": "4h 20m",
     "dep_stn": "PUNE", "arr_stn": "CSMT",
     "on_time": 81, "cancel_rate": 14, "occupancy": 78, "price_tier": 2, "comfort": 3},
    {"id": "11301", "name": "Indrayani Express", "type": "EXPRESS",
     "dep": "17:30", "arr": "21:10", "dur": "3h 40m",
     "dep_stn": "PUNE", "arr_stn": "CSMT",
     "on_time": 76, "cancel_rate": 22, "occupancy": 65, "price_tier": 1, "comfort": 2},
    {"id": "12301", "name": "Rajdhani Express",  "type": "RAJDHANI",
     "dep": "16:35", "arr": "08:15", "dur": "15h 40m",
     "dep_stn": "PUNE", "arr_stn": "NDLS",
     "on_time": 91, "cancel_rate": 11, "occupancy": 95, "price_tier": 5, "comfort": 5},
]

# ── Demo PNR records ──────────────────────────────────────────────────────────
PNR_DEMO = {
    "6230185420": {
        "train": "Deccan Express #11007", "route": "Pune → Mumbai",
        "date": "18 Mar 2026", "cls": "3A", "coach": "B4", "berth": "32 LB",
        "passenger": "S. Kumar", "fare": "₹760",
        "timeline": [
            {"label": "Booking Initiated",  "time": "14 Mar 2026, 10:38", "state": "done",
             "note": "Payment received · PNR generated"},
            {"label": "Waitlisted (WL/14)", "time": "14 Mar 2026, 10:39", "state": "done",
             "note": "Initial position: WL/14"},
            {"label": "Moved to WL/8",      "time": "15 Mar 2026, 06:12", "state": "done",
             "note": "6 cancellations processed by allocation engine"},
            {"label": "Upgraded to RAC/4",  "time": "16 Mar 2026, 20:44", "state": "done",
             "note": "Dynamic Seat Allocation triggered"},
            {"label": "Confirmed — CNF",    "time": "17 Mar 2026, 22:14", "state": "done",
             "note": "Berth B4/32 LB assigned"},
            {"label": "Chart Prepared",     "time": "17 Mar 2026, 23:59", "state": "active",
             "note": "Final chart locked · Boarding confirmed"},
            {"label": "Train Departed",     "time": "18 Mar 2026, 05:15", "state": "pending",
             "note": "Platform 2 · On time"},
            {"label": "Journey Complete",   "time": "18 Mar 2026, 08:50", "state": "pending",
             "note": "CSMT Mumbai"},
        ],
        "wl_history": [14, 14, 12, 10, 8, 8, 6, 4, "RAC", "CNF"],
    },
    "5119274831": {
        "train": "Mumbai Shatabdi #12027", "route": "Mumbai → Pune",
        "date": "12 Mar 2026", "cls": "CC", "coach": "C2", "berth": "45",
        "passenger": "R. Sharma", "fare": "₹635",
        "timeline": [
            {"label": "Booking Confirmed", "time": "08 Mar 2026, 14:20", "state": "done",
             "note": "Direct confirmation — no waitlist"},
            {"label": "Chart Prepared",    "time": "11 Mar 2026, 22:00", "state": "done",
             "note": "Seat C2/45 confirmed"},
            {"label": "Train Departed",    "time": "12 Mar 2026, 06:25", "state": "done",
             "note": "Departed on time"},
            {"label": "Journey Complete",  "time": "12 Mar 2026, 09:30", "state": "done",
             "note": "Arrived — journey complete ✓"},
        ],
        "wl_history": ["CNF", "CNF", "CNF", "CNF"],
    },
}

# ── Heatmap route list ────────────────────────────────────────────────────────
ROUTES = [
    ("Mumbai", "Delhi"),    ("Mumbai", "Pune"),     ("Delhi", "Kolkata"),
    ("Mumbai", "Ahmedabad"),("Bangalore", "Chennai"),("Delhi", "Jaipur"),
    ("Hyderabad", "Chennai"),("Kolkata", "Patna"),   ("Delhi", "Lucknow"),
    ("Mumbai", "Goa"),      ("Chennai", "Bangalore"),("Pune", "Nashik"),
    ("Agra", "Delhi"),      ("Nagpur", "Mumbai"),    ("Bhopal", "Delhi"),
]

# ── Dashboard service names ───────────────────────────────────────────────────
HEALTH_SVCS = [
    "Prediction Engine", "Booking API", "Seat Alloc. Engine",
    "PNR Sync Service",  "Database Cluster",
]

# ── ML chart palette ──────────────────────────────────────────────────────────
MODEL_PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":       "#55A868",
    "Gradient Boosting":   "#C44E52",
    "Stacking Ensemble":   "#FF8C00",
}

# ── App metadata ──────────────────────────────────────────────────────────────
APP_TITLE     = "RailPulse AI"
APP_ICON      = "⚡"
MODEL_VERSION = "v2.0"
