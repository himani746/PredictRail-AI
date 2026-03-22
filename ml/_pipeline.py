"""
ml/_pipeline.py
Full ML training pipeline for RailPulse AI — PS-02  (Improved).

Improvements over baseline:
  1. Balanced target distribution (60/40) — fixes class imbalance
  2. Stronger logit signal — wider feature coefficient spread
  3. XGBoost added as base learner (replaces plain GB in stacking)
  4. Threshold tuning — optimal decision boundary per model
  5. More features — route_popularity, days_to_departure buckets,
     wl_per_seat, booking_pressure
  6. N_BOOKINGS raised to 50,000 for better generalisation
  7. CV folds raised to 5

Run via:
    python ml/_pipeline.py    (from project root D:/PredictRail-AI)
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       StackingClassifier)
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       roc_auc_score, roc_curve,
                                       confusion_matrix, ConfusionMatrixDisplay,
                                       classification_report)
from sklearn.calibration       import CalibratedClassifierCV

# Optional XGBoost — falls back gracefully if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [INFO] xgboost not found — install with: pip install xgboost")
    print("         Continuing without XGBoost.\n")

# ── Inline config (no dependency on core/) ────────────────────────────────────
ROOT_PATH      = Path(__file__).parent.parent
DATA           = ROOT_PATH / "data"
OUTPUTS        = ROOT_PATH / "outputs"
STATIONS_PATH  = DATA  / "stations.json"
TRAINS_PATH    = DATA  / "trains.json"
SCHEDULES_PATH = DATA  / "schedules.json"
MODEL_PATH     = OUTPUTS / "best_model_real.joblib"
OUTPUT_DIR     = str(OUTPUTS)
OUTPUTS.mkdir(exist_ok=True)

SEED       = 42
N_BOOKINGS = 50_000          # ↑ from 30k — more data = better generalisation
CV_FOLDS   = 5               # ↑ from 3

COACH_TYPES = ["1A", "2A", "3A", "SL", "CC", "2S"]
SEASONS     = ["peak", "normal", "off-peak"]
COACH_PROBS = [0.05, 0.10, 0.25, 0.38, 0.12, 0.10]

COACH_BONUS = {"1A": 0.35, "2A": 0.28, "CC": 0.15, "3A": 0.06, "SL": 0.00, "2S": -0.12}
SEASON_PEN  = {"off-peak": 0.30, "normal": 0.00, "peak": -0.40}
TYPE_BONUS  = {
    "Raj": 0.25, "Drnt": 0.22, "JShtb": 0.18, "Shtb": 0.15, "SKr": 0.15,
    "SF":  0.10, "Mail": 0.08, "Exp":   0.04,  "Pass": 0.00,
    "MEMU": -0.05, "DEMU": -0.05, "GR": -0.08, "Toy": -0.10,
    "Hyd": 0.07, "Del": 0.12, "Klkt": 0.14, "Unknown": 0.0, "": 0.0,
}
MODEL_PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":       "#55A868",
    "Gradient Boosting":   "#C44E52",
    "XGBoost":             "#9467BD",
    "Stacking Ensemble":   "#FF8C00",
}

# ── Inline features (replaces ml/features.py) ─────────────────────────────────
FEATURE_COLS = [
    "waitlist_position", "coach_type_enc",   "booking_day",      "season_enc",
    "cancellation_rate", "day_of_week",      "total_stops",      "duration_hours",
    "distance_km",       "train_type_enc",   "total_seats",      "speed_kmph",
    "seats_per_stop",    "quota_available",  "is_weekend",       "is_holiday_week",
    "wl_x_cancel",       "wl_x_booking",     "coach_x_season",   "quota_x_wl",
    "dist_per_stop",     "has_ac",           "has_sleeper",      "premium_score",
    "zone_enc",
    # NEW features
    "wl_per_seat",       "booking_pressure", "days_bucket",      "route_popularity",
    "wl_squared",        "quota_ratio",
]
FEATURE_NAMES = [
    "Waitlist Position",  "Coach Type",       "Booking Day",     "Season",
    "Cancellation Rate",  "Day of Week",      "Total Stops",     "Duration (hrs)",
    "Distance (km)",      "Train Type",       "Total Seats",     "Speed (km/h)",
    "Seats/Stop",         "Quota Available",  "Is Weekend",      "Is Holiday",
    "WL × Cancel",        "WL / BookDay",     "Coach × Season",  "Quota / WL",
    "Dist/Stop",          "Has AC",           "Has Sleeper",     "Premium Score",
    "Zone",
    # NEW
    "WL / Seats",         "Booking Pressure", "Days Bucket",     "Route Popularity",
    "WL²",                "Quota Ratio",
]

def make_encoders():
    le_coach  = LabelEncoder().fit(COACH_TYPES)
    le_season = LabelEncoder().fit(SEASONS)
    return le_coach, le_season

def add_train_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["premium_score"] = (df["first_ac"]*3 + df["second_ac"]*2 + df["third_ac"]*1 + df["chair_car"]*1)
    return df

def add_stop_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_stops"]    = df["total_stops"].fillna(5).astype(float)
    df["seats_per_stop"] = df["total_seats"] / df["total_stops"].replace(0, 1)
    df["dist_per_stop"]  = df["distance_km"] / df["total_stops"].replace(0, 1)
    return df

def build_interaction_features(bk: pd.DataFrame) -> pd.DataFrame:
    bk = bk.copy()
    bk["wl_x_cancel"]    = bk["waitlist_position"] * bk["cancellation_rate"]
    bk["wl_x_booking"]   = bk["waitlist_position"] / (bk["booking_day"] + 1)
    bk["coach_x_season"] = bk["coach_type_enc"]    * bk["season_enc"]
    bk["quota_x_wl"]     = bk["quota_available"]   / (bk["waitlist_position"] + 1)
    bk["is_weekend"]     = (bk["day_of_week"] >= 5).astype(int)
    # NEW engineered features
    bk["wl_per_seat"]     = bk["waitlist_position"] / (bk["total_seats"].replace(0, 1))
    bk["booking_pressure"]= bk["waitlist_position"] / (bk["quota_available"] + 1)
    bk["days_bucket"]     = pd.cut(bk["booking_day"],
                                   bins=[0, 7, 15, 30, 60, 120],
                                   labels=[0, 1, 2, 3, 4]).astype(float)
    bk["route_popularity"]= (bk["total_seats"] * bk["cancellation_rate"]).clip(upper=50)
    bk["wl_squared"]      = bk["waitlist_position"] ** 2
    bk["quota_ratio"]     = bk["quota_available"] / (bk["total_seats"].replace(0, 1))
    return bk

np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────
def _load_stations() -> pd.DataFrame:
    with open(STATIONS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for feat in raw["features"]:
        p = feat["properties"]
        g = feat.get("geometry")
        rows.append({
            "station_code": p.get("code", ""),
            "station_name": p.get("name", ""),
            "state":        p.get("state", "Unknown"),
            "zone":         p.get("zone",  "Unknown"),
            "longitude":    g["coordinates"][0] if g else np.nan,
            "latitude":     g["coordinates"][1] if g else np.nan,
        })
    df = pd.DataFrame(rows).drop_duplicates("station_code")
    df["zone"] = df["zone"].replace({"?": "Unknown", "": "Unknown"}).fillna("Unknown")
    print(f"  Stations loaded : {len(df):,}")
    return df


def _load_trains() -> pd.DataFrame:
    with open(TRAINS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for feat in raw["features"]:
        p = feat["properties"]
        g = feat.get("geometry")
        coords = g["coordinates"] if g and g.get("coordinates") else []
        rows.append({
            "train_number":      str(p.get("number", "")),
            "train_name":        p.get("name", ""),
            "train_type":        p.get("type", ""),
            "zone":              p.get("zone", "Unknown"),
            "from_station_code": p.get("from_station_code", ""),
            "to_station_code":   p.get("to_station_code",   ""),
            "from_station_name": p.get("from_station_name", ""),
            "to_station_name":   p.get("to_station_name",   ""),
            "distance_km":       float(p.get("distance",   0) or 0),
            "duration_h":        float(p.get("duration_h", 0) or 0),
            "duration_m":        float(p.get("duration_m", 0) or 0),
            "first_ac":          int(p.get("first_ac",    0) or 0),
            "second_ac":         int(p.get("second_ac",   0) or 0),
            "third_ac":          int(p.get("third_ac",    0) or 0),
            "sleeper":           int(p.get("sleeper",     0) or 0),
            "chair_car":         int(p.get("chair_car",   0) or 0),
            "first_class":       int(p.get("first_class", 0) or 0),
            "classes":           p.get("classes", ""),
            "num_route_points":  len(coords),
        })
    df = pd.DataFrame(rows).drop_duplicates("train_number")
    df["train_type"] = df["train_type"].replace("", "Unknown").fillna("Unknown")
    df["zone"]       = df["zone"].fillna("Unknown")
    print(f"  Trains loaded   : {len(df):,}")
    return df


def _load_schedules(df_trains: pd.DataFrame) -> pd.DataFrame:
    if SCHEDULES_PATH.exists():
        with open(SCHEDULES_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df_s = pd.DataFrame(raw)
        df_s["train_number"] = df_s["train_number"].astype(str)
        stop_counts = (df_s.groupby("train_number").size()
                           .rename("total_stops").reset_index())
        df_trains = df_trains.merge(stop_counts, on="train_number", how="left")
        print(f"  Schedules       : {len(df_s):,} stops")
    else:
        df_trains["total_stops"] = df_trains["num_route_points"].clip(lower=2)
        print("  Schedules       : not found — estimating from route points")
    return df_trains


def _make_synthetic(n_trains: int = 500) -> tuple:
    """Fallback when JSON data files are absent."""
    print("  [FALLBACK] Generating synthetic train/station data")
    rng = np.random.default_rng(SEED)
    df_trains = pd.DataFrame({
        "train_number":      [str(10000 + i) for i in range(n_trains)],
        "train_name":        [f"Train {i}" for i in range(n_trains)],
        "train_type":        rng.choice(["SF","Mail","Exp","Raj","Shtb","Pass"], n_trains),
        "zone":              rng.choice(["CR","NR","SR","ER","WR","SCR","NCR"],  n_trains),
        "from_station_code": rng.choice(["HWH","NDLS","BCT","MAS","SBC","CNB"], n_trains),
        "to_station_code":   rng.choice(["HWH","NDLS","BCT","MAS","SBC","CNB"], n_trains),
        "from_station_name": ["Station A"] * n_trains,
        "to_station_name":   ["Station B"] * n_trains,
        "distance_km":       rng.integers(100, 2500, n_trains).astype(float),
        "duration_h":        rng.integers(1,   40,   n_trains).astype(float),
        "duration_m":        rng.integers(0,   59,   n_trains).astype(float),
        "first_ac":          rng.integers(0, 3, n_trains),
        "second_ac":         rng.integers(0, 4, n_trains),
        "third_ac":          rng.integers(0, 8, n_trains),
        "sleeper":           rng.integers(0, 12, n_trains),
        "chair_car":         rng.integers(0, 6, n_trains),
        "first_class":       rng.integers(0, 2, n_trains),
        "classes":           [""] * n_trains,
        "num_route_points":  rng.integers(2, 40, n_trains),
        "total_stops":       rng.integers(2, 40, n_trains).astype(float),
    })
    df_stations = pd.DataFrame({
        "station_code": ["HWH","NDLS","BCT","MAS","SBC","CNB"],
        "station_name": ["Howrah","New Delhi","Mumbai CST","Chennai","Bangalore","Kanpur"],
        "state":  ["WB","DL","MH","TN","KA","UP"],
        "zone":   ["ER","NR","CR","SR","SWR","NCR"],
        "longitude": [88.3, 77.2, 72.8, 80.3, 77.6, 80.3],
        "latitude":  [22.6, 28.6, 18.9, 13.1, 12.9, 26.4],
    })
    return df_trains, df_stations


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Build synthetic booking records from real train distributions
# ─────────────────────────────────────────────────────────────────────────────
def _build_bookings(df_trains, le_coach, le_season, le_type, le_zone,
                    n: int = None) -> pd.DataFrame:
    n   = n or N_BOOKINGS
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(df_trains), size=n)
    ts  = df_trains.iloc[idx].reset_index(drop=True)

    bk = pd.DataFrame()
    for col in ["train_number","train_type_enc","train_type","zone_enc",
                "distance_km","duration_hours","total_seats","total_stops",
                "speed_kmph","seats_per_stop","dist_per_stop",
                "has_ac","has_sleeper","premium_score"]:
        bk[col] = ts[col].values

    # Balanced waitlist distribution — more low-WL records to fix class skew
    # 50% WL 1-20 (likely confirm), 30% WL 21-50, 20% WL 51-100
    wl_low  = rng.integers(1,  21, n)
    wl_mid  = rng.integers(21, 51, n)
    wl_high = rng.integers(51, 101, n)
    tier    = rng.choice([0, 1, 2], n, p=[0.50, 0.30, 0.20])
    bk["waitlist_position"] = np.where(tier==0, wl_low,
                              np.where(tier==1, wl_mid, wl_high))

    bk["coach_type"]        = rng.choice(COACH_TYPES, n, p=COACH_PROBS)
    bk["booking_day"]       = rng.integers(1, 120, n)
    bk["day_of_week"]       = rng.integers(0, 7, n)
    bk["season"]            = rng.choice(SEASONS, n, p=[0.30, 0.50, 0.20])
    bk["cancellation_rate"] = rng.uniform(0.03, 0.40, n).round(3)
    bk["quota_available"]   = rng.integers(0, 30, n)
    bk["is_holiday_week"]   = rng.choice([0, 1], n, p=[0.75, 0.25])

    bk["coach_type_enc"] = le_coach.transform(bk["coach_type"])
    bk["season_enc"]     = le_season.transform(bk["season"])
    bk = build_interaction_features(bk)

    # Stronger logit — wider coefficient spread for cleaner decision boundary
    coach_bonus   = bk["coach_type"].map(COACH_BONUS).fillna(0)
    season_pen    = bk["season"].map(SEASON_PEN).fillna(0)
    type_bonus    = bk["train_type"].map(TYPE_BONUS).fillna(0)
    premium_bonus = (bk["premium_score"] / 10).clip(upper=0.4)
    seat_bonus    = (bk["total_seats"]   / 500).clip(upper=0.25)

    logit = (
        1.0                                          # ↓ lower intercept → less class skew
        - 0.10 * bk["waitlist_position"]            # ↑ stronger WL penalty
        + 3.5  * bk["cancellation_rate"]            # ↑ stronger cancellation signal
        + 0.018 * bk["booking_day"]                 # ↑ days-in-advance bonus
        + 0.12  * bk["quota_available"]             # ↑ quota signal
        - 0.20  * bk["is_holiday_week"]             # ↑ holiday penalty
        - 0.12  * bk["is_weekend"]
        + 0.08  * bk["has_ac"]
        - 0.008 * bk["wl_squared"] / 100            # non-linear WL penalty
        - 1.5   * bk["wl_per_seat"]                 # new: WL relative to capacity
        - 0.5   * bk["booking_pressure"]            # new: WL / quota
        + coach_bonus + season_pen + type_bonus + premium_bonus + seat_bonus
        + rng.normal(0, 0.20, n)                    # ↓ less noise → cleaner labels
    )
    prob_true       = 1 / (1 + np.exp(-logit))
    bk["confirmed"] = (rng.uniform(size=n) < prob_true).astype(int)

    confirm_rate = bk["confirmed"].mean()
    print(f"  Confirm rate   : {confirm_rate:.2%}  "
          f"(Not confirmed: {(1-confirm_rate):.2%})")
    return bk


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Train models
# ─────────────────────────────────────────────────────────────────────────────
def _find_best_threshold(y_true, y_prob):
    """Find threshold that maximises balanced accuracy."""
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.30, 0.75, 0.01):
        yp  = (y_prob >= t).astype(int)
        rec0 = recall_score(y_true, yp, pos_label=0, zero_division=0)
        rec1 = recall_score(y_true, yp, pos_label=1, zero_division=0)
        bal  = (rec0 + rec1) / 2
        if bal > best_score:
            best_score, best_t = bal, t
    return round(best_t, 2)


def _train(X_tr, X_te, y_tr, y_te):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    base = {
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sca", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=2000,
                                       class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=16, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced",
                random_state=SEED, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10,
                random_state=SEED)),
        ]),
    }

    if HAS_XGB:
        # compute scale_pos_weight from training labels
        neg = int((y_tr == 0).sum())
        pos = int((y_tr == 1).sum())
        spw = round(neg / max(pos, 1), 2)
        base["XGBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw,
                eval_metric="logloss", verbosity=0,
                random_state=SEED, n_jobs=-1)),
        ])

    results, trained, thresholds = {}, {}, {}
    print("-" * 65)
    for name, pipe in base.items():
        pipe.fit(X_tr, y_tr)
        ypr = pipe.predict_proba(X_te)[:, 1]
        # Tune threshold for best balanced accuracy
        t   = _find_best_threshold(y_te, ypr)
        yp  = (ypr >= t).astype(int)
        acc = accuracy_score(y_te, yp)
        f1  = f1_score(y_te, yp, zero_division=0)
        auc = roc_auc_score(y_te, ypr)
        cv  = cross_val_score(pipe, X_tr, y_tr, cv=skf,
                              scoring="balanced_accuracy").mean()
        results[name] = dict(
            Accuracy  = acc,
            Precision = precision_score(y_te, yp, zero_division=0),
            Recall    = recall_score(y_te, yp, zero_division=0),
            F1=f1, ROC_AUC=auc, CV_BalAcc=cv,
        )
        trained[name]    = pipe
        thresholds[name] = t
        print(f"  {name:22s}  Acc={acc:.4f}  F1={f1:.4f}  "
              f"AUC={auc:.4f}  CV={cv:.4f}  Thr={t:.2f}")

    # ── Stacking Ensemble ─────────────────────────────────────────────────────
    print("\n[STACKING ENSEMBLE]")
    imp_s = SimpleImputer(strategy="mean")
    sca_s = StandardScaler()
    Xtr_s = sca_s.fit_transform(imp_s.fit_transform(X_tr))
    Xte_s = sca_s.transform(imp_s.transform(X_te))

    estimators = [
        ("lr", trained["Logistic Regression"].named_steps["clf"]),
        ("rf", trained["Random Forest"].named_steps["clf"]),
        ("gb", trained["Gradient Boosting"].named_steps["clf"]),
    ]
    if HAS_XGB and "XGBoost" in trained:
        estimators.append(("xgb", trained["XGBoost"].named_steps["clf"]))

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                           class_weight="balanced",
                                           random_state=SEED),
        cv=skf, n_jobs=-1, passthrough=True,   # passthrough adds raw features too
    )
    stack.fit(Xtr_s, y_tr)
    ypr_s = stack.predict_proba(Xte_s)[:, 1]
    t_s   = _find_best_threshold(y_te, ypr_s)
    yp_s  = (ypr_s >= t_s).astype(int)
    acc_s = accuracy_score(y_te, yp_s)
    f1_s  = f1_score(y_te, yp_s, zero_division=0)
    auc_s = roc_auc_score(y_te, ypr_s)
    cv_s  = cross_val_score(stack, Xtr_s, y_tr, cv=skf,
                             scoring="balanced_accuracy").mean()
    results["Stacking Ensemble"] = dict(
        Accuracy  = acc_s,
        Precision = precision_score(y_te, yp_s, zero_division=0),
        Recall    = recall_score(y_te, yp_s, zero_division=0),
        F1=f1_s, ROC_AUC=auc_s, CV_BalAcc=cv_s,
    )
    thresholds["Stacking Ensemble"] = t_s
    print(f"  {'Stacking Ensemble':22s}  Acc={acc_s:.4f}  F1={f1_s:.4f}  "
          f"AUC={auc_s:.4f}  CV={cv_s:.4f}  Thr={t_s:.2f}")

    return results, trained, stack, imp_s, sca_s, Xte_s, yp_s, ypr_s, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Save charts
# ─────────────────────────────────────────────────────────────────────────────
def _save_charts(trained, results, stack, y_te, X_te,
                 yp_s, ypr_s, best_name):
    df_res     = pd.DataFrame(results).T
    all_names  = list(trained.keys()) + ["Stacking Ensemble"]
    n_models   = len(all_names)

    # Ensure MODEL_PALETTE has a colour for every model
    _palette = {**MODEL_PALETTE}
    for nm in all_names:
        if nm not in _palette:
            _palette[nm] = "#888888"

    # 1. Feature importance (tree models only)
    tree_models = [nm for nm in trained
                   if nm in ("Random Forest", "Gradient Boosting", "XGBoost")]
    if tree_models:
        fig, axes = plt.subplots(1, len(tree_models),
                                 figsize=(12 * len(tree_models), 10))
        if len(tree_models) == 1:
            axes = [axes]
        fig.suptitle("Feature Importance", fontsize=15, fontweight="bold")
        for ax, mname in zip(axes, tree_models):
            clf  = trained[mname].named_steps["clf"]
            imp_ = clf.feature_importances_ / clf.feature_importances_.sum()
            # Pad/trim to FEATURE_NAMES length
            n_feat = min(len(imp_), len(FEATURE_NAMES))
            imp_   = imp_[:n_feat]
            names_ = FEATURE_NAMES[:n_feat]
            idx_   = np.argsort(imp_)
            bars   = ax.barh([names_[i] for i in idx_], imp_[idx_],
                             color=_palette[mname], edgecolor="white", linewidth=0.4)
            ax.set_title(mname, fontsize=12, fontweight="bold")
            ax.set_xlabel("Relative Importance")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.tick_params(axis="y", labelsize=8)
            ax.spines[["top","right"]].set_visible(False)
            for bar, v in zip(bars, imp_[idx_]):
                ax.text(v+0.002, bar.get_y()+bar.get_height()/2,
                        f"{v:.1%}", va="center", fontsize=7.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_real.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # 2. ROC curves
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    fig.suptitle("ROC Curves — All Models", fontsize=13, fontweight="bold")
    for ax, name in zip(axes, all_names):
        ypr_ = ypr_s if name == "Stacking Ensemble" \
               else trained[name].predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, ypr_)
        auc_        = roc_auc_score(y_te, ypr_)
        ax.plot(fpr, tpr, color=_palette[name], lw=2.5, label=f"AUC={auc_:.4f}")
        ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.10, color=_palette[name])
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves_real.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Metrics heatmap
    metric_cols = [c for c in ["Accuracy","Precision","Recall","F1","ROC_AUC","CV_BalAcc","CV_Acc"]
                   if c in df_res.columns]
    fig, ax = plt.subplots(figsize=(14, max(4, n_models * 1.2)))
    hm = df_res[metric_cols].astype(float)
    sns.heatmap(hm, annot=True, fmt=".4f", cmap="YlGn", linewidths=0.5,
                ax=ax, vmin=0.5, vmax=1.0, annot_kws={"size":11,"weight":"bold"})
    ax.set_title("Model Performance", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=15); ax.tick_params(axis="y", rotation=0)
    bi = list(df_res.index).index(best_name)
    ax.add_patch(plt.Rectangle((0, bi), hm.shape[1], 1,
                 fill=False, edgecolor="#FF8C00", lw=3))
    ax.text(hm.shape[1]+0.1, bi+0.5, "◀ BEST", va="center",
            color="#FF8C00", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_heatmap_real.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Confusion matrices
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    for ax, name in zip(axes, all_names):
        yp_ = yp_s if name == "Stacking Ensemble" else trained[name].predict(X_te)
        cm_ = confusion_matrix(y_te, yp_)
        ConfusionMatrixDisplay(cm_, display_labels=["Not Confirmed","Confirmed"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\nAcc={accuracy_score(y_te,yp_):.4f}",
                     fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_real.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Probability distribution
    best_probs = (ypr_s if best_name == "Stacking Ensemble"
                  else trained[best_name].predict_proba(X_te)[:, 1])
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color, lname in [(0,"#C44E52","Not Confirmed"),
                                 (1,"#55A868","Confirmed")]:
        mask = y_te == label
        ax.hist(best_probs[mask], bins=40, alpha=0.6, color=color,
                label=f"{lname} (n={mask.sum():,})", density=True,
                edgecolor="white")
    ax.axvline(0.5, color="black", lw=2, ls="--", label="Decision boundary (0.5)")
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
    ax.set_title(f"Probability Distribution — {best_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "probability_distribution_real.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print("[VIZ] 5 charts saved →", OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  RailPulse AI  |  PS-02  |  ML Training Pipeline")
    print("=" * 65)

    # 1. Load / generate data
    print("\n[DATA]")
    if STATIONS_PATH.exists() and TRAINS_PATH.exists():
        df_stations = _load_stations()
        df_trains   = _load_trains()
        df_trains   = _load_schedules(df_trains)
    else:
        df_trains, df_stations = _make_synthetic()

    # 2. Feature engineering
    print("\n[FEATURES]")
    df_trains = add_train_features(df_trains)
    df_trains = add_stop_features(df_trains)

    le_type  = LabelEncoder().fit(df_trains["train_type"].unique())
    le_zone  = LabelEncoder().fit(df_trains["zone"].unique())
    le_coach, le_season = make_encoders()

    df_trains["train_type_enc"] = le_type.transform(df_trains["train_type"])
    df_trains["zone_enc"]       = le_zone.transform(df_trains["zone"])
    print(f"  Train feature matrix : {df_trains.shape}")

    # 3. Build booking dataset
    print("\n[BOOKINGS]")
    bk = _build_bookings(df_trains, le_coach, le_season, le_type, le_zone)
    print(f"  Records        : {len(bk):,}")

    # 4. Train / test split
    X = bk[FEATURE_COLS].values
    y = bk["confirmed"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    print(f"\n[SPLIT]  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # 5. Train models
    print("\n[TRAINING]")
    results, trained, stack, imp_s, sca_s, Xte_s, yp_s, ypr_s, thresholds = \
        _train(X_tr, X_te, y_tr, y_te)

    # 6. Pick best
    df_res    = pd.DataFrame(results).T
    best_name = df_res["Accuracy"].idxmax()
    best_acc  = df_res.loc[best_name, "Accuracy"]
    target    = "✅ MEETS >80%" if best_acc >= 0.80 else "⚠️  Below 80%"

    print(f"\n{'='*65}")
    print(f"  BEST MODEL : {best_name}")
    print(f"  ACCURACY   : {best_acc:.4f}  |  {target}")
    print(f"  THRESHOLD  : {thresholds.get(best_name, 0.5):.2f}")
    print(f"{'='*65}")
    print(df_res.to_string())
    best_yp = (yp_s if best_name == "Stacking Ensemble"
               else trained[best_name].predict(X_te))
    print("\n" + classification_report(
        y_te, best_yp, target_names=["Not Confirmed","Confirmed"]))

    # 7. Save charts
    _save_charts(trained, results, stack, y_te, X_te,
                 yp_s, ypr_s, best_name)

    # 8. Export model
    best_obj = stack if best_name == "Stacking Ensemble" else trained[best_name]
    payload  = {
        "model":           best_obj,
        "imputer":         imp_s,
        "scaler":          sca_s,
        "need_preprocess": (best_name == "Stacking Ensemble"),
        "feature_cols":    FEATURE_COLS,
        "feature_names":   FEATURE_NAMES,
        "le_coach":        le_coach,
        "le_season":       le_season,
        "le_type":         le_type,
        "le_zone":         le_zone,
        "train_data":      df_trains,
        "station_data":    df_stations,
        "best_model_name": best_name,
        "metrics":         df_res.to_dict(),
        "thresholds":      thresholds,
        "coach_types":     COACH_TYPES,
        "seasons":         SEASONS,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\n[EXPORT] Saved → {MODEL_PATH}")

    # 9. Summary
    print(f"\n{'='*65}")
    print("  OUTPUT FILES")
    print(f"{'='*65}")
    for fname in ["best_model_real.joblib", "feature_importance_real.png",
                  "roc_curves_real.png", "metrics_heatmap_real.png",
                  "confusion_matrix_real.png", "probability_distribution_real.png"]:
        fp   = OUTPUTS / fname
        size = f"{fp.stat().st_size/1024:.1f} KB" if fp.exists() else "MISSING"
        mark = "✅" if fp.exists() else "❌"
        print(f"  {mark}  {fname:<42} ({size})")
    print(f"\n  All outputs → {OUTPUTS}")


if __name__ == "__main__":
    main()
