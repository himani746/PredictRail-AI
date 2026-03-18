"""
SmartTransit AI — PS-02
Real Data Pipeline  (stations.json + trains.json + schedules.json schema)
=========================================================================
Run:
    python smarttransit_real.py

Outputs (saved next to this script in outputs/ folder):
    best_model_real.joblib
    feature_importance.png
    roc_curves.png
    metrics_heatmap.png
    confusion_matrix.png
    probability_distribution.png
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       StackingClassifier)
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score, roc_auc_score,
                                       roc_curve, confusion_matrix,
                                       ConfusionMatrixDisplay,
                                       classification_report)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))   # folder where script lives
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ↓↓ Change these if your JSON files are in a different folder ↓↓
STATIONS_PATH  = os.path.join(BASE, "stations.json")
TRAINS_PATH    = os.path.join(BASE, "trains.json")
# schedules.json is optional — if present it will be used for stop counts
SCHEDULES_PATH = os.path.join(BASE, "schedules.json")

SEED = 42
np.random.seed(SEED)

print("=" * 65)
print("  SmartTransit AI  |  PS-02  |  Real Data Pipeline")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD & PARSE REAL JSON FILES
# ─────────────────────────────────────────────────────────────────────────────

# ── stations.json  (GeoJSON FeatureCollection) ────────────────────────────────
print("\n[LOAD] stations.json ...")
with open(STATIONS_PATH, "r", encoding="utf-8") as f:
    raw_stations = json.load(f)

station_rows = []
for feat in raw_stations["features"]:
    p = feat["properties"]
    g = feat.get("geometry")
    station_rows.append({
        "station_code": p.get("code", ""),
        "station_name": p.get("name", ""),
        "state":        p.get("state", "Unknown"),
        "zone":         p.get("zone",  "Unknown"),
        "address":      p.get("address", ""),
        "longitude":    g["coordinates"][0] if g else np.nan,
        "latitude":     g["coordinates"][1] if g else np.nan,
    })
df_stations = pd.DataFrame(station_rows).drop_duplicates("station_code")
# Clean bad zone values
df_stations["zone"] = df_stations["zone"].replace({"?": "Unknown", "": "Unknown"})
df_stations["zone"] = df_stations["zone"].fillna("Unknown")
print(f"  Stations loaded : {len(df_stations):,}")

# ── trains.json  (GeoJSON FeatureCollection) ──────────────────────────────────
print("[LOAD] trains.json ...")
with open(TRAINS_PATH, "r", encoding="utf-8") as f:
    raw_trains = json.load(f)

train_rows = []
for feat in raw_trains["features"]:
    p = feat["properties"]
    # derive geometry coords if available
    g = feat.get("geometry")
    coords = g["coordinates"] if g and g.get("coordinates") else []
    train_rows.append({
        "train_number":        str(p.get("number", "")),
        "train_name":          p.get("name", ""),
        "train_type":          p.get("type", ""),
        "zone":                p.get("zone", "Unknown"),
        "from_station_code":   p.get("from_station_code", ""),
        "to_station_code":     p.get("to_station_code", ""),
        "from_station_name":   p.get("from_station_name", ""),
        "to_station_name":     p.get("to_station_name", ""),
        "distance_km":         float(p.get("distance", 0) or 0),
        "duration_h":          float(p.get("duration_h", 0) or 0),
        "duration_m":          float(p.get("duration_m", 0) or 0),
        "first_ac":            int(p.get("first_ac",    0) or 0),
        "second_ac":           int(p.get("second_ac",   0) or 0),
        "third_ac":            int(p.get("third_ac",    0) or 0),
        "sleeper":             int(p.get("sleeper",     0) or 0),
        "chair_car":           int(p.get("chair_car",   0) or 0),
        "first_class":         int(p.get("first_class", 0) or 0),
        "classes":             p.get("classes", ""),
        "return_train":        str(p.get("return_train", "") or ""),
        "num_route_points":    len(coords),
    })
df_trains = pd.DataFrame(train_rows).drop_duplicates("train_number")
df_trains["duration_hours"] = df_trains["duration_h"] + df_trains["duration_m"] / 60
df_trains["total_seats"]    = (df_trains["first_ac"]   * 24 +
                                df_trains["second_ac"]  * 46 +
                                df_trains["third_ac"]   * 64 +
                                df_trains["sleeper"]    * 72 +
                                df_trains["chair_car"]  * 78 +
                                df_trains["first_class"]* 18)
df_trains["has_ac"]         = ((df_trains["first_ac"] + df_trains["second_ac"] +
                                 df_trains["third_ac"]) > 0).astype(int)
df_trains["has_sleeper"]    = (df_trains["sleeper"] > 0).astype(int)
df_trains["speed_kmph"]     = (df_trains["distance_km"] /
                                df_trains["duration_hours"].replace(0, np.nan)).fillna(0)
print(f"  Trains loaded   : {len(df_trains):,}")

# ── schedules.json  (optional — for stop counts) ──────────────────────────────
stop_counts = None
if os.path.exists(SCHEDULES_PATH):
    print("[LOAD] schedules.json ...")
    with open(SCHEDULES_PATH, "r", encoding="utf-8") as f:
        raw_sched = json.load(f)
    df_sched = pd.DataFrame(raw_sched)
    df_sched["train_number"] = df_sched["train_number"].astype(str)
    stop_counts = (df_sched.groupby("train_number").size()
                            .rename("total_stops").reset_index())
    print(f"  Schedules loaded: {len(df_sched):,} stops across "
          f"{df_sched['train_number'].nunique():,} trains")
else:
    print("[SKIP] schedules.json not found — estimating stops from route points")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE ENGINEERING ON REAL DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[FEATURES] Engineering train-level features ...")

# Merge stop counts if available, else estimate
if stop_counts is not None:
    df_trains = df_trains.merge(stop_counts, on="train_number", how="left")
    df_trains["total_stops"] = df_trains["total_stops"].fillna(
        df_trains["num_route_points"].clip(lower=2))
else:
    df_trains["total_stops"] = df_trains["num_route_points"].clip(lower=2)

df_trains["total_stops"]    = df_trains["total_stops"].fillna(5).astype(float)
df_trains["seats_per_stop"] = (df_trains["total_seats"] /
                                df_trains["total_stops"].replace(0, 1))
df_trains["dist_per_stop"]  = (df_trains["distance_km"] /
                                df_trains["total_stops"].replace(0, 1))

# Encode categoricals
le_type = LabelEncoder()
le_zone = LabelEncoder()
df_trains["train_type"]  = df_trains["train_type"].replace("", "Unknown").fillna("Unknown")
df_trains["zone"]        = df_trains["zone"].fillna("Unknown")
df_trains["train_type_enc"] = le_type.fit_transform(df_trains["train_type"])
df_trains["zone_enc"]       = le_zone.fit_transform(df_trains["zone"])

# Premium score — higher for more premium classes
df_trains["premium_score"] = (df_trains["first_ac"]   * 3 +
                               df_trains["second_ac"]  * 2 +
                               df_trains["third_ac"]   * 1 +
                               df_trains["chair_car"]  * 1)

print(f"  Train features ready. Shape: {df_trains.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SYNTHETIC BOOKING DATASET  (based on REAL train distributions)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[BOOKINGS] Generating realistic booking records from real train data ...")

N_BOOKINGS  = 30_000
COACH_TYPES = ["1A","2A","3A","SL","CC","2S"]
SEASONS     = ["peak","normal","off-peak"]

rng = np.random.default_rng(SEED)

# Sample real trains (with replacement)
idx = rng.integers(0, len(df_trains), size=N_BOOKINGS)
ts  = df_trains.iloc[idx].reset_index(drop=True)

bk = pd.DataFrame()
bk["train_number"]   = ts["train_number"].values
bk["train_type_enc"] = ts["train_type_enc"].values
bk["train_type"]     = ts["train_type"].values
bk["zone_enc"]       = ts["zone_enc"].values
bk["distance_km"]    = ts["distance_km"].values
bk["duration_hours"] = ts["duration_hours"].values
bk["total_seats"]    = ts["total_seats"].values
bk["total_stops"]    = ts["total_stops"].values
bk["speed_kmph"]     = ts["speed_kmph"].values
bk["seats_per_stop"] = ts["seats_per_stop"].values
bk["dist_per_stop"]  = ts["dist_per_stop"].values
bk["has_ac"]         = ts["has_ac"].values
bk["has_sleeper"]    = ts["has_sleeper"].values
bk["premium_score"]  = ts["premium_score"].values

# Booking-level fields
bk["waitlist_position"] = rng.integers(1, 100, size=N_BOOKINGS)
bk["coach_type"]        = rng.choice(COACH_TYPES, size=N_BOOKINGS,
                                      p=[0.05,0.10,0.25,0.38,0.12,0.10])
bk["booking_day"]       = rng.integers(1, 120, size=N_BOOKINGS)
bk["day_of_week"]       = rng.integers(0, 7,   size=N_BOOKINGS)
bk["season"]            = rng.choice(SEASONS, size=N_BOOKINGS, p=[0.30,0.50,0.20])
bk["cancellation_rate"] = rng.uniform(0.03, 0.40, size=N_BOOKINGS).round(3)
bk["quota_available"]   = rng.integers(0, 30,  size=N_BOOKINGS)
bk["is_weekend"]        = (bk["day_of_week"] >= 5).astype(int)
bk["is_holiday_week"]   = rng.choice([0,1], size=N_BOOKINGS, p=[0.75,0.25])

# Encode booking categoricals
le_coach  = LabelEncoder().fit(COACH_TYPES)
le_season = LabelEncoder().fit(SEASONS)
bk["coach_type_enc"] = le_coach.transform(bk["coach_type"])
bk["season_enc"]     = le_season.transform(bk["season"])

# Interaction features
bk["wl_x_cancel"]    = bk["waitlist_position"] * bk["cancellation_rate"]
bk["wl_x_booking"]   = bk["waitlist_position"] / (bk["booking_day"] + 1)
bk["coach_x_season"] = bk["coach_type_enc"]    * bk["season_enc"]
bk["quota_x_wl"]     = bk["quota_available"]   / (bk["waitlist_position"] + 1)

# ── Target variable — driven by REAL train characteristics ────────────────────
coach_bonus = bk["coach_type"].map(
    {"1A":0.30,"2A":0.22,"CC":0.12,"3A":0.05,"SL":0.00,"2S":-0.08}).fillna(0)
season_pen = bk["season"].map(
    {"off-peak":0.20,"normal":0.00,"peak":-0.25}).fillna(0)

# Real train type bonus using actual type codes from trains.json
type_bonus = bk["train_type"].map({
    "Raj":0.20, "Drnt":0.18, "JShtb":0.15, "Shtb":0.12, "SKr":0.12,
    "SF":0.08,  "Mail":0.06, "Exp":0.03,   "Pass":0.00, "MEMU":-0.03,
    "DEMU":-0.03, "GR":-0.05, "Toy":-0.08, "Hyd":0.05,
    "Del":0.10, "Klkt":0.12, "": 0.00
}).fillna(0)

# Premium trains have more confirmed seats (real insight from data)
premium_bonus = (bk["premium_score"] / 10).clip(upper=0.3)

# More total seats → easier confirmation
seat_bonus = (bk["total_seats"] / 500).clip(upper=0.2)

logit = (
    3.0
    - 0.055 * bk["waitlist_position"]
    + 2.0   * bk["cancellation_rate"]
    + 0.010 * bk["booking_day"]
    + 0.08  * bk["quota_available"]
    - 0.10  * bk["is_holiday_week"]
    - 0.08  * bk["is_weekend"]
    + 0.05  * bk["has_ac"]
    + coach_bonus + season_pen + type_bonus + premium_bonus + seat_bonus
    + rng.normal(0, 0.25, size=N_BOOKINGS)
)
prob_true = 1 / (1 + np.exp(-logit))
bk["confirmed"] = (rng.uniform(size=N_BOOKINGS) < prob_true).astype(int)

print(f"  Booking records : {len(bk):,}")
print(f"  Confirmation rate: {bk['confirmed'].mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FEATURES & SPLIT
# ─────────────────────────────────────────────────────────────────────────────
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
    "Waitlist Position",  "Coach Type",        "Booking Day",      "Season",
    "Cancellation Rate",  "Day of Week",       "Total Stops",      "Duration (hrs)",
    "Distance (km)",      "Train Type",        "Total Seats",      "Speed (km/h)",
    "Seats/Stop",         "Quota Available",   "Is Weekend",       "Is Holiday",
    "WL × Cancel",        "WL / BookDay",      "Coach × Season",   "Quota / WL",
    "Dist/Stop",          "Has AC",            "Has Sleeper",      "Premium Score",
    "Zone",
]

X = bk[FEATURE_COLS].values
y = bk["confirmed"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
print(f"\n[SPLIT]  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
base_models = {
    "Logistic Regression": Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sca", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced", random_state=SEED)),
    ]),
    "Random Forest": Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced",
            random_state=SEED, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.07,
            subsample=0.8, random_state=SEED)),
    ]),
}

results = {}
trained = {}

print("\n[TRAINING]")
print("-" * 65)
for name, pipe in base_models.items():
    pipe.fit(X_train, y_train)
    yp  = pipe.predict(X_test)
    ypr = pipe.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, yp)
    prec = precision_score(y_test, yp, zero_division=0)
    rec  = recall_score(y_test, yp, zero_division=0)
    f1   = f1_score(y_test, yp, zero_division=0)
    auc  = roc_auc_score(y_test, ypr)
    cv   = cross_val_score(pipe, X_train, y_train, cv=3, scoring="accuracy").mean()
    results[name] = dict(Accuracy=acc, Precision=prec, Recall=rec,
                         F1=f1, ROC_AUC=auc, CV_Acc=cv)
    trained[name] = pipe
    print(f"  {name:22s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  CV={cv:.4f}")

# ── Stacking Ensemble ─────────────────────────────────────────────────────────
print("\n[TRAINING STACKING ENSEMBLE]")
imp_s = SimpleImputer(strategy="mean")
sca_s = StandardScaler()
X_tr_s = sca_s.fit_transform(imp_s.fit_transform(X_train))
X_te_s = sca_s.transform(imp_s.transform(X_test))

estimators = [
    ("lr", trained["Logistic Regression"].named_steps["clf"]),
    ("rf", trained["Random Forest"].named_steps["clf"]),
    ("gb", trained["Gradient Boosting"].named_steps["clf"]),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
    cv=3, n_jobs=-1,
)
stack.fit(X_tr_s, y_train)
yp_s  = stack.predict(X_te_s)
ypr_s = stack.predict_proba(X_te_s)[:, 1]
acc_s  = accuracy_score(y_test, yp_s)
prec_s = precision_score(y_test, yp_s, zero_division=0)
rec_s  = recall_score(y_test, yp_s, zero_division=0)
f1_s   = f1_score(y_test, yp_s, zero_division=0)
auc_s  = roc_auc_score(y_test, ypr_s)
cv_s   = cross_val_score(stack, X_tr_s, y_train, cv=3, scoring="accuracy").mean()
results["Stacking Ensemble"] = dict(Accuracy=acc_s, Precision=prec_s, Recall=rec_s,
                                     F1=f1_s, ROC_AUC=auc_s, CV_Acc=cv_s)
print(f"  {'Stacking Ensemble':22s}  Acc={acc_s:.4f}  F1={f1_s:.4f}  "
      f"AUC={auc_s:.4f}  CV={cv_s:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SELECT BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
df_res    = pd.DataFrame(results).T
best_name = df_res["Accuracy"].idxmax()
best_acc  = df_res.loc[best_name, "Accuracy"]
target    = "✅ MEETS >80% TARGET" if best_acc >= 0.80 else "⚠️  Below 80%"

print(f"\n{'='*65}")
print(f"  BEST MODEL : {best_name}")
print(f"  ACCURACY   : {best_acc:.4f}  |  {target}")
print(f"{'='*65}")
print(df_res.to_string())
print("\n" + classification_report(
    y_test,
    stack.predict(X_te_s) if best_name=="Stacking Ensemble" else trained[best_name].predict(X_test),
    target_names=["Not Confirmed","Confirmed"]))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":        "#55A868",
    "Gradient Boosting":    "#C44E52",
    "Stacking Ensemble":    "#FF8C00",
}

# ── VIZ 1: Feature Importance ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle("Feature Importance — Real Data Model", fontsize=15, fontweight="bold")
for ax, mname in zip(axes, ["Random Forest","Gradient Boosting"]):
    clf  = trained[mname].named_steps["clf"]
    imp_ = clf.feature_importances_ / clf.feature_importances_.sum()
    idx_ = np.argsort(imp_)
    bars = ax.barh([FEATURE_NAMES[i] for i in idx_], imp_[idx_],
                   color=PALETTE[mname], edgecolor="white", linewidth=0.4)
    ax.set_title(mname, fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Importance")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    for bar, v in zip(bars, imp_[idx_]):
        ax.text(v+0.002, bar.get_y()+bar.get_height()/2,
                f"{v:.1%}", va="center", fontsize=7.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── VIZ 2: ROC Curves ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("ROC Curves — All Models", fontsize=13, fontweight="bold")
for ax, name in zip(axes, list(trained.keys()) + ["Stacking Ensemble"]):
    ypr_ = ypr_s if name=="Stacking Ensemble" else trained[name].predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, ypr_)
    auc_ = roc_auc_score(y_test, ypr_)
    ax.plot(fpr, tpr, color=PALETTE[name], lw=2.5, label=f"AUC={auc_:.4f}")
    ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE[name])
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── VIZ 3: Metrics Heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
hm = df_res[["Accuracy","Precision","Recall","F1","ROC_AUC","CV_Acc"]].astype(float)
sns.heatmap(hm, annot=True, fmt=".4f", cmap="YlGn", linewidths=0.5,
            ax=ax, vmin=0.5, vmax=1.0, annot_kws={"size":11,"weight":"bold"})
ax.set_title("Model Performance — Real Data", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="x", rotation=15); ax.tick_params(axis="y", rotation=0)
bi = list(df_res.index).index(best_name)
ax.add_patch(plt.Rectangle((0,bi), hm.shape[1], 1,
             fill=False, edgecolor="#FF8C00", lw=3))
ax.text(hm.shape[1]+0.1, bi+0.5, "◀ BEST", va="center",
        color="#FF8C00", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "metrics_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── VIZ 4: Confusion Matrices ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
for ax, name in zip(axes, list(trained.keys()) + ["Stacking Ensemble"]):
    yp_ = yp_s if name=="Stacking Ensemble" else trained[name].predict(X_test)
    cm_ = confusion_matrix(y_test, yp_)
    ConfusionMatrixDisplay(cm_, display_labels=["Not Confirmed","Confirmed"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAcc={accuracy_score(y_test,yp_):.4f}",
                 fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── VIZ 5: Probability Distribution ──────────────────────────────────────────
best_probs = ypr_s if best_name=="Stacking Ensemble" else trained[best_name].predict_proba(X_test)[:,1]
fig, ax = plt.subplots(figsize=(9, 5))
for label, color, lname in [(0,"#C44E52","Not Confirmed"),(1,"#55A868","Confirmed")]:
    mask = y_test == label
    ax.hist(best_probs[mask], bins=40, alpha=0.6, color=color,
            label=f"{lname} (n={mask.sum():,})", density=True, edgecolor="white")
ax.axvline(0.5, color="black", lw=2, ls="--", label="Decision boundary")
ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
ax.set_title(f"Probability Distribution — {best_name}", fontsize=12, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "probability_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n[VIZ] All 5 charts saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EXPORT MODEL
# ─────────────────────────────────────────────────────────────────────────────
best_obj = stack if best_name == "Stacking Ensemble" else trained[best_name]

payload = {
    "model":            best_obj,
    "imputer":          imp_s,
    "scaler":           sca_s,
    "need_preprocess":  (best_name == "Stacking Ensemble"),
    "feature_cols":     FEATURE_COLS,
    "feature_names":    FEATURE_NAMES,
    "le_coach":         le_coach,
    "le_season":        le_season,
    "le_type":          le_type,
    "le_zone":          le_zone,
    "train_data":       df_trains,
    "station_data":     df_stations,
    "best_model_name":  best_name,
    "metrics":          df_res.to_dict(),
    "coach_types":      COACH_TYPES,
    "seasons":          SEASONS,
}
joblib.dump(payload, os.path.join(OUT, "best_model_real.joblib"))
print(f"[EXPORT] best_model_real.joblib saved  (model={best_name})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_waitlist_confirmation(
    train_number:        str,
    source_station:      str,
    destination_station: str,
    waitlist_position:   int,
    coach_type:          str,
    booking_day:         int,
    season:              str   = "normal",
    day_of_week:         int   = 0,
    cancellation_rate:   float = 0.15,
    quota_available:     int   = 5,
    is_holiday_week:     int   = 0,
    model_dir:           str   = OUT,
) -> dict:
    """
    Predict PNR confirmation probability using the real-data trained model.

    Parameters
    ----------
    train_number         : e.g. "12301"
    source_station       : station code e.g. "HWH"
    destination_station  : station code e.g. "NDLS"
    waitlist_position    : integer ≥ 1
    coach_type           : 1A | 2A | 3A | SL | CC | 2S
    booking_day          : days before journey (1–120)
    season               : peak | normal | off-peak
    day_of_week          : 0=Mon … 6=Sun
    cancellation_rate    : 0.0–1.0
    quota_available      : estimated quota seats
    is_holiday_week      : 0 or 1
    """
    pl  = joblib.load(os.path.join(model_dir, "best_model_real.joblib"))
    mdl = pl["model"]
    _imp = pl["imputer"]; _sca = pl["scaler"]
    _lc  = pl["le_coach"]; _ls = pl["le_season"]
    _lt  = pl["le_type"];  _lz = pl["le_zone"]
    tf_  = pl["train_data"]
    COACH_TYPES_ = pl["coach_types"]
    SEASONS_     = pl["seasons"]

    # Look up real train
    row = tf_[tf_["train_number"] == str(train_number)]
    if not row.empty:
        t = row.iloc[0]
    else:
        t = tf_.mean(numeric_only=True)
        t["train_type_enc"] = 0
        t["zone_enc"]       = 0

    total_stops    = float(t.get("total_stops", 10))
    duration_hours = float(t.get("duration_hours", 5))
    distance_km    = float(t.get("distance_km", 500))
    total_seats    = float(t.get("total_seats", 100))
    speed_kmph     = float(t.get("speed_kmph", 60))
    seats_per_stop = float(t.get("seats_per_stop", 10))
    dist_per_stop  = float(t.get("dist_per_stop", 50))
    has_ac         = float(t.get("has_ac", 0))
    has_sleeper    = float(t.get("has_sleeper", 0))
    premium_score  = float(t.get("premium_score", 0))
    train_type_enc = float(t.get("train_type_enc", 0))
    zone_enc       = float(t.get("zone_enc", 0))

    coach_enc  = int(_lc.transform([coach_type])[0]) if coach_type in _lc.classes_ else 0
    season_enc = int(_ls.transform([season])[0])     if season     in _ls.classes_ else 1
    is_weekend = int(day_of_week >= 5)

    wl_x_cancel  = waitlist_position * cancellation_rate
    wl_x_booking = waitlist_position / (booking_day + 1)
    coach_x_seas = coach_enc * season_enc
    quota_x_wl   = quota_available / (waitlist_position + 1)

    X_in = np.array([[
        waitlist_position, coach_enc,    booking_day,   season_enc,
        cancellation_rate, day_of_week,  total_stops,   duration_hours,
        distance_km,       train_type_enc, total_seats, speed_kmph,
        seats_per_stop,    quota_available, is_weekend,  is_holiday_week,
        wl_x_cancel,       wl_x_booking,   coach_x_seas, quota_x_wl,
        dist_per_stop,     has_ac,          has_sleeper,  premium_score,
        zone_enc,
    ]], dtype=float)

    if pl["need_preprocess"]:
        X_in = _sca.transform(_imp.transform(X_in))

    prob = float(mdl.predict_proba(X_in)[0, 1])

    band = ("High (>75%)"     if prob >= 0.75 else
            "Medium (50–75%)" if prob >= 0.50 else
            "Low (25–50%)"    if prob >= 0.25 else
            "Very Low (<25%)")

    # Real train info for response
    train_info = {}
    if not tf_[tf_["train_number"]==str(train_number)].empty:
        r = tf_[tf_["train_number"]==str(train_number)].iloc[0]
        train_info = {
            "name":     r.get("train_name",""),
            "type":     r.get("train_type",""),
            "distance": r.get("distance_km",0),
            "from":     r.get("from_station_name",""),
            "to":       r.get("to_station_name",""),
        }

    alternatives = []
    if prob < 0.50:
        for c in [x for x in COACH_TYPES_ if x != coach_type][:2]:
            alternatives.append(f"Try {c} class — may have better quota")
        alternatives.append("Book earlier (more days in advance) for better WL clearance")
        if season == "peak":
            alternatives.append("Off-peak travel improves confirmation by ~20%")

    return {
        "pnr_confirmation_probability": round(prob, 4),
        "confirmed":                    prob >= 0.50,
        "confidence_band":              band,
        "train_info":                   train_info,
        "suggested_alternatives":       alternatives,
        "input_summary": {
            "train_number":      train_number,
            "source":            source_station,
            "destination":       destination_station,
            "waitlist_position": waitlist_position,
            "coach_type":        coach_type,
            "booking_day":       booking_day,
            "season":            season,
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — DEMO with REAL train numbers
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  DEMO — Using Real Train Numbers from trains.json")
print("="*65)

# Pick 5 real train numbers
real_trains = df_trains["train_number"].sample(5, random_state=SEED).tolist()
demo_cases = [
    (real_trains[0], "HWH", "NDLS",  3, "2A",  90, "off-peak", 2, 0.28, 10, 0),
    (real_trains[1], "BCT", "NDLS", 45, "SL",  10, "peak",     5, 0.07,  1, 1),
    (real_trains[2], "MAS", "HWH",  18, "3A",  45, "normal",   3, 0.18,  5, 0),
    (real_trains[3], "NDLS","BCT",   2, "1A", 120, "peak",     1, 0.32, 10, 0),
    (real_trains[4], "CNB", "MAS",  72, "SL",  14, "peak",     6, 0.05,  0, 1),
]
for d in demo_cases:
    r = predict_waitlist_confirmation(*d)
    status = "✅" if r["confirmed"] else "❌"
    tinfo  = r["train_info"]
    print(f"\n  {status}  Train {d[0]} ({tinfo.get('name','')[:30]})")
    print(f"       WL#{d[3]:>3}  {d[4]:<3}  {d[5]:>3}d ahead  {d[6]:<9}"
          f"  → {r['pnr_confirmation_probability']:.2%}  [{r['confidence_band']}]")
    for a in r["suggested_alternatives"]:
        print(f"       💡 {a}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  OUTPUT FILES")
print(f"{'='*65}")
for fname in ["best_model_real.joblib", "feature_importance.png",
              "roc_curves.png", "metrics_heatmap.png",
              "confusion_matrix.png", "probability_distribution.png"]:
    fpath = os.path.join(OUT, fname)
    size  = f"{os.path.getsize(fpath)/1024:.1f} KB" if os.path.exists(fpath) else "missing"
    print(f"  ✅  {fname:<40} ({size})")
print(f"\n  All outputs saved to: {OUT}")
