"""
ml/features.py
Feature column definitions and engineering helpers.
Shared by ml/pipeline.py (training) and ml/predictor.py (inference).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.config import COACH_TYPES, SEASONS

# ── Column lists — single source of truth ────────────────────────────────────
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


# ── Encoder factory ───────────────────────────────────────────────────────────
def make_encoders():
    """Return (le_coach, le_season) fitted on the known value lists."""
    le_coach  = LabelEncoder().fit(COACH_TYPES)
    le_season = LabelEncoder().fit(SEASONS)
    return le_coach, le_season


# ── Train-level feature engineering ──────────────────────────────────────────
def add_train_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration_hours"] = df["duration_h"] + df["duration_m"] / 60
    df["total_seats"] = (
        df["first_ac"]    * 24 +
        df["second_ac"]   * 46 +
        df["third_ac"]    * 64 +
        df["sleeper"]     * 72 +
        df["chair_car"]   * 78 +
        df["first_class"] * 18
    )
    df["has_ac"]        = ((df["first_ac"] + df["second_ac"] + df["third_ac"]) > 0).astype(int)
    df["has_sleeper"]   = (df["sleeper"] > 0).astype(int)
    df["speed_kmph"]    = (df["distance_km"] / df["duration_hours"].replace(0, np.nan)).fillna(0)
    df["premium_score"] = (
        df["first_ac"]  * 3 +
        df["second_ac"] * 2 +
        df["third_ac"]  * 1 +
        df["chair_car"] * 1
    )
    return df


def add_stop_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_stops"]    = df["total_stops"].fillna(5).astype(float)
    df["seats_per_stop"] = df["total_seats"] / df["total_stops"].replace(0, 1)
    df["dist_per_stop"]  = df["distance_km"] / df["total_stops"].replace(0, 1)
    return df


# ── Booking-level interaction features ───────────────────────────────────────
def build_interaction_features(bk: pd.DataFrame) -> pd.DataFrame:
    bk = bk.copy()
    bk["wl_x_cancel"]    = bk["waitlist_position"] * bk["cancellation_rate"]
    bk["wl_x_booking"]   = bk["waitlist_position"] / (bk["booking_day"] + 1)
    bk["coach_x_season"] = bk["coach_type_enc"]    * bk["season_enc"]
    bk["quota_x_wl"]     = bk["quota_available"]   / (bk["waitlist_position"] + 1)
    bk["is_weekend"]     = (bk["day_of_week"] >= 5).astype(int)
    return bk


# ── Single-row array for inference ───────────────────────────────────────────
def single_row_array(
    waitlist_position:  int,
    coach_type:         str,
    booking_day:        int,
    season:             str,
    cancellation_rate:  float,
    day_of_week:        int,
    quota_available:    int,
    is_holiday_week:    int,
    train_row:          dict,
    le_coach:           LabelEncoder,
    le_season:          LabelEncoder,
) -> np.ndarray:
    """Build a 1 × len(FEATURE_COLS) array ready for model.predict_proba()."""
    coach_enc  = int(le_coach.transform([coach_type])[0]) if coach_type in le_coach.classes_ else 0
    season_enc = int(le_season.transform([season])[0])    if season     in le_season.classes_ else 1
    is_weekend = int(day_of_week >= 5)

    wl_x_cancel  = waitlist_position * cancellation_rate
    wl_x_booking = waitlist_position / (booking_day + 1)
    coach_x_seas = coach_enc * season_enc
    quota_x_wl   = quota_available / (waitlist_position + 1)

    t = train_row
    return np.array([[
        waitlist_position,
        coach_enc,
        booking_day,
        season_enc,
        cancellation_rate,
        day_of_week,
        float(t.get("total_stops",     10)),
        float(t.get("duration_hours",   5)),
        float(t.get("distance_km",    500)),
        float(t.get("train_type_enc",   0)),
        float(t.get("total_seats",    100)),
        float(t.get("speed_kmph",      60)),
        float(t.get("seats_per_stop",  10)),
        quota_available,
        is_weekend,
        is_holiday_week,
        wl_x_cancel,
        wl_x_booking,
        coach_x_seas,
        quota_x_wl,
        float(t.get("dist_per_stop",   50)),
        float(t.get("has_ac",           0)),
        float(t.get("has_sleeper",      0)),
        float(t.get("premium_score",    0)),
        float(t.get("zone_enc",         0)),
    ]], dtype=float)
