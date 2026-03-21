"""
ml/predict.py
Prediction API for RailPulse AI.

- predict_waitlist_confirmation() uses the trained joblib model.
- predict_confirmation_fast()     is a lightweight heuristic fallback
  used by the Streamlit UI when no trained model is present.
"""

import os
import numpy as np
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (MODEL_PATH, FEATURE_COLS, COACH_TYPES, SEASONS)


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY: real model prediction
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
) -> dict:
    """
    Predict PNR confirmation probability using the trained ensemble model.

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
    quota_available      : estimated quota seats available
    is_holiday_week      : 0 or 1

    Returns
    -------
    dict with keys:
        pnr_confirmation_probability, confirmed, confidence_band,
        train_info, suggested_alternatives, input_summary
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python -m ml.train` first."
        )

    pl  = joblib.load(MODEL_PATH)
    mdl = pl["model"]
    _imp, _sca = pl["imputer"], pl["scaler"]
    _lc, _ls   = pl["le_coach"], pl["le_season"]
    tf_        = pl["train_data"]

    # Look up real train characteristics
    row = tf_[tf_["train_number"] == str(train_number)]
    t   = row.iloc[0] if not row.empty else tf_.mean(numeric_only=True)

    def _get(col, default=0.0):
        try:
            return float(t[col])
        except (KeyError, TypeError):
            return float(default)

    total_stops    = _get("total_stops", 10)
    duration_hours = _get("duration_hours", 5)
    distance_km    = _get("distance_km", 500)
    total_seats    = _get("total_seats", 100)
    speed_kmph     = _get("speed_kmph", 60)
    seats_per_stop = _get("seats_per_stop", 10)
    dist_per_stop  = _get("dist_per_stop", 50)
    has_ac         = _get("has_ac", 0)
    has_sleeper    = _get("has_sleeper", 0)
    premium_score  = _get("premium_score", 0)
    train_type_enc = _get("train_type_enc", 0)
    zone_enc       = _get("zone_enc", 0)

    coach_enc  = (int(_lc.transform([coach_type])[0])
                  if coach_type in _lc.classes_ else 0)
    season_enc = (int(_ls.transform([season])[0])
                  if season in _ls.classes_ else 1)
    is_weekend = int(day_of_week >= 5)

    wl_x_cancel  = waitlist_position * cancellation_rate
    wl_x_booking = waitlist_position / (booking_day + 1)
    coach_x_seas = coach_enc * season_enc
    quota_x_wl   = quota_available / (waitlist_position + 1)

    X_in = np.array([[
        waitlist_position, coach_enc,      booking_day,     season_enc,
        cancellation_rate, day_of_week,    total_stops,     duration_hours,
        distance_km,       train_type_enc, total_seats,     speed_kmph,
        seats_per_stop,    quota_available, is_weekend,     is_holiday_week,
        wl_x_cancel,       wl_x_booking,   coach_x_seas,   quota_x_wl,
        dist_per_stop,     has_ac,          has_sleeper,    premium_score,
        zone_enc,
    ]], dtype=float)

    if pl["need_preprocess"]:
        X_in = _sca.transform(_imp.transform(X_in))

    prob = float(mdl.predict_proba(X_in)[0, 1])
    return _build_result(prob, train_number, coach_type, booking_day,
                         season, source_station, destination_station,
                         waitlist_position, tf_)


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: heuristic for Streamlit UI when model isn't trained yet
# ─────────────────────────────────────────────────────────────────────────────
def predict_confirmation_fast(
    train_id:      str,
    journey_class: str,
    wl_number:     int,
    days:          int,
) -> float:
    """
    Lightweight heuristic prediction — no model required.
    Used by the Streamlit UI when best_model_real.joblib doesn't exist.

    Returns a probability float 0–100 (percentage, not 0–1).
    """
    _BASE = {"SL": 72, "3A": 65, "2A": 58, "1A": 85, "CC": 78, "EC": 60}
    _TBNS = {"12027": 9, "11007": 5, "12123": -4, "12301": 12, "11301": -6}
    _CBNX = {"11007": 3, "12027": 1, "12123": 2,  "11301": 4,  "12301": 1}

    base        = _BASE.get(journey_class, 65)
    wl_penalty  = min(wl_number * 3.8, 58)
    days_bonus  = min(days * 0.9, 22)
    train_bonus = _TBNS.get(train_id, 0)
    cx_bonus    = _CBNX.get(train_id, 0)
    prob = max(4.0, min(97.0, base - wl_penalty + days_bonus + train_bonus + cx_bonus))
    return round(prob, 1)


def smart_predict(train_id, journey_class, wl_number, days,
                  source="", destination="", season="normal",
                  day_of_week=0, cancellation_rate=0.15,
                  quota_available=5, is_holiday_week=0) -> dict:
    """
    Unified prediction entry-point used by Streamlit pages.
    Tries real model first; falls back to heuristic gracefully.
    """
    try:
        result = predict_waitlist_confirmation(
            train_number=train_id,
            source_station=source,
            destination_station=destination,
            waitlist_position=wl_number,
            coach_type=journey_class,
            booking_day=days,
            season=season,
            day_of_week=day_of_week,
            cancellation_rate=cancellation_rate,
            quota_available=quota_available,
            is_holiday_week=is_holiday_week,
        )
        result["source"] = "real_model"
        return result
    except FileNotFoundError:
        prob_pct = predict_confirmation_fast(train_id, journey_class,
                                             wl_number, days)
        prob     = prob_pct / 100.0
        return {
            **_build_result(prob, train_id, journey_class, days,
                            season, source, destination, wl_number, None),
            "source": "heuristic",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shared result builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_result(prob, train_number, coach_type, booking_day,
                  season, source, destination, waitlist_position, tf_):
    band = (
        "High (>75%)"     if prob >= 0.75 else
        "Medium (50–75%)" if prob >= 0.50 else
        "Low (25–50%)"    if prob >= 0.25 else
        "Very Low (<25%)"
    )

    train_info = {}
    if tf_ is not None:
        row = tf_[tf_["train_number"] == str(train_number)]
        if not row.empty:
            r = row.iloc[0]
            train_info = {
                "name":     r.get("train_name", ""),
                "type":     r.get("train_type", ""),
                "distance": r.get("distance_km", 0),
                "from":     r.get("from_station_name", ""),
                "to":       r.get("to_station_name", ""),
            }

    alternatives = []
    if prob < 0.50:
        for c in [x for x in COACH_TYPES if x != coach_type][:2]:
            alternatives.append(f"Try {c} class — may have better quota availability")
        alternatives.append(
            "Book earlier (more days in advance) for better WL clearance")
        if season == "peak":
            alternatives.append(
                "Off-peak travel improves confirmation probability by ~20%")

    return {
        "pnr_confirmation_probability": round(prob, 4),
        "pnr_confirmation_pct":         round(prob * 100, 1),
        "confirmed":                    prob >= 0.50,
        "confidence_band":              band,
        "train_info":                   train_info,
        "suggested_alternatives":       alternatives,
        "input_summary": {
            "train_number":      train_number,
            "source":            source,
            "destination":       destination,
            "waitlist_position": waitlist_position,
            "coach_type":        coach_type,
            "booking_day":       booking_day,
            "season":            season,
        },
    }
