"""
ml/predictor.py
Inference wrapper used by the Streamlit UI.

Priority:
  1. Use trained model from outputs/best_model_real.joblib if it exists
  2. Fall back to fast heuristic (no model file needed)

Usage:
    from ml.predictor import Predictor
    p = Predictor()
    result = p.predict(train_id="11007", coach_type="3A", wl_no=8,
                       booking_day=30, season="normal", day_of_week=2,
                       cancellation_rate=0.15, quota_available=5,
                       is_holiday_week=0)
    print(result["probability"])   # 0.73
"""

import os
import numpy as np
import joblib
from typing import Optional

from core.config  import MODEL_PATH, CLASS_META, DEMO_TRAINS
from ml.features  import single_row_array, make_encoders


class Predictor:
    """Thin wrapper that loads a saved model (or falls back to heuristic)."""

    def __init__(self):
        self._payload   = None
        self._le_coach  = None
        self._le_season = None
        self._loaded    = False
        self._try_load()

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(
        self,
        train_id:          str,
        coach_type:        str,
        wl_no:             int,
        booking_day:       int,
        season:            str   = "normal",
        day_of_week:       int   = 0,
        cancellation_rate: float = 0.15,
        quota_available:   int   = 5,
        is_holiday_week:   int   = 0,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
            probability     float  0–1
            confirmed       bool   prob >= 0.50
            confidence_band str    High / Medium / Low / Very Low
            model_used      str    "ML Model" or "Heuristic"
            train_info      dict   name, type, distance etc (from train_data)
            alternatives    list[str]  suggestions if prob < 0.50
        """
        if self._loaded and self._payload is not None:
            return self._ml_predict(
                train_id, coach_type, wl_no, booking_day,
                season, day_of_week, cancellation_rate,
                quota_available, is_holiday_week,
            )
        return self._heuristic_predict(train_id, coach_type, wl_no,
                                       booking_day, season, day_of_week)

    @property
    def model_name(self) -> str:
        if self._loaded and self._payload:
            return self._payload.get("best_model_name", "ML Model")
        return "Heuristic"

    @property
    def metrics(self) -> Optional[dict]:
        if self._payload:
            return self._payload.get("metrics")
        return None

    # ── Load ───────────────────────────────────────────────────────────────────
    def _try_load(self):
        if MODEL_PATH.exists():
            try:
                self._payload   = joblib.load(MODEL_PATH)
                self._le_coach  = self._payload["le_coach"]
                self._le_season = self._payload["le_season"]
                self._loaded    = True
            except Exception:
                self._loaded = False
        else:
            # No model file — build default encoders for heuristic
            self._le_coach, self._le_season = make_encoders()

    # ── ML inference ───────────────────────────────────────────────────────────
    def _ml_predict(self, train_id, coach_type, wl_no, booking_day,
                    season, day_of_week, cancellation_rate,
                    quota_available, is_holiday_week) -> dict:
        pl     = self._payload
        mdl    = pl["model"]
        _imp   = pl["imputer"]
        _sca   = pl["scaler"]
        tf_    = pl["train_data"]

        row    = tf_[tf_["train_number"] == str(train_id)]
        t_dict = row.iloc[0].to_dict() if not row.empty else {}
        if not row.empty:
            train_info = {
                "name":     str(t_dict.get("train_name","")),
                "type":     str(t_dict.get("train_type","")),
                "distance": float(t_dict.get("distance_km", 0)),
                "from":     str(t_dict.get("from_station_name","")),
                "to":       str(t_dict.get("to_station_name","")),
            }
        else:
            t_dict     = {c: 0 for c in tf_.columns}
            train_info = {}

        X_in = single_row_array(
            wl_no, coach_type, booking_day, season,
            cancellation_rate, day_of_week, quota_available, is_holiday_week,
            t_dict, self._le_coach, self._le_season,
        )
        if pl["need_preprocess"]:
            X_in = _sca.transform(_imp.transform(X_in))

        prob = float(mdl.predict_proba(X_in)[0, 1])
        return self._package(prob, train_id, coach_type, wl_no,
                             booking_day, season, train_info, "ML Model")

    # ── Heuristic fallback ─────────────────────────────────────────────────────
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

    # ── Shared result builder ──────────────────────────────────────────────────
    @staticmethod
    def _package(prob, train_id, coach_type, wl_no,
                 booking_day, season, train_info, model_used) -> dict:
        band = ("High (>75%)"     if prob >= 0.75 else
                "Medium (50–75%)" if prob >= 0.50 else
                "Low (25–50%)"    if prob >= 0.25 else
                "Very Low (<25%)")

        alts = []
        if prob < 0.50:
            all_cls = list(CLASS_META.keys())
            for c in [x for x in all_cls if x != coach_type][:2]:
                alts.append(f"Try {c} class — may have better quota availability")
            if booking_day < 30:
                alts.append("Booking earlier (30+ days in advance) improves WL clearance significantly")
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
                "train_id":    train_id,
                "coach_type":  coach_type,
                "wl_no":       wl_no,
                "booking_day": booking_day,
                "season":      season,
            },
        }


# ── Module-level singleton (imported by UI pages) ─────────────────────────────
_predictor: Optional[Predictor] = None

def get_predictor() -> Predictor:
    """Return a cached Predictor instance (loads model once)."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
