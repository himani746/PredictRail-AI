# ⚡ RailPulse AI — PredicTrail AI

> **AI-powered waitlist prediction system for Indian Railways**  
> Know whether your waitlisted ticket will confirm — before you book.

---

## The Problem

Every day millions of Indians book train tickets and end up on a waitlist. IRCTC gives you a number — WL/12, WL/34 — and nothing else. No prediction, no guidance, no intelligence. Passengers double-book out of panic, pay extra for Tatkal, or miss travel entirely.

## The Solution

RailPulse AI predicts, with **94% accuracy**, whether a waitlisted booking will confirm. It explains *why*, shows you *what to do*, and suggests *alternative trains on the same route* — all powered by a real machine learning model trained on Indian Railways data.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Test Accuracy | **94.07%** |
| AUC-ROC | **0.987** |
| F1 Score | **0.904** |
| CV Balanced Accuracy | **93.2%** (5-fold) |
| Not Confirmed Recall | **96%** |
| Confirmed Recall | **96%** |
| Trains in database | **5,208** |
| Stations in database | **8,990** |
| Schedule stops | **417,080** |
| Features used | **31** engineered |
| Training records | **50,000** |

---

## Features

### Smart Predictor
- Search all 8,990 Indian stations by name or code
- Train dropdown auto-filters to only trains running on your selected route
- Enter waitlist number, class, season, day of week
- Get a confirmation probability with full plain-English explanation
- Three probability bars — class history, WL safety, time advantage

### What-If Simulator
- Live slider — drag to any WL position, probability updates instantly
- Snap grid showing WL/2, 5, 8, 12, 18, 25 side by side
- Colour coded: green (>75%), amber (45–75%), red (<45%)

### Route-Filtered Alternatives
- When probability is low, shows other trains on the same route
- Each alternative gets its own live model prediction
- Ranked by predicted confirmation rate

### Route Intelligence
- 15 major Indian corridors analysed in real time by the model
- Demand index, average WL pressure, confirm % at WL/10 and WL/3
- Treemap and scatter chart — all model-computed, nothing hardcoded

### Executive Dashboard
- Select any of 5,208 trains for individual analysis
- 14-day booking trend and per-class confirmation rates per train
- All KPIs (accuracy, confirmation rates, n_trains) come from the trained model
- System health monitor

### PNR Tracker
- Full journey timeline — booking → waitlist → RAC → confirmed → departed
- WL position history as a colour-coded sparkline

---

## Project Structure

```
PredictRail-AI/
│
├── app.py                  ← Single-file Streamlit application
│
├── ml/
│   └── _pipeline.py        ← Full ML training pipeline (run this first)
│
├── data/
│   ├── stations.json       ← 8,990 Indian railway stations (GeoJSON)
│   ├── trains.json         ← 5,208 trains with routes and coach info
│   └── schedules.json      ← 417,080 schedule stops
│
├── outputs/                ← Auto-created by pipeline
│   ├── best_model_real.joblib
│   ├── feature_importance_real.png
│   ├── roc_curves_real.png
│   ├── metrics_heatmap_real.png
│   ├── confusion_matrix_real.png
│   └── probability_distribution_real.png
│
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/PredictRail-AI.git
cd PredictRail-AI
pip install -r requirements.txt
```

### 2. Train the model

Run this **once** from the project root. It reads the data files, trains all models, and saves the joblib + 5 diagnostic charts.

```bash
python ml/_pipeline.py
```

Expected output:
```
  BEST MODEL : Gradient Boosting
  ACCURACY   : 0.9407  |  ✅ MEETS >80%
  ✅  best_model_real.joblib
  ✅  feature_importance_real.png
  ...
```

Training takes approximately **5–10 minutes** depending on your machine (XGBoost + Stacking Ensemble with 5-fold CV).

### 3. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

**Demo credentials:** username `demo` / password `demo123`

---

## ML Architecture

```
Input (31 features)
        │
        ├── Logistic Regression ──┐
        ├── Random Forest        ──┤──→ Stacking Meta-Learner ──→ Probability
        ├── Gradient Boosting    ──┤     (+ raw features via passthrough)
        └── XGBoost              ──┘
```

### The 31 Features

**Base features (25):**
Waitlist position, coach type, booking day, season, cancellation rate, day of week, total stops, duration, distance, train type, total seats, speed, seats/stop, quota available, is weekend, is holiday week, WL×cancel, WL/booking day, coach×season, quota/WL, dist/stop, has AC, has sleeper, premium score, zone

**Engineered features (6 — added in v2):**
- `wl_per_seat` — WL relative to train capacity
- `booking_pressure` — WL divided by available quota
- `wl_squared` — non-linear deep-WL penalty
- `days_bucket` — binned booking horizon (0–7d, 7–15d, 15–30d, 30–60d, 60d+)
- `route_popularity` — seats × cancellation rate
- `quota_ratio` — available quota as fraction of total seats

### What drove accuracy from 86% → 94%

| Change | Impact |
|--------|--------|
| Fixed class imbalance (84% → 29% confirm rate in training) | +4% recall on minority class |
| Threshold tuning per model | +2% balanced accuracy |
| 6 new engineered features | Captures non-linear WL dynamics |
| 50k records (up from 30k) | Better generalisation |
| 5-fold stratified CV (up from 3-fold) | More robust evaluation |
| `passthrough=True` in stacking | Meta-learner sees full context |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML models | scikit-learn, XGBoost |
| Data processing | pandas, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Model serialisation | joblib |
| Fonts | Sora, Outfit, IBM Plex Mono |

---

## Notes

- **XGBoost is optional** — the pipeline falls back gracefully to 3 base models if not installed, with minimal accuracy impact
- **The app works without a trained model** — it falls back to a heuristic predictor so you can explore the UI before training
- **Data files are required** for the pipeline to use real train characteristics; without them it generates synthetic data
- **All dashboard values come from the model** — no hardcoded KPIs anywhere in the app

---

## Team

**Team TwinSabers**  
Project: PredicTrail AI  
Built with Python, scikit-learn, Streamlit, and real Indian Railways data.
