"""ui/pages/admin.py"""
import streamlit as st
import pandas as pd
from ui import components as C
from ui import charts
from ml.predictor import get_predictor


def render():
    C.page_header("Admin", "Console",
                  "Revenue analytics · Allocation engine log · Model diagnostics")

    C.kpi_grid(
        ("Revenue Today",    "₹38.4L", "↑ 9.2%",          "up"),
        ("Avg Fare / Ticket","₹796",   "↑ ₹42",            "up"),
        ("Cancellations",    "312",    "↓ 18%",            "down"),
        ("Seats Optimised",  "1,847",  "By AI alloc engine","flat"),
    )

    tab1, tab2, tab3 = st.tabs(
        ["📈  Revenue Breakdown", "🔧  Allocation Engine", "🤖  Model Diagnostics"])

    with tab1:
        C.card_open()
        st.plotly_chart(charts.revenue_donut(), use_container_width=True)
        C.card_close()

    with tab2:
        C.card_open()
        C.section_header("Dynamic Seat Allocation Log")
        alloc = {
            "Time":    ["05:14","05:32","06:01","06:45","07:12","08:00"],
            "Train":   ["11007","12027","12123","12301","11301","11007"],
            "Action":  ["WL→RAC upgrade","RAC→CNF upgrade","Seat realloc",
                        "WL drop cleared","Tatkal seat freed","Chart locked"],
            "Seats":   [12, 4, 6, 18, 2, 0],
            "Trigger": ["Auto","Auto","Manual","Auto","Auto","Scheduled"],
        }
        st.dataframe(pd.DataFrame(alloc), use_container_width=True, hide_index=True)
        C.card_close()

    with tab3:
        predictor = get_predictor()
        C.card_open("neon")
        C.section_header("Model Performance Metrics")

        # Pull real metrics if ML model was trained
        raw_metrics = predictor.metrics
        if raw_metrics and "Accuracy" in raw_metrics:
            import pandas as pd
            df_m = pd.DataFrame(raw_metrics).T[
                ["Accuracy","Precision","Recall","F1","ROC_AUC","CV_Acc"]
            ].astype(float)
            st.dataframe(
                df_m.style.background_gradient(cmap="YlGn"),
                use_container_width=True,
            )
            st.markdown(f'<div style="margin-top:.5rem;font-size:12px;color:#8888aa;">'
                        f'Model in use: <strong style="color:#39ff85;">'
                        f'{predictor.model_name}</strong></div>',
                        unsafe_allow_html=True)
        else:
            # Show demo metrics
            metrics = {
                "Metric":    ["Accuracy","Precision","Recall","F1 Score","AUC-ROC"],
                "Train Set": [86.2, 84.7, 88.1, 86.4, 0.921],
                "Test Set":  [84.3, 82.9, 86.5, 84.7, 0.908],
                "Δ":         ["-1.9","-1.8","-1.6","-1.7","-0.013"],
            }
            st.dataframe(pd.DataFrame(metrics), use_container_width=True,
                         hide_index=True)

        C.badge_row(
            C.badge("AI Model v2.0.3",      "n"),
            C.badge("47 Booking Signals",    "c"),
            C.badge("Updated: 15 Mar 2026",  "g"),
            C.badge("1.2M Past Bookings",    "a"),
        )

        # Show ML output charts if they were generated
        from core.config import OUTPUTS
        import os
        chart_files = {
            "Feature Importance":       OUTPUTS / "feature_importance_real.png",
            "ROC Curves":               OUTPUTS / "roc_curves_real.png",
            "Metrics Heatmap":          OUTPUTS / "metrics_heatmap_real.png",
            "Confusion Matrix":         OUTPUTS / "confusion_matrix_real.png",
            "Probability Distribution": OUTPUTS / "probability_distribution_real.png",
        }
        available = {k: v for k, v in chart_files.items() if v.exists()}
        if available:
            C.neon_divider()
            st.markdown('<div class="sl">ML Pipeline Charts</div>',
                        unsafe_allow_html=True)
            sel = st.selectbox("Select chart", list(available.keys()),
                               label_visibility="collapsed")
            st.image(str(available[sel]), use_container_width=True)
        C.card_close()
