"""ui/pages/heatmap.py"""
import streamlit as st
from ui import components as C
from ui import charts


def render():
    C.page_header("Route", "Intelligence",
                  "Demand heatmap · High-pressure corridors · Dynamic allocation proof")

    C.kpi_grid(
        ("Highest Demand",    "Mumbai → Delhi", "↑ 94 demand index",  "up"),
        ("Peak Avg Waitlist", "WL / 31",        "Rajdhani corridor",  "flat"),
        ("Routes Monitored",  "15",             "Realtime via API",   "up"),
        ("Alloc Actions Today","1,847",          "By AI engine",       "up"),
    )

    fig_heat, df = charts.heatmap_treemap()

    hc1, hc2 = st.columns([3, 2])
    with hc1:
        C.card_open("cyber")
        C.section_header("Demand Treemap",
                          "Block size = demand index · Color = average waitlist intensity")
        st.plotly_chart(fig_heat, use_container_width=True)
        C.card_close()
    with hc2:
        C.card_open()
        C.section_header("Demand vs Waitlist", "Each bubble is one route corridor")
        st.plotly_chart(charts.demand_scatter(df), use_container_width=True)
        C.card_close()

    C.card_open()
    C.section_header("Top 10 High-Pressure Routes")
    top10 = df.nlargest(10, "demand")[["route","demand","wl_avg"]].reset_index(drop=True)
    top10.index += 1
    top10.columns = ["Route","Demand Index","Avg Waitlist"]
    st.dataframe(
        top10.style
             .background_gradient(subset=["Demand Index"], cmap="YlGn")
             .background_gradient(subset=["Avg Waitlist"], cmap="Reds"),
        use_container_width=True,
    )
    C.card_close()
