"""ui/charts.py — all Plotly chart builders (dark-themed, seeded data)."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import date, timedelta

BL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,36,0.5)",
    font=dict(family="Inter", color="#8888aa", size=11),
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#55556a")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#55556a")),
)

def booking_trend(bookings_trend, wl_trend):
    dates = [date.today() - timedelta(days=i) for i in range(13, -1, -1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=bookings_trend, name="Bookings",
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        line=dict(color="#00d4ff", width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=dates, y=wl_trend, name="Waitlisted",
        fill="tozeroy", fillcolor="rgba(255,179,64,0.07)",
        line=dict(color="#ffb340", width=2, dash="dot"), mode="lines"))
    fig.update_layout(**BL, height=200, showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8888aa", size=11), x=0.01, y=0.99))
    return fig

def class_rates(conf_rates):
    classes = ["SL","3A","2A","1A","CC","EC"]
    colors  = ["#39ff85" if r>=75 else "#00d4ff" if r>=60 else "#ffb340" for r in conf_rates]
    fig = go.Figure(go.Bar(x=classes, y=conf_rates, marker_color=colors,
        text=[f"{r}%" for r in conf_rates], textposition="outside",
        textfont=dict(color="#f0f0f8", size=11)))
    fig.add_hline(y=75, line_dash="dash", line_color="rgba(57,255,133,0.35)",
        annotation_text="Target", annotation_font_color="#39ff85", annotation_font_size=10)
    yax = {**BL["yaxis"], "range":[0,105]}
    bl  = {k:v for k,v in BL.items() if k!="yaxis"}
    fig.update_layout(**bl, yaxis=yax, height=200, showlegend=False)
    return fig

def model_accuracy(model_acc):
    months = ["Sep","Oct","Nov","Dec","Jan","Feb","Mar"]
    fig = go.Figure(go.Scatter(x=months, y=model_acc, mode="lines+markers",
        line=dict(color="#39ff85", width=3),
        marker=dict(color="#39ff85", size=7, line=dict(color="#0d0d0f", width=2)),
        fill="tozeroy", fillcolor="rgba(57,255,133,0.05)"))
    yax = {**BL["yaxis"], "range":[70,90]}
    bl  = {k:v for k,v in BL.items() if k!="yaxis"}
    fig.update_layout(**bl, yaxis=yax, height=190, showlegend=False)
    return fig

def gauge(value):
    color = "#39ff85" if value >= 75 else "#ffb340" if value >= 45 else "#ff4757"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix":"%","font":{"size":48,"color":"#f0f0f8","family":"Space Grotesk"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"rgba(255,255,255,0.1)","tickfont":{"color":"#55556a","size":10}},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"rgba(34,34,47,0.8)","borderwidth":0,
            "steps":[{"range":[0,45],"color":"rgba(255,71,87,0.10)"},
                     {"range":[45,75],"color":"rgba(255,179,64,0.10)"},
                     {"range":[75,100],"color":"rgba(57,255,133,0.10)"}],
            "threshold":{"line":{"color":color,"width":3},"thickness":0.85,"value":value},
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=210, margin=dict(l=20,r=20,t=30,b=10))
    return fig

def heatmap(heatmap_demand, heatmap_wl, routes):
    rows = [{"source":s,"dest":t,"demand":heatmap_demand[i],"wl_avg":heatmap_wl[i],
             "route":f"{s} → {t}"} for i,(s,t) in enumerate(routes)]
    df = pd.DataFrame(rows)
    fig = px.treemap(df, path=["source","dest"], values="demand", color="wl_avg",
        color_continuous_scale=["#0d0d0f","#00d4ff","#39ff85"], color_continuous_midpoint=18)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=360,
        coloraxis_colorbar=dict(title="WL",tickfont=dict(color="#8888aa"),title_font=dict(color="#8888aa")))
    fig.update_traces(textfont=dict(color="#f0f0f8", size=12))
    return fig, df

def scatter_demand(df):
    fig = px.scatter(df, x="demand", y="wl_avg", text="route", size="demand",
        color="wl_avg", color_continuous_scale=["#39ff85","#ffb340","#ff4757"],
        hover_name="route", size_max=28)
    fig.update_traces(textposition="top center", textfont=dict(color="#8888aa", size=9))
    fig.update_layout(**BL, height=320, showlegend=False,
        xaxis_title="Demand Index", yaxis_title="Avg Waitlist", coloraxis_showscale=False)
    return fig

def revenue_donut(revenue_slices):
    classes = ["SL","3A","2A","1A","CC","EC","TQ"]
    total   = sum(revenue_slices)
    fig = go.Figure(go.Pie(labels=classes, values=revenue_slices, hole=0.55,
        marker=dict(colors=["#00d4ff","#39ff85","#ffb340","#ff4757","#b46eff","#ff6bba","#73ffd8"],
                    line=dict(color="rgba(0,0,0,0.4)",width=1.5)),
        textfont=dict(color="#f0f0f8", size=11)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280,
        font=dict(color="#8888aa"),
        legend=dict(font=dict(color="#8888aa"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0,r=0,t=10,b=0),
        annotations=[dict(text=f"\u20b9{total}L", x=0.5, y=0.5,
            font=dict(size=20, color="#f0f0f8", family="Space Grotesk"), showarrow=False)])
    return fig
