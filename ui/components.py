"""
ui/components.py
Every reusable HTML/Streamlit building block used by the pages.
All page files import this as:  from ui import components as C
"""

import random
import streamlit as st


# ── Card wrappers ─────────────────────────────────────────────────────────────
def card_open(accent: str = "") -> None:
    cls = f"glass glass-{accent}" if accent else "glass"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)

def card_close() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


# ── Section headers ───────────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="sl">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ss">{subtitle}</div>', unsafe_allow_html=True)

def page_header(title_plain: str, title_accent: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ph">{title_plain} <span class="acc">{title_accent}</span></div>'
        f'<div class="ps">{subtitle}</div>',
        unsafe_allow_html=True,
    )


# ── KPI grid ──────────────────────────────────────────────────────────────────
def kpi_grid(*items) -> None:
    """items: (label, value, delta_text, delta_dir)  delta_dir: up|down|flat"""
    accents = ["n", "c", "a", "r"]
    cells   = ""
    for i, (label, value, delta, direction) in enumerate(items):
        acc    = accents[i % 4]
        cells += (
            f'<div class="kpi kpi-{acc}">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val">{value}</div>'
            f'<div class="kpi-d {direction}">{delta}</div>'
            f'</div>'
        )
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));'
        f'gap:14px;margin-bottom:1.5rem;">{cells}</div>',
        unsafe_allow_html=True,
    )


# ── Badges ────────────────────────────────────────────────────────────────────
def badge(text: str, variant: str = "g") -> str:
    """Return an HTML badge string. variant: n|c|a|r|g"""
    return f'<span class="badge b-{variant}">{text}</span>'

def badge_row(*badges) -> None:
    st.markdown(
        '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">'
        + "".join(badges) + '</div>',
        unsafe_allow_html=True,
    )


# ── Dividers ──────────────────────────────────────────────────────────────────
def neon_divider()  -> None:
    st.markdown('<div class="nd"></div>', unsafe_allow_html=True)

def cyber_divider() -> None:
    st.markdown('<div class="cd"></div>', unsafe_allow_html=True)


# ── Health bars ───────────────────────────────────────────────────────────────
def health_bar(name: str, pct: int) -> None:
    color = "#39ff85" if pct >= 95 else "#00d4ff" if pct >= 85 else "#ffb340"
    st.markdown(f"""
    <div class="hb-row">
        <div class="hb-nm">{name}</div>
        <div class="hb-trk">
            <div class="hb-fil" style="width:{pct}%;background:{color};opacity:.85;"></div>
        </div>
        <div class="hb-val">{pct}%</div>
    </div>""", unsafe_allow_html=True)


# ── Probability bars ──────────────────────────────────────────────────────────
def prob_bar(label: str, value: float, display: str, color: str,
             note: str = "") -> None:
    note_html = (f'<div style="font-size:11px;color:#55556a;margin-top:3px;">{note}</div>'
                 if note else "")
    st.markdown(f"""
    <div class="pb-wrap">
        <div class="pb-row">
            <span>{label}</span>
            <span style="color:{color};font-weight:600;">{display}</span>
        </div>
        <div class="pb-track">
            <div class="pb-fill"
                 style="width:{min(value,100):.0f}%;background:{color};opacity:.8;"></div>
        </div>
        {note_html}
    </div>""", unsafe_allow_html=True)


# ── Timeline ──────────────────────────────────────────────────────────────────
def timeline(steps: list) -> None:
    """steps: list of {label, time, state, note}  state: done|active|pending"""
    st.markdown('<div class="tl">', unsafe_allow_html=True)
    for step in steps:
        st.markdown(f"""
        <div class="tl-s">
            <div class="tl-d {step['state']}"></div>
            <div class="tl-lbl">{step['label']}</div>
            <div class="tl-meta">{step['time']} · {step['note']}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── WL position sparkline ─────────────────────────────────────────────────────
def wl_sparkline(wl_history: list) -> None:
    colors = {"CNF": "#39ff85", "RAC": "#00d4ff"}
    spans  = ""
    for i, v in enumerate(wl_history):
        if str(v) in colors:
            col = colors[str(v)]
        elif isinstance(v, int):
            col = ("#39ff85" if v <= 4 else
                   "#00d4ff" if v <= 8 else
                   "#ffb340" if v <= 15 else "#ff4757")
        else:
            col = "#8888aa"
        label = str(v) if not isinstance(v, int) else f"WL/{v}"
        arrow = " →" if i < len(wl_history) - 1 else ""
        spans += (
            f'<span style="color:{col};font-family:\'JetBrains Mono\','
            f'monospace;font-size:12px;">{label}{arrow}</span> '
        )
    st.markdown(
        f'<div style="padding:10px 12px;background:rgba(255,255,255,0.03);'
        f'border-radius:8px;margin-bottom:1.25rem;line-height:2;">{spans}</div>',
        unsafe_allow_html=True,
    )


# ── Insight callout ───────────────────────────────────────────────────────────
def insight(html_body: str) -> None:
    st.markdown(f'<div class="insight">{html_body}</div>',
                unsafe_allow_html=True)


# ── 2-column info grid ────────────────────────────────────────────────────────
def info_grid(*items) -> None:
    """items: list of (label, value) tuples"""
    cells = "".join([
        f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:11px;">'
        f'<div style="font-size:10px;color:#55556a;text-transform:uppercase;'
        f'letter-spacing:.8px;">{lbl}</div>'
        f'<div style="font-size:14px;font-weight:600;color:#f0f0f8;margin-top:4px;">'
        f'{val}</div></div>'
        for lbl, val in items
    ])
    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;'
        f'margin-bottom:1.25rem;">{cells}</div>',
        unsafe_allow_html=True,
    )


# ── AI says block ─────────────────────────────────────────────────────────────
def ai_says(text: str) -> None:
    st.markdown(f"""
    <div class="ai-says">
        <div class="ai-says-icon">⚡</div>
        <div>
            <div class="ai-says-title">RailPulse AI says</div>
            <div class="ai-says-text">{text}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── Alt train card ────────────────────────────────────────────────────────────
def alt_card(alt: dict) -> None:
    sc     = alt["score"]
    sc_cls = "score-h" if sc >= 80 else "score-m" if sc >= 65 else "score-l"
    ab_cls = "b-n" if alt["wl"] == 0 else "b-a"
    st.markdown(f"""
    <div class="alt-card">
        <div style="flex:1;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
                 font-weight:600;color:#f0f0f8;margin-bottom:3px;">
                {alt['name']}
                <span style="font-size:11px;color:#55556a;">#{alt['id']}</span>
            </div>
            <div style="font-size:12px;color:#8888aa;">
                {alt['dep']} → {alt['arr']} &nbsp;·&nbsp;
                {alt['dur']} &nbsp;·&nbsp; {alt['cls']}
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:14px;flex-shrink:0;">
            <span class="badge {ab_cls}">{alt['avail']}</span>
            <div style="text-align:right;">
                <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
                     font-weight:700;color:#f0f0f8;">{alt['fare']}</div>
                <div style="font-size:10px;color:#55556a;">per person</div>
            </div>
            <div style="text-align:center;min-width:52px;">
                <div class="{sc_cls}">{sc}</div>
                <div style="font-size:10px;color:#55556a;">score</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── Alert feed row ────────────────────────────────────────────────────────────
def alert_row(icon: str, train: str, msg: str, time_str: str) -> None:
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:12px;padding:10px 0;
         border-bottom:1px solid rgba(255,255,255,0.04);">
        <span style="font-size:14px;margin-top:1px;">{icon}</span>
        <div style="flex:1;">
            <div style="font-size:13px;font-weight:600;color:#f0f0f8;
                 margin-bottom:2px;">{train}</div>
            <div style="font-size:12px;color:#8888aa;">{msg}</div>
        </div>
        <div style="font-size:11px;color:#55556a;white-space:nowrap;">{time_str}</div>
    </div>""", unsafe_allow_html=True)
