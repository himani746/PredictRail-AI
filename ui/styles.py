"""ui/styles.py — single CSS source. Call inject() once in app.py."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--obs:#0d0d0f;--obs2:#13131a;--obs3:#1a1a24;--obs4:#22222f;--obs5:#2a2a3a;--bd:rgba(255,255,255,0.07);--bd2:rgba(255,255,255,0.13);--neon:#39ff85;--ndim:rgba(57,255,133,0.12);--nglow:rgba(57,255,133,0.30);--cyber:#00d4ff;--cdim:rgba(0,212,255,0.12);--cglow:rgba(0,212,255,0.30);--amber:#ffb340;--adim:rgba(255,179,64,0.12);--red:#ff4757;--rdim:rgba(255,71,87,0.12);--tp:#f0f0f8;--ts:#8888aa;--th:#55556a;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--obs)!important;color:var(--tp);}
.stApp{background:var(--obs)!important;}
#MainMenu,footer{visibility:hidden;}header{visibility:visible!important;}
.block-container{padding:2rem 2.5rem 3rem!important;max-width:100%!important;}
section[data-testid="stSidebar"]>div{background:var(--obs2)!important;border-right:1px solid var(--bd);padding:1.5rem 1rem 1rem!important;}
section[data-testid="stSidebar"] div[data-testid="stButton"] button{background:transparent!important;border:1px solid transparent!important;color:var(--ts)!important;text-align:left!important;font-size:13px!important;font-weight:400!important;border-radius:8px!important;padding:9px 14px!important;margin-bottom:2px!important;width:100%!important;transition:all .15s!important;}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover{background:rgba(255,255,255,0.05)!important;color:var(--tp)!important;}
.glass{background:rgba(26,26,36,0.75);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--bd);border-radius:16px;padding:1.5rem;margin-bottom:1.25rem;transition:border-color .2s,transform .15s;}
.glass:hover{border-color:var(--bd2);}
.glass-neon{border-color:rgba(57,255,133,0.22);box-shadow:0 0 28px rgba(57,255,133,0.06);}
.glass-cyber{border-color:rgba(0,212,255,0.22);}
.glass-amber{border-color:rgba(255,179,64,0.22);}
.glass-red{border-color:rgba(255,71,87,0.22);}
.kpi{background:var(--obs3);border:1px solid var(--bd);border-radius:14px;padding:1.25rem 1.4rem;position:relative;overflow:hidden;transition:transform .15s;}
.kpi:hover{transform:translateY(-2px);}
.kpi::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;border-radius:3px 0 0 3px;}
.kpi-n::before{background:var(--neon);}.kpi-n{background:linear-gradient(135deg,rgba(57,255,133,0.04) 0%,var(--obs3) 60%);}
.kpi-c::before{background:var(--cyber);}.kpi-c{background:linear-gradient(135deg,rgba(0,212,255,0.04) 0%,var(--obs3) 60%);}
.kpi-a::before{background:var(--amber);}.kpi-a{background:linear-gradient(135deg,rgba(255,179,64,0.04) 0%,var(--obs3) 60%);}
.kpi-r::before{background:var(--red);}.kpi-r{background:linear-gradient(135deg,rgba(255,71,87,0.04) 0%,var(--obs3) 60%);}
.kpi-lbl{font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:var(--th);margin-bottom:10px;}
.kpi-val{font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:800;color:var(--tp);line-height:1;}
.kpi-d{font-size:12px;margin-top:8px;}.up{color:var(--neon);}.down{color:var(--red);}.flat{color:var(--ts);}
.ph{font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--tp);margin-bottom:4px;}
.ph .acc{color:var(--neon);}
.ps{font-size:13px;color:var(--ts);margin-bottom:2rem;}
.sl{font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:600;color:var(--tp);margin-bottom:4px;}
.ss{font-size:12px;color:var(--ts);margin-bottom:1rem;}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;letter-spacing:.4px;}
.b-n{background:var(--ndim);color:var(--neon);border:1px solid var(--nglow);}
.b-c{background:var(--cdim);color:var(--cyber);border:1px solid var(--cglow);}
.b-a{background:var(--adim);color:var(--amber);border:1px solid rgba(255,179,64,.3);}
.b-r{background:var(--rdim);color:var(--red);border:1px solid rgba(255,71,87,.3);}
.b-g{background:rgba(255,255,255,.06);color:var(--ts);border:1px solid var(--bd);}
.nd{height:1px;background:linear-gradient(90deg,transparent,var(--neon),transparent);margin:1.25rem 0;opacity:.3;}
.cd{height:1px;background:linear-gradient(90deg,transparent,var(--cyber),transparent);margin:1.25rem 0;opacity:.3;}
.pb-wrap{margin:.6rem 0;}
.pb-row{display:flex;justify-content:space-between;font-size:12px;color:var(--ts);margin-bottom:5px;}
.pb-track{height:8px;background:var(--obs4);border-radius:4px;overflow:hidden;}
.pb-fill{height:100%;border-radius:4px;}
.tl{position:relative;padding-left:28px;}
.tl::before{content:'';position:absolute;left:6px;top:8px;bottom:8px;width:2px;background:var(--obs4);border-radius:2px;}
.tl-s{position:relative;margin-bottom:22px;}
.tl-d{position:absolute;left:-28px;top:2px;width:14px;height:14px;border-radius:50%;border:2px solid var(--obs4);background:var(--obs2);}
.tl-d.done{background:var(--neon);border-color:var(--neon);box-shadow:0 0 10px var(--nglow);}
.tl-d.active{background:var(--cyber);border-color:var(--cyber);box-shadow:0 0 10px var(--cglow);animation:rp-pulse 1.8s ease-in-out infinite;}
.tl-d.pending{background:var(--obs4);border-color:var(--bd2);}
@keyframes rp-pulse{0%,100%{box-shadow:0 0 8px var(--cglow);}50%{box-shadow:0 0 22px var(--cglow);}}
.tl-lbl{font-size:13px;font-weight:600;color:var(--tp);}
.tl-meta{font-size:11px;color:var(--ts);margin-top:3px;line-height:1.5;}
.alt-card{background:var(--obs3);border:1px solid var(--bd);border-radius:12px;padding:1rem 1.25rem;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between;transition:border-color .2s,transform .15s;cursor:pointer;}
.alt-card:hover{border-color:var(--nglow);transform:translateX(3px);}
.score-h{color:var(--neon);font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:800;}
.score-m{color:var(--amber);font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:800;}
.score-l{color:var(--red);font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:800;}
.hero-pct{font-family:'Space Grotesk',sans-serif;font-size:80px;font-weight:800;line-height:1;letter-spacing:-2px;text-align:center;}
.hero-sublabel{font-size:13px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-top:8px;opacity:.75;text-align:center;}
.hero-eta{font-size:13px;color:var(--ts);margin-top:6px;text-align:center;}
.ai-says{background:linear-gradient(135deg,rgba(0,212,255,0.06),rgba(57,255,133,0.04));border:1px solid rgba(0,212,255,0.2);border-radius:14px;padding:1rem 1.25rem;margin-bottom:1rem;display:flex;gap:12px;align-items:flex-start;}
.ai-says-icon{font-size:22px;flex-shrink:0;margin-top:1px;}
.ai-says-title{font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--cyber);margin-bottom:5px;}
.ai-says-text{font-size:13px;color:var(--ts);line-height:1.65;}
.ai-says-text strong{color:var(--tp);font-weight:500;}
.hb-row{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.hb-nm{font-size:12px;color:var(--ts);width:140px;flex-shrink:0;}
.hb-trk{flex:1;height:5px;background:var(--obs4);border-radius:3px;overflow:hidden;}
.hb-fil{height:100%;border-radius:3px;}
.hb-val{font-size:12px;font-weight:600;color:var(--tp);width:36px;text-align:right;}
.dna{display:flex;gap:4px;align-items:center;}
.dna-dot{width:8px;height:8px;border-radius:50%;cursor:help;}
.decision{background:linear-gradient(135deg,rgba(57,255,133,0.06),rgba(0,212,255,0.04));border:1px solid rgba(57,255,133,0.22);border-radius:14px;padding:1.25rem 1.5rem;}
.decision-title{font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;color:var(--tp);margin-bottom:.75rem;}
.live-ticker{background:rgba(57,255,133,0.04);border:1px solid rgba(57,255,133,0.15);border-radius:10px;padding:.6rem 1rem;font-size:12px;color:var(--ts);display:flex;align-items:center;gap:8px;}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--neon);animation:rp-pulse 1.5s infinite;}
.wif-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;margin-top:.75rem;}
.wif-cell{background:var(--obs4);border:1px solid var(--bd);border-radius:10px;padding:10px;text-align:center;cursor:pointer;transition:all .15s;}
.wif-cell.wif-active{border-color:var(--nglow);background:var(--ndim);}
.wif-wl{font-size:11px;color:var(--th);margin-bottom:4px;}
.wif-pct{font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;}
.wif-bar{height:3px;border-radius:2px;margin-top:6px;}
.insight-callout{background:rgba(0,212,255,0.05);border-left:3px solid var(--cyber);border-radius:0 10px 10px 0;padding:10px 16px;margin-bottom:1.25rem;font-size:13px;color:var(--ts);line-height:1.6;}
.insight-callout strong{color:var(--cyber);}
.pf-avatar{width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,var(--neon),var(--cyber));display:flex;align-items:center;justify-content:center;font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:800;color:#000;margin:0 auto 1rem;}
.pf-name{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;text-align:center;color:var(--tp);}
.pf-sub{font-size:12px;color:var(--ts);text-align:center;margin-top:4px;}
.pf-stat{background:var(--obs4);border-radius:10px;padding:12px;text-align:center;}
.pf-sv{font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:var(--tp);}
.pf-sl{font-size:11px;color:var(--ts);margin-top:2px;}
.auth-wrap{max-width:420px;margin:0 auto;background:var(--obs2);border:1px solid var(--bd);border-radius:20px;padding:2.5rem;box-shadow:0 24px 64px rgba(0,0,0,.6);}
.auth-logo{font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:800;color:var(--tp);text-align:center;margin-bottom:6px;}
.auth-tag{font-size:13px;color:var(--ts);text-align:center;margin-bottom:1.75rem;}
.auth-title{font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--tp);margin-bottom:1.25rem;}
.sb-brand{font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--tp);display:flex;align-items:center;gap:8px;margin-bottom:1.5rem;}
.sb-pulse{width:10px;height:10px;border-radius:50%;background:var(--neon);box-shadow:0 0 10px var(--nglow);animation:rp-pulse 2s infinite;}
.nav-primary-hint{font-size:10px;color:var(--neon);letter-spacing:.8px;text-transform:uppercase;margin-bottom:3px;margin-left:2px;opacity:.7;}
div[data-testid="stTextInput"] input,div[data-testid="stSelectbox"]>div>div,div[data-testid="stDateInput"] input,div[data-testid="stNumberInput"] input{background:var(--obs3)!important;border:1px solid var(--bd2)!important;color:var(--tp)!important;border-radius:10px!important;}
div[data-testid="stTextInput"] label,div[data-testid="stSelectbox"] label,div[data-testid="stDateInput"] label,div[data-testid="stNumberInput"] label{color:var(--th)!important;font-size:11px!important;font-weight:600!important;letter-spacing:.8px!important;text-transform:uppercase!important;}
div[data-testid="stButton"]>button{background:linear-gradient(135deg,#39ff85,#00c96e)!important;color:#000!important;font-weight:700!important;border:none!important;border-radius:10px!important;font-family:'Space Grotesk',sans-serif!important;font-size:14px!important;letter-spacing:.3px!important;transition:opacity .2s,transform .15s!important;}
div[data-testid="stButton"]>button:hover{opacity:.85!important;transform:translateY(-1px)!important;}
[data-testid="stSlider"]>div>div>div{background:var(--neon)!important;}
[data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid var(--bd)!important;}
[data-testid="stTabs"] [role="tab"]{color:var(--ts)!important;font-size:13px!important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--neon)!important;border-bottom:2px solid var(--neon)!important;}
[data-testid="stMetric"]{background:var(--obs3)!important;border:1px solid var(--bd)!important;border-radius:12px!important;padding:1rem!important;}
[data-testid="stMetricLabel"]{color:var(--ts)!important;font-size:11px!important;}
[data-testid="stMetricValue"]{color:var(--tp)!important;font-family:'Space Grotesk',sans-serif!important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:var(--obs2);}::-webkit-scrollbar-thumb{background:var(--obs5);border-radius:3px;}
</style>
"""

def inject():
    """Inject all CSS. Call once at top of app.py."""
    st.markdown(CSS, unsafe_allow_html=True)
