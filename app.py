import math
import re
import statistics
from datetime import timedelta
from pathlib import Path
from statistics import median

import pandas as pd
import requests
import streamlit as st

import model as engine

MODEL_VERSION = getattr(engine, "MODEL_VERSION", "UNKNOWN")
today_et = engine.today_et
run_model = engine.run_model
implied_prob = engine.implied_prob
expected_value = engine.expected_value
fair_ml = engine.fair_ml

def fetch_games_for_date(selected_date=None):
    """Compatibility wrapper so app.py does not crash if GitHub still has the prior model.py."""
    if hasattr(engine, "fetch_games_for_date"):
        return engine.fetch_games_for_date(selected_date)
    # Older production model only had fetch_today_games(). Keep today's slate usable.
    if hasattr(engine, "fetch_today_games") and (selected_date is None or selected_date == today_et()):
        return engine.fetch_today_games()
    raise RuntimeError(
        "Date selection requires the v1.0.3 model.py. Replace model.py in GitHub with the v1.0.3 file, then reboot the app."
    )

APP_VERSION = "3.2.3-TRACKER-LINEUP-SYNC"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT_KEY = "baseball_mlb"


st.set_page_config(page_title="Ninth Signal", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

# Streamlit fragments let free live data update without rerunning the full model.
# On older Streamlit versions, the decorator gracefully falls back to normal rendering.
def _auto_fragment(seconds):
    fragment_fn = getattr(st, "fragment", None)
    if fragment_fn is None:
        return lambda fn: fn
    return fragment_fn(run_every=f"{int(seconds)}s")

st.markdown("""
<style>
:root{--bg:#06111f;--panel:#0b1728;--panel2:#0f2035;--text:#f3f7fb;--muted:#8fa3ba;--blue:#7dd3fc;--green:#86efac;--amber:#fde68a;--red:#fda4af}
.stApp{background:radial-gradient(circle at 15% -5%,rgba(59,130,246,.17),transparent 28%),linear-gradient(180deg,#071321 0%,#06111f 55%,#050d18 100%);color:var(--text)}
.block-container{max-width:980px!important;padding-top:1rem!important;padding-bottom:4rem!important}
header[data-testid="stHeader"]{background:rgba(6,17,31,.78);backdrop-filter:blur(14px);border-bottom:1px solid rgba(148,163,184,.08)}
.hero{padding:17px 4px 9px}.eyebrow{font-size:.69rem;font-weight:950;letter-spacing:.18em;color:#7dd3fc}.title{font-size:2.35rem;font-weight:950;letter-spacing:-.055em;line-height:1;color:#fff}.sub{font-size:.86rem;color:#8fa3ba;margin-top:8px;max-width:720px}.pill{display:inline-flex;margin-top:10px;padding:5px 9px;border-radius:999px;background:rgba(34,197,94,.10);border:1px solid rgba(34,197,94,.22);color:#9ef0b6;font-size:.65rem;font-weight:900;letter-spacing:.05em}
.status{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:10px 12px;margin:8px 0 15px;border-radius:13px;background:rgba(11,23,40,.76);border:1px solid rgba(148,163,184,.10);font-size:.72rem;color:#8fa3ba}.live{color:#86efac;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#22c55e;margin-right:7px;box-shadow:0 0 0 4px rgba(34,197,94,.10)}
.kicker{font-size:.67rem;font-weight:950;letter-spacing:.14em;color:#75ccee;text-transform:uppercase;margin:16px 0 8px}
.best-card{padding:16px;margin:8px 0 14px;border-radius:18px;background:linear-gradient(145deg,rgba(15,38,60,.98),rgba(8,22,38,.99));border:1px solid rgba(34,197,94,.28);box-shadow:0 16px 40px rgba(0,0,0,.20)}.best-top{display:flex;justify-content:space-between;gap:10px}.best-tag{font-size:.62rem;font-weight:950;letter-spacing:.13em;color:#86efac}.best-pick{font-size:1.45rem;font-weight:950;color:#fff;margin-top:3px}.best-game{font-size:.72rem;color:#8fa3ba;margin-top:4px}.badge{padding:5px 8px;border-radius:999px;font-size:.60rem;font-weight:950;white-space:nowrap}.badge-best{color:#a7f3d0;background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.25)}.badge-bet{color:#bae6fd;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.22)}.badge-lean{color:#fde68a;background:rgba(234,179,8,.10);border:1px solid rgba(234,179,8,.22)}.badge-pass{color:#aebdcc;background:rgba(148,163,184,.08);border:1px solid rgba(148,163,184,.15)}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:12px}.metric{padding:8px 9px;border-radius:10px;background:rgba(255,255,255,.028);border:1px solid rgba(255,255,255,.05)}.metric span{display:block;font-size:.53rem;font-weight:900;letter-spacing:.07em;color:#677f98;text-transform:uppercase}.metric b{display:block;font-size:.78rem;color:#eaf2f9;margin-top:2px}
.game-card{margin:9px 0;padding:13px 14px;border-radius:16px;background:linear-gradient(180deg,rgba(14,29,49,.97),rgba(9,21,37,.98));border:1px solid rgba(148,163,184,.10)}.game-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.game-time{font-size:.60rem;color:#6f87a0;font-weight:850;letter-spacing:.05em}.match{font-size:.91rem;font-weight:950;color:#eef5fb;margin-top:3px}.sp{font-size:.64rem;color:#8298af;margin-top:3px}.pick{margin-top:10px;padding:10px 11px;border-radius:11px;background:rgba(5,16,30,.62);display:flex;justify-content:space-between;gap:10px;align-items:center}.pick-main{font-size:.92rem;font-weight:950;color:#f7fafc}.pick-sub{font-size:.62rem;color:#7890aa;margin-top:3px}.lineup-ok{color:#86efac}.lineup-wait{color:#fde68a}
.note{padding:11px 12px;border-radius:12px;background:rgba(59,130,246,.06);border:1px solid rgba(96,165,250,.12);color:#91a7bd;font-size:.72rem;line-height:1.45}
.single-summary{padding:13px 14px;border-radius:14px;background:rgba(15,32,53,.88);border:1px solid rgba(125,211,252,.14);margin:10px 0}.detail-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}.detail{padding:10px;border-radius:11px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05)}.detail span{display:block;font-size:.54rem;text-transform:uppercase;letter-spacing:.07em;color:#6f87a0;font-weight:900}.detail b{display:block;margin-top:3px;font-size:.82rem;color:#eef5fb}.stButton>button{width:100%;min-height:2.8rem;border-radius:11px;font-weight:850!important;background:#123252!important;color:#f8fbff!important;border:1px solid #2d5b82!important;box-shadow:none!important}.stButton>button:hover{background:#174267!important;border-color:#4c86b5!important;color:#fff!important}.stButton>button:focus{color:#fff!important}.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{background:#0f766e!important;color:#fff!important;border-color:#2dd4bf!important}.stButton>button:disabled{background:#17263a!important;color:#8fa3ba!important;border-color:#2a3a4e!important;opacity:1!important}div[data-testid="stRadio"] label,div[data-testid="stRadio"] label p,div[data-testid="stRadio"] span{color:#eef5fb!important;opacity:1!important}div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p{color:#eef5fb!important}div[data-testid="stSelectbox"] label,div[data-testid="stDateInput"] label{color:#dbeafe!important}div[data-testid="stExpander"]{border-radius:14px!important;border:1px solid rgba(148,163,184,.09)!important;background:rgba(7,18,32,.50)!important}
@media(max-width:720px){.block-container{padding-left:.72rem!important;padding-right:.72rem!important}.title{font-size:1.95rem}.metrics{grid-template-columns:repeat(2,1fr)}.detail-grid{grid-template-columns:repeat(2,1fr)}.best-pick{font-size:1.25rem}}

/* v1.4.1 readability fix */
.stButton > button,
.stDownloadButton > button {
    background: #12395f !important;
    color: #ffffff !important;
    border: 1px solid #2d6f9e !important;
    font-weight: 800 !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled {
    background: #203247 !important;
    color: #b9c8d6 !important;
    border: 1px solid #41556a !important;
    opacity: 1 !important;
}
.stButton > button p,
.stDownloadButton > button p {
    color: #ffffff !important;
}
.stButton > button:disabled p,
.stDownloadButton > button:disabled p {
    color: #b9c8d6 !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div {
    color: #dce8f2 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #e9eef4 !important;
    border: 1px solid #9eb2c4 !important;
}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div {
    color: #516273 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #ffffff !important;
    color: #203247 !important;
    border: 1px solid #c6d1dc !important;
}
[data-testid="stFileUploaderDropzone"] button p {
    color: #203247 !important;
}
.stMarkdown p,
.stCaption,
[data-testid="stCaptionContainer"] p {
    color: #aebdcc !important;
}
label,
[data-testid="stWidgetLabel"] p {
    color: #dce8f2 !important;
}
[data-testid="stExpander"] details summary p {
    color: #eaf2f9 !important;
    font-weight: 800 !important;
}


/* v1.4.2 global readability */
html, body, [class*="css"] {
    -webkit-font-smoothing: antialiased !important;
    text-rendering: optimizeLegibility !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background: #061321 !important;
    color: #edf5fb !important;
}
.block-container {
    max-width: 980px !important;
    padding-top: 1.1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-bottom: 3rem !important;
}

/* Global text */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown strong, .stMarkdown b {
    color: #f8fbff !important;
}
.stMarkdown p,
.stMarkdown li,
.stCaption,
[data-testid="stCaptionContainer"] p,
[data-testid="stText"] {
    color: #b8c7d6 !important;
    line-height: 1.55 !important;
}
small {
    color: #9fb0c1 !important;
}

/* Section labels */
.kicker {
    color: #7dd3fc !important;
    font-size: .72rem !important;
    letter-spacing: .14em !important;
    font-weight: 950 !important;
}
.live {
    color: #86efac !important;
}
.status {
    color: #b9c8d6 !important;
    background: #0a1a2b !important;
    border-color: #29425a !important;
}

/* Radio / checkbox / toggle labels */
[data-testid="stRadio"] label,
[data-testid="stRadio"] p,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] p,
[data-testid="stToggle"] label,
[data-testid="stToggle"] p {
    color: #eef5fb !important;
    opacity: 1 !important;
    font-weight: 750 !important;
}
[data-testid="stRadio"] [role="radiogroup"] {
    gap: .7rem !important;
}

/* Inputs */
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    background: #f4f7fa !important;
    color: #162638 !important;
    border-color: #afbecb !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] input {
    color: #162638 !important;
}
[data-testid="stWidgetLabel"] p,
label {
    color: #dbe7f1 !important;
    font-weight: 750 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: #174a73 !important;
    color: #ffffff !important;
    border: 1px solid #4a88b8 !important;
    font-weight: 850 !important;
    min-height: 3rem !important;
    border-radius: 13px !important;
    opacity: 1 !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1d5b8c !important;
    border-color: #6fb7e6 !important;
    color: #ffffff !important;
}
.stButton > button p,
.stDownloadButton > button p {
    color: #ffffff !important;
    opacity: 1 !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled {
    background: #24384c !important;
    color: #c8d4df !important;
    border-color: #4c6277 !important;
    opacity: 1 !important;
}
.stButton > button:disabled p,
.stDownloadButton > button:disabled p {
    color: #c8d4df !important;
    opacity: 1 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    color: #dce8f2 !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p {
    color: #dce8f2 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #edf2f6 !important;
    border: 1px solid #aab9c7 !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #42566a !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #ffffff !important;
    color: #17324a !important;
    border: 1px solid #b6c3cf !important;
}
[data-testid="stFileUploaderDropzone"] button p {
    color: #17324a !important;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #284159 !important;
    background: #081827 !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] details summary {
    color: #eef5fb !important;
}
[data-testid="stExpander"] details summary p {
    color: #eef5fb !important;
    font-weight: 800 !important;
}
[data-testid="stExpander"] svg {
    fill: #dbe7f1 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 14px !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: inherit !important;
}
div[data-testid="stAlert"][data-baseweb="notification"] {
    opacity: 1 !important;
}

/* Cards */
.best-card, .game-card {
    background: linear-gradient(180deg, #102238 0%, #0a1929 100%) !important;
    border-color: #2b4359 !important;
}
.best-pick, .match, .pick-main {
    color: #ffffff !important;
}
.best-game, .sp, .pick-sub, .game-time {
    color: #a9bac9 !important;
}
.metric {
    background: #14283d !important;
    border-color: #31495f !important;
}
.metric span {
    color: #9db0c2 !important;
}
.metric b {
    color: #f8fbff !important;
}
.pick {
    background: #071524 !important;
}

/* Badges */
.badge-best {
    color: #b6f7d0 !important;
    background: #123c2a !important;
    border-color: #2d7a53 !important;
}
.badge-bet {
    color: #d6f0ff !important;
    background: #10344b !important;
    border-color: #2877a4 !important;
}
.badge-lean {
    color: #ffe99a !important;
    background: #3a3011 !important;
    border-color: #806b17 !important;
}
.badge-pass {
    color: #cbd8e4 !important;
    background: #24313e !important;
    border-color: #495b6d !important;
}
.lineup-ok { color: #8df0b7 !important; }
.lineup-wait { color: #ffe27a !important; }

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #2a435c !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] * {
    font-size: .82rem !important;
}

/* Tabs if introduced later */
[data-baseweb="tab-list"] button {
    color: #c9d6e2 !important;
}
[data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #ffffff !important;
    font-weight: 850 !important;
}

/* Mobile tuning */
@media (max-width: 700px) {
    .block-container {
        padding-left: .75rem !important;
        padding-right: .75rem !important;
    }
    .best-pick {
        font-size: 1.25rem !important;
        line-height: 1.2 !important;
    }
    .match {
        font-size: 1rem !important;
        line-height: 1.25 !important;
    }
    .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 8px !important;
    }
    .metric {
        min-height: 66px !important;
    }
    .status {
        display: block !important;
        line-height: 1.45 !important;
    }
    .stButton > button,
    .stDownloadButton > button {
        font-size: .95rem !important;
    }
}


/* v1.5.1 combined upcoming cards */
.combo-card{
    margin:12px 0;padding:15px;border-radius:18px;
    background:linear-gradient(180deg,#102238 0%,#0a1929 100%);
    border:1px solid #2b4359;
}
.combo-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:10px}
.combo-time{font-size:.65rem;color:#8ea5ba;font-weight:850;letter-spacing:.04em}
.combo-match{font-size:1.05rem;font-weight:950;color:#fff;margin-top:3px;line-height:1.25}
.combo-sp{font-size:.68rem;color:#91a6b9;margin-top:4px}
.market-row{
    display:grid;grid-template-columns:72px 1fr auto;gap:10px;align-items:center;
    padding:11px 12px;margin-top:8px;border-radius:12px;background:#071524;border:1px solid #22394f
}
.market-name{font-size:.62rem;font-weight:950;letter-spacing:.10em;color:#7dd3fc}
.market-main{font-size:.95rem;font-weight:900;color:#fff}
.market-sub{font-size:.65rem;color:#9db0c2;margin-top:2px}
.market-grade{font-size:.62rem;font-weight:950;padding:5px 8px;border-radius:999px;white-space:nowrap}
.grade-best{color:#b6f7d0;background:#123c2a;border:1px solid #2d7a53}
.grade-bet{color:#d6f0ff;background:#10344b;border:1px solid #2877a4}
.grade-lean{color:#ffe99a;background:#3a3011;border:1px solid #806b17}
.grade-pass{color:#cbd8e4;background:#24313e;border:1px solid #495b6d}
.grade-wait{color:#cbd8e4;background:#182838;border:1px solid #3d5368}
@media(max-width:700px){
  .market-row{grid-template-columns:58px 1fr auto;gap:7px;padding:10px}
  .market-main{font-size:.88rem}
  .market-sub{font-size:.61rem}
}


/* v1.6 simple workflow */
[data-testid="stRadio"] { margin-bottom: .4rem !important; }
[data-testid="stRadio"] label { font-size: 1rem !important; }
.simple-note { color:#9db0c2;font-size:.72rem;line-height:1.45; }
@media(max-width:700px){
  .hero .sub{font-size:.92rem !important;line-height:1.45 !important;}
  .hero{padding-bottom:.35rem !important;}
  .kicker{margin-top:14px !important;margin-bottom:7px !important;}
}


/* v1.6.1 expander readability */
[data-testid="stExpander"] {
    background: #081827 !important;
    border: 1px solid #284159 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > div {
    background: #081827 !important;
    color: #eef5fb !important;
}
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details summary:hover,
[data-testid="stExpander"] details[open] summary {
    background: #0d1d2e !important;
    color: #eef5fb !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span,
[data-testid="stExpander"] details summary div {
    color: #eef5fb !important;
    opacity: 1 !important;
    font-weight: 850 !important;
}
[data-testid="stExpander"] details summary svg {
    fill: #eef5fb !important;
    color: #eef5fb !important;
}
[data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid #284159 !important;
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
}



/* v1.8 visual bet tracker */
.tracker-title-row{display:flex;align-items:center;justify-content:space-between;margin:6px 0 14px}
.tracker-title{font-size:1.55rem;font-weight:950;color:#fff;line-height:1.1}
.tracker-count{display:inline-flex;align-items:center;justify-content:center;min-width:28px;height:28px;padding:0 8px;border-radius:9px;background:#21354d;color:#fff;font-size:.82rem;margin-left:7px;vertical-align:middle}
.tracker-sub{font-size:.78rem;color:#9eafc0;margin-top:5px}
.visual-bet-card{margin:13px 0;padding:16px;border-radius:18px;background:linear-gradient(180deg,#11253b 0%,#0b1b2d 100%);border:1px solid #2f4c67;box-shadow:0 8px 24px rgba(0,0,0,.18)}
.visual-score-head{display:grid;grid-template-columns:1fr auto 58px;gap:15px;align-items:center}
.score-teams{min-width:0}
.team-row{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;color:#f8fbff;font-size:1rem;font-weight:900;line-height:1.45}
.team-row b{font-size:1.08rem;color:#fff}
.live-meta{border-left:1px solid #29445c;padding-left:13px;min-width:96px}
.live-dot-wrap{font-size:.64rem;font-weight:950;color:#86efac;letter-spacing:.03em}
.mini-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#39d98a;margin-right:6px;box-shadow:0 0 0 4px rgba(57,217,138,.08)}
.inning-meta{font-size:.66rem;color:#c3d0dc;margin-top:8px;white-space:nowrap}
.diamond-mini{width:44px;height:44px;position:relative;opacity:.55}
.diamond-mini i{position:absolute;width:14px;height:14px;border:2px solid #3f5872;transform:rotate(45deg);border-radius:2px}
.diamond-mini i:nth-child(1){left:15px;top:0}
.diamond-mini i:nth-child(2){left:0;top:15px}
.diamond-mini i:nth-child(3){right:0;top:15px}
.diamond-mini i:nth-child(4){left:15px;bottom:0}
.visual-divider{height:1px;background:#2b4359;margin:14px 0}
.bet-section-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.bet-pick{font-size:1.02rem;font-weight:950;color:#fff;line-height:1.2}
.bet-type{font-size:.68rem;color:#9eb0c1;margin-top:3px}
.track-pill{font-size:.61rem;font-weight:950;padding:6px 10px;border-radius:999px;border:1px solid;white-space:nowrap}
.track-good{color:#79edaa !important;border-color:#247a50 !important;background:#0d3526 !important}
.track-neutral{color:#f8df84 !important;border-color:#79621b !important;background:#30290f !important}
.track-risk{color:#ff7f7f !important;border-color:#8c3434 !important;background:#351717 !important}
.progress-label{display:flex;justify-content:space-between;align-items:center;margin-top:14px;font-size:.68rem;color:#a8b8c7}
.progress-label b{font-size:.85rem;color:#fff}
.run-track{height:7px;border-radius:999px;background:#2b3f55;position:relative;margin-top:7px;overflow:visible}
.run-fill{height:7px;border-radius:999px;background:#51d98a}
.run-fill.track-neutral{background:#d4b94d !important}
.run-fill.track-risk{background:#ff6666 !important}
.line-marker{position:absolute;top:-6px;width:2px;height:19px;background:#e9f0f6;border-radius:1px;transform:translateX(-1px);box-shadow:0 0 0 2px rgba(255,255,255,.07)}
.run-axis{position:relative;height:22px;margin-top:6px;color:#8195a8;font-size:.60rem}
.run-axis span:first-child{position:absolute;left:0}
.run-axis span:nth-child(2){position:absolute;transform:translateX(-50%);color:#e6edf4}
.run-axis span:last-child{position:absolute;right:0}
.ml-meter-wrap{height:44px;position:relative;margin:14px 2px 0}
.ml-meter-line{position:absolute;left:0;right:0;top:20px;height:4px;border-radius:999px;background:linear-gradient(90deg,#a94343 0%,#475c70 50%,#2d9f69 100%)}
.ml-meter-mid{position:absolute;left:50%;top:14px;width:1px;height:16px;background:#dce8f2;opacity:.55}
.ml-meter-dot{position:absolute;top:13px;width:17px;height:17px;border-radius:50%;transform:translateX(-50%);background:#51d98a;border:3px solid #dff9ea}
.ml-meter-dot.track-neutral{background:#d4b94d !important;border-color:#fff4bf !important}
.ml-meter-dot.track-risk{background:#ff6666 !important;border-color:#ffd4d4 !important}
.ml-meter-labels{display:flex;justify-content:space-between;gap:8px;color:#879aad;font-size:.58rem;margin-top:-3px}
.ml-meter-labels b{color:#b9c8d6;font-weight:750}
.plain-live-card{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;padding:12px;margin:8px 0;border-radius:14px;background:#102238;border:1px solid #31506a}
@media(max-width:700px){
  .visual-bet-card{padding:14px}
  .visual-score-head{grid-template-columns:1fr auto 42px;gap:10px}
  .team-row{font-size:.92rem}
  .live-meta{min-width:86px;padding-left:10px}
  .diamond-mini{transform:scale(.85);transform-origin:center}
  .bet-pick{font-size:.96rem}
}

/* v1.7.2 top plays */
.top-play-card{
    margin:8px 0;padding:12px 14px;border-radius:14px;
    background:#0c1d2e;border:1px solid #2b465e;
}
.top-play-rank{font-size:.60rem;font-weight:950;color:#7dd3fc;letter-spacing:.08em}
.top-play-main{font-size:.94rem;font-weight:950;color:#fff;margin-top:2px;line-height:1.22}
.top-play-sub{font-size:.64rem;color:#9fb1c2;margin-top:3px}
@media(max-width:700px){
  .top-play-card{padding:11px 12px}
  .top-play-main{font-size:.89rem}
}


/* v1.8.1 bottom navigation */
div[class*="st-key-main_navigation"] {
    position: fixed !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    z-index: 999999 !important;
    margin: 0 !important;
    padding: 8px 12px calc(8px + env(safe-area-inset-bottom)) !important;
    background: rgba(7, 21, 36, .985) !important;
    border-top: 1px solid #263e56 !important;
    box-shadow: 0 -10px 28px rgba(0,0,0,.32) !important;
}
div[class*="st-key-main_navigation"] [role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 8px !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}
div[class*="st-key-main_navigation"] label {
    min-height: 52px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 7px 3px !important;
    border: 1px solid transparent !important;
    border-radius: 13px !important;
    background: transparent !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) {
    background: #102b46 !important;
    border-color: #3477ab !important;
}
div[class*="st-key-main_navigation"] label p {
    color: #94a7ba !important;
    font-size: .72rem !important;
    font-weight: 850 !important;
    white-space: nowrap !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) p {
    color: #76c5ff !important;
}
div[class*="st-key-main_navigation"] input {
    display: none !important;
}
.block-container {
    padding-bottom: 110px !important;
}

/* actual occupied bases */
.diamond-mini i.occupied {
    background: #fbbf24 !important;
    border-color: #fbbf24 !important;
    box-shadow: 0 0 10px rgba(251,191,36,.28) !important;
}
.diamond-mini .base-second { left:15px !important; top:0 !important; }
.diamond-mini .base-third { left:0 !important; top:15px !important; }
.diamond-mini .base-first { right:0 !important; top:15px !important; }
.diamond-mini .base-home { left:15px !important; bottom:0 !important; }

/* quieter status chip, cleaner run axis */
.track-pill {
    font-size: .55rem !important;
    padding: 5px 8px !important;
}
.run-axis span:last-child {
    right: auto !important;
}


/* v1.8.2 professional bottom navigation */
div[class*="st-key-main_navigation"] label {
    min-height: 48px !important;
    border-radius: 10px !important;
}
div[class*="st-key-main_navigation"] label p {
    font-size: .74rem !important;
    letter-spacing: .02em !important;
    text-transform: uppercase !important;
    font-weight: 900 !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) {
    background: #0f2942 !important;
    border-color: #3d6f98 !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) p {
    color: #8fd0ff !important;
}


/* v1.8.3 clearer totals tracker */
.run-summary{
    display:grid !important;
    grid-template-columns:1fr 1fr !important;
    gap:10px !important;
    margin-top:18px !important;
    margin-bottom:14px !important;
}
.run-stat{
    padding:10px 12px !important;
    border-radius:12px !important;
    background:#0b1b2d !important;
    border:1px solid #29445c !important;
}
.run-stat span{
    display:block !important;
    font-size:.56rem !important;
    letter-spacing:.09em !important;
    font-weight:900 !important;
    color:#93a7ba !important;
}
.run-stat b{
    display:block !important;
    margin-top:3px !important;
    font-size:1.55rem !important;
    line-height:1 !important;
    font-weight:950 !important;
    color:#ffffff !important;
}
.line-stat b{color:#dce8f2 !important}
.clear-track{
    height:9px !important;
    margin-top:2px !important;
}
.clear-track .run-fill{height:9px !important}
.clear-track .line-marker{
    top:-7px !important;
    height:23px !important;
    width:3px !important;
    background:#ffffff !important;
}
.clear-axis{
    height:24px !important;
    margin-top:8px !important;
}
.clear-axis .line-axis-label{
    transform:translateX(-50%) !important;
    color:#ffffff !important;
    font-weight:900 !important;
    font-size:.58rem !important;
}
.bet-pick{
    font-size:1.08rem !important;
    letter-spacing:.01em !important;
}
.track-pill{
    font-size:.54rem !important;
    padding:5px 8px !important;
}
.tracker-sub{
    font-size:.72rem !important;
}
.visual-bet-card{
    padding:15px !important;
}

/* Bottom navigation: compact, no extra title-like visual weight */
div[class*="st-key-main_navigation"]{
    padding-top:6px !important;
}
div[class*="st-key-main_navigation"] [role="radiogroup"]{
    gap:6px !important;
}
div[class*="st-key-main_navigation"] label{
    min-height:44px !important;
}
div[class*="st-key-main_navigation"] label p{
    font-size:.66rem !important;
    letter-spacing:.04em !important;
}
@media(max-width:700px){
    .run-stat b{font-size:1.42rem !important}
    .bet-pick{font-size:1rem !important}
}


/* v1.9.0 premium visual system */
:root{
  --bg:#06111d;
  --panel:#0b1b2b;
  --panel2:#10253a;
  --line:#27445f;
  --text:#f7fbff;
  --muted:#93a9bd;
  --cyan:#67c7ff;
  --teal:#37d8c2;
  --green:#43e28f;
  --amber:#f4c95d;
  --red:#ff6b73;
}
[data-testid="stAppViewContainer"]{
    background:
      radial-gradient(circle at 20% -10%, rgba(33,112,170,.16), transparent 32%),
      radial-gradient(circle at 100% 15%, rgba(55,216,194,.08), transparent 28%),
      linear-gradient(180deg,#06111d 0%,#071522 100%) !important;
}
.block-container{
    max-width:920px !important;
}

/* Hero */
.hero{
    padding:10px 0 6px !important;
}
.hero h1{
    font-size:2.05rem !important;
    letter-spacing:-.035em !important;
    text-shadow:0 6px 24px rgba(0,0,0,.28);
}
.hero .sub{
    max-width:620px;
    font-size:.9rem !important;
    color:#98adbf !important;
}
.live-pill{
    box-shadow:0 0 0 1px rgba(67,226,143,.15),0 8px 30px rgba(67,226,143,.08) !important;
}

/* Main status strip */
.status{
    background:linear-gradient(180deg,rgba(15,34,54,.92),rgba(9,24,39,.95)) !important;
    border:1px solid #284762 !important;
    box-shadow:0 10px 28px rgba(0,0,0,.16) !important;
    backdrop-filter:blur(12px);
}

/* Buttons */
.stButton > button,
.stDownloadButton > button{
    background:linear-gradient(180deg,#1b547f 0%,#153f63 100%) !important;
    border:1px solid #4b8ebb !important;
    box-shadow:0 8px 22px rgba(0,0,0,.18) !important;
    transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover{
    transform:translateY(-1px) !important;
    border-color:#75b9e5 !important;
    box-shadow:0 10px 26px rgba(29,91,140,.24) !important;
}
.stButton > button[kind="primary"]{
    background:linear-gradient(135deg,#158b7f 0%,#126b75 100%) !important;
    border-color:#32d4c3 !important;
    box-shadow:0 8px 26px rgba(38,201,182,.18) !important;
}

/* Full-slate cards */
.combo-card{
    position:relative;
    overflow:hidden;
    background:
      linear-gradient(180deg,rgba(18,43,67,.98) 0%,rgba(10,27,44,.98) 100%) !important;
    border:1px solid #31516d !important;
    box-shadow:0 14px 32px rgba(0,0,0,.18) !important;
}
.combo-card::before{
    content:"";
    position:absolute;left:0;top:0;bottom:0;width:3px;
    background:linear-gradient(180deg,#5ac7ff,#32d9c4);
    opacity:.85;
}
.market-row{
    background:rgba(5,18,31,.78) !important;
    border:1px solid #24445e !important;
}
.market-name{
    color:#73cdfc !important;
}
.market-grade{
    box-shadow:0 4px 16px rgba(0,0,0,.16);
}

/* Top plays */
.top-play-card{
    position:relative;
    overflow:hidden;
    background:linear-gradient(135deg,#11263b,#0b1a2a) !important;
    border:1px solid #31516c !important;
    box-shadow:0 10px 28px rgba(0,0,0,.16) !important;
}
.top-play-card::after{
    content:"";
    position:absolute;right:-24px;top:-24px;width:78px;height:78px;border-radius:50%;
    background:radial-gradient(circle,rgba(71,199,255,.14),transparent 68%);
}

/* Tracker hero */
.tracker-hero{
    display:flex;justify-content:space-between;align-items:center;gap:14px;
    margin:4px 0 18px;padding:15px 16px;border-radius:18px;
    background:linear-gradient(135deg,#102941 0%,#0a1d30 70%);
    border:1px solid #31516c;
    box-shadow:0 16px 34px rgba(0,0,0,.18);
}
.tracker-eyebrow{
    font-size:.56rem;font-weight:950;letter-spacing:.14em;color:#69d8ca;
}
.tracker-title{
    margin-top:3px;font-size:1.65rem !important;letter-spacing:-.03em;
}
.tracker-live-orb{
    display:flex;align-items:center;gap:7px;color:#78efaa;font-size:.62rem;font-weight:950;
    border:1px solid #2f7252;background:#0c2d22;padding:7px 10px;border-radius:999px;
}
.tracker-live-orb span{
    width:7px;height:7px;border-radius:50%;background:#43e28f;
    box-shadow:0 0 0 5px rgba(67,226,143,.09),0 0 14px rgba(67,226,143,.4);
}

/* Tracked bet cards */
.visual-bet-card{
    position:relative;
    overflow:hidden;
    background:
      radial-gradient(circle at 92% 8%,rgba(91,196,255,.08),transparent 24%),
      linear-gradient(180deg,#112941 0%,#0b1c2e 100%) !important;
    border:1px solid #355773 !important;
    box-shadow:0 16px 36px rgba(0,0,0,.20) !important;
    border-radius:20px !important;
}
.visual-bet-card::before{
    content:"";
    position:absolute;left:0;right:0;top:0;height:1px;
    background:linear-gradient(90deg,transparent,#5bc9ff,transparent);
    opacity:.7;
}
.team-row{
    font-size:1.01rem !important;
}
.live-dot-wrap{
    color:#72efa6 !important;
}
.market-chip{
    display:inline-flex;align-items:center;
    margin-bottom:5px;padding:3px 7px;border-radius:999px;
    font-size:.51rem;font-weight:950;letter-spacing:.09em;
    color:#8cd7ff;background:#0d2a42;border:1px solid #28587a;
}
.bet-pick{
    font-size:1.12rem !important;
}
.run-summary{
    gap:12px !important;
}
.run-stat{
    background:linear-gradient(180deg,#0c1e30,#091827) !important;
    border:1px solid #294a64 !important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.02);
}
.run-stat b{
    font-size:1.72rem !important;
}
.clear-track{
    background:#223b52 !important;
    box-shadow:inset 0 1px 3px rgba(0,0,0,.28);
}
.run-fill{
    box-shadow:0 0 14px rgba(67,226,143,.22);
}
.run-fill.track-risk{
    box-shadow:0 0 14px rgba(255,107,115,.18);
}
.line-marker{
    box-shadow:0 0 0 2px rgba(255,255,255,.09),0 0 14px rgba(255,255,255,.22) !important;
}

/* Status pills */
.track-pill{
    font-size:.56rem !important;
    letter-spacing:.03em;
    box-shadow:0 4px 16px rgba(0,0,0,.16);
}
.track-good{
    background:linear-gradient(180deg,#0e3c2a,#0b2c20) !important;
}
.track-risk{
    background:linear-gradient(180deg,#41191c,#2f1114) !important;
}
.track-neutral{
    background:linear-gradient(180deg,#3b3112,#29220c) !important;
}

/* Functional diamond */
.diamond-mini{
    filter:drop-shadow(0 6px 12px rgba(0,0,0,.18));
}
.diamond-mini i{
    border-color:#47647e !important;
}
.diamond-mini i.occupied{
    background:#f4c95d !important;
    border-color:#f4c95d !important;
    box-shadow:0 0 12px rgba(244,201,93,.35) !important;
}

/* Expanders */
[data-testid="stExpander"]{
    box-shadow:0 10px 28px rgba(0,0,0,.12) !important;
}

/* Bottom nav */
div[class*="st-key-main_navigation"]{
    background:rgba(5,17,29,.96) !important;
    backdrop-filter:blur(18px) !important;
    border-top:1px solid #29445d !important;
}
div[class*="st-key-main_navigation"] label{
    transition:all .15s ease !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked){
    background:linear-gradient(180deg,#153650,#102b44) !important;
    border-color:#4c82aa !important;
    box-shadow:0 6px 18px rgba(0,0,0,.18) !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) p{
    color:#8fd5ff !important;
}

/* Dataframes + metrics */
.metric{
    background:linear-gradient(180deg,#132a40,#0f2235) !important;
    border-color:#31516c !important;
    box-shadow:0 8px 22px rgba(0,0,0,.12);
}

@media(max-width:700px){
    .tracker-hero{padding:13px 14px}
    .tracker-title{font-size:1.45rem !important}
    .tracker-live-orb{padding:6px 8px}
    .visual-bet-card{border-radius:18px !important}
    .run-stat b{font-size:1.55rem !important}
}


/* v1.9.1 win probability + slate pulse */
.slate-pulse{
    margin:0 0 18px;padding:14px 15px;border-radius:18px;
    background:
      radial-gradient(circle at 90% 0%,rgba(103,199,255,.10),transparent 30%),
      linear-gradient(135deg,#10263b,#0a1b2c);
    border:1px solid #31516c;
    box-shadow:0 14px 30px rgba(0,0,0,.17);
}
.pulse-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
.pulse-kicker{font-size:.52rem;letter-spacing:.12em;font-weight:950;color:#69d8ca}
.pulse-title{font-size:1.05rem;font-weight:950;color:#fff;margin-top:3px}
.pulse-status{font-size:.55rem;font-weight:950;padding:6px 8px;border-radius:999px;border:1px solid}
.pulse-good{color:#75edaa;background:#0c3324;border-color:#29734f}
.pulse-neutral{color:#f2d980;background:#30280e;border-color:#73601b}
.pulse-risk{color:#ff8589;background:#351619;border-color:#813337}
.pulse-grid{
    display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;margin-top:12px
}
.pulse-grid div{
    padding:8px 7px;border-radius:10px;background:rgba(6,18,31,.62);border:1px solid #25435b
}
.pulse-grid span{display:block;font-size:.48rem;letter-spacing:.07em;font-weight:900;color:#8fa4b7}
.pulse-grid b{display:block;margin-top:3px;font-size:.82rem;color:#fff}

.wp-wrap{
    margin-top:13px;padding:10px 11px;border-radius:12px;
    background:rgba(6,18,31,.55);border:1px solid #29465f
}
.wp-title{font-size:.50rem;letter-spacing:.10em;font-weight:950;color:#87a1b8;margin-bottom:7px}
.wp-labels{display:flex;justify-content:space-between;gap:12px;font-size:.58rem;color:#a8b8c7}
.wp-labels span{display:flex;gap:5px;align-items:baseline;min-width:0}
.wp-labels span:last-child{justify-content:flex-end;text-align:right}
.wp-labels b{font-size:.75rem;color:#fff}
.wp-track{
    position:relative;height:7px;margin-top:7px;border-radius:999px;overflow:hidden;
    background:#1d3b55
}
.wp-away{
    height:100%;background:linear-gradient(90deg,#55c9ff,#39d8c2);
    border-radius:999px 0 0 999px
}
.wp-mid{
    position:absolute;left:50%;top:-2px;width:1px;height:11px;background:rgba(255,255,255,.75)
}
.ml-live-wp{
    display:flex;justify-content:space-between;align-items:end;gap:10px;margin-top:15px
}
.ml-live-wp span{font-size:.54rem;letter-spacing:.08em;font-weight:950;color:#8fa5b8}
.ml-live-wp b{font-size:1.65rem;line-height:1;color:#fff}
.live-wp-meter{margin-top:8px !important}
.plain-live-card-wrap{
    margin:8px 0;padding:0;border-radius:14px;background:#0d2033;border:1px solid #31506a;overflow:hidden
}
.plain-live-card-wrap .plain-live-card{
    margin:0;border:0;border-radius:0;background:transparent
}
.plain-live-card-wrap .wp-wrap{
    margin:0 10px 10px
}
@media(max-width:700px){
    .pulse-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
    .pulse-grid div:last-child{grid-column:span 2}
    .ml-live-wp b{font-size:1.48rem}
}


/* v1.9.2 tracker readability + compact bottom nav */

/* Brighter positive run progress */
.run-fill.track-good,
.run-fill.track-neutral.track-good {
    background: linear-gradient(90deg,#35e08f 0%,#72f2b5 100%) !important;
    box-shadow: 0 0 16px rgba(76,235,159,.38) !important;
}
.run-fill.track-neutral {
    background: linear-gradient(90deg,#e5c84f 0%,#f2dc74 100%) !important;
}
.run-fill.track-risk {
    background: linear-gradient(90deg,#ff626c 0%,#ff8a90 100%) !important;
}

/* Make slate pulse more prominent */
.slate-pulse{
    margin: 0 0 20px !important;
    padding: 16px !important;
    border: 1px solid #3b6687 !important;
    background:
      radial-gradient(circle at 85% 0%,rgba(74,209,255,.18),transparent 34%),
      linear-gradient(135deg,#12314b 0%,#0b2135 100%) !important;
    box-shadow: 0 16px 34px rgba(0,0,0,.24) !important;
}
.pulse-title{
    font-size:1.16rem !important;
}
.pulse-grid b{
    font-size:.92rem !important;
}

/* Bottom navigation: thin app-style bar */
div[class*="st-key-main_navigation"] {
    padding: 4px 10px calc(4px + env(safe-area-inset-bottom)) !important;
    min-height: 58px !important;
}
div[class*="st-key-main_navigation"] [role="radiogroup"] {
    gap: 5px !important;
}
div[class*="st-key-main_navigation"] label {
    min-height: 40px !important;
    padding: 4px 3px !important;
    border-radius: 9px !important;
}
div[class*="st-key-main_navigation"] label p {
    font-size: .62rem !important;
    letter-spacing: .045em !important;
}
div[class*="st-key-main_navigation"] [data-testid="stWidgetLabel"],
div[class*="st-key-main_navigation"] > label,
div[class*="st-key-main_navigation"] legend {
    display: none !important;
}
.block-container {
    padding-bottom: 82px !important;
}

/* Keep cards above nav */
.visual-bet-card,
.slate-pulse,
.tracker-hero {
    position: relative;
    z-index: 1;
}


.pulse-sub{
    margin-top:3px;
    font-size:.58rem;
    color:#9eb2c5;
}


/* v2.0.0 mockup-style five-tab bottom navigation */
div[class*="st-key-main_navigation"] {
    position: fixed !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    z-index: 999999 !important;
    margin: 0 !important;
    padding: 7px 14px calc(7px + env(safe-area-inset-bottom)) !important;
    min-height: 76px !important;
    background:
      linear-gradient(180deg,rgba(8,24,40,.97),rgba(5,17,29,.995)) !important;
    border-top: 1px solid #29445d !important;
    box-shadow: 0 -12px 30px rgba(0,0,0,.30) !important;
    backdrop-filter: blur(20px) !important;
}
div[class*="st-key-main_navigation"] [role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(0,1fr)) !important;
    gap: 2px !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}
div[class*="st-key-main_navigation"] label {
    min-width: 0 !important;
    min-height: 62px !important;
    padding: 5px 2px 3px !important;
    border: 0 !important;
    border-radius: 12px !important;
    background: transparent !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 4px !important;
    transition: all .15s ease !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) {
    background: rgba(34,112,177,.10) !important;
    box-shadow: none !important;
}
div[class*="st-key-main_navigation"] input {
    display: none !important;
}
div[class*="st-key-main_navigation"] label p {
    margin: 0 !important;
    color: #7f93a8 !important;
    font-size: .58rem !important;
    font-weight: 750 !important;
    letter-spacing: .01em !important;
    text-transform: none !important;
    white-space: nowrap !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) p {
    color: #3da5ff !important;
    font-weight: 900 !important;
}

/* shared icon shell */
div[class*="st-key-main_navigation"] label::before {
    content:"" !important;
    display:block !important;
    width:25px !important;
    height:25px !important;
    background-color:#74889c !important;
    -webkit-mask-size:contain !important;
    -webkit-mask-repeat:no-repeat !important;
    -webkit-mask-position:center !important;
    mask-size:contain !important;
    mask-repeat:no-repeat !important;
    mask-position:center !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked)::before {
    background-color:#3da5ff !important;
    filter:drop-shadow(0 0 8px rgba(61,165,255,.28)) !important;
}

/* Home */
div[class*="st-key-main_navigation"] label:nth-child(1)::before {
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 10.5 12 3l9 7.5'/%3E%3Cpath d='M5 9.5V21h5v-6h4v6h5V9.5'/%3E%3C/svg%3E") !important;
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 10.5 12 3l9 7.5'/%3E%3Cpath d='M5 9.5V21h5v-6h4v6h5V9.5'/%3E%3C/svg%3E") !important;
}
/* Live */
div[class*="st-key-main_navigation"] label:nth-child(2)::before {
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='2.2'/%3E%3Cpath d='M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.4 10.4 0 0 0 0 14.6M19.3 4.7a10.4 10.4 0 0 1 0 14.6'/%3E%3C/svg%3E") !important;
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='2.2'/%3E%3Cpath d='M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.4 10.4 0 0 0 0 14.6M19.3 4.7a10.4 10.4 0 0 1 0 14.6'/%3E%3C/svg%3E") !important;
}
/* Tracker */
div[class*="st-key-main_navigation"] label:nth-child(3)::before {
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10h4v10M10 20V6h4v14M16 20V12h4v8'/%3E%3Cpath d='m4 7 5-3 4 3 7-5'/%3E%3C/svg%3E") !important;
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10h4v10M10 20V6h4v14M16 20V12h4v8'/%3E%3Cpath d='m4 7 5-3 4 3 7-5'/%3E%3C/svg%3E") !important;
}
/* Bets */
div[class*="st-key-main_navigation"] label:nth-child(4)::before {
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3' width='14' height='18' rx='2'/%3E%3Cpath d='M8 7h8M8 11h8M8 15h5'/%3E%3C/svg%3E") !important;
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3' width='14' height='18' rx='2'/%3E%3Cpath d='M8 7h8M8 11h8M8 15h5'/%3E%3C/svg%3E") !important;
}
/* Account */
div[class*="st-key-main_navigation"] label:nth-child(5)::before {
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='8' r='4'/%3E%3Cpath d='M4 21a8 8 0 0 1 16 0'/%3E%3C/svg%3E") !important;
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='8' r='4'/%3E%3Cpath d='M4 21a8 8 0 0 1 16 0'/%3E%3C/svg%3E") !important;
}

.block-container {
    padding-bottom: 104px !important;
}

/* dedicated page headers */
.page-head{margin:4px 0 16px}
.page-kicker{font-size:.54rem;font-weight:950;letter-spacing:.13em;color:#69d8ca}
.page-title{font-size:1.7rem;font-weight:950;color:#fff;letter-spacing:-.03em;margin-top:3px}
.page-count{display:inline-flex;min-width:27px;height:27px;align-items:center;justify-content:center;padding:0 7px;border-radius:8px;background:#20364f;font-size:.78rem;vertical-align:middle}
.page-sub{font-size:.72rem;color:#9db0c2;margin-top:5px}
.live-page-card{margin:10px 0;padding:14px;border-radius:17px;background:linear-gradient(180deg,#11273d,#0b1c2e);border:1px solid #34556f}
.account-card{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0 16px}
.account-card div{padding:12px;border-radius:12px;background:#0d2032;border:1px solid #29475f}
.account-card span{display:block;font-size:.52rem;font-weight:900;letter-spacing:.09em;color:#8ea4b8}
.account-card b{display:block;margin-top:4px;font-size:.72rem;color:#fff;word-break:break-word}
@media(max-width:700px){
    div[class*="st-key-main_navigation"] {padding-left:8px !important;padding-right:8px !important}
    div[class*="st-key-main_navigation"] label::before {width:23px !important;height:23px !important}
    div[class*="st-key-main_navigation"] label p {font-size:.54rem !important}
}


/* v2.0.1 — full-width native-style bottom navigation */
div[class*="st-key-main_navigation"] {
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100vw !important;
    max-width: none !important;
    min-height: 78px !important;
    padding: 8px 12px calc(8px + env(safe-area-inset-bottom)) !important;
    border-radius: 0 !important;
    background: rgba(5,17,29,.995) !important;
    border-top: 1px solid #29445d !important;
    box-shadow: 0 -10px 28px rgba(0,0,0,.30) !important;
}

/* Fill the entire bottom width instead of centering inside a constrained wrapper */
div[class*="st-key-main_navigation"] [role="radiogroup"] {
    width: 100% !important;
    max-width: none !important;
    grid-template-columns: repeat(5, 1fr) !important;
    gap: 0 !important;
    margin: 0 !important;
}

/* Remove every Streamlit radio-control visual */
div[class*="st-key-main_navigation"] input,
div[class*="st-key-main_navigation"] label > div:first-child,
div[class*="st-key-main_navigation"] [data-baseweb="radio"],
div[class*="st-key-main_navigation"] [role="radio"] > div:first-child,
div[class*="st-key-main_navigation"] svg[data-testid="stMarkdownIcon"] {
    display: none !important;
}

/* Pure tab targets: no circular control, no selected pill/card */
div[class*="st-key-main_navigation"] label {
    min-height: 58px !important;
    padding: 5px 2px 3px !important;
    margin: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Active state comes only from icon + label color, like the mockup */
div[class*="st-key-main_navigation"] label::before {
    width: 26px !important;
    height: 26px !important;
    margin-bottom: 3px !important;
    background-color: #6f8397 !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked)::before {
    background-color: #3da5ff !important;
    filter: drop-shadow(0 0 8px rgba(61,165,255,.30)) !important;
}
div[class*="st-key-main_navigation"] label p {
    color: #74889c !important;
    font-size: .58rem !important;
    font-weight: 720 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
div[class*="st-key-main_navigation"] label:has(input:checked) p {
    color: #3da5ff !important;
    font-weight: 850 !important;
}

/* Reserve exact space for the fixed bar */
.block-container {
    padding-bottom: 108px !important;
}

@media(max-width:700px){
    div[class*="st-key-main_navigation"]{
        padding-left: 4px !important;
        padding-right: 4px !important;
    }
    div[class*="st-key-main_navigation"] label::before{
        width:24px !important;
        height:24px !important;
    }
    div[class*="st-key-main_navigation"] label p{
        font-size:.55rem !important;
    }
}


/* ===== Ninth Signal v3 mobile UX ===== */
.ninth-hero{padding-top:8px!important;padding-bottom:6px!important}
.ninth-hero .title{font-size:2.15rem!important}
.ninth-hero .sub{font-size:.78rem!important;margin-top:6px!important}
.ninth-hero .pill{margin-top:9px!important}
.ninth-status{
    margin:8px 0 12px!important;
    padding:9px 11px!important;
}
.ninth-status>div{width:100%}

/* Date and refresh are compact, not the focus */
div[data-testid="stDateInput"]{margin-top:4px!important}
div[class*="st-key-refresh_scores_top"] button{
    min-height:42px!important;
    border-radius:12px!important;
    background:#102c45!important;
    border:1px solid #315d80!important;
    color:#b9d8ee!important;
    font-size:.72rem!important;
}

/* Cleaner board hierarchy */
.board-head{margin:12px 0 8px}
.board-head span{display:block;color:#74d3f7;font-size:.59rem;font-weight:950;letter-spacing:.13em}
.board-head b{display:block;color:#fff;font-size:1.14rem;margin-top:2px}

/* True segmented control for Single Game / Full Slate */
div[class*="st-key-production_view_mode"] [role="radiogroup"]{
    display:grid!important;
    grid-template-columns:1fr 1fr!important;
    gap:5px!important;
    padding:4px!important;
    border-radius:14px!important;
    background:#081a2b!important;
    border:1px solid #24435c!important;
}
div[class*="st-key-production_view_mode"] label{
    min-height:42px!important;
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    border-radius:10px!important;
    background:transparent!important;
    border:0!important;
    padding:0 8px!important;
}
div[class*="st-key-production_view_mode"] label:has(input:checked){
    background:#133451!important;
    box-shadow:inset 0 0 0 1px #3d7ca9!important;
}
div[class*="st-key-production_view_mode"] input,
div[class*="st-key-production_view_mode"] label > div:first-child,
div[class*="st-key-production_view_mode"] [data-baseweb="radio"]{
    display:none!important;
}
div[class*="st-key-production_view_mode"] label p{
    margin:0!important;
    font-size:.72rem!important;
    font-weight:850!important;
    color:#8399ac!important;
}
div[class*="st-key-production_view_mode"] label:has(input:checked) p{
    color:#fff!important;
}

/* Fixed full-width bottom tab bar using actual buttons */
div[class*="st-key-ninth_nav_"]{
    position:fixed!important;
    bottom:0!important;
    z-index:999999!important;
    width:20vw!important;
    margin:0!important;
    padding:0!important;
    background:#051522!important;
    border-top:1px solid #29465e!important;
}
div[class*="st-key-ninth_nav_board_"]{left:0!important}
div[class*="st-key-ninth_nav_live_"]{left:20vw!important}
div[class*="st-key-ninth_nav_tracker_"]{left:40vw!important}
div[class*="st-key-ninth_nav_bets_"]{left:60vw!important}
div[class*="st-key-ninth_nav_more_"]{left:80vw!important}

div[class*="st-key-ninth_nav_"] button{
    height:78px!important;
    min-height:78px!important;
    width:100%!important;
    border:0!important;
    border-radius:0!important;
    background:#051522!important;
    box-shadow:none!important;
    color:#71869a!important;
    padding:7px 1px calc(7px + env(safe-area-inset-bottom))!important;
    display:flex!important;
    flex-direction:column!important;
    justify-content:center!important;
    align-items:center!important;
    gap:5px!important;
}
div[class*="st-key-ninth_nav_"] button p{
    margin:0!important;
    font-size:.54rem!important;
    font-weight:800!important;
    line-height:1!important;
    color:inherit!important;
}
div[class*="st-key-ninth_nav_"] button::before{
    content:""!important;
    display:block!important;
    width:25px!important;
    height:25px!important;
    background-color:#70869a!important;
    -webkit-mask-size:contain!important;
    -webkit-mask-repeat:no-repeat!important;
    -webkit-mask-position:center!important;
    mask-size:contain!important;
    mask-repeat:no-repeat!important;
    mask-position:center!important;
}
div[class*="st-key-ninth_nav_"][class*="_active"] button{
    color:#46a8ff!important;
}
div[class*="st-key-ninth_nav_"][class*="_active"] button::before{
    background-color:#46a8ff!important;
    filter:drop-shadow(0 0 7px rgba(70,168,255,.28));
}
/* Board */
div[class*="st-key-ninth_nav_board_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='4' width='16' height='16' rx='2'/%3E%3Cpath d='M8 8h8M8 12h8M8 16h5'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='4' width='16' height='16' rx='2'/%3E%3Cpath d='M8 8h8M8 12h8M8 16h5'/%3E%3C/svg%3E");
}
/* Live */
div[class*="st-key-ninth_nav_live_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='2.2'/%3E%3Cpath d='M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.4 10.4 0 0 0 0 14.6M19.3 4.7a10.4 10.4 0 0 1 0 14.6'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='2.2'/%3E%3Cpath d='M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.4 10.4 0 0 0 0 14.6M19.3 4.7a10.4 10.4 0 0 1 0 14.6'/%3E%3C/svg%3E");
}
/* Tracker */
div[class*="st-key-ninth_nav_tracker_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10h4v10M10 20V6h4v14M16 20V12h4v8'/%3E%3Cpath d='m4 7 5-3 4 3 7-5'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10h4v10M10 20V6h4v14M16 20V12h4v8'/%3E%3Cpath d='m4 7 5-3 4 3 7-5'/%3E%3C/svg%3E");
}
/* Bets */
div[class*="st-key-ninth_nav_bets_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 3h12v18H6z'/%3E%3Cpath d='M9 8h6M9 12h6M9 16h4'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 3h12v18H6z'/%3E%3Cpath d='M9 8h6M9 12h6M9 16h4'/%3E%3C/svg%3E");
}
/* More */
div[class*="st-key-ninth_nav_more_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Ccircle cx='5' cy='12' r='2'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3Ccircle cx='19' cy='12' r='2'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Ccircle cx='5' cy='12' r='2'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3Ccircle cx='19' cy='12' r='2'/%3E%3C/svg%3E");
}

.block-container{padding-bottom:104px!important}
@media(max-width:700px){
    .hero{padding-left:0!important;padding-right:0!important}
    .title{font-size:2.05rem!important}
    .sub{max-width:92%!important}
}


/* ===== Ninth Signal v3.1 branded header ===== */
div[data-testid="stImage"]:has(img[src*="ninth_signal_mark"]){
    max-width:128px;
    margin:0 auto;
}
div[data-testid="stImage"]:has(img[src*="ninth_signal_mark"]) img{
    border-radius:22px;
    filter:drop-shadow(0 10px 22px rgba(0,0,0,.22));
}
.branded-hero-copy{
    padding-top:5px!important;
    padding-bottom:5px!important;
}
.branded-hero-copy .title{
    font-size:2.18rem!important;
}
.branded-hero-copy .eyebrow{
    font-size:.62rem!important;
}
.branded-hero-copy .sub{
    margin-top:6px!important;
}
@media(max-width:700px){
    div[data-testid="stHorizontalBlock"]:has(img[src*="ninth_signal_mark"]){
        gap:.5rem!important;
    }
    div[data-testid="stImage"]:has(img[src*="ninth_signal_mark"]){
        max-width:92px;
    }
    .branded-hero-copy .title{
        font-size:1.88rem!important;
    }
    .branded-hero-copy .sub{
        font-size:.70rem!important;
        line-height:1.35!important;
    }
}


/* v3.2.1 embedded brand banner */
.ninth-brand-header{
    position:relative;
    display:grid;
    grid-template-columns:128px minmax(0,1fr);
    gap:18px;
    align-items:center;
    margin:4px 0 14px;
    padding:18px 18px 18px 16px;
    border-radius:26px;
    overflow:hidden;
    background:
        radial-gradient(circle at 16% 28%, rgba(0,185,255,.30), transparent 28%),
        radial-gradient(circle at 84% 78%, rgba(0,185,255,.12), transparent 24%),
        linear-gradient(90deg, rgba(2,12,31,.98) 0%, rgba(3,23,56,.98) 48%, rgba(2,12,28,.98) 100%);
    border:1px solid rgba(71,139,255,.22);
    box-shadow:0 18px 42px rgba(0,0,0,.32), inset 0 0 0 1px rgba(255,255,255,.02);
}
.ninth-brand-header::before{
    content:"";
    position:absolute;
    inset:0;
    pointer-events:none;
    background:
        linear-gradient(135deg, transparent 0%, rgba(56,189,248,.08) 34%, transparent 35%),
        repeating-linear-gradient(90deg, transparent 0 46px, rgba(71,139,255,.05) 46px 47px);
    opacity:.55;
}
.ninth-brand-header::after{
    content:"";
    position:absolute;
    right:-74px;
    top:-72px;
    width:240px;
    height:240px;
    border-radius:50%;
    pointer-events:none;
    background:radial-gradient(circle, rgba(34,211,238,.22) 0%, rgba(34,211,238,.08) 48%, transparent 70%);
    filter:blur(8px);
}
.ninth-brand-mark{
    position:relative;
    z-index:1;
    width:128px;
    height:128px;
    display:flex;
    align-items:center;
    justify-content:center;
}
.ninth-brand-mark img{
    width:100%;
    height:100%;
    object-fit:contain;
    filter:drop-shadow(0 16px 28px rgba(0,0,0,.34));
}
.branded-hero-copy{
    position:relative;
    z-index:1;
    padding:0 !important;
}
.ninth-brand-header .eyebrow{
    font-size:.68rem !important;
    font-weight:950 !important;
    letter-spacing:.22em !important;
    color:#82ddff !important;
    margin-bottom:4px !important;
}
.ninth-brand-header .title{
    font-size:clamp(2.15rem, 5vw, 3.9rem) !important;
    line-height:.95 !important;
    letter-spacing:-.06em !important;
    font-weight:1000 !important;
    color:#f4f8ff !important;
    text-shadow:0 10px 26px rgba(0,0,0,.30);
}
.ninth-brand-header .title .signal{
    background:linear-gradient(180deg, #f7fbff 0%, #bfdcff 42%, #1da8ff 100%);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
}
.ninth-brand-header .sub{
    font-size:.92rem !important;
    color:#c8d7e8 !important;
    margin-top:8px !important;
    max-width:460px !important;
    line-height:1.42 !important;
}
.ninth-brand-header .pill{
    display:inline-flex;
    margin-top:14px !important;
    padding:8px 16px !important;
    border-radius:999px;
    background:rgba(34,197,94,.08) !important;
    border:1px solid rgba(34,197,94,.30) !important;
    color:#aef5c2 !important;
    font-size:.74rem !important;
    font-weight:950 !important;
    letter-spacing:.09em !important;
    box-shadow:0 8px 18px rgba(0,0,0,.18);
}
.ninth-full-logo{
    width:min(100%,720px);
    margin:2px auto 14px;
}
.ninth-full-logo img{
    display:block;
    width:100%;
    height:auto;
    border-radius:18px;
}
@media(max-width:700px){
    .ninth-brand-header{
        grid-template-columns:96px minmax(0,1fr);
        gap:12px;
        padding:14px 14px 14px 12px;
        border-radius:22px;
    }
    .ninth-brand-mark{
        width:96px;
        height:96px;
    }
    .ninth-brand-header .title{
        font-size:2rem !important;
    }
    .ninth-brand-header .sub{
        font-size:.82rem !important;
        max-width:100% !important;
    }
    .ninth-brand-header .pill{
        margin-top:12px !important;
        padding:7px 13px !important;
        font-size:.68rem !important;
    }
}


/* ===== Ninth Signal v3.2 automatic free data ===== */
.free-data-note{
    display:flex;
    align-items:center;
    gap:7px;
    margin:7px 0 9px;
    color:#7f96aa;
    font-size:.63rem;
    line-height:1.35;
}
.free-data-note span,
.auto-fresh span{
    flex:0 0 auto;
    width:7px;
    height:7px;
    border-radius:50%;
    background:#27d17f;
    box-shadow:0 0 0 4px rgba(39,209,127,.10);
}
.auto-fresh{
    display:flex;
    align-items:center;
    justify-content:flex-end;
    gap:7px;
    margin:8px 2px 4px;
    color:#708aa0;
    font-size:.51rem;
    font-weight:900;
    letter-spacing:.07em;
}
.auto-age{
    color:#5f788e;
    font-size:.58rem;
}


/* ===== v3.2.2 clearer lineup state ===== */
.lineup-feed-diag{
    margin-top:5px;
    color:#6f8ca6;
    font-size:.56rem;
    font-weight:750;
    letter-spacing:.015em;
}
.combo-time{
    white-space:normal !important;
}


/* ===== v3.2.3 tracker/lineup sync ===== */
.tracker-gate-diag{
    margin-top:3px;
    color:#6f8ca6;
    font-size:.54rem;
    font-weight:800;
}

</style>
""", unsafe_allow_html=True)


def team_key(name):
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    aliases = {
        "oaklandathletics":"athletics", "athletics":"athletics", "laangels":"losangelesangels",
        "losangelesangels":"losangelesangels", "dbacks":"arizonadiamondbacks",
        "arizonadiamondbacks":"arizonadiamondbacks", "whitesox":"chicagowhitesox",
        "chicagowhitesox":"chicagowhitesox", "redsox":"bostonredsox", "bostonredsox":"bostonredsox",
        "bluejays":"torontobluejays", "torontobluejays":"torontobluejays",
    }
    return aliases.get(s, s)


def valid_odds(v):
    try:
        x = float(v)
        if not math.isfinite(x) or abs(x) < 100:
            return None
        return int(round(x))
    except Exception:
        return None


def no_vig_pair(a, b):
    a, b = valid_odds(a), valid_odds(b)
    if a is None or b is None:
        return None, None
    pa, pb = implied_prob(a), implied_prob(b)
    s = pa + pb
    return (pa/s, pb/s) if s > 0 else (None, None)


@st.cache_data(ttl=75, show_spinner=False)
def fetch_odds(api_key):
    if not api_key:
        return {"events":[],"error":"ODDS_API_KEY is not configured.","quota":{}}
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/odds",
            params={"apiKey":api_key,"regions":"us","markets":"h2h","oddsFormat":"american","dateFormat":"iso"},
            timeout=25,
        )
    except requests.RequestException:
        return {"events":[],"error":"Could not reach The Odds API.","quota":{}}
    quota={"remaining":r.headers.get("x-requests-remaining"),"used":r.headers.get("x-requests-used"),"last":r.headers.get("x-requests-last")}
    if r.status_code==401:
        return {"events":[],"error":"The Odds API rejected ODDS_API_KEY (401). Update the Streamlit secret with a valid key.","quota":quota}
    if r.status_code==429:
        return {"events":[],"error":"The Odds API credit/rate limit was reached (429).","quota":quota}
    if r.status_code>=400:
        return {"events":[],"error":f"The Odds API returned HTTP {r.status_code}.","quota":quota}
    try:
        ev=r.json()
    except Exception:
        ev=[]
    return {"events":ev if isinstance(ev,list) else [],"error":"","quota":quota}



@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds_event_list(api_key):
    """Fetch current MLB event IDs only. The provider documents this endpoint as quota-free."""
    if not api_key:
        return {"events": [], "error": "ODDS_API_KEY is not configured.", "quota": {}}
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/events",
            params={"apiKey": api_key, "dateFormat": "iso"},
            timeout=25,
        )
    except requests.RequestException:
        return {"events": [], "error": "Could not reach The Odds API event list.", "quota": {}}
    quota={"remaining":r.headers.get("x-requests-remaining"),"used":r.headers.get("x-requests-used"),"last":r.headers.get("x-requests-last")}
    if r.status_code==401:
        return {"events": [], "error": "The Odds API rejected ODDS_API_KEY (401).", "quota": quota}
    if r.status_code>=400:
        return {"events": [], "error": f"The Odds API event list returned HTTP {r.status_code}.", "quota": quota}
    try:
        ev=r.json()
    except Exception:
        ev=[]
    return {"events": ev if isinstance(ev,list) else [], "error": "", "quota": quota}


def fetch_single_game_odds(api_key, game):
    """Fetch h2h odds for one explicitly selected MLB event."""
    listing=fetch_odds_event_list(api_key)
    if listing.get("error"):
        return listing
    event=match_event(listing.get("events",[]), game)
    if not event:
        return {"events": [], "error": "Could not match this MLB game to The Odds API event list yet.", "quota": listing.get("quota",{})}
    event_id=event.get("id")
    if not event_id:
        return {"events": [], "error": "Matched event did not contain an Odds API event ID.", "quota": listing.get("quota",{})}
    try:
        r=requests.get(
            f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/events/{event_id}/odds",
            params={"apiKey":api_key,"regions":"us","markets":"h2h","oddsFormat":"american","dateFormat":"iso"},
            timeout=25,
        )
    except requests.RequestException:
        return {"events": [], "error": "Could not reach The Odds API for this game.", "quota": {}}
    quota={"remaining":r.headers.get("x-requests-remaining"),"used":r.headers.get("x-requests-used"),"last":r.headers.get("x-requests-last")}
    if r.status_code==401:
        return {"events": [], "error": "The Odds API rejected ODDS_API_KEY (401).", "quota": quota}
    if r.status_code==429:
        return {"events": [], "error": "The Odds API credit/rate limit was reached (429).", "quota": quota}
    if r.status_code>=400:
        return {"events": [], "error": f"The Odds API returned HTTP {r.status_code} for this game.", "quota": quota}
    try:
        ev=r.json()
    except Exception:
        ev={}
    return {"events": [ev] if isinstance(ev,dict) and ev else [], "error": "" if ev else "No live moneyline was returned for this game.", "quota": quota}


@st.cache_data(ttl=75, show_spinner=False)
def fetch_full_slate_totals(api_key):
    if not api_key: return {"events":[],"error":"ODDS_API_KEY is not configured.","quota":{}}
    try:
        r=requests.get(f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/odds",params={"apiKey":api_key,"regions":"us","markets":"totals","oddsFormat":"american","dateFormat":"iso"},timeout=25)
    except requests.RequestException:
        return {"events":[],"error":"Could not reach The Odds API for totals.","quota":{}}
    quota={"remaining":r.headers.get("x-requests-remaining"),"used":r.headers.get("x-requests-used"),"last":r.headers.get("x-requests-last")}
    if r.status_code==401: return {"events":[],"error":"The Odds API rejected ODDS_API_KEY (401).","quota":quota}
    if r.status_code==429: return {"events":[],"error":"The Odds API credit/rate limit was reached (429).","quota":quota}
    if r.status_code>=400: return {"events":[],"error":f"The Odds API returned HTTP {r.status_code} for totals.","quota":quota}
    try: ev=r.json()
    except Exception: ev=[]
    return {"events":ev if isinstance(ev,list) else [],"error":"","quota":quota}


def fetch_single_game_totals(api_key, game):
    listing=fetch_odds_event_list(api_key)
    if listing.get("error"): return listing
    event=match_event(listing.get("events",[]),game)
    if not event: return {"events":[],"error":"Could not match this MLB game to The Odds API event list yet.","quota":listing.get("quota",{})}
    event_id=event.get("id")
    try:
        r=requests.get(f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/events/{event_id}/odds",params={"apiKey":api_key,"regions":"us","markets":"totals","oddsFormat":"american","dateFormat":"iso"},timeout=25)
    except requests.RequestException:
        return {"events":[],"error":"Could not reach The Odds API for this game's total.","quota":{}}
    quota={"remaining":r.headers.get("x-requests-remaining"),"used":r.headers.get("x-requests-used"),"last":r.headers.get("x-requests-last")}
    if r.status_code>=400: return {"events":[],"error":f"The Odds API returned HTTP {r.status_code} for this game's total.","quota":quota}
    try: ev=r.json()
    except Exception: ev={}
    return {"events":[ev] if isinstance(ev,dict) and ev else [],"error":"" if ev else "No live total was returned for this game.","quota":quota}


def totals_market(event):
    if not event: return None
    rows=[]
    for book in event.get("bookmakers",[]):
        title=book.get("title") or book.get("key") or "book"
        for m in book.get("markets",[]):
            if m.get("key")!="totals": continue
            by_point={}
            for o in m.get("outcomes",[]):
                name=str(o.get("name","")).strip().lower()
                try: point=float(o.get("point"))
                except Exception: continue
                price=valid_odds(o.get("price"))
                if price is None or name not in ("over","under"): continue
                by_point.setdefault(point,{})[name]=(price,title)
            for point,pair in by_point.items():
                if "over" in pair and "under" in pair: rows.append({"point":point,"over":pair["over"][0],"under":pair["under"][0],"book":title})
    if not rows: return None
    counts={}
    for r in rows:
        counts[r["point"]]=counts.get(r["point"],0)+1
    if not counts:
        return None
    maxn=max(counts.values())
    candidate_points=sorted([p for p,n in counts.items() if n==maxn])
    # Never average two tied market totals into a synthetic line (e.g. 8.0 and 8.5 -> 8.25).
    # Pick an actual quoted point closest to the median of all quoted book totals.
    all_points=sorted(r["point"] for r in rows)
    center=float(statistics.median(all_points))
    point=min(candidate_points,key=lambda p:(abs(float(p)-center),float(p)))
    same=[r for r in rows if abs(float(r["point"])-float(point))<1e-9]
    if not same:
        return None
    over_prices=[r["over"] for r in same if valid_odds(r.get("over")) is not None]
    under_prices=[r["under"] for r in same if valid_odds(r.get("under")) is not None]
    if not over_prices or not under_prices:
        return None
    oc=int(round(statistics.median(over_prices)))
    uc=int(round(statistics.median(under_prices)))
    ob=max(same,key=lambda r:r["over"])
    ub=max(same,key=lambda r:r["under"])
    po,pu=no_vig_pair(oc,uc)
    # Defensive fallback: a malformed book/consensus pair should never crash the full slate.
    if po is None or pu is None:
        po,pu=no_vig_pair(ob["over"],ub["under"])
    if po is None or pu is None:
        return None
    return {"total":point,"over_best":ob["over"],"under_best":ub["under"],"over_book":ob["book"],"under_book":ub["book"],
            "over_market_prob":float(po),"under_market_prob":float(pu),"books":len(same)}


def poisson_total_probs(lam,line):
    lam=max(.1,float(lam)); line=float(line); probs=[]; p=math.exp(-lam); probs.append(p)
    for k in range(1,40): p=p*lam/k; probs.append(p)
    if abs(line-round(line))<1e-9:
        n=int(round(line)); push=probs[n] if 0<=n<len(probs) else 0.; under=sum(probs[:max(n,0)]); over=max(0.,1.-under-push)
    else:
        cutoff=math.floor(line); under=sum(probs[:cutoff+1]); push=0.; over=max(0.,1.-under)
    s=over+under+push
    return (over/s,under/s,push/s) if s>0 else (.5,.5,0.)


def totals_ev(win,lose,odds):
    o=float(odds); profit=o/100. if o>0 else 100./abs(o); return float(win)*profit-float(lose)


def total_fair_ml(win,lose):
    d=float(win)+float(lose); return fair_ml(float(win)/d) if d>0 else None

TOTALS_MODEL_WEIGHT = 0.80
TOTALS_RESIDUAL_SD = 3.92
TOTALS_MAX_OFFICIAL = 3

def _normal_cdf(x, mean, sd):
    sd=max(0.25,float(sd))
    z=(float(x)-float(mean))/(sd*math.sqrt(2.0))
    return 0.5*(1.0+math.erf(z))

def production_total_probs(model_total, market_total):
    mean=float(model_total)
    line=float(market_total)
    if abs(line-round(line)) < 1e-9:
        n=int(round(line))
        under=_normal_cdf(n-0.5, mean, TOTALS_RESIDUAL_SD)
        over=1.0-_normal_cdf(n+0.5, mean, TOTALS_RESIDUAL_SD)
        push=max(0.0,1.0-over-under)
    else:
        under=_normal_cdf(line, mean, TOTALS_RESIDUAL_SD)
        over=1.0-under
        push=0.0
    s=over+under+push
    return (over/s,under/s,push/s) if s>0 else (.5,.5,0.)

def totals_grade(edge):
    edge=float(edge)
    if edge >= .125:
        return "BEST BET"
    if edge >= .075:
        return "BET"
    if edge >= .05:
        return "LEAN"
    return "PASS"

def build_total_pick(model_total, tm):
    if not tm:
        return None
    try:
        market_total=float(tm["total"])
        mpo=float(tm.get("over_market_prob"))
        mpu=float(tm.get("under_market_prob"))
    except (TypeError, ValueError, KeyError):
        return None
    if not all(math.isfinite(v) for v in (market_total,mpo,mpu)):
        return None
    calibrated_total = TOTALS_MODEL_WEIGHT*float(model_total) + (1.0-TOTALS_MODEL_WEIGHT)*market_total
    op,up,push = production_total_probs(calibrated_total, market_total)
    d=op+up
    op_np=op/d if d>0 else .5
    up_np=up/d if d>0 else .5
    oe=op_np-mpo
    ue=up_np-mpu
    if oe >= ue:
        side="OVER"; prob=op; lose=up; edge=oe; odds=tm["over_best"]; book=tm["over_book"]
    else:
        side="UNDER"; prob=up; lose=op; edge=ue; odds=tm["under_best"]; book=tm["under_book"]
    ev=totals_ev(prob,lose,odds)
    return {
        "side":side,"prob":prob,"edge":edge,"ev":ev,"odds":odds,"book":book,
        "grade":totals_grade(edge),"push":push,"calibrated_total":calibrated_total,
        "market_total":float(tm["total"]),"books":tm["books"],
        "over_prob":op,"under_prob":up,"over_edge":oe,"under_edge":ue,
        "over_odds":tm["over_best"],"under_odds":tm["under_best"],
        "over_book":tm["over_book"],"under_book":tm["under_book"],
    }


def totals_download_row(game_row, total_ctx, total_pick):
    """Flatten one totals recommendation into a CSV-friendly row."""
    out = {
        "GamePk": game_row.get("GamePk"),
        "GameDate": game_row.get("GameDate"),
        "Start_Time": game_row.get("Start_Time"),
        "Away_Team": game_row.get("Away_Team"),
        "Home_Team": game_row.get("Home_Team"),
        "Away_SP": game_row.get("Away_SP"),
        "Home_SP": game_row.get("Home_SP"),
        "Lineups_Confirmed": game_row.get("Lineups_Confirmed"),
        "Raw_Model_Total": total_ctx.get("Projected_Total"),
        "Base_Total": total_ctx.get("Base_Total"),
        "Park_Factor_Context": total_ctx.get("Park_Factor"),
        "Weather_Factor_Context": total_ctx.get("Weather_Factor"),
        "Temperature_F": total_ctx.get("Temp"),
        "Wind": total_ctx.get("Wind"),
        "Humidity": total_ctx.get("Humidity"),
        "Precip": total_ctx.get("Precip"),
    }
    if total_pick:
        out.update({
            "Market_Total": total_pick.get("market_total"),
            "Side": total_pick.get("side"),
            "Grade": total_pick.get("grade"),
            "Odds": total_pick.get("odds"),
            "Book": total_pick.get("book"),
            "Bet_Probability": total_pick.get("prob"),
            "Edge": total_pick.get("edge"),
            "EV": total_pick.get("ev"),
            "Calibrated_Total": total_pick.get("calibrated_total"),
            "Model_Weight": TOTALS_MODEL_WEIGHT,
            "Over_Probability": total_pick.get("over_prob"),
            "Under_Probability": total_pick.get("under_prob"),
            "Over_Edge": total_pick.get("over_edge"),
            "Under_Edge": total_pick.get("under_edge"),
            "Over_Odds": total_pick.get("over_odds"),
            "Under_Odds": total_pick.get("under_odds"),
            "Over_Book": total_pick.get("over_book"),
            "Under_Book": total_pick.get("under_book"),
            "Books_In_Consensus": total_pick.get("books"),
        })
    return out



def game_state(game):
    """Return PREGAME / LIVE / FINAL / OTHER.

    Safety rule: stale MLB Preview/Scheduled status can never keep a game actionable
    after its scheduled first-pitch time. Explicit delay/postponement states are the
    only exception.
    """
    if not game:
        return "OTHER"

    abstract = str(game.get("AbstractGameState") or "").strip().lower()
    detailed = str(game.get("DetailedState") or "").strip().lower()
    code = str(game.get("StatusCode") or "").strip().upper()

    # Terminal / active MLB states always win.
    if abstract == "final" or any(x in detailed for x in ("final","completed","game over")):
        return "FINAL"
    if abstract == "live" or any(x in detailed for x in ("in progress","manager challenge","review","warmup")):
        return "LIVE"

    # Explicit delay/postponement should not be forced live merely because the
    # original scheduled start time passed.
    explicit_delay = any(
        x in detailed for x in (
            "delayed", "delay", "postponed", "suspended", "rain delay",
            "weather delay", "delayed start"
        )
    )
    if explicit_delay:
        return "PREGAME"

    # Time is the hard safety gate for all other stale Preview/Scheduled states.
    try:
        start = pd.to_datetime(game.get("GameDate"), utc=True)
        now = pd.Timestamp.now(tz="UTC")
        if now >= start:
            return "LIVE"
        return "PREGAME"
    except Exception:
        pass

    # Only fall back to MLB Preview/Scheduled if time parsing failed.
    if abstract == "preview" or code in {"S","P"} or any(x in detailed for x in ("scheduled","pre-game","pregame")):
        return "PREGAME"

    return "OTHER"

def is_pregame(game):
    return game_state(game) == "PREGAME"

def game_state_label(game):
    state = game_state(game)
    if state == "LIVE":
        return "STARTED / LIVE — betting recommendations disabled"
    if state == "FINAL":
        return "FINAL — betting recommendations disabled"
    if state == "PREGAME":
        return "PREGAME"
    return "STATUS UNKNOWN"




@st.cache_data(ttl=60, show_spinner=False)
def fetch_fresh_lineup_counts(game_pk):
    """Read current batting-order counts from MLB's free live game feed."""
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live",
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        teams = ((data.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        away_order = ((teams.get("away") or {}).get("battingOrder") or [])[:9]
        home_order = ((teams.get("home") or {}).get("battingOrder") or [])[:9]
        away_count = len(away_order)
        home_count = len(home_order)
        return {
            "away_count": away_count,
            "home_count": home_count,
            "away_ready": away_count >= 8,
            "home_ready": home_count >= 8,
            "teams_ready": int(away_count >= 8) + int(home_count >= 8),
        }
    except Exception:
        return {
            "away_count": 0,
            "home_count": 0,
            "away_ready": False,
            "home_ready": False,
            "teams_ready": 0,
        }


def lineup_feed_status(game_pk):
    snap = fetch_fresh_lineup_counts(game_pk)
    ready = int(snap.get("teams_ready", 0))
    label = "LINEUPS CONFIRMED" if ready >= 2 else f"AWAITING LINEUPS • {ready}/2"
    return label, snap


def _game_hours_to_start(game):
    try:
        start = pd.to_datetime(game.get("GameDate"), utc=True)
        now = pd.Timestamp.now(tz="UTC")
        return float((start - now).total_seconds() / 3600.0)
    except Exception:
        return None


def current_lineup_snapshot(games, max_hours=5.0):
    """Check only near-term pregame games so free MLB traffic stays modest."""
    out = {}
    for g in games or []:
        if not is_pregame(g):
            continue
        hrs = _game_hours_to_start(g)
        if hrs is not None and (hrs < -0.25 or hrs > max_hours):
            continue
        gp = g.get("GamePk")
        if gp is None:
            continue
        out[str(gp)] = fetch_fresh_lineup_counts(gp)
    return out


@_auto_fragment(60)
def lineup_auto_refresh_watcher(games):
    """Refresh free lineup data every 60s and rerun the model only after a real lineup change."""
    snapshot = current_lineup_snapshot(games, max_hours=5.0)
    compact = {
        k: (
            int(v.get("away_count", 0)),
            int(v.get("home_count", 0)),
            int(v.get("teams_ready", 0)),
        )
        for k, v in snapshot.items()
    }
    previous = st.session_state.get("_ninth_lineup_snapshot")
    st.session_state["_ninth_lineup_snapshot"] = compact

    # Once MLB posts or changes a lineup, rerun the complete app. The production
    # engine clears its dynamic feed cache on run_model(), so lineup adjustments
    # and Tracker qualification are recalculated immediately.
    if previous is not None and compact != previous:
        st.rerun()

@st.cache_data(ttl=20, show_spinner=False)
def fetch_fresh_scoreboard(date_text):
    """Fetch fresh MLB scores/innings directly from the free schedule endpoint.

    This intentionally bypasses model.py's long-lived JSON cache so live scores update.
    """
    try:
        day = pd.Timestamp(date_text).date().strftime("%Y-%m-%d")
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": day, "hydrate": "linescore"},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    board = {}
    for block in data.get("dates", []):
        for g in block.get("games", []):
            teams = g.get("teams", {}) or {}
            away = teams.get("away", {}) or {}
            home = teams.get("home", {}) or {}
            linescore = g.get("linescore", {}) or {}
            status = g.get("status", {}) or {}
            board[str(g.get("gamePk"))] = {
                "GamePk": g.get("gamePk"),
                "Away": (away.get("team", {}) or {}).get("name"),
                "Home": (home.get("team", {}) or {}).get("name"),
                "Away_Score": away.get("score"),
                "Home_Score": home.get("score"),
                "Current_Inning": linescore.get("currentInning"),
                "Current_Inning_Ordinal": linescore.get("currentInningOrdinal"),
                "Inning_State": linescore.get("inningState"),
                "Inning_Half": linescore.get("inningHalf"),
                "Outs": linescore.get("outs"),
                "On_First": bool((linescore.get("offense", {}) or {}).get("first")),
                "On_Second": bool((linescore.get("offense", {}) or {}).get("second")),
                "On_Third": bool((linescore.get("offense", {}) or {}).get("third")),
                "AbstractGameState": status.get("abstractGameState", ""),
                "DetailedState": status.get("detailedState", ""),
                "StatusCode": status.get("statusCode", ""),
                "GameDate": g.get("gameDate"),
            }
    return board

def live_score_text(game):
    if not game:
        return ""
    away = game.get("Away") or "Away"
    home = game.get("Home") or "Home"
    a = game.get("Away_Score")
    h = game.get("Home_Score")
    if a is None or h is None:
        return f"{away} @ {home}"
    try:
        return f"{away} {int(a)} — {home} {int(h)}"
    except Exception:
        return f"{away} {a} — {home} {h}"

def inning_status_text(game):
    """Human-friendly inning/outs label using the free MLB linescore."""
    if not game:
        return ""
    state = game_state(game)
    if state == "FINAL":
        return "FINAL"
    if state != "LIVE":
        return game_state_label(game)

    ordinal = game.get("Current_Inning_Ordinal")
    inning = game.get("Current_Inning")
    inning_state = str(game.get("Inning_State") or "").strip()
    inning_half = str(game.get("Inning_Half") or "").strip()

    # Prefer MLB's inningState (Top/Middle/Bottom/End) when present.
    half = inning_state or inning_half
    if not half and inning:
        half = f"Inning {inning}"

    if ordinal and half:
        if ordinal.lower() not in half.lower():
            label = f"{half} {ordinal}"
        else:
            label = half
    elif ordinal:
        label = str(ordinal)
    elif half:
        label = half
    elif inning:
        label = f"Inning {inning}"
    else:
        label = "LIVE"

    outs = game.get("Outs")
    try:
        outs = int(outs)
        if outs >= 0 and str(half).lower() not in ("middle", "end"):
            label += f" • {outs} out" + ("" if outs == 1 else "s")
    except Exception:
        pass
    return label


def event_match(event, game):
    if not is_pregame(game):
        return None
    if team_key(event.get("away_team")) != team_key(game.get("Away")) or team_key(event.get("home_team")) != team_key(game.get("Home")):
        return None
    try:
        e=pd.to_datetime(event.get("commence_time"),utc=True); g=pd.to_datetime(game.get("GameDate"),utc=True)
        return abs((e-g).total_seconds())
    except Exception:
        return 0


def match_event(events, game):
    c=[]
    for e in events:
        s=event_match(e,game)
        if s is not None: c.append((s,e))
    if not c: return None
    c.sort(key=lambda z:z[0]); return c[0][1]


def moneyline_market(event):
    if not event: return None
    away_k, home_k = team_key(event.get("away_team")), team_key(event.get("home_team"))
    prices={away_k:[],home_k:[]}
    books={away_k:[],home_k:[]}
    updates=[]
    for book in event.get("bookmakers",[]):
        title=book.get("title") or book.get("key") or "book"
        for m in book.get("markets",[]):
            if m.get("key")!="h2h": continue
            if m.get("last_update"): updates.append(m.get("last_update"))
            for o in m.get("outcomes",[]):
                k=team_key(o.get("name")); p=valid_odds(o.get("price"))
                if k in prices and p is not None:
                    prices[k].append(p); books[k].append((p,title))
    if not prices[away_k] or not prices[home_k]: return None
    away_cons=int(round(statistics.median(prices[away_k]))); home_cons=int(round(statistics.median(prices[home_k])))
    away_best=max(books[away_k], key=lambda x:x[0]); home_best=max(books[home_k], key=lambda x:x[0])
    return {
        "away_consensus":away_cons,"home_consensus":home_cons,
        "away_best":away_best[0],"home_best":home_best[0],"away_book":away_best[1],"home_book":home_best[1],
        "books":min(len(prices[away_k]),len(prices[home_k])),"last_update":max(updates) if updates else None,
    }


def model_alpha(confidence, lineup_confirmed):
    # Research champion selected ~70% model weight. Production starts slightly more conservative until lineups are confirmed.
    a = 0.70 if lineup_confirmed else 0.60
    if confidence < 70: a -= 0.10
    elif confidence < 80: a -= 0.05
    return max(0.45,min(0.70,a))


def thresholds(odds):
    o=float(odds); b_edge,b_ev,a_edge,a_ev=.025,.045,.045,.075
    if o<=-200: b_edge+=.010; b_ev+=.010; a_edge+=.010; a_ev+=.015
    if o>=300: b_edge+=.015; b_ev+=.025; a_edge+=.020; a_ev+=.035
    return b_edge,b_ev,a_edge,a_ev


def grade(prob, odds, confidence, lineup_confirmed):
    imp=implied_prob(odds); edge=prob-imp; ev=expected_value(prob,odds)
    b_edge,b_ev,a_edge,a_ev=thresholds(odds)
    # Official bets require known starters. Unconfirmed lineups may still qualify, but need stronger confidence.
    official_conf = 78 if lineup_confirmed else 82
    if odds>=500: verdict="PASS"
    elif confidence>=official_conf and edge>=a_edge and ev>=a_ev: verdict="BEST BET"
    elif confidence>=max(70,official_conf-8) and edge>=b_edge and ev>=b_ev: verdict="BET"
    elif edge>=.010 and ev>=.015: verdict="LEAN"
    else: verdict="PASS"
    if odds>=300 and verdict in ("BEST BET","BET"): verdict="LEAN"
    return verdict,edge,ev,imp



def smart_card_label(side, confidence, lineup_confirmed):
    """Edge-driven selection layer; model probabilities/calibration stay unchanged."""
    if side.get("odds") is None or side.get("edge") is None or side.get("ev") is None:
        return "MODEL ONLY"

    odds=float(side["odds"])
    edge=float(side["edge"])
    legacy=side.get("verdict","PASS")

    # Preserve hard production rejections (invalid/very long prices, etc.).
    if legacy == "PASS":
        return "PASS"

    # Thin historical sample for +200 and longer dogs: require materially more edge.
    if odds >= 200:
        if lineup_confirmed and confidence >= 82 and edge >= .15 and legacy in ("BEST BET","BET"):
            return "BEST BET"
        if lineup_confirmed and confidence >= 80 and edge >= .12 and legacy in ("BEST BET","BET"):
            return "BET"
        if edge >= .075:
            return "LEAN"
        return "PASS"

    # Frozen price-bucket audit supports edge as the primary gate.
    # 10%+ = strongest zone, 7.5–10% = bettable, 5–7.5% = lean, <5% = pass.
    if legacy in ("BEST BET","BET") and edge >= .10:
        return "BEST BET"
    if legacy in ("BEST BET","BET") and edge >= .075:
        return "BET"
    if edge >= .05:
        return "LEAN"
    return "PASS"


def smart_score(side, confidence):
    if side.get("edge") is None or side.get("ev") is None:
        return -999.0
    return float(side["edge"])*100 + float(side["ev"])*35 + max(0, confidence-70)*0.03

def cls(v):
    return {"BEST BET":"badge-best","BET":"badge-bet","LEAN":"badge-lean","PASS":"badge-pass","MODEL ONLY":"badge-lean"}.get(v,"badge-pass")


def build_candidates(model_df, games, events):
    """Build one candidate for every modeled MLB game."""
    game_map={g.get("GamePk"):g for g in games}
    out=[]
    for _,r in model_df.iterrows():
        g=game_map.get(r["GamePk"],{})
        event=match_event(events,g) if is_pregame(g) else None
        m=moneyline_market(event) if is_pregame(g) else None
        engine_confirmed=bool(r["Away_Lineup_Used"] and r["Home_Lineup_Used"])
        lineup_label, lineup_feed = lineup_feed_status(r["GamePk"]) if is_pregame(g) else ("", {})
        feed_confirmed = bool(lineup_feed.get("teams_ready", 0) >= 2)
        # One source of truth for Board grading and Tracker qualification:
        # confirmed if either the engine has loaded both lineups OR the fresh MLB
        # batting-order feed shows both teams ready.
        confirmed = bool(engine_confirmed or feed_confirmed)
        conf=int(r["Model_Confidence"])
        alpha=model_alpha(conf,confirmed)
        market_available=False
        side_rows=[]

        if m:
            am,hm=no_vig_pair(m["away_consensus"],m["home_consensus"])
            if am is not None and hm is not None:
                market_available=True
                sides=[
                    (r["Away"],float(r["Away_WinProb"]),am,m["away_best"],m["away_book"]),
                    (r["Home"],float(r["Home_WinProb"]),hm,m["home_best"],m["home_book"]),
                ]
                for team,raw,market_p,price,book in sides:
                    cal=market_p+alpha*(raw-market_p); cal=max(.001,min(.999,cal))
                    verdict,edge,ev,imp=grade(cal,price,conf,confirmed)
                    side_rows.append({"team":team,"raw":raw,"market_prob":market_p,"prob":cal,"odds":price,"book":book,"verdict":verdict,"edge":edge,"ev":ev,"fair":fair_ml(cal)})

        if not market_available:
            side_rows=[
                {"team":r["Away"],"raw":float(r["Away_WinProb"]),"market_prob":None,"prob":float(r["Away_WinProb"]),"odds":None,"book":None,"verdict":"MODEL ONLY","edge":None,"ev":None,"fair":fair_ml(float(r["Away_WinProb"]))},
                {"team":r["Home"],"raw":float(r["Home_WinProb"]),"market_prob":None,"prob":float(r["Home_WinProb"]),"odds":None,"book":None,"verdict":"MODEL ONLY","edge":None,"ev":None,"fair":fair_ml(float(r["Home_WinProb"]))},
            ]

        rank={"BEST BET":5,"BET":4,"LEAN":3,"MODEL ONLY":2,"PASS":1}
        side_rows.sort(key=lambda x:(rank[x["verdict"]], x["edge"] if x["edge"] is not None else -999, x["ev"] if x["ev"] is not None else -999, x["prob"]),reverse=True)
        for z in side_rows:
            z["selection"] = smart_card_label(z, conf, confirmed)
            z["smart_score"] = smart_score(z, conf)
        selection_rank={"BEST BET":5,"BET":4,"LEAN":3,"MODEL ONLY":2,"PASS":1}
        side_rows.sort(key=lambda z:(selection_rank.get(z.get("selection"),0), z.get("smart_score",-999), z.get("prob",0)), reverse=True)
        best=side_rows[0]
        out.append({
            "GamePk":r["GamePk"],"game":r["Game"],"away":r["Away"],"home":r["Home"],"time":r.get("TimeLabel",g.get("TimeLabel","")),
            "away_sp":r.get("Away_SP") or "TBD","home_sp":r.get("Home_SP") or "TBD","lineup_confirmed":confirmed,
            "lineup_display": ("LINEUPS CONFIRMED" if confirmed else lineup_label),
            "engine_lineup_confirmed": engine_confirmed,
            "feed_lineup_confirmed": feed_confirmed,
            "away_lineup_count": int(lineup_feed.get("away_count", 0)),
            "home_lineup_count": int(lineup_feed.get("home_count", 0)),
            "lineup_teams_ready": int(lineup_feed.get("teams_ready", 0)),
            "confidence":conf,"alpha":alpha,"books":m["books"] if market_available else 0,"best":best,"all":side_rows,
            "away_proj":float(r["Away_Proj_Runs"]),"home_proj":float(r["Home_Proj_Runs"]),
            "lineup_status":r["Lineup_Status"],"confidence_reasons":r.get("Confidence_Reasons",""),"market_available":market_available,
            "model_row": r.to_dict(),
            "game_state": game_state(g),
            "pregame": is_pregame(g),
        })
    order={"BEST BET":5,"BET":4,"LEAN":3,"MODEL ONLY":2,"PASS":1}
    out.sort(key=lambda x:(order.get(x["best"].get("selection"),0), x["best"].get("smart_score",-999)),reverse=True)
    return out


def _diag_num(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def game_diagnostics_df(x, slate_date):
    """One-row export with the live model inputs/outputs for a selected game."""
    r = dict(x.get("model_row") or {})
    away = next(z for z in x["all"] if z["team"] == x["away"])
    home = next(z for z in x["all"] if z["team"] == x["home"])
    row = {
        "Slate_Date": str(slate_date),
        "GamePk": x.get("GamePk"),
        "Game": x.get("game"),
        "Time": x.get("time"),
        "Away": x.get("away"),
        "Home": x.get("home"),
        "Away_SP": x.get("away_sp"),
        "Home_SP": x.get("home_sp"),
        "Lineup_Status": x.get("lineup_status"),
        "Lineups_Confirmed": bool(x.get("lineup_confirmed") or x.get("feed_lineup_confirmed") or int(x.get("lineup_teams_ready") or 0) >= 2),
        "Model_Confidence": x.get("confidence"),
        "Confidence_Reasons": x.get("confidence_reasons"),
        "Market_Available": x.get("market_available"),
        "Model_Weight": x.get("alpha") if x.get("market_available") else None,
        "Market_Weight": (1-x.get("alpha")) if x.get("market_available") else None,
        "Books_In_Consensus": x.get("books"),
        "Away_Raw_Model_Prob": away.get("raw"),
        "Home_Raw_Model_Prob": home.get("raw"),
        "Away_Calibrated_Prob": away.get("prob") if x.get("market_available") else None,
        "Home_Calibrated_Prob": home.get("prob") if x.get("market_available") else None,
        "Away_Market_NoVig_Prob": away.get("market_prob"),
        "Home_Market_NoVig_Prob": home.get("market_prob"),
        "Away_Best_Odds": away.get("odds"),
        "Home_Best_Odds": home.get("odds"),
        "Away_Best_Book": away.get("book"),
        "Home_Best_Book": home.get("book"),
        "Away_Edge": away.get("edge"),
        "Home_Edge": home.get("edge"),
        "Away_EV": away.get("ev"),
        "Home_EV": home.get("ev"),
        "Away_Fair_ML": away.get("fair"),
        "Home_Fair_ML": home.get("fair"),
        "Away_Card_Label": away.get("selection"),
        "Home_Card_Label": home.get("selection"),
        "Away_Proj_Runs": x.get("away_proj"),
        "Home_Proj_Runs": x.get("home_proj"),
        "Model_Version": MODEL_VERSION,
        "App_Version": APP_VERSION,
    }
    # Preserve the most useful engine-level diagnostic inputs when available.
    wanted = [
        "Away_SP_Hand","Home_SP_Hand","Away_SP_Quality","Home_SP_Quality",
        "Away_SP_Starts","Home_SP_Starts","Away_SP_SeasonERA","Home_SP_SeasonERA",
        "Away_SP_SeasonFIP","Home_SP_SeasonFIP","Away_SP_RecentERA","Home_SP_RecentERA",
        "Away_SP_RecentFIP","Home_SP_RecentFIP","Away_SP_ExpIP","Home_SP_ExpIP",
        "Away_Base_Offense","Home_Base_Offense","Away_Platoon_Factor","Home_Platoon_Factor",
        "Away_Lineup_Factor","Home_Lineup_Factor","Away_Lineup_Used","Home_Lineup_Used",
        "Away_Offense","Home_Offense",
    ]
    for k in wanted:
        if k in r:
            row[k] = _diag_num(r.get(k))
    return pd.DataFrame([row])


def slate_export_df(candidates):
    rows=[]
    for x in candidates:
        b=x["best"]
        rows.append({
            "Game":x["game"],"Time":x["time"],
            "Pick":f"{b['team']} ML" if x["market_available"] else f"{b['team']} model lean",
            "Odds":b["odds"],"Book":b["book"],"Edge_Driven_Card":b.get("selection"),
            "Legacy_Grade":b["verdict"],"Calibrated_Prob":b["prob"] if x["market_available"] else None,
            "Model_Prob":b["raw"],"Edge":b["edge"],"EV":b["ev"],"Fair_ML":b["fair"],
            "Confidence":x["confidence"],"Lineups_Confirmed":x["lineup_confirmed"],
            "Market_Available":x["market_available"],"Model_Weight":x["alpha"] if x["market_available"] else None,
            "Model_Version":MODEL_VERSION
        })
    return pd.DataFrame(rows)


TRACKER_DIR = Path(".mlb_tracker")
TRACKER_PATH = TRACKER_DIR / "model_recommendations.csv"
TRACKER_COLUMNS = [
    "Record_Key","Logged_At_ET","Slate_Date","GamePk","Game","Start_Time_UTC",
    "Market","Pick","Side","Market_Line","Odds","Book","Grade",
    "Model_Probability","Edge","EV","Fair_Line","Model_Weight","Market_Weight",
    "Lineups_Confirmed","Model_Confidence","App_Version","Model_Version",
    "Result","Units","Final_Away_Score","Final_Home_Score","Final_Total",
    "Graded_At_ET",
]


TRACKER_MIN_CONFIDENCE_ML = 80
TRACKER_MIN_CONFIDENCE_TOTAL = 80
TRACKER_REQUIRE_CONFIRMED_LINEUPS = True

def empty_tracker():
    return pd.DataFrame(columns=TRACKER_COLUMNS)

def _tracker_clean(df):
    if df is None or df.empty:
        return empty_tracker()

    out = df.copy()
    for c in TRACKER_COLUMNS:
        if c not in out.columns:
            out[c] = None

    out = out[TRACKER_COLUMNS].copy()

    # Older tracker CSVs can load entirely blank timestamp/text columns as
    # float64. Later assigning an ISO timestamp string then raises TypeError.
    object_cols = [
        "Record_Key","Logged_At_ET","Slate_Date","Game","Start_Time_UTC",
        "Market","Pick","Side","Book","Grade","App_Version","Model_Version",
        "Result","Graded_At_ET",
    ]
    for c in object_cols:
        if c in out.columns:
            out[c] = out[c].astype("object")

    numeric_cols = [
        "GamePk","Market_Line","Odds","Model_Probability","Edge","EV",
        "Fair_Line","Model_Weight","Market_Weight","Model_Confidence",
        "Units","Final_Away_Score","Final_Home_Score","Final_Total",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def load_tracker():
    if "_model_tracker_df" in st.session_state:
        return _tracker_clean(st.session_state["_model_tracker_df"])
    try:
        if TRACKER_PATH.exists():
            df = pd.read_csv(TRACKER_PATH)
        else:
            df = empty_tracker()
    except Exception:
        df = empty_tracker()
    st.session_state["_model_tracker_df"] = _tracker_clean(df)
    return _tracker_clean(df)

def save_tracker(df):
    df = _tracker_clean(df)
    st.session_state["_model_tracker_df"] = df.copy()
    try:
        TRACKER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TRACKER_PATH.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(TRACKER_PATH)
        st.session_state["_tracker_storage_error"] = ""
        return True
    except Exception as e:
        st.session_state["_tracker_storage_error"] = str(e)
        return False

def _now_et_iso():
    return pd.Timestamp.now(tz="America/New_York").isoformat()

def _game_lookup(games):
    return {str(g.get("GamePk")): g for g in games}

def _append_tracker_row(row):
    df = load_tracker()
    key = str(row.get("Record_Key"))
    if key in set(df["Record_Key"].astype(str)):
        return False
    full = {c: row.get(c) for c in TRACKER_COLUMNS}
    full["Result"] = full.get("Result") or "PENDING"
    if full.get("Units") is None:
        full["Units"] = 0.0
    df = pd.concat([df, pd.DataFrame([full])], ignore_index=True)
    save_tracker(df)
    return True


def tracker_qualification(candidate, market_type):
    """Return (qualified, reason). The live board can still show early signals,
    but only mature pregame signals enter the headline forward-performance ledger.
    """
    if not candidate or not candidate.get("pregame"):
        return False, "not pregame"
    effective_lineups_confirmed = bool(
        candidate.get("lineup_confirmed")
        or candidate.get("feed_lineup_confirmed")
        or int(candidate.get("lineup_teams_ready") or 0) >= 2
    )
    if TRACKER_REQUIRE_CONFIRMED_LINEUPS and not effective_lineups_confirmed:
        return False, f'lineups not confirmed ({int(candidate.get("lineup_teams_ready") or 0)}/2)'
    conf = int(candidate.get("confidence") or 0)
    min_conf = TRACKER_MIN_CONFIDENCE_TOTAL if str(market_type).upper() == "TOTAL" else TRACKER_MIN_CONFIDENCE_ML
    if conf < min_conf:
        return False, f"confidence {conf} < {min_conf}"
    return True, "qualified"


def tracker_candidate_status(candidate, market_type):
    """Small UI/debug helper: return the exact forward-tracker gate status."""
    qualified, reason = tracker_qualification(candidate, market_type)
    return {
        "qualified": bool(qualified),
        "reason": reason,
        "confidence": int(candidate.get("confidence") or 0),
        "lineups_confirmed": bool(
            candidate.get("lineup_confirmed")
            or candidate.get("feed_lineup_confirmed")
            or int(candidate.get("lineup_teams_ready") or 0) >= 2
        ),
    }


def track_current_official_recommendations(candidates, games, slate_date):
    """Freeze the first official ML signal for each game. Never overwrite later line moves."""
    game_map = _game_lookup(games)
    added = 0
    for x in candidates:
        qualified, _reason = tracker_qualification(x, "MONEYLINE")
        if not qualified or not x.get("market_available"):
            continue
        b = x.get("best") or {}
        if b.get("selection") not in ("BET", "BEST BET"):
            continue
        g = game_map.get(str(x.get("GamePk")), {})
        row = {
            "Record_Key": f'{x.get("GamePk")}|MONEYLINE',
            "Logged_At_ET": _now_et_iso(),
            "Slate_Date": str(slate_date),
            "GamePk": x.get("GamePk"),
            "Game": x.get("game"),
            "Start_Time_UTC": g.get("GameDate"),
            "Market": "MONEYLINE",
            "Pick": b.get("team"),
            "Side": b.get("team"),
            "Market_Line": None,
            "Odds": b.get("odds"),
            "Book": b.get("book"),
            "Grade": b.get("selection"),
            "Model_Probability": b.get("prob"),
            "Edge": b.get("edge"),
            "EV": b.get("ev"),
            "Fair_Line": b.get("fair"),
            "Model_Weight": x.get("alpha"),
            "Market_Weight": 1 - float(x.get("alpha", 0)) if x.get("alpha") is not None else None,
            "Lineups_Confirmed": bool(x.get("lineup_confirmed") or x.get("feed_lineup_confirmed") or int(x.get("lineup_teams_ready") or 0) >= 2),
            "Model_Confidence": x.get("confidence"),
            "App_Version": APP_VERSION,
            "Model_Version": MODEL_VERSION,
            "Result": "PENDING",
            "Units": 0.0,
        }
        added += int(_append_tracker_row(row))
    return added

def track_current_total_recommendations(candidates, games, model_df, totals_payload, slate_date):
    """Freeze the first official totals signal for each game when totals odds are loaded."""
    if not st.session_state.get("totals_loaded") or model_df is None or model_df.empty:
        return 0
    added = 0
    game_map = _game_lookup(games)
    for x in candidates:
        qualified, _reason = tracker_qualification(x, "TOTAL")
        if not qualified:
            continue
        mr = model_df.loc[model_df["GamePk"] == x["GamePk"]]
        if mr.empty:
            continue
        g = game_map.get(str(x.get("GamePk")), {})
        ev = match_event(totals_payload.get("events", []), g) if g else None
        tm = totals_market(ev)
        if not tm:
            continue
        ctx = engine.totals_projection(mr.iloc[0].to_dict()) if hasattr(engine, "totals_projection") else {
            "Projected_Total": x["away_proj"] + x["home_proj"]
        }
        tp = build_total_pick(float(ctx["Projected_Total"]), tm)
        if not tp or tp.get("grade") not in ("BET", "BEST BET"):
            continue
        row = {
            "Record_Key": f'{x.get("GamePk")}|TOTAL',
            "Logged_At_ET": _now_et_iso(),
            "Slate_Date": str(slate_date),
            "GamePk": x.get("GamePk"),
            "Game": x.get("game"),
            "Start_Time_UTC": g.get("GameDate"),
            "Market": "TOTAL",
            "Pick": f'{tp.get("side")} {tp.get("market_total")}',
            "Side": tp.get("side"),
            "Market_Line": tp.get("market_total"),
            "Odds": tp.get("odds"),
            "Book": tp.get("book"),
            "Grade": tp.get("grade"),
            "Model_Probability": tp.get("prob"),
            "Edge": tp.get("edge"),
            "EV": tp.get("ev"),
            "Fair_Line": None,
            "Model_Weight": TOTALS_MODEL_WEIGHT,
            "Market_Weight": 1 - TOTALS_MODEL_WEIGHT,
            "Lineups_Confirmed": bool(x.get("lineup_confirmed") or x.get("feed_lineup_confirmed") or int(x.get("lineup_teams_ready") or 0) >= 2),
            "Model_Confidence": x.get("confidence"),
            "App_Version": APP_VERSION,
            "Model_Version": MODEL_VERSION,
            "Result": "PENDING",
            "Units": 0.0,
        }
        added += int(_append_tracker_row(row))
    return added

@st.cache_data(ttl=120, show_spinner=False)
def tracker_results_for_date(date_text):
    """Free MLB Stats API result lookup. This does not use Odds API credits."""
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": str(date_text)},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}
    out = {}
    for block in data.get("dates", []):
        for g in block.get("games", []):
            status = g.get("status", {}) or {}
            away = g.get("teams", {}).get("away", {}) or {}
            home = g.get("teams", {}).get("home", {}) or {}
            out[str(g.get("gamePk"))] = {
                "abstract": str(status.get("abstractGameState", "")),
                "detailed": str(status.get("detailedState", "")),
                "away_score": away.get("score"),
                "home_score": home.get("score"),
                "away_name": (away.get("team", {}) or {}).get("name"),
                "home_name": (home.get("team", {}) or {}).get("name"),
            }
    return out

def _american_profit(odds):
    try:
        o = float(odds)
        return o / 100.0 if o > 0 else 100.0 / abs(o)
    except Exception:
        return 0.0

def grade_tracker(force=False):
    """Automatically grade pending recommendations once MLB marks the game final."""
    df = load_tracker()

    # Defensive normalization for tracker files created before v2.0.2.
    if "Graded_At_ET" in df.columns:
        df["Graded_At_ET"] = df["Graded_At_ET"].astype("object")
    if df.empty:
        return 0
    pending_mask = df["Result"].fillna("PENDING").astype(str).eq("PENDING")
    if not pending_mask.any():
        return 0

    now = pd.Timestamp.now(tz="UTC")
    last = st.session_state.get("_tracker_last_grade_check")
    if not force and last is not None:
        try:
            if (now - pd.Timestamp(last)).total_seconds() < 60:
                return 0
        except Exception:
            pass
    st.session_state["_tracker_last_grade_check"] = now.isoformat()

    changed = 0
    for date_text in df.loc[pending_mask, "Slate_Date"].dropna().astype(str).unique():
        results = tracker_results_for_date(date_text)
        if not results:
            continue
        idxs = df.index[pending_mask & df["Slate_Date"].astype(str).eq(date_text)]
        for i in idxs:
            rec = results.get(str(df.at[i, "GamePk"]))
            if not rec:
                continue
            abstract = rec.get("abstract", "").lower()
            detailed = rec.get("detailed", "").lower()
            if any(x in detailed for x in ("postponed", "cancelled", "canceled")):
                df.at[i, "Result"] = "VOID"
                df.at[i, "Units"] = 0.0
                df.at[i, "Graded_At_ET"] = _now_et_iso()
                changed += 1
                continue
            if abstract != "final" and "final" not in detailed and "game over" not in detailed:
                continue
            try:
                away_score = int(rec.get("away_score"))
                home_score = int(rec.get("home_score"))
            except Exception:
                continue

            df.at[i, "Final_Away_Score"] = away_score
            df.at[i, "Final_Home_Score"] = home_score
            df.at[i, "Final_Total"] = away_score + home_score

            market = str(df.at[i, "Market"]).upper()
            odds = df.at[i, "Odds"]
            result = "LOSS"
            if market == "MONEYLINE":
                pick = str(df.at[i, "Pick"])
                winner = rec.get("away_name") if away_score > home_score else rec.get("home_name")
                result = "WIN" if team_key(pick) == team_key(winner) else "LOSS"
            elif market == "TOTAL":
                try:
                    line = float(df.at[i, "Market_Line"])
                except Exception:
                    continue
                final_total = away_score + home_score
                side = str(df.at[i, "Side"]).upper()
                if abs(final_total - line) < 1e-9:
                    result = "PUSH"
                elif side == "OVER":
                    result = "WIN" if final_total > line else "LOSS"
                else:
                    result = "WIN" if final_total < line else "LOSS"

            units = _american_profit(odds) if result == "WIN" else (-1.0 if result == "LOSS" else 0.0)
            df.at[i, "Result"] = result
            df.at[i, "Units"] = units
            df.at[i, "Graded_At_ET"] = _now_et_iso()
            changed += 1

    if changed:
        save_tracker(df)
    return changed

def import_diagnostics_tracker(uploaded):
    """Backfill an older recommendation from a downloaded single-game diagnostics CSV."""
    try:
        d = pd.read_csv(uploaded)
    except Exception as e:
        return 0, f"Could not read diagnostics CSV: {e}"
    if d.empty:
        return 0, "Diagnostics CSV is empty."
    r = d.iloc[0]
    required = {"GamePk","Away","Home","Away_Card_Label","Home_Card_Label"}
    if not required.issubset(d.columns):
        return 0, "This does not look like an MLB game diagnostics CSV."

    choices = []
    for prefix in ("Away", "Home"):
        label = str(r.get(f"{prefix}_Card_Label", ""))
        if label in ("BET", "BEST BET"):
            choices.append(prefix)
    if not choices:
        return 0, "That diagnostics file did not contain an official BET/BEST BET."
    prefix = choices[0]
    team = r.get(prefix)
    row = {
        "Record_Key": f'{r.get("GamePk")}|MONEYLINE',
        "Logged_At_ET": _now_et_iso(),
        "Slate_Date": str(r.get("Slate_Date")),
        "GamePk": r.get("GamePk"),
        "Game": r.get("Game"),
        "Start_Time_UTC": None,
        "Market": "MONEYLINE",
        "Pick": team,
        "Side": team,
        "Market_Line": None,
        "Odds": r.get(f"{prefix}_Best_Odds"),
        "Book": r.get(f"{prefix}_Best_Book"),
        "Grade": r.get(f"{prefix}_Card_Label"),
        "Model_Probability": r.get(f"{prefix}_Calibrated_Prob"),
        "Edge": r.get(f"{prefix}_Edge"),
        "EV": r.get(f"{prefix}_EV"),
        "Fair_Line": r.get(f"{prefix}_Fair_ML"),
        "Model_Weight": r.get("Model_Weight"),
        "Market_Weight": r.get("Market_Weight"),
        "Lineups_Confirmed": r.get("Lineups_Confirmed"),
        "Model_Confidence": r.get("Model_Confidence"),
        "App_Version": r.get("App_Version"),
        "Model_Version": r.get("Model_Version"),
        "Result": "PENDING",
        "Units": 0.0,
    }
    added = int(_append_tracker_row(row))
    if added:
        grade_tracker(force=True)
        odds_txt = row.get("Odds")
        try:
            odds_txt = f'{float(odds_txt):+.0f}'
        except Exception:
            odds_txt = str(odds_txt)
        return 1, f"Imported {team} {odds_txt}."
    return 0, "That game/market is already in the tracker."

def tracker_performance_summary(df):
    if df is None or df.empty:
        return {"wins":0,"losses":0,"pushes":0,"voids":0,"pending":0,"units":0.0,"roi":0.0,"graded":0}
    res = df["Result"].fillna("PENDING").astype(str)
    wins = int((res=="WIN").sum())
    losses = int((res=="LOSS").sum())
    pushes = int((res=="PUSH").sum())
    voids = int((res=="VOID").sum())
    pending = int((res=="PENDING").sum())
    completed = wins + losses + pushes
    units = pd.to_numeric(df["Units"], errors="coerce").fillna(0).sum()
    roi = units / completed if completed else 0.0
    return {"wins":wins,"losses":losses,"pushes":pushes,"voids":voids,"pending":pending,"units":units,"roi":roi,"graded":completed}

def tracker_split_table(df):
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for market, g in df.groupby("Market", dropna=False):
        s = tracker_performance_summary(g)
        record = f'{s["wins"]}-{s["losses"]}' + (f'-{s["pushes"]}P' if s["pushes"] else "")
        rows.append({
            "Market": market,
            "Graded": s["graded"],
            "Record": record,
            "Units": round(s["units"], 2),
            "ROI %": round(s["roi"] * 100, 1),
            "Pending": s["pending"],
        })
    return pd.DataFrame(rows)




@st.cache_data(ttl=20, show_spinner=False)
def fetch_live_win_probability(game_pk):
    """Current team win probability from the free MLB contextMetrics endpoint."""
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/contextMetrics",
            timeout=10,
        )
        r.raise_for_status()
        data = r.json() or {}
        away = data.get("awayWinProbability")
        home = data.get("homeWinProbability")
        if away is None or home is None:
            return {}
        away = float(away)
        home = float(home)
        # Defensive normalization in case an implementation returns 0-1 rather than 0-100.
        if away <= 1.0 and home <= 1.0:
            away *= 100.0
            home *= 100.0
        total = away + home
        if total > 0 and abs(total - 100.0) > 0.5:
            away = away / total * 100.0
            home = home / total * 100.0
        return {
            "away": max(0.0, min(100.0, away)),
            "home": max(0.0, min(100.0, home)),
        }
    except Exception:
        return {}

def _picked_team_wp(rec, game, win_prob):
    if not win_prob:
        return None
    pick = str(rec.get("Pick") or rec.get("Side") or "")
    away = str(game.get("Away") or "")
    home = str(game.get("Home") or "")
    if team_key(pick) == team_key(away):
        return float(win_prob.get("away"))
    if team_key(pick) == team_key(home):
        return float(win_prob.get("home"))
    return None

def _game_win_probability_html(game, win_prob):
    if not win_prob:
        return ""
    away = str(game.get("Away") or "Away")
    home = str(game.get("Home") or "Home")
    ap = float(win_prob.get("away", 50))
    hp = float(win_prob.get("home", 50))
    return (
        f'<div class="wp-wrap">'
        f'<div class="wp-title">MLB LIVE WIN PROBABILITY</div>'
        f'<div class="wp-labels"><span>{away}<b>{ap:.0f}%</b></span>'
        f'<span>{home}<b>{hp:.0f}%</b></span></div>'
        f'<div class="wp-track">'
        f'<div class="wp-away" style="width:{ap:.1f}%"></div>'
        f'<div class="wp-mid"></div>'
        f'</div>'
        f'</div>'
    )

def _live_tracker_bucket(rec, game, win_prob):
    """Return ON TRACK / NEUTRAL / NEEDS HELP for slate-level monitoring."""
    result = str(rec.get("Result", "PENDING") or "PENDING").upper()
    if result == "WIN":
        return "FINAL_WIN"
    if result == "LOSS":
        return "FINAL_LOSS"
    if result == "PUSH":
        return "FINAL_PUSH"
    if result == "VOID":
        return "FINAL_VOID"

    market = str(rec.get("Market") or "").upper()
    if market == "MONEYLINE":
        wp = _picked_team_wp(rec, game, win_prob)
        if wp is not None:
            if wp >= 60:
                return "ON_TRACK"
            if wp <= 40:
                return "NEEDS_HELP"
            return "NEUTRAL"

        # Fallback to score if live WP is temporarily unavailable.
        pick = str(rec.get("Pick") or "")
        away = str(game.get("Away") or "")
        home = str(game.get("Home") or "")
        a = _safe_int(game.get("Away_Score"), 0)
        h = _safe_int(game.get("Home_Score"), 0)
        if team_key(pick) == team_key(away):
            diff = a - h
        elif team_key(pick) == team_key(home):
            diff = h - a
        else:
            diff = 0
        return "ON_TRACK" if diff > 0 else ("NEEDS_HELP" if diff < 0 else "NEUTRAL")

    if market == "TOTAL":
        side = str(rec.get("Side") or "").upper()
        try:
            line = float(rec.get("Market_Line"))
        except Exception:
            return "NEUTRAL"
        runs = _safe_int(game.get("Away_Score"), 0) + _safe_int(game.get("Home_Score"), 0)
        frac = _inning_fraction(game)
        expected_to_now = line * frac
        ratio = runs / max(expected_to_now, 0.75)

        if side == "UNDER":
            if runs >= line:
                return "NEEDS_HELP"
            if ratio <= 0.95:
                return "ON_TRACK"
            if ratio <= 1.20:
                return "NEUTRAL"
            return "NEEDS_HELP"
        else:
            if runs > line:
                return "ON_TRACK"
            if ratio >= 1.05:
                return "ON_TRACK"
            if ratio >= 0.80:
                return "NEUTRAL"
            return "NEEDS_HELP"

    return "NEUTRAL"

def _slate_tracking_summary(tracker_df, games, fresh_scoreboard, slate_date):
    if tracker_df is None or tracker_df.empty:
        return {
            "tracked":0,"live":0,"final":0,"wins":0,"losses":0,"pushes":0,
            "on_track":0,"neutral":0,"needs_help":0,"units":0.0,"status":"NO TRACKED BETS"
        }

    today_rows = tracker_df[tracker_df["Slate_Date"].astype(str) == str(slate_date)].copy()
    if today_rows.empty:
        return {
            "tracked":0,"live":0,"final":0,"wins":0,"losses":0,"pushes":0,
            "on_track":0,"neutral":0,"needs_help":0,"units":0.0,"status":"NO TRACKED BETS"
        }

    game_map = {}
    for g0 in games:
        gf = fresh_scoreboard.get(str(g0.get("GamePk")), g0)
        game_map[str(g0.get("GamePk"))] = gf

    out = {
        "tracked":len(today_rows),"live":0,"final":0,"wins":0,"losses":0,"pushes":0,
        "on_track":0,"neutral":0,"needs_help":0,"units":0.0
    }

    for _, rec in today_rows.iterrows():
        g = game_map.get(str(rec.get("GamePk")), {})
        state = game_state(g)
        result = str(rec.get("Result","PENDING") or "PENDING").upper()

        if state == "FINAL" or result in ("WIN","LOSS","PUSH","VOID"):
            out["final"] += 1
            if result == "WIN":
                out["wins"] += 1
            elif result == "LOSS":
                out["losses"] += 1
            elif result == "PUSH":
                out["pushes"] += 1
            try:
                out["units"] += float(rec.get("Units") or 0)
            except Exception:
                pass
            continue

        if state == "LIVE":
            out["live"] += 1
            wp = fetch_live_win_probability(rec.get("GamePk"))
            bucket = _live_tracker_bucket(rec, g, wp)
            if bucket == "ON_TRACK":
                out["on_track"] += 1
            elif bucket == "NEEDS_HELP":
                out["needs_help"] += 1
            else:
                out["neutral"] += 1

    pulse = (out["wins"] - out["losses"]) * 2 + out["on_track"] - out["needs_help"]
    if out["tracked"] == 0:
        status = "NO TRACKED BETS"
    elif pulse >= 2:
        status = "SLATE POSITIVE"
    elif pulse <= -2:
        status = "SLATE UNDER PRESSURE"
    else:
        status = "SLATE MIXED"
    out["status"] = status
    return out

def _slate_pulse_html(summary):
    record = f'{summary["wins"]}-{summary["losses"]}'
    if summary["pushes"]:
        record += f'-{summary["pushes"]}P'
    status_cls = (
        "pulse-good" if summary["status"] == "SLATE POSITIVE"
        else ("pulse-risk" if summary["status"] == "SLATE UNDER PRESSURE" else "pulse-neutral")
    )
    return (
        f'<div class="slate-pulse">'
        f'<div class="pulse-head"><div><div class="pulse-kicker">TODAY\'S TRACKED SLATE</div>'
        f'<div class="pulse-title">{summary["status"]}</div><div class="pulse-sub">Final results + live tracked bets</div></div>'
        f'<div class="pulse-status {status_cls}">{summary["tracked"]} TRACKED</div></div>'
        f'<div class="pulse-grid">'
        f'<div><span>FINAL</span><b>{record}</b></div>'
        f'<div><span>LIVE</span><b>{summary["live"]}</b></div>'
        f'<div><span>ON TRACK</span><b>{summary["on_track"]}</b></div>'
        f'<div><span>NEEDS HELP</span><b>{summary["needs_help"]}</b></div>'
        f'<div><span>FINAL UNITS</span><b>{summary["units"]:+.2f}u</b></div>'
        f'</div>'
        f'</div>'
    )

def tracked_rows_for_game(game_pk, tracker_df=None):
    if tracker_df is None:
        tracker_df = load_tracker()
    if tracker_df is None or tracker_df.empty:
        return pd.DataFrame(columns=TRACKER_COLUMNS)
    return tracker_df[tracker_df["GamePk"].astype(str) == str(game_pk)].copy()


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def _inning_fraction(game):
    """Approximate fraction of regulation game completed, for visual pace only."""
    inning = max(1, _safe_int(game.get("Current_Inning"), 1))
    outs = max(0, min(3, _safe_int(game.get("Outs"), 0)))
    half = str(game.get("Inning_State") or game.get("Inning_Half") or "").lower()
    completed_halves = max(0, (inning - 1) * 2)
    if "bottom" in half or "middle" in half:
        completed_halves += 1
    elif "end" in half:
        completed_halves += 2
    frac = (completed_halves * 3 + outs) / 54.0
    return max(0.02, min(1.0, frac))

def _odds_text(v):
    try:
        return f"{int(float(v)):+d}"
    except Exception:
        return ""

def _tracker_result_badge(rec):
    result = str(rec.get("Result", "PENDING") or "PENDING").upper()
    if result == "WIN":
        return "WIN", "track-good"
    if result == "LOSS":
        return "LOSS", "track-risk"
    if result == "PUSH":
        return "PUSH", "track-neutral"
    if result == "VOID":
        return "VOID", "track-neutral"
    return None, None

def _total_visual(rec, game):
    side = str(rec.get("Side") or "").upper()
    try:
        line = float(rec.get("Market_Line"))
    except Exception:
        return "", "track-neutral", "TRACKING"
    away_score = _safe_int(game.get("Away_Score"), 0)
    home_score = _safe_int(game.get("Home_Score"), 0)
    runs = away_score + home_score
    state = game_state(game)

    final_badge, final_cls = _tracker_result_badge(rec)
    if final_badge:
        status, status_cls = final_badge, final_cls
    else:
        frac = _inning_fraction(game)
        expected_to_now = line * frac
        # Visual pace heuristic only — not a live probability model.
        if side == "UNDER":
            ratio = runs / max(expected_to_now, 0.75)
            if runs >= line:
                status, status_cls = "NEEDS SCORING", "track-risk"
            elif ratio <= 0.90:
                status, status_cls = "ON TRACK", "track-good"
            elif ratio <= 1.20:
                status, status_cls = "ON TRACK", "track-neutral"
            else:
                status, status_cls = "NEEDS SCORING", "track-risk"
        else:
            ratio = runs / max(expected_to_now, 0.75)
            if runs > line:
                status, status_cls = "ON TRACK", "track-good"
            elif ratio >= 1.10:
                status, status_cls = "ON TRACK", "track-good"
            elif ratio >= 0.80:
                status, status_cls = "ON TRACK", "track-neutral"
            else:
                status, status_cls = "NEEDS SCORING", "track-risk"

    scale_max = max(line * 1.65, runs + 2, 12)
    fill_pct = max(0, min(100, runs / scale_max * 100))
    line_pct = max(2, min(96, line / scale_max * 100))
    pick = f"{side} {line:.1f} {_odds_text(rec.get('Odds'))}".strip()

    html = (
        f'<div class="bet-section-head"><div>'
        f'<div class="market-chip">TOTAL</div>'
        f'<div class="bet-pick">{pick}</div></div>'
        f'<div class="track-pill {status_cls}">{status}</div></div>'
        f'<div class="run-summary">'
        f'<div class="run-stat"><span>CURRENT RUNS</span><b>{runs:g}</b></div>'
        f'<div class="run-stat line-stat"><span>BET LINE</span><b>{line:g}</b></div>'
        f'</div>'
        f'<div class="run-track clear-track">'
        f'<div class="run-fill {status_cls}" style="width:{fill_pct:.1f}%"></div>'
        f'<div class="line-marker" style="left:{line_pct:.1f}%"></div>'
        f'</div>'
        f'<div class="run-axis clear-axis"><span>0</span>'
        f'<span class="line-axis-label" style="left:{line_pct:.1f}%">LINE {line:g}</span></div>'
    )
    return html, status_cls, status


def _moneyline_visual(rec, game, win_prob=None):
    pick = str(rec.get("Pick") or "")
    odds = _odds_text(rec.get("Odds"))
    away = str(game.get("Away") or "")
    home = str(game.get("Home") or "")
    away_score = _safe_int(game.get("Away_Score"), 0)
    home_score = _safe_int(game.get("Home_Score"), 0)

    final_badge, final_cls = _tracker_result_badge(rec)
    picked_wp = _picked_team_wp(rec, game, win_prob)

    if final_badge:
        status, status_cls = final_badge, final_cls
    elif picked_wp is not None:
        if picked_wp >= 60:
            status, status_cls = "ON TRACK", "track-good"
        elif picked_wp <= 40:
            status, status_cls = "NEEDS HELP", "track-risk"
        else:
            status, status_cls = "LIVE", "track-neutral"
    else:
        if team_key(pick) == team_key(away):
            margin = away_score - home_score
        elif team_key(pick) == team_key(home):
            margin = home_score - away_score
        else:
            margin = 0
        status = "LEADING" if margin > 0 else ("TRAILING" if margin < 0 else "TIED")
        status_cls = "track-good" if margin > 0 else ("track-risk" if margin < 0 else "track-neutral")

    if picked_wp is not None:
        meter = max(2, min(98, picked_wp))
        wp_main = f"{picked_wp:.0f}%"
        wp_label = "CURRENT WIN PROBABILITY"
    else:
        if team_key(pick) == team_key(away):
            margin = away_score - home_score
        else:
            margin = home_score - away_score
        meter = max(8, min(92, 50 + margin * 8))
        wp_main = "—"
        wp_label = "LIVE WIN PROBABILITY"

    html = (
        f'<div class="bet-section-head"><div>'
        f'<div class="market-chip">MONEYLINE</div>'
        f'<div class="bet-pick">{pick} {odds}</div></div>'
        f'<div class="track-pill {status_cls}">{status}</div></div>'
        f'<div class="ml-live-wp">'
        f'<span>{wp_label}</span><b>{wp_main}</b></div>'
        f'<div class="ml-meter-wrap live-wp-meter">'
        f'<div class="ml-meter-line"></div>'
        f'<div class="ml-meter-mid"></div>'
        f'<div class="ml-meter-dot {status_cls}" style="left:{meter:.1f}%"></div>'
        f'</div>'
        f'<div class="ml-meter-labels"><span>0%</span><b>50%</b><span>100%</span></div>'
    )
    return html, status_cls, status

def _score_rows(game):
    away = str(game.get("Away") or "Away")
    home = str(game.get("Home") or "Home")
    away_score = game.get("Away_Score")
    home_score = game.get("Home_Score")
    try:
        away_score = int(away_score)
    except Exception:
        away_score = "-"
    try:
        home_score = int(home_score)
    except Exception:
        home_score = "-"
    return (
        f'<div class="team-row"><span>{away}</span><b>{away_score}</b></div>'
        f'<div class="team-row"><span>{home}</span><b>{home_score}</b></div>'
    )

def _visual_tracked_card(rec, game, win_prob=None):
    market = str(rec.get("Market") or "").upper()
    state = game_state(game)
    state_label = "FINAL" if state == "FINAL" else "LIVE"
    inning = "FINAL" if state == "FINAL" else inning_status_text(game)
    if market == "TOTAL":
        bet_html, _, _ = _total_visual(rec, game)
    else:
        bet_html, _, _ = _moneyline_visual(rec, game, win_prob=win_prob)

    return (
        f'<div class="visual-bet-card">'
        f'<div class="visual-score-head">'
        f'<div class="score-teams">{_score_rows(game)}</div>'
        f'<div class="live-meta"><div class="live-dot-wrap"><span class="mini-dot"></span>{state_label}</div>'
        f'<div class="inning-meta">{inning}</div></div>'
        f'<div class="diamond-mini">'
        f'<i class="base-second {"occupied" if game.get("On_Second") else ""}"></i>'
        f'<i class="base-third {"occupied" if game.get("On_Third") else ""}"></i>'
        f'<i class="base-first {"occupied" if game.get("On_First") else ""}"></i>'
        f'<i class="base-home"></i>'
        f'</div>'
        f'</div>'
        f'{_game_win_probability_html(game, win_prob)}'
        f'<div class="visual-divider"></div>'
        f'{bet_html}'
        f'</div>'
    )


def render_live_scoreboard(games, fresh_scoreboard, tracker_df, slate_date):
    """Premium live bet tracker with MLB win probability and slate-level tracking."""
    fresh_games = []
    for g0 in games:
        g = fresh_scoreboard.get(str(g0.get("GamePk")), g0)
        if game_state(g) in ("LIVE", "FINAL"):
            fresh_games.append(g)

    live_list = [g for g in fresh_games if game_state(g) == "LIVE"]
    final_list = [g for g in fresh_games if game_state(g) == "FINAL"]

    tracked_live = []
    for g in live_list:
        rows = tracked_rows_for_game(g.get("GamePk"), tracker_df)
        if not rows.empty:
            for _, rec in rows.iterrows():
                tracked_live.append((g, rec))

    tracked_final = []
    for g in final_list:
        rows = tracked_rows_for_game(g.get("GamePk"), tracker_df)
        if not rows.empty:
            for _, rec in rows.iterrows():
                tracked_final.append((g, rec))

    st.markdown(
        f'<div class="tracker-hero">'
        f'<div><div class="tracker-eyebrow">LIVE MODEL MONITOR</div>'
        f'<div class="tracker-title">Bet Tracker <span class="tracker-count">{len(tracked_live)}</span></div>'
        f'<div class="tracker-sub">Win probability, live bet progress, and today\'s slate status</div></div>'
        f'<div class="tracker-live-orb"><span></span>LIVE</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    summary = _slate_tracking_summary(tracker_df, games, fresh_scoreboard, slate_date)
    st.markdown(_slate_pulse_html(summary), unsafe_allow_html=True)

    if tracked_live:
        st.markdown('<div class="kicker">Live Tracked Bets</div>', unsafe_allow_html=True)
        for g, rec in tracked_live:
            wp = fetch_live_win_probability(g.get("GamePk"))
            st.markdown(_visual_tracked_card(rec, g, win_prob=wp), unsafe_allow_html=True)
    else:
        st.info("No tracked bets are live right now.")

    if tracked_final:
        with st.expander(f"Completed Tracked Bets — {len(tracked_final)}", expanded=False):
            for g, rec in tracked_final:
                st.markdown(_visual_tracked_card(rec, g, win_prob=None), unsafe_allow_html=True)

    untracked_live = [g for g in live_list if tracked_rows_for_game(g.get("GamePk"), tracker_df).empty]
    if untracked_live:
        with st.expander(f"Other Live Games — {len(untracked_live)}", expanded=False):
            for g in untracked_live:
                wp = fetch_live_win_probability(g.get("GamePk"))
                st.markdown(
                    f'<div class="plain-live-card-wrap">'
                    f'<div class="plain-live-card"><div><div class="score-state">{inning_status_text(g)}</div>'
                    f'<div class="score-main">{live_score_text(g)}</div></div><div class="score-badge">LIVE</div></div>'
                    f'{_game_win_probability_html(g, wp)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )



def render_live_games_page(games, fresh_scoreboard):
    """Dedicated scores page: all live games first, finals collapsed below."""
    fresh_games = []
    for g0 in games:
        g = fresh_scoreboard.get(str(g0.get("GamePk")), g0)
        state = game_state(g)
        if state in ("LIVE", "FINAL"):
            fresh_games.append(g)

    live_list = [g for g in fresh_games if game_state(g) == "LIVE"]
    final_list = [g for g in fresh_games if game_state(g) == "FINAL"]

    st.markdown(
        f'<div class="page-head">'
        f'<div class="page-kicker">LIVE SCOREBOARD</div>'
        f'<div class="page-title">Live <span class="page-count">{len(live_list)}</span></div>'
        f'<div class="page-sub">Scores, inning status, baserunners and MLB live win probability.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if live_list:
        for g in live_list:
            wp = fetch_live_win_probability(g.get("GamePk"))
            st.markdown(
                f'<div class="live-page-card">'
                f'<div class="visual-score-head">'
                f'<div class="score-teams">{_score_rows(g)}</div>'
                f'<div class="live-meta"><div class="live-dot-wrap"><span class="mini-dot"></span>LIVE</div>'
                f'<div class="inning-meta">{inning_status_text(g)}</div></div>'
                f'<div class="diamond-mini">'
                f'<i class="base-second {"occupied" if g.get("On_Second") else ""}"></i>'
                f'<i class="base-third {"occupied" if g.get("On_Third") else ""}"></i>'
                f'<i class="base-first {"occupied" if g.get("On_First") else ""}"></i>'
                f'<i class="base-home"></i>'
                f'</div>'
                f'</div>'
                f'{_game_win_probability_html(g, wp)}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No games are currently live.")

    if final_list:
        with st.expander(f"Final Games — {len(final_list)}", expanded=False):
            for g in final_list:
                st.markdown(
                    f'<div class="plain-live-card"><div>'
                    f'<div class="score-state">FINAL</div>'
                    f'<div class="score-main">{live_score_text(g)}</div>'
                    f'</div><div class="score-badge">FINAL</div></div>',
                    unsafe_allow_html=True,
                )


@_auto_fragment(20)
def render_auto_live_page(games, slate_date):
    """Free live scoreboard refreshes every 20 seconds. No Odds API calls."""
    fresh = fetch_fresh_scoreboard(slate_date)
    render_live_games_page(games, fresh)
    updated = pd.Timestamp.now(tz="America/New_York")
    st.markdown(
        f'<div class="auto-fresh"><span></span>LIVE DATA • UPDATED {updated.strftime("%-I:%M:%S %p")}</div>',
        unsafe_allow_html=True,
    )

@_auto_fragment(20)
def render_auto_tracker_page(games, slate_date):
    """Refresh scores, live win probability and tracker state without paid odds calls."""
    # grade_tracker is internally throttled to once per minute.
    grade_tracker(force=False)
    fresh = fetch_fresh_scoreboard(slate_date)
    tracker_df = load_tracker()
    render_live_scoreboard(games, fresh, tracker_df, slate_date)
    updated = pd.Timestamp.now(tz="America/New_York")
    st.markdown(
        f'<div class="auto-fresh"><span></span>AUTO TRACKING • UPDATED {updated.strftime("%-I:%M:%S %p")}</div>',
        unsafe_allow_html=True,
    )

@_auto_fragment(60)
def render_auto_slate_status(games, slate_date):
    """Free slate status heartbeat for Board; never touches paid market endpoints."""
    fresh = fetch_fresh_scoreboard(slate_date)
    states = []
    for g0 in games:
        gf = fresh.get(str(g0.get("GamePk")), g0)
        states.append(game_state(gf))
    pre = sum(1 for x in states if x == "PREGAME")
    live = sum(1 for x in states if x == "LIVE")
    final = sum(1 for x in states if x == "FINAL")
    now_et = pd.Timestamp.now(tz="America/New_York")
    st.markdown(
        f'<div class="status ninth-status"><div><span class="dot"></span>'
        f'<span class="live">{slate_date.strftime("%b %-d")}</span> '
        f'• {pre} upcoming • {live} live • {final} final'
        f'<span class="auto-age"> • auto {now_et.strftime("%-I:%M %p")}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

def render_account_page():
    st.markdown(
        """<div class="ninth-full-logo"><img src="data:image/png;base64,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" alt="Ninth Signal"></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="page-head">'
        f'<div class="page-kicker">NINTH SIGNAL</div>'
        f'<div class="page-title">More</div>'
        f'<div class="page-sub">Model information, technical details, and advanced resources.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="account-card">'
        f'<div><span>APP</span><b>{APP_VERSION}</b></div>'
        f'<div><span>MODEL</span><b>{MODEL_VERSION}</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Model details & limitations", expanded=False):
        st.write(
            "Production recommendations are pregame-only. Live views are for tracking and "
            "score context; they do not generate in-game betting recommendations."
        )
    with st.expander("Data & downloads", expanded=False):
        st.caption("Use the Board and Bets tabs for slate/game downloads and performance exports.")


def render_performance_page():
    tracker_df = load_tracker()
    perf = tracker_performance_summary(tracker_df)
    record_display = f'{perf["wins"]}-{perf["losses"]}' + (f'-{perf["pushes"]}P' if perf["pushes"] else "")

    st.markdown('<div class="kicker">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metrics">'
        f'<div class="metric"><span>Record</span><b>{record_display}</b></div>'
        f'<div class="metric"><span>Units</span><b>{perf["units"]:+.2f}</b></div>'
        f'<div class="metric"><span>ROI</span><b>{perf["roi"]*100:+.1f}%</b></div>'
        f'<div class="metric"><span>Pending</span><b>{perf["pending"]}</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption("Headline tracker includes qualified pregame BET/BEST BET signals only: confirmed lineups, confidence ≥80 and valid odds.")

    if not tracker_df.empty:
        split = tracker_split_table(tracker_df)
        if not split.empty:
            st.dataframe(split, use_container_width=True, hide_index=True)

        recent_cols = ["Slate_Date","Game","Market","Pick","Odds","Grade","Result","Units"]
        recent = tracker_df.sort_values(["Slate_Date","Logged_At_ET"], ascending=False)
        st.dataframe(recent[[c for c in recent_cols if c in recent.columns]].head(50), use_container_width=True, hide_index=True)

    if st.button("Refresh Results (free)", use_container_width=True, key="perf_refresh_results"):
        tracker_results_for_date.clear()
        n = grade_tracker(force=True)
        st.success(f"Updated {n} completed recommendation(s)." if n else "No new finals to grade yet.")
        st.rerun()

    tracker_df = load_tracker()
    st.download_button(
        "Download Performance Tracker",
        data=tracker_df.to_csv(index=False).encode("utf-8"),
        file_name="mlb_model_recommendation_tracker.csv",
        mime="text/csv",
        use_container_width=True,
        key="perf_download_tracker",
    )

    with st.expander("Tracker backup / restore", expanded=False):
        restore_file = st.file_uploader("Restore tracker backup", type=["csv"], key="perf_restore_upload")
        if st.button("Merge Tracker Backup", use_container_width=True, disabled=(restore_file is None), key="perf_restore_btn"):
            try:
                incoming = _tracker_clean(pd.read_csv(restore_file))
                current = load_tracker()
                merged = pd.concat([current, incoming], ignore_index=True)
                merged = merged.drop_duplicates(subset=["Record_Key"], keep="first")
                save_tracker(merged)
                grade_tracker(force=True)
                st.success(f"Tracker restored/merged: {len(merged)} total records.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not restore tracker: {e}")

        diag_file = st.file_uploader("Import earlier Game Diagnostics CSV", type=["csv"], key="perf_diag_import")
        if st.button("Import Earlier Pick", use_container_width=True, disabled=(diag_file is None), key="perf_diag_btn"):
            n,msg = import_diagnostics_tracker(diag_file)
            if n:
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)


st.markdown("""
<div class="ninth-brand-header">
  <div class="ninth-brand-mark">
    <img src="data:image/png;base64,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" alt="Ninth Signal logo mark">
  </div>
  <div class="hero ninth-hero branded-hero-copy">
    <div class="eyebrow">BASEBALL MARKET SIGNALS</div>
    <div class="title"><span>Ninth</span> <span class="signal">Signal</span></div>
    <div class="sub">Pregame model signals, live tracking, and forward performance.</div>
    <div class="pill">MODEL LIVE</div>
  </div>
</div>
""", unsafe_allow_html=True)

try:
    api_key=st.secrets.get("ODDS_API_KEY","")
except Exception:
    api_key=""


slate_date=st.date_input(
    "Slate date",
    value=today_et(),
    min_value=today_et(),
    max_value=today_et()+timedelta(days=14),
    help="Current/upcoming MLB dates only.",
    label_visibility="collapsed",
)
st.markdown(
    '<div class="free-data-note"><span></span>Scores, game status and Tracker update automatically. '
    'Betting lines only update when you load them.</div>',
    unsafe_allow_html=True,
)

if "odds_payload" not in st.session_state:
    st.session_state.odds_payload={"events":[],"error":"","quota":{}}
    st.session_state.odds_loaded=False
    st.session_state.odds_loaded_at=None
    st.session_state.odds_scope=None

if "totals_payload" not in st.session_state:
    st.session_state.totals_payload={"events":[],"error":"","quota":{}}
    st.session_state.totals_loaded=False
    st.session_state.totals_scope=None

odds_payload=st.session_state.odds_payload if st.session_state.get("odds_loaded") else {"events":[],"error":"","quota":{}}
totals_payload=st.session_state.totals_payload if st.session_state.get("totals_loaded") else {"events":[],"error":"","quota":{}}

with st.spinner("Loading MLB schedule, starters, lineups and model data…"):
    games=fetch_games_for_date(slate_date)
    model_df=run_model(games) if games else pd.DataFrame()
    candidates=build_candidates(model_df,games,odds_payload.get("events",[])) if not model_df.empty else []

fresh_scoreboard = fetch_fresh_scoreboard(slate_date)

# Forward-test tracker: freeze the first official recommendation at the price that triggered it.
_new_ml = track_current_official_recommendations(candidates, games, slate_date)
_new_totals = track_current_total_recommendations(candidates, games, model_df, totals_payload, slate_date)
_graded_now = grade_tracker(force=False)
if _new_ml or _new_totals:
    st.toast(f"Tracked {_new_ml + _new_totals} new official model recommendation(s).")
if _graded_now:
    st.toast(f"Auto-graded {_graded_now} completed recommendation(s).")

quota=(totals_payload.get("quota",{}) if st.session_state.get("totals_loaded") else odds_payload.get("quota",{}))
if st.session_state.get("odds_loaded") or st.session_state.get("totals_loaded"):
    qtxt=f"Odds credits remaining: {quota.get('remaining')}" if quota.get("remaining") is not None else "Odds loaded manually"
else:
    qtxt="Market not loaded • 0 Odds API credits used"
priced_games=sum(1 for x in candidates if x.get("market_available") and x.get("pregame"))
fresh_states=[]
for _g0 in games:
    _gf=fresh_scoreboard.get(str(_g0.get("GamePk")),_g0)
    fresh_states.append(game_state(_gf))
pregame_games=sum(1 for s in fresh_states if s=="PREGAME")
live_games=sum(1 for s in fresh_states if s=="LIVE")
final_games=sum(1 for s in fresh_states if s=="FINAL")
render_auto_slate_status(games, slate_date)
lineup_auto_refresh_watcher(games)

if odds_payload.get("error"):
    st.error(odds_payload["error"])

if not games:
    st.info(f"No MLB games were returned for {slate_date.strftime('%B %-d, %Y')}.")
    st.stop()

if not candidates:
    st.warning("The model could not produce game rows for today.")
else:
    if "ninth_page" not in st.session_state:
        st.session_state["ninth_page"] = "Board"

    main_view = st.session_state.get("ninth_page", "Board")

    def _ninth_nav_button(label, slug):
        active = main_view == label
        key = f"ninth_nav_{slug}_{'active' if active else 'idle'}"
        if st.button(label, key=key, use_container_width=True):
            st.session_state["ninth_page"] = label
            st.rerun()

    _ninth_nav_button("Board", "board")
    _ninth_nav_button("Live", "live")
    _ninth_nav_button("Tracker", "tracker")
    _ninth_nav_button("Bets", "bets")
    _ninth_nav_button("More", "more")

    if main_view == "Live":
        render_auto_live_page(games, slate_date)
        st.stop()

    if main_view == "Tracker":
        render_auto_tracker_page(games, slate_date)
        st.caption("Live tracking only — no in-game betting recommendations.")
        st.stop()

    if main_view == "Bets":
        render_performance_page()
        st.stop()

    if main_view == "More":
        render_account_page()
        st.stop()

    st.markdown('<div class="board-head"><span>BETTING BOARD</span><b>Choose a workflow</b></div>', unsafe_allow_html=True)
    mode = st.radio(
        "View mode",
        ["Single Game", "Full Slate"],
        horizontal=True,
        label_visibility="collapsed",
        key="production_view_mode",
    )

    def start_sort(x):
        try:
            g=next(g for g in games if g.get("GamePk")==x["GamePk"])
            return pd.to_datetime(g.get("GameDate"),utc=True)
        except Exception:
            return pd.Timestamp.max.tz_localize("UTC")

    if mode == "Single Game":
        chrono = sorted(candidates, key=start_sort)
        upcoming_single = [x for x in chrono if x.get("pregame")]
        single_group = "Upcoming"
        single_pool = upcoming_single
        if not single_pool:
            st.info("No upcoming games remain. Use **Live** for scores or **Tracker** for tracked bets.")
            st.stop()
        labels = [f"{x['time']} • {x['away']} @ {x['home']}" + (f" • {x['game_state']}" if not x.get("pregame") else "") for x in single_pool]
        st.markdown('<div class="kicker">Matchup</div>', unsafe_allow_html=True)
        selected_label = st.selectbox("Choose matchup", labels, index=0, key="single_game_matchup", label_visibility="collapsed")
        x = single_pool[labels.index(selected_label)]
        selected_game = next((g for g in games if g.get("GamePk") == x["GamePk"]), None)

        selected_state = game_state(selected_game)
        if selected_state != "PREGAME":
            st.warning(game_state_label(selected_game) + ". Historical/pregame prices are not shown as actionable live bets.")
        pull_single = st.button(
            "Update This Game Odds",
            use_container_width=True,
            type="primary",
            disabled=(selected_state != "PREGAME"),
        )
        if pull_single:
            with st.spinner("Updating this game's moneyline + total…"):
                st.session_state.odds_payload = fetch_single_game_odds(api_key, selected_game)
                st.session_state.totals_payload = fetch_single_game_totals(api_key, selected_game)
            st.session_state.odds_loaded = True
            st.session_state.odds_loaded_at = pd.Timestamp.now(tz="America/New_York")
            st.session_state.odds_scope = f"single game: {x['away']} @ {x['home']}"
            st.session_state.totals_loaded = True
            st.session_state.totals_scope = f"single game total: {x['away']} @ {x['home']}"
            st.rerun()

        b = x["best"]
        away_side = next(z for z in x["all"] if z["team"] == x["away"])
        home_side = next(z for z in x["all"] if z["team"] == x["home"])
        lineup_text = "Lineups confirmed" if x["lineup_confirmed"] else f'Awaiting lineups • {x.get("lineup_teams_ready",0)}/2 teams posted'
        _trk_ok, _trk_reason = tracker_qualification(x, "MONEYLINE")

        st.markdown('<div class="kicker">Moneyline</div>', unsafe_allow_html=True)
        if x.get("market_available") and (x.get("best") or {}).get("selection") in ("BET","BEST BET"):
            if _trk_ok:
                st.caption("Tracker status: **QUALIFIED** — this recommendation is eligible to be frozen in forward performance.")
            else:
                st.caption(f"Tracker status: **EARLY SIGNAL** — not yet counted in headline performance ({_trk_reason}).")
        if x["market_available"]:
            st.markdown(f'''<div class="best-card"><div class="best-top"><div><div class="best-tag">{b['selection']}</div><div class="best-pick">{b['team']} ML {b['odds']:+d}</div><div class="best-game">{x['away']} @ {x['home']} • {x['time']} • Best price: {b['book']}</div></div><div class="badge {cls(b['selection'])}">{b['selection']}</div></div><div class="metrics"><div class="metric"><span>Win chance</span><b>{b['prob']*100:.1f}%</b></div><div class="metric"><span>Edge vs price</span><b>{b['edge']*100:+.1f}%</b></div><div class="metric"><span>EV</span><b>{b['ev']*100:+.1f}%</b></div><div class="metric"><span>Fair line</span><b>{b['fair']:+d}</b></div></div><div class="best-game" style="margin-top:10px">{lineup_text} • Model weight {x['alpha']*100:.0f}% / market {(1-x['alpha'])*100:.0f}% • {x['books']} books in consensus</div></div>''', unsafe_allow_html=True)
        else:
            fav = away_side if away_side['prob'] >= home_side['prob'] else home_side
            st.markdown(f'''<div class="best-card"><div class="best-top"><div><div class="best-tag">MODEL VIEW</div><div class="best-pick">{fav['team']} {fav['prob']*100:.1f}%</div><div class="best-game">{x['away']} @ {x['home']} • {x['time']} • Live moneyline not available</div></div><div class="badge badge-lean">MODEL ONLY</div></div><div class="metrics"><div class="metric"><span>{x['away']} win</span><b>{away_side['prob']*100:.1f}%</b></div><div class="metric"><span>{x['home']} win</span><b>{home_side['prob']*100:.1f}%</b></div><div class="metric"><span>{x['away']} fair</span><b>{away_side['fair']:+d}</b></div><div class="metric"><span>{x['home']} fair</span><b>{home_side['fair']:+d}</b></div></div><div class="best-game" style="margin-top:10px">{lineup_text} • Model confidence {x['confidence']}/100 • No BET/LEAN verdict without a live price</div></div>''', unsafe_allow_html=True)
            st.info("This game is modeled and selectable. A betting verdict appears automatically when a valid two-way moneyline is available.")

        st.markdown('<div class="kicker">Totals</div>', unsafe_allow_html=True)
        row_for_total=model_df.loc[model_df["GamePk"]==x["GamePk"]].iloc[0].to_dict()
        tctx=engine.totals_projection(row_for_total) if hasattr(engine,"totals_projection") else {"Projected_Total":x['away_proj']+x['home_proj'],"Base_Total":x['away_proj']+x['home_proj'],"Park_Factor":1.,"Weather_Factor":1.,"Weather_Available":False}
        tev=match_event(totals_payload.get("events",[]),selected_game) if st.session_state.get("totals_loaded") else None
        tm=totals_market(tev)
        raw_total=float(tctx["Projected_Total"])

        if tm:
            tp=build_total_pick(raw_total,tm)
            if tp is None:
                st.warning("A totals market was returned, but its price pair was incomplete/invalid. Refresh the total or try again later.")
            else:
                st.markdown(f'''<div class="best-card"><div class="best-top"><div><div class="best-tag">TOTALS • {tp["grade"]}</div><div class="best-pick">{tp["side"]} {tp["market_total"]:.1f} {tp["odds"]:+d}</div><div class="best-game">{tp["book"]} • Model {raw_total:.2f} • Calibrated {tp["calibrated_total"]:.2f} • {tp["books"]} books</div></div><div class="badge {cls(tp["grade"])}">{tp["grade"]}</div></div><div class="metrics"><div class="metric"><span>Bet probability</span><b>{tp["prob"]*100:.1f}%</b></div><div class="metric"><span>Edge</span><b>{tp["edge"]*100:+.1f}%</b></div><div class="metric"><span>EV</span><b>{tp["ev"]*100:+.1f}%</b></div><div class="metric"><span>Model weight</span><b>{TOTALS_MODEL_WEIGHT*100:.0f}%</b></div></div><div class="best-game" style="margin-top:10px">Over {tp["over_odds"]:+d} • {tp["over_prob"]*100:.1f}% | Under {tp["under_odds"]:+d} • {tp["under_prob"]*100:.1f}% • Park/weather are context only.</div></div>''',unsafe_allow_html=True)
        else:
            temp_txt = f'{float(tctx["Temp"]):.0f}°F' if tctx.get("Temp") is not None and pd.notna(tctx.get("Temp")) else "—"
            st.markdown(f'''<div class="best-card"><div class="best-top"><div><div class="best-tag">TOTALS MODEL VIEW</div><div class="best-pick">Projected total {raw_total:.2f}</div><div class="best-game">Load this game's total only when you want an official market grade.</div></div><div class="badge badge-lean">MODEL ONLY</div></div><div class="metrics"><div class="metric"><span>Projected total</span><b>{raw_total:.2f}</b></div><div class="metric"><span>Park context</span><b>{float(tctx.get("Park_Factor",1.0)):.3f}</b></div><div class="metric"><span>Temperature</span><b>{temp_txt}</b></div><div class="metric"><span>Lineups</span><b>{"CONFIRMED" if x["lineup_confirmed"] else "MODEL"}</b></div></div></div>''',unsafe_allow_html=True)


        st.markdown('<div class="kicker">More</div>', unsafe_allow_html=True)
        with st.expander("Download detailed game analysis", expanded=False):
            total_download_row = totals_download_row(row_for_total, tctx, tp if tm and 'tp' in locals() else None)
            total_download_df = pd.DataFrame([total_download_row])
            st.download_button(
                "Download Totals Detail",
                data=total_download_df.to_csv(index=False).encode("utf-8"),
                file_name=f"mlb_game_totals_{x['GamePk']}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"download_total_{x['GamePk']}",
            )
            diag = game_diagnostics_df(x, slate_date)
            st.download_button(
                "Download Full Game Analysis",
                diag.to_csv(index=False).encode("utf-8"),
                file_name=f"mlb_game_diagnostics_{x['GamePk']}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"download_diag_{x['GamePk']}",
            )
            st.caption("Use these files when you want a deeper breakdown in ChatGPT.")

    else:
        update_full_slate = st.button("Load Full Slate Lines", use_container_width=True, type="primary", key="update_full_slate_odds")
        if update_full_slate:
            fetch_odds.clear()
            fetch_full_slate_totals.clear()
            with st.spinner("Updating full-slate moneyline + totals…"):
                st.session_state.odds_payload = fetch_odds(api_key)
                st.session_state.totals_payload = fetch_full_slate_totals(api_key)
            st.session_state.odds_loaded = True
            st.session_state.odds_loaded_at = pd.Timestamp.now(tz="America/New_York")
            st.session_state.odds_scope = "full slate"
            st.session_state.totals_loaded = True
            st.session_state.totals_scope = "full slate totals"
            st.rerun()

        upcoming = sorted([x for x in candidates if x.get("pregame")], key=start_sort)
        live_now = sorted([x for x in candidates if x.get("game_state") == "LIVE"], key=start_sort)
        final_now = sorted([x for x in candidates if x.get("game_state") == "FINAL"], key=start_sort)

        if not st.session_state.get("odds_loaded") or not st.session_state.get("totals_loaded"):
            st.caption("Load current lines to activate Best Bet / Bet / Lean grades.")
        if not upcoming:
            st.info("No upcoming games remain on this slate.")
        else:
            # Build totals once so each game card can show ML + Total together.
            total_map = {}
            if st.session_state.get("totals_loaded"):
                for cx in upcoming:
                    mr = model_df.loc[model_df["GamePk"] == cx["GamePk"]]
                    if mr.empty:
                        continue
                    ctx = engine.totals_projection(mr.iloc[0].to_dict()) if hasattr(engine,"totals_projection") else {
                        "Projected_Total": cx["away_proj"] + cx["home_proj"]
                    }
                    game_obj = next((g for g in games if g.get("GamePk") == cx["GamePk"]), None)
                    ev = match_event(totals_payload.get("events", []), game_obj) if game_obj else None
                    tm = totals_market(ev)
                    tp = build_total_pick(float(ctx["Projected_Total"]), tm) if tm else None
                    total_map[cx["GamePk"]] = (tp, ctx)

            # Top Plays = strongest actionable markets only.
            # Upcoming Games below stays purely chronological for easy scanning.
            top_plays = []
            grade_rank = {"BEST BET":3, "BET":2, "LEAN":1}

            for cx in upcoming:
                b0 = cx.get("best") or {}
                if cx.get("market_available") and b0.get("selection") in grade_rank:
                    top_plays.append({
                        "game": cx,
                        "market": "ML",
                        "grade": b0.get("selection"),
                        "main": f'{b0.get("team")} ML {b0.get("odds"):+d}',
                        "book": b0.get("book"),
                        "edge": float(b0.get("edge") or 0),
                        "ev": float(b0.get("ev") or 0),
                    })

                tp0 = (total_map.get(cx["GamePk"]) or (None,None))[0]
                if tp0 and tp0.get("grade") in grade_rank:
                    top_plays.append({
                        "game": cx,
                        "market": "TOTAL",
                        "grade": tp0.get("grade"),
                        "main": f'{tp0.get("side")} {tp0.get("market_total"):.1f} {tp0.get("odds"):+d}',
                        "book": tp0.get("book"),
                        "edge": float(tp0.get("edge") or 0),
                        "ev": float(tp0.get("ev") or 0),
                    })

            top_plays = sorted(
                top_plays,
                key=lambda p: (
                    -grade_rank.get(p["grade"], 0),
                    -p["edge"],
                    -p["ev"],
                    start_sort(p["game"]),
                ),
            )

            st.markdown('<div class="kicker">Top Plays</div>', unsafe_allow_html=True)
            if top_plays:
                for n, p in enumerate(top_plays[:5], start=1):
                    gx = p["game"]
                    st.markdown(
                        f'<div class="top-play-card">'
                        f'<div class="top-play-rank">#{n} • {p["grade"]} • {p["market"]} • {gx["time"]}</div>'
                        f'<div class="top-play-main">{p["main"]}</div>'
                        f'<div class="top-play-sub">{gx["away"]} @ {gx["home"]} • {p["book"]} • Edge {p["edge"]*100:+.1f}% • EV {p["ev"]*100:+.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            elif st.session_state.get("odds_loaded") or st.session_state.get("totals_loaded"):
                st.caption("No BET / BEST BET / LEAN plays currently qualify.")
            else:
                st.caption("Update Full Slate Odds to rank the strongest current plays.")

            st.markdown('<div class="kicker">Upcoming Games — Chronological</div>', unsafe_allow_html=True)

            for cx in sorted(upcoming, key=start_sort):
                b = cx.get("best") or {}
                if cx.get("market_available"):
                    ml_grade = b.get("selection","PASS")
                    ml_main = f'{b.get("team")} ML {b.get("odds"):+d}' if b.get("odds") is not None else "Moneyline unavailable"
                    ml_sub = f'{b.get("book")} • Edge {b.get("edge",0)*100:+.1f}% • EV {b.get("ev",0)*100:+.1f}%'
                else:
                    ml_grade = "MODEL"
                    ml_main = "Model only"
                    ml_sub = f'Model fair: {cx["away"]} {fair_ml(next(z["prob"] for z in cx["all"] if z["team"]==cx["away"])):+d} / {cx["home"]} {fair_ml(next(z["prob"] for z in cx["all"] if z["team"]==cx["home"])):+d}'

                tp, tctx = total_map.get(cx["GamePk"], (None, None))
                if tp:
                    total_grade = tp.get("grade","PASS")
                    total_main = f'{tp.get("side")} {tp.get("market_total"):.1f} {tp.get("odds"):+d}'
                    total_sub = f'{tp.get("book")} • Edge {tp.get("edge",0)*100:+.1f}% • EV {tp.get("ev",0)*100:+.1f}%'
                else:
                    total_grade = "MODEL"
                    raw_total = None
                    mr = model_df.loc[model_df["GamePk"] == cx["GamePk"]]
                    if not mr.empty:
                        ctx0 = engine.totals_projection(mr.iloc[0].to_dict()) if hasattr(engine,"totals_projection") else {"Projected_Total":cx["away_proj"]+cx["home_proj"]}
                        raw_total = float(ctx0["Projected_Total"])
                    total_main = "Model only"
                    total_sub = f'Model total {raw_total:.2f}' if raw_total is not None else "Model total unavailable"

                def grade_class(g):
                    return {
                        "BEST BET":"grade-best","BET":"grade-bet","LEAN":"grade-lean",
                        "PASS":"grade-pass","MODEL":"grade-wait","MODEL ONLY":"grade-wait"
                    }.get(g,"grade-wait")

                lineup_label = cx.get("lineup_display") or ("LINEUPS CONFIRMED" if cx.get("lineup_confirmed") else "AWAITING LINEUPS • 0/2")
                away_lc = int(cx.get("away_lineup_count", 0) or 0)
                home_lc = int(cx.get("home_lineup_count", 0) or 0)
                lineup_diag = (
                    f'Lineup feed: {cx["away"]} {away_lc}/9 • {cx["home"]} {home_lc}/9'
                    if not cx.get("lineup_confirmed")
                    else 'Both starting lineups loaded • lineup adjustment active'
                )
                tracker_diag = tracker_candidate_status(cx, "MONEYLINE")
                tracker_text = (
                    "Tracker ready"
                    if tracker_diag["qualified"]
                    else f'Tracker waiting: {tracker_diag["reason"]}'
                )
                html = (
                    f'<div class="combo-card"><div class="combo-head"><div>'
                    f'<div class="combo-time">{cx["time"]} • {lineup_label}</div>'
                    f'<div class="combo-match">{cx["away"]} @ {cx["home"]}</div>'
                    f'<div class="combo-sp">{cx["away_sp"]} vs {cx["home_sp"]}</div>'
                    f'<div class="lineup-feed-diag">{lineup_diag}</div>'
                    f'<div class="tracker-gate-diag">{tracker_text}</div></div></div>'
                    f'<div class="market-row"><div class="market-name">ML</div><div><div class="market-main">{ml_main}</div>'
                    f'<div class="market-sub">{ml_sub}</div></div><div class="market-grade {grade_class(ml_grade)}">{ml_grade}</div></div>'
                    f'<div class="market-row"><div class="market-name">TOTAL</div><div><div class="market-main">{total_main}</div>'
                    f'<div class="market-sub">{total_sub}</div></div><div class="market-grade {grade_class(total_grade)}">{total_grade}</div></div>'
                    f'</div>'
                )
                st.markdown(html, unsafe_allow_html=True)

        st.markdown('<div class="kicker">Downloads</div>', unsafe_allow_html=True)
        with st.expander("Download detailed analysis", expanded=False):
            export_df = slate_export_df(candidates)
            st.download_button(
                "Download Moneyline Analysis",
                export_df.to_csv(index=False).encode("utf-8"),
                file_name="mlb_production_moneyline_board.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_full_slate_board_v150",
            )
            if st.session_state.get("totals_loaded"):
                totals_export_rows=[]
                for cx in upcoming:
                    mr=model_df.loc[model_df["GamePk"]==cx["GamePk"]]
                    if mr.empty: continue
                    row_dict=mr.iloc[0].to_dict()
                    ctx=engine.totals_projection(row_dict) if hasattr(engine,"totals_projection") else {"Projected_Total":cx['away_proj']+cx['home_proj']}
                    game_obj=next((g for g in games if g.get("GamePk")==cx["GamePk"]),None)
                    ev=match_event(totals_payload.get("events",[]),game_obj) if game_obj else None
                    tm=totals_market(ev)
                    tp=build_total_pick(float(ctx["Projected_Total"]),tm) if tm else None
                    totals_export_rows.append(totals_download_row(row_dict,ctx,tp))
                if totals_export_rows:
                    totals_export_df=pd.DataFrame(totals_export_rows)
                    st.download_button(
                        "Download Totals Analysis",
                        data=totals_export_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"mlb_totals_board_{slate_date.strftime('%Y-%m-%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_full_totals_csv_v150",
                    )
st.markdown('<div class="kicker">More</div>', unsafe_allow_html=True)
with st.expander("Model details & limitations", expanded=False):
    st.write("Moneyline uses the validated starting-pitcher + offense/platoon + lineup engine. Totals use the validated pitcher/run-environment framework. Bullpen and run lines remain excluded.")
    st.write("BET thresholds: moneyline BEST BET 10%+ edge, BET 7.5%+, LEAN 5%+. Totals BEST BET 12.5%+, BET 7.5%+, LEAN 5%+. Odds pulls remain manual-only.")
    st.write("Forward tracker counts only qualified pregame BET/BEST BET signals with confirmed lineups, confidence ≥80 and valid odds, then grades them from MLB final scores.")
    st.caption(f"App {APP_VERSION} • Engine {MODEL_VERSION}")
