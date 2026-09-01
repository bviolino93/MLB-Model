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

APP_VERSION = "2.0.1-FULL-WIDTH-NAV"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT_KEY = "baseball_mlb"

st.set_page_config(page_title="MLB Edge", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

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
        confirmed=bool(r["Away_Lineup_Used"] and r["Home_Lineup_Used"])
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
        "Lineups_Confirmed": x.get("lineup_confirmed"),
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
    return out[TRACKER_COLUMNS]

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
    if TRACKER_REQUIRE_CONFIRMED_LINEUPS and not candidate.get("lineup_confirmed"):
        return False, "lineups not confirmed"
    conf = int(candidate.get("confidence") or 0)
    min_conf = TRACKER_MIN_CONFIDENCE_TOTAL if str(market_type).upper() == "TOTAL" else TRACKER_MIN_CONFIDENCE_ML
    if conf < min_conf:
        return False, f"confidence {conf} < {min_conf}"
    return True, "qualified"

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
            "Lineups_Confirmed": x.get("lineup_confirmed"),
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
            "Lineups_Confirmed": x.get("lineup_confirmed"),
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
    if df.empty:
        return 0
    pending_mask = df["Result"].fillna("PENDING").astype(str).eq("PENDING")
    if not pending_mask.any():
        return 0

    now = pd.Timestamp.now(tz="UTC")
    last = st.session_state.get("_tracker_last_grade_check")
    if not force and last is not None:
        try:
            if (now - pd.Timestamp(last)).total_seconds() < 120:
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

def render_account_page():
    st.markdown(
        f'<div class="page-head">'
        f'<div class="page-kicker">MLB EDGE</div>'
        f'<div class="page-title">Account</div>'
        f'<div class="page-sub">App information, model version and advanced resources.</div>'
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


st.markdown(f"""
<div class="hero">
  <div class="eyebrow">MLB EDGE • PRODUCTION</div>
  <div class="title">MLB Edge</div>
  <div class="sub">Live MLB market intelligence with model-ranked plays and real-time tracking.</div>
  <div class="pill">MODEL LIVE • {APP_VERSION}</div>
</div>
""", unsafe_allow_html=True)

try:
    api_key=st.secrets.get("ODDS_API_KEY","")
except Exception:
    api_key=""

st.markdown('<div class="kicker">Date</div>', unsafe_allow_html=True)
slate_date=st.date_input(
    "Slate date",
    value=today_et(),
    min_value=today_et(),
    max_value=today_et()+timedelta(days=14),
    help="Current/upcoming MLB dates only.",
    label_visibility="collapsed",
)
free_refresh=st.button("Refresh Scores + Tracker (free)",use_container_width=True)
st.caption("Refreshes scores, game status and live tracker data for free. Odds only update when you press an odds button.")
if free_refresh:
    # A Streamlit button click already triggers a script rerun. Clear caches here and
    # continue through this same run so the selected bottom tab is preserved.
    _selected_view = st.session_state.get("main_navigation", "Home")
    st.cache_data.clear()
    try:
        fetch_fresh_scoreboard.clear()
    except Exception:
        pass
    try:
        fetch_live_win_probability.clear()
    except Exception:
        pass
    st.session_state["main_navigation"] = _selected_view

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
st.markdown(f'<div class="status"><div><span class="dot"></span><span class="live">MLB EDGE</span> &nbsp; {slate_date.strftime("%b %-d")} • {pregame_games} upcoming • {live_games} live • {final_games} final</div><div>{qtxt}</div></div>',unsafe_allow_html=True)

if odds_payload.get("error"):
    st.error(odds_payload["error"])

if not st.session_state.get("odds_loaded"):
    st.caption("Odds are not loaded yet. Choose a mode below, then update only what you want.")

if not games:
    st.info(f"No MLB games were returned for {slate_date.strftime('%B %-d, %Y')}.")
    st.stop()

if not candidates:
    st.warning("The model could not produce game rows for today.")
else:
    main_view = st.radio(
        "Main view",
        ["Home", "Live", "Tracker", "Bets", "Account"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )

    if main_view == "Live":
        render_live_games_page(games, fresh_scoreboard)
        st.stop()

    if main_view == "Tracker":
        tracker_df = load_tracker()
        render_live_scoreboard(games, fresh_scoreboard, tracker_df, slate_date)
        st.caption("Visual tracking only — no in-game recommendations.")
        st.stop()

    if main_view == "Bets":
        render_performance_page()
        st.stop()

    if main_view == "Account":
        render_account_page()
        st.stop()

    st.markdown('<div class="kicker">Betting Board</div>', unsafe_allow_html=True)
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

        st.caption("One tap updates both the moneyline and total for this game.")
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
        lineup_text = "Confirmed lineups" if x["lineup_confirmed"] else "Lineups not fully confirmed"
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
        st.markdown('<div class="kicker">Full Slate Odds</div>', unsafe_allow_html=True)
        update_full_slate = st.button("Update Full Slate Odds", use_container_width=True, type="primary", key="update_full_slate_odds")
        st.caption("One tap updates moneyline + totals for all upcoming games. Manual only.")
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
            st.caption("Model view shown below. Tap **Update Full Slate Odds** above for current BET / LEAN / PASS grades.")
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

                lineup_label = "LINEUPS ✓" if cx.get("lineup_confirmed") else "LINEUPS WAIT"
                html = (
                    f'<div class="combo-card"><div class="combo-head"><div>'
                    f'<div class="combo-time">{cx["time"]} • {lineup_label}</div>'
                    f'<div class="combo-match">{cx["away"]} @ {cx["home"]}</div>'
                    f'<div class="combo-sp">{cx["away_sp"]} vs {cx["home_sp"]}</div></div></div>'
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
