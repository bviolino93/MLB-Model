import re
import math
import time
import html
import statistics
import io
import json
import zipfile
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from scipy.stats import poisson

from model import (
    MODEL_VERSION,
    fetch_today_games,
    run_model,
    implied_prob,
    expected_value,
    fair_ml,
)

APP_VERSION = "0.14.1-PIT-BULLPEN-ROSTER-FIX"

st.set_page_config(page_title="MLB Model", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1rem; padding-bottom: 3rem;}
div[data-testid="stMetricValue"] {font-size: 1.65rem;}
.stButton > button {width: 100%; min-height: 3rem; font-weight: 700; border-radius: 10px;}
.bet-card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 14px 16px; margin: 8px 0;}
.bet-big {font-size: 1.15rem; font-weight: 800;}

.mlb-hero {
    padding: 18px 20px; margin: 2px 0 14px 0; border-radius: 18px;
    background: linear-gradient(135deg, rgba(17,34,58,.96), rgba(8,20,36,.98));
    border: 1px solid rgba(148,163,184,.14);
}
.mlb-kicker {font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:#8EA4BE; font-weight:800;}
.mlb-title {font-size:1.72rem; line-height:1.05; font-weight:900; color:#F5F8FC; margin-top:4px;}
.mlb-sub {font-size:.82rem; color:#9FB2C8; margin-top:7px;}

.slate-card {
    margin: 10px 0 4px 0; padding: 15px; border-radius: 16px;
    background: linear-gradient(180deg, rgba(14,28,48,.96), rgba(9,21,37,.98));
    border: 1px solid rgba(148,163,184,.13);
}
.slate-card-top {display:flex; align-items:flex-start; justify-content:space-between; gap:10px;}
.slate-time {font-size:.72rem; color:#8EA4BE; font-weight:750;}
.slate-matchup {font-size:1.02rem; color:#F2F6FB; font-weight:900; margin-top:3px;}
.slate-matchup span {color:#7D93AD; font-weight:700;}
.slate-badge {padding:5px 9px; border-radius:999px; font-size:.67rem; font-weight:900; white-space:nowrap;}
.badge-strong {background:rgba(35,197,94,.20); color:#91F2AE; border:1px solid rgba(35,197,94,.30);}
.badge-bet {background:rgba(34,197,94,.15); color:#8AE9A5; border:1px solid rgba(34,197,94,.22);}
.badge-lean {background:rgba(234,179,8,.14); color:#F7D76B; border:1px solid rgba(234,179,8,.24);}
.badge-pass {background:rgba(148,163,184,.10); color:#AEBAC8; border:1px solid rgba(148,163,184,.15);}

.slate-reco {margin-top:12px; padding:12px; border-radius:12px; background:rgba(5,16,30,.56);}
.slate-reco-label {font-size:.65rem; text-transform:uppercase; letter-spacing:.11em; color:#7890AA; font-weight:800;}
.slate-reco-value {font-size:1.08rem; color:#F6F9FC; font-weight:900; margin-top:2px;}
.slate-reco-meta {font-size:.76rem; color:#9EB0C4; margin-top:3px;}

.slate-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:10px;}
.slate-box {padding:9px 10px; border-radius:10px; background:rgba(255,255,255,.025); border:1px solid rgba(148,163,184,.08);}
.slate-box-label {font-size:.62rem; color:#7187A0; text-transform:uppercase; font-weight:800;}
.slate-box-value {font-size:.86rem; color:#EAF0F7; font-weight:850; margin-top:2px;}
.slate-footer {font-size:.70rem; color:#7E93AA; margin-top:10px;}
.slate-footer span {margin:0 5px;}

.market-row {margin:7px 0; padding:10px 11px; border-radius:10px; background:rgba(13,26,45,.72); border:1px solid rgba(148,163,184,.10);}
.market-row-title {color:#E8EEF8; font-size:.86rem; font-weight:800;}
.market-row-sub {margin-top:3px; color:#8EA4BE; font-size:.73rem; font-weight:650;}

@media (max-width: 720px) {
    .block-container {padding-left:.75rem; padding-right:.75rem;}
    .slate-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
    .mlb-title {font-size:1.45rem;}
}

/* ===== v0.9 production betting UI ===== */
.app-head{
  display:flex;justify-content:space-between;align-items:flex-end;gap:12px;
  padding:14px 15px;margin:2px 0 15px;border-radius:18px;
  background:linear-gradient(145deg,rgba(13,31,52,.99),rgba(7,18,32,.99));
  border:1px solid rgba(73,188,255,.15);box-shadow:0 14px 36px rgba(0,0,0,.20);
}
.app-eyebrow{font-size:.59rem;font-weight:950;letter-spacing:.16em;color:#67d1ff}
.app-head-title{font-size:1.35rem;font-weight:950;color:#f8fafc;letter-spacing:-.035em;margin-top:2px}
.app-head-sub{font-size:.67rem;color:#748da7;margin-top:3px}
.app-live{font-size:.58rem;font-weight:900;color:#98aec5;white-space:nowrap}
.section-kicker{
  margin:15px 0 8px;font-size:.66rem;font-weight:950;letter-spacing:.13em;
  color:#75ccee;text-transform:uppercase;
}
.topbet-card{
  display:grid;grid-template-columns:34px 38px 1fr 42px;gap:9px;align-items:start;
  padding:12px;margin:8px 0;border-radius:16px;
  background:linear-gradient(180deg,rgba(14,29,49,.97),rgba(9,21,37,.98));
  border:1px solid rgba(148,163,184,.10);
}
.topbet-card.a{border-color:rgba(34,197,94,.34)}
.topbet-card.b{border-color:rgba(56,189,248,.30)}
.topbet-card.c{border-color:rgba(250,204,21,.22)}
.topbet-rank{font-size:.82rem;font-weight:950;color:#7890aa;padding-top:3px}
.topbet-game{font-size:.60rem;font-weight:850;letter-spacing:.05em;color:#71869f;text-transform:uppercase}
.topbet-pick{font-size:1.00rem;font-weight:950;color:#f7f9fc;margin-top:3px}
.topbet-note{font-size:.66rem;color:#7890aa;margin-top:3px}
.topbet-grade{
  width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-size:.96rem;font-weight:950;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
}
.topbet-grade.a{border-color:rgba(34,197,94,.34)}
.topbet-grade.b{border-color:rgba(56,189,248,.30)}
.topbet-grade.c{border-color:rgba(250,204,21,.23)}
.topbet-metrics{grid-column:3 / 5;display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.topbet-metrics div{padding:6px 8px;border-radius:9px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.045)}
.topbet-metrics span{display:block;font-size:.53rem;font-weight:850;letter-spacing:.06em;color:#61768f;text-transform:uppercase}
.topbet-metrics b{display:block;font-size:.74rem;color:#dfe8f2;margin-top:2px}
.team-logo{object-fit:contain;border-radius:50%;background:#fff;padding:2px;box-shadow:0 2px 10px rgba(0,0,0,.20)}
.team-logo-fallback{border-radius:50%;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.10);color:#b7c7d8;font-size:.58rem;font-weight:950}
.game-market-row{display:grid;grid-template-columns:28px 30px 1fr;gap:8px;align-items:center;padding:9px 2px;border-bottom:1px solid rgba(148,163,184,.07)}
.game-market-row:last-child{border-bottom:none}
.game-market-rank{font-size:.66rem;font-weight:850;color:#657b93}
.game-market-grade{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);font-size:.74rem;font-weight:900;color:#e8eef6}
.game-market-pick{font-size:.84rem;font-weight:850;color:#e7eef7}
.game-market-meta{font-size:.64rem;color:#71869f;margin-top:2px}
.game-head{
  display:grid;grid-template-columns:1fr 25px 1fr;gap:8px;align-items:center;
  padding:10px 11px;margin:1px 0 10px;border-radius:13px;
  background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.055);
}
.game-team{display:flex;gap:9px;align-items:center;min-width:0}
.game-team.home{justify-content:flex-end;text-align:right}
.game-team span{display:block;font-size:.50rem;font-weight:850;letter-spacing:.07em;text-transform:uppercase;color:#667e97}
.game-team b{display:block;font-size:.76rem;font-weight:900;color:#ecf2f8}
.game-at{text-align:center;font-size:.70rem;font-weight:900;color:#58718b}
div[data-testid="stExpander"]{border-radius:14px !important;border:1px solid rgba(148,163,184,.09) !important;background:rgba(7,18,32,.50) !important}
@media(max-width:720px){
  .topbet-card{grid-template-columns:28px 34px 1fr 38px}
  .topbet-metrics{grid-column:1 / 5}
  .app-head-title{font-size:1.16rem}
}


/* ===== v0.9.2 CFB-parity visual system ===== */
:root{
  --bg:#06111f;
  --panel:#0b1728;
  --panel2:#0e1d31;
  --text:#eef5fb;
  --muted:#8fa3ba;
  --blue:#7dd3fc;
  --green:#86efac;
}
.stApp{
  background:
    radial-gradient(circle at 18% -4%, rgba(59,130,246,.18), transparent 30%),
    radial-gradient(circle at 88% 4%, rgba(56,189,248,.08), transparent 22%),
    linear-gradient(180deg,#071321 0%,#06111f 44%,#050d18 100%);
  color:var(--text);
}
.block-container{
  max-width:940px !important;
  padding-top:1rem !important;
  padding-bottom:4rem !important;
}
header[data-testid="stHeader"]{
  background:rgba(6,17,31,.76);
  backdrop-filter:blur(14px);
  border-bottom:1px solid rgba(148,163,184,.08);
}
h1,h2,h3{letter-spacing:-.03em}
[data-testid="stMarkdownContainer"] p{line-height:1.45}

/* Main product hero — mirrors CFB Edge */
.edge-hero{padding:20px 2px 12px}
.edge-kicker{
  font-size:.72rem;font-weight:900;letter-spacing:.18em;color:#7dd3fc;margin-bottom:6px
}
.edge-title{
  font-size:2.55rem;line-height:1;font-weight:900;letter-spacing:-.055em;color:#fff
}
.edge-subtitle{
  margin-top:10px;color:#8fa3ba;max-width:650px;font-size:.95rem
}
.version-pill{
  display:inline-flex;margin-top:12px;padding:5px 9px;border-radius:999px;
  background:rgba(59,130,246,.11);border:1px solid rgba(96,165,250,.22);
  color:#bfdbfe;font-size:.68rem;font-weight:800;letter-spacing:.05em
}
.status-strip{
  display:flex;justify-content:space-between;align-items:center;gap:10px;
  padding:10px 12px;margin:0 0 14px;border-radius:13px;
  background:rgba(11,23,40,.68);border:1px solid rgba(148,163,184,.10);
  color:#8fa3ba;font-size:.76rem
}
.status-live{display:flex;align-items:center;gap:7px;color:#86efac;font-weight:850}
.status-dot{
  width:7px;height:7px;border-radius:999px;background:#22c55e;
  box-shadow:0 0 0 4px rgba(34,197,94,.10)
}

/* Controls */
div[role="radiogroup"]{
  gap:8px;
}
div[role="radiogroup"] label{
  border-radius:12px !important;
}
.stButton>button{
  border-radius:13px !important;
  min-height:48px !important;
  font-weight:850 !important;
}
[data-baseweb="select"]>div,
[data-testid="stNumberInput"] input{
  border-radius:12px !important;
}
div[data-testid="stAlert"]{
  border-radius:14px;
}

/* Single game premium result */
.game-detail-head{
  display:grid;grid-template-columns:1fr 24px 1fr;gap:8px;align-items:center;
  padding:11px 12px;margin:4px 0 8px;border-radius:14px;
  background:rgba(255,255,255,.026);border:1px solid rgba(255,255,255,.055)
}
.game-team{display:flex;align-items:center;gap:9px;min-width:0}
.game-team.home{justify-content:flex-end;text-align:right}
.game-team span{
  display:block;color:#667e97;font-size:.51rem;font-weight:850;
  text-transform:uppercase;letter-spacing:.07em
}
.game-team b{
  display:block;color:#ecf2f8;font-size:.78rem;font-weight:900;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis
}
.game-at{text-align:center;color:#58718b;font-size:.72rem;font-weight:900}
.game-detail-sub{color:#71879f;font-size:.66rem;margin:0 2px 10px}

.proj-grid{
  display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:8px 0 14px
}
.proj-box{
  padding:10px;border-radius:11px;background:rgba(255,255,255,.028);
  border:1px solid rgba(255,255,255,.05)
}
.proj-box span{
  display:block;font-size:.56rem;color:#697f97;text-transform:uppercase;
  letter-spacing:.07em;font-weight:850
}
.proj-box b{display:block;margin-top:3px;color:#eaf1f8;font-size:.92rem;font-weight:900}

.result-hero{
  border-radius:20px;padding:16px;margin:9px 0 15px;
  background:linear-gradient(135deg,rgba(15,30,50,.96),rgba(8,20,35,.97));
  border:1px solid rgba(148,163,184,.11)
}
.result-hero.a{border-color:rgba(34,197,94,.32)}
.result-hero.b{border-color:rgba(56,189,248,.28)}
.result-hero.c{border-color:rgba(250,204,21,.22)}
.result-kicker{
  color:#70869f;font-size:.61rem;font-weight:900;letter-spacing:.11em;text-transform:uppercase
}
.result-row{
  display:grid;grid-template-columns:42px 1fr 44px;gap:11px;align-items:center;margin-top:7px
}
.result-pick{font-size:1.08rem;color:#f8fafc;font-weight:950;letter-spacing:-.02em}
.result-sub{font-size:.70rem;color:#7f94ae;margin-top:3px}
.result-grade{
  width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  font-size:1.05rem;font-weight:950;background:rgba(255,255,255,.045);
  border:1px solid rgba(255,255,255,.08)
}
.result-metrics{
  display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:11px
}
.result-metrics div{
  padding:7px 9px;border-radius:10px;background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.045)
}
.result-metrics span{
  display:block;color:#60768e;font-size:.54rem;font-weight:850;
  text-transform:uppercase;letter-spacing:.06em
}
.result-metrics b{display:block;margin-top:2px;color:#dfe8f2;font-size:.78rem}

/* CFB-like market board */
.market-board{display:flex;flex-direction:column;gap:9px;margin-top:8px}
.market-card{
  display:flex;align-items:center;gap:12px;padding:12px 13px;border-radius:15px;
  background:linear-gradient(180deg,rgba(14,29,49,.92),rgba(10,23,40,.92));
  border:1px solid rgba(148,163,184,.10)
}
.market-grade{
  flex:0 0 38px;width:38px;height:38px;border-radius:11px;
  display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:950;
  color:#fff;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08)
}
.market-grade.a{border-color:rgba(34,197,94,.28)}
.market-grade.b{border-color:rgba(56,189,248,.28)}
.market-grade.c{border-color:rgba(250,204,21,.23)}
.market-grade.d{border-color:rgba(148,163,184,.13)}
.market-main{min-width:0;flex:1}
.market-pick{
  font-size:.98rem;font-weight:850;color:#f8fafc;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis
}
.market-sub{margin-top:3px;font-size:.72rem;color:#7f94ae}
.market-tag{margin-left:auto;font-size:.62rem;font-weight:900;letter-spacing:.08em;color:#94a3b8}

/* Slate header/card parity */
.app-head{
  display:flex;align-items:flex-end;justify-content:space-between;gap:12px;
  padding:14px 15px;margin:2px 0 14px;border-radius:18px;
  background:linear-gradient(145deg,rgba(13,31,52,.99),rgba(7,18,32,.99));
  border:1px solid rgba(73,188,255,.16);box-shadow:0 14px 36px rgba(0,0,0,.22)
}
.app-live{display:flex;align-items:center;gap:6px}
.app-live::before{
  content:"";width:7px;height:7px;border-radius:50%;background:#22c55e;
  box-shadow:0 0 10px rgba(34,197,94,.65)
}
.section-kicker{
  font-size:.72rem !important;letter-spacing:.17em !important;font-weight:900 !important;
  color:#7dd3fc !important;margin-top:23px !important;margin-bottom:6px !important
}
.topbet-card{
  border-radius:15px !important;
  background:linear-gradient(180deg,rgba(14,29,49,.97),rgba(9,21,37,.97)) !important
}
.game-market-grade.a{border-color:rgba(34,197,94,.30)}
.game-market-grade.b{border-color:rgba(56,189,248,.28)}
.game-market-grade.c{border-color:rgba(250,204,21,.22)}
div[data-testid="stExpander"]{
  border-radius:15px !important;border:1px solid rgba(148,163,184,.09) !important;
  background:rgba(7,18,32,.50) !important;overflow:hidden
}
div[data-testid="stExpander"] summary{min-height:49px}

@media(max-width:700px){
  .block-container{padding-left:1rem !important;padding-right:1rem !important}
  .edge-title{font-size:2.15rem}
  .status-strip{align-items:flex-start;flex-direction:column;gap:5px}
  .proj-grid{grid-template-columns:repeat(3,1fr)}
  .result-row{grid-template-columns:36px 1fr 40px}
  .result-metrics{grid-template-columns:repeat(3,1fr)}
  .market-card{padding:11px 12px}
  .market-pick{font-size:.94rem}
}


/* ===== v0.9.3 CLEAN UX ===== */
.bet-pill{
  display:inline-flex;align-items:center;justify-content:center;
  min-width:74px;padding:6px 10px;border-radius:999px;
  font-size:.62rem;font-weight:950;letter-spacing:.08em;
  border:1px solid rgba(148,163,184,.12);
  background:rgba(255,255,255,.035);color:#cbd5e1;
}
.bet-pill.best{
  color:#bbf7d0;background:rgba(34,197,94,.10);border-color:rgba(34,197,94,.30);
}
.bet-pill.bet{
  color:#bae6fd;background:rgba(56,189,248,.10);border-color:rgba(56,189,248,.28);
}
.bet-pill.lean{
  color:#fde68a;background:rgba(250,204,21,.08);border-color:rgba(250,204,21,.20);
}
.bet-pill.pass{
  color:#94a3b8;background:rgba(148,163,184,.05);border-color:rgba(148,163,184,.11);
}

/* Single-game top play: much tighter and no logo collision */
.clean-top-card{
  padding:14px 14px 13px;margin:7px 0 12px;border-radius:17px;
  background:linear-gradient(145deg,rgba(13,29,49,.98),rgba(8,20,34,.98));
  border:1px solid rgba(148,163,184,.10);
}
.clean-top-card.best{border-color:rgba(34,197,94,.30)}
.clean-top-card.bet{border-color:rgba(56,189,248,.28)}
.clean-top-card.lean{border-color:rgba(250,204,21,.20)}
.clean-top-head{
  display:flex;align-items:center;justify-content:space-between;gap:10px;
}
.clean-matchup{
  display:flex;align-items:center;gap:8px;min-width:0;
  color:#8297af;font-size:.66rem;font-weight:850;letter-spacing:.06em;text-transform:uppercase
}
.clean-matchup-logos{display:flex;align-items:center;gap:3px;flex:0 0 auto}
.clean-top-pick{
  margin-top:9px;color:#f8fafc;font-size:1.16rem;font-weight:950;letter-spacing:-.025em
}
.clean-top-sub{
  margin-top:3px;color:#7890a8;font-size:.66rem
}
.clean-top-metrics{
  display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:11px
}
.clean-top-metrics div{
  padding:7px 8px;border-radius:9px;background:rgba(255,255,255,.022);
  border:1px solid rgba(255,255,255,.04)
}
.clean-top-metrics span{
  display:block;font-size:.50rem;color:#61778e;font-weight:850;
  letter-spacing:.06em;text-transform:uppercase
}
.clean-top-metrics b{
  display:block;margin-top:2px;color:#e2e8f0;font-size:.76rem;font-weight:900
}

/* Actionable market rows */
.action-row{
  display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;
  padding:10px 2px;border-bottom:1px solid rgba(148,163,184,.07)
}
.action-row:last-child{border-bottom:none}
.action-main{min-width:0}
.action-pick{
  color:#eef5fb;font-size:.90rem;font-weight:900;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis
}
.action-meta{
  margin-top:2px;color:#71869f;font-size:.63rem
}
.action-type{
  color:#71869f;font-size:.56rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase
}

/* Pass rows are intentionally subdued */
.pass-row{
  display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;
  padding:9px 1px;border-bottom:1px solid rgba(148,163,184,.055)
}
.pass-row:last-child{border-bottom:none}
.pass-pick{font-size:.82rem;font-weight:800;color:#b5c2d1}
.pass-meta{margin-top:2px;font-size:.60rem;color:#61758c}
.pass-type{
  font-size:.54rem;font-weight:900;letter-spacing:.08em;color:#5f7389;text-transform:uppercase
}

/* Make manual tools visually secondary */
.advanced-note{
  color:#6f8398;font-size:.63rem;margin:5px 0 2px
}
div[data-testid="stExpander"] details summary p{
  font-weight:800;
}

/* Slate cards: replace grade box emphasis with verdict chip emphasis */
.topbet-card{
  grid-template-columns:30px 36px 1fr auto !important;
  padding:11px 12px !important;
}
.topbet-grade{
  width:auto !important;height:auto !important;min-width:68px !important;
  padding:6px 9px !important;border-radius:999px !important;
  font-size:.58rem !important;letter-spacing:.07em !important;
}
.topbet-grade.best{
  color:#bbf7d0;background:rgba(34,197,94,.09);border-color:rgba(34,197,94,.28)
}
.topbet-grade.bet{
  color:#bae6fd;background:rgba(56,189,248,.09);border-color:rgba(56,189,248,.27)
}
.topbet-grade.lean{
  color:#fde68a;background:rgba(250,204,21,.07);border-color:rgba(250,204,21,.20)
}
.topbet-grade.pass{
  color:#94a3b8;background:rgba(148,163,184,.05);border-color:rgba(148,163,184,.11)
}
.topbet-metrics{
  grid-column:3 / 5 !important;
}
.game-market-grade{
  width:auto !important;height:auto !important;min-width:62px !important;
  padding:5px 8px !important;border-radius:999px !important;
  font-size:.55rem !important;letter-spacing:.06em !important
}
.game-market-grade.best{
  color:#bbf7d0;border-color:rgba(34,197,94,.28);background:rgba(34,197,94,.08)
}
.game-market-grade.bet{
  color:#bae6fd;border-color:rgba(56,189,248,.27);background:rgba(56,189,248,.08)
}
.game-market-grade.lean{
  color:#fde68a;border-color:rgba(250,204,21,.20);background:rgba(250,204,21,.06)
}
.game-market-grade.pass{
  color:#94a3b8;border-color:rgba(148,163,184,.11);background:rgba(148,163,184,.04)
}
.game-market-row{
  grid-template-columns:26px auto 1fr !important;
  padding:8px 2px !important
}

/* Reduce vertical bulk */
.section-kicker{
  margin-top:18px !important;
}
.app-head{
  padding:12px 14px !important;
  margin-bottom:11px !important
}
.proj-grid{
  margin-bottom:10px !important
}

@media(max-width:700px){
  .clean-top-pick{font-size:1.08rem}
  .clean-top-metrics{grid-template-columns:repeat(3,1fr)}
  .action-row{grid-template-columns:auto 1fr}
  .action-type{grid-column:2}
  .topbet-card{grid-template-columns:26px 32px 1fr auto !important}
  .topbet-metrics{grid-column:1 / 5 !important}
}


/* ===== v0.9.4 STREAMLINED ===== */
.stButton > button{
  background:rgba(15,29,47,.92) !important;
  color:#d7e2ee !important;
  border:1px solid rgba(148,163,184,.16) !important;
  box-shadow:none !important;
}
.stButton > button:disabled{
  background:rgba(15,29,47,.42) !important;
  color:#607388 !important;
  border-color:rgba(148,163,184,.08) !important;
  opacity:1 !important;
}
.custom-result-summary{
  margin-top:9px;padding:10px 12px;border-radius:12px;
  background:rgba(255,255,255,.025);
  border:1px solid rgba(148,163,184,.08);
}
.custom-result-summary b{color:#edf4fb;font-size:.84rem}
.custom-result-summary span{
  display:block;margin-top:3px;color:#758ba2;font-size:.63rem;line-height:1.45
}


/* ===== v0.10 BACKTEST LAB ===== */
.bt-note{
  padding:10px 12px;border-radius:12px;margin:5px 0 12px;
  background:rgba(56,189,248,.055);border:1px solid rgba(56,189,248,.13);
  color:#9fb5ca;font-size:.68rem;line-height:1.5
}
.bt-note b{color:#d8edf8}
.bt-metrics{
  display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:8px 0 13px
}
.bt-metric{
  padding:9px;border-radius:11px;background:rgba(255,255,255,.025);
  border:1px solid rgba(148,163,184,.07)
}
.bt-metric span{
  display:block;color:#657b92;font-size:.52rem;font-weight:850;
  text-transform:uppercase;letter-spacing:.06em
}
.bt-metric b{display:block;margin-top:3px;color:#edf4fb;font-size:.88rem;font-weight:900}
.bt-good{color:#86efac !important}
.bt-bad{color:#fca5a5 !important}
@media(max-width:700px){.bt-metrics{grid-template-columns:repeat(2,1fr)}}


/* ===== v0.10.1 HISTORICAL BUILDER ===== */
.credit-box{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:8px 0 12px}
.credit-cell{padding:10px;border-radius:11px;background:rgba(255,255,255,.025);border:1px solid rgba(148,163,184,.08)}
.credit-cell span{display:block;color:#6f879e;font-size:.50rem;font-weight:850;text-transform:uppercase;letter-spacing:.06em}
.credit-cell b{display:block;color:#edf4fb;font-size:.90rem;margin-top:3px}
.guard-box{padding:10px 12px;border-radius:11px;background:rgba(34,197,94,.04);border:1px solid rgba(34,197,94,.12);color:#9db4c8;font-size:.68rem;line-height:1.5}
.guard-box b{color:#bbf7d0}
@media(max-width:700px){.credit-box{grid-template-columns:1fr}}


/* ===== v0.10.2 SMART HISTORY / MOBILE READABILITY ===== */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stWidgetLabel"] p,
.stMultiSelect label p,
.stDateInput label p,
.stNumberInput label p,
.stCheckbox label p,
.stFileUploader label p {
  color:#dce8f3 !important;
  opacity:1 !important;
}
.phase-grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:7px;
  margin:8px 0 10px;
}
.phase-cell{
  padding:10px;
  border-radius:11px;
  background:rgba(255,255,255,.025);
  border:1px solid rgba(148,163,184,.10);
}
.phase-cell.recommended{
  background:rgba(34,197,94,.05);
  border-color:rgba(34,197,94,.20);
}
.phase-cell span{
  display:block;
  color:#91a7ba;
  font-size:.50rem;
  font-weight:850;
  text-transform:uppercase;
  letter-spacing:.06em;
}
.phase-cell b{
  display:block;
  color:#f1f7fb;
  font-size:.80rem;
  margin-top:2px;
}
.phase-cell em{
  display:block;
  color:#b0c5d6;
  font-size:.63rem;
  font-style:normal;
  margin-top:2px;
}
@media(max-width:700px){
  .phase-grid{grid-template-columns:1fr}
  [data-testid="stExpander"] summary{
    background:#102236 !important;
  }
}

</style>
""", unsafe_allow_html=True)





ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT_KEY = "baseball_mlb"

def get_odds_api_key():
    try:
        return str(st.secrets.get("ODDS_API_KEY", "")).strip()
    except Exception:
        return ""

def _team_key(name):
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    aliases = {
        "oaklandathletics": "athletics",
        "athletics": "athletics",
        "losangelesangels": "losangelesangels",
        "laangels": "losangelesangels",
        "arizonadiamondbacks": "arizonadiamondbacks",
        "dbacks": "arizonadiamondbacks",
        "chicagowhitesox": "chicagowhitesox",
        "whitesox": "chicagowhitesox",
        "bostonredsox": "bostonredsox",
        "redsox": "bostonredsox",
        "torontobluejays": "torontobluejays",
        "bluejays": "torontobluejays",
        "tampabayrays": "tampabayrays",
    }
    return aliases.get(s, s)

def _median_num(vals):
    vals = [float(x) for x in vals if x is not None]
    return float(median(vals)) if vals else None

@st.cache_data(ttl=90, show_spinner=False)
def fetch_general_mlb_odds(api_key):
    """
    Current general MLB market from The Odds API.
    One US-region request returns moneyline, run line and total markets.
    Never surface a raw requests exception because it can include the secret key.
    """
    if not api_key:
        return {
            "events": [],
            "error": "ODDS_API_KEY is not configured.",
            "error_code": "missing_key",
            "quota": {},
        }

    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=25,
        )
    except requests.RequestException:
        return {
            "events": [],
            "error": "Could not reach The Odds API.",
            "error_code": "network",
            "quota": {},
        }

    quota = {
        "remaining": r.headers.get("x-requests-remaining"),
        "used": r.headers.get("x-requests-used"),
        "last": r.headers.get("x-requests-last"),
    }

    if r.status_code == 401:
        return {
            "events": [],
            "error": "The Odds API rejected ODDS_API_KEY (401 Unauthorized). The configured key is invalid, expired, revoked, or was not updated in this Streamlit app.",
            "error_code": "unauthorized",
            "quota": quota,
        }

    if r.status_code == 429:
        return {
            "events": [],
            "error": "The Odds API request/credit limit was reached (429).",
            "error_code": "rate_limit",
            "quota": quota,
        }

    if r.status_code >= 400:
        return {
            "events": [],
            "error": f"The Odds API returned HTTP {r.status_code}.",
            "error_code": f"http_{r.status_code}",
            "quota": quota,
        }

    try:
        events = r.json()
    except Exception:
        return {
            "events": [],
            "error": "The Odds API returned an unreadable response.",
            "error_code": "bad_json",
            "quota": quota,
        }

    return {
        "events": events if isinstance(events, list) else [],
        "error": "",
        "error_code": "",
        "quota": quota,
    }


def _event_match_score(event, game):
    if _team_key(event.get("away_team")) != _team_key(game.get("Away")):
        return None
    if _team_key(event.get("home_team")) != _team_key(game.get("Home")):
        return None

    # Doubleheaders can have the same two teams twice. Match the nearest start time.
    try:
        e = pd.to_datetime(event.get("commence_time"), utc=True)
        g = pd.to_datetime(game.get("GameDate"), utc=True)
        return abs((e - g).total_seconds())
    except Exception:
        return 0.0

def match_odds_event(events, game):
    choices = []
    for e in events:
        score = _event_match_score(e, game)
        if score is not None:
            choices.append((score, e))
    if not choices:
        return None
    choices.sort(key=lambda x: x[0])
    return choices[0][1]

def consensus_from_event(event):
    """
    Build a general/consensus line by taking the median across available books.
    For spreads/totals, the consensus point is the median point and the consensus
    price is the median price among books at the closest available point.
    """
    if not event:
        return None

    h2h = {}
    spreads = {}
    totals = {"Over": [], "Under": []}
    providers = set()
    updates = []

    for book in event.get("bookmakers", []):
        providers.add(book.get("title") or book.get("key") or "book")
        if book.get("last_update"):
            updates.append(book.get("last_update"))
        for market in book.get("markets", []):
            key = market.get("key")
            if market.get("last_update"):
                updates.append(market.get("last_update"))

            if key == "h2h":
                for o in market.get("outcomes", []):
                    name = o.get("name")
                    price = o.get("price")
                    if name is not None and price is not None:
                        h2h.setdefault(_team_key(name), []).append(float(price))

            elif key == "spreads":
                for o in market.get("outcomes", []):
                    name = o.get("name")
                    point = o.get("point")
                    price = o.get("price")
                    if name is not None and point is not None and price is not None:
                        spreads.setdefault(_team_key(name), []).append((float(point), float(price)))

            elif key == "totals":
                for o in market.get("outcomes", []):
                    name = str(o.get("name", "")).title()
                    point = o.get("point")
                    price = o.get("price")
                    if name in totals and point is not None and price is not None:
                        totals[name].append((float(point), float(price)))

    away_key = _team_key(event.get("away_team"))
    home_key = _team_key(event.get("home_team"))

    out = {
        "event_id": event.get("id"),
        "commence_time": event.get("commence_time"),
        "away_team": event.get("away_team"),
        "home_team": event.get("home_team"),
        "provider_count": len(providers),
        "providers": ", ".join(sorted(providers)),
        "last_update": max(updates) if updates else None,
        "away_ml": None,
        "home_ml": None,
        "away_rl": None,
        "away_rl_odds": None,
        "home_rl": None,
        "home_rl_odds": None,
        "total": None,
        "over_odds": None,
        "under_odds": None,
    }

    if h2h.get(away_key):
        out["away_ml"] = int(round(_median_num(h2h[away_key])))
    if h2h.get(home_key):
        out["home_ml"] = int(round(_median_num(h2h[home_key])))

    def spread_consensus(items):
        if not items:
            return None, None
        point = _median_num([x[0] for x in items])
        nearest = min(abs(x[0] - point) for x in items)
        prices = [x[1] for x in items if abs(abs(x[0] - point) - nearest) < 1e-9]
        return float(point), int(round(_median_num(prices)))

    out["away_rl"], out["away_rl_odds"] = spread_consensus(spreads.get(away_key, []))
    out["home_rl"], out["home_rl_odds"] = spread_consensus(spreads.get(home_key, []))

    all_total_points = [x[0] for side in totals.values() for x in side]
    if all_total_points:
        total_line = _median_num(all_total_points)
        out["total"] = float(total_line)

        for side_name, odds_key in [("Over", "over_odds"), ("Under", "under_odds")]:
            items = totals[side_name]
            if items:
                nearest = min(abs(x[0] - total_line) for x in items)
                prices = [x[1] for x in items if abs(abs(x[0] - total_line) - nearest) < 1e-9]
                out[odds_key] = int(round(_median_num(prices)))

    return out

def _market_for_game_raw(games, events, game_pk):
    game = next((g for g in games if g["GamePk"] == game_pk), None)
    if not game:
        return None
    return consensus_from_event(match_odds_event(events, game))


def _clean_consensus_market(market):
    """
    Sanitize a parsed market and never manufacture invalid odds.
    Expected optional source arrays can be used when present; otherwise
    the existing consensus values are sanitized in place.
    """
    if not market:
        return market

    m = dict(market)

    # Sanitize already-aggregated values.
    for key in ["away_ml", "home_ml", "away_rl_odds", "home_rl_odds", "over_odds", "under_odds"]:
        if key in m:
            m[key] = sanitize_market_price(m.get(key))

    # If book-level arrays exist, recompute robust median and best prices.
    array_map = {
        "away_ml_prices": "away_ml",
        "home_ml_prices": "home_ml",
        "away_rl_odds_prices": "away_rl_odds",
        "home_rl_odds_prices": "home_rl_odds",
        "over_odds_prices": "over_odds",
        "under_odds_prices": "under_odds",
    }
    for arr_key, out_key in array_map.items():
        if arr_key in m and isinstance(m.get(arr_key), (list, tuple)):
            med = median_valid(m.get(arr_key))
            if med is not None:
                m[out_key] = med
            m[out_key + "_best"] = best_price_for_bettor(m.get(arr_key))

    # Book counts, if parser didn't already supply them.
    if not m.get("ml_books_count"):
        ml_arrays = []
        for k in ["away_ml_prices", "home_ml_prices"]:
            if isinstance(m.get(k), (list, tuple)):
                ml_arrays.extend([sanitize_market_price(v) for v in m[k]])
        if ml_arrays:
            m["ml_books_count"] = max(1, len([v for v in ml_arrays if v is not None]) // 2)

    if not m.get("rl_books_count"):
        rl_arrays = []
        for k in ["away_rl_odds_prices", "home_rl_odds_prices"]:
            if isinstance(m.get(k), (list, tuple)):
                rl_arrays.extend([sanitize_market_price(v) for v in m[k]])
        if rl_arrays:
            m["rl_books_count"] = max(1, len([v for v in rl_arrays if v is not None]) // 2)

    if not m.get("total_books_count"):
        t_arrays = []
        for k in ["over_odds_prices", "under_odds_prices"]:
            if isinstance(m.get(k), (list, tuple)):
                t_arrays.extend([sanitize_market_price(v) for v in m[k]])
        if t_arrays:
            m["total_books_count"] = max(1, len([v for v in t_arrays if v is not None]) // 2)

    # Explicitly invalidate impossible pair states instead of fabricating zero.
    if m.get("away_ml") is None and m.get("home_ml") is None:
        m["away_ml"] = None
        m["home_ml"] = None
    if m.get("away_rl_odds") is None and m.get("home_rl_odds") is None:
        m["away_rl_odds"] = None
        m["home_rl_odds"] = None
    if m.get("over_odds") is None and m.get("under_odds") is None:
        m["over_odds"] = None
        m["under_odds"] = None

    return m


def market_for_game(games, events, game_pk):
    return _clean_consensus_market(_market_for_game_raw(games, events, game_pk))

def _verdict_rank(v):
    return {"STRONG BET": 4, "BET": 3, "LEAN": 2, "PASS": 1, "NO LINE": 0}.get(str(v), 0)

def _verdict_class(v):
    return {
        "STRONG BET": "badge-strong",
        "BET": "badge-bet",
        "LEAN": "badge-lean",
    }.get(str(v), "badge-pass")


def nickname(team):
    special = {"Boston Red Sox": "Red Sox", "Chicago White Sox": "White Sox", "Toronto Blue Jays": "Blue Jays"}
    return special.get(team, team.split()[-1])


def clean_ocr_image(image):
    img = ImageOps.exif_transpose(image).convert("L")
    if img.width < 1800:
        scale = 1800 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img = ImageOps.autocontrast(img)
    return img.filter(ImageFilter.SHARPEN)


def ocr_text(image):
    img = clean_ocr_image(image)
    # Use two segmentation modes because sportsbook screenshots often mix
    # compact tables and isolated labels/prices.
    a = pytesseract.image_to_string(img, config="--psm 6")
    b = pytesseract.image_to_string(img, config="--psm 11")
    return a + "\n\n===== OCR ALT PASS =====\n\n" + b


def american_numbers(text):
    vals = []
    for m in re.finditer(r'(?<!\d)([+-]\s?\d{2,4})(?!\d)', text):
        try:
            n = int(m.group(1).replace(" ", ""))
            if 100 <= abs(n) <= 1000:
                vals.append(n)
        except Exception:
            pass
    return vals


def nearby_odds(text, key, window=180):
    low = text.lower()
    idx = low.find(key.lower())
    if idx < 0:
        return []
    return american_numbers(text[max(0, idx - 30):idx + window])


def normalize_734_text(text):
    t = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
        .replace("½", ".5")
        .replace("1/2", ".5")
    )

    # 734's small ½ glyph is commonly read by Tesseract as %, y%, '%, '4, etc.
    # Examples seen in actual screenshots:
    #   o8½  -> 08% / o8% / o8y%
    #   u8½  -> u8% / u8y%
    #   -1½  -> -1'% / -1'4
    #   +1½  -> +1%
    #
    # Normalize those OCR shapes before market parsing.
    t = re.sub(r"(?i)\b[o0]\s*([6-9]|1[0-4])\s*y?\s*%", r"o\1.5", t)
    t = re.sub(r"(?i)\bu\s*([6-9]|1[0-4])\s*y?\s*%", r"u\1.5", t)

    # Full-game run line 1½.
    t = re.sub(r"(?<!\d)([+-])\s*1\s*['’`´]?\s*[%4](?!\d)", r"\g<1>1.5", t)
    t = re.sub(r"(?<!\d)([+-])\s*1\s*[.,|:/]\s*5\b", r"\g<1>1.5", t)
    t = re.sub(r"(?<!\d)([+-])\s*1\s+5\b", r"\g<1>1.5", t)
    t = re.sub(r"(?<!\d)([+-])15(?!\d)", r"\g<1>1.5", t)

    # Occasionally "o" is recognized as zero.
    t = re.sub(r"(?i)(?<!\w)0(?=\s*(?:[6-9]|1[0-4])(?:\.5)?)", "o", t)

    return t


def valid_total(x):
    try:
        x = float(x)
        # Full-game MLB totals normally live in this range. This intentionally
        # rejects F5 totals such as 4.5.
        return 6.0 <= x <= 14.5
    except Exception:
        return False



def _ocr_tokens(image):
    """
    OCR with coordinates. 734 is a fixed table layout, so row position is more
    reliable than trying to understand every glyph in the screenshot.
    """
    img = clean_ocr_image(image)
    data = pytesseract.image_to_data(
        img,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )

    tokens = []
    n = len(data["text"])
    for i in range(n):
        raw = str(data["text"][i]).strip()
        if not raw:
            continue
        tokens.append({
            "text": raw,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "cx": int(data["left"][i]) + int(data["width"][i]) / 2,
            "cy": int(data["top"][i]) + int(data["height"][i]) / 2,
        })
    return tokens


def _token_american_odds(s):
    s = (
        str(s)
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .replace(",", "")
        .strip()
    )
    m = re.search(r"([+-])\s*(\d{3,4})", s)
    if not m:
        return None
    val = int(m.group(1) + m.group(2))
    return val if 100 <= abs(val) <= 1000 else None


def _find_team_market_row(tokens, team):
    """
    Find the first 734 full-game row for a team. We specifically require at
    least two American prices on the same horizontal band; that skips the
    schedule header and selects the full-game row before the 1H row.
    """
    keys = {team.lower(), nickname(team).lower()}
    candidates = []

    for tok in tokens:
        tt = re.sub(r"[^a-z]", "", tok["text"].lower())
        if not tt:
            continue

        matched = False
        for key in keys:
            kk = re.sub(r"[^a-z]", "", key)
            if kk and (kk in tt or tt in kk):
                matched = True
                break
        if not matched:
            continue

        y = tok["cy"]
        band = [
            x for x in tokens
            if abs(x["cy"] - y) <= max(24, tok["height"] * 0.9)
        ]

        odds = []
        for x in band:
            val = _token_american_odds(x["text"])
            if val is not None:
                odds.append((x["left"], val, x["text"]))

        # Sometimes sign and digits get split into adjacent OCR tokens.
        ordered = sorted(band, key=lambda z: z["left"])
        for j in range(len(ordered) - 1):
            combo = ordered[j]["text"] + ordered[j + 1]["text"]
            val = _token_american_odds(combo)
            if val is not None:
                odds.append((ordered[j]["left"], val, combo))

        # Deduplicate by x/price.
        unique = []
        seen = set()
        for item in sorted(odds, key=lambda z: z[0]):
            sig = (round(item[0] / 10), item[1])
            if sig not in seen:
                seen.add(sig)
                unique.append(item)

        if len(unique) >= 2:
            candidates.append((tok["top"], unique, band))

    if not candidates:
        return None

    # First qualifying occurrence is the full-game row; later one is 1H.
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def _tess_read_variants(image, whitelist=None):
    """
    Lightweight multi-pass Tesseract reader.
    Tries several threshold/segmentation variants and returns all non-empty reads.
    """
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)

    variants = [
        gray,
        gray.point(lambda p: 255 if p > 145 else 0),
        gray.point(lambda p: 255 if p > 175 else 0),
    ]

    configs = ["--psm 7", "--psm 6", "--psm 11"]
    if whitelist:
        configs = [c + f" -c tessedit_char_whitelist={whitelist}" for c in configs]

    reads = []
    for variant in variants:
        for cfg in configs:
            try:
                s = pytesseract.image_to_string(variant, config=cfg).strip()
                if s:
                    reads.append(s)
            except Exception:
                pass

    # Preserve order while removing duplicates.
    out = []
    seen = set()
    for s in reads:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _crop_text(image, box, psm=7, mode="generic"):
    """
    Crop one fixed 734 cell and OCR only that cell.
    No heavy ML packages: safe for Streamlit Community Cloud.
    """
    crop = ImageOps.exif_transpose(image).crop(box)

    # Upscale hard because the sportsbook glyphs are small.
    crop = crop.resize((crop.width * 4, crop.height * 4))

    if mode == "middle":
        whitelist = "+-().0123456789½"
    elif mode == "total":
        whitelist = "oOuU()+-%.0123456789½"
    else:
        whitelist = None

    reads = _tess_read_variants(crop, whitelist=whitelist)
    return " || ".join(reads)


def _find_f5_header_y(image):
    """
    734 mobile screenshots use a stable vertical table layout.
    The 1st-5 header begins at ~51.8% of image height in the supplied iPhone
    screenshots. Using geometry here is more reliable than OCR'ing the header.
    """
    return int(image.height * 0.518)


def _parse_single_american(text):
    vals = american_numbers(text)
    if not vals:
        return None

    # ML cell should contain exactly one market price. Use the value that
    # appears most often across OCR passes.
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda v: counts[v])




def _parse_parenthesized_american(text):
    s = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
    )

    hits = [int(x.replace(" ", "")) for x in re.findall(r"\(\s*([+-]\s*\d{3,4})\s*\)", s)]
    hits = [x for x in hits if 100 <= abs(x) <= 1000]
    if hits:
        counts = {}
        for v in hits:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=lambda v: counts[v])

    vals = american_numbers(s)
    if vals:
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=lambda v: counts[v])

    return None


def _looks_like_spread_cell(text):
    """
    ML cells are just -137 / +118.
    Spread cells on 734 contain the spread plus juice in parentheses, e.g.
    -1½ (+109). We can classify the market without correctly reading ½.
    """
    s = text.replace("−", "-").replace("–", "-").replace("—", "-")
    if re.search(r"\(\s*[+-]\s*\d{3,4}\s*\)", s):
        return True
    # Fallback for OCR dropping parentheses but retaining multiple sign tokens.
    signs = re.findall(r"[+-]", s)
    return len(signs) >= 2


def _parse_total_cell(text):
    """
    Parse a 734 total cell from several Tesseract passes joined by ||.
    Vote across passes instead of trusting one OCR result.
    """
    pieces = [x.strip() for x in text.split("||") if x.strip()]
    candidates = []

    for piece in pieces:
        raw = (
            piece.replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace("＋", "+")
            .replace("½", ".5")
        )
        low = raw.lower().replace(" ", "")

        side = None
        if low.startswith("o") or low.startswith("0"):
            side = "over"
        elif low.startswith("u"):
            side = "under"

        odds = _parse_parenthesized_american(raw)

        # Look for a plausible MLB full-game total 6-14.
        m = re.search(r"[ou0]?(\d{1,2})", low)
        line = None
        if m:
            base = int(m.group(1))
            if 6 <= base <= 14:
                # Treat any explicit .5, %, or a trailing 5 after base as a half.
                half = (
                    ".5" in low
                    or "%" in low
                    or bool(re.search(rf"{base}5(?!\d)", low))
                )
                line = float(base) + (0.5 if half else 0.0)

        if side and line is not None and odds is not None:
            candidates.append((side, line, odds))

    if not candidates:
        return None, None, None

    counts = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1

    return max(counts, key=lambda c: counts[c])


def parse_734_image(image, away, home, text_hint=""):
    """
    734 fixed-cell reader.

    The supplied 734 screenshots have a stable three-column table:
      left   = team
      middle = ML or spread
      right  = total

    Instead of asking OCR to understand the whole page, this function finds
    the 1st-5 header, crops the two full-game middle cells and two total cells,
    and OCRs each cell independently.

    Row 1 is away; row 2 is home.
    """
    result = {
        "away_ml": None, "home_ml": None,
        "away_rl_side": None, "away_rl_odds": None,
        "home_rl_side": None, "home_rl_odds": None,
        "total_line": None, "over_odds": None, "under_odds": None,
    }

    w, h = image.size
    f5_y = _find_f5_header_y(image)

    # In the supplied 1206px-wide screenshots, each full-game row is ~155px.
    # Scale by image width so this also works on different iPhone resolutions.
    row_h = max(90, int(w * 0.129))

    row2_top = max(0, f5_y - row_h)
    row1_top = max(0, row2_top - row_h)

    # 734 table columns in normalized x coordinates.
    # Keep a little padding away from the vertical borders.
    mid_x1 = int(w * 0.335)
    mid_x2 = int(w * 0.665)
    tot_x1 = int(w * 0.665)
    tot_x2 = int(w * 0.995)

    pad_y = max(4, int(row_h * 0.08))
    row1 = (row1_top + pad_y, row1_top + row_h - pad_y)
    row2 = (row2_top + pad_y, row2_top + row_h - pad_y)

    away_mid = _crop_text(image, (mid_x1, row1[0], mid_x2, row1[1]), psm=7, mode="middle")
    home_mid = _crop_text(image, (mid_x1, row2[0], mid_x2, row2[1]), psm=7, mode="middle")
    away_tot = _crop_text(image, (tot_x1, row1[0], tot_x2, row1[1]), psm=7, mode="total")
    home_tot = _crop_text(image, (tot_x1, row2[0], tot_x2, row2[1]), psm=7, mode="total")

    spread_screen = _looks_like_spread_cell(away_mid) or _looks_like_spread_cell(home_mid)

    if spread_screen:
        result["away_rl_odds"] = _parse_parenthesized_american(away_mid)
        result["home_rl_odds"] = _parse_parenthesized_american(home_mid)
        # Do not trust OCR for ½. Sides get assigned after ML + spread screenshots
        # are merged, using which team is the moneyline favorite.
    else:
        result["away_ml"] = _parse_single_american(away_mid)
        result["home_ml"] = _parse_single_american(home_mid)

    a_side, a_line, a_odds = _parse_total_cell(away_tot)
    h_side, h_line, h_odds = _parse_total_cell(home_tot)

    if a_line is not None:
        result["total_line"] = a_line
    elif h_line is not None:
        result["total_line"] = h_line

    if a_side == "over" and a_odds is not None:
        result["over_odds"] = a_odds
    elif a_side == "under" and a_odds is not None:
        result["under_odds"] = a_odds

    if h_side == "under" and h_odds is not None:
        result["under_odds"] = h_odds
    elif h_side == "over" and h_odds is not None:
        result["over_odds"] = h_odds

    # Whole-image text parser is now only a fallback for totals, not ML/RL.
    fallback = parse_734_lines_text_only(text_hint, away, home)
    for k in ["total_line", "over_odds", "under_odds"]:
        if result.get(k) is None and fallback.get(k) is not None:
            result[k] = fallback[k]

    # Save cell OCR for the troubleshooting expander.
    result["_debug"] = {
        "away_middle": away_mid,
        "home_middle": home_mid,
        "away_total": away_tot,
        "home_total": home_tot,
        "f5_y": f5_y,
    }

    return result


def parse_734_lines_text_only(text, away, home):
    result = {
        "away_ml": None, "home_ml": None,
        "away_rl_side": None, "away_rl_odds": None,
        "home_rl_side": None, "home_rl_odds": None,
        "total_line": None, "over_odds": None, "under_odds": None,
    }

    t = normalize_734_text(text)
    raw_lines = [re.sub(r"\s+", " ", x).strip() for x in t.splitlines() if x.strip()]
    upper_text = t.upper()

    # Important for multi-screenshot uploads:
    # one 734 screenshot is usually MONEY LINE + TOTAL,
    # another is SPREAD + TOTAL.
    # Do not let the spread screenshot overwrite ML with +109/-139.
    header_moneyline = "MONEY LINE" in upper_text or "MONEYLINE" in upper_text
    header_spread = "SPREAD" in upper_text or "RUN LINE" in upper_text

    # Do not rely on the header alone. On iPhone screenshots Tesseract may miss
    # the MONEY LINE / SPREAD label even when the prices themselves are clear.
    # Presence of a full-game +/-1.5 token is a strong spread-screen signal.
    pre_f5_probe = re.split(r"(?i)1st\s*5\s*innings|first\s*5\s*innings", t)[0]
    has_fullgame_spread = bool(re.search(r"(?<!\d)[+-]\s*1\.5(?!\d)", pre_f5_probe))
    is_spread_screen = header_spread or has_fullgame_spread
    is_moneyline_screen = header_moneyline or not is_spread_screen

    # Build nearby OCR windows because 734 sometimes breaks team / price / total
    # into separate lines.
    windows = list(raw_lines)
    for i in range(len(raw_lines) - 1):
        windows.append(raw_lines[i] + " " + raw_lines[i + 1])
    for i in range(len(raw_lines) - 2):
        windows.append(raw_lines[i] + " " + raw_lines[i + 1] + " " + raw_lines[i + 2])

    away_keys = [away.lower(), nickname(away).lower()]
    home_keys = [home.lower(), nickname(home).lower()]

    def team_line(team_keys):
        # Prefer the full-game team rows before "1st 5 Innings".
        pre_f5 = []
        for line in raw_lines:
            if "1st 5" in line.lower() or "first 5" in line.lower():
                break
            pre_f5.append(line)

        for line in pre_f5:
            ll = line.lower()
            if any(k in ll for k in team_keys):
                return line
        return None

    away_row = team_line(away_keys)
    home_row = team_line(home_keys)

    # -------------------------
    # MONEYLINE
    # -------------------------
    if is_moneyline_screen:
        for side, row in [("away", away_row), ("home", home_row)]:
            if not row:
                continue

            # Strip the O/U portion first. That prevents total juice (-111/-119)
            # from ever being mistaken for the moneyline.
            price_zone = re.split(r"(?i)\b(?:over|under|o|u)\s*[6-9]", row, maxsplit=1)[0]
            odds = american_numbers(price_zone)
            if odds:
                result[f"{side}_ml"] = odds[-1]

        # If OCR split team name and ML price onto different lines, search a
        # short window after the team name, still above the F5 section.
        pre_f5_rows = re.split(r"(?i)1st\s*5\s*innings|first\s*5\s*innings", t)[0]
        for side, keys in [("away", away_keys), ("home", home_keys)]:
            if result[f"{side}_ml"] is not None:
                continue
            for key in keys:
                m = re.search(re.escape(key) + r".{0,90}", pre_f5_rows, re.I | re.S)
                if m:
                    zone = re.split(r"(?i)\b(?:over|under|o|u)\s*[6-9]", m.group(0), maxsplit=1)[0]
                    odds = american_numbers(zone)
                    if odds:
                        result[f"{side}_ml"] = odds[0]
                        break

    # -------------------------
    # RUN LINE / SPREAD
    # -------------------------
    if is_spread_screen:
        spread_pat = re.compile(r"([+-]1\.5)\D{0,35}?\(?\s*([+-]\d{3,4})\s*\)?", re.I)

        # First choice: parse each full-game team row.
        for side, row in [("away", away_row), ("home", home_row)]:
            if not row:
                continue
            m = spread_pat.search(row)
            if m:
                result[f"{side}_rl_side"] = m.group(1)
                result[f"{side}_rl_odds"] = int(m.group(2))

        # Second choice: parse all spread/price pairs above 1st 5. 734 displays
        # the away team first and home team second, so row order is deterministic.
        if result["away_rl_odds"] is None or result["home_rl_odds"] is None:
            pairs = []
            for m in spread_pat.finditer(pre_f5_probe):
                pair = (m.group(1), int(m.group(2)))
                if pair not in pairs:
                    pairs.append(pair)

            if len(pairs) >= 2:
                if result["away_rl_odds"] is None:
                    result["away_rl_side"], result["away_rl_odds"] = pairs[0]
                if result["home_rl_odds"] is None:
                    result["home_rl_side"], result["home_rl_odds"] = pairs[1]

        # Third choice: locate +/-1.5 and take the nearest American price.
        if result["away_rl_odds"] is None or result["home_rl_odds"] is None:
            loose = []
            for m in re.finditer(r"(?<!\d)([+-]1\.5)(?!\d)", pre_f5_probe):
                chunk = pre_f5_probe[m.start():m.start()+70]
                odds = american_numbers(chunk)
                if odds:
                    loose.append((m.group(1), odds[0]))
            if len(loose) >= 2:
                if result["away_rl_odds"] is None:
                    result["away_rl_side"], result["away_rl_odds"] = loose[0]
                if result["home_rl_odds"] is None:
                    result["home_rl_side"], result["home_rl_odds"] = loose[1]

    # -------------------------
    # FULL-GAME TOTAL
    # -------------------------
    # Only inspect text above the 1st-5 section so 4½ cannot be mistaken for
    # the game total.
    pre_f5_text = re.split(r"(?i)1st\s*5\s*innings|first\s*5\s*innings", t)[0]

    # Handles: o8.5 (-111), O 8.5 -111, u8.5 (-119)
    over_pat = re.compile(
        r"(?i)\b(?:over|ovr|o)\s*([6-9]|1[0-4])(?:\.5)?"
        r"(?P<half>\.5)?\D{0,30}?\(\s*([+-]\d{3,4})\s*\)"
    )
    under_pat = re.compile(
        r"(?i)\b(?:under|undr|u)\s*([6-9]|1[0-4])(?:\.5)?"
        r"(?P<half>\.5)?\D{0,30}?\(\s*([+-]\d{3,4})\s*\)"
    )

    # Simpler patterns are more reliable after our 734 normalization.
    simple_over = re.compile(r"(?i)\b(?:over|o)\s*([6-9]|1[0-4])(\.5)?\s*\(\s*([+-]\d{3,4})\s*\)")
    simple_under = re.compile(r"(?i)\b(?:under|u)\s*([6-9]|1[0-4])(\.5)?\s*\(\s*([+-]\d{3,4})\s*\)")

    mo = simple_over.search(pre_f5_text)
    mu = simple_under.search(pre_f5_text)

    if mo:
        total = float(mo.group(1)) + (0.5 if mo.group(2) else 0.0)
        if valid_total(total):
            result["total_line"] = total
            result["over_odds"] = int(mo.group(3))

    if mu:
        total = float(mu.group(1)) + (0.5 if mu.group(2) else 0.0)
        if valid_total(total):
            if result["total_line"] is None:
                result["total_line"] = total
            result["under_odds"] = int(mu.group(3))

    # Row-aware fallback from actual 734 layout.
    # OCR examples:
    # "Marlins -137 o8.5 (-111)"
    # "Nationals +118 u8.5 (-119)"
    # "Marlins -1.5 (+109) o8.5 (-111)"
    # "Nationals +1.5 (-139) u8.5 (-119)"
    if result["over_odds"] is None and away_row:
        m = simple_over.search(away_row)
        if m:
            total = float(m.group(1)) + (0.5 if m.group(2) else 0.0)
            result["total_line"] = total
            result["over_odds"] = int(m.group(3))

    if result["under_odds"] is None and home_row:
        m = simple_under.search(home_row)
        if m:
            total = float(m.group(1)) + (0.5 if m.group(2) else 0.0)
            if result["total_line"] is None:
                result["total_line"] = total
            result["under_odds"] = int(m.group(3))

    return result


def parse_734_lines(text, away, home):
    # Backward-compatible text parser used for OCR troubleshooting/fallback.
    return parse_734_lines_text_only(text, away, home)



def sync_parsed_to_widgets(parsed, game_pk):
    mapping = {
        "away_ml": f"away_ml_{game_pk}",
        "home_ml": f"home_ml_{game_pk}",
        "away_rl_odds": f"away_rl_odds_{game_pk}",
        "home_rl_odds": f"home_rl_odds_{game_pk}",
        "total_line": f"total_line_{game_pk}",
        "over_odds": f"over_odds_{game_pk}",
        "under_odds": f"under_odds_{game_pk}",
    }
    for src_key, widget_key in mapping.items():
        value = parsed.get(src_key)
        if value is not None:
            st.session_state[widget_key] = value

    if parsed.get("away_rl_side") in ["+1.5", "-1.5"]:
        st.session_state[f"away_rl_side_{game_pk}"] = parsed["away_rl_side"]
    if parsed.get("home_rl_side") in ["+1.5", "-1.5"]:
        st.session_state[f"home_rl_side_{game_pk}"] = parsed["home_rl_side"]


def juice_thresholds(odds):
    """
    Require more model edge as the price gets more expensive.

    Baseline:
      BET        edge >= 2.5%, EV >= 5%
      STRONG BET edge >= 4.0%, EV >= 8%

    Heavy favorite / high-juice prices must clear tougher thresholds.
    """
    odds = int(odds)

    # Positive odds and modest favorite prices.
    if odds >= -149:
        return {
            "bet_edge": 0.025, "bet_ev": 0.05,
            "strong_edge": 0.040, "strong_ev": 0.08,
            "tier": "Normal price",
        }

    # Moderate juice.
    if odds >= -179:
        return {
            "bet_edge": 0.030, "bet_ev": 0.06,
            "strong_edge": 0.050, "strong_ev": 0.09,
            "tier": "Moderate juice",
        }

    # Heavy juice.
    if odds >= -199:
        return {
            "bet_edge": 0.040, "bet_ev": 0.07,
            "strong_edge": 0.060, "strong_ev": 0.10,
            "tier": "Heavy juice",
        }

    # -200 through -249: materially tougher hurdle.
    if odds >= -249:
        return {
            "bet_edge": 0.050, "bet_ev": 0.08,
            "strong_edge": 0.070, "strong_ev": 0.12,
            "tier": "Very heavy juice",
        }

    # -250 or worse: only allow a BET with a genuinely exceptional edge.
    return {
        "bet_edge": 0.070, "bet_ev": 0.12,
        "strong_edge": 0.090, "strong_ev": 0.15,
        "tier": "Extreme juice",
    }


def bet_grade(model_prob, odds, confidence):
    imp = implied_prob(odds)
    edge = model_prob - imp
    ev = expected_value(model_prob, odds)
    t = juice_thresholds(odds)

    if confidence >= 80 and edge >= t["strong_edge"] and ev >= t["strong_ev"]:
        verdict = "STRONG BET"
    elif confidence >= 70 and edge >= t["bet_edge"] and ev >= t["bet_ev"]:
        verdict = "BET"
    elif edge > 0 and ev > 0:
        verdict = "LEAN"
    else:
        verdict = "PASS"

    return verdict, edge, ev, imp



DECISION_VERSION = "0.9.0"

def valid_american_odds(value):
    """Return a valid American price as int, else None."""
    try:
        if value is None:
            return None
        x = float(value)
        if not math.isfinite(x):
            return None
        # American odds should not be 0 or inside (-100, +100).
        if abs(x) < 100:
            return None
        return int(round(x))
    except Exception:
        return None


def sanitize_market_price(value):
    """Normalize incoming/manual/API prices and reject placeholders."""
    return valid_american_odds(value)


def median_valid(values):
    vals = [sanitize_market_price(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return int(round(statistics.median(vals)))


def best_price_for_bettor(values):
    """
    For American odds, the numerically larger price is always better for the bettor:
    +120 > +110 and -105 > -115.
    """
    vals = [sanitize_market_price(v) for v in values]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def market_source_summary(market):
    """Human-readable per-market book counts."""
    if not market:
        return ""
    parts = []
    for label, key in [("ML", "ml_books_count"), ("RL", "rl_books_count"), ("Total", "total_books_count")]:
        c = market.get(key)
        if c:
            parts.append(f"{label}: {int(c)} book{'s' if int(c) != 1 else ''}")
    return " • ".join(parts)



def no_vig_pair(odds_a, odds_b):
    """Normalize two-way implied probabilities so they sum to 1."""
    oa = sanitize_market_price(odds_a)
    ob = sanitize_market_price(odds_b)
    if oa is None or ob is None:
        return None, None
    try:
        pa = implied_prob(float(oa))
        pb = implied_prob(float(ob))
        s = pa + pb
        if s <= 0:
            return None, None
        return pa / s, pb / s
    except Exception:
        return None, None


def calibration_alpha(market_type, confidence):
    """
    Conservative market-aware shrinkage.
    MLB projection remains the signal; current consensus is the stabilizing prior.
    Lower-confidence and experimental markets get more shrinkage.
    """
    try:
        c = max(0.0, min(100.0, float(confidence)))
    except Exception:
        c = 70.0
    q = max(0.0, min(1.0, (c - 60.0) / 35.0))

    if market_type == "moneyline":
        return 0.35 + 0.25 * q      # ~35%-60% model
    if market_type == "runline":
        return 0.25 + 0.20 * q      # ~25%-45% model
    if market_type == "total":
        return 0.20 + 0.20 * q      # ~20%-40% model
    return 0.35


def calibrated_probability(raw_prob, market_prob, market_type, confidence):
    a = calibration_alpha(market_type, confidence)
    raw = max(0.001, min(0.999, float(raw_prob)))
    market = max(0.001, min(0.999, float(market_prob)))
    return max(0.001, min(0.999, market + a * (raw - market)))


def decision_thresholds(market_type, odds):
    """
    A/B thresholds for the production betting layer.
    Totals and run lines are intentionally stricter because their probability
    models are less proven than MLB moneyline probabilities.
    """
    odds = float(odds)

    if market_type == "moneyline":
        b_edge, b_ev, a_edge, a_ev = .025, .045, .045, .075
    elif market_type == "runline":
        b_edge, b_ev, a_edge, a_ev = .035, .060, .060, .100
    else:  # total
        b_edge, b_ev, a_edge, a_ev = .040, .070, .065, .110

    # Extra protection for expensive favorites.
    if odds <= -200:
        b_edge += .010
        b_ev += .010
        a_edge += .010
        a_ev += .015

    # Longshot ML tail protection.
    if market_type == "moneyline" and odds >= 300:
        b_edge += .015
        b_ev += .025
        a_edge += .020
        a_ev += .035

    return {
        "b_edge": b_edge, "b_ev": b_ev,
        "a_edge": a_edge, "a_ev": a_ev,
    }


def decision_grade(prob, odds, confidence, market_type):
    imp = implied_prob(odds)
    edge = float(prob) - imp
    ev = expected_value(float(prob), odds)
    t = decision_thresholds(market_type, odds)

    # Extreme MLB longshots are not allowed onto the official card.
    if market_type == "moneyline" and float(odds) >= 500:
        verdict = "PASS"
    elif confidence >= 78 and edge >= t["a_edge"] and ev >= t["a_ev"]:
        verdict = "STRONG BET"
    elif confidence >= 65 and edge >= t["b_edge"] and ev >= t["b_ev"]:
        verdict = "BET"
    elif edge >= .010 and ev >= .015:
        verdict = "LEAN"
    else:
        verdict = "PASS"

    # +300 to +499 can never be an A/B official play.
    if market_type == "moneyline" and float(odds) >= 300 and verdict in {"STRONG BET", "BET"}:
        verdict = "LEAN"

    return verdict, edge, ev, imp


def grade_meta(verdict):
    return {
        "STRONG BET": ("A", 4, "BEST BET"),
        "BET": ("B", 3, "BET"),
        "LEAN": ("C", 2, "LEAN"),
        "PASS": ("D", 1, "PASS"),
        "NO LINE": ("D", 0, "NO LINE"),
    }.get(str(verdict), ("D", 0, "PASS"))




def american_profit_per_unit(odds):
    odds = float(odds)
    if odds == 0 or abs(odds) < 100:
        return None
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def settle_binary_bet(won, odds, push=False):
    if push:
        return 0.0
    p = american_profit_per_unit(odds)
    if p is None:
        return None
    return p if bool(won) else -1.0


def max_drawdown_from_units(units):
    equity = peak = max_dd = 0.0
    for u in units:
        equity += float(u)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def calibration_bucket(prob):
    p = float(prob)
    if p < .50: return "<50%"
    if p < .55: return "50–55%"
    if p < .60: return "55–60%"
    if p < .65: return "60–65%"
    if p < .70: return "65–70%"
    return "70%+"


def edge_bucket(edge):
    e = float(edge)
    if e < 0: return "<0%"
    if e < .02: return "0–2%"
    if e < .04: return "2–4%"
    if e < .06: return "4–6%"
    if e < .08: return "6–8%"
    return "8%+"


def odds_bucket(odds):
    o = int(odds)
    if o <= -200: return "≤ -200"
    if o <= -150: return "-199 to -150"
    if o < 100: return "-149 to -100"
    if o < 150: return "+100 to +149"
    if o < 200: return "+150 to +199"
    if o < 300: return "+200 to +299"
    return "+300+"


def normalize_backtest_columns(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    aliases = {
        "GameDate":"Date", "game_date":"Date", "market_type":"Market_Type",
        "Market":"Bet", "Pick":"Bet", "raw_model_prob":"Raw_Model_Prob",
        "market_prob":"Market_NoVig_Prob", "calibrated_prob":"Calibrated_Prob",
        "edge":"Edge", "ev":"EV", "confidence":"Confidence",
        "verdict":"Verdict", "result":"Result", "odds":"Odds",
    }
    out = out.rename(columns={k:v for k,v in aliases.items() if k in out.columns})
    required = [
        "Date","Game","Market_Type","Bet","Odds","Result","Raw_Model_Prob",
        "Market_NoVig_Prob","Calibrated_Prob","Edge","EV","Verdict","Confidence"
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return None, missing

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Odds"] = pd.to_numeric(out["Odds"], errors="coerce")
    for c in ["Raw_Model_Prob","Market_NoVig_Prob","Calibrated_Prob","Edge","EV","Confidence"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Result"] = out["Result"].astype(str).str.upper().str.strip()
    out["Market_Type"] = out["Market_Type"].astype(str).str.upper().str.strip()
    out["Verdict"] = out["Verdict"].astype(str).str.upper().str.strip().replace({
        "BEST BET":"STRONG BET", "A":"STRONG BET", "B":"BET", "C":"LEAN", "D":"PASS"
    })

    out = out.dropna(subset=["Date","Odds","Calibrated_Prob","Edge","EV"])
    out = out[out["Result"].isin(["WIN","LOSS","PUSH"])]
    out = out[out["Odds"].apply(lambda x: valid_american_odds(x) is not None)]
    return out, []


def attach_backtest_pnl(df):
    out = df.copy()
    out["Won"] = out["Result"].eq("WIN")
    out["Push"] = out["Result"].eq("PUSH")
    out["Units"] = [
        settle_binary_bet(w, o, p)
        for w, o, p in zip(out["Won"], out["Odds"], out["Push"])
    ]
    out = out.dropna(subset=["Units"])
    out["Win"] = out["Won"].astype(int)
    out["Loss"] = out["Result"].eq("LOSS").astype(int)
    out["Season"] = out["Date"].dt.year
    out["Edge_Bucket"] = out["Edge"].apply(edge_bucket)
    out["Odds_Bucket"] = out["Odds"].apply(odds_bucket)
    return out


def summarize_bets(df):
    if df is None or df.empty:
        return {"Bets":0,"Wins":0,"Losses":0,"Pushes":0,"Hit_Rate":0.0,"Units":0.0,
                "ROI":0.0,"Avg_Odds":None,"Max_Drawdown":0.0}
    wins = int((df["Result"]=="WIN").sum())
    losses = int((df["Result"]=="LOSS").sum())
    pushes = int((df["Result"]=="PUSH").sum())
    bets = wins + losses
    units = float(df["Units"].sum())
    return {
        "Bets":bets, "Wins":wins, "Losses":losses, "Pushes":pushes,
        "Hit_Rate":wins/bets if bets else 0.0,
        "Units":units, "ROI":units/bets if bets else 0.0,
        "Avg_Odds":float(df["Odds"].mean()) if len(df) else None,
        "Max_Drawdown":max_drawdown_from_units(df.sort_values("Date")["Units"].tolist()),
    }


def grouped_backtest_summary(df, group_col):
    rows = []
    for key, grp in df.groupby(group_col, dropna=False):
        s = summarize_bets(grp)
        rows.append({
            group_col:key,
            "Bets":s["Bets"],
            "Record":f'{s["Wins"]}-{s["Losses"]}-{s["Pushes"]}',
            "Hit %":round(s["Hit_Rate"]*100,1),
            "Units":round(s["Units"],2),
            "ROI %":round(s["ROI"]*100,1),
            "Avg Odds":round(s["Avg_Odds"],1) if s["Avg_Odds"] is not None else None,
            "Max DD":round(s["Max_Drawdown"],2),
        })
    return pd.DataFrame(rows)


def daily_top_card(df, n=5):
    x = df[df["Verdict"].isin(["STRONG BET","BET"])].copy()
    if x.empty:
        return x
    grade_score = x["Verdict"].map({"STRONG BET":2,"BET":1}).fillna(0)
    x["_rank_score"] = (
        grade_score*100 + x["Calibrated_Prob"]*45 + x["Edge"]*32
        + x["EV"].clip(upper=.30)*12 + x["Confidence"]*.08
    )
    x = x.sort_values(["Date","Game","_rank_score"], ascending=[True,True,False])
    x = x.drop_duplicates(["Date","Game"], keep="first")
    x = x.sort_values(["Date","_rank_score"], ascending=[True,False])
    x = x.groupby("Date", group_keys=False).head(int(n))
    return x.drop(columns=["_rank_score"], errors="ignore")


def calibration_table(df, prob_col):
    tmp = df[~df["Push"]].copy()
    if tmp.empty:
        return pd.DataFrame()
    tmp["Probability Bucket"] = tmp[prob_col].apply(calibration_bucket)
    rows = []
    order = ["<50%","50–55%","55–60%","60–65%","65–70%","70%+"]
    for bucket in order:
        grp = tmp[tmp["Probability Bucket"] == bucket]
        if grp.empty:
            continue
        pred = float(grp[prob_col].mean())
        actual = float(grp["Won"].mean())
        rows.append({
            "Probability Bucket":bucket,
            "Bets":len(grp),
            "Avg Predicted %":round(pred*100,1),
            "Actual Win %":round(actual*100,1),
            "Calibration Error":round(abs(pred-actual)*100,1),
        })
    return pd.DataFrame(rows)



HISTORICAL_ODDS_ENDPOINT = f"{ODDS_API_BASE}/historical/sports/{ODDS_SPORT_KEY}/odds"
HISTORY_CACHE_DIR = Path(".mlb_history_cache")
HISTORY_DEFAULT_START = date(2023, 4, 1)
HISTORY_DEFAULT_END = date(2025, 10, 5)
HISTORY_SNAPSHOT_HOUR_UTC = 15

def calendar_dates(start_date, end_date):
    cur = start_date
    out = []
    while cur <= end_date:
        out.append(cur)
        cur += timedelta(days=1)
    return out

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_mlb_regular_season_dates(start_date, end_date):
    """
    Fetch actual MLB regular-season game dates from MLB's free Stats API,
    one calendar year at a time. Splitting the request prevents long
    multi-year schedule queries from returning incomplete/truncated ranges.

    This consumes ZERO Odds API credits.
    """
    if end_date < start_date:
        return [], {}, "End date is before start date."

    url = "https://statsapi.mlb.com/api/v1/schedule"
    all_dates = []
    season_counts = {}

    for year in range(start_date.year, end_date.year + 1):
        chunk_start = max(start_date, date(year, 1, 1))
        chunk_end = min(end_date, date(year, 12, 31))

        params = {
            "sportId": 1,
            "gameType": "R",
            "startDate": chunk_start.strftime("%Y-%m-%d"),
            "endDate": chunk_end.strftime("%Y-%m-%d"),
            "hydrate": "none",
        }

        try:
            resp = requests.get(url, params=params, timeout=25)
        except requests.RequestException:
            return [], season_counts, f"Could not reach the free MLB schedule service for {year}."

        if resp.status_code != 200:
            return [], season_counts, f"MLB schedule service returned HTTP {resp.status_code} for {year}."

        try:
            payload = resp.json()
        except Exception:
            return [], season_counts, f"MLB schedule response for {year} was not valid JSON."

        year_dates = []
        for day in payload.get("dates", []) or []:
            raw = day.get("date")
            games = day.get("games") or []
            if not raw or not games:
                continue
            try:
                d = date.fromisoformat(raw)
            except Exception:
                continue

            if chunk_start <= d <= chunk_end:
                year_dates.append(d)

        year_dates = sorted(set(year_dates))
        season_counts[year] = len(year_dates)
        all_dates.extend(year_dates)

    return sorted(set(all_dates)), season_counts, None

def historical_credit_estimate(game_dates, markets, regions=("us",)):
    snapshots = len(game_dates)
    per_snapshot = 10 * max(1, len(markets)) * max(1, len(regions))
    return {
        "snapshots": snapshots,
        "per_snapshot_max": per_snapshot,
        "max_credits": snapshots * per_snapshot,
    }

def phase_credit_plan(game_dates):
    n = len(game_dates)
    return {
        "moneyline": n * 10,
        "runline": n * 10,
        "total": n * 10,
        "all_three": n * 30,
    }

def _history_cache_file(snapshot_date, markets):
    market_key = "-".join(sorted(markets))
    return HISTORY_CACHE_DIR / f"{snapshot_date.isoformat()}__{market_key}.json"

def _redacted_history_error(resp):
    if resp.status_code == 401:
        return "Historical Odds authorization failed. Confirm the paid plan is active and your Streamlit secret contains a valid key."
    if resp.status_code == 422:
        return "The historical request was rejected. Check the selected date and markets."
    if resp.status_code == 429:
        return "The Odds API quota/rate limit was reached."
    return f"Historical Odds request failed with HTTP {resp.status_code}."

def fetch_historical_snapshot(snapshot_date, markets, force=False):
    cache_file = _history_cache_file(snapshot_date, markets)
    if cache_file.exists() and not force:
        try:
            return json.loads(cache_file.read_text()), {"cached": True, "remaining": None, "used": None, "last": 0}, None
        except Exception:
            pass
    try:
        api_key = st.secrets.get("ODDS_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        return None, {}, "ODDS_API_KEY is missing from Streamlit Secrets."
    HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_dt = datetime(snapshot_date.year, snapshot_date.month, snapshot_date.day, HISTORY_SNAPSHOT_HOUR_UTC, 0, 0, tzinfo=timezone.utc)
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": snapshot_dt.isoformat().replace("+00:00", "Z"),
    }
    try:
        resp = requests.get(HISTORICAL_ODDS_ENDPOINT, params=params, timeout=25)
    except requests.RequestException:
        return None, {}, "Network error while contacting The Odds API."
    meta = {
        "cached": False,
        "remaining": resp.headers.get("x-requests-remaining"),
        "used": resp.headers.get("x-requests-used"),
        "last": resp.headers.get("x-requests-last"),
    }
    if resp.status_code != 200:
        return None, meta, _redacted_history_error(resp)
    try:
        payload = resp.json()
    except Exception:
        return None, meta, "The historical odds response was not valid JSON."
    try:
        cache_file.write_text(json.dumps(payload))
    except Exception:
        pass
    return payload, meta, None

def historical_payload_games(payload):
    if isinstance(payload, dict):
        data = payload.get("data", [])
        return data if isinstance(data, list) else []
    if isinstance(payload, list):
        return payload
    return []

def _median_or_none(values):
    vals=[]
    for v in values:
        try:
            f=float(v)
            if math.isfinite(f): vals.append(f)
        except Exception:
            pass
    return float(median(vals)) if vals else None

def flatten_historical_snapshot(payload, requested_snapshot_date):
    rows=[]
    for game in historical_payload_games(payload):
        home, away = game.get("home_team"), game.get("away_team")
        if not home or not away:
            continue
        home_ml=[]; away_ml=[]; home_spread_pts=[]; home_spread_px=[]; away_spread_pts=[]; away_spread_px=[]; total_pts=[]; over_px=[]; under_px=[]
        books_h2h=set(); books_spread=set(); books_total=set()
        for book in game.get("bookmakers", []) or []:
            bkey=book.get("key") or book.get("title") or "book"
            for market in book.get("markets", []) or []:
                mkey=market.get("key"); outcomes=market.get("outcomes", []) or []
                if mkey=="h2h":
                    found=False
                    for o in outcomes:
                        price=valid_american_odds(o.get("price")); name=o.get("name")
                        if price is None: continue
                        if name==home: home_ml.append(price); found=True
                        elif name==away: away_ml.append(price); found=True
                    if found: books_h2h.add(bkey)
                elif mkey=="spreads":
                    found=False
                    for o in outcomes:
                        price=valid_american_odds(o.get("price")); name=o.get("name")
                        try: point=float(o.get("point"))
                        except Exception: point=None
                        if price is None or point is None: continue
                        if name==home: home_spread_pts.append(point); home_spread_px.append(price); found=True
                        elif name==away: away_spread_pts.append(point); away_spread_px.append(price); found=True
                    if found: books_spread.add(bkey)
                elif mkey=="totals":
                    found=False
                    for o in outcomes:
                        price=valid_american_odds(o.get("price")); name=str(o.get("name","")).lower()
                        try: point=float(o.get("point"))
                        except Exception: point=None
                        if price is None or point is None: continue
                        total_pts.append(point)
                        if name=="over": over_px.append(price); found=True
                        elif name=="under": under_px.append(price); found=True
                    if found: books_total.add(bkey)
        rows.append({
            "Snapshot_Date": requested_snapshot_date.isoformat(),
            "Snapshot_Hour_UTC": HISTORY_SNAPSHOT_HOUR_UTC,
            "Event_ID": game.get("id"),
            "Commence_Time": game.get("commence_time"),
            "Away_Team": away,
            "Home_Team": home,
            "Away_ML": _median_or_none(away_ml),
            "Home_ML": _median_or_none(home_ml),
            "Away_RL": _median_or_none(away_spread_pts),
            "Away_RL_Odds": _median_or_none(away_spread_px),
            "Home_RL": _median_or_none(home_spread_pts),
            "Home_RL_Odds": _median_or_none(home_spread_px),
            "Total": _median_or_none(total_pts),
            "Over_Odds": _median_or_none(over_px),
            "Under_Odds": _median_or_none(under_px),
            "ML_Books": len(books_h2h),
            "RL_Books": len(books_spread),
            "Total_Books": len(books_total),
        })
    return rows

def cached_history_manifest(markets):
    HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix="__"+"-".join(sorted(markets))+".json"
    out=set()
    for p in HISTORY_CACHE_DIR.glob(f"*{suffix}"):
        try: out.add(date.fromisoformat(p.name.split("__",1)[0]))
        except Exception: pass
    return out

def build_cache_zip():
    HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(HISTORY_CACHE_DIR.glob("*.json")):
            zf.writestr(p.name,p.read_bytes())
    return buf.getvalue()

def restore_cache_zip(uploaded_file):
    HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    restored=0
    with zipfile.ZipFile(uploaded_file) as zf:
        for member in zf.infolist():
            name=Path(member.filename).name
            if not name.endswith(".json") or "__" not in name: continue
            (HISTORY_CACHE_DIR/name).write_bytes(zf.read(member)); restored+=1
    return restored


MLB_RESULTS_CACHE_DIR = Path(".mlb_results_cache")


def american_to_implied_prob(odds):
    o = valid_american_odds(odds)
    if o is None:
        return None
    o = float(o)
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return 100.0 / (o + 100.0)


def two_way_no_vig_prob(odds_a, odds_b):
    pa = american_to_implied_prob(odds_a)
    pb = american_to_implied_prob(odds_b)
    if pa is None or pb is None:
        return None, None
    denom = pa + pb
    if denom <= 0:
        return None, None
    return pa / denom, pb / denom


def clean_historical_moneyline_df(df, max_hours_to_first_pitch=18.0):
    """
    Clean the paid historical market export into one usable pregame ML row per event.
    Keeps games 0–18 hours from first pitch and valid two-way American prices.
    """
    out = df.copy()

    required = [
        "Snapshot_Date", "Event_ID", "Commence_Time",
        "Away_Team", "Home_Team", "Away_ML", "Home_ML"
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return None, {"missing": missing}

    out["Snapshot_Date"] = pd.to_datetime(out["Snapshot_Date"], errors="coerce", utc=True)
    out["Commence_Time"] = pd.to_datetime(out["Commence_Time"], errors="coerce", utc=True)
    out["Away_ML"] = pd.to_numeric(out["Away_ML"], errors="coerce")
    out["Home_ML"] = pd.to_numeric(out["Home_ML"], errors="coerce")

    # Snapshot hour is 15:00 UTC by design in v0.10.3.
    out["Snapshot_Timestamp"] = (
        out["Snapshot_Date"].dt.normalize()
        + pd.to_timedelta(pd.to_numeric(out.get("Snapshot_Hour_UTC", 15), errors="coerce").fillna(15), unit="h")
    )
    out["Hours_To_First_Pitch"] = (
        (out["Commence_Time"] - out["Snapshot_Timestamp"]).dt.total_seconds() / 3600.0
    )

    out["Away_ML_Valid"] = out["Away_ML"].apply(lambda x: valid_american_odds(x) is not None)
    out["Home_ML_Valid"] = out["Home_ML"].apply(lambda x: valid_american_odds(x) is not None)

    cleaned = out[
        out["Away_ML_Valid"]
        & out["Home_ML_Valid"]
        & out["Hours_To_First_Pitch"].between(0, float(max_hours_to_first_pitch), inclusive="both")
    ].copy()

    # One row per historical event.
    cleaned = (
        cleaned.sort_values(["Commence_Time", "Snapshot_Timestamp"])
        .drop_duplicates(["Event_ID"], keep="last")
    )

    no_vig = cleaned.apply(
        lambda r: two_way_no_vig_prob(r["Away_ML"], r["Home_ML"]),
        axis=1
    )
    cleaned["Away_Market_Prob"] = [x[0] for x in no_vig]
    cleaned["Home_Market_Prob"] = [x[1] for x in no_vig]

    cleaned["Season"] = cleaned["Commence_Time"].dt.year
    cleaned["Game_Date_UTC"] = cleaned["Commence_Time"].dt.date.astype(str)

    stats = {
        "raw_rows": len(out),
        "clean_rows": len(cleaned),
        "invalid_away_ml": int((~out["Away_ML_Valid"]).sum()),
        "invalid_home_ml": int((~out["Home_ML_Valid"]).sum()),
        "outside_window": int((~out["Hours_To_First_Pitch"].between(0, float(max_hours_to_first_pitch), inclusive="both")).sum()),
        "missing": [],
    }
    return cleaned, stats


def _mlb_results_cache_file(season):
    MLB_RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MLB_RESULTS_CACHE_DIR / f"mlb_results_{int(season)}.json"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_mlb_results_season(season):
    """
    Pull one season of final MLB regular-season results from MLB's free Stats API.
    Zero Odds API credits.
    """
    cache_file = _mlb_results_cache_file(season)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text()), None, True
        except Exception:
            pass

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "gameType": "R",
        "season": int(season),
        "startDate": f"{int(season)}-03-20",
        "endDate": f"{int(season)}-10-10",
        "hydrate": "linescore",
    }

    try:
        resp = requests.get(url, params=params, timeout=35)
    except requests.RequestException:
        return None, f"Could not reach MLB's free results service for {season}.", False

    if resp.status_code != 200:
        return None, f"MLB results service returned HTTP {resp.status_code} for {season}.", False

    try:
        payload = resp.json()
    except Exception:
        return None, f"MLB results response for {season} was not valid JSON.", False

    try:
        cache_file.write_text(json.dumps(payload))
    except Exception:
        pass

    return payload, None, False


def flatten_mlb_results(payload):
    rows = []
    if not isinstance(payload, dict):
        return rows

    for day in payload.get("dates", []) or []:
        for game in day.get("games", []) or []:
            status = ((game.get("status") or {}).get("abstractGameState") or "").lower()
            detailed = ((game.get("status") or {}).get("detailedState") or "").lower()
            if status != "final" and "final" not in detailed and "completed" not in detailed:
                continue

            teams = game.get("teams") or {}
            away_obj = teams.get("away") or {}
            home_obj = teams.get("home") or {}

            away_team = ((away_obj.get("team") or {}).get("name"))
            home_team = ((home_obj.get("team") or {}).get("name"))
            away_score = away_obj.get("score")
            home_score = home_obj.get("score")

            try:
                away_score = int(away_score)
                home_score = int(home_score)
            except Exception:
                continue

            rows.append({
                "MLB_GamePk": game.get("gamePk"),
                "Result_Commence_Time": game.get("gameDate"),
                "Result_Game_Date": day.get("date"),
                "Away_Team_Result": away_team,
                "Home_Team_Result": home_team,
                "Away_Score": away_score,
                "Home_Score": home_score,
                "Winner": away_team if away_score > home_score else home_team,
                "Away_Win": int(away_score > home_score),
                "Home_Win": int(home_score > away_score),
                "Final_Total_Runs": away_score + home_score,
            })
    return rows


def normalize_team_name_for_match(name):
    if name is None:
        return ""
    s = str(name).lower().strip()
    replacements = {
        "d-backs": "diamondbacks",
        "arizona d-backs": "arizona diamondbacks",
        "athletics": "oakland athletics",
        "a's": "oakland athletics",
        "la angels": "los angeles angels",
    }
    s = replacements.get(s, s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_moneyline_master(clean_market_df, results_df):
    """
    Match cleaned historical market rows to free MLB final results using
    normalized away/home team names plus commence date proximity.
    """
    market = clean_market_df.copy()
    results = results_df.copy()

    market["_away_key"] = market["Away_Team"].map(normalize_team_name_for_match)
    market["_home_key"] = market["Home_Team"].map(normalize_team_name_for_match)
    market["_game_day"] = market["Commence_Time"].dt.date.astype(str)

    results["Result_Commence_Time"] = pd.to_datetime(
        results["Result_Commence_Time"], errors="coerce", utc=True
    )
    results["_away_key"] = results["Away_Team_Result"].map(normalize_team_name_for_match)
    results["_home_key"] = results["Home_Team_Result"].map(normalize_team_name_for_match)
    results["_game_day"] = results["Result_Commence_Time"].dt.date.astype(str)

    # Exact team/date match handles virtually all games and doubleheaders because
    # commence times differ. Merge candidates, then pick nearest time.
    cand = market.merge(
        results,
        on=["_away_key", "_home_key", "_game_day"],
        how="left",
        suffixes=("", "_res")
    )

    cand["_time_diff_min"] = (
        (cand["Commence_Time"] - cand["Result_Commence_Time"]).abs().dt.total_seconds() / 60.0
    )

    cand = (
        cand.sort_values(["Event_ID", "_time_diff_min"], na_position="last")
        .drop_duplicates(["Event_ID"], keep="first")
    )

    cand["Result_Matched"] = cand["MLB_GamePk"].notna()
    cand["Market_Favorite"] = np.where(
        cand["Home_Market_Prob"] >= cand["Away_Market_Prob"],
        cand["Home_Team"], cand["Away_Team"]
    )
    cand["Favorite_Prob"] = cand[["Home_Market_Prob", "Away_Market_Prob"]].max(axis=1)
    cand["Favorite_Won"] = np.where(
        cand["Result_Matched"],
        (cand["Winner"] == cand["Market_Favorite"]).astype(int),
        np.nan
    )

    keep = [
        "Season", "Game_Date_UTC", "Event_ID", "MLB_GamePk", "Commence_Time",
        "Snapshot_Timestamp", "Hours_To_First_Pitch",
        "Away_Team", "Home_Team", "Away_ML", "Home_ML",
        "Away_Market_Prob", "Home_Market_Prob",
        "ML_Books", "Away_Score", "Home_Score", "Winner",
        "Away_Win", "Home_Win", "Final_Total_Runs",
        "Market_Favorite", "Favorite_Prob", "Favorite_Won",
        "Result_Matched"
    ]
    keep = [c for c in keep if c in cand.columns]
    return cand[keep].copy()


def market_calibration_summary(master_df):
    x = master_df[master_df["Result_Matched"]].copy()
    if x.empty:
        return pd.DataFrame()

    rows = []
    for side in ["Away", "Home"]:
        pcol = f"{side}_Market_Prob"
        wcol = f"{side}_Win"
        tmp = x[[pcol, wcol]].dropna().copy()
        tmp["Bucket"] = pd.cut(
            tmp[pcol],
            bins=[0,.40,.45,.50,.55,.60,.65,.70,1.0],
            labels=["<40%","40–45%","45–50%","50–55%","55–60%","60–65%","65–70%","70%+"],
            include_lowest=True,
            right=False
        )
        for bucket, grp in tmp.groupby("Bucket", observed=True):
            if grp.empty:
                continue
            rows.append({
                "Probability Bucket": str(bucket),
                "Team-Sides": len(grp),
                "Avg Market %": round(grp[pcol].mean()*100,1),
                "Actual Win %": round(grp[wcol].mean()*100,1),
                "Abs Error": round(abs(grp[pcol].mean()-grp[wcol].mean())*100,1),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby("Probability Bucket", as_index=False)
        .agg({
            "Team-Sides":"sum",
            "Avg Market %":"mean",
            "Actual Win %":"mean",
            "Abs Error":"mean"
        })
        .round({"Avg Market %":1,"Actual Win %":1,"Abs Error":1})
    )


def moneyline_market_brier(master_df):
    x = master_df[master_df["Result_Matched"]].dropna(
        subset=["Home_Market_Prob", "Home_Win"]
    )
    if x.empty:
        return None
    return float(((x["Home_Market_Prob"] - x["Home_Win"].astype(float))**2).mean())

PIT_FEATURES = [
    "Home_Season_WinPct",
    "Away_Season_WinPct",
    "Home_Season_RunDiffPG",
    "Away_Season_RunDiffPG",
    "Home_Last10_WinPct",
    "Away_Last10_WinPct",
    "Home_Last10_RunDiffPG",
    "Away_Last10_RunDiffPG",
    "Home_Rest_Days",
    "Away_Rest_Days",
]


def _safe_mean(vals, default=np.nan):
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else default


def build_point_in_time_features(master_df, min_prior_games=12):
    """
    Construct historical team-state features using ONLY games completed before
    the current game's first pitch. No current-game result is used.

    This is intentionally a separate PIT validation model, not a retroactive
    claim that the live production engine had these exact historical inputs.
    """
    df = master_df.copy()
    df["Commence_Time"] = pd.to_datetime(df["Commence_Time"], errors="coerce", utc=True)
    df = df[df["Result_Matched"].astype(bool)].copy()
    df = df.dropna(subset=[
        "Commence_Time","Away_Team","Home_Team","Away_Score","Home_Score",
        "Away_Market_Prob","Home_Market_Prob","Away_ML","Home_ML"
    ])
    df = df.sort_values(["Commence_Time","Event_ID"]).reset_index(drop=True)

    team_hist = {}
    rows = []

    for _, r in df.iterrows():
        away = r["Away_Team"]
        home = r["Home_Team"]
        game_time = r["Commence_Time"]

        ah = team_hist.get(away, [])
        hh = team_hist.get(home, [])

        def summarize(hist):
            if not hist:
                return {
                    "n":0,"winpct":np.nan,"rdpg":np.nan,
                    "l10_winpct":np.nan,"l10_rdpg":np.nan,
                    "rest":np.nan
                }
            wins = [x["win"] for x in hist]
            rds = [x["rd"] for x in hist]
            last10 = hist[-10:]
            last_time = hist[-1]["time"]
            rest = max(0.0, (game_time - last_time).total_seconds()/86400.0)
            return {
                "n":len(hist),
                "winpct":_safe_mean(wins),
                "rdpg":_safe_mean(rds),
                "l10_winpct":_safe_mean([x["win"] for x in last10]),
                "l10_rdpg":_safe_mean([x["rd"] for x in last10]),
                "rest":rest,
            }

        a = summarize(ah)
        h = summarize(hh)

        rows.append({
            "Event_ID": r["Event_ID"],
            "Season": int(r["Season"]),
            "Commence_Time": game_time,
            "Away_Team": away,
            "Home_Team": home,
            "Away_ML": float(r["Away_ML"]),
            "Home_ML": float(r["Home_ML"]),
            "Away_Market_Prob": float(r["Away_Market_Prob"]),
            "Home_Market_Prob": float(r["Home_Market_Prob"]),
            "Home_Win": int(r["Home_Win"]),
            "Away_Win": int(r["Away_Win"]),
            "Home_Prior_Games": h["n"],
            "Away_Prior_Games": a["n"],
            "Home_Season_WinPct": h["winpct"],
            "Away_Season_WinPct": a["winpct"],
            "Home_Season_RunDiffPG": h["rdpg"],
            "Away_Season_RunDiffPG": a["rdpg"],
            "Home_Last10_WinPct": h["l10_winpct"],
            "Away_Last10_WinPct": a["l10_winpct"],
            "Home_Last10_RunDiffPG": h["l10_rdpg"],
            "Away_Last10_RunDiffPG": a["l10_rdpg"],
            "Home_Rest_Days": h["rest"],
            "Away_Rest_Days": a["rest"],
        })

        # Only after feature capture do we append the current result.
        away_score = float(r["Away_Score"])
        home_score = float(r["Home_Score"])
        team_hist.setdefault(away, []).append({
            "time": game_time,
            "win": int(away_score > home_score),
            "rd": away_score - home_score,
        })
        team_hist.setdefault(home, []).append({
            "time": game_time,
            "win": int(home_score > away_score),
            "rd": home_score - away_score,
        })

    out = pd.DataFrame(rows)
    out["PIT_Eligible"] = (
        (out["Home_Prior_Games"] >= int(min_prior_games))
        & (out["Away_Prior_Games"] >= int(min_prior_games))
    )
    return out


def _standardize_train_apply(train_x, apply_x):
    mu = np.nanmean(train_x, axis=0)
    sd = np.nanstd(train_x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    train_fill = np.where(np.isnan(train_x), mu, train_x)
    apply_fill = np.where(np.isnan(apply_x), mu, apply_x)
    return (train_fill-mu)/sd, (apply_fill-mu)/sd, mu, sd


def _sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0/(1.0+np.exp(-z))


def fit_ridge_logit(x, y, ridge=1.0):
    """
    Small dependency-free ridge logistic regression using scipy.optimize.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x1 = np.column_stack([np.ones(len(x)), x])

    def loss(beta):
        p = _sigmoid(x1 @ beta)
        eps = 1e-12
        nll = -np.sum(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
        penalty = float(ridge) * np.sum(beta[1:]**2)
        return nll + penalty

    res = minimize(loss, np.zeros(x1.shape[1]), method="L-BFGS-B")
    return res.x


def predict_logit(beta, x):
    x = np.asarray(x, dtype=float)
    x1 = np.column_stack([np.ones(len(x)), x])
    return _sigmoid(x1 @ beta)


def pit_feature_matrix(df):
    x = pd.DataFrame(index=df.index)
    # Differences encode home-minus-away team state.
    x["Season_WinPct_Diff"] = df["Home_Season_WinPct"] - df["Away_Season_WinPct"]
    x["Season_RunDiff_Diff"] = df["Home_Season_RunDiffPG"] - df["Away_Season_RunDiffPG"]
    x["Last10_WinPct_Diff"] = df["Home_Last10_WinPct"] - df["Away_Last10_WinPct"]
    x["Last10_RunDiff_Diff"] = df["Home_Last10_RunDiffPG"] - df["Away_Last10_RunDiffPG"]
    x["Rest_Diff"] = df["Home_Rest_Days"] - df["Away_Rest_Days"]
    return x


def brier_score_binary(prob, y):
    p = np.asarray(pd.to_numeric(pd.Series(prob), errors="coerce"), dtype=float)
    yy = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=float)
    mask = np.isfinite(p) & np.isfinite(yy)
    if not np.any(mask):
        return np.nan
    return float(np.mean((p[mask] - yy[mask])**2))


def log_loss_binary(prob, y):
    p = np.asarray(pd.to_numeric(pd.Series(prob), errors="coerce"), dtype=float)
    yy = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=float)
    mask = np.isfinite(p) & np.isfinite(yy)
    if not np.any(mask):
        return np.nan
    pp = np.clip(p[mask], 1e-6, 1-1e-6)
    yv = yy[mask]
    return float(-np.mean(yv*np.log(pp) + (1-yv)*np.log(1-pp)))


def blend_prob(model_prob, market_prob, model_weight):
    w = float(model_weight)
    mp = np.asarray(model_prob, dtype=float)
    mkp = np.asarray(market_prob, dtype=float)
    return np.clip(w*mp + (1-w)*mkp, .01, .99)


def american_unit_profit(odds, won):
    o = valid_american_odds(odds)
    if o is None:
        return np.nan
    o = float(o)
    if bool(won):
        return o/100.0 if o > 0 else 100.0/abs(o)
    return -1.0


def simulate_ml_bets(df, prob_col, min_edge=.025, min_ev=.045, max_dog=299):
    """
    Bet whichever side has positive model-vs-market edge and passes thresholds.
    One bet max per game.
    """
    rows = []
    for _, r in df.iterrows():
        hp = float(r[prob_col])
        ap = 1.0 - hp
        hm = float(r["Home_Market_Prob"])
        am = float(r["Away_Market_Prob"])

        candidates = [
            ("HOME", hp, hm, r["Home_ML"], int(r["Home_Win"])),
            ("AWAY", ap, am, r["Away_ML"], int(r["Away_Win"])),
        ]

        best = None
        for side, p, mp, odds, won in candidates:
            edge = p-mp
            try:
                ev = expected_value(p, odds)
            except Exception:
                continue
            o = float(odds)
            if o >= 300 or edge < min_edge or ev < min_ev:
                continue
            rank = edge + 0.50*ev
            if best is None or rank > best["rank"]:
                best = {
                    "Side":side,"Prob":p,"MarketProb":mp,"Odds":o,
                    "Edge":edge,"EV":ev,"Won":won,"rank":rank
                }

        if best is not None:
            rows.append({
                "Event_ID":r["Event_ID"],
                "Season":r["Season"],
                "Commence_Time":r["Commence_Time"],
                "Away_Team":r["Away_Team"],
                "Home_Team":r["Home_Team"],
                **best,
                "Units":american_unit_profit(best["Odds"], best["Won"])
            })

    return pd.DataFrame(rows)


def summarize_sim_bets(bets):
    if bets is None or bets.empty:
        return {"Bets":0,"Wins":0,"Losses":0,"Hit":0.0,"Units":0.0,"ROI":0.0}
    wins = int(bets["Won"].sum())
    losses = int(len(bets)-wins)
    units = float(bets["Units"].sum())
    return {
        "Bets":len(bets),
        "Wins":wins,
        "Losses":losses,
        "Hit":wins/len(bets) if len(bets) else 0.0,
        "Units":units,
        "ROI":units/len(bets) if len(bets) else 0.0,
    }


def run_point_in_time_backtest(master_df, min_prior_games=12):
    """
    Development:
      Train on 2023 PIT-eligible games
      Validate candidate market-blend weights on 2024

    Holdout:
      Refit on 2023+2024 PIT-eligible games
      Freeze selected blend weight
      Evaluate 2025 exactly once

    The raw PIT model excludes current-game sportsbook price from its features.
    """
    pit = build_point_in_time_features(master_df, min_prior_games=min_prior_games)
    pit = pit[pit["PIT_Eligible"]].copy()

    fmat = pit_feature_matrix(pit)
    pit = pd.concat([pit.reset_index(drop=True), fmat.reset_index(drop=True)], axis=1)
    fcols = list(fmat.columns)

    train23 = pit[pit["Season"] == 2023].copy()
    val24 = pit[pit["Season"] == 2024].copy()
    hold25 = pit[pit["Season"] == 2025].copy()

    if len(train23) < 500 or len(val24) < 500 or len(hold25) < 500:
        raise ValueError("Not enough PIT-eligible games in one or more seasons.")

    X23 = train23[fcols].to_numpy(float)
    X24 = val24[fcols].to_numpy(float)
    X25 = hold25[fcols].to_numpy(float)
    y23 = train23["Home_Win"].to_numpy(float)
    y24 = val24["Home_Win"].to_numpy(float)
    y25 = hold25["Home_Win"].to_numpy(float)

    X23z, X24z, mu23, sd23 = _standardize_train_apply(X23, X24)
    beta23 = fit_ridge_logit(X23z, y23, ridge=2.0)
    raw24 = predict_logit(beta23, X24z)
    val24["PIT_Raw_Prob"] = raw24

    # Choose blend weight using 2024 Brier only; 2025 remains untouched.
    weights = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    blend_rows = []
    for w in weights:
        p = blend_prob(raw24, val24["Home_Market_Prob"].to_numpy(float), w)
        blend_rows.append({
            "Model_Weight":w,
            "Brier_2024":brier_score_binary(p, y24),
            "LogLoss_2024":log_loss_binary(p, y24)
        })
    blend_table = pd.DataFrame(blend_rows).sort_values(["Brier_2024","LogLoss_2024"])
    best_weight = float(blend_table.iloc[0]["Model_Weight"])

    # Refit on all 2023+2024, then score untouched 2025.
    dev = pit[pit["Season"].isin([2023,2024])].copy()
    Xdev = dev[fcols].to_numpy(float)
    ydev = dev["Home_Win"].to_numpy(float)
    Xdevz, X25z, mu, sd = _standardize_train_apply(Xdev, X25)
    beta_dev = fit_ridge_logit(Xdevz, ydev, ridge=2.0)
    raw25 = predict_logit(beta_dev, X25z)

    # Also score dev for audit with same final fit (descriptive only, not OOS).
    Xdevz2 = np.where(np.isnan(Xdev), mu, Xdev)
    Xdevz2 = (Xdevz2-mu)/sd
    rawdev = predict_logit(beta_dev, Xdevz2)

    dev["PIT_Raw_Prob"] = rawdev
    hold25["PIT_Raw_Prob"] = raw25
    dev["PIT_Calibrated_Prob"] = blend_prob(
        dev["PIT_Raw_Prob"], dev["Home_Market_Prob"], best_weight
    )
    hold25["PIT_Calibrated_Prob"] = blend_prob(
        hold25["PIT_Raw_Prob"], hold25["Home_Market_Prob"], best_weight
    )

    metrics = []
    for label, frame in [("2024 validation", val24), ("2025 holdout", hold25)]:
        if label.startswith("2024"):
            raw = frame["PIT_Raw_Prob"]
            cal = blend_prob(raw, frame["Home_Market_Prob"], best_weight)
        else:
            raw = frame["PIT_Raw_Prob"]
            cal = frame["PIT_Calibrated_Prob"]

        metrics.append({
            "Sample":label,
            "Games":len(frame),
            "Market Brier":brier_score_binary(frame["Home_Market_Prob"], frame["Home_Win"]),
            "Raw PIT Brier":brier_score_binary(raw, frame["Home_Win"]),
            "Calibrated Brier":brier_score_binary(cal, frame["Home_Win"]),
            "Market Log Loss":log_loss_binary(frame["Home_Market_Prob"], frame["Home_Win"]),
            "Raw PIT Log Loss":log_loss_binary(raw, frame["Home_Win"]),
            "Calibrated Log Loss":log_loss_binary(cal, frame["Home_Win"]),
        })

    hold_bets_raw = simulate_ml_bets(hold25, "PIT_Raw_Prob")
    hold_bets_cal = simulate_ml_bets(hold25, "PIT_Calibrated_Prob")

    return {
        "pit":pit,
        "dev":dev,
        "val24":val24,
        "hold25":hold25,
        "blend_table":blend_table,
        "best_weight":best_weight,
        "metrics":pd.DataFrame(metrics),
        "hold_bets_raw":hold_bets_raw,
        "hold_bets_cal":hold_bets_cal,
        "hold_raw_summary":summarize_sim_bets(hold_bets_raw),
        "hold_cal_summary":summarize_sim_bets(hold_bets_cal),
        "feature_cols":fcols,
    }

PITCHER_CACHE_DIR = Path(".mlb_pitcher_cache")


def _pitcher_schedule_cache_file(season):
    PITCHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PITCHER_CACHE_DIR / f"mlb_schedule_pitchers_{int(season)}.json"


def _pitcher_gamelog_cache_file(season, pitcher_id):
    PITCHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PITCHER_CACHE_DIR / f"pitcher_{int(pitcher_id)}_{int(season)}_gamelog.json"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_season_schedule_with_pitchers(season):
    """
    Free MLB Stats API. Pull final regular-season schedule with probable/starter IDs.
    Zero Odds API credits.
    """
    cache_file = _pitcher_schedule_cache_file(season)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text()), None, True
        except Exception:
            pass

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "gameType": "R",
        "season": int(season),
        "startDate": f"{int(season)}-03-20",
        "endDate": f"{int(season)}-10-10",
        "hydrate": "probablePitcher",
    }
    try:
        resp = requests.get(url, params=params, timeout=40)
    except requests.RequestException:
        return None, f"Could not reach MLB schedule service for {season}.", False

    if resp.status_code != 200:
        return None, f"MLB schedule service returned HTTP {resp.status_code} for {season}.", False

    try:
        payload = resp.json()
    except Exception:
        return None, f"MLB schedule response for {season} was not valid JSON.", False

    try:
        cache_file.write_text(json.dumps(payload))
    except Exception:
        pass

    return payload, None, False


def flatten_schedule_pitchers(payload):
    rows = []
    if not isinstance(payload, dict):
        return rows

    for day in payload.get("dates", []) or []:
        for game in day.get("games", []) or []:
            teams = game.get("teams") or {}
            away = teams.get("away") or {}
            home = teams.get("home") or {}
            away_team = ((away.get("team") or {}).get("name"))
            home_team = ((home.get("team") or {}).get("name"))
            away_pp = away.get("probablePitcher") or {}
            home_pp = home.get("probablePitcher") or {}

            rows.append({
                "MLB_GamePk": game.get("gamePk"),
                "Pitcher_GameDate": game.get("gameDate"),
                "Away_Team_Pitcher": away_team,
                "Home_Team_Pitcher": home_team,
                "Away_Team_ID": ((away.get("team") or {}).get("id")),
                "Home_Team_ID": ((home.get("team") or {}).get("id")),
                "Away_Starter_ID": away_pp.get("id"),
                "Away_Starter_Name": away_pp.get("fullName"),
                "Home_Starter_ID": home_pp.get("id"),
                "Home_Starter_Name": home_pp.get("fullName"),
            })
    return rows


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pitcher_season_gamelog(season, pitcher_id):
    """
    One free MLB Stats API call per pitcher-season, cached on disk.
    Game logs are later filtered strictly to appearances BEFORE the target game.
    """
    cache_file = _pitcher_gamelog_cache_file(season, pitcher_id)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text()), None, True
        except Exception:
            pass

    url = f"https://statsapi.mlb.com/api/v1/people/{int(pitcher_id)}/stats"
    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": int(season),
        "gameType": "R",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
    except requests.RequestException:
        return None, f"Could not fetch pitcher {pitcher_id} game log.", False

    if resp.status_code != 200:
        return None, f"Pitcher {pitcher_id} game log returned HTTP {resp.status_code}.", False

    try:
        payload = resp.json()
    except Exception:
        return None, f"Pitcher {pitcher_id} game log was not valid JSON.", False

    try:
        cache_file.write_text(json.dumps(payload))
    except Exception:
        pass

    return payload, None, False


def parse_pitcher_gamelog(payload):
    """
    Parse MLB gameLog stats into dated appearances.
    """
    rows = []
    for block in (payload or {}).get("stats", []) or []:
        for split in block.get("splits", []) or []:
            stat = split.get("stat") or {}
            game = split.get("game") or {}
            raw_date = split.get("date")
            if not raw_date:
                continue
            try:
                d = pd.to_datetime(raw_date, utc=True)
            except Exception:
                continue

            def num(key, default=np.nan):
                try:
                    return float(stat.get(key))
                except Exception:
                    return default

            innings = stat.get("inningsPitched")
            # MLB encodes partial innings as baseball notation: 5.1 = 5 1/3, 5.2 = 5 2/3.
            # Treating this as a normal decimal biases every rate stat, so parse outs explicitly.
            try:
                txt = str(innings).strip()
                if "." in txt:
                    whole, outs = txt.split(".", 1)
                    outs_i = int(outs)
                    ip = float(int(whole)) + (outs_i / 3.0 if outs_i in (0, 1, 2) else float("nan"))
                else:
                    ip = float(txt)
            except Exception:
                ip = np.nan

            rows.append({
                "date": d,
                "gamePk": game.get("gamePk"),
                "gamesStarted": num("gamesStarted", 0.0),
                "inningsPitched": ip,
                "era": num("era"),
                "whip": num("whip"),
                "strikeOuts": num("strikeOuts", 0.0),
                "baseOnBalls": num("baseOnBalls", 0.0),
                "hits": num("hits", 0.0),
                "homeRuns": num("homeRuns", 0.0),
                "earnedRuns": num("earnedRuns", 0.0),
                "battersFaced": num("battersFaced", 0.0),
                "numberOfPitches": num("numberOfPitches", np.nan),
                "hitBatsmen": num("hitBatsmen", 0.0),
            })
    return pd.DataFrame(rows)


def summarize_pitcher_before_game(gamelog_df, game_time, min_starts=3):
    """Point-in-time starter profile using only starts before game_time.

    v0.13 keeps the original v0.12 features for apples-to-apples benchmarking and
    adds a more baseball-specific profile: FIP-style components, exponential
    recency weighting, shrinkage toward league priors, workload, pitch-count and
    rest features. Nothing from the target game is used.
    """
    if gamelog_df is None or gamelog_df.empty:
        return None

    g = gamelog_df.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce", utc=True)
    g = g[g["date"] < game_time].copy()
    if "gamesStarted" in g.columns:
        g = g[g["gamesStarted"] >= 1].copy()
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < int(min_starts):
        return None

    def col(name, default=0.0):
        if name not in g.columns:
            return pd.Series(default, index=g.index, dtype=float)
        return pd.to_numeric(g[name], errors="coerce").fillna(default).astype(float)

    ip_s = col("inningsPitched")
    er_s = col("earnedRuns")
    so_s = col("strikeOuts")
    bb_s = col("baseOnBalls")
    hr_s = col("homeRuns")
    h_s = col("hits")
    pitches_s = pd.to_numeric(g.get("numberOfPitches", pd.Series(np.nan, index=g.index)), errors="coerce")

    ip = float(ip_s.sum()); er = float(er_s.sum()); so = float(so_s.sum())
    bb = float(bb_s.sum()); hr = float(hr_s.sum()); hits = float(h_s.sum())
    if ip <= 0:
        return None

    last5 = g.tail(5).copy()
    l5_ip = float(pd.to_numeric(last5["inningsPitched"], errors="coerce").fillna(0).sum())
    l5_er = float(pd.to_numeric(last5["earnedRuns"], errors="coerce").fillna(0).sum())
    l5_so = float(pd.to_numeric(last5["strikeOuts"], errors="coerce").fillna(0).sum())
    l5_bb = float(pd.to_numeric(last5["baseOnBalls"], errors="coerce").fillna(0).sum())
    if l5_ip <= 0:
        return None

    # Original v0.12 features.
    season_era = 9.0 * er / ip
    season_k9 = 9.0 * so / ip
    season_bb9 = 9.0 * bb / ip
    season_hr9 = 9.0 * hr / ip
    season_whip = (hits + bb) / ip
    l5_era = 9.0 * l5_er / l5_ip
    l5_k9 = 9.0 * l5_so / l5_ip
    l5_bb9 = 9.0 * l5_bb / l5_ip

    # FIP-style estimator. The constant is fixed and therefore cannot leak target-game info.
    fip_const = 3.20
    season_fip = (13.0 * hr + 3.0 * bb - 2.0 * so) / ip + fip_const
    season_kbb9 = 9.0 * (so - bb) / ip

    # Exponential recency weighting by START, half-life = 3 starts.
    n = len(g)
    age = np.arange(n - 1, -1, -1, dtype=float)  # newest start age 0
    weights = np.power(0.5, age / 3.0)
    wip = float(np.sum(weights * ip_s.to_numpy(float)))
    if wip <= 0:
        return None
    wer = float(np.sum(weights * er_s.to_numpy(float)))
    wso = float(np.sum(weights * so_s.to_numpy(float)))
    wbb = float(np.sum(weights * bb_s.to_numpy(float)))
    whr = float(np.sum(weights * hr_s.to_numpy(float)))
    whits = float(np.sum(weights * h_s.to_numpy(float)))
    ew_era = 9.0 * wer / wip
    ew_k9 = 9.0 * wso / wip
    ew_bb9 = 9.0 * wbb / wip
    ew_hr9 = 9.0 * whr / wip
    ew_whip = (whits + wbb) / wip
    ew_fip = (13.0 * whr + 3.0 * wbb - 2.0 * wso) / wip + fip_const
    ew_kbb9 = 9.0 * (wso - wbb) / wip

    # Empirical-Bayes shrinkage toward broad MLB starter priors. We intentionally
    # use fixed priors rather than season-end league stats to avoid look-ahead.
    prior_ip = 20.0
    prior = {"ERA": 4.20, "K9": 8.50, "BB9": 3.20, "HR9": 1.20, "WHIP": 1.30, "FIP": 4.20, "KBB9": 5.30}
    shrink_w = ip / (ip + prior_ip)
    shr_era = shrink_w * season_era + (1.0 - shrink_w) * prior["ERA"]
    shr_k9 = shrink_w * season_k9 + (1.0 - shrink_w) * prior["K9"]
    shr_bb9 = shrink_w * season_bb9 + (1.0 - shrink_w) * prior["BB9"]
    shr_hr9 = shrink_w * season_hr9 + (1.0 - shrink_w) * prior["HR9"]
    shr_whip = shrink_w * season_whip + (1.0 - shrink_w) * prior["WHIP"]
    shr_fip = shrink_w * season_fip + (1.0 - shrink_w) * prior["FIP"]
    shr_kbb9 = shrink_w * season_kbb9 + (1.0 - shrink_w) * prior["KBB9"]

    last_start = g.iloc[-1]
    last_date = pd.to_datetime(last_start["date"], utc=True)
    days_rest = max(0.0, (game_time - last_date).total_seconds() / 86400.0)
    cutoff14 = game_time - pd.Timedelta(days=14)
    cutoff30 = game_time - pd.Timedelta(days=30)
    work14 = float(pd.to_numeric(g.loc[g["date"] >= cutoff14, "inningsPitched"], errors="coerce").fillna(0).sum())
    work30 = float(pd.to_numeric(g.loc[g["date"] >= cutoff30, "inningsPitched"], errors="coerce").fillna(0).sum())
    last_pitch = pd.to_numeric(pd.Series([last_start.get("numberOfPitches", np.nan)]), errors="coerce").iloc[0]
    p5 = pd.to_numeric(g.tail(5).get("numberOfPitches", pd.Series(dtype=float)), errors="coerce")
    avg_pitches5 = float(p5.mean()) if len(p5) and p5.notna().any() else np.nan

    return {
        "Starts": int(len(g)),
        "ERA": season_era,
        "K9": season_k9,
        "BB9": season_bb9,
        "HR9": season_hr9,
        "WHIP_Approx": season_whip,
        "Last5_ERA": l5_era,
        "Last5_K9": l5_k9,
        "Last5_BB9": l5_bb9,
        # v0.13 engineered features
        "Season_IP": ip,
        "IP_Per_Start": ip / max(1.0, float(len(g))),
        "FIP": season_fip,
        "KBB9": season_kbb9,
        "EW_ERA": ew_era,
        "EW_K9": ew_k9,
        "EW_BB9": ew_bb9,
        "EW_HR9": ew_hr9,
        "EW_WHIP": ew_whip,
        "EW_FIP": ew_fip,
        "EW_KBB9": ew_kbb9,
        "Shrunk_ERA": shr_era,
        "Shrunk_K9": shr_k9,
        "Shrunk_BB9": shr_bb9,
        "Shrunk_HR9": shr_hr9,
        "Shrunk_WHIP": shr_whip,
        "Shrunk_FIP": shr_fip,
        "Shrunk_KBB9": shr_kbb9,
        "Form_ERA_Delta": ew_era - shr_era,
        "Form_FIP_Delta": ew_fip - shr_fip,
        "Form_KBB9_Delta": ew_kbb9 - shr_kbb9,
        "Days_Rest": days_rest,
        "Workload14_IP": work14,
        "Workload30_IP": work30,
        "Last_Start_Pitches": float(last_pitch) if pd.notna(last_pitch) else np.nan,
        "Avg_Pitches_Last5": avg_pitches5,
    }


def attach_pitchers_to_master(master_df, schedule_df):
    m = master_df.copy()
    s = schedule_df.copy()

    m["Commence_Time"] = pd.to_datetime(m["Commence_Time"], errors="coerce", utc=True)
    s["Pitcher_GameDate"] = pd.to_datetime(s["Pitcher_GameDate"], errors="coerce", utc=True)

    # Prefer direct MLB_GamePk match from Moneyline Master.
    if "MLB_GamePk" in m.columns:
        merged = m.merge(
            s,
            on="MLB_GamePk",
            how="left",
            suffixes=("", "_sched")
        )
    else:
        m["_away_key"] = m["Away_Team"].map(normalize_team_name_for_match)
        m["_home_key"] = m["Home_Team"].map(normalize_team_name_for_match)
        s["_away_key"] = s["Away_Team_Pitcher"].map(normalize_team_name_for_match)
        s["_home_key"] = s["Home_Team_Pitcher"].map(normalize_team_name_for_match)
        m["_day"] = m["Commence_Time"].dt.date.astype(str)
        s["_day"] = s["Pitcher_GameDate"].dt.date.astype(str)
        merged = m.merge(
            s,
            on=["_away_key","_home_key","_day"],
            how="left",
            suffixes=("", "_sched")
        )
    return merged


def build_pitcher_feature_table(master_df, min_prior_team_games=12, min_pitcher_starts=3):
    """
    Enrich the existing PIT team-state table with point-in-time starter metrics.
    """
    pit_team = build_point_in_time_features(
        master_df,
        min_prior_games=min_prior_team_games
    )
    pit_team = pit_team[pit_team["PIT_Eligible"]].copy()

    seasons = sorted(int(x) for x in pit_team["Season"].dropna().unique())
    schedule_rows = []
    schedule_errors = []

    for season in seasons:
        payload, err, _ = fetch_season_schedule_with_pitchers(season)
        if err:
            schedule_errors.append(err)
            continue
        schedule_rows.extend(flatten_schedule_pitchers(payload))

    if schedule_errors and not schedule_rows:
        raise ValueError("Could not retrieve historical starter identities.")

    schedule_df = pd.DataFrame(schedule_rows)

    # Match starters onto PIT rows using MLB game identifiers from the master file.
    master_small = master_df.copy()
    master_small["Commence_Time"] = pd.to_datetime(master_small["Commence_Time"], errors="coerce", utc=True)
    # Preserve the historical market snapshot timing on the PIT rows.
    # v0.12.1.2 only merged MLB_GamePk here, which silently dropped
    # Hours_To_First_Pitch and caused every audit window to filter to zero rows.
    timing_cols = [c for c in [
        "Event_ID", "MLB_GamePk", "Snapshot_Timestamp", "Hours_To_First_Pitch"
    ] if c in master_small.columns]
    pit_plus = pit_team.merge(
        master_small[timing_cols].drop_duplicates("Event_ID"),
        on="Event_ID",
        how="left"
    )

    # If an older master has Snapshot_Timestamp but not Hours_To_First_Pitch,
    # reconstruct the interval directly.
    if "Hours_To_First_Pitch" not in pit_plus.columns and "Snapshot_Timestamp" in pit_plus.columns:
        snap = pd.to_datetime(pit_plus["Snapshot_Timestamp"], errors="coerce", utc=True)
        start = pd.to_datetime(pit_plus["Commence_Time"], errors="coerce", utc=True)
        pit_plus["Hours_To_First_Pitch"] = (start - snap).dt.total_seconds() / 3600.0

    pit_plus = attach_pitchers_to_master(pit_plus, schedule_df)

    pitcher_ids = set()
    for col in ["Away_Starter_ID","Home_Starter_ID"]:
        if col in pit_plus.columns:
            for v in pit_plus[col].dropna().unique():
                try:
                    pitcher_ids.add(int(v))
                except Exception:
                    pass

    # Load each pitcher-season game log once, cached.
    logs = {}
    total = len(pitcher_ids)
    progress = st.progress(0) if total else None
    status = st.empty() if total else None

    # Determine seasons per pitcher from target rows to avoid unnecessary calls.
    pitcher_seasons = set()
    for _, r in pit_plus.iterrows():
        season = int(r["Season"])
        for c in ["Away_Starter_ID","Home_Starter_ID"]:
            try:
                pitcher_seasons.add((season, int(r[c])))
            except Exception:
                pass

    for i, (season, pid) in enumerate(sorted(pitcher_seasons), start=1):
        if status is not None:
            status.caption(f"Loading free MLB pitcher history • {i:,} of {len(pitcher_seasons):,}")
        payload, err, _ = fetch_pitcher_season_gamelog(season, pid)
        if not err and payload:
            logs[(season, pid)] = parse_pitcher_gamelog(payload)
        if progress is not None:
            progress.progress(i / max(1, len(pitcher_seasons)))

    if progress is not None:
        progress.empty()
    if status is not None:
        status.empty()

    feat_rows = []
    for _, r in pit_plus.iterrows():
        season = int(r["Season"])
        game_time = pd.to_datetime(r["Commence_Time"], utc=True)
        try:
            aid = int(r["Away_Starter_ID"])
            hid = int(r["Home_Starter_ID"])
        except Exception:
            continue

        a = summarize_pitcher_before_game(
            logs.get((season, aid)), game_time, min_starts=min_pitcher_starts
        )
        h = summarize_pitcher_before_game(
            logs.get((season, hid)), game_time, min_starts=min_pitcher_starts
        )
        if not a or not h:
            continue

        row = r.to_dict()
        for k, v in a.items():
            row[f"Away_SP_{k}"] = v
        for k, v in h.items():
            row[f"Home_SP_{k}"] = v
        feat_rows.append(row)

    return pd.DataFrame(feat_rows)


def pitcher_feature_matrix(df):
    x = pit_feature_matrix(df).copy()
    x["SP_ERA_Diff"] = df["Away_SP_ERA"] - df["Home_SP_ERA"]
    x["SP_K9_Diff"] = df["Home_SP_K9"] - df["Away_SP_K9"]
    x["SP_BB9_Diff"] = df["Away_SP_BB9"] - df["Home_SP_BB9"]
    x["SP_HR9_Diff"] = df["Away_SP_HR9"] - df["Home_SP_HR9"]
    x["SP_WHIP_Diff"] = df["Away_SP_WHIP_Approx"] - df["Home_SP_WHIP_Approx"]
    x["SP_Last5_ERA_Diff"] = df["Away_SP_Last5_ERA"] - df["Home_SP_Last5_ERA"]
    x["SP_Last5_K9_Diff"] = df["Home_SP_Last5_K9"] - df["Away_SP_Last5_K9"]
    x["SP_Last5_BB9_Diff"] = df["Away_SP_Last5_BB9"] - df["Home_SP_Last5_BB9"]
    x["SP_Starts_Diff"] = df["Home_SP_Starts"] - df["Away_SP_Starts"]
    return x


def run_pitcher_point_in_time_backtest(master_df, min_prior_games=12, min_pitcher_starts=3):
    pit = build_pitcher_feature_table(
        master_df,
        min_prior_team_games=min_prior_games,
        min_pitcher_starts=min_pitcher_starts
    )
    if pit.empty:
        raise ValueError("No PIT games had sufficient starter history.")

    fmat = pitcher_feature_matrix(pit)
    pit = pd.concat([pit.reset_index(drop=True), fmat.reset_index(drop=True)], axis=1)
    fcols = list(fmat.columns)

    train23 = pit[pit["Season"] == 2023].copy()
    val24 = pit[pit["Season"] == 2024].copy()
    hold25 = pit[pit["Season"] == 2025].copy()

    if min(len(train23), len(val24), len(hold25)) < 400:
        raise ValueError(
            f"Insufficient PIT pitcher coverage: 2023={len(train23)}, "
            f"2024={len(val24)}, 2025={len(hold25)}."
        )

    X23 = train23[fcols].to_numpy(float)
    X24 = val24[fcols].to_numpy(float)
    y23 = train23["Home_Win"].to_numpy(float)
    y24 = val24["Home_Win"].to_numpy(float)

    X23z, X24z, _, _ = _standardize_train_apply(X23, X24)
    beta23 = fit_ridge_logit(X23z, y23, ridge=3.0)
    raw24 = predict_logit(beta23, X24z)
    val24["PIT_Pitcher_Raw_Prob"] = raw24

    weights = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    blend_rows = []
    for w in weights:
        p = blend_prob(raw24, val24["Home_Market_Prob"].to_numpy(float), w)
        blend_rows.append({
            "Model_Weight":w,
            "Brier_2024":brier_score_binary(p, y24),
            "LogLoss_2024":log_loss_binary(p, y24)
        })
    blend_table = pd.DataFrame(blend_rows).sort_values(["Brier_2024","LogLoss_2024"])
    best_weight = float(blend_table.iloc[0]["Model_Weight"])

    dev = pit[pit["Season"].isin([2023,2024])].copy()
    Xdev = dev[fcols].to_numpy(float)
    ydev = dev["Home_Win"].to_numpy(float)
    X25 = hold25[fcols].to_numpy(float)
    y25 = hold25["Home_Win"].to_numpy(float)

    Xdevz, X25z, _, _ = _standardize_train_apply(Xdev, X25)
    beta_dev = fit_ridge_logit(Xdevz, ydev, ridge=3.0)
    raw25 = predict_logit(beta_dev, X25z)

    hold25["PIT_Pitcher_Raw_Prob"] = raw25
    hold25["PIT_Pitcher_Calibrated_Prob"] = blend_prob(
        raw25, hold25["Home_Market_Prob"].to_numpy(float), best_weight
    )

    metrics = pd.DataFrame([
        {
            "Sample":"2024 validation",
            "Games":len(val24),
            "Market Brier":brier_score_binary(val24["Home_Market_Prob"], val24["Home_Win"]),
            "Raw Pitcher PIT Brier":brier_score_binary(val24["PIT_Pitcher_Raw_Prob"], val24["Home_Win"]),
            "Calibrated Brier":brier_score_binary(
                blend_prob(val24["PIT_Pitcher_Raw_Prob"], val24["Home_Market_Prob"], best_weight),
                val24["Home_Win"]
            ),
        },
        {
            "Sample":"2025 holdout",
            "Games":len(hold25),
            "Market Brier":brier_score_binary(hold25["Home_Market_Prob"], hold25["Home_Win"]),
            "Raw Pitcher PIT Brier":brier_score_binary(hold25["PIT_Pitcher_Raw_Prob"], hold25["Home_Win"]),
            "Calibrated Brier":brier_score_binary(
                hold25["PIT_Pitcher_Calibrated_Prob"], hold25["Home_Win"]
            ),
        }
    ])

    raw_bets = simulate_ml_bets(
        hold25.rename(columns={"PIT_Pitcher_Raw_Prob":"_P"}),
        "_P"
    )
    cal_bets = simulate_ml_bets(
        hold25.rename(columns={"PIT_Pitcher_Calibrated_Prob":"_P"}),
        "_P"
    )

    return {
        "pit":pit,
        "val24":val24,
        "hold25":hold25,
        "metrics":metrics,
        "blend_table":blend_table,
        "best_weight":best_weight,
        "raw_bets":raw_bets,
        "cal_bets":cal_bets,
        "raw_summary":summarize_sim_bets(raw_bets),
        "cal_summary":summarize_sim_bets(cal_bets),
        "feature_cols":fcols,
    }


def _mark_doubleheaders(df):
    """Flag team/date duplicates without relying on post-game scores."""
    x = df.copy()
    x["_audit_day"] = pd.to_datetime(x["Commence_Time"], errors="coerce", utc=True).dt.date.astype(str)
    x["_audit_matchup"] = x.apply(
        lambda r: "|".join(sorted([
            normalize_team_name_for_match(r.get("Away_Team")),
            normalize_team_name_for_match(r.get("Home_Team")),
        ])), axis=1
    )
    counts = x.groupby(["_audit_day", "_audit_matchup"])["Event_ID"].transform("nunique")
    x["Audit_Doubleheader"] = counts > 1
    return x.drop(columns=["_audit_day", "_audit_matchup"], errors="ignore")


def _audit_subset_backtest(pit, max_hours_to_first_pitch=None, exclude_doubleheaders=True,
                           min_starter_starts=3, label="Audit"):
    """
    Refit the exact same walk-forward model on a stricter eligible subset.
    2023 fit -> 2024 chooses market/model blend -> 2023+24 refit -> 2025 holdout.
    No threshold optimization is performed on 2025.
    """
    d = _mark_doubleheaders(pit)
    if "Hours_To_First_Pitch" not in d.columns:
        raise ValueError(
            "Audit timing is missing from the PIT table. Rebuild with a Moneyline Master that includes "
            "Snapshot_Timestamp / Hours_To_First_Pitch."
        )
    d["Hours_To_First_Pitch"] = pd.to_numeric(d["Hours_To_First_Pitch"], errors="coerce")
    if max_hours_to_first_pitch is not None:
        d = d[
            d["Hours_To_First_Pitch"].notna()
            & (d["Hours_To_First_Pitch"] >= 0)
            & (d["Hours_To_First_Pitch"] <= float(max_hours_to_first_pitch))
        ].copy()
    if exclude_doubleheaders:
        d = d[~d["Audit_Doubleheader"]].copy()
    d = d[(pd.to_numeric(d["Away_SP_Starts"], errors="coerce") >= int(min_starter_starts)) &
          (pd.to_numeric(d["Home_SP_Starts"], errors="coerce") >= int(min_starter_starts))].copy()

    if d.empty:
        return None
    fmat = pitcher_feature_matrix(d)
    d = pd.concat([d.reset_index(drop=True), fmat.reset_index(drop=True)], axis=1)
    fcols = list(fmat.columns)
    train23 = d[d["Season"] == 2023].copy()
    val24 = d[d["Season"] == 2024].copy()
    hold25 = d[d["Season"] == 2025].copy()
    if min(len(train23), len(val24), len(hold25)) < 150:
        return {"Label": label, "Error": f"Too little coverage: {len(train23)}/{len(val24)}/{len(hold25)}"}

    X23, X24 = train23[fcols].to_numpy(float), val24[fcols].to_numpy(float)
    y23, y24 = train23["Home_Win"].to_numpy(float), val24["Home_Win"].to_numpy(float)
    X23z, X24z, _, _ = _standardize_train_apply(X23, X24)
    b23 = fit_ridge_logit(X23z, y23, ridge=3.0)
    raw24 = predict_logit(b23, X24z)
    weights = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    blends=[]
    for w in weights:
        pp=blend_prob(raw24, val24["Home_Market_Prob"].to_numpy(float), w)
        blends.append((w,brier_score_binary(pp,y24),log_loss_binary(pp,y24)))
    blends=sorted(blends,key=lambda z:(z[1],z[2]))
    best_weight=float(blends[0][0])

    dev=d[d["Season"].isin([2023,2024])].copy()
    Xdev, X25 = dev[fcols].to_numpy(float), hold25[fcols].to_numpy(float)
    ydev, y25 = dev["Home_Win"].to_numpy(float), hold25["Home_Win"].to_numpy(float)
    Xdevz, X25z, _, _ = _standardize_train_apply(Xdev, X25)
    bdev=fit_ridge_logit(Xdevz,ydev,ridge=3.0)
    raw25=predict_logit(bdev,X25z)
    cal25=blend_prob(raw25,hold25["Home_Market_Prob"].to_numpy(float),best_weight)
    hold25=hold25.copy()
    hold25["Audit_Raw_Prob"]=raw25
    hold25["Audit_Cal_Prob"]=cal25
    bets=simulate_ml_bets(hold25.rename(columns={"Audit_Cal_Prob":"_P"}),"_P")
    bs=summarize_sim_bets(bets)
    return {
        "Label":label,"Games_2023":len(train23),"Games_2024":len(val24),"Games_2025":len(hold25),
        "Model_Weight":best_weight,
        "Market_Brier_2025":brier_score_binary(hold25["Home_Market_Prob"],hold25["Home_Win"]),
        "Model_Brier_2025":brier_score_binary(raw25,y25),
        "Cal_Brier_2025":brier_score_binary(cal25,y25),
        "Brier_Improvement":brier_score_binary(hold25["Home_Market_Prob"],hold25["Home_Win"])-brier_score_binary(cal25,y25),
        "Bets":bs["Bets"],"Wins":bs["Wins"],"Losses":bs["Losses"],"Units":bs["Units"],"ROI":bs["ROI"],
        "hold25":hold25,"bets":bets,
    }


def run_pitcher_audit(master_df, min_prior_games=12, min_pitcher_starts=3):
    """
    Deliberately hostile audit of the pitcher result. Starter identity from MLB's historical record
    is retrospective and therefore cannot by itself prove that the starter was known at the odds snapshot.
    We stress the result by moving snapshots closer to first pitch and removing doubleheaders.
    """
    pit=build_pitcher_feature_table(master_df,min_prior_team_games=min_prior_games,min_pitcher_starts=min_pitcher_starts)
    if pit.empty:
        raise ValueError("No PIT games had sufficient starter history for the audit.")
    specs=[
        (18,False,"Baseline ≤18h"),
        (18,True,"No doubleheaders ≤18h"),
        (12,True,"No DH • ≤12h"),
        (6,True,"No DH • ≤6h"),
        (3,True,"No DH • ≤3h"),
    ]
    results=[]
    full={}
    for hrs,no_dh,label in specs:
        r=_audit_subset_backtest(pit,max_hours_to_first_pitch=hrs,exclude_doubleheaders=no_dh,
                                 min_starter_starts=min_pitcher_starts,label=label)
        if r:
            full[label]=r
            results.append({k:v for k,v in r.items() if k not in ("hold25","bets")})
    table=pd.DataFrame(results)
    timing = pd.to_numeric(pit.get("Hours_To_First_Pitch"), errors="coerce") if "Hours_To_First_Pitch" in pit.columns else pd.Series(dtype=float)
    timing_diag = {
        "pit_rows": int(len(pit)),
        "timing_nonnull": int(timing.notna().sum()) if len(timing) else 0,
        "timing_min": float(timing.min()) if timing.notna().any() else None,
        "timing_median": float(timing.median()) if timing.notna().any() else None,
        "timing_max": float(timing.max()) if timing.notna().any() else None,
        "within_18h": int(((timing >= 0) & (timing <= 18)).sum()) if len(timing) else 0,
        "within_12h": int(((timing >= 0) & (timing <= 12)).sum()) if len(timing) else 0,
        "within_6h": int(((timing >= 0) & (timing <= 6)).sum()) if len(timing) else 0,
        "within_3h": int(((timing >= 0) & (timing <= 3)).sum()) if len(timing) else 0,
    }
    return {"pit":pit,"table":table,"results":full,"timing_diagnostics":timing_diag}


def _integrity_prepare_subset(pit, max_hours=6, min_starts=3, established_starts=None, trim_extremes=False):
    d=_mark_doubleheaders(pit)
    d["Hours_To_First_Pitch"]=pd.to_numeric(d.get("Hours_To_First_Pitch"),errors="coerce")
    d=d[d["Hours_To_First_Pitch"].between(0,float(max_hours),inclusive="both")].copy()
    d=d[~d["Audit_Doubleheader"]].copy()
    need=int(established_starts if established_starts is not None else min_starts)
    d=d[(pd.to_numeric(d["Away_SP_Starts"],errors="coerce")>=need)&(pd.to_numeric(d["Home_SP_Starts"],errors="coerce")>=need)].copy()
    if trim_extremes and not d.empty:
        era=(pd.to_numeric(d["Away_SP_ERA"],errors="coerce")-pd.to_numeric(d["Home_SP_ERA"],errors="coerce")).abs()
        k9=(pd.to_numeric(d["Away_SP_K9"],errors="coerce")-pd.to_numeric(d["Home_SP_K9"],errors="coerce")).abs()
        bb9=(pd.to_numeric(d["Away_SP_BB9"],errors="coerce")-pd.to_numeric(d["Home_SP_BB9"],errors="coerce")).abs()
        d=d[(era<=5.0)&(k9<=8.0)&(bb9<=6.0)].copy()
    return d


def _integrity_fit_eval(d, mode="correct", seed=122, cap_probs=False, label="Correct starters"):
    if d.empty:
        return {"Test":label,"Error":"No eligible rows"}
    work=d.copy().reset_index(drop=True)
    sp_cols=[c for c in work.columns if c.startswith("Away_SP_") or c.startswith("Home_SP_")]
    if mode=="scramble":
        rng=np.random.default_rng(seed)
        for season in sorted(work["Season"].dropna().unique()):
            idx=work.index[work["Season"]==season].to_numpy()
            perm=rng.permutation(idx)
            work.loc[idx,sp_cols]=work.loc[perm,sp_cols].to_numpy()
    elif mode=="swap":
        metrics=sorted({c.replace("Away_SP_","") for c in work.columns if c.startswith("Away_SP_")})
        for m in metrics:
            a,h=f"Away_SP_{m}",f"Home_SP_{m}"
            if a in work.columns and h in work.columns:
                tmp=work[a].copy(); work[a]=work[h].to_numpy(); work[h]=tmp.to_numpy()
    fmat=pit_feature_matrix(work).copy() if mode=="team_only" else pitcher_feature_matrix(work).copy()
    work=pd.concat([work.reset_index(drop=True),fmat.reset_index(drop=True)],axis=1)
    fcols=list(fmat.columns)
    tr=work[work["Season"]==2023].copy(); va=work[work["Season"]==2024].copy(); ho=work[work["Season"]==2025].copy()
    if min(len(tr),len(va),len(ho))<150:
        return {"Test":label,"Error":f"Too little coverage: {len(tr)}/{len(va)}/{len(ho)}"}
    Xtr,Xva=tr[fcols].to_numpy(float),va[fcols].to_numpy(float)
    ytr,yva=tr["Home_Win"].to_numpy(float),va["Home_Win"].to_numpy(float)
    Xtrz,Xvaz,_,_=_standardize_train_apply(Xtr,Xva)
    b=fit_ridge_logit(Xtrz,ytr,ridge=3.0); rawva=predict_logit(b,Xvaz)
    candidates=[]
    for w in [0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
        pp=blend_prob(rawva,va["Home_Market_Prob"].to_numpy(float),w)
        candidates.append((w,brier_score_binary(pp,yva),log_loss_binary(pp,yva)))
    bw=float(sorted(candidates,key=lambda z:(z[1],z[2]))[0][0])
    dev=work[work["Season"].isin([2023,2024])].copy()
    Xdev,X25=dev[fcols].to_numpy(float),ho[fcols].to_numpy(float)
    ydev,y25=dev["Home_Win"].to_numpy(float),ho["Home_Win"].to_numpy(float)
    Xdz,X25z,_,_=_standardize_train_apply(Xdev,X25)
    bd=fit_ridge_logit(Xdz,ydev,ridge=3.0); raw25=predict_logit(bd,X25z)
    if cap_probs:
        raw25=np.clip(raw25,.20,.80)
    cal25=blend_prob(raw25,ho["Home_Market_Prob"].to_numpy(float),bw)
    ho=ho.copy(); ho["Integrity_Prob"]=cal25
    bets=simulate_ml_bets(ho,"Integrity_Prob"); bs=summarize_sim_bets(bets)
    mb=brier_score_binary(ho["Home_Market_Prob"],ho["Home_Win"]); rb=brier_score_binary(raw25,y25); cb=brier_score_binary(cal25,y25)
    return {"Test":label,"Games_2025":len(ho),"Model_Weight":bw,"Market_Brier":mb,"Raw_Model_Brier":rb,"Cal_Brier":cb,"Brier_Improvement":mb-cb,"Bets":bs["Bets"],"Wins":bs["Wins"],"Losses":bs["Losses"],"Units":bs["Units"],"ROI":bs["ROI"],"hold25":ho,"bets":bets}



def _causality_corrupt_holdout(raw_hold, mode="correct", seed=123):
    """Corrupt starter identity ONLY at inference time.

    Critical design point: the model is trained on the correct historical features.  A
    placebo is then applied only to the 2025 holdout, so the regression cannot simply
    relearn the inverse mapping (the flaw in a train+test swapped-starter placebo).
    """
    h = raw_hold.copy().reset_index(drop=True)
    sp_cols = [c for c in h.columns if c.startswith("Away_SP_") or c.startswith("Home_SP_")]
    if mode == "correct":
        return h
    if mode == "scramble":
        rng = np.random.default_rng(seed)
        # Shuffle complete starter matchup rows, preserving internally coherent pitcher lines.
        perm = rng.permutation(len(h))
        h.loc[:, sp_cols] = h.loc[perm, sp_cols].to_numpy()
        return h
    if mode == "swap":
        metrics = sorted({c.replace("Away_SP_", "") for c in h.columns if c.startswith("Away_SP_")})
        for m in metrics:
            a, b = f"Away_SP_{m}", f"Home_SP_{m}"
            if a in h.columns and b in h.columns:
                av = h[a].copy()
                h[a] = h[b].to_numpy()
                h[b] = av.to_numpy()
        return h
    if mode == "lagged":
        # Assign the preceding eligible game's starter feature row. This is knowingly
        # wrong and should materially hurt a genuinely starter-driven signal.
        order = pd.to_datetime(h["Commence_Time"], errors="coerce", utc=True).sort_values().index
        ordered = h.loc[order, sp_cols].copy()
        shifted = ordered.shift(1)
        shifted.iloc[0] = ordered.iloc[-1].to_numpy()
        h.loc[order, sp_cols] = shifted.to_numpy()
        return h
    raise ValueError(f"Unknown causality placebo mode: {mode}")


def _fit_correct_model_and_eval_holdout(d, holdout_mode="correct", seed=123, label="Correct starters"):
    """Train/validate on correct features; corrupt only the untouched 2025 inference rows."""
    if d.empty:
        return {"Test": label, "Type": "Inference placebo", "Error": "No eligible rows"}
    raw = d.copy().reset_index(drop=True)
    correct_f = pitcher_feature_matrix(raw).reset_index(drop=True)
    work = pd.concat([raw, correct_f], axis=1)
    fcols = list(correct_f.columns)
    tr = work[work["Season"] == 2023].copy()
    va = work[work["Season"] == 2024].copy()
    ho_raw = raw[raw["Season"] == 2025].copy().reset_index(drop=True)
    ho_meta = work[work["Season"] == 2025].copy().reset_index(drop=True)
    if min(len(tr), len(va), len(ho_raw)) < 150:
        return {"Test": label, "Type": "Inference placebo", "Error": f"Too little coverage: {len(tr)}/{len(va)}/{len(ho_raw)}"}

    # Select blend using ONLY correct 2024 features.
    Xtr, Xva = tr[fcols].to_numpy(float), va[fcols].to_numpy(float)
    ytr, yva = tr["Home_Win"].to_numpy(float), va["Home_Win"].to_numpy(float)
    Xtrz, Xvaz, _, _ = _standardize_train_apply(Xtr, Xva)
    b23 = fit_ridge_logit(Xtrz, ytr, ridge=3.0)
    raw24 = predict_logit(b23, Xvaz)
    choices = []
    for w in [0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
        p = blend_prob(raw24, va["Home_Market_Prob"].to_numpy(float), w)
        choices.append((w, brier_score_binary(p, yva), log_loss_binary(p, yva)))
    bw = float(sorted(choices, key=lambda z:(z[1],z[2]))[0][0])

    # Refit on 2023+2024 correct data.
    dev = work[work["Season"].isin([2023, 2024])].copy()
    Xdev = dev[fcols].to_numpy(float)
    ydev = dev["Home_Win"].to_numpy(float)
    Xdummy = ho_meta[fcols].to_numpy(float)
    Xdevz, _, mu, sd = _standardize_train_apply(Xdev, Xdummy)
    bd = fit_ridge_logit(Xdevz, ydev, ridge=3.0)

    # Only now corrupt starter identity in 2025.
    corrupted = _causality_corrupt_holdout(ho_raw, mode=holdout_mode, seed=seed)
    hf = pitcher_feature_matrix(corrupted).reset_index(drop=True)
    Xh = hf[fcols].to_numpy(float)
    Xhfill = np.where(np.isnan(Xh), mu, Xh)
    Xhz = (Xhfill - mu) / sd
    raw25 = predict_logit(bd, Xhz)
    y25 = ho_meta["Home_Win"].to_numpy(float)
    mkt = ho_meta["Home_Market_Prob"].to_numpy(float)
    cal25 = blend_prob(raw25, mkt, bw)

    out = ho_meta.copy()
    out["Causality_Raw_Prob"] = raw25
    out["Causality_Prob"] = cal25
    bets = simulate_ml_bets(out, "Causality_Prob")
    bs = summarize_sim_bets(bets)
    mb = brier_score_binary(mkt, y25)
    rb = brier_score_binary(raw25, y25)
    cb = brier_score_binary(cal25, y25)
    return {
        "Test": label, "Type": "Inference placebo" if holdout_mode != "correct" else "Baseline",
        "Games_2025": len(out), "Model_Weight": bw, "Market_Brier": mb,
        "Raw_Model_Brier": rb, "Cal_Brier": cb, "Brier_Improvement": mb-cb,
        "Bets": bs["Bets"], "Wins": bs["Wins"], "Losses": bs["Losses"],
        "Units": bs["Units"], "ROI": bs["ROI"], "hold25": out, "bets": bets,
    }


def _fit_eval_ablation(d, drop_cols, label):
    """Retrain with selected derived features removed, preserving 2023→2024→2025 protocol."""
    if d.empty:
        return {"Test": label, "Type": "Feature ablation", "Error": "No eligible rows"}
    raw = d.copy().reset_index(drop=True)
    fmat = pitcher_feature_matrix(raw).reset_index(drop=True)
    keep = [c for c in fmat.columns if c not in set(drop_cols)]
    if not keep:
        return {"Test": label, "Type": "Feature ablation", "Error": "No features left"}
    work = pd.concat([raw, fmat], axis=1)
    tr = work[work["Season"]==2023].copy(); va=work[work["Season"]==2024].copy(); ho=work[work["Season"]==2025].copy()
    if min(len(tr),len(va),len(ho)) < 150:
        return {"Test": label, "Type": "Feature ablation", "Error": f"Too little coverage: {len(tr)}/{len(va)}/{len(ho)}"}
    Xtr,Xva=tr[keep].to_numpy(float),va[keep].to_numpy(float)
    ytr,yva=tr["Home_Win"].to_numpy(float),va["Home_Win"].to_numpy(float)
    Xtrz,Xvaz,_,_=_standardize_train_apply(Xtr,Xva)
    b=fit_ridge_logit(Xtrz,ytr,ridge=3.0); rawva=predict_logit(b,Xvaz)
    candidates=[]
    for w in [0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
        pp=blend_prob(rawva,va["Home_Market_Prob"].to_numpy(float),w)
        candidates.append((w,brier_score_binary(pp,yva),log_loss_binary(pp,yva)))
    bw=float(sorted(candidates,key=lambda z:(z[1],z[2]))[0][0])
    dev=work[work["Season"].isin([2023,2024])].copy()
    Xdev,X25=dev[keep].to_numpy(float),ho[keep].to_numpy(float)
    ydev,y25=dev["Home_Win"].to_numpy(float),ho["Home_Win"].to_numpy(float)
    Xdz,X25z,_,_=_standardize_train_apply(Xdev,X25)
    bd=fit_ridge_logit(Xdz,ydev,ridge=3.0); raw25=predict_logit(bd,X25z)
    cal25=blend_prob(raw25,ho["Home_Market_Prob"].to_numpy(float),bw)
    out=ho.copy(); out["Causality_Prob"]=cal25
    bets=simulate_ml_bets(out,"Causality_Prob"); bs=summarize_sim_bets(bets)
    mb=brier_score_binary(ho["Home_Market_Prob"],ho["Home_Win"]); rb=brier_score_binary(raw25,y25); cb=brier_score_binary(cal25,y25)
    return {"Test":label,"Type":"Feature ablation","Games_2025":len(ho),"Model_Weight":bw,"Market_Brier":mb,"Raw_Model_Brier":rb,"Cal_Brier":cb,"Brier_Improvement":mb-cb,"Bets":bs["Bets"],"Wins":bs["Wins"],"Losses":bs["Losses"],"Units":bs["Units"],"ROI":bs["ROI"],"hold25":out,"bets":bets}


def run_pitcher_causality_audit(master_df, min_prior_games=12, min_pitcher_starts=3):
    pit = build_pitcher_feature_table(master_df, min_prior_team_games=min_prior_games, min_pitcher_starts=min_pitcher_starts)
    if pit.empty:
        raise ValueError("No PIT pitcher rows were built.")
    strict = _integrity_prepare_subset(pit, max_hours=6, min_starts=min_pitcher_starts)
    if strict.empty:
        raise ValueError("No eligible ≤6h non-doubleheader rows were available.")

    tests = []
    # Inference-time placebos: model never gets to relearn the corrupted mapping.
    for mode, label in [
        ("correct", "Correct starters • ≤6h"),
        ("scramble", "2025 scrambled starters • inference-only"),
        ("swap", "2025 opponent starters • inference-only"),
        ("lagged", "2025 lagged wrong starters • inference-only"),
    ]:
        tests.append(_fit_correct_model_and_eval_holdout(strict, holdout_mode=mode, seed=123, label=label))

    # Feature-family ablations, each independently re-fit and validated.
    groups = {
        "Remove ERA": ["SP_ERA_Diff"],
        "Remove K/9": ["SP_K9_Diff"],
        "Remove BB/9": ["SP_BB9_Diff"],
        "Remove HR/9": ["SP_HR9_Diff"],
        "Remove WHIP": ["SP_WHIP_Diff"],
        "Remove recent form": ["SP_Last5_ERA_Diff","SP_Last5_K9_Diff","SP_Last5_BB9_Diff"],
        "Remove starter experience": ["SP_Starts_Diff"],
        "Remove ALL pitcher features": [
            "SP_ERA_Diff","SP_K9_Diff","SP_BB9_Diff","SP_HR9_Diff","SP_WHIP_Diff",
            "SP_Last5_ERA_Diff","SP_Last5_K9_Diff","SP_Last5_BB9_Diff","SP_Starts_Diff"
        ],
    }
    for name, cols in groups.items():
        tests.append(_fit_eval_ablation(strict, cols, name))

    table = pd.DataFrame([{k:v for k,v in r.items() if k not in ("hold25","bets")} for r in tests])
    full = {r.get("Test"): r for r in tests}

    # Robustness segments for the true-starter baseline.
    seg=[]; base=full.get("Correct starters • ≤6h",{}); bets=base.get("bets") if isinstance(base,dict) else None
    if bets is not None and not bets.empty:
        x=bets.copy(); x["Commence_Time"]=pd.to_datetime(x["Commence_Time"],errors="coerce",utc=True); x["Month"]=x["Commence_Time"].dt.strftime("%Y-%m")
        for name,g in x.groupby("Month"):
            seg.append({"Segment":name,**summarize_sim_bets(g)})
        for name,g in [("Favorites",x[x["Odds"]<0]),("Underdogs",x[x["Odds"]>0])]:
            seg.append({"Segment":name,**summarize_sim_bets(g)})
    return {"pit":pit,"table":table,"results":full,"segments":pd.DataFrame(seg)}


PITCHER_V2_FEATURES = [
    "Season_WinPct_Diff","Season_RunDiff_Diff","Last10_WinPct_Diff","Last10_RunDiff_Diff","Rest_Diff",
    "V2_Shrunk_FIP_Diff","V2_Shrunk_ERA_Diff","V2_Shrunk_KBB9_Diff",
    "V2_EW_FIP_Diff","V2_EW_KBB9_Diff","V2_Form_FIP_Diff","V2_Form_KBB9_Diff",
    "V2_IPPerStart_Diff","V2_DaysRest_Diff","V2_Workload14_Diff","V2_Workload30_Diff",
    "V2_LastStartPitches_Diff","V2_AvgPitches5_Diff","V2_Starts_Diff",
]


def pitcher_v2_feature_matrix(df):
    """Engineered v0.13 starter features. Signs are oriented so positive generally
    means a home-side advantage when practical; ridge logistic is free to learn
    either direction. Missing pitch-count fields are imputed from development data.
    """
    x = pit_feature_matrix(df).copy()
    x["V2_Shrunk_FIP_Diff"] = df["Away_SP_Shrunk_FIP"] - df["Home_SP_Shrunk_FIP"]
    x["V2_Shrunk_ERA_Diff"] = df["Away_SP_Shrunk_ERA"] - df["Home_SP_Shrunk_ERA"]
    x["V2_Shrunk_KBB9_Diff"] = df["Home_SP_Shrunk_KBB9"] - df["Away_SP_Shrunk_KBB9"]
    x["V2_EW_FIP_Diff"] = df["Away_SP_EW_FIP"] - df["Home_SP_EW_FIP"]
    x["V2_EW_KBB9_Diff"] = df["Home_SP_EW_KBB9"] - df["Away_SP_EW_KBB9"]
    x["V2_Form_FIP_Diff"] = df["Away_SP_Form_FIP_Delta"] - df["Home_SP_Form_FIP_Delta"]
    x["V2_Form_KBB9_Diff"] = df["Home_SP_Form_KBB9_Delta"] - df["Away_SP_Form_KBB9_Delta"]
    x["V2_IPPerStart_Diff"] = df["Home_SP_IP_Per_Start"] - df["Away_SP_IP_Per_Start"]
    x["V2_DaysRest_Diff"] = df["Home_SP_Days_Rest"] - df["Away_SP_Days_Rest"]
    x["V2_Workload14_Diff"] = df["Away_SP_Workload14_IP"] - df["Home_SP_Workload14_IP"]
    x["V2_Workload30_Diff"] = df["Away_SP_Workload30_IP"] - df["Home_SP_Workload30_IP"]
    x["V2_LastStartPitches_Diff"] = df["Away_SP_Last_Start_Pitches"] - df["Home_SP_Last_Start_Pitches"]
    x["V2_AvgPitches5_Diff"] = df["Home_SP_Avg_Pitches_Last5"] - df["Away_SP_Avg_Pitches_Last5"]
    x["V2_Starts_Diff"] = df["Home_SP_Starts"] - df["Away_SP_Starts"]
    return x


def _fit_feature_model_walkforward(d, feature_builder, label, ridge=3.0):
    """Fixed 2023 train -> 2024 blend selection -> 2023+24 refit -> 2025 holdout.
    Betting thresholds are exactly the same simulate_ml_bets defaults used in v0.12.3.
    """
    raw = d.copy().reset_index(drop=True)
    fmat = feature_builder(raw).reset_index(drop=True)
    work = pd.concat([raw, fmat], axis=1)
    fcols = list(fmat.columns)
    tr = work[work["Season"] == 2023].copy()
    va = work[work["Season"] == 2024].copy()
    ho = work[work["Season"] == 2025].copy()
    if min(len(tr), len(va), len(ho)) < 150:
        return {"Model":label,"Error":f"Too little coverage: {len(tr)}/{len(va)}/{len(ho)}"}

    Xtr, Xva = tr[fcols].to_numpy(float), va[fcols].to_numpy(float)
    ytr, yva = tr["Home_Win"].to_numpy(float), va["Home_Win"].to_numpy(float)
    Xtrz, Xvaz, _, _ = _standardize_train_apply(Xtr, Xva)
    b23 = fit_ridge_logit(Xtrz, ytr, ridge=ridge)
    raw24 = predict_logit(b23, Xvaz)
    choices=[]
    for w in [0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
        p=blend_prob(raw24, va["Home_Market_Prob"].to_numpy(float), w)
        choices.append((w,brier_score_binary(p,yva),log_loss_binary(p,yva)))
    bw,b24,ll24=sorted(choices,key=lambda z:(z[1],z[2]))[0]
    bw=float(bw)

    dev=work[work["Season"].isin([2023,2024])].copy()
    Xdev,X25=dev[fcols].to_numpy(float),ho[fcols].to_numpy(float)
    ydev,y25=dev["Home_Win"].to_numpy(float),ho["Home_Win"].to_numpy(float)
    Xdz,X25z,mu,sd=_standardize_train_apply(Xdev,X25)
    bd=fit_ridge_logit(Xdz,ydev,ridge=ridge)
    raw25=predict_logit(bd,X25z)
    mkt=ho["Home_Market_Prob"].to_numpy(float)
    cal25=blend_prob(raw25,mkt,bw)
    out=ho.copy(); out["V13_Raw_Prob"]=raw25; out["V13_Cal_Prob"]=cal25
    bets=simulate_ml_bets(out,"V13_Cal_Prob")
    bs=summarize_sim_bets(bets)
    coef=pd.DataFrame({"Feature":["Intercept"]+fcols,"Coefficient":bd})
    coef["Abs_Coefficient"]=coef["Coefficient"].abs()
    coef=coef.sort_values("Abs_Coefficient",ascending=False).reset_index(drop=True)
    return {
        "Model":label,"Games_2025":len(ho),"Model_Weight":bw,
        "Validation_2024_Brier":float(b24),"Validation_2024_LogLoss":float(ll24),
        "Market_Brier":brier_score_binary(mkt,y25),"Raw_Model_Brier":brier_score_binary(raw25,y25),
        "Cal_Brier":brier_score_binary(cal25,y25),"Market_LogLoss":log_loss_binary(mkt,y25),
        "Raw_Model_LogLoss":log_loss_binary(raw25,y25),"Cal_LogLoss":log_loss_binary(cal25,y25),
        "Brier_Improvement":brier_score_binary(mkt,y25)-brier_score_binary(cal25,y25),
        "Bets":bs["Bets"],"Wins":bs["Wins"],"Losses":bs["Losses"],"Units":bs["Units"],"ROI":bs["ROI"],
        "hold25":out,"bets":bets,"coefficients":coef,
    }


def run_pitcher_model_2_test(master_df,min_prior_games=12,min_pitcher_starts=3):
    """v0.13 benchmark test. Builds one PIT dataset then compares the frozen
    v0.12.3 feature recipe against Pitcher Model 2.0 on the identical strict sample.
    """
    pit=build_pitcher_feature_table(master_df,min_prior_team_games=min_prior_games,min_pitcher_starts=min_pitcher_starts)
    if pit.empty:
        raise ValueError("No PIT pitcher rows were built.")
    strict=_integrity_prepare_subset(pit,max_hours=6,min_starts=min_pitcher_starts)
    if strict.empty:
        raise ValueError("No eligible ≤6h non-doubleheader rows were available.")

    base=_fit_feature_model_walkforward(strict,pitcher_feature_matrix,"v0.12.3 benchmark",ridge=3.0)
    v2=_fit_feature_model_walkforward(strict,pitcher_v2_feature_matrix,"v0.13 Pitcher Model 2.0",ridge=5.0)
    results=[base,v2]
    table=pd.DataFrame([{k:v for k,v in r.items() if k not in ("hold25","bets","coefficients")} for r in results])

    seg=[]
    if not v2.get("Error") and isinstance(v2.get("bets"),pd.DataFrame) and not v2["bets"].empty:
        x=v2["bets"].copy(); x["Commence_Time"]=pd.to_datetime(x["Commence_Time"],errors="coerce",utc=True)
        x["Month"]=x["Commence_Time"].dt.strftime("%Y-%m")
        for name,g in x.groupby("Month"):
            seg.append({"Segment":name,**summarize_sim_bets(g)})
        for name,g in [("Favorites",x[x["Odds"]<0]),("Underdogs",x[x["Odds"]>0])]:
            seg.append({"Segment":name,**summarize_sim_bets(g)})

    return {
        "pit":pit,"strict_rows":len(strict),"table":table,
        "results":{r.get("Model"):r for r in results},"segments":pd.DataFrame(seg),
        "coefficients":v2.get("coefficients",pd.DataFrame()),
    }


# ===== v0.14.1 point-in-time bullpen research layer =====
BULLPEN_CACHE_DIR = Path(".mlb_bullpen_cache")
BULLPEN_CACHE_DIR.mkdir(exist_ok=True)


def _bullpen_cache_file(season, team_id):
    # v0.14.1 deliberately uses a new cache namespace so the empty/broken
    # team-level gameLog responses from v0.14.0 can never be reused.
    return BULLPEN_CACHE_DIR / f"v141_team_{int(team_id)}_{int(season)}_relief_rows.json"


def _historical_team_roster(season, team_id):
    """Return pitcher IDs who appeared on a team's historical season roster.

    MLB's generic /stats?teamId=... gameLog query does not return the
    per-pitcher appearance ledger we need.  v0.14.1 instead gets the historical
    full-season roster, then reuses the already-tested player gameLog endpoint
    from the starter PIT model.
    """
    url = f"https://statsapi.mlb.com/api/v1/teams/{int(team_id)}/roster"
    attempts = ["fullSeason", "40Man", "active"]
    last_err = None
    for roster_type in attempts:
        try:
            resp = requests.get(
                url,
                params={"rosterType": roster_type, "season": int(season), "hydrate": "person"},
                timeout=35,
            )
        except requests.RequestException as ex:
            last_err = f"roster request failed: {ex}"
            continue
        if resp.status_code != 200:
            last_err = f"roster HTTP {resp.status_code}"
            continue
        try:
            payload = resp.json()
        except Exception:
            last_err = "roster response was not valid JSON"
            continue

        ids = []
        for item in payload.get("roster", []) or []:
            person = item.get("person") or {}
            pos = item.get("position") or person.get("primaryPosition") or {}
            ptype = str(pos.get("type", "")).lower()
            pcode = str(pos.get("code", ""))
            pabbr = str(pos.get("abbreviation", "")).upper()
            # Pitchers are normally code 1 / type Pitcher. Include two-way
            # players defensively because their pitching gameLog is still valid.
            is_pitcher = (ptype == "pitcher") or (pcode == "1") or (pabbr in {"P", "TWP"})
            pid = person.get("id")
            if is_pitcher and pid is not None:
                try:
                    ids.append(int(pid))
                except Exception:
                    pass
        ids = sorted(set(ids))
        if ids:
            return ids, None, roster_type
        last_err = f"{roster_type} roster contained no pitcher IDs"
    return [], last_err or "no historical roster data", None


def _parse_player_relief_rows_for_team(payload, pitcher_id, team_id, season):
    """Parse one pitcher's historical gameLog and keep only relief work for team.

    The split-level team ID is required when available so a traded pitcher's
    appearances for another club do not leak into this team's bullpen history.
    If MLB omits a split team entirely, the row is retained only as a fallback;
    the roster itself still anchors the pitcher to the requested team-season.
    """
    rows = []
    target_team = int(team_id)
    for block in (payload or {}).get("stats", []) or []:
        for split in block.get("splits", []) or []:
            stat = split.get("stat") or {}
            game = split.get("game") or {}
            raw_date = split.get("date")
            if not raw_date:
                continue
            try:
                d = pd.to_datetime(raw_date, errors="raise", utc=True)
            except Exception:
                continue

            split_team = split.get("team") or {}
            split_team_id = split_team.get("id")
            if split_team_id is not None:
                try:
                    if int(split_team_id) != target_team:
                        continue
                except Exception:
                    continue

            def num(key, default=0.0):
                try:
                    v = stat.get(key)
                    return float(v) if v not in (None, "", "-") else float(default)
                except Exception:
                    return float(default)

            if num("gamesStarted", 0.0) >= 1.0:
                continue

            try:
                txt = str(stat.get("inningsPitched", 0.0)).strip()
                if "." in txt:
                    whole, outs = txt.split(".", 1)
                    oi = int(outs)
                    ip = float(int(whole)) + (oi / 3.0 if oi in (0, 1, 2) else float("nan"))
                else:
                    ip = float(txt)
            except Exception:
                ip = float("nan")
            if not np.isfinite(ip) or ip <= 0:
                continue

            rows.append({
                "date": d.isoformat(),
                "gamePk": game.get("gamePk"),
                "team_id": target_team,
                "season": int(season),
                "player_id": int(pitcher_id),
                "inningsPitched": float(ip),
                "earnedRuns": num("earnedRuns"),
                "strikeOuts": num("strikeOuts"),
                "baseOnBalls": num("baseOnBalls"),
                "homeRuns": num("homeRuns"),
                "hits": num("hits"),
                "hitBatsmen": num("hitBatsmen"),
                "battersFaced": num("battersFaced"),
                "numberOfPitches": num("numberOfPitches", np.nan),
            })
    return rows


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_team_pitching_gamelog(season, team_id):
    """Build one team's relief-appearance ledger from roster + player gameLogs.

    This replaces v0.14.0's invalid assumption that the generic MLB /stats
    endpoint can return every pitcher appearance for a team-season in one call.
    It costs ZERO Odds API credits. MLB Stats API calls are cached both at the
    player-season level and as a finished team-season relief ledger.
    """
    cache_file = _bullpen_cache_file(season, team_id)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if (cached or {}).get("_relief_rows"):
                return cached, None, True
        except Exception:
            pass

    pitcher_ids, roster_err, roster_type = _historical_team_roster(season, team_id)
    if not pitcher_ids:
        return None, f"Team {team_id} {season} historical roster failed: {roster_err}", False

    rows = []
    errors = []
    cache_hits = 0

    # Keep the nested pool small. Outer build_bullpen_feature_table already
    # parallelizes team-seasons; too much concurrency can make MLB throttle us.
    def load_pitcher(pid):
        payload, err, cached = fetch_pitcher_season_gamelog(int(season), int(pid))
        if err or not payload:
            return [], err or f"Pitcher {pid} returned no gameLog payload", cached
        return _parse_player_relief_rows_for_team(payload, pid, team_id, season), None, cached

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(pitcher_ids)))) as ex:
        futs = {ex.submit(load_pitcher, pid): pid for pid in pitcher_ids}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                prow, err, cached = fut.result()
            except Exception as exc:
                prow, err, cached = [], f"Pitcher {pid} parse failed: {exc}", False
            if cached:
                cache_hits += 1
            if err:
                errors.append(err)
            if prow:
                rows.extend(prow)

    if rows:
        # A traded/re-added player can show up more than once in roster payloads;
        # ensure one pitcher/game appearance only.
        dedup = {}
        for row in rows:
            key = (row.get("player_id"), row.get("gamePk"), row.get("date"))
            dedup[key] = row
        rows = list(dedup.values())
        rows.sort(key=lambda r: (str(r.get("date")), int(r.get("player_id") or 0)))

    out = {
        "_relief_rows": rows,
        "_meta": {
            "team_id": int(team_id),
            "season": int(season),
            "roster_type": roster_type,
            "pitchers_requested": len(pitcher_ids),
            "player_cache_hits": cache_hits,
            "player_errors": len(errors),
        },
    }
    if rows:
        try:
            cache_file.write_text(json.dumps(out, allow_nan=True))
        except Exception:
            pass
        return out, None, False

    sample = "; ".join(errors[:2])
    return None, (
        f"Team {team_id} {season} roster loaded {len(pitcher_ids)} pitchers but produced "
        f"no team relief appearances. {sample}"
    ).strip(), False


def parse_team_reliever_gamelog(payload, team_id=None, season=None):
    """Parse the v0.14.1 synthetic relief ledger (with legacy fallback)."""
    if isinstance(payload, dict) and "_relief_rows" in payload:
        df = pd.DataFrame(payload.get("_relief_rows") or [])
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        return df.dropna(subset=["date"]).copy()

    # Legacy parser retained so old valid cached payloads (if any) remain usable.
    rows = []
    for block in (payload or {}).get("stats", []) or []:
        for split in block.get("splits", []) or []:
            stat = split.get("stat") or {}
            game = split.get("game") or {}
            person = split.get("player") or split.get("person") or {}
            raw_date = split.get("date")
            if not raw_date:
                continue
            try:
                d = pd.to_datetime(raw_date, errors="raise", utc=True)
            except Exception:
                continue
            def num(key, default=0.0):
                try:
                    v = stat.get(key)
                    return float(v) if v not in (None, "", "-") else float(default)
                except Exception:
                    return float(default)
            if num("gamesStarted", 0.0) >= 1.0:
                continue
            try:
                txt = str(stat.get("inningsPitched", 0.0)).strip()
                if "." in txt:
                    whole, outs = txt.split(".", 1); oi = int(outs)
                    ip = float(int(whole)) + (oi / 3.0 if oi in (0, 1, 2) else float("nan"))
                else:
                    ip = float(txt)
            except Exception:
                ip = float("nan")
            if not np.isfinite(ip) or ip <= 0:
                continue
            rows.append({
                "date": d, "gamePk": game.get("gamePk"),
                "team_id": int(team_id) if team_id is not None else np.nan,
                "season": int(season) if season is not None else d.year,
                "player_id": person.get("id"), "inningsPitched": float(ip),
                "earnedRuns": num("earnedRuns"), "strikeOuts": num("strikeOuts"),
                "baseOnBalls": num("baseOnBalls"), "homeRuns": num("homeRuns"),
                "hits": num("hits"), "hitBatsmen": num("hitBatsmen"),
                "battersFaced": num("battersFaced"),
                "numberOfPitches": num("numberOfPitches", np.nan),
            })
    return pd.DataFrame(rows)

def summarize_bullpen_before_game(relief_df, game_time, min_relief_ip=12.0):
    """Build a bullpen profile using only relief appearances before target game.

    Features cover underlying skill plus availability/fatigue. Calendar-day cutoffs
    are intentionally conservative: same-day appearances are excluded, which avoids
    accidentally using Game 1 relief work for a Game 1 target. Target doubleheaders
    are removed by the strict research sample anyway.
    """
    if relief_df is None or relief_df.empty:
        return None
    g = relief_df.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce", utc=True)
    target_day = pd.Timestamp(game_time).tz_convert("UTC").normalize()
    g = g[g["date"].dt.normalize() < target_day].copy()
    if g.empty:
        return None

    def sums(df):
        ip = float(pd.to_numeric(df["inningsPitched"], errors="coerce").fillna(0).sum())
        er = float(pd.to_numeric(df["earnedRuns"], errors="coerce").fillna(0).sum())
        so = float(pd.to_numeric(df["strikeOuts"], errors="coerce").fillna(0).sum())
        bb = float(pd.to_numeric(df["baseOnBalls"], errors="coerce").fillna(0).sum())
        hr = float(pd.to_numeric(df["homeRuns"], errors="coerce").fillna(0).sum())
        h = float(pd.to_numeric(df["hits"], errors="coerce").fillna(0).sum())
        pit = pd.to_numeric(df["numberOfPitches"], errors="coerce")
        pitches = float(pit.fillna(0).sum()) if pit.notna().any() else np.nan
        if ip <= 0:
            return {"IP":0.0,"ERA":np.nan,"FIP":np.nan,"KBB9":np.nan,"WHIP":np.nan,"Pitches":pitches}
        return {
            "IP": ip,
            "ERA": 9.0 * er / ip,
            "FIP": (13.0*hr + 3.0*bb - 2.0*so) / ip + 3.20,
            "KBB9": 9.0 * (so - bb) / ip,
            "WHIP": (h + bb) / ip,
            "Pitches": pitches,
        }

    season = sums(g)
    if season["IP"] < float(min_relief_ip):
        return None

    d1 = target_day - pd.Timedelta(days=1)
    d2 = target_day - pd.Timedelta(days=2)
    last1 = g[g["date"].dt.normalize() == d1].copy()
    last3 = g[g["date"].dt.normalize() >= target_day - pd.Timedelta(days=3)].copy()
    last7 = g[g["date"].dt.normalize() >= target_day - pd.Timedelta(days=7)].copy()
    last14 = g[g["date"].dt.normalize() >= target_day - pd.Timedelta(days=14)].copy()
    s1, s3, s7, s14 = sums(last1), sums(last3), sums(last7), sums(last14)

    # Fixed priors prevent early-season extremes from dominating without peeking at
    # the target season's final league averages.
    prior_ip = 25.0
    w = season["IP"] / (season["IP"] + prior_ip)
    shr_fip = w*season["FIP"] + (1-w)*4.20
    shr_era = w*season["ERA"] + (1-w)*4.20
    shr_kbb9 = w*season["KBB9"] + (1-w)*5.10
    shr_whip = w*season["WHIP"] + (1-w)*1.32

    heavy1 = 0
    if not last1.empty:
        p = pd.to_numeric(last1["numberOfPitches"], errors="coerce")
        heavy1 = int((p >= 20).fillna(False).sum())
    day1_arms = set(last1["player_id"].dropna().tolist()) if "player_id" in last1 else set()
    day2_df = g[g["date"].dt.normalize() == d2]
    day2_arms = set(day2_df["player_id"].dropna().tolist()) if "player_id" in day2_df else set()
    b2b = len(day1_arms.intersection(day2_arms))

    return {
        "BP_IP": season["IP"],
        "BP_Shrunk_FIP": shr_fip,
        "BP_Shrunk_ERA": shr_era,
        "BP_Shrunk_KBB9": shr_kbb9,
        "BP_Shrunk_WHIP": shr_whip,
        "BP_Last7_FIP": s7["FIP"],
        "BP_Last7_KBB9": s7["KBB9"],
        "BP_Last14_FIP": s14["FIP"],
        "BP_Last14_KBB9": s14["KBB9"],
        "BP_Last1_IP": s1["IP"],
        "BP_Last3_IP": s3["IP"],
        "BP_Last1_Pitches": s1["Pitches"],
        "BP_Last3_Pitches": s3["Pitches"],
        "BP_Last1_Arms": int(last1["player_id"].nunique()) if not last1.empty else 0,
        "BP_Last3_Arms": int(last3["player_id"].nunique()) if not last3.empty else 0,
        "BP_Heavy20_Last1": heavy1,
        "BP_BackToBack_Arms": int(b2b),
    }


def build_bullpen_feature_table(master_df, min_prior_team_games=12, min_pitcher_starts=3, min_bullpen_ip=12.0):
    """Add point-in-time bullpen quality + availability to the frozen v0.13 starter table."""
    pit = build_pitcher_feature_table(
        master_df,
        min_prior_team_games=min_prior_team_games,
        min_pitcher_starts=min_pitcher_starts,
    )
    if pit.empty:
        return pit, pd.DataFrame()
    need = [c for c in ["Away_Team_ID","Home_Team_ID"] if c not in pit.columns]
    if need:
        raise ValueError("Historical schedule did not supply MLB team IDs needed for bullpen logs: " + ", ".join(need))

    pairs=set()
    for _,r in pit.iterrows():
        try:
            season=int(r["Season"])
        except Exception:
            continue
        for c in ["Away_Team_ID","Home_Team_ID"]:
            try:
                pairs.add((season,int(r[c])))
            except Exception:
                pass
    if not pairs:
        raise ValueError("No team-season pairs were available for bullpen history.")

    status=st.empty(); prog=st.progress(0)
    ledgers={}; errors=[]; cache_hits=0
    pairs_sorted=sorted(pairs)
    max_workers=min(6,max(1,len(pairs_sorted)))
    status.caption(f"Loading roster-based MLB bullpen histories • {len(pairs_sorted)} team-seasons")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut={ex.submit(fetch_team_pitching_gamelog,season,tid):(season,tid) for season,tid in pairs_sorted}
        done=0
        for f in as_completed(fut):
            season,tid=fut[f]
            done+=1
            try:
                payload,err,cached=f.result()
            except Exception as exc:
                payload,err,cached=None,str(exc),False
            if cached: cache_hits+=1
            if err:
                errors.append(err)
            else:
                df=parse_team_reliever_gamelog(payload,team_id=tid,season=season)
                if not df.empty:
                    ledgers[(season,tid)]=df
                else:
                    errors.append(f"Team {tid} {season} returned no roster/player relief rows.")
            prog.progress(min(100,int(done*100/len(pairs_sorted))),text=f"Bullpen histories • {done}/{len(pairs_sorted)}")
    status.empty(); prog.empty()

    if len(ledgers) < max(10, int(len(pairs_sorted)*0.70)):
        sample="; ".join(errors[:3])
        raise ValueError(f"Bullpen roster/player coverage too low ({len(ledgers)}/{len(pairs_sorted)} team-seasons). {sample}")

    rows=[]
    feat_names=None
    for _,r in pit.iterrows():
        try:
            season=int(r["Season"]); aid=int(r["Away_Team_ID"]); hid=int(r["Home_Team_ID"])
            gt=pd.to_datetime(r["Commence_Time"],errors="coerce",utc=True)
        except Exception:
            continue
        if pd.isna(gt):
            continue
        af=summarize_bullpen_before_game(ledgers.get((season,aid)),gt,min_relief_ip=min_bullpen_ip)
        hf=summarize_bullpen_before_game(ledgers.get((season,hid)),gt,min_relief_ip=min_bullpen_ip)
        if af is None or hf is None:
            continue
        rec=r.to_dict()
        feat_names=list(af.keys())
        for k,v in af.items(): rec[f"Away_{k}"]=v
        for k,v in hf.items(): rec[f"Home_{k}"]=v
        rows.append(rec)
    out=pd.DataFrame(rows)
    quality=pd.DataFrame([{
        "Team_Seasons_Requested":len(pairs_sorted),
        "Team_Seasons_Loaded":len(ledgers),
        "Cache_Hits":cache_hits,
        "PIT_Pitcher_Rows":len(pit),
        "PIT_Bullpen_Rows":len(out),
        "Coverage":len(out)/len(pit) if len(pit) else 0.0,
        "Fetch_Errors":len(errors),
    }])
    return out, quality


BULLPEN_FEATURES = [
    "BP_Shrunk_FIP_Diff","BP_Shrunk_ERA_Diff","BP_Shrunk_KBB9_Diff","BP_Shrunk_WHIP_Diff",
    "BP_Last7_FIP_Diff","BP_Last7_KBB9_Diff","BP_Last14_FIP_Diff","BP_Last14_KBB9_Diff",
    "BP_Last1_IP_Diff","BP_Last3_IP_Diff","BP_Last1_Pitches_Diff","BP_Last3_Pitches_Diff",
    "BP_Last1_Arms_Diff","BP_Last3_Arms_Diff","BP_Heavy20_Last1_Diff","BP_BackToBack_Arms_Diff",
]


def pitcher_bullpen_feature_matrix(df):
    x=pitcher_v2_feature_matrix(df).copy()
    # Quality signs: positive = home bullpen advantage when practical.
    x["BP_Shrunk_FIP_Diff"] = df["Away_BP_Shrunk_FIP"] - df["Home_BP_Shrunk_FIP"]
    x["BP_Shrunk_ERA_Diff"] = df["Away_BP_Shrunk_ERA"] - df["Home_BP_Shrunk_ERA"]
    x["BP_Shrunk_KBB9_Diff"] = df["Home_BP_Shrunk_KBB9"] - df["Away_BP_Shrunk_KBB9"]
    x["BP_Shrunk_WHIP_Diff"] = df["Away_BP_Shrunk_WHIP"] - df["Home_BP_Shrunk_WHIP"]
    x["BP_Last7_FIP_Diff"] = df["Away_BP_Last7_FIP"] - df["Home_BP_Last7_FIP"]
    x["BP_Last7_KBB9_Diff"] = df["Home_BP_Last7_KBB9"] - df["Away_BP_Last7_KBB9"]
    x["BP_Last14_FIP_Diff"] = df["Away_BP_Last14_FIP"] - df["Home_BP_Last14_FIP"]
    x["BP_Last14_KBB9_Diff"] = df["Home_BP_Last14_KBB9"] - df["Away_BP_Last14_KBB9"]
    # Fatigue signs: positive = away bullpen worked harder, therefore home advantage.
    x["BP_Last1_IP_Diff"] = df["Away_BP_Last1_IP"] - df["Home_BP_Last1_IP"]
    x["BP_Last3_IP_Diff"] = df["Away_BP_Last3_IP"] - df["Home_BP_Last3_IP"]
    x["BP_Last1_Pitches_Diff"] = df["Away_BP_Last1_Pitches"] - df["Home_BP_Last1_Pitches"]
    x["BP_Last3_Pitches_Diff"] = df["Away_BP_Last3_Pitches"] - df["Home_BP_Last3_Pitches"]
    x["BP_Last1_Arms_Diff"] = df["Away_BP_Last1_Arms"] - df["Home_BP_Last1_Arms"]
    x["BP_Last3_Arms_Diff"] = df["Away_BP_Last3_Arms"] - df["Home_BP_Last3_Arms"]
    x["BP_Heavy20_Last1_Diff"] = df["Away_BP_Heavy20_Last1"] - df["Home_BP_Heavy20_Last1"]
    x["BP_BackToBack_Arms_Diff"] = df["Away_BP_BackToBack_Arms"] - df["Home_BP_BackToBack_Arms"]
    # Recent windows can be empty. Preserve the row and let train-only standardization
    # impute from the development distribution rather than deleting games.
    return x.replace([np.inf,-np.inf],np.nan)


def _fit_feature_model_walkforward_impute(d, feature_builder, label, ridge=6.0):
    """Same frozen walk-forward protocol, with train-only median imputation.

    v0.13 had very few missing engineered starter fields. Bullpen 7/14-day windows
    can legitimately be empty early in a season, so this version learns medians
    from the training/development sample only and applies them forward.
    """
    raw=d.copy().reset_index(drop=True)
    fmat=feature_builder(raw).reset_index(drop=True)
    work=pd.concat([raw,fmat],axis=1); fcols=list(fmat.columns)
    tr=work[work["Season"]==2023].copy(); va=work[work["Season"]==2024].copy(); ho=work[work["Season"]==2025].copy()
    if min(len(tr),len(va),len(ho))<150:
        return {"Model":label,"Error":f"Too little coverage: {len(tr)}/{len(va)}/{len(ho)}"}

    med23=tr[fcols].median(numeric_only=True).reindex(fcols).fillna(0.0)
    Xtr=tr[fcols].fillna(med23).to_numpy(float); Xva=va[fcols].fillna(med23).to_numpy(float)
    ytr=tr["Home_Win"].to_numpy(float); yva=va["Home_Win"].to_numpy(float)
    Xtrz,Xvaz,_,_=_standardize_train_apply(Xtr,Xva)
    b23=fit_ridge_logit(Xtrz,ytr,ridge=ridge); raw24=predict_logit(b23,Xvaz)
    choices=[]
    for w in [0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
        pp=blend_prob(raw24,va["Home_Market_Prob"].to_numpy(float),w)
        choices.append((w,brier_score_binary(pp,yva),log_loss_binary(pp,yva)))
    bw,b24,ll24=sorted(choices,key=lambda z:(z[1],z[2]))[0]; bw=float(bw)

    dev=work[work["Season"].isin([2023,2024])].copy()
    meddev=dev[fcols].median(numeric_only=True).reindex(fcols).fillna(0.0)
    Xdev=dev[fcols].fillna(meddev).to_numpy(float); X25=ho[fcols].fillna(meddev).to_numpy(float)
    ydev=dev["Home_Win"].to_numpy(float); y25=ho["Home_Win"].to_numpy(float)
    Xdz,X25z,_,_=_standardize_train_apply(Xdev,X25)
    bd=fit_ridge_logit(Xdz,ydev,ridge=ridge); raw25=predict_logit(bd,X25z)
    mkt=ho["Home_Market_Prob"].to_numpy(float); cal25=blend_prob(raw25,mkt,bw)
    out=ho.copy(); out["V14_Raw_Prob"]=raw25; out["V14_Cal_Prob"]=cal25
    bets=simulate_ml_bets(out,"V14_Cal_Prob"); bs=summarize_sim_bets(bets)
    coef=pd.DataFrame({"Feature":["Intercept"]+fcols,"Coefficient":bd}); coef["Abs_Coefficient"]=coef["Coefficient"].abs(); coef=coef.sort_values("Abs_Coefficient",ascending=False).reset_index(drop=True)
    return {
        "Model":label,"Games_2025":len(ho),"Model_Weight":bw,
        "Validation_2024_Brier":float(b24),"Validation_2024_LogLoss":float(ll24),
        "Market_Brier":brier_score_binary(mkt,y25),"Raw_Model_Brier":brier_score_binary(raw25,y25),
        "Cal_Brier":brier_score_binary(cal25,y25),"Market_LogLoss":log_loss_binary(mkt,y25),
        "Raw_Model_LogLoss":log_loss_binary(raw25,y25),"Cal_LogLoss":log_loss_binary(cal25,y25),
        "Brier_Improvement":brier_score_binary(mkt,y25)-brier_score_binary(cal25,y25),
        "Bets":bs["Bets"],"Wins":bs["Wins"],"Losses":bs["Losses"],"Units":bs["Units"],"ROI":bs["ROI"],
        "hold25":out,"bets":bets,"coefficients":coef,
    }


def run_pitcher_bullpen_test(master_df,min_prior_games=12,min_pitcher_starts=3,min_bullpen_ip=12.0):
    """Frozen v0.13 champion versus v0.14 Pitcher 2.0 + PIT Bullpen.

    Both are refit on the exact same bullpen-eligible strict rows. 2025 never chooses
    weights or thresholds. Promotion is based on Brier, not backtest ROI.
    """
    pit,quality=build_bullpen_feature_table(master_df,min_prior_team_games=min_prior_games,min_pitcher_starts=min_pitcher_starts,min_bullpen_ip=min_bullpen_ip)
    if pit.empty:
        raise ValueError("No point-in-time pitcher + bullpen rows were built.")
    strict=_integrity_prepare_subset(pit,max_hours=6,min_starts=min_pitcher_starts)
    if strict.empty:
        raise ValueError("No eligible ≤6h non-doubleheader pitcher + bullpen rows were available.")

    # Frozen champion is rerun on the identical rows. Use the same v0.13 fitter/recipe.
    base=_fit_feature_model_walkforward(strict,pitcher_v2_feature_matrix,"v0.13 frozen Pitcher 2.0",ridge=5.0)
    v14=_fit_feature_model_walkforward_impute(strict,pitcher_bullpen_feature_matrix,"v0.14 Pitcher 2.0 + Bullpen",ridge=6.0)
    results=[base,v14]
    table=pd.DataFrame([{k:v for k,v in r.items() if k not in ("hold25","bets","coefficients")} for r in results])

    seg=[]
    if not v14.get("Error") and isinstance(v14.get("bets"),pd.DataFrame) and not v14["bets"].empty:
        x=v14["bets"].copy(); x["Commence_Time"]=pd.to_datetime(x["Commence_Time"],errors="coerce",utc=True); x["Month"]=x["Commence_Time"].dt.strftime("%Y-%m")
        for name,g in x.groupby("Month"):
            seg.append({"Segment":name,**summarize_sim_bets(g)})
        for name,g in [("Favorites",x[x["Odds"]<0]),("Underdogs",x[x["Odds"]>0])]:
            seg.append({"Segment":name,**summarize_sim_bets(g)})

    return {
        "pit":pit,"strict_rows":len(strict),"quality":quality,"table":table,
        "results":{r.get("Model"):r for r in results},"segments":pd.DataFrame(seg),
        "coefficients":v14.get("coefficients",pd.DataFrame()),
    }

def run_pitcher_integrity_test(master_df,min_prior_games=12,min_pitcher_starts=3):
    pit=build_pitcher_feature_table(master_df,min_prior_team_games=min_prior_games,min_pitcher_starts=min_pitcher_starts)
    if pit.empty:
        raise ValueError("No PIT pitcher rows were built.")
    strict=_integrity_prepare_subset(pit,max_hours=6,min_starts=min_pitcher_starts)
    established=_integrity_prepare_subset(pit,max_hours=6,min_starts=min_pitcher_starts,established_starts=max(5,min_pitcher_starts))
    trimmed=_integrity_prepare_subset(pit,max_hours=6,min_starts=min_pitcher_starts,trim_extremes=True)
    specs=[
        (strict,"correct",False,"Correct starters • ≤6h"),
        (strict,"team_only",False,"Team-only placebo • ≤6h"),
        (strict,"scramble",False,"Scrambled starters placebo • ≤6h"),
        (strict,"swap",False,"Swapped starters placebo • ≤6h"),
        (established,"correct",False,"Established starters ≥5 • ≤6h"),
        (trimmed,"correct",False,"Trim extreme SP mismatches • ≤6h"),
        (strict,"correct",True,"Probability cap 20–80% • ≤6h"),
    ]
    results=[]; full={}
    for d,mode,cap,label in specs:
        r=_integrity_fit_eval(d,mode=mode,seed=122,cap_probs=cap,label=label)
        full[label]=r; results.append({k:v for k,v in r.items() if k not in ("hold25","bets")})
    seg=[]; base=full.get("Correct starters • ≤6h",{}); bets=base.get("bets") if isinstance(base,dict) else None
    if bets is not None and not bets.empty:
        x=bets.copy(); x["Commence_Time"]=pd.to_datetime(x["Commence_Time"],errors="coerce",utc=True); x["Month"]=x["Commence_Time"].dt.strftime("%Y-%m")
        for name,g in x.groupby("Month"):
            seg.append({"Segment":name,**summarize_sim_bets(g)})
        for name,g in [("Favorites",x[x["Odds"]<0]),("Underdogs",x[x["Odds"]>0])]:
            seg.append({"Segment":name,**summarize_sim_bets(g)})
    return {"pit":pit,"table":pd.DataFrame(results),"results":full,"segments":pd.DataFrame(seg)}

def user_verdict(verdict):
    """User-facing betting label. Letter grades stay internal only."""
    return {
        "STRONG BET": "BEST BET",
        "BET": "BET",
        "LEAN": "LEAN",
        "PASS": "PASS",
        "NO LINE": "NO LINE",
    }.get(str(verdict), "PASS")


def verdict_class(verdict):
    return {
        "STRONG BET": "best",
        "BET": "bet",
        "LEAN": "lean",
        "PASS": "pass",
        "NO LINE": "pass",
    }.get(str(verdict), "pass")


def verdict_rank(verdict):
    return {
        "STRONG BET": 4,
        "BET": 3,
        "LEAN": 2,
        "PASS": 1,
        "NO LINE": 0,
    }.get(str(verdict), 0)


def market_rank_score(candidate, confidence):
    """Practical ranking: grade first, then win chance, edge, EV and confidence."""
    _, grank, _ = grade_meta(candidate.get("verdict"))
    p = float(candidate.get("calibrated_prob", 0.5))
    e = float(candidate.get("edge", 0.0))
    v = float(candidate.get("ev", 0.0))
    c = float(confidence)

    # Small reliability haircut for lower-confidence games.
    p_adj = max(0.0, p - max(0.0, 75.0 - c) * .0015)
    return grank * 100.0 + p_adj * 45.0 + e * 32.0 + v * 12.0 + c * .08


def evaluate_game_markets(r, market):
    """Build and rank all six full-game markets for one MLB matchup."""
    conf = int(r["Model_Confidence"])
    rows = []

    if not market:
        return rows

    # ---------- Moneyline ----------
    if market.get("away_ml") is not None and market.get("home_ml") is not None:
        away_mkt, home_mkt = no_vig_pair(market["away_ml"], market["home_ml"])
        if away_mkt is None or home_mkt is None:
            away_mkt, home_mkt = None, None
        if away_mkt is not None and home_mkt is not None:
            for side, raw_prob, odds, mkt_prob, team in [
                ("away", float(r["Away_WinProb"]), int(market["away_ml"]), away_mkt, r["Away"]),
                ("home", float(r["Home_WinProb"]), int(market["home_ml"]), home_mkt, r["Home"]),
            ]:
                cal = calibrated_probability(raw_prob, mkt_prob, "moneyline", conf)
                verdict, edge, ev, imp = decision_grade(cal, odds, conf, "moneyline")
                rows.append({
                    "verdict": verdict,
                    "market": f"{team} ML",
                    "market_type": "MONEYLINE",
                    "odds": odds,
                    "raw_prob": raw_prob,
                    "market_prob": mkt_prob,
                    "calibrated_prob": cal,
                    "implied_prob": imp,
                    "edge": edge,
                    "ev": ev,
                    "fair": fair_ml(cal),
                })

    # ---------- Run line ----------
    if (
        market.get("away_rl") is not None and market.get("away_rl_odds") is not None
        and market.get("home_rl") is not None and market.get("home_rl_odds") is not None
        and abs(abs(float(market["away_rl"])) - 1.5) < 1e-8
        and abs(abs(float(market["home_rl"])) - 1.5) < 1e-8
    ):
        away_mkt, home_mkt = no_vig_pair(market["away_rl_odds"], market["home_rl_odds"])
        for side, point, odds, mkt_prob in [
            ("away", float(market["away_rl"]), int(market["away_rl_odds"]), away_mkt),
            ("home", float(market["home_rl"]), int(market["home_rl_odds"]), home_mkt),
        ]:
            prefix = "Away" if side == "away" else "Home"
            team = r["Away"] if side == "away" else r["Home"]
            prob_col = f"{prefix}_{'+' if point > 0 else '-'}1.5_Prob"
            if prob_col in r.index:
                raw_prob = float(r[prob_col])
                cal = calibrated_probability(raw_prob, mkt_prob, "runline", conf)
                verdict, edge, ev, imp = decision_grade(cal, odds, conf, "runline")
                rows.append({
                    "verdict": verdict,
                    "market": f"{team} {point:+g}",
                    "market_type": "RUN LINE",
                    "odds": odds,
                    "raw_prob": raw_prob,
                    "market_prob": mkt_prob,
                    "calibrated_prob": cal,
                    "implied_prob": imp,
                    "edge": edge,
                    "ev": ev,
                    "fair": fair_ml(cal),
                })

    # ---------- Total ----------
    if (
        market.get("total") is not None
        and market.get("over_odds") is not None
        and market.get("under_odds") is not None
    ):
        total_line = float(market["total"])
        over_mkt, under_mkt = no_vig_pair(market["over_odds"], market["under_odds"])

        for side, odds, mkt_prob in [
            ("Over", int(market["over_odds"]), over_mkt),
            ("Under", int(market["under_odds"]), under_mkt),
        ]:
            probs = total_market_probs(float(r["Model_Total"]), total_line, side)
            decisive = probs["win"] + probs["loss"]
            raw_prob = probs["win"] / decisive if decisive > 0 else 0.5
            cal = calibrated_probability(raw_prob, mkt_prob, "total", conf)
            verdict, edge, ev, imp = decision_grade(cal, odds, conf, "total")
            rows.append({
                "verdict": verdict,
                "market": f"{side} {total_line:g}",
                "market_type": "TOTAL",
                "odds": odds,
                "raw_prob": raw_prob,
                "market_prob": mkt_prob,
                "calibrated_prob": cal,
                "implied_prob": imp,
                "edge": edge,
                "ev": ev,
                "fair": fair_ml(cal),
                "push_prob": probs.get("push", 0.0),
            })

    for x in rows:
        x["rank_score"] = market_rank_score(x, conf)

    return sorted(
        rows,
        key=lambda x: (grade_meta(x["verdict"])[1], x["rank_score"]),
        reverse=True,
    )


def mlb_team_logo(team_name):
    """Stable MLB team logo URLs; falls back to initials if a mapping is unavailable."""
    ids = {
        "Arizona Diamondbacks": 109, "Atlanta Braves": 144, "Baltimore Orioles": 110,
        "Boston Red Sox": 111, "Chicago Cubs": 112, "Chicago White Sox": 145,
        "Cincinnati Reds": 113, "Cleveland Guardians": 114, "Colorado Rockies": 115,
        "Detroit Tigers": 116, "Houston Astros": 117, "Kansas City Royals": 118,
        "Los Angeles Angels": 108, "Los Angeles Dodgers": 119, "Miami Marlins": 146,
        "Milwaukee Brewers": 158, "Minnesota Twins": 142, "New York Mets": 121,
        "New York Yankees": 147, "Athletics": 133, "Oakland Athletics": 133,
        "Philadelphia Phillies": 143, "Pittsburgh Pirates": 134, "San Diego Padres": 135,
        "San Francisco Giants": 137, "Seattle Mariners": 136, "St. Louis Cardinals": 138,
        "Tampa Bay Rays": 139, "Texas Rangers": 140, "Toronto Blue Jays": 141,
        "Washington Nationals": 120,
    }
    tid = ids.get(str(team_name))
    return f"https://www.mlbstatic.com/team-logos/{tid}.svg" if tid else ""


def logo_html(team, size=32):
    url = mlb_team_logo(team)
    initials = "".join(x[0] for x in str(team).split()[:2]).upper()
    if url:
        return f'<img class="team-logo" src="{html.escape(url)}" alt="{html.escape(str(team))}" style="width:{size}px;height:{size}px;">'
    return f'<div class="team-logo-fallback" style="width:{size}px;height:{size}px;">{html.escape(initials)}</div>'

def icon(v):
    return "🟢" if v in ["BET", "STRONG BET"] else "🟡" if v == "LEAN" else "⚪"


def add_market(rows, name, prob, odds, conf):
    verdict, edge, ev, imp = bet_grade(prob, odds, conf)
    rows.append({
        "Bet": name, "Verdict": verdict, "Odds": int(odds),
        "Model Prob": prob, "Implied Prob": imp, "Edge": edge,
        "EV": ev, "Model Fair": fair_ml(prob),
    })



def total_market_probs(model_total, line, side):
    """Experimental Poisson total distribution centered on the model total."""
    lam = max(0.10, float(model_total))
    line = float(line)
    is_integer = abs(line - round(line)) < 1e-9

    if is_integer:
        n = int(round(line))
        p_push = float(poisson.pmf(n, lam))
        if side == "Over":
            p_win = float(1.0 - poisson.cdf(n, lam))
            p_loss = float(poisson.cdf(n - 1, lam))
        else:
            p_win = float(poisson.cdf(n - 1, lam))
            p_loss = float(1.0 - poisson.cdf(n, lam))
    else:
        n = int(line // 1)
        p_push = 0.0
        if side == "Over":
            p_win = float(1.0 - poisson.cdf(n, lam))
            p_loss = float(poisson.cdf(n, lam))
        else:
            p_win = float(poisson.cdf(n, lam))
            p_loss = float(1.0 - poisson.cdf(n, lam))

    return {"win": p_win, "push": p_push, "loss": p_loss}

def total_expected_value(probs, odds):
    odds = float(odds)
    profit = odds / 100.0 if odds > 0 else 100.0 / abs(odds)
    return probs["win"] * profit - probs["loss"]

def total_bet_grade(model_total, line, side, odds, confidence):
    probs = total_market_probs(model_total, line, side)
    decisive = probs["win"] + probs["loss"]
    conditional_win = probs["win"] / decisive if decisive > 0 else 0.5
    imp = implied_prob(odds)
    edge = conditional_win - imp
    ev = total_expected_value(probs, odds)
    t = juice_thresholds(odds)

    if confidence >= 80 and edge >= t["strong_edge"] and ev >= t["strong_ev"]:
        verdict = "STRONG BET"
    elif confidence >= 70 and edge >= t["bet_edge"] and ev >= t["bet_ev"]:
        verdict = "BET"
    elif edge > 0 and ev > 0:
        verdict = "LEAN"
    else:
        verdict = "PASS"

    return verdict, edge, ev, imp, conditional_win, probs

st.markdown(
    f"""
    <div class="edge-hero">
      <div class="edge-kicker">MLB BETTING MODEL</div>
      <div class="edge-title">MLB Edge</div>
      <div class="edge-subtitle">Pick a matchup or run the full slate, load the current market, and get a ranked betting board in seconds.</div>
      <div class="version-pill">{APP_VERSION} • Engine {MODEL_VERSION}</div>
    </div>
    <div class="status-strip">
      <div class="status-live"><span class="status-dot"></span> Live model ready</div>
      <div>Best Bet &nbsp;•&nbsp; Bet &nbsp;•&nbsp; Lean &nbsp;•&nbsp; Pass</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "games" not in st.session_state:
    with st.spinner("Loading today's MLB schedule..."):
        st.session_state.games = fetch_today_games()

if st.button("↻ Refresh Games + Market"):
    with st.spinner("Refreshing schedule and general market..."):
        st.session_state.games = fetch_today_games()
        fetch_general_mlb_odds.clear()
        st.session_state.general_odds_payload = fetch_general_mlb_odds(get_odds_api_key())
    for k in ["last_results"]:
        st.session_state.pop(k, None)
    st.rerun()

games = st.session_state.games
if not games:
    st.warning("No MLB games were found for today.")
    st.stop()

odds_api_key = get_odds_api_key()
if "general_odds_payload" not in st.session_state:
    st.session_state.general_odds_payload = fetch_general_mlb_odds(odds_api_key)

labels = {}
for g in games:
    away_sp = g["Away_SP"] or "TBD"
    home_sp = g["Home_SP"] or "TBD"
    time_text = f" — {g['TimeLabel']} ET" if g["TimeLabel"] else ""
    labels[f"{g['Away']} @ {g['Home']}{time_text} | {away_sp} vs {home_sp}"] = g["GamePk"]

mode = st.radio("Betting Board", ["Single Game", "Full Slate", "Backtest Lab"], horizontal=True, label_visibility="collapsed")

if mode == "Backtest Lab":
    st.markdown(
        """
        <div class="app-head">
          <div>
            <div class="app-eyebrow">MLB EDGE</div>
            <div class="app-head-title">Backtest Lab</div>
            <div class="app-head-sub">Test the production betting logic without consuming live Odds API credits.</div>
          </div>
          <div class="app-live">LOCAL TEST</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bt-note">
          <b>Backtest calculations make zero Odds API calls.</b>
          The Historical Data Builder below is the only part of this tab that can use paid historical credits,
          and it cannot run until you explicitly confirm the spend and press the build button.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Historical CSV format", expanded=False):
        st.caption(
            "Required: Date, Game, Market_Type, Bet, Odds, Result, Raw_Model_Prob, "
            "Market_NoVig_Prob, Calibrated_Prob, Edge, EV, Verdict, Confidence."
        )
        sample = pd.DataFrame([{
            "Date":"2025-08-01",
            "Game":"Away Team @ Home Team",
            "Market_Type":"MONEYLINE",
            "Bet":"Home Team ML",
            "Odds":-120,
            "Result":"WIN",
            "Raw_Model_Prob":0.57,
            "Market_NoVig_Prob":0.54,
            "Calibrated_Prob":0.55,
            "Edge":0.045,
            "EV":0.083,
            "Verdict":"BET",
            "Confidence":76,
        }])
        st.dataframe(sample, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV template",
            sample.to_csv(index=False).encode("utf-8"),
            file_name="mlb_backtest_template.csv",
            mime="text/csv",
        )


    st.markdown('<div class="section-kicker">HISTORICAL DATA BUILDER</div>', unsafe_allow_html=True)
    st.caption(
        "The builder first finds actual MLB regular-season game dates from MLB's free schedule service. "
        "That lookup uses zero Odds API credits."
    )

    with st.expander("Build historical market dataset", expanded=True):
        hc1, hc2 = st.columns(2)
        with hc1:
            hist_start = st.date_input(
                "Historical start",
                value=date(2023,3,30),
                min_value=date(2020,6,6),
                max_value=date.today(),
                key="hist_start"
            )
        with hc2:
            hist_end = st.date_input(
                "Historical end",
                value=date(2025,9,28),
                min_value=date(2020,6,6),
                max_value=date.today(),
                key="hist_end"
            )

        with st.spinner("Finding actual MLB game dates by season…"):
            mlb_game_dates, season_date_counts, schedule_err = fetch_mlb_regular_season_dates(
                hist_start, hist_end
            )

        if schedule_err:
            st.error(schedule_err)
            mlb_game_dates = []
            season_date_counts = {}
        elif mlb_game_dates:
            st.success(
                f"Found {len(mlb_game_dates):,} actual MLB regular-season game dates. "
                "Offseason and no-game dates will be skipped."
            )

            if season_date_counts:
                season_cols = st.columns(min(3, len(season_date_counts)))
                for i, (season_year, season_count) in enumerate(sorted(season_date_counts.items())):
                    with season_cols[i % len(season_cols)]:
                        st.metric(str(season_year), f"{season_count:,} game dates")

                suspicious = [
                    (yr, cnt) for yr, cnt in season_date_counts.items()
                    if cnt < 150 and date(yr, 4, 1) >= hist_start and date(yr, 9, 1) <= hist_end
                ]
                if suspicious:
                    st.warning(
                        "One or more full-season schedule counts look unusually low. "
                        "Do not run paid historical downloads until the counts are reviewed."
                    )
        else:
            st.warning("No MLB regular-season game dates were found in this range.")

        hist_market_labels = st.multiselect(
            "Historical markets",
            ["Moneyline","Run Line","Total"],
            default=["Moneyline"],
            key="hist_market_labels",
            help="Recommended: build Moneyline first. Add Run Line and Total later only if budget remains."
        )
        market_map = {"Moneyline":"h2h","Run Line":"spreads","Total":"totals"}
        hist_markets = [market_map[x] for x in hist_market_labels]

        if hist_end < hist_start:
            st.error("Historical end date must be on or after the start date.")
            hist_est = {"snapshots":0,"per_snapshot_max":0,"max_credits":0}
        elif not hist_markets:
            st.warning("Select at least one market.")
            hist_est = {"snapshots":0,"per_snapshot_max":0,"max_credits":0}
        else:
            hist_est = historical_credit_estimate(mlb_game_dates, hist_markets)

        phase_plan = phase_credit_plan(mlb_game_dates)
        cached_dates = cached_history_manifest(hist_markets) if hist_markets else set()
        selected_dates = set(mlb_game_dates)
        cached_in_range = len(cached_dates & selected_dates)
        remaining_snapshots = max(0, hist_est["snapshots"] - cached_in_range)
        remaining_credit_ceiling = remaining_snapshots * hist_est["per_snapshot_max"]

        c1, c2, c3 = st.columns(3)
        c1.metric("MLB game dates", f"{hist_est['snapshots']:,}")
        c2.metric("Max / new snapshot", f"{hist_est['per_snapshot_max']:,}")
        c3.metric("Remaining ceiling", f"{remaining_credit_ceiling:,}")

        st.markdown(
            f"""
            <div class="phase-grid">
              <div class="phase-cell recommended">
                <span>Phase 1</span><b>Moneyline</b><em>≤ {phase_plan['moneyline']:,} credits</em>
              </div>
              <div class="phase-cell">
                <span>Phase 2</span><b>Run Line</b><em>+ ≤ {phase_plan['runline']:,}</em>
              </div>
              <div class="phase-cell">
                <span>Phase 3</span><b>Total</b><em>+ ≤ {phase_plan['total']:,}</em>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"All three markets together have a conservative ceiling of {phase_plan['all_three']:,} credits. "
            "Moneyline-first is the safest way to stay inside the 20K plan."
        )
        st.caption(
            f"{cached_in_range:,} selected MLB game-date snapshots are already cached and will be skipped."
        )

        suggested_cap = min(
            20000,
            max(10, remaining_credit_ceiling if remaining_credit_ceiling else 10)
        )
        hard_cap = st.number_input(
            "Hard credit cap for this run",
            min_value=10,
            max_value=100000,
            value=suggested_cap,
            step=10,
            key="history_hard_cap"
        )

        if remaining_credit_ceiling > 20000:
            st.warning(
                "This selection can exceed a 20K plan. Reduce the date range or markets before running it."
            )
        elif remaining_credit_ceiling > 0:
            st.info(
                f"This selection fits under a 20K conservative ceiling with "
                f"{20000 - remaining_credit_ceiling:,} credits of headroom."
            )

        confirm_history = st.checkbox(
            "I understand this button can consume paid Historical Odds credits.",
            key="confirm_history_spend"
        )

        st.info(
            "Credit guard: only actual MLB regular-season game dates are requested; cached dates are skipped; "
            "one league-wide snapshot is requested per game date; successful responses are cached immediately; "
            "the run stops before the hard cap is exceeded. Snapshot time is 15:00 UTC."
        )

        restore_zip = st.file_uploader(
            "Restore prior historical cache (optional ZIP)",
            type=["zip"],
            key="history_cache_restore"
        )
        if restore_zip is not None and st.button("Restore cache ZIP", key="restore_history_zip"):
            try:
                restored_n = restore_cache_zip(restore_zip)
                st.success(f"Restored {restored_n:,} cached snapshot files.")
            except Exception as ex:
                st.error(f"Could not restore cache ZIP: {ex}")

        full_season_years = [
            yr for yr in range(hist_start.year, hist_end.year + 1)
            if hist_start <= date(yr, 4, 1) and hist_end >= date(yr, 9, 1)
        ]
        schedule_sanity_ok = all(
            season_date_counts.get(yr, 0) >= 150 for yr in full_season_years
        )

        if full_season_years and not schedule_sanity_ok:
            st.error(
                "Schedule sanity check failed. Paid historical downloading is disabled "
                "until each included full MLB season has at least 150 distinct game dates."
            )

        run_history = st.button(
            "Build Historical Market Dataset",
            type="primary",
            disabled=(
                not confirm_history
                or hist_end < hist_start
                or not hist_markets
                or not mlb_game_dates
                or remaining_snapshots == 0
                or not schedule_sanity_ok
            ),
            key="run_history_builder"
        )

        if remaining_snapshots == 0 and hist_est["snapshots"] > 0:
            st.success("Every selected MLB game-date snapshot is already cached. No Odds API call is needed.")

        if run_history:
            dates_to_fetch = [d for d in mlb_game_dates if d not in cached_dates]
            actual_last = None
            remaining_hdr = None
            estimated_spend = 0
            stopped_reason = None

            progress = st.progress(0)
            status = st.empty()
            total_new = len(dates_to_fetch)

            for i, d in enumerate(dates_to_fetch, start=1):
                if estimated_spend + hist_est["per_snapshot_max"] > hard_cap:
                    stopped_reason = f"Stopped before exceeding the hard credit cap ({hard_cap:,})."
                    break

                status.caption(
                    f"Fetching MLB game date {d.isoformat()} • {i:,} of {total_new:,} new snapshots"
                )
                payload, meta, err = fetch_historical_snapshot(d, hist_markets, force=False)

                if err:
                    stopped_reason = f"Stopped on {d.isoformat()}: {err}"
                    break

                estimated_spend += hist_est["per_snapshot_max"]
                actual_last = meta.get("last")
                remaining_hdr = meta.get("remaining")
                progress.progress(min(i / max(1,total_new), 1.0))

            all_rows = []
            for d in mlb_game_dates:
                cf = _history_cache_file(d, hist_markets)
                if not cf.exists():
                    continue
                try:
                    all_rows.extend(
                        flatten_historical_snapshot(json.loads(cf.read_text()), d)
                    )
                except Exception:
                    continue

            progress.empty()
            status.empty()

            if stopped_reason:
                st.warning(stopped_reason)

            if all_rows:
                hist_df = pd.DataFrame(all_rows)
                hist_df["Commence_Time"] = pd.to_datetime(
                    hist_df["Commence_Time"], errors="coerce", utc=True
                )
                hist_df = (
                    hist_df
                    .sort_values(["Snapshot_Date","Commence_Time","Away_Team","Home_Team"])
                    .drop_duplicates(["Snapshot_Date","Event_ID"], keep="last")
                )

                st.success(
                    f"Historical market dataset ready: {len(hist_df):,} game rows across "
                    f"{hist_df['Snapshot_Date'].nunique():,} cached MLB game dates."
                )
                if remaining_hdr is not None:
                    st.caption(
                        f"The Odds API reports {remaining_hdr} credits remaining. "
                        f"Last-request cost: {actual_last if actual_last is not None else '—'}."
                    )

                st.dataframe(hist_df.head(100), use_container_width=True, hide_index=True)

                st.download_button(
                    "Download Historical Market CSV",
                    hist_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"mlb_historical_market_{hist_start}_{hist_end}.csv",
                    mime="text/csv",
                    key="download_history_csv"
                )
                st.download_button(
                    "Download Cache ZIP",
                    build_cache_zip(),
                    file_name=f"mlb_history_cache_{hist_start}_{hist_end}.zip",
                    mime="application/zip",
                    key="download_history_cache"
                )
            else:
                st.info("No cached historical game rows are available yet.")

    st.divider()

    st.markdown('<div class="section-kicker">MONEYLINE MASTER DATASET</div>', unsafe_allow_html=True)
    st.caption(
        "Turn the paid historical Moneyline export into a clean, result-matched master dataset. "
        "MLB final scores come from MLB's free Stats API, so this step uses zero Odds API credits."
    )

    ml_history_upload = st.file_uploader(
        "Upload Historical Market CSV",
        type=["csv"],
        key="moneyline_master_upload",
        help="Use the Moneyline CSV downloaded from Historical Data Builder."
    )

    if ml_history_upload is not None:
        try:
            ml_raw = pd.read_csv(ml_history_upload)
            ml_clean, clean_stats = clean_historical_moneyline_df(
                ml_raw, max_hours_to_first_pitch=18.0
            )

            if clean_stats.get("missing"):
                st.error("Missing columns: " + ", ".join(clean_stats["missing"]))
            elif ml_clean is None or ml_clean.empty:
                st.error("No usable Moneyline rows remained after cleaning.")
            else:
                st.markdown(
                    f"""
                    <div class="bt-metrics">
                      <div class="bt-metric"><span>Raw Rows</span><b>{clean_stats['raw_rows']:,}</b></div>
                      <div class="bt-metric"><span>Usable Games</span><b>{clean_stats['clean_rows']:,}</b></div>
                      <div class="bt-metric"><span>Invalid Away ML</span><b>{clean_stats['invalid_away_ml']:,}</b></div>
                      <div class="bt-metric"><span>Invalid Home ML</span><b>{clean_stats['invalid_home_ml']:,}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    "Cleaning rule: valid two-way American prices and 0–18 hours before first pitch, "
                    "then one row per historical event."
                )

                seasons = sorted(
                    int(x) for x in ml_clean["Season"].dropna().unique().tolist()
                )

                if st.button(
                    "Build Moneyline Master Dataset",
                    type="primary",
                    key="build_moneyline_master"
                ):
                    result_rows = []
                    result_errors = []
                    cache_notes = []

                    prog = st.progress(0)
                    stat = st.empty()

                    for i, season in enumerate(seasons, start=1):
                        stat.caption(f"Getting final MLB results for {season}…")
                        payload, err, cached = fetch_mlb_results_season(season)
                        if err:
                            result_errors.append(err)
                            continue
                        result_rows.extend(flatten_mlb_results(payload))
                        cache_notes.append(f"{season}: {'cached' if cached else 'downloaded free'}")
                        prog.progress(i / max(1, len(seasons)))

                    prog.empty()
                    stat.empty()

                    if result_errors:
                        for err in result_errors:
                            st.warning(err)

                    if not result_rows:
                        st.error("No MLB final results were available.")
                    else:
                        results_df = pd.DataFrame(result_rows)
                        master_df = build_moneyline_master(ml_clean, results_df)

                        matched = int(master_df["Result_Matched"].sum())
                        total = len(master_df)
                        match_rate = matched / total if total else 0.0

                        st.session_state["moneyline_master_df"] = master_df

                        st.success(
                            f"Moneyline Master ready: {matched:,} of {total:,} games matched "
                            f"to final MLB results ({match_rate*100:.1f}%)."
                        )
                        st.caption(" • ".join(cache_notes))

                        if match_rate < .985:
                            st.warning(
                                "Result match rate is below 98.5%. Review unmatched games before using this for model validation."
                            )

        except Exception as ex:
            st.error(f"Could not process Historical Market CSV: {ex}")

    if "moneyline_master_df" in st.session_state:
        master_df = st.session_state["moneyline_master_df"].copy()
        matched_df = master_df[master_df["Result_Matched"]].copy()

        if not matched_df.empty:
            st.markdown('<div class="section-kicker">MARKET BASELINE</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            fav_win = matched_df["Favorite_Won"].mean()
            brier = moneyline_market_brier(matched_df)
            with m1:
                st.metric("Matched games", f"{len(matched_df):,}")
            with m2:
                st.metric("Market favorite win %", f"{fav_win*100:.1f}%")
            with m3:
                st.metric("Market Brier", f"{brier:.4f}" if brier is not None else "—")
            with m4:
                med_books = pd.to_numeric(
                    matched_df.get("ML_Books"), errors="coerce"
                ).median()
                st.metric("Median books", f"{med_books:.0f}" if pd.notna(med_books) else "—")

            cal = market_calibration_summary(matched_df)
            if not cal.empty:
                st.dataframe(cal, use_container_width=True, hide_index=True)

            by_season = (
                matched_df.groupby("Season")
                .agg(
                    Games=("Event_ID","count"),
                    Favorite_Win_Pct=("Favorite_Won","mean"),
                    Avg_Favorite_Prob=("Favorite_Prob","mean"),
                    Median_Books=("ML_Books","median")
                )
                .reset_index()
            )
            by_season["Favorite Win %"] = (by_season["Favorite_Win_Pct"]*100).round(1)
            by_season["Avg Favorite %"] = (by_season["Avg_Favorite_Prob"]*100).round(1)
            by_season = by_season[
                ["Season","Games","Favorite Win %","Avg Favorite %","Median_Books"]
            ]
            st.dataframe(by_season, use_container_width=True, hide_index=True)

        with st.expander("Preview / download Moneyline Master", expanded=False):
            st.dataframe(master_df.head(150), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Moneyline Master CSV",
                master_df.to_csv(index=False).encode("utf-8"),
                file_name="mlb_moneyline_master_2023_2025.csv",
                mime="text/csv",
                key="download_moneyline_master"
            )

    st.divider()

    st.markdown('<div class="section-kicker">POINT-IN-TIME MONEYLINE TEST</div>', unsafe_allow_html=True)
    st.caption(
        "This test uses the Moneyline Master file and creates a separate historical PIT model from information "
        "available before each game. It does not pretend the current live engine had identical historical inputs."
    )

    pit_upload = st.file_uploader(
        "Upload Moneyline Master CSV",
        type=["csv"],
        key="pit_master_upload",
        help="Use mlb_moneyline_master_2023_2025.csv from the previous step."
    )

    if pit_upload is not None:
        try:
            pit_master = pd.read_csv(pit_upload)
            required = [
                "Event_ID","Season","Commence_Time","Away_Team","Home_Team",
                "Away_ML","Home_ML","Away_Market_Prob","Home_Market_Prob",
                "Away_Score","Home_Score","Away_Win","Home_Win","Result_Matched"
            ]
            missing = [c for c in required if c not in pit_master.columns]

            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                matched_n = int(pit_master["Result_Matched"].astype(bool).sum())
                st.info(
                    f"Loaded {len(pit_master):,} master rows • {matched_n:,} matched final results."
                )

                min_prior = st.slider(
                    "Minimum prior games per team",
                    min_value=5,
                    max_value=25,
                    value=12,
                    step=1,
                    key="pit_min_prior",
                    help="A game is eligible only after both teams have this many earlier games in the historical ledger."
                )

                if st.button(
                    "Run Point-in-Time Moneyline Backtest",
                    type="primary",
                    key="run_pit_backtest"
                ):
                    with st.spinner("Building lagged team states, fitting 2023, validating 2024, and testing untouched 2025…"):
                        pit_result = run_point_in_time_backtest(
                            pit_master,
                            min_prior_games=min_prior
                        )
                    st.session_state["pit_result"] = pit_result

        except Exception as ex:
            st.error(f"Could not run point-in-time backtest: {ex}")

    if "pit_result" in st.session_state:
        r = st.session_state["pit_result"]
        metrics = r["metrics"].copy()
        best_weight = r["best_weight"]
        raw_sum = r["hold_raw_summary"]
        cal_sum = r["hold_cal_summary"]

        st.success(
            f"Walk-forward test complete. 2024 selected a {best_weight*100:.0f}% PIT-model / "
            f"{(1-best_weight)*100:.0f}% market blend. 2025 was not used to choose that weight."
        )

        st.markdown('<div class="section-kicker">2025 HOLDOUT</div>', unsafe_allow_html=True)

        hold = r["hold25"]
        market_brier = brier_score_binary(hold["Home_Market_Prob"], hold["Home_Win"])
        raw_brier = brier_score_binary(hold["PIT_Raw_Prob"], hold["Home_Win"])
        cal_brier = brier_score_binary(hold["PIT_Calibrated_Prob"], hold["Home_Win"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Holdout games", f"{len(hold):,}")
        with c2:
            st.metric("Market Brier", f"{market_brier:.4f}")
        with c3:
            st.metric("Raw PIT Brier", f"{raw_brier:.4f}",
                      delta=f"{market_brier-raw_brier:+.4f} vs market")
        with c4:
            st.metric("Calibrated Brier", f"{cal_brier:.4f}",
                      delta=f"{market_brier-cal_brier:+.4f} vs market")

        st.dataframe(
            metrics.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.markdown('<div class="section-kicker">2025 BET SIMULATION</div>', unsafe_allow_html=True)
        st.caption(
            "Default test thresholds: edge ≥2.5%, EV ≥4.5%, no +300 or longer dogs, one moneyline bet max per game."
        )

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Raw PIT model**")
            st.metric("ROI", f"{raw_sum['ROI']*100:+.1f}%")
            st.caption(
                f"{raw_sum['Wins']}-{raw_sum['Losses']} • "
                f"{raw_sum['Units']:+.2f}u • {raw_sum['Bets']} bets"
            )
        with b2:
            st.markdown("**Market-calibrated PIT model**")
            st.metric("ROI", f"{cal_sum['ROI']*100:+.1f}%")
            st.caption(
                f"{cal_sum['Wins']}-{cal_sum['Losses']} • "
                f"{cal_sum['Units']:+.2f}u • {cal_sum['Bets']} bets"
            )

        st.markdown('<div class="section-kicker">2024 BLEND SELECTION</div>', unsafe_allow_html=True)
        blend_show = r["blend_table"].copy()
        blend_show["Model Weight %"] = (blend_show["Model_Weight"]*100).round(0).astype(int)
        blend_show = blend_show[["Model Weight %","Brier_2024","LogLoss_2024"]]
        st.dataframe(
            blend_show.round(4),
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Download point-in-time outputs", expanded=False):
            hold_export = r["hold25"].copy()
            st.download_button(
                "Download 2025 Holdout Predictions",
                hold_export.to_csv(index=False).encode("utf-8"),
                file_name="mlb_pit_holdout_2025.csv",
                mime="text/csv",
                key="download_pit_holdout"
            )
            if not r["hold_bets_cal"].empty:
                st.download_button(
                    "Download 2025 Calibrated Bet Ledger",
                    r["hold_bets_cal"].to_csv(index=False).encode("utf-8"),
                    file_name="mlb_pit_bets_2025.csv",
                    mime="text/csv",
                    key="download_pit_bets"
                )

        st.caption(
            "Interpretation guard: this is a clean point-in-time validation model built from lagged team results. "
            "It is not yet a historical replay of every starter, lineup, bullpen, park, weather and travel input used by the live engine."
        )

    st.divider()

    st.markdown('<div class="section-kicker">PIT STARTING PITCHER TEST</div>', unsafe_allow_html=True)
    st.caption(
        "Adds historical starting-pitcher quality to the point-in-time model using MLB's free Stats API. "
        "Pitcher game logs are filtered strictly to starts before each target game. Zero Odds API credits."
    )

    pitcher_upload = st.file_uploader(
        "Upload Moneyline Master CSV for Pitcher Test",
        type=["csv"],
        key="pit_pitcher_master_upload"
    )

    if pitcher_upload is not None:
        try:
            pitcher_master = pd.read_csv(pitcher_upload)

            pc1, pc2 = st.columns(2)
            with pc1:
                pitcher_min_team = st.slider(
                    "Prior team games",
                    min_value=5,
                    max_value=25,
                    value=12,
                    step=1,
                    key="pitcher_min_team"
                )
            with pc2:
                pitcher_min_starts = st.slider(
                    "Prior starter starts",
                    min_value=1,
                    max_value=8,
                    value=3,
                    step=1,
                    key="pitcher_min_starts"
                )

            st.info(
                "First run can take several minutes because the app builds and caches historical pitcher game logs. "
                "Those requests use MLB's free API, not The Odds API."
            )

            if st.button(
                "Run PIT Starting Pitcher Backtest",
                type="primary",
                key="run_pitcher_pit"
            ):
                with st.spinner(
                    "Building historical starter identities and prior-start features…"
                ):
                    pitcher_result = run_pitcher_point_in_time_backtest(
                        pitcher_master,
                        min_prior_games=pitcher_min_team,
                        min_pitcher_starts=pitcher_min_starts
                    )
                st.session_state["pitcher_pit_result"] = pitcher_result

        except Exception as ex:
            st.error(f"Could not run pitcher PIT backtest: {ex}")

    if "pitcher_pit_result" in st.session_state:
        pr = st.session_state["pitcher_pit_result"]
        hold = pr["hold25"]
        rs = pr["raw_summary"]
        cs = pr["cal_summary"]

        st.success(
            f"Pitcher walk-forward complete. 2024 selected "
            f"{pr['best_weight']*100:.0f}% pitcher model / "
            f"{(1-pr['best_weight'])*100:.0f}% market."
        )

        st.markdown('<div class="section-kicker">2025 PITCHER HOLDOUT</div>', unsafe_allow_html=True)
        pm = pr["metrics"].copy()
        st.dataframe(pm.round(4), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raw pitcher PIT**")
            st.metric("ROI", f"{rs['ROI']*100:+.1f}%")
            st.caption(
                f"{rs['Wins']}-{rs['Losses']} • {rs['Units']:+.2f}u • {rs['Bets']} bets"
            )
        with c2:
            st.markdown("**Market-calibrated pitcher PIT**")
            st.metric("ROI", f"{cs['ROI']*100:+.1f}%")
            st.caption(
                f"{cs['Wins']}-{cs['Losses']} • {cs['Units']:+.2f}u • {cs['Bets']} bets"
            )

        with st.expander("Pitcher feature set", expanded=False):
            st.write(pr["feature_cols"])

        with st.expander("Download pitcher PIT outputs", expanded=False):
            st.download_button(
                "Download 2025 Pitcher Holdout",
                hold.to_csv(index=False).encode("utf-8"),
                file_name="mlb_pit_pitcher_holdout_2025.csv",
                mime="text/csv",
                key="download_pitcher_holdout"
            )
            if not pr["cal_bets"].empty:
                st.download_button(
                    "Download 2025 Pitcher Bet Ledger",
                    pr["cal_bets"].to_csv(index=False).encode("utf-8"),
                    file_name="mlb_pit_pitcher_bets_2025.csv",
                    mime="text/csv",
                    key="download_pitcher_bets"
                )

        st.caption(
            "Interpretation guard: historical probable/starter identity coverage can be incomplete. "
            "Only games with sufficient prior-start history for both starters enter this test."
        )

    st.divider()
    st.markdown('<div class="section-kicker">v0.12.1.3 • PITCHER LEAKAGE AUDIT</div>', unsafe_allow_html=True)
    st.caption(
        "Try to break the pitcher result before production. This reruns the same walk-forward test under stricter "
        "pregame windows and removes doubleheaders. It also fixes MLB innings notation (5.2 = 5⅔ IP). "
        "v0.12.1.3 preserves the historical snapshot timing used by the stress windows. Zero Odds API credits."
    )
    st.warning(
        "Important: MLB's historical starter identity is retrospective. This audit can stress-test the result, "
        "but it cannot prove the listed starter was publicly known at the exact historical odds snapshot."
    )
    audit_upload = st.file_uploader(
        "Upload Moneyline Master CSV for Audit",
        type=["csv"], key="pitcher_audit_master_upload"
    )
    if audit_upload is not None:
        try:
            audit_master=pd.read_csv(audit_upload)
            ac1,ac2=st.columns(2)
            with ac1:
                audit_team=st.slider("Audit prior team games",5,25,12,1,key="audit_team")
            with ac2:
                audit_starts=st.slider("Audit prior starter starts",1,8,3,1,key="audit_starts")
            # Two-phase launch: on mobile, immediately acknowledge the tap and rerun before
            # starting the long MLB Stats API / PIT calculation. This prevents the button from
            # appearing to do nothing while the synchronous audit is starting.
            if st.button("Run Pitcher Leakage Audit",type="primary",key="run_pitcher_audit"):
                st.session_state["pitcher_audit_pending"] = True
                st.session_state["pitcher_audit_params"] = (int(audit_team), int(audit_starts))
                st.session_state.pop("pitcher_audit_result", None)
                st.rerun()

            if st.session_state.get("pitcher_audit_pending", False):
                run_team, run_starts = st.session_state.get("pitcher_audit_params", (audit_team, audit_starts))
                st.info("Audit started. Keep this page open — the first run can take several minutes while MLB pitcher data is fetched/cached.")
                progress = st.progress(5, text="Starting pitcher leakage audit…")
                try:
                    progress.progress(15, text="Building point-in-time pitcher features…")
                    result = run_pitcher_audit(
                        audit_master, min_prior_games=int(run_team), min_pitcher_starts=int(run_starts)
                    )
                    progress.progress(100, text="Pitcher leakage audit complete.")
                    st.session_state["pitcher_audit_result"] = result
                    st.session_state["pitcher_audit_pending"] = False
                    st.session_state["pitcher_audit_just_completed"] = True
                    # Force a clean render pass after the long synchronous calculation.
                    # This is especially important on Streamlit mobile, where widgets below
                    # a completed long-running callback can fail to paint until the next rerun.
                    st.rerun()
                except Exception as audit_ex:
                    st.session_state["pitcher_audit_pending"] = False
                    st.error(f"Pitcher audit failed: {audit_ex}")
        except Exception as ex:
            st.error(f"Could not prepare pitcher audit: {ex}")

    if "pitcher_audit_result" in st.session_state:
        ar = st.session_state.get("pitcher_audit_result") or {}
        at = ar.get("table", pd.DataFrame())
        results_map = ar.get("results", {}) or {}

        if st.session_state.pop("pitcher_audit_just_completed", False):
            st.success("Audit complete — results are ready below.")
        else:
            st.success("Pitcher leakage audit results loaded.")

        if at is None or at.empty:
            st.error(
                "The audit finished but produced no stress-test rows. "
                "This usually means no games met the starter-history / time-window rules."
            )
            pit_rows = len(ar.get("pit", [])) if ar.get("pit") is not None else 0
            st.caption(f"PIT feature rows built: {pit_rows:,} • Stress-test rows: 0")
            diag = ar.get("timing_diagnostics", {}) or {}
            with st.expander("Audit timing diagnostics", expanded=True):
                if diag:
                    st.json(diag)
                else:
                    st.write("No timing diagnostics were produced.")
                if results_map:
                    st.write(results_map)
        else:
            show = at.copy()
            numeric_cols = [
                "Model_Weight","Market_Brier_2025","Model_Brier_2025",
                "Cal_Brier_2025","Brier_Improvement","ROI","Units",
                "Games_2023","Games_2024","Games_2025","Bets","Wins","Losses"
            ]
            for c in numeric_cols:
                if c in show.columns:
                    show[c] = pd.to_numeric(show[c], errors="coerce")

            # Add an explicit audit verdict so the user does not have to interpret raw columns.
            def _audit_status(row):
                imp = float(row.get("Brier_Improvement", 0) or 0)
                wt = float(row.get("Model_Weight", 0) or 0)
                roi = float(row.get("ROI", 0) or 0)
                n = float(row.get("Games_2025", 0) or 0)
                if n < 250:
                    return "WARNING — small sample"
                if imp > 0 and wt > 0 and roi > 0:
                    return "PASS"
                if imp > 0 and wt > 0:
                    return "WARNING — predictive only"
                return "FAIL"

            show.insert(1, "Audit_Status", show.apply(_audit_status, axis=1))

            st.markdown("**2025 stress-test ladder**")
            st.dataframe(
                show.round({
                    "Model_Weight":2,"Market_Brier_2025":4,"Model_Brier_2025":4,
                    "Cal_Brier_2025":4,"Brier_Improvement":4,"ROI":4,"Units":2
                }),
                use_container_width=True,
                hide_index=True
            )

            strict = results_map.get("No DH • ≤3h") or results_map.get("No DH • ≤6h")
            if strict and not strict.get("Error"):
                imp = float(strict.get("Brier_Improvement", 0) or 0)
                roi = float(strict.get("ROI", 0) or 0)
                wt = float(strict.get("Model_Weight", 0) or 0)
                games = int(strict.get("Games_2025", 0) or 0)
                bets = int(strict.get("Bets", 0) or 0)
                units = float(strict.get("Units", 0) or 0)
                if imp > 0 and wt > 0:
                    st.success(
                        f"Strict-window signal survived: {games:,} 2025 games • "
                        f"Brier improvement {imp:+.4f} • validation model weight {wt*100:.0f}% • "
                        f"{bets:,} bets • {units:+.2f}u • ROI {roi*100:+.1f}%."
                    )
                else:
                    st.warning(
                        "The strict-window test did not preserve the original predictive signal. "
                        "Do not promote the pitcher model to production yet."
                    )
            else:
                st.warning("No usable ≤3h/≤6h strict-window result was produced. Review the ladder above.")

            st.download_button(
                "Download Audit Summary CSV",
                at.to_csv(index=False).encode("utf-8"),
                file_name="mlb_pit_pitcher_audit_summary.csv",
                mime="text/csv",
                key="dl_pitcher_audit_summary_main"
            )

            with st.expander("Download individual holdouts", expanded=False):
                for label, r in results_map.items():
                    if isinstance(r, dict) and "hold25" in r:
                        safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
                        st.download_button(
                            f"Download {label} Holdout",
                            r["hold25"].to_csv(index=False).encode("utf-8"),
                            file_name=f"mlb_pit_pitcher_audit_{safe}_2025.csv",
                            mime="text/csv",
                            key=f"dl_holdout_{safe}"
                        )

            st.caption(
                "Audit rule: we want positive out-of-sample Brier improvement that persists as the odds snapshot "
                "moves closer to first pitch, without tuning thresholds on 2025."
            )



    st.divider()
    st.markdown('<div class="section-kicker">v0.12.2 • PITCHER INTEGRITY TEST</div>', unsafe_allow_html=True)
    st.caption("Harder validation of the starter signal. Uses the same Moneyline Master and zero Odds API credits. The key question: does the edge collapse when the correct starter information is destroyed?")
    st.info("PASS logic: correct starters should beat market; team-only should be weaker; scrambled/swapped starter placebos should materially deteriorate. If placebos stay strong, suspect leakage elsewhere.")
    integrity_upload=st.file_uploader("Upload Moneyline Master CSV for Integrity Test",type=["csv"],key="pitcher_integrity_upload")
    if integrity_upload is not None:
        try:
            integrity_master=pd.read_csv(integrity_upload)
            ic1,ic2=st.columns(2)
            with ic1: integ_team=st.slider("Integrity prior team games",5,25,12,1,key="integ_team")
            with ic2: integ_starts=st.slider("Integrity prior starter starts",1,8,3,1,key="integ_starts")
            if st.button("Run Pitcher Integrity Test",type="primary",key="run_pitcher_integrity"):
                st.session_state["pitcher_integrity_pending"]=True
                st.session_state["pitcher_integrity_params"]=(int(integ_team),int(integ_starts))
                st.session_state.pop("pitcher_integrity_result",None)
                st.rerun()
            if st.session_state.get("pitcher_integrity_pending",False):
                rt,rs=st.session_state.get("pitcher_integrity_params",(integ_team,integ_starts))
                st.info("Integrity test started. Keep this page open; cached MLB pitcher data will be reused where available.")
                prog=st.progress(8,text="Building PIT pitcher table…")
                try:
                    out=run_pitcher_integrity_test(integrity_master,min_prior_games=rt,min_pitcher_starts=rs)
                    prog.progress(100,text="Pitcher integrity test complete.")
                    st.session_state["pitcher_integrity_result"]=out
                    st.session_state["pitcher_integrity_pending"]=False
                    st.session_state["pitcher_integrity_just_completed"]=True
                    st.rerun()
                except Exception as ex:
                    st.session_state["pitcher_integrity_pending"]=False
                    st.error(f"Pitcher integrity test failed: {ex}")
        except Exception as ex:
            st.error(f"Could not prepare integrity test: {ex}")
    if "pitcher_integrity_result" in st.session_state:
        ir=st.session_state["pitcher_integrity_result"] or {}; tab=ir.get("table",pd.DataFrame())
        if st.session_state.pop("pitcher_integrity_just_completed",False):
            st.success("Integrity test complete — results below.")
        if tab is None or tab.empty:
            st.error("Integrity test returned no rows.")
        else:
            st.markdown("**Integrity ladder**")
            st.dataframe(tab.round({"Model_Weight":2,"Market_Brier":4,"Raw_Model_Brier":4,"Cal_Brier":4,"Brier_Improvement":4,"Units":2,"ROI":4}),use_container_width=True,hide_index=True)
            rows={r.get("Test"):r for _,r in tab.iterrows()}
            corr=rows.get("Correct starters • ≤6h"); scr=rows.get("Scrambled starters placebo • ≤6h")
            if corr is not None and scr is not None and pd.isna(corr.get("Error",np.nan)) and pd.isna(scr.get("Error",np.nan)):
                cimp=float(corr.get("Brier_Improvement",0) or 0); simp=float(scr.get("Brier_Improvement",0) or 0); cwt=float(corr.get("Model_Weight",0) or 0); swt=float(scr.get("Model_Weight",0) or 0)
                if cimp>0 and cwt>0 and (simp<=0 or swt<cwt):
                    st.success("PASS signal: correct starter information materially outperforms the scrambled-starter placebo.")
                else:
                    st.warning("WARNING: the placebo did not deteriorate enough. Do not move the pitcher model into production yet.")
            seg=ir.get("segments",pd.DataFrame())
            if seg is not None and not seg.empty:
                with st.expander("Correct-starter 2025 robustness splits",expanded=True):
                    st.dataframe(seg.round({"Hit":3,"Units":2,"ROI":4}),use_container_width=True,hide_index=True)
            st.download_button("Download Integrity Summary CSV",tab.to_csv(index=False).encode("utf-8"),file_name="mlb_pit_pitcher_integrity_summary.csv",mime="text/csv",key="dl_integrity_summary")
            if seg is not None and not seg.empty:
                st.download_button("Download Integrity Segments CSV",seg.to_csv(index=False).encode("utf-8"),file_name="mlb_pit_pitcher_integrity_segments.csv",mime="text/csv",key="dl_integrity_segments")
    st.divider()
    st.markdown('<div class="section-kicker">v0.12.3 • PITCHER CAUSALITY AUDIT</div>', unsafe_allow_html=True)
    st.caption("Inference-only corruption + feature ablation. The trained model does NOT get to relearn swapped/scrambled starters. Uses the same Moneyline Master and zero Odds API credits.")
    st.info("Key test: correct 2025 starters should beat inference-only scrambled, opponent-starter, and lagged-wrong-starter controls. Feature ablations show which starter metrics actually matter.")
    caus_upload=st.file_uploader("Upload Moneyline Master CSV for Causality Audit",type=["csv"],key="pitcher_causality_upload")
    if caus_upload is not None:
        try:
            caus_master=pd.read_csv(caus_upload)
            cc1,cc2=st.columns(2)
            with cc1: caus_team=st.slider("Causality prior team games",5,25,12,1,key="caus_team")
            with cc2: caus_starts=st.slider("Causality prior starter starts",1,8,3,1,key="caus_starts")
            if st.button("Run Pitcher Causality Audit",type="primary",key="run_pitcher_causality"):
                st.session_state["pitcher_causality_pending"]=True
                st.session_state["pitcher_causality_params"]=(int(caus_team),int(caus_starts))
                st.session_state.pop("pitcher_causality_result",None)
                st.rerun()
            if st.session_state.get("pitcher_causality_pending",False):
                rt,rs=st.session_state.get("pitcher_causality_params",(caus_team,caus_starts))
                st.info("Causality audit started. Keep this page open; cached MLB pitcher data will be reused.")
                prog=st.progress(8,text="Building strict PIT pitcher table…")
                try:
                    out=run_pitcher_causality_audit(caus_master,min_prior_games=rt,min_pitcher_starts=rs)
                    prog.progress(100,text="Pitcher causality audit complete.")
                    st.session_state["pitcher_causality_result"]=out
                    st.session_state["pitcher_causality_pending"]=False
                    st.session_state["pitcher_causality_just_completed"]=True
                    st.rerun()
                except Exception as ex:
                    st.session_state["pitcher_causality_pending"]=False
                    st.error(f"Pitcher causality audit failed: {ex}")
        except Exception as ex:
            st.error(f"Could not prepare causality audit: {ex}")
    if "pitcher_causality_result" in st.session_state:
        cr=st.session_state["pitcher_causality_result"] or {}; ct=cr.get("table",pd.DataFrame())
        if st.session_state.pop("pitcher_causality_just_completed",False):
            st.success("Causality audit complete — results below.")
        if ct is None or ct.empty:
            st.error("Causality audit returned no rows.")
        else:
            st.markdown("**Causality ladder**")
            st.dataframe(ct.round({"Model_Weight":2,"Market_Brier":4,"Raw_Model_Brier":4,"Cal_Brier":4,"Brier_Improvement":4,"Units":2,"ROI":4}),use_container_width=True,hide_index=True)
            rows={r.get("Test"):r for _,r in ct.iterrows()}
            corr=rows.get("Correct starters • ≤6h")
            bad=[rows.get("2025 scrambled starters • inference-only"),rows.get("2025 opponent starters • inference-only"),rows.get("2025 lagged wrong starters • inference-only")]
            if corr is not None and pd.isna(corr.get("Error",np.nan)):
                c_raw=float(corr.get("Raw_Model_Brier",np.nan)); mkt=float(corr.get("Market_Brier",np.nan))
                placebo_raw=[float(r.get("Raw_Model_Brier",np.nan)) for r in bad if r is not None and pd.isna(r.get("Error",np.nan))]
                deteriorated=[p > c_raw + 0.004 for p in placebo_raw if np.isfinite(p)]
                if c_raw < mkt and len(deteriorated)>=2 and sum(deteriorated)>=2:
                    st.success("PASS signal: correct starters beat market and at least two inference-only wrong-starter controls deteriorated materially.")
                else:
                    st.warning("WARNING: causality controls did not deteriorate enough. Do not promote the starter model yet.")
            seg=cr.get("segments",pd.DataFrame())
            if seg is not None and not seg.empty:
                with st.expander("Correct-starter 2025 robustness splits",expanded=True):
                    st.dataframe(seg.round({"Hit":3,"Units":2,"ROI":4}),use_container_width=True,hide_index=True)
            st.download_button("Download Causality Summary CSV",ct.to_csv(index=False).encode("utf-8"),file_name="mlb_pit_pitcher_causality_summary.csv",mime="text/csv",key="dl_causality_summary")
            if seg is not None and not seg.empty:
                st.download_button("Download Causality Segments CSV",seg.to_csv(index=False).encode("utf-8"),file_name="mlb_pit_pitcher_causality_segments.csv",mime="text/csv",key="dl_causality_segments")
    st.divider()
    st.markdown('<div class="section-kicker">v0.13.0 • PITCHER MODEL 2.0</div>', unsafe_allow_html=True)
    st.caption("Engineer the starter signal instead of tuning betting thresholds. Uses the same strict ≤6h, no-doubleheader sample and the same 2023 → 2024 → 2025 walk-forward protocol.")
    st.info("Adds fixed-prior shrinkage, FIP-style components, exponentially weighted recent form, K-BB skill, innings/start, starter rest, recent workload and pitch-count features. The v0.12.3 benchmark is rerun on the exact same rows for a fair comparison.")
    v13_upload=st.file_uploader("Upload Moneyline Master CSV for Pitcher Model 2.0",type=["csv"],key="pitcher_v13_upload")
    if v13_upload is not None:
        try:
            v13_master=pd.read_csv(v13_upload)
            vc1,vc2=st.columns(2)
            with vc1: v13_team=st.slider("v0.13 prior team games",5,25,12,1,key="v13_team")
            with vc2: v13_starts=st.slider("v0.13 prior starter starts",1,8,3,1,key="v13_starts")
            if st.button("Run Pitcher Model 2.0 Test",type="primary",key="run_v13_pitcher"):
                st.session_state["v13_pitcher_pending"]=True
                st.session_state["v13_pitcher_params"]=(int(v13_team),int(v13_starts))
                st.session_state.pop("v13_pitcher_result",None)
                st.rerun()
            if st.session_state.get("v13_pitcher_pending",False):
                rt,rs=st.session_state.get("v13_pitcher_params",(v13_team,v13_starts))
                st.info("Pitcher Model 2.0 test started. Cached MLB starter histories will be reused where available.")
                prog=st.progress(8,text="Building enhanced point-in-time starter profiles…")
                try:
                    out=run_pitcher_model_2_test(v13_master,min_prior_games=rt,min_pitcher_starts=rs)
                    prog.progress(100,text="Pitcher Model 2.0 comparison complete.")
                    st.session_state["v13_pitcher_result"]=out
                    st.session_state["v13_pitcher_pending"]=False
                    st.session_state["v13_pitcher_just_completed"]=True
                    st.rerun()
                except Exception as ex:
                    st.session_state["v13_pitcher_pending"]=False
                    st.error(f"Pitcher Model 2.0 test failed: {ex}")
        except Exception as ex:
            st.error(f"Could not prepare Pitcher Model 2.0 test: {ex}")

    if "v13_pitcher_result" in st.session_state:
        vr=st.session_state["v13_pitcher_result"] or {}; vt=vr.get("table",pd.DataFrame())
        if st.session_state.pop("v13_pitcher_just_completed",False):
            st.success("Pitcher Model 2.0 test complete — benchmark comparison below.")
        if vt is None or vt.empty:
            st.error("Pitcher Model 2.0 returned no comparison rows.")
        else:
            st.markdown("**2025 untouched holdout — same rows, same bet thresholds**")
            show_cols=[c for c in ["Model","Games_2025","Model_Weight","Market_Brier","Raw_Model_Brier","Cal_Brier","Brier_Improvement","Bets","Wins","Losses","Units","ROI"] if c in vt.columns]
            st.dataframe(vt[show_cols].round({"Model_Weight":2,"Market_Brier":4,"Raw_Model_Brier":4,"Cal_Brier":4,"Brier_Improvement":4,"Units":2,"ROI":4}),use_container_width=True,hide_index=True)
            try:
                b=vt[vt["Model"]=="v0.12.3 benchmark"].iloc[0]
                n=vt[vt["Model"]=="v0.13 Pitcher Model 2.0"].iloc[0]
                if float(n["Cal_Brier"]) < float(b["Cal_Brier"]) - .001:
                    st.success(f"MODEL 2.0 PASS: calibrated Brier improved by {float(b['Cal_Brier'])-float(n['Cal_Brier']):.4f} versus the frozen v0.12.3 benchmark.")
                elif float(n["Cal_Brier"]) <= float(b["Cal_Brier"]):
                    st.info("MODEL 2.0 NEUTRAL: predictive accuracy is roughly tied with v0.12.3. Do not promote based on ROI alone.")
                else:
                    st.warning("MODEL 2.0 FAIL: v0.13 did not beat the frozen v0.12.3 benchmark on 2025 Brier. Keep v0.12.3 as the research baseline.")
            except Exception:
                pass
            seg=vr.get("segments",pd.DataFrame())
            if seg is not None and not seg.empty:
                with st.expander("v0.13 2025 betting robustness splits",expanded=True):
                    st.dataframe(seg.round({"Hit":3,"Units":2,"ROI":4}),use_container_width=True,hide_index=True)
            coef=vr.get("coefficients",pd.DataFrame())
            if coef is not None and not coef.empty:
                with st.expander("Pitcher Model 2.0 fitted coefficients",expanded=False):
                    st.caption("Standardized development-set coefficients. Useful for diagnosis, not causal interpretation.")
                    st.dataframe(coef.round({"Coefficient":4,"Abs_Coefficient":4}),use_container_width=True,hide_index=True)
            st.download_button("Download v0.13 Comparison CSV",vt.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_model_2_comparison.csv",mime="text/csv",key="dl_v13_compare")
            if seg is not None and not seg.empty:
                st.download_button("Download v0.13 Segments CSV",seg.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_model_2_segments.csv",mime="text/csv",key="dl_v13_segments")
            if coef is not None and not coef.empty:
                st.download_button("Download v0.13 Coefficients CSV",coef.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_model_2_coefficients.csv",mime="text/csv",key="dl_v13_coefficients")
            v2res=(vr.get("results") or {}).get("v0.13 Pitcher Model 2.0",{})
            if isinstance(v2res,dict):
                hold=v2res.get("hold25"); bets=v2res.get("bets")
                if isinstance(hold,pd.DataFrame) and not hold.empty:
                    st.download_button("Download v0.13 2025 Holdout CSV",hold.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_model_2_holdout_2025.csv",mime="text/csv",key="dl_v13_hold")
                if isinstance(bets,pd.DataFrame) and not bets.empty:
                    st.download_button("Download v0.13 2025 Bets CSV",bets.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_model_2_bets_2025.csv",mime="text/csv",key="dl_v13_bets")
    st.divider()
    st.markdown('<div class="section-kicker">v0.14.1 • PIT BULLPEN TEST</div>', unsafe_allow_html=True)
    st.caption("Frozen v0.13 Pitcher Model 2.0 versus Pitcher 2.0 + point-in-time bullpen quality and availability. Same ≤6h, no-doubleheader walk-forward protocol.")
    st.info("Bullpen data comes from historical MLB full-season rosters plus each pitcher's free MLB game log, filtered strictly to relief appearances for that team before each target game. Adds season-shrunk relief skill, 7/14-day form, 1/3-day workload, heavy-use arms and back-to-back availability. 2025 never selects the blend or betting thresholds.")
    v14_upload=st.file_uploader("Upload Moneyline Master CSV for Bullpen Test",type=["csv"],key="bullpen_v14_upload")
    if v14_upload is not None:
        try:
            v14_master=pd.read_csv(v14_upload)
            bc1,bc2,bc3=st.columns(3)
            with bc1: v14_team=st.slider("v0.14 prior team games",5,25,12,1,key="v14_team")
            with bc2: v14_starts=st.slider("v0.14 prior starter starts",1,8,3,1,key="v14_starts")
            with bc3: v14_bpip=st.slider("Minimum prior bullpen IP",6,30,12,1,key="v14_bpip")
            if st.button("Run Pitcher + Bullpen Test",type="primary",key="run_v14_bullpen"):
                st.session_state["v14_bullpen_pending"]=True
                st.session_state["v14_bullpen_params"]=(int(v14_team),int(v14_starts),float(v14_bpip))
                st.session_state.pop("v14_bullpen_result",None)
                st.rerun()
            if st.session_state.get("v14_bullpen_pending",False):
                rt,rs,rip=st.session_state.get("v14_bullpen_params",(v14_team,v14_starts,v14_bpip))
                st.info("Bullpen test started. First run may take a few minutes while historical rosters and pitcher-season relief logs are cached; reruns should be much faster.")
                prog14=st.progress(5,text="Building point-in-time pitcher + bullpen states…")
                try:
                    out=run_pitcher_bullpen_test(v14_master,min_prior_games=rt,min_pitcher_starts=rs,min_bullpen_ip=rip)
                    prog14.progress(100,text="Pitcher + bullpen comparison complete.")
                    st.session_state["v14_bullpen_result"]=out
                    st.session_state["v14_bullpen_pending"]=False
                    st.session_state["v14_bullpen_just_completed"]=True
                    st.rerun()
                except Exception as ex:
                    st.session_state["v14_bullpen_pending"]=False
                    st.error(f"Pitcher + bullpen test failed: {ex}")
        except Exception as ex:
            st.error(f"Could not prepare bullpen test: {ex}")

    if "v14_bullpen_result" in st.session_state:
        br=st.session_state["v14_bullpen_result"] or {}; bt14=br.get("table",pd.DataFrame())
        if st.session_state.pop("v14_bullpen_just_completed",False):
            st.success("Bullpen test complete — frozen champion comparison below.")
        q=br.get("quality",pd.DataFrame())
        if isinstance(q,pd.DataFrame) and not q.empty:
            qq=q.iloc[0]
            st.caption(f"Bullpen coverage: {int(qq.get('PIT_Bullpen_Rows',0)):,}/{int(qq.get('PIT_Pitcher_Rows',0)):,} PIT rows • {int(qq.get('Team_Seasons_Loaded',0))}/{int(qq.get('Team_Seasons_Requested',0))} team-seasons loaded • {int(qq.get('Cache_Hits',0))} cache hits")
        if bt14 is None or bt14.empty:
            st.error("Bullpen test returned no comparison rows.")
        else:
            st.markdown("**2025 untouched holdout — identical bullpen-eligible rows**")
            cols=[c for c in ["Model","Games_2025","Model_Weight","Validation_2024_Brier","Market_Brier","Raw_Model_Brier","Cal_Brier","Brier_Improvement","Bets","Wins","Losses","Units","ROI"] if c in bt14.columns]
            st.dataframe(bt14[cols].round({"Model_Weight":2,"Validation_2024_Brier":4,"Market_Brier":4,"Raw_Model_Brier":4,"Cal_Brier":4,"Brier_Improvement":4,"Units":2,"ROI":4}),use_container_width=True,hide_index=True)
            try:
                base=bt14[bt14["Model"]=="v0.13 frozen Pitcher 2.0"].iloc[0]
                new=bt14[bt14["Model"]=="v0.14 Pitcher 2.0 + Bullpen"].iloc[0]
                delta=float(base["Cal_Brier"])-float(new["Cal_Brier"])
                val_delta=float(base["Validation_2024_Brier"])-float(new["Validation_2024_Brier"])
                if delta>=.001 and val_delta>=0:
                    st.success(f"BULLPEN PASS: 2025 calibrated Brier improves by {delta:.4f} and 2024 validation does not deteriorate. Bullpen earns the next research slot.")
                elif delta>0 and val_delta>=-.0005:
                    st.info(f"BULLPEN NEUTRAL: small 2025 Brier gain ({delta:.4f}). Keep v0.13 frozen until the gain is stronger; do not promote on ROI alone.")
                else:
                    st.warning("BULLPEN FAIL: point-in-time bullpen features did not improve the frozen champion cleanly. Keep v0.13 and discard/tighten this bullpen layer.")
            except Exception:
                pass
            seg=br.get("segments",pd.DataFrame())
            if isinstance(seg,pd.DataFrame) and not seg.empty:
                with st.expander("v0.14 2025 betting robustness splits",expanded=True):
                    st.dataframe(seg.round({"Hit":3,"Units":2,"ROI":4}),use_container_width=True,hide_index=True)
            coef=br.get("coefficients",pd.DataFrame())
            if isinstance(coef,pd.DataFrame) and not coef.empty:
                bpcoef=coef[coef["Feature"].astype(str).str.startswith("BP_")].copy()
                with st.expander("Bullpen feature coefficients",expanded=False):
                    st.caption("Standardized development-set coefficients. Diagnostic only; promotion is based on out-of-sample Brier.")
                    st.dataframe((bpcoef if not bpcoef.empty else coef.head(25)).round({"Coefficient":4,"Abs_Coefficient":4}),use_container_width=True,hide_index=True)
            st.download_button("Download v0.14 Comparison CSV",bt14.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_bullpen_comparison.csv",mime="text/csv",key="dl_v14_compare")
            if isinstance(q,pd.DataFrame) and not q.empty:
                st.download_button("Download v0.14 Data Quality CSV",q.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_bullpen_data_quality.csv",mime="text/csv",key="dl_v14_quality")
            if isinstance(seg,pd.DataFrame) and not seg.empty:
                st.download_button("Download v0.14 Segments CSV",seg.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_bullpen_segments.csv",mime="text/csv",key="dl_v14_segments")
            if isinstance(coef,pd.DataFrame) and not coef.empty:
                st.download_button("Download v0.14 Coefficients CSV",coef.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_bullpen_coefficients.csv",mime="text/csv",key="dl_v14_coefficients")
            v14res=(br.get("results") or {}).get("v0.14 Pitcher 2.0 + Bullpen",{})
            if isinstance(v14res,dict):
                hold=v14res.get("hold25"); bets=v14res.get("bets")
                if isinstance(hold,pd.DataFrame) and not hold.empty:
                    st.download_button("Download v0.14 2025 Holdout CSV",hold.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_bullpen_holdout_2025.csv",mime="text/csv",key="dl_v14_hold")
                if isinstance(bets,pd.DataFrame) and not bets.empty:
                    st.download_button("Download v0.14 2025 Bets CSV",bets.to_csv(index=False).encode("utf-8"),file_name="mlb_pitcher_bullpen_bets_2025.csv",mime="text/csv",key="dl_v14_bets")

    st.divider()
    uploaded_bt = st.file_uploader(
        "Upload completed backtest dataset",
        type=["csv"],
        key="mlb_backtest_csv",
        help="Processed locally in Streamlit. This does not use The Odds API.",
    )

    if uploaded_bt is None:
        st.info("Build/download the historical market dataset above. The full backtest still needs point-in-time model outputs and final results merged into the completed dataset.")
    else:
        try:
            raw_bt = pd.read_csv(uploaded_bt)
            bt, missing = normalize_backtest_columns(raw_bt)

            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            elif bt is None or bt.empty:
                st.error("No valid historical betting rows were found.")
            else:
                bt = attach_backtest_pnl(bt)

                st.markdown('<div class="section-kicker">FILTERS</div>', unsafe_allow_html=True)
                min_date = bt["Date"].min().date()
                max_date = bt["Date"].max().date()

                c1, c2 = st.columns(2)
                with c1:
                    start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
                with c2:
                    end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

                market_opts = sorted(bt["Market_Type"].dropna().unique().tolist())
                selected_markets = st.multiselect("Markets", market_opts, default=market_opts)

                verdict_opts = ["STRONG BET","BET","LEAN","PASS"]
                selected_verdicts = st.multiselect(
                    "Verdicts",
                    verdict_opts,
                    default=["STRONG BET","BET"],
                    format_func=user_verdict,
                )

                filt = bt[
                    (bt["Date"].dt.date >= start_date)
                    & (bt["Date"].dt.date <= end_date)
                    & (bt["Market_Type"].isin(selected_markets))
                    & (bt["Verdict"].isin(selected_verdicts))
                ].copy()

                if filt.empty:
                    st.warning("No rows match the current filters.")
                else:
                    s = summarize_bets(filt)
                    units_cls = "bt-good" if s["Units"] > 0 else ("bt-bad" if s["Units"] < 0 else "")
                    roi_cls = "bt-good" if s["ROI"] > 0 else ("bt-bad" if s["ROI"] < 0 else "")
                    avg_odds = f'{s["Avg_Odds"]:+.0f}' if s["Avg_Odds"] is not None else "—"

                    st.markdown('<div class="section-kicker">RESULTS</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="bt-metrics">
                          <div class="bt-metric"><span>Record</span><b>{s['Wins']}-{s['Losses']}-{s['Pushes']}</b></div>
                          <div class="bt-metric"><span>Hit Rate</span><b>{s['Hit_Rate']*100:.1f}%</b></div>
                          <div class="bt-metric"><span>Units</span><b class="{units_cls}">{s['Units']:+.2f}u</b></div>
                          <div class="bt-metric"><span>ROI</span><b class="{roi_cls}">{s['ROI']*100:+.1f}%</b></div>
                          <div class="bt-metric"><span>Bets</span><b>{s['Bets']}</b></div>
                          <div class="bt-metric"><span>Avg Odds</span><b>{avg_odds}</b></div>
                          <div class="bt-metric"><span>Max Drawdown</span><b>{s['Max_Drawdown']:.2f}u</b></div>
                          <div class="bt-metric"><span>Seasons</span><b>{filt['Season'].nunique()}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown('<div class="section-kicker">PRODUCTION CARD SIMULATION</div>', unsafe_allow_html=True)
                    c5, c10 = st.columns(2)
                    for col, n in [(c5,5),(c10,10)]:
                        card = daily_top_card(filt, n=n)
                        cs = summarize_bets(card)
                        with col:
                            st.markdown(f"**Top {n} daily**")
                            st.metric("ROI", f"{cs['ROI']*100:+.1f}%")
                            st.caption(f"{cs['Wins']}-{cs['Losses']}-{cs['Pushes']} • {cs['Units']:+.2f}u • {cs['Bets']} bets")

                    st.markdown('<div class="section-kicker">SEASON STABILITY</div>', unsafe_allow_html=True)
                    st.dataframe(grouped_backtest_summary(filt, "Season"), use_container_width=True, hide_index=True)

                    st.markdown('<div class="section-kicker">BREAKDOWNS</div>', unsafe_allow_html=True)
                    breakdown_col = st.selectbox(
                        "Breakdown",
                        ["Market_Type","Verdict","Edge_Bucket","Odds_Bucket"],
                        format_func=lambda x: {
                            "Market_Type":"By market",
                            "Verdict":"By verdict",
                            "Edge_Bucket":"By edge bucket",
                            "Odds_Bucket":"By odds bucket",
                        }[x],
                    )
                    breakdown = grouped_backtest_summary(filt, breakdown_col)
                    if breakdown_col == "Verdict" and not breakdown.empty:
                        breakdown["Verdict"] = breakdown["Verdict"].map(user_verdict)
                    st.dataframe(breakdown, use_container_width=True, hide_index=True)

                    st.markdown('<div class="section-kicker">CALIBRATION</div>', unsafe_allow_html=True)
                    prob_source = st.radio(
                        "Probability source",
                        ["Calibrated_Prob","Raw_Model_Prob"],
                        horizontal=True,
                        format_func=lambda x: "Calibrated v0.9" if x == "Calibrated_Prob" else "Raw model",
                    )
                    st.dataframe(calibration_table(filt, prob_source), use_container_width=True, hide_index=True)

                    decided = filt[~filt["Push"]].copy()
                    if not decided.empty:
                        raw_brier = ((decided["Raw_Model_Prob"] - decided["Won"].astype(float))**2).mean()
                        cal_brier = ((decided["Calibrated_Prob"] - decided["Won"].astype(float))**2).mean()
                        winner = "Calibrated v0.9" if cal_brier < raw_brier else "Raw model"
                        st.caption(
                            f"Brier score — Raw {raw_brier:.4f} • Calibrated {cal_brier:.4f} • "
                            f"Lower is better → {winner}"
                        )

                    with st.expander("Audit / Export", expanded=False):
                        cols = [
                            "Date","Game","Market_Type","Bet","Odds","Result","Units",
                            "Raw_Model_Prob","Market_NoVig_Prob","Calibrated_Prob",
                            "Edge","EV","Verdict","Confidence"
                        ]
                        audit = filt[cols].sort_values("Date")
                        st.dataframe(audit, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download filtered backtest",
                            audit.to_csv(index=False).encode("utf-8"),
                            file_name="mlb_backtest_filtered.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.error(f"Could not read the historical CSV: {e}")

    st.divider()
    st.caption(
        "Backtest Lab uses only the uploaded historical file. It does not call The Odds API. "
        "Results are only as trustworthy as the historical prices and point-in-time inputs in that file."
    )
    st.stop()

selected_game = None
if mode == "Single Game":
    selected_label = st.selectbox("Game", list(labels.keys()))
    selected_pk = labels[selected_label]
    selected_game = next(g for g in games if g["GamePk"] == selected_pk)
    st.info(f"**Probable pitchers:** {selected_game['Away_SP'] or 'TBD'} vs {selected_game['Home_SP'] or 'TBD'}")

if st.button("Run Model", type="primary"):
    selected_games = [selected_game] if mode == "Single Game" else games
    with st.spinner("Running MLB model. Statcast and bullpen data can take a little while..."):
        try:
            df = run_model(selected_games)
        except Exception as e:
            st.error(f"Model run failed: {e}")
            st.stop()
    if df.empty:
        st.warning("The model did not return any results.")
        st.stop()
    st.session_state.last_results = df

if "last_results" in st.session_state:
    df = st.session_state.last_results

    if len(df) == 1:
        row = df.iloc[0]
        away, home = row["Away"], row["Home"]
        conf = int(row["Model_Confidence"])

        st.markdown('<div class="section-kicker">GAME ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="game-detail-head">
              <div class="game-team">
                {logo_html(away, 36)}
                <div><span>Away</span><b>{html.escape(str(away))}</b></div>
              </div>
              <div class="game-at">@</div>
              <div class="game-team home">
                <div><span>Home</span><b>{html.escape(str(home))}</b></div>
                {logo_html(home, 36)}
              </div>
            </div>
            <div class="game-detail-sub">
              {html.escape(str(row.get('Away_SP', '') or 'TBD'))} vs {html.escape(str(row.get('Home_SP', '') or 'TBD'))}
              • Confidence {conf}/100 ({html.escape(str(row['Confidence_Grade']))})
            </div>
            <div class="proj-grid">
              <div class="proj-box"><span>{html.escape(str(away))} Runs</span><b>{row['Away_Proj_Runs']:.2f}</b></div>
              <div class="proj-box"><span>{html.escape(str(home))} Runs</span><b>{row['Home_Proj_Runs']:.2f}</b></div>
              <div class="proj-box"><span>Model Total</span><b>{row['Model_Total']:.2f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-kicker">CURRENT MARKET</div>', unsafe_allow_html=True)

        payload = st.session_state.get("general_odds_payload") or {}
        events = payload.get("events", [])
        market_error = payload.get("error", "")
        quota = payload.get("quota", {})

        current_market = market_for_game(games, events, int(row["GamePk"]))

        if not odds_api_key:
            st.warning(
                "Automatic market feed is not configured. Add ODDS_API_KEY to Streamlit Secrets."
            )
            st.caption('Streamlit Secrets format: ODDS_API_KEY = "your_new_key_here"')
        elif market_error:
            safe_market_error = str(market_error)
            if odds_api_key:
                safe_market_error = safe_market_error.replace(str(odds_api_key), "[REDACTED]")
            safe_market_error = re.sub(r"apiKey=[^&\s]+", "apiKey=[REDACTED]", safe_market_error, flags=re.I)

            error_code = payload.get("error_code", "")
            st.warning(f"General market unavailable: {safe_market_error}")

            if error_code == "unauthorized":
                st.info(
                    "Fix: create/confirm a valid The Odds API key, then open Streamlit → Manage app → Settings → Secrets "
                    'and replace ODDS_API_KEY. Save the secret and reboot/redeploy the app.'
                )
            elif error_code == "missing_key":
                st.info(
                    'Add this exact secret name in Streamlit: ODDS_API_KEY = "your_valid_key"'
                )
        elif not current_market:
            st.warning(
                "No current consensus market matched this game. "
                "Use the manual line editor below if you still want to grade it."
            )
        else:
            remaining = quota.get("remaining")
            cap = f" • API credits remaining: {remaining}" if remaining not in [None, ""] else ""
            st.success(
                f"Market connected • {current_market.get('provider_count', 0)} US book(s){cap}."
            )
            st.caption(
                "Consensus = median current line/price across available US sportsbooks. "
                "This is a generic market snapshot, not a verified opener or closing line."
            )

            # Load the current market directly into the editable widgets.
            auto_values = {
                f"away_ml_{row['GamePk']}": current_market.get("away_ml"),
                f"home_ml_{row['GamePk']}": current_market.get("home_ml"),
                f"away_rl_odds_{row['GamePk']}": current_market.get("away_rl_odds"),
                f"home_rl_odds_{row['GamePk']}": current_market.get("home_rl_odds"),
                f"total_line_{row['GamePk']}": current_market.get("total"),
                f"over_odds_{row['GamePk']}": current_market.get("over_odds"),
                f"under_odds_{row['GamePk']}": current_market.get("under_odds"),
            }
            if current_market.get("away_rl") is not None:
                auto_values[f"away_rl_side_{row['GamePk']}"] = f"{float(current_market['away_rl']):+.1f}"
            if current_market.get("home_rl") is not None:
                auto_values[f"home_rl_side_{row['GamePk']}"] = f"{float(current_market['home_rl']):+.1f}"

            # Only overwrite on the first load for this exact market snapshot.
            snapshot_key = (
                current_market.get("event_id"),
                current_market.get("last_update"),
                current_market.get("away_ml"),
                current_market.get("home_ml"),
                current_market.get("away_rl"),
                current_market.get("home_rl"),
                current_market.get("total"),
            )
            state_snapshot_key = f"market_snapshot_{row['GamePk']}"
            if st.session_state.get(state_snapshot_key) != snapshot_key:
                for k, v in auto_values.items():
                    if v is not None:
                        st.session_state[k] = v
                st.session_state[state_snapshot_key] = snapshot_key

            market_cols = st.columns(3)
            market_cols[0].metric(
                "Moneyline",
                f"{away} {int(current_market['away_ml']):+d}" if current_market.get("away_ml") is not None else "—",
                f"{home} {int(current_market['home_ml']):+d}" if current_market.get("home_ml") is not None else None,
            )
            market_cols[1].metric(
                "Run Line",
                (
                    f"{away} {float(current_market['away_rl']):+.1f} "
                    f"{int(current_market['away_rl_odds']):+d}"
                    if current_market.get("away_rl") is not None and current_market.get("away_rl_odds") is not None
                    else "—"
                ),
                (
                    f"{home} {float(current_market['home_rl']):+.1f} "
                    f"{int(current_market['home_rl_odds']):+d}"
                    if current_market.get("home_rl") is not None and current_market.get("home_rl_odds") is not None
                    else None
                ),
            )
            market_cols[2].metric(
                "Total",
                f"{float(current_market['total']):.1f}" if current_market.get("total") is not None else "—",
                (
                    f"O {int(current_market['over_odds']):+d} • U {int(current_market['under_odds']):+d}"
                    if current_market.get("over_odds") is not None and current_market.get("under_odds") is not None
                    else None
                ),
            )

            provider_text = current_market.get("providers") or ""
            update_text = current_market.get("last_update") or ""
            if provider_text:
                st.caption(f"Books: {provider_text}")
            if update_text:
                st.caption(f"Market last update: {update_text}")

        auto_market_ok = bool(current_market) and not bool(market_error)

        # Clean automatic betting board.
        # User-facing labels are verbal; internal A/B/C/D rank logic is preserved.
        if auto_market_ok:
            auto_candidates = evaluate_game_markets(row, current_market)

            if auto_candidates:
                best_auto = auto_candidates[0]
                top_label = user_verdict(best_auto["verdict"])
                top_cls = verdict_class(best_auto["verdict"])

                # Keep matchup identity separate from the pick so logos never collide with text.
                matchup_logo_html = (
                    '<div class="clean-matchup-logos">'
                    + logo_html(away, 27)
                    + logo_html(home, 27)
                    + '</div>'
                )

                st.markdown(
                    f"""
                    <div class="clean-top-card {top_cls}">
                      <div class="clean-top-head">
                        <div class="clean-matchup">
                          {matchup_logo_html}
                          <span>{html.escape(str(away))} @ {html.escape(str(home))}</span>
                        </div>
                        <div class="bet-pill {top_cls}">{top_label}</div>
                      </div>
                      <div class="clean-top-pick">{html.escape(best_auto['market'])} {int(best_auto['odds']):+d}</div>
                      <div class="clean-top-sub">{best_auto['market_type']} • Market-calibrated probability</div>
                      <div class="clean-top-metrics">
                        <div><span>Win Chance</span><b>{best_auto['calibrated_prob']*100:.1f}%</b></div>
                        <div><span>Edge</span><b>{best_auto['edge']*100:+.1f}%</b></div>
                        <div><span>EV</span><b>{best_auto['ev']*100:+.1f}%</b></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                actionable = [
                    c for c in auto_candidates
                    if c["verdict"] in {"STRONG BET", "BET", "LEAN"}
                ]
                passes = [c for c in auto_candidates if c["verdict"] == "PASS"]

                st.markdown('<div class="section-kicker">RANKED MARKETS</div>', unsafe_allow_html=True)

                if actionable:
                    for c in actionable:
                        label = user_verdict(c["verdict"])
                        cls = verdict_class(c["verdict"])
                        st.markdown(
                            f"""
                            <div class="action-row">
                              <div class="bet-pill {cls}">{label}</div>
                              <div class="action-main">
                                <div class="action-pick">{html.escape(c['market'])} {int(c['odds']):+d}</div>
                                <div class="action-meta">
                                  Win {c['calibrated_prob']*100:.1f}% • Edge {c['edge']*100:+.1f}% •
                                  EV {c['ev']*100:+.1f}% • Fair {int(c['fair']):+d}
                                </div>
                              </div>
                              <div class="action-type">{html.escape(c['market_type'])}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No Bet or Lean qualifies in the current market.")

                if passes:
                    with st.expander(f"Show all markets • {len(passes)} pass{'es' if len(passes) != 1 else ''}", expanded=False):
                        for c in passes:
                            st.markdown(
                                f"""
                                <div class="pass-row">
                                  <div>
                                    <div class="pass-pick">{html.escape(c['market'])} {int(c['odds']):+d}</div>
                                    <div class="pass-meta">
                                      Win {c['calibrated_prob']*100:.1f}% • Edge {c['edge']*100:+.1f}% • EV {c['ev']*100:+.1f}%
                                    </div>
                                  </div>
                                  <div class="pass-type">{html.escape(c['market_type'])}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        use_manual_fallback = False
        if not auto_market_ok:
            use_manual_fallback = st.checkbox(
                "Use manual odds fallback",
                value=False,
                help="Turn this on only if you want to enter real sportsbook odds manually while the automatic feed is unavailable.",
                key=f"manual_market_fallback_{row['GamePk']}",
            )

        # Safe defaults if a market is unavailable.
        defaults = {
            f"away_ml_{row['GamePk']}": 100,
            f"home_ml_{row['GamePk']}": -110,
            f"away_rl_side_{row['GamePk']}": "+1.5",
            f"away_rl_odds_{row['GamePk']}": -110,
            f"home_rl_side_{row['GamePk']}": "-1.5",
            f"home_rl_odds_{row['GamePk']}": 100,
            f"total_line_{row['GamePk']}": 9.0,
            f"over_odds_{row['GamePk']}": -110,
            f"under_odds_{row['GamePk']}": -110,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        manual_expander_enabled = auto_market_ok or use_manual_fallback
        with st.expander(
            "Advanced tools • Edit market manually" if auto_market_ok else "Advanced tools • Enter manual market odds",
            expanded=bool(use_manual_fallback and not auto_market_ok),
        ):
            if auto_market_ok:
                st.caption(
                    "The automatic consensus is loaded above. Edit only if you want to test a different sportsbook price."
                )
            elif use_manual_fallback:
                st.caption(
                    "Automatic odds are unavailable. Enter actual sportsbook prices before grading."
                )
            else:
                st.caption(
                    "Automatic odds are unavailable. Enable 'Use manual odds fallback' above before entering or grading odds."
                )

            ml1, ml2 = st.columns(2)
            away_ml = ml1.number_input(f"{away} ML", step=5, key=f"away_ml_{row['GamePk']}")
            home_ml = ml2.number_input(f"{home} ML", step=5, key=f"home_ml_{row['GamePk']}")

            rl1, rl2 = st.columns(2)
            away_rl_side = rl1.selectbox(
                f"{away} run line",
                ["+1.5", "-1.5"],
                key=f"away_rl_side_{row['GamePk']}",
            )
            away_rl_odds = rl1.number_input(
                f"{away} RL odds", step=5, key=f"away_rl_odds_{row['GamePk']}"
            )
            home_rl_side = rl2.selectbox(
                f"{home} run line",
                ["-1.5", "+1.5"],
                key=f"home_rl_side_{row['GamePk']}",
            )
            home_rl_odds = rl2.number_input(
                f"{home} RL odds", step=5, key=f"home_rl_odds_{row['GamePk']}"
            )

            t1, t2, t3 = st.columns(3)
            total_line = t1.number_input(
                "Total", step=0.5, key=f"total_line_{row['GamePk']}"
            )
            over_odds = t2.number_input(
                "Over odds", step=5, key=f"over_odds_{row['GamePk']}"
            )
            under_odds = t3.number_input(
                "Under odds", step=5, key=f"under_odds_{row['GamePk']}"
            )

        # Read active values whether or not the expander was opened.
        away_ml = st.session_state[f"away_ml_{row['GamePk']}"]
        home_ml = st.session_state[f"home_ml_{row['GamePk']}"]
        away_rl_side = st.session_state[f"away_rl_side_{row['GamePk']}"]
        away_rl_odds = st.session_state[f"away_rl_odds_{row['GamePk']}"]
        home_rl_side = st.session_state[f"home_rl_side_{row['GamePk']}"]
        home_rl_odds = st.session_state[f"home_rl_odds_{row['GamePk']}"]
        total_line = st.session_state[f"total_line_{row['GamePk']}"]
        over_odds = st.session_state[f"over_odds_{row['GamePk']}"]
        under_odds = st.session_state[f"under_odds_{row['GamePk']}"]

        # Custom-price testing is secondary. Do not repeat the live-market board.
        grade_allowed = auto_market_ok or use_manual_fallback
        grade_clicked = False

        # When the live API market is present, the primary recommendation is already above.
        # Keep edited-price testing inside Advanced Tools only by requiring explicit manual fallback
        # when the automatic feed is unavailable.
        if not auto_market_ok:
            grade_clicked = st.button(
                "Grade Edited Market",
                disabled=not grade_allowed,
                help=None if grade_allowed else "Enable manual odds fallback first.",
            )

        if grade_clicked:
            markets = []
            add_market(markets, f"{away} ML", row["Away_WinProb"], away_ml, conf)
            add_market(markets, f"{home} ML", row["Home_WinProb"], home_ml, conf)

            away_rl_prob = row["Away_+1.5_Prob"] if away_rl_side == "+1.5" else row["Away_-1.5_Prob"]
            home_rl_prob = row["Home_+1.5_Prob"] if home_rl_side == "+1.5" else row["Home_-1.5_Prob"]
            add_market(markets, f"{away} {away_rl_side}", away_rl_prob, away_rl_odds, conf)
            add_market(markets, f"{home} {home_rl_side}", home_rl_prob, home_rl_odds, conf)

            over_verdict, over_edge, over_ev, over_imp, over_model_prob, over_probs = total_bet_grade(
                row["Model_Total"], total_line, "Over", over_odds, conf
            )
            under_verdict, under_edge, under_ev, under_imp, under_model_prob, under_probs = total_bet_grade(
                row["Model_Total"], total_line, "Under", under_odds, conf
            )

            markets.extend([
                {
                    "Bet": f"Over {total_line:g}",
                    "Verdict": over_verdict,
                    "Odds": int(over_odds),
                    "Model Prob": over_model_prob,
                    "Implied Prob": over_imp,
                    "Edge": over_edge,
                    "EV": over_ev,
                    "Model Fair": fair_ml(over_model_prob),
                },
                {
                    "Bet": f"Under {total_line:g}",
                    "Verdict": under_verdict,
                    "Odds": int(under_odds),
                    "Model Prob": under_model_prob,
                    "Implied Prob": under_imp,
                    "Edge": under_edge,
                    "EV": under_ev,
                    "Model Fair": fair_ml(under_model_prob),
                },
            ])

            market_df = pd.DataFrame(markets)
            rank = {"STRONG BET": 3, "BET": 2, "LEAN": 1, "PASS": 0}
            market_df["_rank"] = market_df["Verdict"].map(rank)
            market_df = market_df.sort_values(
                ["_rank", "EV", "Edge"], ascending=False
            ).drop(columns="_rank")

            best = market_df.iloc[0]
            st.markdown(
                f"""
                <div class="custom-result-summary">
                  <b>{user_verdict(best['Verdict'])} — {html.escape(str(best['Bet']))} {int(best['Odds']):+d}</b>
                  <span>
                    Win {best['Model Prob']*100:.1f}% • Edge {best['Edge']*100:+.1f}% •
                    EV {best['EV']*100:+.1f}% • Fair {int(best['Model Fair']):+d}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Edited-price test only. The automatic live-market recommendation remains the primary view above.")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Result CSV", data=csv, file_name=f"mlb_model_{away.replace(' ', '_')}_at_{home.replace(' ', '_')}.csv", mime="text/csv")

    else:
        st.divider()

        payload = st.session_state.get("general_odds_payload") or {}
        events = payload.get("events", [])
        market_error = payload.get("error", "")
        quota = payload.get("quota", {})

        st.markdown(
            f"""
            <div class="app-head">
              <div>
                <div class="app-eyebrow">MLB EDGE</div>
                <div class="app-head-title">Today's Slate</div>
                <div class="app-head-sub">Moneyline • Run Line • Totals • Market-calibrated probabilities</div>
              </div>
              <div class="app-live">● MODEL LIVE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not odds_api_key:
            st.warning("Automatic market feed is not configured. Add ODDS_API_KEY to Streamlit Secrets.")
        elif market_error:
            st.warning(f"General market unavailable: {market_error}")
        else:
            remaining = quota.get("remaining")
            cap = f" • API credits remaining: {remaining}" if remaining not in [None, ""] else ""
            st.caption(
                f"Consensus market loaded for {len(events)} MLB event(s){cap}. "
                "Lines are median current snapshots across available US books."
            )

        game_rows = []
        all_market_rows = []

        for _, r in df.iterrows():
            game_pk = int(r["GamePk"])
            market = market_for_game(games, events, game_pk)
            candidates = evaluate_game_markets(r, market)

            meta = next((g for g in games if int(g["GamePk"]) == game_pk), {})
            kickoff = meta.get("TimeLabel", "")
            game_date = meta.get("GameDate", "")

            if candidates:
                best = candidates[0]
                best_verdict = best["verdict"]
                best_market = best["market"]
            else:
                best = None
                best_verdict = "NO LINE"
                best_market = "No current market"

            row_obj = {
                "game_pk": game_pk,
                "kickoff_et": kickoff,
                "game_date": game_date,
                "away": str(r["Away"]),
                "home": str(r["Home"]),
                "away_proj": float(r["Away_Proj_Runs"]),
                "home_proj": float(r["Home_Proj_Runs"]),
                "model_total": float(r["Model_Total"]),
                "confidence": int(r["Model_Confidence"]),
                "data_status": str(r.get("Data_Status", "")),
                "market": market,
                "candidates": candidates,
                "best": best,
                "best_verdict": best_verdict,
                "best_market": best_market,
            }
            game_rows.append(row_obj)

            for c in candidates:
                all_market_rows.append({
                    "GamePk": game_pk,
                    "Game": f"{r['Away']} @ {r['Home']}",
                    "Kickoff_ET": kickoff,
                    "Market": c["market"],
                    "Market_Type": c["market_type"],
                    "Odds": c["odds"],
                    "Internal_Grade": grade_meta(c["verdict"])[0],
                    "Verdict": user_verdict(c["verdict"]),
                    "Raw_Model_Prob": c["raw_prob"],
                    "Market_NoVig_Prob": c["market_prob"],
                    "Calibrated_Prob": c["calibrated_prob"],
                    "Edge": c["edge"],
                    "EV": c["ev"],
                    "Fair_Odds": c["fair"],
                    "Confidence": int(r["Model_Confidence"]),
                    "Rank_Score": c["rank_score"],
                })

        # One official bet per game for the quick card.
        official_games = [
            g for g in game_rows
            if g["best"] is not None and g["best"]["verdict"] in {"STRONG BET", "BET"}
        ]
        official_games.sort(
            key=lambda g: g["best"]["rank_score"],
            reverse=True,
        )

        lean_games = [
            g for g in game_rows
            if g["best"] is not None and g["best"]["verdict"] == "LEAN"
        ]
        lean_games.sort(key=lambda g: g["best"]["rank_score"], reverse=True)

        top_n = st.radio(
            "Card size",
            [5, 10],
            horizontal=True,
            index=0,
            format_func=lambda n: f"Top {n}",
            label_visibility="collapsed",
            key="mlb_top_n",
        )
        show_n = int(top_n or 5)

        st.markdown('<div class="section-kicker">SLATE BETTING CARD</div>', unsafe_allow_html=True)
        st.caption(
            "Top Bets shows only official Best Bet / Bet plays. Moneylines, run lines and totals all compete, ranked by win probability, edge, EV and model confidence."
        )

        def render_top_bet(g, rank):
            b = g["best"]
            label = user_verdict(b["verdict"])
            cls = verdict_class(b["verdict"])
            odds_txt = f"{int(b['odds']):+d}"
            logo_team = g["away"] if b["market"].startswith(g["away"]) else g["home"]
            if b["market_type"] == "TOTAL":
                logo = (
                    '<div style="display:flex;gap:2px">'
                    + logo_html(g["away"], 30)
                    + logo_html(g["home"], 30)
                    + '</div>'
                )
            else:
                logo = logo_html(logo_team, 32)

            st.markdown(
                f"""
                <div class="topbet-card {cls}">
                  <div class="topbet-rank">#{rank}</div>
                  <div>{logo}</div>
                  <div>
                    <div class="topbet-game">{html.escape(g['kickoff_et'])} ET • {html.escape(g['away'])} @ {html.escape(g['home'])}</div>
                    <div class="topbet-pick">{html.escape(b['market'])} {odds_txt}</div>
                    <div class="topbet-note">{b['market_type']} • Confidence {g['confidence']}/100</div>
                  </div>
                  <div class="topbet-grade {cls}">{label}</div>
                  <div class="topbet-metrics">
                    <div><span>Win Chance</span><b>{b['calibrated_prob']*100:.1f}%</b></div>
                    <div><span>Edge</span><b>{b['edge']*100:+.1f}%</b></div>
                    <div><span>EV</span><b>{b['ev']*100:+.1f}%</b></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if official_games:
            selected = official_games[:show_n]
            for i, g in enumerate(selected, 1):
                render_top_bet(g, i)
            if len(selected) < show_n:
                st.caption(
                    f"Only {len(selected)} official bet(s) qualify today. The card is not padded with weaker leans."
                )
        else:
            st.info("No official bets currently qualify. The model will not manufacture a Top 5.")

        if lean_games:
            with st.expander(f"Next Best Leans • {min(show_n, len(lean_games))}", expanded=False):
                for i, g in enumerate(lean_games[:show_n], 1):
                    render_top_bet(g, i)

        best_n = sum(1 for g in game_rows if g["best"] is not None and g["best"]["verdict"] == "STRONG BET")
        bet_n = sum(1 for g in game_rows if g["best"] is not None and g["best"]["verdict"] == "BET")
        lean_n = sum(1 for g in game_rows if g["best"] is not None and g["best"]["verdict"] == "LEAN")
        st.caption(f"Slate pool: {best_n} Best Bet • {bet_n} Bet • {lean_n} Lean")

        # Chronological game navigation.
        st.markdown('<div class="section-kicker">ALL GAMES</div>', unsafe_allow_html=True)
        st.caption(
            "Games are in start-time order. Open a matchup to see the best market first, then every ML, run line and total ranked underneath."
        )

        def game_sort_key(g):
            try:
                x = pd.to_datetime(g.get("game_date"), utc=True, errors="coerce")
                if pd.isna(x):
                    return pd.Timestamp.max.tz_localize("UTC")
                return x
            except Exception:
                return pd.Timestamp.max.tz_localize("UTC")

        for g in sorted(game_rows, key=game_sort_key):
            best_grade = grade_meta(g["best_verdict"])[0]
            with st.expander(
                f"{g['kickoff_et']} • {g['away']} @ {g['home']}",
                expanded=False,
            ):
                st.markdown(
                    f"""
                    <div class="game-head">
                      <div class="game-team">
                        {logo_html(g['away'], 34)}
                        <div><span>Away</span><b>{html.escape(g['away'])}</b></div>
                      </div>
                      <div class="game-at">@</div>
                      <div class="game-team home">
                        <div><span>Home</span><b>{html.escape(g['home'])}</b></div>
                        {logo_html(g['home'], 34)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"Projected score {g['away_proj']:.1f}–{g['home_proj']:.1f} • "
                    f"Model total {g['model_total']:.1f} • Confidence {g['confidence']}/100"
                )

                source_summary = market_source_summary(g["market"])
                if source_summary:
                    st.caption(source_summary)

                if not g["candidates"]:
                    st.info("No current consensus market matched this game.")
                else:
                    for i, c in enumerate(g["candidates"], 1):
                        label = user_verdict(c["verdict"])
                        cls = verdict_class(c["verdict"])
                        st.markdown(
                            f"""
                            <div class="game-market-row">
                              <div class="game-market-rank">#{i}</div>
                              <div class="game-market-grade {cls}">{label}</div>
                              <div>
                                <div class="game-market-pick">{html.escape(c['market'])} {int(c['odds']):+d}</div>
                                <div class="game-market-meta">
                                  {c['market_type']} • Win {c['calibrated_prob']*100:.1f}% •
                                  Edge {c['edge']*100:+.1f}% • EV {c['ev']*100:+.1f}% • Fair {int(c['fair']):+d}
                                </div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with st.expander("Model vs Market detail", expanded=False):
                        detail = pd.DataFrame([
                            {
                                "Market": c["market"],
                                "Type": c["market_type"],
                                "Raw Model %": round(c["raw_prob"]*100, 1),
                                "Market No-Vig %": round(c["market_prob"]*100, 1),
                                "Calibrated %": round(c["calibrated_prob"]*100, 1),
                                "Edge %": round(c["edge"]*100, 1),
                                "EV %": round(c["ev"]*100, 1),
                                "Verdict": user_verdict(c["verdict"]),
                            }
                            for c in g["candidates"]
                        ])
                        st.dataframe(detail, use_container_width=True, hide_index=True)

        with st.expander("Export / Audit", expanded=False):
            if all_market_rows:
                market_level_df = pd.DataFrame(all_market_rows).sort_values(
                    ["Rank_Score"], ascending=[False]
                )
                st.dataframe(market_level_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download Ranked Market CSV",
                    data=market_level_df.to_csv(index=False).encode("utf-8"),
                    file_name="mlb_model_v090_ranked_markets.csv",
                    mime="text/csv",
                )

            st.download_button(
                "⬇️ Download Projection CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="mlb_model_v090_projections.csv",
                mime="text/csv",
            )

        st.caption(
            "Decision layer v0.9 uses conservative market-aware probability shrinkage. User-facing grades are Best Bet / Bet / Lean / Pass. "
            "The current consensus is a stabilizing prior, not a closing line. "
            "Run-line and total probabilities remain less proven than moneyline probabilities."
        )

st.divider()
st.caption("General market lines are median current consensus snapshots across available US books. Decision v0.9 shrinks raw model probabilities toward the no-vig market before grading. Model outputs are analytical estimates, not guarantees.")
