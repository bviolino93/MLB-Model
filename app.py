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

import requests

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

APP_VERSION = "0.10.1-HISTORY-BUILDER"

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

def historical_dates(start_date, end_date):
    cur = start_date
    out = []
    while cur <= end_date:
        out.append(cur)
        cur += timedelta(days=1)
    return out

def historical_credit_estimate(start_date, end_date, markets, regions=("us",)):
    days = len(historical_dates(start_date, end_date))
    per_snapshot = 10 * max(1, len(markets)) * max(1, len(regions))
    return {"snapshots": days, "per_snapshot_max": per_snapshot, "max_credits": days * per_snapshot}

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
    st.caption("Use this only after activating a paid Historical Odds plan. Nothing below calls the API until you explicitly press Build Historical Market Dataset.")

    with st.expander("Build historical market dataset", expanded=True):
        hc1, hc2 = st.columns(2)
        with hc1:
            hist_start = st.date_input("Historical start", value=HISTORY_DEFAULT_START, min_value=date(2020,6,6), max_value=date.today(), key="hist_start")
        with hc2:
            hist_end = st.date_input("Historical end", value=HISTORY_DEFAULT_END, min_value=date(2020,6,6), max_value=date.today(), key="hist_end")
        hist_market_labels = st.multiselect("Historical markets", ["Moneyline","Run Line","Total"], default=["Moneyline","Run Line","Total"], key="hist_market_labels")
        market_map={"Moneyline":"h2h","Run Line":"spreads","Total":"totals"}
        hist_markets=[market_map[x] for x in hist_market_labels]
        if hist_end < hist_start:
            st.error("Historical end date must be on or after the start date.")
            hist_est={"snapshots":0,"per_snapshot_max":0,"max_credits":0}
        elif not hist_markets:
            st.warning("Select at least one market.")
            hist_est={"snapshots":0,"per_snapshot_max":0,"max_credits":0}
        else:
            hist_est=historical_credit_estimate(hist_start,hist_end,hist_markets)
        cached_dates=cached_history_manifest(hist_markets) if hist_markets else set()
        selected_dates=set(historical_dates(hist_start,hist_end)) if hist_end>=hist_start else set()
        cached_in_range=len(cached_dates & selected_dates)
        remaining_snapshots=max(0,hist_est["snapshots"]-cached_in_range)
        remaining_credit_ceiling=remaining_snapshots*hist_est["per_snapshot_max"]
        c1,c2,c3=st.columns(3)
        c1.metric("Calendar snapshots", f"{hist_est['snapshots']:,}")
        c2.metric("Max / new snapshot", f"{hist_est['per_snapshot_max']:,}")
        c3.metric("Remaining ceiling", f"{remaining_credit_ceiling:,}")
        st.caption(f"{cached_in_range:,} selected snapshots are already cached and will be skipped. This is a conservative ceiling; empty responses cost 0 and usage is based on markets actually returned.")
        hard_cap=st.number_input("Hard credit cap for this run", min_value=10, max_value=100000, value=min(20000,max(10,remaining_credit_ceiling if remaining_credit_ceiling else 10)), step=10, key="history_hard_cap")
        confirm_history=st.checkbox("I understand this button can consume paid Historical Odds credits.", key="confirm_history_spend")
        st.info("Credit guard: cached dates are skipped; one league-wide snapshot is requested per date; successful responses are cached immediately; the run stops before the hard cap is exceeded. Snapshot time is 15:00 UTC.")
        restore_zip=st.file_uploader("Restore prior historical cache (optional ZIP)", type=["zip"], key="history_cache_restore")
        if restore_zip is not None and st.button("Restore cache ZIP", key="restore_history_zip"):
            try:
                restored_n=restore_cache_zip(restore_zip)
                st.success(f"Restored {restored_n:,} cached snapshot files.")
            except Exception as e:
                st.error(f"Could not restore cache ZIP: {e}")
        run_history=st.button("Build Historical Market Dataset", type="primary", disabled=(not confirm_history or hist_end<hist_start or not hist_markets or remaining_snapshots==0), key="run_history_builder")
        if remaining_snapshots==0 and hist_est["snapshots"]>0:
            st.success("Every selected snapshot is already cached. No API call is needed.")
        if run_history:
            dates_to_fetch=[d for d in historical_dates(hist_start,hist_end) if d not in cached_dates]
            actual_last=None; remaining_hdr=None; estimated_spend=0; stopped_reason=None
            progress=st.progress(0); status=st.empty(); total_new=len(dates_to_fetch)
            for i,d in enumerate(dates_to_fetch,start=1):
                if estimated_spend + hist_est["per_snapshot_max"] > hard_cap:
                    stopped_reason=f"Stopped at the hard credit cap ({hard_cap:,})."; break
                status.caption(f"Fetching {d.isoformat()} • {i:,} of {total_new:,} new snapshots")
                payload,meta,err=fetch_historical_snapshot(d,hist_markets,force=False)
                if err:
                    stopped_reason=f"Stopped on {d.isoformat()}: {err}"; break
                estimated_spend += hist_est["per_snapshot_max"]
                actual_last=meta.get("last"); remaining_hdr=meta.get("remaining")
                progress.progress(min(i/max(1,total_new),1.0))
            all_rows=[]
            for d in historical_dates(hist_start,hist_end):
                cf=_history_cache_file(d,hist_markets)
                if not cf.exists(): continue
                try: all_rows.extend(flatten_historical_snapshot(json.loads(cf.read_text()),d))
                except Exception: continue
            progress.empty(); status.empty()
            if stopped_reason: st.warning(stopped_reason)
            if all_rows:
                hist_df=pd.DataFrame(all_rows)
                hist_df["Commence_Time"]=pd.to_datetime(hist_df["Commence_Time"],errors="coerce",utc=True)
                hist_df=hist_df.sort_values(["Snapshot_Date","Commence_Time","Away_Team","Home_Team"]).drop_duplicates(["Snapshot_Date","Event_ID"],keep="last")
                st.success(f"Historical market dataset ready: {len(hist_df):,} game rows across {hist_df['Snapshot_Date'].nunique():,} cached dates.")
                if remaining_hdr is not None:
                    st.caption(f"The Odds API reports {remaining_hdr} credits remaining. Last-request cost: {actual_last if actual_last is not None else '—'}.")
                st.dataframe(hist_df.head(100),use_container_width=True,hide_index=True)
                st.download_button("Download Historical Market CSV", hist_df.to_csv(index=False).encode("utf-8"), file_name=f"mlb_historical_market_{hist_start}_{hist_end}.csv", mime="text/csv", key="download_history_csv")
                st.download_button("Download Cache ZIP", build_cache_zip(), file_name=f"mlb_history_cache_{hist_start}_{hist_end}.zip", mime="application/zip", key="download_history_cache")
            else:
                st.info("No cached historical game rows are available yet.")
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
