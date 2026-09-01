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

APP_VERSION = "1.3.0-FORWARD-TRACKER"
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
    """Return one of PREGAME / LIVE / FINAL / OTHER using MLB schedule status first."""
    if not game:
        return "OTHER"
    abstract = str(game.get("AbstractGameState") or "").strip().lower()
    detailed = str(game.get("DetailedState") or "").strip().lower()
    code = str(game.get("StatusCode") or "").strip().upper()

    if abstract == "live" or any(x in detailed for x in ("in progress","manager challenge","review","warmup")):
        return "LIVE"
    if abstract == "final" or any(x in detailed for x in ("final","completed","game over")):
        return "FINAL"

    # MLB codes that are normally pregame/scheduled states.
    if abstract == "preview" or code in {"S","P"} or any(x in detailed for x in ("scheduled","pre-game","pregame","delayed start")):
        return "PREGAME"

    # Time fallback protects us if MLB status is stale/missing.
    try:
        start = pd.to_datetime(game.get("GameDate"), utc=True)
        now = pd.Timestamp.now(tz="UTC")
        # Once first pitch time has arrived, never expose a pregame betting grade
        # unless MLB explicitly still reports a preview/delayed-start state.
        if now >= start:
            return "LIVE"
        return "PREGAME"
    except Exception:
        return "OTHER"

def is_pregame(game):
    return game_state(game) == "PREGAME"

def game_state_label(game):
    state = game_state(game)
    if state == "LIVE":
        return "LIVE — betting recommendations disabled"
    if state == "FINAL":
        return "FINAL — betting recommendations disabled"
    if state == "PREGAME":
        return "PREGAME"
    return "STATUS UNKNOWN"

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

def track_current_official_recommendations(candidates, games, slate_date):
    """Freeze the first official ML signal for each game. Never overwrite later line moves."""
    game_map = _game_lookup(games)
    added = 0
    for x in candidates:
        if not x.get("pregame") or not x.get("market_available"):
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
        if not x.get("pregame"):
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


st.markdown(f"""
<div class="hero">
  <div class="eyebrow">MLB EDGE • PRODUCTION</div>
  <div class="title">MLB Edge</div>
  <div class="sub">Moneyline and totals now run as separate production markets. Totals use the validated pitcher + run-environment core with an 80% model / 20% market blend, conservative edge tiers, and a maximum of three official totals plays.</div>
  <div class="pill">MODEL LIVE • {APP_VERSION}</div>
</div>
""", unsafe_allow_html=True)

try:
    api_key=st.secrets.get("ODDS_API_KEY","")
except Exception:
    api_key=""

st.markdown('<div class="kicker">Slate Controls</div>', unsafe_allow_html=True)
ctrl1,ctrl2,ctrl3=st.columns([3,2,2])
with ctrl1:
    slate_date=st.date_input("Slate date",value=today_et(),min_value=today_et(),max_value=today_et()+timedelta(days=14),help="Current/upcoming MLB dates only.")
with ctrl2:
    st.caption("Moneyline market")
    load_market=st.button("Load Full Slate ML Odds",use_container_width=True)
with ctrl3:
    st.caption("Totals market")
    load_totals_market=st.button("Load Full Slate Totals Odds",use_container_width=True)
st.caption("Nothing calls The Odds API automatically. Moneyline and totals are separate manual pulls so you control credits.")
st.caption("Pregame only: once MLB marks a game live/final (or scheduled first-pitch time has passed), its odds are removed from BET/LEAN cards and single-game odds buttons are disabled.")

free_refresh=st.button("Refresh MLB schedule/model data (free)",use_container_width=True)
if free_refresh:
    st.cache_data.clear()
    st.rerun()

if "odds_payload" not in st.session_state:
    st.session_state.odds_payload={"events":[],"error":"","quota":{}}
    st.session_state.odds_loaded=False
    st.session_state.odds_loaded_at=None
    st.session_state.odds_scope=None

if "totals_payload" not in st.session_state:
    st.session_state.totals_payload={"events":[],"error":"","quota":{}}
    st.session_state.totals_loaded=False
    st.session_state.totals_scope=None

if load_market:
    fetch_odds.clear()
    st.session_state.odds_payload=fetch_odds(api_key)
    st.session_state.odds_loaded=True
    st.session_state.odds_loaded_at=pd.Timestamp.now(tz="America/New_York")
    st.session_state.odds_scope="full slate"

if load_totals_market:
    fetch_full_slate_totals.clear()
    st.session_state.totals_payload=fetch_full_slate_totals(api_key)
    st.session_state.totals_loaded=True
    st.session_state.totals_scope="full slate totals"

odds_payload=st.session_state.odds_payload if st.session_state.get("odds_loaded") else {"events":[],"error":"","quota":{}}
totals_payload=st.session_state.totals_payload if st.session_state.get("totals_loaded") else {"events":[],"error":"","quota":{}}

with st.spinner("Loading MLB schedule, starters, lineups and model data…"):
    games=fetch_games_for_date(slate_date)
    model_df=run_model(games) if games else pd.DataFrame()
    candidates=build_candidates(model_df,games,odds_payload.get("events",[])) if not model_df.empty else []

# Forward-test tracker: freeze the first official recommendation at the price that triggered it.
_new_ml = track_current_official_recommendations(candidates, games, slate_date)
_new_totals = track_current_total_recommendations(candidates, games, model_df, totals_payload, slate_date)
_graded_now = grade_tracker(force=False)
if _new_ml or _new_totals:
    st.toast(f"Tracked {_new_ml + _new_totals} new official model recommendation(s).")
if _graded_now:
    st.toast(f"Auto-graded {_graded_now} completed recommendation(s).")

quota=odds_payload.get("quota",{})
if st.session_state.get("odds_loaded"):
    qtxt=f"Odds credits remaining: {quota.get('remaining')}" if quota.get("remaining") is not None else "Live odds loaded manually"
else:
    qtxt="Market not loaded • 0 Odds API credits used"
priced_games=sum(1 for x in candidates if x.get("market_available") and x.get("pregame"))
pregame_games=sum(1 for x in candidates if x.get("pregame"))
started_games=sum(1 for x in candidates if x.get("game_state") in ("LIVE","FINAL"))
st.markdown(f'<div class="status"><div><span class="dot"></span><span class="live">MODEL LIVE</span> &nbsp; {slate_date.strftime("%b %-d")} • {pregame_games} pregame • {priced_games} priced • {started_games} started/final</div><div>{qtxt}</div></div>',unsafe_allow_html=True)

if odds_payload.get("error"):
    st.error(odds_payload["error"])

if not st.session_state.get("odds_loaded"):
    st.info("Model-only mode. In **Single Game**, choose a matchup and load only that game's odds. In **Full Slate**, use the full-slate button above. No Odds API call has been made yet.")

if not games:
    st.info(f"No MLB games were returned for {slate_date.strftime('%B %-d, %Y')}.")
    st.stop()

if not candidates:
    st.warning("The model could not produce game rows for today.")
else:
    st.markdown('<div class="kicker">Forward Performance</div>', unsafe_allow_html=True)
    tracker_df = load_tracker()
    perf = tracker_performance_summary(tracker_df)
    perf_record = f'{perf["wins"]}-{perf["losses"]}' + (f'-{perf["pushes"]}P' if perf["pushes"] else "")
    with st.expander(
        f'📈 Model Tracker • {perf_record} • {perf["units"]:+.2f}u • {perf["pending"]} pending',
        expanded=False,
    ):
        if tracker_df.empty:
            st.info("No official BET/BEST BET recommendations have been frozen yet. The tracker logs them automatically the first time they become official.")
        else:
            record_display = f'{perf["wins"]}-{perf["losses"]}' + (f'-{perf["pushes"]}P' if perf["pushes"] else "")
            st.markdown(
                f'<div class="metrics"><div class="metric"><span>Record</span><b>{record_display}</b></div><div class="metric"><span>Units</span><b>{perf["units"]:+.2f}</b></div><div class="metric"><span>ROI</span><b>{perf["roi"]*100:+.1f}%</b></div><div class="metric"><span>Pending</span><b>{perf["pending"]}</b></div></div>',
                unsafe_allow_html=True,
            )
            split = tracker_split_table(tracker_df)
            if not split.empty:
                st.dataframe(split, use_container_width=True, hide_index=True)

            graded = tracker_df[tracker_df["Result"].isin(["WIN","LOSS","PUSH"])].copy()
            if not graded.empty:
                graded["Cum_Units"] = pd.to_numeric(graded["Units"], errors="coerce").fillna(0).cumsum()
                st.line_chart(graded[["Cum_Units"]], use_container_width=True)

            show_cols = ["Slate_Date","Game","Market","Pick","Odds","Grade","Edge","EV","Result","Units","App_Version"]
            recent = tracker_df.sort_values(["Slate_Date","Logged_At_ET"], ascending=False)
            st.dataframe(recent[[c for c in show_cols if c in recent.columns]].head(50), use_container_width=True, hide_index=True)

        c1,c2=st.columns(2)
        with c1:
            if st.button("Refresh Results Now (free)", use_container_width=True, key="tracker_refresh_results"):
                tracker_results_for_date.clear()
                n = grade_tracker(force=True)
                st.success(f"Updated {n} completed recommendation(s)." if n else "No new finals to grade yet.")
                st.rerun()
        with c2:
            tracker_df = load_tracker()
            st.download_button(
                "Download Tracker Backup CSV",
                data=tracker_df.to_csv(index=False).encode("utf-8"),
                file_name="mlb_model_recommendation_tracker.csv",
                mime="text/csv",
                use_container_width=True,
                key="tracker_backup_download",
            )

        st.caption("Official BET/BEST BET signals are frozen automatically at the first qualifying price. Refreshes never overwrite them. MLB final scores are graded with the free MLB Stats API, not The Odds API.")

        restore_file = st.file_uploader("Restore / merge tracker backup", type=["csv"], key="tracker_restore_upload")
        if st.button("Merge Tracker Backup", use_container_width=True, disabled=(restore_file is None), key="tracker_restore_btn"):
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

        st.markdown("**Backfill an earlier model recommendation**")
        diag_file = st.file_uploader("Import a downloaded Game Diagnostics CSV", type=["csv"], key="tracker_diag_import")
        if st.button("Import Diagnostics Pick", use_container_width=True, disabled=(diag_file is None), key="tracker_diag_btn"):
            n,msg = import_diagnostics_tracker(diag_file)
            if n:
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

        if st.session_state.get("_tracker_storage_error"):
            st.warning("Tracker is currently saved in this app session, but local file persistence failed. Download a backup CSV before leaving the app.")
        else:
            st.caption("Storage note: the app also writes a local tracker file. Streamlit Cloud can replace its runtime filesystem during redeploys, so the downloadable backup is the safest long-term copy.")

    st.markdown('<div class="kicker">Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "View mode",
        ["🎯 Single Game", "📋 Full Slate"],
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

    if mode == "🎯 Single Game":
        chrono = sorted(candidates, key=start_sort)
        labels = [f"{x['time']} • {x['away']} @ {x['home']}" + (f" • {x['game_state']}" if not x.get("pregame") else "") for x in chrono]
        selected_label = st.selectbox("Choose matchup", labels, index=0, key="single_game_matchup")
        x = chrono[labels.index(selected_label)]
        selected_game = next((g for g in games if g.get("GamePk") == x["GamePk"]), None)

        st.caption("Selecting a matchup does **not** call the odds API. Tap below only when you want this game's live moneyline.")
        selected_state = game_state(selected_game)
        if selected_state != "PREGAME":
            st.warning(game_state_label(selected_game) + ". Historical/pregame prices are not shown as actionable live bets.")
        pull_single = st.button(
            "Load / Refresh Odds for This Game",
            use_container_width=True,
            type="primary",
            disabled=(selected_state != "PREGAME"),
        )
        if pull_single:
            payload = fetch_single_game_odds(api_key, selected_game)
            st.session_state.odds_payload = payload
            st.session_state.odds_loaded = True
            st.session_state.odds_loaded_at = pd.Timestamp.now(tz="America/New_York")
            st.session_state.odds_scope = f"single game: {x['away']} @ {x['home']}"
            st.rerun()

        if st.session_state.get("odds_loaded") and st.session_state.get("odds_scope"):
            st.caption(f"Loaded odds scope: **{st.session_state.odds_scope}**")

        b = x["best"]
        away_side = next(z for z in x["all"] if z["team"] == x["away"])
        home_side = next(z for z in x["all"] if z["team"] == x["home"])
        lineup_text = "Confirmed lineups" if x["lineup_confirmed"] else "Lineups not fully confirmed"

        st.markdown('<div class="kicker">Single Game Analysis</div>', unsafe_allow_html=True)
        if x["market_available"]:
            st.markdown(f'''<div class="best-card"><div class="best-top"><div><div class="best-tag">{b['selection']}</div><div class="best-pick">{b['team']} ML {b['odds']:+d}</div><div class="best-game">{x['away']} @ {x['home']} • {x['time']} • Best price: {b['book']}</div></div><div class="badge {cls(b['selection'])}">{b['selection']}</div></div><div class="metrics"><div class="metric"><span>Win chance</span><b>{b['prob']*100:.1f}%</b></div><div class="metric"><span>Edge vs price</span><b>{b['edge']*100:+.1f}%</b></div><div class="metric"><span>EV</span><b>{b['ev']*100:+.1f}%</b></div><div class="metric"><span>Fair line</span><b>{b['fair']:+d}</b></div></div><div class="best-game" style="margin-top:10px">{lineup_text} • Model weight {x['alpha']*100:.0f}% / market {(1-x['alpha'])*100:.0f}% • {x['books']} books in consensus</div></div>''', unsafe_allow_html=True)
        else:
            fav = away_side if away_side['prob'] >= home_side['prob'] else home_side
            st.markdown(f'''<div class="best-card"><div class="best-top"><div><div class="best-tag">MODEL VIEW</div><div class="best-pick">{fav['team']} {fav['prob']*100:.1f}%</div><div class="best-game">{x['away']} @ {x['home']} • {x['time']} • Live moneyline not available</div></div><div class="badge badge-lean">MODEL ONLY</div></div><div class="metrics"><div class="metric"><span>{x['away']} win</span><b>{away_side['prob']*100:.1f}%</b></div><div class="metric"><span>{x['home']} win</span><b>{home_side['prob']*100:.1f}%</b></div><div class="metric"><span>{x['away']} fair</span><b>{away_side['fair']:+d}</b></div><div class="metric"><span>{x['home']} fair</span><b>{home_side['fair']:+d}</b></div></div><div class="best-game" style="margin-top:10px">{lineup_text} • Model confidence {x['confidence']}/100 • No BET/LEAN verdict without a live price</div></div>''', unsafe_allow_html=True)
            st.info("This game is modeled and selectable. A betting verdict appears automatically when a valid two-way moneyline is available.")

        st.markdown('<div class="kicker">Matchup Detail</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="single-summary"><div class="match">{x['away']} @ {x['home']}</div><div class="sp">{x['away_sp']} vs {x['home_sp']}</div><div class="detail-grid"><div class="detail"><span>{x['away']} win</span><b>{away_side['prob']*100:.1f}%</b></div><div class="detail"><span>{x['home']} win</span><b>{home_side['prob']*100:.1f}%</b></div><div class="detail"><span>Projected runs</span><b>{x['away_proj']:.2f} – {x['home_proj']:.2f}</b></div><div class="detail"><span>{x['away']} fair ML</span><b>{away_side['fair']:+d}</b></div><div class="detail"><span>{x['home']} fair ML</span><b>{home_side['fair']:+d}</b></div><div class="detail"><span>Model confidence</span><b>{x['confidence']}/100</b></div></div></div>''', unsafe_allow_html=True)

        st.markdown('<div class="kicker">Both Sides</div>', unsafe_allow_html=True)
        for side in [away_side, home_side]:
            if x['market_available']:
                headline=f"{side['team']} ML {side['odds']:+d}"
                subtitle=f"{side['book']} • Fair {side['fair']:+d}"
                details=f'''<div class="metrics"><div class="metric"><span>Win chance</span><b>{side['prob']*100:.1f}%</b></div><div class="metric"><span>Market no-vig</span><b>{side['market_prob']*100:.1f}%</b></div><div class="metric"><span>Edge</span><b>{side['edge']*100:+.1f}%</b></div><div class="metric"><span>EV</span><b>{side['ev']*100:+.1f}%</b></div></div>'''
            else:
                headline=side['team']
                subtitle=f"Fair line {side['fair']:+d} • Waiting for market"
                details=f'''<div class="metrics"><div class="metric"><span>Model win chance</span><b>{side['prob']*100:.1f}%</b></div><div class="metric"><span>Fair line</span><b>{side['fair']:+d}</b></div></div>'''
            st.markdown(f'''<div class="game-card"><div class="game-head"><div><div class="pick-main">{headline}</div><div class="pick-sub">{subtitle}</div></div><div class="badge {cls(side['selection'])}">{side['selection']}</div></div>{details}</div>''', unsafe_allow_html=True)

        if x['market_available']:
            st.caption(f"{x['lineup_status']}. Calibration: {x['alpha']*100:.0f}% model / {(1-x['alpha'])*100:.0f}% market.")
        else:
            st.caption(f"{x['lineup_status']}. Model-only projection shown until a valid live moneyline is available.")
        if x.get("confidence_reasons"):
            st.caption(f"Data notes: {x['confidence_reasons']}")

        st.markdown('<div class="kicker">Totals</div>', unsafe_allow_html=True)
        st.caption("Production totals are separate from moneyline. The market is only pulled when you tap the button below.")
        pull_total=st.button(
            "Load / Refresh Total for This Game",
            use_container_width=True,
            key=f"pull_total_{x['GamePk']}",
            disabled=(game_state(selected_game) != "PREGAME"),
        )
        if pull_total:
            st.session_state.totals_payload=fetch_single_game_totals(api_key,selected_game)
            st.session_state.totals_loaded=True
            st.session_state.totals_scope=f"single game total: {x['away']} @ {x['home']}"
            st.rerun()

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
            st.markdown(f'''<div class="best-card"><div class="best-top"><div><div class="best-tag">TOTALS MODEL VIEW</div><div class="best-pick">Projected total {raw_total:.2f}</div><div class="best-game">Load this game's total only when you want an official market grade.</div></div><div class="badge badge-lean">MODEL ONLY</div></div><div class="metrics"><div class="metric"><span>Projected total</span><b>{raw_total:.2f}</b></div><div class="metric"><span>Park context</span><b>{float(tctx.get("Park_Factor",1.0)):.3f}</b></div><div class="metric"><span>Temperature</span><b>{temp_txt}</b></div><div class="metric"><span>Lineups</span><b>{"CONFIRMED" if x["lineup_confirmed"] else "WAIT"}</b></div></div></div>''',unsafe_allow_html=True)


        st.markdown('<div class="kicker">Totals Download</div>', unsafe_allow_html=True)
        total_download_row = totals_download_row(row_for_total, tctx, tp if tm and 'tp' in locals() else None)
        total_download_df = pd.DataFrame([total_download_row])
        st.download_button(
            "Download This Game's Totals CSV",
            data=total_download_df.to_csv(index=False).encode("utf-8"),
            file_name=f"mlb_game_totals_{x['GamePk']}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"download_total_{x['GamePk']}",
        )
        st.caption("This download does not call The Odds API. It exports the totals data already loaded for this game.")

        st.markdown('<div class="kicker">Game Data</div>', unsafe_allow_html=True)
        if "show_single_downloads" not in st.session_state:
            st.session_state.show_single_downloads = False
        if not st.session_state.show_single_downloads:
            if st.button("Open Game Downloads", key="open_single_downloads", use_container_width=True):
                st.session_state.show_single_downloads = True
                st.rerun()
        else:
            diag = game_diagnostics_df(x, slate_date)
            st.caption("Download the selected matchup's model inputs and outputs. Send this CSV to ChatGPT when you want a detailed 'why the model likes this side' breakdown.")
            st.download_button(
                "Download Selected Game Diagnostics CSV",
                diag.to_csv(index=False).encode("utf-8"),
                file_name=f"mlb_game_diagnostics_{x['GamePk']}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"download_diag_{x['GamePk']}",
            )
            if st.button("Close Downloads", key="close_single_downloads", use_container_width=True):
                st.session_state.show_single_downloads = False
                st.rerun()

    else:
        official=sorted([x for x in candidates if x["market_available"] and x["best"].get("selection") in ("BEST BET","BET")], key=lambda x:x["best"].get("smart_score",-999), reverse=True)[:5]
        secondary=sorted([x for x in candidates if x["market_available"] and x["best"].get("selection")=="LEAN"], key=lambda x:x["best"].get("smart_score",-999), reverse=True)
        priced=[x for x in candidates if x["market_available"]]

        if not priced:
            st.markdown('<div class="kicker">Today\'s Model View</div>',unsafe_allow_html=True)
            st.info("Live odds are not loaded. The slate remains fully selectable in model-only mode; load full-slate odds only when you want priced recommendations.")
        else:
            st.markdown('<div class="kicker">Today\'s Plays</div>',unsafe_allow_html=True)
            plays=official + secondary
            if not plays:
                st.info("No game currently reaches the 5% LEAN threshold. The model is passing the slate at these prices.")
            else:
                st.caption(f"{len(official)} official bet{'s' if len(official)!=1 else ''} • {len(secondary)} lean{'s' if len(secondary)!=1 else ''}. BEST BET starts at 10% edge, BET at 7.5%, LEAN at 5%. No minimum number of bets is forced.")
                rank=1
                for x in plays[:8]:
                    b=x["best"]
                    is_official=b["selection"] in ("BEST BET","BET")
                    rank_label=f"#{rank}" if is_official else "WATCH"
                    if is_official:
                        rank+=1
                    st.markdown(f'''<div class="game-card"><div class="game-head"><div><div class="game-time">{rank_label} • {x['time']}</div><div class="match">{x['away']} @ {x['home']}</div><div class="sp">{x['away_sp']} vs {x['home_sp']}</div></div><div class="badge {cls(b['selection'])}">{b['selection']}</div></div><div class="pick"><div><div class="pick-main">{b['team']} ML {b['odds']:+d}</div><div class="pick-sub">{b['book']} • Edge {b['edge']*100:+.1f}% • EV {b['ev']*100:+.1f}% • Fair {b['fair']:+d} • Win {b['prob']*100:.1f}%</div></div><div class="{'lineup-ok' if x['lineup_confirmed'] else 'lineup-wait'}" style="font-size:.60rem;font-weight:900">{'LINEUPS ✓' if x['lineup_confirmed'] else 'LINEUPS WAIT'}</div></div></div>''',unsafe_allow_html=True)

        other_games=[x for x in sorted(candidates,key=start_sort) if x["best"].get("selection") not in ("BEST BET","BET","LEAN")]
        st.markdown('<div class="kicker">Other Games</div>',unsafe_allow_html=True)
        st.caption(f"{len(other_games)} game{'s' if len(other_games)!=1 else ''} currently outside the play list. Expand any matchup for the full model view.")
        for x in other_games:
            b=x["best"]
            with st.expander(f"{x['time']}  •  {x['away']} @ {x['home']}  —  {b['selection']}",expanded=False):
                if x['market_available']:
                    pick_html=f"<div class='pick'><div><div class='pick-main'>{b['team']} ML {b['odds']:+d}</div><div class='pick-sub'>Win {b['prob']*100:.1f}% • Edge {b['edge']*100:+.1f}% • EV {b['ev']*100:+.1f}% • Fair {b['fair']:+d} • {b['book']}</div></div></div>"
                    cap=f"Calibration: {x['alpha']*100:.0f}% model / {(1-x['alpha'])*100:.0f}% market."
                else:
                    away_side=next(z for z in x['all'] if z['team']==x['away'])
                    home_side=next(z for z in x['all'] if z['team']==x['home'])
                    pick_html=f"<div class='pick'><div><div class='pick-main'>Model: {x['away']} {away_side['prob']*100:.1f}% • {x['home']} {home_side['prob']*100:.1f}%</div><div class='pick-sub'>Fair lines {x['away']} {away_side['fair']:+d} • {x['home']} {home_side['fair']:+d} • live moneyline unavailable</div></div></div>"
                    cap="Model-only projection; no betting verdict until a valid live market is available."
                st.markdown(f'''<div class="game-card" style="margin-top:0"><div class="game-head"><div><div class="match">{x['away']} @ {x['home']}</div><div class="sp">{x['away_sp']} vs {x['home_sp']}</div></div><div class="badge {cls(b['selection'])}">{b['selection']}</div></div>{pick_html}</div>''',unsafe_allow_html=True)
                st.caption(f"Model projected runs: {x['away']} {x['away_proj']:.2f} — {x['home']} {x['home_proj']:.2f}. {cap} {x['lineup_status']}.")
                if x.get("confidence_reasons"):
                    st.caption(f"Data notes: {x['confidence_reasons']}")
        st.markdown('<div class="kicker">Totals Plays</div>', unsafe_allow_html=True)
        if not st.session_state.get("totals_loaded"):
            st.info("Totals odds are not loaded. Use **Load Full Slate Totals Odds** only when you want priced totals recommendations.")
        else:
            total_rows=[]
            for cx in sorted(candidates,key=start_sort):
                if not cx.get("pregame"):
                    continue
                mr=model_df.loc[model_df["GamePk"]==cx["GamePk"]]
                if mr.empty:
                    continue
                ctx=engine.totals_projection(mr.iloc[0].to_dict()) if hasattr(engine,"totals_projection") else {"Projected_Total":cx['away_proj']+cx['home_proj']}
                game_obj=next((g for g in games if g.get("GamePk")==cx["GamePk"]),None)
                ev=match_event(totals_payload.get("events",[]),game_obj) if game_obj else None
                tm=totals_market(ev)
                if not tm:
                    continue
                tp=build_total_pick(float(ctx["Projected_Total"]),tm)
                if tp is None:
                    continue
                total_rows.append((cx,tp,ctx))

            official_totals=sorted(
                [z for z in total_rows if z[1]["grade"] in ("BEST BET","BET")],
                key=lambda z:(z[1]["edge"],z[1]["ev"]),
                reverse=True
            )[:TOTALS_MAX_OFFICIAL]
            lean_totals=sorted(
                [z for z in total_rows if z[1]["grade"]=="LEAN"],
                key=lambda z:(z[1]["edge"],z[1]["ev"]),
                reverse=True
            )

            st.caption(f"Maximum {TOTALS_MAX_OFFICIAL} official totals plays. BEST BET starts at 12.5% edge, BET at 7.5%, LEAN at 5%. No minimum number of bets is forced.")
            if not official_totals and not lean_totals:
                st.info("No total currently reaches the 5% LEAN threshold.")
            else:
                rank=1
                for cx,tp,ctx in official_totals + lean_totals[:5]:
                    official=tp["grade"] in ("BEST BET","BET")
                    label=f"#{rank}" if official else "WATCH"
                    if official:
                        rank+=1
                    st.markdown(f'''<div class="game-card"><div class="game-head"><div><div class="game-time">{label} • {cx["time"]}</div><div class="match">{cx["away"]} @ {cx["home"]}</div><div class="sp">{cx["away_sp"]} vs {cx["home_sp"]}</div></div><div class="badge {cls(tp["grade"])}">{tp["grade"]}</div></div><div class="pick"><div><div class="pick-main">{tp["side"]} {tp["market_total"]:.1f} {tp["odds"]:+d}</div><div class="pick-sub">{tp["book"]} • Edge {tp["edge"]*100:+.1f}% • EV {tp["ev"]*100:+.1f}% • Raw model {float(ctx["Projected_Total"]):.2f} • Calibrated {tp["calibrated_total"]:.2f}</div></div><div class="{"lineup-ok" if cx["lineup_confirmed"] else "lineup-wait"}" style="font-size:.60rem;font-weight:900">{"LINEUPS ✓" if cx["lineup_confirmed"] else "LINEUPS WAIT"}</div></div></div>''',unsafe_allow_html=True)

            other_totals=[z for z in total_rows if z[1]["grade"]=="PASS"]
            with st.expander(f"Other totals — {len(other_totals)} PASS",expanded=False):
                for cx,tp,ctx in other_totals:
                    st.write(f'{cx["time"]} • {cx["away"]} @ {cx["home"]} — {tp["side"]} {tp["market_total"]:.1f} • edge {tp["edge"]*100:+.1f}% • model {float(ctx["Projected_Total"]):.2f}')


        if st.session_state.get("totals_loaded"):
            totals_export_rows = []
            for cx in sorted(candidates, key=start_sort):
                if not cx.get("pregame"):
                    continue
                mr = model_df.loc[model_df["GamePk"] == cx["GamePk"]]
                if mr.empty:
                    continue
                row_dict = mr.iloc[0].to_dict()
                ctx = engine.totals_projection(row_dict) if hasattr(engine, "totals_projection") else {
                    "Projected_Total": cx["away_proj"] + cx["home_proj"]
                }
                game_obj = next((g for g in games if g.get("GamePk") == cx["GamePk"]), None)
                ev = match_event(totals_payload.get("events", []), game_obj) if game_obj else None
                tm = totals_market(ev)
                tp = build_total_pick(float(ctx["Projected_Total"]), tm) if tm else None
                totals_export_rows.append(totals_download_row(row_dict, ctx, tp))
            if totals_export_rows:
                totals_export_df = pd.DataFrame(totals_export_rows)
                st.download_button(
                    "Download Full Slate Totals CSV",
                    data=totals_export_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"mlb_totals_board_{slate_date.strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_full_totals_csv",
                )
                st.caption("Exports the totals board already in memory. No additional Odds API credits are used.")

        st.markdown('<div class="kicker">Downloads</div>', unsafe_allow_html=True)
        if "show_slate_downloads" not in st.session_state:
            st.session_state.show_slate_downloads = False
        if not st.session_state.show_slate_downloads:
            if st.button("Open Full Slate Downloads", key="open_slate_downloads", use_container_width=True):
                st.session_state.show_slate_downloads = True
                st.rerun()
        else:
            export_df = slate_export_df(candidates)
            st.download_button(
                "Download Today's Moneyline Board",
                export_df.to_csv(index=False).encode("utf-8"),
                file_name="mlb_production_moneyline_board.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_full_slate_board",
            )
            if st.button("Close Downloads", key="close_slate_downloads", use_container_width=True):
                st.session_state.show_slate_downloads = False
                st.rerun()

st.markdown('<div class="kicker">Model Guardrails</div>',unsafe_allow_html=True)
st.markdown("""
<div class="note"><b>Production scope:</b> moneyline + totals. Moneyline keeps the frozen starting-pitcher + offense/platoon + lineup engine. Totals use the same live baseball core, an 80% model / 20% market calibration informed by 2024 validation, and the historical residual spread from the totals audit. Bullpen remains excluded; park/weather are context only because park failed to improve the integrity audit. Run lines remain excluded. Diagnostics Download keeps the same edge-driven bet-selection thresholds and adds closable download panels plus selected-game diagnostic export: BEST BET starts at 10% edge, BET at 7.5%, LEAN at 5%, and PASS below 5%, while +200 or longer dogs remain materially stricter and require confirmed lineups for official status. EV remains visible and influences ranking but is not a separate hard gate. Odds API pulls are manual-only. Moneyline and totals are separate requests so you control which market consumes credits. Single Game and Full Slate each support explicit totals pulls. Totals grading is BEST BET at 12.5%+ edge, BET at 7.5%+, LEAN at 5%+, with at most three official totals plays per slate. Date changes, mode changes, matchup selection and model-only views consume zero odds credits.</div>
""",unsafe_allow_html=True)

with st.expander("Research basis & limitations",expanded=False):
    st.write("The research sequence rejected a team-only moneyline model, isolated a starting-pitcher signal through placebo/causality tests, improved it with Pitcher Model 2.0, rejected bullpen, and promoted offense/platoon plus lineup information. Totals then failed in the simple baseline, passed after adding pitcher + run-environment structure, and survived scrambled/no-starter integrity checks before this production promotion.")
    st.write("Historical results are research evidence, not a guarantee of forward profitability. The live engine also uses current data feeds and is not an exact replay of the historical PIT logistic model, so forward tracking remains necessary. v1.3 adds an automatic forward-performance ledger: first official BET/BEST BET signals are frozen before first pitch and later auto-graded from MLB final scores.")
    st.caption(f"App {APP_VERSION} • Engine {MODEL_VERSION}")
