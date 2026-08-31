import math
import re
import statistics
from statistics import median

import pandas as pd
import requests
import streamlit as st

from model import MODEL_VERSION, fetch_today_games, run_model, implied_prob, expected_value, fair_ml, TEAM_IDS, load_team_ids

APP_VERSION = "1.0.1-UX-RESTORE"
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
.single-summary{padding:13px 14px;border-radius:14px;background:rgba(15,32,53,.88);border:1px solid rgba(125,211,252,.14);margin:10px 0}.detail-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}.detail{padding:10px;border-radius:11px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05)}.detail span{display:block;font-size:.54rem;text-transform:uppercase;letter-spacing:.07em;color:#6f87a0;font-weight:900}.detail b{display:block;margin-top:3px;font-size:.82rem;color:#eef5fb}.stButton>button{width:100%;min-height:2.8rem;border-radius:11px;font-weight:850}div[data-testid="stExpander"]{border-radius:14px!important;border:1px solid rgba(148,163,184,.09)!important;background:rgba(7,18,32,.50)!important}
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


def event_match(event, game):
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


def cls(v):
    return {"BEST BET":"badge-best","BET":"badge-bet","LEAN":"badge-lean","PASS":"badge-pass"}.get(v,"badge-pass")


def build_candidates(model_df, games, events):
    game_map={g.get("GamePk"):g for g in games}
    out=[]
    for _,r in model_df.iterrows():
        g=game_map.get(r["GamePk"],{})
        event=match_event(events,g); m=moneyline_market(event)
        if not m: continue
        am,hm=no_vig_pair(m["away_consensus"],m["home_consensus"])
        if am is None: continue
        confirmed=bool(r["Away_Lineup_Used"] and r["Home_Lineup_Used"])
        conf=int(r["Model_Confidence"]); alpha=model_alpha(conf,confirmed)
        sides=[
            (r["Away"],float(r["Away_WinProb"]),am,m["away_best"],m["away_book"]),
            (r["Home"],float(r["Home_WinProb"]),hm,m["home_best"],m["home_book"]),
        ]
        side_rows=[]
        for team,raw,market_p,price,book in sides:
            cal=market_p+alpha*(raw-market_p); cal=max(.001,min(.999,cal))
            verdict,edge,ev,imp=grade(cal,price,conf,confirmed)
            side_rows.append({"team":team,"raw":raw,"market_prob":market_p,"prob":cal,"odds":price,"book":book,"verdict":verdict,"edge":edge,"ev":ev,"fair":fair_ml(cal)})
        rank={"BEST BET":4,"BET":3,"LEAN":2,"PASS":1}
        side_rows.sort(key=lambda x:(rank[x["verdict"]],x["edge"],x["ev"],x["prob"]),reverse=True)
        best=side_rows[0]
        out.append({
            "GamePk":r["GamePk"],"game":r["Game"],"away":r["Away"],"home":r["Home"],"time":r.get("TimeLabel",g.get("TimeLabel","")),
            "away_sp":r.get("Away_SP") or "TBD","home_sp":r.get("Home_SP") or "TBD","lineup_confirmed":confirmed,
            "confidence":conf,"alpha":alpha,"books":m["books"],"best":best,"all":side_rows,
            "away_proj":float(r["Away_Proj_Runs"]),"home_proj":float(r["Home_Proj_Runs"]),
            "lineup_status":r["Lineup_Status"],"confidence_reasons":r.get("Confidence_Reasons","")
        })
    order={"BEST BET":4,"BET":3,"LEAN":2,"PASS":1}
    out.sort(key=lambda x:(order[x["best"]["verdict"]],x["best"]["edge"],x["best"]["ev"]),reverse=True)
    return out


st.markdown(f"""
<div class="hero">
  <div class="eyebrow">MLB EDGE • PRODUCTION</div>
  <div class="title">Moneyline Model</div>
  <div class="sub">Pitcher 2.0 + offense/platoon, with confirmed lineups as a live upgrade. Bullpen intentionally excluded after failing holdout promotion.</div>
  <div class="pill">MODEL LIVE • {APP_VERSION}</div>
</div>
""", unsafe_allow_html=True)

try:
    api_key=st.secrets.get("ODDS_API_KEY","")
except Exception:
    api_key=""

c1,c2=st.columns([4,1])
with c2:
    if st.button("↻ Refresh",use_container_width=True):
        fetch_odds.clear(); st.cache_data.clear(); st.rerun()

with st.spinner("Loading today's MLB slate, starters, lineups and market…"):
    games=fetch_today_games()
    model_df=run_model(games) if games else pd.DataFrame()
    odds_payload=fetch_odds(api_key)
    candidates=build_candidates(model_df,games,odds_payload.get("events",[])) if not model_df.empty else []

quota=odds_payload.get("quota",{})
qtxt=f"Odds credits remaining: {quota.get('remaining')}" if quota.get("remaining") is not None else "Current consensus moneyline"
st.markdown(f'<div class="status"><div><span class="dot"></span><span class="live">MODEL LIVE</span> &nbsp; {len(games)} games found</div><div>{qtxt}</div></div>',unsafe_allow_html=True)

if odds_payload.get("error"):
    st.error(odds_payload["error"])

if not games:
    st.info("No MLB games were returned for today.")
    st.stop()

if not candidates:
    st.warning("The model loaded the slate, but no games could be matched to a two-way moneyline market yet. Refresh closer to game time if lines are not posted.")
else:
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
        labels = [f"{x['time']} • {x['away']} @ {x['home']}" for x in chrono]
        selected_label = st.selectbox("Choose matchup", labels, index=0)
        x = chrono[labels.index(selected_label)]
        b = x["best"]
        away_side = next(z for z in x["all"] if z["team"] == x["away"])
        home_side = next(z for z in x["all"] if z["team"] == x["home"])
        lineup_text = "Confirmed lineups" if x["lineup_confirmed"] else "Lineups not fully confirmed"

        st.markdown('<div class="kicker">Single Game Analysis</div>', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="best-card">
          <div class="best-top"><div><div class="best-tag">{b['verdict']}</div><div class="best-pick">{b['team']} ML {b['odds']:+d}</div><div class="best-game">{x['away']} @ {x['home']} • {x['time']} • Best price: {b['book']}</div></div><div class="badge {cls(b['verdict'])}">{b['verdict']}</div></div>
          <div class="metrics">
            <div class="metric"><span>Win chance</span><b>{b['prob']*100:.1f}%</b></div>
            <div class="metric"><span>Edge vs price</span><b>{b['edge']*100:+.1f}%</b></div>
            <div class="metric"><span>EV</span><b>{b['ev']*100:+.1f}%</b></div>
            <div class="metric"><span>Fair line</span><b>{b['fair']:+d}</b></div>
          </div>
          <div class="best-game" style="margin-top:10px">{lineup_text} • Model weight {x['alpha']*100:.0f}% / market {(1-x['alpha'])*100:.0f}% • {x['books']} books in consensus</div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('<div class="kicker">Matchup Detail</div>', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="single-summary">
          <div class="match">{x['away']} @ {x['home']}</div>
          <div class="sp">{x['away_sp']} vs {x['home_sp']}</div>
          <div class="detail-grid">
            <div class="detail"><span>{x['away']} win</span><b>{away_side['prob']*100:.1f}%</b></div>
            <div class="detail"><span>{x['home']} win</span><b>{home_side['prob']*100:.1f}%</b></div>
            <div class="detail"><span>Projected runs</span><b>{x['away_proj']:.2f} – {x['home_proj']:.2f}</b></div>
            <div class="detail"><span>{x['away']} fair ML</span><b>{away_side['fair']:+d}</b></div>
            <div class="detail"><span>{x['home']} fair ML</span><b>{home_side['fair']:+d}</b></div>
            <div class="detail"><span>Model confidence</span><b>{x['confidence']}/100</b></div>
          </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('<div class="kicker">Both Sides</div>', unsafe_allow_html=True)
        for side in [away_side, home_side]:
            st.markdown(f'''
            <div class="game-card">
              <div class="game-head"><div><div class="pick-main">{side['team']} ML {side['odds']:+d}</div><div class="pick-sub">{side['book']} • Fair {side['fair']:+d}</div></div><div class="badge {cls(side['verdict'])}">{side['verdict']}</div></div>
              <div class="metrics">
                <div class="metric"><span>Win chance</span><b>{side['prob']*100:.1f}%</b></div>
                <div class="metric"><span>Market no-vig</span><b>{side['market_prob']*100:.1f}%</b></div>
                <div class="metric"><span>Edge</span><b>{side['edge']*100:+.1f}%</b></div>
                <div class="metric"><span>EV</span><b>{side['ev']*100:+.1f}%</b></div>
              </div>
            </div>
            ''', unsafe_allow_html=True)

        st.caption(f"{x['lineup_status']}. Calibration: {x['alpha']*100:.0f}% model / {(1-x['alpha'])*100:.0f}% market.")
        if x.get("confidence_reasons"):
            st.caption(f"Data notes: {x['confidence_reasons']}")

    else:
        official=[x for x in candidates if x["best"]["verdict"] in ("BEST BET","BET")]
        best=(official or candidates)[0]
        b=best["best"]
        lineup_text="Confirmed lineups" if best["lineup_confirmed"] else "Lineups not fully confirmed"
        st.markdown('<div class="kicker">Top Recommendation</div>',unsafe_allow_html=True)
        st.markdown(f'''
        <div class="best-card">
          <div class="best-top"><div><div class="best-tag">{b['verdict']}</div><div class="best-pick">{b['team']} ML {b['odds']:+d}</div><div class="best-game">{best['away']} @ {best['home']} • {best['time']} • {b['book']}</div></div><div class="badge {cls(b['verdict'])}">{b['verdict']}</div></div>
          <div class="metrics">
            <div class="metric"><span>Win chance</span><b>{b['prob']*100:.1f}%</b></div>
            <div class="metric"><span>Edge vs price</span><b>{b['edge']*100:+.1f}%</b></div>
            <div class="metric"><span>EV</span><b>{b['ev']*100:+.1f}%</b></div>
            <div class="metric"><span>Fair line</span><b>{b['fair']:+d}</b></div>
          </div>
          <div class="best-game" style="margin-top:10px">{lineup_text} • Model weight {best['alpha']*100:.0f}% / market {100-best['alpha']*100:.0f}% • {best['books']} books in consensus</div>
        </div>
        ''',unsafe_allow_html=True)

        st.markdown('<div class="kicker">Official Card</div>',unsafe_allow_html=True)
        if not official:
            st.info("No moneyline cleared the official BET threshold right now. The strongest available lean is shown above.")
        else:
            for i,x in enumerate(official[:5],1):
                b=x["best"]
                st.markdown(f'''
                <div class="game-card">
                  <div class="game-head"><div><div class="game-time">#{i} • {x['time']}</div><div class="match">{x['away']} @ {x['home']}</div><div class="sp">{x['away_sp']} vs {x['home_sp']}</div></div><div class="badge {cls(b['verdict'])}">{b['verdict']}</div></div>
                  <div class="pick"><div><div class="pick-main">{b['team']} ML {b['odds']:+d}</div><div class="pick-sub">{b['book']} • Win {b['prob']*100:.1f}% • Edge {b['edge']*100:+.1f}% • EV {b['ev']*100:+.1f}% • Fair {b['fair']:+d}</div></div><div class="{'lineup-ok' if x['lineup_confirmed'] else 'lineup-wait'}" style="font-size:.60rem;font-weight:900">{'LINEUPS ✓' if x['lineup_confirmed'] else 'LINEUPS WAIT'}</div></div>
                </div>
                ''',unsafe_allow_html=True)

        st.markdown('<div class="kicker">Full Slate</div>',unsafe_allow_html=True)
        for x in sorted(candidates,key=start_sort):
            b=x["best"]
            with st.expander(f"{x['time']}  •  {x['away']} @ {x['home']}  —  {b['verdict']}",expanded=False):
                st.markdown(f'''
                <div class="game-card" style="margin-top:0">
                  <div class="game-head"><div><div class="match">{x['away']} @ {x['home']}</div><div class="sp">{x['away_sp']} vs {x['home_sp']}</div></div><div class="badge {cls(b['verdict'])}">{b['verdict']}</div></div>
                  <div class="pick"><div><div class="pick-main">{b['team']} ML {b['odds']:+d}</div><div class="pick-sub">Win {b['prob']*100:.1f}% • Edge {b['edge']*100:+.1f}% • EV {b['ev']*100:+.1f}% • Fair {b['fair']:+d} • {b['book']}</div></div></div>
                </div>
                ''',unsafe_allow_html=True)
                st.caption(f"Model projected runs: {x['away']} {x['away_proj']:.2f} — {x['home']} {x['home_proj']:.2f}. Calibration: {x['alpha']*100:.0f}% model / {(1-x['alpha'])*100:.0f}% market. {x['lineup_status']}.")
                if x.get("confidence_reasons"):
                    st.caption(f"Data notes: {x['confidence_reasons']}")

        export=[]
        for x in candidates:
            b=x["best"]
            export.append({"Game":x["game"],"Time":x["time"],"Pick":f"{b['team']} ML","Odds":b["odds"],"Book":b["book"],"Verdict":b["verdict"],"Calibrated_Prob":b["prob"],"Edge":b["edge"],"EV":b["ev"],"Fair_ML":b["fair"],"Confidence":x["confidence"],"Lineups_Confirmed":x["lineup_confirmed"],"Model_Weight":x["alpha"],"Model_Version":MODEL_VERSION})
        st.download_button("Download Today's Moneyline Board",pd.DataFrame(export).to_csv(index=False).encode("utf-8"),file_name="mlb_production_moneyline_board.csv",mime="text/csv",use_container_width=True)

st.markdown('<div class="kicker">Model Guardrails</div>',unsafe_allow_html=True)
st.markdown("""
<div class="note"><b>Production scope:</b> moneyline only. Starting pitcher + offense/platoon are the core signal. Confirmed batting orders strengthen the live projection; if lineups are not confirmed, the model leans more heavily on the market. Bullpen is neutral because the historical bullpen layer failed to improve the frozen holdout champion. Run lines and totals are intentionally excluded from v1.0.</div>
""",unsafe_allow_html=True)

with st.expander("Research basis & limitations",expanded=False):
    st.write("The research sequence rejected a team-only model, isolated a starting-pitcher signal through placebo/causality tests, improved it with Pitcher Model 2.0, rejected the bullpen layer, and promoted offense/platoon plus actual lineup information. v1.0 converts that research into a live moneyline workflow rather than continuing to add features.")
    st.write("Historical results are research evidence, not a guarantee of forward profitability. The live engine also uses current data feeds and is not an exact replay of the historical PIT logistic model, so forward tracking remains necessary.")
    st.caption(f"App {APP_VERSION} • Engine {MODEL_VERSION}")
