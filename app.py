import json, math, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm

APP_VERSION = "1.1.3-TOTALS-INTEGRITY"
MLB_API = "https://statsapi.mlb.com/api/v1"
CACHE = Path(".mlb_totals_v112_cache")
CACHE.mkdir(exist_ok=True)

st.set_page_config(page_title="MLB Advanced Totals Lab", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#071321 0%,#06111f 50%,#050d18 100%);color:#eef5fb}
.block-container{max-width:980px;padding-top:1rem;padding-bottom:4rem}
.kicker{font-size:.72rem;font-weight:900;letter-spacing:.18em;color:#7dd3fc;text-transform:uppercase}
.sub{color:#9aacc0;max-width:760px;line-height:1.55}
.stButton>button,.stDownloadButton>button{min-height:48px;border-radius:13px;font-weight:800;background:#164e75;color:white;border:1px solid #38bdf8}
div[data-testid="stMetric"]{background:rgba(255,255,255,.025);border:1px solid rgba(148,163,184,.10);padding:10px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

# -------------------- generic helpers --------------------
def sf(x, d=np.nan):
    try:
        if x in (None,"","-","--"): return d
        return float(x)
    except Exception: return d

def ipdec(x):
    try:
        s=str(x)
        if "." not in s: return float(s)
        a,b=s.split(".",1); return float(a)+float(b)/3.0
    except Exception: return 0.0

def valid_odds(o):
    try:
        o=float(o); return np.isfinite(o) and abs(o)>=100 and o!=0
    except Exception: return False

def imp(o):
    o=float(o)
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def profit(o):
    o=float(o); return o/100 if o>0 else 100/abs(o)

def novig(o,u):
    a,b=imp(o),imp(u); z=a+b
    return a/z,b/z

def cache_json(path, url, params=None):
    path=Path(path)
    if path.exists():
        try: return json.loads(path.read_text())
        except Exception: pass
    r=requests.get(url,params=params,timeout=35); r.raise_for_status(); data=r.json()
    path.write_text(json.dumps(data))
    return data

# -------------------- historical MLB starter + venue --------------------
def schedule_cache(season): return CACHE/f"schedule_{season}.json"
def pitcher_cache(season,pid): return CACHE/f"pitcher_{season}_{pid}.json"

def fetch_season_schedule(season):
    # One free MLB Stats API request per season, cached.
    return cache_json(schedule_cache(season), f"{MLB_API}/schedule", {
        "sportId":1,"season":int(season),"gameType":"R","hydrate":"probablePitcher,venue"
    })

def flatten_schedule(payload):
    rows=[]
    for d in payload.get("dates",[]):
        for g in d.get("games",[]):
            away=g.get("teams",{}).get("away",{}); home=g.get("teams",{}).get("home",{})
            rows.append({
                "MLB_GamePk":g.get("gamePk"),
                "Venue":g.get("venue",{}).get("name",""),
                "Away_SP_ID":(away.get("probablePitcher") or {}).get("id"),
                "Home_SP_ID":(home.get("probablePitcher") or {}).get("id"),
                "Away_SP":(away.get("probablePitcher") or {}).get("fullName"),
                "Home_SP":(home.get("probablePitcher") or {}).get("fullName"),
                "GameNumber":g.get("gameNumber",1),"DoubleHeader":g.get("doubleHeader","N")
            })
    return pd.DataFrame(rows)

def fetch_pitcher_log(season,pid):
    return cache_json(pitcher_cache(season,pid), f"{MLB_API}/people/{int(pid)}/stats", {
        "stats":"gameLog","group":"pitching","season":int(season)
    })

def parse_pitcher_log(payload):
    rows=[]
    stats=payload.get("stats",[])
    if not stats: return pd.DataFrame()
    for s in stats[0].get("splits",[]):
        st=s.get("stat",{})
        rows.append({
            "Date":pd.to_datetime(s.get("date"),errors="coerce",utc=True),
            "GS":sf(st.get("gamesStarted"),0),"IP":ipdec(st.get("inningsPitched",0)),
            "ER":sf(st.get("earnedRuns"),0),"H":sf(st.get("hits"),0),"BB":sf(st.get("baseOnBalls"),0),
            "K":sf(st.get("strikeOuts"),0),"HR":sf(st.get("homeRuns"),0),"Pitches":sf(st.get("numberOfPitches"),np.nan)
        })
    d=pd.DataFrame(rows).dropna(subset=["Date"]).sort_values("Date")
    return d

def pitcher_summary(log, game_time, min_starts=3):
    if log is None or log.empty: return None
    x=log[(log.Date < game_time) & (log.GS>0)].copy()
    if len(x)<min_starts: return None
    def calc(z):
        ip=z.IP.sum(); er=z.ER.sum(); bb=z.BB.sum(); k=z.K.sum(); hr=z.HR.sum(); h=z.H.sum()
        if ip<=0: return {}
        era=9*er/ip; k9=9*k/ip; bb9=9*bb/ip; hr9=9*hr/ip; whip=(h+bb)/ip
        fip=(13*hr+3*bb-2*k)/ip+3.20
        return {"ERA":era,"K9":k9,"BB9":bb9,"HR9":hr9,"WHIP":whip,"FIP":fip,"KBB9":k9-bb9,"IPGS":ip/max(1,len(z))}
    allr=calc(x); last5=calc(x.tail(5)); last3=calc(x.tail(3))
    if not allr: return None
    last_date=x.Date.max(); rest=max(0,(game_time-last_date).total_seconds()/86400)
    # Regress volatile rate stats toward fixed league-ish priors.
    n=len(x); shrink=min(1.0,n/12.0)
    pri={"ERA":4.20,"FIP":4.20,"K9":8.6,"BB9":3.1,"HR9":1.2,"WHIP":1.30,"KBB9":5.5,"IPGS":5.3}
    out={"Starts":n,"Rest":rest}
    for k in pri:
        v=allr.get(k,pri[k]); out[k]=shrink*v+(1-shrink)*pri[k]
        out[f"L5_{k}"]=last5.get(k,v) if last5 else v
        out[f"L3_{k}"]=last3.get(k,v) if last3 else v
    return out

# Static neutral-to-moderate park multipliers. Research feature, not a live claim.
PARK={
"Coors Field":1.10,"Great American Ball Park":1.05,"Fenway Park":1.04,"Yankee Stadium":1.03,"Citizens Bank Park":1.03,
"Globe Life Field":1.02,"American Family Field":1.02,"Wrigley Field":1.01,"Daikin Park":1.01,"Minute Maid Park":1.01,
"Nationals Park":1.01,"Rate Field":1.01,"Guaranteed Rate Field":1.01,"Rogers Centre":1.00,"Kauffman Stadium":1.00,
"Busch Stadium":1.00,"Angel Stadium":1.00,"Sutter Health Park":1.00,"George M. Steinbrenner Field":1.00,
"loanDepot park":0.99,"Chase Field":0.99,"Progressive Field":0.99,"Target Field":0.99,"Comerica Park":0.98,
"Dodger Stadium":0.98,"Truist Park":0.98,"Citi Field":0.98,"PNC Park":0.98,"Petco Park":0.97,"T-Mobile Park":0.97,
"Oracle Park":0.96,"Tropicana Field":0.97
}

# -------------------- market merge --------------------
def merge_totals(master, totals):
    m=master.copy(); t=totals.copy()
    m["Snapshot_Timestamp"]=pd.to_datetime(m["Snapshot_Timestamp"],utc=True,errors="coerce").dt.floor("s")
    t["Snapshot_Timestamp"]=pd.to_datetime(t["Snapshot_Timestamp"],utc=True,errors="coerce").dt.floor("s")
    for c in ["Event_ID"]: m[c]=m[c].astype(str); t[c]=t[c].astype(str)
    keep=["Event_ID","Snapshot_Timestamp","Total_Line","Over_Odds","Under_Odds","Over_Market_Prob","Under_Market_Prob","Total_Books"]
    missing=set(keep)-set(t.columns)
    if missing: raise ValueError("Totals file missing: "+", ".join(sorted(missing)))
    t=t[keep].drop_duplicates(["Event_ID","Snapshot_Timestamp"],keep="last")
    return m.merge(t,on=["Event_ID","Snapshot_Timestamp"],how="left")

# -------------------- point-in-time team run features --------------------
def build_team_pit(df,min_games=12):
    x=df.copy(); x["Commence_Time"]=pd.to_datetime(x.Commence_Time,utc=True,errors="coerce")
    x=x[x.Result_Matched.astype(bool)].dropna(subset=["Commence_Time","Away_Team","Home_Team","Away_Score","Home_Score","Final_Total_Runs"]).sort_values(["Commence_Time","Event_ID"])
    hist={}; rows=[]
    for _,r in x.iterrows():
        gt=r.Commence_Time; a=r.Away_Team; h=r.Home_Team
        def summ(team, venue_side):
            z=hist.get(team,[])
            if not z: return None
            recent10=z[-10:]; recent5=z[-5:]
            side=[q for q in z if q["side"]==venue_side]
            return {
                "N":len(z),"RF":np.mean([q["rf"] for q in z]),"RA":np.mean([q["ra"] for q in z]),
                "TOT":np.mean([q["rf"]+q["ra"] for q in z]),"STD":np.std([q["rf"]+q["ra"] for q in z],ddof=1) if len(z)>2 else 3.0,
                "L10_RF":np.mean([q["rf"] for q in recent10]),"L10_RA":np.mean([q["ra"] for q in recent10]),"L10_TOT":np.mean([q["rf"]+q["ra"] for q in recent10]),
                "L5_RF":np.mean([q["rf"] for q in recent5]),"L5_RA":np.mean([q["ra"] for q in recent5]),"L5_TOT":np.mean([q["rf"]+q["ra"] for q in recent5]),
                "SIDE_RF":np.mean([q["rf"] for q in side]) if len(side)>=5 else np.nan,
                "SIDE_RA":np.mean([q["ra"] for q in side]) if len(side)>=5 else np.nan,
                "Rest":max(0,(gt-z[-1]["time"]).total_seconds()/86400)
            }
        aa=summ(a,"away"); hh=summ(h,"home")
        if aa and hh:
            rec=r.to_dict()
            for k,v in aa.items(): rec[f"Away_{k}"]=v
            for k,v in hh.items(): rec[f"Home_{k}"]=v
            rec["PIT_Eligible"]=(aa["N"]>=min_games and hh["N"]>=min_games)
            rows.append(rec)
        ascore=float(r.Away_Score); hscore=float(r.Home_Score)
        hist.setdefault(a,[]).append({"time":gt,"rf":ascore,"ra":hscore,"side":"away"})
        hist.setdefault(h,[]).append({"time":gt,"rf":hscore,"ra":ascore,"side":"home"})
    return pd.DataFrame(rows)

TEAM_FEATURES=[
"Away_RF","Away_RA","Away_TOT","Away_STD","Away_L10_RF","Away_L10_RA","Away_L10_TOT","Away_L5_RF","Away_L5_RA","Away_L5_TOT","Away_SIDE_RF","Away_SIDE_RA","Away_Rest",
"Home_RF","Home_RA","Home_TOT","Home_STD","Home_L10_RF","Home_L10_RA","Home_L10_TOT","Home_L5_RF","Home_L5_RA","Home_L5_TOT","Home_SIDE_RF","Home_SIDE_RA","Home_Rest"]
SP_KEYS=["ERA","FIP","K9","BB9","HR9","WHIP","KBB9","IPGS","L5_ERA","L5_FIP","L5_KBB9","L3_ERA","L3_FIP","Rest","Starts"]

def attach_context(pit,min_starts=3):
    # Schedule lookups
    sched=[]
    for s in sorted(pd.to_numeric(pit.Season,errors="coerce").dropna().astype(int).unique()):
        try: sched.append(flatten_schedule(fetch_season_schedule(s)))
        except Exception as e: st.warning(f"MLB schedule {s} could not be loaded: {e}")
    if not sched: return pd.DataFrame(), pd.DataFrame([{"PIT_Rows":len(pit),"Context_Rows":0}])
    sch=pd.concat(sched,ignore_index=True).drop_duplicates("MLB_GamePk")
    d=pit.copy(); d["MLB_GamePk"]=pd.to_numeric(d.MLB_GamePk,errors="coerce")
    d=d.merge(sch,on="MLB_GamePk",how="left",suffixes=("","_sched"))
    pairs=set()
    for _,r in d.iterrows():
        s=int(r.Season)
        for c in ["Away_SP_ID","Home_SP_ID"]:
            if pd.notna(r.get(c)): pairs.add((s,int(r[c])))
    logs={}; errs=[]; prog=st.progress(0); status=st.empty()
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut={ex.submit(fetch_pitcher_log,s,p):(s,p) for s,p in sorted(pairs)}
        for i,f in enumerate(as_completed(fut),1):
            key=fut[f]
            try: logs[key]=parse_pitcher_log(f.result())
            except Exception as e: errs.append(f"{key}:{e}")
            prog.progress(i/max(1,len(fut))); status.caption(f"Free MLB starter histories • {i}/{len(fut)}")
    prog.empty(); status.empty()
    rows=[]
    for _,r in d.iterrows():
        gt=pd.to_datetime(r.Commence_Time,utc=True,errors="coerce")
        if pd.isna(gt): continue
        try: s=int(r.Season); aid=int(r.Away_SP_ID); hid=int(r.Home_SP_ID)
        except Exception: continue
        a=pitcher_summary(logs.get((s,aid)),gt,min_starts); h=pitcher_summary(logs.get((s,hid)),gt,min_starts)
        if a is None or h is None: continue
        rec=r.to_dict()
        for k,v in a.items(): rec[f"Away_SP_{k}"]=v
        for k,v in h.items(): rec[f"Home_SP_{k}"]=v
        rec["Park_Factor"]=PARK.get(str(r.get("Venue","")),1.0)
        rows.append(rec)
    q={"PIT_Rows":len(pit),"Schedule_Matched":int(d.Away_SP_ID.notna().sum()),"Starter_Seasons":len(pairs),"Starter_Errors":len(errs),"Context_Rows":len(rows)}
    return pd.DataFrame(rows),pd.DataFrame([q])

# -------------------- models --------------------
def ridge_fit(X,y,l2=12.0):
    X=np.asarray(X,float); y=np.asarray(y,float)
    mu=np.nanmean(X,axis=0); sd=np.nanstd(X,axis=0); sd=np.where(sd<1e-8,1,sd)
    xx=np.where(np.isfinite(X),X,mu); z=(xx-mu)/sd; A=np.c_[np.ones(len(z)),z]
    I=np.eye(A.shape[1]); I[0,0]=0; b=np.linalg.solve(A.T@A+l2*I,A.T@y)
    return b,mu,sd

def ridge_pred(model,X):
    b,mu,sd=model; X=np.asarray(X,float); xx=np.where(np.isfinite(X),X,mu); z=(xx-mu)/sd
    return np.c_[np.ones(len(z)),z]@b

def basic_matrix(d):
    cols=["Away_RF","Away_RA","Away_L10_RF","Away_L10_RA","Home_RF","Home_RA","Home_L10_RF","Home_L10_RA","Away_Rest","Home_Rest"]
    return d[cols].apply(pd.to_numeric,errors="coerce")

def advanced_matrix(d):
    x=d[TEAM_FEATURES].apply(pd.to_numeric,errors="coerce").copy()
    # matchup summaries that let the model learn offense-vs-prevention interactions
    x["Season_Matchup_Runs"]=(pd.to_numeric(d.Away_RF,errors="coerce")+pd.to_numeric(d.Home_RA,errors="coerce")+pd.to_numeric(d.Home_RF,errors="coerce")+pd.to_numeric(d.Away_RA,errors="coerce"))/2
    x["Recent10_Matchup_Runs"]=(pd.to_numeric(d.Away_L10_RF,errors="coerce")+pd.to_numeric(d.Home_L10_RA,errors="coerce")+pd.to_numeric(d.Home_L10_RF,errors="coerce")+pd.to_numeric(d.Away_L10_RA,errors="coerce"))/2
    x["Recent5_Matchup_Runs"]=(pd.to_numeric(d.Away_L5_RF,errors="coerce")+pd.to_numeric(d.Home_L5_RA,errors="coerce")+pd.to_numeric(d.Home_L5_RF,errors="coerce")+pd.to_numeric(d.Away_L5_RA,errors="coerce"))/2
    for side in ["Away","Home"]:
        for k in SP_KEYS:
            x[f"{side}_SP_{k}"]=pd.to_numeric(d[f"{side}_SP_{k}"],errors="coerce")
    x["Combined_SP_FIP"]=pd.to_numeric(d.Away_SP_FIP,errors="coerce")+pd.to_numeric(d.Home_SP_FIP,errors="coerce")
    x["Combined_SP_ERA"]=pd.to_numeric(d.Away_SP_ERA,errors="coerce")+pd.to_numeric(d.Home_SP_ERA,errors="coerce")
    x["Combined_SP_IPGS"]=pd.to_numeric(d.Away_SP_IPGS,errors="coerce")+pd.to_numeric(d.Home_SP_IPGS,errors="coerce")
    x["Combined_SP_HR9"]=pd.to_numeric(d.Away_SP_HR9,errors="coerce")+pd.to_numeric(d.Home_SP_HR9,errors="coerce")
    x["Park_Factor"]=pd.to_numeric(d.Park_Factor,errors="coerce")
    return x.replace([np.inf,-np.inf],np.nan)

def empirical_probs(pred,line,resids):
    # Distribution is learned from training residuals rather than assuming a perfect normal.
    vals=pred+resids
    if abs(float(line)-round(float(line)))<1e-9:
        po=float(np.mean(vals>line)); pu=float(np.mean(vals<line)); pp=max(0.0,1-po-pu)
    else:
        po=float(np.mean(vals>line)); pu=1-po; pp=0.0
    return po,pu,pp

def evaluate_model(d, matrix_fn, label, l2=12.0):
    d=d.copy().reset_index(drop=True); X=matrix_fn(d); y=pd.to_numeric(d.Final_Total_Runs,errors="coerce")
    s=pd.to_numeric(d.Season,errors="coerce"); tr=s==2023; va=s==2024; ho=s==2025
    if min(tr.sum(),va.sum(),ho.sum())<150: raise ValueError(f"Insufficient 2023/24/25 rows for {label}: {tr.sum()}/{va.sum()}/{ho.sum()}")
    m=ridge_fit(X.loc[tr],y.loc[tr],l2); p23=ridge_pred(m,X.loc[tr]); resid=(y.loc[tr].to_numpy()-p23)
    p24=ridge_pred(m,X.loc[va]); line24=pd.to_numeric(d.loc[va, "Total_Line"],errors="coerce").to_numpy(); y24=y.loc[va].to_numpy()
    bestw=0; best=1e9
    for w in np.arange(0,1.01,.1):
        q=w*p24+(1-w)*line24; rm=float(np.sqrt(np.mean((q-y24)**2)))
        if rm<best: best=rm; bestw=float(w)
    # refit 2023+24 after selecting weight
    dev=tr|va; m2=ridge_fit(X.loc[dev],y.loc[dev],l2); pdev=ridge_pred(m2,X.loc[dev]); resdev=y.loc[dev].to_numpy()-pdev
    p25=ridge_pred(m2,X.loc[ho]); line25=pd.to_numeric(d.loc[ho,"Total_Line"],errors="coerce").to_numpy(); y25=y.loc[ho].to_numpy(); cal=bestw*p25+(1-bestw)*line25
    out=d.loc[ho].copy(); out["Raw_Model_Total"]=p25; out["Calibrated_Total"]=cal; out["Model_Weight"]=bestw
    sides=[]; edges=[]; evs=[]; results=[]; units=[]; probs=[]
    for r,pred in zip(out.itertuples(),cal):
        po,pu,pp=empirical_probs(float(pred),float(r.Total_Line),resdev)
        try: om,um=novig(r.Over_Odds,r.Under_Odds)
        except Exception: om,um=r.Over_Market_Prob,r.Under_Market_Prob
        eo=po-float(om); eu=pu-float(um)
        if eo>=eu: side="OVER"; pr=po; edge=eo; odds=float(r.Over_Odds)
        else: side="UNDER"; pr=pu; edge=eu; odds=float(r.Under_Odds)
        ev=pr*profit(odds)-(1-pr-pp)
        actual=float(r.Final_Total_Runs); ln=float(r.Total_Line); push=actual==ln
        win=(actual>ln) if side=="OVER" else (actual<ln)
        unit=0.0 if push else (profit(odds) if win else -1.0)
        sides.append(side); edges.append(edge); evs.append(ev); results.append("PUSH" if push else ("WIN" if win else "LOSS")); units.append(unit); probs.append(pr)
    out["Side"]=sides; out["Bet_Prob"]=probs; out["Edge"]=edges; out["EV"]=evs; out["Result"]=results; out["Units"]=units
    metrics={"Model":label,"2024_Selected_Model_Weight":bestw,"2024_Validation_RMSE":best,"2025_Games":len(out),
             "2025_Market_RMSE":float(np.sqrt(np.mean((line25-y25)**2))),"2025_Raw_RMSE":float(np.sqrt(np.mean((p25-y25)**2))),
             "2025_Calibrated_RMSE":float(np.sqrt(np.mean((cal-y25)**2))),"Residual_SD_Dev":float(np.std(resdev,ddof=1))}
    buckets=[]
    for e0 in [0,.025,.05,.075,.10,.125,.15]:
        b=out[(out.Edge>=e0)&(out.EV>=0)].copy()
        buckets.append({"Model":label,"Min_Edge":e0,"Bets":len(b),"Wins":int((b.Result=="WIN").sum()),"Losses":int((b.Result=="LOSS").sum()),"Pushes":int((b.Result=="PUSH").sum()),"Units":float(b.Units.sum()),"ROI":float(b.Units.sum()/len(b)) if len(b) else np.nan})
    return metrics,pd.DataFrame(buckets),out


# -------------------- integrity audit helpers --------------------
def fit_advanced_frozen(d, l2=18.0):
    d=d.copy().reset_index(drop=True)
    X=advanced_matrix(d)
    y=pd.to_numeric(d.Final_Total_Runs,errors="coerce")
    s=pd.to_numeric(d.Season,errors="coerce")
    tr=s==2023; va=s==2024; dev=tr|va; ho=s==2025
    if min(tr.sum(),va.sum(),ho.sum())<100:
        raise ValueError(f"Insufficient rows for frozen audit: {tr.sum()}/{va.sum()}/{ho.sum()}")
    m=ridge_fit(X.loc[tr],y.loc[tr],l2)
    p24=ridge_pred(m,X.loc[va]); line24=pd.to_numeric(d.loc[va,'Total_Line'],errors='coerce').to_numpy(); y24=y.loc[va].to_numpy()
    bestw=0.0; best=1e9
    for w in np.arange(0,1.01,.1):
        q=w*p24+(1-w)*line24
        rm=float(np.sqrt(np.mean((q-y24)**2)))
        if rm<best: best=rm; bestw=float(w)
    m2=ridge_fit(X.loc[dev],y.loc[dev],l2)
    pdev=ridge_pred(m2,X.loc[dev]); resdev=y.loc[dev].to_numpy()-pdev
    return {"model":m2,"weight":bestw,"resids":resdev,"validation_rmse":best,"columns":list(X.columns)}

def score_holdout(fit, hold_df, label):
    h=hold_df.copy().reset_index(drop=True)
    Xh=advanced_matrix(h)
    p=ridge_pred(fit['model'],Xh)
    lines=pd.to_numeric(h.Total_Line,errors='coerce').to_numpy()
    y=pd.to_numeric(h.Final_Total_Runs,errors='coerce').to_numpy()
    cal=fit['weight']*p+(1-fit['weight'])*lines
    rows=[]
    for i,r in enumerate(h.itertuples()):
        pred=float(cal[i]); line=float(r.Total_Line)
        po,pu,pp=empirical_probs(pred,line,fit['resids'])
        try: om,um=novig(r.Over_Odds,r.Under_Odds)
        except Exception: om,um=float(r.Over_Market_Prob),float(r.Under_Market_Prob)
        eo,eu=po-om,pu-um
        if eo>=eu: side='OVER'; pr=po; edge=eo; odds=float(r.Over_Odds)
        else: side='UNDER'; pr=pu; edge=eu; odds=float(r.Under_Odds)
        ev=pr*profit(odds)-(1-pr-pp)
        actual=float(r.Final_Total_Runs); push=(actual==line)
        win=(actual>line) if side=='OVER' else (actual<line)
        unit=0.0 if push else (profit(odds) if win else -1.0)
        rows.append((side,pr,edge,ev,'PUSH' if push else ('WIN' if win else 'LOSS'),unit))
    h['Raw_Model_Total']=p; h['Calibrated_Total']=cal
    h[['Side','Bet_Prob','Edge','EV','Result','Units']]=pd.DataFrame(rows,index=h.index)
    met={
        'Test':label,'Games':len(h),'Model_Weight':fit['weight'],
        'Market_RMSE':float(np.sqrt(np.mean((lines-y)**2))),
        'Raw_RMSE':float(np.sqrt(np.mean((p-y)**2))),
        'Calibrated_RMSE':float(np.sqrt(np.mean((cal-y)**2))),
    }
    b=h[(h.Edge>=.075)&(h.EV>=0)].copy()
    met.update({'Bets_7_5pct':len(b),'Wins':int((b.Result=='WIN').sum()),'Losses':int((b.Result=='LOSS').sum()),'Pushes':int((b.Result=='PUSH').sum()),'Units':float(b.Units.sum()),'ROI':float(b.Units.sum()/len(b)) if len(b) else np.nan})
    return met,h

def neutralize_starters(h):
    z=h.copy()
    for c in z.columns:
        if c.startswith('Away_SP_') or c.startswith('Home_SP_'):
            z[c]=np.nan
    return z

def scramble_starters(h, seed=113):
    z=h.copy(); rng=np.random.default_rng(seed)
    for prefix in ['Away_SP_','Home_SP_']:
        cols=[c for c in z.columns if c.startswith(prefix)]
        if cols:
            idx=rng.permutation(len(z))
            z.loc[:,cols]=z.loc[:,cols].to_numpy()[idx]
    return z

def opponent_starters(h):
    z=h.copy()
    a=[c for c in z.columns if c.startswith('Away_SP_')]
    for ac in a:
        hc='Home_SP_'+ac[len('Away_SP_'):]
        if hc in z.columns:
            av=z[ac].copy(); z[ac]=z[hc].values; z[hc]=av.values
    return z

def remove_park(h):
    z=h.copy(); z['Park_Factor']=1.0; return z

def remove_recent_runs(h):
    z=h.copy()
    for c in z.columns:
        if '_L10_' in c or '_L5_' in c:
            z[c]=np.nan
    return z

def segment_table(hold):
    rows=[]
    def add(name,x):
        if x.empty: return
        bets=x[(x.Edge>=.075)&(x.EV>=0)]
        rows.append({'Segment':name,'Games':len(x),'Bets':len(bets),'Wins':int((bets.Result=='WIN').sum()),'Losses':int((bets.Result=='LOSS').sum()),'Pushes':int((bets.Result=='PUSH').sum()),'Units':float(bets.Units.sum()),'ROI':float(bets.Units.sum()/len(bets)) if len(bets) else np.nan,'Avg_Edge':float(bets.Edge.mean()) if len(bets) else np.nan})
    add('OVER',hold[hold.Side=='OVER']); add('UNDER',hold[hold.Side=='UNDER'])
    for lo,hi,label in [(0,7.99,'Total <=7.5'),(8,8.99,'Total 8-8.5'),(9,9.99,'Total 9-9.5'),(10,99,'Total 10+')]:
        add(label,hold[(hold.Total_Line>=lo)&(hold.Total_Line<=hi)])
    t=pd.to_datetime(hold.Commence_Time,utc=True,errors='coerce')
    for m in sorted(t.dt.month.dropna().unique()):
        add(pd.Timestamp(2025,int(m),1).strftime('%B'),hold[t.dt.month==m])
    return pd.DataFrame(rows)

# -------------------- UI --------------------
st.markdown('<div class="kicker">MLB MODEL • TOTALS INTEGRITY AUDIT</div>',unsafe_allow_html=True)
st.title('v1.1.3 Advanced Totals Integrity Audit')
st.markdown('<div class="sub">This freezes v1.1.2 and tries to break it. Correct 2023/2024 data determine the model and market blend; 2025 is then attacked with tighter timing, starter placebos, park ablation, recent-run ablation, and segment checks. No Odds API calls.</div>',unsafe_allow_html=True)
st.warning('Promotion standard: correct-context totals should beat the market on 2025, while wrong/scrambled starter tests should materially deteriorate. Betting ROI is secondary to predictive RMSE and robustness.')

master_file=st.file_uploader('1. Upload mlb_moneyline_master_2023_2025.csv',type=['csv'],key='master113')
totals_file=st.file_uploader('2. Upload mlb_historical_totals_market_2023_2025.csv',type=['csv'],key='totals113')
if master_file and totals_file:
    master=pd.read_csv(master_file); totals=pd.read_csv(totals_file)
    try: merged=merge_totals(master,totals)
    except Exception as e: st.error(str(e)); st.stop()
    merged=merged[merged.Total_Line.notna()].copy()
    c1,c2,c3=st.columns(3); c1.metric('Priced master games',f'{len(merged):,}'); c2.metric('2025 games',f"{(pd.to_numeric(merged.Season,errors='coerce')==2025).sum():,}"); c3.metric('Odds API credits','0')
    min_games=st.slider('Minimum prior team games',8,25,12,key='mg113')
    min_starts=st.slider('Minimum prior starter starts',2,8,3,key='ms113')
    if st.button('Run Totals Integrity Audit',key='run113'):
        try:
            with st.spinner('Building point-in-time team run environment…'):
                pit=build_team_pit(merged,min_games=min_games)
                pit=pit[pit.PIT_Eligible].copy(); pit['Hours_To_First_Pitch']=pd.to_numeric(pit.Hours_To_First_Pitch,errors='coerce')
                pit=pit[(pit.Hours_To_First_Pitch>=0)&(pit.Hours_To_First_Pitch<=6)].copy()
            with st.spinner('Attaching historical starters and park context…'):
                ctx,quality=attach_context(pit,min_starts=min_starts)
            if ctx.empty: raise ValueError('No starter-context rows were built.')
            if 'DoubleHeader' in ctx.columns: ctx=ctx[ctx.DoubleHeader.astype(str).eq('N')].copy()
            fit=fit_advanced_frozen(ctx,l2=18.0)
            h=ctx[pd.to_numeric(ctx.Season,errors='coerce')==2025].copy()
            tests=[]; holds={}
            for label,hh in [
                ('Correct context <=6h',h),
                ('Correct context <=3h',h[pd.to_numeric(h.Hours_To_First_Pitch,errors='coerce')<=3]),
                ('Correct context <=1h',h[pd.to_numeric(h.Hours_To_First_Pitch,errors='coerce')<=1]),
                ('Scrambled 2025 starters',scramble_starters(h)),
                ('Opponent starters',opponent_starters(h)),
                ('No starter information',neutralize_starters(h)),
                ('No park factor',remove_park(h)),
                ('No recent run environment',remove_recent_runs(h)),
            ]:
                if len(hh)<40: continue
                met,ho=score_holdout(fit,hh,label); tests.append(met); holds[label]=ho
            summary=pd.DataFrame(tests)
            correct=holds.get('Correct context <=6h',pd.DataFrame())
            segments=segment_table(correct) if not correct.empty else pd.DataFrame()
            st.session_state['v113_summary']=summary; st.session_state['v113_segments']=segments; st.session_state['v113_hold']=correct; st.session_state['v113_quality']=quality
        except Exception as e: st.exception(e)

    summary=st.session_state.get('v113_summary')
    if isinstance(summary,pd.DataFrame) and not summary.empty:
        st.subheader('Integrity Stress Tests')
        st.dataframe(summary,use_container_width=True,hide_index=True)
        try:
            corr=summary[summary.Test=='Correct context <=6h'].iloc[0]
            scr=summary[summary.Test=='Scrambled 2025 starters'].iloc[0]
            opp=summary[summary.Test=='Opponent starters'].iloc[0]
            passes=(corr.Calibrated_RMSE<corr.Market_RMSE and scr.Calibrated_RMSE>corr.Calibrated_RMSE and opp.Calibrated_RMSE>corr.Calibrated_RMSE)
            if passes: st.success('INTEGRITY SIGNAL PASSES CORE PLACEBO CHECKS — correct context beats market and both scrambled/opponent starter tests deteriorate. Review timing and segment stability before promotion.')
            else: st.error('INTEGRITY SIGNAL DOES NOT CLEANLY PASS — do not promote totals yet.')
        except Exception: pass
        seg=st.session_state.get('v113_segments')
        if isinstance(seg,pd.DataFrame) and not seg.empty:
            st.subheader('2025 Robustness Segments')
            st.dataframe(seg,use_container_width=True,hide_index=True)
        q=st.session_state.get('v113_quality')
        if isinstance(q,pd.DataFrame):
            with st.expander('Data quality'): st.dataframe(q,use_container_width=True,hide_index=True)
        c1,c2,c3=st.columns(3)
        c1.download_button('Download Audit Summary',summary.to_csv(index=False).encode(),'mlb_totals_v113_integrity_summary.csv','text/csv')
        c2.download_button('Download Segments',st.session_state['v113_segments'].to_csv(index=False).encode(),'mlb_totals_v113_segments.csv','text/csv')
        c3.download_button('Download Correct Holdout',st.session_state['v113_hold'].to_csv(index=False).encode(),'mlb_totals_v113_holdout_2025.csv','text/csv')

st.caption('Zero Odds API calls. Uses the uploaded historical Moneyline Master + historical totals market and the free cached MLB Stats API starter histories.')
