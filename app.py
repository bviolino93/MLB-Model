import os, json, math, zipfile, io, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import norm

APP_VERSION = "1.1.1-TOTALS-BACKTEST"
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
CACHE = Path(".mlb_totals_history_cache")
CACHE.mkdir(exist_ok=True)

st.set_page_config(page_title="MLB Totals Backtest Lab", page_icon="⚾", layout="wide")
st.markdown("""
<style>
.stApp{background:#061525;color:#eef6ff}.block-container{max-width:920px;padding-top:1.2rem}
h1,h2,h3,label,p,span{color:#eef6ff}.stButton>button,.stDownloadButton>button{background:#164d73!important;color:white!important;border:1px solid #3f8db9!important;font-weight:700}.stButton>button:hover,.stDownloadButton>button:hover{background:#1d638f!important}.stAlert{border-radius:14px}.metric-card{background:#0c2035;border:1px solid #1e3a54;border-radius:16px;padding:16px}.kicker{color:#78d5ff;font-size:.82rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}
</style>
""", unsafe_allow_html=True)


def api_key():
    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        return ""

def valid_american(x):
    try:
        x=float(x); return np.isfinite(x) and abs(x)>=100
    except Exception: return False

def implied(o):
    o=float(o)
    return 100/(o+100) if o>0 else (-o)/((-o)+100)

def profit_per_unit(o):
    o=float(o); return o/100 if o>0 else 100/(-o)

def norm_team(s):
    s=str(s).lower().replace("oakland athletics","athletics").replace("a's","athletics")
    return re.sub(r"[^a-z0-9]","",s)

def cache_file(snapshot):
    key=pd.Timestamp(snapshot).strftime("%Y%m%dT%H%M%SZ")
    return CACHE/f"totals_{key}.json"

def fetch_hist(snapshot, key, force=False):
    p=cache_file(snapshot)
    if p.exists() and not force:
        return json.loads(p.read_text()), True, None, {}
    url=f"{ODDS_BASE}/historical/sports/{SPORT}/odds"
    params={"apiKey":key,"regions":"us","markets":"totals","oddsFormat":"american","date":pd.Timestamp(snapshot).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        r=requests.get(url,params=params,timeout=30)
        hdr={k.lower():v for k,v in r.headers.items()}
        if r.status_code!=200:
            msg=f"Historical totals request failed ({r.status_code})."
            if r.status_code==401: msg+=" Check ODDS_API_KEY / subscription access."
            if r.status_code==429: msg+=" Quota/rate limit reached."
            return None,False,msg,hdr
        data=r.json(); p.write_text(json.dumps(data))
        return data,False,None,hdr
    except Exception as e:
        return None,False,f"Historical totals request failed: {type(e).__name__}",{}

def extract_games(payload):
    if not payload: return []
    if isinstance(payload,dict):
        d=payload.get("data",payload)
        if isinstance(d,dict): return d.get("data",d.get("events",[])) if isinstance(d,dict) else []
        if isinstance(d,list): return d
    return []

def median_or_nan(vals):
    vals=[float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.median(vals)) if vals else np.nan

def flatten_payload(payload,snapshot):
    rows=[]
    for e in extract_games(payload):
        books=e.get("bookmakers") or []
        quotes=[]
        for b in books:
            for m in b.get("markets",[]):
                if m.get("key")!="totals": continue
                over=under=None
                for o in m.get("outcomes",[]):
                    name=str(o.get("name","")).lower()
                    if name=="over": over=o
                    elif name=="under": under=o
                if over and under and valid_american(over.get("price")) and valid_american(under.get("price")):
                    try:
                        lo=float(over.get("point")); lu=float(under.get("point"))
                    except Exception: continue
                    if abs(lo-lu)>.01: continue
                    quotes.append((lo,float(over["price"]),float(under["price"]),b.get("key") or b.get("title")))
        if not quotes: continue
        lines=np.array([q[0] for q in quotes],float); med=float(np.median(lines))
        same=[q for q in quotes if abs(q[0]-med)<=0.26]
        if not same: same=quotes
        line=float(np.median([q[0] for q in same]))
        oo=median_or_nan([q[1] for q in same]); uo=median_or_nan([q[2] for q in same])
        if not(valid_american(oo) and valid_american(uo)): continue
        po,pu=implied(oo),implied(uo); z=po+pu
        rows.append({
            "Snapshot_Timestamp":pd.Timestamp(snapshot).isoformat(),"Event_ID":e.get("id"),"Commence_Time":e.get("commence_time"),
            "Away_Team":e.get("away_team"),"Home_Team":e.get("home_team"),"Total_Line":line,"Over_Odds":oo,"Under_Odds":uo,
            "Over_Market_Prob":po/z,"Under_Market_Prob":pu/z,"Total_Books":len(same)
        })
    return rows

def merge_totals(master,totals):
    m=master.copy(); t=totals.copy()
    for d in (m,t): d["Commence_Time"]=pd.to_datetime(d["Commence_Time"],utc=True,errors="coerce")
    # exact event id first
    exact=m.merge(t.drop(columns=["Away_Team","Home_Team","Commence_Time"],errors="ignore"),on="Event_ID",how="left",suffixes=("","_tot"))
    missing=exact["Total_Line"].isna() if "Total_Line" in exact else pd.Series(True,index=exact.index)
    # fallback team/date mapping
    if missing.any():
        t2=t.copy(); t2["ak"]=t2["Away_Team"].map(norm_team); t2["hk"]=t2["Home_Team"].map(norm_team); t2["date"]=t2["Commence_Time"].dt.date
        lookup={(r.ak,r.hk,r.date):r for r in t2.itertuples()}
        cols=["Total_Line","Over_Odds","Under_Odds","Over_Market_Prob","Under_Market_Prob","Total_Books"]
        for i in exact.index[missing]:
            r=exact.loc[i]; key=(norm_team(r["Away_Team"]),norm_team(r["Home_Team"]),r["Commence_Time"].date() if pd.notna(r["Commence_Time"]) else None)
            hit=lookup.get(key)
            if hit:
                for c in cols: exact.at[i,c]=getattr(hit,c)
    return exact

def build_pit_team_features(master,min_games=12):
    d=master.copy(); d=d[d["Result_Matched"].fillna(False).astype(bool)].copy()
    d["Commence_Time"]=pd.to_datetime(d["Commence_Time"],utc=True,errors="coerce"); d=d.sort_values("Commence_Time")
    histories={}; rows=[]
    for _,r in d.iterrows():
        a,h=r["Away_Team"],r["Home_Team"]; ah=histories.get(a,[]); hh=histories.get(h,[])
        if len(ah)>=min_games and len(hh)>=min_games:
            def feats(hist):
                season=hist; last10=hist[-10:]
                def av(arr,k): return float(np.mean([x[k] for x in arr]))
                return av(season,"rf"),av(season,"ra"),av(last10,"rf"),av(last10,"ra")
            asf,asa,a10f,a10a=feats(ah); hsf,hsa,h10f,h10a=feats(hh)
            rec=r.to_dict(); rec.update({"Away_Season_RF":asf,"Away_Season_RA":asa,"Away_Last10_RF":a10f,"Away_Last10_RA":a10a,
                "Home_Season_RF":hsf,"Home_Season_RA":hsa,"Home_Last10_RF":h10f,"Home_Last10_RA":h10a})
            rows.append(rec)
        ar=float(r["Away_Score"]); hr=float(r["Home_Score"])
        histories.setdefault(a,[]).append({"rf":ar,"ra":hr}); histories.setdefault(h,[]).append({"rf":hr,"ra":ar})
    return pd.DataFrame(rows)

def feature_matrix(d):
    x=pd.DataFrame(index=d.index)
    # matchup expected scoring components, all available before game
    x["AwayOff_vs_HomeDef"]=(pd.to_numeric(d["Away_Season_RF"],errors="coerce")+pd.to_numeric(d["Home_Season_RA"],errors="coerce"))/2
    x["HomeOff_vs_AwayDef"]=(pd.to_numeric(d["Home_Season_RF"],errors="coerce")+pd.to_numeric(d["Away_Season_RA"],errors="coerce"))/2
    x["AwayRecent_vs_HomeRecentDef"]=(pd.to_numeric(d["Away_Last10_RF"],errors="coerce")+pd.to_numeric(d["Home_Last10_RA"],errors="coerce"))/2
    x["HomeRecent_vs_AwayRecentDef"]=(pd.to_numeric(d["Home_Last10_RF"],errors="coerce")+pd.to_numeric(d["Away_Last10_RA"],errors="coerce"))/2
    x["Season_Total_Environment"]=x["AwayOff_vs_HomeDef"]+x["HomeOff_vs_AwayDef"]
    x["Recent_Total_Environment"]=x["AwayRecent_vs_HomeRecentDef"]+x["HomeRecent_vs_AwayRecentDef"]
    return x

def fit_ridge_reg(X,y,ridge=8.0):
    X=np.asarray(X,float); y=np.asarray(y,float)
    mu=np.nanmean(X,axis=0); sd=np.nanstd(X,axis=0); sd=np.where(sd<1e-8,1,sd)
    Xs=np.nan_to_num((X-mu)/sd,nan=0.0); A=np.c_[np.ones(len(Xs)),Xs]
    I=np.eye(A.shape[1]); I[0,0]=0
    beta=np.linalg.solve(A.T@A+ridge*I,A.T@y)
    return beta,mu,sd

def predict_reg(model,X):
    beta,mu,sd=model; X=np.asarray(X,float); Xs=np.nan_to_num((X-mu)/sd,nan=0.0); return np.c_[np.ones(len(Xs)),Xs]@beta

def total_probs(mu,line,sigma):
    sigma=max(float(sigma),1.5); line=float(line)
    if abs(line-round(line))<1e-8:
        n=int(round(line)); p_under=norm.cdf(n-0.5,loc=mu,scale=sigma); p_over=1-norm.cdf(n+0.5,loc=mu,scale=sigma); p_push=max(0,1-p_under-p_over)
    else:
        p_under=norm.cdf(line,loc=mu,scale=sigma); p_over=1-p_under; p_push=0
    return p_over,p_under,p_push

def eval_walkforward(merged,min_games=12):
    pit=build_pit_team_features(merged,min_games)
    pit=pit[pit["Total_Line"].notna()].copy(); pit["Season"]=pd.to_numeric(pit["Season"],errors="coerce")
    X=feature_matrix(pit); y=pd.to_numeric(pit["Final_Total_Runs"],errors="coerce")
    train=pit["Season"]==2023; val=pit["Season"]==2024; hold=pit["Season"]==2025
    if train.sum()<300 or val.sum()<300 or hold.sum()<300: raise ValueError(f"Need adequate 2023/2024/2025 totals coverage. Rows: {train.sum()}/{val.sum()}/{hold.sum()}")
    model=fit_ridge_reg(X.loc[train],y.loc[train]); pred_train=predict_reg(model,X.loc[train]); sigma=float(np.std(y.loc[train].to_numpy()-pred_train,ddof=1))
    pval=predict_reg(model,X.loc[val]); phold=predict_reg(model,X.loc[hold])
    # Select market blend on 2024 by RMSE; market line is a strong prior.
    weights=np.arange(0,1.01,.1); bestw=0; bestr=1e9
    linev=pd.to_numeric(pit.loc[val,"Total_Line"],errors="coerce").to_numpy(); yv=y.loc[val].to_numpy()
    for w in weights:
        blend=w*pval+(1-w)*linev; rmse=float(np.sqrt(np.mean((blend-yv)**2)))
        if rmse<bestr: bestr=rmse; bestw=float(w)
    lineh=pd.to_numeric(pit.loc[hold,"Total_Line"],errors="coerce").to_numpy(); yh=y.loc[hold].to_numpy(); blendh=bestw*phold+(1-bestw)*lineh
    out=pit.loc[hold].copy(); out["Raw_Model_Total"]=phold; out["Calibrated_Total"]=blendh; out["Model_Weight"]=bestw; out["Residual_Sigma"]=sigma
    overp=[]; underp=[]; pushp=[]; side=[]; edge=[]; ev=[]; result=[]; units=[]
    for r in out.itertuples():
        po,pu,pp=total_probs(r.Calibrated_Total,r.Total_Line,sigma); overp.append(po); underp.append(pu); pushp.append(pp)
        eo=po-float(r.Over_Market_Prob); eu=pu-float(r.Under_Market_Prob)
        if eo>=eu: s="OVER"; e=eo; pr=po; odds=float(r.Over_Odds)
        else: s="UNDER"; e=eu; pr=pu; odds=float(r.Under_Odds)
        actual=float(r.Final_Total_Runs); line=float(r.Total_Line)
        won=(actual>line) if s=="OVER" else (actual<line); push=(actual==line)
        evalue=pr*profit_per_unit(odds)-(1-pr-pp)
        u=0.0 if push else (profit_per_unit(odds) if won else -1.0)
        side.append(s); edge.append(e); ev.append(evalue); result.append("PUSH" if push else ("WIN" if won else "LOSS")); units.append(u)
    out["Over_Model_Prob"]=overp; out["Under_Model_Prob"]=underp; out["Push_Prob"]=pushp; out["Best_Side"]=side; out["Best_Edge"]=edge; out["Best_EV"]=ev; out["Result"]=result; out["Units"]=units
    summary=[]
    for e0 in [0.00,.025,.05,.075,.10]:
        b=out[(out["Best_Edge"]>=e0)&(out["Best_EV"]>=0)].copy(); dec=b[b.Result!="PUSH"]
        summary.append({"Min_Edge":e0,"Bets":len(b),"Wins":int((b.Result=="WIN").sum()),"Losses":int((b.Result=="LOSS").sum()),"Pushes":int((b.Result=="PUSH").sum()),"Units":float(b.Units.sum()),"ROI":float(b.Units.sum()/len(b)) if len(b) else np.nan})
    metrics=pd.DataFrame([{"2024_Selected_Model_Weight":bestw,"2024_Validation_RMSE":bestr,"2025_Games":len(out),"2025_Market_Line_RMSE":float(np.sqrt(np.mean((lineh-yh)**2))),"2025_Raw_Model_RMSE":float(np.sqrt(np.mean((phold-yh)**2))),"2025_Calibrated_RMSE":float(np.sqrt(np.mean((blendh-yh)**2))),"Residual_Sigma_2023":sigma}])
    return out,pd.DataFrame(summary),metrics

st.markdown('<div class="kicker">MLB MODEL • TOTALS VALIDATION</div>',unsafe_allow_html=True)
st.title("Totals Backtest Lab")
st.write("Build the missing historical totals market, then evaluate a strict point-in-time totals baseline with 2023 training → 2024 validation → untouched 2025 holdout. Nothing calls The Odds API until you explicitly press the historical-build button.")
st.warning("Your existing historical file was moneyline-only: it contains no historical totals. The totals market must be collected before a real totals betting backtest is possible.")

master_file=st.file_uploader("Upload mlb_moneyline_master_2023_2025.csv",type=["csv"],key="master")
if master_file:
    master=pd.read_csv(master_file)
    need={"Snapshot_Timestamp","Event_ID","Commence_Time","Away_Team","Home_Team","Final_Total_Runs","Result_Matched","Season"}
    missing=need-set(master.columns)
    if missing: st.error("Missing required master columns: "+", ".join(sorted(missing)))
    else:
        snaps=sorted(pd.to_datetime(master["Snapshot_Timestamp"],utc=True,errors="coerce").dropna().dt.floor("s").unique())
        cached=sum(cache_file(x).exists() for x in snaps); new=len(snaps)-cached; ceiling=new*10
        c1,c2,c3=st.columns(3); c1.metric("Historical snapshots",len(snaps)); c2.metric("Already cached",cached); c3.metric("Max new credits",f"{ceiling:,}")
        st.caption("Totals only + US region = up to 10 credits per historical snapshot. With this master there are normally ~547 unique 15:00 UTC snapshots (~5,470 credits if none are cached). Empty responses cost 0 according to the API's historical billing behavior.")
        cap=st.number_input("Hard credit ceiling",min_value=0,max_value=20000,value=min(6000,max(0,ceiling)),step=100)
        confirm=st.checkbox(f"I understand this run can use up to {ceiling:,} new Odds API credits.")
        if st.button("Build / Resume Historical Totals Market",disabled=not confirm):
            key=api_key()
            if not key: st.error("ODDS_API_KEY is missing from Streamlit secrets.")
            elif ceiling>cap: st.error(f"Blocked: estimated new-credit ceiling {ceiling:,} exceeds your hard cap of {cap:,}.")
            else:
                bar=st.progress(0); status=st.empty(); rows=[]; used_before=None; last_headers={}
                for i,s in enumerate(snaps,1):
                    payload,was_cached,err,hdr=fetch_hist(s,key)
                    if err: st.error(err); break
                    rows.extend(flatten_payload(payload,s)); last_headers=hdr or last_headers
                    status.caption(f"Totals history • {i}/{len(snaps)} snapshots • {'cache' if was_cached else 'API'}")
                    bar.progress(i/len(snaps))
                else:
                    totals=pd.DataFrame(rows).drop_duplicates(["Event_ID","Snapshot_Timestamp"])
                    st.session_state["totals_hist"]=totals
                    st.success(f"Historical totals market ready: {len(totals):,} priced event snapshots.")
                    if last_headers:
                        rem=last_headers.get("x-requests-remaining"); used=last_headers.get("x-requests-used"); last=last_headers.get("x-requests-last")
                        if rem is not None: st.caption(f"Odds API headers — remaining: {rem} • used: {used} • last request: {last}")
        # restore from session or upload existing totals file
        existing=st.file_uploader("Or upload an existing historical totals CSV",type=["csv"],key="totals_existing")
        totals=st.session_state.get("totals_hist")
        if existing is not None: totals=pd.read_csv(existing)
        if isinstance(totals,pd.DataFrame) and not totals.empty:
            st.download_button("Download Historical Totals Market CSV",totals.to_csv(index=False).encode(),"mlb_historical_totals_market_2023_2025.csv","text/csv")
            merged=merge_totals(master,totals)
            coverage=int(merged["Total_Line"].notna().sum()); st.metric("Master games with historical total",f"{coverage:,} / {len(merged):,}")
            min_games=st.slider("Minimum prior team games",8,25,12)
            if st.button("Run 2023 → 2024 → 2025 Totals Backtest"):
                with st.spinner("Building point-in-time team scoring states and running untouched 2025 holdout…"):
                    try: hold,buckets,metrics=eval_walkforward(merged,min_games)
                    except Exception as e: st.error(f"Totals backtest failed: {e}")
                    else:
                        st.session_state["totals_results"]=(hold,buckets,metrics)
        if "totals_results" in st.session_state:
            hold,buckets,metrics=st.session_state["totals_results"]
            st.subheader("Holdout results")
            st.dataframe(metrics,use_container_width=True,hide_index=True)
            st.subheader("Edge threshold audit")
            show=buckets.copy(); show["Min_Edge"]=show["Min_Edge"].map(lambda x:f"{x*100:.1f}%"); show["ROI"]=show["ROI"].map(lambda x:"—" if pd.isna(x) else f"{x*100:+.1f}%")
            st.dataframe(show,use_container_width=True,hide_index=True)
            st.info("This is the first totals baseline, not an official betting system. We promote totals only if predictive accuracy and betting buckets remain coherent out of sample; we do not tune thresholds on 2025.")
            st.download_button("Download 2025 Totals Holdout CSV",hold.to_csv(index=False).encode(),"mlb_totals_holdout_2025.csv","text/csv")
            st.download_button("Download Totals Backtest Summary CSV",buckets.to_csv(index=False).encode(),"mlb_totals_backtest_summary.csv","text/csv")
            st.download_button("Download Totals Metrics CSV",metrics.to_csv(index=False).encode(),"mlb_totals_backtest_metrics.csv","text/csv")

st.divider(); st.caption(f"{APP_VERSION} • Research only • Historical Odds API calls are always explicit and guarded by a hard credit ceiling.")
