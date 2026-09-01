import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

MODEL_VERSION = "1.1.0-PRODUCTION-ML-TOTALS-RESEARCH"
MLB_API = "https://statsapi.mlb.com/api"

BASE_RUNS_PER_TEAM = 4.45
HOME_RUN_ADVANTAGE = 0.12
PYTH_EXPONENT = 1.83
STARTER_IMPACT = 1.35

TEAM_IDS = {}
_json_cache = {}
_pitcher_cache = {}
_hitting_cache = {}
_platoon_cache = {}
_feed_cache = {}
_hitter_cache = {}
_hand_cache = {}

# Totals research context only. These factors do NOT alter the frozen moneyline engine.
PARKS = {
    "Coors Field": {"factor": 1.10, "lat": 39.7559, "lon": -104.9942},
    "Great American Ball Park": {"factor": 1.05, "lat": 39.0979, "lon": -84.5082},
    "Fenway Park": {"factor": 1.04, "lat": 42.3467, "lon": -71.0972},
    "Yankee Stadium": {"factor": 1.03, "lat": 40.8296, "lon": -73.9262},
    "Citizens Bank Park": {"factor": 1.03, "lat": 39.9061, "lon": -75.1665},
    "Globe Life Field": {"factor": 1.02, "lat": 32.7473, "lon": -97.0847},
    "American Family Field": {"factor": 1.02, "lat": 43.0280, "lon": -87.9712},
    "Daikin Park": {"factor": 1.01, "lat": 29.7573, "lon": -95.3555},
    "Minute Maid Park": {"factor": 1.01, "lat": 29.7573, "lon": -95.3555},
    "Wrigley Field": {"factor": 1.01, "lat": 41.9484, "lon": -87.6553},
    "Nationals Park": {"factor": 1.01, "lat": 38.8730, "lon": -77.0074},
    "Oriole Park at Camden Yards": {"factor": 1.00, "lat": 39.2839, "lon": -76.6217},
    "Rogers Centre": {"factor": 1.00, "lat": 43.6414, "lon": -79.3894},
    "Kauffman Stadium": {"factor": 1.00, "lat": 39.0517, "lon": -94.4803},
    "Busch Stadium": {"factor": 1.00, "lat": 38.6226, "lon": -90.1928},
    "Angel Stadium": {"factor": 1.00, "lat": 33.8003, "lon": -117.8827},
    "loanDepot park": {"factor": 0.99, "lat": 25.7781, "lon": -80.2197},
    "Chase Field": {"factor": 0.99, "lat": 33.4453, "lon": -112.0667},
    "Progressive Field": {"factor": 0.99, "lat": 41.4962, "lon": -81.6852},
    "Target Field": {"factor": 0.99, "lat": 44.9817, "lon": -93.2776},
    "Comerica Park": {"factor": 0.98, "lat": 42.3390, "lon": -83.0485},
    "Dodger Stadium": {"factor": 0.98, "lat": 34.0739, "lon": -118.2400},
    "Truist Park": {"factor": 0.98, "lat": 33.8908, "lon": -84.4677},
    "Citi Field": {"factor": 0.98, "lat": 40.7571, "lon": -73.8458},
    "PNC Park": {"factor": 0.98, "lat": 40.4469, "lon": -80.0057},
    "Petco Park": {"factor": 0.97, "lat": 32.7076, "lon": -117.1570},
    "T-Mobile Park": {"factor": 0.97, "lat": 47.5914, "lon": -122.3325},
    "Oracle Park": {"factor": 0.96, "lat": 37.7786, "lon": -122.3893},
    "Sutter Health Park": {"factor": 1.00, "lat": 38.5803, "lon": -121.5137},
    "George M. Steinbrenner Field": {"factor": 1.00, "lat": 27.9799, "lon": -82.5067},
    "Tropicana Field": {"factor": 0.97, "lat": 27.7682, "lon": -82.6534},
    "Rate Field": {"factor": 1.01, "lat": 41.8300, "lon": -87.6338},
    "Guaranteed Rate Field": {"factor": 1.01, "lat": 41.8300, "lon": -87.6338},
}
_totals_weather_cache = {}


def today_et():
    return datetime.now(ZoneInfo("America/New_York")).date()


def now_et():
    return datetime.now(ZoneInfo("America/New_York"))


def season_now():
    return today_et().year


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


def safe_float(x, default=np.nan):
    try:
        if x in (None, "", "-", "--"):
            return default
        return float(x)
    except Exception:
        return default


def ip_to_decimal(ip):
    try:
        s = str(ip)
        if "." not in s:
            return float(s)
        whole, outs = s.split(".", 1)
        return float(whole) + float(outs) / 3.0
    except Exception:
        return 0.0


def fair_ml(prob):
    p = clamp(prob, 0.001, 0.999)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def implied_prob(odds):
    o = float(odds)
    if o == 0 or abs(o) < 100:
        raise ValueError(f"Invalid American odds: {odds}")
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)


def expected_value(prob, odds):
    o = float(odds)
    profit = o / 100.0 if o > 0 else 100.0 / abs(o)
    return float(prob) * profit - (1.0 - float(prob))


def win_prob(runs_for, runs_against):
    rf = max(0.01, float(runs_for))
    ra = max(0.01, float(runs_against))
    return (rf ** PYTH_EXPONENT) / ((rf ** PYTH_EXPONENT) + (ra ** PYTH_EXPONENT))


def get_json(url, params=None, cache_key=None):
    key = cache_key or (url, tuple(sorted((params or {}).items())))
    if key in _json_cache:
        return _json_cache[key]
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception:
        data = {}
    _json_cache[key] = data
    return data


def load_team_ids():
    global TEAM_IDS
    data = get_json(f"{MLB_API}/v1/teams", {"sportId": 1}, cache_key="teams")
    TEAM_IDS = {t.get("name"): t.get("id") for t in data.get("teams", []) if t.get("name") and t.get("id")}
    return TEAM_IDS


def fetch_games_for_date(selected_date=None):
    """Fetch the free MLB schedule for a selected current/upcoming date.

    This function never calls The Odds API and therefore never consumes odds credits.
    """
    if not TEAM_IDS:
        load_team_ids()
    if selected_date is None:
        selected_date = today_et()
    try:
        day = pd.Timestamp(selected_date).date().strftime("%Y-%m-%d")
    except Exception:
        day = today_et().strftime("%Y-%m-%d")
    data = get_json(
        f"{MLB_API}/v1/schedule",
        {"sportId": 1, "date": day, "hydrate": "probablePitcher,venue"},
        cache_key=("schedule", day),
    )
    games = []
    for block in data.get("dates", []):
        for g in block.get("games", []):
            away = g.get("teams", {}).get("away", {})
            home = g.get("teams", {}).get("home", {})
            asp = away.get("probablePitcher", {}) or {}
            hsp = home.get("probablePitcher", {}) or {}
            game_date = g.get("gameDate")
            try:
                dt = pd.to_datetime(game_date, utc=True).tz_convert("America/New_York")
                time_label = dt.strftime("%-I:%M %p")
                hours_to_game = (dt.to_pydatetime() - now_et()).total_seconds() / 3600.0
            except Exception:
                time_label = ""
                hours_to_game = np.nan
            games.append({
                "GamePk": g.get("gamePk"),
                "Away": away.get("team", {}).get("name"),
                "Home": home.get("team", {}).get("name"),
                "Away_SP": asp.get("fullName"),
                "Away_SP_ID": asp.get("id"),
                "Home_SP": hsp.get("fullName"),
                "Home_SP_ID": hsp.get("id"),
                "Venue": g.get("venue", {}).get("name", ""),
                "GameDate": game_date,
                "TimeLabel": time_label,
                "HoursToGame": hours_to_game,
                "GameNumber": int(safe_float(g.get("gameNumber"), 1)),
                "DoubleHeader": g.get("doubleHeader", "N"),
                "AbstractGameState": (g.get("status", {}) or {}).get("abstractGameState", ""),
                "DetailedState": (g.get("status", {}) or {}).get("detailedState", ""),
                "StatusCode": (g.get("status", {}) or {}).get("statusCode", ""),
            })
    return games


def fetch_today_games():
    """Backward-compatible alias for today's slate."""
    return fetch_games_for_date(today_et())


def pitcher_hand(player_id):
    if not player_id:
        return "U"
    if player_id in _hand_cache:
        return _hand_cache[player_id]
    data = get_json(f"{MLB_API}/v1/people/{player_id}", cache_key=("person", player_id))
    try:
        hand = data["people"][0].get("pitchHand", {}).get("code", "U")
    except Exception:
        hand = "U"
    _hand_cache[player_id] = hand
    return hand


def _pitcher_season_stats(player_id):
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "season", "group": "pitching", "season": season_now()},
        cache_key=("pitcher_season", season_now(), player_id),
    )
    try:
        return data["stats"][0]["splits"][0]["stat"]
    except Exception:
        return {}


def _pitcher_game_log(player_id):
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "gameLog", "group": "pitching", "season": season_now()},
        cache_key=("pitcher_log", season_now(), player_id),
    )
    rows = []
    try:
        for s in data.get("stats", [])[0].get("splits", []):
            st = s.get("stat", {})
            started = safe_float(st.get("gamesStarted"), 0) > 0
            rows.append({
                "Date": pd.to_datetime(s.get("date"), errors="coerce"),
                "Started": int(started),
                "IP": ip_to_decimal(st.get("inningsPitched", 0)),
                "ER": safe_float(st.get("earnedRuns"), 0),
                "H": safe_float(st.get("hits"), 0),
                "BB": safe_float(st.get("baseOnBalls"), 0),
                "K": safe_float(st.get("strikeOuts"), 0),
                "HR": safe_float(st.get("homeRuns"), 0),
                "Pitches": safe_float(st.get("numberOfPitches"), np.nan),
            })
    except Exception:
        pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
    return df


def _rates_from_counts(ip, er, h, bb, k, hr):
    if ip <= 0:
        return None
    era = 9.0 * er / ip
    k9 = 9.0 * k / ip
    bb9 = 9.0 * bb / ip
    hr9 = 9.0 * hr / ip
    whip = (h + bb) / ip
    fip = ((13.0 * hr + 3.0 * bb - 2.0 * k) / ip) + 3.15
    return {"ERA": era, "K9": k9, "BB9": bb9, "HR9": hr9, "WHIP": whip, "FIP": fip}


def _aggregate_starts(df):
    if df is None or df.empty:
        return None
    x = df[df["Started"] == 1].copy()
    if x.empty:
        return None
    ip = float(x["IP"].sum())
    return _rates_from_counts(
        ip,
        float(x["ER"].sum()), float(x["H"].sum()), float(x["BB"].sum()),
        float(x["K"].sum()), float(x["HR"].sum()),
    )


def starter_quality_v2(player_id):
    if not player_id:
        return {
            "Quality": 0.985, "Starts": 0, "IP": 0, "RecentStarts": 0,
            "SeasonERA": np.nan, "SeasonFIP": np.nan, "RecentERA": np.nan,
            "RecentK9": np.nan, "RecentBB9": np.nan, "RecentPitches": np.nan,
        }
    if player_id in _pitcher_cache:
        return _pitcher_cache[player_id]

    st = _pitcher_season_stats(player_id)
    log = _pitcher_game_log(player_id)
    starts = int(safe_float(st.get("gamesStarted"), 0))
    ip = ip_to_decimal(st.get("inningsPitched", 0))
    er = safe_float(st.get("earnedRuns"), np.nan)
    h = safe_float(st.get("hits"), np.nan)
    bb = safe_float(st.get("baseOnBalls"), np.nan)
    k = safe_float(st.get("strikeOuts"), np.nan)
    hr = safe_float(st.get("homeRuns"), np.nan)

    season_rates = None
    if all(np.isfinite(v) for v in [ip, er, h, bb, k, hr]) and ip > 0:
        season_rates = _rates_from_counts(ip, er, h, bb, k, hr)

    recent = log[log["Started"] == 1].head(5) if not log.empty else pd.DataFrame()
    recent_rates = _aggregate_starts(recent)
    recent_pitches = float(recent["Pitches"].dropna().mean()) if not recent.empty and recent["Pitches"].notna().any() else np.nan

    # Fixed league priors; recent form is deliberately shrunk toward the season baseline.
    prior = {"ERA": 4.20, "FIP": 4.20, "K9": 8.60, "BB9": 3.20, "HR9": 1.25, "WHIP": 1.30}
    s = season_rates or prior.copy()
    season_weight = clamp(ip / 70.0, 0.15, 1.0) if ip > 0 else 0.15
    shrunk_s = {k0: season_weight * s.get(k0, prior[k0]) + (1 - season_weight) * prior[k0] for k0 in prior}

    if recent_rates:
        recent_ip = float(recent["IP"].sum())
        rw = clamp(recent_ip / 28.0, 0.10, 0.75)
        rr = {k0: rw * recent_rates.get(k0, shrunk_s[k0]) + (1 - rw) * shrunk_s[k0] for k0 in prior}
    else:
        rr = shrunk_s.copy()

    # Research-informed composite: skill components + recency. Lower run-prevention metrics are better.
    z = (
        0.17 * ((4.20 - shrunk_s["FIP"]) / 0.85)
        + 0.12 * ((4.20 - shrunk_s["ERA"]) / 1.00)
        + 0.15 * ((shrunk_s["K9"] - 8.60) / 1.80)
        + 0.12 * ((3.20 - shrunk_s["BB9"]) / 1.00)
        + 0.08 * ((1.25 - shrunk_s["HR9"]) / 0.45)
        + 0.10 * ((1.30 - shrunk_s["WHIP"]) / 0.18)
        + 0.12 * ((4.20 - rr["FIP"]) / 0.95)
        + 0.08 * ((4.20 - rr["ERA"]) / 1.15)
        + 0.04 * ((rr["K9"] - 8.60) / 2.00)
        + 0.02 * ((3.20 - rr["BB9"]) / 1.10)
    )
    raw_quality = 1.0 + 0.16 * math.tanh(z / 1.25)
    role_shrink = clamp(starts / 6.0, 0.25, 1.0)
    quality = 1.0 + (raw_quality - 1.0) * role_shrink

    result = {
        "Quality": clamp(quality, 0.82, 1.18),
        "Starts": starts,
        "IP": ip,
        "RecentStarts": len(recent),
        "SeasonERA": shrunk_s["ERA"],
        "SeasonFIP": shrunk_s["FIP"],
        "SeasonK9": shrunk_s["K9"],
        "SeasonBB9": shrunk_s["BB9"],
        "RecentERA": rr["ERA"],
        "RecentFIP": rr["FIP"],
        "RecentK9": rr["K9"],
        "RecentBB9": rr["BB9"],
        "RecentPitches": recent_pitches,
    }
    _pitcher_cache[player_id] = result
    return result


def expected_sp_ip(player_id):
    if not player_id:
        return 4.5
    log = _pitcher_game_log(player_id)
    starts = log[log["Started"] == 1].head(5) if not log.empty else pd.DataFrame()
    if starts.empty:
        p = starter_quality_v2(player_id)
        return clamp((p.get("IP", 0) / max(1, p.get("Starts", 0))) if p.get("Starts", 0) else 5.0, 4.0, 6.4)
    vals = starts["IP"].to_numpy(dtype=float)
    weights = np.arange(len(vals), 0, -1, dtype=float)
    ip = float(np.average(vals, weights=weights))
    pitches = starts["Pitches"].dropna()
    if len(pitches):
        avgp = float(pitches.mean())
        if avgp >= 95:
            ip += 0.15
        elif avgp <= 75:
            ip -= 0.25
    return clamp(ip, 4.0, 6.8)


def _team_hitting_stats(team_name, stats_type="season", start_date=None, end_date=None):
    if not TEAM_IDS:
        load_team_ids()
    tid = TEAM_IDS.get(team_name)
    if not tid:
        return {}
    key = (team_name, stats_type, str(start_date), str(end_date))
    if key in _hitting_cache:
        return _hitting_cache[key]
    params = {"stats": stats_type, "group": "hitting", "season": season_now()}
    if start_date:
        params["startDate"] = str(start_date)
    if end_date:
        params["endDate"] = str(end_date)
    data = get_json(f"{MLB_API}/v1/teams/{tid}/stats", params, cache_key=("hit", key))
    try:
        out = data["stats"][0]["splits"][0]["stat"]
    except Exception:
        out = {}
    _hitting_cache[key] = out
    return out


def _hitting_rates(stat):
    if not stat:
        return None
    pa = safe_float(stat.get("plateAppearances"), np.nan)
    ab = safe_float(stat.get("atBats"), np.nan)
    bb = safe_float(stat.get("baseOnBalls"), 0)
    hbp = safe_float(stat.get("hitByPitch"), 0)
    sf = safe_float(stat.get("sacFlies"), 0)
    so = safe_float(stat.get("strikeOuts"), 0)
    hr = safe_float(stat.get("homeRuns"), 0)
    hits = safe_float(stat.get("hits"), 0)
    doubles = safe_float(stat.get("doubles"), 0)
    triples = safe_float(stat.get("triples"), 0)
    runs = safe_float(stat.get("runs"), 0)
    ops = safe_float(stat.get("ops"), np.nan)
    if not np.isfinite(pa):
        pa = (ab if np.isfinite(ab) else 0) + bb + hbp + sf
    if pa <= 0:
        return None
    singles = max(0.0, hits - doubles - triples - hr)
    tb = singles + 2*doubles + 3*triples + 4*hr
    iso = (tb / max(ab, 1)) - (hits / max(ab, 1)) if np.isfinite(ab) and ab > 0 else 0.145
    return {
        "PA": pa, "OPS": ops if np.isfinite(ops) else 0.720,
        "Kpct": so/pa, "BBpct": bb/pa, "HRpct": hr/pa,
        "ISO": iso, "RPA": runs/pa,
    }


def team_offense(team_name):
    season_stat = _team_hitting_stats(team_name, "season")
    season = _hitting_rates(season_stat) or {"PA": 0, "OPS": .720, "Kpct": .225, "BBpct": .082, "HRpct": .030, "ISO": .145, "RPA": .115}
    start = today_et() - timedelta(days=14)
    recent_stat = _team_hitting_stats(team_name, "byDateRange", start, today_et())
    recent = _hitting_rates(recent_stat)
    if recent and recent["PA"] >= 120:
        rw = clamp(recent["PA"] / (recent["PA"] + 350.0), 0.15, 0.45)
    else:
        rw = 0.0
    ops = (1-rw)*season["OPS"] + rw*(recent["OPS"] if recent else season["OPS"])
    k = (1-rw)*season["Kpct"] + rw*(recent["Kpct"] if recent else season["Kpct"])
    bb = (1-rw)*season["BBpct"] + rw*(recent["BBpct"] if recent else season["BBpct"])
    hr = (1-rw)*season["HRpct"] + rw*(recent["HRpct"] if recent else season["HRpct"])
    iso = (1-rw)*season["ISO"] + rw*(recent["ISO"] if recent else season["ISO"])
    composite = (
        0.45*((ops-.720)/.070) + 0.16*((.225-k)/.035) + 0.15*((bb-.082)/.025)
        + 0.14*((hr-.030)/.010) + 0.10*((iso-.145)/.040)
    )
    factor = 1.0 + 0.10*math.tanh(composite/1.8)
    return {"Factor": clamp(factor,.88,1.12), "OPS": ops, "RecentUsed": bool(rw>0), "PA": season["PA"]}


def team_platoon(team_name, opposing_hand):
    if opposing_hand not in ("L", "R"):
        return {"Factor": 1.0, "OPS": .720, "Available": False, "PA": 0}
    key = (team_name, opposing_hand)
    if key in _platoon_cache:
        return _platoon_cache[key]
    if not TEAM_IDS:
        load_team_ids()
    tid = TEAM_IDS.get(team_name)
    if not tid:
        return {"Factor": 1.0, "OPS": .720, "Available": False, "PA": 0}
    sit = "vl" if opposing_hand == "L" else "vr"
    data = get_json(
        f"{MLB_API}/v1/teams/{tid}/stats",
        {"stats": "season", "group": "hitting", "season": season_now(), "sitCodes": sit},
        cache_key=("platoon", season_now(), tid, sit),
    )
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
    except Exception:
        stat = {}
    rates = _hitting_rates(stat)
    if not rates or rates["PA"] < 80:
        result = {"Factor": 1.0, "OPS": .720, "Available": False, "PA": rates["PA"] if rates else 0}
    else:
        # Heavy shrinkage: platoon earns a modest adjustment rather than replacing team strength.
        shrink = rates["PA"] / (rates["PA"] + 180.0)
        ops = shrink*rates["OPS"] + (1-shrink)*.720
        result = {"Factor": clamp((ops/.720)**0.28,.93,1.07), "OPS": ops, "Available": True, "PA": rates["PA"]}
    _platoon_cache[key] = result
    return result


def game_feed(game_pk):
    if game_pk in _feed_cache:
        return _feed_cache[game_pk]
    data = get_json(f"{MLB_API}/v1.1/game/{game_pk}/feed/live", cache_key=("feed", game_pk))
    _feed_cache[game_pk] = data
    return data


def get_lineup(game_pk, side):
    try:
        team = game_feed(game_pk)["liveData"]["boxscore"]["teams"][side]
        order = team.get("battingOrder", [])
        players = team.get("players", {})
        return [{"id": pid, "name": players.get(f"ID{pid}", {}).get("person", {}).get("fullName", "")} for pid in order[:9]]
    except Exception:
        return []


def hitter_ops(player_id):
    key = (season_now(), player_id)
    if key in _hitter_cache:
        return _hitter_cache[key]
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "season", "group": "hitting", "season": season_now()},
        cache_key=("hitter", *key),
    )
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
        ops = safe_float(stat.get("ops"), .720)
        pa = safe_float(stat.get("plateAppearances"), 0)
    except Exception:
        ops, pa = .720, 0
    if pa < 20:
        w = pa/(pa+50.0)
        ops = w*ops + (1-w)*.720
    _hitter_cache[key] = ops
    return ops


def lineup_factor(lineup):
    if len(lineup) < 8:
        return 1.0
    weights = np.array([1.15,1.12,1.10,1.08,1.04,1.00,.96,.92,.88], dtype=float)
    vals = np.array([hitter_ops(p["id"]) for p in lineup[:9]], dtype=float)
    weighted = float(np.average(vals, weights=weights[:len(vals)]))
    return clamp((weighted/.720)**0.38,.91,1.09)


def final_offense(team_name, lineup, opposing_hand):
    base = team_offense(team_name)
    platoon = team_platoon(team_name, opposing_hand)
    lineup_used = len(lineup) >= 8
    lf = lineup_factor(lineup) if lineup_used else 1.0
    if lineup_used:
        factor = 0.60*base["Factor"] + 0.20*platoon["Factor"] + 0.20*lf
    else:
        factor = 0.75*base["Factor"] + 0.25*platoon["Factor"]
    return {
        "Factor": clamp(factor,.86,1.14), "BaseFactor": base["Factor"],
        "PlatoonFactor": platoon["Factor"], "PlatoonOPS": platoon["OPS"],
        "PlatoonAvailable": platoon["Available"], "LineupFactor": lf,
        "LineupUsed": lineup_used, "RecentOffenseUsed": base["RecentUsed"],
    }


def starter_run_factor(quality, expected_ip):
    ip = clamp(expected_ip, 4.0, 6.8)
    sp_component = math.exp(-STARTER_IMPACT * (float(quality)-1.0))
    # Research rejected the bullpen layer. Remaining innings are deliberately neutral.
    return clamp((ip/9.0)*sp_component + ((9.0-ip)/9.0)*1.0, .84, 1.16)


def confidence_score(g, away_sp, home_sp, away_off, home_off):
    score = 100
    reasons = []
    if not g.get("Away_SP_ID"):
        score -= 24; reasons.append("away starter unknown")
    if not g.get("Home_SP_ID"):
        score -= 24; reasons.append("home starter unknown")
    if away_sp.get("Starts",0) < 3:
        score -= 10; reasons.append("away starter small sample")
    if home_sp.get("Starts",0) < 3:
        score -= 10; reasons.append("home starter small sample")
    if not away_off.get("PlatoonAvailable"):
        score -= 3
    if not home_off.get("PlatoonAvailable"):
        score -= 3
    if not away_off.get("LineupUsed"):
        score -= 7; reasons.append("away lineup unconfirmed")
    if not home_off.get("LineupUsed"):
        score -= 7; reasons.append("home lineup unconfirmed")
    score = int(clamp(score, 35, 100))
    grade = "HIGH" if score >= 85 else "MEDIUM" if score >= 70 else "LOW"
    return score, grade, " | ".join(reasons)



def totals_weather_info(venue, game_date):
    default = {"Temp": np.nan, "Wind": np.nan, "Humidity": np.nan, "Precip": np.nan, "Factor": 1.00, "Available": False}
    park = PARKS.get(venue)
    if not park or not game_date:
        return default
    key = (venue, str(game_date))
    if key in _totals_weather_cache:
        return _totals_weather_cache[key]
    try:
        game_dt = pd.to_datetime(game_date, utc=True)
        data = get_json(
            "https://api.open-meteo.com/v1/forecast",
            {"latitude": park["lat"], "longitude": park["lon"],
             "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m",
             "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC", "forecast_days": 14},
            cache_key=("totals_weather", venue, str(game_date)),
        )
        hourly=data.get("hourly",{}); times=pd.to_datetime(hourly.get("time",[]),utc=True)
        if len(times)==0: raise ValueError("weather unavailable")
        idx=int(np.argmin(np.abs((times-game_dt).total_seconds())))
        temp=safe_float(hourly.get("temperature_2m",[])[idx],np.nan)
        humidity=safe_float(hourly.get("relative_humidity_2m",[])[idx],np.nan)
        precip=safe_float(hourly.get("precipitation_probability",[])[idx],np.nan)
        wind=safe_float(hourly.get("wind_speed_10m",[])[idx],np.nan)
        factor=1.0
        if not pd.isna(temp): factor*=1.0+(temp-72.0)*0.0015
        result={"Temp":temp,"Wind":wind,"Humidity":humidity,"Precip":precip,"Factor":clamp(factor,.96,1.04),"Available":True}
    except Exception:
        result=default
    _totals_weather_cache[key]=result
    return result


def totals_projection(row):
    """Production totals core.

    The validated historical totals work found the durable signal in starting-pitcher
    quality plus team run environment. Park did not improve the integrity audit, so
    park/weather are informational only here rather than hard multipliers.
    """
    base_total = (
        safe_float(row.get("Away_Proj_Runs"), BASE_RUNS_PER_TEAM)
        + safe_float(row.get("Home_Proj_Runs"), BASE_RUNS_PER_TEAM)
    )
    venue = row.get("Venue", "")
    park = PARKS.get(venue, {})
    park_factor = safe_float(park.get("factor"), 1.0)
    wx = totals_weather_info(venue, row.get("GameDate"))

    projected = clamp(base_total, 5.5, 14.5)

    return {
        "Base_Total": float(base_total),
        "Projected_Total": float(projected),
        "Park_Factor": float(park_factor),
        "Park_Known": bool(park),
        "Weather_Factor": float(safe_float(wx.get("Factor"), 1.0)),
        "Weather_Available": bool(wx.get("Available")),
        "Temp": wx.get("Temp"),
        "Wind": wx.get("Wind"),
        "Humidity": wx.get("Humidity"),
        "Precip": wx.get("Precip"),
        "Production_Note": "Park/weather shown as context only; not multiplied into the production total.",
    }

def reset_dynamic_caches():
    # Keep static person/team metadata, but refresh game feeds so lineups can appear during the day.
    _feed_cache.clear()


def run_model(games_to_run):
    if not games_to_run:
        return pd.DataFrame()
    if not TEAM_IDS:
        load_team_ids()
    reset_dynamic_caches()
    rows = []
    for g in games_to_run:
        asp = starter_quality_v2(g.get("Away_SP_ID"))
        hsp = starter_quality_v2(g.get("Home_SP_ID"))
        ahand = pitcher_hand(g.get("Away_SP_ID"))
        hhand = pitcher_hand(g.get("Home_SP_ID"))
        aip = expected_sp_ip(g.get("Away_SP_ID"))
        hip = expected_sp_ip(g.get("Home_SP_ID"))

        aline = get_lineup(g.get("GamePk"), "away")
        hline = get_lineup(g.get("GamePk"), "home")
        aoff = final_offense(g.get("Away"), aline, hhand)
        hoff = final_offense(g.get("Home"), hline, ahand)

        away_runs = BASE_RUNS_PER_TEAM * aoff["Factor"] * starter_run_factor(hsp["Quality"], hip)
        home_runs = BASE_RUNS_PER_TEAM * hoff["Factor"] * starter_run_factor(asp["Quality"], aip) + HOME_RUN_ADVANTAGE
        away_prob = win_prob(away_runs, home_runs)
        home_prob = 1.0-away_prob
        conf, conf_grade, conf_reasons = confidence_score(g, asp, hsp, aoff, hoff)

        rows.append({
            "Date": str(today_et()), "GamePk": g.get("GamePk"), "Game": f"{g.get('Away')} @ {g.get('Home')}",
            "Away": g.get("Away"), "Home": g.get("Home"), "Venue": g.get("Venue"), "TimeLabel": g.get("TimeLabel",""),
            "GameDate": g.get("GameDate"), "HoursToGame": g.get("HoursToGame"),
            "Away_SP": g.get("Away_SP"), "Home_SP": g.get("Home_SP"), "Away_SP_Hand": ahand, "Home_SP_Hand": hhand,
            "Away_SP_Quality": asp["Quality"], "Home_SP_Quality": hsp["Quality"],
            "Away_SP_Starts": asp["Starts"], "Home_SP_Starts": hsp["Starts"],
            "Away_SP_SeasonERA": asp.get("SeasonERA"), "Home_SP_SeasonERA": hsp.get("SeasonERA"),
            "Away_SP_SeasonFIP": asp.get("SeasonFIP"), "Home_SP_SeasonFIP": hsp.get("SeasonFIP"),
            "Away_SP_RecentERA": asp.get("RecentERA"), "Home_SP_RecentERA": hsp.get("RecentERA"),
            "Away_SP_RecentFIP": asp.get("RecentFIP"), "Home_SP_RecentFIP": hsp.get("RecentFIP"),
            "Away_SP_ExpIP": aip, "Home_SP_ExpIP": hip,
            "Away_Base_Offense": aoff["BaseFactor"], "Home_Base_Offense": hoff["BaseFactor"],
            "Away_Platoon_Factor": aoff["PlatoonFactor"], "Home_Platoon_Factor": hoff["PlatoonFactor"],
            "Away_Lineup_Factor": aoff["LineupFactor"], "Home_Lineup_Factor": hoff["LineupFactor"],
            "Away_Lineup_Used": aoff["LineupUsed"], "Home_Lineup_Used": hoff["LineupUsed"],
            "Away_Offense": aoff["Factor"], "Home_Offense": hoff["Factor"],
            "Away_Proj_Runs": away_runs, "Home_Proj_Runs": home_runs,
            "Away_WinProb": away_prob, "Home_WinProb": home_prob,
            "Away_FairML": fair_ml(away_prob), "Home_FairML": fair_ml(home_prob),
            "Model_Confidence": conf, "Confidence_Grade": conf_grade, "Confidence_Reasons": conf_reasons,
            "Lineup_Status": "CONFIRMED" if aoff["LineupUsed"] and hoff["LineupUsed"] else "PARTIAL/UNCONFIRMED",
            "Model_Version": MODEL_VERSION,
        })
    return pd.DataFrame(rows)
