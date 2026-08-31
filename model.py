
import math
import time
import warnings
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from pybaseball import statcast_pitcher
from scipy.stats import skellam

warnings.filterwarnings("ignore")

MODEL_VERSION = "0.5.4-LITE-MARKET-SAFE"
APP_VERSION = "0.6.0-WEB"
MLB_API = "https://statsapi.mlb.com/api"

BASE_RUNS_PER_TEAM = 4.45
HOME_RUN_ADVANTAGE = 0.12
PYTH_EXPONENT = 1.83

STARTER_IMPACT = 1.35
BULLPEN_IMPACT = 0.90
TEAM_STRENGTH_EXPONENT = 0.50

USE_PARK = True
USE_PLATOON = True
USE_WEATHER = True
USE_TRAVEL_REST = True
USE_BULLPEN_USAGE = True
USE_TEAM_STRENGTH = True

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

team_offense_cache = {}
team_strength_cache = {}
platoon_cache = {}
pitch_log_cache = {}
pitcher_season_cache = {}
starter_quality_cache = {}
pitcher_hand_cache = {}
roster_cache = {}
reliever_cache = {}
bullpen_cache = {}
feed_cache = {}
hitter_cache = {}
weather_cache = {}
rest_cache = {}

TEAM_IDS = {}
PROBABLE_BY_TEAM = {}

def today_et():
    return datetime.now(ZoneInfo("America/New_York")).date()

def season_now():
    return today_et().year

def get_json(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def safe_float(x, default=np.nan):
    try:
        if x in [None, "", "-", "--"]:
            return default
        return float(x)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))

def ip_to_decimal(ip):
    try:
        s = str(ip)
        if "." not in s:
            return float(s)
        whole, outs = s.split(".")
        return float(whole) + float(outs) / 3
    except Exception:
        return 0.0

def fair_ml(prob):
    prob = clamp(prob, 0.001, 0.999)
    if prob >= 0.50:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))

def implied_prob(odds):
    odds = float(odds)
    if odds == 0 or abs(odds) < 100:
        raise ValueError(f"Invalid American odds: {odds}")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def expected_value(prob, odds):
    odds = float(odds)
    if odds > 0:
        profit = odds / 100.0
    else:
        profit = 100.0 / abs(odds)
    return prob * profit - (1.0 - prob)

def win_prob(runs_for, runs_against):
    return (runs_for ** PYTH_EXPONENT) / (
        runs_for ** PYTH_EXPONENT + runs_against ** PYTH_EXPONENT
    )

def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return 0.0
    r = 3958.8
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def run_line_probs(away_runs, home_runs):
    away_runs = max(0.10, float(away_runs))
    home_runs = max(0.10, float(home_runs))
    return {
        "Away_-1.5_Prob": 1 - skellam.cdf(1, away_runs, home_runs),
        "Away_+1.5_Prob": 1 - skellam.cdf(-2, away_runs, home_runs),
        "Home_-1.5_Prob": skellam.cdf(-2, away_runs, home_runs),
        "Home_+1.5_Prob": skellam.cdf(1, away_runs, home_runs),
    }

def load_team_ids():
    global TEAM_IDS
    data = get_json(f"{MLB_API}/v1/teams", {"sportId": 1})
    TEAM_IDS = {t["name"]: t["id"] for t in data.get("teams", [])}
    return TEAM_IDS

def fetch_today_games():
    global TEAM_IDS
    if not TEAM_IDS:
        load_team_ids()

    today = today_et()
    schedule = get_json(
        f"{MLB_API}/v1/schedule",
        {
            "sportId": 1,
            "date": today.strftime("%Y-%m-%d"),
            "hydrate": "probablePitcher,venue",
        },
    )
    games = []
    for block in schedule.get("dates", []):
        for g in block.get("games", []):
            away = g["teams"]["away"]["team"]["name"]
            home = g["teams"]["home"]["team"]["name"]
            away_sp = g["teams"]["away"].get("probablePitcher", {})
            home_sp = g["teams"]["home"].get("probablePitcher", {})
            game_date = g.get("gameDate")
            try:
                game_dt = pd.to_datetime(game_date, utc=True)
                game_et = game_dt.tz_convert("America/New_York")
                time_label = game_et.strftime("%-I:%M %p")
            except Exception:
                time_label = ""
            games.append(
                {
                    "GamePk": g["gamePk"],
                    "Away": away,
                    "Home": home,
                    "Away_SP": away_sp.get("fullName"),
                    "Away_SP_ID": away_sp.get("id"),
                    "Home_SP": home_sp.get("fullName"),
                    "Home_SP_ID": home_sp.get("id"),
                    "Venue": g.get("venue", {}).get("name", ""),
                    "GameDate": game_date,
                    "GameNumber": safe_float(g.get("gameNumber"), 1),
                    "DoubleHeader": g.get("doubleHeader", "N"),
                    "TimeLabel": time_label,
                }
            )
    return games

def set_probable_pitchers(games):
    global PROBABLE_BY_TEAM
    PROBABLE_BY_TEAM = {}
    for g in games:
        if g["Away_SP_ID"]:
            PROBABLE_BY_TEAM[g["Away"]] = g["Away_SP_ID"]
        if g["Home_SP_ID"]:
            PROBABLE_BY_TEAM[g["Home"]] = g["Home_SP_ID"]

def pitcher_hand(player_id):
    if not player_id:
        return "U"
    if player_id in pitcher_hand_cache:
        return pitcher_hand_cache[player_id]
    data = get_json(f"{MLB_API}/v1/people/{player_id}")
    try:
        hand = data["people"][0].get("pitchHand", {}).get("code", "U")
    except Exception:
        hand = "U"
    pitcher_hand_cache[player_id] = hand
    return hand

def team_offense(team_name):
    if team_name in team_offense_cache:
        return team_offense_cache[team_name]
    team_id = TEAM_IDS.get(team_name)
    result = {"OPS": 0.720, "RunsPG": BASE_RUNS_PER_TEAM, "Factor": 1.00}
    if not team_id:
        return result
    data = get_json(
        f"{MLB_API}/v1/teams/{team_id}/stats",
        {"stats": "season", "group": "hitting", "season": season_now()},
    )
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
        ops = safe_float(stat.get("ops"), 0.720)
        runs = safe_float(stat.get("runs"), 0)
        gp = safe_float(stat.get("gamesPlayed"), 1)
        runs_pg = runs / max(1, gp)
        ops_factor = (ops / 0.720) ** 0.50
        run_factor = (runs_pg / BASE_RUNS_PER_TEAM) ** 0.30
        factor = (0.65 * ops_factor + 0.35 * run_factor) ** 1.20
        result = {"OPS": ops, "RunsPG": runs_pg, "Factor": clamp(factor, 0.82, 1.18)}
    except Exception:
        pass
    team_offense_cache[team_name] = result
    return result

def team_strength(team_name):
    if not USE_TEAM_STRENGTH:
        return {"RunsFor": np.nan, "RunsAgainst": np.nan, "RunDiffPG": 0.0, "Factor": 1.00, "Available": False}
    if team_name in team_strength_cache:
        return team_strength_cache[team_name]
    team_id = TEAM_IDS.get(team_name)
    result = {"RunsFor": np.nan, "RunsAgainst": np.nan, "RunDiffPG": 0.0, "Factor": 1.00, "Available": False}
    if not team_id:
        return result
    hitting = get_json(
        f"{MLB_API}/v1/teams/{team_id}/stats",
        {"stats": "season", "group": "hitting", "season": season_now()},
    )
    pitching = get_json(
        f"{MLB_API}/v1/teams/{team_id}/stats",
        {"stats": "season", "group": "pitching", "season": season_now()},
    )
    try:
        hs = hitting["stats"][0]["splits"][0]["stat"]
        ps = pitching["stats"][0]["splits"][0]["stat"]
        rf = safe_float(hs.get("runs"), np.nan)
        gp = safe_float(hs.get("gamesPlayed"), np.nan)
        ra = safe_float(ps.get("runs"), np.nan)
        if pd.isna(rf) or pd.isna(ra) or pd.isna(gp) or gp <= 0:
            raise ValueError()
        rd = (rf - ra) / gp
        factor = clamp(1.0 + rd / 10.0, 0.92, 1.08)
        result = {"RunsFor": rf, "RunsAgainst": ra, "RunDiffPG": rd, "Factor": factor, "Available": True}
    except Exception:
        pass
    team_strength_cache[team_name] = result
    return result

def team_platoon(team_name, opposing_pitcher_hand):
    if not USE_PLATOON or opposing_pitcher_hand not in ["L", "R"]:
        return {"OPS": 0.720, "Factor": 1.00, "Available": False}
    key = (team_name, opposing_pitcher_hand)
    if key in platoon_cache:
        return platoon_cache[key]
    team_id = TEAM_IDS.get(team_name)
    if not team_id:
        return {"OPS": 0.720, "Factor": 1.00, "Available": False}
    sit_code = "vl" if opposing_pitcher_hand == "L" else "vr"
    data = get_json(
        f"{MLB_API}/v1/teams/{team_id}/stats",
        {"stats": "season", "group": "hitting", "season": season_now(), "sitCodes": sit_code},
    )
    result = {"OPS": 0.720, "Factor": 1.00, "Available": False}
    try:
        splits = data["stats"][0]["splits"]
        if splits:
            ops = safe_float(splits[0]["stat"].get("ops"), np.nan)
            if not pd.isna(ops):
                result = {"OPS": ops, "Factor": clamp((ops / 0.720) ** 0.40, 0.88, 1.12), "Available": True}
    except Exception:
        pass
    platoon_cache[key] = result
    return result

def pitching_log(player_id):
    if not player_id:
        return pd.DataFrame()
    if player_id in pitch_log_cache:
        return pitch_log_cache[player_id]
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "gameLog", "group": "pitching", "season": season_now()},
    )
    rows = []
    try:
        for s in data["stats"][0]["splits"]:
            stat = s.get("stat", {})
            rows.append(
                {
                    "Date": s.get("date"),
                    "IP": ip_to_decimal(stat.get("inningsPitched", 0)),
                    "Pitches": safe_float(stat.get("numberOfPitches"), np.nan),
                    "Started": int(safe_float(stat.get("gamesStarted"), 0) > 0),
                }
            )
    except Exception:
        pass
    df = pd.DataFrame(rows)
    if len(df):
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    pitch_log_cache[player_id] = df
    return df

def pitcher_season(player_id):
    if not player_id:
        return {"Games": 0, "Starts": 0, "IP": 0, "IP_per_Start": np.nan}
    if player_id in pitcher_season_cache:
        return pitcher_season_cache[player_id]
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "season", "group": "pitching", "season": season_now()},
    )
    result = {"Games": 0, "Starts": 0, "IP": 0, "IP_per_Start": np.nan}
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
        gp = safe_float(stat.get("gamesPlayed"), 0)
        starts = safe_float(stat.get("gamesStarted"), 0)
        ip = ip_to_decimal(stat.get("inningsPitched", 0))
        result = {"Games": gp, "Starts": starts, "IP": ip, "IP_per_Start": ip / starts if starts > 0 else np.nan}
    except Exception:
        pass
    pitcher_season_cache[player_id] = result
    return result

def starter_profile(player_id):
    season = pitcher_season(player_id)
    log = pitching_log(player_id)
    gp, starts = season["Games"], season["Starts"]
    start_rate = starts / gp if gp > 0 else 0
    recent = log.head(8)
    recent_pitches = recent["Pitches"].dropna().mean() if len(recent) else np.nan
    recent_ip = recent["IP"].mean() if len(recent) else np.nan
    recent_start_rate = recent["Started"].mean() if len(recent) else 0
    if starts >= 5 and start_rate >= 0.50:
        role = "NORMAL_STARTER"
    elif gp >= 5 and start_rate < 0.20 and (
        (not pd.isna(recent_pitches) and recent_pitches <= 35)
        or (not pd.isna(recent_ip) and recent_ip <= 2.0)
    ):
        role = "OPENER"
    elif starts >= 1 or recent_start_rate >= 0.25:
        role = "SPOT_STARTER"
    else:
        role = "ROLE_UNCERTAIN"
    return {
        "Role": role,
        "Games": gp,
        "Starts": starts,
        "StartRate": start_rate,
        "IP_per_Start": season["IP_per_Start"],
        "RecentPitches": recent_pitches,
        "RecentIP": recent_ip,
    }

def expected_sp_ip(player_id, profile):
    log = pitching_log(player_id)
    role = profile["Role"]
    season_ip_start = profile["IP_per_Start"]
    recent_starts = log[log["Started"] == 1].head(5) if len(log) else pd.DataFrame()
    recent_all = log.head(6) if len(log) else pd.DataFrame()
    recent_start_ip = np.nan
    if len(recent_starts):
        vals = recent_starts["IP"].values
        weights = np.arange(len(vals), 0, -1)
        recent_start_ip = np.average(vals, weights=weights)
    recent_all_ip = recent_all["IP"].mean() if len(recent_all) else np.nan
    recent_pitches = recent_all["Pitches"].dropna().mean() if len(recent_all) else np.nan

    if role == "NORMAL_STARTER":
        if not pd.isna(recent_start_ip) and not pd.isna(season_ip_start):
            ip = 0.60 * recent_start_ip + 0.40 * season_ip_start
        elif not pd.isna(recent_start_ip):
            ip = recent_start_ip
        elif not pd.isna(season_ip_start):
            ip = season_ip_start
        else:
            ip = 5.25
        if not pd.isna(recent_pitches):
            if recent_pitches >= 95:
                ip += 0.20
            elif recent_pitches <= 75:
                ip -= 0.30
        return clamp(ip, 4.0, 7.0)

    if role == "OPENER":
        return clamp(recent_all_ip if not pd.isna(recent_all_ip) else 1.5, 0.7, 3.0)

    if role == "SPOT_STARTER":
        vals = [x for x in [recent_start_ip, recent_all_ip, season_ip_start] if not pd.isna(x)]
        return clamp(np.mean(vals) if vals else 4.0, 2.25, 5.50)

    return clamp(recent_all_ip if not pd.isna(recent_all_ip) else 3.75, 2.0, 5.25)

def starter_quality(player_id):
    if not player_id:
        return {"Quality": 1.00, "PA": 0}
    if player_id in starter_quality_cache:
        return starter_quality_cache[player_id]
    try:
        df = statcast_pitcher(f"{season_now()}-03-15", today_et().strftime("%Y-%m-%d"), player_id)
        terminal = df[df["events"].notna()].copy()
        pa = len(terminal)
        if pa == 0:
            raise ValueError()
        xw = terminal["estimated_woba_using_speedangle"].dropna()
        xwoba = xw.mean() if len(xw) else 0.315
        k_pct = terminal["events"].isin(["strikeout", "strikeout_double_play"]).mean()
        bb_pct = terminal["events"].isin(["walk", "intent_walk"]).mean()
        batted = df[df["launch_speed"].notna()]
        ev = batted["launch_speed"].mean() if len(batted) else 88.5
        hard_hit = (batted["launch_speed"] >= 95).mean() if len(batted) else 0.390
        composite = (
            0.35 * ((0.315 - xwoba) / 0.030)
            + 0.25 * ((k_pct - 0.225) / 0.060)
            + 0.15 * ((0.082 - bb_pct) / 0.030)
            + 0.10 * ((88.5 - ev) / 2.0)
            + 0.15 * ((0.390 - hard_hit) / 0.070)
        )
        raw_quality = 1.0 + 0.20 * np.tanh(composite / 1.5)
        shrink = min(1.0, pa / 250)
        quality = 1.0 + (raw_quality - 1.0) * shrink
        result = {"Quality": quality, "PA": pa}
    except Exception:
        result = {"Quality": 1.00, "PA": 0}
    starter_quality_cache[player_id] = result
    return result

def adjusted_starter_quality(raw_quality, pa, role):
    role_baselines = {
        "NORMAL_STARTER": 1.000,
        "SPOT_STARTER": 0.985,
        "OPENER": 0.975,
        "ROLE_UNCERTAIN": 0.970,
        "UNKNOWN": 0.975,
    }
    fallback = role_baselines.get(role, 0.980)
    if pa <= 0:
        return fallback
    if pa >= 100:
        return raw_quality
    sw = pa / 100
    return clamp(sw * raw_quality + (1 - sw) * fallback, 0.80, 1.20)

def active_roster(team_name):
    if team_name in roster_cache:
        return roster_cache[team_name]
    team_id = TEAM_IDS.get(team_name)
    if not team_id:
        return []
    data = get_json(
        f"{MLB_API}/v1/teams/{team_id}/roster",
        {"rosterType": "active", "season": season_now()},
    )
    roster = []
    for p in data.get("roster", []):
        roster.append(
            {
                "id": p["person"]["id"],
                "name": p["person"]["fullName"],
                "position": p.get("position", {}).get("abbreviation"),
            }
        )
    roster_cache[team_name] = roster
    return roster

def reliever_stats(player_id):
    if player_id in reliever_cache:
        return reliever_cache[player_id]
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "season", "group": "pitching", "season": season_now()},
    )
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
        result = {
            "ERA": safe_float(stat.get("era"), 4.20),
            "WHIP": safe_float(stat.get("whip"), 1.30),
            "K9": safe_float(stat.get("strikeoutsPer9Inn"), 8.5),
            "Saves": safe_float(stat.get("saves"), 0),
            "Holds": safe_float(stat.get("holds"), 0),
        }
    except Exception:
        result = None
    reliever_cache[player_id] = result
    return result

def reliever_usage(player_id):
    log = pitching_log(player_id)
    if len(log) == 0:
        return {"Availability": 1.00, "UsageWeight": 0.90, "Pitches1": 0, "Pitches2": 0}
    today_ts = pd.Timestamp(today_et())
    temp = log.copy()
    temp["DaysAgo"] = (today_ts - temp["Date"]).dt.days
    p1 = temp.loc[temp["DaysAgo"] <= 1, "Pitches"].fillna(0).sum()
    p2 = temp.loc[temp["DaysAgo"] <= 2, "Pitches"].fillna(0).sum()
    penalty = 0
    if p1 >= 30:
        penalty += 0.18
    elif p1 >= 20:
        penalty += 0.10
    if p2 >= 45:
        penalty += 0.10
    availability = clamp(1 - penalty, 0.65, 1.00)
    appearances_7 = len(temp[temp["DaysAgo"] <= 7])
    usage_weight = (0.85 + min(appearances_7 * 0.05, 0.25)) * (0.75 + 0.25 * availability)
    return {"Availability": availability, "UsageWeight": usage_weight, "Pitches1": p1, "Pitches2": p2}

def bullpen_quality(team_name):
    if team_name in bullpen_cache:
        return bullpen_cache[team_name]
    roster = active_roster(team_name)
    probable_id = PROBABLE_BY_TEAM.get(team_name)
    rows = []
    for p in roster:
        if p["id"] == probable_id:
            continue
        if p["position"] not in ["P", "RP", "CP", "SP"]:
            continue
        profile = starter_profile(p["id"])
        if profile["Role"] == "NORMAL_STARTER" and profile["StartRate"] >= 0.60:
            continue
        stats = reliever_stats(p["id"])
        if not stats:
            continue
        usage = reliever_usage(p["id"])
        era_score = 4.20 / max(stats["ERA"], 1.50)
        whip_score = 1.30 / max(stats["WHIP"], 0.70)
        k_score = max(stats["K9"], 4.0) / 8.5
        quality = clamp(0.45 * era_score + 0.35 * whip_score + 0.20 * k_score, 0.75, 1.30)
        leverage = stats["Saves"] + stats["Holds"]
        leverage_weight = 1.0 + min(leverage / 40, 0.60)
        final_weight = leverage_weight * usage["UsageWeight"] if USE_BULLPEN_USAGE else leverage_weight
        rows.append(
            {
                "Name": p["name"],
                "Quality": quality,
                "Availability": usage["Availability"],
                "Weight": final_weight,
            }
        )
        time.sleep(0.002)
    if not rows:
        result = {"Quality": 1.00, "Availability": 1.00, "Limited": "", "Relievers": 0}
    else:
        weights = np.array([x["Weight"] for x in rows])
        qualities = np.array([x["Quality"] for x in rows])
        avails = np.array([x["Availability"] for x in rows])
        raw_quality = np.average(qualities, weights=weights)
        availability = np.average(avails, weights=weights)
        adjusted_quality = raw_quality * (0.75 + 0.25 * availability)
        limited = [x["Name"] for x in rows if x["Availability"] < 0.85]
        result = {
            "Quality": clamp(adjusted_quality, 0.75, 1.30),
            "Availability": availability,
            "Limited": ", ".join(limited[:5]),
            "Relievers": len(rows),
        }
    bullpen_cache[team_name] = result
    return result

def game_feed(game_pk):
    if game_pk in feed_cache:
        return feed_cache[game_pk]
    data = get_json(f"{MLB_API}/v1.1/game/{game_pk}/feed/live")
    feed_cache[game_pk] = data
    return data

def get_lineup(game_pk, side):
    feed = game_feed(game_pk)
    try:
        team = feed["liveData"]["boxscore"]["teams"][side]
        order = team.get("battingOrder", [])
        players = team.get("players", {})
        return [
            {"id": pid, "name": players.get(f"ID{pid}", {}).get("person", {}).get("fullName")}
            for pid in order
        ]
    except Exception:
        return []

def hitter_ops(player_id):
    if player_id in hitter_cache:
        return hitter_cache[player_id]
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "season", "group": "hitting", "season": season_now()},
    )
    try:
        ops = safe_float(data["stats"][0]["splits"][0]["stat"].get("ops"), 0.720)
    except Exception:
        ops = 0.720
    hitter_cache[player_id] = ops
    return ops

def lineup_factor(lineup):
    if len(lineup) < 8:
        return 1.00
    weights = np.array([1.15, 1.12, 1.10, 1.08, 1.04, 1.00, 0.96, 0.92, 0.88])
    ops_values = [hitter_ops(p["id"]) for p in lineup[:9]]
    weighted_ops = np.average(ops_values, weights=weights[:len(ops_values)])
    return clamp((weighted_ops / 0.720) ** 0.45, 0.88, 1.12)

def final_offense(team_name, lineup, opposing_hand):
    base = team_offense(team_name)
    platoon = team_platoon(team_name, opposing_hand)
    lf = lineup_factor(lineup)
    lineup_used = len(lineup) >= 8
    if lineup_used:
        final = 0.50 * base["Factor"] + 0.20 * platoon["Factor"] + 0.30 * lf
    else:
        final = 0.70 * base["Factor"] + 0.30 * platoon["Factor"]
    return {
        "Factor": clamp(final, 0.80, 1.20),
        "BaseFactor": base["Factor"],
        "PlatoonFactor": platoon["Factor"],
        "PlatoonOPS": platoon["OPS"],
        "PlatoonAvailable": platoon["Available"],
        "LineupFactor": lf,
        "LineupUsed": lineup_used,
    }

def weather_info(venue, game_date):
    default = {"Temp": np.nan, "Wind": np.nan, "Humidity": np.nan, "Precip": np.nan, "Factor": 1.00, "Available": False}
    if not USE_WEATHER:
        return default
    park = PARKS.get(venue)
    if not park:
        return default
    key = (venue, str(game_date))
    if key in weather_cache:
        return weather_cache[key]
    try:
        game_dt = pd.to_datetime(game_date, utc=True)
        data = get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": park["lat"],
                "longitude": park["lon"],
                "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
                "forecast_days": 7,
            },
        )
        hourly = data.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), utc=True)
        if len(times) == 0:
            raise ValueError()
        diffs = np.abs((times - game_dt).total_seconds())
        idx = int(np.argmin(diffs))
        temp = safe_float(hourly.get("temperature_2m", [])[idx], np.nan)
        humidity = safe_float(hourly.get("relative_humidity_2m", [])[idx], np.nan)
        precip = safe_float(hourly.get("precipitation_probability", [])[idx], np.nan)
        wind = safe_float(hourly.get("wind_speed_10m", [])[idx], np.nan)
        factor = 1.00
        if not pd.isna(temp):
            factor *= 1 + (temp - 72) * 0.0015
        result = {
            "Temp": temp,
            "Wind": wind,
            "Humidity": humidity,
            "Precip": precip,
            "Factor": clamp(factor, 0.96, 1.04),
            "Available": True,
        }
    except Exception:
        result = default
    weather_cache[key] = result
    return result

def team_rest_context(team_name, today_venue, game_number=1):
    if not USE_TRAVEL_REST:
        return {"PlayedYesterday": False, "TravelMiles": 0, "ExtraInningsPrev": False, "Factor": 1.00}
    key = (team_name, today_venue, game_number)
    if key in rest_cache:
        return rest_cache[key]
    team_id = TEAM_IDS.get(team_name)
    result = {"PlayedYesterday": False, "TravelMiles": 0, "ExtraInningsPrev": False, "Factor": 1.00}
    if not team_id:
        return result

    yesterday = today_et() - timedelta(days=1)
    data = get_json(
        f"{MLB_API}/v1/schedule",
        {"sportId": 1, "teamId": team_id, "date": yesterday.strftime("%Y-%m-%d"), "hydrate": "venue"},
    )
    prev_games = [g for block in data.get("dates", []) for g in block.get("games", [])]
    factor = 1.00
    played = len(prev_games) > 0
    travel_miles = 0
    extra = False

    if played:
        prev = prev_games[-1]
        prev_venue = prev.get("venue", {}).get("name", "")
        prev_park = PARKS.get(prev_venue)
        today_park = PARKS.get(today_venue)
        if prev_park and today_park:
            travel_miles = haversine(prev_park["lat"], prev_park["lon"], today_park["lat"], today_park["lon"])
        try:
            prev_feed = game_feed(prev["gamePk"])
            inning = safe_float(prev_feed["liveData"]["linescore"].get("currentInning", 9), 9)
            extra = inning > 9
        except Exception:
            extra = False
        if travel_miles >= 2000:
            factor *= 0.985
        elif travel_miles >= 1000:
            factor *= 0.9925
        if extra:
            factor *= 0.9925
    else:
        factor *= 1.005

    if game_number >= 2:
        factor *= 0.985

    result = {
        "PlayedYesterday": played,
        "TravelMiles": travel_miles,
        "ExtraInningsPrev": extra,
        "Factor": clamp(factor, 0.96, 1.02),
    }
    rest_cache[key] = result
    return result

def pitching_factor(sp_quality, bp_quality, sp_ip):
    sp_ip = clamp(sp_ip, 0.7, 7.0)
    bp_ip = 9.0 - sp_ip
    starter_run_factor = math.exp(-STARTER_IMPACT * (sp_quality - 1.00))
    bullpen_run_factor = math.exp(-BULLPEN_IMPACT * (bp_quality - 1.00))
    factor = (sp_ip / 9.0) * starter_run_factor + (bp_ip / 9.0) * bullpen_run_factor
    return clamp(factor, 0.72, 1.32)

def confidence_score(
    away_sp_id, home_sp_id, away_profile, home_profile, away_sp, home_sp,
    away_lineup, home_lineup, away_bp, home_bp, park_known, weather_available,
    away_platoon_available, home_platoon_available, away_strength_available, home_strength_available
):
    score = 100
    reasons = []
    if not away_sp_id:
        score -= 18; reasons.append("AWAY SP UNKNOWN")
    if not home_sp_id:
        score -= 18; reasons.append("HOME SP UNKNOWN")
    if len(away_lineup) < 8:
        score -= 10; reasons.append("AWAY LINEUP MISSING")
    if len(home_lineup) < 8:
        score -= 10; reasons.append("HOME LINEUP MISSING")
    if away_sp["PA"] < 100:
        score -= 7; reasons.append("AWAY SP SMALL SAMPLE")
    if home_sp["PA"] < 100:
        score -= 7; reasons.append("HOME SP SMALL SAMPLE")
    for side, profile in [("AWAY", away_profile), ("HOME", home_profile)]:
        if profile["Role"] == "OPENER":
            score -= 5; reasons.append(f"{side} OPENER")
        elif profile["Role"] == "SPOT_STARTER":
            score -= 4
        elif profile["Role"] in ["ROLE_UNCERTAIN", "UNKNOWN"]:
            score -= 7
    if away_bp["Availability"] < 0.90:
        score -= 3
    if home_bp["Availability"] < 0.90:
        score -= 3
    if not park_known:
        score -= 4; reasons.append("PARK UNKNOWN")
    if USE_WEATHER and not weather_available:
        score -= 3
    if USE_PLATOON and not away_platoon_available:
        score -= 2
    if USE_PLATOON and not home_platoon_available:
        score -= 2
    if USE_TEAM_STRENGTH and not away_strength_available:
        score -= 2
    if USE_TEAM_STRENGTH and not home_strength_available:
        score -= 2
    score = int(clamp(score, 30, 100))
    grade = "HIGH" if score >= 85 else "MEDIUM" if score >= 70 else "LOW"
    return {"Score": score, "Grade": grade, "Reasons": " | ".join(reasons)}

def reset_dynamic_caches():
    bullpen_cache.clear()
    feed_cache.clear()
    weather_cache.clear()
    rest_cache.clear()
    platoon_cache.clear()
    team_offense_cache.clear()
    team_strength_cache.clear()
    roster_cache.clear()

def run_model(games_to_run):
    if not games_to_run:
        return pd.DataFrame()

    if not TEAM_IDS:
        load_team_ids()

    set_probable_pitchers(games_to_run)
    reset_dynamic_caches()
    rows = []

    for g in games_to_run:
        away_profile = starter_profile(g["Away_SP_ID"]) if g["Away_SP_ID"] else {
            "Role": "UNKNOWN", "Starts": 0, "StartRate": 0, "RecentPitches": np.nan, "IP_per_Start": np.nan
        }
        home_profile = starter_profile(g["Home_SP_ID"]) if g["Home_SP_ID"] else {
            "Role": "UNKNOWN", "Starts": 0, "StartRate": 0, "RecentPitches": np.nan, "IP_per_Start": np.nan
        }

        away_raw = starter_quality(g["Away_SP_ID"])
        home_raw = starter_quality(g["Home_SP_ID"])
        away_sp = {
            "Quality": adjusted_starter_quality(away_raw["Quality"], away_raw["PA"], away_profile["Role"]),
            "RawQuality": away_raw["Quality"],
            "PA": away_raw["PA"],
        }
        home_sp = {
            "Quality": adjusted_starter_quality(home_raw["Quality"], home_raw["PA"], home_profile["Role"]),
            "RawQuality": home_raw["Quality"],
            "PA": home_raw["PA"],
        }

        away_hand = pitcher_hand(g["Away_SP_ID"])
        home_hand = pitcher_hand(g["Home_SP_ID"])
        away_sp_ip = expected_sp_ip(g["Away_SP_ID"], away_profile) if g["Away_SP_ID"] else 4.5
        home_sp_ip = expected_sp_ip(g["Home_SP_ID"], home_profile) if g["Home_SP_ID"] else 4.5

        away_bp = bullpen_quality(g["Away"])
        home_bp = bullpen_quality(g["Home"])

        away_lineup = get_lineup(g["GamePk"], "away")
        home_lineup = get_lineup(g["GamePk"], "home")

        away_off = final_offense(g["Away"], away_lineup, home_hand)
        home_off = final_offense(g["Home"], home_lineup, away_hand)

        away_strength = team_strength(g["Away"])
        home_strength = team_strength(g["Home"])

        park = PARKS.get(g["Venue"])
        park_known = park is not None
        park_factor = park["factor"] if USE_PARK and park else 1.00

        weather = weather_info(g["Venue"], g["GameDate"])
        weather_factor = weather["Factor"] if USE_WEATHER else 1.00

        away_rest = team_rest_context(g["Away"], g["Venue"], g["GameNumber"])
        home_rest = team_rest_context(g["Home"], g["Venue"], g["GameNumber"])

        away_pitch_factor = pitching_factor(home_sp["Quality"], home_bp["Quality"], home_sp_ip)
        home_pitch_factor = pitching_factor(away_sp["Quality"], away_bp["Quality"], away_sp_ip)

        away_strength_mult = away_strength["Factor"] ** TEAM_STRENGTH_EXPONENT if USE_TEAM_STRENGTH else 1.00
        home_strength_mult = home_strength["Factor"] ** TEAM_STRENGTH_EXPONENT if USE_TEAM_STRENGTH else 1.00

        away_runs = (
            BASE_RUNS_PER_TEAM
            * away_off["Factor"]
            * away_pitch_factor
            * park_factor
            * weather_factor
            * away_rest["Factor"]
            * away_strength_mult
        )

        home_runs = (
            BASE_RUNS_PER_TEAM
            * home_off["Factor"]
            * home_pitch_factor
            * park_factor
            * weather_factor
            * home_rest["Factor"]
            * home_strength_mult
            + HOME_RUN_ADVANTAGE
        )

        total = away_runs + home_runs
        away_prob = win_prob(away_runs, home_runs)
        home_prob = 1 - away_prob
        rl = run_line_probs(away_runs, home_runs)

        confidence = confidence_score(
            g["Away_SP_ID"], g["Home_SP_ID"], away_profile, home_profile,
            away_sp, home_sp, away_lineup, home_lineup, away_bp, home_bp,
            park_known, weather["Available"], away_off["PlatoonAvailable"],
            home_off["PlatoonAvailable"], away_strength["Available"], home_strength["Available"]
        )

        warnings_list = []
        if away_profile["Role"] == "OPENER":
            warnings_list.append("AWAY OPENER")
        if home_profile["Role"] == "OPENER":
            warnings_list.append("HOME OPENER")
        if len(away_lineup) < 8:
            warnings_list.append("AWAY LINEUP MISSING")
        if len(home_lineup) < 8:
            warnings_list.append("HOME LINEUP MISSING")
        if away_sp["PA"] < 100:
            warnings_list.append("AWAY SP SMALL SAMPLE")
        if home_sp["PA"] < 100:
            warnings_list.append("HOME SP SMALL SAMPLE")
        if not park_known:
            warnings_list.append("PARK UNKNOWN")

        rows.append({
            "Date": str(today_et()),
            "GamePk": g["GamePk"],
            "Game": f"{g['Away']} @ {g['Home']}",
            "Away": g["Away"],
            "Home": g["Home"],
            "Venue": g["Venue"],
            "Away_SP": g["Away_SP"],
            "Home_SP": g["Home_SP"],
            "Away_SP_Hand": away_hand,
            "Home_SP_Hand": home_hand,
            "Away_SP_Role": away_profile["Role"],
            "Home_SP_Role": home_profile["Role"],
            "Away_SP_RawQuality": away_sp["RawQuality"],
            "Home_SP_RawQuality": home_sp["RawQuality"],
            "Away_SP_Quality": away_sp["Quality"],
            "Home_SP_Quality": home_sp["Quality"],
            "Away_SP_PA": away_sp["PA"],
            "Home_SP_PA": home_sp["PA"],
            "Away_SP_ExpIP": away_sp_ip,
            "Home_SP_ExpIP": home_sp_ip,
            "Away_BP_Quality": away_bp["Quality"],
            "Home_BP_Quality": home_bp["Quality"],
            "Away_BP_Availability": away_bp["Availability"],
            "Home_BP_Availability": home_bp["Availability"],
            "Away_BP_Limited": away_bp["Limited"],
            "Home_BP_Limited": home_bp["Limited"],
            "Away_Lineup_Used": away_off["LineupUsed"],
            "Home_Lineup_Used": home_off["LineupUsed"],
            "Away_Platoon_OPS": away_off["PlatoonOPS"],
            "Home_Platoon_OPS": home_off["PlatoonOPS"],
            "Away_Platoon_Factor": away_off["PlatoonFactor"],
            "Home_Platoon_Factor": home_off["PlatoonFactor"],
            "Away_Base_Offense": away_off["BaseFactor"],
            "Home_Base_Offense": home_off["BaseFactor"],
            "Away_Offense": away_off["Factor"],
            "Home_Offense": home_off["Factor"],
            "Away_RunDiffPG": away_strength["RunDiffPG"],
            "Home_RunDiffPG": home_strength["RunDiffPG"],
            "Away_TeamStrength": away_strength["Factor"],
            "Home_TeamStrength": home_strength["Factor"],
            "Away_TeamStrength_Mult": away_strength_mult,
            "Home_TeamStrength_Mult": home_strength_mult,
            "Park_Factor": park_factor,
            "Weather_Temp": weather["Temp"],
            "Weather_Wind": weather["Wind"],
            "Weather_Humidity": weather["Humidity"],
            "Weather_Precip": weather["Precip"],
            "Weather_Factor": weather_factor,
            "Away_Played_Yesterday": away_rest["PlayedYesterday"],
            "Home_Played_Yesterday": home_rest["PlayedYesterday"],
            "Away_Travel_Miles": away_rest["TravelMiles"],
            "Home_Travel_Miles": home_rest["TravelMiles"],
            "Away_Prev_Extra_Innings": away_rest["ExtraInningsPrev"],
            "Home_Prev_Extra_Innings": home_rest["ExtraInningsPrev"],
            "Away_Rest_Factor": away_rest["Factor"],
            "Home_Rest_Factor": home_rest["Factor"],
            "Away_FacingPitch_Factor": away_pitch_factor,
            "Home_FacingPitch_Factor": home_pitch_factor,
            "Away_Proj_Runs": away_runs,
            "Home_Proj_Runs": home_runs,
            "Projected_Run_Diff_AwayMinusHome": away_runs - home_runs,
            "Model_Total": total,
            "Away_WinProb": away_prob,
            "Home_WinProb": home_prob,
            "Away_FairML": fair_ml(away_prob),
            "Home_FairML": fair_ml(home_prob),
            "Away_-1.5_Prob": rl["Away_-1.5_Prob"],
            "Away_-1.5_FairML": fair_ml(rl["Away_-1.5_Prob"]),
            "Away_+1.5_Prob": rl["Away_+1.5_Prob"],
            "Away_+1.5_FairML": fair_ml(rl["Away_+1.5_Prob"]),
            "Home_-1.5_Prob": rl["Home_-1.5_Prob"],
            "Home_-1.5_FairML": fair_ml(rl["Home_-1.5_Prob"]),
            "Home_+1.5_Prob": rl["Home_+1.5_Prob"],
            "Home_+1.5_FairML": fair_ml(rl["Home_+1.5_Prob"]),
            "Model_Confidence": confidence["Score"],
            "Confidence_Grade": confidence["Grade"],
            "Confidence_Reasons": confidence["Reasons"],
            "RunLine_Status": "EXPERIMENTAL",
            "Data_Status": "OK" if not warnings_list else " | ".join(warnings_list),
            "Model_Version": MODEL_VERSION,
        })

    return pd.DataFrame(rows)
