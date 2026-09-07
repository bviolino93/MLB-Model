"""Ninth Signal — MLB betting app (single-file build).

This file contains BOTH the projection engine and the Streamlit app. There is
no separate model.py: everything the app needs is in here. Upload just this one
file to your repo.

Layout:
  PART 1  projection engine   (was model.py)
  PART 2  engine namespace    (lets the app code below call engine.* unchanged)
  PART 3  Streamlit app       (was app.py)
  PART 4  calibration panel   (new -- expander at the bottom of the app)
"""

import argparse
import json
import math
import re
import statistics
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =============================================================================
# PART 1 -- PROJECTION ENGINE
# =============================================================================
# MLB projection engine — rebuilt.
#
# DROP-IN REPLACEMENT for the previous model.py. The public interface is unchanged,
# so app.py works without edits:
#
#     MODEL_VERSION, today_et, now_et, fetch_games_for_date, fetch_today_games,
#     run_model, totals_projection, implied_prob, expected_value, fair_ml,
#     reset_dynamic_caches
#
# WHAT CHANGED AND WHY
# --------------------
# The previous engine squashed every input through tanh() with small amplitudes,
# then compressed the result again through blend weights, fractional exponents and
# clamps. Measured end to end, its entire possible output range was 8.22 to 9.92
# runs -- 1.7 runs of spread against market lines that span 6.5 to 11.5. It was
# projecting "about 9 runs" for every game.
#
# Five fixes:
#
#   1. RUN-SPACE MODEL, NOT FACTOR-SQUASH. Offense and pitching are converted to
#      actual runs per game using published run-scoring relationships, then
#      combined. No tanh anywhere.
#
#   2. PARK AND WEATHER ARE APPLIED. The old totals_projection computed both and
#      then discarded them (`projected = clamp(base_total, 5.5, 14.5)`).
#
#   3. BULLPEN EXISTS. The old model held the ~3.5 innings after the starter at
#      exactly league average for every team. Now modeled.
#
#   4. LEVEL ANCHOR CORRECTED. Old neutral output was 4.45*2 + 0.12 = 9.02 against
#      a sample mean market line of 8.02. That single constant was the entire
#      +0.87 run bias.
#
#   5. WIN PROBABILITY CAN NOW REACH REAL FAVORITES. The old model could not
#      produce a favorite stronger than about -135, which mechanically manufactured
#      a fake edge on every market dog priced worse than that.
#
# READ THIS BEFORE BETTING
# ------------------------
# The constants in the CALIBRATION block below are derived from published
# run-scoring relationships, NOT from a backtest of your data. They are a defensible
# starting point, not a validated model. Run `python model.py --backtest 45` and
# check the reported regression slope BEFORE putting money behind this. Instructions
# are in that function's docstring.
#
# Moneyline is computed but has NOT been revalidated. Leave it alone until the
# totals slope comes back near 1.0.

MODEL_VERSION = "2.0.0-DISPERSION-REBUILD"
MLB_API = "https://statsapi.mlb.com/api"

# =============================================================================
# CALIBRATION BLOCK -- these are the knobs. Tune these, not the code below.
# =============================================================================

# League baselines. Update once a season from actual league totals.
LEAGUE_RUNS_PER_TEAM = 4.30   # league average runs scored per team per game
LEAGUE_OPS = 0.720            # league average team OPS
LEAGUE_FIP = 4.20             # league average FIP
LEAGUE_BULLPEN_ERA = 4.05     # league average relief ERA

# Runs scale roughly as OPS^1.7 across the observed team range. This is the
# single most important dispersion parameter in the model.
#   OPS .640 -> factor 0.82   (~3.5 R/G)
#   OPS .720 -> factor 1.00   (~4.3 R/G)
#   OPS .800 -> factor 1.20   (~5.1 R/G)
# If your backtest slope comes back above 1.0, the model is still too flat and
# this exponent should go UP. Below 1.0, bring it down.
OFFENSE_EXPONENT = 1.70
OFFENSE_CLAMP = (0.74, 1.30)

# ERA/FIP -> RA9. Runs allowed run about 8% above earned runs.
RA9_MULTIPLIER = 1.08

# Pitching factor clamp, as a multiple of league RA9.
PITCHING_CLAMP = (0.62, 1.42)

# Home field has TWO separate components and they must not be conflated:
#   1. a small RUN edge (home teams score marginally more) -- affects TOTALS
#   2. batting last (the home team stops batting once ahead in the 9th and
#      wins immediately in extras) -- affects WIN PROBABILITY only
# The old model had only #1, set to 0.12, which produced a 51.5% home win rate
# against an MLB reality of 54.0%. Inflating the run edge to close that gap
# would have pushed every projected total up by ~0.3 runs and reintroduced the
# level bias we just removed. So the structural piece is applied as a logit
# shift on win probability instead, leaving totals untouched.
HOME_RUN_ADVANTAGE = 0.15          # runs; feeds totals AND win prob
HOME_STRUCTURAL_LOGIT = 0.1005     # win prob only; calibrated to 0.540

# Run distribution. MLB team-game runs are overdispersed relative to Poisson
# (mean ~4.30, variance ~9.60). Negative binomial: var = mu + mu^2/k.
RUN_DIST_K = 3.49
RUN_DIST_MAX = 32                  # runs cap for the convolution

# Price band you are actually willing to bet. Moneylines outside this band are
# graded PASS regardless of how much the model likes them -- they still appear
# on the full board for reference, they just stop being recommendations.
# This is a staking preference, not a model judgement: a -300 favourite at a
# true 78% is the same quality of bet as a +200 dog at a true 38%, it simply
# ties up more capital per unit of profit. Adjustable in the Diagnostics panel.
ML_PRICE_FLOOR = -175              # do not recommend favourites shorter than this
ML_PRICE_CEILING = 300             # do not recommend dogs longer than this

# Retained for reference only -- win_prob no longer uses it. See win_prob().
PYTH_EXPONENT = 1.83

# Park factors are already regressed toward 1.0 by the source. Applying them at
# full strength is correct; set below 1.0 only if backtesting says otherwise.
PARK_WEIGHT = 1.00

# Weather. Temperature is the only reliably usable signal without park
# orientation data for wind direction.
TEMP_RUNS_PER_DEGREE = 0.0025   # per degree F above/below 72
WEATHER_CLAMP = (0.94, 1.06)

# Global level correction, applied last to the total. Leave at 0.0 until you
# have backtest output. If the backtest reports a mean residual of -0.4, set
# this to -0.4.
TOTAL_CALIBRATION_OFFSET = 0.0

# Absolute sanity bounds on the final projected total.
TOTAL_CLAMP = (5.5, 15.5)

# Shrinkage. Small samples get pulled toward league average.
SP_SEASON_IP_ANCHOR = 70.0    # innings at which season stats are trusted fully
SP_RECENT_IP_ANCHOR = 28.0    # innings of recent form for max recent weight
SP_RECENT_MAX_WEIGHT = 0.35   # recent form never exceeds this share
OFFENSE_RECENT_PA_ANCHOR = 350.0
OFFENSE_RECENT_MAX_WEIGHT = 0.40
PLATOON_PA_ANCHOR = 180.0
PLATOON_CLAMP = (0.90, 1.10)
LINEUP_CLAMP = (0.88, 1.12)
BULLPEN_IP_ANCHOR = 120.0

# How the three offense components combine. Must sum to 1.0.
OFFENSE_BLEND_WITH_LINEUP = {"base": 0.62, "platoon": 0.18, "lineup": 0.20}
OFFENSE_BLEND_NO_LINEUP = {"base": 0.78, "platoon": 0.22}

# =============================================================================

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

TEAM_IDS = {}
_json_cache = {}
_pitcher_cache = {}
_hitting_cache = {}
_platoon_cache = {}
_bullpen_cache = {}
_feed_cache = {}
_hitter_cache = {}
_hand_cache = {}
_totals_weather_cache = {}


# ---------------------------------------------------------------- utilities --

def today_et():
    return datetime.now(ZoneInfo("America/New_York")).date()


def now_et():
    return datetime.now(ZoneInfo("America/New_York"))


def season_now():
    return today_et().year


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def ip_to_decimal(ip):
    """MLB innings pitched are recorded as 5.1 / 5.2 meaning 5 1/3 / 5 2/3."""
    v = safe_float(ip, 0.0)
    if not math.isfinite(v):
        return 0.0
    whole = int(v)
    frac = round((v - whole) * 10)
    if frac == 1:
        return whole + 1.0 / 3.0
    if frac == 2:
        return whole + 2.0 / 3.0
    return float(whole)


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


_NB_CACHE = {}


def _nb_pmf(mu, k=None, nmax=None):
    """Negative binomial PMF over 0..nmax runs. Cached on rounded mean."""
    k = RUN_DIST_K if k is None else k
    nmax = RUN_DIST_MAX if nmax is None else nmax
    mu = max(0.15, round(float(mu), 3))
    key = (mu, k, nmax)
    if key in _NB_CACHE:
        return _NB_CACHE[key]
    p = k / (k + mu)
    lp, lq = math.log(p), math.log1p(-p)
    lgk = math.lgamma(k)
    out = np.array([
        math.exp(math.lgamma(x + k) - lgk - math.lgamma(x + 1) + k * lp + x * lq)
        for x in range(nmax + 1)
    ])
    out = out / out.sum()
    if len(_NB_CACHE) > 4000:
        _NB_CACHE.clear()
    _NB_CACHE[key] = out
    return out


def win_prob(runs_for, runs_against, home_is_second=True):
    """P(first team wins). Convention: win_prob(away_runs, home_runs).

    Replaces the Pythagorean formula. Pythagorean is a SEASON-level identity;
    applied to a single game it overstates favorites badly, because one game's
    outcome is dominated by the lumpiness of run scoring rather than by the
    difference in means. Measured error at the extremes:

        proj runs     Pythagorean     this model
        5.2 v 3.8        -178            -159
        6.0 v 3.0        -356            -239

    Here both teams' run totals are modelled as negative binomial and the two
    distributions are convolved exactly, so P(win) falls out by construction.
    Ties go to extra innings, which the home side wins about 52% of the time.
    A logit shift then adds the batting-last advantage.
    """
    rf = max(0.01, float(runs_for))
    ra = max(0.01, float(runs_against))

    A = _nb_pmf(rf)
    H = _nb_pmf(ra)
    cumH = np.cumsum(H)
    p_first = float((A[1:] * cumH[:-1]).sum())      # first team outscores
    p_tie = float((A * H).sum())
    p_first += p_tie * 0.48                          # extras: home wins ~52%

    if home_is_second and HOME_STRUCTURAL_LOGIT:
        p_home = clamp(1.0 - p_first, 1e-6, 1 - 1e-6)
        z = math.log(p_home / (1 - p_home)) + HOME_STRUCTURAL_LOGIT
        p_first = 1.0 - 1.0 / (1.0 + math.exp(-z))

    return clamp(p_first, 0.001, 0.999)


def get_json(url, params=None, cache_key=None):
    if cache_key is not None and cache_key in _json_cache:
        return _json_cache[cache_key]
    try:
        r = requests.get(url, params=params or {}, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        data = {}
    if cache_key is not None:
        _json_cache[cache_key] = data
    return data


def load_team_ids():
    data = get_json(f"{MLB_API}/v1/teams", {"sportId": 1}, cache_key=("teams", season_now()))
    for t in data.get("teams", []):
        TEAM_IDS[t.get("name")] = t.get("id")
    return TEAM_IDS


# ------------------------------------------------------------------ schedule --

def fetch_games_for_date(selected_date=None):
    d = selected_date or today_et()
    data = get_json(
        f"{MLB_API}/v1/schedule",
        {"sportId": 1, "date": str(d),
         "hydrate": "probablePitcher,team,venue,linescore,game(content(summary))"},
        cache_key=("sched", str(d)),
    )
    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            teams = g.get("teams", {})
            away, home = teams.get("away", {}), teams.get("home", {})
            gd = g.get("gameDate")
            hours = np.nan
            if gd:
                try:
                    hours = (pd.to_datetime(gd, utc=True)
                             - pd.Timestamp.utcnow()).total_seconds() / 3600.0
                except Exception:
                    hours = np.nan
            time_label = ""
            if gd:
                try:
                    time_label = (pd.to_datetime(gd, utc=True)
                                  .tz_convert("America/New_York")
                                  .strftime("%-I:%M %p ET"))
                except Exception:
                    time_label = ""
            games.append({
                "GamePk": g.get("gamePk"),
                "GameDate": gd,
                "TimeLabel": time_label,
                "HoursToGame": hours,
                "Away": away.get("team", {}).get("name"),
                "Home": home.get("team", {}).get("name"),
                "Venue": g.get("venue", {}).get("name", ""),
                "Away_SP": away.get("probablePitcher", {}).get("fullName"),
                "Home_SP": home.get("probablePitcher", {}).get("fullName"),
                "Away_SP_ID": away.get("probablePitcher", {}).get("id"),
                "Home_SP_ID": home.get("probablePitcher", {}).get("id"),
            })
    return games


def fetch_today_games():
    return fetch_games_for_date(today_et())


# ------------------------------------------------------------------ pitching --

def pitcher_hand(player_id):
    if not player_id:
        return None
    if player_id in _hand_cache:
        return _hand_cache[player_id]
    data = get_json(f"{MLB_API}/v1/people/{player_id}", cache_key=("person", player_id))
    try:
        hand = data["people"][0]["pitchHand"]["code"]
    except Exception:
        hand = None
    _hand_cache[player_id] = hand
    return hand


def _pitcher_season_stats(player_id):
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "season", "group": "pitching", "season": season_now()},
        cache_key=("psn", season_now(), player_id),
    )
    try:
        return data["stats"][0]["splits"][0]["stat"]
    except Exception:
        return {}


def _pitcher_game_log(player_id):
    data = get_json(
        f"{MLB_API}/v1/people/{player_id}/stats",
        {"stats": "gameLog", "group": "pitching", "season": season_now()},
        cache_key=("plog", season_now(), player_id),
    )
    rows = []
    try:
        splits = data["stats"][0]["splits"]
    except Exception:
        splits = []
    for s in splits:
        st = s.get("stat", {})
        rows.append({
            "Date": s.get("date"),
            "Started": 1 if safe_float(st.get("gamesStarted"), 0) >= 1 else 0,
            "IP": ip_to_decimal(st.get("inningsPitched", 0)),
            "ER": safe_float(st.get("earnedRuns"), np.nan),
            "H": safe_float(st.get("hits"), np.nan),
            "BB": safe_float(st.get("baseOnBalls"), np.nan),
            "K": safe_float(st.get("strikeOuts"), np.nan),
            "HR": safe_float(st.get("homeRuns"), np.nan),
            "Pitches": safe_float(st.get("numberOfPitches"), np.nan),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df


def _fip_from_counts(ip, hr, bb, k):
    """FIP constant is absorbed by anchoring the result at LEAGUE_FIP."""
    if not ip or ip <= 0:
        return np.nan
    if not all(math.isfinite(v) for v in [hr, bb, k]):
        return np.nan
    return (13.0 * hr + 3.0 * bb - 2.0 * k) / ip + 3.15


def starter_quality_v2(player_id):
    """Return the starter's expected RA9, shrunk toward league average.

    The old version returned a 'Quality' multiplier squashed through tanh with
    amplitude 0.16, which compressed an ace and a replacement-level starter into
    a 0.60-run difference. This returns runs allowed per 9 directly, so an ace
    at 3.0 and a bad starter at 6.0 stay 3.0 runs apart before innings weighting.
    """
    blank = {
        "RA9": LEAGUE_FIP * RA9_MULTIPLIER, "Quality": 1.0, "Starts": 0, "IP": 0,
        "RecentStarts": 0, "SeasonERA": np.nan, "SeasonFIP": np.nan,
        "RecentERA": np.nan, "RecentFIP": np.nan, "RecentK9": np.nan,
        "RecentBB9": np.nan, "RecentPitches": np.nan,
    }
    if not player_id:
        return blank
    if player_id in _pitcher_cache:
        return _pitcher_cache[player_id]

    st = _pitcher_season_stats(player_id)
    log = _pitcher_game_log(player_id)

    starts = int(safe_float(st.get("gamesStarted"), 0))
    ip = ip_to_decimal(st.get("inningsPitched", 0))
    er = safe_float(st.get("earnedRuns"), np.nan)
    bb = safe_float(st.get("baseOnBalls"), np.nan)
    k = safe_float(st.get("strikeOuts"), np.nan)
    hr = safe_float(st.get("homeRuns"), np.nan)

    season_era = (9.0 * er / ip) if (ip > 0 and math.isfinite(er)) else np.nan
    season_fip = _fip_from_counts(ip, hr, bb, k)

    # Blend ERA and FIP, favouring FIP as the more stable predictor.
    if math.isfinite(season_fip) and math.isfinite(season_era):
        season_skill = 0.70 * season_fip + 0.30 * season_era
    elif math.isfinite(season_fip):
        season_skill = season_fip
    elif math.isfinite(season_era):
        season_skill = season_era
    else:
        season_skill = LEAGUE_FIP

    # Shrink toward league average based on sample size.
    w_season = clamp(ip / SP_SEASON_IP_ANCHOR, 0.0, 1.0)
    skill = w_season * season_skill + (1 - w_season) * LEAGUE_FIP

    # Recent form, capped so a hot or cold month cannot dominate.
    recent = log[log["Started"] == 1].head(5) if not log.empty else pd.DataFrame()
    recent_era = recent_fip = np.nan
    if not recent.empty:
        r_ip = float(recent["IP"].sum())
        if r_ip > 0:
            r_er, r_hr = recent["ER"].sum(), recent["HR"].sum()
            r_bb, r_k = recent["BB"].sum(), recent["K"].sum()
            recent_era = 9.0 * r_er / r_ip
            recent_fip = _fip_from_counts(r_ip, r_hr, r_bb, r_k)
            if math.isfinite(recent_fip):
                w_recent = clamp(r_ip / SP_RECENT_IP_ANCHOR, 0.0, 1.0) * SP_RECENT_MAX_WEIGHT
                skill = (1 - w_recent) * skill + w_recent * recent_fip

    # Relievers and openers masquerading as starters get pulled to league average.
    role_weight = clamp(starts / 6.0, 0.25, 1.0)
    skill = role_weight * skill + (1 - role_weight) * LEAGUE_FIP

    ra9 = clamp(skill * RA9_MULTIPLIER, 2.20, 8.00)
    recent_pitches = (float(recent["Pitches"].dropna().mean())
                      if not recent.empty and recent["Pitches"].notna().any() else np.nan)

    result = {
        "RA9": ra9,
        "Quality": LEAGUE_FIP * RA9_MULTIPLIER / ra9,   # >1 is good, kept for display
        "Starts": starts,
        "IP": ip,
        "RecentStarts": len(recent),
        "SeasonERA": season_era,
        "SeasonFIP": season_fip,
        "RecentERA": recent_era,
        "RecentFIP": recent_fip,
        "RecentK9": (9.0 * recent["K"].sum() / recent["IP"].sum())
                    if (not recent.empty and recent["IP"].sum() > 0) else np.nan,
        "RecentBB9": (9.0 * recent["BB"].sum() / recent["IP"].sum())
                     if (not recent.empty and recent["IP"].sum() > 0) else np.nan,
        "RecentPitches": recent_pitches,
    }
    _pitcher_cache[player_id] = result
    return result


def expected_sp_ip(player_id):
    if not player_id:
        return 5.0
    log = _pitcher_game_log(player_id)
    starts = log[log["Started"] == 1].head(5) if not log.empty else pd.DataFrame()
    if starts.empty:
        p = starter_quality_v2(player_id)
        n = max(1, p.get("Starts", 0))
        return clamp(p.get("IP", 0) / n if p.get("Starts", 0) else 5.0, 4.0, 6.4)
    vals = starts["IP"].to_numpy(dtype=float)
    weights = np.arange(len(vals), 0, -1, dtype=float)
    ip = float(np.average(vals, weights=weights))
    pitches = starts["Pitches"].dropna()
    if len(pitches):
        avg = float(pitches.mean())
        if avg >= 95:
            ip += 0.15
        elif avg <= 75:
            ip -= 0.25
    return clamp(ip, 3.8, 6.8)


def team_bullpen(team_name):
    """Relief-corps RA9, shrunk toward league average.

    NEW. The previous model held every team's bullpen at exactly league average,
    which froze roughly 40% of every game as a constant. Real relief ERAs span
    about 2.90 to 5.20.

    The MLB API relief split is requested via sitCodes=rp. If that call fails or
    returns nothing, this falls back to league average, which reproduces the old
    behaviour rather than crashing.
    """
    if team_name in _bullpen_cache:
        return _bullpen_cache[team_name]
    if not TEAM_IDS:
        load_team_ids()
    tid = TEAM_IDS.get(team_name)
    fallback = {"RA9": LEAGUE_BULLPEN_ERA * RA9_MULTIPLIER, "IP": 0.0, "Available": False}
    if not tid:
        _bullpen_cache[team_name] = fallback
        return fallback

    data = get_json(
        f"{MLB_API}/v1/teams/{tid}/stats",
        {"stats": "season", "group": "pitching", "season": season_now(), "sitCodes": "rp"},
        cache_key=("bp", season_now(), tid),
    )
    stat = {}
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
    except Exception:
        stat = {}

    ip = ip_to_decimal(stat.get("inningsPitched", 0))
    er = safe_float(stat.get("earnedRuns"), np.nan)
    if ip <= 0 or not math.isfinite(er):
        _bullpen_cache[team_name] = fallback
        return fallback

    era = 9.0 * er / ip
    w = clamp(ip / BULLPEN_IP_ANCHOR, 0.0, 1.0)
    shrunk = w * era + (1 - w) * LEAGUE_BULLPEN_ERA
    result = {"RA9": clamp(shrunk * RA9_MULTIPLIER, 2.80, 7.00),
              "IP": ip, "Available": True}
    _bullpen_cache[team_name] = result
    return result


# ------------------------------------------------------------------- offense --

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
    if stats_type == "byDateRange":
        params["startDate"] = str(start_date)
        params["endDate"] = str(end_date)
    data = get_json(f"{MLB_API}/v1/teams/{tid}/stats", params,
                    cache_key=("hit", season_now(), tid, stats_type, str(start_date)))
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
    except Exception:
        stat = {}
    _hitting_cache[key] = stat
    return stat


def _hitting_rates(stat):
    if not stat:
        return None
    pa = safe_float(stat.get("plateAppearances"), np.nan)
    ab = safe_float(stat.get("atBats"), np.nan)
    bb = safe_float(stat.get("baseOnBalls"), 0)
    hbp = safe_float(stat.get("hitByPitch"), 0)
    sf = safe_float(stat.get("sacFlies"), 0)
    ops = safe_float(stat.get("ops"), np.nan)
    if not math.isfinite(pa):
        pa = (ab if math.isfinite(ab) else 0) + bb + hbp + sf
    if pa <= 0:
        return None
    return {"PA": pa, "OPS": ops if math.isfinite(ops) else LEAGUE_OPS}


def team_offense(team_name):
    """Offense as a multiple of league-average runs scored.

    Old version: 1.0 + 0.10*tanh(composite/1.8), which mapped the entire MLB OPS
    range (.640 to .800) onto factors of 0.972 to 1.028 -- a 0.24-run spread.
    New version uses the empirical runs ~ OPS^1.7 relationship, giving 0.82 to
    1.20, or roughly 1.6 runs.
    """
    season_stat = _team_hitting_stats(team_name, "season")
    season = _hitting_rates(season_stat) or {"PA": 0, "OPS": LEAGUE_OPS}

    start = today_et() - timedelta(days=14)
    recent = _hitting_rates(_team_hitting_stats(team_name, "byDateRange", start, today_et()))

    if recent and recent["PA"] >= 120:
        rw = clamp(recent["PA"] / (recent["PA"] + OFFENSE_RECENT_PA_ANCHOR),
                   0.0, OFFENSE_RECENT_MAX_WEIGHT)
        ops = (1 - rw) * season["OPS"] + rw * recent["OPS"]
    else:
        rw = 0.0
        ops = season["OPS"]

    # Shrink toward league average early in the season.
    w = clamp(season["PA"] / 1500.0, 0.0, 1.0)
    ops = w * ops + (1 - w) * LEAGUE_OPS

    factor = clamp((ops / LEAGUE_OPS) ** OFFENSE_EXPONENT, *OFFENSE_CLAMP)
    return {"Factor": factor, "OPS": ops, "RecentUsed": bool(rw > 0), "PA": season["PA"]}


def team_platoon(team_name, opposing_hand):
    if opposing_hand not in ("L", "R"):
        return {"Factor": 1.0, "OPS": LEAGUE_OPS, "Available": False, "PA": 0}
    key = (team_name, opposing_hand)
    if key in _platoon_cache:
        return _platoon_cache[key]
    if not TEAM_IDS:
        load_team_ids()
    tid = TEAM_IDS.get(team_name)
    if not tid:
        return {"Factor": 1.0, "OPS": LEAGUE_OPS, "Available": False, "PA": 0}

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
        result = {"Factor": 1.0, "OPS": LEAGUE_OPS, "Available": False,
                  "PA": rates["PA"] if rates else 0}
    else:
        shrink = rates["PA"] / (rates["PA"] + PLATOON_PA_ANCHOR)
        ops = shrink * rates["OPS"] + (1 - shrink) * LEAGUE_OPS
        result = {"Factor": clamp((ops / LEAGUE_OPS) ** OFFENSE_EXPONENT, *PLATOON_CLAMP),
                  "OPS": ops, "Available": True, "PA": rates["PA"]}
    _platoon_cache[key] = result
    return result


def game_feed(game_pk):
    if game_pk in _feed_cache:
        return _feed_cache[game_pk]
    data = get_json(f"{MLB_API}/v1.1/game/{game_pk}/feed/live", cache_key=None)
    _feed_cache[game_pk] = data
    return data


def get_lineup(game_pk, side):
    try:
        team = game_feed(game_pk)["liveData"]["boxscore"]["teams"][side]
        order = team.get("battingOrder", [])
        players = team.get("players", {})
        return [{"id": pid,
                 "name": players.get(f"ID{pid}", {}).get("person", {}).get("fullName", "")}
                for pid in order[:9]]
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
        ops = safe_float(stat.get("ops"), LEAGUE_OPS)
        pa = safe_float(stat.get("plateAppearances"), 0)
    except Exception:
        ops, pa = LEAGUE_OPS, 0
    if pa < 50:
        w = pa / (pa + 100.0)
        ops = w * ops + (1 - w) * LEAGUE_OPS
    _hitter_cache[key] = ops
    return ops


def lineup_factor(lineup):
    if len(lineup) < 8:
        return 1.0
    weights = np.array([1.15, 1.12, 1.10, 1.08, 1.04, 1.00, .96, .92, .88], dtype=float)
    vals = np.array([hitter_ops(p["id"]) for p in lineup[:9]], dtype=float)
    weighted = float(np.average(vals, weights=weights[:len(vals)]))
    return clamp((weighted / LEAGUE_OPS) ** OFFENSE_EXPONENT, *LINEUP_CLAMP)


def final_offense(team_name, lineup, opposing_hand):
    base = team_offense(team_name)
    platoon = team_platoon(team_name, opposing_hand)
    lineup_used = len(lineup) >= 8
    lf = lineup_factor(lineup) if lineup_used else 1.0

    # Renormalise over AVAILABLE components only. Previously an unavailable
    # platoon split was set to a neutral 1.0 and still given its full blend
    # weight -- which is not an absence, it is an assertion that the team is
    # exactly league average against that handedness. That silently shrank
    # every team's deviation from neutral by 22% (no lineup, no platoon) to
    # 38% (lineups only), in both directions. Dropping the weight instead of
    # blending against 1.0 removes that compression.
    parts = [(OFFENSE_BLEND_WITH_LINEUP["base"] if lineup_used
              else OFFENSE_BLEND_NO_LINEUP["base"], base["Factor"])]
    if platoon["Available"]:
        parts.append(((OFFENSE_BLEND_WITH_LINEUP if lineup_used
                       else OFFENSE_BLEND_NO_LINEUP)["platoon"], platoon["Factor"]))
    if lineup_used:
        parts.append((OFFENSE_BLEND_WITH_LINEUP["lineup"], lf))
    _w = sum(w for w, _ in parts)
    factor = sum(w * v for w, v in parts) / _w if _w else base["Factor"]

    return {
        "Factor": clamp(factor, *OFFENSE_CLAMP),
        "BaseFactor": base["Factor"], "PlatoonFactor": platoon["Factor"],
        "PlatoonOPS": platoon["OPS"], "PlatoonAvailable": platoon["Available"],
        "LineupFactor": lf, "LineupUsed": lineup_used,
        "RecentOffenseUsed": base["RecentUsed"],
    }


# ------------------------------------------------------------ park / weather --

def totals_weather_info(venue, game_date):
    default = {"Temp": np.nan, "Wind": np.nan, "Humidity": np.nan, "Precip": np.nan,
               "Factor": 1.00, "Available": False}
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
             "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
             "timezone": "UTC", "forecast_days": 14},
            cache_key=("wx", venue, str(game_date)),
        )
        hourly = data.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), utc=True)
        if len(times) == 0:
            raise ValueError("weather unavailable")
        idx = int(np.argmin(np.abs((times - game_dt).total_seconds())))
        temp = safe_float(hourly.get("temperature_2m", [])[idx], np.nan)
        humidity = safe_float(hourly.get("relative_humidity_2m", [])[idx], np.nan)
        precip = safe_float(hourly.get("precipitation_probability", [])[idx], np.nan)
        wind = safe_float(hourly.get("wind_speed_10m", [])[idx], np.nan)

        factor = 1.0
        if math.isfinite(temp):
            factor *= 1.0 + (temp - 72.0) * TEMP_RUNS_PER_DEGREE
        result = {"Temp": temp, "Wind": wind, "Humidity": humidity, "Precip": precip,
                  "Factor": clamp(factor, *WEATHER_CLAMP), "Available": True}
    except Exception:
        result = default
    _totals_weather_cache[key] = result
    return result


def totals_projection(row):
    """Final projected total, with park and weather APPLIED.

    The old version computed park_factor and weather factor and then threw both
    away: `projected = clamp(base_total, 5.5, 14.5)`. That is why every Coors
    game produced an UNDER -- the model projected ~9.3 into an 11.0 line because
    it could not see the park.
    """
    base_total = (safe_float(row.get("Away_Proj_Runs"), LEAGUE_RUNS_PER_TEAM)
                  + safe_float(row.get("Home_Proj_Runs"), LEAGUE_RUNS_PER_TEAM))

    venue = row.get("Venue", "")
    park = PARKS.get(venue, {})
    raw_park = safe_float(park.get("factor"), 1.0)
    park_factor = 1.0 + (raw_park - 1.0) * PARK_WEIGHT

    wx = totals_weather_info(venue, row.get("GameDate"))
    weather_factor = safe_float(wx.get("Factor"), 1.0)

    projected = base_total * park_factor * weather_factor + TOTAL_CALIBRATION_OFFSET
    projected = clamp(projected, *TOTAL_CLAMP)

    return {
        "Base_Total": float(base_total),
        "Projected_Total": float(projected),
        "Park_Factor": float(park_factor),
        "Park_Known": bool(park),
        "Weather_Factor": float(weather_factor),
        "Weather_Available": bool(wx.get("Available")),
        "Temp": wx.get("Temp"), "Wind": wx.get("Wind"),
        "Humidity": wx.get("Humidity"), "Precip": wx.get("Precip"),
        "Production_Note": "Park and weather ARE applied to the production total.",
    }


# ---------------------------------------------------------------- confidence --

def confidence_score(g, away_sp, home_sp, away_off, home_off, away_bp, home_bp):
    score = 100
    reasons = []
    if not g.get("Away_SP_ID"):
        score -= 24; reasons.append("away starter unknown")
    if not g.get("Home_SP_ID"):
        score -= 24; reasons.append("home starter unknown")
    if away_sp.get("Starts", 0) < 3:
        score -= 10; reasons.append("away starter small sample")
    if home_sp.get("Starts", 0) < 3:
        score -= 10; reasons.append("home starter small sample")
    if not away_bp.get("Available"):
        score -= 5; reasons.append("away bullpen unavailable")
    if not home_bp.get("Available"):
        score -= 5; reasons.append("home bullpen unavailable")
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


# ------------------------------------------------------------------ the model --

def _pitching_factor(sp_ra9, bp_ra9, sp_ip):
    """Blend starter and bullpen into one runs-allowed multiplier."""
    ip = clamp(sp_ip, 3.0, 7.5)
    combined_ra9 = (ip / 9.0) * sp_ra9 + ((9.0 - ip) / 9.0) * bp_ra9
    league_ra9 = LEAGUE_RUNS_PER_TEAM * RA9_MULTIPLIER
    return clamp(combined_ra9 / league_ra9, *PITCHING_CLAMP), combined_ra9


def reset_dynamic_caches():
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

        abp = team_bullpen(g.get("Away"))
        hbp = team_bullpen(g.get("Home"))

        aline = get_lineup(g.get("GamePk"), "away")
        hline = get_lineup(g.get("GamePk"), "home")
        aoff = final_offense(g.get("Away"), aline, hhand)
        hoff = final_offense(g.get("Home"), hline, ahand)

        # Away scores against the HOME pitching staff, and vice versa.
        home_pitch_factor, home_ra9 = _pitching_factor(hsp["RA9"], hbp["RA9"], hip)
        away_pitch_factor, away_ra9 = _pitching_factor(asp["RA9"], abp["RA9"], aip)

        away_runs = LEAGUE_RUNS_PER_TEAM * aoff["Factor"] * home_pitch_factor
        home_runs = (LEAGUE_RUNS_PER_TEAM * hoff["Factor"] * away_pitch_factor
                     + HOME_RUN_ADVANTAGE)

        away_prob = win_prob(away_runs, home_runs)
        home_prob = 1.0 - away_prob
        conf, conf_grade, conf_reasons = confidence_score(g, asp, hsp, aoff, hoff, abp, hbp)

        rows.append({
            "Date": str(today_et()), "GamePk": g.get("GamePk"),
            "Game": f"{g.get('Away')} @ {g.get('Home')}",
            "Away": g.get("Away"), "Home": g.get("Home"), "Venue": g.get("Venue"),
            "TimeLabel": g.get("TimeLabel", ""), "GameDate": g.get("GameDate"),
            "HoursToGame": g.get("HoursToGame"),
            "Away_SP": g.get("Away_SP"), "Home_SP": g.get("Home_SP"),
            "Away_SP_Hand": ahand, "Home_SP_Hand": hhand,
            "Away_SP_Quality": asp["Quality"], "Home_SP_Quality": hsp["Quality"],
            "Away_SP_RA9": asp["RA9"], "Home_SP_RA9": hsp["RA9"],
            "Away_SP_Starts": asp["Starts"], "Home_SP_Starts": hsp["Starts"],
            "Away_SP_SeasonERA": asp.get("SeasonERA"), "Home_SP_SeasonERA": hsp.get("SeasonERA"),
            "Away_SP_SeasonFIP": asp.get("SeasonFIP"), "Home_SP_SeasonFIP": hsp.get("SeasonFIP"),
            "Away_SP_RecentERA": asp.get("RecentERA"), "Home_SP_RecentERA": hsp.get("RecentERA"),
            "Away_SP_RecentFIP": asp.get("RecentFIP"), "Home_SP_RecentFIP": hsp.get("RecentFIP"),
            "Away_SP_ExpIP": aip, "Home_SP_ExpIP": hip,
            "Away_Bullpen_RA9": abp["RA9"], "Home_Bullpen_RA9": hbp["RA9"],
            "Away_Bullpen_Available": abp["Available"], "Home_Bullpen_Available": hbp["Available"],
            "Away_Staff_RA9": away_ra9, "Home_Staff_RA9": home_ra9,
            "Away_Base_Offense": aoff["BaseFactor"], "Home_Base_Offense": hoff["BaseFactor"],
            "Away_Platoon_Factor": aoff["PlatoonFactor"], "Home_Platoon_Factor": hoff["PlatoonFactor"],
            "Away_Lineup_Factor": aoff["LineupFactor"], "Home_Lineup_Factor": hoff["LineupFactor"],
            "Away_Lineup_Used": aoff["LineupUsed"], "Home_Lineup_Used": hoff["LineupUsed"],
            "Away_Offense": aoff["Factor"], "Home_Offense": hoff["Factor"],
            "Away_Proj_Runs": away_runs, "Home_Proj_Runs": home_runs,
            "Away_WinProb": away_prob, "Home_WinProb": home_prob,
            "Away_FairML": fair_ml(away_prob), "Home_FairML": fair_ml(home_prob),
            "Model_Confidence": conf, "Confidence_Grade": conf_grade,
            "Confidence_Reasons": conf_reasons,
            "Lineup_Status": ("CONFIRMED" if aoff["LineupUsed"] and hoff["LineupUsed"]
                              else "PARTIAL/UNCONFIRMED"),
            "Model_Version": MODEL_VERSION,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ backtest --

def _final_scores_for_date(d):
    data = get_json(f"{MLB_API}/v1/schedule",
                    {"sportId": 1, "date": str(d), "hydrate": "linescore"},
                    cache_key=("final", str(d)))
    out = {}
    for block in data.get("dates", []):
        for g in block.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            t = g.get("teams", {})
            a = safe_float(t.get("away", {}).get("score"), np.nan)
            h = safe_float(t.get("home", {}).get("score"), np.nan)
            if math.isfinite(a) and math.isfinite(h):
                out[g.get("gamePk")] = a + h
    return out


def backtest(days_back=45, verbose=True):
    """Dispersion check against completed games.

    WHAT THIS DOES AND DOES NOT TELL YOU
    ------------------------------------
    This runs the current model over recent completed games and regresses the
    actual total on the projected total. It uses CURRENT season-to-date stats,
    so it contains look-ahead bias. It is NOT a profitability test and it will
    flatter the model.

    Use it for exactly one thing: checking whether the model has enough spread.

        slope near 1.0  -> dispersion is roughly right
        slope above 1.3 -> still too flat, raise OFFENSE_EXPONENT
        slope below 0.8 -> overshooting, lower OFFENSE_EXPONENT

    Also reports the mean residual. If that is, say, -0.4, set
    TOTAL_CALIBRATION_OFFSET = -0.4 and rerun.

    A real profitability test requires point-in-time stats and closing lines.
    This is not that.
    """
    end = today_et() - timedelta(days=1)
    proj, actual = [], []

    for i in range(days_back):
        d = end - timedelta(days=i)
        try:
            games = fetch_games_for_date(d)
            finals = _final_scores_for_date(d)
        except Exception:
            continue
        if not games or not finals:
            continue
        games = [g for g in games if g.get("GamePk") in finals
                 and g.get("Away_SP_ID") and g.get("Home_SP_ID")]
        if not games:
            continue
        df = run_model(games)
        if df.empty:
            continue
        for _, r in df.iterrows():
            t = totals_projection(r.to_dict())
            proj.append(t["Projected_Total"])
            actual.append(finals[r["GamePk"]])
        if verbose:
            print(f"  {d}: {len(games)} games", flush=True)

    if len(proj) < 30:
        print(f"\nOnly {len(proj)} games collected. Need 30+ for a usable read.")
        return None

    p, a = np.array(proj), np.array(actual)
    slope, intercept = np.polyfit(p, a, 1)
    resid = a - p

    result = {
        "n": len(p), "slope": float(slope), "intercept": float(intercept),
        "mean_residual": float(resid.mean()), "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "proj_sd": float(p.std()), "actual_sd": float(a.std()),
    }
    if not verbose:
        return result

    print("\n" + "=" * 58)
    print(f"BACKTEST  n={len(p)}  model={MODEL_VERSION}")
    print("=" * 58)
    print(f"  projected total: mean {p.mean():6.2f}   sd {p.std():5.2f}")
    print(f"  actual total:    mean {a.mean():6.2f}   sd {a.std():5.2f}")
    print(f"  mean residual (actual - projected): {resid.mean():+6.2f}")
    print(f"  MAE  {np.abs(resid).mean():5.2f}    RMSE {np.sqrt((resid**2).mean()):5.2f}")
    print(f"  regression: actual = {slope:.2f} * projected + {intercept:.2f}")
    print("-" * 58)
    if slope > 1.3:
        print(f"  -> STILL TOO FLAT. Raise OFFENSE_EXPONENT (currently "
              f"{OFFENSE_EXPONENT}) toward {OFFENSE_EXPONENT * slope:.2f} and rerun.")
    elif slope < 0.8:
        print(f"  -> OVERSHOOTING. Lower OFFENSE_EXPONENT (currently "
              f"{OFFENSE_EXPONENT}) toward {OFFENSE_EXPONENT * slope:.2f} and rerun.")
    else:
        print("  -> Dispersion looks reasonable.")
    if abs(resid.mean()) > 0.25:
        print(f"  -> Set TOTAL_CALIBRATION_OFFSET = {resid.mean():+.2f} and rerun.")
    print("=" * 58)
    print("Reminder: look-ahead bias is present. This is a dispersion check,")
    print("not evidence the model beats a closing line.\n")
    return result


def selftest():
    """Offline sanity check of the factor ranges. No network required."""
    print(f"\n{MODEL_VERSION} — dispersion self-test (no API calls)\n")

    print("OFFENSE (team OPS -> runs scored)")
    for ops in (0.640, 0.680, 0.720, 0.760, 0.800):
        f = clamp((ops / LEAGUE_OPS) ** OFFENSE_EXPONENT, *OFFENSE_CLAMP)
        print(f"  OPS {ops:.3f} -> factor {f:.3f} -> {LEAGUE_RUNS_PER_TEAM * f:5.2f} R/G")

    print("\nPITCHING (starter FIP -> runs allowed, league-average bullpen)")
    bp = LEAGUE_BULLPEN_ERA * RA9_MULTIPLIER
    for fip, ip in ((2.80, 6.2), (3.50, 5.9), (4.20, 5.4), (5.00, 5.0), (5.60, 4.6)):
        pf, ra9 = _pitching_factor(fip * RA9_MULTIPLIER, bp, ip)
        print(f"  FIP {fip:.2f} ({ip:.1f} IP) -> staff RA9 {ra9:5.2f} -> factor {pf:.3f}"
              f" -> {LEAGUE_RUNS_PER_TEAM * pf:5.2f} R/G allowed")

    lo_off = clamp((0.640 / LEAGUE_OPS) ** OFFENSE_EXPONENT, *OFFENSE_CLAMP)
    hi_off = clamp((0.800 / LEAGUE_OPS) ** OFFENSE_EXPONENT, *OFFENSE_CLAMP)
    lo_pf = _pitching_factor(2.80 * RA9_MULTIPLIER, 3.10 * RA9_MULTIPLIER, 6.4)[0]
    hi_pf = _pitching_factor(5.60 * RA9_MULTIPLIER, 5.20 * RA9_MULTIPLIER, 4.4)[0]

    lo = LEAGUE_RUNS_PER_TEAM * lo_off * lo_pf * 2 + HOME_RUN_ADVANTAGE
    neu = LEAGUE_RUNS_PER_TEAM * 2 + HOME_RUN_ADVANTAGE
    hi = LEAGUE_RUNS_PER_TEAM * hi_off * hi_pf * 2 + HOME_RUN_ADVANTAGE

    print("\nTOTAL RANGE (before park/weather)")
    print(f"  low {lo:5.2f}   neutral {neu:5.2f}   high {hi:5.2f}   spread {hi - lo:.2f} runs")
    print(f"  at Coors (x1.10): {hi * 1.10:5.2f}")
    print(f"  at Oracle (x0.96): {lo * 0.96:5.2f}")
    print("  [old model spread was 1.70 runs, 8.22 to 9.92]")

    print("\nWIN PROBABILITY (exact convolution vs old Pythagorean)")
    def _pyth(a, h):
        return a ** PYTH_EXPONENT / (a ** PYTH_EXPONENT + h ** PYTH_EXPONENT)
    print("  %-16s %10s %10s" % ("away v home", "old ML", "new ML"))
    for a, h in ((4.30, 4.30), (4.6, 4.2), (5.0, 4.0), (5.4, 3.6), (6.0, 3.0)):
        old = fair_ml(_pyth(a, h + 0.12))
        new = fair_ml(win_prob(a, h + HOME_RUN_ADVANTAGE))
        print("  %-16s %10d %10d" % (f"{a} v {h}", old, new))
    _even = 1 - win_prob(LEAGUE_RUNS_PER_TEAM,
                         LEAGUE_RUNS_PER_TEAM + HOME_RUN_ADVANTAGE)
    print(f"  even teams -> home wins {_even:.3f}  (MLB actual ~0.540)")

    print("\nMONEYLINE (most lopsided matchup the model can build)")
    a = LEAGUE_RUNS_PER_TEAM * hi_off * hi_pf
    h = LEAGUE_RUNS_PER_TEAM * lo_off * lo_pf + HOME_RUN_ADVANTAGE
    wp = win_prob(a, h)
    print(f"  away {a:.2f} vs home {h:.2f} -> win prob {wp:.3f} -> fair ML {fair_ml(wp):+d}")
    print("  [old model could not exceed -134]\n")

# =============================================================================
# POINT-IN-TIME BACKTEST
# =============================================================================
# The quick backtest() above re-runs the model with TODAY's statistics against
# games already played. For a game in June that means feeding the model July,
# August and September data that did not exist yet, plus a "recent form" window
# anchored to today rather than to the game. The further back you go the more
# incoherent the inputs, which is why its slope decayed from 1.24 (30d) to 0.89
# (90d) and its residual bounced +0.44 / -0.04 / +0.77 across consecutive
# stretches of the same season.
#
# The functions below rebuild each game's inputs as they stood the morning of
# that game, and optionally compare the result against the closing total.
#
# COST: this is much slower than backtest(). Team hitting splits must be
# fetched per (team, date), so a 14-day window is roughly 500-900 API calls.
# Start small.

PIT_SEASON_START = "{}-03-01"


def _season_start(as_of):
    return PIT_SEASON_START.format(as_of.year)


def _pit_team_offense(team_name, as_of, use_recent=True):
    """Offense factor using only games played before `as_of`."""
    prior_day = as_of - timedelta(days=1)
    season = _hitting_rates(
        _team_hitting_stats(team_name, "byDateRange", _season_start(as_of), prior_day)
    )
    if not season or season["PA"] < 200:
        return {"Factor": 1.0, "OPS": LEAGUE_OPS, "PA": season["PA"] if season else 0,
                "Available": False}

    ops = season["OPS"]
    if use_recent:
        recent = _hitting_rates(
            _team_hitting_stats(team_name, "byDateRange",
                                as_of - timedelta(days=15), prior_day)
        )
        if recent and recent["PA"] >= 120:
            rw = clamp(recent["PA"] / (recent["PA"] + OFFENSE_RECENT_PA_ANCHOR),
                       0.0, OFFENSE_RECENT_MAX_WEIGHT)
            ops = (1 - rw) * ops + rw * recent["OPS"]

    w = clamp(season["PA"] / 1500.0, 0.0, 1.0)
    ops = w * ops + (1 - w) * LEAGUE_OPS
    return {"Factor": clamp((ops / LEAGUE_OPS) ** OFFENSE_EXPONENT, *OFFENSE_CLAMP),
            "OPS": ops, "PA": season["PA"], "Available": True}


def _pit_starter(player_id, as_of):
    """Starter RA9 and expected IP from game-log rows dated before `as_of`.

    Uses the cached full-season game log and filters by date, so this costs no
    extra API calls beyond the one log fetch per pitcher.
    """
    league_ra9 = LEAGUE_FIP * RA9_MULTIPLIER
    if not player_id:
        return {"RA9": league_ra9, "ExpIP": 5.0, "Starts": 0, "Available": False}

    log = _pitcher_game_log(player_id)
    if log.empty:
        return {"RA9": league_ra9, "ExpIP": 5.0, "Starts": 0, "Available": False}

    cutoff = str(as_of)
    prior = log[log["Date"].astype(str) < cutoff]
    if prior.empty:
        return {"RA9": league_ra9, "ExpIP": 5.0, "Starts": 0, "Available": False}

    ip = float(prior["IP"].sum())
    if ip <= 0:
        return {"RA9": league_ra9, "ExpIP": 5.0, "Starts": 0, "Available": False}

    er = float(prior["ER"].sum())
    hr = float(prior["HR"].sum())
    bb = float(prior["BB"].sum())
    k = float(prior["K"].sum())
    starts = int(prior["Started"].sum())

    era = 9.0 * er / ip
    fip = _fip_from_counts(ip, hr, bb, k)
    if math.isfinite(fip) and math.isfinite(era):
        skill = 0.70 * fip + 0.30 * era
    elif math.isfinite(fip):
        skill = fip
    elif math.isfinite(era):
        skill = era
    else:
        skill = LEAGUE_FIP

    w_season = clamp(ip / SP_SEASON_IP_ANCHOR, 0.0, 1.0)
    skill = w_season * skill + (1 - w_season) * LEAGUE_FIP

    # recent form = last 5 starts BEFORE this game
    rs = prior[prior["Started"] == 1].head(5)
    if not rs.empty:
        r_ip = float(rs["IP"].sum())
        if r_ip > 0:
            r_fip = _fip_from_counts(r_ip, float(rs["HR"].sum()),
                                     float(rs["BB"].sum()), float(rs["K"].sum()))
            if math.isfinite(r_fip):
                w_r = clamp(r_ip / SP_RECENT_IP_ANCHOR, 0.0, 1.0) * SP_RECENT_MAX_WEIGHT
                skill = (1 - w_r) * skill + w_r * r_fip

    role = clamp(starts / 6.0, 0.25, 1.0)
    skill = role * skill + (1 - role) * LEAGUE_FIP

    if not rs.empty:
        vals = rs["IP"].to_numpy(dtype=float)
        exp_ip = float(np.average(vals, weights=np.arange(len(vals), 0, -1, dtype=float)))
    else:
        exp_ip = 5.0
    return {"RA9": clamp(skill * RA9_MULTIPLIER, 2.20, 8.00),
            "ExpIP": clamp(exp_ip, 3.8, 6.8), "Starts": starts, "Available": True}


def _pit_bullpen(team_name, as_of):
    """Relief RA9 before `as_of`. Falls back to league average if the split is
    not available for a date range (the MLB API is inconsistent here)."""
    if not TEAM_IDS:
        load_team_ids()
    tid = TEAM_IDS.get(team_name)
    fallback = {"RA9": LEAGUE_BULLPEN_ERA * RA9_MULTIPLIER, "Available": False}
    if not tid:
        return fallback
    data = get_json(
        f"{MLB_API}/v1/teams/{tid}/stats",
        {"stats": "byDateRange", "group": "pitching", "season": as_of.year,
         "sitCodes": "rp", "startDate": _season_start(as_of),
         "endDate": str(as_of - timedelta(days=1))},
        cache_key=("pitbp", tid, str(as_of)),
    )
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
        ip = ip_to_decimal(stat.get("inningsPitched", 0))
        er = safe_float(stat.get("earnedRuns"), np.nan)
    except Exception:
        return fallback
    if ip <= 0 or not math.isfinite(er):
        return fallback
    era = 9.0 * er / ip
    w = clamp(ip / BULLPEN_IP_ANCHOR, 0.0, 1.0)
    shrunk = w * era + (1 - w) * LEAGUE_BULLPEN_ERA
    return {"RA9": clamp(shrunk * RA9_MULTIPLIER, 2.80, 7.00), "Available": True}


def _pit_project(g, as_of, use_recent=True):
    """Rebuild one game's projected total as it would have stood that morning.

    Lineups and platoon splits are deliberately excluded: lineups are not
    posted at projection time, and the MLB API does not expose point-in-time
    platoon splits. Weather is excluded too -- historical forecasts are not
    retrievable, and using actual observed weather would be look-ahead.
    Park IS applied.
    """
    asp = _pit_starter(g.get("Away_SP_ID"), as_of)
    hsp = _pit_starter(g.get("Home_SP_ID"), as_of)
    abp = _pit_bullpen(g.get("Away"), as_of)
    hbp = _pit_bullpen(g.get("Home"), as_of)
    aoff = _pit_team_offense(g.get("Away"), as_of, use_recent)
    hoff = _pit_team_offense(g.get("Home"), as_of, use_recent)

    home_pf, _ = _pitching_factor(hsp["RA9"], hbp["RA9"], hsp["ExpIP"])
    away_pf, _ = _pitching_factor(asp["RA9"], abp["RA9"], asp["ExpIP"])

    away_runs = LEAGUE_RUNS_PER_TEAM * aoff["Factor"] * home_pf
    home_runs = LEAGUE_RUNS_PER_TEAM * hoff["Factor"] * away_pf + HOME_RUN_ADVANTAGE

    park = PARKS.get(g.get("Venue", ""), {})
    pf = 1.0 + (safe_float(park.get("factor"), 1.0) - 1.0) * PARK_WEIGHT
    total = clamp((away_runs + home_runs) * pf + TOTAL_CALIBRATION_OFFSET, *TOTAL_CLAMP)

    return {
        "projected": total,
        "inputs_ok": all([asp["Available"], hsp["Available"],
                          aoff["Available"], hoff["Available"]]),
        "bullpen_ok": abp["Available"] and hbp["Available"],
        "away_sp_ra9": asp["RA9"], "home_sp_ra9": hsp["RA9"],
        "away_off": aoff["Factor"], "home_off": hoff["Factor"],
        "park_factor": pf,
    }


# --- closing lines ---------------------------------------------------------

def _norm_team(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _closing_totals_for_date(api_key, d):
    """Consensus closing total per game from The Odds API historical endpoint.

    Historical odds are a separate paid add-on. If the call fails or is not
    included in your plan this returns {} and the benchmark is skipped rather
    than silently reporting nothing.
    """
    if not api_key:
        return {}, "no ODDS_API_KEY configured"
    snap = f"{d}T23:00:00Z"
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/historical/sports/{ODDS_SPORT_KEY}/odds",
            params={"apiKey": api_key, "regions": "us", "markets": "totals",
                    "oddsFormat": "american", "date": snap},
            timeout=25,
        )
        if r.status_code in (401, 403):
            return {}, "historical odds not included in this API plan"
        if r.status_code == 422:
            return {}, "historical endpoint rejected the date"
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        return {}, f"historical odds request failed: {e}"

    events = payload.get("data", payload) or []
    out = {}
    for ev in events:
        pts = []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "totals":
                    continue
                for oc in mk.get("outcomes", []):
                    p = safe_float(oc.get("point"), np.nan)
                    if math.isfinite(p):
                        pts.append(p)
        if pts:
            key = (_norm_team(ev.get("away_team")), _norm_team(ev.get("home_team")))
            out[key] = float(statistics.median(pts))
    return out, None


def pit_backtest(days_back=14, use_recent=True, use_lines=True,
                 api_key=None, progress=None):
    """Honest backtest: point-in-time inputs, optional closing-line benchmark.

    Returns a dict, or None if too few games were assembled.
    """
    end = today_et() - timedelta(days=1)
    proj, actual, lines = [], [], []
    skipped = 0
    bullpen_ok = 0
    line_note = None

    for i in range(days_back):
        d = end - timedelta(days=i)
        if progress:
            progress(i + 1, days_back, str(d))
        try:
            games = fetch_games_for_date(d)
            finals = _final_scores_for_date(d)
        except Exception:
            continue
        if not games or not finals:
            continue

        day_lines = {}
        if use_lines:
            day_lines, note = _closing_totals_for_date(api_key, d)
            if note and not line_note:
                line_note = note

        for g in games:
            gp = g.get("GamePk")
            if gp not in finals or not g.get("Away_SP_ID") or not g.get("Home_SP_ID"):
                continue
            try:
                res = _pit_project(g, d, use_recent)
            except Exception:
                skipped += 1
                continue
            if not res["inputs_ok"]:
                skipped += 1
                continue
            proj.append(res["projected"])
            actual.append(finals[gp])
            bullpen_ok += 1 if res["bullpen_ok"] else 0
            key = (_norm_team(g.get("Away")), _norm_team(g.get("Home")))
            lines.append(day_lines.get(key))

    if len(proj) < 30:
        return {"error": f"only {len(proj)} usable games -- widen the window",
                "n": len(proj), "skipped": skipped}

    p, a = np.array(proj), np.array(actual)
    slope, intercept = np.polyfit(p, a, 1)
    resid = a - p

    out = {
        "n": int(len(p)), "skipped": int(skipped),
        "bullpen_coverage": round(bullpen_ok / len(p), 3),
        "slope": float(slope), "intercept": float(intercept),
        "mean_residual": float(resid.mean()),
        "model_mae": float(np.abs(resid).mean()),
        "model_rmse": float(np.sqrt((resid ** 2).mean())),
        "proj_sd": float(p.std()), "actual_sd": float(a.std()),
        "line_note": line_note,
    }

    # --- benchmark against the closing line -------------------------------
    idx = [j for j, v in enumerate(lines) if v is not None]
    out["n_with_line"] = len(idx)
    if len(idx) >= 30:
        pl = p[idx]
        al = a[idx]
        ln = np.array([lines[j] for j in idx], dtype=float)
        out["line_mae"] = float(np.abs(al - ln).mean())
        out["model_mae_matched"] = float(np.abs(al - pl).mean())
        out["mae_gap"] = out["model_mae_matched"] - out["line_mae"]
        out["line_mean"] = float(ln.mean())
        out["model_mean_matched"] = float(pl.mean())
        out["corr_model_line"] = float(np.corrcoef(pl, ln)[0, 1])
        # does disagreement with the line predict the actual deviation?
        edge = pl - ln
        dev = al - ln
        out["edge_slope"] = float(np.polyfit(edge, dev, 1)[0])
        out["edge_corr"] = float(np.corrcoef(edge, dev)[0, 1])
        out["beats_line"] = bool(out["mae_gap"] < 0)
    return out


# =============================================================================
# PART 2 -- ENGINE NAMESPACE
# =============================================================================
# The app code below was written against `import model as engine`. Rather than
# edit ~4500 lines, expose the same names through a namespace object. This is
# snapshotted HERE, before the app redefines fetch_games_for_date -- otherwise
# that wrapper would call itself forever.

engine = types.SimpleNamespace(
    MODEL_VERSION=MODEL_VERSION,
    today_et=today_et,
    now_et=now_et,
    season_now=season_now,
    fetch_games_for_date=fetch_games_for_date,
    fetch_today_games=fetch_today_games,
    run_model=run_model,
    totals_projection=totals_projection,
    implied_prob=implied_prob,
    expected_value=expected_value,
    fair_ml=fair_ml,
    win_prob=win_prob,
    reset_dynamic_caches=reset_dynamic_caches,
    PARKS=PARKS,
)


# =============================================================================
# FIRST FIVE INNINGS (F5)
# =============================================================================
# Why F5 rather than the full game.
#
# The full-game model measured out at edge correlation 0.02 against closing
# lines over 187 point-in-time games -- i.e. its disagreements with the market
# carried no information. Two structural reasons to expect F5 to be different:
#
#   1. NO BULLPEN. Roughly 40% of a full game is relief innings, and relief
#      usage is the least knowable input in the whole model -- you do not know
#      who is available, who is warming, or how the manager will sequence them.
#      F5 is mostly the starter, which is the input we model best.
#   2. THINNER MARKET. Fewer books post F5, limits are lower, and far less
#      sharp money shapes the number. Full-game MLB totals are among the most
#      efficient markets in sport; F5 is not.
#
# This is a measurement build, not a betting board. It projects F5 totals and
# scores them against F5 closing lines so we can read edge correlation for this
# market specifically. If it comes back near 0.02 like the full game, F5 is
# dead too and we stop. Nothing here feeds the board yet, by design.

LEAGUE_F5_RUNS_PER_TEAM = 2.30     # league average runs per team, innings 1-5
F5_RUN_DIST_K = 2.10               # lower mean -> refit dispersion
F5_HOME_RUN_ADVANTAGE = 0.06       # smaller than full game; no walk-off effect
F5_TOTAL_CLAMP = (2.5, 10.0)
F5_CALIBRATION_OFFSET = 0.0


def _f5_pitching_factor(sp_ra9, bp_ra9, sp_ip):
    """Runs-allowed multiplier over innings 1-5 only.

    The starter covers min(expected_IP, 5) of those innings; anything short is
    covered by relief. A starter going 6+ means the bullpen never appears in
    the F5 window at all, which is the point of this market.
    """
    sp_innings = clamp(min(float(sp_ip), 5.0), 0.0, 5.0)
    bp_innings = 5.0 - sp_innings
    combined = (sp_innings * sp_ra9 + bp_innings * bp_ra9) / 5.0
    league = LEAGUE_F5_RUNS_PER_TEAM * (9.0 / 5.0) * RA9_MULTIPLIER
    return clamp(combined / league, *PITCHING_CLAMP), combined


def f5_projection(row):
    """Projected first-five total. Park applies; weather is left out because
    its effect is concentrated in ball carry over a full game and the F5
    signal is not worth the extra failure mode."""
    a_sp = safe_float(row.get("Away_SP_RA9"), LEAGUE_FIP * RA9_MULTIPLIER)
    h_sp = safe_float(row.get("Home_SP_RA9"), LEAGUE_FIP * RA9_MULTIPLIER)
    a_bp = safe_float(row.get("Away_Bullpen_RA9"), LEAGUE_BULLPEN_ERA * RA9_MULTIPLIER)
    h_bp = safe_float(row.get("Home_Bullpen_RA9"), LEAGUE_BULLPEN_ERA * RA9_MULTIPLIER)
    a_ip = safe_float(row.get("Away_SP_ExpIP"), 5.0)
    h_ip = safe_float(row.get("Home_SP_ExpIP"), 5.0)
    a_off = safe_float(row.get("Away_Offense"), 1.0)
    h_off = safe_float(row.get("Home_Offense"), 1.0)

    home_pf, _ = _f5_pitching_factor(h_sp, h_bp, h_ip)
    away_pf, _ = _f5_pitching_factor(a_sp, a_bp, a_ip)

    away = LEAGUE_F5_RUNS_PER_TEAM * a_off * home_pf
    home = LEAGUE_F5_RUNS_PER_TEAM * h_off * away_pf + F5_HOME_RUN_ADVANTAGE

    park = PARKS.get(row.get("Venue", ""), {})
    pf = 1.0 + (safe_float(park.get("factor"), 1.0) - 1.0) * PARK_WEIGHT
    total = clamp((away + home) * pf + F5_CALIBRATION_OFFSET, *F5_TOTAL_CLAMP)
    return {"F5_Away_Runs": away, "F5_Home_Runs": home,
            "F5_Projected_Total": total, "F5_Park_Factor": pf,
            "F5_SP_Covers": min(a_ip, 5.0) + min(h_ip, 5.0)}


def _pit_f5_project(g, as_of, use_recent=True):
    """Point-in-time F5 projection, same cutoff discipline as _pit_project."""
    asp = _pit_starter(g.get("Away_SP_ID"), as_of)
    hsp = _pit_starter(g.get("Home_SP_ID"), as_of)
    abp = _pit_bullpen(g.get("Away"), as_of)
    hbp = _pit_bullpen(g.get("Home"), as_of)
    aoff = _pit_team_offense(g.get("Away"), as_of, use_recent)
    hoff = _pit_team_offense(g.get("Home"), as_of, use_recent)
    row = {
        "Away_SP_RA9": asp["RA9"], "Home_SP_RA9": hsp["RA9"],
        "Away_Bullpen_RA9": abp["RA9"], "Home_Bullpen_RA9": hbp["RA9"],
        "Away_SP_ExpIP": asp["ExpIP"], "Home_SP_ExpIP": hsp["ExpIP"],
        "Away_Offense": aoff["Factor"], "Home_Offense": hoff["Factor"],
        "Venue": g.get("Venue", ""),
    }
    out = f5_projection(row)
    out["inputs_ok"] = all([asp["Available"], hsp["Available"],
                            aoff["Available"], hoff["Available"]])
    return out


def _f5_final_scores_for_date(d):
    """Actual runs through 5 innings, from the linescore innings array."""
    data = get_json(f"{MLB_API}/v1/schedule",
                    {"sportId": 1, "date": str(d), "hydrate": "linescore"},
                    cache_key=("f5final", str(d)))
    out = {}
    for block in data.get("dates", []):
        for g in block.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            innings = (g.get("linescore", {}) or {}).get("innings", []) or []
            if len(innings) < 5:
                continue          # shortened game: no valid F5
            tot = 0.0
            ok = True
            for inn in innings[:5]:
                a = safe_float((inn.get("away", {}) or {}).get("runs"), np.nan)
                h = safe_float((inn.get("home", {}) or {}).get("runs"), np.nan)
                # bottom 5 can be legitimately unplayed if home leads
                if not math.isfinite(a):
                    ok = False
                    break
                tot += a + (h if math.isfinite(h) else 0.0)
            if ok:
                out[g.get("gamePk")] = tot
    return out


F5_MARKET_KEYS = ("totals_1st_5_innings", "totals_h1", "totals_1st_half")


def _f5_closing_totals_for_date(api_key, d):
    """Consensus F5 closing total per game. Tries the known market keys in
    order, since The Odds API naming has varied."""
    if not api_key:
        return {}, "no ODDS_API_KEY configured"
    snap = f"{d}T23:00:00Z"
    last_note = None
    for mkey in F5_MARKET_KEYS:
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{ODDS_SPORT_KEY}/odds",
                params={"apiKey": api_key, "regions": "us", "markets": mkey,
                        "oddsFormat": "american", "date": snap},
                timeout=25,
            )
            if r.status_code in (401, 403):
                return {}, "historical odds not included in this API plan"
            if r.status_code in (404, 422):
                last_note = f"market key '{mkey}' not served"
                continue
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            last_note = f"F5 odds request failed: {e}"
            continue

        events = payload.get("data", payload) or []
        out = {}
        for ev in events:
            pts = []
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk.get("key") != mkey:
                        continue
                    for oc in mk.get("outcomes", []):
                        p = safe_float(oc.get("point"), np.nan)
                        if math.isfinite(p):
                            pts.append(p)
            if pts:
                key = (_norm_team(ev.get("away_team")), _norm_team(ev.get("home_team")))
                out[key] = float(statistics.median(pts))
        if out:
            return out, None
        last_note = f"market key '{mkey}' returned no F5 lines"
    return {}, last_note or "no F5 market available"


def f5_backtest(days_back=14, use_recent=True, api_key=None, progress=None):
    """Point-in-time F5 backtest with the closing-line benchmark.

    Same discipline as pit_backtest. The number that matters is edge_corr:
    the full-game model scored 0.02 on it, which is why we are here.
    """
    end = today_et() - timedelta(days=1)
    proj, actual, lines = [], [], []
    skipped = 0
    line_note = None

    for i in range(days_back):
        d = end - timedelta(days=i)
        if progress:
            progress(i + 1, days_back, str(d))
        try:
            games = fetch_games_for_date(d)
            finals = _f5_final_scores_for_date(d)
        except Exception:
            continue
        if not games or not finals:
            continue

        day_lines, note = _f5_closing_totals_for_date(api_key, d)
        if note and not line_note:
            line_note = note

        for g in games:
            gp = g.get("GamePk")
            if gp not in finals or not g.get("Away_SP_ID") or not g.get("Home_SP_ID"):
                continue
            try:
                res = _pit_f5_project(g, d, use_recent)
            except Exception:
                skipped += 1
                continue
            if not res["inputs_ok"]:
                skipped += 1
                continue
            proj.append(res["F5_Projected_Total"])
            actual.append(finals[gp])
            key = (_norm_team(g.get("Away")), _norm_team(g.get("Home")))
            lines.append(day_lines.get(key))

    if len(proj) < 30:
        return {"error": f"only {len(proj)} usable F5 games -- widen the window",
                "n": len(proj), "skipped": skipped, "line_note": line_note}

    p, a = np.array(proj), np.array(actual)
    slope, intercept = np.polyfit(p, a, 1)
    resid = a - p
    out = {
        "n": int(len(p)), "skipped": int(skipped),
        "slope": float(slope), "intercept": float(intercept),
        "mean_residual": float(resid.mean()),
        "model_mae": float(np.abs(resid).mean()),
        "proj_sd": float(p.std()), "actual_sd": float(a.std()),
        "actual_mean": float(a.mean()), "proj_mean": float(p.mean()),
        "line_note": line_note,
    }

    idx = [j for j, v in enumerate(lines) if v is not None]
    out["n_with_line"] = len(idx)
    if len(idx) >= 30:
        pl, al = p[idx], a[idx]
        ln = np.array([lines[j] for j in idx], dtype=float)
        out["line_mae"] = float(np.abs(al - ln).mean())
        out["model_mae_matched"] = float(np.abs(al - pl).mean())
        out["mae_gap"] = out["model_mae_matched"] - out["line_mae"]
        out["line_mean"] = float(ln.mean())
        edge, dev = pl - ln, al - ln
        out["edge_slope"] = float(np.polyfit(edge, dev, 1)[0])
        out["edge_corr"] = float(np.corrcoef(edge, dev)[0, 1])
        # what that edge slope is worth at -110, given the residual spread
        from math import erf, sqrt
        b = out["edge_slope"]
        sd = float(np.std(dev)) or 1.0
        z = b * 1.0 / sd
        winp = 0.5 * (1 + erf(z / sqrt(2)))
        out["implied_win_rate"] = float(winp)
        out["implied_ev"] = float(winp * (100 / 110) - (1 - winp))
    return out


# =============================================================================
# PART 2.5 -- DIAGNOSTICS (defined before the app so the
#              router can call it; see NOTE ON PLACEMENT below)
# =============================================================================
# One button, one file. Everything needed to review the model is bundled into a
# single JSON export: config constants, per-game model inputs and outputs,
# market comparison, calibration results and the tracker.

def _json_safe(o):
    """NaN/NumPy/Timestamp -> something json.dumps can handle."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if (o != o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and o != o:
        return None
    return str(o)


def _clean_record(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and v != v:
            out[k] = None
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            out[k] = _json_safe(v)
        else:
            out[k] = v
    return out


def diagnostics_bundle():
    """Assemble the full review package as one dict."""
    bundle = {}

    bundle["meta"] = {
        "generated_at_et": _now_et_iso(),
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "python": sys.version.split()[0],
        "packages": {
            "streamlit": getattr(st, "__version__", "?"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "requests": requests.__version__,
        },
    }

    # Every tunable constant, so the numbers below can be reproduced exactly.
    bundle["config"] = {
        "LEAGUE_RUNS_PER_TEAM": LEAGUE_RUNS_PER_TEAM,
        "LEAGUE_OPS": LEAGUE_OPS,
        "LEAGUE_FIP": LEAGUE_FIP,
        "LEAGUE_BULLPEN_ERA": LEAGUE_BULLPEN_ERA,
        "OFFENSE_EXPONENT": OFFENSE_EXPONENT,
        "OFFENSE_CLAMP": list(OFFENSE_CLAMP),
        "PITCHING_CLAMP": list(PITCHING_CLAMP),
        "RA9_MULTIPLIER": RA9_MULTIPLIER,
        "HOME_RUN_ADVANTAGE": HOME_RUN_ADVANTAGE,
        "PYTH_EXPONENT": PYTH_EXPONENT,
        "PARK_WEIGHT": PARK_WEIGHT,
        "TEMP_RUNS_PER_DEGREE": TEMP_RUNS_PER_DEGREE,
        "TOTAL_CALIBRATION_OFFSET": TOTAL_CALIBRATION_OFFSET,
        "TOTAL_CLAMP": list(TOTAL_CLAMP),
        "TOTALS_MODEL_WEIGHT": TOTALS_MODEL_WEIGHT,
        "TOTALS_RESIDUAL_SD": TOTALS_RESIDUAL_SD,
        "TOTALS_GRADE_THRESHOLDS": {"BEST BET": 0.125, "BET": 0.075, "LEAN": 0.05},
    }

    # Offline factor ranges -- proves dispersion without needing the API.
    _bp = LEAGUE_BULLPEN_ERA * RA9_MULTIPLIER
    bundle["factor_ranges"] = {
        "offense": {
            f"{o:.3f}": round(LEAGUE_RUNS_PER_TEAM
                              * clamp((o / LEAGUE_OPS) ** OFFENSE_EXPONENT,
                                      *OFFENSE_CLAMP), 3)
            for o in (0.640, 0.680, 0.720, 0.760, 0.800)
        },
        "pitching": {
            f"FIP{f:.2f}": round(LEAGUE_RUNS_PER_TEAM
                                 * _pitching_factor(f * RA9_MULTIPLIER, _bp, ip)[0], 3)
            for f, ip in ((2.80, 6.2), (3.50, 5.9), (4.20, 5.4),
                          (5.00, 5.0), (5.60, 4.6))
        },
    }

    # Calibration result, if the panel has been run this session.
    bundle["calibration"] = st.session_state.get("cal_result")

    # Per-game: model inputs, projection, park/weather, market, pick.
    rows = []
    mdf = globals().get("model_df")
    gms = globals().get("games") or []
    payload = globals().get("totals_payload") or {}
    if mdf is not None and hasattr(mdf, "empty") and not mdf.empty:
        for _, r in mdf.iterrows():
            d = r.to_dict()
            try:
                ctx = totals_projection(d)
            except Exception as e:
                ctx = {"error": str(e)}
            rec = _clean_record(d)
            rec.update(_clean_record(ctx))
            gobj = next((g for g in gms
                         if g.get("GamePk") == d.get("GamePk")), None)
            try:
                ev = match_event(payload.get("events", []), gobj) if gobj else None
                tm = totals_market(ev) if ev else None
            except Exception:
                tm = None
            if tm:
                rec["Market_Total"] = tm.get("total")
                rec["Market_Over_Odds"] = tm.get("over_best")
                rec["Market_Under_Odds"] = tm.get("under_best")
                rec["Market_Books"] = tm.get("books")
                try:
                    tp = build_total_pick(float(ctx["Projected_Total"]), tm)
                except Exception:
                    tp = None
                if tp:
                    rec["Pick_Side"] = tp["side"]
                    rec["Pick_Prob"] = tp["prob"]
                    rec["Pick_Edge"] = tp["edge"]
                    rec["Pick_EV"] = tp["ev"]
                    rec["Pick_Grade"] = tp["grade"]
                    rec["Calibrated_Total"] = tp["calibrated_total"]
            else:
                rec["Market_Total"] = None
            rows.append(rec)
    bundle["slate"] = rows
    bundle["slate_note"] = (
        "empty means no slate was loaded in this session -- open the Board "
        "page first, then export"
    )

    # Full tracker history.
    try:
        tdf = load_tracker()
        bundle["tracker"] = [_clean_record(x) for x in tdf.to_dict("records")]
    except Exception as e:
        bundle["tracker"] = {"error": str(e)}

    return bundle



# NOTE ON PLACEMENT
# This block used to sit at the very bottom of the file, which meant it only
# executed if the script reached the end. The page router above calls
# st.stop() on the Live / Tracker / Bets / More views, and also when no
# upcoming games remain -- which is what happens once the slate has started.
# Clicking a button triggers a full script rerun, so if that rerun hit any
# st.stop() first, the button's handler never ran and nothing happened.
# Diagnostics is now a function, invoked from the "More" route, so it renders
# and reruns reliably regardless of slate state.


def render_diagnostics():
    st.markdown('<div class="kicker">Diagnostics</div>', unsafe_allow_html=True)

    with st.expander("Moneyline price band", expanded=False):
        st.caption(
            "Moneylines outside this band are graded PASS no matter how much the "
            "model likes them. They still show on the full board -- they just stop "
            "being recommendations. This is a staking preference, not a model "
            "judgement."
        )
        _mb1, _mb2 = st.columns(2)
        _mb1.slider("Shortest favourite you would bet", -400, -100,
                    int(ML_PRICE_FLOOR), step=5, key="ml_floor")
        _mb2.slider("Longest dog you would bet", 100, 500,
                    int(ML_PRICE_CEILING), step=10, key="ml_ceiling")
        st.caption(
            f"Currently recommending only prices between "
            f"{int(st.session_state.get('ml_floor', ML_PRICE_FLOOR))} and "
            f"+{int(st.session_state.get('ml_ceiling', ML_PRICE_CEILING))}. "
            "Reload the board after changing these."
        )


    with st.expander("Model calibration", expanded=False):
        st.caption(
            "Regresses actual totals on projected totals over recent completed "
            "games. Uses current season-to-date stats, so it carries look-ahead "
            "bias -- a dispersion check, NOT a profitability test."
        )
        _cal_days = st.slider("Days of completed games", 14, 90, 45, key="cal_days")
        if st.button("Run calibration check", key="cal_run"):
            with st.spinner("Pulling completed games and re-running the model..."):
                _res = backtest(days_back=_cal_days, verbose=False)
            st.session_state["cal_result"] = _res
        _res = st.session_state.get("cal_result")
        if _res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Games", _res["n"])
            c2.metric("Slope", f"{_res['slope']:.2f}", help="Target is 1.0")
            c3.metric("Mean residual", f"{_res['mean_residual']:+.2f}")
            c4, c5 = st.columns(2)
            c4.metric("Projected SD", f"{_res['proj_sd']:.2f}")
            c5.metric("MAE", f"{_res['mae']:.2f}")
            if _res["slope"] > 1.3:
                st.error(
                    f"Still too flat. Raise OFFENSE_EXPONENT (currently "
                    f"{OFFENSE_EXPONENT}) toward "
                    f"{OFFENSE_EXPONENT * _res['slope']:.2f} and rerun."
                )
            elif _res["slope"] < 0.8:
                st.error(
                    f"Overshooting. Lower OFFENSE_EXPONENT (currently "
                    f"{OFFENSE_EXPONENT}) toward "
                    f"{OFFENSE_EXPONENT * _res['slope']:.2f} and rerun."
                )
            else:
                st.success("Dispersion looks reasonable.")
            if abs(_res["mean_residual"]) > 0.25:
                st.warning(
                    f"Set TOTAL_CALIBRATION_OFFSET = "
                    f"{_res['mean_residual']:+.2f} and rerun."
                )
            st.caption("Both constants are in the CALIBRATION BLOCK at the top of this file.")

    with st.expander("Export diagnostics bundle", expanded=False):
        st.caption(
            "One file containing everything needed to review the model: config "
            "constants, factor ranges, every game on the loaded slate with its "
            "model inputs / projection / park / weather / market / pick, the "
            "calibration result, and the full tracker."
        )
        st.caption(
            "Load the Board page first so a slate is in memory, and run the "
            "calibration check above if you want it included."
        )
        if st.button("Build diagnostics bundle", key="diag_build"):
            with st.spinner("Assembling..."):
                try:
                    _b = diagnostics_bundle()
                    st.session_state["diag_bundle"] = json.dumps(
                        _b, indent=2, default=_json_safe
                    )
                    st.session_state["diag_counts"] = (
                        len(_b.get("slate") or []),
                        len(_b.get("tracker") or []),
                        bool(_b.get("calibration")),
                    )
                except Exception as e:
                    st.session_state["diag_bundle"] = None
                    st.error(f"Could not build bundle: {e}")
        if st.session_state.get("diag_bundle"):
            _g, _t, _c = st.session_state.get("diag_counts", (0, 0, False))
            st.success(
                f"{_g} games on slate, {_t} tracker rows, "
                f"calibration {'included' if _c else 'not run'}"
            )
            st.download_button(
                "Download diagnostics bundle",
                data=st.session_state["diag_bundle"].encode("utf-8"),
                file_name=f"ninth_signal_diagnostics_{today_et()}.json",
                mime="application/json",
                use_container_width=True,
                key="diag_download",
            )


    with st.expander("First-five-innings edge test (F5)", expanded=False):
        st.caption(
            "The full-game model measured edge correlation 0.02 against closing "
            "lines -- its disagreements with the market carried no information. "
            "F5 removes the bullpen (the least knowable input) and trades a very "
            "efficient market for a thinner one. This tests whether that helps."
        )
        st.caption(
            "This is a measurement only. Nothing here feeds the board until the "
            "numbers justify it."
        )
        _f1, _f2 = st.columns(2)
        _f5_days = _f1.slider("Days", 7, 45, 21, key="f5_days")
        _f5_recent = _f2.checkbox("Recent form", value=True, key="f5_recent")
        if st.button("Run F5 edge test", key="f5_run"):
            _fb = st.progress(0.0, text="starting...")

            def _fprog(i, n, label):
                _fb.progress(i / n, text=f"{label}  ({i}/{n} days)")

            try:
                st.session_state["f5_result"] = f5_backtest(
                    days_back=_f5_days, use_recent=_f5_recent,
                    api_key=st.secrets.get("ODDS_API_KEY", ""), progress=_fprog)
            except Exception as e:
                st.session_state["f5_result"] = {"error": str(e)}
            _fb.empty()

        _fr = st.session_state.get("f5_result")
        if _fr and _fr.get("error"):
            st.error(_fr["error"])
            if _fr.get("line_note"):
                st.warning(f"Lines: {_fr['line_note']}")
        elif _fr:
            g1, g2, g3 = st.columns(3)
            g1.metric("Games", _fr["n"])
            g2.metric("Slope", f"{_fr['slope']:.2f}")
            g3.metric("Mean residual", f"{_fr['mean_residual']:+.2f}")
            h1, h2 = st.columns(2)
            h1.metric("Projected F5", f"{_fr['proj_mean']:.2f}")
            h2.metric("Actual F5", f"{_fr['actual_mean']:.2f}")

            st.markdown("**Versus the F5 closing line**")
            if _fr.get("line_note"):
                st.warning(f"Lines unavailable: {_fr['line_note']}")
            elif _fr.get("n_with_line", 0) < 30:
                st.warning(
                    f"Only {_fr.get('n_with_line', 0)} games matched an F5 line.")
            else:
                k1, k2, k3 = st.columns(3)
                k1.metric("Model MAE", f"{_fr['model_mae_matched']:.2f}")
                k2.metric("Line MAE", f"{_fr['line_mae']:.2f}")
                k3.metric("Gap", f"{_fr['mae_gap']:+.2f}", delta_color="inverse")
                j1, j2, j3 = st.columns(3)
                j1.metric("Edge corr", f"{_fr['edge_corr']:.2f}",
                          help="Full game scored 0.02")
                j2.metric("Edge slope", f"{_fr['edge_slope']:.2f}")
                j3.metric("Implied win rate",
                          f"{_fr['implied_win_rate']*100:.1f}%",
                          help="Break-even at -110 is 52.4%")
                st.caption(f"Matched on {_fr['n_with_line']} games.")

                if _fr["implied_ev"] > 0.01:
                    st.success(
                        f"Implied EV {_fr['implied_ev']*100:+.1f}% per bet at "
                        f"-110. This is the first positive signal in the "
                        f"project. Confirm on a second window before betting."
                    )
                elif _fr["edge_corr"] < 0.10:
                    st.error(
                        f"Edge correlation {_fr['edge_corr']:.2f} -- no better "
                        f"than the full game. F5 is not the answer either."
                    )
                else:
                    st.warning(
                        f"Edge correlation {_fr['edge_corr']:.2f}: some signal, "
                        f"but implied EV {_fr['implied_ev']*100:+.1f}% is not "
                        f"enough to beat vig. Break-even needs edge slope ~0.26."
                    )

    with st.expander("Point-in-time backtest (advanced)", expanded=False):
        st.caption(
            "Rebuilds each game's inputs as they stood that morning -- season "
            "stats cut off the day before, recent form anchored to the game -- "
            "then optionally compares the projection against the closing total. "
            "This is the honest version. The quick check above uses today's "
            "stats and cannot measure level bias."
        )
        st.caption(
            "Excluded by design: lineups (not posted at projection time), platoon "
            "splits (no point-in-time source) and weather (historical forecasts "
            "are not retrievable). Park IS applied."
        )
        st.warning(
            "Slow. Team splits are fetched per team per date, so roughly 500-900 "
            "API calls for 14 days. Start small and do not close the tab."
        )
        _pd1, _pd2 = st.columns(2)
        _pit_days = _pd1.slider("Days", 7, 45, 14, key="pit_days")
        _pit_recent = _pd2.checkbox("Include 14-day recent form", value=True,
                                    key="pit_recent",
                                    help="Doubles the number of API calls")
        _pit_lines = st.checkbox(
            "Benchmark against closing lines", value=True, key="pit_lines",
            help="Requires the historical odds add-on on your Odds API plan")
        if st.button("Run point-in-time backtest", key="pit_run"):
            _bar = st.progress(0.0, text="starting...")

            def _prog(i, n, label):
                _bar.progress(i / n, text=f"{label}  ({i}/{n} days)")

            try:
                st.session_state["pit_result"] = pit_backtest(
                    days_back=_pit_days, use_recent=_pit_recent,
                    use_lines=_pit_lines,
                    api_key=st.secrets.get("ODDS_API_KEY", ""),
                    progress=_prog)
            except Exception as e:
                st.session_state["pit_result"] = {"error": str(e)}
            _bar.empty()

        _pr = st.session_state.get("pit_result")
        if _pr and _pr.get("error"):
            st.error(_pr["error"])
        elif _pr:
            st.markdown("**Projection quality**")
            a1, a2, a3 = st.columns(3)
            a1.metric("Games", _pr["n"])
            a2.metric("Slope", f"{_pr['slope']:.2f}")
            a3.metric("Mean residual", f"{_pr['mean_residual']:+.2f}")
            b1, b2, b3 = st.columns(3)
            b1.metric("Model MAE", f"{_pr['model_mae']:.2f}")
            b2.metric("Projected SD", f"{_pr['proj_sd']:.2f}")
            b3.metric("Bullpen coverage", f"{_pr['bullpen_coverage']*100:.0f}%")
            if _pr.get("skipped"):
                st.caption(f"{_pr['skipped']} games skipped for incomplete inputs.")

            if abs(_pr["mean_residual"]) > 0.20:
                st.info(
                    f"Point-in-time level bias is {_pr['mean_residual']:+.2f}. "
                    "Unlike the quick check this number is trustworthy -- if it "
                    "holds across two different windows, put it in "
                    "TOTAL_CALIBRATION_OFFSET.")
            else:
                st.success("No meaningful level bias. Leave the offset at 0.0.")

            st.markdown("**Versus the closing line**")
            if _pr.get("line_note"):
                st.warning(f"Lines unavailable: {_pr['line_note']}")
            elif _pr.get("n_with_line", 0) < 30:
                st.warning(
                    f"Only {_pr.get('n_with_line', 0)} games matched a closing "
                    "line -- not enough to compare.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Model MAE", f"{_pr['model_mae_matched']:.2f}")
                c2.metric("Line MAE", f"{_pr['line_mae']:.2f}")
                c3.metric("Gap", f"{_pr['mae_gap']:+.2f}", delta_color="inverse")
                d1, d2 = st.columns(2)
                d1.metric("Edge slope", f"{_pr['edge_slope']:.2f}",
                          help="1.0 means your disagreement with the line is fully "
                               "predictive. 0.0 means it is noise.")
                d2.metric("Edge corr", f"{_pr['edge_corr']:.2f}")
                st.caption(f"Matched on {_pr['n_with_line']} games.")

                if _pr["mae_gap"] < -0.05:
                    st.success(
                        f"Model beats the closing line by {abs(_pr['mae_gap']):.2f} "
                        "runs of MAE. That is the first real evidence of an edge. "
                        "Confirm on a second window before staking.")
                elif _pr["mae_gap"] > 0.05:
                    st.error(
                        f"Line beats the model by {_pr['mae_gap']:.2f} runs of MAE. "
                        "No edge. Betting into this loses to vig regardless of how "
                        "good the calibration looks.")
                else:
                    st.warning("Model and line are within noise of each other. "
                               "No demonstrated edge.")
                if _pr["edge_corr"] < 0.15:
                    st.error(
                        f"Edge correlation {_pr['edge_corr']:.2f}: your "
                        "disagreements with the line carry almost no information "
                        "about the outcome. This is the same failure the old model "
                        "had at 0.05.")


# =============================================================================
# PART 3 -- STREAMLIT APP
# =============================================================================

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

APP_VERSION = "3.2.8-TRACKER-MIDNIGHT-CARRY"
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


/* ===== v3.2.4 official pregame tracker ===== */
.pregame-track-card{
    margin:10px 0 14px;
    padding:16px;
    border-radius:20px;
    background:
        radial-gradient(circle at 94% 8%, rgba(56,189,248,.10), transparent 30%),
        linear-gradient(135deg, rgba(18,48,75,.98), rgba(5,25,42,.98));
    border:1px solid rgba(78,139,181,.50);
    box-shadow:0 12px 28px rgba(0,0,0,.18);
}
.pregame-track-top{
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap:12px;
    padding-bottom:12px;
    border-bottom:1px solid rgba(91,137,170,.28);
}
.pregame-track-time{
    color:#8ca6bc;
    font-size:.57rem;
    font-weight:900;
    letter-spacing:.08em;
}
.pregame-track-game{
    margin-top:5px;
    color:#fff;
    font-size:1.06rem;
    font-weight:950;
    line-height:1.22;
}
.pregame-track-grade{
    flex:0 0 auto;
    padding:7px 10px;
    border-radius:999px;
    background:rgba(14,165,233,.13);
    border:1px solid rgba(56,189,248,.52);
    color:#8bddff;
    font-size:.57rem;
    font-weight:950;
    letter-spacing:.06em;
}
.pregame-track-line{
    display:flex;
    align-items:center;
    gap:12px;
    margin-top:13px;
}
.pregame-track-line span{
    color:#71d2f5;
    font-size:.58rem;
    font-weight:950;
    letter-spacing:.10em;
}
.pregame-track-line b{
    color:#fff;
    font-size:1rem;
}
.pregame-track-meta{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:7px;
    margin-top:12px;
}
.pregame-track-meta span{
    display:flex;
    flex-direction:column;
    gap:2px;
    padding:8px 9px;
    border-radius:12px;
    background:rgba(2,15,28,.42);
    color:#718ca3;
    font-size:.49rem;
    font-weight:850;
    letter-spacing:.06em;
}
.pregame-track-meta b{
    color:#d9e8f4;
    font-size:.66rem;
    letter-spacing:0;
}


/* ===== v3.2.8 midnight tracker carry ===== */
.midnight-carry-note{
    margin:8px 0 12px;
    padding:9px 11px;
    border-radius:12px;
    background:rgba(56,189,248,.07);
    border:1px solid rgba(56,189,248,.18);
    color:#9bc7df;
    font-size:.61rem;
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

    Start-time guard:
    MLB can initialize a game as Live / Top 1st / 0 outs several minutes before
    first pitch. Before the scheduled GameDate, Ninth Signal always treats the
    game as PREGAME. This prevents false live cards and fake pre-first-pitch
    win-probability displays.
    """
    if not game:
        return "OTHER"

    abstract = str(game.get("AbstractGameState") or "").strip().lower()
    detailed = str(game.get("DetailedState") or "").strip().lower()
    code = str(game.get("StatusCode") or "").strip().upper()

    # A true terminal state is always final.
    if abstract == "final" or any(x in detailed for x in ("final", "completed", "game over")):
        return "FINAL"

    explicit_delay = any(
        x in detailed for x in (
            "delayed", "delay", "postponed", "suspended", "rain delay",
            "weather delay", "delayed start"
        )
    )

    # Scheduled first pitch is the hard pregame boundary.
    try:
        start = pd.to_datetime(game.get("GameDate"), utc=True)
        now = pd.Timestamp.now(tz="UTC")

        if now < start:
            return "PREGAME"

        # Once the scheduled time has arrived, explicit delay/postponement
        # remains pregame/non-actionable until MLB actually resumes/starts it.
        if explicit_delay:
            return "PREGAME"

        # At/after scheduled time, trust an explicit MLB live state.
        if abstract == "live" or any(
            x in detailed for x in ("in progress", "manager challenge", "review", "warmup")
        ):
            return "LIVE"

        # Preserve the existing safety rule: after scheduled first pitch, a
        # stale Preview/Scheduled flag cannot keep betting recommendations live.
        return "LIVE"
    except Exception:
        pass

    # If time parsing failed, fall back to MLB state.
    if explicit_delay:
        return "PREGAME"
    if abstract == "live" or any(x in detailed for x in ("in progress", "manager challenge", "review", "warmup")):
        return "LIVE"
    if abstract == "preview" or code in {"S", "P"} or any(x in detailed for x in ("scheduled", "pre-game", "pregame")):
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
    # Routed through the price-neutral grader. The legacy body below is kept
    # for reference but no longer runs.
    v, edge, ev, imp, redge = ml_grade_v2(prob, odds, confidence, lineup_confirmed)
    return v, edge, ev, imp

def _grade_legacy(prob, odds, confidence, lineup_confirmed):
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



# --- price-neutral moneyline metrics ---------------------------------------
# A flat probability-edge gate structurally excludes favorites: the same model
# conviction yields ~3.5x less probability edge at -350 than at +250, because
# probability space compresses at the extremes. Measured, 0.40 runs of
# conviction gives 4.2% edge at +250 but only 1.2% at -350. Grading on run
# conviction and EV instead makes the test price-neutral.

def _wp_to_runs(p, base=None):
    """Run differential that reproduces win probability p. Bisection."""
    base = LEAGUE_RUNS_PER_TEAM if base is None else base
    p = clamp(float(p), 0.005, 0.995)
    lo, hi = -6.0, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if win_prob(base + mid, base + HOME_RUN_ADVANTAGE) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def run_edge(model_prob, market_prob):
    """Model conviction in RUNS. Price-neutral: 0.40 means the model likes
    this side by 0.40 runs more than the market does."""
    try:
        return _wp_to_runs(model_prob) - _wp_to_runs(market_prob)
    except Exception:
        return 0.0


# RUN CONVICTION is the gate. It is the only price-neutral measure available:
# 0.40 runs means the same thing at -300 as at +300. EV is kept only as a
# positivity sanity check, NOT as a second gate -- an EV floor reintroduces the
# dog bias, because equal conviction always yields more EV at longer prices
# (0.40 runs = 14.8% EV at +250 but 2.3% at -250).
#
# NOT BACKTESTED. These values are set so a favorite and a dog with equal
# conviction face an equal test. Whether 0.28 runs is the right cutoff at all
# is unknown.
ML_BET_RUNS, ML_BET_EV = 0.28, 0.015
ML_BEST_RUNS, ML_BEST_EV = 0.48, 0.030


def _ml_price_band():
    """Live price band, overridable from the Diagnostics panel."""
    try:
        return (float(st.session_state.get("ml_floor", ML_PRICE_FLOOR)),
                float(st.session_state.get("ml_ceiling", ML_PRICE_CEILING)))
    except Exception:
        return float(ML_PRICE_FLOOR), float(ML_PRICE_CEILING)


def ml_grade_v2(prob, odds, confidence, lineup_confirmed, market_prob=None):
    """Price-neutral moneyline grade. Returns (verdict, edge, ev, imp, redge)."""
    imp = implied_prob(odds)
    edge = prob - imp
    ev = expected_value(prob, odds)
    redge = run_edge(prob, market_prob if market_prob is not None else imp)

    official_conf = 78 if lineup_confirmed else 82
    o = float(odds)

    # Staking preference: outside the band it is not a recommendation.
    floor, ceiling = _ml_price_band()
    if o < 0 and o < floor:
        return "PASS", edge, ev, imp, redge
    if o > 0 and o > ceiling:
        return "PASS", edge, ev, imp, redge

    if o >= 500:
        return "PASS", edge, ev, imp, redge
    # Long dogs: thin sample, model error is amplified by the price.
    if o >= 300:
        cap = "LEAN"
    # Heavy favorites: a model error costs far more than it wins.
    elif o <= -250:
        cap = "BET" if confidence >= 85 else "LEAN"
    else:
        cap = "BEST BET"

    if confidence >= official_conf and redge >= ML_BEST_RUNS and ev >= ML_BEST_EV:
        v = "BEST BET"
    elif (confidence >= max(70, official_conf - 8)
          and redge >= ML_BET_RUNS and ev >= ML_BET_EV):
        v = "BET"
    elif redge >= 0.15 and ev >= 0.015:
        v = "LEAN"
    else:
        v = "PASS"

    order = ["PASS", "LEAN", "BET", "BEST BET"]
    if order.index(v) > order.index(cap):
        v = cap
    return v, edge, ev, imp, redge


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

    # Price-neutral path: the legacy probability-edge buckets below are what
    # excluded favorites in the first place, so trust the grader's verdict for
    # anything that is not a long dog.
    if odds < 200:
        return legacy

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
    """Return (qualified, reason) for freezing an official recommendation.

    Tracker policy: if the Board has an official BET / BEST BET, it belongs in
    forward tracking. There is no second hidden confidence hurdle. The official
    grading logic already incorporates model confidence.
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

    return True, "official-bet eligible"


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
    """Freeze every official ML BET / BEST BET at its first qualifying price."""
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
    """Freeze every official totals BET / BEST BET at its first qualifying price."""
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


def _tracker_date_set(slate_date):
    """Normalize Tracker scope to one or many YYYY-MM-DD slate dates."""
    if isinstance(slate_date, (list, tuple, set)):
        vals = slate_date
    else:
        vals = [slate_date]
    out = set()
    for v in vals:
        try:
            out.add(str(pd.Timestamp(v).date()))
        except Exception:
            out.add(str(v))
    return out


def _active_tracker_dates(tracker_df, selected_date):
    """Carry unfinished prior-day official bets across midnight.

    The Tracker follows the bet/game until it reaches a terminal result instead
    of disappearing just because the calendar date changed.
    """
    dates = {str(pd.Timestamp(selected_date).date())}

    if tracker_df is None or tracker_df.empty:
        return sorted(dates)

    pending = tracker_df[
        ~tracker_df["Result"].astype(str).str.upper().isin(["WIN", "LOSS", "PUSH", "VOID"])
    ].copy()

    # Keep pending bets from the previous two calendar days. MLB games cannot
    # realistically remain active beyond this window, while this also covers
    # long extra-inning / delayed games crossing midnight.
    today = pd.Timestamp.now(tz="America/New_York").date()
    for value in pending.get("Slate_Date", pd.Series(dtype=str)).astype(str):
        try:
            d = pd.Timestamp(value).date()
            age = (today - d).days
            if 0 <= age <= 2:
                dates.add(str(d))
        except Exception:
            continue

    return sorted(dates)


def _slate_tracking_summary(tracker_df, games, fresh_scoreboard, slate_date):
    if tracker_df is None or tracker_df.empty:
        return {
            "tracked":0,"upcoming":0,"live":0,"final":0,"wins":0,"losses":0,"pushes":0,
            "on_track":0,"neutral":0,"needs_help":0,"units":0.0,"status":"NO TRACKED BETS"
        }

    active_dates = _tracker_date_set(slate_date)
    today_rows = tracker_df[tracker_df["Slate_Date"].astype(str).isin(active_dates)].copy()
    if today_rows.empty:
        return {
            "tracked":0,"upcoming":0,"live":0,"final":0,"wins":0,"losses":0,"pushes":0,
            "on_track":0,"neutral":0,"needs_help":0,"units":0.0,"status":"NO TRACKED BETS"
        }

    game_map = {}
    for g0 in games:
        gf = fresh_scoreboard.get(str(g0.get("GamePk")), g0)
        game_map[str(g0.get("GamePk"))] = gf

    out = {
        "tracked":len(today_rows),"upcoming":0,"live":0,"final":0,"wins":0,"losses":0,"pushes":0,
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

        if state == "PREGAME":
            out["upcoming"] += 1
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

def _slate_pulse_html(summary, cross_day=False):
    record = f'{summary["wins"]}-{summary["losses"]}'
    if summary["pushes"]:
        record += f'-{summary["pushes"]}P'
    status_cls = (
        "pulse-good" if summary["status"] == "SLATE POSITIVE"
        else ("pulse-risk" if summary["status"] == "SLATE UNDER PRESSURE" else "pulse-neutral")
    )
    return (
        f'<div class="slate-pulse">'
        f'<div class="pulse-head"><div><div class="pulse-kicker">{"ACTIVE TRACKED SLATE" if cross_day else "TODAY\'S TRACKED SLATE"}</div>'
        f'<div class="pulse-title">{summary["status"]}</div><div class="pulse-sub">Upcoming + live + final official bets</div></div>'
        f'<div class="pulse-status {status_cls}">{summary["tracked"]} TRACKED</div></div>'
        f'<div class="pulse-grid">'
        f'<div><span>UPCOMING</span><b>{summary.get("upcoming",0)}</b></div>'
        f'<div><span>LIVE</span><b>{summary["live"]}</b></div>'
        f'<div><span>FINAL</span><b>{record}</b></div>'
        f'<div><span>ON TRACK</span><b>{summary["on_track"]}</b></div>'
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


def _pregame_tracked_card(rec, game):
    """Compact card for an official tracked bet that has not started yet."""
    market = str(rec.get("Market") or "").upper()
    grade = str(rec.get("Grade") or "BET").upper()
    pick = str(rec.get("Pick") or "").strip()
    odds = _odds_text(rec.get("Odds"))
    if odds and odds not in pick:
        pick = f"{pick} {odds}".strip()

    try:
        start = pd.to_datetime(game.get("GameDate"), utc=True).tz_convert("America/New_York")
        start_text = start.strftime("%-I:%M %p")
    except Exception:
        start_text = "Pregame"

    try:
        edge_text = f'{float(rec.get("Edge")):+.1%}'
    except Exception:
        edge_text = "—"
    try:
        ev_text = f'{float(rec.get("EV")):+.1%}'
    except Exception:
        ev_text = "—"

    market_label = "TOTAL" if market == "TOTAL" else "MONEYLINE"
    away = game.get("Away") or str(rec.get("Game") or "").split(" @ ")[0] or "Away"
    home = game.get("Home") or (
        str(rec.get("Game") or "").split(" @ ")[1]
        if " @ " in str(rec.get("Game") or "")
        else "Home"
    )

    return (
        f'<div class="pregame-track-card">'
        f'<div class="pregame-track-top"><div>'
        f'<div class="pregame-track-time">{start_text} • OFFICIAL TRACKED BET</div>'
        f'<div class="pregame-track-game">{away} @ {home}</div>'
        f'</div><div class="pregame-track-grade">{grade}</div></div>'
        f'<div class="pregame-track-line"><span>{market_label}</span><b>{pick}</b></div>'
        f'<div class="pregame-track-meta">'
        f'<span>EDGE <b>{edge_text}</b></span>'
        f'<span>EV <b>{ev_text}</b></span>'
        f'<span>FROZEN <b>{str(rec.get("Book") or "Market")}</b></span>'
        f'</div>'
        f'</div>'
    )


def _visual_tracked_card(rec, game, win_prob=None):
    market = str(rec.get("Market") or "").upper()
    state = game_state(game)
    state_label = "FINAL" if state == "FINAL" else ("LIVE" if state == "LIVE" else "PREGAME")
    inning = "FINAL" if state == "FINAL" else (inning_status_text(game) if state == "LIVE" else "Awaiting first pitch")
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



def _live_progress_sort_key(item):
    """Sort live tracked bets from furthest progressed game to least progressed.

    Uses inning/half/outs via _inning_fraction(). Multiple official bets from the
    same game naturally stay together because they share the same progress value.
    """
    g, _rec = item
    try:
        progress = float(_inning_fraction(g))
    except Exception:
        progress = 0.0

    # Secondary keys keep ordering deterministic within the same progress point.
    try:
        inning = int(g.get("Current_Inning") or 0)
    except Exception:
        inning = 0

    half = str(g.get("Inning_State") or g.get("Inning_Half") or "").strip().lower()
    half_rank = {
        "top": 0,
        "middle": 1,
        "bottom": 2,
        "end": 3,
    }.get(half, 0)

    try:
        outs = int(g.get("Outs") or 0)
    except Exception:
        outs = 0

    return (progress, inning, half_rank, outs)


def render_live_scoreboard(games, fresh_scoreboard, tracker_df, slate_date):
    """Official bet tracker: upcoming, live and completed tracked recommendations."""
    game_map = {}
    for g0 in games:
        g = fresh_scoreboard.get(str(g0.get("GamePk")), g0)
        game_map[str(g0.get("GamePk"))] = g

    active_dates = _tracker_date_set(slate_date)
    today_rows = (
        tracker_df[tracker_df["Slate_Date"].astype(str).isin(active_dates)].copy()
        if tracker_df is not None and not tracker_df.empty
        else pd.DataFrame(columns=TRACKER_COLUMNS)
    )

    tracked_upcoming = []
    tracked_live = []
    tracked_final = []

    for _, rec in today_rows.iterrows():
        g = game_map.get(str(rec.get("GamePk")), {})
        if not g:
            continue
        state = game_state(g)
        result = str(rec.get("Result", "PENDING") or "PENDING").upper()

        if state == "FINAL" or result in ("WIN", "LOSS", "PUSH", "VOID"):
            tracked_final.append((g, rec))
        elif state == "LIVE":
            tracked_live.append((g, rec))
        elif state == "PREGAME":
            tracked_upcoming.append((g, rec))

    # Upcoming official bets: earliest scheduled first pitch first.
    # Multiple official bets from the same game stay adjacent because they share
    # the same GameDate and GamePk.
    def _start_key(item):
        g, _rec = item
        try:
            start = pd.to_datetime(g.get("GameDate"), utc=True)
        except Exception:
            start = pd.Timestamp.max.tz_localize("UTC")
        try:
            game_pk = int(g.get("GamePk") or 0)
        except Exception:
            game_pk = 0
        return (start, game_pk)

    tracked_upcoming.sort(key=_start_key)

    # Live Tracker priority: games furthest along are shown first.
    # Example: Bottom 8th appears above Top 6th, which appears above Top 2nd.
    tracked_live.sort(key=_live_progress_sort_key, reverse=True)

    active_count = len(tracked_upcoming) + len(tracked_live)

    st.markdown(
        f'<div class="tracker-hero">'
        f'<div><div class="tracker-eyebrow">OFFICIAL MODEL LEDGER</div>'
        f'<div class="tracker-title">Bet Tracker <span class="tracker-count">{active_count}</span></div>'
        f'<div class="tracker-sub">Official bets • live first by game progress • upcoming next by start time</div></div>'
        f'<div class="tracker-live-orb"><span></span>AUTO</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    summary = _slate_tracking_summary(tracker_df, games, fresh_scoreboard, slate_date)
    st.markdown(
        _slate_pulse_html(summary, cross_day=(len(active_dates) > 1)),
        unsafe_allow_html=True,
    )

    # Priority order in Tracker:
    # 1) Live tracked bets
    # 2) Upcoming tracked bets
    # 3) Completed tracked bets
    if tracked_live:
        st.markdown(
            f'<div class="kicker">Live Tracked Bets — {len(tracked_live)}</div>',
            unsafe_allow_html=True,
        )
        for g, rec in tracked_live:
            wp = fetch_live_win_probability(g.get("GamePk"))
            st.markdown(_visual_tracked_card(rec, g, win_prob=wp), unsafe_allow_html=True)

    if tracked_upcoming:
        st.markdown(
            f'<div class="kicker">Upcoming Tracked Bets — {len(tracked_upcoming)}</div>',
            unsafe_allow_html=True,
        )
        for g, rec in tracked_upcoming:
            st.markdown(_pregame_tracked_card(rec, g), unsafe_allow_html=True)

    if not tracked_live and not tracked_upcoming:
        st.info("No official tracked bets are upcoming or live right now.")

    if tracked_final:
        with st.expander(f"Completed Tracked Bets — {len(tracked_final)}", expanded=False):
            for g, rec in tracked_final:
                st.markdown(_visual_tracked_card(rec, g, win_prob=None), unsafe_allow_html=True)

    # Other live games remain secondary and never clutter the official ledger.
    live_games = [g for g in game_map.values() if game_state(g) == "LIVE"]
    untracked_live = [
        g for g in live_games
        if tracked_rows_for_game(g.get("GamePk"), tracker_df).empty
    ]
    untracked_live.sort(
        key=lambda g: _live_progress_sort_key((g, None)),
        reverse=True,
    )
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
    """Refresh Tracker across midnight without using paid odds calls.

    If an official bet from yesterday is still pending/live after midnight,
    Ninth Signal automatically keeps yesterday's schedule + scoreboard attached
    until the bet reaches a final result.
    """
    # grade_tracker is internally throttled to once per minute.
    grade_tracker(force=False)
    tracker_df = load_tracker()
    active_dates = _active_tracker_dates(tracker_df, slate_date)

    combined_games = {}
    combined_fresh = {}

    for d_text in active_dates:
        try:
            d = pd.Timestamp(d_text).date()
        except Exception:
            d = slate_date

        # Reuse already-loaded current-page games where possible.
        if str(d) == str(pd.Timestamp(slate_date).date()):
            day_games = games
        else:
            try:
                day_games = fetch_games_for_date(d)
            except Exception:
                day_games = []

        for g in day_games or []:
            combined_games[str(g.get("GamePk"))] = g

        fresh_day = fetch_fresh_scoreboard(d)
        combined_fresh.update(fresh_day or {})

    render_live_scoreboard(
        list(combined_games.values()),
        combined_fresh,
        tracker_df,
        active_dates,
    )

    updated = pd.Timestamp.now(tz="America/New_York")
    carry = [d for d in active_dates if d != str(pd.Timestamp(slate_date).date())]
    carry_text = f' • CARRYING {", ".join(carry)}' if carry else ""
    st.markdown(
        f'<div class="auto-fresh"><span></span>AUTO TRACKING{carry_text} • UPDATED {updated.strftime("%-I:%M:%S %p")}</div>',
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
    # Decorative 477 KB base64 logo removed -- it was 56% of the entire file
    # and rendered only on this settings page, directly above the text header
    # below. The brand mark in the global header is unaffected.
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
    min_value=today_et()-timedelta(days=2),
    max_value=today_et()+timedelta(days=14),
    help="Board date. Tracker automatically carries unfinished prior-day bets across midnight.",
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

# Only the Board consumes model_df / candidates. Live and Tracker take `games`
# and the live scoreboard; Bets and More take neither. Running the full
# projection before the router meant every page paid for per-pitcher game logs,
# team splits, bullpen and lineup fetches it then threw away.
_view = st.session_state.get("ninth_page", "Board")

# The Board's model run is now explicit. Nothing projects until you ask for it,
# so opening the app costs one schedule call. The flag is keyed to the slate
# date, so changing dates correctly requires a fresh run.
_board_ready = st.session_state.get("board_loaded_for") == str(slate_date)
_needs_model = (_view == "Board") and _board_ready

if _needs_model:
    with st.spinner("Loading MLB schedule, starters, lineups and model data…"):
        games = fetch_games_for_date(slate_date)
        model_df = run_model(games) if games else pd.DataFrame()
        candidates = build_candidates(
            model_df, games, odds_payload.get("events", [])
        ) if not model_df.empty else []
else:
    with st.spinner("Loading schedule…"):
        games = fetch_games_for_date(slate_date)
    model_df = pd.DataFrame()
    candidates = []

fresh_scoreboard = fetch_fresh_scoreboard(slate_date)

# Forward-test tracker: freeze the first official recommendation at the price
# that triggered it. Requires candidates, so it only runs on the Board. Grading
# of already-tracked bets is independent and always runs.
if _needs_model:
    _new_ml = track_current_official_recommendations(candidates, games, slate_date)
    _new_totals = track_current_total_recommendations(
        candidates, games, model_df, totals_payload, slate_date)
else:
    _new_ml = _new_totals = 0
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

if _needs_model and not candidates:
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
        # Diagnostics also lives here: the "More" nav button is the last item in
        # the bar and can sit underneath the preview overlay on mobile, making
        # it untappable. Only one route renders per run, so no key collisions.
        render_diagnostics()
        st.stop()

    if main_view == "More":
        render_account_page()
        render_diagnostics()
        st.stop()

    if not _board_ready:
        st.markdown(
            '<div class="board-head"><span>BETTING BOARD</span>'
            '<b>Model not loaded</b></div>', unsafe_allow_html=True)
        st.caption(
            f"{len(games)} games on the {slate_date.strftime('%b %-d')} slate. "
            "Running the model fetches starter game logs, team splits, bullpen "
            "numbers and lineups for every game, so it is the slow part of the "
            "app. Nothing else needs it."
        )
        if st.button("Run model for this slate", key="board_run",
                     type="primary", use_container_width=True):
            st.session_state["board_loaded_for"] = str(slate_date)
            st.rerun()
        st.caption("Live, Tracker and Bets work without it.")
        st.stop()

    _bh1, _bh2 = st.columns([3, 1])
    _bh1.markdown('<div class="board-head"><span>BETTING BOARD</span>'
                  '<b>Choose a workflow</b></div>', unsafe_allow_html=True)
    if _bh2.button("Refresh", key="board_refresh", use_container_width=True,
                   help="Re-run the model with the latest lineups and stats"):
        try:
            reset_dynamic_caches()
        except Exception:
            pass
        st.session_state["board_loaded_for"] = str(slate_date)
        st.rerun()

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
