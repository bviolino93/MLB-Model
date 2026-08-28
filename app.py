import re

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

APP_VERSION = "0.6.3-SMARTER-OCR"

st.set_page_config(page_title="MLB Model", page_icon="⚾", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {max-width: 760px; padding-top: 1rem; padding-bottom: 3rem;}
div[data-testid="stMetricValue"] {font-size: 1.65rem;}
.stButton > button {width: 100%; min-height: 3rem; font-weight: 700;}
.bet-card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 14px 16px; margin: 8px 0;}
.bet-big {font-size: 1.15rem; font-weight: 800;}
</style>
""", unsafe_allow_html=True)


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

    # Common OCR variations for baseball run lines.
    t = re.sub(r'([+-])\s*1\s*[.,|:/]\s*5\b', r'\g<1>1.5', t)
    t = re.sub(r'([+-])\s*1\s+5\b', r'\g<1>1.5', t)

    # "+15" / "-15" is a common OCR miss for "+1.5" / "-1.5".
    # It is safe here because American odds are filtered to 3+ digits elsewhere.
    t = re.sub(r'(?<!\d)([+-])15(?!\d)', r'\g<1>1.5', t)

    return t


def valid_total(x):
    try:
        x = float(x)
        # Full-game MLB totals normally live in this range. This intentionally
        # rejects F5 totals such as 4.5.
        return 6.0 <= x <= 14.5
    except Exception:
        return False


def parse_734_lines(text, away, home):
    result = {
        "away_ml": None, "home_ml": None,
        "away_rl_side": "+1.5", "away_rl_odds": None,
        "home_rl_side": "-1.5", "home_rl_odds": None,
        "total_line": None, "over_odds": None, "under_odds": None,
    }

    t = normalize_734_text(text)
    raw_lines = [re.sub(r"\\s+", " ", x).strip() for x in t.splitlines() if x.strip()]

    # Also inspect 2- and 3-line windows because OCR often puts the team/market
    # name on one line and the price on the next line.
    windows = list(raw_lines)
    for i in range(len(raw_lines) - 1):
        windows.append(raw_lines[i] + " " + raw_lines[i + 1])
    for i in range(len(raw_lines) - 2):
        windows.append(raw_lines[i] + " " + raw_lines[i + 1] + " " + raw_lines[i + 2])

    away_keys = [away.lower(), nickname(away).lower()]
    home_keys = [home.lower(), nickname(home).lower()]

    # -------------------------
    # MONEYLINE
    # -------------------------
    team_hits = {"away": [], "home": []}
    for line in windows:
        ll = line.lower()
        odds = american_numbers(line)
        if not odds:
            continue
        if any(k in ll for k in away_keys):
            team_hits["away"].extend(odds)
        if any(k in ll for k in home_keys):
            team_hits["home"].extend(odds)

    if team_hits["away"]:
        result["away_ml"] = team_hits["away"][0]
    if team_hits["home"]:
        result["home_ml"] = team_hits["home"][0]

    # -------------------------
    # RUN LINE / SPREAD
    # -------------------------
    spread_pat = re.compile(r'([+-]1\\.5)\\D{0,80}?([+-]\\s?\\d{3,4})(?!\\d)', re.I)
    spread_candidates = []

    for line in windows:
        ll = line.lower()
        for m in spread_pat.finditer(line):
            side = m.group(1).replace(" ", "")
            odds = int(m.group(2).replace(" ", ""))
            if not (100 <= abs(odds) <= 1000):
                continue

            spread_candidates.append((side, odds, line))

            if any(k in ll for k in away_keys):
                result["away_rl_side"] = side
                result["away_rl_odds"] = odds
            if any(k in ll for k in home_keys):
                result["home_rl_side"] = side
                result["home_rl_odds"] = odds

    # Fallback for table-style 734 screenshots where team names are detached
    # from the spread prices. Prefer one +1.5 and one -1.5 candidate.
    plus = [(s, o, l) for s, o, l in spread_candidates if s == "+1.5"]
    minus = [(s, o, l) for s, o, l in spread_candidates if s == "-1.5"]

    if result["away_rl_odds"] is None and result["home_rl_odds"] is None:
        if plus and minus:
            # Away underdog convention is common, but do not assume blindly.
            # Use ML direction when available to determine favorite.
            if result["away_ml"] is not None and result["home_ml"] is not None:
                away_is_dog = result["away_ml"] > 0 and result["home_ml"] < 0
                home_is_dog = result["home_ml"] > 0 and result["away_ml"] < 0

                if away_is_dog:
                    result["away_rl_side"], result["away_rl_odds"] = plus[0][0], plus[0][1]
                    result["home_rl_side"], result["home_rl_odds"] = minus[0][0], minus[0][1]
                elif home_is_dog:
                    result["away_rl_side"], result["away_rl_odds"] = minus[0][0], minus[0][1]
                    result["home_rl_side"], result["home_rl_odds"] = plus[0][0], plus[0][1]

    # "Spread +1.5 -130 -1.5 +100" style row.
    if result["away_rl_odds"] is None or result["home_rl_odds"] is None:
        for line in windows:
            if "spread" not in line.lower() and "run line" not in line.lower():
                continue
            pairs = [(m.group(1), int(m.group(2).replace(" ", ""))) for m in spread_pat.finditer(line)]
            if len(pairs) >= 2 and result["away_ml"] is not None and result["home_ml"] is not None:
                plus_pair = next((p for p in pairs if p[0] == "+1.5"), None)
                minus_pair = next((p for p in pairs if p[0] == "-1.5"), None)
                if plus_pair and minus_pair:
                    if result["away_ml"] > result["home_ml"]:
                        result["away_rl_side"], result["away_rl_odds"] = plus_pair
                        result["home_rl_side"], result["home_rl_odds"] = minus_pair
                    else:
                        result["away_rl_side"], result["away_rl_odds"] = minus_pair
                        result["home_rl_side"], result["home_rl_odds"] = plus_pair
                    break

    # -------------------------
    # TOTAL
    # -------------------------
    # Accept Over 9 -104, O 9 -104, O9 -104, etc.
    over_pat = re.compile(
        r'\\b(?:over|ovr|o)\\s*[:\\-]?\\s*([0-9]{1,2}(?:\\.5)?)\\D{0,80}?([+-]\\s?\\d{3,4})(?!\\d)',
        re.I,
    )
    under_pat = re.compile(
        r'\\b(?:under|undr|u)\\s*[:\\-]?\\s*([0-9]{1,2}(?:\\.5)?)\\D{0,80}?([+-]\\s?\\d{3,4})(?!\\d)',
        re.I,
    )

    for line in windows:
        mo = over_pat.search(line)
        if mo and valid_total(mo.group(1)):
            result["total_line"] = float(mo.group(1))
            result["over_odds"] = int(mo.group(2).replace(" ", ""))

        mu = under_pat.search(line)
        if mu and valid_total(mu.group(1)):
            if result["total_line"] is None:
                result["total_line"] = float(mu.group(1))
            result["under_odds"] = int(mu.group(2).replace(" ", ""))

    # Table-style fallback:
    # "Total 9 -104 -126" or "9 O -104 U -126".
    if result["over_odds"] is None or result["under_odds"] is None:
        for line in windows:
            ll = line.lower()
            if "total" not in ll and not re.search(r'\\b[ou]\\b', ll):
                continue

            total_nums = re.findall(r'(?<!\\d)([6-9](?:\\.5)?|1[0-4](?:\\.5)?)(?!\\d)', line)
            total_nums = [float(x) for x in total_nums if valid_total(x)]
            odds = american_numbers(line)

            if total_nums and len(odds) >= 2:
                if result["total_line"] is None:
                    result["total_line"] = total_nums[0]

                # If explicit O/U positions exist, use them. Otherwise 734's
                # full-game total display is conventionally Over first, Under second.
                if result["over_odds"] is None:
                    result["over_odds"] = odds[0]
                if result["under_odds"] is None:
                    result["under_odds"] = odds[1]
                break

    # Last-resort nearby-label fallback with a wider OCR window.
    if result["over_odds"] is None:
        vals = nearby_odds(t, "over", window=320)
        if vals:
            result["over_odds"] = vals[0]
    if result["under_odds"] is None:
        vals = nearby_odds(t, "under", window=320)
        if vals:
            result["under_odds"] = vals[0]

    return result



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


def bet_grade(model_prob, odds, confidence):
    imp = implied_prob(odds)
    edge = model_prob - imp
    ev = expected_value(model_prob, odds)
    if confidence >= 80 and edge >= 0.04 and ev >= 0.08:
        verdict = "STRONG BET"
    elif confidence >= 70 and edge >= 0.025 and ev >= 0.05:
        verdict = "BET"
    elif edge > 0 and ev > 0:
        verdict = "LEAN"
    else:
        verdict = "PASS"
    return verdict, edge, ev, imp


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

    if confidence >= 80 and edge >= 0.04 and ev >= 0.08:
        verdict = "STRONG BET"
    elif confidence >= 70 and edge >= 0.025 and ev >= 0.05:
        verdict = "BET"
    elif edge > 0 and ev > 0:
        verdict = "LEAN"
    else:
        verdict = "PASS"

    return verdict, edge, ev, imp, conditional_win, probs

st.title("⚾ MLB Model")
st.caption(f"App {APP_VERSION} • Engine {MODEL_VERSION}")

if "games" not in st.session_state:
    with st.spinner("Loading today's MLB schedule..."):
        st.session_state.games = fetch_today_games()

if st.button("🔄 Refresh Today's Games"):
    with st.spinner("Refreshing schedule..."):
        st.session_state.games = fetch_today_games()
    for k in ["last_results", "parsed_lines", "ocr_raw"]:
        st.session_state.pop(k, None)
    st.rerun()

games = st.session_state.games
if not games:
    st.warning("No MLB games were found for today.")
    st.stop()

labels = {}
for g in games:
    away_sp = g["Away_SP"] or "TBD"
    home_sp = g["Home_SP"] or "TBD"
    time_text = f" — {g['TimeLabel']} ET" if g["TimeLabel"] else ""
    labels[f"{g['Away']} @ {g['Home']}{time_text} | {away_sp} vs {home_sp}"] = g["GamePk"]

mode = st.radio("Run mode", ["Single Game", "Full Slate"], horizontal=True)
selected_game = None
if mode == "Single Game":
    selected_label = st.selectbox("Game", list(labels.keys()))
    selected_pk = labels[selected_label]
    selected_game = next(g for g in games if g["GamePk"] == selected_pk)
    st.info(f"**Probable pitchers:** {selected_game['Away_SP'] or 'TBD'} vs {selected_game['Home_SP'] or 'TBD'}")

if st.button("▶️ Run Model", type="primary"):
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

        st.divider()
        st.subheader(row["Game"])
        c1, c2, c3 = st.columns(3)
        c1.metric(away, f"{row['Away_Proj_Runs']:.2f}")
        c2.metric(home, f"{row['Home_Proj_Runs']:.2f}")
        c3.metric("Model Total", f"{row['Model_Total']:.2f}")
        c1, c2 = st.columns(2)
        c1.metric(f"{away} Win", f"{row['Away_WinProb']*100:.1f}%", f"Fair {int(row['Away_FairML']):+d}")
        c2.metric(f"{home} Win", f"{row['Home_WinProb']*100:.1f}%", f"Fair {int(row['Home_FairML']):+d}")
        st.caption(f"Confidence: {conf}/100 ({row['Confidence_Grade']}) • {row['Data_Status']}")

        st.divider()
        st.subheader("📸 Upload 734 Lines Screenshots")
        uploads = st.file_uploader(
            "Upload one or more screenshots from 734 Games",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )

        if uploads:
            st.caption(f"{len(uploads)} screenshot(s) selected")
            for i, uploaded in enumerate(uploads, start=1):
                uploaded.seek(0)
                image = Image.open(uploaded)
                st.image(image, caption=f"Sportsbook screenshot {i}", use_container_width=True)

            if st.button("🔎 Read Lines From Screenshots"):
                with st.spinner("Reading sportsbook lines from all screenshots..."):
                    try:
                        merged = {
                            "away_ml": None,
                            "home_ml": None,
                            "away_rl_side": "+1.5",
                            "away_rl_odds": None,
                            "home_rl_side": "-1.5",
                            "home_rl_odds": None,
                            "total_line": None,
                            "over_odds": None,
                            "under_odds": None,
                        }
                        all_text = []

                        for uploaded in uploads:
                            uploaded.seek(0)
                            image = Image.open(uploaded)
                            shot_text = ocr_text(image)
                            all_text.append(shot_text)
                            shot_parsed = parse_734_lines(shot_text, away, home)

                            for key, value in shot_parsed.items():
                                if value is not None:
                                    merged[key] = value

                        st.session_state.parsed_lines = merged
                        st.session_state.ocr_raw = "\n\n===== NEXT SCREENSHOT =====\n\n".join(all_text)
                        sync_parsed_to_widgets(merged, row["GamePk"])
                        st.success("Screenshots read. Extracted values were loaded into the boxes below.")
                        st.write(
                            "**Detected:** "
                            f"{away} ML {merged.get('away_ml')} | "
                            f"{home} ML {merged.get('home_ml')} | "
                            f"{away} {merged.get('away_rl_side')} {merged.get('away_rl_odds')} | "
                            f"{home} {merged.get('home_rl_side')} {merged.get('home_rl_odds')} | "
                            f"Total {merged.get('total_line')} | "
                            f"Over {merged.get('over_odds')} | Under {merged.get('under_odds')}"
                        )
                    except Exception as e:
                        st.error(f"Could not read screenshots: {e}")

        if "ocr_raw" in st.session_state:
            with st.expander("OCR text / troubleshooting"):
                st.code(st.session_state.ocr_raw)

        p = st.session_state.get("parsed_lines", {})

        defaults = {
            f"away_ml_{row['GamePk']}": int(p.get("away_ml") if p.get("away_ml") is not None else 100),
            f"home_ml_{row['GamePk']}": int(p.get("home_ml") if p.get("home_ml") is not None else -110),
            f"away_rl_side_{row['GamePk']}": p.get("away_rl_side", "+1.5"),
            f"away_rl_odds_{row['GamePk']}": int(p.get("away_rl_odds") if p.get("away_rl_odds") is not None else -110),
            f"home_rl_side_{row['GamePk']}": p.get("home_rl_side", "-1.5"),
            f"home_rl_odds_{row['GamePk']}": int(p.get("home_rl_odds") if p.get("home_rl_odds") is not None else 100),
            f"total_line_{row['GamePk']}": float(p.get("total_line") if p.get("total_line") is not None else 9.0),
            f"over_odds_{row['GamePk']}": int(p.get("over_odds") if p.get("over_odds") is not None else -110),
            f"under_odds_{row['GamePk']}": int(p.get("under_odds") if p.get("under_odds") is not None else -110),
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        st.subheader("Sportsbook Lines")
        st.caption("Confirm the extracted values. If OCR misses something, just edit the box manually.")

        ml1, ml2 = st.columns(2)
        away_ml = ml1.number_input(f"{away} ML", step=5, key=f"away_ml_{row['GamePk']}")
        home_ml = ml2.number_input(f"{home} ML", step=5, key=f"home_ml_{row['GamePk']}")

        rl1, rl2 = st.columns(2)
        away_side0 = p.get("away_rl_side", "+1.5")
        away_rl_side = rl1.selectbox(f"{away} run line", ["+1.5", "-1.5"], key=f"away_rl_side_{row['GamePk']}")
        away_rl_odds = rl1.number_input(f"{away} RL odds", step=5, key=f"away_rl_odds_{row['GamePk']}")
        home_side0 = p.get("home_rl_side", "-1.5")
        home_rl_side = rl2.selectbox(f"{home} run line", ["-1.5", "+1.5"], key=f"home_rl_side_{row['GamePk']}")
        home_rl_odds = rl2.number_input(f"{home} RL odds", step=5, key=f"home_rl_odds_{row['GamePk']}")

        t1, t2, t3 = st.columns(3)
        total_line = t1.number_input("Total", step=0.5, key=f"total_line_{row['GamePk']}")
        over_odds = t2.number_input("Over odds", step=5, key=f"over_odds_{row['GamePk']}")
        under_odds = t3.number_input("Under odds", step=5, key=f"under_odds_{row['GamePk']}")

        if st.button("✅ Should I Bet?", type="primary"):
            markets = []
            add_market(markets, f"{away} ML", row["Away_WinProb"], away_ml, conf)
            add_market(markets, f"{home} ML", row["Home_WinProb"], home_ml, conf)
            away_rl_prob = row["Away_+1.5_Prob"] if away_rl_side == "+1.5" else row["Away_-1.5_Prob"]
            home_rl_prob = row["Home_+1.5_Prob"] if home_rl_side == "+1.5" else row["Home_-1.5_Prob"]
            add_market(markets, f"{away} {away_rl_side}", away_rl_prob, away_rl_odds, conf)
            add_market(markets, f"{home} {home_rl_side}", home_rl_prob, home_rl_odds, conf)

            market_df = pd.DataFrame(markets)
            rank = {"STRONG BET": 3, "BET": 2, "LEAN": 1, "PASS": 0}
            market_df["_rank"] = market_df["Verdict"].map(rank)
            market_df = market_df.sort_values(["_rank", "EV"], ascending=False).drop(columns="_rank")
            best = market_df.iloc[0]

            if best["Verdict"] in ["STRONG BET", "BET"]:
                st.success(f"{icon(best['Verdict'])} **{best['Verdict']}: {best['Bet']} {int(best['Odds']):+d}**\n\nModel edge: **{best['Edge']*100:+.1f}%** • EV: **{best['EV']*100:+.1f}%**")
            elif best["Verdict"] == "LEAN":
                st.warning(f"🟡 **LEAN ONLY: {best['Bet']} {int(best['Odds']):+d}**\n\nPositive value, but it does **not** clear the bet threshold.")
            else:
                st.info("⚪ **NO BET / PASS** — none of the entered ML or run-line prices clear the model's bet threshold.")

            st.markdown("#### Every market")
            for _, m in market_df.iterrows():
                st.markdown(
                    f"""<div class="bet-card"><div class="bet-big">{icon(m['Verdict'])} {m['Verdict']} — {m['Bet']} {int(m['Odds']):+d}</div>
                    Model {m['Model Prob']*100:.1f}% • Implied {m['Implied Prob']*100:.1f}% • Edge {m['Edge']*100:+.1f}% • EV {m['EV']*100:+.1f}% • Fair {int(m['Model Fair']):+d}</div>""",
                    unsafe_allow_html=True,
                )

            gap = row["Model_Total"] - total_line
            st.markdown("#### Total")

            over_verdict, over_edge, over_ev, over_imp, over_model_prob, over_probs = total_bet_grade(
                row["Model_Total"], total_line, "Over", over_odds, conf
            )
            under_verdict, under_edge, under_ev, under_imp, under_model_prob, under_probs = total_bet_grade(
                row["Model_Total"], total_line, "Under", under_odds, conf
            )

            total_markets = [
                {
                    "Bet": f"Over {total_line:g}",
                    "Verdict": over_verdict,
                    "Odds": int(over_odds),
                    "Model Prob": over_model_prob,
                    "Push Prob": over_probs["push"],
                    "Implied Prob": over_imp,
                    "Edge": over_edge,
                    "EV": over_ev,
                },
                {
                    "Bet": f"Under {total_line:g}",
                    "Verdict": under_verdict,
                    "Odds": int(under_odds),
                    "Model Prob": under_model_prob,
                    "Push Prob": under_probs["push"],
                    "Implied Prob": under_imp,
                    "Edge": under_edge,
                    "EV": under_ev,
                },
            ]

            total_rank = {"STRONG BET": 3, "BET": 2, "LEAN": 1, "PASS": 0}
            total_markets = sorted(
                total_markets,
                key=lambda x: (total_rank[x["Verdict"]], x["EV"]),
                reverse=True,
            )
            best_total = total_markets[0]

            if best_total["Verdict"] in ["STRONG BET", "BET"]:
                st.success(
                    f"{icon(best_total['Verdict'])} **TOTAL {best_total['Verdict']}: "
                    f"{best_total['Bet']} {best_total['Odds']:+d}**\n\n"
                    f"Model edge: **{best_total['Edge']*100:+.1f}%** • EV: **{best_total['EV']*100:+.1f}%**"
                )
            elif best_total["Verdict"] == "LEAN":
                st.warning(
                    f"🟡 **TOTAL LEAN: {best_total['Bet']} {best_total['Odds']:+d}**\n\n"
                    "Positive model value, but it does **not** clear the bet threshold."
                )
            else:
                st.info("⚪ **TOTAL: PASS** — neither side clears the model's bet threshold.")

            st.write(
                f"Model total **{row['Model_Total']:.2f}** vs market **{total_line:.1f}** "
                f"→ projection gap **{gap:+.2f} runs**."
            )

            for m in total_markets:
                push_text = f" • Push {m['Push Prob']*100:.1f}%" if m["Push Prob"] > 0 else ""
                st.markdown(
                    f"""<div class="bet-card"><div class="bet-big">{icon(m['Verdict'])} {m['Verdict']} — {m['Bet']} {m['Odds']:+d}</div>
                    Model {m['Model Prob']*100:.1f}% • Implied {m['Implied Prob']*100:.1f}% • Edge {m['Edge']*100:+.1f}% • EV {m['EV']*100:+.1f}%{push_text}</div>""",
                    unsafe_allow_html=True,
                )

            st.caption(
                "BET = confidence ≥70, edge ≥2.5 percentage points, EV ≥5%. "
                "STRONG BET = confidence ≥80, edge ≥4 points, EV ≥8%. "
                "Positive value below those levels = LEAN. "
                "Total probabilities are EXPERIMENTAL and use a Poisson distribution centered on the model total."
            )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Result CSV", data=csv, file_name=f"mlb_model_{away.replace(' ', '_')}_at_{home.replace(' ', '_')}.csv", mime="text/csv")

    else:
        st.divider()
        st.subheader("Full Slate")
        table = df[["Game", "Away_Proj_Runs", "Home_Proj_Runs", "Model_Total", "Away_WinProb", "Home_WinProb", "Away_FairML", "Home_FairML", "Model_Confidence", "Confidence_Grade"]].copy()
        table["Away_WinProb"] = (table["Away_WinProb"] * 100).round(1)
        table["Home_WinProb"] = (table["Home_WinProb"] * 100).round(1)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download Full Slate CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="mlb_model_full_slate.csv", mime="text/csv")

st.divider()
st.caption("Screenshot OCR is best-effort; always verify parsed sportsbook prices. Run-line and total probabilities are experimental. Model outputs are analytical estimates, not guarantees.")
