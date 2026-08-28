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

APP_VERSION = "0.6.7-734-FIXED-CELLS"

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


def _crop_text(image, box, psm=7):
    crop = image.crop(box)
    # Enlarging just the target cell dramatically improves Tesseract on 734.
    crop = crop.resize((crop.width * 2, crop.height * 2))
    crop = ImageOps.grayscale(crop)
    crop = ImageOps.autocontrast(crop)
    crop = crop.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(crop, config=f"--psm {psm}").strip()


def _find_f5_header_y(image):
    """
    Find the top of the '1st 5 Innings' header. This lets the screenshot be
    slightly scrolled and still keeps the cell crops aligned.
    """
    img = clean_ocr_image(image)
    data = pytesseract.image_to_data(
        img,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )

    hits = []
    for i, raw in enumerate(data["text"]):
        s = re.sub(r"[^a-z0-9]", "", str(raw).lower())
        if s in {"1st", "ist", "1s", "innings"} or "inning" in s:
            hits.append(int(data["top"][i]))

    if hits:
        # The first innings-related heading below the full-game rows is F5.
        # Ignore any very-high-page noise.
        usable = [y for y in hits if y > image.height * 0.35]
        if usable:
            return min(usable)

    # Fallback tuned to the supplied 734 iPhone screenshots.
    return int(image.height * 0.518)


def _parse_single_american(text):
    vals = american_numbers(text)
    return vals[0] if vals else None


def _parse_parenthesized_american(text):
    s = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
    )
    m = re.search(r"\(\s*([+-]\s*\d{3,4})\s*\)", s)
    if m:
        return int(m.group(1).replace(" ", ""))
    vals = american_numbers(s)
    # Spread cell usually contains a spread token plus one American price.
    return vals[-1] if vals else None


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
    Parse one 734 total cell. The ½ glyph is often OCR'd as %, Z, 7, etc.,
    so use the leading O/U + base run number and treat those known suffixes
    as .5.
    """
    raw = (
        text.replace("−", "-")
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

    # Juice is usually in parentheses and is much easier to read than ½.
    odds = _parse_parenthesized_american(raw)

    # Find the first plausible baseball full-game total number.
    m = re.search(r"[ou0]?\s*(\d{1,2})", low)
    line = None
    if m:
        base = int(m.group(1))
        if 6 <= base <= 14:
            # Explicit .5 or common OCR substitutions for the small ½ glyph.
            half = bool(re.search(r"\.5|%|z|y%|7(?=\s*\()", low))
            line = float(base) + (0.5 if half else 0.0)

    return side, line, odds


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

    away_mid = _crop_text(image, (mid_x1, row1[0], mid_x2, row1[1]), psm=7)
    home_mid = _crop_text(image, (mid_x1, row2[0], mid_x2, row2[1]), psm=7)
    away_tot = _crop_text(image, (tot_x1, row1[0], tot_x2, row1[1]), psm=7)
    home_tot = _crop_text(image, (tot_x1, row2[0], tot_x2, row2[1]), psm=7)

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
            "Upload 734 screenshots (Money Line and/or Spread)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )

        if uploads:
            st.caption(f"{len(uploads)} screenshot(s) selected — order does not matter; each screenshot is read by table-cell position.")
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
                            "away_rl_side": None,
                            "away_rl_odds": None,
                            "home_rl_side": None,
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
                            shot_parsed = parse_734_image(image, away, home, shot_text)

                            for key, value in shot_parsed.items():
                                if key.startswith("_"):
                                    continue
                                if value is not None:
                                    merged[key] = value

                            if shot_parsed.get("_debug"):
                                all_text.append(
                                    "\n===== FIXED CELL OCR =====\n"
                                    + str(shot_parsed["_debug"])
                                )

                        # If we have both RL prices but the ½ glyph was unreadable,
                        # assign standard MLB +/-1.5 sides using the ML favorite.
                        if (
                            merged.get("away_rl_odds") is not None
                            and merged.get("home_rl_odds") is not None
                            and (
                                merged.get("away_rl_side") is None
                                or merged.get("home_rl_side") is None
                            )
                            and merged.get("away_ml") is not None
                            and merged.get("home_ml") is not None
                        ):
                            if merged["away_ml"] < 0 and merged["home_ml"] > 0:
                                merged["away_rl_side"] = "-1.5"
                                merged["home_rl_side"] = "+1.5"
                            elif merged["home_ml"] < 0 and merged["away_ml"] > 0:
                                merged["away_rl_side"] = "+1.5"
                                merged["home_rl_side"] = "-1.5"
                            elif merged["away_ml"] < merged["home_ml"]:
                                merged["away_rl_side"] = "-1.5"
                                merged["home_rl_side"] = "+1.5"
                            else:
                                merged["away_rl_side"] = "+1.5"
                                merged["home_rl_side"] = "-1.5"

                        st.session_state.parsed_lines = merged
                        st.session_state.ocr_raw = "\n\n===== NEXT SCREENSHOT =====\n\n".join(all_text)
                        sync_parsed_to_widgets(merged, row["GamePk"])
                        st.success("Screenshots read. Extracted values were loaded into the boxes below.")
                        st.write(
                            "**Detected:** "
                            f"{away} ML {merged.get('away_ml')} | "
                            f"{home} ML {merged.get('home_ml')} | "
                            f"{away} RL {merged.get('away_rl_side')} {merged.get('away_rl_odds')} | "
                            f"{home} RL {merged.get('home_rl_side')} {merged.get('home_rl_odds')} | "
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
