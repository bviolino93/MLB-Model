MLB Edge v0.8.0-GENERAL-MARKET

What changed
------------
- Automatic current MLB line feed from The Odds API.
- Pulls h2h (moneyline), spreads (run line), and totals in one US-region request.
- Builds a general/consensus market using medians across available US books.
- Matches odds events to MLB Stats API games, including nearest-time handling for doubleheaders.
- Full Slate is now ranked:
    STRONG BET > BET > LEAN > PASS > NO LINE
    then by edge and EV.
- Each game card highlights the best current market.
- "All Markets" dropdown shows every grade for that game.
- Full-slate CSV now includes the market snapshot, provider count and last update.
- Existing 734 screenshot/manual single-game workflow is preserved.
- Existing juice-aware thresholds are unchanged.
- Projection engine model.py is unchanged.

Required Streamlit secret
-------------------------
ODDS_API_KEY = "YOUR_KEY_HERE"

Do not put the real key directly in app.py.
