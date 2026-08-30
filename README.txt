MLB Edge v0.8.1-AUTO-MARKET

What changed
------------
- Removes the sportsbook screenshot/OCR workflow from the visible Single Game experience.
- Single Game now automatically loads the same generic consensus market used by Full Slate.
- Consensus is the median current line/price across available US sportsbooks from The Odds API.
- Moneyline, run line and total all load automatically.
- The current consensus can still be edited manually inside a collapsed expander.
- Single-game market grading now ranks ML, run line and totals together.
- Full Slate automatic market workflow remains unchanged.
- Existing juice-aware betting thresholds remain unchanged.
- Projection engine model.py remains unchanged.

Required Streamlit secret
-------------------------
ODDS_API_KEY = "YOUR_KEY_HERE"

Do not put the real key directly in app.py.

Important market terminology
----------------------------
The feed is a current generic/consensus market snapshot.
It should not be called an opener or closing line unless separately verified.
