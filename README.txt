MLB EDGE v1.0.0 — PRODUCTION MONEYLINE

Purpose
-------
Live MLB moneyline-only production app based on the research components that survived holdout testing:
- Starting Pitcher Model 2.0
- Team offense + platoon
- Confirmed actual lineup as a live upgrade
- Market-aware probability calibration

Deliberately excluded
---------------------
- Bullpen model (failed to improve the frozen research champion)
- Run lines
- Totals
- Historical backtest builders/audit UI

Deploy
------
1. Replace app.py and model.py in the Streamlit/GitHub repo.
2. Replace requirements.txt.
3. In Streamlit Secrets configure:

   ODDS_API_KEY = "YOUR_VALID_KEY"

Never paste a live Odds API key into chat or commit it to GitHub.

4. Redeploy/reboot the Streamlit app.
5. Open the app and tap Refresh when you want a fresh slate/lineup/market pull.

Live decision layer
-------------------
- Moneyline consensus probability is derived from the no-vig median market.
- Best available observed book price is used to evaluate the bet.
- Core model weight starts around 60% before both lineups are confirmed.
- With confirmed lineups and sufficient confidence it can rise to 70%, reflecting the research champion while remaining market-aware.
- Expensive favorites and longshots face stricter thresholds.
- +300 to +499 cannot become an official BET/BEST BET; +500 or longer is PASS.

Important limitation
--------------------
The live production engine is research-informed but is not a bit-for-bit replay of the historical PIT logistic model. Track forward results before treating historical ROI as an expectation.
