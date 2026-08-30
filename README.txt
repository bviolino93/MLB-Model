MLB Edge v0.10.0-BACKTEST-LAB

Adds a third app mode: Backtest Lab.

API CREDIT SAFETY
=================
Backtest Lab makes ZERO Odds API calls.
It operates entirely on a historical CSV uploaded by the user.

Required historical CSV columns
===============================
Date
Game
Market_Type
Bet
Odds
Result
Raw_Model_Prob
Market_NoVig_Prob
Calibrated_Prob
Edge
EV
Verdict
Confidence

Result must be WIN, LOSS, or PUSH.

Included analysis
=================
- Record / hit rate / units / ROI
- Average odds
- Max drawdown
- Season-by-season stability
- Market breakdown
- Verdict breakdown
- Edge buckets
- Odds buckets
- Top 5 daily simulation
- Top 10 daily simulation
- Raw-model probability calibration
- v0.9 calibrated probability calibration
- Brier-score comparison
- Filtered audit CSV export

Methodology
===========
The app does not fabricate historical odds and does not label historical
prices as closing lines unless the supplied dataset actually contains verified
closing lines. Point-in-time quality of the source dataset remains critical.
