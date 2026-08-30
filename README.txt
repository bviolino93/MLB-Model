MLB Model v0.14.0 — Point-in-Time Bullpen Test

Purpose
-------
Freeze v0.13 Pitcher Model 2.0 as the research champion and test whether genuinely point-in-time bullpen quality and availability add predictive value.

Protocol
--------
- Same Moneyline Master input: mlb_moneyline_master_2023_2025.csv
- Strict <=6 hours to first pitch
- Doubleheaders removed
- 2023 fit
- 2024 selects model/market blend
- 2023+2024 refit
- 2025 untouched holdout
- Same betting thresholds; ROI never selects the model
- Frozen v0.13 champion rerun on the exact same bullpen-eligible rows

Bullpen features
----------------
- Season-to-date relief FIP/ERA/K-BB9/WHIP with fixed-prior shrinkage
- Prior 7-day and 14-day relief form
- Prior 1-day and 3-day innings/pitches
- Number of relievers used
- Relievers throwing >=20 pitches yesterday
- Relievers used on back-to-back prior days

Data integrity
--------------
Bullpen history is sourced from MLB Stats API pitcher gameLog data. Only appearances with gamesStarted == 0 are treated as relief appearances, and only appearances before the target game date are used. Target doubleheaders are excluded by the strict research sample.

First run
---------
The app requests roughly one free MLB pitching game-log payload per team-season and caches it in .mlb_bullpen_cache. First run can take a few minutes; reruns should be much faster. This uses zero The Odds API credits.

Main output
-----------
mlb_pitcher_bullpen_comparison.csv

Promotion rule
--------------
Promote bullpen only if it improves predictive accuracy (especially calibrated 2025 Brier) versus frozen v0.13 without degrading 2024 validation. Do not promote because ROI looks better.
