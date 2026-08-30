MLB Edge v0.12.0-PIT-PITCHER

New
===
Adds a historical point-in-time starting-pitcher validation layer.

Data source
===========
MLB Stats API only.
ZERO Odds API credits are used.

Historical pitcher workflow
===========================
1. Upload mlb_moneyline_master_2023_2025.csv
2. Free MLB schedule data supplies historical starter/probable-pitcher IDs
3. One free game-log request per pitcher-season is cached
4. For each target game, the app uses ONLY prior starts before first pitch
5. Current-game and future pitcher results are excluded

Pitcher features
================
- prior starts
- ERA
- K/9
- BB/9
- HR/9
- approximate WHIP
- last-5-start ERA
- last-5-start K/9
- last-5-start BB/9

These are combined with the existing lagged team features:
- season win %
- season run differential/game
- last-10 win %
- last-10 run differential/game
- rest difference

Walk-forward
============
2023: fit
2024: choose model/market blend by Brier score
2023+2024: refit
2025: untouched holdout

Important
=========
This is still a validation model, not a perfect replay of the live production
engine. It specifically tests whether point-in-time starter information adds
predictive value beyond the historical market benchmark.

First run may take several minutes because pitcher logs are downloaded from
MLB's free API and cached.
