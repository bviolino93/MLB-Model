MLB MODEL v0.16.0 — ACTUAL LINEUP RESEARCH TEST

Purpose
-------
Freeze v0.15 Pitcher 2.0 + Offense/Platoon as the research champion and test whether actual historical batting-order strength adds predictive value on the same eligible games.

Protocol
--------
- 2023 train
- 2024 validation chooses model/market blend
- 2023+2024 refit
- 2025 untouched holdout
- <=6 hours to first pitch
- doubleheaders excluded
- same moneyline betting thresholds
- promotion judged on Brier/log loss, not ROI

Lineup layer
------------
- Historical MLB boxscore starting batting orders
- Individual hitter game logs strictly BEFORE target game
- Fixed small-sample shrinkage
- Season-to-date and 14-day hitter quality
- Weighted batting-order quality
- Lineup strength relative to team offensive baseline

IMPORTANT INTEGRITY LIMIT
-------------------------
Historical MLB boxscores identify the ACTUAL lineup that played. They do NOT prove that the exact lineup was publicly known at the historical odds snapshot. Therefore v0.16 is a retrospective upper-bound research test, not point-in-time certification. A positive result requires a later lineup-publication-timing audit before live promotion.

First run can be slow because boxscores and hitter-season logs are cached. Uses zero Odds API credits.
