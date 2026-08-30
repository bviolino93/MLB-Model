MLB Edge v0.11.0-POINT-IN-TIME

What this release does
======================
Adds a real point-in-time Moneyline validation workflow using the
Moneyline Master dataset.

Important distinction
=====================
This is NOT a claim that the live v0.5.4 production engine can be historically
replayed perfectly from the current master file.

The current live engine depends on historical starter, lineup, bullpen,
platoon, park, weather, rest/travel and team-strength inputs that were not
stored point-in-time in the Moneyline Master CSV.

Instead, v0.11.0 builds a clean PIT validation model using only lagged team
results available before each game:
- season win percentage
- season run differential per game
- last-10 win percentage
- last-10 run differential per game
- rest-day difference

No current-game result enters a feature.

Walk-forward design
===================
2023:
  fit raw PIT logistic model

2024:
  validate raw model and choose market-blend weight by Brier score

2023 + 2024:
  refit raw PIT model

2025:
  untouched holdout evaluation

The 2025 season is not used to select the blend weight.

The app compares
================
- market Brier score
- raw PIT model Brier score
- calibrated PIT + market Brier score
- log loss
- 2025 moneyline bet simulation

Default bet simulation
======================
- edge >= 2.5%
- EV >= 4.5%
- no +300 or longer underdogs
- one bet maximum per game
- 1 unit risk per bet

This is a validation layer, not a profitability guarantee.

Included files
==============
app.py
model.py
requirements.txt
README.txt
mlb_pit_holdout_2025_precomputed.csv
mlb_pit_summary_precomputed.csv
