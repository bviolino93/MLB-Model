MLB Model v0.15.0 — Point-in-Time Offense + Platoon Test

Purpose
-------
Freeze v0.13 Pitcher Model 2.0 and test whether point-in-time team offense and handedness matchup add out-of-sample value.

Protocol
--------
- Same Moneyline Master CSV as prior PIT tests.
- 2023 train -> 2024 validation/blend selection -> 2025 untouched holdout.
- Strict <=6 hours before first pitch and no doubleheaders.
- Frozen v0.13 champion is refit on the exact same offense-eligible rows.
- Same betting thresholds. Promotion is based on Brier/log loss, not ROI.

New inputs
----------
- Season-to-date team hitting rates.
- Prior-14-day offense.
- K%, BB%, HR/PA, ISO, run/PA, fixed-weight wOBA-like quality.
- Team hitting versus the opposing starter's throwing hand (L/R), shrunk hard toward overall offense.
- Platoon split is used only after the configured minimum prior PA; otherwise the model falls back to overall offense and records the coverage.

Data / leakage controls
-----------------------
- Free MLB Stats API only; zero Odds API historical credits.
- Hitting game logs are filtered to calendar days strictly before the target game.
- Same-day target-game hitting cannot enter the features.
- Fixed batting weights and shrinkage constants; no target-season final league averages.
- Historical starter identity retains the same retrospective caveat documented in v0.12-v0.14.

Default settings
----------------
- Prior team games: 12
- Prior starter starts: 3
- Minimum prior platoon PA: 80

Primary output
--------------
mlb_pitcher_offense_platoon_comparison.csv

If v0.15 does not improve the frozen v0.13 champion cleanly on calibrated 2025 Brier without validation deterioration, reject the offense/platoon layer.
