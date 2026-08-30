MLB Edge v0.10.4-MONEYLINE-MASTER

Purpose
=======
Convert the paid 2023-2025 historical Moneyline export into a clean master
dataset with actual final MLB results and a market benchmark.

ZERO additional Odds API credits are required for this step.

Workflow
========
Backtest Lab -> Moneyline Master Dataset

1. Upload the Historical Market CSV from the paid Moneyline pull.
2. The app cleans it:
   - valid two-way American Moneyline prices only
   - 0 to 18 hours before first pitch
   - one row per historical Event_ID
3. Click Build Moneyline Master Dataset.
4. Final scores/results are pulled from MLB's free Stats API.
5. Historical market odds are converted to no-vig win probabilities.
6. Games are matched to results by normalized away/home teams, UTC game date,
   and nearest commence time.
7. Download mlb_moneyline_master_2023_2025.csv.

Outputs
=======
- Away/Home historical Moneyline
- Away/Home no-vig market probability
- MLB final score
- winner
- away/home win flag
- favorite and favorite probability
- favorite result
- bookmaker count
- hours before first pitch
- season
- result-match status

Market baseline
===============
The app displays:
- matched game count
- market favorite win rate
- market Brier score
- probability calibration buckets
- season-by-season favorite win rate / implied probability / book depth

Important
=========
This creates the historical MARKET + RESULT master dataset.

It does not yet reconstruct the production model's historical point-in-time
features/probabilities. That is the next validation layer. We should not use
future/full-season information when building historical model predictions.
