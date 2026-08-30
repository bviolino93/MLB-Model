MLB Model v0.13.0 — Pitcher Model 2.0

Purpose
- Moves from pitcher-signal validation to pitcher-signal engineering.
- Keeps the v0.12.3 causality benchmark intact and reruns it on the exact same strict sample.
- Adds a second starter model with more baseball-specific point-in-time features.
- Does NOT loosen or optimize betting thresholds on 2025.

Pitcher Model 2.0 features
- Fixed-prior empirical-Bayes shrinkage for ERA, K/9, BB/9, HR/9, WHIP, FIP and K-BB/9.
- FIP-style skill components.
- Exponentially weighted recent form with a fixed 3-start half-life.
- Recent-vs-baseline form deltas.
- Innings per start / starter experience.
- Days rest before the target start.
- 14-day and 30-day prior workload.
- Last-start and last-five pitch counts when MLB game-log data exposes them.
- Existing point-in-time team-form features remain in the model.

Validation protocol
- Strict <=6 hours to first pitch.
- Doubleheaders removed.
- 2023 = train.
- 2024 = select ONLY model-vs-market blend weight.
- 2023+2024 = refit.
- 2025 = untouched holdout.
- v0.12.3 benchmark ridge = 3.0.
- v0.13 Model 2.0 ridge = 5.0 (fixed in advance; not selected on 2025).
- Same bet thresholds as prior research: 2.5% model-vs-market edge, 4.5% EV, no +300 or longer dogs.

Run
1. Deploy app.py, model.py, requirements.txt.
2. Open Backtest Lab -> v0.13.0 Pitcher Model 2.0.
3. Upload mlb_moneyline_master_2023_2025.csv.
4. Leave defaults at 12 prior team games / 3 prior starter starts.
5. Run Pitcher Model 2.0 Test.
6. Send mlb_pitcher_model_2_comparison.csv back for analysis.

Interpretation
- Primary criterion is 2025 calibrated Brier improvement vs the frozen v0.12.3 benchmark.
- ROI is secondary and should not determine promotion.
- If Model 2.0 fails to improve holdout Brier, retain v0.12.3 as the research baseline.
- This remains a research validator. It is not yet wired into the live production slate model.

Data caveat
- Historical starter identities come from MLB's historical schedule/game records. Prior causality tests materially deteriorated when starter assignment was corrupted, but exact historical 'starter known at snapshot' provenance is still not independently archived in this build.
