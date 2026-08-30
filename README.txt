MLB Model v0.12.3 — Pitcher Causality Audit

Purpose
- Fixes the conceptual flaw in the old swapped-starter placebo: swapping train + validation + holdout lets the model simply relearn the inverse mapping.
- v0.12.3 trains on correct 2023/2024 starter features, then corrupts ONLY the untouched 2025 inference rows.
- Adds inference-only scrambled starters, opponent-starter swap, and lagged-wrong-starter controls.
- Adds feature-family ablations for ERA, K/9, BB/9, HR/9, WHIP, recent form, starter experience, and all pitcher features.
- Uses strict <=6h, no-doubleheader sample.
- Keeps 2023 train -> 2024 validation -> 2025 holdout.
- Uses free MLB Stats API and local cache; zero Odds API historical credits.

Run
1. Deploy app.py/model.py/requirements.txt.
2. Open Backtest Lab -> v0.12.3 Pitcher Causality Audit.
3. Upload mlb_moneyline_master_2023_2025.csv.
4. Leave defaults at 12 prior team games / 3 prior starter starts.
5. Run the audit.
6. Download mlb_pit_pitcher_causality_summary.csv and mlb_pit_pitcher_causality_segments.csv.

Interpretation
- Correct starters should outperform the market.
- Inference-only wrong-starter controls should materially worsen because the trained model cannot relearn the corruption.
- Feature ablations identify which pitcher-stat families contribute incremental OOS information.
