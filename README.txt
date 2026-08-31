MLB v0.16.2 — Moneyline Price-Bucket Audit

Purpose
- Re-run the frozen v0.16 lineup champion on the same strict 2025 holdout.
- Diagnose favorites vs underdogs and moneyline price buckets.
- Diagnose edge buckets.
- Do NOT tune thresholds from this output.

Workflow
1. Deploy app.py/model.py/requirements.txt.
2. Upload mlb_moneyline_master_2023_2025.csv in the v0.16.2 section.
3. Leave defaults: 12 prior team games, 3 starter starts, 80 platoon PA, 20 hitter PA.
4. Run Moneyline Price-Bucket Audit.
5. Download mlb_v0162_moneyline_price_bucket_audit.csv and send it back for review.

No Odds API historical credits are used.
