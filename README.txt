MLB Model v0.12.1 — Pitcher Leakage Audit

What changed
- Adds a hostile audit of the PIT starting-pitcher result.
- Reruns walk-forward validation at <=18h, <=12h, <=6h, and <=3h before first pitch.
- Removes doubleheaders in conservative audit rows.
- Fixes MLB innings notation: 5.1 = 5 1/3 IP and 5.2 = 5 2/3 IP.
- Keeps 2025 untouched: 2024 still chooses the model/market blend.
- Produces downloadable audit summary and strict-window holdouts.
- Uses MLB Stats API only; zero The Odds API credits.

Important limitation
MLB's current historical schedule record is retrospective. The starter attached to an old game cannot by itself prove that pitcher was publicly announced at the exact historical odds snapshot. The audit therefore treats starter identity as a potential leakage vector and stress-tests it; it does not claim to eliminate that uncertainty.

Recommended run
1. Open Backtest Lab.
2. Go to v0.12.1 Pitcher Leakage Audit.
3. Upload mlb_moneyline_master_2023_2025.csv.
4. Leave 12 prior team games / 3 prior starter starts.
5. Run Pitcher Leakage Audit.
6. Download mlb_pit_pitcher_audit_summary.csv and send it back for review.
