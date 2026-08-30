MLB Model v0.12.2 — Pitcher Integrity Test

Purpose: attack the apparent starting-pitcher signal before production use.

New integrity tests use a strict <=6h pregame window and remove doubleheaders:
- Correct starter model
- Team-only placebo (removes all starter features)
- Scrambled-starter placebo (starter feature blocks reassigned within season)
- Swapped-starter placebo (away/home starter histories reversed)
- Established-starter subset (>=5 prior starts)
- Extreme starter-mismatch trim
- Probability cap (20%-80%)
- 2025 month and favorite/underdog robustness splits

Walk-forward remains fixed:
2023 train -> 2024 selects market/model blend -> 2023+2024 refit -> 2025 holdout.
No 2025 threshold optimization is performed.

Run:
Upload mlb_moneyline_master_2023_2025.csv in the v0.12.2 Pitcher Integrity Test section, leave 12 prior team games / 3 prior starter starts, run, then download both integrity CSVs.
