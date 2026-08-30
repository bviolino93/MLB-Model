MLB Edge v0.10.2-SMART-HISTORY

Main improvement
================
Historical Odds planning now uses ACTUAL MLB regular-season game dates from
MLB's free Stats API rather than every calendar day.

This means:
- offseason dates are skipped
- no-game dates are skipped
- the displayed Odds API credit ceiling is much more realistic
- the free schedule lookup consumes zero Odds API credits

Default workflow
================
2023-03-30 through 2025-09-28
Moneyline only (Phase 1)

Then, only after downloading and saving the cache:
Phase 2: Run Line
Phase 3: Total

The app shows conservative credit ceilings for each phase and all three
combined. It warns when a selected run can exceed a 20K plan.

Safeguards retained
===================
- explicit paid-credit confirmation
- hard credit cap
- cached dates skipped
- successful snapshots saved immediately
- downloadable Historical Market CSV
- downloadable Cache ZIP
- API key is never displayed in errors

UI
==
Dark-mode/mobile label readability has been improved.

Methodology
===========
The snapshot remains fixed at 15:00 UTC as a consistent pregame baseline.
It is NOT described as a verified closing line.
