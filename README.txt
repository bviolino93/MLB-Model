MLB Edge v0.10.3-SEASON-SCHEDULE-FIX

Fix
===
The free MLB schedule lookup now runs one calendar year at a time instead of
requesting the entire 2023-2025 range in one call.

Why
===
The prior multi-year schedule request returned only 188 distinct game dates,
which was clearly incomplete for three MLB seasons and understated the paid
Historical Odds credit estimate.

New safeguards
==============
- Fetch 2023, 2024, 2025 schedule ranges separately
- Combine and deduplicate dates
- Display distinct game-date count for each season
- Full included seasons must each show at least 150 distinct regular-season
  game dates before the paid Historical Odds build button is enabled
- If the schedule sanity check fails, paid downloading is blocked

Credit behavior
===============
The MLB schedule lookup remains free and consumes zero Odds API credits.
No Historical Odds credits are used merely by opening the app.

Recommended workflow
====================
1. Deploy v0.10.3.
2. Open Backtest Lab -> Historical Data Builder.
3. Confirm 2023, 2024, and 2025 each show roughly a full season of distinct
   game dates, and total dates are around the expected multi-season range.
4. Confirm Moneyline credit estimate.
5. Only then activate the paid Historical Odds plan.
6. Start with Moneyline only.

Historical snapshot remains 15:00 UTC and is a consistent pregame baseline,
not a verified closing line.
