MLB Edge v0.10.1-HISTORY-BUILDER

Adds a guarded Historical Data Builder for The Odds API inside Backtest Lab.

Default window: 2023-04-01 through 2025-10-05
Snapshot: one league-wide MLB snapshot per date at 15:00 UTC
Markets: h2h, spreads, totals
Region: us

Historical usage ceiling is 10 credits x markets x regions per snapshot. With all 3 default markets, max is 30 credits per nonempty snapshot. Empty responses cost 0.

Safeguards:
- no API call just by opening the app
- explicit paid-credit confirmation checkbox
- hard credit cap checked before every request
- successful snapshots cached immediately
- cached dates skipped
- downloadable cache ZIP
- cache ZIP restore after redeploy
- downloadable historical market CSV
- API key never displayed in errors
- API remaining/last-cost headers displayed after requests

15:00 UTC is a consistent pregame snapshot, not a verified closing line.
The builder creates the historical market half of the backtest dataset. Point-in-time model outputs and final game results still need to be merged for the complete production backtest.
