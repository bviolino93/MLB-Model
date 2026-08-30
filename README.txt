MLB Edge v0.11.2-ARRAY-FIX

Fix
====
The point-in-time backtest reached the Brier/log-loss evaluation step, but
those helpers assumed pandas Series and called `.notna()`.

Some walk-forward predictions are NumPy arrays, which caused:
  'numpy.ndarray' object has no attribute 'notna'

v0.11.2 updates the probability-scoring helpers to accept both pandas Series
and NumPy arrays using NumPy finite-value masks.

No backtest methodology, features, thresholds, season split, market blend
logic, or Odds API usage changed.

This fix consumes zero Odds API credits.
