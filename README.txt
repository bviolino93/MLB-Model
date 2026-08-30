MLB Edge v0.11.1-SCIPY-FIX

Fix
====
v0.11.0 uses scipy.optimize.minimize to fit the ridge logistic point-in-time
model, but app.py omitted:

    from scipy.optimize import minimize

That caused:
    name 'minimize' is not defined

v0.11.1 adds the missing import and explicitly includes scipy in
requirements.txt.

No backtest methodology, thresholds, historical data, model features,
market calibration, or Odds API usage changed.

This fix consumes zero Odds API credits.
