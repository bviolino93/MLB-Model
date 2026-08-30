MLB Edge v0.8.2-ODDS-KEY-HOTFIX

What this fixes
===============
- 401 Unauthorized from The Odds API is now shown as a clean, actionable message.
- Raw requests exceptions no longer expose the full API URL.
- ODDS_API_KEY is explicitly redacted from any user-visible error text.
- 429 / rate-limit errors get their own friendly message.
- Single Game automatic consensus market remains the default workflow.
- Manual market editing remains available as a fallback.

Important
=========
If an API key has appeared in a screenshot or error message, rotate it at The Odds API and then update Streamlit Secrets:

ODDS_API_KEY = "your_new_key_here"

Do not put the real key directly in app.py.

Model logic
===========
Unchanged.
