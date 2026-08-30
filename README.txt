MLB Edge v0.9.3-CLEAN-UX

Purpose
=======
Tighten the MLB mobile UI after visual review and remove confusing letter grades from the user-facing betting experience.

What changed
============
- Removed visible A/B/C/D grades from the main app.
- User-facing verdicts are now:
  BEST BET
  BET
  LEAN
  PASS
- Letter grades remain internal only for ranking/backtesting compatibility.
- Fixed total-card logo placement so team logos never overlap the pick text.
- Matchup identity is now separated from the betting recommendation.
- Top Market card is smaller and cleaner on iPhone.
- Ranked Markets shows actionable BET / LEAN plays first.
- PASS markets are de-emphasized and collapsed under "Show all markets".
- Manual market editing is buried under Advanced Tools.
- Disabled/custom grading is no longer visually competing with the main recommendation.
- Full Slate Top 5 / Top 10 also uses verbal verdict pills instead of letter grades.
- Chronological game dropdowns retain the same ranking logic with clearer labels.
- Vertical spacing and card heights were reduced for a more sportsbook-like mobile experience.

Model logic
===========
No projection or threshold changes were made in this release.
v0.9.1 market sanitation and v0.9 calibrated decision logic remain intact.
