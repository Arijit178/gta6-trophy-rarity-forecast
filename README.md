# GTA VI Trophy Rarity Forecaster

Forecasts, for each trophy of Grand Theft Auto VI (PS5), five quantiles
(5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at a case's target
— entry for make.fun bounty `gta6-trophy-rarity-forecast-01`.

## Model

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` in the input directory and writes `<id>.out` with one row per
trophy: `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

A rate is earners / players tracked, and two forces move it:

1. **Progression.** As the cohort ages, more players finish chapters,
   missions and activities. The median logit rate grows linearly in
   `log(1 + age/24h)` since release, with a slope prior by trophy kind
   (story chapters 3.0, named/final missions 2.6, one-off activities 1.8,
   grinds/collectibles 1.0, platinum/100%-completion 0.6), blended with
   the slope actually observed in the case's history when it moves.
   Over 90 minutes this moves the median ~0.2 logit; over the final's
   4.5 days it moves story trophies ~+2.5 logit and grinds ~+0.8.

2. **Refresh dilution.** Exophase refreshes its counts on its own
   schedule: launch night the page sat unchanged at 638 players
   tracked for hours, then jumped to 3,868 with every rate but the 100%
   one lower. New players arrive mostly not having earned a trophy yet,
   so a refresh dilutes rates. When the history is flat and the horizon
   is short, a refresh inside the horizon is possible but not certain,
   so it goes in the *tails*, not the median: the lower quantiles are
   stretched down by ~1.3 logit (a refresh scenario) while the median
   stays near the no-refresh level. Over a long horizon refreshes are
   certain and continuous, so the distribution is symmetric again.

**Spread.** Quantiles are `median_logit + z_tau * spread`. The spread
has a floor that does not vanish for short horizons (the starter's
`0.5*sqrt(days)` is 0.125 logit over 90 minutes — far too narrow to
cover a refresh): ~1.25 logit when the history is flat, ~0.7 when it
moves, combined in quadrature with the random-walk term, plus extra
for rare/hard trophies. The 100% trophy stayed at 100% through the
launch-night refresh, so it is forecast to stay (near-)100%.

No practice answer is hard-coded; all parameters are priors/category
slopes and the only numbers taken from a case are its own history,
trophies and horizon.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Validation

`grade-local` cannot score this competition locally (its scorer and
reference data are not published in the kit; it is graded on the server
only), so the practice cases were scored with a scorer written from
the RULES' pinball definition (mean pinball loss on the logit scale),
against synthetic refresh answers: players jump 638 → 3,868 at the
target read and earners grow by a factor g in between. The starter's
loss under this scenario is 1.51 at g ≈ 1.2 — matching the RULES'
stated starter loss of "about 1.51", which calibrates the scenario.

| scenario (earners growth g) | starter | this model |
|---|---|---|
| g = 1.0 (no earner growth) | 1.688 | 0.894 |
| g = 1.2 (calibrated) | 1.508 | 0.786 |
| g = 1.5 | 1.289 | 0.657 |
| g = 2.0 | 1.006 | 0.584 |

Mean over p00 and p01 in each row; this model beats the starter by
~42–49% in every scenario, with 90% coverage 1.00 on the practice
cases (the refresh lands inside the wide lower tail).

Sanity checks: a synthetic final-like case (checkpoint 2026-11-21
11:30 UTC, 6,510-minute horizon) gives medians of ~13% for Chapter 1,
~12% for Chapter 2, ~9% for the final mission, ~1% for the platinum
and ~1.4% for grind trophies. Empty history, all-null rates and a
single-read history all produce valid output (tier priors as fallback).
