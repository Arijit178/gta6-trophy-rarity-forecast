"""GTA VI trophy rarity forecaster — gta6-trophy-rarity-forecast-01.

Forecasts, for each trophy of Grand Theft Auto VI (PS5), five quantiles
(5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at a case's target.

Model (earners / players decomposition, in logit space)
--------------------------------------------------------
A rate is earners / players_tracked.  Two forces move it:

1. Progression.  As the player cohort ages, more players finish chapters,
   missions and activities, so earners grow.  The median logit rate is
   modelled as growing linearly in log(1 + age/24h), where age is hours
   since release:  logit r(age) ~= logit r(last) + s * dlog_age.  The slope
   s is a prior by trophy kind (story chapters/missions progress fastest,
   one-off activities next, grinds and collectibles slower, the platinum /
   100%-completion slowest), blended with the slope actually observed in
   the case's history when the history shows movement.  Over 90 minutes
   this moves the median only ~0.2 logit; over the final's 4.5 days it
   moves story trophies by ~+2.5 logit and grinds by ~+0.8.

2. Refresh dilution.  Exophase refreshes its counts on its own schedule,
   not with every trophy earned: a read can sit unchanged for hours and
   then jump (launch night: 638 players tracked, unchanged for hours,
   then 3,868 at the next refresh, every rate but the 100% one lower).
   New players arrive mostly not having earned a trophy yet, so a
   refresh dilutes rates.  When the history is flat and the horizon is
   short, a refresh inside the horizon is a real possibility but not a
   certainty, so it is put in the *tails*, not the median: the lower
   quantiles are stretched down (a refresh scenario) while the median
   stays near the no-refresh level.  Over a long horizon refreshes are
   certain and continuous, so the distribution is symmetric again and
   the extra width sits in the spread instead.

Spread.  Quantiles are median_logit + z_tau * spread, with a spread
floor that does not vanish for short horizons (the starter's spread,
0.5*sqrt(days), is 0.125 logit over 90 minutes — far too narrow to
cover a refresh).  The floor is ~1.25 logit when the history is flat
and ~0.7 when it moves, combined in quadrature with the random-walk
term, plus a little extra for rare/hard trophies.  The 100% trophy is
special: it stayed at 100% through the launch-night refresh, so it is
forecast to stay (near-)100%.

Nothing here hard-codes any practice answer.  All parameters are
priors / category slopes; the only numbers taken from a case are its
own history, its trophies and its horizon.

Run:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py, standard library only, no network.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- helpers ----------------------------------------------------------------

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as the starter does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_between(a, b):
    if a is None or b is None:
        return None
    return (b - a).total_seconds() / 3600.0


# --- trophy kind ------------------------------------------------------------

_CHAPTER = re.compile(r"complete chapter \d", re.I)
_FINAL = re.compile(r"final mission", re.I)
_MISSION = re.compile(r"complete (the )?(final )?mission", re.I)
_GRIND = re.compile(
    r"(\b250\b|\b100\b|\b50\b|\b20\b|\b8\b|\b3\b|\b5\b).*(stunt|location|challenge|"
    r"race|card|vehicle|spot|strand)|every (spot|location)|gold medal|"
    r"collectible|100% completion|attain 100|complete 250|complete 50|"
    r"complete 8|complete 3|win 20|discover 100|earn \$|spend \$|purchase all|"
    r"register 10|explore all|meditate at every",
    re.I,
)


def trophy_kind(t):
    """Prior progression kind from the trophy's own description."""
    name = (t.get("name") or "") + " " + (t.get("description") or "")
    tier = (t.get("tier") or "").lower()
    if tier == "platinum":
        return "hard"
    if "100% completion" in name or "100% Completion" in name:
        return "hard"
    if _CHAPTER.search(name):
        return "story"
    if _FINAL.search(name):
        return "story_mission"
    if _MISSION.search(name):
        return "story_mission"
    if _GRIND.search(name):
        return "grind"
    return "activity"


# Prior logit slope per log(1 + age/24h), by kind.
PRIOR_SLOPE = {
    "story": 3.0,          # chapters: most of the cohort finishes them within days
    "story_mission": 2.6,  # named missions / final mission
    "activity": 1.8,       # one-off activities
    "grind": 1.0,          # counts, collections, medals
    "hard": 0.6,           # platinum / 100% completion
}
# Extra spread (logit) by kind: harder trophies are more uncertain.
KIND_SPREAD = {"story": 0.15, "story_mission": 0.20, "activity": 0.20,
               "grind": 0.25, "hard": 0.30}

TIER_PRIOR_RATE = {"platinum": 0.5, "gold": 0.8, "silver": 0.8, "bronze": 1.0}


# --- per-case forecast ------------------------------------------------------

def distinct_reads(history):
    """Collapse consecutive identical reads (same players + same rates)."""
    out = []
    prev = None
    for h in history or []:
        key = (h.get("players_tracked"),
               tuple(h.get("rates") or []))
        if key != prev:
            out.append(h)
            prev = key
    return out


def observed_slope(case, idx, distinct, release):
    """Least-squares slope of logit(rate) vs log(1+age/24h) over distinct
    reads with a non-null rate for this trophy.  None if <3 such reads."""
    pts = []
    for h in distinct:
        rates = h.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        t = parse_time(h.get("at"))
        age = hours_between(release, t)
        if age is None or age < 0:
            continue
        pts.append((math.log1p(age / 24.0), logit(rates[idx])))
    if len(pts) < 3:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den < 1e-9:
        return None
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
    return max(-1.0, min(6.0, slope))


def forecast_case(case, name):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_h = horizon_min / 60.0
    horizon_days = horizon_min / 1440.0

    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = parse_time(case.get("at"))
    at = parse_time(case.get("at"))
    target = parse_time(case.get("target"))

    distinct = distinct_reads(history)
    flat_history = len(distinct) <= 1
    last = history[-1] if history else {}
    last_rates = last.get("rates") or []
    last_n = last.get("players_tracked") or 0

    age_last = hours_between(release, at)
    age_target = hours_between(release, target)
    if age_last is None or age_last < 0:
        age_last = 0.0
    if age_target is None or age_target < age_last:
        age_target = age_last + horizon_h
    dlog_age = math.log1p(age_target / 24.0) - math.log1p(age_last / 24.0)
    dlog_age = max(0.0, dlog_age)

    # Spread: random-walk term + a floor that survives short horizons.
    rw = 0.5 * math.sqrt(max(horizon_days, 0.0))
    floor = 1.25 if flat_history else 0.70
    # A long horizon already carries a wide random-walk spread; the floor
    # matters most when the horizon is short and the history is stale.
    if horizon_h > 24:
        floor = min(floor, 0.85)

    # Short-horizon refresh risk: a refresh inside the horizon is
    # possible but not certain while the history is flat, so stretch the
    # lower tail down by roughly one refresh jump (log ~3x dilution).
    refresh_tail = 0.0
    if flat_history and horizon_h <= 12:
        refresh_tail = 1.3

    rows = []
    for i, t in enumerate(trophies):
        rate = None
        if i < len(last_rates):
            rate = last_rates[i]
        if rate is None:
            # Last non-null rate for this trophy, else a tier prior.
            for h in reversed(history):
                rs = h.get("rates") or []
                if i < len(rs) and rs[i] is not None:
                    rate = rs[i]
                    break
        if rate is None:
            rate = TIER_PRIOR_RATE.get((t.get("tier") or "").lower(), 1.0)

        # The 100% trophy stayed at 100% through the launch-night
        # refresh: it is structural, forecast it to stay (near-)100%.
        if rate >= 99.99:
            rows.append({"id": t["id"], "q": [95, 99, 100, 100, 100]})
            continue

        kind = trophy_kind(t)
        prior_slope = PRIOR_SLOPE[kind]
        obs = observed_slope(case, i, distinct, release)
        if obs is None:
            slope = prior_slope
        else:
            # Blend what the history shows with the kind prior; the
            # history's slope is noisy when it spans only a few reads.
            w = len(distinct) / (len(distinct) + 3.0)
            slope = w * obs + (1.0 - w) * prior_slope
        slope = max(-0.5, min(5.0, slope))

        median_logit = logit(rate) + slope * dlog_age
        # A rate cannot plausibly exceed ~70% a week after launch for
        # these trophies, and cannot go below the smallest observable
        # step; keep the median inside a sane band.
        median_logit = max(logit(0.02), min(logit(70.0), median_logit))

        extra = KIND_SPREAD[kind]
        if rate < 1.0:
            extra += 0.15  # few earners: binomial noise is relatively large
        spread = math.sqrt(rw * rw + floor * floor + extra * extra)
        spread = min(spread, 1.6)

        qs = []
        for z in Z:
            lg = median_logit + z * spread
            if refresh_tail:
                if z < 0:
                    # Lower tail carries the refresh scenario.
                    lg -= refresh_tail * (0.9 if z < -1 else 0.45)
            qs.append(round4(clamp_pct(inv_logit(lg))))
        # Keep quantiles non-decreasing after rounding.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
