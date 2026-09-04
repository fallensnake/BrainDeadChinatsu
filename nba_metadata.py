#!/usr/bin/env python3
"""
PHASE 1 (fast): dump every NBA market's metadata + resolution. NO candlesticks, so this
finishes in ~1-2 min. Everything here comes free in the list endpoints. After this runs you
can immediately see result_binary for every settled market.

Then run nba_enrich_prices.py to add the tip-off prices (the slow part).

  pip install requests
  python nba_metadata.py
"""
from __future__ import annotations
import os, sys, csv, time, datetime as dt
from typing import Optional, Iterable
import requests

# --- ADD THESE LINES ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
# -----------------------

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SEASON_START = dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc).timestamp()
SEASON_END   = dt.datetime(2026, 7, 15, tzinfo=dt.timezone.utc).timestamp()
OUT = "nba_markets_meta.csv"
SLEEP = 0.05
SERIES_INCLUDE_OVERRIDE = [
    # game & period outcomes
    "KXNBAGAME",
    "KXNBA1HWINNER", "KXNBA2HWINNER",
    "KXNBA1QWINNER", "KXNBA2QWINNER", "KXNBA3QWINNER", "KXNBA4QWINNER",
    # spreads
    "KXNBASPREAD",
    "KXNBA1HSPREAD", "KXNBA2HSPREAD",
    "KXNBA1QSPREAD", "KXNBA2QSPREAD", "KXNBA3QSPREAD", "KXNBA4QSPREAD",
    # totals
    "KXNBATOTAL", "KXNBATEAMTOTAL",
    "KXNBA1HTOTAL", "KXNBA2HTOTAL",
    "KXNBA1QTOTAL", "KXNBA2QTOTAL", "KXNBA3QTOTAL", "KXNBA4QTOTAL",
    # in-game events
    "KXNBAOVERTIME", "KXNBAFIRSTBASKET", "KXNBA30COMEBACK",
    # player props (single game)
    "KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBABLK", "KXNBASTL",
    "KXNBA3PT", "KXNBAFTM", "KXNBAPA", "KXNBAPR", "KXNBARA", "KXNBAPRA",
    "KXNBA2D", "KXNBA3D",
    # head-to-head props (single game)
    "KXNBAH2HPTS", "KXNBAH2H3PT", "KXNBAH2HPRA", "KXNBAH2HBENCHPTS", "KXNBAH2HTEAM3PT",
]
SERIES_EXCLUDE = ("WNBA",)
# Used only if auto-discovery returns nothing. Core NBA series (props get added by discovery).
KNOWN_NBA_SERIES = ["KXNBA", "KXNBAGAME", "KXNBASERIES", "KXNBAEAST", "KXNBAWEST"]

SESSION = requests.Session()

def get(path, params=None):
    for attempt in range(5):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1)); continue
        if r.status_code == 200:
            time.sleep(SLEEP); return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt); continue
        print(f"HTTP {r.status_code} {path}: {r.text[:150]}", file=sys.stderr); return None
    return None

def paginate(path, params, key):
    params = dict(params)
    while True:
        data = get(path, params)
        if not data: return
        for item in data.get(key, []) or []: yield item
        cur = data.get("cursor")
        if not cur: return
        params["cursor"] = cur

def to_ts(iso):
    if not iso: return None
    try: return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except Exception: return None

def discover_series():
    if SERIES_INCLUDE_OVERRIDE: return SERIES_INCLUDE_OVERRIDE
    found = set()
    # The /series list is paginated (default 100/page). Page through ALL of it, or NBA series
    # past the first 100 get missed entirely.
    for params in ({"category": "Sports", "limit": 1000}, {"limit": 1000}):
        for s in paginate("/series", params, "series"):
            tk = (s.get("ticker") or "").upper()
            if "NBA" in tk and not any(x in tk for x in SERIES_EXCLUDE):
                found.add(s["ticker"])
        if found: break
    if not found:
        print("Discovery empty; falling back to KNOWN_NBA_SERIES.", file=sys.stderr)
        found = set(KNOWN_NBA_SERIES)
    return sorted(found)

def classify(series, title=""):
    t = series.upper(); blob = (series + " " + (title or "")).upper()
    if "SERIES" in t: return "series_winner"
    if t == "KXNBA" or "CHAMPION" in blob or "TITLE" in blob: return "champion"
    if "EAST" in t: return "east_conf_winner"
    if "WEST" in t: return "west_conf_winner"
    if "SPREAD" in blob: return "spread"
    if "TOTAL" in blob or "OVER" in blob: return "total_points"
    if any(k in blob for k in ("POINT","PTS","REB","AST","ASSIST","REBOUND",
                               "DOUBLE","TRIPLE","BLOCK","STEAL","THREE","3PT")):
        return "player_prop"
    if "GAME" in t or "WINNER" in blob or "BEAT" in blob: return "game_winner"
    return "other"

FIELDS = ["market_ticker","event_ticker","series_ticker","market_type","title",
          "yes_sub_title","no_sub_title","result","result_binary","status",
          "open_time","close_time","expected_expiration_time","volume","open_interest"]

def main():
    series_list = discover_series()
    if not series_list:
        print("No NBA series found. Set SERIES_INCLUDE_OVERRIDE.", file=sys.stderr); return
    print(f"Series: {', '.join(series_list)}")
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="", encoding="utf-8"); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    seen = set(); n = 0
    for series in series_list:
        print(f"=== {series} ===")
        for base in ("/markets", "/historical/markets"):
            for m in paginate(base, {"series_ticker": series, "limit": 1000}, "markets"):
                tk = m.get("ticker")
                if not tk or tk in seen: continue
                seen.add(tk)
                ct = to_ts(m.get("close_time"))
                if ct and not (SEASON_START <= ct <= SEASON_END): continue
                res = (m.get("result") or "").lower()
                w.writerow({
                    "market_ticker": tk, "event_ticker": m.get("event_ticker"),
                    "series_ticker": series, "market_type": classify(series, m.get("title","")),
                    "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"),
                    "no_sub_title": m.get("no_sub_title"), "result": m.get("result"),
                    "result_binary": {"yes":1,"no":0}.get(res),
                    "status": m.get("status"), "open_time": m.get("open_time"),
                    "close_time": m.get("close_time"),
                    "expected_expiration_time": m.get("expected_expiration_time"),
                    "volume": m.get("volume_fp") or m.get("volume"),
                    "open_interest": m.get("open_interest_fp") or m.get("open_interest"),
                })
                n += 1
        f.flush(); print(f"  total so far: {n}")
    f.close(); print(f"Done. {n} markets -> {OUT}")

if __name__ == "__main__":
    main()
