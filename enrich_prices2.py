#!/usr/bin/env python3
"""
PHASE 2: read nba_markets_meta.csv and append the tip-off price for each market, writing ONE
combined file (nba_markets_enriched.csv) = every metadata column + price columns at the end.

Speed:
  - bulk strike_date prefetch: one paginated /events sweep per series builds an
    event_ticker -> tip-off map, so markets look up their game's start time locally instead of
    each firing its own /events call (that per-market call was quietly doubling request count).
  - threaded candle fetches; routes each market to the live OR historical candlestick endpoint
    directly via /historical/cutoff; only prices SETTLED markets by default.
Resumable: re-running skips market_tickers already in the output file.

  pip install requests
  python nba_enrich_prices.py
"""
from __future__ import annotations
import os, sys, csv, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
META_IN = "nba_markets_meta.csv"
OUT = "nba_markets_enriched.csv"

THREADS = 16                 # network-bound, so more threads = faster (backs off on 429s)
PRE_MIN, POST_MIN = 90, 20
PERIOD = 1
EXPECTED_TO_TIPOFF_MIN = 150
ONLY_SETTLED = True
PREFETCH_EVENTS = True       # bulk-load strike_dates up front (the speedup)
SLEEP = 0.03

PRICE_FIELDS = ["tipoff_ts_used", "tipoff_source", "candle_ts",
                "price_at_start_yes_bid", "price_at_start_yes_ask", "price_at_start_mid",
                "price_at_start_last", "n_candles_in_window"]

SESSION = requests.Session()

def get(path, params=None):
    for attempt in range(5):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1)); continue
        if r.status_code == 200:
            time.sleep(SLEEP); return r.json()
        if r.status_code == 404: return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt); continue
        return None
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
    try: return int(dt.datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp())
    except Exception: return None

def prob(v):
    if v in (None, ""): return None
    try: f = float(v)
    except (TypeError, ValueError): return None
    return f if f <= 1.0 else f / 100.0

def candle_close(c, field):
    b = c.get(field)
    if not isinstance(b, dict): return None
    for k in ("close","close_dollars","mean","mean_dollars"):
        if k in b: return prob(b[k])
    return None

# event_ticker -> tip-off ts. Seeded in bulk by prefetch; misses fall back to a single call.
_evt = {}

def prefetch_strike_dates(series_set):
    """One paginated /events sweep per series. Fills _evt so per-market calls aren't needed."""
    n = 0
    for series in sorted(series_set):
        for ev in paginate("/events", {"series_ticker": series, "limit": 1000}, "events"):
            et = ev.get("event_ticker")
            if et and et not in _evt:
                _evt[et] = to_ts(ev.get("strike_date"))
                n += 1
    print(f"prefetched strike_date for {n} events across {len(series_set)} series")

def strike_ts(event_ticker):
    if event_ticker in _evt: return _evt[event_ticker]          # bulk hit (or prior lookup)
    d = get(f"/events/{event_ticker}", {"with_nested_markets": "false"}) or {}
    ts = to_ts((d.get("event") or {}).get("strike_date"))
    _evt[event_ticker] = ts                                     # cache the miss too
    return ts

CUTOFF = None
def settled_cutoff():
    global CUTOFF
    if CUTOFF is None:
        d = get("/historical/cutoff") or {}
        CUTOFF = to_ts(d.get("market_settled_ts")) or d.get("market_settled_ts") or 0
        print(f"historical cutoff ts = {CUTOFF}")
    return CUTOFF

def fetch_candles(series, ticker, settle_ts, start_ts, end_ts):
    p = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": PERIOD}
    historical_first = settle_ts and settle_ts < settled_cutoff()
    paths = ([f"/historical/markets/{ticker}/candlesticks",
              f"/series/{series}/markets/{ticker}/candlesticks"]
             if historical_first else
             [f"/series/{series}/markets/{ticker}/candlesticks",
              f"/historical/markets/{ticker}/candlesticks"])
    for path in paths:
        d = get(path, p)
        if d and d.get("candlesticks"):
            return d["candlesticks"]
    return []

def anchor_for(row):
    s = strike_ts(row["event_ticker"])
    if s: return s, "strike_date"
    ee = to_ts(row.get("expected_expiration_time"))
    if ee: return ee - EXPECTED_TO_TIPOFF_MIN*60, "expected_expiration-150m"
    ct = to_ts(row.get("close_time"))
    if ct: return ct, "close_time"
    return None, "none"

def pick(candles, anchor):
    before = [c for c in candles if c.get("end_period_ts",0) <= anchor]
    if before: return max(before, key=lambda c: c["end_period_ts"])
    return min(candles, key=lambda c: c.get("end_period_ts",0)) if candles else None

def work(row):
    """Return the original meta row dict with price columns merged in."""
    out = dict(row)
    for f in PRICE_FIELDS: out.setdefault(f, None)
    anchor, src = anchor_for(row)
    out["tipoff_ts_used"] = anchor
    out["tipoff_source"] = src
    if not anchor:
        return out
    settle = to_ts(row.get("expected_expiration_time")) or to_ts(row.get("close_time"))
    candles = fetch_candles(row["series_ticker"], row["market_ticker"], settle,
                            anchor - PRE_MIN*60, anchor + POST_MIN*60)
    out["n_candles_in_window"] = len(candles)
    c = pick(candles, anchor)
    if c:
        yb, ya = candle_close(c,"yes_bid"), candle_close(c,"yes_ask")
        out["price_at_start_yes_bid"] = yb
        out["price_at_start_yes_ask"] = ya
        out["price_at_start_mid"] = (yb+ya)/2 if (yb is not None and ya is not None) else None
        out["price_at_start_last"] = candle_close(c,"price")
        out["candle_ts"] = c.get("end_period_ts")
    return out

def main():
    if not os.path.exists(META_IN):
        print(f"Run nba_metadata.py first ({META_IN} missing).", file=sys.stderr); return

    with open(META_IN, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        meta_cols = reader.fieldnames or []
        all_rows = list(reader)
    fieldnames = meta_cols + [c for c in PRICE_FIELDS if c not in meta_cols]

    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            done = {r["market_ticker"] for r in csv.DictReader(f)}

    rows = []
    for r in all_rows:
        if r["market_ticker"] in done: continue
        if ONLY_SETTLED and (r.get("status") not in ("settled","finalized")
                             and r.get("result_binary") in ("", None)):
            continue
        rows.append(r)
    print(f"{len(rows)} markets to price ({len(done)} already done).")

    if PREFETCH_EVENTS and rows:
        prefetch_strike_dates({r["series_ticker"] for r in rows if r.get("series_ticker")})

    new = not os.path.exists(OUT)
    out = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    if new: w.writeheader()
    n = 0
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for fut in as_completed(futs):
            w.writerow(fut.result()); out.flush()
            n += 1
            if n % 50 == 0: print(f"  priced {n}/{len(rows)}")
    out.close(); print(f"Done. {n} rows -> {OUT}")

if __name__ == "__main__":
    main()