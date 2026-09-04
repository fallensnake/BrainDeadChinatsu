#!/usr/bin/env python3
"""
PHASE 2 (batched): read nba_markets_meta.csv and append the tip-off price for each market,
writing ONE combined file (nba_markets_enriched.csv) = meta columns + price columns.

Big speedup: markets in the same game share a tip-off window, so we fetch up to 100 at a time
via GET /markets/candlesticks instead of one call per market. That cuts request count ~100x,
which is what the rate limit actually counts.

  - groups markets by tip-off ts, batches <=100 tickers/call (include_latest_before_start=true
    so even no-trade props still return the last price before tip-off)
  - bulk strike_date prefetch (one /events sweep per series) so no per-market /events calls
  - connection pool sized to THREADS (default requests pool caps at 10 -> silent bottleneck)
  - FALLBACK: archived markets (settled before Kalshi's historical cutoff) have no batch
    endpoint, so those drop to per-market /historical calls. The run prints the batch/fallback
    split -- if fallback dominates, your data is mostly historical and that part stays slow.

Resumable. Join-free: feed the output straight into the FLB analysis.

  pip install requests
  python nba_enrich_prices.py
"""
from __future__ import annotations
import os, sys, csv, time, datetime as dt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
META_IN = "nba_markets_meta.csv"
OUT = "nba_markets_enriched.csv"

THREADS = 20
BATCH_SIZE = 100             # max tickers per batch call (API hard limit)
PRE_MIN, POST_MIN = 30, 10   # window around tip-off; 40 candles x100 tickers < 10k cap
PERIOD = 1
EXPECTED_TO_TIPOFF_MIN = 150
ONLY_SETTLED = True
PREFETCH_EVENTS = True
SLEEP = 0.03

PRICE_FIELDS = ["tipoff_ts_used", "tipoff_source", "candle_ts",
                "price_at_start_yes_bid", "price_at_start_yes_ask", "price_at_start_mid",
                "price_at_start_last", "n_candles_in_window"]

SESSION = requests.Session()
# Size the connection pool to the worker count, or threads 11+ block on the default cap of 10.
_adapter = requests.adapters.HTTPAdapter(pool_connections=THREADS, pool_maxsize=THREADS)
SESSION.mount("https://", _adapter)

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
    # close first; fall through to previous_* for the synthetic before-start candle (OHLC null)
    for k in ("close","close_dollars","mean","mean_dollars","previous","previous_dollars"):
        v = b.get(k)
        if v not in (None, ""): return prob(v)
    return None

# ---- strike_date prefetch (event_ticker -> tip-off ts; /events covers historical too) ----
_evt = {}
def prefetch_strike_dates(series_set):
    n = 0
    for series in sorted(series_set):
        for ev in paginate("/events", {"series_ticker": series, "limit": 1000}, "events"):
            et = ev.get("event_ticker")
            if et and et not in _evt:
                _evt[et] = to_ts(ev.get("strike_date")); n += 1
    print(f"prefetched strike_date for {n} events across {len(series_set)} series")

def strike_ts(event_ticker):
    if event_ticker in _evt: return _evt[event_ticker]
    d = get(f"/events/{event_ticker}", {"with_nested_markets": "false"}) or {}
    ts = to_ts((d.get("event") or {}).get("strike_date"))
    _evt[event_ticker] = ts
    return ts

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

# ---- candlestick fetch: batch (live tier) + single fallback (historical tier) ----
def batch_candles(tickers, start_ts, end_ts):
    p = {"market_tickers": ",".join(tickers), "start_ts": int(start_ts), "end_ts": int(end_ts),
         "period_interval": PERIOD, "include_latest_before_start": "true"}
    d = get("/markets/candlesticks", p)
    out = {}
    if d:
        for m in d.get("markets", []) or []:
            out[m.get("market_ticker")] = m.get("candlesticks") or []
    return out

CUTOFF = None
def settled_cutoff():
    global CUTOFF
    if CUTOFF is None:
        d = get("/historical/cutoff") or {}
        CUTOFF = to_ts(d.get("market_settled_ts")) or d.get("market_settled_ts") or 0
        print(f"historical cutoff ts = {CUTOFF}")
    return CUTOFF

def single_candles(series, ticker, settle_ts, start_ts, end_ts):
    p = {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": PERIOD}
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

# ---- row assembly ----
def merge_price(row, candles, anchor):
    out = dict(row)
    for f in PRICE_FIELDS: out.setdefault(f, None)
    out["tipoff_ts_used"] = anchor
    out["tipoff_source"] = row.get("_src")
    out["n_candles_in_window"] = len(candles)
    if anchor and candles:
        c = pick(candles, anchor)
        if c:
            yb, ya = candle_close(c,"yes_bid"), candle_close(c,"yes_ask")
            out["price_at_start_yes_bid"] = yb
            out["price_at_start_yes_ask"] = ya
            out["price_at_start_mid"] = (yb+ya)/2 if (yb is not None and ya is not None) else None
            out["price_at_start_last"] = candle_close(c,"price")
            out["candle_ts"] = c.get("end_period_ts")
    return out

def batch_task(anchor, chunk):
    start, end = anchor - PRE_MIN*60, anchor + POST_MIN*60
    res = batch_candles([r["market_ticker"] for r in chunk], start, end)
    hits, misses = [], []
    for r in chunk:
        candles = res.get(r["market_ticker"]) or []
        (hits if candles else misses).append((r, candles))
    return [merge_price(r, c, anchor) for r, c in hits], [r for r, _ in misses]

def fallback_task(row):
    anchor = row["_anchor"]
    settle = to_ts(row.get("expected_expiration_time")) or to_ts(row.get("close_time"))
    candles = single_candles(row["series_ticker"], row["market_ticker"], settle,
                             anchor - PRE_MIN*60, anchor + POST_MIN*60)
    return merge_price(row, candles, anchor)

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
    if not rows: return

    if PREFETCH_EVENTS:
        prefetch_strike_dates({r["series_ticker"] for r in rows if r.get("series_ticker")})

    # group by tip-off ts; rows with no anchor get written with null prices
    groups, no_anchor = defaultdict(list), []
    for r in rows:
        a, src = anchor_for(r)
        r["_anchor"], r["_src"] = a, src
        (groups[a].append(r) if a else no_anchor.append(r))

    # build batch chunks: <=100 tickers, all sharing one tip-off window
    chunks = []
    for anchor, grp in groups.items():
        for i in range(0, len(grp), BATCH_SIZE):
            chunks.append((anchor, grp[i:i+BATCH_SIZE]))
    print(f"{len(chunks)} batch calls cover {sum(len(g) for g in groups.values())} markets")

    new = not os.path.exists(OUT)
    out = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    if new: w.writeheader()

    written, miss_rows = 0, []
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = [ex.submit(batch_task, a, c) for a, c in chunks]
        for fut in as_completed(futs):
            hits, misses = fut.result()
            for row in hits: w.writerow(row); written += 1
            miss_rows.extend(misses)
            out.flush()
            if written % 500 < len(hits): print(f"  batched {written} priced, {len(miss_rows)} to fallback")

    for r in no_anchor: w.writerow(merge_price(r, [], None)); written += 1

    print(f"batch phase done: {written} priced, {len(miss_rows)} markets need per-market fallback")
    if miss_rows:
        settled_cutoff()
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            futs = [ex.submit(fallback_task, r) for r in miss_rows]
            n = 0
            for fut in as_completed(futs):
                w.writerow(fut.result()); written += 1; n += 1
                out.flush()
                if n % 100 == 0: print(f"  fallback {n}/{len(miss_rows)}")

    out.close()
    print(f"Done. {written} rows -> {OUT}")

if __name__ == "__main__":
    main()